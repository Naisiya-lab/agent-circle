"""Serial, frozen, resumable experiment runner. No answer scoring."""
from __future__ import annotations

from collections import Counter
from html import escape
import json
import os
from pathlib import Path
import time
import uuid

from .pipeline import transport, run_sample
from .protocol import VERSION, ProtocolError

BASE = transport.BASE
DEFAULT_SOURCE = BASE / "prepared/mmtab_eval.json"
DEFAULT_RUN = "circle_mmtab_eval"
PROMPT_NAMES = {"understand": "01_understand.txt", "select": "02_select.txt",
                "solve": "04_solve.txt", "check": "05_check.txt"}


def prompt_map(cfg=None):
    folder = Path(__file__).parent / "prompts"
    prompts = {role: (folder/name).read_text(encoding="utf-8") for role, name in PROMPT_NAMES.items()}
    for role, name in (cfg or {}).get("prompt_overrides", {}).items():
        transport.require(role in PROMPT_NAMES, "Unknown prompt role")
        path = (BASE/name).resolve()
        transport.require(path.is_relative_to(folder.resolve()) and path.is_file(), "Selector prompt must be inside runtime prompts")
        prompts[role] = path.read_text(encoding="utf-8")
    return prompts


def code_files():
    return sorted(list(Path(__file__).parent.rglob("*.py")) +
                  list((Path(__file__).parent/"prompts").rglob("*.txt")) +
                  list((BASE/"table_tools").glob("*.py")) + [BASE/"agent_circle.py"])


def run_root(name):
    transport.require(transport.SAFE_ID.fullmatch(name) and name not in (".", ".."), "Unsafe run name")
    import os
    return Path(os.environ.get("AGENT_CIRCLE_RUNS", str(BASE / "runs"))) / name


def acquire_lock(root):
    lock = root / "run.lock"
    try:
        with lock.open("x", encoding="utf-8") as file:
            file.write(str(os.getpid()))
    except FileExistsError as exc:
        raise ValueError("Run is locked. Verify the owning process has exited before removing: " + str(lock)) from exc
    return lock


def prepare(cfg, name=DEFAULT_RUN, source=DEFAULT_SOURCE, limit=18):
    transport.proxy_mode(cfg)
    transport.uses_image(cfg, "select")
    transport.require(cfg.get("loop") == {"max_rework_rounds": 1, "extraction": "rectangle_merge_closure", "coordinate_base": 1},
                      "This version supports exactly one rework round and 1-based merge-closed rectangles")
    transport.require(cfg.get("communication") == {"version": VERSION, "max_readbacks_per_stage": 1},
                      "Tool message version and one readback per stage must be explicit")
    transport.require(type(limit) is int and limit > 0, "--limit must be positive")
    root = run_root(name)
    root.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(root)
    try:
        source = Path(source).resolve()
        records = transport.read_json(source)
        transport.require(len(records) >= limit, "Source has fewer samples than --limit")
        # Explicit field projection: references, source labels and hidden solutions never enter the workflow.
        fields = ("id", "question", "image", "image_sha256", "task_type", "dataset_name", "image_id")
        rows = [{k: item[k] for k in fields if k in item} for item in records[:limit]]
        transport.require(len({row["id"] for row in rows}) == limit, "Duplicate sample IDs")
        for row in rows:
            transport.require(transport.SAFE_ID.fullmatch(row["id"]) and row["id"] not in (".", ".."), "Unsafe sample ID")
            image = Path(row["image"])
            transport.require(image.is_file() and transport.digest(image.read_bytes()) == row["image_sha256"],
                              "Prepared image changed: " + row["id"])
        prompts = prompt_map(cfg)
        direct = cfg.get("experiment", {}).get("mode") == "baseline"
        snapshot = {"workflow": "direct_model_v1" if direct else VERSION, "config": cfg, "source": str(source),
                    "source_sha256": transport.digest(source.read_bytes()),
                    "selected_sha256": transport.digest(json.dumps(rows, sort_keys=True).encode()),
                    "implementation_sha256": {str(p.relative_to(BASE)): transport.digest(p.read_bytes()) for p in code_files()},
                    "note": ("Direct M(image, original question); no agent-role JSON constraint. No automatic scoring." if direct else
                             "Tool-backed U -> region request -> version-bound extraction -> text solver -> coordinate resolution -> visual check. "
                             "One targeted rework; one optional cell-query readback per select/check stage. No automatic scoring.")}
        if (root/"config.json").exists():
            transport.require(transport.read_json(root/"config.json") == snapshot,
                              "Frozen code, prompts, config or source changed; use a new --run name")
            transport.require(transport.read_json(root/"selected.json") == rows, "Frozen samples changed")
            transport.require(transport.read_json(root/"prompts_snapshot.json") == prompts, "Frozen prompts changed")
        else:
            transport.require(not any(p.name != "run.lock" for p in root.iterdir()), "New run directory must be empty")
            transport.save_json(root/"config.json", snapshot)
            transport.save_json(root/"selected.json", rows)
            transport.save_json(root/"prompts_snapshot.json", prompts)
            for file in code_files():
                dest = root/"code_snapshot"/file.relative_to(BASE)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(file.read_bytes())
        report(root)
        return root, rows, prompts
    finally:
        lock.unlink(missing_ok=True)


def report(root):
    root = Path(root)
    rows = transport.read_json(root/"selected.json")
    cfg = transport.read_json(root/"config.json")["config"] if (root/"config.json").exists() else {}
    detail_limit = cfg.get("report_detail_limit", len(rows))
    done = sum((root/"samples"/r["id"]/"result.json").exists() for r in rows)
    start = max(0, done-detail_limit)
    detail_ids = {r["id"] for r in rows[start:start+detail_limit]}
    results, cards, usage = [], [], Counter()
    role_usage = {}
    verification = Counter()
    errors = 0
    tool_calls, tool_seconds = 0, 0
    for row in rows:
        folder = root/"samples"/row["id"]
        result_path = folder/"result.json"
        result = transport.read_json(result_path) if result_path.exists() else None
        if result:
            results.append(result)
            verification[result["verification_status"]] += 1
        elif (folder/"error.json").exists():
            errors += 1
        events = folder/"tool_events.jsonl"
        if events.exists():
            for line in events.read_text(encoding="utf-8").splitlines():
                if line:
                    event = json.loads(line)
                    tool_calls += 1
                    tool_seconds += event["seconds"]
        details = []
        for path in sorted(folder.glob("r*.json")) if row["id"] in detail_ids else []:
            record = transport.read_json(path)
            details.append("<details><summary>"+escape(path.stem)+"</summary><pre>"+
                           escape(json.dumps(record.get("output", record), ensure_ascii=False, indent=2))+"</pre></details>")
            if path.name.endswith("03_extract.json"):
                for subtable in record["subtables"]:
                    # Source HTML was sanitized by table_tool. Sandbox is an extra display boundary.
                    frame = '<style>table{border-collapse:collapse}td,th{border:1px solid #999;padding:4px}</style>'+subtable["html"]
                    details.append('<iframe sandbox="" srcdoc="'+escape(frame, quote=True)+'"></iframe>')
        for path in folder.glob("attempts/*.json"):
            record = transport.read_json(path)
            counter = role_usage.setdefault(record["role"], Counter())
            counter["calls"] += 1
            counter["seconds"] += record["seconds"]
            tokens = record["raw_response"].get("usage", {})
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if type(tokens.get(key)) is int:
                    counter[key] += tokens[key]
            reasoning = tokens.get("completion_tokens_details", {}).get("reasoning_tokens")
            if type(reasoning) is int:
                counter["reasoning_tokens"] += reasoning
        label = result["verification_status"] if result else "error" if (folder/"error.json").exists() else "pending"
        if row["id"] not in detail_ids:
            continue
        image = Path(row["image"]).as_uri()
        cards.append('<article data-status="'+label+'"><h2>'+escape(row["id"])+" · "+escape(row.get("task_type", ""))+"</h2>"+
                     '<p class="badge">执行核查状态：'+label+'（不是人工正确性标签）</p>'+
                     '<img loading="lazy" src="'+escape(image, quote=True)+'"><h3>原题</h3><pre>'+escape(row["question"])+
                     "</pre><h3>最终候选回答</h3><pre>"+escape(result["answer"] if result else "尚无结果")+
                     "</pre><details><summary>流程记录与终止原因</summary><pre>"+
                     escape(json.dumps(result, ensure_ascii=False, indent=2) if result else "pending")+
                     "</pre></details>"+"".join(details)+"</article>")
    for counter in role_usage.values():
        usage.update(counter)
    summary = {"selected": len(rows), "completed": len(results), "errors": errors,
               "verification_status": dict(verification), "reworked": sum(r["rework_used"] for r in results),
               "execution_counts": {"model_calls": usage.get("calls", 0), "tool_calls": tool_calls,
                                    "rework_rounds": sum(r["counts"]["rework_rounds"] for r in results)},
               "tool_seconds": round(tool_seconds, 6),
               "api_usage": dict(usage), "by_role": {k: dict(v) for k, v in role_usage.items()},
               "note": "No automatic accuracy calculation. passed means checker acceptance only. "
                       "Recorded response seconds are summed request durations, not end-to-end wall time. "
                       "Unreturned/failed transport attempts may have unreported provider token usage."}
    transport.save_json(root/"results.json", results)
    transport.save_json(root/"summary.json", summary)
    (root/"predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in results), encoding="utf-8")
    options = "".join('<option value="'+s+'">'+s+'</option>' for s in ("passed", "needs_revision", "uncertain", "tool_failed", "pending", "error"))
    page = ('<!doctype html><meta charset="utf-8"><title>Region feedback loop</title>'
            '<style>body{font-family:system-ui;margin:24px;background:#f5f7fa}article{background:white;padding:20px;margin:20px 0}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%;max-height:650px}iframe{width:100%;height:280px;border:0}'
            'summary{cursor:pointer;padding:8px}.badge{color:#92400e}</style>'
            '<h1>外部工具接入版：区域提取与答案反馈闭环</h1><p>模型核查通过不等于答案正确。汇总含全部样本；网页显示至多'+str(detail_limit)+'条附近样本，完整输出保存在samples目录。</p><pre>'+escape(json.dumps(summary, ensure_ascii=False, indent=2))+
            '</pre><label>核查状态 <select id="filter"><option value="all">全部</option>'+options+'</select></label>'+"".join(cards)+
            '<script>document.getElementById("filter").onchange=function(){document.querySelectorAll("article").forEach(a=>a.hidden=this.value!=="all"&&a.dataset.status!==this.value)}</script>')
    (root/"review.html").write_text(page, encoding="utf-8")
    return summary


def run(cfg, name=DEFAULT_RUN, source=DEFAULT_SOURCE, limit=18, max_samples=0, retry_errors=False):
    transport.require(type(max_samples) is int and max_samples >= 0, "--max-samples must be nonnegative")
    root, rows, prompts = prepare(cfg, name, source, limit)
    lock = acquire_lock(root)
    code, session = 0, None
    try:
        pending = [row for row in rows if not (root/"samples"/row["id"]/"result.json").exists()
                   and (retry_errors or not (root/"samples"/row["id"]/"error.json").exists()
                        or transport.saved_error_is_transient(root/"samples"/row["id"]/"error.json"))]
        if max_samples:
            pending = pending[:max_samples]
        keys = transport.load_api_keys(cfg) if pending else {}
        progress = transport.RunProgress(len(pending), len(rows), sum((root/"samples"/r["id"]/"result.json").exists() for r in rows))
        client = transport.RoleApiClient(cfg, keys, root)
        client.progress = progress.show
        session = {"id": uuid.uuid4().hex, "workers": 1, "pid": os.getpid(), "started_at": transport.stamp(),
                   "proxy_mode": transport.proxy_mode(cfg),
                   "selected_ids_this_session": [r["id"] for r in pending]}
        transport.save_json(root/"sessions"/(session["id"]+".json"), session)
        live_cfg = dict(cfg, _retry_protocol_errors=retry_errors)
        print("Understand -> Registered table tools -> Select -> Extract -> Solve -> Resolve coordinates -> Check | workers: 1")
        print("API proxy mode: " + transport.proxy_mode(cfg) + " (client policy; does not override TUN or system routes).")
        print("At most ONE targeted rework. Details: run.log; failed samples do not stop subsequent samples.")
        for row in pending:
            folder = root/"samples"/row["id"]
            for attempt in range(cfg.get("sample_retry_rounds", 2)+1):
                try:
                    run_sample(row, folder, live_cfg, prompts, client, root/"run.log")
                    (folder/"error.json").unlink(missing_ok=True)
                    progress.completed += 1
                    break
                except (ValueError, transport.ApiFailure, KeyError, TypeError) as exc:
                    fatal = isinstance(exc, transport.ApiFailure) and exc.fatal
                    transient = isinstance(exc, transport.ApiFailure) and exc.retryable and not fatal
                    error = {"id": row["id"], "time": transport.stamp(), "message": str(exc), "type": type(exc).__name__,
                             "retryable": transient, "fatal": fatal, "sample_attempt": attempt+1}
                    if isinstance(exc, ProtocolError):
                        error.update(role=exc.role, stage=exc.stage, protocol_errors=exc.errors)
                    transport.save_json(folder/"error.json", error)
                    transport.log(root/"run.log", row["id"]+" ERROR "+str(exc))
                    if fatal:
                        code = 1
                        raise
                    if transient and attempt < cfg.get("sample_retry_rounds", 2):
                        transport.wait_with_progress(cfg.get("sample_retry_delay_seconds", 30), progress.show,
                                                     row["id"], "sample retry", "saved stages retained")
                        continue
                    progress.errors += 1
                    code = 1
                    break
            progress.finished += 1
            progress.show()
            if progress.finished % cfg.get("report_every_samples", 1) == 0 or progress.finished == len(pending):
                report(root)
    except KeyboardInterrupt:
        code = 130
        print("\nInterrupted. Completed responses/stages are retained; rerun the same command to resume.")
        transport.log(root/"run.log", "Interrupted; saved stages retained")
    except transport.ApiFailure as exc:
        code = 1
        print("\nStopped for fatal API/account or prolonged outage failure: "+str(exc))
    finally:
        try:
            summary = report(root)
            if session:
                session.update(ended_at=transport.stamp(), exit_code=code, interrupted=code == 130)
                transport.save_json(root/"sessions"/(session["id"]+".json"), session)
            print("\nLog: "+str(root/"run.log")+"\nResults: "+str(root/"review.html"))
            print("Completed: "+str(summary["completed"])+"; checker states: "+json.dumps(summary["verification_status"]))
        finally:
            lock.unlink(missing_ok=True)
    return code
