"""Host-bound tool execution, coordinate references and communication metadata."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import time

from table_tools import TableTools, ProgramError
from . import transport

REFERENCE = re.compile(r"\{\{coord:([^{}]+)\}\}")
TRANSIENT_TOOL_ERRORS = {"STATE_IO_ERROR"}


def digest_value(value):
    return transport.digest(json.dumps(value, ensure_ascii=False, sort_keys=True).encode())


def failure(code, message, **details):
    return {"ok": False, "data": None, "error": {"code": code, "message": message, "details": details}}


class ToolSession:
    """Execute tools without turning execution errors into content-routing decisions."""
    def __init__(self, row, understanding, sample_dir, prefix, cfg=None, budgets=None):
        self.sample_dir, self.prefix = Path(sample_dir), prefix
        self.cfg = cfg or {}
        self.budgets = budgets
        self.operation_scope = ""
        self.service = None
        self.understanding = understanding
        self.registration = None
        for attempt in range(self.cfg.get("tools", {}).get("execution_retries", 2)+1):
            started = time.monotonic()
            try:
                self.service = TableTools(row["id"], understanding, state_dir=self.sample_dir/"tools")
                self.registration = {"ok": True, "context_id": self.service.context_id,
                                     "table_version": self.service.table_version}
            except ProgramError as exc:
                self.registration = {"ok": False, "data": None, "error": exc.record()}
            self.event("host", "register_tables", {"sample_id": row["id"]}, self.registration, started, attempt)
            if self.registration["ok"] or self.registration["error"]["code"] not in TRANSIENT_TOOL_ERRORS:
                break

    def event(self, role, name, args, response, started, retry=0):
        record = {"time": transport.stamp(), "round": self.prefix, "role": role, "tool": name,
                  "arguments": args, "response": response, "execution_retry": retry,
                  "seconds": round(time.monotonic()-started, 6)}
        with (self.sample_dir/"tool_events.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False)+"\n")

    def call(self, role, name, arguments, stage, *, target=None, required=True):
        response = self.registration
        for attempt in range(self.cfg.get("tools", {}).get("execution_retries", 2)+1):
            started = time.monotonic()
            args = dict(arguments, table_version=self.service.table_version) if self.service else arguments
            token = None
            if self.budgets is not None:
                key = digest_value({"stage": stage, "scope": self.operation_scope, "tool": name,
                    "arguments": args, "registration": self.registration, "attempt": attempt,
                    "retry_plan": self.cfg.get("_tool_retry_plan", {}).get("id")})
                token = self.budgets.take(role, "tool", key)
                if token is None:
                    response = failure("TOOL_BUDGET_EXHAUSTED", "This role has used its per-sample tool allowance")
                    break
                cached = self.budgets.receipt(token)
                if cached is not None:
                    response = cached
                    if response["ok"] or response["error"]["code"] not in TRANSIENT_TOOL_ERRORS:
                        break
                    continue
            if self.service is None:
                response = self.registration
            else:
                # Sample/table identity comes from the host's active session.
                response = self.service.call(role, name, args)
            if token is not None:
                self.budgets.complete(token, response)
            self.event(role, name, args, response, started, attempt)
            if response["ok"] or response["error"]["code"] not in TRANSIENT_TOOL_ERRORS or self.service is None:
                break
        transport.save_json(self.sample_dir/(stage+".json"), {"role": "program", "tool": name, "input": args, "output": response})
        return response

    def catalog(self):
        reply = self.call("select", "list_tables", {}, self.prefix+"00t_index")
        if reply["ok"]:
            return dict(reply["data"], index_available=True)
        # HTML remains structured message content even when no reliable grid
        # exists. No row counts or mappings are invented for these tables.
        return {"tables": self.understanding["tables"], "index_available": False, "tool_response": reply}

    def extract(self, selection, indexed):
        regions = selection["regions"]
        fallback = regions == []
        if fallback and self.service is not None:
            regions = [{"id": f"full_{number}", "source_table_id": table["id"],
                        "rows": [1, table["row_count"]], "columns": [1, table["column_count"]]}
                       for number, table in enumerate(indexed["tables"], 1)]
        resolution = {"mode": "full_table_after_empty_selection" if fallback else "selected_regions",
                      "requested_regions": selection["regions"], "effective_requests": regions}
        transport.save_json(self.sample_dir/(self.prefix+"selection_resolution.json"), resolution)
        program_regions = [{k: region[k] for k in ("id", "source_table_id", "rows", "columns")} for region in regions]
        response = self.call("select", "extract_subtables", {"regions": program_regions}, self.prefix+"03t_extract")
        if response["ok"]:
            data = dict(response["data"], mapping_available=True)
        else:
            data = {"subtables": [], "available_tables": self.understanding["tables"],
                    "mapping_available": False, "extraction_id": None}
        data.update(selection_resolution=resolution, tool_response=response)
        transport.save_json(self.sample_dir/(self.prefix+"03_extract.json"), {"role": "program", **data})
        return data


def route_checker(judgment):
    """Dispatch from the agent's explicit stage labels; do not interpret its prose."""
    stages = [issue.get("source_stage") for issue in judgment.get("issues", [])]
    priority = ("understand", "select", "solve")
    explicit = judgment.get("return_to")
    target = explicit if explicit in priority else "none"
    if target == "none" and judgment.get("status") != "passed" and "unknown" not in stages:
        target = next((stage for stage in priority if stage in stages), "none")
    return dict(judgment, return_to=target)


def wire_subtables(extraction):
    fields = ("id", "source_table_id", "table_version", "source_rows", "source_columns", "html")
    return [{key: table[key] for key in fields if key in table} for table in extraction["subtables"]]


def resolve_answer(solver, session, extraction):
    """Convert requested references; retain direct answers and resolved portions."""
    requests = solver.get("coordinate_requests", [])
    tokens = set(REFERENCE.findall(solver["answer"]))
    errors, receipts, resolved, by_id = [], [], {}, {}
    ignored = 0
    for item in requests if isinstance(requests, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] not in tokens:
            ignored += 1
            continue
        rid = item["id"]
        if rid in by_id and by_id[rid] != item:
            errors.append(failure("CONFLICTING_COORDINATE_REQUEST", "One reference has conflicting requests", reference=rid))
        else:
            by_id[rid] = item
    conflicted = {x["error"]["details"]["reference"] for x in errors}
    for rid in tokens - set(by_id):
        errors.append(failure("MISSING_COORDINATE_REQUEST", "No request for this reference", reference=rid))
    groups = defaultdict(list)
    for rid, request in by_id.items():
        if rid in conflicted:
            continue
        component = request.get("component")
        if component not in ("position", "top_left", "bottom_right", "row", "column", "rowspan", "colspan") or request.get("format") not in ("tuple", "json"):
            errors.append(failure("INVALID_COORDINATE_ARGUMENTS", "Provide a known component and output format", reference=rid))
            continue
        position = request.get("position")
        if not isinstance(request.get("subtable_id"), str) or not isinstance(position, list) or len(position) != 2 or any(type(v) is not int for v in position) or type(request.get("output_base", 1)) is not int or request.get("output_base", 1) not in (0, 1):
            errors.append(failure("INVALID_COORDINATE_ARGUMENTS", "Provide subtable_id, [row,column] integers and output_base 0 or 1", reference=rid))
            continue
        groups[request["subtable_id"], request.get("output_base", 1)].append(request)
    counter = 0
    batch_size = max(1, int(session.cfg.get("tools", {}).get("position_batch_size", 64)))
    for (subtable_id, base), members in groups.items():
        positions = list(dict.fromkeys(tuple(r["position"]) for r in members))
        mapped = {}
        for start in range(0, len(positions), batch_size):
            counter += 1
            reply = session.call("solve", "map_coordinates", {"extraction_id": extraction.get("extraction_id"),
                "subtable_id": subtable_id, "direction": "to_source", "positions": [list(p) for p in positions[start:start+batch_size]],
                "output_base": base}, session.prefix+f"04t_map_{counter:02d}")
            receipts.append(reply)
            if not reply["ok"]:
                errors.append(reply)
                continue
            mapped.update((tuple(x["input"]["position"]), x["mapped"]) for x in reply["data"]["positions"])
        for request in members:
            result = mapped.get(tuple(request["position"]))
            if result is None:
                continue
            values = {"position": result["position"], "top_left": result["cell_anchor"],
                      "bottom_right": [result["cell_rows"][1], result["cell_columns"][1]],
                      "row": result["position"][0], "column": result["position"][1],
                      "rowspan": result["cell_rows"][1]-result["cell_rows"][0]+1,
                      "colspan": result["cell_columns"][1]-result["cell_columns"][0]+1}
            value = values[request["component"]]
            text = ("("+", ".join(str(v) for v in value)+")") if type(value) is list and request["format"] == "tuple" else json.dumps(value)
            resolved[request["id"]] = {"value": value, "text": text}
    # Preserve unmatched placeholders and direct numeric coordinates. They are
    # visible to the solver/checker, never silently deleted or guessed.
    answer = REFERENCE.sub(lambda m: resolved[m[1]]["text"] if m[1] in resolved else m[0], solver["answer"])
    if "{{coord:" in REFERENCE.sub("", answer):
        errors.append(failure("INVALID_COORDINATE_REFERENCE", "An incomplete coordinate reference remains in the answer"))
    candidate = dict(solver, answer=answer)
    record = {"role": "program", "answer_template": solver["answer"], "resolved_references": resolved,
              "mapping_results": receipts, "errors": errors, "ignored_unused_requests": ignored,
              "unresolved_references": sorted(tokens - set(resolved)), "candidate": candidate}
    transport.save_json(session.sample_dir/(session.prefix+"04_coordinates.json"), record)
    return candidate, record


def call_counts(folder):
    paths = list((Path(folder)/"attempts").glob("*.json"))
    records = [transport.read_json(p) for p in paths]
    events = []
    event_path = Path(folder)/"tool_events.jsonl"
    if event_path.exists():
        events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line]
    return {"model_calls": len(records), "tool_calls": len(events),
            "tool_queries": sum(r.get("output", {}).get("action") == "inspect" for r in records if isinstance(r.get("output"), dict)),
            "image_accesses": sum(r.get("image_count", int(r.get("image_attached", False))) for r in records),
            "format_repairs": sum(r.get("protocol", {}).get("format_retry_index", 0) > 0 for r in records),
            "tool_input_repairs": len({r["stage"] for r in records if "_tool_repair_" in r["stage"]}),
            "tool_execution_retries": sum(e.get("execution_retry", 0) > 0 for e in events)}
