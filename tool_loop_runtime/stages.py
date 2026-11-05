"""U -> region S -> table tool -> F -> answer C, at most one targeted rework."""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
import time
import uuid

from . import transport
from .protocol import VERSION, FORMAT_RETRIES, ProtocolError, decode, errors_for
from .providers import profile, build_request


def payload(role, row, context, cfg, prompt, attach_image=None):
    content = json.dumps(context, ensure_ascii=False)
    image_attached = transport.uses_image(cfg, role) if attach_image is None else attach_image
    if image_attached:
        image = Path(row["image"])
        data = image.read_bytes()
        transport.require(transport.digest(data) == row["image_sha256"], "Prepared image changed")
        mime = mimetypes.guess_type(str(image))[0] or "image/jpeg"
        content = [{"type": "image_url", "image_url": {
            "url": "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")}},
                   {"type": "text", "text": content}]
    if profile(cfg, role) is not None:
        return build_request(cfg, role, [{"role": "system", "content": prompt}, {"role": "user", "content": content}])
    body = {"model": cfg["models"][role], "stream": cfg.get("stream", True),
            "max_tokens": cfg["max_tokens"][role],
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": content}]}
    if role in ("select", "solve") and not transport.uses_image(cfg, role):
        body.update(transport.text_api(cfg, role)["request_options"])
        if body["stream"]:
            body["stream_options"] = {"include_usage": True}
    else:
        body["thinking"] = {"type": cfg["thinking"]}
        body["reasoning_effort"] = cfg["reasoning_effort"][role]
    return body


def _format_request(body, role, prompt, previous_text, errors):
    # Repair representation only, on the same model. Do not resend an image or
    # the complete upstream history to fix duplicated keys/JSON punctuation.
    fields = {
        "understand": "tables: [{id: string, html: string}]",
        "select": "action: extract, regions: [{id, source_table_id, rows: [first,last], columns: [first,last]}]; or action: inspect, queries: [{source_table_id, positions}]",
        "solve": "answer: string; optional coordinate_requests for program conversion",
        "check": "action: judge, status: string, issues: list; or action: inspect, queries: list",
        "select_direct": "subtables: [{id, source_table_id, html, source_rows: list, source_columns: list}]; uncertainties: list",
        "solve_direct": "answer: string containing the final response and literal final coordinates; status and review_request",
        "check_final": "action: judge, status: string, issues: list",
        "understand_final": "tables: [{id: string, html: string}]",
    }
    repair = {"previous_output": previous_text, "interface_errors": errors,
              "required_interface": fields[role],
              "instruction": "Correct only the JSON representation and required interface fields. Preserve the existing table text, numbers, region decisions and answer. Do not invent missing content."}
    return dict(body, messages=[{"role": "system", "content": prompt},
                                {"role": "user", "content": json.dumps(repair, ensure_ascii=False)}])


def invoke(role, stage, row, context, cfg, prompt, client, sample_dir, log_path, *, interface=None, attach_image=None):
    interface = interface or role
    path = sample_dir / (stage + ".json")
    retry_plan = cfg.get("_tool_retry_plan")
    if retry_plan and role==retry_plan["owner"] and stage.startswith(f"r{retry_plan.get('content_reworks_used', 0):02d}_") and "_tool_repair_" not in stage and "_after_read_" not in stage:
        context=dict(context,tool_retry_context={"id":retry_plan["id"],"previous_output":retry_plan["previous_output"],
            "tool_errors":retry_plan["tool_errors"],"previous_table_version":retry_plan["previous_table_version"]})
    body = payload(role, row, context, cfg, prompt, attach_image)
    fingerprint = transport.semantic_request_fingerprint(body)
    record = transport.read_json(path) if path.exists() else None
    shared = cfg.get("initial_understanding_cache")
    if record is None and shared and role == "understand" and stage == "r00_01_understand":
        source = Path(shared)/"samples"/row["id"]/(stage+".json")
        transport.require(source.is_file(), "Shared understanding is not ready")
        reused = transport.read_json(source)
        transport.require(reused["base_request_fingerprint"] == fingerprint and reused["protocol"]["version"] == VERSION
                          and reused["protocol"]["status"] == "accepted"
                          and reused["endpoint"] == transport.api_binding(cfg, role)[0], "Shared understanding does not match this request")
        record = dict(reused, seconds=0.0, time=transport.stamp(),
                      reused_from={"path": str(source.resolve()), "sha256": transport.digest(source.read_bytes()),
                                   "source_seconds": reused["seconds"], "source_attempt_id": reused["attempt_id"]})
        transport.save_json(path, record)

    def persist():
        transport.save_json(sample_dir / "attempts" / (stage + "_" + record["attempt_id"] + ".json"), record)
        transport.save_json(path, record)

    if record is not None:
        if record.get("reused_from"):
            origin = record["reused_from"]
            transport.require(Path(origin["path"]).is_file() and transport.digest(Path(origin["path"]).read_bytes()) == origin["sha256"],
                              "Shared understanding changed after reuse; use a new run")
        transport.require(record["protocol"]["version"] == VERSION, "Cached protocol changed; use a new run")
        transport.require(record["endpoint"] == transport.api_binding(cfg, role)[0], "Cached endpoint changed")
        if record["base_request_fingerprint"] != fingerprint:
            # Frozen config/prompt/source checks happen before the sample runs.
            # A changed upstream/tool-feedback message needs a fresh request,
            # not a fatal cache gate or reuse of an unrelated response.
            transport.log(log_path, f"{row['id']} {stage} input changed; prior response remains in attempts")
            record = None
        if record is not None and record["protocol"]["status"] == "accepted":
            transport.log(log_path, f"{row['id']} {stage} resume cached")
            return record["output"]
        if record is not None and record["protocol"]["status"] == "transport_failed":
            record = None
        elif record is not None and record["protocol"]["status"] == "exhausted":
            if not cfg.get("_retry_protocol_errors"):
                raise ProtocolError(record["protocol"]["errors"], role, stage)
            record = None

    max_repairs = cfg.get("communication", {}).get("format_retries", FORMAT_RETRIES)
    while True:
        if record is None or record["protocol"]["status"] == "retry_pending":
            previous = record
            repair_index = 0 if previous is None else previous["protocol"]["format_retry_index"] + 1
            request = body if previous is None else _format_request(
                body, interface, prompt,
                previous["raw_response"]["choices"][0]["message"].get("content") or "",
                previous["protocol"]["errors"])
            transport.log(log_path, f"{row['id']} {stage} requesting format_retry={repair_index}")
            started = time.monotonic()
            failure = None
            try:
                raw = client.call(request, row["id"], stage, role=role)
            except transport.ApiFailure as exc:
                failure = exc
                raw = {"api_error": {"message": str(exc), "http_status": exc.http_status,
                                     "retryable": exc.retryable, "fatal": exc.fatal}}
            images = sum(1 for m in request["messages"] if isinstance(m.get("content"), list)
                         for item in m["content"] if item.get("type") == "image_url")
            record = {"role": role, "stage": stage, "time": transport.stamp(), "attempt_id": uuid.uuid4().hex,
                      "parent_attempt_id": previous["attempt_id"] if previous else None,
                      "tool_retry_id": retry_plan["id"] if retry_plan else None,
                      "base_request_fingerprint": fingerprint,
                      "request_fingerprint": raw.get("_transport", {}).get("effective_request_fingerprint", transport.semantic_request_fingerprint(request)), "input": context,
                      "image_attached": bool(images), "image_count": images, "interface": interface,
                      "model": cfg["models"][role], "endpoint": transport.api_binding(cfg, role)[0],
                      "request_options": profile(cfg, role).get("request_options", {}) if profile(cfg, role) is not None else {},
                      "requested_max_tokens": request.get("max_completion_tokens", request.get("max_tokens")),
                      "request_max_tokens": raw.get("_transport", {}).get("effective_max_tokens", request.get("max_completion_tokens", request.get("max_tokens"))),
                      "request_reasoning_effort": request.get("reasoning_effort"),
                      "seconds": round(time.monotonic()-started, 3), "raw_response": raw, "output": None,
                      "protocol": {"version": VERSION, "status": "transport_failed" if failure else "pending",
                                   "format_retry_index": repair_index, "errors": [str(failure)] if failure else []}}
            persist()
            if failure:
                failure.role, failure.stage = role, stage
                raise failure
        try:
            choice = record["raw_response"]["choices"][0]
            # A length-truncated response is accepted when its interface can be
            # read. Preserve finish_reason; do not equate truncation with failure.
            output = decode(choice["message"].get("content"), interface)
        except ProtocolError as exc:
            record["protocol"]["errors"] = exc.errors
            exhausted = record["protocol"]["format_retry_index"] >= max_repairs
            record["protocol"]["status"] = "exhausted" if exhausted else "retry_pending"
            persist()
            if exhausted:
                raise ProtocolError(exc.errors, role, stage) from exc
            continue
        record["output"] = output
        record["protocol"].update(status="accepted", errors=[])
        persist()
        transport.log(log_path, f"{row['id']} {stage} saved seconds={record['seconds']}")
        return output
