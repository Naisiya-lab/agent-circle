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


def payload(role, row, context, cfg, prompt):
    content = json.dumps(context, ensure_ascii=False)
    image_attached = transport.uses_image(cfg, role)
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
    if role in ("select", "solve") and not image_attached:
        body.update(transport.text_api(cfg, role)["request_options"])
        if body["stream"]:
            body["stream_options"] = {"include_usage": True}
    else:
        body["thinking"] = {"type": cfg["thinking"]}
        body["reasoning_effort"] = cfg["reasoning_effort"][role]
    return body


def invoke(role, stage, row, context, cfg, prompt, client, sample_dir, log_path):
    path = sample_dir / (stage + ".json")
    body = payload(role, row, context, cfg, prompt)
    fingerprint = transport.semantic_request_fingerprint(body)
    record = transport.read_json(path) if path.exists() else None

    shared = cfg.get("initial_understanding_cache")
    if record is None and shared and role == "understand" and stage == "r00_01_understand":
        source = Path(shared)/"samples"/row["id"]/(stage+".json")
        transport.require(source.is_file(), "Shared understanding is not ready; use mmtab_selector_ab.py run")
        reused = transport.read_json(source)
        transport.require(reused["base_request_fingerprint"] == fingerprint and reused["protocol"]["version"] == VERSION
                          and reused["protocol"]["status"] == "accepted" and not errors_for(role, reused["output"])
                          and reused["endpoint"] == transport.api_binding(cfg, role)[0], "Shared understanding does not match this request")
        record = dict(reused, seconds=0.0, time=transport.stamp(),
                      reused_from={"path": str(source.resolve()), "sha256": transport.digest(source.read_bytes()),
                                   "source_seconds": reused["seconds"], "source_attempt_id": reused["attempt_id"]})
        # Reuse is not a paid attempt. Keep the shared provenance in the stage file.
        transport.save_json(path, record)

    def persist():
        transport.save_json(sample_dir / "attempts" / (stage + "_" + record["attempt_id"] + ".json"), record)
        transport.save_json(path, record)

    if record is not None:
        if record.get("reused_from"):
            origin = record["reused_from"]
            transport.require(Path(origin["path"]).is_file() and transport.digest(Path(origin["path"]).read_bytes()) == origin["sha256"],
                              "Shared understanding changed after reuse; use a new run")
        transport.require(record["base_request_fingerprint"] == fingerprint and record["protocol"]["version"] == VERSION,
                          "Cached request/protocol changed; use a new run")
        transport.require(record["endpoint"] == transport.api_binding(cfg, role)[0], "Cached endpoint changed")
        if record["protocol"]["status"] == "accepted":
            transport.require(not errors_for(role, record["output"]), "Cached output violates interface")
            transport.log(log_path, f"{row['id']} {stage} resume cached")
            return record["output"]
        if record["protocol"]["status"] == "exhausted":
            if not cfg.get("_retry_protocol_errors"):
                raise ProtocolError(record["protocol"]["errors"], role, stage)
            record = None

    while True:
        if record is None or record["protocol"]["status"] == "retry_pending":
            previous = record
            retry = 0 if previous is None else previous["protocol"]["format_retry_index"] + 1
            request = body
            if previous is not None:
                previous_text = previous["raw_response"]["choices"][0]["message"].get("content") or ""
                feedback = ("Return one complete JSON object in your role's specified format. "
                            "Correct the representation only. Interface errors: " + "; ".join(previous["protocol"]["errors"]))
                request = dict(body, messages=body["messages"] + [{"role": "assistant", "content": previous_text},
                                                                 {"role": "user", "content": feedback}])
            transport.log(log_path, f"{row['id']} {stage} requesting format_retry={retry}")
            then = time.monotonic()
            raw = client.call(request, row["id"], stage, role=role)
            record = {"role": role, "stage": stage, "time": transport.stamp(), "attempt_id": uuid.uuid4().hex,
                      "parent_attempt_id": previous["attempt_id"] if previous else None,
                      "base_request_fingerprint": fingerprint,
                      "request_fingerprint": transport.semantic_request_fingerprint(request),
                      "input": context, "image_attached": transport.uses_image(cfg, role),
                      "model": cfg["models"][role], "endpoint": transport.api_binding(cfg, role)[0],
                      "request_options": profile(cfg, role).get("request_options", {}) if profile(cfg, role) is not None else (transport.text_api(cfg, role)["request_options"] if role in ("select", "solve") and not transport.uses_image(cfg, role) else {}),
                      "request_max_tokens": cfg["max_tokens"][role], "request_reasoning_effort": body.get("reasoning_effort"),
                      "seconds": round(time.monotonic()-then, 3), "raw_response": raw, "output": None,
                      "protocol": {"version": VERSION, "status": "pending", "format_retry_index": retry, "errors": []}}
            # Persist the provider response before decoding or consuming it.
            persist()
        try:
            choice = record["raw_response"]["choices"][0]
            if choice.get("finish_reason") not in (None, "stop"):
                raise ProtocolError(["Non-complete response: " + str(choice.get("finish_reason"))], role, stage)
            output = decode(choice["message"].get("content"), role)
        except ProtocolError as exc:
            record["protocol"]["errors"] = exc.errors
            exhausted = record["protocol"]["format_retry_index"] >= FORMAT_RETRIES
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

