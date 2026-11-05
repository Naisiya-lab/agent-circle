"""Minimal executable message interfaces, without task-correctness judgments."""
import re
from .wire_json import DECODER, _object_spans, ProtocolError

VERSION = "tool_messages_v3"
FORMAT_RETRIES = 2


def errors_for(role, obj):
    errors = []
    def field(value, key, typ):
        item = value.get(key) if isinstance(value, dict) else None
        if type(item) is not typ:
            errors.append(key + " must be " + typ.__name__)
            return None
        return item
    if not isinstance(obj, dict):
        return ["Return one JSON object"]
    if role in ("understand", "select", "solve", "check") and obj.get("action") == "view_image":
        return []
    if role == "select_direct":
        for table in field(obj, "subtables", list) or []:
            for key in ("id", "source_table_id", "html"):
                field(table, key, str)
            for key in ("source_rows", "source_columns"):
                field(table, key, list)
        return errors
    if role == "solve_direct":
        field(obj, "answer", str)
        if obj.get("coordinate_requests") or "{{coord:" in str(obj.get("answer", "")):
            errors.append("Write final coordinates directly in answer")
        return errors
    if role in ("check_final", "understand_final"):
        if obj.get("action") in ("inspect", "view_image"):
            return ["Return the final judgment" if role == "check_final" else "Return the HTML tables"]
        role = role.split("_")[0]
    if role == "understand":
        for table in field(obj, "tables", list) or []:
            field(table, "id", str)
            field(table, "html", str)
    elif role in ("select", "check"):
        action = field(obj, "action", str)
        if action == "inspect":
            query_errors(obj, errors)
        elif role == "select" and action == "extract":
            for region in field(obj, "regions", list) or []:
                field(region, "id", str)
                field(region, "source_table_id", str)
                for name in ("rows", "columns"):
                    pair = field(region, name, list)
                    if pair is not None and (len(pair) != 2 or any(type(v) is not int for v in pair)):
                        errors.append(name + " needs [first,last] integers for extraction")
        elif role == "check" and action == "judge":
            field(obj, "status", str)
            issues = field(obj, "issues", list)
            if issues is not None and any(not isinstance(i, dict) for i in issues):
                errors.append("issues must contain objects")
            # No status/issue consistency rules or required explanatory prose.
        elif action is not None:
            errors.append("action must be inspect or " + ("extract" if role == "select" else "judge"))
    elif role == "solve":
        field(obj, "answer", str)
        # Optional coordinate requests are handled by the requested tool, not a
        # pre-handoff answer gate. Direct coordinates/text are valid messages.
    else:
        raise ValueError("Unknown role " + role)
    return errors


def query_errors(obj, errors):
    queries = obj.get("queries")
    if not isinstance(queries, list):
        errors.append("inspect requires a queries list")
        return
    for query in queries:
        if not isinstance(query, dict) or not isinstance(query.get("source_table_id"), str):
            errors.append("Each query needs source_table_id")
            continue
        positions = query.get("positions")
        if not isinstance(positions, list) or any(not isinstance(p, list) or len(p) != 2 or any(type(v) is not int for v in p) for p in positions):
            errors.append("Query positions must contain [row,column] integers")


def decode(text, role):
    if not isinstance(text, str) or not text.strip():
        raise ProtocolError(["Empty response content"], role)
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.S | re.I)
    if fence:
        stripped = fence.group(1)
    try:
        value = DECODER.decode(stripped)
    except ValueError as original:
        spans = list(_object_spans(stripped))
        if len(spans) != 1:
            raise ProtocolError(["JSON cannot be read: " + str(original)], role) from original
        start, end = spans[0]
        try:
            value = DECODER.decode(stripped[start:end])
        except ValueError as exc:
            raise ProtocolError(["JSON cannot be read: " + str(exc)], role) from exc
    errors = errors_for(role, value)
    if errors:
        raise ProtocolError(errors, role)
    return value
