"""Typed handoff contracts. Table mechanics are checked separately by table_tool."""
from .wire_json import DECODER, _object_spans, ProtocolError
import re

VERSION = "tool_messages_v2"
FORMAT_RETRIES = 1


def errors_for(role, obj):
    errors = []
    def field(obj, key, typ):
        if not isinstance(obj, dict) or type(obj.get(key)) is not typ:
            errors.append(key + " must be " + typ.__name__)
            return None
        return obj[key]
    def enum(obj, key, choices):
        value = field(obj, key, str)
        if value is not None and value not in choices:
            errors.append(key + " must be one of " + ", ".join(choices))
        return value
    if type(obj) is not dict:
        return ["Return one JSON object"]
    if role == "understand":
        enum(obj, "schema_version", ("table_html_v3",))
        enum(obj, "coverage", ("complete", "partial"))
        tables = field(obj, "tables", list)
        field(obj, "uncertainties", list)
        for table in tables or []:
            field(table, "id", str)
            field(table, "html", str)
    elif role == "select":
        action = enum(obj, "action", ("extract", "inspect"))
        if action == "inspect":
            query_errors(obj, errors)
        else:
            regions = field(obj, "regions", list)
            field(obj, "uncertainties", list)
            for item in regions or []:
                field(item, "id", str)
                field(item, "source_table_id", str)
                for key in ("rows", "columns"):
                    pair = field(item, key, list)
                    if pair is not None and (len(pair) != 2 or any(type(v) is not int for v in pair)):
                        errors.append(key + " must contain [first,last] integers")
    elif role == "solve":
        enum(obj, "status", ("answered", "ambiguous", "insufficient_information"))
        field(obj, "answer", str)
        field(obj, "review_request", str)
        if "coordinate_requests" in obj:
            requests = field(obj, "coordinate_requests", list)
            if requests is not None and len(requests) > 256:
                errors.append("At most 256 coordinate references per answer")
            for request in requests or []:
                rid = field(request, "id", str)
                if rid is not None and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", rid):
                    errors.append("coordinate request id must be a short identifier")
                field(request, "subtable_id", str)
                position = field(request, "position", list)
                if position is not None and (len(position) != 2 or any(type(v) is not int or v < 1 for v in position)):
                    errors.append("coordinate position must be [row,column], 1-based integers")
                enum(request, "component", ("position", "top_left", "bottom_right", "row", "column", "rowspan", "colspan"))
                enum(request, "format", ("tuple", "json"))
                base = field(request, "output_base", int)
                if base is not None and base not in (0, 1):
                    errors.append("output_base must be 0 or 1")
    elif role == "check":
        action = enum(obj, "action", ("judge", "inspect"))
        if action == "inspect":
            query_errors(obj, errors)
            return errors
        status = enum(obj, "status", ("passed", "needs_revision", "uncertain"))
        if "return_to" in obj:
            errors.append("Do not output return_to; the host routes from each issue's source_stage")
        issues = field(obj, "issues", list)
        for item in issues or []:
            for name in ("observation", "conflict", "location"):
                value = field(item, name, str)
                if value is not None and not value.strip():
                    errors.append(name + " must not be empty")
            enum(item, "source_stage", ("understand", "select", "solve", "unknown"))
        if status == "passed" and issues:
            errors.append("passed requires issues=[]")
        if status == "needs_revision":
            if not issues:
                errors.append("needs_revision requires concrete issues")
            elif any(type(i) is dict and i.get("source_stage") == "unknown" for i in issues):
                errors.append("Use uncertain when an issue's source_stage is unknown")
    else:
        raise ValueError("Unknown role " + role)
    return errors


def query_errors(obj, errors):
    queries = obj.get("queries")
    if type(queries) is not list or not 1 <= len(queries) <= 32:
        errors.append("inspect requires 1..32 queries")
        return
    for query in queries:
        if type(query) is not dict or type(query.get("source_table_id")) is not str:
            errors.append("Each query needs source_table_id")
            continue
        positions = query.get("positions")
        if type(positions) is not list or not 1 <= len(positions) <= 64:
            errors.append("Each query needs 1..64 positions")
        elif any(type(p) is not list or len(p) != 2 or any(type(v) is not int or v < 1 for v in p) for p in positions):
            errors.append("Query positions must be 1-based [row,column] integers")


def decode(text, role):
    if not isinstance(text, str) or not text.strip():
        raise ProtocolError(["Empty response content"], role)
    stripped = text.strip()
    try:
        value = DECODER.decode(stripped)
    except ValueError:
        matches = []
        for start, end in _object_spans(text):
            try:
                candidate = DECODER.decode(text[start:end])
            except ValueError:
                continue
            if not errors_for(role, candidate):
                matches.append(candidate)
        if len(matches) != 1:
            raise ProtocolError(["Expected one unambiguous role JSON object"], role)
        value = matches[0]
    errors = errors_for(role, value)
    if errors:
        raise ProtocolError(errors, role)
    return value
