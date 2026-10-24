"""Provider-neutral function definitions and strict JSON argument validation."""
from __future__ import annotations

from copy import deepcopy

VERSION = "table_programs_v1"


def obj(properties, required):
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


STRING = {"type": "string", "minLength": 1}
POSITION = {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 2, "maxItems": 2}
POSITIONS = {"type": "array", "items": POSITION, "minItems": 1, "maxItems": 64}
REGION = obj({"id": STRING, "source_table_id": STRING, "rows": POSITION, "columns": POSITION},
             ["id", "source_table_id", "rows", "columns"])

SPECS = {
    "list_tables": {
        "roles": ["select", "check"],
        "description": "Return program-counted table sizes and indexed HTML for the current recognized-table version. This does not verify fidelity to the image.",
        "description_zh": "读取当前版本的逻辑行列数和带位置的全表 HTML。只反映识别结果，不代表已核实原图。",
        "parameters": obj({"table_version": STRING}, ["table_version"]),
    },
    "extract_subtables": {
        "roles": ["select"],
        "description": "Extract one or more inclusive, 1-based rectangles from the recognized tables. Expand through intersecting merged cells until all are complete. Preserve blanks, text, spans and source positions. Return an immutable extraction ID; separated regions remain separate.",
        "description_zh": "按原表从1开始的闭区间提取一个或多个矩形，自动扩大至完整保留相交合并单元格。保留空白、文本、跨度和来源坐标；分离区域不拼接。返回不可变的提取结果ID。",
        "parameters": obj({"table_version": STRING,
                           "regions": {"type": "array", "items": REGION, "minItems": 1, "maxItems": 32}},
                          ["table_version", "regions"]),
    },
    "inspect_cells": {
        "roles": ["select", "check"],
        "description": "Read specified logical positions in a recognized full table. Return each queried position, the owning merged-cell anchor/range, and unchanged cell content. Source positions are always 1-based. This tool does not read image pixels.",
        "description_zh": "查询全表指定逻辑位置的原文、所属单元格锚点及合并范围。原表查询位置固定从1开始。此工具不读取图片，也不判断原文是否识别正确。",
        "parameters": obj({"table_version": STRING, "source_table_id": STRING, "positions": POSITIONS},
                          ["table_version", "source_table_id", "positions"]),
    },
    "map_coordinates": {
        "roles": ["solve", "check"],
        "description": "Map positions between a program-generated subtable and its recognized source table. Use the extraction ID returned by extract_subtables. Inputs are 1-based; output_base defaults to 1 and may be 0 only when the task requires it. Return both the mapped grid position and the merged-cell anchor/range, without confusing them.",
        "description_zh": "使用程序生成的提取结果ID，将子表坐标与原表坐标相互换算。输入始终从1开始；output_base默认1，原题明确要求时可设0。分别返回所查询网格位置与所属合并单元格的锚点、范围。",
        "parameters": obj({"table_version": STRING, "extraction_id": STRING, "subtable_id": STRING,
                           "direction": {"type": "string", "enum": ["to_source", "to_subtable"]},
                           "positions": POSITIONS, "output_base": {"type": "integer", "enum": [0, 1]}},
                          ["table_version", "extraction_id", "subtable_id", "direction", "positions"]),
    },
}


def definitions_for(role):
    """Host chooses the role. A model cannot grant itself more tool permissions."""
    if role not in ("understand", "select", "solve", "check"):
        raise ValueError("Unknown agent role")
    return [{"type": "function", "function": {"name": name, "description": spec["description"],
                                               "parameters": deepcopy(spec["parameters"])}}
            for name, spec in SPECS.items() if role in spec["roles"]]


def validate(value, schema, path="arguments"):
    """Only the JSON-schema subset used above; reject booleans as integers."""
    types = {"object": dict, "array": list, "string": str, "integer": int}
    if type(value) is not types[schema["type"]]:
        return [path + " must be " + schema["type"]]
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(path + " must be one of " + str(schema["enum"]))
    if schema["type"] == "object":
        for key in schema["required"]:
            if key not in value:
                errors.append(path + "." + key + " is required")
        for key, item in value.items():
            if type(key) is not str:
                errors.append(path + " must have string JSON keys")
                continue
            if key not in schema["properties"]:
                errors.append(path + "." + key + " is not allowed")
            else:
                errors.extend(validate(item, schema["properties"][key], path+"."+key))
    elif schema["type"] == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            errors.append(path + " has invalid length")
        for number, item in enumerate(value):
            errors.extend(validate(item, schema["items"], f"{path}[{number}]"))
    elif schema["type"] == "integer" and value < schema.get("minimum", value):
        errors.append(path + " is below its minimum")
    elif schema["type"] == "string" and len(value) < schema.get("minLength", 0):
        errors.append(path + " must not be empty")
    return errors
