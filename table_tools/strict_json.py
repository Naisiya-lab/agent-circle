"""Read JSON without guessing between conflicting values."""
import json


def same_value(left, right):
    return json.dumps(left, sort_keys=True, ensure_ascii=False, allow_nan=False) == json.dumps(right, sort_keys=True, ensure_ascii=False, allow_nan=False)


def pairs(items):
    result = {}
    for key, value in items:
        if key in result and not same_value(result[key], value):
            raise ValueError("Conflicting duplicate JSON key: " + key)
        result[key] = value
    return result


def constant(value):
    raise ValueError("Non-standard JSON constant: " + value)


DECODER = json.JSONDecoder(object_pairs_hook=pairs, parse_constant=constant)
