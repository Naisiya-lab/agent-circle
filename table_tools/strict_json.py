"""Strict JSON only: reject duplicate keys, NaN, Infinity and competing objects."""
import json


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def constant(value):
    raise ValueError("Non-standard JSON constant: " + value)


DECODER = json.JSONDecoder(object_pairs_hook=pairs, parse_constant=constant)
