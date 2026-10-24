"""Strict JSON parsing shared by the local role contracts."""

import json

class ProtocolError(ValueError):
    def __init__(self, errors, role=None, stage=None):
        self.errors = list(errors)
        self.role, self.stage = role, stage
        super().__init__("; ".join(self.errors))

def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result

def _constant(value):
    raise ValueError("non-standard JSON constant: " + value)

DECODER = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant)

def _object_spans(text):
    """Skip complete outer objects/arrays, including malformed JSON; never fish inside them."""
    i = 0
    while i < len(text):
        if text[i] not in "{[":
            i += 1
            continue
        start, stack, quoted, escaped = i, [], False, False
        while i < len(text):
            char = text[i]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack or (stack[-1], char) not in (("{", "}"), ("[", "]")):
                    break
                stack.pop()
                if not stack:
                    i += 1
                    yield start, i
                    break
            i += 1
        else:
            return
        if stack:
            # A damaged outer object is not a license to reuse its nested fragments.
            return
