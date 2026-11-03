"""JSON message reading: collapse equal duplicates, never choose conflicting values."""

import json

class ProtocolError(ValueError):
    def __init__(self, errors, role=None, stage=None):
        self.errors = list(errors)
        self.role, self.stage = role, stage
        super().__init__("; ".join(self.errors))

from table_tools.strict_json import DECODER


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
