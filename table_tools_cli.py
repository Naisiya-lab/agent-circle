"""Offline entry point for the agents' local table programs; no API keys/network."""
import argparse
import json
from pathlib import Path
import sys

from table_tools import TableTools, ProgramError, definitions_for
from table_tools.contracts import SPECS, VERSION
from table_tools.strict_json import DECODER

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("describe", "call"))
    parser.add_argument("--role", required=True, choices=("understand", "select", "solve", "check"))
    parser.add_argument("--context", type=Path, help="Host-controlled JSON with sample_id and understanding")
    parser.add_argument("--request", type=Path, help="JSON with tool and arguments; stdin when omitted")
    parser.add_argument("--state-dir", type=Path, default=BASE/"tool_workspace")
    args = parser.parse_args()
    if args.command == "describe":
        result = {"tool_version": VERSION, "tools": definitions_for(args.role),
                  "descriptions_zh": {name: spec["description_zh"] for name, spec in SPECS.items() if args.role in spec["roles"]}}
    else:
        try:
            if args.context is None:
                raise ProgramError("INVALID_CONTEXT", "--context is required for call")
            context = DECODER.decode(args.context.read_text(encoding="utf-8-sig"))
            service = TableTools(context["sample_id"], context["understanding"], state_dir=args.state_dir)
            request = DECODER.decode(args.request.read_text(encoding="utf-8-sig") if args.request else sys.stdin.read())
            if type(request) is not dict or set(request) != {"tool", "arguments"} or type(request["tool"]) is not str:
                raise ProgramError("INVALID_REQUEST", "Request must contain exactly tool and arguments")
            result = service.call(args.role, request["tool"], request["arguments"])
        except ProgramError as exc:
            result = {"ok": False, "tool_version": VERSION, "data": None, "error": exc.record()}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result = {"ok": False, "tool_version": VERSION, "data": None,
                      "error": {"code": "INVALID_INPUT", "message": str(exc)}}
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
