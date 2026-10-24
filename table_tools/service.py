"""Versioned table programs with immutable provenance and role-bound dispatch."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

from .html_grid import build_index, extract, TableToolError
from .strict_json import DECODER
from .contracts import VERSION, SPECS, definitions_for, validate


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent/(".tmp_"+uuid.uuid4().hex+".json")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class ProgramError(ValueError):
    def __init__(self, code, message, *, details=None, recovery="correct_arguments"):
        self.code, self.details, self.recovery = code, details or {}, recovery
        super().__init__(message)

    def record(self):
        return {"code": self.code, "message": str(self), "details": self.details, "recovery": self.recovery}


class TableTools:
    """One instance is bound by the host to one sample and one table version.

    Agents supply neither input HTML nor file paths in tool calls. The host
    registers the understanding output once. A new output needs a new instance.
    """
    def __init__(self, sample_id, understanding, *, state_dir=None):
        if type(sample_id) is not str or not sample_id:
            raise ProgramError("INVALID_CONTEXT", "The host must supply a sample ID")
        if type(understanding) is not dict or type(understanding.get("tables")) is not list:
            raise ProgramError("INVALID_UNDERSTANDING", "Expected tables: [{id, html}]", recovery="revise_understanding")
        for item in understanding["tables"]:
            if type(item) is not dict or type(item.get("id")) is not str or type(item.get("html")) is not str:
                raise ProgramError("INVALID_UNDERSTANDING", "Each table needs string id and html", recovery="revise_understanding")
        coverage, uncertainties = understanding.get("coverage", "unspecified"), understanding.get("uncertainties", [])
        if coverage not in ("complete", "partial", "unspecified") or type(uncertainties) is not list:
            raise ProgramError("INVALID_UNDERSTANDING", "Invalid optional coverage/uncertainties fields", recovery="revise_understanding")
        normalized = {"tables": deepcopy(understanding["tables"]), "coverage": coverage, "uncertainties": deepcopy(uncertainties)}
        try:
            self._tables, self._catalog = build_index(normalized)
        except TableToolError as exc:
            raise ProgramError("HTML_GRID_INVALID", str(exc), recovery="revise_understanding") from exc
        # Include source markup, not just normalized cells: attribute-only changes
        # also invalidate references. Indexing never overwrites the source HTML.
        self.table_version = fingerprint({"grid": self._catalog["table_version"], "tables": normalized["tables"]})
        self._catalog["table_version"] = self.table_version
        self.context_id = "ctx_"+fingerprint({"sample_id": sample_id, "table_version": self.table_version, "tool_version": VERSION})
        self._extractions = {}
        self._lock = threading.RLock()
        # The directory is controlled by the host, never by model arguments.
        self._folder = Path(state_dir).resolve()/self.context_id if state_dir is not None else None
        # Nested run/sample directories plus full version hashes can exceed
        # Windows' legacy path limit. Use native extended paths internally.
        if self._folder is not None and os.name == "nt":
            location = str(self._folder)
            if not location.startswith("\\\\?\\"):
                location = "\\\\?\\UNC\\"+location[2:] if location.startswith("\\\\") else "\\\\?\\"+location
                self._folder = Path(location)
        if self._folder is not None:
            metadata = {"tool_version": VERSION, "context_id": self.context_id, "sample_id": sample_id,
                        "table_version": self.table_version}
            try:
                self._folder.mkdir(parents=True, exist_ok=True)
                path = self._folder/"context.json"
                if path.exists() and DECODER.decode(path.read_text(encoding="utf-8")) != metadata:
                    raise ProgramError("STATE_CONTEXT_MISMATCH", "Stored context does not match this sample/version", recovery="restore_matching_context")
                write_json(path, metadata)
                write_json(self._folder/"source_tables.json", normalized)
            except OSError as exc:
                raise ProgramError("STATE_IO_ERROR", "Cannot initialize tool storage", recovery="fix_host_storage") from exc
            except ValueError as exc:
                if isinstance(exc, ProgramError):
                    raise
                raise ProgramError("STATE_CONTEXT_MISMATCH", "Stored context JSON is invalid", recovery="restore_matching_context") from exc

    def describe(self, role):
        return {"tool_version": VERSION, "context_id": self.context_id, "table_version": self.table_version,
                "coordinate_base": 1, "tools": definitions_for(role)}

    def call(self, role, name, arguments):
        started = time.monotonic()
        with self._lock:
            try:
                if type(name) is not str or name not in SPECS:
                    raise ProgramError("UNKNOWN_TOOL", "Unknown table program")
                if role not in SPECS[name]["roles"]:
                    raise ProgramError("ROLE_NOT_ALLOWED", "This agent cannot call this program")
                if isinstance(arguments, str):
                    try:
                        arguments = DECODER.decode(arguments)
                    except ValueError as exc:
                        raise ProgramError("INVALID_ARGUMENTS", "Tool arguments must be one strict JSON object") from exc
                errors = validate(arguments, SPECS[name]["parameters"])
                if errors:
                    raise ProgramError("INVALID_ARGUMENTS", "Invalid table-program parameters", details={"errors": errors})
                if arguments["table_version"] != self.table_version:
                    raise ProgramError("STALE_TABLE_VERSION", "Use positions and mappings for the current table version",
                                       details={"expected_version": self.table_version}, recovery="refresh_context")
                data = getattr(self, "_"+name)(arguments)
                response = {"ok": True, "tool": name, "tool_version": VERSION, "context_id": self.context_id,
                            "table_version": self.table_version, "data": data, "error": None}
            except ProgramError as exc:
                response = {"ok": False, "tool": name, "tool_version": VERSION, "context_id": self.context_id,
                            "table_version": self.table_version, "data": None, "error": exc.record()}
            except OSError:
                response = {"ok": False, "tool": name, "tool_version": VERSION, "context_id": self.context_id,
                            "table_version": self.table_version, "data": None,
                            "error": ProgramError("STATE_IO_ERROR", "Cannot persist tool result", recovery="fix_host_storage").record()}
            if self._folder is not None:
                event = {"role": role, "tool": name, "arguments": arguments, "response": response,
                         "seconds": round(time.monotonic()-started, 6)}
                try:
                    with (self._folder/"calls.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps(event, ensure_ascii=False, default=lambda value: "<non-JSON argument>")+"\n")
                except OSError:
                    response.update(ok=False, data=None, error=ProgramError("STATE_IO_ERROR", "Cannot persist tool audit log", recovery="fix_host_storage").record())
            return deepcopy(response)

    def _list_tables(self, args):
        return deepcopy(self._catalog)

    def _extract_subtables(self, args):
        plan = {"regions": deepcopy(args["regions"])}
        try:
            subtables = extract(self._tables, plan, self.table_version)
        except TableToolError as exc:
            raise ProgramError("INVALID_REGION", str(exc), recovery="reselect_region") from exc
        for sub in subtables:
            sub["effective_rows"] = [sub["source_rows"][0], sub["source_rows"][-1]]
            sub["effective_columns"] = [sub["source_columns"][0], sub["source_columns"][-1]]
        key = "x_"+fingerprint({"tool_version": VERSION, "context_id": self.context_id, "selection": plan})
        stored = {"tool_version": VERSION, "context_id": self.context_id, "table_version": self.table_version,
                  "extraction_id": key, "selection": plan, "subtables": subtables}
        self._extractions[key] = stored
        if self._folder is not None:
            write_json(self._folder/(key+".json"), stored)
        # Detailed per-cell lineage stays in program state, not repeated in every model message.
        fields = ("id", "source_table_id", "table_version", "requested_rows", "requested_columns",
                  "effective_rows", "effective_columns", "source_rows", "source_columns", "expanded_for_merges", "html")
        return {"extraction_id": key, "subtables": [{k: s[k] for k in fields} for s in subtables],
                "note": "Positions refer to the recognized source table, not verified image coordinates."}

    def _load_extraction(self, key):
        if not re.fullmatch(r"x_[0-9a-f]{64}", key):
            raise ProgramError("UNKNOWN_EXTRACTION", "Invalid extraction ID")
        if key not in self._extractions and self._folder is not None:
            path = self._folder/(key+".json")
            if path.is_file():
                try:
                    stored = DECODER.decode(path.read_text(encoding="utf-8"))
                    if type(stored) is not dict:
                        raise ValueError("stored extraction is not an object")
                    if stored.get("context_id") != self.context_id or stored.get("table_version") != self.table_version or stored.get("tool_version") != VERSION:
                        raise ValueError("wrong context")
                    args = {"table_version": self.table_version, "regions": stored["selection"]["regions"]}
                    if validate(args, SPECS["extract_subtables"]["parameters"]):
                        raise ValueError("wrong selection")
                    expected = "x_"+fingerprint({"tool_version": VERSION, "context_id": self.context_id, "selection": stored["selection"]})
                    if expected != key:
                        raise ValueError("wrong ID")
                    # Rebuild from registered HTML, never trust edited saved coordinates or HTML.
                    self._extract_subtables(args)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ProgramError("INVALID_STORED_EXTRACTION", "Stored extraction cannot be restored", recovery="extract_again") from exc
        if key not in self._extractions:
            raise ProgramError("UNKNOWN_EXTRACTION", "This extraction does not belong to the current sample/version", recovery="extract_again")
        return self._extractions[key]

    def _inspect_cells(self, args):
        tid = args["source_table_id"]
        if tid not in self._tables:
            raise ProgramError("UNKNOWN_TABLE", "Unknown recognized table ID")
        table, cells = self._tables[tid], []
        for r, c in args["positions"]:
            if (r, c) not in table.grid:
                raise ProgramError("POSITION_OUT_OF_BOUNDS", "Query outside recognized table", details={"position": [r, c]})
            cell = table.grid[r, c]
            cells.append({"queried_position": [r, c], "coordinate_base": 1,
                          "cell_id": table.cell_id(cell), "anchor": [cell.row, cell.column],
                          "rows": [cell.row, cell.bottom], "columns": [cell.column, cell.right],
                          "content_html": cell.content, "attributes": deepcopy(cell.attributes),
                          "empty_html": cell.content == ""})
        return {"source_table_id": tid, "cells": cells}

    def _map_coordinates(self, args):
        stored = self._load_extraction(args["extraction_id"])
        sub = next((s for s in stored["subtables"] if s["id"] == args["subtable_id"]), None)
        if sub is None:
            raise ProgramError("UNKNOWN_SUBTABLE", "Unknown subtable ID in this extraction")
        table = self._tables[sub["source_table_id"]]
        row_map, col_map = sub["source_rows"], sub["source_columns"]
        reverse_rows = {v: i for i, v in enumerate(row_map, 1)}
        reverse_cols = {v: i for i, v in enumerate(col_map, 1)}
        base = args.get("output_base", 1)
        shift = 1-base
        results = []
        for row, col in args["positions"]:
            if args["direction"] == "to_source":
                if not (1 <= row <= len(row_map) and 1 <= col <= len(col_map)):
                    raise ProgramError("POSITION_OUTSIDE_SUBTABLE", "Local position is outside the extracted grid", details={"position": [row, col]})
                sr, sc = row_map[row-1], col_map[col-1]
                cell = table.grid[sr, sc]
                mapped = [sr, sc]
                anchor = [cell.row, cell.column]
                rows, columns = [cell.row, cell.bottom], [cell.column, cell.right]
            else:
                if row not in reverse_rows or col not in reverse_cols:
                    raise ProgramError("POSITION_OUTSIDE_SUBTABLE", "Source position is not included in this subtable", details={"position": [row, col]})
                sr, sc = row, col
                cell = table.grid[sr, sc]
                mapped = [reverse_rows[row], reverse_cols[col]]
                anchor = [reverse_rows[cell.row], reverse_cols[cell.column]]
                rows = [reverse_rows[cell.row], reverse_rows[cell.bottom]]
                columns = [reverse_cols[cell.column], reverse_cols[cell.right]]
            results.append({"input": {"position": [row, col], "base": 1},
                            "mapped": {"position": [v-shift for v in mapped], "base": base,
                                       "cell_anchor": [v-shift for v in anchor],
                                       "cell_rows": [v-shift for v in rows], "cell_columns": [v-shift for v in columns]},
                            "source_cell": {"id": table.cell_id(cell), "rows": [cell.row, cell.bottom],
                                            "columns": [cell.column, cell.right], "base": 1}})
        return {"extraction_id": args["extraction_id"], "subtable_id": sub["id"], "direction": args["direction"],
                "positions": results}
