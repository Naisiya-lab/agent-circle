"""Deterministic operations on recognized HTML, never image or answer grading."""
from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from html.parser import HTMLParser
import hashlib
import json
import re


def safe_attributes(attrs):
    """Preserve declarative cell/row metadata without scripts or remote resources."""
    result = {}
    for name, value in attrs:
        if value is None or name in ("data-source-row", "data-source-col", "data-cell-id", "data-local-row", "data-local-col"):
            continue
        if name in ("id", "class", "title", "scope", "headers", "abbr", "align", "valign", "bgcolor", "role", "aria-label") or name.startswith("data-"):
            result[name] = value
        elif name == "style":
            declarations = []
            for declaration in value.split(";"):
                key, sep, item = declaration.partition(":")
                key, item = key.strip().lower(), item.strip()
                if sep and key in ("color", "background", "background-color", "font-weight", "font-style", "text-decoration", "text-align", "vertical-align", "white-space"):
                    if re.fullmatch(r"[a-zA-Z0-9#.,()%\s-]+", item) and not re.search(r"url|expression|import|javascript|binding|behavior", item, re.I):
                        declarations.append(key+":"+item)
            if declarations:
                result[name] = ";".join(declarations)
    return result


def render_attributes(attrs):
    return "".join(' '+key+'="'+escape(value, quote=True)+'"' for key, value in sorted(attrs.items()))


class TableToolError(ValueError):
    def __init__(self, message, return_to="understand"):
        self.return_to = return_to
        super().__init__(message)


@dataclass
class Cell:
    row: int
    column: int
    rowspan: int
    colspan: int
    tag: str
    content: str
    attributes: dict = field(default_factory=dict)

    @property
    def bottom(self):
        return self.row + self.rowspan - 1

    @property
    def right(self):
        return self.column + self.colspan - 1


class Reader(HTMLParser):
    """Require explicit rows/cells. Never invent missing trailing blank cells."""
    INLINE = {"b", "strong", "i", "em", "u", "s", "sub", "sup", "span",
              "small", "p", "div", "br", "a", "font", "code"}
    GROUPS = {"thead", "tbody", "tfoot"}

    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.rows, self.current_row, self.current_cell = [], None, None
        self.in_table = False
        self.table_count = 0
        self.caption = None
        self.in_caption = False
        self.caption_parts = []
        self.stack = []
        self.table_attributes = {}
        self.row_attributes, self.row_groups, self.row_group_ids = [], [], []
        self.current_group, self.group_serial, self.current_row_attributes = None, 0, {}
        self.feed(source)
        self.close()
        if self.stack or self.in_table or self.in_caption or self.current_group is not None or self.current_row is not None or self.current_cell is not None:
            raise TableToolError("Unclosed HTML table/row/cell or formatting tag")
        if self.table_count != 1 or not self.rows:
            raise TableToolError("Each table HTML must contain exactly one nonempty table")

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.in_table or self.table_count:
                raise TableToolError("Nested or multiple tables in one HTML entry")
            self.table_count += 1
            self.in_table = True
            self.table_attributes = safe_attributes(attrs)
        elif tag in self.GROUPS:
            if not self.in_table or self.current_row is not None or self.current_group is not None or self.in_caption:
                raise TableToolError("Row group outside table or inside row")
            self.group_serial += 1
            self.current_group = (self.group_serial, tag, safe_attributes(attrs))
        elif tag == "caption":
            if not self.in_table or self.current_row is not None or self.caption is not None:
                raise TableToolError("Invalid or repeated table caption")
            self.caption = ""
            self.in_caption = True
        elif tag == "tr":
            if not self.in_table or self.current_row is not None or self.in_caption:
                raise TableToolError("Nested row or row outside table")
            self.current_row = []
            self.current_row_attributes = safe_attributes(attrs)
        elif tag in ("td", "th"):
            if self.current_row is None or self.current_cell is not None:
                raise TableToolError("Cell outside row or nested cell")
            if len({key for key, _ in attrs}) != len(attrs):
                raise TableToolError("Duplicate cell attributes")
            original_attributes = safe_attributes(attrs)
            attrs = dict(attrs)
            spans = []
            for key in ("rowspan", "colspan"):
                try:
                    value = int(attrs.get(key, "1"))
                except (ValueError, TypeError) as exc:
                    raise TableToolError("Invalid " + key) from exc
                if not 1 <= value <= 2000:
                    raise TableToolError(key + " must be a positive integer <= 2000; expand rowspan=0 explicitly")
                spans.append(value)
            self.current_cell = [tag, *spans, [], original_attributes]
        elif tag in self.INLINE:
            if self.current_cell is None and not self.in_caption:
                raise TableToolError("Formatting outside cell/caption: " + tag)
            # Preserve safe presentation, including explicit highlights and text format.
            parts = self.current_cell[3] if self.current_cell is not None else self.caption_parts
            parts.append("<" + tag + render_attributes(safe_attributes(attrs)) + ">")
            if tag != "br":
                self.stack.append(tag)
        else:
            raise TableToolError("Unsupported HTML element: " + tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag != "br":
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "table":
            if not self.in_table or self.current_row is not None or self.in_caption or self.current_group is not None:
                raise TableToolError("Invalid table closing tag")
            self.in_table = False
        elif tag in self.GROUPS:
            if self.current_row is not None or self.current_group is None or self.current_group[1] != tag:
                raise TableToolError("Row group ends inside row")
            self.current_group = None
            self.group_serial += 1
        elif tag == "caption":
            if self.stack or not self.in_caption:
                raise TableToolError("Unclosed formatting in caption")
            self.caption = "".join(self.caption_parts)
            self.in_caption = False
        elif tag == "tr":
            if self.current_row is None or self.current_cell is not None:
                raise TableToolError("Invalid row closing tag")
            self.rows.append(self.current_row)
            self.row_attributes.append(self.current_row_attributes)
            self.row_groups.append(self.current_group)
            self.row_group_ids.append(self.group_serial)
            self.current_row = None
        elif tag in ("td", "th"):
            if self.current_cell is None or self.current_cell[0] != tag or self.stack:
                raise TableToolError("Invalid cell closing tag or unclosed formatting")
            kind, rs, cs, parts, attrs = self.current_cell
            self.current_row.append((kind, rs, cs, "".join(parts), attrs))
            self.current_cell = None
        elif tag in self.INLINE and tag != "br":
            if not self.stack or self.stack.pop() != tag:
                raise TableToolError("Mismatched inline HTML tags")
            parts = self.current_cell[3] if self.current_cell is not None else self.caption_parts
            parts.append("</" + tag + ">")
        elif tag != "br":
            raise TableToolError("Unsupported closing HTML tag: " + tag)

    def handle_data(self, data):
        if self.current_cell is not None:
            self.current_cell[3].append(escape(data))
        elif self.in_caption:
            self.caption_parts.append(escape(data))
        elif data.strip():
            raise TableToolError("Text outside a table cell/caption")


class Table:
    def __init__(self, table_id, source):
        self.id = table_id
        reader = Reader(source)
        self.caption = reader.caption or ""
        self.table_attributes = reader.table_attributes
        self.row_attributes, self.row_groups = reader.row_attributes, reader.row_groups
        self.rows = len(reader.rows)
        if self.rows > 2000:
            raise TableToolError("More than 2000 logical rows")
        self.grid, self.cells = {}, []
        for r, row in enumerate(reader.rows, 1):
            c = 1
            for kind, rs, cs, content, attrs in row:
                while (r, c) in self.grid:
                    c += 1
                cell = Cell(r, c, rs, cs, kind, content, attrs)
                if cell.bottom > self.rows or cell.right > 500:
                    raise TableToolError(f"{table_id}: span outside declared rows or >500 columns at ({r},{c})")
                if any(group != reader.row_group_ids[r-1] for group in reader.row_group_ids[r-1:cell.bottom]):
                    raise TableToolError(f"{table_id}: rowspan crosses an explicit row-group boundary at ({r},{c})")
                for rr in range(r, cell.bottom + 1):
                    for cc in range(c, cell.right + 1):
                        if (rr, cc) in self.grid:
                            raise TableToolError(f"{table_id}: overlapping spans at ({rr},{cc})")
                        self.grid[rr, cc] = cell
                self.cells.append(cell)
                c = cell.right + 1
        self.columns = max((c for r, c in self.grid), default=0)
        if not self.columns:
            raise TableToolError(table_id + ": no logical columns")
        for r in range(1, self.rows + 1):
            for c in range(1, self.columns + 1):
                if (r, c) not in self.grid:
                    raise TableToolError(f"{table_id}: missing explicit cell at ({r},{c}); no blank was invented")
        self.html = self.render((1, self.rows, 1, self.columns))

    def cell_id(self, cell):
        return f"{self.id}:r{cell.row}c{cell.column}"

    def render(self, bounds):
        top, bottom, left, right = bounds
        parts = ["<table"+render_attributes(self.table_attributes)+">"]
        if self.caption:
            parts.append("<caption>" + self.caption + "</caption>")
        active_group = None
        for r in range(top, bottom + 1):
            group = self.row_groups[r-1]
            if group != active_group:
                if active_group is not None:
                    parts.append("</"+active_group[1]+">")
                if group is not None:
                    parts.append("<"+group[1]+render_attributes(group[2])+">")
                active_group = group
            parts.append(f'<tr data-source-row="{r}" data-local-row="{r-top+1}"'+render_attributes(self.row_attributes[r-1])+'>')
            for cell in self.cells:
                if cell.row != r or not left <= cell.column <= right:
                    continue
                attrs = (f' data-cell-id="{escape(self.cell_id(cell), quote=True)}"'
                         f' data-source-row="{cell.row}" data-source-col="{cell.column}"'
                         f' data-local-row="{cell.row-top+1}" data-local-col="{cell.column-left+1}"')
                attrs += render_attributes(cell.attributes)
                if cell.rowspan != 1:
                    attrs += f' rowspan="{cell.rowspan}"'
                if cell.colspan != 1:
                    attrs += f' colspan="{cell.colspan}"'
                parts.append(f"<{cell.tag}{attrs}>{cell.content}</{cell.tag}>")
            parts.append("</tr>")
        if active_group is not None:
            parts.append("</"+active_group[1]+">")
        return "".join(parts) + "</table>"

    def closure(self, rows, columns):
        top, bottom = rows
        left, right = columns
        if not (1 <= top <= bottom <= self.rows and 1 <= left <= right <= self.columns):
            raise TableToolError(f"{self.id}: region outside 1-based {self.rows}x{self.columns} grid", "select")
        while True:
            bounds = top, bottom, left, right
            for cell in self.cells:
                if cell.row <= bottom and cell.bottom >= top and cell.column <= right and cell.right >= left:
                    top, bottom = min(top, cell.row), max(bottom, cell.bottom)
                    left, right = min(left, cell.column), max(right, cell.right)
            if bounds == (top, bottom, left, right):
                return bounds


def build_index(understanding):
    tables = {}
    for item in understanding["tables"]:
        tid = item["id"]
        if not tid or tid in tables:
            raise TableToolError("Empty or duplicate source table ID")
        tables[tid] = Table(tid, item["html"])
    if not tables:
        raise TableToolError("No recognized tables")
    payload = [{"id": t.id, "row_count": t.rows, "column_count": t.columns, "html": t.html}
               for t in tables.values()]
    version = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return tables, {"table_version": version, "tables": payload,
                    "coverage": understanding["coverage"], "uncertainties": understanding["uncertainties"]}


def extract(index, selection, version):
    result, ids = [], set()
    for region in selection["regions"]:
        rid, tid = region["id"], region["source_table_id"]
        if not rid or rid in ids or tid not in index:
            raise TableToolError("Empty/duplicate region ID or unknown source table: " + tid, "select")
        ids.add(rid)
        table = index[tid]
        top, bottom, left, right = table.closure(region["rows"], region["columns"])
        cells = []
        for cell in table.cells:
            if top <= cell.row <= bottom and left <= cell.column <= right:
                cells.append({"cell_id": table.cell_id(cell),
                              "local_rows": [cell.row-top+1, cell.bottom-top+1],
                              "local_columns": [cell.column-left+1, cell.right-left+1],
                              "source_rows": [cell.row, cell.bottom],
                              "source_columns": [cell.column, cell.right]})
        result.append({"id": rid, "source_table_id": tid, "table_version": version,
                       "requested_rows": region["rows"], "requested_columns": region["columns"],
                       "source_rows": list(range(top, bottom+1)),
                       "source_columns": list(range(left, right+1)),
                       "expanded_for_merges": [top, bottom] != region["rows"] or [left, right] != region["columns"],
                       "html": table.render((top, bottom, left, right)), "source_cells": cells})
    if not result:
        raise TableToolError("Choose at least one region; use the complete table for absence/global judgments", "select")
    return result
