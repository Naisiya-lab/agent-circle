"""Standalone program tests: no agents, API keys or network calls."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from table_tools import TableTools, ProgramError, definitions_for
from table_tools.contracts import SPECS
from table_tools.html_grid import Table
from table_tools_demo import EXAMPLE_HTML, run_demo

BASE = Path(__file__).resolve().parents[1]


def source(html=EXAMPLE_HTML):
    return {"tables": [{"id": "t1", "html": html}]}


class ToolCase(unittest.TestCase):
    def setUp(self):
        self.service = TableTools("sample", source())
        self.v = self.service.table_version

    def call(self, tool, args=None, role="select", service=None):
        instance = service or self.service
        return instance.call(role, tool, {"table_version": instance.table_version, **(args or {})})

    def extracted(self, rows=(3, 3), columns=(1, 2), service=None):
        result = self.call("extract_subtables", {"regions": [{"id": "s1", "source_table_id": "t1", "rows": list(rows), "columns": list(columns)}]}, service=service)
        self.assertTrue(result["ok"], result)
        return result["data"]

    def mapped(self, extraction, positions, direction="to_source", **extra):
        return self.call("map_coordinates", {"extraction_id": extraction["extraction_id"], "subtable_id": "s1",
                         "direction": direction, "positions": positions, **extra}, role="solve")

    def test_index_logical_rows_and_columns_not_tag_counts(self):
        catalog = self.call("list_tables")["data"]["tables"][0]
        self.assertEqual((catalog["row_count"], catalog["column_count"]), (6, 3))
        expected = [["Item", "Values", "Values"], ["A", "4005000000", ""],
                    ["A", "95", "22.1%"], ["", "", ""], ["", "", ""], ["B", "4005000000", ""]]
        actual = self.call("inspect_cells", {"source_table_id": "t1", "positions": [[r, c] for r in range(1, 7) for c in range(1, 4)]})["data"]["cells"]
        self.assertEqual([x["content_html"] for x in actual], [v for row in expected for v in row])
        self.assertEqual(actual[6]["anchor"], [2, 1])

    def test_repeats_and_independent_blanks_remain_separate(self):
        cells = self.call("inspect_cells", {"source_table_id": "t1", "positions": [[2, 2], [6, 2], [4, 1], [5, 1]]})["data"]["cells"]
        self.assertEqual(cells[0]["content_html"], cells[1]["content_html"])
        self.assertNotEqual(cells[0]["cell_id"], cells[1]["cell_id"])
        self.assertTrue(cells[2]["empty_html"])
        self.assertNotEqual(cells[2]["cell_id"], cells[3]["cell_id"])

    def test_extraction_closes_rowspan(self):
        sub = self.extracted()["subtables"][0]
        self.assertEqual(sub["requested_rows"], [3, 3])
        self.assertEqual(sub["effective_rows"], [2, 3])
        self.assertEqual(sub["source_rows"], [2, 3])
        self.assertTrue(sub["expanded_for_merges"])
        self.assertIn('rowspan="2"', sub["html"])

    def test_extraction_closes_colspan(self):
        sub = self.extracted((1, 1), (2, 2))["subtables"][0]
        self.assertEqual(sub["source_columns"], [2, 3])
        self.assertIn('colspan="2"', sub["html"])

    def test_multiple_regions_do_not_compress_gaps(self):
        result = self.call("extract_subtables", {"regions": [
            {"id": "a", "source_table_id": "t1", "rows": [2, 2], "columns": [2, 2]},
            {"id": "b", "source_table_id": "t1", "rows": [6, 6], "columns": [2, 2]}]})
        self.assertTrue(result["ok"])
        self.assertEqual([x["source_rows"] for x in result["data"]["subtables"]], [[2], [6]])

    def test_nonzero_offset_mapping_in_both_directions(self):
        extraction = self.extracted((6, 6), (2, 3))
        a = self.mapped(extraction, [[1, 1]])
        b = self.mapped(extraction, [[6, 2]], "to_subtable")
        self.assertEqual(a["data"]["positions"][0]["mapped"]["position"], [6, 2])
        self.assertEqual(b["data"]["positions"][0]["mapped"]["position"], [1, 1])

    def test_merged_covered_position_is_not_anchor(self):
        result = self.mapped(self.extracted(), [[2, 1]])
        item = result["data"]["positions"][0]
        self.assertEqual(item["mapped"]["position"], [3, 1])
        self.assertEqual(item["mapped"]["cell_anchor"], [2, 1])
        self.assertEqual(item["source_cell"]["rows"], [2, 3])

    def test_explicit_zero_based_output_does_not_modify_mapping(self):
        extraction = self.extracted()
        result = self.mapped(extraction, [[2, 2]], output_base=0)
        self.assertEqual(result["data"]["positions"][0]["mapped"]["position"], [2, 1])
        self.assertEqual(result["data"]["positions"][0]["source_cell"]["base"], 1)
        again = self.mapped(extraction, [[2, 2]])
        self.assertEqual(again["data"]["positions"][0]["mapped"]["position"], [3, 2])

    def test_mapping_outside_subtable_is_a_failure_not_partial_success(self):
        result = self.mapped(self.extracted(), [[1, 1], [99, 1]])
        self.assertFalse(result["ok"])
        self.assertIsNone(result["data"])
        self.assertEqual(result["error"]["code"], "POSITION_OUTSIDE_SUBTABLE")

    def test_source_position_not_selected_is_rejected(self):
        result = self.mapped(self.extracted(), [[6, 1]], "to_subtable")
        self.assertFalse(result["ok"])

    def test_stale_version_is_rejected_for_every_operation(self):
        extraction = self.extracted()
        cases = [("list_tables", {}, "select"),
                 ("extract_subtables", {"regions": [{"id": "s", "source_table_id": "t1", "rows": [1, 1], "columns": [1, 1]}]}, "select"),
                 ("inspect_cells", {"source_table_id": "t1", "positions": [[1, 1]]}, "check"),
                 ("map_coordinates", {"extraction_id": extraction["extraction_id"], "subtable_id": "s1", "direction": "to_source", "positions": [[1, 1]]}, "solve")]
        for name, args, role in cases:
            with self.subTest(name=name):
                result = self.call(name, {"table_version": "old", **args}, role)
                self.assertEqual(result["error"]["code"], "STALE_TABLE_VERSION")

    def test_other_sample_extraction_is_rejected(self):
        extraction = self.extracted()
        other = TableTools("other", source())
        result = other.call("solve", "map_coordinates", {"table_version": other.table_version, "extraction_id": extraction["extraction_id"],
                            "subtable_id": "s1", "direction": "to_source", "positions": [[1, 1]]})
        self.assertEqual(result["error"]["code"], "UNKNOWN_EXTRACTION")

    def test_content_and_attribute_changes_invalidate_versions(self):
        for html in (EXAMPLE_HTML.replace('>95<', '>93<'), EXAMPLE_HTML.replace('yellow', 'blue')):
            with self.subTest(html=html):
                updated = TableTools("sample", source(html))
                self.assertNotEqual(updated.table_version, self.v)

    def test_returned_objects_cannot_mutate_program_state(self):
        first = self.extracted()
        first["subtables"][0]["source_rows"][0] = 999
        second = self.extracted()
        self.assertEqual(second["subtables"][0]["source_rows"], [2, 3])
        result = self.mapped(second, [[1, 1]])
        self.assertEqual(result["data"]["positions"][0]["mapped"]["position"], [2, 1])

    def test_idempotent_extraction_id(self):
        self.assertEqual(self.extracted()["extraction_id"], self.extracted()["extraction_id"])

    def test_region_errors_are_specific(self):
        for row_range, table, rid in (([7, 7], "t1", "s"), ([3, 2], "t1", "s"), ([1, 1], "bad", "s")):
            result = self.call("extract_subtables", {"regions": [{"id": rid, "source_table_id": table, "rows": row_range, "columns": [1, 1]}]})
            self.assertEqual(result["error"]["code"], "INVALID_REGION")

    def test_duplicate_region_ids_are_rejected(self):
        region = {"id": "same", "source_table_id": "t1", "rows": [1, 1], "columns": [1, 1]}
        result = self.call("extract_subtables", {"regions": [region, region]})
        self.assertEqual(result["error"]["code"], "INVALID_REGION")

    def test_empty_unknown_and_boolean_arguments_are_rejected(self):
        for regions in ([], [{"id": "s", "source_table_id": "t1", "rows": [True, 2], "columns": [1, 1]}]):
            result = self.call("extract_subtables", {"regions": regions})
            self.assertEqual(result["error"]["code"], "INVALID_ARGUMENTS")
        result = self.call("list_tables", {"filename": "never_read.txt"})
        self.assertEqual(result["error"]["code"], "INVALID_ARGUMENTS")

    def test_strict_string_arguments_reject_duplicate_keys_and_nan(self):
        for value in ('{"table_version":"x","table_version":"y"}', '{"table_version":NaN}', '{} {}'):
            result = self.service.call("select", "list_tables", value)
            self.assertEqual(result["error"]["code"], "INVALID_ARGUMENTS")

    def test_role_boundaries_do_not_allow_solver_to_reselect(self):
        self.assertEqual(definitions_for("understand"), [])
        self.assertEqual([d["function"]["name"] for d in definitions_for("solve")], ["map_coordinates"])
        result = self.call("extract_subtables", {"regions": []}, role="solve")
        self.assertEqual(result["error"]["code"], "ROLE_NOT_ALLOWED")
        self.assertEqual(self.call("list_tables", role="solve")["error"]["code"], "ROLE_NOT_ALLOWED")

    def test_no_arbitrary_function_or_path_execution(self):
        result = self.call("__import__", {"code": "anything"})
        self.assertEqual(result["error"]["code"], "UNKNOWN_TOOL")
        result = self.mapped({"extraction_id": "../../outside"}, [[1, 1]])
        self.assertEqual(result["error"]["code"], "UNKNOWN_EXTRACTION")

    def test_schema_queries_are_copies(self):
        definitions = definitions_for("select")
        definitions[0]["function"]["parameters"]["properties"].clear()
        self.assertIn("table_version", SPECS["list_tables"]["parameters"]["properties"])

    def test_all_rectangles_preserve_cells_spans_and_roundtrip_positions(self):
        original = Table("t1", EXAMPLE_HTML)
        for top in range(1, 7):
            for bottom in range(top, 7):
                for left in range(1, 4):
                    for right in range(left, 4):
                        extraction = self.extracted((top, bottom), (left, right))
                        sub = extraction["subtables"][0]
                        parsed = Table("local", sub["html"])
                        self.assertEqual(parsed.rows, len(sub["source_rows"]))
                        self.assertEqual(parsed.columns, len(sub["source_columns"]))
                        local = [[r, c] for r in range(1, parsed.rows+1) for c in range(1, parsed.columns+1)]
                        forward = self.mapped(extraction, local)
                        self.assertTrue(forward["ok"])
                        positions = [p["mapped"]["position"] for p in forward["data"]["positions"]]
                        backward = self.mapped(extraction, positions, "to_subtable")
                        self.assertEqual([p["mapped"]["position"] for p in backward["data"]["positions"]], local)
                        for (r, c), (sr, sc) in zip(local, positions):
                            a, b = parsed.grid[r, c], original.grid[sr, sc]
                            self.assertEqual((a.content, a.rowspan, a.colspan), (b.content, b.rowspan, b.colspan))


class GridTests(unittest.TestCase):
    def test_highlight_and_formatting_survive_extraction(self):
        service = TableTools("x", source())
        args = {"table_version": service.table_version, "regions": [{"id": "s", "source_table_id": "t1", "rows": [3, 3], "columns": [2, 2]}]}
        sub = service.call("select", "extract_subtables", args)["data"]["subtables"][0]
        self.assertIn('background-color:yellow', sub["html"])
        self.assertIn('data-highlight="true"', sub["html"])

    def test_table_group_row_and_inline_attributes_are_preserved_safely(self):
        html = '<table class="a"><tbody style="background:yellow"><tr data-kind="record"><td title="value"><span style="font-weight:bold" onclick="bad()">8</span></td></tr></tbody></table>'
        result = Table("t", html).html
        for fragment in ('class="a"', 'background:yellow', 'data-kind="record"', 'title="value"', 'font-weight:bold'):
            self.assertIn(fragment, result)
        self.assertNotIn('onclick', result)

    def test_no_external_url_or_executable_style_is_rendered(self):
        html = '<table><tr><td style="background:url(https://example.invalid);color:red" onclick="bad()">A</td></tr></table>'
        result = Table("t", html).html
        self.assertNotIn("https", result)
        self.assertNotIn("onclick", result)
        self.assertIn('color:red', result)

    def test_explicit_blank_rows_and_whitespace_are_not_invented_or_trimmed(self):
        table = Table("t", '<table><tr><td> </td></tr><tr><td></td></tr></table>')
        self.assertEqual([c.content for c in table.cells], [' ', ''])

    def test_fully_merged_covered_row_is_not_deleted(self):
        table = Table("t", '<table><tr><td rowspan="2">A</td></tr><tr></tr></table>')
        self.assertEqual((table.rows, table.columns, len(table.cells)), (2, 1, 1))

    def test_transitive_merge_closure(self):
        html = '<table><tr><td rowspan="2">A</td><td>B</td><td>C</td></tr><tr><td colspan="2">D</td></tr></table>'
        table = Table("t", html)
        self.assertEqual(table.closure([2, 2], [1, 2]), (1, 2, 1, 3))

    def test_invalid_grids_are_reported_without_guessed_cells(self):
        cases = ['<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>',
                 '<table><tr><td rowspan="9">A</td></tr></table>',
                 '<table><tr><td>A</td><td rowspan="2">B</td></tr><tr><td colspan="2">C</td></tr></table>',
                 '<table><tr><td colspan="2" colspan="3">A</td></tr></table>']
        for html in cases:
            with self.subTest(html=html), self.assertRaises(ProgramError) as cm:
                TableTools("x", source(html))
            self.assertEqual(cm.exception.code, "HTML_GRID_INVALID")

    def test_conflicting_groups_fail_but_omitted_closing_group_is_readable(self):
        with self.assertRaises(ProgramError):
            TableTools('x',source('<table><tbody><tr><td>x</td></tr></thead></table>'))
        service=TableTools('x',source('<table><tbody><tr><td>x</td></tr></table>'))
        self.assertEqual(service._catalog['tables'][0]['row_count'],1)

    def test_rowspan_cannot_cross_explicit_groups(self):
        html = '<table><thead><tr><td rowspan="2">x</td></tr></thead><tbody><tr></tr></tbody></table>'
        with self.assertRaises(ProgramError):
            TableTools("x", source(html))

    def test_entities_caption_and_inline_content_survive(self):
        table = Table("t", '<table><caption>A &amp; B</caption><tr><td>x<sub>2</sub>&lt;3<br/>z</td></tr></table>')
        self.assertEqual(table.cells[0].content, 'x<sub>2</sub>&lt;3<br>z')
        self.assertIn('<caption>A &amp; B</caption>', table.html)

    def test_inputs_are_not_mutated_and_minimal_tables_are_accepted(self):
        value = source()
        before = deepcopy(value)
        TableTools("x", value)
        self.assertEqual(value, before)


class StoredAndCLITests(unittest.TestCase):
    def setUp(self):
        parent = (BASE/"offline_checks").resolve()
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="table_programs_", dir=parent)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(parent)  # Verified cleanup target stays in the workspace.
        self.addCleanup(self.temp.cleanup)

    def extract(self, service):
        result = service.call("select", "extract_subtables", {"table_version": service.table_version, "regions": [
            {"id": "s", "source_table_id": "t1", "rows": [3, 3], "columns": [1, 2]}]})
        self.assertTrue(result["ok"], result)
        return result["data"]["extraction_id"]

    def test_persisted_extraction_restores_after_restart(self):
        first = TableTools("x", source(), state_dir=self.root)
        key = self.extract(first)
        second = TableTools("x", source(), state_dir=self.root)
        result = second.call("solve", "map_coordinates", {"table_version": second.table_version, "extraction_id": key,
                             "subtable_id": "s", "direction": "to_source", "positions": [[2, 2]]})
        self.assertEqual(result["data"]["positions"][0]["mapped"]["position"], [3, 2])
        self.assertTrue((self.root/first.context_id/"source_tables.json").is_file())
        self.assertTrue((self.root/first.context_id/"calls.jsonl").is_file())

    def test_saved_html_and_mapping_are_recomputed_not_trusted(self):
        first = TableTools("x", source(), state_dir=self.root)
        key = self.extract(first)
        path = self.root/first.context_id/(key+".json")
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["subtables"][0]["source_rows"] = [999, 1000]
        stored["subtables"][0]["html"] = "bad"
        path.write_text(json.dumps(stored), encoding="utf-8")
        second = TableTools("x", source(), state_dir=self.root)
        result = second.call("solve", "map_coordinates", {"table_version": second.table_version, "extraction_id": key,
                             "subtable_id": "s", "direction": "to_source", "positions": [[2, 2]]})
        self.assertEqual(result["data"]["positions"][0]["mapped"]["position"], [3, 2])

    def test_wrong_saved_context_is_not_reused(self):
        service = TableTools("x", source(), state_dir=self.root)
        key = self.extract(service)
        path = self.root/service.context_id/(key+".json")
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["context_id"] = "foreign"
        path.write_text(json.dumps(stored), encoding="utf-8")
        restored = TableTools("x", source(), state_dir=self.root)
        response = restored.call("solve", "map_coordinates", {"table_version": restored.table_version, "extraction_id": key,
                                 "subtable_id": "s", "direction": "to_source", "positions": [[1, 1]]})
        self.assertEqual(response["error"]["code"], "INVALID_STORED_EXTRACTION")

    def test_storage_failure_is_not_reported_as_bad_table(self):
        with patch("table_tools.service.write_json", side_effect=OSError("offline simulated storage problem")):
            with self.assertRaises(ProgramError) as cm:
                TableTools("x", source(), state_dir=self.root)
        self.assertEqual(cm.exception.code, "STATE_IO_ERROR")

    def test_long_state_paths_work_with_nested_run_directories(self):
        state = self.root/("nested_run_"+"a"*105)
        service = None
        try:
            service = TableTools("long-path-sample", source(), state_dir=state)
            key = self.extract(service)
            restored = TableTools("long-path-sample", source(), state_dir=state)
            response = restored.call("solve", "map_coordinates", {"table_version": restored.table_version,
                "extraction_id": key, "subtable_id": "s", "direction": "to_source", "positions": [[2, 2]]})
            self.assertTrue(response["ok"], response)
            self.assertEqual(response["data"]["positions"][0]["mapped"]["position"], [3, 2])
        finally:
            if service is not None:
                for file in service._folder.iterdir():
                    file.unlink()
                service._folder.rmdir()
                state.rmdir()

    def test_cli_separate_processes_share_only_their_bound_context(self):
        output = self.root/"example"
        demo = run_demo(output)
        cmd = [sys.executable, str(BASE/"table_tools_cli.py"), "call", "--role", "solve", "--context", str(output/"context.json"),
               "--request", str(output/"map_request.json"), "--state-dir", str(output/"state")]
        process = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, encoding="utf-8", check=True)
        response = json.loads(process.stdout)
        self.assertEqual(response["data"]["positions"][0]["mapped"]["position"], [3, 2])
        self.assertEqual(demo["api_calls"], 0)

    def test_cli_malformed_request_has_machine_readable_failure(self):
        output = self.root/"example"
        run_demo(output)
        cmd = [sys.executable, str(BASE/"table_tools_cli.py"), "call", "--role", "select", "--context", str(output/"context.json")]
        process = subprocess.run(cmd, input='{"tool":"list_tables","arguments":NaN}', capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(process.returncode, 1)
        self.assertFalse(json.loads(process.stdout)["ok"])

    def test_tools_import_without_importing_the_workflow(self):
        script = "import sys; from table_tools import TableTools; assert not any(n.startswith('region_loop_runtime') for n in sys.modules)"
        subprocess.run([sys.executable, "-c", script], cwd=BASE, check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
