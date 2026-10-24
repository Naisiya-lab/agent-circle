"""Agent/tool communication and bounded feedback loops, with fake model responses."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from table_tools import ProgramError
from table_tools_demo import EXAMPLE_HTML
from tool_loop_runtime import pipeline as flow, runner
from tool_loop_runtime.protocol import ProtocolError, decode

BASE = Path(__file__).resolve().parents[1]


def understood(html='<table><tr><th>Item</th><th>Value</th></tr><tr><td>A</td><td>8</td></tr></table>'):
    return {"schema_version": "table_html_v3", "tables": [{"id": "t1", "html": html}], "coverage": "complete", "uncertainties": []}


def selected(rows=(1, 2), columns=(1, 2)):
    return {"action": "extract", "regions": [{"id": "s1", "source_table_id": "t1", "rows": list(rows), "columns": list(columns)}], "uncertainties": []}


def solved(answer="8", requests=None):
    value = {"status": "answered", "answer": answer, "review_request": ""}
    if requests is not None:
        value["coordinate_requests"] = requests
    return value


def request(rid="p1", position=(1, 1), component="position", base=1, fmt="tuple"):
    return {"id": rid, "subtable_id": "s1", "position": list(position), "component": component, "output_base": base, "format": fmt}


def checked(target="none", status=None):
    return {"action": "judge", "status": status or ("passed" if target == "none" else "needs_revision"),
            "issues": [] if target == "none" else [{"observation": "The image has value 9", "conflict": "The current output has value 8",
                                                       "source_stage": target, "location": "t1 r2c2"}]}


def inspect(table="t1", positions=((2, 2),)):
    return {"action": "inspect", "queries": [{"source_table_id": table, "positions": [list(p) for p in positions]}]}


def context(body):
    content = body["messages"][1]["content"]
    return json.loads(content[-1]["text"] if type(content) is list else content)


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def call(self, body, sid, stage, *, role):
        self.calls.append((role, stage, deepcopy(body)))
        if not self.replies:
            raise AssertionError("Unexpected model request: "+role)
        value = self.replies.pop(0)
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            value = value(body)
        return {"choices": [{"finish_reason": "stop", "message": {"content": value if type(value) is str else json.dumps(value)}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}}


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        parent = (BASE/"offline_checks").resolve()
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="tool_integration_", dir=parent)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(parent)
        self.addCleanup(self.temp.cleanup)
        image = self.root/"image.jpg"
        image.write_bytes(b"offline fake image")
        self.row = {"id": "test", "question": "Find the requested value or position.", "image": str(image),
                    "image_sha256": flow.transport.digest(image.read_bytes()), "task_type": "TQA"}
        self.cfg = flow.transport.read_json(BASE/"configs/legacy_mixed_A.json")
        self.prompts = runner.prompt_map()
        self.folder = self.root/"samples/test"

    def execute(self, replies):
        client = FakeClient(replies)
        result = flow.run_sample(self.row, self.folder, self.cfg, self.prompts, client, self.root/"run.log")
        return result, client

    def test_normal_path_has_four_model_calls_and_minimal_solver_input(self):
        result, client = self.execute([understood(), selected(), solved(), checked()])
        self.assertEqual([x[0] for x in client.calls], ["understand", "select", "solve", "check"])
        self.assertEqual(result["counts"], {"model_calls": 4, "tool_calls": 3, "rework_rounds": 0})
        self.assertEqual(set(context(client.calls[2][2])), {"task", "subtables"})
        self.assertNotIn("source_cells", context(client.calls[2][2])["subtables"][0])
        self.assertEqual(result["verification_status"], "passed")
        self.assertTrue(result["identity"]["extraction_id"].startswith("x_"))
        self.assertTrue(result["identity"]["candidate_id"].startswith("a_"))
        for role, _, body in client.calls:
            self.assertEqual(type(body["messages"][1]["content"]) is list, role in ("understand", "check"))

    def test_empty_selection_supplies_full_table_without_rework_or_absence_claim(self):
        empty = {"action":"extract", "regions":[], "uncertainties":[]}
        answer = dict(solved('The target cannot be established'), status='insufficient_information')
        result, client = self.execute([understood(), empty, answer, checked()])
        self.assertFalse(result['rework_used'])
        self.assertEqual(result['counts'], {'model_calls':4, 'tool_calls':3, 'rework_rounds':0})
        self.assertEqual(result['status'], 'insufficient_information')
        self.assertEqual(result['answer'], answer['answer'])
        actual = context(client.calls[2][2])['subtables'][0]
        self.assertEqual(actual['source_rows'], [1,2])
        self.assertEqual(actual['source_columns'], [1,2])
        self.assertEqual(flow.transport.read_json(self.folder/'r00_02_select.json')['output'], empty)
        resolution = flow.transport.read_json(self.folder/'r00_selection_resolution.json')
        self.assertEqual(resolution['mode'], 'full_table_after_empty_selection')
        self.assertEqual(context(client.calls[-1][2])['selection_resolution'], resolution)

    def test_empty_selection_supplies_all_tables_and_valid_coordinate_mapping(self):
        u = understood()
        u['tables'].append({'id':'t2','html':'<table><tr><td colspan="2">Marked</td></tr><tr><td>A</td><td>B</td></tr></table>'})
        req = dict(request(position=(2,2)),subtable_id='full_2')
        result, client = self.execute([u, {'action':'extract','regions':[], 'uncertainties':[]}, solved('{{coord:p1}}',[req]), checked()])
        subs = context(client.calls[2][2])['subtables']
        self.assertEqual([s['source_table_id'] for s in subs], ['t1','t2'])
        self.assertIn('colspan="2"', subs[1]['html'])
        self.assertEqual(result['answer'], '(2, 2)')
        self.assertFalse(result['rework_used'])

    def test_empty_selection_during_rework_does_not_add_a_second_round(self):
        result, _ = self.execute([understood(), selected(), solved(), checked('select'),
                                 {'action':'extract','regions':[], 'uncertainties':[]}, solved('[]'), checked()])
        self.assertEqual(result['counts']['rework_rounds'],1)
        self.assertEqual(result['verification_status'],'passed')
        self.assertEqual(result['answer'],'[]')

    def test_background_and_emphasis_survive_all_handoffs_and_merge_expansion(self):
        html = '<table><tr><th style="background-color:#aaffff;font-weight:bold" colspan="2">Heading</th><th>Other</th></tr><tr><td>A</td><td bgcolor="#aaffff"><strong>73.4</strong></td><td></td></tr></table>'
        result, client = self.execute([understood(html), selected((1,2),(2,2)), solved('73.4'), checked()])
        stages = [context(c[2]) for c in client.calls]
        for wire in (stages[1]['table_index']['tables'][0]['html'], stages[2]['subtables'][0]['html'], stages[3]['subtables'][0]['html']):
            self.assertIn('background-color:#aaffff',wire)
            self.assertIn('bgcolor="#aaffff"',wire)
            self.assertIn('<strong>73.4</strong>',wire)
            self.assertIn('colspan="2"',wire)
        self.assertEqual(stages[2]['subtables'][0]['source_columns'],[1,2])
        self.assertEqual(result['verification_status'],'passed')

    def test_checker_issues_choose_earliest_stage_without_model_return_to(self):
        judge = checked('solve')
        judge['issues'].extend(checked('select')['issues']+checked('understand')['issues'])
        result, client = self.execute([understood(), selected(), solved(), judge, understood(), selected(), solved(), checked()])
        self.assertEqual([c[0] for c in client.calls],['understand','select','solve','check','understand','select','solve','check'])
        self.assertEqual(result['rework_target'],'understand')
        saved = flow.transport.read_json(self.folder/'r00_05_check.json')['output']
        self.assertNotIn('return_to',saved)
        route = flow.transport.read_json(self.folder/'r00_check_routing.json')
        self.assertEqual(route['return_to'],'understand')
        self.assertEqual(route['judgment'],judge)
        self.assertNotIn('previous_selection',context(client.calls[5][2]))

    def test_unknown_source_is_not_forced_to_understanding_or_another_stage(self):
        judge = checked('unknown',status='uncertain')
        judge['issues'].extend(checked('solve')['issues'])
        result, client = self.execute([understood(), selected(), solved(), judge])
        self.assertEqual(len(client.calls),4)
        self.assertFalse(result['rework_used'])
        self.assertEqual(result['verification_status'],'uncertain')
        self.assertEqual(result['checker_output']['return_to'],'none')

    def test_uncertain_without_issues_does_not_invent_a_route(self):
        result, _ = self.execute([understood(), selected(), solved(), checked(status='uncertain')])
        self.assertEqual(result['checker_output']['return_to'],'none')
        self.assertEqual(result['verification_status'],'uncertain')

    def test_resume_after_empty_selection_keeps_full_table_fallback(self):
        empty = {'action':'extract','regions':[], 'uncertainties':[]}
        first = FakeClient([understood(),empty,KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            flow.run_sample(self.row,self.folder,self.cfg,self.prompts,first,self.root/'run.log')
        result, client = self.execute([solved('[]'),checked()])
        self.assertEqual([c[0] for c in client.calls],['solve','check'])
        self.assertEqual(context(client.calls[0][2])['subtables'][0]['source_rows'],[1,2])
        self.assertFalse(result['rework_used'])

    def test_local_and_original_positions_are_both_program_marked(self):
        result, client = self.execute([understood(EXAMPLE_HTML), selected((6, 6), (2, 3)), solved(), checked()])
        html = context(client.calls[2][2])["subtables"][0]["html"]
        self.assertIn('data-local-row="1"', html)
        self.assertIn('data-local-col="1"', html)
        self.assertIn('data-source-row="6"', html)
        self.assertIn('data-source-col="2"', html)

    def test_coordinate_resolution_precedes_check_and_does_not_rewrite_other_numbers(self):
        output = solved('Value 95 at {{coord:p1}}; 4005000000', [request()])
        result, client = self.execute([understood(EXAMPLE_HTML), selected((6, 6), (2, 3)), output, checked()])
        self.assertEqual(result["answer"], 'Value 95 at (6, 2); 4005000000')
        self.assertEqual(context(client.calls[-1][2])["candidate"]["answer"], result["answer"])
        self.assertEqual(result["solver_output"]["answer"], output["answer"])
        self.assertEqual(result["counts"]["model_calls"], 4)
        self.assertEqual(result["counts"]["tool_calls"], 4)

    def test_merge_corners_span_and_covered_position_are_distinct(self):
        output = solved('{{coord:p}} {{coord:tl}} {{coord:br}} span={{coord:span}}', [
            request("p", (2, 1)), request("tl", (2, 1), "top_left"),
            request("br", (2, 1), "bottom_right"), request("span", (2, 1), "rowspan")])
        result, _ = self.execute([understood(EXAMPLE_HTML), selected((3, 3), (1, 2)), output, checked()])
        self.assertEqual(result["answer"], '(3, 1) (2, 1) (3, 1) span=2')
        self.assertEqual(len(list(self.folder.glob('r00_04t_map*.json'))), 1)

    def test_zero_based_and_json_output_are_explicit(self):
        output = solved('{"location":{{coord:p1}}}', [request(position=(2, 1), base=0, fmt="json")])
        result, _ = self.execute([understood(EXAMPLE_HTML), selected((3, 3), (1, 2)), output, checked()])
        self.assertEqual(json.loads(result["answer"]), {"location": [2, 0]})

    def test_more_than_64_coordinates_are_batched_without_extra_model_calls(self):
        html = '<table><tr>'+''.join('<td>'+str(i)+'</td>' for i in range(130))+'</tr></table>'
        requests = [request("p"+str(i), (1, i+1), fmt="json") for i in range(128)]
        output = solved('['+','.join('{{coord:p'+str(i)+'}}' for i in range(128))+']', requests)
        result, client = self.execute([understood(html), selected((1, 1), (1, 130)), output, checked()])
        self.assertEqual(json.loads(result["answer"])[-1], [1, 128])
        self.assertEqual(len(list(self.folder.glob('r00_04t_map*.json'))), 2)
        self.assertEqual(len(client.calls), 4)

    def test_repeated_reference_is_resolved_consistently(self):
        result, _ = self.execute([understood(), selected(), solved('{{coord:p1}}={{coord:p1}}', [request(position=(2, 2))]), checked()])
        self.assertEqual(result["answer"], '(2, 2)=(2, 2)')

    def test_selector_readback_receives_tool_result_not_an_interpretation(self):
        result, client = self.execute([understood(), inspect(), selected(), solved(), checked()])
        self.assertEqual([x[0] for x in client.calls], ["understand", "select", "select", "solve", "check"])
        second = context(client.calls[2][2])
        self.assertEqual(second["tool_results"][0]["data"]["cells"][0]["content_html"], "8")
        self.assertEqual(second["previous_request"]["action"], "inspect")
        self.assertEqual(result["counts"]["rework_rounds"], 0)

    def test_checker_readback_retains_same_candidate(self):
        result, client = self.execute([understood(), selected(), solved(), inspect(), checked()])
        self.assertEqual(context(client.calls[-1][2])["candidate"], context(client.calls[-2][2])["candidate"])
        self.assertIn("tool_results", context(client.calls[-1][2]))
        self.assertEqual(result["counts"]["model_calls"], 5)

    def test_failed_read_is_returned_as_failure_not_missing_cell(self):
        result, client = self.execute([understood(), inspect("absent"), selected(), solved(), checked()])
        response = context(client.calls[2][2])["tool_results"][0]
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "UNKNOWN_TABLE")
        self.assertEqual(result["verification_status"], "passed")

    def test_model_cannot_override_host_version_in_read_request(self):
        query = inspect()
        query["queries"][0]["table_version"] = "foreign"
        result, client = self.execute([understood(), query, selected(), solved(), checked()])
        reply = context(client.calls[2][2])["tool_results"][0]
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["table_version"], result["identity"]["table_version"])

    def test_repeated_selector_read_is_bounded(self):
        result, client = self.execute([understood(), inspect(), inspect()])
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(result["verification_status"], "tool_failed")
        self.assertIsNone(result["final"])

    def test_repeated_checker_read_keeps_candidate_unverified(self):
        result, client = self.execute([understood(), selected(), solved(), inspect(), inspect()])
        self.assertEqual(result["answer"], "8")
        self.assertIsNone(result["checker_output"])
        self.assertEqual(result["verification_status"], "tool_failed")

    def test_selection_rework_preserves_table_but_changes_extraction(self):
        result, client = self.execute([understood(), selected(), solved(), checked("select"), selected((2, 2), (2, 2)), solved(), checked()])
        self.assertEqual([x[0] for x in client.calls], ["understand", "select", "solve", "check", "select", "solve", "check"])
        first = flow.transport.read_json(self.folder/'r00_03_extract.json')
        last = flow.transport.read_json(self.folder/'r01_03_extract.json')
        self.assertNotEqual(first["extraction_id"], last["extraction_id"])
        self.assertEqual(context(client.calls[1][2])["table_index"]["table_version"], context(client.calls[4][2])["table_index"]["table_version"])
        self.assertNotIn("repair_feedback", context(client.calls[5][2]))

    def test_understanding_rework_invalidates_positions_and_all_downstream_artifacts(self):
        updated = understood('<table><tr><td>Only</td></tr></table>')
        result, client = self.execute([understood(), selected(), solved(), checked("understand"), updated,
                                       selected((1, 1), (1, 1)), solved("Only"), checked()])
        first = context(client.calls[1][2])["table_index"]["table_version"]
        second = context(client.calls[5][2])["table_index"]["table_version"]
        self.assertNotEqual(first, second)
        self.assertNotIn("previous_selection", context(client.calls[5][2]))
        self.assertEqual(result["identity"]["table_version"], second)
        self.assertEqual(result["artifacts"]["extraction"], 'r01_03_extract.json')
        state = flow.transport.read_json(self.folder/'communication_state.json')
        self.assertEqual(state["identity"], result["identity"])

    def test_solver_rework_reuses_exact_subtables_and_extraction(self):
        result, client = self.execute([understood(), selected(), solved(), checked("solve"), solved("9"), checked()])
        self.assertEqual([x[0] for x in client.calls], ["understand", "select", "solve", "check", "solve", "check"])
        first, second = context(client.calls[2][2]), context(client.calls[4][2])
        self.assertEqual(first["subtables"], second["subtables"])
        self.assertIn("repair_feedback", second)
        self.assertEqual(len(list(self.folder.glob('*03t_extract.json'))), 1)

    def test_invalid_reference_returns_to_solver_and_uses_shared_quota(self):
        result, client = self.execute([understood(), selected(), solved('{{coord:missing}}'), solved("8"), checked()])
        self.assertEqual([x[0] for x in client.calls], ["understand", "select", "solve", "solve", "check"])
        self.assertEqual(result["rework_target"], "solve")
        self.assertEqual(result["counts"]["rework_rounds"], 1)

    def test_twice_bad_coordinate_does_not_publish_template_as_answer(self):
        invalid = solved('{{coord:p1}}', [request(position=(999, 1))])
        result, client = self.execute([understood(), selected(), invalid, invalid])
        self.assertEqual(result["answer"], "")
        self.assertIsNone(result["final"])
        self.assertEqual(result["verification_status"], "tool_failed")

    def test_duplicate_and_unused_coordinate_requests_fail_explicitly(self):
        for output in (solved('{{coord:p1}}', [request(), request()]), solved('8', [request()])):
            # Use separate sample state so cached requests do not cross cases.
            self.folder = self.root/('case_'+str(len(list(self.root.iterdir()))))
            result, _ = self.execute([understood(), selected(), output, output])
            self.assertEqual(result["verification_status"], "tool_failed")

    def test_invalid_html_returns_understanding_and_does_not_guess_blanks(self):
        bad = understood('<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>')
        result, client = self.execute([bad, understood(), selected(), solved(), checked("select")])
        self.assertEqual([x[0] for x in client.calls], ["understand", "understand", "select", "solve", "check"])
        self.assertEqual(result["verification_status"], "needs_revision")
        self.assertEqual(result["termination_reason"], "rework_budget_exhausted")

    def test_second_checker_failure_is_not_converted_to_pass(self):
        result, _ = self.execute([understood(), selected(), solved(), checked("select"), selected(), solved(), checked("understand")])
        self.assertEqual(result["verification_status"], "needs_revision")
        self.assertEqual(result["counts"]["model_calls"], 7)

    def test_host_storage_failure_does_not_ask_model_to_fix_table(self):
        with patch('tool_loop_runtime.communication.TableTools', side_effect=ProgramError('STATE_IO_ERROR', 'storage failed', recovery='fix_host_storage')):
            result, client = self.execute([understood()])
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(result["rework_used"])
        self.assertEqual(result["verification_status"], "tool_failed")

    def test_resume_after_query_does_not_repeat_model_requests(self):
        first = FakeClient([understood(), inspect(), KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            flow.run_sample(self.row, self.folder, self.cfg, self.prompts, first, self.root/'run.log')
        result, second = self.execute([selected(), solved(), checked()])
        self.assertEqual([x[0] for x in second.calls], ["select", "solve", "check"])
        self.assertEqual(result["counts"]["model_calls"], 5)

    def test_resume_after_coordinate_resolution_has_same_candidate(self):
        output = solved('{{coord:p1}}', [request(position=(2, 2))])
        first = FakeClient([understood(), selected(), output, KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            flow.run_sample(self.row, self.folder, self.cfg, self.prompts, first, self.root/'run.log')
        result, client = self.execute([checked()])
        self.assertEqual([x[0] for x in client.calls], ["check"])
        self.assertEqual(result["answer"], '(2, 2)')
        self.assertEqual(result["counts"]["model_calls"], 4)

    def test_format_repair_has_separate_model_attempt_count(self):
        result, client = self.execute([understood(), 'bad JSON', selected(), solved(), checked()])
        self.assertEqual(result["counts"]["model_calls"], 5)
        self.assertEqual(result["counts"]["rework_rounds"], 0)

    def test_snapshot_pins_external_tools_and_has_no_reference_leak(self):
        source = self.root/'selected.json'
        flow.transport.save_json(source, [dict(self.row, reference='GOLD_DO_NOT_SEND')])
        run_root = self.root/'prepared_run'
        with patch.object(runner, 'run_root', return_value=run_root):
            root, rows, _ = runner.prepare(self.cfg, 'region_loop_tools_offline', source, 1)
        self.assertNotIn('reference', rows[0])
        snapshot = flow.transport.read_json(root/'config.json')
        self.assertIn('table_tools/service.py', [p.replace('\\','/') for p in snapshot['implementation_sha256']])

    def test_report_separates_model_tool_and_rework_counts(self):
        self.execute([understood(), selected(), solved(), checked()])
        flow.transport.save_json(self.root/'selected.json', [self.row])
        summary = runner.report(self.root)
        self.assertEqual(summary['execution_counts'], {'model_calls': 4, 'tool_calls': 3, 'rework_rounds': 0})
        self.assertNotIn('accuracy', summary)


class ProtocolTests(unittest.TestCase):
    def test_actions_cannot_be_silently_inferred_from_natural_language(self):
        with self.assertRaises(ProtocolError):
            decode(json.dumps({'regions': [], 'uncertainties': []}), 'select')

    def test_coordinate_types_and_components_are_explicit(self):
        for item in (request(position=(True, 1)), dict(request(), output_base=True), dict(request(), component='guess')):
            with self.assertRaises(ProtocolError):
                decode(json.dumps(solved('{{coord:p1}}', [item])), 'solve')

    def test_pass_requires_consistent_route(self):
        with self.assertRaises(ProtocolError):
            decode(json.dumps(dict(checked(), return_to='solve')), 'check')

    def test_checker_requires_observation_conflict_location_and_stage(self):
        for field in ('observation','conflict','location','source_stage'):
            value = checked('select')
            value['issues'][0].pop(field)
            with self.assertRaises(ProtocolError):
                decode(json.dumps(value),'check')

    def test_unknown_source_requires_uncertain_status(self):
        with self.assertRaises(ProtocolError):
            decode(json.dumps(checked('unknown')),'check')
        self.assertEqual(decode(json.dumps(checked('unknown',status='uncertain')),'check')['status'],'uncertain')

    def test_pass_with_an_issue_or_legacy_checker_fields_is_rejected(self):
        for value in (checked('select',status='passed'), {'action':'judge','status':'needs_revision','return_to':'select','issues':[{'problem':'x','location':'y'}]}):
            with self.assertRaises(ProtocolError):
                decode(json.dumps(value),'check')


if __name__ == '__main__':
    unittest.main()
