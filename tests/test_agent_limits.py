"""Bounded real workflow regressions using fake model responses, no APIs."""
import json
import unittest
from copy import deepcopy
from unittest.mock import patch
import test_loop as fixtures
from test_loop import understood, selected, solved, checked, inspect, request, context, FakeClient
from tool_loop_runtime import pipeline as flow
from tool_loop_runtime.budgets import Budgets


def direct_selection(html=None, rows=None, columns=None):
    return {"subtables": [{"id": "s1", "source_table_id": "t1",
            "html": html or understood()["tables"][0]["html"],
            "source_rows": rows or [1, 2], "source_columns": columns or [1, 2]}], "uncertainties": []}


# Reuse the fixture without inheriting the old behavior-specific tests.
class LimitTests(unittest.TestCase):
    setUp = fixtures.IntegrationTests.setUp
    execute = fixtures.IntegrationTests.execute

    def test_selector_limit_switches_to_html_and_does_not_execute_sixth_tool(self):
        result, client = self.execute([understood()] + [inspect()]*5 + [direct_selection(), solved('8'), checked()])
        self.assertEqual(result['answer'], '8')
        self.assertEqual(result['agent_budgets']['by_role']['select']['tool'], 5)
        self.assertEqual(result['execution_status'], 'model_fallback')
        self.assertEqual(result['counts']['rework_rounds'], 0)
        self.assertEqual(result['counts']['tool_calls'], 6)  # host registration + 5 selector calls
        solve_input = context(next(c[2] for c in client.calls if c[0]=='solve'))
        self.assertEqual(solve_input['subtables'][0]['source_rows'], [1,2])
        self.assertNotIn('tool_results', solve_input)
        self.assertNotIn('table_version', solve_input['subtables'][0])

    def test_checker_has_separate_five_query_budget(self):
        result, client = self.execute([understood(),selected(),solved()] + [inspect()]*6 + [checked()])
        counts = result['agent_budgets']['by_role']
        self.assertEqual(counts['select']['tool'], 2)
        self.assertEqual(counts['check']['tool'], 5)
        self.assertEqual(result['answer'], '8')
        self.assertTrue(client.calls[-1][1].endswith('_direct'))

    def test_images_are_independent_and_initial_attachment_is_not_charged(self):
        self.cfg=flow.transport.read_json(fixtures.BASE/'configs/legacy_mixed_B.json')
        result, client = self.execute([understood()] + [{'action':'view_image'}]*6 + [direct_selection(),solved(),checked()])
        counts = result['agent_budgets']['by_role']['select']
        self.assertEqual(counts, {'tool':1,'image':5})
        calls = [c for c in client.calls if c[0]=='select']
        self.assertEqual(sum(isinstance(c[2]['messages'][1]['content'],list) for c in calls),6)
        self.assertIsInstance(calls[-1][2]['messages'][1]['content'],str)

    def test_budget_survives_checker_rework(self):
        result, _ = self.execute([understood(),selected(),solved(),checked('select')] +
            [inspect()]*4 + [direct_selection(),solved(),checked('understand')])
        self.assertEqual(result['agent_budgets']['by_role']['select']['tool'],5)
        self.assertEqual(result['counts']['rework_rounds'],1)
        self.assertEqual(result['verification_status'],'needs_revision')

    def test_resume_replays_receipts_without_double_charging(self):
        first=FakeClient([understood(),inspect(),inspect(),KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            flow.run_sample(self.row,self.folder,self.cfg,self.prompts,first,self.root/'run.log')
        result, client=self.execute([inspect(),inspect(),inspect(),direct_selection(),solved(),checked()])
        self.assertEqual(result['agent_budgets']['by_role']['select']['tool'],5)
        self.assertEqual(result['counts']['tool_calls'],7)  # registration runs again, agent calls are replayed
        self.assertEqual(result['answer'],'8')
        self.assertNotIn('understand',[c[0] for c in client.calls])

    def test_real_invalid_grids_use_direct_prompts_without_inspect(self):
        grids=[
            '<table><tr><th>Tickets sold</th></tr><tr><th>Day</th><th>Tickets</th></tr><tr><td>Sunday</td><td>53</td></tr></table>',
            '<table><tr><th>Peanuts</th></tr><tr><th>Stem</th><th>Leaf</th></tr><tr><td>5</td><td>0 9</td></tr></table>',
            '<table><tr><th>Dancer</th><th>Conductor</th></tr><tr><th>Income</th><td>33000</td><td>51500</td></tr></table>',
            '<table><tr><th>Blocks from school</th></tr><tr><th>Name</th><th>Blocks</th></tr><tr><td>Omar</td><td>10</td></tr></table>']
        for i,html in enumerate(grids):
            self.folder=self.root/('grid'+str(i))
            result,client=self.execute([understood(html),direct_selection(html),solved('answer'),checked()])
            self.assertEqual([c[0] for c in client.calls],['understand','select','solve','check'])
            self.assertTrue(all(c[1].endswith('_direct') for c in client.calls[1:]))
            self.assertEqual(result['counts']['tool_queries'],0)
            self.assertEqual(result['execution_status'],'model_fallback')
            self.assertEqual(context(client.calls[1][2])['tables'],understood(html)['tables'])
            self.assertNotIn('tool_results',context(client.calls[2][2]))

    def test_coordinate_calls_stop_at_five_then_model_finishes_answer(self):
        groups=[dict(request('p'+str(i)),subtable_id='missing'+str(i)) for i in range(6)]
        output=solved(' '.join('{{coord:p'+str(i)+'}}' for i in range(6)),groups)
        result,client=self.execute([understood(),selected(),output,solved('(2,2)'),checked()])
        self.assertEqual(result['agent_budgets']['by_role']['solve']['tool'],5)
        self.assertEqual(result['answer'],'(2,2)')
        self.assertEqual(result['execution_status'],'model_fallback')
        self.assertEqual(len([c for c in client.calls if c[0]=='solve']),2)

    def test_direct_interface_repair_keeps_same_role_and_correct_contract(self):
        bad=understood('<table><tr><th>Title</th></tr><tr><td>A</td><td>8</td></tr></table>')
        result,client=self.execute([bad,selected(),direct_selection(),solved(),checked()])
        repair=context(client.calls[2][2])
        self.assertIn('subtables:',repair['required_interface'])
        self.assertEqual(result['answer'],'8')
        self.assertEqual(result['counts']['format_repairs'],1)

    def test_empty_inspect_cannot_loop_forever(self):
        result,_=self.execute([understood()]+[{'action':'inspect','queries':[]}]*5+[direct_selection(),solved(),checked()])
        self.assertEqual(result['agent_budgets']['by_role']['select']['tool'],5)
        self.assertEqual(result['answer'],'8')

    def test_unknown_mapping_is_forwarded_not_invented_or_rejected(self):
        bad=understood('<table><tr><th>Title</th></tr><tr><td>A</td><td>8</td></tr></table>')
        selection=direct_selection(rows=[None,2])
        result,client=self.execute([bad,selection,solved('8'),checked()])
        self.assertEqual(context(client.calls[2][2])['subtables'][0]['source_rows'],[None,2])
        self.assertEqual(result['execution_status'],'model_fallback')

    def test_image_allowance_survives_understanding_rework(self):
        view = {'action':'view_image'}
        result,client=self.execute([view]*3+[understood(),selected(),solved(),checked('understand')]
            +[view]*3+[understood(),selected(),solved(),checked()])
        self.assertEqual(result['agent_budgets']['by_role']['understand']['image'],5)
        self.assertEqual(result['agent_budgets']['by_role']['check']['image'],0)
        self.assertEqual(result['counts']['rework_rounds'],1)
        self.assertEqual(result['answer'],'8')
        self.assertTrue(result['artifacts']['understanding'].endswith('_direct.json'))

    def test_resume_image_request_does_not_reset_or_double_charge(self):
        first=FakeClient([{'action':'view_image'},KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            flow.run_sample(self.row,self.folder,self.cfg,self.prompts,first,self.root/'run.log')
        result,client=self.execute([understood(),selected(),solved(),checked()])
        self.assertEqual(result['agent_budgets']['by_role']['understand']['image'],1)
        self.assertEqual(result['answer'],'8')

    def test_transient_tool_retries_also_respect_five_call_limit(self):
        from table_tools import TableTools
        original=TableTools.call
        def failing_query(service,role,name,args):
            if name=='inspect_cells':
                return {'ok':False,'data':None,'error':{'code':'STATE_IO_ERROR','message':'offline simulated storage failure'}}
            return original(service,role,name,args)
        with patch.object(TableTools,'call',failing_query):
            result,_=self.execute([understood(),inspect(),inspect(),direct_selection(),solved(),checked()])
        self.assertEqual(result['agent_budgets']['by_role']['select']['tool'],5)
        self.assertEqual(result['counts']['tool_calls'],6)
        self.assertEqual(result['counts']['tool_execution_retries'],2)
        self.assertEqual(result['execution_status'],'model_fallback')
