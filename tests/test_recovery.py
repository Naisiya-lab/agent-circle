"""Regression cases from Qwen failures; no external models or API keys."""
import io
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tool_loop_runtime import pipeline, runner, baseline, recovery, transport
from tool_loop_runtime.protocol import decode, ProtocolError
from table_tools import TableTools, ProgramError
from table_tools.html_grid import Table
from test_loop import direct_selected, understood, selected, solved, checked, request, inspect, context, FakeClient

ROOT = Path(__file__).resolve().parents[1]


def raw_reply(content, finish='stop'):
    return {'_raw_response': {'choices':[{'finish_reason':finish,'message':{'content':content}}],
                              'usage':{'prompt_tokens':10,'completion_tokens':4,'total_tokens':14}}}


class HandoffRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='rc_',dir=ROOT/'offline_checks')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        image=self.root/'x.jpg';image.write_bytes(b'fake image')
        self.row={'id':'x','question':'Count rows and columns','image':str(image),
                  'image_sha256':transport.digest(image.read_bytes()),'task_type':'TSD'}
        self.cfg=transport.read_json(ROOT/'configs/legacy_mixed_B.json')
        self.folder=self.root/'samples/x'
        self.prompts=runner.prompt_map(self.cfg)

    def execute(self,replies,cfg=None):
        client=FakeClient(replies)
        result=pipeline.run_sample(self.row,self.folder,cfg or self.cfg,self.prompts,client,self.root/'run.log')
        return result,client

    def test_real_repeated_source_stage_is_read_without_retry(self):
        text='{"action":"judge","status":"needs_revision","issues":[{"observation":"3 rows","conflict":"1 row","source_stage":"select","location":"t1","source_stage":"select"}]}'
        cfg=deepcopy(self.cfg);cfg['loop']['max_rework_rounds']=0
        result,client=self.execute([understood(),selected(),solved('1'),text],cfg)
        self.assertEqual(result['verification_status'],'needs_revision')
        self.assertEqual(len(client.calls),4)
        self.assertEqual(result['counts']['format_repairs'],0)
        self.assertEqual(result['checker_output']['issues'][0]['source_stage'],'select')

    def test_conflicting_duplicate_goes_back_to_same_role_with_specific_error(self):
        conflict='{"action":"judge","status":"passed","status":"uncertain","issues":[]}'
        result,client=self.execute([understood(),selected(),solved(),conflict,checked()])
        self.assertEqual([c[0] for c in client.calls],['understand','select','solve','check','check'])
        fix=context(client.calls[-1][2])
        self.assertIn('Conflicting duplicate JSON key: status',' '.join(fix['interface_errors']))
        self.assertEqual(fix['previous_output'],conflict)
        self.assertIsInstance(client.calls[-1][2]['messages'][1]['content'],str)
        self.assertEqual(result['counts']['format_repairs'],1)
        self.assertEqual(result['counts']['rework_rounds'],0)

    def test_length_response_with_readable_json_is_accepted(self):
        result,client=self.execute([understood(),selected(),raw_reply(json.dumps(solved('8')),'length'),checked()])
        self.assertEqual(result['answer'],'8')
        self.assertEqual(len(client.calls),4)
        saved=transport.read_json(self.folder/'r00_04_solve.json')
        self.assertEqual(saved['raw_response']['choices'][0]['finish_reason'],'length')
        self.assertEqual(result['counts']['format_repairs'],0)

    def test_two_format_repairs_then_preserve_solver_answer(self):
        result,client=self.execute([understood(),selected(),solved('8'),'broken','still broken','still broken'])
        self.assertEqual(result['answer'],'8')
        self.assertEqual(result['execution_status'],'communication_failed')
        self.assertEqual(result['verification_status'],'communication_failed')
        self.assertEqual(result['communication_error']['role'],'check')
        self.assertEqual(result['counts']['format_repairs'],2)
        self.assertEqual(result['counts']['rework_rounds'],0)
        self.assertEqual([c[0] for c in client.calls][-3:],['check']*3)

    def test_resume_communication_failure_reuses_three_completed_roles(self):
        self.execute([understood(),selected(),solved('8'),'bad','bad','bad'])
        result,client=self.execute([checked()],dict(self.cfg,_retry_protocol_errors=True))
        self.assertEqual([c[0] for c in client.calls],['check'])
        self.assertEqual(result['answer'],'8')
        self.assertEqual(result['execution_status'],'completed')

    def test_failed_understanding_format_is_not_passed_as_raw_text(self):
        result,client=self.execute(['not json','not json','not json'])
        self.assertEqual([c[0] for c in client.calls],['understand']*3)
        self.assertEqual(result['execution_status'],'communication_failed')
        self.assertEqual(result['answer'],'')
        self.assertNotIn('raw_output',json.dumps(result))

    def test_invalid_region_is_fixed_by_selector_without_consuming_checker_round(self):
        result,client=self.execute([understood(),selected((1,999)),selected(),solved(),checked()])
        self.assertEqual([c[0] for c in client.calls],['understand','select','select','solve','check'])
        self.assertIn('tool_execution_feedback',context(client.calls[2][2]))
        self.assertEqual(result['counts']['tool_input_repairs'],1)
        self.assertEqual(result['counts']['rework_rounds'],0)

    def test_extra_region_metadata_is_not_sent_as_tool_arguments(self):
        selection=selected();selection['regions'][0]['explanation']='harmless additional metadata'
        result,client=self.execute([understood(),selection,solved('8'),checked()])
        self.assertEqual(len(client.calls),4)
        self.assertEqual(result['execution_status'],'completed')
        original=transport.read_json(self.folder/'r00_selection_resolution.json')['requested_regions'][0]
        self.assertIn('explanation',original)

    def test_exhausted_region_repairs_use_original_html_in_solver(self):
        invalid=selected((1,999))
        result,client=self.execute([understood(),invalid,invalid,invalid,direct_selected(),solved('2 by 2'),checked()])
        self.assertEqual(result['execution_status'],'model_fallback')
        solve_input=context(next(c[2] for c in client.calls if c[0]=='solve'))
        self.assertEqual(solve_input['tables'],understood()['tables'])
        self.assertNotIn('mapping_available',solve_input)
        self.assertEqual(solve_input['subtables'],direct_selected()['subtables'])
        self.assertEqual(result['counts']['rework_rounds'],0)

    def test_queries_batch_and_respect_total_budget(self):
        html='<table><tr>'+''.join('<td>x</td>' for _ in range(130))+'</tr></table>'
        query=inspect(positions=tuple((1,i) for i in range(1,131)))
        result,client=self.execute([understood(html),query,inspect(),inspect(),direct_selected(html),solved('130'),checked()])
        after=context(client.calls[2][2])
        self.assertEqual([len(x['data']['cells']) for x in after['tool_results']],[64,64,2])
        self.assertEqual(result['counts']['tool_queries'],3)
        self.assertEqual(result['counts']['rework_rounds'],0)
        self.assertEqual(result['agent_budgets']['by_role']['select']['tool'],5)

    def test_one_checker_rework_remains_after_many_tool_queries(self):
        result,client=self.execute([understood(),inspect(),inspect(),selected(),solved('8'),checked('solve'),solved('9'),checked('solve')])
        self.assertEqual([c[0] for c in client.calls].count('check'),2)
        self.assertEqual(result['counts']['rework_rounds'],1)
        self.assertEqual(result['verification_status'],'needs_revision')
        self.assertEqual(result['answer'],'9')

    def test_direct_coordinate_and_unused_invalid_request_pass_unchanged(self):
        output=solved('(7, 3)',[{'id':'unused','position':'bad'}])
        result,client=self.execute([understood(),selected(),output,checked()])
        self.assertEqual(result['answer'],'(7, 3)')
        self.assertEqual(result['counts']['model_calls'],4)
        self.assertFalse(list(self.folder.glob('*04t_map*')))

    def test_conflicting_coordinate_requests_are_not_silently_overwritten(self):
        output=solved('{{coord:p1}}',[request(),request(position=(2,2))])
        result,client=self.execute([understood(),selected(),output,solved('(2, 2)'),checked()])
        feedback=context(client.calls[3][2])['tool_execution_feedback']
        self.assertEqual(feedback[0]['error']['code'],'CONFLICTING_COORDINATE_REQUEST')
        self.assertEqual(result['answer'],'(2, 2)')
        self.assertEqual(result['counts']['rework_rounds'],0)

    def test_single_request_http400_reaches_next_sample_in_runner(self):
        source=self.root/'source.json';transport.save_json(source,[self.row,dict(self.row,id='y')])
        run=self.root/'run'
        bad=transport.ApiFailure('HTTP 400: invalid request',fatal=False,retryable=False,http_status=400)
        client=FakeClient([bad,understood(),selected(),solved('8'),checked()])
        with patch.object(runner,'run_root',return_value=run),patch.object(transport,'load_api_keys',return_value={}),patch.object(transport,'RoleApiClient',return_value=client):
            code=runner.run(self.cfg,'fixture',source,2)
        self.assertEqual(code,1)
        self.assertEqual(transport.read_json(run/'samples/x/result.json')['execution_status'],'communication_failed')
        self.assertEqual(transport.read_json(run/'samples/y/result.json')['answer'],'8')
        self.assertEqual(transport.read_json(run/'summary.json')['completed'],2)
        self.assertEqual(transport.read_json(run/'summary.json')['errors'],1)

    def test_explicit_tool_retry_refreshes_only_caller_and_downstream(self):
        invalid=selected((1,999))
        self.cfg['tools']['input_retries']=0
        self.execute([understood(),invalid,direct_selected(),solved('8'),checked()])
        cfg=recovery.prepare_tool_retry(self.folder,self.cfg,retry_errors=True)
        result,client=self.execute([selected(),solved('8'),checked()],cfg)
        recovery.finish_tool_retry(self.folder)
        self.assertEqual([x[0] for x in client.calls],['select','solve','check'])
        self.assertEqual(result['execution_status'],'completed')
        self.assertIn('tool_retry_context',context(client.calls[0][2]))
        self.assertTrue(list((self.folder/'retry_stages').glob('*/*02_select*.json')))
        self.assertGreater(result['counts']['model_calls'],len(client.calls))

    def test_interrupted_tool_retry_reuses_new_completed_stages(self):
        invalid=selected((1,999))
        self.cfg['tools']['input_retries']=0
        self.execute([understood(),invalid,direct_selected(),solved('8'),checked()])
        cfg=recovery.prepare_tool_retry(self.folder,self.cfg,retry_errors=True)
        with self.assertRaises(KeyboardInterrupt):
            self.execute([selected(),KeyboardInterrupt()],cfg)
        self.assertTrue(recovery.needs_run(self.folder))
        resumed=recovery.prepare_tool_retry(self.folder,self.cfg)
        result,client=self.execute([solved('8'),checked()],resumed)
        recovery.finish_tool_retry(self.folder)
        self.assertEqual([x[0] for x in client.calls],['solve','check'])
        self.assertEqual(result['execution_status'],'completed')

    def test_checker_communication_failure_during_tool_retry_resumes_checker_only(self):
        invalid=selected((1,999))
        self.cfg['tools']['input_retries']=0
        self.execute([understood(),invalid,direct_selected(),solved('8'),checked()])
        cfg=recovery.prepare_tool_retry(self.folder,self.cfg,retry_errors=True)
        failed,_=self.execute([selected(),solved('8'),'bad','bad','bad'],cfg)
        recovery.finish_tool_retry(self.folder)
        self.assertEqual(failed['execution_status'],'communication_failed')
        resumed=recovery.prepare_tool_retry(self.folder,dict(self.cfg,_retry_protocol_errors=True))
        result,client=self.execute([checked()],resumed)
        recovery.finish_tool_retry(self.folder)
        self.assertEqual([x[0] for x in client.calls],['check'])
        self.assertEqual(result['execution_status'],'completed')
        self.assertEqual(result['answer'],'8')

    def test_tool_retry_preserves_already_consumed_checker_rework(self):
        invalid=selected((1,999))
        self.cfg['tools']['input_retries']=0
        first,_=self.execute([understood(),invalid,direct_selected(),solved('8'),checked('solve'),solved('8'),checked(status='uncertain')])
        self.assertEqual(first['counts']['rework_rounds'],1)
        cfg=recovery.prepare_tool_retry(self.folder,self.cfg,retry_errors=True)
        result,client=self.execute([selected(),solved('8'),checked('understand')],cfg)
        recovery.finish_tool_retry(self.folder)
        self.assertEqual([x[0] for x in client.calls],['select','solve','check'])
        self.assertEqual(result['counts']['rework_rounds'],1)
        self.assertEqual(result['verification_status'],'needs_revision')

    def test_error_history_and_retry_eligibility(self):
        recovery.record_error(self.folder,{'message':'old failure','fatal':False})
        self.assertTrue(recovery.needs_run(self.folder))
        limited={'execution_status':'tool_limited','answer':'8'}
        transport.save_json(self.folder/'result.json',limited)
        self.assertFalse(recovery.needs_run(self.folder))
        self.assertTrue(recovery.needs_run(self.folder,retry_errors=True))
        recovery.begin_retry(self.folder);recovery.clear_error(self.folder)
        events=[json.loads(x) for x in (self.folder/'recovery_history.jsonl').read_text().splitlines()]
        self.assertIn('old failure',json.dumps(events))
        self.assertIn(limited,[x['record'] for x in events])


class BaselineRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='br_',dir=ROOT/'offline_checks')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        image=self.root/'x.jpg';image.write_bytes(b'fixture')
        self.rows=[{'id':str(i),'question':'Original question','image':str(image),
                    'image_sha256':transport.digest(image.read_bytes()),'task_type':'TQA'} for i in range(2)]
        self.source=self.root/'source.json';transport.save_json(self.source,self.rows)
        from tool_loop_runtime.configuration import load_config
        with patch.dict('os.environ',{'MODEL_ENDPOINT':'http://127.0.0.1:8000/v1','SERVED_MODEL_NAME':'M'}):
            self.cfg=load_config(ROOT/'configs/models/qwen25_vl_7b.json')
        self.cfg['experiment']['mode']='baseline'

    def test_length_truncated_baseline_is_kept(self):
        client=FakeClient([raw_reply('partial but useful','length'),raw_reply('next')])
        root=self.root/'run'
        with patch.object(runner,'run_root',return_value=root),patch.object(transport,'RoleApiClient',return_value=client):
            code=baseline.run(self.cfg,'fixture',self.source,2)
        self.assertEqual(code,0)
        result=transport.read_json(root/'samples/0/direct_result.json')
        self.assertEqual(result['answer'],'partial but useful')
        self.assertEqual(result['finish_reason'],'length')
        self.assertEqual(transport.read_json(root/'summary.json')['completed'],2)

    def test_bad_request_does_not_stop_baseline_shard_and_error_can_resume(self):
        client=FakeClient([transport.ApiFailure('HTTP 400: bad request',http_status=400),raw_reply('second')])
        root=self.root/'run'
        with patch.object(runner,'run_root',return_value=root),patch.object(transport,'RoleApiClient',return_value=client):
            code=baseline.run(self.cfg,'fixture',self.source,2)
        self.assertEqual(code,1)
        self.assertEqual(transport.read_json(root/'samples/1/direct_result.json')['answer'],'second')
        retry=FakeClient([raw_reply('first recovered')])
        with patch.object(runner,'run_root',return_value=root),patch.object(transport,'RoleApiClient',return_value=retry):
            code=baseline.run(self.cfg,'fixture',self.source,2)
        self.assertEqual(code,0)
        self.assertEqual(len(retry.calls),1)
        self.assertEqual(transport.read_json(root/'summary.json')['completed'],2)
        self.assertTrue((root/'samples/0/recovery_history.jsonl').exists())


class ExportTests(unittest.TestCase):
    def test_export_preserves_partial_answer_failure_metadata_and_denominator(self):
        import agent_circle
        import sys
        with tempfile.TemporaryDirectory(dir=ROOT/'offline_checks') as d:
            base=Path(d);root=base/'run';output=base/'export.jsonl'
            transport.save_json(root/'config.json',{'config':{'experiment':{'mode':'circle'},'models':{'solve':'M'}}})
            transport.save_json(root/'selected.json',[{'id':'done'},{'id':'pending'}])
            transport.save_json(root/'samples/done/result.json',{'id':'done','answer':'8','execution_status':'communication_failed','verification_status':'communication_failed'})
            argv=['agent_circle.py','export','--run','run','--output',str(output)]
            with patch.dict('os.environ',{'AGENT_CIRCLE_RUNS':str(base)}),patch.object(sys,'argv',argv):
                self.assertEqual(agent_circle.main(),0)
            rows=[json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['question_id'] for r in rows],['done','pending'])
            self.assertEqual(rows[0]['text'],'8')
            self.assertEqual(rows[0]['metadata']['execution_status'],'communication_failed')
            self.assertEqual(rows[1]['text'],'')
            self.assertFalse(rows[1]['metadata']['completed'])


class GridCompatibilityTests(unittest.TestCase):
    def test_optional_cell_row_and_group_ends_preserve_grid(self):
        a=Table('t','<table><tbody><tr><th>H<th>V<tr><td>A<td>8</table>')
        b=Table('t','<table><tbody><tr><th>H</th><th>V</th></tr><tr><td>A</td><td>8</td></tr></tbody></table>')
        self.assertEqual(a.html,b.html)
        self.assertEqual([(c.row,c.column,c.content) for c in a.cells],[(c.row,c.column,c.content) for c in b.cells])

    def test_presentation_tags_do_not_prevent_mapping(self):
        table=Table('t','<html><body><table><tr><td><label>A</label></td><td><mark>8</mark></td></tr></table></body></html>')
        self.assertEqual((table.rows,table.columns),(1,2))
        self.assertEqual(table.cells[0].content,'A')
        self.assertIn('8',table.cells[1].content)

    def test_rowspan_zero_respects_group_and_does_not_invent_cells(self):
        table=Table('t','<table><tbody><tr><td rowspan="0">A</td><td>1</td></tr><tr><td>2</td></tr></tbody></table>')
        self.assertEqual(table.grid[2,1].rowspan,2)
        self.assertEqual(len(table.cells),3)

    def test_previous_row_column_and_region_caps_are_not_task_limits(self):
        self.assertEqual(Table('t','<table>'+('<tr><td>x</td></tr>'*2001)+'</table>').rows,2001)
        self.assertEqual(Table('t','<table><tr>'+('<td>x</td>'*501)+'</tr></table>').columns,501)
        service=TableTools('many',understood())
        regions=[{'id':str(i),'source_table_id':'t1','rows':[1,1],'columns':[1,1]} for i in range(40)]
        response=service.call('select','extract_subtables',{'table_version':service.table_version,'regions':regions})
        self.assertTrue(response['ok'])
        self.assertEqual(len(response['data']['subtables']),40)


class ContextBudgetTests(unittest.TestCase):
    def test_context_overflow_changes_only_output_budget(self):
        msg="This model's maximum context length is 32768 tokens. However, you requested 33182 tokens (29086 in the messages, 4096 in the completion)."
        body={'messages':[{'role':'user','content':'unchanged'}],'max_tokens':4096,'temperature':0,'model':'M'}
        before=deepcopy(body)
        updated,receipt=transport.context_budget_adjustment(body,msg)
        self.assertEqual(updated,dict(before,max_tokens=3682))
        self.assertEqual(body,before)
        self.assertTrue(receipt['input_unchanged'])
        self.assertIsNone(transport.context_budget_adjustment(body,msg.replace('29086','40000')))

    def test_actual_http400_is_corrected_even_without_network_retries(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'offline_checks') as d:
            cfg=transport.read_json(ROOT/'configs/legacy_mixed_A.json');cfg.update(transport_retries=0)
            root=Path(d)
            message="This model's maximum context length is 32768 tokens. However, you requested 33182 tokens (29086 in the messages, 4096 in the completion)."
            error=transport.urllib.error.HTTPError(cfg['endpoint'],400,'bad request',{},io.BytesIO(message.encode()))
            self.addCleanup(error.close)
            ok=io.BytesIO(b'{"choices":[{"message":{"content":"8"},"finish_reason":"stop"}]}')
            ok.headers={'Content-Type':'application/json'}
            calls=[]
            def opened(req,timeout):
                calls.append(json.loads(req.data))
                if len(calls)==1:raise error
                return ok
            with patch.object(transport.urllib.request.OpenerDirector,'open',side_effect=opened):
                raw=transport.ApiClient(cfg,'offline-key',root).call({'messages':[],'max_tokens':4096,'stream':False},'x','solve')
            self.assertEqual([x['max_tokens'] for x in calls],[4096,3682])
            self.assertEqual(raw['_transport']['effective_max_tokens'],3682)
            self.assertEqual(len(raw['_transport']['request_adjustments']),1)

    def test_http400_clears_old_network_outage_state(self):
        import time
        with tempfile.TemporaryDirectory(dir=ROOT/'offline_checks') as d:
            cfg=transport.read_json(ROOT/'configs/legacy_mixed_A.json')
            client=transport.ApiClient(cfg,'offline-key',Path(d))
            client.outage_started=time.monotonic()-1
            error=transport.urllib.error.HTTPError(cfg['endpoint'],400,'bad',{},io.BytesIO(b'bad request'))
            self.addCleanup(error.close)
            with patch.object(transport.urllib.request.OpenerDirector,'open',side_effect=error):
                with self.assertRaises(transport.ApiFailure):client.call({'stream':False},'x','solve')
            self.assertIsNone(client.outage_started)
            self.assertEqual(client.ready_at,0.0)

    def test_unrelated_http400_is_not_retried_or_marked_fatal(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'offline_checks') as d:
            cfg=transport.read_json(ROOT/'configs/legacy_mixed_A.json')
            error=transport.urllib.error.HTTPError(cfg['endpoint'],400,'bad',{},io.BytesIO(b'unsupported parameter'))
            self.addCleanup(error.close)
            with patch.object(transport.urllib.request.OpenerDirector,'open',side_effect=error) as opened:
                with self.assertRaises(transport.ApiFailure) as caught:
                    transport.ApiClient(cfg,'offline-key',Path(d)).call({'stream':False},'x','solve')
            self.assertFalse(caught.exception.fatal)
            self.assertEqual(opened.call_count,1)


if __name__=='__main__':unittest.main()
