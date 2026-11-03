"""Launcher tests use synthetic manifests and fake workers, never model weights or GPUs."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('full_four_gpu', ROOT/'scripts/full_qwen25_7b_4gpu.py')
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)

FAKE_WORKER = """
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('action')
p.add_argument('--mode');p.add_argument('--route');p.add_argument('--model-profile')
p.add_argument('--source');p.add_argument('--run');p.add_argument('--output')
a=p.parse_args();root=Path(os.environ['AGENT_CIRCLE_RUNS'])/a.run
events=Path(os.environ['FAKE_EVENTS'])
def event(name):
 events.mkdir(exist_ok=True)
 (events/(a.run+'_'+name+'.json')).write_text(json.dumps({'event':name,'mode':a.mode,'run':a.run,'endpoint':os.environ.get('MODEL_ENDPOINT'),'when':time.monotonic_ns()}))
if a.action=='export':
 rows=json.loads((root/'selected.json').read_text())
 results={r['id']:r for r in json.loads((root/'results.json').read_text())}
 Path(a.output).write_text(''.join(json.dumps({'question_id':r['id'],'text':results.get(r['id'],{}).get('answer',''),'model_id':'M'})+'\\n' for r in rows))
else:
 rows=json.loads(Path(a.source).read_text());root.mkdir(parents=True,exist_ok=True)
 (root/'selected.json').write_text(json.dumps(rows));event('start');results=[]
 for r in rows:
  d=root/'samples'/r['id'];d.mkdir(parents=True,exist_ok=True)
  if a.mode=='circle' and os.environ.get('FAKE_EARLY_STOP')=='1' and r['id']=='id0':
   (d/'error.json').write_text(json.dumps({'fatal':True,'retryable':False}));(root/'results.json').write_text(json.dumps(results));event('end');raise SystemExit(1)
  if r['id']=='id1':
   (d/'error.json').write_text(json.dumps({'fatal':False,'retryable':False}));continue
  result={'id':r['id'],'answer':'fixture','verification_status':'tool_failed' if r['id']=='id2' else 'passed'}
  (d/('result.json' if a.mode=='circle' else 'direct_result.json')).write_text(json.dumps(result));results.append(result)
 (root/'results.json').write_text(json.dumps(results));event('end');raise SystemExit(1)
"""


class FourGpuLauncherTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'offline_checks').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'offline_checks')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base/'source.json'
        self.rows = [{'id':f'id{i}','question':f'Q{i}','image':'remote/image.png',
                      'image_sha256':'fixture','task_type':'TQA','dataset_name':'fixture'} for i in range(9)]
        launch.save(self.source,self.rows)
        self.project=self.base/'project';self.project.mkdir()
        (self.project/'agent_circle.py').write_text(FAKE_WORKER)
        for rel in ('configs/runtime.json','configs/models/qwen25_vl_7b.json'):
            launch.save(self.project/rel,{})
        self.weights=self.base/'weights';self.weights.mkdir()
        launch.save(self.weights/'config.json',{'fixture':True})
        self.args=SimpleNamespace(project=self.project,source=self.source,weights=self.weights,
            shards_root=self.base/'shards',control_root=self.base/'control',runs_root=self.base/'runs',
            experiment='fixture',gpus=['0','1','2','3'],first_port=8000,model_name='M',
            context=32768,memory_utilization=0.85,max_num_seqs=2,start_timeout=3)

    def test_partition_preserves_fields_and_has_exact_coverage(self):
        original=self.source.read_bytes()
        rows,paths=launch.split_manifest(self.source,self.args.shards_root)
        self.assertEqual([len(launch.read(p)) for p in paths],[3,2,2,2])
        self.assertEqual({r['id'] for p in paths for r in launch.read(p)},{r['id'] for r in rows})
        self.assertEqual(launch.read(paths[0]),self.rows[::4])
        self.assertEqual(self.source.read_bytes(),original)
        launch.split_manifest(self.source,self.args.shards_root)
        launch.save(paths[1],[])
        with self.assertRaisesRegex(ValueError,'differs'):
            launch.split_manifest(self.source,self.args.shards_root)

    def test_duplicate_source_ids_rejected(self):
        launch.save(self.source,self.rows+[self.rows[0]])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            launch.split_manifest(self.source,self.args.shards_root)

    def test_completion_means_terminal_processing_not_correctness(self):
        root=self.args.runs_root/'test'
        launch.save(root/'samples/id0/result.json',{'verification_status':'tool_failed','answer':''})
        launch.save(root/'samples/id1/error.json',{'retryable':False})
        status=launch.scan_run(root,self.rows[:2],'circle')
        self.assertTrue(status['all_processed'])
        self.assertTrue(launch.phase_complete([status],[1]))
        self.assertFalse(launch.phase_complete([status],[130]))
        self.assertFalse(launch.scan_run(root,self.rows[:3],'circle')['all_processed'])
        launch.save(root/'samples/id1/error.json',{'retryable':True})
        self.assertTrue(launch.scan_run(root,self.rows[:2],'circle')['all_processed'])
        launch.save(root/'samples/id1/error.json',{'retryable':True,'fatal':True})
        self.assertFalse(launch.scan_run(root,self.rows[:2],'circle')['all_processed'])

    def test_plan_refuses_generation_or_core_changes(self):
        with redirect_stdout(io.StringIO()):
            launch.prepare(self.args);launch.prepare(self.args)
            self.args.context=16384
            with self.assertRaisesRegex(ValueError,'Frozen launch plan'):
                launch.prepare(self.args)

    def test_services_reuse_gpu0_and_launch_three_single_gpu_copies(self):
        self.args.control_root.mkdir()
        seen={};spawned=[]
        def probe(port,*_):
            seen[port]=seen.get(port,0)+1
            return {'id':'M'} if port==8000 or seen[port]>1 else None
        class Process:
            def __init__(self,pid):self.pid=pid
            def poll(self):return None
        def popen(command,**kwargs):
            spawned.append((command,kwargs))
            return Process(100+len(spawned))
        with patch.object(launch,'server_probe',side_effect=probe),patch.object(launch.subprocess,'Popen',side_effect=popen),redirect_stdout(io.StringIO()):
            launch.ensure_servers(self.args)
        self.assertEqual([kwargs['env']['CUDA_VISIBLE_DEVICES'] for _,kwargs in spawned],['1','2','3'])
        for command,_ in spawned:
            self.assertEqual(command[command.index('--tensor-parallel-size')+1],'1')
        self.assertTrue(launch.read(self.args.control_root/'services.json')[0]['reused'])

    def test_existing_service_wrong_model_is_rejected(self):
        opener=SimpleNamespace(open=lambda *_args,**_kwargs:io.BytesIO(b'{"data":[{"id":"OTHER"}]}'))
        with patch.object(launch.urllib.request,'build_opener',return_value=opener):
            with self.assertRaisesRegex(ValueError,'another model'):
                launch.server_probe(8000,'M',self.weights,32768)

    def argv(self):
        a=self.args
        return ['launcher','run','--project',str(a.project),'--source',str(a.source),
                '--weights',str(a.weights),'--shards-root',str(a.shards_root),
                '--control-root',str(a.control_root),'--runs-root',str(a.runs_root),
                '--experiment',a.experiment,'--model-name','M']

    def trace(self,events):
        return sorted([json.loads(p.read_text()) for p in events.glob('*.json')],key=lambda r:r['when'])

    def test_real_fake_processes_wait_for_all_circle_shards_before_baseline(self):
        events=self.base/'events.jsonl'
        with patch.dict('os.environ',{'FAKE_EVENTS':str(events),'FAKE_EARLY_STOP':'0'}),patch.object(sys,'argv',self.argv()),patch.object(launch,'ensure_servers'),redirect_stdout(io.StringIO()):
            self.assertEqual(launch.main(),0)
        trace=self.trace(events)
        circle_ends=[i for i,r in enumerate(trace) if r['mode']=='circle' and r['event']=='end']
        baseline_starts=[i for i,r in enumerate(trace) if r['mode']=='baseline' and r['event']=='start']
        self.assertEqual(len(circle_ends),4);self.assertEqual(len(baseline_starts),4)
        self.assertLess(max(circle_ends),min(baseline_starts))
        for mode in ('circle','baseline'):
            data=[json.loads(s) for s in (self.args.control_root/'exports'/f'{mode}_all.jsonl').read_text().splitlines()]
            self.assertEqual([r['question_id'] for r in data],[r['id'] for r in self.rows])
            self.assertEqual(data[1]['text'],'')  # Nonretryable failure retained, not dropped.
        self.assertFalse((self.args.control_root/'launcher.lock').exists())
        with patch.dict('os.environ',{'FAKE_EVENTS':str(events),'FAKE_EARLY_STOP':'0'}),patch.object(sys,'argv',self.argv()),patch.object(launch,'ensure_servers'),redirect_stdout(io.StringIO()):
            self.assertEqual(launch.main(),0)
        self.assertEqual(len(self.trace(events)),16)  # Resume never reruns finished workers.

    def test_incomplete_circle_shard_prevents_baseline_launch(self):
        events=self.base/'events.jsonl'
        with patch.dict('os.environ',{'FAKE_EVENTS':str(events),'FAKE_EARLY_STOP':'1'}),patch.object(sys,'argv',self.argv()),patch.object(launch,'ensure_servers'),redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'next phase will NOT start'):
                launch.main()
        trace=self.trace(events)
        self.assertTrue(all(r['mode']=='circle' for r in trace))
        self.assertFalse((self.args.control_root/'launcher.lock').exists())


if __name__=='__main__':
    unittest.main()
