import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tool_loop_runtime.configuration import load_config
from tool_loop_runtime.dataset import prepare_manifest
from tool_loop_runtime.providers import validate_homogeneous
from tool_loop_runtime import stages, baseline, transport

ROOT=Path(__file__).resolve().parents[1]


class PortableTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'offline_checks').mkdir(exist_ok=True)

    def test_same_checkpoint_for_every_role_and_baseline_without_provider_flags(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'offline_checks') as d:
            p=Path(d);image=p/'x.jpg';image.write_bytes(b'fixture')
            profile=p/'m.json';profile.write_text(json.dumps({'model_id':'M','endpoint':'http://127.0.0.1:8000/v1','require_key':False}))
            with patch.dict('os.environ',{'MODEL_ENDPOINT':'http://127.0.0.1:8000/v1','SERVED_MODEL_NAME':'M'}):
                cfg=load_config(profile,'B')
            row={'id':'x','question':'ORIGINAL QUESTION','image':str(image),'image_sha256':transport.digest(image.read_bytes())}
            for role in ['understand','select','solve','check']:
                body=stages.payload(role,row,{'task':row['question']},cfg,'role prompt')
                self.assertEqual(body['model'],'M');self.assertNotIn('thinking',body);self.assertNotIn('reasoning_effort',body)
                self.assertEqual(transport.api_binding(cfg,role)[0],'http://127.0.0.1:8000/v1/chat/completions')
                self.assertEqual(isinstance(body['messages'][1]['content'],list),role!='solve')
            body=baseline.body_for(row,cfg)
            self.assertEqual(body['messages'][0]['content'][1]['text'],'ORIGINAL QUESTION')
            self.assertEqual(body['max_tokens'],cfg['max_tokens']['solve'])
            cfg['models']['check']='OTHER'
            with self.assertRaisesRegex(ValueError,'same model'):validate_homogeneous(cfg)

    def test_eval_manifest_preserves_ids_and_never_includes_gold(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'offline_checks') as d:
            p=Path(d);images=p/'images';images.mkdir();(images/'TabFact_a.html.csv.jpg').write_bytes(b'fixture')
            dataset=p/'official.json';dataset.write_text(json.dumps([{'item_id':'one','image_id':'TabFact_a.html.csv','input':'Q',
                'output':'SECRET','answer_list':['SECRET'],'task_type':'TQA','dataset_name':'WTQ'},
                {'item_id':'two','image_id':'missing','input':'Q2','output':'SECRET','task_type':'T2T','dataset_name':'ToTTo'}]))
            out=p/'manifest.json';meta=prepare_manifest(dataset,images,out,exclude=['T2T'])
            self.assertEqual(meta['count'],1);self.assertNotIn('SECRET',out.read_text());self.assertEqual(json.loads(out.read_text())[0]['id'],'one')
            with self.assertRaisesRegex(ValueError,'no image'):prepare_manifest(dataset,images,p/'all.json')
            self.assertFalse((p/'all.json').exists())


if __name__=='__main__':unittest.main()
