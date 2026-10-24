"""Local fake HTTP provider; no external API, credentials, model, or GPU required."""
from contextlib import redirect_stdout
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from tool_loop_runtime import baseline, runner, transport
from tool_loop_runtime.configuration import load_config

ROOT=Path(__file__).resolve().parents[1]


class HttpSmoke(unittest.TestCase):
    def test_direct_and_circle_use_one_model_through_real_local_http(self):
        (ROOT/'offline_checks').mkdir(exist_ok=True)
        received=[]
        replies=['8',{'schema_version':'table_html_v3','tables':[{'id':'t1','html':'<table><tr><td>A</td><td>8</td></tr></table>'}],
                       'coverage':'complete','uncertainties':[]},
                 {'action':'extract','regions':[{'id':'s1','source_table_id':'t1','rows':[1,1],'columns':[1,2]}],'uncertainties':[]},
                 {'status':'answered','answer':'8','review_request':''},
                 {'action':'judge','status':'passed','issues':[]}]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                obj=json.loads(self.rfile.read(int(self.headers['Content-Length'])));received.append(obj)
                value=replies.pop(0);content=value if isinstance(value,str) else json.dumps(value)
                data=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':content}}],
                                 'usage':{'prompt_tokens':2,'completion_tokens':1,'total_tokens':3}}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        server=HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory(dir=ROOT/'offline_checks') as d:
                p=Path(d);image=p/'sample.jpg';image.write_bytes(b'fixture')
                source=p/'manifest.json';transport.save_json(source,[{'id':'one','question':'What is A?','image':str(image),
                    'image_sha256':transport.digest(image.read_bytes()),'task_type':'TQA','dataset_name':'fixture','reference':'DO_NOT_SEND'}])
                profile=p/'model.json';transport.save_json(profile,{'model_id':'M','require_key':False})
                with patch.dict('os.environ',{'MODEL_ENDPOINT':f'http://127.0.0.1:{server.server_port}/v1','SERVED_MODEL_NAME':'M'}):cfg=load_config(profile,'B')
                cfg.update(stream=False,timeout_seconds=3,transport_retries=0,sample_retry_rounds=0,request_deadline_seconds=5)
                def path(name):return p/name
                with patch.object(runner,'run_root',side_effect=path),redirect_stdout(io.StringIO()):
                    cfg['experiment']['mode']='baseline';self.assertEqual(baseline.run(deepcopy(cfg),'direct',source,1),0,(p/'direct/samples/one/error.json').read_text() if (p/'direct/samples/one/error.json').exists() else '')
                    cfg['experiment']['mode']='circle';self.assertEqual(runner.run(deepcopy(cfg),'circle',source,1),0,(p/'circle/samples/one/error.json').read_text() if (p/'circle/samples/one/error.json').exists() else '')
                self.assertEqual(transport.read_json(p/'direct/results.json')[0]['answer'],'8')
                self.assertEqual(transport.read_json(p/'circle/results.json')[0]['answer'],'8')
                self.assertEqual(len(received),5);self.assertTrue(all(b['model']=='M' for b in received))
                self.assertTrue(all('DO_NOT_SEND' not in json.dumps(b) for b in received))
                self.assertTrue(all('thinking' not in b and 'reasoning_effort' not in b for b in received))
        finally:server.shutdown();server.server_close();thread.join(timeout=2)


if __name__=='__main__':unittest.main()
