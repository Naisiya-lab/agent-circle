"""API proxy policy, role dispatch and retries; no external requests or real keys."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tool_loop_runtime import transport as tr

BASE = Path(__file__).resolve().parents[1]


class Response(io.BytesIO):
    headers = {'Content-Type':'application/json'}


def response():
    return Response(b'{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}')


class NetworkTests(unittest.TestCase):
    def setUp(self):
        parent = (BASE/'offline_checks').resolve()
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='direct_network_', dir=parent)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(parent)
        self.addCleanup(self.temp.cleanup)
        self.cfg = tr.read_json(BASE/'configs/legacy_mixed_A.json')

    def test_direct_ignores_auto_proxy_without_changing_global_state(self):
        global_opener = tr.urllib.request._opener
        with patch.dict(os.environ, {'HTTPS_PROXY':'http://proxy.invalid:8080'}):
            before = dict(os.environ)
            with patch.object(tr.urllib.request,'getproxies',side_effect=AssertionError('Proxy lookup forbidden')):
                opener = tr.build_api_opener(self.cfg)
            self.assertEqual(dict(os.environ), before)
        self.assertIs(tr.urllib.request._opener, global_opener)
        self.assertFalse(any(getattr(h,'proxies',{}) for h in opener.handlers))

    def test_system_mode_is_explicit_opt_in(self):
        with patch.object(tr.urllib.request,'getproxies',return_value={'https':'http://proxy.invalid:8080'}) as proxies:
            opener = tr.build_api_opener(dict(self.cfg,network={'proxy_mode':'system'}))
        proxies.assert_called_once()
        self.assertTrue(any(getattr(h,'proxies',{}).get('https')=='http://proxy.invalid:8080' for h in opener.handlers))
        self.assertEqual(tr.proxy_mode({}), 'direct')
        for network in ([], {'proxy_mode':'typo'}):
            with self.assertRaises(ValueError):
                tr.proxy_mode({'network':network})

    def test_both_providers_use_local_opener_and_unchanged_model_payload(self):
        bindings = {tr.api_binding(self.cfg,r):'offline-key' for r in ('understand','select','solve','check')}
        received = []
        def opened(request,timeout):
            received.append((request.full_url,json.loads(request.data)))
            return response()
        with patch.object(tr.urllib.request.OpenerDirector,'open',side_effect=opened), \
             patch.object(tr.urllib.request,'urlopen',side_effect=AssertionError('Global opener forbidden')):
            client = tr.RoleApiClient(self.cfg,bindings,self.root)
            for role in ('understand','select','solve','check'):
                body = {'stream':False,'model':self.cfg['models'][role],'messages':[]}
                result = client.call(body,'test',role,role=role)
                self.assertEqual(received[-1],(tr.api_binding(self.cfg,role)[0],body))
                self.assertEqual(result['_transport']['proxy_mode'],'direct')
        self.assertEqual(len(client.clients),2)

    def test_retry_keeps_direct_opener_and_records_timing(self):
        cfg = dict(self.cfg,retry_base_seconds=0,cooldown_seconds=0)
        with patch.object(tr.urllib.request.OpenerDirector,'open',side_effect=[OSError('offline disconnect'),response()]) as opened:
            result = tr.ApiClient(cfg,'offline-key',self.root).call({'stream':False},'test','stage')
        self.assertEqual(opened.call_count,2)
        self.assertEqual(result['_transport']['successful_attempt'],2)
        self.assertGreaterEqual(result['_transport']['response_headers_seconds'],0)
        self.assertNotIn('offline-key',(self.root/'run.log').read_text())

    def test_fatal_auth_does_not_retry_or_leak_key(self):
        failure = tr.urllib.error.HTTPError(self.cfg['endpoint'],401,'auth',{},io.BytesIO(b'bad offline-key'))
        self.addCleanup(failure.close)
        with patch.object(tr.urllib.request.OpenerDirector,'open',side_effect=failure) as opened:
            with self.assertRaises(tr.ApiFailure) as result:
                tr.ApiClient(self.cfg,'offline-key',self.root).call({'stream':False},'test','stage')
        self.assertTrue(result.exception.fatal)
        self.assertEqual(opened.call_count,1)
        self.assertNotIn('offline-key',(self.root/'api_attempts.jsonl').read_text())


if __name__ == '__main__':
    unittest.main()
