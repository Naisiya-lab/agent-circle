"""Collector checks use small synthetic runs and never call a model service."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('stall_collector', ROOT/'scripts/collect_stall_diagnostics.py')
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class StallDiagnosticsTests(unittest.TestCase):
    def test_numeric_latest_stage_selection_does_not_select_query_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in ('r01_02_select_after_read_999.json', 'r01_02_select_after_read_1000.json',
                         'r01_02_select_after_read_173815.json', 'r01_02_select_after_read_173816_query_001_001.json',
                         'r00_05_check.json'):
                (folder/name).write_text('{}')
            names, counts = collector.recent_stages(folder, 2)
            self.assertEqual(names, ['r01_02_select_after_read_173815.json', 'r01_02_select_after_read_1000.json'])
            self.assertEqual(counts['model_stage_files'], 4)

    def test_progress_detection_uses_latest_tail_and_run_log_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            console, log = root/'console.log', root/'run.log'
            console.write_text('x'*3000+'\r[session 1/2] elapsed 1 min | TABMWP_2444 r01_02_select_after_read_35634 request attempt 1\n')
            self.assertEqual(collector.discover(console, log, 512), 'TABMWP_2444')
            console.write_text('Interrupted')
            log.write_text('2026-11-05 TABMWP_33750 r01_02_select_after_read_173815 requesting format_retry=0\n')
            self.assertEqual(collector.discover(console, log, 512), 'TABMWP_33750')

    def test_archive_is_bounded_and_collects_all_four_samples_without_recursion(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            control = project/'runs/exp'
            control.mkdir(parents=True)
            originals = {}
            for index in range(4):
                sid = f'TABMWP_{index}'
                run = project/f'runs/exp_circle_s{index}'
                sample = run/'samples'/sid
                sample.mkdir(parents=True)
                image = project/f'{index}.png'
                image.write_bytes(b'fixture image')
                (run/'selected.json').write_text(json.dumps([{'id':sid, 'question':'q', 'image':str(image)}, {'id':'unrelated', 'question':'do not include'}]))
                (run/'config.json').write_text(json.dumps({'api_key':'secret-value', 'max_tokens':4096}))
                (control/f'circle_s{index}.console.log').write_text('IGNORE_OLD_LOG\n'*10000+f'| {sid} r01_02_select_after_read_10000 request attempt 1\n')
                (sample/'r01_02_select_after_read_10000.json').write_text(json.dumps({'output':{'action':'inspect'}, 'input':{'tool_results':[{'ok':True}]}}))
                (sample/'attempts').mkdir()
                (sample/'attempts/huge.json').write_text('UNRELATED_ATTEMPT'*10000)
                originals[str(sample/'r01_02_select_after_read_10000.json')] = (sample/'r01_02_select_after_read_10000.json').read_bytes()
            args = SimpleNamespace(project=project, runs_root=None, control_root=None, experiment='exp', sample=[],
                                   output=project/'out.tar.gz', recent=2, log_tail_kb=1, max_file_mb=1, max_total_mb=2, images=True)
            with patch.object(collector.subprocess, 'run', return_value=subprocess.CompletedProcess([],0,b'fixture',b'')), redirect_stdout(io.StringIO()):
                archive = collector.collect(args)
            with tarfile.open(archive) as package:
                names = package.getnames()
                self.assertFalse(any('/attempts/' in name or '/streams/' in name or '/selected.json' in name for name in names))
                self.assertLessEqual(sum(m.size for m in package.getmembers()), 2*collector.MIB)
                for index in range(4):
                    self.assertIn(f's{index}/sample_TABMWP_{index}/sample.json', names)
                    self.assertIn(f's{index}/sample_TABMWP_{index}/image.png', names)
                    config = json.load(package.extractfile(f's{index}/config.json'))
                    self.assertEqual(config['api_key'], '[REDACTED]')
                    self.assertEqual(config['max_tokens'], 4096)
                    self.assertLessEqual(package.getmember(f's{index}/logs/circle_s{index}.console.log.tail.txt').size, 1024)
                manifest = json.load(package.extractfile('manifest.json'))
                self.assertEqual(len(manifest['shards']), 4)
            for path, contents in originals.items():
                self.assertEqual(Path(path).read_bytes(), contents)

    def test_manifest_survives_size_budget_and_existing_archive_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'out.tar.gz'
            bundle = collector.Bundle(output, collector.MIB, collector.MIB, 128)
            self.assertTrue(bundle.add('large.txt', b'x'*(collector.MIB-256*1024)))
            self.assertFalse(bundle.add('excess.txt', b'y'))
            bundle.finish({'test':True})
            with tarfile.open(output) as package:
                self.assertIn('manifest.json', package.getnames())
                self.assertNotIn('excess.txt', package.getnames())
            with self.assertRaises(FileExistsError):
                collector.Bundle(output, collector.MIB, collector.MIB, 128)


if __name__ == '__main__':
    unittest.main()
