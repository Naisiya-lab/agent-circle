"""Bounded, read-only collection of four stalled agent-loop samples.

Does not import the workflow, contact model APIs, replay stages or stop workers.
Only the newly created .tar.gz is written. Large logs are read from their tails;
large sample directories are scanned by filename without loading all records.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import heapq
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

MIB = 1024 * 1024
STAGE = re.compile(r"r(\d+)_(01_understand|02_select|04_solve|05_check)(?:_tool_repair_(\d+))?(?:_after_read_(\d+))?\.json$")
PROGRESS = re.compile(r"\|\s+([A-Za-z0-9_.-]+)\s+(r\d+_\S+)")
RUN_LINE = re.compile(r"\b([A-Za-z0-9_.-]+)\s+(r\d+_\S+)\s+(?:requesting|attempt=|receiving|saved|resume)")
SECRET_KEYS = {'api_key', 'apikey', 'authorization', 'password', 'secret', 'access_token', 'refresh_token'}


def scrub(value):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if k.lower() in SECRET_KEYS else scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r'data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+', '[IMAGE_DATA_OMITTED]', value)
        value = re.sub(r'(?i)(Bearer\s+)\S+', r'\1[REDACTED]', value)
        value = re.sub(r'(?i)(api[_-]?key[\"\s:=]+)[^\s\",}]+', r'\1[REDACTED]', value)
    return value


def tail(path, limit):
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size-limit))
        data = stream.read(limit)
    return data.decode('utf-8', errors='replace').replace('\r', '\n'), size


def discover(console, run_log, limit):
    for path, pattern in ((console, PROGRESS), (run_log, RUN_LINE)):
        if path.is_file():
            text, _ = tail(path, limit)
            matches = list(pattern.finditer(text))
            if matches:
                return matches[-1].group(1)
    return None


def stage_key(name):
    match = STAGE.fullmatch(name)
    if not match:
        return None
    return (int(match[1]), int(match[2][:2]), int(match[3] or 0), int(match[4] or 0))


def recent_stages(folder, count):
    counts = Counter()
    def candidates():
        with os.scandir(folder) as entries:
            for entry in entries:
                counts['directory_entries'] += 1
                key = stage_key(entry.name)
                if key is not None and entry.is_file(follow_symlinks=False):
                    counts['model_stage_files'] += 1
                    yield key, entry.name
    selected = [name for _, name in heapq.nlargest(count, candidates())]
    return selected, dict(counts)


class Bundle:
    def __init__(self, output, total_limit, file_limit, tail_limit):
        self.output, self.total_limit = output, total_limit
        self.file_limit, self.tail_limit = file_limit, tail_limit
        self.total, self.entries, self.skipped, self.names = 0, [], [], set()
        self.section_end = None
        self.archive = tarfile.open(output, mode='x:gz')

    def add(self, name, data, source=None, **metadata):
        if name in self.names:
            return False
        ceiling = self.total_limit if name == 'manifest.json' else self.total_limit - 256 * 1024
        if self.section_end is not None and name != 'manifest.json':
            ceiling = min(ceiling, self.section_end)
        if self.total + len(data) > ceiling:
            self.skipped.append({'path': name, 'reason': 'total_size_budget', 'bytes': len(data)})
            return False
        info = tarfile.TarInfo(name)
        info.size, info.mode = len(data), 0o644
        self.archive.addfile(info, io.BytesIO(data))
        self.names.add(name)
        self.total += len(data)
        self.entries.append({'path': name, 'bytes': len(data), 'source': str(source) if source else None, **metadata})
        return True

    def obj(self, name, value, **metadata):
        data = (json.dumps(scrub(value), ensure_ascii=False, indent=2)+'\n').encode('utf-8')
        return self.add(name, data, **metadata)

    def file(self, path, name, *, log=False, image=False):
        if not path.is_file():
            self.skipped.append({'path': name, 'reason': 'missing'})
            return
        try:
            size = path.stat().st_size
            if log:
                text, size = tail(path, self.tail_limit)
                self.add(name+'.tail.txt', scrub(text).encode('utf-8'), path, original_bytes=size, tail_only=True)
            elif size > self.file_limit:
                self.skipped.append({'path': name, 'reason': 'file_size_budget', 'bytes': size})
                if not image:
                    text, _ = tail(path, min(16384, self.tail_limit))
                    self.add(name+'.excerpt.txt', scrub(text).encode('utf-8'), path, original_bytes=size, tail_only=True)
            else:
                # Read at most the advertised limit even if a live file grows.
                with path.open('rb') as stream:
                    raw = stream.read(self.file_limit+1)
                if len(raw) > self.file_limit:
                    self.skipped.append({'path': name, 'reason': 'file_grew_past_limit'})
                elif image:
                    self.add(name, raw, path, sha256=hashlib.sha256(raw).hexdigest())
                else:
                    text = raw.decode('utf-8-sig', errors='replace')
                    if path.suffix == '.json':
                        try:
                            self.obj(name, json.loads(text), source=str(path))
                        except ValueError:
                            self.add(name+'.unparsed.txt', scrub(text).encode('utf-8'), path)
                    else:
                        self.add(name, scrub(text).encode('utf-8'), path)
        except OSError as error:
            self.skipped.append({'path': name, 'reason': type(error).__name__, 'message': str(error)})

    def finish(self, summary):
        self.section_end = None
        summary.update(files=self.entries, skipped=self.skipped, raw_bytes=self.total,
                       total_limit_bytes=self.total_limit,
                       note='Selected excerpts only; absence from archive does not mean absence from run. No API calls or changes to experiment files.')
        if not self.obj('manifest.json', summary):
            self.archive.close()
            raise ValueError('Archive manifest exceeded size budget')
        self.archive.close()


def selected_row(path, sid):
    # Read the small frozen shard manifest, but export only the one selected row.
    if not path.is_file() or path.stat().st_size > 64*MIB:
        return None
    with path.open(encoding='utf-8-sig') as stream:
        return next((row for row in json.load(stream) if row.get('id') == sid), None)


def collect(args):
    project = args.project.resolve()
    runs = args.runs_root.resolve() if args.runs_root else project/'runs'
    control = args.control_root.resolve() if args.control_root else runs/args.experiment
    output = args.output.resolve() if args.output else project/('stall_diagnostics_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.tar.gz')
    override = {}
    for item in args.sample:
        shard, sid = item.split('=', 1)
        if shard not in ('s0', 's1', 's2', 's3') or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', sid):
            raise ValueError('Use --sample s0=sample_id, with a plain sample folder name')
        override[int(shard[1])] = sid
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = Bundle(output, args.max_total_mb*MIB, args.max_file_mb*MIB, args.log_tail_kb*1024)
    summary = {'created_at': datetime.now(timezone.utc).isoformat(), 'experiment': args.experiment, 'shards': []}
    try:
        for filename in ('plan.json', 'services.json', 'circle_phase.json', 'launcher.lock'):
            bundle.file(control/filename, 'control/'+filename)
        for index in range(4):
            # Share remaining capacity across shards; one large table must not
            # consume the whole archive before the other stalled samples.
            bundle.section_end = bundle.total + max(0, (bundle.total_limit-bundle.total-512*1024)//(4-index))
            run = runs/(args.experiment+f'_circle_s{index}')
            console = control/f'circle_s{index}.console.log'
            sid = override.get(index) or discover(console, run/'run.log', bundle.tail_limit)
            entry = {'shard': index, 'sample_id': sid}
            summary['shards'].append(entry)
            print(f's{index}: collecting {sid or "sample not found"}', flush=True)
            for path in (console, control/f'server_gpu{index}.log', run/'run.log', run/'api_attempts.jsonl'):
                bundle.file(path, f's{index}/logs/'+path.name, log=True)
            for filename in ('config.json', 'summary.json', 'prompts_snapshot.json', 'run.lock'):
                bundle.file(run/filename, f's{index}/'+filename)
            for filename in ('pipeline.py', 'stages.py', 'communication.py', 'protocol.py', 'transport.py', 'recovery.py'):
                bundle.file(run/'code_snapshot/tool_loop_runtime'/filename, f's{index}/code_snapshot/'+filename)
            if not sid:
                entry['issue'] = 'Could not infer current sample; rerun with --sample sN=ID'
                continue
            folder = run/'samples'/sid
            if not folder.is_dir():
                entry['issue'] = 'Sample directory not found'
                continue
            prefix = f's{index}/sample_{sid}/'
            latest, counts = recent_stages(folder, args.recent)
            entry.update(counts, recent_model_stages=latest)
            # Latest exchanges have priority over supplementary old-round files.
            for name in latest:
                bundle.file(folder/name, prefix+name)
            for filename in ('communication_state.json', 'loop_state.json', 'result.json', 'error.json', 'tool_retry_plan.json'):
                bundle.file(folder/filename, prefix+filename)
            for filename in ('tool_events.jsonl', 'recovery_history.jsonl'):
                bundle.file(folder/filename, prefix+filename, log=True)
            for rnd in ('r00', 'r01'):
                for name in ('01_understand', '00t_index', 'table_index', '02_select', '02_select_after_read_001', '02_select_after_read_002',
                             '03_extract', '04_solve', 'candidate', '05_check', 'check_routing'):
                    path = folder/f'{rnd}_{name}.json'
                    if path.is_file():
                        bundle.file(path, prefix+path.name)
            # Selected stage inputs contain the previous request + tool results;
            # do not traverse attempts/, streams/, tools/, or read huge JSONL files.
            row = selected_row(run/'selected.json', sid)
            if row:
                bundle.obj(prefix+'sample.json', row)
                if args.images and row.get('image'):
                    image = Path(row['image'])
                    if not image.is_absolute():
                        image = project/image
                    bundle.file(image, prefix+'image'+image.suffix, image=True)
        bundle.section_end = None
        for name in ('pipeline.py', 'stages.py', 'communication.py', 'protocol.py'):
            bundle.file(project/'tool_loop_runtime'/name, 'current_code/'+name)
        versions = {}
        for name in ('vllm', 'torch', 'transformers'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = 'not installed in collector environment'
        bundle.obj('environment.json', {'python': sys.version, 'packages': versions})
        for label, command in (
            ('git_head', ['git', '-C', str(project), 'rev-parse', 'HEAD']),
            ('gpu', ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv']),
            ('disk', ['df', '-h', str(runs)]),
            ('inodes', ['df', '-i', str(runs)]),
        ):
            try:
                result = subprocess.run(command, capture_output=True, timeout=10, cwd=project)
                bundle.add(label+'.txt', scrub((result.stdout+result.stderr)[:16384].decode('utf-8', errors='replace')).encode('utf-8'))
            except (OSError, subprocess.TimeoutExpired) as error:
                summary[label] = str(error)
    finally:
        bundle.finish(summary)
    print(f'Archive: {output}\nCompressed size: {output.stat().st_size/MIB:.2f} MiB\nCollected raw size: {bundle.total/MIB:.2f} MiB', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--experiment', default='qwen25_7b_full4gpu_v2')
    parser.add_argument('--runs-root', type=Path)
    parser.add_argument('--control-root', type=Path)
    parser.add_argument('--sample', action='append', default=[], help='Override autodetection, e.g. s0=TABMWP_2444')
    parser.add_argument('--recent', type=int, default=8)
    parser.add_argument('--log-tail-kb', type=int, default=128)
    parser.add_argument('--max-file-mb', type=int, default=2)
    parser.add_argument('--max-total-mb', type=int, default=32)
    parser.add_argument('--images', action='store_true', help='Include only the four current sample images, subject to file/total limits')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', args.experiment):
        parser.error('Use a plain experiment name')
    if args.recent < 1 or args.log_tail_kb < 1 or args.max_file_mb < 1 or args.max_total_mb < 1:
        parser.error('Collection limits must be positive')
    collect(args)


if __name__ == '__main__':
    main()
