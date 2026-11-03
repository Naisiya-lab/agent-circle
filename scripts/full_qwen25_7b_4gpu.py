"""Run existing agent_circle.py unchanged: four circle shards, barrier, four baselines."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def split_manifest(source, folder):
    """Interleave the frozen manifest, retaining every field, with no answer inspection."""
    source, folder = Path(source), Path(folder)
    rows = read(source)
    require(isinstance(rows, list) and len(rows) >= 4, 'Manifest must contain at least four samples')
    ids = [row['id'] for row in rows]
    require(len(set(ids)) == len(ids), 'Source has duplicate sample IDs')
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(4):
        part = rows[index::4]
        path = folder / f'mmtab_eval_s{index}of4.json'
        if path.exists():
            require(read(path) == part, f'Existing shard differs: {path}; use a new experiment name')
        else:
            save(path, part)
        paths.append(path.resolve())
    # Also check the persisted data, rather than just the in-memory partition.
    partitioned = [row['id'] for path in paths for row in read(path)]
    require(Counter(partitioned) == Counter(ids), 'Shard coverage or uniqueness check failed')
    return rows, paths


def run_name(args, mode, index):
    return f'{args.experiment}_{mode}_s{index}'


def scan_run(root, rows, mode):
    """Job completion only. Never judge, modify or route an agent's output."""
    root = Path(root)
    result_name = 'result.json' if mode == 'circle' else 'direct_result.json'
    counts = dict(selected=len(rows), terminal_records=0, nonretryable_errors=0,
                  retryable_errors=0, pending=0, service_failures=0)
    for row in rows:
        folder = root / 'samples' / row['id']
        if (folder/'error.json').exists() and read(folder/'error.json').get('fatal', False):
            counts['service_failures'] += 1
        result_path=folder/result_name
        retry_plan=folder/'tool_retry_plan.json'
        if result_path.exists() and retry_plan.exists() and read(retry_plan).get('state')=='in_progress' and read(retry_plan).get('from_result_sha256')==sha(result_path):
            counts['pending'] += 1
        elif result_path.is_file():
            counts['terminal_records'] += 1
        elif (folder / 'error.json').is_file():
            error = read(folder / 'error.json')
            key = 'retryable_errors' if error.get('retryable', False) else 'nonretryable_errors'
            counts[key] += 1
        else:
            counts['pending'] += 1
    # A recoverable failure remains eligible for a later retry, but is still
    # an attempted item in this pass. Never drop it from exports/denominators.
    counts['all_processed'] = counts['pending'] == 0 and counts['service_failures'] == 0
    summary = root / 'summary.json'
    if summary.exists():
        counts['verification_status'] = read(summary).get('verification_status', {})
    return counts


def phase_status(args, mode, shard_paths):
    return [dict(shard=i, run=run_name(args, mode, i),
                 **scan_run(args.runs_root / run_name(args, mode, i), read(path), mode))
            for i, path in enumerate(shard_paths)]


def phase_complete(statuses, codes=None):
    return all(s['all_processed'] for s in statuses) and (
        codes is None or all(code in (0, 1) for code in codes))


def server_probe(port, name, weights, context, timeout=3):
    url = f'http://127.0.0.1:{port}/v1/models'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        raise ValueError(f'Port {port} returned HTTP {exc.code}; check the existing service') from exc
    except urllib.error.URLError:
        return None
    cards = [item for item in body.get('data', []) if item.get('id') == name]
    require(cards, f'Port {port} serves another model, expected {name}; no process was stopped')
    card = cards[0]
    root = card.get('root')
    if root and Path(root).is_absolute():
        require(Path(root).resolve() == weights.resolve(),
                f'Port {port} has different model weights: {root}')
    if card.get('max_model_len') is not None:
        require(int(card['max_model_len']) == context,
                f'Port {port} uses a different context length; expected {context}')
    return card


def model_files(weights):
    result = {}
    for path in sorted(weights.iterdir()):
        if path.is_file() and (path.suffix == '.json' or path.suffix == '.safetensors'):
            info = path.stat()
            result[path.name] = {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns}
            if path.suffix == '.json':
                result[path.name]['sha256'] = sha(path)
    return result


def prepare(args):
    require((args.project / 'agent_circle.py').is_file(), 'Project entry point not found')
    require(args.weights.is_dir(), f'Model directory not found: {args.weights}')
    require(len(args.gpus) == 4 and len(set(args.gpus)) == 4, 'Specify four distinct GPU indices')
    require(1 <= args.first_port <= 65532, 'Four consecutive ports must fit in 1..65535')
    require(0 < args.memory_utilization <= 1, 'Invalid GPU memory utilization')
    require(args.experiment and all(c.isalnum() or c in '_.-' for c in args.experiment)
            and args.experiment not in ('.', '..'), 'Invalid experiment name')
    rows, shards = split_manifest(args.source, args.shards_root)
    files = [args.project / 'agent_circle.py', args.project / 'configs/runtime.json',
             args.project / 'configs/models/qwen25_vl_7b.json']
    files += sorted((args.project / 'tool_loop_runtime').rglob('*.py'))
    files += sorted((args.project / 'tool_loop_runtime/prompts').rglob('*.txt'))
    files += sorted((args.project / 'table_tools').glob('*.py'))
    files += [p for p in (args.project / 'scripts/full_qwen25_7b_4gpu.py',
                          args.project / 'scripts/full_qwen25_7b_4gpu.sh',
                          args.project / 'scripts/full_qwen25_7b_4gpu.env.example.sh') if p.is_file()]
    plan = {'source': str(args.source), 'source_sha256': sha(args.source), 'count': len(rows),
            'shards': [{'path': str(p), 'sha256': sha(p), 'count': len(read(p))} for p in shards],
            'phase_order': ['circle', 'baseline'], 'route': 'B',
            'weights': str(args.weights), 'weight_file_identity': model_files(args.weights),
            'served_model_name': args.model_name, 'gpus': args.gpus,
            'ports': [args.first_port + i for i in range(4)],
            'context': args.context, 'memory_utilization': args.memory_utilization,
            'max_num_seqs': args.max_num_seqs, 'runs_root': str(args.runs_root),
            'python': sys.executable, 'python_version': sys.version,
            'environment': {k: os.environ.get(k) for k in
                            ('MODEL_REVISION', 'GENERATION_TEMPERATURE')},
            'implementation_sha256': {p.relative_to(args.project).as_posix(): sha(p) for p in files},
            'note': 'Execution launcher only; core, prompts, output limits and repair rules are unchanged.'}
    path = args.control_root / 'plan.json'
    if path.exists():
        require(read(path) == plan, 'Frozen launch plan changed; restore settings or use a new experiment name')
    else:
        save(path, plan)
    print(f'Manifest: {len(rows)} samples; shards: {[len(read(p)) for p in shards]}', flush=True)
    return rows, shards


def ensure_servers(args):
    records, waiting = [], []
    try:
        for index, gpu in enumerate(args.gpus):
            port = args.first_port + index
            card = server_probe(port, args.model_name, args.weights, args.context)
            record = {'gpu': gpu, 'port': port, 'reused': card is not None, 'started_at': stamp()}
            if card is not None:
                record['model_card'] = card
                print(f'GPU {gpu} / port {port}: reuse existing matching service '
                      '(GPU placement follows its original launch command)', flush=True)
            else:
                pidfile = args.control_root / f'server_gpu{gpu}.json'
                old = read(pidfile) if pidfile.exists() else None
                alive = False
                if old:
                    try:
                        os.kill(old['pid'], 0)
                        alive = True
                    except ProcessLookupError:
                        pass
                if alive:
                    record['pid'] = old['pid']
                    print(f'GPU {gpu} / port {port}: waiting for previously launched PID {old["pid"]}', flush=True)
                    waiting.append((record, None))
                else:
                    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
                               '--model', str(args.weights), '--served-model-name', args.model_name,
                               '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
                               '--tensor-parallel-size', '1', '--gpu-memory-utilization', str(args.memory_utilization),
                               '--max-model-len', str(args.context), '--max-num-seqs', str(args.max_num_seqs),
                               '--generation-config', 'vllm', '--trust-remote-code']
                    if os.environ.get('MODEL_REVISION'):
                        command += ['--revision', os.environ['MODEL_REVISION']]
                    log = args.control_root / f'server_gpu{gpu}.log'
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                    with log.open('ab') as output:
                        process = subprocess.Popen(command, cwd=args.project, env=env, stdin=subprocess.DEVNULL,
                                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    record.update(pid=process.pid, log=str(log))
                    save(pidfile, record)
                    waiting.append((record, process))
                    print(f'GPU {gpu} / port {port}: starting PID {process.pid}; {log}', flush=True)
            records.append(record)
        save(args.control_root / 'services.json', records)
        deadline, next_message = time.monotonic() + args.start_timeout, 0
        while waiting:
            remaining = []
            for record, process in waiting:
                if process is not None and process.poll() is not None:
                    raise ValueError(f'Service on GPU {record["gpu"]} exited; see server_gpu{record["gpu"]}.log')
                card = server_probe(record['port'], args.model_name, args.weights, args.context)
                if card is None:
                    remaining.append((record, process))
                else:
                    record['model_card'] = card
                    record['ready_at'] = stamp()
                    print(f'GPU {record["gpu"]} / port {record["port"]}: ready', flush=True)
            waiting = remaining
            require(not waiting or time.monotonic() < deadline, 'Service startup timed out; inspect server logs')
            if waiting:
                if time.monotonic() >= next_message:
                    print('Waiting for ports: ' + ', '.join(str(r['port']) for r, _ in waiting), flush=True)
                    next_message = time.monotonic() + 20
                time.sleep(2)
        save(args.control_root / 'services.json', records)
    except BaseException:
        save(args.control_root / 'services.json', records)
        raise


def stop_workers(processes):
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def export_phase(args, mode, rows, shards):
    folder = args.control_root / 'exports'
    folder.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for index in range(4):
        out = folder / f'{mode}_s{index}.jsonl'
        subprocess.run([sys.executable, str(args.project / 'agent_circle.py'), 'export',
                        '--run', run_name(args, mode, index), '--output', str(out)],
                       cwd=args.project, env=dict(os.environ, AGENT_CIRCLE_RUNS=str(args.runs_root)), check=True)
        expected = {row['id'] for row in read(shards[index])}
        actual = set()
        for line in out.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            key = item['question_id']
            require(key not in predictions, 'Duplicate exported prediction ID')
            predictions[key] = item
            actual.add(key)
        require(actual == expected, 'Exported IDs do not match the frozen shard')
    require(set(predictions) == {row['id'] for row in rows}, 'Merged export does not cover the source')
    output = folder / f'{mode}_all.jsonl'
    output.write_text(''.join(json.dumps(predictions[row['id']], ensure_ascii=False) + '\n'
                              for row in rows), encoding='utf-8')
    print(f'{mode}: exported {len(rows)} predictions, including failed/empty items: {output}', flush=True)


def execute_phase(args, mode, rows, shards):
    current = phase_status(args, mode, shards)
    if phase_complete(current) and not getattr(args, "retry_errors", False):
        print(f'{mode}: all samples already processed; reusing saved results', flush=True)
        export_phase(args, mode, rows, shards)
        return
    processes, started = [], time.monotonic()
    record = {'phase': mode, 'started_at': stamp(), 'workers': []}
    try:
        for index, path in enumerate(shards):
            port = args.first_port + index
            name = run_name(args, mode, index)
            env = dict(os.environ, MODEL_ENDPOINT=f'http://127.0.0.1:{port}/v1',
                       SERVED_MODEL_NAME=args.model_name, API_PROXY_MODE='direct',
                       AGENT_CIRCLE_RUNS=str(args.runs_root), PYTHONUNBUFFERED='1')
            command = [sys.executable, '-u', str(args.project / 'agent_circle.py'), 'run',
                       '--mode', mode, '--route', 'B',
                       '--model-profile', str(args.project / 'configs/models/qwen25_vl_7b.json'),
                       '--source', str(path), '--run', name]
            if getattr(args, 'retry_errors', False):
                command += ['--retry-errors']
            log = args.control_root / f'{mode}_s{index}.console.log'
            with log.open('ab') as output:
                process = subprocess.Popen(command, cwd=args.project, env=env, stdin=subprocess.DEVNULL,
                                           stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            record['workers'].append({'shard': index, 'pid': process.pid, 'port': port, 'run': name, 'log': str(log)})
            print(f'{mode} shard {index}: PID {process.pid}, port {port}, log {log}', flush=True)
        save(args.control_root / f'{mode}_phase.json', record)
        next_message = 0
        while any(p.poll() is None for p in processes):
            if time.monotonic() >= next_message:
                print(f'{mode}: running shard indices ' +
                      ', '.join(str(i) for i, p in enumerate(processes) if p.poll() is None), flush=True)
                next_message = time.monotonic() + 60
            time.sleep(2)
        codes = [p.wait() for p in processes]
    except BaseException:
        stop_workers(processes)
        record.update(interrupted_at=stamp(), exit_codes=[p.poll() for p in processes])
        save(args.control_root / f'{mode}_phase.json', record)
        raise
    statuses = phase_status(args, mode, shards)
    record.update(ended_at=stamp(), wall_seconds=round(time.monotonic()-started, 3),
                  exit_codes=codes, shards=statuses, all_processed=phase_complete(statuses, codes))
    save(args.control_root / f'{mode}_phase.json', record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    require(record['all_processed'],
            f'{mode} stopped before every shard was processed; the next phase will NOT start. '
            'Inspect logs and rerun the same launcher to resume. Original errors/results remain.')
    export_phase(args, mode, rows, shards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'status'), nargs='?', default='run')
    for name in ('project', 'source', 'weights', 'shards-root', 'control-root', 'runs-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--model-name', default='qwen25-vl-7b')
    parser.add_argument('--retry-errors', action='store_true', help='Retry communication/tool-limited results without rerunning completed answers')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--first-port', type=int, default=8000)
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--max-num-seqs', type=int, default=2)
    parser.add_argument('--memory-utilization', type=float, default=0.85)
    parser.add_argument('--start-timeout', type=float, default=1800)
    args = parser.parse_args()
    for key in ('project', 'source', 'weights', 'shards_root', 'control_root', 'runs_root'):
        setattr(args, key, getattr(args, key).resolve())
    args.gpus = args.gpus.split(',')
    require(all(gpu.isdigit() for gpu in args.gpus), 'GPU indices must be nonnegative integers')
    args.control_root.mkdir(parents=True, exist_ok=True)
    if args.action == 'status':
        paths = [args.shards_root / f'mmtab_eval_s{i}of4.json' for i in range(4)]
        require(all(p.exists() for p in paths), 'No prepared four-shard experiment found')
        for mode in ('circle', 'baseline'):
            print(json.dumps({'phase': mode, 'shards': phase_status(args, mode, paths)}, ensure_ascii=False, indent=2))
        return 0
    lock = args.control_root / 'launcher.lock'
    token = uuid.uuid4().hex
    try:
        with lock.open('x', encoding='utf-8') as file:
            json.dump({'pid': os.getpid(), 'hostname': socket.gethostname(), 'token': token, 'started_at': stamp()}, file)
    except FileExistsError as exc:
        raise ValueError(f'Launcher is locked: {lock}; verify its PID has exited before removing a stale lock') from exc
    try:
        rows, paths = prepare(args)
        if args.action == 'prepare':
            return 0
        ensure_servers(args)
        execute_phase(args, 'circle', rows, paths)
        print('Barrier satisfied: every circle shard processed. Starting baselines.', flush=True)
        execute_phase(args, 'baseline', rows, paths)
        print('Both phases processed. Services remain available; failures remain in exports.', flush=True)
        return 0
    finally:
        if lock.exists() and read(lock).get('token') == token:
            lock.unlink()


if __name__ == '__main__':
    try:
        def interrupted(*_):
            raise KeyboardInterrupt()
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, interrupted)
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nLauncher interrupted; test workers stopped, model services retained. Rerun to resume.', file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'Launcher stopped: {exc}', file=sys.stderr)
        raise SystemExit(1)
