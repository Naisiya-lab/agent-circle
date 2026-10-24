"""Direct M(image, original question), using the same provider and model as the circle."""
import base64
from html import escape
import json
import mimetypes
from pathlib import Path
import time
import uuid
from . import transport, runner
from .providers import build_request


def body_for(row, cfg):
    image = Path(row['image'])
    transport.require(transport.digest(image.read_bytes()) == row['image_sha256'], 'Image changed')
    url = 'data:'+(mimetypes.guess_type(str(image))[0] or 'image/jpeg')+';base64,'+base64.b64encode(image.read_bytes()).decode('ascii')
    return build_request(cfg, 'solve', [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': url}}, {'type': 'text', 'text': row['question']}]}])


def run(cfg, name, source, limit, max_samples=0, retry_errors=False, prepare_only=False):
    root, rows, _ = runner.prepare(cfg, name, source, limit)
    transport.require(cfg['experiment']['mode'] == 'baseline', 'Expected baseline configuration')
    report(root)
    if prepare_only: return 0
    lock = runner.acquire_lock(root)
    code = 0
    try:
        pending = [r for r in rows if not (root/'samples'/r['id']/'direct_result.json').exists()
                   and (retry_errors or not (root/'samples'/r['id']/'error.json').exists()
                        or transport.saved_error_is_transient(root/'samples'/r['id']/'error.json'))]
        if max_samples: pending = pending[:max_samples]
        endpoint, env = transport.api_binding(cfg, 'solve')
        import os, getpass, sys
        key = os.environ.get(env)
        if not key and not cfg['role_apis']['solve'].get('require_key', True): key = 'EMPTY'
        if pending and not key:
            transport.require(sys.stdin.isatty(), 'Set '+env+' for noninteractive execution')
            key = getpass.getpass('API key for '+cfg['models']['solve']+' (hidden): ').strip()
            transport.require(bool(key), 'Empty API key')
        client = transport.RoleApiClient(cfg, {(endpoint,env):key} if pending else {}, root)
        completed=sum((root/'samples'/r['id']/'direct_result.json').exists() for r in rows)
        progress = transport.RunProgress(len(pending),len(rows),completed)
        client.progress = progress.show
        started = time.monotonic()
        for row in pending:
            folder = root/'samples'/row['id'];folder.mkdir(parents=True,exist_ok=True)
            try:
                body = body_for(row,cfg); then = time.monotonic()
                fingerprint=transport.semantic_request_fingerprint(body)
                saved=folder/'direct_response.json'
                if saved.exists() and not retry_errors:
                    record=transport.read_json(saved)
                    transport.require(record['request_fingerprint']==fingerprint,'Saved baseline request changed')
                    raw=record['raw_response']
                else:
                    raw = client.call(body,row['id'],'direct',role='solve')
                    record = {'model':cfg['models']['solve'],'image_sha256':row['image_sha256'],
                              'request_fingerprint':fingerprint,
                              'seconds':round(time.monotonic()-then,3),'raw_response':raw}
                    transport.save_json(saved,record)
                choice = raw['choices'][0]; content=choice['message'].get('content')
                transport.require(choice.get('finish_reason') in (None,'stop') and isinstance(content,str) and bool(content.strip()), 'Incomplete or empty baseline response')
                transport.save_json(folder/'direct_result.json',{'id':row['id'],'answer':content,'status':'answered',
                                    'model_calls':1,'seconds_this_session':record['seconds'],'completed_at':transport.stamp()})
                (folder/'error.json').unlink(missing_ok=True);progress.completed+=1
            except (ValueError,transport.ApiFailure,KeyError,TypeError) as exc:
                fatal=isinstance(exc,transport.ApiFailure) and exc.fatal
                transport.save_json(folder/'error.json',{'id':row['id'],'message':str(exc),'type':type(exc).__name__,
                    'retryable':isinstance(exc,transport.ApiFailure) and exc.retryable,'fatal':fatal})
                code=1;progress.errors+=1
                if fatal:raise
            progress.finished+=1;progress.show()
            if progress.finished % cfg.get('report_every_samples',1)==0 or progress.finished==len(pending):report(root)
        transport.save_json(root/'sessions'/('direct_'+uuid.uuid4().hex+'.json'),
                            {'mode':'baseline','elapsed_seconds':round(time.monotonic()-started,3),
                             'selected_ids_this_session':[r['id'] for r in pending]})
    except KeyboardInterrupt:code=130
    except transport.ApiFailure:code=1
    finally:
        report(root);lock.unlink(missing_ok=True)
    return code


def report(root):
    root=Path(root);rows=transport.read_json(root/'selected.json');results=[];usage={};cards=[];seconds=0
    for row in rows:
        folder=root/'samples'/row['id'];result=folder/'direct_result.json'
        if result.exists():
            r=transport.read_json(result);results.append(r);seconds+=r['seconds_this_session']
            for k,v in transport.read_json(folder/'direct_response.json')['raw_response'].get('usage',{}).items():
                if type(v) is int:usage[k]=usage.get(k,0)+v
        else:r={'answer':'pending/error'}
        if len(cards)<100:cards.append('<article><h2>'+escape(row['id'])+'</h2><pre>'+escape(row['question'])+'</pre><pre>'+escape(r['answer'])+'</pre></article>')
    summary={'selected':len(rows),'completed':len(results),'errors':sum((root/'samples'/r['id']/'error.json').exists() for r in rows),
             'successful_response_seconds':round(seconds,3),'api_usage':usage,'note':'Direct model baseline; no correctness judgment.'}
    transport.save_json(root/'results.json',results);transport.save_json(root/'summary.json',summary)
    (root/'predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in results),encoding='utf-8')
    (root/'review.html').write_text('<meta charset="utf-8"><style>pre{white-space:pre-wrap}article{border-bottom:1px solid #aaa}</style><h1>Direct model baseline</h1><pre>'+escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards),encoding='utf-8')
    return summary
