"""Portable MMTab-eval entry point. Dataset/model/output paths are CLI or shell inputs."""
import argparse
import json
from pathlib import Path
from tool_loop_runtime import runner, baseline
from tool_loop_runtime.configuration import load_config
from tool_loop_runtime.dataset import prepare_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    data=commands.add_parser('prepare-data')
    for flag in ['dataset','images','output']:data.add_argument('--'+flag,type=Path,required=True)
    data.add_argument('--include-tasks',nargs='*',default=[]);data.add_argument('--exclude-tasks',nargs='*',default=[])
    data.add_argument('--limit',type=int,default=0);data.add_argument('--shards',type=int,default=1);data.add_argument('--shard-index',type=int,default=0)
    for command in ['prepare','run']:
        p=commands.add_parser(command);p.add_argument('--mode',choices=['baseline','circle'],required=True)
        p.add_argument('--route',choices=['A','B'],default='B');p.add_argument('--model-profile',type=Path,required=True)
        p.add_argument('--runtime-config',type=Path);p.add_argument('--source',type=Path,required=True);p.add_argument('--run',required=True)
        p.add_argument('--limit',type=int,default=0);p.add_argument('--max-samples',type=int,default=0);p.add_argument('--retry-errors',action='store_true')
    p=commands.add_parser('report');p.add_argument('--run',required=True)
    p=commands.add_parser('export');p.add_argument('--run',required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare-data':
        print(json.dumps(prepare_manifest(args.dataset,args.images,args.output,args.include_tasks,args.exclude_tasks,args.limit,args.shard_index,args.shards),ensure_ascii=False,indent=2));return 0
    if args.command in ('report','export'):
        root=runner.run_root(args.run);cfg=runner.transport.read_json(root/'config.json')['config']
        if args.command=='report':
            print(json.dumps((baseline.report if cfg['experiment']['mode']=='baseline' else runner.report)(root),ensure_ascii=False,indent=2));return 0
        rows=runner.transport.read_json(root/'selected.json');args.output.parent.mkdir(parents=True,exist_ok=True)
        filename='direct_result.json' if cfg['experiment']['mode']=='baseline' else 'result.json'
        exported=[]
        for row in rows:
            path=root/'samples'/row['id']/filename
            result=runner.transport.read_json(path) if path.exists() else {}
            exported.append({'question_id':row['id'],'text':result.get('answer',''),
                'model_id':cfg['models']['solve'], 'metadata':{'completed':bool(result),
                'execution_status':result.get('execution_status','pending_or_error'),
                'verification_status':result.get('verification_status'), 'answer_origin':result.get('answer_origin')}})
        # Read authoritative sample files rather than a periodically refreshed
        # report. Preserve every selected ID, including pending/failed items.
        args.output.write_text(''.join(json.dumps(item,ensure_ascii=False)+'\n' for item in exported),encoding='utf-8')
        return 0
    cfg=load_config(args.model_profile,args.route,args.runtime_config);cfg['experiment']['mode']=args.mode
    limit=args.limit or len(runner.transport.read_json(args.source))
    if args.mode=='baseline':return baseline.run(cfg,args.run,args.source,limit,args.max_samples,args.retry_errors,args.command=='prepare')
    if args.command=='prepare':
        root,rows,_=runner.prepare(cfg,args.run,args.source,limit);print(f'Prepared {len(rows)} samples; no API calls. {root}');return 0
    return runner.run(cfg,args.run,args.source,limit,args.max_samples,args.retry_errors)


if __name__=='__main__':raise SystemExit(main())
