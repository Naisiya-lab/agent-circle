"""Offline interface/tool replay of an archived run; not a correctness evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tool_loop_runtime.protocol import decode, ProtocolError
from table_tools import TableTools, ProgramError


def replay(archive):
    samples, attempts, grids, successful_grids = [], [], [], []
    grid_cache = {}
    with tarfile.open(archive, "r:*") as package:
        names={m.name:m for m in package.getmembers() if m.isfile()}
        def read(name):
            with package.extractfile(names[name]) as file:
                return json.load(file)
        for name in names:
            if '/attempts/' not in name or not name.endswith('.json'):
                continue
            record=read(name)
            role=record.get('role')
            choices=record.get('raw_response',{}).get('choices') or []
            if not choices or role not in ('understand','select','solve','check'):
                continue
            choice=choices[0]
            try:
                decode(choice.get('message',{}).get('content'),role)
                accepted,error=True,None
            except ProtocolError as exc:
                accepted,error=False,str(exc)
            attempts.append({'path':name,'role':role,'previous_protocol':record.get('protocol',{}).get('status'),
                             'finish_reason':choice.get('finish_reason'),'readable_now':accepted,'remaining_error':error})
        for name in names:
            if '/samples/' not in name or not name.endswith('/error.json'):
                continue
            error=read(name)
            stage=error.get('stage')
            stage_path=name.rsplit('/',1)[0]+'/'+str(stage)+'.json'
            result={'path':name,'id':error.get('id'),'role':error.get('role'),'old_error':error.get('message')}
            if stage_path in names:
                stage_record=read(stage_path)
                choices=stage_record.get('raw_response',{}).get('choices') or []
                if choices:
                    try:
                        decode(choices[0].get('message',{}).get('content'),stage_record['role'])
                        result.update(readable_now=True,remaining_error=None)
                    except ProtocolError as exc:
                        result.update(readable_now=False,remaining_error=str(exc))
            samples.append(result)
        for name in names:
            if '/samples/' not in name or not name.endswith('/result.json'):
                continue
            result=read(name)
            if result.get('verification_status')!='tool_failed':
                continue
            stage=result.get('artifacts',{}).get('understanding')
            stage_path=name.rsplit('/',1)[0]+'/'+str(stage)
            if stage_path not in names:
                continue
            output=read(stage_path).get('output')
            if not output:
                continue
            previous_errors=[x.get('tool_error',{}) for x in result.get('loop_history',[]) if x.get('source')=='table_tool']
            entry={'id':result.get('id'),'path':name,'previous_final_tool_error':previous_errors[-1].get('code') if previous_errors else None}
            try:
                service=TableTools(result['id'],output)
                entry.update(index_available=True,table_version=service.table_version)
            except ProgramError as exc:
                entry.update(index_available=False,tool_error=exc.record())
            grids.append(entry)
        for name in names:
            if not name.endswith('/tool_events.jsonl'):
                continue
            with package.extractfile(names[name]) as file:
                events=[json.loads(line) for line in file if line.strip()]
            for event in events:
                if event.get('tool')!='register_tables' or not event.get('response',{}).get('ok'):
                    continue
                folder=name.rsplit('/',1)[0]
                stage_path=folder+'/'+event['round']+'01_understand.json'
                if stage_path in grid_cache or stage_path not in names:
                    continue
                grid_cache[stage_path]=True
                output=read(stage_path).get('output')
                if not output:continue
                entry={'path':stage_path,'still_indexable':False,'same_dimensions':False}
                try:
                    service=TableTools('replay',output)
                    entry['still_indexable']=True
                    old_path=folder+'/'+event['round']+'table_index.json'
                    if old_path in names:
                        old=read(old_path)
                        before=[(x['id'],x['row_count'],x['column_count']) for x in old['tables']]
                        after=[(x['id'],x['row_count'],x['column_count']) for x in service._catalog['tables']]
                        entry['same_dimensions']=before==after
                        # Comparing normalized per-cell facts avoids confusing a
                        # harmless markup normalization with changed table data.
                        from table_tools.html_grid import Table
                        def facts(table):
                            return [(c.row,c.column,c.rowspan,c.colspan,c.content,c.attributes) for c in table.cells]
                        old_facts={x['id']:facts(Table(x['id'],x['html'])) for x in old['tables']}
                        entry['same_cells_and_merges']=all(old_facts[k]==facts(v) for k,v in service._tables.items())
                except ProgramError as exc:
                    entry['tool_error']=exc.record()
                successful_grids.append(entry)
    return {'scope':'Offline replay of already recorded responses only. No model calls, images, answer scoring, or claim of accuracy improvement.',
            'archive':str(Path(archive).resolve()),'recorded_attempts':len(attempts),
            'attempts_readable_now':sum(x['readable_now'] for x in attempts),
            'attempts_by_role':dict(Counter(x['role'] for x in attempts)),
            'terminal_error_samples':len(samples),
            'terminal_errors_with_saved_response':sum('readable_now' in x for x in samples),
            'terminal_saved_responses_readable_now':sum(x.get('readable_now',False) for x in samples),
            'previous_tool_failed_samples_with_html':len(grids),
            'previous_tool_failed_now_indexable':sum(x['index_available'] for x in grids),
            'previous_grid_failures_now_indexable':sum(x['index_available'] and x['previous_final_tool_error']=='HTML_GRID_INVALID' for x in grids),
            'previous_successful_grids':len(successful_grids),
            'previous_successful_grids_still_indexable':sum(x['still_indexable'] for x in successful_grids),
            'previous_successful_grids_same_dimensions':sum(x['same_dimensions'] for x in successful_grids),
            'previous_successful_grids_same_cells_and_merges':sum(x.get('same_cells_and_merges',False) for x in successful_grids),
            'remaining_grid_errors':dict(Counter(x['tool_error']['code'] for x in grids if not x['index_available'])),
            'sample_replay':samples,'grid_replay':grids,'attempt_replay':attempts,'successful_grid_replay':successful_grids}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=replay(args.archive)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sample_replay','grid_replay','attempt_replay','successful_grid_replay')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
