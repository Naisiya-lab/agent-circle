"""Explicit retry eligibility and append-only histories; never score answers."""
import json
import uuid
from . import transport


def result_is_retryable(result, include_tool_failures=False):
    if result.get("execution_status") == "communication_failed":
        return True
    return include_tool_failures and (result.get("execution_status") == "tool_limited" or result.get("verification_status") == "tool_failed")


def needs_run(folder, result_name="result.json", retry_errors=False):
    path = folder/result_name
    if not path.exists():
        # An error record is diagnostic state, not a permanent skip marker.
        return True
    plan_path=folder/"tool_retry_plan.json"
    if plan_path.exists():
        plan=transport.read_json(plan_path)
        if plan.get("state")=="in_progress" and plan.get("from_result_sha256")==transport.digest(path.read_bytes()):
            return True
    return result_is_retryable(transport.read_json(path), retry_errors)


def history(folder, event, value):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/"recovery_history.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps({"time": transport.stamp(), "event": event, "record": value}, ensure_ascii=False)+"\n")


def begin_retry(folder, result_name="result.json"):
    for name in (result_name, "error.json"):
        path = folder/name
        if path.exists():
            history(folder, "previous_"+name, transport.read_json(path))


def record_error(folder, error):
    history(folder, "error", error)
    transport.save_json(folder/"error.json", error)


def clear_error(folder):
    path = folder/"error.json"
    if path.exists():
        history(folder, "resolved_error", transport.read_json(path))
        path.unlink()


def prepare_tool_retry(folder, cfg, retry_errors=False):
    """Refresh the failed tool's input provider and dependent cached stages.

    This is an explicit execution retry, not an inferred semantic-error route.
    Keep the plan stable across interruptions; every archived stage also keeps
    its original model response in attempts/.
    """
    result_path=folder/"result.json"
    plan_path=folder/"tool_retry_plan.json"
    if not result_path.exists():
        return cfg
    result=transport.read_json(result_path)
    previous=transport.read_json(plan_path) if plan_path.exists() else None
    digest=transport.digest(result_path.read_bytes())
    if previous and ((previous.get("state")=="in_progress" and previous.get("from_result_sha256")==digest)
                     or result.get("execution_status")=="communication_failed"):
        return _activate_tool_plan(folder,cfg,previous)
    if not retry_errors or not result_is_retryable(result,True) or result.get("execution_status")=="communication_failed":
        return cfg
    availability=result.get("tool_availability",{})
    if not availability.get("index_available",False):
        owner,roles='understand',('understand','select','solve','check')
    elif not availability.get("extraction_available",False):
        owner,roles='select',('select','solve','check')
    else:
        owner,roles='solve',('solve','check')
    key={'understand':'understanding','select':'selection','solve':'solver'}[owner]
    artifact=result.get('artifacts',{}).get(key)
    source=folder/artifact if artifact else None
    previous_output=transport.read_json(source).get('output') if source and source.exists() else None
    plan={'id':uuid.uuid4().hex,'state':'in_progress','from_result_sha256':digest,'owner':owner,
          'roles':list(roles),'previous_output':previous_output,
          'tool_errors':availability.get('errors',[]),'previous_table_version':result.get('table_version'),
          'content_reworks_used':result.get('counts',{}).get('rework_rounds',0),
          'previous_result':result}
    # Persist first so interruption during archiving is resumable.
    transport.save_json(plan_path,plan)
    return _activate_tool_plan(folder,cfg,plan)


def _activate_tool_plan(folder,cfg,plan):
    archive=folder/'retry_stages'/plan['id']
    for path in folder.glob('r*.json'):
        record=transport.read_json(path)
        if record.get('role') not in plan['roles'] or 'raw_response' not in record:
            continue
        # If this plan is already partway through a retry, its new responses
        # must remain reusable rather than being invalidated a second time.
        if record.get('tool_retry_id')==plan['id']:
            continue
        archive.mkdir(parents=True,exist_ok=True)
        path.replace(archive/path.name)
    return dict(cfg,_tool_retry_plan=plan)


def finish_tool_retry(folder):
    path=folder/'tool_retry_plan.json'
    if path.exists():
        plan=transport.read_json(path)
        plan.update(state='finished',finished_at=transport.stamp())
        transport.save_json(path,plan)
