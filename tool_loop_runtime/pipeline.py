"""Four-agent execution; only checker labels initiate the single content rework."""
from pathlib import Path
import time

from . import transport
from .stages import invoke
from .protocol import ProtocolError
from .communication import ToolSession, wire_subtables, resolve_answer, digest_value, call_counts, route_checker

INPUT_TOOL_ERRORS = {"INVALID_ARGUMENTS", "INVALID_REGION", "UNKNOWN_TABLE"}


def invoke_stage(role, stage, row, context, cfg, prompts, client, folder, log_path, session):
    base, readback = stage, 0
    output = invoke(role, stage, row, context, cfg, prompts[role], client, folder, log_path)
    while role in ("select", "check") and output.get("action") == "inspect":
        replies = []
        batch_size = max(1, int(cfg.get("tools", {}).get("position_batch_size", 64)))
        for number, query in enumerate(output["queries"], 1):
            positions = query["positions"]
            for batch, start in enumerate(range(0, len(positions), batch_size), 1):
                arguments = {"source_table_id": query["source_table_id"], "positions": positions[start:start+batch_size]}
                replies.append(session.call(role, "inspect_cells", arguments, stage+f"_query_{number:03d}_{batch:03d}"))
        readback += 1
        stage = base+f"_after_read_{readback:03d}"
        following = dict(context, previous_request=output, tool_results=replies)
        output = invoke(role, stage, row, following, cfg, prompts[role], client, folder, log_path)
    return output, stage


def solver_context(row, extraction):
    context = {"task": row["question"], "subtables": wire_subtables(extraction)}
    if not extraction["tool_response"]["ok"]:
        context.update(tables=extraction["available_tables"], mapping_available=False,
                       tool_results=[extraction["tool_response"]])
    return context


def run_sample(row, sample_dir, cfg, prompts, client, log_path):
    folder = Path(sample_dir)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    target, feedback = "understand", None
    understanding = selection = solver = candidate = checked = None
    session = indexed = extraction = resolution = candidate_id = None
    last_candidate, last_solver, answer_stage = None, None, None
    answer_origin = None
    history, artifacts = [], {}
    limit = cfg["loop"]["max_rework_rounds"]
    input_repairs = cfg.get("tools", {}).get("input_retries", 2)
    retry_plan = cfg.get("_tool_retry_plan")
    start_round = retry_plan.get("content_reworks_used", 0) if retry_plan else 0
    if retry_plan:
        previous = retry_plan["previous_result"]
        target = retry_plan["owner"]
        artifacts = dict(previous.get("artifacts", {}))
        history = list(previous.get("loop_history", []))
        candidate = last_candidate = previous.get("final")
        solver = last_solver = previous.get("solver_output")
        answer_stage = previous.get("answer_stage")
        answer_origin = previous.get("answer_origin")
        # Reconstruct only the already available upstream tool state. Source
        # identity is rebuilt and checked by the same table program as usual.
        if target != "understand":
            understanding = transport.read_json(folder/artifacts["understanding"])["output"]
            session = ToolSession(row, understanding, folder, f"r{start_round:02d}_", cfg)
            indexed = session.catalog()
            if target == "solve":
                selection = transport.read_json(folder/artifacts["selection"])["output"]
                extraction = session.extract(selection, indexed)
    communication_error = None
    termination, verification = "", "not_checked"

    def identity():
        service = session.service if session else None
        return {"sample_id": row["id"], "context_id": service.context_id if service else None,
                "table_version": service.table_version if service else None,
                "extraction_id": extraction.get("extraction_id") if extraction else None, "candidate_id": candidate_id}

    def checkpoint(stage):
        transport.save_json(folder/"communication_state.json", {"identity": identity(), "round": iteration,
                            "stage": stage, "artifacts": artifacts, "target": target, "history": history})

    for iteration in range(start_round, limit+1):
        prefix = f"r{iteration:02d}_"
        previous_candidate, previous_solver = candidate, solver
        checked, resolution = None, None
        checkpoint("begin_round")
        try:
            if target == "understand":
                context = {"instruction": "Read the complete table image and return its complete HTML tables."}
                if understanding is not None:
                    context.update(previous_output=understanding, repair_feedback=feedback)
                understanding = invoke("understand", prefix+"01_understand", row, context, cfg,
                                       prompts["understand"], client, folder, log_path)
                artifacts["understanding"] = prefix+"01_understand.json"
                session = ToolSession(row, understanding, folder, prefix, cfg)
                indexed = session.catalog()
                transport.save_json(folder/(prefix+"table_index.json"), indexed)
                artifacts["index"] = prefix+"table_index.json"
                selection = None
            else:
                session.prefix = prefix
            checkpoint("indexed")
            if target in ("understand", "select"):
                context = {"task": row["question"], "table_index": indexed}
                if target == "select" and feedback:
                    context.update(previous_selection=selection, repair_feedback=feedback)
                selection, stage = invoke_stage("select", prefix+"02_select", row, context, cfg,
                                                prompts, client, folder, log_path, session)
                artifacts["selection"] = stage+".json"
                extraction = session.extract(selection, indexed)
                # Repair current tool parameters with their originating agent.
                # This is not a host-assigned upstream/content rework.
                for repair in range(1, input_repairs+1):
                    reply = extraction["tool_response"]
                    if reply["ok"] or session.service is None or reply["error"]["code"] not in INPUT_TOOL_ERRORS:
                        break
                    repair_context = dict(context, previous_selection=selection, tool_execution_feedback=[reply])
                    selection, stage = invoke_stage("select", prefix+f"02_select_tool_repair_{repair:02d}", row,
                        repair_context, cfg, prompts, client, folder, log_path, session)
                    artifacts["selection"] = stage+".json"
                    extraction = session.extract(selection, indexed)
                artifacts["extraction"] = prefix+"03_extract.json"
                artifacts["selection_resolution"] = prefix+"selection_resolution.json"
            checkpoint("extracted")
            context = solver_context(row, extraction)
            if target == "solve" and feedback:
                context.update(previous_output=previous_solver, previous_answer=previous_candidate, repair_feedback=feedback)
            stage = prefix+"04_solve"
            solver = invoke("solve", stage, row, context, cfg, prompts["solve"], client, folder, log_path)
            artifacts["solver"] = stage+".json"
            candidate, resolution = resolve_answer(solver, session, extraction)
            last_candidate, last_solver, answer_stage = candidate, solver, stage
            answer_origin = {"stage": stage, "table_version": identity()["table_version"], "extraction_id": identity()["extraction_id"]}
            for repair in range(1, input_repairs+1):
                if not resolution["errors"] or not extraction["mapping_available"]:
                    break
                repair_context = dict(context, previous_output=solver, previous_answer=candidate,
                                      tool_execution_feedback=resolution["errors"])
                stage = prefix+f"04_solve_tool_repair_{repair:02d}"
                solver = invoke("solve", stage, row, repair_context, cfg, prompts["solve"], client, folder, log_path)
                artifacts["solver"] = stage+".json"
                candidate, resolution = resolve_answer(solver, session, extraction)
                last_candidate, last_solver, answer_stage = candidate, solver, stage
                answer_origin = {"stage": stage, "table_version": identity()["table_version"], "extraction_id": identity()["extraction_id"]}
            candidate_id = "a_"+digest_value({"table_version": identity()["table_version"], "extraction_id": identity()["extraction_id"], "candidate": candidate})
            transport.save_json(folder/(prefix+"candidate.json"), {"identity": identity(), "candidate": candidate,
                                "coordinate_requests": solver.get("coordinate_requests", [])})
            artifacts["candidate"] = prefix+"candidate.json"
            checkpoint("candidate_ready")
            context = {"task": row["question"], "table_index": indexed, "selection": selection,
                       "selection_resolution": extraction["selection_resolution"], "subtables": extraction["subtables"],
                       "candidate": candidate, "coordinate_resolution": resolution["resolved_references"]}
            if not extraction["tool_response"]["ok"]:
                context.update(available_tables=extraction["available_tables"], tool_results=[extraction["tool_response"]])
            if resolution["errors"]:
                context["coordinate_tool_errors"] = resolution["errors"]
            checked, stage = invoke_stage("check", prefix+"05_check", row, context, cfg,
                                          prompts, client, folder, log_path, session)
            artifacts["check"] = stage+".json"
            judgment = checked
            checked = route_checker(judgment)
            transport.save_json(folder/(prefix+"check_routing.json"), {
                "role": "host", "judgment": judgment, "return_to": checked["return_to"],
                "rule": "Dispatch from agent-labeled stages only; unknown sources do not force a route."})
            artifacts["check_routing"] = prefix+"check_routing.json"
            checkpoint("checked")
        except (ProtocolError, transport.ApiFailure) as exc:
            if isinstance(exc, transport.ApiFailure) and exc.fatal:
                raise  # Account/service outage handling, not a response-quality gate.
            communication_error = {"id": row["id"], "time": transport.stamp(), "message": str(exc),
                "type": type(exc).__name__, "role": getattr(exc, "role", None), "stage": getattr(exc, "stage", None),
                "retryable": True, "fatal": False, "category": "communication_failed"}
            if isinstance(exc, ProtocolError):
                communication_error["protocol_errors"] = exc.errors
            verification, termination = "communication_failed", "communication_repair_exhausted"
            history.append({"iteration": iteration, "source": "communication", "error": communication_error})
            # Keep the most recent readable solver answer even if the checker
            # or a later content-rework request could not be delivered/parsed.
            candidate, solver = last_candidate, last_solver
            break
        feedback = dict(checked, source="answer_checker")
        history.append({"iteration": iteration, "source": "answer_checker", "status": checked["status"],
                        "return_to": checked["return_to"], "issues": checked.get("issues", [])})
        verification = checked["status"]
        if checked["status"] == "passed":
            termination = "checker_passed"
            break
        if checked["return_to"] == "none":
            termination = "checker_without_route"
            break
        if iteration == limit:
            termination = "rework_budget_exhausted"
            break
        target = checked["return_to"]
    tool_limited = extraction is not None and (not extraction["tool_response"]["ok"] or bool(resolution and resolution["errors"]))
    execution = "communication_failed" if communication_error else "tool_limited" if tool_limited else "completed"
    result = {"id": row["id"], "status": candidate.get("status", "unspecified") if candidate else "not_answered",
              "answer": candidate["answer"] if candidate else "", "final": candidate, "solver_output": solver,
              "answer_stage": answer_stage, "answer_origin": answer_origin, "execution_status": execution,
              "tool_availability": {"index_available": bool(session and session.service),
                  "extraction_available": bool(extraction and extraction["tool_response"]["ok"]),
                  "errors": ([extraction["tool_response"]] if extraction and not extraction["tool_response"]["ok"] else [])
                            + (resolution["errors"] if resolution else [])}, "communication_error": communication_error,
              "verification_status": verification, "termination_reason": termination, "checker_output": checked,
              "loop_history": history, "rework_used": iteration > 0,
              "rework_target": history[0].get("return_to") if iteration else None,
              "identity": identity(), "table_version": identity()["table_version"], "artifacts": artifacts,
              "counts": {**call_counts(folder), "rework_rounds": iteration},
              "seconds_this_session": round(time.monotonic()-started, 3), "completed_at": transport.stamp(),
              "note": "Checker status is not accuracy. Communication and tool limitations preserve available answers and remain retryable."}
    transport.save_json(folder/"result.json", result)
    checkpoint("completed")
    transport.save_json(folder/"loop_state.json", {"completed": True, "history": history,
                        "verification_status": verification, "execution_status": execution, "termination_reason": termination})
    return result
