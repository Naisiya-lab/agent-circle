"""Four-agent execution; only checker labels initiate the single content rework."""
from pathlib import Path
import time

from . import transport
from .stages import invoke
from .protocol import ProtocolError
from .budgets import Budgets
from .communication import ToolSession, wire_subtables, resolve_answer, digest_value, call_counts, route_checker

INPUT_TOOL_ERRORS = {"INVALID_ARGUMENTS", "INVALID_REGION", "UNKNOWN_TABLE"}


def direct_stage(role, stage, row, context, cfg, prompts, client, folder, log_path, session, budgets, reason):
    stage += "_direct"
    budgets.fallback(role, stage, reason)
    transport.log(log_path, f"{row['id']} {stage} direct completion: {reason}")
    if role == "select":
        feedback = context.get("repair_feedback")
        context = {"task": row["question"], "tables": session.understanding["tables"],
                   "uncertainties": session.understanding.get("uncertainties", [])}
        if feedback:
            context["repair_feedback"] = feedback
        prompt, interface = prompts["select_direct"], "select_direct"
    elif role == "solve":
        context = {k: v for k, v in context.items() if k not in ("tool_results", "tool_execution_feedback", "mapping_available")}
        prompt, interface = prompts["solve_direct"], "solve_direct"
    else:
        prompt = "\n\n".join(p for p in prompts[role].split("\n\n") if '"action":"inspect"' not in p)
        prompt += "\nReturn your final judgment from the supplied materials." if role == "check" else "\nReturn the complete HTML tables from the supplied materials."
        interface = role + "_final"
    return invoke(role, stage, row, context, cfg, prompt, client, folder, log_path, interface=interface,
                  attach_image=False if reason == "image_budget_exhausted" else None), stage


def invoke_stage(role, stage, row, context, cfg, prompts, client, folder, log_path, session, budgets=None):
    budgets = budgets or session.budgets
    base, readback = stage, 0
    observations = []
    if session is not None:
        session.operation_scope = base
    if role in ("select", "check") and session.service is None:
        return direct_stage(role, stage, row, context, cfg, prompts, client, folder, log_path,
                            session, budgets, "grid_unavailable")
    prompt = prompts[role]
    if transport.uses_image(cfg, role):
        prompt += '\nIf you need another look at the original image, return {"action":"view_image"}.'
    output = invoke(role, stage, row, context, cfg, prompt, client, folder, log_path)
    while output.get("action") in ("inspect", "view_image"):
        replies, image_requested = [], output["action"] == "view_image"
        reason = None
        if image_requested:
            key = digest_value({"stage": stage, "context": context, "request": output,
                                "retry_plan": cfg.get("_tool_retry_plan", {}).get("id")})
            if not transport.uses_image(cfg, role):
                reason = "text_role_has_no_image_input"
            elif budgets.take(role, "image", key) is None:
                reason = "image_budget_exhausted"
        else:
            batch_size = max(1, int(cfg.get("tools", {}).get("position_batch_size", 64)))
            for number, query in enumerate(output["queries"], 1):
                positions = query["positions"]
                for batch, start in enumerate(range(0, len(positions), batch_size), 1):
                    arguments = {"source_table_id": query["source_table_id"], "positions": positions[start:start+batch_size]}
                    reply = session.call(role, "inspect_cells", arguments, stage+f"_query_{number:03d}_{batch:03d}")
                    replies.append(reply)
                    if not reply["ok"] and reply["error"]["code"] == "TOOL_BUDGET_EXHAUSTED":
                        reason = "tool_budget_exhausted"
                        break
                if reason:
                    break
            # Empty inspect requests must consume an allowance too.
            if not replies and budgets.take(role, "tool", stage+":empty_query") is None:
                reason = "tool_budget_exhausted"
        observations.extend(replies)
        if reason:
            final_context = dict(context, tool_results=observations) if observations else context
            return direct_stage(role, base, row, final_context, cfg, prompts, client, folder, log_path,
                                session, budgets, reason)
        readback += 1
        stage = base+f"_after_read_{readback:03d}"
        following = dict(context, previous_request=output, tool_results=observations)
        # A query of recognized HTML does not itself request another image.
        output = invoke(role, stage, row, following, cfg, prompt, client, folder, log_path,
                        attach_image=image_requested)
    return output, stage


def model_extraction(selection, session, reason):
    resolution = {"mode": "model_direct", "requested_regions": [], "effective_requests": []}
    data = {"subtables": selection["subtables"], "available_tables": session.understanding["tables"],
            "mapping_available": False, "mapping_origin": "model", "extraction_id": None,
            "selection_resolution": resolution, "tool_response": {"ok": False, "data": None,
            "error": {"code": "MODEL_DIRECT_EXTRACTION", "message": reason}},
            "uncertainties": selection.get("uncertainties", [])}
    transport.save_json(session.sample_dir/(session.prefix+"03_extract.json"), {"role": "model", **data})
    transport.save_json(session.sample_dir/(session.prefix+"selection_resolution.json"), resolution)
    return data


def direct_resolution(solver, session):
    record = {"role": "model", "resolved_references": {}, "errors": [], "mapping_results": [],
              "candidate": solver, "coordinate_origin": "model"}
    transport.save_json(session.sample_dir/(session.prefix+"04_coordinates.json"), record)
    return solver, record


def solver_context(row, extraction):
    context = {"task": row["question"], "subtables": wire_subtables(extraction)}
    if not extraction["tool_response"]["ok"]:
        context.update(tables=extraction["available_tables"], mapping_available=False,
                       tool_results=[extraction["tool_response"]])
    if extraction.get("mapping_origin") == "model":
        context.pop("tool_results", None)
        context.pop("mapping_available", None)
        context["uncertainties"] = extraction.get("uncertainties", [])
    return context


def run_sample(row, sample_dir, cfg, prompts, client, log_path):
    folder = Path(sample_dir)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    budgets = Budgets(folder, cfg)
    target, feedback = "understand", None
    understanding = selection = solver = candidate = checked = None
    session = indexed = extraction = resolution = candidate_id = None
    last_candidate, last_solver, answer_stage = None, None, None
    answer_origin = None
    history, artifacts = [], {}
    limit = min(1, cfg["loop"]["max_rework_rounds"])
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
            session = ToolSession(row, understanding, folder, f"r{start_round:02d}_", cfg, budgets)
            indexed = session.catalog()
            if target == "solve":
                selection = transport.read_json(folder/artifacts["selection"])["output"]
                extraction = (model_extraction(selection, session, "restored_model_extraction") if "subtables" in selection
                              else session.extract(selection, indexed))
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
                understanding, stage = invoke_stage("understand", prefix+"01_understand", row, context, cfg,
                                       prompts, client, folder, log_path, None, budgets)
                artifacts["understanding"] = stage+".json"
                session = ToolSession(row, understanding, folder, prefix, cfg, budgets)
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
                session.operation_scope = stage
                extraction = (model_extraction(selection, session, "direct_selection") if "subtables" in selection
                              else session.extract(selection, indexed))
                # Repair current tool parameters with their originating agent.
                # This is not a host-assigned upstream/content rework.
                for repair in range(1, input_repairs+1):
                    reply = extraction["tool_response"]
                    if reply["ok"] or "subtables" in selection or session.service is None or reply["error"]["code"] not in INPUT_TOOL_ERRORS:
                        break
                    repair_context = dict(context, previous_selection=selection, tool_execution_feedback=[reply])
                    selection, stage = invoke_stage("select", prefix+f"02_select_tool_repair_{repair:02d}", row,
                        repair_context, cfg, prompts, client, folder, log_path, session)
                    artifacts["selection"] = stage+".json"
                    session.operation_scope = stage
                    extraction = (model_extraction(selection, session, "direct_selection") if "subtables" in selection
                                  else session.extract(selection, indexed))
                if not extraction["tool_response"]["ok"] and "subtables" not in selection:
                    selection, stage = direct_stage("select", prefix+"02_select", row, context, cfg, prompts,
                        client, folder, log_path, session, budgets, extraction["tool_response"]["error"]["code"])
                    artifacts["selection"] = stage+".json"
                    extraction = model_extraction(selection, session, "extraction_failed")
                artifacts["extraction"] = prefix+"03_extract.json"
                artifacts["selection_resolution"] = prefix+"selection_resolution.json"
            checkpoint("extracted")
            context = solver_context(row, extraction)
            if target == "solve" and feedback:
                context.update(previous_output=previous_solver, previous_answer=previous_candidate, repair_feedback=feedback)
            stage = prefix+"04_solve"
            direct_solve = extraction.get("mapping_origin") == "model" or session.service is None
            if direct_solve:
                solver, stage = direct_stage("solve", stage, row, context, cfg, prompts, client, folder, log_path,
                                              session, budgets, "model_extraction")
            else:
                solver, stage = invoke_stage("solve", stage, row, context, cfg, prompts, client, folder, log_path, session)
            artifacts["solver"] = stage+".json"
            session.operation_scope = stage
            candidate, resolution = (direct_resolution(solver, session) if direct_solve or stage.endswith("_direct")
                                     else resolve_answer(solver, session, extraction))
            last_candidate, last_solver, answer_stage = candidate, solver, stage
            answer_origin = {"stage": stage, "table_version": identity()["table_version"], "extraction_id": identity()["extraction_id"]}
            for repair in range(1, input_repairs+1):
                if not resolution["errors"] or not extraction["mapping_available"]:
                    break
                repair_context = dict(context, previous_output=solver, previous_answer=candidate,
                                      tool_execution_feedback=resolution["errors"])
                stage = prefix+f"04_solve_tool_repair_{repair:02d}"
                if any(e.get("error", {}).get("code") == "TOOL_BUDGET_EXHAUSTED" for e in resolution["errors"]):
                    break
                solver, stage = invoke_stage("solve", stage, row, repair_context, cfg, prompts, client, folder, log_path, session)
                artifacts["solver"] = stage+".json"
                session.operation_scope = stage
                candidate, resolution = (direct_resolution(solver, session) if stage.endswith("_direct")
                                         else resolve_answer(solver, session, extraction))
                last_candidate, last_solver, answer_stage = candidate, solver, stage
                answer_origin = {"stage": stage, "table_version": identity()["table_version"], "extraction_id": identity()["extraction_id"]}
            if resolution["errors"]:
                solver, stage = direct_stage("solve", prefix+"04_solve", row, context, cfg, prompts,
                    client, folder, log_path, session, budgets, "coordinate_conversion_failed")
                artifacts["solver"] = stage+".json"
                candidate, resolution = direct_resolution(solver, session)
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
    model_fallback = extraction is not None and (extraction.get("mapping_origin") == "model" or bool(resolution and resolution.get("coordinate_origin") == "model"))
    execution = "communication_failed" if communication_error else "model_fallback" if model_fallback else "tool_limited" if tool_limited else "completed"
    result = {"id": row["id"], "status": candidate.get("status", "unspecified") if candidate else "not_answered",
              "answer": candidate["answer"] if candidate else "", "final": candidate, "solver_output": solver,
              "answer_stage": answer_stage, "answer_origin": answer_origin, "execution_status": execution,
              "tool_availability": {"index_available": bool(session and session.service),
                  "extraction_available": bool(extraction and extraction["tool_response"]["ok"]),
                  "errors": ([session.registration] if session and not session.registration["ok"] else [])
                            + ([extraction["tool_response"]] if extraction and not extraction["tool_response"]["ok"] else [])
                            + (resolution["errors"] if resolution else [])}, "communication_error": communication_error,
              "verification_status": verification, "termination_reason": termination, "checker_output": checked,
              "loop_history": history, "rework_used": iteration > 0,
              "rework_target": history[0].get("return_to") if iteration else None,
              "identity": identity(), "table_version": identity()["table_version"], "artifacts": artifacts,
              "agent_budgets": budgets.summary(),
              "counts": {**call_counts(folder), "rework_rounds": iteration},
              "seconds_this_session": round(time.monotonic()-started, 3), "completed_at": transport.stamp(),
              "note": "Checker status is not accuracy. Communication and tool limitations preserve available answers and remain retryable."}
    transport.save_json(folder/"result.json", result)
    checkpoint("completed")
    transport.save_json(folder/"loop_state.json", {"completed": True, "history": history,
                        "verification_status": verification, "execution_status": execution, "termination_reason": termination})
    return result
