"""Tool-backed, version-bound four-agent loop with one targeted rework."""
from pathlib import Path
import time

from . import transport
from .stages import invoke
from .communication import ToolSession, ToolStageFailure, wire_subtables, resolve_answer, digest_value, call_counts, route_checker


def invoke_stage(role, stage, row, context, cfg, prompts, client, folder, log_path, session):
    output = invoke(role, stage, row, context, cfg, prompts[role], client, folder, log_path)
    if role not in ("select", "check") or output["action"] != "inspect":
        return output, stage
    replies = []
    for number, query in enumerate(output["queries"], 1):
        arguments = {key: query[key] for key in ("source_table_id", "positions")}
        response = session.call(role, "inspect_cells", arguments, stage+f"_query_{number:02d}", required=False)
        if not response["ok"] and response["error"].get("recovery") == "fix_host_storage":
            raise ToolStageFailure(response["error"])
        replies.append(response)
    # One readback is a communication step inside this stage, not a new workflow round.
    following = dict(context, previous_request=output, tool_results=replies,
                     instruction="Use the query results to return the final extract/judge action. No further cell query is available in this stage.")
    stage += "_after_read"
    output = invoke(role, stage, row, following, cfg, prompts[role], client, folder, log_path)
    if output["action"] == "inspect":
        raise ToolStageFailure({"code": "READ_LIMIT_EXCEEDED", "message": "The stage requested another read after its single readback"})
    return output, stage


def run_sample(row, sample_dir, cfg, prompts, client, log_path):
    folder = Path(sample_dir)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    target, feedback = "understand", None
    understanding = selection = solver = candidate = checked = None
    session = indexed = extraction = resolution = candidate_id = None
    history, artifacts = [], {}
    limit = cfg["loop"]["max_rework_rounds"]

    def identity():
        return {"sample_id": row["id"], "context_id": session.service.context_id if session else None,
                "table_version": session.service.table_version if session else None,
                "extraction_id": extraction["extraction_id"] if extraction else None, "candidate_id": candidate_id}

    def checkpoint(stage):
        transport.save_json(folder/"communication_state.json", {"identity": identity(), "round": iteration,
                            "stage": stage, "artifacts": artifacts, "target": target, "history": history})

    for iteration in range(limit+1):
        prefix = f"r{iteration:02d}_"
        previous_candidate = candidate
        previous_solver = solver
        solver = candidate = checked = resolution = candidate_id = None
        for key in ("solver", "candidate", "check", "check_routing"):
            artifacts.pop(key, None)
        if target in ("understand", "select"):
            extraction = None
            artifacts.pop("extraction", None)
            artifacts.pop("selection", None)
            artifacts.pop("selection_resolution", None)
        if target == "understand":
            session = indexed = None
            artifacts.pop("index", None)
        checkpoint("begin_round")
        try:
            if target == "understand":
                context = {"instruction": "Read the complete table image and return its complete HTML tables."}
                if understanding is not None:
                    context.update(previous_output=understanding, repair_feedback=feedback)
                understanding = invoke("understand", prefix+"01_understand", row, context, cfg,
                                       prompts["understand"], client, folder, log_path)
                artifacts["understanding"] = prefix+"01_understand.json"
                session = ToolSession(row, understanding, folder, prefix)
                indexed = session.catalog()
                transport.save_json(folder/(prefix+"table_index.json"), indexed)
                artifacts["index"] = prefix+"table_index.json"
                # No old region positions are forwarded when the recognized table changes.
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
                artifacts["extraction"] = prefix+"03_extract.json"
                artifacts["selection_resolution"] = prefix+"selection_resolution.json"
            checkpoint("extracted")
            context = {"task": row["question"], "subtables": wire_subtables(extraction)}
            if target == "solve" and feedback:
                context.update(previous_output=previous_solver, previous_answer=previous_candidate, repair_feedback=feedback)
            solver = invoke("solve", prefix+"04_solve", row, context, cfg, prompts["solve"], client, folder, log_path)
            artifacts["solver"] = prefix+"04_solve.json"
            candidate, resolution = resolve_answer(solver, session, extraction)
            candidate_id = "a_"+digest_value({"context_id": session.service.context_id,
                                             "extraction_id": extraction["extraction_id"], "candidate": candidate})
            transport.save_json(folder/(prefix+"candidate.json"), {"identity": identity(), "candidate": candidate,
                                "coordinate_requests": solver.get("coordinate_requests", [])})
            artifacts["candidate"] = prefix+"candidate.json"
            checkpoint("candidate_ready")
            context = {"task": row["question"], "table_index": indexed, "selection": selection,
                       "selection_resolution": extraction["selection_resolution"],
                       "subtables": extraction["subtables"], "candidate": candidate,
                       "coordinate_resolution": resolution["resolved_references"]}
            checked, stage = invoke_stage("check", prefix+"05_check", row, context, cfg,
                                          prompts, client, folder, log_path, session)
            artifacts["check"] = stage+".json"
            judgment = checked
            checked = route_checker(judgment)
            transport.save_json(folder/(prefix+"check_routing.json"), {
                "role": "host", "judgment": judgment, "return_to": checked["return_to"],
                "rule": "Earliest agent-labeled stage: understand > select > solve; any unknown source means no forced route."})
            artifacts["check_routing"] = prefix+"check_routing.json"
            checkpoint("checked")
        except ToolStageFailure as exc:
            feedback = {"source": "table_tool", "return_to": exc.return_to or "none", "tool_error": exc.error,
                        "issues": [{"problem": exc.error["message"], "location": exc.error.get("code", "table program")}]}
            transport.save_json(folder/(prefix+"tool_error.json"), feedback)
            history.append({"iteration": iteration, **feedback})
            transport.log(log_path, row["id"]+" tool error "+str(exc.error))
            if exc.return_to and iteration < limit:
                target = exc.return_to
                checkpoint("rework_requested")
                continue
            verification = "tool_failed"
            termination = "rework_budget_exhausted" if exc.return_to else "host_or_tool_communication_error"
            break
        feedback = dict(checked, source="answer_checker")
        history.append({"iteration": iteration, "source": "answer_checker", "status": checked["status"],
                        "return_to": checked["return_to"], "issues": checked["issues"]})
        if checked["status"] == "passed":
            verification, termination = "passed", "checker_passed"
            break
        if checked["return_to"] == "none":
            verification, termination = checked["status"], "checker_uncertain_without_route"
            break
        if iteration >= limit:
            verification, termination = checked["status"], "rework_budget_exhausted"
            break
        target = checked["return_to"]
    result = {"id": row["id"], "status": candidate["status"] if candidate else "not_answered",
              "answer": candidate["answer"] if candidate else "", "final": candidate, "solver_output": solver,
              "verification_status": verification, "termination_reason": termination, "checker_output": checked,
              "loop_history": history, "rework_used": iteration > 0,
              "rework_target": history[0]["return_to"] if iteration else None,
              "identity": identity(), "table_version": identity()["table_version"], "artifacts": artifacts,
              "counts": {**call_counts(folder), "rework_rounds": iteration},
              "seconds_this_session": round(time.monotonic()-started, 3), "completed_at": transport.stamp(),
              "note": "Checker acceptance is a model judgment, not a correctness label. Tool calls include any idempotent replay after interruption."}
    transport.save_json(folder/"result.json", result)
    checkpoint("completed")
    transport.save_json(folder/"loop_state.json", {"completed": True, "history": history,
                        "verification_status": verification, "termination_reason": termination})
    return result
