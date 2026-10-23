"""Hash-pinned official-training batches with spawn isolation and shared admission.

The design manifest is deliberately not executable. This runner does not resume
failed Agents: each new attempt has a new directory. Explicit completed receipts
may be reused after checking their inputs, settings and current Word again.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time
import uuid

from .document_batch_admission import BatchAdmissionBroker
from .document_delivery_artifacts import DocumentDeliveryArtifacts
from .document_multiagent import MultiAgentSettings, run_multiagent
from .document_multiagent_cli import prepare_training_input
from .document_task_plan import build_document_task_plan
from .run_state import atomic_json, json_digest

RUNTIME_SCHEMA = "cityagent-multiagent-batch-runtime-v1"
FILE_FIELDS = ("official_original", "source_docx", "manifest", "materials",
               "full_source_inventory", "excluded_content")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pinned_file(record, base):
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("Each input needs an explicit path and exact file SHA256")
    path = (Path(base) / record["path"]).resolve()
    sha = record.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64 or file_sha(path) != sha:
        raise ValueError(f"Input file SHA256 differs: {path}")
    return {"path": str(path), "sha256": sha}


def _load_document(document, settings):
    # Repeat byte checks in the worker; no input may change after preflight.
    for field in FILE_FIELDS:
        _pinned_file(document[field], Path("/"))
    manifest, materials, index = prepare_training_input(
        document["manifest"]["path"], document["manifest"]["sha256"],
        materials_path=document["materials"]["path"], materials_sha256=document["materials"]["sha256"],
        source_docx=document["source_docx"]["path"])
    if manifest.get("document_sha256") != document["source_docx"]["sha256"]:
        raise ValueError("Canonical manifest does not belong to the pinned DOCX")
    plan = build_document_task_plan(manifest, target_tasks=settings.tasks,
                                   max_tasks=settings.tasks, material_index=index)
    return manifest, materials, plan


def prepare_batch(path, expected_sha256, *, settings=MultiAgentSettings()):
    """Validate every listed file and material index before starting any worker."""
    path = Path(path).resolve()
    if file_sha(path) != expected_sha256:
        raise ValueError("Batch runtime file SHA256 differs")
    runtime = json.loads(path.read_bytes())
    scope = runtime.get("scope")
    permitted_scope = (scope == "official-training" or
                       scope == "official-test" and runtime.get("test_inference_only") is True)
    if (runtime.get("schema") != RUNTIME_SCHEMA or runtime.get("status") != "ready"
            or not permitted_scope):
        raise ValueError("Only explicit ready official-training runtime manifests are executable; design is not runtime")
    source = runtime.get("documents")
    if not isinstance(source, list) or not source:
        raise ValueError("Batch requires a nonempty document list")
    documents, ids = [], set()
    for position, item in enumerate(source):
        label = item.get("document_id") if isinstance(item, dict) else None
        if not isinstance(label, str) or not label.strip() or label in ids:
            raise ValueError("document_id labels must be nonempty and unique")
        ids.add(label)
        if (item.get("scope") != scope or item.get("runtime_conversion_required") is not False
                or item.get("evaluation_outputs_are_not_model_inputs") is not True):
            raise ValueError("Each document must assert official training, prepared conversion and no evaluation inputs")
        document = {"document_id": label, "title": item.get("title", label), "position": position, "scope": scope,
                    **{field: _pinned_file(item.get(field), path.parent) for field in FILE_FIELDS}}
        manifest, materials, plan = _load_document(document, settings)
        document.update(document_sha256=manifest["document_sha256"],
                        manifest_identity_sha256=json_digest(manifest),
                        materials_identity_sha256=json_digest(materials), plan=plan,
                        input_identity_sha256=json_digest({key: document[key]["sha256"] for key in FILE_FIELDS}))
        # Labels never become directory components or model request identities.
        document["document_key"] = json_digest({"position": position, "document_id": label,
                                                "input": document["input_identity_sha256"]})
        if "reuse_result" in item:
            document["reuse_result"] = _pinned_file(item["reuse_result"], path.parent)
        documents.append(document)
    return {"schema": RUNTIME_SCHEMA, "scope": scope, "scope_is_operator_assertion": True,
            "runtime_path": str(path), "runtime_file_sha256": expected_sha256, "documents": documents,
            "model_calls": 0, "status": "inputs_validated"}


@dataclass(frozen=True)
class BatchSettings:
    document_workers: int = 3
    global_main_model_inflight: int = 2
    global_summary_model_inflight: int = 1
    settlement_grace_seconds: float = 30

    def __post_init__(self):
        for name in ("document_workers", "global_main_model_inflight", "global_summary_model_inflight"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (not isinstance(self.settlement_grace_seconds, (int, float))
                or not math.isfinite(self.settlement_grace_seconds) or self.settlement_grace_seconds < 0):
            raise ValueError("Invalid settlement grace")


def _within(path, root):
    path = Path(path).resolve()
    path.relative_to(Path(root).resolve())
    return path


def verify_current_delivery(run_directory, result, *, require_complete=True, manifest_identity_sha256=None):
    """Read exact current workspace/receipt/Word; mtime and row counts are irrelevant."""
    run_directory = Path(run_directory).resolve()
    receipt = deepcopy(result.get("delivery"))
    if not isinstance(receipt, dict) or not result.get("artifact_selection", {}).get("current_workspace_verified"):
        raise ValueError("No verified current workspace artifact")
    for field in ("json_path", "docx_path", "status_path"):
        _within(receipt[field], run_directory)
    if json.loads(Path(receipt["status_path"]).read_bytes()) != receipt:
        raise ValueError("Current artifact status differs from result receipt")
    snapshot = json.loads((run_directory / "merge" / "workspace" / "workspace.json").read_bytes())
    if (manifest_identity_sha256 is not None
            and snapshot.get("identity", {}).get("manifest_sha256") != manifest_identity_sha256):
        raise ValueError("Workspace belongs to another canonical manifest")
    manager = DocumentDeliveryArtifacts(run_directory / "merge" / "delivery_artifacts",
                                        receipt["document_id"], scope=receipt["scope"])
    if not manager.matches_snapshot(snapshot, receipt):
        raise ValueError("Current workspace content differs from delivery")
    manager.read_page(snapshot, section="summary", max_chars=1,
                      expected_file_sha256=receipt["file_sha256"], receipt=receipt)
    if require_complete:
        if (result.get("complete") is not True or receipt.get("complete") is not True
                or receipt.get("incomplete") is not False or receipt.get("phase") != "completed"
                or receipt.get("requested_failure_marker") is not False):
            raise ValueError("Word exists but workflow did not complete")
        trace = json.loads((run_directory / "merge" / "document_workflow.json").read_bytes())
        from .document_workflow import DocumentWorkflow
        statement = receipt.get("agent_file_self_check", {}).get("statement")
        finish, origin = trace.get("accepted_finish"), trace.get("actual_finish_origin")
        binding = (statement or {}).get("verified_file_binding", {})
        semantic = (statement or {}).get("model_statement", {})
        # Quality relaxation is opt-in and bound to both workflow and workspace.
        # It never replaces native finish or inspection of this exact file.
        speed_gaps = (snapshot.get("identity", {}).get("quality_policy") == "speed_v1_explicit_gaps"
            and trace.get("identity", {}).get("settings", {}).get("allow_quality_gaps") is True
            and trace.get("identity", {}).get("run_identity", {}).get("speed_priority") is True
            and (statement or {}).get("quality_policy") == "speed_v1_explicit_gaps"
            and (statement or {}).get("exhaustive_quality_verified") is False)
        remaining = semantic.get("remaining_issues")
        issues_valid = (type(remaining) is list and all(type(item) is str and item.strip() for item in remaining)
                        and (not remaining or speed_gaps))
        if (not isinstance(finish, dict) or not isinstance(origin, dict)
                or not all(origin.get(key) for key in ("call_id", "request_id", "request_sha256", "response_sha256"))
                or receipt.get("actual_finish_origin") != origin
                or receipt.get("agent_file_self_check", {}).get("performed") is not True
                or trace.get("artifact_delivery") != receipt
                or (statement or {}).get("semantic_state_sha256") != DocumentWorkflow._semantic_identity(snapshot)
                or statement != trace.get("model_self_check")
                or (statement or {}).get("call_id") != origin["call_id"]
                or finish.get("model_self_check_sha256") != json_digest(statement)
                or binding.get("draft_sha256") != snapshot.get("draft_sha256")
                or binding.get("file_sha256") != receipt["file_sha256"]
                or finish.get("file_sha256") != receipt["file_sha256"]
                or semantic.get("inspected_export") is not True or semantic.get("complete") is not True
                or not issues_valid):
            raise ValueError("No current native finish and same-file model self-check")
    return receipt


def _stage_delivery(run_directory, result, document):
    try:
        receipt = verify_current_delivery(run_directory, result, require_complete=result.get("complete") is True,
                                          manifest_identity_sha256=document["manifest_identity_sha256"])
        target = Path(run_directory) / "staged_delivery" / ("candidate.docx" if result.get("complete") else "partial.docx")
        target.parent.mkdir(parents=True, exist_ok=False)
        raw = Path(receipt["docx_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt["file_sha256"]:
            raise ValueError("Word changed while staging")
        target.write_bytes(raw)
        if file_sha(target) != receipt["file_sha256"]:
            raise ValueError("Staged Word hash differs")
        return {"verified_complete": result.get("complete") is True, "path": str(target),
                "file_sha256": receipt["file_sha256"], "source_path": receipt["docx_path"],
                "source_receipt": receipt}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"verified_complete": False, "error": f"{type(error).__name__}: {error}",
                "preserved_partial_path": result.get("partial_docx")}


def _commit_delivery(record):
    """Promote validated bytes after settlement; keep expensive checks in workers."""
    staged = record["staged_delivery"]
    if not staged.get("path"):
        return deepcopy(staged)
    try:
        directory = Path(record["run_directory"])
        source = _within(staged["path"], directory / "staged_delivery")
        if file_sha(source) != staged["file_sha256"]:
            raise ValueError("Staged Word changed before publication")
        target = directory / "delivery" / ("report.docx" if staged["verified_complete"] else "partial.docx")
        target.parent.mkdir(parents=True, exist_ok=False)
        source.replace(target)
        return {**staged, "path": str(target)}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"verified_complete": False, "error": f"{type(error).__name__}: {error}",
                "preserved_partial_path": staged.get("path")}


def _default_worker(document, directory, model_config, settings):
    manifest, materials, _plan = _load_document(document, settings)
    if document.get("scope", "official-training") != "official-training":
        model_config = {**model_config, "input_scope": document["scope"]}
    return run_multiagent(manifest, directory, model_config, settings=settings,
                          materials=materials, materials_source_path=document["source_docx"]["path"])


def _worker_entry(document, directory, record_path, model_config, settings, accepted_at,
                  generation_deadline, execution_identity, worker):
    started = time.monotonic()
    deadline = min(accepted_at + settings.report_timeout, generation_deadline)
    try:
        if deadline <= started:
            raise TimeoutError("Document expired before worker startup; no model sent")
        effective = replace(settings, report_timeout=deadline - started)
        # Even injected CPU test workers receive the actual admission config.
        result = worker(document, Path(directory), model_config, effective)
        if not isinstance(result, dict):
            raise TypeError("Document worker must return a result object")
        staged_delivery = _stage_delivery(directory, result, document)
        # A worker's success cannot authorize publication while one of its
        # service requests is still unknown. The parent owns final settlement.
        record = {"status": "awaiting_settlement", "complete": False, "result": result,
                  "staged_delivery": staged_delivery}
    except Exception as error:
        record = {"status": "failed", "complete": False,
                  "error": {"type": type(error).__name__, "message": str(error)}}
    record.update(document_id=document["document_id"], document_key=document["document_key"],
                  document_sha256=document["document_sha256"], input_identity_sha256=document["input_identity_sha256"],
                  execution_identity_sha256=execution_identity, run_directory=str(directory),
                  accepted_at=accepted_at, worker_started_at=started, worker_finished_at=time.monotonic(),
                  pid=__import__("os").getpid(), new_attempt=True)
    atomic_json(Path(record_path), record)


def run_batch(prepared, directory, model_config, *, settings=MultiAgentSettings(),
              batch_settings=BatchSettings(), worker=None, quiescence_observer=None,
              quiescence_observer_factory=None, on_document_complete=None):
    """Schedule any positive list length. Default workers run the real pipeline."""
    # Recheck the entire runtime, not just each document as it reaches the queue.
    checked = prepare_batch(prepared["runtime_path"], prepared["runtime_file_sha256"], settings=settings)
    if checked != prepared:
        raise ValueError("Prepared batch identity/settings changed before execution")
    documents = prepared["documents"]
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "records").mkdir()
    started = time.monotonic()
    generation_seconds = math.ceil(len(documents) / batch_settings.document_workers) * settings.report_timeout
    generation_deadline = started + generation_seconds
    settlement_deadline = generation_deadline + batch_settings.settlement_grace_seconds
    execution_identity = json_digest({"model": model_config, "settings": asdict(settings)})
    atomic_json(directory / "input-identity.json", prepared)
    atomic_json(directory / "batch-settings.json", {"settings": asdict(settings), "batch": asdict(batch_settings),
        "model": model_config, "total_model_call_budget": len(documents) * settings.max_model_calls,
        "generation_deadline": generation_deadline, "settlement_deadline": settlement_deadline,
        "execution_identity_sha256": execution_identity, "resume_supported": False})
    results, pending, active = {}, list(documents), {}
    # Reuse is explicit and all-or-reject; a stale success cannot silently skip work.
    for document in list(pending):
        if "reuse_result" not in document:
            continue
        previous = json.loads(Path(document["reuse_result"]["path"]).read_bytes())
        if (previous.get("input_identity_sha256") != document["input_identity_sha256"]
                or previous.get("execution_identity_sha256") != execution_identity or not previous.get("complete")):
            raise ValueError("Explicit reuse receipt has different inputs/settings or is incomplete")
        receipt = verify_current_delivery(previous["run_directory"], previous["result"],
                                          manifest_identity_sha256=document["manifest_identity_sha256"])
        if file_sha(previous["publication"]["path"]) != receipt["file_sha256"]:
            raise ValueError("Previously published Word changed")
        results[document["document_key"]] = {**previous, "document_id": document["document_id"],
            "document_key": document["document_key"], "status": "reused_complete", "new_attempt": False}
        pending.remove(document)
    context = multiprocessing.get_context("spawn")
    if quiescence_observer_factory is not None:
        if quiescence_observer is not None:
            raise ValueError("Use one quiescence observer or factory")
        quiescence_observer = quiescence_observer_factory(directory)
    broker = BatchAdmissionBroker(directory, {"main": batch_settings.global_main_model_inflight,
        "summary": batch_settings.global_summary_model_inflight},
        [doc["document_key"] for doc in documents], generation_deadline, quiescence_observer,
        settlement_deadline=settlement_deadline)
    notified = set()

    def finish_settled_documents(snapshot, *, final=False):
        """Publish independently settled documents once, including explicit reuse.

        Before a new result reaches this helper its supervisor has exited and
        the broker has disabled further admission for that document. Completed
        siblings therefore need not wait for another document's idle probe.
        """
        requests = {}
        for request in snapshot["requests"]:
            requests.setdefault(request["document_id"], []).append(request)
        for document in documents:
            key = document["document_key"]
            if key not in results or key in notified:
                continue
            record = results[key]
            own_requests = requests.get(key, [])
            unsettled = any(row["state"] in {"pending", "active", "unknown"} for row in own_requests)
            if unsettled and not final:
                continue
            record["generation_settlement"] = {
                "batch_id": snapshot["batch_id"], "settled": not unsettled,
                "requests": [{"request_id": row["request_id"], "state": row["state"]}
                             for row in own_requests],
                "reused_prior_result": not record.get("new_attempt", False),
            }
            if unsettled:
                record.update(complete=False, status="unsettled_generation", service_cancellation_confirmed=False)
                record["publication"] = {"verified_complete": False, "reason": "unsettled_generation",
                    "preserved_partial_path": record.get("result", {}).get("partial_docx")}
            elif record.get("new_attempt") and "result" in record:
                publication = _commit_delivery(record)
                record.update(publication=publication, complete=publication["verified_complete"],
                              status="completed" if publication["verified_complete"] else "failed")
            if record.get("accepted_at") is not None:
                record["document_completion_seconds"] = time.monotonic() - record["accepted_at"]
            if record.get("batch_record_path") and record.get("new_attempt"):
                atomic_json(Path(record["batch_record_path"]), record)
            notified.add(key)
            if on_document_complete is not None:
                try:
                    on_document_complete(document, deepcopy(record))
                except Exception as error:
                    # A report's publication error must not cancel its siblings.
                    record.update(complete=False, status="publication_failed",
                        completion_callback_error={"type": type(error).__name__, "message": str(error)})
                    if record.get("batch_record_path") and record.get("new_attempt"):
                        atomic_json(Path(record["batch_record_path"]), record)

    try:
        finish_settled_documents(broker.snapshot())
        while pending or active:
            now = time.monotonic()
            while pending and len(active) < batch_settings.document_workers and now < generation_deadline:
                document = pending.pop(0)
                key = document["document_key"]
                attempt = uuid.uuid4().hex
                run_dir = directory / "documents" / document["document_sha256"] / settings.section_prompt_variant / attempt
                record_path = directory / "records" / (key + "-" + attempt + ".json")
                config = deepcopy(model_config)
                config["batch_admission"] = broker.config_for(key)
                config["batch_admission"]["deadline"] = min(now + settings.report_timeout, generation_deadline)
                process = context.Process(target=_worker_entry, args=(document, str(run_dir), str(record_path),
                    config, settings, now, generation_deadline, execution_identity, worker or _default_worker))
                try:
                    process.start()
                except Exception as error:
                    broker.mark_document_unknown(key, {"reason": "process_start_failed"})
                    results[key] = {"document_id": document["document_id"], "document_key": key,
                        "status": "worker_start_failed", "complete": False, "new_attempt": False,
                        "error": {"type": type(error).__name__, "message": str(error)}}
                    continue
                active[key] = {"process": process, "document": document, "accepted": now,
                               "deadline": min(now + settings.report_timeout, generation_deadline),
                               "record_path": record_path, "run_directory": str(run_dir)}
                now = time.monotonic()
            if now >= generation_deadline:
                for document in pending:
                    results[document["document_key"]] = {"document_id": document["document_id"],
                        "document_key": document["document_key"], "status": "expired_before_start",
                        "complete": False, "new_attempt": False, "model_calls": 0}
                pending.clear()
            for key, info in list(active.items()):
                process = info["process"]
                stop_at = min(info["deadline"] + batch_settings.settlement_grace_seconds, settlement_deadline)
                forced = process.is_alive() and now >= stop_at
                if forced:
                    # Only our own document supervisor is stopped. This is not
                    # evidence that a service generation or child has stopped.
                    process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1)
                if process.is_alive():
                    continue
                process.join(timeout=0)
                broker.mark_document_unknown(key, {"reason": "document_process_exit",
                    "exitcode": process.exitcode, "supervisor_terminated": forced})
                if info["record_path"].is_file() and not forced:
                    record = json.loads(info["record_path"].read_bytes())
                else:
                    record = {"document_id": info["document"]["document_id"], "document_key": key,
                        "status": "deadline_exceeded" if forced else "worker_crashed", "complete": False,
                        "new_attempt": True, "run_directory": info["run_directory"], "exitcode": process.exitcode,
                        "service_cancellation_confirmed": False}
                record.update(batch_queue_seconds=info["accepted"] - started,
                              document_wall_seconds=time.monotonic() - info["accepted"],
                              batch_record_path=str(info["record_path"]))
                results[key] = record
                atomic_json(info["record_path"], record)
                del active[key]
            finish_settled_documents(broker.snapshot())
            if pending or active:
                time.sleep(0.05)
        # No new jobs now; allow only bounded settlement of uncertain requests.
        end_settlement = min(settlement_deadline, time.monotonic() + batch_settings.settlement_grace_seconds)
        while time.monotonic() < end_settlement:
            snapshot = broker.snapshot()
            finish_settled_documents(snapshot)
            if not any(pool["occupied"] or pool["pending"] for pool in snapshot["pools"].values()):
                break
            time.sleep(0.05)
    finally:
        for key, info in active.items():
            broker.mark_document_unknown(key, {"reason": "batch_supervisor_exit"})
            if info["process"].is_alive():
                info["process"].terminate()
                info["process"].join(timeout=1)
        broker.close()
    admission = broker.snapshot()
    finish_settled_documents(admission, final=True)
    for record in results.values():
        if record.get("batch_record_path") and record.get("new_attempt"):
            atomic_json(Path(record["batch_record_path"]), record)
    elapsed = time.monotonic() - started
    ordered = [results[doc["document_key"]] for doc in documents]
    new_complete = sum(row["complete"] and row["new_attempt"] for row in ordered)
    result = {"schema": "cityagent-multiagent-batch-result-v1", "batch_id": broker.batch_id,
        "status": "completed" if all(row["complete"] for row in ordered) else "partial_failure",
        "complete": all(row["complete"] for row in ordered), "seconds": elapsed,
        "new_complete_reports": new_complete, "reused_complete_reports": sum(
            row["complete"] and not row["new_attempt"] for row in ordered),
        "new_complete_reports_per_hour": new_complete * 3600 / elapsed,
        "documents": ordered, "admission": admission,
        "throughput_scope": "this_batch_only_including_failures_queue_and_settlement",
        "semantic_correctness_verified": False, "resume_supported": False}
    atomic_json(directory / "batch-result.json", result)
    return result
