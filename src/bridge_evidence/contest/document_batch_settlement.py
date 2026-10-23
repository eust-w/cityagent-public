"""Bind stopped children to shared admission evidence before parent aggregation.

This observes the authenticated broker, never the model service. An earlier
missing DONE/usage stays unknown even when later service idle frees its lease.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import time

from .document_batch_admission import AdmissionClient
from .document_generation_settlement import _Refused, _read, _ref, _save, _trusted_children
from .run_state import json_digest


_INFLIGHT = {"pending", "active", "unknown"}
_STATES = _INFLIGHT | {"complete", "quiescent", "not_sent", "expired", "blocked"}


def _idle_proof(row, *, model, batch_id, refs):
    later = row.get("evidence", {}).get("later_service_idle")
    if not isinstance(later, dict) or later.get("confirmed_idle") is not True:
        raise _Refused("BATCH_QUIESCENT_PROOF_MISSING")
    proof, ref = later.get("proof"), later.get("proof_ref")
    if not isinstance(proof, dict) or not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise _Refused("BATCH_QUIESCENT_PROOF_INVALID")
    saved = _read(Path(ref["path"]), refs)
    if refs[-1] != ref or saved != proof:
        raise _Refused("BATCH_QUIESCENT_PROOF_CHANGED")
    samples, service = proof.get("samples"), proof.get("service", {})
    if (proof.get("schema") != "cityagent-batch-service-quiescence-v1"
            or proof.get("method") != "later_whole_service_idle"
            or proof.get("pool") != row["pool"] or service.get("pool") != row["pool"]
            or service.get("model") != model or proof.get("confirmed_idle") is not True
            or proof.get("same_service_process") is not True
            or proof.get("cancellation_confirmed") is not False
            or not proof.get("process_identity_sha256")
            or json_digest(service.get("process_identity")) != proof["process_identity_sha256"]
            or not isinstance(samples, list) or len(samples) < 2):
        raise _Refused("BATCH_QUIESCENT_SERVICE_IDENTITY_MISMATCH")
    last = samples[-2:]
    if (any(s.get("idle") is not True or s.get("model_identity", {}).get("model") != model
            or s.get("model_identity", {}).get("max_model_len") != 262144 for s in last)
            or json_digest(last[0]["model_identity"]) != json_digest(last[1]["model_identity"])
            or proof.get("model_identity_sha256") != json_digest(last[1]["model_identity"])
            or last[1].get("started_after_seconds", -1) - last[0].get("completed_after_seconds", 0) < .5):
        raise _Refused("BATCH_QUIESCENT_SAMPLES_INVALID")
    original_ref = proof.get("snapshot_ref")
    if not isinstance(original_ref, dict) or not isinstance(original_ref.get("path"), str):
        raise _Refused("BATCH_QUIESCENT_SNAPSHOT_MISSING")
    original = _read(Path(original_ref["path"]), refs)
    if refs[-1] != original_ref or original.get("batch_id") != batch_id:
        raise _Refused("BATCH_QUIESCENT_SNAPSHOT_MISMATCH")
    matches = [r for r in original.get("requests", []) if r.get("request_id") == row["request_id"]]
    if (len(matches) != 1 or matches[0].get("state") != "unknown"
            or any(matches[0].get(key) != row.get(key) for key in
                   ("batch_id", "document_id", "agent_id", "pool", "lease_id"))
            or original.get("pools", {}).get(row["pool"], {}).get("active") != 0):
        raise _Refused("BATCH_QUIESCENT_REQUEST_ASSOCIATION_MISMATCH")
    return ref


def observe_batch_child_settlement(dispatch, *, run_directory, model, manifest_sha256,
        deadline, batch_config, client=None, clock=time.monotonic, sleep=time.sleep):
    """Wait at most 900 seconds and never past the document/batch deadline.

    ``client.snapshot(timeout=...)`` can be injected for CPU-only tests. All
    process/artifact checks use the existing stopped-child verification chain.
    """
    directory = Path(run_directory).absolute()
    output = directory / "post-child-batch-settlement"
    output.mkdir(exist_ok=False)
    started = clock()
    receipt = {"schema": "cityagent-post-child-batch-settlement-v1", "outcome": "unknown",
        "may_continue_aggregation": False, "proof_kind": "shared_admission_child_settlement",
        "cancellation_confirmed": False, "prior_requests_reclassified": False,
        "model_retry_permitted": False, "historical_done_and_usage": "Unchanged; missing values remain unknown.",
        "started_at_monotonic": started, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "unknown_calls": [], "unknown_call_ids": [], "input_refs": [], "observations": [], "error_code": None}
    receipt_path = output / "receipt.json"
    try:
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)
                or not isinstance(batch_config, dict)
                or any(not isinstance(batch_config.get(k), str) or not batch_config[k]
                       for k in ("batch_id", "document_id"))
                or type(batch_config.get("deadline")) not in (int, float)
                or not math.isfinite(batch_config["deadline"])):
            raise _Refused("BATCH_SETTLEMENT_CONFIGURATION_INVALID")
        stop_at = min(deadline, batch_config["deadline"], started + 900)
        receipt.update(batch_id=batch_config["batch_id"], document_id=batch_config["document_id"],
            shared_deadline_monotonic=deadline, observation_deadline_monotonic=stop_at)
        refs, pids, unknowns = _trusted_children(dispatch, directory, model, manifest_sha256, deadline)
        pairs = [(u["task_id"], u["call_id"]) for u in unknowns]
        if len(set(pairs)) != len(pairs):
            raise _Refused("BATCH_DUPLICATE_UNKNOWN_CALL")
        receipt.update(input_refs=refs, unknown_calls=unknowns,
            unknown_call_ids=list(dict.fromkeys(u["call_id"] for u in unknowns)))
        agents = {f"{pool}:{task}" for task in dispatch["task_ids"] for pool in ("main", "summary")}
        observer = client if client is not None else AdmissionClient(batch_config)
        associations = {}
        while clock() < stop_at:
            snapshot = observer.snapshot(timeout=min(5, stop_at - clock()))
            snap_path = output / f"snapshot-{len(receipt['observations']):04d}.json"
            _save(snap_path, snapshot)
            sample = {"snapshot_ref": _ref(snap_path), "at_monotonic": clock(), "associations": []}
            receipt["observations"].append(sample)
            if clock() >= stop_at:
                raise _Refused("BATCH_SETTLEMENT_DEADLINE_EXCEEDED")
            if (not isinstance(snapshot, dict) or snapshot.get("batch_id") != batch_config["batch_id"]
                    or not isinstance(snapshot.get("requests"), list) or not isinstance(snapshot.get("pools"), dict)):
                raise _Refused("BATCH_SNAPSHOT_IDENTITY_MISMATCH")
            if snapshot.get("closed") is True or batch_config["document_id"] in snapshot.get("stopped_documents", []):
                raise _Refused("BATCH_DOCUMENT_OR_BROKER_STOPPED")
            if any(row.get("document_id") == batch_config["document_id"] and row.get("state") in _INFLIGHT
                   and row.get("agent_id") not in agents for row in snapshot["requests"]):
                raise _Refused("BATCH_UNEXPECTED_DOCUMENT_INFLIGHT")
            children = [row for row in snapshot["requests"] if row.get("document_id") == batch_config["document_id"]
                        and row.get("agent_id") in agents]
            if any(row.get("batch_id") != batch_config["batch_id"] or row.get("state") not in _STATES
                   or row.get("pool") != row["agent_id"].split(":", 1)[0] for row in children):
                raise _Refused("BATCH_CHILD_ROW_IDENTITY_MISMATCH")
            ids = [row.get("request_id") for row in children]
            if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
                raise _Refused("BATCH_CHILD_REQUEST_ID_INVALID")
            waiting = [row for row in children if row["state"] in _INFLIGHT]
            sample["inflight_child_request_ids"] = [r["request_id"] for r in waiting]
            proof_refs = []
            for task_id, call_id in pairs:
                matches = [row for row in children if row["agent_id"] == "main:" + task_id
                    and row.get("settlement_evidence", {}).get("call_id") == call_id]
                if len(matches) != 1:
                    raise _Refused("BATCH_UNKNOWN_CALL_ASSOCIATION_MISSING_OR_AMBIGUOUS")
                row = matches[0]
                key = (task_id, call_id)
                identity = (row["request_id"], row.get("lease_id"))
                if not isinstance(identity[1], str) or not identity[1]:
                    raise _Refused("BATCH_CHILD_LEASE_ID_MISSING")
                if key in associations and associations[key] != identity:
                    raise _Refused("BATCH_UNKNOWN_CALL_ASSOCIATION_CHANGED")
                associations[key] = identity
                if row["settlement_evidence"].get("model") != model:
                    raise _Refused("BATCH_UNKNOWN_CALL_MODEL_MISMATCH")
                binding = {"task_id": task_id, "call_id": call_id, "request_id": identity[0],
                           "lease_id": identity[1], "state": row["state"]}
                sample["associations"].append(binding)
                if row["state"] == "quiescent":
                    binding["idle_proof_ref"] = _idle_proof(row, model=model,
                        batch_id=batch_config["batch_id"], refs=proof_refs)
                elif row["state"] == "complete":
                    completion = row.get("completion_evidence")
                    if (not isinstance(completion, dict) or completion.get("call_id") != call_id
                            or completion.get("model") != model):
                        raise _Refused("BATCH_COMPLETE_CALL_IDENTITY_MISMATCH")
                elif row["state"] not in _INFLIGHT:
                    raise _Refused("BATCH_UNKNOWN_CALL_NOT_PROVEN_SETTLED")
            sample["proof_refs"] = proof_refs
            relevant_pools = {"main"} | {r["pool"] for r in waiting}
            for pool in relevant_pools:
                state = snapshot["pools"].get(pool)
                if not isinstance(state, dict):
                    raise _Refused("BATCH_POOL_STATE_MISSING")
                if state.get("blocked_reason"):
                    receipt["broker_blocked_reason"] = state["blocked_reason"]
                    raise _Refused("BATCH_POOL_RECOVERY_BLOCKED")
            _save(receipt_path, receipt)
            if not waiting:
                current_refs, current_pids, current_unknowns = _trusted_children(
                    dispatch, directory, model, manifest_sha256, deadline)
                if current_refs != refs or current_pids != pids or current_unknowns != unknowns:
                    raise _Refused("BATCH_TRUSTED_CHILDREN_CHANGED")
                if any(_ref(Path(ref["path"])) != ref for ref in proof_refs):
                    raise _Refused("BATCH_QUIESCENT_PROOF_CHANGED")
                if clock() >= stop_at:
                    raise _Refused("BATCH_SETTLEMENT_DEADLINE_EXCEEDED")
                receipt.update(outcome="batch_children_settled", may_continue_aggregation=True)
                break
            sleep(min(.25, max(0, stop_at - clock())))
        if not receipt["may_continue_aggregation"]:
            raise _Refused("BATCH_SETTLEMENT_DEADLINE_EXCEEDED")
    except _Refused as exc:
        receipt["error_code"] = str(exc)
    except Exception as exc:
        receipt.update(error_code="BATCH_SETTLEMENT_CHECK_ERROR", error_type=type(exc).__name__)
    receipt.update(finished_at_monotonic=clock(), finished_at_utc=datetime.now(timezone.utc).isoformat())
    _save(receipt_path, receipt)
    if receipt["may_continue_aggregation"] and clock() >= receipt["observation_deadline_monotonic"]:
        receipt.update(outcome="unknown", may_continue_aggregation=False,
                       error_code="BATCH_SETTLEMENT_DEADLINE_EXCEEDED")
        _save(receipt_path, receipt)
    return {**receipt, "receipt_ref": _ref(receipt_path)}
