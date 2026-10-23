"""Real child artifact checks with synthetic broker snapshots; no model calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from bridge_evidence.contest import document_generation_settlement as original
from bridge_evidence.contest.document_batch_quiescence import BatchServiceQuiescence
from bridge_evidence.contest.document_batch_settlement import observe_batch_child_settlement
from test_document_generation_settlement import stopped_dispatch, Clock, models, metrics, MODEL, MANIFEST, save


CONFIG = {"url": "http://127.0.0.1:9", "batch_id": "trusted-batch", "document_id": "own-document",
          "auth_token": "test-auth-do-not-persist", "deadline": 120.0}


def row(state="unknown", **overrides):
    value = {"batch_id": CONFIG["batch_id"], "document_id": CONFIG["document_id"],
        "request_id": "lease-request", "lease_id": "lease-token", "agent_id": "main:one", "pool": "main",
        "state": state, "settlement_state": "unknown", "settlement_evidence": {
            "call_id": "original-unknown-call", "model": MODEL, "done_received": False}}
    if state == "complete":
        value["completion_evidence"] = {"call_id": "original-unknown-call", "model": MODEL, "done_received": True}
    value.update(overrides)
    return value


def snapshot(*rows, blocked_reason=None):
    return {"batch_id": CONFIG["batch_id"], "closed": False, "stopped_documents": [],
        "pools": {pool: {"active": sum(r["pool"] == pool and r["state"] == "active" for r in rows),
                          "unknown": sum(r["pool"] == pool and r["state"] == "unknown" for r in rows),
                          "blocked": bool(blocked_reason), "blocked_reason": blocked_reason}
                  for pool in ("main", "summary")},
        "requests": list(rows)}


class Client:
    def __init__(self, snapshots, *, on_snapshot=None):
        self.snapshots = snapshots
        self.calls = []
        self.on_snapshot = on_snapshot

    def snapshot(self, *, timeout):
        assert 0 < timeout <= 5
        if self.on_snapshot:
            self.on_snapshot()
        index = len(self.calls)
        self.calls.append(timeout)
        return deepcopy(self.snapshots[min(index, len(self.snapshots) - 1)])


def observe(stopped_dispatch, snapshots, *, client=None, clock=None, config=None, **kwargs):
    directory, dispatch = stopped_dispatch
    clock = clock or Clock()
    client = client or Client(snapshots)
    result = observe_batch_child_settlement(dispatch, run_directory=directory, model=MODEL,
        manifest_sha256=MANIFEST, deadline=dispatch["deadline_monotonic"], batch_config=config or CONFIG,
        client=client, clock=clock, sleep=clock.sleep, **kwargs)
    return result, client, clock


def idle_row(tmp_path, monkeypatch):
    monkeypatch.setenv("CITYAGENT_STOP_OBSERVER_SERVER_PID", "123")
    clock = Clock()
    def fetch(url, *, timeout, headers):
        return 200, models() if url.endswith("/models") else metrics()
    probe = BatchServiceQuiescence({"base_url": "http://127.0.0.1:18110", "model": MODEL},
        tmp_path / "broker-service-proof", clock=clock, sleep=clock.sleep, fetch=fetch,
        process_identity=lambda pid, port: {"pid": pid, "starttime_ticks": "123", "cmdline_sha256": "a" * 64})
    proof = probe("main", 120.0, snapshot(row()))
    assert proof["confirmed_idle"]
    return row("quiescent", evidence={"later_service_idle": proof,
        "request_completion_known": False, "usage_known": False})


def test_complete_and_quiescent_evidence_preserve_original_child_result(stopped_dispatch, tmp_path, monkeypatch):
    directory, _ = stopped_dispatch
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    settled = idle_row(tmp_path, monkeypatch)
    foreign = row("active", document_id="other-document", request_id="foreign-request", lease_id="foreign-lease")
    result, client, _ = observe(stopped_dispatch, [snapshot(row()), snapshot(settled, foreign)])
    assert result["may_continue_aggregation"] is True and result["outcome"] == "batch_children_settled"
    assert len(client.calls) == 2 and result["unknown_call_ids"] == ["original-unknown-call"]
    assert result["cancellation_confirmed"] is False and result["prior_requests_reclassified"] is False
    assert result["model_retry_permitted"] is False
    assert all(p.read_bytes() == raw for p, raw in before.items())
    child = json.loads((directory / "tasks/one/result.json").read_text())
    assert child["status"] == "failed" and child["usage"] is None and child["done_received"] is False
    path = Path(result["receipt_ref"]["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result["receipt_ref"]["sha256"]
    assert all(b"test-auth-do-not-persist" not in p.read_bytes()
               for p in path.parent.rglob("*") if p.is_file())


def test_matching_late_complete_allows_aggregation_without_service_poll(stopped_dispatch):
    result, client, _ = observe(stopped_dispatch, [snapshot(row("complete"))])
    assert result["may_continue_aggregation"] and len(client.calls) == 1
    assert result["observations"][0]["associations"][0]["state"] == "complete"


@pytest.mark.parametrize("change", ["document", "agent", "call", "missing_call", "wrong_field", "duplicate", "batch", "model", "missing_model"])
def test_exact_unknown_call_association_is_required(stopped_dispatch, change):
    candidate = row("complete")
    if change == "document": candidate["document_id"] = "foreign-document"
    if change == "agent": candidate["agent_id"] = "main:wrong-task"
    if change == "call": candidate["settlement_evidence"]["call_id"] = "foreign-call"
    if change == "missing_call": candidate.pop("settlement_evidence")
    if change == "wrong_field":
        candidate["evidence"] = candidate.pop("settlement_evidence")
    if change == "batch": candidate["batch_id"] = "foreign-batch"
    if change == "model": candidate["settlement_evidence"]["model"] = "foreign-model"
    if change == "missing_model": candidate["settlement_evidence"].pop("model")
    rows = [candidate]
    if change == "duplicate": rows.append(row("complete", request_id="another-lease", lease_id="another-token"))
    result, _, _ = observe(stopped_dispatch, [snapshot(*rows)])
    assert not result["may_continue_aggregation"] and result["error_code"] is not None


@pytest.mark.parametrize("change", ["call", "missing", "model", "missing_model"])
def test_late_complete_must_describe_the_same_original_call(stopped_dispatch, change):
    candidate = row("complete")
    if change == "call": candidate["completion_evidence"]["call_id"] = "other"
    if change == "missing": candidate.pop("completion_evidence")
    if change == "model": candidate["completion_evidence"]["model"] = "other"
    if change == "missing_model": candidate["completion_evidence"].pop("model")
    result, _, _ = observe(stopped_dispatch, [snapshot(candidate)])
    assert result["error_code"] == "BATCH_COMPLETE_CALL_IDENTITY_MISMATCH"


@pytest.mark.parametrize("state", ["not_sent", "expired", "blocked"])
def test_old_attempt_cannot_be_treated_as_never_sent(stopped_dispatch, state):
    result, _, _ = observe(stopped_dispatch, [snapshot(row(state))])
    assert result["error_code"] == "BATCH_UNKNOWN_CALL_NOT_PROVEN_SETTLED"


def test_additional_child_summary_and_main_requests_must_clear(stopped_dispatch):
    extra_main = row("active", request_id="extra-main", lease_id="extra-main-lease", settlement_evidence={})
    extra_summary = row("unknown", request_id="extra-summary", lease_id="extra-summary-lease", pool="summary",
                        agent_id="summary:one", settlement_evidence={"call_id": "summary-call"})
    final_main = {**extra_main, "state": "complete"}
    final_summary = {**extra_summary, "state": "complete"}
    result, client, _ = observe(stopped_dispatch, [snapshot(row("complete"), extra_main, extra_summary),
        snapshot(row("complete"), final_main, final_summary)])
    assert result["may_continue_aggregation"] and len(client.calls) == 2
    assert result["observations"][0]["inflight_child_request_ids"] == ["extra-main", "extra-summary"]


def test_identity_cannot_move_to_another_lease_while_waiting(stopped_dispatch):
    result, _, _ = observe(stopped_dispatch, [snapshot(row()),
        snapshot(row("complete", request_id="replacement", lease_id="replacement-token"))])
    assert result["error_code"] == "BATCH_UNKNOWN_CALL_ASSOCIATION_CHANGED"


@pytest.mark.parametrize("reason", ["quiescence_observer_unavailable", "quiescence_deadline_elapsed", "quiescence_attempts_exhausted"])
def test_broker_terminal_block_ends_wait_early(stopped_dispatch, reason):
    result, client, clock = observe(stopped_dispatch, [snapshot(row(), blocked_reason=reason)])
    assert result["error_code"] == "BATCH_POOL_RECOVERY_BLOCKED" and result["broker_blocked_reason"] == reason
    assert len(client.calls) == 1 and clock.sleeps == []


def test_transient_blocked_flag_alone_is_not_a_terminal_failure(stopped_dispatch):
    value = snapshot(row())
    value["pools"]["main"]["blocked"] = True
    result, client, _ = observe(stopped_dispatch, [value, snapshot(row("complete"))])
    assert result["may_continue_aggregation"] and len(client.calls) == 2


def test_wait_is_bounded_by_document_deadline_and_rpc_remaining_time(stopped_dispatch):
    clock = Clock()
    clock.value = 119.8
    result, client, _ = observe(stopped_dispatch, [snapshot(row())], clock=clock)
    assert result["error_code"] == "BATCH_SETTLEMENT_DEADLINE_EXCEEDED"
    assert clock.value == 120 and max(client.calls) <= .2 + 1e-9


def test_expired_deadline_never_calls_broker(stopped_dispatch):
    clock = Clock()
    clock.value = 120
    result, client, _ = observe(stopped_dispatch, [], clock=clock)
    assert result["error_code"] == "BATCH_SETTLEMENT_DEADLINE_EXCEEDED" and client.calls == []


def test_long_document_has_a_separate_nine_hundred_second_cap(stopped_dispatch):
    directory, dispatch = stopped_dispatch
    dispatch["deadline_monotonic"] = 5000.0
    save(directory / "dispatch-summary.json", dispatch)
    claim = json.loads((directory / "dispatch-claim.json").read_text())
    claim["deadline_monotonic"] = 5000.0
    save(directory / "dispatch-claim.json", claim)
    clock = Clock()
    # Advance at each poll without performing any real wait.
    client = Client([snapshot(row())], on_snapshot=lambda: setattr(clock, "value", clock.value + 4))
    result, _, _ = observe(stopped_dispatch, [], client=client, clock=clock, config={**CONFIG, "deadline": 5000.0})
    assert result["observation_deadline_monotonic"] == 1000.0
    assert result["error_code"] == "BATCH_SETTLEMENT_DEADLINE_EXCEEDED"


def test_mutated_artifact_during_broker_observation_is_rejected(stopped_dispatch):
    directory, _ = stopped_dispatch
    def mutate():
        path = directory / "tasks/one/result.json"
        value = json.loads(path.read_text()); value["saved_rows"].append({"text": "changed"}); save(path, value)
    result, _, _ = observe(stopped_dispatch, [], client=Client([snapshot(row("complete"))], on_snapshot=mutate))
    assert not result["may_continue_aggregation"] and result["error_code"] == "CHILD_RESULT_IDENTITY_MISMATCH"


def test_stopped_child_pid_must_still_be_stopped_at_admission(stopped_dispatch, monkeypatch):
    calls = []
    def stopped(pid):
        calls.append(pid)
        return len(calls) == 1
    monkeypatch.setattr(original, "_process_stopped", stopped)
    result, _, _ = observe(stopped_dispatch, [snapshot(row("complete"))])
    assert result["error_code"] == "OWNED_CHILD_OR_PROCESS_GROUP_STILL_EXISTS"


def test_untrusted_child_artifacts_are_checked_before_any_broker_rpc(stopped_dispatch):
    directory, _ = stopped_dispatch
    (directory / "tasks/one/process-receipt.json").unlink()
    result, client, _ = observe(stopped_dispatch, [snapshot(row("complete"))])
    assert not result["may_continue_aggregation"] and client.calls == []


@pytest.mark.parametrize("change", ["pool", "model", "foreign_request", "raw_file", "no_proof"])
def test_quiescent_requires_matching_persisted_idle_proof(stopped_dispatch, tmp_path, monkeypatch, change):
    candidate = idle_row(tmp_path, monkeypatch)
    later = candidate["evidence"]["later_service_idle"]
    if change == "pool": later["proof"]["pool"] = "summary"
    if change == "model": later["proof"]["service"]["model"] = "foreign"
    if change == "foreign_request": candidate["request_id"] = "other-request"
    if change == "raw_file": Path(later["proof_ref"]["path"]).write_text("{}")
    if change == "no_proof": candidate["evidence"] = {"later_service_idle": {"confirmed_idle": True}}
    result, _, _ = observe(stopped_dispatch, [snapshot(candidate)])
    assert not result["may_continue_aggregation"] and result["error_code"] is not None


@pytest.mark.parametrize("change", ["batch", "closed", "stopped_document"])
def test_foreign_or_closed_broker_cannot_authorize_parent(stopped_dispatch, change):
    value = snapshot(row("complete"))
    if change == "batch": value["batch_id"] = "foreign"
    if change == "closed": value["closed"] = True
    if change == "stopped_document": value["stopped_documents"] = [CONFIG["document_id"]]
    result, _, _ = observe(stopped_dispatch, [value])
    assert not result["may_continue_aggregation"]


def test_broker_error_and_late_success_are_not_admission_evidence(stopped_dispatch):
    clock = Clock()
    client = Client([snapshot(row("complete"))], on_snapshot=lambda: setattr(clock, "value", 121))
    result, _, _ = observe(stopped_dispatch, [], client=client, clock=clock)
    assert result["error_code"] == "BATCH_SETTLEMENT_DEADLINE_EXCEEDED"


@pytest.mark.parametrize("state", ["pending", "active", "unknown"])
def test_unexpected_inflight_agent_in_same_document_is_refused(stopped_dispatch, state):
    unexpected = row(state, agent_id="main:not-in-child-plan", request_id="rogue-request", lease_id="rogue-lease")
    result, _, _ = observe(stopped_dispatch, [snapshot(row("complete"), unexpected)])
    assert result["error_code"] == "BATCH_UNEXPECTED_DOCUMENT_INFLIGHT"
