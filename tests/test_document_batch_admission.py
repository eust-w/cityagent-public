"""Local HTTP + spawned processes only: no model, training data or GPU."""
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
import os
import socket
import struct
import threading
import time

import pytest

from bridge_evidence.contest.document_batch_admission import (
    AdmissionBlocked, AdmissionClient, AdmissionDeadlineExceeded, AdmissionError, AdmissionUnavailable,
    BatchAdmissionBroker,
)


def _parallel_worker(config, pool, agent, start, lock, counters, peaks, output, ready=None, rounds=2):
    try:
        client = AdmissionClient(config)
        if ready is not None:
            ready.put(agent)
        start.wait(10)
        for attempt in range(rounds):
            lease = client.acquire(pool, agent, time.monotonic() + 10)
            index = 0 if pool == "main" else 1
            with lock:
                counters[index] += 1
                peaks[index] = max(peaks[index], counters[index])
            time.sleep(0.05)
            with lock:
                counters[index] -= 1
            client.settle(lease, "complete", {"response_id": f"{agent}-{attempt}", "done": True})
        output.put((agent, "ok"))
    except BaseException as exc:
        output.put((agent, type(exc).__name__ + ": " + str(exc)))


def _crash_after_lease(config, output):
    lease = AdmissionClient(config).acquire("main", "crashing-child", time.monotonic() + 10)
    output.send(lease)
    output.close()
    os._exit(17)


def _wait(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Expected broker state did not arrive")


def _broker(tmp_path, **kwargs):
    return BatchAdmissionBroker(tmp_path, kwargs.pop("limits", {"main": 2, "summary": 1}),
        kwargs.pop("document_ids", ["a", "b", "c"]), time.monotonic() + 20, **kwargs)


def test_spawned_processes_share_main_and_summary_caps_and_preserve_request_identity(tmp_path):
    context = mp.get_context("spawn")
    lock, start, output = context.Lock(), context.Event(), context.Queue()
    counters, peaks = context.Array("i", [0, 0]), context.Array("i", [0, 0])
    processes = []
    with _broker(tmp_path) as broker:
        try:
            for i in range(8):
                pool = "main" if i < 6 else "summary"
                process = context.Process(target=_parallel_worker,
                    args=(broker.config_for("abc"[i % 3]), pool, f"agent-{i}", start,
                          lock, counters, peaks, output))
                process.start()
                processes.append(process)
            start.set()
            results = [output.get(timeout=15) for _ in processes]
            for process in processes:
                process.join(3)
                assert process.exitcode == 0
            assert all(state == "ok" for _, state in results), results
            assert list(peaks) == [2, 1]
            state = broker.snapshot()
            assert len(state["requests"]) == 16
            assert {r["document_id"] for r in state["requests"]} == {"a", "b", "c"}
            assert len({r["lease_id"] for r in state["requests"]}) == 16
            assert all(r["state"] == "complete" for r in state["requests"])
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()  # Only our test-owned synthetic workers.
                process.join(3)
    events = [json.loads(line) for line in (tmp_path / "admission-events.jsonl").read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert sum(event["event"] == "enqueued" for event in events) == 16
    occupancy = {"main": 0, "summary": 0}
    for event in events:
        if event["event"] != "transition":
            continue
        row = event["request"]
        if row["state"] == "active":
            occupancy[row["pool"]] += 1
        elif event["previous_state"] == "active":
            occupancy[row["pool"]] -= 1
        assert 0 <= occupancy[row["pool"]] <= (2 if row["pool"] == "main" else 1)


def test_queued_documents_round_robin_and_within_document_fifo(tmp_path):
    with _broker(tmp_path, limits={"main": 1}) as broker, ThreadPoolExecutor(max_workers=6) as threads:
        clients = {doc: AdmissionClient(broker.config_for(doc)) for doc in "abc"}
        blocker = clients["a"].acquire("main", "blocker", time.monotonic() + 10)
        futures = []
        for doc, agent in [("a", "a1"), ("a", "a2"), ("b", "b1"), ("b", "b2"), ("c", "c1"), ("c", "c2")]:
            futures.append((doc, agent, threads.submit(clients[doc].acquire, "main", agent, time.monotonic() + 10)))
            _wait(lambda: broker.snapshot()["pools"]["main"]["pending"] == len(futures))
        clients["a"].settle(blocker, "complete", {"done": True})
        for expected in ["b1", "c1", "a1", "b2", "c2", "a2"]:
            doc, _, future = next(item for item in futures if item[1] == expected)
            lease = future.result(timeout=3)
            assert lease["agent_id"] == expected
            clients[doc].settle(lease, "complete", {"done": True})
        grants = [json.loads(line)["request"]["agent_id"]
            for line in (tmp_path / "admission-events.jsonl").read_text().splitlines()
            if json.loads(line)["event"] == "transition" and json.loads(line)["request"]["state"] == "active"]
        assert grants == ["blocker", "b1", "c1", "a1", "b2", "c2", "a2"]


def test_queue_deadline_does_not_admit_and_independent_pool_can_proceed(tmp_path):
    with _broker(tmp_path, limits={"main": 1, "summary": 1}) as broker:
        a, b = AdmissionClient(broker.config_for("a")), AdmissionClient(broker.config_for("b"))
        active = a.acquire("main", "main", time.monotonic() + 10)
        with pytest.raises(AdmissionDeadlineExceeded):
            b.acquire("main", "expires", time.monotonic() + 0.1)
        expired = next(r for r in broker.snapshot()["requests"] if r["agent_id"] == "expires")
        assert expired["state"] in {"expired", "not_sent"} and "granted_at" not in expired
        with pytest.raises(AdmissionDeadlineExceeded):
            b.acquire("main", "already-expired", time.monotonic() - 1)
        assert not any(r["agent_id"] == "already-expired" for r in broker.snapshot()["requests"])
        summary = b.acquire("summary", "summary", time.monotonic() + 2)
        b.settle(summary, "complete", {"done": True})
        a.settle(active, "complete", {"done": True})


def test_unknown_blocks_only_its_pool_and_never_becomes_not_sent(tmp_path):
    with _broker(tmp_path) as broker:
        a, b = AdmissionClient(broker.config_for("a")), AdmissionClient(broker.config_for("b"))
        lease = a.acquire("main", "uncertain", time.monotonic() + 10)
        proof = {"disconnected": True, "server_stop_confirmed": False}
        a.settle(lease, "unknown", proof)
        assert a.settle(lease, "unknown", proof)["state"] == "unknown"
        with pytest.raises(AdmissionError, match="cannot be reclassified"):
            a.settle(lease, "not_sent", {"not_sent": True})
        with pytest.raises(AdmissionBlocked, match="quiescence_observer_unavailable"):
            b.acquire("main", "blocked", time.monotonic() + 0.1)
        summary = b.acquire("summary", "compressor", time.monotonic() + 2)
        b.settle(summary, "not_sent", {"preflight_failed": True})
        state = broker.snapshot()
        assert state["pools"]["main"]["unknown"] == 1
        assert state["pools"]["main"]["occupied"] == 1 and state["pools"]["main"]["blocked"]


def test_observer_waits_for_known_active_then_releases_only_with_later_idle_proof(tmp_path):
    observed = []
    def observer(pool, deadline, snapshot):
        observed.append(snapshot)
        assert pool == "main" and deadline > time.monotonic()
        assert snapshot["pools"][pool]["active"] == 0
        return {"confirmed_idle": True, "service_identity": "synthetic-owned-server", "samples": [0, 0]}

    with _broker(tmp_path, quiescence_observer=observer) as broker:
        a, b = AdmissionClient(broker.config_for("a")), AdmissionClient(broker.config_for("b"))
        lost = a.acquire("main", "lost", time.monotonic() + 10)
        active = b.acquire("main", "still-running", time.monotonic() + 10)
        original_evidence = {"connection_lost": True, "call_id": "child-original-call",
                             "request_id": "upstream-original-request", "usage_known": False}
        a.settle(lost, "unknown", original_evidence)
        time.sleep(0.08)
        assert observed == []
        b.settle(active, "complete", {"response_id": "known-done"})
        _wait(lambda: broker.snapshot()["pools"]["main"]["unknown"] == 0)
        row = next(r for r in broker.snapshot()["requests"] if r["request_id"] == lost["request_id"])
        assert row["state"] == "quiescent"
        assert row["evidence"]["request_completion_known"] is False
        assert row["evidence"]["usage_known"] is False
        assert row["settlement_state"] == "unknown" and row["settlement_evidence"] == original_evidence
        assert a.settle(lost, "unknown", original_evidence)["state"] == "quiescent"
        late = a.settle(lost, "complete", {"response_id": "late-original-done", "done": True})
        assert late["state"] == "complete" and late["settlement_evidence"] == original_evidence
        assert late["completion_evidence"] == {"response_id": "late-original-done", "done": True}
        later = b.acquire("main", "after-idle", time.monotonic() + 2)
        b.settle(later, "complete", {"done": True})


@pytest.mark.parametrize("mode", ["false", "error", "truthy", "late"])
def test_missing_invalid_or_late_idle_proof_keeps_unknown_and_observation_is_bounded(tmp_path, mode):
    calls = []
    def observer(pool, deadline, snapshot):
        calls.append(pool)
        if mode == "error":
            raise RuntimeError("synthetic unavailable metrics")
        if mode == "late":
            time.sleep(0.06)
            return {"confirmed_idle": True}
        if mode == "truthy":
            return {"confirmed_idle": 1}
        return {"confirmed_idle": False}

    with _broker(tmp_path, quiescence_observer=observer, observer_max_attempts=2,
                 observer_retry_interval=0.01, observer_timeout=0.03 if mode == "late" else 1) as broker:
        client = AdmissionClient(broker.config_for("a"))
        lease = client.acquire("main", "lost", time.monotonic() + 10)
        client.settle(lease, "unknown", {"disconnected": True})
        expected_calls = 1 if mode == "late" else 2
        _wait(lambda: len(calls) == expected_calls and not broker.snapshot()["pools"]["main"]["observing"])
        time.sleep(0.06)
        assert len(calls) == expected_calls
        assert broker.snapshot()["pools"]["main"]["unknown"] == 1


def test_process_exit_does_not_release_lease_parent_marks_unknown_and_close_preserves_it(tmp_path):
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    broker = _broker(tmp_path)
    child = context.Process(target=_crash_after_lease, args=(broker.config_for("a"), send))
    try:
        child.start()
        assert receive.poll(5)
        lease = receive.recv()
        child.join(3)
        assert child.exitcode == 17
        assert broker.snapshot()["pools"]["main"]["active"] == 1
        broker.mark_document_unknown("a", {"exitcode": 17})
        assert broker.snapshot()["pools"]["main"]["unknown"] == 1
        with pytest.raises(AdmissionError, match="worker has stopped"):
            AdmissionClient(broker.config_for("a")).acquire("summary", "orphan-child", time.monotonic() + 3)
        other = AdmissionClient(broker.config_for("b")).acquire("summary", "other", time.monotonic() + 3)
        assert other["state"] == "active"
    finally:
        broker.close()
        if child.is_alive():
            child.terminate()
        child.join(3)
        send.close()
        receive.close()
    state = broker.snapshot()
    assert state["closed"]
    assert next(r for r in state["requests"] if r["lease_id"] == lease["lease_id"])["state"] == "unknown"
    assert state["pools"]["summary"]["unknown"] == 1
    with pytest.raises(FileExistsError):
        _broker(tmp_path)


def test_settlement_is_idempotent_but_cannot_change_evidence_or_document(tmp_path):
    with _broker(tmp_path) as broker:
        a, b = AdmissionClient(broker.config_for("a")), AdmissionClient(broker.config_for("b"))
        lease = a.acquire("main", "main-agent", time.monotonic() + 3)
        proof = {"response_id": "real-response", "done": True}
        done = a.settle(lease, "complete", proof)
        seq = broker.snapshot()["sequence"]
        assert a.settle(lease, "complete", proof) == done
        assert broker.snapshot()["sequence"] == seq
        with pytest.raises(AdmissionError, match="changed its evidence"):
            a.settle(lease, "complete", {"response_id": "other-response"})
        with pytest.raises(AdmissionError, match="Cannot settle"):
            a.settle(lease, "unknown", {})
        with pytest.raises(AdmissionError, match="foreign lease"):
            b.settle(lease, "complete", proof)
        tampered = broker.config_for("a")
        tampered["document_id"] = "b"
        with pytest.raises(AdmissionError, match="identity mismatch"):
            AdmissionClient(tampered).snapshot()


def test_expired_active_lease_becomes_unknown_not_free_slot(tmp_path):
    with _broker(tmp_path) as broker:
        client = AdmissionClient(broker.config_for("a"))
        lease = client.acquire("main", "deadline-active", time.monotonic() + 0.1)
        _wait(lambda: broker.snapshot()["pools"]["main"]["unknown"] == 1)
        assert broker.snapshot()["pools"]["main"]["blocked"]
        # Later actual request-bound completion is evidence; PID exit was not.
        assert client.settle(lease, "complete", {"response_id": "late-actual-done", "done": True})["state"] == "complete"


def test_single_waiting_document_can_use_entire_capacity_and_json_config_is_portable(tmp_path):
    with _broker(tmp_path) as broker:
        client = AdmissionClient(json.loads(json.dumps(broker.config_for("a"))))
        first = client.acquire("main", "child-1", time.monotonic() + 2)
        second = client.acquire("main", "main-agent", time.monotonic() + 2)
        assert client.snapshot()["pools"]["main"]["active"] == 2
        client.settle(first, "complete", {"done": True})
        client.settle(second, "complete", {"done": True})


def test_close_does_not_wait_for_stuck_observer_or_claim_its_unknown_is_stopped(tmp_path):
    entered, finish = threading.Event(), threading.Event()

    def observer(pool, deadline, snapshot):
        entered.set()
        finish.wait(5)
        return {"confirmed_idle": True, "service_identity": "late-result"}

    broker = _broker(tmp_path, quiescence_observer=observer)
    try:
        client = AdmissionClient(broker.config_for("a"))
        lease = client.acquire("main", "unknown", time.monotonic() + 10)
        client.settle(lease, "unknown", {"connection_lost": True})
        assert entered.wait(2)
        started = time.monotonic()
        broker.close()
        assert time.monotonic() - started < 1
        assert broker.snapshot()["pools"]["main"]["unknown"] == 1
        before = (tmp_path / "admission-events.jsonl").read_bytes()
        finish.set()
        time.sleep(0.03)
        assert (tmp_path / "admission-events.jsonl").read_bytes() == before
    finally:
        finish.set()
        broker.close()


def test_expired_document_can_be_observed_idle_without_reviving_its_generation_deadline(tmp_path):
    observations = []

    def observer(pool, deadline, snapshot):
        observations.append(snapshot)
        assert deadline > time.monotonic()
        assert all(row["deadline"] < time.monotonic() for row in snapshot["requests"])
        return {"confirmed_idle": True, "service_identity": "same-synthetic-service"}

    with _broker(tmp_path, quiescence_observer=observer) as broker:
        a, b = AdmissionClient(broker.config_for("a")), AdmissionClient(broker.config_for("b"))
        doc_deadline = time.monotonic() + 0.08
        original = a.acquire("main", "expires", doc_deadline)
        _wait(lambda: bool(observations) and broker.snapshot()["pools"]["main"]["unknown"] == 0)
        assert next(r for r in broker.snapshot()["requests"] if r["lease_id"] == original["lease_id"])["state"] == "quiescent"
        with pytest.raises(AdmissionDeadlineExceeded):
            a.acquire("main", "must-not-revive", doc_deadline)
        later = b.acquire("main", "still-valid-document", time.monotonic() + 2)
        b.settle(later, "complete", {"done": True})


def test_settlement_grace_allows_idle_observation_but_never_new_generation(tmp_path):
    cutoff = time.monotonic() + 0.18
    settlement_cutoff = cutoff + 0.6

    def observer(pool, deadline, snapshot):
        assert deadline == settlement_cutoff
        time.sleep(max(0, cutoff + 0.04 - time.monotonic()))
        return {"confirmed_idle": True, "service_identity": "same-synthetic-service"}

    with BatchAdmissionBroker(tmp_path, {"main": 1}, ["a", "b"], cutoff,
                              observer, settlement_deadline=settlement_cutoff) as broker:
        client = AdmissionClient(broker.config_for("a"))
        lease = client.acquire("main", "original", cutoff)
        client.settle(lease, "unknown", {"connection_lost": True})
        _wait(lambda: next(r for r in broker.snapshot()["requests"] if r["lease_id"] == lease["lease_id"])["state"] == "quiescent")
        assert cutoff < time.monotonic() < settlement_cutoff
        with pytest.raises(AdmissionDeadlineExceeded):
            AdmissionClient(broker.config_for("b")).acquire("main", "late", settlement_cutoff)
        assert len(broker.snapshot()["requests"]) == 1


def test_exhausted_observation_fails_pending_promptly_without_releasing_unknown(tmp_path):
    with _broker(tmp_path, quiescence_observer=lambda *_: {"confirmed_idle": False},
                 observer_max_attempts=2, observer_retry_interval=0.01) as broker:
        a, b = AdmissionClient(broker.config_for("a")), AdmissionClient(broker.config_for("b"))
        lease = a.acquire("main", "unknown", time.monotonic() + 10)
        a.settle(lease, "unknown", {"connection_lost": True})
        started = time.monotonic()
        with pytest.raises(AdmissionBlocked, match="quiescence_attempts_exhausted"):
            b.acquire("main", "never-admitted", time.monotonic() + 10)
        assert time.monotonic() - started < 1
        state = broker.snapshot()
        assert state["pools"]["main"]["unknown"] == state["pools"]["main"]["occupied"] == 1
        blocked = next(r for r in state["requests"] if r["agent_id"] == "never-admitted")
        assert blocked["state"] == "blocked" and "granted_at" not in blocked
        assert blocked["evidence"]["not_sent"] and blocked["evidence"]["unknown_requests_retained"]


def test_actual_late_completion_supersedes_slow_observer_without_releasing_new_work(tmp_path):
    entered, finish = threading.Event(), threading.Event()

    def observer(pool, deadline, snapshot):
        entered.set()
        finish.wait(3)
        return {"confirmed_idle": True, "service_identity": "now-obsolete-proof"}

    with _broker(tmp_path, quiescence_observer=observer) as broker:
        try:
            client = AdmissionClient(broker.config_for("a"))
            first = client.acquire("main", "old", time.monotonic() + 10)
            client.settle(first, "unknown", {"connection_lost": True})
            assert entered.wait(2)
            client.settle(first, "complete", {"response_id": "old-native-completion", "done": True})
            current = client.acquire("main", "new", time.monotonic() + 1)
            # A fresh unknown after a genuine new request must not be cleared by
            # the old request's delayed observer response.
            client.settle(current, "unknown", {"new_connection_lost": True})
            before = broker.snapshot()
            assert before["pools"]["main"]["unknown"] == 1
            # The new recovery probe is valid; keep it from starting for this
            # synthetic check by settling its own actual completion immediately.
            client.settle(current, "complete", {"response_id": "new-native-completion", "done": True})
            latest = client.acquire("main", "latest", time.monotonic() + 1)
            finish.set()
            time.sleep(0.03)
            row = next(r for r in broker.snapshot()["requests"] if r["lease_id"] == latest["lease_id"])
            assert row["state"] == "active"
            client.settle(latest, "complete", {"done": True})
        finally:
            finish.set()


@pytest.mark.parametrize("later_idle", [False, True])
def test_system_unknown_accepts_first_client_call_evidence_without_losing_origin(tmp_path, later_idle):
    observer = (lambda *_: {"confirmed_idle": True, "service_identity": "synthetic"}) if later_idle else None
    with _broker(tmp_path, quiescence_observer=observer) as broker:
        client = AdmissionClient(broker.config_for("a"))
        lease = client.acquire("main", "child-agent", time.monotonic() + 0.07)
        target = "quiescent" if later_idle else "unknown"
        _wait(lambda: next(r for r in broker.snapshot()["requests"] if r["lease_id"] == lease["lease_id"])["state"] == target)
        evidence = {"call_id": "real-child-call", "request_id": "real-upstream-request",
                    "generation_attempted": True, "usage_known": False}
        row = client.settle(lease, "unknown", evidence)
        assert row["state"] == target
        assert row["settlement_evidence"] == evidence and row["settlement_state"] == "unknown"
        assert client.settle(lease, "unknown", evidence) == row
        with pytest.raises(AdmissionError, match="changed its evidence"):
            client.settle(lease, "unknown", {**evidence, "call_id": "wrong-call"})
        events = [json.loads(line) for line in (tmp_path / "admission-events.jsonl").read_text().splitlines()]
        recorded = [e for e in events if e["event"] == "settlement_recorded"]
        assert len(recorded) == 1 and recorded[0]["request"]["settlement_evidence"] == evidence


def _reset_control_connections(broker, monkeypatch, path, *, after_commit, reset_limit):
    """Inject actual TCP reset, optionally after durable broker mutation."""
    original = broker._server.RequestHandlerClass.do_POST
    resets, lock = [], threading.Lock()

    def do_post(handler):
        with lock:
            reset = handler.path == path and len(resets) < reset_limit
            if reset:
                slot = len(resets)
                resets.append(None)
        if not reset:
            return original(handler)
        handler.connection.settimeout(2)
        body = json.loads(handler.rfile.read(int(handler.headers["Content-Length"])))
        resets[slot] = body
        if after_commit:
            broker._dispatch(path, body)
        handler.close_connection = True
        handler.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        handler.connection.close()

    monkeypatch.setattr(broker._server.RequestHandlerClass, "do_POST", do_post)
    return resets


@pytest.mark.parametrize("after_commit", [False, True])
def test_acquire_tcp_reset_replays_same_request_without_allocating_another_lease(tmp_path, monkeypatch, after_commit):
    with _broker(tmp_path) as broker:
        resets = _reset_control_connections(broker, monkeypatch, "/acquire", after_commit=after_commit, reset_limit=1)
        client = AdmissionClient(broker.config_for("a"))
        lease = client.acquire("main", "one-generation", time.monotonic() + 2)
        assert len(resets) == 1 and resets[0]["request_id"] == lease["request_id"]
        assert len(client.transport_events) == 1 and client.transport_events[0]["path"] == "/acquire"
        assert len(broker.snapshot()["requests"]) == 1 and broker.snapshot()["pools"]["main"]["active"] == 1
        client.settle(lease, "complete", {"call_id": "one-call", "done": True})
    events = [json.loads(line) for line in (tmp_path / "admission-events.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "enqueued" for e in events) == 1
    assert sum(e["event"] == "transition" and e["request"]["state"] == "active" for e in events) == 1


@pytest.mark.parametrize("state", ["complete", "unknown"])
@pytest.mark.parametrize("all_acks_lost", [False, True])
def test_settle_lost_ack_replays_exact_evidence_and_never_claims_success_without_ack(
        tmp_path, monkeypatch, state, all_acks_lost):
    with _broker(tmp_path) as broker:
        client = AdmissionClient(broker.config_for("a"))
        lease = client.acquire("main", "one-generation", time.monotonic() + 5)
        resets = _reset_control_connections(broker, monkeypatch, "/settle", after_commit=True,
                                            reset_limit=100 if all_acks_lost else 1)
        evidence = {"call_id": "one-call", "done": state == "complete", "usage_known": state == "complete"}
        if all_acks_lost:
            with pytest.raises(AdmissionUnavailable, match=r"path=/settle, attempts=4"):
                client.settle(lease, state, evidence)
        else:
            assert client.settle(lease, state, evidence)["state"] == state
        assert len(resets) == (4 if all_acks_lost else 1)
        assert all(body == resets[0] for body in resets)
        row = broker.snapshot()["requests"][0]
        assert row["state"] == state and row["settlement_evidence"] == evidence
        assert broker.snapshot()["pools"]["main"]["unknown"] == (1 if state == "unknown" else 0)
    events = [json.loads(line) for line in (tmp_path / "admission-events.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "settlement_recorded" for e in events) == 1
    assert sum(e["event"] == "transition" and e["request"]["state"] == state for e in events) == 1


def test_control_rpc_retries_share_one_timeout_and_do_not_retry_application_rejections(tmp_path, monkeypatch):
    with _broker(tmp_path) as broker:
        client = AdmissionClient(broker.config_for("a"))
        original_open, attempts = client._opener.open, []

        def slow_reset(request, timeout):
            attempts.append((time.monotonic(), request.data, timeout))
            time.sleep(min(0.02, timeout))
            raise ConnectionResetError("synthetic transient control-plane failure")

        monkeypatch.setattr(client._opener, "open", slow_reset)
        started = time.monotonic()
        with pytest.raises(AdmissionUnavailable, match="path=/snapshot"):
            client.snapshot(timeout=0.045)
        assert time.monotonic() - started < 0.09
        assert 1 < len(attempts) <= 4 and all(body == attempts[0][1] for _, body, _ in attempts)
        assert all(stamp < started + 0.045 for stamp, _, _ in attempts)
        monkeypatch.setattr(client._opener, "open", original_open)
        before = len(client.transport_events)
        with pytest.raises(AdmissionError, match="Unknown admission pool"):
            client.acquire("bad-pool", "invalid", time.monotonic() + 1)
        assert len(client.transport_events) == before
        assert broker.snapshot()["requests"] == []


@pytest.mark.parametrize("burst", range(3))
def test_synchronized_spawn_bursts_keep_caps_without_losing_control_connections(tmp_path, burst):
    context = mp.get_context("spawn")
    lock, start, ready, output = context.Lock(), context.Event(), context.Queue(), context.Queue()
    counters, peaks = context.Array("i", [0, 0]), context.Array("i", [0, 0])
    processes = []
    with BatchAdmissionBroker(tmp_path, {"main": 2, "summary": 1}, ["a", "b", "c"],
                              time.monotonic() + 30) as broker:
        try:
            for index in range(16):
                pool = "main" if index < 12 else "summary"
                process = context.Process(target=_parallel_worker,
                    args=(broker.config_for("abc"[index % 3]), pool, f"burst-{burst}-agent-{index}",
                          start, lock, counters, peaks, output, ready, 4))
                process.start()
                processes.append(process)
            # All 16 processes reach the barrier before releasing the burst;
            # process spawn timing cannot accidentally stagger the first HTTPs.
            assert len({ready.get(timeout=15) for _ in processes}) == 16
            start.set()
            results = [output.get(timeout=15) for _ in processes]
            assert all(result == "ok" for _, result in results), results
            for process in processes:
                process.join(3)
                assert process.exitcode == 0
            assert list(peaks) == [2, 1]
            rows = broker.snapshot()["requests"]
            assert len(rows) == len({row["lease_id"] for row in rows}) == 64
            assert all(row["state"] == "complete" for row in rows)
        finally:
            start.set()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(3)
    started_event = json.loads((tmp_path / "admission-events.jsonl").read_text().splitlines()[0])
    assert started_event["request_queue_size"] >= 128
