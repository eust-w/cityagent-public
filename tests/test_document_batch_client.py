"""Real client/gateway admission wiring with synthetic streams; no model calls."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import socket
import threading
import time
from urllib.parse import urlsplit

import pytest

from bridge_evidence.contest.document_batch_admission import BatchAdmissionBroker
from bridge_evidence.contest.document_batch_client import BatchClientAdmission
from bridge_evidence.contest.document_model_gateway import AuditedModelGateway
from bridge_evidence.contest.document_multiagent_worker import make_client
from bridge_evidence.contest.local_model_client import LocalModelClient
from bridge_evidence.contest.reliable_generation import ReliableGenerationError, StreamPolicy
from test_reliable_document_gateway import JSONResponse, SSEResponse, complete, event


class FakeAdmissionClient:
    def __init__(self, on_acquire=None, lease_deadline=None):
        self.on_acquire = on_acquire
        self.lease_deadline = lease_deadline
        self.acquired = []
        self.settled = []
        self.entered = threading.Event()
        self.finished = threading.Event()

    def acquire(self, pool, agent_id, deadline):
        self.entered.set()
        if self.on_acquire:
            self.on_acquire()
        lease = {"request_id": f"request-{len(self.acquired)}", "lease_id": f"lease-{len(self.acquired)}"}
        if self.lease_deadline is not None:
            lease["deadline"] = self.lease_deadline
        self.acquired.append({"lease": lease, "pool": pool, "agent_id": agent_id, "deadline": deadline})
        return lease

    def settle(self, lease, disposition, evidence):
        self.settled.append({"lease": deepcopy(lease), "state": disposition, "evidence": deepcopy(evidence)})
        self.finished.set()
        return {"state": disposition}


class SequenceOpener:
    def __init__(self, streams):
        self.streams = list(streams)
        self.requests = []

    def open(self, request, *, timeout):
        self.requests.append((request.full_url, json.loads(request.data), timeout))
        if request.full_url.endswith("/tokenize"):
            return JSONResponse()
        return self.streams.pop(0)


def route(*, admission=None, raw=None, timeout=1.0, lifetime=10.0, on_cancel=None, recovery=None):
    deadline = time.monotonic() + lifetime
    client = LocalModelClient("http://127.0.0.1:9", "own-model", timeout=timeout,
        max_model_len=512, max_output_tokens=16, enable_thinking=False, temperature=1)
    client.enable_reliable_generation(StreamPolicy(), deadline=deadline, on_cancel=on_cancel)
    stream = SSEResponse(complete() if raw is None else raw, None)
    client._opener = SequenceOpener([stream])
    fake = admission or FakeAdmissionClient()
    client.batch_admission = BatchClientAdmission({}, pool="main", agent_id="child-1",
        deadline=deadline, admission_client=fake)
    gateway = AuditedModelGateway(client, deadline=deadline, max_calls=5, max_output_tokens=16,
        recovery_prompt=recovery)
    payload = {"model": "own-model", "messages": [{"role": "user", "content": "合成报告原文"}],
        "stream": False, "max_tokens": 16, "temperature": 1,
        "chat_template_kwargs": {"enable_thinking": False}}
    return client, gateway, fake, payload


def generations(client):
    return [row for row in client.calls if row.get("endpoint") == "/v1/chat/completions"]


def test_queue_is_not_a_generation_and_does_not_spend_per_request_timeout():
    fake = FakeAdmissionClient()
    client, gateway, _, payload = route(admission=fake, timeout=.05)

    def queued():
        assert client.calls == [] and gateway._issued == 0
        time.sleep(.12)  # More than twice the scaled per-request 900-second cap.
        assert client.calls == [] and gateway._issued == 0

    fake.on_acquire = queued
    result = gateway.forward(payload)
    assert result["choices"][0]["finish_reason"] == "stop"
    assert len(generations(client)) == gateway._issued == 1
    receipt = generations(client)[0]
    assert .01 < receipt["wall_seconds_limit"] <= .05
    assert gateway.receipts[-1]["batch_queue_seconds"] >= .1
    assert fake.settled[0]["state"] == "complete"


def test_gateway_outer_and_client_inner_scopes_share_exactly_one_lease():
    client, gateway, fake, payload = route()
    gateway.forward(payload)
    assert len(fake.acquired) == len(fake.settled) == 1
    assert client.batch_admission.active is None
    assert generations(client)[0]["batch_admission"]["state"] == "complete"
    assert fake.settled[0]["evidence"]["request_id"] == gateway.receipts[-1]["request_id"]


def test_tokenization_uses_no_generation_lease_and_direct_client_acquires_one():
    client, _, fake, payload = route()
    assert client.count_tokens("synthetic input") == 23
    assert fake.acquired == [] and generations(client) == []
    result = client.generate(payload["messages"])
    assert result["status"] == "success"
    assert len(fake.acquired) == len(fake.settled) == len(generations(client)) == 1
    assert fake.settled[0]["state"] == "complete"


def test_deadline_expiring_during_queue_cannot_send_after_a_late_grant():
    fake = FakeAdmissionClient(on_acquire=lambda: time.sleep(.04))
    client, gateway, _, payload = route(admission=fake, lifetime=.02)
    with pytest.raises(TimeoutError, match="expired"):
        gateway.forward(payload)
    assert client.calls == [] and client._opener.requests == []
    assert gateway._issued == 0
    assert fake.settled[0]["state"] == "not_sent"


@pytest.mark.parametrize("through_gateway", [False, True], ids=["direct", "gateway"])
def test_broker_shorter_lease_deadline_limits_real_stream_wall_budget(through_gateway):
    fake = FakeAdmissionClient()
    client, gateway, _, payload = route(admission=fake, timeout=2, lifetime=10)
    fake.lease_deadline = time.monotonic() + .25
    if through_gateway:
        gateway.forward(payload)
    else:
        client.generate(payload["messages"])
    receipt = generations(client)[0]
    assert 0 < receipt["wall_seconds_limit"] <= .25
    generation_open = [row for row in client._opener.requests if row[0].endswith("/v1/chat/completions")]
    assert len(generation_open) == 1 and 0 < generation_open[0][2] <= .25
    assert fake.settled[0]["state"] == "complete"


@pytest.mark.parametrize("through_gateway", [False, True], ids=["direct", "gateway"])
def test_broker_expired_lease_cannot_dispatch_with_longer_document_deadline(through_gateway):
    fake = FakeAdmissionClient(lease_deadline=time.monotonic() - 1)
    client, gateway, _, payload = route(admission=fake, lifetime=10)
    with pytest.raises(TimeoutError, match="expired"):
        if through_gateway:
            gateway.forward(payload)
        else:
            client.generate(payload["messages"])
    assert generations(client) == []
    assert not any(row[0].endswith("/v1/chat/completions") for row in client._opener.requests)
    assert gateway._issued == 0
    assert fake.settled[0]["state"] == "not_sent"


@pytest.mark.parametrize("failure", ["usage", "business_callback", "length"])
def test_received_done_releases_even_when_later_business_validation_fails(failure):
    raw = complete(finish="length" if failure == "length" else "stop")
    if failure == "usage":
        raw = raw.replace(b'"prompt_tokens":23', b'"prompt_tokens":24').replace(b'"total_tokens":26', b'"total_tokens":27')
    client, gateway, fake, payload = route(raw=raw)
    if failure == "business_callback":
        def reject(*_):
            raise OSError("synthetic persistence failure after complete response")
        gateway.on_response = reject
    with pytest.raises((ValueError, RuntimeError)):
        gateway.forward(payload)
    assert generations(client)[0]["done_received"] is True
    assert fake.settled[0]["state"] == "complete"
    assert fake.settled[0]["evidence"]["done_received"] is True
    assert client.batch_admission.active is None


@pytest.mark.parametrize("failure", ["invalid_model", "before_request", "prepared_callback"])
def test_pre_dispatch_validation_or_persistence_failure_settles_not_sent(failure):
    client, gateway, fake, payload = route()
    if failure == "invalid_model":
        payload["model"] = "foreign-model"
    else:
        def fail(*_):
            raise ValueError("synthetic pre-dispatch rejection")
        if failure == "before_request":
            gateway.before_request = fail
        else:
            gateway.on_prepared_request = fail
    with pytest.raises((ValueError, RuntimeError)):
        gateway.forward(payload)
    assert generations(client) == []
    assert not any(url.endswith("/v1/chat/completions") for url, _, _ in client._opener.requests)
    assert fake.settled[0]["state"] == "not_sent"
    assert gateway._issued == 0
    assert gateway.receipts[-1]["generation_attempted"] is False
    assert gateway.receipts[-1]["batch_dispatch_state"] == "not_sent"


def test_stream_preflight_failure_before_open_is_not_sent():
    client, gateway, fake, payload = route()
    def failed_guard_setup():
        raise ValueError("synthetic tokenizer guard initialization failure")
    client.reliable_generation_raw_guard_factory = failed_guard_setup
    with pytest.raises(ValueError, match="guard initialization"):
        gateway.forward(payload)
    assert not any(url.endswith("/v1/chat/completions") for url, _, _ in client._opener.requests)
    assert generations(client) == []
    assert fake.settled[0]["state"] == "not_sent"
    assert gateway._issued == 0
    assert gateway.receipts[-1]["generation_attempted"] is False
    assert gateway.receipts[-1]["batch_dispatch_state"] == "not_sent"


def test_midstream_disconnect_remains_unknown_instead_of_freeing_capacity():
    client, gateway, fake, payload = route(raw=[event({"content": "部分输出"}), OSError("synthetic broken stream")])
    with pytest.raises(ReliableGenerationError):
        gateway.forward(payload)
    assert len(generations(client)) == 1
    receipt = generations(client)[0]
    assert receipt["generation_attempted"] is True and receipt["done_received"] is False
    assert fake.settled[0]["state"] == "unknown"
    assert fake.settled[0]["evidence"]["response_id"] == receipt["response_id"]
    assert gateway._issued == 1
    assert gateway.receipts[-1]["generation_attempted"] is True
    assert gateway.receipts[-1].get("batch_dispatch_state") != "not_sent"


def test_gateway_closing_while_queued_never_sends_a_late_upstream_request():
    release = threading.Event()
    fake = FakeAdmissionClient(on_acquire=lambda: release.wait(2))
    client, gateway, _, payload = route(admission=fake)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(gateway.forward, payload)
        assert fake.entered.wait(1)
        gateway.__exit__(None, None, None)
        release.set()
        with pytest.raises(RuntimeError, match="no longer active"):
            future.result(timeout=2)
    assert client._opener.requests == [] and client.calls == []
    assert gateway._issued == 0
    assert fake.settled[0]["state"] == "not_sent"


def test_real_http_caller_disconnect_while_queued_never_dispatches_upstream():
    release = threading.Event()
    fake = FakeAdmissionClient(on_acquire=lambda: release.wait(2))
    client, gateway, _, payload = route(admission=fake)
    with gateway:
        target = urlsplit(gateway.url)
        connection = socket.create_connection((target.hostname, target.port), timeout=2)
        raw = json.dumps(payload).encode()
        try:
            connection.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: {target.hostname}\r\n"
                f"Content-Length: {len(raw)}\r\nContent-Type: application/json\r\n\r\n").encode() + raw)
            assert fake.entered.wait(1)
            connection.shutdown(socket.SHUT_RDWR)
        finally:
            connection.close()
            release.set()
        assert fake.finished.wait(2)
    assert client._opener.requests == [] and generations(client) == []
    assert gateway._issued == 0
    assert fake.settled[0]["state"] == "not_sent"


def test_caller_disappearing_during_preflight_is_rechecked_at_actual_transport():
    alive = {"value": True}
    client, gateway, fake, payload = route()
    gateway.on_prepared_request = lambda *_: alive.update(value=False)
    with pytest.raises(RuntimeError, match="no longer active"):
        gateway.forward(payload, request_alive=lambda: alive["value"])
    assert generations(client) == []
    assert not any(url.endswith("/v1/chat/completions") for url, _, _ in client._opener.requests)
    assert fake.settled[0]["state"] == "not_sent"
    assert gateway._issued == 0
    assert gateway.receipts[-1]["generation_attempted"] is False
    assert gateway.receipts[-1]["batch_dispatch_state"] == "not_sent"


def test_proved_stopped_repetition_recovery_acquires_a_fresh_lease():
    def idle_after_disconnect(receipt, _response):
        return {"status": "unknown", "stop_observation": {
            "response_id": receipt["response_id"], "stopped_observed": True,
            "method": "service_idle_after_disconnect", "same_service_process": True,
            "cancellation_confirmed": False, "model_identity_sha256": "a" * 64,
            "process_identity_sha256": "b" * 64,
            "samples": [{"idle": True, "after_disconnect_seconds": 1},
                        {"idle": True, "after_disconnect_seconds": 2}]}}
    client, gateway, fake, payload = route(raw=event({"content": "H" * 600}),
        on_cancel=idle_after_disconnect, recovery=lambda _: "保留已保存条目，只完成剩余动作。")
    client._opener.streams.append(SSEResponse(complete(), None))
    result = gateway.forward(payload)
    assert result["choices"][0]["finish_reason"] == "stop"
    assert len(fake.acquired) == len(fake.settled) == len(generations(client)) == 2
    assert fake.acquired[0]["lease"] != fake.acquired[1]["lease"]
    assert [row["state"] for row in fake.settled] == ["complete", "complete"]
    assert generations(client)[0]["error_code"] == "content_repetition"
    assert gateway.receipts[1]["recovery_of_call_id"] == generations(client)[0]["call_id"]


def test_real_localhost_broker_and_configured_client_settle_the_same_document_lease(tmp_path):
    deadline = time.monotonic() + 10
    with BatchAdmissionBroker(tmp_path / "broker", {"main": 2, "summary": 1}, ["document-a"], deadline) as broker:
        client, gateway, _, payload = route()
        client.configure_batch_admission(broker.config_for("document-a"), pool="main", agent_id="main-agent", deadline=deadline)
        gateway.forward(payload)
        snapshot = broker.snapshot()
        assert snapshot["pools"]["main"]["occupied"] == 0
        assert len(snapshot["requests"]) == 1
        row = snapshot["requests"][0]
        assert (row["document_id"], row["agent_id"], row["state"]) == ("document-a", "main-agent", "complete")
        assert row["evidence"]["call_id"] == generations(client)[0]["call_id"]
        assert row["evidence"]["request_id"] == gateway.receipts[-1]["request_id"]


def test_real_broker_retains_unknown_for_broken_stream(tmp_path):
    deadline = time.monotonic() + 10
    with BatchAdmissionBroker(tmp_path / "broker", {"main": 1}, ["document-a"], deadline) as broker:
        client, gateway, _, payload = route(raw=[event({"content": "部分"}), OSError("disconnect")])
        client.configure_batch_admission(broker.config_for("document-a"), pool="main", agent_id="child", deadline=deadline)
        with pytest.raises(ReliableGenerationError):
            gateway.forward(payload)
        pool = broker.snapshot()["pools"]["main"]
        assert pool["active"] == 0 and pool["unknown"] == pool["occupied"] == 1
        assert pool["blocked"] is True


def test_make_client_assigns_main_and_summary_to_their_separate_pools_without_requests(tmp_path):
    deadline = time.monotonic() + 10
    with BatchAdmissionBroker(tmp_path / "broker", {"main": 2, "summary": 1}, ["document-a"], deadline) as broker:
        config = {"base_url": "http://127.0.0.1:9", "model": "own-model", "enable_thinking": False,
            "batch_admission": broker.config_for("document-a"),
            "summary": {"base_url": "http://127.0.0.1:10", "model": "own-summary"}}
        main = make_client(config, deadline)
        summary = make_client(config, deadline, summary=True)
        assert main.batch_admission.pool == "main"
        assert summary.batch_admission.pool == "summary"
        assert main.batch_admission.client.config == summary.batch_admission.client.config == broker.config_for("document-a")
        assert main.enable_thinking is False and summary.enable_thinking is True
        assert main.max_model_len == summary.max_model_len == 262144
        assert main.max_output_tokens == summary.max_output_tokens == 262144
        assert main.dynamic_output_budget and summary.dynamic_output_budget
        assert main.reliable_generation_policy.wall_seconds == summary.reliable_generation_policy.wall_seconds == 900
        assert main.calls == summary.calls == [] and broker.snapshot()["requests"] == []
