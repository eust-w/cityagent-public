"""Cross-branch receipt/recovery contracts with synthetic local transport."""
from __future__ import annotations

import json
import time

import pytest

from bridge_evidence.contest.local_model_client import LocalModelClient
from bridge_evidence.contest.reliable_generation import StreamPolicy


def stream_bytes():
    frames = [
        {"choices": [{"index": 0, "delta": {"content": "完成"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
    ]
    return b"".join(("data: " + json.dumps({"id": "response-1", "model": "own-model", **frame})
                     + "\n\n").encode() for frame in frames) + b"data: [DONE]\n\n"


class StreamResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self):
        self.raw = stream_bytes()

    def geturl(self):
        return "http://127.0.0.1:9/v1/chat/completions"

    def set_read_timeout(self, timeout):
        pass

    def read1(self, size):
        raw, self.raw = self.raw, b""
        return raw

    def close(self):
        pass


def client_and_payload(events):
    client = LocalModelClient("http://127.0.0.1:9", "own-model", temperature=1,
                              receipt_sink=events.append)
    client.request_context = {"reservation_id": "task-attempt-1"}
    client.enable_reliable_generation(StreamPolicy(), deadline=time.monotonic() + 30)
    payload = {"model": "own-model", "messages": [{"role": "user", "content": "证据"}],
               "max_tokens": 16, "temperature": 1, "chat_template_kwargs": {"enable_thinking": False}}
    return client, payload


@pytest.mark.parametrize("interrupted", [False, True])
def test_stream_reserves_before_transport_and_persists_terminal_identity(interrupted):
    events = []
    client, payload = client_and_payload(events)

    class Opener:
        def open(self, request, timeout):
            assert len(events) == 1
            assert events[0]["status"] == "running"
            assert events[0]["reservation_id"] == "task-attempt-1"
            if interrupted:
                raise KeyboardInterrupt("executor interrupted")
            return StreamResponse()

    client._opener = Opener()
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            client._request("/v1/chat/completions", payload, kind="generation")
    else:
        client._request("/v1/chat/completions", payload, kind="generation")
    assert len(client.calls) == 1
    assert {event["call_id"] for event in events} == {client.calls[0]["call_id"]}
    assert events[-1] == client.calls[0]
    assert events[-1]["status"] == ("interrupted" if interrupted else "success")
    assert events[-1]["reservation_id"] == "task-attempt-1"
    if interrupted:
        assert events[-1]["timing_complete"] is False
        assert events[-1]["usage_known"] is False


def test_stream_journal_failure_prevents_network_send():
    client, payload = client_and_payload([])

    def failed_sink(receipt):
        raise OSError("journal unavailable")

    class Opener:
        def open(self, *args, **kwargs):
            pytest.fail("A failed reservation must never send a model request")

    client.receipt_sink = failed_sink
    client._opener = Opener()
    with pytest.raises(OSError, match="journal unavailable"):
        client._request("/v1/chat/completions", payload, kind="generation")
    assert len(client.calls) == 1
    assert client.calls[0]["generation_attempted"] is False


@pytest.mark.parametrize("deadline", [0, -1])
def test_expired_stream_deadline_cannot_become_unbounded(deadline):
    client, payload = client_and_payload([])
    client.reliable_generation_deadline = deadline

    class Opener:
        def open(self, *args, **kwargs):
            pytest.fail("Expired streams must not open a connection")

    client._opener = Opener()
    with pytest.raises(RuntimeError):
        client._request("/v1/chat/completions", payload, kind="generation")
    assert client.calls[-1]["generation_attempted"] is False


def test_stream_rechecks_deadline_after_synchronous_reservation():
    now = [0.0]
    events = []
    client, payload = client_and_payload(events)
    client.enable_reliable_generation(StreamPolicy(clock=lambda: now[0]), deadline=10)

    def slow_journal(event):
        events.append(event)
        if event["status"] == "running":
            now[0] = 11

    class Opener:
        def open(self, *args, **kwargs):
            pytest.fail("Deadline expired while synchronously reserving the call")

    client.receipt_sink = slow_journal
    client._opener = Opener()
    with pytest.raises(RuntimeError):
        client._request("/v1/chat/completions", payload, kind="generation")
    assert events[0]["status"] == "running"
    assert events[-1]["generation_attempted"] is False


def test_failed_admitted_stream_journals_final_lease_settlement():
    from test_document_batch_client import route

    client, _, _, payload = route(raw=b"data: invalid\n\n")
    events = []
    client.receipt_sink = events.append
    with pytest.raises(RuntimeError):
        client._request("/v1/chat/completions", payload, kind="generation")
    terminal = [event for event in events if event.get("kind") == "generation"][-1]
    assert terminal["status"] == "failed"
    assert terminal["batch_admission"]["state"] == "unknown"


@pytest.mark.parametrize("local_count", [22, 23])
def test_gateway_template_count_is_fresh_and_mismatch_prevents_generation(local_count):
    from test_document_batch_client import route

    client, gateway, _, payload = route()
    seen = []
    payload["chat_template_kwargs"]["custom_flag"] = "preserved"

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.append(kwargs)
            return "complete template"

        def encode(self, prompt, **kwargs):
            return list(range(local_count))

    client.chat_tokenizer = Tokenizer()
    if local_count == 22:
        with pytest.raises(RuntimeError, match="token counts differ"):
            gateway.forward(payload)
        assert len(client.calls) == 1
        assert client.calls[0]["status"] == "failed"
        assert client.calls[0]["token_counts_match"] is False
    else:
        assert gateway.count_request(payload) == gateway.count_request(payload) == 23
        assert len(client.calls) == 2
        assert all(call["token_counts_match"] is True for call in client.calls)
    assert all(kwargs["custom_flag"] == "preserved" for kwargs in seen)
    assert all(call["kind"] == "tokenize" for call in client.calls)


@pytest.mark.parametrize("policy", ["review_output_tokens", "temperature"])
def test_resume_rejects_changed_review_or_generation_policy(tmp_path, policy):
    from test_divide_recovery import make

    workflow, _ = make(tmp_path)
    workflow.run()
    resumed, client = make(tmp_path, resume=True)
    if policy == "review_output_tokens":
        resumed.review_output_tokens = 4096
    else:
        client.temperature = 1
    with pytest.raises(ValueError, match="identity"):
        resumed.run()
    assert client.calls == []
