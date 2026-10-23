"""Document entry and real transport wiring with synthetic inputs only.

The SDK runner is an in-memory double; the actual gateway/client/SSE parser
remain in the integration path. No socket, subprocess, report or model is used.
"""
import argparse
import json
import socket
import subprocess
from types import SimpleNamespace

import pytest

from bridge_evidence.contest import document_cli as cli
from bridge_evidence.contest import document_model_gateway as gateway_module
from bridge_evidence.contest import document_workflow as module
from bridge_evidence.contest import local_model_client as client_module
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.reliable_generation import StreamPolicy


@pytest.fixture(autouse=True)
def no_external_actions(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("This synthetic wiring test must not use sockets, DNS or subprocesses")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


class Clock:
    now = 1000.0
    def monotonic(self):
        return self.now


def parsed(*flags):
    parser = argparse.ArgumentParser()
    cli.add_arguments(parser)
    return parser.parse_args(flags)


def test_document_cli_preserves_explicit_sampling_defaults():
    args = parsed()
    assert args.document_enable_thinking is False
    assert args.document_temperature == 0
    assert cli._settings(args).enable_thinking is False
    explicit = cli._settings(parsed("--document-enable-thinking", "--document-temperature", "1"))
    assert explicit.enable_thinking is True and explicit.temperature == 1


@pytest.mark.parametrize("value", [0, -1, 901, True, float("nan"), float("inf")])
def test_document_request_cap_is_positive_finite_and_bounded(value):
    with pytest.raises(ValueError):
        StreamPolicy(wall_seconds=value)


def test_worker_factory_explicitly_enables_sse_without_changing_generic_defaults(monkeypatch):
    from bridge_evidence.contest import document_multiagent_worker as worker
    clock = Clock()
    monkeypatch.setattr(worker, "time", clock)
    supplied = worker.make_client({"base_url": "http://127.0.0.1:9", "model": "synthetic",
                                   "enable_thinking": False}, deadline=clock.now + 1200)
    assert (supplied.timeout, supplied.enable_thinking, supplied.temperature) == (900, False, 1)
    assert supplied.reliable_generation_policy.wall_seconds == 900
    assert supplied.reliable_generation_deadline == 2200
    generic = client_module.LocalModelClient("http://127.0.0.1:9", "synthetic")
    assert generic.timeout == 600 and generic.reliable_generation_policy is None


def make_workflow(tmp_path, monkeypatch, *, remaining=1200, client_timeout=900,
                  supplied_timeout=900, request_cap=900, runner=None, max_sessions=1, client=None):
    clock = Clock()
    for imported in (module, client_module, gateway_module):
        monkeypatch.setattr(imported, "time", clock)
    client = client or client_module.LocalModelClient("http://127.0.0.1:9", "synthetic",
        timeout=client_timeout, enable_thinking=True, temperature=1)
    client.enable_reliable_generation(StreamPolicy(wall_seconds=request_cap, clock=clock.monotonic),
                                      deadline=clock.now + remaining)
    settings = module.DocumentSettings(enable_thinking=True, temperature=1,
                                       max_sessions=max_sessions, report_timeout=1200)
    wf = module.DocumentWorkflow(manifest_from_text("仅供测试的合成文本"), client,
        {"model": "synthetic", "timeout": supplied_timeout}, tmp_path,
        settings=settings, external_deadline=clock.now + remaining,
        react_runner=runner or (lambda *_args: {}))
    return wf, client, clock


@pytest.mark.parametrize("remaining,client_cap,supplied,request_cap,expected", [
    (1200, 900, 900, 900, 900), (23, 900, 900, 900, 23),
    (1200, 17, 900, 900, 17), (1200, 900, 17, 900, 900),
    (1200, 900, 900, 300, 300)])
def test_generation_uses_smallest_transport_policy_and_report_deadline(
        tmp_path, monkeypatch, remaining, client_cap, supplied, request_cap, expected):
    wf, client, _clock = make_workflow(tmp_path, monkeypatch, remaining=remaining,
        client_timeout=client_cap, supplied_timeout=supplied, request_cap=request_cap)
    before_deadline = wf.deadline
    client._opener = Opener(sse({"content": "合成状态"}) + sse(finish="stop") +
        sse(usage={"prompt_tokens": 23, "completion_tokens": 3, "total_tokens": 26}) + b"data: [DONE]\n\n")
    direct_gateway(wf).forward(request_payload("合成请求"))
    assert client.calls[-1]["wall_seconds_limit"] == expected
    assert wf.deadline == client.reliable_generation_deadline == before_deadline
    assert client.reliable_generation_policy.wall_seconds == request_cap
    assert client.reliable_generation_on_cancel is None
    assert client.calls[-1]["automatic_retry"] is False


def test_expired_report_cannot_use_sdk_grace_to_start_generation(tmp_path, monkeypatch):
    wf, client, clock = make_workflow(tmp_path, monkeypatch, remaining=1)
    clock.now += 1
    with pytest.raises(TimeoutError):
        direct_gateway(wf).forward(request_payload("过期请求"))
    assert client.calls == []


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def request_payload(prompt):
    return {"model": "synthetic", "max_tokens": 4096, "temperature": 1,
            "chat_template_kwargs": {"enable_thinking": True},
            "messages": [{"role": "user", "content": prompt}], "stream": False}


def sse(delta=None, finish=None, usage=None):
    value = {"id": "synthetic-response", "model": "synthetic", "choices": [] if usage else
        [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        value["usage"] = usage
    return b"data: " + packed(value) + b"\n\n"


class JSONResponse:
    status = 200
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        return False
    def read(self):
        # A synthetic count, deliberately not a real model/tokenizer result.
        return packed({"count": 23})


class StreamResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}
    def __init__(self, body):
        self.body, self.closed = body, False
    def set_read_timeout(self, seconds):
        self.last_timeout = seconds
    def read1(self, _size):
        body, self.body = self.body, b""
        return body
    def close(self):
        self.closed = True


class Opener:
    def __init__(self, body):
        self.response = StreamResponse(body)
        self.requests = []
    def open(self, request, *, timeout):
        self.requests.append((request.full_url, json.loads(request.data), timeout))
        return JSONResponse() if request.full_url.endswith("/tokenize") else self.response


def in_process_gateway(monkeypatch):
    class InProcessGateway(gateway_module.AuditedModelGateway):
        instances = []
        @property
        def url(self):
            return "http://127.0.0.1:9/v1"
        def __enter__(self):
            self.instances.append(self)
            return self
        def __exit__(self, *_args):
            return False
    monkeypatch.setattr(module, "AuditedModelGateway", InProcessGateway)
    return InProcessGateway


def test_workflow_selects_explicit_sse_route_and_preserves_semantic_gate(tmp_path, monkeypatch):
    gate = in_process_gateway(monkeypatch)
    raw = (sse({"content": "合成状态"}) + sse(finish="stop") +
           sse(usage={"prompt_tokens": 23, "completion_tokens": 3, "total_tokens": 26}) +
           b"data: [DONE]\n\n")
    observed = []
    def runner(prompt, _handlers, config):
        observed.append(config["timeout"])
        reply = gate.instances[-1].forward({"model": "synthetic", "max_tokens": 4096,
            "messages": [{"role": "system", "content": config["system_prompt"]},
                         {"role": "user", "content": prompt}], "stream": False,
            "temperature": config["temperature"],
            "chat_template_kwargs": {"enable_thinking": config["enable_thinking"]}})
        assert reply["choices"][0]["message"]["content"] == "合成状态"
        # A transport-complete synthetic answer is insufficient for delivery.
        return {"status": "completed", "native_final_event": True, "tool_calls": []}
    wf, client, _clock = make_workflow(tmp_path, monkeypatch, runner=runner)
    client._opener = Opener(raw)
    with pytest.raises(module.DocumentWorkflowFailure, match="before accepted delivery"):
        wf.run()
    assert observed == [900]
    generation = [row for row in client.calls if row["kind"] == "agent_generation"]
    assert len(generation) == 1 and generation[0]["done_received"] is True
    assert generation[0]["wall_seconds_limit"] == 900
    wire = client._opener.requests[-1][1]
    assert wire["stream"] is True and wire["chat_template_kwargs"]["enable_thinking"] is True
    assert wire["temperature"] == 1 and wire["repetition_penalty"] == 1.1
    assert wire["presence_penalty"] == 1.5
    assert wf.workspace.delivery_status()["ready"] is False


def test_unknown_stream_stops_outer_session_loop_and_remains_blocked_on_resume(tmp_path, monkeypatch):
    gate = in_process_gateway(monkeypatch)
    observed = []
    def runner(prompt, _handlers, config):
        observed.append(config["timeout"])
        with pytest.raises(RuntimeError):
            gate.instances[-1].forward(request_payload(prompt))
        return {"status": "error", "native_final_event": False, "tool_calls": []}
    wf, client, _clock = make_workflow(tmp_path, monkeypatch, runner=runner, max_sessions=3)
    # A complete HTTP response with an unfinished SSE body cannot prove that
    # the backend stopped, and missing usage is never treated as zero.
    client._opener = Opener(sse({"reasoning_content": "合成未完成状态"}))
    with pytest.raises(module.DocumentWorkflowFailure, match="outcome is unknown"):
        wf.run()
    assert observed == [900]
    assert len([row for row in client.calls if row["kind"] == "agent_generation"]) == 1
    assert client.calls[-1]["usage"] is None
    assert client._opener.response.closed is True
    saved = json.loads(wf.trace_path.read_text())
    assert saved["model_dispatches"][0]["status"] == "outcome_unknown"
    restored, new_client, _clock = make_workflow(tmp_path, monkeypatch, runner=runner, max_sessions=3)
    with pytest.raises(module.DocumentWorkflowFailure, match="outcome is unknown"):
        restored.run()
    assert new_client.calls == [] and observed == [900]


def test_unknown_receipt_is_durable_when_sdk_runner_raises_before_return(tmp_path, monkeypatch):
    gate = in_process_gateway(monkeypatch)
    def runner(prompt, _handlers, _config):
        try:
            gate.instances[-1].forward(request_payload(prompt))
        except RuntimeError:
            # The atomic gate exists already, without an outer SDK outcome.
            durable = json.loads((tmp_path / "document_workflow.json").read_text())
            assert durable["model_dispatches"][0]["status"] == "outcome_unknown"
            raise RuntimeError("synthetic SDK raised before returning")
        pytest.fail("The unfinished stream must fail")
    wf, client, _clock = make_workflow(tmp_path, monkeypatch, runner=runner, max_sessions=3)
    client._opener = Opener(sse({"reasoning_content": "合成未完成状态"}))
    with pytest.raises(module.DocumentWorkflowFailure, match="synthetic SDK raised"):
        wf.run()
    assert len(wf.sessions) == 1 and wf.sessions[0]["native_final_event"] is False
    durable = json.loads(wf.trace_path.read_text())
    actual = client.calls[-1]
    assert durable["model_dispatches"][0]["call_id"] == actual["call_id"]
    assert durable["model_dispatches"][0]["request_id"] == actual["request_id"]
    restored, new_client, _clock = make_workflow(tmp_path, monkeypatch, max_sessions=3)
    with pytest.raises(module.DocumentWorkflowFailure, match="outcome is unknown"):
        restored.run()
    assert new_client.calls == []


def direct_gateway(wf):
    gateway = gateway_module.AuditedModelGateway(wf.client, deadline=wf.deadline,
        max_calls=8, max_output_tokens=4096, before_request=wf._check_budget,
        on_dispatch=wf._on_dispatch)
    gateway.receipts = wf.gateway_receipts
    gateway.on_receipt = wf._on_model_receipt
    return gateway


@pytest.mark.parametrize("simulate_old_latch_window", [False, True])
def test_receipt_checkpoint_blocks_fresh_client_before_sdk_or_public_statecheck(
        tmp_path, monkeypatch, simulate_old_latch_window):
    wf, client, _clock = make_workflow(tmp_path, monkeypatch)
    client._opener = Opener(sse({"reasoning_content": "合成未完成状态"}))
    gateway = direct_gateway(wf)
    with pytest.raises(RuntimeError):
        gateway.forward(request_payload("合成请求"))
    # No SDK runner, outer outcome, or post-SDK public sticky check ran. This
    # is the checkpoint a process may leave immediately after receipt storage.
    durable = json.loads(wf.trace_path.read_text())
    assert durable["model_dispatches"][0]["call_id"] == client.calls[-1]["call_id"]
    assert durable["model_dispatches"][0]["status"] == "outcome_unknown"
    if simulate_old_latch_window:
        # Older checkpoints called HTTP 200 settled and did not copy SSE DONE.
        # Recovery still derives the unknown state from actual call evidence.
        durable["model_dispatches"][0]["status"] = "settled"
        durable["gateway_receipts"][0].pop("done_received", None)
        wf.trace_path.write_text(json.dumps(durable))
    restored, new_client, _clock = make_workflow(tmp_path, monkeypatch)
    assert restored.model_dispatches[0]["status"] == "outcome_unknown"
    with pytest.raises(module.DocumentWorkflowFailure, match="outcome is unknown"):
        restored.run()
    assert new_client.calls == []


def test_complete_length_response_remains_failed_without_false_unknown_cancel(tmp_path, monkeypatch):
    wf, client, _clock = make_workflow(tmp_path, monkeypatch)
    raw = (sse({"content": "合成长输出"}) + sse(finish="length") +
           sse(usage={"prompt_tokens": 23, "completion_tokens": 3, "total_tokens": 26}) +
           b"data: [DONE]\n\n")
    client._opener = Opener(raw)
    with pytest.raises(ValueError, match="incomplete or truncated"):
        direct_gateway(wf).forward(request_payload("合成请求"))
    actual = client.calls[-1]
    assert actual["status"] == "failed" and actual["done_received"] is True
    assert actual["usage_known"] is True and actual["cancel_status"] == "not_needed_stream_completed"
    assert all(row["status"] == "settled" for row in wf.model_dispatches)
    assert wf.workspace.delivery_status()["ready"] is False
    restored, new_client, _clock = make_workflow(tmp_path, monkeypatch)
    assert all(row["status"] == "settled" for row in restored.model_dispatches)
    restored._check_budget()
    assert new_client.calls == []
