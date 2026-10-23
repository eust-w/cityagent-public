"""Raw/native repetition recovery through the real transport and gateway, no GPU."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from bridge_evidence.contest.document_workflow import DocumentWorkflow
from bridge_evidence.contest.generation_stop_observer import generation_stopped
from bridge_evidence.contest.reliable_generation import ReliableGenerationError
from test_repetition_recovery import Clock, SSEResponse, StreamSequence, TokenResponse, packed, route, stopped


def raw_event(text="", *, delta=None, finish=None, native=None, usage=None):
    choice = {"index": 0, "delta": delta or {}, "finish_reason": finish,
              "token_ids": list(text.encode("ascii"))}
    if native is not None:
        choice["stop_reason"] = native
    row = {"id": "raw-response", "model": "own-model", "choices": [choice] if usage is None else [],
           **({"usage": usage} if usage is not None else {})}
    return b"data: " + packed(row) + b"\n\n"


class RawSequence(StreamSequence):
    def open(self, request, *, timeout):
        payload = json.loads(request.data)
        self.requests.append((request.full_url, deepcopy(payload), timeout))
        if request.full_url.endswith("/tokenize"):
            return TokenResponse(self.count(payload))
        mode = next(self.modes)
        # A complete-looking parsed tool invocation is deliberately available
        # in the failing event: repetition must invalidate the whole response.
        save = {"tool_calls": [{"index": 0, "id": "must-never-execute", "type": "function",
            "function": {"name": "save_entries", "arguments": '{"entries":[]}'}}]}
        if mode == "raw":
            prefix = '{"entries":[{"id":"'
            raw = [raw_event(prefix), *[raw_event("H" * 64) for _ in range(15)],
                   raw_event("H" * 64, delta=save)]
        elif mode == "contract":
            prefix = '<tool_call><function=complete_review><parameter=basis>'
            raw = [raw_event(prefix), raw_event("abcdefghijklmnopqrstuvwxyz" * 8, delta=save)]
        elif mode.startswith("native"):
            native = "repetition_detected"
            finish = None if mode == "native_unterminated" else "tool_calls"
            raw = [raw_event('{"entries":[]}', delta=save, finish=finish, native=native)]
        else:
            count = self.count(payload)
            raw = [raw_event("done", delta={"content": "done"}), raw_event(finish="stop"),
                   raw_event(usage={"prompt_tokens": count, "completion_tokens": 4, "total_tokens": count + 4}),
                   b"data: [DONE]\n\n"]
        stream = SSEResponse(raw, Clock())
        self.streams.append(stream)
        return stream


def raw_route(tmp_path, modes, *, observer=stopped):
    client, gateway, payload, feedback = route(tmp_path, [], observer=observer)
    tokenizer = {"decoder": {"type": "ByteLevel"}, "model": {"type": "BPE",
                 "vocab": {chr(i): i for i in range(33, 127)}}, "added_tokens": []}
    path = tmp_path / "tokenizer.json"
    path.write_bytes(packed(tokenizer))
    client.enable_reliable_generation(client.reliable_generation_policy,
        deadline=client.reliable_generation_deadline, on_cancel=observer,
        capture_directory=tmp_path / "raw-sse", raw_guard_tokenizer_path=path,
        raw_guard_tokenizer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        native_repetition_detection=True)
    client._opener = RawSequence(modes)
    if "contract" in modes:
        client.reliable_tool_argument_contract = True
        payload["tools"] = [{"type": "function", "function": {
            "name": "complete_review", "parameters": {"type": "object",
                "properties": {"basis": {"type": "string", "maxLength": 32}},
                "required": ["basis"], "additionalProperties": False}}}]
    # Reprepare because the complete input now includes native protection.
    payload = gateway.prepare_request(payload)
    return client, gateway, payload, feedback


@pytest.mark.parametrize("mode,error", [("raw", "raw_repetition"), ("native", "native_repetition"),
                                      ("contract", "tool_argument_contract")])
def test_once_recovered_clean_output_preserves_original_history_and_never_publishes_failed_tools(tmp_path, mode, error):
    client, gateway, payload, feedback = raw_route(tmp_path, [mode, "success"])
    original = deepcopy(payload)
    published = []
    gateway.on_response = lambda response, audit: published.append(deepcopy(response))
    response = gateway.forward(payload)
    assert response["choices"][0]["message"]["content"] == "done"
    assert payload == original and len(feedback) == 1 and len(published) == 1
    assert all(not p["choices"][0]["message"].get("tool_calls") for p in published)
    calls = [r for r in client.calls if r["kind"] == "agent_generation"]
    assert len(calls) == 2 and calls[0]["error_code"] == error
    assert calls[0]["recovery_authorized"] is True and generation_stopped(calls[0])
    assert calls[0]["raw_sse_capture"]["path"]
    wire = [p for url, p, _ in client._opener.requests if url.endswith("/chat/completions")]
    assert wire[1]["messages"][:-1] == original["messages"]
    assert "H" * 32 not in json.dumps(wire[1]) and "must-never-execute" not in json.dumps(wire[1])
    assert "条目d1" in wire[1]["messages"][-1]["content"]
    assert all(p["repetition_penalty"] == 1.1 for p in wire)
    assert all(p["repetition_detection"] == {"min_pattern_size": 1, "max_pattern_size": 64, "min_count": 16} for p in wire)
    # Primary workflow must recognize the same explicitly authorized failure.
    workflow = object.__new__(DocumentWorkflow)
    workflow.settings = SimpleNamespace(single_document_agent=True)
    workflow.model_dispatches, workflow._persist = [], lambda: None
    workflow._on_model_receipt(gateway.receipts[0])
    assert getattr(workflow, "_single_agent_terminal_failure", None) is None


@pytest.mark.parametrize("mode,error_code", [("raw", "raw_repetition"), ("contract", "tool_argument_contract")])
def test_raw_without_stopped_proof_cannot_recover_or_publish_complete_looking_tool(tmp_path, mode, error_code):
    client, gateway, payload, feedback = raw_route(tmp_path, [mode], observer=None)
    published = []
    gateway.on_response = lambda *args: published.append(args)
    with pytest.raises(ReliableGenerationError) as error:
        gateway.forward(payload)
    assert error.value.code == error_code and not generation_stopped(error.value.receipt)
    assert not published and not feedback
    assert len([r for r in client.calls if r["kind"] == "agent_generation"]) == 1


def test_native_terminal_per_request_proof_recovers_without_waiting_for_other_request_idle(tmp_path):
    def peer_busy(receipt, response):
        assert response.closed
        return {"status": "not_confirmed", "stop_observation": {
            "method": "service_idle_after_disconnect", "response_id": receipt["response_id"],
            "stopped_observed": False, "samples": [{"idle": False}],
            "reason": "Another Agent is still generating"}}
    client, gateway, payload, feedback = raw_route(tmp_path, ["native", "success"], observer=peer_busy)
    gateway.forward(payload)
    failed = feedback[0]
    assert failed["cancel_status"] == "not_confirmed"
    assert failed["stop_observation"]["stopped_observed"] is False
    assert failed["native_stop_observation"]["response_id"] == failed["response_id"]
    assert generation_stopped(failed)


def test_native_flag_without_terminal_event_stops_output_but_does_not_authorize_retry(tmp_path):
    client, gateway, payload, feedback = raw_route(tmp_path, ["native_unterminated"], observer=None)
    with pytest.raises(ReliableGenerationError) as error:
        gateway.forward(payload)
    assert error.value.code == "native_repetition"
    assert not generation_stopped(error.value.receipt) and not feedback


@pytest.mark.parametrize("mode", ["raw", "native", "contract"])
def test_second_failure_does_not_retry_or_publish_partial_result(tmp_path, mode):
    client, gateway, payload, feedback = raw_route(tmp_path, [mode, mode])
    published = []
    gateway.on_response = lambda *args: published.append(args)
    with pytest.raises(ReliableGenerationError):
        gateway.forward(payload)
    assert len(feedback) == 1 and not published
    assert len([r for r in client.calls if r["kind"] == "agent_generation"]) == 2
    assert not gateway.receipts[-1].get("recovery_authorized")


def test_raw_only_failure_confirms_source_received_without_claiming_safe_output(tmp_path):
    from bridge_evidence.contest.document_source_delivery import SourceDeliveryLedger
    from bridge_evidence.contest.evidence_manifest import manifest_from_text
    client, gateway, payload, feedback = raw_route(tmp_path, ["raw"], observer=None)
    manifest = manifest_from_text(payload["messages"][0]["content"])
    unit = manifest["units"][0]
    ledger = SourceDeliveryLedger(manifest, tmp_path / "ledger", run_id="raw-only", agent_id="child")
    ledger.register_initial_message(payload["messages"][0]["content"],
        [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}])
    gateway.on_prepared_request = ledger.observe_request
    gateway.on_input_observed = lambda sent, audit: ledger.confirm_request(audit["request_id"], evidence=audit)
    published = []
    gateway.on_response = lambda *args: published.append(args)
    with pytest.raises(ReliableGenerationError):
        gateway.forward(payload)
    receipt = next(r for r in client.calls if r["kind"] == "agent_generation")
    assert receipt["content_characters"] == receipt["reasoning_characters"] == receipt["tool_call_count"] == 0
    assert receipt["raw_generation_observation"]["generated_tokens_observed"] >= 1024
    assert receipt["input_delivery_confirmed"] is True
    assert gateway.receipts[0]["input_delivery_evidence"] == "raw_generated_stream_data"
    assert ledger.confirmed_ranges() == {unit["id"]: [[0, len(unit["text"])]]}
    assert not ledger.pending_initial_messages() and not published and not feedback


@pytest.mark.parametrize("raw", [None, {}, {"generated_tokens_observed": 0},
    {"generated_tokens_observed": True}, {"prompt_token_ids": [1, 2]},
    {"generated_tokens_observed": 3, "tokenizer_sha256": "unverified"}])
def test_raw_configuration_or_prompt_ids_do_not_confirm_source(tmp_path, raw):
    client, gateway, payload, _ = raw_route(tmp_path, ["success"])
    observed = []
    gateway.on_input_observed = lambda *args: observed.append(args)
    gateway._observe_input(payload, {}, {"http_status": 200, "raw_generation_observation": raw,
        "return_token_ids": True, "prompt_token_ids": [1, 2, 3]})
    assert not observed
