"""Proved failed JSON pairs have a renderable, audited wire view; no models."""
from copy import deepcopy
import io
import json
import socket
import subprocess
from urllib.error import HTTPError
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

from bridge_evidence.contest import tong_react_adapter as adapter
from bridge_evidence.contest.document_context_budget import allocate_sdk_wire_requests, canonical
from bridge_evidence.contest.document_history_continuity import HistoryPrefixGuard
from bridge_evidence.contest.local_model_client import LocalModelClient


@pytest.fixture(autouse=True)
def no_external_actions(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("External actions are forbidden")
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)


def packed(value):
    return canonical(value).decode()


def fixture(mixed=True):
    raw = '{"entries": '
    origin = "settled-generation-one"
    record = {"tool_name": "save_entries", "arguments_text_sha256": adapter._hash(raw),
              "schema_sha256": adapter._hash(adapter._section_tool_schema("save_entries")),
              "diagnostic": {"path": "$", "reason": "invalid_json_syntax", "character_offset": 12,
                             "line": 1, "column": 13}, "handler_executed": False, "saved": False,
              "arguments_repaired_by_host": False, "agent_call_id": origin, "receipt_index": 0}
    def call(key, arguments):
        return {"id": key, "type": "function", "function": {"name": "save_entries", "arguments": arguments}}
    calls = ([call("good", '{"entries": []}')] if mixed else []) + [call("bad", raw)]
    if mixed:
        calls += [call("aborted-bad", '{"entries":'), call("aborted-valid", '{"entries": []}')]
    recovery = {"code": "SDK_TOOL_ARGUMENT_JSON", "agent_call_id": origin,
                "adapter_parse_error_sha256": adapter._hash(record),
                "tool_call_id_sha256": adapter._hash("bad"), "all_batch_tool_calls_paired": True,
                "generation_terminal_verified": True, "paired_batch_results": []}
    results = []
    for call in calls:
        key = call["id"]
        if key == "good":
            value = {"saved": True, "saved_count": 0}
        else:
            rejected = key == "bad"
            code = "SDK_TOOL_ARGUMENT_JSON" if rejected else "SDK_TOOL_BATCH_ABORTED"
            error = {"code": code, "handler_executed": False, "saved": False,
                     "observed_sdk_rejection": rejected}
            if rejected:
                error["adapter_parse_error_sha256"] = adapter._hash(record)
            else:
                error["blocked_by_tool_call_id_sha256"] = adapter._hash("bad")
            value = {"ok": False, "tool": "save_entries", "error": error,
                     "arguments_repaired_by_host": False}
            recovery["paired_batch_results"].append({"tool_name": "save_entries",
                "tool_call_id_sha256": adapter._hash(key), "arguments_text_sha256": adapter._hash(call["function"]["arguments"]),
                "result_sha256": adapter._hash(value),
                "code": code, "observed_sdk_rejection": rejected})
        results.append({"role": "tool", "tool_call_id": key, "content": packed(value)})
    payload = {"model": "synthetic", "messages": [{"role": "system", "content": "Synthetic instructions"},
        {"role": "user", "content": "Synthetic source"},
        {"role": "assistant", "content": "", "reasoning_content": "synthetic private text", "tool_calls": calls},
        *results, {"role": "user", "content": "Regenerate only the invalid JSON"}], "max_tokens": 262144,
        "tools": [{"type": "function", "function": {"name": "save_entries", "parameters": adapter._section_tool_schema("save_entries")}}]}
    session = {"argument_parse_errors": [record], "parse_recovery": [recovery], "tool_history_projections": []}
    return payload, session


def test_exact_mixed_batch_preserves_success_and_aborted_valid_pairs_and_originals():
    payload, session = fixture()
    before = deepcopy(payload)
    result = adapter._project_rejected_tool_history(payload, session)
    assert payload == before
    assistant = result["messages"][2]
    assert [x["id"] for x in assistant["tool_calls"]] == ["good", "aborted-valid"]
    assert assistant["content"] == before["messages"][2]["content"]
    assert assistant["reasoning_content"] == before["messages"][2]["reasoning_content"]
    for key in ["good", "aborted-valid"]:
        assert next(m for m in result["messages"] if m.get("tool_call_id") == key) == next(m for m in before["messages"] if m.get("tool_call_id") == key)
    note = next(m["content"] for m in result["messages"] if m.get("content", "").startswith("CITYAGENT_REJECTED_TOOL_HISTORY_V1"))
    quoted = json.loads(note.split("\n")[-1])
    assert [q["original_tool_call"]["id"] for q in quoted] == ["bad", "aborted-bad"]
    assert quoted[0]["original_tool_call"]["function"]["arguments"] == '{"entries": '
    assert all(q["handler_executed"] is False and q["saved"] is False for q in quoted)
    assert len(session["tool_history_projections"]) == 1
    HistoryPrefixGuard().observe_complete_request(result)
    for message in result["messages"]:
        for call in message.get("tool_calls", []):
            json.loads(call["function"]["arguments"])


def test_single_failed_pair_is_text_and_repeated_projection_is_idempotent():
    payload, session = fixture(False)
    result = adapter._project_rejected_tool_history(payload, session)
    assert not any(m.get("tool_calls") or m.get("role") == "tool" for m in result["messages"])
    assert adapter._project_rejected_tool_history(result, session) == result
    assert adapter._project_rejected_tool_history(payload, session) == result
    assert len(session["tool_history_projections"]) == 1
    HistoryPrefixGuard().observe_complete_request(result)


@pytest.mark.parametrize("fault", ["unknown_generation", "missing_owned", "unpaired", "wrong_raw",
    "wrong_result", "duplicate_receipt", "missing_result", "duplicate_id", "wrong_origin"])
def test_unproven_or_uncertain_malformed_history_fails_before_count(fault):
    payload, session = fixture()
    if fault == "unknown_generation": session["parse_recovery"][0]["generation_terminal_verified"] = False
    if fault == "missing_owned": session["argument_parse_errors"] = []
    if fault == "unpaired": session["parse_recovery"][0]["all_batch_tool_calls_paired"] = False
    if fault == "wrong_raw": payload["messages"][2]["tool_calls"][1]["function"]["arguments"] = '{"entries":'
    if fault == "wrong_result": payload["messages"][4]["content"] = '{"ok":true}'
    if fault == "duplicate_receipt": session["parse_recovery"] *= 2
    if fault == "missing_result": payload["messages"].pop(4)
    if fault == "duplicate_id": payload["messages"][2]["tool_calls"][0]["id"] = "bad"
    if fault == "wrong_origin": session["parse_recovery"][0]["agent_call_id"] = "unknown"
    with pytest.raises(adapter.AgentCallAssociationError):
        adapter._project_rejected_tool_history(payload, session)
    assert session["tool_history_projections"] == []


def test_valid_json_schema_error_and_business_strings_are_not_projected():
    payload, session = fixture(False)
    payload["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"entries":"[]"}'
    assert adapter._project_rejected_tool_history(payload, session) == payload
    assert session["tool_history_projections"] == []


def native_sdk_fixture(mixed=True):
    payload, session = fixture(mixed)
    session["argument_parse_errors"] = []
    recovery = session["parse_recovery"][0]
    recovery.pop("adapter_parse_error_sha256")
    session["sdk_tool_errors"] = [{"type": "ToolErrorAction", "tool_name": "read_section",
        "code": "SDK_TOOL_ARGUMENT_JSON", "agent_call_id": recovery["agent_call_id"],
        "tool_call_id_sha256": adapter._hash("bad"), "event_sha256": adapter._hash({"synthetic": True})}]
    for call in payload["messages"][2]["tool_calls"]:
        call["function"]["name"] = "read_section"
    for message in payload["messages"]:
        if message.get("role") != "tool" or message["tool_call_id"] == "good":
            continue
        value = json.loads(message["content"])
        value["tool"] = "read_section"
        value["error"].pop("adapter_parse_error_sha256", None)
        message["content"] = packed(value)
        proof = next(p for p in recovery["paired_batch_results"]
                     if p["tool_call_id_sha256"] == adapter._hash(message["tool_call_id"]))
        proof.update(tool_name="read_section", result_sha256=adapter._hash(value))
    return payload, session


def test_sdk_owned_rejection_is_projected_without_an_adapter_parse_receipt():
    payload, session = native_sdk_fixture()
    original = deepcopy(payload)
    projected = adapter._project_rejected_tool_history(payload, session)
    assert payload == original and session["argument_parse_errors"] == []
    assert [c["id"] for c in projected["messages"][2]["tool_calls"]] == ["good", "aborted-valid"]
    assert next(m for m in projected["messages"] if m.get("tool_call_id") == "good") == original["messages"][3]
    for message in projected["messages"]:
        for call in message.get("tool_calls", []):
            json.loads(call["function"]["arguments"])
    assert adapter._project_rejected_tool_history(payload, session) == projected
    HistoryPrefixGuard().observe_complete_request(projected)


@pytest.mark.parametrize("fault", ["no_event", "wrong_event_id", "wrong_origin", "wrong_tool",
    "wrong_event_type", "duplicate_event", "event_hash_missing", "root_arguments_changed",
    "wrong_result", "unsettled", "unpaired", "duplicate_proof"])
def test_native_sdk_rejection_still_requires_exact_settled_proof(fault):
    payload, session = native_sdk_fixture(False)
    event = session["sdk_tool_errors"][0]
    recovery = session["parse_recovery"][0]
    if fault == "no_event": session["sdk_tool_errors"] = []
    if fault == "wrong_event_id": event["tool_call_id_sha256"] = adapter._hash("other")
    if fault == "wrong_origin": event["agent_call_id"] = "other"
    if fault == "wrong_tool": event["tool_name"] = "read_source"
    if fault == "wrong_event_type": event["type"] = "ToolJsonSchemaErrorAction"
    if fault == "duplicate_event": session["sdk_tool_errors"] *= 2
    if fault == "event_hash_missing": event.pop("event_sha256")
    if fault == "root_arguments_changed": payload["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"cursor": '
    if fault == "wrong_result": payload["messages"][3]["content"] = '{"ok":true}'
    if fault == "unsettled": recovery["generation_terminal_verified"] = False
    if fault == "unpaired": recovery["all_batch_tool_calls_paired"] = False
    if fault == "duplicate_proof": recovery["paired_batch_results"] *= 2
    with pytest.raises(adapter.AgentCallAssociationError):
        adapter._project_rejected_tool_history(payload, session)
    assert session["tool_history_projections"] == []


def test_actual_openai_builder_counts_and_sends_same_stable_projected_body():
    payload, session = fixture()
    history = HistoryPrefixGuard()
    counted, sent = [], []
    def project(value):
        return adapter._project_rejected_tool_history(value, session)
    def prepare(value):
        history.observe_complete_request(value)
        for message in value["messages"]:
            for call in message.get("tool_calls", []):
                json.loads(call["function"]["arguments"])
        count = len(canonical(value["messages"]))
        assert count < 262144
        prepared = deepcopy(value); prepared["max_tokens"] = 262144-count
        counted.append(deepcopy(prepared))
        return prepared
    def transport(request):
        sent.append(json.loads(request.read()))
        return httpx.Response(200, json={"id": "mock-only", "object": "chat.completion", "model": "synthetic",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Synthetic"}}]})
    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        client = OpenAI(api_key="synthetic", base_url="http://127.0.0.1:9/v1", http_client=http, max_retries=0)
        with allocate_sdk_wire_requests("http://127.0.0.1:9/v1", prepare, project):
            client.chat.completions.create(**payload)
            next_payload = deepcopy(payload)
            next_payload["messages"].append({"role": "assistant", "content": "Synthetic next message"})
            client.chat.completions.create(**next_payload)
    assert sent == counted
    assert all(s["max_tokens"] == 262144-len(canonical(s["messages"])) for s in sent)
    assert history.snapshot()["poisoned"] is False
    assert len(sent) == 2


def test_allocation_still_rejects_post_projection_input_mutation():
    payload, session = fixture(False)
    calls = []
    def mutation(value):
        value["messages"].append({"role": "user", "content": "Unapproved"})
        value["max_tokens"] = 1
        return value
    with httpx.Client(transport=httpx.MockTransport(lambda request: calls.append(request))) as http:
        client = OpenAI(api_key="synthetic", base_url="http://127.0.0.1:9/v1", http_client=http, max_retries=0)
        with allocate_sdk_wire_requests("http://127.0.0.1:9/v1", mutation,
                lambda value: adapter._project_rejected_tool_history(value, session)):
            with pytest.raises(ValueError, match="changed the complete model input"):
                client.chat.completions.create(**payload)
    assert calls == []


def test_http_400_diagnostic_preserves_only_hash_and_safe_coordinates():
    private = "PRIVATE_SOURCE_OR_KEY_MUST_NOT_LEAK"
    raw = json.dumps({"error": {"message": private + " Expecting value: line 1 column 13 (char 12)",
                                "type": "BadRequestError", "code": 400}}).encode()
    class Opener:
        def open(self, request, **kwargs):
            raise HTTPError(request.full_url, 400, "synthetic", {}, io.BytesIO(raw))
    client = LocalModelClient("http://127.0.0.1:9", "synthetic")
    client._opener = Opener()
    with pytest.raises(RuntimeError, match="HTTP 400"):
        client._request("/tokenize", {"model": "synthetic", "messages": []}, kind="tokenize")
    receipt = client.calls[-1]
    assert private not in json.dumps(receipt)
    assert receipt["usage"] is None and receipt["output_sha256"] is None
    assert receipt["http_error_diagnostic"]["json_location"] == {"line": 1, "column": 13, "character_offset": 12}
    assert receipt["http_error_body_bytes_read"] == len(raw)
    assert receipt["http_error_body_truncated"] is False


@pytest.mark.parametrize("settled", [True, False])
def test_run_react_wires_projection_only_after_real_settlement_callback(settled):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from bridge_evidence.contest import document_context_budget as budget
    from bridge_evidence.contest import document_context_compression as native
    from test_section_agent_adapter import Tool, Parameters, fake_sdk, TextAction, basic
    from test_partial_native_batch_recovery import Memory, ToolResultMessage, call
    from test_strict_argument_parse_recovery import ToolErrorAction

    raw = '{"entries": '
    memory = Memory([call("save_entries", raw, "bad")])
    state, saved = {"steps": 0, "origin": "generation-one"}, []
    api = {"messages": SimpleNamespace(ToolResultMessage=ToolResultMessage),
           "memory": SimpleNamespace(MessageRole=SimpleNamespace(TOOL="tool"))}
    @contextmanager
    def allocator(url, prepare, project):
        state["project"] = project
        yield
    def script(registry, settings, outputs, prompt):
        state["steps"] += 1
        if state["steps"] == 1:
            try:
                registry["save_entries"]._normalize_and_validate_params(raw, strict_json=False)
            except adapter._StrictToolArgumentJSONError as exc:
                yield ToolErrorAction(str(exc))
            else:
                pytest.fail("Malformed JSON must fail before handler")
        else:
            assert settled is True
            result_message = memory.messages[1].chunks[0].content
            payload = {"model": "synthetic", "messages": [{"role": "assistant", "content": "",
                "tool_calls": [{"id": "bad", "type": "function", "function": {"name": "save_entries", "arguments": raw}}]},
                {"role": "tool", "tool_call_id": "bad", "content": packed(result_message.content)},
                {"role": "user", "content": "Repair JSON"}]}
            projected = state["project"](payload)
            assert not any(m.get("tool_calls") or m.get("role") == "tool" for m in projected["messages"])
            assert memory.messages[0].chunks[0].content.tool_call_messages[0].arguments == raw
            state["origin"] = "generation-two"
            args = registry["save_entries"]._normalize_and_validate_params(json.dumps({"entries": [basic()]}))
            outputs.append(registry["save_entries"](**args))
            yield TextAction()
    sdk, *_ = fake_sdk(script)
    adapter._REGISTRATIONS.clear(); adapter._REGISTERED_TOOL_CLASSES.clear()
    try:
        with patch.object(adapter, "_load_tongagents", return_value=sdk), \
                patch.object(adapter, "_load_tool_schema_api", return_value=(Tool, Parameters)), \
                patch.object(native, "load_native_api", return_value=api), \
                patch.object(budget, "allocate_sdk_wire_requests", allocator):
            outcome = adapter.run_react("Synthetic source", {"save_entries": lambda entries: saved.extend(entries) or {"saved": True}},
                {"profile": "section", "url": "http://127.0.0.1:9/v1", "model": "synthetic",
                 "dynamic_output_budget": True, "prepare_generation_request": lambda value: value,
                 "current_agent_call_id": lambda: state["origin"], "initial_memory_factory": lambda: memory,
                 "same_agent_parse_recovery": True, "before_same_agent_continue": lambda: None,
                 "parse_recovery_generation_is_settled": lambda: settled})
        assert outcome["step_invocations"] == (2 if settled else 1)
        assert outcome["parse_recovery"][0]["generation_terminal_verified"] is settled
        assert outcome["status"] == ("completed" if settled else "error")
        assert len(outcome["tool_history_projections"]) == (1 if settled else 0)
        assert len(saved) == (1 if settled else 0)
    finally:
        adapter._REGISTRATIONS.clear(); adapter._REGISTERED_TOOL_CLASSES.clear()
