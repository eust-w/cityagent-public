"""Native handoff proof with plain local memory, never an SDK/model execution."""
from copy import deepcopy
import json
import socket
import subprocess
import threading
from types import SimpleNamespace

import pytest

from bridge_evidence.contest import tong_react_adapter as adapter


class ToolResultMessage:
    def __init__(self, name, content, tool_call_id):
        self.name, self.content, self.tool_call_id = name, deepcopy(content), tool_call_id

    def __eq__(self, other):
        return type(other) is type(self) and vars(self) == vars(other)


class Memory:
    def __init__(self, messages):
        self.messages = messages

    def get_messages(self):
        return self.messages[:]

    def add_chunk(self, *args, **kwargs):
        raise AssertionError("A terminal proof must never synthesize native memory")


def message(content):
    return SimpleNamespace(chunks=[SimpleNamespace(content=content)])


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("No network, subprocess, SDK or model in boundary tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)


@pytest.fixture
def boundary():
    origin = "actual-synthetic-generation"
    specifications = [
        ("read_source", {"evidence_id": "E-synthetic", "restart": False}),
        ("complete_section", {"summary": "Saved this assigned synthetic section", "unresolved": []}),
    ]
    calls, rows, returns = [], [], []
    for index, (name, arguments) in enumerate(specifications):
        call = SimpleNamespace(tool_name=name, arguments=json.dumps(arguments), tool_call_id=f"tool-{index}")
        response = {"ok": True, "tool": name, "result": {"synthetic": index}}
        calls.append(call)
        returns.append(ToolResultMessage(name, response, call.tool_call_id))
        rows.append({"name": name, "args_sha256": adapter._hash(arguments),
                     "result_sha256": adapter._hash(response), "status": "success", "agent_call_id": origin})
    memory = Memory([message(SimpleNamespace(tool_call_messages=calls))] + [message(x) for x in returns])
    state = {"tool_lock": threading.RLock(), "tool_calls": rows, "tool_budget_exceeded": False,
             "current_agent_call_id": lambda: origin, "argument_normalizations": [],
             "native_final_event": False, "response_text": ""}
    handoff = {"origin_call_id": origin, "entries_content_sha256": "synthetic-content-hash",
               "document_delivery_complete": False}
    settings = {"terminal_handoff_state": lambda: deepcopy(handoff),
                "terminal_generation_is_settled": lambda: True}
    return SimpleNamespace(memory=memory, context=SimpleNamespace(memory=memory), session=state,
                           settings=settings, origin=origin, handoff=handoff, calls=calls, returns=returns)


def prove(boundary):
    before = deepcopy(boundary.memory.messages)
    try:
        return adapter._terminal_section_handoff(boundary.context, boundary.session, boundary.settings)
    finally:
        assert boundary.memory.messages == before
        assert boundary.session["native_final_event"] is False
        assert boundary.session["response_text"] == ""


def test_complete_same_batch_returns_proof_without_fake_final_or_memory(boundary):
    proof = prove(boundary)
    assert proof["terminal_handoff_origin_call_id"] == boundary.origin
    assert proof["terminal_handoff_sha256"] == adapter._hash(boundary.handoff)
    assert proof["terminal_handoff_proof"] == {
        "all_batch_tool_calls_paired": True, "batch_tool_count": 2,
        "native_tool_call_ids_sha256": adapter._hash([c.tool_call_id for c in boundary.calls])}
    assert "native_final_event" not in proof and "response_text" not in proof


def test_document_finish_uses_same_exact_batch_proof_without_extra_generation(boundary):
    call = boundary.calls[-1]
    call.tool_name = 'finish_report'
    arguments = {'basis': '已检查真实文件', 'notes': ''}
    call.arguments = json.dumps(arguments)
    response = {'ok': True, 'tool': 'finish_report', 'result': {'ok': True}}
    boundary.returns[-1].name = 'finish_report'
    boundary.returns[-1].content = response
    boundary.session['tool_calls'][-1].update(name='finish_report',
        args_sha256=adapter._hash(arguments), result_sha256=adapter._hash(response))
    boundary.settings['terminal_completion_tool'] = 'finish_report'
    assert prove(boundary)['terminal_handoff_proof']['batch_tool_count'] == 2
    # A claimed current state alone cannot turn a rejected finish into success.
    response['result']['ok'] = False
    boundary.session['tool_calls'][-1]['result_sha256'] = adapter._hash(response)
    with pytest.raises(adapter.AgentCallAssociationError, match='successful completed'):
        prove(boundary)


@pytest.mark.parametrize("missing", [0, 1])
def test_missing_native_result_refuses_even_with_all_handler_receipts(boundary, missing):
    boundary.memory.messages.pop(missing + 1)
    with pytest.raises(adapter.AgentCallAssociationError, match="unpaired"):
        prove(boundary)


def test_active_callback_on_another_thread_refuses(boundary):
    held, release = threading.Event(), threading.Event()
    def callback():
        with boundary.session["tool_lock"]:
            held.set()
            assert release.wait(3)
    thread = threading.Thread(target=callback)
    thread.start()
    try:
        assert held.wait(3)
        with pytest.raises(adapter.AgentCallAssociationError, match="active callback"):
            prove(boundary)
    finally:
        release.set()
        thread.join(3)
        assert not thread.is_alive()


@pytest.mark.parametrize("origin", [None, "another-generation"])
def test_generation_mismatch_refuses(boundary, origin):
    boundary.session["current_agent_call_id"] = lambda: origin
    with pytest.raises(adapter.AgentCallAssociationError, match="settled current generation"):
        prove(boundary)


@pytest.mark.parametrize("settled", [False, None, 1])
def test_generation_must_be_explicitly_settled(boundary, settled):
    boundary.settings["terminal_generation_is_settled"] = lambda: settled
    with pytest.raises(adapter.AgentCallAssociationError, match="settled current generation"):
        prove(boundary)


@pytest.mark.parametrize("flag", ["agent_call_id_invalid", "tool_budget_exceeded"])
def test_failure_latches_forbid_terminal_acceptance(boundary, flag):
    boundary.session[flag] = True
    with pytest.raises(adapter.AgentCallAssociationError, match="settled current generation"):
        prove(boundary)


def test_no_handoff_continues_without_even_consulting_settlement(boundary):
    boundary.settings["terminal_handoff_state"] = lambda: None
    boundary.settings["terminal_generation_is_settled"] = lambda: pytest.fail("No handoff to settle")
    assert prove(boundary) is None


def test_mutation_invalidated_callback_none_does_not_reuse_prior_handoff(boundary):
    assert prove(boundary) is not None
    boundary.settings["terminal_handoff_state"] = lambda: None
    assert prove(boundary) is None


@pytest.mark.parametrize("field,value", [
    ("name", "get_neighbors"), ("args_sha256", "0" * 64),
    ("result_sha256", "0" * 64), ("agent_call_id", "another-generation"),
])
def test_same_count_wrong_execution_receipt_is_not_a_batch_proof(boundary, field, value):
    boundary.session["tool_calls"][0][field] = value
    with pytest.raises(adapter.AgentCallAssociationError):
        prove(boundary)


def test_changed_actual_native_return_refuses_original_receipt_hash(boundary):
    boundary.returns[0].content["result"]["synthetic"] = "changed after real return"
    with pytest.raises(adapter.AgentCallAssociationError, match="exactly proved"):
        prove(boundary)


def test_changed_actual_native_arguments_refuse_original_receipt_hash(boundary):
    arguments = json.loads(boundary.calls[0].arguments)
    arguments["evidence_id"] = "E-other"
    boundary.calls[0].arguments = json.dumps(arguments)
    with pytest.raises(adapter.AgentCallAssociationError, match="exactly proved"):
        prove(boundary)


def test_duplicate_identical_siblings_consume_distinct_equal_receipts(boundary):
    duplicate = deepcopy(boundary.calls[0]); duplicate.tool_call_id = "tool-identical"
    boundary.calls.insert(1, duplicate)
    boundary.memory.messages.insert(2, message(ToolResultMessage(
        duplicate.tool_name, boundary.returns[0].content, duplicate.tool_call_id)))
    boundary.session["tool_calls"].insert(1, deepcopy(boundary.session["tool_calls"][0]))
    assert prove(boundary)["terminal_handoff_proof"]["batch_tool_count"] == 3


def test_duplicate_receipt_cannot_cover_different_sibling(boundary):
    different = deepcopy(boundary.calls[0]); different.tool_call_id = "tool-different"
    args = json.loads(different.arguments); args["evidence_id"] = "E-other"
    different.arguments = json.dumps(args)
    boundary.calls.insert(1, different)
    boundary.memory.messages.insert(2, message(ToolResultMessage(
        different.tool_name, boundary.returns[0].content, different.tool_call_id)))
    boundary.session["tool_calls"].insert(1, deepcopy(boundary.session["tool_calls"][0]))
    with pytest.raises(adapter.AgentCallAssociationError, match="exactly proved"):
        prove(boundary)


def test_latest_native_batch_must_itself_contain_complete_section(boundary):
    later = SimpleNamespace(tool_name="read_source", arguments=boundary.calls[0].arguments,
                            tool_call_id="later-call")
    boundary.memory.messages += [message(SimpleNamespace(tool_call_messages=[later])),
                                 message(ToolResultMessage(later.tool_name, boundary.returns[0].content, later.tool_call_id))]
    with pytest.raises(adapter.AgentCallAssociationError, match="complete executed native batch"):
        prove(boundary)


@pytest.mark.parametrize("has_audit", [False, True])
def test_container_normalization_needs_actual_exact_validation_audit(boundary, has_audit):
    original = json.loads(boundary.calls[1].arguments)
    original["unresolved"] = "[]"
    boundary.calls[1].arguments = json.dumps(original)
    schema = adapter._normalization_schema_for_tool("complete_section")
    normalized, changes, _ = adapter._normalize_container_arguments(original, schema)
    if has_audit:
        boundary.session["argument_normalizations"] = [{
            "tool_name": "complete_section", "agent_call_id": boundary.origin,
            "original_arguments_sha256": adapter._hash(original),
            "normalized_arguments_sha256": adapter._hash(normalized),
            "schema_sha256": adapter._hash(schema), "conversions": changes,
            "native_validation_passed": True}]
        assert prove(boundary) is not None
    else:
        with pytest.raises(adapter.AgentCallAssociationError, match="exactly proved"):
            prove(boundary)
