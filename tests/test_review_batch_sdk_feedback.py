"""Review-batch feedback over synthetic SDK memory; no model or network calls."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
from jsonschema import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest import document_context_compression as native
from bridge_evidence.contest import tong_react_adapter as adapter
from bridge_evidence.contest.document_review_submission import (
    REVIEW_BATCH_LIMIT, REVIEW_COLLECTIONS, REVIEW_REJECTED_ACTION, submit_review_schema,
)
from test_partial_native_batch_recovery import Memory, ToolErrorAction, ToolResultMessage, call
from test_section_agent_adapter import Parameters, Tool


class ToolJsonSchemaErrorAction:
    def __init__(self, tool_call_id):
        self.tool_call_id = tool_call_id


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("This test must not start a process, SDK or network connection")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)
    monkeypatch.setattr(adapter, "_load_tongagents", blocked)
    monkeypatch.setattr(adapter, "_load_tool_schema_api", lambda: (Tool, Parameters))
    monkeypatch.setattr(native, "load_native_api", lambda: {
        "messages": SimpleNamespace(ToolResultMessage=ToolResultMessage),
        "memory": SimpleNamespace(MessageRole=SimpleNamespace(TOOL="tool")),
    })


def paired(memory, *, key="bad", name="submit_review", code="SDK_TOOL_ARGUMENT_SCHEMA", state=None):
    event = (ToolJsonSchemaErrorAction(key) if code == "SDK_TOOL_ARGUMENT_SCHEMA"
             else ToolErrorAction(name, key))
    error = {"code": code, "tool_name": name, "agent_call_id": "generation-one",
        "tool_call_id_sha256": adapter._hash(key),
        "action": "Repair the indicated argument paths."}
    return adapter._paired_parse_feedback(SimpleNamespace(), event, error,
        (state or {}).get("tool_calls", []), context=SimpleNamespace(memory=memory),
        tool_schema=adapter._normalization_schema_for_tool(name),
        actual_responses=(state or {}).get("tool_response_snapshots"))


def results(memory):
    return {chunk.content.tool_call_id: chunk.content.content
        for message in memory.get_messages() for chunk in message.chunks
        if isinstance(chunk.content, ToolResultMessage)}


def assert_rejected(result, count):
    assert result["error"]["handler_executed"] is False
    assert result["error"]["saved"] is False
    assert result["arguments_repaired_by_host"] is False
    receipt = result["batch_receipt"]
    assert receipt["status"] == "rejected"
    assert receipt["applied"] is False
    assert receipt["action_count"] == count
    assert receipt["limit"] == REVIEW_BATCH_LIMIT
    assert receipt["accepted"] == {key: [] for key in REVIEW_COLLECTIONS}
    assert receipt["workspace_version"] is None  # SDK cannot observe the actual store.
    assert receipt["instruction"] == result["error"]["action"] == REVIEW_REJECTED_ACTION


def test_actual_registered_schema_rejects_six_before_handler_and_reports_whole_batch():
    args = {"selections": [f"candidate-{i}" for i in range(REVIEW_BATCH_LIMIT + 1)]}
    tool = type("SyntheticReview", (Tool,), {})()
    adapter._install_tool_schema(tool, "submit_review", submit_review_schema())
    with pytest.raises(ValidationError):
        tool._normalize_and_validate_params(json.dumps(args))
    raw = json.dumps(args)
    memory = Memory([call("submit_review", raw, "bad")])
    _, feedback = paired(memory)
    result = results(memory)["bad"]
    assert_rejected(result, 6)
    assert any(issue["path"] == "selections" and issue["reason"] == "maxItems"
        for issue in result["error"]["argument_repair"]["issues"])
    assert REVIEW_REJECTED_ACTION in result["error"]["argument_repair"]["action"]
    assert REVIEW_REJECTED_ACTION in feedback
    assert memory.messages[0].chunks[0].content.tool_call_messages[0].arguments == raw


def test_bad_last_item_does_not_claim_earlier_items_in_that_submission_were_saved():
    args = {"selections": ["candidate-one"],
        "patches": [{"candidate_id": "candidate-two", "changes": {}}]}
    memory = Memory([call("submit_review", json.dumps(args), "bad")])
    paired(memory)
    result = results(memory)["bad"]
    assert_rejected(result, 2)
    assert result["error"]["argument_repair"]["issues"][0]["path"] == "patches[0].changes"
    assert "all still-needed decisions from THIS failed batch" in result["error"]["action"]


@pytest.mark.parametrize("raw,code", [
    ('{"reviews":', "SDK_TOOL_ARGUMENT_JSON"),
    ('{"reviews":"[invalid JSON"}', "SDK_TOOL_ARGUMENT_SCHEMA"),
])
def test_malformed_arguments_do_not_invent_an_action_count_or_repair_values(raw, code):
    memory = Memory([call("submit_review", raw, "bad")])
    receipt, feedback = paired(memory, code=code)
    assert_rejected(results(memory)["bad"], None)
    assert REVIEW_REJECTED_ACTION in feedback
    assert memory.messages[0].chunks[0].content.tool_call_messages[0].arguments == raw
    assert raw not in json.dumps(receipt)


def saved_prefix(memory):
    args = {"selections": ["already-saved"]}
    batch = memory.messages[0].chunks[0].content.tool_call_messages
    batch.insert(0, call("submit_review", json.dumps(args), "saved"))
    captured = []
    saved = {"ok": True, "changed": True, "version": 17, "batch_receipt": {
        "status": "saved", "workspace_version": 17, "accepted": {"selections": ["already-saved"]}}}
    state = {"handlers": {"submit_review": lambda **kwargs: captured.append(kwargs) or deepcopy(saved)},
        "tool_calls": [], "tool_response_snapshots": {}, "max_calls": 20,
        "profile": "reading", "tool_lock": threading.RLock(),
        "current_agent_call_id": lambda: "generation-one"}
    bound = inspect.signature(adapter._TOOLS["submit_review"][0]).bind(**args)
    bound.apply_defaults()
    response = adapter._invoke_active(state, "submit_review", dict(bound.arguments))
    assert response["ok"] is True
    return state, response, captured


@pytest.mark.parametrize("retained", [False, True])
def test_preceding_successful_review_is_preserved_or_restored_without_reexecution(retained):
    memory = Memory([call("submit_review", '{"reviews":', "bad")])
    state, saved, captured = saved_prefix(memory)
    if retained:
        memory.add_chunk(ToolResultMessage(name="submit_review", content=saved, tool_call_id="saved"), "tool")
    receipt, feedback = paired(memory, state=state, code="SDK_TOOL_ARGUMENT_JSON")
    actual = results(memory)
    assert actual["saved"] == saved
    assert actual["saved"]["result"]["batch_receipt"]["status"] == "saved"
    assert "batch_receipt" not in actual["saved"]  # Never relabel the successful SDK envelope.
    assert_rejected(actual["bad"], None)
    assert receipt["preserved_executed_tool_count"] == 1
    assert receipt["restored_actual_tool_result_count"] == int(not retained)
    assert len(captured) == len(state["tool_calls"]) == 1
    assert "Do not resubmit earlier successful batches" in feedback


def test_later_unexecuted_review_gets_receipt_even_when_other_tool_caused_rejection():
    memory = Memory([call("search_in_document", '{"query":', "bad"),
        call("submit_review", '{"selections":["not-yet-saved"]}', "later")])
    _, feedback = paired(memory, name="search_in_document", code="SDK_TOOL_ARGUMENT_JSON")
    actual = results(memory)
    assert "batch_receipt" not in actual["bad"]
    assert_rejected(actual["later"], 1)
    assert actual["later"]["error"]["code"] == "SDK_TOOL_BATCH_ABORTED"
    assert actual["later"]["error"]["observed_sdk_rejection"] is False
    assert "For the unexecuted submit_review" in feedback
    assert REVIEW_REJECTED_ACTION in feedback


@pytest.mark.parametrize("fault", ["missing_execution_proof", "already_answered"])
def test_no_rejection_receipt_is_fabricated_when_unexecuted_identity_is_unproved(fault):
    memory = Memory([call("submit_review", '{"reviews":', "bad")])
    if fault == "missing_execution_proof":
        memory.messages[0].chunks[0].content.tool_call_messages.insert(0,
            call("submit_review", '{"selections":["unknown-status"]}', "unknown"))
    else:
        memory.add_chunk(ToolResultMessage(name="submit_review", content={"ok": True}, tool_call_id="bad"), "tool")
    before = list(memory.messages)
    before_results = deepcopy(results(memory))
    before_arguments = [item.arguments for item in memory.messages[0].chunks[0].content.tool_call_messages]
    with pytest.raises(adapter.AgentCallAssociationError):
        paired(memory, code="SDK_TOOL_ARGUMENT_JSON")
    assert memory.messages == before
    assert results(memory) == before_results
    assert [item.arguments for item in memory.messages[0].chunks[0].content.tool_call_messages] == before_arguments
    assert not any("batch_receipt" in result for result in results(memory).values())


def test_nonreview_feedback_remains_outside_review_batch_contract():
    memory = Memory([call("search_in_document", '{"query":', "bad")])
    _, feedback = paired(memory, name="search_in_document", code="SDK_TOOL_ARGUMENT_JSON")
    assert "batch_receipt" not in results(memory)["bad"]
    assert REVIEW_REJECTED_ACTION not in feedback


def test_tool_description_sets_aggregate_not_merge_member_limit():
    description = adapter._TOOLS["submit_review"][1]
    assert f"合计最多{REVIEW_BATCH_LIMIT}项" in description
    assert "一组合并算1项，不拆其成员" in description
    assert "等待回执" in description and "不重交以前成功的批次" in description
