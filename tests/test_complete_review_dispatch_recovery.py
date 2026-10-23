"""Exact native defaults must reach real gap feedback, never bypass identity.

Synthetic native frames and local workspaces only: no SDK, model, network or
service requests. These tests certify dispatcher/recovery behavior, not Word
quality or an actual delivery finish.
"""
from copy import deepcopy
import json
import socket
import subprocess
import threading
from types import SimpleNamespace

import pytest

from bridge_evidence.contest import document_context_compression as native
from bridge_evidence.contest import tong_react_adapter as adapter
from test_document_candidate_review import case
from test_main_gap_workflow import workflow
from test_partial_native_batch_recovery import Memory, ToolErrorAction, ToolResultMessage, call


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("No network, model, SDK or subprocess in dispatcher tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)
    assert adapter._ACTIVE_SESSION is None


def session(handler):
    return {"handlers": {"complete_review": handler}, "tool_calls": [],
        "tool_response_snapshots": {}, "tool_lock": threading.RLock(), "max_calls": 100,
        "profile": "reading", "tool_budget_exceeded": False,
        "run_id": "synthetic-run", "session_id": "synthetic-session",
        "current_agent_call_id": lambda: "generation-one", "argument_normalizations": []}


def frame(arguments, name="complete_review", key="review-one", origin="generation-one"):
    return {"generation_call_id": origin, "native_tool_call_id": key, "tool_name": name,
        "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))}


def checks():
    return {"病害明细": "已核对现有合成病害，历史未知仍保留",
        "维修建议": "已核对现有合成建议，未声称原文没有措施"}


@pytest.mark.parametrize("include_notes", [False, True])
def test_omitted_checks_reach_real_gap_rejection_then_model_checks_are_accepted(case, monkeypatch, include_notes):
    wf = workflow(case, failed=True)
    actual_handler = wf._additional_reading_handlers()["complete_review"]
    received = []

    def observed(**arguments):
        received.append((deepcopy(arguments), adapter.current_tool_call_identity()))
        return actual_handler(**arguments)

    state = session(observed)
    initial = case.workspace.checkpoint_path.read_bytes()
    original = {"basis": "已核对合成候选"}
    if include_notes:
        original["notes"] = "仍有历史未知"
    native_frame = frame(original)
    state["native_dispatch"] = native_frame
    with monkeypatch.context() as active:
        active.setattr(adapter, "_ACTIVE_SESSION", state)
        first = adapter._complete_review(**original)
        assert not first["ok"]
        assert first["result"]["error"]["code"] == "GAP_CHECK_REQUIRED"
        assert received[0][0] == {"basis": original["basis"], "notes": original.get("notes", "")}
        assert "gap_checks" not in received[0][0]
        assert received[0][1]["native_tool_call_id"] == "review-one"
        assert received[0][1]["args_sha256"] == adapter._hash(received[0][0])
        assert state["tool_calls"][0]["handler_result_sha256"] == adapter._hash(first["result"])
        assert case.workspace.checkpoint_path.read_bytes() == initial
        assert state["native_dispatch"] == native_frame
        assert wf._gap_active()
        corrected = {**original, "gap_checks": checks()}
        state["native_dispatch"] = frame(corrected, key="review-two")
        second = adapter._complete_review(**corrected)
        assert second["ok"] and second["result"]["model_gap_checks"] == checks()
        assert received[-1][0]["gap_checks"] == checks()
        assert received[-1][1]["native_tool_call_id"] == "review-two"
        assert not wf._gap_active() and wf._quick_current()
        assert adapter.current_tool_call_identity() is None
    assert [row["status"] for row in state["tool_calls"]] == ["failed", "success"]
    assert all(row.get("native_tool_call_id") for row in state["tool_calls"])


@pytest.mark.parametrize("fault", ["native_name", "origin", "basis", "checks", "explicit_null", "unknown_field"])
def test_real_wrapper_rejects_actual_dispatch_identity_mismatch_without_handler(case, monkeypatch, fault):
    wf = workflow(case, failed=True)
    actual_handler = wf._additional_reading_handlers()["complete_review"]
    received = []

    def observed(**arguments):
        received.append(deepcopy(arguments))
        return actual_handler(**arguments)

    state = session(observed)
    native_arguments = {"basis": "原始审核说明", "notes": "未决问题", "gap_checks": checks()}
    dispatch = deepcopy(native_arguments)
    state["native_dispatch"] = frame(native_arguments)
    if fault == "native_name":
        state["native_dispatch"]["tool_name"] = "finish_report"
    elif fault == "origin":
        state["native_dispatch"]["generation_call_id"] = "other-generation"
    elif fault == "basis":
        dispatch["basis"] = "被替换的审核说明"
    elif fault == "checks":
        dispatch["gap_checks"]["维修建议"] = "未经原始生成提出的替换判断"
    elif fault == "explicit_null":
        native_arguments["gap_checks"] = None
        dispatch["gap_checks"] = None
        state["native_dispatch"] = frame(native_arguments)
    else:
        native_arguments["unrecognized_argument"] = "不能忽略"
        state["native_dispatch"] = frame(native_arguments)
    before = case.workspace.checkpoint_path.read_bytes()
    native_frame_before = deepcopy(state["native_dispatch"])
    with monkeypatch.context() as active:
        active.setattr(adapter, "_ACTIVE_SESSION", state)
        response = adapter._complete_review(**dispatch)
        assert not response["ok"] and response["error"]["code"] == "AGENT_CALL_ID_INVALID"
        assert response["error"]["recoverable"] is False
        assert adapter.current_tool_call_identity() is None
    assert received == []
    assert case.workspace.checkpoint_path.read_bytes() == before
    assert wf._gap_active()
    assert state["native_dispatch"] == native_frame_before


def test_bound_defaults_omit_only_known_absent_none_and_preserve_explicit_null():
    original = {"basis": "synthetic"}
    frozen = deepcopy(original)
    actual = adapter._bound_dispatch_arguments("complete_review", original)
    assert actual == {"basis": "synthetic", "notes": ""}
    assert original == frozen
    explicit = {"basis": "synthetic", "gap_checks": None}
    assert adapter._bound_dispatch_arguments("complete_review", explicit) == {
        "basis": "synthetic", "notes": "", "gap_checks": None}
    assert adapter._bound_dispatch_arguments("finish", {}) == {}
    assert adapter._bound_dispatch_arguments("finish", {"self_check": None}) == {"self_check": None}
    assert adapter._bound_dispatch_arguments("read_source_range", {"evidence_id": "E1"}) == {
        "evidence_id": "E1", "start": 0, "end": None}
    assert adapter._bound_dispatch_arguments("review_candidates", {
        "operations": [], "task_reviews": []}) == {"operations": [], "task_reviews": [], "issue_reviews": None}


@pytest.mark.parametrize("gap_active", [False, True])
def test_later_bad_json_sibling_preserves_real_review_result_with_same_argument_hash(case, monkeypatch, gap_active):
    wf = workflow(case, failed=gap_active)
    state = session(wf._additional_reading_handlers()["complete_review"])
    original = {"basis": "合成审核", "notes": "尚未提供分类补漏结论"}
    state["native_dispatch"] = frame(original, key="valid-first")
    before = case.workspace.checkpoint_path.read_bytes()
    with monkeypatch.context() as active:
        active.setattr(adapter, "_ACTIVE_SESSION", state)
        actual_response = adapter._complete_review(**original)
    if gap_active:
        assert actual_response["result"]["error"]["code"] == "GAP_CHECK_REQUIRED"
        assert case.workspace.checkpoint_path.read_bytes() == before
    else:
        assert actual_response["ok"]
    after_actual_handler = case.workspace.checkpoint_path.read_bytes()
    calls = [call("complete_review", state["native_dispatch"]["arguments"], "valid-first"),
        call("save_recommendations", '{"entries":', "bad-second")]
    memory = Memory(calls)
    if gap_active:
        # An actual business rejection is already paired in native memory.
        # Snapshot-only failed returns are deliberately not restored by the
        # existing proof policy, which this default-value fix must not relax.
        memory.add_chunk(ToolResultMessage(name="complete_review", content=actual_response,
            tool_call_id="valid-first"), "tool")
    event = ToolErrorAction("save_recommendations", "bad-second")
    error = {"code": "SDK_TOOL_ARGUMENT_JSON", "tool_name": "save_recommendations",
        "agent_call_id": "generation-one", "tool_call_id_sha256": adapter._hash("bad-second"),
        "action": "Regenerate only the invalid tool arguments"}
    api = {"messages": SimpleNamespace(ToolResultMessage=ToolResultMessage),
        "memory": SimpleNamespace(MessageRole=SimpleNamespace(TOOL="tool"))}
    frozen_native = [call.arguments for call in calls]
    with monkeypatch.context() as local:
        local.setattr(native, "load_native_api", lambda: api)
        receipt, feedback = adapter._paired_parse_feedback(SimpleNamespace(), event, error,
            state["tool_calls"], context=SimpleNamespace(memory=memory),
            actual_responses=state["tool_response_snapshots"], argument_normalizations=[])
    assert receipt["all_batch_tool_calls_paired"] is True
    assert receipt["restored_actual_tool_result_count"] == int(not gap_active)
    if not gap_active:
        assert receipt["restored_actual_tool_results"][0]["handler_reexecuted"] is False
    assert memory.messages[1].chunks[0].content.content == actual_response
    assert memory.messages[2].chunks[0].content.content["error"]["handler_executed"] is False
    assert [call.arguments for call in calls] == frozen_native
    assert case.workspace.checkpoint_path.read_bytes() == after_actual_handler
    assert wf._gap_active() is gap_active


def test_terminal_batch_hash_uses_same_omitted_default_projection(case, monkeypatch):
    # Exercise only proof mechanics with complete_review as the designated
    # terminal tool. Production delivery still terminates at finish_report.
    wf = workflow(case)
    state = session(wf._additional_reading_handlers()["complete_review"])
    arguments = {"basis": "合成候选的快速审核"}
    state["native_dispatch"] = frame(arguments)
    with monkeypatch.context() as active:
        active.setattr(adapter, "_ACTIVE_SESSION", state)
        response = adapter._complete_review(**arguments)
    assert response["ok"]
    memory = Memory([call("complete_review", state["native_dispatch"]["arguments"], "review-one")])
    memory.add_chunk(ToolResultMessage(name="complete_review", content=response, tool_call_id="review-one"), "tool")
    settings = {"terminal_completion_tool": "complete_review",
        "terminal_handoff_state": lambda: {"origin_call_id": "generation-one", "synthetic": True},
        "terminal_generation_is_settled": lambda: True}
    native_frame = deepcopy(state["native_dispatch"])
    proof = adapter._terminal_section_handoff(SimpleNamespace(memory=memory), state, settings)
    assert proof["terminal_handoff_proof"]["batch_tool_count"] == 1
    assert state["native_dispatch"] == native_frame
    assert len(state["tool_calls"]) == 1


def test_missing_failed_business_return_is_still_not_restored_from_snapshot(case, monkeypatch):
    wf = workflow(case, failed=True)
    state = session(wf._additional_reading_handlers()["complete_review"])
    original = {"basis": "未提供分类结论"}
    state["native_dispatch"] = frame(original, key="failed-first")
    with monkeypatch.context() as active:
        active.setattr(adapter, "_ACTIVE_SESSION", state)
        response = adapter._complete_review(**original)
    assert response["result"]["error"]["code"] == "GAP_CHECK_REQUIRED"
    calls = [call("complete_review", state["native_dispatch"]["arguments"], "failed-first"),
        call("save_recommendations", '{"entries":', "bad-second")]
    memory = Memory(calls)
    native_calls, native_answers = native.native_tool_pairs(memory.get_messages())
    error = {"agent_call_id": "generation-one"}
    with pytest.raises(adapter.NativeBatchRecoveryError) as caught:
        adapter._prove_stopped_native_batch(memory.get_messages(), native_calls,
            native_answers, "bad-second", error, state["tool_calls"],
            actual_responses=state["tool_response_snapshots"], argument_normalizations=[])
    assert caught.value.proof_reason == "actual_return_execution_not_confirmed"
    assert len(memory.messages) == 1
    assert len(state["tool_calls"]) == 1
    assert wf._gap_active()
