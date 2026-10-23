"""Speed-mode limits and degraded delivery never fabricate evidence or file checks."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
from jsonschema import Draft202012Validator

from test_document_candidate_review import case
from test_direct_delivery_worker import section
from test_managed_file_self_check import Harness
from bridge_evidence.contest import document_multiagent_worker as worker
from bridge_evidence.contest.document_multiagent import MultiAgentSettings
from bridge_evidence.contest.document_speed import QuickReviewTools, budget_receipt, constrain_tools, quick_review_schema
from bridge_evidence.contest.document_speed_workflow import SpeedMergeWorkflow
from bridge_evidence.contest.document_workspace import DocumentWorkspace, WorkspaceIdentityError
from bridge_evidence.contest.evidence_manifest import manifest_from_text


def test_fast_mode_is_explicit():
    assert not MultiAgentSettings().speed_priority
    with pytest.raises(ValueError):
        MultiAgentSettings(speed_priority=True)
    assert MultiAgentSettings(direct_delivery=True, speed_priority=True).speed_priority


def test_stage_tool_selection_avoids_forced_grammar_and_preserves_history():
    payload = {"messages": [{"role": "user", "content": "原文"}], "tools": [
        {"type": "function", "function": {"name": n}} for n in ("submit_section", "read_source")]}
    original = deepcopy(payload)
    result = constrain_tools(payload, {"submit_section"}, named="submit_section")
    assert result["tool_choice"] == "auto"
    assert result["messages"] == original["messages"] and payload == original
    assert len(result["tools"]) == 1
    with pytest.raises(ValueError):
        constrain_tools(payload, {"unknown"})


def test_main_selection_enforces_output_cap_without_changing_history_or_context():
    from bridge_evidence.contest.local_model_client import LocalModelClient
    workflow = object.__new__(SpeedMergeWorkflow)
    workflow.client = LocalModelClient("http://127.0.0.1:9", "own-model", max_model_len=262144,
        max_output_tokens=262144, dynamic_output_budget=True)
    workflow._finish_is_current = lambda: False
    workflow._speed_progress = lambda: {"stage": "inspect_delivery"}
    payload = {"messages": [{"role": "user", "content": "完整材料，不裁剪"}],
        "tools": [{"type": "function", "function": {"name": "inspect_delivery"}}]}
    selected = workflow._before_continuous_request(payload)
    assert workflow.client.generation_output_cap == 1024
    assert workflow.client.max_model_len == 262144
    assert selected["messages"] == payload["messages"]
    independent_client = LocalModelClient("http://127.0.0.1:9", "own-model")
    assert independent_client.generation_output_cap is None


def test_native_tool_selection_precedes_allocation_and_cannot_rewrite_history():
    from bridge_evidence.contest.tong_react_adapter import _select_generation_tools
    payload = {"messages": [{"role": "user", "content": "完整原文"}], "temperature": 1.0,
        "tools": [{"type": "function", "function": {"name": "submit_section"}}]}
    selected = _select_generation_tools(payload, lambda p: constrain_tools(p, {"submit_section"}, named="submit_section"))
    assert selected["tool_choice"] == "auto"
    for key, value in (("messages", []), ("temperature", 0), ("max_tokens", 1)):
        with pytest.raises(ValueError, match="history or inference"):
            _select_generation_tools(payload, lambda p: dict(p, **{key: value}))
    with pytest.raises(ValueError, match="unregistered schema"):
        _select_generation_tools(payload, lambda p: dict(p, tools=[{"function": {"name": "invented"}}]))


def test_budget_uses_actual_requests_not_queue_waiting_or_tokenizer_calls():
    calls = [{"kind": "tokenize", "seconds": 50},
        {"kind": "agent_generation", "seconds": 40, "queue_seconds": 200},
        {"kind": "agent_generation", "seconds": 80, "generation_attempted": False}]
    receipt = budget_receipt(calls, soft_seconds=60, max_calls=6, saved_counts={"total": 2})
    assert receipt["active_request_seconds"] == 40 and receipt["soft_seconds_remaining"] == 20
    assert not receipt["wrap_up"] and receipt["actual_model_requests"] == 1
    assert budget_receipt(calls, soft_seconds=30, max_calls=6)["wrap_up"]
    assert budget_receipt(calls, soft_seconds=60, max_calls=3)["wrap_up"]


def test_tool_choices_use_same_rendered_input_but_distinct_request_hashes():
    from bridge_evidence.contest.document_context_budget import tokenize_chat_payload, request_input_hash
    payload = {"model": "own-model", "messages": [{"role": "user", "content": "完整原文"}],
        "tools": [{"type": "function", "function": {"name": "submit_section", "parameters": {"type": "object"}}}]}
    hashes = set()
    for choice in ("auto", "required", {"type": "function", "function": {"name": "submit_section"}}):
        request = dict(payload, tool_choice=choice)
        assert tokenize_chat_payload(request) == tokenize_chat_payload(payload)
        hashes.add(request_input_hash(request))
    assert len(hashes) == 3
    with pytest.raises(ValueError, match="exclusion policy"):
        tokenize_chat_payload(dict(payload, tool_choice="none"))
    del payload["tools"]
    assert "tools" not in tokenize_chat_payload(dict(payload, tool_choice="none"))
    for choice in ("required", {"type": "function", "function": {"name": "missing"}}):
        with pytest.raises(ValueError):
            tokenize_chat_payload(dict(payload, tool_choice=choice))


def test_quick_review_accepts_gaps_without_marking_questions_resolved(case):
    reviewer = QuickReviewTools(case.workspace, case.handoff, case.summary)
    before = case.workspace.snapshot()
    result = reviewer.quick_review("快速查看候选后保留主要信息", ["历史状态未核实"], selections=[case.ids["left_name"]])
    assert result["ok"], result
    state = case.workspace.snapshot()
    assert state["facts"] == before["facts"]
    assert all(q["status"] == "unknown" for q in state["questions"].values())
    assert state["draft"]["基本信息"]["桥梁名称"] == "桥梁甲"
    assert result["exhaustive_quality_verified"] is False and result["remaining_issues"]


def test_bad_quick_patch_cannot_defer_questions_or_change_existing_output(case):
    reviewer = QuickReviewTools(case.workspace, case.handoff, case.summary)
    before = case.workspace.checkpoint_path.read_bytes()
    result = reviewer.quick_review("允许缺项", ["未知"], patches=[{"candidate_id": "foreign", "changes": {"value": 1}}])
    assert not result["ok"]
    assert case.workspace.checkpoint_path.read_bytes() == before


def test_quick_schema_allows_more_than_five_decisions_and_never_empty_patch():
    payload = {"basis": "快速审核", "remaining_issues": [], "selections": [f"candidate-{i}" for i in range(12)]}
    Draft202012Validator(quick_review_schema()).validate(payload)
    payload["patches"] = [{"candidate_id": "candidate", "changes": {}}]
    assert list(Draft202012Validator(quick_review_schema()).iter_errors(payload))


def test_unread_ranges_are_preserved_and_policy_bound_to_workspace_identity(tmp_path):
    manifest = manifest_from_text("已读。\n未读。")
    workspace = DocumentWorkspace(manifest, tmp_path / "speed", {}, require_semantic_review=False, allow_quality_gaps=True)
    workspace.apply_update({"base_version": 0, "draft": {"基本信息": {}, "病害明细": [], "维修建议": []},
        "draft_fact_ids": [], "draft_bindings": {}})
    status = workspace.delivery_status()
    assert status["ready"] and status["reading_progress"]["unread_ranges"]
    assert status["quality_gaps_allowed"] and workspace.snapshot()["reads"] == {}
    with pytest.raises(WorkspaceIdentityError):
        DocumentWorkspace(manifest, tmp_path / "speed", {}, require_semantic_review=False)
    strict = DocumentWorkspace(manifest, tmp_path / "strict", {}, require_semantic_review=False)
    assert "unread_source_ranges" in strict.delivery_status()["blockers"]


def test_speed_child_can_handoff_partial_reading_with_explicit_gap(section, monkeypatch):
    unit = section.manifest["units"][0]
    monkeypatch.setattr(worker, "section_prompt", lambda *a, **k: (
        unit["id"] + ":" + unit["text"], [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}]))
    section.request["speed_priority"] = True
    def script(handlers, gateway, config):
        assert "允许遗漏" in config["system_prompt"]
        assert "quality_gaps_allowed" in config["before_native_model"]()
        entry = {**section.defect["row"], "source_refs": section.defect["source_refs"], "status": "explicit"}
        assert handlers["save_defects"]([entry])["ok"]
        receipt = handlers["handoff_section"]("其他材料未读完")
        assert receipt["handed_off"]
    result = section.run(script)
    assert result["status"] == "completed" and result["handoff"]["unread_primary"]
    assert len(result["entries"]["reads"]) == 1
    assert not result["handoff"]["exhaustive_quality_verified"]


def test_file_check_accepts_declared_quality_gap_only_in_opted_in_mode(tmp_path):
    h = Harness(tmp_path / "strict")
    h.read_all()
    statement = {"inspected_export": True, "complete": True, "basis": "快速检查实际文件", "remaining_issues": ["部分成因未提取"]}
    assert not h.finish(statement)["ok"]
    h.wf.settings = replace(h.wf.settings, allow_quality_gaps=True)
    assert h.finish(statement)["ok"]
    assert h.wf.model_self_check["model_statement"] == statement
    assert h.wf.model_self_check["exhaustive_quality_verified"] is False
    h.revise()
    assert not h.finish(statement)["ok"]  # Old Word must not satisfy new content.


def test_speed_flag_cannot_replace_actual_word_readback(tmp_path):
    h = Harness(tmp_path)
    h.wf.settings = replace(h.wf.settings, allow_quality_gaps=True)
    assert not h.finish({"inspected_export": True, "complete": True,
        "basis": "允许内容缺项", "remaining_issues": ["历史未核对"]})["ok"]


def test_quick_ack_is_invalidated_by_later_candidate_change(case):
    workflow = object.__new__(SpeedMergeWorkflow)
    workflow.workspace, workflow.child_summary = case.workspace, case.summary
    workflow.quick_review_receipt = {"semantic_state_sha256": workflow._semantic_identity(case.workspace.snapshot())}
    assert not workflow._coverage_pending(case.workspace.snapshot())
    case.workspace.apply_update({"base_version": case.workspace.snapshot()["version"],
        "questions": [{"id": case.issue, "status": "unknown", "reason": "新增变化"}]})
    assert workflow._coverage_pending(case.workspace.snapshot())


def test_speed_finish_with_missing_advice_text_still_requires_and_binds_real_word(tmp_path):
    """v19 C1: an exportable row must not hit legacy meaningless_row at finish."""
    from bridge_evidence.contest.document_workflow import DocumentWorkflow
    h = Harness(tmp_path)
    h.wf.settings = replace(h.wf.settings, allow_quality_gaps=True)
    h.wf.__class__ = SpeedMergeWorkflow
    h.wf.child_summary = []
    h.wf.workspace.apply_update({"base_version": h.wf.workspace.snapshot()["version"],
        "draft_patch": [{"op": "append_row", "section": "维修建议",
            "row": {"病害部位": "合成桥", "建议类别": "未知"}, "fact_ids": ["name"]}]})
    state = h.wf.workspace.snapshot()
    with pytest.raises(ValueError, match="meaningless_row"):
        DocumentWorkflow._validate_delivery_schema(state["draft"])
    h.wf.quick_review_receipt = {"semantic_state_sha256": h.wf._semantic_identity(state),
        "remaining_issues": ["建议内容未提取"]}
    statement = {"inspected_export": True, "complete": True, "basis": "实际文件保留缺项",
        "remaining_issues": ["建议内容未提取"]}
    assert not h.finish(statement)["ok"]
    h.read_all()
    assert h.finish(statement)["ok"]
    assert h.wf._finish_is_current()
    assert (tmp_path / "quality-gaps.json").exists()
    assert "建议内容" not in h.wf.workspace.snapshot()["draft"]["维修建议"][0]
    bad = deepcopy(state["draft"])
    bad["维修建议"][0]["will_be_silently_dropped"] = "invalid"
    with pytest.raises(ValueError, match="unsupported fields"):
        SpeedMergeWorkflow._validate_delivery_schema(bad)


def test_failed_source_lookup_consumes_allowance_without_repeating_lookup(monkeypatch):
    from bridge_evidence.contest.document_multiagent import MergeWorkflow
    wf = object.__new__(SpeedMergeWorkflow)
    monkeypatch.setattr(MergeWorkflow, "_observe_native_tool", lambda *a: None)
    wf._finish_is_current = lambda: False
    wf._speed_progress = lambda: {"stage": "review_edits", "budget": {"wrap_up": False}}
    request = {"messages": [], "tools": [{"function": {"name": name}} for name in
        ("complete_review", "search_in_document", "read_source")]}
    assert len(wf._before_continuous_request(request)["tools"]) == 3
    wf._observe_native_tool("search_in_document", {}, {"ok": False, "error": {"code": "TOOL_ARGUMENTS"}})
    selected = wf._before_continuous_request(request)
    assert [t["function"]["name"] for t in selected["tools"]] == ["complete_review"]


@pytest.mark.parametrize("missing", [None, "workspace", "workflow", "statement"])
def test_batch_publication_respects_bound_quality_policy_and_same_file(tmp_path, missing):
    from pathlib import Path
    from test_document_multiagent_batch import _word_result
    from bridge_evidence.contest.document_multiagent_batch import verify_current_delivery
    from bridge_evidence.contest.document_workflow import DocumentWorkflow
    from bridge_evidence.contest.document_delivery_artifacts import DocumentDeliveryArtifacts
    from bridge_evidence.contest.run_state import atomic_json, json_digest
    root = tmp_path / "run"
    result = _word_result({"document_id": "synthetic", "document_key": "a" * 64,
        "manifest_identity_sha256": "b" * 64}, root)
    workspace_path = root / "merge/workspace/workspace.json"
    state = json.loads(workspace_path.read_text())
    if missing != "workspace":
        state["identity"]["quality_policy"] = "speed_v1_explicit_gaps"
    atomic_json(workspace_path, state)
    old = result["delivery"]
    manager = DocumentDeliveryArtifacts(root / "merge/delivery_artifacts", old["document_id"], scope=old["scope"])
    receipt = manager.export(state, incomplete=False)
    statement = deepcopy(old["agent_file_self_check"]["statement"])
    statement.update(semantic_state_sha256=DocumentWorkflow._semantic_identity(state),
        verified_file_binding={"workspace_version": state["version"], "draft_sha256": state["draft_sha256"],
            "file_sha256": receipt["file_sha256"]})
    statement["model_statement"]["remaining_issues"] = ["历史变化未核实"]
    if missing != "statement":
        statement.update(quality_policy="speed_v1_explicit_gaps", exhaustive_quality_verified=False)
    receipt.update(complete=True, incomplete=False, phase="completed", actual_finish_origin=old["actual_finish_origin"],
        agent_file_self_check={"performed": True, "statement": statement})
    trace = {"identity": {} if missing == "workflow" else {"settings": {"allow_quality_gaps": True},
        "run_identity": {"speed_priority": True}}, "actual_finish_origin": old["actual_finish_origin"],
        "artifact_delivery": receipt, "model_self_check": statement,
        "accepted_finish": {"file_sha256": receipt["file_sha256"], "model_self_check_sha256": json_digest(statement)}}
    atomic_json(root / "merge/document_workflow.json", trace)
    atomic_json(Path(receipt["status_path"]), receipt)
    result["delivery"] = receipt
    if missing:
        with pytest.raises(ValueError, match="No current native finish"):
            verify_current_delivery(root, result)
    else:
        assert verify_current_delivery(root, result) == receipt
        Path(receipt["docx_path"]).write_bytes(b"tampered")
        with pytest.raises(ValueError):
            verify_current_delivery(root, result)
