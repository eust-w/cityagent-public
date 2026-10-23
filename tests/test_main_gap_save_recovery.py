"""Save recovery must preserve committed rows without per-call save exhaustion.

All source and model decisions in this module are synthetic CPU-only fixtures.
The integration sidecar separately verifies actual model correction and Word
delivery; these tests do not certify semantic coverage or invent that evidence.
"""
from copy import deepcopy
import time
from types import SimpleNamespace

import pytest

from bridge_evidence.contest import document_candidate_review as engine
from bridge_evidence.contest.document_speed_workflow import SpeedMergeWorkflow
from bridge_evidence.contest.document_workspace import DocumentWorkspace
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import json_digest


@pytest.fixture
def recovery(tmp_path):
    manifest = manifest_from_text(
        "甲处锈蚀\n乙处裂缝\n丙处渗水\n"
        "建议甲处除锈\n建议乙处封闭裂缝\n建议丙处疏通排水\n建议丁处修补表面\n")
    workspace = DocumentWorkspace(manifest, tmp_path / "main", {"model": "synthetic"})
    for unit in manifest["units"]:
        workspace.record_read(unit["id"], 0, len(unit["text"]))
    workspace.apply_update({"base_version": workspace.snapshot()["version"],
        "draft": {"基本信息": {}, "病害明细": [], "维修建议": []},
        "draft_fact_ids": [], "draft_bindings": {}})
    workflow = object.__new__(SpeedMergeWorkflow)
    workflow.workspace, workflow.manifest = workspace, manifest
    workflow.directory, workflow.handoff_evidence = workspace.directory, {}
    workflow.child_summary = [{"task_id": "synthetic-failed-child", "status": "failed"}]
    workflow.source_delivery = SimpleNamespace(confirmed_ranges=lambda: workspace.snapshot()["reads"])
    workflow._prepare_review_word = lambda payload, result: result
    return workflow


def advice(workflow, index):
    unit = workflow.manifest["units"][index + 3]
    return {"建议内容": unit["text"], "source_refs": [{"evidence_id": unit["id"]}],
        "status": "explicit"}


def native(monkeypatch, payload, call="save-1"):
    invocation = {"tool_name": "save_recommendations", "run_id": "synthetic-run",
        "session_id": "synthetic-session", "generation_call_id": "synthetic-generation",
        "native_tool_call_id": call, "args_sha256": json_digest(payload)}
    monkeypatch.setattr(engine, "_current_invocation", lambda: invocation)
    return invocation


def test_third_and_fourth_single_entry_saves_do_not_exhaust_two_call_allowance(recovery):
    handler = recovery._additional_reading_handlers()["save_recommendations"]
    before = recovery.workspace.snapshot()
    receipts = [handler(entries=[advice(recovery, index)]) for index in range(4)]
    assert all(receipt["ok"] and receipt["added_count"] == 1 for receipt in receipts)
    for index, receipt in enumerate(receipts, 1):
        progress = receipt["save_progress"]
        assert progress["outcome"] == "saved" and progress["saved_indices"] == [0]
        assert progress["already_saved_indices"] == []
        assert progress["attempt"] == index
        assert progress["workspace_version"] == before["version"] + index
        assert progress["operation_workspace_version"] == receipt["version"]
        assert progress["batch_applied"] is True
        assert progress["save_calls_limited_by"] == "shared_main_model_tool_time_budget"
    after = recovery.workspace.snapshot()
    assert after["draft"]["维修建议"] == [
        {"建议内容": advice(recovery, index)["建议内容"]} for index in range(4)]
    assert after["version"] == before["version"] + 4
    assert all(identifier in after["facts"] for identifiers in after["draft_bindings"].values()
        for identifier in identifiers)


def test_explicit_failures_do_not_block_later_corrected_saves(recovery):
    handler = recovery._additional_reading_handlers()["save_recommendations"]
    initial = recovery.workspace.checkpoint_path.read_bytes()
    for _ in range(3):
        rejected = handler(entries=[])
        assert not rejected["ok"] and rejected["error"]["details"]["applied"] is False
        assert rejected["save_progress"]["outcome"] == "not_saved"
        assert rejected["save_progress"]["saved_indices"] == []
        assert rejected["save_progress"]["error"]["path"] == "entries"
        assert recovery.workspace.checkpoint_path.read_bytes() == initial
    for index in range(3):
        assert handler(entries=[advice(recovery, index)])["ok"]
    assert len(recovery.workspace.snapshot()["draft"]["维修建议"]) == 3


def test_bad_row_rejects_whole_batch_then_field_only_correction_retains_prior_rows(recovery):
    handler = recovery._additional_reading_handlers()["save_recommendations"]
    assert handler(entries=[advice(recovery, 0)])["ok"]
    before = recovery.workspace.checkpoint_path.read_bytes()
    payload = {"entries": [advice(recovery, 1), advice(recovery, 2)]}
    payload["entries"][1]["建议内容"] = " "
    rejected = handler(**payload)
    assert not rejected["ok"]
    assert rejected["error"]["details"]["applied"] is False
    assert rejected["error"]["details"]["path"] == "entries[1].建议内容"
    assert rejected["save_progress"]["outcome"] == "not_saved"
    assert rejected["save_progress"]["batch_applied"] is False
    assert rejected["save_progress"]["error"]["path"] == "entries[1].建议内容"
    assert rejected["save_progress"]["saved_indices"] == []
    assert recovery.workspace.checkpoint_path.read_bytes() == before
    payload["entries"][1]["建议内容"] = advice(recovery, 2)["建议内容"]
    corrected = handler(**payload)
    assert corrected["ok"] and corrected["added_count"] == 2
    assert corrected["save_progress"]["saved_indices"] == [0, 1]
    assert len(recovery.workspace.snapshot()["draft"]["维修建议"]) == 3


def test_duplicate_noop_preserves_contents_then_still_allows_new_rows(recovery):
    handler = recovery._additional_reading_handlers()["save_recommendations"]
    assert handler(entries=[advice(recovery, 0)])["ok"]
    before = recovery.workspace.checkpoint_path.read_bytes()
    for _ in range(3):
        repeated = handler(entries=[advice(recovery, 0)])
        assert repeated["ok"] and not repeated["changed"] and repeated["added_count"] == 0
        assert repeated["save_progress"]["outcome"] == "unchanged"
        assert repeated["save_progress"]["saved_indices"] == []
        assert repeated["save_progress"]["already_saved_indices"] == [0]
        assert recovery.workspace.checkpoint_path.read_bytes() == before
    for index in (1, 2):
        assert handler(entries=[advice(recovery, index)])["ok"]
    assert len(recovery.workspace.snapshot()["draft"]["维修建议"]) == 3


def test_old_exhausted_gap_state_can_resume_without_reopening_closed_state(recovery):
    record = recovery._gap_state()
    record["save_attempts"]["save_recommendations"] = 2
    record["limits"]["save_attempts_per_section"] = 2
    recovery._persist_gap_state()
    del recovery.main_gap_repair
    assert recovery._additional_reading_handlers()["save_recommendations"](
        entries=[advice(recovery, 0)])["ok"]
    record = recovery._gap_state()
    assert "save_attempts_per_section" not in record["limits"]
    assert record["limits"]["save_calls"] == "shared_main_model_tool_time_budget"
    assert record["save_attempts"]["save_recommendations"] == 3
    record["closed"] = True
    record["model_checks"] = {"病害明细": "synthetic-check", "维修建议": "synthetic-check"}
    recovery._persist_gap_state()
    del recovery.main_gap_repair
    before = recovery.workspace.checkpoint_path.read_bytes()
    assert not recovery._additional_reading_handlers()["save_recommendations"](
        entries=[advice(recovery, 1)])["ok"]
    assert recovery._gap_state()["closed"] is True
    assert recovery.workspace.checkpoint_path.read_bytes() == before


def test_save_tools_remain_enabled_after_old_save_counter_exceeds_two(recovery):
    recovery._gap_state()["save_attempts"].update(save_defects=3, save_recommendations=9)
    recovery._finish_is_current = lambda: False
    recovery._speed_progress = lambda: {"stage": "review_edits", "budget": {"wrap_up": True}}
    names = ["save_defects", "save_recommendations", "complete_review", "finish_report"]
    payload = {"messages": [], "tools": [{"type": "function", "function": {"name": name}}
        for name in names]}
    result = recovery._before_continuous_request(payload)
    assert {tool["function"]["name"] for tool in result["tools"]} == set(names) - {"finish_report"}
    assert payload["tools"] == [{"type": "function", "function": {"name": name}} for name in names]


def test_actual_main_source_delivery_failure_can_be_fixed_without_losing_saved_rows(recovery):
    handler = recovery._additional_reading_handlers()["save_recommendations"]
    assert handler(entries=[advice(recovery, 0)])["ok"]
    before = recovery.workspace.checkpoint_path.read_bytes()
    confirmed = recovery.source_delivery.confirmed_ranges
    recovery.source_delivery.confirmed_ranges = lambda: {}
    for index in (1, 2):
        result = handler(entries=[advice(recovery, index)])
        assert result["error"]["code"] == "MAIN_SOURCE_NOT_DELIVERED"
        assert result["error"]["details"]["applied"] is False
        assert recovery.workspace.checkpoint_path.read_bytes() == before
    recovery.source_delivery.confirmed_ranges = confirmed
    for index in (1, 2):
        assert handler(entries=[advice(recovery, index)])["ok"]
    assert len(recovery.workspace.snapshot()["draft"]["维修建议"]) == 3


def test_exact_native_replay_recovers_postcommit_failure_without_duplicate_save(recovery, monkeypatch):
    handler = recovery._additional_reading_handlers()["save_recommendations"]
    payload = {"entries": [advice(recovery, 0)]}
    invocation = native(monkeypatch, payload)
    real_atomic = engine.atomic_json

    def fail_final_journal(path, value):
        if path.name == "candidate-review.json" and any(
                row.get("phase") == "complete" for row in value.get("calls", {}).values()):
            raise OSError("synthetic post-commit journal failure")
        return real_atomic(path, value)

    monkeypatch.setattr(engine, "atomic_json", fail_final_journal)
    first = handler(**payload)
    assert first["error"]["details"]["applied"] is None
    assert first["save_progress"]["outcome"] == "commit_unconfirmed"
    assert first["save_progress"]["batch_applied"] is None
    assert first["save_progress"]["saved_indices"] == []
    committed = recovery.workspace.checkpoint_path.read_bytes()
    attempts = recovery._gap_state()["save_attempts"]["save_recommendations"]
    assert len(recovery.workspace.snapshot()["draft"]["维修建议"]) == 1
    monkeypatch.setattr(engine, "atomic_json", real_atomic)
    again = handler(**payload)
    assert again["ok"] and again["replayed"]
    assert again["save_progress"]["replayed"]
    assert again["save_progress"]["attempt"] == attempts
    assert recovery._gap_state()["save_attempts"]["save_recommendations"] == attempts
    assert recovery.workspace.checkpoint_path.read_bytes() == committed
    monkeypatch.setattr(engine, "_current_invocation", lambda: None)
    for index in (1, 2):
        assert handler(entries=[advice(recovery, index)])["ok"]
    before_replay = recovery.workspace.checkpoint_path.read_bytes()
    version_after_more_saves = recovery.workspace.snapshot()["version"]
    attempts = recovery._gap_state()["save_attempts"]["save_recommendations"]
    monkeypatch.setattr(engine, "_current_invocation", lambda: invocation)
    late_replay = handler(**payload)
    assert late_replay["replayed"]
    assert late_replay["save_progress"]["workspace_version"] == version_after_more_saves
    assert late_replay["save_progress"]["operation_workspace_version"] == again["version"]
    assert late_replay["save_progress"]["operation_workspace_version"] < version_after_more_saves
    assert recovery._gap_state()["save_attempts"]["save_recommendations"] == attempts
    assert recovery.workspace.checkpoint_path.read_bytes() == before_replay


def test_closed_gap_allows_only_exact_proven_receipt_replay(recovery, monkeypatch):
    handler = recovery._additional_reading_handlers()["save_recommendations"]
    payload = {"entries": [advice(recovery, 0)]}
    native(monkeypatch, payload)
    first = handler(**payload)
    assert first["ok"]
    recovery._gap_state()["closed"] = True
    recovery._persist_gap_state()
    before = recovery.workspace.checkpoint_path.read_bytes()
    again = handler(**payload)
    assert again["ok"] and again["replayed"]
    assert recovery.workspace.checkpoint_path.read_bytes() == before
    monkeypatch.setattr(engine, "_current_invocation", lambda: None)
    assert not handler(entries=[advice(recovery, 1)])["ok"]
    assert recovery.workspace.checkpoint_path.read_bytes() == before


def test_batch_size_limit_remains_atomic_and_does_not_disable_later_save(recovery):
    handler = recovery._additional_reading_handlers()["save_recommendations"]
    before = recovery.workspace.checkpoint_path.read_bytes()
    result = handler(entries=[deepcopy(advice(recovery, 0)) for _ in range(6)])
    assert not result["ok"] and result["error"]["details"]["applied"] is False
    assert result["error"]["details"]["path"] == "entries"
    assert recovery.workspace.checkpoint_path.read_bytes() == before
    for index in range(3):
        assert handler(entries=[advice(recovery, index)])["ok"]
    assert len(recovery.workspace.snapshot()["draft"]["维修建议"]) == 3


def test_continuing_small_saves_does_not_reset_main_model_or_time_budget(recovery):
    recovery.settings = SimpleNamespace(single_document_agent=True,
        dynamic_output_budget=False, max_model_calls=16)
    recovery.native_compression = None
    recovery.model_dispatches = []
    recovery.client = SimpleNamespace(timeout=900)
    recovery.deadline = time.monotonic() + 900
    deadline = recovery.deadline
    calls = [{"kind": "generation", "seconds": 1} for _ in range(15)]
    recovery._model_calls = lambda: deepcopy(calls)
    recovery._check_budget()
    handler = recovery._additional_reading_handlers()["save_recommendations"]
    for index in range(4):
        assert handler(entries=[advice(recovery, index)])["ok"]
    assert recovery.settings.max_model_calls == 16
    assert recovery.main_request_output_cap == 1024
    assert recovery.client.timeout == 900 and recovery.deadline == deadline
    assert len(recovery._model_calls()) == 15
    calls.append({"kind": "generation", "seconds": 1})
    with pytest.raises(RuntimeError, match="cumulative model request budget exhausted"):
        recovery._check_budget()
    recovery.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError, match="cumulative time budget exhausted"):
        recovery._check_budget()
