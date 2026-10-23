"""Business completion stays separate from SDK final and later backend idleness."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from bridge_evidence.contest import document_multiagent as pipeline
from bridge_evidence.contest.document_entries import EntryStore
from bridge_evidence.contest.document_workspace import DocumentWorkspace
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import atomic_json, json_digest


def fixture(tmp_path):
    manifest = manifest_from_text("合成桥。支座锈蚀，建议除锈。")
    unit = manifest["units"][0]
    task = {"id": "section-a", "primary_evidence_ids": [unit["id"]],
            "headings": [{"title": "检测结果与养护建议"}]}
    store = EntryStore(manifest, task["id"], tmp_path / "child-entries")
    store.record_read(unit["id"], 0, len(unit["text"]))
    store.save_entries([{"id": "bridge", "section": "基本信息", "field": "桥梁名称", "value": "合成桥",
        "source_refs": [{"evidence_id": unit["id"]}], "status": "explicit"}])
    child = {"task_id": task["id"], "manifest_sha256": json_digest(manifest),
             "entries": store.snapshot(), "handoff": None, "status": "failed",
             "model_calls": 1, "generation_attempts_settled": False, "unknown_generation_calls": ["call-1"]}
    workspace = DocumentWorkspace(manifest, tmp_path / "workspace", {"model": "synthetic"})
    return manifest, task, child, workspace


def bare_merger(workspace, task):
    merge = object.__new__(pipeline.MergeWorkflow)
    merge.workspace = workspace
    merge.child_summary = [{"task_id": task["id"]}]
    merge._check_budget = lambda: None
    return merge


def test_import_creates_model_coverage_task_even_after_read_and_basic_saved(tmp_path):
    _, task, child, workspace = fixture(tmp_path)
    pipeline.import_candidates(workspace, [child], tasks=[task])
    state = workspace.snapshot()
    question = state["questions"][pipeline.coverage_question_id(task["id"])]
    assert question["status"] == "open"
    assert all(section in question["question"] for section in ("基本信息", "病害明细", "维修建议"))
    assert state["draft"]["病害明细"] == []
    assert state["draft"]["维修建议"] == []
    assert any("unresolved_question:" in blocker for blocker in workspace.delivery_status()["blockers"])


@pytest.mark.parametrize("state_change", ["missing", "unknown", "invalid"])
def test_coverage_cannot_be_bypassed_by_unknown_retraction_or_invalidation(tmp_path, state_change):
    _, task, child, workspace = fixture(tmp_path)
    pipeline.import_candidates(workspace, [child], tasks=[task])
    state = workspace.snapshot()
    key = pipeline.coverage_question_id(task["id"])
    state["questions"][key].update(status="resolved", reason="模型核对后说明三类覆盖。")
    if state_change == "missing":
        del state["questions"][key]
    elif state_change == "unknown":
        state["questions"][key]["status"] = "unknown"
    else:
        state["questions"][key]["valid"] = False
    merge = bare_merger(SimpleNamespace(snapshot=lambda: state, delivery_status=lambda: {'ready': False, 'blockers': []}), task)
    assert merge._coverage_pending(state) == [key]
    assert merge._finish()["ok"] is False


def test_empty_rows_do_not_force_invention_when_model_reports_no_relevant_items(tmp_path):
    _, task, child, workspace = fixture(tmp_path)
    pipeline.import_candidates(workspace, [child], tasks=[task])
    state = workspace.snapshot()
    state["questions"][pipeline.coverage_question_id(task["id"])].update(
        status="resolved", reason="合成测试：模型说明本范围三类别均无相关条目。")
    merge = bare_merger(workspace, task)
    assert merge._coverage_pending(state) == []  # Only model decision recorded, no host semantic approval.


def test_early_native_final_returns_actual_saved_counts_and_same_session_repair(tmp_path):
    _, task, child, workspace = fixture(tmp_path)
    pipeline.import_candidates(workspace, [child], tasks=[task])
    merge = bare_merger(workspace, task)
    merge._finish_is_current = lambda: False
    feedback = merge._continue_after_native_exit({"native_final_event": True, "response_text": "已总结。"})
    assert "同一会话" in feedback and '"病害明细": 0' in feedback and '"维修建议": 0' in feedback
    progress = json.loads(feedback.split('\n', 1)[1])
    assert progress['next_action']['tool'] == 'read_workspace_page'
    assert progress['stage'] == 'resolve_candidates'
    assert progress['semantic_approval'] is False


def test_valid_file_finish_stops_but_empty_native_final_needs_final_ack(tmp_path):
    _, task, _, workspace = fixture(tmp_path)
    merge = bare_merger(workspace, task)
    merge._finish_is_current = lambda: True
    assert merge._continue_after_native_exit({"native_final_event": True, "response_text": "已交付"}) is None
    assert "非空" in merge._continue_after_native_exit({"native_final_event": False, "response_text": ""})


def test_later_quiescence_allows_merge_without_rewriting_failed_child_or_unknown_usage(tmp_path, monkeypatch):
    manifest, _, child, _ = fixture(tmp_path)
    def dispatch(tasks, **kwargs):
        row = deepcopy(child)
        row["task_id"] = tasks[0]["id"]
        row["entries"]["identity"]["task_id"] = row["task_id"]
        path = tmp_path / "immutable-child-result.json"
        atomic_json(path, row)
        return {"status": "unknown", "results": [{"task_id": row["task_id"], "result_path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "settled": False}]}
    observed = []
    def observe(dispatch_result, **kwargs):
        observed.append(kwargs)
        return {"outcome": "quiescent_observed", "may_continue_aggregation": True,
                "historical_usage_known": False}
    monkeypatch.setattr(pipeline, "make_client", lambda *a, **k: SimpleNamespace(
        base_url="http://unused", model="synthetic", api_key="not-a-secret"))
    class Merger:
        artifact_delivery = None
        def __init__(self, source, client, config, directory, **kwargs):
            assert kwargs["child_summary"][0]["status"] == "failed"
            self.workspace = DocumentWorkspace(source, directory, {"model": "synthetic"})
        def run(self):
            assert any(key.startswith("Q_coverage_") for key in self.workspace.snapshot()["questions"])
            raise RuntimeError("Synthetic main Agent has started but has not finished its Word")
        def _model_calls(self):
            return [{"kind": "agent_generation"}]
    result = pipeline.run_multiagent(manifest, tmp_path / "run", {"model": "synthetic"}, dispatcher=dispatch,
        workflow_type=Merger, settlement_observer=observe)
    assert observed
    assert result["parent_model_calls"] == 1
    assert result["child_call_accounting_complete"] is False
    assert result["child_model_calls"] is None
    assert result["child_results"][0]["status"] == "failed"
    assert result["status"] == "failed" and result["delivery"] is None
    assert json.loads((tmp_path / "immutable-child-result.json").read_text())["generation_attempts_settled"] is False
