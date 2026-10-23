"""Direct handoff wiring: actual stores/source receipts, no inference service."""
from copy import deepcopy
import json

import pytest
from docx import Document

from bridge_evidence.contest.document_entries import EntryStore
from bridge_evidence.contest.document_merge_progress import build_handoff_evidence, inspection_progress
from bridge_evidence.contest.document_multiagent import MergeWorkflow, MultiAgentSettings, import_candidates, coverage_question_id
from bridge_evidence.contest.document_multiagent_cli import parser
from bridge_evidence.contest.document_source_delivery import SourceDeliveryLedger
from bridge_evidence.contest.document_workspace import DocumentWorkspace
from bridge_evidence.contest.docx_writer import write_extraction_docx
from bridge_evidence.contest.evidence_manifest import manifest_from_text


def fixture(tmp_path):
    manifest = manifest_from_text("合成桥，左幅支座锈蚀，建议除锈。\n未交接的其他原文不得自动认定主Agent已读。")
    source = manifest["units"][0]
    store = EntryStore(manifest, "task-one", tmp_path / "child")
    store.record_read(source["id"], 0, len(source["text"]))
    entries = [
        {"id": "bridge", "section": "基本信息", "field": "桥梁名称", "value": "合成桥"},
        {"id": "defect", "section": "病害明细", "row": {"病害部位": "左幅支座", "病害描述": "锈蚀"}},
        {"id": "advice", "section": "维修建议", "row": {"维修建议": "除锈"}},
    ]
    # Use the delivery schema's actual advice column, not a display alias.
    from bridge_evidence.contest.docx_writer import ADVICE_FIELDS
    entries[2]["row"] = {ADVICE_FIELDS[-1]: "除锈"}
    store.save_entries([{**entry, "source_refs": [{"evidence_id": source["id"]}],
                         "status": "explicit"} for entry in entries])
    task = {"id": "task-one", "primary_evidence_ids": [source["id"]], "headings": []}
    child = {"task_id": task["id"], "status": "completed", "entries": store.snapshot(),
             "handoff": {"unresolved": []}}
    merge = object.__new__(MergeWorkflow)
    merge.direct_delivery = True
    merge.manifest = manifest
    merge.directory = tmp_path / "merge"
    merge.workspace = DocumentWorkspace(manifest, merge.directory, {"model": "synthetic"})
    merge.child_summary = [{"task_id": task["id"], "status": child["status"], "handoff": child["handoff"],
        "entry_count": 3, "primary_evidence_ids": task["primary_evidence_ids"], "headings": []}]
    merge.handoff_evidence = build_handoff_evidence(manifest, [child])
    import_candidates(merge.workspace, [child], tasks=[task])
    merge.source_delivery = SourceDeliveryLedger(manifest, tmp_path / "delivered", run_id="test", agent_id="main")
    return merge, task


def test_initial_handoff_has_all_candidates_and_only_actually_received_sources(tmp_path):
    merge, _ = fixture(tmp_path)
    mode, prompt = merge._entry()
    assert mode == "navigation"
    assert len(merge.handoff_bundle["candidates"]) == 3
    assert not merge.handoff_bundle["remaining"]["candidate_ids"]
    assert "未交接的其他原文" not in prompt
    assert merge.source_delivery.confirmed_ranges() == {}
    merge._register_initial_source(prompt, mode, 0, [])
    merge.source_delivery.observe_request({"messages": [{"role": "user", "content": prompt}]}, {"request_id": "request-1"})
    merge.source_delivery.confirm_request("request-1", evidence={"input_delivery_confirmed": True})
    assert set(merge.source_delivery.confirmed_ranges()) == {"E000001"}
    assert merge.source_delivery.pending_initial_messages() == []


def test_unreceived_handoff_recovery_never_registers_the_whole_report(tmp_path):
    merge, _ = fixture(tmp_path)
    mode, prompt = merge._entry()
    merge._register_initial_source(prompt, mode, 0, [])
    pending = merge.source_delivery.pending_initial_messages()
    recovered = prompt + "\n恢复当前已保存工作区。"
    merge._register_initial_source(recovered, mode, 1, pending)
    merge.source_delivery.observe_request({"messages": [{"role": "user", "content": recovered}]}, {"request_id": "request-2"})
    merge.source_delivery.confirm_request("request-2", evidence={"input_delivery_confirmed": True})
    assert set(merge.source_delivery.confirmed_ranges()) == {"E000001"}
    assert merge.source_delivery.pending_initial_messages() == []


def test_candidate_patch_reuses_original_fact_and_requires_new_word_inspection(tmp_path):
    merge, task = fixture(tmp_path)
    handlers = merge._additional_reading_handlers()
    assert set(handlers) == {"submit_review"}
    handler = handlers["submit_review"]
    before = merge.workspace.snapshot()
    fact_id = next(key for key, value in merge.handoff_evidence.items() if value["section"] == "病害明细")
    old_artifact = {"workspace_version": before["version"], "draft_sha256": before["draft_sha256"],
                    "file_sha256": "old-word", "structure_valid": True, "content_validated": True}
    old_reads = [{**old_artifact, "view": view, "cursor": 0, "end": 1, "total_chars": 1}
                 for view in ("basic", "advice", "defects")]
    receipt = handler(patches=[{"candidate_id": fact_id,
        "changes": {"病害描述": "支座锈蚀"}}], reviews=[{"question_id": coverage_question_id(task["id"]), "status": "resolved",
        "reason": "合成模型核对：基本信息、病害和建议分别有对应原文，三类已保存。"}])
    assert receipt["ok"], receipt
    after = merge.workspace.snapshot()
    assert after["facts"] == before["facts"]
    assert after["draft_bindings"] == before["draft_bindings"]
    assert not merge._coverage_pending(after)
    assert len(inspection_progress(after, old_artifact, old_reads)["pending_reads"]) == 3
    target = tmp_path / "report.docx"
    write_extraction_docx(after["draft"], str(target))
    exported = "\n".join(cell.text for table in Document(target).tables for row in table.rows for cell in row.cells)
    assert "支座锈蚀" in exported and "除锈" in exported and "合成桥" in exported


def test_wrong_candidate_cannot_partially_patch_or_resolve_coverage(tmp_path):
    merge, task = fixture(tmp_path)
    handler = merge._additional_reading_handlers()["submit_review"]
    before = deepcopy(merge.workspace.snapshot())
    receipt = handler(selections=["foreign-candidate"],
        reviews=[{"question_id": coverage_question_id(task["id"]), "status": "resolved", "reason": "不能在错误批次中生效"}])
    assert not receipt["ok"]
    assert merge.workspace.snapshot() == before
    assert merge._coverage_pending(before)


def test_direct_flag_is_opt_in_and_cli_is_explicit():
    assert not MultiAgentSettings().direct_delivery
    assert MultiAgentSettings(direct_delivery=True).direct_delivery
    with pytest.raises(ValueError, match="boolean"):
        MultiAgentSettings(direct_delivery="true")
    args = parser().parse_args(["--manifest", "source.json", "--manifest-sha256", "0" * 64,
        "--scope", "official-training", "--output-dir", "out", "--direct-delivery"])
    assert args.direct_delivery


def test_direct_continuation_names_review_tool_without_filling_a_verdict(tmp_path):
    merge, task = fixture(tmp_path)
    question_id = coverage_question_id(task["id"])
    before = merge.workspace.snapshot()
    progress = {"stage": "resolve_candidates", "target_question_id": question_id,
        "target_question": before["questions"][question_id],
        "next_action": {"tool": "read_workspace_page"}}
    action = merge._review_action(progress)
    assert action["next_action"] == {"tool": "submit_review", "required_question_id": question_id,
        "model_must_supply_status_and_reason": True}
    assert progress["next_action"]["tool"] == "read_workspace_page"
    assert merge.workspace.snapshot() == before
    for unchanged in ({"stage": "inspect_current_file", "next_action": {"tool": "inspect_delivery"}},
                      {"stage": "resolve_candidates", "target_question_id": "missing",
                       "next_action": {"tool": "update_workspace"}}):
        assert merge._review_action(unchanged) == unchanged
    merge.direct_delivery = False
    assert merge._review_action(progress) is progress
