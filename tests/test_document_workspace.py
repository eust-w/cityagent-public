from copy import deepcopy
import json

import pytest

from bridge_evidence.contest.document_workspace import (
    DocumentWorkspace,
    WorkspaceConflictError,
    WorkspaceError,
    WorkspaceIdentityError,
)
from bridge_evidence.contest.evidence_manifest import manifest_from_text


def make_workspace(tmp_path, text="桥名：合成桥\n左幅1号梁有裂缝\n", **kwargs):
    manifest = manifest_from_text(text)
    manifest.update(kwargs)
    return DocumentWorkspace(manifest, tmp_path, {"model": "synthetic", "prompt": "v1"})


def read_all(workspace):
    for unit in workspace.manifest["units"]:
        workspace.record_read(unit["id"], 0, len(unit["text"]))


def fact(workspace, record_id="bridge", index=0, **kwargs):
    unit = workspace.manifest["units"][index]
    return {"id": record_id, "value": "合成桥", "source_refs": [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}],
            "status": "explicit", "depends_on": [], **kwargs}


def update(workspace, **kwargs):
    return workspace.apply_update({"base_version": workspace.snapshot()["version"], **kwargs})


def set_draft(workspace, ids=None, bindings=None, **kwargs):
    return update(workspace, draft={"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [], "病害明细": []},
                  draft_fact_ids=["bridge"] if ids is None else ids,
                  draft_bindings={"基本信息.桥梁名称": ["bridge"]} if bindings is None else bindings, **kwargs)


@pytest.mark.parametrize("patch", [
    {"op": "set_basic", "field": "全桥技术状况评分", "value": "88.8", "fact_ids": ["bridge"]},
    {"op": "set_basic", "field": "总体评分", "value": {"score": "88.8"}, "fact_ids": ["bridge"]},
    {"op": "append_row", "section": "病害明细", "row": {"病害类型": True}, "fact_ids": ["bridge"]},
    {"op": "append_row", "section": "维修建议", "row": {"额外建议": "补写"}, "fact_ids": ["bridge"]},
])
def test_unexportable_patch_is_rejected_before_any_workspace_change(tmp_path, patch):
    workspace = make_workspace(tmp_path)
    read_all(workspace)
    update(workspace, facts=[fact(workspace)])
    set_draft(workspace)
    before = workspace.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceError, match="unsupported fields|exportable scalar"):
        update(workspace, draft_patch=[patch])
    assert workspace.checkpoint_path.read_bytes() == before


def test_full_draft_uses_same_exportable_field_contract_as_incremental_updates(tmp_path):
    workspace = make_workspace(tmp_path)
    read_all(workspace)
    update(workspace, facts=[fact(workspace)])
    before = workspace.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceError, match="unsupported fields"):
        update(workspace,
               draft={"基本信息": {"桥梁名称": "合成桥", "全桥技术状况评分": "88.8"}, "维修建议": [], "病害明细": []},
               draft_fact_ids=["bridge"],
               draft_bindings={"基本信息.桥梁名称": ["bridge"], "基本信息.全桥技术状况评分": ["bridge"]})
    assert workspace.checkpoint_path.read_bytes() == before


def review(workspace, **kwargs):
    state = workspace.snapshot()
    return {"workspace_version": state["version"], "draft_sha256": state["draft_sha256"],
            "checked_evidence_ids": state["reading_progress"]["fully_read_evidence_ids"],
            "unsupported_claims": [], "missing_items": [], "conflicts": [], "unresolved_items": [],
            "passed": True, "model_call_ids": ["synthetic-review-call"], **kwargs}


def ready_workspace(tmp_path):
    workspace = make_workspace(tmp_path)
    read_all(workspace)
    set_draft(workspace, facts=[fact(workspace)])
    workspace.record_review(review(workspace))
    assert workspace.delivery_status()["ready"]
    return workspace


def test_checkpoint_resumes_bound_to_manifest_and_complete_run_identity(tmp_path):
    workspace = ready_workspace(tmp_path)
    before = workspace.snapshot()
    restored = DocumentWorkspace(workspace.manifest, tmp_path, {"model": "synthetic", "prompt": "v1"})
    assert restored.snapshot() == before
    assert restored.delivery_status()["ready"]
    with pytest.raises(WorkspaceIdentityError):
        DocumentWorkspace(workspace.manifest, tmp_path, {"model": "different", "prompt": "v1"})
    changed = deepcopy(workspace.manifest)
    changed["units"][0]["text"] += "改动"
    with pytest.raises(WorkspaceIdentityError):
        DocumentWorkspace(changed, tmp_path, {"model": "synthetic", "prompt": "v1"})
    before["facts"]["bridge"]["value"] = "不能通过快照篡改"
    assert restored.snapshot()["facts"]["bridge"]["value"] == "合成桥"


def test_independent_instances_cannot_overwrite_a_newer_checkpoint(tmp_path):
    workspace = make_workspace(tmp_path)
    other = make_workspace(tmp_path)
    stale = other.snapshot()["version"]
    workspace.record_read("E000001", 0, 2)
    with pytest.raises(WorkspaceConflictError):
        other.apply_update({"base_version": stale, "questions": [{"id": "q"}]})
    assert other.snapshot()["reads"] == {"E000001": [[0, 2]]}
    assert other.snapshot()["questions"] == {}


def test_reads_merge_but_gaps_and_exclusions_do_not_authorize_source_refs(tmp_path):
    workspace = make_workspace(tmp_path, "0123456789")
    workspace.record_read("E000001", 0, 3)
    workspace.record_read("E000001", 5, 8)
    update(workspace, exclusions=[{"evidence_id": "E000001", "start": 3, "end": 5, "reason": "版式占位"}])
    with pytest.raises(WorkspaceError, match="not actually read"):
        update(workspace, facts=[fact(workspace, source_refs=[{"evidence_id": "E000001", "start": 0, "end": 8}])])
    workspace.record_read("E000001", 3, 5)
    assert workspace.snapshot()["reads"] == {"E000001": [[0, 8]]}
    version = workspace.snapshot()["version"]
    workspace.record_read("E000001", 1, 5)
    assert workspace.snapshot()["version"] == version
    update(workspace, facts=[fact(workspace, source_refs=[{"evidence_id": "E000001", "start": 0, "end": 8}])])
    with pytest.raises(WorkspaceError, match="source_path"):
        update(workspace, facts=[fact(workspace, source_refs=[{"evidence_id": "E000001", "start": 0, "end": 8, "source_path": "other-report"}])])
    for start, end in ((-1, 2), (0, 11), (True, 2), (3, 2)):
        with pytest.raises(WorkspaceError):
            workspace.record_read("E000001", start, end)


def test_dependency_change_invalidates_transitive_records_draft_and_review(tmp_path):
    workspace = make_workspace(tmp_path)
    read_all(workspace)
    update(workspace, facts=[fact(workspace), fact(workspace, "damage", 1, depends_on=["bridge"])],
           relations=[{"id": "location", "subject_id": "damage", "object_id": "bridge", "depends_on": []}])
    set_draft(workspace, ids=["location"], bindings={"基本信息.桥梁名称": ["location"]})
    workspace.record_review(review(workspace))
    old_review = review(workspace)
    state = update(workspace, facts=[{"id": "bridge", "value": "更正后的桥名"}])
    assert not state["facts"]["damage"]["valid"]
    assert not state["relations"]["location"]["valid"]
    assert not state["draft_valid"]
    assert state["review"] is None
    assert not workspace.delivery_status()["ready"]
    with pytest.raises(WorkspaceConflictError):
        workspace.record_review(old_review)
    # Reaffirming only a dependent relation cannot hide a still-invalid parent.
    with pytest.raises(WorkspaceError, match="missing/invalid"):
        update(workspace, relations=[{"id": "location"}])
    update(workspace, facts=[{"id": "damage"}], relations=[{"id": "location"}])
    assert not workspace.snapshot()["draft_valid"]
    set_draft(workspace, ids=["location"], bindings={"基本信息.桥梁名称": ["location"]})
    assert not workspace.delivery_status()["ready"]
    workspace.record_review(review(workspace))
    assert workspace.delivery_status()["ready"]
    state = update(workspace, retract_fact_ids=["bridge"])
    assert "bridge" not in state["facts"]
    assert not state["relations"]["location"]["valid"]
    assert not workspace.delivery_status()["ready"]


def test_relation_endpoints_invalidate_even_without_explicit_dependency(tmp_path):
    workspace = make_workspace(tmp_path)
    read_all(workspace)
    update(workspace, facts=[fact(workspace), fact(workspace, "damage", 1)],
           relations=[{"id": "located", "subject_id": "damage", "object_id": "bridge", "source_refs": fact(workspace)["source_refs"]}])
    state = update(workspace, retract_fact_ids=["bridge"])
    assert not state["relations"]["located"]["valid"]


def test_reverse_review_requires_sources_and_cannot_claim_unread_or_excluded_text(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.record_read("E000001", 0, len(workspace.manifest["units"][0]["text"]))
    set_draft(workspace, facts=[fact(workspace)])
    with pytest.raises(WorkspaceError, match="unread_source_ranges"):
        workspace.record_review(review(workspace))
    with pytest.raises(WorkspaceError, match="not actually read"):
        workspace.record_review(review(workspace, checked_evidence_ids=["E000001", "E000002"]))
    update(workspace, exclusions=[{"evidence_id": "E000002", "start": 0, "end": len(workspace.manifest["units"][1]["text"]), "reason": "模型判断该段与当前输出任务无关"}])
    with pytest.raises(WorkspaceError, match="reverse_source_review_incomplete"):
        workspace.record_review(review(workspace, checked_evidence_ids=[]))
    workspace.record_review(review(workspace))
    assert workspace.delivery_status()["ready"]
    assert "E000002" not in workspace.snapshot()["reads"]


def test_review_records_exact_scope_when_a_unit_is_partly_excluded(tmp_path):
    workspace = make_workspace(tmp_path, "相关信息不相关页码")
    workspace.record_read("E000001", 0, 4)
    update(workspace, exclusions=[{"evidence_id": "E000001", "start": 4, "end": 9, "reason": "无关的页码装饰"}])
    set_draft(workspace, facts=[fact(workspace, source_refs=[{"evidence_id": "E000001", "start": 0, "end": 4}])])
    state = workspace.record_review(review(workspace, checked_evidence_ids=["E000001"]))
    assert state["review"]["checked_read_ranges"] == [{"evidence_id": "E000001", "start": 0, "end": 4}]
    assert workspace.delivery_status()["ready"]


def test_reverse_review_cannot_omit_fully_read_source_absent_from_draft(tmp_path):
    workspace = ready_workspace(tmp_path)
    with pytest.raises(WorkspaceError, match="reverse_source_review_incomplete"):
        workspace.record_review(review(workspace, checked_evidence_ids=["E000001"]))


def test_manifest_copy_cannot_rewrite_source_under_a_saved_receipt(tmp_path):
    workspace = make_workspace(tmp_path)
    original = workspace.manifest["units"][0]["text"]
    returned = workspace.manifest
    returned["units"][0]["text"] = "替换证据"
    assert workspace.manifest["units"][0]["text"] == original


@pytest.mark.parametrize("field", ["unsupported_claims", "missing_items", "conflicts", "unresolved_items"])
def test_review_with_findings_cannot_be_marked_passed(tmp_path, field):
    workspace = ready_workspace(tmp_path)
    with pytest.raises(WorkspaceError, match="unresolved_findings"):
        workspace.record_review(review(workspace, **{field: ["待核对事项"]}))
    workspace.record_review(review(workspace, passed=False, **{field: ["待核对事项"]}))
    assert "semantic_review_not_passed" in workspace.delivery_status()["blockers"]


def test_review_is_bound_to_hash_version_and_actual_model_receipts(tmp_path):
    workspace = ready_workspace(tmp_path)
    with pytest.raises(WorkspaceConflictError, match="hash"):
        workspace.record_review(review(workspace, draft_sha256="older-draft"))
    with pytest.raises(WorkspaceConflictError, match="facts version"):
        workspace.record_review(review(workspace, facts_version=-1))
    with pytest.raises(WorkspaceError, match="model call receipts"):
        workspace.record_review(review(workspace, model_call_ids=[]))
    receipt = review(workspace)
    receipt.pop("missing_items")
    with pytest.raises(WorkspaceError, match="missing_items"):
        workspace.record_review(receipt)


def test_unknown_values_are_preserved_and_semantics_are_not_string_matched(tmp_path):
    workspace = make_workspace(tmp_path)
    read_all(workspace)
    update(workspace, facts=[fact(workspace, value="synthetic bridge", status="inferred"),
                              {"id": "history", "value": None, "source_refs": [], "depends_on": [], "status": "uncertain"}],
           questions=[{"id": "old-inspection", "status": "unknown", "reason": "报告未提供上次检测资料"}])
    state = update(workspace, draft={"基本信息": {"桥梁名称": "synthetic bridge", "上一次定检结果": "未知"}, "维修建议": [], "病害明细": []},
                   draft_fact_ids=["bridge", "history"], draft_bindings={"基本信息.桥梁名称": ["bridge"], "基本信息.上一次定检结果": ["history"]})
    assert state["facts"]["history"]["value"] is None
    assert state["facts"]["bridge"]["value"] == "synthetic bridge"
    workspace.record_review(review(workspace))
    assert workspace.delivery_status()["ready"]
    update(workspace, questions=[{"id": "unanswered", "status": "open"}])
    with pytest.raises(WorkspaceError, match="unresolved_question"):
        workspace.record_review(review(workspace))
    with pytest.raises(WorkspaceError, match="reason"):
        update(workspace, questions=[{"id": "unanswered", "status": "resolved"}])


def test_draft_has_item_level_bindings_and_not_just_global_fact_ids(tmp_path):
    workspace = make_workspace(tmp_path)
    read_all(workspace)
    update(workspace, facts=[fact(workspace)])
    with pytest.raises(WorkspaceError, match="own draft binding"):
        set_draft(workspace, bindings={})
    with pytest.raises(WorkspaceError, match="included in draft_fact_ids"):
        set_draft(workspace, bindings={"基本信息.桥梁名称": ["unrelated"]})
    with pytest.raises(WorkspaceError, match="own draft binding"):
        update(workspace, draft={"基本信息": {}, "维修建议": [], "病害明细": [{"病害部位": "未知"}]}, draft_fact_ids=["bridge"], draft_bindings={})
    set_draft(workspace)


def test_failed_update_is_atomic_and_dependencies_cannot_cycle(tmp_path):
    workspace = ready_workspace(tmp_path)
    original = workspace.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceError, match="Cyclic"):
        update(workspace, facts=[{"id": "bridge", "depends_on": ["bridge"]}])
    assert workspace.checkpoint_path.read_bytes() == original
    with pytest.raises(WorkspaceError, match="Unknown evidence"):
        update(workspace, questions=[{"id": "new-question"}], facts=[fact(workspace, source_refs=[{"evidence_id": "another-report", "start": 0, "end": 1}])])
    assert workspace.checkpoint_path.read_bytes() == original
    assert workspace.delivery_status()["ready"]


def test_atomic_write_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    workspace = ready_workspace(tmp_path)
    original = workspace.checkpoint_path.read_bytes()

    def fail_replace(*_):
        raise OSError("simulated disk failure before rename")

    monkeypatch.setattr("bridge_evidence.contest.document_workspace.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        update(workspace, questions=[{"id": "new-question"}])
    assert workspace.checkpoint_path.read_bytes() == original
    assert not list(tmp_path.glob(".workspace-*.tmp"))
    assert json.loads(original)["review"]["passed"]


def test_image_inventory_and_exclusion_do_not_count_as_read_receipts(tmp_path):
    workspace = make_workspace(tmp_path, images=[{"id": "I00001", "source_path": "word/media/image1.png"}])
    read_all(workspace)
    image_fact = fact(workspace, source_refs=[{"image_id": "I00001"}])
    with pytest.raises(WorkspaceError, match="not actually read"):
        update(workspace, facts=[image_fact])
    update(workspace, exclusions=[{"image_id": "I00001", "reason": "装饰性图标，与报告结论无关"}])
    with pytest.raises(WorkspaceError, match="not actually read"):
        update(workspace, facts=[image_fact])
    workspace.record_read("I00001", 0, 1)
    set_draft(workspace, facts=[image_fact])
    with pytest.raises(WorkspaceError, match="reverse_source_review_incomplete"):
        workspace.record_review(review(workspace))
    receipt = review(workspace)
    receipt["checked_evidence_ids"].append("I00001")
    workspace.record_review(receipt)
    assert workspace.delivery_status()["ready"]
    assert "I00001" not in workspace.snapshot()["reads"]
    assert workspace.snapshot()["image_reads"] == ["I00001"]


def test_long_draft_is_built_with_small_independent_patch_updates(tmp_path):
    descriptions = [f"左幅第{i + 1}号梁：" + "合成裂缝检测记录" * 24 for i in range(64)]
    workspace = make_workspace(tmp_path, "桥名：合成桥\n" + "\n".join(descriptions))
    read_all(workspace)
    update(workspace, facts=[fact(workspace)], draft_patch=[
        {"op": "set_basic", "field": "桥梁名称", "value": "合成桥", "fact_ids": ["bridge"]}])
    for index, description in enumerate(descriptions):
        payload = {"base_version": workspace.snapshot()["version"],
                   "facts": [fact(workspace, f"damage-{index}", index + 1, value=description)],
                   "draft_patch": [{"op": "append_row", "section": "病害明细",
                                    "row": {"病害部位": f"左幅第{index + 1}号梁", "病害描述": description},
                                    "fact_ids": [f"damage-{index}"]}]}
        # Each generated update stays small as the accumulated report grows.
        assert len(json.dumps(payload, ensure_ascii=False)) < 1500
        workspace.apply_update(payload)
    state = workspace.snapshot()
    assert len(json.dumps(state["draft"], ensure_ascii=False)) > 12000
    assert [row["病害描述"] for row in state["draft"]["病害明细"]] == descriptions
    assert state["draft_bindings"]["病害明细.63"] == ["damage-63"]
    assert set(state["draft_fact_ids"]) == {"bridge", *(f"damage-{i}" for i in range(64))}
    assert state["review"] is None


def test_patch_delete_reindexes_bindings_and_replace_uses_the_new_index(tmp_path):
    workspace = make_workspace(tmp_path, "桥名：合成桥\n甲构件\n乙构件\n丙构件\n")
    read_all(workspace)
    update(workspace, facts=[fact(workspace), *(fact(workspace, name, i + 1) for i, name in enumerate(("A", "B", "C")))],
           draft_patch=[{"op": "set_basic", "field": "桥梁名称", "value": "合成桥", "fact_ids": ["bridge"]},
                        *({"op": "append_row", "section": "病害明细", "row": {"病害部位": name}, "fact_ids": [name]}
                          for name in ("A", "B", "C")),
                        {"op": "append_row", "section": "维修建议", "row": {"建议内容": "观察C"}, "fact_ids": ["C"]}])
    prior = workspace.snapshot()["version"]
    state = update(workspace, draft_patch=[
        {"op": "remove_row", "section": "病害明细", "index": 1},
        {"op": "replace_row", "section": "病害明细", "index": 1, "row": {"病害部位": "C精确部位"}, "fact_ids": ["C"]},
        {"op": "append_row", "section": "病害明细", "row": {"病害部位": "B重新核实"}, "fact_ids": ["B"]},
        {"op": "remove_row", "section": "维修建议", "index": 0}])
    assert state["version"] == prior + 1
    assert state["draft"]["病害明细"] == [{"病害部位": "A"}, {"病害部位": "C精确部位"}, {"病害部位": "B重新核实"}]
    assert state["draft_bindings"] == {"基本信息.桥梁名称": ["bridge"], "病害明细.0": ["A"], "病害明细.1": ["C"], "病害明细.2": ["B"]}
    state = update(workspace, draft_patch=[{"op": "remove_row", "section": "病害明细", "index": 2}])
    assert "B" not in state["draft_fact_ids"]
    assert "B" in state["facts"]  # Removing an output row does not retract its evidence.


@pytest.mark.parametrize("explicit", ["draft", "draft_fact_ids", "draft_bindings"])
def test_incremental_and_full_draft_updates_cannot_be_mixed(tmp_path, explicit):
    workspace = make_workspace(tmp_path)
    original = workspace.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceError, match="mutually exclusive"):
        update(workspace, draft_patch=[{"op": "remove_row", "section": "病害明细", "index": 0}], **{explicit: {}})
    assert workspace.checkpoint_path.read_bytes() == original


def test_patch_failure_is_atomic_including_earlier_ops_and_new_facts(tmp_path):
    workspace = ready_workspace(tmp_path)
    original = workspace.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceError, match="row index"):
        update(workspace, facts=[fact(workspace, "new")], draft_patch=[
            {"op": "set_basic", "field": "桥梁名称", "value": "不应落盘", "fact_ids": ["new"]},
            {"op": "remove_row", "section": "病害明细", "index": 999}])
    assert workspace.checkpoint_path.read_bytes() == original
    assert workspace.delivery_status()["ready"]
    # A later removal does not make an invalid earlier operation acceptable.
    with pytest.raises(WorkspaceError, match="active fact"):
        update(workspace, draft_patch=[
            {"op": "append_row", "section": "病害明细", "row": {"病害描述": "不存在的依据"}, "fact_ids": ["missing"]},
            {"op": "remove_row", "section": "病害明细", "index": 0}])
    assert workspace.checkpoint_path.read_bytes() == original


def test_patch_respects_stale_versions_and_invalidates_prior_review(tmp_path):
    workspace = ready_workspace(tmp_path)
    stale = workspace.snapshot()["version"]
    state = update(workspace, draft_patch=[{"op": "set_basic", "field": "桥梁名称", "value": "合成桥", "fact_ids": ["bridge"]}])
    assert state["review"] is None
    assert not workspace.delivery_status()["ready"]
    with pytest.raises(WorkspaceConflictError):
        workspace.apply_update({"base_version": stale, "draft_patch": [
            {"op": "set_basic", "field": "桥梁名称", "value": "过期更改", "fact_ids": ["bridge"]}]})
    workspace.record_review(review(workspace))
    assert workspace.delivery_status()["ready"]


def test_patch_cannot_resurrect_invalid_dependency_without_revalidation(tmp_path):
    workspace = make_workspace(tmp_path)
    read_all(workspace)
    update(workspace, facts=[fact(workspace), fact(workspace, "damage", 1, depends_on=["bridge"])],
           draft_patch=[{"op": "set_basic", "field": "桥梁名称", "value": "合成桥", "fact_ids": ["bridge"]},
                        {"op": "append_row", "section": "病害明细", "row": {"病害描述": "裂缝"}, "fact_ids": ["damage"]}])
    workspace.record_review(review(workspace))
    update(workspace, facts=[{"id": "bridge", "value": "更正桥名"}])
    with pytest.raises(WorkspaceError, match="missing or invalid"):
        update(workspace, draft_patch=[{"op": "set_basic", "field": "桥梁名称", "value": "更正桥名", "fact_ids": ["bridge"]}])
    assert not workspace.snapshot()["draft_valid"]
    state = update(workspace, facts=[{"id": "damage"}], draft_patch=[
        {"op": "set_basic", "field": "桥梁名称", "value": "更正桥名", "fact_ids": ["bridge"]},
        {"op": "replace_row", "section": "病害明细", "index": 0, "row": {"病害描述": "重新核对的裂缝"}, "fact_ids": ["damage"]}])
    assert state["draft_valid"] and state["review"] is None
    assert not workspace.delivery_status()["ready"]
    workspace.record_review(review(workspace))
    assert workspace.delivery_status()["ready"]
