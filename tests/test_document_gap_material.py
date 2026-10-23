"""No-network source organization and serialized-budget checks."""
from copy import deepcopy

import pytest

from bridge_evidence.contest.document_gap_material import add_gap_material
from bridge_evidence.contest.document_handoff_bundle import (
    _Sources, build_handoff_bundle, serialize_handoff_bundle, validate_handoff_bundle)
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import json_digest
from test_document_material_index import make_material_case


def bundle_for(manifest, children):
    return build_handoff_bundle(manifest, {}, children, {
        "identity": {"manifest_sha256": json_digest(manifest)}, "version": 0,
        "facts": {}, "questions": {}})


def child(task, ids):
    return {"task_id": task, "status": "complete", "entry_count": 0, "primary_evidence_ids": ids}


def test_failed_scope_first_no_unassigned_source_and_input_unchanged():
    manifest = manifest_from_text("第一组原文\n失败组原文\n没有分工的原文")
    ids = [unit["id"] for unit in manifest["units"]]
    children = [child("a", ids[:1]), child("b", ids[1:2])]
    bundle = bundle_for(manifest, children)
    before = deepcopy(bundle)
    result = add_gap_material(manifest, bundle, children, ["b"])
    assert result["gap_material"]["added_evidence_ids"] == [ids[1], ids[0]]
    assert ids[2] not in {row["evidence_id"] for row in result["source_map"].values()}
    assert result["gap_material"]["not_added_unit_count"] == 0
    assert result["gap_material"]["partial_original_source_only"]
    assert not result["gap_material"]["source_delivery_confirmed"]
    assert bundle == before and result is not bundle
    assert validate_handoff_bundle(manifest, result)
    assert len(serialize_handoff_bundle(result)) - len(serialize_handoff_bundle(bundle)) <= 8000


def test_no_explicit_assignment_does_not_expand_compact_boundary_scope():
    manifest = manifest_from_text("甲\n乙\n丙")
    children = [{"task_id": "a", "scope": {"first_source": "E000001", "last_source": "E000003"}}]
    result = add_gap_material(manifest, bundle_for(manifest, children), children, [])
    assert result["source_map"] == {}
    assert result["gap_material"]["assigned_unit_count"] == 0


def test_foreign_source_and_bundle_identity_rejected():
    manifest = manifest_from_text("甲\n乙")
    children = [child("a", ["E000001"])]
    bundle = bundle_for(manifest, children)
    with pytest.raises(ValueError, match="outside"):
        add_gap_material(manifest, bundle, [child("a", ["E999999"])], [])
    foreign = deepcopy(bundle); foreign["document_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another"):
        add_gap_material(manifest, foreign, children, [])


def test_previously_provided_source_not_repeated_partial_source_finished_exactly():
    manifest = manifest_from_text("完整的原文单元\n另一个原文单元")
    children = [child("a", [u["id"] for u in manifest["units"]])]
    bundle = bundle_for(manifest, children)
    sources = _Sources(manifest)
    key, row = sources.source({"evidence_id": "E000001"})
    bundle["source_map"][key] = row
    key, row = sources.source({"evidence_id": "E000002", "start": 0, "end": 2})
    bundle["source_map"][key] = row
    result = add_gap_material(manifest, bundle, children, [])
    assert result["gap_material"]["added_evidence_ids"] == ["E000002"]
    rows = [r for r in result["source_map"].values() if r["evidence_id"] == "E000002"]
    assert sorted((r["start"], r["end"]) for r in rows) == [(0, 2), (2, len(manifest["units"][1]["text"]))]
    assert len([r for r in result["source_map"].values() if r["evidence_id"] == "E000001"]) == 1


def test_budget_skips_whole_oversized_unit_without_sentence_clipping():
    manifest = manifest_from_text("不可截断的完整句子。" * 2000 + "\n小单元")
    children = [child("a", [u["id"] for u in manifest["units"]])]
    bundle = bundle_for(manifest, children)
    result = add_gap_material(manifest, bundle, children, [], max_additional_characters=1000)
    assert "E000001" not in result["gap_material"]["added_evidence_ids"]
    assert "E000002" in result["gap_material"]["added_evidence_ids"]
    assert result["gap_material"]["not_added_unit_count"] == 1
    assert len(serialize_handoff_bundle(result)) - len(serialize_handoff_bundle(bundle)) <= 1000
    assert add_gap_material(manifest, bundle, children, [], max_additional_characters=0) == bundle


def test_table_full_row_header_candidate_and_merged_anchor_are_indivisible(tmp_path):
    case = make_material_case(tmp_path / "source")
    manifest = case["manifest"]
    table = manifest["tables"][0]
    table["header_kind"] = "first_row_candidate"
    row = next(u for u in manifest["units"] if u.get("table_id") == table["id"] and u.get("row_index") == 3)
    children = [child("failed", [row["id"]])]
    bundle = bundle_for(manifest, children)
    sources = _Sources(manifest)
    headers, anchors = sources.related({"evidence_id": row["id"]})
    assert headers and anchors
    result = add_gap_material(manifest, bundle, children, ["failed"])
    exposed = result["source_map"]
    for ref in [{"evidence_id": row["id"]}, *headers, *anchors]:
        key, source = sources.source({"evidence_id": ref["evidence_id"]})
        assert exposed[key] == source
    group = result["gap_material"]["context_groups"][0]
    assert group["header_kind"] == "first_row_candidate"
    assert group["table_id"] == table["id"]
    assert validate_handoff_bundle(manifest, result)
    small = add_gap_material(manifest, bundle, children, ["failed"], max_additional_characters=500)
    assert not small["source_map"]  # Never expose row without its required context.
    for identifier in result["gap_material"]["added_evidence_ids"]:
        key, full = sources.source({"evidence_id": identifier})
        assert result["source_map"][key] == full


def test_total_bundle_bound_and_exact_new_character_accounting():
    manifest = manifest_from_text("单元" * 100 + "\n" + "另一个单元" * 100)
    children = [child("a", [u["id"] for u in manifest["units"]])]
    bundle = bundle_for(manifest, children)
    bundle["semantics"] += "限" * 65000
    assert len(serialize_handoff_bundle(bundle)) < 68000
    result = add_gap_material(manifest, bundle, children, [])
    assert len(serialize_handoff_bundle(result)) <= 68000
    assert len(serialize_handoff_bundle(result)) - len(serialize_handoff_bundle(bundle)) <= 8000
