"""Local synthetic memo persistence/budget tests; no SDK, model or report data."""
from copy import deepcopy
import hashlib
import json

import pytest

from bridge_evidence.contest.document_workspace import (
    DocumentWorkspace, WorkspaceConflictError, WorkspaceError, WorkspaceIdentityError,
)
from bridge_evidence.contest.document_workspace_tools import WorkspaceTools
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.tong_react_adapter import wrap_tool_response


IDENTITY = {"model": "synthetic-notes-test", "prompt": "fixture-only"}


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def workspace(tmp_path):
    return DocumentWorkspace(manifest_from_text("合成桥\n左幅一号梁裂缝\n"), tmp_path, IDENTITY)


def update(store, **values):
    return store.apply_update({"base_version": store.snapshot()["version"], **values})


def supported_fact(store, **values):
    unit = store.manifest["units"][0]
    return {"id": "F1", "value": "合成桥", "status": "explicit",
            "source_refs": [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}], **values}


def read_all(store):
    for unit in store.manifest["units"]:
        store.record_read(unit["id"], 0, len(unit["text"]))


def ready_workspace(tmp_path):
    store = workspace(tmp_path)
    read_all(store)
    state = update(store, facts=[supported_fact(store)],
                   draft={"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [], "病害明细": []},
                   draft_fact_ids=["F1"], draft_bindings={"基本信息.桥梁名称": ["F1"]})
    store.record_review({"workspace_version": state["version"], "draft_sha256": state["draft_sha256"],
                         "checked_evidence_ids": state["reading_progress"]["fully_read_evidence_ids"],
                         "passed": True, "model_call_ids": ["synthetic-review-receipt"],
                         **{key: [] for key in ("unsupported_claims", "missing_items", "conflicts", "unresolved_items")}})
    assert store.delivery_status()["ready"]
    return store


def test_notes_replace_survive_restore_and_clear_without_mutable_aliases(tmp_path):
    store = workspace(tmp_path)
    notes = {"next": {"tool": "read_source", "evidence_id": "E000002"},
             "reason": "继续核对，尚未完成", "flags": [True, False, None, 0, 1.5]}
    state = update(store, working_notes=notes)
    notes["reason"] = "caller mutation"
    state["working_notes"]["next"]["tool"] = "caller mutation"
    restored = DocumentWorkspace(store.manifest, tmp_path, IDENTITY)
    assert restored.snapshot()["working_notes"]["reason"] == "继续核对，尚未完成"
    assert restored.snapshot()["working_notes"]["next"]["tool"] == "read_source"
    update(restored, questions=[{"id": "Q1", "question": "待核对"}])
    assert restored.snapshot()["working_notes"] == store.snapshot()["working_notes"]
    update(restored, working_notes={"next": "new task"})
    assert store.snapshot()["working_notes"] == {"next": "new task"}
    update(store, working_notes={})
    assert restored.snapshot()["working_notes"] == {}


def test_legacy_checkpoint_defaults_to_empty_without_rewriting_or_invalidating_review(tmp_path):
    store = ready_workspace(tmp_path)
    state = json.loads(store.checkpoint_path.read_text())
    state.pop("working_notes")
    store.checkpoint_path.write_bytes(packed(state))
    before = store.checkpoint_path.read_bytes()
    restored = DocumentWorkspace(store.manifest, tmp_path, IDENTITY)
    assert restored.snapshot()["working_notes"] == {}
    assert restored.delivery_status()["ready"]
    assert restored.checkpoint_path.read_bytes() == before


def test_notes_use_utf8_serialized_limit_and_allow_exact_boundary(tmp_path):
    store = workspace(tmp_path)
    prefix = "桥" * 1200
    notes = {"note": prefix}
    notes["note"] += "x" * (4096 - len(packed(notes)))
    assert len(packed(notes)) == 4096
    assert update(store, working_notes=notes)["working_notes"] == notes
    before = store.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceError, match="4096 UTF-8 bytes"):
        update(store, working_notes={"note": notes["note"] + "x"})
    assert store.checkpoint_path.read_bytes() == before


@pytest.mark.parametrize("notes", [
    [], None, "memo", {"x": float("nan")}, {"x": [float("inf")]},
    {"x": float("-inf")}, {1: "lossy-key"}, {"nested": {False: "lossy-key"}},
    {"tuple": (1, 2)}, {"set": {1}}, {"bad-unicode": "\ud800"},
])
def test_non_json_or_nonfinite_notes_never_change_checkpoint(tmp_path, notes):
    store = workspace(tmp_path)
    update(store, working_notes={"preserve": True})
    before = store.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceError, match="finite JSON object"):
        update(store, working_notes=notes)
    assert store.checkpoint_path.read_bytes() == before


def test_cyclic_notes_are_rejected_without_partial_persistence(tmp_path):
    store = workspace(tmp_path)
    cyclic = {}
    cyclic["self"] = cyclic
    before = store.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceError, match="finite JSON object"):
        update(store, working_notes=cyclic)
    assert store.checkpoint_path.read_bytes() == before


def test_stale_notes_and_notes_with_invalid_fact_fail_atomically(tmp_path):
    store = workspace(tmp_path)
    version = store.snapshot()["version"]
    other = DocumentWorkspace(store.manifest, tmp_path, IDENTITY)
    update(store, working_notes={"next": "read source"})
    before = store.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceConflictError):
        other.apply_update({"base_version": version, "working_notes": {"stale": True}})
    with pytest.raises(WorkspaceError, match="not actually read"):
        update(store, working_notes={"must-not-save": True}, facts=[supported_fact(store)])
    assert store.checkpoint_path.read_bytes() == before
    read_all(store)
    before = store.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceError, match="4096"):
        update(store, working_notes={"too-long": "x" * 4096}, facts=[supported_fact(store)])
    assert store.checkpoint_path.read_bytes() == before


def test_note_only_update_preserves_evidence_semantics_but_invalidates_review(tmp_path):
    store = ready_workspace(tmp_path)
    before = store.snapshot()
    after = update(store, working_notes={"next": "inspect historical comparison"})
    for key in ("facts", "facts_version", "relations", "questions", "draft", "draft_sha256",
                "draft_fact_ids", "draft_bindings", "draft_valid", "reads", "image_reads", "exclusions", "reading_progress"):
        assert after[key] == before[key], key
    assert after["version"] == before["version"] + 1
    assert after["review"] is None
    assert "semantic_review_missing" in store.delivery_status()["blockers"]


def test_notes_cannot_grant_coverage_supply_source_or_create_missing_fact(tmp_path):
    store = workspace(tmp_path)
    before = store.snapshot()
    state = update(store, working_notes={"facts": [supported_fact(store)], "read": "E000001",
                                        "draft": {"病害明细": [{"病害描述": "笔记不是依据"}]}})
    assert state["facts"] == {} and state["draft"] is None
    assert state["reads"] == before["reads"] == {}
    assert state["reading_progress"] == before["reading_progress"]
    checkpoint = store.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceError, match="Unknown evidence ID"):
        update(store, facts=[supported_fact(store, source_refs=[{"evidence_id": "working_notes", "start": 0, "end": 1}])])
    with pytest.raises(WorkspaceError, match="missing/invalid"):
        update(store, facts=[supported_fact(store, source_refs=[], depends_on=["notes-fact"])])
    assert store.checkpoint_path.read_bytes() == checkpoint


def test_unknown_update_fields_remain_rejected_even_with_valid_notes(tmp_path):
    store = workspace(tmp_path)
    before = store.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceError, match="Unknown update fields"):
        update(store, working_notes={"valid": True}, working_notes_history=[])
    assert store.checkpoint_path.read_bytes() == before


def test_saved_malformed_notes_fail_closed_on_restore(tmp_path):
    store = workspace(tmp_path)
    state = json.loads(store.checkpoint_path.read_text())
    state["working_notes"] = ["invalid persisted shape"]
    store.checkpoint_path.write_bytes(packed(state))
    before = store.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceIdentityError, match="working_notes"):
        DocumentWorkspace(store.manifest, tmp_path, IDENTITY)
    assert store.checkpoint_path.read_bytes() == before


def test_full_summary_and_notes_page_return_only_latest_memo_not_history(tmp_path):
    store = workspace(tmp_path)
    update(store, working_notes={"old": "PREVIOUS-NOTES-MUST-NOT-REENTER-PROMPT"})
    current = {"next": {"evidence_id": "E000002", "start": 0, "end": 8}}
    update(store, working_notes=current)
    tools = WorkspaceTools(store, count_tokens=len)
    summary = tools.summary()
    page = tools.read_page("working_notes", 0, 1)
    assert summary["mode"] == "full" and summary["working_notes"] == current
    assert page["items"] == [current] and page["total"] == 1 and page["continuation"] is None
    assert "PREVIOUS-NOTES" not in json.dumps([summary, page])
    assert "history" not in summary and "working_notes" not in summary.get("unexposed_fields", [])
    assert tools.read_page("working_notes_history", 0, 1)["ok"] is False
    assert tools.read_page("working_notes", 1, 1)["items"] == []
    page["items"][0]["next"]["start"] = 100
    assert store.snapshot()["working_notes"] == current


def test_paged_summary_has_hash_and_reader_for_large_notes_without_truncating(tmp_path):
    store = workspace(tmp_path)
    notes = {"next": "完整模型工作笔记" + "x" * 2700}
    update(store, working_notes=notes,
           questions=[{"id": f"Q{i}", "question": "synthetic task " * 25} for i in range(20)])
    counter = lambda text: len(text.encode("utf-8"))
    tools = WorkspaceTools(store, count_tokens=counter, max_tool_tokens=2200, native_response_budget=True)
    summary = tools.summary()
    assert summary["ok"] and summary["mode"] == "paged"
    assert summary["ready"] is False and summary["draft_valid"] is False
    assert summary["counts"]["questions"] == 20
    assert summary["reading_progress"]["text_coverage_complete"] is False
    assert summary["update_effect"]["base_version"] == 0
    assert summary["update_effect"]["update_version"] == summary["version"] == 1
    assert summary["update_effect"]["draft_changed"] is False
    assert summary["update_effect"]["draft_row_delta"] == {"病害明细": 0, "维修建议": 0}
    assert summary["draft_invalidation_feedback"]["diagnostic_only"] is True
    assert "working_notes" not in summary
    assert summary["counts"]["working_notes"] == 1
    assert summary["working_notes_ref"] == {"present": True, "sha256": hashlib.sha256(packed(notes)).hexdigest(),
                                           "view": "working_notes", "cursor": 0, "limit": 1, "is_evidence": False}
    assert len(packed(wrap_tool_response("read_workspace", summary))) <= 2200
    before = store.checkpoint_path.read_bytes()
    page = tools.read_page("working_notes", 0, 1)
    fragments = []
    while True:
        assert page["ok"] and page["mode"] == "canonical_json_fragment" and "items" not in page
        assert page["fragment_is_complete_record"] is False
        assert len(packed(wrap_tool_response("read_workspace_page", page))) <= 2200
        assert page["item_start"] == sum(len(part) for part in fragments)
        fragments.append(page["fragment"])
        if not page["continuation"]:
            break
        page = tools.read_page(**page["continuation"]["arguments"])
    joined = "".join(fragments)
    assert hashlib.sha256(joined.encode("utf-8")).hexdigest() == page["item_sha256"]
    assert json.loads(joined) == notes
    larger = WorkspaceTools(store, count_tokens=counter, max_tool_tokens=4096, native_response_budget=True)
    assert larger.read_page("working_notes", 0, 1)["items"] == [notes]
    assert store.checkpoint_path.read_bytes() == before
    update(store, working_notes={"next": "small memo fits alongside paged status"})
    smaller = tools.summary()
    assert smaller["mode"] == "paged" and smaller["working_notes"] == store.snapshot()["working_notes"]
    update(store, working_notes={})
    empty = tools.summary()
    assert empty["working_notes_ref"]["present"] is False
    assert empty["counts"]["working_notes"] == 0 and tools.read_page("working_notes", 0, 1)["items"] == []
