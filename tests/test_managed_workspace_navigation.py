"""Host-owned page positions stay lossless, version-bound and budget-aware."""
from copy import deepcopy
import hashlib
import json

import pytest

from bridge_evidence.contest.document_workspace_tools import ManagedEntriesPager, WorkspaceTools
from bridge_evidence.contest.tong_react_adapter import wrap_tool_response


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Workspace:
    def __init__(self, identity="C1", **values):
        self.state = {"identity": {"document": identity}, "version": 3, "facts_version": 1,
                      "facts": {}, "relations": {}, "questions": {}, "draft": {},
                      "draft_bindings": {}, "history": [], **values}

    def snapshot(self):
        return deepcopy(self.state)

    def delivery_status(self):
        return {"workspace_version": self.state["version"], "ready": False, "blockers": []}


def workspace_tools(**kwargs):
    workspace = Workspace(facts={f"F{i}": {"id": f"F{i}", "value": f"事实{i}"} for i in range(4)})
    return workspace, WorkspaceTools(workspace, count_tokens=len, **kwargs)


def test_views_independent_complete_and_explicit_restart():
    workspace, tools = workspace_tools()
    workspace.state["questions"] = {"Q1": {"id": "Q1", "text": "年份"}}
    first = tools.read_page_managed("facts", limit=2)
    assert [item["id"] for item in first["items"]] == ["F0", "F1"]
    assert first["continuation"]["arguments"] == {"view": "facts", "limit": 2, "restart": False}
    assert tools.read_page_managed("questions")["items"][0]["id"] == "Q1"
    assert [item["id"] for item in tools.read_page_managed("facts", limit=2)["items"]] == ["F2", "F3"]
    done = tools.read_page_managed("facts")
    assert done["status"] == "already_read" and done["items"] == []
    assert done["navigation"]["read_complete"]
    assert done["navigation"]["read_is_not_semantic_review"]
    assert tools.read_page_managed("facts", limit=1, restart=True)["items"][0]["id"] == "F0"


def test_changed_version_restarts_the_view_and_does_not_conceal_restart():
    workspace, tools = workspace_tools()
    tools.read_page_managed("facts", limit=2)
    workspace.state["version"] += 1
    workspace.state["facts"]["F0"]["value"] = "修订事实"
    result = tools.read_page_managed("facts", limit=1)
    assert result["start"] == 0 and result["version"] == 4
    assert result["items"][0]["value"] == "修订事实"
    assert result["navigation"]["version_changed"]
    assert result["navigation"]["previous_version"] == 3
    assert "discard old fragments" in result["navigation"]["action"]


def test_large_record_fragments_have_no_gaps_and_native_budget_includes_navigation():
    record = {"id": "F0", "value": "测量数据甲😀" * 600}
    workspace = Workspace(facts={"F0": record})
    tools = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=850, native_response_budget=True)
    fragments, position = [], 0
    for _ in range(200):
        page = tools.read_page_managed("facts", limit=1)
        assert page["ok"] and page["mode"] == "canonical_json_fragment"
        assert page["item_start"] == position
        assert len(encoded(wrap_tool_response("read_workspace_page", page))) <= 850
        assert not page["fragment_is_complete_record"]
        fragments.append(page["fragment"])
        position = page["item_end"]
        if page["continuation"] is None:
            break
        assert set(page["continuation"]["arguments"]) == {"view", "limit", "restart"}
    else:
        pytest.fail("Managed fragments made no bounded progress")
    recovered = "".join(fragments)
    assert json.loads(recovered) == record
    assert hashlib.sha256(recovered.encode()).hexdigest() == page["item_sha256"]
    assert tools.read_page_managed("facts")["status"] == "already_read"


def test_version_changes_during_fragment_restart_at_zero():
    workspace = Workspace(facts={"F0": {"id": "F0", "value": "长段落" * 600}})
    tools = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=850)
    first = tools.read_page_managed("facts", limit=1)
    assert first["item_end"] > 0
    workspace.state["version"] += 1
    workspace.state["facts"]["F0"]["value"] = "完全不同的新段落" * 600
    second = tools.read_page_managed("facts", limit=1)
    assert second["item_start"] == 0
    assert second["item_sha256"] != first["item_sha256"]
    assert second["navigation"]["version_changed"]


def test_fragment_offset_survives_navigation_reload():
    workspace = Workspace(facts={"F0": {"id": "F0", "value": "片段" * 1000}})
    tools = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=850)
    first = tools.read_page_managed("facts", limit=1)
    saved = tools.managed_navigation_snapshot()
    resumed = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=850)
    resumed.configure_managed_navigation(saved)
    following = resumed.read_page_managed("facts", limit=1)
    assert following["item_start"] == first["item_end"]
    assert following["item_sha256"] == first["item_sha256"]


def test_failed_page_or_progress_save_never_advances():
    workspace, tools = workspace_tools()
    tools.read_page_managed("facts", limit=1)
    before = tools.managed_navigation_snapshot()
    assert not tools.read_page_managed("facts", limit=0)["ok"]
    assert tools.managed_navigation_snapshot() == before
    workspace.state["facts"]["F1"]["value"] = "long" * 500
    old_budget = tools.max_tool_tokens
    tools.max_tool_tokens = 400
    assert not tools.read_page_managed("facts", limit=1)["ok"]
    assert tools.managed_navigation_snapshot() == before
    tools.max_tool_tokens = old_budget

    def failed_save(_state):
        raise OSError("disk unavailable")

    tools.configure_managed_navigation(before, failed_save)
    assert not tools.read_page_managed("facts", limit=1)["ok"]
    assert tools.managed_navigation_snapshot() == before
    tools.configure_managed_navigation(before)
    assert tools.read_page_managed("facts", limit=1)["items"][0]["id"] == "F1"


def test_persisted_navigation_can_resume_only_same_document():
    workspace, tools = workspace_tools()
    saved = []
    tools.configure_managed_navigation(on_progress=saved.append)
    tools.read_page_managed("facts", limit=1)
    resumed = WorkspaceTools(workspace, count_tokens=len)
    resumed.configure_managed_navigation(saved[-1])
    assert resumed.read_page_managed("facts", limit=1)["items"][0]["id"] == "F1"
    other = WorkspaceTools(Workspace("other-document"), count_tokens=len)
    with pytest.raises(ValueError, match="different document"):
        other.configure_managed_navigation(saved[-1])


def test_racing_workspace_update_restarts_without_advancing_old_version(monkeypatch):
    workspace, tools = workspace_tools()
    tools.read_page_managed("facts", limit=1)
    original = WorkspaceTools.read_page
    changed = False

    def read_page(instance, *args, **kwargs):
        nonlocal changed
        if not changed:
            workspace.state["version"] += 1
            changed = True
        return original(instance, *args, **kwargs)

    monkeypatch.setattr(WorkspaceTools, "read_page", read_page)
    result = tools.read_page_managed("facts", limit=1)
    assert result["version"] == 4 and result["start"] == 0
    assert result["navigation"]["version_changed"]


class Entries:
    def __init__(self):
        self.version = 1
        self.entries = [{"id": str(i), "value": "条目"} for i in range(4)]
        self.fail = False

    def snapshot(self):
        return {"identity": {"document": "C1", "task": "defects"}, "version": self.version,
                "counts": {"total": len(self.entries)}}

    def read_entries(self, cursor=0, limit=8):
        if self.fail:
            raise OSError("temporary failure")
        end = min(cursor + limit, len(self.entries))
        return {"ok": True, "version": self.version, "entries": deepcopy(self.entries[cursor:end]),
                "cursor": cursor, "next_cursor": end if end < len(self.entries) else None,
                "counts": {"total": len(self.entries)}, "delivery_complete": False}


def test_entries_continue_complete_restart_version_and_resume():
    store = Entries()
    saved = []
    pager = ManagedEntriesPager(store, count_tokens=len, on_progress=saved.append)
    assert [entry["id"] for entry in pager.read_entries(limit=2)["entries"]] == ["0", "1"]
    resumed = ManagedEntriesPager(store, count_tokens=len, state=saved[-1])
    assert [entry["id"] for entry in resumed.read_entries(limit=2)["entries"]] == ["2", "3"]
    assert resumed.read_entries()["status"] == "already_read"
    assert resumed.read_entries(restart=True, limit=1)["entries"][0]["id"] == "0"
    store.version += 1
    result = resumed.read_entries(limit=1)
    assert result["cursor"] == 0 and result["navigation"]["version_changed"]
    assert result["delivery_complete"] is False


def test_entry_read_failure_and_oversize_entry_keep_pointer():
    store = Entries()
    pager = ManagedEntriesPager(store, count_tokens=len, max_tool_tokens=900)
    pager.read_entries(limit=1)
    before = pager.managed_navigation_snapshot()
    store.fail = True
    assert not pager.read_entries(limit=1)["ok"]
    assert pager.managed_navigation_snapshot() == before
    store.fail = False
    store.entries[1]["value"] = "很长" * 1000
    assert pager.read_entries(limit=1)["status"] == "budget_exceeded"
    assert pager.managed_navigation_snapshot() == before
    store.entries[1]["value"] = "正常"
    assert pager.read_entries(limit=1)["entries"][0]["id"] == "1"


def test_entries_reduce_page_to_fit_without_skipping():
    store = Entries()
    for entry in store.entries:
        entry["value"] = "事实" * 100
    pager = ManagedEntriesPager(store, count_tokens=len, max_tool_tokens=850)
    actual = []
    while True:
        result = pager.read_entries(limit=8)
        assert result["ok"]
        assert len(encoded(wrap_tool_response("read_entries", result))) <= 850
        actual.extend(result["entries"])
        if not result["continuation"]:
            break
    assert actual == store.entries


def test_entry_version_race_restarts_and_progress_save_failure_does_not_skip(monkeypatch):
    store = Entries()
    pager = ManagedEntriesPager(store, count_tokens=len)
    pager.read_entries(limit=1)
    original = store.read_entries
    changed = False

    def read_entries(*args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            store.version += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "read_entries", read_entries)
    result = pager.read_entries(limit=1)
    assert result["cursor"] == 0 and result["navigation"]["version_changed"]
    before = pager.managed_navigation_snapshot()

    def fail_save(_state):
        raise OSError("disk failed")

    pager.navigation.on_progress = fail_save
    assert not pager.read_entries(limit=1)["ok"]
    assert pager.managed_navigation_snapshot() == before


def test_pager_with_real_entry_store_preserves_entries_and_provenance(tmp_path):
    from bridge_evidence.contest.document_entries import EntryStore
    from bridge_evidence.contest.evidence_manifest import manifest_from_text

    manifest = manifest_from_text("桥名甲\n2018年")
    store = EntryStore(manifest, "overview", tmp_path / "entries")
    unit = manifest["units"][0]
    store.record_read(unit["id"], 0, len(unit["text"]))
    store.save_entries([{"id": "name", "section": "基本信息", "field": "桥梁名称", "value": "桥名甲",
                         "status": "explicit", "source_refs": [{"evidence_id": unit["id"]}]}])
    pager = ManagedEntriesPager(store, count_tokens=len)
    result = pager.read_entries()
    assert result["entries"] == store.read_entries()["entries"]
    assert result["delivery_complete"] is False
    assert pager.read_entries()["status"] == "already_read"
