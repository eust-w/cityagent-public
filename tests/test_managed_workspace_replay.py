"""Trusted invocation replays cannot consume another workspace/entry page."""
from copy import deepcopy
import hashlib
import json
from threading import RLock

import pytest

from bridge_evidence.contest import document_workspace_tools as module
from bridge_evidence.contest.document_workspace_tools import ManagedEntriesPager, WorkspaceTools
from test_managed_workspace_navigation import Entries, Workspace, encoded, workspace_tools


def invocation(generation="g1", call="c1", arguments=None):
    return {"run_id": "run-C1", "session_id": "session-main", "generation_call_id": generation,
            "native_tool_call_id": call,
            "args_sha256": hashlib.sha256(encoded(arguments or {"view": "facts", "limit": 1}).encode()).hexdigest()}


@pytest.fixture
def active(monkeypatch):
    value = {"identity": invocation()}
    monkeypatch.setattr(module, "_current_call_identity", lambda: deepcopy(value["identity"]))
    return value


def same_page(replayed, original):
    assert replayed == {**original, "replayed": True}


def test_exact_native_replay_then_new_generation_advances(active):
    workspace, tools = workspace_tools()
    first = tools.read_page_managed("facts", limit=1)
    assert first["ok"] and not first["replayed"]
    checkpoint = tools.managed_navigation_snapshot()
    same_page(tools.read_page_managed("facts", limit=1), first)
    assert tools.managed_navigation_snapshot() == checkpoint
    active["identity"] = invocation("g2", "c2")
    second = tools.read_page_managed("facts", limit=1)
    assert second["start"] == 1 and second["items"][0]["id"] == "F1"
    assert second["version"] == first["version"] == workspace.state["version"]


def test_same_generation_new_native_same_object_never_consumes_a_second_page(active):
    _, tools = workspace_tools()
    first = tools.read_page_managed("facts", limit=1)
    checkpoint = tools.managed_navigation_snapshot()
    active["identity"] = invocation("g1", "c2", {"view": "facts", "limit": 4, "restart": True})
    same_page(tools.read_page_managed("facts", limit=4, restart=True), first)
    assert tools.managed_navigation_snapshot() == checkpoint
    active["identity"] = invocation("g2", "c3")
    assert tools.read_page_managed("facts", limit=1)["start"] == 1


def test_different_views_in_same_generation_progress_independently(active):
    workspace, tools = workspace_tools()
    workspace.state["questions"] = {"Q1": {"id": "Q1"}, "Q2": {"id": "Q2"}}
    first = tools.read_page_managed("facts", limit=1)
    active["identity"] = invocation("g1", "c2", {"view": "questions", "limit": 1})
    question = tools.read_page_managed("questions", limit=1)
    assert question["items"][0]["id"] == "Q1"
    active["identity"] = invocation("g1", "c3")
    same_page(tools.read_page_managed("facts", limit=1), first)


@pytest.mark.parametrize("change", ["arguments", "view"])
def test_native_identity_cannot_be_reused_for_different_arguments_or_view(active, change):
    _, tools = workspace_tools()
    tools.read_page_managed("facts", limit=1)
    before = tools.managed_navigation_snapshot()
    if change == "arguments":
        active["identity"] = invocation(arguments={"view": "facts", "limit": 2})
        result = tools.read_page_managed("facts", limit=2)
    else:
        result = tools.read_page_managed("questions", limit=1)
    assert not result["ok"] and "replay changed arguments or object" in result["error"]["message"]
    assert tools.managed_navigation_snapshot() == before


def test_version_change_in_same_generation_replays_original_then_next_generation_restarts(active):
    workspace, tools = workspace_tools()
    first = tools.read_page_managed("facts", limit=1)
    workspace.state["version"] += 1
    workspace.state["facts"]["F0"]["value"] = "改后事实"
    active["identity"] = invocation("g1", "c2")
    same_page(tools.read_page_managed("facts", limit=1), first)
    active["identity"] = invocation("g2", "c3")
    new = tools.read_page_managed("facts", limit=1)
    assert new["version"] == 4 and new["start"] == 0 and not new["replayed"]
    assert new["items"][0]["value"] == "改后事实"
    assert new["navigation"]["version_changed"]


def test_reload_restores_page_receipt_and_cursor_atomically(active, tmp_path):
    workspace, tools = workspace_tools()
    target = tmp_path / "navigation.json"
    tools.configure_managed_navigation(on_progress=lambda value: target.write_text(encoded(value)))
    first = tools.read_page_managed("facts", limit=1)
    saved = json.loads(target.read_text())
    resumed = WorkspaceTools(workspace, count_tokens=len)
    resumed.configure_managed_navigation(saved)
    same_page(resumed.read_page_managed("facts", limit=1), first)
    assert resumed.managed_navigation_snapshot() == saved
    active["identity"] = invocation("g2", "c2")
    assert resumed.read_page_managed("facts", limit=1)["start"] == 1


def test_checkpoint_write_failure_records_neither_receipt_nor_position(active):
    workspace, tools = workspace_tools()
    fail = {"enabled": True}

    def save(_state):
        if fail["enabled"]:
            raise OSError("disk write failed")

    tools.configure_managed_navigation(on_progress=save)
    before = tools.managed_navigation_snapshot()
    assert not tools.read_page_managed("facts", limit=1)["ok"]
    assert tools.managed_navigation_snapshot() == before
    fail["enabled"] = False
    first = tools.read_page_managed("facts", limit=1)
    assert first["start"] == 0 and not first["replayed"]
    same_page(tools.read_page_managed("facts", limit=1), first)


def test_fragment_replay_restore_and_new_generation_have_no_gaps(active):
    record = {"id": "F0", "value": "长记录甲😀" * 300}
    workspace = Workspace(facts={"F0": record})
    tools = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=850, native_response_budget=True)
    fragments = []
    offset = 0
    for index in range(100):
        active["identity"] = invocation(f"g{index}", f"c{index}")
        page = tools.read_page_managed("facts", limit=1)
        assert page["ok"] and page["item_start"] == offset
        same_page(tools.read_page_managed("facts", limit=1), page)
        assert len(encoded(module.wrap_tool_response("read_workspace_page", page))) <= 850
        fragments.append(page["fragment"])
        offset = page["item_end"]
        saved = tools.managed_navigation_snapshot()
        tools = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=850, native_response_budget=True)
        tools.configure_managed_navigation(saved)
        same_page(tools.read_page_managed("facts", limit=1), page)
        if page["continuation"] is None:
            break
    else:
        pytest.fail("fragment pagination did not finish")
    assert json.loads("".join(fragments)) == record
    active["identity"] = invocation("g-next", "c-next")
    done = tools.read_page_managed("facts", limit=1)
    assert done["status"] == "already_read"
    same_page(tools.read_page_managed("facts", limit=1), done)


def test_entries_replay_after_recovery_does_not_skip_and_version_changes_are_explicit(active):
    store = Entries()
    pager = ManagedEntriesPager(store, count_tokens=len)
    first = pager.read_entries(limit=1)
    before = pager.managed_navigation_snapshot()
    resumed = ManagedEntriesPager(store, count_tokens=len, state=before)
    same_page(resumed.read_entries(limit=1), first)
    assert resumed.managed_navigation_snapshot() == before
    active["identity"] = invocation("g1", "other-native")
    same_page(resumed.read_entries(limit=1), first)
    active["identity"] = invocation("g2", "new-native")
    assert resumed.read_entries(limit=1)["entries"][0]["id"] == "1"
    store.version += 1
    store.entries[0]["value"] = "new"
    active["identity"] = invocation("g3", "new-version")
    changed = resumed.read_entries(limit=1)
    assert changed["cursor"] == 0 and changed["navigation"]["version_changed"]


def test_entries_checkpoint_failure_does_not_commit_journal_or_cursor(active):
    store = Entries()
    fail = {"enabled": True}

    def save(_state):
        if fail["enabled"]:
            raise OSError("disk write failed")

    pager = ManagedEntriesPager(store, count_tokens=len, on_progress=save)
    before = pager.managed_navigation_snapshot()
    assert not pager.read_entries(limit=1)["ok"]
    assert pager.managed_navigation_snapshot() == before
    fail["enabled"] = False
    first = pager.read_entries(limit=1)
    assert first["cursor"] == 0
    same_page(pager.read_entries(limit=1), first)


def test_invalid_trusted_identity_cannot_advance(active):
    _, tools = workspace_tools()
    before = tools.managed_navigation_snapshot()
    active["identity"] = {"generation_call_id": "g1"}
    assert not tools.read_page_managed("facts", limit=1)["ok"]
    assert tools.managed_navigation_snapshot() == before


def test_unresolved_questions_excludes_only_valid_resolved_and_keeps_full_view(active):
    workspace, tools = workspace_tools()
    workspace.state["questions"] = {
        "Q1": {"id": "Q1", "valid": True, "status": "resolved"},
        "Q2": {"id": "Q2", "valid": True, "status": "open"},
        "Q3": {"id": "Q3", "valid": False, "status": "resolved"},
        "Q4": {"id": "Q4", "valid": True, "status": "unknown"},
        "Q5": {"id": "Q5", "valid": True, "status": "conflict"},
    }
    pending = tools.read_page_managed("unresolved_questions", limit=2)
    assert [item["id"] for item in pending["items"]] == ["Q2", "Q3"]
    active["identity"] = invocation("g2", "next-pending", {"view": "unresolved_questions", "limit": 2})
    rest = tools.read_page_managed("unresolved_questions", limit=2)
    assert [item["id"] for item in rest["items"]] == ["Q4", "Q5"]
    active["identity"] = invocation("g3", "all-questions", {"view": "questions"})
    assert len(tools.read_page_managed("questions")["items"]) == 5
    workspace.state["questions"]["Q2"]["status"] = "resolved"
    workspace.state["version"] += 1
    active["identity"] = invocation("g4", "new-version", {"view": "unresolved_questions"})
    newer = tools.read_page_managed("unresolved_questions")
    assert [item["id"] for item in newer["items"]] == ["Q3", "Q4", "Q5"]
    assert newer["navigation"]["version_changed"]


def test_questions_next_page_uses_real_adapter_identity_without_changing_workspace_version(monkeypatch):
    """Actual adapter dispatch/context, synthetic native frames; no model/SDK."""
    from bridge_evidence.contest import tong_react_adapter as adapter

    workspace = Workspace(questions={f"Q{i}": {"id": f"Q{i}", "status": "open"} for i in range(3)})
    tools = WorkspaceTools(workspace, count_tokens=len, native_response_budget=True)
    origin = {"id": "generation-1"}
    session = {"run_id": "run-C1", "session_id": "main-C1", "tool_lock": RLock(),
        "handlers": {"read_workspace_page": tools.read_page_managed},
        "current_agent_call_id": lambda: origin["id"], "tool_calls": [], "max_calls": 100,
        "tool_budget_exceeded": False, "count_tokens": len, "max_tool_tokens": 4096}
    monkeypatch.setattr(adapter, "_ACTIVE_SESSION", session)

    def dispatch(native_id):
        session["native_dispatch"] = {"generation_call_id": origin["id"],
            "tool_name": "read_workspace_page", "native_tool_call_id": native_id,
            "arguments": encoded({"view": "questions", "limit": 1})}
        result = adapter._read_workspace_page("questions", limit=1)
        assert result["ok"], result
        assert adapter.current_tool_call_identity() is None
        return result["result"]

    first = dispatch("native-1")
    assert first["items"][0]["id"] == "Q0"
    same_page(dispatch("native-1"), first)
    same_page(dispatch("native-2"), first)
    origin["id"] = "generation-2"
    second = dispatch("native-3")
    assert second["items"][0]["id"] == "Q1"
    assert first["version"] == second["version"] == workspace.state["version"]
    assert session["tool_calls"][-1]["native_tool_call_id"] == "native-3"
    assert session["tool_calls"][-1]["agent_call_id"] == "generation-2"
