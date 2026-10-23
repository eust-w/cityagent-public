"""Lossless, version-bound transport of large workspace items; no model calls."""
import hashlib
import inspect
import json
from threading import Lock

import pytest

from bridge_evidence.contest import tong_react_adapter as adapter
from bridge_evidence.contest.document_workspace_tools import WorkspaceTools
from test_document_workspace_tools import SyntheticWorkspace, native_session


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def collect(workspace, view, *, counter=len, budget=800, start=None):
    """Consumer uses managed navigation and still verifies every item hash."""
    tools = WorkspaceTools(workspace, count_tokens=counter, max_tool_tokens=budget,
                           native_response_budget=True)
    session = native_session({"read_workspace_page": tools.read_page_managed}, counter, budget)
    session["max_calls"] = 1000
    call = start or {"view": view, "limit": 4, "restart": False}
    items, pending, packets = [], "", []
    expected_end, identity = 0, None
    while call:
        assert set(call) == {"view", "limit", "restart"}
        reply = adapter._invoke_active(session, "read_workspace_page", call)
        assert reply["ok"], reply
        assert counter(canonical(reply)) <= budget
        packet = reply["result"]
        packets.append(packet)
        if packet.get("mode") == "canonical_json_fragment":
            assert "items" not in packet and packet["fragment_is_complete_record"] is False
            current = (packet["version"], packet["view"], packet["item_index"], packet["item_sha256"])
            if identity is None:
                identity = current
            assert current == identity
            assert packet["item_start"] == expected_end
            assert packet["item_end"] > expected_end
            assert len(packet["fragment"]) == packet["item_end"] - expected_end
            pending += packet["fragment"]
            expected_end = packet["item_end"]
            if expected_end == packet["item_total_chars"]:
                assert hashlib.sha256(pending.encode("utf-8")).hexdigest() == packet["item_sha256"]
                items.append(json.loads(pending))
                pending, expected_end, identity = "", 0, None
        else:
            assert not pending, "Partial JSON must finish before whole items resume"
            items.extend(packet["items"])
        call = packet["continuation"]["arguments"] if packet["continuation"] else None
    assert not pending
    return items, packets, session


@pytest.mark.parametrize("view", ["facts", "draft_defects", "review_findings", "working_notes"])
@pytest.mark.parametrize("counter", [len, lambda text: len(text.encode("utf-8"))])
def test_all_large_item_kinds_round_trip_unicode_escaping_and_exact_native_budget(view, counter):
    value = '甲😀 e\u0301 "引号" \\路径\n\t\x00末尾' * 90
    if view == "facts":
        item = {"id": "F1", "value": value, "source_refs": [{"evidence_id": "E1", "start": 0, "end": 3}]}
        workspace = SyntheticWorkspace(facts={"F1": item})
    elif view == "draft_defects":
        item = {"序号": 1, "病害描述": value, "位置": None}
        workspace = SyntheticWorkspace(draft={"病害明细": [item]})
    elif view == "review_findings":
        finding = {"reason": value, "path": "病害明细.0"}
        item = {"category": "unresolved_items", "index": 0, "finding": finding}
        workspace = SyntheticWorkspace(review={"unresolved_items": [finding]})
    else:
        item = {"next": value, "is_evidence": False}
        workspace = SyntheticWorkspace(working_notes=item)
    before = workspace.snapshot()
    actual, packets, _ = collect(workspace, view, counter=counter)
    assert actual == [item] and len(packets) > 1
    assert all(p["mode"] == "canonical_json_fragment" for p in packets)
    assert workspace.snapshot() == before


def test_mixed_whole_and_large_items_recover_once_in_order_without_changing_legacy_pages():
    records = {"F0": {"id": "F0", "value": "小"}, "F1": {"id": "F1", "value": "大" * 2800},
               "F2": {"id": "F2", "value": "尾"}}
    workspace = SyntheticWorkspace(facts=records)
    recovered, packets, _ = collect(workspace, "facts")
    assert recovered == list(records.values())
    first, last = packets[0], packets[-1]
    assert "mode" not in first and first["items"] == [records["F0"]]
    assert first["continuation"]["arguments"] == {"view": "facts", "limit": 4, "restart": False}
    # Explicit cursors remain a host-internal transport API, with their old
    # exact lossless continuation, while the native model need not copy them.
    legacy = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=800,
                            native_response_budget=True).read_page("facts", 0, 4)
    assert legacy["continuation"]["arguments"] == {"view": "facts", "cursor": 1, "limit": 4}
    assert "mode" not in last and last["items"] == [records["F2"]]
    assert last["version"] == 3


def test_stale_fragment_returns_restart_without_any_new_fragment_and_can_recover():
    workspace = SyntheticWorkspace(facts={"F1": {"id": "F1", "value": "旧" * 1800}})
    tools = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=1100, native_response_budget=True)
    first = tools.read_page("facts", 0, 1)
    workspace.state["version"] += 1
    workspace.state["facts"]["F1"]["value"] = "新" * 1800
    # A saved low-level fragment cursor is host data, not a native-tool call.
    failed = adapter.wrap_tool_response("read_workspace_page",
        tools.read_page(**first["continuation"]["arguments"]))
    assert not failed["ok"] and failed["error"]["code"] == "WORKSPACE_PAGE_STALE"
    assert "fragment" not in failed["result"] and "items" not in failed["result"]
    assert "Discard" in failed["error"]["action"]
    assert len(canonical(failed)) <= 1100
    assert failed["result"]["restart"]["arguments"]["cursor"] == 0
    restored, _, _ = collect(workspace, "facts", budget=1100,
                            start={"view": "facts", "limit": 1, "restart": True})
    assert restored == [workspace.state["facts"]["F1"]]


@pytest.mark.parametrize("kwargs", [{"item_start": 1}, {"item_start": True}, {"item_start": -1},
    {"item_start": 1.0}, {"expected_version": True}, {"expected_version": -1},
    {"expected_version": "3"}, {"item_start": 999999, "expected_version": 3}])
def test_invalid_fragment_positions_are_rejected_without_writes_and_valid_retry_succeeds(kwargs):
    workspace = SyntheticWorkspace(facts={"F1": {"id": "F1", "value": "数据" * 500}})
    before = workspace.snapshot()
    tools = WorkspaceTools(workspace, count_tokens=len)
    result = tools.read_page("facts", 0, 1, **kwargs)
    assert not result["ok"] and "fragment" not in result
    actual, _, _ = collect(workspace, "facts")
    assert actual == [workspace.state["facts"]["F1"]] and workspace.snapshot() == before


def test_empty_end_page_and_explicit_stale_whole_page_do_not_create_fragments():
    tools = WorkspaceTools(SyntheticWorkspace(), count_tokens=len)
    empty = tools.read_page("facts", 0, 1, expected_version=3)
    assert empty["items"] == [] and empty["continuation"] is None and "fragment" not in empty
    assert not tools.read_page("facts", 0, 1, item_start=1, expected_version=3)["ok"]
    assert tools.read_page("facts", 0, 1, expected_version=2)["status"] == "version_conflict"


def test_fragment_continuation_is_stateless_and_exactly_replayable_after_tool_reconstruction():
    workspace = SyntheticWorkspace(facts={"F1": {"id": "F1", "value": "完整单条" * 500}})
    first_tools = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=800, native_response_budget=True)
    first = first_tools.read_page("facts", 0, 1)
    continuation = first["continuation"]["arguments"]
    expected = first_tools.read_page(**continuation)
    new_tools = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=800, native_response_budget=True)
    assert new_tools.read_page(**continuation) == expected
    assert expected["item_start"] == first["item_end"]
    assert expected["item_sha256"] == first["item_sha256"]


def test_protocol_budget_failure_preserves_record_and_larger_budget_can_resume_from_start():
    workspace = SyntheticWorkspace(facts={"F1": {"id": "F1", "value": "完整" * 500}})
    before = workspace.snapshot()
    # Synthetic tokenizer imposes a protocol cost only on fragments; even the
    # one-character envelope cannot fit, while the explicit error can.
    def counter(text):
        return len(text) + (1000 if '"mode":"canonical_json_fragment"' in text else 0)
    tools = WorkspaceTools(workspace, count_tokens=counter, max_tool_tokens=800, native_response_budget=True)
    reply = adapter._invoke_active(native_session({"read_workspace_page": tools.read_page_managed}, counter, 800),
                                   "read_workspace_page", {"view": "facts", "limit": 1, "restart": False})
    assert not reply["ok"] and "fragment" not in reply["result"]
    assert counter(canonical(reply)) <= 800 and workspace.snapshot() == before
    actual, _, _ = collect(workspace, "facts", counter=counter, budget=1900)
    assert actual == [workspace.state["facts"]["F1"]]
    with pytest.raises(ValueError, match="minimal budget error"):
        WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=1, native_response_budget=True)


def test_native_managed_page_arguments_keep_positions_out_of_model_contract(monkeypatch):
    calls = []
    session = native_session({"read_workspace_page": lambda **args: calls.append(args) or {"received": args}}, len, 4096)
    session["tool_lock"] = Lock()
    monkeypatch.setattr(adapter, "_ACTIVE_SESSION", session)
    assert adapter._read_workspace_page("facts")["ok"]
    assert adapter._read_workspace_page("facts", limit=1, restart=True)["ok"]
    assert calls == [{"view": "facts", "limit": 8, "restart": False},
                     {"view": "facts", "limit": 1, "restart": True}]
    assert not adapter._read_workspace_page("facts", limit=True)["ok"]
    assert not adapter._read_workspace_page("facts", restart=1)["ok"]
    assert len(calls) == 2
    fn, description = adapter._TOOLS["read_workspace_page"]
    parameters = inspect.signature(fn).parameters
    assert set(parameters) == {"view", "limit", "restart"}
    assert parameters["limit"].default == 8
    assert parameters["restart"].default is False
    assert "自动续页" in description and "无需位置或版本参数" in description
    assert "不要跨版本拼接片段" in description
