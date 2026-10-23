from copy import deepcopy
import json

import pytest

from bridge_evidence.contest.document_workspace import DocumentWorkspace
from bridge_evidence.contest.document_workspace_tools import WorkspaceTools
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest import tong_react_adapter as adapter


def tokens(value):
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def native_session(handlers, counter, budget):
    # Real dispatcher/wrapper with local synthetic handlers, without the SDK.
    return {"handlers": handlers, "tool_calls": [], "max_calls": 100,
            "tool_budget_exceeded": False, "count_tokens": counter, "max_tool_tokens": budget}


class SyntheticWorkspace:
    def __init__(self, **fields):
        self.state = {
            "version": 3, "facts_version": 1, "facts": {}, "relations": {}, "questions": {},
            "draft": None, "draft_fact_ids": [], "draft_bindings": {}, "draft_sha256": None,
            "draft_valid": False, "reads": {}, "image_reads": [], "exclusions": [], "review": None,
            "reading_progress": {"unread_ranges": [], "unverified_image_ids": [],
                                 "unverified_content": [], "review_required_evidence_ids": []},
            "identity": {"private": "must-never-be-in-prompt"}, "history": [{"large": "past-state" * 10000}],
            **fields,
        }

    def snapshot(self):
        return deepcopy(self.state)

    def delivery_status(self):
        return {"workspace_version": self.state["version"], "ready": False,
                "blockers": ["draft_missing", "semantic_review_missing"]}


def test_small_working_state_omits_huge_history_and_identity():
    workspace = SyntheticWorkspace(facts={"F1": {"id": "F1", "value": "桥名"}})
    result = WorkspaceTools(workspace, count_tokens=len).summary()
    assert result["ok"] and result["mode"] == "full"
    assert result["facts"] == workspace.state["facts"]
    assert result["delivery"] == {"ready": False, "workspace_version": 3, "draft_sha256": None,
                                  "blocker_count": 2, "blockers_view": "blockers"}
    assert "identity" not in result and "history" not in result
    assert "must-never-be-in-prompt" not in json.dumps(result)
    assert result["version"] == result["facts_version"] + 2
    result["facts"]["F1"]["value"] = "调用方改动"
    assert workspace.state["facts"]["F1"]["value"] == "桥名"


def test_large_state_has_counts_and_bounded_paging_navigation():
    workspace = SyntheticWorkspace(facts={f"F{i:04d}": {"id": f"F{i:04d}", "value": "事实" * 100} for i in range(100)})
    tools = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=2000)
    result = tools.summary()
    assert result["ok"] and result["mode"] == "paged"
    assert result["version"] == 3 and result["counts"]["facts"] == 100
    assert result["counts"]["blockers"] == 2
    assert "facts" not in result and tokens(result) <= 2000
    assert result["pagination"]["tool"] == "read_workspace_page"
    assert result["blockers"]["by_kind"] == {"draft_missing": 1, "semantic_review_missing": 1}
    assert result["reading_progress"]["counts"]["unread_ranges"] == 0
    assert result["review"] is None


def test_pages_shrink_without_skipping_or_truncating_records():
    facts = {f"F{i:03d}": {"id": f"F{i:03d}", "value": "甲😀" * 40} for i in reversed(range(12))}
    tools = WorkspaceTools(SyntheticWorkspace(facts=facts), count_tokens=len, max_tool_tokens=600)
    cursor, recovered = 0, []
    while True:
        page = tools.read_page("facts", cursor, 10)
        assert page["ok"] and tokens(page) <= 600
        assert page["start"] == cursor and page["end"] > cursor
        recovered.extend(page["items"])
        if not page["continuation"]:
            break
        assert page["continuation"]["version"] == page["version"] == 3
        cursor = page["continuation"]["arguments"]["cursor"]
    assert recovered == [facts[key] for key in sorted(facts)]


def test_single_oversize_item_starts_explicit_json_fragment_and_preserves_state():
    record = {"id": "F1", "value": "完整原文" * 500}
    workspace = SyntheticWorkspace(facts={"F1": record})
    result = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=650).read_page("facts", 0, 5)
    assert result["ok"] and result["mode"] == "canonical_json_fragment"
    assert result["item_index"] == 0 and result["item_start"] == 0
    assert result["item_end"] < result["item_total_chars"]
    assert result["continuation"]["arguments"]["expected_version"] == 3
    assert "items" not in result and workspace.state["facts"]["F1"] == record
    assert tokens(result) <= 650


@pytest.mark.parametrize("cursor,limit", [(True, 1), (0, True), (-1, 1), (0, 0), (0, -1), (1.0, 1), (0, 1.0)])
def test_page_arguments_require_strict_integers(cursor, limit):
    result = WorkspaceTools(SyntheticWorkspace(), count_tokens=len).read_page("facts", cursor, limit)
    assert result["ok"] is False and result["status"] == "error"


def test_empty_and_end_pages_and_unknown_views():
    tools = WorkspaceTools(SyntheticWorkspace(), count_tokens=len)
    page = tools.read_page("facts", 0, 10)
    assert page["items"] == [] and page["start"] == page["end"] == page["total"] == 0
    assert page["continuation"] is None
    assert tools.read_page("facts", 1, 10)["ok"] is False
    assert tools.read_page("history", 0, 10)["ok"] is False
    assert tools.read_page("identity", 0, 10)["ok"] is False


def test_all_requested_views_keep_order_and_review_categories():
    workspace = SyntheticWorkspace(
        relations={"R2": {"id": "R2"}, "R1": {"id": "R1"}}, questions={"Q1": {"id": "Q1"}},
        draft={"基本信息": {"年份": "2018", "桥名": "合成桥"}, "维修建议": [{"建议内容": "甲"}, {"建议内容": "乙"}],
               "病害明细": [{"病害描述": "丙"}]},
        draft_bindings={"病害明细.0": ["F2"], "基本信息.桥名": ["F1"]},
        reads={"E2": [[0, 2]], "E1": [[1, 3], [5, 7]]}, image_reads=["I2", "I1"],
        exclusions=[{"evidence_id": "E1", "reason": "甲"}, {"image_id": "I2", "reason": "乙"}],
        review={"passed": False, "unsupported_claims": [{"reason": "a"}], "missing_items": [{"reason": "b"}],
                "conflicts": [{"reason": "c"}], "unresolved_items": [{"reason": "d"}]},
    )
    tools = WorkspaceTools(workspace, count_tokens=len)
    assert [item["id"] for item in tools.read_page("relations", 0, 10)["items"]] == ["R1", "R2"]
    assert tools.read_page("questions", 0, 10)["items"] == [{"id": "Q1"}]
    assert tools.read_page("draft_advice", 1, 10)["items"] == [{"建议内容": "乙"}]
    assert tools.read_page("draft_defects", 0, 10)["items"] == [{"病害描述": "丙"}]
    assert tools.read_page("draft_basic", 0, 10)["items"] == [{"field": k, "value": v} for k, v in sorted(workspace.state["draft"]["基本信息"].items())]
    assert tools.read_page("draft_bindings", 0, 10)["items"] == [{"path": k, "record_ids": v} for k, v in sorted(workspace.state["draft_bindings"].items())]
    assert tools.read_page("reads", 0, 1)["items"] == [{"evidence_id": "E1", "ranges": [[1, 3], [5, 7]]}]
    assert tools.read_page("exclusions", 0, 10)["items"] == workspace.state["exclusions"]
    assert tools.read_page("image_reads", 0, 10)["items"] == ["I2", "I1"]
    assert [item["category"] for item in tools.read_page("review_findings", 0, 10)["items"]] == ["unsupported_claims", "missing_items", "conflicts", "unresolved_items"]


def test_view_version_changes_are_visible_and_summary_never_mixes_versions():
    workspace = SyntheticWorkspace(facts={"F1": {"id": "F1"}, "F2": {"id": "F2"}})
    tools = WorkspaceTools(workspace, count_tokens=len)
    first = tools.read_page("facts", 0, 1)
    workspace.state["version"] += 1
    second = tools.read_page("facts", 1, 1)
    assert first["version"] != second["version"]

    def changing_status():
        workspace.state["version"] += 1
        return {"workspace_version": workspace.state["version"], "ready": False, "blockers": []}

    workspace.delivery_status = changing_status
    result = tools.summary()
    assert result["ok"] is False and "changed" in result["error"]["message"]


def test_real_synthetic_workspace_has_paged_unread_intervals(tmp_path):
    manifest = manifest_from_text("合成正文\n未读取的第二行")
    workspace = DocumentWorkspace(manifest, tmp_path, {"model": "synthetic"})
    workspace.record_read("E000001", 0, len(manifest["units"][0]["text"]))
    tools = WorkspaceTools(workspace, count_tokens=len)
    result = tools.read_page("unread_ranges", 0, 10)
    assert result["items"] == [{"evidence_id": "E000002", "start": 0, "end": len("未读取的第二行")}]
    assert "history" not in tools.summary()


def test_large_review_keeps_compact_status_and_all_findings_are_pageable():
    review = {"passed": False, "draft_sha256": "d" * 64, "recorded_version": 3,
              "protocol": "synthetic", "unsupported_claims": [], "missing_items": [{"reason": "遗漏" * 100} for _ in range(30)],
              "conflicts": [], "unresolved_items": []}
    workspace = SyntheticWorkspace(review=review)
    tools = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=2300)
    result = tools.summary()
    assert result["ok"] and result["mode"] == "paged"
    assert result["review"]["passed"] is False and result["review"]["protocol"] == "synthetic"
    assert result["review"]["issue_counts"]["missing_items"] == 30
    assert result["review"]["findings_view"] == "review_findings"
    assert "missing_items" not in result["review"]
    page = tools.read_page("review_findings", 29, 1)
    assert page["items"] == [{"category": "missing_items", "index": 29, "finding": review["missing_items"][29]}]


def test_bounded_delivery_does_not_return_full_blockers_or_unread_ranges():
    workspace = SyntheticWorkspace()
    workspace.delivery_status = lambda: {"ready": False, "workspace_version": 3, "draft_sha256": "a" * 64,
                                         "blockers": [f"invalidated_record:F{i}" for i in range(1000)],
                                         "reading_progress": {"unread_ranges": ["x"] * 1000}}
    result = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=500).bounded_delivery()
    assert result["ready"] is False and result["blocker_count"] == 1000
    assert result["blockers_view"] == "blockers" and tokens(result) <= 500
    assert "blockers" not in result and "reading_progress" not in result


def test_smallest_accepted_budget_never_emits_an_oversize_error():
    for budget in range(100, 600):
        try:
            tools = WorkspaceTools(SyntheticWorkspace(), count_tokens=len, max_tool_tokens=budget)
        except ValueError:
            continue
        for result in (tools.summary(), tools.read_page("facts", True, 1), tools.read_page("facts", 0, 1)):
            assert tokens(result) <= budget
        break
    else:
        pytest.fail("No accepted budget")


def test_native_summary_near_payload_limit_returns_readable_paged_status():
    workspace = SyntheticWorkspace(facts={"F1": {"id": "F1", "value": ""}})
    size = tokens(WorkspaceTools(workspace, count_tokens=len).summary())
    workspace.state["facts"]["F1"]["value"] = "甲" * (4096 - size)
    before = workspace.snapshot()
    bare = WorkspaceTools(workspace, count_tokens=len).summary()
    assert tokens(bare) == 4096 and bare["mode"] == "full"
    assert tokens(adapter.wrap_tool_response("read_workspace", bare)) > 4096

    tools = WorkspaceTools(workspace, count_tokens=len, native_response_budget=True)
    result = adapter._invoke_active(native_session({"read_workspace": tools.summary}, len, 4096),
                                    "read_workspace", {})
    assert result["ok"] and result["result"]["mode"] == "paged"
    assert result["result"]["version"] == before["version"]
    assert result["result"]["counts"]["facts"] == 1
    assert tokens(result) <= 4096 and workspace.snapshot() == before


@pytest.mark.parametrize("counter", [len, lambda text: len(text.encode("utf-8"))])
def test_native_pages_shrink_for_complete_wrapper_and_recover_every_record(counter):
    facts = {f"F{i:03d}": {"id": f"F{i:03d}", "value": '原文😀带"引号"' * 35} for i in range(6)}
    workspace = SyntheticWorkspace(facts=facts)
    before = workspace.snapshot()
    bare = WorkspaceTools(workspace, count_tokens=counter, max_tool_tokens=100000).read_page("facts", 0, 2)
    budget = counter(adapter._json_bytes(bare).decode())
    assert counter(adapter._json_bytes(adapter.wrap_tool_response("read_workspace_page", bare)).decode()) > budget
    tools = WorkspaceTools(workspace, count_tokens=counter, max_tool_tokens=budget, native_response_budget=True)
    session = native_session({"read_workspace_page": tools.read_page_managed}, counter, budget)
    call = {"view": "facts", "limit": 2, "restart": False}
    received = []
    while call:
        result = adapter._invoke_active(session, "read_workspace_page", call)
        assert result["ok"], result
        assert counter(adapter._json_bytes(result).decode()) <= budget
        page = result["result"]
        assert page["version"] == before["version"] and page["start"] == len(received)
        if not received:
            assert len(page["items"]) == 1  # Two bare records fit; two wrapped records do not.
        received.extend(page["items"])
        call = page["continuation"]["arguments"] if page["continuation"] else None
        assert call is None or set(call) == {"view", "limit", "restart"}
    assert received == [facts[key] for key in sorted(facts)]
    assert workspace.snapshot() == before


@pytest.mark.parametrize("name", ["read_workspace", "update_workspace", "review_draft"])
def test_summary_uses_exact_authorized_caller_envelope(name):
    penalties = {"read_workspace": 101, "update_workspace": 701, "review_draft": 503}

    def counter(text):
        return len(text) + penalties.get(json.loads(text).get("tool"), 0)

    workspace = SyntheticWorkspace(facts={"F1": {"id": "F1", "value": "甲" * 3000}})
    bare = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=100000).summary()
    budget = counter(adapter._json_bytes(adapter.wrap_tool_response(name, bare)).decode()) - 1
    tools = WorkspaceTools(workspace, count_tokens=counter, max_tool_tokens=budget, native_response_budget=True)
    session = native_session({name: lambda **kwargs: tools.summary(tool_name=name)}, counter, budget)
    args = {"update": {}} if name == "update_workspace" else {}
    result = adapter._invoke_active(session, name, args)
    assert result["ok"] and result["result"]["mode"] == "paged"
    assert counter(adapter._json_bytes(result).decode()) <= budget
    assert result["result"]["version"] == workspace.state["version"]


def test_real_committed_update_returns_current_native_summary_and_preserves_full_disk_draft(tmp_path):
    text = "合成桥；" + "合成裂缝记录" * 600
    manifest = manifest_from_text(text)
    identity = {"model": "synthetic", "run": "native-budget"}

    def create(directory):
        workspace = DocumentWorkspace(manifest, directory, identity)
        workspace.record_read("E000001", 0, len(text))
        return workspace

    preview = create(tmp_path / "preview")
    workspace = create(tmp_path / "actual")
    value = "合成裂缝记录" * 600
    draft = {"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [], "病害明细": [{"病害描述": value}]}
    update = {"base_version": workspace.snapshot()["version"],
              "facts": [{"id": "F1", "value": value, "status": "explicit",
                         "source_refs": [{"evidence_id": "E000001", "start": 0, "end": len(text)}]}],
              "draft": draft, "draft_fact_ids": ["F1"],
              "draft_bindings": {"基本信息.桥梁名称": ["F1"], "病害明细.0": ["F1"]}}
    preview.apply_update(update)
    bare = WorkspaceTools(preview, count_tokens=len, max_tool_tokens=100000).summary()
    budget = tokens(bare)
    assert tokens(adapter.wrap_tool_response("update_workspace", bare)) > budget
    tools = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=budget, native_response_budget=True)

    def update_handler(update_json):
        workspace.apply_update(json.loads(update_json))
        return tools.summary(tool_name="update_workspace")

    session = native_session({"update_workspace": update_handler}, len, budget)
    reply = adapter._invoke_active(session, "update_workspace", {"update": update})
    after = workspace.snapshot()
    assert reply["ok"] and reply["result"]["mode"] == "paged"
    assert reply["result"]["version"] == after["version"] == update["base_version"] + 1
    assert reply["result"]["draft_sha256"] == after["draft_sha256"]
    assert tokens(reply) <= budget and after["draft"] == draft
    on_disk = workspace.checkpoint_path.read_bytes()
    for view in ("facts", "draft_basic", "draft_defects"):
        result = tools.read_page(view, 0, 10)
        assert result["ok"]
    assert workspace.checkpoint_path.read_bytes() == on_disk
    restored = DocumentWorkspace(manifest, tmp_path / "actual", identity)
    assert restored.snapshot() == after  # No full value/history lost to prompt paging.


def test_native_oversize_record_is_labeled_json_transport_not_partial_fact():
    record = {"id": "F1", "value": "不可截断" * 500}
    workspace = SyntheticWorkspace(facts={"F1": record})
    before = workspace.snapshot()
    tools = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=650, native_response_budget=True)
    reply = adapter._invoke_active(native_session({"read_workspace_page": tools.read_page_managed}, len, 650),
                                    "read_workspace_page", {"view": "facts", "limit": 1, "restart": False})
    assert reply["ok"] and reply["result"]["mode"] == "canonical_json_fragment"
    assert reply["result"]["fragment_is_complete_record"] is False and "items" not in reply["result"]
    assert reply["result"]["version"] == before["version"]
    assert tokens(reply) <= 650 and workspace.snapshot() == before


def test_native_bounded_delivery_counts_finish_envelope():
    workspace = SyntheticWorkspace()
    workspace.delivery_status = lambda: {"ready": False, "workspace_version": 3,
                                         "draft_sha256": "a" * 64, "blockers": ["missing"]}

    def counter(text):
        obj = json.loads(text)
        return len(text) + (300 if obj.get("tool") == "finish" and "ready" in obj.get("result", {}) else 0)

    bare = WorkspaceTools(workspace, count_tokens=len).bounded_delivery()
    budget = counter(adapter._json_bytes(adapter.wrap_tool_response("finish", bare)).decode())
    for limit in (budget - 1, budget):
        tools = WorkspaceTools(workspace, count_tokens=counter, max_tool_tokens=limit, native_response_budget=True)
        reply = adapter._invoke_active(native_session({"finish": tools.bounded_delivery}, counter, limit), "finish", {})
        if limit == budget:
            assert reply["result"] == bare and counter(adapter._json_bytes(reply).decode()) == budget
        else:
            assert not reply["ok"] and reply["result"]["status"] == "budget_exceeded"


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_native_response_budget_requires_boolean(value):
    with pytest.raises(ValueError, match="native_response_budget"):
        WorkspaceTools(SyntheticWorkspace(), count_tokens=len, native_response_budget=value)


@pytest.mark.parametrize("name", [None, [], "finish", "history", "read_workspace_page", "private_arbitrary_name"])
def test_summary_caller_is_a_fixed_host_only_choice(name):
    with pytest.raises(ValueError, match="Summary caller"):
        WorkspaceTools(SyntheticWorkspace(), count_tokens=len).summary(tool_name=name)
