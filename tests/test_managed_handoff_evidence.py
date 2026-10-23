"""Synthetic handoff evidence must survive the model-facing managed pager.

Counts use deterministic JSON character lengths, not model token estimates.
No model requests are made by these regressions.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from bridge_evidence.contest import document_workspace_tools as module
from bridge_evidence.contest.document_merge_progress import MergeWorkspaceTools
from bridge_evidence.contest.tong_react_adapter import wrap_tool_response
from test_managed_workspace_navigation import Workspace, encoded
from test_managed_workspace_replay import invocation


@pytest.fixture
def handoff_fixture():
    return json.loads((Path(__file__).parent / "fixtures" /
                       "synthetic_managed_handoff.json").read_text())


def setup(handoff_fixture, *, budget=100_000, only_first=False):
    page_input = handoff_fixture["workspace_page_input"]
    records = deepcopy(page_input["items"][:1] if only_first else page_input["items"])
    workspace = Workspace(identity="synthetic-handoff-document", version=page_input["version"],
                          facts={item["id"]: item for item in records})
    tools = MergeWorkspaceTools(workspace, count_tokens=len, max_tool_tokens=budget,
                                native_response_budget=True,
                                handoff_evidence=handoff_fixture["handoff_evidence"])
    return workspace, tools


@pytest.mark.parametrize("view", ["facts", "unbound_facts"])
def test_synthetic_managed_page_delivers_verbatim_handoff_projection(handoff_fixture, view):
    original = handoff_fixture["workspace_page_input"]
    assert original["version"] == 17 and len(original["items"]) == 10
    assert all("handoff_evidence" not in item for item in original["items"])
    workspace, tools = setup(handoff_fixture)
    before = workspace.snapshot()

    # Exercise the model-facing interface, including its native response
    # envelope, rather than only the lower-level read_page projection.
    page = tools.read_page_managed(view, limit=handoff_fixture["arguments"]["limit"])
    assert page["ok"] and page["version"] == original["version"]
    assert len(page["items"]) == len(original["items"])
    for item, original_item in zip(page["items"], original["items"]):
        expected = handoff_fixture["handoff_evidence"][item["id"]]
        assert item == {**original_item, "handoff_evidence": expected}
        assert expected["source_refs"] == item["source_refs"]
        assert expected["candidate_value"] == item["value"]
        assert expected["source_excerpts"]
        assert expected["excerpts_are_verbatim_not_model_summary"] is True
        assert expected["semantic_approval"] is False
    assert workspace.snapshot() == before
    assert len(encoded(wrap_tool_response("read_workspace_page", page))) <= tools.max_tool_tokens


def test_synthetic_handoff_fragment_budget_durable_replay_and_fresh_version(handoff_fixture, monkeypatch):
    # Deliberately long fictional source excerpts exceed this small accounting
    # budget. Include the native envelope in every assertion.
    budget = 1100
    workspace, tools = setup(handoff_fixture, budget=budget, only_first=True)
    original = deepcopy(next(iter(workspace.state["facts"].values())))
    evidence = deepcopy(handoff_fixture["handoff_evidence"][original["id"]])
    expected = {**original, "handoff_evidence": evidence}
    active = {"identity": None}
    monkeypatch.setattr(module, "_current_call_identity", lambda: deepcopy(active["identity"]))
    fragments, offset = [], 0
    for index in range(200):
        active["identity"] = invocation(f"g{index}", f"call{index}")
        page = tools.read_page_managed("facts", limit=1)
        assert page["ok"] and page["mode"] == "canonical_json_fragment"
        assert page["item_start"] == offset and page["item_end"] > offset
        assert len(encoded(wrap_tool_response("read_workspace_page", page))) <= budget
        if page["continuation"]:
            assert page["continuation"]["arguments"] == {"view": "facts", "limit": 1, "restart": False}
        fragments.append(page["fragment"])
        offset = page["item_end"]
        checkpoint = tools.managed_navigation_snapshot()
        resumed = MergeWorkspaceTools(workspace, count_tokens=len, max_tool_tokens=budget,
                                       native_response_budget=True,
                                       handoff_evidence=handoff_fixture["handoff_evidence"])
        resumed.configure_managed_navigation(checkpoint)
        assert resumed.read_page_managed("facts", limit=1) == {**page, "replayed": True}
        assert resumed.managed_navigation_snapshot() == checkpoint
        tools = resumed
        if page["continuation"] is None:
            break
    else:
        pytest.fail("Handoff evidence fragment pagination did not finish")
    recovered = "".join(fragments)
    assert len(fragments) > 1
    assert json.loads(recovered) == expected
    assert hashlib.sha256(recovered.encode()).hexdigest() == page["item_sha256"]
    assert "handoff_evidence" not in workspace.snapshot()["facts"][original["id"]]

    # An original invocation still replays its original version; a fresh one
    # restarts and must not mislabel a revised candidate with stale evidence.
    workspace.state["version"] += 1
    workspace.state["facts"][original["id"]]["value"] = "汇总模型已修订"
    assert tools.read_page_managed("facts", limit=1) == {**page, "replayed": True}
    active["identity"] = invocation("new-version", "new-call")
    tools.max_tool_tokens = 10_000
    revised = tools.read_page_managed("facts", limit=1)
    assert revised["ok"] and revised["start"] == 0
    assert revised["navigation"]["version_changed"]
    assert revised["version"] == original["record_version"] + 1
    assert revised["items"][0]["value"] == "汇总模型已修订"
    assert "handoff_evidence" not in revised["items"][0]
