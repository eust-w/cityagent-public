"""Whole-draft semantic review lookup, using in-memory model protocol doubles."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from bridge_evidence.contest import document_workflow as module
from bridge_evidence.contest.run_state import json_digest
from test_document_workflow import FakeModel, workflow, read, prepare, source_fact


def defect(value):
    return {"病害部位": value, "病害类型": "裂缝", "病害描述": "裂缝一条，长2m", "是否新增": None,
            "上一次定检状态": None, "发展程度": None}


def seed(wf, extra_rows=0):
    read(wf)
    rows = [defect("左幅1号梁")]
    rows.extend(defect("其他构件" + str(i) + "长描述" * 40) for i in range(extra_rows))
    prepare(wf, draft={"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [], "病害明细": rows},
            draft_bindings={"基本信息.桥梁名称": ["name"], **{f"病害明细.{i}": ["name"] for i in range(len(rows))}})


def reverse(wf):
    unit = wf.manifest["units"][-1]
    return {"direction": "REVERSE", "output_items": [], "related_output": [],
            "source_spans": [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}],
            "source": unit["text"], "document_background": {"桥梁名称": "合成桥"}}


def decision(payload, **kwargs):
    return {"passed": True, "checked_output_paths": [], "checked_evidence_spans": payload["source_spans"],
            "unsupported_claims": [], "missing_items": [], "conflicts": [], "unresolved_items": [], **kwargs}


class ToolModel(FakeModel):
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy
        self.requests = []
        self.max_model_len = 3000

    def _request(self, path, payload, *, kind):
        assert kind == "generation"
        self.requests.append(deepcopy(payload))
        receipt = {"call_id": f"lookup-{len(self.requests)}", "kind": kind, "status": "success"}
        self.calls.append(receipt)
        message, reason = self.strategy(payload)
        return {"choices": [{"finish_reason": reason, "message": message}],
                "usage": {"prompt_tokens": self._tokenize(payload), "completion_tokens": 30}}, receipt


def tool(name="review_search_draft", arguments=None):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": "lookup-call", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments or {"query": "左幅1号梁", "cursor": 0, "limit": 5}, ensure_ascii=False)}}]}


def configured(tmp_path, strategy):
    client = ToolModel(strategy)
    settings = module.DocumentSettings(max_model_len=3000, output_tokens=300, tool_tokens=650,
                                       history_reserve_tokens=100, max_model_calls=30)
    wf = workflow(tmp_path, client=client, settings=settings, text="桥梁名称：合成桥\n正文左幅1号梁裂缝一条，长2m。\n结论重复记录同一裂缝。")
    seed(wf, 180)
    wf._active_review_items = wf._output_items(wf.workspace.snapshot())
    return wf


def test_duplicate_mention_can_see_existing_draft_without_same_source_id(tmp_path):
    observed = []
    def decide(payload):
        if payload["direction"] == "REVERSE":
            # All rows bind only the bridge-name/body fact, not this repeated
            # conclusion. Whole draft values remain visible for comparison.
            if any(s["evidence_id"] == wf.manifest["units"][-1]["id"] for s in payload["source_spans"]):
                observed.append(payload)
                assert payload["related_output"] == []
                assert payload["draft_catalog"]["mode"] == "complete_values"
                assert any(item["path"] == "病害明细.0" and item["value"]["病害部位"] == "左幅1号梁"
                           for item in payload["draft_catalog"]["items"])
        return {}
    wf = workflow(tmp_path, client=FakeModel(decide),
                  text="桥梁名称：合成桥\n正文左幅1号梁裂缝一条，长2m。\n" + "分隔段落。" * 4000 + "\n结论重复记录同一裂缝。")
    seed(wf)
    result = wf._perform_review_draft()
    assert observed
    assert result["review"]["missing_items"] == []
    assert len(wf.workspace.snapshot()["draft"]["病害明细"]) == 1


def test_paged_reviewer_searches_whole_frozen_draft_and_counts_both_calls(tmp_path):
    def strategy(request):
        payload = json.loads(request["messages"][1]["content"])
        if request["messages"][-1]["role"] != "tool":
            assert payload["draft_catalog"]["mode"] == "paged"
            assert payload["related_output"] == []
            assert request["tools"]
            return tool(), "tool_calls"
        found = json.loads(request["messages"][-1]["content"])
        assert found["items"][0]["path"] == "病害明细.0"
        assert found["items"][0]["value"]["病害部位"] == "左幅1号梁"
        return {"role": "assistant", "content": json.dumps(decision(payload), ensure_ascii=False)}, "stop"
    wf = configured(tmp_path, strategy)
    result = wf._review_request(reverse(wf))
    assert result["model_call_ids"] == ["lookup-1", "lookup-2"]
    assert len(wf.client.calls) == 2
    assert len(result["draft_lookup"]["tool_calls"]) == 1
    assert all(d["status"] == "settled" for d in wf.model_dispatches)


@pytest.mark.parametrize("unresolved", [False, True])
def test_paged_omission_claim_requires_draft_lookup_or_explicit_unknown(tmp_path, unresolved):
    def strategy(request):
        payload = json.loads(request["messages"][1]["content"])
        category = "unresolved_items" if unresolved else "missing_items"
        answer = decision(payload, passed=False, **{category: [{"reason": "未能完成全稿核对"}]})
        return {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}, "stop"
    wf = configured(tmp_path, strategy)
    if unresolved:
        result = wf._review_request(reverse(wf))
        assert result["missing_items"] == [] and result["unresolved_items"]
    else:
        with pytest.raises(ValueError, match="omission without reading"):
            wf._review_request(reverse(wf))
    # A native final is immutable; it cannot be silently replaced by a second
    # direct generation after the returned SDK boundary.
    assert len(wf.client.calls) == 1


def test_review_tool_cannot_mutate_workspace(tmp_path):
    def strategy(request):
        payload = json.loads(request["messages"][1]["content"])
        if request["messages"][-1]["role"] != "tool":
            return tool("update_workspace", {"update_json": "{}"}), "tool_calls"
        assert json.loads(request["messages"][-1]["content"])["ok"] is False
        return {"role": "assistant", "content": json.dumps(decision(payload, passed=False,
            unresolved_items=[{"reason": "无可用修改工具"}]), ensure_ascii=False)}, "stop"
    wf = configured(tmp_path, strategy)
    before = json_digest(wf.workspace.snapshot())
    result = wf._review_request(reverse(wf))
    assert result["unresolved_items"]
    assert json_digest(wf.workspace.snapshot()) == before


def test_native_tool_limit_stops_before_executing_extra_lookup(tmp_path):
    def strategy(_request):
        message = tool()
        second = deepcopy(message["tool_calls"][0])
        second.update(id="second")
        second["function"]["arguments"] = json.dumps({"query": "other", "cursor": 0, "limit": 5})
        message["tool_calls"].append(second)
        return message, "tool_calls"
    wf = configured(tmp_path, strategy)
    wf.settings = replace(wf.settings, review_iterations=1)
    with pytest.raises(ValueError, match="tool budget"):
        wf._review_request(reverse(wf))
    assert len(wf.client.calls) == 1
    assert all(d["status"] == "settled" for d in wf.model_dispatches)


def test_new_semantic_state_keeps_old_partial_without_blocking_same_source_batch(tmp_path):
    def strategy(request):
        payload = json.loads(request["messages"][1]["content"])
        return {"role": "assistant", "content": json.dumps(decision(payload), ensure_ascii=False)}, "stop"
    wf = configured(tmp_path, strategy)
    payload = reverse(wf)
    first = wf._review_request(payload)
    old_paths = list((tmp_path / "semantic_review" / "partial").glob("*.json"))
    assert len(old_paths) == 1
    old_bytes = old_paths[0].read_bytes()
    wf.workspace.apply_update({"base_version": wf.workspace.snapshot()["version"],
        "questions": [{"id": "q-new", "question": "新增的跨段关联疑问", "status": "open"}]})
    second = wf._review_request(payload)
    assert second["model_call_ids"] != first["model_call_ids"]
    assert len(wf.client.calls) == 2
    assert len(list((tmp_path / "semantic_review" / "partial").glob("*.json"))) == 2
    assert old_paths[0].read_bytes() == old_bytes


def test_pending_final_response_is_consumed_at_model_call_limit_without_new_generation(tmp_path, monkeypatch):
    from bridge_evidence.contest.document_review_checkpoint import NativeReviewCheckpoint
    settings = module.DocumentSettings(max_model_calls=1)
    wf = workflow(tmp_path, settings=settings)
    seed(wf)
    payload = reverse(wf)
    wf._active_review_items = wf._output_items(wf.workspace.snapshot())

    def interrupt_before_decision(*_args, **_kwargs):
        raise RuntimeError("synthetic crash after final model response")

    with monkeypatch.context() as patch:
        patch.setattr(wf, "_native_final_decision", interrupt_before_decision)
        with pytest.raises(RuntimeError, match="after final model response"):
            wf._review_request(payload)
    assert len(wf.client.calls) == 1
    checkpoint = wf._partial_checkpoint(payload, json_digest(wf._active_review_items))
    assert isinstance(checkpoint, NativeReviewCheckpoint)
    assert checkpoint.load()["status"] == "returned"
    restored = workflow(tmp_path, settings=settings)
    restored._active_review_items = restored._output_items(restored.workspace.snapshot())
    result = restored._review_request(payload)
    assert result["passed"]
    assert restored.client.calls == []
    assert result["model_call_ids"] == [wf.client.calls[0]["call_id"]]
    assert result["model_call_id"] == wf.client.calls[0]["call_id"]
