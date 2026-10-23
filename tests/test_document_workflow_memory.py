"""Bounded workspace handlers through the real workflow, using synthetic state.

The SDK context is an in-memory double and the review gateway forwards only to
FakeModel below. No HTTP listener, SDK process, real report or model is used.
"""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from bridge_evidence.contest import document_workflow as module
from bridge_evidence.contest import tong_react_adapter as adapter
from bridge_evidence.contest.document_model_gateway import AuditedModelGateway
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from test_document_workflow import install_native_review_driver


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class FakeModel:
    model = "synthetic-memory-model"
    base_url = "http://never-contacted.invalid"
    max_model_len = 65536
    timeout = 180.0

    def __init__(self, findings=None):
        self.calls = []
        self.payloads = []
        self.findings = findings or []

    @staticmethod
    def count_tokens(text):
        return max(1, (len(text) + 3) // 4)

    def count_chat_tokens(self, messages):
        return self.count_tokens(compact(messages)) + 12

    def _tokenize(self, payload):
        return self.count_chat_tokens(payload["messages"])

    def _request(self, path, payload, *, kind):
        assert path == "/v1/chat/completions" and kind == "generation"
        review_input = json.loads(payload["messages"][-1]["content"])
        self.payloads.append(deepcopy(review_input))
        findings = deepcopy(self.findings) if review_input["direction"] == "FORWARD" else []
        decision = {"passed": not findings,
                    "checked_output_paths": [item["path"] for item in review_input.get("output_items", [])],
                    "checked_evidence_spans": deepcopy(review_input.get("source_spans", [])),
                    "unsupported_claims": findings, "missing_items": [], "conflicts": [], "unresolved_items": []}
        content = compact(decision)
        assert self.count_tokens(content) <= payload["max_tokens"]
        receipt = {"kind": kind, "call_id": f"synthetic-review-{len(self.calls) + 1}", "status": "success"}
        self.calls.append(receipt)
        return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": self._tokenize(payload), "completion_tokens": self.count_tokens(content)}}, receipt


@pytest.fixture
def no_network_gateway(monkeypatch):
    class SDKContext:
        def __init__(self, client, **kwargs):
            self.client, self.options = client, kwargs
            self.url = "http://in-memory-gateway.invalid/v1"
            self.receipts = []
            self.on_success = None
            self.on_receipt = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def factory(client, **kwargs):
        if kwargs.get("request_kind") == "generation":
            # forward() is synchronous against FakeModel; never enter() this
            # real gateway, so no socket or server is created.
            return AuditedModelGateway(client, **kwargs)
        return SDKContext(client, **kwargs)

    monkeypatch.setattr(module, "AuditedModelGateway", factory)


class _CheckedHandlers(Exception):
    """Stop the synthetic run once assertions have inspected actual handlers."""


def exercise_handlers(wf, assertion):
    def runner(_prompt, handlers, _config):
        assertion(handlers)
        raise _CheckedHandlers()

    wf.react_runner = runner
    with pytest.raises(module.DocumentWorkflowFailure) as caught:
        wf.run()
    # Do not let the workflow's error wrapper hide an assertion failure.
    if not isinstance(caught.value.__cause__, _CheckedHandlers):
        raise caught.value.__cause__


def make_workflow(tmp_path, *, text="桥梁名称：合成桥。\n", client=None, max_sessions=30):
    return install_native_review_driver(module.DocumentWorkflow(
        manifest_from_text(text), client or FakeModel(), {"model": FakeModel.model}, tmp_path,
        settings=module.DocumentSettings(tool_tokens=768, max_sessions=max_sessions, max_model_calls=30),
        react_runner=lambda *_args: None,
    ))


def seed_large_state(wf, count=28):
    unit = wf.manifest["units"][0]
    wf.workspace.record_read(unit["id"], 0, len(unit["text"]))
    facts = [{"id": f"F{index:03d}", "value": f"第{index}条完整内容：" + "合成详细描述😀" * 12,
              "status": "explicit", "source_refs": [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}]}
             for index in range(count)]
    rows = [{"病害部位": f"构件{index}", "病害描述": item["value"]} for index, item in enumerate(facts)]
    advice = [{"建议类别": "原文建议", "建议内容": "合成完整建议" * 15, "病害部位": f"构件{index}"}
              for index in range(12)]
    bindings = {"基本信息.桥梁名称": ["F000"],
                **{f"病害明细.{index}": [item["id"]] for index, item in enumerate(facts)},
                **{f"维修建议.{index}": ["F000"] for index in range(len(advice))}}
    wf.workspace.apply_update({
        "base_version": wf.workspace.snapshot()["version"], "facts": facts,
        "draft": {"基本信息": {"桥梁名称": "合成桥"}, "维修建议": advice, "病害明细": rows},
        "draft_fact_ids": [item["id"] for item in facts], "draft_bindings": bindings,
        "questions": [{"id": "Q1", "question": "合成问题", "status": "unknown",
                       "reason": "HISTORY_ONLY_SENTINEL" * 400}],
    })
    wf.workspace.apply_update({"base_version": wf.workspace.snapshot()["version"],
                               "questions": [{"id": "Q1", "reason": "当前只保留这个说明"}]})
    state = wf.workspace.snapshot()
    assert wf.client.count_tokens(compact(state)) > wf.settings.tool_tokens * 3
    assert "HISTORY_ONLY_SENTINEL" in compact(state["history"])
    return state


def assert_bounded(wf, result):
    rendered = compact(result)
    assert wf.client.count_tokens(rendered) <= wf.settings.tool_tokens
    assert "history" not in result and "identity" not in result
    assert "HISTORY_ONLY_SENTINEL" not in rendered


def collect_pages(wf, handlers, view):
    cursor, result, versions = 0, [], set()
    while True:
        page = handlers["read_workspace_page"](view=view, limit=50, restart=False)
        assert page["ok"], page
        assert_bounded(wf, page)
        assert page["start"] == cursor
        versions.add(page["version"])
        result.extend(page["items"])
        if page["continuation"] is None:
            assert page["end"] == page["total"] == len(result)
            break
        assert "cursor" not in page["continuation"]["arguments"]
        next_cursor = page["end"]
        assert next_cursor > cursor
        cursor = next_cursor
    assert len(versions) == 1
    return result


def test_large_read_and_update_handlers_return_bounded_state_without_history(tmp_path, no_network_gateway):
    wf = make_workflow(tmp_path)
    seed_large_state(wf)

    def check(handlers):
        before = handlers["read_workspace"]()
        assert before["ok"] and before["mode"] == "paged"
        assert_bounded(wf, before)
        updated = handlers["update_workspace"](update_json=compact({
            "base_version": before["version"],
            "questions": [{"id": "Q1", "reason": "更新后的完整说明"}],
            "draft_patch": [{"op": "append_row", "section": "病害明细",
                             "row": {"病害部位": "新构件", "病害描述": "新记录"}, "fact_ids": ["F001"]}],
        }))
        assert updated["ok"] and updated["mode"] == "paged"
        assert updated["version"] > before["version"]
        assert_bounded(wf, updated)
        assert updated["counts"]["draft_defects"] == before["counts"]["draft_defects"] + 1
        assert wf.workspace.snapshot()["questions"]["Q1"]["reason"] == "更新后的完整说明"

    exercise_handlers(wf, check)


def test_workspace_page_handlers_reconstruct_all_facts_and_draft_rows(tmp_path, no_network_gateway):
    wf = make_workflow(tmp_path)
    state = seed_large_state(wf)

    def check(handlers):
        assert collect_pages(wf, handlers, "facts") == [state["facts"][key] for key in sorted(state["facts"])]
        assert collect_pages(wf, handlers, "draft_basic") == [{"field": "桥梁名称", "value": "合成桥"}]
        assert collect_pages(wf, handlers, "draft_advice") == state["draft"]["维修建议"]
        assert collect_pages(wf, handlers, "draft_defects") == state["draft"]["病害明细"]
        assert collect_pages(wf, handlers, "draft_bindings") == [
            {"path": key, "record_ids": value} for key, value in sorted(state["draft_bindings"].items())]

    exercise_handlers(wf, check)


def test_review_handler_bounds_large_findings_but_preserves_all_on_disk_and_pages(tmp_path, no_network_gateway):
    findings = [{"path": "基本信息.桥梁名称", "reason": f"待核对事项{index}：" + "合成复核说明" * 20,
                 "source_refs": ["E000001"]} for index in range(30)]
    wf = make_workflow(tmp_path, client=FakeModel(findings))
    seed_large_state(wf)
    # Review runs only after a returned reader stage; inspect its bounded
    # summary and managed pages separately from queue handoff acceptance.
    review_result = wf._perform_review_draft()

    def check(handlers):
        result = review_result
        assert result["ok"] and result["mode"] == "paged"
        assert_bounded(wf, result)
        assert result["review"]["passed"] is False and result["review"]["findings_view"] == "review_findings"
        assert "unsupported_claims" not in result["review"]
        saved = wf.workspace.snapshot()["review"]["unsupported_claims"]
        assert saved and all(item in findings for item in saved)
        assert result["review"]["issue_counts"]["unsupported_claims"] == len(saved)
        paged = collect_pages(wf, handlers, "review_findings")
        assert [item["finding"] for item in paged] == saved
        assert {item["category"] for item in paged} == {"unsupported_claims"}
        assert wf.client.calls and {payload["direction"] for payload in wf.client.payloads} >= {"FORWARD", "REVERSE"}

    exercise_handlers(wf, check)


def test_failed_finish_returns_only_status_and_navigation_not_full_unread_ranges(tmp_path, no_network_gateway):
    wf = make_workflow(tmp_path, text="\n".join(f"第{i}段合成报告正文" for i in range(120)))
    seed_large_state(wf)
    expected_unread = wf.workspace.snapshot()["reading_progress"]["unread_ranges"]
    assert len(expected_unread) == 119

    def check(handlers):
        result = handlers["finish"]()
        assert result["ok"] is False and result["ready"] is False
        assert_bounded(wf, result)
        assert result["next_action"] == {"tool": "read_source", "arguments": {"evidence_id": "E000002"}}
        assert "reading_progress" not in result and "unread_ranges" not in compact(result)
        assert "unread_source_ranges" in collect_pages(wf, handlers, "blockers")
        assert collect_pages(wf, handlers, "unread_ranges") == expected_unread

    exercise_handlers(wf, check)


@pytest.mark.parametrize("navigation_fallback", [False, True])
def test_next_session_can_repair_rejected_update_from_bounded_safe_feedback(
        tmp_path, no_network_gateway, navigation_fallback):
    class RecoveryModel(FakeModel):
        def count_chat_tokens(self, messages):
            # Exercise the real recovery fallback without requiring a huge
            # document or changing the tokenizer for any dispatched request.
            if navigation_fallback and '"counts":' in messages[-1]["content"]:
                return self.max_model_len
            return super().count_chat_tokens(messages)

    wf = make_workflow(tmp_path, client=RecoveryModel(), max_sessions=2)
    before = seed_large_state(wf)
    unit = wf.manifest["units"][0]
    refs = [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}]
    secret = "PRIVATE_FAILED_UPDATE_VALUE" * 100
    sdk_error = {"type": "ToolErrorAction", "tool_name": "update_workspace",
                 "code": "SDK_TOOL_ARGUMENT_JSON", "action": "Regenerate valid tool-argument JSON.",
                 "event_sha256": "synthetic-event-hash"}
    calls, prompts = [], []

    def runner(prompt, handlers, config):
        prompts.append(prompt)
        native = {"handlers": handlers, "tool_calls": [], "max_calls": 12,
                  "count_tokens": wf.client.count_tokens, "max_tool_tokens": wf.settings.tool_tokens}
        if len(prompts) == 1:
            failures = [
                ({"base_version": before["version"], "facts": [
                    {"id": "F_OLD_ERROR", "value": secret, "status": "explicit",
                     "source_refs": refs, "depends_on": ["UNDECLARED_FACT"]}]}, "DEPENDENCY_INVALID"),
                ({"facts": [{"id": "F_NEW", "value": secret, "source_refs": refs}]}, "WORKSPACE_STALE"),
                ({"base_version": before["version"], "facts": [
                    {"id": "F_NEW", "value": secret, "source_refs": refs}]}, "FACT_FIELDS"),
            ]
            for update, code in failures:
                result = adapter._invoke_active(native, "update_workspace", {"update": update})
                assert result["ok"] is False and result["error"]["code"] == code
                # Actual validation rejected the update atomically; no partial
                # fact from a failed call should become a future dependency.
                assert wf.workspace.snapshot() == before
            # Extra receipt diagnostics and old workspace history must never
            # become the next model's recovery prompt.
            native["tool_calls"][-1]["error"]["details"] = {"private": "PRIVATE_DIAGNOSTIC"}
            native["tool_calls"][-1]["result_summary"] = "PRIVATE_RESULT_SUMMARY"
            calls.extend(deepcopy(native["tool_calls"]))
            return {"status": "failed", "failure_reason": "tool_error", "native_final_event": False,
                    "tool_calls": native["tool_calls"], "sdk_tool_errors": [sdk_error]}

        recovery = json.loads(prompt.splitlines()[-1])
        repairs = recovery["previous_handler_tool_errors"]
        assert len(repairs) == 2
        assert [item["code"] for item in repairs] == ["WORKSPACE_STALE", "FACT_FIELDS"]
        assert all(item["name"] == "update_workspace" for item in repairs)
        assert "read_workspace" in repairs[0]["action"] and "base_version" in repairs[0]["fields"]
        assert "status" in repairs[1]["fields"] and "explicit" in repairs[1]["action"]
        assert all(set(item) == {"name", "code", "action", "fields"} for item in repairs)
        assert len(compact(repairs).encode("utf-8")) <= 4099
        assert recovery["previous_sdk_tool_errors"] == [sdk_error]
        for private in ("PRIVATE_FAILED_UPDATE_VALUE", "PRIVATE_DIAGNOSTIC", "PRIVATE_RESULT_SUMMARY",
                        "HISTORY_ONLY_SENTINEL", "UNDECLARED_FACT", "DEPENDENCY_INVALID"):
            assert private not in prompt
        assert ("通过 read_outline 导航" in prompt) is navigation_fallback
        # Follow the actual recovery actions using the real adapter and
        # workflow handlers, rather than asserting only a prompt string.
        current = adapter._invoke_active(native, "read_workspace", {})
        assert current["ok"]
        fixed = adapter._invoke_active(native, "update_workspace", {"update": {
            "base_version": current["result"]["version"],
            "facts": [{"id": "F_NEW", "value": "合成桥", "source_refs": refs, "status": "explicit"}],
        }})
        assert fixed["ok"], fixed
        saved = wf.workspace.snapshot()
        assert saved["facts"]["F_NEW"]["value"] == "合成桥"
        assert saved["version"] > before["version"]
        raise _CheckedHandlers()

    wf.react_runner = runner
    with pytest.raises(module.DocumentWorkflowFailure) as caught:
        wf.run()
    if not isinstance(caught.value.__cause__, _CheckedHandlers):
        raise caught.value.__cause__
    assert len(prompts) == 2
    assert wf.sessions[0]["tool_calls"] == calls


def test_restored_handler_feedback_rejects_oversized_or_malformed_repairs():
    valid = {"name": "update_workspace", "status": "failed", "error": {
        "code": "WORKSPACE_STALE", "action": "Call read_workspace again.", "fields": ["base_version"]}}
    outcome = {"tool_calls": [valid,
        {**valid, "name": "not_an_enabled_tool"},
        {**valid, "error": {"code": "WORKSPACE_STALE", "action": "x" * 3000}},
        {**valid, "error": {"code": "WORKSPACE_STALE", "action": "Read workspace", "fields": {"private": "state"}}},
        {**valid, "status": "success"}]}
    assert module._recent_handler_errors(outcome, {"update_workspace": object()}) == [{
        "name": "update_workspace", "code": "WORKSPACE_STALE",
        "action": "Call read_workspace again.", "fields": ["base_version"]}]
