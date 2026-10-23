"""Long reverse reviews and restart recovery through the actual workflow loop.

The model is an in-memory protocol double. Its next action is derived only
from the request it receives, so a reconstructed instance has no hidden cursor
or notes. No SDK process, listener, report dataset or remote service is used.
"""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from bridge_evidence.contest import document_workflow as module
from bridge_evidence.contest.document_review_lookup import LOOKUP_TOOLS
from bridge_evidence.contest.run_state import json_digest
from test_document_workflow import SyntheticNativeGateway, completed, workflow
from test_document_workflow_lookup import ToolModel, decision, reverse, seed


TEXT = "桥梁名称：合成桥\n正文左幅1号梁裂缝一条，长2m。\n结论仍在描述同一病害；必须保留的原文任务标记：PRIMARY-SOURCE-KEEP。"


def bounded_native_driver(prompt, handlers, config):
    """Simulate SDK return at its tool limit, retaining actual gateway evidence.

    No history is rewritten inside a session. Only the production workflow can
    start another session and provide the saved, explicitly non-evidence notes.
    """
    gateway = SyntheticNativeGateway.active[-1]
    messages = [{"role": "system", "content": config["system_prompt"]},
                {"role": "user", "content": prompt}]
    calls = []
    for _ in range(config["max_iterations"]):
        request = {"model": gateway.client.model, "messages": deepcopy(messages),
                   "tools": deepcopy(LOOKUP_TOOLS), "tool_choice": "auto", "stream": False,
                   "max_tokens": config["max_tokens"], "temperature": config["temperature"],
                   "chat_template_kwargs": {"enable_thinking": config["enable_thinking"]}}
        message = gateway.forward(request)["choices"][0]["message"]
        messages.append(deepcopy(message))
        if not message.get("tool_calls"):
            return completed(calls, profile="reviewer", response_text=message["content"])
        for tool in message["tool_calls"]:
            name = tool["function"]["name"]
            if len(calls) >= config["max_calls"]:
                # Match the SDK's pre-argument-processing rejection. This is a
                # fully returned local scheduling limit, not a failed HTTP call.
                calls.append({"name": name, "status": "failed",
                              "args": {"invalid_or_unprocessed": True}, "args_sha256": None,
                              "error": {"code": "TOOL_CALL_BUDGET"}})
                return {**completed(calls, profile="reviewer"), "status": "failed",
                        "native_final_event": False, "failure_reason": "tool_budget_exceeded"}
            arguments = json.loads(tool["function"]["arguments"])
            result = handlers[name](**arguments)
            calls.append({"name": name, "args": arguments, "args_sha256": json_digest(arguments),
                          "status": "success", "result": result,
                          "agent_call_id": config["current_agent_call_id"]()})
            messages.append({"role": "tool", "tool_call_id": tool["id"],
                             "content": json.dumps(result, ensure_ascii=False)})
    raise AssertionError("Synthetic SDK exhausted iterations before reaching its expected return boundary")


class PagedReviewer(ToolModel):
    def __init__(self, namespace="initial", *, fail_on=None):
        super().__init__(self.respond)
        self.max_model_len = 5000
        self.namespace = namespace
        self.fail_on = fail_on
        self.page_requests = []
        self.visible_pages = []
        self.saved_notes = []

    def _tokenize(self, payload):
        # Count the actual complete message history AND tool definitions, not
        # just the user text. Production uses the served model's tokenizer.
        return self.count_tokens(json.dumps({"messages": payload["messages"],
                                            "tools": payload.get("tools", [])}, ensure_ascii=False)) + 12

    def _request(self, path, payload, *, kind):
        if len(self.requests) + 1 == self.fail_on:
            self.requests.append(deepcopy(payload))
            self.calls.append({"call_id": f"{self.namespace}:failed-{len(self.requests)}", "kind": kind,
                               "status": "failed", "endpoint": path, "input_sha256": json_digest(payload)})
            raise ConnectionError("synthetic paged-review transport interruption")
        result, receipt = super()._request(path, payload, kind=kind)
        receipt.update(call_id=f"{self.namespace}:{receipt['call_id']}", endpoint=path,
                       input_sha256=json_digest(payload))
        return result, receipt

    def tool(self, name, arguments):
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"{self.namespace}:tool-{len(self.requests)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}, "tool_calls"

    def respond(self, request):
        primary = json.loads(request["messages"][1]["content"])
        assert primary["direction"] == "REVERSE"
        assert primary["draft_catalog"]["mode"] == "paged"
        last = request["messages"][-1]
        if last["role"] == "tool":
            reply = json.loads(last["content"])
            assert reply["ok"], reply
            if "items" in reply:
                self.visible_pages.append(deepcopy(reply))
                notes = {"next_cursor": reply["next_cursor"], "checked_count": reply["end"],
                         "last_checked_path": reply["items"][-1]["path"],
                         "primary_evidence": primary["source_spans"],
                         "reason": "合成测试已查看这一页，原文仍须按对象语义核对", "unresolved": []}
                self.saved_notes.append(deepcopy(notes))
                return self.tool("review_save_notes", {"notes_json": json.dumps(notes, ensure_ascii=False)})
            assert "notes_sha256" in reply
            prior_save = request["messages"][-2]["tool_calls"][0]
            assert prior_save["function"]["name"] == "review_save_notes"
            notes = json.loads(json.loads(prior_save["function"]["arguments"])["notes_json"])
            assert json_digest(notes) == reply["notes_sha256"]
            assert notes["primary_evidence"] == primary["source_spans"]
            # This extra call is rejected by the SDK's two-tool limit before
            # lookup execution. The next session must recover via saved notes.
            return self.tool("review_save_notes", {"notes_json": json.dumps(notes, ensure_ascii=False)})
        else:
            continuation = primary.get("native_review_continuation")
            if continuation:
                assert continuation["notes_are_evidence"] is False
                notes = continuation["saved_notes"]
                assert notes["primary_evidence"] == primary["source_spans"]
                cursor = notes["next_cursor"]
                if cursor is None:
                    assert notes["checked_count"] == primary["draft_catalog"]["total_items"]
                    return {"role": "assistant", "content": json.dumps(decision(primary), ensure_ascii=False)}, "stop"
            else:
                cursor = 0
        self.page_requests.append(cursor)
        return self.tool("review_read_draft_page", {"cursor": cursor, "limit": 100})


def configured(directory, client, *, initialize=True):
    settings = module.DocumentSettings(max_model_len=client.max_model_len, output_tokens=300, tool_tokens=400,
                                       history_reserve_tokens=100, review_iterations=512, review_batch_items=1,
                                       session_iterations=3, session_calls=2, max_sessions=256,
                                       max_model_calls=600, max_review_attempts=1)
    wf = workflow(directory, client=client, settings=settings, text=TEXT)
    if initialize:
        seed(wf, extra_rows=180)
    wf._active_review_items = wf._output_items(wf.workspace.snapshot())

    def review(payload):
        runner = wf.react_runner
        wf.react_runner = bounded_native_driver
        try:
            with patch.object(module, "AuditedModelGateway", SyntheticNativeGateway):
                return module.DocumentWorkflow._review_request(wf, payload)
        finally:
            wf.react_runner = runner

    wf._review_request = review
    return wf


def assert_source_and_context_budget(clients, primary, settings):
    requests = [request for client in clients for request in client.requests]
    for client in clients:
        for request in client.requests:
            source_assignment = json.loads(request["messages"][1]["content"])
            assert source_assignment["source"] == primary["source"]
            assert source_assignment["source_spans"] == primary["source_spans"]
            assert "PRIMARY-SOURCE-KEEP" in source_assignment["source"]
            assert client._tokenize(request) + request["max_tokens"] <= settings.max_model_len
    memories = []
    for request in requests:
        primary = json.loads(request["messages"][1]["content"])
        continuation = primary.get("native_review_continuation")
        if continuation:
            assert continuation["notes_are_evidence"] is False
            memories.append(continuation["saved_notes"])
    assert len({note["checked_count"] for note in memories}) >= 15
    # Every returned session starts a bounded new history. The saved notes
    # accompany the same complete source, never replace it.
    drops = [(before, after) for before, after in zip(requests, requests[1:])
             if len(after["messages"]) < len(before["messages"])]
    assert len(drops) >= 15
    assert max(len(request["messages"]) for request in requests) <= 7


def assert_complete_paging(clients, expected_items):
    pages = [page for client in clients for page in client.visible_pages]
    assert len(pages) >= 15
    assert [item for page in pages for item in page["items"]] == [
        {"path": item["path"], "value": item["value"]} for item in expected_items]
    cursor = 0
    for page in pages:
        assert page["start"] == cursor
        assert page["end"] > cursor
        cursor = page["end"]
    assert cursor == len(expected_items)
    issued = [cursor for client in clients for cursor in client.page_requests]
    assert issued == [page["start"] for page in pages]
    assert len(issued) == len(set(issued)) and issued.count(0) == 1


def assert_accounting(wf, clients, outcome, *, failed=0):
    calls = [call for client in clients for call in client.calls]
    successful = [call["call_id"] for call in calls if call["status"] == "success"]
    assert outcome["model_call_ids"] == successful
    assert len(successful) > wf.settings.session_calls
    assert len(calls) == len(successful) + failed
    assert len({call["call_id"] for call in calls}) == len(calls)
    receipt = wf._receipt("completed", "navigation")
    assert receipt["review_generation_requests"] == len(calls)
    assert [call["call_id"] for call in receipt["model_calls"]] == [call["call_id"] for call in calls]
    assert len(receipt["gateway_receipts"]) == len(calls)
    assert len(wf.model_dispatches) == len(calls)
    assert all(dispatch["status"] == "settled" for dispatch in wf.model_dispatches)


def test_long_reverse_review_saves_notes_repeatedly_without_losing_source_or_overflowing(tmp_path):
    client = PagedReviewer()
    wf = configured(tmp_path, client)
    primary = reverse(wf)
    expected = wf._output_items(wf.workspace.snapshot())

    result = wf._review_request(primary)

    assert result["passed"]
    paused = [session for session in wf.review_sessions if session.get("continuation_state") == "budget_paused"]
    assert len(paused) >= 15
    assert all(session["status"] == "returned" and not session["terminal_failure"] for session in paused)
    assert_complete_paging([client], expected)
    assert_source_and_context_budget([client], primary, wf.settings)
    assert_accounting(wf, [client], result)
    # The accumulated page bodies alone exceed model context. This is a real
    # long-history test, not a short request that would fit without notes.
    assert client.count_tokens(json.dumps(client.visible_pages, ensure_ascii=False)) > client.max_model_len
    names = [call["name"] for call in result["draft_lookup"]["tool_calls"]]
    assert names.count("review_read_draft_page") == names.count("review_save_notes") == len(client.visible_pages)


@pytest.mark.parametrize("fail_on", [18, 19], ids=["after-note-save", "before-next-page"])
def test_unknown_request_preserves_notes_cursor_and_accounting_without_automatic_retry(tmp_path, fail_on):
    first = PagedReviewer(fail_on=fail_on)
    wf = configured(tmp_path, first)
    primary = reverse(wf)
    with pytest.raises(RuntimeError, match="Unsettled or unknown model request"):
        wf._review_request(primary)
    assert len(first.requests) == fail_on
    checkpoint = wf._partial_checkpoint(primary, json_digest(wf._active_review_items))
    saved = checkpoint.load()
    assert saved["status"] == "returned"
    assert saved["iterations_spent"] == fail_on
    assert saved["tool_calls_spent"] == fail_on - 1
    assert saved["lookup_state"]["notes"]["checked_count"] > 0
    assert saved["lookup_state"]["notes"] == first.saved_notes[-1]
    assert wf.review_sessions[saved["session_index"]]["terminal_failure"]
    assert wf.model_dispatches[-1]["status"] == "outcome_unknown"
    recorded_ids = [call["call_id"] for call in wf._model_calls()]
    assert saved["model_call_ids"] == recorded_ids
    assert len(recorded_ids) == len(set(recorded_ids)) == fail_on

    second = PagedReviewer(namespace="restored")
    resumed = configured(tmp_path, second, initialize=False)
    assert resumed.identity == wf.identity
    assert [call["call_id"] for call in resumed._model_calls()] == recorded_ids
    with pytest.raises(RuntimeError, match="Unsettled or unknown model request"):
        resumed._review_request(primary)

    assert second.requests == second.calls == []
    assert checkpoint.load() == saved
    assert len(resumed.gateway_receipts) == fail_on
    assert resumed.workspace.snapshot()["review"] is None
    # All successfully executed lookups and their source positions survive;
    # no cursor is advanced and no incomplete review is published as passed.
    assert saved["lookup_state"]["seen_paths"] == sorted(
        item["path"] for page in first.visible_pages for item in page["items"])
    assert not resumed._receipt("failed", "navigation")["delivery"]["ready"]


def test_restart_continues_from_durable_returned_budget_boundary_without_repeating_pages(tmp_path, monkeypatch):
    first = PagedReviewer()
    wf = configured(tmp_path, first)
    primary = reverse(wf)
    expected = wf._output_items(wf.workspace.snapshot())
    original_persist = wf._persist
    interrupted = []

    def persist_then_interrupt():
        original_persist()
        if (not interrupted and len(wf.review_sessions) == 8
                and wf.review_sessions[-1].get("continuation_state") == "budget_paused"):
            interrupted.append(True)
            raise RuntimeError("synthetic interruption after durable returned budget boundary")

    with monkeypatch.context() as patcher:
        patcher.setattr(wf, "_persist", persist_then_interrupt)
        with pytest.raises(RuntimeError, match="after durable returned budget boundary"):
            wf._review_request(primary)
    checkpoint = wf._partial_checkpoint(primary, json_digest(wf._active_review_items))
    saved = checkpoint.load()
    assert saved["status"] == "returned"
    assert len(first.requests) == 24
    assert all(dispatch["status"] == "settled" for dispatch in wf.model_dispatches)
    assert len(first.visible_pages) == 8

    second = PagedReviewer(namespace="restored")
    resumed = configured(tmp_path, second, initialize=False)
    result = resumed._review_request(primary)

    assert result["passed"]
    continuation = json.loads(second.requests[0]["messages"][1]["content"])["native_review_continuation"]
    assert continuation["saved_notes"] == saved["lookup_state"]["notes"]
    assert second.page_requests[0] == saved["lookup_state"]["notes"]["next_cursor"]
    assert result["draft_lookup"]["tool_calls"][:len(saved["lookup_state"]["calls"])] == saved["lookup_state"]["calls"]
    assert_complete_paging([first, second], expected)
    assert_source_and_context_budget([first, second], primary, resumed.settings)
    assert_accounting(resumed, [first, second], result)


def test_restart_reuses_durable_returned_final_without_any_regeneration(tmp_path, monkeypatch):
    first = PagedReviewer()
    wf = configured(tmp_path, first)
    primary = reverse(wf)
    expected = wf._output_items(wf.workspace.snapshot())

    def interrupt_before_decision(*args, **kwargs):
        raise RuntimeError("synthetic interruption after durable returned final")

    with monkeypatch.context() as patcher:
        patcher.setattr(wf, "_native_final_decision", interrupt_before_decision)
        with pytest.raises(RuntimeError, match="after durable returned final"):
            wf._review_request(primary)
    checkpoint = wf._partial_checkpoint(primary, json_digest(wf._active_review_items))
    saved = checkpoint.load()
    assert saved["status"] == "returned"
    assert saved["model_call_ids"] == [call["call_id"] for call in first.calls]
    assert wf.review_sessions[saved["session_index"]]["outcome"]["native_final_event"] is True
    assert_complete_paging([first], expected)

    second = PagedReviewer(namespace="restored")
    resumed = configured(tmp_path, second, initialize=False)
    result = resumed._review_request(primary)

    assert result["passed"]
    assert second.requests == second.calls == second.page_requests == []
    assert_complete_paging([first, second], expected)
    assert_source_and_context_budget([first, second], primary, resumed.settings)
    assert_accounting(resumed, [first, second], result)
