"""Same native Agent business progress, using only synthetic SDK/receipts."""
from copy import deepcopy
import json
import socket
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bridge_evidence.contest import tong_react_adapter as adapter
from bridge_evidence.contest.document_workflow import DocumentWorkflow
from test_section_agent_adapter import fake_sdk, Tool, Parameters


class TextAction:
    def __init__(self, content="finished", reason="stop"):
        self.content, self.reason = content, reason

    def model_dump(self, **kwargs):
        return {"content": self.content, "is_final_response": True,
                "response_meta": {"finish_reason": self.reason}}


class ToolErrorAction:
    def model_dump(self, **kwargs):
        return {"tool_name": "read_outline", "message": "synthetic SDK failure"}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("This suite must not start processes or use a network")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)
    assert adapter._ACTIVE_SESSION is None
    adapter._REGISTRATIONS.clear()
    adapter._REGISTERED_TOOL_CLASSES.clear()


def execute(script, callback=lambda state: None, *, enabled=True, advance=True, **overrides):
    calls, prompts, checks = [], [], []
    current = {"id": None}
    memory = SimpleNamespace(markers=[])
    def events(registry, settings, outputs, prompt):
        prompts.append(prompt)
        if advance or current["id"] is None:
            current["id"] = "actual-generation-" + str(len(prompts))
        yield from script(registry, len(prompts), prompt)
    sdk, registry, settings, contexts, _ = fake_sdk(events)
    config = {"profile": "section", "url": "http://127.0.0.1:8000/v1", "model": "synthetic-own",
              "max_calls": 20, "current_agent_call_id": lambda: current["id"],
              "initial_memory_factory": lambda: memory}
    if enabled:
        config.update(continue_after_native_exit=lambda state: (calls.append(deepcopy(state)), callback(state))[1],
                      native_exit_is_settled=lambda: True,
                      before_same_agent_continue=lambda: checks.append(True))
    config.update(overrides)
    with patch.object(adapter, "_load_tongagents", return_value=sdk), \
            patch.object(adapter, "_load_tool_schema_api", return_value=(Tool, Parameters)):
        result = adapter.run_react("initial source", {"read_outline": lambda: {"outline": []}}, config)
    return SimpleNamespace(result=result, calls=calls, prompts=prompts, checks=checks,
                           settings=settings, contexts=contexts, memory=memory)


def routine(registry, step, prompt):
    registry["read_outline"]()
    yield TextAction("partial" if step == 1 else "finished")


def test_continue_reuses_agent_context_memory_and_budget_without_replaying_tools():
    run = execute(routine, lambda state: "Save missing advice rows" if state["step_invocation"] == 1 else None)
    assert run.result["status"] == "completed"
    assert run.prompts == ["initial source", "Save missing advice rows"]
    assert len(run.contexts) == len(run.settings) == 1
    assert run.contexts[0].memory is run.memory
    assert len(run.result["tool_calls"]) == 2
    assert run.checks == [True]
    assert run.calls[0]["step_tool_call_count"] == 1
    assert run.calls[1]["tool_call_count"] == 2
    receipts = run.result["business_continuations"]
    assert len(receipts) == 1 and receipts[0]["same_agent"] and receipts[0]["same_memory"]
    assert "Save missing advice rows" not in json.dumps(receipts)
    assert run.result["response_text"] == "finished"


@pytest.mark.parametrize("empty_event", [False, True])
def test_empty_successful_stop_can_continue(empty_event):
    def script(registry, step, prompt):
        registry["read_outline"]()
        if step > 1:
            yield TextAction()
        elif empty_event:
            yield TextAction("")
    run = execute(script, lambda state: "Continue saving" if state["step_invocation"] == 1 else None)
    assert run.result["status"] == "completed" and len(run.prompts) == 2


def test_legacy_default_and_explicit_none_do_not_start_another_step():
    for enabled in (False, True):
        run = execute(routine, enabled=enabled)
        assert run.result["status"] == "completed"
        assert run.prompts == ["initial source"] and not run.checks


def test_previous_final_cannot_accept_later_empty_exit():
    def script(registry, step, prompt):
        registry["read_outline"]()
        if step == 1:
            yield TextAction("old final must not survive")
    run = execute(script, lambda state: "Continue" if state["step_invocation"] == 1 else None)
    assert run.result["status"] == "failed"
    assert run.result["failure_reason"] == "missing_or_empty_native_final_response"
    assert run.result["native_final_event"] is False and run.result["response_text"] == ""


def test_following_timeout_keeps_prior_saved_work_but_never_prior_final():
    def script(registry, step, prompt):
        if step == 2:
            raise TimeoutError("synthetic timeout")
        yield from routine(registry, step, prompt)
    run = execute(script, lambda state: "Continue")
    assert run.result["status"] == "error"
    assert run.result["native_final_event"] is False
    assert len(run.result["tool_calls"]) == 1 and len(run.calls) == 1


def test_unknown_or_unfinished_generation_does_not_invoke_business_callback():
    run = execute(routine, lambda state: "Must not run", native_exit_is_settled=lambda: False)
    assert not run.calls and len(run.prompts) == 1
    assert run.result["status"] == "failed"
    assert run.result["failure_reason"] == "business_continuation_unsettled_generation"


def test_repeated_generation_identity_stops_without_another_business_callback():
    run = execute(routine, lambda state: "Continue", advance=False)
    assert len(run.calls) == 1 and len(run.prompts) == 2
    assert run.result["failure_reason"] == "business_continuation_without_generation_progress"


@pytest.mark.parametrize("mode", ["truncated", "sdk_error", "exception", "tool_budget"])
def test_error_exit_does_not_request_business_recovery(mode):
    def script(registry, step, prompt):
        registry["read_outline"]()
        if mode == "exception":
            raise RuntimeError("synthetic failure")
        if mode == "sdk_error":
            yield ToolErrorAction()
        elif mode == "tool_budget":
            registry["read_outline"]()
            yield TextAction()
        else:
            yield TextAction(reason="length")
    run = execute(script, lambda state: "Must not run", max_calls=1 if mode == "tool_budget" else 20)
    assert not run.calls and len(run.prompts) == 1
    assert run.result["status"] in {"failed", "error"}


def test_original_shared_budget_can_refuse_before_next_step():
    def exhausted():
        raise TimeoutError("Original report deadline exhausted")
    run = execute(routine, lambda state: "Continue", before_same_agent_continue=exhausted)
    assert run.result["status"] == "error" and len(run.prompts) == 1
    assert not run.result["business_continuations"]


def test_hard_shared_call_bound_does_not_allow_infinite_final_only_steps():
    run = execute(routine, lambda state: "Continue", max_calls=1)
    assert run.result["status"] == "failed" and len(run.prompts) == 1
    assert run.result["failure_reason"] == "business_continuation_budget_exhausted"


@pytest.mark.parametrize("value", ["", "  ", {}, False])
def test_invalid_feedback_cannot_start_another_step(value):
    run = execute(routine, lambda state: value)
    assert run.result["status"] == "error" and len(run.prompts) == 1


@pytest.mark.parametrize("missing", ["native_exit_is_settled", "before_same_agent_continue", "current_agent_call_id"])
def test_continuation_requires_receipt_guard_identity_and_shared_budget(missing):
    run = execute(routine, **{missing: None})
    assert run.result["status"] == "error" and not run.settings


def workflow_receipts():
    wf = object.__new__(DocumentWorkflow)
    wf._active_agent_call_id = "actual-1"
    wf._single_agent_terminal_failure = None
    wf.model_dispatches = [{"status": "settled"}]
    wf.client = SimpleNamespace(calls=[{"kind": "agent_generation", "call_id": "actual-1", "request_id": "request-1",
        "status": "success", "http_status": 200, "done_received": True, "finish_reason": "stop"}])
    wf.gateway_receipts = [{"request_kind": "agent_generation", "call_id": "actual-1", "request_id": "request-1",
        "status": "success", "http_status": 200, "finish_reason": "stop"}]
    wf._model_forward_lock = threading.Lock()
    return wf


def test_workflow_requires_matching_successful_actual_stop_receipts():
    assert workflow_receipts()._native_exit_is_settled() is True
    mutations = [lambda w: w.model_dispatches[0].update(status="outcome_unknown"),
        lambda w: w.model_dispatches[0].update(status="pending"),
        lambda w: w.client.calls[-1].update(done_received=False, cancel_status="cancellation_confirmed"),
        lambda w: w.client.calls[-1].update(status="failed"),
        lambda w: w.client.calls[-1].update(http_status=400),
        lambda w: w.gateway_receipts[-1].update(status="rejected"),
        lambda w: w.gateway_receipts[-1].update(finish_reason="length"),
        lambda w: w.gateway_receipts[-1].update(finish_reason="tool_calls"),
        lambda w: w.gateway_receipts[-1].update(call_id="different"),
        lambda w: w.gateway_receipts[-1].update(request_id="different"),
        lambda w: setattr(w, "_single_agent_terminal_failure", {}),
        lambda w: setattr(w, "_active_agent_call_id", None),
        lambda w: w._model_forward_lock.acquire()]
    for mutate in mutations:
        wf = workflow_receipts()
        mutate(wf)
        assert wf._native_exit_is_settled() is False


def test_workflow_is_explicit_opt_in_and_current_finish_stops_nonempty_final_only():
    wf = workflow_receipts()
    assert wf._continue_after_native_exit({}) is None
    assert wf._native_exit_continuation_config() == {}
    class Merge(DocumentWorkflow):
        def _continue_after_native_exit(self, state):
            return "Complete saved Word self-check"
    wf.__class__ = Merge
    wf._finish_is_current = lambda: False
    config = wf._native_exit_continuation_config()
    assert config["continue_after_native_exit"]({}) == "Complete saved Word self-check"
    wf._finish_is_current = lambda: True
    assert config["continue_after_native_exit"]({"native_final_event": True, "response_text": "finished"}) is None
    assert config["continue_after_native_exit"]({"native_final_event": True, "response_text": ""}) == "Complete saved Word self-check"
    assert config["continue_after_native_exit"]({"native_final_event": False, "response_text": ""}) == "Complete saved Word self-check"
