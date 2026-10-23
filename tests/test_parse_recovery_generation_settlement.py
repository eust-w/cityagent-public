"""A syntax receipt alone cannot authorize recovered model-history rendering."""
from copy import deepcopy
from threading import Lock
from types import SimpleNamespace

import pytest

from bridge_evidence.contest.document_workflow import DocumentWorkflow


@pytest.fixture
def workflow():
    row = {"kind": "agent_generation", "call_id": "call", "request_id": "request",
           "status": "success", "http_status": 200, "done_received": True,
           "finish_reason": "tool_calls", "usage_known": True,
           "generation_attempted": True, "prompt_tokens_checked": 100,
           "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
    result = object.__new__(DocumentWorkflow)
    result.client = SimpleNamespace(calls=[row], max_model_len=262144)
    result.gateway_receipts = [{**deepcopy(row), "request_kind": "agent_generation"}]
    result.model_dispatches = [{"status": "settled", "request_id": "request"}]
    result._model_forward_lock = Lock()
    result._active_agent_call_id = "call"
    result._single_agent_terminal_failure = None
    return result


@pytest.mark.parametrize("finish", ["stop", "tool_calls", "function_call"])
def test_known_tool_or_stop_result_can_render_recovery(workflow, finish):
    workflow.client.calls[-1]["finish_reason"] = finish
    workflow.gateway_receipts[-1]["finish_reason"] = finish
    assert workflow._parse_recovery_generation_is_settled()
    assert workflow._model_forward_lock.acquire(blocking=False)
    workflow._model_forward_lock.release()


@pytest.mark.parametrize("field,value", [
    ("done_received", False), ("status", "failed"), ("http_status", 500),
    ("finish_reason", "length"), ("usage_known", False), ("usage", None),
    ("request_id", "other"), ("call_id", "other"), ("prompt_tokens_checked", 99),
    ("usage", {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 121}),
    ("usage", {"prompt_tokens": True, "completion_tokens": 20, "total_tokens": 21}),
])
def test_uncertain_mismatched_or_invalid_client_result_refused(workflow, field, value):
    workflow.client.calls[-1][field] = value
    assert not workflow._parse_recovery_generation_is_settled()


@pytest.mark.parametrize("scenario", ["pending", "duplicate_call", "duplicate_audit",
    "unsettled_generation", "gateway_only", "lock_busy", "terminal_failure", "no_origin"])
def test_inflight_or_ambiguous_origin_refused(workflow, scenario):
    if scenario == "pending":
        workflow.model_dispatches.append({"status": "pending"})
    elif scenario == "duplicate_call":
        workflow.client.calls.insert(0, deepcopy(workflow.client.calls[-1]))
    elif scenario == "duplicate_audit":
        workflow.gateway_receipts.insert(0, deepcopy(workflow.gateway_receipts[-1]))
    elif scenario == "unsettled_generation":
        workflow.client.calls.insert(0, {"kind": "agent_generation", "call_id": "old",
            "generation_attempted": True, "done_received": False})
    elif scenario == "gateway_only":
        workflow.gateway_receipts.insert(0, {"request_kind": "agent_generation",
            "call_id": "unknown", "generation_attempted": True})
    elif scenario == "lock_busy":
        workflow._model_forward_lock.acquire()
    elif scenario == "terminal_failure":
        workflow._single_agent_terminal_failure = {"status": "unknown"}
    else:
        workflow._active_agent_call_id = None
    assert not workflow._parse_recovery_generation_is_settled()


def test_resumed_client_uses_persisted_receipts_without_relabeling_old_usage(workflow):
    old = {**deepcopy(workflow.client.calls[0]), "call_id": "previous", "request_id": "previous-request"}
    workflow.model_calls_before_resume = [old]
    workflow.gateway_receipts.insert(0, {**deepcopy(old), "request_kind": "agent_generation"})
    assert workflow._parse_recovery_generation_is_settled()
    # A stopped old stream need not acquire fabricated DONE or usage metadata.
    old.update(status="failed", done_received=False, usage_known=False, usage=None,
               response_id="previous-response", cancel_status="not_confirmed")
    assert not workflow._parse_recovery_generation_is_settled()
    old["stop_observation"] = {"response_id": old["response_id"], "stopped_observed": True,
        "method": "service_idle_after_disconnect", "same_service_process": True,
        "cancellation_confirmed": False, "model_identity_sha256": "model",
        "process_identity_sha256": "process", "samples": [
            {"idle": True, "after_disconnect_seconds": 1},
            {"idle": True, "after_disconnect_seconds": 1.5}]}
    assert workflow._parse_recovery_generation_is_settled()
    assert old["done_received"] is False and old["usage"] is None
    assert old["cancel_status"] == "not_confirmed"
    old["stop_observation"]["same_service_process"] = False
    assert not workflow._parse_recovery_generation_is_settled()


def test_resume_does_not_hide_duplicate_current_call_or_missing_old_receipt(workflow):
    workflow.model_calls_before_resume = [deepcopy(workflow.client.calls[0])]
    assert not workflow._parse_recovery_generation_is_settled()
    workflow.model_calls_before_resume = []
    workflow.gateway_receipts.insert(0, {"request_kind": "agent_generation", "call_id": "old",
        "generation_attempted": True})
    assert not workflow._parse_recovery_generation_is_settled()
