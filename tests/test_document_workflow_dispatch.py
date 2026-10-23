"""Durable intent is distinct from a completed or even attempted HTTP call."""
import json

import pytest

from bridge_evidence.contest import document_workflow as module
from test_document_workflow import workflow, read, prepare
from bridge_evidence.contest.run_state import json_digest


def test_review_dispatch_is_durable_before_generation_and_settles(tmp_path):
    wf = workflow(tmp_path)
    read(wf)
    prepare(wf)
    observed = []

    def before_reply(_payload):
        stored = json.loads(wf.trace_path.read_text())
        pending = [d for d in stored["model_dispatches"] if d["status"] == "pending"]
        assert len(pending) == 1
        observed.append(pending[0]["request_id"])
        # The pending intent is not a fabricated successful HTTP receipt.
        assert all(c.get("request_id") != pending[0]["request_id"] for c in stored["model_calls"])

    wf.client.before_reply = before_reply
    wf._perform_review_draft()
    stored = json.loads(wf.trace_path.read_text())
    assert observed
    assert all(d["status"] == "settled" and d["generation_attempted"] for d in stored["model_dispatches"])
    assert {c["request_id"] for c in stored["model_calls"]} == set(observed)
    assert all(r["request_id"] in observed for r in stored["gateway_receipts"])


def test_unknown_dispatch_is_not_replayed_or_counted_as_completed_http(tmp_path):
    wf = workflow(tmp_path)
    wf._on_dispatch({"request_id": "interrupted-request", "request_kind": "agent_generation",
                     "request_sha256": "a" * 64, "started_at": "2026-10-13T01:00:00+00:00"})
    restored = workflow(tmp_path)
    with pytest.raises(module.DocumentWorkflowFailure, match="outcome is unknown") as caught:
        restored.run()
    receipt = caught.value.receipt
    assert receipt["uncertain_generation_attempts"] == 1
    assert receipt["agent_generation_requests"] == 0
    assert receipt["review_generation_requests"] == 0
    assert receipt["uncertain_budget_seconds"] > 0
    assert restored.client.calls == []
    twice = workflow(tmp_path)
    assert twice.uncertain_budget_seconds == restored.uncertain_budget_seconds
    assert len(twice.model_dispatches) == 1


def test_terminal_receipt_recovers_unsettled_dispatch_without_uncertainty(tmp_path):
    wf = workflow(tmp_path)
    wf._on_dispatch({"request_id": "denied", "request_kind": "generation"})
    wf.gateway_receipts.append({"request_id": "denied", "status": "rejected", "generation_attempted": False})
    wf._persist()
    restored = workflow(tmp_path)
    assert restored.model_dispatches[0]["status"] == "settled"
    assert restored.model_dispatches[0]["generation_attempted"] is False
    assert restored.uncertain_budget_seconds == 0
    restored._check_budget()


def test_failed_dispatch_persistence_does_not_leave_in_memory_intent(tmp_path, monkeypatch):
    wf = workflow(tmp_path)
    def fail(*_args, **_kwargs):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(module, "atomic_json", fail)
    with pytest.raises(OSError):
        wf._on_dispatch({"request_id": "not-sent"})
    assert wf.model_dispatches == []
    assert wf.client.calls == []


def test_checkpoint_exists_before_first_tokenizer_operation(tmp_path):
    wf = workflow(tmp_path)
    def fail_entry():
        stored = json.loads(wf.trace_path.read_text())
        assert stored["status"] == "running"
        assert stored["model_dispatches"] == []
        raise ValueError("synthetic entry failure")
    wf._entry = fail_entry
    with pytest.raises(module.DocumentWorkflowFailure):
        wf.run()
    assert json.loads(wf.trace_path.read_text())["status"] == "failed"


def test_later_terminal_receipt_reconciles_unknown_dispatch_and_releases_reservation(tmp_path):
    wf = workflow(tmp_path)
    wf._on_dispatch({"request_id": "eventually-denied", "request_kind": "generation"})
    uncertain = workflow(tmp_path)
    uncertain._persist("failed")
    uncertain.gateway_receipts.append({"request_id": "eventually-denied", "status": "rejected",
                                       "generation_attempted": False})
    uncertain._persist("failed")
    restored = workflow(tmp_path)
    assert restored.model_dispatches[0]["status"] == "settled"
    assert restored.uncertain_budget_seconds == 0
    restored._check_budget()


def test_sdk_invalid_tool_arguments_remain_available_in_local_response_sidecar(tmp_path):
    from test_document_workflow_lookup import configured, reverse, tool

    def malformed(_request):
        message = tool()
        message["tool_calls"][0]["function"]["arguments"] = '{"query":'
        return message, "tool_calls"

    wf = configured(tmp_path, malformed)
    with pytest.raises(ValueError):
        wf._review_request(reverse(wf))
    records = list((tmp_path / "model_responses").glob("*.json"))
    assert len(records) == 1
    saved = json.loads(records[0].read_text())
    assert saved["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"query":'
    audit = wf.gateway_receipts[0]
    assert saved["call_id"] == audit["call_id"]
    assert saved["request_id"] == audit["request_id"]
    assert saved["response_sha256"] == json_digest(saved["response"]) == audit["response_sha256"]
    assert len(wf.client.calls) == 1
    # Summary receipts identify the reply without echoing its contents.
    assert "response" not in audit
