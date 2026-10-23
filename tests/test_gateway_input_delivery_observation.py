"""Input-delivery evidence through the real gateway/client, without a model.

The ordinary-response cases use a local scripted HTTP server. Stream cases
use scripted transport bytes. Neither path can call the configured GPU service.
"""
from copy import deepcopy
import time

import pytest

from bridge_evidence.contest.document_model_gateway import AuditedModelGateway
from bridge_evidence.contest.local_model_client import LocalModelClient
from bridge_evidence.contest.reliable_generation import ReliableGenerationError
from test_document_model_gateway import completion, sdk_payload, synthetic_upstream
from test_repetition_recovery import route


def run_response(response, *, http_status=200, observed=None, callback=None):
    notifications = [] if observed is None else observed
    prepared, delivered = [], []
    with synthetic_upstream(response, generation_status=http_status) as (url, requests):
        client = LocalModelClient(url, "own-model", timeout=5)
        gateway = AuditedModelGateway(client, deadline=time.monotonic() + 15,
            max_calls=2, max_output_tokens=16,
            on_prepared_request=lambda payload, audit: prepared.append((payload, audit)),
            on_input_observed=callback or (lambda payload, audit: notifications.append((payload, audit))))
        gateway.on_success = lambda payload, receipt: delivered.append(deepcopy(receipt))
        error, result = None, None
        try:
            result = gateway.forward(sdk_payload())
        except Exception as exc:
            error = exc
    return gateway, client, prepared, notifications, delivered, requests, result, error


def test_valid_response_confirms_exact_prepared_request_before_success():
    gateway, client, prepared, observed, delivered, requests, result, error = run_response(completion())
    assert error is None and result == completion()
    assert len(observed) == len(prepared) == len(delivered) == 1
    payload, audit = observed[0]
    assert payload == prepared[0][0] == requests[-1][1] == sdk_payload()
    assert audit["request_id"] == prepared[0][1]["request_id"] == client.calls[-1]["request_id"]
    assert audit["input_delivery_confirmed"] is True
    assert audit["input_delivery_evidence"] == "complete_response"
    assert "response_validated" not in audit  # Hook precedes output validation.
    assert gateway.receipts[-1]["status"] == "success"


@pytest.mark.parametrize("bad", ["length", "wrong_usage", "malformed_tool_json"])
def test_unusable_output_does_not_erase_proven_input_delivery(bad):
    response = completion()
    if bad == "length":
        response["choices"][0]["finish_reason"] = "length"
    elif bad == "wrong_usage":
        response["usage"]["prompt_tokens"] += 1
    else:
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"entries":'
    gateway, _, _, observed, delivered, _, result, error = run_response(response)
    assert len(observed) == 1 and observed[0][1]["input_delivery_confirmed"] is True
    if bad == "malformed_tool_json":
        # The gateway preserves the model's answer; SDK parsing is a separate
        # gate. Do not repair, execute or infer successful tool storage here.
        assert error is None and result == response
        assert result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"entries":'
    else:
        assert error is not None and result is None and delivered == []
        assert gateway.receipts[-1]["status"] == "rejected"


@pytest.mark.parametrize("response,http_status", [
    ({"error": {"message": "synthetic rejected input"}}, 400),
    ({"id": "header-only", "choices": [], "usage": {"prompt_tokens": 23}}, 200),
    ({"id": "no-generation", "choices": [{"index": 0, "message": {}, "finish_reason": None}],
      "usage": {"prompt_tokens": 23, "completion_tokens": 0, "total_tokens": 23}}, 200),
])
def test_http_header_or_dispatch_without_generation_does_not_confirm(response, http_status):
    gateway, _, prepared, observed, delivered, _, result, error = run_response(response, http_status=http_status)
    assert prepared and error is not None and result is None
    assert observed == delivered == []
    assert gateway.receipts[-1].get("input_delivery_confirmed") is not True


@pytest.mark.parametrize("mode,confirmed", [("loop", True), ("timeout", False)])
def test_failed_stream_requires_observed_generated_data(tmp_path, mode, confirmed):
    client, gateway, payload, feedback = route(tmp_path, [mode], observer=None, recovery=False)
    observed = []
    gateway.on_input_observed = lambda sent, audit: observed.append((sent, audit))
    with pytest.raises(ReliableGenerationError):
        gateway.forward(payload)
    assert bool(observed) is confirmed
    assert not feedback
    receipt = next(r for r in client.calls if r["kind"] == "agent_generation")
    assert receipt.get("input_delivery_confirmed", False) is confirmed
    if confirmed:
        assert observed[0][1]["input_delivery_evidence"] == "generated_stream_data"
        assert observed[0][0]["messages"] == payload["messages"]
        assert receipt["content_characters"] >= 512
        assert receipt["done_received"] is False
    assert gateway.receipts[-1]["status"] == "rejected"


def test_confirmation_persistence_failure_blocks_sdk_delivery_without_hiding_generation():
    observed = []

    def broken_persist(payload, audit):
        observed.append((deepcopy(payload), deepcopy(audit)))
        raise OSError("synthetic persistence failure with private path")

    gateway, client, _, _, delivered, _, result, error = run_response(completion(), callback=broken_persist)
    assert isinstance(error, RuntimeError) and "Could not persist confirmed model input delivery" in str(error)
    assert "private path" not in str(error)
    assert len(observed) == 1 and result is None and delivered == []
    assert gateway.receipts[-1]["failure_stage"] == "input_observed_callback"
    assert gateway.receipts[-1]["status"] == "rejected"
    assert gateway.receipts[-1]["input_delivery_confirmed"] is True  # Transport fact, not successful disk commit.
    assert len([r for r in client.calls if r["kind"] == "agent_generation"]) == 1


def test_input_callback_cannot_mutate_wire_payload_or_result():
    def mutate(payload, audit):
        payload["messages"].clear()
        audit["request_id"] = "not-the-real-request"
        audit["input_delivery_confirmed"] = False

    gateway, _, prepared, _, delivered, requests, result, error = run_response(completion(), callback=mutate)
    assert error is None and result == completion()
    assert requests[-1][1] == prepared[0][0] == sdk_payload()
    assert gateway.receipts[-1]["request_id"] == prepared[0][1]["request_id"]
    assert gateway.receipts[-1]["input_delivery_confirmed"] is True
    assert delivered
