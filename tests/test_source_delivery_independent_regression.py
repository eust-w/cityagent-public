"""Independent integration/fault tests; scripted CPU HTTP only, no model."""
from copy import deepcopy
import json
import time

import pytest

from bridge_evidence.contest.document_entries import EntryStore
from bridge_evidence.contest.document_model_gateway import AuditedModelGateway
from bridge_evidence.contest.document_source_delivery import SourceDeliveryLedger, RECEIPT_FIELD
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.local_model_client import LocalModelClient
from bridge_evidence.contest import document_source_delivery as delivery_module
from test_document_model_gateway import completion, sdk_payload, synthetic_upstream


def fixture(tmp_path, callback=None):
    manifest = manifest_from_text("一号桥原始名称\n本次检查支座锈蚀\n2017年检查无明显变化")
    store = EntryStore(manifest, "scoring", tmp_path / "entries")
    ledger = SourceDeliveryLedger(manifest, tmp_path / "delivery", run_id="offline", agent_id="scoring",
                                  on_confirmed=callback or store.record_read)
    spans = [{"evidence_id": u["id"], "start": 0, "end": len(u["text"]), "text": u["text"]}
             for u in manifest["units"]]
    page = ledger.prepare_page({"ok": True, "spans": spans[:1], "context": spans[1:]}, tool="read_source")
    payload = sdk_payload()
    payload["messages"][-1]["content"] = json.dumps({"ok": True, "tool": "read_source", "result": page}, ensure_ascii=False)
    return manifest, store, ledger, page, payload


def stored(store):
    return json.loads(store.path.read_text())


def expected(manifest):
    return {u["id"]: [[0, len(u["text"])]] for u in manifest["units"]}


@pytest.mark.parametrize("output", ["valid", "bad_output", "http400"])
def test_real_gateway_to_ledger_to_entry_store_separates_delivery_from_output(tmp_path, output):
    manifest, store, ledger, page, payload = fixture(tmp_path)
    response = completion(finish_reason="length" if output == "bad_output" else "tool_calls")
    with synthetic_upstream(response, generation_status=400 if output == "http400" else 200) as (url, requests):
        client = LocalModelClient(url, "own-model", timeout=5)
        gateway = AuditedModelGateway(client, deadline=time.monotonic() + 15, max_calls=2, max_output_tokens=16,
            on_prepared_request=ledger.observe_request,
            on_input_observed=lambda wire, audit: ledger.confirm_request(audit["request_id"], evidence=audit))
        if output == "valid":
            gateway.forward(payload)
        else:
            with pytest.raises(Exception):
                gateway.forward(payload)
    assert requests[-1][1] == payload
    confirmed = output != "http400"
    assert ledger.confirmed_ranges() == (expected(manifest) if confirmed else {})
    assert stored(store)["reads"] == ledger.confirmed_ranges()
    assert ledger.is_confirmed(page[RECEIPT_FIELD]) is confirmed
    assert stored(store)["entries"] == {}  # Delivered text does not manufacture facts or approval.
    assert gateway.receipts[-1]["status"] == ("success" if output == "valid" else "rejected")


@pytest.mark.parametrize("fail_index", [1, 2, 3])
@pytest.mark.parametrize("when", ["before_write", "after_write"])
def test_each_range_callback_failure_recovers_real_store_without_duplicate_versions(tmp_path, fail_index, when):
    manifest, store, ledger, page, payload = fixture(tmp_path)
    callback_index = 0

    def interrupted(identifier, start, end):
        nonlocal callback_index
        callback_index += 1
        if callback_index == fail_index and when == "before_write":
            raise OSError("synthetic per-range failure")
        store.record_read(identifier, start, end)
        if callback_index == fail_index and when == "after_write":
            raise OSError("synthetic after-write failure")

    ledger.on_confirmed = interrupted
    ledger.observe_request(payload, {"request_id": "request-1"})
    with pytest.raises(OSError):
        ledger.confirm_request("request-1", evidence={"input_delivery_confirmed": True})
    disk = json.loads(ledger.path.read_text())
    assert disk["requests"]["request-1"]["status"] == "confirmed"
    assert disk["requests"]["request-1"]["applied"] is False
    assert len(stored(store)["reads"]) == fail_index - (when == "before_write")
    restored_store = EntryStore(manifest, "scoring", tmp_path / "entries")
    restored = SourceDeliveryLedger(manifest, tmp_path / "delivery", run_id="offline", agent_id="scoring",
                                    on_confirmed=restored_store.record_read)
    assert stored(restored_store)["reads"] == expected(manifest) == restored.confirmed_ranges()
    assert stored(restored_store)["version"] == 3  # Callback replays are idempotent, not extra changes.
    assert stored(restored_store)["entries"] == {}
    assert restored.state()["requests"]["request-1"]["applied"] is True
    restored.confirm_request("request-1", evidence={"input_delivery_confirmed": True})
    assert stored(restored_store)["version"] == 3


@pytest.mark.parametrize("when", ["before_rename", "after_rename", "after_apply_rename"])
def test_confirmation_disk_failure_reconciles_real_durable_state_and_restores(tmp_path, monkeypatch, when):
    manifest, store, ledger, page, payload = fixture(tmp_path)
    ledger.observe_request(payload, {"request_id": "request-1"})
    actual_write = delivery_module.atomic_json
    calls = 0

    def broken(path, value):
        nonlocal calls
        calls += 1
        if calls == 1 and when == "before_rename":
            raise OSError("before rename")
        actual_write(path, value)
        if (calls == 1 and when == "after_rename") or (calls == 2 and when == "after_apply_rename"):
            raise OSError("after durable rename")

    monkeypatch.setattr(delivery_module, "atomic_json", broken)
    with pytest.raises(OSError):
        ledger.confirm_request("request-1", evidence={"input_delivery_confirmed": True})
    assert ledger.state() == json.loads(ledger.path.read_text())
    monkeypatch.setattr(delivery_module, "atomic_json", actual_write)
    restored = SourceDeliveryLedger(manifest, tmp_path / "delivery", run_id="offline", agent_id="scoring",
                                    on_confirmed=store.record_read)
    if when == "before_rename":
        assert not restored.confirmed_ranges() and stored(store)["reads"] == {}
        assert restored.pending_page(page[RECEIPT_FIELD]) == page
        restored.confirm_request("request-1", evidence={"input_delivery_confirmed": True})
    assert stored(store)["reads"] == expected(manifest) == restored.confirmed_ranges()
    assert stored(store)["entries"] == {} and stored(store)["version"] == 3


def test_unobserved_compressed_initial_prompt_recovers_exact_original_without_promotion(tmp_path):
    manifest, store, ledger, _, _ = fixture(tmp_path)
    units = manifest["units"]
    prompt = json.dumps({"source_units": units}, ensure_ascii=False)
    ranges = [{"evidence_id": u["id"], "start": 0, "end": len(u["text"])} for u in units]
    receipt = ledger.register_initial_message(prompt, ranges)
    ledger.observe_request({"messages": [{"role": "user", "content": "摘要：已经读过桥梁报告"}]}, "compressed")
    ledger.confirm_request("compressed", evidence={"kind": "generation_observed", "generated_tokens": 1})
    assert stored(store)["reads"] == {}
    restored = SourceDeliveryLedger(manifest, tmp_path / "delivery", run_id="offline", agent_id="scoring",
                                    on_confirmed=store.record_read)
    recovered = restored.pending_initial_messages()
    assert recovered == [{"role": "user", "content": prompt}]
    restored.observe_request({"messages": recovered}, "recovered")
    restored.confirm_request("recovered", evidence={"kind": "generation_observed", "generated_tokens": 1})
    assert restored.is_confirmed(receipt) and stored(store)["reads"] == expected(manifest)
    assert stored(store)["entries"] == {} and restored.pending_initial_messages() == []
