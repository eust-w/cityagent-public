"""No model/network: exact request receipts, failures, recovery and isolation."""
from copy import deepcopy
import json

import pytest

from bridge_evidence.contest.document_source_delivery import SourceDeliveryLedger, RECEIPT_FIELD
from bridge_evidence.contest.document_tools import DocumentTools
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.managed_navigation_journal import ManagedNavigationJournal
from bridge_evidence.contest import document_source_delivery as delivery_module
from bridge_evidence.contest import tong_react_adapter as adapter


def _ledger(tmp_path, text="1 甲章\n甲正文\n甲末尾\n2 乙章\n乙正文", **kwargs):
    return SourceDeliveryLedger(manifest_from_text(text), tmp_path, run_id="run-1", agent_id="child-1", **kwargs)


def _request(page, request_id="request-1"):
    return {"messages": [{"role": "tool", "tool_call_id": "native-1", "content": json.dumps(
        {"ok": True, "tool": "read_source", "result": page}, ensure_ascii=False)}]}, {"request_id": request_id}


def _confirm(ledger, page, request_id="request-1"):
    payload, audit = _request(page, request_id)
    ledger.observe_request(payload, audit)
    return ledger.confirm_request(request_id, evidence={"request_id": request_id, "input_delivery_confirmed": True})


def _source_page(ledger, **overrides):
    source = ledger.manifest["units"][1]
    page = {"ok": True, "status": "read", "spans": [{"evidence_id": source["id"],
        "start": 0, "end": len(source["text"]), "text": source["text"]}], "context": []}
    page.update(overrides)
    return page


def _identity(generation="generation-1", native="native-1", args="args"):
    return {"run_id": "run-1", "session_id": "session-1", "generation_call_id": generation,
            "native_tool_call_id": native, "args_sha256": args}


def test_tools_prepare_but_only_confirmed_request_grants_spans_and_real_context(tmp_path):
    received = []
    ledger = _ledger(tmp_path, on_confirmed=lambda *span: received.append(span))
    tools = DocumentTools(ledger.manifest, count_tokens=len, max_tool_tokens=4000,
                          delivery_ledger=ledger, on_read=lambda *_: pytest.fail("Premature read callback"))
    page = tools.automatic_handlers()["read_source"]("E000002")
    assert received == [] and ledger.confirmed_ranges() == {}
    assert page["context"] and all("grants_coverage" not in s for s in page["context"])
    assert page["coverage_confirmation"] == "await_upstream_receipt" and page["semantic_review"] is False
    _confirm(ledger, page)
    expected = {(s["evidence_id"], s["start"], s["end"]) for s in page["spans"] + page["context"] if s["text"]}
    assert set(received) == expected
    assert ledger.is_confirmed(page[RECEIPT_FIELD])
    # Output JSON rejection is downstream of known input delivery.
    ledger.reject_request("request-1", "output JSON was invalid")
    assert ledger.is_confirmed(page[RECEIPT_FIELD])
    assert ledger.state()["requests"]["request-1"]["status"] == "confirmed"
    _confirm(ledger, page, "request-2")
    assert len(received) == len(expected), "Repeated history must not reapply old coverage"


@pytest.mark.parametrize("mutation", ["summary", "ids_only", "assistant_role", "unknown_receipt"])
def test_summaries_ids_or_forged_receipts_cannot_grant_original_coverage(tmp_path, mutation):
    ledger = _ledger(tmp_path)
    page = ledger.prepare_page(_source_page(ledger), tool="read_source")
    altered = deepcopy(page)
    payload, audit = _request(altered)
    if mutation == "summary":
        payload["messages"][0]["content"] = "总结：已经读取甲正文。"
    elif mutation == "ids_only":
        altered["spans"] = [{"evidence_id": "E000002"}]
        payload, audit = _request(altered)
    elif mutation == "assistant_role":
        payload["messages"][0]["role"] = "assistant"
    else:
        altered[RECEIPT_FIELD] = "foreign-document-receipt"
        payload, audit = _request(altered)
    ledger.observe_request(payload, audit)
    ledger.confirm_request("request-1", evidence={"kind": "generation_observed", "generated_tokens": 1})
    assert ledger.confirmed_ranges() == {}
    assert not ledger.is_confirmed(page[RECEIPT_FIELD])


def test_truncated_final_page_counts_only_actual_slice_and_keeps_page_pending(tmp_path):
    ledger = _ledger(tmp_path)
    page = ledger.prepare_page(_source_page(ledger), tool="read_source")
    partial = deepcopy(page)
    partial["spans"][0].update(end=2, text=partial["spans"][0]["text"][:2])
    _confirm(ledger, partial)
    assert ledger.confirmed_ranges() == {"E000002": [[0, 2]]}
    assert not ledger.is_confirmed(page[RECEIPT_FIELD])
    _confirm(ledger, page, "request-2")
    assert ledger.is_confirmed(page[RECEIPT_FIELD])
    assert ledger.confirmed_ranges() == {"E000002": [[0, len(page["spans"][0]["text"])]]}


def test_mismatched_original_text_is_never_accepted(tmp_path):
    ledger = _ledger(tmp_path)
    page = _source_page(ledger)
    page["spans"][0]["text"] = "模型补写的其他文字"
    with pytest.raises(ValueError, match="differs"):
        ledger.prepare_page(page, tool="read_source")


def test_initial_input_requires_actual_exact_message_and_survives_compression(tmp_path):
    ledger = _ledger(tmp_path)
    span = _source_page(ledger)["spans"][0]
    content = "请阅读原文：" + span["text"]
    identifier = ledger.register_initial_message(content, [span])
    assert ledger.pending_initial_messages() == [{"role": "user", "content": content}]
    ledger.observe_request({"messages": [{"role": "user", "content": "阅读摘要"}]}, {"request_id": "compressed"})
    ledger.confirm_request("compressed", evidence={"kind": "upstream_response", "response_id": "response-1"})
    assert not ledger.is_confirmed(identifier) and not ledger.confirmed_ranges()
    ledger.observe_request({"messages": ledger.pending_initial_messages()}, {"request_id": "recovered"})
    ledger.confirm_request("recovered", evidence={"kind": "upstream_response", "response_id": "response-2"})
    assert ledger.is_confirmed(identifier) and ledger.pending_initial_messages() == []
    assert ledger.confirmed_ranges()


def test_failed_unconfirmed_request_remains_pending_and_invalid_ack_is_rejected(tmp_path):
    ledger = _ledger(tmp_path)
    page = ledger.prepare_page(_source_page(ledger), tool="read_source")
    payload, audit = _request(page)
    ledger.observe_request(payload, audit)
    ledger.reject_request("request-1", "HTTP 400")
    assert not ledger.confirmed_ranges() and ledger.pending_page(page[RECEIPT_FIELD]) == page
    with pytest.raises(ValueError, match="upstream"):
        ledger.confirm_request("request-1", evidence={"http_status": 200})
    with pytest.raises(ValueError, match="matching"):
        ledger.confirm_request("request-1", evidence={"input_delivery_confirmed": True, "request_id": "other"})


def test_initial_recovery_wrapper_does_not_repeat_already_delivered_source(tmp_path):
    ledger = _ledger(tmp_path)
    span = _source_page(ledger)["spans"][0]
    original = "原始任务\n" + span["text"]
    recovery = original + "\n准确恢复提示"
    ledger.register_initial_message(original, [span])
    ledger.register_initial_message(recovery, [span])
    assert ledger.pending_initial_messages() == [{"role": "user", "content": original}]
    ledger.observe_request({"messages": [{"role": "user", "content": recovery}]}, {"request_id": "recovery"})
    ledger.confirm_request("recovery", evidence={"input_delivery_confirmed": True})
    assert ledger.pending_initial_messages() == []


@pytest.mark.parametrize("changed", ["run", "agent", "document", "manifest"])
def test_ledger_restoration_rejects_cross_identity(tmp_path, changed):
    first = _ledger(tmp_path)
    manifest, run, agent = deepcopy(first.manifest), "run-1", "child-1"
    if changed == "run": run = "run-2"
    elif changed == "agent": agent = "main"
    elif changed == "document": manifest = manifest_from_text("不同报告但仍有E000001")
    else: manifest["parser_version"] = "different-parser"
    with pytest.raises(ValueError, match="identity"):
        SourceDeliveryLedger(manifest, tmp_path, run_id=run, agent_id=agent)


def test_two_agents_delivery_is_not_silently_merged(tmp_path):
    child = _ledger(tmp_path / "child")
    main = SourceDeliveryLedger(child.manifest, tmp_path / "main", run_id="run-1", agent_id="main")
    page = child.prepare_page(_source_page(child), tool="read_source")
    _confirm(child, page)
    _confirm(main, page)
    assert child.confirmed_ranges() and not main.confirmed_ranges()
    child_tools = DocumentTools(child.manifest, count_tokens=len, max_tool_tokens=4000, delivery_ledger=child)
    child_tools.automatic_handlers()["read_source"]("E000002")
    main_tools = DocumentTools(main.manifest, count_tokens=len, max_tool_tokens=4000, delivery_ledger=main)
    with pytest.raises(ValueError, match="manifest/run/Agent"):
        main_tools.automatic_handlers(state=child_tools.navigation_state())


def test_navigation_replays_pending_page_after_restore_then_continues_after_ack(tmp_path):
    ledger = _ledger(tmp_path / "ledger", "很长的原文" * 400 + "\n尾段")
    saved = []
    tools = DocumentTools(ledger.manifest, count_tokens=len, max_tool_tokens=1600, delivery_ledger=ledger)
    read = tools.automatic_handlers(on_progress=lambda state: saved.append(deepcopy(state)))["read_section"]
    first = read("root", max_units=1)
    assert first["ok"] and first["continuation"]
    pending = read("root", max_units=1)
    assert pending["replayed"] and pending["spans"] == first["spans"]
    restored = SourceDeliveryLedger(ledger.manifest, tmp_path / "ledger", run_id="run-1", agent_id="child-1")
    again = DocumentTools(ledger.manifest, count_tokens=len, max_tool_tokens=1600, delivery_ledger=restored)
    read_again = again.automatic_handlers(state=saved[-1])["read_section"]
    assert read_again("root")[RECEIPT_FIELD] == first[RECEIPT_FIELD]
    _confirm(restored, pending)
    next_page = read_again("root", max_units=1)
    assert next_page["spans"][0]["start"] == first["spans"][-1]["end"]
    assert next_page[RECEIPT_FIELD] != first[RECEIPT_FIELD]


def test_same_native_call_and_same_generation_do_not_advance_source_twice(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path)
    current = _identity()
    monkeypatch.setattr(adapter, "current_tool_call_identity", lambda: deepcopy(current))
    tools = DocumentTools(ledger.manifest, count_tokens=len, max_tool_tokens=4000, delivery_ledger=ledger)
    read = tools.automatic_handlers()["read_section"]
    first = read("root", max_units=1)
    _confirm(ledger, first)
    assert read("root", max_units=1)["spans"] == first["spans"]
    current["native_tool_call_id"] = "native-2"
    assert read("root", max_units=1)["spans"] == first["spans"]
    current.update(generation_call_id="generation-2", native_tool_call_id="native-3")
    second = read("root", max_units=1)
    assert second["spans"][0]["evidence_id"] == "E000002"
    current.update(generation_call_id="generation-1", native_tool_call_id="native-1")
    assert read("root", max_units=1)["spans"] == first["spans"]
    assert list(tools.navigation_state()["objects"].values())[0]["pending_delivery_receipt"] == second[RECEIPT_FIELD]


def test_native_replay_cannot_change_arguments_and_failed_journal_commit_is_atomic():
    journal = ManagedNavigationJournal()
    identity = _identity()
    journal.record(identity, "object", {"ok": True, "page": 1, "replayed": False})
    with pytest.raises(ValueError, match="changed"):
        journal.replay({**identity, "args_sha256": "different"}, "object")
    def fail(_): raise OSError("disk full")
    failed = ManagedNavigationJournal(on_progress=fail)
    with pytest.raises(OSError): failed.record(identity, "object", {"ok": True})
    assert failed.replay(identity, "object") is None


def test_disk_failure_cannot_leave_in_memory_confirmation_or_advance_navigation(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path)
    page = ledger.prepare_page(_source_page(ledger), tool="read_source")
    payload, audit = _request(page)
    ledger.observe_request(payload, audit)
    def fail(*_): raise OSError("disk full")
    monkeypatch.setattr(delivery_module, "atomic_json", fail)
    with pytest.raises(OSError):
        ledger.confirm_request("request-1", evidence={"input_delivery_confirmed": True})
    assert not ledger.is_confirmed(page[RECEIPT_FIELD]) and not ledger.confirmed_ranges()
    tools = DocumentTools(ledger.manifest, count_tokens=len, max_tool_tokens=4000, delivery_ledger=ledger)
    before = tools.navigation_state()
    assert not tools.automatic_handlers()["read_source"]("E000003")["ok"]
    assert tools.navigation_state() == before


def test_failed_coverage_callback_recovers_from_durable_confirmation_idempotently(tmp_path):
    received = set()
    def fail_after_write(*span):
        received.add(span)
        raise OSError("callback interrupted")
    ledger = _ledger(tmp_path, on_confirmed=fail_after_write)
    page = ledger.prepare_page(_source_page(ledger), tool="read_source")
    with pytest.raises(OSError): _confirm(ledger, page)
    assert ledger.is_confirmed(page[RECEIPT_FIELD])
    restored = _ledger(tmp_path, on_confirmed=lambda *span: received.add(span))
    assert len(received) == 1 and restored.state()["requests"]["request-1"]["applied"]


def test_actual_receipt_token_cost_is_used_in_pagination_not_cheap_zero_placeholder(tmp_path):
    ledger = _ledger(tmp_path, "原文" * 3000)
    count = lambda text: len(text) - 63 * text.count("0" * 64)
    tools = DocumentTools(ledger.manifest, count_tokens=count, max_tool_tokens=1600,
                          native_response_budget=True, delivery_ledger=ledger)
    page = tools.automatic_handlers()["read_source"]("E000001")
    assert page["ok"] and page["continuation"]
    wire = json.dumps(adapter.wrap_tool_response("read_source", page), ensure_ascii=False,
                      sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert count(wire) <= 1600
    assert tools.receipts[-1]["result_tokens"] == count(wire)
    _confirm(ledger, page)
    following = tools.automatic_handlers()["read_source"]("E000001")
    assert following["spans"][0]["start"] == page["spans"][0]["end"]


def test_unexpected_final_page_growth_rolls_back_outbox_and_position(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path)
    original = ledger.prepare_page
    def grow(result, **kwargs):
        page = original(result, **kwargs)
        page["unexpected_envelope_growth"] = "X" * 10000
        return page
    monkeypatch.setattr(ledger, "prepare_page", grow)
    tools = DocumentTools(ledger.manifest, count_tokens=len, max_tool_tokens=2000,
                          native_response_budget=True, delivery_ledger=ledger)
    before = tools.navigation_state()
    result = tools.automatic_handlers()["read_source"]("E000002")
    assert not result["ok"] and "response budget" in result["error"]["message"]
    assert tools.navigation_state() == before
    assert ledger.state()["pages"] == {} and ledger.confirmed_ranges() == {}
    monkeypatch.setattr(ledger, "prepare_page", original)
    assert tools.automatic_handlers()["read_source"]("E000002")["ok"]
