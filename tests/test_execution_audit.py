from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from bridge_evidence.contest.execution_audit import (
    STAGES,
    audit_execution_records,
    validate_execution_record,
)

FINGERPRINT = "f" * 64


def make_record(input_file: Path, output_file: Path, *, engine="tongagents") -> dict:
    """Synthetic schema fixture only; it is not real SDK acceptance evidence."""
    call = {"call_id": "call-1", "status": "success", "prompt_sha256": "a" * 64,
            "output_sha256": "b" * 64, "prompt_tokens": 123, "output_tokens": 45,
            "finish_reason": "stop", "seconds": 0.25, "errors": []}
    stages = [{"stage": stage, "status": "success", "duration_seconds": 0.01} for stage in STAGES]
    stages[1].update(model_call_ids=["call-1"], accepted_model_call_ids=["call-1"])
    return {"schema_version": 1,
            "document": {"relative_path": input_file.name, "sha256": hashlib.sha256(input_file.read_bytes()).hexdigest()},
            "run_fingerprint": FINGERPRINT, "status": "success", "seconds": 0.5,
            "model_validation": {"valid": True}, "model_calls": [call],
            "workflow": {"engine": engine, "non_sdk": engine == "direct", "status": "success",
                "sdk_version": "1.2.3", "verification": "verified", "native_trace_validated": True,
                "native_events": [{"type": kind, "data": {"node_name": stage}}
                    for stage in STAGES for kind in ("node_start", "node_end")], "stage_records": stages},
            "output": {"name": output_file.name, "sha256": hashlib.sha256(output_file.read_bytes()).hexdigest()}}


@pytest.fixture
def receipt(tmp_path):
    source = tmp_path / "input" / "bridge.doc"
    source.parent.mkdir()
    source.write_bytes(b"original report")
    output = tmp_path / "result" / "bridge.docx"
    output.parent.mkdir()
    output.write_bytes(b"output bytes; DOCX structure checked by package audit")
    record = make_record(source, output)
    return source, output, record


def check(receipt, record=None, **kwargs):
    _, output, original = receipt
    return validate_execution_record(original if record is None else record,
        expected_document=original["document"], expected_run_fingerprint=FINGERPRINT,
        output_path=output, **kwargs)


def test_complete_schema_can_be_checked_without_claiming_real_sdk_test(receipt):
    assert check(receipt) == []


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("schema_version", True), ("run_fingerprint", "e" * 64),
    ("seconds", float("nan")), ("seconds", True), ("model_validation", {"valid": False}),
    ("model_validation", {"valid": True, "truncated": True}), ("model_calls", []),
])
def test_invalid_top_level_never_passes(receipt, field, value):
    record = deepcopy(receipt[2])
    record[field] = value
    assert check(receipt, record)


@pytest.mark.parametrize("field,value", [
    ("finish_reason", "length"), ("output_tokens", 0), ("output_tokens", True),
    ("prompt_tokens", 0), ("output_sha256", None), ("prompt_sha256", "old-filename"),
    ("seconds", 0), ("mock", True), ("simulated", True), ("errors", ["invalid_json"]),
])
def test_invalid_model_call_never_passes(receipt, field, value):
    record = deepcopy(receipt[2])
    record["model_calls"][0][field] = value
    assert check(receipt, record)


@pytest.mark.parametrize("field,value", [
    ("engine", "direct"), ("sdk_version", None), ("verification", "unverified"),
    ("verification", "mock"), ("non_sdk", True), ("native_trace_validated", False),
    ("status", "error"), ("native_events", []),
])
def test_missing_unverified_or_direct_sdk_never_passes_contest(receipt, field, value):
    record = deepcopy(receipt[2])
    record["workflow"][field] = value
    assert check(receipt, record)


def test_native_failure_order_and_call_links_are_required(receipt):
    for mutate in (
        lambda w: w["native_events"].pop(),
        lambda w: w["native_events"].reverse(),
        lambda w: w["native_events"].append({"type": "node_exception", "data": {"node_name": "infer"}}),
        lambda w: w["native_events"].insert(0, {"type": "node_process", "data": {"node_name": "infer"}}),
        lambda w: w["stage_records"][1].update(model_call_ids=["unrelated"]),
        lambda w: w["stage_records"][1].update(accepted_model_call_ids=[]),
        lambda w: w["stage_records"][2].update(status="error"),
    ):
        record = deepcopy(receipt[2])
        mutate(record["workflow"])
        assert check(receipt, record)


def test_failed_retry_is_history_and_cannot_be_accepted(receipt):
    record = deepcopy(receipt[2])
    failed = deepcopy(record["model_calls"][0])
    failed.update(call_id="failed-attempt", status="failed", finish_reason="length", errors=["truncated"])
    record["model_calls"].insert(0, failed)
    stage = record["workflow"]["stage_records"][1]
    stage["model_call_ids"].insert(0, "failed-attempt")
    assert check(receipt, record) == []
    stage["accepted_model_call_ids"].insert(0, "failed-attempt")
    assert check(receipt, record)
    record["model_calls"].reverse()
    assert check(receipt, record)


def test_legacy_is_explicit_and_never_sdk_passed(receipt, tmp_path):
    source, output, _ = receipt
    record = make_record(source, output, engine="direct")
    record["workflow"].update(verification="non-sdk-baseline", native_events=[], native_trace_validated=False, sdk_version=None)
    assert check(receipt, record, profile="legacy") == []
    assert check(receipt, record)
    records = tmp_path / "records"
    records.mkdir()
    (records / "receipt.json").write_text(json.dumps(record))
    audit = audit_execution_records(source.parent, output.parent, records, FINGERPRINT, profile="legacy")
    assert audit["passed"] is True
    assert audit["sdk_passed"] is False


def test_output_bytes_input_bytes_and_stale_receipts_are_bound(receipt, tmp_path):
    source, output, record = receipt
    records = tmp_path / "records"
    records.mkdir()
    record_file = records / "content-and-path-digest.json"
    record_file.write_text(json.dumps(record))
    assert audit_execution_records(source.parent, output.parent, records, FINGERPRINT)["passed"]
    output.write_bytes(b"different report")
    assert not audit_execution_records(source.parent, output.parent, records, FINGERPRINT)["passed"]
    source.write_bytes(b"same filename changed input")
    audit = audit_execution_records(source.parent, output.parent, records, FINGERPRINT)
    assert not audit["passed"]
    assert any("stale" in error for error in audit["errors"])


def test_old_filename_cache_and_duplicate_receipts_cannot_pass(receipt, tmp_path):
    source, output, record = receipt
    records = tmp_path / "records"
    records.mkdir()
    (records / "bridge.json").write_text(json.dumps({"source_path": str(source), "data": {}}))
    assert not audit_execution_records(source.parent, output.parent, records, FINGERPRINT)["passed"]
    (records / "bridge.json").write_text(json.dumps(record))
    (records / "duplicate.json").write_text(json.dumps(record))
    assert not audit_execution_records(source.parent, output.parent, records, FINGERPRINT)["passed"]


def test_empty_inputs_and_output_collisions_are_rejected(receipt, tmp_path):
    source, output, record = receipt
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not audit_execution_records(empty, output.parent, empty, FINGERPRINT)["passed"]
    source.with_suffix(".docx").write_bytes(source.read_bytes())
    records = tmp_path / "records"
    records.mkdir()
    (records / "one.json").write_text(json.dumps(record))
    result = audit_execution_records(source.parent, output.parent, records, FINGERPRINT)
    assert any("collision" in error for error in result["errors"])


def test_malformed_receipts_fail_without_raising(receipt):
    for record in (None, [], "old state", {"schema_version": 1}, {"document": []}):
        assert validate_execution_record(
            record, expected_document=receipt[2]["document"], expected_run_fingerprint=FINGERPRINT,
            output_path=receipt[1])
    assert validate_execution_record(receipt[2], expected_document={"relative_path": []},
        expected_run_fingerprint=FINGERPRINT, output_path=receipt[1])


def test_saved_model_response_must_match_final_successful_call(receipt):
    record = deepcopy(receipt[2])
    record["raw_output"] = '{"model": "response"}'
    record["model_calls"][-1]["output_sha256"] = hashlib.sha256(record["raw_output"].encode()).hexdigest()
    assert check(receipt, record) == []
    record["raw_output"] += "changed"
    assert check(receipt, record)
    record["raw_output"] = "\ud800"
    assert check(receipt, record)


def test_native_end_control_node_is_allowed_only_after_render(receipt):
    record = deepcopy(receipt[2])
    events = record["workflow"]["native_events"]
    end = [{"type": kind, "data": {"node_name": "__end__"}} for kind in ("node_start", "node_end")]
    record["workflow"]["native_events"] = events + end
    assert check(receipt, record) == []
    record["workflow"]["native_events"] = events[:2] + end + events[2:]
    assert check(receipt, record)
    record["workflow"]["native_events"] = events + end[:1]
    assert check(receipt, record)


def test_recursive_inputs_match_relative_path_and_ignore_word_lockfiles(receipt, tmp_path):
    source, output, record = receipt
    nested = source.parent / "2026"
    nested.mkdir()
    moved = nested / source.name
    source.rename(moved)
    (nested / "~$bridge.docx").write_text("Word temporary lock")
    record["document"]["relative_path"] = "2026/bridge.doc"
    records = tmp_path / "records"
    records.mkdir()
    (records / "nested.json").write_text(json.dumps(record))
    audit = audit_execution_records(source.parent, output.parent, records, FINGERPRINT)
    assert audit["passed"] is True, audit["errors"]
    assert audit["expected_inputs"] == 1


def test_fact_graph_cannot_be_rebound_by_changing_only_hash(receipt):
    from bridge_evidence.contest.facts import build_fact_graph
    from bridge_evidence.contest.run_state import json_digest
    record = deepcopy(receipt[2])
    record['final_data'] = {'基本信息': {'桥梁名称': '测试桥'}, '维修建议': [], '病害明细': []}
    record['facts'] = build_fact_graph(record).to_dict()
    record['facts_sha256'] = json_digest(record['facts'])
    assert check(receipt, record) == []
    record['facts']['three_table_sha256'] = 'tables:' + '0' * 64
    record['facts_sha256'] = json_digest(record['facts'])
    assert any('derived graph' in e for e in check(receipt, record))
