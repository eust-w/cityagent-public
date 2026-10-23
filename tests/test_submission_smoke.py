"""No local fixture or self-declared verified bit can certify a live package."""

import importlib
import json
import sys

import pytest


def smoke():
    return importlib.import_module("scripts.smoke_submission")


def test_smoke_cannot_accept_a_bare_verified_claim(tmp_path):
    report = smoke().validate_smoke_artifacts(
        tmp_path, tmp_path / "inputs", tmp_path / "run"
    )
    assert not report["passed"]
    assert report["evidence_kind"] == "artifact-consistency-not-remote-attestation"
    assert any("run_manifest" in error for error in report["errors"])


def test_smoke_builds_generation_not_help_from_absolute_staged_entrypoint(tmp_path):
    command = smoke().build_command(
        code_dir=tmp_path / "code",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "smoke",
        model_manifest=tmp_path / "model.json",
        candidate_config=tmp_path / "config.json",
        sdk_acceptance=tmp_path / "sdk.json",
        python=sys.executable,
        backend="vllm-api",
        backend_url="http://127.0.0.1:8000/v1",
    )
    assert command[1] == str(tmp_path / "code/run.py")
    assert "--help" not in command and "--check" not in command
    assert command[command.index("--engine") + 1] == "tongagents"
    assert command[command.index("--rsi-mode") + 1] == "off"


def test_missing_environment_has_durable_failed_evidence_no_false_pass(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "run.py").write_text('raise RuntimeError("missing actual SDK")')
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "sample.doc").write_text("not a real doc")
    result = smoke().run_smoke(
        code_dir=code,
        input_dir=inputs,
        output_dir=tmp_path / "probe",
        model_manifest=tmp_path / "absent-model.json",
        candidate_config=tmp_path / "absent-config.json",
        sdk_acceptance=tmp_path / "absent-sdk.json",
        python=sys.executable,
        formal=False,
    )
    assert not result["passed"]
    assert result["formal_accepted"] is False
    saved = json.loads((tmp_path / "probe/smoke-evidence.json").read_text())
    assert saved["passed"] is False
    assert saved["errors"]


def test_formal_smoke_requires_existing_shared_budget_and_idle_window(tmp_path):
    result = smoke().run_smoke(
        code_dir=tmp_path,
        input_dir=tmp_path,
        output_dir=tmp_path / "probe",
        model_manifest=tmp_path / "model",
        candidate_config=tmp_path / "candidate",
        sdk_acceptance=tmp_path / "sdk",
        python=sys.executable,
        formal=True,
    )
    assert not result["passed"]
    assert any("shared budget" in e for e in result["errors"])
    assert not (tmp_path / "budget.sqlite").exists()


def test_manifest_denominator_cannot_silently_change(tmp_path):
    evaluate = importlib.import_module("scripts.evaluate_multitask")
    report = tmp_path / "report.docx"
    gold = tmp_path / "gold.json"
    report.write_bytes(b"source")
    gold.write_text("{}")
    from bridge_evidence.contest.run_state import file_digest

    case = {
        "id": "a",
        "report": str(report),
        "report_sha256": file_digest(report),
        "gold": str(gold),
        "gold_sha256": file_digest(gold),
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "purpose": "development-regression",
                "metric_denominator": 2,
                "samples": [case],
            }
        )
    )
    with pytest.raises(ValueError, match="denominator"):
        evaluate.load_cases(manifest)


def test_staged_timeout_terminates_only_its_owned_process_group(tmp_path):
    import os
    import subprocess

    command = [
        sys.executable,
        "-c",
        'import time; print("started", flush=True); time.sleep(60)',
    ]
    with (tmp_path / "log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            smoke().run_staged_process(
                command, cwd=tmp_path, env=os.environ.copy(), stdout=log, timeout=0.1
            )
    assert "started" in (tmp_path / "log").read_text()


def test_frozen_smoke_explicitly_sets_workflow_before_entrypoint_rsi_validation(
    tmp_path,
):
    command = smoke().build_command(
        code_dir=tmp_path,
        input_dir=tmp_path,
        output_dir=tmp_path / "smoke",
        model_manifest=tmp_path / "model",
        candidate_config=tmp_path / "config",
        sdk_acceptance=tmp_path / "sdk",
        python=sys.executable,
        workflow="bounded-react",
        rsi_bundle=tmp_path / "bundle",
        rsi_sandbox_root=tmp_path / "sandbox",
        rsi_release=tmp_path / "release",
        runtime_options={
            "max_new_tokens": 4096,
            "max_attempts": 3,
            "fusion": "model-preserving",
        },
    )
    assert command[command.index("--workflow") + 1] == "bounded-react"
    assert command[command.index("--max-new-tokens") + 1] == "4096"
    assert command[command.index("--max-attempts") + 1] == "3"


def test_smoke_ledger_binding_rejects_altered_generation_output():
    call = {
        "call_id": "actual-1",
        "status": "success",
        "seconds": 0.1,
        "prompt_sha256": "a" * 64,
        "output_sha256": "b" * 64,
        "output_tokens": 10,
        "finish_reason": "stop",
    }
    receipt = {**call, "output_sha256": "c" * 64, "backend_request_id": "backend-1"}
    errors = smoke().validate_ledger_call(call, 0.1, "success", receipt)
    assert any("output_sha256" in error for error in errors)
    receipt["output_sha256"] = call["output_sha256"]
    assert smoke().validate_ledger_call(call, 0.1, "success", receipt) == []
    call["status"] = "failed"
    assert smoke().validate_ledger_call(call, 0.1, "success", receipt) == []


@pytest.mark.parametrize(
    "status,field,value",
    [
        ("success", "seconds", 0.01),
        ("failed", "prompt_sha256", "x" * 64),
        ("failed", "output_sha256", "x" * 64),
    ],
)
def test_smoke_binds_failed_calls_and_actual_duration(status, field, value):
    call = {
        "call_id": "c",
        "status": status,
        "seconds": 1.5,
        "prompt_sha256": "a" * 64,
        "output_sha256": "b" * 64,
        "output_tokens": 1,
        "finish_reason": "stop" if status == "success" else "length",
    }
    receipt = {**call, "backend_request_id": "request"}
    call[field] = value
    assert smoke().validate_ledger_call(call, 1.5, status, receipt)


def test_completed_batch_timing_cannot_omit_failed_cost_or_undercount(tmp_path):
    from bridge_evidence.contest.execution_audit import audit_timing_records
    from bridge_evidence.contest.timing import TimingLedger

    records_dir, results = tmp_path / "work/records", tmp_path / "result"
    records_dir.mkdir(parents=True)
    results.mkdir()
    ledger = TimingLedger(records_dir.parent / "timing.sqlite")
    calls = []
    for call_id, seconds, status in [
        ("failed", 2.0, "failed"),
        ("good", 3.0, "success"),
    ]:
        call = {
            "call_id": call_id,
            "role": "extraction",
            "seconds": seconds,
            "status": status,
            "prompt_sha256": "a" * 64,
            "output_sha256": "b" * 64,
            "output_tokens": 12,
            "finish_reason": "length" if status == "failed" else "stop",
        }
        calls.append(call)
        ledger.start(call_id, role="extraction")
        ledger.finish(
            call_id,
            seconds,
            status=status,
            receipt={**call, "backend_request_id": call_id},
        )
    record = {"schema_version": 2, "seconds": 5.0, "model_calls": calls}
    (records_dir / "report.json").write_text(json.dumps(record))
    timing = results / "infer_time.json"
    timing.write_text('{"infer_time": 5.0}')
    assert audit_timing_records(results, records_dir)["passed"]
    timing.write_text('{"infer_time": 0.001}')
    assert not audit_timing_records(results, records_dir)["passed"]
    timing.write_text('{"infer_time": 3.0}')
    record["model_calls"] = calls[1:]
    record["seconds"] = 3.0
    (records_dir / "report.json").write_text(json.dumps(record))
    assert not audit_timing_records(results, records_dir)["passed"]


def test_strong_batch_audit_does_not_downgrade_legacy_receipts(tmp_path):
    from test_execution_audit import FINGERPRINT, make_record

    from bridge_evidence.contest.execution_audit import audit_execution_records

    source, output = tmp_path / "input/sample.docx", tmp_path / "result/sample.docx"
    source.parent.mkdir()
    output.parent.mkdir()
    source.write_bytes(b"unit source")
    output.write_bytes(b"unit output")
    records = tmp_path / "work/records"
    records.mkdir(parents=True)
    (records / "record.json").write_text(json.dumps(make_record(source, output)))
    assert audit_execution_records(source.parent, output.parent, records, FINGERPRINT)[
        "passed"
    ]
    strong = audit_execution_records(
        source.parent, output.parent, records, FINGERPRINT, require_strong=True
    )
    assert not strong["passed"]
    assert any("fresh original" in e or "raw_output" in e for e in strong["errors"])
    assert any("ledger" in e for e in strong["errors"])
