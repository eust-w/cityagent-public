"""Endpoint changes must invalidate current experiment comparison evidence."""

import json
from copy import deepcopy

import pytest

from bridge_evidence.contest import backends, rsi_acceptance
from bridge_evidence.contest.run_state import file_digest
from bridge_evidence.evaluation.generation import METRIC_VERSION


def identity(tmp_path):
    evaluator = tmp_path / "scripts/evaluate_multitask.py"
    evaluator.parent.mkdir()
    evaluator.write_text("# isolated evidence fixture\n")
    return {
        "code_files": {"run.py": "d" * 64},
        "options": {
            "protocol": "complete-multitask-report-v2",
            "metric_version": METRIC_VERSION,
            "evaluator_sha256": file_digest(evaluator),
            "hardware": "same-device", "warmup": 1,
            "backend": "vllm-api", "engine": "direct",
            "url": "http://127.0.0.1:8000/v1",
            "manifest": {"samples": [{"id": "a", "report_sha256": "a" * 64,
                                        "gold_sha256": "b" * 64}]},
            "model": {"files": {"base": {"weights": "c" * 64}}},
        },
    }


def test_comparison_identity_changes_when_service_endpoint_changes(tmp_path):
    before = identity(tmp_path)
    after = deepcopy(before)
    after["options"]["url"] = "http://127.0.0.1:8011/v1"
    assert rsi_acceptance.comparison_identity(before, source_root=tmp_path) != (
        rsi_acceptance.comparison_identity(after, source_root=tmp_path))


def test_current_formal_comparison_requires_explicit_endpoint(tmp_path):
    value = identity(tmp_path)
    del value["options"]["url"]
    with pytest.raises(ValueError, match="endpoint"):
        rsi_acceptance.comparison_identity(value, source_root=tmp_path)


@pytest.mark.parametrize("url", ["http://example.com/v1", "http://user:secret@127.0.0.1/v1",
                                  "http://127.0.0.1:8011/v1?token=secret", "https://127.0.0.1/v1"])
def test_formal_comparison_rejects_nonlocal_or_credentialed_endpoint(tmp_path, url):
    value = identity(tmp_path)
    value["options"]["url"] = url
    with pytest.raises(ValueError, match="endpoint"):
        rsi_acceptance.comparison_identity(value, source_root=tmp_path)


def test_normalized_endpoint_matches_actual_backend_url():
    url = "http://127.0.0.1:8011/v1/"
    assert backends.normalize_api_url(url) == backends.APIBackend(url, "model", None).url


@pytest.mark.parametrize("other", [None, "http://127.0.0.1:8011/v1"])
def test_recovery_from_another_or_unbound_endpoint_is_rejected(tmp_path, other):
    from test_rsi_receipt_acceptance import recovery_artifacts

    path, _, value, root = recovery_artifacts(tmp_path)
    # Unit fixture supplies real durable events and ledger, not release proof.
    if other is None:
        del value["options"]["url"]
    else:
        value["options"]["url"] = other
    with pytest.raises(ValueError, match="endpoint"):
        rsi_acceptance.audit_recovery_evidence(
            path, identity=value, bundle_digest=None, source_root=root)


@pytest.mark.parametrize("recorded,expected,accepted", [
    (None, "http://127.0.0.1:8000/v1", False),
    ("http://127.0.0.1:8011/v1", "http://127.0.0.1:8000/v1", False),
    ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1", True),
    ("http://127.0.0.1:8011/v1", "http://127.0.0.1:8011/v1", True),
    ("http://127.0.0.1:8011/v1/", "http://127.0.0.1:8011/v1", True),
])
def test_parallel_proof_requires_matching_explicit_endpoint(tmp_path, monkeypatch, recorded, expected, accepted):
    from pathlib import Path

    from bridge_evidence.contest import execution_audit, rsi_pipeline

    # Isolate endpoint admission; a unit fixture is never real SDK acceptance.
    monkeypatch.setattr(execution_audit, "_native_errors", lambda _: [])
    sdk = tmp_path / "sdk.json"
    sdk.write_text("{}")
    receipts = []
    for index in range(2):
        output = tmp_path / f"out-{index}.docx"
        output.write_bytes(b"unit-output")
        receipts.append({"document_sha256": str(index), "workflow": {},
                         "model_call_ids": [str(index)], "output_path": str(output),
                         "output_sha256": file_digest(output)})
    proof = {"schema_version": 1, "sdk_acceptance_sha256": file_digest(sdk),
             "runtime_sha256": file_digest(Path(__file__).resolve().parents[1] / "run.py"),
             "protected_sha256": rsi_pipeline.parallel_runtime_digest(),
             "overlap": True, "receipts": receipts}
    if recorded is not None:
        proof["url"] = recorded
    path = tmp_path / "parallel.json"
    path.write_text(json.dumps(proof))
    assert rsi_pipeline.verify_parallel_sdk(path, sdk, expected) is accepted
