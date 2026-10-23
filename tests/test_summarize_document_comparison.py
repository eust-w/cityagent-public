"""Offline synthetic receipts and DOCX files only; no models or real labels."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

from docx import Document
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("document_summary", ROOT / "experiments/summarize_document_comparison.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
from bridge_evidence.contest.document_workspace import DocumentWorkspace
from bridge_evidence.contest.docx_writer import write_extraction_docx


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return audit.file_digest(path)


class Run:
    def __init__(self, root, count=2, *, review_version="bidirectional-semantic-review-v2", workflow_review_version=None):
        self.review_version = review_version
        self.workflow_review_version = workflow_review_version or review_version
        self.root, self.work, self.result, self.code = root, root / "work", root / "result", root / "code"
        self.batch, self.status = root / "inputs/manifest.json", root / "status.json"
        self.result.mkdir()
        self.entries, self.sources, self.manifests, self.reports = [], [], {}, {}
        for n in range(count):
            name = f"TRAIN-{n + 1}"
            value = {"evidence_id": name, "training_line": n + 1, "text": f"合成{name}桥。\n历史状态未知。",
                     "source_sha256": audit.digest({"original": name})}
            path = self.batch.parent / (name + ".json")
            sha = save(path, value)
            self.entries.append({"path": path.name, "sha256": sha})
            self.sources.append({"path": path.name, "sha256": sha, "source_path": str(path), "evidence_id": name})
            manifest = audit.manifest_from_text(value["text"])
            manifest.update(training_source={k: v for k, v in value.items() if k != "text"}, source_json_sha256=sha)
            self.manifests[name] = manifest
            self.reports[name] = {"基本信息": {"桥梁名称": name + "桥", "上一次定检结果": "未知"}, "维修建议": [], "病害明细": []}
        batch_sha = save(self.batch, {"schema_version": 1, "scope": audit.SCOPE, "entries": self.entries})
        code = {}
        for relative in ("run.py", *["src/bridge_evidence/contest/" + name for name in (
                "document_workflow.py", "document_workspace.py", "document_delivery_validation.py", "tong_react_adapter.py")]):
            path = self.code / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic hash identity, not executable experiment code\n")
            code[relative] = audit.file_digest(path)
        self.identity = {"workflow": "document-agent", "engine": "tongagents", "code": code,
                         "versions": {"review_version": review_version},
                         "settings": {"report_timeout": 100, "max_model_calls": 20},
                         "document_total_report_timeout": 110, "served_model": "synthetic-own-model",
                         "sources": {s["source_path"]: s["sha256"] for s in self.sources},
                         "training_evidence_batch": {"scope": audit.SCOPE, "manifest_sha256": batch_sha, "entries": self.sources}}
        self.fingerprint = audit.digest(self.identity)
        save(self.work / "run_manifest.json", {"schema_version": 1, "run_fingerprint": self.fingerprint, "identity": self.identity})
        self.documents = {}
        for source in self.sources:
            self.fail(source["evidence_id"])
        self.finish()

    def directory(self, name):
        return self.work / "documents" / name

    def fail(self, name):
        self.documents[name] = {"name": name, "status": "failed", "seconds": 4, "reused": False}
        save(self.directory(name) / "document_summary.json", self.documents[name])

    def complete(self, name):
        directory, manifest, report = self.directory(name), self.manifests[name], self.reports[name]
        source_sha = manifest["source_json_sha256"]
        identity = {"source": audit.digest(manifest), "settings": self.identity["settings"], "model": self.identity["served_model"],
                    "versions": {"review": self.workflow_review_version},
                    "run_identity": {"run_fingerprint": self.fingerprint, "source_sha256": source_sha}}
        workspace = DocumentWorkspace(manifest, directory / "workspace", identity)
        for unit in manifest["units"]:
            workspace.record_read(unit["id"], 0, len(unit["text"]))
        state = workspace.snapshot()
        refs = [{"evidence_id": u["id"], "start": 0, "end": len(u["text"])} for u in manifest["units"]]
        state = workspace.apply_update({"base_version": state["version"], "facts": [
            {"id": "F1", "value": name + "桥", "source_refs": refs, "status": "explicit"}],
            "draft": report, "draft_fact_ids": ["F1"],
            "draft_bindings": {"基本信息.桥梁名称": ["F1"], "基本信息.上一次定检结果": ["F1"]}})
        state = workspace.record_review({"draft_sha256": state["draft_sha256"], "workspace_version": state["version"],
            "checked_evidence_ids": [u["id"] for u in manifest["units"]], "model_call_ids": ["review"], "passed": True,
            "protocol": self.workflow_review_version, "unsupported_claims": [], "missing_items": [],
            "conflicts": [], "unresolved_items": []})
        calls, dispatches, gateways = [], [], []
        for call_id, kind in (("controller", "agent_generation"), ("review", "generation")):
            request_id, input_sha = name + "-" + call_id, audit.digest({"request": call_id})
            response = {"choices": [{"finish_reason": "stop", "message": {"content": "synthetic"}}],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 5}}
            calls.append({"call_id": call_id, "request_id": request_id, "kind": kind, "status": "success", "http_status": 200,
                          "endpoint": "/v1/chat/completions", "input_sha256": input_sha,
                          "finish_reason": "stop", "usage": response["usage"]})
            dispatches.append({"request_id": request_id, "request_sha256": input_sha, "status": "settled"})
            gateways.append({"request_id": request_id, "call_id": call_id, "request_sha256": input_sha,
                             "status": "success", "generation_attempted": True, "finish_reason": "stop", "usage": response["usage"],
                             "prompt_tokens_checked": 20, "response_sha256": audit.digest(response)})
            save(directory / "model_responses" / (audit.digest({"call_id": call_id}) + ".json"), {
                "workflow_identity_sha256": audit.digest(identity), "call_id": call_id, "request_id": request_id,
                "request_sha256": input_sha, "response_sha256": audit.digest(response), "response": response})
        sessions = [{"status": "completed", "native_final_event": True,
                     "tool_calls": [{"name": "finish", "status": "success"}]}]
        trace = {"identity": identity, "status": "completed", "model_calls": calls, "model_dispatches": dispatches,
                 "gateway_receipts": gateways, "sessions": sessions, "delivery": workspace.delivery_status(),
                 "accepted_finish": {"workspace_version": state["version"], "draft_sha256": state["draft_sha256"]}}
        save(directory / "document_workflow.json", trace)
        save(directory / "source_manifest.json", manifest)
        binding_sha = save(directory / "source_binding.json", {"run_fingerprint": self.fingerprint,
            "original_source_sha256": source_sha, "manifest_sha256": audit.digest(manifest), "document": None})
        output = self.result / (name + ".docx")
        write_extraction_docx(report, str(output))
        execution = {**trace, "sdk_sessions": sessions, "cli_status": "completed", "run_fingerprint": self.fingerprint,
            "source_sha256": source_sha, "source_binding_sha256": binding_sha, "report_sha256": audit.digest(report),
            "output": {"name": output.name, "sha256": audit.file_digest(output)}, "final_data": report,
            "workspace_version": state["version"], "draft_sha256": state["draft_sha256"], "seconds": 10,
            "delivery_validation": audit.validate_delivery_docx(report, output)}
        save(directory / "report.json", report)
        execution_sha = save(directory / "execution.json", execution)
        save(directory / "publication.json", {"run_fingerprint": self.fingerprint, "status": "committed", "execution_sha256": execution_sha})
        self.documents[name] = {"name": name, "status": "completed", "seconds": 12, "reused": False,
                                "output": output.name, "execution_sha256": execution_sha}
        save(directory / "document_summary.json", self.documents[name])
        self.finish()

    def rebind_execution(self, name):
        directory = self.directory(name)
        sha = audit.file_digest(directory / "execution.json")
        publication = audit.read_json(directory / "publication.json")
        publication["execution_sha256"] = sha
        save(directory / "publication.json", publication)
        self.documents[name]["execution_sha256"] = sha
        save(directory / "document_summary.json", self.documents[name])
        self.finish()

    def finish(self):
        complete = sum(r["status"] == "completed" for r in self.documents.values())
        summary = {"workflow": "document-agent", "run_fingerprint": self.fingerprint, "attempt_id": "attempt1",
            "reports": len(self.documents), "documents": list(self.documents.values()), "completed_reports": complete,
            "failed_reports": len(self.documents) - complete, "unfinished_reports": 0, "reused_reports": 0,
            "seconds": 20, "cumulative_seconds": 20, "cumulative_timing_complete": True, "attempts_with_unknown_wall_time": [],
            "started_at": "2026-10-13T00:00:02+00:00", "training_evidence_manifest_sha256": audit.file_digest(self.batch)}
        save(self.work / "summary.json", summary)
        save(self.work / "attempts/attempt1.summary.json", summary)
        save(self.work / "attempts/attempt1.preflight.json", {"run_fingerprint": self.fingerprint, "model_calls": []})
        save(self.status, {"status": "completed" if complete == len(self.documents) else "failed",
            "returncode": 0 if complete == len(self.documents) else 1, "summary": summary,
            "started_at": "2026-10-13T00:00:00+00:00", "finished_at": "2026-10-13T00:00:25+00:00", "wall_seconds": 25})

    def scoring(self):
        entries = []
        for name, report in self.reports.items():
            path = self.root / "labels" / (name + ".json")
            entries.append({"evidence_id": name, "path": path.name, "sha256": save(path, report)})
        path = self.root / "labels/manifest.json"
        save(path, {"schema_version": 1, "never_include_in_model_inputs": True, "entries": entries})
        return path

    def semantic(self):
        entries = []
        for name, report in self.reports.items():
            manifest = self.manifests[name]
            entries.append({"evidence_id": name, "input_json_sha256": manifest["source_json_sha256"],
                "original_report_sha256": manifest["training_source"]["source_sha256"], "report_sha256": audit.digest(report),
                "reviewer": "synthetic-reviewer", "coverage": "complete_supplied_text", "findings": [{"category": "A"}],
                "omissions": [], "accepted": True})
        path = self.root / "semantic.json"
        save(path, {"schema_version": 1, "independent": True, "entries": entries})
        return path

    def summarize(self, **kwargs):
        return audit.summarize(batch_manifest=self.batch, work_dir=self.work, result_dir=self.result,
                               code_root=self.code, supervisor_status=self.status, **kwargs)


def test_no_labels_does_not_invent_quality_or_semantic_acceptance(tmp_path):
    run = Run(tmp_path)
    run.complete("TRAIN-1")
    result = run.summarize()
    assert result["selected_reports"] == 2
    assert result["internal_completed_reports"] == result["content_fidelity_verified_reports"] == 1
    assert result["scoring"]["status"] == "not_scored" and result["metrics"] is None
    assert result["throughput"]["internally_completed_reports_per_hour"] == 144
    assert result["throughput"]["accepted_reports_per_hour"] is None
    assert result["generations"]["known_http_attempts"] == 2  # One SDK session, two actual receipts.
    assert result["generations"]["complete"] is False  # Failed report has no ledger, not zero invented calls.
    assert "images are not verified" in str(result["limitations"])


def test_failed_and_missing_predictions_retain_all_label_false_negatives(tmp_path):
    run = Run(tmp_path, 3)
    run.complete("TRAIN-1")
    (run.directory("TRAIN-2") / "document_summary.json").unlink()
    result = run.summarize(scoring_manifest=run.scoring())
    assert result["metrics"]["reports_total"] == 3
    assert result["metrics"]["reports_valid"] == 1
    assert result["metrics"]["overall"]["fn"] == 4
    assert result["metrics"]["overall"]["f1"] == .5


def test_all_failed_is_scored_zero_without_dropping_denominator(tmp_path):
    run = Run(tmp_path, 3)
    result = run.summarize(scoring_manifest=run.scoring())
    assert result["metrics"]["reports_total"] == result["failed_missing_or_invalid_reports"] == 3
    assert result["metrics"]["overall"]["f1"] == 0
    assert result["metrics"]["overall"]["fn"] == 6
    assert result["throughput"]["internally_completed_reports_per_hour"] == 0
    assert result["throughput"]["accepted_reports_per_hour"] is None


@pytest.mark.parametrize("artifact", ["input", "code", "report", "docx", "binding", "publication", "missing_receipt", "sidecar", "stale_review", "stale_finish"])
def test_tampered_or_missing_artifacts_never_become_valid_predictions(tmp_path, artifact):
    run = Run(tmp_path, 1)
    run.complete("TRAIN-1")
    directory = run.directory("TRAIN-1")
    if artifact in {"input", "code", "report", "docx", "binding"}:
        path = {"input": run.batch.parent / "TRAIN-1.json", "code": run.code / "run.py", "report": directory / "report.json",
                "docx": run.result / "TRAIN-1.docx", "binding": directory / "source_binding.json"}[artifact]
        if artifact == "report":
            value = audit.read_json(path)
            value["基本信息"]["桥梁名称"] = "无依据改名"
            save(path, value)
        else:
            path.write_bytes(path.read_bytes() + b" ")
    elif artifact == "missing_receipt":
        (directory / "execution.json").unlink()
    elif artifact == "sidecar":
        path = directory / "model_responses" / (audit.digest({"call_id": "review"}) + ".json")
        value = audit.read_json(path)
        value["response"]["choices"][0]["message"]["content"] = "altered"
        save(path, value)
    else:
        filename = {"publication": "publication.json", "stale_review": "workspace/workspace.json",
                    "stale_finish": "document_workflow.json"}[artifact]
        path = directory / filename
        value = audit.read_json(path)
        if artifact == "publication":
            value["status"] = "prepared"
        elif artifact == "stale_review":
            value["review"]["recorded_version"] -= 1
        else:
            value["accepted_finish"]["workspace_version"] -= 1
        save(path, value)
    result = run.summarize(scoring_manifest=run.scoring())
    assert result["metrics"]["reports_total"] == 1
    assert result["metrics"]["reports_valid"] == 0, result
    assert result["metrics"]["overall"]["fn"] == 2


def test_docx_readback_catches_changed_cells_even_if_output_hash_is_rebound(tmp_path):
    run = Run(tmp_path, 1)
    run.complete("TRAIN-1")
    output = run.result / "TRAIN-1.docx"
    document = Document(output)
    document.tables[0].rows[1].cells[1].text = "Changed bridge"
    document.save(output)
    path = run.directory("TRAIN-1") / "execution.json"
    execution = audit.read_json(path)
    execution["output"]["sha256"] = audit.file_digest(output)
    save(path, execution)
    run.rebind_execution("TRAIN-1")
    result = run.summarize(scoring_manifest=run.scoring())
    assert result["internal_completed_reports"] == 1
    assert result["content_fidelity_verified_reports"] == 0
    assert result["metrics"]["reports_valid"] == 0


@pytest.mark.parametrize("mode", ["report", "export", "http"])
def test_late_success_cannot_pass_completion_gate(tmp_path, mode):
    run = Run(tmp_path, 1)
    run.complete("TRAIN-1")
    directory = run.directory("TRAIN-1")
    if mode == "export":
        run.documents["TRAIN-1"]["seconds"] = 111
        save(directory / "document_summary.json", run.documents["TRAIN-1"])
        run.finish()
    else:
        execution = audit.read_json(directory / "execution.json")
        if mode == "report":
            execution["seconds"] = 101
        else:
            execution["gateway_receipts"][-1]["status"] = "failed"
            trace = audit.read_json(directory / "document_workflow.json")
            trace["gateway_receipts"] = execution["gateway_receipts"]
            save(directory / "document_workflow.json", trace)
        save(directory / "execution.json", execution)
        run.rebind_execution("TRAIN-1")
    result = run.summarize(scoring_manifest=run.scoring())
    assert result["metrics"]["reports_valid"] == 0


@pytest.mark.parametrize("mode", ["old_status", "missing_preflight", "unknown_attempt", "extra_attempt", "clock_mismatch", "unbound_supervisor"])
def test_incomplete_wall_time_is_unknown_not_summed_model_seconds(tmp_path, mode):
    run = Run(tmp_path, 1)
    run.complete("TRAIN-1")
    status = audit.read_json(run.status)
    if mode == "old_status":
        status.pop("started_at")
    elif mode == "missing_preflight":
        (run.work / "attempts/attempt1.preflight.json").unlink()
    elif mode == "extra_attempt":
        save(run.work / "attempts/earlier.preflight.json", {"run_fingerprint": run.fingerprint})
    elif mode == "clock_mismatch":
        status["wall_seconds"] = 1
    elif mode == "unbound_supervisor":
        status.pop("summary")
    else:
        summary = audit.read_json(run.work / "summary.json")
        summary["cumulative_timing_complete"] = False
        summary["attempts_with_unknown_wall_time"] = ["earlier"]
        save(run.work / "summary.json", summary)
        status["summary"] = summary
    save(run.status, status)
    result = run.summarize()
    assert result["internal_completed_reports"] == 1
    assert result["timing"]["status"] == "unknown"
    assert result["throughput"]["internally_completed_reports_per_hour"] is None


@pytest.mark.parametrize("mode", ["valid", "stale", "wrong_original", "unmarked_B", "not_allowed_B", "C", "D", "omission"])
def test_external_semantic_acceptance_is_bound_and_not_manufactured(tmp_path, mode):
    run = Run(tmp_path, 1)
    run.complete("TRAIN-1")
    path = run.semantic()
    value = audit.read_json(path)
    row = value["entries"][0]
    if mode == "stale":
        row["report_sha256"] = "0" * 64
    elif mode == "wrong_original":
        row["original_report_sha256"] = "0" * 64
    elif mode == "omission":
        row["omissions"] = [{"evidence_id": "E000001", "reason": "synthetic omission"}]
    elif mode != "valid":
        row["findings"] = [{"category": mode if mode in {"C", "D"} else "B",
                             "marked_as_inference": mode != "unmarked_B",
                             "allowed_by_output_contract": mode != "not_allowed_B"}]
    save(path, value)
    result = run.summarize(semantic_manifest=path)
    assert result["throughput"]["accepted_reports_per_hour"] == (144 if mode == "valid" else None)
    assert result["metrics"] is None


def test_invalid_label_manifest_keeps_unscored_batch_instead_of_partial_metric(tmp_path):
    run = Run(tmp_path, 2)
    path = run.scoring()
    value = audit.read_json(path)
    value["entries"].pop()
    save(path, value)
    result = run.summarize(scoring_manifest=path)
    assert result["selected_reports"] == 2 and result["metrics"] is None
    assert result["scoring"]["status"] == "invalid"


def test_pending_dispatch_is_unknown_not_an_extra_actual_call(tmp_path):
    run = Run(tmp_path, 1)
    directory = run.directory("TRAIN-1")
    save(directory / "document_workflow.json", {"model_calls": [], "gateway_receipts": [],
         "model_dispatches": [{"request_id": "reserved-before-crash", "status": "pending"}]})
    result = run.summarize()
    assert result["generations"]["known_http_attempts"] == 0
    assert result["generations"]["outcome_unknown_dispatches"] == 1
    assert result["generations"]["complete"] is False


def test_summary_cannot_shrink_selected_denominator(tmp_path):
    run = Run(tmp_path, 2)
    run.complete("TRAIN-1")
    run.documents.pop("TRAIN-2")
    run.finish()
    result = run.summarize(scoring_manifest=run.scoring())
    assert result["selected_reports"] == result["metrics"]["reports_total"] == 2
    assert result["metrics"]["reports_valid"] == 0
    assert "summary_drops_selected_reports" in result["run_errors"]


def test_no_model_input_discovery_or_unrequested_label_read(tmp_path, monkeypatch):
    run = Run(tmp_path, 1)
    forbidden = run.batch.parent / "unlisted-labels.json"
    forbidden.write_text("Do not read")
    original = audit.read_json
    def checked(path):
        assert Path(path) != forbidden
        return original(path)
    monkeypatch.setattr(audit, "read_json", checked)
    result = run.summarize()
    assert result["selected_reports"] == 1


def test_missing_source_retains_recorded_selected_identity(tmp_path):
    run = Run(tmp_path, 2)
    (run.batch.parent / "TRAIN-1.json").unlink()
    result = run.summarize(scoring_manifest=run.scoring())
    assert result["metrics"]["reports_total"] == 2
    assert result["documents"][0]["evidence_id"] == "TRAIN-1"
    assert result["metrics"]["overall"]["fn"] == 4


@pytest.mark.parametrize("bad", [None, ["broken"], [{"kind": "generation"}], "invalid"])
def test_malformed_call_receipts_preserve_batch_denominator(tmp_path, bad):
    run = Run(tmp_path, 2)
    directory = run.directory("TRAIN-1")
    save(directory / "document_workflow.json", {"model_calls": bad, "gateway_receipts": [], "model_dispatches": []})
    result = run.summarize(scoring_manifest=run.scoring())
    assert result["selected_reports"] == result["metrics"]["reports_total"] == 2
    assert result["metrics"]["overall"]["fn"] == 4
    assert result["generations"]["complete"] is False


def test_missing_run_metadata_does_not_drop_explicit_selection(tmp_path):
    run = Run(tmp_path, 2)
    (run.work / "run_manifest.json").unlink()
    result = run.summarize(scoring_manifest=run.scoring())
    assert result["selected_reports"] == result["metrics"]["reports_total"] == 2
    assert result["metrics"]["reports_valid"] == 0


def test_recoverable_pre_generation_rejection_is_not_a_generation_call(tmp_path):
    run = Run(tmp_path, 1)
    run.complete("TRAIN-1")
    directory = run.directory("TRAIN-1")
    for filename in ("document_workflow.json", "execution.json"):
        value = audit.read_json(directory / filename)
        value["gateway_receipts"].insert(0, {"status": "failed", "prompt_tokens_checked": 90000})
        save(directory / filename, value)
    run.rebind_execution("TRAIN-1")
    result = run.summarize()
    assert result["internal_completed_reports"] == 1
    assert result["generations"]["known_http_attempts"] == 2
    assert result["generations"]["complete"] is True


@pytest.mark.parametrize("version", ["bidirectional-semantic-review-v2", "bidirectional-semantic-review-v3", "bidirectional-semantic-review-v4"])
def test_supported_review_protocol_remains_bound_through_run_workflow_workspace(tmp_path, version):
    run = Run(tmp_path, 1, review_version=version)
    run.complete("TRAIN-1")
    result = run.summarize(scoring_manifest=run.scoring())
    assert result["internal_completed_reports"] == result["metrics"]["reports_valid"] == 1


@pytest.mark.parametrize("mode", ["unknown_version", "run_workflow_mismatch", "workspace_mismatch"])
def test_review_version_changes_cannot_be_silently_mixed(tmp_path, mode):
    run = Run(tmp_path, 1, review_version="unimplemented-v9" if mode == "unknown_version" else "bidirectional-semantic-review-v3",
              workflow_review_version="bidirectional-semantic-review-v2" if mode == "run_workflow_mismatch" else None)
    run.complete("TRAIN-1")
    if mode == "workspace_mismatch":
        path = run.directory("TRAIN-1") / "workspace/workspace.json"
        value = audit.read_json(path)
        value["review"]["protocol"] = "bidirectional-semantic-review-v2"
        save(path, value)
    result = run.summarize(scoring_manifest=run.scoring())
    assert result["metrics"]["reports_total"] == 1
    assert result["metrics"]["reports_valid"] == 0
