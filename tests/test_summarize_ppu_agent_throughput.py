"""Synthetic offline receipt/metric checks; no reports, SDK, GPU or network."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/summarize_ppu_agent_throughput.py"
SPEC = importlib.util.spec_from_file_location("agent_throughput_summary_tests", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def batch(tmp_path):
    selection = {"schema_version": 1, "dataset_split": "training-source-development", "samples": []}
    scoring = {"schema_version": 1, "never_include_in_model_inputs": True, "samples": []}
    inputs, golds = tmp_path / "input", tmp_path / "scoring"
    inputs.mkdir()
    golds.mkdir()
    defect = {"序号": "8", "病害部位": "主梁", "病害描述": "合成裂缝", "是否新增": "未知"}
    gold = {"基本信息": {"桥梁名称": "合成桥", "总体评分": "88.00"},
            "维修建议": [{"建议内容": "合成建议", "建议类别": "未知"}],
            "病害明细": [defect, deepcopy(defect)]}
    for number in range(2):
        sample_id = f"SFT-TRAIN-L{number + 1}"
        source = inputs / f"synthetic-{number}.docx"
        source.write_bytes(f"synthetic input {number}".encode())
        gold_path = golds / (sample_id + ".gold.json")
        save(gold_path, gold)
        sample = {"id": sample_id, "input_docx": str(source), "input_docx_sha256": summary.file_digest(source),
                  "gold_path": str(gold_path), "gold_sha256": summary.file_digest(gold_path)}
        selection["samples"].append(sample)
        scoring["samples"].append({"id": sample_id, "gold_path": str(gold_path),
                                  "gold_sha256": sample["gold_sha256"]})
    # Scoring can contain unrelated short-field probes; they are not extra reports.
    scoring["samples"].append({"id": "SFT-TRAIN-L999", "gold_path": "not-used", "gold_sha256": "not-used"})
    selection_path, scoring_path = tmp_path / "selection.json", tmp_path / "scoring.json"
    save(selection_path, selection)
    save(scoring_path, scoring)
    work, result = tmp_path / "work", tmp_path / "result"
    result.mkdir()
    identity = {"sources": {sample["input_docx"]: sample["input_docx_sha256"] for sample in selection["samples"]},
                "workflow": "divide-react", "document_workers": 2}
    fingerprint = summary.digest(identity)
    save(work / "run_manifest.json", {"identity": identity, "run_fingerprint": fingerprint})
    final = {"run_fingerprint": fingerprint, "reports": 2, "completed_reports": 2,
             "seconds": 100, "documents": [], "document_seconds_sum": 160,
             "timing_scope": "synthetic whole-batch wall"}
    for number, sample in enumerate(selection["samples"]):
        name = Path(sample["input_docx"]).stem
        directory = work / "documents" / name
        manifest = {"document_sha256": sample["input_docx_sha256"],
                    "original_document_sha256": sample["input_docx_sha256"],
                    "units": [{"id": "E1", "text": "abc"}]}
        save(directory / "manifest.json", manifest)
        prediction = deepcopy(gold)
        prediction["基本信息"]["总体评分"] = "88"
        save(directory / "report.json", prediction)
        out_path = result / (name + ".docx")
        out_path.write_bytes(b"synthetic published Word")
        calls = [{"call_id": f"tokenize-{number}", "kind": "tokenize", "status": "success", "token_count": 88},
                 {"call_id": f"generation-{number}", "kind": "generation", "status": "success",
                  "finish_reason": "stop", "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}]
        task = {"id": "T1", "status": "completed", "accepted_call_id": f"generation-{number}",
                "spans": [{"coverage_id": "E1", "start": 0, "end": 3}],
                "attempts": [{"call_id": f"generation-{number}", "status": "success"}]}
        agent = {"status": "completed", "native_final_event": True, "external_model_used": False,
                 "tool_calls": [{"name": "extract_chunk", "status": "success"}, {"name": "finish", "status": "success"}]}
        execution = {"status": "success", "mode": "divide-react", "run_fingerprint": fingerprint,
                     "manifest_sha256": summary.digest(manifest), "document_sha256": manifest["document_sha256"],
                     "report_sha256": summary.digest(prediction), "model_calls": calls, "tasks": [task],
                     "agent_runs": [agent], "reviews": [], "coverage": {"complete": True},
                     "output": {"name": out_path.name, "sha256": summary.file_digest(out_path)}}
        save(directory / "execution.json", execution)
        final["documents"].append({"name": name, "status": "completed", "output": out_path.name, "seconds": 80})
    save(work / "summary.json", final)
    return SimpleNamespace(root=tmp_path, selection=selection_path, scoring=scoring_path, work=work,
                           result=result, samples=selection["samples"], final=final)


def evaluate(batch, **kwargs):
    return summary.summarize(batch.selection, batch.scoring, batch.work, batch.result, **kwargs)


def execution_path(batch, number=0):
    return batch.work / "documents" / Path(batch.samples[number]["input_docx"]).stem / "execution.json"


def mutate_execution(batch, change, number=0):
    path = execution_path(batch, number)
    record = summary.read_json(path)
    change(record)
    save(path, record)


def rebind_identity(batch, identity):
    """Change synthetic identity consistently, so a profile check is exercised."""
    fingerprint = summary.digest(identity)
    save(batch.work / "run_manifest.json", {"identity": identity, "run_fingerprint": fingerprint})
    for number in range(2):
        mutate_execution(batch, lambda record: record.update(run_fingerprint=fingerprint), number)
    batch.final["run_fingerprint"] = fingerprint
    save(batch.work / "summary.json", batch.final)


def react_profile_batch(batch):
    identity = summary.read_json(batch.work / "run_manifest.json")["identity"]
    settings = {"chunk_tokens": 6000, "output_tokens": 4096, "max_split_depth": 3,
                "max_extraction_attempts": 2, "max_tasks": 512, "max_context_units": 12, "max_tool_tokens": 6000}
    identity.update(settings=settings, sdk_version="2.7.20", served_model="synthetic-own-model",
                    adapter_sha256="a" * 64, code={"run.py": "b" * 64},
                    model_identity={"kind": "dynamic_lora", "adapter_path": "/synthetic/adapter"},
                    agent_max_iterations=8, agent_max_calls=12, model_timeout=900,
                    max_model_len=65536, chat_tokenization_mode="server_chat_template",
                    document_worker_backend="spawn-processes", future_comparison_field="preserved")
    batch.final["workflow"] = "divide-react"
    rebind_identity(batch, identity)
    for number in range(2):
        def change(record):
            record["settings"] = deepcopy(settings)
            for call in record["model_calls"]:
                call["model"] = identity["served_model"]
            for run in record["agent_runs"]:
                run["sdk_version"] = identity["sdk_version"]
        mutate_execution(batch, change, number)
    return identity


def test_complete_batch_uses_wall_and_multiset_f1_with_visible_controller_unknowns(batch):
    result = evaluate(batch)
    assert result["status"] == "completed"
    assert result["batch_wall_seconds"] == 100  # not document sum 160
    assert result["completed_reports_per_hour"] == 72
    assert result["metrics"]["sections"]["defects"]["tp"] == 4  # duplicates retained
    assert result["metrics"]["overall"]["f1"] == 1
    assert result["basic_field_metrics"]["总体评分"]["f1"] == 1
    observed = result["observed_work"]
    assert observed["worker_generation_requests"] == 2
    assert observed["http_requests_by_kind"] == {"tokenize": 2, "generation": 2}
    assert observed["worker_tokens"]["total_tokens"]["complete_sum"] == 60
    assert observed["sdk_controller_generation_requests"] is None
    assert observed["whole_workflow_tokens"] is None
    assert result["all_reports_audit_complete"]


def test_missing_failed_document_keeps_all_gold_fn_and_does_not_inflate_throughput(batch):
    directory = execution_path(batch, 1).parent
    for path in directory.iterdir():
        path.unlink()
    batch.final["documents"][1] = {"name": batch.final["documents"][1]["name"], "status": "failed",
                                    "error": "synthetic worker crash", "seconds": None}
    batch.final["completed_reports"] = 1
    save(batch.work / "summary.json", batch.final)
    result = evaluate(batch)
    assert result["status"] == "completed_with_failures"
    assert result["reports_expected"] == 2 and result["reports_completed"] == 1
    assert result["records"][1]["status"] == "missing"
    assert result["metrics"]["sections"]["defects"]["fn"] == 2
    assert result["metrics"]["sections"]["basic"]["fn"] == 2
    assert result["completed_reports_per_hour"] == 36


@pytest.mark.parametrize("mode", ["length", "coverage_gap", "no_sdk_final", "report_hash", "word_hash", "unresolved_review"])
def test_completion_claim_does_not_bypass_actual_evidence_gate(batch, mode):
    def change(record):
        if mode == "length":
            record["model_calls"][1]["finish_reason"] = "length"
        elif mode == "coverage_gap":
            record["tasks"][0]["spans"][0]["end"] = 2
        elif mode == "no_sdk_final":
            record["agent_runs"][0]["native_final_event"] = False
        elif mode == "report_hash":
            record["report_sha256"] = "wrong"
        elif mode == "word_hash":
            record["output"]["sha256"] = "wrong"
        elif mode == "unresolved_review":
            record["reviews"] = [{"resolved": False, "attempts": []}]
    mutate_execution(batch, change)
    result = evaluate(batch)
    assert result["reports_completed"] == 1
    assert not result["records"][0]["score"]["prediction_valid"]
    assert result["records"][0]["score"]["overall"]["fn"] == 5


def test_truncated_attempt_can_recover_but_is_visible_and_missing_usage_is_unknown(batch):
    def change(record):
        record["model_calls"].insert(1, {"call_id": "failed-first", "kind": "generation", "status": "failed",
                                          "finish_reason": "length", "usage": None})
        record["tasks"][0]["attempts"].insert(0, {"call_id": "failed-first", "status": "failed"})
    mutate_execution(batch, change)
    result = evaluate(batch)
    assert result["reports_completed"] == 2
    observed = result["observed_work"]
    assert observed["worker_generation_requests"] == 3
    assert observed["worker_generation_length_finishes"] == 1
    assert observed["extraction_retry_attempts"] == 1
    tokens = observed["worker_tokens"]["total_tokens"]
    assert tokens == {"observed_sum": 60, "receipts_with_value": 2, "receipts_missing_value": 1, "complete_sum": None}


def test_missing_final_batch_summary_never_gets_a_throughput_number(batch):
    (batch.work / "summary.json").unlink()
    result = evaluate(batch)
    assert result["status"] == "incomplete"
    assert result["completed_reports_per_hour"] is None
    assert result["batch_wall_seconds"] is None
    assert result["metrics"]["reports_total"] == 2
    assert result["metrics"]["sections"]["defects"]["fn"] == 4


def test_different_source_selection_invalidates_batch_instead_of_matching_names_only(batch):
    path = batch.work / "run_manifest.json"
    bound = summary.read_json(path)
    bound["identity"]["sources"][batch.samples[0]["input_docx"]] = "wrong-source"
    save(path, bound)
    result = evaluate(batch)
    assert result["reports_completed"] == 0 and not result["timing_valid"]
    assert "run_inputs_differ_from_fixed_selection" in result["global_errors"]


def test_missing_or_changed_gold_aborts_rather_than_shrinking_denominator(batch):
    Path(batch.samples[0]["gold_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Gold binding mismatch"):
        evaluate(batch)


def test_malformed_audit_counts_failure_instead_of_crashing_or_dropping_document(batch):
    def change(record):
        record["model_calls"] = None
    mutate_execution(batch, change)
    result = evaluate(batch)
    assert result["reports_completed"] == 1
    assert result["metrics"]["reports_total"] == 2
    assert result["observed_work"]["malformed_audit_records_or_containers"] == 1


def test_cli_output_is_new_and_contains_no_label_or_prediction_bodies(batch):
    destination = batch.root / "evaluation.json"
    args = ["--selection", str(batch.selection), "--scoring", str(batch.scoring),
            "--work-dir", str(batch.work), "--result-dir", str(batch.result), "--out-json", str(destination)]
    assert summary.main(args) == 0
    report = summary.read_json(destination)
    assert all("gold" not in record and "prediction" not in record for record in report["records"])
    with pytest.raises(FileExistsError):
        summary.main(args)


def test_strict_react_profile_retains_complete_comparison_identity(batch):
    identity = react_profile_batch(batch)
    result = evaluate(batch, require_tongagents_react=True)
    assert result["reports_completed"] == 2 and result["all_reports_audit_complete"]
    assert result["evaluation_profile"] == "tongagents-react"
    assert result["identity"] == identity
    assert result["identity"]["future_comparison_field"] == "preserved"
    assert result["identity"]["agent_max_iterations"] == 8
    assert result["identity"]["model_timeout"] == 900
    assert result["identity"]["chat_tokenization_mode"] == "server_chat_template"


@pytest.mark.parametrize("field", ["sdk_version", "served_model", "adapter_sha256", "model_identity", "code",
                                   "agent_max_iterations", "agent_max_calls", "model_timeout", "max_model_len",
                                   "document_workers", "settings", "chat_tokenization_mode"])
def test_strict_profile_rejects_missing_required_identity_field(batch, field):
    identity = react_profile_batch(batch)
    identity.pop(field)
    rebind_identity(batch, identity)
    result = evaluate(batch, require_tongagents_react=True)
    assert result["reports_completed"] == 0
    assert any("react_profile_missing_or_invalid:" + field in error for error in result["global_errors"])
    assert result["metrics"]["reports_total"] == 2
    assert result["metrics"]["overall"]["fn"] == 10


@pytest.mark.parametrize("change,expected", [
    ({"document_workers": 9}, "react_profile_document_workers_out_of_range"),
    ({"agent_max_calls": True}, "react_profile_missing_or_invalid:agent_max_calls"),
    ({"model_timeout": 0}, "react_profile_missing_or_invalid:model_timeout"),
    ({"adapter_sha256": "not-a-sha"}, "react_profile_missing_or_invalid:adapter_sha256"),
    ({"max_model_len": 10000}, "react_profile_chunk_budget_exceeds_context"),
    ({"workflow": "divide-fixed"}, "react_profile_identity_workflow_mismatch"),
])
def test_strict_profile_rejects_invalid_bound_identity_values(batch, change, expected):
    identity = react_profile_batch(batch)
    identity.update(change)
    rebind_identity(batch, identity)
    result = evaluate(batch, require_tongagents_react=True)
    assert expected in result["global_errors"]
    assert result["completed_reports_per_hour"] is None


@pytest.mark.parametrize("mode,expected", [
    ("execution_mode", "react_profile_execution_mode_mismatch"),
    ("execution_settings", "react_profile_execution_settings_mismatch"),
    ("sdk_version", "react_profile_execution_sdk_version_mismatch"),
    ("model", "react_profile_accepted_generation_model_mismatch"),
])
def test_strict_profile_binds_execution_to_identity(batch, mode, expected):
    react_profile_batch(batch)
    def change(record):
        if mode == "execution_mode":
            record["mode"] = "divide-fixed"
        elif mode == "execution_settings":
            record["settings"]["output_tokens"] = 512
        elif mode == "sdk_version":
            record["agent_runs"][0]["sdk_version"] = "other-sdk-version"
        elif mode == "model":
            record["model_calls"][1]["model"] = "different-model"
    mutate_execution(batch, change)
    result = evaluate(batch, require_tongagents_react=True)
    assert result["reports_completed"] == 1
    assert expected in result["records"][0]["errors"]


def test_strict_profile_checks_summary_workflow_and_cli_flag(batch):
    react_profile_batch(batch)
    batch.final["workflow"] = "divide-fixed"
    save(batch.work / "summary.json", batch.final)
    destination = batch.root / "strict-evaluation.json"
    assert summary.main(["--selection", str(batch.selection), "--scoring", str(batch.scoring),
                         "--work-dir", str(batch.work), "--result-dir", str(batch.result),
                         "--out-json", str(destination), "--require-tongagents-react"]) == 1
    result = summary.read_json(destination)
    assert "react_profile_summary_workflow_mismatch" in result["global_errors"]


def test_generic_default_remains_usable_for_fixed_workflow(batch):
    identity = summary.read_json(batch.work / "run_manifest.json")["identity"]
    identity["workflow"] = "divide-fixed"
    batch.final["workflow"] = "divide-fixed"
    rebind_identity(batch, identity)
    for number in range(2):
        mutate_execution(batch, lambda record: record.update(mode="divide-fixed", agent_runs=[]), number)
    result = evaluate(batch)
    assert result["all_reports_audit_complete"]
    assert result["evaluation_profile"] == "generic-divide-workflow"
    assert evaluate(batch, require_tongagents_react=True)["reports_completed"] == 0
