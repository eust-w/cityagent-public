"""Offline synthetic paired metrics; no source reports, models, SDK or GPU."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/compare_ppu_agent_throughput.py"
SPEC = importlib.util.spec_from_file_location("paired_agent_summary_tests", SCRIPT)
paired = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paired)
single = paired.single


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


GOLD = {"基本信息": {"桥梁名称": "合成桥", "总体评分": "85.9"},
        "维修建议": [{"建议内容": "合成建议"}],
        "病害明细": [{"病害部位": "主梁", "病害描述": "合成裂缝"},
                      {"病害部位": "主梁", "病害描述": "合成裂缝"}]}


def evaluation(predictions=None, *, workers=1, wall=200):
    predictions = [deepcopy(GOLD), deepcopy(GOLD)] if predictions is None else predictions
    identity = {
        "workflow": "divide-react", "sdk_version": "2.7.20", "served_model": "own-model",
        "adapter_sha256": "a" * 64, "model_identity": {"kind": "dynamic_lora", "adapter_path": "/own/adapter"},
        "settings": {"chunk_tokens": 6000, "output_tokens": 4096, "max_split_depth": 3,
                     "max_extraction_attempts": 2, "max_tasks": 512, "max_context_units": 12, "max_tool_tokens": 6000},
        "agent_max_iterations": 8, "agent_max_calls": 12, "model_timeout": 900,
        "max_model_len": 65536, "document_workers": workers, "effective_document_workers": min(workers, 2),
        "isolate_documents": True, "document_worker_backend": "spawn-processes",
        "chat_tokenization_mode": "server_chat_template", "endpoint": "http://127.0.0.1:8000",
        "code": {"run.py": "b" * 64, "src/contest.py": "c" * 64},
        "sources": {"/frozen/A.docx": "1" * 64, "/frozen/B.docx": "2" * 64},
        "model_catalog": [{"id": "own-model", "created": 100, "root": "/own/adapter", "parent": "own-base",
                           "permission": [{"id": "perm-1", "created": 100, "allow_sampling": True}]}],
        "preflight_model_calls": [{"kind": "models", "endpoint": "/v1/models", "method": "GET",
                                   "model": "own-model", "status": "success", "http_status": 200,
                                   "input_sha256": "d" * 64, "output_sha256": "e" * 64,
                                   "response_sha256": "e" * 64, "call_id": "uuid-1", "started_at": "time-1", "seconds": 0.1}],
    }
    records, field_scores = [], {}
    for index, (name, prediction) in enumerate(zip(("A", "B"), predictions)):
        score = single.score_report(GOLD, prediction)
        records.append({"id": f"SFT-TRAIN-L{index + 1}", "name": name,
                        "input_docx_sha256": str(index + 1) * 64, "gold_sha256": "f" * 64,
                        "status": "completed" if prediction is not None else "failed",
                        "errors": [] if prediction is not None else ["synthetic failure"], "score": score})
        basic = prediction["基本信息"] if prediction is not None else {}
        for field in set(GOLD["基本信息"]) | set(basic):
            one_gold = {"基本信息": {field: GOLD["基本信息"][field]} if field in GOLD["基本信息"] else {},
                        "维修建议": [], "病害明细": []}
            one_prediction = {"基本信息": {field: basic[field]} if field in basic else {},
                              "维修建议": [], "病害明细": []} if prediction is not None else None
            field_scores.setdefault(field, []).append(single.score_report(one_gold, one_prediction))
    completed = sum(prediction is not None for prediction in predictions)
    return {"schema_version": 1, "evaluation_profile": "tongagents-react", "identity": identity,
            "run_fingerprint": single.digest(identity), "global_errors": [], "records": records,
            "sample_ids": [row["id"] for row in records], "fixed_selection_sha256": "0" * 64,
            "scoring_manifest_sha256": "9" * 64, "dataset_split": "official-training-source-development",
            "reports_expected": 2, "reports_completed": completed, "reports_failed_or_missing": 2 - completed,
            "timing_valid": True, "batch_wall_seconds": wall, "all_reports_audit_complete": completed == 2,
            "status": "completed" if completed == 2 else "completed_with_failures",
            "metrics": single.aggregate_reports(row["score"] for row in records),
            "basic_field_metrics": {field: single.aggregate_reports(scores)["sections"]["basic"]
                                    for field, scores in field_scores.items()}}


@pytest.fixture
def pair(tmp_path):
    reference, candidate = evaluation(), evaluation(workers=4, wall=100)
    ref_path, cand_path = tmp_path / "reference.json", tmp_path / "candidate.json"
    save(ref_path, reference)
    save(cand_path, candidate)
    return SimpleNamespace(root=tmp_path, reference=reference, candidate=candidate,
                           ref_path=ref_path, cand_path=cand_path)


def compare(pair):
    save(pair.ref_path, pair.reference)
    save(pair.cand_path, pair.candidate)
    return paired.compare(pair.ref_path, pair.cand_path)


def rebind(pair):
    pair.candidate["run_fingerprint"] = single.digest(pair.candidate["identity"])


def test_only_allowed_workers_and_runtime_metadata_differences_pass(pair):
    model = pair.candidate["identity"]["model_catalog"][0]
    model["created"] = 200
    model["permission"][0].update(id="perm-2", created=200)
    pair.candidate["identity"]["preflight_model_calls"][0].update(
        call_id="uuid-2", started_at="time-2", seconds=0.3, response_sha256="7" * 64, output_sha256="7" * 64)
    rebind(pair)
    result = compare(pair)
    assert result["comparable"]
    assert result["performance"]["throughput_ratio_candidate_over_reference"] == 2
    assert result["metrics"]["overall"]["delta"]["f1"] == 0
    assert result["identity_differences_not_allowed"] == []
    assert len(result["allowed_identity_changes"]) >= 8
    assert result["completeness"]["both_audit_complete"]
    assert "No semantic passing threshold" in result["quality_verdict"]


@pytest.mark.parametrize("field,value", [
    ("sdk_version", "2.8.0"), ("served_model", "different-model"), ("adapter_sha256", "8" * 64),
    ("agent_max_iterations", 12), ("agent_max_calls", 14), ("model_timeout", 1800),
    ("max_model_len", 131072), ("chat_tokenization_mode", "verified_local_template+server_prompt_count"),
    ("isolate_documents", False), ("document_worker_backend", "serial"),
    ("endpoint", "http://127.0.0.1:8001"), ("unrecognized_future_setting", True),
])
def test_semantic_budget_runtime_and_unknown_field_changes_block_comparison(pair, field, value):
    pair.candidate["identity"][field] = value
    rebind(pair)
    result = compare(pair)
    assert not result["comparable"]
    assert any(change["path"] == "identity." + field for change in result["identity_differences_not_allowed"])
    assert result["performance"]["throughput_ratio_candidate_over_reference"] is None
    assert result["metrics"]["any_defined_label_metric_decreased"] is None


@pytest.mark.parametrize("mode", ["code", "settings", "model_path", "catalog_root", "permission", "preflight_model"])
def test_nested_model_code_budget_and_permission_changes_are_not_ignored(pair, mode):
    identity = pair.candidate["identity"]
    if mode == "code":
        identity["code"]["run.py"] = "8" * 64
    elif mode == "settings":
        identity["settings"]["output_tokens"] = 2048
    elif mode == "model_path":
        identity["model_identity"]["adapter_path"] = "/other/adapter"
    elif mode == "catalog_root":
        identity["model_catalog"][0]["root"] = "/other/adapter"
    elif mode == "permission":
        identity["model_catalog"][0]["permission"][0]["allow_sampling"] = False
    else:
        identity["preflight_model_calls"][0]["model"] = "different-model"
    rebind(pair)
    assert not compare(pair)["comparable"]


@pytest.mark.parametrize("mode", ["selection", "scoring", "source", "gold", "sample_ids"])
def test_fixed_dataset_bindings_must_match(pair, mode):
    if mode == "selection":
        pair.candidate["fixed_selection_sha256"] = "5" * 64
    elif mode == "scoring":
        pair.candidate["scoring_manifest_sha256"] = "5" * 64
    elif mode == "source":
        pair.candidate["records"][0]["input_docx_sha256"] = "5" * 64
    elif mode == "gold":
        pair.candidate["records"][0]["gold_sha256"] = "5" * 64
    else:
        pair.candidate["sample_ids"].reverse()
    result = compare(pair)
    assert not result["comparable"]
    assert result["dataset_differences"]


def test_complete_audits_can_still_have_quality_regressions(pair):
    worse = deepcopy(GOLD)
    worse["基本信息"]["总体评分"] = "wrong"
    worse["病害明细"].pop()  # Duplicate rows must retain multiplicity.
    pair.candidate = evaluation([worse, deepcopy(GOLD)], workers=4, wall=100)
    result = compare(pair)
    assert result["comparable"] and result["completeness"]["both_audit_complete"]
    assert result["performance"]["throughput_ratio_candidate_over_reference"] == 2
    assert result["metrics"]["any_defined_label_metric_decreased"] is True
    assert result["metrics"]["overall"]["delta"]["f1"] < 0
    assert result["metrics"]["sections"]["defects"]["delta"]["fn"] == 1
    assert result["metrics"]["basic_fields"]["总体评分"]["delta"]["f1"] == -0.5


def test_failed_report_keeps_gold_fn_and_reduces_successful_report_rate(pair):
    pair.candidate = evaluation([deepcopy(GOLD), None], workers=4, wall=100)
    result = compare(pair)
    assert result["comparable"]
    assert not result["completeness"]["both_audit_complete"]
    assert result["completeness"]["completed"]["delta"] == -1
    assert result["performance"]["throughput_ratio_candidate_over_reference"] == 1
    assert result["metrics"]["overall"]["delta"]["fn"] == 5
    assert result["completeness"]["status_changes"] == [{"id": "SFT-TRAIN-L2", "reference": "completed", "candidate": "failed"}]


def test_cached_aggregate_f1_is_not_trusted(pair):
    pair.candidate["metrics"]["overall"]["f1"] = 0.01
    pair.candidate["metrics"]["sections"]["defects"]["macro_report_f1"] = 0.01
    pair.candidate["records"][0]["score"]["overall"]["f1"] = 0.01
    pair.candidate["records"][0]["score"]["macro_section_f1"] = 0.01
    result = compare(pair)
    assert result["metrics"]["overall"]["candidate"]["f1"] == 1
    assert result["metrics"]["sections"]["defects"]["candidate"]["macro_report_f1"] == 1
    assert result["metrics"]["per_report"]["SFT-TRAIN-L1"]["macro_section_f1"]["candidate"] == 1


def test_pooled_tie_does_not_hide_macro_or_single_report_degradation(pair):
    one_defect = deepcopy(GOLD)
    one_defect["病害明细"].pop()
    no_defects = deepcopy(GOLD)
    no_defects["病害明细"] = []
    pair.reference = evaluation([deepcopy(one_defect), deepcopy(one_defect)])
    pair.candidate = evaluation([deepcopy(GOLD), no_defects], workers=4, wall=100)
    result = compare(pair)
    metrics = result["metrics"]
    assert result["comparable"] and result["completeness"]["both_audit_complete"]
    assert metrics["overall"]["delta"]["f1"] == 0
    assert all(change["delta"]["f1"] == 0 for change in metrics["sections"].values())
    assert metrics["sections"]["defects"]["reference"]["macro_report_f1"] == pytest.approx(2 / 3)
    assert metrics["sections"]["defects"]["candidate"]["macro_report_f1"] == 0.5
    assert metrics["sections"]["defects"]["delta"]["macro_report_f1"] == pytest.approx(-1 / 6)
    assert metrics["sections"]["defects"]["reference"]["macro_report_support"] == 2
    assert metrics["per_report"]["SFT-TRAIN-L2"]["sections"]["defects"]["delta"]["f1"] == pytest.approx(-2 / 3)
    assert metrics["any_defined_label_metric_decreased"] is True
    decreased = {row["metric"] for row in metrics["decreased_precision_recall_or_f1"]}
    assert "sections.defects.macro_report_f1" in decreased
    assert "per_report.SFT-TRAIN-L2.sections.defects.f1" in decreased


def test_even_pooled_and_macro_ties_do_not_hide_one_report_getting_worse(pair):
    one_defect = deepcopy(GOLD)
    one_defect["病害明细"].pop()
    pair.reference = evaluation([deepcopy(one_defect), deepcopy(GOLD)])
    pair.candidate = evaluation([deepcopy(GOLD), deepcopy(one_defect)], workers=4, wall=100)
    result = compare(pair)
    assert result["metrics"]["overall"]["delta"]["f1"] == 0
    assert result["metrics"]["sections"]["defects"]["delta"]["macro_report_f1"] == 0
    assert result["metrics"]["per_report"]["SFT-TRAIN-L2"]["overall"]["delta"]["f1"] < 0
    assert result["metrics"]["any_defined_label_metric_decreased"] is True


def test_stdout_exposes_completion_drop_even_when_pooled_f1_and_throughput_improve(pair, capsys):
    wrong = deepcopy(GOLD)
    wrong["基本信息"] = {"桥梁名称": "错误桥名", "总体评分": "0"}
    wrong["维修建议"] = [{"建议内容": "错误建议"}]
    wrong["病害明细"] = [{"病害部位": "错误部位", "病害描述": "错误病害"}] * 2
    pair.reference = evaluation([wrong, deepcopy(GOLD)], wall=200)
    pair.candidate = evaluation([deepcopy(GOLD), None], workers=4, wall=50)
    result = compare(pair)
    assert result["metrics"]["overall"]["delta"]["f1"] > 0
    assert result["performance"]["throughput_ratio_candidate_over_reference"] == 2
    destination = pair.root / "failed-report-comparison.json"
    assert paired.main(["--reference", str(pair.ref_path), "--candidate", str(pair.cand_path),
                        "--out-json", str(destination)]) == 0  # Comparable, not passed.
    printed = json.loads(capsys.readouterr().out)
    assert printed["completeness"]["completed"]["delta"] == -1
    assert not printed["completeness"]["both_audit_complete"]
    assert printed["any_defined_label_metric_decreased"] is True
    assert "No semantic passing threshold" in printed["quality_verdict"]


def test_partial_prediction_for_failed_report_is_rejected(pair):
    pair.candidate["records"][0]["status"] = "failed"
    pair.candidate["records"][0]["score"]["prediction_valid"] = False
    with pytest.raises(ValueError, match="cannot retain partial predictions"):
        compare(pair)


def test_missing_final_wall_blocks_ranking(pair):
    pair.candidate.update(timing_valid=False, batch_wall_seconds=None,
                          all_reports_audit_complete=False, status="incomplete")
    result = compare(pair)
    assert not result["comparable"]
    assert result["performance"]["completed_reports_per_hour"]["candidate"] is None


def test_generic_profile_cannot_be_labeled_as_tongagents_benchmark(pair):
    pair.candidate["evaluation_profile"] = "generic-divide-workflow"
    assert not compare(pair)["comparable"]


def test_invalid_basic_field_counts_do_not_silently_pass(pair):
    pair.candidate["basic_field_metrics"]["总体评分"]["tp"] += 1
    with pytest.raises(ValueError, match="Inconsistent"):
        compare(pair)


def test_unknown_preflight_response_hash_is_not_ignored(pair):
    extra = {"call_id": "id", "kind": "other", "endpoint": "/unknown", "method": "GET",
             "status": "success", "http_status": 200, "response_sha256": "0" * 64}
    pair.reference["identity"]["preflight_model_calls"].append(deepcopy(extra))
    pair.candidate["identity"]["preflight_model_calls"].append({**extra, "response_sha256": "1" * 64})
    pair.reference["run_fingerprint"] = single.digest(pair.reference["identity"])
    rebind(pair)
    assert not compare(pair)["comparable"]


def test_cli_does_not_overwrite_existing_comparison(pair):
    destination = pair.root / "comparison.json"
    args = ["--reference", str(pair.ref_path), "--candidate", str(pair.cand_path), "--out-json", str(destination)]
    assert paired.main(args) == 0
    assert single.read_json(destination)["comparable"]
    with pytest.raises(FileExistsError):
        paired.main(args)
