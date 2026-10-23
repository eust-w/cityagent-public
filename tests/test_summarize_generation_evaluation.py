"""Synthetic CPU-only evaluation evidence; no real reports, model, SDK or GPU."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_generation_evaluation.py"
SPEC = importlib.util.spec_from_file_location("generation_summary_tests", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class SyntheticTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "<system>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<assistant>"

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        return {"input_ids": list(text.encode("utf-8"))}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path):
    tokenizer = SyntheticTokenizer()
    defect = {"序号": "19", "病害部位": "P1", "病害描述": "合成裂缝", "是否新增": ""}
    prediction = {"基本信息": {"桥梁名称": "合成桥", "报告日期": "2018-01-01"},
                  "维修建议": [], "病害明细": [defect, deepcopy(defect)]}
    content = [{"instruction": "synthetic extract", "input": f"synthetic report {index}",
                "output": json.dumps(prediction, ensure_ascii=False), "meta": {"year": "2018"}}
               for index in range(5)]
    content.append({**content[0], "input": "synthetic training", "meta": {"year": "2017"}})
    sft = tmp_path / "sft.jsonl"
    sft.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in content), encoding="utf-8")
    rows = summary.bench.load_rows(sft)
    tokenized = [summary.bench.tokenize_row(row, tokenizer) for row in rows]
    selection = summary.selection_helper.build_manifest(rows, tokenized, sft_sha256=summary.bench.file_digest(sft))
    selection_path = tmp_path / "selection.json"
    save(selection_path, selection)
    args = SimpleNamespace(sft=sft, selection_manifest=selection_path, tokenizer=tmp_path / "tokenizer",
                           generation_dir=[], partition="diagnostic")
    return SimpleNamespace(root=tmp_path, args=args, tokenizer=tokenizer, rows=rows,
                           selection=selection, prediction=prediction, jobs=[])


def job(fixture, *, partition="diagnostic", shards=1, shard_index=0, **parameters):
    directory = fixture.root / f"job-{len(fixture.jobs)}"
    directory.mkdir()
    params = {"model": "/synthetic/model", "adapter": "/synthetic/adapter",
              "sft": str(fixture.args.sft), "selection_manifest": str(fixture.args.selection_manifest),
              "out_dir": str(directory), "partition": partition, "shards": shards,
              "shard_index": shard_index, "limit": None, "sample_sha256": None,
              "max_model_len": 100000, "max_new_tokens": 1000, "gpu_memory_gib": 24,
              "sample_timeout_seconds": 1200, "prefill_chunk_size": 0, "prefill_min_tokens": 32768,
              **parameters}
    selected = summary.runner.select_samples(fixture.selection, fixture.rows, partition=partition,
                                             shards=shards, shard_index=shard_index,
                                             limit=params["limit"], sample_sha256=params["sample_sha256"])
    manifest = {key: hashlib.sha256(key.encode()).hexdigest() for key in summary.SOURCE_HASHES}
    manifest.update(sft_sha256=summary.bench.file_digest(fixture.args.sft),
                    selection_manifest_sha256=summary.bench.file_digest(fixture.args.selection_manifest),
                    parameters=params, selected_sample_sha256=[sample["sample_sha256"] for sample, _ in selected],
                    versions={"python": "synthetic", "transformers": "synthetic"},
                    generation="synthetic greedy full-prompt single-attempt")
    save(directory / "manifest.json", manifest)
    result = SimpleNamespace(directory=directory, manifest=manifest, selected=selected)
    fixture.jobs.append(result)
    fixture.args.generation_dir.append(directory)
    fixture.args.partition = partition
    return result


def output(fixture, generation, index=0, *, mode="success"):
    sample, row = generation.selected[index]
    ids = summary.bench.tokenize_row(row, fixture.tokenizer)["input_ids"][:sample["prompt_tokens"]]
    raw = json.dumps(fixture.prediction, ensure_ascii=False)
    call = {"call_id": "synthetic-" + sample["sample_sha256"], "utc": "2026-10-09T00:00:00Z",
            "attempt": 1, "prompt_sha256": summary.prompt_ids_sha256(ids), "prompt_tokens": len(ids),
            "output_tokens": 100, "seconds": 0.2, "finish_reason": "stop", "status": "success", "errors": [],
            "output_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    record = {**sample, "prompt_ids_sha256": summary.prompt_ids_sha256(ids), "model_calls": [call],
              "raw_output": raw, "prediction": deepcopy(fixture.prediction), "status": "success",
              "finish_reason": "stop", "output_tokens": 100, "seconds": 0.2, "prefill_chunk_size": 0}
    if mode == "context_budget":
        record.update(model_calls=[], raw_output="", prediction=None, status="failed",
                      error="context_budget: synthetic full prompt cannot fit")
        for key in ("finish_reason", "output_tokens", "seconds"):
            record.pop(key)
    elif mode in {"backend", "length", "invalid_json", "metric_schema"}:
        record.update(prediction=None, status="failed", error="InferenceFailure")
        call.update(status="failed", errors=["synthetic failure"])
        if mode == "backend":
            call.update(output_sha256=None, output_tokens=None, finish_reason=None,
                        errors=["backend_exception:RuntimeError"])
            record.update(raw_output="", output_tokens=None, finish_reason=None)
        elif mode == "length":
            call.update(finish_reason="length", errors=["incomplete_generation:length"])
            record["finish_reason"] = "length"
        elif mode == "invalid_json":
            record["raw_output"] = "synthetic invalid JSON"
            call.update(output_sha256=hashlib.sha256(record["raw_output"].encode()).hexdigest(), errors=["invalid_json"])
        else:
            collision = deepcopy(fixture.prediction)
            collision["基本信息"][" 桥梁名称 "] = "colliding field name"
            record.update(raw_output=json.dumps(collision, ensure_ascii=False), error="metric_schema: collision")
            call.update(status="success", errors=[], output_sha256=hashlib.sha256(record["raw_output"].encode()).hexdigest())
    path = generation.directory / (sample["sample_sha256"] + ".json")
    save(path, record)
    return path, record


def child_summary(generation):
    save(generation.directory / "summary.json", {
        "status": "completed", "samples": len(generation.selected),
        "partition": generation.manifest["parameters"]["partition"],
        "manifest_sha256": summary.bench.file_digest(generation.directory / "manifest.json"),
        "metrics": {"forged_perfect_score": 1.0},
    })


def summarize(fixture):
    return summary.summarize(fixture.args, tokenizer=fixture.tokenizer)


def test_full_partition_missing_and_failed_keep_all_fn_even_with_completed_child_summary(fixture):
    generation = job(fixture, partition="final_acceptance")
    output(fixture, generation, 0)
    output(fixture, generation, 1, mode="length")
    child_summary(generation)
    result = summarize(fixture)
    assert result["status"] == "incomplete"
    assert result["counts"] == {"expected": 3, "success": 1, "failed": 1, "missing": 1}
    assert not result["all_results_present"] and not result["all_predictions_valid"]
    assert result["partition"] == "final_acceptance"
    assert result["metrics"]["raw_model"]["reports_total"] == 3
    assert result["metrics"]["raw_model"]["reports_failed"] == 2
    assert result["metrics"]["raw_model"]["sections"]["defects"]["fn"] == 4
    assert [record["status"] for record in result["records"]] == ["success", "failed", "missing"]
    assert all(record["strategies"]["raw_model"]["score"]["overall"]["predicted_count"] == 0
               for record in result["records"][1:])


def test_disjoint_shards_merge_with_duplicate_rows_and_candidate_difference_visible(fixture):
    for index in range(2):
        generation = job(fixture, shards=2, shard_index=index)
        output(fixture, generation)
        child_summary(generation)
    result = summarize(fixture)
    assert result["status"] == "completed" and result["all_predictions_valid"]
    assert result["counts"] == {"expected": 2, "success": 2, "failed": 0, "missing": 0}
    for record in result["records"]:
        assert record["semantic_preservation"]["verified"]
        assert record["candidate_audit"]["row_counts"]["病害明细"] == {"before": 2, "after": 2}
        assert record["strategies"]["raw_model"]["score"]["sections"]["defects"]["tp"] == 2
        assert record["strategies"]["model_preserving"]["score"]["sections"]["defects"]["tp"] == 2
    # The derived 建议统计 is a visible extra basic fact, not silently excluded from candidate metrics.
    assert result["metrics"]["raw_model"]["sections"]["basic"]["fp"] == 0
    assert result["metrics"]["model_preserving"]["sections"]["basic"]["fp"] == 2
    assert len(result["jobs"]) == 2


@pytest.mark.parametrize("mode", ["backend", "length", "invalid_json", "metric_schema"])
def test_explicit_failure_is_none_and_never_counted_as_success(fixture, mode):
    generation = job(fixture)
    output(fixture, generation, 0, mode=mode)
    output(fixture, generation, 1)
    result = summarize(fixture)
    assert result["status"] == "completed_with_failures"
    assert result["all_results_present"] and not result["all_predictions_valid"]
    assert result["counts"]["failed"] == 1
    assert result["metrics"]["raw_model"]["reports_failed"] == 1
    assert not result["records"][0]["strategies"]["raw_model"]["score"]["prediction_valid"]


def test_overlong_records_without_backend_calls_stay_in_denominator(fixture):
    generation = job(fixture, max_model_len=100)
    for index in range(len(generation.selected)):
        output(fixture, generation, index, mode="context_budget")
    result = summarize(fixture)
    assert result["status"] == "completed_with_failures"
    assert result["counts"]["failed"] == 2
    assert result["metrics"]["raw_model"]["overall"]["tp"] == 0
    assert result["metrics"]["raw_model"]["overall"]["fn"] > 0


def test_no_results_yet_is_incomplete_with_full_frozen_denominator(fixture):
    job(fixture)
    result = summarize(fixture)
    assert result["status"] == "incomplete"
    assert result["counts"] == {"expected": 2, "success": 0, "failed": 0, "missing": 2}
    assert result["metrics"]["raw_model"]["reports_total"] == 2
    assert result["metrics"]["raw_model"]["overall"]["f1"] == 0


def test_duplicate_results_rejected_even_if_one_failed_and_one_success(fixture):
    first = job(fixture)
    second = job(fixture)
    output(fixture, first, mode="backend")
    output(fixture, second)
    with pytest.raises(ValueError, match="Duplicate generation result"):
        summarize(fixture)


@pytest.mark.parametrize("field,value", [("max_model_len", 200000), ("sample_timeout_seconds", 600),
                                         ("adapter", "/synthetic/another-adapter")])
def test_different_inference_parameters_must_be_summarized_separately(fixture, field, value):
    job(fixture, shards=2, shard_index=0)
    job(fixture, shards=2, shard_index=1, **{field: value})
    with pytest.raises(ValueError, match="parameters differ"):
        summarize(fixture)


@pytest.mark.parametrize("field", ["adapter_sha256", "runner_sha256", "inference_core_sha256", "base_index_sha256"])
def test_source_hash_mismatch_between_shards_rejected(fixture, field):
    job(fixture, shards=2, shard_index=0)
    other = job(fixture, shards=2, shard_index=1)
    other.manifest[field] = "f" * 64
    save(other.directory / "manifest.json", other.manifest)
    with pytest.raises(ValueError, match="source/adapter"):
        summarize(fixture)


@pytest.mark.parametrize("field,value", [("sft_sha256", "f" * 64), ("selection_manifest_sha256", "f" * 64)])
def test_generation_manifest_cannot_bind_other_inputs(fixture, field, value):
    generation = job(fixture)
    generation.manifest[field] = value
    save(generation.directory / "manifest.json", generation.manifest)
    with pytest.raises(ValueError, match="hash mismatch"):
        summarize(fixture)


@pytest.mark.parametrize("change", ["line", "partition", "sample", "prompt_ids", "call_prompt", "token_count", "raw", "prediction", "finish", "no_call"])
def test_record_binding_or_complete_output_receipt_tampering_rejected(fixture, change):
    generation = job(fixture)
    path, record = output(fixture, generation)
    if change == "line":
        record["line"] += 1
    elif change == "partition":
        record["partition"] = "final_acceptance"
    elif change == "sample":
        record["sample_sha256"] = "f" * 64
    elif change == "prompt_ids":
        record["prompt_ids_sha256"] = "f" * 64
        record["model_calls"][0]["prompt_sha256"] = "f" * 64  # Internally consistent forgery still fails CPU re-tokenization.
    elif change == "call_prompt":
        record["model_calls"][0]["prompt_sha256"] = "f" * 64
    elif change == "token_count":
        record["output_tokens"] += 1
    elif change == "raw":
        record["raw_output"] += " "
    elif change == "prediction":
        record["prediction"]["病害明细"].pop()
    elif change == "finish":
        record["finish_reason"] = record["model_calls"][0]["finish_reason"] = "length"
    else:
        record.update(status="failed", prediction=None, raw_output="", model_calls=[], error="context_budget: forged")
    save(path, record)
    with pytest.raises(ValueError):
        summarize(fixture)


def test_partition_mixing_rejected(fixture):
    job(fixture, partition="diagnostic")
    fixture.args.partition = "final_acceptance"
    with pytest.raises(ValueError, match="partition mismatch"):
        summarize(fixture)


def test_backend_call_receipt_cannot_be_reused_for_another_sample(fixture):
    generation = job(fixture)
    _, first = output(fixture, generation, 0)
    path, second = output(fixture, generation, 1)
    second["model_calls"][0]["call_id"] = first["model_calls"][0]["call_id"]
    save(path, second)
    with pytest.raises(ValueError, match="Duplicate backend call ID"):
        summarize(fixture)


def test_original_diagnostic_runner_layout_has_implicit_zero_prefill(fixture):
    generation = job(fixture)
    generation.manifest["parameters"].pop("prefill_chunk_size")
    generation.manifest["parameters"].pop("prefill_min_tokens")
    save(generation.directory / "manifest.json", generation.manifest)
    for index in range(len(generation.selected)):
        path, record = output(fixture, generation, index)
        record.pop("prefill_chunk_size")
        save(path, record)
    assert summarize(fixture)["status"] == "completed"


def test_new_runner_cannot_omit_explicit_zero_prefill_receipt(fixture):
    generation = job(fixture)
    path, record = output(fixture, generation)
    record.pop("prefill_chunk_size")
    save(path, record)
    with pytest.raises(ValueError, match="prefill setting"):
        summarize(fixture)


@pytest.mark.parametrize("change", ["sft", "prompt_count", "token_hash", "partition_count", "missing_sample"])
def test_frozen_cohort_and_cpu_tokenization_binding_fail_closed(fixture, change):
    job(fixture)
    manifest = deepcopy(fixture.selection)
    if change == "sft":
        fixture.args.sft.write_text(fixture.args.sft.read_text() + "\n")
    elif change == "prompt_count":
        manifest["samples"][0]["prompt_tokens"] += 1
    elif change == "token_hash":
        manifest["samples"][0]["full_sequence_token_sha256"] = "f" * 64
    elif change == "partition_count":
        manifest["partition_counts"]["diagnostic"] += 1
    else:
        manifest["samples"].pop()
    save(fixture.args.selection_manifest, manifest)
    with pytest.raises(ValueError):
        summarize(fixture)


@pytest.mark.parametrize("change", ["deduplicate", "new_fact", "normalize", "unknown_section"])
def test_independent_semantic_guard_rejects_candidate_fact_changes(fixture, monkeypatch, change):
    generation = job(fixture)
    output(fixture, generation)
    original = summary.preserve_model_report
    def corrupt(raw):
        candidate, audit = original(raw)
        if change == "deduplicate":
            candidate["病害明细"].pop()
        elif change == "new_fact":
            candidate["基本信息"]["总体评分"] = "90"
        elif change == "normalize":
            candidate["病害明细"][0]["是否新增"] = "否"
        else:
            candidate["unknown"] = "invented"
        return candidate, audit
    monkeypatch.setattr(summary, "preserve_model_report", corrupt)
    with pytest.raises(ValueError, match="Candidate changed"):
        summarize(fixture)


def test_cli_refuses_existing_output_before_loading_tokenizer(fixture):
    destination = fixture.root / "existing.json"
    destination.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        summary.main(["--sft", str(fixture.args.sft), "--selection-manifest", str(fixture.args.selection_manifest),
                      "--tokenizer", str(fixture.args.tokenizer), "--output", str(destination),
                      "--generation-dir", str(fixture.root), "--partition", "diagnostic"])
    assert destination.read_text() == "existing evidence"
