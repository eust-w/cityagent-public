#!/usr/bin/env python3
"""CPU-only, full-partition generation evaluation from frozen SFT evidence.

Missing and explicitly failed predictions retain every gold false negative.
Integrity errors and duplicate results are rejected, never selected by score.
These are internal frozen-label agreement metrics, not competition scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.inference_core import _parse_report, prompt_ids_sha256
from bridge_evidence.contest.model_preserving_fusion import preserve_model_report
from bridge_evidence.evaluation.generation import aggregate_reports, score_report


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("summary_generation_runner", ROOT / "scripts/evaluate_generation_nvidia.py")
selection_helper = load_script("summary_selection_helper", ROOT / "scripts/prepare_generation_manifest.py")
bench = runner.bench
HEX = re.compile(r"[0-9a-f]{64}\Z")
PARTITIONS = ("diagnostic", "final_acceptance")
SOURCE_HASHES = (
    "sft_sha256", "selection_manifest_sha256", "runner_sha256", "helper_sha256",
    "inference_core_sha256", "metrics_sha256", "base_config_sha256", "base_index_sha256",
    "adapter_sha256", "adapter_config_sha256",
)
SELECTION_PARAMETERS = {"out_dir", "shard_index", "shards", "limit", "sample_sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path, snapshots):
    data = path.read_bytes()
    snapshots[path.resolve()] = hashlib.sha256(data).hexdigest()
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Nonfinite JSON value: " + value)


def same(actual, expected, message):
    require(type(actual) is type(expected) and actual == expected, message)


def sha(value, name):
    require(isinstance(value, str) and HEX.fullmatch(value) is not None, name + ": invalid SHA256")
    return value


def positive_int(value, name):
    require(type(value) is int and value > 0, name + ": positive integer required")
    return value


def validate_selection(manifest, rows, sft_sha, tokenizer):
    """Recompute token hashes on CPU; no model or answer is passed to a backend."""
    same(manifest.get("schema_version"), 1, "Unsupported selection manifest version")
    same(manifest.get("sft_sha256"), sft_sha, "Selection SFT hash mismatch")
    same(manifest.get("source_row_count"), len(rows), "Selection source-row count mismatch")
    same(manifest.get("holdout_year"), "2018", "Selection holdout year mismatch")
    expected = {row["sample_sha256"]: row for row in rows if row["year"] == "2018"}
    samples = manifest.get("samples")
    require(isinstance(samples, list) and bool(samples), "Selection samples must be nonempty")
    by_id, prompts = {}, {}
    for sample in samples:
        require(isinstance(sample, dict), "Selection sample must be an object")
        sample_id = sha(sample.get("sample_sha256"), "sample_sha256")
        require(sample_id not in by_id and sample_id in expected, "Duplicate or unknown frozen sample")
        row = expected[sample_id]
        for field, value in (("line", row["line"]), ("year", "2018"),
                             ("prompt_content_sha256", row["prompt_sha256"])):
            same(sample.get(field), value, "Selection/SFT " + field + " mismatch")
        require(sample.get("partition") in PARTITIONS, "Unknown frozen partition")
        tokenized = bench.tokenize_row(row, tokenizer)
        for frozen, recomputed in (("prompt_tokens", "prompt_tokens"), ("full_tokens", "tokens"),
                                   ("answer_tokens", "answer_tokens"),
                                   ("full_sequence_token_sha256", "token_sha256")):
            same(sample.get(frozen), tokenized[recomputed], "Frozen token metadata mismatch: " + frozen)
        same(sample.get("prompt_length_bucket"), selection_helper.prompt_length_bucket(tokenized["prompt_tokens"]),
             "Frozen prompt bucket mismatch")
        prompts[sample_id] = prompt_ids_sha256(tokenized["input_ids"][:tokenized["prompt_tokens"]])
        by_id[sample_id] = sample
    same(set(by_id), set(expected), "Selection must cover the complete internal 2018 holdout")
    same(manifest.get("sample_count"), len(samples), "Selection sample count mismatch")
    for name, field in (("cohort_sha256", "sample_sha256"),
                        ("cohort_prompt_content_sha256", "prompt_content_sha256")):
        same(manifest.get(name), bench.digest(sorted(sample[field] for sample in samples)), name + " mismatch")
    groups = {partition: [sample for sample in samples if sample["partition"] == partition]
              for partition in PARTITIONS}
    same(manifest.get("partition_counts"), {key: len(group) for key, group in groups.items()},
         "Frozen partition count mismatch")
    for name, field in (("partition_sha256", "sample_sha256"),
                        ("partition_prompt_content_sha256", "prompt_content_sha256")):
        same(manifest.get(name), {key: bench.digest(sorted(s[field] for s in group))
                                 for key, group in groups.items()}, "Frozen " + name + " mismatch")
    return by_id, prompts


def validate_job(manifest, selection, rows, partition, selection_sha, sft_sha):
    require(isinstance(manifest, dict), "Generation manifest must be an object")
    for key in SOURCE_HASHES:
        sha(manifest.get(key), key)
    same(manifest["sft_sha256"], sft_sha, "Generation SFT hash mismatch")
    same(manifest["selection_manifest_sha256"], selection_sha, "Generation selection hash mismatch")
    params = manifest.get("parameters")
    require(isinstance(params, dict), "Generation parameters missing")
    same(params.get("partition"), partition, "Diagnostic/final-acceptance partition mismatch")
    for key in ("model", "adapter", "sft", "selection_manifest", "out_dir"):
        require(isinstance(params.get(key), str) and Path(params[key]).is_absolute(),
                "Generation parameter must be an absolute path: " + key)
    for key in ("max_model_len", "max_new_tokens", "gpu_memory_gib", "sample_timeout_seconds", "shards"):
        positive_int(params.get(key), key)
    require(type(params.get("shard_index")) is int, "shard_index must be integer")
    if params.get("limit") is not None:
        positive_int(params["limit"], "limit")
    require(type(params.get("prefill_chunk_size", 0)) is int and params.get("prefill_chunk_size", 0) >= 0,
            "Invalid prefill chunk size")
    positive_int(params.get("prefill_min_tokens", 32768), "prefill_min_tokens")
    chosen = runner.select_samples(selection, rows, partition=partition, shards=params["shards"],
                                   shard_index=params["shard_index"], limit=params.get("limit"),
                                   sample_sha256=params.get("sample_sha256"))
    same(manifest.get("selected_sample_sha256"), [sample["sample_sha256"] for sample, _ in chosen],
         "Generation selected samples differ from its frozen selection/shard arguments")
    require(isinstance(manifest.get("versions"), dict) and bool(manifest["versions"]), "Runtime versions missing")
    require(isinstance(manifest.get("generation"), str) and bool(manifest["generation"]), "Generation strategy missing")
    # Selection controls may differ; inference budgets, source/adapter identities,
    # backend versions and all remaining arguments must match exactly.
    return {"source_hashes": {key: manifest[key] for key in SOURCE_HASHES},
            "parameters": {key: value for key, value in params.items() if key not in SELECTION_PARAMETERS},
            "versions": manifest["versions"], "generation": manifest["generation"]}


def validate_record(record, sample, prompt_hash, params):
    require(isinstance(record, dict), "Generation result must be an object")
    for key, value in sample.items():
        same(record.get(key), value, "Result/frozen sample mismatch: " + key)
    same(record.get("prompt_ids_sha256"), prompt_hash, "Result prompt token hash mismatch")
    chunk = params.get("prefill_chunk_size", 0)
    expected_chunk = chunk if chunk and sample["prompt_tokens"] >= params.get("prefill_min_tokens", 32768) else 0
    # The original frozen runner predates chunked prefill and records neither
    # flag nor per-result value. New manifests must retain the explicit receipt.
    actual_chunk = record.get("prefill_chunk_size", 0) if "prefill_chunk_size" not in params else record.get("prefill_chunk_size")
    same(actual_chunk, expected_chunk, "Result prefill setting differs from its manifest")
    require(record.get("status") in {"success", "failed"}, "Unknown generation result status")
    calls = record.get("model_calls")
    require(isinstance(calls, list) and len(calls) <= 1, "Frozen generation requires zero or one backend attempt")
    raw = record.get("raw_output")
    require(isinstance(raw, str), "raw_output must be a string")
    over_budget = sample["prompt_tokens"] + params["max_new_tokens"] > params["max_model_len"]
    if not calls:
        require(over_budget and record["status"] == "failed" and record.get("prediction") is None
                and not raw and str(record.get("error", "")).startswith("context_budget:"),
                "Missing backend call is only valid for a proven context-budget failure")
        return None
    require(not over_budget, "Backend called with an over-budget full prompt")
    call = calls[0]
    require(isinstance(call, dict), "Call receipt must be an object")
    for key in ("call_id", "utc"):
        require(isinstance(call.get(key), str) and bool(call[key]), "Missing call " + key)
    same(call.get("attempt"), 1, "Unexpected backend attempt index")
    same(call.get("prompt_sha256"), prompt_hash, "Call prompt token hash mismatch")
    same(call.get("prompt_tokens"), sample["prompt_tokens"], "Call prompt length mismatch")
    require(type(call.get("seconds")) in (int, float) and math.isfinite(call["seconds"])
            and call["seconds"] >= 0, "Invalid call duration")
    same(record.get("seconds"), call["seconds"], "Result/call duration mismatch")
    same(record.get("finish_reason"), call.get("finish_reason"), "Result/call finish mismatch")
    same(record.get("output_tokens"), call.get("output_tokens"), "Result/call token count mismatch")
    if call.get("output_sha256") is not None:
        same(call["output_sha256"], hashlib.sha256(raw.encode("utf-8")).hexdigest(), "Raw model output hash mismatch")
    else:
        require(not raw, "Raw output has no receipt hash")
    require(isinstance(call.get("errors"), list) and all(isinstance(e, str) and e for e in call["errors"]),
            "Invalid call errors")
    if call.get("status") == "success":
        require(call["errors"] == [] and call.get("finish_reason") == "stop", "Successful call lacks complete stop receipt")
        tokens = positive_int(call.get("output_tokens"), "actual output tokens")
        require(tokens <= params["max_new_tokens"], "Actual output exceeds reserved budget")
        sha(call.get("output_sha256"), "Call output hash")
        parsed = _parse_report(raw)
        if record["status"] == "success":
            same(record.get("prediction"), parsed, "Stored prediction differs from complete raw JSON")
            return parsed
        # The runner may reject a structurally valid model JSON at the stricter
        # metric-schema boundary (e.g. normalized field-name collisions).
        require(record.get("prediction") is None and str(record.get("error", "")).startswith("metric_schema:"),
                "Successful call was discarded without a metric-schema failure")
        try:
            score_report({"基本信息": {}, "维修建议": [], "病害明细": []}, parsed)
        except ValueError:
            return None
        raise ValueError("Claimed metric-schema failure is not reproducible")
    require(call.get("status") == "failed" and bool(call["errors"]) and record["status"] == "failed"
            and record.get("prediction") is None and bool(record.get("error")), "Invalid failure receipt")
    return None


def verify_preservation(raw, candidate):
    """Independently check exact semantic values and duplicate-row multiplicity."""
    require(set(candidate) == set(raw), "Candidate changed top-level sections")
    for key in raw:
        if key not in {"基本信息", "维修建议", "病害明细"}:
            same(candidate[key], raw[key], "Candidate changed an unknown section")
    same({key: value for key, value in candidate["基本信息"].items() if key != "建议统计"},
         {key: value for key, value in raw["基本信息"].items() if key != "建议统计"},
         "Candidate changed semantic basic information")
    multiset_hashes = {}
    for section in ("维修建议", "病害明细"):
        def rows(report):
            return Counter(json.dumps({key: value for key, value in row.items() if key != "序号"},
                                      ensure_ascii=False, sort_keys=True, allow_nan=False)
                           for row in report[section])
        before, after = rows(raw), rows(candidate)
        same(after, before, "Candidate changed semantic row multiset: " + section)
        multiset_hashes[section] = bench.digest(sorted(before.items()))
    return {"verified": True, "row_multiset_sha256": multiset_hashes,
            "allowed_changes": ["row 序号", "基本信息.建议统计 derived from actual advice rows"]}


def summarize(args, *, tokenizer):
    require(args.partition in PARTITIONS, "Unknown requested partition")
    snapshots = {}
    selection = read_json(args.selection_manifest, snapshots)
    sft_sha = bench.file_digest(args.sft)
    snapshots[args.sft.resolve()] = sft_sha
    rows = bench.load_rows(args.sft)
    samples, prompts = validate_selection(selection, rows, sft_sha, tokenizer)
    by_id = {row["sample_sha256"]: row for row in rows}
    selected = [sample for sample in selection["samples"] if sample["partition"] == args.partition]
    require(bool(selected), "Requested frozen partition is empty")
    selection_sha = snapshots[args.selection_manifest.resolve()]
    common, jobs, found, seen_dirs, call_ids = None, [], {}, set(), set()
    for directory in args.generation_dir:
        directory = directory.resolve()
        require(directory not in seen_dirs, "Duplicate generation directory")
        seen_dirs.add(directory)
        manifest = read_json(directory / "manifest.json", snapshots)
        signature = validate_job(manifest, selection, rows, args.partition, selection_sha, sft_sha)
        if common is None:
            common = signature
        else:
            same(signature, common, "Generation source/adapter/runtime/parameters differ; summarize runs separately")
        selected_ids = set(manifest["selected_sample_sha256"])
        observed = []
        for path in sorted(directory.glob("*.json")):
            if not HEX.fullmatch(path.stem):
                continue
            require(path.stem in selected_ids, "Result was not selected in its generation manifest")
            require(path.stem not in found, "Duplicate generation result; replacements/best-of are not supported")
            record = read_json(path, snapshots)
            prediction = validate_record(record, samples[path.stem], prompts[path.stem], manifest["parameters"])
            for call in record["model_calls"]:
                require(call["call_id"] not in call_ids, "Duplicate backend call ID across generation results")
                call_ids.add(call["call_id"])
            found[path.stem] = (record, prediction, path)
            observed.append(path.stem)
        summary = None
        summary_path = directory / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path, snapshots)
            same(summary.get("manifest_sha256"), snapshots[(directory / "manifest.json").resolve()],
                 "Child summary belongs to another manifest")
            same(summary.get("partition"), args.partition, "Child summary partition mismatch")
            same(summary.get("samples"), len(selected_ids), "Child summary selection count mismatch")
        jobs.append({"generation_dir": str(directory), "manifest_sha256": snapshots[(directory / "manifest.json").resolve()],
                     "manifest": manifest, "observed_sample_sha256": observed,
                     "missing_selected_sample_sha256": sorted(selected_ids - set(observed)),
                     "child_summary": {"present": summary is not None,
                                       "sha256": snapshots.get(summary_path.resolve()),
                                       "status": summary.get("status") if summary else None},
                     "completion_note": "Child summary status is provenance only; completeness is recomputed from frozen IDs."})
    require(bool(jobs), "At least one generation directory is required")
    records, counts = [], Counter()
    for sample in selected:
        sample_id = sample["sample_sha256"]
        row = by_id[sample_id]
        gold = json.loads(row["output"], object_pairs_hook=unique_object, parse_constant=reject_constant)
        score_report(gold, None)  # Broken gold is fatal, never relabelled a model failure.
        record, prediction, path = found.get(sample_id, (None, None, None))
        status = record["status"] if record is not None else "missing"
        counts[status] += 1
        raw_score = score_report(gold, prediction)
        candidate, audit, invariant = None, None, None
        if prediction is not None:
            candidate, audit = preserve_model_report(prediction)
            invariant = verify_preservation(prediction, candidate)
        candidate_score = score_report(gold, candidate)
        records.append({**sample, "status": status, "prediction_valid": prediction is not None,
                        "generation_path": str(path) if path else None,
                        "generation_file_sha256": snapshots.get(path.resolve()) if path else None,
                        "prompt_ids_sha256": prompts[sample_id],
                        "model_calls": record.get("model_calls", []) if record else [],
                        "error": record.get("error") if record else "missing_result",
                        "finish_reason": record.get("finish_reason") if record else None,
                        "output_tokens": record.get("output_tokens") if record else None,
                        "strategies": {"raw_model": {"prediction_sha256": bench.digest(prediction) if prediction is not None else None,
                                                     "score": raw_score},
                                       "model_preserving": {"prediction_sha256": bench.digest(candidate) if candidate is not None else None,
                                                            "score": candidate_score}},
                        "candidate_audit": audit, "semantic_preservation": invariant})
    for path, expected_sha in snapshots.items():
        same(bench.file_digest(path), expected_sha, "Input changed while summarizing: " + str(path))
    counts = {key: counts[key] for key in ("success", "failed", "missing")}
    return {"schema_version": 1, "utc": datetime.now(timezone.utc).isoformat(),
            "status": "incomplete" if counts["missing"] else ("completed_with_failures" if counts["failed"] else "completed"),
            "partition": args.partition, "counts": {"expected": len(selected), **counts},
            "all_results_present": not counts["missing"], "all_predictions_valid": counts["success"] == len(selected),
            "interpretation": "Internal frozen-label agreement, not official competition score or proof of factual correctness. No pass threshold is implied.",
            "scope": "Full frozen partition, including missing/failed/context-budget cases as prediction=None and all gold false negatives.",
            "partition_warning": "diagnostic is development data; final_acceptance must not be used for tuning",
            "replacement_policy": "reject_duplicate_results_no_best_of",
            "provenance": {"sft": str(args.sft.resolve()), "sft_sha256": sft_sha,
                           "selection_manifest": str(args.selection_manifest.resolve()),
                           "selection_manifest_sha256": selection_sha,
                           "frozen_partition_sha256": selection["partition_sha256"][args.partition],
                           "token_verification": "CPU retokenization of full frozen 2018 cohort; exact full-sequence and prompt-prefix hashes",
                           "tokenizer": str(args.tokenizer.resolve()),
                           "tokenizer_class": type(tokenizer).__module__ + "." + type(tokenizer).__qualname__,
                           "common_generation_signature": common,
                           "source_sha256": {str(path.relative_to(ROOT)): bench.file_digest(path) for path in (
                               Path(__file__), Path(bench.__file__), ROOT / "scripts/prepare_generation_manifest.py",
                               ROOT / "scripts/evaluate_generation_nvidia.py",
                               ROOT / "src/bridge_evidence/contest/inference_core.py",
                               ROOT / "src/bridge_evidence/contest/model_preserving_fusion.py",
                               ROOT / "src/bridge_evidence/evaluation/generation.py")}},
            "jobs": jobs, "records": records,
            "metrics": {name: aggregate_reports(record["strategies"][name]["score"] for record in records)
                        for name in ("raw_model", "model_preserving")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sft", "selection-manifest", "tokenizer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, action="append", required=True,
                        help="Exact child output directory containing manifest.json; repeat for disjoint shards")
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    from transformers import AutoTokenizer  # CPU only; never downloads or loads model weights.
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    result = summarize(args, tokenizer=tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bench.write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "partition": result["partition"], "counts": result["counts"]}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
