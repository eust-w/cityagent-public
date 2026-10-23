#!/usr/bin/env python3
"""Offline scoring of a fixed training-source batch of complete Agent reports.

Labels are read only here, after inference; this script has no model/network SDK.
Example (paths on the PPU, or an intact offline copy):
  python experiments/summarize_ppu_agent_throughput.py \
    --selection /benchmark/agent-selection.json --scoring /benchmark/scoring.json \
    --work-dir /runs/candidate/work --result-dir /runs/candidate/result \
    --require-tongagents-react --out-json /runs/candidate/evaluation.json

Only complete, source-bound, hash-verified report/Word pairs receive predictions.
Missing, failed and invalid reports remain in the fixed denominator with all gold
false negatives. A failed/truncated attempt may be retried successfully; an
accepted extraction/review must have a successful, normally stopped receipt.
Batch time comes from the final CLI summary, never summed document/request times.
Worker generation receipts omit TongAgents controller HTTP calls; those missing
calls/tokens are explicitly unknown, not inferred from SDK events or tokenization.
F1 is the repository's strict normalized multiset proxy against training labels,
not official scoring, factual verification, or an independent generalization test.
The explicit TongAgents benchmark profile checks consistent recorded SDK/model/
budget identities. It does not revalidate live tensors or compare two runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.evaluation.generation import aggregate_reports, score_report


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=reject)


def index_rows(rows: Any, key: str) -> dict[str, dict]:
    if not isinstance(rows, list):
        raise ValueError("Expected an array of records")
    result = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get(key), str)
                or not row[key] or row[key] in result):
            raise ValueError(f"Missing/duplicate record identity: {key}")
        result[row[key]] = row
    return result


def finite_nonnegative(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def sha256_string(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def react_profile_errors(identity: dict, final_summary: dict | None) -> list[str]:
    """Check recorded benchmark identity, without importing SDK/model code."""
    errors = []
    if identity.get("workflow") != "divide-react":
        errors.append("react_profile_identity_workflow_mismatch")
    if not final_summary or final_summary.get("workflow") != "divide-react":
        errors.append("react_profile_summary_workflow_mismatch")
    for field in ("sdk_version", "served_model"):
        if not nonempty_string(identity.get(field)):
            errors.append("react_profile_missing_or_invalid:" + field)
    if not sha256_string(identity.get("adapter_sha256")):
        errors.append("react_profile_missing_or_invalid:adapter_sha256")
    code = identity.get("code")
    if (not isinstance(code, dict) or not code
            or any(not nonempty_string(name) or not sha256_string(value) for name, value in code.items())):
        errors.append("react_profile_missing_or_invalid:code")
    model_identity = identity.get("model_identity")
    if not isinstance(model_identity, dict) or model_identity.get("kind") not in {"dynamic_lora", "merged_export"}:
        errors.append("react_profile_missing_or_invalid:model_identity")
    elif not nonempty_string(model_identity.get("adapter_path" if model_identity["kind"] == "dynamic_lora" else "path")):
        errors.append("react_profile_missing_or_invalid:model_identity_path")
    for field in ("agent_max_iterations", "agent_max_calls", "model_timeout", "max_model_len", "document_workers"):
        if not positive_integer(identity.get(field)):
            errors.append("react_profile_missing_or_invalid:" + field)
    if positive_integer(identity.get("document_workers")) and identity["document_workers"] > 8:
        errors.append("react_profile_document_workers_out_of_range")
    if identity.get("chat_tokenization_mode") not in {
        "server_chat_template", "verified_local_template+server_prompt_count",
    }:
        errors.append("react_profile_missing_or_invalid:chat_tokenization_mode")
    settings = identity.get("settings")
    required_settings = {"chunk_tokens", "output_tokens", "max_split_depth", "max_extraction_attempts",
                         "max_tasks", "max_context_units", "max_tool_tokens"}
    if (not isinstance(settings, dict) or not required_settings.issubset(settings)
            or any(not positive_integer(value) for value in settings.values())):
        errors.append("react_profile_missing_or_invalid:settings")
    elif (positive_integer(identity.get("max_model_len"))
          and settings["chunk_tokens"] + settings["output_tokens"] >= identity["max_model_len"]):
        errors.append("react_profile_chunk_budget_exceeds_context")
    return errors


def frozen_samples(selection_path: Path, scoring_path: Path) -> tuple[dict, list[dict]]:
    selection, scoring = read_json(selection_path), read_json(scoring_path)
    if selection.get("schema_version") != 1 or scoring.get("schema_version") != 1:
        raise ValueError("Unsupported frozen selection/scoring schema")
    if scoring.get("never_include_in_model_inputs") is not True:
        raise ValueError("Scoring labels must be explicitly separate from model inputs")
    selected = index_rows(selection.get("samples"), "id")
    labels = index_rows(scoring.get("samples"), "id")
    if not selected:
        raise ValueError("The fixed selection is empty")
    result, names = [], set()
    for sample_id, sample in selected.items():
        if sample_id not in labels:
            raise ValueError(f"Frozen scoring record missing: {sample_id}")
        label = labels[sample_id]
        source = Path(sample["input_docx"])
        gold_path = Path(sample["gold_path"])
        name = source.stem
        if name in names or name in {"", ".", ".."}:
            raise ValueError("Frozen inputs do not have unique document names")
        names.add(name)
        if (label.get("gold_path") != str(gold_path)
                or label.get("gold_sha256") != sample.get("gold_sha256")
                or file_digest(gold_path) != sample.get("gold_sha256")):
            raise ValueError(f"Gold binding mismatch: {sample_id}")
        if file_digest(source) != sample.get("input_docx_sha256"):
            raise ValueError(f"Frozen input changed: {sample_id}")
        if source.resolve() == gold_path.resolve():
            raise ValueError("Input and scoring label paths must be different")
        gold = read_json(gold_path)
        score_report(gold, None)  # Reject invalid gold; never shrink the denominator.
        result.append({**sample, "name": name, "gold": gold})
    return selection, result


def coverage_valid(manifest: dict, tasks: list[dict]) -> bool:
    """Recompute primary text coverage; this is not a semantic-recall check."""
    units = index_rows(manifest.get("units"), "id")
    if not units:
        return False
    intervals = {key: [] for key in units}
    for task in tasks:
        if task.get("status") not in {"completed", "split"}:
            return False
        if task["status"] == "split":
            continue
        for span in task.get("spans", []):
            key, start, end = span.get("coverage_id"), span.get("start"), span.get("end")
            if (key not in units or type(start) is not int or type(end) is not int
                    or not 0 <= start <= end <= len(units[key]["text"])):
                return False
            intervals[key].append((start, end))
    for key, unit in units.items():
        cursor = 0
        for start, end in sorted(intervals[key]):
            if start > cursor:
                return False
            cursor = max(cursor, end)
        if cursor != len(unit["text"]):
            return False
    return True


def validate_prediction(sample: dict, row: dict | None, execution: dict, manifest: dict,
                        prediction: dict, result_dir: Path, fingerprint: str, *,
                        react_identity: dict | None = None) -> list[str]:
    errors = []
    if row is None or row.get("status") != "completed":
        errors.append("batch_does_not_mark_document_completed")
    if execution.get("status") != "success":
        errors.append("workflow_not_successful")
    if execution.get("run_fingerprint") != fingerprint:
        errors.append("execution_run_fingerprint_mismatch")
    if react_identity is not None:
        if execution.get("mode") != "divide-react":
            errors.append("react_profile_execution_mode_mismatch")
        if execution.get("settings") != react_identity.get("settings"):
            errors.append("react_profile_execution_settings_mismatch")
    if (execution.get("manifest_sha256") != digest(manifest)
            or execution.get("document_sha256") != manifest.get("document_sha256")
            or manifest.get("original_document_sha256", manifest.get("document_sha256"))
            != sample["input_docx_sha256"]):
        errors.append("source_manifest_binding_mismatch")
    tasks = execution.get("tasks", [])
    if (execution.get("coverage", {}).get("complete") is not True
            or not tasks or not coverage_valid(manifest, tasks)):
        errors.append("incomplete_text_coverage")
    if execution.get("report_sha256") != digest(prediction):
        errors.append("report_hash_mismatch")
    calls = index_rows(execution.get("model_calls", []), "call_id")
    accepted = [task.get("accepted_call_id") for task in tasks if task.get("status") == "completed"]
    for review in execution.get("reviews", []):
        successful = [a for a in review.get("attempts", []) if a.get("status") == "success"]
        if review.get("resolved") is not True or not successful:
            errors.append("unresolved_review")
        else:
            accepted.append(successful[-1].get("call_id"))
    if not accepted:
        errors.append("no_accepted_model_generation")
    for call_id in accepted:
        call = calls.get(call_id, {})
        if (call.get("kind") != "generation" or call.get("status") != "success"
                or call.get("finish_reason") != "stop"):
            errors.append("accepted_generation_missing_failed_or_truncated")
        if react_identity is not None and call.get("model") != react_identity.get("served_model"):
            errors.append("react_profile_accepted_generation_model_mismatch")
    if execution.get("mode") == "divide-react":
        runs = execution.get("agent_runs", [])
        if not runs or any(run.get("status") != "completed"
                           or run.get("native_final_event") is not True
                           or run.get("output_truncated") is True
                           or run.get("external_model_used") is not False
                           or not run.get("tool_calls") for run in runs):
            errors.append("sdk_completion_not_verified")
        if react_identity is not None and any(run.get("sdk_version") != react_identity.get("sdk_version") for run in runs):
            errors.append("react_profile_execution_sdk_version_mismatch")
    elif execution.get("mode") != "divide-fixed":
        errors.append("unknown_workflow_mode")
    output = execution.get("output", {})
    expected_name = sample["name"] + ".docx"
    output_path = result_dir / expected_name
    if (output.get("name") != expected_name or not row or row.get("output") != expected_name
            or not output_path.is_file() or file_digest(output_path) != output.get("sha256")):
        errors.append("published_word_missing_or_hash_mismatch")
    return sorted(set(errors))


def observed_work(executions: list[dict]) -> dict:
    calls, runs, tasks, reviews = [], [], [], []
    malformed = 0

    def records(container: dict, key: str) -> list[dict]:
        nonlocal malformed
        value = container.get(key, [])
        if not isinstance(value, list):
            malformed += 1
            return []
        malformed += sum(not isinstance(row, dict) for row in value)
        return [row for row in value if isinstance(row, dict)]

    for execution in executions:
        # Request IDs are unique in each LocalModelClient. Do not inflate totals
        # by replaying duplicate audit rows; show the duplicates as a limitation.
        calls.extend(records(execution, "model_calls"))
        runs.extend(records(execution, "agent_runs"))
        tasks.extend(records(execution, "tasks"))
        reviews.extend(records(execution, "reviews"))
    unique, duplicates = {}, 0
    for call in calls:
        key = call.get("call_id")
        if not isinstance(key, str) or key in unique:
            duplicates += 1
        else:
            unique[key] = call
    calls = list(unique.values())
    generations = [call for call in calls if call.get("kind") == "generation"]
    tokens = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(call["usage"].get(field) if isinstance(call.get("usage"), dict) else None)
                  for call in generations]
        present = [value for value in values if type(value) is int and value >= 0]
        tokens[field] = {"observed_sum": sum(present), "receipts_with_value": len(present),
                         "receipts_missing_value": len(values) - len(present),
                         "complete_sum": sum(present) if len(present) == len(values) else None}
    tool_calls = [call for run in runs for call in records(run, "tool_calls")]
    extraction_attempts = [records(task, "attempts") for task in tasks]
    review_attempts = [records(review, "attempts") for review in reviews]
    return {
        "scope": "existing receipts, including failures; worker extraction/review only for generation and tokens",
        "http_requests_by_kind": dict(Counter(str(call.get("kind")) for call in calls)),
        "duplicate_or_missing_request_ids": duplicates,
        "malformed_audit_records_or_containers": malformed,
        "worker_generation_requests": len(generations),
        "worker_generation_failures": sum(call.get("status") != "success" for call in generations),
        "worker_generation_length_finishes": sum(call.get("finish_reason") == "length" for call in generations),
        "worker_tokens": tokens,
        "sdk_sessions": len(runs),
        "sdk_sessions_completed": sum(run.get("status") == "completed" for run in runs),
        "sdk_tool_calls": len(tool_calls),
        "sdk_tool_failures": sum(call.get("status") != "success" for call in tool_calls),
        "sdk_tool_calls_by_name": dict(Counter(str(call.get("name")) for call in tool_calls)),
        "sdk_controller_generation_requests": None,
        "sdk_controller_tokens": None,
        "whole_workflow_generation_requests": None,
        "whole_workflow_tokens": None,
        "missing_controller_receipts_note": "SDK native events/tools do not establish HTTP generation counts or usage",
        "extraction_attempts": sum(map(len, extraction_attempts)),
        "extraction_retry_attempts": sum(max(0, len(attempts) - 1) for attempts in extraction_attempts),
        "split_tasks": sum(task.get("status") == "split" for task in tasks),
        "review_attempts": sum(map(len, review_attempts)),
        "review_retry_attempts": sum(max(0, len(attempts) - 1) for attempts in review_attempts),
    }


def summarize(selection_path: Path, scoring_path: Path, work_dir: Path, result_dir: Path, *,
              require_tongagents_react: bool = False) -> dict:
    selection, samples = frozen_samples(selection_path, scoring_path)
    global_errors, final_summary, identity = [], None, {}
    fingerprint = ""
    try:
        bound = read_json(work_dir / "run_manifest.json")
        identity, fingerprint = bound["identity"], bound["run_fingerprint"]
        if not isinstance(identity, dict):
            identity = {}
            raise ValueError("Run identity must be an object")
        if digest(identity) != fingerprint:
            global_errors.append("run_identity_hash_mismatch")
        expected_sources = {str(Path(s["input_docx"]).resolve()): s["input_docx_sha256"] for s in samples}
        if identity.get("sources") != expected_sources:
            global_errors.append("run_inputs_differ_from_fixed_selection")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        global_errors.append("run_manifest_unavailable_or_invalid:" + type(exc).__name__)
    summaries = {}
    try:
        final_summary = read_json(work_dir / "summary.json")
        summaries = index_rows(final_summary.get("documents"), "name")
        if (final_summary.get("run_fingerprint") != fingerprint
                or final_summary.get("reports") != len(samples)
                or set(summaries) != {s["name"] for s in samples}):
            global_errors.append("batch_summary_binding_or_denominator_mismatch")
        if any(row.get("status") not in {"completed", "failed"} for row in summaries.values()):
            global_errors.append("batch_summary_nonterminal_document")
        if final_summary.get("completed_reports") != sum(row.get("status") == "completed" for row in summaries.values()):
            global_errors.append("batch_summary_completed_count_mismatch")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        global_errors.append("final_batch_summary_unavailable_or_invalid:" + type(exc).__name__)
        final_summary = None
    if require_tongagents_react:
        global_errors.extend(react_profile_errors(identity, final_summary))
    records, executions, scores, fields = [], [], [], {}
    for sample in samples:
        directory = work_dir / "documents" / sample["name"]
        prediction, execution = None, None
        errors = list(global_errors)
        try:
            execution = read_json(directory / "execution.json")
            if not isinstance(execution, dict):
                raise ValueError("Execution must be an object")
            executions.append(execution)
            manifest = read_json(directory / "manifest.json")
            prediction = read_json(directory / "report.json")
            errors.extend(validate_prediction(sample, summaries.get(sample["name"]), execution,
                                              manifest, prediction, result_dir, fingerprint,
                                              react_identity=identity if require_tongagents_react else None))
            score_report(sample["gold"], prediction)  # Validate metric schema before accepting.
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append("document_evidence_unavailable_or_invalid:" + type(exc).__name__)
        if errors:
            prediction = None
        score = score_report(sample["gold"], prediction)
        scores.append(score)
        basic_gold = sample["gold"]["基本信息"]
        basic_prediction = prediction["基本信息"] if prediction is not None else {}
        for field in sorted(set(basic_gold) | set(basic_prediction)):
            one_gold = {"基本信息": {field: basic_gold[field]} if field in basic_gold else {},
                        "维修建议": [], "病害明细": []}
            one_prediction = ({"基本信息": {field: basic_prediction[field]} if field in basic_prediction else {},
                               "维修建议": [], "病害明细": []} if prediction is not None else None)
            fields.setdefault(field, []).append(score_report(one_gold, one_prediction))
        records.append({"id": sample["id"], "name": sample["name"],
                        "input_docx_sha256": sample["input_docx_sha256"], "gold_sha256": sample["gold_sha256"],
                        "status": "completed" if prediction is not None else "failed" if execution is not None else "missing",
                        "errors": sorted(set(errors)), "score": score,
                        "workflow_status": execution.get("status") if execution else None,
                        "document_seconds": summaries.get(sample["name"], {}).get("seconds")})
    completed = sum(record["status"] == "completed" for record in records)
    wall = final_summary.get("seconds") if final_summary else None
    timing_valid = bool(not global_errors and finite_nonnegative(wall) and wall > 0)
    if not timing_valid:
        wall = None
    metrics = aggregate_reports(scores)
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "incomplete" if final_summary is None else "completed" if completed == len(samples)
                  else "completed_with_failures",
        "interpretation": "training_label_proxy_not_official_score_or_independent_generalization",
        "evaluation_profile": "tongagents-react" if require_tongagents_react else "generic-divide-workflow",
        "fixed_selection_sha256": file_digest(selection_path), "scoring_manifest_sha256": file_digest(scoring_path),
        "dataset_split": selection.get("dataset_split"), "benchmark_scope": selection.get("scope"),
        "sample_ids": [s["id"] for s in samples],
        "work_dir": str(work_dir), "result_dir": str(result_dir), "run_fingerprint": fingerprint or None,
        "global_errors": global_errors, "timing_valid": timing_valid,
        "batch_wall_seconds": wall, "reports_expected": len(samples), "reports_completed": completed,
        "reports_failed_or_missing": len(samples) - completed,
        "completion_rate": completed / len(samples),
        "completion_measure": "structural/audit completion, not a semantic-quality passing threshold",
        "completed_reports_per_hour": completed * 3600 / wall if wall is not None else None,
        "all_reports_audit_complete": completed == len(samples) and timing_valid,
        "timing_scope": final_summary.get("timing_scope") if final_summary else None,
        "timing_note": "Final CLI batch wall includes failures, process startup, conversion, SDK/model retries and Word publication; excludes model-server startup and CLI preflight",
        "identity": identity,
        "metrics": metrics,
        "basic_field_metrics": {field: aggregate_reports(values)["sections"]["basic"] for field, values in sorted(fields.items())},
        "observed_work": observed_work(executions), "records": records,
        "limitations": ["Text coverage does not establish semantic recall or evidence correctness.",
                        "Training labels can contain factual errors; exact-label F1 does not settle source conflicts.",
                        "No partial JSON rescue, row deduplication, or omission of failed reports in the denominator.",
                        "This fixed short/medium-report batch does not establish long-report performance.",
                        "Identity checks verify recorded bindings, not live model tensors; comparing runs still requires matching model/code/semantic budgets and examining all fixed-denominator PRF metrics.",
                        "SDK controller request counts/tokens are unavailable in current workflow receipts."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("selection", "scoring", "work-dir", "result-dir", "out-json"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--require-tongagents-react", action="store_true",
                        help="Require consistent divide-react, SDK, model and budget identities for this benchmark")
    args = parser.parse_args(argv)
    report = summarize(args.selection, args.scoring, args.work_dir, args.result_dir,
                       require_tongagents_react=args.require_tongagents_react)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "reports_expected", "reports_completed",
                     "batch_wall_seconds", "completed_reports_per_hour")}, ensure_ascii=False))
    return 0 if report["all_reports_audit_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
