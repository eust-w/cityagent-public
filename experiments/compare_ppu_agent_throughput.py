#!/usr/bin/env python3
"""Compare two offline complete-Agent evaluation files, without running models.

  python experiments/compare_ppu_agent_throughput.py \
    --reference C1/evaluation.json --candidate C4/evaluation.json \
    --out-json comparison.json

Inputs must come from summarize_ppu_agent_throughput.py's explicit
--require-tongagents-react profile. Frozen samples/gold and every semantic
identity field must match. The only identity exceptions are requested/effective
document worker counts and narrowly listed HTTP/model-catalog metadata. All
exceptions and actual differences are reported. Unknown new fields are compared,
not silently ignored. Counts and PRF are recomputed from each report's metric
receipts, including failures, rather than trusting a cached aggregate F1.

Exit zero means comparable evidence, not quality passed or a recommended winner.
No semantic passing threshold is defined. Pooled, macro and per-report metric
regressions are shown even when every report passed structural/audit gates or
the pooled scores are unchanged. This does not revalidate live
model tensors or establish performance on long reports or the official test set.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_ppu_agent_throughput as single


def differences(before: Any, after: Any, path: str = "") -> list[dict]:
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            child = path + "." + str(key) if path else str(key)
            if key not in before or key not in after:
                result.append({"path": child, "reference_present": key in before,
                               "candidate_present": key in after,
                               "reference": before.get(key), "candidate": after.get(key)})
            else:
                result.extend(differences(before[key], after[key], child))
        return result
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [change for index, (left, right) in enumerate(zip(before, after))
                for change in differences(left, right, f"{path}[{index}]")]
    return [] if type(before) is type(after) and before == after else [
        {"path": path, "reference": before, "candidate": after}]


def comparison_identity(identity: dict) -> tuple[dict, dict]:
    """Remove only explicit concurrency or nonsemantic receipt metadata."""
    result, removed = deepcopy(identity), {}

    def omit(container: dict, key: str, path: str):
        if key in container:
            removed[path + "." + key if path else key] = container.pop(key)

    for key in ("document_workers", "effective_document_workers"):
        omit(result, key, "")
    # Do not omit the catalog or permissions themselves: model id/root/parent,
    # owners and every permission flag remain part of the exact comparison.
    for index, model in enumerate(result.get("model_catalog", [])):
        path = f"model_catalog[{index}]"
        omit(model, "created", path)
        for number, permission in enumerate(model.get("permission", [])):
            for key in ("id", "created"):
                omit(permission, key, f"{path}.permission[{number}]")
    for index, call in enumerate(result.get("preflight_model_calls", [])):
        path = f"preflight_model_calls[{index}]"
        for key in ("call_id", "started_at", "seconds"):
            omit(call, key, path)
        catalog_call = (call.get("kind") == "models" and call.get("endpoint") == "/v1/models"
                        and bool(result.get("model_catalog")))
        info_call = (call.get("kind") == "model_identity" and call.get("endpoint") == "/model_info"
                     and isinstance(result.get("model_identity", {}).get("model_info"), dict))
        # Those raw response hashes can change with catalog creation times.
        # Their parsed semantic bodies are retained above/in model_identity.
        # No hashes are discarded for tokenize/generation/unknown endpoints.
        if call.get("method") == "GET" and (catalog_call or info_call):
            for key in ("response_sha256", "output_sha256"):
                omit(call, key, path)
    return result, removed


def metric_counts(value: dict) -> dict:
    keys = ("tp", "fp", "fn", "predicted_count", "gold_count")
    if (not isinstance(value, dict) or any(type(value.get(key)) is not int or value[key] < 0 for key in keys)
            or value["tp"] + value["fp"] != value["predicted_count"]
            or value["tp"] + value["fn"] != value["gold_count"]):
        raise ValueError("Inconsistent nonnegative metric counts")
    tp, predicted, gold = value["tp"], value["predicted_count"], value["gold_count"]
    return {**{key: value[key] for key in keys},
            "precision": tp / predicted if predicted else 0.0 if gold else None,
            "recall": tp / gold if gold else 0.0 if predicted else None,
            "f1": 2 * tp / (predicted + gold) if predicted + gold else None}


def load_evaluation(path: Path) -> dict:
    evaluation = single.read_json(path)
    if not isinstance(evaluation, dict) or evaluation.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation schema")
    errors = []
    if evaluation.get("evaluation_profile") != "tongagents-react":
        errors.append("evaluation_not_produced_with_required_tongagents_react_profile")
    identity = evaluation.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Missing complete run identity")
    errors.extend(single.react_profile_errors(identity, {"workflow": identity.get("workflow")}))
    if single.digest(identity) != evaluation.get("run_fingerprint"):
        errors.append("run_identity_fingerprint_mismatch")
    if evaluation.get("global_errors") != []:
        errors.append("evaluation_has_global_errors_or_missing_error_list")
    records = single.index_rows(evaluation.get("records"), "id")
    ids = evaluation.get("sample_ids")
    if (not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids)
            or len(set(ids)) != len(ids) or set(ids) != set(records)
            or evaluation.get("reports_expected") != len(ids)):
        raise ValueError("Fixed evaluation denominator or sample identities inconsistent")
    if (not isinstance(identity.get("sources"), dict) or len(identity["sources"]) != len(ids)
            or any(not single.sha256_string(value) for value in identity["sources"].values())):
        errors.append("identity_source_hashes_missing_or_count_mismatch")
    else:
        bound_sources = {Path(path).stem: value for path, value in identity["sources"].items()}
        if len(bound_sources) != len(ids) or any(bound_sources.get(record.get("name")) != record.get("input_docx_sha256")
                                               for record in records.values()):
            errors.append("document_sources_do_not_match_run_identity")
    for key in ("fixed_selection_sha256", "scoring_manifest_sha256"):
        if not single.sha256_string(evaluation.get(key)):
            errors.append("missing_frozen_manifest_hash:" + key)
    for record in records.values():
        if record.get("status") not in {"completed", "failed", "missing"}:
            raise ValueError("Invalid per-document evaluation status")
        valid = record["status"] == "completed"
        if record["score"].get("prediction_valid") is not valid:
            raise ValueError("Prediction validity conflicts with completion status")
        if valid and record.get("errors") != []:
            raise ValueError("Completed report has audit errors")
        if not valid and any(section.get("predicted_count") != 0 for section in record["score"]["sections"].values()):
            raise ValueError("Failed/missing report cannot retain partial predictions")
        if any(not single.sha256_string(record.get(key)) for key in ("input_docx_sha256", "gold_sha256")):
            errors.append("missing_document_or_gold_hash:" + record["id"])
    metrics = single.aggregate_reports(record["score"] for record in records.values())
    completed = sum(record["status"] == "completed" for record in records.values())
    if (evaluation.get("reports_completed") != completed
            or evaluation.get("reports_failed_or_missing") != len(ids) - completed):
        errors.append("evaluation_completion_counts_inconsistent")
    wall = evaluation.get("batch_wall_seconds")
    timing_valid = (evaluation.get("timing_valid") is True
                    and single.finite_nonnegative(wall) and wall > 0)
    if not timing_valid:
        errors.append("final_batch_wall_time_unavailable_or_invalid")
    if evaluation.get("status") not in {"completed", "completed_with_failures"}:
        errors.append("evaluation_not_terminal")
    if evaluation.get("all_reports_audit_complete") is not (completed == len(ids) and timing_valid):
        errors.append("evaluation_audit_complete_claim_inconsistent")
    fields = evaluation.get("basic_field_metrics")
    if not isinstance(fields, dict):
        raise ValueError("Basic-field metrics missing")
    fields = {key: metric_counts(value) for key, value in fields.items()}
    # All nonempty basic facts appear in exactly one field projection.
    for key in ("tp", "fp", "fn", "predicted_count", "gold_count"):
        if sum(field[key] for field in fields.values()) != metrics["sections"]["basic"][key]:
            errors.append("basic_field_count_totals_inconsistent:" + key)
    preflight = identity.get("preflight_model_calls")
    catalog = identity.get("model_catalog")
    if (not isinstance(preflight, list) or any(not isinstance(call, dict) for call in preflight)
            or not isinstance(catalog, list) or not catalog
            or any(not isinstance(model, dict) or not isinstance(model.get("permission", []), list)
                   or any(not isinstance(item, dict) for item in model.get("permission", [])) for model in catalog)):
        raise ValueError("Preflight receipts/model catalog missing or malformed")
    if any(call.get("status") != "success" or call.get("http_status") != 200 for call in preflight):
        errors.append("preflight_http_evidence_not_successful")
    return {"evaluation": evaluation, "identity": identity, "records": records, "metrics": metrics,
            "fields": fields, "completed": completed, "expected": len(ids), "wall": wall,
            "errors": errors, "file_sha256": single.file_digest(path), "path": str(path)}


def metric_change(reference: dict, candidate: dict) -> dict:
    before, after = metric_counts(reference), metric_counts(candidate)
    # These section macros come from aggregate_reports(), recomputed from each
    # report's counts above. Do not lose them when selecting micro-count fields.
    for key in ("macro_report_f1", "macro_report_support"):
        if key in reference or key in candidate:
            before[key], after[key] = reference.get(key), candidate.get(key)
    return {"reference": before, "candidate": after,
            "delta": {key: after[key] - before[key] if before[key] is not None and after[key] is not None else None
                      for key in before}}


def scalar_change(reference: float | None, candidate: float | None) -> dict:
    return {"reference": reference, "candidate": candidate,
            "delta": candidate - reference if reference is not None and candidate is not None else None}


def compare(reference_path: Path, candidate_path: Path) -> dict:
    reference, candidate = load_evaluation(reference_path), load_evaluation(candidate_path)
    ref_eval, cand_eval = reference["evaluation"], candidate["evaluation"]
    left_identity, left_allowed = comparison_identity(reference["identity"])
    right_identity, right_allowed = comparison_identity(candidate["identity"])
    identity_differences = differences(left_identity, right_identity, "identity")
    binding_keys = ("fixed_selection_sha256", "scoring_manifest_sha256", "dataset_split", "sample_ids")
    dataset_differences = differences({key: ref_eval.get(key) for key in binding_keys},
                                      {key: cand_eval.get(key) for key in binding_keys}, "fixed_dataset")
    sample_keys = ("name", "input_docx_sha256", "gold_sha256")
    dataset_differences.extend(differences(
        {key: {field: row.get(field) for field in sample_keys} for key, row in reference["records"].items()},
        {key: {field: row.get(field) for field in sample_keys} for key, row in candidate["records"].items()},
        "document_bindings"))
    for sample_id in set(reference["records"]) & set(candidate["records"]):
        left, right = reference["records"][sample_id], candidate["records"][sample_id]
        dataset_differences.extend(differences(
            {key: value["gold_count"] for key, value in left["score"]["sections"].items()},
            {key: value["gold_count"] for key, value in right["score"]["sections"].items()},
            f"gold_denominator.{sample_id}"))
    empty = {key: 0 for key in ("tp", "fp", "fn", "predicted_count", "gold_count")}
    fields = {field: metric_change(reference["fields"].get(field, empty), candidate["fields"].get(field, empty))
              for field in sorted(set(reference["fields"]) | set(candidate["fields"]))}
    for field, changes in fields.items():
        if changes["reference"]["gold_count"] != changes["candidate"]["gold_count"]:
            dataset_differences.append({"path": "gold_basic_field_denominator." + field,
                                        "reference": changes["reference"]["gold_count"],
                                        "candidate": changes["candidate"]["gold_count"]})
    sections = {name: metric_change(reference["metrics"]["sections"][name], candidate["metrics"]["sections"][name])
                for name in reference["metrics"]["sections"]}
    overall = metric_change(reference["metrics"]["overall"], candidate["metrics"]["overall"])
    macro_section_f1 = scalar_change(reference["metrics"]["macro_section_f1"], candidate["metrics"]["macro_section_f1"])
    per_report = {}
    for sample_id in sorted(set(reference["records"]) & set(candidate["records"])):
        left, right = reference["records"][sample_id], candidate["records"][sample_id]
        # Recompute derived overall and macro values from section counts, not
        # the cached overall/macro fields in the original evaluation JSON.
        left_metrics = single.aggregate_reports([left["score"]])
        right_metrics = single.aggregate_reports([right["score"]])
        per_report[sample_id] = {
            "name": left["name"], "status": {"reference": left["status"], "candidate": right["status"]},
            "overall": metric_change(left_metrics["overall"], right_metrics["overall"]),
            "sections": {name: metric_change(left["score"]["sections"][name], right["score"]["sections"][name])
                         for name in left_metrics["sections"]},
            "macro_section_f1": scalar_change(left_metrics["macro_section_f1"], right_metrics["macro_section_f1"]),
        }
    errors = [{"side": side, "error": error} for side, run in (("reference", reference), ("candidate", candidate))
              for error in run["errors"]]
    comparable = not errors and not identity_differences and not dataset_differences
    metric_drops = []
    comparisons = [("overall", overall), *[("sections." + key, value) for key, value in sections.items()],
                   *[("basic_fields." + key, value) for key, value in fields.items()]]
    for sample_id, report in per_report.items():
        comparisons.append(("per_report." + sample_id + ".overall", report["overall"]))
        comparisons.extend(("per_report." + sample_id + ".sections." + name, change)
                           for name, change in report["sections"].items())
    for scope, change in comparisons:
        for metric in ("precision", "recall", "f1", "macro_report_f1"):
            delta = change["delta"].get(metric)
            if delta is not None and delta < 0:
                metric_drops.append({"metric": scope + "." + metric, "reference": change["reference"][metric],
                                     "candidate": change["candidate"][metric], "delta": delta})
    for scope, change in [("macro_section_f1", macro_section_f1),
                          *[("per_report." + key + ".macro_section_f1", report["macro_section_f1"])
                            for key, report in per_report.items()]]:
        if change["delta"] is not None and change["delta"] < 0:
            metric_drops.append({"metric": scope, **change})
    rates = {side: run["completed"] * 3600 / run["wall"] if comparable else None
             for side, run in (("reference", reference), ("candidate", candidate))}
    status_changes = [{"id": sample_id, "reference": reference["records"][sample_id]["status"],
                       "candidate": candidate["records"][sample_id]["status"]}
                      for sample_id in sorted(set(reference["records"]) & set(candidate["records"]))
                      if reference["records"][sample_id]["status"] != candidate["records"][sample_id]["status"]]
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "comparable" if comparable else "blocked_by_inconsistent_or_incomplete_evidence",
        "comparable": comparable, "errors": errors,
        "reference": {"path": reference["path"], "sha256": reference["file_sha256"],
                      "run_fingerprint": ref_eval["run_fingerprint"]},
        "candidate": {"path": candidate["path"], "sha256": candidate["file_sha256"],
                      "run_fingerprint": cand_eval["run_fingerprint"]},
        "comparison_identity_sha256": single.digest(left_identity) if not identity_differences else None,
        "identity_differences_not_allowed": identity_differences, "dataset_differences": dataset_differences,
        "allowed_identity_changes": differences(left_allowed, right_allowed),
        "identity_allowlist": ["document_workers and effective_document_workers",
                               "model_catalog[*].created and permission[*].id/created only",
                               "preflight_model_calls[*].call_id/started_at/seconds only",
                               "GET /v1/models or /model_info response/output SHA only when corresponding parsed body is retained in identity"],
        "completeness": {"reports_expected": {"reference": reference["expected"], "candidate": candidate["expected"]},
                         "completed": {"reference": reference["completed"], "candidate": candidate["completed"],
                                       "delta": candidate["completed"] - reference["completed"]},
                         "failed_or_missing": {"reference": reference["expected"] - reference["completed"],
                                               "candidate": candidate["expected"] - candidate["completed"]},
                         "status_changes": status_changes,
                         "both_audit_complete": ref_eval.get("all_reports_audit_complete") is True
                                                and cand_eval.get("all_reports_audit_complete") is True},
        "performance": {"scope": "structural/audit-complete reports per final whole-batch wall time",
                        "batch_wall_seconds": {"reference": reference["wall"], "candidate": candidate["wall"]},
                        "completed_reports_per_hour": rates,
                        "throughput_ratio_candidate_over_reference": rates["candidate"] / rates["reference"]
                            if comparable and rates["reference"] else None},
        "metrics": {"interpretation": "strict_training_label_proxy_not_official_score", "overall": overall,
                    "sections": sections, "basic_fields": fields, "macro_section_f1": macro_section_f1,
                    "per_report": per_report,
                    "regression_checks": "pooled overall/section/basic-field PRF, section macro_report_f1, macro_section_f1, and each report's overall/section PRF and macro_section_f1",
                    "raw_deltas_comparable": comparable,
                    "decreased_precision_recall_or_f1": metric_drops,
                    "any_defined_label_metric_decreased": bool(metric_drops) if comparable else None},
        "quality_verdict": "No semantic passing threshold is defined; audit completion or unchanged pooled metrics do not establish report-level quality nonregression.",
        "limitations": ["Offline recorded-identity comparison does not verify live model tensors.",
                        "Server/runtime options absent from recorded identities cannot be verified here.",
                        "Failures retain all gold false negatives; partial JSON is never scored as success.",
                        "No significance claim or long-document extrapolation follows from this small fixed batch.",
                        "When comparison is blocked, numerical deltas remain diagnostic and must not rank candidates."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("reference", "candidate", "out-json"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(args.reference, args.candidate)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    import json
    with args.out_json.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "completeness": result["completeness"],
                      "performance": result["performance"],
                      "any_defined_label_metric_decreased": result["metrics"]["any_defined_label_metric_decreased"],
                      "quality_verdict": result["quality_verdict"]},
                     ensure_ascii=False))
    return 0 if result["comparable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
