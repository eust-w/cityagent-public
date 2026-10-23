#!/usr/bin/env python3
"""Independent V2 protocol; score each frozen report once, including failures."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.backends import (
    AccountedBackend,
    APIBackend,
    ModelManifest,
    normalize_api_url,
)
from bridge_evidence.contest.candidate import validate_config
from bridge_evidence.contest.rsi_acceptance import comparison_identity
from bridge_evidence.contest.run_state import (
    atomic_json,
    bind_work_directory,
    build_run_identity,
    file_digest,
    json_digest,
)
from bridge_evidence.contest.timing import TimingLedger
from bridge_evidence.evaluation.generation import (
    METRIC_VERSION,
    aggregate_reports,
    score_report,
)

PROTOCOL = "complete-multitask-report-v2"


def load_cases(path):
    manifest = json.loads(path.read_text())
    cases = manifest.get("samples")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("purpose") != "development-regression"
        or not isinstance(cases, list)
        or not cases
    ):
        raise ValueError(
            "explicit nonempty V2 development-regression manifest required"
        )
    if manifest.get("metric_denominator", len(cases)) != len(cases):
        raise ValueError("frozen metric denominator differs from report count")
    ids = set()
    entities = {}
    for case in cases:
        if case["id"] in ids:
            raise ValueError("duplicate evaluation sample")
        ids.add(case["id"])
        if manifest.get("split_verification", {}).get("strict"):
            entity, split = case.get("bridge_entity"), case.get("split")
            if (
                not case.get("entity_verified")
                or not entity
                or not case.get("entity_evidence")
            ):
                raise ValueError("verified bridge identity required in strict manifest")
            if entities.setdefault(entity, split) != split:
                raise ValueError("bridge entity appears across splits")
        for key in ("report", "gold", "diagnostic_annotations"):
            if key == "diagnostic_annotations" and key not in case:
                continue
            if file_digest(Path(case[key])) != case[key + "_sha256"]:
                raise ValueError("frozen " + key + " changed")
    return manifest, cases


def evaluate(args):
    from bridge_evidence.contest.rsi_runtime import WorkLock

    with WorkLock(args.output_dir):
        return _evaluate_locked(args)


def _evaluate_locked(args):
    start = time.perf_counter()
    manifest, cases = load_cases(args.manifest)
    model = ModelManifest.load(args.model_manifest).validate()
    config = validate_config(json.loads(args.candidate_config.read_text()))
    from concurrent.futures import ThreadPoolExecutor

    from bridge_evidence.contest.rsi import StrategyBundle
    from bridge_evidence.contest.rsi_pipeline import pipeline, verify_parallel_sdk
    from bridge_evidence.contest.rsi_runtime import Dispatcher
    from bridge_evidence.contest.rsi_sandbox import PolicyController, verify_root

    cleanup = ThreadPoolExecutor(max_workers=1)
    if config["engine_parameters"]:
        raise ValueError(
            "API engine parameters require separate server deployment evidence"
        )
    bundle = StrategyBundle.load(args.rsi_bundle) if args.rsi_bundle else None
    if args.rsi_mode != "off" and (
        not bundle or not args.rsi_sandbox_root or bundle.config != config
    ):
        raise ValueError("RSI requires matching frozen bundle and sandbox")
    sandbox_digest = (
        verify_root(args.rsi_sandbox_root) if args.rsi_sandbox_root else None
    )
    if (
        args.report_workers > 1
        and args.engine == "tongagents"
        and not (
            args.sdk_parallel_acceptance
            and verify_parallel_sdk(args.sdk_parallel_acceptance, args.sdk_acceptance, args.url,
                                    workers=args.report_workers)
        )
    ):
        raise ValueError("real parallel SDK proof required")
    spec = importlib.util.spec_from_file_location("contest_runtime", ROOT / "run.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    args.workflow = config["workflow"]
    entry.validate_visual_options(args)
    model_identity = model.identity()
    visual_conditions = entry.configure_visual_runtime(args, model_identity)
    identity = build_run_identity(
        Path(model.base),
        ROOT,
        {
            "model": model_identity,
            "visual_evidence": visual_conditions,
            "visual_trial_nonce": uuid4().hex if getattr(args, "visual_evidence", "off") != "off" else None,
            "rsi_mode": args.rsi_mode,
            "rsi_bundle": bundle.as_dict() if bundle else None,
            "sandbox_digest": sandbox_digest,
            "report_workers": args.report_workers,
            "config": config,
            "manifest": manifest,
            "metric_version": METRIC_VERSION,
            "protocol": PROTOCOL,
            "evaluator_sha256": file_digest(Path(__file__)),
            "hardware": args.hardware,
            "warmup": args.warmup,
            "backend": args.backend,
            "url": args.url,
            "engine": args.engine,
            "sdk_acceptance_sha256": file_digest(args.sdk_acceptance)
            if args.sdk_acceptance
            else None,
        },
        model_files=model_identity["files"]["base"],
    )
    fingerprint = bind_work_directory(args.output_dir, identity, resume=False)
    args.visual_trial_id = fingerprint
    from bridge_evidence.contest.fast_tokenizer import load_tokenizer

    tokenizer = load_tokenizer(model.tokenizer)
    backend = APIBackend(
        args.url,
        model.served_model,
        tokenizer,
        kind=args.backend,
        workers=config["batch_size"],
        seed=config["seed"],
    )
    backend.probe()
    ledger = TimingLedger(args.output_dir / "timing.sqlite")
    generate = AccountedBackend(backend, ledger)
    ledger.recover()
    # A common warmup is measured separately and never used as a speed benchmark.
    warm_start = time.perf_counter()
    warmup_ledger = TimingLedger(args.output_dir / "warmup.sqlite")
    warmup_generate = AccountedBackend(backend, warmup_ledger, role="warmup")
    warmup_generations = []
    for _ in range(args.warmup):
        prompt = [tokenizer.eos_token_id]
        warmup_generations.append(
            {"prompt_ids": prompt, "response": warmup_generate(prompt, 1)}
        )
    atomic_json(
        args.output_dir / "warmup.json",
        {
            "protocol": "actual-warmup-v1",
            "seconds": time.perf_counter() - warm_start,
            "requests": args.warmup,
            "generations": warmup_generations,
            "timing_sha256": file_digest(args.output_dir / "warmup.sqlite"),
        },
    )
    raw_scores, final_scores, details = [], [], []
    batch_start = time.perf_counter()
    policies = {}
    report_timings = {}

    def observe_report_timing(index, timing):
        report_timings[index] = timing
        atomic_json(
            args.output_dir / "pipeline-timing" / f"{index:06d}.json",
            {"index": index, "id": cases[index]["id"],
             "run_fingerprint": fingerprint, **timing},
        )

    def prepare_case(case):
        from types import SimpleNamespace

        runtime = SimpleNamespace(
            work_dir=args.output_dir,
            workflow=config["workflow"],
            candidate=config,
            text_dir=None,
            backend=args.backend,
            visual_evidence=getattr(args, "visual_evidence", "off"),
            visual_display_transforms=getattr(args, "visual_display_transforms", False),
            visual_capability=getattr(args, "visual_capability", None),
            visual_capability_binding=args.visual_capability_binding,
            visual_model_identity_sha256=args.visual_model_identity_sha256,
            visual_trial_id=fingerprint,
        )
        return entry.prepare_document(runtime, Path(case["report"]), tokenizer)

    def run_case(index, prepared):
        case = cases[index]
        path = Path(case["report"])
        runtime = entry.build_parser().parse_args(
            [
                "--input-dir",
                str(path.parent),
                "--result-dir",
                str(args.output_dir / "result"),
                "--work-dir",
                str(args.output_dir),
                "--engine",
                args.engine,
                "--workflow",
                config["workflow"],
                "--max-attempts",
                "2",
                "--label-conventions",
                getattr(args, "label_conventions", "auto"),
                *(["--max-model-len", str(args.max_model_len)] if getattr(args, "max_model_len", None) else []),
            ]
        )
        runtime.candidate = config
        runtime.backend = args.backend
        runtime.visual_evidence = getattr(args, "visual_evidence", "off")
        runtime.visual_display_transforms = getattr(args, "visual_display_transforms", False)
        runtime.visual_capability = getattr(args, "visual_capability", None)
        runtime.visual_capability_binding = args.visual_capability_binding
        runtime.visual_model_identity_sha256 = args.visual_model_identity_sha256
        runtime.visual_trial_id = fingerprint
        runtime.model_identity = model_identity
        runtime.sdk_acceptance = args.sdk_acceptance
        runtime.result_dir.mkdir(exist_ok=True)
        runtime.rsi_mode = args.rsi_mode
        policy = None
        if args.rsi_mode != "off":
            policy = PolicyController(bundle, args.rsi_sandbox_root)
            policies[index] = policy
        return entry.process_document(
            runtime,
            path,
            tokenizer,
            dispatcher,
            fingerprint=fingerprint,
            prepared=prepared,
            policy=policy,
        )

    with Dispatcher(generate, capacity=config["batch_size"]) as dispatcher:
        dispatcher.begin_foreground()
        for index, record in pipeline(
            cases, prepare_case, run_case, workers=args.report_workers,
            timing_observer=observe_report_timing,
        ):
            case = cases[index]
            # Only the scorer receives labels; strategy inputs come from execution state.
            gold = json.loads(Path(case["gold"]).read_text())
            scored = score_execution_record(gold, record, workflow=config["workflow"])
            from bridge_evidence.contest.semantic_diagnostics import diagnose_record

            annotations = (
                json.loads(Path(case["diagnostic_annotations"]).read_text())
                if case.get("diagnostic_annotations")
                else None
            )
            diagnostic_record = (
                record if isinstance(record, dict) else getattr(record, "record", {})
            )
            diagnostics = diagnose_record(diagnostic_record, annotations=annotations)
            raw_scores.append(scored["raw_score"])
            final_scores.append(scored["final_score"])
            details.append(
                {
                    "id": case["id"],
                    "status": scored["status"],
                    "error": scored["error"],
                    "wall_seconds": scored["processing_wall_seconds"],
                    "processing_wall_seconds": scored["processing_wall_seconds"],
                    "end_to_end_wall_seconds": report_timings[index]["end_to_end_wall_seconds"],
                    "pipeline_timing": report_timings.pop(index),
                    "raw_score": raw_scores[-1],
                    "final_score": final_scores[-1],
                    "semantic_diagnostics": diagnostics,
                    "bundle_sha256": bundle.digest
                    if bundle and args.rsi_mode != "off"
                    else None,
                }
            )
            atomic_json(
                args.output_dir / "progress.json",
                {"completed": len(details), "denominator": len(cases)},
            )
            if index in policies:
                cleanup.submit(policies.pop(index).close)
        dispatcher.end_foreground()
    cleanup.shutdown(wait=True)
    for policy in policies.values():
        policy.close()
    ledger.span("batch-wall", "batch_wall", time.perf_counter() - start)
    aggregate = aggregate_reports(final_scores)
    conditions = comparison_identity(identity, source_root=ROOT)
    result = {
        "run_id": uuid4().hex,
        "protocol": "frozen-rsi-v3" if args.rsi_mode == "frozen" else PROTOCOL,
        "comparison_identity": conditions,
        "protected_sha256": json_digest(identity["code_files"]),
        "bundle_sha256": bundle.digest if bundle and args.rsi_mode != "off" else None,
        "latency": latency_quantiles(
            [d["wall_seconds"] for d in details if d["wall_seconds"] is not None]
        ),
        # Keep the legacy latency field bound to existing process_document
        # receipts. The new metric includes preparation/queue time and failures;
        # it is an experiment observation, not new RSI acceptance evidence.
        "latency_scope": "process_document_only_excludes_preparation_and_queue",
        **end_to_end_latency_summary(details, denominator=len(cases)),
        "completed": len(details),
        "cache_reused": False,
        "metric_version": METRIC_VERSION,
        "purpose": "development-regression",
        "generalization_claim": False,
        "diagnostic_protocol": "semantic-evidence-diagnostics-v1",
        "denominator": len(cases),
        "config": config,
        "conditions": conditions,
        "f1": [aggregate["sections"][s]["f1"] for s in ("basic", "advice", "defects")],
        "wall_seconds": time.perf_counter() - start,
        "postwarm_batch_wall_seconds": time.perf_counter() - batch_start,
        "total_wall_seconds": time.perf_counter() - start,
        "failures": aggregate["reports_failed"],
        "raw_metrics": aggregate_reports(raw_scores),
        "final_metrics": aggregate,
        "timing": ledger.summary(),
        "reports": details,
        "repeated": False,
        "run_fingerprint": fingerprint,
    }
    atomic_json(args.output_dir / "evaluation.json", result)
    atomic_json(
        args.output_dir / "result" / "infer_time.json",
        {"infer_time": ledger.summary()["generation_seconds"]},
    )
    return result


def score_execution_record(gold, record, *, workflow):
    """Score the production parser's output contract; invalid records keep all FN.

    Whole-report inference accepts a complete fenced JSON document and its
    defined trailing-comma normalization. Chunked inference serializes merged
    data as plain JSON, so it retains that separate contract. Neither path
    extracts substrings, recovers truncated JSON, or replaces missing raw output
    with final data. Gold validation stays outside the per-record error handler.
    """
    failed_score = score_report(gold, None)
    details = {
        "status": "failed", "error": None, "processing_wall_seconds": None,
        "raw_score": failed_score, "final_score": failed_score,
    }
    if isinstance(record, Exception):
        details["error"] = str(record)
        failure_record = getattr(record, "record", {})
        if isinstance(failure_record, dict):
            details["processing_wall_seconds"] = failure_record.get("report_wall_seconds")
        return details
    if not isinstance(record, dict):
        details["error"] = "evaluation_record_invalid:execution_record_must_be_object"
        return details
    details["processing_wall_seconds"] = record.get("report_wall_seconds")
    try:
        raw_output = record["raw_output"]
        if not isinstance(raw_output, str):
            raise ValueError("raw_output_must_be_string")
        if workflow == "whole":
            from bridge_evidence.contest.inference_core import _parse_report

            raw = _parse_report(raw_output)
        else:
            raw = json.loads(raw_output)
        raw_score = score_report(gold, raw)
        final_score = score_report(gold, record["final_data"])
        if not raw_score["prediction_valid"] or not final_score["prediction_valid"]:
            raise ValueError("missing_raw_or_final_prediction")
    except (KeyError, TypeError, ValueError, RecursionError) as error:
        details["error"] = "evaluation_record_invalid:" + type(error).__name__ + ":" + str(error)
        return details
    details.update(status="success", raw_score=raw_score, final_score=final_score)
    return details


def latency_quantiles(values):
    values = sorted(values)

    def percentile(p):
        if not values:
            return None
        position = (len(values) - 1) * p
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {"p50": percentile(0.5), "p95": percentile(0.95)}


def end_to_end_latency_summary(reports, *, denominator):
    """Reject omitted timings instead of silently dropping failed reports."""
    values = [report.get("end_to_end_wall_seconds") for report in reports]
    if len(values) != denominator or any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in values
    ):
        raise ValueError("end-to-end latency requires one finite timing per report")
    return {
        "end_to_end_latency_protocol": "preparation-submission-to-report-terminal-v1",
        "end_to_end_latency_seconds": latency_quantiles(values),
        "end_to_end_latency_count": len(values),
        "end_to_end_latency_denominator": denominator,
    }


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual-evidence", choices=("off", "same-model"), default="off")
    ap.add_argument("--visual-capability", type=Path)
    ap.add_argument("--visual-display-transforms", action="store_true",
                    help="Explicit same-model source-pixel crop/quarter-turn/flip views")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-manifest", type=Path, required=True)
    ap.add_argument("--candidate-config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--rsi-mode", choices=("off", "frozen"), default="off")
    ap.add_argument("--rsi-bundle", type=Path)
    ap.add_argument("--rsi-sandbox-root", type=Path)
    ap.add_argument("--report-workers", type=int, choices=(1, 2, 4), default=1)
    ap.add_argument("--max-model-len", type=int, help="whole-workflow context; run.py default if omitted")
    ap.add_argument("--label-conventions", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--sdk-parallel-acceptance", type=Path)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--url", type=normalize_api_url, default="http://127.0.0.1:8000/v1")
    ap.add_argument("--backend", choices=("vllm-api", "sglang-api"), default="vllm-api")
    ap.add_argument("--engine", choices=("direct", "tongagents"), default="direct")
    ap.add_argument("--sdk-acceptance", type=Path)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.warmup < 0:
        ap.error("warmup must be nonnegative")
    print(
        json.dumps(
            {
                k: v
                for k, v in evaluate(args).items()
                if k not in ("reports", "raw_metrics", "final_metrics")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
