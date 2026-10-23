#!/usr/bin/env python3
"""Official-cloud-only parameter search; candidates require separate RSI release acceptance."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.candidate import select_candidate
from bridge_evidence.contest.meta_optimization import (
    CloudBudget,
    candidate_configs,
    diagnose_baseline,
)
from bridge_evidence.contest.run_state import atomic_json, file_digest


def write_search_selection(
    directory,
    baseline,
    winner,
    *,
    cloud_remaining_seconds,
    model_manifest_sha256,
    evaluation_manifest_sha256,
):
    """Publish search evidence, while keeping the formal configuration on baseline.

    This legacy search does not run the complete current RSI acceptance protocol.
    Even two passing screening runs cannot certify latency, recovery, or receipts.
    """
    directory = Path(directory)
    candidate = winner is not baseline
    atomic_json(directory / "frozen-config.json", baseline["config"])
    if candidate:
        atomic_json(directory / "provisional-config.json", winner["config"])
    atomic_json(
        directory / "selection.json",
        {
            "selected": "baseline",
            "evidence": baseline,
            "search_winner": "candidate" if candidate else "baseline",
            "search_winner_evidence": winner,
            "provisional_config": "provisional-config.json" if candidate else None,
            "release_qualified": False,
            "missing_release_gates": [
                "independent_execution_receipts",
                "latency_p50_p95",
                "fault_recovery_start_finish",
                "source_coverage_conflicts",
                "current_executor_and_backend_identity",
                "formal_rsi_release_verification",
            ],
            "cloud_remaining_seconds": cloud_remaining_seconds,
            "model_manifest_sha256": model_manifest_sha256,
            "evaluation_manifest_sha256": evaluation_manifest_sha256,
            "scope": "development-regression screening; formal inference retains baseline",
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-manifest", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--initial-candidate", type=Path)
    ap.add_argument("--retest-limit-seconds", type=int, default=18000)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--budget-ledger", type=Path, required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--engine", choices=("direct", "tongagents"), default="direct")
    ap.add_argument("--sdk-acceptance", type=Path)
    ap.add_argument("--errors", type=Path)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--trial-limit-seconds", type=int, default=3600)
    ap.add_argument("--supported-parameters", nargs="*", default=[])
    args = ap.parse_args()
    if not str(args.output_dir.resolve()).startswith("/workspace/") or not str(
        args.budget_ledger.resolve()
    ).startswith("/workspace/"):
        ap.error("meta optimization must run inside official /workspace")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(args.baseline.read_text())
    budget = CloudBudget(args.budget_ledger)
    errors = (
        json.loads(args.errors.read_text())
        if args.errors
        else diagnose_baseline(baseline)
    )
    atomic_json(args.output_dir / "error-analysis.json", errors)
    candidates = candidate_configs(
        errors, args.seed, args.max_candidates, args.supported_parameters
    )
    # Server-owned flags cannot be changed by the API protocol. They require an explicit server deployment experiment.
    candidates = [c for c in candidates if not c["engine_parameters"]]
    histories = []
    if args.initial_candidate:
        initial = json.loads(args.initial_candidate.read_text())
        if any(
            initial.get(key) != baseline.get(key)
            for key in ("conditions", "denominator", "protocol", "metric_version")
        ):
            ap.error(
                "initial candidate must use identical baseline evaluation conditions"
            )
        from bridge_evidence.contest.candidate import validate_config

        validate_config(initial["config"])
        histories.append(initial)
        candidates = [c for c in candidates if c != initial["config"]][
            : args.max_candidates - 1
        ]
    if args.trial_limit_seconds <= 30 or args.retest_limit_seconds <= 30:
        ap.error("positive trial and retest limits above shutdown margin required")
    frozen_model = file_digest(args.model_manifest)
    frozen_data = file_digest(args.manifest)

    def trial(config, index):
        if (
            file_digest(args.manifest) != frozen_data
            or file_digest(args.model_manifest) != frozen_model
        ):
            raise ValueError("experiment identity changed")
        path = args.output_dir / f"trial-{index}"
        config_path = args.output_dir / f"candidate-{index}.json"
        atomic_json(config_path, config)
        cap = min(
            args.retest_limit_seconds
            if index == "retest"
            else args.trial_limit_seconds,
            int(budget.remaining()) - (1830 if index == "retest" else 30),
        )
        if cap < 1:
            raise RuntimeError("cloud budget exhausted")
        job = budget.start("candidate" if index != "retest" else "final-retest", cap)
        start = time.perf_counter()
        status = "failed"
        try:
            with (args.output_dir / f"trial-{index}.log").open("x") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "scripts/evaluate_multitask.py"),
                        "--manifest",
                        str(args.manifest),
                        "--model-manifest",
                        str(args.model_manifest),
                        "--candidate-config",
                        str(config_path),
                        "--output-dir",
                        str(path),
                        "--hardware",
                        args.hardware,
                        "--engine",
                        args.engine,
                        *(
                            ["--sdk-acceptance", str(args.sdk_acceptance)]
                            if args.sdk_acceptance
                            else []
                        ),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = process.wait(timeout=cap)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
                if code:
                    raise subprocess.CalledProcessError(code, process.args)
            result = json.loads((path / "evaluation.json").read_text())
            status = "complete"
            return result
        finally:
            budget.finish(job, time.perf_counter() - start, status)

    atomic_json(args.output_dir / "history.json", histories)
    for i, config in enumerate(candidates):
        if budget.remaining() < args.trial_limit_seconds:
            break
        try:
            result = trial(config, i)
            histories.append(result)
        except Exception as error:
            histories.append({"config": config, "error": str(error)})
        atomic_json(args.output_dir / "history.json", histories)
    provisional = [{**r, "repeated": True} for r in histories if "f1" in r]
    winner = select_candidate(baseline, provisional)
    if winner is not baseline and budget.remaining() > 1860:
        try:
            repeated = trial(winner["config"], "retest")
        except Exception as error:
            repeated = {"error": str(error)}
        repeated["repeated"] = True
        histories.append({**repeated, "search_stage": "independent-search-retest"})
        atomic_json(args.output_dir / "history.json", histories)
        # Both measured runs must pass; never select from a lucky first run.
        if "f1" in repeated and select_candidate(baseline, [repeated]) is not baseline:
            winner = repeated
        else:
            winner = baseline
    else:
        winner = baseline
    write_search_selection(
        args.output_dir,
        baseline,
        winner,
        cloud_remaining_seconds=budget.remaining(),
        model_manifest_sha256=frozen_model,
        evaluation_manifest_sha256=frozen_data,
    )


if __name__ == "__main__":
    main()
