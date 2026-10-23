#!/usr/bin/env python3
"""Bounded inference-only recursive search on the official cloud, sharing the old budget."""

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
from bridge_evidence.contest.backends import normalize_api_url
from bridge_evidence.contest.candidate import validate_config
from bridge_evidence.contest.meta_optimization import CloudBudget
from bridge_evidence.contest.rsi import StrategyBundle
from bridge_evidence.contest.rsi_acceptance import (
    audit_independent_receipts,
    audited_optimizer_trial,
    load_receipt,
    replacement_errors,
)
from bridge_evidence.contest.rsi_evolution import EvolutionStore
from bridge_evidence.contest.rsi_experiments import search_history
from bridge_evidence.contest.rsi_sandbox import sandbox_call, verify_root
from bridge_evidence.contest.run_state import atomic_json, file_digest, json_digest


def run_job(budget, phase, cap, command, log):
    cap = min(cap, int(budget.remaining()) - 30)
    if cap < 30:
        raise RuntimeError("insufficient shared experiment budget")
    job = budget.start(phase, cap)
    started, status = time.perf_counter(), "failed"
    try:
        with Path(log).open("x") as output:
            process = subprocess.Popen(
                command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=cap)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            status = "complete" if code == 0 else "failed"
            if code:
                raise RuntimeError("experiment process failed: " + phase)
    finally:
        budget.finish(job, time.perf_counter() - started, status)


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "model-manifest",
        "baseline-config",
        "sandbox-root",
        "budget-ledger",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--url", type=normalize_api_url, default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--engine", choices=("direct", "tongagents"), default="tongagents"
    )
    parser.add_argument("--sdk-acceptance", type=Path)
    parser.add_argument("--sdk-parallel-acceptance", type=Path)
    parser.add_argument("--legacy-history", type=Path)
    parser.add_argument("--optimizer-parent-bundle", type=Path)
    parser.add_argument("--optimizer-child-bundle", type=Path)
    parser.add_argument("--trial-limit-seconds", type=int, default=14400)
    parser.add_argument("--retest-limit-seconds", type=int, default=18000)
    args = parser.parse_args()
    if any(
        not str(p.resolve()).startswith("/workspace/")
        for p in (args.output_dir, args.budget_ledger, args.sandbox_root)
    ):
        parser.error("official /workspace paths required")
    config = validate_config(json.loads(args.baseline_config.read_text()))
    if config["workflow"] == "whole":
        parser.error("whole baseline cannot start RSI search; supply an explicit chunked or bounded-react baseline")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    verify_root(args.sandbox_root)
    budget = CloudBudget(args.budget_ledger)
    model_sha, data_sha = file_digest(args.model_manifest), file_digest(args.manifest)
    # Evaluation and recovery must use the exact same explicit baseline workflow.
    actor_config = config
    parent = StrategyBundle.baseline(actor_config)
    atomic_json(args.output_dir / "initial-bundle.json", parent.as_dict())
    atomic_json(args.output_dir / "anchor-config.json", config)
    legacy = (
        json.loads(args.legacy_history.read_text())
        if args.legacy_history and args.legacy_history.is_file()
        else []
    )
    legacy_used = len({json_digest(r["config"]) for r in legacy if "config" in r})
    if legacy_used >= 8:
        atomic_json(
            args.output_dir / "selection.json",
            {
                "selected": "baseline",
                "reason": "eight-candidate campaign cap already exhausted",
            },
        )
        return
    common = [
        "--model-manifest",
        str(args.model_manifest),
        "--hardware",
        args.hardware,
        "--url",
        args.url,
        "--engine",
        args.engine,
        "--rsi-sandbox-root",
        str(args.sandbox_root),
    ]
    if args.sdk_acceptance:
        common += ["--sdk-acceptance", str(args.sdk_acceptance)]
    if args.sdk_parallel_acceptance:
        common += [
            "--sdk-parallel-acceptance",
            str(args.sdk_parallel_acceptance),
            "--report-workers",
            "2",
        ]

    def check_identity():
        if (
            file_digest(args.model_manifest) != model_sha
            or file_digest(args.manifest) != data_sha
        ):
            raise ValueError("fixed model/data changed")

    def evaluation(bundle, name, mode="frozen", cap=None, *, audited=False):
        check_identity()
        cfg = args.output_dir / (name + ".config.json")
        bp = args.output_dir / (name + ".bundle.json")
        atomic_json(cfg, config if mode == "off" else bundle.config)
        atomic_json(bp, bundle.as_dict())
        output = args.output_dir / name
        command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_multitask.py"),
            "--manifest",
            str(args.manifest),
            "--candidate-config",
            str(cfg),
            "--rsi-mode",
            mode,
            "--output-dir",
            str(output),
            *common,
        ]
        if mode != "off":
            command += ["--rsi-bundle", str(bp)]
        run_job(
            budget,
            "rsi-" + name,
            cap or args.trial_limit_seconds,
            command,
            args.output_dir / (name + ".log"),
        )
        if audited:
            return audited_optimizer_trial(output, source_root=ROOT)
        return json.loads((output / "evaluation.json").read_text())

    def recovery(bundle, name, mode="frozen"):
        bp = args.output_dir / (name + ".recovery-bundle.json")
        atomic_json(bp, bundle.as_dict())
        output = args.output_dir / (name + "-recovery")
        run_job(
            budget,
            "rsi-recovery",
            1200,
            [
                sys.executable,
                str(ROOT / "scripts/benchmark_rsi_recovery.py"),
                "--model-manifest",
                str(args.model_manifest),
                "--bundle",
                str(bp),
                "--sandbox-root",
                str(args.sandbox_root),
                "--output-dir",
                str(output),
                "--mode",
                mode,
                "--url",
                args.url,
            ],
            args.output_dir / (name + "-recovery.log"),
        )
        return output / "recovery.json"

    selected = None
    report = {
        "selected": "baseline",
        "candidate_verified": False,
        "optimizer_verified": False,
        "generalization_claim": False,
        "official_score": None,
        "model_manifest_sha256": model_sha,
        "manifest_sha256": data_sha,
        "training": False,
    }
    atomic_json(args.output_dir / "selection.json", report)
    try:
        # Do not spend search budget that is needed by independent full retest.
        reserve = args.retest_limit_seconds + 2400
        if budget.remaining() < args.trial_limit_seconds + reserve:
            raise RuntimeError(
                "insufficient budget for anchor and independent acceptance"
            )
        evaluation(parent, "anchor", mode="off")
        anchor_recovery = recovery(parent, "anchor", mode="off")
        anchor = load_receipt(
            args.output_dir / "anchor", anchor_recovery, source_root=ROOT
        )
        with EvolutionStore(
            args.output_dir / "evolution.sqlite", model_sha, data_sha
        ) as store:
            store.add(parent, 0)
            if args.optimizer_parent_bundle or args.optimizer_child_bundle:
                if not args.optimizer_parent_bundle or not args.optimizer_child_bundle:
                    raise ValueError("both optimizer arms required")
                if (
                    budget.remaining() >= 4 * args.trial_limit_seconds + reserve
                    and legacy_used + 5 <= 8
                ):
                    from bridge_evidence.contest.rsi_search_comparison import (
                        compare_optimizers,
                    )

                    comparison = compare_optimizers(
                        StrategyBundle.load(args.optimizer_parent_bundle),
                        StrategyBundle.load(args.optimizer_child_bundle),
                        parent,
                        anchor,
                        args.sandbox_root,
                        lambda bundle, name, **kwargs: evaluation(
                            bundle, name, audited=True, **kwargs
                        ),
                        lambda candidate: store.add(candidate, 1),
                        trial_cap=args.trial_limit_seconds,
                        budget=budget,
                    )
                    atomic_json(
                        args.output_dir / "optimizer-comparison.json", comparison
                    )
                    report["optimizer_verified"] = comparison["optimizer_verified"]
                else:
                    report["optimizer_comparison"] = "insufficient-budget"
            search_state = {"config": actor_config, "generation": 0, "seed": 17}
            used = legacy_used + len(store.history())
            completed_outcomes = []
            recent_records = args.output_dir / "anchor" / "records"
            for generation in range(1, 4):
                if used >= 8 or budget.remaining() < args.trial_limit_seconds + reserve:
                    break
                # Same fixed inference model proposes code. Failure leaves the previous strategy intact.
                target = ("choose_action", "propose_candidates", "update_search")[
                    generation - 1
                ]
                parent_path = args.output_dir / f"g{generation}-parent.bundle.json"
                atomic_json(parent_path, parent.as_dict())
                code_dir = args.output_dir / f"g{generation}-code"
                feedback_path = args.output_dir / f"g{generation}-search-feedback.json"
                atomic_json(feedback_path, search_history(completed_outcomes))
                try:
                    run_job(
                        budget,
                        "rsi-code-proposal",
                        900,
                        [
                            sys.executable,
                            str(ROOT / "scripts/propose_rsi_strategy.py"),
                            "--parent-bundle",
                            str(parent_path),
                            "--model-manifest",
                            str(args.model_manifest),
                            "--sandbox-root",
                            str(args.sandbox_root),
                            "--records-dir",
                            str(recent_records),
                            "--search-history",
                            str(feedback_path),
                            "--reports-manifest",
                            str(args.manifest),
                            "--output-dir",
                            str(code_dir),
                            "--target",
                            target,
                            "--url",
                            args.url,
                        ],
                        args.output_dir / f"g{generation}-code.log",
                    )
                    code_candidate = StrategyBundle.load(
                        code_dir / "candidate.bundle.json"
                    )
                except Exception as error:
                    store.event(
                        "code-proposal-rejected",
                        {"generation": generation, "error": type(error).__name__},
                    )
                    code_candidate = parent
                proposals = sandbox_call(
                    args.sandbox_root,
                    code_candidate,
                    "propose_candidates",
                    [search_history(completed_outcomes), search_state],
                    budget=budget,
                )
                if not isinstance(proposals, list) or len(proposals) > 8:
                    raise ValueError("invalid bounded search output")
                outcomes = []
                qualified_outcomes = []
                if code_candidate.digest != parent.digest:
                    proposals = [parent.config, *proposals]
                for proposed in proposals:
                    if (
                        used >= 8
                        or budget.remaining() < args.trial_limit_seconds + reserve
                    ):
                        break
                    candidate_config = validate_config(proposed)
                    if (
                        candidate_config["workflow"] == "whole"
                        or candidate_config["engine_parameters"]
                    ):
                        continue
                    candidate = StrategyBundle(
                        code_candidate.source, candidate_config, parent.digest
                    )
                    if not store.add(candidate, generation):
                        continue
                    used += 1
                    name = f"g{generation}-candidate-{used}"
                    trial_started = time.perf_counter()
                    try:
                        result = evaluation(candidate, name)
                        result["status"] = "complete"
                        recent_records = args.output_dir / name / "records"
                        qualified = (
                            all(x >= y for x, y in zip(result["f1"], anchor["f1"]))
                            and result["failures"] <= anchor["failures"]
                        )
                        if qualified:
                            qualified_outcomes.append(result)
                        if (
                            qualified
                            and result["wall_seconds"] <= 0.8 * anchor["wall_seconds"]
                        ):
                            rc = recovery(candidate, name)
                            first = load_receipt(
                                args.output_dir / name, rc, source_root=ROOT
                            )
                            evaluation(
                                candidate,
                                name + "-retest",
                                cap=args.retest_limit_seconds,
                            )
                            repeated_rc = recovery(candidate, name + "-retest")
                            second = load_receipt(
                                args.output_dir / (name + "-retest"),
                                repeated_rc,
                                source_root=ROOT,
                            )
                            errors = replacement_errors(anchor, [first, second])
                            if not errors:
                                audit_independent_receipts([anchor, first, second])
                                selected = candidate
                                store.outcome(candidate.digest, result)
                                atomic_json(
                                    args.output_dir / "release.json",
                                    {
                                        "schema_version": 1,
                                        "bundle_sha256": candidate.digest,
                                        "comparison_identity": anchor[
                                            "comparison_identity"
                                        ],
                                        "protected_sha256": anchor["protected_sha256"],
                                        "model_manifest_sha256": model_sha,
                                        "anchor": str(
                                            args.output_dir
                                            / "anchor"
                                            / "evaluation.json"
                                        ),
                                        "runs": [
                                            str(
                                                args.output_dir
                                                / name
                                                / "evaluation.json"
                                            ),
                                            str(
                                                args.output_dir
                                                / (name + "-retest")
                                                / "evaluation.json"
                                            ),
                                        ],
                                        "receipts_sha256": [
                                            first["evaluation_sha256"],
                                            second["evaluation_sha256"],
                                        ],
                                        "anchor_sha256": anchor["evaluation_sha256"],
                                        "metrics": [first, second],
                                        "baseline": anchor,
                                    },
                                )
                                report.update(
                                    selected="candidate",
                                    candidate_verified=True,
                                    bundle_sha256=candidate.digest,
                                    evidence=[
                                        first["evaluation_sha256"],
                                        second["evaluation_sha256"],
                                    ],
                                    anchor_sha256=anchor["evaluation_sha256"],
                                )
                                break
                    except Exception as error:
                        result = {
                            "status": "failed",
                            "error": type(error).__name__,
                            "wall_seconds": time.perf_counter() - trial_started,
                            "config": candidate.config,
                        }
                    store.outcome(candidate.digest, result)
                    outcomes.append(result)
                completed_outcomes.extend(outcomes)
                atomic_json(args.output_dir / "history.json", store.history())
                if selected:
                    break
                if not outcomes:
                    break
                search_state = sandbox_call(
                    args.sandbox_root,
                    code_candidate,
                    "update_search",
                    [search_state, search_history(outcomes)],
                    budget=budget,
                )
                if (
                    not isinstance(search_state, dict)
                    or set(search_state) - {"config", "generation", "seed"}
                    or search_state.get("seed") != 17
                ):
                    raise ValueError("invalid search-state contract")
                validate_config(search_state["config"])
                if not qualified_outcomes:
                    # Failed/slow trials still inform the next proposal; keep the qualified actor.
                    continue
                best = min(qualified_outcomes, key=lambda r: r["wall_seconds"])
                parent = next(
                    StrategyBundle.load(
                        args.output_dir / (f"g{generation}-candidate-{i}.bundle.json")
                    )
                    for i in range(legacy_used + 2, used + 1)
                    if (
                        args.output_dir / f"g{generation}-candidate-{i}.bundle.json"
                    ).is_file()
                    and json.loads(
                        (
                            args.output_dir / f"g{generation}-candidate-{i}.config.json"
                        ).read_text()
                    )
                    == best["config"]
                )
                # Search code modifications have their own evidence; this default run does not claim A/B verification.
                # Only compare_optimizers can grant optimizer_verified; code generation itself cannot.
    except Exception as error:
        report["reason"] = type(error).__name__ + ": " + str(error)
    if selected:
        atomic_json(args.output_dir / "frozen-rsi-bundle.json", selected.as_dict())
    report["cloud_remaining_seconds"] = budget.remaining()
    atomic_json(args.output_dir / "selection.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
