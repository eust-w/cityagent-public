#!/usr/bin/env python3
"""Three fresh full-denominator diagnostics; this command never releases RSI.

The unchanged evaluator owns scoring and strong evidence checks. A capped arm
retains its records and call ledger, but receives no synthesized metric. Run only
against the original shared cloud ledger; existing jobs are never recovered here.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_rsi_evolution import run_job

from bridge_evidence.contest.backends import ModelManifest
from bridge_evidence.contest.candidate import validate_config
from bridge_evidence.contest.meta_optimization import CloudBudget
from bridge_evidence.contest.rsi import StrategyBundle
from bridge_evidence.contest.rsi_pipeline import verify_parallel_sdk
from bridge_evidence.contest.rsi_sandbox import verify_root
from bridge_evidence.contest.run_state import atomic_json, file_digest, json_digest
from bridge_evidence.contest.sdk_acceptance import validate_acceptance
from bridge_evidence.contest.vision_capability import validate_vision_capability

ARMS = (
    ("baseline", "off"),
    ("candidate-first", "frozen"),
    ("candidate-retest", "frozen"),
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "manifest",
        "model-manifest",
        "baseline-config",
        "candidate-bundle",
        "sdk-acceptance",
        "sandbox-root",
        "budget-ledger",
        "output-dir",
    ):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--hardware", required=True)
    result.add_argument(
        "--visual-evidence", choices=("off", "same-model"), default="off"
    )
    result.add_argument("--visual-capability", type=Path)
    result.add_argument("--sdk-parallel-acceptance", type=Path)
    result.add_argument("--report-workers", type=int, choices=(1, 2), default=1)
    result.add_argument("--expected-denominator", type=int, default=79)
    result.add_argument("--arm-limit-seconds", type=int, default=21600)
    result.add_argument("--reserve-seconds", type=int, default=180)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument(
        "--backend", choices=("vllm-api", "sglang-api"), default="vllm-api"
    )
    result.add_argument("--url", default="http://127.0.0.1:8000/v1")
    return result


def workspace_path(path):
    return Path("/workspace") in path.resolve().parents


def idle_ledger(path):
    # mode=ro prevents accidentally creating a new 24-hour budget.
    if not path.is_file():
        raise ValueError("the original existing cloud budget ledger is required")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        if db.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0]:
            raise RuntimeError(
                "existing cloud job active; do not recover or preempt it"
            )


def source_digest(root):
    paths = [root / "run.py"]
    for folder in ("src", "scripts"):
        paths.extend(sorted((root / folder).rglob("*.py")))
    return json_digest({str(p.relative_to(root)): file_digest(p) for p in paths})


def input_identity(args):
    paths = {
        "manifest": args.manifest,
        "model_manifest": args.model_manifest,
        "baseline_config": args.baseline_config,
        "candidate_bundle": args.candidate_bundle,
        "sdk_acceptance": args.sdk_acceptance,
    }
    if args.visual_capability:
        paths["visual_capability"] = args.visual_capability
    if args.sdk_parallel_acceptance:
        paths["sdk_parallel_acceptance"] = args.sdk_parallel_acceptance
    return {
        **{key: file_digest(path) for key, path in paths.items()},
        "source": source_digest(args.source_root),
    }


def validate(args):
    if args.source_root.resolve() != ROOT:
        raise ValueError("source-root must identify this immutable evaluator checkout")
    if not all(
        workspace_path(p)
        for p in (args.output_dir, args.budget_ledger, args.sandbox_root)
    ):
        raise ValueError(
            "official /workspace output, sandbox and original budget required"
        )
    if args.output_dir.exists():
        raise FileExistsError("fresh output directory required; cache cannot be reused")
    if (
        args.expected_denominator < 1
        or args.arm_limit_seconds < 60
        or args.reserve_seconds < 90
        or args.warmup < 1
    ):
        raise ValueError(
            "positive denominator, >=60s arm, >=90s reserve and real warmup required"
        )
    endpoint = urlsplit(args.url)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in ("127.0.0.1", "localhost", "::1")
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("fixed local loopback model service required")
    if not args.hardware.strip():
        raise ValueError("explicit hardware identity required")
    baseline = validate_config(json.loads(args.baseline_config.read_text()))
    candidate = StrategyBundle.load(args.candidate_bundle)
    if any(
        config["workflow"] == "whole" or config["engine_parameters"]
        for config in (baseline, candidate.config)
    ):
        raise ValueError(
            "chunked/bounded-react baseline and no server reconfiguration required"
        )
    if baseline["seed"] != candidate.config["seed"]:
        raise ValueError("baseline and candidate must use the same seed")
    model = ModelManifest.load(args.model_manifest).validate()
    if not model.served_model:
        raise ValueError("explicit unchanged served model alias required")
    if args.visual_evidence == "same-model":
        if not args.visual_capability:
            raise ValueError(
                "same-model visual diagnostics require actual capability proof"
            )
        declared_identity = json.loads(args.visual_capability.read_text())[
            "model_identity"
        ]
        if declared_identity.get("manifest") != model.as_dict():
            raise ValueError("visual capability belongs to another model manifest")
        # Check actual probe images, raw replies and ledger now. The evaluator
        # rehashes current model bytes under its cloud lease before any request.
        validate_vision_capability(args.visual_capability, declared_identity)
    elif args.visual_capability:
        raise ValueError("visual capability supplied while visual evidence is off")
    manifest = json.loads(args.manifest.read_text())
    cases = manifest.get("samples")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("purpose") != "development-regression"
        or not isinstance(cases, list)
        or len(cases) != args.expected_denominator
        or manifest.get("metric_denominator") != args.expected_denominator
    ):
        raise ValueError("full frozen development manifest denominator required")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate report identity")
    # Never read labels here. The unchanged evaluator checks file hashes and scores.
    for case in cases:
        for field in ("report", "gold"):
            digest = case.get(field + "_sha256", "")
            if (
                not Path(case[field]).is_file()
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("frozen report and label file identities required")
    idle_ledger(args.budget_ledger)
    validate_acceptance(args.sdk_acceptance)
    sandbox = verify_root(args.sandbox_root)
    if args.report_workers > 1 and not (
        args.sdk_parallel_acceptance
        and verify_parallel_sdk(args.sdk_parallel_acceptance, args.sdk_acceptance, args.url)
    ):
        raise ValueError("real matching parallel SDK proof required for two workers")
    return baseline, candidate, ids, sandbox, input_identity(args)


def partial_evidence(output, denominator):
    summary = {
        "progress": {"completed": 0, "denominator": denominator},
        "execution_records": {"total": 0, "statuses": {}},
        "calls": {"total": 0, "statuses": {}, "incomplete_call_ids": []},
    }
    path = output / "progress.json"
    if path.is_file():
        progress = json.loads(path.read_text())
        summary["progress"] = progress
        if progress.get("denominator") != denominator:
            summary["progress_error"] = (
                "progress denominator differs from frozen manifest"
            )
    records = sorted((output / "records").glob("*.json"))
    statuses = Counter()
    for path in records:
        try:
            statuses[json.loads(path.read_text()).get("status", "unknown")] += 1
        except (OSError, ValueError):
            statuses["unreadable"] += 1
    summary["execution_records"] = {"total": len(records), "statuses": dict(statuses)}
    ledger = output / "timing.sqlite"
    if ledger.is_file():
        try:
            with sqlite3.connect(
                ledger.resolve().as_uri() + "?mode=ro", uri=True
            ) as db:
                calls = db.execute("SELECT id,status,seconds FROM calls").fetchall()
            summary["calls"] = {
                "total": len(calls),
                "statuses": dict(Counter(row[1] for row in calls)),
                "incomplete_call_ids": [row[0] for row in calls if row[1] == "running"],
                "completed_call_seconds": sum(row[2] or 0 for row in calls),
                "accounting": "unique calls; overlapping seconds are not batch wall",
            }
        except sqlite3.Error as error:
            summary["calls"]["read_error"] = str(error)
    return summary


def check_evaluation(path, ids, config, bundle_sha):
    value = json.loads(path.read_text())
    reports = value.get("reports", [])
    if (
        value.get("denominator") != len(ids)
        or value.get("completed") != len(ids)
        or len(reports) != len(ids)
        or {row.get("id") for row in reports} != set(ids)
        or value.get("config") != config
        or value.get("bundle_sha256") != bundle_sha
        or value.get("cache_reused") is not False
        or not isinstance(value.get("comparison_identity"), str)
        or len(value["comparison_identity"]) != 64
        or value.get("conditions") != value["comparison_identity"]
    ):
        raise ValueError(
            "evaluation does not match full frozen arm identity/denominator"
        )
    return value


def run(args):
    baseline, candidate, ids, sandbox, identity = validate(args)
    budget = CloudBudget(args.budget_ledger)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    baseline_path = args.output_dir / "baseline.config.json"
    candidate_path = args.output_dir / "candidate.config.json"
    bundle_path = args.output_dir / "candidate.bundle.json"
    atomic_json(baseline_path, baseline)
    atomic_json(candidate_path, candidate.config)
    atomic_json(bundle_path, candidate.as_dict())
    result = {
        "schema_version": 1,
        "purpose": "full-report-diagnostic-only",
        "visual_evidence": {
            "mode": args.visual_evidence,
            "capability_sha256": file_digest(args.visual_capability)
            if args.visual_capability
            else None,
        },
        "baseline_retained": True,
        "formal_release_evidence": False,
        "rsi_enabled": False,
        "training": False,
        "expected_denominator": len(ids),
        "input_identity": identity,
        "sandbox_sha256": sandbox,
        "candidate_bundle_sha256": candidate.digest,
        "arms": [],
        "status": "running",
        "model_policy": "same explicit manifest; each evaluator binds actual model files",
        "budget_ledger": str(args.budget_ledger),
        "cloud_limit_seconds": 86400,
    }
    status_path = args.output_dir / "diagnostic-campaign.json"
    atomic_json(status_path, result)
    for index, (name, mode) in enumerate(ARMS):
        if input_identity(args) != identity:
            result["status"] = "input-changed"
            break
        idle_ledger(args.budget_ledger)
        left = len(ARMS) - index
        # Reserve ten seconds per process for run_job's TERM/KILL cleanup.
        cap = min(
            args.arm_limit_seconds,
            int((budget.remaining() - args.reserve_seconds) / left) - 10,
        )
        if cap < 60:
            result["status"] = "insufficient-budget"
            break
        output = args.output_dir / name
        command = [
            sys.executable,
            str(args.source_root / "scripts/evaluate_multitask.py"),
            "--manifest",
            str(args.manifest),
            "--model-manifest",
            str(args.model_manifest),
            "--candidate-config",
            str(baseline_path if mode == "off" else candidate_path),
            "--output-dir",
            str(output),
            "--hardware",
            args.hardware,
            "--rsi-mode",
            mode,
            "--rsi-sandbox-root",
            str(args.sandbox_root),
            "--engine",
            "tongagents",
            "--sdk-acceptance",
            str(args.sdk_acceptance),
            "--report-workers",
            str(args.report_workers),
            "--warmup",
            str(args.warmup),
            "--backend",
            args.backend,
            "--url",
            args.url,
        ]
        command += ["--visual-evidence", args.visual_evidence]
        if args.visual_capability:
            command += ["--visual-capability", str(args.visual_capability)]
        if args.sdk_parallel_acceptance:
            command += ["--sdk-parallel-acceptance", str(args.sdk_parallel_acceptance)]
        if mode != "off":
            command += ["--rsi-bundle", str(bundle_path)]
        arm = {
            "name": name,
            "mode": mode,
            "cap_seconds": cap,
            "output_dir": str(output),
            "status": "running",
            "evaluation_complete": False,
            "expected_denominator": len(ids),
            "candidate_bundle_sha256": candidate.digest if mode != "off" else None,
        }
        result["arms"].append(arm)
        atomic_json(status_path, result)
        try:
            run_job(
                budget,
                "rsi-diagnostic-" + name,
                cap,
                command,
                args.output_dir / (name + ".log"),
            )
            evaluation = check_evaluation(
                output / "evaluation.json",
                ids,
                baseline if mode == "off" else candidate.config,
                arm["candidate_bundle_sha256"],
            )
            arm.update(
                status="evaluated",
                evaluation_complete=True,
                evaluation_sha256=file_digest(output / "evaluation.json"),
                evaluation_run_id=evaluation.get("run_id"),
                comparison_identity=evaluation["comparison_identity"],
            )
        except subprocess.TimeoutExpired as error:
            arm.update(status="hard-cap", error=str(error))
        except (OSError, ValueError, KeyError, TypeError) as error:
            arm.update(
                status="invalid-evaluation",
                error=type(error).__name__ + ": " + str(error),
            )
        except RuntimeError as error:
            arm.update(status="failed", error=str(error))
        try:
            arm.update(partial_evidence(output, len(ids)))
        except (OSError, ValueError, KeyError, TypeError) as error:
            arm["evidence_read_error"] = str(error)
        conditions = arm.get("comparison_identity")
        if conditions:
            previous = result.setdefault("comparison_identity", conditions)
            if previous != conditions:
                result["status"] = "conditions-changed"
                arm["evaluation_complete"] = False
                arm["status"] = "conditions-changed"
        result["remaining_seconds"] = budget.remaining()
        atomic_json(status_path, result)
        if result["status"] == "conditions-changed":
            break
    if input_identity(args) != identity:
        result["status"] = "input-changed"
    elif result["status"] == "running":
        result["status"] = "diagnostics-finished"
    result["all_evaluations_complete"] = (
        len(result["arms"]) == len(ARMS)
        and all(arm["evaluation_complete"] for arm in result["arms"])
        and result["status"] == "diagnostics-finished"
    )
    result["remaining_seconds"] = budget.remaining()
    atomic_json(status_path, result)
    return result


def main():
    arguments = parser()
    args = arguments.parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError, KeyError, sqlite3.Error) as error:
        arguments.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
