#!/usr/bin/env python3
"""Wait for the existing experiment; qualify RSI without resetting the 24h ledger."""

import argparse
import fcntl
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_rsi_evolution import run_job

from bridge_evidence.contest.backends import normalize_api_url
from bridge_evidence.contest.candidate import validate_config
from bridge_evidence.contest.meta_optimization import CloudBudget
from bridge_evidence.contest.run_state import atomic_json


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "manifest", "model-manifest", "sdk-acceptance"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--url", type=normalize_api_url, default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=ROOT / "configs/contest/chunked-baseline.json",
        help="Explicit chunked or bounded-react baseline for same-executor RSI comparisons",
    )
    args = parser.parse_args()
    try:
        baseline = validate_config(json.loads(args.baseline_config.read_text()))
    except (OSError, ValueError) as error:
        parser.error("invalid baseline configuration: " + str(error))
    if baseline["workflow"] == "whole":
        parser.error("whole baseline cannot start RSI search; choose chunked or bounded-react")
    if not str(args.root.resolve()).startswith("/workspace/"):
        parser.error("official /workspace required")
    with (args.root / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = args.root / "rsi-campaign-status.json"
        status = {
            "state": "waiting-existing-experiment",
            "submitted": False,
            "training": False,
            "official_strategy": "baseline",
            "source": str(ROOT),
        }
        atomic_json(status_path, status)
        ledger = args.root / "budget.sqlite"
        # Do not recover, stop or alter any job owned by the old wrapper.
        while True:
            with sqlite3.connect(ledger) as db:
                running = db.execute(
                    "SELECT id FROM jobs WHERE status='running'"
                ).fetchall()
            if not running:
                break
            time.sleep(30)
        budget = CloudBudget(ledger)
        experiment = args.root / ("rsi-" + ROOT.name)
        experiment.mkdir(exist_ok=False)
        sandbox = experiment / "sandbox-root"
        status.update(state="isolating", remaining_seconds=budget.remaining())
        atomic_json(status_path, status)
        try:
            if budget.remaining() < 34800 + 1800:
                raise RuntimeError(
                    "insufficient shared budget for fresh anchor and two complete receipts"
                )
            run_job(
                budget,
                "rsi-sandbox-acceptance",
                900,
                [
                    sys.executable,
                    str(ROOT / "scripts/probe_rsi_sandbox.py"),
                    "--root",
                    str(sandbox),
                    "--prepare",
                ],
                experiment / "sandbox.log",
            )
            parallel = experiment / "parallel"
            run_job(
                budget,
                "rsi-parallel-sdk-acceptance",
                900,
                [
                    sys.executable,
                    str(ROOT / "scripts/probe_rsi_parallel_sdk.py"),
                    "--model-manifest",
                    str(args.model_manifest),
                    "--sdk-acceptance",
                    str(args.sdk_acceptance),
                    "--output-dir",
                    str(parallel),
                    "--url",
                    args.url,
                ],
                experiment / "parallel.log",
            )
            proof = parallel / "parallel-acceptance.json"
            command = [
                sys.executable,
                str(ROOT / "scripts/run_rsi_evolution.py"),
                "--manifest",
                str(args.manifest),
                "--model-manifest",
                str(args.model_manifest),
                "--baseline-config",
                str(args.baseline_config),
                "--sandbox-root",
                str(sandbox),
                "--budget-ledger",
                str(ledger),
                "--output-dir",
                str(experiment / "search"),
                "--hardware",
                args.hardware,
                "--url",
                args.url,
                "--sdk-acceptance",
                str(args.sdk_acceptance),
                "--sdk-parallel-acceptance",
                str(proof),
            ]
            status.update(
                state="running-inference-only-evolution", experiment=str(experiment)
            )
            atomic_json(status_path, status)
            subprocess.run(command, check=True)
            selection = json.loads((experiment / "search/selection.json").read_text())
            status.update(state="complete", selection=selection)
        except Exception as error:
            status.update(
                state="baseline-retained",
                error=type(error).__name__ + ": " + str(error),
            )
        status["remaining_seconds"] = budget.remaining()
        atomic_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
