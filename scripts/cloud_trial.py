#!/usr/bin/env python3
"""Single-PPU budget wrapper. Timeout kills only this process group, not existing services."""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.meta_optimization import CloudBudget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=Path, required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--cap-seconds", type=int, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if not args.command:
        ap.error("experiment command required")
    if args.command[0] == "--":
        args.command = args.command[1:]
    for path in (args.budget, args.log):
        if not str(path.resolve()).startswith("/workspace/"):
            ap.error("official /workspace paths required")
    budget = CloudBudget(args.budget)
    job = budget.start(args.phase, args.cap_seconds)
    start = time.perf_counter()
    status = "failed"
    try:
        with args.log.open("x") as log:
            process = subprocess.Popen(
                args.command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(
                    timeout=min(args.cap_seconds, max(1, budget.remaining() - 30))
                )
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            status = "complete" if code == 0 else "failed"
            print(
                "Experiment",
                job,
                status,
                "seconds",
                time.perf_counter() - start,
                flush=True,
            )
            raise SystemExit(code)
    finally:
        budget.finish(job, time.perf_counter() - start, status)


if __name__ == "__main__":
    main()
