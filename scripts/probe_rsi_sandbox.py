#!/usr/bin/env python3
"""Real namespace/seccomp acceptance; no model request, no simulated verified status."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.rsi import StrategyBundle
from bridge_evidence.contest.rsi_sandbox import SandboxWorker, prepare_root
from bridge_evidence.contest.run_state import atomic_json, file_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare_root(args.root)
    bundle = StrategyBundle.baseline()
    results = []
    for probe in (
        "read",
        "write",
        "network",
        "fork",
        "exec",
        "labels",
        "memory",
        "mount",
        "ptrace",
        "flood",
    ):
        worker = SandboxWorker(args.root, bundle, probe=probe)
        try:
            worker.send({"function": "probe", "args": [probe]})
            if probe == "flood":
                import time

                started = time.monotonic()
                try:
                    worker.read(2)
                    value = {"blocked": False}
                except (TimeoutError, RuntimeError):
                    # Parent enforces wall-clock termination even when RLIMIT_CPU is not reached.
                    worker.close()
                    value = {"blocked": time.monotonic() - started <= 3}
            else:
                value = worker.read(1)
            results.append(
                {
                    "probe": probe,
                    "blocked": value == {"blocked": True},
                    "kernel": worker.kernel_proof,
                }
            )
        finally:
            if probe != "flood":
                worker.close()
    worker = SandboxWorker(args.root, bundle)
    try:
        action = worker.invoke(
            "choose_action",
            [{"phase": "schedule", "batch_size": 1, "remaining_calls": 2}],
            timeout=1,
        )
    finally:
        worker.close()
    passed = all(r["blocked"] for r in results) and action == {
        "action": "extract",
        "batch_size": 1,
    }
    identity = json.loads(
        (args.root.parent / (args.root.name + ".identity.json")).read_text()
    )
    report = {
        "passed": passed,
        "runtime_digest": identity["digest"],
        "probes": results,
        "worker_sha256": file_digest(
            ROOT / "src/bridge_evidence/contest/rsi_worker.py"
        ),
        "launcher_sha256": file_digest(
            ROOT / "src/bridge_evidence/contest/rsi_namespace.py"
        ),
    }
    atomic_json(args.root.parent / (args.root.name + ".acceptance.json"), report)
    print(json.dumps(report))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
