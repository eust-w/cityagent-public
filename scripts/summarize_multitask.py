#!/usr/bin/env python3
"""Recompute a complete frozen denominator from finished and interrupted run records."""

import argparse
import importlib.util
import json
import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.execution_audit import validate_execution_record
from bridge_evidence.contest.run_state import (
    atomic_json,
    document_identity,
    json_digest,
    record_name,
)
from bridge_evidence.contest.timing import TimingLedger
from bridge_evidence.evaluation.generation import (
    METRIC_VERSION,
    aggregate_reports,
    score_report,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--budget-ledger", type=Path)
    ap.add_argument("--job-id")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location(
        "multitask_protocol", ROOT / "scripts/evaluate_multitask.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    manifest, cases = mod.load_cases(args.manifest)
    run = json.loads((args.run_dir / "run_manifest.json").read_text())
    options = run["identity"]["options"]
    raw_scores, final_scores, details = [], [], []
    for case in cases:
        path = Path(case["report"])
        doc = document_identity(path, path.parent)
        record = args.run_dir / "records" / record_name(doc)
        raw = final = None
        issues = ["missing or interrupted result"]
        if record.is_file():
            value = json.loads(record.read_text())
            engine = value.get("workflow", {}).get("engine")
            issues = validate_execution_record(
                value,
                expected_document=doc,
                expected_run_fingerprint=run["run_fingerprint"],
                output_path=args.run_dir / "result" / (path.stem + ".docx"),
                profile="contest-sdk" if engine == "tongagents" else "legacy",
            )
            if not issues:
                raw = json.loads(value["raw_output"])
                final = value["final_data"]
        gold = json.loads(Path(case["gold"]).read_text())
        raw_scores.append(score_report(gold, raw))
        final_scores.append(score_report(gold, final))
        details.append({"id": case["id"], "errors": issues})
    aggregate = aggregate_reports(final_scores)
    ledger = TimingLedger(args.run_dir / "timing.sqlite")
    measured_wall = ledger.summary()["spans"].get("batch_wall")
    if args.budget_ledger or args.job_id:
        if not (args.budget_ledger and args.job_id):
            ap.error("budget ledger and job id must be supplied together")
        with sqlite3.connect(args.budget_ledger) as db:
            job = db.execute(
                "SELECT seconds,status FROM jobs WHERE id=?", (args.job_id,)
            ).fetchone()
        if (
            not job
            or job[1] == "running"
            or job[0] is None
            or not math.isfinite(job[0])
            or job[0] <= 0
        ):
            ap.error("finished, measured experiment required before recovery")
        ledger.recover()
        measured_wall = job[0]

    result = {
        "protocol": mod.PROTOCOL,
        "metric_version": METRIC_VERSION,
        "purpose": "development-regression",
        "denominator": len(cases),
        "f1": [aggregate["sections"][s]["f1"] for s in ("basic", "advice", "defects")],
        "failures": aggregate["reports_failed"],
        "final_metrics": aggregate,
        "raw_metrics": aggregate_reports(raw_scores),
        "details": details,
        "timing": ledger.summary(),
        "config": options["config"],
        "repeated": False,
        "conditions": json_digest(
            {
                "hardware": options["hardware"],
                "model": options["model"],
                "manifest_sha256": mod.file_digest(args.manifest),
                "metric": METRIC_VERSION,
                "warmup": options["warmup"],
                "backend": options["backend"],
                "engine": options.get(
                    "engine",
                    "tongagents" if options.get("sdk_acceptance_sha256") else "direct",
                ),
                "sdk_acceptance_sha256": options.get("sdk_acceptance_sha256"),
            }
        ),
        "wall_seconds": measured_wall,
        "recovered_job_id": args.job_id,
        "complete_batch": not any(d["errors"] for d in details),
    }
    atomic_json(args.run_dir / "recomputed-evaluation.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("details", "raw_metrics", "final_metrics")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
