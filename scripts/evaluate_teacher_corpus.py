from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bridge_evidence.config import load_settings
from bridge_evidence.evaluation.teacher import compare_teacher_target
from bridge_evidence.orchestrator import BridgeEvidenceOrchestrator


def _filter_current_summary(teacher: dict[str, Any]) -> dict[str, Any]:
    filtered = dict(teacher)
    filtered["targets"] = {
        field: target
        for field, target in teacher.get("targets", {}).items()
        if not field.startswith("scores.previous.")
        and target.get("status") != "benchmark_convention_conflict"
    }
    return filtered


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    included = [
        field
        for row in rows
        for field in row.get("comparison", {}).get("fields", [])
        if field.get("included_in_objective")
    ]
    return {
        "samples": len(rows),
        "successful_samples": sum("comparison" in row for row in rows),
        "matched_fields": sum(bool(field.get("matched")) for field in included),
        "total_fields": len(included),
        "objective_score": (
            sum(bool(field.get("matched")) for field in included) / len(included)
            if included
            else 0.0
        ),
        "failures": [
            {"teacher": row["teacher"], "error": row["error"]}
            for row in rows
            if "error" in row
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run source reports blind, then score them against a teacher corpus")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--llm-mode", choices=("off", "auto", "on"), default="off")
    parser.add_argument("--split", action="append", choices=("dev_teacher", "locked_audit", "temporal_holdout"))
    parser.add_argument("--teacher", action="append", help="Optional teacher filename filter; repeatable")
    parser.add_argument(
        "--workflow-profile",
        choices=("full", "summary"),
        default="full",
        help="summary keeps the real ReAct planner/retrieval/review loop but omits unrelated detail generation",
    )
    parser.add_argument("--project-root", type=Path, default=Path(".."))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_splits = set(args.split or ())
    selected_teachers = set(args.teacher or ())
    samples = [
        sample
        for sample in manifest.get("samples", [])
        if not selected_splits or sample["split"] in selected_splits
        if not selected_teachers or sample["teacher"] in selected_teachers
    ]

    output_dir = args.output_dir.resolve()
    states_dir = output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings(args.config, llm_mode=args.llm_mode)
    settings.work_dir = output_dir / "work"
    # Every evaluation run owns its cache.  In online mode this is essential:
    # an earlier run's semantic/model response must not masquerade as a fresh
    # network call in the execution audit.
    settings.cache_dir = output_dir / f"cache-{args.llm_mode}"
    runner = BridgeEvidenceOrchestrator(settings)

    rows: list[dict[str, Any]] = []
    for sample in samples:
        teacher_path = manifest_path.parent / sample["teacher"]
        teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
        source_path = project_root / teacher["source"]["path"]
        row: dict[str, Any] = {
            "split": sample["split"],
            "family": sample["family"],
            "teacher": sample["teacher"],
            "source": str(source_path),
        }
        actual_hash = _sha256(source_path)
        if actual_hash != teacher["source"]["sha256"]:
            row["error"] = f"source hash mismatch: {actual_hash}"
            rows.append(row)
            continue
        try:
            state = runner.process(
                source_path,
                None,
                json_dir=states_dir,
                render=False,
                workflow_profile=args.workflow_profile,
            )
            row["comparison"] = compare_teacher_target(
                state.model_dump(mode="json"),
                _filter_current_summary(teacher),
            )
            row["metrics"] = state.metrics.model_dump(mode="json")
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)

    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split_rows[row["split"]].append(row)
    aggregates = {split: _aggregate(items) for split, items in sorted(split_rows.items())}
    aggregates["all"] = _aggregate(rows)

    thresholds = manifest.get("acceptance_thresholds", {})
    evaluated_splits = set(split_rows)
    gates = {
        "all_objective": aggregates["all"]["objective_score"] >= float(thresholds.get("all_objective_score", 1.0)),
        "locked_audit": (
            "locked_audit" not in evaluated_splits
            or aggregates["locked_audit"]["objective_score"]
            >= float(thresholds.get("locked_audit_objective_score", 1.0))
        ),
        "temporal_holdout": (
            "temporal_holdout" not in evaluated_splits
            or aggregates["temporal_holdout"]["objective_score"]
            >= float(thresholds.get("temporal_holdout_objective_score", 1.0))
        ),
        "no_processing_failures": not aggregates["all"]["failures"],
    }
    report = {
        "schema_version": "1.0",
        "llm_mode": args.llm_mode,
        "workflow_profile": args.workflow_profile,
        "manifest": str(manifest_path),
        "aggregates": aggregates,
        "gates": gates,
        "passed": all(gates.values()),
        "samples": rows,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "aggregates": aggregates, "gates": gates, "passed": report["passed"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
