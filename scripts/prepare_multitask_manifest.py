#!/usr/bin/env python3
"""Bind existing SFT gold to original cloud reports; never pass labels to inference."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.run_state import (
    atomic_json,
    file_digest,
    json_digest,
    validate_output_names,
)


def bind_bridge_entities(cases, registry=None, *, strict=False):
    """Attach reviewer-backed canonical IDs, checking all registry splits before selection."""
    registry = registry or {}
    rows = registry.get("reports", {})
    if registry and (registry.get("schema_version") != 1 or not isinstance(rows, dict)):
        raise ValueError("bridge registry v1 required")
    assignments = {}
    for row in rows.values():
        entity, split = row.get("bridge_entity"), row.get("split")
        if (
            not isinstance(entity, str)
            or not entity.strip()
            or not isinstance(split, str)
            or not split
        ):
            raise ValueError("canonical bridge_entity and split required")
        if assignments.setdefault(entity, split) != split:
            raise ValueError("bridge entity assigned to multiple splits")
    bound = []
    for case in cases:
        row = rows.get(case["report_sha256"])
        verified = bool(row and row.get("evidence") and registry.get("evidence"))
        if strict and not verified:
            raise ValueError("bridge verification evidence required for every report")
        bound.append(
            {
                **case,
                "bridge_entity": row["bridge_entity"] if row else None,
                "entity_verified": verified,
                "entity_evidence": row.get("evidence") if row else None,
                "split": row["split"] if row else "development-regression",
            }
        )
    return bound


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", type=Path, required=True)
    ap.add_argument("--reports-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--year", default="2018")
    ap.add_argument("--diagnostic-only", action="store_true")
    ap.add_argument("--bridge-registry", type=Path)
    ap.add_argument("--strict-entity-split", action="store_true")
    ap.add_argument(
        "--split", help="Select a split after validating the complete bridge registry"
    )
    args = ap.parse_args()
    if not str(args.output_dir.resolve()).startswith("/workspace/"):
        ap.error("official cloud artifacts only")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    by_name = {}
    for report in args.reports_root.rglob("*"):
        if report.is_file() and report.suffix.lower() in (".doc", ".docx"):
            by_name.setdefault(report.name, []).append(report)
    cases = []
    with args.sft.open() as f:
        for line in f:
            row = json.loads(line)
            meta = row.get("meta", {})
            if str(meta.get("year", "")).removesuffix("年") != args.year:
                continue
            paths = by_name.get(meta.get("report"), [])
            if meta.get("report_sha256"):
                paths = [p for p in paths if file_digest(p) == meta["report_sha256"]]
            if len(paths) != 1:
                raise ValueError(
                    "cannot uniquely bind source report " + str(meta.get("report"))
                )
            report = paths[0]
            sample = json_digest(
                {k: row[k] for k in ("instruction", "input", "output")}
            )
            gold = args.output_dir / (sample + ".gold.json")
            atomic_json(gold, json.loads(row["output"]))
            cases.append(
                {
                    "id": sample,
                    "report": str(report),
                    "report_sha256": file_digest(report),
                    "gold": str(gold),
                    "gold_sha256": file_digest(gold),
                    "bridge_entity": None,
                    "entity_verified": False,
                    "split": "development-regression",
                    "source_characters": len(row["input"]),
                }
            )
    registry = (
        json.loads(args.bridge_registry.read_text()) if args.bridge_registry else None
    )
    cases = bind_bridge_entities(cases, registry, strict=args.strict_entity_split)
    if args.split:
        cases = [case for case in cases if case["split"] == args.split]
    validate_output_names([Path(c["report"]) for c in cases])
    cases.sort(key=lambda c: (c["source_characters"], c["id"]))
    if args.diagnostic_only and cases:
        # Source length only, never gold content: short, median, long, and longest.
        indices = sorted(set((0, len(cases) // 3, 2 * len(cases) // 3, len(cases) - 1)))
        cases = [cases[i] for i in indices]
    if not cases:
        raise ValueError("empty regression manifest")
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": 2,
            "purpose": "development-regression",
            "samples": cases,
            "scope": "source-length diagnostic"
            if args.diagnostic_only
            else "all observed 2018 development reports",
            "sft_sha256": file_digest(args.sft),
            "metric_denominator": len(cases),
            "generalization_claim": False,
            "split_verification": {
                "strict": args.strict_entity_split,
                "bridge_registry_sha256": file_digest(args.bridge_registry)
                if args.bridge_registry
                else None,
                "all_entities_verified": all(case["entity_verified"] for case in cases),
                "independence": "review-required; entity disjointness alone is not independent data",
            },
        },
    )
    print("Frozen regression denominator", len(cases))


if __name__ == "__main__":
    main()
