#!/usr/bin/env python3
"""Index audited allowed reports, query facts or compare supported measurements."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest.execution_audit import audit_execution_records
from bridge_evidence.contest.report_qa import ReportIndex, validate_index_ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--records", type=Path)
    ap.add_argument("--allowed-reports", type=Path)
    ap.add_argument("--input-dir", type=Path)
    ap.add_argument("--result-dir", type=Path)
    ap.add_argument("--run-fingerprint")
    ap.add_argument("--timing-ledger", type=Path)
    ap.add_argument(
        "--audit-profile", choices=("contest-sdk", "legacy"), default="contest-sdk"
    )
    ap.add_argument("--bridge")
    ap.add_argument("--year", type=int)
    ap.add_argument("--component")
    ap.add_argument("--disease")
    ap.add_argument("--compare-years", nargs="+", type=int)
    ap.add_argument(
        "--kind",
        choices=("occurrences", "defect-identities", "measurements", "scores"),
        default="occurrences",
    )
    ap.add_argument("--attribute")
    ap.add_argument("--unit")
    ap.add_argument(
        "--statistic", choices=("maximum", "minimum", "mean", "total", "unspecified")
    )
    ap.add_argument("--minimum", type=float)
    ap.add_argument("--maximum", type=float)
    args = ap.parse_args()
    index = ReportIndex(args.index)
    if args.records:
        if not all(
            (
                args.allowed_reports,
                args.input_dir,
                args.result_dir,
                args.run_fingerprint,
                args.timing_ledger,
            )
        ):
            ap.error(
                "ingestion requires --allowed-reports, --input-dir, --result-dir, --run-fingerprint, --timing-ledger"
            )
        audit = audit_execution_records(
            args.input_dir,
            args.result_dir,
            args.records,
            args.run_fingerprint,
            profile=args.audit_profile,
        )
        if not audit["passed"]:
            ap.error("execution evidence audit failed: " + "; ".join(audit["errors"]))
        allowed = set(json.loads(args.allowed_reports.read_text()))
        records = [json.loads(path.read_text()) for path in args.records.glob("*.json")]
        # Validate the entire import before writing any graph.
        from bridge_evidence.contest.facts import build_fact_graph

        for record in records:
            validate_index_ledger(record, args.timing_ledger)
            build_fact_graph(record, strict=True, allowed_sha256=allowed)
        for record in records:
            index.ingest(record, allowed_sha256=allowed, strict=True)
    filters = dict(component=args.component, disease=args.disease)
    if args.kind == "measurements":
        filters.update(
            attribute=args.attribute,
            unit=args.unit,
            statistic=args.statistic,
            minimum=args.minimum,
            maximum=args.maximum,
        )
        if args.compare_years:
            if not args.attribute:
                ap.error("--attribute required for measurement comparison")
            result = index.compare_measurements(
                args.bridge, args.compare_years, **filters
            )
        else:
            result = index.measurements(bridge=args.bridge, year=args.year, **filters)
    elif args.kind == "scores":
        result = index.scores(
            bridge=args.bridge, year=args.year, attribute=args.attribute
        )
    elif args.kind == "defect-identities":
        result = index.count(
            bridge=args.bridge, year=args.year, basis="defect-identities", **filters
        )
    else:
        result = (
            index.compare(args.bridge, args.compare_years, **filters)
            if args.compare_years
            else index.query(bridge=args.bridge, year=args.year, **filters)
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
