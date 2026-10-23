#!/usr/bin/env python3
"""Score frozen document-agent artifacts offline; no network/model requests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.evaluation.document_artifacts import evaluate_artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts-index", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True, help="Previously frozen cohort file SHA-256")
    parser.add_argument("--index-sha256", required=True, help="Previously frozen run artifact index SHA-256")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New score JSON; refuses overwrite")
    args = parser.parse_args(argv)
    try:
        result = evaluate_artifacts(args.manifest, args.artifacts_index, data_root=args.data_root,
                                    artifact_root=args.artifact_root, manifest_sha256=args.manifest_sha256,
                                    index_sha256=args.index_sha256)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Offline evaluation rejected: {exc}\n")
    print(json.dumps({"output": str(args.output), "reports_total": result["final"]["reports_total"],
                      "reports_failed": result["final"]["reports_failed"],
                      "raw_available_reports": result["raw_available_reports"],
                      "release_qualified": False}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
