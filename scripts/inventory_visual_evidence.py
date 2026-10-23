#!/usr/bin/env python3
"""Inventory every frozen development source without reading labels or calling models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.run_state import atomic_json
from bridge_evidence.contest.visual_inventory import inventory_docx


def inventory_manifest(
    manifest_path: Path, output: Path, expected_sha256: str, *, workers: int = 2
) -> dict:
    if workers not in (1, 2):
        raise ValueError("inventory workers must be 1 or 2")
    if manifest_path.stat().st_size > 16 * 1024**2:
        raise ValueError("manifest metadata limit exceeded")
    with manifest_path.open("rb") as handle:
        raw_manifest = handle.read(16 * 1024**2 + 1)
    if len(raw_manifest) > 16 * 1024**2:
        raise ValueError("manifest metadata limit exceeded")
    if hashlib.sha256(raw_manifest).hexdigest() != expected_sha256:
        raise ValueError("frozen manifest SHA256 mismatch")
    manifest = json.loads(raw_manifest)
    samples = manifest.get("samples")
    if (
        manifest.get("purpose") != "development-regression"
        or not isinstance(samples, list)
        or not samples
        or manifest.get("metric_denominator") != len(samples)
    ):
        raise ValueError("explicit complete frozen development manifest required")
    ids = [row.get("id") for row in samples]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("unique nonempty report identities required")
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": 1,
        "manifest_sha256": expected_sha256,
        "metric_denominator": len(samples),
        "processed": 0,
        "failed": 0,
        "reports": [],
        "interpretation_verified": False,
        "document_complete": False,
        "model_calls": 0,
    }
    kinds, extensions, unresolved, blobs = Counter(), Counter(), Counter(), {}

    def one(row):
        report_id = row["id"]
        filename = hashlib.sha256(report_id.encode()).hexdigest() + ".json"
        try:
            expected = row.get("report_sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("frozen report SHA256 required")
            value = inventory_docx(Path(row["report"]), expected_sha256=expected)
            result = {
                "id": report_id,
                "status": "inventoried_uninterpreted",
                "record": filename,
                "report_sha256": value["document_sha256"],
                "counts": value["counts"],
            }
        except Exception as error:
            value = {
                "schema_version": 1,
                "id": report_id,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "document_complete": False,
                "interpretation_verified": False,
            }
            result = {
                "id": report_id,
                "status": "failed",
                "record": filename,
                "error_type": type(error).__name__,
            }
        atomic_json(output / filename, value)
        return result, value

    atomic_json(output / "summary.json", summary)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, row): i for i, row in enumerate(samples)}
        results = {}
        for future in as_completed(futures):
            row, value = future.result()
            results[futures[future]] = row
            summary["processed"] += 1
            if row["status"] == "failed":
                summary["failed"] += 1
            else:
                kinds.update(value["counts"]["by_kind"])
                unresolved.update(value["counts"]["unresolved_types"])
                for occurrence in value["occurrences"]:
                    for reference in occurrence["relationships"]:
                        blob = reference.get("blob")
                        if blob and blob["sha256"] not in blobs:
                            blobs[blob["sha256"]] = blob["size_bytes"]
                            extensions[blob["extension"]] += 1
            summary.update(
                {
                    "reports": [results[i] for i in sorted(results)],
                    "pending_report_ids": [
                        ids[i] for i in range(len(ids)) if i not in results
                    ],
                    "by_kind": dict(kinds),
                    "media_extensions": dict(extensions),
                    "unresolved_types": dict(unresolved),
                    "unique_blobs": len(blobs),
                    "unique_blob_bytes": sum(blobs.values()),
                }
            )
            atomic_json(output / "summary.json", summary)
    summary["inventory_complete"] = (
        summary["processed"] == len(samples) and summary["failed"] == 0
    )
    summary["task_cost_inputs"] = {
        "image_occurrences": kinds["image"],
        "special_object_occurrences": sum(kinds.values()) - kinds["image"],
        "unique_bound_blobs": len(blobs),
        "estimated_model_seconds": None,
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(
        Path("/workspace")
    ) or args.output_dir.resolve() == Path("/workspace"):
        parser.error("official inventory artifacts must be under /workspace")
    summary = inventory_manifest(
        args.manifest, args.output_dir, args.manifest_sha256, workers=args.workers
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "metric_denominator",
                    "processed",
                    "failed",
                    "inventory_complete",
                    "by_kind",
                    "media_extensions",
                    "unique_blobs",
                    "model_calls",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["inventory_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
