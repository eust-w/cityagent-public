#!/usr/bin/env python3
"""Materialize approved overlays and chunk SFT after bridge-entity partitioning."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.long_document import DocumentSnapshot, plan_chunks
from bridge_evidence.contest.run_state import atomic_json, file_digest
from bridge_evidence.contest.training_revisions import (
    apply_revision,
    build_subtasks,
    entity_partition,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--revision-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    reports = entity_partition(json.loads(args.manifest.read_text())["samples"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = []
    for report in reports:
        snapshot = DocumentSnapshot.from_docx(Path(report["report"]))
        original = json.loads(Path(report["gold"]).read_text())
        revision_path = args.revision_dir / (report["id"] + ".json")
        original_sha = file_digest(Path(report["gold"]))
        if not revision_path.is_file():
            summary.append({"id": report["id"], "status": "no-approved-overlay"})
            continue
        revision = json.loads(revision_path.read_text())
        fixed = apply_revision(original, snapshot, revision)
        plan = plan_chunks(snapshot, tokenizer)
        subtasks = build_subtasks(snapshot, plan, revision.get("subtasks", []))
        atomic_json(args.output_dir / (report["id"] + ".label.json"), fixed)
        with (args.output_dir / (report["id"] + ".sft.jsonl")).open("x") as f:
            for row in subtasks:
                f.write(
                    json.dumps(
                        {
                            **row,
                            "bridge_entity": report["bridge_entity"],
                            "split": report["split"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        if file_digest(Path(report["gold"])) != original_sha:
            raise RuntimeError("original labels changed")
        summary.append(
            {
                "id": report["id"],
                "split": report["split"],
                "original_sha256": original_sha,
                "subtasks": len(subtasks),
            }
        )
    atomic_json(
        args.output_dir / "revision-manifest.json",
        {
            "schema_version": 1,
            "records": summary,
            "claim": "development regression only; independent generalization not established",
        },
    )


if __name__ == "__main__":
    main()
