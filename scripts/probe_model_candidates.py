#!/usr/bin/env python3
"""Report mounted candidate architectures and explicit eligibility gates, without downloading."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.run_state import atomic_json, file_digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=Path, default=Path("/model"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--eligibility-evidence", type=Path)
    args = ap.parse_args()
    eligibility = (
        json.loads(args.eligibility_evidence.read_text())
        if args.eligibility_evidence
        else {}
    )
    rows = []
    for name in ("Qwen3.8-27B", "Qwen3.5-9B", "Qwen3.6-35B-A3B"):
        config = args.model_root / name / "config.json"
        row = {
            "name": name,
            "mounted": config.is_file(),
            "eligibility": "unverified",
            "ppu_generation": "unverified",
            "adapter": "fresh-base-specific-training-required",
        }
        if config.is_file():
            row["model_type"] = json.loads(config.read_text()).get("model_type")
            row["config_sha256"] = file_digest(config)
        if (
            name in eligibility.get("allowed_models", [])
            and eligibility.get("official_source")
            and eligibility.get("team_scope")
        ):
            row["eligibility"] = "documented-pending-official-confirmation"
            row["eligibility_evidence_sha256"] = file_digest(args.eligibility_evidence)
        rows.append(row)
    atomic_json(
        args.output,
        {"schema_version": 1, "candidates": rows, "automatic_activation": False},
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
