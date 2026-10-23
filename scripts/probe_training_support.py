#!/usr/bin/env python3
"""Read-only single-PPU training probe; no vendor dependency replacement."""

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.run_state import atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not str(args.output.resolve()).startswith("/workspace/"):
        ap.error("official cloud artifacts required")
    result = {
        "training_verified": False,
        "external_training": False,
        "model_type": json.loads((args.model / "config.json").read_text()).get(
            "model_type"
        ),
    }
    for module in ("torch", "transformers", "peft"):
        try:
            importlib.import_module(module)
            result[module] = {
                "imported": True,
                "version": importlib.metadata.version(module),
            }
        except Exception as e:
            result[module] = {"imported": False, "error_type": type(e).__name__}
    if result["torch"]["imported"]:
        import torch

        result["device_count"] = torch.cuda.device_count()
        result["memory_free_total"] = [
            torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count())
        ]
    result["next_gate"] = (
        "approved revision dataset and isolated single-PPU backward/adapter-save acceptance; current NVIDIA/FSDP path is not PPU validation"
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
