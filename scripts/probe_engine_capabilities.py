#!/usr/bin/env python3
"""Vendor capability discovery is not an optimization performance certificate."""

import argparse
import dataclasses
import importlib.metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.run_state import atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from vllm.engine.arg_utils import EngineArgs

    fields = {f.name for f in dataclasses.fields(EngineArgs)}
    controls = {
        "prefix_cache": "enable_prefix_caching",
        "graph_execution": "enforce_eager",
        "quantization": "quantization",
        "speculative_decoding": "speculative_config",
    }
    result = {
        "vendor_vllm": importlib.metadata.version("vllm"),
        "features": {
            name: {
                "argument_recognized": flag in fields,
                "ppu_generation_verified": False,
                "quality_speed_verified": False,
                "enabled_by_this_probe": False,
            }
            for name, flag in controls.items()
        },
        "sglang": None,
    }
    try:
        result["sglang"] = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        pass
    atomic_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
