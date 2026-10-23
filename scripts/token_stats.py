#!/usr/bin/env python3
"""Report full-sequence SFT token lengths using a local tokenizer, without weights.

Outputs contain aggregate lengths and input/config hashes, never report text.
No runtime or training-speed estimate is inferred from token counts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "_token_stats_helpers", Path(__file__).with_name("benchmark_lora.py"))
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def read_samples(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                                for key in ("instruction", "input", "output")):
                raise ValueError(f"Line {line_number}: nonempty instruction/input/output strings required")
            content = {key: row[key] for key in ("instruction", "input", "output")}
            rows.append({**content, "line": line_number, "year": "",
                         "sample_sha256": bench.digest(content),
                         "prompt_sha256": bench.digest({key: row[key] for key in ("instruction", "input")})})
    if not rows:
        raise ValueError("SFT is empty")
    return rows


def summarize(rows, tokenizer, max_lengths):
    if not max_lengths or any(type(length) is not int or length <= 0 for length in max_lengths):
        raise ValueError("max-lengths must be positive integers")
    # Use the same template, EOS and tokenization as the current training helper.
    lengths = [{key: value for key, value in bench.tokenize_row(row, tokenizer).items()
                if key in {"tokens", "prompt_tokens", "answer_tokens"}} for row in rows]
    if not lengths:
        raise ValueError("SFT is empty")

    def distribution(key):
        values = sorted(row[key] for row in lengths)
        return {"min": values[0], "max": values[-1], "mean": statistics.mean(values),
                **{f"p{percent}": values[min(len(values) - 1, max(0, (len(values) * percent + 99) // 100 - 1))]
                   for percent in (50, 90, 95, 99)}}

    return {
        "schema_version": 1, "samples": len(lengths), "full_sequence_tokens": sum(row["tokens"] for row in lengths),
        "prompt": distribution("prompt_tokens"), "answer_including_eos": distribution("answer_tokens"),
        "full_sequence": distribution("tokens"),
        "coverage": [{"max_length": length, "complete_samples": sum(row["tokens"] <= length for row in lengths),
                      "total_samples": len(lengths)} for length in sorted(set(max_lengths))],
        "truncation": False, "speed_estimate": None,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional NEW JSON file; otherwise print aggregate JSON")
    parser.add_argument("--max-lengths", nargs="+", type=int, default=[8192, 16384, 32768, 65536])
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.model.is_dir() or not (args.model / "config.json").is_file() or not args.sft.is_file():
        parser.error("Explicit local model directory with config.json and an SFT JSONL file are required")
    if args.output is not None:
        output, model = args.output.resolve(), args.model.resolve()
        if output.exists() or model == output or model in output.parents:
            parser.error("Output must be a NEW file outside the model directory")
    if any(length <= 0 for length in args.max_lengths):
        parser.error("max-lengths must be positive")
    rows = read_samples(args.sft)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=False, local_files_only=True)
    result = summarize(rows, tokenizer, args.max_lengths)
    result.update(sft_sha256=bench.file_digest(args.sft),
                  model_config_sha256=bench.file_digest(args.model / "config.json"),
                  helper_sha256=bench.file_digest(Path(bench.__file__)))
    if args.output is not None:
        bench.write_json(args.output, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
