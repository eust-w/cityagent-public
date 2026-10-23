#!/usr/bin/env python3
"""Freeze all 2018 generation cases without loading a model or discarding long inputs.

Diagnostic selection uses only prompt length and the instruction/input content hash.
Gold answers bind the sample identity but cannot affect its partition. The rest is
an internal final-acceptance partition, not a claim of a pristine blind test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any


_BENCH_PATH = Path(__file__).with_name("benchmark_lora.py")
_SPEC = importlib.util.spec_from_file_location("generation_manifest_benchmark", _BENCH_PATH)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

BUCKETS = (
    {"name": "lt_8k", "min_inclusive": 1, "max_inclusive": 8191},
    {"name": "8k_to_16k", "min_inclusive": 8192, "max_inclusive": 16383},
    {"name": "16k_to_32k", "min_inclusive": 16384, "max_inclusive": 32768},
    {"name": "gt_32k", "min_inclusive": 32769, "max_inclusive": None},
)
PARTITIONS = ("diagnostic", "final_acceptance")


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name}: lowercase SHA-256 required")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name}: positive integer required")
    return value


def prompt_length_bucket(tokens: int) -> str:
    _positive_int(tokens, "prompt_tokens")
    return next(bucket["name"] for bucket in BUCKETS
                if tokens >= bucket["min_inclusive"]
                and (bucket["max_inclusive"] is None or tokens <= bucket["max_inclusive"]))


def _ids_hash(samples: list[dict], field: str) -> str:
    return bench.digest(sorted(sample[field] for sample in samples))


def build_manifest(rows: list[dict], token_metadata: list[dict], *,
                   sft_sha256: str, seed: int = 20261009) -> dict:
    """Bind complete benchmark.load_rows output to complete tokenize_row/audit metadata.

    Metadata must cover every SFT row, including non-2018 and previously skipped
    rows. `status` is deliberately ignored: benchmark bucket/limit decisions must
    not determine the generation denominator. The seed is provenance only; SHA
    ordering is deliberately deterministic and is not randomized with the seed.
    """
    _sha(sft_sha256, "sft_sha256")
    if type(seed) is not int:
        raise ValueError("seed: integer required")
    if not isinstance(rows, list) or not rows or not isinstance(token_metadata, list):
        raise ValueError("nonempty rows and complete token_metadata lists required")

    metadata_by_sample = {}
    for item in token_metadata:
        if not isinstance(item, dict):
            raise ValueError("token metadata rows must be objects")
        sample_id = _sha(item.get("sample_sha256"), "metadata sample_sha256")
        if sample_id in metadata_by_sample:
            raise ValueError(f"duplicate token metadata sample: {sample_id}")
        metadata_by_sample[sample_id] = item

    seen_samples, seen_prompts, seen_lines = set(), set(), set()
    samples = []
    for row in rows:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("instruction", "input", "output")
        ):
            raise ValueError("rows must come from benchmark.load_rows")
        sample_id = _sha(row.get("sample_sha256"), "sample_sha256")
        prompt_id = _sha(row.get("prompt_sha256"), "prompt content SHA-256")
        if sample_id != bench.digest({key: row[key] for key in ("instruction", "input", "output")}):
            raise ValueError("sample hash does not match SFT content")
        if prompt_id != bench.digest({key: row[key] for key in ("instruction", "input")}):
            raise ValueError("prompt content hash does not match SFT content")
        line = _positive_int(row.get("line"), "SFT line")
        year = row.get("year")
        if not isinstance(year, str) or re.fullmatch(r"(?:19|20)\d{2}", year) is None:
            raise ValueError("explicit normalized year required")
        if sample_id in seen_samples or prompt_id in seen_prompts or line in seen_lines:
            raise ValueError("duplicate SFT sample, prompt, or line")
        seen_samples.add(sample_id)
        seen_prompts.add(prompt_id)
        seen_lines.add(line)
        metadata = metadata_by_sample.get(sample_id)
        if metadata is None:
            raise ValueError(f"missing token metadata for SFT line {line}")
        for key, expected in (("prompt_sha256", prompt_id), ("line", line), ("year", year)):
            if metadata.get(key) != expected or type(metadata.get(key)) is not type(expected):
                raise ValueError(f"token metadata {key} mismatch at SFT line {line}")
        split = "holdout" if year == "2018" else "train"
        if "split" in metadata and metadata["split"] != split:
            raise ValueError(f"token metadata split mismatch at SFT line {line}")
        prompt_tokens = _positive_int(metadata.get("prompt_tokens"), "prompt_tokens")
        full_tokens = _positive_int(metadata.get("tokens"), "tokens")
        answer_tokens = _positive_int(metadata.get("answer_tokens"), "answer_tokens")
        if prompt_tokens + answer_tokens != full_tokens:
            raise ValueError(f"inconsistent token lengths at SFT line {line}")
        token_sha = _sha(metadata.get("token_sha256"), "full sequence token SHA-256")
        if year == "2018":
            samples.append({
                "sample_sha256": sample_id, "prompt_content_sha256": prompt_id,
                "full_sequence_token_sha256": token_sha,
                "line": line, "year": year, "prompt_tokens": prompt_tokens,
                "full_tokens": full_tokens, "answer_tokens": answer_tokens,
                "prompt_length_bucket": prompt_length_bucket(prompt_tokens),
            })
    if set(metadata_by_sample) != seen_samples:
        raise ValueError("token metadata contains samples not present in SFT")
    if not samples:
        raise ValueError("no 2018 holdout samples")

    # Prompt hashes contain no answer. Sorting by sample_sha256 here would let
    # a corrected gold answer change which report is opened for diagnosis.
    samples.sort(key=lambda item: item["prompt_content_sha256"])
    seen_buckets = Counter()
    for sample in samples:
        bucket = sample["prompt_length_bucket"]
        sample["partition"] = "diagnostic" if seen_buckets[bucket] < 2 else "final_acceptance"
        seen_buckets[bucket] += 1
    partitioned = {name: [sample for sample in samples if sample["partition"] == name]
                   for name in PARTITIONS}
    return {
        "schema_version": 1, "sft_sha256": sft_sha256, "source_row_count": len(rows),
        "holdout_year": "2018", "seed": seed,
        "selection": {
            "diagnostics_per_bucket": 2,
            "order_by": "prompt_content_sha256 ascending",
            "seed_role": "provenance only; selection does not use randomness",
            "buckets": [dict(bucket) for bucket in BUCKETS],
            "selection_inputs": ["year", "prompt_tokens", "prompt_content_sha256"],
            "no_length_exclusions": True,
        },
        "sample_count": len(samples),
        "partition_counts": {name: len(group) for name, group in partitioned.items()},
        "bucket_counts": {bucket["name"]: seen_buckets[bucket["name"]] for bucket in BUCKETS},
        "cohort_sha256": _ids_hash(samples, "sample_sha256"),
        "cohort_prompt_content_sha256": _ids_hash(samples, "prompt_content_sha256"),
        "partition_sha256": {name: _ids_hash(group, "sample_sha256") for name, group in partitioned.items()},
        "partition_prompt_content_sha256": {
            name: _ids_hash(group, "prompt_content_sha256") for name, group in partitioned.items()},
        "interpretation": (
            "Internal validation: 2018 teacher-forced losses were already observed. "
            "Use diagnostics for changes; freeze the candidate before inspecting final_acceptance. "
            "Count context/output-budget failures and invalid outputs in each full partition denominator. "
            "Repeated final_acceptance tuning makes it development data, not evidence of blind generalization."
        ),
        "samples": samples,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sft", type=Path, required=True)
    result.add_argument("--token-manifest", type=Path, required=True,
                        help="Frozen benchmark/training manifest containing the complete audit array")
    result.add_argument("--output", type=Path, required=True, help="New JSON file; existing files are never overwritten")
    result.add_argument("--seed", type=int, default=20261009)
    result.add_argument("--expected-holdout-count", type=int,
                        help="Optional independent count assertion (81 for the audited current SFT)")
    return result


def main(argv: list[str] | None = None) -> dict:
    args = parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.expected_holdout_count is not None:
        _positive_int(args.expected_holdout_count, "expected_holdout_count")
    source_hash = bench.file_digest(args.sft)
    source_metadata_bytes = args.token_manifest.read_bytes()
    source_metadata = json.loads(source_metadata_bytes)
    if not isinstance(source_metadata, dict) or source_metadata.get("sft_sha256") != source_hash:
        raise ValueError("token manifest SFT SHA-256 does not match the provided SFT")
    rows = bench.load_rows(args.sft)
    if bench.file_digest(args.sft) != source_hash:
        raise ValueError("SFT changed during manifest preparation")
    manifest = build_manifest(rows, source_metadata.get("audit"), sft_sha256=source_hash, seed=args.seed)
    if args.expected_holdout_count is not None and manifest["sample_count"] != args.expected_holdout_count:
        raise ValueError("holdout count does not match the independent expected count")
    manifest["provenance"] = {
        "sft_path": str(args.sft.resolve()), "token_manifest_path": str(args.token_manifest.resolve()),
        "token_manifest_sha256": hashlib.sha256(source_metadata_bytes).hexdigest(),
        "script_sha256": bench.file_digest(Path(__file__)),
        "benchmark_helper_sha256": bench.file_digest(_BENCH_PATH),
        "tokenization": {key: source_metadata[key] for key in
                         ("config_sha256", "script_sha256", "helper_sha256", "prompt_format", "versions")
                         if key in source_metadata},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bench.write_json(args.output, manifest)
    print(json.dumps({"output": str(args.output), "sample_count": manifest["sample_count"],
                      "partition_counts": manifest["partition_counts"],
                      "cohort_sha256": manifest["cohort_sha256"]}, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    main()
