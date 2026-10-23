#!/usr/bin/env python3
"""Offline strict comparison of complete 16 x 256-token PPU throughput rounds.

  python experiments/summarize_ppu_throughput.py /runs/sg-c1 /runs/sg-c2 \
    /runs/sg-c2-repeat /runs/vllm-c1 --out-json /reports/throughput.json \
    --out-md /reports/throughput.md

Uses existing manifest.json, frozen-samples.json, results.jsonl and summary.json.
No inference, network access, GPU operations, or input mutation. Output files
must be new. Directories with max_tokens=32 are excluded as separate warmup runs;
embedded warmup batches are excluded too. Each measured batch remains one round.
Repeat groups use the EXACT label plus concurrency: use the same label on repeat
runs. No automatic framework-name guessing or suffix stripping occurs.

Before comparing rounds, validates frozen input hash/order, all 16 unique IDs and
indices, full request hashes INCLUDING model, each ID's messages/request/options,
and same-ID prompt-token counts. Every completion must be successful, finished
with length and exactly 256 tokens (4096 total). Failures/partial or incompatible
rounds remain visible, never receive a primary throughput score, and make the
overall cross-round comparison invalid (exit 1). Whole-round wall time is never
replaced with the sum of per-request latencies. This is fixed-output screening,
not a complete-report quality evaluation or F1.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


EXPECTED_SAMPLES = 16
EXPECTED_OUTPUT = 256


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def valid_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stats(values: list[Any]) -> dict[str, Any]:
    numbers = sorted(value for value in values if valid_number(value) and value >= 0)
    return {"count": len(numbers), "median": statistics.median(numbers) if numbers else None,
            "min": min(numbers) if numbers else None, "max": max(numbers) if numbers else None}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_round(directory: Path, manifest: dict[str, Any], frozen: dict[str, Any],
                  batch: dict[str, Any], rows: list[dict[str, Any]], inherited_errors: list[str]) -> dict[str, Any]:
    errors = list(inherited_errors)
    arguments = manifest.get("arguments", {})
    sample_rows = frozen.get("samples", [])
    ids = [sample.get("id") for sample in sample_rows]
    frozen_by_id = {sample.get("id"): sample for sample in sample_rows}
    if len(ids) != EXPECTED_SAMPLES or len(set(ids)) != EXPECTED_SAMPLES:
        errors.append("frozen_input_must_have_16_unique_sample_ids")
    if ids != manifest.get("sample_ids_in_order"):
        errors.append("manifest_and_frozen_sample_order_differ")
    if len(rows) != EXPECTED_SAMPLES:
        errors.append(f"round_has_{len(rows)}_receipts_expected_16")
    indices = [row.get("sample_index") for row in rows]
    indices_valid = all(valid_count(index) for index in indices)
    ordered = sorted(rows, key=lambda row: row["sample_index"]) if indices_valid else rows
    closure = bool(indices_valid and sorted(indices) == list(range(EXPECTED_SAMPLES))
                   and [row.get("sample_id") for row in ordered] == ids
                   and len(set(row.get("sample_id") for row in rows)) == EXPECTED_SAMPLES)
    if not closure:
        errors.append("sample_id_or_index_closure_failed")
    wall = batch.get("summary", {}).get("wall_seconds")
    if not valid_number(wall) or wall <= 0:
        errors.append("missing_or_invalid_whole_round_wall_seconds")
    start, end = batch.get("monotonic_start_seconds"), batch.get("monotonic_end_seconds")
    if (not valid_number(start) or not valid_number(end) or end <= start
            or not valid_number(wall) or not math.isclose(end - start, wall, rel_tol=1e-9, abs_tol=1e-6)):
        errors.append("whole_round_monotonic_clock_mismatch")
    concurrency = arguments.get("concurrency")
    if concurrency not in (1, 2, 4, 8) or isinstance(concurrency, bool) or batch.get("concurrency") != concurrency:
        errors.append("batch_and_manifest_concurrency_mismatch")
    peak = batch.get("peak_observed_client_inflight", batch.get("peak_active_client_workers"))
    if not valid_count(peak) or not valid_count(concurrency) or not 1 <= peak <= concurrency:
        errors.append("invalid_client_inflight_peak")
    per_id, prompt_total, output_total = [], 0, 0
    missing_usage, failures = 0, 0
    for row in ordered:
        sample_id = row.get("sample_id")
        prefix = f"sample={sample_id}:"
        request = row.get("request") or {}
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        prompt_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if row.get("success") is not True:
            failures += 1
            errors.append(prefix + "request_failed")
        if row.get("done_received") is not True:
            errors.append(prefix + "missing_stream_done")
        if row.get("finish_reason") != "length":
            errors.append(prefix + "finish_reason_not_length")
        if (request.get("ignore_eos") is not True or request.get("max_tokens") != EXPECTED_OUTPUT
                or isinstance(request.get("max_tokens"), bool)):
            errors.append(prefix + "not_fixed_256_request")
        if request.get("model") != arguments.get("model") or not request.get("model"):
            errors.append(prefix + "model_alias_mismatch")
        request_sha, messages_sha = digest(request), digest(request.get("messages"))
        if request_sha != row.get("request_sha256"):
            errors.append(prefix + "saved_request_hash_mismatch")
        if messages_sha != row.get("messages_sha256"):
            errors.append(prefix + "saved_messages_hash_mismatch")
        source = frozen_by_id.get(sample_id)
        if not source or request.get("messages") != source.get("messages"):
            errors.append(prefix + "request_not_frozen_sample_messages")
        if not valid_count(prompt_tokens) or not valid_count(output_tokens):
            missing_usage += 1
            errors.append(prefix + "missing_or_invalid_usage")
        if valid_count(prompt_tokens):
            prompt_total += prompt_tokens
        if valid_count(output_tokens):
            output_total += output_tokens
        if not valid_count(output_tokens) or output_tokens != EXPECTED_OUTPUT:
            errors.append(prefix + "completion_tokens_not_256")
        lifetime = row.get("batch_arrival_to_completion_seconds")
        if (not valid_number(lifetime) or lifetime < 0 or not valid_number(wall)
                or lifetime > wall + 1e-6):
            errors.append(prefix + "arrival_to_completion_outside_round_wall")
        per_id.append({"sample_id": sample_id, "messages_sha256": messages_sha,
                       "request_sha256": request_sha, "prompt_tokens": prompt_tokens})
    if output_total != EXPECTED_SAMPLES * EXPECTED_OUTPUT:
        errors.append(f"round_completion_tokens_{output_total}_expected_4096")
    signature = {"frozen_samples_sha256": digest(frozen), "sample_ids_in_order": ids, "per_id": per_id}
    valid = not errors
    return {"directory": str(directory.resolve()), "round_id": batch.get("batch_id"),
            "label": arguments.get("label", directory.name), "concurrency": concurrency,
            "started_at": batch.get("started_at"), "ended_at": batch.get("ended_at"),
            "whole_round_wall_seconds": wall, "request_count": len(rows), "failure_count": failures,
            "missing_usage_count": missing_usage, "sample_id_closure_valid": closure,
            "completion_tokens": output_total, "prompt_tokens": prompt_total,
            "peak_observed_client_inflight": peak,
            "finish_reasons": dict(Counter(str(row.get("finish_reason")) for row in rows)),
            "valid_for_comparison": valid, "invalid_reasons": errors,
            "aggregate_completion_tokens_per_second": output_total / wall if valid else None,
            "aggregate_prompt_tokens_per_second": prompt_total / wall if valid else None,
            "latencies": {field: stats([row.get(field) for row in rows]) for field in (
                "ttft_observed_seconds", "end_to_end_seconds", "client_dispatch_wait_seconds",
                "batch_arrival_to_completion_seconds")},
            "comparison_signature": signature, "comparison_signature_sha256": digest(signature),
            "request_models": sorted({str(row.get("request", {}).get("model")) for row in rows}),
            "measurement_script_sha256": manifest.get("script_sha256"),
            "measurement_base_script_sha256": manifest.get("base_script_sha256"),
            "server_configuration_receipt": manifest.get("server_configuration_receipt"),
            "quality_evaluated": False}


def inspect_directory(directory: Path) -> dict[str, Any]:
    result = {"directory": str(directory.resolve()), "excluded": False, "errors": [], "rounds": []}
    try:
        manifest = load_json(directory / "manifest.json")
        arguments = manifest.get("arguments", {})
        if arguments.get("max_tokens") == 32:
            result.update({"excluded": True, "exclusion_reason": "32_token_warmup_directory"})
            return result
        errors = []
        if arguments.get("max_tokens") != EXPECTED_OUTPUT or arguments.get("ignore_eos") is not True:
            errors.append("directory_is_not_fixed_256_probe")
        frozen = load_json(directory / "frozen-samples.json")
        if frozen.get("dataset_split") != "train" or not frozen.get("source"):
            errors.append("missing_explicit_training_provenance")
        if digest(frozen) != manifest.get("frozen_samples_sha256"):
            errors.append("frozen_sample_hash_mismatch")
        report = load_json(directory / "summary.json")
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line.strip()]
        batches = [batch for batch in report.get("batches", []) if batch.get("phase") == "measured"]
        measured = [row for row in rows if row.get("phase") == "measured"]
        if len(batches) != arguments.get("repetitions") or not batches:
            errors.append("measured_batch_count_mismatch_or_no_complete_summary")
        batch_ids = [batch.get("batch_id") for batch in batches]
        if len(set(batch_ids)) != len(batch_ids):
            errors.append("duplicate_measured_batch_ids")
        if any(row.get("batch_id") not in batch_ids for row in measured):
            errors.append("measured_receipts_outside_declared_batches")
        for batch in report.get("batches", []):
            if batch.get("phase") == "warmup" and not batch.get("summary", {}).get("valid_for_comparison"):
                errors.append("invalid_embedded_warmup_batch")
        result["rounds"] = [inspect_round(directory, manifest, frozen, batch,
                               [row for row in measured if row.get("batch_id") == batch.get("batch_id")], errors)
                            for batch in batches]
        result["errors"] = errors
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
        result["errors"].append(f"unreadable_or_incomplete_artifacts:{type(error).__name__}:{error}")
    return result


def compare(directories: list[Path]) -> dict[str, Any]:
    inspected = [inspect_directory(directory) for directory in directories]
    rounds = [row for run in inspected for row in run["rounds"]]
    reference = next((row for row in rounds if row["valid_for_comparison"]), None)
    if reference:
        for row in rounds:
            if row["valid_for_comparison"] and row["comparison_signature_sha256"] != reference["comparison_signature_sha256"]:
                row["valid_for_comparison"] = False
                row["invalid_reasons"].append("cross_round_frozen_order_request_model_options_or_prompt_tokens_mismatch")
                row["aggregate_completion_tokens_per_second"] = None
                row["aggregate_prompt_tokens_per_second"] = None
    valid = bool(rounds) and all(row["valid_for_comparison"] for row in rounds) and not any(
        run["errors"] for run in inspected if not run["excluded"])
    groups = []
    for label, concurrency in sorted({(row["label"], row["concurrency"]) for row in rounds}, key=str):
        members = [row for row in rounds if row["label"] == label and row["concurrency"] == concurrency]
        accepted = [row for row in members if row["valid_for_comparison"]]
        groups.append({"label": label, "concurrency": concurrency, "round_count": len(members),
            "valid_round_count": len(accepted), "invalid_round_count": len(members) - len(accepted),
            "whole_round_wall_seconds": stats([row["whole_round_wall_seconds"] for row in accepted]),
            "aggregate_completion_tokens_per_second": stats([row["aggregate_completion_tokens_per_second"] for row in accepted]),
            "aggregate_prompt_tokens_per_second": stats([row["aggregate_prompt_tokens_per_second"] for row in accepted]),
            "per_round_median_ttft_seconds": stats([row["latencies"]["ttft_observed_seconds"]["median"] for row in accepted]),
            "per_round_median_http_latency_seconds": stats([row["latencies"]["end_to_end_seconds"]["median"] for row in accepted]),
        })
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "probe_description": "Fixed 256-token generation over 16 training samples; not complete-report quality",
            "cross_round_comparison_valid": valid, "ranking_permitted": valid,
            "reference_round": {"directory": reference["directory"], "round_id": reference["round_id"]} if reference else None,
            "reference_signature_sha256": reference["comparison_signature_sha256"] if reference else None,
            "runs": inspected, "repeat_groups": groups, "quality_evaluated": False}


def markdown(report: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    def number(value: Any) -> str:
        return f"{value:.3f}" if valid_number(value) else "—"

    def interval(value: dict[str, Any]) -> str:
        return f'{number(value["median"])} [{number(value["min"])}, {number(value["max"])}]' if value["count"] else "—"

    lines = ["PPU 并发吞吐复核", "", "这是 16 份训练样本、每份固定生成 256 tokens 的筛选，不是完整报告质量评测。",
             f'跨轮比较：{"通过" if report["cross_round_comparison_valid"] else "拒绝（存在不完整或不可比数据）"}。', "",
             "| label | 客户端并发上限 | 批次 | ID闭包 | 失败 | 完整wall秒 | 输出tokens | 总输出tokens/s | TTFT中位秒 | HTTP延迟中位秒 | 有效 |",
             "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for run in report["runs"]:
        for row in run["rounds"]:
            lines.append("| " + " | ".join(map(cell, [row["label"], row["concurrency"],
                Path(row["directory"]).name + "/" + str(row["round_id"]),
                "16/16" if row["sample_id_closure_valid"] else f'{row["request_count"]}/16 异常',
                row["failure_count"], number(row["whole_round_wall_seconds"]), row["completion_tokens"],
                number(row["aggregate_completion_tokens_per_second"]),
                number(row["latencies"]["ttft_observed_seconds"]["median"]),
                number(row["latencies"]["end_to_end_seconds"]["median"]), row["valid_for_comparison"]])) + " |")
    lines.extend(["", "重复轮汇总：先计算每轮指标，再取中位数 [最小值, 最大值]，不把重复轮拼成一轮。", "",
        "| label | 客户端并发上限 | 有效轮/总轮 | 每轮wall秒 | 每轮总输出tokens/s | 每轮TTFT中位秒 |",
        "|---|---:|---:|---:|---:|---:|"])
    for group in report["repeat_groups"]:
        lines.append("| " + " | ".join(map(cell, [group["label"], group["concurrency"],
            f'{group["valid_round_count"]}/{group["round_count"]}', interval(group["whole_round_wall_seconds"]),
            interval(group["aggregate_completion_tokens_per_second"]), interval(group["per_round_median_ttft_seconds"])])) + " |")
    for run in report["runs"]:
        if run["excluded"]:
            lines.extend(["", f'排除 {cell(run["directory"])}：{run["exclusion_reason"]}。'])
        if run["errors"]:
            lines.extend(["", f'{cell(run["directory"])}：' + "; ".join(run["errors"])])
        for row in run["rounds"]:
            if row["invalid_reasons"]:
                lines.extend(["", f'{cell(row["directory"])}/{cell(row["round_id"])}：' + "; ".join(row["invalid_reasons"])])
    lines.extend(["", "服务有效配置、权重一致性和设备独占仍须结合 controller 记录核验；脚本不操作或验证远端服务。", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()
    if len({path.resolve() for path in args.run_dirs}) != len(args.run_dirs):
        parser.error("Duplicate run directories would double-count a round")
    for path in (args.out_json, args.out_md):
        if path and path.exists():
            parser.error(f"Refusing to overwrite {path}")
    report = compare(args.run_dirs)
    for path, content in ((args.out_json, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                          (args.out_md, markdown(report))):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
    if not args.out_json and not args.out_md:
        print(markdown(report))
    return 0 if report["cross_round_comparison_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
