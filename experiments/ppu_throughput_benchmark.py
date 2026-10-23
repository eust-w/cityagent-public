#!/usr/bin/env python3
"""Bounded-concurrency benchmark over an entire frozen training sample list.

  python experiments/ppu_throughput_benchmark.py --samples /runs/frozen16.json \
    --endpoint http://127.0.0.1:8010/v1/chat/completions --model cityagent-epoch002 \
    --concurrency 4 --max-tokens 256 --ignore-eos --out-dir /runs/c4 \
    --label sglang-bf16-c4

Run concurrency 1/2/4/8 separately with identical samples and server settings;
do not run those experiments simultaneously. This script cannot establish GPU
exclusivity or verify server max-running-requests. Optional --server-config-json
attaches a caller-supplied configuration receipt, not independent verification.

Imports the adjacent ppu_framework_benchmark.py without changing it. Each sample
gets its own request/SSE/events/result files. Every pass uses the ENTIRE list.
At most C jobs are in flight; a finished slot immediately admits the next sample.
No retries, truncation, test-set discovery, model loading, or GPU control occurs.

Every sample is considered available at batch start. Per-request HTTP TTFT and
HTTP end-to-end include server queuing; client dispatch wait is separate. Exact
batch-arrival-to-completion latency additionally includes client wait and local
receipt persistence. Batch wall time includes all failures and all samples.

--ignore-eos is opt-in: fixed-length probes require success, valid usage, exactly
max_tokens completion tokens and finish_reason=length. They are not task quality
tests. Normal-EOS mode reports stopped/truncated generations separately and does
not claim JSON or semantic quality. Missing usage is never treated as zero.
Only complete, fully valid batches get primary aggregate token-throughput rates;
partial known-token lower bounds remain explicitly ineligible for comparison.

Warmup passes are explicit (default 0) and excluded from measurement windows.
--timeout is a socket-operation timeout, NOT a batch deadline. The caller must
enforce an outer hard deadline if required; a killed run without summary.json is
unfinished. Preserve its partial receipts, but never rank it as a completed run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any, Callable
import urllib.parse


BASE_PATH = Path(__file__).resolve().with_name("ppu_framework_benchmark.py")
SPEC = importlib.util.spec_from_file_location("ppu_throughput_serial_base", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def valid_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def token_count(row: dict[str, Any], field: str) -> int | None:
    usage = row.get("usage")
    value = usage.get(field) if isinstance(usage, dict) else None
    return value if valid_count(value) else None


def aggregate(values: list[Any]) -> dict[str, Any]:
    numbers = sorted(value for value in values if isinstance(value, (int, float))
                     and not isinstance(value, bool) and math.isfinite(value) and value >= 0)
    return {"count": len(numbers), "median": statistics.median(numbers) if numbers else None,
            "min": min(numbers) if numbers else None, "max": max(numbers) if numbers else None,
            "p95_nearest_rank": numbers[math.ceil(.95 * len(numbers)) - 1] if numbers else None}


def protocol_check(row: dict[str, Any], fixed: bool) -> dict[str, Any]:
    reasons = []
    if row.get("success") is not True:
        reasons.append("request_not_successful")
    for field in ("prompt_tokens", "completion_tokens"):
        if token_count(row, field) is None:
            reasons.append("missing_or_invalid_" + field)
    if fixed:
        request = row.get("request") or {}
        maximum = request.get("max_tokens")
        if request.get("ignore_eos") is not True:
            reasons.append("ignore_eos_not_explicitly_true")
        if not valid_count(maximum) or maximum == 0:
            reasons.append("invalid_max_tokens")
        elif token_count(row, "completion_tokens") != maximum:
            reasons.append("completion_tokens_not_equal_to_max_tokens")
        if row.get("finish_reason") != "length":
            reasons.append("finish_reason_not_length")
    return {"valid": not reasons, "invalid_reasons": reasons}


def summarize_rows(rows: list[dict[str, Any]], wall_seconds: float,
                   expected_count: int, fixed: bool) -> dict[str, Any]:
    successful = [row for row in rows if row.get("success") is True]
    checks = [protocol_check(row, fixed) for row in rows]
    eligible = [row for row, check in zip(rows, checks) if check["valid"]]
    complete = len(rows) == expected_count
    valid = complete and bool(rows) and len(eligible) == expected_count and wall_seconds > 0
    tokens = {}
    for field in ("prompt_tokens", "completion_tokens"):
        known = [token_count(row, field) for row in successful if token_count(row, field) is not None]
        total = sum(known)
        missing = len(successful) - len(known)
        tokens[field] = {
            "known_successful_tokens": total, "successful_requests_with_usage": len(known),
            "successful_requests_missing_usage": missing,
            "aggregate_tokens_per_second": total / wall_seconds if valid else None,
            "observed_successful_tokens_per_second_lower_bound": total / wall_seconds
                if known and wall_seconds > 0 else None,
            "lower_bound_is_eligible_for_comparison": False,
        }
    # Fixed-length-invalid requests never enter the fixed-length latency statistics.
    latency_rows = eligible if fixed else successful
    latencies = {field: aggregate([row.get(field) for row in latency_rows]) for field in (
        "ttft_observed_seconds", "end_to_end_seconds", "client_dispatch_wait_seconds",
        "batch_arrival_to_completion_seconds")}
    return {"expected_request_count": expected_count, "finished_request_count": len(rows),
            "success_count": len(successful), "failure_count": len(rows) - len(successful),
            "unrecorded_request_count": max(0, expected_count - len(rows)),
            "batch_complete": complete, "valid_for_comparison": valid,
            "protocol_valid_count": len(eligible), "protocol_invalid_count": len(rows) - len(eligible),
            "wall_seconds": wall_seconds, "tokens": tokens, "latencies": latencies,
            "all_request_arrival_to_completion_seconds": aggregate(
                [row.get("batch_arrival_to_completion_seconds") for row in rows]),
            "finish_reasons": dict(Counter(str(row.get("finish_reason")) for row in rows)),
            "successful_stop_count": sum(row.get("finish_reason") == "stop" for row in successful),
            "successful_length_count": sum(row.get("finish_reason") == "length" for row in successful),
            "successful_requests_per_second": len(successful) / wall_seconds if wall_seconds > 0 else None,
            "probe_type": "fixed_length_probe" if fixed else "normal_eos_generation",
            "quality_evaluated": False}


def invalidate_comparison(summary: dict[str, Any], reason: str) -> None:
    summary["valid_for_comparison"] = False
    summary.setdefault("additional_invalid_reasons", []).append(reason)
    for tokens in summary["tokens"].values():
        tokens["aggregate_tokens_per_second"] = None


def run_batch(args: argparse.Namespace, samples: list[dict[str, Any]], directory: Path,
              phase: str, repetition: int,
              on_result: Callable[[dict[str, Any]], None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory.mkdir()
    arrival_at, start = base.utc_now(), time.perf_counter()
    lock = threading.Lock()
    active, peak = 0, 0

    def execute(index: int) -> dict[str, Any]:
        nonlocal active, peak
        sample = samples[index]
        destination = directory / f"sample-{index:04d}"
        dispatched_at, dispatch = base.utc_now(), time.perf_counter()
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            try:
                row = base.run_request(args, sample, destination, phase, repetition)
            except Exception as error:
                # A local persistence/preflight failure is still a completed failed job.
                row = {"phase": phase, "repetition": repetition, "sample_id": sample["id"],
                       "sample_sha256": sample["sample_sha256"],
                       "messages_sha256": sample["messages_sha256"],
                       "request": base.make_payload(args, sample), "success": False,
                       "error": {"type": type(error).__name__, "message": str(error)},
                       "usage": None, "finish_reason": None, "quality_evaluated": False}
            completed_at, completed = base.utc_now(), time.perf_counter()
            row.update({"batch_id": directory.name, "sample_index": index,
                        "request_directory": str(destination.resolve()),
                        "batch_arrival_at": arrival_at, "worker_dispatched_at": dispatched_at,
                        "worker_completed_at": completed_at,
                        "client_dispatch_wait_seconds": dispatch - start,
                        "worker_lifecycle_seconds": completed - dispatch,
                        "batch_arrival_to_completion_seconds": completed - start,
                        "measurement_protocol": protocol_check(row, args.ignore_eos)})
            try:
                destination.mkdir(exist_ok=True)
                base.save_json(destination / "throughput-result.json", row)
            except OSError as error:
                row["success"] = False
                row["error"] = {"type": type(error).__name__, "message": str(error),
                                "stage": "throughput_receipt_persistence"}
                row["measurement_protocol"] = protocol_check(row, args.ignore_eos)
            return row
        finally:
            with lock:
                active -= 1

    records, next_index = [], 0
    with ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="ppu-benchmark") as pool:
        pending = set()
        while next_index < min(args.concurrency, len(samples)):
            pending.add(pool.submit(execute, next_index))
            next_index += 1
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            ready = [future.result() for future in done]
            # Refill before journal/console I/O to avoid holding available workers idle.
            for _ in ready:
                if next_index < len(samples):
                    pending.add(pool.submit(execute, next_index))
                    next_index += 1
            for row in ready:
                records.append(row)
                on_result(row)
    end, ended_at = time.perf_counter(), base.utc_now()
    records.sort(key=lambda row: row["sample_index"])
    closure_valid = ([row["sample_id"] for row in records] == [sample["id"] for sample in samples]
                     and [row["sample_index"] for row in records] == list(range(len(samples))))
    summary = summarize_rows(records, end - start, len(samples), args.ignore_eos)
    if not closure_valid:
        invalidate_comparison(summary, "sample_identity_closure_failed")
    result = {"batch_id": directory.name, "phase": phase, "repetition": repetition,
              "started_at": arrival_at, "ended_at": ended_at,
              "monotonic_start_seconds": start, "monotonic_end_seconds": end,
              "concurrency": args.concurrency, "peak_active_client_workers": peak,
              "peak_observed_client_inflight": peak,
              "expected_sample_ids": [sample["id"] for sample in samples],
              "finished_sample_ids": [row["sample_id"] for row in records],
              "sample_id_closure_valid": closure_valid, "summary": summary}
    base.save_json(directory / "batch.json", result)
    return result, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="Full /v1/chat/completions URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="unspecified")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--warmup", type=int, default=0, help="Full-list warmup passes, excluded from measured wall time")
    parser.add_argument("--repetitions", type=int, default=1, help="Measured full-list batches")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--omit-seed", action="store_true")
    parser.add_argument("--stream-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--thinking-field", choices=("chat_template_kwargs", "enable_thinking", "omit"), default="chat_template_kwargs")
    parser.add_argument("--timeout", type=float, default=1800, help="Socket-operation timeout, not a batch deadline")
    parser.add_argument("--api-key-env")
    parser.add_argument("--server-config-json", type=Path, help="Optional existing caller-supplied server configuration receipt")
    args = parser.parse_args()
    url = urllib.parse.urlparse(args.endpoint)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        parser.error("endpoint must be an HTTP(S) URL without embedded credentials")
    if args.max_tokens < 1 or args.warmup < 0 or args.repetitions < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("max-tokens/repetitions/timeout must be positive; warmup must be nonnegative")
    if args.api_key_env and not os.environ.get(args.api_key_env):
        parser.error(f"Environment variable {args.api_key_env} is not set")
    provenance, samples = base.load_samples(args.samples)
    server_receipt = None
    if args.server_config_json:
        raw = args.server_config_json.read_bytes()
        server_receipt = {"path": str(args.server_config_json.resolve()),
                          "sha256": hashlib.sha256(raw).hexdigest(), "content": json.loads(raw),
                          "independently_verified": False}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frozen = {"dataset_split": "train", "source": provenance["source"],
              "samples": [{"id": sample["id"], "messages": sample["messages"]} for sample in samples]}
    base.save_json(args.out_dir / "frozen-samples.json", frozen)
    base.save_json(args.out_dir / "manifest.json", {
        "schema_version": 1, "started_at": base.utc_now(), "provenance": provenance,
        "frozen_samples_sha256": base.sha256(frozen), "samples_per_batch": len(samples),
        "sample_ids_in_order": [sample["id"] for sample in samples],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_script_sha256": hashlib.sha256(BASE_PATH.read_bytes()).hexdigest(),
        "server_configuration_receipt": server_receipt, "server_exclusivity_verified": False,
        "request_execution": "bounded concurrent full-list batches; no retries",
        "batch_arrival_policy": "all frozen samples available at batch start",
        "ttft_note": "HTTP TTFT includes service queue; client dispatch wait is separate. "
                     "Arrival-to-completion includes both waits and local per-request persistence.",
        "timeout_note": "Socket-operation timeout only; caller supplies any hard batch deadline.",
        "completion_note": "No final summary.json means unfinished; partial rates cannot rank candidates.",
        "quality_evaluated": False,
    })
    batches, measured_rows = [], []
    measurement_start = measurement_end = None
    measurement_started_at = measurement_ended_at = None
    with (args.out_dir / "results.jsonl").open("x", encoding="utf-8") as journal:
        def record(row: dict[str, Any]) -> None:
            journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            print(json.dumps({key: row.get(key) for key in (
                "batch_id", "sample_index", "sample_id", "phase", "success", "finish_reason",
                "usage", "ttft_observed_seconds", "batch_arrival_to_completion_seconds",
                "measurement_protocol", "error")}, ensure_ascii=False), flush=True)

        for phase, passes in (("warmup", args.warmup), ("measured", args.repetitions)):
            for repetition in range(passes):
                if phase == "measured" and measurement_start is None:
                    measurement_started_at, measurement_start = base.utc_now(), time.perf_counter()
                batch, rows = run_batch(args, samples, args.out_dir / f"batch-{len(batches):04d}",
                                        phase, repetition, record)
                batches.append(batch)
                if phase == "measured":
                    measured_rows.extend(rows)
                    measurement_end, measurement_ended_at = time.perf_counter(), base.utc_now()
    measured_wall = measurement_end - measurement_start
    measured_summary = summarize_rows(measured_rows, measured_wall,
                                      len(samples) * args.repetitions, args.ignore_eos)
    warmups = [batch for batch in batches if batch["phase"] == "warmup"]
    invalid_warmups = sum(not batch["summary"]["valid_for_comparison"] for batch in warmups)
    if invalid_warmups:
        invalidate_comparison(measured_summary, "invalid_warmup_batch")
    if any(not batch["sample_id_closure_valid"] for batch in batches if batch["phase"] == "measured"):
        invalidate_comparison(measured_summary, "sample_identity_closure_failed")
    report = {"schema_version": 1, "label": args.label, "concurrency": args.concurrency,
              "peak_observed_client_inflight": max(batch["peak_observed_client_inflight"]
                                                    for batch in batches if batch["phase"] == "measured"),
              "measurement_started_at": measurement_started_at, "measurement_ended_at": measurement_ended_at,
              "measurement_window_seconds": measured_wall,
              "measured_batches": args.repetitions, "warmup_batches": len(warmups),
              "warmup_failure_count": sum(batch["summary"]["failure_count"] for batch in warmups),
              "warmup_invalid_batch_count": invalid_warmups,
              "summary": measured_summary, "batches": batches,
              "comparison_status": "valid" if measured_summary["valid_for_comparison"] else "invalid_or_partial",
              "quality_evaluated": False}
    base.save_json(args.out_dir / "summary.json", report)
    print(json.dumps({"comparison_status": report["comparison_status"], "summary": measured_summary},
                     ensure_ascii=False, allow_nan=False), flush=True)
    # Missing usage is an invalid throughput measurement, even with successful HTTP.
    return 0 if measured_summary["valid_for_comparison"] and not report["warmup_failure_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
