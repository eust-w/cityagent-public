#!/usr/bin/env python3
"""Serial, streaming OpenAI-compatible inference benchmark (stdlib only).

Input JSON, explicitly declaring training provenance (not proof of provenance)::

  {"dataset_split": "train", "source": "training report / extraction receipt",
   "samples": [{"id": "chunk-1", "messages": [
       {"role": "user", "content": "An explicit training-derived prompt"}]}]}

Each sample may use ``prompt`` instead of ``messages``. A single sample object
with dataset_split/source is also accepted. No datasets are discovered/read by
this script beyond this explicit JSON file. No input is shortened.

Example (output directory MUST NOT already exist)::

  python experiments/ppu_framework_benchmark.py --samples /tmp/train_probe.json \
    --endpoint http://127.0.0.1:8000/v1/chat/completions \
    --model cityagent-qwen36-27b-epoch-002 --max-tokens 256 \
    --out-dir /tmp/vllm-bf16-probe --label vllm-bf16

Run the same file/arguments against SGLang in a NEW output directory. Explicit
warmups are excluded from aggregate measurements. There is no retry or parallel
request. Stop on the first failed request by default. Other clients must be
excluded operationally; serial execution here cannot make a server exclusive.

Protocol notes:
* chat_template_kwargs.enable_thinking=false is sent by default, but is NOT a
  universal OpenAI field. Select --thinking-field enable_thinking or omit only
  after checking the server/template; omission does not guarantee thinking off.
* --no-stream-usage / --omit-seed are explicit compatibility switches. Missing
  usage means token throughput is unavailable; characters are never tokens.
* --ignore-eos is an opt-in server extension for fixed-length throughput probes.
  finish_reason=length is not treated as a complete extraction or quality pass.
* TTFT measures the first observed nonempty model-output SSE delta, including
  reasoning/tool arguments. SSE can batch multiple tokens. Decode throughput
  is explicitly approximate; no tokenizer or per-token timestamps are invented.
* Timeout is the urllib socket-operation timeout, not a whole-request deadline.
* Requests, raw SSE and generated text are saved locally (can contain report text).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_samples(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = path.read_bytes()
    envelope = json.loads(raw)
    if not isinstance(envelope, dict) or envelope.get("dataset_split") != "train":
        raise ValueError("Input must explicitly declare dataset_split='train'")
    if not isinstance(envelope.get("source"), str) or not envelope["source"].strip():
        raise ValueError("Input must provide a nonempty training source description")
    rows = envelope.get("samples", [envelope])
    if not isinstance(rows, list) or not rows:
        raise ValueError("samples must be a nonempty list")
    samples, ids = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Sample {index} must be an object")
        if row.get("dataset_split", "train") != "train":
            raise ValueError(f"Sample {index} is not declared training data")
        sample_id = str(row.get("id", index))
        if sample_id in ids:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        ids.add(sample_id)
        if ("messages" in row) == ("prompt" in row):
            raise ValueError(f"Sample {sample_id}: use exactly one of prompt/messages")
        messages = row.get("messages")
        if "prompt" in row:
            messages = [{"role": "user", "content": row["prompt"]}]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Sample {sample_id}: messages must be a nonempty list")
        for message in messages:
            if (not isinstance(message, dict)
                    or message.get("role") not in {"system", "user", "assistant", "tool"}
                    or not isinstance(message.get("content"), str)):
                raise ValueError(f"Sample {sample_id}: only text chat messages supported")
        samples.append({"id": sample_id, "messages": messages,
                        "sample_sha256": sha256(row),
                        "messages_sha256": sha256(messages)})
    provenance = {"input_path": str(path.resolve()),
                  "input_file_sha256": hashlib.sha256(raw).hexdigest(),
                  "dataset_split": "train", "source": envelope["source"],
                  "provenance_is_user_declared": True}
    return provenance, samples


def make_payload(args: argparse.Namespace, sample: dict[str, Any]) -> dict[str, Any]:
    payload = {"model": args.model, "messages": sample["messages"],
               "temperature": 0, "max_tokens": args.max_tokens, "stream": True}
    if not args.omit_seed:
        payload["seed"] = args.seed
    if args.stream_usage:
        payload["stream_options"] = {"include_usage": True}
    if args.thinking_field == "chat_template_kwargs":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif args.thinking_field == "enable_thinking":
        payload["enable_thinking"] = False
    if args.ignore_eos:
        payload["ignore_eos"] = True
    return payload


def run_request(args: argparse.Namespace, sample: dict[str, Any],
                output_dir: Path, phase: str, repetition: int) -> dict[str, Any]:
    payload = make_payload(args, sample)
    record: dict[str, Any] = {
        "schema_version": 1, "label": args.label, "phase": phase,
        "repetition": repetition, "sample_id": sample["id"],
        "sample_sha256": sample["sample_sha256"],
        "messages_sha256": sample["messages_sha256"],
        "request_sha256": sha256(payload), "endpoint": args.endpoint,
        "request": payload, "started_at": utc_now(), "success": False,
        "error": None, "usage": None, "finish_reason": None,
        "done_received": False, "output_delta_count": 0,
        "first_output_delta_seconds": None, "last_output_delta_seconds": None,
        "first_content_delta_seconds": None, "content_text": "",
        "reasoning_text": "", "tool_call_deltas": [], "response_model": None,
        "http_status": None, "http_response_headers": {},
    }
    output_dir.mkdir()
    save_json(output_dir / "request.json", payload)
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {args.api_key_env} is not set")
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(args.endpoint, data=canonical_bytes(payload),
                                     headers=headers, method="POST")
    # Disable inherited HTTP proxies: benchmark the explicitly selected endpoint.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.perf_counter()
    try:
        with (output_dir / "response.sse").open("xb") as raw_stream, \
                (output_dir / "events.jsonl").open("x", encoding="utf-8") as events, \
                opener.open(request, timeout=args.timeout) as response:
            record["http_status"] = response.status
            record["http_response_headers"] = dict(response.headers.items())
            if "text/event-stream" not in response.headers.get("Content-Type", "").lower():
                raw_stream.write(response.read())
                raise ValueError("Expected text/event-stream response")
            data_lines: list[str] = []

            def consume_event() -> bool:
                if not data_lines:
                    return False
                elapsed = time.perf_counter() - started
                data = "\n".join(data_lines)
                data_lines.clear()
                if data == "[DONE]":
                    record["done_received"] = True
                    return True
                event = json.loads(data)
                events.write(json.dumps({"elapsed_seconds": elapsed, "event": event},
                                        ensure_ascii=False) + "\n")
                events.flush()
                if not isinstance(event, dict):
                    raise ValueError("SSE data event is not a JSON object")
                if event.get("error"):
                    raise ValueError("Server stream error: " + json.dumps(event["error"]))
                if event.get("model"):
                    record["response_model"] = event["model"]
                if event.get("usage") is not None:
                    record["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise ValueError("Unexpected multiple output choices")
                    if choice.get("finish_reason") is not None:
                        record["finish_reason"] = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if not isinstance(content, str) or not isinstance(reasoning, str):
                        raise ValueError("Unsupported non-text output delta")
                    tool_calls = delta.get("tool_calls") or []
                    if tool_calls:
                        record["tool_call_deltas"].extend(tool_calls)
                    tool_output = any(call.get("function", {}).get("arguments")
                                      for call in tool_calls)
                    if content or reasoning or tool_output:
                        record["output_delta_count"] += 1
                        if record["first_output_delta_seconds"] is None:
                            record["first_output_delta_seconds"] = elapsed
                        record["last_output_delta_seconds"] = elapsed
                    if content and record["first_content_delta_seconds"] is None:
                        record["first_content_delta_seconds"] = elapsed
                    record["content_text"] += content
                    record["reasoning_text"] += reasoning
                return False

            for line in response:
                raw_stream.write(line)
                # SSE is UTF-8; malformed data is retained and fails visibly.
                decoded = line.decode("utf-8").rstrip("\r\n")
                if not decoded:
                    if consume_event():
                        break
                elif decoded.startswith("data:"):
                    data_lines.append(decoded[5:].removeprefix(" "))
            if data_lines:
                consume_event()
            if not record["done_received"] or record["finish_reason"] is None:
                raise ValueError("Incomplete stream: missing [DONE] or finish_reason")
            record["success"] = True
    except urllib.error.HTTPError as error:
        record["http_status"] = error.code
        record["http_response_headers"] = dict(error.headers.items())
        try:
            body = error.read().decode("utf-8", errors="replace")
        finally:
            error.close()
        record["error"] = {"type": type(error).__name__, "message": str(error),
                           "response_body": body}
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        record["end_to_end_seconds"] = time.perf_counter() - started
        record["ended_at"] = utc_now()
    first, last = record["first_output_delta_seconds"], record["last_output_delta_seconds"]
    record["ttft_observed_seconds"] = first
    span = last - first if first is not None and last is not None else None
    record["observed_output_span_seconds"] = span
    record["completion_tokens_per_end_to_end_second"] = None
    record["approx_completion_tokens_per_decode_second"] = None
    usage = record["usage"]
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
        record["completion_tokens_per_end_to_end_second"] = tokens / record["end_to_end_seconds"]
        if span and span > 0 and tokens > 1:
            record["approx_completion_tokens_per_decode_second"] = (tokens - 1) / span
    record["generation_hit_limit"] = record["finish_reason"] == "length"
    record["quality_evaluated"] = False
    save_json(output_dir / "result.json", record)
    return record


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in results if row["phase"] == "measured"]
    successful = [row for row in measured if row["success"]]
    metrics = {}
    for key in ("ttft_observed_seconds", "end_to_end_seconds",
                "completion_tokens_per_end_to_end_second",
                "approx_completion_tokens_per_decode_second"):
        values = [row[key] for row in successful if row[key] is not None]
        metrics[key] = {"count": len(values), "median": statistics.median(values)
                        if values else None, "min": min(values) if values else None,
                        "max": max(values) if values else None}
    return {"measured_requests": len(measured), "successful_requests": len(successful),
            "failed_requests": len(measured) - len(successful),
            "warmup_requests": len(results) - len(measured),
            "generation_hit_limit_count": sum(row["generation_hit_limit"] for row in successful),
            "metrics": metrics, "quality_evaluated": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="Full /v1/chat/completions URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="unspecified")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--omit-seed", action="store_true")
    parser.add_argument("--stream-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--thinking-field", choices=("chat_template_kwargs", "enable_thinking", "omit"),
                        default="chat_template_kwargs")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0, help="Warmup passes over every sample")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--api-key-env", help="Environment variable name; key is never saved")
    args = parser.parse_args()
    url = urllib.parse.urlparse(args.endpoint)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        parser.error("endpoint must be an HTTP(S) URL without embedded credentials")
    if (args.max_tokens < 1 or args.repetitions < 1 or args.warmup < 0 or args.timeout <= 0):
        parser.error("max-tokens, repetitions, timeout must be positive; warmup must be nonnegative")
    if args.api_key_env and not os.environ.get(args.api_key_env):
        parser.error(f"Environment variable {args.api_key_env} is not set")
    provenance, samples = load_samples(args.samples)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    save_json(args.out_dir / "manifest.json", {
        "started_at": utc_now(), "provenance": provenance,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version, "request_execution": "serial, no retries",
        "server_exclusivity_verified": False, "quality_evaluated": False,
        "timing_note": "TTFT is first observed output SSE delta; delta may contain multiple tokens. "
                       "Decode rate is (completion_tokens - 1)/(last_delta - first_delta), "
                       "an estimate; usage may include special tokens not seen as text.",
    })
    results, failed = [], False
    with (args.out_dir / "results.jsonl").open("x", encoding="utf-8") as journal:
        for phase, passes in (("warmup", args.warmup), ("measured", args.repetitions)):
            for repetition in range(passes):
                for sample in samples:
                    row = run_request(args, sample, args.out_dir / f"request-{len(results):04d}",
                                      phase, repetition)
                    results.append(row)
                    journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                    journal.flush()
                    print(json.dumps({key: row[key] for key in (
                        "phase", "sample_id", "success", "finish_reason", "usage",
                        "ttft_observed_seconds", "end_to_end_seconds", "error")}, ensure_ascii=False), flush=True)
                    failed = failed or not row["success"]
                    if failed and not args.continue_on_error:
                        save_json(args.out_dir / "summary.json", summarize(results))
                        return 1
    save_json(args.out_dir / "summary.json", summarize(results))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
