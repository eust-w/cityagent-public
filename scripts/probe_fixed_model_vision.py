#!/usr/bin/env python3
"""Bounded synthetic image probe against an explicitly fixed model service.

This is capability evidence only, never report-image evidence or RSI acceptance.
It owns the existing cloud budget lease; do not wrap it in another cloud trial.
The native-LoRA identity and cumulative budget ledger must be configured
explicitly. The manifest-locked policy binds an authorized baseline; no model
fallback exists.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import signal
import sqlite3
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.backends import ModelManifest
from bridge_evidence.contest.meta_optimization import CloudBudget
from bridge_evidence.contest.model_identity import load_identity
from bridge_evidence.contest.model_json import parse_json_object
from bridge_evidence.contest.run_state import atomic_json, file_digest, json_digest
from bridge_evidence.contest.timing import TimingLedger

WORKSPACE = Path("/workspace")
ORIGINAL_BUDGET = Path(os.environ.get(
    "CITYAGENT_BUDGET_LEDGER", "/workspace/__cityagent_unconfigured_budget__/budget.sqlite"
))
FIXED_ALIAS = os.environ.get("CITYAGENT_FIXED_MODEL_ALIAS", "__cityagent_unconfigured_model__")
FIXED_ADAPTER_SHA256 = os.environ.get("CITYAGENT_FIXED_ADAPTER_SHA256", "")
MAX_RESPONSE_BYTES = 128 * 1024
PROMPT = (
    "Look at the attached image. Identify the fill color of the rectangle on "
    "the left and the fill color of the rectangle on the right. "
    'Return exactly one JSON object with keys "left" and "right", whose values '
    "are lowercase English color names. Do not include explanations or markdown."
)


def synthetic_png(*, swapped=False):
    """Two plain colored rectangles; expected values never enter the request text."""
    width, height = 320, 160
    colors = [(255, 0, 0), (0, 0, 255)]
    if swapped:
        colors.reverse()
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            color = (255, 255, 255)
            if 32 <= y < 128:
                if 32 <= x < 128:
                    color = colors[0]
                elif 192 <= x < 288:
                    color = colors[1]
            row.extend(color)
        rows.append(bytes(row))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def _workspace_path(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(WORKSPACE.resolve()) or resolved == WORKSPACE.resolve():
        raise ValueError("official /workspace descendant required")
    return resolved


def validate_inputs(args):
    for path in (args.model_manifest, args.output_dir, args.budget_ledger):
        _workspace_path(path)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("fresh output directory required")
    if args.budget_ledger.resolve() != ORIGINAL_BUDGET.resolve():
        raise ValueError("the original cumulative budget ledger is required")
    if not args.budget_ledger.is_file():
        raise ValueError("existing budget ledger required; refusing to initialize a budget")
    with sqlite3.connect(args.budget_ledger.resolve().as_uri() + "?mode=ro", uri=True) as db:
        columns = [row[1] for row in db.execute("PRAGMA table_info(jobs)")]
        if columns != ["id", "phase", "start", "seconds", "status", "cap"]:
            raise ValueError("invalid existing cloud budget ledger")
    url = urlsplit(args.api_url)
    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "::1")
            or url.username or url.password or url.query or url.fragment
            or url.path.rstrip("/") != "/v1"):
        raise ValueError("numeric loopback /v1 endpoint required")
    if not 1 <= args.limit_seconds <= 900 or not 0 < args.request_timeout <= 180:
        raise ValueError("probe limit must be 1..900 seconds and request timeout at most 180")
    receipt = args.prepared_identity
    if bool(receipt) != bool(args.prepared_identity_sha256):
        raise ValueError("prepared identity requires its explicitly pinned SHA256")
    if receipt:
        _workspace_path(receipt)
        if not receipt.is_file() or len(args.prepared_identity_sha256) != 64:
            raise ValueError("invalid prepared identity receipt")
    # Structural validation precedes the lease; hashing actual model bytes happens
    # inside the metered interval below.
    model = ModelManifest.load(args.model_manifest)
    policy = getattr(args, "model_policy", "fixed-native-lora")
    if policy == "fixed-native-lora":
        if (model.served_model != FIXED_ALIAS or model.adapter is None or model.merged
                or model.adapter_sha256 != FIXED_ADAPTER_SHA256
                or Path(model.base).name != "Qwen3.6-27B"
                or Path(model.tokenizer).resolve() != Path(model.base).resolve()):
            raise ValueError("fixed Qwen3.6-27B native-LoRA manifest required; no model fallback")
    elif policy == "manifest-locked":
        if (not isinstance(model.served_model, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}", model.served_model)):
            raise ValueError("manifest-locked probe requires an explicit served model alias")
        tokenizer = Path(model.tokenizer)
        tokenizer_config = tokenizer / "tokenizer_config.json"
        if (not tokenizer_config.is_file()
                or not any((tokenizer / name).is_file() for name in
                           ("tokenizer.json", "tokenizer.model", "spiece.model"))):
            raise ValueError("manifest-locked probe requires bound tokenizer config and artifacts")
        roots = {Path(path) for path in (model.base, model.tokenizer, model.merged) if path}
        processors = [root / name for root in roots
                      for name in ("preprocessor_config.json", "processor_config.json")
                      if (root / name).is_file()]
        if not processors:
            raise ValueError("manifest-locked probe requires bound processor config")
        for path in (tokenizer_config, *processors):
            if not isinstance(json.loads(path.read_text()), dict):
                raise ValueError("tokenizer and processor configs must be JSON objects")
    else:
        raise ValueError("unknown model policy")
    return model


@contextmanager
def deadline(seconds):
    """Real wall-clock bound, including slow streaming responses and identity I/O."""
    old_handler = signal.getsignal(signal.SIGALRM)
    previous = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(signum, frame):
        raise TimeoutError("fixed-model vision probe deadline exceeded")

    signal.signal(signal.SIGALRM, expired)
    delay = min(seconds, previous[0]) if previous[0] else seconds
    signal.setitimer(signal.ITIMER_REAL, delay)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        remaining = previous[0] - (time.monotonic() - started)
        if previous[0] and remaining > 0:
            signal.setitimer(signal.ITIMER_REAL, remaining, previous[1])


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("inference endpoint redirects are forbidden")


class BackendHTTPError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(f"loopback backend returned HTTP {status}")
        self.status, self.body = status, body


def request_json(url, *, timeout, payload=None, call_id=None):
    headers = {"Content-Type": "application/json"}
    if call_id:
        headers["X-Request-ID"] = call_id
    api_key = os.environ.get("CITYAGENT_LOCAL_API_KEY")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
    )
    # Explicitly ignore proxy environment variables, even for the loopback request.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with deadline(timeout):
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            with error:
                raw = error.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("backend error exceeds bounded receipt size") from error
            raise BackendHTTPError(error.code, raw.decode("utf-8", errors="replace")) from error
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("backend response exceeds bounded receipt size")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("backend response must be an object")
    return value


def checked_response(value, seen_request_ids, *, expected_alias=None):
    if value.get("model") != (expected_alias or FIXED_ALIAS):
        raise ValueError("backend returned a different model alias")
    request_id = value.get("id")
    if (not isinstance(request_id, str) or not request_id
            or request_id in seen_request_ids):
        raise ValueError("unique backend request ID required")
    usage = value.get("usage")
    if (not isinstance(usage, dict)
            or any(type(usage.get(key)) is not int or usage[key] <= 0
                   for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
        raise ValueError("nonempty actual usage required")
    choices = value.get("choices")
    if (not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict) or choices[0].get("index") != 0
            or choices[0].get("finish_reason") != "stop"):
        raise ValueError("one complete nontruncated response required")
    message = choices[0].get("message")
    if (not isinstance(message, dict) or message.get("role") != "assistant"
            or not isinstance(message.get("content"), str)):
        raise ValueError("actual assistant content required")
    seen_request_ids.add(request_id)
    return message["content"]


def run_probe(args):
    model = validate_inputs(args)
    budget = CloudBudget(args.budget_ledger)  # Existing ledger only, never recover/reset.
    job = budget.start("fixed-model-vision-capability", args.limit_seconds)
    started = time.monotonic()
    status = "failed"
    created_output = False
    report = {
        "schema_version": 1, "protocol": "fixed-model-vision-capability-v1",
        "purpose": "synthetic capability diagnostic only", "budget_job_id": job,
        "formal_release_evidence": False, "production_multimodal_verified": False,
        "rsi_enabled": False, "model_or_lora_modified": False,
        "passed": False, "calls": [], "maximum_model_requests": 2,
        "endpoint": args.api_url, "seed": 17, "temperature": 0.0, "max_tokens": 256,
        "served_model": model.served_model,
        "model_policy": getattr(args, "model_policy", "fixed-native-lora"),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "limits": {"wall_seconds": args.limit_seconds, "request_seconds": args.request_timeout},
        "limitations": [
            "Synthetic geometry does not establish report-image transcription quality.",
            "Local model hashes and served alias are recorded; server weights are not introspected.",
            "This receipt cannot promote RSI or resolve report-image source coverage.",
        ],
    }
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        created_output = True
        report["model_manifest_sha256"] = file_digest(args.model_manifest)
        report["probe_source_sha256"] = file_digest(Path(__file__))
        report["response_parser_sha256"] = file_digest(
            ROOT / "src/bridge_evidence/contest/model_json.py"
        )
        ledger = TimingLedger(args.output_dir / "timing.sqlite")
        with deadline(args.limit_seconds):
            identity = (
                load_identity(model, args.prepared_identity,
                              expected_sha256=args.prepared_identity_sha256)
                if args.prepared_identity else model.identity()
            )
            report["model_identity"] = identity
            report["model_identity_sha256"] = json_digest(identity)
            models = request_json(args.api_url.rstrip("/") + "/models", timeout=min(30, args.request_timeout))
            report["models_response"] = models
            if model.served_model not in [item.get("id") for item in models.get("data", [])
                                          if isinstance(item, dict)]:
                raise ValueError("configured model alias is not served")
            seen = set()
            for index in range(2):
                pixels = synthetic_png(swapped=bool(index))
                expected = {"left": "red", "right": "blue"} if index == 0 else {
                    "left": "blue", "right": "red",
                }
                (args.output_dir / f"synthetic-{index}.png").write_bytes(pixels)
                payload = {
                    "model": model.served_model, "temperature": 0.0, "seed": 17,
                    "max_tokens": 256, "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(pixels).decode(),
                        }},
                    ]}],
                }
                call_id = uuid4().hex
                record = {"call_id": call_id, "image_sha256": hashlib.sha256(pixels).hexdigest(),
                          "request_sha256": json_digest(payload), "expected": expected,
                          "capability_match": False}
                report["calls"].append(record)
                ledger.claim(call_id, role="synthetic-vision-probe")
                call_started = time.monotonic()
                call_status = "failed"
                try:
                    result = request_json(
                        args.api_url.rstrip("/") + "/chat/completions",
                        payload=payload, call_id=call_id, timeout=args.request_timeout,
                    )
                    atomic_json(args.output_dir / f"response-{call_id}.json", result)
                    record["response_sha256"] = json_digest(result)
                    content = checked_response(result, seen, expected_alias=model.served_model)
                    record.update(backend_request_id=result["id"], usage=result["usage"],
                                  finish_reason=result["choices"][0]["finish_reason"], content=content)
                    call_status = "success"
                    try:
                        answer = parse_json_object(content)
                    except (TypeError, ValueError):
                        answer = None
                    record["capability_match"] = answer == expected
                except BaseException as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                    if isinstance(error, BackendHTTPError):
                        record["backend_error"] = {"status": error.status, "body": error.body}
                    raise
                finally:
                    record["seconds"] = time.monotonic() - call_started
                    record["status"] = call_status
                    ledger.finish(call_id, record["seconds"], status=call_status, receipt=record)
                    atomic_json(args.output_dir / "vision-probe.json", report)
            report["passed"] = len(report["calls"]) == 2 and all(
                record["capability_match"] for record in report["calls"]
            )
            status = "complete" if report["passed"] else "failed"
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            status = "interrupted"
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["status"] = status
        try:
            if created_output:
                atomic_json(args.output_dir / "vision-probe.json", report)
        finally:
            budget.finish(job, time.monotonic() - started, status)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument(
        "--model-policy", choices=("fixed-native-lora", "manifest-locked"),
        default="fixed-native-lora",
        help="Explicit manifest-locked opt-in for a separately authorized baseline; no fallback",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--limit-seconds", type=int, default=600)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--prepared-identity", type=Path)
    parser.add_argument("--prepared-identity-sha256")
    result = run_probe(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("passed", "status", "wall_seconds")}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
