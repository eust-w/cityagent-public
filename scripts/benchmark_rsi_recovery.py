#!/usr/bin/env python3
"""Fixed controlled fault suite with actual model generations, never mocked acceptance."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.backends import (
    AccountedBackend,
    APIBackend,
    ModelManifest,
    normalize_api_url,
)
from bridge_evidence.contest.chunked_inference import infer_chunked
from bridge_evidence.contest.long_document import DocumentSnapshot
from bridge_evidence.contest.rsi import StrategyBundle
from bridge_evidence.contest.rsi_acceptance import RECOVERY_KINDS, RECOVERY_TEXT
from bridge_evidence.contest.rsi_sandbox import PolicyController, verify_root
from bridge_evidence.contest.run_state import (
    atomic_json,
    build_run_identity,
    file_digest,
    json_digest,
)
from bridge_evidence.contest.timing import TimingLedger


def prepare_policy(policy, *, timeout=30):
    """Warm only before injecting a fault; never wait here during recovery."""
    deadline = time.monotonic_ns() + int(timeout * 1e9)
    while True:
        remaining = (deadline - time.monotonic_ns()) / 1e9
        if remaining <= 0 or policy.error:
            raise RuntimeError("policy preparation failed or timed out")
        if policy.wait_ready(min(2.5, remaining)):
            return


def run_cases(output_dir, bundle, generate, tokenizer, ledger, *, policy_factory=None):
    """Run the fixed suite; caller owns hardware, model and cumulative budget."""
    snapshot = DocumentSnapshot.from_text(RECOVERY_TEXT)
    cases, actual = {}, []
    for kind in RECOVERY_KINDS:
        injected = False
        journal = {"schema_version": 1, "case": kind, "events": [], "requests": []}
        journal_path = output_dir / (kind + ".events.json")

        def event(name, **values):
            point = {
                "event": name,
                "monotonic_ns": time.monotonic_ns(),
                "wall_ns": time.time_ns(),
                **values,
            }
            journal["events"].append(point)
            atomic_json(journal_path, journal)
            return point["monotonic_ns"]

        def request(ids, maximum):
            index = len(journal["requests"])
            event("call_start", request_index=index)
            response = generate(ids, maximum)
            journal["requests"].append({"prompt_ids": ids, "response": dict(response)})
            actual.append(response["call_id"])
            event("call_finish", request_index=index)
            return response

        fault_time = resume_time = None

        def fault_backend(ids, maximum):
            nonlocal injected, fault_time, resume_time
            if not injected:
                injected = True
                if kind == "timeout":
                    fault_time = event("fault", injection=kind)
                    raise TimeoutError("controlled timeout injection")
                if kind == "oom":
                    fault_time = event("fault", injection=kind)
                    raise OSError("controlled out of memory injection")
                response = request(
                    ids, min(1, maximum) if kind == "length" else maximum
                )
                # Measure recovery only once the real fault is observable.
                fault_time = event("fault", injection=kind)
                if kind == "interrupt":
                    raise KeyboardInterrupt("controlled process-boundary interruption")
                if kind == "format":
                    # Journal retains genuine receipt; only failed parser input is changed.
                    response = {
                        **response,
                        "text": "{invalid",
                        "fault_injected": "format",
                    }
                return response
            if resume_time is None:
                resume_time = event("resume")
            return request(ids, maximum)

        started = event("started")
        policy = None
        try:
            if policy_factory:
                prepare_start = event("policy_prepare_start")
                policy = policy_factory()
                prepare_policy(policy)
                prepare_end = event("policy_ready")
                ledger.span(
                    kind + ":policy_prepare_seconds",
                    "recovery_policy_prepare_seconds",
                    (prepare_end - prepare_start) / 1e9,
                )
            try:
                result = infer_chunked(
                    snapshot,
                    tokenizer,
                    fault_backend,
                    config=bundle.config,
                    cache_dir=output_dir / kind,
                    policy=policy,
                )
            except KeyboardInterrupt:
                result = infer_chunked(
                    snapshot,
                    tokenizer,
                    fault_backend,
                    config=bundle.config,
                    cache_dir=output_dir / kind,
                    policy=policy,
                )
            if (
                not result["coverage"]["complete"]
                or result["conflicts"]
                or resume_time is None
            ):
                raise ValueError("fault recovery did not produce complete evidence")
            atomic_json(output_dir / (kind + ".json"), result)
            finished = event("complete")
            durations = {
                "start_seconds": (resume_time - fault_time) / 1e9,
                "finish_seconds": (finished - fault_time) / 1e9,
                "total_seconds": (finished - started) / 1e9,
            }
            for name, seconds in durations.items():
                ledger.span(kind + ":" + name, "recovery_" + name, seconds)
            cases[kind] = {
                **durations,
                "coverage": result["coverage"],
                "controlled_injection": True,
                "final_sha256": json_digest(result["data"]),
            }
        finally:
            if policy:
                policy.close()
    return cases, actual


def create_backend(args, model, tokenizer):
    return APIBackend(args.url, model.served_model, tokenizer, timeout=180)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sandbox-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("off", "frozen"), default="frozen")
    parser.add_argument("--url", type=normalize_api_url, default="http://127.0.0.1:8000/v1")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not str(args.output_dir.resolve()).startswith("/workspace/"):
        parser.error("official /workspace required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sandbox_digest = verify_root(args.sandbox_root)
    bundle = StrategyBundle.load(args.bundle)
    model = ModelManifest.load(args.model_manifest).validate()
    identity = model.identity()
    protected = build_run_identity(
        Path(model.base), ROOT, {}, model_files=identity["files"]["base"]
    )["code_files"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model.tokenizer, local_files_only=True, trust_remote_code=True
    )
    ledger = TimingLedger(args.output_dir / "timing.sqlite")
    backend = create_backend(args, model, tokenizer)
    generate = AccountedBackend(backend, ledger)
    cases, actual = run_cases(
        args.output_dir,
        bundle,
        generate,
        tokenizer,
        ledger,
        policy_factory=(lambda: PolicyController(bundle, args.sandbox_root))
        if args.mode == "frozen"
        else None,
    )
    report = {
        "protocol": "rsi-recovery-v2",
        "url": backend.url,
        "mode": args.mode,
        "config": bundle.config,
        "sandbox_digest": sandbox_digest,
        "protected_sha256": json_digest(protected),
        "model": identity,
        "bundle_sha256": bundle.digest if args.mode == "frozen" else None,
        "artifacts": {
            name: file_digest(args.output_dir / name)
            for name in [
                "timing.sqlite",
                *(
                    kind + suffix
                    for kind in RECOVERY_KINDS
                    for suffix in (".json", ".events.json")
                ),
            ]
        },
        "cases": cases,
        "actual_model_call_ids": actual,
        "driver_sha256": file_digest(Path(__file__)),
        "scope": "controlled fault injections with real model requests; not natural hardware-failure evidence",
    }
    atomic_json(args.output_dir / "recovery.json", report)
    print(json.dumps({"cases": len(cases), "actual_requests": len(actual)}))


if __name__ == "__main__":
    main()
