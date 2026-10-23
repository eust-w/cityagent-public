#!/usr/bin/env python3
"""Supplemental real executor SIGKILL test; never substitutes for RSI release gates.

The CLI uses the existing model service and owns one lease in the shared cloud
budget. A fixed synthetic source exercises a crash after durable raw receipt
commit. The supervisor kills only its own Popen child, confirms its exit, then
starts a new executor. All new-process preparation remains in recovery timing.
"""

import argparse
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bridge_evidence.contest import chunked_inference as ci
from bridge_evidence.contest.backends import (
    AccountedBackend,
    APIBackend,
    ModelManifest,
    normalize_api_url,
)
from bridge_evidence.contest.execution_audit import rebuild_chunked_result
from bridge_evidence.contest.long_document import (
    ChunkPlan, DocumentSnapshot, parse_fragment as trusted_parse_fragment,
)
from bridge_evidence.contest.meta_optimization import CloudBudget
from bridge_evidence.contest.rsi import StrategyBundle
from bridge_evidence.contest.rsi_acceptance import (
    PROTECTED_RUNTIME_FILES,
    RECOVERY_TEXT,
    audit_call_ledger,
    audit_strategy_execution,
)
from bridge_evidence.contest.rsi_runtime import TaskJournal, WorkLock
from bridge_evidence.contest.rsi_sandbox import PolicyController, verify_root
from bridge_evidence.contest.run_state import atomic_json, file_digest, json_digest
from bridge_evidence.contest.timing import TimingLedger


def clock_event(name, **values):
    return {
        "event": name,
        "monotonic_ns": time.monotonic_ns(),
        "wall_ns": time.time_ns(),
        **values,
    }


def protected_identity():
    paths = [ROOT / name for name in PROTECTED_RUNTIME_FILES]
    paths += sorted((ROOT / "src/bridge_evidence/contest").rglob("*.py"))
    return {path.relative_to(ROOT).as_posix(): file_digest(path) for path in paths}


class RecoveryPreconditionError(RuntimeError):
    """No valid raw fragment reached the requested crash window."""


def parser_identity():
    path = "src/bridge_evidence/contest/long_document.py"
    decoder = "src/bridge_evidence/contest/model_json.py"
    return {"path": path, "sha256": file_digest(ROOT / path),
            "dependencies": {decoder: file_digest(ROOT / decoder)}}


def ledger_calls(path):
    # Preserve exact backend receipts, not just aggregate call counts/timing.
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as db:
        return db.execute(
            "SELECT id,role,start,seconds,status,receipt FROM calls ORDER BY id"
        ).fetchall()


def audit_precondition(directory, *, before=None, ready=None):
    """Revalidate durable raw offline with the current parser and bound source.

    Passing the live pre-crash state is for the supervisor. Later audits use
    its immutable snapshot, since successful recovery advances the checkpoint.
    No tokenizer, model service or candidate execution is needed here.
    """
    directory = Path(directory)
    if before is None:
        before = json.loads((directory / "precondition.checkpoint.json").read_text())
    if ready is None:
        ready = json.loads((directory / "kill-ready.json").read_text())
    identity = json.loads((directory / "crash-window.identity.json").read_text())
    source = directory / "source.txt"
    snapshot = DocumentSnapshot.from_text(source.read_text())
    parser = parser_identity()
    if (identity.get("parser") != parser
            or identity.get("source_sha256") != file_digest(source)
            or source.read_text() != RECOVERY_TEXT
            or identity.get("snapshot_sha256") != snapshot.digest
            or identity.get("protected_sha256") != json_digest(protected_identity())
            or identity.get("driver_sha256") != file_digest(Path(__file__))
            or identity.get("preparation_driver_sha256") != file_digest(
                ROOT / "scripts/benchmark_rsi_recovery.py")):
        raise ValueError("precondition parser/source/execution identity changed")
    checkpoints = list((directory / "work/cache").glob("*/checkpoint.sqlite"))
    if len(checkpoints) != 1:
        raise ValueError("one fixed synthetic checkpoint required")
    checkpoint = checkpoints[0]
    calls = before["calls"]
    if (len(calls) not in (1, 2) or before["fragments"]
            or len(before["tasks"]) != 1 or before["initial_tasks"] != 1
            or before["issued"] != len(calls)):
        raise ValueError("expected one valid durable raw within ordinary two-attempt budget")
    task = ci._restore_task(before["tasks"][0])
    instruction = ci._instruction(identity["config"])
    ci._validate_plan(ChunkPlan(snapshot.digest, [task], identity["config"]["input_budget"],
                               identity["config"]["output_budget"], instruction),
                      snapshot, identity["config"], instruction)
    if (before["pending"] != [task.task_id]
            or before["attempts"].get(task.task_id, {}).get("count") != len(calls)
            or calls[-1]["call_id"] != ready.get("call_id")
            or calls[-1].get("response_validation") != "pending"
            or any(call.get("response_validation") != "failed" for call in calls[:-1])):
        raise ValueError("checkpoint differs from the valid crash sync receipt")
    raw_hashes, audit_calls, fragment = {}, [], None
    for call in calls:
        path = ci._raw_receipt_path(checkpoint.parent, call["call_id"])
        receipt = call.get("raw_receipt", {})
        if (receipt.get("path") != path.relative_to(checkpoint.parent).as_posix()
                or path.is_symlink() or path.stat().st_size > 2 * ci._raw_limit(identity["config"]) + 65536):
            raise ValueError("precondition raw receipt path/size changed")
        stored = json.loads(path.read_text())
        raw = stored.get("raw")
        raw_sha256 = hashlib.sha256(raw.encode()).hexdigest() if isinstance(raw, str) else None
        if (stored.get("schema_version") != 1
                or stored.get("cache_identity") != checkpoint.parent.name
                or stored.get("call_id") != call["call_id"]
                or stored.get("task_id") != task.task_id
                or call.get("task_id") != task.task_id
                or stored.get("source_ranges") != task.owned_keys
                or call.get("source_ranges") != task.owned_keys
                or stored.get("prompt_sha256") != call.get("prompt_sha256")
                or stored.get("latency_seconds") != call.get("seconds")
                or raw_sha256 is None or stored.get("raw_sha256") != raw_sha256
                or stored.get("output_sha256") != raw_sha256
                or call.get("output_sha256") != raw_sha256
                or stored.get("truncated") is not False
                or receipt.get("truncated") is not False
                or receipt.get("sha256") != json_digest(stored)):
            raise ValueError("precondition raw receipt identity/hash mismatch")
        audited_call = dict(call)
        if call is calls[-1]:
            # Earlier ordinary validation failures remain failures. Requiring
            # successful completion of those receipts would reject a valid retry.
            ci._complete(audited_call, {"text": raw, **{key: stored[key] for key in (
                "call_id", "finish_reason", "output_tokens", "latency_seconds")}},
                         stored["latency_seconds"], identity["config"]["output_budget"])
        audit_calls.append(audited_call)
        raw_hashes[call["call_id"]] = file_digest(path)
        if call is calls[-1]:
            fragment = trusted_parse_fragment(raw, task, snapshot, call["call_id"])
            expected_ready = {
                "protocol": "valid-raw-precondition-v1", "parser": parser,
                "source_sha256": file_digest(source), "snapshot_sha256": snapshot.digest,
                "task_id": task.task_id, "raw_sha256": raw_sha256,
                "fragment_sha256": json_digest(asdict(fragment)),
            }
            if ready.get("precondition") != expected_ready:
                raise ValueError("forged or stale valid-raw precondition receipt")
    audited = audit_call_ledger(directory / "timing.sqlite", [{
        "schema_version": 3, "model_calls": audit_calls,
    }])
    if audited["failures"]:
        raise ValueError("model receipt was not durably successful before SIGKILL")
    result = {
        "protocol": "valid-raw-precondition-v1", "status": "passed",
        "valid_call_id": calls[-1]["call_id"], "parser_sha256": parser["sha256"],
        "parser": parser,
        "source_sha256": file_digest(source), "snapshot_sha256": snapshot.digest,
        "fragment_sha256": json_digest(asdict(fragment)),
        "initial_call_ids": [call["call_id"] for call in calls],
        "raw_receipts_sha256": raw_hashes,
        "checkpoint_sha256": json_digest(before),
        "ledger_calls_sha256": json_digest(ledger_calls(directory / "timing.sqlite")),
    }
    saved = directory / "precondition.json"
    if saved.exists() and json.loads(saved.read_text()) != result:
        raise ValueError("precondition evidence changed or resume repeated a model request")
    return result


def create_backend(args, model, tokenizer):
    return APIBackend(getattr(args, "url", "http://127.0.0.1:8000/v1"),
                      model.served_model, tokenizer, timeout=180)


def runtime(args, *, event=lambda *a, **kw: None):
    event("runtime_import_started")
    from bridge_evidence.contest.fast_tokenizer import load_tokenizer
    event("runtime_import_finished")

    event("manifest_validation_started")
    model = ModelManifest.load(args.model_manifest).validate()
    event("manifest_validation_finished")
    prepared = getattr(args, "prepared_identity", None)
    event("model_identity_started", prepared=bool(prepared))
    if prepared:
        from bridge_evidence.contest.model_identity import load_identity

        identity = load_identity(
            model, prepared, expected_sha256=args.prepared_identity_sha256,
        )
    else:
        identity = model.identity()
    event("model_identity_finished", prepared=bool(prepared))
    event("tokenizer_load_started")
    tokenizer = load_tokenizer(model.tokenizer)
    event("tokenizer_load_finished")
    event("backend_setup_started")
    backend = create_backend(args, model, tokenizer)
    event("backend_setup_finished")
    return tokenizer, backend, identity


def validate_sandbox_mode(args):
    mode = getattr(args, "mode", "off")
    if mode not in ("off", "frozen"):
        raise ValueError("unsupported recovery mode")
    if mode == "frozen" and not getattr(args, "sandbox_root", None):
        raise ValueError("frozen recovery requires --sandbox-root")


def worker(args, *, runtime_factory=runtime):
    # Injected factories exist only for unit tests; the public CLI always uses
    # the actual manifest, tokenizer and resident API service above.
    validate_sandbox_mode(args)
    from benchmark_rsi_recovery import prepare_policy

    directory = args.output_dir
    stage = args.worker_stage
    bundle = StrategyBundle.load(args.bundle)
    source = directory / "source.txt"
    snapshot = DocumentSnapshot.from_text(source.read_text())
    events, policy = [], None
    original_parser = ci.parse_fragment
    valid_raw_seen = False

    def event(name, **values):
        point = clock_event(name, pid=os.getpid(), **values)
        events.append(point)
        atomic_json(directory / (stage + ".events.json"), events)
        return point

    with WorkLock(directory / "work"):
        event("owner_lock_acquired")
        try:
            event("runtime_preparation_started")
            tokenizer, backend, model_identity = (
                runtime_factory(args, event=event) if runtime_factory is runtime
                else runtime_factory(args)
            )
            event("runtime_preparation_finished")
            # Off mode never executes candidate code, so an optional stale
            # sandbox must neither block recovery nor enter its identity.
            event("sandbox_verification_started", mode=args.mode)
            sandbox_digest = verify_root(args.sandbox_root) if args.mode == "frozen" else None
            event("sandbox_verification_finished", mode=args.mode)
            event("execution_identity_started")
            identity = {
                "model": model_identity,
                # Injected unit-test backends without a URL are not API evidence.
                "url": getattr(backend, "url", None),
                "source_sha256": file_digest(source),
                "snapshot_sha256": snapshot.digest,
                "parser": parser_identity(),
                "protected_sha256": json_digest(protected_identity()),
                "driver_sha256": file_digest(Path(__file__)),
                "preparation_driver_sha256": file_digest(
                    ROOT / "scripts/benchmark_rsi_recovery.py"
                ),
                "mode": args.mode,
                "config": bundle.config,
                "bundle_sha256": bundle.digest if args.mode == "frozen" else None,
                "sandbox_digest": sandbox_digest,
            }
            atomic_json(directory / (stage + ".identity.json"), identity)
            if stage == "resume" and identity != json.loads((directory / "crash-window.identity.json").read_text()):
                raise ValueError("source, strategy, model or endpoint changed before checkpoint recovery")
            event("execution_identity_finished")
            if args.mode == "frozen":
                event("controller_setup_started")
                policy = PolicyController(bundle, args.sandbox_root)
                prepare_policy(policy)
                event("controller_setup_finished")
            event("ready_for_inference")
            if stage == "crash-window":

                def pause_after_durable_raw(raw, task, document, call_id):
                    nonlocal valid_raw_seen
                    # infer_chunked committed raw + checkpoint synchronously
                    # before calling its parser. Validation here does not commit
                    # the fragment: the hook never returns it to the executor.
                    # Invalid raw propagates unchanged into ordinary bounded retry.
                    event("raw_validation_started", call_id=call_id)
                    try:
                        fragment = original_parser(raw, task, document, call_id)
                    except Exception as error:
                        event("raw_validation_failed", call_id=call_id,
                              error_type=type(error).__name__, error=str(error))
                        raise
                    valid_raw_seen = True
                    event("raw_validation_passed", call_id=call_id)
                    atomic_json(
                        directory / "kill-ready.json",
                        {
                            **clock_event("durable_raw_ready", pid=os.getpid()),
                            "call_id": call_id,
                            "precondition": {
                                "protocol": "valid-raw-precondition-v1",
                                "parser": identity["parser"],
                                "source_sha256": identity["source_sha256"],
                                "snapshot_sha256": document.digest,
                                "task_id": task.task_id,
                                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                                "fragment_sha256": json_digest(asdict(fragment)),
                            },
                        },
                    )
                    time.sleep(600)
                    raise TimeoutError("supervisor never reached the SIGKILL point")

                ci.parse_fragment = pause_after_durable_raw
            ledger = TimingLedger(directory / "timing.sqlite")
            try:
                result = ci.infer_chunked(
                    snapshot, tokenizer, AccountedBackend(backend, ledger),
                    config=bundle.config, cache_dir=directory / "work/cache",
                    policy=policy,
                )
            except Exception as error:
                if stage == "crash-window" and not valid_raw_seen and ledger.summary()["calls"]:
                    atomic_json(directory / "precondition-failed.json", {
                        **event("precondition_failed"), "status": "failed",
                        "reason": "no-valid-raw-within-inference-budget",
                        "model_calls": ledger.summary()["calls"],
                        "parser": identity["parser"],
                        "source_sha256": identity["source_sha256"],
                        "error_type": type(error).__name__, "error": str(error),
                    })
                    raise RecoveryPreconditionError(
                        "recovery precondition failed: no valid raw within inference budget"
                    ) from error
                raise
            atomic_json(directory / "result.json", result)
            event("result_committed")
        finally:
            ci.parse_fragment = original_parser
            if policy:
                policy.close()


def run_process_recovery(args, *, command_factory=None):
    validate_sandbox_mode(args)
    # Fail before model identity I/O if a copied validation package is incomplete.
    # The real worker binds the same complete protected file set again.
    protected_identity()
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "source.txt").write_text(RECOVERY_TEXT)
    children, events = [], []
    started = time.monotonic()
    deadline = started + args.limit_seconds

    def event(name, **values):
        point = clock_event(name, **values)
        events.append(point)
        atomic_json(directory / "supervisor.events.json", events)
        return point

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("process recovery experiment deadline")
        return value

    def command(stage):
        if command_factory:
            return command_factory(stage)
        result = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker-stage", stage,
            "--output-dir", str(directory),
            "--model-manifest", str(args.model_manifest),
            "--bundle", str(args.bundle),
            "--mode", args.mode,
            "--url", getattr(args, "url", "http://127.0.0.1:8000/v1"),
        ]
        if getattr(args, "sandbox_root", None) is not None:
            result += ["--sandbox-root", str(args.sandbox_root)]
        if getattr(args, "prepared_identity", None):
            result += ["--prepared-identity", str(args.prepared_identity),
                       "--prepared-identity-sha256", args.prepared_identity_sha256]
        return result

    def launch(stage, log):
        process = subprocess.Popen(command(stage), stdout=log, stderr=subprocess.STDOUT)
        children.append(process)
        event("worker_launched", stage=stage, pid=process.pid)
        return process

    event("experiment_started")
    try:
        with (directory / "workers.log").open("x") as log:
            initial = launch("crash-window", log)
            ready_path = directory / "kill-ready.json"
            while not ready_path.is_file():
                if initial.poll() is not None:
                    if (directory / "precondition-failed.json").is_file():
                        raise RecoveryPreconditionError(
                            "recovery precondition failed: no valid raw within inference budget"
                        )
                    raise RuntimeError("initial executor exited before durable raw receipt")
                time.sleep(min(0.01, remaining()))
            ready = json.loads(ready_path.read_text())
            if ready["pid"] != initial.pid or initial.poll() is not None:
                raise ValueError("crash sync point does not belong to the live child")
            checkpoints = list((directory / "work/cache").glob("*/checkpoint.sqlite"))
            if len(checkpoints) != 1:
                raise ValueError("one fixed synthetic checkpoint required")
            checkpoint = checkpoints[0]
            with TaskJournal(checkpoint, checkpoint.parent.name) as journal:
                before = journal.load()
            precondition = audit_precondition(directory, before=before, ready=ready)
            atomic_json(directory / "precondition.checkpoint.json", before)
            atomic_json(directory / "precondition.json", precondition)
            event("valid_raw_precondition_verified", call_id=ready["call_id"])
            call = before["calls"][-1]
            raw_path = checkpoint.parent / call["raw_receipt"]["path"]
            raw_sha256 = file_digest(raw_path)
            ledger = TimingLedger(directory / "timing.sqlite")
            before_ledger = ledger.summary()
            if (before_ledger["calls"] != len(before["calls"]) or before_ledger["failures"]
                    or before_ledger["incomplete_call_ids"]):
                raise ValueError("model receipt was not durably successful before SIGKILL")
            fault = event("sigkill_sent", pid=initial.pid, signal=signal.SIGKILL)
            initial.kill()
            killed_returncode = initial.wait(timeout=min(10, remaining()))
            if killed_returncode != -signal.SIGKILL:
                raise ValueError("owned executor did not terminate by SIGKILL")
            event("old_executor_exited", pid=initial.pid, returncode=killed_returncode)
            # No global ledger recovery and no signalling by PID/name. Only
            # after wait() proves the old owner is dead may another take its lock.
            resumed = launch("resume", log)
            if resumed.wait(timeout=remaining()) != 0:
                raise RuntimeError("resumed executor failed")
            finished = event("resumed_executor_exited", pid=resumed.pid)
        result = json.loads((directory / "result.json").read_text())
        identity = json.loads((directory / "crash-window.identity.json").read_text())
        if identity != json.loads((directory / "resume.identity.json").read_text()):
            raise ValueError("source, strategy, model or execution identity changed on resume")
        if (identity["source_sha256"] != file_digest(directory / "source.txt")
                or result["snapshot_sha256"] != identity["snapshot_sha256"]):
            raise ValueError("source identity differs from recovered evidence")
        if file_digest(raw_path) != raw_sha256 or ledger.summary() != before_ledger:
            raise ValueError("resume changed raw evidence or repeated a model request")
        if audit_precondition(directory) != precondition:
            raise ValueError("resume changed valid-raw precondition evidence")
        with TaskJournal(checkpoint, checkpoint.parent.name) as journal:
            after = journal.load()
        if (after["issued"] != before["issued"] or after["attempts"] != before["attempts"]
                or after["pending"] or not after["fragments"]):
            raise ValueError("resume repeated an attempt or left pending work")
        expected_calls = [dict(value) for value in before["calls"]]
        stored = json.loads(raw_path.read_text())
        expected_calls[-1].update(
            status="success", response_validation="success",
            finish_reason=stored["finish_reason"], output_tokens=stored["output_tokens"],
        )
        if after["calls"] != expected_calls:
            raise ValueError("resume changed initial model call receipts")
        rebuilt = rebuild_chunked_result(result)
        if (rebuilt["data"] != result["data"] or not rebuilt["coverage"]["complete"]
                or rebuilt["conflicts"]):
            raise ValueError("recovered final evidence failed reconstruction")
        audited = audit_call_ledger(directory / "timing.sqlite", [result])
        if audited["call_ids"] != precondition["initial_call_ids"]:
            raise ValueError("recovered call is not the original actual receipt")
        strategy = audit_strategy_execution(
            [result], identity["bundle_sha256"], identity["sandbox_digest"]
        ) if args.mode == "frozen" else None
        policy_phases = None
        if args.mode == "frozen":
            initial_decisions = before.get("decisions", [])
            recovered_decisions = result.get("rsi", {}).get("decisions", [])
            if recovered_decisions[:len(initial_decisions)] != initial_decisions:
                raise ValueError("recovery lost or changed the durable initial strategy receipts")
            initial_strategy = audit_strategy_execution(
                [{"rsi": {"decisions": initial_decisions}}],
                identity["bundle_sha256"], identity["sandbox_digest"],
            )
            resumed_decisions = recovered_decisions[len(initial_decisions):]
            resumed_executed = sum(decision.get("fallback") is False for decision in resumed_decisions)
            policy_phases = {
                "before_crash": initial_strategy,
                "resume": {
                    "status": "not-needed" if not resumed_decisions else "executed" if resumed_executed else "fallback-only",
                    "decisions": len(resumed_decisions),
                    "executed_decisions": resumed_executed,
                },
            }
        resumed_events = json.loads((directory / "resume.events.json").read_text())
        marks = {value["event"]: value["monotonic_ns"] for value in resumed_events}
        fault_ns = fault["monotonic_ns"]
        if not fault_ns < marks["owner_lock_acquired"] <= marks["ready_for_inference"] < finished["monotonic_ns"]:
            raise ValueError("recovery event order invalid")
        report = {
            "protocol": "rsi-process-recovery-supplement-v1",
            "scope": "fixed synthetic source; actual executor SIGKILL after durable model response; no model service restart",
            "formal_release_evidence": False,
            **identity,
            "killed_returncode": killed_returncode,
            "pids": {"initial": initial.pid, "resume": resumed.pid},
            "model_calls_before": before_ledger["calls"],
            "model_calls_after": ledger.summary()["calls"],
            "reused_call_ids": [ready["call_id"]],
            "raw_receipt_sha256": raw_sha256,
            "precondition": precondition,
            "ledger": audited,
            "strategy_execution": strategy,
            "policy_phases": policy_phases,
            "recovery": {
                "owner_lock_seconds": (marks["owner_lock_acquired"] - fault_ns) / 1e9,
                "start_seconds": (marks["ready_for_inference"] - fault_ns) / 1e9,
                "finish_seconds": (finished["monotonic_ns"] - fault_ns) / 1e9,
            },
            "total_seconds": time.monotonic() - started,
            "snapshot": asdict(DocumentSnapshot.from_text(RECOVERY_TEXT)),
            "artifacts": {
                name: file_digest(directory / name)
                for name in (
                    "source.txt", "kill-ready.json", "timing.sqlite", "result.json",
                    "crash-window.identity.json", "resume.identity.json",
                    "crash-window.events.json", "resume.events.json", "supervisor.events.json",
                    "precondition.checkpoint.json", "precondition.json",
                )
            },
        }
        atomic_json(directory / "process-recovery.json", report)
        return report
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "model-manifest", "bundle"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sandbox-root", type=Path, help="Required for frozen policy execution; unused in off mode")
    parser.add_argument("--mode", choices=("off", "frozen"), default="off")
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument("--limit-seconds", type=int, default=900)
    parser.add_argument("--prepared-identity", type=Path)
    parser.add_argument("--prepared-identity-sha256")
    parser.add_argument("--worker-stage", choices=("crash-window", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--url", type=normalize_api_url, default="http://127.0.0.1:8000/v1")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_sandbox_mode(args)
    except ValueError as error:
        parser.error(str(error))
    if bool(args.prepared_identity) != bool(args.prepared_identity_sha256):
        parser.error("prepared identity requires its pinned SHA256")
    if args.worker_stage:
        worker(args)
        return
    if (not args.budget_ledger or args.limit_seconds < 30
            or any(not str(path.resolve()).startswith("/workspace/") for path in (args.output_dir, args.budget_ledger))):
        parser.error("official /workspace output/shared budget and limit >= 30 required")
    budget = CloudBudget(args.budget_ledger)
    args.limit_seconds = min(args.limit_seconds, int(budget.remaining()) - 30)
    if args.limit_seconds < 30:
        raise RuntimeError("insufficient cumulative cloud budget")
    job = budget.start("rsi-process-recovery-supplement", args.limit_seconds)
    started, status = time.perf_counter(), "failed"
    try:
        report = run_process_recovery(args)
        status = "complete"
        print(json.dumps({"recovery": report["recovery"], "model_calls": report["model_calls_after"]}))
    finally:
        budget.finish(job, time.perf_counter() - started, status)


if __name__ == "__main__":
    main()
