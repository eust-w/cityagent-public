"""Trusted proposal/replay/shadow orchestration. No labels enter online proposals."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .inference_core import build_prompt
from .long_document import (
    ChunkPlan,
    ChunkTask,
    DocumentSnapshot,
    SourceRegion,
    SourceSlice,
    source_metadata,
)
from .rsi import SIGNATURES, StrategyBundle, apply_patch, validate_action
from .rsi_sandbox import PolicyController, SandboxWorker, policy_state
from .run_state import atomic_json, json_digest


def search_history(outcomes):
    """Search workers see cost/structural feedback, never scores, errors, or paths."""
    from .candidate import validate_config

    result = []
    for outcome in outcomes[-8:]:
        row = {
            "status": "complete" if outcome.get("status") == "complete" else "failed"
        }
        for key in (
            "failures",
            "omissions",
            "conflicts",
            "wall_seconds",
            "generation_seconds",
            "calls",
            "failed_calls",
            "executed_decisions",
            "fallback_decisions",
        ):
            value = outcome.get(key)
            if (
                type(value) in (int, float)
                and math.isfinite(value)
                and 0 <= value <= 1e9
            ):
                row[key] = value
        if isinstance(outcome.get("config"), dict):
            try:
                row["config"] = validate_config(outcome["config"])
            except ValueError:
                pass
        result.append(row)
    return result


def generation_history(records):
    """Construct an allowlisted view, not a deletion-based redaction of scored records."""
    result = []
    for record in records[-5:]:
        calls = record.get("model_calls", [])
        result.append(
            {
                "tasks": len(record.get("chunk_plan", {}).get("tasks", [])),
                "calls": len(calls),
                "failed_calls": sum(c.get("status") != "success" for c in calls),
                "length_failures": sum(
                    c.get("finish_reason") == "length" for c in calls
                ),
                "generation_seconds": sum(c.get("seconds", 0) for c in calls),
                "covered": record.get("coverage", {}).get("complete") is True,
                "conflicts": len(record.get("conflicts", [])),
            }
        )
    return result


def propose_patch(
    parent, history, tokenizer, generate, *, target="choose_action", outcomes=()
):
    if target not in SIGNATURES:
        raise ValueError("unknown patch target")
    payload = {
        "parent_sha256": parent.digest,
        "target_function": target,
        "source": parent.source,
        "history": generation_history(history),
        "outcomes": search_history(outcomes),
        "contract": "pure JSON, allowed actions only; no imports, attributes, files, tools or budget changes",
    }
    instruction = 'Return only JSON {"parent_sha256":...,"target_function":...,"new_body":...}. Improve inference cost and errors; retain source coverage. Helpers: base_action, default_candidates, default_search, get, copy_dict, len, min, max, abs, sorted, range, enumerate.'
    calls = []
    for attempt in range(2):
        ids = build_prompt(
            tokenizer,
            json.dumps(payload, ensure_ascii=False),
            max_model_len=12288,
            max_new_tokens=4096,
            instruction=instruction,
        )
        response = generate.call_with_role(ids, 4096, role="rsi-code-proposal")
        calls.append(
            {
                "call_id": response["call_id"],
                "finish_reason": response.get("finish_reason"),
                "output_sha256": json_digest(response),
                "latency_seconds": response.get("latency_seconds"),
            }
        )
        try:
            if response.get("finish_reason") != "stop":
                raise ValueError("incomplete code proposal")
            patch = json.loads(response["text"])
            if patch.get("target_function") != target:
                raise ValueError("patch target changed")
            bundle = apply_patch(parent, patch)
            return bundle, calls
        except (ValueError, SyntaxError, TypeError) as error:
            # Do not forward arbitrary errors, file names or report text.
            payload["feedback"] = (
                type(error).__name__
                + ": proposal does not satisfy fixed patch contract"
            )
    raise ValueError("two code proposal attempts exhausted")


def replay(bundle, root, *, records=()):
    states = [
        {
            "phase": "schedule",
            "batch_size": b,
            "remaining_calls": 20,
            "remaining_seconds": 1200,
        }
        for b in (1, 2, 4)
    ]
    states += [
        {
            "phase": "failure",
            "error_kind": kind,
            "attempt": attempt,
            "depth": 0,
            "max_depth": 8,
            "max_attempts": 2,
            "remaining_calls": 10,
            "remaining_seconds": 100,
        }
        for kind in ("length", "temporary", "format")
        for attempt in (1, 2)
    ]
    recorded = []
    for record in records[-5:]:
        for decision in record.get("rsi", {}).get("decisions", [])[-32:]:
            try:
                state = policy_state(decision["state"])
            except (KeyError, ValueError, TypeError):
                continue
            if state not in recorded:
                recorded.append(state)
    evidence = []
    for origin, state in [("recorded", s) for s in recorded] + [
        ("edge-probe", s) for s in states
    ]:
        values = []
        for _ in range(2):
            worker = SandboxWorker(root, bundle)
            try:
                values.append(
                    validate_action(
                        worker.invoke("choose_action", [state], timeout=1), state
                    )
                )
            finally:
                worker.close()
        if state["phase"] == "schedule" and values[0]["action"] != "extract":
            raise ValueError("healthy task cannot be dropped")
        if values[0] != values[1]:
            raise ValueError("strategy is nondeterministic")
        evidence.append(
            {"origin": origin, "state_sha256": json_digest(state), "action": values[0]}
        )
    return evidence


def _shadow_scope(snapshot, selected):
    """Rebase exact owned slices to a complete two-task snapshot with original provenance."""
    regions, groups = [], []
    sources = snapshot.sources()
    for task in selected:
        slices = []
        for source in task.owned:
            original = sources.get(source.source_id)
            if (
                original is None
                or original.text[source.start : source.end] != source.text
            ):
                raise ValueError("shadow source ownership changed")
            source_id = "shadow-" + json_digest(source.key)[:20]
            location = {
                "original_source_id": source.source_id,
                "original_start": source.start,
                "original_end": source.end,
            }
            regions.append(
                SourceRegion(
                    source_id,
                    len(regions),
                    source.text,
                    original.kind,
                    location,
                    tuple(
                        {k: v for k, v in cell.items() if k != "text"}
                        for cell in original.cells
                    ),
                    original.header,
                    original.years,
                )
            )
            slices.append(
                SourceSlice(
                    source_id,
                    0,
                    len(source.text),
                    source.text,
                    source_metadata(regions[-1]),
                )
            )
        groups.append(slices)
    if len({r.source_id for r in regions}) != len(regions):
        raise ValueError("duplicate shadow ownership")
    scoped = DocumentSnapshot(
        snapshot.document_sha256, regions, "bounded-shadow-source-v2"
    )
    return scoped, groups


def shadow(parent, candidate, snapshot, tasks, tokenizer, generate, *, sandbox_root):
    """Run both strategies through the real trusted executor on the same two tasks.

    No cache is permitted. Only PolicyController can execute source, in its isolated
    one-shot worker. Passing this structural probe makes no semantic quality claim.
    """
    from .chunked_inference import _instruction, infer_chunked

    selected = tasks[:2]
    if len(selected) != 2:
        raise ValueError("two shadow tasks required")
    scoped, groups = _shadow_scope(snapshot, selected)
    runs = []

    class ShadowBackend:
        def __call__(self, ids, maximum):
            return generate.call_with_role(ids, maximum, role="rsi-shadow")

        def call_with_role(self, ids, maximum, *, role):
            return self(ids, maximum)

    for bundle in (parent, candidate):
        instruction = _instruction(bundle.config)
        planned = []
        for slices in groups:
            task = ChunkTask(
                json_digest([s.key for s in slices])[:24], list(slices), []
            )
            ids = build_prompt(
                tokenizer,
                task.text(),
                max_model_len=bundle.config["input_budget"]
                + bundle.config["output_budget"],
                max_new_tokens=bundle.config["output_budget"],
                instruction=instruction,
            )
            task.prompt_tokens = len(ids)
            planned.append(task)
        plan = ChunkPlan(
            scoped.digest,
            planned,
            bundle.config["input_budget"],
            bundle.config["output_budget"],
            instruction,
        )
        policy = PolicyController(bundle, sandbox_root)
        try:
            if not policy.wait_ready():
                raise ValueError("shadow isolated policy worker not ready")
            result = infer_chunked(
                scoped,
                tokenizer,
                ShadowBackend(),
                config=bundle.config,
                policy=policy,
                prepared_plan=plan,
            )
        finally:
            policy.close()
        decisions = result.get("rsi", {}).get("decisions", [])
        executed = sum(d.get("fallback") is False for d in decisions)
        if not executed:
            raise ValueError(
                "shadow strategy never executed; fallback cannot verify evolution"
            )
        if not result["coverage"]["complete"] or result["conflicts"]:
            raise ValueError("shadow incomplete coverage or conflicts")
        runs.append(
            {
                "bundle_sha256": bundle.digest,
                "record": result,
                "executed_decisions": executed,
                "fallback_decisions": len(decisions) - executed,
            }
        )
    traces = [
        [
            {
                "phase": d["state"]["phase"],
                "action": d["action"],
                "fallback": d["fallback"],
            }
            for d in r["record"]["rsi"]["decisions"]
        ]
        for r in runs
    ]
    effectiveness = {
        "strategy_exercised": True,
        "policy_effect_observed": traces[0] != traces[1],
        "semantic_quality_verified": False,
        "execution_rates": {
            r["bundle_sha256"]: r["executed_decisions"]
            / (r["executed_decisions"] + r["fallback_decisions"])
            for r in runs
        },
    }
    return {
        "schema_version": 2,
        "executor": "infer_chunked",
        "snapshot_sha256": scoped.digest,
        "effectiveness": effectiveness,
        "original_snapshot_sha256": snapshot.digest,
        "task_ids": [t.task_id for t in selected],
        "runs": runs,
    }


def shadow_evidence(shadow_result, parent_digest, candidate_digest):
    """Recheck actual executor records, never upgrade historical config-only probes."""
    if (
        not isinstance(shadow_result, dict)
        or shadow_result.get("schema_version") != 2
        or shadow_result.get("executor") != "infer_chunked"
    ):
        raise ValueError("real controller shadow evidence required")
    runs = shadow_result.get("runs", [])
    if len(runs) != 2 or {r.get("bundle_sha256") for r in runs} != {
        parent_digest,
        candidate_digest,
    }:
        raise ValueError("both parent and candidate shadow runs required")
    calls = []
    for run in runs:
        record = run["record"]
        rsi = record.get("rsi", {})
        decisions = rsi.get("decisions", [])
        raw_snapshot, raw_plan = record["snapshot"], record["chunk_plan"]
        checked_snapshot = DocumentSnapshot(
            **{
                **raw_snapshot,
                "regions": [SourceRegion(**r) for r in raw_snapshot["regions"]],
            }
        )
        checked_plan = ChunkPlan(
            raw_plan["snapshot_sha256"],
            [
                ChunkTask(
                    t["task_id"],
                    [SourceSlice(**v) for v in t["owned"]],
                    [SourceSlice(**v) for v in t["context"]],
                    t["prompt_tokens"],
                )
                for t in raw_plan["tasks"]
            ],
            raw_plan["input_budget"],
            raw_plan["output_budget"],
            raw_plan["instruction"],
        )
        if (
            checked_snapshot.digest != record["snapshot_sha256"]
            or checked_plan.digest != record["plan_sha256"]
            or checked_plan.snapshot_sha256 != checked_snapshot.digest
            or not checked_plan.coverage(checked_snapshot)["complete"]
        ):
            raise ValueError("shadow source plan coverage changed")
        expected = {t.task_id: set(t.owned_keys) for t in checked_plan.tasks}
        fragments = record.get("fragments", [])
        if (
            len(fragments) != len(expected)
            or {f["task_id"] for f in fragments} != set(expected)
            or any(set(f["covered"]) != expected[f["task_id"]] for f in fragments)
        ):
            raise ValueError("shadow source fragments incomplete")
        if (
            rsi.get("bundle_sha256") != run["bundle_sha256"]
            or rsi.get("initial_tasks") != 2
            or record.get("snapshot_sha256") != shadow_result["snapshot_sha256"]
            or record.get("coverage", {}).get("complete") is not True
            or record.get("conflicts")
            or not decisions
        ):
            raise ValueError("shadow executor identities or source coverage invalid")
        executed = 0
        for decision in decisions:
            state = policy_state(decision["state"])
            if (
                state != decision["state"]
                or json_digest(state) != decision["state_sha256"]
                or decision["bundle_sha256"] != run["bundle_sha256"]
            ):
                raise ValueError("shadow action identity changed")
            validate_action(decision["action"], state)
            if decision.get("fallback") is False:
                proof = decision.get("kernel_proof", {})
                if (
                    proof.get("seccomp") != 2
                    or proof.get("no_new_privs") != 1
                    or proof.get("capabilities") != 0
                    or proof.get("namespaces") != 6
                    or proof.get("readonly_root") is not True
                    or len(decision.get("sandbox_runtime_digest", "")) != 64
                ):
                    raise ValueError("shadow policy lacks actual isolation evidence")
                executed += 1
        if not executed:
            raise ValueError("shadow strategy never executed")
        current = record.get("model_calls", [])
        if not 2 <= len(current) <= 10 or any(not c.get("call_id") for c in current):
            raise ValueError("bounded shadow model receipts required")
        successes = {c["call_id"] for c in current if c.get("status") == "success"}
        if not set(record.get("contributing_call_ids", [])) <= successes:
            raise ValueError("shadow merge used unsuccessful calls")
        calls.extend(current)
    if len({c["call_id"] for c in calls}) != len(calls):
        raise ValueError("shadow calls must be freshly generated")
    return calls


def write_ready(
    directory, bundle, proposal_calls, replay_evidence, shadow_calls, *, ledger_path
):
    shadow_records = shadow_calls
    shadow_calls = shadow_evidence(shadow_records, bundle.parent_sha256, bundle.digest)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / (bundle.digest + ".bundle.json"), bundle.as_dict())
    import shutil

    from .run_state import file_digest

    receipt_path = directory / (bundle.digest + ".timing.sqlite")
    shutil.copy2(ledger_path, receipt_path)
    evidence = {
        "schema_version": 2,
        "shadow_records": shadow_records,
        "strategy_exercised": True,
        "ledger_sha256": file_digest(receipt_path),
        "bundle_sha256": bundle.digest,
        "parent_sha256": bundle.parent_sha256,
        "static": True,
        "replay": True,
        "shadow": True,
        "replay_evidence": replay_evidence,
        "proposal_calls": proposal_calls,
        "shadow_calls": shadow_calls,
        "shadow_call_ids": [c["call_id"] for c in shadow_calls],
    }
    atomic_json(directory / (bundle.digest + ".evidence.json"), evidence)
    atomic_json(directory / "ready.json", {"bundle_sha256": bundle.digest})


def load_ready(directory, parent):
    directory = Path(directory)
    pointer = json.loads((directory / "ready.json").read_text())["bundle_sha256"]
    if (
        not isinstance(pointer, str)
        or len(pointer) != 64
        or any(c not in "0123456789abcdef" for c in pointer)
    ):
        raise ValueError("unsafe ready strategy identity")
    bundle = StrategyBundle.load(directory / (pointer + ".bundle.json"))
    evidence = json.loads((directory / (pointer + ".evidence.json")).read_text())
    if (
        bundle.digest != pointer
        or bundle.parent_sha256 != parent.digest
        or evidence["bundle_sha256"] != pointer
    ):
        raise ValueError("ready strategy parent mismatch")
    if evidence.get("schema_version") != 2:
        raise ValueError("legacy config-only shadow cannot verify strategy evolution")
    calls = shadow_evidence(
        evidence.get("shadow_records"), parent.digest, bundle.digest
    )
    by_id = {c["call_id"]: c for c in calls}
    if (
        not evidence.get("proposal_calls")
        or not evidence.get("replay_evidence")
        or len(evidence["proposal_calls"]) > 2
        or evidence.get("shadow_call_ids") != [c["call_id"] for c in calls]
    ):
        raise ValueError("incomplete shadow/proposal evidence")
    import sqlite3

    from .rsi_acceptance import backend_request_key
    from .run_state import file_digest

    ledger = directory / (pointer + ".timing.sqlite")
    if file_digest(ledger) != evidence["ledger_sha256"]:
        raise ValueError("ready strategy call ledger changed")
    with sqlite3.connect("file:" + str(ledger.resolve()) + "?mode=ro", uri=True) as db:
        expected = [
            (c["call_id"], "rsi-code-proposal") for c in evidence["proposal_calls"]
        ]
        expected += [(c["call_id"], "rsi-shadow") for c in calls]
        if len({key for key, _ in expected}) != len(expected):
            raise ValueError("duplicate proposal/shadow receipt")
        backend_requests = set()
        for key, role in expected:
            row = db.execute(
                "SELECT role,status,seconds,receipt FROM calls WHERE id=?", (key,)
            ).fetchone()
            receipt = json.loads(row[3]) if row and row[3] else {}
            if (
                not row
                or row[0] != role
                or row[1] != "success"
                or row[2] <= 0
                or not receipt.get("backend_request_id")
                or (
                    role == "rsi-shadow"
                    and receipt.get("prompt_sha256") != by_id[key]["prompt_sha256"]
                )
                or (role == "rsi-shadow" and row[2] != by_id[key]["seconds"])
                or (
                    role == "rsi-shadow"
                    and any(
                        receipt.get(field) != by_id[key].get(field)
                        for field in ("output_sha256", "output_tokens", "finish_reason")
                    )
                )
            ):
                raise ValueError(
                    "actual backend receipt required for online activation"
                )
            backend_key = backend_request_key(receipt)
            if backend_key in backend_requests:
                raise ValueError(
                    "duplicate actual backend request in proposal/shadow receipts"
                )
            backend_requests.add(backend_key)
    return bundle, evidence
