"""Equal-budget optimizer-code comparison, distinct from actor publication."""

import ast
import math
import time

from .candidate import validate_config
from .rsi import StrategyBundle, validate_source
from .rsi_acceptance import audit_independent_receipts, is_audited_evaluation
from .rsi_experiments import search_history
from .rsi_sandbox import sandbox_call


def choose_hash(bundle):
    tree = validate_source(bundle.source)
    return ast.dump(
        next(n for n in tree.body if n.name == "choose_action"),
        include_attributes=False,
    )


def _digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _evidence_error(value, *, bundle=None):
    """Only the trusted auditor can mint an in-memory comparison receipt."""
    if not is_audited_evaluation(value):
        return "missing or modified independent evaluation audit"
    if (
        value.get("audit_scope") not in {"optimizer-trial-only", "formal-rsi-receipt"}
        or value.get("audit_passed") is not True
        or value.get("cache_reused") is not False
        or value.get("mock")
        or value.get("simulated")
        or not isinstance(value.get("run_id"), str)
        or not value["run_id"]
        or not all(
            _digest(value.get(key)) for key in ("evaluation_sha256", "timing_sha256")
        )
    ):
        return "incomplete independent evaluation receipt"
    if bundle is not None and (
        value.get("status") != "complete" or value.get("bundle_sha256") != bundle.digest
    ):
        return "trial did not complete with the requested frozen bundle"
    ledger = value.get("generation_ledger")
    if not isinstance(ledger, dict):
        return "missing actual generation ledger"
    for key in ("call_ids", "backend_request_keys"):
        ids = ledger.get(key)
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, str) or not item for item in ids)
            or len(ids) != len(set(ids))
        ):
            return "missing or duplicate actual generation identities"
    return None


def _metrics(value):
    values = value.get("f1")
    return (
        isinstance(values, list)
        and len(values) == 3
        and all(
            type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in values
        )
        and type(value.get("failures")) is int
        and value["failures"] >= 0
        and type(value.get("wall_seconds")) in (int, float)
        and math.isfinite(value["wall_seconds"])
        and value["wall_seconds"] > 0
    )


def compare_optimizers(
    parent,
    child,
    actor,
    anchor,
    root,
    evaluate,
    register,
    *,
    trial_cap,
    history=(),
    budget=None,
):
    if (
        len({choose_hash(bundle) for bundle in (parent, child, actor)}) != 1
        or parent.config != child.config
        or parent.config != actor.config
    ):
        raise ValueError("optimizer comparison must keep actor identical")
    if (
        type(trial_cap) not in (int, float)
        or not math.isfinite(trial_cap)
        or trial_cap <= 0
    ):
        raise ValueError("positive finite comparison trial budget required")
    if (
        _evidence_error(anchor)
        or not _metrics(anchor)
        or type(anchor.get("denominator")) is not int
        or anchor["denominator"] < 1
        or anchor.get("completed") != anchor["denominator"]
        or not anchor.get("comparison_identity")
    ):
        raise ValueError("independently audited complete anchor required")
    arms = {"parent": parent, "child": child}
    states = {
        name: {"config": actor.config, "seed": 17, "generation": 0} for name in arms
    }
    visible = {name: search_history(list(history)) for name in arms}
    results = {name: [] for name in arms}
    elapsed = {name: 0.0 for name in arms}
    protocol_errors = []
    for round_number in range(2):
        for name in ("parent", "child"):
            started = time.perf_counter()
            proposals = sandbox_call(
                root,
                arms[name],
                "propose_candidates",
                [visible[name], states[name]],
                budget=budget,
            )
            if not isinstance(proposals, list) or not proposals:
                raise ValueError("optimizer did not return bounded proposals")
            config = validate_config(proposals[round_number % len(proposals)])
            if config["workflow"] == "whole" or config["engine_parameters"]:
                raise ValueError("unsupported optimizer trial")
            candidate = StrategyBundle(arms[name].source, config, actor.digest)
            register(candidate)
            value = evaluate(
                candidate, f"optimizer-{name}-{round_number}", cap=trial_cap
            )
            if not isinstance(value, dict):
                raise TypeError(
                    "optimizer trial must return an audited receipt or failure"
                )
            elapsed[name] += time.perf_counter() - started
            results[name].append(value)
            evidence_error = _evidence_error(value, bundle=candidate)
            if not evidence_error and (
                value.get("comparison_identity") != anchor["comparison_identity"]
                or value.get("completed") != anchor["denominator"]
                or value.get("denominator") != anchor["denominator"]
            ):
                evidence_error = "trial changed the complete comparison conditions"
            if evidence_error:
                protocol_errors.append(f"{name}/{round_number}: {evidence_error}")
            observed = search_history(
                [
                    {
                        **value,
                        "status": value.get("status")
                        if not evidence_error
                        else "failed",
                        "config": candidate.config,
                    }
                ]
            )[0]
            visible[name].append(observed)
            update_started = time.perf_counter()
            states[name] = sandbox_call(
                root,
                arms[name],
                "update_search",
                [states[name], [observed]],
                budget=budget,
            )
            elapsed[name] += time.perf_counter() - update_started
            if not isinstance(states[name], dict) or states[name].get("seed") != 17:
                raise ValueError("optimizer changed fixed search contract")

    def qualified(value):
        return (
            _metrics(value)
            and value.get("status") == "complete"
            and value.get("completed") == anchor["denominator"]
            and value.get("denominator") == anchor["denominator"]
            and value.get("comparison_identity") == anchor["comparison_identity"]
            and all(x >= y for x, y in zip(value["f1"], anchor["f1"]))
            and value["failures"] <= anchor["failures"]
            and value["wall_seconds"] <= 0.8 * anchor["wall_seconds"]
        )

    def utility(values):
        eligible = [r for r in values if qualified(r)]
        return len(eligible), -min(
            (r["wall_seconds"] for r in eligible), default=float("inf")
        )

    within = all(v <= 2 * trial_cap for v in elapsed.values())
    receipts = [anchor, *results["parent"], *results["child"]]
    for key in ("run_id", "evaluation_sha256", "timing_sha256"):
        values = [receipt.get(key) for receipt in receipts]
        if len({str(value) for value in values}) != len(values):
            protocol_errors.append("independent trials reused " + key)
    try:
        audit_independent_receipts(receipts)
    except (ValueError, TypeError, AttributeError):
        protocol_errors.append(
            "independent trials reused or lack actual model receipts"
        )
    return {
        "optimizer_verified": within
        and not protocol_errors
        and utility(results["child"]) > utility(results["parent"]),
        "scope": "fixed-development-regression equal-budget comparison",
        "generalization_claim": False,
        "trials_per_arm": 2,
        "trial_cap_seconds": trial_cap,
        "wall_seconds": elapsed,
        "evidence_errors": protocol_errors,
        "results": results,
        "parent_sha256": parent.digest,
        "child_sha256": child.digest,
    }
