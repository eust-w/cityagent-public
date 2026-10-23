"""Optimizer comparison contracts; sandbox/evaluation fixtures are unit-only."""

import json
from copy import deepcopy
from typing import ClassVar

import pytest

from bridge_evidence.contest import rsi_acceptance as acceptance
from bridge_evidence.contest import rsi_search_comparison as comparison
from bridge_evidence.contest.rsi import StrategyBundle, apply_patch
from bridge_evidence.contest.run_state import json_digest


@pytest.fixture
def harness(monkeypatch):
    class Harness:
        parent = StrategyBundle.baseline()
        child = apply_patch(
            parent,
            {
                "parent_sha256": parent.digest,
                "target_function": "update_search",
                "new_body": "next_state = default_search(search_state, outcomes)\nreturn next_state",
            },
        )
        actor = parent
        anchor: ClassVar[dict] = {
            "denominator": 3,
            "f1": [0.8, 0.9, 0.75],
            "failures": 0,
            "comparison_identity": "fixed",
            "wall_seconds": 100,
            "run_id": "anchor",
            "generation_ledger": {
                "call_ids": ["anchor-call"],
                "backend_request_keys": ["anchor-request"],
            },
        }

        def __init__(self):
            self.clock = 0.0
            self.budget = object()
            self.registered, self.evaluations, self.proposals, self.updates = (
                [],
                [],
                [],
                [],
            )
            self.transform = lambda result, label: result
            self.proposal_value = None
            self.changed_seed = False
            self.trial_elapsed = 5.0
            self.wrap = lambda result, label: result
            self.anchor = self.seal(
                {
                    **self.anchor,
                    "status": "complete",
                    "completed": 3,
                    "audit_passed": True,
                    "cache_reused": False,
                    "evaluation_sha256": json_digest("anchor-evaluation"),
                    "timing_sha256": json_digest("anchor-timing"),
                }
            )

        @staticmethod
        def seal(result):
            # Only this unit fixture replaces artifact I/O. Mint via the real
            # auditor entrypoint; comparison must still require its live seal.
            with monkeypatch.context() as local:
                local.setattr(
                    acceptance,
                    "_audited_evaluation",
                    lambda *args, **kwargs: (deepcopy(result), {}),
                )
                return acceptance.audited_optimizer_trial(
                    "unit-only", source_root="unit-only"
                )

        def sandbox(self, root, bundle, function, arguments, *, budget=None):
            assert root == "private-root" and budget is self.budget
            self.clock += 0.25
            name = "parent" if bundle.digest == self.parent.digest else "child"
            if function == "propose_candidates":
                history, state = deepcopy(arguments)
                self.proposals.append((name, history, state))
                if self.proposal_value is not None:
                    return self.proposal_value
                return [{**self.actor.config, "batch_size": 1 + state["generation"]}]
            assert function == "update_search"
            state, outcomes = deepcopy(arguments)
            self.updates.append((name, state, outcomes))
            return {
                **state,
                "generation": state["generation"] + 1,
                "seed": 99 if self.changed_seed else 17,
                "last_wall_seconds": outcomes[0]["wall_seconds"],
            }

        def evaluate(self, candidate, label, *, cap):
            assert candidate in self.registered
            self.evaluations.append((candidate, label, cap))
            self.clock += self.trial_elapsed
            result = {
                "status": "complete",
                "completed": 3,
                "denominator": 3,
                "comparison_identity": "fixed",
                "f1": [0.8, 0.9, 0.75],
                "failures": 0,
                "wall_seconds": 70 if "parent" in label else 60,
                "audit_passed": True,
                "cache_reused": False,
                "run_id": label,
                "bundle_sha256": candidate.digest,
                "evaluation_sha256": json_digest([label, "evaluation"]),
                "timing_sha256": json_digest([label, "timing"]),
                "generation_ledger": {
                    "call_ids": [label + "-call"],
                    "backend_request_keys": [label + "-request"],
                },
            }
            result = self.transform(result, label)
            return (
                self.wrap(self.seal(result), label)
                if result.get("status") == "complete"
                else result
            )

        def run(self, **kwargs):
            return comparison.compare_optimizers(
                self.parent,
                self.child,
                self.actor,
                self.anchor,
                "private-root",
                self.evaluate,
                self.registered.append,
                trial_cap=kwargs.pop("trial_cap", 100),
                budget=self.budget,
                **kwargs,
            )

    value = Harness()
    monkeypatch.setattr(comparison, "sandbox_call", value.sandbox)
    monkeypatch.setattr(comparison.time, "perf_counter", lambda: value.clock)
    return value


def test_equal_history_budget_actor_and_feedback_drive_two_real_rounds(harness):
    prior = {
        "status": "complete",
        "wall_seconds": 80,
        "failures": 0,
        "f1": [1, 1, 1],
        "secret_label_path": "/labels/gold.json",
    }
    result = harness.run(history=[prior])
    assert result["optimizer_verified"]
    assert result["generalization_claim"] is False
    assert result["trials_per_arm"] == 2
    assert [entry[1] for entry in harness.evaluations] == [
        "optimizer-parent-0",
        "optimizer-child-0",
        "optimizer-parent-1",
        "optimizer-child-1",
    ]
    assert {entry[2] for entry in harness.evaluations} == {100}
    assert (
        harness.proposals[0][1]
        == harness.proposals[1][1]
        == [{"status": "complete", "wall_seconds": 80, "failures": 0}]
    )
    for name, history, state in harness.proposals[2:]:
        assert state["generation"] == 1 and state["seed"] == 17
        assert history[-1]["wall_seconds"] == (70 if name == "parent" else 60)
        assert state["last_wall_seconds"] == history[-1]["wall_seconds"]
        assert "f1" not in history[-1]
    assert all(
        candidate.parent_sha256 == harness.actor.digest
        for candidate in harness.registered
    )
    assert {comparison.choose_hash(candidate) for candidate in harness.registered} == {
        comparison.choose_hash(harness.actor)
    }
    assert [candidate.config["batch_size"] for candidate in harness.registered] == [
        1,
        1,
        2,
        2,
    ]


@pytest.mark.parametrize("change", ["actor", "config"])
def test_comparison_rejects_actor_or_config_difference_before_execution(
    harness, change
):
    if change == "actor":
        harness.child = apply_patch(
            harness.child,
            {
                "parent_sha256": harness.child.digest,
                "target_function": "choose_action",
                "new_body": "return {'action': 'fail'}",
            },
        )
    else:
        harness.child = StrategyBundle(
            harness.child.source,
            {**harness.child.config, "batch_size": 2},
            harness.parent.digest,
        )
    with pytest.raises(ValueError, match="actor identical"):
        harness.run()
    assert not harness.evaluations and not harness.proposals


@pytest.mark.parametrize(
    "change",
    [
        {"completed": 2},
        {"denominator": 4},
        {"comparison_identity": "different"},
        {"mock": True},
        {"simulated": True},
        {"f1": [0.79, 0.9, 0.75]},
        {"f1": [1.01, 0.9, 0.75]},
        {"failures": 1},
        {"wall_seconds": 81},
        {"wall_seconds": 0},
    ],
)
def test_incomplete_simulated_failed_quality_and_slow_trials_cannot_verify(
    harness, change
):
    harness.transform = lambda result, label: (
        {**result, **change} if "child" in label else result
    )
    assert not harness.run()["optimizer_verified"]


def test_equal_utility_is_not_recursive_optimizer_improvement(harness):
    harness.transform = lambda result, label: {**result, "wall_seconds": 70}
    assert not harness.run()["optimizer_verified"]


def test_search_work_and_trials_count_toward_each_arm_budget(harness):
    harness.trial_elapsed = 9.75
    # Each trial plus proposal and state update is 10.25 seconds: 20.5 per arm.
    result = harness.run(trial_cap=10)
    assert result["wall_seconds"] == {"parent": 20.5, "child": 20.5}
    assert not result["optimizer_verified"]


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        [{"workflow": "whole"}],
        [{"engine_parameters": {"enforce_eager": True}}],
        [{"batch_size": 3}],
    ],
)
def test_invalid_or_unsupported_proposals_do_not_reach_evaluation(harness, value):
    harness.proposal_value = value
    with pytest.raises(ValueError):
        harness.run()
    assert not harness.evaluations


def test_changed_random_seed_cannot_define_next_generation(harness):
    harness.changed_seed = True
    with pytest.raises(ValueError, match="fixed search contract"):
        harness.run()
    assert len(harness.evaluations) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"audit_passed": False},
        {"cache_reused": True},
        {"status": "failed"},
        {"generation_ledger": {}},
        {"evaluation_sha256": ""},
        {"bundle_sha256": "a" * 64},
    ],
)
def test_summary_flags_without_bound_actual_evidence_cannot_verify(harness, change):
    harness.transform = lambda result, label: (
        {**result, **change} if "child" in label else result
    )
    assert not harness.run()["optimizer_verified"]


@pytest.mark.parametrize("field", ["call_ids", "backend_request_keys"])
def test_reused_actual_receipts_cannot_verify_optimizer(harness, field):
    def reuse(result, label):
        result["generation_ledger"][field] = ["reused"]
        return result

    harness.transform = reuse
    assert not harness.run()["optimizer_verified"]


def test_failed_actual_feedback_stays_failed_in_next_generation(harness):
    harness.transform = lambda result, label: {
        **result,
        "status": "failed",
        "completed": 2,
    }
    result = harness.run()
    assert not result["optimizer_verified"]
    assert all(update[2][0]["status"] == "failed" for update in harness.updates)
    assert all(entry[1][-1]["status"] == "failed" for entry in harness.proposals[2:])


@pytest.mark.parametrize(
    "copy_receipt", [dict, lambda value: json.loads(json.dumps(value))]
)
def test_copying_even_valid_audit_flags_cannot_mint_trial_receipt(
    harness, copy_receipt
):
    harness.wrap = lambda result, label: (
        copy_receipt(result) if "child" in label else result
    )
    result = harness.run()
    assert not result["optimizer_verified"]
    assert "independent evaluation audit" in result["evidence_errors"][0]
    assert all(
        item[2][0]["status"] == "failed"
        for item in harness.updates
        if item[0] == "child"
    )


def test_mutation_after_real_audit_invalidates_trial(harness):
    def mutate(result, label):
        if "child" in label:
            result["wall_seconds"] = 1
        return result

    harness.wrap = mutate
    assert not harness.run()["optimizer_verified"]


def test_raw_anchor_flags_do_not_allow_work_to_start(harness):
    harness.anchor = dict(harness.anchor)
    with pytest.raises(ValueError, match="audited complete anchor"):
        harness.run()
    assert not harness.evaluations and not harness.proposals


@pytest.mark.parametrize("field", ["run_id", "evaluation_sha256", "timing_sha256"])
def test_repeated_run_and_artifact_identity_cannot_verify(harness, field):
    harness.transform = lambda result, label: {**result, field: harness.anchor[field]}
    assert not harness.run()["optimizer_verified"]


@pytest.mark.parametrize("field", ["call_ids", "backend_request_keys"])
def test_reusing_anchor_requests_cannot_verify(harness, field):
    def reuse(result, label):
        result["generation_ledger"][field] = harness.anchor["generation_ledger"][field]
        return result

    harness.transform = reuse
    assert not harness.run()["optimizer_verified"]


def test_warmup_cannot_reuse_another_trials_generation(harness):
    def reuse(result, label):
        if label == "optimizer-child-1":
            result["warmup_ledger"] = {
                "call_ids": ["optimizer-parent-0-call"],
                "backend_request_keys": ["optimizer-parent-0-request"],
            }
        return result

    harness.transform = reuse
    assert not harness.run()["optimizer_verified"]


def test_different_active_actor_is_not_an_optimizer_only_comparison(harness):
    harness.actor = apply_patch(
        harness.actor,
        {
            "parent_sha256": harness.actor.digest,
            "target_function": "choose_action",
            "new_body": "return {'action': 'fail'}",
        },
    )
    with pytest.raises(ValueError, match="actor identical"):
        harness.run()
    assert not harness.evaluations


@pytest.mark.parametrize("cap", [0, -1, True, float("inf"), float("nan"), "100"])
def test_invalid_trial_budget_cannot_start_work(harness, cap):
    with pytest.raises(ValueError, match="finite comparison trial budget"):
        harness.run(trial_cap=cap)
    assert not harness.evaluations and not harness.proposals


def test_nonfinite_metrics_cannot_receive_audit_seal(harness):
    harness.transform = lambda result, label: {
        **result,
        "f1": [float("nan"), 0.9, 0.75],
    }
    with pytest.raises(ValueError, match="JSON compliant"):
        harness.run()


@pytest.mark.parametrize(
    "change",
    [{"completed": 2}, {"denominator": 4}, {"comparison_identity": "different"}],
)
def test_invalid_parent_cannot_artificially_make_child_win(harness, change):
    harness.transform = lambda result, label: (
        {**result, **change} if "parent" in label else result
    )
    assert not harness.run()["optimizer_verified"]
    assert all(
        update[2][0]["status"] == "failed"
        for update in harness.updates
        if update[0] == "parent"
    )
