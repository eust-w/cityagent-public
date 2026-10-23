import copy
import threading
import time

import pytest

from bridge_evidence.contest.rsi import (
    StrategyBundle,
    apply_patch,
    base_action,
    validate_action,
    validate_source,
)
from bridge_evidence.contest.rsi_acceptance import replacement_errors
from bridge_evidence.contest.rsi_evolution import EvolutionStore, OnlineCoordinator
from bridge_evidence.contest.rsi_runtime import Dispatcher, TaskJournal, WorkLock


def test_bundle_patch_is_content_bound_and_function_only():
    bundle = StrategyBundle.baseline()
    patched = apply_patch(
        bundle,
        {
            "parent_sha256": bundle.digest,
            "target_function": "choose_action",
            "new_body": "return {'action': 'extract', 'batch_size': 1}",
        },
    )
    assert patched.digest != bundle.digest
    assert patched.parent_sha256 == bundle.digest
    with pytest.raises(ValueError):
        apply_patch(
            bundle,
            {
                "parent_sha256": "0" * 64,
                "target_function": "choose_action",
                "new_body": "return {}",
            },
        )
    with pytest.raises(ValueError):
        apply_patch(
            bundle,
            {
                "parent_sha256": bundle.digest,
                "target_function": "judge",
                "new_body": "return {}",
            },
        )


@pytest.mark.parametrize(
    "body",
    [
        "import os\nreturn {}",
        "return open('/etc/passwd')",
        "return state.__class__",
        "return eval('1')",
        "return {'action': str.__subclasses__()}",
    ],
)
def test_unsafe_python_is_rejected(body):
    source = StrategyBundle.baseline().source.replace("return base_action(state)", body)
    with pytest.raises(ValueError):
        validate_source(source)


def test_actions_cannot_increase_limits_or_add_tools():
    state = {
        "phase": "schedule",
        "batch_size": 2,
        "remaining_calls": 4,
        "attempt": 0,
        "max_attempts": 2,
        "depth": 0,
        "max_depth": 8,
    }
    assert validate_action(base_action(state), state)["batch_size"] == 2
    for action in (
        {"action": "extract", "batch_size": 4},
        {"action": "shell", "command": "true"},
        {"action": "extract", "batch_size": 1, "url": "http://example.com"},
    ):
        with pytest.raises(ValueError):
            validate_action(action, state)


def test_dispatcher_limits_global_concurrency_and_returns_completed_first():
    active = maximum = 0
    mutex = threading.Lock()

    def backend(ids, limit):
        nonlocal active, maximum
        with mutex:
            active += 1
            maximum = max(maximum, active)
        time.sleep(ids[0])
        with mutex:
            active -= 1
        return {"text": str(ids[0])}

    with Dispatcher(backend, capacity=2) as dispatcher:
        dispatcher.begin_foreground()
        with pytest.raises(RuntimeError):
            dispatcher.submit(([0], 1), priority="experiment")
        results = list(dispatcher.completed([([0.04], 1), ([0.001], 1)]))
        assert results[0][0] == 1
        dispatcher.end_foreground()
    assert maximum == 2


def test_work_lock_rejects_second_owner(tmp_path):
    with WorkLock(tmp_path):
        with pytest.raises(RuntimeError):
            with WorkLock(tmp_path):
                pass


def test_task_journal_restores_split_graph_without_replaying_parent(tmp_path):
    with TaskJournal(tmp_path / "tasks.sqlite", "identity") as journal:
        journal.save(
            {
                "pending": ["child-a", "child-b"],
                "split": {"parent": ["child-a", "child-b"]},
            }
        )
    with TaskJournal(tmp_path / "tasks.sqlite", "identity") as journal:
        assert journal.load()["pending"] == ["child-a", "child-b"]
    with pytest.raises(ValueError):
        TaskJournal(tmp_path / "tasks.sqlite", "changed")


def receipt(run_id):
    return {
        "run_id": run_id,
        "protocol": "frozen-rsi-v3",
        "comparison_identity": "same",
        "bundle_sha256": "a" * 64,
        "denominator": 79,
        "completed": 79,
        "f1": [0.9, 0.8, 0.7],
        "failures": 0,
        "missing_sources": 0,
        "conflicts": 0,
        "wall_seconds": 80,
        "latency": {"p50": 1, "p95": 2},
        "recovery": {"cases": {"timeout": {"start_seconds": 1, "finish_seconds": 2}}},
        "cache_reused": False,
        "audit_passed": True,
    }


def test_rsi_gate_requires_two_runs_and_zero_quality_or_recovery_regression():
    baseline = receipt("base")
    baseline["wall_seconds"] = 100
    first, second = receipt("one"), receipt("two")
    assert replacement_errors(baseline, [first, second]) == []
    assert replacement_errors(baseline, [first, first])
    worse = copy.deepcopy(second)
    worse["f1"][0] -= 0.00001
    assert replacement_errors(baseline, [first, worse])
    worse = copy.deepcopy(second)
    worse["recovery"]["cases"]["timeout"]["finish_seconds"] = 3
    assert replacement_errors(baseline, [first, worse])
    first["protocol"] = "mixed-policy-development"
    assert replacement_errors(baseline, [first, second])


def test_evolution_global_cap_and_online_activation_are_bounded(tmp_path):
    with EvolutionStore(tmp_path / "evolution.sqlite", "model", "evaluation") as store:
        initial = StrategyBundle.baseline()
        store.add(initial, generation=0)
        coordinator = OnlineCoordinator(initial, store)
        assert coordinator.pin().digest == initial.digest
        assert not coordinator.offer(
            initial, evidence={"static": True, "replay": True, "shadow": False}
        )
        assert coordinator.pin().digest == initial.digest
        for i in range(7):
            bundle = apply_patch(
                initial,
                {
                    "parent_sha256": initial.digest,
                    "target_function": "choose_action",
                    "new_body": f"x = {i}\nreturn base_action(state)",
                },
            )
            store.add(bundle, generation=1)
        with pytest.raises(RuntimeError):
            store.add(
                apply_patch(
                    initial,
                    {
                        "parent_sha256": initial.digest,
                        "target_function": "choose_action",
                        "new_body": "x = 100\nreturn base_action(state)",
                    },
                ),
                generation=1,
            )


def test_rsi_chunked_checkpoint_uses_final_graph_and_v3_evidence(tmp_path):
    from test_chunked_integration import CharTokenizer, FragmentBackend

    from bridge_evidence.contest.chunked_inference import infer_chunked
    from bridge_evidence.contest.long_document import DocumentSnapshot
    from bridge_evidence.contest.rsi_sandbox import PolicyController

    bundle = StrategyBundle.baseline({"input_budget": 3000, "output_budget": 500})
    snapshot = DocumentSnapshot.from_text("测试桥\n裂缝\n裂缝")
    first = FragmentBackend(length_once=True)
    policy = PolicyController(bundle)
    result = infer_chunked(
        snapshot,
        CharTokenizer(),
        first,
        config=bundle.config,
        cache_dir=tmp_path,
        policy=policy,
    )
    assert result["coverage"]["complete"]
    assert result["rsi"]["bundle_sha256"] == bundle.digest
    assert any(d["action"]["action"] == "split" for d in result["rsi"]["decisions"])
    second = FragmentBackend(length_once=True)
    resumed = infer_chunked(
        snapshot,
        CharTokenizer(),
        second,
        config=bundle.config,
        cache_dir=tmp_path,
        policy=PolicyController(bundle),
    )
    assert second.calls == 0
    assert resumed["chunk_plan"] == result["chunk_plan"]
    assert resumed["model_calls"] == result["model_calls"]


def test_candidate_worker_unavailable_never_executes_in_host():
    from bridge_evidence.contest.rsi_sandbox import PolicyController

    parent = StrategyBundle.baseline()
    bundle = apply_patch(
        parent,
        {
            "parent_sha256": parent.digest,
            "target_function": "choose_action",
            "new_body": "return {'action': 'fail'}",
        },
    )
    policy = PolicyController(bundle)
    action = policy.choose({"phase": "schedule", "batch_size": 1, "remaining_calls": 2})
    assert action["action"] == "extract"
    assert policy.decisions[-1]["fallback"] is True


def test_dispatcher_call_roles_are_preserved():
    class Backend:
        def __call__(self, ids, limit):
            return {"text": "normal"}

        def call_with_role(self, ids, limit, *, role):
            return {"text": role}

    with Dispatcher(Backend()) as dispatcher:
        assert (
            dispatcher.call_with_role([1], 1, role="react-repair")["text"]
            == "react-repair"
        )


def test_online_activation_pins_boundary_and_survives_restart(tmp_path, monkeypatch):
    # This tests coordinator lifecycle only; actual shadow acceptance has independent tests.
    from bridge_evidence.contest import rsi_experiments
    monkeypatch.setattr(rsi_experiments, "shadow_evidence", lambda *args: [
        {"call_id": key} for key in ("one", "two", "three", "four")
    ])
    parent = StrategyBundle.baseline()
    child = apply_patch(
        parent,
        {
            "parent_sha256": parent.digest,
            "target_function": "choose_action",
            "new_body": "return {'action': 'extract', 'batch_size': 1}",
        },
    )
    with EvolutionStore(tmp_path / "evolution.sqlite", "model", "data") as store:
        store.add(parent, 0)
        store.add(child, 1)
        online = OnlineCoordinator(parent, store)
        assert online.pin().digest == parent.digest
        evidence = {
            "static": True,
            "replay": True,
            "shadow": True,
            "worker_ready": True,
            "bundle_sha256": child.digest,
            "shadow_call_ids": ["one", "two", "three", "four"],
        }
        assert online.offer(child, evidence=evidence)
        assert online.active.digest == parent.digest
        assert online.pin().digest == child.digest
        assert [online.report_done() for _ in range(5)] == [False] * 4 + [True]
    with EvolutionStore(tmp_path / "evolution.sqlite", "model", "data") as store:
        restored = OnlineCoordinator(parent, store)
        assert restored.active.digest == child.digest
        assert restored.completed == 5 and restored.updates == 1
        restored.rollback("failure")
        assert restored.pin().digest == parent.digest


def test_recovery_priority_precedes_queued_extraction_and_context_is_preserved():
    from bridge_evidence.contest.rsi_runtime import REPORT_DEADLINE, REQUEST_PRIORITY

    started, release = threading.Event(), threading.Event()
    order = []

    def backend(ids, limit):
        if ids[0] == "block":
            started.set()
            assert release.wait(1)
        order.append((ids[0], REPORT_DEADLINE.get()))
        return {"text": ids[0]}

    token = REPORT_DEADLINE.set(123)
    try:
        with Dispatcher(backend) as dispatcher:
            blocking = dispatcher.submit((["block"], 1))
            assert started.wait(1)
            regular = dispatcher.submit((["regular"], 1))
            priority_token = REQUEST_PRIORITY.set("recovery")
            try:
                recovery = dispatcher.submit((["recovery"], 1))
            finally:
                REQUEST_PRIORITY.reset(priority_token)
            release.set()
            blocking.result(timeout=1)
            recovery.result(timeout=1)
            regular.result(timeout=1)
        assert order == [("block", 123), ("recovery", 123), ("regular", 123)]
    finally:
        REPORT_DEADLINE.reset(token)


def test_nested_function_and_signature_mutation_rejected():
    parent = StrategyBundle.baseline()
    with pytest.raises(ValueError):
        apply_patch(
            parent,
            {
                "parent_sha256": parent.digest,
                "target_function": "choose_action",
                "new_body": "def nested():\n    return {}\nreturn {}",
            },
        )
    with pytest.raises(ValueError):
        validate_source(
            parent.source.replace("choose_action(state)", "choose_action(state=None)")
        )


def test_release_flag_cannot_replace_actual_evidence(tmp_path):
    import json

    from bridge_evidence.contest.rsi_acceptance import verify_release

    path = tmp_path / "release.json"
    path.write_text(
        json.dumps({"schema_version": 1, "bundle_sha256": "wrong", "verified": True})
    )
    with pytest.raises(ValueError, match="identity"):
        verify_release(path, StrategyBundle.baseline().digest, source_root=tmp_path)


def test_third_react_and_depth_overflow_are_rejected():
    state = {
        "phase": "merge",
        "allow_repair": True,
        "react_calls": 2,
        "remaining_calls": 4,
        "remaining_seconds": 100,
    }
    assert base_action(state)["action"] == "fail"
    with pytest.raises(ValueError):
        validate_action({"action": "repair"}, state)
    assert (
        base_action(
            {
                "phase": "failure",
                "error_kind": "length",
                "depth": 8,
                "max_depth": 8,
                "remaining_calls": 10,
            }
        )["action"]
        == "fail"
    )


def test_sandbox_worker_uses_only_explicit_internal_library_paths():
    from bridge_evidence.contest import rsi_namespace

    assert set(rsi_namespace.WORKER_ENV) == {"LD_LIBRARY_PATH"}
    paths = rsi_namespace.WORKER_ENV["LD_LIBRARY_PATH"].split(":")
    assert "/usr/local/lib" in paths and "/lib" in paths
    assert all(
        p.startswith("/") and "/workspace" not in p and ".." not in p for p in paths
    )
