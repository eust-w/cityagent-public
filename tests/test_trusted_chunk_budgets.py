"""Chunked trusted budgets apply equally with and without a strategy worker."""

import shutil

import pytest
from test_chunked_integration import CharTokenizer, FragmentBackend

from bridge_evidence.contest import chunked_inference as ci
from bridge_evidence.contest.candidate import validate_config
from bridge_evidence.contest.long_document import DocumentSnapshot
from bridge_evidence.contest.rsi import StrategyBundle
from bridge_evidence.contest.rsi_runtime import REPORT_DEADLINE, TaskJournal
from bridge_evidence.contest.rsi_sandbox import PolicyController
from bridge_evidence.contest.run_state import json_digest

CONFIG = validate_config({"input_budget": 3000, "output_budget": 500})


def policy_for(mode):
    return PolicyController(StrategyBundle.baseline(CONFIG)) if mode == "frozen" else None


class Clock:
    now = 100.0

    def monotonic(self):
        return self.now

    def perf_counter(self):
        return self.now

    def time(self):
        return 1_000_000_000 + self.now


@pytest.mark.parametrize("mode", ["off", "frozen"])
@pytest.mark.parametrize("outer", [None, 700.0])
def test_same_deadline_and_context_restored(monkeypatch, mode, outer):
    clock, observed = Clock(), []
    monkeypatch.setattr(ci, "time", clock)
    backend = FragmentBackend()

    def generate(ids, maximum):
        observed.append(REPORT_DEADLINE.get())
        return backend(ids, maximum)

    token = REPORT_DEADLINE.set(outer)
    try:
        result = ci.infer_chunked(
            DocumentSnapshot.from_text("测试桥"), CharTokenizer(), generate,
            config=CONFIG, policy=policy_for(mode),
        )
        assert result["coverage"]["complete"]
        assert observed == [min(1300, outer) if outer is not None else 1300]
        assert REPORT_DEADLINE.get() == outer
    finally:
        REPORT_DEADLINE.reset(token)


@pytest.mark.parametrize("mode", ["off", "frozen"])
def test_attempt_limit_cannot_be_expanded_by_caller(mode):
    attempts = []

    def generate(ids, maximum):
        attempts.append(ids)
        raise TimeoutError("synthetic backend failure")

    with pytest.raises(ci.ChunkFailure):
        ci.infer_chunked(
            DocumentSnapshot.from_text("测试桥"), CharTokenizer(), generate,
            config=CONFIG, policy=policy_for(mode), max_attempts=9,
        )
    assert len(attempts) == 2


@pytest.mark.parametrize("mode", ["off", "frozen"])
def test_expired_report_never_starts_another_request_and_restores_context(monkeypatch, mode):
    clock, calls = Clock(), []
    monkeypatch.setattr(ci, "time", clock)

    def generate(ids, maximum):
        calls.append(ids)
        clock.now += 1201
        raise TimeoutError("request consumed report deadline")

    token = REPORT_DEADLINE.set(None)
    try:
        with pytest.raises(ci.ChunkFailure):
            ci.infer_chunked(
                DocumentSnapshot.from_text("测试桥"), CharTokenizer(), generate,
                config=CONFIG, policy=policy_for(mode),
            )
        assert len(calls) == 1
        assert REPORT_DEADLINE.get() is None
    finally:
        REPORT_DEADLINE.reset(token)


@pytest.mark.parametrize("mode", ["off", "frozen"])
def test_split_tree_cannot_exceed_four_initial_tasks_plus_two(mode):
    snapshot = DocumentSnapshot.from_text("裂" * 256)
    assert len(ci.prepare_chunk_plan(snapshot, CharTokenizer(), CONFIG).tasks) == 1
    calls = []

    def generate(ids, maximum):
        calls.append(ids)
        return {"text": "{", "finish_reason": "length", "output_tokens": maximum}

    with pytest.raises(ci.ChunkFailure):
        ci.infer_chunked(snapshot, CharTokenizer(), generate, config=CONFIG, policy=policy_for(mode))
    assert len(calls) == 6


@pytest.mark.parametrize("mode", ["off", "frozen"])
def test_resume_carries_spent_budget_and_restores_deadline(monkeypatch, tmp_path, mode):
    clock, backend, observed = Clock(), FragmentBackend(), []
    monkeypatch.setattr(ci, "time", clock)
    snapshot = DocumentSnapshot.from_text("测试桥\n" + "裂" * 500 + "\n" + "裂" * 500)
    assert len(ci.prepare_chunk_plan(snapshot, CharTokenizer(), CONFIG).tasks) == 2

    def first(ids, maximum):
        if backend.calls:
            raise KeyboardInterrupt("executor interrupted before second receipt")
        response = backend(ids, maximum)
        clock.now += 800
        return response

    token = REPORT_DEADLINE.set(None)
    try:
        with pytest.raises(KeyboardInterrupt):
            ci.infer_chunked(
                snapshot, CharTokenizer(), first, config=CONFIG,
                policy=policy_for(mode), cache_dir=tmp_path,
            )
        assert REPORT_DEADLINE.get() is None

        def resumed(ids, maximum):
            observed.append(REPORT_DEADLINE.get() - clock.monotonic())
            return backend(ids, maximum)

        result = ci.infer_chunked(
            snapshot, CharTokenizer(), resumed, config=CONFIG,
            policy=policy_for(mode), cache_dir=tmp_path,
        )
        assert result["coverage"]["complete"]
        assert observed == [400]
        assert backend.calls == 2
        assert REPORT_DEADLINE.get() is None
    finally:
        REPORT_DEADLINE.reset(token)


@pytest.mark.parametrize("mode", ["off", "frozen"])
def test_interruption_time_cannot_reset_an_expired_report_budget(monkeypatch, tmp_path, mode):
    clock, backend = Clock(), FragmentBackend()
    monkeypatch.setattr(ci, "time", clock)
    snapshot = DocumentSnapshot.from_text("测试桥\n" + "裂" * 500 + "\n" + "裂" * 500)

    def first(ids, maximum):
        if backend.calls:
            raise KeyboardInterrupt("unfinished second request")
        response = backend(ids, maximum)
        clock.now += 800
        return response

    with pytest.raises(KeyboardInterrupt):
        ci.infer_chunked(
            snapshot, CharTokenizer(), first, config=CONFIG,
            policy=policy_for(mode), cache_dir=tmp_path,
        )
    clock.now += 500
    resumed = FragmentBackend()
    with pytest.raises(ci.ChunkFailure):
        ci.infer_chunked(
            snapshot, CharTokenizer(), resumed, config=CONFIG,
            policy=policy_for(mode), cache_dir=tmp_path,
        )
    assert resumed.calls == 0
    assert REPORT_DEADLINE.get() is None


@pytest.mark.parametrize("mode", ["off", "frozen"])
def test_depth_eight_failure_is_not_split_again(tmp_path, mode):
    snapshot = DocumentSnapshot.from_text("裂" * 256)

    def interrupt(ids, maximum):
        raise KeyboardInterrupt("before durable response")

    with pytest.raises(KeyboardInterrupt):
        ci.infer_chunked(
            snapshot, CharTokenizer(), interrupt, config=CONFIG,
            policy=policy_for(mode), cache_dir=tmp_path,
        )
    checkpoint = next(tmp_path.glob("*/checkpoint.sqlite"))
    with TaskJournal(checkpoint, checkpoint.parent.name) as journal:
        before = journal.load()
        journal.update(fields={"depths": {task["task_id"]: 8 for task in before["tasks"]}})
    calls = []

    def truncated(ids, maximum):
        calls.append(ids)
        return {"text": "{", "finish_reason": "length", "output_tokens": maximum}

    with pytest.raises(ci.ChunkFailure):
        ci.infer_chunked(
            snapshot, CharTokenizer(), truncated, config=CONFIG,
            policy=policy_for(mode), cache_dir=tmp_path,
        )
    assert len(calls) == 1
    with TaskJournal(checkpoint, checkpoint.parent.name) as journal:
        assert journal.load()["tasks"] == before["tasks"]


@pytest.mark.parametrize("legacy_schema", [None, 2])
def test_old_budget_checkpoint_is_not_reused(tmp_path, legacy_schema):
    snapshot = DocumentSnapshot.from_text("测试桥")
    ci.infer_chunked(snapshot, CharTokenizer(), FragmentBackend(), config=CONFIG, cache_dir=tmp_path)
    checkpoint = next(tmp_path.glob("*/checkpoint.sqlite"))
    if legacy_schema == 2:
        old_key = json_digest({
            "schema": 2, "max_attempts": 2, "snapshot": snapshot.digest,
            "config": CONFIG, "instruction": ci._instruction(CONFIG), "strategy": None,
        })
    else:
        plan = ci.prepare_chunk_plan(snapshot, CharTokenizer(), CONFIG)
        old_key = json_digest({
            "snapshot": snapshot.digest, "plan": plan.digest,
            "config": CONFIG, "strategy": None,
        })
    old = tmp_path / old_key
    if checkpoint.parent != old:
        shutil.move(str(checkpoint.parent), str(old))
        with TaskJournal(old / "checkpoint.sqlite", checkpoint.parent.name) as journal:
            journal.db.execute("UPDATE checkpoint SET identity=? WHERE id=1", (old_key,))
            journal.db.commit()
    backend = FragmentBackend()
    ci.infer_chunked(snapshot, CharTokenizer(), backend, config=CONFIG, cache_dir=tmp_path)
    assert backend.calls == 1
    assert old.is_dir()


def test_current_checkpoint_must_carry_exact_trusted_budget_version(tmp_path):
    snapshot = DocumentSnapshot.from_text("测试桥")
    ci.infer_chunked(snapshot, CharTokenizer(), FragmentBackend(), config=CONFIG, cache_dir=tmp_path)
    checkpoint = next(tmp_path.glob("*/checkpoint.sqlite"))
    with TaskJournal(checkpoint, checkpoint.parent.name) as journal:
        journal.update(fields={"trusted_budget": {"version": "older"}})
    backend = FragmentBackend()
    with pytest.raises(ValueError, match="budget"):
        ci.infer_chunked(snapshot, CharTokenizer(), backend, config=CONFIG, cache_dir=tmp_path)
    assert backend.calls == 0
