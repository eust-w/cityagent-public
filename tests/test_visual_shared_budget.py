"""Visual requests and derived text share one immutable inference budget."""

from copy import deepcopy

import pytest
from test_chunked_integration import CharTokenizer, FragmentBackend

from bridge_evidence.contest import chunked_inference as ci
from bridge_evidence.contest.candidate import validate_config
from bridge_evidence.contest.long_document import DocumentSnapshot
from bridge_evidence.contest.rsi import StrategyBundle
from bridge_evidence.contest.rsi_runtime import REPORT_DEADLINE, TaskJournal
from bridge_evidence.contest.rsi_sandbox import PolicyController

CONFIG = validate_config({"input_budget": 3000, "output_budget": 500})


def prefix(**overrides):
    return {
        "identity": "a" * 64, "initial_text_tasks": 1, "initial_visual_jobs": 1,
        "total_call_limit": 10, "used_calls": 1, "elapsed_seconds": 200.0,
        **overrides,
    }


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
def test_visual_time_and_inherited_deadline_apply_to_text_without_context_leak(monkeypatch, mode, outer):
    clock, observed, backend = Clock(), [], FragmentBackend()
    monkeypatch.setattr(ci, "time", clock)

    def generate(ids, maximum):
        observed.append(REPORT_DEADLINE.get())
        return backend(ids, maximum)

    token = REPORT_DEADLINE.set(outer)
    try:
        result = ci.infer_chunked(
            DocumentSnapshot.from_text("测试桥"), CharTokenizer(), generate,
            config=CONFIG, policy=policy_for(mode), prefix_budget=prefix(),
        )
        assert observed == [min(1100, outer) if outer is not None else 1100]
        assert REPORT_DEADLINE.get() == outer
        # Only the text response enters text model_calls; integration owns visual receipts.
        assert len(result["model_calls"]) == 1
        assert result["shared_budget"]["text_issued_calls"] == 1
        assert result["shared_budget"]["total_issued_calls"] == 2
        assert result["shared_budget"]["total_elapsed_seconds"] == 200
    finally:
        REPORT_DEADLINE.reset(token)


@pytest.mark.parametrize("mode", ["off", "frozen"])
def test_visual_derived_text_and_splits_cannot_expand_predeclared_call_limit(mode):
    snapshot = DocumentSnapshot.from_text("\n".join("裂" * 600 for _ in range(8)))
    assert len(ci.prepare_chunk_plan(snapshot, CharTokenizer(), CONFIG).tasks) > 1
    calls = []

    def truncated(ids, maximum):
        calls.append(ids)
        return {"text": "{", "finish_reason": "length", "output_tokens": maximum}

    with pytest.raises(ci.ChunkFailure) as failure:
        ci.infer_chunked(
            snapshot, CharTokenizer(), truncated, config=CONFIG,
            policy=policy_for(mode), prefix_budget=prefix(),
        )
    # N was one original text task + one visual job, not len(derived text plan).
    assert len(calls) == 9
    budget = failure.value.partial_evidence["shared_budget"]
    assert budget["total_issued_calls"] == budget["total_call_limit"] == 10


@pytest.mark.parametrize("mode", ["off", "frozen"])
def test_visual_time_at_deadline_never_starts_text_model_request(mode):
    backend = FragmentBackend()
    with pytest.raises(ci.ChunkFailure):
        ci.infer_chunked(
            DocumentSnapshot.from_text("测试桥"), CharTokenizer(), backend,
            config=CONFIG, policy=policy_for(mode),
            prefix_budget=prefix(elapsed_seconds=1200),
        )
    assert backend.calls == 0


@pytest.mark.parametrize("bad", [
    {"identity": "not-a-sha"},
    {"initial_text_tasks": True}, {"initial_text_tasks": -1},
    {"initial_visual_jobs": 0}, {"used_calls": -1}, {"used_calls": 2},
    {"used_calls": True}, {"total_call_limit": 11},
    {"elapsed_seconds": float("nan")}, {"elapsed_seconds": float("inf")},
    {"elapsed_seconds": -1}, {"elapsed_seconds": True}, {"elapsed_seconds": 1201},
    {"labels": "not-allowed"},
])
def test_invalid_prefix_is_rejected_before_checkpoint_or_request(tmp_path, bad):
    backend = FragmentBackend()
    with pytest.raises(ValueError, match="prefix"):
        ci.infer_chunked(
            DocumentSnapshot.from_text("测试桥"), CharTokenizer(), backend,
            config=CONFIG, prefix_budget=prefix(**bad), cache_dir=tmp_path,
        )
    assert backend.calls == 0 and not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["off", "frozen"])
def test_resume_subtracts_visual_once_and_separate_saved_text_time(monkeypatch, tmp_path, mode):
    clock, backend, observed = Clock(), FragmentBackend(), []
    monkeypatch.setattr(ci, "time", clock)
    snapshot = DocumentSnapshot.from_text("测试桥\n" + "裂" * 500 + "\n" + "裂" * 500)
    assert len(ci.prepare_chunk_plan(snapshot, CharTokenizer(), CONFIG).tasks) == 2
    budget = prefix(initial_text_tasks=2, total_call_limit=14)

    def first(ids, maximum):
        if backend.calls:
            raise KeyboardInterrupt("second text request interrupted")
        result = backend(ids, maximum)
        clock.now += 300
        return result

    with pytest.raises(KeyboardInterrupt):
        ci.infer_chunked(
            snapshot, CharTokenizer(), first, config=CONFIG,
            policy=policy_for(mode), cache_dir=tmp_path, prefix_budget=budget,
        )
    checkpoint = next(tmp_path.glob("*/checkpoint.sqlite"))
    with TaskJournal(checkpoint, checkpoint.parent.name) as journal:
        saved = journal.load()
    assert saved["spent"] == 300
    assert saved["prefix_budget"]["elapsed_seconds"] == 200
    # Visual replay adds only its new validation overhead. Saved 200 seconds is
    # not rerun, added to text spent, or subtracted twice.
    budget = {**budget, "elapsed_seconds": 225}
    clock.now += 25

    def resumed(ids, maximum):
        observed.append(REPORT_DEADLINE.get() - clock.monotonic())
        return backend(ids, maximum)

    result = ci.infer_chunked(
        snapshot, CharTokenizer(), resumed, config=CONFIG,
        policy=policy_for(mode), cache_dir=tmp_path, prefix_budget=budget,
    )
    assert observed == [675]  # 1200 - 225 visual - 300 prior text.
    assert backend.calls == 2  # First successful text fragment is reused.
    assert result["shared_budget"]["total_elapsed_seconds"] == 525
    assert result["shared_budget"]["total_issued_calls"] == 4  # Visual + 3 text reservations.
    assert len(list(tmp_path.glob("*/checkpoint.sqlite"))) == 1


def test_interruption_gap_outside_visual_recovery_remains_charged(monkeypatch, tmp_path):
    clock, backend, observed = Clock(), FragmentBackend(), []
    monkeypatch.setattr(ci, "time", clock)
    snapshot = DocumentSnapshot.from_text("测试桥\n" + "裂" * 500 + "\n" + "裂" * 500)
    budget = prefix(initial_text_tasks=2, total_call_limit=14)

    def first(ids, maximum):
        if backend.calls:
            raise KeyboardInterrupt("second text request interrupted")
        response = backend(ids, maximum)
        clock.now += 300
        return response

    with pytest.raises(KeyboardInterrupt):
        ci.infer_chunked(snapshot, CharTokenizer(), first, config=CONFIG,
                         cache_dir=tmp_path, prefix_budget=budget)
    clock.now += 125  # 100 seconds downtime, then 25 seconds visual replay.

    def resumed(ids, maximum):
        observed.append(REPORT_DEADLINE.get() - clock.monotonic())
        return backend(ids, maximum)

    result = ci.infer_chunked(
        snapshot, CharTokenizer(), resumed, config=CONFIG, cache_dir=tmp_path,
        prefix_budget={**budget, "elapsed_seconds": 225},
    )
    assert observed == [575]  # 1200 - 200 visual - 300 text - 125 interruption.
    assert result["shared_budget"]["total_elapsed_seconds"] == 625


@pytest.mark.parametrize("mode", ["off", "frozen"])
def test_completed_text_cache_reuses_responses_without_visual_reissue(monkeypatch, tmp_path, mode):
    clock = Clock()
    monkeypatch.setattr(ci, "time", clock)
    snapshot = DocumentSnapshot.from_text("测试桥")
    first = ci.infer_chunked(
        snapshot, CharTokenizer(), FragmentBackend(), config=CONFIG,
        policy=policy_for(mode), cache_dir=tmp_path, prefix_budget=prefix(),
    )
    backend = FragmentBackend()
    second = ci.infer_chunked(
        snapshot, CharTokenizer(), backend, config=CONFIG,
        policy=policy_for(mode), cache_dir=tmp_path, prefix_budget=prefix(elapsed_seconds=210),
    )
    assert backend.calls == 0
    assert first["model_calls"] == second["model_calls"]
    assert second["shared_budget"]["total_issued_calls"] == 2
    assert second["shared_budget"]["total_elapsed_seconds"] == 210


@pytest.mark.parametrize("change", [{"elapsed_seconds": 199}, {"used_calls": 0}])
def test_recovery_cannot_reset_visual_usage(tmp_path, change):
    snapshot = DocumentSnapshot.from_text("测试桥")
    ci.infer_chunked(snapshot, CharTokenizer(), FragmentBackend(), config=CONFIG,
                     cache_dir=tmp_path, prefix_budget=prefix())
    backend = FragmentBackend()
    with pytest.raises(ValueError, match="reset"):
        ci.infer_chunked(snapshot, CharTokenizer(), backend, config=CONFIG,
                         cache_dir=tmp_path, prefix_budget=prefix(**change))
    assert backend.calls == 0


def test_dynamic_checkpoint_issued_calls_still_include_visual_reservations(tmp_path):
    snapshot = DocumentSnapshot.from_text("测试桥")
    ci.infer_chunked(snapshot, CharTokenizer(), FragmentBackend(), config=CONFIG,
                     cache_dir=tmp_path, prefix_budget=prefix())
    checkpoint = next(tmp_path.glob("*/checkpoint.sqlite"))
    with TaskJournal(checkpoint, checkpoint.parent.name) as journal:
        journal.update(fields={"issued": 10})  # 10 text + 1 visual > fixed total 10.
    backend = FragmentBackend()
    with pytest.raises(ci.ChunkFailure, match="checkpoint budgets"):
        ci.infer_chunked(snapshot, CharTokenizer(), backend, config=CONFIG,
                         cache_dir=tmp_path, prefix_budget=prefix())
    assert backend.calls == 0


def test_changed_prefix_plan_never_reuses_text_cache(tmp_path):
    snapshot = DocumentSnapshot.from_text("测试桥")
    first = ci.infer_chunked(snapshot, CharTokenizer(), FragmentBackend(), config=CONFIG,
                            cache_dir=tmp_path, prefix_budget=prefix())
    backend = FragmentBackend()
    second = ci.infer_chunked(snapshot, CharTokenizer(), backend, config=CONFIG,
                             cache_dir=tmp_path, prefix_budget=prefix(identity="b" * 64))
    assert backend.calls == 1
    assert first["model_calls"][0]["call_id"] != second["model_calls"][0]["call_id"]
    assert len(list(tmp_path.glob("*/checkpoint.sqlite"))) == 2


def test_missing_or_modified_checkpoint_prefix_is_rejected(tmp_path):
    snapshot = DocumentSnapshot.from_text("测试桥")
    ci.infer_chunked(snapshot, CharTokenizer(), FragmentBackend(), config=CONFIG,
                     cache_dir=tmp_path, prefix_budget=prefix())
    checkpoint = next(tmp_path.glob("*/checkpoint.sqlite"))
    with TaskJournal(checkpoint, checkpoint.parent.name) as journal:
        journal.update(fields={"prefix_budget": None})
    backend = FragmentBackend()
    with pytest.raises(ValueError, match="prefix"):
        ci.infer_chunked(snapshot, CharTokenizer(), backend, config=CONFIG,
                         cache_dir=tmp_path, prefix_budget=prefix())
    assert backend.calls == 0


def test_none_prefix_preserves_existing_no_image_checkpoint_identity(tmp_path):
    snapshot = DocumentSnapshot.from_text("测试桥")
    first = ci.infer_chunked(snapshot, CharTokenizer(), FragmentBackend(), config=CONFIG,
                            cache_dir=tmp_path)
    backend = FragmentBackend()
    second = ci.infer_chunked(snapshot, CharTokenizer(), backend, config=CONFIG,
                             cache_dir=tmp_path, prefix_budget=None)
    assert backend.calls == 0 and first["model_calls"] == second["model_calls"]
    assert "shared_budget" not in second
    checkpoint = next(tmp_path.glob("*/checkpoint.sqlite"))
    with TaskJournal(checkpoint, checkpoint.parent.name) as journal:
        before = deepcopy(journal.load())
    assert "prefix_budget" not in before
