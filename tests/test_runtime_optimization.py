import copy
import json
import threading
import time

import pytest
from test_chunked_integration import CharTokenizer, FragmentBackend

from bridge_evidence.contest import chunked_inference as ci
from bridge_evidence.contest.candidate import validate_config
from bridge_evidence.contest.long_document import DocumentSnapshot, plan_chunks
from bridge_evidence.contest.rsi import StrategyBundle, base_action, validate_action
from bridge_evidence.contest.rsi_runtime import REPORT_DEADLINE, Dispatcher, TaskJournal
from bridge_evidence.contest.rsi_sandbox import PolicyController

CONFIG = validate_config({"input_budget": 3000, "output_budget": 500})


def test_repair_requires_two_reserved_calls():
    state = {
        "phase": "merge",
        "allow_repair": True,
        "react_calls": 0,
        "remaining_calls": 1,
        "remaining_seconds": 10,
    }
    assert base_action(state) == {"action": "fail"}
    with pytest.raises(ValueError, match="repair|budget"):
        validate_action({"action": "repair"}, state)


def test_repair_checks_deadline_before_second_request():
    calls = []

    def generate(ids, maximum):
        REPORT_DEADLINE.set(time.monotonic() - 1)
        return {
            "text": json.dumps(
                {"tool": "read_evidence", "arguments": {"source_ids": ["p0"]}}
            ),
            "output_tokens": 10,
            "finish_reason": "stop",
        }

    token = REPORT_DEADLINE.set(time.monotonic() + 60)
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            ci.bounded_repair(
                DocumentSnapshot.from_text("桥总体评分82"),
                {"conflicts": [{"field": "总体评分"}]},
                CharTokenizer(),
                generate,
                CONFIG,
                calls,
            )
        assert len(calls) == 1
    finally:
        REPORT_DEADLINE.reset(token)


def test_repair_persists_selection_and_resumes_without_reissuing_it():
    snapshot = DocumentSnapshot.from_text("桥总体评分82")
    merged = {
        "data": {"基本信息": {}},
        "conflicts": [{"field": "总体评分"}],
        "contributing_call_ids": [],
    }
    stages, durable, calls = {}, {}, []

    def persist():
        durable.clear()
        durable.update(copy.deepcopy(stages))
        if "react-tool-selection" in stages.get("responses", {}):
            raise KeyboardInterrupt("simulated crash after durable response")

    def selection(ids, maximum):
        return {
            "text": json.dumps(
                {"tool": "read_evidence", "arguments": {"source_ids": ["p0"]}}
            ),
            "output_tokens": 10,
            "finish_reason": "stop",
        }

    with pytest.raises(KeyboardInterrupt):
        ci.bounded_repair(
            snapshot,
            copy.deepcopy(merged),
            CharTokenizer(),
            selection,
            CONFIG,
            calls,
            checkpoint=stages,
            persist=persist,
        )
    count = 0

    def repair(ids, maximum):
        nonlocal count
        count += 1
        return {
            "text": json.dumps(
                {"resolved": {"总体评分": {"value": "82", "source_id": "p0"}}}
            ),
            "output_tokens": 10,
            "finish_reason": "stop",
        }

    result = ci.bounded_repair(
        snapshot,
        merged,
        CharTokenizer(),
        repair,
        CONFIG,
        calls,
        checkpoint=durable,
        persist=lambda: None,
    )
    assert count == 1
    assert result["data"]["基本信息"]["总体评分"] == "82"
    assert len(calls) == 2


def test_prepared_plan_avoids_replanning_and_checks_identity(monkeypatch):
    snapshot = DocumentSnapshot.from_text("测试桥\n裂缝")
    plan = plan_chunks(snapshot, CharTokenizer(), input_budget=3000, output_budget=500)
    monkeypatch.setattr(ci, "plan_chunks", lambda *a, **k: pytest.fail("replanned"))
    result = ci.infer_chunked(
        snapshot, CharTokenizer(), FragmentBackend(), config=CONFIG, prepared_plan=plan
    )
    assert result["coverage"]["complete"]
    plan.snapshot_sha256 = "0" * 64
    with pytest.raises(ValueError, match="plan"):
        ci.infer_chunked(
            snapshot,
            CharTokenizer(),
            FragmentBackend(),
            config=CONFIG,
            prepared_plan=plan,
        )


def test_checkpoint_recovery_does_not_replan(monkeypatch, tmp_path):
    snapshot = DocumentSnapshot.from_text("测试桥\n裂缝")
    bundle = StrategyBundle.baseline(CONFIG)
    first = ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        FragmentBackend(),
        config=CONFIG,
        policy=PolicyController(bundle),
        cache_dir=tmp_path,
    )
    monkeypatch.setattr(
        ci, "plan_chunks", lambda *a, **k: pytest.fail("replanned on recovery")
    )
    backend = FragmentBackend()
    resumed = ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        backend,
        config=CONFIG,
        policy=PolicyController(bundle),
        cache_dir=tmp_path,
    )
    assert backend.calls == 0
    assert resumed["model_calls"] == first["model_calls"]


def test_journal_delta_only_updates_changed_receipt(tmp_path):
    with TaskJournal(tmp_path / "state.sqlite", "same") as journal:
        state = {
            "tasks": [{"task_id": "a"}],
            "calls": [{"call_id": "a", "raw": "x" * 10000}],
            "pending": ["a"],
        }
        journal.save(state)
        queries = []
        journal.db.set_trace_callback(queries.append)
        journal.update(fields={"pending": []}, records={"calls": {1: {"call_id": "b"}}})
        assert not any("x" * 100 in query for query in queries)
        loaded = journal.load()
        assert loaded["calls"] == [state["calls"][0], {"call_id": "b"}]
        assert loaded["pending"] == []


def test_dispatch_refills_slot_before_slowest_sibling_finishes():
    snapshot = DocumentSnapshot.from_text("测试桥\n" + ("裂缝" * 700 + "\n") * 4)
    config = validate_config({**CONFIG, "batch_size": 2, "overlap": 0})
    plan = plan_chunks(
        snapshot, CharTokenizer(), input_budget=3000, output_budget=500, overlap=0
    )
    assert len(plan.tasks) >= 3
    slow_started, third_started = threading.Event(), threading.Event()
    lock = threading.Lock()
    count = 0
    backend = FragmentBackend()

    def generate(ids, maximum):
        nonlocal count
        with lock:
            count += 1
            current = count
        if current == 1:
            slow_started.set()
            assert third_started.wait(1), "batch barrier prevented refill"
        elif current >= 3:
            third_started.set()
        return backend(ids, maximum)

    with Dispatcher(generate, capacity=2) as dispatcher:
        result = ci.infer_chunked(snapshot, CharTokenizer(), dispatcher, config=config)
    assert result["coverage"]["complete"]
    assert result["model_calls"][0]["status"] == "success"


def test_repair_recovers_after_durable_response_before_validation(
    monkeypatch, tmp_path
):
    bundle = StrategyBundle.baseline({**CONFIG, "workflow": "bounded-react"})
    snapshot = DocumentSnapshot.from_text("测试桥总体评分82")
    merge = ci.merge_fragments

    def conflict(*args):
        result = merge(*args)
        result["conflicts"] = [{"field": "总体评分", "values": ["81", "82"]}]
        return result

    monkeypatch.setattr(ci, "merge_fragments", conflict)
    original_update = TaskJournal.update
    crashed = False

    def crashing_update(journal, **kwargs):
        nonlocal crashed
        original_update(journal, **kwargs)
        responses = kwargs.get("fields", {}).get("repair", {}).get("responses", {})
        if not crashed and "react-tool-selection" in responses:
            crashed = True
            raise KeyboardInterrupt("crash after selection response committed")

    monkeypatch.setattr(TaskJournal, "update", crashing_update)

    class Backend(FragmentBackend):
        roles = []

        def call_with_role(self, ids, maximum, *, role):
            self.roles.append(role)
            value = (
                {"tool": "read_evidence", "arguments": {"source_ids": ["p0"]}}
                if role == "react-tool-selection"
                else {"resolved": {"总体评分": {"value": "82", "source_id": "p0"}}}
            )
            return {
                "text": json.dumps(value),
                "output_tokens": 50,
                "finish_reason": "stop",
            }

    backend = Backend()
    with pytest.raises(KeyboardInterrupt):
        ci.infer_chunked(
            snapshot,
            CharTokenizer(),
            backend,
            config=bundle.config,
            policy=PolicyController(bundle),
            cache_dir=tmp_path,
        )
    result = ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        backend,
        config=bundle.config,
        policy=PolicyController(bundle),
        cache_dir=tmp_path,
    )
    assert backend.calls == 1
    assert backend.roles == ["react-tool-selection", "react-repair"]
    assert len(result["model_calls"]) == result["rsi"]["issued_calls"] == 3
    assert all(c["status"] == "success" for c in result["model_calls"])
    again = ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        backend,
        config=bundle.config,
        policy=PolicyController(bundle),
        cache_dir=tmp_path,
    )
    assert again["model_calls"] == result["model_calls"]
    assert backend.roles == ["react-tool-selection", "react-repair"]


def test_sibling_success_survives_failure_and_is_not_regenerated(tmp_path):
    snapshot = DocumentSnapshot.from_text("测试桥\n" + ("裂缝" * 700 + "\n") * 2)
    bundle = StrategyBundle.baseline({**CONFIG, "batch_size": 2, "overlap": 0})
    calls = []
    backend = FragmentBackend()

    def generate(ids, maximum):
        text = "".join(map(chr, ids))
        payload = json.JSONDecoder().raw_decode(text[text.index('{"owned":') :])[0]
        key = payload["owned"][0]["key"]
        calls.append(key)
        if len(calls) == 1:
            raise RuntimeError("permanent backend failure")
        return backend(ids, maximum)

    with Dispatcher(generate, capacity=2) as dispatcher:
        with pytest.raises(ci.ChunkFailure):
            ci.infer_chunked(
                snapshot,
                CharTokenizer(),
                dispatcher,
                config=bundle.config,
                policy=PolicyController(bundle),
                cache_dir=tmp_path,
                max_attempts=1,
            )
    stored = [
        json.loads(p.read_text())
        for p in tmp_path.rglob("*.json")
        if p.parent.name != "raw-responses"
    ]
    assert len(stored) == 1
    successful = stored[0]["call"]["source_ranges"][0]
    assert calls.count(successful) == 1
    with pytest.raises(ci.ChunkFailure):
        ci.infer_chunked(
            snapshot,
            CharTokenizer(),
            generate,
            config=bundle.config,
            policy=PolicyController(bundle),
            cache_dir=tmp_path,
            max_attempts=1,
        )
    assert calls.count(successful) == 1


def test_legacy_journal_migration_is_atomic_and_preserves_empty_lists(tmp_path):
    import sqlite3

    path = tmp_path / "state.sqlite"
    old = {"calls": [], "tasks": [{"task_id": "parent"}], "pending": ["parent"]}
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE checkpoint (id INTEGER PRIMARY KEY, identity TEXT, state TEXT)"
        )
        db.execute("INSERT INTO checkpoint VALUES (1, ?, ?)", ("same", json.dumps(old)))
    with TaskJournal(path, "same") as journal:
        assert journal.load() == old
        journal.update(fields={"pending": []})
        assert journal.load() == {**old, "pending": []}


def test_recovered_inflight_call_retains_failure_evidence(tmp_path):
    snapshot = DocumentSnapshot.from_text("测试桥")
    bundle = StrategyBundle.baseline(CONFIG)

    def crash(ids, maximum):
        raise KeyboardInterrupt("lost process")

    with pytest.raises(KeyboardInterrupt):
        ci.infer_chunked(
            snapshot,
            CharTokenizer(),
            crash,
            config=CONFIG,
            policy=PolicyController(bundle),
            cache_dir=tmp_path,
        )
    result = ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        FragmentBackend(),
        config=CONFIG,
        policy=PolicyController(bundle),
        cache_dir=tmp_path,
    )
    assert result["model_calls"][0]["errors"]
    assert "interrupted" in result["model_calls"][0]["errors"][0]
    assert result["model_calls"][0]["seconds"] > 0
    assert result["rsi"]["issued_calls"] == 2


def test_prepared_plan_rejects_forged_header_context_before_model_request():
    from dataclasses import replace

    snapshot = DocumentSnapshot.from_text("测试桥\n裂缝")
    plan = ci.prepare_chunk_plan(snapshot, CharTokenizer(), CONFIG)
    part = plan.tasks[0].owned[0]
    plan.tasks[0].owned[0] = replace(
        part, metadata={**part.metadata, "header": ["伪造评分99"]}
    )
    backend = FragmentBackend()
    with pytest.raises(ValueError, match="metadata|source"):
        ci.infer_chunked(
            snapshot, CharTokenizer(), backend, config=CONFIG, prepared_plan=plan
        )
    assert backend.calls == 0


def test_dispatch_request_ids_are_bound_to_durable_reservations(tmp_path):
    from bridge_evidence.contest.rsi_runtime import REQUEST_CALL_ID

    seen = []
    backend = FragmentBackend()

    def generate(ids, maximum):
        seen.append(REQUEST_CALL_ID.get())
        return backend(ids, maximum)

    with Dispatcher(generate, capacity=1) as dispatcher:
        result = ci.infer_chunked(
            DocumentSnapshot.from_text("测试桥"),
            CharTokenizer(),
            dispatcher,
            config=CONFIG,
            cache_dir=tmp_path,
        )
    assert seen == [c["call_id"] for c in result["model_calls"]]
    assert seen[0]


def test_rsi_off_recovery_reuses_dynamic_split_graph_without_replan(
    tmp_path, monkeypatch
):
    snapshot = DocumentSnapshot.from_text("测试桥\n裂缝")
    first = ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        FragmentBackend(length_once=True),
        config=CONFIG,
        cache_dir=tmp_path,
    )
    monkeypatch.setattr(ci, "plan_chunks", lambda *a, **k: pytest.fail("replanned"))
    backend = FragmentBackend()
    result = ci.infer_chunked(
        snapshot, CharTokenizer(), backend, config=CONFIG, cache_dir=tmp_path
    )
    assert backend.calls == 0
    assert result["chunk_plan"] == first["chunk_plan"]


def test_success_cache_recovers_response_lost_before_checkpoint_commit(
    tmp_path, monkeypatch
):
    original_update = TaskJournal.update
    crashed = False

    def crash_before_commit(journal, **kwargs):
        nonlocal crashed
        if not crashed and kwargs.get("records", {}).get("fragments"):
            crashed = True
            raise KeyboardInterrupt("process dies after cache publication")
        return original_update(journal, **kwargs)

    monkeypatch.setattr(TaskJournal, "update", crash_before_commit)
    snapshot = DocumentSnapshot.from_text("测试桥")
    backend = FragmentBackend()
    with pytest.raises(KeyboardInterrupt):
        ci.infer_chunked(
            snapshot, CharTokenizer(), backend, config=CONFIG, cache_dir=tmp_path
        )
    result = ci.infer_chunked(
        snapshot, CharTokenizer(), backend, config=CONFIG, cache_dir=tmp_path
    )
    assert backend.calls == 1
    assert len(result["model_calls"]) == 1
    assert result["model_calls"][0]["status"] == "success"
    assert result["model_calls"][0]["errors"] == []


def test_repair_rejects_cross_scope_resolution_before_backend():
    snapshot = DocumentSnapshot.from_text("左幅评分81，右幅评分82")
    merged = {
        "conflicts": [
            {
                "field": "总体评分",
                "observations": [
                    {"scope": {"span": "左幅"}},
                    {"scope": {"span": "右幅"}},
                ],
            }
        ]
    }
    backend = FragmentBackend()
    with pytest.raises(ValueError, match="scope"):
        ci.bounded_repair(snapshot, merged, CharTokenizer(), backend, CONFIG, [])
    assert backend.calls == 0


def test_repair_records_exact_evidence_without_rewriting_observations():
    snapshot = DocumentSnapshot.from_text("桥总体评分82")
    observations = [
        {"field": "总体评分", "value": "81", "scope": {}},
        {"field": "总体评分", "value": "82", "scope": {}},
    ]
    merged = {
        "data": {"基本信息": {}},
        "conflicts": [{"field": "总体评分", "observations": observations}],
        "basic_observations": copy.deepcopy(observations),
        "contributing_call_ids": [],
    }
    responses = iter(
        [
            {"tool": "read_evidence", "arguments": {"source_ids": ["p0"]}},
            {"resolved": {"总体评分": {"value": "82", "source_id": "p0"}}},
        ]
    )

    def generate(ids, maximum):
        return {
            "text": json.dumps(next(responses)),
            "finish_reason": "stop",
            "output_tokens": 10,
        }

    result = ci.bounded_repair(snapshot, merged, CharTokenizer(), generate, CONFIG, [])
    assert result["basic_observations"] == observations
    repair = result["repair_evidence"][0]
    assert repair["value"] == "82"
    assert repair["evidence"]["text"] == "82"
    assert (
        snapshot.sources()["p0"].text[
            repair["evidence"]["start"] : repair["evidence"]["end"]
        ]
        == "82"
    )
    assert len(repair["supersedes"]) == 2


def test_repair_cannot_introduce_an_unsupported_value():
    responses = iter(
        [
            {"tool": "read_evidence", "arguments": {"source_ids": ["p0"]}},
            {"resolved": {"总体评分": {"value": "99", "source_id": "p0"}}},
        ]
    )

    def generate(ids, maximum):
        return {
            "text": json.dumps(next(responses)),
            "finish_reason": "stop",
            "output_tokens": 10,
        }

    with pytest.raises(ValueError, match="source|support"):
        ci.bounded_repair(
            DocumentSnapshot.from_text("桥总体评分82"),
            {
                "data": {"基本信息": {}},
                "conflicts": [{"field": "总体评分"}],
                "contributing_call_ids": [],
            },
            CharTokenizer(),
            generate,
            CONFIG,
            [],
        )


def test_changed_attempt_budget_does_not_reuse_checkpoint(tmp_path):
    snapshot = DocumentSnapshot.from_text("测试桥")
    backend = FragmentBackend()
    ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        backend,
        config=CONFIG,
        cache_dir=tmp_path,
        max_attempts=1,
    )
    ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        backend,
        config=CONFIG,
        cache_dir=tmp_path,
        max_attempts=2,
    )
    assert backend.calls == 2


def test_repair_receipt_binds_visible_evidence_ranges():
    snapshot = DocumentSnapshot.from_text("桥总体评分82")
    calls = []
    responses = iter(
        [
            {"tool": "read_evidence", "arguments": {"source_ids": ["p0"]}},
            {"resolved": {"总体评分": {"value": "82", "source_id": "p0"}}},
        ]
    )

    def generate(ids, maximum):
        return {
            "text": json.dumps(next(responses)),
            "finish_reason": "stop",
            "output_tokens": 10,
        }

    ci.bounded_repair(
        snapshot,
        {
            "data": {"基本信息": {}},
            "conflicts": [{"field": "总体评分"}],
            "contributing_call_ids": [],
        },
        CharTokenizer(),
        generate,
        CONFIG,
        calls,
    )
    assert calls[-1]["source_ranges"] == ["p0:0:7"]


def test_direct_accounted_backend_recovery_uses_its_own_ledger(tmp_path):
    from bridge_evidence.contest.backends import AccountedBackend
    from bridge_evidence.contest.timing import TimingLedger

    class InterruptedBatch:
        def batch(self, requests):
            raise KeyboardInterrupt("local model process interrupted")

    class SuccessfulBatch(FragmentBackend):
        def batch(self, requests):
            return [self(*request) for request in requests]

    snapshot = DocumentSnapshot.from_text("测试桥")
    ledger = TimingLedger(tmp_path / "timing.sqlite")
    cache = tmp_path / "cache"
    with pytest.raises(KeyboardInterrupt):
        ci.infer_chunked(
            snapshot,
            CharTokenizer(),
            AccountedBackend(InterruptedBatch(), ledger),
            config=CONFIG,
            cache_dir=cache,
        )
    with ledger.connect() as db:
        interrupted_id, interrupted_seconds = db.execute(
            "SELECT id,seconds FROM calls"
        ).fetchone()
    result = ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        AccountedBackend(SuccessfulBatch(), ledger),
        config=CONFIG,
        cache_dir=cache,
    )
    interrupted = next(
        call for call in result["model_calls"] if call["call_id"] == interrupted_id
    )
    assert interrupted["timing_source"] == "durable-ledger"
    assert interrupted["seconds"] == interrupted_seconds


@pytest.mark.parametrize(
    "selected_text",
    [
        "2021年右幅总体评分88",
        "2020年左幅桥面系评分88",
        "总体评分88",
        "2020年左幅总体评分80，2021年右幅总体评分88",
        "2020年左幅总体评分80及2021年右幅总体评分88",
        "2020年左幅总体评分80，裂缝数量88",
    ],
)
def test_repair_requires_field_and_inherited_scope_in_same_claim(selected_text):
    snapshot = DocumentSnapshot.from_text(
        "2020年左幅总体评分80\n2020年左幅总体评分81\n" + selected_text
    )
    observations = [
        {"field": "总体评分", "value": value, "scope": {"year": 2020, "span": "左幅"}}
        for value in ["80", "81"]
    ]
    merged = {
        "data": {"基本信息": {"总体评分": "80"}},
        "conflicts": [{"field": "总体评分", "observations": observations}],
        "basic_observations": copy.deepcopy(observations),
        "contributing_call_ids": [],
    }
    before = copy.deepcopy(merged)
    responses = iter(
        [
            {"tool": "read_evidence", "arguments": {"source_ids": ["p2"]}},
            {"resolved": {"总体评分": {"value": "88", "source_id": "p2"}}},
        ]
    )

    def generate(ids, maximum):
        return {
            "text": json.dumps(next(responses)),
            "finish_reason": "stop",
            "output_tokens": 10,
        }

    with pytest.raises(ValueError, match="scope|field|support"):
        ci.bounded_repair(snapshot, merged, CharTokenizer(), generate, CONFIG, [])
    assert merged == before


def test_repair_accepts_supported_same_scope_and_saves_support_quote():
    snapshot = DocumentSnapshot.from_text("2020年左幅总体评分88")
    observations = [
        {"field": "总体评分", "value": v, "scope": {"year": 2020, "span": "左幅"}}
        for v in ["80", "81"]
    ]
    merged = {
        "data": {"基本信息": {}},
        "conflicts": [{"field": "总体评分", "observations": observations}],
        "contributing_call_ids": [],
    }
    responses = iter(
        [
            {"tool": "read_evidence", "arguments": {"source_ids": ["p0"]}},
            {"resolved": {"总体评分": {"value": "88", "source_id": "p0"}}},
        ]
    )

    def generate(ids, maximum):
        return {
            "text": json.dumps(next(responses)),
            "finish_reason": "stop",
            "output_tokens": 10,
        }

    result = ci.bounded_repair(snapshot, merged, CharTokenizer(), generate, CONFIG, [])
    assert result["repair_evidence"][0]["support_quote"] == "2020年左幅总体评分88"
    assert result["repair_evidence"][0]["scope"] == {"year": 2020, "span": "左幅"}


@pytest.mark.parametrize("failure_type", ["backend", "length"])
def test_repair_failure_preserves_accounted_backend_latency(failure_type):
    calls = []

    def generate(ids, maximum):
        if failure_type == "backend":
            error = TimeoutError("timed backend failure")
            error.latency_seconds = 0.125
            raise error
        return {
            "text": "{",
            "finish_reason": "length",
            "output_tokens": maximum,
            "latency_seconds": 0.125,
        }

    with pytest.raises((TimeoutError, ValueError)):
        ci.bounded_repair(
            DocumentSnapshot.from_text("总体评分82"),
            {"conflicts": [{"field": "总体评分"}]},
            CharTokenizer(),
            generate,
            CONFIG,
            calls,
        )
    assert calls[0]["seconds"] == 0.125
    assert calls[0]["latency_seconds"] == 0.125


def test_failed_model_response_has_bound_raw_receipt_outside_success_cache(tmp_path):
    raw = '{"基本信息":{"桥梁名称":"测试'

    def generate(ids, maximum):
        return {"text": raw, "finish_reason": "stop", "output_tokens": 30}

    with pytest.raises(ci.ChunkFailure) as failed:
        ci.infer_chunked(
            DocumentSnapshot.from_text("测试桥"),
            CharTokenizer(),
            generate,
            config=CONFIG,
            cache_dir=tmp_path,
            max_attempts=1,
        )
    call = failed.value.model_calls[0]
    receipts = list(tmp_path.rglob("raw-responses/*.json"))
    assert len(receipts) == 1
    stored = json.loads(receipts[0].read_text())
    assert stored["raw"] == raw
    assert stored["output_sha256"] == call["output_sha256"]
    assert stored["call_id"] == call["call_id"]
    assert stored["source_ranges"] == call["source_ranges"]
    assert call["response_validation"] == "failed"
    assert not list(receipts[0].parent.parent.glob("*.json"))


def test_raw_response_resumes_after_crash_before_parse_without_another_call(
    tmp_path, monkeypatch
):
    original = ci.parse_fragment
    interrupted = False

    def interrupted_parse(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("crash before model response validation")
        return original(*args, **kwargs)

    monkeypatch.setattr(ci, "parse_fragment", interrupted_parse)
    snapshot = DocumentSnapshot.from_text("测试桥")
    backend = FragmentBackend()
    with pytest.raises(KeyboardInterrupt):
        ci.infer_chunked(
            snapshot,
            CharTokenizer(),
            backend,
            config=CONFIG,
            cache_dir=tmp_path,
            max_attempts=1,
        )
    result = ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        backend,
        config=CONFIG,
        cache_dir=tmp_path,
        max_attempts=1,
    )
    assert backend.calls == 1
    assert len(result["model_calls"]) == 1
    assert result["model_calls"][0]["response_validation"] == "success"
    assert result["coverage"]["complete"]


def test_raw_response_journal_gap_does_not_regenerate(tmp_path, monkeypatch):
    original = TaskJournal.update
    interrupted = False

    def stop_before_receipt_commit(journal, **kwargs):
        nonlocal interrupted
        records = kwargs.get("records", {}).get("calls", {})
        if not interrupted and any(
            c.get("response_validation") == "pending" for c in records.values()
        ):
            interrupted = True
            raise KeyboardInterrupt("raw durable but checkpoint not committed")
        return original(journal, **kwargs)

    monkeypatch.setattr(TaskJournal, "update", stop_before_receipt_commit)
    snapshot = DocumentSnapshot.from_text("测试桥")
    backend = FragmentBackend()
    with pytest.raises(KeyboardInterrupt):
        ci.infer_chunked(
            snapshot,
            CharTokenizer(),
            backend,
            config=CONFIG,
            cache_dir=tmp_path,
            max_attempts=1,
        )
    result = ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        backend,
        config=CONFIG,
        cache_dir=tmp_path,
        max_attempts=1,
    )
    assert backend.calls == 1
    assert result["model_calls"][0]["status"] == "success"


def test_oversized_failed_raw_is_bounded_and_cannot_be_replayed_as_success(tmp_path):
    raw = "x" * 70000
    calls = 0

    def generate(ids, maximum):
        nonlocal calls
        calls += 1
        return {"text": raw, "finish_reason": "stop", "output_tokens": 10}

    snapshot = DocumentSnapshot.from_text("测试桥")
    with pytest.raises(ci.ChunkFailure) as failed:
        ci.infer_chunked(
            snapshot,
            CharTokenizer(),
            generate,
            config=CONFIG,
            cache_dir=tmp_path,
            max_attempts=1,
        )
    receipt = json.loads(next(tmp_path.rglob("raw-responses/*.json")).read_text())
    assert receipt["truncated"] is True
    assert len(receipt["raw"].encode()) <= 65536
    assert receipt["output_sha256"] == failed.value.model_calls[0]["output_sha256"]
    with pytest.raises(ci.ChunkFailure):
        ci.infer_chunked(
            snapshot,
            CharTokenizer(),
            generate,
            config=CONFIG,
            cache_dir=tmp_path,
            max_attempts=1,
        )
    assert calls == 1


def test_completed_raw_can_be_validated_after_model_deadline_without_new_request(
    tmp_path, monkeypatch
):
    snapshot = DocumentSnapshot.from_text("测试桥")
    bundle = StrategyBundle.baseline(CONFIG)
    original = ci.parse_fragment

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("before parse")

    monkeypatch.setattr(ci, "parse_fragment", interrupted)
    backend = FragmentBackend()
    with pytest.raises(KeyboardInterrupt):
        ci.infer_chunked(
            snapshot,
            CharTokenizer(),
            backend,
            config=CONFIG,
            cache_dir=tmp_path,
            policy=PolicyController(bundle),
            max_attempts=1,
        )
    path = next(tmp_path.rglob("checkpoint.sqlite"))
    import sqlite3

    with sqlite3.connect(path) as db:
        identity = db.execute("SELECT identity FROM checkpoint").fetchone()[0]
    with TaskJournal(path, identity) as journal:
        journal.update(fields={"spent": 1201.0})
    monkeypatch.setattr(ci, "parse_fragment", original)
    result = ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        backend,
        config=CONFIG,
        cache_dir=tmp_path,
        policy=PolicyController(bundle),
        max_attempts=1,
    )
    assert backend.calls == 1
    assert result["coverage"]["complete"]


def test_tampered_unvalidated_raw_cannot_trigger_regeneration_or_success(
    tmp_path, monkeypatch
):
    snapshot = DocumentSnapshot.from_text("测试桥")
    backend = FragmentBackend()
    original = ci.parse_fragment
    monkeypatch.setattr(
        ci, "parse_fragment", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        ci.infer_chunked(
            snapshot, CharTokenizer(), backend, config=CONFIG, cache_dir=tmp_path
        )
    path = next(tmp_path.rglob("raw-responses/*.json"))
    raw = json.loads(path.read_text())
    raw["raw"] = raw["raw"].replace("测试桥", "替换桥")
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(ci, "parse_fragment", original)
    with pytest.raises(ci.ChunkFailure, match="raw response receipt"):
        ci.infer_chunked(
            snapshot, CharTokenizer(), backend, config=CONFIG, cache_dir=tmp_path
        )
    assert backend.calls == 1


def test_raw_response_replay_matches_actual_accounted_receipt(tmp_path, monkeypatch):
    from bridge_evidence.contest.backends import AccountedBackend
    from bridge_evidence.contest.timing import TimingLedger

    class Backend(FragmentBackend):
        def batch(self, requests):
            return [self(*request) for request in requests]

    raw_backend = Backend()
    ledger = TimingLedger(tmp_path / "timing.sqlite")
    backend = AccountedBackend(raw_backend, ledger)
    snapshot = DocumentSnapshot.from_text("测试桥")
    original = ci.parse_fragment
    monkeypatch.setattr(
        ci, "parse_fragment", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        ci.infer_chunked(
            snapshot,
            CharTokenizer(),
            backend,
            config=CONFIG,
            cache_dir=tmp_path / "cache",
            max_attempts=1,
        )
    monkeypatch.setattr(ci, "parse_fragment", original)
    result = ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        backend,
        config=CONFIG,
        cache_dir=tmp_path / "cache",
        max_attempts=1,
    )
    assert raw_backend.calls == ledger.summary()["calls"] == 1
    assert result["model_calls"][0]["status"] == "success"
    with ledger.connect() as db:
        seconds, receipt = db.execute("SELECT seconds,receipt FROM calls").fetchone()
    assert result["model_calls"][0]["seconds"] == seconds
    assert (
        json.loads(receipt)["output_sha256"]
        == result["model_calls"][0]["output_sha256"]
    )


@pytest.mark.parametrize("fault_type", [OSError, KeyboardInterrupt])
def test_fragment_cache_write_failure_reuses_raw_without_new_attempt(
    tmp_path, monkeypatch, fault_type
):
    """Raw is durable before a fragment write: storage failure is not model failure."""
    from bridge_evidence.contest.rsi_runtime import TaskJournal

    snapshot = DocumentSnapshot.from_text("测试桥")
    backend = FragmentBackend()
    original = ci.atomic_json

    def fail_fragment(path, value):
        if path.parent.name != "raw-responses":
            raise fault_type("synthetic fragment write interrupted")
        return original(path, value)

    with monkeypatch.context() as patcher:
        patcher.setattr(ci, "atomic_json", fail_fragment)
        with pytest.raises((ci.ChunkFailure, KeyboardInterrupt)):
            ci.infer_chunked(
                snapshot,
                CharTokenizer(),
                backend,
                config=CONFIG,
                max_attempts=1,
                cache_dir=tmp_path,
            )
    assert backend.calls == 1
    resumed = FragmentBackend()
    result = ci.infer_chunked(
        snapshot,
        CharTokenizer(),
        resumed,
        config=CONFIG,
        max_attempts=1,
        cache_dir=tmp_path,
    )
    assert resumed.calls == 0
    assert len(result["model_calls"]) == 1
    assert result["model_calls"][0]["status"] == "success"
    assert result["model_calls"][0]["errors"] == []
    checkpoint = next(tmp_path.glob("*/checkpoint.sqlite"))
    with TaskJournal(checkpoint, checkpoint.parent.name) as journal:
        state = journal.load()
    assert state["issued"] == 1
    assert [attempt["count"] for attempt in state["attempts"].values()] == [1]
    assert not state["pending"]
    assert len(state["fragments"]) == 1


def test_generation_evidence_keeps_actual_accepted_raw_on_resume(tmp_path):
    import hashlib

    snapshot = DocumentSnapshot.from_text("测试桥")
    first = FragmentBackend(invalid_once=True)
    result = ci.infer_chunked(
        snapshot, CharTokenizer(), first, config=CONFIG, cache_dir=tmp_path
    )
    evidence = result["generation_evidence"]
    assert evidence["schema_version"] == 1
    accepted = {
        call["call_id"]: call
        for call in result["model_calls"]
        if call["status"] == "success"
    }
    assert set(evidence["responses"]) == set(accepted)
    for call_id, raw in evidence["responses"].items():
        assert (
            hashlib.sha256(raw.encode()).hexdigest()
            == accepted[call_id]["output_sha256"]
        )
        assert json.loads(raw)["covered"]
    resumed_backend = FragmentBackend()
    resumed = ci.infer_chunked(
        snapshot, CharTokenizer(), resumed_backend, config=CONFIG, cache_dir=tmp_path
    )
    assert resumed_backend.calls == 0
    assert resumed["generation_evidence"] == evidence
