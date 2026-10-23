"""Deterministic metadata-only probes; no model service or network is contacted."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from bridge_evidence.contest.document_batch_quiescence import BatchServiceQuiescence


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, duration):
        assert duration == .5
        self.sleeps.append(duration)
        self.now += duration


def models(model="own-main", *, root="same-root", context=262144, parent=None):
    cards = [{"id": model, "parent": parent, "root": root, "max_model_len": context}]
    if parent is not None:
        cards[0].pop("max_model_len")
        cards.append({"id": parent, "parent": None, "max_model_len": context})
    return json.dumps({"data": cards}).encode()


def metrics(model="own-main", *, running=(0, 0), waiting=(0, 0)):
    return "\n".join(f'vllm:num_requests_{name}{{model_name="{model}",engine="{i}"}} {value}'
        for name, values in (("running", running), ("waiting", waiting))
        for i, value in enumerate(values)).encode()


def snapshot(pool="main"):
    return {"batch_id": "batch-1", "sequence": 8,
        "pools": {pool: {"active": 0, "unknown": 1}},
        "requests": [{"pool": pool, "request_id": "unknown-original", "state": "unknown",
                      "usage": None, "done_received": False, "cancel_status": "not_confirmed"}]}


class Fetch:
    def __init__(self, *, model="own-main", metric_bodies=None, model_bodies=None):
        self.model = model
        self.metric_bodies = metric_bodies or [metrics(model)]
        self.model_bodies = model_bodies or [models(model)]
        self.counts = {"models": 0, "metrics": 0}
        self.calls = []

    def __call__(self, url, *, timeout, headers):
        assert url.rsplit("/", 1)[-1] in {"models", "metrics"}
        assert "/v1/v1" not in url and 0 < timeout <= 2
        self.calls.append((url, timeout, dict(headers)))
        kind = url.rsplit("/", 1)[-1]
        bodies = self.model_bodies if kind == "models" else self.metric_bodies
        index = self.counts[kind]
        self.counts[kind] += 1
        return 200, bodies[min(index, len(bodies) - 1)]


@pytest.fixture
def setup_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("CITYAGENT_STOP_OBSERVER_SERVER_PID", "123")
    monkeypatch.setenv("CITYAGENT_SUMMARY_STOP_OBSERVER_SERVER_PID", "456")
    monkeypatch.setenv("MAIN_TEST_KEY", "main-secret-never-persist")
    monkeypatch.setenv("SUMMARY_TEST_KEY", "summary-secret-never-persist")
    config = {"base_url": "http://127.0.0.1:18110/v1", "model": "own-main", "api_key_env": "MAIN_TEST_KEY",
        "summary": {"base_url": "http://127.0.0.1:18284", "model": "own-summary", "api_key_env": "SUMMARY_TEST_KEY"}}
    created = []
    def make(*, fetch=None, clock=None, process=None, profile=None):
        clock = clock or Clock()
        fetch = fetch or Fetch()
        def identity(pid, port):
            created.append((pid, port))
            return {"pid": pid, "starttime_ticks": "100", "cmdline_sha256": "a" * 64}
        obj = BatchServiceQuiescence(profile or config, tmp_path, fetch=fetch, clock=clock,
            sleep=clock.sleep, process_identity=process or identity)
        return obj, fetch, clock
    return make, config, created


def test_confirmed_idle_persists_raw_proof_but_old_unknown_is_unchanged(setup_probe, tmp_path):
    make, _, pids = setup_probe
    probe, fetch, clock = make()
    assert pids == [(123, 18110), (456, 18284)] and fetch.calls == []
    original = snapshot()
    before = deepcopy(original)
    result = probe("main", 200, original)
    assert result["confirmed_idle"] is True and original == before
    proof = result["proof"]
    assert proof["same_service_process"] is True and proof["seconds"] == .5
    assert proof["cancellation_confirmed"] is False and proof["prior_requests_reclassified"] is False
    assert proof["request_completion_known"] is False and proof["usage_known"] is False
    assert len(proof["samples"]) == 2 and clock.sleeps == [.5]
    assert all(s["idle"] for s in proof["samples"])
    assert len(fetch.calls) == 4 and all(x[2]["Authorization"] == "Bearer main-secret-never-persist" for x in fetch.calls)
    for sample in proof["samples"]:
        for key in ("/v1/models", "/metrics"):
            ref = sample[key]
            assert hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    stored = Path(result["proof_ref"]["path"])
    assert json.loads(stored.read_text()) == proof
    assert hashlib.sha256(stored.read_bytes()).hexdigest() == result["proof_ref"]["sha256"]
    assert all(b"secret-never-persist" not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file())


def test_summary_uses_its_own_pinned_port_pid_key_and_model(setup_probe):
    make, _, pids = setup_probe
    probe, fetch, _ = make(fetch=Fetch(model="own-summary"))
    assert probe("summary", 200, snapshot("summary"))["confirmed_idle"]
    assert pids[2:] == [(456, 18284)] * 4
    assert all(x[0].startswith("http://127.0.0.1:18284/") for x in fetch.calls)
    assert all(x[2]["Authorization"] == "Bearer summary-secret-never-persist" for x in fetch.calls)


def test_busy_engine_and_queued_work_reset_consecutive_idle(setup_probe):
    make, _, _ = setup_probe
    probe, _, _ = make(fetch=Fetch(metric_bodies=[metrics(), metrics(running=(0, 1)),
        metrics(waiting=(0, 1)), metrics(), metrics()]))
    result = probe("main", 200, snapshot())
    assert result["confirmed_idle"]
    assert [r["consecutive_idle"] for r in result["proof"]["samples"]] == [1, 0, 0, 1, 2]


@pytest.mark.parametrize("deadline", [100, 100.49, 100.5, 100.7, 200])
def test_busy_or_insufficient_time_never_exceeds_deadline_or_ten_seconds(setup_probe, deadline):
    make, _, _ = setup_probe
    probe, fetch, clock = make(fetch=Fetch(metric_bodies=[metrics(waiting=(1, 0))]))
    result = probe("main", deadline, snapshot())
    assert not result["confirmed_idle"] and clock.now <= min(deadline, 110)
    assert all(timeout <= min(2, deadline - 100) for _, timeout, _ in fetch.calls)


def test_sleep_must_really_separate_samples(setup_probe):
    make, _, _ = setup_probe
    probe, _, clock = make()
    probe.sleep = lambda _: None
    result = probe("main", 200, snapshot())
    assert not result["confirmed_idle"] and clock.now == 100
    assert "at least 0.5" in result["proof"]["reason"]


@pytest.mark.parametrize("body", [b"no_metric 0", metrics(waiting=()),
    metrics(running=("NaN", 0)), metrics(running=(-1, 0)), metrics(running=(0,)),
    metrics() + b"\n" + metrics(), metrics(model="foreign-model")])
def test_bad_or_incomplete_engine_metrics_fail_closed_with_original_bytes(setup_probe, body):
    make, _, _ = setup_probe
    probe, _, _ = make(fetch=Fetch(metric_bodies=[body]))
    result = probe("main", 200, snapshot())
    assert not result["confirmed_idle"]
    assert Path(result["proof"]["samples"][0]["/metrics"]["path"]).read_bytes() == body


@pytest.mark.parametrize("body", [models(context=65536), models(model="other"), b"bad-json"])
def test_wrong_model_context_or_invalid_metadata_fail_closed(setup_probe, body):
    make, _, _ = setup_probe
    probe, _, _ = make(fetch=Fetch(model_bodies=[body]))
    result = probe("main", 200, snapshot())
    assert not result["confirmed_idle"]
    assert Path(result["proof"]["samples"][0]["/v1/models"]["path"]).read_bytes() == body


def test_lora_matches_actual_base_model_metrics_and_context(setup_probe):
    make, _, _ = setup_probe
    probe, _, _ = make(fetch=Fetch(model_bodies=[models(parent="base")], metric_bodies=[metrics(model="base")]))
    assert probe("main", 200, snapshot())["confirmed_idle"]


@pytest.mark.parametrize("change", ["model", "engine", "process_before", "process_during"])
def test_changed_identity_never_releases_capacity(setup_probe, change):
    make, _, _ = setup_probe
    fetch = Fetch(model_bodies=[models(), models(root="replacement")] if change == "model" else None,
        metric_bodies=[metrics(), metrics(running=(0,), waiting=(0,))] if change == "engine" else None)
    calls = 0
    def process(pid, port):
        nonlocal calls
        calls += 1
        changed = (change == "process_before" and calls >= 3) or (change == "process_during" and calls >= 4)
        return {"pid": pid, "starttime_ticks": "replacement" if changed else "original"}
    probe, _, _ = make(fetch=fetch, process=process)
    result = probe("main", 200, snapshot())
    assert not result["confirmed_idle"] and "changed" in result["proof"]["reason"]


@pytest.mark.parametrize("pid", ["", "bad", "0", "-1"])
def test_missing_or_invalid_pid_never_fetches(setup_probe, monkeypatch, pid):
    make, _, _ = setup_probe
    monkeypatch.setenv("CITYAGENT_STOP_OBSERVER_SERVER_PID", pid)
    probe, fetch, _ = make()
    result = probe("main", 200, snapshot())
    assert not result["confirmed_idle"] and fetch.calls == []
    assert "PID" in result["proof"]["reason"]


def test_pin_failure_is_not_repaired_by_later_environment_change(setup_probe, monkeypatch):
    make, _, _ = setup_probe
    monkeypatch.delenv("CITYAGENT_STOP_OBSERVER_SERVER_PID")
    probe, fetch, _ = make()
    monkeypatch.setenv("CITYAGENT_STOP_OBSERVER_SERVER_PID", "123")
    assert not probe("main", 200, snapshot())["confirmed_idle"] and not fetch.calls


def test_active_or_inconsistent_snapshot_is_not_observed(setup_probe):
    make, _, _ = setup_probe
    probe, fetch, _ = make()
    for field in ("active", "unknown"):
        value = snapshot()
        value["pools"]["main"][field] = 1 if field == "active" else 2
        assert not probe("main", 200, value)["confirmed_idle"]
    value = snapshot()
    value["requests"].append({"pool": "main", "state": "active"})
    assert not probe("main", 200, value)["confirmed_idle"]
    assert not fetch.calls


def test_http_error_preserves_response_and_late_success_stays_unknown(setup_probe):
    make, _, _ = setup_probe
    probe, _, clock = make(fetch=lambda *a, **k: (503, b"service-busy"))
    result = probe("main", 200, snapshot())
    sample = result["proof"]["samples"][0]["/v1/models"]
    assert not result["confirmed_idle"] and sample["http_status"] == 503
    assert Path(sample["path"]).read_bytes() == b"service-busy"
    def late(*args, **kwargs):
        clock.now += kwargs["timeout"] + .01
        return 200, models()
    probe.fetch = late
    result = probe("main", 100.2, snapshot())
    assert not result["confirmed_idle"] and "deadline" in result["proof"]["reason"].lower()


def test_fetch_exception_and_missing_summary_profile_fail_closed(setup_probe):
    make, config, _ = setup_probe
    def failed(*args, **kwargs):
        raise TimeoutError("metadata timeout")
    probe, _, _ = make(fetch=failed)
    assert not probe("main", 200, snapshot())["confirmed_idle"]
    config = {k: v for k, v in config.items() if k != "summary"}
    probe, fetch, _ = make(profile=config)
    assert not probe("summary", 200, snapshot("summary"))["confirmed_idle"] and not fetch.calls


def test_repeated_callbacks_have_distinct_proof_directories(setup_probe):
    make, _, _ = setup_probe
    probe, _, _ = make()
    one = probe("main", 200, snapshot())
    two = probe("main", 200, snapshot())
    assert one["confirmed_idle"] and two["confirmed_idle"]
    assert one["proof_ref"]["path"] != two["proof_ref"]["path"]
    assert Path(one["proof_ref"]["path"]).is_file()


@pytest.mark.parametrize("url", ["http://192.0.2.3:18110", "http://model.invalid:18110",
    "http://127.0.0.1:18110/v1/chat/completions", "http://user:pass@127.0.0.1:18110"])
def test_local_pid_cannot_prove_remote_or_unexpected_endpoint(setup_probe, url):
    make, config, _ = setup_probe
    config = deepcopy(config)
    config["base_url"] = url
    probe, fetch, _ = make(profile=config)
    assert not probe("main", 200, snapshot())["confirmed_idle"] and fetch.calls == []


def test_single_idle_sample_is_insufficient_near_deadline(setup_probe):
    make, _, _ = setup_probe
    probe, fetch, clock = make()
    result = probe("main", 100.5, snapshot())
    assert not result["confirmed_idle"] and len(fetch.calls) == 2 and clock.sleeps == []


def test_absent_process_is_recorded_as_pin_error_without_fetch(setup_probe):
    make, _, _ = setup_probe
    def absent(*args):
        raise FileNotFoundError("no such serving PID")
    probe, fetch, _ = make(process=absent)
    result = probe("main", 200, snapshot())
    assert not result["confirmed_idle"] and fetch.calls == []
    assert result["proof"]["service"]["pin_error_type"] == "FileNotFoundError"
