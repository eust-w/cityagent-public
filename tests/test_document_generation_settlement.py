"""Only saved synthetic dispatches and metadata GET doubles; no model/network."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError

import pytest

from bridge_evidence.contest import document_generation_settlement as module


MODEL = "own-model"
MANIFEST = "a" * 64


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def stopped_dispatch(tmp_path):
    # Check the production PID+PGID probe against a truly reaped process.
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    assert process.wait(timeout=5) == 0
    directory = tmp_path / "children"
    task = directory / "tasks" / "one"
    save(task / "assignment.json", {"id": "one", "source_ids": ["E000001"]})
    save(task / "result.json", {"task_id": "one", "manifest_sha256": MANIFEST, "status": "failed",
         "generation_attempts_settled": False, "unknown_generation_calls": ["original-unknown-call"],
         "usage": None, "done_received": False, "saved_rows": [{"text": "synthetic saved fact"}]})
    data = {"phase": "failed", "pid": process.pid, "returncode": 2, "process_settled": True,
        "generation_attempts_settled": False, "result_path": str(task / "result.json"),
        "sha256": ref(task / "result.json")["sha256"], "assignment_ref": ref(task / "assignment.json")}
    save(task / "process-receipt.json", data)
    row = {**data, "task_id": "one", "task_directory": str(task), "runner_done": True,
        "native_terminal": True, "native_result_matches": True, "association_error": None,
        "native_status": "failed", "settled": False, "bg_process_id": "bg-one"}
    native = directory / "native-background-processes.json"
    save(native, {"bg-one": {"status": "failed"}})
    claim = {"schema": "cityagent-native-subagent-dispatch-v1", "run_id": "trusted-run",
             "model": MODEL, "task_ids": ["one"], "deadline_monotonic": 120.0}
    save(directory / "dispatch-claim.json", claim)
    dispatch = {**claim, "status": "unknown", "results": [row], "native_registry_path": str(native)}
    save(directory / "dispatch-summary.json", dispatch)
    return directory, dispatch


class Clock:
    def __init__(self):
        self.value = 100.0
        self.sleeps = []

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        assert 0 <= seconds <= 1
        self.sleeps.append(seconds)
        self.value += seconds


def models(*, model=MODEL, parent=None, context=262144, root="same-root"):
    return json.dumps({"data": [{"id": model, "parent": parent, "max_model_len": context, "root": root}]}).encode()


def metrics(running=(0, 0), waiting=(0, 0)):
    lines = ["# TYPE vllm:num_requests_running gauge"]
    for name, values in (("running", running), ("waiting", waiting)):
        for engine, value in enumerate(values):
            lines.append(f'vllm:num_requests_{name}{{model_name="{MODEL}",engine="{engine}"}} {value}')
    return ("\n".join(lines) + "\n").encode()


class Fetch:
    def __init__(self, *, metric_bodies=None, model_bodies=None, failure=None, clock=None):
        self.metrics = metric_bodies or [metrics()]
        self.models = model_bodies or [models()]
        self.failure = failure
        self.calls = []
        self.indices = {"models": 0, "metrics": 0}
        self.clock = clock

    def __call__(self, url, *, timeout, headers):
        assert url in {"http://localhost:18110/v1/models", "http://localhost:18110/metrics"}
        assert 0 < timeout <= 5
        self.calls.append((url, timeout, deepcopy(headers)))
        if self.failure is not None:
            raise self.failure
        kind = "models" if url.endswith("models") else "metrics"
        data = self.models if kind == "models" else self.metrics
        index = self.indices[kind]
        self.indices[kind] += 1
        if self.clock is not None:
            self.clock.value += .01
        return 200, data[min(index, len(data) - 1)]


def observe(stopped_dispatch, *, fetch=None, clock=None, **kwargs):
    directory, dispatch = stopped_dispatch
    clock = clock or Clock()
    fetch = fetch or Fetch(clock=clock)
    result = module.observe_generation_quiescence(dispatch, run_directory=directory,
        base_url="http://localhost:18110/v1", model=MODEL, manifest_sha256=MANIFEST,
        deadline=120.0, fetch=fetch, clock=clock, sleep=clock.sleep, **kwargs)
    return result, fetch, clock


def rewrite_summary(stopped_dispatch):
    directory, dispatch = stopped_dispatch
    save(directory / "dispatch-summary.json", dispatch)


def test_two_idle_observations_allow_parent_but_never_reclassify_old_unknown(stopped_dispatch):
    directory, _ = stopped_dispatch
    originals = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    result, fetch, clock = observe(stopped_dispatch, api_key="secret-never-persist")
    assert result["outcome"] == "quiescent_observed" and result["may_continue_aggregation"] is True
    assert result["cancellation_confirmed"] is False and result["prior_requests_reclassified"] is False
    assert result["model_retry_permitted"] is False and result["unknown_call_ids"] == ["original-unknown-call"]
    assert len(fetch.calls) == 4 and len(result["observations"]) == 2 and clock.sleeps
    assert {key: result["model_identity"][key] for key in ("model", "parent", "max_model_len", "root")} == {
        "model": MODEL, "parent": None, "max_model_len": 262144, "root": "same-root"}
    assert len(result["model_identity"]["selected_identity_sha256"]) == 64
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    original_result = json.loads((directory / "tasks/one/result.json").read_bytes())
    assert original_result["usage"] is None and original_result["done_received"] is False and original_result["status"] == "failed"
    assert ref(Path(result["receipt_ref"]["path"])) == result["receipt_ref"]
    for path in (directory / "generation-quiescence").iterdir():
        assert b"secret-never-persist" not in path.read_bytes()
    assert all(call[2]["Authorization"] == "Bearer secret-never-persist" for call in fetch.calls)


def test_busy_resets_continuous_idle_counter(stopped_dispatch):
    fetch = Fetch(metric_bodies=[metrics(), metrics(running=(0, 1)), metrics(), metrics()])
    result, _, _ = observe(stopped_dispatch, fetch=fetch)
    assert result["outcome"] == "quiescent_observed"
    assert [row["consecutive_idle_samples"] for row in result["observations"]] == [1, 0, 1, 2]


def test_busy_cannot_extend_shared_deadline_or_observation_cap(stopped_dispatch):
    result, fetch, clock = observe(stopped_dispatch, fetch=Fetch(metric_bodies=[metrics(waiting=(1, 0))]),
                                  max_wait_seconds=.7, poll_interval=.3)
    assert result["outcome"] == "unknown" and result["may_continue_aggregation"] is False
    assert result["error_code"] == "OBSERVATION_DEADLINE_EXCEEDED"
    assert clock.value <= 100.7 + 1e-9 and all(value <= 1 for value in clock.sleeps)


@pytest.mark.parametrize("body", [b"unrelated_metric 0\n", metrics(waiting=()),
    metrics(running=("NaN", 0)), metrics(running=("+Inf", 0)), metrics(running=(-1, 0)),
    metrics(running=(0, 0), waiting=(0,)), metrics() + metrics(),
    b'vllm:num_requests_running{engine="0"} broken\nvllm:num_requests_waiting{engine="0"} 0\n'])
def test_missing_invalid_duplicate_or_incomplete_metrics_refused(stopped_dispatch, body):
    result, _, _ = observe(stopped_dispatch, fetch=Fetch(metric_bodies=[body]))
    assert result["outcome"] == "unknown" and result["may_continue_aggregation"] is False
    assert result["error_code"].startswith("METRICS_")


@pytest.mark.parametrize("body", [models(model="other"), models(parent="missing-base"), models(context=65536),
    b'{"data": []}', b'{"data":[{"id":"own-model","max_model_len":262144}]}'])
def test_model_parent_or_context_mismatch_refused(stopped_dispatch, body):
    result, _, _ = observe(stopped_dispatch, fetch=Fetch(model_bodies=[body]))
    assert result["outcome"] == "unknown" and result["may_continue_aggregation"] is False


def test_identity_must_remain_stable_across_both_samples(stopped_dispatch):
    result, _, _ = observe(stopped_dispatch, fetch=Fetch(model_bodies=[models(), models(root="replaced-root")]))
    assert result["outcome"] == "unknown" and result["error_code"] == "MODEL_IDENTITY_CHANGED_DURING_OBSERVATION"


@pytest.mark.parametrize("adapter_context", [None, 65536])
def test_real_lora_shape_uses_stable_selected_and_base_card_context(stopped_dispatch, adapter_context):
    selected = {"id": MODEL, "parent": "qwen36-27b-base-5090", "root": "epoch-002-adapter"}
    if adapter_context is not None:
        selected["max_model_len"] = adapter_context
    base = {"id": "qwen36-27b-base-5090", "parent": None, "root": "base-path", "max_model_len": 262144}
    body = json.dumps({"data": [base, selected]}).encode()
    result, _, _ = observe(stopped_dispatch, fetch=Fetch(model_bodies=[body]))
    assert result["outcome"] == "quiescent_observed"
    assert result["model_identity"]["parent"] == base["id"] and result["model_identity"]["max_model_len"] == 262144
    assert result["model_identity"]["base"]["id"] == base["id"]
    assert result["model_identity"]["selected_identity_sha256"] != result["model_identity"]["base_identity_sha256"]


def test_explicit_expected_parent_and_lora_base_identity_are_enforced(stopped_dispatch):
    base = {"id": "base", "parent": None, "root": "base-path", "max_model_len": 262144}
    selected = {"id": MODEL, "parent": "base", "root": "adapter"}
    body = json.dumps({"data": [base, selected]}).encode()
    result, _, _ = observe(stopped_dispatch, fetch=Fetch(model_bodies=[body]), expected_parent="different-base")
    assert result["outcome"] == "unknown" and result["error_code"] == "MODEL_PARENT_OR_CONTEXT_CHANGED"


def test_changed_base_card_with_same_model_name_is_not_stable_identity(stopped_dispatch):
    base = {"id": "base", "parent": None, "root": "base-path", "max_model_len": 262144}
    selected = {"id": MODEL, "parent": "base", "root": "adapter"}
    bodies = [json.dumps({"data": [base, selected]}).encode(),
              json.dumps({"data": [{**base, "root": "changed-base-path"}, selected]}).encode()]
    result, _, _ = observe(stopped_dispatch, fetch=Fetch(model_bodies=bodies))
    assert result["outcome"] == "unknown" and result["error_code"] == "MODEL_IDENTITY_CHANGED_DURING_OBSERVATION"


def test_per_get_display_metadata_is_preserved_but_is_not_model_identity(stopped_dispatch):
    first = json.loads(models())
    second = deepcopy(first)
    first["data"][0].update(created=1, permission=[{"id": "display-one"}])
    second["data"][0].update(created=2, permission=[{"id": "display-two"}])
    result, _, _ = observe(stopped_dispatch, fetch=Fetch(model_bodies=[json.dumps(first).encode(), json.dumps(second).encode()]))
    assert result["outcome"] == "quiescent_observed"
    refs = [sample["requests"][0]["body_ref"] for sample in result["observations"]]
    assert refs[0]["sha256"] != refs[1]["sha256"]


@pytest.mark.parametrize("failure", [TimeoutError("synthetic timeout"), OSError("synthetic I/O"),
    HTTPError("http://localhost:18110/v1/models", 503, "synthetic", None, None)])
def test_http_errors_fail_closed_without_model_retry(stopped_dispatch, failure):
    result, fetch, _ = observe(stopped_dispatch, fetch=Fetch(failure=failure))
    assert result["outcome"] == "unknown" and result["may_continue_aggregation"] is False
    assert len(fetch.calls) == 1 and result["model_retry_permitted"] is False


@pytest.mark.parametrize("field,value", [("process_settled", False), ("runner_done", False),
    ("native_terminal", False), ("native_result_matches", False), ("association_error", "bad_join")])
def test_unstopped_unpaired_or_untrusted_dispatch_does_not_get_backend(stopped_dispatch, field, value):
    stopped_dispatch[1]["results"][0][field] = value
    rewrite_summary(stopped_dispatch)
    result, fetch, _ = observe(stopped_dispatch)
    assert result["outcome"] == "unknown" and fetch.calls == []


def test_actual_live_child_pid_blocks_gets_even_if_dispatch_claims_stopped(stopped_dispatch):
    directory, dispatch = stopped_dispatch
    dispatch["results"][0]["pid"] = os.getpid()
    path = directory / "tasks/one/process-receipt.json"
    data = json.loads(path.read_bytes())
    data["pid"] = os.getpid()
    save(path, data)
    rewrite_summary(stopped_dispatch)
    result, fetch, _ = observe(stopped_dispatch)
    assert result["error_code"] == "OWNED_CHILD_OR_PROCESS_GROUP_STILL_EXISTS" and fetch.calls == []


@pytest.mark.parametrize("change", ["result_hash", "assignment_hash", "missing_result", "summary_identity"])
def test_result_and_original_dispatch_must_be_trusted_before_gets(stopped_dispatch, change):
    directory, dispatch = stopped_dispatch
    if change == "result_hash":
        (directory / "tasks/one/result.json").write_text('{"changed":true}')
    elif change == "assignment_hash":
        (directory / "tasks/one/assignment.json").write_text('{"id":"one","changed":true}')
    elif change == "missing_result":
        (directory / "tasks/one/result.json").unlink()
    else:
        dispatch["run_id"] = "untrusted-replacement"
    result, fetch, _ = observe(stopped_dispatch)
    assert result["outcome"] == "unknown" and fetch.calls == []


def test_artifact_mutation_during_observation_blocks_late_admission(stopped_dispatch):
    directory, _ = stopped_dispatch
    fetch = Fetch()

    def changing(url, **kwargs):
        result = fetch(url, **kwargs)
        if len(fetch.calls) == 4:
            (directory / "tasks/one/result.json").write_text('{"changed":true}')
        return result

    result, _, _ = observe(stopped_dispatch, fetch=changing)
    assert result["outcome"] == "unknown" and result["error_code"] == "TRUSTED_INPUT_CHANGED_DURING_OBSERVATION"


def test_deadline_crossed_during_get_never_allows_parent(stopped_dispatch):
    clock, fetch = Clock(), Fetch()

    def slow(url, **kwargs):
        result = fetch(url, **kwargs)
        clock.value = 121
        return result

    result, _, _ = observe(stopped_dispatch, fetch=slow, clock=clock)
    assert result["outcome"] == "unknown" and result["may_continue_aggregation"] is False


def test_observation_does_not_overwrite_earlier_receipt(stopped_dispatch):
    result, _, _ = observe(stopped_dispatch)
    original = Path(result["receipt_ref"]["path"]).read_bytes()
    with pytest.raises(FileExistsError):
        observe(stopped_dispatch)
    assert Path(result["receipt_ref"]["path"]).read_bytes() == original
