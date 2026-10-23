"""Native API doubles plus real synthetic subprocesses; no SDK/model/network."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from bridge_evidence.contest import document_subagents as module


class FakeSpecRegistry:
    def __init__(self):
        self.specs = {}

    def register(self, spec):
        assert spec.name not in self.specs
        self.specs[spec.name] = spec

    def get(self, name):
        return self.specs.get(name)


class FakeBackgroundRegistry:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.lock = threading.Lock()
        self.rows = {}

    def save(self):
        self.file_path.write_text(json.dumps(self.rows), encoding="utf-8")


class FakeSDK:
    """Reproduce native delegate's asynchronous runner and terminal states."""
    SubAgentSpec = SimpleNamespace
    BackgroundProcessRegistry = FakeBackgroundRegistry

    def __init__(self, *, echo=False, query_mode=None, stopped_while_running=False):
        self.spec_registry = FakeSpecRegistry()
        self.runners = {}
        self.delegates = []
        self.threads = []
        self.echo = echo
        self.query_mode = query_mode
        self.stopped_while_running = stopped_while_running
        self.stop_calls = []
        self.query_calls = []
        self.default_registrations = []

    def SubAgentRegistry(self):
        return self.spec_registry

    def register_subagent_runner(self, subagent_type, runner, *, is_default=False):
        assert self.spec_registry.get(subagent_type) is not None
        assert callable(runner.run)
        self.runners[subagent_type] = runner
        self.default_registrations.append(is_default)

    def delegate_subagent(self, **kwargs):
        self.delegates.append(kwargs)
        registry = kwargs["registry"]
        index = len(self.delegates)
        bg_id, session_id = f"bg-{index}", f"session-{index}"
        with registry.lock:
            registry.rows[bg_id] = {"bg_process_id": bg_id, "sub_session_id": session_id,
                "parent_session_id": kwargs["parent_session_id"], "subagent_type": kwargs["subagent_type"],
                "prompt": kwargs["prompt"], "status": "pending", "result": None}
            registry.save()

        def worker():
            with registry.lock:
                if registry.rows[bg_id]["status"] not in module._TERMINAL:
                    registry.rows[bg_id]["status"] = "running"
                registry.save()
            try:
                result = "echo" if self.echo else self.runners[kwargs["subagent_type"]].run(
                    prompt=kwargs["prompt"], similar_task_ids=kwargs["similar_task_ids"])
                status = "completed"
            except Exception as exc:
                result, status = None, "failed"
                with registry.lock:
                    registry.rows[bg_id]["error"] = type(exc).__name__
            with registry.lock:
                if registry.rows[bg_id]["status"] not in module._TERMINAL:
                    registry.rows[bg_id].update(status=status, result=result)
                registry.save()

        thread = threading.Thread(target=worker, daemon=True)
        self.threads.append(thread)
        thread.start()
        return {"bg_process_id": bg_id, "sub_session_id": session_id}

    def query_bg_process_status(self, bg_process_id, *, registry):
        self.query_calls.append(bg_process_id)
        if self.query_mode == "missing":
            return None
        with registry.lock:
            value = deepcopy(registry.rows[bg_process_id])
            if self.stopped_while_running and value["status"] == "running":
                registry.rows[bg_process_id]["status"] = "stopped"
                value["status"] = "stopped"
                registry.save()
        if self.query_mode == "wrong_id":
            value["bg_process_id"] = "wrong-bg"
        if self.query_mode == "wrong_result" and value["status"] == "completed":
            value["result"] = "not-the-runner-result"
        return value

    def stop_bg_process(self, bg_process_id, *, reason, registry):
        self.stop_calls.append((bg_process_id, reason))
        with registry.lock:
            if registry.rows[bg_process_id]["status"] not in module._TERMINAL:
                registry.rows[bg_process_id]["status"] = "stopped"
                registry.save()
        return True


_CHILD = r'''
import json, os, time
from pathlib import Path
task = json.loads(Path("assignment.json").read_text())
started = time.monotonic()
time.sleep(task.get("sleep", 0.01))
result = {"task_id": task.get("id", task.get("task_id")),
          "status": task.get("child_status", "completed"),
          "generation_attempts_settled": task.get("generation_settled", True),
          "pid": os.getpid(), "started": started, "ended": time.monotonic(),
          "saved_rows": [{"value": task.get("literal", "synthetic")}]}
if task.get("wrong_id"):
    result["task_id"] = "other"
if not task.get("missing_result"):
    Path("result.json").write_text("broken" if task.get("bad_json") else json.dumps(result))
raise SystemExit(task.get("exit_code", 0))
'''


def command(task, directory):
    # Literal model-authored strings remain data, never shell command text.
    return [sys.executable, "-c", _CHILD]


def dispatch(tmp_path, tasks, *, sdk=None, timeout=5, **kwargs):
    sdk = sdk or FakeSDK()
    output = module.dispatch_tasks(tasks, run_directory=tmp_path / "dispatch", worker_command=command,
                                   model="own-synthetic-model", timeout_seconds=timeout, sdk=sdk, **kwargs)
    for thread in sdk.threads:
        thread.join(timeout=2)
    return output, sdk


def test_four_native_tasks_use_two_isolated_processes_and_short_artifact_refs(tmp_path):
    output, sdk = dispatch(tmp_path, [{"id": f"task-{n}", "sleep": .18} for n in range(4)], tools=["read_source"])
    assert output["status"] == "finished" and output["settled"] is True
    assert output["semantic_quality_accepted"] is None
    assert len(sdk.delegates) == 4 and sdk.query_calls
    assert sdk.default_registrations == [False]
    specs = list(sdk.spec_registry.specs.values())
    assert len(specs) == 1 and specs[0].model == "own-synthetic-model" and specs[0].tools == ["read_source"]
    rows = output["results"]
    assert all(row["settled"] and row["native_status"] == "completed" and row["runner_done"] for row in rows)
    results = [json.loads(Path(row["result_path"]).read_text()) for row in rows]
    assert len({item["pid"] for item in results}) == 4
    events = sorted([(item["started"], 1) for item in results] + [(item["ended"], -1) for item in results])
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    assert peak == 2 and active == 0
    assert len({row["task_directory"] for row in rows}) == 4
    for row in rows:
        assert row["sha256"] == hashlib.sha256(Path(row["result_path"]).read_bytes()).hexdigest()
    registry = json.loads(Path(output["native_registry_path"]).read_text())
    for item in registry.values():
        result = json.loads(item["result"])
        assert set(result) == {"task_id", "status", "result_ref", "process_receipt_ref"}
        assert len(item["result"]) < 1200 and "saved_rows" not in item["result"]


def test_failed_exit_preserves_result_and_is_settled_without_retry(tmp_path):
    output, sdk = dispatch(tmp_path, [{"id": "bad", "exit_code": 2, "child_status": "failed"}, {"id": "good"}])
    assert output["status"] == "failed" and output["settled"] is True
    bad = output["results"][0]
    assert bad["returncode"] == 2 and bad["native_status"] == "failed" and bad["child_status"] == "failed"
    assert Path(bad["result_path"]).is_file() and bad["sha256"] and bad["retry_allowed"] is False
    assert json.loads(Path(bad["result_path"]).read_text())["saved_rows"]
    assert len(sdk.delegates) == 2


def test_unknown_upstream_cannot_be_completed_or_retried(tmp_path):
    output, sdk = dispatch(tmp_path, [{"id": "task", "generation_settled": False}], timeout=.3)
    row = output["results"][0]
    assert row["process_settled"] is True and row["native_terminal"] is True
    assert row["generation_attempts_settled"] is False and row["settled"] is False
    assert output["status"] == "unknown" and output["retry_allowed"] is False and len(sdk.delegates) == 1


def test_native_stop_alone_does_not_end_running_process(tmp_path):
    output, sdk = dispatch(tmp_path, [{"id": "task", "sleep": .2}], sdk=FakeSDK(stopped_while_running=True))
    row = output["results"][0]
    assert output["status"] == "failed" and row["settled"] is True
    assert row["native_status"] == "stopped" and row["returncode"] == 0
    assert json.loads(Path(row["result_path"]).read_text())["ended"] >= json.loads(Path(row["result_path"]).read_text())["started"] + .19
    assert len(sdk.query_calls) >= 2


def test_deadline_closes_owned_process_and_does_not_invent_backend_stop(tmp_path):
    start = time.monotonic()
    output, sdk = dispatch(tmp_path, [{"id": "task", "sleep": 20}], timeout=.15)
    assert time.monotonic() - start < 2
    row = output["results"][0]
    assert row["process_settled"] is True and row["returncode"] is not None
    assert row["generation_attempts_settled"] is None
    assert output["status"] == "unknown" and row["settled"] is False and row["retry_allowed"] is False
    assert sdk.stop_calls and len(sdk.delegates) == 1


@pytest.mark.parametrize("fields", [{"missing_result": True}, {"bad_json": True}, {"wrong_id": True}])
def test_missing_invalid_or_misassociated_result_never_finishes(tmp_path, fields):
    output, _ = dispatch(tmp_path, [{"id": "task", **fields}], timeout=.3)
    assert output["status"] == "unknown" and output["settled"] is False
    if not fields.get("missing_result"):
        assert output["results"][0]["result_path"] and output["results"][0]["sha256"]


@pytest.mark.parametrize("sdk", [FakeSDK(echo=True), FakeSDK(query_mode="wrong_id"), FakeSDK(query_mode="wrong_result"), FakeSDK(query_mode="missing")])
def test_native_echo_bad_identity_result_or_missing_query_cannot_finish(tmp_path, sdk):
    output, _ = dispatch(tmp_path, [{"id": "task"}], sdk=sdk, timeout=.3)
    assert output["status"] != "finished" and output["settled"] is False
    assert output["retry_allowed"] is False


def test_task_data_stays_literal_and_aliases_agree(tmp_path):
    literal = "$(touch SHOULD_NOT_EXIST); `printf unsafe` 中文"
    output, _ = dispatch(tmp_path, [{"id": "one", "task_id": "one", "literal": literal}])
    assert output["status"] == "finished"
    assert json.loads(Path(output["results"][0]["result_path"]).read_text())["saved_rows"][0]["value"] == literal
    assert not (tmp_path / "dispatch" / "tasks" / "one" / "SHOULD_NOT_EXIST").exists()


@pytest.mark.parametrize("tasks", [[{"id": "../bad"}], [{"id": "same"}, {"id": "same"}],
                                   [{"id": "one", "task_id": "two"}], [], ["bad"]])
def test_invalid_tasks_rejected_before_native_dispatch(tmp_path, tasks):
    sdk = FakeSDK()
    with pytest.raises(ValueError):
        dispatch(tmp_path, tasks, sdk=sdk)
    assert not sdk.delegates


def test_bounded_deadline_required_and_min_of_parent_and_timeout_used(tmp_path):
    sdk = FakeSDK()
    with pytest.raises(ValueError):
        module.dispatch_tasks([{"id": "one"}], run_directory=tmp_path, worker_command=command, model="own", sdk=sdk)
    deadline = time.monotonic() + 2
    output, _ = dispatch(tmp_path, [{"id": "one"}], timeout=5, deadline=deadline)
    assert output["deadline_monotonic"] == deadline


def test_existing_dispatch_cannot_be_restarted_with_fresh_sdk(tmp_path):
    output, _ = dispatch(tmp_path, [{"id": "one"}])
    assert output["status"] == "finished"
    sdk = FakeSDK()
    with pytest.raises(FileExistsError):
        dispatch(tmp_path, [{"id": "one"}], sdk=sdk)
    assert not sdk.delegates


def test_final_native_query_is_checked_before_returning_finished(tmp_path):
    class FinalQueryChanges(FakeSDK):
        completed_queries = 0

        def query_bg_process_status(self, *args, **kwargs):
            row = super().query_bg_process_status(*args, **kwargs)
            if row["status"] == "completed":
                self.completed_queries += 1
                if self.completed_queries > 1:
                    row["sub_session_id"] = "changed-after-completion"
            return row

    output, _ = dispatch(tmp_path, [{"id": "one"}], sdk=FinalQueryChanges())
    assert output["status"] == "unknown" and output["settled"] is False
    assert output["results"][0]["association_error"] == "NATIVE_FINAL_QUERY_UNKNOWN"


def test_unknown_worker_returns_promptly_without_spending_unused_parent_budget(tmp_path):
    start = time.monotonic()
    output, _ = dispatch(tmp_path, [{"id": "one", "generation_settled": False}], timeout=30)
    assert time.monotonic() - start < 2
    assert output["status"] == "unknown" and "DISPATCH_DEADLINE_EXCEEDED" not in output["errors"]


def test_trusted_worker_receives_exact_parent_budget_and_deadline(tmp_path):
    seen = []
    deadline = time.monotonic() + 2
    task = {"id": "one", "max_model_calls": 32, "global_deadline_monotonic": deadline, "source_ids": ["E000001"]}

    def trusted_command(value, directory):
        seen.append(deepcopy(value))
        (directory / "request.json").write_text(json.dumps(value))
        return command(value, directory)

    output = module.dispatch_tasks([task], run_directory=tmp_path, worker_command=trusted_command,
                                   model="own", deadline=deadline, sdk=FakeSDK())
    assert output["status"] == "finished" and seen == [task]
    assert json.loads((tmp_path / "tasks" / "one" / "assignment.json").read_text()) == task


def test_sdk_is_lazy_until_dispatch(monkeypatch):
    def forbidden(_):
        raise AssertionError("No real SDK imports allowed in synthetic tests")
    monkeypatch.setattr(module.importlib, "import_module", forbidden)
    # Pure helpers and constructing a fake SDK have no native import side effect.
    assert module._status({"status": "COMPLETED"}) == "completed"
    assert FakeSDK().runners == {}
