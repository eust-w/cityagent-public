"""Native TongAgents delegation with isolated, bounded document workers.

The native runner is an actual subprocess runner, never the SDK's echo runner
and never a nested run_react() in a SDK thread. A native terminal state alone
does not prove that a child or its upstream generation stopped. Both ledgers
are retained, including failed result artifacts; this module has no retry or
semantic acceptance policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Sequence
import uuid


_TERMINAL = frozenset({"completed", "failed", "stopped", "timeout"})
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_POLL_SECONDS = 0.05
_CLEANUP_SECONDS = 5.0


def _encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _save(path: Path, value: Any) -> None:
    name = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".subagent-", delete=False) as out:
            name = out.name
            out.write(_encoded(value) + b"\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
        name = None
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def _ref(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _load_sdk() -> SimpleNamespace:
    # Compiled SDK packages remain optional until actual dispatch. These names
    # are the public signatures verified in the installed TongAgents 2.7.20.
    specs = importlib.import_module("tongagents.hooks.subagent_registry")
    delegate = importlib.import_module("tongagents.agent_runtime.delegate")
    background = importlib.import_module("tongagents.agent_runtime.background_process")
    return SimpleNamespace(SubAgentSpec=specs.SubAgentSpec, SubAgentRegistry=specs.SubAgentRegistry,
                           BackgroundProcessRegistry=background.BackgroundProcessRegistry,
                           register_subagent_runner=delegate.register_subagent_runner,
                           delegate_subagent=delegate.delegate_subagent,
                           query_bg_process_status=delegate.query_bg_process_status,
                           stop_bg_process=delegate.stop_bg_process)


def _status(record: dict | None) -> str | None:
    value = record.get("status") if isinstance(record, dict) else None
    value = getattr(value, "value", value)
    return value.lower() if isinstance(value, str) else None


def _group_alive(process: subprocess.Popen) -> bool:
    if os.name != "posix":
        return process.poll() is None
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_owned(process: subprocess.Popen) -> None:
    """Signal only our still-live Popen leader, then reap within a fixed bound.

    A surviving group after the leader already exited is reported unknown;
    it is not killed by guessing that an old numeric PGID still belongs to us.
    Neither killing this process nor SDK stop proves a backend model stopped.
    """
    if process.poll() is not None:
        return
    for sig, allowance in ((signal.SIGTERM, 1.0), (signal.SIGKILL, _CLEANUP_SECONDS - 1.0)):
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            else:
                process.terminate() if sig == signal.SIGTERM else process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=allowance)
            return
        except subprocess.TimeoutExpired:
            continue


class _RunnerFailure(RuntimeError):
    """Safe code only: worker logs and report text never become SDK errors."""


@dataclass
class _Task:
    task_id: str
    directory: Path
    assignment_ref: dict
    prompt: str
    argv: list[str]
    lock: threading.RLock = field(default_factory=threading.RLock)
    done: threading.Event = field(default_factory=threading.Event)
    invoked: bool = False
    process: subprocess.Popen | None = None
    data: dict = field(default_factory=dict)
    runner_result: str | None = None
    native: dict | None = None
    bg_process_id: str | None = None
    sub_session_id: str | None = None
    association_error: str | None = None

    def save(self) -> None:
        _save(self.directory / "process-receipt.json", self.data)

    def result_row(self) -> dict:
        with self.lock:
            data = dict(self.data)
            native_status = _status(self.native)
            terminal = native_status in _TERMINAL
            paired = (native_status != "completed" or
                      (self.done.is_set() and self.native.get("result") == self.runner_result))
            settled = (terminal and paired and self.done.is_set() and
                       data.get("process_settled") is True and
                       (data.get("generation_attempts_settled") is True or data.get("never_launched") is True)
                       and self.association_error is None)
            return {**data, "task_id": self.task_id, "task_directory": str(self.directory),
                    "result_path": data.get("result_path"), "sha256": data.get("sha256"),
                    "bg_process_id": self.bg_process_id, "sub_session_id": self.sub_session_id,
                    "native_status": native_status, "native_terminal": terminal,
                    "runner_done": self.done.is_set(), "native_result_matches": paired,
                    "association_error": self.association_error,
                    "settled": settled, "retry_allowed": False}


class _ProcessRunner:
    def __init__(self, tasks: list[_Task], *, max_workers: int, deadline: float):
        self.tasks = {task.prompt: task for task in tasks}
        self.slots = threading.BoundedSemaphore(max_workers)
        self.deadline = deadline
        self.cancel = threading.Event()

    def run(self, *, prompt: str, similar_task_ids: list[int]) -> str:
        task = self.tasks.get(prompt)
        if task is None or similar_task_ids:
            raise _RunnerFailure("UNTRUSTED_SUBAGENT_ASSIGNMENT")
        with task.lock:
            if task.invoked:
                raise _RunnerFailure("SUBAGENT_RUNNER_ALREADY_INVOKED")
            task.invoked = True
            task.data["phase"] = "waiting_for_process_slot"
            task.save()
        acquired = False
        try:
            while not acquired:
                if self.cancel.is_set() or time.monotonic() >= self.deadline:
                    raise _RunnerFailure("DEADLINE_BEFORE_CHILD_LAUNCH")
                acquired = self.slots.acquire(timeout=min(_POLL_SECONDS, max(0, self.deadline - time.monotonic())))
            # No launch after a cancellation while waiting in the native pool.
            if self.cancel.is_set() or time.monotonic() >= self.deadline:
                raise _RunnerFailure("DEADLINE_BEFORE_CHILD_LAUNCH")
            assignment = Path(task.assignment_ref["path"])
            if _ref(assignment) != task.assignment_ref:
                raise _RunnerFailure("ASSIGNMENT_CHANGED_BEFORE_LAUNCH")
            with (task.directory / "stdout.log").open("wb") as stdout, (task.directory / "stderr.log").open("wb") as stderr:
                process = subprocess.Popen(task.argv, cwd=task.directory, stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr, shell=False, start_new_session=True)
                with task.lock:
                    task.process = process
                    task.data.update(phase="running", pid=process.pid, never_launched=False,
                                     generation_attempts_settled=None,
                                     launched_at_monotonic=time.monotonic())
                    task.save()
                while process.poll() is None:
                    if self.cancel.is_set() or time.monotonic() >= self.deadline:
                        with task.lock:
                            task.data["deadline_exceeded"] = True
                        _stop_owned(process)
                        break
                    try:
                        process.wait(timeout=min(_POLL_SECONDS, max(0, self.deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
            with task.lock:
                task.data.update(returncode=process.poll(), process_settled=(process.poll() is not None and not _group_alive(process)),
                                 finished_at_monotonic=time.monotonic())
                self._read_result(task)
                if task.data.get("process_settled") is not True:
                    raise _RunnerFailure("CHILD_PROCESS_EXIT_UNKNOWN")
                if task.data.get("generation_attempts_settled") is not True:
                    raise _RunnerFailure("CHILD_UPSTREAM_SETTLEMENT_UNKNOWN")
                if task.data.get("deadline_exceeded"):
                    raise _RunnerFailure("CHILD_DEADLINE_EXCEEDED")
                if process.returncode != 0:
                    raise _RunnerFailure("CHILD_EXIT_NONZERO")
                if task.data.get("child_status") != "completed":
                    raise _RunnerFailure("CHILD_REPORTED_FAILURE")
                task.data["phase"] = "completed"
                task.save()
                task.runner_result = _encoded({"task_id": task.task_id, "status": "completed",
                    "result_ref": {"path": task.data["result_path"], "sha256": task.data["sha256"]},
                    "process_receipt_ref": _ref(task.directory / "process-receipt.json")}).decode("utf-8")
                return task.runner_result
        except Exception as exc:
            with task.lock:
                task.data.update(phase="failed", error_code=str(exc) if isinstance(exc, _RunnerFailure) else "CHILD_RUNNER_ERROR",
                                 error_type=type(exc).__name__)
                if task.process is None:
                    task.data.update(never_launched=True, process_settled=True, generation_attempts_settled=True)
                else:
                    _stop_owned(task.process)
                    task.data.update(returncode=task.process.poll(),
                                     process_settled=(task.process.poll() is not None and not _group_alive(task.process)))
                task.save()
            raise _RunnerFailure(task.data["error_code"]) from None
        finally:
            if acquired:
                self.slots.release()
            task.done.set()

    @staticmethod
    def _read_result(task: _Task) -> None:
        path = task.directory / "result.json"
        if not path.is_file() or path.is_symlink():
            raise _RunnerFailure("CHILD_RESULT_MISSING")
        raw = path.read_bytes()
        task.data.update(result_path=str(path), sha256=hashlib.sha256(raw).hexdigest())
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError):
            raise _RunnerFailure("CHILD_RESULT_INVALID_JSON") from None
        if not isinstance(value, dict) or value.get("status") not in {"completed", "failed"}:
            raise _RunnerFailure("CHILD_RESULT_INVALID_STATUS")
        if any(value[key] != task.task_id for key in ("id", "task_id") if key in value):
            raise _RunnerFailure("CHILD_RESULT_ID_MISMATCH")
        task.data.update(child_status=value["status"], generation_attempts_settled=value.get("generation_attempts_settled") is True)


def dispatch_tasks(tasks: Sequence[dict], *, run_directory: Path | str,
                   worker_command: Callable[[dict, Path], list[str]], model: str,
                   tools: Sequence[str] = (), system_prompt: str = "", max_workers: int = 2,
                   timeout_seconds: float | None = None, deadline: float | None = None,
                   sdk: Any = None) -> dict:
    """Delegate once, then join native state to owned processes and artifacts.

    ``deadline`` is the parent's absolute time.monotonic() deadline. Task model
    call allocations and that same deadline belong in the trusted worker's
    request.json; the dispatcher neither calls a model nor renews a budget.
    ``finished`` describes settled execution, not accepted extraction quality.
    A fresh claim is mandatory, even for an earlier failed or unknown attempt.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A concrete own-model identifier is required")
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    if not isinstance(system_prompt, str) or any(not isinstance(tool, str) or not tool for tool in tools):
        raise ValueError("Invalid native spec fields")
    now = time.monotonic()
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite monotonic timestamp")
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        deadline = min(deadline, now + timeout_seconds) if deadline is not None else now + timeout_seconds
    if deadline is None or deadline <= now:
        raise ValueError("A future bounded deadline is required")
    normalized = []
    seen = set()
    for value in tasks:
        if not isinstance(value, dict):
            raise ValueError("Each task must be an object")
        task = json.loads(_encoded(value))
        task_id = task.get("id", task.get("task_id"))
        if (not isinstance(task_id, str) or not _ID.fullmatch(task_id) or task_id in seen or
                ("id" in task and "task_id" in task and task["id"] != task["task_id"])):
            raise ValueError("Task IDs must be unique safe strings; aliases must agree")
        seen.add(task_id)
        normalized.append((task_id, task))
    if not normalized:
        raise ValueError("At least one task is required")
    api = _load_sdk() if sdk is None else sdk
    directory = Path(run_directory).absolute()
    directory.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    claim = {"schema": "cityagent-native-subagent-dispatch-v1", "run_id": run_id,
             "model": model, "deadline_monotonic": deadline, "max_workers": max_workers,
             "task_ids": [item[0] for item in normalized], "retry_allowed": False}
    # Existing failed/unknown dispatches are deliberately not reopened.
    with (directory / "dispatch-claim.json").open("xb") as stream:
        stream.write(_encoded(claim) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    prepared = []
    for task_id, task in normalized:
        task_directory = directory / "tasks" / task_id
        task_directory.mkdir(parents=True, exist_ok=False)
        assignment = task_directory / "assignment.json"
        _save(assignment, task)
        argv = worker_command(json.loads(_encoded(task)), task_directory)
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in argv):
            raise ValueError("worker_command must return a nonempty argv list of strings")
        assignment_ref = _ref(assignment)
        prompt = _encoded({"task_id": task_id, "assignment_ref": assignment_ref}).decode("utf-8")
        prepared.append(_Task(task_id, task_directory, assignment_ref, prompt, argv, data={
            "phase": "prepared", "never_launched": True, "process_settled": True,
            "generation_attempts_settled": True, "argv_sha256": hashlib.sha256(_encoded(argv)).hexdigest(),
            "assignment_ref": assignment_ref}))
        prepared[-1].save()
    runner = _ProcessRunner(prepared, max_workers=max_workers, deadline=deadline)
    spec_registry = api.SubAgentRegistry()
    native_type = "cityagent_document_" + run_id
    spec = api.SubAgentSpec(name=native_type, type="sub", description="Isolated CityAgent document task",
        source=__name__, model=model, tools=list(tools), system_prompt=system_prompt,
        metadata={"run_id": run_id, "runner": "owned_subprocess", "max_workers": max_workers})
    spec_registry.register(spec)
    api.register_subagent_runner(native_type, runner, is_default=False)
    native_file = directory / "native-background-processes.json"
    registry = api.BackgroundProcessRegistry(file_path=native_file)
    parent_session_id = "cityagent-document-" + run_id
    errors = []

    def observe_native(task: _Task) -> None:
        native = api.query_bg_process_status(task.bg_process_id, registry=registry)
        if not isinstance(native, dict) or _status(native) not in _TERMINAL | {"pending", "running"}:
            task.association_error = "NATIVE_STATUS_UNKNOWN"
            raise _RunnerFailure(task.association_error)
        pins = {"bg_process_id": task.bg_process_id, "sub_session_id": task.sub_session_id,
                "parent_session_id": parent_session_id, "subagent_type": native_type, "prompt": task.prompt}
        if any(native[key] != value for key, value in pins.items() if key in native):
            task.association_error = "NATIVE_STATUS_IDENTITY_MISMATCH"
            raise _RunnerFailure(task.association_error)
        task.native = native

    def snapshot() -> dict:
        rows = [task.result_row() for task in prepared]
        settled = all(row["settled"] for row in rows)
        finished = settled and not errors and all(row["phase"] == "completed" and row["native_status"] == "completed" for row in rows)
        value = {**claim, "status": "finished" if finished else ("failed" if settled else "unknown"),
                 "settled": settled, "parent_session_id": parent_session_id,
                 "subagent_type": native_type, "native_registry_path": str(native_file),
                 "results": rows, "errors": list(errors), "semantic_quality_accepted": None}
        _save(directory / "dispatch-summary.json", value)
        return value

    try:
        for task in prepared:
            ids = api.delegate_subagent(prompt=task.prompt, subagent_type=native_type,
                parent_session_id=parent_session_id, description="CityAgent task " + task.task_id,
                timeout_seconds=max(1, math.ceil(deadline - time.monotonic())), similar_task_ids=[], registry=registry)
            if (not isinstance(ids, dict) or not isinstance(ids.get("bg_process_id"), str) or not ids["bg_process_id"] or
                    not isinstance(ids.get("sub_session_id"), str) or not ids["sub_session_id"]):
                task.association_error = "INVALID_NATIVE_DELEGATE_IDS"
                raise _RunnerFailure(task.association_error)
            task.bg_process_id, task.sub_session_id = ids["bg_process_id"], ids["sub_session_id"]
            if (sum(other.bg_process_id == task.bg_process_id for other in prepared) != 1 or
                    sum(other.sub_session_id == task.sub_session_id for other in prepared) != 1):
                task.association_error = "DUPLICATE_NATIVE_BACKGROUND_ID"
                raise _RunnerFailure(task.association_error)
        joined = False
        while time.monotonic() < deadline:
            for task in prepared:
                observe_native(task)
            current = snapshot()
            if current["settled"] or all(_status(task.native) in _TERMINAL and task.done.is_set() for task in prepared):
                # Unknown upstream settlement cannot be repaired by waiting or
                # repeating a child. Return its evidence promptly, still unknown.
                joined = True
                break
            if all(_status(task.native) in _TERMINAL for task in prepared) and any(not task.invoked for task in prepared):
                # In particular, a native echo completion cannot be accepted.
                raise _RunnerFailure("NATIVE_TERMINAL_WITHOUT_OWN_RUNNER")
            time.sleep(min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
        if not joined:
            errors.append("DISPATCH_DEADLINE_EXCEEDED")
    except Exception as exc:
        errors.append(str(exc) if isinstance(exc, _RunnerFailure) else "NATIVE_DISPATCH_ERROR")
    finally:
        runner.cancel.set()
        # This is a native state request only. Actual workers independently
        # observe cancel and reap; no authoritative backend usage is invented.
        for task in prepared:
            if task.bg_process_id and _status(task.native) not in _TERMINAL:
                try:
                    api.stop_bg_process(task.bg_process_id, reason="owned-dispatch-ending", registry=registry)
                except Exception:
                    errors.append("NATIVE_STOP_STATE_UNKNOWN")
        cleanup_deadline = time.monotonic() + _CLEANUP_SECONDS + 1.0
        for task in prepared:
            if task.invoked:
                task.done.wait(timeout=max(0, cleanup_deadline - time.monotonic()))
            if task.bg_process_id:
                try:
                    observe_native(task)
                except Exception:
                    task.association_error = "NATIVE_FINAL_QUERY_UNKNOWN"
    return snapshot()
