"""RSI evidence exercises trusted execution; test workers never qualify real acceptance."""

import json

import pytest

from bridge_evidence.contest.rsi import StrategyBundle, apply_patch, base_action
from bridge_evidence.contest.rsi_sandbox import PolicyController


def bundles():
    parent = StrategyBundle.baseline({"input_budget": 4000, "output_budget": 500})
    child = apply_patch(
        parent,
        {
            "parent_sha256": parent.digest,
            "target_function": "choose_action",
            "new_body": "return {'action': 'fail'}",
        },
    )
    return parent, child


def test_controller_reports_why_strategy_did_not_execute():
    parent, _ = bundles()
    with PolicyController(parent) as policy:
        policy.choose(
            {
                "phase": "schedule",
                "batch_size": 1,
                "remaining_calls": 2,
                "labels": {"secret": "value"},
                "path": "/private/data",
            }
        )
        assert policy.decisions[-1]["fallback_reason"] == "sandbox-unavailable"
        assert "labels" not in policy.decisions[-1]["state"]
        assert policy.telemetry()["executed_decisions"] == 0
        assert policy.telemetry()["execution_rate"] == 0


def test_failure_outcomes_are_sanitized_before_reaching_search_code():
    from bridge_evidence.contest.rsi_experiments import search_history

    outcomes = search_history(
        [
            {
                "status": "failed",
                "error": "/labels/gold.json SECRET",
                "f1": [0.2, 0.3, 0.4],
                "failures": 2,
                "wall_seconds": 4.0,
                "omissions": 1,
                "config": StrategyBundle.baseline().config,
                "output_dir": "/private/results",
            }
        ]
    )
    assert outcomes[0]["status"] == "failed"
    assert outcomes[0]["failures"] == 2
    assert outcomes[0]["wall_seconds"] == 4.0
    encoded = json.dumps(outcomes)
    assert all(v not in encoded for v in ("f1", "SECRET", "/private", "/labels"))


def test_replay_includes_real_sanitized_states_and_keeps_edge_probes(monkeypatch):
    from bridge_evidence.contest import rsi_experiments as exp

    seen = []

    class Worker:
        def __init__(self, *a, **kw):
            pass

        def invoke(self, fn, args, **kw):
            seen.append(args[0])
            return base_action(args[0])

        def close(self):
            pass

    monkeypatch.setattr(exp, "SandboxWorker", Worker)
    records = [
        {
            "rsi": {
                "decisions": [
                    {
                        "state": {
                            "phase": "schedule",
                            "remaining_calls": 7,
                            "remaining_seconds": 10,
                            "batch_size": 1,
                            "structure": {"table_rows": 11, "characters": 500},
                            "label": "hidden",
                            "path": "/private",
                        }
                    }
                ]
            }
        }
    ]
    evidence = exp.replay(StrategyBundle.baseline(), "test-root", records=records)
    assert any(v.get("structure", {}).get("table_rows") == 11 for v in seen)
    assert any(v.get("error_kind") == "length" for v in seen)
    assert all("path" not in v and "label" not in v for v in seen)
    assert any(e["origin"] == "recorded" for e in evidence)


def test_shadow_cannot_succeed_when_controller_rejects_extraction(monkeypatch):
    from test_chunked_integration import CharTokenizer, FragmentBackend

    from bridge_evidence.contest import rsi_experiments as exp
    from bridge_evidence.contest.long_document import (
        DocumentSnapshot,
        plan_chunks,
        split_task,
    )

    parent, child = bundles()
    snapshot = DocumentSnapshot.from_text("first\nsecond\n")
    plan = plan_chunks(snapshot, CharTokenizer(), input_budget=4000, output_budget=500)
    tasks = split_task(plan.tasks[0], snapshot)

    class Policy:
        def __init__(self, bundle, root):
            self.bundle, self.decisions = bundle, []

        def wait_ready(self, *a, **kw):
            return True

        def choose(self, state):
            action = (
                {"action": "fail"}
                if self.bundle.digest == child.digest
                else base_action(state)
            )
            self.decisions.append(
                {
                    "state": state,
                    "action": action,
                    "fallback": False,
                    "bundle_sha256": self.bundle.digest,
                }
            )
            return action

        def close(self):
            pass

    class Backend(FragmentBackend):
        def call_with_role(self, ids, maximum, role):
            result = self(ids, maximum)
            value = json.loads(result["text"])
            prompt = "".join(map(chr, ids))
            owned = json.JSONDecoder().raw_decode(prompt[prompt.index('{"owned":') :])[
                0
            ]["owned"]
            value["基本信息"] = {"桥梁名称": "测试桥"}
            value["basic_sources"] = {"桥梁名称": owned[0]["source_id"]}
            result["text"] = json.dumps(value, ensure_ascii=False)
            result.update(call_id=f"call-{self.calls}", latency_seconds=0.1)
            return result

    monkeypatch.setattr(exp, "PolicyController", Policy)
    with pytest.raises(Exception, match="(fail|reject|incomplete|coverage|exhausted)"):
        exp.shadow(
            parent,
            child,
            snapshot,
            tasks,
            CharTokenizer(),
            Backend(),
            sandbox_root="test-root",
        )


def test_config_only_ready_flags_cannot_activate_strategy(tmp_path):
    from bridge_evidence.contest.rsi_evolution import EvolutionStore, OnlineCoordinator

    parent, child = bundles()
    with EvolutionStore(tmp_path / "evolution.sqlite", "model", "evaluation") as store:
        store.add(parent, 0)
        store.add(child, 1)
        online = OnlineCoordinator(parent, store)
        assert not online.offer(
            child,
            evidence={
                "static": True,
                "replay": True,
                "shadow": True,
                "worker_ready": True,
                "bundle_sha256": child.digest,
                "shadow_call_ids": ["a", "b", "c", "d"],
            },
        )


@pytest.mark.parametrize(
    "failure,reason",
    [(TimeoutError, "decision-timeout"), (ValueError, "invalid-action")],
)
def test_worker_failure_is_attributed_and_late_worker_never_reused(
    monkeypatch, failure, reason
):
    import time

    from bridge_evidence.contest import rsi_sandbox

    killed = []

    class Process:
        pid = 12345

        def poll(self):
            return None

    class Worker:
        born = time.monotonic()
        process = Process()
        invoked = 0

        def poll(self):
            return self.process.poll()

        def terminate(self):
            rsi_sandbox.os.killpg(self.process.pid, 9)

        def invoke(self, name, args, timeout):
            assert timeout <= 0.005
            self.invoked += 1
            raise failure("not copied into telemetry /labels/secret")

        def close(self):
            pass

    monkeypatch.setattr(rsi_sandbox.os, "killpg", lambda pid, sig: killed.append(pid))
    policy = PolicyController(StrategyBundle.baseline())
    policy.root = "test-root"
    worker = Worker()
    policy.pool.append(worker)
    try:
        state = {"phase": "schedule", "batch_size": 1, "remaining_calls": 2}
        assert policy.choose(state)["action"] == "extract"
        assert policy.decisions[-1]["fallback_reason"] == reason
        assert policy.choose(state)["action"] == "extract"
        assert worker.invoked == 1
        assert killed == [12345]
        assert "/labels" not in json.dumps(policy.decisions)
    finally:
        policy.close()


@pytest.mark.parametrize("prompt", [
    "source-fragment-v1", "source-fragment-concise-v1", "source-fragment-exact-v1",
    "source-fragment-exact-v2",
])
def test_shadow_records_actual_actions_and_rejects_all_fallback_evidence(monkeypatch, prompt):
    from test_chunked_integration import CharTokenizer, FragmentBackend

    from bridge_evidence.contest import rsi_experiments as exp
    from bridge_evidence.contest.long_document import (
        DocumentSnapshot,
        plan_chunks,
        split_task,
    )
    from bridge_evidence.contest.rsi_sandbox import policy_state
    from bridge_evidence.contest.run_state import json_digest

    parent = StrategyBundle.baseline({"input_budget": 4000, "output_budget": 500,
                                      "prompt": prompt})
    child = apply_patch(
        parent,
        {
            "parent_sha256": parent.digest,
            "target_function": "choose_action",
            "new_body": "value = 1\nreturn base_action(state)",
        },
    )
    snapshot = DocumentSnapshot.from_text("测试桥\n裂缝")
    planned = plan_chunks(
        snapshot, CharTokenizer(), input_budget=4000, output_budget=500
    )

    class Policy:
        def __init__(self, bundle, root):
            self.bundle, self.decisions = bundle, []

        def wait_ready(self):
            return True

        def choose(self, state):
            state = policy_state(state)
            action = base_action(state)
            self.decisions.append(
                {
                    "state": state,
                    "state_sha256": json_digest(state),
                    "bundle_sha256": self.bundle.digest,
                    "action": action,
                    "fallback": False,
                    "kernel_proof": {
                        "seccomp": 2,
                        "no_new_privs": 1,
                        "capabilities": 0,
                        "namespaces": 6,
                        "readonly_root": True,
                    },
                    "sandbox_runtime_digest": "0" * 64,
                }
            )
            return action

        def close(self):
            pass

    class Backend(FragmentBackend):
        def call_with_role(self, ids, maximum, role):
            result = self(ids, maximum)
            data = json.loads(result["text"])
            prompt = "".join(map(chr, ids))
            owned = json.JSONDecoder().raw_decode(prompt[prompt.index('{"owned":') :])[
                0
            ]["owned"]
            data["基本信息"] = {"桥梁名称": "测试桥"}
            data["basic_sources"] = {"桥梁名称": owned[0]["source_id"]}
            result.update(
                text=json.dumps(data, ensure_ascii=False),
                call_id=f"call-{self.calls}",
                latency_seconds=0.1,
            )
            return result

    monkeypatch.setattr(exp, "PolicyController", Policy)
    result = exp.shadow(
        parent,
        child,
        snapshot,
        split_task(planned.tasks[0], snapshot),
        CharTokenizer(),
        Backend(),
        sandbox_root="unit-test",
    )
    calls = exp.shadow_evidence(result, parent.digest, child.digest)
    assert len(calls) == 4 and len({c["call_id"] for c in calls}) == 4
    assert result["effectiveness"]["strategy_exercised"] is True
    assert result["effectiveness"]["policy_effect_observed"] is False
    assert result["effectiveness"]["semantic_quality_verified"] is False
    for decision in result["runs"][1]["record"]["rsi"]["decisions"]:
        decision["fallback"] = True
    with pytest.raises(ValueError, match="never executed"):
        exp.shadow_evidence(result, parent.digest, child.digest)


def test_restart_does_not_upgrade_legacy_activation_flags(tmp_path):
    from bridge_evidence.contest.rsi_evolution import EvolutionStore, OnlineCoordinator

    parent, child = bundles()
    with EvolutionStore(tmp_path / "evolution.sqlite", "model", "evaluation") as store:
        store.add(parent, 0)
        store.add(child, 1)
        store.event("experimental-activate", {"bundle_sha256": child.digest})
        online = OnlineCoordinator(parent, store)
        assert online.pin().digest == parent.digest


def test_ready_receipt_must_bind_actual_model_output(monkeypatch, tmp_path):
    from bridge_evidence.contest import rsi_experiments as exp
    from bridge_evidence.contest.timing import TimingLedger

    parent, child = bundles()
    # Isolate receipt validation; full executor/namespace checks are covered separately.
    calls = [
        {
            "call_id": "shadow",
            "prompt_sha256": "prompt",
            "seconds": 0.1,
            "output_sha256": "expected",
            "output_tokens": 3,
            "finish_reason": "stop",
        }
    ]
    monkeypatch.setattr(exp, "shadow_evidence", lambda *a: calls)
    ledger = TimingLedger(tmp_path / "ledger.sqlite")
    for key, role in [("proposal", "rsi-code-proposal"), ("shadow", "rsi-shadow")]:
        ledger.start(key, role=role)
        ledger.finish(
            key,
            0.1,
            status="success",
            receipt={
                "backend_request_id": "test-only",
                "prompt_sha256": "prompt",
                "output_sha256": "tampered",
                "output_tokens": 3,
                "finish_reason": "stop",
            },
        )
    exp.write_ready(
        tmp_path / "ready",
        child,
        [{"call_id": "proposal"}],
        [{"origin": "unit-test"}],
        {},
        ledger_path=ledger.path,
    )
    with pytest.raises(ValueError, match="receipt"):
        exp.load_ready(tmp_path / "ready", parent)


@pytest.mark.parametrize(
    "instances,rejected",
    [
        ((None, None), True),
        (("engine-one", "engine-one"), True),
        (("engine-one", "engine-two"), False),
    ],
)
def test_ready_rejects_duplicate_backend_request_identity(
    monkeypatch, tmp_path, instances, rejected
):
    from bridge_evidence.contest import rsi_experiments as exp
    from bridge_evidence.contest.timing import TimingLedger

    parent, child = bundles()
    calls = [
        {
            "call_id": "shadow",
            "prompt_sha256": "prompt",
            "seconds": 0.1,
            "output_sha256": "output",
            "output_tokens": 3,
            "finish_reason": "stop",
        }
    ]
    # Isolate ledger deduplication from independently tested source/worker evidence.
    monkeypatch.setattr(exp, "shadow_evidence", lambda *args: calls)
    ledger = TimingLedger(tmp_path / "ledger.sqlite")
    for (call_id, role), instance in zip(
        [("proposal", "rsi-code-proposal"), ("shadow", "rsi-shadow")], instances
    ):
        ledger.claim(call_id, role=role)
        ledger.finish(
            call_id,
            0.1,
            status="success",
            receipt={
                "backend_request_id": "0",
                "backend_instance_id": instance,
                "prompt_sha256": "prompt",
                "output_sha256": "output",
                "output_tokens": 3,
                "finish_reason": "stop",
            },
        )
    exp.write_ready(
        tmp_path / "ready",
        child,
        [{"call_id": "proposal"}],
        [{"origin": "unit-test"}],
        {},
        ledger_path=ledger.path,
    )
    if rejected:
        with pytest.raises(ValueError, match="duplicate.*backend"):
            exp.load_ready(tmp_path / "ready", parent)
    else:
        restored, _ = exp.load_ready(tmp_path / "ready", parent)
        assert restored.digest == child.digest


def test_root_attestation_rejects_modified_content_before_spawn(tmp_path):
    from bridge_evidence.contest import rsi_sandbox as sandbox
    from bridge_evidence.contest.run_state import file_digest, json_digest

    root = tmp_path / "root"
    root.mkdir()
    runtime = root / "worker.py"
    runtime.write_text("first")
    files = {"worker.py": file_digest(runtime)}
    (tmp_path / "root.identity.json").write_text(
        json.dumps({"files": files, "digest": json_digest(files)})
    )
    verified = sandbox._verify_runtime(root)
    verified.check_unchanged()
    runtime.write_text("other")  # Same size; mtime/ctime/inode checks still reject it.
    with pytest.raises(ValueError, match="changed"):
        verified.check_unchanged()


def test_root_attestation_rejects_change_during_full_hash(monkeypatch, tmp_path):
    from bridge_evidence.contest import rsi_sandbox as sandbox
    from bridge_evidence.contest.run_state import file_digest, json_digest

    root = tmp_path / "root"
    root.mkdir()
    runtime = root / "worker.py"
    runtime.write_text("first")
    files = {"worker.py": file_digest(runtime)}
    (tmp_path / "root.identity.json").write_text(
        json.dumps({"files": files, "digest": json_digest(files)})
    )

    def racing_digest(path):
        digest = file_digest(path)
        Path(path).write_text("other")
        return digest

    from pathlib import Path

    monkeypatch.setattr(sandbox, "file_digest", racing_digest)
    with pytest.raises(ValueError, match="changed"):
        sandbox._verify_runtime(root)


def test_prewarm_round_hashes_once_before_starting_both_workers(monkeypatch):
    import time

    from bridge_evidence.contest import rsi_sandbox as sandbox

    clock = [time.monotonic()]
    validations, created = [], []

    class Attestation:
        def check_unchanged(self):
            pass

    def full_validation(root):
        validations.append(root)
        clock[0] += 4  # A real slow persistent volume can exceed worker's 3 s lifetime.
        return Attestation()

    class Process:
        def poll(self):
            return None

    class Worker:
        def __init__(self, root, bundle, *, _verified_runtime=None):
            if _verified_runtime is None:
                full_validation(root)
            self.born = clock[0]
            self.process = Process()
            created.append(self)
            if len(created) > 2:
                raise RuntimeError("first worker expired while hashing for the second")

        def poll(self):
            return self.process.poll()

        def close(self):
            pass

    monkeypatch.setattr(sandbox, "_verify_runtime", full_validation, raising=False)
    monkeypatch.setattr(sandbox, "SandboxWorker", Worker)
    monkeypatch.setattr(sandbox.time, "monotonic", lambda: clock[0])
    policy = PolicyController(StrategyBundle.baseline(), "test-root")
    try:
        policy.ready.wait(1)
        with policy.lock:
            assert len(validations) == 1
            assert len(policy.pool) == 2
            assert all(not policy._expired(worker) for worker in policy.pool)
    finally:
        policy.close()


def test_controller_skips_expired_worker_when_fresh_worker_is_ready(monkeypatch):
    import time

    from bridge_evidence.contest.rsi_sandbox import PolicyController

    monkeypatch.setattr(
        "bridge_evidence.contest.rsi_sandbox.os.killpg", lambda *args: None
    )

    class Process:
        pid = 12345

        def poll(self):
            return None

    class Worker:
        def __init__(self, age):
            self.born = time.monotonic() - age
            self.process = Process()
            self.kernel_proof = {"seccomp": 2}
            self.identity = {"digest": "0" * 64}

        def poll(self):
            return self.process.poll()

        def invoke(self, name, args, timeout):
            assert timeout <= 0.005
            return {"action": "extract", "batch_size": 1}

        def close(self):
            pass

    policy = PolicyController(StrategyBundle.baseline())
    policy.root = "test-root"
    policy.pool.extend([Worker(2.9), Worker(0)])
    try:
        policy.choose({"phase": "schedule", "batch_size": 1, "remaining_calls": 2})
        assert policy.decisions[-1]["fallback"] is False
        assert len(policy.retired) == 2
    finally:
        policy.close()


@pytest.mark.parametrize("escape", ["identity-path", "symlink", "unlisted-file"])
def test_root_attestation_rejects_path_escape_and_file_set_changes(tmp_path, escape):
    from bridge_evidence.contest import rsi_sandbox as sandbox
    from bridge_evidence.contest.run_state import file_digest, json_digest

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("outside")
    worker = root / "worker.py"
    worker.write_text("worker")
    files = {"worker.py": file_digest(worker)}
    if escape == "identity-path":
        files = {"../outside.py": file_digest(outside)}
    elif escape == "symlink":
        worker.unlink()
        worker.symlink_to(outside)
    else:
        (root / "unexpected.py").write_text("unlisted")
    (tmp_path / "root.identity.json").write_text(
        json.dumps({"files": files, "digest": json_digest(files)})
    )
    with pytest.raises(ValueError, match="(escape|changed)"):
        sandbox._verify_runtime(root)


def test_worker_watchdog_kills_idle_session_during_slow_prewarm():
    import signal
    import subprocess
    import sys
    import threading
    import time

    from bridge_evidence.contest import rsi_sandbox as sandbox

    born = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    lifetime = None
    try:
        lifetime = sandbox._WorkerLifetime(process, born)
        # The prewarm thread does not poll/retire workers during a slow root hash.
        threading.Event().wait(3.3)
        assert lifetime.poll() == -signal.SIGKILL
        assert lifetime.expired is True
    finally:
        if lifetime:
            lifetime.close()
        elif process.poll() is None:
            process.kill()
        process.wait(timeout=2)


def test_worker_watchdog_close_cancels_late_callback(monkeypatch):
    import signal
    import subprocess
    import sys
    import time

    from bridge_evidence.contest import rsi_sandbox as sandbox

    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    original_killpg, kills = sandbox.os.killpg, []

    def killpg(pid, sig):
        kills.append((pid, sig))
        original_killpg(pid, sig)

    monkeypatch.setattr(sandbox.os, "killpg", killpg)
    lifetime = None
    try:
        lifetime = sandbox._WorkerLifetime(process, time.monotonic())
        lifetime.close()
        process.wait(timeout=2)
        lifetime._expire()  # Timer can already be queued when close() cancels it.
        assert kills == [(process.pid, signal.SIGKILL)]
        assert lifetime.expired is False
    finally:
        if lifetime:
            lifetime.close()
        elif process.poll() is None:
            process.kill()
        process.wait(timeout=2)


def test_worker_watchdog_does_not_signal_reaped_pid(monkeypatch):
    import time

    from bridge_evidence.contest import rsi_sandbox as sandbox

    class Process:
        pid = 12345
        returncode = None

        def poll(self):
            self.returncode = 0
            return self.returncode

    kills = []
    monkeypatch.setattr(sandbox.os, "killpg", lambda *args: kills.append(args))
    lifetime = sandbox._WorkerLifetime(Process(), time.monotonic())
    try:
        assert lifetime.poll() == 0
        lifetime._expire()
        lifetime.close()
        assert kills == []
    finally:
        lifetime.close()


def test_controller_foreground_does_not_import_schema(monkeypatch):
    import builtins

    policy = PolicyController(StrategyBundle.baseline())
    original = builtins.__import__

    def no_schema_import(name, *args, **kwargs):
        if name.endswith("inference_core"):
            pytest.fail("schema loaded on foreground decision path")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_schema_import)
    try:
        assert policy.choose({"phase": "schedule", "remaining_calls": 1})["action"] == "extract"
    finally:
        policy.close()


def test_stdlib_archive_keeps_code_and_native_extensions_separate(tmp_path):
    import sys
    import zipfile

    from bridge_evidence.contest import rsi_sandbox as sandbox

    source, root = tmp_path / "stdlib", tmp_path / "root"
    for folder in ("encodings", "lib-dynload", "__pycache__", "test", "site-packages"):
        (source / folder).mkdir(parents=True)
    (source / "os.py").write_text("# os")
    (source / "encodings/__init__.py").write_text("# encoding bootstrap")
    (source / "lib-dynload/_test.so").write_bytes(b"native-library")
    (source / "__pycache__/os.pyc").write_bytes(b"cache")
    (source / "test/test_os.py").write_text("# tests")
    (source / "site-packages/private.py").write_text("# unrelated package")
    destination = sandbox._stage_stdlib(source, root)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    archive = root / f"lib/python{version.replace('.', '')}.zip"
    assert destination == root / f"lib/python{version}"
    assert (destination / "lib-dynload/_test.so").read_bytes() == b"native-library"
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == {"os.py", "encodings/__init__.py"}
        assert all(entry.compress_type == zipfile.ZIP_STORED for entry in zipped.infolist())
    assert len([p for p in root.rglob("*") if p.is_file()]) == 3


def test_stdlib_archive_rejects_source_symlink_escape(tmp_path):
    from bridge_evidence.contest import rsi_sandbox as sandbox

    source = tmp_path / "stdlib"
    source.mkdir()
    private = tmp_path / "private.py"
    private.write_text("# outside selected runtime")
    (source / "os.py").symlink_to(private)
    with pytest.raises(ValueError, match="symlink"):
        sandbox._stage_stdlib(source, tmp_path / "root")


def test_real_interpreter_bootstraps_encoding_and_worker_imports_from_archive(tmp_path):
    import os
    import subprocess
    import sys
    import sysconfig
    from pathlib import Path

    from bridge_evidence.contest import rsi_sandbox as sandbox

    root = tmp_path / "runtime"
    sandbox._stage_stdlib(Path(sysconfig.get_path("stdlib")), root)
    code = (
        "import ast, ctypes, json, os, resource, socket, sys, encodings; "
        "print(json.dumps({'encoding': encodings.__file__, 'paths': sys.path}))"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-c", code],
        env={"PATH": os.defpath, "PYTHONHOME": str(root)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    imported = json.loads(result.stdout)
    assert imported["encoding"].startswith(str(root / "lib"))
    assert ".zip/encodings/__init__.py" in imported["encoding"]
    assert all(not path or path.startswith(str(root)) for path in imported["paths"])


def _stdlib_bootstrap_source():
    import ast
    from pathlib import Path

    from bridge_evidence.contest import rsi_sandbox

    source = Path(rsi_sandbox.__file__).with_name("rsi_worker.py").read_text()
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "_install_stdlib_map"]
    assert functions, "trusted memory importer must be installed before worker imports"
    return ast.get_source_segment(source, functions[0])


def test_stdlib_code_map_contains_bound_compiled_modules(tmp_path):
    import _frozen_importlib_external
    import marshal
    import sys

    from bridge_evidence.contest import rsi_sandbox as sandbox

    source, root = tmp_path / "stdlib", tmp_path / "root"
    (source / "example").mkdir(parents=True)
    (source / "example/__init__.py").write_text("answer = 42")
    sandbox._stage_stdlib(source, root)
    payload = marshal.loads((root / "stdlib.marshal").read_bytes())
    assert payload["python_version"] == sys.version
    assert payload["cache_tag"] == sys.implementation.cache_tag
    assert payload["marshal_version"] == marshal.version
    assert payload["magic_number"] == _frozen_importlib_external.MAGIC_NUMBER
    package, code = payload["modules"]["example"]
    assert package is True
    assert code.co_filename == f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/example/__init__.py"
    namespace = {}
    exec(code, namespace)
    assert namespace["answer"] == 42


def test_parent_attestation_rejects_valid_code_map_content_tampering(tmp_path):
    import marshal

    from bridge_evidence.contest import rsi_sandbox as sandbox
    from bridge_evidence.contest.run_state import file_digest, json_digest

    source, root = tmp_path / "stdlib", tmp_path / "root"
    source.mkdir()
    (source / "example.py").write_text("answer = 42")
    sandbox._stage_stdlib(source, root)
    files = {str(p.relative_to(root)): file_digest(p) for p in root.rglob("*") if p.is_file()}
    (tmp_path / "root.identity.json").write_text(json.dumps({"files": files, "digest": json_digest(files)}))
    attestation = sandbox._verify_runtime(root)
    payload = marshal.loads((root / "stdlib.marshal").read_bytes())
    original = payload["modules"]["example"][1]
    payload["modules"]["example"] = (False, compile("answer = 43", original.co_filename, "exec"))
    (root / "stdlib.marshal").write_bytes(marshal.dumps(payload))
    with pytest.raises(ValueError, match="changed"):
        attestation.check_unchanged()
    with pytest.raises(ValueError, match="changed"):
        sandbox._verify_runtime(root)


@pytest.mark.parametrize("tamper", ["version", "tag", "marshal", "magic", "path", "code", "name"])
def test_memory_importer_rejects_tampered_map_before_install(tmp_path, tamper):
    import _frozen_importlib_external
    import marshal
    import sys

    source = _stdlib_bootstrap_source()
    namespace = {}
    exec(source, namespace)
    filename = f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/example.py"
    payload = {
        "schema_version": 1,
        "python_version": sys.version,
        "cache_tag": sys.implementation.cache_tag,
        "marshal_version": marshal.version,
        "magic_number": _frozen_importlib_external.MAGIC_NUMBER,
        "modules": {"example": (False, compile("answer = 42", filename, "exec"))},
    }
    if tamper in ("version", "tag", "marshal", "magic"):
        field = {"version": "python_version", "tag": "cache_tag", "marshal": "marshal_version", "magic": "magic_number"}[tamper]
        payload[field] = "changed"
    elif tamper == "path":
        payload["modules"]["example"] = (False, compile("answer = 42", "/host/secret.py", "exec"))
    elif tamper == "code":
        payload["modules"]["example"] = (False, "not compiled code")
    else:
        payload["modules"]["../escape"] = payload["modules"].pop("example")
    path = tmp_path / "tampered.marshal"
    path.write_bytes(marshal.dumps(payload))
    before = list(sys.meta_path)
    with pytest.raises(ValueError):
        namespace["_install_stdlib_map"](str(path))
    assert sys.meta_path == before


def test_memory_importer_rejects_oversize_map_before_deserializing(tmp_path):
    import sys

    namespace = {}
    exec(_stdlib_bootstrap_source(), namespace)
    path = tmp_path / "oversize.marshal"
    with path.open("wb") as output:
        output.truncate(64 * 1024 * 1024 + 1)
    before = list(sys.meta_path)
    with pytest.raises(ValueError, match="exceeds limit"):
        namespace["_install_stdlib_map"](str(path))
    assert sys.meta_path == before


def _worker_import_rss_bytes(platform, maxrss, proc_status):
    """Measure this exec image, without Linux's retained pre-exec RSS peak."""
    if platform.startswith("linux"):
        # Linux exec retains old_mm's high-water mark in signal->maxrss, which
        # getrusage includes. /proc/self/status VmHWM belongs to the current mm.
        # See linux/fs/exec.c, kernel/sys.c and docs.kernel.org/filesystems/proc.html.
        rows = [line.split() for line in proc_status.splitlines()
                if line.startswith("VmHWM:")]
        if (len(rows) != 1 or len(rows[0]) != 3 or rows[0][2] != "kB"
                or not rows[0][1].isascii() or not rows[0][1].isdigit()
                or int(rows[0][1]) <= 0):
            raise ValueError("Missing or malformed Linux VmHWM measurement")
        return int(rows[0][1]) * 1024
    return maxrss if platform == "darwin" else maxrss * 1024


@pytest.mark.parametrize("peak_kib", [80_000, 262_144, 300_000])
def test_worker_rss_uses_linux_exec_peak_without_hiding_over_limit_values(peak_kib):
    # A fork/exec child can retain a much larger pre-exec getrusage maximum.
    status = f"Name:\tpython\nVmHWM:\t{peak_kib} kB\nVmRSS:\t12000 kB\n"
    measured = _worker_import_rss_bytes("linux", 600_000, status)
    assert measured == peak_kib * 1024
    assert (measured < 256 * 1024 * 1024) == (peak_kib < 262_144)


@pytest.mark.parametrize("status", [
    "", "VmRSS: 12000 kB\n", "VmHWM: 1 MB\n", "VmHWM: -1 kB\n",
    "VmHWM: 0 kB\n", "VmHWM: invalid kB\n", "VmHWM: 1 kB extra\n",
    "VmHWM: 1 kB\nVmHWM: 2 kB\n",
])
def test_worker_rss_rejects_missing_or_malformed_linux_measurement(status):
    with pytest.raises(ValueError, match="VmHWM"):
        _worker_import_rss_bytes("linux", 1, status)


def test_worker_rss_preserves_darwin_byte_units():
    assert _worker_import_rss_bytes("darwin", 80_000_000, "") == 80_000_000


def test_real_memory_importer_loads_worker_modules_and_keeps_builtin_priority(tmp_path):
    import inspect
    import marshal
    import os
    import subprocess
    import sys
    import sysconfig

    from bridge_evidence.contest import rsi_sandbox as sandbox

    root = tmp_path / "runtime"
    sandbox._stage_stdlib(sysconfig.get_path("stdlib"), root)
    payload = marshal.loads((root / "stdlib.marshal").read_bytes())
    for name in ("errno", "zipimport"):
        filename = f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/{name}.py"
        payload["modules"][name] = (False, compile("raise AssertionError('shadowed builtin or frozen')", filename, "exec"))
    (root / "stdlib.marshal").write_bytes(marshal.dumps(payload))
    bootstrap = _stdlib_bootstrap_source()
    code = inspect.getsource(_worker_import_rss_bytes) + "\n" + bootstrap
    code += f"\n_install_stdlib_map({str(root / 'stdlib.marshal')!r})\n" + (
        "import sys\n"
        "sys.path[:] = [p for p in sys.path if p.endswith('lib-dynload')]\n"
        "sys.modules.pop('errno', None); sys.modules.pop('zipimport', None)\n"
        "import errno, zipimport\n"
        "import ast, ctypes, json, os, resource, socket, selectors, marshal, _frozen_importlib\n"
        "assert ast.__loader__.__class__.__name__ == 'MemoryStdlibFinder'\n"
        "assert selectors.__loader__.__class__.__name__ == 'MemoryStdlibFinder'\n"
        "assert sys.meta_path.index(_frozen_importlib.BuiltinImporter) < 2\n"
        "assert sys.meta_path.index(_frozen_importlib.FrozenImporter) < 2\n"
        "assert errno.__spec__.loader is _frozen_importlib.BuiltinImporter\n"
        "assert zipimport.__spec__.loader is _frozen_importlib.FrozenImporter\n"
        "status = ''\n"
        "if sys.platform.startswith('linux'):\n"
        "    with open('/proc/self/status', encoding='ascii') as source:\n"
        "        status = source.read()\n"
        "rusage_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "rss_bytes = _worker_import_rss_bytes(sys.platform, rusage_peak, status)\n"
        "print(json.dumps({'modules': len(sys.modules), 'rss_bytes': rss_bytes, 'ru_maxrss': rusage_peak}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-c", code],
        env={"PATH": os.defpath, "PYTHONHOME": str(root)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    measured = json.loads(result.stdout)
    assert measured["modules"] > 10
    assert 0 < measured["rss_bytes"] < 256 * 1024 * 1024, measured
