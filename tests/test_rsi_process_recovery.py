"""Real SIGKILL/resume protocol using a fake backend, never PPU acceptance."""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def driver_module():
    spec = importlib.util.spec_from_file_location(
        "process_recovery_driver", ROOT / "scripts/benchmark_rsi_process_recovery.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_incomplete_copied_source_fails_before_starting_model_worker(tmp_path, monkeypatch):
    driver = driver_module()
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    output = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        driver.run_process_recovery(
            SimpleNamespace(output_dir=output),
            command_factory=lambda stage: pytest.fail("model worker launched for incomplete package"),
        )
    assert not output.exists()


def test_runtime_uses_pinned_prepared_identity_without_reloading_model(tmp_path, monkeypatch):
    driver = driver_module()
    import bridge_evidence.contest.model_identity as identities

    model = SimpleNamespace(tokenizer="fixed-tokenizer", served_model="fixed-lora")
    model.validate = lambda: model
    model.identity = lambda: pytest.fail("unnecessary full model rehash")
    monkeypatch.setattr(driver.ModelManifest, "load", lambda path: model)
    calls = []
    def load_identity(manifest, receipt, *, expected_sha256):
        calls.append((manifest, receipt, expected_sha256))
        return {"verified": "identity-fixture"}
    monkeypatch.setattr(identities, "load_identity", load_identity)
    tokenizer = object()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)))
    monkeypatch.setattr(driver, "APIBackend", lambda *a, **kw: "resident-api")
    args = SimpleNamespace(model_manifest=tmp_path / "model.json",
                           prepared_identity=tmp_path / "identity.json",
                           prepared_identity_sha256="a" * 64)
    assert driver.runtime(args) == (tokenizer, "resident-api", {"verified": "identity-fixture"})
    assert calls == [(model, args.prepared_identity, "a" * 64)]


def fake_runtime(args):
    from bridge_evidence.contest.run_state import atomic_json

    if args.worker_stage == "resume":
        time.sleep(0.15)  # A new executor's setup must remain inside recovery latency.

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(message["content"] for message in messages)

        def __call__(self, text, **kwargs):
            return {"input_ids": list(map(ord, text))}

    class Backend:
        def __call__(self, ids, maximum):
            assert args.worker_stage == "crash-window", "resume invoked the backend"
            counter = args.output_dir / "fake-calls.json"
            count = json.loads(counter.read_text()) if counter.exists() else 0
            atomic_json(counter, count + 1)
            prompt = "".join(map(chr, ids))
            payload = json.JSONDecoder().raw_decode(prompt[prompt.index('{"owned":'):])[0]
            refs = payload["owned"]
            value = {
                "covered": [ref["key"] for ref in refs],
                "基本信息": {"桥梁名称": "RSI恢复验证桥"},
                "basic_sources": {"桥梁名称": "p0"},
                "维修建议": [],
                "病害明细": [],
            }
            return {
                "text": json.dumps(value, ensure_ascii=False),
                "finish_reason": "stop",
                "output_tokens": 100,
                "backend_request_id": "fake-sigkill-1",
                "backend_instance_id": "fake-only-service-instance",
            }

        def batch(self, requests):
            return [self(*request) for request in requests]

    return Tokenizer(), Backend(), {"test_only": True}


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL and flock")
@pytest.mark.parametrize("mode", ["off", "frozen"])
def test_supervisor_kills_own_worker_then_resumes_durable_raw_without_model(tmp_path, mode):
    from bridge_evidence.contest.rsi import StrategyBundle

    driver = driver_module()
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps(StrategyBundle.baseline().as_dict()))
    args = SimpleNamespace(
        output_dir=tmp_path / "run", bundle=bundle, mode=mode,
        model_manifest=tmp_path / "model.json", sandbox_root=tmp_path / "sandbox",
        limit_seconds=30,
    )

    def command(stage):
        return [
            sys.executable, str(Path(__file__).resolve()), "--test-worker",
            str(args.output_dir), str(bundle), stage, mode,
        ]

    result = driver.run_process_recovery(args, command_factory=command)
    assert result["protocol"] == "rsi-process-recovery-supplement-v1"
    assert result["killed_returncode"] == -9
    assert result["pids"]["initial"] != result["pids"]["resume"]
    assert result["model_calls_before"] == result["model_calls_after"] == 1
    assert result["reused_call_ids"] == result["ledger"]["call_ids"]
    assert result["recovery"]["finish_seconds"] >= result["recovery"]["start_seconds"] > 0
    assert result["recovery"]["start_seconds"] >= result["recovery"]["owner_lock_seconds"]
    assert result["recovery"]["start_seconds"] >= 0.15
    assert json.loads((args.output_dir / "fake-calls.json").read_text()) == 1
    assert result["model"] == {"test_only": True}
    assert result["formal_release_evidence"] is False
    if mode == "frozen":
        from bridge_evidence.contest.rsi_runtime import TaskJournal

        recovered = json.loads((args.output_dir / "result.json").read_text())
        decisions = recovered["rsi"]["decisions"]
        assert len(decisions) == 1
        assert decisions[0]["kernel_proof"]["pid"] == result["pids"]["initial"]
        assert recovered["rsi"]["controller"]["executed_decisions"] == 0
        checkpoint = next((args.output_dir / "work/cache").glob("*/checkpoint.sqlite"))
        with TaskJournal(checkpoint, checkpoint.parent.name) as journal:
            assert journal.load()["decisions"] == decisions
        assert result["strategy_execution"]["executed_decisions"] == 1
        assert result["policy_phases"]["before_crash"]["executed_decisions"] == 1
        assert result["policy_phases"]["resume"] == {
            "status": "not-needed", "decisions": 0, "executed_decisions": 0,
        }


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL and flock")
def test_supervisor_timeout_terminates_its_unready_child(tmp_path):
    driver = driver_module()
    pid_path = tmp_path / "pid"
    args = SimpleNamespace(output_dir=tmp_path / "run", limit_seconds=0.3)

    def command(stage):
        return [
            sys.executable, "-c",
            "import os,sys,time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(30)",
            str(pid_path),
        ]

    with pytest.raises(TimeoutError, match="experiment deadline"):
        driver.run_process_recovery(args, command_factory=command)
    pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not (args.output_dir / "process-recovery.json").exists()


if __name__ == "__main__":
    assert sys.argv[1] == "--test-worker"
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    driver = driver_module()
    mode = sys.argv[5]
    if mode == "frozen":
        # This is explicitly a synthetic policy-proof fixture to test durable
        # checkpoint plumbing on macOS. It cannot certify real Linux isolation.
        from bridge_evidence.contest.rsi import base_action
        from bridge_evidence.contest.rsi_sandbox import policy_state
        from bridge_evidence.contest.run_state import json_digest

        class FixturePolicy:
            error = None

            def __init__(self, bundle, root):
                self.bundle, self.decisions = bundle, []

            def wait_ready(self, timeout):
                return True

            def choose(self, state):
                state = policy_state(state)
                action = base_action(state)
                self.decisions.append({
                    "state": state, "state_sha256": json_digest(state),
                    "action": action, "bundle_sha256": self.bundle.digest,
                    "fallback": False, "fallback_reason": None,
                    "sandbox_runtime_digest": "f" * 64,
                    "kernel_proof": {
                        "pid": os.getpid(), "seccomp": 2, "no_new_privs": 1,
                        "capabilities": 0, "namespaces": 6, "readonly_root": True,
                    },
                })
                return action

            def telemetry(self):
                return {"executed_decisions": len(self.decisions)}

            def close(self):
                pass

        driver.PolicyController = FixturePolicy
        driver.verify_root = lambda root: "f" * 64
    driver.worker(
        SimpleNamespace(
            output_dir=Path(sys.argv[2]), bundle=Path(sys.argv[3]),
            worker_stage=sys.argv[4], mode=mode, model_manifest=None,
            sandbox_root=Path("/test-only-sandbox") if mode == "frozen" else None,
        ),
        runtime_factory=fake_runtime,
    )
