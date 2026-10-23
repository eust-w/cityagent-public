"""Off-mode recovery has no candidate worker; frozen isolation stays mandatory."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bridge_evidence.contest.rsi import StrategyBundle
from bridge_evidence.contest.rsi_acceptance import RECOVERY_TEXT

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def driver(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "off_sandbox_recovery_driver", ROOT / "scripts/benchmark_rsi_process_recovery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return module


def arguments(tmp_path, *, mode="off"):
    return ["--output-dir", str(tmp_path / "run"), "--model-manifest", str(tmp_path / "model.json"),
            "--bundle", str(tmp_path / "bundle.json"), "--mode", mode]


def test_off_cli_accepts_omitted_sandbox(driver, tmp_path):
    args = driver.build_parser().parse_args(arguments(tmp_path))
    assert args.mode == "off" and args.sandbox_root is None


@pytest.mark.parametrize("worker_stage", [None, "crash-window", "resume"])
def test_frozen_missing_sandbox_rejected_before_budget_or_worker(driver, monkeypatch, tmp_path, worker_stage, capsys):
    argv = arguments(tmp_path, mode="frozen")
    if worker_stage:
        argv += ["--worker-stage", worker_stage]
    monkeypatch.setattr(sys, "argv", ["recovery.py", *argv])
    monkeypatch.setattr(driver, "CloudBudget", lambda *a, **kw: pytest.fail("budget accessed before validation"))
    monkeypatch.setattr(driver, "worker", lambda *a, **kw: pytest.fail("worker entered before validation"))
    monkeypatch.setattr(driver, "run_process_recovery", lambda *a, **kw: pytest.fail("supervisor entered before validation"))
    with pytest.raises(SystemExit) as caught:
        driver.main()
    assert caught.value.code == 2
    assert "frozen recovery requires --sandbox-root" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_direct_frozen_worker_rejects_missing_sandbox_before_model_io(driver, tmp_path):
    args = SimpleNamespace(mode="frozen", sandbox_root=None, output_dir=tmp_path / "absent")
    with pytest.raises(ValueError, match="frozen.*sandbox"):
        driver.worker(args, runtime_factory=lambda _: pytest.fail("model runtime entered"))
    assert not args.output_dir.exists()


def test_direct_frozen_supervisor_rejects_missing_sandbox_before_output(driver, tmp_path, monkeypatch):
    args = SimpleNamespace(mode="frozen", sandbox_root=None, output_dir=tmp_path / "absent")
    monkeypatch.setattr(driver, "protected_identity", lambda: pytest.fail("package IO before validation"))
    with pytest.raises(ValueError, match="frozen.*sandbox"):
        driver.run_process_recovery(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("provided", [False, True])
def test_off_worker_never_checks_unused_sandbox_and_binds_none(driver, tmp_path, monkeypatch, provided):
    output = tmp_path / "run"
    output.mkdir()
    (output / "source.txt").write_text(RECOVERY_TEXT)
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps(StrategyBundle.baseline().as_dict()))
    args = SimpleNamespace(mode="off", sandbox_root=tmp_path / "stale" if provided else None,
                           output_dir=output, worker_stage="crash-window", bundle=bundle)
    monkeypatch.setattr(driver, "verify_root", lambda *_: pytest.fail("off mode checked unused sandbox"))
    monkeypatch.setattr(driver, "PolicyController", lambda *a: pytest.fail("off mode created candidate worker"))
    monkeypatch.setattr(driver.ci, "parse_fragment", driver.ci.parse_fragment)

    def stop(*a, **kw):
        assert kw["policy"] is None
        raise InterruptedError("synthetic stop before model request")

    monkeypatch.setattr(driver.ci, "infer_chunked", stop)
    with pytest.raises(InterruptedError):
        driver.worker(args, runtime_factory=lambda _: (
            object(), SimpleNamespace(url="http://127.0.0.1:8011/v1"), {"test_only": True}))
    identity = json.loads((output / "crash-window.identity.json").read_text())
    assert identity["mode"] == "off" and identity["sandbox_digest"] is None
    assert identity["bundle_sha256"] is None


def test_frozen_worker_still_rejects_unverified_sandbox_before_inference(driver, tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    (output / "source.txt").write_text(RECOVERY_TEXT)
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps(StrategyBundle.baseline().as_dict()))
    root = tmp_path / "sandbox"
    args = SimpleNamespace(mode="frozen", sandbox_root=root, output_dir=output,
                           worker_stage="crash-window", bundle=bundle)
    checked = []

    def reject(path):
        checked.append(path)
        raise ValueError("synthetic stale sandbox evidence")

    monkeypatch.setattr(driver, "verify_root", reject)
    monkeypatch.setattr(driver.ci, "infer_chunked", lambda *a, **kw: pytest.fail("unverified policy inferred"))
    with pytest.raises(ValueError, match="stale sandbox"):
        driver.worker(args, runtime_factory=lambda _: (object(), object(), {"test_only": True}))
    assert checked == [root]


@pytest.mark.parametrize("mode", ["off", "frozen"])
def test_child_command_omits_none_and_keeps_frozen_sandbox(driver, tmp_path, monkeypatch, mode):
    argv = arguments(tmp_path, mode=mode)
    if mode == "frozen":
        argv += ["--sandbox-root", str(tmp_path / "sandbox")]
    args = driver.build_parser().parse_args(argv)
    commands = []

    def stop(command, **kwargs):
        commands.append(command)
        raise InterruptedError("synthetic stop before child launch")

    monkeypatch.setattr(driver.subprocess, "Popen", stop)
    with pytest.raises(InterruptedError):
        driver.run_process_recovery(args)
    if mode == "off":
        assert "--sandbox-root" not in commands[0] and "None" not in commands[0]
    else:
        assert commands[0][commands[0].index("--sandbox-root") + 1] == str(args.sandbox_root)
