"""Exercise real coordinator CLI branches without starting models or cloud jobs."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from bridge_evidence.contest.candidate import validate_config
from bridge_evidence.contest.rsi import StrategyBundle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:8000/v1"
CUSTOM_URL = "http://127.0.0.1:8011/v1"


def cli_module(monkeypatch, name):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name + "_endpoint_test", ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_cli(monkeypatch, tmp_path, module, name, *, url=None, workflow="chunked"):
    config = validate_config({"workflow": workflow})
    for filename, value in (("baseline", config), ("model", {}), ("manifest", {})):
        (tmp_path / (filename + ".json")).write_text(json.dumps(value))
    output, budget, sandbox = tmp_path / "output", tmp_path / "budget.sqlite", tmp_path / "sandbox"
    command = [name, "--manifest", str(tmp_path / "manifest.json"),
               "--model-manifest", str(tmp_path / "model.json"),
               "--baseline-config", str(tmp_path / "baseline.json"), "--hardware", "synthetic-endpoint-test"]
    if name == "run_rsi_campaign":
        output.mkdir()
        command += ["--root", str(output), "--sdk-acceptance", str(tmp_path / "sdk.json")]
    else:
        command += ["--output-dir", str(output), "--budget-ledger", str(budget), "--sandbox-root", str(sandbox)]
    if url is not None:
        command += ["--url", url]
    resolve = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda path, *a, **kw: (
        Path("/workspace") / path.name if path in (output, budget, sandbox) else resolve(path, *a, **kw)))
    monkeypatch.setattr(module.sys, "argv", command)
    return output, config


class UnspentBudget:
    def __init__(self, path):
        pass

    def remaining(self):
        return 80000


def argument(command, flag):
    assert command.count(flag) == 1, command
    return command[command.index(flag) + 1]


@pytest.mark.parametrize("url", [None, CUSTOM_URL, CUSTOM_URL + "/"])
def test_campaign_forwards_endpoint_to_actual_sdk_and_evolution_launches(monkeypatch, tmp_path, url):
    module = cli_module(monkeypatch, "run_rsi_campaign")
    output, _ = prepare_cli(monkeypatch, tmp_path, module, "run_rsi_campaign", url=url)
    with sqlite3.connect(output / "budget.sqlite") as db:
        db.execute("CREATE TABLE jobs(id TEXT, status TEXT)")
    commands = []
    monkeypatch.setattr(module, "CloudBudget", UnspentBudget)
    monkeypatch.setattr(module, "run_job", lambda budget, phase, cap, command, log: commands.append(command))

    def execute(command, *, check):
        commands.append(command)
        destination = Path(argument(command, "--output-dir"))
        destination.mkdir()
        (destination / "selection.json").write_text('{"selected":"baseline"}')
        assert check is True

    monkeypatch.setattr(module.subprocess, "run", execute)
    module.main()
    assert [Path(c[1]).name for c in commands] == [
        "probe_rsi_sandbox.py", "probe_rsi_parallel_sdk.py", "run_rsi_evolution.py"]
    assert "--url" not in commands[0]  # The isolation probe never contacts the model.
    for command in commands[1:]:
        assert argument(command, "--url") == (url or DEFAULT_URL).rstrip("/")
    assert json.loads((output / "rsi-campaign-status.json").read_text())["state"] == "complete"


@pytest.mark.parametrize("url", [None, CUSTOM_URL, CUSTOM_URL + "/"])
def test_evolution_forwards_endpoint_through_anchor_optimizer_candidate_retest_and_proposal(
    monkeypatch, tmp_path, url
):
    module = cli_module(monkeypatch, "run_rsi_evolution")
    output, config = prepare_cli(monkeypatch, tmp_path, module, "run_rsi_evolution", url=url)
    optimizer = tmp_path / "optimizer.json"
    optimizer.write_text(json.dumps(StrategyBundle.baseline(config).as_dict()))
    module.sys.argv.extend(["--optimizer-parent-bundle", str(optimizer), "--optimizer-child-bundle", str(optimizer)])
    commands = []

    def run_job(budget, phase, cap, command, log):
        commands.append(command)
        if phase == "rsi-code-proposal":
            raise RuntimeError("synthetic boundary test does not generate code")
        if Path(command[1]).name != "evaluate_multitask.py":
            return
        destination = Path(argument(command, "--output-dir"))
        destination.mkdir()
        candidate = json.loads(Path(argument(command, "--candidate-config")).read_text())
        (destination / "evaluation.json").write_text(json.dumps({
            "config": candidate, "f1": [0.9, 0.8, 0.7], "failures": 0,
            "wall_seconds": 100 if destination.name == "anchor" else 60}))

    def receipt(directory, recovery, *, source_root):
        name = directory.name
        return {
            "run_id": name, "protocol": "frozen-rsi-v3", "comparison_identity": "same-synthetic-identity",
            "protected_sha256": "0" * 64, "evaluation_sha256": name, "bundle_sha256": "1" * 64,
            "denominator": 2, "completed": 2, "f1": [0.9, 0.8, 0.7], "failures": 0,
            "missing_sources": 0, "conflicts": 0, "wall_seconds": 100 if name == "anchor" else 60,
            "latency": {"p50": 1, "p95": 2},
            "recovery": {"cases": {"timeout": {"start_seconds": 1, "finish_seconds": 2}}},
            "cache_reused": False, "audit_passed": True,
            "generation_ledger": {"call_ids": [name], "backend_request_keys": ["synthetic-" + name]},
        }

    def sandbox(root, bundle, function, values, **kwargs):
        return [{**config, "batch_size": 2}] if function == "propose_candidates" else values[0]

    from bridge_evidence.contest import rsi_search_comparison

    def compare(parent, child, actor, anchor, root, evaluate, register, **kwargs):
        evaluate(actor, "optimizer-endpoint-test", cap=14400)
        return {"optimizer_verified": False}

    monkeypatch.setattr(module, "CloudBudget", UnspentBudget)
    monkeypatch.setattr(module, "verify_root", lambda *a: None)
    monkeypatch.setattr(module, "run_job", run_job)
    monkeypatch.setattr(module, "load_receipt", receipt)
    monkeypatch.setattr(module, "sandbox_call", sandbox)
    monkeypatch.setattr(module, "audited_optimizer_trial", lambda *a, **kw: {})
    monkeypatch.setattr(rsi_search_comparison, "compare_optimizers", compare)
    module.main()
    evaluations = [c for c in commands if Path(c[1]).name == "evaluate_multitask.py"]
    assert [Path(argument(c, "--output-dir")).name for c in evaluations] == [
        "anchor", "optimizer-endpoint-test", "g1-candidate-2", "g1-candidate-2-retest"]
    recoveries = [c for c in commands if Path(c[1]).name == "benchmark_rsi_recovery.py"]
    assert [argument(c, "--mode") for c in recoveries] == ["off", "frozen", "frozen"]
    assert len([c for c in commands if Path(c[1]).name == "propose_rsi_strategy.py"]) == 1
    for command in commands:
        assert argument(command, "--url") == (url or DEFAULT_URL).rstrip("/")
    assert json.loads((output / "selection.json").read_text())["selected"] == "candidate"


@pytest.mark.parametrize("name", ["run_rsi_campaign", "run_rsi_evolution"])
@pytest.mark.parametrize("url", ["https://127.0.0.1:8011/v1", "http://remote.invalid:8011/v1",
                                  "http://user:secret@127.0.0.1:8011/v1", "http://127.0.0.1:8011/v1?route=other"])
def test_invalid_endpoint_fails_before_budget_isolation_or_output(monkeypatch, tmp_path, capsys, name, url):
    module = cli_module(monkeypatch, name)
    output, _ = prepare_cli(monkeypatch, tmp_path, module, name, url=url)
    monkeypatch.setattr(module, "CloudBudget", lambda *a: pytest.fail("budget accessed"))
    monkeypatch.setattr(module, "run_job", lambda *a: pytest.fail("job launched"))
    if hasattr(module, "verify_root"):
        monkeypatch.setattr(module, "verify_root", lambda *a: pytest.fail("isolation accessed"))
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert "invalid" in error
    assert "unrecognized arguments" not in error
    assert not output.exists() or not list(output.iterdir())


@pytest.mark.parametrize("name", ["run_rsi_campaign", "run_rsi_evolution"])
def test_custom_endpoint_does_not_bypass_invalid_baseline(monkeypatch, tmp_path, capsys, name):
    module = cli_module(monkeypatch, name)
    output, _ = prepare_cli(monkeypatch, tmp_path, module, name, url=CUSTOM_URL, workflow="whole")
    monkeypatch.setattr(module, "CloudBudget", lambda *a: pytest.fail("budget accessed"))
    monkeypatch.setattr(module, "run_job", lambda *a: pytest.fail("job launched"))
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 2
    assert "whole baseline" in capsys.readouterr().err
    assert not output.exists() or not list(output.iterdir())
