"""Exercise the search CLI release boundary with explicitly synthetic trials."""

import importlib.util
import json
from pathlib import Path

import pytest

from bridge_evidence.contest.candidate import validate_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "rsi_search_cli", ROOT / "scripts/run_rsi_evolution.py"
)
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)


def setup_cli(monkeypatch, tmp_path, workflow="chunked"):
    output = tmp_path / "output"
    budget_path, sandbox = tmp_path / "budget.sqlite", tmp_path / "sandbox"
    config = validate_config({"workflow": workflow})
    for name, value in [("baseline", config), ("model", {}), ("manifest", {})]:
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    resolve = Path.resolve
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda path, *args, **kwargs: (
            Path("/workspace") / path.name
            if path in (output, budget_path, sandbox)
            else resolve(path, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        search.sys,
        "argv",
        [
            "run_rsi_evolution.py",
            "--manifest", str(tmp_path / "manifest.json"),
            "--model-manifest", str(tmp_path / "model.json"),
            "--baseline-config", str(tmp_path / "baseline.json"),
            "--sandbox-root", str(sandbox),
            "--budget-ledger", str(budget_path),
            "--output-dir", str(output),
            "--hardware", "synthetic-test",
        ],
    )
    return output, config


def test_whole_baseline_rejected_before_environment_or_cloud_work(
    monkeypatch, tmp_path, capsys
):
    output, _ = setup_cli(monkeypatch, tmp_path, "whole")
    monkeypatch.setattr(
        search, "verify_root", lambda *args: pytest.fail("environment work started")
    )
    monkeypatch.setattr(
        search, "CloudBudget", lambda *args: pytest.fail("budget work started")
    )
    with pytest.raises(SystemExit) as stopped:
        search.main()
    assert stopped.value.code == 2
    assert "whole" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("collision", ["call_ids", "backend_request_keys", None])
def test_release_requires_independent_anchor_and_both_trials(
    monkeypatch, tmp_path, collision
):
    output, config = setup_cli(monkeypatch, tmp_path, "bounded-react")

    class Budget:
        def __init__(self, path):
            pass

        def remaining(self):
            return 80000

    def run_job(budget, phase, cap, command, log):
        if phase == "rsi-code-proposal":
            raise RuntimeError("no code proposal in this synthetic boundary test")
        if "evaluate_multitask.py" not in command[1]:
            return
        destination = Path(command[command.index("--output-dir") + 1])
        cfg = Path(command[command.index("--candidate-config") + 1])
        destination.mkdir()
        (destination / "evaluation.json").write_text(
            json.dumps({
                "config": json.loads(cfg.read_text()),
                "f1": [0.9, 0.8, 0.7],
                "wall_seconds": 100 if destination.name == "anchor" else 60,
                "failures": 0,
            })
        )

    def receipt(directory, recovery, *, source_root):
        name = directory.name
        ledger = {"call_ids": [name], "backend_request_keys": ["vendor-" + name]}
        if collision and name in ("anchor", "g1-candidate-2"):
            ledger[collision] = ["reused"]
        return {
            "run_id": name,
            "protocol": "frozen-rsi-v3",
            "comparison_identity": "same-test-identity",
            "protected_sha256": "0" * 64,
            "evaluation_sha256": name,
            "bundle_sha256": "1" * 64,
            "denominator": 2,
            "completed": 2,
            "f1": [0.9, 0.8, 0.7],
            "failures": 0,
            "missing_sources": 0,
            "conflicts": 0,
            "wall_seconds": 100 if name == "anchor" else 60,
            "latency": {"p50": 1, "p95": 2},
            "recovery": {"cases": {"timeout": {
                "start_seconds": 1, "finish_seconds": 2,
            }}},
            "cache_reused": False,
            "audit_passed": True,
            "generation_ledger": ledger,
        }

    proposals = [0]

    def sandbox_call(root, bundle, function, values, **kwargs):
        if function == "update_search":
            return values[0]
        proposals[0] += 1
        return [{**config, "batch_size": 2}] if proposals[0] == 1 else []

    monkeypatch.setattr(search, "CloudBudget", Budget)
    monkeypatch.setattr(search, "verify_root", lambda *args: None)
    monkeypatch.setattr(search, "run_job", run_job)
    monkeypatch.setattr(search, "load_receipt", receipt)
    monkeypatch.setattr(search, "sandbox_call", sandbox_call)
    if collision is None:
        from bridge_evidence.contest import rsi_search_comparison
        from bridge_evidence.contest.rsi import StrategyBundle

        bundle_path = tmp_path / "optimizer.json"
        bundle_path.write_text(json.dumps(StrategyBundle.baseline(config).as_dict()))
        search.sys.argv.extend([
            "--optimizer-parent-bundle", str(bundle_path),
            "--optimizer-child-bundle", str(bundle_path),
        ])
        audited = []

        def audited_optimizer_trial(directory, *, source_root):
            audited.append(directory)
            return {"audit_scope": "optimizer-trial-only", "trusted-test-marker": True}

        def compare(parent, child, actor, anchor, root, evaluate, register, **kwargs):
            trial = evaluate(actor, "optimizer-boundary-test", cap=14400)
            assert trial.get("trusted-test-marker") is True
            return {"optimizer_verified": True}

        monkeypatch.setattr(
            search, "audited_optimizer_trial", audited_optimizer_trial, raising=False
        )
        monkeypatch.setattr(rsi_search_comparison, "compare_optimizers", compare)
    search.main()
    selection = json.loads((output / "selection.json").read_text())
    if collision:
        assert selection["candidate_verified"] is False
        assert selection["selected"] == "baseline"
        assert not (output / "release.json").exists()
        assert not (output / "frozen-rsi-bundle.json").exists()
    else:
        assert selection["candidate_verified"] is True
        assert selection["optimizer_verified"] is True
        assert audited == [output / "optimizer-boundary-test"]
        assert (output / "release.json").exists()
    # Evaluation and recovery must retain the exact caller-supplied baseline.
    for name in ("initial-bundle.json", "anchor.recovery-bundle.json"):
        assert json.loads((output / name).read_text())["config"] == config
