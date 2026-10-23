"""Legacy parameter search must never bypass the stricter RSI release gate."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "offline_meta", ROOT / "scripts/run_meta_optimization.py"
)
meta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta)


def test_repeated_search_winner_remains_provisional(tmp_path):
    baseline = {"config": {"batch_size": 1}, "f1": [0.9, 0.9, 0.9], "wall_seconds": 100}
    winner = {
        "config": {"batch_size": 2},
        "f1": [0.95, 0.95, 0.95],
        "wall_seconds": 60,
        "repeated": True,
        "verified": True,
    }
    meta.write_search_selection(
        tmp_path,
        baseline,
        winner,
        cloud_remaining_seconds=1000,
        model_manifest_sha256="model",
        evaluation_manifest_sha256="data",
    )
    assert (
        json.loads((tmp_path / "frozen-config.json").read_text()) == baseline["config"]
    )
    assert (
        json.loads((tmp_path / "provisional-config.json").read_text())
        == winner["config"]
    )
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["selected"] == "baseline"
    assert selection["search_winner"] == "candidate"
    assert selection["release_qualified"] is False
    assert {
        "latency_p50_p95",
        "fault_recovery_start_finish",
        "independent_execution_receipts",
    } <= set(selection["missing_release_gates"])
    assert selection["search_winner_evidence"] == winner


def test_no_search_winner_exports_only_baseline(tmp_path):
    baseline = {"config": {"batch_size": 1}}
    meta.write_search_selection(
        tmp_path,
        baseline,
        baseline,
        cloud_remaining_seconds=1,
        model_manifest_sha256="model",
        evaluation_manifest_sha256="data",
    )
    assert not (tmp_path / "provisional-config.json").exists()
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["search_winner"] == "baseline"
    assert selection["selected"] == "baseline"
    assert selection["release_qualified"] is False


def test_legacy_cli_retest_does_not_promote_winner(monkeypatch, tmp_path):
    from bridge_evidence.contest.candidate import validate_config

    output = tmp_path / "search"
    budget_path = tmp_path / "budget.sqlite"
    baseline = {
        "config": validate_config({}),
        "conditions": "fixed-conditions",
        "f1": [0.9, 0.9, 0.9],
        "wall_seconds": 100,
        "failures": 0,
        "denominator": 2,
        "protocol": "v2",
        "metric_version": "fixed",
    }
    candidate = {
        **baseline,
        "config": validate_config({"batch_size": 2}),
        "wall_seconds": 60,
    }
    for name, value in [
        ("baseline", baseline),
        ("initial", candidate),
        ("model", {}),
        ("manifest", {}),
    ]:
        (tmp_path / f"{name}.json").write_text(json.dumps(value))

    class Budget:
        def __init__(self, path):
            pass

        def remaining(self):
            return 80000

        def start(self, *args):
            return "job"

        def finish(self, *args):
            pass

    class Process:
        def __init__(self, command, **kwargs):
            self.args = command
            trial = Path(command[command.index("--output-dir") + 1])
            trial.mkdir()
            (trial / "evaluation.json").write_text(json.dumps(candidate))

        def wait(self, timeout):
            return 0

    # Exercise the unchanged CLI locally without launching any real cloud trial.
    resolve = Path.resolve
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda path, *a, **kw: (
            Path("/workspace") / path.name
            if path in (output, budget_path)
            else resolve(path, *a, **kw)
        ),
    )
    monkeypatch.setattr(meta, "CloudBudget", Budget)
    monkeypatch.setattr(meta, "candidate_configs", lambda *args: [])
    monkeypatch.setattr(meta, "diagnose_baseline", lambda baseline: {})
    monkeypatch.setattr(meta.subprocess, "Popen", Process)
    monkeypatch.setattr(
        meta.sys,
        "argv",
        [
            "run_meta_optimization.py",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--model-manifest",
            str(tmp_path / "model.json"),
            "--baseline",
            str(tmp_path / "baseline.json"),
            "--initial-candidate",
            str(tmp_path / "initial.json"),
            "--output-dir",
            str(output),
            "--budget-ledger",
            str(budget_path),
            "--hardware",
            "test-only",
        ],
    )
    meta.main()
    assert json.loads((output / "frozen-config.json").read_text()) == baseline["config"]
    assert (
        json.loads((output / "provisional-config.json").read_text())
        == candidate["config"]
    )
    assert (
        json.loads((output / "selection.json").read_text())["release_qualified"]
        is False
    )
    history = json.loads((output / "history.json").read_text())
    assert history[-1]["search_stage"] == "independent-search-retest"
