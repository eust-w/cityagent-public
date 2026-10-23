"""Campaign validates the actual search workflow before spending cloud budget."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def campaign_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "rsi_campaign_cli", ROOT / "scripts/run_rsi_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_cli(monkeypatch, tmp_path, campaign, config=None):
    root = tmp_path / "campaign"
    root.mkdir()
    resolve = Path.resolve
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda path, *args, **kwargs: (
            Path("/workspace/campaign")
            if path == root
            else resolve(path, *args, **kwargs)
        ),
    )
    command = [
        "run_rsi_campaign.py",
        "--root", str(root),
        "--manifest", str(tmp_path / "reports.json"),
        "--model-manifest", str(tmp_path / "model.json"),
        "--sdk-acceptance", str(tmp_path / "sdk.json"),
        "--hardware", "synthetic-cli-boundary-test",
    ]
    if config is not None:
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(config))
        command += ["--baseline-config", str(path)]
    monkeypatch.setattr(campaign.sys, "argv", command)
    return root


@pytest.mark.parametrize("config", [{"workflow": "whole"}, {"batch_size": 3}])
def test_invalid_baseline_fails_before_lock_or_cloud_work(
    monkeypatch, tmp_path, capsys, config
):
    campaign = campaign_module(monkeypatch)
    root = configure_cli(monkeypatch, tmp_path, campaign, config)
    monkeypatch.setattr(
        campaign, "CloudBudget", lambda *args: pytest.fail("budget was accessed")
    )
    monkeypatch.setattr(
        campaign, "run_job", lambda *args: pytest.fail("cloud work was started")
    )
    with pytest.raises(SystemExit) as stopped:
        campaign.main()
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert "baseline" in error
    assert "unrecognized arguments" not in error
    assert not list(root.iterdir())


@pytest.mark.parametrize("workflow", [None, "bounded-react"])
def test_campaign_forwards_a_search_compatible_baseline(
    monkeypatch, tmp_path, workflow
):
    campaign = campaign_module(monkeypatch)
    root = configure_cli(
        monkeypatch,
        tmp_path,
        campaign,
        None if workflow is None else {"workflow": workflow},
    )
    with sqlite3.connect(root / "budget.sqlite") as db:
        db.execute("CREATE TABLE jobs(id TEXT, status TEXT)")

    class Budget:
        def __init__(self, path):
            assert path == root / "budget.sqlite"

        def remaining(self):
            return 60000

    launched = []

    def execute(command, *, check):
        launched.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        (output / "selection.json").write_text('{"selected":"baseline"}')
        assert check is True

    monkeypatch.setattr(campaign, "CloudBudget", Budget)
    monkeypatch.setattr(campaign, "run_job", lambda *args: None)
    monkeypatch.setattr(campaign.subprocess, "run", execute)
    campaign.main()
    assert len(launched) == 1
    command = launched[0]
    baseline = Path(command[command.index("--baseline-config") + 1])
    assert json.loads(baseline.read_text())["workflow"] == (workflow or "chunked")
    assert json.loads((root / "rsi-campaign-status.json").read_text())["state"] == "complete"
