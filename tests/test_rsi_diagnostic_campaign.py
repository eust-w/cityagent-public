"""Diagnostic campaigns preserve the full denominator and never promote a policy."""

import importlib.util
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from bridge_evidence.contest.meta_optimization import CloudBudget
from bridge_evidence.contest.rsi import StrategyBundle
from bridge_evidence.contest.run_state import atomic_json, file_digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "diagnostic_campaign", ROOT / "scripts/run_rsi_diagnostic_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "workspace_path", lambda path: True)
    monkeypatch.setattr(module, "verify_root", lambda path: "sandbox-digest")
    monkeypatch.setattr(module, "validate_acceptance", lambda path: {})
    monkeypatch.setattr(module, "verify_parallel_sdk", lambda *args: True)
    return module


@pytest.fixture
def configured(campaign, tmp_path):
    base = tmp_path / "model"
    base.mkdir()
    atomic_json(base / "config.json", {})
    model = tmp_path / "model.json"
    atomic_json(
        model,
        {
            "base": str(base),
            "tokenizer": str(base),
            "merged": str(base),
            "served_model": "fixed-model",
        },
    )
    samples = []
    for i in range(79):
        report, gold = tmp_path / f"report-{i}.docx", tmp_path / f"gold-{i}.json"
        report.write_bytes(b"fixture document")
        gold.write_text("{}")
        samples.append(
            {
                "id": str(i),
                "report": str(report),
                "gold": str(gold),
                "report_sha256": file_digest(report),
                "gold_sha256": file_digest(gold),
            }
        )
    manifest = tmp_path / "manifest.json"
    atomic_json(
        manifest,
        {
            "schema_version": 2,
            "purpose": "development-regression",
            "metric_denominator": 79,
            "samples": samples,
        },
    )
    baseline = StrategyBundle.baseline()
    candidate = StrategyBundle.baseline({**baseline.config, "batch_size": 2})
    config, bundle = tmp_path / "baseline.json", tmp_path / "candidate.json"
    atomic_json(config, baseline.config)
    atomic_json(bundle, candidate.as_dict())
    for name in ("sdk.json", "parallel.json"):
        atomic_json(tmp_path / name, {})
    ledger = tmp_path / "budget.sqlite"
    CloudBudget(ledger)
    command = [
        "--source-root",
        str(ROOT),
        "--manifest",
        str(manifest),
        "--model-manifest",
        str(model),
        "--baseline-config",
        str(config),
        "--candidate-bundle",
        str(bundle),
        "--sandbox-root",
        str(tmp_path / "sandbox"),
        "--budget-ledger",
        str(ledger),
        "--output-dir",
        str(tmp_path / "campaign"),
        "--hardware",
        "fixture-only",
        "--sdk-acceptance",
        str(tmp_path / "sdk.json"),
        "--sdk-parallel-acceptance",
        str(tmp_path / "parallel.json"),
        "--report-workers",
        "2",
    ]
    return campaign.parser().parse_args(command)


class FakeBudget:
    def __init__(self, remaining=3600):
        self.left = remaining
        self.active = False

    def remaining(self):
        return self.left


def successful_result(command, *, failures=0):
    output = Path(command[command.index("--output-dir") + 1])
    output.mkdir()
    manifest = json.loads(Path(command[command.index("--manifest") + 1]).read_text())
    config = json.loads(
        Path(command[command.index("--candidate-config") + 1]).read_text()
    )
    mode = command[command.index("--rsi-mode") + 1]
    bundle = (
        StrategyBundle.load(command[command.index("--rsi-bundle") + 1])
        if mode != "off"
        else None
    )
    value = {
        "run_id": output.name,
        "comparison_identity": "1" * 64,
        "conditions": "1" * 64,
        "denominator": 79,
        "completed": 79,
        "config": config,
        "bundle_sha256": bundle.digest if bundle else None,
        "cache_reused": False,
        "reports": [
            {"id": case["id"], "status": "failed" if failures else "success"}
            for case in manifest["samples"]
        ],
        "failures": failures,
        "f1": [0.0, 0.0, 0.0],
    }
    atomic_json(output / "evaluation.json", value)
    return value


def test_three_fresh_sequential_arms_same_candidate_and_fixed_model(
    campaign, configured, monkeypatch
):
    budget = FakeBudget()
    monkeypatch.setattr(campaign, "CloudBudget", lambda path: budget)
    launched = []

    def execute(actual, phase, cap, command, log):
        assert actual is budget and not budget.active
        budget.active = True
        launched.append((cap, command))
        assert command[command.index("--engine") + 1] == "tongagents"
        assert command[command.index("--warmup") + 1] == "1"
        assert command[command.index("--model-manifest") + 1] == str(
            configured.model_manifest
        )
        successful_result(command, failures=79)
        budget.left -= cap + 10
        budget.active = False

    monkeypatch.setattr(campaign, "run_job", execute)
    result = campaign.run(configured)
    assert len(launched) == 3
    assert max(cap for cap, _ in launched) - min(cap for cap, _ in launched) <= 1
    assert len({c[c.index("--output-dir") + 1] for _, c in launched}) == 3
    assert [c[c.index("--rsi-mode") + 1] for _, c in launched] == [
        "off",
        "frozen",
        "frozen",
    ]
    assert (
        launched[1][1][launched[1][1].index("--rsi-bundle") + 1]
        == launched[2][1][launched[2][1].index("--rsi-bundle") + 1]
    )
    assert result["baseline_retained"] and not result["formal_release_evidence"]
    assert result["expected_denominator"] == 79
    assert all(a["evaluation_complete"] for a in result["arms"])
    assert not any(a.get("qualified") for a in result["arms"])


def test_hard_cap_preserves_progress_and_calls_without_inventing_scores(
    campaign, configured, monkeypatch
):
    budget = FakeBudget()
    monkeypatch.setattr(campaign, "CloudBudget", lambda path: budget)

    def execute(actual, phase, cap, command, log):
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        atomic_json(output / "progress.json", {"completed": 2, "denominator": 79})
        atomic_json(
            output / "records/one.json",
            {"status": "failed", "model_calls": [{"call_id": "a"}]},
        )
        with sqlite3.connect(output / "timing.sqlite") as db:
            db.execute("CREATE TABLE calls(id TEXT, status TEXT, seconds REAL)")
            db.executemany(
                "INSERT INTO calls VALUES(?,?,?)",
                [("a", "success", 1), ("b", "running", None)],
            )
        actual.left -= cap
        raise subprocess.TimeoutExpired(command, cap)

    monkeypatch.setattr(campaign, "run_job", execute)
    result = campaign.run(configured)
    assert len(result["arms"]) == 3
    for arm in result["arms"]:
        assert arm["status"] == "hard-cap"
        assert not arm["evaluation_complete"]
        assert arm["progress"]["denominator"] == 79
        assert arm["calls"]["total"] == 2
        assert arm["calls"]["incomplete_call_ids"] == ["b"]
        assert arm["execution_records"]["total"] == 1
        assert "f1" not in arm
    assert not result["all_evaluations_complete"]


@pytest.mark.parametrize(
    "change",
    [
        "existing-output",
        "running-job",
        "wrong-denominator",
        "invalid-parallel",
        "engine-parameters",
    ],
)
def test_invalid_inputs_do_not_mutate_outputs_or_start_budget(
    campaign, configured, monkeypatch, change
):
    if change == "existing-output":
        configured.output_dir.mkdir()
        (configured.output_dir / "keep").write_text("untouched")
    elif change == "running-job":
        CloudBudget(configured.budget_ledger).start("other-owner", 100)
    elif change == "wrong-denominator":
        value = json.loads(configured.manifest.read_text())
        value["metric_denominator"] = 78
        atomic_json(configured.manifest, value)
    elif change == "invalid-parallel":
        monkeypatch.setattr(campaign, "verify_parallel_sdk", lambda *args: False)
    elif change == "engine-parameters":
        value = json.loads(configured.baseline_config.read_text())
        value["engine_parameters"] = {"enable_prefix_caching": True}
        atomic_json(configured.baseline_config, value)
    monkeypatch.setattr(
        campaign,
        "CloudBudget",
        lambda path: pytest.fail("budget mutation before validation"),
    )
    monkeypatch.setattr(
        campaign, "run_job", lambda *args: pytest.fail("model work before validation")
    )
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        campaign.run(configured)
    assert (
        not configured.output_dir.exists()
        or (configured.output_dir / "keep").read_text() == "untouched"
    )


def test_input_change_stops_later_arms(campaign, configured, monkeypatch):
    monkeypatch.setattr(campaign, "CloudBudget", lambda path: FakeBudget())
    launched = []

    def execute(actual, phase, cap, command, log):
        launched.append(command)
        successful_result(command)
        configured.model_manifest.write_text(
            configured.model_manifest.read_text() + " "
        )

    monkeypatch.setattr(campaign, "run_job", execute)
    result = campaign.run(configured)
    assert len(launched) == 1
    assert result["status"] == "input-changed"
    assert result["baseline_retained"]


def test_partial_result_cannot_claim_full_completion(campaign, configured, monkeypatch):
    monkeypatch.setattr(campaign, "CloudBudget", lambda path: FakeBudget())

    def execute(actual, phase, cap, command, log):
        value = successful_result(command)
        value["reports"].pop()
        atomic_json(
            Path(command[command.index("--output-dir") + 1]) / "evaluation.json", value
        )

    monkeypatch.setattr(campaign, "run_job", execute)
    result = campaign.run(configured)
    assert not result["all_evaluations_complete"]
    assert all(a["status"] == "invalid-evaluation" for a in result["arms"])


def test_insufficient_budget_does_not_launch_partial_arm(
    campaign, configured, monkeypatch
):
    monkeypatch.setattr(campaign, "CloudBudget", lambda path: FakeBudget(300))
    monkeypatch.setattr(
        campaign, "run_job", lambda *args: pytest.fail("budget cannot fund arm")
    )
    result = campaign.run(configured)
    assert result["status"] == "insufficient-budget"
    assert result["arms"] == []
    assert not result["all_evaluations_complete"]


def test_changed_actual_conditions_stop_comparison(campaign, configured, monkeypatch):
    monkeypatch.setattr(campaign, "CloudBudget", lambda path: FakeBudget())
    launched = []

    def execute(actual, phase, cap, command, log):
        launched.append(command)
        value = successful_result(command)
        value["comparison_identity"] = value["conditions"] = str(len(launched)) * 64
        atomic_json(
            Path(command[command.index("--output-dir") + 1]) / "evaluation.json", value
        )

    monkeypatch.setattr(campaign, "run_job", execute)
    result = campaign.run(configured)
    assert len(launched) == 2
    assert result["status"] == "conditions-changed"
    assert not result["all_evaluations_complete"]
    assert result["baseline_retained"]


def test_all_diagnostic_arms_forward_same_explicit_visual_proof(
    campaign, configured, tmp_path, monkeypatch
):
    model = json.loads(configured.model_manifest.read_text())
    # A unit-test fixture, never a real capability grant.
    from bridge_evidence.contest.backends import ModelManifest

    proof = tmp_path / "visual-proof.json"
    atomic_json(
        proof,
        {
            "model_identity": {
                "manifest": ModelManifest(**model).as_dict(),
                "files": {"base": {"config.json": "a" * 64}},
            }
        },
    )
    configured.visual_evidence = "same-model"
    configured.visual_capability = proof
    monkeypatch.setattr(
        campaign,
        "validate_vision_capability",
        lambda path, identity: {"mode": "same-model"},
    )
    monkeypatch.setattr(campaign, "CloudBudget", lambda path: FakeBudget())
    commands = []

    def execute(actual, phase, cap, command, log):
        commands.append(command)
        successful_result(command)

    monkeypatch.setattr(campaign, "run_job", execute)
    result = campaign.run(configured)
    assert len(commands) == 3
    for command in commands:
        assert command[command.index("--visual-evidence") + 1] == "same-model"
        assert command[command.index("--visual-capability") + 1] == str(proof)
    assert result["visual_evidence"]["mode"] == "same-model"
    assert result["input_identity"]["visual_capability"] == file_digest(proof)
