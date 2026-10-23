"""The SIGKILL window requires source-valid actual raw, within ordinary retries."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def driver_module():
    spec = importlib.util.spec_from_file_location(
        "valid_raw_recovery_driver", ROOT / "scripts/benchmark_rsi_process_recovery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fixture(tmp_path, scenario):
    from bridge_evidence.contest.rsi import StrategyBundle

    driver = driver_module()
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps(StrategyBundle.baseline().as_dict()))
    args = SimpleNamespace(output_dir=tmp_path / "run", bundle=bundle,
                           mode="off", model_manifest=None, limit_seconds=30)

    def command(stage):
        return [sys.executable, str(Path(__file__).resolve()), "--test-worker",
                str(args.output_dir), str(bundle), stage, scenario]

    return driver.run_process_recovery(args, command_factory=command)


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL and flock")
@pytest.mark.parametrize("scenario", ["invalid-first", "invalid-tokens-first"])
def test_invalid_first_raw_retries_before_kill_and_replays_without_new_call(tmp_path, scenario):
    report = run_fixture(tmp_path, scenario)
    output = tmp_path / "run"
    assert report["model_calls_before"] == report["model_calls_after"] == 2
    assert json.loads((output / "fake-calls.json").read_text()) == 2
    result = json.loads((output / "result.json").read_text())
    assert [call["response_validation"] for call in result["model_calls"]] == ["failed", "success"]
    assert report["ledger"]["call_ids"] == [call["call_id"] for call in result["model_calls"]]
    assert report["reused_call_ids"] == [result["model_calls"][-1]["call_id"]]
    precondition = driver_module().audit_precondition(output)
    assert report["precondition"] == precondition
    assert precondition["valid_call_id"] == report["reused_call_ids"][0]
    assert precondition["parser_sha256"] == driver_module().parser_identity()["sha256"]
    before = json.loads((output / "precondition.checkpoint.json").read_text())
    assert before["fragments"] == []
    assert before["calls"][-1]["response_validation"] == "pending"


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL and flock")
def test_all_invalid_raw_fails_precondition_without_kill_or_resume(tmp_path):
    with pytest.raises(RuntimeError, match="recovery precondition failed"):
        run_fixture(tmp_path, "invalid-all")
    output = tmp_path / "run"
    assert json.loads((output / "fake-calls.json").read_text()) == 2
    assert not (output / "kill-ready.json").exists()
    assert not (output / "resume.identity.json").exists()
    assert not (output / "process-recovery.json").exists()
    failure = json.loads((output / "precondition-failed.json").read_text())
    assert failure["status"] == "failed"
    assert failure["reason"] == "no-valid-raw-within-inference-budget"
    assert failure["model_calls"] == 2
    assert "sigkill_sent" not in [event["event"] for event in json.loads(
        (output / "supervisor.events.json").read_text())]


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL and flock")
def test_source_valid_fragment_with_all_empty_basic_fails_final_recovery(tmp_path):
    from bridge_evidence.contest.inference_core import BASIC_FIELDS
    from bridge_evidence.contest.run_state import file_digest, json_digest

    with pytest.raises(RuntimeError, match="resumed executor failed"):
        run_fixture(tmp_path, "all-empty-basic")
    output = tmp_path / "run"
    driver = driver_module()
    before = json.loads((output / "precondition.checkpoint.json").read_text())
    precondition = json.loads((output / "precondition.json").read_text())
    assert precondition["status"] == "passed"
    assert driver.audit_precondition(output) == precondition
    assert "meaningless_report" in (output / "workers.log").read_text()
    assert not (output / "result.json").exists()
    assert not (output / "process-recovery.json").exists()
    assert not (output / "precondition-failed.json").exists()

    events = json.loads((output / "supervisor.events.json").read_text())
    assert "sigkill_sent" in [event["event"] for event in events]
    assert "old_executor_exited" in [event["event"] for event in events]
    assert any(event["event"] == "worker_launched" and event["stage"] == "resume" for event in events)
    assert json.loads((output / "fake-calls.json").read_text()) == 1
    assert driver.TimingLedger(output / "timing.sqlite").summary()["calls"] == 1
    assert json_digest(driver.ledger_calls(output / "timing.sqlite")) == precondition["ledger_calls_sha256"]

    checkpoint = next((output / "work/cache").glob("*/checkpoint.sqlite"))
    with driver.TaskJournal(checkpoint, checkpoint.parent.name) as journal:
        after = journal.load()
    assert after["issued"] == before["issued"] == 1
    assert after["attempts"] == before["attempts"]
    assert len(after["fragments"]) == 1
    assert after["calls"][0]["response_validation"] == "success"
    call_id = precondition["valid_call_id"]
    assert [call["call_id"] for call in after["calls"]] == [call_id]
    raw_path = checkpoint.parent / before["calls"][0]["raw_receipt"]["path"]
    assert file_digest(raw_path) == precondition["raw_receipts_sha256"][call_id]
    stored = json.loads(raw_path.read_text())
    assert json.loads(stored["raw"])["基本信息"] == {field: None for field in BASIC_FIELDS}


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL and flock")
def test_forged_ready_for_invalid_raw_is_reparsed_before_sigkill(tmp_path):
    with pytest.raises(ValueError):
        run_fixture(tmp_path, "forge-ready")
    output = tmp_path / "run"
    assert (output / "kill-ready.json").exists()
    assert not (output / "resume.identity.json").exists()
    assert not (output / "process-recovery.json").exists()
    assert "sigkill_sent" not in [event["event"] for event in json.loads(
        (output / "supervisor.events.json").read_text())]


@pytest.fixture
def completed_run(tmp_path):
    if os.name != "posix":
        pytest.skip("requires SIGKILL and flock")
    run_fixture(tmp_path, "valid")
    return tmp_path / "run"


@pytest.mark.parametrize("field", ["parser", "raw_sha256", "fragment_sha256", "source_sha256"])
def test_offline_audit_rejects_forged_ready_binding(completed_run, field):
    path = completed_run / "kill-ready.json"
    ready = json.loads(path.read_text())
    ready["precondition"][field] = "forged"
    path.write_text(json.dumps(ready))
    with pytest.raises(ValueError, match="forged or stale"):
        driver_module().audit_precondition(completed_run)


def test_offline_audit_rejects_changed_document(completed_run):
    (completed_run / "source.txt").write_text("a different bridge source")
    with pytest.raises(ValueError, match="parser/source/execution identity changed"):
        driver_module().audit_precondition(completed_run)


def test_offline_audit_rejects_changed_parser_dependency(completed_run, monkeypatch):
    driver = driver_module()
    original = driver.parser_identity

    def changed_parser():
        value = original()
        value["dependencies"]["src/bridge_evidence/contest/model_json.py"] = "0" * 64
        return value

    monkeypatch.setattr(driver, "parser_identity", changed_parser)
    with pytest.raises(ValueError, match="parser/source/execution identity changed"):
        driver.audit_precondition(completed_run)


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL and flock")
def test_successful_resume_cannot_pass_after_an_additional_model_call(tmp_path):
    with pytest.raises(ValueError, match="repeated a model request"):
        run_fixture(tmp_path, "extra-resume-call")
    output = tmp_path / "run"
    assert (output / "result.json").exists()  # Reconstructed output alone is insufficient.
    assert not (output / "process-recovery.json").exists()


def test_runtime_phase_events_retain_complete_preparation(tmp_path, monkeypatch):
    driver = driver_module()
    model = SimpleNamespace(tokenizer="local", served_model="fixed")
    model.validate = lambda: model
    model.identity = lambda: {"test_only": True}
    monkeypatch.setattr(driver.ModelManifest, "load", lambda _: model)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a, **kw: object())))
    monkeypatch.setattr(driver, "create_backend", lambda *a: object())
    events = []
    driver.runtime(SimpleNamespace(model_manifest=tmp_path / "manifest"),
                   event=lambda name, **values: events.append(name))
    assert events == [phase + suffix for phase in (
        "runtime_import", "manifest_validation", "model_identity", "tokenizer_load", "backend_setup",
    ) for suffix in ("_started", "_finished")]


def fixture_runtime(args):
    from test_rsi_process_recovery import fake_runtime

    tokenizer, original, identity = fake_runtime(args)

    class Backend:
        def __call__(self, ids, maximum):
            response = original(ids, maximum)
            count = json.loads((args.output_dir / "fake-calls.json").read_text())
            response["backend_request_id"] = "test-valid-raw-" + str(count)
            (args.output_dir / "fixture-valid-response.txt").write_text(response["text"])
            if args.scenario in ("invalid-all", "forge-ready") or args.scenario == "invalid-first" and count == 1:
                response["text"] = "{not valid JSON"
            if args.scenario == "invalid-tokens-first" and count == 1:
                response["output_tokens"] = 0
            if args.scenario == "all-empty-basic":
                from bridge_evidence.contest.inference_core import BASIC_FIELDS

                value = json.loads(response["text"])
                value["基本信息"] = {field: None for field in sorted(BASIC_FIELDS)}
                value["basic_sources"] = {}
                value["basic_quotes"] = {}
                response["text"] = json.dumps(value, ensure_ascii=False)
            return response

        def batch(self, requests):
            return [self(*request) for request in requests]

    return tokenizer, Backend(), identity


if __name__ == "__main__":
    assert sys.argv[1] == "--test-worker"
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    driver = driver_module()
    args = SimpleNamespace(output_dir=Path(sys.argv[2]), bundle=Path(sys.argv[3]),
                           worker_stage=sys.argv[4], scenario=sys.argv[5],
                           mode="off", sandbox_root=None)
    if args.scenario == "forge-ready":
        driver.ci.parse_fragment = lambda raw, task, snapshot, call_id: driver.trusted_parse_fragment(
            (args.output_dir / "fixture-valid-response.txt").read_text(), task, snapshot, call_id)
    driver.worker(args, runtime_factory=fixture_runtime)
    if args.scenario == "extra-resume-call" and args.worker_stage == "resume":
        # An actual accounted fake-backend call after otherwise successful replay.
        backend = lambda ids, maximum: {
            "text": "{}", "finish_reason": "stop", "output_tokens": 1,
            "backend_request_id": "unexpected-extra-successful-request",
            "backend_instance_id": "test-only",
        }
        driver.AccountedBackend(backend, driver.TimingLedger(args.output_dir / "timing.sqlite"))([1], 10)
