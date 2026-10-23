"""Report latency includes preparation/queue failures without changing RSI proofs."""

import importlib.machinery
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from bridge_evidence.contest.rsi_pipeline import pipeline

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_multitask.py"
spec = importlib.util.spec_from_file_location("multitask_latency_test", SCRIPT)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def test_latency_includes_delayed_preparation_processing_and_report_queue():
    timings = {}

    def prepare(item):
        time.sleep(0.025)
        return item

    def process(index, prepared):
        time.sleep(0.04 if index == 0 else 0.025)
        return prepared

    results = list(pipeline([0, 1, 2], prepare, process, workers=1,
                            timing_observer=lambda i, t: timings.update({i: t})))
    assert results == [(0, 0), (1, 1), (2, 2)]
    assert set(timings) == {0, 1, 2}
    for timing in timings.values():
        assert timing["preparation_wall_seconds"] >= 0.025
        assert timing["pipeline_processing_wall_seconds"] >= 0.025
        assert timing["end_to_end_wall_seconds"] >= 0.05
        assert timing["status"] == "success"
        assert timing["end_to_end_wall_seconds"] == pytest.approx(sum(
            timing[key] for key in ("preparation_queue_wall_seconds", "preparation_wall_seconds",
                                   "prepared_to_processing_wall_seconds", "pipeline_processing_wall_seconds")))
    assert timings[1]["prepared_to_processing_wall_seconds"] >= 0.03


def test_out_of_order_terminal_timings_follow_report_identity():
    release_first = threading.Event()
    timings = {}

    def prepare(item):
        if item == 0:
            assert release_first.wait(3)
        return item

    stream = pipeline([0, 1], prepare, lambda i, p: p, workers=2,
                      timing_observer=lambda i, t: timings.update({i: t}))
    try:
        assert next(stream) == (1, 1)
        assert set(timings) == {1}
        release_first.set()
        assert list(stream) == [(0, 0)]
    finally:
        release_first.set()
        stream.close()
    assert set(timings) == {0, 1}
    assert timings[1]["terminal_monotonic_seconds"] < timings[0]["terminal_monotonic_seconds"]


def test_preparation_and_processing_failures_keep_finite_timings_and_denominator():
    timings, processed = {}, []

    def prepare(item):
        if item == 0:
            time.sleep(0.01)
            raise ValueError("conversion failed")
        return item

    def process(index, value):
        processed.append(index)
        if index == 1:
            raise RuntimeError("render failed")
        return value

    results = dict(pipeline([0, 1, 2], prepare, process, workers=2,
                            timing_observer=lambda i, t: timings.update({i: t})))
    assert isinstance(results[0], ValueError)
    assert isinstance(results[1], RuntimeError)
    assert results[2] == 2
    assert sorted(processed) == [1, 2]
    assert timings[0]["terminal_stage"] == "prepare"
    assert timings[0]["pipeline_processing_wall_seconds"] is None
    assert timings[0]["end_to_end_wall_seconds"] >= 0.01
    assert timings[1]["terminal_stage"] == "process"
    assert timings[0]["status"] == timings[1]["status"] == "failed"
    summary = evaluator.end_to_end_latency_summary(list(timings.values()), denominator=3)
    assert summary["end_to_end_latency_count"] == summary["end_to_end_latency_denominator"] == 3
    assert summary["end_to_end_latency_seconds"]["p95"] > 0


@pytest.mark.parametrize("values,denominator", [([1], 2), ([None], 1), ([float("nan")], 1),
                                               ([float("inf")], 1), ([-1], 1), ([True], 1)])
def test_latency_summary_rejects_missing_and_invalid_observations(values, denominator):
    with pytest.raises(ValueError, match="one finite timing per report"):
        evaluator.end_to_end_latency_summary(
            [{"end_to_end_wall_seconds": value} for value in values], denominator=denominator)


def test_observer_is_optional_and_failure_return_contract_is_unchanged():
    def prepare(item):
        if item:
            raise ValueError("failure")
        return item

    results = dict(pipeline([0, 1], prepare, lambda i, p: p))
    assert results[0] == 0
    assert isinstance(results[1], ValueError)


@pytest.mark.parametrize("all_fail", [False, True])
def test_evaluator_persists_every_report_timing_including_preparation_failures(
    monkeypatch, tmp_path, all_fail
):
    gold = {"基本信息": {"桥梁名称": "测试桥"}, "维修建议": [], "病害明细": []}
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold))
    cases = [{"id": str(i), "report": str(tmp_path / f"{i}.docx"), "gold": str(gold_path)}
             for i in range(2)]
    config_path = tmp_path / "candidate.json"
    config_path.write_text('{"workflow":"chunked"}')
    monkeypatch.setattr(evaluator, "load_cases", lambda path: ({"samples": cases}, cases))
    model = SimpleNamespace(base=str(tmp_path), tokenizer=str(tmp_path), served_model="test",
                            identity=lambda: {"files": {"base": {}}})
    model.validate = lambda: model
    monkeypatch.setattr(evaluator.ModelManifest, "load", lambda path: model)
    monkeypatch.setattr(evaluator, "build_run_identity", lambda *a, **kw: {"code_files": {}})
    monkeypatch.setattr(evaluator, "bind_work_directory", lambda *a, **kw: "test-fingerprint")
    monkeypatch.setattr(evaluator, "comparison_identity", lambda *a, **kw: "test-identity")
    monkeypatch.setattr(evaluator, "APIBackend", lambda *a, **kw: SimpleNamespace(probe=lambda: None))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a, **kw: SimpleNamespace(eos_token_id=0))))
    entry = ModuleType("contest_runtime")
    entry.validate_visual_options = lambda args: None
    entry.configure_visual_runtime = lambda args, identity: {}
    entry.build_parser = lambda: SimpleNamespace(parse_args=lambda argv: SimpleNamespace(
        result_dir=tmp_path / "output" / "result"))

    def prepare(runtime, path, tokenizer):
        time.sleep(0.01)
        if path.stem == "0" or all_fail:
            raise ValueError("conversion failed")
        return {"prepared": True}

    def process(*a, **kw):
        time.sleep(0.01)
        return {"final_data": gold, "raw_output": json.dumps(gold), "report_wall_seconds": 0.005}

    entry.prepare_document, entry.process_document = prepare, process

    class Loader:
        def create_module(self, spec):
            return entry

        def exec_module(self, module):
            pass

    original_spec = importlib.util.spec_from_file_location
    monkeypatch.setattr(evaluator.importlib.util, "spec_from_file_location", lambda name, path: (
        importlib.machinery.ModuleSpec(name, Loader()) if name == "contest_runtime"
        else original_spec(name, path)))
    output = tmp_path / "output"
    output.mkdir()
    args = SimpleNamespace(
        manifest=tmp_path / "manifest.json", model_manifest=tmp_path / "model.json",
        candidate_config=config_path, output_dir=output, rsi_bundle=None, rsi_mode="off",
        rsi_sandbox_root=None, report_workers=2, engine="direct", sdk_acceptance=None,
        hardware="test", warmup=0, backend="vllm-api", url="http://127.0.0.1:8011/v1",
        visual_evidence="off", visual_capability=None, visual_capability_binding=None,
        visual_model_identity_sha256=None)
    result = evaluator._evaluate_locked(args)
    assert result["completed"] == result["denominator"] == 2
    assert result["failures"] == (2 if all_fail else 1)
    assert result["end_to_end_latency_count"] == result["end_to_end_latency_denominator"] == 2
    assert result["end_to_end_latency_seconds"]["p50"] >= 0.01
    expected_processing = None if all_fail else 0.005
    assert result["latency"]["p50"] == expected_processing
    assert "excludes_preparation" in result["latency_scope"]
    persisted = json.loads((output / "evaluation.json").read_text())
    assert persisted == result
    assert len(list((output / "pipeline-timing").glob("*.json"))) == 2
    for detail in result["reports"]:
        receipt = json.loads((output / "pipeline-timing" / f"{int(detail['id']):06d}.json").read_text())
        assert receipt["id"] == detail["id"]
        assert receipt["run_fingerprint"] == "test-fingerprint"
        assert detail["end_to_end_wall_seconds"] == receipt["end_to_end_wall_seconds"]
        assert detail["processing_wall_seconds"] == detail["wall_seconds"]
    assert (output / "result" / "infer_time.json").is_file()
