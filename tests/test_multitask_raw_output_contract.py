"""Evaluate production whole JSON fences without rescuing malformed reports."""

import copy
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from bridge_evidence.contest.inference_core import _parse_report
from bridge_evidence.evaluation.generation import METRIC_VERSION, score_report

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("multitask_raw_contract", ROOT / "scripts/evaluate_multitask.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
GOLD = {"基本信息": {"桥梁名称": "测试桥"}, "维修建议": [{"建议内容": "修补裂缝"}],
        "病害明细": [{"病害类型": "裂缝"}]}
RAW = json.dumps(GOLD, ensure_ascii=False)


def record(raw=RAW, final=None):
    return {"raw_output": raw, "final_data": copy.deepcopy(GOLD if final is None else final),
            "report_wall_seconds": 1.5}


@pytest.mark.parametrize("raw", [RAW, "```json\n" + RAW + "\n```", "```JSON\n" + RAW + "\n```",
                                 RAW[:-1] + ",}"])
def test_whole_uses_exact_production_parse_contract(raw):
    parsed = _parse_report(raw)
    scored = evaluator.score_execution_record(GOLD, record(raw), workflow="whole")
    assert scored["status"] == "success" and scored["error"] is None
    assert scored["raw_score"] == score_report(GOLD, parsed)
    assert scored["final_score"] == score_report(GOLD, GOLD)
    assert scored["processing_wall_seconds"] == 1.5


@pytest.mark.parametrize("raw", ["", RAW[:-1], "```json\n" + RAW, "说明:" + RAW, RAW + "unfinished",
                                 "[" + RAW + "]", None, {},
                                 '{"基本信息":{"桥梁名称":"桥","桥梁名称":"另桥"},"维修建议":[],"病害明细":[]}',
                                 '{"基本信息":{"总体评分":NaN},"维修建议":[],"病害明细":[]}'])
def test_whole_invalid_output_is_not_salvaged_or_replaced_by_valid_final(raw):
    scored = evaluator.score_execution_record(GOLD, record(raw), workflow="whole")
    assert scored["status"] == "failed"
    assert scored["error"].startswith("evaluation_record_invalid:")
    assert scored["raw_score"] == scored["final_score"] == score_report(GOLD, None)
    assert scored["final_score"]["overall"]["fn"] == 3


@pytest.mark.parametrize("workflow", ["chunked", "bounded-react"])
def test_chunked_retains_plain_merged_json_contract(workflow):
    assert evaluator.score_execution_record(GOLD, record(), workflow=workflow)["status"] == "success"
    fenced = evaluator.score_execution_record(GOLD, record("```json\n" + RAW + "\n```"), workflow=workflow)
    assert fenced["status"] == "failed"
    # Merged chunk data need not satisfy whole-only meaningful-basic admission.
    merged = {"基本信息": {}, "维修建议": [], "病害明细": []}
    result = evaluator.score_execution_record(GOLD, record(json.dumps(merged), merged), workflow=workflow)
    assert result["status"] == "success"
    assert result["raw_score"] == score_report(GOLD, merged)


@pytest.mark.parametrize("change", ["missing_raw", "missing_final", "invalid_final", "null_final", "null_raw"])
def test_missing_or_invalid_output_preserves_all_false_negatives(change):
    value = record()
    if change == "missing_raw":
        del value["raw_output"]
    elif change == "missing_final":
        del value["final_data"]
    elif change == "invalid_final":
        value["final_data"]["维修建议"] = "not rows"
    elif change == "null_final":
        value["final_data"] = None
    else:
        value["raw_output"] = "null"
    result = evaluator.score_execution_record(GOLD, value, workflow="chunked")
    assert result["status"] == "failed"
    assert result["raw_score"] == result["final_score"] == score_report(GOLD, None)


def test_invalid_gold_is_a_dataset_error_not_a_failed_prediction():
    with pytest.raises(ValueError, match="all three sections"):
        evaluator.score_execution_record({}, record(), workflow="whole")


def test_full_evaluation_keeps_first_failure_and_continues_after_bad_output(monkeypatch, tmp_path):
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(RAW)
    cases = [{"id": str(i), "report": str(tmp_path / f"{i}.docx"), "gold": str(gold_path)} for i in range(4)]
    config = tmp_path / "config.json"
    config.write_text('{"workflow":"whole"}')
    monkeypatch.setattr(evaluator, "load_cases", lambda _: ({"samples": cases}, cases))
    model = SimpleNamespace(base=str(tmp_path), tokenizer=str(tmp_path), served_model="fixture",
                            identity=lambda: {"files": {"base": {}}})
    model.validate = lambda: model
    monkeypatch.setattr(evaluator.ModelManifest, "load", lambda _: model)
    monkeypatch.setattr(evaluator, "build_run_identity", lambda *a, **kw: {"code_files": {}})
    monkeypatch.setattr(evaluator, "bind_work_directory", lambda *a, **kw: "fixture-run")
    monkeypatch.setattr(evaluator, "comparison_identity", lambda *a, **kw: "fixture-conditions")
    monkeypatch.setattr(evaluator, "APIBackend", lambda *a, **kw: SimpleNamespace(probe=lambda: None))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a, **kw: SimpleNamespace(eos_token_id=0))))
    entry = ModuleType("contest_runtime")
    entry.validate_visual_options = lambda args: None
    entry.configure_visual_runtime = lambda args, identity: {}
    entry.build_parser = lambda: SimpleNamespace(parse_args=lambda argv: SimpleNamespace(result_dir=tmp_path / "out" / "result"))
    entry.prepare_document = lambda args, path, tokenizer: path
    visited = []

    def process(args, path, *a, **kw):
        index = int(path.stem)
        visited.append(index)
        if index == 0:
            failure = RuntimeError("context exceeded")
            failure.record = {"report_wall_seconds": 0.1}
            raise failure
        if index == 1:
            return record("```json\n" + RAW + "\n```")
        if index == 2:
            return record(RAW[:-1])
        return record()

    entry.process_document = process

    class Loader:
        def create_module(self, spec):
            return entry

        def exec_module(self, module):
            pass

    original_spec = importlib.util.spec_from_file_location
    monkeypatch.setattr(evaluator.importlib.util, "spec_from_file_location", lambda name, path: (
        importlib.machinery.ModuleSpec(name, Loader()) if name == "contest_runtime" else original_spec(name, path)))
    output = tmp_path / "out"
    output.mkdir()
    args = SimpleNamespace(manifest=tmp_path / "manifest.json", model_manifest=tmp_path / "model.json",
        candidate_config=config, output_dir=output, rsi_bundle=None, rsi_mode="off", rsi_sandbox_root=None,
        report_workers=1, engine="direct", sdk_acceptance=None, hardware="fixture", warmup=0,
        backend="vllm-api", url="http://127.0.0.1:8011/v1", visual_evidence="off", visual_capability=None,
        visual_capability_binding=None, visual_model_identity_sha256=None)
    result = evaluator._evaluate_locked(args)
    assert visited == [0, 1, 2, 3]
    assert result["completed"] == result["denominator"] == result["end_to_end_latency_count"] == 4
    assert result["metric_version"] == METRIC_VERSION
    assert result["failures"] == 2
    assert [r["status"] for r in result["reports"]] == ["failed", "success", "failed", "success"]
    assert result["raw_metrics"]["reports_failed"] == result["final_metrics"]["reports_failed"] == 2
    assert result["final_metrics"]["overall"]["gold_count"] == 12
    assert result["final_metrics"]["overall"]["fn"] == 6
    assert result["final_metrics"]["overall"]["recall"] == 0.5
    assert json.loads((output / "evaluation.json").read_text()) == result
