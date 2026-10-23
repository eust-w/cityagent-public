"""Synthetic prompt/receipt tests; no real model or SDK acceptance claims."""

import hashlib
import json

import pytest
from docx import Document

import run
from bridge_evidence.contest import inference_core
from bridge_evidence.contest.candidate import validate_config
from test_inference_core import FakeBackend, FakeTokenizer, response


@pytest.fixture
def report_runtime(tmp_path):
    source_dir, result_dir, work_dir = [tmp_path / name for name in ("source", "result", "work")]
    for directory in (source_dir, result_dir, work_dir):
        directory.mkdir()
    source = source_dir / "synthetic.docx"
    document = Document()
    document.add_paragraph("合成测试桥定期检测报告，仅用于测试控制流程。")
    document.save(source)
    args = run.build_parser().parse_args([
        "--input-dir", str(source_dir), "--result-dir", str(result_dir),
        "--work-dir", str(work_dir), "--workflow", "whole", "--engine", "direct",
        "--max-model-len", "8192", "--max-new-tokens", "512", "--max-attempts", "2",
    ])
    return args, source


def test_flat_candidate_changes_only_prompt_configuration():
    original = validate_config({"workflow": "whole"})
    flat = validate_config({"workflow": "whole", "prompt": "whole-flat-v1"})
    assert {key for key in original if original[key] != flat[key]} == {"prompt"}


@pytest.mark.parametrize("workflow", ["chunked", "bounded-react"])
def test_fragment_workflow_rejects_whole_flat_prompt(workflow):
    with pytest.raises(ValueError, match="whole"):
        validate_config({"workflow": workflow, "prompt": "whole-flat-v1"})


@pytest.mark.parametrize("prompt", ["source-fragment-v1", "source-fragment-concise-v1"])
def test_legacy_whole_instruction_remains_byte_identical(prompt):
    assert inference_core.whole_instruction(prompt) == inference_core.INSTRUCTION


@pytest.mark.parametrize("prompt", [None, "source-fragment-v1", "whole-flat-v1"])
def test_whole_runtime_selects_opt_in_prompt_only(report_runtime, prompt):
    args, source = report_runtime
    if prompt is not None:
        args.candidate = validate_config({"workflow": "whole", "prompt": prompt})
    tokenizer = FakeTokenizer()
    backend = FakeBackend(response())
    record = run.process_document(args, source, tokenizer, backend, fingerprint="a" * 64)
    expected = (inference_core.WHOLE_FLAT_INSTRUCTION if prompt == "whole-flat-v1"
                else inference_core.INSTRUCTION)
    assert tokenizer.templates[0][0][0]["content"] == expected
    assert record["status"] == "success"
    assert len(backend.calls) == 1 and backend.calls[0][1] == args.max_new_tokens


def test_flat_template_overhead_is_charged_before_any_model_call():
    tokenizer = FakeTokenizer()
    old_ids = inference_core.build_prompt(tokenizer, "合成报告", max_model_len=8192, max_new_tokens=64)
    backend = FakeBackend(response())
    with pytest.raises(inference_core.ContextBudgetError):
        inference_core.infer_document(
            "合成报告", tokenizer, backend, max_model_len=len(old_ids) + 64, max_new_tokens=64,
            instruction=inference_core.whole_instruction("whole-flat-v1"),
        )
    assert backend.calls == []


@pytest.mark.parametrize("prompt", ["source-fragment-v1", "whole-flat-v1"])
def test_failed_nested_output_is_preserved_without_relaxing_parser_or_retry(report_runtime, prompt):
    args, source = report_runtime
    args.candidate = validate_config({"workflow": "whole", "prompt": prompt})
    raw = json.dumps({"基本信息": {"桥梁名称": "合成桥", "技术状况": {"评分": 80}},
                      "维修建议": [], "病害明细": []}, ensure_ascii=False)
    backend = FakeBackend(response(raw=raw), response(raw=raw))
    with pytest.raises(run.DocumentProcessingError) as caught:
        run.process_document(args, source, FakeTokenizer(), backend, fingerprint="a" * 64)
    record = caught.value.record
    assert record["status"] == "failed" and record["submission_ready"] is False
    assert all(call["errors"] == ["invalid_basic_shape"] for call in record["model_calls"])
    assert len(backend.calls) == 2 and backend.calls[0] == backend.calls[1]
    assert record["raw_output"] == raw
    assert record["raw_output_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert record["raw_output_call_id"] == record["model_calls"][-1]["call_id"]
    assert not list(args.result_dir.glob("*.docx"))
    saved = json.loads(next((args.work_dir / "records").glob("*.json")).read_text())
    assert saved == record


def test_failure_receipt_binds_last_received_text_when_later_call_raises(report_runtime):
    args, source = report_runtime
    backend = FakeBackend(response(raw="invalid JSON"), TimeoutError("synthetic timeout"))
    with pytest.raises(run.DocumentProcessingError) as caught:
        run.process_document(args, source, FakeTokenizer(), backend, fingerprint="a" * 64)
    record = caught.value.record
    assert record["raw_output"] == "invalid JSON"
    assert record["raw_output_call_id"] == record["model_calls"][0]["call_id"]
    assert record["model_calls"][1]["output_sha256"] is None


def test_invalid_unicode_failure_can_still_persist_diagnostic_receipt(report_runtime):
    args, source = report_runtime
    backend = FakeBackend(response(raw="\ud800"), response(raw="\ud800"))
    with pytest.raises(run.DocumentProcessingError) as caught:
        run.process_document(args, source, FakeTokenizer(), backend, fingerprint="a" * 64)
    record = caught.value.record
    assert record["status"] == "failed" and "raw_output" not in record
    assert json.loads(record["raw_output_json"]) == "\ud800"
    assert record["raw_output_json_sha256"] == hashlib.sha256(record["raw_output_json"].encode()).hexdigest()
    assert not list(args.result_dir.glob("*.docx"))


def test_backend_only_failure_does_not_invent_an_empty_model_response(report_runtime):
    args, source = report_runtime
    backend = FakeBackend(TimeoutError("first"), TimeoutError("second"))
    with pytest.raises(run.DocumentProcessingError) as caught:
        run.process_document(args, source, FakeTokenizer(), backend, fingerprint="a" * 64)
    record = caught.value.record
    assert record["status"] == "failed" and len(record["model_calls"]) == 2
    assert "raw_output" not in record and "raw_output_sha256" not in record
    assert not list(args.result_dir.glob("*.docx"))


def test_flat_prompt_overflow_keeps_zero_calls_and_failure_receipt(report_runtime):
    args, source = report_runtime
    args.candidate = validate_config({"workflow": "whole", "prompt": "whole-flat-v1"})
    args.max_model_len = args.max_new_tokens + 100
    backend = FakeBackend(response())
    with pytest.raises(run.DocumentProcessingError) as caught:
        run.process_document(args, source, FakeTokenizer(), backend, fingerprint="a" * 64)
    record = caught.value.record
    assert record["status"] == "failed" and record["model_calls"] == []
    assert backend.calls == [] and "raw_output" not in record
    assert not list(args.result_dir.glob("*.docx"))
