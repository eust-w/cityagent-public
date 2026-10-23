"""Synthetic DOCX/receipt tests; no real corpus, model, SDK or GPU access."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from docx import Document
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/audit_generated_postprocessing.py"
SPEC = importlib.util.spec_from_file_location("postprocessing_audit_tests", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "training"
    year = root / "2018年"
    year.mkdir(parents=True)
    document = year / "Synthetic report.docx"
    doc = Document()
    doc.add_paragraph("Synthetic source for postprocessing audit")
    doc.save(document)
    defect = {"序号": "17", "病害部位": "P1", "病害描述": "裂缝", "病害类型": "裂缝",
              "是否新增": "", "上一次定检状态": "", "发展程度": ""}
    prediction = {"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [], "病害明细": [defect, deepcopy(defect)]}
    content = {"instruction": "extract", "input": "Synthetic source", "output": json.dumps(prediction, ensure_ascii=False)}
    meta = {"year": "2018年", "report": document.name, "report_sha256": audit.bench.file_digest(document)}
    sft = tmp_path / "sft.jsonl"
    save(sft, {**content, "meta": meta})
    row = audit.bench.load_rows(sft)[0]
    generation_dir = tmp_path / "results"
    job = generation_dir / "job"
    job.mkdir(parents=True)
    manifest = {"sft_sha256": audit.bench.file_digest(sft), "selected_sample_sha256": [row["sample_sha256"]],
                "parameters": {"partition": "diagnostic", "max_new_tokens": 32768}}
    save(job / "manifest.json", manifest)
    raw = json.dumps(prediction, ensure_ascii=False)
    record = {"sample_sha256": row["sample_sha256"], "line": 1, "year": "2018", "partition": "diagnostic",
              "prompt_content_sha256": row["prompt_sha256"], "prompt_ids_sha256": "b" * 64,
              "prompt_tokens": 30, "output_tokens": 50, "status": "success", "finish_reason": "stop",
              "raw_output": raw, "prediction": prediction,
              "model_calls": [{"status": "success", "errors": [], "prompt_sha256": "b" * 64,
                               "prompt_tokens": 30, "output_tokens": 50, "finish_reason": "stop",
                               "output_sha256": hashlib.sha256(raw.encode()).hexdigest()}]}
    path = job / (row["sample_sha256"] + ".json")
    save(path, record)
    mapping = tmp_path / "documents.json"
    save(mapping, [{"sample_sha256": row["sample_sha256"], "path": str(document),
                    "source_sha256": meta["report_sha256"]}])
    args = SimpleNamespace(sft=sft, generation_dir=generation_dir, document_root=None,
                           document_manifest=mapping, sample_sha256=None)
    return SimpleNamespace(args=args, row=row, meta=meta, document=document, path=path,
                           record=record, manifest=manifest, root=root, tmp=tmp_path)


def test_real_pure_pipeline_preserves_duplicate_rows_and_records_legacy_inventions(fixture):
    result = audit.audit(fixture.args)
    assert result["status"] == "completed"
    record = result["records"][0]
    candidate = record["strategies"]["model_preserving"]["report"]
    assert len(candidate["病害明细"]) == 2
    assert [row["是否新增"] for row in candidate["病害明细"]] == ["", ""]
    assert record["candidate_changes"]["rows"]["病害明细"]["removed"] == []
    assert record["candidate_changes"]["rows"]["病害明细"]["added"] == []
    assert "90.00" in record["strategies"]["legacy"]["report"]["基本信息"]["总体结论"]
    assert "总体结论" not in candidate["基本信息"]
    assert record["prediction_sha256"] == record["strategies"]["raw_model"]["report_sha256"]
    assert record["source_sha256"] == audit.bench.file_digest(fixture.document)
    assert result["metrics"]["raw_model"]["sections"]["defects"]["tp"] == 2
    assert json.loads(fixture.path.read_text()) == fixture.record


@pytest.mark.parametrize("change", ["manifest_sft", "record_status", "eos", "output_tokens", "raw_hash", "stored_prediction"])
def test_bad_receipt_blocks_summary_instead_of_being_dropped(fixture, change):
    record = deepcopy(fixture.record)
    if change == "manifest_sft":
        manifest = deepcopy(fixture.manifest)
        manifest["sft_sha256"] = "c" * 64
        save(fixture.path.parent / "manifest.json", manifest)
    elif change == "record_status":
        record["status"] = "failed"
    elif change == "eos":
        record["model_calls"][0]["finish_reason"] = "length"
    elif change == "output_tokens":
        record["model_calls"][0]["output_tokens"] += 1
    elif change == "raw_hash":
        record["raw_output"] += " "
    else:
        record["prediction"]["基本信息"]["桥梁名称"] = "Forged after generation"
    save(fixture.path, record)
    result = audit.audit(fixture.args)
    assert result["status"] == "blocked"
    assert len(result["errors"]) == 1
    assert "metrics" not in result


def test_changed_training_report_is_rejected_before_legacy_extraction(fixture, monkeypatch):
    fixture.document.write_bytes(b"different source bytes")
    monkeypatch.setattr(audit.legacy, "extract_table_facts", lambda _: pytest.fail("Must reject before source extraction"))
    result = audit.audit(fixture.args)
    assert result["status"] == "blocked"
    assert "report SHA-256 mismatch" in result["errors"][0]["error"]


def test_duplicate_generation_sample_blocks_aggregate_even_when_first_succeeds(fixture):
    duplicate = fixture.args.generation_dir / "duplicate"
    duplicate.mkdir()
    save(duplicate / "manifest.json", fixture.manifest)
    save(duplicate / fixture.path.name, fixture.record)
    result = audit.audit(fixture.args)
    assert len(result["records"]) == 1
    assert len(result["errors"]) == 1
    assert "metrics" not in result


def test_explicit_subset_cannot_silently_ignore_a_missing_requested_result(fixture):
    fixture.args.sample_sha256 = ["f" * 64]
    with pytest.raises(ValueError, match="result is missing"):
        audit.audit(fixture.args)


def test_exact_root_path_rejects_traversal_and_does_not_search_alternate_names(fixture):
    path, sha = audit.source_path(fixture.row["sample_sha256"], fixture.meta, document_root=fixture.root)
    assert path == fixture.document
    assert sha == fixture.meta["report_sha256"]
    bad = {**fixture.meta, "report": "../Synthetic report.docx"}
    with pytest.raises(ValueError, match="Unsafe"):
        audit.source_path(fixture.row["sample_sha256"], bad, document_root=fixture.root)
    with pytest.raises(FileNotFoundError):
        audit.source_path(fixture.row["sample_sha256"], {**fixture.meta, "report": "Different.docx"},
                          document_root=fixture.root)


def test_multiset_diff_exposes_one_lost_duplicate(fixture):
    before = fixture.record["prediction"]
    after = deepcopy(before)
    after["病害明细"].pop()
    changes = audit.changes(before, after)["rows"]["病害明细"]
    assert changes["removed"][0]["count"] == 1
    assert changes["before_count"] == 2 and changes["after_count"] == 1


def test_cli_writes_blocked_evidence_and_refuses_overwrite(fixture):
    fixture.record["finish_reason"] = "length"
    save(fixture.path, fixture.record)
    output = fixture.tmp / "audit.json"
    args = ["--sft", str(fixture.args.sft), "--generation-dir", str(fixture.args.generation_dir),
            "--document-manifest", str(fixture.args.document_manifest), "--output", str(output)]
    with pytest.raises(SystemExit) as stopped:
        audit.main(args)
    assert stopped.value.code == 1
    assert json.loads(output.read_text())["status"] == "blocked"
    with pytest.raises(FileExistsError):
        audit.main(args)
