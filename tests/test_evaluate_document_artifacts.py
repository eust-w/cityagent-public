"""Synthetic documents only: offline scoring must not manufacture model evidence."""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest import document_cli
from bridge_evidence.contest.run_state import file_digest, json_digest
from bridge_evidence.evaluation.generation import METRIC_VERSION


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return file_digest(path)


def evaluator():
    name = "bridge_evidence.evaluation.document_artifacts"
    assert importlib.util.find_spec(name) is not None, "offline artifact evaluator is missing"
    return importlib.import_module(name)


@pytest.fixture
def corpus(tmp_path):
    data_root, artifacts = tmp_path / "data", tmp_path / "artifacts"
    data_root.mkdir(); artifacts.mkdir()
    fingerprint = hashlib.sha256(b"synthetic run").hexdigest()
    cases, outputs = [], []
    for n in range(2):
        identifier = f"sample-{n}"
        source = data_root / f"source-{n}.docx"
        label = data_root / f"label-{n}.docx"
        source.write_bytes(f"synthetic source {n}".encode())
        label.write_bytes(f"synthetic label {n}".encode())
        gold = {"基本信息": {"桥梁名称": f"合成桥{n}", "总体评分": 0}, "维修建议": [],
                "病害明细": [{"病害部位": "桥面", "病害类型": "裂缝"}] * 2}
        gold_path = data_root / f"gold-{n}.json"
        save(gold_path, gold)
        cases.append({"id": identifier, "source": source.name, "source_sha256": file_digest(source),
                      "label": label.name, "label_sha256": file_digest(label),
                      "gold": gold_path.name, "gold_sha256": file_digest(gold_path)})
        directory = artifacts / identifier
        directory.mkdir()
        output = artifacts / f"{identifier}.docx"
        manifest = {"document_sha256": file_digest(source), "original_document_sha256": file_digest(source),
                    "unverified_content": []}
        document_cli._bind_source(directory, manifest, source, fingerprint, file_digest(source))
        receipt = {"status": "completed", "delivery": {"ready": True},
                   "draft_sha256": json_digest(gold), "final_data": gold}
        document_cli._publish({"data": gold, "receipt": receipt}, directory, output,
                              fingerprint, file_digest(source))
        outputs.append({"id": identifier, "directory": identifier, "output": output.name,
                        "execution_sha256": file_digest(directory / "execution.json"),
                        "publication_sha256": file_digest(directory / "publication.json")})
    manifest = {"schema_version": 1, "purpose": "development-regression",
                "metric_version": METRIC_VERSION, "cohort_name": "synthetic-two",
                "metric_denominator": 2, "samples": cases}
    manifest_path = tmp_path / "manifest.json"
    save(manifest_path, manifest)
    index = {"schema_version": 1, "cohort_sha256": file_digest(manifest_path),
             "workflow": "document-agent", "run_id": "synthetic-run", "run_fingerprint": fingerprint,
             "processing_scope": "full-source", "samples": outputs}
    index_path = tmp_path / "index.json"
    save(index_path, index)
    return dict(root=tmp_path, data_root=data_root, artifacts=artifacts, manifest=manifest,
                manifest_path=manifest_path, index=index, index_path=index_path)


def evaluate(corpus):
    return evaluator().evaluate_artifacts(
        corpus["manifest_path"], corpus["index_path"], data_root=corpus["data_root"],
        artifact_root=corpus["artifacts"], manifest_sha256=file_digest(corpus["manifest_path"]),
        index_sha256=file_digest(corpus["index_path"]))


def refresh_index(corpus):
    save(corpus["index_path"], corpus["index"])


def repin_execution(corpus, n=0):
    entry = corpus["index"]["samples"][n]
    directory = corpus["artifacts"] / entry["directory"]
    entry["execution_sha256"] = file_digest(directory / "execution.json")
    marker = json.loads((directory / "publication.json").read_text())
    marker["execution_sha256"] = entry["execution_sha256"]
    entry["publication_sha256"] = save(directory / "publication.json", marker)
    refresh_index(corpus)


def test_complete_publication_scores_each_report_and_preserves_duplicate_rows(corpus):
    result = evaluate(corpus)
    assert result["metric_version"] == METRIC_VERSION
    assert result["final"]["reports_total"] == result["final"]["reports_valid"] == 2
    assert result["final"]["sections"]["defects"]["tp"] == 4
    assert result["final"]["sections"]["defects"]["f1"] == 1.0
    assert result["raw"]["reports_valid"] == 0
    assert result["raw_available_reports"] == 0
    assert result["raw"]["sections"]["defects"]["recall"] == 0.0
    assert result["release_qualified"] is False
    assert result["inference_timing_verified"] is False


@pytest.mark.parametrize("missing", ["index_entry", "report.json", "execution.json", "publication.json", "docx"])
def test_missing_results_are_failures_and_remain_in_denominator(corpus, missing):
    if missing == "index_entry":
        corpus["index"]["samples"].pop(0); refresh_index(corpus)
    elif missing == "docx":
        (corpus["artifacts"] / "sample-0.docx").unlink()
    else:
        (corpus["artifacts"] / "sample-0" / missing).unlink()
    result = evaluate(corpus)
    assert result["final"]["reports_total"] == 2
    assert result["final"]["reports_failed"] == 1
    assert result["final"]["sections"]["defects"]["fn"] == 2
    assert result["final"]["sections"]["defects"]["recall"] == 0.5


@pytest.mark.parametrize("mutation", ["report", "docx", "execution", "publication", "wrong_source",
                                      "prepared", "failed", "draft", "source_manifest", "source_binding"])
def test_tampering_wrong_source_and_unpublished_results_fail_closed(corpus, mutation):
    directory = corpus["artifacts"] / "sample-0"
    if mutation in {"report", "source_manifest", "source_binding"}:
        filename = {"report": "report.json", "source_manifest": "source_manifest.json",
                    "source_binding": "source_binding.json"}[mutation]
        with (directory / filename).open("a") as handle:
            handle.write("tampered")
    elif mutation == "docx":
        with (corpus["artifacts"] / "sample-0.docx").open("ab") as handle:
            handle.write(b"changed")
    elif mutation in {"execution", "publication"}:
        with (directory / f"{mutation}.json").open("a") as handle:
            handle.write(" ")
    elif mutation == "prepared":
        marker = json.loads((directory / "publication.json").read_text()); marker["status"] = "prepared"
        corpus["index"]["samples"][0]["publication_sha256"] = save(directory / "publication.json", marker)
        refresh_index(corpus)
    else:
        receipt = json.loads((directory / "execution.json").read_text())
        if mutation == "wrong_source": receipt["source_sha256"] = "a" * 64
        elif mutation == "failed": receipt["status"] = "failed"
        else: receipt["draft_sha256"] = "b" * 64
        save(directory / "execution.json", receipt); repin_execution(corpus)
    before = {p: p.read_bytes() for p in corpus["artifacts"].rglob("*") if p.is_file()}
    result = evaluate(corpus)
    assert result["final"]["reports_failed"] == 1
    assert result["details"][0]["error"]
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("kind", ["sample_id", "source", "output_id", "output_path", "unknown_output"])
def test_duplicate_or_unknown_sample_mapping_rejects_entire_evaluation(corpus, kind):
    if kind in {"sample_id", "source"}:
        second, first = corpus["manifest"]["samples"][1], corpus["manifest"]["samples"][0]
        for key in (["id"] if kind == "sample_id" else ["source", "source_sha256"]): second[key] = first[key]
        save(corpus["manifest_path"], corpus["manifest"])
        corpus["index"]["cohort_sha256"] = file_digest(corpus["manifest_path"])
    elif kind == "output_id": corpus["index"]["samples"].append(copy.deepcopy(corpus["index"]["samples"][0]))
    elif kind == "output_path": corpus["index"]["samples"][1]["output"] = corpus["index"]["samples"][0]["output"]
    else: corpus["index"]["samples"][0]["id"] = "outside-cohort"
    refresh_index(corpus)
    with pytest.raises(ValueError): evaluate(corpus)


@pytest.mark.parametrize("kind", ["source", "label", "gold", "denominator", "metric", "cohort", "unsupported"])
def test_changed_frozen_inputs_or_protocol_are_rejected(corpus, kind):
    if kind in {"source", "label", "gold"}:
        path = corpus["data_root"] / corpus["manifest"]["samples"][0][kind]
        path.write_bytes(path.read_bytes() + b" ")
    elif kind in {"denominator", "metric"}:
        corpus["manifest"]["metric_denominator" if kind == "denominator" else "metric_version"] = 1 if kind == "denominator" else "new-score"
        save(corpus["manifest_path"], corpus["manifest"])
        corpus["index"]["cohort_sha256"] = file_digest(corpus["manifest_path"])
    elif kind == "cohort": corpus["index"]["cohort_sha256"] = "f" * 64
    else: corpus["index"]["workflow"] = "divide-fixed"
    refresh_index(corpus)
    with pytest.raises(ValueError): evaluate(corpus)


@pytest.mark.parametrize("path", ["../source-0.docx", "/etc/passwd", "初赛测试集/source.docx"])
def test_unsafe_or_test_set_paths_rejected_before_access(corpus, path):
    corpus["manifest"]["samples"][0]["source"] = path
    save(corpus["manifest_path"], corpus["manifest"])
    corpus["index"]["cohort_sha256"] = file_digest(corpus["manifest_path"]); refresh_index(corpus)
    with pytest.raises(ValueError): evaluate(corpus)


def test_symlink_result_is_not_accepted(corpus):
    path = corpus["artifacts"] / "sample-0" / "report.json"
    target = corpus["root"] / "borrowed.json"
    path.rename(target); path.symlink_to(target)
    assert evaluate(corpus)["final"]["reports_failed"] == 1


def test_missing_raw_never_reuses_final_and_bound_raw_is_scored_separately(corpus):
    entry = corpus["index"]["samples"][0]
    case = corpus["manifest"]["samples"][0]
    raw = {"schema_version": 1, "stage": "prepublication-three-table-json",
           "source_sha256": case["source_sha256"], "run_fingerprint": corpus["index"]["run_fingerprint"],
           "data": json.loads((corpus["data_root"] / case["gold"]).read_text())}
    raw["data"]["病害明细"].pop()
    path = corpus["artifacts"] / "sample-0" / "raw.json"
    entry["raw"] = {"path": "sample-0/raw.json", "sha256": save(path, raw)}; refresh_index(corpus)
    result = evaluate(corpus)
    assert result["raw_available_reports"] == 1
    assert result["raw"]["reports_total"] == 2
    assert result["raw"]["sections"]["defects"]["tp"] == 1
    assert result["final"]["sections"]["defects"]["tp"] == 4
    raw["source_sha256"] = "c" * 64
    entry["raw"]["sha256"] = save(path, raw); refresh_index(corpus)
    result = evaluate(corpus)
    assert result["raw_available_reports"] == 0
    assert result["final"]["reports_valid"] == 2
    assert result["details"][0]["raw_error"]


def test_cli_offline_result_and_manifest_hash_guard(corpus):
    output = corpus["root"] / "scores.json"
    command = [sys.executable, str(ROOT / "scripts/evaluate_document_artifacts.py"),
               "--manifest", str(corpus["manifest_path"]), "--artifacts-index", str(corpus["index_path"]),
               "--data-root", str(corpus["data_root"]), "--artifact-root", str(corpus["artifacts"]),
               "--manifest-sha256", file_digest(corpus["manifest_path"]),
               "--index-sha256", file_digest(corpus["index_path"]), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["final"]["reports_total"] == 2
    command[command.index("--manifest-sha256") + 1] = "0" * 64
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0


def test_missing_source_binding_hash_cannot_disable_file_validation(corpus):
    path = corpus["artifacts"] / "sample-0" / "execution.json"
    receipt = json.loads(path.read_text()); receipt.pop("source_binding_sha256")
    save(path, receipt); repin_execution(corpus)
    assert evaluate(corpus)["final"]["reports_failed"] == 1


def test_changed_docx_still_rejected_after_rehashing_outer_receipts(corpus):
    from docx import Document
    path = corpus["artifacts"] / "sample-0.docx"
    document = Document(path)
    document.tables[0].rows[1].cells[1].text = "未审核的另一座桥"
    document.save(path)
    receipt_path = corpus["artifacts"] / "sample-0" / "execution.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["output"]["sha256"] = file_digest(path)
    receipt["delivery_validation"]["docx_sha256"] = file_digest(path)
    save(receipt_path, receipt); repin_execution(corpus)
    assert evaluate(corpus)["final"]["reports_failed"] == 1


def test_bound_document_outside_explicit_roots_is_not_read(corpus, monkeypatch):
    directory = corpus["artifacts"] / "sample-0"
    binding = json.loads((directory / "source_binding.json").read_text())
    binding["document"]["path"] = str(corpus["root"] / "forbidden-label-copy.docx")
    receipt = json.loads((directory / "execution.json").read_text())
    receipt["source_binding_sha256"] = save(directory / "source_binding.json", binding)
    save(directory / "execution.json", receipt); repin_execution(corpus)
    original = Path.read_bytes
    def read_bytes(path):
        assert path.name != "forbidden-label-copy.docx", "untrusted path was accessed"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert evaluate(corpus)["final"]["reports_failed"] == 1


def test_unverified_native_inventory_without_scope_receipt_is_not_complete(corpus):
    directory = corpus["artifacts"] / "sample-0"
    source_manifest = json.loads((directory / "source_manifest.json").read_text())
    source_manifest["unverified_content"] = [{"source_path": "word/document.xml/object[1]"}]
    save(directory / "source_manifest.json", source_manifest)
    binding = json.loads((directory / "source_binding.json").read_text())
    binding["manifest_sha256"] = json_digest(source_manifest)
    receipt = json.loads((directory / "execution.json").read_text())
    receipt["source_binding_sha256"] = save(directory / "source_binding.json", binding)
    save(directory / "execution.json", receipt); repin_execution(corpus)
    assert evaluate(corpus)["final"]["reports_failed"] == 1


@pytest.mark.parametrize("field", ["manifest_sha256", "index_sha256"])
def test_explicit_frozen_hashes_are_required(corpus, field):
    kwargs = dict(data_root=corpus["data_root"], artifact_root=corpus["artifacts"],
                  manifest_sha256=file_digest(corpus["manifest_path"]), index_sha256=file_digest(corpus["index_path"]))
    kwargs[field] = None
    with pytest.raises(ValueError):
        evaluator().evaluate_artifacts(corpus["manifest_path"], corpus["index_path"], **kwargs)


@pytest.mark.parametrize("failure", ["missing_docx", "prepared"])
def test_valid_independent_raw_survives_final_publication_failure(corpus, failure):
    entry = corpus["index"]["samples"][0]
    case = corpus["manifest"]["samples"][0]
    raw = {"schema_version": 1, "stage": "prepublication-three-table-json",
           "source_sha256": case["source_sha256"], "run_fingerprint": corpus["index"]["run_fingerprint"],
           "data": json.loads((corpus["data_root"] / case["gold"]).read_text())}
    entry["raw"] = {"path": "sample-0/raw.json",
                    "sha256": save(corpus["artifacts"] / "sample-0/raw.json", raw)}
    if failure == "missing_docx":
        (corpus["artifacts"] / "sample-0.docx").unlink()
    else:
        path = corpus["artifacts"] / "sample-0/publication.json"
        marker = json.loads(path.read_text()); marker["status"] = "prepared"
        entry["publication_sha256"] = save(path, marker)
    refresh_index(corpus)
    result = evaluate(corpus)
    assert result["final"]["reports_failed"] == 1
    assert result["raw_available_reports"] == 1
    assert result["raw"]["sections"]["defects"]["tp"] == 2


def test_conflicting_original_source_in_manifest_is_rejected(corpus):
    directory = corpus["artifacts"] / "sample-0"
    manifest = json.loads((directory / "source_manifest.json").read_text())
    manifest["original_document_sha256"] = corpus["manifest"]["samples"][1]["source_sha256"]
    save(directory / "source_manifest.json", manifest)
    binding = json.loads((directory / "source_binding.json").read_text())
    binding["manifest_sha256"] = json_digest(manifest)
    receipt = json.loads((directory / "execution.json").read_text())
    receipt["source_binding_sha256"] = save(directory / "source_binding.json", binding)
    save(directory / "execution.json", receipt); repin_execution(corpus)
    assert evaluate(corpus)["final"]["reports_failed"] == 1


def test_json_boolean_is_not_equivalent_to_numeric_reviewed_value(corpus):
    path = corpus["artifacts"] / "sample-0/execution.json"
    receipt = json.loads(path.read_text())
    receipt["final_data"]["基本信息"]["总体评分"] = False
    save(path, receipt); repin_execution(corpus)
    assert evaluate(corpus)["final"]["reports_failed"] == 1
