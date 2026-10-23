#!/usr/bin/env python3
"""Compare pure postprocessing on explicitly selected, already generated diagnostics.

No model, SDK or converter is loaded. Exact local training DOCX paths and hashes
are required. Scores are frozen-label agreement, not adjudications of source truth.
This audit covers existing successful outputs only, never the full holdout score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.inference_core import _parse_report
from bridge_evidence.contest.model_preserving_fusion import preserve_model_report
from bridge_evidence.evaluation.generation import aggregate_reports, score_report


def _load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_script("postprocessing_benchmark", ROOT / "scripts/benchmark_lora.py")
legacy = _load_script("postprocessing_legacy_run", ROOT / "run.py")
HEX = re.compile(r"[0-9a-f]{64}\Z")
SECTIONS = ("基本信息", "维修建议", "病害明细")


def _json_bytes(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def bind_generation(path, row, sft_sha256):
    """Validate provenance and complete-output receipts before postprocessing."""
    record, record_sha = _json_bytes(path)
    manifest, manifest_sha = _json_bytes(path.parent / "manifest.json")
    if manifest.get("sft_sha256") != sft_sha256:
        raise ValueError("Generation manifest belongs to a different SFT")
    if manifest.get("parameters", {}).get("partition") != "diagnostic" or record.get("partition") != "diagnostic":
        raise ValueError("Only explicit diagnostic generations may be audited")
    if path.stem != row["sample_sha256"] or record.get("sample_sha256") != path.stem:
        raise ValueError("Generation sample SHA does not bind to the frozen SFT")
    if path.stem not in manifest.get("selected_sample_sha256", []):
        raise ValueError("Generation was not selected in its job manifest")
    if record.get("line") != row["line"] or record.get("prompt_content_sha256") != row["prompt_sha256"]:
        raise ValueError("Generation line/prompt binding mismatch")
    if row["year"] != "2018" or record.get("year") != "2018":
        raise ValueError("Only the internal 2018 holdout may be audited")
    calls = record.get("model_calls")
    if (record.get("status") != "success" or record.get("finish_reason") != "stop"
            or not isinstance(calls, list) or len(calls) != 1):
        raise ValueError("A complete successful single-attempt generation is required")
    call = calls[0]
    raw = record.get("raw_output")
    tokens = record.get("output_tokens")
    budget = manifest.get("parameters", {}).get("max_new_tokens")
    if (not isinstance(call, dict) or call.get("status") != "success" or call.get("errors") != []
            or call.get("finish_reason") != "stop" or type(tokens) is not int or tokens < 1
            or type(budget) is not int or tokens > budget or call.get("output_tokens") != tokens
            or call.get("prompt_tokens") != record.get("prompt_tokens")
            or not isinstance(record.get("prompt_ids_sha256"), str)
            or not HEX.fullmatch(record["prompt_ids_sha256"])
            or call.get("prompt_sha256") != record["prompt_ids_sha256"]
            or not isinstance(raw, str)
            or call.get("output_sha256") != hashlib.sha256(raw.encode("utf-8")).hexdigest()):
        raise ValueError("Generation EOS/token/hash receipt is inconsistent")
    if _parse_report(raw) != record.get("prediction"):
        raise ValueError("Stored prediction differs from the completed raw model JSON")
    return record, {"generation_path": str(path.resolve()), "generation_file_sha256": record_sha,
                    "generation_manifest_sha256": manifest_sha,
                    "prediction_sha256": bench.digest(record["prediction"])}


def source_path(sample, meta, *, document_root=None, documents=None):
    expected_sha = meta.get("report_sha256")
    if not isinstance(expected_sha, str) or not HEX.fullmatch(expected_sha):
        raise ValueError("SFT metadata must contain the original report SHA-256")
    if documents is not None:
        item = documents.get(sample)
        if item is None or item.get("source_sha256") != expected_sha:
            raise ValueError("Missing or mismatched explicit document mapping")
        path = Path(item["path"])
        if not path.is_absolute():
            raise ValueError("Document manifest paths must be absolute")
    else:
        if document_root is None:
            raise ValueError("An explicit local training document root or manifest is required")
        year, name = meta.get("year"), meta.get("report")
        if any(not isinstance(x, str) or Path(x).name != x or x in {".", ".."} for x in (year, name)):
            raise ValueError("Unsafe source filename/year in SFT metadata")
        root = Path(document_root).resolve()
        path = (root / year / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source path escapes the explicit training root")
    if path.suffix.lower() != ".docx":
        raise ValueError("This pure audit supports original training DOCX only; no implicit conversion")
    if bench.file_digest(path) != expected_sha:
        raise ValueError("Original report SHA-256 mismatch")
    return path, expected_sha


def changes(before, after):
    """Exact row multisets preserve duplicates; only display sequence is excluded."""
    result = {"basic": [], "rows": {}}
    a, b = before["基本信息"], after["基本信息"]
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b or a[key] != b[key]:
            result["basic"].append({"field": key, "before_present": key in a, "after_present": key in b,
                                    "before": a.get(key), "after": b.get(key)})
    for section in SECTIONS[1:]:
        def counter(rows):
            return Counter(json.dumps({k: v for k, v in row.items() if k != "序号"},
                                      ensure_ascii=False, sort_keys=True) for row in rows)
        old, new = counter(before[section]), counter(after[section])
        result["rows"][section] = {
            "before_count": len(before[section]), "after_count": len(after[section]),
            "removed": [{"row": json.loads(row), "count": count} for row, count in (old - new).items()],
            "added": [{"row": json.loads(row), "count": count} for row, count in (new - old).items()],
            "ordered_json_changed": before[section] != after[section],
        }
    return result


def compare(record, row, document):
    original = deepcopy(record["prediction"])
    facts = legacy.extract_table_facts(document)
    temporal = legacy.detect_temporal_baseline(document, facts)
    defects = legacy.extract_defect_rows_from_tables(document)
    advice = legacy.extract_advice_from_docx(document)
    baseline = legacy._fuse_existing_success_path(deepcopy(original), document, facts, defects, advice, temporal)
    candidate, candidate_audit = preserve_model_report(original)
    gold = json.loads(row["output"])
    reports = {"raw_model": original, "legacy": baseline, "model_preserving": candidate}
    return {"strategies": {name: {"report": data, "report_sha256": bench.digest(data),
                                   "counts": {section: len(data[section]) for section in SECTIONS},
                                   "score": score_report(gold, data)} for name, data in reports.items()},
            "legacy_changes": changes(original, baseline),
            "candidate_changes": changes(original, candidate), "candidate_audit": candidate_audit,
            "extraction": {"facts": facts, "temporal": temporal,
                           "table_advice_count": len(advice), "table_defect_count": len(defects)}}


def audit(args):
    sft_sha = bench.file_digest(args.sft)
    rows = {row["sample_sha256"]: row for row in bench.load_rows(args.sft)}
    with args.sft.open(encoding="utf-8") as handle:
        meta = {i: json.loads(line).get("meta", {}) for i, line in enumerate(handle, 1) if line.strip()}
    if bench.file_digest(args.sft) != sft_sha:
        raise ValueError("SFT changed during reading")
    documents = None
    mapping_sha = None
    if args.document_manifest:
        mapping, mapping_sha = _json_bytes(args.document_manifest)
        if not isinstance(mapping, list):
            raise ValueError("Document manifest must be a list of sample_sha256/path/source_sha256 objects")
        documents = {item["sample_sha256"]: item for item in mapping}
        if len(documents) != len(mapping):
            raise ValueError("Duplicate document manifest samples")
    found = sorted(path for path in args.generation_dir.rglob("*.json") if HEX.fullmatch(path.stem))
    requested = set(args.sample_sha256 or [])
    if any(not HEX.fullmatch(value) for value in requested):
        raise ValueError("Requested sample IDs must be lowercase SHA-256")
    if requested and requested - {path.stem for path in found}:
        raise ValueError("Requested diagnostic result is missing")
    paths = [path for path in found if not requested or path.stem in requested]
    if not paths:
        raise ValueError("No existing diagnostic output JSON selected")
    results, errors, seen = [], [], set()
    for path in paths:
        try:
            if path.stem in seen:
                raise ValueError("Duplicate generation sample in input directory")
            seen.add(path.stem)
            row = rows[path.stem]
            record, provenance = bind_generation(path, row, sft_sha)
            document, source_sha = source_path(path.stem, meta[row["line"]],
                                              document_root=args.document_root, documents=documents)
            comparison = compare(record, row, document)
            if bench.file_digest(document) != source_sha or bench.file_digest(path) != provenance["generation_file_sha256"]:
                raise ValueError("Source or generation changed during postprocessing")
            results.append({"sample_sha256": path.stem, "line": row["line"],
                            "source_path": str(document.resolve()), "source_sha256": source_sha,
                            **provenance, **comparison})
        except Exception as error:
            errors.append({"sample_sha256": path.stem, "generation_path": str(path),
                           "error": type(error).__name__ + ": " + str(error)})
    output = {"schema_version": 1, "status": "blocked" if errors else "completed",
              "interpretation": "Agreement with frozen labels is not proof of factual correctness; labels may contradict source reports.",
              "scope": "Selected existing successful diagnostic outputs only; not full-cohort inference quality or failure rate.",
              "sft_sha256": sft_sha, "document_manifest_sha256": mapping_sha,
              "selected_sample_sha256": [path.stem for path in paths],
              "source_code_sha256": {str(path.relative_to(ROOT)): bench.file_digest(path) for path in
                                      (Path(__file__), ROOT / "run.py", ROOT / "src/bridge_evidence/contest/model_preserving_fusion.py",
                                       ROOT / "src/bridge_evidence/contest/constraint_cleaner.py",
                                       ROOT / "src/bridge_evidence/contest/defect_chunker.py",
                                       ROOT / "src/bridge_evidence/contest/table_facts.py",
                                       ROOT / "src/bridge_evidence/contest/temporal_router.py",
                                       ROOT / "src/bridge_evidence/contest/inference_core.py",
                                       ROOT / "src/bridge_evidence/domain_rules.py", ROOT / "scripts/benchmark_lora.py",
                                       ROOT / "src/bridge_evidence/evaluation/generation.py")},
              "records": results, "errors": errors}
    if not errors:
        output["metrics"] = {name: aggregate_reports(record["strategies"][name]["score"] for record in results)
                             for name in ("raw_model", "legacy", "model_preserving")}
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sft", "generation-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--document-root", type=Path, help="Exact local official training report root; year/name from SFT only")
    source.add_argument("--document-manifest", type=Path, help="Explicit JSON array: sample_sha256, absolute path, source_sha256")
    parser.add_argument("--sample-sha256", action="append", help="Explicit existing sample selection; repeat for more samples")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bench.write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "compared": len(result["records"]), "errors": len(result["errors"])}, ensure_ascii=False))
    if result["errors"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
