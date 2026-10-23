"""Offline, frozen-cohort scoring of document-agent publications.

This verifies artifact consistency, not model execution authenticity, semantic
source completeness, timing, or RSI promotion. It never calls a model or repairs
publication state. Input and artifact indexes are separate to preserve one
denominator across different runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from ..contest.document_delivery_validation import validate_delivery_data, validate_delivery_docx
from ..contest.run_state import json_digest
from .generation import METRIC_VERSION, aggregate_reports, score_report

PROTOCOL = "document-publication-artifacts-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SAMPLE_KEYS = {"id", "source", "source_sha256", "label", "label_sha256", "gold", "gold_sha256"}
_ENTRY_KEYS = {"id", "directory", "output", "execution_sha256", "publication_sha256"}


def _sha(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("Expected a lowercase SHA-256")
    return value


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty identifier")
    return value


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _json(path, expected=None):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked file: {path.name}")
    content = path.read_bytes()
    if expected is not None and hashlib.sha256(content).hexdigest() != _sha(expected):
        raise ValueError(f"Frozen file hash mismatch: {path.name}")
    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")
    return json.loads(content, object_pairs_hook=_pairs, parse_constant=invalid_constant)


def _relative(root, value):
    """Only explicit, normalized paths under a caller-selected root are read."""
    _text(value)
    part = PurePosixPath(value)
    if (part.is_absolute() or str(part) != value or any(p in {".", ".."} for p in part.parts)
            or "\\" in value or not part.parts or "初赛测试集" in value):
        raise ValueError("Expected a training/development relative path without traversal")
    path = root
    for component in part.parts:
        path /= component
        if path.is_symlink():
            raise ValueError("Symlinked input or artifact path is not allowed")
    return path


def _bound_document(value, roots):
    """A saved absolute conversion path cannot grant access outside these roots."""
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("Bound document path must be absolute")
    for root in roots:
        try:
            relative = Path(value).relative_to(root)
        except ValueError:
            continue
        return _relative(root, relative.as_posix())
    raise ValueError("Bound document is outside data/artifact roots")


def _load_cohort(path, expected, root):
    manifest = _json(path, expected)
    keys = {"schema_version", "purpose", "metric_version", "cohort_name", "metric_denominator", "samples"}
    if (not isinstance(manifest, dict) or set(manifest) != keys
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["purpose"] != "development-regression" or manifest["metric_version"] != METRIC_VERSION
            or not isinstance(manifest["samples"], list) or not manifest["samples"]
            or type(manifest["metric_denominator"]) is not int
            or manifest["metric_denominator"] != len(manifest["samples"])):
        raise ValueError("Invalid frozen development manifest/metric denominator")
    _text(manifest["cohort_name"])
    seen = {key: set() for key in ("id", "source", "source_sha256", "label", "gold")}
    cases = []
    for case in manifest["samples"]:
        if not isinstance(case, dict) or set(case) != _SAMPLE_KEYS:
            raise ValueError("Invalid manifest sample fields")
        for key, values in seen.items():
            value = _text(case[key])
            if value in values:
                raise ValueError(f"Duplicate manifest {key}")
            values.add(value)
        paths = {}
        for key in ("source", "label", "gold"):
            paths[key] = _relative(root, case[key])
            expected_hash = _sha(case[key + "_sha256"])
            if not paths[key].is_file() or _digest(paths[key]) != expected_hash:
                raise ValueError(f"Frozen {key} is missing or changed for {case['id']}")
        gold = _json(paths["gold"], case["gold_sha256"])
        score_report(gold, None)  # Validate metric values before accepting any results.
        cases.append((case, gold))
    return manifest, cases


def _load_index(path, expected, cohort_hash, cases):
    index = _json(path, expected)
    keys = {"schema_version", "cohort_sha256", "workflow", "run_id", "run_fingerprint", "processing_scope", "samples"}
    if (not isinstance(index, dict) or set(index) != keys
            or type(index["schema_version"]) is not int or index["schema_version"] != 1
            or index["cohort_sha256"] != cohort_hash or index["workflow"] != "document-agent"
            or index["processing_scope"] not in {"full-source", "training-text-only"}
            or not isinstance(index["samples"], list)):
        raise ValueError("Invalid artifact index; only document-agent publications are supported")
    _text(index["run_id"]); _sha(index["run_fingerprint"])
    ids = {case["id"] for case, _ in cases}
    seen = {key: set() for key in ("id", "directory", "output", "execution_sha256")}
    entries = {}
    for entry in index["samples"]:
        if (not isinstance(entry, dict) or set(entry) - _ENTRY_KEYS - {"raw"}
                or not _ENTRY_KEYS.issubset(entry) or entry["id"] not in ids):
            raise ValueError("Unknown or malformed artifact sample")
        for key, values in seen.items():
            value = _text(entry[key])
            if value in values:
                raise ValueError(f"Duplicate artifact {key}")
            values.add(value)
        # Validate syntax here; missing/invalid artifact files become per-report failures.
        for key in ("directory", "output"):
            value = entry[key]
            part = PurePosixPath(value)
            if part.is_absolute() or str(part) != value or ".." in part.parts or "\\" in value or "初赛测试集" in value:
                raise ValueError("Invalid artifact relative path")
        _sha(entry["execution_sha256"]); _sha(entry["publication_sha256"])
        entries[entry["id"]] = entry
    return index, entries


def _published(case, entry, index, data_root, artifact_root):
    # These helpers are pure checks. _completed() is deliberately NOT called:
    # that runtime recovery helper can mutate prepared publication markers.
    from ..contest.document_cli import _source_publication_context, _verify_processing_scope_file

    directory = _relative(artifact_root, entry["directory"])
    output = _relative(artifact_root, entry["output"])
    for name in ("report.json", "execution.json", "publication.json", "source_binding.json", "source_manifest.json"):
        _relative(artifact_root, (PurePosixPath(entry["directory"]) / name).as_posix())
    receipt = _json(directory / "execution.json", entry["execution_sha256"])
    marker = _json(directory / "publication.json", entry["publication_sha256"])
    report = _json(directory / "report.json")
    validate_delivery_data(report)
    fingerprint, source_hash = index["run_fingerprint"], case["source_sha256"]
    if (not isinstance(receipt, dict) or not isinstance(marker, dict)
            or marker.get("status") != "committed" or marker.get("run_fingerprint") != fingerprint
            or marker.get("execution_sha256") != entry["execution_sha256"]
            or receipt.get("status") not in {"success", "completed"} or receipt.get("cli_status") != "completed"
            or receipt.get("run_fingerprint") != fingerprint or receipt.get("source_sha256") != source_hash
            or receipt.get("report_sha256") != json_digest(report)
            or receipt.get("draft_sha256") != json_digest(report)
            or json_digest(receipt.get("final_data")) != json_digest(report)
            or receipt.get("delivery", {}).get("ready") is not True):
        raise ValueError("Result is unpublished, unsuccessful, or has mismatched report/source/run bindings")
    binding = _json(directory / "source_binding.json", _sha(receipt.get("source_binding_sha256")))
    manifest = _json(directory / "source_manifest.json")
    if (not isinstance(binding, dict) or not isinstance(manifest, dict)
            or binding.get("run_fingerprint") != fingerprint
            or binding.get("original_source_sha256") != source_hash
            or binding.get("manifest_sha256") != json_digest(manifest)):
        raise ValueError("Source manifest binding mismatch")
    document = binding.get("document")
    if document is None:
        if index["processing_scope"] != "training-text-only" or manifest.get("source_json_sha256") != source_hash:
            raise ValueError("Missing parsed document cannot claim full-source processing")
    else:
        if not isinstance(document, dict) or set(document) != {"path", "sha256"}:
            raise ValueError("Invalid bound document")
        suffix = Path(case["source"]).suffix.lower()
        if (index["processing_scope"] != "full-source" or suffix not in {".doc", ".docx"}
                or manifest.get("original_document_sha256") != source_hash
                or (suffix == ".docx" and document["sha256"] != source_hash)):
            raise ValueError("Original source conflicts with parsed document manifest")
        target = _bound_document(document["path"], (data_root, artifact_root))
        if (not target.is_file() or _digest(target) != _sha(document["sha256"])
                or manifest.get("document_sha256") != document["sha256"]):
            raise ValueError("Parsed/converted document hash mismatch")
    if not output.is_file() or receipt.get("output") != {"name": output.name, "sha256": _digest(output)}:
        raise ValueError("Published DOCX hash/name mismatch")
    context = _source_publication_context(receipt, report, manifest)
    _relative(artifact_root, (PurePosixPath(entry["directory"]) / "processing_scope.json").as_posix())
    _verify_processing_scope_file(directory, context)
    fields = ("processing_scope_sha256", "processing_note_sha256", "source_object_attestation_sha256")
    if context is not None:
        if any(receipt.get(key) != context[key] or marker.get(key) != context[key] for key in fields):
            raise ValueError("Source object publication scope mismatch")
        if receipt.get("processing_note_version") != "source-object-processing-note-v1":
            raise ValueError("Unsupported source processing note version")
        validation = validate_delivery_docx(report, output, processing_note=context["note"])
    else:
        if any(key in receipt or key in marker for key in (*fields, "processing_note_version")):
            raise ValueError("Unexpected source scope fields")
        validation = validate_delivery_docx(report, output)
    if validation != receipt.get("delivery_validation"):
        raise ValueError("DOCX readback differs from recorded delivery validation")
    return report


def _raw(case, entry, index, root):
    reference = entry.get("raw")
    if reference is None:
        return None
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Invalid raw artifact reference")
    path = _relative(root, reference["path"])
    if path == _relative(root, entry["directory"]) / "report.json":
        raise ValueError("Final report cannot be declared as raw")
    value = _json(path, _sha(reference["sha256"]))
    if (not isinstance(value, dict)
            or set(value) != {"schema_version", "stage", "source_sha256", "run_fingerprint", "data"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["stage"] != "prepublication-three-table-json"
            or value["source_sha256"] != case["source_sha256"]
            or value["run_fingerprint"] != index["run_fingerprint"]):
        raise ValueError("Raw artifact is not bound to this report/run/stage")
    validate_delivery_data(value["data"])
    return value["data"]


def evaluate_artifacts(manifest_path, index_path, *, data_root, artifact_root, manifest_sha256, index_sha256):
    """Read one immutable cohort and artifact index. Never generate or modify results."""
    manifest_path, index_path = Path(manifest_path), Path(index_path)
    _sha(manifest_sha256); _sha(index_sha256)
    data_root, artifact_root = Path(data_root).resolve(), Path(artifact_root).resolve()
    manifest, cases = _load_cohort(manifest_path, manifest_sha256, data_root)
    index, entries = _load_index(index_path, index_sha256, manifest_sha256, cases)
    raw_scores, final_scores, details = [], [], []
    for case, gold in cases:
        entry = entries.get(case["id"])
        prediction = raw = None
        error = raw_error = None
        try:
            if entry is None:
                raise ValueError("Missing artifact index entry")
            prediction = _published(case, entry, index, data_root, artifact_root)
            score_report(gold, prediction)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            prediction = None
            error = str(exc)
        if entry is not None:
            try:
                raw = _raw(case, entry, index, artifact_root)
                score_report(gold, raw)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                raw = None
                raw_error = str(exc)
        raw_score, final_score = score_report(gold, raw), score_report(gold, prediction)
        raw_scores.append(raw_score); final_scores.append(final_score)
        details.append({"id": case["id"], "source_sha256": case["source_sha256"],
                        "status": "success" if prediction is not None else "failed", "error": error,
                        "raw_status": "available" if raw is not None else "unavailable", "raw_error": raw_error,
                        "raw": raw_score, "final": final_score})
    # Detect concurrent input replacement instead of emitting a mixed snapshot.
    _json(manifest_path, manifest_sha256); _json(index_path, index_sha256)
    _load_cohort(manifest_path, manifest_sha256, data_root)
    return {"schema_version": 1, "protocol": PROTOCOL, "metric_version": METRIC_VERSION,
            "scope": "offline development-regression artifact consistency; internal proxy, not competition score",
            "cohort_name": manifest["cohort_name"], "cohort_sha256": manifest_sha256,
            "artifacts_index_sha256": index_sha256, "run_id": index["run_id"],
            "run_fingerprint": index["run_fingerprint"], "workflow": index["workflow"],
            "processing_scope": index["processing_scope"], "metric_denominator": len(cases),
            "raw_stage": "explicit prepublication parsed three-table artifact, not individual model responses",
            "raw_available_reports": sum(item["prediction_valid"] for item in raw_scores),
            "raw_missing_policy": "retain full cohort; unavailable raw predicts nothing and retains all gold FN",
            "raw": aggregate_reports(raw_scores), "final": aggregate_reports(final_scores), "details": details,
            "model_execution_verified": False, "inference_timing_verified": False, "release_qualified": False,
            "limitations": ["Hashes prove consistency with supplied frozen indexes, not model execution authenticity.",
                            "No semantic source-completeness, independent model review, speed, or RSI promotion claim.",
                            "Only document-agent committed publications are supported; divide/legacy artifacts are unsupported.",
                            "Gold JSON must be audited against its separately hashed original label before cohort freeze."]}
