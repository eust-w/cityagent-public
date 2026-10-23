"""Content-bound batch boundary for the document agent; no model loading here."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from uuid import uuid4

from . import divide_cli
from .document_delivery_validation import validate_delivery_data, validate_delivery_docx
from .docx_writer import write_numbered_extraction_docx as write_extraction_docx
from .evidence_manifest import build_manifest
from .inference_core import _parse_report
from .local_model_client import validate_temperature
from .run_state import atomic_json, bind_work_directory, file_digest, json_digest, validate_output_names
from .training_evidence_batch import load_training_evidence_batch, verify_training_evidence_batch

VERSION = "document-cli-request-policy-v1"
_DEFAULTS = dict(max_model_len=65536, output_tokens=4096, tool_tokens=4096,
                 max_sessions=12, session_iterations=8, session_calls=12, review_iterations=32,
                 review_batch_items=0,
                 report_timeout=1200.0, max_review_attempts=2, history_reserve_tokens=4096,
                 max_model_calls=192, enable_thinking=False, temperature=0.0, dynamic_output_budget=False)
_OCR_LIMITATIONS = ["CPU OCR reads text characters only; it does not understand diagrams, damage images or measurements",
                    "OCR observations remain unverified and require source-grounded model review"]


def add_arguments(parser):
    """Keep existing whole/divide options unchanged and avoid duplicate flags."""
    parser.add_argument("--training-evidence-manifest", type=Path,
                        help="Explicit hash-bound official-training text batch; no dataset discovery; does not cover images/layout")
    for name, default in _DEFAULTS.items():
        if name in {"enable_thinking", "dynamic_output_budget"}:
            parser.add_argument("--document-" + name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=default,
                                help=("Enable own-model thinking; included in identity" if name == "enable_thinking" else
                                      "Allocate each complete request from exact server-tokenized remaining context; disabled by default"))
            continue
        parser.add_argument("--document-" + name.replace("_", "-"), type=type(default), default=default,
                            help=f"document-agent {name} (default: {default})")
    parser.add_argument("--document-total-report-timeout", type=float, default=None,
                        help="Optional fresh-run seconds per report, from worker preparation through DOCX publication; incompatible with --resume")
    parser.add_argument("--document-ocr", choices=("none", "tesseract"), default="none",
                        help="Explicit local CPU text OCR only; disabled by default; no external vision API")
    parser.add_argument("--document-ocr-language", default="chi_sim+eng")
    parser.add_argument("--document-ocr-timeout", type=int, default=30)
    parser.add_argument("--document-vision", choices=("none", "own-model"), default="none",
                        help="Explicit own-model image observations using the report's common budget")
    parser.add_argument("--document-vision-config", type=Path,
                        help="Pinned JSON with native processor Python, source audit and PPU runtime paths")


class _BoundOCR:
    """Keep actual OCR bytes/configuration bound to tool receipts and resumes."""
    def __init__(self, delegate, identity):
        self.delegate, self.identity = delegate, identity
        self.version, self.model = identity["version"], identity["language"]
        self.cache_key = "tesseract:" + json_digest(identity)

    def verify(self):
        artifacts = [self.identity["executable"], *self.identity["traineddata"].values()]
        if any((item.get("resolved_path") is not None
                and str(Path(item["path"]).resolve(strict=True)) != item["resolved_path"])
               or file_digest(Path(item["path"])) != item["sha256"] for item in artifacts):
            raise ValueError("CPU OCR executable or language data changed during the bound run")

    def recognize(self, image_bytes, *, region):
        self.verify()
        previous = os.environ.get("TESSDATA_PREFIX")
        try:
            # Workers process one report each. Pin the model data actually
            # opened by Tesseract rather than relying on a mutable environment.
            os.environ["TESSDATA_PREFIX"] = self.identity["tessdata_directory"]
            observation = self.delegate.recognize(image_bytes, region=region)
            self.verify()
            return observation
        finally:
            if previous is None:
                os.environ.pop("TESSDATA_PREFIX", None)
            else:
                os.environ["TESSDATA_PREFIX"] = previous


def _make_ocr_backend(args):
    kind = getattr(args, "document_ocr", "none")
    if kind == "none":
        return None, {"backend": "none", "status": "disabled", "limitations": _OCR_LIMITATIONS}
    if kind != "tesseract":
        raise ValueError("Only explicitly selected local Tesseract CPU OCR is supported")
    language = getattr(args, "document_ocr_language", "chi_sim+eng")
    timeout = getattr(args, "document_ocr_timeout", 30)
    if (type(timeout) is not int or timeout <= 0 or not isinstance(language, str)
            or not re.fullmatch(r"[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)?(?:\+[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)?)*", language)):
        raise ValueError("CPU OCR needs valid language names and a positive integer timeout")
    from .document_tools import make_tesseract_backend
    previous = os.environ.get("TESSDATA_PREFIX")
    try:
        pinned = getattr(args, "document_vision_identity", {})
        if pinned.get("backend") == "tesseract":
            os.environ["TESSDATA_PREFIX"] = pinned["tessdata_directory"]
        backend = make_tesseract_backend(language=language, timeout_seconds=timeout)
        executable = Path(backend.executable).resolve(strict=True)
        executable_hash = file_digest(executable)
        version = subprocess.run([str(executable), "--version"], capture_output=True, text=True,
                                 timeout=timeout, check=True)
        version = (version.stdout + "\n" + version.stderr).strip()
        if not version:
            raise RuntimeError("Tesseract did not report a version")
        listing = subprocess.run([str(executable), "--list-langs"], capture_output=True, text=True,
                                 timeout=timeout, check=True)
        match = re.search(r'List of available languages in "([^"]+)"', listing.stdout + "\n" + listing.stderr)
        if match is None:
            raise RuntimeError("Cannot identify Tesseract's actual language-data directory")
        directory = Path(match[1]).resolve(strict=True)
        traineddata = {}
        for name in language.split("+"):
            path = directory / (name + ".traineddata")
            # Package managers commonly symlink language packs. Bind both the
            # lookup path and its target so retargeting cannot bypass hashing.
            traineddata[name] = {"path": str(path), "resolved_path": str(path.resolve(strict=True)),
                                 "sha256": file_digest(path)}
        if file_digest(executable) != executable_hash:
            raise ValueError("CPU OCR executable changed while probing its version")
        identity = {"backend": "tesseract", "status": "available", "device": "cpu", "version": version,
                    "executable": {"path": str(executable), "sha256": executable_hash},
                    "language": language, "timeout_seconds": timeout, "tessdata_directory": str(directory),
                    "traineddata": traineddata, "pillow_version": importlib.metadata.version("Pillow"),
                    "shared_libraries_checksum_verified": False, "external_vision_api": False,
                    "limitations": _OCR_LIMITATIONS}
        backend.executable = str(executable)
        return _BoundOCR(backend, identity), identity
    finally:
        if previous is None:
            os.environ.pop("TESSDATA_PREFIX", None)
        else:
            os.environ["TESSDATA_PREFIX"] = previous


def _workflow_types():
    # SDK/workflow imports remain lazy, including in spawned workers.
    from .document_workflow import DocumentSettings, DocumentWorkflow
    return DocumentSettings, DocumentWorkflow


def _make_vision_backend(args):
    kind = getattr(args, "document_vision", "none")
    configuration = getattr(args, "document_vision_config", None)
    if kind == "none":
        if configuration is not None:
            raise ValueError("Vision configuration requires explicit --document-vision own-model")
        backend, identity = _make_ocr_backend(args)
        return backend, None, identity
    if kind != "own-model" or configuration is None or getattr(args, "document_ocr", "none") != "none":
        raise ValueError("Select own-model vision with its configuration and CPU OCR disabled")
    from .document_vision_runtime import OwnModelVisionFactory
    factory = OwnModelVisionFactory(configuration, model=args.served_model,
                                   model_identity=args.document_model_identity, base_url=args.vllm_url)
    return None, factory, copy.deepcopy(factory.identity)


def _settings(args):
    settings_type, _ = _workflow_types()
    values = {name: getattr(args, "document_" + name, default) for name, default in _DEFAULTS.items()}
    for name, value in values.items():
        if name in {"enable_thinking", "dynamic_output_budget"}:
            valid = type(value) is bool
        elif name == "temperature":
            validate_temperature(value)
            valid = True
        elif name == "report_timeout":
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        elif name == "review_batch_items":
            valid = type(value) is int and value >= 0
        else:
            valid = type(value) is int and value > 0
        if not valid:
            if name == "enable_thinking":
                raise ValueError("document-enable-thinking must be a boolean")
            if name == "review_batch_items":
                raise ValueError("document-review-batch-items must be a nonnegative integer")
            raise ValueError(f"document-{name.replace('_', '-')} must be positive and finite")
    if not values["dynamic_output_budget"] and values["output_tokens"] + values["tool_tokens"] + values["history_reserve_tokens"] >= values["max_model_len"]:
        raise ValueError("Document output/tool/history budgets must leave room within model context")
    return settings_type(**values)


def _total_report_timeout(args):
    value = getattr(args, "document_total_report_timeout", None)
    if value is not None:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("document-total-report-timeout must be positive and finite")
        if getattr(args, "resume", False):
            raise ValueError("document-total-report-timeout is fresh-run only; cannot combine with --resume")
    return value


def _check_total_deadline(deadline, stage):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(f"Document total report deadline exhausted at {stage}")


def _safe_name(name):
    if (not isinstance(name, str) or not name or Path(name).name != name
            or name in {".", ".."} or "\\" in name):
        raise ValueError("Unsafe document output name")
    return name


def _overlap(left, right):
    return left == right or left in right.parents or right in left.parents


def _prepare(args):
    if getattr(args, "text_dir", None):
        raise ValueError("Document agent requires structural originals or explicit training text inputs")
    batch_path = getattr(args, "training_evidence_manifest", None)
    if sum(bool(value) for value in (args.input_dir, args.training_evidence_json, batch_path)) != 1:
        raise ValueError("Specify exactly one explicit --input-dir, --training-evidence-json or --training-evidence-manifest; no dataset discovery")
    if _overlap(args.work_dir, args.result_dir):
        raise ValueError("Work and result directories must be separate, non-overlapping directories")
    if batch_path is not None:
        if args.limit is not None:
            raise ValueError("--limit cannot truncate an explicit training evidence manifest")
        rows, identity = load_training_evidence_batch(batch_path, forbidden_roots=(args.work_dir, args.result_dir))
        args.document_training_batch_identity = identity
        return [(_safe_name(name), manifest, None, original) for name, manifest, original in rows]
    if args.training_evidence_json:
        source = args.training_evidence_json
        if source.is_symlink() or any(root == source or root in source.parents for root in (args.work_dir, args.result_dir)):
            raise ValueError("Training source must be outside work/result directories and not a symlink")
        name, manifest = divide_cli._training_manifest(source)
        return [(_safe_name(name), manifest, None, source)]
    if not args.input_dir.is_dir():
        raise ValueError("Input directory does not exist")
    if any(_overlap(args.input_dir, root) for root in (args.work_dir, args.result_dir)):
        raise ValueError("Input and work/result directories must not overlap")
    files = sorted(p for p in args.input_dir.rglob("*") if p.is_file()
                   and p.suffix.lower() in {".doc", ".docx"} and not p.name.startswith("~$"))
    if args.limit is not None:
        if type(args.limit) is not int or args.limit < 1:
            raise ValueError("limit must be positive")
        files = files[:args.limit]
    if not files:
        raise ValueError("Input directory has no reports")
    if any(p.is_symlink() for p in files):
        raise ValueError("Symlink inputs are unsupported")
    validate_output_names(files)
    if any(p.suffix.lower() == ".doc" for p in files) and not shutil.which("soffice"):
        raise ValueError("LibreOffice is required for .doc conversion")
    return [(_safe_name(p.stem), None, p, p) for p in files]


def _model_identity(args):
    client = divide_cli._make_client(args)
    if args.served_model not in client.models():
        raise ValueError("Requested model is not served by local model service")
    weights = args.served_adapter / "adapter_model.safetensors"
    config = args.served_adapter / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise ValueError("Local served adapter weights and config are required for execution identity")
    adapter = {"weights_sha256": file_digest(weights), "config_sha256": file_digest(config)}
    if args.served_merged_model:
        identity = divide_cli._merged_identity(args, client, adapter["weights_sha256"])
        if identity["binding_source"] == "model_catalog.alias+model_info.model_path":
            files = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
            artifacts = identity["verified_artifacts"]
            if not all(name in artifacts for name in files):
                raise ValueError("SGLang local chat rendering requires verified tokenizer/template artifacts")
            args.chat_tokenizer_directory = args.served_merged_model
            args.chat_tokenizer_hashes = {name: artifacts[name]["sha256"] for name in files}
            divide_cli._MergedChatTokenizer(args.chat_tokenizer_directory, args.chat_tokenizer_hashes)
    else:
        entries = [item for item in client.model_catalog if item.get("id") == args.served_model]
        # vLLM can expose a base model and its LoRA under the same ID. A base
        # entry is not an adapter candidate; multiple adapter candidates still
        # make routing ambiguous, even when just one has our requested path.
        adapters = [item for item in entries if item.get("parent") is not None]
        if (len(adapters) != 1 or not isinstance(adapters[0].get("parent"), str)
                or not adapters[0]["parent"].strip()
                or adapters[0].get("root") != str(args.served_adapter)):
            raise ValueError("Served model catalog does not bind this model ID to requested LoRA path")
        parent = adapters[0]["parent"]
        bases = [item for item in client.model_catalog
                 if item.get("id") == parent and item.get("parent") is None]
        if (len(bases) != 1 or not isinstance(bases[0].get("root"), str)
                or not bases[0]["root"].strip()):
            raise ValueError("Served LoRA parent is not uniquely traceable to a base model catalog entry")
        identity = {"kind": "dynamic_lora", "adapter_path": str(args.served_adapter),
                    "binding_source": "model_catalog.id+root+parent",
                    "base_model_id": parent, "base_model_root": bases[0]["root"],
                    "runtime_weights_checksum_verified": False,
                    "base_weights_full_checksum_verified": False}
    return client, {"adapter": adapter, "deployment": identity}


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_code_snapshot(args):
    for relative, expected in getattr(args, "document_code_hashes", {}).items():
        if file_digest(args.document_code_root / relative) != expected:
            raise ValueError("Code/prompt/tool source changed during the bound run")


def _verify_training_batch(args, original=None):
    identity = getattr(args, "document_training_batch_identity", None)
    if identity is not None:
        verify_training_evidence_batch(identity, original=original)


def _bind_source(directory, manifest, target, fingerprint, source_hash):
    """Pin the parser's actual input, including a converted .doc's bytes."""
    target_hash = file_digest(target) if target else None
    if target and manifest.get("document_sha256", target_hash) != target_hash:
        raise ValueError("Document changed during manifest parsing")
    binding = {"run_fingerprint": fingerprint, "original_source_sha256": source_hash,
               "manifest_sha256": json_digest(manifest),
               "document": {"path": str(target), "sha256": target_hash} if target else None}
    path = directory / "source_binding.json"
    manifest_path = directory / "source_manifest.json"
    if path.exists():
        if _load_json(path) != binding:
            raise ValueError("Parsed/converted source changed since the document checkpoint")
        _verify_source_binding(directory, fingerprint, source_hash)
    else:
        atomic_json(manifest_path, manifest)
        atomic_json(path, binding)
    return file_digest(path)


def _verify_source_binding(directory, fingerprint, source_hash):
    path = directory / "source_binding.json"
    binding = _load_json(path)
    if (binding.get("run_fingerprint") != fingerprint
            or binding.get("original_source_sha256") != source_hash
            or binding.get("manifest_sha256") != json_digest(_load_json(directory / "source_manifest.json"))):
        raise ValueError("Bound source manifest identity/hash mismatch")
    document = binding.get("document")
    if document is not None:
        target = Path(document["path"])
        if target.is_symlink() or file_digest(target) != document["sha256"]:
            raise ValueError("Parsed/converted document changed during the bound run")
    return file_digest(path)


_PROCESSING_NOTE_VERSION = "source-object-processing-note-v1"
_SOURCE_BINDING_FIELDS = ("source_catalog_sha256", "scope_sha256", "semantic_input_sha256", "attestation_sha256")


def _require_sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid {label} SHA256")
    return value


def processing_scope_note(processing_scope):
    """Present trusted accepted scope verbatim; never decide semantic relevance.

    This pure formatter is shared by export and completed-result verification.
    Each occurrence/locator/uncertainty/limitation is retained, not summarized by
    another model or inserted into the original engineering conclusions.
    """
    if not isinstance(processing_scope, dict) or processing_scope.get("object_gate_passed") is not True:
        raise ValueError("Processing scope must have an accepted object gate")
    rows = processing_scope.get("objects")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Processing scope requires explicit source object occurrences")
    if processing_scope.get("all_source_semantics_verified") is not False:
        raise ValueError("Unrecognized-object delivery cannot claim all source semantics verified")
    states = {"unread": "未识别", "candidate": "已有候选表示，语义尚未验证", "semantics_verified": "该对象语义已验证"}
    modes = {"not_relied_on": "本次三表未依赖此对象", "outside_task": "不属于本次三表任务范围"}
    lines = ["本文件的完成范围为原报告的三表信息抽取。以下源对象的处置经同一自有模型独立审核接受；处置通过不代表原对象已经被完整识别。",
             "未识别对象未作为三表证据使用；下列限制继续适用。"]
    seen = set()
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("Processing scope object must be an object")
        oid, locator = row.get("object_id"), row.get("locator")
        if (not isinstance(oid, str) or re.fullmatch(r"so_[0-9a-f]{32}", oid) is None
                or oid in seen or not isinstance(locator, str) or not locator.strip()):
            raise ValueError("Processing scope requires distinct occurrence IDs and original locators")
        seen.add(oid)
        limitations = row.get("source_limitations")
        if (row.get("review_status") != "accepted" or row.get("object_gate_passed") is not True
                or row.get("representation_state") not in states or row.get("disposition") not in modes
                or not isinstance(row.get("model_call_id"), str) or not row["model_call_id"].strip()
                or not isinstance(limitations, list)
                or any(not isinstance(v, str) or not v.strip() for v in limitations)
                or (row["representation_state"] != "semantics_verified" and not limitations)):
            raise ValueError("Processing scope contains an unaccepted or incomplete occurrence")
        lines.append(f"{number}. 对象 {oid}；原文位置：{locator}；表示状态：{states[row['representation_state']]}；处置：{modes[row['disposition']]}。")
        lines.extend(f"   适用限制：{value}" for value in limitations)
    return "\n".join(lines)


def _source_publication_context(receipt, data, manifest):
    """Validate bindings from the trusted workflow, not offline model approval.

    The workflow must replay the actual source-object ledger and real model
    receipts before returning this structure. Here we check that publication is
    the same frozen result, scope, complete native inventory and accepted finish.
    """
    inventory = manifest.get("unverified_content", [])
    if not isinstance(inventory, list):
        raise ValueError("Bound manifest has an invalid source object inventory")
    delivery = receipt.get("delivery", {})
    finish = receipt.get("accepted_finish")
    source_present = "source_objects" in receipt or (isinstance(delivery, dict) and "source_objects" in delivery)
    finish_source_present = isinstance(finish, dict) and any(str(k).startswith("source_object_") for k in finish)
    if not inventory:
        if source_present or finish_source_present:
            raise ValueError("Source object receipt is inconsistent with an empty native inventory")
        return None
    source = receipt.get("source_objects")
    if (not isinstance(source, dict) or not isinstance(delivery, dict)
            or source != delivery.get("source_objects")
            or source.get("passed") is not True or source.get("catalog_ready") is not True
            or source.get("blockers") != [] or type(source.get("inventory_count")) is not int
            or source["inventory_count"] != len(inventory)):
        raise ValueError("Native objects require the same complete accepted source coverage in receipt and delivery")
    for field in _SOURCE_BINDING_FIELDS:
        _require_sha(source.get(field), field)
    draft_sha = json_digest(data)
    version = receipt.get("workspace_version")
    if (type(version) is not int or version < 0 or not isinstance(finish, dict)
            or finish.get("workspace_version") != version or delivery.get("workspace_version") != version
            or receipt.get("draft_sha256") != draft_sha or delivery.get("draft_sha256") != draft_sha
            or finish.get("draft_sha256") != draft_sha or json_digest(receipt.get("final_data")) != draft_sha
            or delivery.get("ready") is not True
            or finish.get("source_object_attestation_sha256") != source["attestation_sha256"]
            or finish.get("source_object_semantic_input_sha256") != source["semantic_input_sha256"]
            or finish.get("source_object_processing_scope_sha256") != json_digest(source.get("processing_scope"))):
        raise ValueError("Source scope, accepted finish and reviewed draft are not bound to the same state")
    scope, coverage = source.get("processing_scope"), source.get("coverage")
    if (not isinstance(scope, dict) or not isinstance(coverage, dict)
            or scope.get("source_catalog_sha256") != source["source_catalog_sha256"]
            or scope.get("scope_sha256") != source["scope_sha256"]
            or scope.get("unverified_content") != inventory
            or scope.get("object_gate_passed") is not True
            or coverage.get("scope_sha256") != source["scope_sha256"]
            or coverage.get("object_gate_passed") is not True
            or coverage.get("total_objects") != len(inventory)
            or coverage.get("blocking_count") != 0 or coverage.get("accepted_count") != len(inventory)
            or coverage.get("all_source_semantics_verified") is not False):
        raise ValueError("Processing scope does not preserve the complete accepted native inventory")
    rows, core = scope.get("objects"), coverage.get("objects")
    locators = [item.get("source_path") if isinstance(item, dict) else None for item in inventory]
    if (not isinstance(rows, list) or not isinstance(core, list)
            or len(rows) != len(inventory) or len(core) != len(inventory)
            or any(not isinstance(v, str) or not v for v in locators)
            or len(set(locators)) != len(locators)
            or any(not isinstance(row, dict) or not isinstance(row.get("object_id"), str) for row in rows + core)
            or any(not isinstance(row.get("locator"), str) for row in rows)):
        raise ValueError("Processing scope occurrence inventory is malformed")
    core_index = {row.get("object_id"): row for row in core}
    if (len(core_index) != len(core) or {row.get("locator") for row in rows} != set(locators)
            or any({key: value for key, value in row.items() if key not in {"locator", "kind"}}
                   != core_index.get(row.get("object_id")) for row in rows)):
        raise ValueError("Visible processing scope differs from the accepted occurrence decisions")
    expected_attestation = json_digest({"catalog_sha256": source["source_catalog_sha256"],
        "scope_sha256": source["scope_sha256"], "objects": core, "passed": source["passed"], "blockers": source["blockers"]})
    if expected_attestation != source["attestation_sha256"]:
        raise ValueError("Source object attestation checksum differs from accepted coverage")
    note = processing_scope_note(scope)
    return {"scope": copy.deepcopy(scope), "note": note,
            "processing_scope_sha256": json_digest(scope),
            "processing_note_sha256": json_digest(note),
            "source_object_attestation_sha256": source["attestation_sha256"]}


def _verify_processing_scope_file(directory, context):
    path = directory / "processing_scope.json"
    if context is None:
        if path.exists() or path.is_symlink():
            raise ValueError("Unexpected processing scope for a report without native objects")
        return
    if (path.is_symlink() or not path.is_file() or _load_json(path) != context["scope"]
            or json_digest(_load_json(path)) != context["processing_scope_sha256"]):
        raise ValueError("Saved processing scope is missing or differs from accepted scope")


def _completed(name, directory, output, fingerprint, source_hash):
    """Reuse actual completed artifacts, never infer success from a filename."""
    execution = directory / "execution.json"
    publication = directory / "publication.json"
    marker = _load_json(publication) if publication.is_file() else None
    if not output.exists():
        # A completed terminal record with a missing result is corruption, not
        # a checkpoint. An interrupted publication has no completed terminal.
        terminal = directory / "document_summary.json"
        if ((marker is not None and marker.get("status") == "committed")
                or (terminal.is_file() and _load_json(terminal).get("status") == "completed")):
            raise ValueError("Completed output is missing")
        return None
    if output.is_symlink() or not output.is_file() or not execution.is_file():
        raise ValueError("Untracked or invalid existing output")
    receipt = _load_json(execution)
    source_binding_hash = _verify_source_binding(directory, fingerprint, source_hash)
    report = directory / "report.json"
    report_data = _load_json(report) if report.is_file() else None
    if (not isinstance(marker, dict) or marker.get("run_fingerprint") != fingerprint
            or marker.get("status") not in {"prepared", "committed"}
            or marker.get("execution_sha256") != file_digest(execution)):
        raise ValueError("Completed execution receipt hash mismatch")
    if (receipt.get("cli_status") != "completed" or receipt.get("run_fingerprint") != fingerprint
            or receipt.get("source_sha256") != source_hash or not report.is_file()
            or receipt.get("source_binding_sha256") != source_binding_hash
            or receipt.get("report_sha256") != json_digest(report_data)
            or receipt.get("output") != {"name": output.name, "sha256": file_digest(output)}):
        raise ValueError("Completed result identity/hash mismatch")
    context = _source_publication_context(receipt, report_data, _load_json(directory / "source_manifest.json"))
    _verify_processing_scope_file(directory, context)
    if context is not None:
        for field in ("processing_scope_sha256", "processing_note_sha256", "source_object_attestation_sha256"):
            if receipt.get(field) != context[field] or marker.get(field) != context[field]:
                raise ValueError("Completed source scope publication binding differs")
        if receipt.get("processing_note_version") != _PROCESSING_NOTE_VERSION:
            raise ValueError("Completed processing note version differs")
        validation = validate_delivery_docx(report_data, output, processing_note=context["note"])
    else:
        if any(key in receipt or key in marker for key in ("processing_scope_sha256", "processing_note_sha256",
                                                         "source_object_attestation_sha256", "processing_note_version")):
            raise ValueError("Unexpected source scope metadata without native objects")
        validation = validate_delivery_docx(report_data, output)
    if validation != receipt.get("delivery_validation"):
        raise ValueError("Completed DOCX readback differs from its delivery audit")
    terminal = directory / "document_summary.json"
    if terminal.is_file():
        previous = _load_json(terminal)
        if (previous.get("status") == "completed"
                and previous.get("execution_sha256") != file_digest(execution)):
            raise ValueError("Completed execution receipt hash mismatch")
    if marker["status"] == "prepared":
        atomic_json(publication, {**marker, "status": "committed"})
    return {"name": name, "status": "completed", "output": output.name,
            "execution_sha256": file_digest(execution), "reused": True,
            "model_calls_this_attempt": 0}


def _publish(result, directory, output, fingerprint, source_hash, *, deadline_guard=None):
    check = deadline_guard if deadline_guard is not None else lambda stage: None
    check("before_export")
    data, receipt = result.get("data"), result.get("receipt")
    if (not isinstance(data, dict) or not isinstance(data.get("基本信息"), dict)
            or not isinstance(data.get("病害明细"), list) or not isinstance(data.get("维修建议"), list)
            or not isinstance(receipt, dict) or receipt.get("status") not in {"success", "completed"}):
        raise ValueError("Workflow did not return successful Chinese three-table data and receipt")
    # These checks bind publication to the model-reviewed draft; they do not
    # judge the source's meaning or manufacture missing report values.
    _parse_report(json.dumps(data, ensure_ascii=False, allow_nan=False))
    validate_delivery_data(data)
    if (receipt.get("delivery", {}).get("ready") is not True
            or receipt.get("draft_sha256") != json_digest(data)
            or json_digest(receipt.get("final_data")) != json_digest(data)):
        raise ValueError("Published tables differ from the workflow's accepted draft")
    source_binding_hash = _verify_source_binding(directory, fingerprint, source_hash)
    context = _source_publication_context(receipt, data, _load_json(directory / "source_manifest.json"))
    scope_path = directory / "processing_scope.json"
    if context is not None and not scope_path.exists() and not scope_path.is_symlink():
        atomic_json(scope_path, context["scope"])
    _verify_processing_scope_file(directory, context)
    pending = output.parent / (".pending-" + uuid4().hex + ".docx")
    try:
        check("before_docx_write")
        if context is None:
            write_extraction_docx(data, str(pending))
        else:
            write_extraction_docx(data, str(pending), processing_note=context["note"])
        check("after_docx_write")
        delivery_validation = (validate_delivery_docx(data, pending) if context is None else
                               validate_delivery_docx(data, pending, processing_note=context["note"]))
        check("after_export_readback")
        if _verify_source_binding(directory, fingerprint, source_hash) != source_binding_hash:
            raise ValueError("Bound source changed during report publication")
        # Never overwrite a prior model receipt or expose an unfinished DOCX.
        bound = {**receipt, "cli_status": "completed", "run_fingerprint": fingerprint,
                 "source_sha256": source_hash, "report_sha256": json_digest(data),
                 "source_binding_sha256": source_binding_hash,
                 "delivery_validation": delivery_validation,
                 "output": {"name": output.name, "sha256": delivery_validation["docx_sha256"]}}
        scope_binding = {} if context is None else {key: context[key] for key in
            ("processing_scope_sha256", "processing_note_sha256", "source_object_attestation_sha256")}
        if context is not None:
            bound.update(scope_binding, processing_note_version=_PROCESSING_NOTE_VERSION)
        atomic_json(directory / "report.json", data)
        atomic_json(directory / "execution.json", bound)
        marker = {"run_fingerprint": fingerprint, "status": "prepared", **scope_binding,
                  "execution_sha256": file_digest(directory / "execution.json")}
        atomic_json(directory / "publication.json", marker)
        if file_digest(pending) != delivery_validation["docx_sha256"]:
            raise ValueError("DOCX changed after delivery readback")
        check("before_atomic_publication")
        _verify_processing_scope_file(directory, context)
        if context is not None:
            check("after_processing_scope_recheck")
        os.link(pending, output)  # Atomic publication, refuses an existing file.
        atomic_json(directory / "publication.json", {**marker, "status": "committed"})
    finally:
        pending.unlink(missing_ok=True)
    return file_digest(directory / "execution.json")


def _run_document(name, manifest, source, original, args, settings, fingerprint, source_hash, convert_doc):
    started = time.perf_counter()
    total_timeout = _total_report_timeout(args)
    # This optional fresh-run horizon also pays for conversion, source binding
    # and export. The existing resumable workflow budget remains independent.
    deadline = time.monotonic() + total_timeout if total_timeout is not None else None
    check = lambda stage: _check_total_deadline(deadline, stage)
    directory = args.work_dir / "documents" / name
    output = args.result_dir / (name + ".docx")
    outcome = {"name": name, "status": "failed", "reused": False, "worker_pid": os.getpid()}
    previous_cwd = Path.cwd()
    env = {"LOG_TRACE_LOG_DIR": str(directory / "sdk-logs"), "LOG_TRACE_LOG_FILE": "tongagents.log",
           "LOG_TRACE_DB_PATH": str(directory / "sdk-logs" / "traces.db"), "LOG_TRACE_BACKEND": "local"}
    previous_env = {key: os.environ.get(key) for key in env}
    result, client = None, None
    try:
        check("before_preparation")
        directory.mkdir(parents=True, exist_ok=True)
        os.chdir(directory)
        os.environ.update(env)
        _verify_code_snapshot(args)
        _verify_training_batch(args, original)
        if file_digest(original) != source_hash:
            raise ValueError("Input changed before document execution")
        if args.resume:
            prior = _completed(name, directory, output, fingerprint, source_hash)
            if prior is not None:
                _verify_training_batch(args, original)
                outcome.update(prior)
                return outcome
        if output.exists():
            raise FileExistsError("Refusing to overwrite existing output")
        vision_backend, vision_factory, vision_identity = _make_vision_backend(args)
        if vision_identity != getattr(args, "document_vision_identity", vision_identity):
            raise ValueError("Image backend identity changed between preflight and worker startup")
        _, workflow_type = _workflow_types()
        check("before_client")
        client = divide_cli._make_client(args)
        config = {"url": client.base_url + "/v1", "model": args.served_model, "api_key": client.api_key,
                  "max_tokens": settings.output_tokens, "timeout": args.model_timeout,
                  "enable_thinking": settings.enable_thinking,
                  "temperature": settings.temperature,
                  "max_iterations": settings.session_iterations, "max_calls": settings.session_calls}
        target = None
        check("before_source_preparation")
        if getattr(args, "document_training_batch_identity", None) is not None:
            # Each worker reopens its own original JSON, never a global input
            # flag or another entry from the batch. Bind the parsed bytes too.
            parsed_name, parsed_manifest = divide_cli._training_manifest(original)
            if parsed_name != name or json_digest(parsed_manifest) != json_digest(manifest):
                raise ValueError("Training batch source changed before worker parsing")
            manifest = parsed_manifest
        if source is not None:
            target = Path(convert_doc(source, directory / "converted") if source.suffix.lower() == ".doc" else source)
            check("after_conversion")
            if target.is_symlink():
                raise ValueError("Parsed/converted input must not be a symlink")
            target = target.resolve()
            manifest = build_manifest(target)
            manifest["original_document_sha256"] = source_hash
        check("after_source_preparation")
        _bind_source(directory, manifest, target, fingerprint, source_hash)
        check("before_workflow")
        run_identity = {"run_fingerprint": fingerprint, "source_sha256": source_hash}
        workflow_options = {}
        if vision_factory is not None:
            workflow_options["vision_backend_factory"] = vision_factory
        if total_timeout is not None:
            run_identity["document_total_report_timeout"] = total_timeout
            workflow_options["external_deadline"] = deadline
        result = workflow_type(manifest, client, config, directory, settings=settings, source_path=target,
                               vision_backend=vision_backend,
                               run_identity=run_identity, **workflow_options).run()
        check("after_workflow")
        if file_digest(original) != source_hash:
            raise ValueError("Input changed during document execution")
        _verify_training_batch(args, original)
        _verify_code_snapshot(args)
        if vision_backend is not None:
            vision_backend.verify()
        if vision_factory is not None:
            vision_factory.verify()
        publish_options = {"deadline_guard": check} if total_timeout is not None else {}
        digest = _publish(result, directory, output, fingerprint, source_hash, **publish_options)
        _verify_training_batch(args, original)
        outcome.update(status="completed", output=output.name, execution_sha256=digest)
    except Exception as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        failure = {"cli_status": "failed", "run_fingerprint": fingerprint,
                   "source_sha256": source_hash, "error": outcome["error"]}
        receipt = getattr(exc, "receipt", None)
        if receipt is None and isinstance(result, dict):
            receipt = result.get("receipt")
        if isinstance(receipt, dict):
            failure["workflow_receipt"] = receipt
        elif client is not None:
            failure["model_calls"] = copy.deepcopy(getattr(client, "calls", []))
        atomic_json(directory / "attempts" / (uuid4().hex + ".failure.json"), failure)
    finally:
        os.chdir(previous_cwd)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        outcome["seconds"] = time.perf_counter() - started
        atomic_json(directory / "document_summary.json", outcome)
    return outcome


def run_document_agent(args, *, code_root: Path, convert_doc: callable):
    args = copy.copy(args)
    total_timeout = _total_report_timeout(args)
    # Preserve symlink detection before resolving input files.
    for key in ("training_evidence_json", "training_evidence_manifest", "input_dir", "work_dir", "result_dir"):
        value = getattr(args, key, None)
        if value is not None and (Path(value).is_symlink() or (
                key == "training_evidence_manifest" and any(parent.is_symlink() for parent in Path(value).absolute().parents))):
            raise ValueError(f"Symlink {key} is unsupported")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    code_root = code_root.resolve()
    if args.engine != "tongagents" or args.fusion_policy != "model-preserving":
        raise ValueError("document-agent requires tongagents and model-preserving")
    if type(args.document_workers) is not int or not 1 <= args.document_workers <= 8:
        raise ValueError("Document workers must be an integer between 1 and 8")
    if type(args.model_timeout) not in (int, float) or not math.isfinite(args.model_timeout) or args.model_timeout <= 0:
        raise ValueError("Model timeout must be positive and finite")
    settings = _settings(args)
    args.max_model_len, args.chunk_output_tokens = settings.max_model_len, settings.output_tokens
    prepared = _prepare(args)
    if not args.resume and args.result_dir.exists() and any(args.result_dir.iterdir()):
        raise FileExistsError("Use an empty result directory or a content-verified --resume")
    if args.resume and not (args.work_dir / "run_manifest.json").is_file():
        raise ValueError("Resume requires an existing bound run_manifest.json")
    client = model_identity = None
    if getattr(args, "document_vision", "none") == "own-model":
        client, model_identity = _model_identity(args)
        args.document_model_identity = copy.deepcopy(model_identity)
    image_label = "Own-model vision" if getattr(args, "document_vision", "none") == "own-model" else "CPU OCR"
    try:
        _, _, vision_identity = _make_vision_backend(args)
    except Exception as exc:
        if args.check:
            print(f"{image_label} unavailable: {type(exc).__name__}: {exc}", flush=True)
            if image_label == "CPU OCR":
                print("; ".join(_OCR_LIMITATIONS), flush=True)
        raise
    args.document_vision_identity = vision_identity
    if args.check:
        print(f"{image_label} {vision_identity['status']}: {vision_identity['backend']}", flush=True)
        print("; ".join(vision_identity.get("limitations", [])), flush=True)
    if client is None:
        client, model_identity = _model_identity(args)
        args.document_model_identity = copy.deepcopy(model_identity)
    _, workflow_type = _workflow_types()
    workflow_module = sys.modules.get(workflow_type.__module__)
    package_versions = {name: importlib.metadata.version(name) for name in ("tongagents", "python-docx")}
    contest = code_root / "src/bridge_evidence/contest"
    # Including helpers binds prompt/tool code as well as parser/export behavior.
    code_files = [code_root / "run.py", *sorted(contest.glob("*.py"))]
    sources = {str(original): file_digest(original) for _, _, _, original in prepared}
    for _, manifest, _, original in prepared:
        if manifest is not None and manifest["source_json_sha256"] != sources[str(original)]:
            raise ValueError("Training evidence changed before binding run identity")
    _verify_training_batch(args)
    worker_count = min(args.document_workers, len(prepared))
    identity = {"schema_version": 1, "workflow": "document-agent", "cli_version": VERSION,
                "versions": {key.lower(): getattr(workflow_module, key, None)
                             for key in ("VERSION", "PROMPT_VERSION", "REVIEW_VERSION")},
                "code": {str(p.relative_to(code_root)): file_digest(p) for p in code_files},
                "sources": sources, "settings": asdict(settings), "model_identity": model_identity,
                "vision_backend": vision_identity,
                "served_model": args.served_model, "endpoint": client.base_url,
                "model_catalog": client.model_catalog, "packages": package_versions,
                "model_timeout": args.model_timeout, "engine": args.engine, "fusion_policy": args.fusion_policy,
                "document_workers": args.document_workers, "effective_document_workers": worker_count,
                "document_worker_backend": "spawn-processes", "max_tasks_per_child": 1,
                "result_dir": str(args.result_dir), "chat_tokenizer_hashes": getattr(args, "chat_tokenizer_hashes", None)}
    if total_timeout is not None:
        identity["document_total_report_timeout"] = total_timeout
    if getattr(args, "document_training_batch_identity", None) is not None:
        identity["training_evidence_batch"] = args.document_training_batch_identity
    if args.check:
        if "training_evidence_batch" in identity:
            print(f"Explicit training text diagnostic: {len(prepared)} reports; original images/layout are not covered.", flush=True)
        print("Document preflight passed; real SDK/model execution is not yet verified.", flush=True)
        return None
    fingerprint = bind_work_directory(args.work_dir, identity, resume=args.resume)
    args.document_code_root, args.document_code_hashes = code_root, identity["code"]
    args.result_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = {name + ".docx" for name, _, _, _ in prepared}
    unexpected = [p.name for p in args.result_dir.iterdir()
                  if p.name not in expected_outputs | {"infer_time.json"} and not p.name.startswith(".pending-")]
    if unexpected:
        raise ValueError(f"Result directory has untracked files: {unexpected}")
    attempt_id = uuid4().hex
    previous_seconds = 0.0
    previous_attempts = {p.name.removesuffix(".preflight.json")
                         for p in (args.work_dir / "attempts").glob("*.preflight.json")}
    timed_attempts = set()
    for path in (args.work_dir / "attempts").glob("*.summary.json"):
        previous = _load_json(path)
        if previous.get("run_fingerprint") != fingerprint:
            raise ValueError("Attempt history has a different run identity")
        elapsed = previous.get("seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid attempt wall time")
        previous_seconds += elapsed
        timed_attempts.add(path.name.removesuffix(".summary.json"))
    unknown_attempts = sorted(previous_attempts - timed_attempts)
    atomic_json(args.work_dir / "attempts" / (attempt_id + ".preflight.json"),
                {"run_fingerprint": fingerprint, "model_calls": client.calls})
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    outcomes = {}
    jobs = [(name, manifest, source, original, args, settings, fingerprint,
             sources[str(original)], convert_doc) for name, manifest, source, original in prepared]
    interrupted = None
    try:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("spawn"),
                                 max_tasks_per_child=1) as pool:
            futures = {}
            for job in jobs:
                try:
                    futures[pool.submit(_run_document, *job)] = job[0]
                except Exception as exc:
                    outcomes[job[0]] = {"name": job[0], "status": "failed", "seconds": None,
                                        "error": f"Worker submission failed: {type(exc).__name__}: {exc}"}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    outcomes[name] = future.result()
                except Exception as exc:
                    outcomes[name] = {"name": name, "status": "failed", "seconds": None,
                                      "error": f"Worker failed: {type(exc).__name__}: {exc}"}
        try:
            _verify_training_batch(args)
        except Exception as exc:
            # Existing artifacts remain available as evidence, but changed
            # batch inputs cannot produce a successful throughput result.
            for row in outcomes.values():
                if row["status"] == "completed":
                    row.update(status="failed", reused=False, completed_before_input_revalidation=True,
                               error=f"Batch input identity changed: {type(exc).__name__}: {exc}")
            raise
    except BaseException as exc:
        interrupted = exc
    finally:
        for name, _, _, _ in prepared:
            if name not in outcomes:
                outcomes[name] = {"name": name, "status": "unfinished", "seconds": None,
                                  "error": f"Batch interrupted: {type(interrupted).__name__}"}
        seconds = time.perf_counter() - started
        documents = [outcomes[name] for name, _, _, _ in prepared]
        completed = sum(d["status"] == "completed" for d in documents)
        reused = sum(d.get("reused", False) for d in documents)
        failed = sum(d["status"] == "failed" for d in documents)
        unfinished = len(documents) - completed - failed
        cumulative_seconds = previous_seconds + seconds
        summary = {"workflow": "document-agent", "run_fingerprint": fingerprint, "attempt_id": attempt_id,
                   "started_at": started_at, "reports": len(documents), "documents": documents,
                   "completed_reports": completed, "newly_completed_reports": completed - reused,
                   "reused_reports": reused, "failed_reports": failed, "unfinished_reports": unfinished,
                   "seconds": seconds, "reports_per_hour": (completed - reused) * 3600 / seconds if seconds > 0 else None,
                   "completed_reports_per_hour": (completed - reused) * 3600 / seconds if seconds > 0 else None,
                   "cumulative_seconds": cumulative_seconds,
                   "cumulative_timing_complete": not unknown_attempts,
                   "attempts_with_unknown_wall_time": unknown_attempts,
                   "cumulative_reports_per_hour": completed * 3600 / cumulative_seconds
                       if cumulative_seconds > 0 and not unknown_attempts else None,
                   "document_workers": args.document_workers, "effective_document_workers": worker_count,
                   "document_worker_backend": "spawn-processes", "submission_ready": False,
                   "failures": [d for d in documents if d["status"] != "completed"],
                   "timing_scope": "current attempt wall time including worker startup/shutdown, resume verification, conversion, workflow retries and DOCX publication; reused reports excluded from throughput"}
        if "training_evidence_batch" in identity:
            summary["input_scope"] = identity["training_evidence_batch"]["scope"]
            summary["input_limitations"] = identity["training_evidence_batch"]["limitations"]
            summary["training_evidence_manifest_sha256"] = identity["training_evidence_batch"]["manifest_sha256"]
        for d in documents:
            atomic_json(args.work_dir / "documents" / d["name"] / "document_summary.json", d)
        atomic_json(args.work_dir / "attempts" / (attempt_id + ".summary.json"), summary)
        atomic_json(args.work_dir / "summary.json", summary)
        atomic_json(args.result_dir / "infer_time.json", {"infer_time": math.ceil(cumulative_seconds),
                    "timing_complete": not unknown_attempts,
                    "timing_scope": "recorded attempt wall times; a lower bound if timing_complete is false"})
    if interrupted is not None:
        raise interrupted
    if failed or unfinished:
        raise RuntimeError(f"{failed} document-agent report(s) failed, {unfinished} unfinished; see work summary")
    return summary
