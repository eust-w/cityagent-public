"""Explicit on-host contest inference; originals and request traces stay on host."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from uuid import uuid4

from .artifact_publish import publish_no_replace
from .document_multiagent import MultiAgentSettings
from .document_multiagent_batch import (BatchSettings, RUNTIME_SCHEMA, file_sha,
    prepare_batch, run_batch, verify_current_delivery)
from .document_multiagent_cli import add_execution_arguments, model_configuration, validate_execution_arguments
from .document_batch_quiescence import BatchServiceQuiescence
from .evidence_manifest import build_manifest
from .mineru_preparation import prepare_document
from .run_state import atomic_json


def source_files(root):
    root = Path(root).resolve(strict=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and
        p.suffix.lower() in {".doc", ".docx"} and not p.name.startswith("~$"))
    if not files:
        raise ValueError("No DOC/DOCX in the explicitly selected input directory")
    names = set()
    for p in files:
        if not p.resolve().is_relative_to(root):
            raise ValueError("Input symlink leaves the selected directory")
        name = (p.stem + ".docx").casefold()
        if name in names:
            raise ValueError("Input basenames collide in the flat result directory")
        names.add(name)
    return files


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha(path)}


def convert_source(source, directory, soffice):
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / (source.stem + ".docx")
    if source.suffix.lower() == ".docx":
        shutil.copyfile(source, target)
        return target
    executable = shutil.which(soffice)
    if executable is None:
        raise ValueError("LibreOffice required for legacy DOC conversion")
    profile = (directory / "lo-profile").resolve().as_uri()
    command = [executable, "-env:UserInstallation=" + profile, "--headless", "--convert-to", "docx",
               "--outdir", str(directory), str(source)]
    process = subprocess.run(command, capture_output=True, timeout=300)
    (directory / "conversion.stdout.log").write_bytes(process.stdout)
    (directory / "conversion.stderr.log").write_bytes(process.stderr)
    if process.returncode or not target.is_file() or not target.stat().st_size:
        raise ValueError("DOC conversion failed; diagnostics retained on host")
    return target


def prepare_inputs(input_dir, directory, scope, parser_python, soffice="soffice", *, text_table_projection=False):
    if scope not in {"official-training", "official-test"}:
        raise ValueError("Explicit inference scope required")
    files = source_files(input_dir)
    directory.mkdir(parents=True, exist_ok=False)
    inventory = [{"name": p.name, "output_name": p.stem + ".docx", **binding(p)} for p in files]
    atomic_json(directory / "original-inventory.json", {"scope": scope, "files": inventory})
    documents = []
    for number, (source, original) in enumerate(zip(files, inventory), 1):
        item = directory / f"document-{number:03d}"
        item.mkdir()
        converted = convert_source(source, item / "converted", soffice)
        parse_dir = item / "parsed"
        prepare_document(converted, parse_dir, source_root=converted.parent,
            input_scope=scope, python_executable=parser_python, expected_sha256=file_sha(converted))
        full = build_manifest(converted)
        inference_manifest = deepcopy(full)
        excluded = {}
        if text_table_projection:
            excluded = {key: inference_manifest.get(key, []) for key in ("images", "unverified_content")}
            inference_manifest.update(images=[], unverified_content=[])
        source_scope = {"mode": "text_table_projection" if text_table_projection else "full_document",
            "projection_explicitly_requested": text_table_projection,
            "excluded_images": len(excluded.get("images", [])),
            "excluded_unverified_content": len(excluded.get("unverified_content", [])),
            "whole_original_report_verified": False}
        atomic_json(item / "full-source-inventory.json", full)
        manifest_path = item / ("source-projection.json" if text_table_projection else "source-manifest.json")
        atomic_json(manifest_path, inference_manifest)
        atomic_json(item / "excluded-content.json", excluded)
        atomic_json(item / "source-scope.json", source_scope)
        if file_sha(source) != original["sha256"]:
            raise ValueError("Original input changed during preparation")
        documents.append({"document_id": f"document-{number:03d}", "title": source.name,
            "scope": scope, "runtime_conversion_required": False,
            "source_scope": source_scope,
            "evaluation_outputs_are_not_model_inputs": True,
            "official_original": binding(source), "source_docx": binding(converted),
            "manifest": binding(manifest_path),
            "materials": binding(parse_dir / "materials.json"),
            "full_source_inventory": binding(item / "full-source-inventory.json"),
            "excluded_content": binding(item / "excluded-content.json")})
        atomic_json(directory / "preparation-progress.json", {"prepared": number, "total": len(files)})
    runtime = directory / "batch.runtime.json"
    atomic_json(runtime, {"schema": RUNTIME_SCHEMA, "status": "ready", "scope": scope,
        "test_inference_only": scope == "official-test", "text_table_projection": text_table_projection,
        "documents": documents})
    return runtime


def validate_source_scope(prepared, *, text_table_projection=False):
    """Check actual pinned bytes, including older prepared-batch manifests.

    A label in a runtime JSON cannot opt the operator into excluding evidence.
    Full source preservation is the default; the legacy text/table experiment
    requires the explicit CLI switch on every invocation, also when reused.
    """
    scopes = {}
    for document in prepared["documents"]:
        full = json.loads(Path(document["full_source_inventory"]["path"]).read_bytes())
        actual = json.loads(Path(document["manifest"]["path"]).read_bytes())
        excluded = json.loads(Path(document["excluded_content"]["path"]).read_bytes())
        projection = {**full, "images": [], "unverified_content": []}
        is_projection = actual != full
        if is_projection and (not text_table_projection or actual != projection):
            raise ValueError("Prepared manifest excludes or changes original evidence; only --text-table-projection "
                             "explicitly permits excluding images and unverified content")
        expected = {key: full.get(key, []) for key in ("images", "unverified_content")} if is_projection else {}
        if any(excluded.get(key, []) != expected.get(key, []) for key in ("images", "unverified_content")):
            raise ValueError("Excluded-content receipt differs from the actual source projection")
        scopes[document["document_id"]] = {
            "mode": "text_table_projection" if is_projection else "full_document",
            "projection_explicitly_requested": text_table_projection,
            "excluded_images": len(expected.get("images", [])),
            "excluded_unverified_content": len(expected.get("unverified_content", [])),
            "whole_original_report_verified": False}
    return scopes


def publish_report(document, record, result_dir):
    """Publish the exact inspected bytes under the original basename, once."""
    if not record.get("complete"):
        return {"published": False, "reason": record.get("status", "incomplete")}
    if record.get("generation_settlement", {}).get("settled") is not True:
        raise ValueError("Final publication requires settled document generation")
    receipt = verify_current_delivery(record["run_directory"], record["result"],
        manifest_identity_sha256=document["manifest_identity_sha256"])
    if receipt.get("scope", {}).get("submission_format") is not True:
        raise ValueError("Only the inspected clean submission format can be published")
    name = Path(document["official_original"]["path"]).stem + ".docx"
    target = Path(result_dir) / name
    raw = Path(receipt["docx_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["file_sha256"]:
        raise ValueError("Inspected Word changed before publication")
    # Keep the final basename invisible until write, flush and close all
    # succeed. Only clean a temporary file this invocation actually created.
    pending = target.parent / (".pending-" + uuid4().hex + ".docx")
    owns_pending = False
    try:
        stream = pending.open("xb")
        owns_pending = True
        with stream:
            if stream.write(raw) != len(raw):
                raise OSError("Incomplete staged Word write")
            stream.flush()
            os.fsync(stream.fileno())
        if file_sha(pending) != receipt["file_sha256"]:
            raise ValueError("Staged Word changed before publication")
        # All publishers of a given result directory share target-keyed locks,
        # also on filesystems where publication falls back from link to rename.
        lock_dir = target.parent.resolve().parent / ".cityagent-publish-locks"
        publish_no_replace(pending, target, lock_dir)
    finally:
        if owns_pending:
            pending.unlink(missing_ok=True)
    return {"published": True, "output_name": name, "sha256": receipt["file_sha256"]}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--scope", choices=["official-training", "official-test"], required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--parser-python", required=True)
    p.add_argument("--soffice", default="soffice")
    p.add_argument("--prepared-batch", type=Path)
    p.add_argument("--prepared-batch-sha256")
    p.add_argument("--text-table-projection", action="store_true",
                   help="Explicitly exclude images/unverified objects for the text/table experiment; "
                        "this does not verify completeness of the original report")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--document-workers", type=int, default=2)
    p.add_argument("--global-main-model-inflight", type=int, default=4)
    add_execution_arguments(p)
    return p


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    validate_execution_arguments(args, cli)
    started = time.monotonic()
    root = args.input_dir.resolve(strict=True)
    selected = source_files(root)
    if args.work_dir.resolve().is_relative_to(root) or args.result_dir.resolve().is_relative_to(root):
        cli.error("Output and working directories must be outside the immutable source directory")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    if args.prepared_batch:
        if not args.prepared_batch_sha256 or file_sha(args.prepared_batch) != args.prepared_batch_sha256:
            cli.error("Prepared batch requires its exact SHA256")
        runtime = args.prepared_batch
    else:
        runtime = prepare_inputs(root, args.work_dir / "inputs", args.scope, args.parser_python, args.soffice,
                                 text_table_projection=args.text_table_projection)
    settings = MultiAgentSettings(tasks=args.tasks, workers=args.workers,
        max_model_calls=args.max_model_calls, report_timeout=args.report_timeout,
        compression_trigger_tokens=args.compression_trigger_tokens,
        direct_delivery=args.direct_delivery, speed_priority=args.speed_priority,
        section_prompt_variant=args.section_prompt_variant)
    prepared = prepare_batch(runtime, file_sha(runtime), settings=settings)
    source_scopes = validate_source_scope(prepared, text_table_projection=args.text_table_projection)
    if prepared["scope"] != args.scope or {d["official_original"]["path"] for d in prepared["documents"]} != {str(p.resolve()) for p in selected}:
        raise ValueError("Prepared inputs differ from the explicit input directory/scope")
    preparation_seconds = time.monotonic() - started
    atomic_json(args.work_dir / "preflight.json", {"status": "validated", "documents": len(selected),
        "scope": args.scope, "source_scopes": source_scopes,
        "runtime": binding(runtime), "preparation_seconds": preparation_seconds})
    if not args.execute:
        print(json.dumps({"status": "prepared", "documents": len(selected), "model_calls": 0,
                          "source_scopes": source_scopes}))
        return 0
    model = model_configuration(args, cli)
    model["submission_format"] = True
    args.result_dir.mkdir(parents=True, exist_ok=False)
    publications = {}
    def publish(document, record):
        try:
            publications[document["document_id"]] = publish_report(document, record, args.result_dir)
        except Exception as exc:
            publications[document["document_id"]] = {"published": False, "error_type": type(exc).__name__,
                "reason": str(exc)}
        atomic_json(args.work_dir / "publication-progress.json", publications)
    result = run_batch(prepared, args.work_dir / "run", model, settings=settings,
        batch_settings=BatchSettings(args.document_workers, args.global_main_model_inflight, 1),
        quiescence_observer_factory=lambda directory: BatchServiceQuiescence(model, directory / "service-settlement"),
        on_document_complete=publish)
    atomic_json(args.result_dir / "infer_time.json", {"infer_time": math.ceil(result["seconds"])})
    published = sum(r.get("published", False) for r in publications.values())
    summary = {"status": "completed" if result["complete"] and published == len(selected) else "partial_failure",
        "documents": len(selected), "published": published, "preparation_seconds": preparation_seconds,
        "inference_seconds": result["seconds"], "total_seconds": time.monotonic() - started,
        "scope": args.scope, "test_data_exported": False, "summary_model_enabled": bool(model.get("summary")),
        "source_scopes": source_scopes,
        "result_dir": str(args.result_dir.resolve()), "publications": publications}
    atomic_json(args.work_dir / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "documents", "published", "inference_seconds", "total_seconds")}))
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
