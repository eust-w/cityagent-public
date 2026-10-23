#!/usr/bin/env python3
"""Measure existing DOC conversion cache reuse with real, hash-pinned reports.

Three independent cold/hit pairs retain every report, source binding, manifest
and complete OPC part. No model, SDK, label or test-set input is used. This is a
repeated-preparation diagnostic, not an inference or end-to-end recovery trial.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cache_preparation_helpers", ROOT / "scripts/benchmark_document_preparation.py")
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)
SCHEMA = "cityagent-conversion-cache-diagnostic-v1"
FORBIDDEN_MODULES = {"torch", "transformers", "vllm", "tongagents", "openai", "sglang"}
CODE_PATHS = ("scripts/benchmark_conversion_cache.py", "scripts/benchmark_document_preparation.py",
              "run.py", "src/bridge_evidence/contest/evidence_manifest.py")


def install_process_audit(path):
    """Persist actual Python subprocess launch events; never record environments."""
    path = Path(path)

    def audit(event, args):
        if event != "subprocess.Popen":
            return
        executable, arguments, cwd, _ = args
        argv = [os.fsdecode(item) for item in arguments] if isinstance(arguments, (list, tuple)) else [str(arguments)]
        office = Path(os.fsdecode(executable)).name == "soffice"
        kind = ("version" if office and "--version" in argv else
                "convert" if office and "--convert-to" in argv else "other")
        record = {"event": event, "kind": kind, "argv": argv, "monotonic": time.perf_counter(),
                  "cwd": os.fsdecode(cwd) if cwd else None}
        with path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    sys.addaudithook(audit)


def subprocess_counts(directory):
    path = Path(directory) / "subprocess-events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return {kind: sum(event["kind"] == kind for event in events) for kind in ("version", "convert", "other")}


def prepare_worker(request_path):
    request = json.loads(Path(request_path).read_text())
    directory, source = Path(request["directory"]), Path(request["source"]["path"])
    started = time.perf_counter()
    record = {"source": request["source"], "status": "failed", "model_requests": 0}
    try:
        prep.install_cpu_guard()
        install_process_audit(directory / "subprocess-events.jsonl")
        if prep.digest(source) != request["source"]["sha256"]:
            raise ValueError("Source changed before preparation")
        spec = importlib.util.spec_from_file_location("cache_contest_entry", ROOT / "run.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        from bridge_evidence.contest.evidence_manifest import build_manifest

        stage = time.perf_counter()
        target = (entry.convert_doc_to_docx(source, Path(request["cache_directory"]))
                  if source.suffix.lower() == ".doc" else source)
        record["conversion_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        binding = entry.bind_source_evidence(source, target)
        record["source_binding_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        manifest = build_manifest(target)
        record["parsing_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        parts = prep.package_part_hashes(target)
        target_sha = prep.digest(target)
        if (binding["source_sha256"] != request["source"]["sha256"]
                or binding["source_sha256"] != prep.digest(source)
                or binding["prepared_docx_sha256"] != target_sha
                or manifest["document_sha256"] != target_sha):
            raise ValueError("Source or converted document changed during preparation")
        content = prep.content_digest(manifest, request["source"]["sha256"], parts)
        record["integrity_seconds"] = time.perf_counter() - stage
        prep.save(directory / "package-parts.json", parts)
        prep.save(directory / "source-manifest.json", manifest)
        prep.save(directory / "source-binding.json", binding)
        record.update(status="success", target_sha256=target_sha, source_binding=binding,
                      source_binding_sha256=prep.json_digest({key: value for key, value in binding.items()
                                                            if key != "prepared_docx_path"}),
                      content_sha256=content, package_parts_sha256=prep.json_digest(parts),
                      package_part_count=len(parts), manifest_sha256=prep.digest(directory / "source-manifest.json"),
                      counts={key: len(manifest[key]) for key in ("units", "tables", "images", "unverified_content")},
                      parser_version=manifest["parser_version"],
                      source_inventory={"visual_or_native_evidence_required": bool(manifest["images"] or manifest["unverified_content"]),
                                        "whole_original_report_verified": False,
                                        "publication_eligible_from_this_cpu_run": False},
                      loaded_model_or_sdk_modules=sorted(name for name in sys.modules
                                                       if name.split(".")[0] in FORBIDDEN_MODULES))
    except Exception as error:  # noqa: BLE001 - retain every real worker failure in the denominator
        record.update(error_type=type(error).__name__, error=str(error)[:2000])
    record["worker_seconds"] = time.perf_counter() - started
    record["subprocess_counts"] = subprocess_counts(directory)
    prep.save(directory / "receipt.json", record)
    return 0 if record["status"] == "success" else 2


def run_report(source, directory, cache_directory, timeout):
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    prep.save(request, {"source": source, "directory": str(directory), "cache_directory": str(cache_directory)})
    started = time.perf_counter()
    process = None
    try:
        with (directory / "worker.log").open("wb") as log:
            process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--worker-request", str(request)],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                record = {"source": source, "status": "failed", "error_type": "PreparationTimeout", "model_requests": 0}
            else:
                receipt = directory / "receipt.json"
                record = json.loads(receipt.read_text()) if receipt.exists() else {
                    "source": source, "status": "failed", "error_type": "WorkerExitedWithoutReceipt", "returncode": code,
                    "model_requests": 0}
                if not isinstance(record, dict) or record.get("status") not in {"success", "failed"}:
                    raise ValueError("Malformed worker receipt")
                if code and record["status"] == "success":
                    record.update(status="failed", error_type="InconsistentWorkerExit", returncode=code)
        record["subprocess_counts"] = subprocess_counts(directory)
    except Exception as error:  # noqa: BLE001 - retain launch/read/audit failures instead of shortening the batch
        record = {"source": source, "status": "failed", "error_type": type(error).__name__,
                  "error": str(error)[:2000], "model_requests": 0,
                  "subprocess_counts": {"version": 0, "convert": 0, "other": 0}, "audit_complete": False}
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    record["wall_seconds"] = time.perf_counter() - started
    prep.save(directory / "observed-receipt.json", record)
    return record


def compare_pair(cold, hit):
    errors = []
    if cold["source"] != hit["source"]:
        errors.append("different_input")
    if cold["status"] != "success" or hit["status"] != "success":
        errors.append("failed_preparation")
    else:
        for key in ("target_sha256", "content_sha256", "package_parts_sha256", "source_binding_sha256"):
            if cold[key] != hit[key]:
                errors.append(key)
    doc = Path(cold["source"]["path"]).suffix.lower() == ".doc"
    for phase, record in (("cold", cold), ("hit", hit)):
        expected = {"version": int(doc), "convert": int(doc and phase == "cold"), "other": 0}
        if record["subprocess_counts"] != expected:
            errors.append(phase + "_process_counts")
    return {"source_sha256": cold["source"]["sha256"], "qualified": not errors, "errors": errors}


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def summarize(rounds, denominator):
    by_pair = {}
    arms = {}
    for item in rounds:
        by_pair.setdefault(item["pair"], {})[item["phase"]] = item
    comparisons = []
    for pair, phases in by_pair.items():
        if any(len(phases[phase]["reports"]) != denominator for phase in ("cold", "hit")):
            raise ValueError("Changed report denominator")
        comparisons.extend({"pair": pair, **compare_pair(cold, hit)} for cold, hit in
                           zip(phases["cold"]["reports"], phases["hit"]["reports"], strict=True))
    for phase in ("cold", "hit"):
        selected = [item for item in rounds if item["phase"] == phase]
        records = [record for item in selected for record in item["reports"]]
        values = [record["wall_seconds"] for record in records]
        arms[phase] = {"attempted_reports": len(records), "failures": sum(record["status"] != "success" for record in records),
                       "batch_wall_seconds": [item["wall_seconds"] for item in selected],
                       "median_batch_wall_seconds": statistics.median(item["wall_seconds"] for item in selected),
                       "report_service_p50_seconds": quantile(values, .5), "report_service_p95_seconds": quantile(values, .95),
                       "subprocess_counts": {kind: sum(record["subprocess_counts"][kind] for record in records)
                                             for kind in ("version", "convert", "other")}}
    qualified = all(item["qualified"] for item in comparisons)
    reduction = 1 - arms["hit"]["median_batch_wall_seconds"] / arms["cold"]["median_batch_wall_seconds"]
    return {"schema": SCHEMA, "paired_reports": len(comparisons), "arms": arms,
            "failures": {key: arm["failures"] for key, arm in arms.items()}, "comparisons": comparisons,
            "all_pairs_qualified": qualified, "observed_preparation_wall_reduction": reduction,
            "qualified_preparation_wall_reduction": reduction if qualified else None,
            "quantile_method": "R7 linear interpolation at (n-1)*p; per-report service wall after admission",
            "inference_quality_or_speed_claim": False, "end_to_end_recovery_claim": False, "model_requests": 0}


def public_results(identity, rounds, summary):
    environment = {key: identity[key] for key in ("python", "platform", "processor", "cpu_count", "libraries")}
    phases = []
    allowed = {"status", "error_type", "target_sha256", "source_binding_sha256", "content_sha256", "package_parts_sha256",
               "package_part_count", "counts", "source_inventory", "parser_version", "loaded_model_or_sdk_modules",
               "subprocess_counts", "wall_seconds", "worker_seconds", "conversion_seconds", "source_binding_seconds",
               "parsing_seconds", "integrity_seconds", "model_requests"}
    for item in rounds:
        reports = [{"source_sha256": record["source"]["sha256"], "source_format": Path(record["source"]["path"]).suffix,
                    **{key: value for key, value in record.items() if key in allowed}} for record in item["reports"]]
        phases.append({**{key: value for key, value in item.items() if key != "reports"}, "reports": reports})
    return {"schema": SCHEMA, "code": identity["code"], "code_unchanged": identity["code_unchanged"],
            "environment": environment, "manifest_file_sha256": identity["manifest_file_sha256"],
            "equality_policy": prep.EQUALITY_POLICY, "repeats": identity["repeats"],
            "cache_policy": identity["cache_policy"], "scope": identity["scope"],
            "converter_versions": sorted({record["source_binding"]["conversion"]["converter"]
                for item in rounds for record in item["reports"] if record.get("source_binding", {}).get("conversion")}),
            "rounds": phases, "summary": summary}


def benchmark(manifest_path, output_dir, repeats=3, timeout=60):
    if not 3 <= repeats <= 10 or not 1 <= timeout <= 300:
        raise ValueError("Use 3..10 repeats and a 1..300 second per-report limit")
    inputs = prep.load_inputs(manifest_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    identity = {"schema": SCHEMA, "inputs": inputs, "manifest_file_sha256": prep.digest(manifest_path),
                "repeats": repeats, "report_timeout_seconds": timeout, "python": sys.version,
                "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
                "libraries": {name: importlib.metadata.version(name) for name in ("python-docx", "lxml")},
                "code": {path: prep.digest(ROOT / path) for path in CODE_PATHS},
                "cache_policy": "one independent empty conversion cache per report and pair; cold then same-cache hit; sequential; fresh worker each preparation; OS cache not flushed",
                "scope": "local repeated conversion, actual source binding, parsing and integrity diagnostic; no inference, visual interpretation or end-to-end recovery"}
    for path in CODE_PATHS:
        target = output_dir / "executed-source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
        if prep.digest(target) != identity["code"][path]:
            raise ValueError("Source changed while saving execution snapshot")
    prep.save(output_dir / "identity.json", identity)
    rounds = []
    for pair in range(1, repeats + 1):
        for phase in ("cold", "hit"):
            started = time.perf_counter()
            records = [run_report(source, output_dir / f"pair-{pair}/{phase}/report-{index:03d}",
                                  output_dir / f"pair-{pair}/caches/report-{index:03d}", timeout)
                       for index, source in enumerate(inputs["reports"])]
            rounds.append({"pair": pair, "phase": phase, "denominator": len(inputs["reports"]),
                           "wall_seconds": time.perf_counter() - started, "reports": records})
            prep.save(output_dir / "rounds.json", rounds)
            print(json.dumps({key: value for key, value in rounds[-1].items() if key != "reports"}), flush=True)
    identity["code_unchanged"] = all(prep.digest(ROOT / path) == value for path, value in identity["code"].items())
    prep.save(output_dir / "identity.json", identity)
    summary = summarize(rounds, len(inputs["reports"]))
    if not identity["code_unchanged"]:
        summary.update(all_pairs_qualified=False, qualified_preparation_wall_reduction=None, invalid_reason="code_changed")
    prep.save(output_dir / "summary.json", summary)
    prep.save(output_dir / "public-results.json", public_results(identity, rounds, summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report-timeout", type=float, default=60)
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_request:
        return prepare_worker(args.worker_request)
    if not args.manifest or not args.output_dir:
        parser.error("--manifest and --output-dir are required")
    result = benchmark(args.manifest, args.output_dir, args.repeats, args.report_timeout)
    return 0 if result["all_pairs_qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
