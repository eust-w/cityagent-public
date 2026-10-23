#!/usr/bin/env python3
"""Real local conversion/manifest benchmark; never loads a model or calls an SDK.

Only explicit hash-pinned training inputs are accepted. Every report/round has a
new process, converter profile and cache. Failed work remains in the denominator.
This measures CPU preparation, not extraction F1, visual understanding or PPU speed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cityagent-cpu-preparation-benchmark-v2"
EQUALITY_POLICY = "complete-uncompressed-opc-part-bytes-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def load_inputs(path):
    value = json.loads(Path(path).read_text())
    if value.get("scope") != "official-training" or not value.get("reports"):
        raise ValueError("Explicit nonempty official-training report manifest required")
    root = Path(value["training_root"]).resolve(strict=True)
    forbidden = {"初赛测试集", "线上测试集", "测试集", "赛题一_训练集标签", "labels", "test"}
    if any(part in forbidden for part in root.parts):
        raise ValueError("Training root must not be a test or label directory")
    reports, seen = [], set()
    for item in value["reports"]:
        source = Path(item["path"]).resolve(strict=True)
        source.relative_to(root)
        if (source.suffix.lower() not in {".doc", ".docx"}
                or any(part in forbidden for part in source.parts)
                or source in seen or digest(source) != item["sha256"]):
            raise ValueError("Duplicate, unsupported or changed training report")
        seen.add(source)
        reports.append({"path": str(source), "sha256": item["sha256"]})
    return {"scope": value["scope"], "training_root": str(root), "reports": reports}


def package_part_hashes(path):
    """Hash all package files, including media, embeddings and relationships.

    ZIP timestamps/compression do not change file contents. No XML attributes or
    timestamps inside parts are normalized: even converter-generated ObjectID
    changes must fail this strict comparison rather than silently losing proof.
    """
    with zipfile.ZipFile(path) as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate ZIP part names cannot establish byte identity")
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(names)}


def content_digest(manifest, source_sha256, package_parts):
    # Retain every evidence unit/cell, location, image and unverified object.
    # Only the run-local root filename and possibly timestamp-sensitive converted
    # ZIP identity are omitted; every uncompressed package byte is bound below.
    return json_digest({"original_source_sha256": source_sha256,
                        "equality_policy": EQUALITY_POLICY, "package_parts": package_parts,
                        "manifest": {key: value for key, value in manifest.items()
                                     if key not in {"source_path", "document_sha256"}}})


def install_cpu_guard():
    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise RuntimeError("Network access is forbidden in CPU preparation")
        if event == "import" and args[0].split(".")[0] in {
            "torch", "transformers", "vllm", "tongagents", "openai", "sglang"
        }:
            raise RuntimeError("Model/SDK import is forbidden in CPU preparation")
    sys.addaudithook(audit)


def prepare_worker(request_path):
    request = json.loads(Path(request_path).read_text())
    directory = Path(request["directory"])
    source = Path(request["source"]["path"])
    started = time.perf_counter()
    record = {"source": request["source"], "status": "failed", "model_requests": 0}
    try:
        install_cpu_guard()
        if digest(source) != request["source"]["sha256"]:
            raise ValueError("Source changed before preparation")
        spec = importlib.util.spec_from_file_location("cpu_contest_entry", ROOT / "run.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        from bridge_evidence.contest.evidence_manifest import build_manifest

        convert_start = time.perf_counter()
        target = (entry.convert_doc_to_docx(source, directory / "conversion")
                  if source.suffix.lower() == ".doc" else source)
        conversion_seconds = time.perf_counter() - convert_start
        parse_start = time.perf_counter()
        manifest = build_manifest(target)
        parsing_seconds = time.perf_counter() - parse_start
        parts = package_part_hashes(target)
        save(directory / "package-parts.json", parts)
        save(directory / "source-manifest.json", manifest)
        if digest(source) != request["source"]["sha256"]:
            raise ValueError("Source changed during preparation")
        record.update(status="success", conversion_seconds=conversion_seconds,
                      parsing_seconds=parsing_seconds, target_sha256=digest(target),
                      manifest_sha256=digest(directory / "source-manifest.json"),
                      content_sha256=content_digest(manifest, request["source"]["sha256"], parts),
                      package_parts_sha256=json_digest(parts), package_part_count=len(parts),
                      equality_policy=EQUALITY_POLICY,
                      counts={key: len(manifest[key]) for key in
                              ("units", "tables", "images", "unverified_content")},
                      source_inventory={
                          "visual_or_native_evidence_required": bool(manifest["images"] or manifest["unverified_content"]),
                          "whole_original_report_verified": False,
                          "publication_eligible_from_this_cpu_run": False,
                          "limitations": manifest["limitations"]},
                      parser_version=manifest["parser_version"],
                      loaded_model_or_sdk_modules=[name for name in sys.modules if name.split(".")[0]
                          in {"torch", "transformers", "vllm", "tongagents", "openai", "sglang"}])
    except Exception as error:  # noqa: BLE001 - record any real preparation failure in the denominator
        record.update(error_type=type(error).__name__, error=str(error)[:2000])
    record["worker_seconds"] = time.perf_counter() - started
    save(directory / "receipt.json", record)
    return 0 if record["status"] == "success" else 2


def run_report(source, directory, timeout):
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    save(request, {"source": source, "directory": str(directory)})
    started = time.perf_counter()
    with (directory / "worker.log").open("wb") as log:
        process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                                    "--worker-request", str(request)], stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            record = {"source": source, "status": "failed", "error_type": "PreparationTimeout",
                      "error": "Report and its converter exceeded the fixed time limit", "model_requests": 0}
        else:
            receipt = directory / "receipt.json"
            record = json.loads(receipt.read_text()) if receipt.exists() else {
                "source": source, "status": "failed", "error_type": "WorkerExitedWithoutReceipt",
                "returncode": code, "model_requests": 0}
            if code and record.get("status") == "success":
                record.update(status="failed", error_type="InconsistentWorkerExit", returncode=code)
    record["wall_seconds"] = time.perf_counter() - started
    save(directory / "observed-receipt.json", record)
    return record


def summarize(rounds, denominator):
    reference = {}
    inventories = {}
    mismatches = []
    for item in rounds:
        for record in item["reports"]:
            if record["status"] != "success":
                continue
            source = record["source"]["sha256"]
            if "counts" in record:
                inventories.setdefault(source, {"source": record["source"], "counts": record["counts"],
                                               "source_inventory": record["source_inventory"]})
            prior = reference.setdefault(source, record["content_sha256"])
            if prior != record["content_sha256"]:
                mismatches.append({"round": item["round"], "workers": item["workers"], "source_sha256": source})
    arms = {}
    for workers in (1, 2):
        selected = [item for item in rounds if item["workers"] == workers]
        arms[str(workers)] = {"rounds": len(selected), "attempted_reports": denominator * len(selected),
            "failures": sum(record["status"] != "success" for item in selected for record in item["reports"]),
            "batch_wall_seconds": [item["wall_seconds"] for item in selected],
            "median_batch_wall_seconds": statistics.median(item["wall_seconds"] for item in selected)}
    all_success = all(arm["failures"] == 0 for arm in arms.values())
    comparable = all_success and not mismatches
    ratio = arms["1"]["median_batch_wall_seconds"] / arms["2"]["median_batch_wall_seconds"]
    return {"arms": arms, "content_mismatches": mismatches,
            "report_inventories": list(inventories.values()),
            "all_reports_succeeded_with_equal_content": comparable,
            "observed_median_wall_ratio_serial_over_parallel": ratio,
            "validated_cpu_speedup_ratio": ratio if comparable else None,
            "inference_quality_or_speed_claim": False, "model_requests": 0}


def benchmark(manifest_path, output_dir, repeats=3, timeout=60):
    if repeats < 3 or repeats > 10 or not 1 <= timeout <= 300:
        raise ValueError("Use 3..10 repeats and a 1..300 second per-report limit")
    inputs = load_inputs(manifest_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    import shutil
    converter = shutil.which("soffice")
    converter_version = None
    if converter:
        try:
            result = subprocess.run([converter, "--version"], capture_output=True, text=True, timeout=10, check=False)
            converter_version = {"returncode": result.returncode, "text": (result.stdout + result.stderr).strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            converter_version = {"error_type": type(error).__name__}
    identity = {"schema": SCHEMA, "equality_policy": EQUALITY_POLICY,
                "inputs": inputs, "manifest_file_sha256": digest(manifest_path),
                "repeats": repeats, "report_timeout_seconds": timeout,
                "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                "cpu_count": os.cpu_count(), "converter": converter, "converter_version": converter_version,
                "libraries": {name: importlib.metadata.version(name) for name in ("python-docx", "lxml")},
                "code": {path: digest(ROOT / path) for path in ("run.py", "src/bridge_evidence/contest/evidence_manifest.py",
                                                              "scripts/benchmark_document_preparation.py")},
                "cache_policy": "fresh private conversion cache per report and arm; OS cache not flushed",
                "scope": "local CPU conversion plus evidence inventory; images not interpreted; no model/SDK calls"}
    save(output_dir / "identity.json", identity)
    rounds = []
    for repeat in range(repeats):
        # Alternate arm order to reduce systematic filesystem-cache/order bias.
        for workers in ((1, 2) if repeat % 2 == 0 else (2, 1)):
            directory = output_dir / f"round-{repeat + 1}-workers-{workers}"
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_report, source, directory / f"report-{index:03d}", timeout)
                           for index, source in enumerate(inputs["reports"])]
                records = [future.result() for future in futures]
            rounds.append({"round": repeat + 1, "workers": workers, "wall_seconds": time.perf_counter() - started,
                           "denominator": len(inputs["reports"]), "reports": records})
            save(output_dir / "rounds.json", rounds)
            print(json.dumps({key: rounds[-1][key] for key in ("round", "workers", "wall_seconds", "denominator")}), flush=True)
    result = {"schema": SCHEMA, "equality_policy": EQUALITY_POLICY,
              **summarize(rounds, len(inputs["reports"]))}
    save(output_dir / "summary.json", result)
    return result


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
    return 0 if result["all_reports_succeeded_with_equal_content"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
