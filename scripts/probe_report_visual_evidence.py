#!/usr/bin/env python3
"""Plan all frozen sources, then sample real images using the unchanged model.

All plans, raw responses and source paths stay in the official workspace. This
owns the ORIGINAL CloudBudget lease and never certifies full-report/RSI readiness.
The two CPU preparation subprocesses and the model share one wall-clock cap.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from probe_fixed_model_vision import (
    FIXED_ADAPTER_SHA256,
    FIXED_ALIAS,
    ORIGINAL_BUDGET,
    deadline,
    request_json,
)

from bridge_evidence.contest.backends import AccountedBackend, APIBackend, ModelManifest
from bridge_evidence.contest.long_document import DocumentSnapshot
from bridge_evidence.contest.meta_optimization import CloudBudget
from bridge_evidence.contest.model_identity import load_identity
from bridge_evidence.contest.rsi_runtime import REPORT_DEADLINE, REQUEST_CALL_ID
from bridge_evidence.contest.run_state import atomic_json, file_digest, json_digest
from bridge_evidence.contest.timing import TimingLedger
from bridge_evidence.contest.vision_capability import validate_vision_capability
from bridge_evidence.contest.visual_evidence import (
    execute_visual_jobs,
    plan_visual_jobs,
)

WORKSPACE = Path("/workspace")
APPROVED_SOURCE_ROOT = Path(os.environ.get(
    "CITYAGENT_APPROVED_SOURCE_ROOT", "/workspace/__cityagent_unconfigured_sources__"
))
EXPECTED_DENOMINATOR = 79


def _workspace(path):
    path = Path(path).resolve()
    if path == WORKSPACE.resolve() or not path.is_relative_to(WORKSPACE.resolve()):
        raise ValueError("official /workspace descendant required")
    return path


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def source_policy(args):
    """Resolve an explicit development scope without widening the legacy policy."""
    policy = getattr(args, "model_policy", "fixed-native-lora")
    denominator = getattr(args, "expected_denominator", None)
    source_root = getattr(args, "source_root", None)
    if denominator is not None and (type(denominator) is not int or denominator < 1):
        raise ValueError("expected denominator must be a positive integer")
    if policy == "fixed-native-lora":
        if denominator is not None and denominator != EXPECTED_DENOMINATOR:
            raise ValueError("legacy policy retains the original fixed denominator")
        if source_root is not None and Path(source_root).resolve() != APPROVED_SOURCE_ROOT.resolve():
            raise ValueError("legacy policy retains the original approved source root")
        return policy, EXPECTED_DENOMINATOR, APPROVED_SOURCE_ROOT.resolve()
    if policy != "manifest-locked":
        raise ValueError("unknown model policy")
    if denominator is None or source_root is None:
        raise ValueError("manifest-locked policy requires explicit denominator and source root")
    root, workspace = Path(source_root).resolve(), WORKSPACE.resolve()
    if not root.is_dir() or not (
        root == APPROVED_SOURCE_ROOT.resolve()
        or (root != workspace and root.is_relative_to(workspace))
    ):
        raise ValueError("source root must be original training root or an explicit /workspace development directory")
    return policy, denominator, root


def validate_inputs(args):
    policy, _, _ = source_policy(args)
    for path in (args.manifest, args.model_manifest, args.output_dir, args.budget_ledger):
        _workspace(path)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("fresh independent diagnostic output directory required")
    if (args.budget_ledger.resolve() != ORIGINAL_BUDGET.resolve()
            or not args.budget_ledger.is_file()):
        raise ValueError("existing original cumulative budget ledger required; no new budget")
    with sqlite3.connect(args.budget_ledger.resolve().as_uri() + "?mode=ro", uri=True) as db:
        if [row[1] for row in db.execute("PRAGMA table_info(jobs)")] != [
            "id", "phase", "start", "seconds", "status", "cap",
        ]:
            raise ValueError("invalid existing budget ledger")
    if (not _sha(args.manifest_sha256) or args.workers not in (1, 2)
            or type(args.max_jobs) is not int or not 1 <= args.max_jobs <= 64
            or type(args.limit_seconds) is not int or not 1 <= args.limit_seconds <= 7200
            or type(args.max_tokens) is not int or not 1 <= args.max_tokens <= 4096
            or type(args.request_timeout) not in (int, float)
            or not math.isfinite(args.request_timeout) or not 0 < args.request_timeout <= 180
            or type(getattr(args, "visual_display_transforms", False)) is not bool):
        raise ValueError("invalid bounded diagnostic configuration")
    endpoint = urlsplit(args.api_url)
    if (endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "::1")
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
            or endpoint.path.rstrip("/") != "/v1"):
        raise ValueError("numeric loopback /v1 endpoint required")
    if not args.planning_only and not args.visual_capability:
        raise ValueError("actual fixed-model visual capability receipt required")
    if args.visual_capability:
        _workspace(args.visual_capability)
        if not args.visual_capability.is_file():
            raise ValueError("visual capability receipt missing")
    if bool(args.prepared_identity) != bool(args.prepared_identity_sha256):
        raise ValueError("prepared identity requires an explicitly pinned SHA256")
    if args.prepared_identity:
        _workspace(args.prepared_identity)
        if not args.prepared_identity.is_file() or not _sha(args.prepared_identity_sha256):
            raise ValueError("invalid prepared model identity receipt")
    model = ModelManifest.load(args.model_manifest)
    if policy == "fixed-native-lora":
        if (model.served_model != FIXED_ALIAS or model.adapter is None or model.merged
                or model.adapter_sha256 != FIXED_ADAPTER_SHA256
                or Path(model.base).name != "Qwen3.6-27B"
                or Path(model.tokenizer).resolve() != Path(model.base).resolve()):
            raise ValueError("unchanged Qwen3.6-27B native-LoRA manifest required")
    elif (not isinstance(model.served_model, str)
          or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}", model.served_model)):
        raise ValueError("manifest-locked model requires an explicit served model alias")
    return model


def source_cases(args):
    """Read only manifest metadata and original DOCX bytes, never any gold path."""
    if args.manifest.stat().st_size > 16 * 1024**2:
        raise ValueError("bounded frozen manifest required")
    if file_digest(args.manifest) != args.manifest_sha256:
        raise ValueError("frozen manifest SHA256 changed")
    _, denominator, source_root = source_policy(args)
    manifest = json.loads(args.manifest.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("frozen development manifest must be an object")
    samples = manifest.get("samples")
    if (manifest.get("schema_version") != 2
            or manifest.get("purpose") != "development-regression"
            or not isinstance(samples, list) or len(samples) != denominator
            or type(manifest.get("metric_denominator")) is not int
            or manifest.get("metric_denominator") != denominator):
        raise ValueError(f"all {denominator} frozen development reports are required")
    cases, ids = [], set()
    for row in samples:
        if not isinstance(row, dict) or not isinstance(row.get("report"), str):
            raise ValueError("frozen report metadata and source path required")
        report_id = row.get("id")
        if (not isinstance(report_id, str) or not 0 < len(report_id) <= 256
                or report_id in ids or not _sha(row.get("report_sha256"))):
            raise ValueError("unique frozen report identity required")
        ids.add(report_id)
        path = Path(row["report"]).resolve()
        if (not path.is_relative_to(source_root)
                or path.suffix.lower() != ".docx" or not path.is_file()):
            raise ValueError("approved original development DOCX required")
        # No gold/annotation field is forwarded into preparation or generation.
        cases.append({"id": report_id, "report": str(path),
                      "report_sha256": row["report_sha256"]})
    if args.report_id is not None and args.report_id not in ids:
        raise ValueError("requested report ID is outside the frozen manifest")
    for case in cases:
        if file_digest(Path(case["report"])) != case["report_sha256"]:
            raise ValueError("frozen original report SHA256 changed")
    return cases


def _plan_one(case, model_sha, config, directory):
    """Only CPU/document work; invoked in a parent-owned bounded subprocess."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    result = {"id": case["id"], "report_sha256": case["report_sha256"], "status": "failed"}
    started = time.monotonic()
    try:
        source = Path(case["report"])
        if file_digest(source) != case["report_sha256"]:
            raise ValueError("frozen source changed before planning")
        native = DocumentSnapshot.from_docx(source)
        plan = plan_visual_jobs(source, native, model_sha, config)
        atomic_json(directory / "plan.json", plan)
        reasons = Counter((item["kind"] + ": " + item["reason"]) for item in plan["unresolved"])
        blobs = [ref["blob"] for occurrence in plan["inventory"]["occurrences"]
                 for ref in occurrence["relationships"] if ref.get("blob")]
        result.update(
            status="planned", plan_sha256=plan["plan_sha256"],
            plan_file_sha256=file_digest(directory / "plan.json"),
            supported_unique_jobs=len(plan["jobs"]),
            unsupported_occurrences=len(plan["unresolved"]), unresolved_types=dict(reasons),
            inventory_counts=plan["inventory"]["counts"],
            max_bound_blob_bytes=max((blob["size_bytes"] for blob in blobs), default=0),
            max_supported_image_bytes=max((job["blob"]["size_bytes"] for job in plan["jobs"]), default=0),
            max_supported_pixels=max((job["view"]["width"] * job["view"]["height"]
                                      for job in plan["jobs"]), default=0),
        )
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    result["planning_seconds"] = time.monotonic() - started
    atomic_json(directory / "summary.json", result)
    return result


def _stop_owned(children):
    for process, _, _ in children:
        if process.poll() is None:
            process.terminate()
    for process, _, log in children:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        finally:
            log.close()


def plan_all(cases, model_sha, config, output, *, workers=2, progress=None):
    """Bounded CPU overlap; only these owned children are stopped at the cap."""
    output.mkdir()
    jobs_dir = output / "jobs"
    jobs_dir.mkdir()
    running, results, next_index = [], {}, 0
    try:
        while next_index < len(cases) or running:
            while next_index < len(cases) and len(running) < workers:
                index = next_index
                next_index += 1
                directory = output / json_digest(cases[index]["id"])
                job_path = jobs_dir / f"{index}.json"
                atomic_json(job_path, {"case": cases[index], "model_sha": model_sha,
                                       "config": config, "directory": str(directory)})
                log = (jobs_dir / f"{index}.log").open("xb")
                try:
                    process = subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), "--_plan-job", str(job_path)],
                        stdout=log, stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    log.close()
                    raise
                running.append((process, index, log))
            completed = [item for item in running if item[0].poll() is not None]
            if not completed:
                time.sleep(0.05)
                continue
            for process, index, log in completed:
                log.close()
                running.remove((process, index, log))
                directory = output / json_digest(cases[index]["id"])
                summary_path = directory / "summary.json"
                if process.returncode != 0 or not summary_path.is_file():
                    result = {"id": cases[index]["id"], "status": "failed",
                              "error_type": "PlanningProcessFailure", "returncode": process.returncode}
                else:
                    result = json.loads(summary_path.read_text())
                    if result["id"] != cases[index]["id"]:
                        raise ValueError("planning child returned a different report")
                result["plan_directory"] = str(directory)
                results[index] = result
                if progress:
                    progress([results[i] for i in sorted(results)])
        return [results[i] for i in range(len(cases))]
    finally:
        _stop_owned(running)


class LoopbackBackend(APIBackend):
    def _request(self, route, value=None):
        remaining = REPORT_DEADLINE.get()
        timeout = self.timeout if remaining is None else min(self.timeout, remaining - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("report visual deadline exhausted")
        return request_json(self.url + "/" + route, timeout=timeout, payload=value,
                            call_id=REQUEST_CALL_ID.get())


def _summary(report):
    rows = report["reports"]
    reasons = Counter()
    for row in rows:
        reasons.update(row.get("unresolved_types", {}))
    return {
        "processed_reports": len(rows),
        "planned_reports": sum(row["status"] == "planned" for row in rows),
        "failed_reports": sum(row["status"] != "planned" for row in rows),
        "supported_unique_jobs": sum(row.get("supported_unique_jobs", 0) for row in rows),
        "unsupported_occurrences": sum(row.get("unsupported_occurrences", 0) for row in rows),
        "unresolved_types": dict(reasons),
        "max_bound_blob_bytes": max((row.get("max_bound_blob_bytes", 0) for row in rows), default=0),
        "max_supported_image_bytes": max((row.get("max_supported_image_bytes", 0) for row in rows), default=0),
        "max_supported_pixels": max((row.get("max_supported_pixels", 0) for row in rows), default=0),
    }


def run_probe(args):
    model = validate_inputs(args)
    policy, denominator, source_root = source_policy(args)
    budget = CloudBudget(args.budget_ledger)
    job_id = budget.start("report-visual-evidence-diagnostic", args.limit_seconds)
    started = time.monotonic()
    global_deadline = started + args.limit_seconds
    status, created = "failed", False
    report = {
        "schema_version": 1, "protocol": "report-visual-evidence-diagnostic-v1",
        "purpose": "development-source-planning-and-bounded-image-diagnostic",
        "expected_denominator": denominator, "reports": [],
        "model_policy": policy, "source_root": str(source_root), "endpoint": args.api_url,
        "planning_complete": False, "document_complete": False,
        "formal_release_evidence": False, "rsi_enabled": False,
        "semantic_quality_verified": False, "full_report_evaluation": False,
        "model_or_lora_modified": False, "labels_read": False,
        "budget_job_id": job_id, "manifest_sha256": args.manifest_sha256,
        "max_jobs": args.max_jobs, "limit_seconds": args.limit_seconds,
        "actual_model_calls": 0, "trial_id": uuid4().hex,
    }
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        created = True
        with deadline(args.limit_seconds):
            cases = source_cases(args)
            report["source_hashes_verified"] = len(cases)
            identity = (
                load_identity(model, args.prepared_identity,
                              expected_sha256=args.prepared_identity_sha256)
                if args.prepared_identity else model.identity()
            )
            model_sha = json_digest(identity)
            report.update(model_identity=identity, model_identity_sha256=model_sha,
                          model_manifest_sha256=file_digest(args.model_manifest),
                          probe_source_sha256=file_digest(Path(__file__)))
            if not args.planning_only:
                report["visual_capability"] = validate_vision_capability(args.visual_capability, identity)
            config = {"trial_id": report["trial_id"], "max_tokens": args.max_tokens}
            if getattr(args, "visual_display_transforms", False):
                config["allow_display_transforms"] = True
            report["visual_config"] = dict(config)

            def progress(rows):
                report["reports"] = rows
                report["planning"] = _summary(report)
                atomic_json(args.output_dir / "report-visual-probe.json", report)

            planning_started = time.monotonic()
            try:
                rows = plan_all(cases, model_sha, config, args.output_dir / "plans",
                                workers=args.workers, progress=progress)
            finally:
                report["planning_wall_seconds"] = time.monotonic() - planning_started
            report["planning_complete"] = len(rows) == len(cases) and all(row["status"] == "planned" for row in rows)
            if not report["planning_complete"]:
                raise ValueError("full frozen source planning incomplete; no model pilot started")
            if args.planning_only:
                status = "complete"
            else:
                eligible = [row for row in rows if row["supported_unique_jobs"] > 0]
                if args.report_id:
                    eligible = [row for row in eligible if row["id"] == args.report_id]
                if not eligible:
                    raise ValueError("no supported visual jobs in requested reports")
                selected = min(eligible, key=lambda row: (row["supported_unique_jobs"], row["id"]))
                plan_path = Path(selected["plan_directory"]) / "plan.json"
                if file_digest(plan_path) != selected["plan_file_sha256"]:
                    raise ValueError("selected visual plan changed")
                plan = json.loads(plan_path.read_text())
                ledger = TimingLedger(args.output_dir / "timing.sqlite")
                backend = LoopbackBackend(args.api_url, model.served_model, None,
                                          timeout=args.request_timeout, seed=17)
                generate = AccountedBackend(backend, ledger)
                report["selected_report_id"] = selected["id"]
                report["selected_report_sha256"] = selected["report_sha256"]
                report["selection"] = "explicit-report-id" if args.report_id else "fewest-supported-jobs-then-id; no labels"
                visual_deadline = min(global_deadline, time.monotonic() + 1200)
                token = REPORT_DEADLINE.set(visual_deadline)
                try:
                    def check_deadline():
                        # The core durably catches backend exceptions, including
                        # the one-shot SIGALRM. The absolute deadline must still
                        # prevent every later dispatch once that timer has fired.
                        if time.monotonic() >= min(global_deadline, REPORT_DEADLINE.get()):
                            raise TimeoutError("global/report visual deadline exhausted")

                    check_deadline()
                    report["served_models"] = backend.probe()
                    check_deadline()

                    def before_call(state):
                        check_deadline()
                        if len(state["prior_calls"]) >= args.max_jobs:
                            raise RuntimeError("diagnostic unique-job limit reached; remaining tasks are unexecuted")

                    visual = execute_visual_jobs(plan, generate, args.output_dir / "visual-cache", budget=before_call)
                finally:
                    REPORT_DEADLINE.reset(token)
                # Keep complete original plan and typed real evidence in the cloud.
                atomic_json(args.output_dir / "visual-evidence.json", {"plan": plan, **visual})
                executed = {raw["job_id"] for raw in visual["raw"]}
                outcomes = Counter(item["response"]["outcome"] for item in visual["evidence"])
                durations = [call["seconds"] for call in visual["calls"]
                             if type(call.get("seconds")) in (int, float)
                             and math.isfinite(call["seconds"]) and call["seconds"] >= 0]
                report["visual"] = {
                    "planned_jobs": len(plan["jobs"]), "attempted_jobs": len(visual["calls"]),
                    "unexecuted_jobs": len(plan["jobs"]) - len(executed),
                    "parsed_jobs": len(visual["evidence"]), "outcomes": dict(outcomes),
                    "errors": visual["errors"], "spent_seconds": visual["spent_seconds"],
                    "evidence_sha256": file_digest(args.output_dir / "visual-evidence.json"),
                    "source_unresolved_occurrences": len(plan["unresolved"]),
                    "call_seconds": durations,
                    "prompt_tokens": sum(call["prompt_tokens"] for call in visual["calls"]
                                         if type(call.get("prompt_tokens")) is int),
                    "output_tokens": sum(call["output_tokens"] for call in visual["calls"]
                                         if type(call.get("output_tokens")) is int),
                    "semantic_quality_verified": False,
                }
                report["actual_model_calls"] = ledger.summary()["calls"]
                report["timing"] = ledger.summary()
                if time.monotonic() >= visual_deadline:
                    raise TimeoutError("global/report visual deadline exhausted")
                status = "complete" if visual["evidence"] else "failed"
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            status = "interrupted"
    finally:
        if status == "complete" and time.monotonic() >= global_deadline:
            status = "failed"
            report.update(error_type="TimeoutError", error="global diagnostic deadline exhausted")
        report["status"] = status
        report["wall_seconds"] = time.monotonic() - started
        try:
            if created:
                timing_path = args.output_dir / "timing.sqlite"
                if timing_path.is_file():
                    report["timing"] = TimingLedger(timing_path).summary()
                    report["actual_model_calls"] = report["timing"]["calls"]
                atomic_json(args.output_dir / "report-visual-probe.json", report)
        finally:
            budget.finish(job_id, time.monotonic() - started, status)
    return report


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--_plan-job":
        task = json.loads(Path(sys.argv[2]).read_text())
        result = _plan_one(task["case"], task["model_sha"], task["config"], task["directory"])
        return 0 if result["status"] == "planned" else 1
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model-manifest", "output-dir", "budget-ledger"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--model-policy", choices=("fixed-native-lora", "manifest-locked"),
                        default="fixed-native-lora")
    parser.add_argument("--expected-denominator", type=int,
                        help="Required explicit full development count in manifest-locked mode")
    parser.add_argument("--source-root", type=Path,
                        help="Original training root or explicit /workspace development source directory")
    parser.add_argument("--visual-display-transforms", action="store_true",
                        help="Bind supported display transforms to the visual plan; unsupported sources remain unresolved")
    parser.add_argument("--visual-capability", type=Path)
    parser.add_argument("--prepared-identity", type=Path)
    parser.add_argument("--prepared-identity-sha256")
    parser.add_argument("--planning-only", action="store_true")
    parser.add_argument("--report-id")
    parser.add_argument("--max-jobs", type=int, default=8)
    parser.add_argument("--limit-seconds", type=int, default=1200)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    report = run_probe(parser.parse_args())
    curated = {key: report[key] for key in (
        "status", "expected_denominator", "planning_complete", "actual_model_calls",
        "wall_seconds", "document_complete", "formal_release_evidence", "rsi_enabled",
    )}
    curated["planning"] = report.get("planning")
    if report.get("visual"):
        curated["visual"] = {key: report["visual"][key] for key in (
            "planned_jobs", "attempted_jobs", "unexecuted_jobs", "parsed_jobs", "outcomes",
            "spent_seconds", "call_seconds", "prompt_tokens", "output_tokens",
            "source_unresolved_occurrences", "evidence_sha256", "semantic_quality_verified",
        )}
        curated["selected_report_sha256"] = report["selected_report_sha256"]
    if report.get("error_type"):
        curated["error_type"] = report["error_type"]
    print(json.dumps(curated, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
