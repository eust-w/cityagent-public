"""Bounded preparation/report overlap; isolated report state and SDK proof gates."""

import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .run_state import file_digest, json_digest


def pipeline(files, prepare, process, *, workers=2, timing_observer=None):
    """Yield terminal reports, optionally observing submission-to-terminal timings.

    The observer runs on the consuming thread before each yield. Timestamps use
    one process's monotonic clock; terminal timestamps are captured in the
    report worker, so slow consumption/scoring is not charged to that report.
    Existing callers and the bounded preparation queue are unchanged.
    """
    if workers not in (1, 2, 4):
        raise ValueError("one, two or four reports in flight")
    timings = {}

    def prepare_timed(index, item):
        timing = timings[index]
        timing["preparation_started_monotonic_seconds"] = time.monotonic()
        try:
            return prepare(item)
        finally:
            timing["preparation_finished_monotonic_seconds"] = time.monotonic()

    def submit_preparation(executor, index):
        if timing_observer is None:
            return executor.submit(prepare, files[index])
        timings[index] = {
            "protocol": "preparation-submission-to-report-terminal-v1",
            "clock": "time.monotonic",
            "preparation_submitted_monotonic_seconds": time.monotonic(),
        }
        return executor.submit(prepare_timed, index, files[index])

    # Every report slot gets its own preparation ahead (at least two), so four
    # workers do not wait on two serial rounds of document conversion at start.
    depth = max(2, workers)
    with (
        ThreadPoolExecutor(max_workers=depth) as preparation,
        ThreadPoolExecutor(max_workers=workers) as reports,
    ):
        prepared = {
            i: submit_preparation(preparation, i) for i in range(min(depth, len(files)))
        }
        active, cursor = {}, 0

        def execute(index, future):
            if timing_observer is None:
                return process(index, future.result())
            timing = timings[index]
            timing["terminal_stage"] = "prepare"
            timing["status"] = "failed"
            try:
                value = future.result()
                timing["processing_started_monotonic_seconds"] = time.monotonic()
                timing["terminal_stage"] = "process"
                result = process(index, value)
                timing["status"] = "success"
                return result
            finally:
                terminal = time.monotonic()
                timing["terminal_monotonic_seconds"] = terminal
                timing["end_to_end_wall_seconds"] = (
                    terminal - timing["preparation_submitted_monotonic_seconds"]
                )
                timing["preparation_wall_seconds"] = (
                    timing["preparation_finished_monotonic_seconds"]
                    - timing["preparation_started_monotonic_seconds"]
                )
                timing["preparation_queue_wall_seconds"] = (
                    timing["preparation_started_monotonic_seconds"]
                    - timing["preparation_submitted_monotonic_seconds"]
                )
                processing_start = timing.get("processing_started_monotonic_seconds")
                timing["prepared_to_processing_wall_seconds"] = (
                    processing_start - timing["preparation_finished_monotonic_seconds"]
                    if processing_start is not None else None
                )
                timing["pipeline_processing_wall_seconds"] = (
                    terminal - processing_start if processing_start is not None else None
                )

        while cursor < len(files) or active:
            while cursor < len(files) and len(active) < workers:
                future = reports.submit(execute, cursor, prepared.pop(cursor))
                active[future] = cursor
                if cursor + depth < len(files):
                    prepared[cursor + depth] = submit_preparation(preparation, cursor + depth)
                cursor += 1
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                index = active.pop(future)
                try:
                    value = future.result()
                except Exception as error:
                    value = error
                if timing_observer is not None:
                    timing_observer(index, timings.pop(index))
                yield index, value


def approved_reports(manifest_path, reports):
    value = json.loads(Path(manifest_path).read_text())
    if (
        value.get("purpose") != "development-regression"
        or value.get("schema_version") != 2
    ):
        raise ValueError("explicit approved development manifest required")
    # Only construct a generation view; never forward gold paths or metrics.
    allowed = {
        str(Path(s["report"]).resolve()): s["report_sha256"] for s in value["samples"]
    }
    for report in reports:
        if allowed.get(str(report.resolve())) != file_digest(report):
            raise ValueError("report outside approved development input")
    return {
        "purpose": "development-regression",
        "reports": [{"sha256": file_digest(p)} for p in reports],
    }


def verify_parallel_sdk(path, acceptance_path, url="http://127.0.0.1:8000/v1", workers=2):
    from .backends import normalize_api_url
    from .execution_audit import _native_errors

    if not acceptance_path:
        return False
    try:
        value = json.loads(Path(path).read_text())
        if normalize_api_url(value.get("url")) != normalize_api_url(url):
            return False
        if (
            value.get("schema_version") != 1
            or value.get("sdk_acceptance_sha256") != file_digest(acceptance_path)
            or value.get("runtime_sha256")
            != file_digest(Path(__file__).parents[3] / "run.py")
            or value.get("protected_sha256") != parallel_runtime_digest()
            or value.get("overlap") is not True
        ):
            return False
        receipts = value["receipts"]
        # The proof must cover at least as many overlapping workflows as are used.
        if (len(receipts) < max(2, workers)
                or len({r["document_sha256"] for r in receipts}) != len(receipts)):
            return False
        return all(
            not _native_errors(r["workflow"])
            and r["model_call_ids"]
            and file_digest(Path(r["output_path"])) == r["output_sha256"]
            for r in receipts
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def parallel_runtime_digest():
    root = Path(__file__).parents[3]
    paths = [
        "run.py",
        "src/bridge_evidence/contest/rsi_runtime.py",
        "src/bridge_evidence/contest/rsi_pipeline.py",
        "src/bridge_evidence/contest/tong_workflow.py",
    ]
    return json_digest({name: file_digest(root / name) for name in paths})
