"""Offline bounded search; fixed weights and immutable evaluation denominator."""

from __future__ import annotations

import math
import random
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from .candidate import DEFAULT, validate_config


class CloudBudget:
    def __init__(self, path, maximum_seconds=86400):
        self.path = str(path)
        self.maximum = maximum_seconds
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, phase TEXT, start REAL, seconds REAL, status TEXT, cap REAL)"
            )

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def used(self, db):
        rows = db.execute("SELECT start,seconds,status FROM jobs").fetchall()
        return sum(
            seconds if seconds is not None else max(0, time.time() - start)
            for start, seconds, status in rows
        )

    def remaining(self):
        with self.connect() as db:
            return max(0, self.maximum - self.used(db))

    def start(self, phase, maximum_seconds):
        if (
            type(maximum_seconds) not in (int, float)
            or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0
        ):
            raise ValueError("positive job limit required")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            remaining = self.maximum - self.used(db)
            if remaining < maximum_seconds:
                raise RuntimeError("cloud experiment budget exhausted or insufficient")
            if db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='running'"
            ).fetchone()[0]:
                raise RuntimeError(
                    "single PPU experiment already running; recover prior job first"
                )
            job = uuid4().hex
            db.execute(
                "INSERT INTO jobs VALUES (?,?,?,NULL,?,?)",
                (job, phase, time.time(), "running", maximum_seconds),
            )
        return job

    def finish(self, job, seconds=None, status="complete"):
        with self.connect() as db:
            row = db.execute(
                "SELECT start,seconds,status FROM jobs WHERE id=?", (job,)
            ).fetchone()
            if not row:
                raise ValueError("unknown experiment")
            duration = max(0, time.time() - row[0]) if seconds is None else seconds
            if (
                type(duration) not in (int, float)
                or not math.isfinite(duration)
                or duration < 0
                or status not in ("complete", "failed", "interrupted")
            ):
                raise ValueError("invalid experiment time or status")
            if row[1] is not None:
                if row[1] != duration or row[2] != status:
                    raise ValueError("conflicting experiment completion")
                return
            db.execute(
                "UPDATE jobs SET seconds=?,status=? WHERE id=?", (duration, status, job)
            )

    def recover(self):
        with self.connect() as db:
            for job, start in db.execute(
                "SELECT id,start FROM jobs WHERE status='running'"
            ).fetchall():
                db.execute(
                    "UPDATE jobs SET seconds=?,status=? WHERE id=?",
                    (max(0, time.time() - start), "interrupted", job),
                )


def candidate_configs(errors, seed=17, maximum=8, supported_parameters=()):
    if not 1 <= maximum <= 8:
        raise ValueError("at most eight candidates")
    baseline = validate_config({**DEFAULT, "seed": seed})
    candidates = [baseline]
    for batch in (2, 4):
        candidates.append(validate_config({**baseline, "batch_size": batch}))
    if errors.get("length_failures", 0) or errors.get("gold_false_negatives", 0):
        candidates.extend(
            [
                validate_config({**baseline, "input_budget": 4096}),
                validate_config({**baseline, "output_budget": 6144}),
            ]
        )
    if errors.get("format_failures", 0) or errors.get("gold_false_positives", 0):
        candidates.append(
            validate_config({**baseline, "prompt": "source-fragment-concise-v1"})
        )
    for flag in ("enable_prefix_caching", "enforce_eager"):
        if flag in supported_parameters:
            candidates.append(
                validate_config(
                    {
                        **baseline,
                        "engine_parameters": {flag: flag == "enable_prefix_caching"},
                    }
                )
            )
    rest = candidates[1:]
    random.Random(seed).shuffle(rest)
    return [baseline, *rest][:maximum]


def diagnose_baseline(result):
    """Use program failures and aggregate gold differences, never label text in prompts."""
    failures = [str(item.get("error") or "") for item in result.get("reports", [])]
    failures += [" ".join(item.get("errors", [])) for item in result.get("details", [])]
    sections = result.get("final_metrics", {}).get("sections", {})
    return {
        "length_failures": sum(
            any(
                word in error.lower()
                for word in ("length", "context", "budget", "truncat")
            )
            for error in failures
        ),
        "format_failures": sum(
            any(
                word in error.lower()
                for word in (
                    "json",
                    "format",
                    "schema",
                    "coverage",
                    "source",
                    "conflict",
                )
            )
            for error in failures
        ),
        "gold_false_negatives": sum(s.get("fn", 0) for s in sections.values()),
        "gold_false_positives": sum(s.get("fp", 0) for s in sections.values()),
    }
