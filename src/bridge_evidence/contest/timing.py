"""Durable, idempotent call accounting; interrupted calls remain visible."""

import json
import math
import sqlite3
import time
from pathlib import Path


class TimingLedger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, role TEXT NOT NULL, start REAL NOT NULL, seconds REAL, status TEXT NOT NULL, receipt TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS spans (id TEXT PRIMARY KEY, kind TEXT NOT NULL, seconds REAL NOT NULL)"
            )

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def start(self, call_id, *, role):
        if not call_id or not role:
            raise ValueError("call id and role required")
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO calls VALUES (?,?,?,NULL,?,NULL)",
                (call_id, role, time.time(), "running"),
            )

    def claim(self, call_id, *, role):
        """Reserve a dispatch exactly once, independently of receipt replay."""
        self.claim_many([call_id], role=role)

    def claim_many(self, call_ids, *, role):
        call_ids = list(call_ids)
        if (not role or not call_ids or any(not value for value in call_ids)
                or len(set(call_ids)) != len(call_ids)):
            raise ValueError("unique call ids and role required")
        try:
            with self.connect() as db:
                db.executemany(
                    "INSERT INTO calls VALUES (?,?,?,NULL,?,NULL)",
                    [(value, role, time.time(), "running") for value in call_ids],
                )
        except sqlite3.IntegrityError as error:
            raise ValueError("call reservation already dispatched") from error

    def finish(self, call_id, seconds, *, status, receipt=None):
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds < 0
            or status not in ("success", "failed", "interrupted")
        ):
            raise ValueError("invalid call timing/status")
        encoded = (
            json.dumps(receipt, sort_keys=True, ensure_ascii=False) if receipt else None
        )
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT seconds,status,receipt FROM calls WHERE id=?", (call_id,)
            ).fetchone()
            if not row:
                raise ValueError("call was not started")
            if row[0] is not None:
                if row != (seconds, status, encoded):
                    raise ValueError("conflicting duplicate receipt")
                return
            db.execute(
                "UPDATE calls SET seconds=?,status=?,receipt=? WHERE id=?",
                (seconds, status, encoded, call_id),
            )

    def span(self, span_id, kind, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid span")
        with self.connect() as db:
            old = db.execute(
                "SELECT kind,seconds FROM spans WHERE id=?", (span_id,)
            ).fetchone()
            if old and old != (kind, seconds):
                raise ValueError("conflicting span")
            db.execute(
                "INSERT OR IGNORE INTO spans VALUES (?,?,?)", (span_id, kind, seconds)
            )

    def recover(self):
        with self.connect() as db:
            for call_id, start in db.execute(
                "SELECT id,start FROM calls WHERE status='running'"
            ).fetchall():
                db.execute(
                    "UPDATE calls SET status=?, seconds=? WHERE id=?",
                    ("interrupted", max(0, time.time() - start), call_id),
                )

    def summary(self):
        with self.connect() as db:
            calls = db.execute(
                "SELECT id,role,seconds,status FROM calls ORDER BY start"
            ).fetchall()
            spans = db.execute(
                "SELECT kind,SUM(seconds) FROM spans GROUP BY kind"
            ).fetchall()
        return {
            "generation_seconds": sum(c[2] or 0 for c in calls),
            "calls": len(calls),
            "failures": sum(c[3] in ("failed", "interrupted") for c in calls),
            "incomplete_call_ids": [c[0] for c in calls if c[3] == "running"],
            "spans": dict(spans),
            "accounting": "unique-call-id-v2",
        }
