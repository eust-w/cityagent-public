"""Persistent bounded lineage and nonblocking experimental activation."""

from __future__ import annotations

import json
import sqlite3
import threading


class EvolutionStore:
    def __init__(self, path, model_identity, evaluation_identity):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.mutex = threading.RLock()
        self.db.executescript("""CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY, model TEXT, evaluation TEXT);
            CREATE TABLE IF NOT EXISTS candidates (sha TEXT PRIMARY KEY, parent TEXT, generation INTEGER, bundle TEXT, outcome TEXT);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, kind TEXT, value TEXT);""")
        row = self.db.execute(
            "SELECT model,evaluation FROM identity WHERE id=1"
        ).fetchone()
        if row and row != (model_identity, evaluation_identity):
            self.db.close()
            raise ValueError("fixed model/evaluation identity changed")
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO identity VALUES (1,?,?)",
                (model_identity, evaluation_identity),
            )

    def add(self, bundle, generation):
        with self.mutex, self.db:
            if type(generation) is not int or not 0 <= generation <= 3:
                raise ValueError("at most three generations")
            existing = self.db.execute(
                "SELECT sha FROM candidates WHERE sha=?", (bundle.digest,)
            ).fetchone()
            if existing:
                return False
            if self.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] >= 8:
                raise RuntimeError("global eight candidate budget exhausted")
            if bundle.parent_sha256:
                parent = self.db.execute(
                    "SELECT generation FROM candidates WHERE sha=?",
                    (bundle.parent_sha256,),
                ).fetchone()
                if not parent or parent[0] >= generation:
                    raise ValueError("parent missing or cyclic generation")
            self.db.execute(
                "INSERT INTO candidates VALUES (?,?,?,?,NULL)",
                (
                    bundle.digest,
                    bundle.parent_sha256,
                    generation,
                    json.dumps(bundle.as_dict(), allow_nan=False),
                ),
            )
            return True

    def outcome(self, digest, value):
        with self.mutex, self.db:
            changed = self.db.execute(
                "UPDATE candidates SET outcome=? WHERE sha=? AND outcome IS NULL",
                (json.dumps(value, allow_nan=False), digest),
            ).rowcount
            if not changed:
                raise ValueError("unknown or already completed candidate")

    def history(self):
        with self.mutex:
            return [
                {
                    "sha256": sha,
                    "parent_sha256": parent,
                    "generation": generation,
                    "bundle": json.loads(bundle),
                    "outcome": json.loads(outcome) if outcome else None,
                }
                for sha, parent, generation, bundle, outcome in self.db.execute(
                    "SELECT * FROM candidates ORDER BY generation,rowid"
                )
            ]

    def event(self, kind, value):
        with self.mutex, self.db:
            self.db.execute(
                "INSERT INTO events(kind,value) VALUES (?,?)",
                (kind, json.dumps(value, allow_nan=False)),
            )

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class OnlineCoordinator:
    def __init__(self, bundle, store):
        self.active, self.previous, self.ready = bundle, bundle, None
        self.active_evidence = self.previous_evidence = self.ready_evidence = None
        self.store = store
        self.completed = self.updates = 0
        self.mutex = threading.Lock()
        # Restore count only; never infer verification from a candidate's outcome.
        with store.mutex:
            self.updates = store.db.execute(
                "SELECT COUNT(*) FROM events WHERE kind='experimental-activate'"
            ).fetchone()[0]
            self.completed = store.db.execute(
                "SELECT COUNT(*) FROM events WHERE kind='report-complete'"
            ).fetchone()[0]
            row = store.db.execute(
                "SELECT value FROM events WHERE kind IN ('experimental-activate','experimental-rollback') ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                activation = json.loads(row[0])
                digest = activation["bundle_sha256"]
                stored = store.db.execute(
                    "SELECT bundle FROM candidates WHERE sha=?", (digest,)
                ).fetchone()
                if stored:
                    value = json.loads(stored[0])
                    if digest != bundle.digest:
                        # Old flag-only activations cannot become verified after an upgrade.
                        from .rsi_experiments import shadow_evidence

                        try:
                            shadow_evidence(
                                activation.get("evidence", {}).get("shadow_records"),
                                value["parent_sha256"],
                                digest,
                            )
                        except (KeyError, ValueError, TypeError, AttributeError):
                            return
                    self.active_evidence = activation.get("evidence")
                    self.previous_evidence = activation.get("previous_evidence")

                    from .rsi import StrategyBundle

                    self.active = StrategyBundle(
                        value["source"],
                        value["config"],
                        value["parent_sha256"],
                        value["abi"],
                    )
                    parent = store.db.execute(
                        "SELECT bundle FROM candidates WHERE sha=?",
                        (self.active.parent_sha256,),
                    ).fetchone()
                    if parent:
                        value = json.loads(parent[0])
                        self.previous = StrategyBundle(
                            value["source"],
                            value["config"],
                            value["parent_sha256"],
                            value["abi"],
                        )

    def pin(self):
        with self.mutex:
            if self.ready is not None and self.updates < 2:
                self.previous, self.active = self.active, self.ready
                self.previous_evidence, self.active_evidence = (
                    self.active_evidence,
                    self.ready_evidence,
                )
                self.ready = self.ready_evidence = None
                self.updates += 1
                self.store.event(
                    "experimental-activate",
                    {
                        "bundle_sha256": self.active.digest,
                        "evidence": self.active_evidence,
                        "previous_evidence": self.previous_evidence,
                    },
                )
            return self.active

    def offer(self, bundle, *, evidence):
        from .rsi_experiments import shadow_evidence

        try:
            calls = shadow_evidence(
                evidence.get("shadow_records"), self.active.digest, bundle.digest
            )
            if evidence.get("shadow_call_ids") != [c["call_id"] for c in calls]:
                return False
        except (ValueError, KeyError, TypeError):
            return False
        if (
            evidence.get("static") is not True
            or evidence.get("replay") is not True
            or evidence.get("shadow") is not True
            or evidence.get("bundle_sha256") != bundle.digest
            or not evidence.get("shadow_call_ids")
            or evidence.get("worker_ready") is not True
        ):
            return False
        with self.mutex:
            if self.updates >= 2 or bundle.parent_sha256 != self.active.digest:
                return False
            self.ready = bundle
            self.ready_evidence = json.loads(json.dumps(evidence, allow_nan=False))
            self.store.event(
                "experimental-ready",
                {"bundle_sha256": bundle.digest, "evidence": evidence},
            )
            return True

    def report_done(self):
        with self.mutex:
            self.completed += 1
            self.store.event("report-complete", {"ordinal": self.completed})
            return self.completed % 5 == 0 and self.updates < 2

    def rollback(self, reason):
        with self.mutex:
            self.store.event(
                "experimental-rollback",
                {
                    "bundle_sha256": self.previous.digest,
                    "from_sha256": self.active.digest,
                    "reason": reason,
                    "evidence": self.previous_evidence,
                },
            )
            self.active, self.ready = self.previous, None
            self.active_evidence, self.ready_evidence = self.previous_evidence, None
