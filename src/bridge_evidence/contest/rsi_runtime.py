"""Single-owner durable state and one bounded model dispatcher."""

from __future__ import annotations

import contextvars
import fcntl
import heapq
import json
import os
import sqlite3
import threading
from concurrent.futures import Future, as_completed
from pathlib import Path

REQUEST_CALL_ID = contextvars.ContextVar("cityagent_request_call_id", default=None)
REQUEST_CALL_IDS = contextvars.ContextVar("cityagent_request_call_ids", default=None)
REPORT_DEADLINE = contextvars.ContextVar("cityagent_report_deadline", default=None)
REQUEST_PRIORITY = contextvars.ContextVar(
    "cityagent_request_priority", default="extraction"
)


class WorkLock:
    def __init__(self, root):
        Path(root).mkdir(parents=True, exist_ok=True)
        self.path = Path(root) / ".execution.lock"
        self.handle = None

    def __enter__(self):
        self.handle = os.fdopen(
            os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "a"
        )
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            raise RuntimeError("work directory already has an active owner") from error
        return self

    def __exit__(self, *args):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


class TaskJournal:
    """Atomic incremental checkpoint; legacy JSON checkpoints migrate on first write.

    Large append-only collections use individual indexed rows. Receipts and graph
    changes can commit together without reserializing historical fragments.
    """

    COLLECTIONS = frozenset({"tasks", "fragments", "calls", "decisions"})

    def __init__(self, path, identity):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS checkpoint (id INTEGER PRIMARY KEY, identity TEXT, state TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS checkpoint_fields (name TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS checkpoint_items (name TEXT NOT NULL, position INTEGER NOT NULL, value TEXT NOT NULL, PRIMARY KEY(name,position))"
        )
        row = self.db.execute("SELECT identity FROM checkpoint WHERE id=1").fetchone()
        if row and row[0] != identity:
            self.db.close()
            raise ValueError("checkpoint identity changed")
        self.identity = identity

    @staticmethod
    def _json(value):
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )

    def _write_fields(self, fields):
        for name, value in fields.items():
            self.db.execute(
                "INSERT OR REPLACE INTO checkpoint_fields VALUES (?,?)",
                (name, self._json(value)),
            )

    def _migrate(self):
        row = self.db.execute("SELECT state FROM checkpoint WHERE id=1").fetchone()
        if row and row[0] != "null":
            state = json.loads(row[0])
            self._write_fields(
                {k: v for k, v in state.items() if k not in self.COLLECTIONS}
            )
            for name in self.COLLECTIONS & state.keys():
                self.db.executemany(
                    "INSERT OR REPLACE INTO checkpoint_items VALUES (?,?,?)",
                    [(name, i, self._json(v)) for i, v in enumerate(state[name])],
                )
                self._write_fields({"__collection__" + name: True})
        self.db.execute(
            "INSERT OR REPLACE INTO checkpoint VALUES (1,?, 'null')", (self.identity,)
        )

    def update(self, *, fields=None, records=None, replace=None):
        """Commit scalar fields and changed collection rows in one durable transaction."""
        with self.db:
            self._migrate()
            self._write_fields(fields or {})
            for name, values in (replace or {}).items():
                if name not in self.COLLECTIONS:
                    raise ValueError("unknown checkpoint collection")
                self.db.execute("DELETE FROM checkpoint_items WHERE name=?", (name,))
                self.db.executemany(
                    "INSERT INTO checkpoint_items VALUES (?,?,?)",
                    [(name, i, self._json(value)) for i, value in enumerate(values)],
                )
                self._write_fields({"__collection__" + name: True})
            for name, values in (records or {}).items():
                if name not in self.COLLECTIONS:
                    raise ValueError("unknown checkpoint collection")
                self.db.executemany(
                    "INSERT OR REPLACE INTO checkpoint_items VALUES (?,?,?)",
                    [(name, i, self._json(value)) for i, value in values.items()],
                )
                self._write_fields({"__collection__" + name: True})

    def save(self, state):
        with self.db:
            self.db.execute("DELETE FROM checkpoint_fields")
            self.db.execute("DELETE FROM checkpoint_items")
            self.db.execute(
                "INSERT OR REPLACE INTO checkpoint VALUES (1,?, 'null')",
                (self.identity,),
            )
            self._write_fields(
                {k: v for k, v in state.items() if k not in self.COLLECTIONS}
            )
            for name in self.COLLECTIONS & state.keys():
                self.db.executemany(
                    "INSERT INTO checkpoint_items VALUES (?,?,?)",
                    [(name, i, self._json(v)) for i, v in enumerate(state[name])],
                )
                self._write_fields({"__collection__" + name: True})

    def load(self):
        row = self.db.execute("SELECT state FROM checkpoint WHERE id=1").fetchone()
        if not row:
            return None
        if row[0] != "null":
            return json.loads(row[0])
        state = {
            key: json.loads(value)
            for key, value in self.db.execute(
                "SELECT name,value FROM checkpoint_fields"
            )
        }
        for name in self.COLLECTIONS:
            values = self.db.execute(
                "SELECT value FROM checkpoint_items WHERE name=? ORDER BY position",
                (name,),
            ).fetchall()
            if values or state.pop("__collection__" + name, False):
                state[name] = [json.loads(v[0]) for v in values]
            state.pop("__collection__" + name, None)
        return state

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Dispatcher:
    PRIORITY = {"recovery": 0, "extraction": 1, "experiment": 2}

    def __init__(self, backend, capacity=1):
        if capacity not in (1, 2, 4, 8):
            raise ValueError("invalid global capacity")
        self.backend, self.capacity = backend, capacity
        self.condition = threading.Condition()
        self.queue, self.sequence, self.foreground, self.active = [], 0, 0, 0
        self.closed = False
        self.threads = [
            threading.Thread(target=self._work, daemon=True) for _ in range(capacity)
        ]
        for thread in self.threads:
            thread.start()

    def begin_foreground(self):
        with self.condition:
            if self.active or any(item[4] == "experiment" for item in self.queue):
                raise RuntimeError("cannot start foreground while requests are active")
            self.foreground += 1

    def end_foreground(self):
        with self.condition:
            self.foreground -= 1
            if self.foreground < 0:
                raise RuntimeError("unbalanced foreground lease")

    def submit(self, request, priority=None):
        priority = priority or REQUEST_PRIORITY.get()
        if priority not in self.PRIORITY:
            raise ValueError("unknown priority")
        with self.condition:
            if self.closed or (priority == "experiment" and self.foreground):
                raise RuntimeError(
                    "background model requests disabled during foreground batch"
                )
            if len(self.queue) >= 16:
                raise RuntimeError("bounded model queue full")
            future = Future()
            self.sequence += 1
            length = len(request[0]) if isinstance(request[0], (list, tuple)) else 0
            heapq.heappush(
                self.queue,
                (
                    self.PRIORITY[priority],
                    length,
                    self.sequence,
                    request,
                    priority,
                    future,
                    contextvars.copy_context(),
                ),
            )
            self.condition.notify()
            return future

    def _work(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.queue or self.closed)
                if not self.queue:
                    return
                _, _, _, request, _, future, context = heapq.heappop(self.queue)
                self.active += 1
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        future.set_result(
                            context.run(request[0])
                            if callable(request[0]) and request[1] is None
                            else context.run(self.backend, *request)
                        )
                    except BaseException as error:
                        future.set_exception(error)
            finally:
                with self.condition:
                    self.active -= 1
                    self.condition.notify_all()

    def completed(self, requests, priority=None):
        futures = {
            self.submit(request, priority): i for i, request in enumerate(requests)
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                result = error
            yield futures[future], result

    def batch(self, requests):
        results = [None] * len(requests)
        for i, result in self.completed(requests):
            results[i] = result
        return results

    def __call__(self, ids, maximum):
        return self.submit((ids, maximum)).result()

    def call_with_role(self, ids, maximum, *, role):
        def run():
            return self.backend.call_with_role(ids, maximum, role=role)

        return self.submit_callable(
            run,
            priority="experiment"
            if role.startswith("rsi-")
            else REQUEST_PRIORITY.get(),
        ).result()

    def submit_callable(self, function, priority="extraction"):
        return self.submit((function, None), priority)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        for thread in self.threads:
            thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
