"""Bounded JSON cache; legacy pickle-based cache files are never opened."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterator


def _reject_constant(value: str) -> None:
    raise ValueError("Cache values must contain only finite JSON numbers")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate object key in cached JSON")
        result[key] = value
    return result


class JsonCache:
    """The get/set subset used by GatewayClient, backed by independent connections.

    Writes are atomic SQLite transactions. Missing keys return the requested
    default; corrupt entries or databases raise instead of hiding the problem.
    There is no pickle compatibility or migration path.
    """

    FILENAME = "cache-json-v1.sqlite3"
    MAX_DEPTH = 64
    MAX_NODES = 250_000
    MAX_KEY_BYTES = 1024

    def __init__(self, directory: str | Path, *, timeout: float = 5.0,
                 max_value_bytes: int = 16 * 1024 * 1024):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError("Cache timeout must be finite and between 0 and 60 seconds")
        if type(max_value_bytes) is not int or not 1 <= max_value_bytes <= 64 * 1024 * 1024:
            raise ValueError("Cache value byte limit must be an integer between 1 and 64 MiB")
        self.path = Path(directory) / self.FILENAME
        self.timeout = float(timeout)
        self.max_value_bytes = max_value_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, value BLOB NOT NULL)")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=self.timeout)
        try:
            connection.execute("PRAGMA trusted_schema = OFF")
            with connection:
                yield connection
        finally:
            connection.close()

    @classmethod
    def _check_key(cls, key: str) -> None:
        if type(key) is not str or not key:
            raise TypeError("Cache key must be a nonempty string")
        if len(key) > cls.MAX_KEY_BYTES or len(key.encode("utf-8")) > cls.MAX_KEY_BYTES:
            raise ValueError("Cache key exceeds the byte limit")

    def _validate(self, value: Any) -> None:
        stack = [(value, 0, False)]
        active: set[int] = set()
        nodes = 0
        while stack:
            node, depth, exiting = stack.pop()
            if exiting:
                active.remove(id(node))
                continue
            nodes += 1
            if nodes > self.MAX_NODES:
                raise ValueError("Cache value exceeds the node limit")
            if depth > self.MAX_DEPTH:
                raise ValueError("Cache value exceeds the depth limit")
            if node is None or type(node) in (bool, int):
                continue
            if type(node) is float:
                if not math.isfinite(node):
                    raise ValueError("Cache values must contain only finite JSON numbers")
            elif type(node) is str:
                if len(node) > self.max_value_bytes or len(node.encode("utf-8")) > self.max_value_bytes:
                    raise ValueError("Cache value exceeds the byte limit")
            elif type(node) in (dict, list):
                if id(node) in active:
                    raise ValueError("Cyclic values are not JSON")
                if len(node) > self.MAX_NODES:
                    raise ValueError("Cache value exceeds the node limit")
                active.add(id(node))
                stack.append((node, depth, True))
                if type(node) is dict:
                    for key, item in node.items():
                        if type(key) is not str:
                            raise TypeError("JSON object keys must be strings")
                        stack.append((key, depth + 1, False))
                        stack.append((item, depth + 1, False))
                else:
                    stack.extend((item, depth + 1, False) for item in node)
            else:
                raise TypeError("Cache values must contain only JSON types")

    def get(self, key: str, default: Any = None) -> Any:
        self._check_key(key)
        with self._connection() as connection:
            # Avoid loading a corrupt oversized payload into Python before its
            # byte length has been checked. TEXT and other storage types fail.
            row = connection.execute(
                "SELECT CASE WHEN typeof(value) = 'blob' AND length(value) <= ? THEN value END, "
                "typeof(value), length(value) FROM entries WHERE key = ?",
                (self.max_value_bytes, key),
            ).fetchone()
        if row is None:
            return default
        payload, storage_type, size = row
        if storage_type != "blob":
            raise ValueError("Cached JSON must be stored as UTF-8 bytes")
        if size > self.max_value_bytes:
            raise ValueError("Cached JSON exceeds the byte limit")
        try:
            value = json.loads(payload.decode("utf-8"), parse_constant=_reject_constant,
                               object_pairs_hook=_unique_object)
        except RecursionError as exc:
            raise ValueError("Cached JSON exceeds the depth limit") from exc
        self._validate(value)
        return value

    def set(self, key: str, value: Any) -> bool:
        self._check_key(key)
        self._validate(value)
        payload = bytearray()
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        for chunk in encoder.iterencode(value):
            encoded = chunk.encode("utf-8")
            if len(payload) + len(encoded) > self.max_value_bytes:
                raise ValueError("Cache value exceeds the byte limit")
            payload.extend(encoded)
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO entries(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, bytes(payload)),
            )
        return True
