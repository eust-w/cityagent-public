from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import multiprocessing
from pathlib import Path
import pickle
import sqlite3

import pytest

from bridge_evidence.agents.json_cache import JsonCache


def _process_write(directory: str, index: int) -> dict:
    cache = JsonCache(directory)
    value = {"parsed": {"index": index, "text": "桥梁"}}
    cache.set(str(index), value)
    return cache.get(str(index))


def _write_marker(path: str) -> None:
    Path(path).write_text("unsafe pickle was executed", encoding="utf-8")


class _Exploit:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return _write_marker, (str(self.marker),)


def test_json_values_persist_and_round_trip_without_mutation(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path)
    value = {"parsed": {"桥": "云样桥", "bool": True, "null": None, "n": 3.5}, "vectors": [[1.0, 2.0]]}
    assert cache.set("one", value) is True
    value["parsed"]["桥"] = "mutated caller value"
    expected = {"parsed": {"桥": "云样桥", "bool": True, "null": None, "n": 3.5}, "vectors": [[1.0, 2.0]]}
    assert JsonCache(tmp_path).get("one") == expected
    assert cache.get("missing") is None
    marker = object()
    assert cache.get("missing", marker) is marker
    assert cache.set("one", {"results": []}) is True
    assert cache.get("one") == {"results": []}


def test_sql_like_key_is_literal_and_does_not_change_other_entries(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path)
    key = "'); DROP TABLE entries; --"
    cache.set("safe", {"parsed": 1})
    cache.set(key, {"parsed": 2})
    assert cache.get(key) == {"parsed": 2}
    assert cache.get("safe") == {"parsed": 1}


def test_threads_share_a_cache_without_sharing_connections(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path)

    def write(index: int) -> int:
        cache.set(str(index), {"index": index})
        return cache.get(str(index))["index"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(write, range(64))) == list(range(64))


def test_processes_can_write_concurrently(tmp_path: Path) -> None:
    JsonCache(tmp_path)
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        values = list(pool.map(_process_write, [str(tmp_path)] * 12, range(12)))
    assert [value["parsed"]["index"] for value in values] == list(range(12))
    cache = JsonCache(tmp_path)
    assert all(cache.get(str(i)) == values[i] for i in range(12))


def test_legacy_pickle_cache_files_are_not_read(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    payload = pickle.dumps(_Exploit(marker))
    (tmp_path / "cache.db").write_bytes(payload)
    (tmp_path / "legacy.val").write_bytes(payload)
    cache = JsonCache(tmp_path)
    assert cache.get("legacy") is None
    cache.set("new", {"parsed": "safe"})
    assert cache.get("new") == {"parsed": "safe"}
    assert not marker.exists()
    assert (tmp_path / "cache.db").read_bytes() == payload


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), {"x": float("nan")}, {1: "not-string-key"}, {"bad": (1, 2)}, b"not-json", object()])
def test_only_finite_json_values_can_be_written(tmp_path: Path, value: object) -> None:
    cache = JsonCache(tmp_path)
    cache.set("key", {"valid": 1})
    with pytest.raises((ValueError, TypeError)):
        cache.set("key", value)
    assert cache.get("key") == {"valid": 1}


def test_value_byte_limit_and_depth_limit_apply_before_writes(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path, max_value_bytes=16)
    assert cache.set("small", {"x": "桥"}) is True
    with pytest.raises(ValueError, match="byte"):
        cache.set("large", {"x": "桥" * 8})
    nested: object = 0
    for _ in range(66):
        nested = [nested]
    with pytest.raises(ValueError, match="depth"):
        JsonCache(tmp_path).set("deep", nested)


@pytest.mark.parametrize("payload", [b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'{"x":1,"x":2}', b'{"x":', b'\xff'])
def test_corrupt_or_nonfinite_json_is_not_treated_as_a_miss(tmp_path: Path, payload: bytes) -> None:
    cache = JsonCache(tmp_path)
    with sqlite3.connect(cache.path) as connection:
        connection.execute("INSERT INTO entries(key, value) VALUES(?, ?)", ("bad", payload))
    with pytest.raises((ValueError, UnicodeError)):
        cache.get("bad")


def test_pickle_in_new_database_is_rejected_without_execution(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path)
    marker = tmp_path / "executed"
    payload = pickle.dumps(_Exploit(marker))
    with sqlite3.connect(cache.path) as connection:
        connection.execute("INSERT INTO entries(key, value) VALUES(?, ?)", ("bad", payload))
    with pytest.raises((ValueError, UnicodeError)):
        cache.get("bad")
    assert not marker.exists()


def test_oversized_database_value_is_rejected_before_loading_payload(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path, max_value_bytes=16)
    with sqlite3.connect(cache.path) as connection:
        connection.execute("INSERT INTO entries(key, value) VALUES(?, ?)", ("large", json.dumps("x" * 32).encode()))
    with pytest.raises(ValueError, match="byte"):
        cache.get("large")


def test_corrupt_database_is_not_recreated_or_silently_ignored(tmp_path: Path) -> None:
    database = tmp_path / "cache-json-v1.sqlite3"
    database.write_bytes(b"not a SQLite database")
    with pytest.raises(sqlite3.DatabaseError):
        JsonCache(tmp_path)
    assert database.read_bytes() == b"not a SQLite database"


@pytest.mark.parametrize("key", [None, True, 1, "", "桥" * 400])
def test_invalid_keys_are_rejected(tmp_path: Path, key: object) -> None:
    cache = JsonCache(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        cache.set(key, {})
    with pytest.raises((ValueError, TypeError)):
        cache.get(key)


@pytest.mark.parametrize("options", [
    {"timeout": float("nan")}, {"timeout": 0}, {"timeout": True},
    {"max_value_bytes": 0}, {"max_value_bytes": True}, {"max_value_bytes": 64 * 1024 * 1024 + 1},
])
def test_invalid_limits_are_rejected(tmp_path: Path, options: dict) -> None:
    with pytest.raises(ValueError):
        JsonCache(tmp_path, **options)


def test_cycles_are_rejected_but_repeated_json_values_are_allowed(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path)
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="Cyclic"):
        cache.set("cycle", cyclic)
    shared = {"x": 1}
    assert cache.set("repeated", [shared, shared]) is True
    assert cache.get("repeated") == [{"x": 1}, {"x": 1}]


def test_unexpected_storage_type_is_not_a_miss(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path)
    with sqlite3.connect(cache.path) as connection:
        connection.execute("INSERT INTO entries(key, value) VALUES(?, ?)", ("text", '{"x":1}'))
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        cache.get("text")


def test_lock_timeout_is_reported_not_silently_treated_as_cache_miss(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path, timeout=0.02)
    cache.set("key", {"x": 1})
    with sqlite3.connect(cache.path) as connection:
        connection.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            cache.get("key")
