"""Idempotent pages keyed by trusted SDK invocation identity, never model args."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def navigation_call_keys(identity, object_key):
    """None means a direct Python call, not an invented native invocation.

    An invocation id deliberately excludes args: changing arguments on replay
    is an identity conflict, not permission to advance another page.
    """
    if identity is None:
        return None
    if not isinstance(identity, dict):
        raise ValueError("Invalid trusted tool-call identity")
    generation = identity.get("generation_call_id")
    native = identity.get("native_tool_call_id")
    if not generation or not native or not identity.get("args_sha256"):
        raise ValueError("Incomplete trusted tool-call identity")
    scope = [identity.get("run_id", identity.get("run")),
             identity.get("session_id", identity.get("session"))]
    return {"native": _digest([scope, native]),
            "generation_object": _digest([scope, generation, object_key]),
            "args_sha256": identity["args_sha256"], "tool_name": identity.get("tool_name")}


class ManagedNavigationJournal:
    """Snapshot is saved atomically together with the caller's page position."""

    def __init__(self, state=None, on_progress=None):
        self._state = deepcopy(state) if state is not None else {"schema_version": 1, "native": {}, "objects": {}}
        if (not isinstance(self._state, dict) or self._state.get("schema_version") != 1
                or not isinstance(self._state.get("native"), dict) or not isinstance(self._state.get("objects"), dict)):
            raise ValueError("Invalid managed navigation journal")
        self.on_progress = on_progress

    def replay(self, identity, object_key):
        keys = navigation_call_keys(identity, object_key)
        if keys is None:
            return None
        row = self._state["native"].get(keys["native"])
        if row is not None:
            if (row["args_sha256"] != keys["args_sha256"] or row.get("tool_name") != keys["tool_name"]
                    or row["object_key"] != object_key):
                raise ValueError("Native tool-call replay changed arguments or object")
            return deepcopy(row["result"])
        native = self._state["objects"].get(keys["generation_object"])
        if native is not None:
            return deepcopy(self._state["native"][native]["result"])
        return None

    def record(self, identity, object_key, result):
        keys = navigation_call_keys(identity, object_key)
        if keys is None:
            return
        previous = self.replay(identity, object_key)
        comparable = lambda page: {k: v for k, v in page.items() if k != "replayed"}
        if previous is not None and comparable(previous) != comparable(result):
            raise ValueError("One generation cannot publish different pages for one object")
        updated = deepcopy(self._state)
        updated["native"][keys["native"]] = {"args_sha256": keys["args_sha256"],
            "tool_name": keys["tool_name"], "object_key": deepcopy(object_key),
            "result": deepcopy(previous if previous is not None else result)}
        updated["objects"][keys["generation_object"]] = keys["native"]
        if self.on_progress is not None:
            self.on_progress(deepcopy(updated))
        self._state = updated

    def snapshot(self):
        return deepcopy(self._state)
