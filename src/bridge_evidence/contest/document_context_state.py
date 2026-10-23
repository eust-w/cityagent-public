"""Exact host progress for context compression, never a model-written summary.

Only actual tool responses and DocumentWorkspace.snapshot() are observations.
A successful source exposure is not a saved fact or semantic acceptance. Full
receipts and workspace snapshots stay on disk; bounded navigation has explicit
pagination and hashes. This module neither calls a model nor grants delivery.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any


def _bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _copy(value: Any) -> Any:
    return json.loads(_bytes(value))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".exact-progress-", delete=False) as stream:
            name = stream.name
            stream.write(_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        name = None
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def _merge(ranges: list[list[int]]) -> list[list[int]]:
    result: list[list[int]] = []
    for start, end in sorted(ranges):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def _span(value: Any) -> dict | None:
    if (not isinstance(value, dict) or not isinstance(value.get("evidence_id"), str)
            or not value["evidence_id"] or type(value.get("start")) is not int
            or type(value.get("end")) is not int or not 0 <= value["start"] <= value["end"]):
        return None
    return {key: value[key] for key in ("evidence_id", "start", "end")}


def _payload(name: str, value: dict) -> tuple[dict, bool | None]:
    # Native envelope and raw handler response are both supported. An explicit
    # outer failure always wins over an inner payload claiming success.
    if "tool" in value and "result" in value:
        if value["tool"] != name or not isinstance(value["result"], dict):
            raise ValueError("Tool result envelope identity differs")
        payload = value["result"]
        success = value.get("ok") if type(value.get("ok")) is bool else None
        if payload.get("ok") is False or payload.get("error"):
            success = False
        return payload, success
    return value, value.get("ok") if type(value.get("ok")) is bool else None


class ExactProgressRecorder:
    """One identity-bound, append-only receipt journal and exact current state.

    observe() may be called once before tools with a trusted initial workspace
    snapshot; no source coverage is invented by doing so. snapshot() contains
    navigation only. read_page() exposes complete saved objects on demand.
    Summary text has no write API and cannot replace authoritative state.
    """
    VIEWS = ("tool_source_ranges", "workspace_reads", "table_continuations", "facts",
             "relations", "draft_output_paths", "draft_bindings", "draft_fact_ids",
             "questions", "failed_saves", "unread_ranges")

    def __init__(self, directory: Path | str, identity: dict):
        if not isinstance(identity, dict):
            raise ValueError("Progress identity must be an object")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = _copy(identity)
        self._thread_lock = threading.RLock()
        self.path = self.directory / "exact-progress.json"
        with self._lock():
            if not self.path.exists():
                _write(self.path, {"schema": "document-exact-progress-v1", "identity": self.identity,
                    "sequence": 0, "receipt_chain_sha256": None, "workspace_ref": None,
                    "workspace_identity": None, "workspace_version": None,
                    "tool_source_ranges": {}, "table_continuations": {}, "failed_saves": [],
                    "tool_outcomes": {"success": 0, "failed": 0, "unknown": 0}})
            self._load()

    @contextmanager
    def _lock(self):
        with self._thread_lock, (self.directory / "exact-progress.lock").open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load(self) -> dict:
        state = json.loads(self.path.read_bytes())
        if state.get("schema") != "document-exact-progress-v1" or state.get("identity") != self.identity:
            raise ValueError("Exact progress identity differs")
        files = sorted((self.directory / "receipts").glob("*.json"))
        if len(files) != state["sequence"]:
            raise RuntimeError("Uncommitted/missing progress receipt; explicit reconstruction required")
        previous = None
        for sequence, path in enumerate(files, 1):
            receipt = json.loads(path.read_bytes())
            if (path.name != f"{sequence:08d}.json" or receipt.get("sequence") != sequence
                    or receipt.get("previous_receipt_sha256") != previous):
                raise ValueError("Progress receipt chain differs")
            previous = _digest(receipt)
        if previous != state["receipt_chain_sha256"]:
            raise ValueError("Progress receipt chain head differs")
        ref = state.get("workspace_ref")
        if ref and hashlib.sha256((self.directory / ref["file"]).read_bytes()).hexdigest() != ref["sha256"]:
            raise ValueError("Exact workspace snapshot hash differs")
        return state

    def observe(self, tool_name: str, args: dict, result: dict, workspace_snapshot: dict) -> dict:
        if not isinstance(tool_name, str) or not tool_name or not all(isinstance(v, dict) for v in (args, result, workspace_snapshot)):
            raise ValueError("observe requires tool name and finite JSON objects")
        args, result, workspace = _copy(args), _copy(result), _copy(workspace_snapshot)
        version = workspace.get("version")
        if (type(version) is not int or version < 0 or workspace.get("workspace_version", version) != version
                or not isinstance(workspace.get("identity"), dict)):
            raise ValueError("Authoritative workspace identity/version is missing")
        for name in ("facts", "relations", "questions", "draft_bindings", "reads"):
            if not isinstance(workspace.get(name), dict):
                raise ValueError("Authoritative workspace collection is missing: " + name)
        payload, success = _payload(tool_name, result)
        with self._lock():
            state = self._load()
            if state["workspace_identity"] is not None and state["workspace_identity"] != workspace["identity"]:
                raise ValueError("Authoritative workspace identity changed")
            if state["workspace_version"] is not None and version < state["workspace_version"]:
                raise ValueError("Authoritative workspace version moved backwards")
            workspace_hash = _digest(workspace)
            if state["workspace_version"] == version and state.get("workspace_ref", {}).get("sha256") != workspace_hash:
                raise ValueError("Workspace changed without a version advance")
            sequence = state["sequence"] + 1
            receipt = {"sequence": sequence, "previous_receipt_sha256": state["receipt_chain_sha256"],
                "tool": tool_name, "args": args, "result": result,
                "successful_response": success, "workspace_version": version,
                "workspace_sha256": workspace_hash}
            state["tool_outcomes"]["success" if success is True else "failed" if success is False else "unknown"] += 1
            if success is True:
                for field in ("spans", "context", "matches"):
                    for value in payload.get(field, []) if isinstance(payload.get(field, []), list) else []:
                        span = _span(value)
                        if span is None:
                            continue
                        # Excerpts/context can be exposed while granting no primary read.
                        kind = "primary" if field == "spans" and payload.get("status") == "read" and payload.get("grants_coverage") is not False and value.get("grants_coverage") is not False else "context_or_excerpt"
                        key = kind + ":" + span["evidence_id"]
                        previous = state["tool_source_ranges"].get(key, {"kind": kind, "evidence_id": span["evidence_id"], "ranges": []})
                        previous["ranges"] = _merge(previous["ranges"] + [[span["start"], span["end"]]])
                        state["tool_source_ranges"][key] = previous
                table_id = payload.get("table_id")
                continuation = payload.get("continuation")
                if isinstance(table_id, str) and table_id:
                    next_row = None
                    if isinstance(continuation, dict) and continuation.get("tool") == "read_table_rows":
                        following = continuation.get("arguments", {})
                        if following.get("table_id") == table_id and type(following.get("start")) is int and following["start"] >= 0:
                            next_row = following["start"]
                    state["table_continuations"][table_id] = {"table_id": table_id,
                        "returned_rows": deepcopy(payload.get("returned_rows")), "next_row": next_row,
                        "continuation": deepcopy(continuation), "receipt_sequence": sequence,
                        "complete_table": None}  # end of requested slice is not end of table
            elif tool_name == "update_workspace":
                error = result.get("error") or payload.get("error") or {}
                if not isinstance(error, dict):
                    error = {"unstructured_error": True}
                error_details = error.get("details") if isinstance(error.get("details"), dict) else {}
                state["failed_saves"].append({"receipt_sequence": sequence,
                    "workspace_version": version, "successful_response": success,
                    # A failed transport/budget response can follow a committed
                    # handler. Only explicit owned atomic diagnostics assert false.
                    "applied": False if error.get("type") == "WorkspaceRepairError" and error_details.get("applied") is False else None,
                    "fields": deepcopy(error.get("fields", [])), "details": deepcopy(error_details),
                    "code": error.get("code"), "receipt_file": f"receipts/{sequence:08d}.json"})
            workspace_file = f"workspaces/{workspace_hash}.json"
            _write(self.directory / workspace_file, workspace)
            _write(self.directory / f"receipts/{sequence:08d}.json", receipt)
            state.update({"sequence": sequence, "receipt_chain_sha256": _digest(receipt),
                "workspace_ref": {"file": workspace_file, "sha256": workspace_hash},
                "workspace_identity": workspace["identity"], "workspace_version": version})
            _write(self.path, state)
            return self._snapshot(state, page_size=16, cursors={})

    def _workspace(self, state: dict) -> dict:
        return json.loads((self.directory / state["workspace_ref"]["file"]).read_bytes()) if state["workspace_ref"] else {}

    def _items(self, state: dict, view: str, *, navigation: bool = False) -> list:
        if view not in self.VIEWS:
            raise ValueError("Unknown exact-progress view")
        workspace = self._workspace(state)
        if view in {"tool_source_ranges", "table_continuations"}:
            return [deepcopy(state[view][key]) for key in sorted(state[view])]
        if view == "failed_saves":
            return deepcopy(state[view])
        if view in {"facts", "relations", "questions"}:
            records = workspace.get(view, {})
            return [{"id": key, **({"status": value.get("status")} if view == "questions" else {})} if navigation
                    else {"id": key, "record": deepcopy(value)} for key, value in sorted(records.items())]
        if view in {"workspace_reads", "draft_bindings"}:
            records = workspace.get("reads" if view == "workspace_reads" else view, {})
            return [{"evidence_id" if view == "workspace_reads" else "path": key,
                     "ranges" if view == "workspace_reads" else "record_ids": deepcopy(value)}
                    for key, value in sorted(records.items())]
        if view == "draft_fact_ids":
            return deepcopy(workspace.get(view, []))
        if view == "unread_ranges":
            return deepcopy(workspace.get("reading_progress", {}).get(view, []))
        draft = workspace.get("draft") or {}
        return [f"基本信息.{key}" for key in sorted(draft.get("基本信息", {}))] + [
            f"{section}.{index}" for section in ("维修建议", "病害明细") for index in range(len(draft.get(section, [])))]

    @staticmethod
    def _page(items: list, cursor: int, limit: int) -> dict:
        if type(cursor) is not int or cursor < 0 or cursor > len(items) or type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("Invalid exact-progress page bounds")
        following = cursor + min(limit, len(items) - cursor)
        return {"items": deepcopy(items[cursor:following]), "total": len(items), "cursor": cursor,
                "next_cursor": following if following < len(items) else None,
                "complete": cursor == 0 and following == len(items)}

    def _snapshot(self, state: dict, *, page_size: int, cursors: dict) -> dict:
        workspace = self._workspace(state)
        policy = workspace.get("identity", {}).get("semantic_review_policy")
        performed = bool(workspace.get("review")) if workspace else None
        status = "not_run_by_user_policy" if policy == "not_run_by_user_policy" and not performed else "observed_review" if performed else "unknown"
        return {"schema": "document-exact-progress-navigation-v1", "identity_sha256": _digest(self.identity),
            "sequence": state["sequence"], "workspace_version": state["workspace_version"],
            "full_state": {"path": str(self.path), "sha256": _digest(state)},
            "workspace_snapshot": {"path": str(self.directory / state["workspace_ref"]["file"]),
                "sha256": state["workspace_ref"]["sha256"]} if state["workspace_ref"] else None,
            "receipt_chain_sha256": state["receipt_chain_sha256"], "tool_outcomes": deepcopy(state["tool_outcomes"]),
            "saved_fact_count": len(workspace.get("facts", {})), "saved_relation_count": len(workspace.get("relations", {})),
            "saved_draft_sha256": workspace.get("draft_sha256"), "draft_valid": workspace.get("draft_valid"),
            "unresolved_question_count": sum(v.get("status") not in {"resolved", "unknown"} for v in workspace.get("questions", {}).values()),
            "semantic_review": {"status": status, "performed": performed, "passed": None},
            "source_exposure_is_not_saved_fact_or_semantic_acceptance": True,
            "pages": {view: self._page(self._items(state, view, navigation=True), cursors.get(view, 0), page_size) for view in self.VIEWS}}

    def snapshot(self, *, page_size: int = 16, cursors: dict | None = None) -> dict:
        if cursors is not None and (not isinstance(cursors, dict) or set(cursors) - set(self.VIEWS)):
            raise ValueError("Unknown exact-progress page cursor")
        with self._lock():
            return self._snapshot(self._load(), page_size=page_size, cursors=cursors or {})

    def read_page(self, view: str, cursor: int = 0, limit: int = 16) -> dict:
        with self._lock():
            state = self._load()
            return {"view": view, "workspace_version": state["workspace_version"],
                    "full_state_sha256": _digest(state), **self._page(self._items(state, view), cursor, limit)}
