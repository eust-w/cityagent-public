"""Receipt-driven model reads with no cursor copying.

The immutable document still uses exact source offsets. This facade owns the
continuation queue, including source fragments and context-recovery pages, so
the model chooses *what* to read rather than reconstructing tool bookkeeping.
Delivery state never means the material has been understood or extracted.
"""
from __future__ import annotations

from copy import deepcopy
from functools import wraps
import hashlib
import json
from typing import Any, Callable

from .document_tools import DocumentTools, _recorded
from .managed_navigation_journal import ManagedNavigationJournal


_AUTOMATIC_NAMES = {"read_section", "read_source", "read_table_rows", "read_outline", "search_in_document"}
_RANGE_NAMES = {"read_source": "read_source_range", "read_section": "read_section_range",
                "read_table_rows": "read_table_range"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _range_references(value: Any) -> Any:
    """The model chooses an object; exact coordinates stay in the program."""
    if isinstance(value, list):
        return [_range_references(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _range_references(item) for key, item in value.items()}
    if isinstance(result.get("source_range"), dict) and "next_read" in result:
        result["next_action"] = "Read the identified source object; the program owns its range and continuation. This failed call did not deliver source text."
    if isinstance(result.get("tool"), str) and isinstance(result.get("arguments"), dict):
        name = result["tool"]
        ordinary = {**{name: name for name in _RANGE_NAMES},
                    **{value: key for key, value in _RANGE_NAMES.items()}}.get(name)
        if ordinary:
            key = {"read_source": "evidence_id", "read_section": "section_id", "read_table_rows": "table_id"}[ordinary]
            if key in result["arguments"]:
                result["tool"] = ordinary
                result["arguments"] = {key: result["arguments"][key], "restart": True}
    return result


def _range_recorded(method: Callable) -> Callable:
    recorded = _recorded(method)

    @wraps(method)
    def call(self: "DocumentReadNavigation", *args: Any, **kwargs: Any) -> dict[str, Any]:
        with self.tools._lock:
            previous = self.tools._response_projection
            self.tools._response_projection = _range_references
            try:
                return recorded(self, *args, **kwargs)
            finally:
                self.tools._response_projection = previous

    return call


def _automatic_recorded(method: Callable) -> Callable:
    recorded = _recorded(method)

    @wraps(method)
    def call(self: "DocumentReadNavigation", *args: Any, **kwargs: Any) -> dict[str, Any]:
        with self.tools._lock:
            previous = self.tools._response_projection
            self._active = None
            self._pending = None
            self._replay_pending = None
            self.tools._response_projection = self._project
            try:
                return recorded(self, *args, **kwargs)
            finally:
                self.tools._response_projection = previous
                self._active = None
                self._pending = None
                self._replay_pending = None

    return call


class DocumentReadNavigation:
    """One document / Agent's resumable reading positions.

    A snapshot is program-owned data, not a model-supplied tool argument.
    Independent instances and object identities never share a cursor. Explicit
    low-level reads remain available and do not advance these positions.
    """

    def __init__(self, tools: DocumentTools, *, state: dict[str, Any] | None = None,
                 on_progress: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.tools = tools
        if on_progress is not None and not callable(on_progress):
            raise ValueError("on_progress must be callable")
        self.on_progress = on_progress
        self._state = {"schema_version": 1, "document_sha256": tools.manifest["document_sha256"], "objects": {}}
        if tools.delivery_ledger is not None:
            self._state["source_delivery_identity"] = deepcopy(tools.delivery_ledger.identity)
        if tools.material_tools is not None:
            self._state["material_identity"] = deepcopy(tools.material_tools.index.identity)
        if state is not None:
            self._state = self._validated_snapshot(state)
        self._active: dict[str, Any] | None = None
        self._pending: dict[str, Any] | None = None
        self._replay_pending = None

    def __getattr__(self, name: str) -> Any:
        # The common receipt decorator uses the same locks, token budget and
        # durable source-read callback as explicit DocumentTools methods.
        return getattr(self.tools, name)

    def _validated_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        if (not isinstance(state, dict) or state.get("schema_version") != 1
                or state.get("document_sha256") != self.tools.manifest["document_sha256"]
                or not isinstance(state.get("objects"), dict)):
            raise ValueError("Navigation snapshot does not match the immutable document")
        result = json.loads(_json(state))
        if self.tools.delivery_ledger is not None:
            expected = self.tools.delivery_ledger.identity
            if result.get("source_delivery_identity") != expected:
                if result.get("objects") or result.get("source_delivery_identity") is not None:
                    raise ValueError("Navigation snapshot belongs to another manifest/run/Agent delivery ledger")
                result["source_delivery_identity"] = deepcopy(expected)
        expected_material = (self.tools.material_tools.index.identity
                             if self.tools.material_tools is not None else None)
        if result.get("material_identity") != expected_material:
            if result.get("objects") or result.get("material_identity") is not None:
                raise ValueError("Navigation snapshot belongs to another material index")
            result["material_identity"] = deepcopy(expected_material)
        for key, value in result["objects"].items():
            if (not isinstance(value, dict) or not isinstance(value.get("identity"), dict)
                    or self._key(value["identity"]) != key
                    or type(value.get("complete")) is not bool
                    or type(value.get("delivered_calls")) is not int or value["delivered_calls"] < 0
                    or not isinstance(value.get("pending_calls"), list)
                    or value["complete"] != (not value["pending_calls"])):
                raise ValueError("Invalid program navigation snapshot")
            for call in value["pending_calls"]:
                self._validate_call(call)
            if self.tools.delivery_ledger is not None and value.get("pending_delivery_receipt"):
                self.tools.delivery_ledger.pending_page(value["pending_delivery_receipt"])
        return result

    @staticmethod
    def _key(identity: dict[str, Any]) -> str:
        return hashlib.sha256(_json(identity).encode()).hexdigest()

    @staticmethod
    def _validate_call(call: dict[str, Any]) -> None:
        if (not isinstance(call, dict) or call.get("tool") not in _AUTOMATIC_NAMES
                or not isinstance(call.get("arguments"), dict)):
            raise ValueError("Invalid source continuation in program navigation snapshot")

    def snapshot(self) -> dict[str, Any]:
        with self.tools._lock:
            return deepcopy(self._state)

    def handlers(self) -> dict[str, Callable]:
        handlers = self.tools.handlers()
        handlers.update({name: getattr(self, name) for name in _AUTOMATIC_NAMES})
        handlers.update({name: getattr(self, name) for name in _RANGE_NAMES.values()})
        handlers["get_neighbors"] = self.get_neighbors
        return handlers

    def _commit_recorded_success(self, result: dict[str, Any]) -> None:
        if self._replay_pending is not None:
            invocation, key = self._replay_pending
            journal = ManagedNavigationJournal(self._state.get("journal"))
            journal.record(invocation, key, result)
            if invocation is not None:
                updated = deepcopy(self._state)
                updated["journal"] = journal.snapshot()
                if self.on_progress is not None:
                    self.on_progress(deepcopy(updated))
                self._state = updated
            return
        if self._pending is None:
            return
        updated = deepcopy(self._state)
        pending = deepcopy(self._pending)
        if self.delivery_ledger is not None:
            pending["pending_delivery_receipt"] = result["source_delivery_receipt"]
        updated["objects"][self._active["key"]] = pending
        journal = ManagedNavigationJournal(updated.get("journal"))
        journal.record(self._active.get("invocation"), self._active["key"], result)
        if self._active.get("invocation") is not None:
            updated["journal"] = journal.snapshot()
        if self.on_progress is not None:
            # Never publish/advance a position whose durable write failed.
            self.on_progress(deepcopy(updated))
        self._state = updated

    @staticmethod
    def _following(result: dict[str, Any], tail: list[dict[str, Any]]) -> list[dict[str, Any]]:
        following = []
        for name in ("continuation", "resume_section"):
            call = result.get(name)
            if call is not None:
                DocumentReadNavigation._validate_call(call)
                following.append(deepcopy(call))
        return [*following, *deepcopy(tail)]

    def _project(self, result: dict[str, Any]) -> dict[str, Any]:
        if self._active is None or "automatic_navigation" in result:
            return result
        if not result.get("ok"):
            return _range_references(result)
        original = result
        result = _range_references(result)
        pending = self._following(original, self._active["tail"])
        identity = self._active["identity"]
        # The model sees a repeatable high-level action, never the source
        # continuation queue or a stale cursor copied from another object.
        result.pop("resume_section", None)
        result["continuation"] = (deepcopy(self._active["next_action"]) if pending else None)
        result["automatic_navigation"] = {
            "object": deepcopy(identity), "delivery_complete": not pending,
            "progress_meaning": "delivered_source_not_understanding_or_extraction",
            "restart_required_to_reread": not pending,
        }
        if self._active.get("invocation") is not None:
            result.setdefault("replayed", False)
        return result

    def _raw_call(self, call: dict[str, Any]) -> dict[str, Any]:
        self._validate_call(call)
        method = getattr(DocumentTools, call["tool"]).__wrapped__
        return method(self.tools, **call["arguments"])

    def _run(self, identity: dict[str, Any], initial: dict[str, Any], *, restart: bool,
             next_action: dict[str, Any], page_size: int | None = None) -> dict[str, Any]:
        if type(restart) is not bool:
            raise ValueError("restart must be a boolean; true explicitly rereads this object from its beginning")
        key = self._key(identity)
        from . import tong_react_adapter
        identity_provider = getattr(tong_react_adapter, "current_tool_call_identity", lambda: None)
        invocation = identity_provider()
        replay = ManagedNavigationJournal(self._state.get("journal")).replay(invocation, key)
        if replay is not None:
            self._replay_pending = (invocation, key)
            replay["replayed"] = True
            return replay
        before = self._state["objects"].get(key)
        if before is not None and self.delivery_ledger is not None and before.get("pending_delivery_receipt"):
            replay = self.delivery_ledger.pending_page(before["pending_delivery_receipt"])
            if replay is not None:
                self._replay_pending = (invocation, key)
                replay["replayed"] = True
                return replay
        if restart or before is None:
            before = {"identity": deepcopy(identity), "complete": False,
                      "delivered_calls": 0, "pending_calls": [deepcopy(initial)]}
        self._active = {"key": key, "identity": identity, "next_action": next_action, "tail": [],
                        "invocation": invocation}
        if before["complete"]:
            return {"ok": True, "status": "already_delivered", "spans": [], "continuation": None,
                    "message": "This object's source pages were already delivered. This does not establish understanding, extraction or review. Use restart=true only for an intentional reread."}
        queue = deepcopy(before["pending_calls"])
        # An oversized table header can first return a source-addressed
        # recovery call rather than text. Follow this routing internally. No
        # source or navigation progress is granted for a failed intermediate
        # result, and each invocation still delivers one bounded response.
        for _ in range(4):
            call, tail = queue[0], queue[1:]
            arguments = call["arguments"]
            if call["tool"] == "read_section" and identity["tool"] == "read_section":
                arguments["max_units"] = page_size
            elif call["tool"] == "read_table_rows" and identity["tool"] == "read_table_rows":
                end = min(arguments["end"], arguments["start"] + page_size)
                if end < arguments["end"]:
                    tail.insert(0, {"tool": "read_table_rows", "arguments": {
                        "table_id": arguments["table_id"], "start": end, "end": arguments["end"]}})
                    arguments["end"] = end
            elif call["tool"] == "read_outline":
                arguments["limit"] = page_size
            elif call["tool"] == "search_in_document":
                arguments["max_matches"] = page_size
            self._active["tail"] = tail
            result = self._raw_call(call)
            if result.get("ok"):
                following = self._following(result, tail)
                self._pending = {"identity": deepcopy(identity), "complete": not following,
                                 "delivered_calls": before["delivered_calls"] + 1,
                                 "pending_calls": following}
                return result
            if (result.get("status") != "budget_exceeded" or not result.get("continuation")
                    or result.get("error", {}).get("recoverable") is not True):
                return result
            queue = self._following(result, tail)
        return {"ok": False, "status": "error", "error": {
            "type": "NavigationRecoveryError", "message": "Source context recovery did not yield a bounded page; no automatic position was advanced.",
            "recoverable": True}}

    @staticmethod
    def _size(value: int, name: str, maximum: int | None = None) -> None:
        if type(value) is not int or value <= 0 or (maximum is not None and value > maximum):
            raise ValueError(f"{name} must be a positive integer" + (f" <= {maximum}" if maximum is not None else ""))

    @_automatic_recorded
    def read_section(self, section_id: str, max_units: int = 8, restart: bool = False) -> dict[str, Any]:
        """Read/continue this chapter; start a new chapter at its own beginning."""
        self._size(max_units, "max_units")
        if self.tools.material_tools is not None and self.tools.material_tools.looks_like_ref(section_id):
            section_id = self.tools.material_tools.index.qualify_ref(section_id)
        identity = {"tool": "read_section", "section_id": section_id}
        return self._run(identity, {"tool": "read_section", "arguments": {
            "section_id": section_id, "cursor": 0, "max_units": max_units}}, restart=restart,
            next_action={"tool": "read_section", "arguments": {"section_id": section_id, "max_units": max_units}}, page_size=max_units)

    @_automatic_recorded
    def read_source(self, evidence_id: str, restart: bool = False) -> dict[str, Any]:
        """Read/continue one original unit, including long text and table context."""
        if self.tools.material_tools is not None and self.tools.material_tools.looks_like_ref(evidence_id):
            evidence_id = self.tools.material_tools.index.qualify_ref(evidence_id)
        identity = {"tool": "read_source", "evidence_id": evidence_id}
        initial = {"tool": "read_source", "arguments": {"evidence_id": evidence_id}}
        if evidence_id.startswith("@piece/"):
            section_id, index = evidence_id[len("@piece/"):].rsplit("/", 1)
            initial["arguments"]["start"] = self.tools._recovery_entries(section_id)[int(index)]["start"]
        return self._run(identity, initial, restart=restart,
                         next_action={"tool": "read_source", "arguments": {"evidence_id": evidence_id}})

    @_automatic_recorded
    def read_table_rows(self, table_id: str, max_rows: int = 8, restart: bool = False) -> dict[str, Any]:
        """Read/continue a table; the program owns row and context continuations."""
        self._size(max_rows, "max_rows")
        table_id = self.tools.canonical_table_id(table_id)
        return self._run({"tool": "read_table_rows", "table_id": table_id},
                         {"tool": "read_table_rows", "arguments": {"table_id": table_id, "start": 0,
                                                                    "end": self.tools._tables[table_id]["row_count"]}},
                         restart=restart, next_action={"tool": "read_table_rows", "arguments": {
                             "table_id": table_id, "max_rows": max_rows}}, page_size=max_rows)

    @_automatic_recorded
    def read_outline(self, limit: int = 20, restart: bool = False) -> dict[str, Any]:
        """Read/continue the inventory; this never advances source reading."""
        self._size(limit, "limit", 100)
        return self._run({"tool": "read_outline"}, {"tool": "read_outline", "arguments": {"cursor": 0, "limit": limit}},
                         restart=restart, next_action={"tool": "read_outline", "arguments": {"limit": limit}}, page_size=limit)

    @_automatic_recorded
    def search_in_document(self, query: str | list[str], scope: str = "default", section_id: str | None = None,
                           max_matches: int = 8, restart: bool = False) -> dict[str, Any]:
        """Search/continue these exact terms and scope; other queries are isolated."""
        self._size(max_matches, "max_matches", 100)
        if self.tools.material_tools is not None and self.tools.material_tools.looks_like_ref(section_id):
            section_id = self.tools.material_tools.index.qualify_ref(section_id)
        identity = {"tool": "search_in_document", "query": query, "scope": scope, "section_id": section_id}
        arguments = {"query": query, "scope": scope, "section_id": section_id, "max_matches": max_matches}
        return self._run(identity, {"tool": "search_in_document", "arguments": {**arguments, "cursor": 0}},
                         restart=restart, next_action={"tool": "search_in_document", "arguments": arguments}, page_size=max_matches)

    @_range_recorded
    def read_source_range(self, evidence_id: str, start: int = 0, end: int | None = None) -> dict[str, Any]:
        """Intentional exact source reread; does not move automatic navigation."""
        return DocumentTools.read_source.__wrapped__(self.tools, evidence_id, start, end)

    @_range_recorded
    def read_section_range(self, section_id: str, cursor: int = 0, max_units: int = 8) -> dict[str, Any]:
        """Intentional exact chapter reread; does not move automatic navigation."""
        return DocumentTools.read_section.__wrapped__(self.tools, section_id, cursor, max_units)

    @_range_recorded
    def read_table_range(self, table_id: str, start: int, end: int) -> dict[str, Any]:
        """Intentional exact table reread; does not move automatic navigation."""
        return DocumentTools.read_table_rows.__wrapped__(self.tools, table_id, start, end)

    @_range_recorded
    def get_neighbors(self, evidence_id: str) -> dict[str, Any]:
        """Neighbor reads keep exact recovery references callable in this facade."""
        return DocumentTools.get_neighbors.__wrapped__(self.tools, evidence_id)
