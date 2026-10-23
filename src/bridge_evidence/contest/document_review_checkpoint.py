"""Atomic partial progress for one frozen semantic-review payload.

This is inert JSON, not a model-call ledger or a semantic decision. The caller
must bind the complete workflow/semantic/payload/catalog identity and reconcile
model_call_ids with its actual request receipts before resuming. Hashes detect
damage and mismatched state; they are not signatures against an actor able to
rewrite both the contents and every hash. Single-worker use only.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from .run_state import atomic_json, json_digest


_SCHEMA = 2
_COUNTERS = ("iterations_spent", "tool_calls_spent", "history_reset_count")
_STATE_FIELDS = {"messages", "lookup_state", "model_call_ids", "pending_response", *_COUNTERS}
_ENVELOPE_FIELDS = {"schema_version", "identity", "identity_sha256", "state",
                    "state_sha256", "checkpoint_sha256"}


def _json_value(value, name):
    """Reject coercions (e.g. tuple to list or integer dictionary keys)."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item, name)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item, name)
        return
    raise ValueError(f"Review checkpoint {name} must contain only finite JSON values")


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"Review checkpoint {name} must be an object")
    try:
        _json_value(value, name)
        # UTF-8 serialization must work before touching an existing checkpoint.
        json_digest(value)
    except (TypeError, RecursionError, UnicodeError) as exc:
        raise ValueError(f"Review checkpoint {name} is not valid JSON") from exc
    return deepcopy(value)


def _state(value):
    state = _object(value, "state")
    if set(state) != _STATE_FIELDS:
        raise ValueError("Review checkpoint state has invalid schema fields")
    messages = state["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("Review checkpoint messages must be a nonempty array")
    for message in messages:
        if (not isinstance(message, dict) or not isinstance(message.get("role"), str) or message.get("role") not in
                {"system", "developer", "user", "assistant", "tool", "function"}):
            raise ValueError("Review checkpoint contains an invalid message")
    if not isinstance(state["lookup_state"], dict):
        raise ValueError("Review checkpoint lookup_state must be an object")
    calls = state["model_call_ids"]
    if (not isinstance(calls, list)
            or any(not isinstance(call, str) or not call.strip() for call in calls)
            or len(calls) != len(set(calls))):
        raise ValueError("Review checkpoint model_call_ids must be distinct nonempty strings")
    for name in _COUNTERS:
        if type(state[name]) is not int or state[name] < 0:
            raise ValueError(f"Review checkpoint {name} must be a nonnegative integer")
    pending = state["pending_response"]
    if pending is not None:
        if not isinstance(pending, dict) or set(pending) != {"call_id", "response"}:
            raise ValueError("Review checkpoint pending_response has invalid schema fields")
        if (not calls or pending["call_id"] != calls[-1]
                or not isinstance(pending["call_id"], str) or not pending["call_id"].strip()
                or state["iterations_spent"] < 1):
            raise ValueError("Review checkpoint pending_response must bind the latest spent model call")
        response = pending["response"]
        if (not isinstance(response, dict) or not isinstance(response.get("choices"), list)
                or not response["choices"]):
            raise ValueError("Review checkpoint pending_response needs nonempty response choices")
        for choice in response["choices"]:
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError("Review checkpoint pending_response needs assistant messages")
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("Review checkpoint pending_response has invalid assistant content")
            tools = message.get("tool_calls")
            if tools is not None:
                if not isinstance(tools, list):
                    raise ValueError("Review checkpoint pending_response has invalid tool calls")
                identifiers = []
                for tool in tools:
                    function = tool.get("function") if isinstance(tool, dict) else None
                    if (not isinstance(tool, dict) or tool.get("type") != "function"
                            or not isinstance(tool.get("id"), str) or not tool["id"].strip()
                            or not isinstance(function, dict)
                            or not isinstance(function.get("name"), str) or not function["name"].strip()
                            or not isinstance(function.get("arguments"), str)):
                        raise ValueError("Review checkpoint pending_response has invalid tool call structure")
                    identifiers.append(tool["id"])
                if len(identifiers) != len(set(identifiers)):
                    raise ValueError("Review checkpoint pending_response has duplicate tool call IDs")
            if content is None and not tools:
                raise ValueError("Review checkpoint pending_response has no assistant content or tool calls")
    return state


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Review checkpoint has duplicate JSON keys")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("Review checkpoint contains a nonfinite JSON number")


class ReviewCheckpoint:
    """Load/save a complete pending message tail without replaying any action.

    ``identity`` is the caller's complete JSON object, compared without key
    aliases or defaults. Constructing this class does not create files. ``save``
    refuses to overwrite a checkpoint belonging to any other identity. Schema 2
    requires ``pending_response`` to be None or a complete ``{call_id,response}``
    record. It preserves a successful model reply that the workflow has not yet
    consumed; this class never executes its tools or treats it as a decision.
    """

    _schema = _SCHEMA
    _state_validator = staticmethod(_state)
    _counters = _COUNTERS

    def __init__(self, path, identity: dict):
        self.path = Path(path)
        self._identity = _object(identity, "identity")
        if not self._identity:
            raise ValueError("Review checkpoint identity must not be empty")
        self._identity_sha256 = json_digest(self._identity)

    def load(self) -> dict | None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except UnicodeError as exc:
            raise ValueError("Cannot restore review checkpoint: invalid UTF-8") from exc
        except OSError as exc:
            raise OSError("Review checkpoint load could not read the file") from exc
        try:
            record = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            if not isinstance(record, dict) or set(record) != _ENVELOPE_FIELDS:
                raise ValueError("invalid checkpoint envelope fields")
            if type(record["schema_version"]) is not int or record["schema_version"] != self._schema:
                raise ValueError("unsupported checkpoint schema version")
            if (record["identity"] != self._identity
                    or record["identity_sha256"] != self._identity_sha256
                    or json_digest(record["identity"]) != self._identity_sha256):
                raise ValueError("checkpoint identity or identity hash differs")
            state = self._state_validator(record["state"])
            if json_digest(state) != record["state_sha256"]:
                raise ValueError("checkpoint state hash differs")
            body = {key: value for key, value in record.items() if key != "checkpoint_sha256"}
            if json_digest(body) != record["checkpoint_sha256"]:
                raise ValueError("checkpoint envelope hash differs")
            return state
        except (ValueError, TypeError, KeyError, RecursionError, UnicodeError) as exc:
            raise ValueError(f"Cannot restore review checkpoint: {exc}") from exc

    def save(self, state: dict) -> None:
        state = self._state_validator(state)
        # Do not silently reset another payload, or overwrite damaged evidence.
        previous = self.load()
        if previous is not None:
            if any(state[name] < previous[name] for name in self._counters):
                raise ValueError("Review checkpoint save cannot decrease spent counters")
            calls = previous["model_call_ids"]
            if state["model_call_ids"][:len(calls)] != calls:
                raise ValueError("Review checkpoint save cannot remove/reorder prior model_call_ids")
        body = {"schema_version": self._schema, "identity": deepcopy(self._identity),
                "identity_sha256": self._identity_sha256,
                "state": state, "state_sha256": json_digest(state)}
        body["checkpoint_sha256"] = json_digest(body)
        try:
            atomic_json(self.path, body)
        except OSError as exc:
            raise OSError("Review checkpoint save could not be persisted") from exc

    def clear(self) -> None:
        """Remove only a valid checkpoint bound to this exact identity."""
        if self.load() is None:
            return
        try:
            self.path.unlink()
        except OSError as exc:
            raise OSError("Review checkpoint clear could not remove the file") from exc


def _native_state(value):
    state = _object(value, "native state")
    if set(state) != {"status", "session_index", "lookup_state", "model_call_ids",
                      "iterations_spent", "tool_calls_spent"}:
        raise ValueError("Invalid native review checkpoint fields")
    if state["status"] not in {"ready", "started", "returned"}:
        raise ValueError("Invalid native review stage")
    index = state["session_index"]
    if ((state["status"] == "ready" and index is not None)
            or (state["status"] != "ready" and (type(index) is not int or index < 0))):
        raise ValueError("Native review stage must bind its real session index")
    if type(state["lookup_state"]) is not dict:
        raise ValueError("Native lookup state must be an object")
    for key in ("iterations_spent", "tool_calls_spent"):
        if type(state[key]) is not int or state[key] < 0:
            raise ValueError("Native review counters must be nonnegative integers")
    calls = state["model_call_ids"]
    if (type(calls) is not list or any(type(v) is not str or not v.strip() for v in calls)
            or len(calls) != len(set(calls)) or len(calls) > state["iterations_spent"]):
        raise ValueError("Native review call IDs must bind spent attempts")
    return state


class NativeReviewCheckpoint(ReviewCheckpoint):
    """Stage metadata only; never restores/replays an opaque SDK checkpoint.

    A started session without a durable returned SDK outcome is a blocker,
    even when an HTTP receipt exists. The workflow must reconcile all call IDs,
    immutable assignment, source and lookup state before continuing.
    """
    _schema = 3
    _state_validator = staticmethod(_native_state)
    _counters = ("iterations_spent", "tool_calls_spent")


class ReviewSplitPlan:
    """Write-once routing metadata, never a parent verdict or decision cache.

    Child verdicts belong to the existing ReviewJournal. Workflow revalidates
    the failed call and recomputes exact children from frozen source and draft.
    Loading a plan alone grants no source coverage.
    """
    def __init__(self, path, identity):
        self.path = Path(path)
        self.identity = _object(identity, "split identity")

    def load(self):
        if not self.path.exists():
            return None
        try:
            record = json.loads(self.path.read_text(encoding="utf-8"),
                                object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            if (type(record) is not dict or set(record) != {
                    "schema_version", "identity", "failed_attempt", "children", "plan_sha256"}
                    or type(record["schema_version"]) is not int or record["schema_version"] != 1
                    or record["identity"] != self.identity
                    or type(record["failed_attempt"]) is not dict
                    or type(record["children"]) is not list or len(record["children"]) != 2
                    or any(type(child) is not dict for child in record["children"])):
                raise ValueError("invalid split identity/schema")
            _json_value(record, "split plan")
            if json_digest({k:v for k,v in record.items() if k!="plan_sha256"}) != record["plan_sha256"]:
                raise ValueError("split plan checksum differs")
            return deepcopy(record)
        except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
            raise ValueError("Invalid or incompatible reverse length split plan") from exc

    def save(self, failed_attempt, children):
        body = {"schema_version":1, "identity":deepcopy(self.identity),
                "failed_attempt":_object(failed_attempt,"failed attempt"),
                "children":deepcopy(children)}
        _json_value(body,"split plan")
        if type(children) is not list or len(children)!=2 or any(type(child) is not dict for child in children):
            raise ValueError("Split plan needs exactly two complete child payloads")
        body["plan_sha256"] = json_digest(body)
        previous = self.load()
        if previous is not None:
            if previous != body:
                raise ValueError("Split plan refuses to overwrite a different original attempt")
            return
        atomic_json(self.path, body)
