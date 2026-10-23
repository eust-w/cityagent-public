"""Atomic, content-bound review decisions for one workflow worker.

Checksums detect inconsistent or damaged records; they are not cryptographic
signatures against an actor able to rewrite the file and all of its hashes.
Semantic scope and evidence-range validation remain the workflow's job.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from .run_state import atomic_json, json_digest


_SCHEMA = 1
_ARRAYS = ("checked_output_paths", "checked_evidence_spans", "unsupported_claims",
           "missing_items", "conflicts", "unresolved_items")
_ISSUES = _ARRAYS[2:]
_COMPACT_PROTOCOLS = {"bidirectional-semantic-review-v3", "bidirectional-semantic-review-v4", "bidirectional-semantic-review-v5", "bidirectional-semantic-review-v6"}


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Review journal {name} must be a nonempty string")
    return value


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"Review journal {name} must be a JSON object")
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Review journal {name} must contain valid JSON values") from exc


def _decision(value):
    result = _object(value, "decision")
    _string(result.get("model_call_id"), "decision.model_call_id")
    if type(result.get("passed")) is not bool:
        raise ValueError("Review journal decision.passed must be a boolean")
    assignment = result.get("review_assignment")
    source_protocol = isinstance(assignment, dict) and assignment.get("protocol") in {"bidirectional-semantic-review-v5", "bidirectional-semantic-review-v6"}
    issues = (*_ISSUES, "citation_gaps") if source_protocol else _ISSUES
    for name in issues if assignment is not None else _ARRAYS:
        if not isinstance(result.get(name), list):
            raise ValueError(f"Review journal decision.{name} must be an array")
    if result["passed"] and any(result[name] for name in issues):
        raise ValueError("Review journal decision cannot pass with unresolved issues")
    if assignment is None:
        if any(name in result for name in ("assignment_confirmation", "assigned_output_paths", "assigned_evidence_spans",
                                            "checked_assignment_id", "checked_all_assigned")):
            raise ValueError("Review journal assignment fields require a complete review_assignment")
        return result
    if not isinstance(assignment, dict):
        raise ValueError("Review journal review_assignment must be an object")
    required = {"id", "protocol", "workflow_sha256", "semantic_state_sha256", "payload_sha256",
                "draft_catalog_sha256", "output_item_count", "source_interval_count"}
    if source_protocol:
        required.add("source_snapshot_sha256")
    if set(assignment) != required or assignment["protocol"] not in _COMPACT_PROTOCOLS:
        raise ValueError("Review journal review_assignment has unsupported fields/protocol")
    hashes = ("workflow_sha256", "semantic_state_sha256", "payload_sha256", "draft_catalog_sha256")
    for name in (*hashes, "source_snapshot_sha256") if source_protocol else hashes:
        value = assignment[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Review journal assignment.{name} must be a SHA256")
    assignment_body = {key: value for key, value in assignment.items() if key != "id"}
    if assignment["id"] != "ra_" + json_digest(assignment_body)[:32]:
        raise ValueError("Review journal assignment ID hash differs")
    paths = result.get("assigned_output_paths")
    spans = result.get("assigned_evidence_spans")
    if (not isinstance(paths, list) or any(not isinstance(p, str) or not p for p in paths)
            or len(set(paths)) != len(paths) or not isinstance(spans, list)):
        raise ValueError("Review journal assigned scopes must be explicit distinct arrays")
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("Review journal assigned evidence span must be an object")
        if set(span) == {"image_id"}:
            _string(span["image_id"], "assigned image_id")
        elif (set(span) != {"evidence_id", "start", "end"}
              or not isinstance(span.get("evidence_id"), str) or not span["evidence_id"]
              or type(span.get("start")) is not int or type(span.get("end")) is not int
              or not 0 <= span["start"] < span["end"]):
            raise ValueError("Review journal assigned source interval is invalid")
    if len({json_digest(s) for s in spans}) != len(spans):
        raise ValueError("Review journal assigned source intervals contain duplicates")
    for name, expected in (("output_item_count", len(paths)), ("source_interval_count", len(spans))):
        if type(assignment[name]) is not int or assignment[name] != expected:
            raise ValueError("Review journal assignment scope counts differ")
    confirmation = result.get("assignment_confirmation")
    if confirmation not in {"compact", "legacy"}:
        raise ValueError("Review journal assignment confirmation mode is invalid")
    has_ack = "checked_assignment_id" in result or "checked_all_assigned" in result
    if confirmation == "compact" or has_ack:
        if result.get("checked_assignment_id") != assignment["id"] or result.get("checked_all_assigned") is not True:
            raise ValueError("Review journal compact acknowledgement is missing or differs")
    for key, expected in (("checked_output_paths", paths), ("checked_evidence_spans", spans)):
        if confirmation == "legacy" or key in result:
            actual = result.get(key)
            if (not isinstance(actual, list) or len(actual) != len(expected)
                    or {json_digest(i) for i in actual} != {json_digest(i) for i in expected}):
                raise ValueError("Review journal legacy echo contradicts assigned scopes")
    return result


def _now():
    return datetime.now(timezone.utc).isoformat()


def _key(semantic_hash, payload_sha256):
    return json_digest({"semantic_hash": semantic_hash, "payload_sha256": payload_sha256})


class ReviewJournal:
    """One atomic file, with all attempts and all first-stored decisions retained."""

    def __init__(self, directory, identity: dict):
        self.directory = Path(directory)
        self.path = self.directory / "review_journal.json"
        self.identity = _object(identity, "identity")
        if self.path.exists():
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
                self._validate(state)
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                raise ValueError(f"Cannot restore review journal: {exc}") from exc
            self._state = state
        else:
            self._commit({"schema_version": _SCHEMA, "identity": self.identity,
                          "identity_sha256": json_digest(self.identity),
                          "active_attempt_id": None, "attempts": [], "records": {}}, "initialize")

    def _bound_decision(self, value, semantic_hash, payload_hash, payload=None):
        decision = _decision(value)
        assignment = decision.get("review_assignment")
        if assignment is None:
            if self.identity.get("protocol") in _COMPACT_PROTOCOLS:
                raise ValueError("Review journal compact decision requires program-assigned scope binding")
            return decision
        if (assignment["protocol"] != self.identity.get("protocol")
                or assignment["workflow_sha256"] != json_digest(self.identity.get("workflow"))
                or assignment["semantic_state_sha256"] != semantic_hash
                or assignment["payload_sha256"] != payload_hash):
            raise ValueError("Review journal assignment workflow/semantic/payload binding differs")
        source_protocol = assignment["protocol"] in {"bidirectional-semantic-review-v5", "bidirectional-semantic-review-v6"}
        if source_protocol and assignment["source_snapshot_sha256"] != self.identity.get("source_snapshot_sha256"):
            raise ValueError("Review journal original source binding differs")
        if payload is not None:
            paths = [item["path"] for item in payload.get("output_items", [])]
            spans = payload.get("source_spans", [])
            if (decision["assigned_output_paths"] != paths or decision["assigned_evidence_spans"] != spans):
                raise ValueError("Review journal assigned scopes differ from complete payload")
            # Reverse catalog contains the complete frozen draft, not just the
            # related output subset. Only Workflow can validate that catalog.
            if (assignment["protocol"] == "bidirectional-semantic-review-v3" and payload.get("direction") == "FORWARD"
                    and assignment["draft_catalog_sha256"] != json_digest(payload.get("output_items", []))):
                raise ValueError("Review journal forward catalog binding differs")
            # v4+ gives every bounded forward batch the full frozen draft
            # lookup. Its catalog is deliberately larger than this batch.
            # Workflow reconciles that catalog against the native request.
        return decision

    def _validate(self, state):
        fields = {"schema_version", "identity", "identity_sha256", "active_attempt_id",
                  "attempts", "records", "state_sha256"}
        if not isinstance(state, dict) or set(state) != fields:
            raise ValueError("invalid journal schema fields")
        if type(state["schema_version"]) is not int or state["schema_version"] != _SCHEMA:
            raise ValueError("unsupported journal schema version")
        if (state["identity"] != self.identity
                or state["identity_sha256"] != json_digest(self.identity)
                or json_digest(state["identity"]) != state["identity_sha256"]):
            raise ValueError("journal identity or identity hash differs")
        body = {key: value for key, value in state.items() if key != "state_sha256"}
        if state["state_sha256"] != json_digest(body):
            raise ValueError("journal state hash differs")
        if not isinstance(state["attempts"], list) or not isinstance(state["records"], dict):
            raise ValueError("invalid journal attempts or records")
        attempts = {}
        active = []
        for attempt in state["attempts"]:
            if not isinstance(attempt, dict) or set(attempt) != {
                    "attempt_id", "semantic_hash", "status", "started_at", "ended_at"}:
                raise ValueError("invalid journal attempt fields")
            attempt_id = _string(attempt["attempt_id"], "attempt_id")
            _string(attempt["semantic_hash"], "semantic_hash")
            _string(attempt["started_at"], "started_at")
            if attempt_id in attempts:
                raise ValueError("duplicate journal attempt_id")
            attempts[attempt_id] = attempt
            if attempt["status"] == "active":
                if attempt["ended_at"] is not None:
                    raise ValueError("active journal attempt has ended_at")
                active.append(attempt_id)
            elif attempt["status"] in {"finished", "superseded"}:
                _string(attempt["ended_at"], "ended_at")
            else:
                raise ValueError("invalid journal attempt status")
        expected_active = [state["active_attempt_id"]] if state["active_attempt_id"] is not None else []
        if active != expected_active or (active and state["attempts"][-1]["attempt_id"] != active[0]):
            raise ValueError("journal active attempt is inconsistent")
        for key, record in state["records"].items():
            if not isinstance(record, dict) or set(record) != {
                    "semantic_hash", "payload_sha256", "decision", "decision_sha256",
                    "attempt_id", "stored_at"}:
                raise ValueError("invalid journal decision record fields")
            _string(record["semantic_hash"], "semantic_hash")
            payload_hash = record["payload_sha256"]
            if (not isinstance(payload_hash, str) or len(payload_hash) != 64
                    or any(char not in "0123456789abcdef" for char in payload_hash)):
                raise ValueError("invalid journal payload hash")
            if key != _key(record["semantic_hash"], payload_hash):
                raise ValueError("journal payload key hash differs")
            decision = self._bound_decision(record["decision"], record["semantic_hash"], payload_hash)
            if record["decision_sha256"] != json_digest(decision):
                raise ValueError("journal decision hash differs")
            origin = attempts.get(record["attempt_id"])
            if origin is None or origin["semantic_hash"] != record["semantic_hash"]:
                raise ValueError("journal decision attempt binding differs")
            _string(record["stored_at"], "stored_at")

    def _commit(self, body, operation):
        sealed = deepcopy(body)
        sealed.pop("state_sha256", None)
        sealed["state_sha256"] = json_digest(sealed)
        self._validate(sealed)
        try:
            atomic_json(self.path, sealed)
        except OSError as exc:
            raise OSError(f"Review journal {operation} could not be persisted") from exc
        self._state = sealed

    @property
    def attempt_count(self):
        return len(self._state["attempts"])

    def active(self, semantic_hash: str) -> dict | None:
        _string(semantic_hash, "semantic_hash")
        for attempt in self._state["attempts"]:
            if (attempt["attempt_id"] == self._state["active_attempt_id"]
                    and attempt["semantic_hash"] == semantic_hash):
                return deepcopy(attempt)
        return None

    def begin(self, semantic_hash: str) -> dict:
        _string(semantic_hash, "semantic_hash")
        current = self.active(semantic_hash)
        if current is not None:
            return {"attempt_id": current["attempt_id"], "started_new": False}
        state = deepcopy(self._state)
        now = _now()
        for attempt in state["attempts"]:
            if attempt["status"] == "active":
                attempt.update(status="superseded", ended_at=now)
        attempt_id = "review-" + uuid4().hex
        state["attempts"].append({"attempt_id": attempt_id, "semantic_hash": semantic_hash,
                                  "status": "active", "started_at": now, "ended_at": None})
        state["active_attempt_id"] = attempt_id
        self._commit(state, "begin")
        return {"attempt_id": attempt_id, "started_new": True}

    def get(self, semantic_hash: str, payload: dict) -> dict | None:
        _string(semantic_hash, "semantic_hash")
        payload_hash = json_digest(_object(payload, "payload"))
        record = self._state["records"].get(_key(semantic_hash, payload_hash))
        return self._bound_decision(record["decision"], semantic_hash, payload_hash, payload) if record is not None else None

    def unchanged_forward(self, semantic_hash: str, payload: dict) -> dict | None:
        """Find a clean prior verdict for the exact same output dependency closure.

        This is only a candidate. Workflow must revalidate the original native
        SDK request and final response using the ORIGINAL assignment. It must
        not mint a new assignment or store that old verdict under a new state.
        Reverse omission review depends on the full draft and is never reused
        here. Findings are also not reused across semantic changes.
        """
        if payload.get("direction") != "FORWARD":
            return None
        payload_hash = json_digest(_object(payload, "payload"))
        for record in reversed(list(self._state["records"].values())):
            if record["semantic_hash"] == semantic_hash or record["payload_sha256"] != payload_hash:
                continue
            decision = self._bound_decision(record["decision"], record["semantic_hash"], payload_hash, payload)
            if (decision.get("passed") is True and decision.get("review_assignment")
                    and all(not decision.get(name) for name in _ISSUES)):
                return {"semantic_hash": record["semantic_hash"], "decision": decision}
        return None

    def put(self, semantic_hash: str, payload: dict, decision) -> None:
        _string(semantic_hash, "semantic_hash")
        payload_hash = json_digest(_object(payload, "payload"))
        value = self._bound_decision(decision, semantic_hash, payload_hash, payload)
        key = _key(semantic_hash, payload_hash)
        previous = self._state["records"].get(key)
        if previous is not None:
            if previous["decision_sha256"] != json_digest(value):
                raise ValueError("Review journal put refuses to overwrite a different decision for the same key")
            return  # Idempotent even if the originating attempt already finished.
        current = self.active(semantic_hash)
        if current is None:
            raise RuntimeError("Review journal put requires an active attempt for this semantic hash")
        state = deepcopy(self._state)
        state["records"][key] = {"semantic_hash": semantic_hash, "payload_sha256": payload_hash,
                                  "decision": value, "decision_sha256": json_digest(value),
                                  "attempt_id": current["attempt_id"], "stored_at": _now()}
        self._commit(state, "put")

    def finish(self, attempt_id: str) -> None:
        _string(attempt_id, "attempt_id")
        state = deepcopy(self._state)
        for attempt in state["attempts"]:
            if attempt["attempt_id"] == attempt_id:
                if attempt["status"] == "finished":
                    return
                if state["active_attempt_id"] != attempt_id:
                    raise RuntimeError("Review journal finish requires the active attempt; this attempt was superseded")
                attempt.update(status="finished", ended_at=_now())
                state["active_attempt_id"] = None
                self._commit(state, "finish")
                return
        raise ValueError("Review journal finish received an unknown attempt_id")
