"""Durable scheduling metadata for bounded semantic review.

This file records work, never a semantic verdict. Completed entries are only
references to independently validated model decisions; the workflow revalidates
their native request/response provenance before any reuse or final acceptance.
"""
from copy import deepcopy
import json

from .run_state import atomic_json, json_digest


VERSION = "bounded-native-review-batches-v1"


def _scope(payload):
    return {"payload_sha256": json_digest(payload), "direction": payload["direction"],
            "output_paths": [item["path"] for item in payload.get("output_items", [])],
            "source_spans": deepcopy(payload.get("source_spans", []))}


class ReviewBatchPlan:
    def __init__(self, path, identity, payloads):
        self.path = path
        self.identity = deepcopy(identity)
        self.scopes = [_scope(payload) for payload in payloads]
        if len({scope["payload_sha256"] for scope in self.scopes}) != len(self.scopes):
            raise ValueError("Review batch plan contains duplicate scopes")
        if path.exists():
            self.state = json.loads(path.read_text(encoding="utf-8"))
            self._validate(self.state)
        else:
            self._commit({"schema_version": 1, "identity": self.identity,
                          "scopes": self.scopes, "routes": {}, "completed": {}, "blocked": None})

    def _validate(self, state):
        if (type(state) is not dict or set(state) != {
                "schema_version", "identity", "scopes", "routes", "completed", "blocked", "state_sha256"}
                or state["schema_version"] != 1 or state["identity"] != self.identity
                or state["scopes"] != self.scopes
                or state["state_sha256"] != json_digest({k: v for k, v in state.items() if k != "state_sha256"})):
            raise ValueError("Review batch plan identity/scope/checksum differs")
        allowed = {scope["payload_sha256"] for scope in self.scopes}
        if type(state["routes"]) is not dict or set(state["routes"]) - allowed:
            raise ValueError("Review batch plan has invalid split routes")
        for parent, children in state["routes"].items():
            if parent in state["completed"] or type(children) is not list or len(children) != 2:
                raise ValueError("Review batch plan cannot complete a delegated parent")
            for child in children:
                if (type(child) is not dict or set(child) != {
                        "payload_sha256", "direction", "output_paths", "source_spans"}
                        or child["direction"] != "REVERSE" or child["output_paths"]):
                    raise ValueError("Review batch plan has invalid child scope")
                allowed.add(child["payload_sha256"])
        if type(state["completed"]) is not dict or set(state["completed"]) - allowed:
            raise ValueError("Review batch plan has invalid completed scopes")
        for reference in state["completed"].values():
            if (type(reference) is not dict or set(reference) != {
                    "decision_sha256", "model_call_ids", "origin_semantic_state_sha256"}
                    or not reference["model_call_ids"]
                    or type(reference["model_call_ids"]) is not list
                    or any(type(call) is not str or not call for call in reference["model_call_ids"])):
                raise ValueError("Review batch plan has invalid model references")
        blocked = state["blocked"]
        if blocked is not None and (type(blocked) is not dict
                or set(blocked) != {"payload_sha256", "error_type"}
                or blocked["payload_sha256"] not in allowed
                or type(blocked["error_type"]) is not str):
            raise ValueError("Review batch plan has invalid blocker")

    def _commit(self, body):
        state = deepcopy(body)
        state.pop("state_sha256", None)
        state["state_sha256"] = json_digest(state)
        self._validate(state)
        atomic_json(self.path, state)
        self.state = state

    def complete(self, payload, decision):
        key = json_digest(payload)
        reference = {"decision_sha256": json_digest(decision),
                     "model_call_ids": list(decision.get("model_call_ids", [decision["model_call_id"]])),
                     "origin_semantic_state_sha256": decision["review_assignment"]["semantic_state_sha256"]}
        state = deepcopy(self.state)
        if key in state["routes"]:
            raise ValueError("Review batch plan cannot invent a delegated parent verdict")
        old = state["completed"].get(key)
        if old is not None and old != reference:
            raise ValueError("Review batch plan cannot replace an accepted decision reference")
        state["completed"][key] = reference
        if state["blocked"] and state["blocked"]["payload_sha256"] == key:
            state["blocked"] = None
        self._commit(state)

    def split(self, parent, children):
        """Routing only; caller must validate the actual length/split proof."""
        key = json_digest(parent)
        scopes = [_scope(payload) for payload in children]
        state = deepcopy(self.state)
        previous = state["routes"].get(key)
        if previous is not None and previous != scopes:
            raise ValueError("Review batch plan split children changed")
        state["routes"][key] = scopes
        if state["blocked"] and state["blocked"]["payload_sha256"] == key:
            state["blocked"] = None
        self._commit(state)

    def leaf_scopes(self):
        return [child for scope in self.scopes
                for child in self.state["routes"].get(scope["payload_sha256"], [scope])]

    def block(self, payload, error):
        state = deepcopy(self.state)
        state["blocked"] = {"payload_sha256": json_digest(payload), "error_type": type(error).__name__}
        self._commit(state)

    def progress(self):
        pending = [scope["payload_sha256"] for scope in self.leaf_scopes()
                   if scope["payload_sha256"] not in self.state["completed"]]
        return {"plan_sha256": self.state["state_sha256"], "batch_count": len(self.leaf_scopes()),
                "delegated_parent_count": len(self.state["routes"]),
                "completed_count": len(self.state["completed"]), "pending_count": len(pending),
                "pending_payload_sha256": pending, "blocked": deepcopy(self.state["blocked"]),
                "is_semantic_verdict": False}
