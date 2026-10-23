"""Candidate source-object disposition ledger; no model, filesystem source reads or I/O except its ledger.

A trusted caller supplies the complete occurrence inventory, actual-read intervals,
workflow identity and a LOCAL receipt_lookup(call_id). That lookup must read the
real gateway's terminal ledger and its original request/response sidecars. It must
not synthesize success from an Agent's statement. Synthetic tests are not evidence
of a real model review. Checksums detect damage, not an attacker rewriting all
records and the trusted ledger. One worker only; no concurrent writer support.

This module checks identity, structure and receipt bindings. Only the independent
own-model review decides semantic relevance, support, omissions or contradictions.
It never changes the source manifest's unverified_content or actual-read ledger.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Callable

from .run_state import atomic_json, json_digest

PROTOCOL = "source-object-dispositions-v1"
SCHEMA = 1
DISPOSITIONS = {"used_as_evidence", "not_relied_on", "outside_task", "unresolved_relevant", "unclassified"}
NONBLOCKING = {"used_as_evidence", "not_relied_on", "outside_task"}
ISSUES = ("unsupported_claims", "missing_items", "conflicts", "unresolved_items")


STREAM_RECEIPT_FIELDS = frozenset({"call_id", "request_id", "kind", "model", "http_status", "status",
    "generation_attempted", "done_received", "usage_known", "finish_reason", "input_sha256",
    "response_sha256", "output_sha256", "raw_sse_sha256", "response_id", "usage"})


def validate_source_transport(audit, request, response):
    """Accept SSE only with the trusted caller's bound complete transport receipt."""
    stream = request.get("stream")
    if stream is None or stream is False:
        return
    if stream is not True:
        raise ValueError("Source request stream flag must be boolean")
    receipt = audit.get("stream_transport_receipt")
    if type(receipt) is not dict or set(receipt) != STREAM_RECEIPT_FIELDS:
        raise ValueError("Source SSE response requires a complete trusted transport receipt")
    choices = response.get("choices")
    usage = receipt.get("usage")
    if (receipt.get("call_id") != audit.get("call_id")
            or receipt.get("request_id") != audit.get("request_id")
            or receipt.get("kind") != audit.get("request_kind")
            or receipt.get("model") != audit.get("model") or receipt.get("model") != response.get("model")
            or receipt.get("status") != "success" or receipt.get("http_status") != 200
            or receipt.get("generation_attempted") is not True
            or receipt.get("done_received") is not True or receipt.get("usage_known") is not True
            or receipt.get("input_sha256") != audit.get("request_sha256")
            or receipt.get("input_sha256") != json_digest(request)
            or receipt.get("response_sha256") != audit.get("response_sha256")
            or receipt.get("response_sha256") != json_digest(response)
            or receipt.get("output_sha256") != receipt.get("response_sha256")
            or type(receipt.get("raw_sse_sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", receipt["raw_sse_sha256"]) is None
            or type(receipt.get("response_id")) is not str or not receipt["response_id"]
            or receipt["response_id"] != response.get("id")
            or type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict
            or receipt.get("finish_reason") not in {"stop", "tool_calls"}
            or receipt["finish_reason"] != audit.get("finish_reason")
            or receipt["finish_reason"] != choices[0].get("finish_reason")
            or type(usage) is not dict or usage != response.get("usage") or usage != audit.get("usage")
            or any(type(usage.get(key)) is not int or usage[key] < 0
                   for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
        raise ValueError("Source SSE response lacks bound DONE, complete usage or matching wire evidence")


def _json(value):
    # JSON round-trip alone silently accepts non-string keys/tuples: reject them.
    def visit(v):
        if v is None or type(v) in (str, bool, int):
            return
        if type(v) is float:
            if not math.isfinite(v):
                raise ValueError("Object disposition requires finite JSON")
            return
        if type(v) is list:
            for item in v:
                visit(item)
            return
        if type(v) is dict and all(type(k) is str for k in v):
            for item in v.values():
                visit(item)
            return
        raise ValueError("Object disposition requires strict JSON values")
    visit(value)
    return deepcopy(value)


def _keys(value, keys, name):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"Invalid {name} fields")


def _str(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(f"Invalid nonempty {name}")
    return value


def _sha(value, name):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid SHA256 {name}")
    return value


def _strings(value, name, *, nonempty=False):
    if type(value) is not list or (nonempty and not value):
        raise ValueError(f"Invalid {name} array")
    for item in value:
        _str(item, name)
    if len(set(value)) != len(value):
        raise ValueError(f"Duplicate {name}")
    return value


def occurrence_id(occurrence: dict) -> str:
    """Stable identity includes source position, bytes and context, never just raw SHA."""
    value = _json(occurrence)
    _keys(value, {"source_sha256", "locator", "kind", "raw_sha256", "native_sha256", "cache_sha256", "context_refs"}, "occurrence")
    for key in ("source_sha256", "raw_sha256"):
        _sha(value[key], key)
    for key in ("native_sha256", "cache_sha256"):
        if value[key] is not None:
            _sha(value[key], key)
    _str(value["locator"], "locator")
    if value["kind"] not in {"ole", "image", "formula", "unsupported"}:
        raise ValueError("Unsupported source object kind")
    if type(value["context_refs"]) is not list:
        raise ValueError("Occurrence context_refs must be an array")
    return "so_" + json_digest(value)[:32]


def _unread():
    return {"state": "unread", "content": None, "semantic_uncertainties": [],
            "glyph_fidelity": "unverified", "inspection": {}, "origin_call_ids": []}


def _representation(value):
    value = _json(value)
    _keys(value, _unread(), "representation")
    if value["state"] not in {"unread", "candidate", "semantics_verified"}:
        raise ValueError("Invalid representation state")
    if value["glyph_fidelity"] not in {"verified", "unverified", "not_applicable"}:
        raise ValueError("Invalid glyph fidelity")
    _strings(value["semantic_uncertainties"], "semantic uncertainties")
    _strings(value["origin_call_ids"], "representation origin calls")
    if type(value["inspection"]) is not dict:
        raise ValueError("Representation inspection must retain evidence as an object")
    if value["state"] == "unread" and (value["content"] is not None or value["inspection"] or value["glyph_fidelity"] != "unverified"):
        raise ValueError("Unread representation cannot claim inspected content/glyphs")
    if value["state"] != "unread" and value["content"] is None:
        raise ValueError("Parsed representation must preserve candidate content")
    if value["state"] == "semantics_verified" and (value["semantic_uncertainties"] or not value["inspection"]):
        raise ValueError("Verified representation requires inspection evidence and no unresolved semantic symbols")
    return value


class SourceObjectDispositions:
    """Append-only, atomically persisted single-worker review ledger.

    identity requires model and source_sha256; include the full workflow/model
    identity here. evidence_units is the immutable {evidence_id: text} source
    catalog, not labels. read_refs must come from actual tool-delivery receipts.
    A representation inspection is material sent to the independent reviewer,
    not proof that a candidate parser understood original glyphs.
    """

    def __init__(self, directory: Path, identity: dict, occurrences: list[dict],
                 evidence_units: dict[str, str], receipt_lookup: Callable[[str], dict]):
        self.path = Path(directory) / "source_object_dispositions.json"
        self.identity = _json(identity)
        if type(self.identity) is not dict:
            raise ValueError("Identity must be an object")
        _str(self.identity.get("model"), "identity model")
        _sha(self.identity.get("source_sha256"), "identity source")
        if not callable(receipt_lookup):
            raise ValueError("A trusted local receipt_lookup is required")
        self.receipt_lookup = receipt_lookup
        self.units = _json(evidence_units)
        if type(self.units) is not dict or any(type(v) is not str for v in self.units.values()):
            raise ValueError("Evidence units must map IDs to original strings")
        for key in self.units:
            _str(key, "evidence_id")
        self.objects = {}
        locators = set()
        if type(occurrences) is not list:
            raise ValueError("Complete occurrence inventory must be an array")
        for occurrence in _json(occurrences):
            oid = occurrence_id(occurrence)
            if occurrence["source_sha256"] != self.identity["source_sha256"]:
                raise ValueError("Occurrence source differs from ledger identity")
            if occurrence["locator"] in locators or oid in self.objects:
                raise ValueError("Duplicate occurrence position")
            locators.add(occurrence["locator"])
            self._refs(occurrence["context_refs"])
            self.objects[oid] = occurrence
        self.header = {"schema_version": SCHEMA, "protocol": PROTOCOL, "identity": self.identity,
                       "occurrences": self.objects, "evidence_units_sha256": json_digest(self.units)}
        self.events = []
        self._reset()
        if self.path.exists():
            self._load()
        else:
            self._write([])

    def _reset(self):
        self.context = None
        self.context_revision = 0
        self.representations = {oid: _unread() for oid in self.objects}
        self.representation_revision = 0
        self.proposals = {}
        self.reviews = {}

    def _refs(self, refs, *, must_read=False):
        if type(refs) is not list:
            raise ValueError("References must be an array")
        seen = set()
        for ref in refs:
            _keys(ref, {"evidence_id", "start", "end"}, "source reference")
            text = self.units.get(ref["evidence_id"])
            if (text is None or type(ref["start"]) is not int or type(ref["end"]) is not int
                    or not 0 <= ref["start"] < ref["end"] <= len(text)):
                raise ValueError("Source reference is outside the original unit")
            key = json_digest(ref)
            if key in seen:
                raise ValueError("Duplicate source reference")
            seen.add(key)
            if must_read:
                if self.context is None:
                    raise ValueError("Bind context before proposing dispositions")
                intervals = sorted((r["start"], r["end"]) for r in self.context["read_refs"]
                                   if r["evidence_id"] == ref["evidence_id"])
                cursor = ref["start"]
                for start, end in intervals:
                    if start <= cursor:
                        cursor = max(cursor, end)
                if cursor < ref["end"]:
                    raise ValueError("Disposition reference has not actually been read")
        return refs

    def _context(self, value):
        value = _json(value)
        _keys(value, {"task", "draft", "facts", "output_paths", "read_refs", "catalog_sha256"}, "context")
        for key in ("task", "draft", "facts"):
            if type(value[key]) is not dict:
                raise ValueError(f"Context {key} must preserve a JSON object")
        if not value["task"]:
            raise ValueError("Task identity/scope must be explicit")
        _strings(value["output_paths"], "output paths")
        self._refs(value["read_refs"])
        _sha(value["catalog_sha256"], "frozen catalog")
        return value

    def bind_context(self, *, task: dict, draft: dict, facts: dict, output_paths: list[str],
                     read_refs: list[dict], catalog_sha256: str) -> str:
        value = self._context(dict(task=task, draft=draft, facts=facts, output_paths=output_paths,
                                   read_refs=read_refs, catalog_sha256=catalog_sha256))
        if value != self.context:
            self._append("context", value)
        return self.scope_sha256

    @property
    def scope_sha256(self):
        if self.context is None:
            raise ValueError("Bind context before using object dispositions")
        return json_digest({"header_sha256": json_digest(self.header), "context": self.context,
                            "context_revision": self.context_revision, "representations": self.representations,
                            "representation_revision": self.representation_revision})

    def _object(self, object_id):
        if object_id not in self.objects:
            raise ValueError("Unknown object occurrence")
        return self.objects[object_id]

    def representation_review_payload(self, object_id: str, representation: dict) -> dict:
        occurrence = self._object(object_id)
        value = _representation(representation)
        if value["state"] != "semantics_verified":
            raise ValueError("Representation review must explicitly propose semantics_verified")
        body = {"protocol": PROTOCOL, "purpose": "source_representation_review",
                "identity_sha256": json_digest(self.header), "object_id": object_id,
                "occurrence": occurrence, "representation": value}
        return {**deepcopy(body), "assignment_id": "sr_" + json_digest(body)[:32]}

    def set_representation(self, object_id: str, representation: dict, *, verification_call_id: str | None = None):
        self._object(object_id)
        value = _representation(representation)
        if value == self.representations[object_id]:
            return
        verification = None
        if value["state"] == "semantics_verified":
            assignment = self.representation_review_payload(object_id, value)
            decision, receipt = self._receipt(verification_call_id, assignment, set(value["origin_call_ids"]))
            verification = {"decision": decision, "receipt": receipt}
        payload = {"object_id": object_id, "representation": value, "verification_call_id": verification_call_id,
                   "verification": verification}
        self._append("representation", payload)

    def _proposal(self, object_id, proposal):
        self._object(object_id)
        value = _json(proposal)
        _keys(value, {"disposition", "rationale", "affected_output_paths", "supporting_refs", "context_refs",
                      "source_limitations", "origin_call_ids"}, "proposal")
        if value["disposition"] not in DISPOSITIONS:
            raise ValueError("Unknown task disposition")
        _str(value["rationale"], "proposal rationale")
        _strings(value["affected_output_paths"], "affected output paths")
        _strings(value["source_limitations"], "source limitations")
        _strings(value["origin_call_ids"], "proposal origin calls", nonempty=True)
        self._refs(value["supporting_refs"], must_read=True)
        self._refs(value["context_refs"], must_read=True)
        if self.context is None:
            raise ValueError("Bind context before proposing dispositions")
        if set(value["affected_output_paths"]) - set(self.context["output_paths"]):
            raise ValueError("Proposal refers to an unknown output path")
        state = self.representations[object_id]["state"]
        if state != "semantics_verified" and not value["source_limitations"]:
            raise ValueError("Unverified source representation must retain explicit limitations")
        mode = value["disposition"]
        if mode == "used_as_evidence" and (state != "semantics_verified" or not value["affected_output_paths"]):
            raise ValueError("used_as_evidence requires verified semantics and explicit output dependencies")
        if mode == "not_relied_on" and (not value["supporting_refs"] or not value["affected_output_paths"]):
            raise ValueError("not_relied_on requires independent support and explicit affected output paths")
        if mode in NONBLOCKING:
            occurrence_context = self.objects[object_id]["context_refs"]
            if not value["context_refs"] or not occurrence_context:
                raise ValueError("Nonblocking dispositions require actually-read object context")
            for ref in occurrence_context:
                cursor = ref["start"]
                for start, end in sorted((r["start"], r["end"]) for r in value["context_refs"]
                                         if r["evidence_id"] == ref["evidence_id"]):
                    if start <= cursor:
                        cursor = max(cursor, end)
                if cursor < ref["end"]:
                    raise ValueError("Disposition must include the complete original occurrence context")
        return value

    def propose(self, object_id: str, proposal: dict) -> str:
        value = self._proposal(object_id, proposal)
        body = {"object_id": object_id, "scope_sha256": self.scope_sha256, "proposal": value}
        proposal_id = "sp_" + json_digest({"sequence": len(self.events) + 1, **body})[:32]
        self._append("proposal", {"proposal_id": proposal_id, **body})
        return proposal_id

    def _current(self, proposal_id):
        row = self.proposals.get(proposal_id)
        if row is None:
            raise ValueError("Unknown proposal")
        if row["scope_sha256"] != self.scope_sha256:
            raise ValueError("Proposal is stale after task/draft/facts/catalog/read/representation change")
        latest = [r for r in self.proposals.values() if r["object_id"] == row["object_id"]
                  and r["scope_sha256"] == self.scope_sha256]
        if latest[-1]["proposal_id"] != proposal_id:
            raise ValueError("Proposal was superseded by a later proposal for this occurrence")
        return row

    def _proposal_ids(self, value):
        ids = [value] if isinstance(value, str) else value
        _strings(ids, "proposal IDs", nonempty=True)
        rows = [self._current(pid) for pid in ids]
        if len({r["object_id"] for r in rows}) != len(rows):
            raise ValueError("Review batch repeats an object occurrence")
        return rows

    def review_payload(self, proposal_ids: str | list[str]) -> dict:
        rows = self._proposal_ids(proposal_ids)
        materials = []
        for row in rows:
            refs = row["proposal"]["context_refs"] + row["proposal"]["supporting_refs"]
            source = [{**ref, "text": self.units[ref["evidence_id"]][ref["start"]:ref["end"]]} for ref in refs]
            materials.append({**deepcopy(row), "occurrence": deepcopy(self.objects[row["object_id"]]),
                              "representation": deepcopy(self.representations[row["object_id"]]), "source": source})
        body = {"protocol": PROTOCOL, "purpose": "source_object_disposition_review", "scope_sha256": self.scope_sha256,
                "task": deepcopy(self.context["task"]), "draft": deepcopy(self.context["draft"]),
                "facts": deepcopy(self.context["facts"]), "catalog_sha256": self.context["catalog_sha256"],
                "proposals": materials}
        return {**body, "assignment_id": "sd_" + json_digest(body)[:32]}

    def _receipt(self, call_id, assignment, forbidden_calls):
        _str(call_id, "model call ID")
        if call_id in forbidden_calls:
            raise ValueError("Disposition review must be independent of proposal/representation generation")
        try:
            bundle = _json(self.receipt_lookup(call_id))
        except Exception:
            raise ValueError("Trusted local model receipt is unavailable") from None
        _keys(bundle, {"audit", "request", "response"}, "trusted receipt bundle")
        audit, request, response = bundle["audit"], bundle["request"], bundle["response"]
        if any(type(item) is not dict for item in (audit, request, response)):
            raise ValueError("Trusted receipt bundle must contain JSON objects")
        if (audit.get("call_id") != call_id or audit.get("status") != "success"
                or audit.get("generation_attempted") is not True or audit.get("response_validated") is not True
                or audit.get("http_status") != 200 or audit.get("finish_reason") != "stop"
                or audit.get("request_kind") != "generation"
                or audit.get("endpoint") != "/v1/chat/completions"
                or audit.get("model") != self.identity["model"]):
            raise ValueError("A terminal successful independent own-model generation receipt is required")
        _str(audit.get("request_id"), "real dispatch request ID")
        if audit.get("request_sha256") != json_digest(request) or audit.get("response_sha256") != json_digest(response):
            raise ValueError("Model request/response SHA256 differs from terminal receipt")
        if request.get("model") != self.identity["model"]:
            raise ValueError("Review request must use the fixed own model")
        validate_source_transport(audit, request, response)
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
            raise ValueError("Review request has no final assignment message")
        try:
            actual_assignment = json.loads(messages[-1]["content"])
        except (ValueError, TypeError, KeyError):
            raise ValueError("Review assignment must be exact JSON in final user message") from None
        if messages[-1].get("role") != "user" or actual_assignment != assignment:
            raise ValueError("Actual model request does not contain the complete frozen assignment")
        choices = response.get("choices")
        if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
                or choices[0].get("finish_reason") != "stop"):
            raise ValueError("Review response must be one complete final choice")
        message = choices[0].get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant" or message.get("tool_calls"):
            raise ValueError("Review response must be a final assistant decision")
        try:
            decision = _json(json.loads(message["content"]))
        except (ValueError, TypeError, KeyError):
            raise ValueError("Review response is not a complete JSON decision") from None
        if type(decision) is not dict:
            raise ValueError("Review decision must be an object")
        return decision, {key: audit[key] for key in ("call_id", "request_id", "request_sha256", "response_sha256", "model")}

    def _review_decisions(self, proposal_ids, call_id):
        rows = self._proposal_ids(proposal_ids)
        assignment = self.review_payload(proposal_ids)
        forbidden = {cid for row in rows for cid in row["proposal"]["origin_call_ids"]}
        forbidden.update(cid for row in rows for cid in self.representations[row["object_id"]]["origin_call_ids"])
        decision, receipt = self._receipt(call_id, assignment, forbidden)
        _keys(decision, {"assignment_id", "decisions"}, "review decision envelope")
        if decision["assignment_id"] != assignment["assignment_id"] or type(decision["decisions"]) is not list:
            raise ValueError("Independent review assignment differs")
        by_id = {}
        for item in decision["decisions"]:
            _keys(item, {"proposal_id", "object_id", "verdict", "explanation", *ISSUES}, "object review decision")
            _str(item["explanation"], "review explanation")
            if item["verdict"] not in {"accepted", "rejected", "unresolved"}:
                raise ValueError("Unknown independent review verdict")
            if item["proposal_id"] in by_id:
                raise ValueError("Duplicate reviewed proposal")
            for name in ISSUES:
                if type(item[name]) is not list:
                    raise ValueError("Review issue arrays must be retained")
            by_id[item["proposal_id"]] = item
        if set(by_id) != {row["proposal_id"] for row in rows}:
            raise ValueError("Independent review must cover every assigned proposal exactly")
        for row in rows:
            item = by_id[row["proposal_id"]]
            if item["object_id"] != row["object_id"]:
                raise ValueError("Reviewed occurrence does not match the assignment")
            if item["verdict"] == "accepted" and row["proposal"]["disposition"] in NONBLOCKING and any(item[name] for name in ISSUES):
                raise ValueError("Accepted nonblocking disposition cannot have unresolved review issues")
        return deepcopy(decision), receipt

    def record_review(self, proposal_ids: str | list[str], model_call_id: str) -> dict:
        ids = [proposal_ids] if isinstance(proposal_ids, str) else deepcopy(proposal_ids)
        result, receipt = self._review_decisions(ids, model_call_id)
        payload = {"proposal_ids": ids, "model_call_id": model_call_id, "decision": result, "receipt": receipt}
        existing = [self.reviews.get(pid) for pid in ids]
        if any(existing):
            if all(item == payload for item in existing):
                return deepcopy(result)
            raise ValueError("Reviewed proposals are immutable; append a new proposal/review")
        self._append("review", payload)
        return deepcopy(result)

    def coverage(self) -> dict:
        scope = self.scope_sha256
        rows = []
        for oid in self.objects:
            proposals = [p for p in self.proposals.values() if p["object_id"] == oid and p["scope_sha256"] == scope]
            proposal = proposals[-1] if proposals else None
            review = self.reviews.get(proposal["proposal_id"]) if proposal else None
            decision = next((d for d in review["decision"]["decisions"] if d["object_id"] == oid), None) if review else None
            mode = proposal["proposal"]["disposition"] if proposal else "unclassified"
            verdict = decision["verdict"] if decision else ("proposed" if proposal else "unclassified")
            representation = self.representations[oid]["state"]
            gate = verdict == "accepted" and mode in NONBLOCKING
            rows.append({"object_id": oid, "representation_state": representation, "disposition": mode,
                         "review_status": verdict, "object_gate_passed": gate,
                         "proposal_id": proposal["proposal_id"] if proposal else None,
                         "model_call_id": review["model_call_id"] if review else None,
                         "source_limitations": deepcopy(proposal["proposal"]["source_limitations"]) if proposal else []})
        return {"scope_sha256": scope, "total_objects": len(rows), "objects": rows,
                "object_gate_passed": all(row["object_gate_passed"] for row in rows),
                "accepted_count": sum(row["review_status"] == "accepted" for row in rows),
                "blocking_count": sum(not row["object_gate_passed"] for row in rows),
                "unverified_representation_count": sum(row["representation_state"] != "semantics_verified" for row in rows),
                "all_source_semantics_verified": all(row["representation_state"] == "semantics_verified" for row in rows)}

    def completion(self, gates: dict) -> dict:
        """Completion recommendation, NOT export/publication and NOT a model-review substitute.

        gates must come from the existing trusted workflow/Word readback. In
        particular a partial result requires its own actual independent review;
        an exhausted budget never manufactures completed_limited.
        """
        names = {"audit_valid", "draft_valid", "native_sdk_observed", "finish_passed", "export_verified",
                 "task_review_passed", "partial_review_passed"}
        _keys(gates, names, "completion gates")
        if any(type(value) is not bool for value in gates.values()):
            raise ValueError("Completion gates must be explicit booleans")
        coverage = self.coverage()
        delivery = all(gates[name] for name in names - {"task_review_passed", "partial_review_passed"})
        if delivery and gates["task_review_passed"] and coverage["object_gate_passed"]:
            status = "completed"
        elif delivery and gates["partial_review_passed"]:
            status = "completed_limited"
        else:
            status = "failed" if not gates["audit_valid"] or not gates["draft_valid"] else "incomplete"
        return {"status": status, "completion_scope": "three_table_extraction", "task_complete": status == "completed",
                "formal_publish_eligible": status == "completed", "coverage": coverage,
                "all_source_semantics_verified": coverage["all_source_semantics_verified"],
                "source_limitations": [row for row in coverage["objects"] if row["source_limitations"]
                                       or row["representation_state"] != "semantics_verified"]}

    def history(self) -> list[dict]:
        return deepcopy(self.events)

    def _apply(self, kind, payload, sequence):
        if kind == "context":
            self.context = self._context(payload)
            self.context_revision += 1
        elif kind == "representation":
            _keys(payload, {"object_id", "representation", "verification_call_id", "verification"}, "representation event")
            oid = payload["object_id"]
            self._object(oid)
            value = _representation(payload["representation"])
            if value["state"] == "semantics_verified":
                assignment = self.representation_review_payload(oid, value)
                decision, receipt = self._receipt(payload["verification_call_id"], assignment, set(value["origin_call_ids"]))
                if payload["verification"] != {"decision": decision, "receipt": receipt}:
                    raise ValueError("Representation verification differs from the original model response")
                _keys(decision, {"assignment_id", "object_id", "representation_sha256", "semantics_verified", "explanation"}, "representation decision")
                if (decision["assignment_id"] != assignment["assignment_id"] or decision["object_id"] != oid
                        or decision["representation_sha256"] != json_digest(value) or decision["semantics_verified"] is not True):
                    raise ValueError("Independent representation review did not verify this exact occurrence/content")
                _str(decision["explanation"], "representation review explanation")
            elif payload["verification_call_id"] is not None or payload["verification"] is not None:
                raise ValueError("Unverified representation cannot attach a verification call")
            self.representations[oid] = value
            self.representation_revision += 1
        elif kind == "proposal":
            _keys(payload, {"proposal_id", "object_id", "scope_sha256", "proposal"}, "proposal event")
            if payload["scope_sha256"] != self.scope_sha256:
                raise ValueError("Proposal scope identity differs")
            self._proposal(payload["object_id"], payload["proposal"])
            body = {key: value for key, value in payload.items() if key != "proposal_id"}
            if payload["proposal_id"] != "sp_" + json_digest({"sequence": sequence, **body})[:32]:
                raise ValueError("Proposal ID differs from its immutable content")
            self.proposals[payload["proposal_id"]] = deepcopy(payload)
        elif kind == "review":
            _keys(payload, {"proposal_ids", "model_call_id", "decision", "receipt"}, "review event")
            decision, receipt = self._review_decisions(payload["proposal_ids"], payload["model_call_id"])
            if decision != payload["decision"] or receipt != payload["receipt"]:
                raise ValueError("Stored review differs from the actual model receipt/response")
            if any(pid in self.reviews for pid in payload["proposal_ids"]):
                raise ValueError("Ledger cannot overwrite existing object reviews")
            for pid in payload["proposal_ids"]:
                self.reviews[pid] = deepcopy(payload)
        else:
            raise ValueError("Unknown source-object ledger event")

    def _sealed(self, events):
        state = {"header": self.header, "events": events}
        return {**state, "state_sha256": json_digest(state)}

    def _write(self, events):
        atomic_json(self.path, self._sealed(events))

    def _append(self, kind, payload):
        payload = _json(payload)
        event = {"sequence": len(self.events) + 1, "kind": kind, "payload": payload,
                 "created_at": datetime.now(timezone.utc).isoformat(),
                 "previous_sha256": self.events[-1]["event_sha256"] if self.events else json_digest(self.header)}
        event["event_sha256"] = json_digest(event)
        # Validate in memory, write atomically, then publish state. A failed write
        # restores the already committed projection; no false accepted result.
        previous = self.history()
        try:
            self._apply(kind, payload, event["sequence"])
            self._write([*self.events, event])
        except Exception:
            self._reset()
            for old in previous:
                self._apply(old["kind"], old["payload"], old["sequence"])
            raise
        self.events.append(event)

    def _load(self):
        try:
            state = _json(json.loads(self.path.read_text(encoding="utf-8")))
            _keys(state, {"header", "events", "state_sha256"}, "ledger")
            body = {key: value for key, value in state.items() if key != "state_sha256"}
            if state["state_sha256"] != json_digest(body) or state["header"] != self.header:
                raise ValueError("Ledger identity/schema/content hash mismatch")
            if type(state["events"]) is not list:
                raise ValueError("Ledger events must be an array")
            previous = json_digest(self.header)
            for expected, event in enumerate(state["events"], 1):
                _keys(event, {"sequence", "kind", "payload", "created_at", "previous_sha256", "event_sha256"}, "ledger event")
                body = {key: value for key, value in event.items() if key != "event_sha256"}
                if type(event["sequence"]) is not int or event["sequence"] != expected or event["previous_sha256"] != previous or event["event_sha256"] != json_digest(body):
                    raise ValueError("Ledger history sequence/hash mismatch")
                _str(event["created_at"], "event timestamp")
                self._apply(event["kind"], event["payload"], expected)
                self.events.append(event)
                previous = event["event_sha256"]
        except (TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid source-object ledger structure") from exc


def count_completions(results: list[dict]) -> dict:
    """All attempts stay in the denominator; limited is never a complete report."""
    counts = Counter({"completed": 0, "completed_limited": 0, "failed": 0, "incomplete": 0})
    for result in results:
        status = result.get("status")
        if status not in counts or type(result.get("task_complete")) is not bool or result["task_complete"] != (status == "completed"):
            raise ValueError("Completion status and task_complete disagree")
        if result.get("completion_scope") != "three_table_extraction":
            raise ValueError("Cannot mix completion contracts in one throughput count")
        counts[status] += 1
    return {"attempted_reports": len(results), "complete_reports": counts["completed"],
            "limited_reports": counts["completed_limited"], "failed_reports": counts["failed"],
            "incomplete_reports": counts["incomplete"]}
