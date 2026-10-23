"""One structural submission: save model entries, then request a section handoff.

IDs, counts and handoff summaries are host bookkeeping. All report values,
evidence references and unresolved issues remain model-authored. A successful
save is not reading coverage, semantic approval or completed document delivery.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
import threading

from .document_entries import EntryStoreError, entry_schema
from .run_state import json_digest


def submit_section_schema() -> dict:
    """Keep the original entry contract, with host-assigned IDs when omitted."""
    item = entry_schema()
    for variant in item["anyOf"]:
        variant["required"].remove("id")
        variant["properties"]["id"]["description"] = (
            "Omit for a new entry: the program assigns a stable ID. "
            "Use an assigned ID only when intentionally correcting that saved entry.")
    return {"type": "object", "additionalProperties": False,
        "required": ["entries", "unresolved", "complete"], "properties": {
            "entries": {"type": "array", "items": item,
                "description": "Model-authored delivery entries with exact source references. An empty array only requests handoff; zero rows never establishes semantic completeness."},
            "unresolved": {"type": "array", "items": {"type": "string"},
                "description": "Unresolved questions stated by the model; [] means none reported, not automatic verification."},
            "complete": {"type": "boolean", "default": True,
                "description": "True saves then requests the existing checked handoff. False saves a batch and keeps this section open."}}}


def _path(parts) -> str:
    return "".join(f"[{part}]" if type(part) is int else ("." if i else "") + str(part)
                   for i, part in enumerate(parts)) or "$"


def _entry_example() -> dict:
    return {"section": "基本信息", "field": "桥梁名称", "value": "<原文中的值>",
            "source_refs": [{"evidence_id": "<阅读工具返回的原文编号>"}], "status": "explicit"}


class _SubmissionError(ValueError):
    def __init__(self, message, *, path, example, code="SECTION_SUBMISSION_ARGUMENTS"):
        super().__init__(message)
        self.path, self.example, self.code = path, example, code


def _strict_json(value, parts=()):
    """Reject non-JSON values before hashing or changing the durable store."""
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is str and not any(0xD800 <= ord(c) <= 0xDFFF for c in value):
        return
    if type(value) is list:
        for i, item in enumerate(value):
            _strict_json(item, (*parts, i))
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for key, item in value.items():
            _strict_json(key, (*parts, key))
            _strict_json(item, (*parts, key))
        return
    raise _SubmissionError("Use complete finite UTF-8 JSON values; no partial JSON is repaired.",
                           path=_path(parts), example=_entry_example())


def _validate_arguments(entries, unresolved, complete):
    from jsonschema import Draft202012Validator

    payload = {"entries": entries, "unresolved": unresolved, "complete": complete}
    _strict_json(payload)
    schema = submit_section_schema()
    # Narrow by the model's explicit section so a row error points to that
    # field, rather than returning only the anyOf entry-level failure.
    checks = [(payload, {**schema, "properties": {**schema["properties"],
        "entries": {"type": "array"}}}, ())]
    if isinstance(entries, list):
        variants = schema["properties"]["entries"]["items"]["anyOf"]
        for index, entry in enumerate(entries):
            variant = next((v for v in variants if isinstance(entry, dict)
                and entry.get("section") in v["properties"]["section"]["enum"]), None)
            checks.append((entry, variant or {"anyOf": variants}, ("entries", index)))
    for value, spec, prefix in checks:
        error = next(Draft202012Validator(spec).iter_errors(value), None)
        if error is None:
            continue
        location = [*prefix, *error.absolute_path]
        if error.validator == "required" and isinstance(error.instance, dict):
            location.append(next(key for key in error.validator_value if key not in error.instance))
        if error.validator == "additionalProperties" and isinstance(error.instance, dict):
            extra = sorted(set(error.instance) - set(error.schema.get("properties", {})))
            if extra:
                location.append(extra[0])
        examples = {"complete": True, "unresolved": ["<仍待核查的问题>"], "entries": [_entry_example()]}
        raise _SubmissionError("Submission field does not match its required structural schema.",
            path=_path(location), example=examples.get(location[0] if location else "", _entry_example()))


class SectionSubmission:
    """A store-bound handler; the existing callback owns handoff validation.

    Omitted IDs are exact-content identities, not semantic deduplication. Equal
    entries in one batch retain their separate occurrence numbers. Replaying an
    omitted-ID entry never overwrites later corrections under its assigned ID;
    an intentional correction must explicitly supply that ID.
    """

    def __init__(self, store, complete_section_callback, task_id):
        if (not isinstance(task_id, str) or not task_id.strip()
                or store.identity.get("task_id") != task_id):
            raise ValueError("Submission task must match its entry store identity")
        if not callable(complete_section_callback):
            raise TypeError("complete_section_callback must be callable")
        self.store, self.complete_section_callback, self.task_id = store, complete_section_callback, task_id
        self._lock = threading.RLock()

    @staticmethod
    def _failure(exc):
        if isinstance(exc, EntryStoreError):
            details = deepcopy(exc.details)
            details.setdefault("example", {"entries": [_entry_example()]})
            code = exc.code
        else:
            details = {"applied": False, "path": exc.path, "example": deepcopy(exc.example)}
            code = exc.code
        return {"ok": False, "status": "error", "saved": False, "handed_off": False,
            "document_delivery_complete": False,
            "error": {"type": type(exc).__name__, "code": code, "message": str(exc),
                      "recoverable": True, "details": details},
            "action": "Only correct the reported field and resubmit. Previously saved entries remain unchanged."}

    def submit_section(self, entries, unresolved, complete=True):
        with self._lock:
            return self._submit(entries, unresolved, complete)

    def _submit(self, entries, unresolved, complete):
        try:
            _validate_arguments(entries, unresolved, complete)
            before = self.store.snapshot()
            if not entries and not complete:
                raise _SubmissionError("An empty submission must request handoff; it cannot save a new batch.",
                    path="entries", example=[_entry_example()], code="SECTION_SUBMISSION_EMPTY")
            prepared, assignments, occurrences, ids = [], [], defaultdict(int), set()
            for index, incoming in enumerate(entries):
                entry = deepcopy(incoming)
                generated = "id" not in entry
                if generated:
                    content = json_digest(entry)
                    occurrence = occurrences[content]
                    occurrences[content] += 1
                    entry["id"] = "entry_" + json_digest({"task_id": self.task_id,
                        "entry": entry, "occurrence": occurrence})
                if entry["id"] in ids:
                    raise _SubmissionError("An explicit ID collides with another entry in this batch.",
                        path=f"entries[{index}].id", example="<an assigned ID for this one entry>")
                ids.add(entry["id"])
                reused = generated and entry["id"] in before["entries"]
                assignments.append({"entry_index": index, "id": entry["id"],
                                    "generated": generated, "reused": reused})
                prepared.append(entry)
            # Validate the entire proposed batch, including replayed entries,
            # before saving any row. The store revalidates atomically on save.
            self.store.validate_entries(prepared)
            pending = [entry for entry, assignment in zip(prepared, assignments) if not assignment["reused"]]
            saved = self.store.save_entries(pending)
        except (_SubmissionError, EntryStoreError) as exc:
            return self._failure(exc)
        if saved.get("ok") is not True:
            # A nonstandard store must never cause a handoff after failed save.
            return {"ok": False, "status": "save_failed", "saved": False, "handed_off": False,
                    "save": deepcopy(saved), "document_delivery_complete": False}
        result = {"ok": True, "status": "saved", "saved": True, "handed_off": False,
            "assigned_ids": [row["id"] for row in assignments], "assignments": assignments,
            "save": deepcopy(saved), "version": saved["version"], "changed": saved["changed"],
            "counts": deepcopy(saved["counts"]), "document_delivery_complete": False,
            "semantic_approval": False}
        if not complete:
            result["action"] = "Batch saved. Continue only the remaining work; use assigned IDs for intentional corrections."
            return result
        summary = "任务 " + self.task_id + "：本次提交 " + str(len(entries)) + " 条；当前已保存条目计数：" + json.dumps(
            saved["counts"], ensure_ascii=False, sort_keys=True) + "。内容和出处见已保存条目，未决问题见 unresolved。"
        try:
            handoff = self.complete_section_callback(summary, deepcopy(unresolved))
        except Exception as exc:
            return {**result, "ok": False, "status": "handoff_failed",
                "error": {"code": "SECTION_HANDOFF_FAILED", "type": type(exc).__name__,
                    "message": str(exc), "recoverable": True,
                    "details": {"path": "complete", "saved_entries_preserved": True,
                                "example": {"entries": [], "unresolved": deepcopy(unresolved), "complete": True}}},
                "action": "Entries are saved. Resolve the reported handoff blocker, then retry handoff with entries=[]; do not rewrite saved entries."}
        if not isinstance(handoff, dict) or (handoff.get("ok") is True and handoff.get("status") != "handed_off"):
            return {**result, "ok": False, "status": "handoff_failed",
                "error": {"code": "SECTION_HANDOFF_RECEIPT", "message": "Handoff callback did not return a verified handoff receipt.",
                          "details": {"saved_entries_preserved": True}},
                "action": "Entries remain saved; the program must inspect the handoff callback result."}
        result["handoff"] = deepcopy(handoff)
        result["ok"] = handoff.get("ok") is True
        result["handed_off"] = result["ok"] and handoff.get("status") == "handed_off"
        result["status"] = "handed_off" if result["handed_off"] else handoff.get("status", "handoff_failed")
        if handoff.get("action"):
            result["action"] = handoff["action"]
        if not result["handed_off"]:
            missing = handoff.get("missing", [])
            if isinstance(missing, list) and missing and isinstance(missing[0], dict) and missing[0].get("evidence_id"):
                result["next_action"] = {"tool": "read_source", "arguments": {"evidence_id": missing[0]["evidence_id"]}}
                result["action"] = "Entries are saved. Read the next missing source object; the program owns its continuation. Save only new findings, then retry handoff."
            result["saved_entries_preserved"] = True
        return result
