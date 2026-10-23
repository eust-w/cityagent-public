"""Homogeneous child save tools over the existing provenance-checked ledger.

No business values are defaulted. Independent new entries can be saved while
bad entries are rejected; explicit corrections stay atomic. Earlier saved
batches survive rejection. The selected tool determines section.
"""
from copy import deepcopy
import json

from .document_entries import entry_schema, EntryStoreError
from .document_section_submission import SectionSubmission, _path, _validate_arguments, _SubmissionError
from .run_state import atomic_json

SAVE_SECTIONS = {"save_basic_info": "基本信息", "save_defects": "病害明细",
                 "save_recommendations": "维修建议"}

# These are spelling aliases, not guesses about the content. Conflicting
# spellings are rejected; both original arguments and this receipt remain in
# the tool trace. Do not repair malformed JSON or infer missing business data.
FIELD_ALIASES = {"save_basic_info": {},
    "save_defects": {"development_degree": "发展程度"},
    "save_recommendations": {"源部件": "病害部位"}}


def normalize_fields(name, incoming, index):
    row, changes = deepcopy(incoming), []
    if not isinstance(row, dict):
        return row, changes, None
    for alias, field in FIELD_ALIASES[name].items():
        if alias not in row:
            continue
        if field in row and row[field] != row[alias]:
            return row, changes, {"path": f"entries[{index}].{alias}",
                "reason": "conflicting_alias", "canonical_field": field,
                "message": "两种字段名的值不同；请依据原文选择，程序没有覆盖任何值。"}
        row[field] = row.pop(alias)
        changes.append({"path": f"entries[{index}].{alias}", "canonical_field": field})
    return row, changes, None


def dedicated_schema(name):
    section = SAVE_SECTIONS[name]
    item = next(v for v in entry_schema()["anyOf"] if v["properties"]["section"]["enum"] == [section])
    item["properties"].pop("section")
    item["required"].remove("section")
    item["required"].remove("id")
    item["properties"]["id"]["description"] = "Omit for new entries; use a returned ID only for an intentional correction."
    item["properties"]["source_refs"]["items"]["description"] = "Copy an exact E evidence_id from the supplied units. Multiple sources use separate objects; never concatenate paths, hashes or E-number ranges. Omit offsets for a whole supplied unit."
    if section != "基本信息":
        row = item["properties"].pop("row")
        item["required"].remove("row")
        item["properties"].update(row["properties"])
        item["required"].append("病害描述" if section == "病害明细" else "建议内容")
    return {"type": "object", "additionalProperties": False, "required": ["entries"],
        "properties": {"entries": {"type": "array", "minItems": 1,
            "items": item, "description": "Prefer small batches of this single type; a complete larger batch is also accepted. Use a real JSON array. No section, row, complete or unresolved fields."}}}


def dedicated_outer_schema(name):
    schema = dedicated_schema(name)
    schema["properties"]["entries"]["items"] = {}
    return schema


def example(name):
    row = {"source_refs": [{"evidence_id": "<已提供的原文编号>"}], "status": "explicit"}
    row.update({"field": "桥梁名称", "value": "<原文值>"} if name == "save_basic_info"
        else {"病害描述" if name == "save_defects" else "建议内容": "<原文支持的简短内容>"})
    return {"entries": [row]}


class DedicatedSectionTools:
    def __init__(self, submission: SectionSubmission):
        self.submission = submission
        self.failures = {}
        path = submission.store.directory / "dedicated-save-errors.json"
        if path.exists():
            state = json.loads(path.read_bytes())
            if state["identity"] != submission.store.identity:
                raise ValueError("Save errors belong to another task")
            self.failures = state["failures"]

    def _persist(self):
        atomic_json(self.submission.store.directory / "dedicated-save-errors.json", {
            "identity": self.submission.store.identity, "failures": self.failures})

    def save(self, name, entries):
        from jsonschema import Draft202012Validator
        outer = next(Draft202012Validator(dedicated_outer_schema(name)).iter_errors({"entries": entries}), None)
        if outer:
            return {"ok": False, "saved": False, "error": {"code": "DEDICATED_CONTAINER",
                "details": {"path": _path(outer.absolute_path), "reason": outer.validator, "example": example(name)}}}
        validator = Draft202012Validator(dedicated_schema(name)["properties"]["entries"]["items"])
        converted, valid_indices, rejected, normalizations = [], [], [], []
        for index, incoming in enumerate(entries):
            incoming, changes, details = normalize_fields(name, incoming, index)
            normalizations.extend(changes)
            error = next(validator.iter_errors(incoming), None) if details is None else None
            if error is not None:
                location = ["entries", index, *error.absolute_path]
                if error.validator == "required":
                    location.append(next(k for k in error.validator_value if k not in error.instance))
                details = {"reason": error.validator, "message": error.message}
                if error.validator == "additionalProperties" and isinstance(error.instance, dict):
                    extra = sorted(set(error.instance) - set(error.schema.get("properties", {})))
                    location.extend(extra[:1])
                    details.update(unexpected_fields=extra, allowed_fields=list(error.schema.get("properties", {})))
                details["path"] = _path(location)
            if details is None:
                row = deepcopy(incoming)
                entry = {k: row.pop(k) for k in ("id", "source_refs", "status", "context") if k in row}
                entry["section"] = SAVE_SECTIONS[name]
                entry.update(row if name == "save_basic_info" else {"row": row})
                try:
                    _validate_arguments([entry], [], False)
                    # Validation only: temporary ID is never committed or returned.
                    self.submission.store.validate_entries([{**entry, "id": entry.get("id", "preflight")}])
                except (EntryStoreError, _SubmissionError) as exc:
                    failure = self.submission._failure(exc)
                    details = deepcopy(failure["error"].get("details", {}))
                    details["path"] = details.get("path", "entries[0]").replace("entries[0]", f"entries[{index}]", 1)
                    details["reason"] = failure["error"]["code"]
                    details["message"] = str(exc)
                if details is None:
                    converted.append(entry);valid_indices.append(index)
            if details is not None:
                rejected.append({**details, "entry_index": index})
        # Corrections can be related. Do not partly apply such a transaction.
        atomic_corrections = bool(rejected and any(isinstance(r, dict) and "id" in r for r in entries))
        result = None
        if converted and not atomic_corrections:
            result = self.submission.submit_section(converted, [], False)
            if not result.get("ok"):
                self.failures[name] = deepcopy(result.get("error", {"code": "SAVE_FAILED"}))
                self._persist()
                return result
        if rejected:
            self.failures[name] = {"rejected": rejected, "atomic_corrections": atomic_corrections}
        else:
            self.failures.pop(name, None)
        self._persist()
        state = self.submission.store.snapshot()
        receipt = {"ok": not rejected, "status": "partially_saved" if result and rejected else "saved" if result else "rejected",
            "saved": result is not None, "changed": bool(result and result.get("changed")),
            "version": state["version"], "counts": state["counts"],
            "saved_indices": valid_indices if result else [],
            "field_normalizations": normalizations,
            "assignments": [{**r, "entry_index": valid_indices[r["entry_index"]]} for r in (result or {}).get("assignments", [])]}
        if rejected:
            receipt.update(error={"code": "DEDICATED_ENTRIES_REJECTED", "details": {
                "path": rejected[0]["path"], "rejected": rejected, "example": example(name),
                "atomic_corrections": atomic_corrections}},
                action="Only resubmit rejected indices with corrected fields. Saved indices are durable: do not rewrite them. If an item has no evidence, omit it and list the gap in handoff_section(notes); never invent source_refs.")
        return receipt

    def complete(self, unresolved, summary=""):
        if self.failures and not unresolved:
            return {"ok": False, "status": "save_errors_pending", "errors": deepcopy(self.failures),
                "action": "A save failed. Correct that batch, or explicitly list the unsaved content in unresolved before handoff. Saved counts do not prove completeness."}
        # Preserve program-observed errors even when the model accepts gaps.
        issues = list(unresolved) + ["保存失败未确认修复：" + name for name in self.failures]
        return self.submission.submit_section([], issues, True)

    def handoff(self, notes=""):
        if not isinstance(notes, str):
            raise ValueError("Handoff notes must be plain text")
        return self.complete([notes] if notes.strip() else [])
