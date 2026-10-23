"""Per-task delivery entries with exact read provenance and atomic upserts.

This ledger validates representation and provenance only. Status and all output
values come from the model; a saved entry is not a finding of fact or a claim of
complete delivery. Character ranges address unchanged manifest unit text.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path
import threading

from .docx_writer import ADVICE_FIELDS, BASIC_FIELDS, DEFECT_FIELDS
from .document_material_refs import (MaterialRefNormalizationError, material_index_identity,
    normalize_material_refs, retain_material_provenance, validate_material_provenance)
from .run_state import atomic_json, json_digest


ENTRY_STATUSES = ("explicit", "inferred", "uncertain", "conflict")
ENTRY_CONTEXT_FIELDS = ("year", "period", "side", "component", "location")
ENTRY_FIELDS = {"基本信息": tuple(name for name, _ in BASIC_FIELDS),
                "病害明细": tuple(DEFECT_FIELDS), "维修建议": tuple(ADVICE_FIELDS)}
_SCHEMA_VERSION = 1


class EntryStoreError(ValueError):
    """Host-authored, content-free actionable failure details."""

    def __init__(self, code: str, message: str, *, path: str, **details):
        super().__init__(message)
        self.code = code
        self.details = {"applied": False, "path": path, **details}

    def as_tool_error(self) -> dict:
        return {"ok": False, "error": {"code": self.code, "message": str(self),
                                       "details": deepcopy(self.details)}}


def entry_schema() -> dict:
    """The single public schema shared by the SDK and structural validator.

    Exportable fields may be omitted from a row. Null/unknown values are kept,
    never filled. Both offsets must be present, or both absent for a whole unit.
    Actual bounds and prior reading are checked against this store's manifest.
    """
    scalar = {"type": ["string", "number", "null"]}
    reference = {"description": "Use an evidence_id returned by the reading tool: canonical E ID or current-document mineru:<documentSHA>#/... alias. With an alias, omit offsets; the host resolves exact canonical spans. Never guess offsets.", "anyOf": [
        {"type": "object", "properties": {"evidence_id": {"type": "string", "minLength": 1}},
         "required": ["evidence_id"], "additionalProperties": False},
        {"type": "object", "properties": {
            "evidence_id": {"type": "string", "minLength": 1},
            "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 0}},
         "required": ["evidence_id", "start", "end"], "additionalProperties": False}]}
    variants = []
    for section, fields in ENTRY_FIELDS.items():
        properties = {"id": {"type": "string", "minLength": 1, "maxLength": 128},
                      "section": {"type": "string", "enum": [section]},
                      "source_refs": {"type": "array", "minItems": 1, "items": reference},
                      "status": {"type": "string", "enum": list(ENTRY_STATUSES)},
                      "context": {"type": "object", "properties": {
                          name: {"type": ["string", "null"]} for name in ENTRY_CONTEXT_FIELDS},
                          "additionalProperties": False}}
        required = ["id", "section", "source_refs", "status"]
        if section == "基本信息":
            properties.update(field={"type": "string", "enum": list(fields)}, value=scalar)
            required += ["field", "value"]
        else:
            properties["row"] = {"type": "object", "minProperties": 1,
                                 "properties": {field: scalar for field in fields},
                                 "additionalProperties": False}
            required += ["row"]
        variants.append({"type": "object", "properties": properties,
                         "required": required, "additionalProperties": False})
    return deepcopy({"anyOf": variants})


def _valid_id(value) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= 128 and value == value.strip()
            and not any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in value))


def _scalar(value) -> bool:
    if value is None or type(value) is int:
        return True
    if type(value) is float:
        return math.isfinite(value)
    return isinstance(value, str) and all(
        char in "\t\n\r" or "\u0020" <= char <= "\ud7ff"
        or "\ue000" <= char <= "\ufffd" or "\U00010000" <= char <= "\U0010ffff"
        for char in value)


def _merge_ranges(ranges: list[list[int]]) -> list[list[int]]:
    result = []
    for start, end in sorted(ranges):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


class EntryStore:
    """One manifest and one task per directory; IDs are local to that identity.

    ``record_read`` is a host callback, called only after source content has been
    sent to this Agent. ``save_entries`` replaces an existing local ID in place;
    retries with identical entries are no-ops. Every batch validates before any
    replacement is written. Separate tasks use separate directories.
    """

    def __init__(self, manifest: dict, task_id: str, directory: str | Path, *, material_index=None):
        if not _valid_id(task_id):
            raise EntryStoreError("TASK_ID", "task_id must be a stable nonempty string",
                                  path="task_id")
        if not isinstance(manifest, dict) or not isinstance(manifest.get("units"), list):
            raise EntryStoreError("MANIFEST", "Manifest units must be an array", path="manifest.units")
        self.manifest = deepcopy(manifest)
        self.material_index = material_index
        self.material_identity = material_index_identity(self.manifest, material_index)
        self._units = {}
        for index, unit in enumerate(self.manifest["units"]):
            if (not isinstance(unit, dict) or not _valid_id(unit.get("id"))
                    or not isinstance(unit.get("text"), str) or unit["id"] in self._units):
                raise EntryStoreError("MANIFEST", "Each unit needs a unique ID and unchanged text",
                                      path=f"manifest.units[{index}]")
            self._units[unit["id"]] = unit["text"]
        try:
            manifest_hash = json_digest(self.manifest)
        except (TypeError, ValueError, UnicodeError) as error:
            raise EntryStoreError("MANIFEST", "Manifest must be finite UTF-8 JSON", path="manifest") from error
        self.identity = {"manifest_sha256": manifest_hash, "task_id": task_id,
                         "task_sha256": json_digest({"manifest_sha256": manifest_hash, "task_id": task_id})}
        if self.material_identity is not None:
            self.identity["material_identity"] = self.material_identity
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "entries.json"
        self._thread_lock = threading.RLock()
        with self._locked():
            if self.path.exists():
                self._load()
            else:
                self._commit({"schema_version": _SCHEMA_VERSION, "identity": self.identity,
                              "version": 0, "entries": {}, "entry_order": [], "reads": {},
                              **({"material_provenance": {}} if self.material_index is not None else {})}, 0)

    @contextmanager
    def _locked(self):
        with self._thread_lock, (self.directory / "entries.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _load(self) -> dict:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise EntryStoreError("STORE_READ", "Cannot read the entry ledger", path="store") from error
        if not isinstance(state, dict) or state.get("identity") != self.identity:
            raise EntryStoreError("STORE_IDENTITY", "Directory belongs to another manifest or task", path="identity")
        expected_fields = {"schema_version", "identity", "version", "entries", "entry_order", "reads"}
        if self.material_index is not None:
            expected_fields.add("material_provenance")
        if (set(state) != expected_fields
                or state["schema_version"] != _SCHEMA_VERSION or type(state["version"]) is not int
                or state["version"] < 0 or not isinstance(state["entries"], dict)
                or not isinstance(state["entry_order"], list) or not isinstance(state["reads"], dict)):
            raise EntryStoreError("STORE_STATE", "Invalid entry ledger structure", path="store")
        try:
            order = state["entry_order"]
            if len(set(order)) != len(order) or set(order) != set(state["entries"]):
                raise ValueError("Entry order differs")
            for evidence_id, ranges in state["reads"].items():
                if not isinstance(ranges, list) or not ranges:
                    raise ValueError("Empty read ranges")
                for pair in ranges:
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise ValueError("Invalid read range")
                    self._range(evidence_id, *pair, path="reads")
                if ranges != _merge_ranges(ranges):
                    raise ValueError("Read ranges are not canonical")
            normalized = self._validate([state["entries"][key] for key in order], state)
            if any(entry != state["entries"][entry["id"]] for entry in normalized):
                raise ValueError("Stored entries are not canonical")
            provenance = state.get("material_provenance", {})
            if not isinstance(provenance, dict) or set(provenance) - set(state["entries"]):
                raise ValueError("Material provenance has no corresponding entry")
            for entry_id, rows in provenance.items():
                validate_material_provenance(rows, state["entries"][entry_id]["source_refs"],
                                             self.manifest, self.material_index)
        except (ValueError, TypeError, KeyError) as error:
            raise EntryStoreError("STORE_STATE", "Invalid persisted entries or read provenance", path="store") from error
        return state

    def _commit(self, state: dict, current_version: int):
        try:
            atomic_json(self.path, state)
        except Exception as error:
            # Rename is atomic. If a later filesystem error obscures whether it
            # succeeded, report uncertainty rather than asserting an unsaved edit.
            applied = False
            try:
                applied = None if json.loads(self.path.read_text(encoding="utf-8")) == state else False
            except (OSError, ValueError):
                pass
            raise EntryStoreError("STORE_WRITE", "Atomic ledger save failed; inspect the ledger before retrying",
                                  path="store", applied=applied, current_version=current_version) from error

    def _range(self, evidence_id, start, end, *, path: str, **details):
        if not isinstance(evidence_id, str) or evidence_id not in self._units:
            raise EntryStoreError("SOURCE_ID", "Reference is not in this manifest", path=path + ".evidence_id", **details)
        length = len(self._units[evidence_id])
        bad_start = type(start) is not int or not 0 <= start <= length
        bad_end = type(end) is not int or not 0 <= end <= length
        if bad_start or bad_end or start > end:
            raise EntryStoreError("SOURCE_RANGE", "Expected integer 0 <= start <= end <= actual_length; no clamping",
                                  path=path + (".start" if bad_start else ".end"),
                                  evidence_id=evidence_id, actual_length=length, **details)

    def record_read(self, evidence_id: str, start: int, end: int) -> dict:
        """Persist an actual host-delivered half-open range, not a model claim."""
        self._range(evidence_id, start, end, path="read")
        with self._locked():
            state = self._load()
            previous = state["reads"].get(evidence_id, [])
            merged = _merge_ranges([*previous, [start, end]])
            changed = previous != merged
            if changed:
                state["reads"][evidence_id] = merged
                state["version"] += 1
                self._commit(state, state["version"] - 1)
            return {"ok": True, "version": state["version"], "changed": changed,
                    "evidence_id": evidence_id, "ranges": deepcopy(merged)}

    def _validate(self, entries, state: dict) -> list[dict]:
        if not isinstance(entries, list):
            raise EntryStoreError("ENTRIES_TYPE", "entries must be an array", path="entries", current_version=state["version"])
        normalized, seen = [], set()
        for index, entry in enumerate(entries):
            path = f"entries[{index}]"
            details = {"entry_index": index, "current_version": state["version"]}
            if not isinstance(entry, dict):
                raise EntryStoreError("ENTRY_TYPE", "Entry must be an object", path=path, **details)
            if not all(isinstance(key, str) for key in entry):
                raise EntryStoreError("ENTRY_FIELDS", "Entry property names must be strings", path=path, **details)
            entry_id = entry.get("id")
            if not _valid_id(entry_id):
                raise EntryStoreError("ENTRY_ID", "Entry needs a stable nonempty local ID", path=path + ".id", **details)
            details["entry_id"] = entry_id
            if entry_id in seen:
                raise EntryStoreError("ENTRY_DUPLICATE", "A batch cannot contain the same ID twice", path=path + ".id", **details)
            seen.add(entry_id)
            section = entry.get("section")
            if not isinstance(section, str) or section not in ENTRY_FIELDS:
                raise EntryStoreError("ENTRY_SECTION", "Use a standard delivery section", path=path + ".section", **details)
            required = {"id", "section", "source_refs", "status", *(('field', 'value') if section == "基本信息" else ('row',))}
            missing, extra = sorted(required - set(entry)), sorted(set(entry) - required - {"context"})
            if missing or extra:
                bad = missing[0] if missing else extra[0]
                raise EntryStoreError("ENTRY_FIELDS", "Entry has missing or unsupported fields", path=path + "." + str(bad),
                                      missing_fields=missing, unsupported_fields=extra, **details)
            if not isinstance(entry["status"], str) or entry["status"] not in ENTRY_STATUSES:
                raise EntryStoreError("ENTRY_STATUS", "Use explicit, inferred, uncertain or conflict", path=path + ".status", **details)
            if "context" in entry:
                context = entry["context"]
                if not isinstance(context, dict) or set(context) - set(ENTRY_CONTEXT_FIELDS):
                    raise EntryStoreError("ENTRY_CONTEXT", "context accepts year, period, side, component and location only",
                                          path=path + ".context", **details)
                for key, value in context.items():
                    if value is not None and (not isinstance(value, str) or not _scalar(value)):
                        raise EntryStoreError("ENTRY_CONTEXT", "Context must be original-source text or null; do not infer missing values",
                                              path=path + ".context." + key, **details)
            if section == "基本信息":
                field = entry["field"]
                if not isinstance(field, str) or field not in ENTRY_FIELDS[section]:
                    raise EntryStoreError("OUTPUT_FIELD", "Field is not recognized by the Word exporter", path=path + ".field", **details)
                values = {field: entry["value"]}
            else:
                values = entry["row"]
                if not isinstance(values, dict) or not values:
                    raise EntryStoreError("ENTRY_ROW", "row must be a nonempty object", path=path + ".row", **details)
            for field, value in values.items():
                value_path = path + (".value" if section == "基本信息" else ".row." + str(field))
                if field not in ENTRY_FIELDS[section]:
                    raise EntryStoreError("OUTPUT_FIELD", "Field is not recognized by the Word exporter", path=value_path, **details)
                if not _scalar(value):
                    raise EntryStoreError("OUTPUT_VALUE", "Value must be an exportable string, finite number or null", path=value_path, **details)
            refs = entry["source_refs"]
            if not isinstance(refs, list) or not refs:
                raise EntryStoreError("SOURCE_REFS", "Every entry needs at least one actually read source reference",
                                      path=path + ".source_refs", **details)
            canonical_refs = []
            for ref_index, ref in enumerate(refs):
                ref_path = path + f".source_refs[{ref_index}]"
                if not isinstance(ref, dict) or set(ref) not in ({"evidence_id"}, {"evidence_id", "start", "end"}):
                    raise EntryStoreError("SOURCE_FIELDS", "Give evidence_id alone, or evidence_id with both start and end",
                                          path=ref_path, **details)
                evidence_id = ref["evidence_id"]
                if not isinstance(evidence_id, str) or evidence_id not in self._units:
                    raise EntryStoreError("SOURCE_ID", "Reference is not in this manifest", path=ref_path + ".evidence_id", **details)
                start, end = (ref["start"], ref["end"]) if "start" in ref else (0, len(self._units[evidence_id]))
                self._range(evidence_id, start, end, path=ref_path, **details)
                if start == end:
                    raise EntryStoreError("SOURCE_EMPTY", "An empty range cannot substantiate an entry", path=ref_path, **details)
                if not any(low <= start and end <= high for low, high in state["reads"].get(evidence_id, [])):
                    raise EntryStoreError("SOURCE_UNREAD", "This entire reference must first be delivered to the Agent",
                                          path=ref_path, evidence_id=evidence_id, start=start, end=end,
                                          actual_length=len(self._units[evidence_id]), **details)
                canonical_refs.append({"evidence_id": evidence_id, "start": start, "end": end})
            previous = state["entries"].get(entry_id)
            if previous and (previous["section"] != section or
                             (section == "基本信息" and previous["field"] != entry["field"])):
                raise EntryStoreError("ENTRY_IDENTITY", "An existing ID cannot change its section or basic field",
                                      path=path + ".id", **details)
            normalized.append({**deepcopy(entry), "source_refs": canonical_refs})
        return normalized

    def validate_entries(self, entries: list[dict]) -> list[dict]:
        """Validate without saving; return entries with exact, expanded references."""
        with self._locked():
            state = self._load()
            prepared, _ = self._prepare_entries(entries, state)
            return self._validate(prepared, state)

    def _prepare_entries(self, entries, state):
        prepared = deepcopy(entries)
        origins = []
        if not isinstance(prepared, list):
            return prepared, origins
        for index, entry in enumerate(prepared):
            origin = []
            if isinstance(entry, dict) and "source_refs" in entry:
                try:
                    entry["source_refs"], origin = normalize_material_refs(entry["source_refs"],
                        self.manifest, self.material_index, path=f"entries[{index}].source_refs")
                except MaterialRefNormalizationError as exc:
                    raise EntryStoreError(exc.code, str(exc), path=exc.path,
                        example=exc.example, current_version=state["version"], entry_index=index) from exc
            origins.append(origin)
        return prepared, origins

    @staticmethod
    def _counts(state: dict) -> dict:
        counts = {section: 0 for section in ENTRY_FIELDS}
        for entry in state["entries"].values():
            counts[entry["section"]] += 1
        counts["total"] = len(state["entries"])
        return counts

    def save_entries(self, entries: list[dict]) -> dict:
        """Atomically upsert one complete model-provided entry/provenance batch."""
        with self._locked():
            state = self._load()
            prepared, origins = self._prepare_entries(entries, state)
            normalized = self._validate(prepared, state)
            provenance = deepcopy(state.get("material_provenance", {}))
            if self.material_index is not None:
                for entry, added in zip(normalized, origins):
                    retained = retain_material_provenance(provenance.get(entry["id"], []),
                                                         added, entry["source_refs"])
                    if retained:
                        provenance[entry["id"]] = retained
                    else:
                        provenance.pop(entry["id"], None)
            changed = any(entry["id"] not in state["entries"] or
                          json_digest(state["entries"][entry["id"]]) != json_digest(entry)
                          for entry in normalized) or provenance != state.get("material_provenance", {})
            if changed:
                old_version = state["version"]
                for entry in normalized:
                    if entry["id"] not in state["entries"]:
                        state["entry_order"].append(entry["id"])
                    state["entries"][entry["id"]] = entry
                if self.material_index is not None:
                    state["material_provenance"] = provenance
                state["version"] += 1
                self._commit(state, old_version)
            return {"ok": True, "version": state["version"], "changed": changed,
                    "saved_ids": [entry["id"] for entry in normalized], "counts": self._counts(state),
                    "delivery_complete": False}

    def snapshot(self) -> dict:
        """Complete authoritative state, with no original unit text or summaries."""
        with self._locked():
            state = self._load()
            return {**deepcopy(state), "counts": self._counts(state), "state_sha256": json_digest(state),
                    "delivery_complete": False}

    def read_entries(self, cursor: int = 0, limit: int = 8) -> dict:
        """Page complete entries; the caller handles model tool-budget limits."""
        if type(cursor) is not int or cursor < 0 or type(limit) is not int or limit <= 0:
            raise EntryStoreError("PAGE_RANGE", "cursor must be nonnegative and limit a positive integer", path="page")
        with self._locked():
            state = self._load()
            if cursor > len(state["entry_order"]):
                raise EntryStoreError("PAGE_RANGE", "cursor exceeds saved entry count", path="page.cursor", current_version=state["version"])
            end = min(cursor + limit, len(state["entry_order"]))
            return {"ok": True, "version": state["version"], "entries": [
                deepcopy(state["entries"][entry_id]) for entry_id in state["entry_order"][cursor:end]],
                "cursor": cursor, "next_cursor": end if end < len(state["entry_order"]) else None,
                "counts": self._counts(state), "delivery_complete": False}
