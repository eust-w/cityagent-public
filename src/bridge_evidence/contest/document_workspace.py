"""Durable reading receipts, versioned evidence records and reviewed drafts.

This store checks provenance and state transitions, not whether prose entails a
claim. That judgement belongs to the model's separately recorded review. Text
offsets are half-open character ranges. Successful image tools record a receipt
with ``record_read(image_id, 0, 1)`` for legacy consumers, or the atomic
``record_image_observation`` sidecar/index transition for new consumers. An image
inventory alone is not a reading receipt.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
from typing import Any, Iterator

from .document_delivery_validation import validate_delivery_data
from .document_material_refs import (MaterialRefNormalizationError, material_index_identity,
    normalize_material_refs, retain_material_provenance, validate_material_provenance)
from .run_state import atomic_json


class WorkspaceError(ValueError):
    """Invalid workspace input or an unsafe state transition."""


class WorkspaceRepairError(WorkspaceError):
    """Host-authored structural diagnostics for one rejected atomic update.

    Only bounded identifiers/locations enter model feedback, never draft prose,
    source text, arbitrary exception attributes, or a factual verdict.
    """

    def __init__(self, message: str, *, code: str, version: int, **details: Any):
        super().__init__(message)
        self.code = code
        self.details = {"current_version": version, "applied": False}
        for key, value in details.items():
            if type(value) is list:
                self.details[key + "_count"] = len(value)
                # Show complete locations, never truncated identifier prefixes.
                safe = [item for item in value if type(item) is str and len(item) <= 96
                        and re.fullmatch(r"[\w.:-]+", item) is not None]
                self.details[key] = safe[:4]
                if len(safe[:4]) != len(value):
                    self.details[key + "_omitted"] = len(value) - len(safe[:4])
            elif type(value) is int and 0 <= value <= 2**31 - 1:
                self.details[key] = value
            elif type(value) is str and value in {"基本信息", "维修建议", "病害明细"}:
                self.details[key] = value


class WorkspaceConflictError(WorkspaceError):
    """An update/review was prepared against an obsolete workspace version."""


class WorkspaceIdentityError(WorkspaceError):
    """A checkpoint belongs to different input, parser, model or run settings."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _working_notes(value: Any) -> dict:
    """One bounded model memo; its contents never grant evidence authority."""
    def check(item: Any) -> None:
        if item is None or type(item) in (str, bool, int, float):
            return
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif type(item) is list:
            for child in item:
                check(child)
        else:
            raise ValueError("Only JSON values are allowed")

    try:
        if type(value) is not dict:
            raise ValueError("The memo must be an object")
        check(value)
        serialized = _json(value)
        size = len(serialized.encode("utf-8"))
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise WorkspaceError("working_notes must be a finite JSON object with string keys and valid UTF-8") from exc
    if size > 4096:
        raise WorkspaceError("working_notes must not exceed 4096 UTF-8 bytes")
    return json.loads(serialized)


def bind_image_observation(value: dict) -> dict:
    """Content identity for a complete image-tool response, excluding cache-hit telemetry.

    This is a binding, not semantic verification. The trusted tool-delivery path
    must call it only for an actually successful original-image observation.
    """
    try:
        result = json.loads(_json(value))
    except (TypeError, ValueError) as exc:
        raise WorkspaceError("Image observation must contain finite JSON") from exc
    if not isinstance(result, dict) or result.get("ok") is not True or result.get("status") != "observed":
        raise WorkspaceError("Only a complete successful image observation can be bound")
    required = {"image_id", "document_sha256", "source_path", "image_sha256", "question", "region",
                "context_refs", "backend", "observation", "judgment"}
    if not required <= set(result):
        raise WorkspaceError("Image observation lacks its complete occurrence/request identity")
    for name in ("image_id", "source_path", "question"):
        if not isinstance(result[name], str) or not result[name].strip():
            raise WorkspaceError(f"Image observation needs nonempty {name}")
    for name in ("document_sha256", "image_sha256"):
        if not isinstance(result[name], str) or re.fullmatch(r"[0-9a-f]{64}", result[name]) is None:
            raise WorkspaceError(f"Image observation needs SHA256 {name}")
    region = result["region"]
    if not isinstance(region, dict) or (region and (set(region) != {"x", "y", "width", "height"}
            or any(type(v) is not int for v in region.values()) or region["x"] < 0 or region["y"] < 0
            or region["width"] <= 0 or region["height"] <= 0)):
        raise WorkspaceError("Image observation has invalid pixel region")
    _ids(result["context_refs"], "image context_refs")
    if not isinstance(result["backend"], dict) or not isinstance(result["backend"].get("identity"), str):
        raise WorkspaceError("Image observation needs its backend identity")
    required_specific = result["backend"].get("requires_observation_id", False)
    if type(required_specific) is not bool:
        raise WorkspaceError("requires_observation_id must be boolean")
    body = {key: val for key, val in result.items() if key not in {"cache", "observation_id", "observation_sha256"}}
    digest = _sha(body)
    if ("observation_sha256" in result and result["observation_sha256"] != digest
            or "observation_id" in result and result["observation_id"] != "io_" + digest):
        raise WorkspaceIdentityError("Image observation identity/hash differs from its complete content")
    return {**result, "observation_id": "io_" + digest, "observation_sha256": digest}


def _ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise WorkspaceError(f"{label} must be a list of nonempty IDs")
    if len(value) != len(set(value)):
        raise WorkspaceError(f"{label} contains duplicate IDs")
    return value


def _ranges(values: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(values):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def _covers(values: list[list[int]], start: int, end: int) -> bool:
    return start == end or any(a <= start and end <= b for a, b in _ranges(values))


class DocumentWorkspace:
    """One immutable manifest/run identity and an atomically saved workspace.

    ``apply_update`` merges lists of records by ID; omitted records survive.
    Explicit retractions use ``retract_fact_ids`` / ``retract_relation_ids`` /
    ``retract_question_ids``. A draft update supplies all three of ``draft``,
    ``draft_fact_ids`` and ``draft_bindings``. Bindings use paths such as
    ``基本信息.桥梁名称`` or ``病害明细.0`` and refer to fact/relation IDs.
    Alternatively, ``draft_patch`` applies ordered ``set_basic``, ``append_row``,
    ``replace_row``, ``remove_row`` and ``bind_output`` operations without
    resending the report. ``bind_output`` only changes an existing item's
    provenance binding; it does not certify semantic support.
    ``working_notes`` replaces a bounded model memo, not facts or source reads;
    omitted notes survive and an empty object clears the memo.

    Every persisted event advances ``version`` (also ``workspace_version``).
    Review callers freeze that version and ``draft_sha256``. Duplicate reads
    are no-ops. Callers must obtain a fresh snapshot after successful tools.
    """

    _META = {"valid", "record_version", "invalidated_by"}
    _SECTIONS = {"基本信息", "维修建议", "病害明细"}

    def __init__(self, manifest: dict, directory: Path, run_identity: dict, *,
                 max_image_observations: int = 2048, max_image_observation_bytes: int = 4 * 1024 * 1024,
                 source_gate=None, require_semantic_review: bool = True, material_index=None, allow_quality_gaps=False):
        if not isinstance(manifest, dict) or not isinstance(run_identity, dict):
            raise WorkspaceError("manifest and run_identity must be objects")
        if type(max_image_observations) is not int or max_image_observations <= 0:
            raise WorkspaceError("max_image_observations must be a positive integer")
        if type(max_image_observation_bytes) is not int or max_image_observation_bytes <= 0:
            raise WorkspaceError("max_image_observation_bytes must be a positive integer")
        if type(require_semantic_review) is not bool:
            raise WorkspaceError("require_semantic_review must be a boolean")
        if type(allow_quality_gaps) is not bool:
            raise WorkspaceError("allow_quality_gaps must be boolean")
        self.allow_quality_gaps = allow_quality_gaps
        self.require_semantic_review = require_semantic_review
        self.max_image_observations = max_image_observations
        self.max_image_observation_bytes = max_image_observation_bytes
        if source_gate is not None and not callable(source_gate):
            raise WorkspaceError("source_gate must be a trusted callable")
        self.source_gate = source_gate
        self._manifest = deepcopy(manifest)
        self.material_index = material_index
        self.material_identity = material_index_identity(self._manifest, material_index)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.directory / "workspace.json"
        self._lock_path = self.directory / "workspace.lock"
        self._identity = {"manifest_sha256": _sha(manifest), "run_identity": deepcopy(run_identity)}
        if self.allow_quality_gaps:
            self._identity["quality_policy"] = "speed_v1_explicit_gaps"
        if self.material_identity is not None:
            self._identity["material_identity"] = self.material_identity
        if not self.require_semantic_review:
            self._identity["semantic_review_policy"] = "not_run_by_user_policy"
        self._units = {item["id"]: item for item in self._manifest.get("units", [])}
        self._images = {item["id"]: item for item in self._manifest.get("images", [])}
        if len(self._units) != len(self._manifest.get("units", [])) or len(self._images) != len(self._manifest.get("images", [])):
            raise WorkspaceError("Duplicate manifest evidence IDs")
        if self._units.keys() & self._images.keys():
            raise WorkspaceError("Text and image IDs must be distinct")
        with self._lock():
            if self.checkpoint_path.exists():
                self._load()
            else:
                self._save({
                    "schema_version": 1, "identity": self._identity, "version": 0,
                    "facts_version": 0, "facts": {}, "relations": {}, "questions": {},
                    "working_notes": {},
                    "reads": {}, "image_reads": [], "image_observations": {}, "image_observation_index_sha256": _sha({}), "exclusions": [],
                    "draft": None, "draft_fact_ids": [], "draft_bindings": {},
                    "draft_sha256": None, "draft_valid": False, "review": None, "history": [],
                    **({"material_provenance": {"facts": {}, "relations": {}}} if self.material_index is not None else {}),
                })

    @property
    def manifest(self) -> dict:
        """A copy: callers cannot change the source underlying saved receipts."""
        return deepcopy(self._manifest)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        # The file lock also protects two independently restored instances.
        with self._lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load(self) -> dict:
        state = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1:
            raise WorkspaceIdentityError("Unsupported workspace schema version")
        if _json(state.get("identity")) != _json(self._identity):
            raise WorkspaceIdentityError("Checkpoint identity differs from manifest or run_identity")
        # Older checkpoints have no memo. Reading one does not rewrite its
        # bytes or fabricate progress; the next ordinary update persists it.
        try:
            state["working_notes"] = _working_notes(state.get("working_notes", {}))
        except WorkspaceError as exc:
            raise WorkspaceIdentityError("Saved working_notes are invalid") from exc
        # Old OCR/image-read-only workspaces remain loadable. Absence means no
        # recoverable observation content, never a fabricated successful read.
        state.setdefault("image_observations", {})
        index = state["image_observations"]
        if not isinstance(index, dict) or len(index) > self.max_image_observations:
            raise WorkspaceIdentityError("Image observation index exceeds its bound or is invalid")
        if (index or "image_observation_index_sha256" in state) and state.get("image_observation_index_sha256") != _sha(index):
            raise WorkspaceIdentityError("Image observation index hash differs")
        for observation_id, entry in index.items():
            self._load_observation_entry(observation_id, entry)
        for collection in ("facts", "relations"):
            for record in state.get(collection, {}).values():
                for ref in record.get("source_refs", []):
                    if "image_id" not in ref:
                        continue
                    ids = [ref["observation_id"]] if "observation_id" in ref else ref.get("observation_ids", [])
                    if any(oid not in index or index[oid]["image_id"] != ref["image_id"] for oid in ids):
                        raise WorkspaceIdentityError("Saved image fact references a missing/mismatched observation")
        if self.material_index is not None:
            provenance = state.get("material_provenance")
            if not isinstance(provenance, dict) or set(provenance) != {"facts", "relations"}:
                raise WorkspaceIdentityError("Saved material provenance collections are invalid")
            for collection, entries in provenance.items():
                if not isinstance(entries, dict) or set(entries) - set(state[collection]):
                    raise WorkspaceIdentityError("Saved material provenance has no corresponding record")
                for record_id, rows in entries.items():
                    try:
                        validate_material_provenance(rows, state[collection][record_id]["source_refs"],
                                                     self._manifest, self.material_index)
                    except (ValueError, TypeError, KeyError) as exc:
                        raise WorkspaceIdentityError("Saved material provenance differs from this record/index") from exc
        elif state.get("material_provenance"):
            raise WorkspaceIdentityError("Saved material provenance needs its original material index")
        return state

    def _save(self, state: dict) -> None:
        content = _json(state)
        name: str | None = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=self.directory, prefix=".workspace-", suffix=".tmp", encoding="utf-8", delete=False) as handle:
                name = handle.name
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.checkpoint_path)
            name = None
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if name is not None:
                Path(name).unlink(missing_ok=True)

    def _snapshot(self, state: dict) -> dict:
        result = deepcopy(state)
        result["workspace_version"] = state["version"]
        result["reading_progress"] = self._reading_progress(state)
        return result

    def snapshot(self) -> dict:
        with self._lock():
            return self._snapshot(self._load())

    def _advance(self, state: dict, event: str, details: dict) -> None:
        state["version"] += 1
        state["history"].append({"version": state["version"], "event": event, **deepcopy(details)})

    @staticmethod
    def _check_version(state: dict, version: Any) -> None:
        if type(version) is not int or version != state["version"]:
            raise WorkspaceConflictError(f"Stale workspace version {version!r}; current version is {state['version']}")

    def _interval(self, evidence_id: str, start: Any, end: Any, *, nonempty: bool = False) -> None:
        if evidence_id not in self._units:
            raise WorkspaceError(f"Unknown evidence ID: {evidence_id}")
        size = len(self._units[evidence_id]["text"])
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= size:
            raise WorkspaceError(f"Invalid interval for {evidence_id}: {start!r}:{end!r}, length={size}")
        if nonempty and start == end:
            raise WorkspaceError("A fact source reference must cover nonempty text")

    def record_read(self, evidence_id: str, start: int, end: int) -> None:
        with self._lock():
            state = self._load()
            if evidence_id in self._images:
                if type(start) is not int or type(end) is not int or (start, end) != (0, 1):
                    raise WorkspaceError("Image read receipts use the interval [0, 1)")
                if evidence_id in state["image_reads"]:
                    return
                state["image_reads"].append(evidence_id)
            else:
                self._interval(evidence_id, start, end)
                previous = state["reads"].get(evidence_id, [])
                current = _ranges(previous + [[start, end]])
                if current == previous:
                    return
                state["reads"][evidence_id] = current
            state["review"] = None
            self._advance(state, "read", {"evidence_id": evidence_id, "start": start, "end": end})
            self._save(state)

    def _observation_path(self, observation_id: str) -> Path:
        if not isinstance(observation_id, str) or re.fullmatch(r"io_[0-9a-f]{64}", observation_id) is None:
            raise WorkspaceIdentityError("Invalid image observation ID")
        return self.directory / "image-observations" / (observation_id + ".json")

    def _load_observation_entry(self, observation_id: str, entry: dict) -> dict:
        required = {"image_id", "observation_sha256", "sidecar_sha256", "requires_observation_id"}
        if (not isinstance(entry, dict) or set(entry) != required or entry["image_id"] not in self._images
                or type(entry["requires_observation_id"]) is not bool):
            raise WorkspaceIdentityError("Image observation index entry is invalid")
        path = self._observation_path(observation_id)
        try:
            if path.stat().st_size > self.max_image_observation_bytes:
                raise WorkspaceIdentityError("Image observation sidecar exceeds its bound")
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise WorkspaceIdentityError("Image observation sidecar is missing or invalid") from exc
        if (not isinstance(data, dict) or set(data) != {"schema_version", "workspace_identity_sha256", "observation"}
                or data["schema_version"] != 1 or data["workspace_identity_sha256"] != _sha(self._identity)
                or _sha(data) != entry["sidecar_sha256"]):
            raise WorkspaceIdentityError("Image observation sidecar identity/hash differs")
        observation = bind_image_observation(data["observation"])
        image_id = entry["image_id"]
        if (observation["observation_id"] != observation_id or observation["observation_sha256"] != entry["observation_sha256"]
                or observation["image_id"] != image_id or observation["document_sha256"] != self._manifest["document_sha256"]
                or observation["source_path"] != self._images[image_id]["source_path"]
                or observation["backend"].get("requires_observation_id", False) != entry["requires_observation_id"]):
            raise WorkspaceIdentityError("Image observation belongs to different source/occurrence/backend")
        return observation

    def load_image_observation(self, image_id: str, observation_id: str) -> dict:
        """Trusted review reads one immutable full sidecar; snapshots retain only IDs/SHA."""
        with self._lock():
            state = self._load()
            entry = state["image_observations"].get(observation_id)
            if entry is None or entry["image_id"] != image_id:
                raise WorkspaceError("Observation was not delivered for this image occurrence")
            return self._load_observation_entry(observation_id, entry)

    def record_image_observation(self, image_id: str, observation: dict) -> None:
        """Called only AFTER complete tool/native response budget validation.

        Persists the complete immutable result before a single atomic index/read
        transition. An orphan sidecar after a crash grants no reading coverage.
        Duplicate semantic content with different cache-hit flags is a no-op.
        """
        observation = bind_image_observation(observation)
        if image_id not in self._images or observation["image_id"] != image_id:
            raise WorkspaceIdentityError("Image observation occurrence differs")
        if (observation["document_sha256"] != self._manifest["document_sha256"]
                or observation["source_path"] != self._images[image_id]["source_path"]
                or any(ref not in self._units for ref in observation["context_refs"])):
            raise WorkspaceIdentityError("Image observation source/context differs from manifest")
        # Cache hit telemetry belongs to each tool call, not observation identity.
        immutable = {key: val for key, val in observation.items() if key != "cache"}
        sidecar = {"schema_version": 1, "workspace_identity_sha256": _sha(self._identity), "observation": immutable}
        # atomic_json uses indentation. Bound actual serialized bytes too.
        if len((json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()) > self.max_image_observation_bytes:
            raise WorkspaceError("Complete image observation exceeds the persistence bound")
        observation_id = immutable["observation_id"]
        entry = {"image_id": image_id, "observation_sha256": immutable["observation_sha256"],
                 "sidecar_sha256": _sha(sidecar),
                 "requires_observation_id": immutable["backend"].get("requires_observation_id", False)}
        with self._lock():
            state = self._load()
            previous = state["image_observations"].get(observation_id)
            if previous is not None:
                if previous != entry:
                    raise WorkspaceIdentityError("Cannot overwrite an immutable image observation")
                return
            if len(state["image_observations"]) >= self.max_image_observations:
                raise WorkspaceError("Image observation count budget exhausted; no source read granted")
            path = self._observation_path(observation_id)
            if path.exists():
                self._load_observation_entry(observation_id, entry)
            else:
                atomic_json(path, sidecar)
            state["image_observations"][observation_id] = entry
            state["image_observation_index_sha256"] = _sha(state["image_observations"])
            if image_id not in state["image_reads"]:
                state["image_reads"].append(image_id)
            state["review"] = None
            self._advance(state, "image_observation", {"image_id": image_id, "observation_id": observation_id,
                                                      "observation_sha256": entry["observation_sha256"]})
            self._save(state)

    def _validate_refs(self, refs: Any, state: dict) -> None:
        if not isinstance(refs, list):
            raise WorkspaceError("source_refs must be a list")
        for ref in refs:
            if not isinstance(ref, dict):
                raise WorkspaceError("Each source reference must be an object")
            if "image_id" in ref:
                if set(ref) - {"image_id", "source_path", "observation_id", "observation_ids", "observation_set_sha256", "observation_limitation"}:
                    raise WorkspaceError("An image reference cannot also contain a text interval")
                image_id = ref["image_id"]
                if image_id not in self._images or image_id not in state["image_reads"]:
                    raise WorkspaceError(f"Image source was not actually read: {image_id}")
                source = self._images[image_id]
                available = {oid: row for oid, row in state["image_observations"].items() if row["image_id"] == image_id}
                if "observation_id" in ref:
                    if set(ref) & {"observation_ids", "observation_set_sha256", "observation_limitation"}:
                        raise WorkspaceError("Specific observation references cannot mix a legacy frozen set")
                    if not isinstance(ref["observation_id"], str) or ref["observation_id"] not in available:
                        raise WorkspaceError("Image observation was not actually delivered for this occurrence")
                elif "observation_ids" in ref:
                    # Existing OCR/image-only records keep their original set;
                    # later crops must not silently expand the evidence scope.
                    identifiers = _ids(ref["observation_ids"], "frozen observation_ids")
                    if (set(identifiers) - set(available) or ref.get("observation_set_sha256") != _sha(identifiers)
                            or not isinstance(ref.get("observation_limitation"), str) or not ref["observation_limitation"]):
                        raise WorkspaceError("Legacy frozen image observation set is invalid")
                else:
                    if any(row["requires_observation_id"] for row in available.values()):
                        raise WorkspaceError("Own-model image facts require a specific observation_id")
                    identifiers = sorted(available)
                    ref.update(observation_ids=identifiers, observation_set_sha256=_sha(identifiers),
                               observation_limitation="Legacy image-only reference; only this frozen delivered set is available, without a specific crop/question claim."
                               if identifiers else "Legacy image-read receipt has no persisted observation content; crop/question semantics are unavailable.")
            else:
                if set(ref) - {"evidence_id", "start", "end", "source_path"} or not {"evidence_id", "start", "end"} <= set(ref):
                    raise WorkspaceError("Text references require evidence_id, start and end")
                evidence_id, start, end = ref["evidence_id"], ref["start"], ref["end"]
                self._interval(evidence_id, start, end, nonempty=True)
                if not _covers(state["reads"].get(evidence_id, []), start, end):
                    raise WorkspaceError(f"Source interval was not actually read: {evidence_id}:{start}:{end}")
                source = self._units[evidence_id]
            if "source_path" in ref and ref["source_path"] != source.get("source_path"):
                raise WorkspaceError("Reference source_path differs from its manifest source")

    def _normalize_record(self, collection: str, incoming: dict, state: dict, *, path=None) -> dict:
        if not isinstance(incoming, dict) or not isinstance(incoming.get("id"), str) or not incoming["id"].strip():
            raise WorkspaceError(f"{collection} records need a nonempty id")
        if set(incoming) & self._META:
            raise WorkspaceError("Record validity/version metadata is managed by the workspace")
        for ref in incoming.get("source_refs", []) if isinstance(incoming.get("source_refs", []), list) else []:
            if isinstance(ref, dict) and set(ref) & {"observation_ids", "observation_set_sha256", "observation_limitation"}:
                raise WorkspaceError("Frozen image observation scope metadata is managed by the workspace")
        record = {k: v for k, v in state[collection].get(incoming["id"], {}).items() if k not in self._META}
        record.update(deepcopy(incoming))
        record.setdefault("depends_on", [])
        _ids(record["depends_on"], "depends_on")
        if collection == "questions":
            record.setdefault("status", "open")
            if record["status"] not in {"open", "resolved", "unknown", "conflict"}:
                raise WorkspaceError("Question status must be open, resolved, unknown or conflict")
            if record["status"] in {"resolved", "unknown"} and (not isinstance(record.get("reason"), str) or not record["reason"].strip()):
                raise WorkspaceError("Resolved/unknown questions require a reason")
        else:
            if collection == "facts" and not {"value", "source_refs", "status"} <= record.keys():
                raise WorkspaceRepairError("Facts require value, source_refs and status",
                    code="FACT_FIELDS", version=state["version"], record_ids=[incoming["id"]],
                    missing_fields=sorted({"value", "source_refs", "status"} - record.keys()))
            record.setdefault("source_refs", [])
            record.setdefault("status", "explicit")
            if record["status"] not in {"explicit", "inferred", "uncertain", "conflict"}:
                raise WorkspaceError("Invalid evidence record status")
            try:
                record["source_refs"], added_provenance = normalize_material_refs(record["source_refs"],
                    self._manifest, self.material_index, path=(path or collection) + ".source_refs")
            except MaterialRefNormalizationError as exc:
                error = WorkspaceRepairError("Material source reference cannot be mapped; the entire update was not saved",
                    code=exc.code, version=state["version"])
                error.details.update(path=exc.path,
                    example={"source_refs": [{"evidence_id": "<exact source ID from the reading tool>"}]},
                    reason=str(exc))
                raise error from exc
            self._validate_refs(record["source_refs"], state)
            if self.material_index is not None:
                provenance = state["material_provenance"][collection]
                retained = retain_material_provenance(provenance.get(record["id"], []),
                                                     added_provenance, record["source_refs"])
                if retained:
                    provenance[record["id"]] = retained
                else:
                    provenance.pop(record["id"], None)
            if not record["source_refs"] and not self._dependencies(record, self._records(state)) and record["status"] != "uncertain":
                raise WorkspaceError("Supported claims need read source_refs or evidence dependencies")
        record.update(valid=True, record_version=state["version"] + 1, invalidated_by=[])
        return record

    @staticmethod
    def _records(state: dict) -> dict:
        return {**state["facts"], **state["relations"], **state["questions"]}

    def _dependencies(self, record: dict, records: dict) -> set[str]:
        dependencies = set(record.get("depends_on", []))
        # Relations cannot evade invalidation by omitting an endpoint from an
        # otherwise optional depends_on list.
        for key in ("subject_id", "object_id"):
            endpoint = record.get(key)
            if isinstance(endpoint, str) and endpoint:
                dependencies.add(endpoint)
        return dependencies

    def _validate_graph(self, state: dict, rewritten: set[str]) -> None:
        records = self._records(state)
        for record_id in rewritten:
            for dependency in self._dependencies(records[record_id], records):
                if dependency not in records or not records[dependency]["valid"]:
                    raise WorkspaceError(f"{record_id} depends on missing/invalid record {dependency}")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(record_id: str) -> None:
            if record_id in visiting:
                raise WorkspaceError(f"Cyclic dependency involving {record_id}")
            if record_id in visited or record_id not in records:
                return
            visiting.add(record_id)
            for dependency in self._dependencies(records[record_id], records):
                visit(dependency)
            visiting.remove(record_id)
            visited.add(record_id)

        for record_id in records:
            visit(record_id)

    def _exclusions(self, incoming: Any, state: dict) -> None:
        if not isinstance(incoming, list):
            raise WorkspaceError("exclusions must be a list")
        for exclusion in incoming:
            if not isinstance(exclusion, dict) or not isinstance(exclusion.get("reason"), str) or not exclusion["reason"].strip():
                raise WorkspaceError("An exclusion needs a nonempty reason")
            if "image_id" in exclusion:
                if set(exclusion) != {"image_id", "reason"} or exclusion["image_id"] not in self._images:
                    raise WorkspaceError("Unknown image exclusion or unexpected fields")
            else:
                if set(exclusion) != {"evidence_id", "start", "end", "reason"}:
                    raise WorkspaceError("Text exclusions require evidence_id, start, end and reason")
                self._interval(exclusion["evidence_id"], exclusion["start"], exclusion["end"], nonempty=True)
            if exclusion not in state["exclusions"]:
                state["exclusions"].append(deepcopy(exclusion))

    def _set_draft(self, update: dict, state: dict) -> None:
        if not {"draft", "draft_fact_ids", "draft_bindings"} <= update.keys():
            raise WorkspaceRepairError("A draft update requires draft, draft_fact_ids and draft_bindings together",
                code="DRAFT_BINDINGS", version=state["version"],
                missing_fields=sorted({"draft", "draft_fact_ids", "draft_bindings"} - update.keys()))
        draft = update["draft"]
        if not isinstance(draft, dict) or set(draft) != self._SECTIONS:
            raise WorkspaceError("draft must contain exactly 基本信息, 维修建议 and 病害明细")
        if not isinstance(draft["基本信息"], dict) or any(not isinstance(draft[name], list) or any(not isinstance(row, dict) for row in draft[name]) for name in ("维修建议", "病害明细")):
            raise WorkspaceError("Draft basic information is an object and both detail sections are lists of objects")
        try:
            # Use the exporter's contract at the atomic write boundary, so a
            # bad field/value is repairable before expensive semantic review.
            # This checks representation only; facts still belong to the model.
            validate_delivery_data(draft)
        except ValueError as exc:
            invalid_paths = []
            for section in ("基本信息", "维修建议", "病害明细"):
                rows = [("基本信息", draft[section])] if section == "基本信息" else [
                    (f"{section}.{index}", row) for index, row in enumerate(draft[section])]
                for location, row in rows:
                    for field, value in row.items():
                        probe = {"基本信息": {}, "维修建议": [], "病害明细": []}
                        if section == "基本信息":
                            probe[section] = {field: value}
                        else:
                            probe[section] = [{field: value}]
                        try:
                            validate_delivery_data(probe)
                        except ValueError:
                            invalid_paths.append(f"{location}.{field}")
            raise WorkspaceRepairError(str(exc), code="DRAFT_SHAPE", version=state["version"],
                                       invalid_paths=invalid_paths) from exc
        ids = _ids(update["draft_fact_ids"], "draft_fact_ids")
        records = {**state["facts"], **state["relations"]}
        invalid_ids = [record_id for record_id in ids if record_id not in records or not records[record_id]["valid"]]
        if invalid_ids:
            raise WorkspaceRepairError("Draft refers to missing or invalid facts/relations",
                code="DEPENDENCY_INVALID", version=state["version"], invalid_ids=invalid_ids)
        bindings = update["draft_bindings"]
        if not isinstance(bindings, dict):
            raise WorkspaceError("draft_bindings must be an object")
        allowed_paths = {f"基本信息.{key}" for key in draft["基本信息"]}
        required_paths = {f"基本信息.{key}" for key, value in draft["基本信息"].items() if value not in (None, "", [], {})}
        for section in ("维修建议", "病害明细"):
            paths = {f"{section}.{i}" for i in range(len(draft[section]))}
            allowed_paths.update(paths)
            required_paths.update(paths)
        if set(bindings) - allowed_paths or required_paths - set(bindings):
            raise WorkspaceRepairError("Every nonempty output field/row needs its own draft binding; unknown paths are not allowed",
                code="DRAFT_BINDINGS", version=state["version"],
                missing_paths=sorted(required_paths - set(bindings)), unknown_paths=sorted(set(bindings) - allowed_paths))
        for path, bound in bindings.items():
            _ids(bound, f"draft_bindings[{path}]")
            if not bound or set(bound) - set(ids):
                raise WorkspaceRepairError("Every draft binding needs active IDs included in draft_fact_ids",
                    code="DRAFT_BINDINGS", version=state["version"], invalid_binding_paths=[path],
                    unlisted_ids=sorted(set(bound) - set(ids)), empty_binding_paths=[path] if not bound else [])
        state["draft"] = deepcopy(draft)
        state["draft_fact_ids"] = ids.copy()
        state["draft_bindings"] = deepcopy(bindings)
        state["draft_sha256"] = _sha(draft)
        state["draft_valid"] = True

    def _patch_draft(self, operations: Any, state: dict) -> None:
        if not isinstance(operations, list) or not operations:
            raise WorkspaceError("draft_patch must be a nonempty list of operations")
        draft = deepcopy(state["draft"]) if state["draft"] is not None else {
            "基本信息": {}, "维修建议": [], "病害明细": []}
        bindings = deepcopy(state["draft_bindings"])
        records = {**state["facts"], **state["relations"]}
        fields = {
            "set_basic": {"op", "field", "value", "fact_ids"},
            "append_row": {"op", "section", "row", "fact_ids"},
            "replace_row": {"op", "section", "index", "row", "fact_ids"},
            "remove_row": {"op", "section", "index"},
            "bind_output": {"op", "path", "fact_ids"},
        }
        for operation_index, operation in enumerate(operations):
            if not isinstance(operation, dict) or not isinstance(operation.get("op"), str):
                raise WorkspaceRepairError("Each draft patch operation needs an op",
                    code="DRAFT_PATCH", version=state["version"], operation_index=operation_index,
                    missing_fields=["op"])
            op = operation["op"]
            if op not in fields or set(operation) != fields[op]:
                raise WorkspaceRepairError(f"Invalid draft patch operation or fields: {op}",
                    code="DRAFT_PATCH", version=state["version"], operation_index=operation_index,
                    missing_fields=sorted(fields.get(op, set()) - operation.keys()),
                    unexpected_fields=sorted(operation.keys() - fields.get(op, set())))
            if op != "remove_row":
                try:
                    ids = _ids(operation["fact_ids"], "draft_patch.fact_ids")
                except WorkspaceError as exc:
                    raise WorkspaceRepairError(str(exc), code="DRAFT_PATCH", version=state["version"],
                        operation_index=operation_index, invalid_fields=["fact_ids"]) from exc
                invalid_ids = [record_id for record_id in ids if record_id not in records or not records[record_id]["valid"]]
                if not ids or invalid_ids:
                    raise WorkspaceRepairError("Draft patch requires active fact/relation IDs",
                        code="DEPENDENCY_INVALID", version=state["version"], operation_index=operation_index,
                        invalid_ids=invalid_ids, missing_fields=["fact_ids"] if not ids else [])
            if op == "bind_output":
                path = operation["path"]
                allowed_paths = {f"基本信息.{field}" for field in draft["基本信息"]}
                allowed_paths.update(f"{section}.{index}" for section in ("维修建议", "病害明细")
                                     for index in range(len(draft[section])))
                if type(path) is not str or path not in allowed_paths:
                    raise WorkspaceRepairError("bind_output requires an existing draft field or row path",
                        code="DRAFT_PATCH", version=state["version"], operation_index=operation_index,
                        unknown_paths=[path] if type(path) is str else [])
                bindings[path] = ids.copy()
                continue
            if op == "set_basic":
                field = operation["field"]
                if not isinstance(field, str) or not field.strip():
                    raise WorkspaceRepairError("set_basic requires a nonempty field name",
                        code="DRAFT_PATCH", version=state["version"], operation_index=operation_index,
                        invalid_fields=["field"])
                draft["基本信息"][field] = deepcopy(operation["value"])
                bindings[f"基本信息.{field}"] = ids.copy()
                continue
            section = operation["section"]
            if section not in ("维修建议", "病害明细"):
                raise WorkspaceRepairError("Row operations require section 维修建议 or 病害明细",
                    code="DRAFT_PATCH", version=state["version"], operation_index=operation_index,
                    invalid_fields=["section"])
            rows = draft[section]
            if op in {"replace_row", "remove_row"}:
                index = operation["index"]
                if type(index) is not int or not 0 <= index < len(rows):
                    raise WorkspaceRepairError(f"Invalid draft row index for {section}: {index!r}",
                        code="DRAFT_PATCH", version=state["version"], operation_index=operation_index,
                        section=section, row_count=len(rows), valid_index_count=len(rows))
            if op == "remove_row":
                del rows[index]
                old_bindings = bindings
                bindings = {path: bound for path, bound in old_bindings.items() if not path.startswith(f"{section}.")}
                for new_index in range(len(rows)):
                    old_index = new_index if new_index < index else new_index + 1
                    bindings[f"{section}.{new_index}"] = old_bindings[f"{section}.{old_index}"]
            else:
                if not isinstance(operation["row"], dict):
                    raise WorkspaceRepairError("Draft row must be an object",
                        code="DRAFT_PATCH", version=state["version"], operation_index=operation_index,
                        invalid_fields=["row"])
                if op == "append_row":
                    index = len(rows)
                    rows.append(deepcopy(operation["row"]))
                else:
                    rows[index] = deepcopy(operation["row"])
                bindings[f"{section}.{index}"] = ids.copy()
        self._set_draft({"draft": draft, "draft_bindings": bindings,
                         "draft_fact_ids": sorted({record_id for bound in bindings.values() for record_id in bound})}, state)

    def _draft_update_impact(self, state: dict, changed: set[str], affected: set[str]) -> list[dict]:
        """Journal structural impact only; never decide or repair a claim.

        Called before any explicit draft patch. The same dependency edges used
        by invalidation map each changed ID to affected bindings. No report
        values, history or inferred engineering judgement are copied here.
        """
        hit = affected & set(state["draft_fact_ids"])
        if not hit:
            return []
        records = self._records(state)
        roots = {record_id: {record_id} for record_id in changed}
        while True:
            updated = False
            for record_id, record in records.items():
                inherited = set().union(*(roots.get(dependency, set())
                    for dependency in self._dependencies(record, records)))
                combined = roots.get(record_id, set()) | inherited
                if combined != roots.get(record_id, set()):
                    roots[record_id] = combined
                    updated = True
            if not updated:
                break
        result, bound_hits = [], set()
        for path, bindings in sorted(state["draft_bindings"].items()):
            relevant = set(bindings) & hit
            if relevant:
                bound_hits.update(relevant)
                result.append({"path": path, "scope": "draft_binding", "affected_bindings": [
                    {"record_id": identifier, "changed_ids": sorted(roots.get(identifier, set()))}
                    for identifier in sorted(relevant)]})
        # A draft can retain unused draft_fact_ids; report that association
        # honestly instead of inventing an output path for it.
        if hit - bound_hits:
            result.append({"path": None, "scope": "draft_fact_ids", "affected_bindings": [
                {"record_id": identifier, "changed_ids": sorted(roots.get(identifier, set()))}
                for identifier in sorted(hit - bound_hits)]})
        return result

    def apply_update(self, update: dict, *, imported_material_provenance=None) -> dict:
        # The keyword is host-only; it is intentionally absent from model update
        # fields/schema. Importers validate the child's pinned index first.
        if not isinstance(update, dict):
            raise WorkspaceError("update must be an object")
        allowed = {"base_version", "facts", "relations", "questions", "retract_fact_ids", "retract_relation_ids", "retract_question_ids", "draft", "draft_fact_ids", "draft_bindings", "draft_patch", "exclusions", "working_notes"}
        if set(update) - allowed:
            raise WorkspaceError(f"Unknown update fields: {sorted(set(update) - allowed)}")
        if "draft_patch" in update and any(key in update for key in ("draft", "draft_fact_ids", "draft_bindings")):
            raise WorkspaceError("draft_patch is mutually exclusive with explicit draft fields")
        with self._lock():
            state = self._load()
            self._check_version(state, update.get("base_version"))
            # Small host-owned transition metadata: saving facts or a memo is
            # not evidence that the deliverable itself changed. Keep the
            # existing atomic update and journal, without storing another draft.
            before_draft = _sha(state["draft"])
            before_bindings = _sha(state["draft_bindings"])
            before_valid = state["draft_valid"]
            before_rows = {section: len((state["draft"] or {}).get(section, []))
                           for section in ("维修建议", "病害明细")}
            if "working_notes" in update:
                state["working_notes"] = _working_notes(update["working_notes"])
            changed: set[str] = set()
            rewritten: set[str] = set()
            facts_changed = False
            for collection, singular in (("facts", "fact"), ("relations", "relation"), ("questions", "question")):
                retracted = _ids(update.get(f"retract_{singular}_ids", []), f"retract_{singular}_ids")
                for record_id in retracted:
                    if record_id not in state[collection]:
                        raise WorkspaceError(f"Cannot retract unknown {singular}: {record_id}")
                    del state[collection][record_id]
                    if self.material_index is not None and collection != "questions":
                        state["material_provenance"][collection].pop(record_id, None)
                    changed.add(record_id)
                incoming = update.get(collection, [])
                if not isinstance(incoming, list):
                    raise WorkspaceError(f"{collection} must be a list")
                for index, item in enumerate(incoming):
                    record = self._normalize_record(collection, item, state, path=f"{collection}[{index}]")
                    record_id = record["id"]
                    if record_id in rewritten or record_id in retracted:
                        raise WorkspaceError(f"Duplicate update/retraction for {record_id}")
                    if any(record_id in state[other] for other in ("facts", "relations", "questions") if other != collection):
                        raise WorkspaceError(f"Record IDs must be unique across collections: {record_id}")
                    state[collection][record_id] = record
                    rewritten.add(record_id)
                    changed.add(record_id)
                facts_changed |= collection != "questions" and bool(retracted or incoming)
            if imported_material_provenance is not None:
                if (self.material_index is None or not isinstance(imported_material_provenance, dict)
                        or set(imported_material_provenance) - {"facts", "relations"}):
                    raise WorkspaceError("Imported material provenance needs the pinned material index")
                for collection, origins in imported_material_provenance.items():
                    if not isinstance(origins, dict) or set(origins) - rewritten or set(origins) - set(state[collection]):
                        raise WorkspaceError("Imported material provenance must accompany the same atomic record update")
                    for record_id, rows in origins.items():
                        validate_material_provenance(rows, state[collection][record_id]["source_refs"],
                                                     self._manifest, self.material_index)
                        state["material_provenance"][collection][record_id] = deepcopy(rows)
            records = self._records(state)
            affected = set(changed)
            while True:
                dependents = {record_id for record_id, record in records.items() if self._dependencies(record, records) & affected}
                following = affected | dependents
                if following == affected:
                    break
                affected = following
            for record_id in affected - rewritten:
                if record_id in records:
                    records[record_id]["valid"] = False
                    records[record_id]["invalidated_by"] = sorted(changed)
            self._validate_graph(state, rewritten)
            if facts_changed or affected & (state["facts"].keys() | state["relations"].keys()):
                state["facts_version"] += 1
            draft_impact = self._draft_update_impact(state, changed, affected)
            if affected & set(state["draft_fact_ids"]):
                state["draft_valid"] = False
            self._exclusions(update.get("exclusions", []), state)
            if "draft_patch" in update:
                self._patch_draft(update["draft_patch"], state)
            elif any(key in update for key in ("draft", "draft_fact_ids", "draft_bindings")):
                self._set_draft(update, state)
            state["review"] = None
            update_effect = {
                "base_version": state["version"], "update_version": state["version"] + 1,
                "draft_changed": before_draft != _sha(state["draft"]),
                "draft_bindings_changed": before_bindings != _sha(state["draft_bindings"]),
                "draft_valid_changed": before_valid != state["draft_valid"],
                "draft_row_delta": {section: len((state["draft"] or {}).get(section, [])) - count
                                    for section, count in before_rows.items()},
            }
            self._advance(state, "update", {"update": update, "invalidated_ids": sorted(affected - rewritten),
                                             "draft_impact": draft_impact, "update_effect": update_effect})
            self._save(state)
            return self._snapshot(state)

    def _reading_progress(self, state: dict) -> dict:
        unread: list[dict] = []
        review_required: list[str] = []
        fully_read: list[str] = []
        for evidence_id, unit in self._units.items():
            size = len(unit["text"])
            reads = state["reads"].get(evidence_id, [])
            excluded = [[row["start"], row["end"]] for row in state["exclusions"] if row.get("evidence_id") == evidence_id]
            covered = _ranges(reads + excluded)
            cursor = 0
            for start, end in covered:
                if cursor < start:
                    unread.append({"evidence_id": evidence_id, "start": cursor, "end": start})
                cursor = max(cursor, end)
            if cursor < size:
                unread.append({"evidence_id": evidence_id, "start": cursor, "end": size})
            if size and not _covers(excluded, 0, size):
                review_required.append(evidence_id)
            if size and _covers(reads, 0, size):
                fully_read.append(evidence_id)
        excluded_images = {row["image_id"] for row in state["exclusions"] if "image_id" in row}
        image_reads = set(state["image_reads"])
        unverified_images = sorted(self._images.keys() - image_reads - excluded_images)
        review_required.extend(sorted(self._images.keys() - excluded_images))
        # A source supporting an output must also be reviewed even if an
        # exclusion was entered for that source elsewhere in the reading plan.
        records = {**state["facts"], **state["relations"]}
        pending = list(state["draft_fact_ids"])
        seen = set()
        while pending:
            record_id = pending.pop()
            if record_id in seen or record_id not in records:
                continue
            seen.add(record_id)
            record = records[record_id]
            pending.extend(self._dependencies(record, records))
            review_required.extend(ref.get("evidence_id", ref.get("image_id")) for ref in record["source_refs"])
        return {"unread_ranges": unread, "unread_evidence_ids": sorted({row["evidence_id"] for row in unread}),
                "fully_read_evidence_ids": fully_read, "review_required_evidence_ids": sorted(set(review_required)),
                "unverified_image_ids": unverified_images,
                "unverified_content": deepcopy(self._manifest.get("unverified_content", [])),
                "text_coverage_complete": not unread, "exclusions": deepcopy(state["exclusions"])}

    def _source_status(self, state: dict):
        if not self._manifest.get("unverified_content"):
            return None
        if self.source_gate is None:
            return None
        # Caller must consume this already loaded snapshot, never reenter the
        # workspace lock. The callback validates actual source/model ledgers.
        result = self.source_gate(deepcopy(state))
        if not isinstance(result, dict) or type(result.get("passed")) is not bool:
            raise WorkspaceError("Invalid trusted source-object gate")
        if (result.get("inventory_count") != len(self._manifest["unverified_content"])
                or result.get("processing_scope", {}).get("unverified_content") != self._manifest["unverified_content"]):
            raise WorkspaceError("Source-object gate does not retain the complete original inventory")
        return result

    def _preparation_blockers(self, state: dict, *, source_status=None) -> list[str]:
        blockers = []
        progress = self._reading_progress(state)
        if state["draft"] is None:
            blockers.append("draft_missing")
        elif not state["draft_valid"]:
            blockers.append("draft_invalidated")
        if progress["unread_ranges"] and not self.allow_quality_gaps:
            blockers.append("unread_source_ranges")
        if progress["unverified_image_ids"]:
            blockers.append("unverified_images")
        if progress["unverified_content"]:
            if source_status is None:
                blockers.append("unverified_manifest_content")
            elif not source_status["passed"]:
                blockers.extend(source_status.get("blockers") or ["source_object_review_incomplete"])
        for collection in ("facts", "relations", "questions"):
            for record_id, record in state[collection].items():
                if not record["valid"]:
                    blockers.append(f"invalidated_record:{record_id}")
                if collection == "questions" and record["status"] not in {"resolved", "unknown"}:
                    blockers.append(f"unresolved_question:{record_id}")
        return blockers

    def record_review(self, review: dict) -> dict:
        if not self.require_semantic_review:
            raise WorkspaceError("Semantic review is disabled by explicit user policy")
        if not isinstance(review, dict):
            raise WorkspaceError("review must be an object")
        with self._lock():
            state = self._load()
            self._check_version(state, review.get("workspace_version"))
            if state["draft_sha256"] is None or review.get("draft_sha256") != state["draft_sha256"]:
                raise WorkspaceConflictError("Review does not match the current draft hash")
            if "facts_version" in review and review["facts_version"] != state["facts_version"]:
                raise WorkspaceConflictError("Review does not match the current facts version")
            if type(review.get("passed")) is not bool:
                raise WorkspaceError("Review requires a boolean passed field")
            checked = _ids(review.get("checked_evidence_ids"), "checked_evidence_ids")
            progress = self._reading_progress(state)
            checked_ranges = []
            for evidence_id in checked:
                if evidence_id in self._images:
                    if evidence_id not in state["image_reads"]:
                        raise WorkspaceError(f"Review claims an unread image: {evidence_id}")
                    checked_ranges.append({"image_id": evidence_id})
                else:
                    ranges = state["reads"].get(evidence_id, [])
                    if evidence_id not in self._units or not any(start < end for start, end in ranges):
                        raise WorkspaceError(f"Review claims a source that was not actually read: {evidence_id}")
                    # Checked IDs mean the actual read portions, not excluded
                    # gaps. Persist their exact scope rather than pretending a
                    # partial read reviewed an entire long paragraph.
                    checked_ranges.extend({"evidence_id": evidence_id, "start": start, "end": end} for start, end in ranges if start < end)
            supplied_observations = review.get("checked_image_observation_ids", [])
            checked_observations = _ids(supplied_observations, "checked_image_observation_ids")
            index = state["image_observations"]
            if any(oid not in index or index[oid]["image_id"] not in checked for oid in checked_observations):
                raise WorkspaceError("Review claims an undelivered/mismatched image observation")
            required_observations = {oid for oid, row in index.items() if row["image_id"] in checked and row["requires_observation_id"]}
            if review["passed"] and required_observations - set(checked_observations):
                raise WorkspaceError("Review did not cover all delivered own-model image observations")
            issue_fields = ("unsupported_claims", "missing_items", "conflicts", "unresolved_items")
            if review.get("protocol") in {"bidirectional-semantic-review-v5", "bidirectional-semantic-review-v6"} or "citation_gaps" in review:
                issue_fields += ("citation_gaps",)
            for field in issue_fields:
                if not isinstance(review.get(field), list):
                    raise WorkspaceError(f"Review requires an explicit {field} list")
            call_ids = _ids(review.get("model_call_ids"), "model_call_ids")
            if not call_ids:
                raise WorkspaceError("A semantic review needs model call receipts")
            source_status = self._source_status(state)
            if source_status is not None:
                if (review.get("source_object_attestation_sha256") != source_status["attestation_sha256"]
                        or review.get("source_object_semantic_input_sha256") != source_status["semantic_input_sha256"]
                        or review.get("source_object_processing_scope_sha256") != _sha(source_status["processing_scope"])):
                    raise WorkspaceConflictError("Review does not match the current source-object attestation")
            if review["passed"]:
                blockers = self._preparation_blockers(state, source_status=source_status)
                if set(progress["review_required_evidence_ids"]) - set(checked):
                    blockers.append("reverse_source_review_incomplete")
                if any(review[field] for field in issue_fields):
                    blockers.append("review_has_unresolved_findings")
                if blockers:
                    raise WorkspaceError(f"Review cannot pass: {', '.join(blockers)}")
            accepted = deepcopy(review)
            accepted.update(facts_version=state["facts_version"], recorded_version=state["version"] + 1,
                            checked_read_ranges=checked_ranges,
                            checked_image_observation_ids=checked_observations,
                            image_observation_index_sha256=_sha(index))
            state["review"] = accepted
            self._advance(state, "review", {"review": accepted})
            self._save(state)
            return self._snapshot(state)

    def delivery_status(self) -> dict:
        with self._lock():
            state = self._load()
            if not self.require_semantic_review:
                # No source semantic gate, fabricated review, or truth judgement.
                blockers = self._preparation_blockers(state, source_status=None)
                return {"ready": not blockers, "blockers": blockers,
                    "draft_sha256": state["draft_sha256"], "workspace_version": state["version"],
                    "facts_version": state["facts_version"], "reading_progress": self._reading_progress(state),
                    "validation_scope": ("structure_version_bindings_with_explicit_read_gaps" if self.allow_quality_gaps else "structure_version_bindings_read_coverage_only"),
                    "quality_gaps_allowed": self.allow_quality_gaps,
                    "semantic_review": {"status": "not_run_by_user_policy", "performed": False,
                                        "passed": None, "independent": False}}
            source_status = self._source_status(state)
            blockers = self._preparation_blockers(state, source_status=source_status)
            review = state["review"]
            if not review:
                blockers.append("semantic_review_missing")
            elif not review["passed"]:
                blockers.append("semantic_review_not_passed")
            elif (review["draft_sha256"] != state["draft_sha256"] or review["facts_version"] != state["facts_version"]
                  or review["recorded_version"] != state["version"]
                  or review.get("image_observation_index_sha256", _sha({})) != _sha(state["image_observations"])):
                blockers.append("semantic_review_stale")
            if source_status is not None and review and (
                    review.get("source_object_attestation_sha256") != source_status["attestation_sha256"]
                    or review.get("source_object_semantic_input_sha256") != source_status["semantic_input_sha256"]
                    or review.get("source_object_processing_scope_sha256") != _sha(source_status["processing_scope"])):
                blockers.append("source_object_review_stale")
            result = {"ready": not blockers, "blockers": blockers, "draft_sha256": state["draft_sha256"],
                    "workspace_version": state["version"], "facts_version": state["facts_version"],
                    "reading_progress": self._reading_progress(state)}
            if source_status is not None:
                result["source_objects"] = deepcopy(source_status)
            return result
