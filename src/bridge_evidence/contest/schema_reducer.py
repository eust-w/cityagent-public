"""Strict evidence-bound chunk output and loss-averse, deterministic reduction.

This module does not infer missing facts, prove semantic entailment, or declare
document coverage. The caller supplies the IDs actually made available to a
model and separately tracks which primary document units were processed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any


SECTIONS = ("基本信息", "维修建议", "病害明细")
_UNKNOWN = frozenset({"未知", "无法判断", "不详", "未提供", "无法确定", "unknown"})


class ChunkValidationError(ValueError):
    """A chunk is incomplete, malformed, or lacks authorized provenance."""


class UnresolvedConflictsError(ValueError):
    """Reduction cannot publish an arbitrary winner for conflicting facts."""

    def __init__(self, conflicts: list[dict]):
        self.conflicts = deepcopy(conflicts)
        super().__init__("Unresolved basic-information conflicts: " +
                         ", ".join(item["field"] for item in conflicts))


def _scalar(value: Any) -> bool:
    return (value is None or isinstance(value, (str, bool)) or type(value) is int
            or (type(value) is float and math.isfinite(value)))


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ChunkValidationError("duplicate_json_key:" + key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ChunkValidationError("nonfinite_json_number:" + value)


def _check_ids(ids: Any, allowed_ids: set[str], primary_ids: set[str] | None,
               *, required: bool = True) -> list[str]:
    if not isinstance(ids, list) or any(not isinstance(item, str) or not item for item in ids):
        raise ChunkValidationError("evidence_ids_must_be_string_list")
    if required and not ids:
        raise ChunkValidationError("missing_evidence_ids")
    if not set(ids).issubset(allowed_ids):
        raise ChunkValidationError("unauthorized_evidence_ids")
    if required and primary_ids is not None and not set(ids).intersection(primary_ids):
        raise ChunkValidationError("context_only_fact_without_primary_evidence")
    return sorted(set(ids))


def _validate(data: Any, allowed_ids: set[str], primary_ids: set[str] | None) -> dict:
    if not isinstance(allowed_ids, (set, frozenset)) or any(
            not isinstance(item, str) or not item for item in allowed_ids):
        raise ChunkValidationError("allowed_ids_must_be_set_of_nonempty_strings")
    if primary_ids is not None:
        if not isinstance(primary_ids, (set, frozenset)) or not primary_ids.issubset(allowed_ids):
            raise ChunkValidationError("primary_ids_must_be_subset_of_allowed_ids")
    if not isinstance(data, dict) or not {*SECTIONS, "evidence"}.issubset(data):
        raise ChunkValidationError("missing_report_sections_or_evidence")
    if any(not _scalar(value) for key, value in data.items()
           if key not in {*SECTIONS, "evidence"}):
        raise ChunkValidationError("extra_top_level_fields_must_be_scalar")
    basic = data["基本信息"]
    if not isinstance(basic, dict) or any(not isinstance(key, str) or not key or not _scalar(value)
                                          for key, value in basic.items()):
        raise ChunkValidationError("invalid_basic_information")
    evidence = data["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != set(SECTIONS):
        raise ChunkValidationError("invalid_evidence_sections")
    basic_evidence = evidence["基本信息"]
    if not isinstance(basic_evidence, dict) or not set(basic_evidence).issubset(basic):
        raise ChunkValidationError("invalid_basic_evidence_fields")
    result = deepcopy(data)
    for field, value in basic.items():
        if field in basic_evidence or not _empty(value):
            try:
                result["evidence"]["基本信息"][field] = _check_ids(
                    basic_evidence.get(field, []), allowed_ids, primary_ids, required=not _empty(value))
            except ChunkValidationError as error:
                path = "evidence.基本信息[" + json.dumps(field, ensure_ascii=False) + "]"
                raise ChunkValidationError(f"{error}:{path}") from error
    for section, content_fields in (("维修建议", {"建议内容"}),
                                    ("病害明细", {"病害部位", "病害类型", "病害描述"})):
        rows, row_evidence = data[section], evidence[section]
        if not isinstance(rows, list) or not isinstance(row_evidence, list) or len(rows) != len(row_evidence):
            raise ChunkValidationError("row_evidence_length_mismatch:" + section)
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or any(not isinstance(key, str) or not key or not _scalar(value)
                                               for key, value in row.items()):
                raise ChunkValidationError("invalid_row:" + section)
            if not any(key in row and not _empty(row[key]) for key in content_fields):
                raise ChunkValidationError("row_has_no_content:" + section)
            try:
                result["evidence"][section][index] = _check_ids(
                    row_evidence[index], allowed_ids, primary_ids)
            except ChunkValidationError as error:
                raise ChunkValidationError(f"{error}:evidence.{section}[{index}]") from error
    return result


def parse_chunk_result(text: str, allowed_ids: set[str],
                       primary_ids: set[str] | None = None) -> dict:
    """Parse one complete JSON object, optionally enclosed in a JSON code fence.

    A nonempty basic field and every list row must cite available evidence. If
    primary IDs are supplied, each contribution must also cite its own chunk;
    neighboring context alone may not produce duplicate report contributions.
    Empty sections are valid: coverage is established by the orchestration code.
    """
    if not isinstance(text, str) or not text.strip():
        raise ChunkValidationError("empty_chunk_output")
    raw = text.strip()
    if raw.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
        if match is None:
            raise ChunkValidationError("invalid_json_fence")
        raw = match.group(1)
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except ChunkValidationError:
        raise
    except (ValueError, TypeError, RecursionError) as error:
        raise ChunkValidationError("invalid_or_incomplete_json") from error
    return _validate(data, allowed_ids, primary_ids)


class EvidenceReducer:
    """Combine validated model contributions without inventing or overwriting facts.

    Exact row values plus the exact evidence-ID set define an overlap identity.
    For each identity the largest within-task multiplicity is retained. This
    deduplicates repeated chunk processing without collapsing equal real rows
    from distinct sources, or multiple equal defects within one source row.
    Set ``deduplicate_overlapping_sources=False`` when task spans are disjoint
    but reuse source IDs; equal rows from different tasks then remain separate.
    """

    def __init__(self, deduplicate_overlapping_sources: bool = True) -> None:
        if type(deduplicate_overlapping_sources) is not bool:
            raise ValueError("deduplicate_overlapping_sources must be a boolean")
        self._deduplicate_overlapping_sources = deduplicate_overlapping_sources
        self._tasks: dict[str, dict] = {}
        self._basic: dict[str, list[dict]] = {}
        self._empty_basic: dict[str, Any] = {}
        self._rows: dict[str, list[dict]] = {section: [] for section in SECTIONS[1:]}
        self._row_lookup: dict[str, dict[str, list[dict]]] = {section: {} for section in SECTIONS[1:]}
        self._resolutions: dict[str, dict] = {}

    @staticmethod
    def _conflict_id(field: str) -> str:
        return "basic-" + hashlib.sha256(field.encode("utf-8")).hexdigest()[:16]

    def add(self, task_id: str, data: dict) -> None:
        """Add a validated task once; identical retries are idempotent.

        Authorization against source IDs belongs to ``parse_chunk_result``;
        the structural check here protects callers from malformed dictionaries.
        """
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a nonempty string")
        # Collect supplied citations solely for structural revalidation. This
        # does not purport to establish that the model actually saw those IDs.
        try:
            evidence = data["evidence"]
            refs = list(evidence["基本信息"].values()) + evidence["维修建议"] + evidence["病害明细"]
            allowed = {item for group in refs if isinstance(group, list)
                       for item in group if isinstance(item, str)}
        except (KeyError, TypeError, AttributeError) as error:
            raise ChunkValidationError("invalid_evidence") from error
        validated = _validate(data, allowed, None)
        if task_id in self._tasks:
            if _canonical(self._tasks[task_id]) != _canonical(validated):
                raise ValueError("task_id already has a different contribution: " + task_id)
            return
        self._tasks[task_id] = deepcopy(validated)
        for field, value in validated["基本信息"].items():
            if _empty(value):
                self._empty_basic.setdefault(field, value)
                continue
            candidates = self._basic.setdefault(field, [])
            known_values = {_canonical(item["value"]) for item in candidates}
            if _canonical(value) not in known_values:
                self._resolutions.pop(self._conflict_id(field), None)
            candidates.append({"value": deepcopy(value),
                               "evidence_ids": validated["evidence"]["基本信息"][field].copy(),
                               "task_id": task_id})
        for section in SECTIONS[1:]:
            counts: Counter[str] = Counter()
            for row, refs in zip(validated[section], validated["evidence"][section]):
                # A source unit may be divided into disjoint character spans
                # that share an evidence ID. In that mode task identity keeps
                # independent occurrences distinct instead of assuming overlap.
                identity = [row, refs] if self._deduplicate_overlapping_sources else [task_id, row, refs]
                key = _canonical(identity)
                occurrence = counts[key]
                counts[key] += 1
                existing = self._row_lookup[section].setdefault(key, [])
                if occurrence < len(existing):
                    existing[occurrence]["task_ids"].append(task_id)
                else:
                    record = {"value": deepcopy(row), "evidence_ids": refs.copy(), "task_ids": [task_id]}
                    existing.append(record)
                    self._rows[section].append(record)

    def conflicts(self) -> list[dict]:
        """Return only unresolved, differing basic-field candidates."""
        result = []
        for field, candidates in self._basic.items():
            conflict_id = self._conflict_id(field)
            if len({_canonical(item["value"]) for item in candidates}) > 1 and conflict_id not in self._resolutions:
                result.append({"id": conflict_id, "field": field, "candidates": deepcopy(candidates)})
        return result

    def resolve(self, conflict_id: str, value: Any, evidence_ids: list[str],
                allowed_ids: set[str]) -> None:
        """Choose an existing candidate or an explicitly unknown value after review.

        ``allowed_ids`` must come from the review's actual accessible evidence,
        including any newly fetched sources; it must not be copied from an
        untrusted model's citations. Review does not create novel fact values.
        """
        conflict = next((item for item in self.conflicts() if item["id"] == conflict_id), None)
        if conflict is None:
            raise ValueError("unknown_or_already_resolved_conflict:" + str(conflict_id))
        if not _scalar(value):
            raise ValueError("resolution_value_must_be_scalar")
        candidate_values = {_canonical(item["value"]) for item in conflict["candidates"]}
        explicit_unknown = value is None or (isinstance(value, str) and value.strip().casefold() in _UNKNOWN)
        if _canonical(value) not in candidate_values and not explicit_unknown:
            raise ValueError("resolution_must_choose_candidate_or_explicit_unknown")
        if not isinstance(allowed_ids, (set, frozenset)) or any(
                not isinstance(item, str) or not item for item in allowed_ids):
            raise ValueError("allowed_ids_must_be_set_of_nonempty_strings")
        refs = _check_ids(evidence_ids, allowed_ids, None)
        self._resolutions[conflict_id] = {"value": deepcopy(value), "evidence_ids": refs,
                                           "candidate_values": sorted(candidate_values)}

    def report(self) -> dict:
        """Return the three report sections; never silently choose a conflict."""
        conflicts = self.conflicts()
        if conflicts:
            raise UnresolvedConflictsError(conflicts)
        basic = deepcopy(self._empty_basic)
        for field, candidates in self._basic.items():
            resolution = self._resolutions.get(self._conflict_id(field))
            basic[field] = deepcopy(resolution["value"] if resolution else candidates[0]["value"])
        return {"基本信息": basic,
                **{section: [deepcopy(record["value"]) for record in self._rows[section]]
                   for section in SECTIONS[1:]}}

    def provenance(self) -> dict:
        """Return a serializable audit, including candidates and review decisions."""
        basic = {}
        for field, candidates in self._basic.items():
            record = {"candidates": deepcopy(candidates),
                      "evidence_ids": sorted({item for candidate in candidates for item in candidate["evidence_ids"]}),
                      "task_ids": [candidate["task_id"] for candidate in candidates]}
            resolution = self._resolutions.get(self._conflict_id(field))
            if resolution:
                record["resolution"] = deepcopy(resolution)
                record["value"] = deepcopy(resolution["value"])
                record["evidence_ids"] = resolution["evidence_ids"].copy()
            elif len({_canonical(candidate["value"]) for candidate in candidates}) == 1:
                record["value"] = deepcopy(candidates[0]["value"])
            else:
                record["conflict_id"] = self._conflict_id(field)
            basic[field] = record
        return {"reduction_policy": {
                    "deduplicate_overlapping_sources": self._deduplicate_overlapping_sources,
                },
                "基本信息": basic,
                **{section: [{"evidence_ids": row["evidence_ids"].copy(), "task_ids": row["task_ids"].copy()}
                             for row in self._rows[section]] for section in SECTIONS[1:]}}
