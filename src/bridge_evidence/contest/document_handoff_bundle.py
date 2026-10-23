"""Bounded merge input with shared verbatim evidence, never semantic approval.

Only representation is deduplicated. Candidate facts, conflicting values and
model-authored context remain distinct. This module does not grant source reads;
the caller registers the validated exposed spans against the exact wire message
and confirms them only after the upstream input-delivery receipt.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re


SCHEMA = "cityagent-compact-handoff-v1"


def serialize_handoff_bundle(bundle):
    """The exact serialization to use when enforcing ``max_characters``."""
    return json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(serialize_handoff_bundle(value).encode()).hexdigest()


def _read(view):
    return {"tool": "read_workspace_page", "arguments": {"view": view}}


class _Sources:
    def __init__(self, manifest):
        self.units = {u["id"]: u for u in manifest["units"]}
        self.tables = {t["id"]: t for t in manifest.get("tables", [])}
        self.cells = {}
        for unit in self.units.values():
            offset = 0
            for cell in unit.get("cells", []):
                self.cells.setdefault(cell.get("source_path"), []).append({
                    "evidence_id": unit["id"], "start": offset, "end": offset + len(cell["text"])})
                offset += len(cell["text"]) + 3

    def source(self, ref):
        if not isinstance(ref, dict) or ref.get("evidence_id") not in self.units:
            raise ValueError("Handoff source is not a canonical E reference")
        unit = self.units[ref["evidence_id"]]
        start, end = ref.get("start", 0), ref.get("end", len(unit["text"]))
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(unit["text"])):
            raise ValueError("Handoff source interval is invalid")
        text = unit["text"][start:end]
        if "text" in ref and ref["text"] != text:
            raise ValueError("Handoff excerpt differs from immutable source text")
        result = {"evidence_id": unit["id"], "start": start, "end": end, "text": text}
        for name in ("table_id", "row_index", "segment_index"):
            if name in unit:
                result[name] = unit[name]
        if unit.get("table_id") in self.tables:
            result["header_kind"] = self.tables[unit["table_id"]].get("header_kind")
        layout, offset = [], 0
        for index, cell in enumerate(unit.get("cells", [])):
            if (cell.get("column_index", index) != index or cell.get("column_span", 1) != 1
                    or cell.get("vertical_merge") or " | " in cell["text"]):
                layout.append({"column": cell.get("column_index", index), "start": offset,
                    "end": offset + len(cell["text"]), "column_span": cell.get("column_span", 1),
                    "merge": cell.get("vertical_merge")})
            offset += len(cell["text"]) + 3
        if layout:
            result["cell_layout"] = layout
        return f'{unit["id"]}:{start}:{end}', result

    def related(self, ref):
        unit = self.units[ref["evidence_id"]]
        table = self.tables.get(unit.get("table_id"), {})
        headers = [{"evidence_id": eid} for eid in table.get("header_evidence_ids", [])
                   if eid in self.units and self.units[eid]["text"]]
        merged = [span for c in unit.get("cells", []) if c.get("merged_from_source")
                  for span in self.cells.get(c["merged_from_source"], []) if span["end"] > span["start"]]
        return headers, merged


def _material_metadata(value):
    # Full parser/XML provenance stays in the persisted handoff ledger. These
    # fields are source navigation relationships, never derived bridge facts.
    fields = ("kind", "table_ref", "canonical_table_id", "row", "column", "row_span", "col_span",
              "parent_table_ref", "parent_table_id", "header_kind", "header_note")
    return {key: deepcopy(value[key]) for key in fields if key in value}


def _workspace_question(question, handoff_evidence):
    result = {k: deepcopy(question[k]) for k in ("question", "status", "valid", "reason", "depends_on")
              if k in question}
    text = question.get("question", "")
    # This is only the exact program-authored duplicate-field envelope. Any
    # unrecognized free-form question is kept intact; no semantic summarizer.
    match = re.fullmatch(r'(核对基本字段“([^”]+)”的多个候选并决定字段归属)：(\[.*\])', text, re.S)
    if match:
        try:
            pairs = json.loads(match[3])
        except (ValueError, TypeError):
            pairs = None
        if (isinstance(pairs, list) and pairs and all(isinstance(p, list) and len(p) == 2
                and isinstance(p[0], str) and p[0] in handoff_evidence
                and handoff_evidence[p[0]].get("field") == match[2]
                and handoff_evidence[p[0]].get("candidate_value") == p[1] for p in pairs)):
            result.update(question=match[1], field=match[2], candidate_ids=[p[0] for p in pairs],
                          original_question_available_in_workspace=True)
    prefix = "核对分工范围的三类交付是否齐全："
    if isinstance(text, str) and text.startswith(prefix):
        try:
            scope = json.loads(text[len(prefix):])
        except (ValueError, TypeError):
            scope = None
        if isinstance(scope, dict) and isinstance(scope.get("task_id"), str):
            result.update(question=prefix[:-1], task_id=scope["task_id"], coverage_scope=scope,
                          original_question_available_in_workspace=True)
    return result


def _output_bindings(snapshot):
    draft = snapshot.get("draft") or {}
    paths = {f"基本信息.{key}": value for key, value in draft.get("基本信息", {}).items()}
    for section in ("病害明细", "维修建议"):
        paths.update({f"{section}.{i}": row for i, row in enumerate(draft.get(section, []))})
    bindings = snapshot.get("draft_bindings", {})
    if set(bindings) - set(paths):
        raise ValueError("Workspace binding refers to a path absent from the current draft")
    return {"draft_valid": snapshot.get("draft_valid"), "draft_sha256": snapshot.get("draft_sha256"),
        "path_bindings": {path: deepcopy(bindings.get(path, [])) for path in paths},
        "empty_paths": [path for path, value in paths.items() if value in (None, "", [], {})],
        "meaning": "Current saved draft bindings only; existence is not semantic approval or Word export confirmation."}


def build_handoff_bundle(manifest, handoff_evidence, child_summary, workspace_snapshot,
                         max_characters=60000):
    """Create one bounded merge input; huge entries remain explicit navigation.

    Character count is the length of ``serialize_handoff_bundle``. No source
    sentence, business value or model context is shortened to meet the bound.
    If even the identity and complete pending-ID inventory cannot fit, fail
    explicitly instead of claiming a complete handoff.
    """
    if type(max_characters) is not int or max_characters <= 0:
        raise ValueError("max_characters must be a positive integer")
    digest = _digest(manifest)
    identity = workspace_snapshot.get("identity", {})
    if identity.get("manifest_sha256", digest) != digest:
        raise ValueError("Workspace belongs to another canonical manifest")
    sources = _Sources(manifest)
    facts = workspace_snapshot.get("facts", {})
    if not isinstance(handoff_evidence, dict) or not isinstance(facts, dict):
        raise ValueError("Handoff evidence and workspace facts must be keyed mappings")
    # Never deduplicate business candidates, even if all their values agree.
    candidate_ids = list(handoff_evidence)
    if any(not isinstance(key, str) or not key for key in candidate_ids):
        raise ValueError("Every candidate needs a stable nonempty fact ID")
    output_bindings = _output_bindings(workspace_snapshot)
    bundle = {"schema": SCHEMA, "document_sha256": manifest["document_sha256"],
        "manifest_sha256": digest, "workspace_version": workspace_snapshot.get("version"),
        "semantics": "All candidates and model_context are unreviewed model claims. Source text is verbatim; candidate headers are navigation only. Presence here does not confirm upstream delivery or semantic review.",
        "candidate_count": len(candidate_ids), "candidate_ids_sha256": _digest(sorted(candidate_ids)),
        "candidates": [], "source_map": {}, "issues": {}, "children": [],
        "current_output": output_bindings,
        "remaining": {"candidate_ids": candidate_ids.copy(), "navigation_only_candidate_ids": [],
            "issue_ids": [], "child_detail_ids": [], "facts": _read("facts"),
            "questions": _read("questions"), "source": {"tool": "read_source", "arguments_note": "Use the listed canonical evidence_id; the program owns pagination."}}}
    issues = {}

    def issue(value, kind, identifier=None):
        key = identifier or "child_issue_" + _digest({"kind": kind, "value": value})[:24]
        record = {"kind": kind, "value": deepcopy(value)}
        if key in issues and issues[key] != record:
            raise ValueError("Conflicting records share an issue ID")
        issues[key] = record
        return key

    def child_issue_ids(task_id, unresolved):
        result = []
        for position, value in enumerate(unresolved):
            # import_candidates uses this exact host-owned ID. Reuse it so
            # review submissions address an actual workspace question rather
            # than a synthetic alias with the same text.
            qid = "Q_" + _digest([task_id, position, value])[:24]
            saved = workspace_snapshot.get("questions", {}).get(qid)
            if saved is not None and saved.get("question") == value:
                key = issue(_workspace_question(saved, handoff_evidence), "workspace_question", qid)
            else:
                key = issue(value, "child_note_not_workspace_question")
            if key not in result:
                result.append(key)
        return result

    for key, question in workspace_snapshot.get("questions", {}).items():
        if question.get("status") != "resolved" or question.get("valid") is not True:
            issue(_workspace_question(question, handoff_evidence), "workspace_question", key)
    child_details = {}
    for child in child_summary:
        task_id = child["task_id"]
        handoff = child.get("handoff") or {}
        child_issues = child_issue_ids(task_id, handoff.get("unresolved", child.get("pending_unresolved", [])))
        record = {"task_id": task_id, "status": child.get("status"),
                  "entry_count": child.get("entry_count"), "issue_ids": list(dict.fromkeys(child_issues))}
        coverage_id = "Q_coverage_" + _digest(task_id)[:24]
        if coverage_id in workspace_snapshot.get("questions", {}):
            record["coverage_question_id"] = coverage_id
        primary_ids = child.get("primary_evidence_ids", [])
        if primary_ids:
            record["scope"] = {"first_source": primary_ids[0], "last_source": primary_ids[-1],
                "assigned_unit_count": len(primary_ids), "boundaries_are_not_semantic_coverage": True}
        bundle["children"].append(record)
        detail = {key: deepcopy(child[key]) for key in ("headings", "primary_evidence_ids", "error") if key in child}
        if detail:
            child_details[task_id] = detail
    candidates = []
    for fact_id, evidence in handoff_evidence.items():
        fact = facts.get(fact_id)
        matches = fact is None or (fact.get("value") == evidence.get("candidate_value")
            and fact.get("source_refs") == evidence.get("source_refs"))
        refs = evidence.get("source_refs", []) if matches else fact.get("source_refs", [])
        candidate = {"fact_id": fact_id, "task_id": evidence.get("task_id"),
            "section": evidence.get("section"),
            "field": evidence.get("field"), "value": deepcopy(fact["value"] if fact is not None else evidence.get("candidate_value")),
            "status": fact.get("status") if fact else None, "valid": fact.get("valid") if fact else None,
            "model_context": deepcopy(evidence.get("model_context", {})) if matches else {},
            "current_output_paths": [path for path, ids in output_bindings["path_bindings"].items() if fact_id in ids],
            "source_keys": [], "header_keys": [], "context_keys": [], "material_context": [],
            "issue_ids": child_issue_ids(evidence.get("task_id"), evidence.get("unresolved", []))}
        if not matches:
            candidate["context_missing"] = ["Child handoff differs from the current workspace fact; prior child context is not reused."]
        own_sources = {}

        def add(ref, destination):
            key, row = sources.source(ref)
            own_sources[key] = row
            if key not in candidate[destination]:
                candidate[destination].append(key)
            return key

        # Validate provided quotes even if the current workspace has superseded
        # this candidate. Never hide a forged/mismatched historical excerpt.
        for quote in evidence.get("source_excerpts", []) + evidence.get("table_headers", []):
            sources.source(quote)
        for original_ref in evidence.get("source_refs", []):
            sources.source(original_ref)
        for ref in refs:
            add(ref, "source_keys")
            headers, merged = sources.related(ref)
            for header in headers:
                add(header, "header_keys")
            for anchor in merged:
                add(anchor, "context_keys")
        if matches:
            for header in evidence.get("table_headers", []):
                add(header, "header_keys")
            for material in evidence.get("material_evidence", []):
                material_identity = material.get("identity", {})
                if (material_identity.get("document_sha256") != manifest["document_sha256"]
                        or material_identity.get("manifest_sha256") != digest):
                    raise ValueError("Material handoff belongs to another document/manifest")
                contexts = [{"source_key": add(span, "context_keys"), "role": span.get("role", "context")}
                            for span in material.get("context", []) if span.get("start") != span.get("end")]
                candidate["material_context"].append({"ref": deepcopy(material.get("source_ref")),
                    "metadata": _material_metadata(material.get("source_metadata", {})), "context": contexts})
        if not candidate["source_keys"]:
            candidate.setdefault("context_missing", []).append("No canonical original-source reference was provided.")
        # Empty optional containers and the redundant child-local entry ID
        # carry no source/claim information. Stable fact_id still addresses the
        # exact persisted candidate; output paths stay explicit even if empty.
        for name in ("field", "model_context", "header_keys", "context_keys", "material_context", "issue_ids"):
            if candidate[name] in (None, {}, []):
                candidate.pop(name)
        candidates.append((candidate, own_sources))

    bundle["remaining"]["issue_ids"] = list(issues)
    bundle["issue_count"] = len(issues)
    bundle["issue_ids_sha256"] = _digest(sorted(issues))
    bundle["remaining"]["child_detail_ids"] = list(child_details)

    def fits(value):
        return len(serialize_handoff_bundle(value)) <= max_characters

    if not fits(bundle):
        raise ValueError("Character budget cannot contain mandatory identity, child states and complete pending IDs")
    # Reserve all IDs first. Optional records are added whole, with omission
    # always represented by the corresponding remaining inventory.
    for key, value in issues.items():
        proposal = deepcopy(bundle)
        proposal["issues"][key] = value
        proposal["remaining"]["issue_ids"].remove(key)
        if fits(proposal):
            bundle = proposal
    for task_id, detail in child_details.items():
        proposal = deepcopy(bundle)
        next(c for c in proposal["children"] if c["task_id"] == task_id).update(detail)
        proposal["remaining"]["child_detail_ids"].remove(task_id)
        if fits(proposal):
            bundle = proposal
    for candidate, own_sources in candidates:
        fact_id = candidate["fact_id"]
        proposal = deepcopy(bundle)
        proposal["candidates"].append(candidate)
        proposal["source_map"].update(own_sources)
        proposal["remaining"]["candidate_ids"].remove(fact_id)
        if fits(proposal):
            bundle = proposal
            continue
        # Large rows are navigable rather than sliced mid-sentence. A stub is
        # not a delivered candidate and stays in remaining.candidate_ids.
        stub = {key: candidate[key] for key in ("fact_id", "task_id", "section", "field", "current_output_paths") if key in candidate}
        stub.update(navigation_only=True, missing=["complete candidate value/context and source excerpts"],
                    evidence_ids=list(dict.fromkeys(s["evidence_id"] for s in own_sources.values())),
                    read=_read("facts"))
        proposal = deepcopy(bundle)
        proposal["candidates"].append(stub)
        proposal["remaining"]["navigation_only_candidate_ids"].append(fact_id)
        if fits(proposal):
            bundle = proposal
    validate_handoff_bundle(manifest, bundle, max_characters=max_characters)
    return bundle


def validate_handoff_bundle(manifest, bundle, max_characters=None):
    """Validate the bundle and return only its actually exposed original spans.

    Feed these spans and the *exact final message* to the delivery ledger. This
    function does not assert that a main-model request accepted that message.
    """
    if (bundle.get("schema") != SCHEMA or bundle.get("document_sha256") != manifest["document_sha256"]
            or bundle.get("manifest_sha256") != _digest(manifest)):
        raise ValueError("Handoff bundle belongs to another document/manifest")
    if max_characters is not None and len(serialize_handoff_bundle(bundle)) > max_characters:
        raise ValueError("Handoff bundle exceeds character budget")
    source = _Sources(manifest)
    exposed = []
    for key, row in bundle["source_map"].items():
        expected_key, expected = source.source(row)
        if key != expected_key or row != expected:
            raise ValueError("Handoff source map differs from canonical text or metadata")
        exposed.append({name: row[name] for name in ("evidence_id", "start", "end", "text")})
    ids = [c["fact_id"] for c in bundle["candidates"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate fact IDs")
    complete = [c["fact_id"] for c in bundle["candidates"] if not c.get("navigation_only")]
    inventory = complete + bundle["remaining"]["candidate_ids"]
    if (len(inventory) != bundle["candidate_count"] or len(inventory) != len(set(inventory))
            or _digest(sorted(inventory)) != bundle["candidate_ids_sha256"]):
        raise ValueError("Candidate inventory was silently removed or duplicated")
    issue_inventory = list(bundle["issues"]) + bundle["remaining"]["issue_ids"]
    if (len(issue_inventory) != bundle["issue_count"] or len(issue_inventory) != len(set(issue_inventory))
            or _digest(sorted(issue_inventory)) != bundle["issue_ids_sha256"]):
        raise ValueError("Unresolved issue inventory was silently removed or duplicated")
    for candidate in bundle["candidates"]:
        if candidate.get("navigation_only"):
            if candidate["fact_id"] not in bundle["remaining"]["candidate_ids"]:
                raise ValueError("Navigation-only candidate is missing from pending IDs")
            continue
        if candidate["fact_id"] in bundle["remaining"]["candidate_ids"]:
            raise ValueError("Complete candidate is also marked pending")
        for name in ("source_keys", "header_keys", "context_keys"):
            if any(key not in bundle["source_map"] for key in candidate.get(name, [])):
                raise ValueError("Candidate references source text absent from the bundle")
        known_issues = set(bundle["issues"]) | set(bundle["remaining"]["issue_ids"])
        if set(candidate.get("issue_ids", [])) - known_issues:
            raise ValueError("Candidate unresolved issue disappeared")
    return exposed
