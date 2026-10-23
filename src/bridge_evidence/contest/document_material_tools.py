"""Bounded MinerU navigation over exact canonical source intervals.

The index owns binding and provenance. This adapter renders canonical original
text only; material headings, table relationships and header candidates remain
navigation metadata. Its continuation queue uses the existing durable reading
facade and source-delivery outbox rather than model-maintained offsets.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from urllib.parse import quote, unquote


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _compact(value):
    """Exact XML and parser provenance stay in the persisted index sidecar."""
    if isinstance(value, list):
        return [_compact(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {key: _compact(item) for key, item in value.items()
            if key not in {"source_provenance", "source_path", "provenance", "source_map",
                           "text", "cells", "heading_spans", "cell_context", "merged_from_source"}
            and "xml" not in key.lower() and not key.startswith("mineru_")}


class UnresolvedMaterialReference(ValueError):
    def __init__(self, result):
        self.result = result
        super().__init__("Material reference has no unambiguous canonical mapping")

    def as_tool_error(self):
        fallback = _compact(self.result.get("fallback", {}))
        return {"ok": False, "status": "unresolved_reference", "error": {
            "type": type(self).__name__, "message": str(self), "recoverable": True},
            "material_ref": self.result.get("qualified_ref", self.result.get("ref")),
            "unresolved": _compact(self.result.get("unresolved", [])),
            "fallback": fallback, "semantic_review": False,
            "next_action": "Use read_source on a listed canonical E candidate to inspect it; candidates are not confirmed mappings. If no candidate is listed, use read_outline or search_in_document."}


class DocumentMaterialTools:
    def __init__(self, tools, index):
        expected = hashlib.sha256(_json(tools.manifest).encode()).hexdigest()
        identity = index.identity
        if (identity.get("document_sha256") != tools.manifest["document_sha256"]
                or identity.get("manifest_sha256") != expected):
            raise ValueError("Material index belongs to another document or canonical manifest")
        self.tools, self.index = tools, index
        self._bundles = {}

    @staticmethod
    def looks_like_ref(ref):
        return isinstance(ref, str) and (ref.startswith(("#/", "mineru:")) or "#/" in ref)

    def normalize(self, ref):
        return self.index.normalize_ref(ref)

    def section_id(self, ref):
        return "@material/" + quote(self.normalize(ref), safe="")

    def _ref(self, section_id):
        if not section_id.startswith("@material/"):
            raise ValueError("Invalid material continuation")
        return self.normalize(unquote(section_id[len("@material/"):]))

    def resolve(self, ref):
        local = self.normalize(ref)
        result = (self.index.section(local) if local in self.index.section_ids
                  else self.index.resolve(local))
        # A section/packet can expose its exact subset while retaining every
        # unresolved reference. An ambiguous individual reference must not
        # silently select one candidate (or claim the whole source was read).
        if result.get("ok") is False and not (local.startswith(("#/sections/", "#/packets/"))
                                               and result.get("spans")):
            raise UnresolvedMaterialReference(result)
        return result

    def canonical_spans(self, ref):
        return [self._span(span) for span in self.resolve(ref).get("spans", [])]

    def canonical_table_id(self, ref):
        result = self.resolve(ref)
        table_id = result.get("metadata", {}).get("canonical_table_id")
        if table_id not in self.tools._tables:
            raise ValueError("Material reference does not identify a canonical table; use read_source to read its exact spans")
        return table_id

    def _span(self, value):
        identifier, start, end = value.get("evidence_id"), value.get("start"), value.get("end")
        unit = self.tools._index.get(identifier)
        if (unit is None or type(start) is not int or type(end) is not int
                or not 0 <= start <= end <= len(unit["text"])):
            raise ValueError("Material reference has an invalid canonical source interval")
        text = unit["text"][start:end]
        if "text" in value and value["text"] != text:
            raise ValueError("Material reference text differs from the canonical source")
        return {"evidence_id": identifier, "start": start, "end": end, "text": text}

    def entries(self, section_id):
        if section_id in self._bundles:
            return deepcopy(self._bundles[section_id])
        ref = self._ref(section_id)
        resolved = self.resolve(ref)
        metadata = {"ref": self.index.qualify_ref(ref),
                    "context": _compact(resolved.get("metadata", {})),
                    "unresolved": _compact(resolved.get("unresolved", [])),
                    "fallback": _compact(resolved.get("fallback", {})),
                    "meaning": "Navigation and candidate headers are not semantic findings; only canonical spans are original text."}
        metadata_text = _json(metadata)
        entries = [{"kind": "context_manifest", "text": metadata_text,
                    "start": 0, "end": len(metadata_text)}]
        primary = [self._span(value) for value in resolved.get("spans", [])]
        primary_keys = {(span["evidence_id"], span["start"], span["end"]) for span in primary}
        seen = set()
        for role, values in (("context", resolved.get("context", [])), ("primary", primary)):
            for value in values:
                span = self._span(value)
                key = (span["evidence_id"], span["start"], span["end"])
                if key in seen or role == "context" and key in primary_keys:
                    continue
                seen.add(key)
                entries.append({"kind": "original", **span,
                    "role": value.get("role", "material_context") if role == "context" else "primary"})
        self._bundles[section_id] = entries
        return deepcopy(entries)

    def page(self, section_id, cursor, max_units):
        entries = self.entries(section_id)
        if (type(cursor) is not int or not 0 <= cursor <= len(entries)
                or type(max_units) is not int or max_units <= 0):
            raise ValueError("Invalid material continuation")
        if cursor == len(entries):
            return {"ok": True, "status": "read", "spans": [], "continuation": None}
        ref = self._ref(section_id)

        def render(stop):
            result = {"ok": True, "status": "read_material", "material_ref": self.index.qualify_ref(ref),
                      "spans": [], "context": [],
                      "continuation": ({"tool": "read_section", "arguments": {
                          "section_id": section_id, "cursor": stop, "max_units": max_units}}
                          if stop < len(entries) else None)}
            for entry in entries[cursor:stop]:
                if entry["kind"] == "context_manifest":
                    result["material_context"] = json.loads(entry["text"])
                else:
                    span = {key: entry[key] for key in ("evidence_id", "start", "end", "text")}
                    if entry["role"] != "primary":
                        span["role"] = entry["role"]
                    result["spans" if entry["role"] == "primary" else "context"].append(span)
            return result

        best = None
        # Metadata does not consume a source-unit allowance.
        limit = min(len(entries), cursor + max_units + (cursor == 0))
        for stop in range(cursor + 1, limit + 1):
            result = render(stop)
            if not self.tools._fits(result):
                break
            best = result
        if best is not None:
            return best
        # The standard source-fragment path bounds even a long header or
        # metadata envelope. It never reattaches an unbounded whole header.
        entry = entries[cursor]
        return self.tools._recovery_piece(section_id, cursor, entry["start"], entry["end"])

    def read_source(self, ref, start=0, end=None):
        if start != 0 or end is not None:
            raise ValueError("Material references use automatic continuation; use their canonical E reference for an explicit character interval")
        return self.page(self.section_id(ref), 0, 64)

    def aliases_for(self, evidence_id):
        return [self.index.qualify_ref(ref) for ref in self.index.aliases_for(evidence_id)]
