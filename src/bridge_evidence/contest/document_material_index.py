"""Document-bound MinerU navigation over immutable canonical evidence.

MinerU controls organization only. Every returned source span is sliced from the
canonical manifest after original DOCX hash/XML validation. Ambiguous aliases
never pick an occurrence, and no method grants delivery or semantic coverage.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import zipfile

from lxml import etree

from .evidence_manifest import build_manifest, _paragraph_events
from .run_state import atomic_json


class MaterialReferenceError(ValueError):
    """A foreign, ambiguous or malformed material reference cannot be trusted."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _path(value):
    return value.split("#", 1)[-1]


def _positions(text, fragment):
    if not fragment:
        return []
    positions, offset = [], text.find(fragment)
    while offset >= 0:
        positions.append(offset)
        offset = text.find(fragment, offset + 1)
    return positions


class DocumentMaterialIndex:
    """Immutable material aliases bound to one DOCX and canonical manifest.

    ``source_path`` should be the actual input DOCX on the execution host, not a
    path copied from another host. A missing source is an explicit failure.
    Public dictionaries are copies; the full provenance artifact is separate
    from compact model views.
    """

    schema = "cityagent-document-material-index-v1"

    def __init__(self, manifest, materials, *, source_path=None,
                 expected_document_sha256=None):
        self._manifest = deepcopy(manifest)
        self._materials = deepcopy(materials)
        self.document_sha256 = manifest.get("document_sha256")
        if (not isinstance(self.document_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", self.document_sha256)
                or materials.get("docx_sha256") != self.document_sha256
                or expected_document_sha256 not in (None, self.document_sha256)):
            raise MaterialReferenceError("Material document SHA differs from canonical manifest")
        self.manifest_sha256 = _digest(manifest)
        self.materials_sha256 = _digest(materials)
        self.identity = {"document_sha256": self.document_sha256,
            "manifest_sha256": self.manifest_sha256, "materials_sha256": self.materials_sha256}
        if source_path is None:
            source_path = next((p for p in (materials.get("source"), manifest.get("source_path"))
                if isinstance(p, str) and Path(p).is_file()), None)
        if source_path is None or not Path(source_path).is_file():
            raise MaterialReferenceError("Original DOCX is required for XML-path validation; pass source_path")
        source = Path(source_path)
        if hashlib.sha256(source.read_bytes()).hexdigest() != self.document_sha256:
            raise MaterialReferenceError("Original DOCX SHA differs from material document")
        # Verify canonical text/path authority against the actual bytes. Path relocation
        # is allowed; evidence identities and text are not allowed to change.
        rebuilt = build_manifest(source)
        if rebuilt["units"] != manifest["units"] or rebuilt["tables"] != manifest["tables"]:
            raise MaterialReferenceError("Canonical units/tables differ from original DOCX XML")
        with zipfile.ZipFile(source) as archive:
            tree = etree.fromstring(archive.read("word/document.xml")).getroottree()
        self._nodes = {tree.getpath(node): node for node in tree.getroot().iter()}
        self._xml_tree = tree
        self._source_verified = {"document_sha256": self.document_sha256,
            "canonical_units_verified": True, "canonical_tables_verified": True,
            "method": "original_docx_sha256_and_rebuilt_canonical_units_tables"}
        self._units = {u["id"]: u for u in manifest["units"]}
        self._unit_order = {u["id"]: i for i, u in enumerate(manifest["units"])}
        self._units_by_path, self._cells_by_path = defaultdict(list), defaultdict(list)
        self._tables_by_path = { _path(t["source_path"]): t for t in manifest["tables"] }
        self._tables_by_id = {t["id"]: t for t in manifest["tables"]}
        for unit in manifest["units"]:
            self._units_by_path[_path(unit["source_path"])].append(unit)
            if unit["kind"] != "table_row":
                continue
            if unit["text"] != " | ".join(c["text"] for c in unit.get("cells", [])):
                raise MaterialReferenceError("Canonical table row/cell text differs")
            cursor = 0
            for cell in unit.get("cells", []):
                self._cells_by_path[_path(cell["source_path"])].append({"unit": unit,
                    "cell": cell, "start": cursor, "end": cursor + len(cell["text"])})
                cursor += len(cell["text"]) + 3
        self._source_map = materials.get("source_map", {})
        self._cell_material_refs = defaultdict(list)
        for packet in materials.get("packets", []):
            for segment in packet.get("segments", []):
                if segment.get("cell") and segment.get("kind") == "text":
                    if segment["ref"] not in self._cell_material_refs[segment["cell"]]:
                        self._cell_material_refs[segment["cell"]].append(segment["ref"])
        self._refs, self._sections, self._packets = {}, {}, {}
        self._build_text_refs()
        self._build_cell_table_refs()
        self._build_organization()
        self._aliases = defaultdict(list)
        for ref, record in self._refs.items():
            for eid in dict.fromkeys(s["evidence_id"] for s in record["spans"]):
                self._aliases[eid].append(self.qualify_ref(ref))

    def validate_manifest(self, manifest):
        if (manifest.get("document_sha256") != self.document_sha256
                or _digest(manifest) != self.manifest_sha256):
            raise MaterialReferenceError("Material index belongs to a different canonical manifest")
        return True

    def normalize_ref(self, ref):
        if not isinstance(ref, str) or not ref:
            raise MaterialReferenceError("Material reference must be a nonempty string")
        if ref.startswith("mineru:"):
            ref = ref[len("mineru:"):]
        if not ref.startswith("#/"):
            match = re.fullmatch(r"([0-9a-fA-F]{64})(#/.*)", ref)
            if not match:
                raise MaterialReferenceError("Malformed material namespace; use mineru:<document_sha256>#/...")
            if match[1].lower() != self.document_sha256:
                raise MaterialReferenceError("Foreign document SHA in material reference")
            ref = match[2]
        if not re.fullmatch(r"#/(?:texts/\d+|tables/\d+(?:/r\d+(?:/c\d+)?)?|sections/[A-Za-z0-9_-]+|packets/[A-Za-z0-9_-]+)", ref):
            raise MaterialReferenceError("Unknown material reference syntax")
        return ref

    def qualify_ref(self, ref):
        return f"mineru:{self.document_sha256}{self.normalize_ref(ref)}"

    def ref_exists(self, ref):
        # Identity errors are intentionally not converted to false.
        ref = self.normalize_ref(ref)
        return ref in self._refs or ref in self._sections or ref in self._packets

    def _span(self, evidence_id, start=0, end=None, **metadata):
        text = self._units[evidence_id]["text"]
        if end is None:
            end = len(text)
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(text):
            raise MaterialReferenceError("Material span outside canonical evidence")
        return {"evidence_id": evidence_id, "start": start, "end": end,
            "text": text[start:end], **metadata}

    def _ordered(self, spans):
        unique = {}
        for span in spans:
            key = (span["evidence_id"], span["start"], span["end"])
            if span["end"] > span["start"]:
                unique.setdefault(key, span)
        return sorted(unique.values(), key=lambda s: (self._unit_order[s["evidence_id"]], s["start"], s["end"]))

    def _record(self, ref, kind, spans, *, status="exact", unresolved=None, **metadata):
        self._refs[ref] = {"kind": kind, "status": status, "spans": self._ordered(spans),
            "metadata": metadata, "unresolved": unresolved or []}

    def _build_text_refs(self):
        grouped = defaultdict(list)
        for packet in self._materials.get("packets", []):
            if packet.get("docx_sha256", self.document_sha256) != self.document_sha256:
                raise MaterialReferenceError("Packet belongs to another document")
            for segment in packet.get("segments", []):
                if segment.get("kind") == "text":
                    grouped[segment["ref"]].append(segment)
        for ref, segments in grouped.items():
            info = self._source_map.get(ref, {})
            size = max(s["offset"][1] for s in segments)
            chars = [None] * size
            for s in segments:
                start, end = s["offset"]
                if (type(start) is not int or type(end) is not int or start < 0
                        or end - start != len(s["text"])):
                    raise MaterialReferenceError("Invalid material segment offset")
                for offset, char in enumerate(s["text"], start):
                    if chars[offset] not in (None, char):
                        raise MaterialReferenceError("Conflicting material segment text")
                    chars[offset] = char
            if any(c is None for c in chars):
                self._record(ref, "paragraph", [], status="unmapped",
                    unresolved=[{"code": "incomplete_segment_text", "ref": ref}])
                continue
            text = "".join(chars)
            if not text:
                self._record(ref, "paragraph", [], status="empty", source_provenance=deepcopy(info))
                continue
            candidates = []
            for xp in info.get("xml_paragraph_candidates", []):
                node = self._nodes.get(xp)
                if node is None:
                    continue
                original_parts = [v for k, v in _paragraph_events(node) if k == "text"]
                if not any(text in part for part in original_parts):
                    continue
                cell_path = info.get("xml_cell_path")
                if cell_path:
                    if not xp.startswith(cell_path + "/"):
                        continue
                    for cell in self._cells_by_path.get(cell_path, []):
                        for offset in _positions(cell["cell"]["text"], text):
                            candidates.append(self._span(cell["unit"]["id"], cell["start"] + offset,
                                cell["start"] + offset + len(text)))
                else:
                    for unit in self._units_by_path.get(xp, []):
                        for offset in _positions(unit["text"], text):
                            candidates.append(self._span(unit["id"], offset, offset + len(text)))
            candidates = self._ordered(candidates)
            status = "exact" if len(candidates) == 1 else "ambiguous" if candidates else "unmapped"
            diagnostics = [] if status == "exact" else [{"code": status + "_xml_text_mapping",
                "ref": ref, "candidate_spans": candidates}]
            self._record(ref, "paragraph", candidates if status == "exact" else [],
                status=status, unresolved=diagnostics, source_provenance=deepcopy(info),
                material_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                cell_ref=next((s.get("cell") for s in segments if s.get("cell")), None))

    def _build_cell_table_refs(self):
        for ref, info in self._source_map.items():
            if not re.fullmatch(r"#/tables/\d+/r\d+/c\d+", ref):
                continue
            xp = info.get("xml_cell_path")
            cells = self._cells_by_path.get(xp, [])
            valid = bool(cells) and xp in self._nodes
            actual_row_span = 1
            for c in cells:
                unit, cell = c["unit"], c["cell"]
                table = self._tables_by_id[unit["table_id"]]
                valid = valid and (_path(table["source_path"]) == info.get("xml_table_path")
                    and unit["row_index"] == info.get("row")
                    and cell["column_index"] == info.get("column")
                    and cell["column_span"] == info.get("col_span", 1))
                if cell.get("vertical_merge") == "restart":
                    # The canonical parser records actual continuation origins. A
                    # material-provided span cannot invent additional merged rows.
                    continuations = {other["row_index"] for other in self._manifest["units"]
                        if other.get("table_id") == unit["table_id"]
                        for continued in other.get("cells", [])
                        if continued.get("merged_from_source") == cell["source_path"]}
                    last_row = unit["row_index"]
                    while last_row + 1 in continuations:
                        last_row += 1
                    actual_row_span = max(actual_row_span, last_row - unit["row_index"] + 1)
            valid = valid and info.get("row_span", 1) == actual_row_span
            # Each nonempty MinerU text anchor must address this physical cell,
            # not merely contain a string that happens to occur elsewhere.
            anchor_issues = []
            for text_ref in self._cell_material_refs.get(ref, []):
                anchor = self._refs.get(text_ref, {})
                candidates = list(anchor.get("spans", []))
                candidates.extend(s for issue in anchor.get("unresolved", [])
                    for s in issue.get("candidate_spans", []))
                if anchor.get("status") == "empty":
                    continue
                if not any(s["evidence_id"] == c["unit"]["id"]
                        and c["start"] <= s["start"] <= s["end"] <= c["end"]
                        for s in candidates for c in cells):
                    anchor_issues.append({"code": "cell_text_anchor_mismatch", "ref": text_ref})
            valid = valid and not anchor_issues
            spans = [self._span(c["unit"]["id"], c["start"], c["end"]) for c in cells] if valid else []
            self._record(ref, "cell", spans, status="exact" if spans and any(s["text"] for s in spans) else "empty" if valid else "unmapped",
                unresolved=[] if valid else [{"code": "xml_cell_mapping_mismatch", "ref": ref,
                    "actual_row_span": actual_row_span}, *anchor_issues],
                source_provenance=deepcopy(info), table_ref=ref.split("/r", 1)[0], row=info.get("row"),
                column=info.get("column"), row_span=actual_row_span, col_span=info.get("col_span", 1),
                canonical_table_id=cells[0]["unit"]["table_id"] if valid else None,
                merged_from_source=[c["cell"].get("merged_from_source") for c in cells if c["cell"].get("merged_from_source")])
        self._table_aliases = {}
        for ref, info in self._source_map.items():
            if not re.fullmatch(r"#/tables/\d+", ref):
                continue
            table = self._tables_by_path.get(info.get("xml_table_path"))
            member_cells = [value for key, value in self._refs.items()
                if value["kind"] == "cell" and key.startswith(ref + "/")]
            consistent = all(c["metadata"]["source_provenance"].get("xml_table_path")
                == info.get("xml_table_path") for c in member_cells)
            if table is None or not consistent:
                self._record(ref, "table", [], status="unmapped",
                    unresolved=[{"code": "xml_table_not_found" if table is None else "xml_table_cell_identity_mismatch",
                        "ref": ref}], source_provenance=deepcopy(info))
                continue
            self._table_aliases[table["id"]] = ref
            row_units = [u for u in self._manifest["units"] if u.get("table_id") == table["id"]]
            # Table/row requests return original full rows (including their separators),
            # never imply that reformatted cell labels cover these canonical characters.
            self._record(ref, "table", [self._span(u["id"]) for u in row_units],
                source_provenance=deepcopy(info), canonical_table_id=table["id"], table_ref=ref,
                parent_table_id=table.get("parent_table_id"), header_kind=table["header_kind"])
            for row in sorted({u["row_index"] for u in row_units}):
                members = [u for u in row_units if u["row_index"] == row]
                self._record(f"{ref}/r{row}", "row", [self._span(u["id"]) for u in members],
                    canonical_table_id=table["id"], table_ref=ref, row=row,
                    source_provenance={"xml_table_path": info.get("xml_table_path"), "row": row},
                    header_kind=table["header_kind"])

    def _build_organization(self):
        for packet in self._materials.get("packets", []):
            ref = "#/packets/" + packet["packet_id"]
            refs = list(dict.fromkeys((s.get("cell") or s.get("ref")) for s in packet.get("segments", [])
                if (s.get("cell") or s.get("ref")) in self._refs))
            self._packets[ref] = {"id": ref, "source_id": packet["packet_id"],
                "section_id": "#/sections/" + packet["section_id"], "refs": refs,
                "headings": deepcopy(packet.get("headings", [])), "region": packet.get("region"),
                "table": packet.get("table")}
        for section in self._materials.get("sections", []):
            ref = "#/sections/" + section["id"]
            source = section.get("source") or {}
            heading_ref = source.get("ref")
            heading = self._refs.get(heading_ref, {})
            spans = deepcopy(heading.get("spans", []))
            indices = [self._unit_order[s["evidence_id"]] for s in spans]
            self._sections[ref] = {"id": ref, "source_id": section["id"],
                "parent": "#/sections/" + section["parent"] if section.get("parent") else None,
                "title": section.get("title", ""), "headings": deepcopy(section.get("headings", [])),
                "region": section.get("region"), "level": section.get("level", 0),
                "basis": section.get("basis"), "heading_ref": heading_ref, "heading_spans": spans,
                "unit_start": min(indices) if indices else None, "unit_end": max(indices) + 1 if indices else None,
                "packet_ids": ["#/packets/" + p for p in section.get("packet_ids", [])],
                "heading_unresolved": deepcopy(heading.get("unresolved", []))}

    @property
    def section_ids(self):
        return tuple(self._sections)

    @property
    def sections(self):
        return deepcopy(list(self._sections.values()))

    @property
    def packets(self):
        return deepcopy(list(self._packets.values()))

    def aliases_for(self, evidence_id):
        return list(self._aliases.get(evidence_id, []))

    def _context(self, record):
        metadata = record["metadata"]
        if record["kind"] == "paragraph" and metadata.get("cell_ref"):
            record = self._refs.get(metadata["cell_ref"], record)
            metadata = record["metadata"]
        table_id = metadata.get("canonical_table_id")
        if not table_id:
            return [], {}
        table = self._tables_by_id[table_id]
        rows = [u for u in self._manifest["units"] if u.get("table_id") == table_id]
        context = []
        row = metadata.get("row")
        if row is not None:
            context.extend(self._span(u["id"], role="same_row") for u in rows if u["row_index"] == row)
        for eid in table.get("header_evidence_ids", []):
            context.append(self._span(eid, role="header" if table["header_kind"] == "explicit" else "header_candidate"))
        # Continued merged cells are supported with the actual anchor's source path.
        for u in rows:
            if row is None or u["row_index"] != row:
                continue
            for cell in u.get("cells", []):
                origin = cell.get("merged_from_source")
                if origin:
                    for anchor in self._cells_by_path.get(_path(origin), []):
                        context.append(self._span(anchor["unit"]["id"], anchor["start"], anchor["end"], role="merged_cell_anchor"))
        parent_id = table.get("parent_table_id")
        parent_ref = self._table_aliases.get(parent_id)
        parent_cells = []
        if parent_id:
            table_path = _path(table["source_path"])
            for cell_path, anchors in self._cells_by_path.items():
                if table_path.startswith(cell_path + "/") and any(a["unit"].get("table_id") == parent_id for a in anchors):
                    parent_cells.append(cell_path)
                    context.extend(self._span(a["unit"]["id"], a["start"], a["end"], role="parent_cell") for a in anchors)
        return self._ordered(context), {"table_ref": self._table_aliases.get(table_id),
            "canonical_table_id": table_id, "header_kind": table["header_kind"],
            "header_note": "候选表头仅作原文上下文，不是模型语义结论。",
            "parent_table_ref": parent_ref, "parent_cell_xml_paths": parent_cells}

    def resolve(self, ref, *, include_context=True):
        ref = self.normalize_ref(ref)
        if ref in self._sections:
            return self.section(ref, include_context=include_context)
        if ref in self._packets:
            return self.packet(ref, include_context=include_context)
        if ref not in self._refs:
            raise MaterialReferenceError("Material reference does not exist in this document")
        record = self._refs[ref]
        context, extra = self._context(record) if include_context else ([], {})
        candidates = [s for issue in record["unresolved"] for s in issue.get("candidate_spans", [])]
        return {"ok": record["status"] in ("exact", "empty"), "ref": ref,
            "qualified_ref": self.qualify_ref(ref), **self.identity, "status": record["status"],
            "spans": deepcopy(record["spans"]), "context": deepcopy(context),
            "metadata": {"kind": record["kind"], **deepcopy(record["metadata"]), **extra},
            "unresolved": deepcopy(record["unresolved"]),
            "fallback": {"mode": "canonical", "evidence_ids": list(dict.fromkeys(s["evidence_id"] for s in candidates))},
            "semantic_review": False}

    resolve_ref = resolve
    resolve_source_ref = resolve

    def section_ref_ids(self, section_id, *, include_descendants=True):
        if isinstance(section_id, str) and section_id.startswith("S"):
            section_id = "#/sections/" + section_id
        section_id = self.normalize_ref(section_id)
        if section_id not in self._sections:
            raise MaterialReferenceError("Material section does not exist")
        selected = {section_id}
        if include_descendants:
            changed = True
            while changed:
                before = len(selected)
                selected.update(s["id"] for s in self._sections.values() if s["parent"] in selected)
                changed = before != len(selected)
        return list(dict.fromkeys(ref for p in self._packets.values() if p["section_id"] in selected for ref in p["refs"]))

    def _combine(self, ref, refs, metadata, include_context):
        resolved = [self.resolve(r, include_context=include_context) for r in refs]
        spans = self._ordered([s for r in resolved for s in r["spans"]])
        unresolved = [u for r in resolved for u in r["unresolved"]]
        return {"ok": not unresolved, "ref": ref, "qualified_ref": self.qualify_ref(ref),
            **self.identity, "status": "partial" if unresolved else "exact",
            "spans": spans, "context": self._ordered([s for r in resolved for s in r["context"]]),
            "metadata": deepcopy(metadata), "unresolved": unresolved,
            "fallback": {"mode": "canonical", "evidence_ids": list(dict.fromkeys(eid for r in resolved for eid in r["fallback"]["evidence_ids"]))},
            "semantic_review": False}

    def section(self, section_id, *, include_descendants=True, include_context=True):
        if isinstance(section_id, str) and section_id.startswith("S"):
            section_id = "#/sections/" + section_id
        section_id = self.normalize_ref(section_id)
        refs = self.section_ref_ids(section_id, include_descendants=include_descendants)
        return self._combine(section_id, refs, self._sections[section_id], include_context)

    def packet(self, packet_id, *, include_context=True):
        if isinstance(packet_id, str) and packet_id.startswith("P"):
            packet_id = "#/packets/" + packet_id
        packet_id = self.normalize_ref(packet_id)
        if packet_id not in self._packets:
            raise MaterialReferenceError("Material packet does not exist")
        packet = self._packets[packet_id]
        return self._combine(packet_id, packet["refs"], packet, include_context)

    def compact_outline(self):
        return {**self.identity, "sections": self.sections, "packet_count": len(self._packets),
            "unresolved_ref_count": sum(bool(r["unresolved"]) for r in self._refs.values()),
            "semantic_review": False}

    def search(self, query, *, section_id=None, limit=20):
        if not isinstance(query, str) or not query or type(limit) is not int or not 1 <= limit <= 100:
            raise MaterialReferenceError("Search needs a nonempty query and limit 1..100")
        allowed = set(self.section_ref_ids(section_id)) if section_id else None
        matches = []
        for ref, record in self._refs.items():
            if allowed is not None and ref not in allowed:
                continue
            if any(query.casefold() in s["text"].casefold() for s in record["spans"]):
                matches.append({"ref": self.qualify_ref(ref), "kind": record["kind"],
                    "evidence_ids": list(dict.fromkeys(s["evidence_id"] for s in record["spans"]))})
            if len(matches) >= limit:
                break
        return {**self.identity, "matches": matches, "semantic_review": False}

    def to_manifest(self):
        return {"schema": self.schema, **self.identity, "source_validation": deepcopy(self._source_verified),
            "refs": deepcopy(self._refs), "sections": self.sections, "packets": self.packets,
            "source_map": deepcopy(self._source_map),
            "policy": {"ordinal_matching": False, "fuzzy_matching": False,
                "ambiguity_auto_selected": False, "coverage_granted": False,
                "model_source_payload": "canonical_evidence_exact_spans"}}

    snapshot = to_manifest

    def save(self, path):
        atomic_json(Path(path), self.to_manifest())
        return Path(path)
