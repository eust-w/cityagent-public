"""Lossless text indexing and bounded reads for the document agent.

Units are in OOXML document order, including tables embedded in cells.  They
are evidence, not predictions: merged values and repeated headers are context
only.  Image occurrences are inventoried but never claimed as interpreted.
Character intervals in chunks refer to the original, unchanged unit text.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import qn

PARSER_VERSION = "evidence-manifest-v2"
_P, _TABLE, _ROW, _CELL = (qn(f"w:{name}") for name in ("p", "tbl", "tr", "tc"))
_TEXT = {qn("w:t"), qn("w:delText"), "{http://schemas.openxmlformats.org/officeDocument/2006/math}t"}
_UNVERIFIED = {
    qn("w:altChunk"), qn("w:object"), qn("w:subDoc"), qn("w:sym"),
    "{http://schemas.openxmlformats.org/drawingml/2006/chart}chart",
    "{http://schemas.openxmlformats.org/drawingml/2006/diagram}relIds",
    "{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath",
    "{urn:schemas-microsoft-com:office:office}OLEObject",
}


def _new_manifest(source_sha256: str, source_path: str, source_kind: str) -> dict[str, Any]:
    return {
        "document_sha256": source_sha256,
        "parser_version": PARSER_VERSION,
        "source_path": source_path,
        "source_kind": source_kind,
        "units": [],
        "tables": [],
        "images": [],
        "unverified_content": [],
        "limitations": [],
    }


def _own_descendants(element: Any, tag: str, stop: set[str]) -> Iterable[Any]:
    for child in element:
        if child.tag == tag:
            yield child
        elif child.tag not in stop:
            yield from _own_descendants(child, tag, stop)


def _paragraph_events(element: Any) -> list[tuple[str, Any]]:
    """Keep text and text-box tables in their original interleaved order."""
    parts: list[str] = []
    events: list[tuple[str, Any]] = []

    def flush() -> None:
        if parts:
            events.append(("text", "".join(parts)))
            parts.clear()

    def visit(node: Any) -> None:
        if node.tag == _TABLE:
            flush()
            events.append(("table", node))
            return
        if node.tag in _TEXT:
            parts.append(node.text or "")
        elif node.tag in {qn("w:tab"), qn("w:ptab")}:
            parts.append("\t")
        elif node.tag in {qn("w:br"), qn("w:cr")}:
            parts.append("\n")
        elif node.tag == qn("w:noBreakHyphen"):
            parts.append("\u2011")
        elif node.tag == qn("w:softHyphen"):
            parts.append("\u00ad")
        else:
            for child in node:
                visit(child)

    visit(element)
    flush()
    return events or [("text", "")]


def _int_property(properties: Any, name: str, default: int) -> int:
    node = properties.find(qn(f"w:{name}")) if properties is not None else None
    if node is None:
        return default
    try:
        return int(node.get(qn("w:val"), str(default)))
    except ValueError:
        raise ValueError(f"Invalid OOXML {name} value") from None


class _Builder:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.recent: list[str] = []
        self.heading = ""
        self.heading_id: str | None = None
        self.story = "body"
        self.part_name = "word/document.xml"

    def source(self, element: Any) -> str:
        return f"{self.part_name}#{element.getroottree().getpath(element)}"

    def emit(self, text: str, kind: str, element: Any, **extra: Any) -> dict[str, Any]:
        unit_id = f"E{len(self.manifest['units']) + 1:06d}"
        unit = {
            "id": unit_id,
            "coverage_id": unit_id,
            "kind": kind,
            "text": text,
            "source_path": self.source(element),
            "context": {
                "story": self.story,
                "section_heading": self.heading,
                "section_heading_id": self.heading_id,
                "preceding_evidence_ids": list(self.recent[-3:]),
                "parent_table_id": extra.pop("parent_table_id", None),
            },
            **extra,
        }
        self.manifest["units"].append(unit)
        if text:
            self.recent.append(unit_id)
        return unit

    def blocks(self, root: Any) -> None:
        for child in root:
            if child.tag == _TABLE:
                self.table(child)
            elif child.tag == _P:
                events = _paragraph_events(child)
                properties = child.find(qn("w:pPr"))
                style = properties.find(qn("w:pStyle")) if properties is not None else None
                style_id = style.get(qn("w:val"), "") if style is not None else ""
                is_heading = bool(re.match(r"(?i)heading\d+$", style_id))
                if is_heading:
                    self.heading = "".join(value for kind, value in events if kind == "text")
                for segment_index, (kind, value) in enumerate(events):
                    if kind == "text":
                        unit = self.emit(value, "paragraph", child, style_id=style_id, segment_index=segment_index)
                        if is_heading:
                            self.heading_id = unit["id"]
                    else:
                        self.table(value)
            else:
                # Content controls and revision wrappers do not erase blocks.
                self.blocks(child)

    def cell_events(self, cell: Any) -> Iterable[tuple[str, Any]]:
        for child in cell:
            if child.tag == _TABLE:
                yield "table", child
            elif child.tag == _P:
                yield from _paragraph_events(child)
            else:
                yield from self.cell_events(child)

    def table(self, element: Any, parent_table_id: str | None = None) -> None:
        table_id = f"T{len(self.manifest['tables']) + 1:05d}"
        rows = list(_own_descendants(element, _ROW, {_TABLE}))
        info = {
            "id": table_id,
            "source_path": self.source(element),
            "parent_table_id": parent_table_id,
            "row_count": len(rows),
            "header_evidence_ids": [],
            "header_kind": "none",
            "context": {"story": self.story, "section_heading": self.heading},
        }
        self.manifest["tables"].append(info)
        vertical: dict[int, tuple[str, str]] = {}
        first_ids: list[str] = []
        first_has_nested = False
        for row_index, row in enumerate(rows):
            properties = row.find(qn("w:trPr"))
            header_node = properties.find(qn("w:tblHeader")) if properties is not None else None
            is_header = header_node is not None and header_node.get(qn("w:val"), "true") not in {"0", "false", "off"}
            column = _int_property(properties, "gridBefore", 0)
            segment_index = 0
            pending: list[dict[str, Any]] = []
            row_ids: list[str] = []

            def flush(force: bool = False) -> None:
                nonlocal segment_index, pending
                if not pending and not force:
                    return
                unit = self.emit(
                    " | ".join(cell["text"] for cell in pending),
                    "table_row", row, table_id=table_id, row_index=row_index,
                    segment_index=segment_index, cells=pending, is_header=is_header,
                    parent_table_id=parent_table_id,
                )
                row_ids.append(unit["id"])
                pending = []
                segment_index += 1

            for cell in _own_descendants(row, _CELL, {_TABLE, _ROW}):
                cell_properties = cell.find(qn("w:tcPr"))
                span = max(1, _int_property(cell_properties, "gridSpan", 1))
                merge_node = cell_properties.find(qn("w:vMerge")) if cell_properties is not None else None
                merge = (merge_node.get(qn("w:val")) or "continue") if merge_node is not None else None
                inherited = vertical.get(column) if merge == "continue" else None
                metadata = {
                    "column_index": column,
                    "column_span": span,
                    "vertical_merge": merge,
                    "merged_from_source": inherited[1] if inherited else None,
                    "merged_context_text": inherited[0] if inherited else "",
                    "source_path": self.source(cell),
                }
                texts: list[str] = []
                all_texts: list[str] = []
                had_event = False
                for kind, item in self.cell_events(cell):
                    had_event = True
                    if kind == "text":
                        text = item
                        texts.append(text)
                        all_texts.append(text)
                    else:
                        # Flush before the nested table, then resume the same
                        # physical row afterwards.  Do not move following text
                        # ahead of the embedded table or duplicate nested text.
                        if texts:
                            pending.append({**metadata, "text": "\n".join(texts)})
                            texts = []
                        flush()
                        self.table(item, parent_table_id=table_id)
                        if row_index == 0:
                            first_has_nested = True
                if texts or not had_event:
                    pending.append({**metadata, "text": "\n".join(texts)})
                value = "\n".join(all_texts)
                for offset in range(span):
                    if merge == "restart":
                        vertical[column + offset] = (value, self.source(cell))
                    elif merge != "continue":
                        vertical.pop(column + offset, None)
                column += span
            flush(force=not row_ids)
            if row_index == 0:
                first_ids = row_ids
            if is_header:
                info["header_evidence_ids"].extend(row_ids)
                info["header_kind"] = "explicit"
        # A one-cell layout row wrapping the report is not a column header.
        if not info["header_evidence_ids"] and not first_has_nested:
            info["header_evidence_ids"] = first_ids
            info["header_kind"] = "first_row_candidate"


_STORY_KINDS = ("header", "footer", "footnotes", "endnotes", "comments")
_STORY_CONTENT_TYPES = {
    f"application/vnd.openxmlformats-officedocument.wordprocessingml.{name}+xml": name
    for name in _STORY_KINDS
}
_STORY_RELATIONSHIPS = {
    f"{prefix}/{name}": name
    for prefix in (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "http://purl.oclc.org/ooxml/officeDocument/relationships",
    )
    for name in _STORY_KINDS
}


def _supplementary_stories(path: Path, document: Any) -> list[tuple[str, Any, Any]]:
    """Discover OPC stories by identity rather than conventional part names.

    Preserve the existing lexical part order and reject known story parts that
    python-docx did not load. An orphan story must not disappear into a claim of
    complete document coverage; interpreting orphan package structure requires
    an explicit repair outside this text parser.
    """
    parts = {str(part.partname).lstrip("/"): part for part in document.part.package.parts}
    kinds = {}
    for name, part in parts.items():
        kind = _STORY_CONTENT_TYPES.get(part.content_type)
        if kind:
            kinds[name] = kind
        elif re.fullmatch(r"word/(header\d+|footer\d+|footnotes|endnotes|comments)\.xml", name):
            kinds[name] = re.sub(r"\d+$", "", Path(name).stem)
    for part in parts.values():
        for relation in part.rels.values():
            kind = _STORY_RELATIONSHIPS.get(relation.reltype)
            if kind is None:
                continue
            if relation.is_external:
                raise ValueError("External OOXML story relationship is not fetched")
            name = str(relation.target_part.partname).lstrip("/")
            if name not in parts:
                raise ValueError("Referenced OOXML story part missing")
            if name in kinds and kinds[name] != kind:
                raise ValueError("Conflicting OOXML story identity")
            kinds[name] = kind
    # OPC permits arbitrary part names. ContentTypes can also declare a story
    # which has no traversable package relationship; fail closed for that case.
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("[Content_Types].xml")
        if info.file_size > 32 * 1024**2:
            raise ValueError("OOXML content type metadata limit exceeded")
        with archive.open(info) as handle:
            metadata = handle.read(32 * 1024**2 + 1)
        if len(metadata) != info.file_size:
            raise ValueError("OOXML content type metadata length mismatch")
        types = parse_xml(metadata)
        if types.getroottree().docinfo.doctype:
            raise ValueError("OOXML content type DTD is not accepted")
        defaults = {}
        declared = {}
        for item in types:
            if not isinstance(item.tag, str):
                continue
            kind = _STORY_CONTENT_TYPES.get(item.get("ContentType"))
            if kind is None:
                continue
            if item.tag.endswith("}Override"):
                declared[item.get("PartName", "").lstrip("/")] = kind
            elif item.tag.endswith("}Default"):
                defaults[item.get("Extension", "").lower()] = kind
        for name in archive.namelist():
            if Path(name).suffix[1:].lower() in defaults and name not in declared:
                # An explicit override supersedes the default content type.
                override = next((item for item in types if isinstance(item.tag, str) and item.get("PartName", "").lstrip("/") == name), None)
                if override is None:
                    declared[name] = defaults[Path(name).suffix[1:].lower()]
        for name, kind in declared.items():
            if name not in parts:
                raise ValueError("Declared OOXML story is missing or unreachable from package relationships")
            if name in kinds and kinds[name] != kind:
                raise ValueError("Conflicting declared OOXML story identity")
            kinds[name] = kind
    stories = []
    for name in sorted(kinds):
        kind, part = kinds[name], parts[name]
        root = parse_xml(part.blob)
        expected = {"header": "hdr", "footer": "ftr"}.get(kind, kind)
        if root.tag != qn(f"w:{expected}"):
            raise ValueError("Unexpected OOXML story root")
        stories.append((kind, root, part))
    return stories


def build_manifest(path: Path | str) -> dict[str, Any]:
    """Index body, header/footer, notes and comments without content filtering.

    DOC must be converted beforehand.  Source paths point into original OOXML;
    body order is retained, supplementary stories follow by package part name.
    """
    path = Path(path)
    if path.suffix.lower() != ".docx":
        raise ValueError("Evidence indexing requires DOCX; convert legacy DOC first")
    manifest = _new_manifest(hashlib.sha256(path.read_bytes()).hexdigest(), str(path), "docx")
    document = Document(str(path))
    builder = _Builder(manifest)
    # Retain legacy IDs only for simple direct body blocks. Every wrapped,
    # nested or split physical occurrence retains its unique manifest ID.
    legacy_paths: dict[str, str] = {}
    table_number = 0
    for block_number, block in enumerate(document.element.body):
        source_path = f"word/document.xml#{block.getroottree().getpath(block)}"
        if block.tag == _P:
            legacy_paths[source_path] = f"p{block_number}"
        elif block.tag == _TABLE:
            for row_number, row in enumerate(block.findall(_ROW)):
                legacy_paths[f"word/document.xml#{row.getroottree().getpath(row)}"] = f"t{table_number}r{row_number}"
            table_number += 1
    stories = [("body", document.element.body, document.part)]
    stories.extend(_supplementary_stories(path, document))
    for story, root, part in stories:
        builder.story = story
        builder.part_name = str(part.partname).lstrip("/")
        builder.recent = []
        builder.heading = ""
        builder.heading_id = None
        builder.blocks(root)
        for element in root.iter():
            if element.tag in _UNVERIFIED:
                manifest["unverified_content"].append({
                    "source_path": builder.source(element),
                    "kind": element.tag.rsplit("}", 1)[-1],
                    "verification": "unverified",
                    "reason": "Embedded/external object, graphic, equation or font symbol needs separate interpretation",
                })
            if element.tag not in {qn("a:blip"), "{urn:schemas-microsoft-com:vml}imagedata"}:
                continue
            relation_id = next((element.get(qn(f"r:{key}")) for key in ("embed", "id", "link") if element.get(qn(f"r:{key}"))), "")
            relation = part.rels.get(relation_id)
            target = str(relation.target_ref) if relation is not None else ""
            manifest["images"].append({
                "id": f"I{len(manifest['images']) + 1:05d}",
                "source_path": builder.source(element),
                "story": story,
                "relationship_id": relation_id,
                "target": target,
                "verification": "unverified",
                "reason": "Image pixels have not been interpreted by the text evidence parser",
            })
    counts: dict[str, int] = {}
    for unit in manifest["units"]:
        counts[unit["source_path"]] = counts.get(unit["source_path"], 0) + 1
    for unit in manifest["units"]:
        source_path = unit["source_path"]
        if source_path in legacy_paths and counts[source_path] == 1:
            unit["legacy_source_id"] = legacy_paths[source_path]
    if manifest["images"]:
        manifest["limitations"].append("Images are inventoried only; text coverage does not establish visual evidence coverage")
    if manifest["unverified_content"]:
        manifest["limitations"].append("Unverified embedded/external content exists; complete text-unit coverage is not complete document coverage")
    manifest["limitations"].append("Evidence order is OOXML order within each story, not rendered page layout")
    return manifest


def manifest_from_text(text: str, source_sha256: str | None = None) -> dict[str, Any]:
    """Bind an already extracted training text, preserving every character.

    This is diagnostic input, not proof that the source Word's images or table
    layout were parsed.  SHA256 supplied by a caller identifies the original
    source; text_sha256 always records the exact UTF-8 evidence text.
    """
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest = _new_manifest(source_sha256 or text_sha256, "training-text", "training_text")
    manifest["text_sha256"] = text_sha256
    manifest["limitations"] = ["Pre-extracted training text; original Word structure and images are not verified"]
    offset = 0
    for index, line in enumerate(text.splitlines(keepends=True) or [""]):
        unit_id = f"E{index + 1:06d}"
        manifest["units"].append({
            "id": unit_id,
            "coverage_id": unit_id,
            "kind": "text_line",
            "text": line,
            "source_path": f"training-text#chars={offset}:{offset + len(line)}",
            "source_start": offset,
            "source_end": offset + len(line),
            "context": {"story": "training_text", "section_heading": "", "preceding_evidence_ids": [f"E{index:06d}"] if index else [], "parent_table_id": None},
        })
        offset += len(line)
    return manifest


def read_evidence(manifest: dict[str, Any], ids: Iterable[str] | str) -> list[dict[str, Any]]:
    """Return requested units in requested order; unknown IDs fail explicitly."""
    if isinstance(ids, str):
        ids = [ids]
    index = {unit["id"]: unit for unit in manifest["units"]}
    result = []
    for evidence_id in ids:
        if evidence_id not in index:
            raise KeyError(f"Unknown evidence ID: {evidence_id}")
        result.append(index[evidence_id])
    return result


def read_table_rows(manifest: dict[str, Any], table_id: str, start: int, end: int) -> list[dict[str, Any]]:
    """Read zero-based physical rows in [start, end), including row fragments."""
    table = next((item for item in manifest["tables"] if item["id"] == table_id), None)
    if table is None:
        raise KeyError(f"Unknown table ID: {table_id}")
    if not (0 <= start <= end <= table["row_count"]):
        raise ValueError(f"Row range must satisfy 0 <= start <= end <= {table['row_count']}")
    return [unit for unit in manifest["units"] if unit.get("table_id") == table_id and start <= unit["row_index"] < end]


def search_evidence(manifest: dict[str, Any], query: str, limit: int = 5) -> list[dict[str, Any]]:
    if not query or not query.strip():
        raise ValueError("Search query must not be blank")
    if limit < 1:
        raise ValueError("Search limit must be positive")
    query = query.casefold()
    return [{**unit, "matched_text": unit["text"]} for unit in manifest["units"] if query in unit["text"].casefold()][:limit]


def get_neighbors(manifest: dict[str, Any], evidence_id: str, radius: int = 1) -> list[dict[str, Any]]:
    if radius < 0:
        raise ValueError("Neighbor radius must be non-negative")
    units = manifest["units"]
    position = next((index for index, unit in enumerate(units) if unit["id"] == evidence_id), None)
    if position is None:
        raise KeyError(f"Unknown evidence ID: {evidence_id}")
    story = units[position]["context"]["story"]
    return [unit for unit in units[max(0, position - radius):position + radius + 1] if unit["context"]["story"] == story]


def outline(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_sha256": manifest["document_sha256"],
        "parser_version": manifest["parser_version"],
        "source_kind": manifest["source_kind"],
        "unit_count": len(manifest["units"]),
        "character_count": sum(len(unit["text"]) for unit in manifest["units"]),
        "tables": manifest["tables"],
        "headings": [{"id": unit["id"], "text": unit["text"]} for unit in manifest["units"] if unit.get("style_id", "").lower().startswith("heading")],
        "unverified_image_count": len(manifest["images"]),
        "unverified_content_count": len(manifest.get("unverified_content", [])),
        "limitations": manifest["limitations"],
    }


def _render_index(manifest: dict[str, Any]) -> tuple[dict, dict, list]:
    return (
        {unit["id"]: unit for unit in manifest["units"]},
        {table["id"]: table for table in manifest["tables"]},
        [unit for unit in manifest["units"] if unit["text"].strip()][:3],
    )


def render_spans(manifest: dict[str, Any], spans: list[dict[str, Any]], *, _index=None) -> dict[str, Any]:
    """Render exact primary intervals plus bounded, separately identified context.

    Document-opening clues, section headings and recent narrative paragraphs
    share a 480-character context budget (160 characters per source excerpt).
    Each excerpt states its character interval.  Table headers and merged-cell
    values remain complete.  Context never grants primary evidence coverage.
    """
    unit_index, table_index, opening_units = _index if _index is not None else _render_index(manifest)
    if not spans:
        return {"text": "", "context_evidence_ids": []}
    primary_starts: dict[str, int] = {}
    for span in spans:
        evidence_id = span["coverage_id"]
        if evidence_id not in unit_index:
            raise KeyError(f"Unknown evidence ID: {evidence_id}")
        start, end = span["start"], span["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(unit_index[evidence_id]["text"]):
            raise ValueError(f"Invalid character interval for {evidence_id}: {start}:{end}")
        primary_starts[evidence_id] = min(start, primary_starts.get(evidence_id, start))
    header_ids = {
        header_id
        for span in spans
        for header_id in table_index.get(unit_index[span["coverage_id"]].get("table_id"), {}).get("header_evidence_ids", [])
    }
    parts: list[str] = []
    context_ids: list[str] = []
    context_remaining = 480

    def add_context(evidence_id: str | None, label: str) -> None:
        nonlocal context_remaining
        if evidence_id not in unit_index or evidence_id in context_ids or evidence_id in header_ids or context_remaining <= 0:
            return
        # A primary interval starting at zero already contains these opening
        # clues.  Later fragments can explicitly repeat a bounded prefix.
        if evidence_id in primary_starts and primary_starts[evidence_id] < 160:
            return
        text = unit_index[evidence_id]["text"]
        size = min(len(text), 160, context_remaining)
        if not text[:size].strip():
            return
        parts.append(f"[Context {label}; {evidence_id} chars 0:{size}; not primary coverage]\n{text[:size]}")
        context_ids.append(evidence_id)
        context_remaining -= size

    # Prefer explicit section identity over nearby arbitrary table rows.
    for span in spans:
        unit = unit_index[span["coverage_id"]]
        add_context(unit.get("context", {}).get("section_heading_id"), "section heading")
    for unit in opening_units:
        add_context(unit["id"], "document opening")
    for span in spans:
        unit = unit_index[span["coverage_id"]]
        for evidence_id in reversed(unit.get("context", {}).get("preceding_evidence_ids", [])):
            previous = unit_index.get(evidence_id, {})
            if previous.get("kind") in {"paragraph", "text_line"} and len(previous.get("text", "")) <= 160:
                add_context(evidence_id, "preceding paragraph")

    seen_tables: set[str] = set()
    for span in spans:
        unit = unit_index[span["coverage_id"]]
        table_id = unit.get("table_id")
        if table_id and table_id not in seen_tables:
            seen_tables.add(table_id)
            table = table_index[table_id]
            if table["header_evidence_ids"]:
                parts.append(f"[Table {table_id}; header context ({table['header_kind']})]")
                for header_id in table["header_evidence_ids"]:
                    parts.append(f"[{header_id}] {unit_index[header_id]['text']}")
                    context_ids.append(header_id)
        merged = [cell["merged_context_text"] for cell in unit.get("cells", []) if cell.get("merged_context_text")]
        if merged:
            parts.append("[Merged-cell context] " + " | ".join(merged))
        start, end = span["start"], span["end"]
        parts.append(f"[{unit['id']} chars {start}:{end}]\n{unit['text'][start:end]}")
    return {"text": "\n\n".join(parts), "context_evidence_ids": list(dict.fromkeys(context_ids))}


def make_chunks(manifest: dict[str, Any], count_tokens: Callable[[str], int], max_tokens: int) -> list[dict[str, Any]]:
    """Greedy evidence chunks with complete, auditable character coverage.

    The token bound includes evidence IDs, interval markers and repeated table
    headers.  Callers must reserve separate space for prompts and output.
    Oversize headers or an indivisible character raise rather than truncate.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    chunks: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    render_index = _render_index(manifest)
    # Bounded exact-token cache: reuse the same rendered interval checked by
    # fits/flush, without estimating token length or retaining whole reports.
    cache: dict[tuple, tuple[dict, int]] = {}

    def measured(spans):
        key = tuple((span["coverage_id"], span["start"], span["end"]) for span in spans)
        if key not in cache:
            rendered = render_spans(manifest, spans, _index=render_index)
            if len(cache) >= 64:
                cache.pop(next(iter(cache)))
            cache[key] = (rendered, count_tokens(rendered["text"]))
        return cache[key]

    def fits(spans: list[dict[str, Any]]) -> bool:
        return measured(spans)[1] <= max_tokens

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        rendered, token_count = measured(pending)
        text, context_ids = rendered["text"], rendered["context_evidence_ids"]
        chunks.append({
            "id": f"C{len(chunks) + 1:05d}",
            "evidence_ids": list(dict.fromkeys(span["coverage_id"] for span in pending)),
            "spans": pending,
            "context_evidence_ids": context_ids,
            "text": text,
            "token_count": token_count,
        })
        pending = []

    for unit in manifest["units"]:
        unit_id, length = unit["id"], len(unit["text"])
        full = {"coverage_id": unit_id, "start": 0, "end": length}
        if fits([*pending, full]):
            pending.append(full)
            continue
        flush()
        if fits([full]):
            pending.append(full)
            continue
        start = 0
        if not length:
            raise ValueError(f"Evidence metadata exceeds max_tokens for {unit_id}")
        while start < length:
            # Token lengths need not be monotonic over characters; bisection
            # finds a valid bound, and every emitted interval is recounted.
            low, high, best = start + 1, length, None
            while low <= high:
                end = (low + high) // 2
                span = {"coverage_id": unit_id, "start": start, "end": end}
                if fits([span]):
                    best = span
                    low = end + 1
                else:
                    high = end - 1
            if best is None:
                raise ValueError(f"Evidence header/context or one character exceeds max_tokens for {unit_id}; raise the budget or read the header separately")
            pending = [best]
            flush()
            start = best["end"]
    flush()
    return chunks
