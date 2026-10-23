"""Unintegrated, read-only DOCX native-object inventory. No model or network I/O.

Input is an immutable evidence-manifest-v1/v2 from build_manifest. Outputs preserve
every unverified_content entry; adjacent context is not a semantic sufficiency
verdict. Original XML, native binary and cached image have separate byte hashes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import posixpath
import re
from urllib.parse import unquote, urlsplit
import xml.parsers.expat
import zipfile

from lxml import etree

from bridge_evidence.contest.evidence_manifest import (
    _Builder, _new_manifest, _STORY_CONTENT_TYPES, _STORY_RELATIONSHIPS, _UNVERIFIED,
)

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
O = "urn:schemas-microsoft-com:office:office"
V = "urn:schemas-microsoft-com:vml"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
SUPPORTED_MANIFESTS = {"evidence-manifest-v1", "evidence-manifest-v2"}
STORY = re.compile(r"word/(document|header\d+|footer\d+|footnotes|endnotes|comments)\.xml$")


class CatalogError(ValueError):
    """Safe diagnostic code; never an underlying parser/ZIP exception string."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class CatalogLimits:
    package_bytes: int = 64 * 1024 * 1024
    total_uncompressed_bytes: int = 256 * 1024 * 1024
    member_bytes: int = 32 * 1024 * 1024
    xml_bytes: int = 8 * 1024 * 1024
    members: int = 4096
    compression_ratio: int = 500


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode())


def occurrence_id(occurrence: dict) -> str:
    """Same compact JSON identity as SourceObjectDispositions.occurrence_id."""
    return "so_" + _digest(occurrence)[:32]


def _member_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or name.startswith("/") or ":" in name
            or any(x in ("", ".", "..") for x in name.rstrip("/").split("/"))):
        raise CatalogError("UNSAFE_ZIP_MEMBER")
    return name


class _Package:
    def __init__(self, blob, limits):
        try:
            self.archive = zipfile.ZipFile(io.BytesIO(blob))
        except (zipfile.BadZipFile, OSError):
            raise CatalogError("INVALID_ZIP") from None
        infos = self.archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise CatalogError("DUPLICATE_ZIP_MEMBER")
        if len(infos) > limits.members or sum(x.file_size for x in infos) > limits.total_uncompressed_bytes:
            raise CatalogError("PACKAGE_LIMIT")
        for info in infos:
            _member_name(info.filename)
            if (info.flag_bits & 1 or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise CatalogError("UNSAFE_ZIP_MEMBER")
            if (info.file_size > limits.member_bytes
                    or info.file_size > max(1, info.compress_size) * limits.compression_ratio):
                raise CatalogError("MEMBER_LIMIT")
        self.names = set(names)
        self.limits = limits
        self.parts = {}
        self.rels = {}

    def read(self, name):
        if name not in self.names:
            raise CatalogError("MISSING_MEMBER")
        try:
            return self.archive.read(name)
        except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError):
            raise CatalogError("MEMBER_READ_FAILED") from None

    def xml(self, name):
        if name not in self.parts:
            blob = self.read(name)
            if len(blob) > self.limits.xml_bytes:
                raise CatalogError("XML_LIMIT")
            self.parts[name] = _XML(blob)
        return self.parts[name]

    def relationships(self, part):
        if part not in self.rels:
            name = posixpath.join(posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels")
            rows = {}
            if name in self.names:
                tree = self.xml(name).root
                if tree.tag != f"{{{REL}}}Relationships":
                    raise CatalogError("RELATIONSHIP_SCHEMA")
                for element in tree:
                    if not isinstance(element.tag, str):
                        continue
                    if element.tag != f"{{{REL}}}Relationship":
                        raise CatalogError("RELATIONSHIP_SCHEMA")
                    rid = element.get("Id")
                    if not rid or rid in rows:
                        raise CatalogError("AMBIGUOUS_RELATIONSHIP")
                    rows[rid] = dict(element.attrib)
            self.rels[part] = rows
        return self.rels[part]

    def resolve(self, part, rid, types):
        relation = self.relationships(part).get(rid)
        if relation is None:
            raise CatalogError("MISSING_RELATIONSHIP")
        if relation.get("TargetMode", "Internal") != "Internal":
            raise CatalogError("EXTERNAL_RELATIONSHIP")
        target = relation.get("Target", "")
        decoded = unquote(target)
        try:
            split = urlsplit(decoded)
        except ValueError:
            raise CatalogError("UNSAFE_RELATIONSHIP_TARGET") from None
        if (not decoded or split.scheme or split.netloc or split.query or split.fragment
                or "\\" in decoded or "\x00" in decoded or decoded.startswith("/")):
            raise CatalogError("UNSAFE_RELATIONSHIP_TARGET")
        member = posixpath.normpath(posixpath.join(posixpath.dirname(part), decoded))
        if member.startswith("../") or member in (".", ".."):
            raise CatalogError("UNSAFE_RELATIONSHIP_TARGET")
        _member_name(member)
        if relation.get("Type") not in {R + "/" + kind for kind in types}:
            raise CatalogError("RELATIONSHIP_TYPE_MISMATCH")
        data = self.read(member)
        return {"relationship_id": rid, "relationship_type": relation["Type"],
                "target": target, "member": member, "sha256": _sha(data), "bytes": len(data)}


class _XML:
    """Secure tree plus exact source-byte spans, not reserialized XML hashes."""

    def __init__(self, blob):
        try:
            self.root = etree.fromstring(blob, etree.XMLParser(
                resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False))
            if self.root.getroottree().docinfo.doctype:
                raise CatalogError("XML_DTD_FORBIDDEN")
            if self.root.getroottree().docinfo.encoding.upper().replace("-", "") not in ("UTF8", "ASCII"):
                raise CatalogError("XML_ENCODING_UNSUPPORTED")
            parser = xml.parsers.expat.ParserCreate()
            stack, spans = [], []

            def start(_name, _attrs):
                pos = parser.CurrentByteIndex
                quote, end = None, pos
                while end < len(blob):
                    c = blob[end]
                    if quote is not None:
                        if c == quote:
                            quote = None
                    elif c in (34, 39):
                        quote = c
                    elif c == 62:
                        break
                    end += 1
                spans.append([pos, None, blob[pos:end].rstrip().endswith(b"/")])
                stack.append((len(spans) - 1, end + 1))

            def end(_name):
                index, open_end = stack.pop()
                pos = parser.CurrentByteIndex
                spans[index][1] = open_end if spans[index][2] else blob.index(b">", pos) + 1

            parser.StartElementHandler = start
            parser.EndElementHandler = end
            parser.Parse(blob, True)
            nodes = [node for node in self.root.iter() if isinstance(node.tag, str)]
            if len(nodes) != len(spans):
                raise CatalogError("XML_SPAN_MISMATCH")
            self.raw = {node: blob[a:b] for node, (a, b, _) in zip(nodes, spans)}
            self.byte_ranges = {node: (a, b) for node, (a, b, _) in zip(nodes, spans)}
            self.order = {node: i for i, node in enumerate(nodes)}
            # Only allow existing structural locators; never execute supplied XPath.
            self.paths = {node.getroottree().getpath(node): node for node in nodes}
        except CatalogError:
            raise
        except (etree.XMLSyntaxError, xml.parsers.expat.ExpatError, ValueError, IndexError):
            raise CatalogError("INVALID_XML") from None


def _location(pkg, path):
    if not isinstance(path, str) or path.count("#") != 1:
        raise CatalogError("INVALID_LOCATOR")
    part, xpath = path.split("#", 1)
    _member_name(part)
    tree = pkg.xml(part)
    if xpath not in tree.paths:
        raise CatalogError("LOCATOR_NOT_FOUND")
    return part, tree, tree.paths[xpath]


def _context(pkg, manifest, part, tree, node):
    units, anchors = [], []
    for unit in manifest["units"]:
        if not unit["source_path"].startswith(part + "#"):
            continue
        _, _, anchor = _location(pkg, unit["source_path"])
        units.append(unit)
        anchors.append(anchor)
    ancestors = {node, *node.iterancestors()}
    containing = [i for i, anchor in enumerate(anchors) if anchor in ancestors]
    if containing:
        deepest = max(len(list(anchors[i].iterancestors())) for i in containing)
        indices = {i for i in containing if len(list(anchors[i].iterancestors())) == deepest}
    else:
        before = [i for i, anchor in enumerate(anchors) if tree.order[anchor] < tree.order[node]]
        after = [i for i, anchor in enumerate(anchors) if tree.order[anchor] > tree.order[node]]
        indices = set(before[-1:] + after[:1])
    selected = set(indices)
    for i in indices:
        selected.update(j for j in (i - 1, i + 1) if 0 <= j < len(units))
    by_id = {unit["id"]: i for i, unit in enumerate(units)}
    tables = {t["id"]: t for t in manifest["tables"]}
    for i in indices:
        unit = units[i]
        ids = [unit.get("context", {}).get("section_heading_id")]
        ids += tables.get(unit.get("table_id"), {}).get("header_evidence_ids", [])
        for eid in ids:
            if eid is not None:
                if eid not in by_id:
                    raise CatalogError("CONTEXT_REFERENCE_MISMATCH")
                selected.add(by_id[eid])
    return [{"evidence_id": units[i]["id"], "start": 0, "end": len(units[i]["text"])}
            for i in sorted(selected) if units[i]["text"].strip()]


def _story_parts(pkg):
    """Find all source stories independently of the supplied manifest version.

    All metadata and story XML goes through the bounded secure package reader.
    Caller-supplied manifest paths never decide which source parts are checked.
    """
    kinds = {name: "body" if name == "word/document.xml" else re.sub(r"\d+$", "", Path(name).stem)
             for name in pkg.names if STORY.fullmatch(name)}
    def add(name, kind):
        _member_name(name)
        if name not in pkg.names:
            raise CatalogError("MISSING_MEMBER")
        if name in kinds and kinds[name] != kind:
            raise CatalogError("STORY_SCHEMA_UNSUPPORTED")
        kinds[name] = kind

    types = pkg.xml("[Content_Types].xml").root
    overrides = {row.get("PartName", "").lstrip("/"): row.get("ContentType")
                 for row in types if isinstance(row.tag, str) and row.tag.endswith("}Override")}
    defaults = {row.get("Extension", "").lower(): row.get("ContentType")
                for row in types if isinstance(row.tag, str) and row.tag.endswith("}Default")}
    for name in set(pkg.names) | set(overrides):
        content_type = overrides.get(name, defaults.get(Path(name).suffix[1:].lower()))
        if content_type in _STORY_CONTENT_TYPES:
            add(name, _STORY_CONTENT_TYPES[content_type])
    for rel_name in sorted(name for name in pkg.names if name.endswith(".rels")):
        directory, filename = posixpath.split(rel_name)
        if posixpath.basename(directory) != "_rels":
            continue
        part = posixpath.join(posixpath.dirname(directory), filename[:-5])
        tree = pkg.xml(rel_name).root
        rows = [dict(row.attrib) for row in tree
                if isinstance(row.tag, str) and row.tag == f"{{{REL}}}Relationship"]
        for relation in rows:
            kind = _STORY_RELATIONSHIPS.get(relation.get("Type"))
            if kind is None:
                continue
            if (not relation.get("Id")
                    or sum(row.get("Id") == relation["Id"] for row in rows) != 1):
                raise CatalogError("AMBIGUOUS_RELATIONSHIP")
            if relation.get("TargetMode", "Internal") != "Internal":
                raise CatalogError("EXTERNAL_RELATIONSHIP")
            target = unquote(relation.get("Target", ""))
            try:
                split = urlsplit(target)
            except ValueError:
                raise CatalogError("UNSAFE_RELATIONSHIP_TARGET") from None
            if (not target or split.scheme or split.netloc or split.query or split.fragment
                    or "\\" in target or "\x00" in target):
                raise CatalogError("UNSAFE_RELATIONSHIP_TARGET")
            name = posixpath.normpath(target.lstrip("/") if target.startswith("/")
                                     else posixpath.join(posixpath.dirname(part), target))
            if name.startswith("../") or name in (".", ".."):
                raise CatalogError("UNSAFE_RELATIONSHIP_TARGET")
            add(name, kind)
    if "word/document.xml" not in kinds:
        raise CatalogError("MAIN_DOCUMENT_MISSING")
    return sorted(kinds.items(), key=lambda value: (value[0] != "word/document.xml", value[0]))


def _verify_text_inventory(pkg, manifest, stories):
    """Replay the pinned builder over already safe XML, never user text guesses.

    This deliberately shares the versioned builder's text representation, including
    table row fragments. It verifies provenance, not interpretation. No second
    DOCX read, package loading, model request or source mutation takes place.
    """
    rebuilt = _new_manifest(manifest["document_sha256"], "not-used", "docx")
    builder = _Builder(rebuilt)
    legacy_paths = {}
    for name, story in stories:
        tree = pkg.xml(name)
        root = tree.root.find(f"{{{W}}}body") if name == "word/document.xml" else tree.root
        if root is None:
            raise CatalogError("STORY_SCHEMA_UNSUPPORTED")
        expected = {"body": "document", "header": "hdr", "footer": "ftr"}.get(story, story)
        if tree.root.tag != f"{{{W}}}{expected}":
            raise CatalogError("STORY_SCHEMA_UNSUPPORTED")
        if name == "word/document.xml" and manifest["parser_version"] == "evidence-manifest-v2":
            table_number = 0
            for block_number, block in enumerate(root):
                if block.tag == f"{{{W}}}p":
                    legacy_paths[name + "#" + block.getroottree().getpath(block)] = f"p{block_number}"
                elif block.tag == f"{{{W}}}tbl":
                    for row_number, row in enumerate(block.findall(f"{{{W}}}tr")):
                        legacy_paths[name + "#" + row.getroottree().getpath(row)] = f"t{table_number}r{row_number}"
                    table_number += 1
        builder.part_name = name
        builder.story = story
        builder.recent, builder.heading, builder.heading_id = [], "", None
        try:
            builder.blocks(root)
        except (ValueError, TypeError, AttributeError, KeyError):
            raise CatalogError("MANIFEST_REBUILD_FAILED") from None
    counts = {}
    for unit in rebuilt["units"]:
        counts[unit["source_path"]] = counts.get(unit["source_path"], 0) + 1
    for unit in rebuilt["units"]:
        path = unit["source_path"]
        if path in legacy_paths and counts[path] == 1:
            unit["legacy_source_id"] = legacy_paths[path]
    if rebuilt["units"] != manifest["units"] or rebuilt["tables"] != manifest["tables"]:
        raise CatalogError("MANIFEST_TEXT_INVENTORY_MISMATCH")


def build_source_object_catalog(docx_path, manifest, *, limits=CatalogLimits()):
    """Catalog every unverified native occurrence; never remove input records.

    Global unsafe-package/schema/source identity errors raise CatalogError.
    Per-entry errors remain in entries with blocking=true. occurrences contains
    only located entries; ledger_ready must be checked before ledger creation.
    Neither ledger_ready nor context_present is task/semantic acceptance.
    """
    if not isinstance(limits, CatalogLimits) or any(type(v) is not int or v <= 0 for v in vars(limits).values()):
        raise CatalogError("INVALID_LIMITS")
    if (not isinstance(manifest, dict) or type(manifest.get("parser_version")) is not str
            or manifest["parser_version"] not in SUPPORTED_MANIFESTS
            or manifest.get("source_kind") != "docx"
            or not all(type(manifest.get(k)) is list for k in ("units", "tables", "images", "unverified_content"))):
        raise CatalogError("MANIFEST_SCHEMA_UNSUPPORTED")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("document_sha256", ""))):
        raise CatalogError("MANIFEST_SOURCE_HASH_INVALID")
    try:
        manifest = json.loads(json.dumps(manifest, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        raise CatalogError("MANIFEST_INVALID_JSON") from None
    seen = set()
    for unit in manifest["units"]:
        if (not isinstance(unit, dict) or not isinstance(unit.get("id"), str) or not unit["id"]
                or unit["id"] in seen or not isinstance(unit.get("text"), str)
                or not isinstance(unit.get("source_path"), str)):
            raise CatalogError("MANIFEST_UNIT_INVALID")
        seen.add(unit["id"])
    with Path(docx_path).open("rb") as stream:
        blob = stream.read(limits.package_bytes + 1)
    if len(blob) > limits.package_bytes:
        raise CatalogError("PACKAGE_LIMIT")
    if _sha(blob) != manifest["document_sha256"]:
        raise CatalogError("SOURCE_HASH_MISMATCH")
    pkg = _Package(blob, limits)
    entries, locators = [], set()
    stories = _story_parts(pkg)
    _verify_text_inventory(pkg, manifest, stories)
    # Schema versions determine record representation, never source coverage.
    # Keep old records intact, but block any newly recognized unlisted instance.
    native_tags = _UNVERIFIED
    # Cross-check the immutable builder inventory against every indexed story.
    actual = set()
    for name, _ in stories:
        tree = pkg.xml(name)
        for xpath, node in tree.paths.items():
            if node.tag in native_tags:
                actual.add(name + "#" + xpath)
    for index, original in enumerate(manifest["unverified_content"]):
        entry = {"inventory_index": index, "inventory_record": deepcopy(original),
                 "occurrence": None, "object_id": None, "native_parts": [], "cache_parts": [],
                 "diagnostics": [], "representation_state": "unread", "task_context_sufficient": None}
        entries.append(entry)
        try:
            if (not isinstance(original, dict) or set(original) != {"source_path", "kind", "verification", "reason"}
                    or original.get("verification") != "unverified"):
                raise CatalogError("INVENTORY_SCHEMA_UNSUPPORTED")
            path = original["source_path"]
            part, tree, node = _location(pkg, path)
            if path in locators:
                raise CatalogError("DUPLICATE_INVENTORY_POSITION")
            locators.add(path)
            if node.tag not in native_tags or original["kind"] != etree.QName(node).localname:
                raise CatalogError("INVENTORY_KIND_MISMATCH")
            oles = list(node.iter(f"{{{O}}}OLEObject"))
            kind = "ole" if oles else "formula" if original["kind"] == "oMath" else "unsupported"
            occurrence = {"source_sha256": manifest["document_sha256"], "locator": path, "kind": kind,
                          "raw_sha256": _sha(tree.raw[node]), "native_sha256": None,
                          "cache_sha256": None, "context_refs": []}
            entry["occurrence"] = occurrence
            entry["raw_basis"] = "exact_original_xml_element_bytes"
            entry["xml_bytes"] = len(tree.raw[node])
            start, end = tree.byte_ranges[node]
            entry["raw_xml_span"] = {"member": part, "start": start, "end": end}
            try:
                occurrence["context_refs"] = _context(pkg, manifest, part, tree, node)
            except CatalogError as exc:
                entry["diagnostics"].append(exc.code)
            if not occurrence["context_refs"]:
                entry["diagnostics"].append("EMPTY_CONTEXT")
            native_refs = [(o.get(f"{{{R}}}id"), {"oleObject", "package"}) for o in oles]
            if node.tag in {f"{{{W}}}altChunk", f"{{{W}}}subDoc"}:
                native_refs.append((node.get(f"{{{R}}}id"), {"aFChunk" if original["kind"] == "altChunk" else "subDocument"}))
            caches = [x for x in node.iter() if x.tag in {f"{{{V}}}imagedata", f"{{{A}}}blip"}]
            handled = {*oles, *caches}
            if node.tag in {f"{{{W}}}altChunk", f"{{{W}}}subDoc"}:
                handled.add(node)
            if any(element not in handled and any(key.startswith("{" + R + "}") for key in element.attrib)
                   for element in node.iter()):
                entry["diagnostics"].append("UNSUPPORTED_RELATIONSHIP_BINDING")
            if any(sum(element.get(f"{{{R}}}{key}") is not None for key in ("id", "embed", "link")) > 1
                   for element in handled):
                entry["diagnostics"].append("AMBIGUOUS_RELATIONSHIP_REFERENCE")
            for image in caches:
                shape = next((x for x in image.iterancestors() if x.tag == f"{{{V}}}shape"), None)
                shape_ids = {o.get("ShapeID") for o in oles if o.get("ShapeID")}
                if shape_ids and (shape is None or shape.get("id") not in shape_ids):
                    entry["diagnostics"].append("CACHE_SHAPE_MISMATCH")
                    continue
                rid = image.get(f"{{{R}}}id") or image.get(f"{{{R}}}embed") or image.get(f"{{{R}}}link")
                try:
                    entry["cache_parts"].append(pkg.resolve(part, rid, {"image"}))
                except CatalogError as exc:
                    entry["diagnostics"].append(exc.code)
            for rid, types in native_refs:
                try:
                    entry["native_parts"].append(pkg.resolve(part, rid, types))
                except CatalogError as exc:
                    entry["diagnostics"].append(exc.code)
            if len(native_refs) > 1 or len(caches) > 1:
                entry["diagnostics"].append("MULTIPLE_REPRESENTATIONS_UNRESOLVED")
            if len(entry["native_parts"]) == 1 and len(native_refs) == 1:
                occurrence["native_sha256"] = entry["native_parts"][0]["sha256"]
            if len(entry["cache_parts"]) == 1 and len(caches) == 1:
                occurrence["cache_sha256"] = entry["cache_parts"][0]["sha256"]
            entry["object_id"] = occurrence_id(occurrence)
        except CatalogError as exc:
            entry["diagnostics"].append(exc.code)
        entry["blocking"] = bool(entry["diagnostics"])
    missing = sorted(actual - locators)
    unexpected = sorted(locators - actual)
    occurrences = [entry["occurrence"] for entry in entries if entry["occurrence"] is not None]
    result = {"schema_version": 1, "catalog_version": "source-object-catalog-candidate-v1",
              "source_sha256": manifest["document_sha256"], "manifest_sha256": _digest(manifest),
              "manifest_parser_version": manifest["parser_version"],
              "image_inventory_sha256": _digest(manifest["images"]), "image_inventory_count": len(manifest["images"]),
              "inventory_sha256": _digest(manifest["unverified_content"]), "inventory_count": len(entries),
              "entries": entries, "occurrences": occurrences,
              "unlisted_xml_occurrences": missing, "unexpected_inventory_locators": unexpected,
              "ledger_ready": not missing and not unexpected and all(not e["blocking"] for e in entries),
              "all_source_semantics_verified": False,
              "text_inventory_verified_against_source": True,
              "context_policy": "Full containing unit(s), adjacent units and section/table header within same XML part; structural context only, not task sufficiency or a read receipt",
              "scope": "Unsupported native-object inventory only; normal image occurrences remain owned by image tools"}
    result["catalog_sha256"] = _digest(result)
    pkg.archive.close()
    return result
