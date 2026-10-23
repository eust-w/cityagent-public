"""Occurrence-preserving OOXML inventory, never a visual interpretation receipt.

Package members are inspected in place. No media is extracted, decoded, edited,
executed, or fetched. Every result remains semantically unverified.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
import stat
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from lxml import etree

from .run_state import file_digest

VERSION = "visual-inventory-v1"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
V = "urn:schemas-microsoft-com:vml"
O = "urn:schemas-microsoft-com:office:office"
IMAGES = {f"{{{A}}}blip", f"{{{V}}}imagedata"}
SPECIAL = {
    **{f"{{{W}}}{name}": name for name in ("altChunk", "object", "subDoc", "sym")},
    f"{{{M}}}oMath": "oMath",
    f"{{{O}}}OLEObject": "OLEObject",
    "{http://schemas.openxmlformats.org/drawingml/2006/chart}chart": "chart",
    "{http://schemas.openxmlformats.org/drawingml/2006/diagram}relIds": "relIds",
}
STORY = re.compile(
    r"word/(document|header\d+|footer\d+|footnotes|endnotes|comments)\.xml$"
)
STORY_CONTENT_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml": "body",
    **{
        f"application/vnd.openxmlformats-officedocument.wordprocessingml.{name}+xml": name
        for name in ("header", "footer", "footnotes", "endnotes", "comments")
    },
}
STORY_RELATIONS = {
    **{
        f"{R}/{name}": name
        for name in ("header", "footer", "footnotes", "endnotes", "comments")
    },
    **{
        f"http://purl.oclc.org/ooxml/officeDocument/relationships/{name}": name
        for name in ("header", "footer", "footnotes", "endnotes", "comments")
    },
}
TRANSFORMS = {
    "srcRect",
    "stretch",
    "tile",
    "xfrm",
    "off",
    "ext",
    "chOff",
    "chExt",
    "extent",
    "effectExtent",
    "rot",
    "positionH",
    "positionV",
}


@dataclass(frozen=True)
class InventoryLimits:
    max_package_bytes: int = 2 * 1024**3
    max_members: int = 50_000
    max_member_bytes: int = 512 * 1024**2
    max_total_bytes: int = 2 * 1024**3
    max_metadata_bytes: int = 32 * 1024**2
    max_metadata_total_bytes: int = 128 * 1024**2
    max_compression_ratio: int = 1000
    max_occurrences: int = 100_000


def _xml(blob: bytes):
    parser = etree.XMLParser(
        resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False
    )
    root = etree.fromstring(blob, parser)
    if root.getroottree().docinfo.doctype or any(
        isinstance(n, etree._Entity) for n in root.iter()
    ):
        raise ValueError("DTD and entity declarations are not accepted")
    return root


def _safe_name(name: str) -> bool:
    return (
        bool(name)
        and not name.startswith("/")
        and "\\" not in name
        and not any(ord(c) < 32 for c in name)
        and all(p not in {"", ".", ".."} for p in name.rstrip("/").split("/"))
        and ":" not in name
    )


class _Package:
    def __init__(self, archive, limits):
        self.archive, self.limits = archive, limits
        infos = archive.infolist()
        if len(infos) > limits.max_members:
            raise ValueError("package member limit exceeded")
        self.members = {}
        total = 0
        for info in infos:
            if not _safe_name(info.filename) or info.filename in self.members:
                raise ValueError("unsafe or duplicate package member")
            if info.flag_bits & 1 or info.compress_type not in {
                zipfile.ZIP_STORED,
                zipfile.ZIP_DEFLATED,
            }:
                raise ValueError("encrypted or unsupported package member")
            if stat.S_IFMT(info.external_attr >> 16) not in {
                0,
                stat.S_IFREG,
                stat.S_IFDIR,
            }:
                raise ValueError("special file package member")
            total += info.file_size
            if (
                info.file_size > limits.max_member_bytes
                or total > limits.max_total_bytes
            ):
                raise ValueError("package expansion limit exceeded")
            if (
                info.file_size
                > max(1, info.compress_size) * limits.max_compression_ratio
            ):
                raise ValueError("package compression ratio limit exceeded")
            self.members[info.filename] = info
        self.metadata_bytes = 0
        self.blobs = {}

    def metadata(self, name):
        info = self.members[name]
        self.metadata_bytes += info.file_size
        if (
            info.file_size > self.limits.max_metadata_bytes
            or self.metadata_bytes > self.limits.max_metadata_total_bytes
        ):
            raise ValueError("XML metadata limit exceeded")
        with self.archive.open(info) as handle:
            data = handle.read(self.limits.max_metadata_bytes + 1)
        if len(data) != info.file_size:
            raise ValueError("inconsistent metadata length")
        return _xml(data)

    def blob(self, name, content_type):
        if name not in self.blobs:
            info = self.members[name]
            digest, size = hashlib.sha256(), 0
            with self.archive.open(info) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    size += len(chunk)
                    if size > self.limits.max_member_bytes:
                        raise ValueError("blob expansion limit exceeded")
                    digest.update(chunk)
            if size != info.file_size:
                raise ValueError("inconsistent blob length")
            self.blobs[name] = {
                "part_name": name,
                "sha256": digest.hexdigest(),
                "size_bytes": size,
                "content_type": content_type,
                "extension": Path(name).suffix.lower(),
            }
        return self.blobs[name]


def _source(part, element):
    return part + "#" + element.getroottree().getpath(element)


def _identifier(document_sha256, source):
    return "V" + hashlib.sha256((document_sha256 + "\n" + source).encode()).hexdigest()


def _target(part, target):
    parsed = urlsplit(target)
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or "\\" in target
    ):
        raise ValueError("unsafe internal relationship target")
    decoded = unquote(parsed.path)
    path = (
        posixpath.normpath(posixpath.join(posixpath.dirname(part), decoded))
        if not decoded.startswith("/")
        else decoded.lstrip("/")
    )
    if not _safe_name(path):
        raise ValueError("relationship target escapes package")
    return path


def inventory_docx(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    limits: InventoryLimits | None = None,
) -> dict:
    """Hash media once per package part while retaining every XML occurrence."""
    path, limits = Path(path), limits or InventoryLimits()
    if path.suffix.lower() != ".docx":
        raise ValueError(
            "DOCX required; legacy DOC conversion must be separately evidenced"
        )
    before = path.stat()
    if before.st_size > limits.max_package_bytes:
        raise ValueError("package size limit exceeded")
    digest = file_digest(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("report SHA256 differs from frozen manifest")
    result = {
        "schema_version": 1,
        "inventory_version": VERSION,
        "document_sha256": digest,
        "source_kind": "docx",
        "occurrences": [],
        "interpretation_verified": False,
        "document_complete": False,
        "limitations": [
            "Inventory binds bytes and XML occurrences only; pixels and special objects remain uninterpreted",
            "OOXML order does not establish rendered page order; alternate branches are both retained",
        ],
    }
    with zipfile.ZipFile(path) as archive:
        package = _Package(archive, limits)
        types = package.metadata("[Content_Types].xml")
        defaults = {
            n.get("Extension", "").lower(): n.get("ContentType")
            for n in types
            if isinstance(n.tag, str) and etree.QName(n).localname == "Default"
        }
        overrides = {
            n.get("PartName", "").lstrip("/"): n.get("ContentType")
            for n in types
            if isinstance(n.tag, str) and etree.QName(n).localname == "Override"
        }
        if "word/document.xml" not in package.members:
            raise ValueError("main document part missing")
        # OPC part names are arbitrary. Content types and relationships, not
        # header1.xml-style names, establish which XML parts carry stories.
        story_kinds = {"word/document.xml": "body"}
        for member in package.members:
            content_type = overrides.get(
                member, defaults.get(Path(member).suffix[1:].lower())
            )
            kind = STORY_CONTENT_TYPES.get(content_type)
            if kind:
                story_kinds[member] = kind
            elif STORY.fullmatch(member):
                story_kinds[member] = (
                    "body"
                    if member == "word/document.xml"
                    else re.sub(r"\d+$", "", Path(member).stem)
                )
        for member, content_type in overrides.items():
            if content_type in STORY_CONTENT_TYPES and member not in package.members:
                raise ValueError("declared OOXML story part missing")
        relationships = {}
        pending = list(story_kinds)
        while pending:
            part = pending.pop()
            if part in relationships:
                continue
            rel_name = posixpath.join(
                posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels"
            )
            rels = {}
            if rel_name in package.members:
                for relation in package.metadata(rel_name):
                    if not isinstance(relation.tag, str):
                        continue
                    rid = relation.get("Id")
                    if not rid or rid in rels:
                        raise ValueError("missing or duplicate relationship identity")
                    rels[rid] = dict(relation.attrib)
                    kind = STORY_RELATIONS.get(relation.get("Type"))
                    if kind:
                        if relation.get("TargetMode", "Internal").lower() != "internal":
                            raise ValueError(
                                "external OOXML story relationship is not fetched"
                            )
                        member = _target(part, relation.get("Target", ""))
                        if member not in package.members:
                            raise ValueError("referenced OOXML story part missing")
                        if member in story_kinds and story_kinds[member] != kind:
                            raise ValueError("conflicting OOXML story identity")
                        story_kinds[member] = kind
                        if member not in relationships:
                            pending.append(member)
            relationships[part] = rels
        for part in sorted(story_kinds, key=lambda n: (n != "word/document.xml", n)):
            root = package.metadata(part)
            story = story_kinds[part]
            expected_root = {"body": "document", "header": "hdr", "footer": "ftr"}.get(
                story, story
            )
            if root.tag != f"{{{W}}}{expected_root}":
                raise ValueError("unexpected OOXML story root")
            rels = relationships[part]
            for element in root.iter():
                if element.tag not in IMAGES and element.tag not in SPECIAL:
                    continue
                if len(result["occurrences"]) >= limits.max_occurrences:
                    raise ValueError("visual occurrence limit exceeded")
                source = _source(part, element)
                kind = "image" if element.tag in IMAGES else SPECIAL[element.tag]
                ancestors = list(element.iterancestors())
                context = next(
                    (
                        n
                        for n in ancestors
                        if etree.QName(n).localname
                        in {"pic", "pict", "drawing", "object"}
                    ),
                    element,
                )
                metadata = [
                    {"tag": n.tag, "attributes": dict(sorted(n.attrib.items()))}
                    for n in context.iter()
                    if isinstance(n.tag, str) and etree.QName(n).localname in TRANSFORMS
                ]
                if element.tag == f"{{{V}}}imagedata":
                    metadata.append(
                        {
                            "tag": element.tag,
                            "attributes": {
                                k: v
                                for k, v in sorted(element.attrib.items())
                                if k
                                in {
                                    "croptop",
                                    "cropbottom",
                                    "cropleft",
                                    "cropright",
                                    "gain",
                                    "blacklevel",
                                    "gamma",
                                    "grayscale",
                                    "bilevel",
                                }
                            },
                        }
                    )
                    metadata.extend(
                        {
                            "tag": n.tag,
                            "attributes": {
                                k: v
                                for k, v in sorted(n.attrib.items())
                                if k
                                in {
                                    "style",
                                    "coordsize",
                                    "coordorigin",
                                    "rotation",
                                    "flip",
                                }
                            },
                        }
                        for n in ancestors
                        if n.tag.startswith("{" + V + "}")
                    )
                occurrence = {
                    "id": _identifier(digest, source),
                    "document_sha256": digest,
                    "source_path": source,
                    "part_name": part,
                    "story": story,
                    "kind": kind,
                    "inline_metadata": dict(sorted(element.attrib.items()))
                    if kind in {"sym", "OLEObject"}
                    else {},
                    "xml_sha256": hashlib.sha256(
                        etree.tostring(element, method="c14n")
                    ).hexdigest(),
                    "display_xml_sha256": hashlib.sha256(
                        etree.tostring(context, method="c14n")
                    ).hexdigest(),
                    "transform_metadata": metadata,
                    "parent_objects": [
                        {
                            "id": _identifier(digest, _source(part, n)),
                            "kind": SPECIAL[n.tag],
                            "source_path": _source(part, n),
                        }
                        for n in ancestors
                        if n.tag in SPECIAL
                    ],
                    "alternate_content": [
                        {
                            "source_path": _source(part, n),
                            "branch": etree.QName(n).localname,
                            "requires": n.get("Requires"),
                        }
                        for n in ancestors
                        if n.tag
                        in {
                            f"{{{MC}}}AlternateContent",
                            f"{{{MC}}}Choice",
                            f"{{{MC}}}Fallback",
                        }
                    ],
                    "relationships": [],
                    "verification": "unverified",
                }
                references = [
                    (key, value)
                    for key, value in element.attrib.items()
                    if key.startswith("{" + R + "}")
                ]
                for attribute, rid in references:
                    relation = rels.get(rid)
                    ref = {
                        "attribute": attribute,
                        "relationship_id": rid,
                        "status": "missing_relationship",
                        "target": None,
                        "blob": None,
                    }
                    if relation:
                        target = relation.get("Target", "")
                        ref.update(
                            {
                                "target": target,
                                "relationship_type": relation.get("Type"),
                                "target_mode": relation.get("TargetMode", "Internal"),
                            }
                        )
                        if relation.get("TargetMode", "Internal").lower() == "external":
                            ref["status"] = "external_not_fetched"
                        else:
                            member = _target(part, target)
                            ref["part_name"] = member
                            ref["status"] = (
                                "bound"
                                if member in package.members
                                and not package.members[member].is_dir()
                                else "missing_target"
                            )
                            if ref["status"] == "bound":
                                ref["blob"] = package.blob(
                                    member,
                                    overrides.get(
                                        member,
                                        defaults.get(Path(member).suffix[1:].lower()),
                                    ),
                                )
                    occurrence["relationships"].append(ref)
                if not references:
                    occurrence["binding_status"] = (
                        "inline_special_object"
                        if kind != "image"
                        else "missing_image_relationship"
                    )
                else:
                    occurrence["binding_status"] = (
                        "bound"
                        if all(
                            r["status"] == "bound" for r in occurrence["relationships"]
                        )
                        else "unresolved_relationship"
                    )
                result["occurrences"].append(occurrence)
        blobs = {b["sha256"]: b for b in package.blobs.values()}
        result["counts"] = {
            "occurrences": len(result["occurrences"]),
            "images": sum(o["kind"] == "image" for o in result["occurrences"]),
            "unique_blobs": len(blobs),
            "unique_blob_bytes": sum(b["size_bytes"] for b in blobs.values()),
            "by_kind": dict(Counter(o["kind"] for o in result["occurrences"])),
            "media_extensions": dict(Counter(b["extension"] for b in blobs.values())),
            "unresolved_types": dict(Counter(o["kind"] for o in result["occurrences"])),
            "binding_status": dict(
                Counter(o["binding_status"] for o in result["occurrences"])
            ),
        }
        result["task_cost_inputs"] = {
            "image_occurrences": result["counts"]["images"],
            "unique_bound_blobs": len(blobs),
            "special_object_occurrences": len(result["occurrences"])
            - result["counts"]["images"],
            "model_calls": 0,
            "decoded_pixels": 0,
            "estimated_model_seconds": None,
        }
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise RuntimeError("report changed during inventory")
    return result
