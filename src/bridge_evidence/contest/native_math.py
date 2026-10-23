"""Bounded, offline OMML syntax candidates; never semantic or glyph verification.

This module has no manifest/workspace integration. Feed original XML *part* bytes,
not reserialized fragments: namespace scope and exact occurrence spans are retained.
Only math runs and subscripts are understood. Unknown syntax stays unresolved.
The AST does not evaluate expressions, normalize symbols, or emit plain-text math.

Primary structure references:
https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.math.officemath
https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.math.subscript
"""
from __future__ import annotations

import hashlib
import json
import re
import xml.parsers.expat
from dataclasses import dataclass

from lxml import etree

VERSION = "omml-run-subscript-candidate-v1"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SPACE = "{http://www.w3.org/XML/1998/namespace}space"


@dataclass(frozen=True)
class NativeMathLimits:
    max_xml_bytes: int = 8 * 1024 * 1024
    max_math_bytes: int = 64 * 1024
    max_total_math_bytes: int = 8 * 1024 * 1024
    max_output_bytes: int = 16 * 1024 * 1024
    max_depth: int = 128
    max_nodes: int = 250_000
    max_occurrences: int = 4096


class _Unresolved(ValueError):
    """Stable diagnostic only; never expose XML parser exception content."""


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _opening_end(blob, start):
    quote = None
    for index in range(start, len(blob)):
        char = blob[index]
        if quote is not None:
            if char == quote:
                quote = None
        elif char in (34, 39):
            quote = char
        elif char == 62:
            return index + 1
    raise _Unresolved("INVALID_XML")


def _bounded_tree(blob, limits):
    """Bound depth/nodes before constructing a tree; retain exact element spans."""
    try:
        blob.decode("utf-8")
    except UnicodeDecodeError:
        raise _Unresolved("XML_ENCODING_UNSUPPORTED") from None
    parser = xml.parsers.expat.ParserCreate(namespace_separator="}")
    stack, spans = [], []
    occurrences = 0
    node_count = 0

    def count_node(*_args):
        nonlocal node_count
        node_count += 1
        if node_count > limits.max_nodes:
            raise _Unresolved("XML_NODE_LIMIT")

    def declaration(version, encoding, _standalone):
        if version != "1.0":
            raise _Unresolved("XML_VERSION_UNSUPPORTED")
        if encoding and encoding.upper().replace("-", "") not in {"UTF8", "ASCII"}:
            raise _Unresolved("XML_ENCODING_UNSUPPORTED")

    def forbidden_dtd(*_args):
        raise _Unresolved("XML_DTD_FORBIDDEN")

    def start(name, _attrs):
        nonlocal occurrences
        if len(stack) >= limits.max_depth:
            raise _Unresolved("XML_DEPTH_LIMIT")
        count_node()
        if name.rsplit("}", 1)[-1] == "oMath":
            occurrences += 1
            if occurrences > limits.max_occurrences:
                raise _Unresolved("MATH_OCCURRENCE_LIMIT")
        offset = parser.CurrentByteIndex
        opening_end = _opening_end(blob, offset)
        empty = blob[offset:opening_end - 1].rstrip().endswith(b"/")
        spans.append([offset, None])
        stack.append((len(spans) - 1, opening_end, empty))

    def end(_name):
        index, opening_end, empty = stack.pop()
        spans[index][1] = opening_end if empty else blob.index(b">", parser.CurrentByteIndex) + 1

    parser.XmlDeclHandler = declaration
    parser.StartDoctypeDeclHandler = forbidden_dtd
    parser.ExternalEntityRefHandler = forbidden_dtd
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CommentHandler = count_node
    parser.ProcessingInstructionHandler = count_node
    try:
        parser.Parse(blob, True)
        root = etree.fromstring(blob, etree.XMLParser(
            resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False,
            remove_comments=False, remove_pis=False,
        ))
    except (xml.parsers.expat.ExpatError, etree.XMLSyntaxError, IndexError, ValueError) as exc:
        if isinstance(exc, _Unresolved):
            raise
        raise _Unresolved("INVALID_XML") from None
    nodes = [node for node in root.iter() if isinstance(node.tag, str)]
    if len(nodes) != len(spans):
        raise _Unresolved("XML_SPAN_MISMATCH")
    return root, dict(zip(nodes, spans))


def _check(node, allowed_attrs=(), required_attrs=(), *, text=False):
    if not isinstance(node.tag, str):
        raise _Unresolved("UNSUPPORTED_XML_NODE")
    if set(node.attrib) - set(allowed_attrs):
        raise _Unresolved("UNSUPPORTED_ATTRIBUTE")
    if set(required_attrs) - set(node.attrib):
        raise _Unresolved("MISSING_ATTRIBUTE")
    if not text and node.text and node.text.strip(" \t\r\n"):
        raise _Unresolved("UNMODELED_TEXT")
    for child in node:
        if not isinstance(child.tag, str):
            raise _Unresolved("UNSUPPORTED_XML_NODE")
        if child.tail and child.tail.strip(" \t\r\n"):
            raise _Unresolved("UNMODELED_TEXT")


def _property(node):
    """Preserve supported formatting explicitly, without interpreting glyphs."""
    containers = {
        M + "sSubPr": {M + "ctrlPr"},
        M + "ctrlPr": {W + "rPr"},
        M + "rPr": {M + "sty"},
        W + "rPr": {W + name for name in ("rFonts", "i", "color", "kern", "sz", "lang")},
    }
    if node.tag in containers:
        _check(node)
        tags = [child.tag for child in node]
        if len(tags) != len(set(tags)) or set(tags) - containers[node.tag]:
            raise _Unresolved("UNSUPPORTED_PROPERTY_STRUCTURE")
        children = [_property(child) for child in node]
    else:
        attrs = {
            M + "sty": ({M + "val"}, {M + "val"}),
            W + "rFonts": ({W + x for x in ("ascii", "hAnsi", "cs", "eastAsia")}, set()),
            W + "i": ({W + "val"}, set()),
            W + "color": ({W + "val"}, {W + "val"}),
            W + "kern": ({W + "val"}, {W + "val"}),
            W + "sz": ({W + "val"}, {W + "val"}),
            W + "lang": ({W + x for x in ("val", "eastAsia", "bidi")}, set()),
        }
        if node.tag not in attrs:
            raise _Unresolved("UNSUPPORTED_PROPERTY")
        _check(node, *attrs[node.tag])
        if len(node):
            raise _Unresolved("UNSUPPORTED_PROPERTY_STRUCTURE")
        value = node.get(M + "val" if node.tag == M + "sty" else W + "val")
        if node.tag == M + "sty" and value not in {"p", "b", "bi", "i"}:
            raise _Unresolved("UNSUPPORTED_PROPERTY_VALUE")
        if node.tag == W + "i" and value not in {None, "0", "1", "true", "false", "on", "off"}:
            raise _Unresolved("UNSUPPORTED_PROPERTY_VALUE")
        if node.tag in {W + "kern", W + "sz"} and (
                not re.fullmatch(r"[0-9]{1,5}", value) or int(value) > 65535):
            raise _Unresolved("UNSUPPORTED_PROPERTY_VALUE")
        if node.tag == W + "color" and value != "auto" and not re.fullmatch(r"[0-9a-fA-F]{6}", value):
            raise _Unresolved("UNSUPPORTED_PROPERTY_VALUE")
        if any(not value for value in node.attrib.values()):
            raise _Unresolved("UNSUPPORTED_PROPERTY_VALUE")
        children = []
    return {"tag": node.tag, "attributes": dict(node.attrib), "children": children}


def _sequence(node):
    _check(node)
    if not len(node):
        raise _Unresolved("EMPTY_MATH_SEQUENCE")
    return {"type": "sequence", "children": [_expression(child) for child in node]}


def _expression(node):
    _check(node)
    children = list(node)
    if node.tag == M + "r":
        properties = []
        for tag in (M + "rPr", W + "rPr"):
            if children and children[0].tag == tag:
                properties.append(_property(children.pop(0)))
        if len(children) != 1 or children[0].tag != M + "t":
            raise _Unresolved("UNSUPPORTED_RUN_STRUCTURE")
        token = children[0]
        _check(token, {SPACE}, text=True)
        if len(token) or token.get(SPACE) not in {None, "default", "preserve"}:
            raise _Unresolved("UNSUPPORTED_TEXT_STRUCTURE")
        return {"type": "run", "text": token.text or "", "text_attributes": dict(token.attrib),
                "properties": properties}
    if node.tag == M + "sSub":
        properties = []
        if children and children[0].tag == M + "sSubPr":
            properties.append(_property(children.pop(0)))
        if [child.tag for child in children] != [M + "e", M + "sub"]:
            raise _Unresolved("UNSUPPORTED_SUBSCRIPT_STRUCTURE")
        return {"type": "subscript", "base": _sequence(children[0]),
                "subscript": _sequence(children[1]), "properties": properties}
    raise _Unresolved("UNSUPPORTED_MATH_ELEMENT")


def extract_omml_candidates(xml_bytes: bytes, *, limits: NativeMathLimits = NativeMathLimits()) -> dict:
    """Inventory each oMath in original XML, including unsupported occurrences.

    Global parser/limit failures return inventory_complete=False and no partial
    inventory. A per-math failure retains its full raw XML, span and diagnostics.
    Caller must bind source_xml_sha256 and spans to its original document package.
    No result changes any manifest, disposition, coverage gate or text source.
    """
    if type(xml_bytes) is not bytes:
        raise TypeError("xml_bytes must be original bytes")
    if (not isinstance(limits, NativeMathLimits)
            or any(type(value) is not int or value <= 0 for value in vars(limits).values())
            or limits.max_depth > 128):
        raise ValueError("Require positive integer limits and max_depth <= 128")
    result = {"parser_version": VERSION, "source_xml_sha256": _sha(xml_bytes),
              "source_xml_bytes": len(xml_bytes), "inventory_complete": False,
              "occurrences": [], "diagnostics": [], "semantics_verified": False,
              "limitations": ["Syntax candidates only; surrounding document/style semantics and glyph fidelity remain unverified."]}
    try:
        if len(xml_bytes) > limits.max_xml_bytes:
            raise _Unresolved("XML_BYTE_LIMIT")
        root, spans = _bounded_tree(xml_bytes, limits)
    except _Unresolved as exc:
        result["diagnostics"] = [str(exc)]
        return result
    # A future/Strict math namespace must not disappear as "no math found".
    maths = [node for node in root.iter()
             if isinstance(node.tag, str) and etree.QName(node).localname == "oMath"]
    # Nested objects overlap. Bound retained bytes before copying any fragment.
    if sum(spans[node][1] - spans[node][0] for node in maths) > limits.max_total_math_bytes:
        result["diagnostics"] = ["MATH_TOTAL_BYTE_LIMIT"]
        return result
    indices = {node: index for index, node in enumerate(maths)}
    # Count the complete canonical JSON envelope plus each occurrence. Namespace
    # contexts/paths/ASTs can amplify output even when raw fragments are small.
    output_bytes = len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if output_bytes > limits.max_output_bytes:
        result["diagnostics"] = ["MATH_OUTPUT_BYTE_LIMIT"]
        return result
    for index, node in enumerate(maths):
        start, end = spans[node]
        raw = xml_bytes[start:end]
        ancestor_index = next((indices[a] for a in node.iterancestors() if a in indices), None)
        row = {"occurrence_index": index, "ancestor_math_occurrence_index": ancestor_index,
               "source_xml_span": {"start": start, "end": end},
               "xml_path": node.getroottree().getpath(node),
               "namespace_context": {key or "": value for key, value in node.nsmap.items()},
               "raw_xml": raw.decode("utf-8"), "raw_sha256": _sha(raw),
               "status": "unresolved", "candidate_ast": None, "diagnostics": [],
               "semantics_verified": False, "glyph_fidelity": "unverified"}
        try:
            if len(raw) > limits.max_math_bytes:
                raise _Unresolved("MATH_BYTE_LIMIT")
            if node.tag != M + "oMath":
                raise _Unresolved("UNSUPPORTED_MATH_NAMESPACE")
            row["candidate_ast"] = _sequence(node)
            row["status"] = "candidate"
        except _Unresolved as exc:
            row["diagnostics"] = [str(exc)]
        output_bytes += len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1
        if output_bytes > limits.max_output_bytes:
            result["occurrences"] = []
            result["diagnostics"] = ["MATH_OUTPUT_BYTE_LIMIT"]
            return result
        result["occurrences"].append(row)
    result["inventory_complete"] = True
    return result
