"""A bounded, local-model TongAgents ReAct bridge.

The SDK is imported lazily and no mock or direct-model fallback exists here.
``completed`` means a native final event or a separately proven section handoff;
the caller must separately validate extraction coverage and business results.
SDK events and tool receipts retain hashes/shape, never complete source text.

TongAgents' decorator registers tools globally.  Register stable dispatchers
once, serialize sessions, and bind the current handlers only for that session.
This also handles SDKs that cache tool instances between successive documents.
"""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import threading
import time
import traceback
import uuid
from typing import Any
from urllib.parse import urlsplit

from .local_model_client import validate_temperature
from .document_main_review_tools import SCHEMAS as MAIN_REVIEW_SCHEMAS, DESCRIPTIONS as MAIN_REVIEW_DESCRIPTIONS


_READING_TOOLS = frozenset({
    "read_outline", "read_table_rows", "read_evidence", "search_in_document", "get_neighbors", "extract_chunk",
    "review_conflict", "split_task", "finish", "read_source", "read_source_range", "read_section_range", "read_table_range", "read_section", "read_image",
    "read_workspace", "read_workspace_page", "update_workspace", "review_draft", "inspect_delivery",
    "save_defects", "save_recommendations",
    "read_source_objects", "read_source_object", "propose_source_object", "review_candidates", "submit_review", "quick_review", *MAIN_REVIEW_SCHEMAS,
})
from .document_review_lookup import LOOKUP_TOOLS
_REVIEW_SCHEMAS = {tool["function"]["name"]: tool["function"] for tool in LOOKUP_TOOLS}
_REVIEW_TOOLS = frozenset(_REVIEW_SCHEMAS)
from .document_dedicated_tools import SAVE_SECTIONS, dedicated_schema, dedicated_outer_schema
_SECTION_ENTRY_TOOLS = frozenset({"save_entries", "read_entries", "complete_section", "handoff_section", "submit_section", *SAVE_SECTIONS})
_SECTION_SOURCE_TOOLS = frozenset({
    "read_outline", "read_source", "read_source_range", "read_section_range", "read_table_range", "read_section", "read_table_rows", "search_in_document", "get_neighbors",
})
_SECTION_TOOLS = _SECTION_SOURCE_TOOLS | _SECTION_ENTRY_TOOLS
_ALLOWED_TOOLS = _READING_TOOLS | _REVIEW_TOOLS | _SECTION_TOOLS
_INTERNAL_RANGE_TOOLS = frozenset({"read_source_range", "read_section_range", "read_table_range"})
_MANAGED_READ_TOOLS = frozenset({"read_source", "read_section", "read_table_rows", "read_outline",
                               "search_in_document", "read_workspace_page", "read_entries", "inspect_delivery"})
NATIVE_REVIEW_ADAPTER_VERSION = "readonly-native-review-handoff-v1"
_SESSION_LOCK = threading.RLock()
_ACTIVE_SESSION: dict[str, Any] | None = None
# Hold decorator references too, so a recycled Python object id cannot reuse a
# registration made against a different SDK/module instance (also useful in tests).
_REGISTRATIONS: list[tuple[Any, set[str]]] = []
# Keep the actual registered native classes; signature-derived schemas are not
# sufficient proof that the SDK rejected these exact arguments.
_REGISTERED_TOOL_CLASSES: list[tuple[Any, dict[str, type]]] = []


class ToolSchemaUnavailableError(RuntimeError):
    """The SDK cannot expose a required native document tool schema."""


class AgentCallAssociationError(RuntimeError):
    """A trusted call-ID provider failed; never guess an association."""


class _SectionHandoffComplete(RuntimeError):
    """Private control signal, raised only after the native tool batch settles."""


class _StrictToolArgumentJSONError(ValueError):
    """Adapter-owned pre-validation rejection; its marker contains no input."""


_STRICT_ARGUMENT_JSON_MARKER = "CITYAGENT_STRICT_TOOL_ARGUMENT_JSON_V1:"


class NativeBatchRecoveryError(AgentCallAssociationError):
    """A stopped SDK batch lacks exact recovery proof, without source text."""

    def __init__(self, reason: str):
        self.proof_reason = reason
        super().__init__("Native stopped-tool-batch recovery proof failed")


def _load_tongagents() -> dict[str, Any]:
    base = importlib.import_module("tongagents.agents.llm.base")
    agent = importlib.import_module("tongagents.agents.llm_agent")
    manager = importlib.import_module("tongagents.tools.tool_manager")
    return {
        "ModelConfig": base.ModelConfig,
        "ModelProvider": base.ModelProvider,
        "ReactAgent": agent.ReactAgent,
        "ReactAgentSetting": agent.ReactAgentSetting,
        "LLMRunContext": agent.LLMRunContext,
        "tool": manager.tool,
        "version": importlib.metadata.version("tongagents"),
    }


def _load_tool_schema_api():
    # Divide tools retain their existing registration and do not need this API.
    base = importlib.import_module("tongagents.tools.base")
    return base.Tool, base.ToolParameters


def _section_tool_schema(name: str) -> dict[str, Any]:
    """Exact public contracts for source reads and section-only handoffs.

    A section handoff is not document delivery or semantic review. Entry
    fields come from the store's shared contract, never a second field list.
    """
    if name in SAVE_SECTIONS:
        return dedicated_schema(name)
    if name == "submit_section":
        from .document_section_submission import submit_section_schema
        return submit_section_schema()
    identifier = {"type": "string", "minLength": 1}
    integer = {"type": "integer", "minimum": 0}
    restart = {"type": "boolean", "default": False,
               "description": "Explicitly reread this object from its beginning; otherwise resume its saved position."}
    properties, required = {
        "read_outline": ({"limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                          "restart": restart}, []),
        "read_source": ({"evidence_id": {**identifier, "description": "Canonical E ID or current document MinerU ref returned by tools, e.g. mineru:<docSHA>#/texts/1, #/tables/1/r0/c0. Whole tables, rows and cells are supported; the program continues pages."}, "restart": restart}, ["evidence_id"]),
        "read_source_range": ({"evidence_id": identifier, "start": {**integer, "default": 0},
                         "end": {"type": ["integer", "null"], "minimum": 1, "default": None}}, ["evidence_id"]),
        "read_section_range": ({"section_id": identifier, "cursor": {**integer, "default": 0},
                          "max_units": {"type": "integer", "minimum": 1, "default": 8}}, ["section_id"]),
        "read_table_range": ({"table_id": identifier, "start": integer,
                             "end": {"type": "integer", "minimum": 1}}, ["table_id", "start", "end"]),
        "read_section": ({"section_id": {**identifier, "description": "Section ID from the outline (root/task/canonical section), or current-document mineru:<docSHA>#/sections/S0001. Do not invent IDs."}, "restart": restart,
                          "max_units": {"type": "integer", "minimum": 1, "default": 8}}, ["section_id"]),
        "read_table_rows": ({"table_id": identifier, "restart": restart,
                             "max_rows": {"type": "integer", "minimum": 1, "default": 8}}, ["table_id"]),
        # qwen3_xml reads the top-level parameter type, not anyOf. Keep one
        # native array contract so keyword arrays survive service parsing.
        "search_in_document": ({"query": {"type": "array", "minItems": 1,
                                  "maxItems": 16, "items": identifier},
                                "scope": {"type": "string", "enum": ["default", "task", "document", "section"], "default": "default"},
                                "section_id": {"type": ["string", "null"], "minLength": 1, "default": None},
                                "restart": restart,
                                "max_matches": {"type": "integer", "minimum": 1, "maximum": 100, "default": 8}}, ["query"]),
        "get_neighbors": ({"evidence_id": identifier}, ["evidence_id"]),
        "read_entries": ({"restart": restart,
                          "limit": {"type": "integer", "minimum": 1, "default": 8}}, []),
        "handoff_section": ({"notes": {"type": "string", "default": "", "description": "Plain text: unresolved questions or remaining gaps. Empty means none reported. Do not encode a JSON array."}}, []),
        "complete_section": ({"summary": identifier,
                              "unresolved": {"type": "array", "items": identifier}}, ["summary", "unresolved"]),
    }.get(name, ({}, []))
    if name == "save_entries":
        from .document_entries import entry_schema
        properties, required = {"entries": {"type": "array", "items": entry_schema()}}, ["entries"]
    elif name not in _SECTION_TOOLS:
        raise ValueError("Unknown section tool")
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": required}


def _validate_section_arguments(name: str, args: dict[str, Any]) -> None:
    # Strict JSON preflight also rejects NaN and Python bools masquerading as
    # integer offsets before the exact public schema reaches a handler.
    _validate_json_value(args)
    from jsonschema import Draft202012Validator
    if next(Draft202012Validator(dedicated_outer_schema(name) if name in SAVE_SECTIONS else _section_tool_schema(name)).iter_errors(args), None) is not None:
        raise ValueError("Section arguments do not match the registered tool schema")
    if any(type(args[key]) is not int for key in ("start", "cursor", "max_units", "max_rows", "limit", "max_matches") if key in args):
        raise ValueError("Reading limits and offsets require actual integers")
    if "end" in args and args["end"] is not None and type(args["end"]) is not int:
        raise ValueError("Range end requires an actual integer or null")
    if name in {"read_source_range", "read_table_range"}:
        start, end = args.get("start", 0), args.get("end")
        if end is not None and end <= start:
            raise ValueError("Range requires 0 <= start < end")
    if name in {"save_entries", "submit_section"}:
        for entry in args["entries"]:
            if "id" in entry and not entry["id"].strip():
                raise ValueError("Entry IDs must be nonempty")
            for ref in entry["source_refs"]:
                if (not ref["evidence_id"].strip() or ("start" in ref) != ("end" in ref)
                        or ("start" in ref and (type(ref["start"]) is not int or type(ref["end"]) is not int
                                               or not ref["start"] < ref["end"]))):
                    raise ValueError("References need an evidence ID and an optional complete half-open range")
        if name == "submit_section" and any(not item.strip() for item in args["unresolved"]):
            raise ValueError("Section unresolved items must be nonempty")
    elif name == "search_in_document":
        queries = args["query"]
        if any(not query.strip() for query in queries):
            raise ValueError("Search phrases must be nonempty")
        scope, section = args.get("scope", "default"), args.get("section_id")
        if ((scope == "section" and not section) or (section is not None and not section.strip())
                or (scope in {"task", "document"} and section is not None)):
            raise ValueError("Search scope and section_id disagree")
        if type(args.get("max_matches", 8)) is not int:
            raise ValueError("Search pagination requires actual integers")
    elif name == "read_outline":
        if type(args.get("limit", 20)) is not int:
            raise ValueError("Outline pagination requires actual integers")
    elif name == "read_entries":
        if type(args.get("limit", 8)) is not int:
            raise ValueError("Entry pagination requires actual integers")
    elif name == "handoff_section":
        pass  # Empty notes explicitly mean no reported unresolved questions.
    elif name == "complete_section":
        if not args["summary"].strip() or any(not item.strip() for item in args["unresolved"]):
            raise ValueError("Section handoff strings must be nonempty")
    elif any(type(value) is str and not value.strip() for value in args.values()):
        raise ValueError("Section source IDs and queries must be nonempty")


def _update_workspace_schema() -> dict[str, Any]:
    """Describe patches without defaulting values or rejecting partial edits.

    Whether an ID already exists is workspace state, not a JSON Schema rule.
    New facts still need value/source_refs/status at workspace validation.
    """
    from .docx_writer import ADVICE_FIELDS, BASIC_FIELDS, DEFECT_FIELDS

    identifier = {"type": "string", "minLength": 1}
    ids = {"type": "array", "items": identifier, "uniqueItems": True}
    bound_ids = {**ids, "minItems": 1}
    scalar = {"type": ["string", "number", "null"]}
    basic = {"type": "object", "additionalProperties": False,
             "properties": {field: scalar for field, _ in BASIC_FIELDS}}
    rows = {section: {"type": "object", "additionalProperties": False,
                     "properties": {field: scalar for field in fields}}
            for section, fields in (("维修建议", ADVICE_FIELDS), ("病害明细", DEFECT_FIELDS))}
    draft = {"type": "object", "required": ["基本信息", "维修建议", "病害明细"],
             "additionalProperties": False, "properties": {"基本信息": basic,
             **{section: {"type": "array", "items": row} for section, row in rows.items()}}}
    refs = {"type": "array", "items": {
        "type": "object", "properties": {
            "evidence_id": {**identifier, "description": "Copy the source tool's canonical E ID or document-bound mineru:<docSHA>#/... alias. ID alone resolves exact source spans; alias offsets are not accepted."}, "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 1}, "image_id": identifier,
            "source_path": identifier, "observation_id": identifier},
        "anyOf": [{"required": ["evidence_id", "start", "end"]},
                  {"required": ["evidence_id"], "not": {"anyOf": [{"required": ["start"]}, {"required": ["end"]}]}},
                  {"required": ["image_id"]}]}}
    status = {"type": "string", "enum": ["explicit", "inferred", "uncertain", "conflict"]}
    facts = {"type": "array", "items": {
        "type": "object", "required": ["id"],
        "description": "New facts require id/value/source_refs/status. Existing IDs may update only changed fields; nothing is auto-filled.",
        "properties": {"id": identifier, "value": {}, "source_refs": refs,
                       "status": status, "depends_on": ids}}}
    relations = {"type": "array", "items": {
        "type": "object", "required": ["id"], "properties": {
            "id": identifier, "subject_id": identifier, "predicate": identifier,
            "object_id": identifier, "source_refs": refs, "status": status, "depends_on": ids}}}
    questions = {"type": "array", "items": {
        "type": "object", "required": ["id"], "properties": {
            "id": identifier, "question": {"type": "string"},
            "status": {"type": "string", "enum": ["open", "resolved", "unknown", "conflict"]},
            "reason": {"type": "string", "description": "Required when status is resolved or unknown."},
            "depends_on": ids}}}
    patch_variants = [
        {"op": {"const": "set_basic"}, "field": {"type": "string", "enum": [field for field, _ in BASIC_FIELDS]},
         "value": scalar, "fact_ids": bound_ids},
        {"op": {"const": "remove_row"}, "section": {"enum": ["维修建议", "病害明细"]},
         "index": {"type": "integer", "minimum": 0}},
        {"op": {"const": "bind_output"}, "path": identifier, "fact_ids": bound_ids},
    ]
    for op in ("append_row", "replace_row"):
        for section, row in rows.items():
            properties = {"op": {"const": op}, "section": {"const": section}, "row": row, "fact_ids": bound_ids}
            if op == "replace_row":
                properties["index"] = {"type": "integer", "minimum": 0}
            patch_variants.append(properties)
    patches = {"type": "array", "minItems": 1, "items": {"oneOf": [
        {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}
        for properties in patch_variants]}, "description":
        "Small atomic edits; first set_basic/append_row creates the empty three-section draft. bind_output only replaces an existing field/row's fact IDs after the model verifies support. Never resend a whole draft to repair one item. A failed update saves none of its fields/facts/operations."}
    row_append = {
        "type": "object", "additionalProperties": False,
        "required": ["fact_id", "section", "row", "source_refs", "status"],
        "description": "Preferred for one NEW detail row. Model supplies all five fields. Host only expands to one fact(value=row) plus one append_row. Every nonempty row field needs support in the supplied refs. No defaults, no inference, no existing-ID overwrite. With row_append, update may contain only base_version and optional working_notes; never mix legacy content fields.",
        "properties": {"fact_id": identifier,
                       "section": {"type": "string", "enum": ["维修建议", "病害明细"]},
                       "row": {"anyOf": list(rows.values()), "description": "One row using the selected section's delivery columns; all values explicitly model-authored, written only once."},
                       "source_refs": refs, "status": status},
        "allOf": [{"if": {"properties": {"section": {"const": section}}},
                   "then": {"properties": {"row": row}}} for section, row in rows.items()]}
    return {"type": "object", "required": ["update"], "additionalProperties": False,
            "properties": {"update": {
                "type": "object", "required": ["base_version"], "additionalProperties": False,
                "description": "Workspace update object. Read the current version first. Other documented workspace fields remain supported and validated by the workspace.",
                "properties": {
                    "base_version": {"type": "integer", "minimum": 0,
                                     "description": "Required: latest version returned by read_workspace; never omit or guess."},
                    "facts": facts, "relations": relations, "questions": questions, "draft_patch": patches,
                    "draft": draft, "draft_fact_ids": ids,
                    "draft_bindings": {"type": "object", "additionalProperties": bound_ids,
                        "description": "Each nonempty basic field uses 基本信息.<standard field>; each detail row uses 维修建议.<zero-based index> or 病害明细.<zero-based index>. IDs must be active and also in draft_fact_ids. Full replacement requires draft/draft_fact_ids/draft_bindings together; prefer draft_patch."},
                    "retract_fact_ids": ids, "retract_relation_ids": ids, "retract_question_ids": ids,
                    "exclusions": {"type": "array", "items": {"type": "object", "properties": {
                        "evidence_id": identifier, "start": {"type": "integer", "minimum": 0},
                        "end": {"type": "integer", "minimum": 1}, "image_id": identifier, "reason": identifier},
                        "required": ["reason"], "additionalProperties": False,
                        "oneOf": [{"required": ["evidence_id", "start", "end"]}, {"required": ["image_id"]}]}},
                    "row_append": row_append,
                    "working_notes": {"anyOf": [{"type": "string"}, {"type": "object"}], "description":
                        "Optional model-authored string or finite JSON object progress notes. Strings are stored unchanged as {text: original string}; objects are preserved. The stored object must fit 4096 UTF-8 bytes in canonical JSON. Replaces current notes; {} clears them. Metadata only, never evidence, engineering facts or Word output. Read current workspace version before writing; source-read delivery may advance it. No values are inferred or completed."}},
                "dependentRequired": {field: [other for other in ("draft", "draft_fact_ids", "draft_bindings") if other != field]
                                      for field in ("draft", "draft_fact_ids", "draft_bindings")},
                "allOf": [{"not": {"required": ["draft_patch", field]}}
                          for field in ("draft", "draft_fact_ids", "draft_bindings")]
                    + [{"not": {"required": ["row_append", field]}} for field in (
                        "facts", "relations", "questions", "draft_patch", "draft", "draft_fact_ids", "draft_bindings",
                        "retract_fact_ids", "retract_relation_ids", "retract_question_ids", "exclusions")]}}}


_SOURCE_PROPOSAL_FIELDS = frozenset({"disposition", "rationale", "affected_output_paths",
                                     "supporting_refs", "context_refs", "source_limitations"})
_SOURCE_DISPOSITIONS = ("not_relied_on", "outside_task", "unresolved_relevant", "unclassified")


def _source_object_schema(name: str) -> dict[str, Any]:
    identifier = {"type": "string", "minLength": 1}
    if name == "read_source_objects":
        properties = {"cursor": {"type": "integer", "minimum": 0, "default": 0},
                      "limit": {"type": "integer", "minimum": 1, "default": 10}}
        required = []
    elif name == "read_source_object":
        properties, required = {"object_id": identifier}, ["object_id"]
    elif name == "propose_source_object":
        refs = {"type": "array", "items": {"type": "object", "additionalProperties": False,
                "required": ["evidence_id", "start", "end"], "properties": {
                    "evidence_id": identifier, "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1}}}}
        strings = {"type": "array", "items": identifier, "uniqueItems": True}
        properties = {"object_id": identifier, "proposal": {"type": "object", "additionalProperties": False,
            "required": sorted(_SOURCE_PROPOSAL_FIELDS),
            "description": "Proposal only. Origin call IDs and acceptance are host-owned; independent review is required. Read complete source context first.",
            "properties": {"disposition": {"type": "string", "enum": list(_SOURCE_DISPOSITIONS)},
                "rationale": identifier, "affected_output_paths": strings, "supporting_refs": refs,
                "context_refs": refs, "source_limitations": strings}}}
        required = ["object_id", "proposal"]
    else:
        raise ValueError("Unknown source-object tool")
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _install_tool_schema(instance: Any, name: str, schema: dict[str, Any]) -> None:
    try:
        tool_base, parameters_class = _load_tool_schema_api()
        if not isinstance(instance, tool_base):
            raise TypeError("Decorator did not return an SDK Tool instance")
        tool_class = type(instance)
        tool_class.parameters = parameters_class(schema)
        # Some SDK versions may cache original metadata separately. Refuse to
        # start an agent unless both public information paths expose the schema.
        for accessor in ("get_original_tool_info", "get_tool_info"):
            exposed = getattr(tool_class, accessor)().parameters.to_json_schema()
            if exposed != schema:
                raise ValueError("SDK did not preserve the document tool schema")
        _install_argument_normalization(tool_class, name, schema)
    except Exception as exc:
        raise ToolSchemaUnavailableError(f"TongAgents cannot expose the required {name} schema") from exc


def _install_update_workspace_schema(instance: Any) -> None:
    _install_tool_schema(instance, "update_workspace", _update_workspace_schema())


def _delivery_schema(name):
    if name == "inspect_delivery":
        return {"type":"object","additionalProperties":False,"required":["view"],
            "properties":{"view":{"type":"string","enum":["summary","basic","advice","defects"]},
                "limit":{"type":"integer","minimum":1,"maximum":16,"default":4},
                "restart":{"type":"boolean","default":False}}}
    return {"type":"object","additionalProperties":False,"properties":{"self_check":{
        "type":"object","additionalProperties":False,
        "required":["inspected_export","complete","basis","remaining_issues"],
        "properties":{"inspected_export":{"type":"boolean"},"complete":{"type":"boolean"},
            "basis":{"type":"string","description":"Your concise public self-check explanation grounded in the inspected actual output and original report; not an independent review verdict."},
            "remaining_issues":{"type":"array","items":{"type":"string"}}}}}}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _strict_json_loads(text: str) -> Any:
    """Decode JSON only; duplicate keys and non-JSON numbers are ambiguous."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Non-finite JSON number")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    _validate_json_value(value)
    return value


def _normalize_container_arguments(arguments: dict[str, Any], schema: dict[str, Any]):
    """Only top-level container fields with an unambiguous schema are decoded.

    This never changes the original argument object/native proposal. Strings
    allowed by a union (including search text and scalar report values) stay
    verbatim. No recursive decoding, business defaults or key repair occurs.
    """
    _validate_json_value(arguments)
    normalized = json.loads(_json_bytes(arguments))
    changes, issues = [], []
    if type(normalized) is not dict:
        return normalized, changes, issues
    for field, spec in schema.get("properties", {}).items():
        declared = spec.get("type")
        types = [declared] if type(declared) is str else declared if type(declared) is list else []
        containers = set(types) & {"array", "object"}
        if "string" in types or len(containers) != 1 or type(normalized.get(field)) is not str:
            continue
        expected = next(iter(containers))
        raw = normalized[field]
        issue = {"path": field, "expected_type": expected, "actual_type": "string"}
        try:
            decoded = _strict_json_loads(raw)
        except (ValueError, TypeError, RecursionError):
            issues.append({**issue, "reason": "invalid_json_container"})
            continue
        if type(decoded) is not (list if expected == "array" else dict):
            issues.append({**issue, "reason": "decoded_container_type_mismatch"})
            continue
        normalized[field] = decoded
        changes.append({**issue, "conversion": "strict_json_loads", "converted_type": expected,
                        "original_value_sha256": _hash(raw), "converted_value_sha256": _hash(decoded)})
    return normalized, changes, issues


def _normalization_schema_for_tool(name):
    if name in MAIN_REVIEW_SCHEMAS:
        return deepcopy(MAIN_REVIEW_SCHEMAS[name])
    if name in _SECTION_TOOLS:
        return _section_tool_schema(name)
    if name == "update_workspace":
        return _update_workspace_schema()
    if name == "review_candidates":
        from .document_candidate_review import review_candidates_schema
        return review_candidates_schema()
    if name == "quick_review":
        from .document_speed import quick_review_schema
        return quick_review_schema()
    if name == "submit_review":
        from .document_review_submission import submit_review_schema
        return submit_review_schema()
    if name in {"finish", "inspect_delivery"}:
        return _delivery_schema(name)
    if name in {"read_source_objects", "read_source_object", "propose_source_object"}:
        return _source_object_schema(name)
    if name in _REVIEW_TOOLS:
        return _REVIEW_SCHEMAS[name]["parameters"]
    return None


def _strict_outer_argument_object(params):
    """Return a strict object or content-free, repeatable parse diagnostics."""
    try:
        value = _strict_json_loads(params)
    except json.JSONDecodeError as exc:
        return None, {"path": "$", "reason": "invalid_json_syntax",
                      "character_offset": exc.pos, "line": exc.lineno, "column": exc.colno}
    except (ValueError, TypeError, RecursionError) as exc:
        reasons = {"Duplicate JSON key": "duplicate_json_key",
                   "Non-finite JSON number": "nonfinite_json_number"}
        reason = ("json_nesting_limit" if isinstance(exc, RecursionError)
                  else reasons.get(str(exc), "invalid_json_value"))
        return None, {"path": "$", "reason": reason}
    if type(value) is not dict:
        return None, {"path": "$", "reason": "outer_object_required", "actual_type": type(value).__name__}
    return value, None


def _reject_outer_argument_json(name, raw, schema, diagnostic):
    """Record only our own pre-schema failure, never arbitrary SDK errors."""
    session = _ACTIVE_SESSION
    record = {"tool_name": name, "arguments_text_sha256": _hash(raw),
              "schema_sha256": _hash(schema), "diagnostic": diagnostic,
              "handler_executed": False, "saved": False, "arguments_repaired_by_host": False}
    if session is not None and name in session.get("handlers", {}):
        records = session.setdefault("argument_parse_errors", [])
        record.update(agent_call_id=_current_agent_call_id(session), receipt_index=len(records))
        records.append(record)
    # SDK 2.7.20 wraps this exception as a generic ToolErrorAction and copies
    # str(exc) exactly. Recovery requires the exact marker AND this receipt;
    # matching words from a business exception are never sufficient.
    raise _StrictToolArgumentJSONError(_STRICT_ARGUMENT_JSON_MARKER + _hash(record))


def _install_argument_normalization(tool_class: type, name: str, schema: dict[str, Any]) -> None:
    """Wrap the actual SDK validation entry, never its model/wire messages."""
    if tool_class.__dict__.get("_cityagent_normalization_schema_sha256") == _hash(schema):
        return
    original = getattr(tool_class, "_normalize_and_validate_params", None)
    if not callable(original):
        raise TypeError("SDK Tool lacks its pre-validation entry")
    frozen_schema = json.loads(_json_bytes(schema))

    def normalize_and_validate(instance, params, *args, **kwargs):
        # Tool.call forwards the native arguments as JSON text in SDK 2.7.20.
        # Decode that outer transport object strictly before examining its
        # container fields; otherwise the SDK first decodes it only *after*
        # our wrapper, leaving escaped entries unchanged at schema validation.
        # Neither the caller's object nor the native proposal text is mutated.
        if type(params) is str:
            arguments, diagnostic = _strict_outer_argument_object(params)
            if diagnostic is not None:
                _reject_outer_argument_json(name, params, frozen_schema, diagnostic)
        else:
            arguments = params
        if type(arguments) is not dict:
            raise ValueError("Tool arguments must be a strict JSON object")
        normalized, changes, issues = _normalize_container_arguments(arguments, frozen_schema)
        normalized_sha256 = _hash(normalized)
        audit = None
        session = _ACTIVE_SESSION
        if session is not None and name in session.get("handlers", {}) and (changes or issues or name in SAVE_SECTIONS):
            audit = {"tool_name": name, "agent_call_id": _current_agent_call_id(session),
                     "original_arguments_sha256": _hash(arguments),
                     "normalized_arguments_sha256": normalized_sha256, "schema_sha256": _hash(frozen_schema),
                     "conversions": changes, "issues": issues, "native_validation_passed": False,
                     "original_native_arguments_unchanged": True}
            if type(params) is str:
                audit["original_arguments_text_sha256"] = _hash(params)
                audit["native_parameter_representation"] = "json_string"
            session.setdefault("argument_normalizations", []).append(audit)
        if name in SAVE_SECTIONS:
            from jsonschema import Draft202012Validator
            # Keep the full fixed schema visible to the model. At the actual
            # SDK validation entry, check the container; the handler validates
            # every row and provenance BEFORE writing any accepted entry.
            if next(Draft202012Validator(dedicated_outer_schema(name)).iter_errors(normalized), None) is None:
                value = normalized
                if audit is not None:
                    audit["validation_scope"] = "container_here_each_row_and_provenance_in_handler"
            else:
                value = original(instance, normalized, *args, **kwargs)
        else:
            value = original(instance, normalized, *args, **kwargs)
        if changes and _hash(value) != normalized_sha256:
            raise ValueError("Native SDK changed decoded container arguments")
        if audit is not None:
            audit["native_validation_passed"] = True
        return value

    tool_class._normalize_and_validate_params = normalize_and_validate
    tool_class._cityagent_normalization_schema_sha256 = _hash(schema)


def _argument_schema_diagnostics(arguments, schema):
    """Content-free repair locations, not suggested report facts/defaults."""
    from jsonschema import Draft202012Validator
    normalized, changes, issues = _normalize_container_arguments(arguments, schema)

    def narrow(spec, value):
        if type(spec) is not dict:
            return spec
        branches = spec.get("anyOf", spec.get("oneOf", []))
        if type(value) is dict and branches:
            matching = [branch for branch in branches if any(
                key in value and value[key] in field.get("enum", [])
                for key, field in branch.get("properties", {}).items())]
            if len(matching) == 1:
                return narrow(matching[0], value)
        result = dict(spec)
        if type(value) is dict and "properties" in spec:
            result["properties"] = {key: narrow(field, value.get(key)) for key, field in spec["properties"].items()}
        if type(value) is list and type(spec.get("items")) is dict:
            result.pop("items")
            result["prefixItems"] = [narrow(spec["items"], item) for item in value]
            result["items"] = spec["items"]
        return result

    def path(parts):
        return "".join(f"[{part}]" if type(part) is int else ("." if index else "") + str(part)
                       for index, part in enumerate(parts))[:192] or "$"

    def shape(spec):
        kind = spec.get("type")
        if type(spec.get("enum")) is list and spec["enum"]:
            return spec["enum"][0]
        if kind == "array":
            return []
        if kind == "object":
            return {}
        if kind == "integer":
            return spec.get("minimum", 0)
        if kind == "string":
            return "<value from source>"
        return None

    diagnostics = []
    for error in Draft202012Validator(narrow(schema, normalized)).iter_errors(normalized):
        spec, actual, location = error.schema, error.instance, list(error.absolute_path)
        if error.validator == "required" and type(actual) is dict:
            missing = [key for key in spec.get("required", []) if key not in actual]
            for field in missing:
                field_spec = spec.get("properties", {}).get(field, {})
                diagnostics.append({"path": path([*location, field]), "actual_type": "missing",
                                    "expected_type": field_spec.get("type", "schema"),
                                    "reason": "required_field_missing",
                                    "minimal_shape_example": {field: shape(field_spec)}})
        else:
            example = [] if spec.get("type") == "array" else {} if spec.get("type") == "object" else None
            detail = {"path": path(location), "actual_type": {dict: "object", list: "array", str: "string",
                       bool: "boolean", int: "integer", float: "number", type(None): "null"}.get(type(actual), "invalid"),
                      "expected_type": spec.get("type", "schema"), "reason": str(error.validator)}
            if example is not None:
                detail["minimal_shape_example"] = example
            diagnostics.append(detail)
        if len(diagnostics) >= 4:
            break
    return {"issues": (issues + diagnostics)[:4], "conversions": changes,
            "examples_are_shapes_not_defaults": True,
            "action": "Repair only the listed argument paths; retain all other fields. Supply required business values from evidence, never from these structural examples."}


def _summary(value: Any, *, depth: int = 0) -> Any:
    """Keep diagnostic shape without putting report text in audit receipts."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return {"type": "str", "chars": len(value),
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}
    if depth >= 6:
        return {"type": type(value).__name__, "sha256": _hash(value)}
    if isinstance(value, dict):
        items = list(value.items())
        return {"type": "dict", "size": len(items), "sha256": _hash(value),
                "fields": {str(k): _summary(v, depth=depth + 1) for k, v in items[:40]}}
    if isinstance(value, (list, tuple)):
        return {"type": "list", "size": len(value), "sha256": _hash(value),
                "items": [_summary(v, depth=depth + 1) for v in value[:8]]}
    # Do not repr SDK objects: their repr may include an entire prompt.
    return {"type": type(value).__name__}


def _argument_receipt(args: dict[str, Any]) -> dict[str, Any]:
    """IDs and short search terms are useful, arbitrary long strings are not."""
    def bounded(value: Any) -> Any:
        if isinstance(value, str):
            return value if len(value) <= 128 else _summary(value)
        if isinstance(value, list):
            return [bounded(item) for item in value] if len(value) <= 32 else _summary(value)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return _summary(value)
    # Document workspace patches and image questions can contain report text
    # even when short. Preserve their identity/shape, not their contents.
    return {key: {"type": "dict", "size": len(value), "sha256": _hash(value)} if key in {"update", "proposal"} and type(value) is dict
            else _summary(value) if key in {"update_json", "region_json", "question", "update", "proposal"}
            else bounded(value) for key, value in args.items()}


def _dump_event(event: Any) -> dict[str, Any]:
    dump = getattr(event, "model_dump", None)
    if callable(dump):
        try:
            value = dump(mode="json")
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return {name: getattr(event, name) for name in (
        "content", "name", "tool_name", "args", "arguments", "result",
        "status", "is_final_response", "response_meta", "code", "message", "tool_call_id",
    ) if hasattr(event, name) and not callable(getattr(event, name))}


def _event_record(event: Any, dumped: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": type(event).__name__,
        "is_final_response": dumped.get("is_final_response") is True,
        "summary": _summary(dumped),
    }


def _sdk_tool_error(event: Any, dumped: dict[str, Any], enabled_tools: set[str]) -> dict[str, Any] | None:
    """SDK parse failures happen before any CityAgent handler is invoked.

    Only known event classes, enabled tool names and fixed repair language are
    exposed. Arbitrary SDK errors/parameters stay hashed in the event receipt.
    """
    event_type = type(event).__name__
    if event_type not in {"ToolErrorAction", "ToolJsonSchemaErrorAction"}:
        return None
    tool_name = dumped.get("tool_name")
    tool_name = tool_name if type(tool_name) is str and tool_name in enabled_tools & _ALLOWED_TOOLS else None
    owned_parse = []
    session = _ACTIVE_SESSION
    if event_type == "ToolErrorAction" and tool_name and session is not None:
        try:
            origin = _current_agent_call_id(session)
        except AgentCallAssociationError:
            # Keep the SDK failure even when its trusted identity provider
            # fails. The invalid session terminates after recording this error;
            # no invented generation association is added to the receipt.
            origin = None
        owned_parse = [row for row in session.get("argument_parse_errors", [])
                       if origin and row.get("agent_call_id") == origin and row.get("tool_name") == tool_name
                       and dumped.get("message") == _STRICT_ARGUMENT_JSON_MARKER + _hash(row)
                       and all(_hash(dumped[field]) == row["arguments_text_sha256"]
                               for field in ("arguments", "params", "parameters") if type(dumped.get(field)) is str)]
    if len(owned_parse) == 1 or dumped.get("message") == "Parameters must be formatted as a valid JSON!":
        code = "SDK_TOOL_ARGUMENT_JSON"
        action = ("Regenerate valid tool-argument JSON. For update_workspace, pass update as an object, not an escaped JSON string; submit a small update."
                  if tool_name == "update_workspace" else "Regenerate only this tool call with its declared parameter names and types. Close every parameter, function and tool-call marker; do not execute or copy incomplete arguments.")
    elif event_type == "ToolJsonSchemaErrorAction":
        code = "SDK_TOOL_ARGUMENT_SCHEMA"
        action = "Check the enabled tool schema and regenerate arguments with the required names and types."
    else:
        code = "SDK_TOOL_ERROR"
        action = "Inspect the enabled tool schema and saved workspace before retrying; do not assume the failed call changed state."
    correlation = {}
    if type(dumped.get("tool_call_id")) is str and dumped["tool_call_id"]:
        correlation["tool_call_id_sha256"] = _hash(dumped["tool_call_id"])
    # Some SDK event versions expose only the raw parameter text. Retain its
    # identity, never its contents, for exact association with a native reply.
    for field in ("arguments", "params", "parameters"):
        if type(dumped.get(field)) is str:
            correlation["arguments_text_sha256"] = _hash(dumped[field])
            break
    if len(owned_parse) == 1:
        record = owned_parse[0]
        correlation.update(adapter_parse_error_sha256=_hash(record),
                           arguments_text_sha256=record["arguments_text_sha256"])
    return {"type": event_type, "tool_name": tool_name, "code": code, "action": action,
            "event_sha256": _hash(dumped), **correlation}


def _error(kind: str, message: str) -> dict[str, str]:
    # Configuration/SDK errors have static messages. Tool exceptions use the
    # separate allowlisted repair mapper below, never their original message.
    return {"type": kind, "message": message}


_ERROR_FEEDBACK_BYTES = 1536
_SOURCE_RANGE_ACTION = (
    "start/end are 0-based half-open character offsets within that evidence_id's own text unit, "
    "not whole-document offsets. Read the source/section tools to obtain its actual unit length, "
    "then use integer 0 <= start < end <= that length. Do not guess, clamp offsets or invent text."
)
_WORKSPACE_REPAIRS = (
    (("A fact source reference must cover nonempty text",), ("Invalid interval for ",),
     "SOURCE_RANGE", ["source_refs", "start", "end"],
     _SOURCE_RANGE_ACTION),
    ((), ("Source interval was not actually read:",),
     "SOURCE_NOT_READ", ["source_refs"],
     "Call read_source for the referenced evidence_id and continue until the needed text is delivered, then correct and resubmit the affected item."),
    ((), ("Image source was not actually read:",),
     "IMAGE_NOT_READ", ["source_refs", "image_id"],
     "Call read_image first. If it is unavailable, record the limitation; do not claim image understanding."),
    (("source_refs must be a list", "Each source reference must be an object",
      "Text references require evidence_id, start and end", "An image reference cannot also contain a text interval",
      "Reference source_path differs from its manifest source"), ("Unknown evidence ID:",),
     "SOURCE_REFERENCE", ["source_refs"],
     "Read the outline/source to obtain valid IDs. Text refs need evidence_id/start/end; image refs need image_id. Do not invent paths."),
    (("Facts require value, source_refs and status", "Invalid evidence record status",
      "Supported claims need read source_refs or evidence dependencies"), (),
     "FACT_FIELDS", ["value", "source_refs", "status", "depends_on"],
     "Provide read evidence or valid dependencies; status must be explicit, inferred, uncertain or conflict. Do not invent support."),
    (("Question status must be open, resolved, unknown or conflict", "Resolved/unknown questions require a reason"), (),
     "QUESTION_FIELDS", ["status", "reason"],
     "Use open/resolved/unknown/conflict; resolved and unknown require a nonempty reason."),
    (("Record validity/version metadata is managed by the workspace",), (),
     "MANAGED_FIELDS", ["valid", "record_version", "invalidated_by"],
     "Remove workspace-managed validity/version fields from the update; submit only editable record fields."),
    (("Draft refers to missing or invalid facts/relations", "Draft patch requires active fact/relation IDs"),
     ("Cyclic dependency involving ",),
     "DEPENDENCY_INVALID", ["depends_on", "subject_id", "object_id", "draft_fact_ids"],
     "Page through facts/relations, repair invalid or missing dependencies and cycles, then bind only active records."),
    (("A draft update requires draft, draft_fact_ids and draft_bindings together", "draft_bindings must be an object",
      "Every nonempty output field/row needs its own draft binding; unknown paths are not allowed",
      "Every draft binding needs active IDs included in draft_fact_ids"), (),
     "DRAFT_BINDINGS", ["draft", "draft_fact_ids", "draft_bindings"],
     "Submit draft/draft_fact_ids/draft_bindings together, binding each nonempty basic field and each detail row to active IDs."),
    (("draft must contain exactly 基本信息, 维修建议 and 病害明细",
      "Draft basic information is an object and both detail sections are lists of objects"), (),
     "DRAFT_SHAPE", ["draft"],
     "Use exactly 基本信息 (object), 维修建议 and 病害明细 (lists of objects), with only the permitted output columns."),
    (("draft_patch must be a nonempty list of operations", "Each draft patch operation needs an op",
      "set_basic requires a nonempty field name", "Row operations require section 维修建议 or 病害明细",
      "Draft row must be an object", "draft_patch is mutually exclusive with explicit draft fields"),
     ("Invalid draft patch operation or fields:", "Invalid draft row index for "),
     "DRAFT_PATCH", ["draft_patch"],
     "Use set_basic/append_row/replace_row/remove_row with their defined fields; page the current draft before using row indices. Do not mix draft_patch and full draft."),
    (("update must be an object",), ("Unknown update fields:",),
     "UPDATE_FIELDS", ["base_version", "facts", "relations", "questions", "draft_patch"],
     "Submit a JSON object using the documented workspace fields and the latest base_version from read_workspace."),
)


def _repair(kind: str, code: str, action: str, *, fields=(), recoverable=True, details=None) -> dict[str, Any]:
    error = {"type": kind, "code": code, "action": action, "recoverable": recoverable}
    if fields:
        error["fields"] = list(fields)
    if details:
        error["details"] = details
    return error


def _tool_error(exc: Exception, name: str, stage: str) -> dict[str, Any]:
    """Map exact owned exception types to static repairs; never echo str(exc).

    Generic messages select an allowlisted repair, without echoing variables.
    The exact WorkspaceRepairError type can additionally expose bounded
    structural locations/IDs constructed at the atomic validation boundary;
    it never exposes draft prose or source text. Unknown exception types and
    arbitrary attributes still receive only the safe static fallback.
    """
    from .document_workspace import WorkspaceConflictError, WorkspaceError, WorkspaceIdentityError, WorkspaceRepairError
    from .document_review_payloads import ReviewPayloadBudgetError

    kind = type(exc)
    if stage == "arguments":
        if name == "update_workspace":
            return _repair("ValueError", "UPDATE_OBJECT_ARGUMENTS",
                           "Pass update as a JSON object, not an escaped JSON string. Use string keys and finite JSON values only.", fields=["update"])
        if name == "propose_source_object":
            return _repair("ValueError", "SOURCE_OBJECT_ARGUMENTS",
                           "Pass object_id and a proposal object with exactly disposition/rationale/affected_output_paths/supporting_refs/context_refs/source_limitations. Do not send origin_call_ids, accepted or representation flags. Use read source intervals; independent review decides acceptance.",
                           fields=["object_id", "proposal"])
        if name == "read_workspace_page":
            return _repair("ValueError", "TOOL_ARGUMENTS",
                           "Use a counts-key view, optional positive integer limit and boolean restart. The program resumes saved positions and detects version changes; use restart=true to reread from the beginning.",
                           fields=["view", "limit", "restart"])
        return _repair("ValueError", "TOOL_ARGUMENTS",
                       "Use the tool schema: nonempty string IDs, string JSON, arrays of strings, and integer pagination bounds (not booleans).")
    if stage == "authorization":
        return _repair("PermissionError", "TOOL_NOT_ENABLED", "Choose a tool enabled in the current session.", recoverable=False)
    if stage == "call_budget":
        return _repair("RuntimeError", "TOOL_CALL_BUDGET", "Stop tool calls and report the unfinished work.", recoverable=False)
    if stage == "agent_call_association":
        return _repair("AgentCallAssociationError", "AGENT_CALL_ID_INVALID",
                       "Stop and request verification of the local model-call receipt association; do not supply or guess a call ID.", recoverable=False)
    if kind is WorkspaceConflictError:
        return _repair("WorkspaceConflictError", "WORKSPACE_STALE",
                       "Call read_workspace again, rebase the update on its current version and recheck affected facts/draft before retrying.",
                       fields=["base_version", "draft_sha256"])
    if kind is WorkspaceIdentityError:
        return _repair("WorkspaceIdentityError", "WORKSPACE_IDENTITY",
                       "Stop and request checkpoint/input identity verification. Do not overwrite or reset this workspace.", recoverable=False)
    if kind is WorkspaceRepairError and exc.code in {"DRAFT_BINDINGS", "DRAFT_SHAPE", "DRAFT_PATCH", "DEPENDENCY_INVALID", "FACT_FIELDS", "SOURCE_MATERIAL", "SOURCE_ALIAS_RANGE"}:
        # This exact owned type is created from validated structural state.
        # It never carries source text/draft values or accepts arbitrary attrs.
        fields = (["draft", "draft_fact_ids", "draft_bindings"] if exc.code == "DRAFT_BINDINGS"
                  else ["draft_patch"] if exc.code == "DRAFT_PATCH"
                  else ["value", "source_refs", "status"] if exc.code == "FACT_FIELDS"
                  else ["source_refs", "evidence_id"] if exc.code in {"SOURCE_MATERIAL", "SOURCE_ALIAS_RANGE"}
                  else ["draft", "draft_patch"])
        action = ("Entire update was NOT saved, including new facts and earlier operations. "
                  "Use current_version, repair the listed paths/IDs/operation, then submit a small atomic update. "
                  "Prefer draft_patch; bind_output(path,fact_ids) changes only a saved item's bindings after you verify its evidence. "
                  "If this was the first draft, use set_basic/append_row with facts; do not assume the rejected draft exists.")
        if exc.code == "FACT_FIELDS":
            action = ("Entire update was NOT saved. New facts require value/source_refs/status; "
                      "status must be explicit, inferred, uncertain or conflict. "
                      "Use current_version and supply the listed missing fields with read evidence or valid dependencies; "
                      "do not invent support. Existing facts may update only changed fields.")
        elif exc.code in {"SOURCE_MATERIAL", "SOURCE_ALIAS_RANGE"}:
            action = ("Entire update was NOT saved. Repair the indicated source_refs path using an exact "
                      "source ID returned by reading tools for this document. Material aliases take evidence_id only; "
                      "the host maps them to canonical ranges. For ambiguous/unmapped material references, "
                      "read the indicated original E evidence and cite it. Never guess offsets or another document's ID.")
        details = json.loads(_json_bytes(exc.details))
        result = _repair("WorkspaceRepairError", exc.code, action, fields=fields, details=details)
        # Unicode paths may be large. Remove complete last list elements while
        # retaining counts and the explicit atomic-failure/current-version facts.
        while len(_json_bytes(result)) > _ERROR_FEEDBACK_BYTES:
            lists = [key for key, value in details.items() if type(value) is list and value]
            if not lists:
                break
            key = max(lists, key=lambda item: len(_json_bytes(details[item])))
            details[key].pop()
            details[key + "_omitted"] = details.get(key + "_omitted", 0) + 1
        return result
    message = exc.args[0] if len(exc.args) == 1 and type(exc.args[0]) is str and len(exc.args[0]) <= 8192 else ""
    if kind is WorkspaceError:
        for exact, prefixes, code, fields, action in _WORKSPACE_REPAIRS:
            if message in exact or (prefixes and message.startswith(prefixes)):
                return _repair("WorkspaceError", code, action, fields=fields)
        if " depends on missing/invalid record " in message:
            return _repair("WorkspaceError", "DEPENDENCY_INVALID",
                           "Page through facts/relations; repair the missing or invalid dependencies before resubmitting.",
                           fields=["depends_on", "subject_id", "object_id"])
        return _repair("WorkspaceError", "WORKSPACE_INVALID",
                       "Read_workspace and page the affected records; correct the update to the documented workspace schema before retrying.")
    if kind is ReviewPayloadBudgetError:
        # Numeric diagnostics are useful, but the source/path arrays may be
        # unbounded or contain document text. Never copy those arrays.
        details = {key: value for key in ("prompt_tokens", "max_model_len", "output_tokens", "minimum_protocol_output_tokens")
                   if type(value := exc.details.get(key)) is int and 0 <= value <= 2**31 - 1} if type(exc.details) is dict else {}
        return _repair("ReviewPayloadBudgetError", "REVIEW_CONTEXT_BUDGET",
                       "Inspect draft bindings and narrow only overly broad source_refs while retaining all necessary context. If the complete evidence still cannot fit, stop for grouped review or a budget change; do not drop evidence or retry unchanged.",
                       fields=["draft_bindings", "source_refs"], details=details)
    if kind is json.JSONDecodeError:
        return _repair("JSONDecodeError", "INVALID_JSON", "Resubmit valid JSON text with quoted keys and no trailing commas.")
    if kind is ValueError and name == "read_workspace_page" and message == "Workspace version changed; discard collected fragments and restart the view":
        return _repair("ValueError", "WORKSPACE_PAGE_STALE",
                       "Discard fragments collected for the previous workspace version. Repeat read_workspace_page with the same view and restart=true; the program selects the current version and saved position. Never combine fragments from different versions.",
                       fields=["view", "limit", "restart"])
    if kind is ValueError and name == "read_workspace_page" and message.startswith((
            "Fragment continuation requires", "item_start exceeds", "item_start must", "expected_version must")):
        return _repair("ValueError", "WORKSPACE_FRAGMENT_RANGE",
                       "Repeat read_workspace_page with the same view and restart=true; the program restores the current record and fragment position. If the failure persists, report it instead of inventing positions. Fragments are incomplete JSON text, not evidence or complete notes.",
                       fields=["view", "limit", "restart"])
    if kind is ValueError and name == "read_workspace_page" and (
            message == "Unknown workspace view; use a summary counts key"
            or message.startswith("Unknown workspace view: ")):
        return _repair("ValueError", "WORKSPACE_VIEW",
                       "Call read_workspace and choose a view from its counts keys. For draft rows use draft_basic, draft_advice or draft_defects. Choose only the view and optional limit/restart; the program owns version and paging.",
                       fields=["view", "limit", "restart"])
    if kind is ValueError and name in {"read_source", "read_section", "read_table_rows", "get_neighbors"} and message.startswith("Invalid character interval for "):
        identifier = {"read_section": "section_id", "read_table_rows": "table_id"}.get(name, "evidence_id")
        return _repair("ValueError", "SOURCE_RANGE",
                       "Select the source object using the ordinary reading tool. The program owns its position; use restart=true to reread, or read_source for a neighbor evidence_id. If the same object still fails, report the blocker instead of guessing offsets.",
                       fields=[identifier])
    if kind is ValueError and name == "review_draft" and message in {
            "Semantic review payload exceeds context; no evidence was truncated",
            "One draft item and its source references exceed review context; narrow its evidence spans"}:
        return _repair("ValueError", "REVIEW_CONTEXT_BUDGET",
                       "Inspect draft bindings; narrow only overly broad evidence spans. Keep necessary context; if it still cannot fit, stop for a budget/protocol change.")
    if kind is ValueError and name == "update_workspace" and message in {
            "Delivery has unsupported sections", "Delivery has unsupported basic columns", "Delivery has unsupported row columns"}:
        return _repair("ValueError", "DRAFT_SHAPE", "Use only the three documented sections and their permitted output columns.", fields=["draft"])
    safe_types = {ValueError: "ValueError", TypeError: "TypeError", KeyError: "KeyError", RuntimeError: "RuntimeError",
                  PermissionError: "PermissionError", TimeoutError: "TimeoutError", ConnectionError: "ConnectionError"}
    return _repair(safe_types.get(kind, "ToolError"), "TOOL_FAILED",
                   "Check arguments and authorized evidence IDs. If the failure persists, stop and report it; do not invent evidence.")


def _tool_budget_error() -> dict[str, Any]:
    return {"ok": False, "error": _repair("ToolResponseBudgetExceeded", "TOOL_RESPONSE_BUDGET",
            "Reduce page/span size; read_workspace before repeating an update. If one item cannot fit, stop for a larger budget. No content was truncated.")}


def _response_tokens(counter: Callable[[str], int], response: dict) -> int:
    count = counter(_json_bytes(response).decode("utf-8"))
    if type(count) is not int or count < 0:
        raise ValueError("count_tokens must return a non-negative integer")
    return count


def wrap_tool_response(name: str, payload: Any) -> dict[str, Any]:
    """The native handler envelope, also used for exact tool-page admission.

    This pure helper does not import or execute the SDK. The caller owns the
    payload snapshot; counting and dispatch must serialize the same envelope.
    """
    failed = isinstance(payload, dict) and (
        payload.get("ok") is False or bool(payload.get("error"))
    )
    response = {"ok": not failed, "tool": name, "result": payload}
    if failed:
        # Local handlers can return an error instead of raising it. Keep their
        # complete payload as data, but carry only fixed, allowlisted repair
        # language into the adapter receipt and future SDK sessions. This same
        # pure wrapper is counted by native tools before granting any reads.
        response["error"] = _returned_tool_error(name, payload)
    return response


def _returned_tool_error(name: str, payload: dict) -> dict[str, Any]:
    from .document_workspace import WorkspaceConflictError, WorkspaceError, WorkspaceIdentityError

    error = payload.get("error")
    kind = error.get("type") if isinstance(error, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    classes = {"ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
               "RuntimeError": RuntimeError, "WorkspaceError": WorkspaceError,
               "WorkspaceConflictError": WorkspaceConflictError, "WorkspaceIdentityError": WorkspaceIdentityError}
    constructor = classes.get(kind) if type(kind) is str else None
    if constructor is not None and type(message) is str and len(message) <= 8192:
        repair = _tool_error(constructor(message), name, "handler")
    else:
        repair = _repair("ToolError", "TOOL_FAILED",
                         "Read the tool schema and current source/workspace state; correct the request. If failure persists, report it without inventing evidence.")
    if isinstance(error, dict) and error.get("recoverable") is False:
        repair["recoverable"] = False
    # Never copy a handler-supplied code/action/details, even when plausible.
    # Unknown protocols remain failures with a bounded static recovery action.
    if len(_json_bytes(repair)) > _ERROR_FEEDBACK_BYTES:
        raise ValueError("Static returned-tool repair exceeds its bound")
    return repair


def _bounded_tool_response(session: dict, response: dict) -> dict:
    counter = session.get("count_tokens")
    if counter is None:
        return response  # Legacy handlers retain their existing result contract.
    if _response_tokens(counter, response) <= session["max_tool_tokens"]:
        return response
    fallback = _tool_budget_error()
    if _response_tokens(counter, fallback) > session["max_tool_tokens"]:
        # Never hand an unchecked or over-budget response to the model if the
        # tokenizer changes/fails after configuration validation.
        raise ValueError("Tool budget cannot fit its minimal error response")
    return fallback


def _validate_json_value(value: Any, active: set[int] | None = None) -> None:
    """Reject Python-only values and lossy key coercion before serialization."""
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if type(value) not in {dict, list}:
        raise ValueError("Unsupported JSON value type")
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError("Circular JSON value")
    active.add(id(value))
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("JSON object keys must be strings")
            children = value.values()
        else:
            children = value
        for child in children:
            _validate_json_value(child, active)
    finally:
        active.remove(id(value))


def _validate_arguments(name: str, args: dict[str, Any]) -> None:
    if name in _SECTION_TOOLS:
        _validate_section_arguments(name, args)
        return
    if name in {"review_candidates", "submit_review", "quick_review", *MAIN_REVIEW_SCHEMAS}:
        _validate_json_value(args)
        from jsonschema import Draft202012Validator
        if next(Draft202012Validator(_normalization_schema_for_tool(name)).iter_errors(args), None) is not None:
            raise ValueError("Candidate review arguments do not match the registered tool schema")
        return
    if name in _REVIEW_TOOLS:
        schema = _REVIEW_SCHEMAS[name]["parameters"]
        if type(args) is not dict or set(args) != set(schema["required"]):
            raise ValueError("Review arguments must match the explicit readonly schema")
        for key, spec in schema["properties"].items():
            value = args[key]
            expected = int if spec["type"] == "integer" else str
            if type(value) is not expected:
                raise ValueError("Review argument has invalid type")
            if expected is int and (value < spec.get("minimum", value) or value > spec.get("maximum", value)):
                raise ValueError("Review integer outside schema bounds")
            if "enum" in spec and value not in spec["enum"]:
                raise ValueError("Review argument outside schema enum")
        return
    if name == "propose_source_object":
        if (set(args) != {"object_id", "proposal"} or type(args["object_id"]) is not str
                or not args["object_id"].strip() or type(args["proposal"]) is not dict):
            raise ValueError("Require object_id and a native proposal JSON object")
        proposal = args["proposal"]
        _validate_json_value(proposal)
        if (set(proposal) != _SOURCE_PROPOSAL_FIELDS or proposal["disposition"] not in _SOURCE_DISPOSITIONS
                or type(proposal["rationale"]) is not str or not proposal["rationale"].strip()):
            raise ValueError("Invalid source-object proposal fields or disposition")
        for key in ("affected_output_paths", "source_limitations"):
            values = proposal[key]
            if (type(values) is not list or any(type(v) is not str or not v.strip() for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError("Proposal paths and limitations must be distinct nonempty strings")
        for key in ("supporting_refs", "context_refs"):
            refs = proposal[key]
            if type(refs) is not list:
                raise ValueError("Proposal source references must be arrays")
            for ref in refs:
                if (type(ref) is not dict or set(ref) != {"evidence_id", "start", "end"}
                        or type(ref["evidence_id"]) is not str or not ref["evidence_id"].strip()
                        or type(ref["start"]) is not int or type(ref["end"]) is not int
                        or not 0 <= ref["start"] < ref["end"]):
                    raise ValueError("Proposal references require evidence_id and integer 0 <= start < end")
        return
    if name == "update_workspace":
        if set(args) != {"update"} or type(args["update"]) is not dict:
            raise ValueError("update must be a JSON object")
        _validate_json_value(args["update"])
        return
    if name == "finish":
        if not args:
            return  # Legacy workflow keeps native finish({}).
        if set(args) != {"self_check"} or type(args["self_check"]) is not dict:
            raise ValueError("finish requires one native self_check object")
        _validate_json_value(args["self_check"])
        return
    if name == "inspect_delivery":
        from jsonschema import Draft202012Validator
        _validate_json_value(args)
        if (next(Draft202012Validator(_delivery_schema(name)).iter_errors(args), None) is not None
                or type(args.get("limit", 4)) is not int):
            raise ValueError("inspect_delivery requires view, optional positive limit and boolean restart")
        return
    if name == "read_workspace_page":
        if (set(args) - {"view", "limit", "restart"} or type(args.get("view")) is not str
                or not args["view"].strip() or type(args.get("limit", 8)) is not int
                or args.get("limit", 8) < 1 or type(args.get("restart", False)) is not bool):
            raise ValueError("read_workspace_page requires view, optional positive limit and boolean restart")
        return
    integer_keys: set[str] = set()
    if name == "read_source_objects":
        cursor, limit = args.get("cursor"), args.get("limit")
        if type(cursor) is not int or type(limit) is not int or cursor < 0 or limit < 1:
            raise ValueError(f"{name} requires integer cursor >= 0 and limit > 0")
        integer_keys = {"cursor", "limit"}
    for key, value in args.items():
        if key in integer_keys:
            continue
        elif key in {"evidence_ids", "extra_evidence_ids", "context_refs"}:
            if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
                raise ValueError(f"{key} must be a list of non-empty strings")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}.{key} must be a non-empty string")


def _invoke(name: str, args: dict[str, Any]) -> dict[str, Any]:
    session = _ACTIVE_SESSION
    if session is None:
        return {"ok": False, "error": _error("NoActiveSession", "No active CityAgent tool session")}
    # SDK worker threads may execute tool callbacks.  They must not acquire
    # the outer session lock (held while step() waits for them).  A separate
    # lock serializes budget accounting and caller-owned mutable task state.
    with session["tool_lock"]:
        if session is not _ACTIVE_SESSION:
            return {"ok": False, "error": _error("ExpiredSession", "The tool session has ended")}
        previous = session.get("active_tool_identity")
        try:
            return _invoke_active(session, name, args)
        finally:
            session["active_tool_identity"] = previous


def _current_agent_call_id(session: dict[str, Any]) -> str | None:
    """Snapshot only caller-owned Python state, never tool/event arguments.

    None means there is no verified association. The callback stays in memory
    and is never passed to the SDK, a tool schema, or a serialized receipt.
    """
    if "current_agent_call_id" not in session:
        return None
    try:
        callback = session["current_agent_call_id"]
        if not callable(callback):
            raise TypeError("invalid trusted callback")
        call_id = callback()
        if call_id is not None and (type(call_id) is not str or not call_id.strip() or len(call_id) > 256):
            raise ValueError("invalid trusted call ID")
    except Exception:
        session["agent_call_id_invalid"] = True
        raise AgentCallAssociationError("Trusted model-call association failed") from None
    return call_id


def current_tool_call_identity() -> dict[str, str] | None:
    """Host-only identity of the currently executing, exactly paired SDK call.

    This is never a model argument. Legacy direct handlers have no such proof
    and return None. Callers receive a detached copy, not mutable session state.
    """
    session = _ACTIVE_SESSION
    if session is None:
        return None
    with session["tool_lock"]:
        value = session.get("active_tool_identity")
        return dict(value) if value is not None else None


def _bound_dispatch_arguments(name: str, original: dict) -> dict:
    """Mirror the typed wrapper's mechanical defaults for receipt matching.

    These two wrappers omit an *absent* optional None object.  Signature
    binding alone adds that object back, making an unchanged native call look
    unrelated to its actual tool return. Explicit nulls and model-supplied
    values stay intact; no semantic review or self-check is synthesized.
    """
    bound = inspect.signature(_TOOLS[name][0]).bind(**original)
    bound.apply_defaults()
    omitted = {"complete_review": "gap_checks", "finish": "self_check"}.get(name)
    if omitted is not None and omitted not in original and bound.arguments.get(omitted) is None:
        bound.arguments.pop(omitted, None)
    return dict(bound.arguments)


def _dispatch_identity(session: dict[str, Any], name: str, args: dict, origin: str | None):
    frame = session.get("native_dispatch")
    if frame is None or origin is None:
        return None
    if frame["generation_call_id"] != origin or frame["tool_name"] != name:
        raise AgentCallAssociationError("Native tool dispatch differs from its generating response")
    original = _strict_json_loads(frame["arguments"])
    if type(original) is not dict:
        raise AgentCallAssociationError("Native tool arguments are not an object")
    schema = _normalization_schema_for_tool(name)
    if schema is not None:
        original, _changes, _issues = _normalize_container_arguments(original, schema)
    if _hash(_bound_dispatch_arguments(name, original)) != _hash(args):
        raise AgentCallAssociationError("Native tool arguments differ from actual dispatch")
    return {"run_id": session["run_id"], "session_id": session["session_id"],
            "generation_call_id": origin, "native_tool_call_id": frame["native_tool_call_id"],
            "tool_name": name, "args_sha256": _hash(args)}


def _invoke_active(session: dict[str, Any], name: str, args: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    original_args = None
    handler_result_sha256 = None
    agent_call_id = None
    status = "failed"
    stage = "authorization"
    try:
        if name not in session["handlers"]:
            raise PermissionError("tool is not enabled for this session")
        stage = "call_budget"
        if len(session["tool_calls"]) >= session["max_calls"]:
            session["tool_budget_exceeded"] = True
            raise RuntimeError("tool call budget exceeded")
        stage = "agent_call_association"
        agent_call_id = _current_agent_call_id(session)
        stage = "arguments"
        if session.get("profile") == "section" and name in _SECTION_SOURCE_TOOLS:
            _validate_section_arguments(name, args)
        _validate_arguments(name, args)
        original_args = json.loads(_json_bytes(args))
        stage = "agent_call_association"
        session["active_tool_identity"] = _dispatch_identity(session, name, original_args, agent_call_id)
        # The SDK sees one object layer. Existing workflow handlers keep their
        # update_json contract; only trusted Python performs the serialization.
        handler_args = ({"update_json": _json_bytes(original_args["update"]).decode("utf-8")}
                        if name == "update_workspace" else args)
        stage = "handler"
        payload = session["handlers"][name](**handler_args)
        # A handler result must be serializable: an opaque Python object is
        # not usable evidence.  Also take a snapshot before a caller mutates it.
        payload = json.loads(_json_bytes(payload))
        response = wrap_tool_response(name, payload)
        status = "success" if response["ok"] else "failed"
        if not response["ok"]:
            handler_result_sha256 = _hash(payload)
    except Exception as exc:
        error = _tool_error(exc, name, stage)
        if len(_json_bytes(error)) > _ERROR_FEEDBACK_BYTES:
            error = _repair("ToolError", "TOOL_FAILED", "Read the tool schema and correct the input; report persistent failure.")
        response = {"ok": False, "tool": name, "error": error}
    response = _bounded_tool_response(session, response)
    if not response["ok"]:
        status = "failed"
    session["tool_calls"].append({
        "name": name, "args": _argument_receipt(original_args) if original_args is not None else {"invalid_or_unprocessed": True},
        "args_sha256": _hash(original_args) if original_args is not None else None,
        "status": status, "result_sha256": _hash(response),
        "result_summary": _summary(response), "seconds": round(time.monotonic() - started, 6),
        **({"agent_call_id": agent_call_id} if agent_call_id is not None else {}),
        **({"native_tool_call_id": session["active_tool_identity"]["native_tool_call_id"]}
           if session.get("active_tool_identity") is not None else {}),
        **({"handler_result_sha256": handler_result_sha256} if handler_result_sha256 is not None else {}),
        **({"error": response["error"]} if "error" in response else {}),
    })
    # The SDK can execute an earlier handler, then reject a later call in the
    # same native batch without persisting any of that batch's tool results.
    # Keep the actual final return privately; the public summary is lossy and
    # must never be used to manufacture a missing result. Observers receive
    # independent copies and must finish before a return is recoverable.
    observer = session.get("on_tool_result")
    if observer is not None:
        observer(name, json.loads(_json_bytes(original_args or {})), json.loads(_json_bytes(response)))
    snapshots = session.get("tool_response_snapshots")
    if snapshots is not None:
        index = len(session["tool_calls"]) - 1
        row = session["tool_calls"][index]
        snapshots[index] = {key: row.get(key) for key in (
            "name", "agent_call_id", "args_sha256", "result_sha256", "status")}
        snapshots[index]["response"] = json.loads(_json_bytes(response))
    return response


# Explicit typed functions give the official SDK a stable JSON schema.  They
# deliberately close over no document, task, or handler instance.
def _read_outline(limit: int = 20, restart: bool = False) -> dict[str, Any]:
    return _invoke("read_outline", {"limit": limit, "restart": restart})


def _read_table_rows(table_id: str, max_rows: int = 8, restart: bool = False) -> dict[str, Any]:
    return _invoke("read_table_rows", {"table_id": table_id, "max_rows": max_rows, "restart": restart})


def _read_evidence(evidence_ids: list[str]) -> dict[str, Any]:
    return _invoke("read_evidence", {"evidence_ids": evidence_ids})


def _read_source(evidence_id: str, restart: bool = False) -> dict[str, Any]:
    return _invoke("read_source", {"evidence_id": evidence_id, "restart": restart})


def _read_source_range(evidence_id: str, start: int = 0, end: int | None = None) -> dict[str, Any]:
    """Read Unicode characters [start,end) within this evidence_id's text unit.

    Offsets are 0-based local characters, not whole-document or byte offsets.
    Omit end to read the actual complete unit; long units return continuation.
    """
    return _invoke("read_source_range", {"evidence_id": evidence_id, "start": start, "end": end})


def _read_section(section_id: str, max_units: int = 8, restart: bool = False) -> dict[str, Any]:
    return _invoke("read_section", {"section_id": section_id, "max_units": max_units, "restart": restart})


def _read_section_range(section_id: str, cursor: int = 0, max_units: int = 8) -> dict[str, Any]:
    return _invoke("read_section_range", {"section_id": section_id, "cursor": cursor, "max_units": max_units})


def _read_table_range(table_id: str, start: int, end: int) -> dict[str, Any]:
    return _invoke("read_table_range", {"table_id": table_id, "start": start, "end": end})


def _save_entries(entries: list[dict]) -> dict[str, Any]:
    return _invoke("save_entries", {"entries": entries})


def _read_entries(limit: int = 8, restart: bool = False) -> dict[str, Any]:
    return _invoke("read_entries", {"limit": limit, "restart": restart})


def _handoff_section(notes: str = "") -> dict[str, Any]:
    return _invoke("handoff_section", {"notes": notes})


def _complete_section(summary: str, unresolved: list[str]) -> dict[str, Any]:
    return _invoke("complete_section", {"summary": summary, "unresolved": unresolved})


def _save_basic_info(entries: list[dict]) -> dict[str, Any]:
    return _invoke("save_basic_info", {"entries": entries})


def _save_defects(entries: list[dict]) -> dict[str, Any]:
    return _invoke("save_defects", {"entries": entries})


def _save_recommendations(entries: list[dict]) -> dict[str, Any]:
    return _invoke("save_recommendations", {"entries": entries})


def _submit_section(entries: list[dict], unresolved: list[str], complete: bool = True) -> dict[str, Any]:
    return _invoke("submit_section", {"entries": entries, "unresolved": unresolved, "complete": complete})


def _review_candidates(operations: list[dict], task_reviews: list[dict],
                       issue_reviews: list[dict] | None = None) -> dict[str, Any]:
    return _invoke("review_candidates", {"operations": operations, "task_reviews": task_reviews,
                                         "issue_reviews": issue_reviews})


def _submit_review(selections: list[str] | None = None, merges: list[dict] | None = None,
                   patches: list[dict] | None = None, exclusions: list[dict] | None = None,
                   reviews: list[dict] | None = None) -> dict[str, Any]:
    return _invoke("submit_review", {"selections": selections, "merges": merges,
        "patches": patches, "exclusions": exclusions, "reviews": reviews})



def _quick_review(basis: str, remaining_issues: list[str], selections: list[str] | None = None,
                  merges: list[dict] | None = None, patches: list[dict] | None = None,
                  exclusions: list[dict] | None = None) -> dict[str, Any]:
    return _invoke("quick_review", dict(basis=basis, remaining_issues=remaining_issues,
        selections=selections, merges=merges, patches=patches, exclusions=exclusions))


def _select_basic_info(selections: dict) -> dict[str, Any]:
    return _invoke("select_basic_info", {"selections": selections})

def _merge_duplicate_entries(groups: list[dict]) -> dict[str, Any]:
    return _invoke("merge_duplicate_entries", {"groups": groups})

def _revise_basic_info(items: list[dict]) -> dict[str, Any]:
    return _invoke("revise_basic_info", {"items": items})

def _revise_defects(items: list[dict]) -> dict[str, Any]:
    return _invoke("revise_defects", {"items": items})

def _revise_recommendations(items: list[dict]) -> dict[str, Any]:
    return _invoke("revise_recommendations", {"items": items})

def _exclude_entries(items: list[dict]) -> dict[str, Any]:
    return _invoke("exclude_entries", {"items": items})

def _complete_review(basis: str, notes: str = "", gap_checks: dict | None = None) -> dict[str, Any]:
    payload = {"basis": basis, "notes": notes}
    if gap_checks is not None:
        payload["gap_checks"] = gap_checks
    return _invoke("complete_review", payload)


def _finish_report(basis: str, notes: str = "") -> dict[str, Any]:
    return _invoke("finish_report", {"basis": basis, "notes": notes})


def _read_image(image_id: str, question: str, region_json: str, context_refs: list[str]) -> dict[str, Any]:
    return _invoke("read_image", {"image_id": image_id, "question": question,
                                  "region_json": region_json, "context_refs": context_refs})


def _read_workspace() -> dict[str, Any]:
    return _invoke("read_workspace", {})


def _read_workspace_page(view: str, limit: int = 8, restart: bool = False) -> dict[str, Any]:
    """Continue this view at the program-saved position for its current version."""
    return _invoke("read_workspace_page", {"view": view, "limit": limit, "restart": restart})


def _read_source_objects(cursor: int = 0, limit: int = 10) -> dict[str, Any]:
    return _invoke("read_source_objects", {"cursor": cursor, "limit": limit})


def _read_source_object(object_id: str) -> dict[str, Any]:
    return _invoke("read_source_object", {"object_id": object_id})


def _propose_source_object(object_id: str, proposal: dict[str, Any]) -> dict[str, Any]:
    return _invoke("propose_source_object", {"object_id": object_id, "proposal": proposal})


def _update_workspace(update: dict[str, Any]) -> dict[str, Any]:
    return _invoke("update_workspace", {"update": update})


def _review_draft() -> dict[str, Any]:
    return _invoke("review_draft", {})


def _search_in_document(query: list[str], scope: str = "default", section_id: str | None = None,
                        max_matches: int = 8, restart: bool = False) -> dict[str, Any]:
    return _invoke("search_in_document", {"query": query, "scope": scope, "section_id": section_id,
                                          "max_matches": max_matches, "restart": restart})


def _get_neighbors(evidence_id: str) -> dict[str, Any]:
    return _invoke("get_neighbors", {"evidence_id": evidence_id})


def _extract_chunk(task_id: str, extra_evidence_ids: list[str]) -> dict[str, Any]:
    return _invoke("extract_chunk", {"task_id": task_id, "extra_evidence_ids": extra_evidence_ids})


def _review_conflict(conflict_id: str, extra_evidence_ids: list[str]) -> dict[str, Any]:
    return _invoke("review_conflict", {"conflict_id": conflict_id, "extra_evidence_ids": extra_evidence_ids})


def _split_task(task_id: str) -> dict[str, Any]:
    return _invoke("split_task", {"task_id": task_id})


def _finish(self_check: dict | None = None) -> dict[str, Any]:
    return _invoke("finish", {} if self_check is None else {"self_check": self_check})


def _inspect_delivery(view: str, limit: int = 4, restart: bool = False) -> dict[str, Any]:
    return _invoke("inspect_delivery", {"view": view, "limit": limit, "restart": restart})


def _review_read_draft_page(cursor: int, limit: int) -> dict[str, Any]:
    return _invoke("review_read_draft_page", {"cursor": cursor, "limit": limit})


def _review_search_draft(query: str, cursor: int, limit: int) -> dict[str, Any]:
    return _invoke("review_search_draft", {"query": query, "cursor": cursor, "limit": limit})


def _review_read_draft_item(path: str, view: str, cursor: int, limit: int) -> dict[str, Any]:
    return _invoke("review_read_draft_item", {"path": path, "view": view, "cursor": cursor, "limit": limit})


def _review_read_source(evidence_id: str, start: int, end: int) -> dict[str, Any]:
    return _invoke("review_read_source", {"evidence_id": evidence_id, "start": start, "end": end})


def _review_read_source_catalog(cursor: int, limit: int) -> dict[str, Any]:
    return _invoke("review_read_source_catalog", {"cursor": cursor, "limit": limit})


def _review_search_source(query: str, cursor: int, limit: int) -> dict[str, Any]:
    return _invoke("review_search_source", {"query": query, "cursor": cursor, "limit": limit})


def _review_read_source_section(section_id: str, cursor: int, max_units: int) -> dict[str, Any]:
    return _invoke("review_read_source_section", {"section_id": section_id, "cursor": cursor, "max_units": max_units})


def _review_save_notes(notes_json: str) -> dict[str, Any]:
    return _invoke("review_save_notes", {"notes_json": notes_json})


_TOOLS = {
    "handoff_section": (_handoff_section, "结束子任务并交接已保存条目。notes为普通文本，只写未决问题或保留缺项；无待办省略或空字符串，不写JSON数组，不重述成果。"),
    "save_basic_info": (_save_basic_info, "保存一批基本信息。每项field/value/source_refs/status；不填section、row、complete。新条目省略id。"),
    "save_defects": (_save_defects, "保存一批病害。每项直接写病害描述、病害部位等已知字段与source_refs/status；不嵌套row，不填section。"),
    "save_recommendations": (_save_recommendations, "保存一批维修建议。每项直接写建议内容等已知字段与source_refs/status；不嵌套row，不填section。"),
    "submit_section": (_submit_section, "提交本子任务的结构化抽取条目与未解决事项。entries可分批，省略条目ID时程序分配；complete=false仅保存，complete=true在真实阅读覆盖与交接检查通过后结束子任务。失败按回执修正，不重复重写已保存内容；子任务交接不代表整篇报告已完成。"),
    "review_candidates": (_review_candidates, "审核已保存的子任务候选：按稳定candidate_id选择、合并、局部修改或排除，并提交task_reviews及必要issue_reviews。依据交接原文做语义判断；无须重抄完整条目、版本或行号，冲突和证据不足时按需回查。此操作不替代Word导出、自查和finish。"),
    "quick_review": (_quick_review, "速度优先集中审核：只提交必要修改和remaining_issues，明确接受未解决质量问题暂缓。保留原始证据和正确候选，不要求逐条覆盖核查；保存后读取实际Word并finish。"),
    "submit_review": (_submit_review, "按同一问题或对象的小批提交审核决定：selections、merges、patches、exclusions、reviews五个集合每次合计最多5项，一组合并算1项，不拆其成员；有关联的修改和问题审核放在同批。selections选择基本字段候选ID；merges明确keep_id/other_ids/reason及可选changes；patches局部修改；exclusions明确排除候选和理由；reviews按question_id/status/reason处理问题。整次提交原子保存，等待回执再提交下一批；失败时修正该失败批仍需的事项，不重交以前成功的批次。程序维护行号、版本和问题归属；省略不改动、不表示已审核。保存后仍须检查实际Word并finish。"),
    "save_entries": (_save_entries, "保存本子任务的原文抽取条目；entries为JSON对象数组，使用共享字段合同。每条保留source_refs与explicit/inferred/uncertain/conflict状态；原文缺失不补写。source_refs可只给evidence_id：工具返回的E编号或当前文档mineru:<docSHA>#/...地址。程序准确展开已读原文范围；MinerU地址不填偏移，E范围的start/end须同时给出且不得猜测。"),
    "read_entries": (_read_entries, "回读本子任务已保存条目，程序按当前版本自动续页；无需填写位置。读完后明确返回完成，需回看时restart=true。limit默认8，超预算可缩小。"),
    "complete_section": (_complete_section, "交接本子任务：summary说明实际处理范围与结果，unresolved明确未解决事项（可空）。仅表示子任务交接，不代表整篇报告交付、Word自查或语义审核通过。"),
    "read_outline": (_read_outline, "读取报告章节和表格索引。程序自动保存索引分页位置；重复调用继续下一页，restart=true从头回看。索引不代表已读正文。"),
    "read_table_rows": (_read_table_rows, "读取指定表格，程序按该表独立进度续读并保留表头与说明，无需填写行号。max_rows默认8；restart=true回看，读完不自动重读。"),
    "read_evidence": (_read_evidence, "按证据编号读取原文片段；所有事实必须来自原文。"),
    "read_source": (_read_source, "按evidence_id读原文：支持工具返回的E编号或当前文档mineru:<docSHA>#/texts/1、#/tables/1/r0/c0等地址，可读整表、行或单元格。程序自动续读及必要上下文，无需字符位置或哈希另字段；再次调用同一对象继续，restart=true回看。读完不代表已提取或核对。"),
    "read_source_range": (_read_source_range, "仅在需要核对工具已给出的准确原文范围时使用。start/end为evidence_id内部Unicode字符半开区间，不猜长度，不作为普通续页方式；普通阅读用read_source。"),
    "read_section_range": (_read_section_range, "仅用于回查工具已给出的准确章节范围；不改变自动阅读进度。普通续读用read_section，不抄写位置。"),
    "read_table_range": (_read_table_range, "仅用于回查工具已给出的准确表格行范围；不改变自动阅读进度。普通续读用read_table_rows，不猜行号。"),
    "read_section": (_read_section, "读取目录中的真实章节；section_id可为root/task/原有章节编号或当前文档mineru:<docSHA>#/sections/S0001地址。程序独立保存位置、长单元续页和上下文，重复调用继续该章；restart=true回看，max_units默认8。"),
    "read_image": (_read_image, "读取报告中的指定图片并围绕 question 核对；首次 region_json 用 {} 读完整图。PNG/JPEG 区域是原始解码像素；EMF/WMF 区域是显式派生 PNG 画布像素，只能在完整图返回其尺寸、转换和坐标映射后指定，不能当作原矢量像素或实测尺寸。context_refs 是相关证据编号列表。"),
    "read_workspace": (_read_workspace, "读取当前报告的共享草稿、证据引用和待解决问题。"),
    "read_workspace_page": (_read_workspace_page, "读read_workspace的counts所指视图：facts、questions、unresolved_questions、draft_basic/draft_advice/draft_defects等，没有draft视图。程序按视图与版本自动续页及长条目片段；只选view，无需位置或版本参数，restart=true回看。版本变化会明确提示并从头读取；不要跨版本拼接片段，片段不是完整事实。未绑定候选不等于遗漏。"),
    "update_workspace": (_update_workspace, "提交当前报告草稿更新；update 是工作区合同规定的 JSON 对象，不是转义后的 JSON 字符串。事实须保留证据引用；长草稿分次提交 draft_patch。"),
    "review_draft": (_review_draft, "冻结并排队独立复核任务；本工具不在阅读会话内启动审阅SDK。收到queued后正常结束本轮，等待外层调度复核；排队不等于审核通过或交付完成。"),
    "read_source_objects": (_read_source_objects, "分页读取原生对象目录；cursor默认0、limit默认10。仅返回元数据，不代表已读原文或已理解公式/图片。"),
    "read_source_object": (_read_source_object, "读取指定对象的原始位置、完整上下文引用、表示状态和当前提议。元数据不等于公式证据；原文须通过read_source实际读取。"),
    "propose_source_object": (_propose_source_object, "提交对象处置提议；proposal须有disposition/rationale/affected_output_paths/supporting_refs/context_refs/source_limitations六项。不得提交origin_call_ids、accepted或表示核验标记；仅允许not_relied_on/outside_task/unresolved_relevant/unclassified，仍需独立审核。"),
    "search_in_document": (_search_in_document, 'query是1..16个原文短语的JSON数组，例如["病害","新增"]，按字面OR搜索。scope默认default：子任务内搜primary，否则全文；document显式全文，section要求section_id，document/task不能另带section_id。程序按查询及范围独立续页，无需cursor；restart=true重看相同查询。max_matches默认8。搜索片段不授予已读覆盖，无命中不证明不存在。'),
    "get_neighbors": (_get_neighbors, "读取指定 evidence_id 及前后相邻原文单元；每段 start/end 是该段 evidence_id 内部的 Unicode 字符半开区间，不可累加为全文偏移。按返回 continuation 续读，超预算则通过 read_section/read_source 分页。"),
    "extract_chunk": (_extract_chunk, "用自有模型抽取指定任务，附加证据编号可为空列表。"),
    "review_conflict": (_review_conflict, "用补充证据复查指定冲突；证据不足应保留未知。"),
    "split_task": (_split_task, "将过长或过密的任务拆为较小任务；拆分不代表子任务已完成。"),
    "finish": (_finish, "提交对实际Word的自查：self_check仅填写inspected_export、complete、basis、remaining_issues。程序依据本次请求收到的真实文件检查回执绑定版本；不要抄写文件哈希或版本。旧版、缺页或未收到文件内容仍不能提交。"),
    "inspect_delivery": (_inspect_delivery, "实际导出并回读当前Word。选择summary/basic/advice/defects，程序按当前文件与view续页，版本变化从头读取；无需cursor，restart=true回看。内容是否正确、完整由你判断。"),
}

_TOOLS.update({name: (globals()["_" + name], spec["description"])
               for name, spec in _REVIEW_SCHEMAS.items()})

_TOOLS.update({name: (globals()["_" + name], MAIN_REVIEW_DESCRIPTIONS[name]) for name in MAIN_REVIEW_SCHEMAS})


def session_boundary() -> dict[str, bool]:
    """Nonblocking application lock observation; not an SDK pause/join claim."""
    acquired = _SESSION_LOCK.acquire(blocking=False)
    try:
        return {"session_lock_available": acquired, "active_session_clear": _ACTIVE_SESSION is None}
    finally:
        if acquired:
            _SESSION_LOCK.release()


def _register_tools(decorator: Callable[..., Any], names: list[str], *, profile: str = "reading") -> None:
    registered = next((names for owner, names in _REGISTRATIONS if owner is decorator), None)
    if registered is None:
        registered = set()
        _REGISTRATIONS.append((decorator, registered))
    classes = next((items for owner, items in _REGISTERED_TOOL_CLASSES if owner is decorator), None)
    if classes is None:
        classes = {}
        _REGISTERED_TOOL_CLASSES.append((decorator, classes))
    for name in names:
        if name not in registered:
            function, description = _TOOLS[name]
            instance = decorator(name=name, description=description, retry_times=1)(function)
            if name == "update_workspace":
                _install_update_workspace_schema(instance)
            elif name in {"review_candidates", "submit_review", "quick_review", *MAIN_REVIEW_SCHEMAS}:
                _install_tool_schema(instance, name, _normalization_schema_for_tool(name))
            elif name in _SECTION_ENTRY_TOOLS or name in _SECTION_SOURCE_TOOLS:
                _install_tool_schema(instance, name, _section_tool_schema(name))
            elif name in {"finish", "inspect_delivery"}:
                _install_tool_schema(instance, name, _delivery_schema(name))
            elif name in {"read_source_objects", "read_source_object", "propose_source_object"}:
                _install_tool_schema(instance, name, _source_object_schema(name))
            elif name in _REVIEW_TOOLS:
                _install_tool_schema(instance, name, _REVIEW_SCHEMAS[name]["parameters"])
            classes[name] = type(instance)
            registered.add(name)
        elif ((profile == "section" and name in _SECTION_TOOLS)
                or name in {"search_in_document", "review_candidates", "submit_review", "quick_review", *MAIN_REVIEW_SCHEMAS}):
            # A cached registration must expose this exact contract too. Never
            # silently run a section handler behind a looser native schema.
            if _registered_tool_schema(decorator, name) != _normalization_schema_for_tool(name):
                raise ToolSchemaUnavailableError("Cached native tool schema differs from the explicit tool contract")


def _registered_tool_schema(decorator: Any, name: str) -> dict[str, Any]:
    """Read both native public metadata paths for this exact registration."""
    classes = next((items for owner, items in _REGISTERED_TOOL_CLASSES if owner is decorator), {})
    try:
        tool_class = classes[name]
        schemas = [getattr(tool_class, accessor)().parameters.to_json_schema()
                   for accessor in ("get_original_tool_info", "get_tool_info")]
        if schemas[0] != schemas[1] or type(schemas[0]) is not dict or schemas[0].get("type") != "object":
            raise ValueError("Native public schemas differ")
        # Tool schemas are local metadata. Never resolve an external reference.
        def local_references(value):
            if type(value) is dict:
                if "$ref" in value and (type(value["$ref"]) is not str or not value["$ref"].startswith("#")):
                    raise ValueError("External tool schema reference")
                for item in value.values(): local_references(item)
            elif type(value) is list:
                for item in value: local_references(item)
        local_references(schemas[0])
        return json.loads(_json_bytes(schemas[0]))
    except Exception as exc:
        raise AgentCallAssociationError("SDK schema rejection lacks exact registered tool metadata") from exc


def _select_generation_tools(payload, selector):
    """Select existing tools before exact counting; never rewrite model history."""
    original = deepcopy(payload)
    selected = selector(deepcopy(payload))
    if type(selected) is not dict:
        raise ValueError("Tool selection must return a complete request")
    unchanged = lambda value: {k: v for k, v in value.items() if k not in {"tools", "tool_choice"}}
    if unchanged(selected) != unchanged(original):
        raise ValueError("Tool selection changed history or inference parameters")
    if any(t not in original.get("tools", []) for t in selected.get("tools", [])):
        raise ValueError("Tool selection introduced an unregistered schema")
    return selected


def _config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    settings = {"profile": "reading", "api_key": "local-cityagent", "max_tokens": 1024, "timeout": 180,
                "max_iterations": 6, "max_calls": 8, "max_history_len": 5, "enable_thinking": False, "temperature": 0, **config}
    if type(settings.get("dynamic_output_budget", False)) is not bool:
        raise ValueError("dynamic_output_budget must be a boolean")
    if settings.get("dynamic_output_budget", False) and not callable(settings.get("prepare_generation_request")):
        raise ValueError("Dynamic SDK mode requires an actual complete-wire allocation callback")
    if settings["profile"] not in {"reading", "reviewer", "section"}:
        raise ValueError("Unknown native SDK tool profile")
    if "initial_memory_factory" in settings:
        if settings["profile"] != "section" or not callable(settings["initial_memory_factory"]):
            raise ValueError("initial_memory_factory is callable and section-only")
        if settings.get("native_compression") is not None:
            raise ValueError("initial_memory_factory cannot be combined with native_compression")
    if "terminal_handoff_state" in settings:
        terminal = settings.get("terminal_completion_tool")
        if not (settings["profile"] == "section" and terminal is None
                or settings["profile"] == "reading" and terminal == "finish_report"):
            raise ValueError("Terminal handoff requires section, or reading finish_report")
        for name in ("terminal_handoff_state", "terminal_generation_is_settled", "current_agent_call_id"):
            if not callable(settings.get(name)):
                raise ValueError(f"Terminal handoff requires callable {name}")
    if type(settings["enable_thinking"]) is not bool:
        raise ValueError("enable_thinking must be a boolean")
    settings["temperature"] = validate_temperature(settings["temperature"])
    url = settings.get("url", "")
    if not isinstance(url, str) or url != url.strip():
        raise ValueError("url must be an HTTP loopback URL")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid local model URL") from exc
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or "?" in url or "#" in url
            or parsed.path.rstrip("/") not in {"", "/v1"}
            or any(ord(character) < 33 for character in url)):
        raise ValueError("Only HTTP localhost, 127.0.0.1 or [::1] without credentials/query/fragment is allowed")
    # Accept the same root and /v1 forms as LocalModelClient.  The SDK consumes
    # an OpenAI base URL, while that client's per-request paths include /v1.
    host = "[::1]" if parsed.hostname == "::1" else parsed.hostname
    settings["url"] = f"http://{host}" + (f":{port}" if port is not None else "") + "/v1"
    if not isinstance(settings.get("model"), str) or not settings["model"].strip():
        raise ValueError("model must be a non-empty served model name")
    if not isinstance(settings["api_key"], str) or "\n" in settings["api_key"] or "\r" in settings["api_key"]:
        raise ValueError("api_key must be a single-line string")
    if "system_prompt" in settings and (
            not isinstance(settings["system_prompt"], str) or not settings["system_prompt"].strip()):
        raise ValueError("system_prompt must be a non-empty string when supplied")
    for name in ("max_tokens", "max_iterations", "max_calls", "max_history_len"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    # Callables remain in memory only; neither settings nor tokenizer objects
    # are emitted in tool receipts, SDK configuration, or fingerprints.
    if "current_agent_call_id" in settings and not callable(settings["current_agent_call_id"]):
        raise ValueError("current_agent_call_id must be callable")
    if "before_same_agent_continue" in settings and not callable(settings["before_same_agent_continue"]):
        raise ValueError("before_same_agent_continue must be callable")
    if ("parse_recovery_generation_is_settled" in settings
            and not callable(settings["parse_recovery_generation_is_settled"])):
        raise ValueError("parse_recovery_generation_is_settled must be callable")
    if "continue_after_native_exit" in settings:
        if settings["profile"] not in {"reading", "section"}:
            raise ValueError("Business continuation is reading/section-only")
        for name in ("continue_after_native_exit", "native_exit_is_settled",
                     "before_same_agent_continue", "current_agent_call_id"):
            if not callable(settings.get(name)):
                raise ValueError(f"Business continuation requires callable {name}")
    if "same_agent_parse_recovery" in settings and type(settings["same_agent_parse_recovery"]) is not bool:
        raise ValueError("same_agent_parse_recovery must be boolean")
    if "on_tool_result" in settings and not callable(settings["on_tool_result"]):
        raise ValueError("on_tool_result must be callable")
    if "before_native_model" in settings and not callable(settings["before_native_model"]):
        raise ValueError("before_native_model must be callable")
    if "select_generation_tools" in settings and not callable(settings["select_generation_tools"]):
        raise ValueError("select_generation_tools must be callable")
    if "run_id" in settings and (type(settings["run_id"]) is not str or not settings["run_id"].strip()):
        raise ValueError("run_id must be a non-empty host-owned string")
    if ("max_tool_tokens" in settings) != ("count_tokens" in settings):
        raise ValueError("max_tool_tokens and count_tokens must be supplied together")
    if "max_tool_tokens" in settings:
        if type(settings["max_tool_tokens"]) is not int or settings["max_tool_tokens"] < 1:
            raise ValueError("max_tool_tokens must be a positive integer")
        if not callable(settings["count_tokens"]):
            raise ValueError("count_tokens must be callable")
        if _response_tokens(settings["count_tokens"], _tool_budget_error()) > settings["max_tool_tokens"]:
            raise ValueError("max_tool_tokens cannot fit its minimal error response")
    if (type(settings["timeout"]) not in {int, float} or not math.isfinite(settings["timeout"])
            or settings["timeout"] <= 0):
        raise ValueError("timeout must be a positive finite number")
    return settings



def _prove_stopped_native_batch(messages, calls, answers, key, sdk_error, tool_calls,
                                actual_responses=None, argument_normalizations=None):
    """Join executed siblings by exact native pairs and durable receipt hashes.

    A handler's name alone is not an identity. Public dispatcher defaults may
    be applied for this comparison only; original native arguments stay intact.
    An ambiguous, missing or unmatched execution receipt forbids continuation.
    """
    index = calls[key][0]
    batch = []
    results = {}
    for chunk in messages[index].chunks:
        batch.extend(getattr(chunk.content, "tool_call_messages", None) or [])
    for message in messages:
        for chunk in message.chunks:
            content = chunk.content
            if type(content).__name__ == "ToolResultMessage":
                results[content.tool_call_id] = content
    batch_ids = [call.tool_call_id for call in batch]
    if key not in batch_ids or any(call_id not in answers and call_id not in batch_ids for call_id in calls):
        raise NativeBatchRecoveryError("pending_batch_not_unique")
    rejected_index = batch_ids.index(key)
    if any(call_id in answers for call_id in batch_ids[rejected_index:]):
        raise NativeBatchRecoveryError("native_answer_after_rejection")

    origin = sdk_error.get("agent_call_id")
    if not origin and tool_calls:
        raise NativeBatchRecoveryError("handler_origin_unavailable")
    rows = [(index, row) for index, row in enumerate(tool_calls) if row.get("agent_call_id") == origin]
    argument_hashes = {}
    for call in batch:
        try:
            original = json.loads(call.arguments)
            if not isinstance(original, dict) or call.tool_name not in _TOOLS:
                continue
            schema = _normalization_schema_for_tool(call.tool_name)
            if schema is not None:
                normalized, changes, _ = _normalize_container_arguments(original, schema)
                if changes:
                    evidence = [row for row in (argument_normalizations or [])
                                if row.get("tool_name") == call.tool_name and row.get("agent_call_id") == origin
                                and row.get("original_arguments_sha256") == _hash(original)
                                and row.get("normalized_arguments_sha256") == _hash(normalized)
                                and row.get("schema_sha256") == _hash(schema)
                                and row.get("native_validation_passed") is True
                                and row.get("conversions") == changes]
                    if not evidence:
                        continue
                    original = normalized
            dispatched = _bound_dispatch_arguments(call.tool_name, original)
            argument_hashes[call.tool_call_id] = _hash(dispatched)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    proved = set()
    missing_results = []
    for receipt_index, row in rows:
        candidates = [call.tool_call_id for call in batch if call.tool_name == row.get("name")
                      and argument_hashes.get(call.tool_call_id) == row.get("args_sha256")
                      and row.get("args_sha256") is not None]
        if len(candidates) != 1:
            raise NativeBatchRecoveryError("handler_identity_ambiguous")
        call_id = candidates[0]
        if call_id in proved or batch_ids.index(call_id) >= rejected_index:
            raise NativeBatchRecoveryError("handler_execution_after_rejection_or_duplicate")
        result = results.get(call_id)
        if not isinstance(row.get("result_sha256"), str):
            raise NativeBatchRecoveryError("handler_result_hash_missing")
        if result is not None:
            if call_id not in answers or _hash(result.content) != row["result_sha256"]:
                raise NativeBatchRecoveryError("native_result_hash_mismatch")
        else:
            snapshot = (actual_responses or {}).get(receipt_index)
            if type(snapshot) is not dict:
                raise NativeBatchRecoveryError("actual_return_snapshot_missing")
            if any(snapshot.get(field) != row.get(field) for field in (
                    "name", "agent_call_id", "args_sha256", "result_sha256", "status")):
                raise NativeBatchRecoveryError("actual_return_identity_mismatch")
            response = snapshot.get("response")
            if (row.get("status") != "success" or type(response) is not dict
                    or response.get("ok") is not True or response.get("tool") != row["name"]):
                raise NativeBatchRecoveryError("actual_return_execution_not_confirmed")
            if _hash(response) != row["result_sha256"]:
                raise NativeBatchRecoveryError("actual_return_hash_mismatch")
            missing_results.append((call_id, row["name"], json.loads(_json_bytes(response)), receipt_index))
        proved.add(call_id)
    if proved != set(batch_ids[:rejected_index]):
        raise NativeBatchRecoveryError("pre_rejection_execution_not_fully_proved")
    # No mutation happens until every existing or missing result in the stopped
    # prefix has been proved. Ambiguous later siblings cannot half-repair memory.
    return batch, proved, missing_results


def _paired_parse_feedback(agent, event, sdk_error, tool_calls, *, context=None, tool_schema=None,
                           actual_responses=None, argument_normalizations=None, argument_parse_errors=None):
    """Record the real SDK parse rejection as its missing failed tool result.

    No argument repair, handler execution or success is manufactured. Continue
    only after an exact native call ID/name and absence of a result are proved.
    """
    from .document_context_compression import load_native_api, native_tool_pairs
    api = load_native_api()
    key = getattr(event, "tool_call_id", None)
    name = sdk_error.get("tool_name")
    if not isinstance(key, str) or not key or not isinstance(name, str):
        raise AgentCallAssociationError("SDK parse failure lacks an exact tool call identity")
    memory = (context if context is not None else agent.run_context).memory
    messages = list(memory.get_messages())
    calls, answers = native_tool_pairs(messages)
    matches = []
    for message in messages:
        for chunk in message.chunks:
            for call in getattr(chunk.content, "tool_call_messages", None) or []:
                if call.tool_call_id == key and call.tool_name == name: matches.append(call)
    if len(matches) != 1 or key not in calls or key in answers:
        raise AgentCallAssociationError("SDK parse failure cannot be joined to an unanswered native tool call")
    raw = matches[0].arguments
    if not isinstance(raw, str):
        raise AgentCallAssociationError("SDK JSON rejection does not refer to native argument text")
    code = sdk_error.get("code", "SDK_TOOL_ARGUMENT_JSON")
    schema_details = {}
    if code == "SDK_TOOL_ARGUMENT_JSON":
        owned_sha = sdk_error.get("adapter_parse_error_sha256")
        if owned_sha is not None:
            records = [row for row in argument_parse_errors or [] if _hash(row) == owned_sha]
            if (len(records) != 1 or type(event).__name__ != "ToolErrorAction"
                    or sdk_error.get("tool_call_id_sha256") != _hash(key)
                    or sdk_error.get("arguments_text_sha256") != _hash(raw)
                    or records[0].get("arguments_text_sha256") != _hash(raw)
                    or records[0].get("agent_call_id") != sdk_error.get("agent_call_id")
                    or not sdk_error.get("agent_call_id") or records[0].get("tool_name") != name
                    or _dump_event(event).get("message") != _STRICT_ARGUMENT_JSON_MARKER + owned_sha
                    or type(tool_schema) is not dict or records[0].get("schema_sha256") != _hash(tool_schema)):
                raise AgentCallAssociationError("Adapter JSON rejection lacks its exact native call and receipt")
            _, diagnostic = _strict_outer_argument_object(raw)
            if diagnostic is None or diagnostic != records[0].get("diagnostic"):
                raise AgentCallAssociationError("Adapter JSON rejection cannot be reproduced exactly")
            schema_details = {"adapter_parse_error_sha256": owned_sha, "arguments_text_sha256": _hash(raw),
                              "argument_repair": {"issues": [diagnostic], "expected_type": "object",
                                  "minimal_shape_example": {k: v for issue in _argument_schema_diagnostics({}, tool_schema).get("issues", [])
                                      for k, v in issue.get("minimal_shape_example", {}).items()}, "examples_are_shapes_not_defaults": True,
                                  "action": "Regenerate only the invalid JSON representation using the enabled tool schema; preserve evidence-grounded values. No handler ran or values were supplied by the host."}}
        else:
            try: json.loads(raw)
            except json.JSONDecodeError: pass
            else: raise AgentCallAssociationError("SDK JSON rejection arguments unexpectedly parse; no continuation guessed")
    elif code == "SDK_TOOL_ARGUMENT_SCHEMA":
        if (type(event).__name__ != "ToolJsonSchemaErrorAction"
                or sdk_error.get("tool_call_id_sha256") != _hash(key)
                or type(tool_schema) is not dict or tool_schema.get("type") != "object"):
            raise AgentCallAssociationError("SDK schema rejection lacks its exact event, call or native schema")
        try:
            from jsonschema.validators import validator_for
            arguments = json.loads(raw)
            _json_bytes(arguments)  # Reject non-JSON numeric constants, without changing the call.
            validator = validator_for(tool_schema)
            validator.check_schema(tool_schema)
            errors = list(validator(tool_schema).iter_errors(arguments))
        except Exception as exc:
            raise AgentCallAssociationError("SDK schema rejection cannot be validated against original arguments") from exc
        if not errors:
            raise AgentCallAssociationError("SDK schema rejection arguments actually satisfy native schema")
        if (sdk_error.get("arguments_text_sha256") is not None
                and sdk_error["arguments_text_sha256"] != _hash(raw)):
            raise AgentCallAssociationError("SDK schema rejection argument identity differs")
        schema_details = {"arguments_text_sha256": _hash(raw), "arguments_sha256": _hash(arguments),
                          "schema_sha256": _hash(tool_schema),
                          "schema_error_count": len(errors),
                          "required_parameters": list(tool_schema.get("required", []))}
        schema_details["argument_repair"] = _argument_schema_diagnostics(arguments, tool_schema)
    else:
        raise AgentCallAssociationError("Only exact SDK JSON or argument-schema rejection can continue")
    batch, proved, missing_results = _prove_stopped_native_batch(
        messages, calls, answers, key, sdk_error, tool_calls, actual_responses, argument_normalizations)
    restored = []
    for call_id, tool_name, response, receipt_index in missing_results:
        memory.add_chunk(api["messages"].ToolResultMessage(name=tool_name, content=response,
                         tool_call_id=call_id), role=api["memory"].MessageRole.TOOL)
        restored.append({"tool_name": tool_name, "tool_call_id_sha256": _hash(call_id),
                         "receipt_index": receipt_index, "result_sha256": _hash(response),
                         "handler_reexecuted": False, "restored_actual_return": True})
    # Confirm the native store retained the exact real return, before recording
    # the rejected/unexecuted calls. No handler is invoked by this repair path.
    current_messages = list(memory.get_messages())
    _, current_answers = native_tool_pairs(current_messages)
    retained_results = {chunk.content.tool_call_id: chunk.content for message in current_messages
                        for chunk in message.chunks if type(chunk.content).__name__ == "ToolResultMessage"}
    if any(call_id not in current_answers or _hash(retained_results[call_id].content) != _hash(response)
           for call_id, _, response, _ in missing_results):
        raise NativeBatchRecoveryError("restored_actual_return_not_retained")
    answers = current_answers
    paired = []
    for call in batch:
        if call.tool_call_id in answers:
            continue
        rejected = call.tool_call_id == key
        error = {"code": code if rejected else "SDK_TOOL_BATCH_ABORTED",
                 "handler_executed": False, "saved": False,
                 "observed_sdk_rejection": rejected,
                 "action": sdk_error["action"] if rejected else
                     "An earlier tool in this exact native batch was rejected by the SDK; this tool was not executed. Regenerate any still-needed call.",
                 **(schema_details if rejected else {"blocked_by_tool_call_id_sha256": _hash(key)})}
        payload = {"ok": False, "tool": call.tool_name, "error": error,
                   "arguments_repaired_by_host": False}
        if call.tool_name == "submit_review":
            from .document_review_submission import rejected_review_receipt
            # The complete stopped-batch proof above, not the SDK error text,
            # establishes that this exact submission never reached its handler.
            # The SDK has no authoritative current workspace version here.
            arguments, _ = _strict_outer_argument_object(call.arguments)
            payload["batch_receipt"] = rejected_review_receipt(arguments)
            error["action"] = payload["batch_receipt"]["instruction"]
            if "argument_repair" in error:
                error["argument_repair"] = {**error["argument_repair"],
                    "action": error["argument_repair"]["action"] + " " + error["action"]}
        memory.add_chunk(api["messages"].ToolResultMessage(name=call.tool_name, content=payload,
                         tool_call_id=call.tool_call_id), role=api["memory"].MessageRole.TOOL)
        paired.append({"tool_name": call.tool_name, "tool_call_id_sha256": _hash(call.tool_call_id),
                       "arguments_text_sha256": _hash(call.arguments),
                       "result_sha256": _hash(payload),
                       "code": error["code"], "observed_sdk_rejection": rejected})
    after_calls, after_answers = native_tool_pairs(memory.get_messages())
    if after_calls != calls or any(call.tool_call_id not in after_answers for call in batch):
        raise AgentCallAssociationError("Native failed or unexecuted batch results were not retained")
    continuation_action = sdk_error["action"]
    if any(row["tool_name"] == "submit_review" for row in paired):
        from .document_review_submission import REVIEW_REJECTED_ACTION
        continuation_action = (REVIEW_REJECTED_ACTION if name == "submit_review" else
            continuation_action + " For the unexecuted submit_review: " + REVIEW_REJECTED_ACTION)
    return {"code":code,"tool_name":name,"tool_call_id_sha256":_hash(key),
        "agent_call_id": sdk_error.get("agent_call_id"),
        "handler_executed":False,"saved":False,"same_agent":True,"same_context":True,
        "paired_failed_tool_result":True, "paired_batch_results": paired,
        "batch_tool_count": len(batch), "preserved_executed_tool_count": len(proved),
        "restored_actual_tool_results": restored, "restored_actual_tool_result_count": len(restored),
        "all_batch_tool_calls_paired": True, **schema_details}, (("上一工具调用在SDK校验参数结构时失败，未调用业务handler、未保存任何修改。"
        if code == "SDK_TOOL_ARGUMENT_SCHEMA" else "上一工具调用在SDK解析JSON时失败，未调用业务handler、未保存任何修改。")
        + continuation_action + " 同批尚未执行的后续调用也已记录为中止，未执行或保存；此前成功的工具结果保持不变。请在当前Agent和工作区中先确认最新版本，再提交有效的小步参数；不要假定失败操作已保存。")


def _native_json_recovery_is_owned(session, recovery, calls):
    """Prove an SDK-owned JSON rejection independently of our strict wrapper.

    Reading tools can be rejected by the SDK before an adapter parse receipt
    exists. Join its exact event, native call, settled generation and failed
    batch receipt; never infer ownership from an error message alone.
    """
    if recovery.get("adapter_parse_error_sha256") is not None:
        return False
    roots = [call for call in calls if _hash(call.get("id")) == recovery.get("tool_call_id_sha256")]
    if len(roots) != 1:
        return False
    root = roots[0]
    function = root.get("function") or {}
    raw = function.get("arguments")
    if type(raw) is not str:
        return False
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        pass
    else:
        return False
    errors = [row for row in session.get("sdk_tool_errors", [])
              if row.get("code") == "SDK_TOOL_ARGUMENT_JSON" and row.get("type") == "ToolErrorAction"
              and row.get("agent_call_id") == recovery.get("agent_call_id")
              and row.get("tool_call_id_sha256") == _hash(root["id"])
              and row.get("tool_name") == function.get("name")
              and type(row.get("event_sha256")) is str and len(row["event_sha256"]) == 64
              and row.get("adapter_parse_error_sha256") is None]
    proofs = [row for row in recovery.get("paired_batch_results", [])
              if row.get("code") == "SDK_TOOL_ARGUMENT_JSON" and row.get("observed_sdk_rejection") is True
              and row.get("tool_call_id_sha256") == _hash(root["id"])
              and row.get("tool_name") == function.get("name")
              and row.get("arguments_text_sha256") == _hash(raw)]
    return len(errors) == len(proofs) == 1


def _project_rejected_tool_history(payload, session):
    """Render proved failed JSON calls as text, without changing native memory.

    Some chat templates json.loads every historical function.arguments. A
    malformed call remains a real failed event, but cannot remain a protocol
    tool call in those templates. Only a settled, exactly paired SDK rejection
    authorizes this projection. Successful siblings stay byte-for-byte equal.
    """
    projected = json.loads(_json_bytes(payload))
    messages = projected.get("messages")
    if type(messages) is not list:
        raise AgentCallAssociationError("Recovery history lacks complete messages")
    original_hash = _hash(payload)
    output, changes, seen_ids = [], [], set()
    index = 0
    while index < len(messages):
        message = messages[index]
        calls = message.get("tool_calls") or []
        if not calls:
            output.append(message); index += 1; continue
        if message.get("role") != "assistant" or type(calls) is not list:
            raise AgentCallAssociationError("Recovery history has malformed tool structure")
        bad = {}
        for call in calls:
            key = call.get("id")
            function = call.get("function") or {}
            name, raw = function.get("name"), function.get("arguments")
            if type(key) is not str or not key or key in seen_ids or type(raw) is not str:
                raise AgentCallAssociationError("Recovery history has ambiguous native tool identities")
            seen_ids.add(key)
            _, diagnostic = _strict_outer_argument_object(raw)
            if diagnostic is None:
                continue
            proofs = []
            for recovery in session.get("parse_recovery", []):
                if (recovery.get("code") != "SDK_TOOL_ARGUMENT_JSON"
                        or recovery.get("generation_terminal_verified") is not True
                        or recovery.get("all_batch_tool_calls_paired") is not True
                        or not recovery.get("agent_call_id")):
                    continue
                owned = [row for row in session.get("argument_parse_errors", [])
                         if _hash(row) == recovery.get("adapter_parse_error_sha256")
                         and row.get("agent_call_id") == recovery["agent_call_id"]
                         and row.get("handler_executed") is False and row.get("saved") is False]
                native_owned = _native_json_recovery_is_owned(session, recovery, calls)
                rows = [row for row in recovery.get("paired_batch_results", [])
                        if row.get("tool_call_id_sha256") == _hash(key) and row.get("tool_name") == name
                        and row.get("arguments_text_sha256") == _hash(raw)
                        and row.get("code") in {"SDK_TOOL_ARGUMENT_JSON", "SDK_TOOL_BATCH_ABORTED"}]
                if len(rows) == 1 and (
                        (rows[0]["code"] == "SDK_TOOL_ARGUMENT_JSON"
                         and rows[0].get("observed_sdk_rejection") is True
                         and recovery.get("tool_call_id_sha256") == _hash(key)
                         and (native_owned or (len(owned) == 1
                              and owned[0].get("arguments_text_sha256") == _hash(raw)
                              and owned[0].get("diagnostic") == diagnostic)))
                        or (rows[0]["code"] == "SDK_TOOL_BATCH_ABORTED"
                            and rows[0].get("observed_sdk_rejection") is False
                            and recovery.get("tool_call_id_sha256") != _hash(key)
                            and (len(owned) == 1 or native_owned))):
                    proofs.append((recovery, rows[0]))
            if len(proofs) != 1:
                raise AgentCallAssociationError("Malformed historical tool arguments lack a unique settled recovery")
            bad[key] = (call, proofs[0][0], proofs[0][1], diagnostic)
        if not bad:
            output.append(message); index += 1; continue
        # A complete native batch must have one adjacent result for every call.
        # Never move text through a still-open tool batch or borrow another ID.
        end = index + 1
        results = []
        while end < len(messages) and messages[end].get("role") == "tool":
            results.append(messages[end]); end += 1
        keys = [call["id"] for call in calls]
        result_keys = [row.get("tool_call_id") for row in results]
        if len(set(result_keys)) != len(result_keys) or sorted(result_keys) != sorted(keys):
            raise AgentCallAssociationError("Recovery history does not contain its complete exact result batch")
        text_records = []
        for result_message in results:
            key = result_message["tool_call_id"]
            if key not in bad:
                continue
            call, recovery, proof, diagnostic = bad[key]
            raw_result = result_message.get("content")
            try:
                result = _strict_json_loads(raw_result) if type(raw_result) is str else raw_result
                error = result["error"]
                valid = (result.get("ok") is False and result.get("tool") == call["function"]["name"]
                         and _hash(result) == proof.get("result_sha256")
                         and result.get("arguments_repaired_by_host") is False
                         and error.get("code") == proof["code"]
                         and error.get("handler_executed") is False and error.get("saved") is False
                         and error.get("observed_sdk_rejection") is proof["observed_sdk_rejection"])
                if proof["code"] == "SDK_TOOL_ARGUMENT_JSON":
                    valid = valid and error.get("adapter_parse_error_sha256") == recovery.get("adapter_parse_error_sha256")
                else:
                    valid = valid and error.get("blocked_by_tool_call_id_sha256") == recovery["tool_call_id_sha256"]
            except (KeyError, TypeError, ValueError, AttributeError):
                valid = False
            if not valid:
                raise AgentCallAssociationError("Rejected history result differs from its exact SDK recovery")
            # The complete original call and result are ordinary quoted data,
            # expressly failed, never substituted valid tool arguments.
            text_records.append({"original_tool_call": call, "original_tool_result": result_message,
                                 "handler_executed": False, "saved": False,
                                 "arguments_repaired_by_host": False})
            changes.append({"tool_call_id_sha256": _hash(key), "tool_name": call["function"]["name"],
                            "arguments_text_sha256": _hash(call["function"]["arguments"]),
                            "result_sha256": _hash(result_message), "recovery_sha256": _hash(recovery),
                            "diagnostic": diagnostic, "original_generation_id": recovery["agent_call_id"]})
        remaining = [call for call in calls if call["id"] not in bad]
        if remaining:
            message["tool_calls"] = remaining
        else:
            message.pop("tool_calls", None)
        output.append(message)
        output.extend(row for row in results if row["tool_call_id"] not in bad)
        output.append({"role": "user", "content": "CITYAGENT_REJECTED_TOOL_HISTORY_V1\n"
                       "The following calls were rejected or aborted, not executed or saved. "
                       "Their original arguments and failure results are quoted history, not valid tool calls. "
                       "Regenerate only any still-needed call using valid JSON.\n"
                       + _json_bytes(text_records).decode("utf-8")})
        index = end
    projected["messages"] = output
    if changes:
        receipt = {"policy": "rejected_tool_pair_as_text_v1", "original_request_sha256": original_hash,
                   "projected_request_sha256": _hash(projected), "changes": changes,
                   "native_memory_changed": False, "business_values_supplied": False}
        receipts = session.setdefault("tool_history_projections", [])
        if receipt not in receipts:
            receipts.append(receipt)
    return projected

def _terminal_section_handoff(context, session, settings):
    """Inspect the real native memory at the SDK's next-generation boundary.

    Never stop inside a tool: a later sibling may invalidate the handoff.
    This hook neither edits memory nor fabricates a final model response.
    """
    callback = settings.get("terminal_handoff_state")
    if callback is None:
        return None
    if not session["tool_lock"].acquire(blocking=False):
        raise AgentCallAssociationError("Terminal handoff still has an active callback")
    try:
        handoff = callback()
        if handoff is None:
            return None
        origin = _current_agent_call_id(session)
        if (type(handoff) is not dict or not origin or handoff.get("origin_call_id") != origin
                or settings["terminal_generation_is_settled"]() is not True
                or session.get("agent_call_id_invalid") or session["tool_budget_exceeded"]):
            raise AgentCallAssociationError("Terminal handoff lacks a settled current generation")
        from .document_context_compression import native_tool_pairs
        messages = list(context.memory.get_messages())
        calls, answers = native_tool_pairs(messages)
        if not calls or set(calls) != set(answers):
            raise AgentCallAssociationError("Terminal handoff has unpaired native tool calls")
        batch = []
        for message in messages:
            for chunk in message.chunks:
                pending = getattr(chunk.content, "tool_call_messages", None)
                if pending:
                    batch = list(pending)
        receipt_rows = [row for row in session["tool_calls"] if row.get("agent_call_id") == origin]
        completion_tool = settings.get("terminal_completion_tool")
        terminal_tools = {completion_tool} if completion_tool else {"complete_section", "handoff_section", "submit_section"}
        if (not any(call.tool_name in terminal_tools for call in batch)
                or not any(row["name"] in terminal_tools and row["status"] == "success" for row in receipt_rows)
                or len(batch) != len(receipt_rows)):
            raise AgentCallAssociationError("Terminal handoff lacks the complete executed native batch")
        results = {chunk.content.tool_call_id: chunk.content for message in messages
                   for chunk in message.chunks if type(chunk.content).__name__ == "ToolResultMessage"}
        # A matching row count is not an execution proof. Join every native
        # call/result to this generation's actual dispatcher receipt. Identical
        # siblings consume identical receipts separately, rather than claiming
        # an identity that their arguments cannot distinguish.
        remaining = list(receipt_rows)
        terminal_success = False
        for call in batch:
            try:
                original = (_strict_json_loads(call.arguments) if type(call.arguments) is str
                            else json.loads(_json_bytes(call.arguments)))
                if type(original) is not dict or call.tool_name not in _TOOLS:
                    raise ValueError("Native arguments are not an enabled object")
                schema = _normalization_schema_for_tool(call.tool_name)
                if schema is not None:
                    normalized, changes, _ = _normalize_container_arguments(original, schema)
                    if changes:
                        audits = [row for row in session.get("argument_normalizations", [])
                                  if row.get("tool_name") == call.tool_name
                                  and row.get("agent_call_id") == origin
                                  and row.get("original_arguments_sha256") == _hash(original)
                                  and row.get("normalized_arguments_sha256") == _hash(normalized)
                                  and row.get("schema_sha256") == _hash(schema)
                                  and row.get("native_validation_passed") is True
                                  and row.get("conversions") == changes]
                        if not audits:
                            raise ValueError("Native normalization has no actual validation receipt")
                        original = normalized
                dispatched = _bound_dispatch_arguments(call.tool_name, original)
                result = results[call.tool_call_id].content
                if (type(result) is not dict or type(result.get("ok")) is not bool
                        or result.get("tool") != call.tool_name):
                    raise ValueError("Native return is not the actual dispatcher envelope")
                expected = {"name": call.tool_name, "args_sha256": _hash(dispatched),
                            "result_sha256": _hash(result),
                            "status": "success" if result["ok"] else "failed"}
                match = next((index for index, row in enumerate(remaining)
                              if all(row.get(key) == value for key, value in expected.items())), None)
                if match is None:
                    raise ValueError("Native call and return lack an exact execution receipt")
                remaining.pop(match)
                if (completion_tool and call.tool_name == completion_tool and result["ok"]
                        and type(result.get("result")) is dict and result["result"].get("ok") is True):
                    terminal_success = True
                elif not completion_tool and result["ok"] and call.tool_name in {"complete_section", "handoff_section"}:
                    terminal_success = True
                elif (result["ok"] and call.tool_name == "submit_section"
                      and dispatched.get("complete") is True
                      and type(result.get("result")) is dict
                      and result["result"].get("status") == "handed_off"):
                    terminal_success = True
            except (KeyError, TypeError, ValueError, RecursionError) as exc:
                raise AgentCallAssociationError("Terminal handoff native batch execution is not exactly proved") from exc
        if remaining:
            raise AgentCallAssociationError("Terminal handoff has unmatched execution receipts")
        if not terminal_success:
            raise AgentCallAssociationError("Terminal handoff lacks a successful completed submission")
        return {"terminal_handoff_sha256": _hash(handoff),
                "terminal_handoff_origin_call_id": origin,
                "terminal_handoff_proof": {"all_batch_tool_calls_paired": True,
                    "batch_tool_count": len(batch),
                    "native_tool_call_ids_sha256": _hash([call.tool_call_id for call in batch])}}
    finally:
        session["tool_lock"].release()


def run_react(prompt: str, handlers: dict[str, Callable[..., Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Run one fresh ReAct session, returning receipts even on SDK failure.

    ``status=completed`` is transport/SDK completion, not business acceptance.
    Tool errors remain in the tool receipts and are returned to the agent as
    structured errors so that it can choose another action.  Native final
    events and tool executions are evidence of SDK use, not HTTP-call counts.
    When config supplies ``system_prompt``, it is used only as the SDK system
    instruction and ``prompt`` is passed only to step(). Omitting it preserves
    the legacy divide workflow's use of prompt in both positions.
    """
    global _ACTIVE_SESSION
    started = time.monotonic()
    result: dict[str, Any] = {
        "status": "error", "sdk_version": None, "events": [], "tool_calls": [],
        "sdk_tool_errors": [],
        "argument_normalizations": [],
        "argument_parse_errors": [],
        "tool_history_projections": [],
        "response_text": "", "native_final_event": False,
        "session_boundary": {"iteration_exited": False, "tool_callbacks_drained": False,
                             "active_session_clear": False, "session_lock_released": False},
        "external_model_used": False, "model_call_count": None,
    }
    try:
        settings = _config(config)
        result["step_invocations"] = 0
        result["parse_recovery"] = []
        result["business_continuations"] = []
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not isinstance(handlers, dict) or not handlers:
            raise ValueError("at least one authorized handler is required")
        allowed = {"reviewer": _REVIEW_TOOLS, "reading": _READING_TOOLS, "section": _SECTION_TOOLS}[settings["profile"]]
        if (set(handlers) - allowed or (settings["profile"] == "reviewer" and set(handlers) != allowed)
                or any(not callable(value) for value in handlers.values())):
            raise ValueError("handlers must contain only callable CityAgent tools")
        # Precise-range methods remain callable in the lower-level document
        # API, but neither ordinary profile advertises or executes them.
        handlers = {name: handler for name, handler in handlers.items() if name not in _INTERNAL_RANGE_TOOLS}
        if not handlers:
            raise ValueError("at least one model-visible tool is required")
        result["profile"] = settings["profile"]
        result["adapter_version"] = NATIVE_REVIEW_ADAPTER_VERSION
        sdk = _load_tongagents()
        result["sdk_version"] = sdk["version"]
        # Lock covers construction, callbacks and event consumption.  Nested
        # sessions are rejected explicitly instead of rebinding outer tools.
        with _SESSION_LOCK:
            if _ACTIVE_SESSION is not None:
                raise RuntimeError("nested CityAgent ReAct sessions are unsupported")
            _ACTIVE_SESSION = {"handlers": dict(handlers), "tool_calls": result["tool_calls"], "profile": settings["profile"],
                               "argument_normalizations": result["argument_normalizations"],
                               "argument_parse_errors": result["argument_parse_errors"],
                               "sdk_tool_errors": result["sdk_tool_errors"],
                               "parse_recovery": result["parse_recovery"],
                               "tool_history_projections": result["tool_history_projections"],
                               "max_calls": settings["max_calls"], "tool_budget_exceeded": False,
                               "max_tool_tokens": settings.get("max_tool_tokens"), "count_tokens": settings.get("count_tokens"),
                               "tool_lock": threading.RLock()}
            _ACTIVE_SESSION["session_id"] = uuid.uuid4().hex
            _ACTIVE_SESSION["run_id"] = settings.get("run_id", _ACTIVE_SESSION["session_id"])
            if settings.get("same_agent_parse_recovery", False):
                _ACTIVE_SESSION["tool_response_snapshots"] = {}
            if "current_agent_call_id" in settings:
                _ACTIVE_SESSION["current_agent_call_id"] = settings["current_agent_call_id"]
            if "on_tool_result" in settings:
                _ACTIVE_SESSION["on_tool_result"] = settings["on_tool_result"]
            allocation_context = None
            try:
                if settings.get("dynamic_output_budget", False):
                    from .document_context_budget import allocate_sdk_wire_requests
                    def projection(payload):
                        if "parse_recovery_generation_is_settled" in settings:
                            payload = _project_rejected_tool_history(payload, _ACTIVE_SESSION)
                        selector = settings.get("select_generation_tools")
                        if selector is not None:
                            payload = _select_generation_tools(payload, selector)
                        return payload
                    if "parse_recovery_generation_is_settled" in settings or "select_generation_tools" in settings:
                        pending_context = allocate_sdk_wire_requests(settings["url"], settings["prepare_generation_request"], projection)
                    else:
                        pending_context = allocate_sdk_wire_requests(settings["url"], settings["prepare_generation_request"])
                    pending_context.__enter__()
                    allocation_context = pending_context
                    result["output_budget_policy"] = "complete_request_remaining_context_v1"
                names = list(handlers)
                _register_tools(sdk["tool"], names, profile=settings["profile"])
                agent_settings = sdk["ReactAgentSetting"](
                    llm_config=sdk["ModelConfig"](
                        model_name=settings["model"], model_provider=sdk["ModelProvider"].OPENAI_COMPATIBLE,
                        url=settings["url"], api_key=settings["api_key"], temperature=settings["temperature"],
                        max_tokens=(None if settings.get("dynamic_output_budget", False) else settings["max_tokens"]), timeout=settings["timeout"], stream=False,
                        # SDK 2.7.20 converts temperature=0 to OpenAI's
                        # NOT_GIVEN.  extra_body preserves the explicit zero
                        # in the final HTTP JSON sent by the OpenAI client.
                        extra_body={"temperature": settings["temperature"], "chat_template_kwargs": {
                            "enable_thinking": settings["enable_thinking"]}},
                    ),
                    system_prompt=settings.get("system_prompt", prompt), max_iterations=settings["max_iterations"],
                    max_call_cnt=settings["max_calls"], max_tool_retry=1,
                    # Real SDK 2.7.20 rejects the second successful same-args
                    # read at 1. The adapter/managed journals own progress and
                    # replay; the unchanged session call budget stays finite.
                    same_tool_retry_max=(settings["max_calls"] if set(names) & _MANAGED_READ_TOOLS else 1),
                    max_history_len=settings["max_history_len"], tool_call_memory=True,
                    tool_identifier_list=names,
                    # Omitting tool_choice leaves tool selection to the model.
                )
                agent_class = sdk["ReactAgent"]
                compression = settings.get("native_compression")
                terminal_signal = _SectionHandoffComplete("Valid section handoff reached the native boundary")
                dispatch_session = _ACTIVE_SESSION

                class ContinuousNativeAgent(agent_class):
                    def _tool_call(self, tool_calls):
                        # The adapter already serializes mutable tools. Execute
                        # each native call through the real SDK, with its exact
                        # native ID in a host-only frame. Matching by arguments
                        # alone cannot distinguish same-batch duplicate calls.
                        answers = []
                        for call in tool_calls:
                            native_id = getattr(call, "tool_call_id", None)
                            if type(native_id) is not str or not native_id.strip():
                                raise AgentCallAssociationError("Native tool call ID is missing")
                            if dispatch_session is not _ACTIVE_SESSION or dispatch_session.get("native_dispatch"):
                                raise AgentCallAssociationError("Native tool dispatch scope is not exclusive")
                            dispatch_session["native_dispatch"] = {
                                "generation_call_id": _current_agent_call_id(dispatch_session),
                                "native_tool_call_id": native_id,
                                "tool_name": call.tool_name, "arguments": call.arguments}
                            try:
                                answers.extend(super()._tool_call(tool_calls=[call]))
                            finally:
                                dispatch_session.pop("native_dispatch", None)
                        return answers

                    def _call_llm_with_compression(self, messages):
                        proof = _terminal_section_handoff(context, _ACTIVE_SESSION, settings)
                        if proof is not None:
                            self.stop()
                            proof["terminal_handoff_proof"].update(
                                stopped_before_next_model_call=True, sdk_stop_returned=True)
                            result.update(proof)
                            raise terminal_signal
                        if compression is not None:
                            messages = compression.before_native_call(self, messages, context=context)
                        before_model = settings.get("before_native_model")
                        if before_model is not None:
                            feedback = before_model()
                            if feedback is not None:
                                if type(feedback) is not str or not feedback.strip():
                                    raise ValueError("before_native_model must return non-empty text or None")
                                from .document_context_compression import load_native_api
                                api = load_native_api()
                                context.memory.add_chunk(api["messages"].UserPromptMessage(content=[
                                    api["messages"].PromptTextContent(text=feedback)]),
                                    role=api["memory"].MessageRole.USER)
                                messages = self._get_msg_from_memory()
                                result.setdefault("native_feedback", []).append({
                                    "text_sha256": _hash(feedback), "chars": len(feedback),
                                    "persisted_in_native_memory": True})
                        return super()._call_llm_with_compression(messages)

                agent_class = ContinuousNativeAgent
                if compression is not None:
                    context = sdk["LLMRunContext"](memory=compression.initial_memory())
                else:
                    memory_factory = settings.get("initial_memory_factory")
                    if memory_factory is not None:
                        memory = memory_factory()
                        if memory is None:
                            raise ValueError("Section memory factory must return a private memory object")
                        context = sdk["LLMRunContext"](memory=memory)
                    else:
                        context = sdk["LLMRunContext"]()
                agent = agent_class(dep_context=context, agent_settings=agent_settings, tools=None)
                next_prompt = prompt
                recovered_error_origins = set()
                continued_exit_origins = set()
                while True:
                    result["step_invocations"] += 1
                    # A final event belongs only to this invocation. If a later
                    # continuation fails/returns empty, an earlier final cannot
                    # make that incomplete run appear successful.
                    result["native_final_event"] = False
                    result["response_text"] = ""
                    result.pop("output_truncated", None)
                    last_parse_error = None
                    first_tool = len(result["tool_calls"])
                    first_sdk_error = len(result["sdk_tool_errors"])
                    def native_events():
                        try:
                            yield from agent.step(next_prompt)
                        except _SectionHandoffComplete as exc:
                            if exc is not terminal_signal:
                                raise
                            result["termination_kind"] = ("document_terminal_finish" if settings.get("terminal_completion_tool")
                                                          else "section_terminal_handoff")

                    for event in native_events():
                        dumped = _dump_event(event)
                        result["events"].append(_event_record(event, dumped))
                        sdk_error = _sdk_tool_error(event, dumped, set(names))
                        if sdk_error is not None:
                            result["sdk_tool_errors"].append(sdk_error)
                            if _ACTIVE_SESSION.get("agent_call_id_invalid"):
                                raise AgentCallAssociationError("Trusted model-call association failed")
                            with _ACTIVE_SESSION["tool_lock"]:
                                agent_call_id = _current_agent_call_id(_ACTIVE_SESSION)
                            if agent_call_id is not None:
                                sdk_error["agent_call_id"] = agent_call_id
                            if sdk_error["code"] in {"SDK_TOOL_ARGUMENT_JSON", "SDK_TOOL_ARGUMENT_SCHEMA"}:
                                last_parse_error = (event, sdk_error)
                        if type(event).__name__ == "TextAction" and dumped.get("is_final_response") is True:
                            result["native_final_event"] = True
                            content = dumped.get("content", getattr(event, "content", ""))
                            result["response_text"] = content.strip() if isinstance(content, str) else ""
                            meta = dumped.get("response_meta")
                            if isinstance(meta, dict) and meta.get("finish_reason") == "length":
                                result["output_truncated"] = True
                    if result.get("termination_kind") in {"section_terminal_handoff", "document_terminal_finish"}:
                        break
                    if result["native_final_event"] or last_parse_error is None or not settings.get("same_agent_parse_recovery", False):
                        continuation = settings.get("continue_after_native_exit")
                        if continuation is None:
                            break
                        # SDK finality alone does not prove that an HTTP request
                        # ended. The caller must verify the actual terminal
                        # generation receipt (including empty-stop responses).
                        # Ordinary tool errors may be business-fixable; an SDK
                        # rejected batch, truncated output or unknown transport
                        # state may not be repaired by this path.
                        if (len(result["sdk_tool_errors"]) != first_sdk_error
                                or result.get("output_truncated")
                                or _ACTIVE_SESSION.get("agent_call_id_invalid")
                                or _ACTIVE_SESSION["tool_budget_exceeded"]):
                            break
                        if not _ACTIVE_SESSION["tool_lock"].acquire(blocking=False):
                            result["failure_reason"] = "business_continuation_tool_callback_pending"
                            break
                        _ACTIVE_SESSION["tool_lock"].release()
                        origin = _current_agent_call_id(_ACTIVE_SESSION)
                        if origin is None or settings["native_exit_is_settled"]() is not True:
                            result["failure_reason"] = "business_continuation_unsettled_generation"
                            break
                        if origin in continued_exit_origins:
                            result["failure_reason"] = "business_continuation_without_generation_progress"
                            break
                        state = {"step_invocation": result["step_invocations"],
                                 "agent_call_id": origin,
                                 "native_final_event": result["native_final_event"],
                                 "response_text": result["response_text"],
                                 "tool_call_count": len(result["tool_calls"]),
                                 "step_tool_call_count": len(result["tool_calls"]) - first_tool}
                        followup = continuation(state)
                        if followup is None:
                            break
                        if type(followup) is not str or not followup.strip():
                            raise ValueError("Business continuation must return nonempty text or None")
                        # No fresh model/tool/time allowance is allocated here.
                        # The gateway and this hook retain their original shared
                        # budgets. The hard bound also prevents a broken SDK that
                        # stops without spending a tool allowance from looping.
                        if result["step_invocations"] >= settings["max_calls"]:
                            result["failure_reason"] = "business_continuation_budget_exhausted"
                            break
                        settings["before_same_agent_continue"]()
                        continued_exit_origins.add(origin)
                        result["business_continuations"].append({
                            "after_step": result["step_invocations"], "agent_call_id": origin,
                            "native_final_event": result["native_final_event"],
                            "response_sha256": _hash(result["response_text"]),
                            "instruction_sha256": _hash(followup), "instruction_chars": len(followup),
                            "same_agent": True, "same_context": True, "same_memory": True})
                        next_prompt = followup
                        continue
                    event, sdk_error = last_parse_error
                    error_origin = sdk_error.get('agent_call_id') or sdk_error.get('tool_call_id_sha256')
                    if error_origin in recovered_error_origins:
                        result['failure_reason'] = 'same_agent_parse_error_without_generation_progress'
                        break
                    recovered_error_origins.add(error_origin)
                    before_continue = settings.get("before_same_agent_continue")
                    if before_continue is not None: before_continue()
                    event, sdk_error = last_parse_error
                    native_schema = (_registered_tool_schema(sdk["tool"], sdk_error["tool_name"])
                                     if sdk_error["code"] == "SDK_TOOL_ARGUMENT_SCHEMA"
                                     or sdk_error.get("adapter_parse_error_sha256") is not None else None)
                    recovery_receipt, next_prompt = _paired_parse_feedback(agent, event, sdk_error, result["tool_calls"], context=context,
                                                                         tool_schema=native_schema,
                                                                         actual_responses=_ACTIVE_SESSION.get("tool_response_snapshots"),
                                                                         argument_normalizations=result["argument_normalizations"],
                                                                         argument_parse_errors=result["argument_parse_errors"])
                    result["parse_recovery"].append(recovery_receipt)
                    settlement = settings.get("parse_recovery_generation_is_settled")
                    if settlement is not None:
                        recovery_receipt["generation_terminal_verified"] = settlement() is True
                        if not recovery_receipt["generation_terminal_verified"]:
                            raise AgentCallAssociationError("SDK parse recovery generation is not authoritatively settled")

                result["session_boundary"]["iteration_exited"] = True
                if _ACTIVE_SESSION.get("agent_call_id_invalid"):
                    result["failure_reason"] = "agent_call_id_invalid"
                elif _ACTIVE_SESSION["tool_budget_exceeded"]:
                    result["failure_reason"] = "tool_budget_exceeded"
                elif not result["tool_calls"] and settings["profile"] in {"reading", "section"}:
                    result["failure_reason"] = "sdk_tool_error" if result["sdk_tool_errors"] else "no_tool_invocation"
                elif result.get("termination_kind") in {"section_terminal_handoff", "document_terminal_finish"}:
                    pass  # Explicit program handoff, never a fabricated native final.
                elif not result["native_final_event"] or not result["response_text"]:
                    result["failure_reason"] = "missing_or_empty_native_final_response"
                elif result.get("output_truncated"):
                    result["failure_reason"] = "final_response_truncated"
                result["status"] = "failed" if "failure_reason" in result else "completed"
            finally:
                if allocation_context is not None:
                    allocation_context.__exit__(None, None, None)
                # Do not clear/rebind a still-running tool callback. A timeout
                # leaves an explicit orphan active marker, blocking all later
                # sessions until external reconciliation. SDK stop is used only
                # at the verified, fully paired section handoff boundary above.
                session = _ACTIVE_SESSION
                remaining = max(0.0, settings["timeout"] - (time.monotonic() - started))
                drained = session["tool_lock"].acquire(timeout=remaining)
                if drained:
                    try:
                        _ACTIVE_SESSION = None
                        result["session_boundary"].update(tool_callbacks_drained=True, active_session_clear=True)
                    finally:
                        session["tool_lock"].release()
                else:
                    result.update(status="failed", failure_reason="tool_callback_not_drained")
        result["session_boundary"]["session_lock_released"] = True
    except Exception as exc:
        result["error"] = _error(type(exc).__name__, "TongAgents ReAct execution failed; inspect event and tool receipts")
        result["error"]["frames"] = [
            {"file": frame.filename, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(exc.__traceback__)[-8:]
        ]
        # Do not include HTTP response bodies or request headers: either can
        # contain source text or credentials.  A status code remains useful.
        if type(getattr(exc, "status_code", None)) is int:
            result["error"]["http_status"] = exc.status_code
        if isinstance(exc, (ImportError, importlib.metadata.PackageNotFoundError)):
            result["error"]["message"] = "Official TongAgents SDK is unavailable; no fallback was executed"
        if isinstance(exc, ToolSchemaUnavailableError):
            result["failure_reason"] = "tool_schema_unavailable"
            result["error"]["message"] = "TongAgents SDK cannot expose the required native document tool schema; no agent was started"
        if isinstance(exc, AgentCallAssociationError):
            result["failure_reason"] = "agent_call_id_invalid"
            result["error"]["message"] = "Trusted local model-call association failed; no association was guessed"
        if isinstance(exc, NativeBatchRecoveryError):
            result["recovery_proof_failure_reason"] = exc.proof_reason
            result["error"]["message"] = "Native tool-batch recovery proof failed; no execution was guessed"
        result["status"] = "error"
    if result["sdk_tool_errors"] and not result["tool_calls"]:
        result.setdefault("failure_reason", "sdk_tool_error")
    result["tool_error_count"] = sum(row["status"] != "success" for row in result["tool_calls"])
    result["sdk_tool_error_count"] = len(result["sdk_tool_errors"])
    result["seconds"] = round(time.monotonic() - started, 6)
    return result
