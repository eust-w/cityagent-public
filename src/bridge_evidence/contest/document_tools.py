"""On-demand reads over an immutable evidence manifest.

Offsets are Python character offsets, with an exclusive end. With a delivery
ledger, original ``spans`` and ``context`` are pending until upstream receipt.
Legacy direct callers retain primary-span callbacks. Image observations record ``(id, 0, 1)``
separately; neither a read nor an image inventory establishes factual correctness.
Tool budgets include the JSON payload. ``native_response_budget=True`` also
counts the exact TongAgents adapter envelope; prompt tokens remain separate.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable
import zipfile

from .evidence_manifest import outline as manifest_outline
from .document_workspace import bind_image_observation
from .document_model_source import ModelSourceView
from .document_task_plan import build_document_navigation
from .tong_react_adapter import wrap_tool_response, _strict_json_loads


_TOOL_NAMES = ("read_outline", "read_source", "read_section", "read_table_rows",
               "search_in_document", "get_neighbors", "read_image")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _search_query_format_hint(query: Any) -> dict[str, Any] | None:
    """Describe a double-encoded array without changing literal search terms."""
    if type(query) is not str:
        return None
    try:
        decoded = _strict_json_loads(query)
    except (ValueError, TypeError, RecursionError):
        return None
    if type(decoded) is not list or not decoded or any(type(term) is not str for term in decoded):
        return None
    hint = {"code": "JSON_ARRAY_ENCODED_AS_STRING", "received_type": "string",
            "search_interpretation": "one_literal_phrase", "search_terms_changed": False,
            "message": "query 实际是一个字符串，本次仍按整句字面搜索，程序未修改搜索词。若意图多关键词 OR，请传真正的数组；示例只说明格式，不会自动执行。",
            "decoded_item_count": len(decoded)}
    if len(decoded) <= 16 and all(term.strip() for term in decoded):
        hint["or_query_example"] = {"query": decoded}
    else:
        hint["array_constraints"] = "OR 数组须包含 1..16 个非空短语；先明确要搜索的短语，再传真正数组。"
    return hint


def _backend_result(value: Any) -> Any:
    value = json.loads(_json(value))
    if isinstance(value, dict) and (value.get("ok") is False or value.get("error")
                                  or value.get("status") in {"failed", "error", "unavailable"}):
        raise RuntimeError("Vision/OCR backend did not produce an observation; occurrence remains unverified")
    if value is None:
        raise RuntimeError("Vision/OCR backend returned no observation")
    return value


def _recorded(method: Callable) -> Callable:
    signature = inspect.signature(method)

    @wraps(method)
    def call(self: "DocumentTools", *args: Any, **kwargs: Any) -> dict[str, Any]:
        with self._lock, self._response_budget_for(method.__name__):
            started = time.monotonic()
            started_at = time.time()
            arguments: dict[str, Any] = {}
            try:
                bound = signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
                arguments = {key: value for key, value in bound.arguments.items() if key != "self"}
                arguments = json.loads(_json(arguments))
                result = method(self, *args, **kwargs)
                result = json.loads(_json(result))
                result = self._model_result(result)
                result_tokens = self._response_tokens(result)
                if result_tokens > self.max_tool_tokens:
                    result = self._over_budget("Response exceeds budget.")
                    result_tokens = self._response_tokens(result)
                if result.get("ok") and self.delivery_ledger is not None and method.__name__ != "read_image":
                    result = self.delivery_ledger.prepare_page(result, tool=method.__name__, arguments=arguments,
                                                              receipt_id=self._response_receipt_id)
                    result_tokens = self._response_tokens(result)
                    if result_tokens > self.max_tool_tokens:
                        if result.get("source_delivery_receipt") == self._response_receipt_id:
                            self.delivery_ledger.discard_unpublished_page(self._response_receipt_id)
                        raise ValueError("Actual delivery receipt exceeds tool response budget; retry a smaller page. No navigation progress committed")
                observed_image = result.get("ok") and method.__name__ == "read_image" and result.get("status") == "observed"
                if observed_image and self.on_image_observation is not None:
                    # The complete native reply already fits. The callback must
                    # atomically persist the immutable observation and read receipt.
                    self.on_image_observation(result["image_id"], deepcopy(result))
                if result.get("ok") and self.on_read is not None:
                    if self.delivery_ledger is None:
                        for span in result.get("spans", []):
                            self.on_read(span["evidence_id"], span["start"], span["end"])
                    if observed_image and self.on_image_observation is None:
                        self.on_read(result["image_id"], 0, 1)
                if result.get("ok") and method.__name__ == "read_image" and result.get("status") == "observed":
                    if self.on_image_observation is None:
                        self.image_observations.append(deepcopy(result))
                    elif not any(item.get("observation_id") == result["observation_id"] for item in self.image_observations):
                        self.image_observations.append({key: result[key] for key in ("image_id", "observation_id", "observation_sha256")})
                        self.image_observations[-1]["storage"] = "workspace_sidecar"
                # Automatic navigation commits only after the final response
                # fits and all durable read callbacks succeeded. Low-level
                # explicit reads do not have this hook or alter navigation.
                commit = getattr(self, "_commit_recorded_success", None)
                if result.get("ok") and callable(commit):
                    commit(result)
            except Exception as exc:
                # Own vision backends expose bounded actionable errors (e.g.
                # crop/context recovery). They are errors, never cacheable observations.
                as_tool_error = getattr(exc, "as_tool_error", None)
                try:
                    structured = as_tool_error() if callable(as_tool_error) else None
                    structured = json.loads(_json(structured))
                    if (not isinstance(structured, dict) or structured.get("ok") is not False
                            or not isinstance(structured.get("error"), dict)):
                        structured = None
                except Exception:
                    structured = None
                result = structured or {"ok": False, "status": "error", "error": {
                    "type": type(exc).__name__, "message": str(exc), "recoverable": True,
                }}
                result["ok"] = False
                result["status"] = "error" if result.get("status") in {None, "observed", "success"} else result["status"]
                result["error"]["recoverable"] = True
                # Error recovery actions must use the same exposed tool
                # contract as successful pages, including precise-read aliases.
                result = self._model_result(result)
                result_tokens = self._response_tokens(result)
                if result_tokens > self.max_tool_tokens:
                    result = self._over_budget("Response exceeds budget.")
                    result_tokens = self._response_tokens(result)
            self.receipts.append({
                "tool": method.__name__, "arguments": arguments,
                "arguments_sha256": _sha(arguments), "result_sha256": _sha(result),
                "document_sha256": self.manifest["document_sha256"],
                "status": result.get("status", "success" if result.get("ok") else "error"),
                "started_at_unix": started_at,
                "seconds": round(time.monotonic() - started, 6),
                "ranges": [{key: span[key] for key in ("evidence_id", "start", "end")}
                           for span in result.get("spans", [])],
                "image_ids": [result["image_id"]] if result.get("ok") and result.get("status") == "observed" else [],
                "observation_ids": [result["observation_id"]] if result.get("ok") and result.get("status") == "observed" else [],
                # Already counted before granting coverage. Do not perform a
                # fallible tokenizer call after recording a successful read.
                "result_tokens": result_tokens,
                "result_token_scope": "native_response" if self.native_response_budget else "payload",
            })
            return result

    return call


class SourceRangeError(ValueError):
    """Controlled metadata for a known unit; never source text or a read receipt."""

    def __init__(self, evidence_id: str, actual_chars: int) -> None:
        super().__init__(f"Invalid character interval for {evidence_id}")
        self.evidence_id, self.actual_chars = evidence_id, actual_chars

    def as_tool_error(self, *, review: bool = False) -> dict[str, Any]:
        if self.actual_chars:
            next_read = {"tool": "review_read_source" if review else "read_source",
                         "arguments": {"evidence_id": self.evidence_id,
                                       "start": 0, "end": self.actual_chars}}
        else:
            # There is no nonempty source interval. Navigation remains an
            # explicit future model choice, not an automatically executed read.
            next_read = {"tool": "review_read_source_catalog", "arguments": {"cursor": 0, "limit": 10}} if review else {
                "tool": "read_outline", "arguments": {}}
        return {"ok": False, "status": "error", "error": {
                    "type": "ValueError", "message": str(self), "recoverable": True,
                    "code": "SOURCE_RANGE", "cause": "invalid_character_interval"},
                "source_range": {"evidence_id": self.evidence_id, "actual_chars": self.actual_chars,
                    "legal_half_open_bounds": {"start": 0, "end": self.actual_chars},
                    "offset_unit": "Unicode characters", "end_exclusive": True,
                    "nonempty_read_available": self.actual_chars > 0,
                    "valid_interval": "0 <= start < end <= actual_chars" if review else "0 <= start <= end <= actual_chars"},
                "next_action": "Choose a valid local character interval using these bounds, or inspect navigation. No source content was read by this failed call.",
                "next_read": next_read}


class DocumentTools:
    """Bounded tools; sections use heading IDs and ``root`` needs no headings.

    Native response budgeting requires the same tokenizer and token limit as
    the TongAgents adapter. Direct consumers still receive a bare payload;
    for them enabling this option is conservative rather than a wire change.
    A section cursor is a zero-based unit index *within that section*.  An
    oversized unit is continued with ``read_source``; its continuation includes
    ``resume_section`` so the next section cursor cannot skip unread characters.
    Oversized automatic context uses an explicit recovery section. Its first
    entry is a lossless, paginated context manifest; subsequent entries read the
    referenced context and requested primary text separately. Recovery calls use
    the same public read_source/read_section signatures and survive reconstruction
    of DocumentTools from the immutable manifest.
    The image backend may implement ``recognize(bytes, region=...)`` and
    ``interpret(observation, question=..., context=..., image=...)``.  A callable
    backend instead receives bytes plus those keyword arguments, and is cached
    only with the full occurrence/question/context identity.
    """

    def __init__(self, manifest: dict[str, Any], *, count_tokens: Callable[[str], int],
                 max_tool_tokens: int = 4096, source_path: str | Path | None = None,
                 vision_backend: Any = None, on_read: Callable[[str, int, int], None] | None = None,
                 native_response_budget: bool = False,
                 compact_model_source: bool = False,
                 on_image_observation: Callable[[str, dict], None] | None = None,
                 task: dict[str, Any] | None = None,
                 delivery_ledger: Any = None, material_index: Any = None) -> None:
        if type(max_tool_tokens) is not int or max_tool_tokens <= 0:
            raise ValueError("max_tool_tokens must be a positive integer")
        if type(native_response_budget) is not bool:
            raise ValueError("native_response_budget must be a boolean")
        if type(compact_model_source) is not bool:
            raise ValueError("compact_model_source must be a boolean")
        self.manifest = deepcopy(manifest)
        self.delivery_ledger = delivery_ledger
        if delivery_ledger is not None and delivery_ledger.manifest != manifest:
            raise ValueError("Source delivery ledger does not match DocumentTools manifest")
        self.count_tokens = count_tokens
        self.max_tool_tokens = max_tool_tokens
        self.native_response_budget = native_response_budget
        self.compact_model_source = compact_model_source
        self.model_source = ModelSourceView(self.manifest)
        self._response_tool: str | None = None
        self._response_projection: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._automatic_navigator: Any = None
        self.source_path = Path(source_path or manifest.get("source_path", ""))
        self.vision_backend = vision_backend
        self.on_read = on_read
        if on_image_observation is not None and not callable(on_image_observation):
            raise ValueError("on_image_observation must be callable")
        if on_image_observation is not None and not native_response_budget:
            raise ValueError("Persisted image delivery requires native_response_budget=True")
        self.on_image_observation = on_image_observation
        self.receipts: list[dict[str, Any]] = []
        # Legacy consumers retain complete returned evidence. With a durable
        # callback, only deduplicated IDs/SHA stay here; full content is loaded
        # from the workspace sidecar for semantic review. Per-call tool receipts
        # retain actual reply hashes, including repeated cache-hit deliveries.
        self.image_observations: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._units = self.manifest["units"]
        self._index = {unit["id"]: unit for unit in self._units}
        if len(self._index) != len(self._units):
            raise ValueError("Duplicate evidence IDs")
        self._positions = {unit["id"]: index for index, unit in enumerate(self._units)}
        self._tables = {table["id"]: table for table in self.manifest.get("tables", [])}
        self._table_units = {table_id: [] for table_id in self._tables}
        for unit in self._units:
            if unit.get("table_id"):
                self._table_units[unit["table_id"]].append(unit)
        self._images = {image["id"]: image for image in self.manifest.get("images", [])}
        self._base_cache: dict[str, Any] = {}
        self._judgment_cache: dict[str, Any] = {}
        self._recovery_cache: dict[str, list[dict[str, Any]]] = {}
        self.navigation = build_document_navigation(self.manifest, material_index=material_index)
        self._sections = self._build_sections()
        self.task = deepcopy(task)
        self._task_id = None
        section_metadata = [{**section, "unit_count": len(self._sections[section["id"]])}
                            for section in self.navigation["sections"]]
        if task is not None:
            if (not isinstance(task, dict) or not isinstance(task.get("id"), str) or not task["id"]
                    or task["id"] in self._sections or task["id"] in self._index
                    or not isinstance(task.get("primary_evidence_ids"), list)
                    or any(not isinstance(eid, str) or eid not in self._index for eid in task["primary_evidence_ids"])
                    or len(set(task["primary_evidence_ids"])) != len(task["primary_evidence_ids"])
                    or (task.get("document_sha256", self.manifest["document_sha256"]) != self.manifest["document_sha256"])):
                raise ValueError("Task must have a unique navigation ID and original primary evidence IDs")
            self._task_id = task["id"]
            self._sections[self._task_id] = list(task["primary_evidence_ids"])
            primary = set(task["primary_evidence_ids"])
            related = [section for section in section_metadata if section["id"] != "root"
                       and primary.intersection(self._sections[section["id"]])]
            remaining = [section for section in section_metadata if section not in related]
            section_metadata = [{"id": self._task_id, "title": "Assigned reading task",
                                 "basis": "primary_assignment_not_semantic_coverage",
                                 "unit_count": len(primary)}, *related, *remaining]
        self._heading_ids = {item["evidence_id"] for item in self.navigation["headings"]}
        self.outline = {**manifest_outline(self.manifest), "sections": section_metadata,
                        "headings": deepcopy(self.navigation["headings"]), "images": list(self._images.values())}
        self.material_tools = None
        if material_index is not None:
            from .document_material_tools import DocumentMaterialTools
            self.material_tools = DocumentMaterialTools(self, material_index)
            material_sections = material_index.compact_outline()
            if isinstance(material_sections, dict):
                material_sections = material_sections.get("sections", [])
            for section_id in material_index.section_ids:
                if section_id in self._sections or section_id in self._index or section_id in self._tables:
                    raise ValueError("Material section identity collides with canonical navigation")
                # Unresolved material sections remain discoverable; their
                # read returns explicit fallback rather than guessing a span.
                self._sections[section_id] = list(dict.fromkeys(
                    span["evidence_id"] for span in material_index.section(section_id).get("spans", [])))
            from .document_material_tools import _compact
            self.outline["sections"] = [*_compact(material_sections), *self.outline["sections"]]
        # Error replies also have to fit.  Reject unusable configurations rather
        # than silently breaking the declared response budget.
        for name in _TOOL_NAMES if native_response_budget else ("read_source",):
            with self._response_budget_for(name):
                if not self._fits(self._over_budget("Response exceeds budget.")):
                    raise ValueError("max_tool_tokens cannot fit a minimal error receipt")

    def handlers(self) -> dict[str, Callable]:
        return {name: getattr(self, name) for name in _TOOL_NAMES}

    def automatic_handlers(self, *, state: dict[str, Any] | None = None,
                           on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Callable]:
        """Model-facing reads without positions; explicit ``handlers`` remain available.

        Restore only this document's previously persisted navigation snapshot.
        ``on_progress`` must durably save the supplied snapshot or raise; a
        failed persistence callback never advances the in-memory position.
        """
        from .document_read_navigation import DocumentReadNavigation
        with self._lock:
            if self._automatic_navigator is None:
                self._automatic_navigator = DocumentReadNavigation(self, state=state, on_progress=on_progress)
            elif state is not None:
                raise ValueError("Automatic navigation is already initialized; restore on a new DocumentTools instance")
            elif on_progress is not None:
                self._automatic_navigator.on_progress = on_progress
            return self._automatic_navigator.handlers()

    def navigation_state(self) -> dict[str, Any]:
        """Accurate delivery positions; not semantic understanding or extraction."""
        if self._automatic_navigator is None:
            state = {"schema_version": 1, "document_sha256": self.manifest["document_sha256"], "objects": {}}
            if self.delivery_ledger is not None:
                state["source_delivery_identity"] = deepcopy(self.delivery_ledger.identity)
            if self.material_tools is not None:
                state["material_identity"] = deepcopy(self.material_tools.index.identity)
            return state
        return self._automatic_navigator.snapshot()

    @contextmanager
    def _response_budget_for(self, name: str):
        previous = self._response_tool
        previous_receipt = getattr(self, "_response_receipt_id", None)
        self._response_tool = name
        self._response_receipt_id = (self.delivery_ledger.reserve_receipt_id()
                                     if self.delivery_ledger is not None and name != "read_image" else None)
        try:
            yield
        finally:
            self._response_tool = previous
            self._response_receipt_id = previous_receipt

    def _response_tokens(self, result: dict[str, Any]) -> int:
        result = self._model_result(result)
        if self.delivery_ledger is not None and result.get("ok") and self._response_tool != "read_image":
            # Reserve the actual receipt envelope during every page-size probe,
            # before an outbox page or a navigation position can be committed.
            result = deepcopy(result)
            if self._response_receipt_id is None:
                raise RuntimeError("Source response budget lacks an actual receipt reservation")
            result.setdefault("source_delivery_receipt", self._response_receipt_id)
            result.setdefault("replayed", False)
        if self.native_response_budget:
            if self._response_tool is None:
                raise RuntimeError("Native response counting requires an active tool name")
            result = wrap_tool_response(self._response_tool, result)
        count = self.count_tokens(_json(result))
        if type(count) is not int or count < 0:
            raise ValueError("count_tokens must return a non-negative integer")
        return count

    def _model_result(self, result: dict[str, Any]) -> dict[str, Any]:
        # Image admission/observation hashes use the original representation.
        # The text-only switch must not change their persistence protocol.
        if self.compact_model_source and self._response_tool != "read_image":
            result = self.model_source.project(result)
        if self._response_projection is not None:
            result = self._response_projection(result)
        if self.delivery_ledger is not None and result.get("ok") and self._response_tool != "read_image":
            def without_legacy_coverage(value):
                if isinstance(value, dict):
                    return {key: without_legacy_coverage(item) for key, item in value.items() if key != "grants_coverage"}
                if isinstance(value, list):
                    return [without_legacy_coverage(item) for item in value]
                return value
            result = without_legacy_coverage(result)
            result.update(coverage_confirmation="await_upstream_receipt", semantic_review=False)
        return result

    def _fits(self, result: dict[str, Any]) -> bool:
        return self._response_tokens(result) <= self.max_tool_tokens

    @staticmethod
    def _over_budget(message: str, recovery: Any = None) -> dict[str, Any]:
        result = {"ok": False, "status": "budget_exceeded", "error": {
            "type": "ToolBudgetExceeded", "message": message, "recoverable": True,
        }}
        if recovery is not None:
            result["recovery"] = recovery
        return result

    def _build_sections(self) -> dict[str, list[str]]:
        return deepcopy(self.navigation["section_evidence_ids"])

    def full_text(self) -> str:
        """Lossless annotated text for admission/token counting; not a read receipt."""
        if self.compact_model_source:
            return self.model_source.full_text()
        return "\n\n".join(
            f"[{unit['id']} chars 0:{len(unit['text'])}; source={unit['source_path']}]\n{unit['text']}"
            for unit in self._units
        )

    def _span(self, evidence_id: str, start: int = 0, end: int | None = None) -> dict[str, Any]:
        unit = self._index[evidence_id]
        if end is None:
            end = len(unit["text"])
        if type(start) is not int or type(end) is not int:
            raise SourceRangeError(evidence_id, len(unit["text"]))
        if not 0 <= start <= end <= len(unit["text"]):
            raise SourceRangeError(evidence_id, len(unit["text"]))
        result = {"evidence_id": evidence_id, "start": start, "end": end,
                  "text": unit["text"][start:end], "source_path": unit["source_path"]}
        if unit.get("table_id"):
            result.update(table_id=unit["table_id"], row_index=unit["row_index"])
            # Cell coordinates and inherited values retain instance provenance;
            # these are context, not added characters in the primary interval.
            result["cell_context"] = [{key: cell[key] for key in (
                "column_index", "column_span", "vertical_merge", "merged_from_source", "merged_context_text", "source_path",
            ) if key in cell} for cell in unit.get("cells", [])]
        return result

    def _read_call(self, evidence_id: str, start: int, end: int) -> dict[str, Any]:
        return {"tool": "read_source", "arguments": {"evidence_id": evidence_id, "start": start, "end": end}}

    def _context(self, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Explicit excerpts for neighbors; full table headers and boundary notes.

        Boundary paragraphs are labeled as such, never classified as true
        captions/notes by a heuristic.  Further context remains discoverable
        through source references and section reads.
        """
        primary = {span["evidence_id"] for span in spans}
        wanted: dict[str, tuple[str, bool]] = {}
        for span in spans:
            unit = self._index[span["evidence_id"]]
            story = unit.get("context", {}).get("story")
            position = self._positions[unit["id"]]
            for j in (position - 1, position + 1):
                if 0 <= j < len(self._units):
                    neighbor = self._units[j]
                    if neighbor.get("context", {}).get("story") == story:
                        wanted.setdefault(neighbor["id"], ("adjacent", False))
            heading = unit.get("context", {}).get("section_heading_id")
            if heading in self._index:
                wanted.setdefault(heading, ("section_heading", False))
            for heading in self.navigation["heading_paths"][position]:
                wanted.setdefault(heading, ("section_heading_navigation_candidate", False))
            table_id = unit.get("table_id")
            while table_id:
                table = self._tables[table_id]
                for header in table.get("header_evidence_ids", []):
                    wanted[header] = (f"table_header:{table.get('header_kind', 'unknown')}", True)
                positions = [self._positions[item["id"]] for item in self._table_units[table_id]]
                if positions:
                    for direction, edge in ((-1, min(positions)), (1, max(positions))):
                        j = edge + direction
                        while 0 <= j < len(self._units):
                            item = self._units[j]
                            if (item.get("context", {}).get("story") != story or item.get("table_id")
                                    or item["id"] in self._heading_ids):
                                break
                            wanted[item["id"]] = ("table_boundary_context", True)
                            # Carry every consecutive annotation after a table;
                            # otherwise carry the immediately adjacent paragraph.
                            annotation = re.match(r"\s*(?:注|备注|说明|单位|Note\b)", item["text"], re.I)
                            if item["text"].strip() and not annotation:
                                break
                            j += direction
                table_id = table.get("parent_table_id")
        result = []
        for evidence_id, (role, complete) in wanted.items():
            if evidence_id in primary:
                continue
            text = self._index[evidence_id]["text"]
            end = len(text) if complete else min(120, len(text))
            context = self._span(evidence_id, 0, end)
            context.update(role=role, grants_coverage=False)
            if end < len(text):
                context["continuation"] = self._read_call(evidence_id, end, len(text))
            result.append(context)
        return result

    def _page(self, spans: list[dict[str, Any]], continuation: Any = None, **extra: Any) -> dict[str, Any]:
        return {"ok": True, "status": "read", "spans": spans, "context": self._context(spans),
                "continuation": continuation, **extra}

    def _recovery_entries(self, section_id: str) -> list[dict[str, Any]]:
        """Deterministic source-addressed recovery, without recursive context.

        Metadata retains every column/merge/source relationship outside the
        inline envelope. Metadata fragments are never original-text coverage.
        Text entries use the original evidence IDs and exact character ranges.
        """
        if section_id.startswith("@material/"):
            if self.material_tools is None:
                raise ValueError("Material continuation requires its original bound index")
            return self.material_tools.entries(section_id)
        if section_id in self._recovery_cache:
            return self._recovery_cache[section_id]
        kind, identifier, first, last = section_id.split("/")
        start, end = int(first), int(last)
        if kind == "@source":
            primary = [self._span(identifier, start, end)]
        elif kind == "@table":
            table = self._tables[identifier]
            if not 0 <= start < end <= table["row_count"]:
                raise ValueError("Invalid recovery table range")
            primary = [self._span(unit["id"]) for unit in self._table_units[identifier]
                       if start <= unit["row_index"] < end]
        else:
            raise ValueError("Unknown recovery section")
        if not primary:
            raise ValueError("Recovery section has no original source")
        primary_ids = {span["evidence_id"] for span in primary}
        contexts = self._context(primary)
        context_ids = {span["evidence_id"] for span in contexts}
        # A merged cell's inherited value must keep its exact origin, including
        # origins farther away than the ordinary neighbor window.
        merged_sources = {cell.get("merged_from_source") for span in primary
                          for cell in span.get("cell_context", []) if cell.get("merged_from_source")}
        for unit in self._units:
            if unit["id"] in primary_ids | context_ids:
                continue
            if any(cell.get("source_path") in merged_sources for cell in unit.get("cells", [])):
                contexts.append({**self._span(unit["id"]), "role": "merged_cell_origin", "grants_coverage": False})
                context_ids.add(unit["id"])
        # A bounded adjacent excerpt has an explicit full-source continuation in
        # normal reads. Recovery follows that continuation too, rather than
        # silently leaving the rest of an annotation behind.
        contexts = [{**self._span(span["evidence_id"]), "role": span["role"], "grants_coverage": False}
                    for span in contexts]
        table_ids = {span["table_id"] for span in [*primary, *contexts] if span.get("table_id")}
        for table_id in list(table_ids):
            parent = self._tables[table_id].get("parent_table_id")
            while parent:
                table_ids.add(parent)
                parent = self._tables[parent].get("parent_table_id")
        metadata = {
            "schema_version": 1, "recovery_section": section_id,
            "primary": [{key: value for key, value in span.items() if key != "text"} for span in primary],
            "context": [{key: value for key, value in span.items() if key != "text"} for span in contexts],
            "tables": [deepcopy(self._tables[key]) for key in sorted(table_ids)],
            "nested_table_ids": [table["id"] for table in self._tables.values()
                                 if table.get("parent_table_id") in table_ids],
            "coverage": "Metadata is not a source read. Continue to read referenced original context and primary ranges.",
        }
        metadata_text = _json(self.model_source.project(metadata) if self.compact_model_source else metadata)
        entries = [{"kind": "context_manifest", "text": metadata_text, "start": 0, "end": len(metadata_text)}]
        entries += [{"kind": "original", "evidence_id": span["evidence_id"],
                     "start": span["start"], "end": span["end"], "role": span.get("role", "primary")}
                    for span in [*contexts, *primary]]
        self._recovery_cache[section_id] = entries
        return entries

    @staticmethod
    def _recovery_piece_id(section_id: str, index: int) -> str:
        return f"@piece/{section_id}/{index}"

    def _recovery_start(self, recovery_section_id: str, **extra: Any) -> dict[str, Any]:
        result = self._over_budget("Inline context exceeds budget. Follow continuation for full context, then original text.")
        result.update(continuation={"tool": "read_section", "arguments": {
            "section_id": recovery_section_id, "cursor": 0, "max_units": 1}}, **extra)
        if not self._fits(result):
            return self._recovery_budget_error()
        return result

    def _recovery_budget_error(self) -> dict[str, Any]:
        # Shorter than the minimum error envelope checked by __init__.
        # A configuration change is required; repeating the read cannot help.
        result = self._over_budget("Increase tool budget.")
        result["error"]["recoverable"] = False
        return result

    def _recovery_piece(self, section_id: str, index: int, start: int, end: int | None) -> dict[str, Any]:
        entries = self._recovery_entries(section_id)
        if not 0 <= index < len(entries):
            raise ValueError("Invalid recovery entry")
        entry = entries[index]
        if end is None:
            end = entry["end"]
        if type(start) is not int or type(end) is not int or not entry["start"] <= start <= end <= entry["end"]:
            raise ValueError("Invalid recovery character range")
        evidence_id = self._recovery_piece_id(section_id, index)
        following = ({"tool": "read_section", "arguments": {
            "section_id": section_id, "cursor": index + 1, "max_units": 1}}
            if index + 1 < len(entries) else None)
        if end < entry["end"]:
            following = self._read_call(evidence_id, end, entry["end"])

        def page(stop: int) -> dict[str, Any]:
            continuation = self._read_call(evidence_id, stop, end) if stop < end else following
            if entry["kind"] == "context_manifest":
                return {"ok": True, "status": "read_context", "spans": [],
                        "context_part": {"context_id": evidence_id, "start": start, "end": stop,
                                         "total_chars": len(entry["text"]), "text": entry["text"][start:stop]},
                        "grants_coverage": False, "continuation": continuation}
            unit = self._index[entry["evidence_id"]]
            return {"ok": True, "status": "read", "spans": [{
                "evidence_id": unit["id"], "start": start, "end": stop, "text": unit["text"][start:stop]}],
                "context_reference": self._read_call(self._recovery_piece_id(section_id, 0), 0, len(entries[0]["text"])),
                "source_role": entry["role"], "continuation": continuation}

        complete = page(end)
        if self._fits(complete):
            return complete
        low, high, best = start + 1, end - 1, None
        while low <= high:
            middle = (low + high) // 2
            candidate = page(middle)
            if self._fits(candidate):
                best, low = candidate, middle + 1
            else:
                high = middle - 1
        if best is not None:
            return best
        # Tiny budgets may not fit even the protocol envelope. This is a
        # configuration limit, not another instruction to repeat the same read.
        return self._recovery_budget_error()

    def _recovery_section(self, section_id: str, cursor: int, max_units: int) -> dict[str, Any]:
        if section_id.startswith("@material/"):
            if self.material_tools is None:
                raise ValueError("Material continuation requires its original bound index")
            return self.material_tools.page(section_id, cursor, max_units)
        entries = self._recovery_entries(section_id)
        if type(cursor) is not int or not 0 <= cursor <= len(entries) or type(max_units) is not int or max_units <= 0:
            raise ValueError("Invalid recovery section cursor or max_units")
        if cursor == len(entries):
            return {"ok": True, "status": "read", "spans": [], "continuation": None}
        entry = entries[cursor]
        return self._recovery_piece(section_id, cursor, entry["start"], entry["end"])

    def _source(self, evidence_id: str, start: int, end: int, **extra: Any) -> dict[str, Any]:
        full = self._span(evidence_id, start, end)
        result = self._page([full], **extra)
        if self._fits(result):
            return result
        # Binary search finds a safe prefix.  Exact token counts are checked on
        # every returned payload; maximality is not required of a tokenizer.
        low, high = start + 1, end - 1
        best = None
        while low <= high:
            middle = (low + high) // 2
            candidate = self._page([self._span(evidence_id, start, middle)],
                                   self._read_call(evidence_id, middle, end), **extra)
            if self._fits(candidate):
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        if best is not None:
            return best
        return self._recovery_start(f"@source/{evidence_id}/{start}/{end}", **extra)

    @_recorded
    def read_outline(self, cursor: int = 0, limit: int = 20) -> dict[str, Any]:
        """Page a shared section/table/image inventory; this is not a read.

        Cursor addresses inventory entries, not source units. Assigned sections
        come first, but root and all other chapters remain explicitly available.
        """
        entries = ([('sections', section) for section in self.outline["sections"]]
                   + [('tables', table) for table in self._tables.values()]
                   + [('images', image) for image in self._images.values()])
        if (type(cursor) is not int or not 0 <= cursor <= len(entries)
                or type(limit) is not int or not 1 <= limit <= 100):
            raise ValueError("cursor must address an inventory entry and limit must be 1..100")

        def page(items: list[tuple[str, dict]], next_cursor: int) -> dict[str, Any]:
            result = {"ok": True, "status": "outline", "unit_count": len(self._units),
                      "table_count": len(self._tables), "image_count": len(self._images),
                      "total_items": len(entries), "cursor": cursor,
                      "default_search_scope": "task" if self._task_id else "document",
                      "outline_complete": next_cursor == len(entries),
                      "continuation": ({"tool": "read_outline", "arguments": {"cursor": next_cursor, "limit": limit}}
                                       if next_cursor < len(entries) else None)}
            for kind, item in items:
                result.setdefault(kind, []).append(item)
            return result

        items: list[tuple[str, dict]] = []
        result = page(items, cursor)
        for index in range(cursor, min(len(entries), cursor + limit)):
            kind, item = entries[index]
            candidate = page([*items, (kind, item)], index + 1)
            if not self._fits(candidate):
                if items:
                    break
                # Do not truncate a source title/header to fake a complete
                # inventory item. Keep its address and an explicit full read.
                compact = {key: item[key] for key in ("id", "unit_count", "basis", "parent_table_id", "row_count")
                           if key in item}
                compact["metadata_complete"] = False
                if item["id"] in self._index:
                    compact["read"] = {"tool": "read_source", "arguments": {"evidence_id": item["id"]}}
                elif self.material_tools is not None and self.material_tools.looks_like_ref(item["id"]):
                    compact["read"] = {"tool": "read_section", "arguments": {"section_id": item["id"]}}
                elif kind == "tables" and item.get("row_count", 0):
                    compact["read"] = {"tool": "read_table_rows", "arguments": {
                        "table_id": item["id"], "start": 0, "end": 1}}
                candidate = page([(kind, compact)], index + 1)
                if not self._fits(candidate):
                    return self._over_budget("One outline address exceeds budget; increase the tool budget.")
            items = [(key, value) for key in ("sections", "tables", "images")
                     for value in candidate.get(key, [])]
            result = candidate
        return result

    @_recorded
    def read_source(self, evidence_id: str, start: int = 0, end: int | None = None) -> dict[str, Any]:
        """Read a whole source unit by default, or an exact explicit interval.

        Long reads return concrete character-range continuations. Recovery
        piece IDs retain their original bounds; no invalid range is clamped.
        """
        if evidence_id.startswith("@piece/"):
            section_id, index = evidence_id[len("@piece/"):].rsplit("/", 1)
            return self._recovery_piece(section_id, int(index), start, end)
        if self.material_tools is not None and self.material_tools.looks_like_ref(evidence_id):
            return self.material_tools.read_source(evidence_id, start, end)
        if end is None:
            end = len(self._index[evidence_id]["text"])
        return self._source(evidence_id, start, end)

    @_recorded
    def read_section(self, section_id: str, cursor: int, max_units: int) -> dict[str, Any]:
        if section_id.startswith(("@source/", "@table/", "@material/")):
            return self._recovery_section(section_id, cursor, max_units)
        if self.material_tools is not None and self.material_tools.looks_like_ref(section_id):
            return self.material_tools.page(self.material_tools.section_id(section_id), cursor, max_units)
        if section_id in self._index and section_id not in self._sections:
            return {"ok": False, "status": "error", "error": {
                "type": "ValueError", "code": "SECTION_EXPECTS_HEADING_ID",
                "message": "This evidence ID identifies a source unit, not a section heading.",
                "recoverable": True,
            }, "next_read": {"tool": "read_source", "arguments": {"evidence_id": section_id}},
                "next_action": "Read this exact source unit with read_source, or inspect read_outline for section IDs. This failed call did not read any source content."}
        ids = self._sections[section_id]
        if type(cursor) is not int or not 0 <= cursor <= len(ids) or type(max_units) is not int or max_units <= 0:
            raise ValueError("cursor must address a section unit and max_units must be positive")
        spans = []
        result = self._page([], section_id=section_id)
        for index in range(cursor, min(len(ids), cursor + max_units)):
            continuation = ({"tool": "read_section", "arguments": {
                "section_id": section_id, "cursor": index + 1, "max_units": max_units,
            }} if index + 1 < len(ids) else None)
            candidate = self._page([*spans, self._span(ids[index])], continuation, section_id=section_id)
            if not self._fits(candidate):
                if spans:
                    break
                result = self._source(ids[index], 0, len(self._index[ids[index]]["text"]),
                                      section_id=section_id, resume_section=continuation)
                return result
            spans.append(self._span(ids[index]))
            result = candidate
        return result

    def canonical_table_id(self, table_id: str) -> str:
        if self.material_tools is not None and self.material_tools.looks_like_ref(table_id):
            return self.material_tools.canonical_table_id(table_id)
        if table_id not in self._tables:
            raise ValueError("Unknown canonical table ID")
        return table_id

    @_recorded
    def read_table_rows(self, table_id: str, start: int, end: int) -> dict[str, Any]:
        table_id = self.canonical_table_id(table_id)
        table = self._tables[table_id]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= table["row_count"]:
            raise ValueError("Rows must satisfy 0 <= start < end <= row_count")
        spans: list[dict[str, Any]] = []
        result = None
        for row in range(start, end):
            units = [unit for unit in self._table_units[table_id] if unit["row_index"] == row]
            # Nested tables remain separate evidence instances, with explicit
            # navigation references rather than fabricated parent-row text.
            candidate_spans = [*spans, *(self._span(unit["id"]) for unit in units)]
            continuation = ({"tool": "read_table_rows", "arguments": {"table_id": table_id, "start": row + 1, "end": end}}
                            if row + 1 < end else None)
            nested = [item["id"] for item in self._tables.values() if item.get("parent_table_id") == table_id]
            candidate = self._page(candidate_spans, continuation, table_id=table_id,
                                   requested_rows=[start, end], returned_rows=[start, row + 1], nested_table_ids=nested)
            if not self._fits(candidate):
                if result is not None:
                    return result
                return self._recovery_start(f"@table/{table_id}/{row}/{end}")
            spans = candidate_spans
            result = candidate
        return result or self._over_budget("No row evidence available")

    @_recorded
    def search_in_document(self, query: str | list[str], scope: str = "default",
                           section_id: str | None = None, cursor: int = 0,
                           max_matches: int = 8) -> dict[str, Any]:
        """Literal case-insensitive OR search; excerpts do not grant coverage."""
        terms = [query] if isinstance(query, str) else query
        if (not isinstance(terms, list) or not 1 <= len(terms) <= 16
                or any(not isinstance(term, str) or not term.strip() for term in terms)):
            raise ValueError("query must be a nonblank phrase or 1..16 nonblank literal phrases")
        terms = list(dict.fromkeys(terms))
        query_format_hint = _search_query_format_hint(query)
        if scope not in {"default", "task", "document", "section"}:
            raise ValueError("scope must be default, task, document, or section")
        if section_id is not None and (not isinstance(section_id, str) or scope not in {"default", "section"}):
            raise ValueError("section_id is only valid with default or section scope")
        if scope == "section" and section_id is None:
            raise ValueError("section scope requires section_id from read_outline")
        material_ranges = None
        if (section_id is not None and self.material_tools is not None
                and self.material_tools.looks_like_ref(section_id)):
            local_section = self.material_tools.normalize(section_id)
            if local_section not in self.material_tools.index.section_ids:
                raise ValueError("Material search scope requires a section reference")
            material_ranges = {}
            for span in self.material_tools.canonical_spans(local_section):
                material_ranges.setdefault(span["evidence_id"], []).append((span["start"], span["end"]))
            effective, ids = "section", list(material_ranges)
        elif section_id is not None:
            if section_id not in self._sections:
                raise ValueError("Unknown section_id; use read_outline or read_source for a source unit")
            effective, ids = "section", self._sections[section_id]
        elif scope == "task" or (scope == "default" and self._task_id is not None):
            if self._task_id is None:
                raise ValueError("task scope requires an assigned reading task")
            effective, ids = "task", self._sections[self._task_id]
        else:
            effective, ids = "document", self._sections["root"]
        if type(max_matches) is not int or not 1 <= max_matches <= 100:
            raise ValueError("max_matches must be 1..100")
        allowed = set(ids)
        patterns = [re.compile(re.escape(term), re.I) for term in terms]
        matches = []
        for unit in self._units:
            if unit["id"] in allowed:
                occurrences = {}
                for term, pattern in zip(terms, patterns):
                    for match in pattern.finditer(unit["text"]):
                        if material_ranges is None or any(a <= match.start() < match.end() <= b
                                for a, b in material_ranges.get(unit["id"], [])):
                            occurrences.setdefault((match.start(), match.end()), term)
                matches.extend((unit, start, end, occurrences[start, end]) for start, end in sorted(occurrences))
        if type(cursor) is not int or not 0 <= cursor <= len(matches):
            raise ValueError("cursor must address a search occurrence")
        hits: list[dict[str, Any]] = []
        def page(rows: list[dict[str, Any]], next_cursor: int) -> dict[str, Any]:
            arguments: dict[str, Any] = {"query": query, "cursor": next_cursor}
            if scope != "default":
                arguments["scope"] = scope
            if section_id is not None:
                arguments["section_id"] = section_id
            if max_matches != 8:
                arguments["max_matches"] = max_matches
            result = {"ok": True, "status": "search", "matches": rows, "scope": effective,
                      "total_matches": len(matches), "omitted_matches": len(matches) - next_cursor,
                      "cursor": cursor, "grants_coverage": False,
                      "continuation": ({"tool": "search_in_document", "arguments": arguments}
                                       if next_cursor < len(matches) else None)}
            if section_id is not None:
                result["section_id"] = section_id
            if query_format_hint is not None:
                result["query_format_hint"] = deepcopy(query_format_hint)
                if "or_query_example" in result["query_format_hint"] and not self._fits(result):
                    # Large example terms must not replace original-text hits
                    # or be clipped into different suggested search phrases.
                    result["query_format_hint"].pop("or_query_example")
                    result["query_format_hint"]["example_omitted_for_budget"] = True
            # Preserve required header/note/heading addresses even when their
            # full original text must be read in a separate bounded call.
            context = [item for item in self._context(rows) if item["role"] != "adjacent"]
            origins = {cell.get("merged_from_source") for row in rows for cell in row.get("cell_context", [])
                       if cell.get("merged_from_source")}
            present = {row["evidence_id"] for row in [*rows, *context]}
            for unit in self._units:
                if unit["id"] not in present and any(cell.get("source_path") in origins for cell in unit.get("cells", [])):
                    context.append({**self._span(unit["id"]), "role": "merged_cell_origin", "grants_coverage": False})
            if context:
                result["context"] = [{"evidence_id": item["evidence_id"], "role": item["role"],
                                      "read": self._read_call(item["evidence_id"], 0, len(self._index[item["evidence_id"]]["text"]))}
                                     for item in context]
                if self._fits(result):
                    for index, item in enumerate(context):
                        candidate = deepcopy(result)
                        candidate["context"][index] = {**item, "read": result["context"][index]["read"]}
                        if self._fits(candidate):
                            result = candidate
            return result

        result = page([], cursor)
        for index in range(cursor, min(len(matches), cursor + max_matches)):
            unit, match_start, match_end, term = matches[index]
            start, end = max(0, match_start - 40), min(len(unit["text"]), match_end + 40)
            if material_ranges is not None:
                left, right = next((a, b) for a, b in material_ranges[unit["id"]]
                                   if a <= match_start < match_end <= b)
                start, end = max(start, left), min(end, right)
            hit = {**self._span(unit["id"], start, end), "match_start": match_start, "match_end": match_end,
                   "matched_query": term, "read": self._read_call(unit["id"], 0, len(unit["text"]))}
            if self.material_tools is not None:
                hit["material_refs"] = self.material_tools.aliases_for(unit["id"])
            candidate = page([*hits, hit], index + 1)
            if not self._fits(candidate):
                break
            hits.append(hit)
            result = candidate
        if cursor < len(matches) and not hits:
            unit = matches[cursor][0]
            return self._over_budget("Search excerpt/context addresses exceed budget; read the exact source with context pagination.",
                                     self._read_call(unit["id"], 0, len(unit["text"])))
        return result

    @_recorded
    def get_neighbors(self, evidence_id: str) -> dict[str, Any]:
        if self.material_tools is not None and self.material_tools.looks_like_ref(evidence_id):
            return self.material_tools.read_source(evidence_id)
        position = self._positions[evidence_id]
        story = self._index[evidence_id].get("context", {}).get("story")
        spans = [self._span(unit["id"]) for unit in self._units[max(0, position - 1):position + 2]
                 if unit.get("context", {}).get("story") == story]
        result = self._page(spans)
        if self._fits(result):
            return result
        return self._over_budget("Neighbors exceed the budget; read each source with character pagination.",
                                 [self._read_call(span["evidence_id"], span["start"], span["end"]) for span in spans])

    def _image_bytes(self, image: dict[str, Any]) -> tuple[bytes, str]:
        if self.manifest.get("source_kind") != "docx":
            raise ValueError("Image bytes require the original DOCX")
        blob = self.source_path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != self.manifest["document_sha256"]:
            raise ValueError("Source DOCX hash does not match the manifest")
        target = image.get("target", "")
        part = image["source_path"].split("#", 1)[0]
        if not target or "\\" in target or target.startswith("/") or re.match(r"^[a-zA-Z][\w+.-]*:", target):
            raise ValueError("External or unsafe image target is unavailable")
        member = posixpath.normpath(posixpath.join(posixpath.dirname(part), target))
        if member.startswith("../") or member in {"..", "."} or PurePosixPath(member).is_absolute():
            raise ValueError("Unsafe image ZIP member")
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            if archive.namelist().count(member) != 1:
                raise ValueError("Image ZIP member is missing or ambiguous")
            info = archive.getinfo(member)
            if info.file_size > 32 * 1024 * 1024:
                raise ValueError("Image exceeds the 32 MiB extraction limit")
            return archive.read(info), member

    @_recorded
    def read_image(self, image_id: str, question: str, region_json: str, context_refs: list[str]) -> dict[str, Any]:
        image = self._images[image_id]
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be blank")
        region = json.loads(region_json)
        if not isinstance(region, dict) or (region and (set(region) != {"x", "y", "width", "height"}
                or any(type(value) is not int for value in region.values())
                or region["x"] < 0 or region["y"] < 0 or region["width"] <= 0 or region["height"] <= 0)):
            raise ValueError("region_json must be {} or pixel integers {x,y,width,height}")
        if not isinstance(context_refs, list) or any(not isinstance(ref, str) for ref in context_refs):
            raise ValueError("context_refs must be evidence IDs")
        context = [self._span(ref) for ref in context_refs]
        if self.vision_backend is None:
            return {"ok": False, "status": "unavailable", "verification": "unverified", "image_id": image_id,
                    "source_path": image["source_path"], "error": {"type": "VisionUnavailable", "recoverable": True,
                    "message": "No vision/OCR backend configured; this occurrence remains unverified."}}
        if not self._fits({"question": question, "context": context}):
            return self._over_budget("Image question/context exceeds the budget; use fewer, smaller context units.")
        blob, member = self._image_bytes(image)
        image_sha = hashlib.sha256(blob).hexdigest()
        backend = self.vision_backend
        verify_inputs = getattr(backend, "verify_image_inputs", None)
        if callable(verify_inputs):
            verify_inputs(blob, member=member)
        identity = str(getattr(backend, "cache_key", f"{type(backend).__module__}.{type(backend).__qualname__}"))
        backend_info = {"identity": identity,
                        "identity_source": "cache_key" if hasattr(backend, "cache_key") else "class_fallback",
                        "version": str(backend.version) if getattr(backend, "version", None) is not None else None,
                        "model": str(backend.model) if getattr(backend, "model", None) is not None else None,
                        "requires_observation_id": getattr(backend, "requires_observation_id", False)}
        base_key = _sha({"bytes": image_sha, "region": region, "backend": backend_info})
        judgment_key = _sha({"base": base_key, "document": self.manifest["document_sha256"], "occurrence": image,
                             "question": question, "context": context})
        recognize = getattr(backend, "recognize", None)
        base_hit = base_key in self._base_cache
        if callable(recognize):
            if not base_hit:
                self._base_cache[base_key] = _backend_result(recognize(blob, region=region))
            observation = deepcopy(self._base_cache[base_key])
        else:
            observation = None
        judgment_hit = judgment_key in self._judgment_cache
        image_context = {**image, "region": deepcopy(region), "image_sha256": image_sha}
        def image_reply(judgment):
            return bind_image_observation({"ok": True, "status": "observed", "verification": "unverified", "semantic_review_required": True,
                    "document_sha256": self.manifest["document_sha256"], "image_id": image_id, "source_path": image["source_path"], "zip_member": member,
                    "image_sha256": image_sha, "question": question, "region": region, "context_refs": context_refs,
                    "backend": backend_info, "observation": deepcopy(observation), "judgment": deepcopy(judgment),
                    "cache": {"base_hit": base_hit, "judgment_hit": judgment_hit}})
        if not judgment_hit:
            preview_response = getattr(backend, "preview_response", None)
            if callable(preview_response):
                preview = preview_response(blob, region=deepcopy(region), question=question,
                                           context=deepcopy(context), image=deepcopy(image_context))
                if (not isinstance(preview, dict) or preview.get("preview_only") is not True
                        or not isinstance(preview.get("response"), dict)
                        or type(preview.get("output_reserve_tokens")) is not int
                        or preview["output_reserve_tokens"] < 1):
                    raise ValueError("Image backend must return a non-observation preview with output reserve")
                # Temporary admission only: no cache, sidecar or read callback
                # sees this preview. The actual full reply is checked again.
                required = self._response_tokens(image_reply(preview["response"])) + preview["output_reserve_tokens"]
                if required > self.max_tool_tokens:
                    return self._over_budget("Complete image response and output reserve exceed the tool budget before generation; "
                                             "use fewer context_refs or a shorter question. No image was read and no model generation was sent.")
            interpret = getattr(backend, "interpret", None)
            if callable(interpret):
                judgment = _backend_result(interpret(deepcopy(observation), question=question, context=deepcopy(context), image=deepcopy(image_context)))
            elif callable(backend):
                judgment = _backend_result(backend(blob, region=region, question=question, context=deepcopy(context), image=deepcopy(image_context)))
            elif callable(recognize):
                judgment = {"status": "unavailable", "reason": "OCR/base recognition has no semantic interpretation backend"}
            else:
                raise TypeError("vision_backend must be callable or expose recognize/interpret")
            self._judgment_cache[judgment_key] = json.loads(_json(judgment))
        return image_reply(self._judgment_cache[judgment_key])


class _TesseractOCR:
    def __init__(self, executable: str, language: str, timeout_seconds: int) -> None:
        self.executable, self.language, self.timeout_seconds = executable, language, timeout_seconds
        self.cache_key = f"tesseract:{executable}:{language}"

    def recognize(self, image_bytes: bytes, *, region: dict[str, int]) -> dict[str, Any]:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as image:
            if region:
                x, y, width, height = (region[key] for key in ("x", "y", "width", "height"))
                if x + width > image.width or y + height > image.height:
                    raise ValueError("OCR region falls outside image pixels")
                image = image.crop((x, y, x + width, y + height))
            with tempfile.TemporaryDirectory(prefix="cityagent-ocr-") as directory:
                path = Path(directory) / "input.png"
                image.save(path, format="PNG")
                result = subprocess.run([self.executable, str(path), "stdout", "-l", self.language],
                                        capture_output=True, text=True, timeout=self.timeout_seconds, check=True)
        return {"kind": "ocr", "text": result.stdout, "verification": "unverified",
                "limitations": ["OCR characters require review; no visual semantics or measurement has been verified"]}


def make_tesseract_backend(*, language: str = "chi_sim+eng", timeout_seconds: int = 30) -> Any:
    """Explicit opt-in CPU OCR; missing executable/languages fail honestly."""
    executable = shutil.which("tesseract")
    if not executable:
        raise RuntimeError("Tesseract is not installed; no CPU OCR backend is available")
    if type(timeout_seconds) is not int or timeout_seconds <= 0 or not language:
        raise ValueError("A language and positive timeout are required")
    available = subprocess.run([executable, "--list-langs"], capture_output=True, text=True,
                               timeout=timeout_seconds, check=True)
    languages = set((available.stdout + "\n" + available.stderr).splitlines())
    missing = set(language.split("+")) - languages
    if missing:
        raise RuntimeError(f"Tesseract language data unavailable: {', '.join(sorted(missing))}")
    return _TesseractOCR(executable, language, timeout_seconds)
