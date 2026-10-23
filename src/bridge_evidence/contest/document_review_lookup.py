"""Read-only, frozen draft lookup for semantic omission review.

Literal search only retrieves candidates. It never decides that two mentions
refer to the same defect or that a source item has been omitted.
"""
from copy import deepcopy
import json
import math

from .run_state import json_digest


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _function(name, description, properties):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False}}}


_PAGE = {"cursor": {"type": "integer", "minimum": 0},
         "limit": {"type": "integer", "minimum": 1, "maximum": 100}}
LOOKUP_TOOLS = [
    _function("review_read_draft_page", "逐页读取整个冻结草稿的字段/行。引用当前段落的条目只是候选，不是全稿。", _PAGE),
    _function("review_search_draft", "在全稿字段/行值中做字面子串检索；无结果不证明遗漏，换词或逐页读全稿。",
              {"query": {"type": "string"}, **_PAGE}),
    _function("review_read_draft_item", "读取某条草稿的完整关联事实或原始证据，view为facts或sources。",
              {"path": {"type": "string"}, "view": {"type": "string", "enum": ["facts", "sources"]}, **_PAGE}),
    _function("review_read_source", "按原报告evidence_id及半开字符范围回读原文；不能修改工作区。",
              {"evidence_id": {"type": "string"}, "start": {"type": "integer", "minimum": 0},
               "end": {"type": "integer", "minimum": 1}}),
    _function("review_read_source_catalog", "分页列出冻结原文位置及明确范围的原文预览；预览不等于完整阅读。", _PAGE),
    _function("review_search_source", "按单个原文短语检索冻结全源，返回实际字符位置；无命中不能判无依据。",
              {"query": {"type": "string"}, **_PAGE}),
    _function("review_read_source_section", "顺序分页读原文或继续超长上下文 recovery；无目录从root读。",
              {"section_id": {"type": "string"}, "cursor": {"type": "integer", "minimum": 0},
               "max_units": {"type": "integer", "minimum": 1, "maximum": 100}}),
    _function("review_save_notes", "保存有界 JSON 工作笔记以便续审；笔记不是原文，也不证明已读全稿，原始证据仍可通过查阅工具回读。",
              {"notes_json": {"type": "string"}}),
]
_SEARCH_SEMANTICS = "literal substring retrieval; no result is not an omission verdict"


class FrozenDraftLookup:
    def __init__(self, items, *, count_tokens, max_tool_tokens, read_source=None, source_manifest=None,
                 compact_model_source=False):
        if type(max_tool_tokens) is not int or max_tool_tokens < 1 or not callable(count_tokens):
            raise ValueError("Lookup needs a positive integer budget and a token counter")
        if not isinstance(items, list) or any(not isinstance(item, dict)
                or not isinstance(item.get("path"), str) or not item["path"].strip()
                or not {"value", "facts", "sources"} <= item.keys()
                or not isinstance(item["facts"], list) or not isinstance(item["sources"], list) for item in items):
            raise ValueError("Frozen draft items need paths, values, facts and sources")
        self.items = deepcopy(items)
        self.by_path = {item["path"]: item for item in self.items}
        if len(self.by_path) != len(self.items):
            raise ValueError("Frozen draft paths must be unique")
        self.values = [{"path": item["path"], "value": item["value"]} for item in self.items]
        self.snapshot_sha256 = json_digest(self.items)
        self.count_tokens = count_tokens
        self.max_tool_tokens = max_tool_tokens
        self.read_source = read_source
        self.source = None
        if source_manifest is not None:
            from .document_review_source import FrozenSourceLookup
            self.source = FrozenSourceLookup(source_manifest, count_tokens=count_tokens,
                max_tool_tokens=max_tool_tokens, compact_model_source=compact_model_source)
        self.seen_paths = set()
        self.calls = []
        self._notes = None
        # Error responses also consume tool context. Reject an unusable budget
        # before starting a review, rather than sending unchecked feedback.
        self._bounded(self._invalid())
        self._bounded(self._too_large(0))
        self._bounded(self._notes_too_large())

    @property
    def notes(self):
        """Original saved working memory, never a replacement for evidence."""
        return deepcopy(self._notes)

    def save_notes(self, notes_json):
        """Same audited operation as the review_save_notes model tool."""
        return self.invoke("review_save_notes", {"notes_json": notes_json})

    @staticmethod
    def _notes_too_large():
        return {"ok": False, "code": "review_notes_too_large", "unresolved": True,
                "next_action": "Shorten the JSON notes, retaining open issues and lookup paths. Prior notes remain unchanged. Reread original evidence through lookup; notes are not source evidence."}

    def _notes_envelope(self, notes):
        return {"snapshot_sha256": self.snapshot_sha256, "notes": notes}

    def _store_notes(self, notes_json):
        def pairs(values):
            result = {}
            for key, value in values:
                if key in result:
                    raise ValueError("Duplicate note field")
                result[key] = value
            return result

        def invalid_constant(_value):
            raise ValueError("Notes must contain finite JSON values")

        try:
            if type(notes_json) is not str:
                raise ValueError("Notes must be JSON text")
            notes = json.loads(notes_json, object_pairs_hook=pairs, parse_constant=invalid_constant)
            if type(notes) is not dict:
                raise ValueError("Notes must be a JSON object")
            self._validate_json(notes, depth=1)
        except (ValueError, TypeError, RecursionError):
            return self._bounded({"ok": False, "code": "invalid_review_notes", "unresolved": True,
                "next_action": "Submit a JSON object with unique keys and finite values. Prior notes remain unchanged; never treat notes as source evidence."})
        if self._tokens(self._notes_envelope(notes)) > self.max_tool_tokens:
            return self._bounded(self._notes_too_large())
        result = self._bounded({"ok": True, "snapshot_sha256": self.snapshot_sha256,
                                "notes_sha256": json_digest(notes)})
        self._notes = notes
        return result

    @staticmethod
    def _validate_json(value, depth=0):
        """Reject Python objects, non-string keys, nonfinite numbers and cycles."""
        if depth > 64:
            raise ValueError("Lookup JSON nesting exceeds supported depth")
        if type(value) is dict:
            for key, child in value.items():
                if type(key) is not str:
                    raise ValueError("Lookup JSON keys must be strings")
                FrozenDraftLookup._validate_json(child, depth + 1)
        elif type(value) is list:
            for child in value:
                FrozenDraftLookup._validate_json(child, depth + 1)
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Lookup JSON numbers must be finite")
        elif value is not None and type(value) not in (str, bool, int):
            raise ValueError("Lookup state must contain only JSON values")

    def export_state(self):
        """Pure data checkpoint; lookup execution is not model exposure proof.

        The checksum detects accidental changes, not forgery by an actor who
        can rewrite the state and its hash. The workflow owns model exposure.
        No tokenizer/backend call is made here, so completed local state can
        still be saved after a model deadline or transport failure.
        """
        state = {"schema_version": 1, "snapshot_sha256": self.snapshot_sha256,
                 "seen_paths": sorted(self.seen_paths), "notes": self.notes,
                 "calls": deepcopy(self.calls)}
        if self.source is not None:
            state.update(schema_version=2, source_snapshot_sha256=self.source.snapshot_sha256)
        self._validate_json(state)
        return {**state, "state_sha256": json_digest(state)}

    def restore_state(self, state):
        """Validate the entire checkpoint before replacing any mutable state."""
        try:
            self._validate_json(state)
            fields = {"schema_version", "snapshot_sha256", "seen_paths", "notes", "calls", "state_sha256"}
            if self.source is not None:
                fields.add("source_snapshot_sha256")
            if type(state) is not dict or set(state) != fields:
                raise ValueError("Invalid state fields")
            if type(state["schema_version"]) is not int or state["schema_version"] != (2 if self.source else 1):
                raise ValueError("Invalid state version")
            if self.source and state["source_snapshot_sha256"] != self.source.snapshot_sha256:
                raise ValueError("Different frozen source")
            if state["snapshot_sha256"] != self.snapshot_sha256:
                raise ValueError("Different frozen draft")
            body = {key: value for key, value in state.items() if key != "state_sha256"}
            if state["state_sha256"] != json_digest(body):
                raise ValueError("Invalid state checksum")
            seen, notes, calls = (state[key] for key in ("seen_paths", "notes", "calls"))
            if (type(seen) is not list or any(type(path) is not str or path not in self.by_path for path in seen)
                    or len(set(seen)) != len(seen)):
                raise ValueError("Invalid seen paths")
            if notes is not None and (type(notes) is not dict or self._tokens(self._notes_envelope(notes)) > self.max_tool_tokens):
                raise ValueError("Invalid or oversized notes")
            if type(calls) is not list:
                raise ValueError("Invalid call list")
            permitted = {tool["function"]["name"] for tool in LOOKUP_TOOLS} | {"invalid_lookup"}

            def digest(value):
                return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

            for call in calls:
                if (type(call) is not dict or set(call) != {"name", "arguments_sha256", "result_sha256", "ok"}
                        or type(call["name"]) is not str or call["name"] not in permitted or type(call["ok"]) is not bool
                        or not digest(call["result_sha256"])
                        or not (digest(call["arguments_sha256"]) or call["arguments_sha256"] is None)
                        or (call["ok"] and (call["name"] == "invalid_lookup" or call["arguments_sha256"] is None))):
                    raise ValueError("Invalid call receipt")
        except Exception:
            # The input may contain source text or a Python object's repr; none
            # of that is included in a restore failure, including its cause.
            raise ValueError("Invalid or incompatible frozen lookup checkpoint") from None
        self.seen_paths, self._notes, self.calls = set(seen), deepcopy(notes), deepcopy(calls)

    @property
    def all_values_seen(self):
        return self.seen_paths == set(self.by_path)

    def metadata(self, *, full=False):
        value = {"snapshot_sha256": self.snapshot_sha256, "total_items": len(self.items),
                 "mode": "complete_values" if full else "paged",
                 "linked_candidates_are_not_complete_draft": True,
                 "policy": "Compare meaning and object/year/side/location across the whole draft. Empty related_output or search results do not prove omission. If paged, inspect all draft values before asserting missing_items; otherwise report unresolved_items."}
        if full:
            value["items"] = deepcopy(self.values)
        return value

    def _tokens(self, value):
        count = self.count_tokens(_json(value))
        if type(count) is not int or count < 0:
            raise ValueError("count_tokens must return a non-negative integer")
        return count

    def _bounded(self, value):
        if self._tokens(value) <= self.max_tool_tokens:
            return value
        raise ValueError("Lookup envelope exceeds tool budget")

    @staticmethod
    def _invalid():
        return {"ok": False, "code": "invalid_or_oversized_lookup", "unresolved": True,
                "next_action": "Use an available tool with valid page/range bounds. Inspect smaller pages or source character ranges; do not infer absence from this error."}

    def _too_large(self, cursor):
        return {"ok": False, "code": "lookup_record_too_large", "unresolved": True,
                "snapshot_sha256": self.snapshot_sha256, "cursor": cursor,
                "next_action": "Use source evidence ranges where available; keep the review unresolved if an indivisible draft value cannot be read. No record was shortened."}

    def _page(self, values, cursor, limit, *, extra=None):
        if (type(cursor) is not int or type(limit) is not int or not 0 <= cursor <= len(values)
                or not 1 <= limit <= 100):
            raise ValueError("Invalid page arguments")
        def page(end):
            return {"ok": True, "snapshot_sha256": self.snapshot_sha256,
                    "items": deepcopy(values[cursor:end]), "start": cursor, "end": end,
                    "total": len(values), "next_cursor": end if end < len(values) else None, **(extra or {})}
        result = self._bounded(page(cursor))
        for end in range(cursor + 1, min(len(values), cursor + limit) + 1):
            candidate = page(end)
            if self._tokens(candidate) > self.max_tool_tokens:
                if end == cursor + 1:
                    return self._bounded(self._too_large(cursor))
                break
            result = candidate
        return result

    def invoke(self, name, arguments):
        permitted = {tool["function"]["name"]: set(tool["function"]["parameters"]["required"])
                     for tool in LOOKUP_TOOLS}
        # Reject non-JSON input without repr/str of user-controlled objects.
        try:
            arguments_sha256 = json_digest(arguments)
        except (ValueError, TypeError):
            arguments_sha256 = None
        authorized_name = name if type(name) is str and name in permitted else None
        try:
            if authorized_name is None or arguments_sha256 is None or not isinstance(arguments, dict) or set(arguments) != permitted[name]:
                raise ValueError("Invalid lookup tool or arguments")
            if name == "review_read_draft_page":
                result = self._page(self.values, **arguments)
            elif name == "review_search_draft":
                query = arguments["query"]
                if not isinstance(query, str) or not query.strip() or len(query) > 512:
                    raise ValueError("Invalid literal query")
                values = [v for v in self.values if query.casefold() in _json(v["value"]).casefold()]
                # Search semantics are part of every successful envelope, so
                # page sizing must include them before accepting each record.
                result = self._page(values, arguments["cursor"], arguments["limit"],
                                    extra={"search_semantics": _SEARCH_SEMANTICS})
            elif name == "review_read_draft_item":
                item = self.by_path[arguments["path"]]
                if arguments["view"] not in {"facts", "sources"}:
                    raise ValueError("Invalid lookup view")
                result = self._page(item[arguments["view"]], arguments["cursor"], arguments["limit"])
            elif name == "review_save_notes":
                result = self._store_notes(**arguments)
            elif name in {"review_read_source_catalog", "review_search_source"}:
                if self.source is None:
                    raise ValueError("Frozen source discovery unavailable")
                from .document_review_source import SOURCE_SEARCH_SEMANTICS
                values = self.source.catalog() if name == "review_read_source_catalog" else self.source.search(arguments["query"])
                result = self._page(values, arguments["cursor"], arguments["limit"], extra={
                    "source_snapshot_sha256": self.source.snapshot_sha256,
                    "search_semantics": SOURCE_SEARCH_SEMANTICS})
            elif self.source is not None:
                result = self._bounded(self.source.read(name, arguments))
            else:
                if self.read_source is None:
                    raise ValueError("Source lookup unavailable")
                identifier, start, end = (arguments[key] for key in ("evidence_id", "start", "end"))
                if (type(identifier) is not str or not identifier.strip() or len(identifier) > 512
                        or type(start) is not int or type(end) is not int or not 0 <= start < end):
                    raise ValueError("Invalid source range")
                value = self.read_source(**arguments)
                if not isinstance(value, dict) or value.get("ok") is not True:
                    raise ValueError("Source lookup failed")
                result = self._bounded(deepcopy(value))
            if result.get("ok") and name in {"review_read_draft_page", "review_search_draft"}:
                self.seen_paths.update(item["path"] for item in result["items"])
        except Exception:
            # Backend exceptions/error bodies can contain paths, source text
            # or credentials. Only a static unresolved response reaches review.
            result = self._bounded(self._invalid())
        self.calls.append({"name": authorized_name or "invalid_lookup", "arguments_sha256": arguments_sha256,
                           "result_sha256": json_digest(result), "ok": result.get("ok", False)})
        return result
