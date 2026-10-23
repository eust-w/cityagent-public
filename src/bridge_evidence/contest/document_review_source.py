"""Frozen original-source discovery for independent semantic review.

This module returns original positions, not semantic matches or conclusions.
Reading here never updates the extraction workspace or its coverage ledger.
"""
from copy import deepcopy
import json
import re

from .document_tools import DocumentTools, SourceRangeError
from .run_state import json_digest


SOURCE_SEARCH_SEMANTICS = (
    "Literal substring positions in the frozen provided manifest only. No hits, "
    "an unread continuation, or a budget error do not prove lack of support. "
    "Read context and compare objects/years/locations; incomplete investigation remains unresolved_items."
)


class FrozenSourceLookup:
    def __init__(self, manifest, *, count_tokens, max_tool_tokens, compact_model_source=False):
        self.manifest = deepcopy(manifest)
        self.snapshot_sha256 = json_digest(self.manifest)
        self.units = self.manifest["units"]
        if (any(type(u.get("id")) is not str or not u["id"] or type(u.get("text")) is not str for u in self.units)
                or len({u["id"] for u in self.units}) != len(self.units)):
            raise ValueError("Frozen source units need distinct IDs and original text")
        self.by_id = {u["id"]: u for u in self.units}
        self.count_tokens, self.max_tool_tokens = count_tokens, max_tool_tokens
        self.compact_model_source = compact_model_source
        from .document_model_source import ModelSourceView
        self.model_source = ModelSourceView(self.manifest)
        # Count the actual reviewer envelope, including renamed recovery calls.
        # Do not grant ordinary workspace reads from an independent reviewer.
        self.tools = DocumentTools(self.manifest, count_tokens=self._source_tokens,
                                   max_tool_tokens=max_tool_tokens, on_read=None,
                                   compact_model_source=compact_model_source)

    def metadata(self):
        return {"source_snapshot_sha256": self.snapshot_sha256, "total_units": len(self.units),
                "catalog_tool": "review_read_source_catalog", "search_tool": "review_search_source",
                "read_tool": "review_read_source", "section_tool": "review_read_source_section",
                "root_section": "root", "scope": "Frozen text units from the provided manifest only; images/native objects are separate. Previews/search hits are not complete context or semantic decisions."}

    def complete_view(self):
        """Exact provided manifest plus separate, unambiguous tool coordinates.

        Preserve table/cell/header/context/limitations and every original field.
        This view is evidence availability, not reading coverage or a verdict.
        The provided manifest may itself be an explicitly limited excerpt.
        """
        return {"manifest": (self.tools.model_source.project(self.manifest)
                             if self.compact_model_source else deepcopy(self.manifest)),
                "unit_read_ranges": [{"evidence_id": u["id"], "start": 0, "end": len(u["text"])}
                                     for u in self.units]}

    @staticmethod
    def _navigation(value):
        if isinstance(value, list):
            return [FrozenSourceLookup._navigation(v) for v in value]
        if isinstance(value, dict):
            result = {k: FrozenSourceLookup._navigation(v) for k, v in value.items()}
            if set(result) == {"tool", "arguments"} and isinstance(result["arguments"], dict):
                result["tool"] = {"read_source": "review_read_source", "read_section": "review_read_source_section"}.get(result["tool"], result["tool"])
            return result
        return value

    def _envelope(self, value):
        if self.compact_model_source:
            value = self.model_source.project(value)
        result = {**self._navigation(value), "source_snapshot_sha256": self.snapshot_sha256,
                  **({"unresolved": True} if value.get("ok") is False else {})}
        # The same key order must reach both tokenizer and actual tool message,
        # including error-only fields added by the wrapper.
        return json.loads(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))

    def _source_tokens(self, text):
        return self.count_tokens(json.dumps(self._envelope(json.loads(text)), ensure_ascii=False,
                                            separators=(",", ":"), allow_nan=False))

    def catalog(self):
        # Every preview is a labelled exact original range with an explicit end;
        # a long heading/body is read through the same lossless source tools.
        return [{"evidence_id": u["id"], "kind": u.get("kind"), "start": 0, "end": len(u["text"]),
                 "source_path": u.get("source_path"),
                 "preview": {"start": 0, "end": min(96, len(u["text"])), "text": u["text"][:96]},
                 "preview_complete": len(u["text"]) <= 96,
                 "continuation": {"tool": "review_read_source", "arguments": {
                     "evidence_id": u["id"], "start": 96, "end": len(u["text"])}} if len(u["text"]) > 96 else None}
                for u in self.units]

    def search(self, query):
        if type(query) is not str or not query.strip() or len(query) > 512:
            raise ValueError("Invalid literal source query")
        # Match on the original string, not a casefolded copy whose Unicode
        # character offsets can differ. This remains literal, non-semantic search.
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        return [{"evidence_id": u["id"], "start": hit.start(), "end": hit.end(),
                 "text": u["text"][hit.start():hit.end()], "source_path": u.get("source_path"),
                 "unit_start": 0, "unit_end": len(u["text"]),
                 "read_context": {"tool": "review_read_source", "arguments": {
                     "evidence_id": u["id"], "start": 0, "end": len(u["text"])}}}
                for u in self.units for hit in pattern.finditer(u["text"])]

    def read(self, name, arguments):
        if name == "review_read_source":
            identifier, start, end = (arguments[key] for key in ("evidence_id", "start", "end"))
            if type(identifier) is not str or type(start) is not int or type(end) is not int:
                raise ValueError("Invalid original range")
            if identifier.startswith("@piece/"):
                # DocumentTools reconstructs and validates this deterministic
                # context/primary entry against the frozen manifest. It is not
                # an ordinary evidence ID or a filesystem location.
                if re.fullmatch(r"@piece/@(?:source|table)/[^/]+/[0-9]+/[0-9]+/[0-9]+", identifier) is None:
                    raise ValueError("Invalid recovery reference")
                if not 0 <= start < end:
                    raise ValueError("Invalid original range")
            elif identifier not in self.by_id:
                raise ValueError("Invalid original range")
            elif not 0 <= start < end <= len(self.by_id[identifier]["text"]):
                # Only a unit in this frozen manifest can supply trusted bounds.
                # Return the controlled error rather than losing its cause in
                # the enclosing lookup's generic exception boundary.
                return self._envelope(SourceRangeError(identifier, len(self.by_id[identifier]["text"])).as_tool_error(review=True))
            result = self.tools.read_source(**arguments)
        elif name == "review_read_source_section":
            section, cursor, units = (arguments[key] for key in ("section_id", "cursor", "max_units"))
            if (type(section) is not str or not section or type(cursor) is not int or cursor < 0
                    or type(units) is not int or not 1 <= units <= 100):
                raise ValueError("Invalid original section page")
            result = self.tools.read_section(**arguments)
        else:
            raise ValueError("Unknown source tool")
        if not result.get("ok") and result.get("status") != "budget_exceeded":
            # Do not expose arbitrary exception strings, filesystem paths, or
            # unsupported backend objects. Budget recovery is a controlled API.
            raise ValueError("Original source unavailable")
        if result.get("ok") is not True:
            result["unresolved"] = True
        return self._envelope(result)
