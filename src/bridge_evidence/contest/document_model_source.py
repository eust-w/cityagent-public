"""Compact source addresses for model input, with a lossless program map.

Evidence IDs and Unicode offsets remain unchanged. OOXML paths identify storage
locations, not evidence semantics; models see stable aliases while the immutable
manifest and this map retain the exact locations. This is a representation layer,
never a source filter or an extraction/validation rule.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


class ModelSourceView:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self._manifest = manifest
        self._path_ids: dict[str, str] = {}
        self._ids: dict[str, str] = {}

        def register(path: Any, identifier: str) -> None:
            if isinstance(path, str) and path and path not in self._path_ids:
                # A physical row/cell can have several text segments. Its first
                # occurrence supplies the canonical address; evidence IDs still
                # identify the individual text segments.
                self._path_ids[path] = identifier
                self._ids[identifier] = path

        for unit in manifest.get("units", []):
            register(unit.get("source_path"), unit["id"])
            for index, cell in enumerate(unit.get("cells", [])):
                register(cell.get("source_path"), f"{unit['id']}.C{index}")
        for table in manifest.get("tables", []):
            register(table.get("source_path"), table["id"])
        for image in manifest.get("images", []):
            register(image.get("source_path"), image["id"])

        def discover(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"source_path", "merged_from_source"}:
                        register(item, f"X{len(self._path_ids) + 1:06d}")
                    discover(item)
            elif isinstance(value, list):
                for item in value:
                    discover(item)

        # Include non-text/native objects and origins without an emitted row.
        # The traversal is deterministic for this frozen manifest.
        discover(manifest)

    def source_map(self) -> dict[str, str]:
        """Exact storage locations for program/audit use, never prompt history."""
        return deepcopy(self._ids)

    def resolve(self, source_id: str) -> str:
        return self._ids[source_id]

    def project(self, value: Any) -> Any:
        """Idempotent projection; preserve all content except path spelling.

        Unknown addresses are kept verbatim rather than silently losing their
        provenance. All addresses from the immutable manifest are registered.
        """
        if isinstance(value, list):
            return [self.project(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if (key in {"source_path", "merged_from_source"}
                    and isinstance(item, str) and item in self._path_ids):
                alias_key = "source_id" if key == "source_path" else "merged_from_id"
                if alias_key in value:
                    # Avoid collapsing conflicting caller-provided metadata.
                    result[key] = self.project(item)
                else:
                    result[alias_key] = self._path_ids[item]
            else:
                result[key] = self.project(item)
        return result

    def full_text(self) -> str:
        """All original text in document order, including empty units.

        Table/row/cell addressing preserves navigation and inherited values.
        Full headers, boundary notes, units and nested table metadata remain
        accessible through ordinary source tools, without duplicating the same
        adjacent prose in every row of this initial view.
        """
        sections = []
        for unit in self._manifest.get("units", []):
            address = f"{unit['id']} chars 0:{len(unit['text'])}"
            if unit.get("table_id"):
                address += f"; {unit['table_id']} row={unit['row_index']}"
                if unit.get("is_header"):
                    address += "; header"
            sections.append(f"[{address}]\n{unit['text']}")
            # Ordinary cells are already represented, in column order, by the
            # exact row text. Preserve nontrivial column/merge relationships in
            # a separate metadata line so it cannot be mistaken for that text.
            contexts = []
            for index, cell in enumerate(unit.get("cells", [])):
                if (cell.get("column_index", index) != index
                        or cell.get("column_span", 1) != 1 or cell.get("vertical_merge")
                        or cell.get("merged_from_source")):
                    context = f"c{cell.get('column_index', index)}"
                    if cell.get("column_span", 1) != 1:
                        context += f" span={cell['column_span']}"
                    if cell.get("merged_from_source"):
                        origin = self._path_ids[cell["merged_from_source"]]
                        context += f" from={origin}"
                        # Origins in emitted cells are already readable using
                        # the E ID before '.C'. Retain inherited text if an
                        # origin has no ordinary evidence unit to navigate to.
                        if not origin.startswith("E"):
                            context += f" value={cell.get('merged_context_text', '')}"
                    elif cell.get("vertical_merge"):
                        context += f" merge={cell['vertical_merge']}"
                    contexts.append(context)
            if contexts:
                sections[-1] += "\n[cell context; not source characters] " + "; ".join(contexts)
        return "\n\n".join(sections)
