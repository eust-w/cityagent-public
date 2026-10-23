"""Pure structural navigation for a small number of document reading tasks.

No unit is rewritten or interpreted.  A primary assignment is a scheduling
decision, never a read receipt, a fact, or evidence of semantic completeness.
Regex headings are navigation candidates; their titles remain original text.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from bisect import bisect_left
from typing import Any

SCHEMA = "document-task-plan-v1"
_STYLE = re.compile(r"heading(\d+)", re.I)
_TOC_STYLE = re.compile(r"(?:toc|contents|目录)\s*\d*", re.I)
_TOC_TITLE = re.compile(r"(?:目\s*录|table\s+of\s+contents|contents)", re.I)
_PAGE_ENTRY = re.compile(r"(?:\t+|\.{2,}|…{2,}|·{3,})\s*(?:\d+|[ivxlcdm]+)\s*$", re.I)
_NUMBERED = re.compile(r"([1-9]\d{0,2}(?:\.[1-9]\d{0,2}){0,5})[\s\u3000]+(\S.*?)\s*$")
_CHINESE = re.compile(r"第[零〇一二三四五六七八九十百千\d]+([章节篇部])\s*\S.*$")
_APPENDIX = re.compile(r"(?:附录|附\s*件|appendix)(?:\s*[A-Za-z\d一二三四五六七八九十]+)?(?:[\s\u3000:：、.．]+\S.*)?$", re.I)
_ANNOTATION = re.compile(r"\s*(?:注|备注|说明|单位|Note\b)", re.I)
_ONLY_MEASUREMENT = re.compile(r"[\d\s.,+−\-/%％×]*\s*(?:mm|cm|m|km|mpa|kn|n|kg|t|℃|°c|%|％)(?:\d|[²³])?", re.I)


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _story(unit: dict[str, Any]) -> str:
    return unit.get("context", {}).get("story", "body")


def _heading(unit: dict[str, Any], position: int) -> dict[str, Any] | None:
    if unit.get("table_id"):
        return None
    text = unit["text"]
    title = text.strip()
    style = unit.get("style_id", "")
    if _TOC_STYLE.fullmatch(style) or _TOC_TITLE.fullmatch(title) or _PAGE_ENTRY.search(title):
        return None
    styled = _STYLE.fullmatch(style)
    if styled and int(styled.group(1)) > 0:
        level, basis = int(styled.group(1)), "heading_style"
    elif len(title) > 160 or not title or "\n" in title or re.search(r"[。；;]", title):
        return None
    elif _APPENDIX.fullmatch(title):
        level, basis = 1, "regex_appendix_navigation_candidate"
    elif chinese := _CHINESE.fullmatch(title):
        level = 2 if chinese.group(1) == "节" else 1
        basis = "regex_chapter_navigation_candidate"
    elif numbered := _NUMBERED.fullmatch(title):
        remainder = numbered.group(2)
        if not re.search(r"[A-Za-z\u3400-\u9fff]", remainder) or _ONLY_MEASUREMENT.fullmatch(remainder):
            return None
        level, basis = numbered.group(1).count(".") + 1, "regex_numbered_navigation_candidate"
    else:
        return None
    return {"evidence_id": unit["id"], "title": text, "level": level,
            "basis": basis, "story": _story(unit), "unit_index": position,
            "grants_reading_coverage": False}



def _material_headings(manifest: dict[str, Any], material_index) -> list[dict[str, Any]]:
    """Bind repaired section titles to canonical units; never match by text alone."""
    material_index.validate_manifest(manifest)
    units = manifest["units"]
    positions = {unit["id"]: i for i, unit in enumerate(units)}
    selected = {}
    for section in material_index.sections:
        if section.get("region") != "content" or type(section.get("level")) is not int or section["level"] < 1:
            continue
        spans = section.get("heading_spans", [])
        if not isinstance(spans, list):
            raise ValueError("Material heading spans must be a list")
        for span in spans:
            identifier = span.get("evidence_id")
            if identifier not in positions:
                raise ValueError("Material heading has unknown canonical evidence")
            unit = units[positions[identifier]]
            start, end = span.get("start"), span.get("end")
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(unit["text"])
                    or span.get("text") != unit["text"][start:end]):
                raise ValueError("Material heading differs from its canonical source range")
            if unit.get("table_id"):
                continue
            heading = {"evidence_id": identifier, "title": unit["text"], "level": section["level"],
                "basis": "mineru_bound_heading_navigation", "story": _story(unit),
                "unit_index": positions[identifier], "material_section_ref": section["id"],
                "grants_reading_coverage": False}
            if identifier in selected and selected[identifier]["level"] != heading["level"]:
                raise ValueError("Conflicting material levels for one canonical heading")
            selected[identifier] = heading
            # A heading may span multiple canonical units; its first exact unit
            # is a stable navigation anchor, while all text remains readable.
            break
    return sorted(selected.values(), key=lambda item: item["unit_index"])

def build_document_navigation(manifest: dict[str, Any], *, material_index=None) -> dict[str, Any]:
    """Share source-addressed chapter candidates between planning and tools.

    Section IDs are existing heading evidence IDs. Regex matches are only
    navigation candidates, never assertions about the document's meaning.
    """
    units = manifest.get("units")
    if not isinstance(units, list):
        raise ValueError("Manifest units must be a list")
    ids = [unit.get("id") for unit in units if isinstance(unit, dict)]
    if (len(ids) != len(units) or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids)
            or any(not isinstance(unit.get("text"), str) or not isinstance(unit.get("context", {}), dict)
                   for unit in units)):
        raise ValueError("Navigation requires unique original evidence IDs and text")
    headings = [heading for i, unit in enumerate(units) if (heading := _heading(unit, i))]
    material_headings = _material_headings(manifest, material_index) if material_index is not None else []
    if material_headings:
        # The source-bound repaired material hierarchy replaces regex guesses;
        # canonical E IDs remain the section IDs used by all reading tools.
        headings = material_headings
    at = {heading["unit_index"]: heading for heading in headings}
    sections = [{"id": "root", "title": "Whole document", "level": 0,
                 "basis": "whole_document_navigation", "parent_section_id": None,
                 "unit_index_start": 0, "unit_index_end": len(units)}]
    stack: list[dict[str, Any]] = []
    paths = []
    previous_story = None
    for position, unit in enumerate(units):
        story = _story(unit)
        if story != previous_story:
            while stack:
                stack.pop()["unit_index_end"] = position
        if heading := at.get(position):
            while stack and stack[-1]["level"] >= heading["level"]:
                stack.pop()["unit_index_end"] = position
            section = {"id": heading["evidence_id"], "title": heading["title"],
                       "level": heading["level"], "basis": heading["basis"], "story": story,
                       "parent_section_id": stack[-1]["id"] if stack else "root",
                       "unit_index_start": position, "unit_index_end": len(units)}
            sections.append(section)
            stack.append(section)
        paths.append([section["id"] for section in stack])
        previous_story = story
    return {"schema": "document-navigation-v1", "document_sha256": manifest.get("document_sha256"),
            "headings": headings, "sections": sections, "heading_paths": paths,
            **({"material_navigation": True} if material_headings else {}),
            "section_evidence_ids": {section["id"]: ids[section["unit_index_start"]:section["unit_index_end"]]
                                     for section in sections}, "grants_reading_coverage": False}


def _range(unit: dict[str, Any]) -> dict[str, Any]:
    result = {"evidence_id": unit["id"], "start": 0, "end": len(unit["text"]),
              "source_path": unit.get("source_path"), "coverage_id": unit.get("coverage_id", unit["id"])}
    for key in ("source_start", "source_end", "table_id", "row_index", "segment_index"):
        if key in unit:
            result[key] = copy.deepcopy(unit[key])
    return result


def _cuts(candidates: list[int], prefix: list[int], count: int) -> list[int]:
    """Deterministic contiguous balancing; characters plus unit overhead, not tokens."""
    chosen = []
    lower = 0
    for ordinal in range(1, count):
        remaining = count - ordinal - 1
        upper = len(candidates) - remaining
        available = candidates[lower:upper]
        ideal = prefix[-1] * ordinal / count
        weights = [prefix[position] for position in available]
        near = bisect_left(weights, ideal)
        indexes = {min(near, len(available) - 1), max(0, near - 1)}
        index = min(indexes, key=lambda j: (abs(weights[j] - ideal), available[j]))
        cut = available[index]
        chosen.append(cut)
        lower += index + 1
    return chosen


def build_document_task_plan(manifest: dict[str, Any], *, target_tasks: int = 4,
                             max_tasks: int = 4, material_index=None) -> dict[str, Any]:
    """Schedule every manifest unit once; retain original IDs and character offsets.

    Parent/nested table families are indivisible intervals.  Reliable/regex
    chapter boundaries are preferred if sufficient; otherwise continuous unit
    boundaries are used.  A large table can result in fewer tasks than requested.
    Context references are original source navigation, not additional facts.
    All-document search/read access must remain available to the caller.
    """
    if type(target_tasks) is not int or type(max_tasks) is not int or target_tasks < 1 or max_tasks < 1:
        raise ValueError("target_tasks and max_tasks must be positive integers")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("document_sha256"), str) or not manifest["document_sha256"]:
        raise ValueError("A source-bound manifest is required")
    units = manifest.get("units")
    tables = manifest.get("tables", [])
    if not isinstance(units, list) or not isinstance(tables, list):
        raise ValueError("Manifest units and tables must be lists")
    positions: dict[str, int] = {}
    for position, unit in enumerate(units):
        if not isinstance(unit, dict) or not isinstance(unit.get("id"), str) or not unit["id"] or unit["id"] in positions:
            raise ValueError("Evidence IDs must be nonempty and unique")
        if not isinstance(unit.get("text"), str) or not isinstance(unit.get("context", {}), dict):
            raise ValueError("Each unit needs unchanged text and a context object")
        positions[unit["id"]] = position
    table_index: dict[str, dict[str, Any]] = {}
    for table in tables:
        if not isinstance(table, dict) or not isinstance(table.get("id"), str) or not table["id"] or table["id"] in table_index:
            raise ValueError("Table IDs must be nonempty and unique")
        table_index[table["id"]] = table
    roots: dict[str, str] = {}
    for table_id in table_index:
        trail: set[str] = set()
        current = table_id
        while True:
            if current in trail:
                raise ValueError("Table parent cycle")
            if current not in table_index:
                raise ValueError("Unknown parent table")
            trail.add(current)
            parent = table_index[current].get("parent_table_id")
            if not parent:
                roots[table_id] = current
                break
            current = parent
    family_positions: dict[str, list[int]] = {}
    table_positions: dict[str, list[int]] = {key: [] for key in table_index}
    for position, unit in enumerate(units):
        table_id = unit.get("table_id")
        if table_id:
            if table_id not in table_index:
                raise ValueError("Unknown unit table")
            table_positions[table_id].append(position)
            family_positions.setdefault(roots[table_id], []).append(position)
    for table_id, table in table_index.items():
        for header in table.get("header_evidence_ids", []):
            if header not in positions:
                raise ValueError("Unknown table header evidence")
            family_positions.setdefault(roots[table_id], []).append(positions[header])
    family_intervals = {key: (min(values), max(values) + 1) for key, values in family_positions.items()}
    legal = [cut for cut in range(1, len(units))
             if not any(start < cut < end for start, end in family_intervals.values())]
    navigation = build_document_navigation(manifest, material_index=material_index)
    headings = navigation["headings"]
    heading_at = {heading["unit_index"]: heading for heading in headings}
    heading_paths = navigation["heading_paths"]
    level_by_story: dict[str, int] = {}
    for heading in headings:
        story = heading["story"]
        level_by_story[story] = min(level_by_story.get(story, heading["level"]), heading["level"])
    preferred = {heading["unit_index"] for heading in headings
                 if heading["level"] == level_by_story[heading["story"]]}
    preferred.update(i for i in range(1, len(units)) if _story(units[i]) != _story(units[i - 1]))
    preferred = sorted(set(legal).intersection(preferred))
    count = min(target_tasks, max_tasks, len(legal) + 1) if units else 0
    use_headings = count > 1 and len(preferred) >= count - 1
    prefix = [0]
    for unit in units:
        prefix.append(prefix[-1] + len(unit["text"]) + 32)
    selected = _cuts(preferred if use_headings else legal, prefix, count) if count > 1 else []
    bounds = [0, *selected, len(units)] if units else []
    manifest_sha = _hash(manifest)
    config = {"target_tasks": target_tasks, "max_tasks": max_tasks,
              "balance_metric": "original_characters_plus_32_per_unit_not_tokens"}
    if material_index is not None:
        config["material_navigation_sha256"] = _hash(navigation["headings"])
    cell_sources: dict[str, str] = {}
    for unit in units:
        for cell in unit.get("cells", []):
            if cell.get("source_path"):
                cell_sources.setdefault(cell["source_path"], unit["id"])
    opening = [unit["id"] for unit in units
               if not unit.get("table_id") and unit["text"].strip()][:3]
    if material_index is not None:
        # A bounded cover prefix supplies source bridge/year/date context, not
        # extracted metadata. No filename-derived year or semantic default.
        opening = []
        opening_characters = 0
        for unit in units:
            if unit.get("table_id") or not unit["text"].strip():
                continue
            if len(opening) == 8 or opening_characters + len(unit["text"]) > 1500:
                break
            opening.append(unit["id"])
            opening_characters += len(unit["text"])
    tasks = []
    for ordinal, (start, end) in enumerate(zip(bounds, bounds[1:]), 1):
        primary = units[start:end]
        primary_ids = [unit["id"] for unit in primary]
        primary_set = set(primary_ids)
        references: dict[str, set[str]] = {}
        unresolved: list[dict[str, str]] = []

        def refer(evidence_id: str | None, role: str) -> None:
            if not evidence_id:
                return
            if evidence_id not in positions:
                item = {"evidence_id": evidence_id, "role": role}
                if item not in unresolved:
                    unresolved.append(item)
            elif evidence_id not in primary_set:
                references.setdefault(evidence_id, set()).add(role)

        for evidence_id in opening:
            refer(evidence_id, "opening_navigation")
        task_tables: set[str] = set()
        for unit in primary:
            context = unit.get("context", {})
            refer(context.get("section_heading_id"), "source_section_heading")
            for evidence_id in context.get("preceding_evidence_ids", []):
                refer(evidence_id, "source_preceding_context")
            for evidence_id in heading_paths[positions[unit["id"]]]:
                refer(evidence_id, "heading_navigation")
            table_id = unit.get("table_id")
            while table_id:
                task_tables.add(table_id)
                table_id = table_index[table_id].get("parent_table_id")
            for cell in unit.get("cells", []):
                merged = cell.get("merged_from_source")
                if merged:
                    evidence_id = cell_sources.get(merged)
                    if evidence_id:
                        refer(evidence_id, "merged_cell_origin")
                    else:
                        unresolved.append({"source_path": merged, "role": "merged_cell_origin"})
        for table_id in sorted(task_tables):
            table = table_index[table_id]
            for header in table.get("header_evidence_ids", []):
                refer(header, "table_header:" + table.get("header_kind", "unknown"))
            family = family_intervals.get(roots[table_id])
            if family:
                for direction, edge in ((-1, family[0]), (1, family[1] - 1)):
                    position = edge + direction
                    while 0 <= position < len(units):
                        neighbor = units[position]
                        if (neighbor.get("table_id") or _story(neighbor) != _story(units[edge])
                                or position in heading_at):
                            break
                        refer(neighbor["id"], "table_boundary_context")
                        # A blank source paragraph does not sever a following
                        # unit/note run.  Keep its original zero/whitespace span
                        # too, while never crossing the next heading or table.
                        if neighbor["text"].strip() and not _ANNOTATION.match(neighbor["text"]):
                            break
                        position += direction
        context_ids = sorted(references, key=positions.__getitem__)
        task_headings = [copy.deepcopy(heading) for heading in headings
                         if start <= heading["unit_index"] < end or heading["evidence_id"] in references]
        task_id = "task-" + _hash({"schema": SCHEMA, "document_sha256": manifest["document_sha256"],
                                   "manifest_sha256": manifest_sha, "config": config,
                                   "unit_index_start": start, "unit_index_end": end})[:24]
        tasks.append({"id": task_id, "ordinal": ordinal, "primary_evidence_ids": primary_ids,
                      "primary_ranges": [_range(unit) for unit in primary],
                      "context_evidence_ids": context_ids,
                      "context_references": [{**_range(units[positions[evidence_id]]),
                                              "roles": sorted(references[evidence_id]),
                                              "grants_reading_coverage": False} for evidence_id in context_ids],
                      "unresolved_context_references": unresolved,
                      "headings": task_headings,
                      "table_ids": [table["id"] for table in tables if table["id"] in task_tables],
                      "table_context": [copy.deepcopy(table) for table in tables if table["id"] in task_tables],
                      "unit_index_start": start, "unit_index_end": end,
                      "original_character_count": sum(len(unit["text"]) for unit in primary),
                      "grants_reading_coverage": False})
    if material_index is not None:
        for task in tasks:
            task["material_section_refs"] = list(dict.fromkeys(heading["material_section_ref"]
                for heading in task["headings"] if heading.get("material_section_ref")))
    assigned = [evidence_id for task in tasks for evidence_id in task["primary_evidence_ids"]]
    return {"schema": SCHEMA, "document_sha256": manifest["document_sha256"],
            "manifest_sha256": manifest_sha, "config": config,
            "partition_basis": ("mineru_heading_boundaries" if navigation.get("material_navigation") else "heading_boundaries")
                if use_headings else "continuous_atomic_fallback",
            "tasks": tasks, "coverage": {
                "manifest_unit_count": len(units), "assigned_unit_count": len(assigned),
                "all_units_assigned_once": assigned == [unit["id"] for unit in units] and len(set(assigned)) == len(assigned),
                "empty_unit_count": sum(not unit["text"] for unit in units),
                "original_character_count": sum(len(unit["text"]) for unit in units),
                "table_families_unbroken": all(not any(start < cut < end for cut in selected)
                                                for start, end in family_intervals.values()),
                "unlocated_table_ids": [key for key in table_index if not table_positions[key]],
                "image_count": len(manifest.get("images", [])),
                "unverified_content_count": len(manifest.get("unverified_content", [])),
                "grants_reading_coverage": False, "semantic_completeness_verified": False,
                "visual_evidence_verified": False},
            "source_limitations": copy.deepcopy(manifest.get("limitations", [])),
            "global_source_access_required": True}
