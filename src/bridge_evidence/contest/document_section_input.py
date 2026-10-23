"""Bounded, lossless source representation for a section Agent's initial input.

This module only removes redundant representation.  It neither extracts facts
nor marks source as read.  The immutable manifest plus ModelSourceView's
source_map retain exact addresses; the worker must save that map separately and
keep full-document source tools available.  Source units are embedded whole or
left for tool paging, never shortened to fit this character budget.
"""
from __future__ import annotations

import copy
import json
from typing import Any

from .document_model_source import ModelSourceView

_PREFIX = "本次任务和只读原文如下；未内嵌的原文通过read_source分页读取。\n"
_NOTE = ("内嵌范围仅表示输入包含原文，不代表模型已理解或完成审核。"
         "CONTEXT也是原文背景；合并格继承值不是本格新增字符，不能视作真空值。"
         "出处别名由独立source_map映射到精确原路径；地址别名不能直接当read_source的evidence_id。"
         "merged_origin_evidence_ids提供可回读原文ID列表，同一格的多个segment须一并查看；null表示无已索引原文单元映射，可全文检索。"
         "task.id和headings.evidence_id可作为read_section的section_id；read_outline提供同一章节编号。"
         "搜索默认限本任务primary原文，query数组是多个原文短语的OR；需要跨段核对时显式用scope=document回查全文。")
_CELL_FIELDS = ("column_index", "column_span", "vertical_merge", "merged_from_source",
                "merged_context_text", "source_path")
_UNIT_FIELDS = ("kind", "table_id", "row_index", "segment_index", "is_header",
                "source_start", "source_end")
_TABLE_FIELDS = ("id", "parent_table_id", "row_count", "header_evidence_ids",
                 "header_kind", "source_path")
_HEADING_FIELDS = ("evidence_id", "level", "basis", "story")
_CELL_COLUMNS = ("column_index", "column_span", "vertical_merge", "merged_from_id",
                 "merged_context_text", "source_id", "text_binding")
_ABSENT = {"absent_source_field": True}


class SectionInputBudgetError(ValueError):
    """Mandatory navigation itself cannot fit; no IDs are silently removed."""

    def __init__(self, limit: int, metadata_characters: int) -> None:
        self.limit = limit
        self.metadata_characters = metadata_characters
        super().__init__(f"Section navigation requires {metadata_characters} characters; budget is {limit}; no evidence IDs were dropped")


def _fields(value: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: copy.deepcopy(value[name]) for name in names if name in value}


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _ids(value: Any, *, name: str, index: dict[str, Any], unique: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or item not in index for item in value):
        raise ValueError(f"{name} must reference original manifest IDs")
    if unique and len(set(value)) != len(value):
        raise ValueError(f"{name} must not repeat primary responsibility")
    return list(dict.fromkeys(value))


def _cells(unit: dict[str, Any], view: ModelSourceView) -> list[list[Any]]:
    cells = unit["cells"]
    if not isinstance(cells, list) or any(not isinstance(cell, dict) for cell in cells):
        raise ValueError("Cells must preserve their original structural objects")
    # A separator inside a cell is ambiguous only if the cell boundaries are
    # discarded.  Validate reconstruction and retain the exact half-open range
    # for every cell, including blank ones.  Never infer ranges if it disagrees.
    reconstructable = (all(isinstance(cell.get("text"), str) for cell in cells)
                       and " | ".join(cell["text"] for cell in cells) == unit["text"])
    result = []
    cursor = 0
    for cell in cells:
        projected = view.project(_fields(cell, _CELL_FIELDS))
        values = []
        for field in _CELL_COLUMNS[:-1]:
            original_name = {"merged_from_id": "merged_from_source", "source_id": "source_path"}.get(field, field)
            values.append(copy.deepcopy(projected.get(field, projected.get(original_name, _ABSENT))))
        if reconstructable:
            binding = [cursor, cursor + len(cell["text"])]
            cursor = binding[1] + 3
        elif "text" in cell:
            binding = {"text": copy.deepcopy(cell["text"])}
        else:
            binding = copy.deepcopy(_ABSENT)
        result.append([*values, binding])
    return result


def compact_section_prompt(manifest: dict[str, Any], task: dict[str, Any],
                           max_initial_characters: int = 60000, *, material_index=None) -> tuple[str, list[dict[str, Any]]]:
    """Return the actual bounded prompt and exact ranges embedded in that prompt.

    CONTEXT IDs have admission priority; original ordering inside each list is
    retained.  All primary/context IDs remain in navigation even when no whole
    unit fits.  `exposed` lists only embedded source characters, not a read or
    understanding receipt.  The budget covers prefix, JSON and structural notes,
    not merely the source rows.  It is Unicode characters, not model tokens.
    """
    if material_index is not None:
        return _material_section_prompt(manifest, task, max_initial_characters, material_index)
    if type(max_initial_characters) is not int or max_initial_characters < 1:
        raise ValueError("max_initial_characters must be a positive integer")
    if not isinstance(manifest, dict) or not isinstance(task, dict):
        raise ValueError("A manifest and task object are required")
    if not isinstance(task.get("id"), str) or not task["id"]:
        raise ValueError("Task ID is required")
    units = manifest.get("units")
    tables = manifest.get("tables", [])
    if not isinstance(units, list) or not isinstance(tables, list):
        raise ValueError("Manifest units and tables must be lists")
    index: dict[str, dict[str, Any]] = {}
    for unit in units:
        if (not isinstance(unit, dict) or not isinstance(unit.get("id"), str)
                or not unit["id"] or unit["id"] in index or not isinstance(unit.get("text"), str)):
            raise ValueError("Manifest needs unique IDs and unchanged unit text")
        index[unit["id"]] = unit
    table_index: dict[str, dict[str, Any]] = {}
    for table in tables:
        if (not isinstance(table, dict) or not isinstance(table.get("id"), str)
                or not table["id"] or table["id"] in table_index):
            raise ValueError("Manifest needs unique table IDs")
        table_index[table["id"]] = table
    primary = _ids(task.get("primary_evidence_ids"), name="primary_evidence_ids", index=index, unique=True)
    context = _ids(task.get("context_evidence_ids", []), name="context_evidence_ids", index=index)
    headings = task.get("headings", [])
    if not isinstance(headings, list) or any(not isinstance(item, dict) or item.get("evidence_id") not in index for item in headings):
        raise ValueError("Headings must reference original manifest evidence")
    wanted_tables = set(_ids(task.get("table_ids", []), name="table_ids", index=table_index))
    ordered_ids = list(dict.fromkeys(context + primary))
    for evidence_id in ordered_ids:
        table_id = index[evidence_id].get("table_id")
        ancestors: set[str] = set()
        while table_id:
            if table_id not in table_index or table_id in ancestors:
                raise ValueError("Invalid table parent relationship")
            wanted_tables.add(table_id)
            ancestors.add(table_id)
            table_id = table_index[table_id].get("parent_table_id")
    # Validate even table metadata which has no emitted source row.
    for table_id in list(wanted_tables):
        ancestors = set()
        while table_id:
            if table_id not in table_index or table_id in ancestors:
                raise ValueError("Invalid table parent relationship")
            wanted_tables.add(table_id)
            ancestors.add(table_id)
            table_id = table_index[table_id].get("parent_table_id")
    table_rows = []
    for table in tables:
        if table["id"] in wanted_tables:
            _ids(table.get("header_evidence_ids", []), name="table header", index=index)
            row = _fields(table, _TABLE_FIELDS)
            if "story" in table.get("context", {}):
                row["story"] = table["context"]["story"]
            table_rows.append(row)
    view = ModelSourceView(manifest)
    navigation = view.project({"id": task["id"], "primary_evidence_ids": primary,
                               "context_evidence_ids": context,
                               "headings": [_fields(heading, _HEADING_FIELDS) for heading in headings],
                               "tables": table_rows,
                               "context_roles": {item["evidence_id"]: copy.deepcopy(item.get("roles", []))
                                                 for item in task.get("context_references", [])},
                               "unresolved_context_references": copy.deepcopy(task.get("unresolved_context_references", []))})
    cell_origins: dict[str, list[str]] = {}
    for unit in units:
        for cell in unit.get("cells", []):
            if cell.get("source_path"):
                origins = cell_origins.setdefault(cell["source_path"], [])
                if unit["id"] not in origins:
                    origins.append(unit["id"])
    merged_origins = {}
    for evidence_id in ordered_ids:
        for cell in index[evidence_id].get("cells", []):
            origin = cell.get("merged_from_source")
            if origin:
                alias = view.project({"merged_from_source": origin})
                address = alias.get("merged_from_id", alias.get("merged_from_source"))
                merged_origins[address] = cell_origins.get(origin)
    document = {"schema": "document-section-input-v1", "task": navigation,
                "source_units": [], "source_aliases": "ModelSourceView.source_map",
                "cell_fields": list(_CELL_COLUMNS),
                "cell_encoding": "cells每项按cell_fields位置对应，长度相同；absent_source_field:true表示原字段不存在，与null不同。text_binding=[start,end]表示本行原文半开字符区间；{text:原文}是无法校验拼接时保留的完整单元格文字。",
                "merged_origin_evidence_ids": merged_origins,
                "structure_note": _NOTE}
    if "document_sha256" in manifest:
        document["document_sha256"] = manifest["document_sha256"]
    metadata_size = len(_PREFIX) + len(_encode(document))
    if metadata_size > max_initial_characters:
        raise SectionInputBudgetError(max_initial_characters, metadata_size)
    primary_set = set(primary)
    admitted: list[dict[str, Any]] = []
    encoded_rows: list[str] = []
    exposed: list[dict[str, Any]] = []
    used = metadata_size
    for evidence_id in ordered_ids:
        unit = index[evidence_id]
        row = {"evidence_id": evidence_id, "start": 0, "end": len(unit["text"]),
               "role": "PRIMARY" if evidence_id in primary_set else "CONTEXT",
               "text": unit["text"]}
        row.update(_fields(unit, _UNIT_FIELDS))
        if "source_path" in unit:
            row["source_path"] = unit["source_path"]
        unit_context = unit.get("context", {})
        if "story" in unit_context:
            row["story"] = unit_context["story"]
        if "parent_table_id" in unit:
            row["parent_table_id"] = copy.deepcopy(unit["parent_table_id"])
        elif "parent_table_id" in unit_context:
            row["parent_table_id"] = copy.deepcopy(unit_context["parent_table_id"])
        if "cells" in unit:
            row["cells"] = _cells(unit, view)
        row = view.project(row)
        encoded = _encode(row)
        addition = len(encoded) + bool(encoded_rows)  # Array comma after the first row.
        if used + addition <= max_initial_characters:
            admitted.append(row)
            encoded_rows.append(encoded)
            exposed.append({"evidence_id": evidence_id, "start": 0, "end": len(unit["text"])})
            used += addition
    document["source_units"] = admitted
    prompt = _PREFIX + _encode(document)
    if len(prompt) != used or len(prompt) > max_initial_characters:
        raise AssertionError("Actual section prompt disagrees with whole-prompt admission accounting")
    return prompt, exposed


_MATERIAL_NOTE = (
    "材料按来源章节和表格组织；units.text是canonical原句/整行，evidence_id可直接引用或read_source。"
    "已提供的原文可以直接理解和保存，无须为证明阅读重复调用；未提供或上下文不足时再回查。"
    "section_ref/packet_ref仅供导航，不是已提取事实。任务编号及canonical章节E编号可read_section，"
    "#/sections/引用用read_section，#/packets/引用用read_source；本地引用绑定当前document_sha256。"
    "scope=document保留全文搜索。"
    "PRIMARY是本任务责任，CONTEXT是原文背景；材料出现不代表语义完整或审核通过。"
    "header_kind=first_row_candidate仅是首行候选，不等于明确表头。"
    "表格行的cells=[列起点,跨列数,原文起点,原文终点]，仅在需要消除歧义时给出；"
    "无法校验行内拼接时cells第三项{text}保留完整单元格；merged_origins是合并格原单元的E编号，"
    "继承背景不等于本行新增字符。"
    "历史比较须分别保留对应年份、前后状态、位置和原文限定词，跨段判断同时附相关原文出处；"
    "不要把未新增改写成未加重，也不要用概要替代原文已有的明细。"
)


def _material_section_prompt(manifest, task, budget, material_index):
    """One compact material representation; canonical whole units are the source.

    MinerU packets decide grouping/context, not facts or text substitutions.
    Unmapped units keep canonical fallback access and initial admission.  The
    ledger receives only the exact units actually embedded under the full budget.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError("max_initial_characters must be a positive integer")
    if not isinstance(manifest, dict) or not isinstance(task, dict) or not task.get("id"):
        raise ValueError("A source manifest and identified task are required")
    material_index.validate_manifest(manifest)
    units = manifest["units"]
    index = {unit["id"]: unit for unit in units}
    if len(index) != len(units) or any(not isinstance(u.get("text"), str) for u in units):
        raise ValueError("Manifest needs unique IDs and unchanged unit text")
    primary = _ids(task.get("primary_evidence_ids"), name="primary_evidence_ids", index=index, unique=True)
    primary_set = set(primary)
    context = _ids(task.get("context_evidence_ids", []), name="context_evidence_ids", index=index)
    groups = {}
    membership = {}
    unresolved = []

    def source_id(span):
        if not isinstance(span, dict) or span.get("evidence_id") not in index:
            raise ValueError("Material span has no canonical source")
        identifier = span["evidence_id"]
        start, end = span.get("start"), span.get("end")
        if (type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(index[identifier]["text"])
                or span.get("text") != index[identifier]["text"][start:end]):
            raise ValueError("Material span differs from its canonical source")
        return identifier

    for packet in material_index.packets:
        resolved = material_index.packet(packet["id"])
        identifiers = list(dict.fromkeys(source_id(span) for span in resolved.get("spans", [])))
        relevant = [identifier for identifier in identifiers if identifier in primary_set]
        if not relevant:
            continue
        key = packet["id"]
        groups[key] = {"packet_ref": key, "section_ref": packet.get("section_id"), "units": []}
        for identifier in relevant:
            membership.setdefault(identifier, key)
        for span in resolved.get("context", []):
            identifier = source_id(span)
            if identifier not in primary_set and identifier not in context:
                context.append(identifier)
        for item in resolved.get("unresolved", []):
            if item not in unresolved:
                unresolved.append(copy.deepcopy(item))

    # Headers/merged origins are source dependencies, not inferred values. Keep
    # IDs and source characters once; never repeat long paths or cell metadata.
    table_index = {table["id"]: table for table in manifest.get("tables", [])}
    origins = {}
    for unit in units:
        for cell in unit.get("cells", []):
            if cell.get("source_path"):
                origins.setdefault(cell["source_path"], []).append(unit["id"])
    wanted_tables = set()
    ordered = list(dict.fromkeys(context + primary))
    for identifier in ordered:
        unit = index[identifier]
        table_id = unit.get("table_id")
        seen = set()
        while table_id:
            if table_id in seen or table_id not in table_index:
                raise ValueError("Invalid table parent relationship")
            seen.add(table_id); wanted_tables.add(table_id)
            table = table_index[table_id]
            for header in _ids(table.get("header_evidence_ids", []), name="table header", index=index):
                if header not in primary_set and header not in context:
                    context.append(header)
            table_id = table.get("parent_table_id")
        for cell in unit.get("cells", []):
            for origin_id in origins.get(cell.get("merged_from_source"), []):
                if origin_id not in primary_set and origin_id not in context:
                    context.append(origin_id)
    ordered = list(dict.fromkeys(context + primary))
    context = [identifier for identifier in context if identifier not in primary_set]
    sections = [{"section_ref": section["id"], "title": section.get("title"), "parent": section.get("parent")}
        for section in material_index.sections if section["id"] in set(task.get("material_section_refs", []))]
    document = {"schema": "document-section-material-input-v1", "document_sha256": manifest["document_sha256"], "task": {
        "id": task["id"], "primary_evidence_ids": primary, "context_evidence_ids": context,
        "sections": sections}, "materials": [],
        "tables": [{"id": table["id"], "parent": table.get("parent_table_id"),
                    "headers": table.get("header_evidence_ids", []), "header_kind": table.get("header_kind")}
                   for table in manifest.get("tables", []) if table["id"] in wanted_tables],
        "unresolved_material_references": unresolved,
        "structure_note": _MATERIAL_NOTE}
    metadata_size = len(_PREFIX) + len(_encode(document))
    if metadata_size > budget:
        raise SectionInputBudgetError(budget, metadata_size)
    admitted_groups = {}
    exposed = []
    for identifier in ordered:
        unit = index[identifier]
        row = {"evidence_id": identifier, "text": unit["text"],
               "role": "PRIMARY" if identifier in primary_set else "CONTEXT"}
        if unit.get("table_id"):
            row.update({key: unit[key] for key in ("table_id", "row_index", "segment_index") if key in unit})
        cells = unit.get("cells", [])
        if cells:
            reconstructable = (all(isinstance(c.get("text"), str) for c in cells)
                and " | ".join(c["text"] for c in cells) == unit["text"])
            ambiguous = (not reconstructable or any(" | " in c.get("text", "") or c.get("column_span", 1) != 1
                or c.get("vertical_merge") or c.get("merged_from_source") for c in cells))
            if ambiguous:
                layout = []; cursor = 0
                for cell in cells:
                    end = cursor + len(cell.get("text", ""))
                    layout.append([cell.get("column_index"), cell.get("column_span"), cursor, end] if reconstructable
                        else [cell.get("column_index"), cell.get("column_span"), {"text": cell.get("text")}])
                    cursor = end + 3
                row["cells"] = layout
            inherited = [{"column": cell.get("column_index"), "evidence_ids": list(dict.fromkeys(origins.get(cell["merged_from_source"], [])))}
                for cell in cells if cell.get("merged_from_source")]
            if inherited:
                row["merged_origins"] = inherited
        key = membership.get(identifier, "canonical-fallback") if identifier in primary_set else "shared-context"
        group = admitted_groups.get(key)
        new_group = group is None
        if new_group:
            group = copy.deepcopy(groups.get(key, {"group": key, "units": []}))
            document["materials"].append(group)
        group["units"].append(row)
        if len(_PREFIX) + len(_encode(document)) <= budget:
            admitted_groups[key] = group
            exposed.append({"evidence_id": identifier, "start": 0, "end": len(unit["text"])})
        else:
            group["units"].pop()
            if new_group:
                document["materials"].remove(group)
    prompt = _PREFIX + _encode(document)
    if len(prompt) > budget:
        raise AssertionError("Material prompt exceeded its whole-message character budget")
    return prompt, exposed
