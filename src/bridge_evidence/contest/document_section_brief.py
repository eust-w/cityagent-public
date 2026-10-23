"""Objective scope and input inventory for a section Agent's first request.

This module does not decide which bridge facts exist or which business fields a
chapter owns. It describes the trusted source partition and the exact source
ranges embedded by the existing input builder. Delivery remains the ledger's
responsibility; understanding and unresolved questions remain the model's.
"""
from __future__ import annotations

import json
from typing import Any

from .document_section_input import SectionInputBudgetError, compact_section_prompt
from .document_section_prompts import CONCISE_SECTION_POLICY, validate_section_prompt_variant
from .run_state import json_digest


_PREFIX = "本次只处理下列分工原文；先保存已有依据的条目，再针对具体缺口回查。\n"
_POLICY = {
    "first_action": "先理解本组已经内嵌的PRIMARY，直接submit_section保存能确认的条目；无需先读目录或重新登记阅读。必要缺口仍可先查证。",
    "responsibility": "本组只抽取PRIMARY中的相关信息；CONTEXT用于桥名、年份、对象和上下文关联，不因出现背景就接管它的独立抽取。",
    "other_tasks": "其他分工独立处理各自PRIMARY中的评分、病害和建议。字段定义是共同保存格式，不是本组必须补齐的整份报告清单；没有按标题推断字段专属归属。",
    "summary": "PRIMARY内的摘要仍须保留其明确给出的基本信息、病害和建议；无需为了另加同义正文出处而重抽其他章节，重复或差异交主Agent核对。",
    "navigation": "章节标题、目录条目和阅读地图是原文导航，不是要求逐章访问的待办清单；目录中提及其他章不扩大本组职责。",
    "lookup": "已有条目的证据缺失、指代、年份、左右幅、构件、表头或冲突不清时，可全文定点回查；跨分工未决问题交unresolved。不要为补齐其他分工的独立字段而遍历全文。",
    "technical_diagnostics": "技术别名映射诊断只说明MinerU别名与canonical E的映射状态，不自动意味着业务缺证；已内嵌canonical E原文可直接使用。真实缺口仍由模型判断。",
    "finish": "本组相关条目已保存、未决问题已列明即可complete=true；不等待其他分工字段齐全。内嵌数量不代表已理解、已提取或审核通过。",
}


def speed_section_input(prompt: str) -> str:
    """Project navigation only, after admission; source text/IDs never change.

    The caller saves the complete input separately. Material aliases remain
    usable by the reader, but are not a second citation vocabulary in this
    first request. Missing navigation can be obtained with read_outline.
    """
    prefix, raw = prompt.split("\n", 1)
    document = json.loads(raw)
    brief = document.get("assignment_brief")
    if brief:
        brief["work_policy"].update(
            first_action="直接用save_basic_info/save_defects/save_recommendations保存本分工主要条目；完成后handoff_section(notes)。",
            summary="摘要/概述主要交基本信息，避免重复提交其他分工负责的整套病害和建议。独有事实或差异仍保留；不要只凭标题丢弃信息。",
            finish="已有条目保存后handoff_section(notes)交接，缺项可如实记录，不为补齐其他分工反复查阅。")
        for scope in [brief["scope"], *brief["other_tasks"]]:
            for section in scope["sections"]:
                section.pop("section_ref", None)
                section.pop("parent", None)
    for group in document.get("materials", []):
        group.pop("packet_ref", None)
        group.pop("section_ref", None)
    if "unresolved_material_references" in document:
        diagnostics = document.pop("unresolved_material_references")
        document["material_mapping_diagnostic_count"] = len(diagnostics)
    if "materials" in document:
        document["structure_note"] = (
            "只用units.evidence_id中的E编号引用和read_source；多个出处用多个source_refs对象，不能拼路径或编号范围。"
            "各行text保持原文，表格headers是表头E编号；header_kind=first_row_candidate仅是候选。"
            "cells=[列起点,跨列数,原文起点,原文终点]或第三项{text}；merged_origins为合并格来源E编号。"
            "PRIMARY为本任务范围，CONTEXT仅供关联；年份、左右幅、对象关系由模型理解。"
            "章节标题仅导航，必要时read_outline取得章节编号。映射诊断已落盘，不代表已给出的E原文不可用。")
    return prefix + "\n" + json.dumps(document, ensure_ascii=False, separators=(",", ":"))


def _ids(task: dict, units: dict) -> list[str]:
    values = task.get("primary_evidence_ids")
    if (not isinstance(task.get("id"), str) or not task["id"] or not isinstance(values, list)
            or any(not isinstance(e, str) or e not in units for e in values)
            or len(values) != len(set(values))):
        raise ValueError("Section brief requires a task with unique canonical PRIMARY IDs")
    return values


def _validated_tasks(manifest: dict, task: dict, task_plan: dict | None, units: dict) -> list[dict]:
    _ids(task, units)
    if task_plan is None:
        return [task]
    if (not isinstance(task_plan, dict)
            or task_plan.get("document_sha256") != manifest.get("document_sha256")
            or task_plan.get("manifest_sha256") != json_digest(manifest)):
        raise ValueError("Section brief task plan belongs to a different document or manifest")
    tasks = task_plan.get("tasks")
    if not isinstance(tasks, list) or any(not isinstance(t, dict) for t in tasks):
        raise ValueError("Section brief task plan requires original task objects")
    task_ids = [_task.get("id") for _task in tasks]
    if any(not isinstance(identifier, str) for identifier in task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("Section brief task plan requires unique task IDs")
    current = [candidate for candidate in tasks if candidate.get("id") == task["id"]]
    if len(current) != 1 or current[0] != task:
        raise ValueError("Current task differs from the trusted task plan")
    assigned = [e for candidate in tasks for e in _ids(candidate, units)]
    if len(set(assigned)) != len(assigned) or set(assigned) != set(units):
        raise ValueError("Section brief task plan must partition this manifest exactly once")
    return tasks


def _ranges(ids: list[str], positions: dict[str, int]) -> list[dict[str, Any]]:
    """Compact contiguous original-unit ranges; never infer an E-number format."""
    groups: list[list[str]] = []
    for identifier in sorted(ids, key=positions.__getitem__):
        if not groups or positions[identifier] != positions[groups[-1][-1]] + 1:
            groups.append([])
        groups[-1].append(identifier)
    return [{"first": group[0], "last": group[-1], "unit_count": len(group)} for group in groups]


def _section_maps(manifest: dict, tasks: list[dict], material_index) -> dict[str, list[dict]]:
    """Use exact material spans or the plan's headings, never title classifiers."""
    maps = {task["id"]: [] for task in tasks}
    primary = {task["id"]: set(task["primary_evidence_ids"]) for task in tasks}
    units = {unit["id"]: unit for unit in manifest["units"]}
    if material_index is not None:
        material_index.validate_manifest(manifest)
        for section in material_index.sections:
            resolved = material_index.section(section["id"], include_descendants=False, include_context=False)
            section_ids = set()
            for span in resolved.get("spans", []):
                eid, start, end = span.get("evidence_id"), span.get("start"), span.get("end")
                if (eid not in units or type(start) is not int or type(end) is not int
                        or not 0 <= start <= end <= len(units[eid]["text"])
                        or span.get("text") != units[eid]["text"][start:end]):
                    raise ValueError("Material section map differs from canonical source")
                section_ids.add(eid)
            for task in tasks:
                relevant = section_ids & primary[task["id"]]
                if relevant:
                    row = {"section_ref": section["id"], "title": section.get("title")}
                    if section.get("parent"):
                        row["parent"] = section["parent"]
                    if section.get("region") and section["region"] != "content":
                        row["region"] = section["region"]
                    if not section_ids <= primary[task["id"]]:
                        row["section_shared_between_tasks"] = True
                    maps[task["id"]].append(row)
        return maps
    for task in tasks:
        for heading in task.get("headings", []):
            eid = heading.get("evidence_id")
            if eid not in units or heading.get("title") not in (units[eid]["text"], units[eid]["text"].strip()):
                raise ValueError("Task heading differs from canonical source")
            if eid in primary[task["id"]]:
                maps[task["id"]].append({"section_ref": eid, "title": heading["title"],
                                        "basis": "task_plan_heading"})
    return maps


def build_section_brief(manifest: dict, task: dict, exposed: list[dict], *,
                        material_index=None, task_plan: dict | None = None) -> dict:
    """Describe source scope and request embedding, without granting coverage."""
    if not isinstance(manifest, dict) or not isinstance(task, dict):
        raise ValueError("Section brief requires a manifest and task")
    units = {unit["id"]: unit for unit in manifest["units"]}
    if len(units) != len(manifest["units"]):
        raise ValueError("Section brief manifest requires unique canonical IDs")
    tasks = _validated_tasks(manifest, task, task_plan, units)
    positions = {unit["id"]: i for i, unit in enumerate(manifest["units"])}
    sections = _section_maps(manifest, tasks, material_index)
    seen: dict[str, list[tuple[int, int]]] = {}
    for span in exposed:
        eid, start, end = span.get("evidence_id"), span.get("start"), span.get("end")
        if (eid not in units or type(start) is not int or type(end) is not int
                or not 0 <= start <= end <= len(units[eid]["text"])):
            raise ValueError("Embedded source span is outside this manifest")
        seen.setdefault(eid, []).append((start, end))
    complete = set()
    for eid, spans in seen.items():
        cursor = 0
        for start, end in sorted(spans):
            if start > cursor:
                break
            cursor = max(cursor, end)
        if cursor == len(units[eid]["text"]):
            complete.add(eid)
    primary = task["primary_evidence_ids"]
    missing = [eid for eid in primary if eid not in complete]
    def scope(candidate, *, other=False):
        rows = sections[candidate["id"]]
        # Other Agents' full subheading maps are not this Agent's reading plan.
        # Retain their trusted parent headings and exact assigned source ranges.
        if other and material_index is not None:
            section_ids = {row["section_ref"] for row in rows}
            top = [row for row in rows if row.get("parent") not in section_ids]
        else:
            top = rows
        result = {"task_id": candidate["id"], "primary_unit_count": len(candidate["primary_evidence_ids"]),
            "source_ranges": _ranges(candidate["primary_evidence_ids"], positions),
            "sections": top}
        if len(top) != len(rows):
            result["additional_subsection_count"] = len(rows) - len(top)
        return result
    return {"schema": "document-section-brief-v1", "scope": scope(task),
        "input_provision": {"meaning": "embedded_in_this_request_not_delivery_or_understanding",
            "primary_total_units": len(primary), "primary_fully_embedded_units": len(primary) - len(missing),
            "all_primary_units_embedded": not missing, "unembedded_or_partial_primary_ids": missing,
            "source_reads_needed_for_unembedded_ids": [eid for eid in missing if units[eid]["text"]],
            "context_embedded_units": len(set(seen) - set(primary)),
            "counts_scope": "only_assigned_parsed_text_and_tables_not_entire_original_report"},
        "other_tasks_map_available": task_plan is not None,
        "other_tasks": [scope(candidate, other=True) for candidate in tasks if candidate["id"] != task["id"]],
        "work_policy": dict(_POLICY)}


def briefed_section_prompt(manifest: dict, task: dict, max_initial_characters: int = 60000, *,
                           material_index=None, task_plan: dict | None = None,
                           prompt_variant: str = "scoped_v1") -> tuple[str, list[dict]]:
    """Keep one source representation, with the brief inside the total budget."""
    validate_section_prompt_variant(prompt_variant)
    if type(max_initial_characters) is not int or max_initial_characters < 1:
        raise ValueError("max_initial_characters must be a positive integer")
    source_budget = max_initial_characters
    while source_budget > 0:
        source_prompt, exposed = compact_section_prompt(manifest, task, source_budget, material_index=material_index)
        document = json.loads(source_prompt.split("\n", 1)[1])
        brief = build_section_brief(manifest, task, exposed, material_index=material_index, task_plan=task_plan)
        # The old material task map omitted front matter/TOC and duplicates this
        # complete scope map. Source units, dependencies and exact IDs stay intact.
        document["task"].pop("sections", None)
        document = {"assignment_brief": brief, **document}
        prompt = _PREFIX + json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if len(prompt) <= max_initial_characters:
            if prompt_variant == "concise_v2":
                # Freeze source admission using the baseline first: shorter
                # instructions must not admit extra original text in arm B.
                document["assignment_brief"]["work_policy"] = dict(CONCISE_SECTION_POLICY)
                variant_prompt = _PREFIX + json.dumps(document, ensure_ascii=False,
                    separators=(",", ":"), allow_nan=False)
                if len(variant_prompt) > len(prompt):
                    raise ValueError("Concise section instructions must not enlarge the baseline source budget")
                return variant_prompt, exposed
            return prompt, exposed
        # Re-run whole-unit admission and then recompute the inventory. A unit
        # which no longer fits is never described as embedded in the new brief.
        source_budget -= len(prompt) - max_initial_characters
    raise SectionInputBudgetError(max_initial_characters, len(prompt))
