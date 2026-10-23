"""Bounded verbatim gap material from existing child assignments only.

This is source organization, never a disease detector or a completeness claim.
It does not grant reading coverage: register the final validated message with
the main Agent's source-delivery ledger and await actual upstream confirmation.
"""
from copy import deepcopy

from .document_handoff_bundle import _Sources, serialize_handoff_bundle, validate_handoff_bundle


def _missing(rows, identifier, length):
    cursor, missing = 0, []
    ranges = sorted((row["start"], row["end"]) for row in rows.values()
                    if row["evidence_id"] == identifier)
    for start, end in ranges:
        if cursor < start:
            missing.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < length:
        missing.append((cursor, length))
    return missing


def add_gap_material(manifest, bundle, child_summary, failed_tasks, *, max_additional_characters=8000):
    if type(max_additional_characters) is not int or max_additional_characters < 0:
        raise ValueError("Additional gap-material budget must be a nonnegative integer")
    validate_handoff_bundle(manifest, bundle, max_characters=68000)
    original_size = len(serialize_handoff_bundle(bundle))
    limit = min(68000, original_size + max_additional_characters)
    result = deepcopy(bundle)
    sources = _Sources(manifest)
    failed = {row.get("task_id", row.get("id")) if isinstance(row, dict) else row
              for row in failed_tasks}
    # Compact bundle scopes contain boundary navigation; they are not enough
    # to reconstruct possibly non-contiguous original task assignments.
    scopes, seen_tasks = [], set()
    for child in child_summary:
        task = child["task_id"]
        if task in seen_tasks:
            raise ValueError("Child task IDs must be unique")
        seen_tasks.add(task)
        ids = child.get("primary_evidence_ids", [])
        if not isinstance(ids, list) or any(not isinstance(eid, str) or eid not in sources.units for eid in ids):
            raise ValueError("Child assignment references a source outside this canonical manifest")
        scopes.append((task, list(dict.fromkeys(ids))))
    scopes = [scope for scope in scopes if scope[0] in failed] + [scope for scope in scopes if scope[0] not in failed]
    eligible = list(dict.fromkeys(eid for _, ids in scopes for eid in ids if sources.units[eid]["text"]))
    added, groups = [], []
    pending = {eid for eid in eligible if _missing(result["source_map"], eid, len(sources.units[eid]["text"]))}

    def navigation(current_sources, new_ids, context_groups, pending_ids):
        return {"added_evidence_ids": new_ids,
            "not_added_unit_count": len(pending_ids),
            "assigned_unit_count": len(eligible), "context_groups": context_groups,
            "partial_original_source_only": True, "source_delivery_confirmed": False,
            "meaning": "逐字原文导航；仅补既有子Agent分工范围及必要表格上下文，不是模型事实、语义审核或完整覆盖证明。候选表头仍须模型判断。"}

    result["gap_material"] = navigation(result["source_map"], added, groups, pending)
    if len(serialize_handoff_bundle(result)) > limit:
        return deepcopy(bundle)  # Even navigation cannot fit; do not claim additions.

    def group_for(identifier):
        # Full physical row including canonical segments, header/candidate
        # header and merged-cell anchors are one indivisible budget group.
        queue, refs, seen = [{"evidence_id": identifier}], [], set()
        while queue:
            ref = queue.pop(0)
            # A merged-cell anchor may be a narrow canonical span. Supply its
            # whole unit as well so every new E ID can safely mean a full unit
            # in the main Agent's simple evidence-only save interface.
            ref = {"evidence_id": ref["evidence_id"]}
            key, row = sources.source(ref)
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
            unit = sources.units[row["evidence_id"]]
            if unit.get("table_id") is not None:
                if ref.get("start", 0) == 0 and ref.get("end", len(unit["text"])) == len(unit["text"]):
                    queue.extend({"evidence_id": eid} for eid, sibling in sources.units.items()
                        if sibling["text"] and sibling.get("table_id") == unit["table_id"]
                        and sibling.get("row_index") == unit.get("row_index") and eid != unit["id"])
                headers, merged = sources.related(ref)
                queue.extend(headers + merged)
        return refs

    for task, ids in scopes:
        for identifier in ids:
            unit = sources.units[identifier]
            if not unit["text"] or not _missing(result["source_map"], identifier, len(unit["text"])):
                continue
            proposal = deepcopy(result)
            new_ids = added.copy()
            for ref in group_for(identifier):
                _, complete = sources.source(ref)
                eid = complete["evidence_id"]
                # Subtract all already supplied source intervals. Remaining
                # fragments finish existing units; no budget-driven clipping.
                for start, end in _missing(proposal["source_map"], eid, len(sources.units[eid]["text"])):
                    start, end = max(start, complete["start"]), min(end, complete["end"])
                    if start >= end:
                        continue
                    key, row = sources.source({"evidence_id": eid, "start": start, "end": end})
                    proposal["source_map"][key] = row
                    if eid not in new_ids:
                        new_ids.append(eid)
            context = {"task_id": task, "primary_evidence_id": identifier,
                "source_keys": [key for key, row in proposal["source_map"].items()
                                if key not in result["source_map"]]}
            if unit.get("table_id"):
                table = sources.tables.get(unit["table_id"], {})
                context.update(table_id=unit["table_id"], row_index=unit.get("row_index"),
                    parent_table_id=table.get("parent_table_id"), header_kind=table.get("header_kind"))
            new_groups = groups + [context]
            proposal_pending = pending - set(new_ids)
            proposal["gap_material"] = navigation(proposal["source_map"], new_ids, new_groups, proposal_pending)
            if len(serialize_handoff_bundle(proposal)) <= limit:
                validate_handoff_bundle(manifest, proposal, max_characters=limit)
                result, added, groups, pending = proposal, new_ids, new_groups, proposal_pending
    validate_handoff_bundle(manifest, result, max_characters=limit)
    return result
