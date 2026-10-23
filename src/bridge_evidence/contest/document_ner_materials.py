"""Opt-in NER-assisted source admission; never creates facts or read receipts.

The immutable manifest remains the source authority. Lexical mentions only
prioritize whole units. Dates, scores, history, conclusions and advice have a
separate safety net because this three-prompt model was not trained for them.
Omitted units remain addressable through the existing document tools.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .run_state import atomic_json, json_digest

SCHEMA = "cityagent-ner-materials-v1"
PROMPTS = ("构件及部位", "病害类型", "尺寸或距离")
PROTECT = re.compile(r"评分|得分|技术状况|养护|维修|建议|结论|风险|安全|耐久|检测时间|检测日期|检测单位|委托单位|上次|上一次|前次|新增|加重|发展|历次|[12][0-9]{3}\s*年|[〇○零一二三四五六七八九]{4}年")


def build_index(manifest, mentions, *, inference=None):
    """Bind exact model spans to canonical IDs, expanding complete table families."""
    units = manifest["units"]
    by_id = {u["id"]: u for u in units}
    selected, reasons, checked = set(), {}, []

    def keep(eid, reason):
        if eid in by_id:
            selected.add(eid)
            reasons.setdefault(eid, set()).add(reason)

    for m in mentions:
        eid, a, b = m.get("evidence_id"), m.get("start"), m.get("end")
        if (eid not in by_id or type(a) is not int or type(b) is not int
                or not 0 <= a < b <= len(by_id[eid]["text"])
                or m.get("text") != by_id[eid]["text"][a:b] or m.get("type") not in PROMPTS):
            raise ValueError("NER mention does not match canonical source")
        checked.append(deepcopy(m))
        keep(eid, "ner_lexical_mention_not_a_confirmed_fact")
    cover = 0
    for u in units:
        if not u.get("table_id") and u["text"].strip() and cover < 16:
            keep(u["id"], "cover_prefix")
            cover += 1
        if PROTECT.search(u["text"]):
            keep(u["id"], "non_ner_business_context")
    if not checked:
        for u in units:
            keep(u["id"], "no_mentions_full_source_fallback")

    tables = {t["id"]: t for t in manifest.get("tables", [])}
    def root(tid):
        trail = set()
        while tid:
            if tid not in tables or tid in trail:
                raise ValueError("Invalid canonical table family")
            trail.add(tid)
            parent = tables[tid].get("parent_table_id")
            if not parent:
                return tid
            tid = parent

    # Close structural dependencies, including merged cell origins. Table rows
    # are not filtered independently: numeric-only rows can carry real facts.
    origins = {}
    for u in units:
        for c in u.get("cells", []):
            origins.setdefault(c.get("source_path"), set()).add(u["id"])
    while True:
        before = set(selected)
        families = {root(by_id[e]["table_id"]) for e in selected if by_id[e].get("table_id")}
        families.update(root(tid) for tid, t in tables.items()
                        if any(e in selected for e in t.get("header_evidence_ids", [])))
        for u in units:
            if u.get("table_id") and root(u["table_id"]) in families:
                keep(u["id"], "whole_table_family")
        for eid in list(selected):
            u = by_id[eid]
            context = u.get("context", {})
            keep(context.get("section_heading_id"), "section_heading")
            # One-hop context, not an accidental recursive walk to page one.
            if reasons[eid] != {"source_context"}:
                for e in context.get("preceding_evidence_ids", []):
                    keep(e, "source_context")
            if u.get("table_id"):
                for e in tables[u["table_id"]].get("header_evidence_ids", []):
                    keep(e, "table_header_or_candidate")
            for c in u.get("cells", []):
                if c.get("merged_from_source"):
                    for e in origins.get(c["merged_from_source"], []):
                        keep(e, "merged_cell_origin")
        if selected == before:
            break
    return {"schema": SCHEMA, "document_sha256": manifest["document_sha256"],
        "manifest_sha256": json_digest(manifest), "mentions": checked,
        "selected_evidence_ids": [u["id"] for u in units if u["id"] in selected],
        "selection_reasons": {e: sorted(v) for e, v in reasons.items()},
        "deferred_evidence_ids": [u["id"] for u in units if u["id"] not in selected],
        "inference": inference or {}, "not_full_report_coverage": True,
        "statistics": {"source_units": len(units), "selected_units": len(selected),
            "source_characters": sum(len(u["text"]) for u in units),
            "selected_characters": sum(len(u["text"]) for u in units if u["id"] in selected)}}


def load_index(manifest, path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("NER index file SHA differs")
    value = json.loads(raw)
    if (value.get("schema") != SCHEMA or value.get("document_sha256") != manifest["document_sha256"]
            or value.get("manifest_sha256") != json_digest(manifest)):
        raise ValueError("NER index belongs to another document or manifest")
    rebuilt = build_index(manifest, value["mentions"], inference=value.get("inference"))
    if value != rebuilt:
        raise ValueError("NER selection differs from deterministic source admission")
    return value


def pin_index(manifest, config, directory):
    """Freeze verified optional index once; the legacy path has no dependency."""
    path, digest = config.get("ner_index_path"), config.get("ner_index_sha256")
    if not path and not digest:
        return config
    if not path or not digest:
        raise ValueError("Both NER index path and SHA are required")
    value = load_index(manifest, path, digest)
    destination = Path(directory) / "ner-materials.json"
    atomic_json(destination, value)
    return {**config, "ner_index_path": str(destination.resolve()),
            "ner_index_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def project_prompt(manifest, task, prompt, exposed, index):
    """Shrink the actual user message and exactly the same delivery ranges."""
    prefix, raw = prompt.split("\n", 1)
    document = json.loads(raw)
    if "materials" not in document:
        raise ValueError("NER experiment requires the repaired material path")
    selected = set(index["selected_evidence_ids"])
    # Shared cover/heading context is kept even when not selected by the model.
    primary = set(task["primary_evidence_ids"])
    keep = {r["evidence_id"] for r in exposed if r["evidence_id"] in selected or r["evidence_id"] not in primary}
    for group in document["materials"]:
        group["units"] = [u for u in group["units"] if u["evidence_id"] in keep]
    document["materials"] = [g for g in document["materials"] if g["units"]]
    if "assignment_brief" in document:
        provision = document["assignment_brief"]["input_provision"]
        missing = [e for e in task["primary_evidence_ids"] if e not in keep]
        provision.update(primary_fully_embedded_units=len(primary) - len(missing),
            all_primary_units_embedded=not missing, deferred_primary_unit_count=len(missing),
            deferred_source_policy="未提供范围留在全文工具中；仅因当前条目存在明确缺口才回查，不是必须逐个读取的待办。")
        # The full omitted-ID list stays in the immutable index/manifest. It
        # was both repeated token overhead and a misleading reading checklist.
        provision.pop("unembedded_or_partial_primary_ids", None)
        provision.pop("source_reads_needed_for_unembedded_ids", None)
    document["ner_reading_policy"] = (
        "材料由小NER模型候选及独立的评分/历史/日期/结论建议保护层选取，均为完整原文单元；"
        "NER命中不是病害事实，未命中也不代表没有信息。否定、年份、左右幅和关系仍由你判断。"
        "先保存当前材料内的条目；有明确缺口再用全文工具回查。未提供原文不算已读，不声称穷尽。")
    document["ner_deferred_primary_count"] = len(primary - keep)
    return prefix + "\n" + json.dumps(document, ensure_ascii=False, separators=(",", ":")), [r for r in exposed if r["evidence_id"] in keep]


def admitted_primary_complete(manifest, task, exposed, index):
    """First-action routing only, never a claim of full-source reading/quality.

    Use the baseline's save-first behavior when all selected whole PRIMARY
    units are present. Deferred units must not turn an NER view into a command
    to enumerate all missing IDs. Later targeted source tools remain available.
    """
    selected = set(index["selected_evidence_ids"]) & set(task["primary_evidence_ids"])
    delivered = {}
    for r in exposed:
        delivered.setdefault(r["evidence_id"], []).append((r["start"], r["end"]))
    for u in manifest["units"]:
        if u["id"] not in selected:
            continue
        cursor = 0
        for a, b in sorted(delivered.get(u["id"], [])):
            if a > cursor:
                break
            cursor = max(cursor, b)
        if cursor != len(u["text"]):
            return False
    return True
