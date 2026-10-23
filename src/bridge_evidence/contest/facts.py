"""Versioned, immutable report facts. Never infer physical identity from similarity.

Adapters retain the three-table contract verbatim. Location precision is only
claimed when present in the actual parser inventory; whole-mode has document
precision. Strict ingestion validates allowed input, successful call and source
visibility; it is not a substitute for independent execution-receipt validation.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal
from hashlib import sha256
from typing import Any

SCHEMA_VERSION = "cityagent-facts-v1"
NORMALIZATION_VERSION = "identity-measurement-v1"


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _id(kind: str, *parts: Any) -> str:
    return kind + ":" + sha256(_json(parts).encode()).hexdigest()


def _text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def normalize_identity(bridge: str, location: str, component: str = "") -> dict:
    """Canonicalize explicit identifiers only; absent fields remain None."""
    value = (
        re.sub(r"\s+", "", _text(location))
        .replace("左线", "左幅")
        .replace("右线", "右幅")
    )

    def number(pattern: str) -> str | None:
        match = re.search(pattern, value, re.I)
        return str(int(match.group(1))) if match else None

    sides = {
        side for token, side in (("左幅", "left"), ("右幅", "right")) if token in value
    }
    position = next(
        (
            p
            for p in ("跨中", "跨端", "墩顶", "梁端", "内侧", "外侧", "上游", "下游")
            if p in value
        ),
        None,
    )
    kind = component or next(
        (
            p
            for p in (
                "支座",
                "桥墩",
                "桥台",
                "主梁",
                "横梁",
                "盖梁",
                "桥面",
                "伸缩缝",
                "梁",
                "墩",
                "台",
            )
            if p in value
        ),
        "",
    )
    kind = {"墩": "桥墩", "台": "桥台", "梁": "主梁"}.get(kind, kind)
    return {
        "bridge": re.sub(r"\s+", "", _text(bridge)),
        "side": next(iter(sides)) if len(sides) == 1 else None,
        "span": number(r"(?<![\d~～\-至])(?:第)?(\d+)[#号]?跨"),
        "pier": number(r"(?:P)(\d+)(?:号|#)?(?:桥)?墩")
        or number(r"(\d+)[#号]?(?:桥)?墩"),
        "abutment": number(r"(\d+)[#号]?(?:桥)?台"),
        "component": kind or None,
        "component_number": number(r"(\d+)[#号](?:支座|主梁|梁|横梁|盖梁|伸缩缝)"),
        "position": position,
    }


def identity_conflicts(left: dict, right: dict) -> tuple[str, ...]:
    return tuple(
        k
        for k in (
            "bridge",
            "side",
            "span",
            "pier",
            "abutment",
            "component_number",
            "position",
        )
        if left.get(k) is not None and right.get(k) is not None and left[k] != right[k]
    )


def assess_relation(
    left: dict,
    right: dict,
    *,
    left_defect_id=None,
    right_defect_id=None,
    evidence_ids=(),
    evidence_bound=False,
) -> dict:
    """Only an explicit stable defect identifier plus matching scope confirms an edge.

    Fuzzy string scores are never promoted to identity probabilities. Missing
    dimensions are not evidence of equality; conflicts are negative evidence.
    """
    conflicts = identity_conflicts(left, right)
    result = {
        "type": "same-defect",
        "status": "uncertain",
        "evidence_ids": list(evidence_ids),
        "rule_version": NORMALIZATION_VERSION,
        "reason": "insufficient identity evidence",
    }
    if conflicts:
        return {
            **result,
            "status": "rejected",
            "reason": "explicit identity conflict",
            "conflicts": list(conflicts),
        }
    anchored = all(
        left.get(k) and left[k] == right.get(k) for k in ("bridge", "side", "component")
    )
    anchored = anchored and any(
        left.get(k) is not None and left[k] == right.get(k)
        for k in ("span", "pier", "abutment", "component_number")
    )
    if left_defect_id and right_defect_id and left_defect_id != right_defect_id:
        return {
            **result,
            "status": "rejected",
            "reason": "explicit defect identifier conflict",
        }
    if (
        anchored
        and evidence_bound
        and evidence_ids
        and left_defect_id
        and left_defect_id == right_defect_id
    ):
        return {
            **result,
            "status": "confirmed",
            "reason": "explicit defect identifier and compatible physical scope",
        }
    return result


@dataclass(frozen=True)
class Measurement:
    attribute: str
    value: float | None
    lower: float | None
    upper: float | None
    unit: str
    statistic: str
    raw: str
    rule_version: str = NORMALIZATION_VERSION


_UNITS = {
    "mm": ("mm", Decimal("1")),
    "毫米": ("mm", Decimal("1")),
    "cm": ("mm", Decimal("10")),
    "厘米": ("mm", Decimal("10")),
    "m": ("mm", Decimal("1000")),
    "米": ("mm", Decimal("1000")),
    "mm2": ("mm2", Decimal("1")),
    "cm2": ("mm2", Decimal("100")),
    "m2": ("mm2", Decimal("1000000")),
}
_UNIT = r"mm2|cm2|m2|mm|cm|毫米|厘米|m|米"
_MEASUREMENT = re.compile(
    r"(?P<stat>最大|最小|平均|累计|总计)?\s*(?P<attr>宽度|裂缝宽|宽|长度|长|深度|深|面积|厚度|厚)"
    r"\s*[:：=为约≤≥<>]?\s*(?P<a>\d+(?:\.\d+)?)\s*"
    + r"(?:(?P<unita>"
    + _UNIT
    + r")\s*)?"
    + r"(?:(?:～|~|至|—|-)\s*(?P<b>\d+(?:\.\d+)?)\s*(?P<unitb>"
    + _UNIT
    + r")?|(?![～~至—-]))"
)


def parse_measurements(text: str) -> tuple[Measurement, ...]:
    values = []
    for match in _MEASUREMENT.finditer(_text(text)):
        raw = match.group()
        first_unit, second_unit = match["unita"], match["unitb"]
        if not (first_unit or second_unit):
            continue
        unit, factor = _UNITS[first_unit or second_unit]
        first = float(Decimal(match["a"]) * factor)
        second_factor = _UNITS[second_unit][1] if second_unit else factor
        if second_unit and _UNITS[second_unit][0] != unit:
            continue
        second = float(Decimal(match["b"]) * second_factor) if match["b"] else None
        attr = (
            "area"
            if match["attr"] == "面积"
            else "width"
            if "宽" in match["attr"]
            else "length"
            if "长" in match["attr"]
            else "depth"
            if "深" in match["attr"]
            else "thickness"
        )
        stat = {
            "最大": "maximum",
            "最小": "minimum",
            "平均": "mean",
            "累计": "total",
            "总计": "total",
        }.get(match["stat"], "unspecified")
        if "≤" in raw or "<" in raw:
            value, low, high = None, None, first
        elif "≥" in raw or ">" in raw:
            value, low, high = None, first, None
        elif second is not None:
            value, low, high = None, min(first, second), max(first, second)
        else:
            value, low, high = first, None, None
        values.append(Measurement(attr, value, low, high, unit, stat, raw))
    return tuple(values)


@dataclass(frozen=True)
class FactGraph:
    """Canonical JSON backing prevents callers mutating nested graph state."""

    payload_json: str

    def to_dict(self) -> dict:
        return json.loads(self.payload_json)

    @property
    def digest(self) -> str:
        return sha256(self.payload_json.encode()).hexdigest()


def _provenance(record: dict) -> dict:
    raw = record.get("provenance") or []
    if isinstance(raw, list):
        return {
            (p["section"], str(p.get("field", p.get("number", "")))): dict(p)
            for p in raw
        }
    result = {}
    for field, item in raw.get("基本信息", {}).items():
        result[("基本信息", field)] = dict(item)
    for section in ("病害明细", "维修建议"):
        for index, item in enumerate(raw.get(section, []), 1):
            result[(section, str(index))] = dict(item)
    return result


def _source_inventory(record: dict) -> dict:
    manifest = record.get("evidence_manifest") or {}
    sources = {s["id"]: s for s in manifest.get("units", [])}
    for s in (record.get("snapshot") or {}).get("regions", []):
        key = s.get("source_id", s.get("id"))
        if key:
            sources[key] = s
    return sources


def _year(value: Any) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return int(match.group()) if match else None


def _source_inspection_year(excerpts: list[str], provenance: dict) -> int | None:
    """A historical row cannot inherit the enclosing report's current cycle."""
    years = set()
    for text in excerpts:
        for match in re.finditer(
            r"(?<![A-Za-z0-9_.:/#-])(?:19|20)\d{2}(?![A-Za-z0-9])", text
        ):
            before, after = text[:match.start()], text[match.end():]
            if re.search(r"报告(?:日期|时间|编制日期|编制时间)?[：:\s]*$", before):
                continue
            if (
                re.match(r"年|[-/.]\d{1,2}(?!\d)", after)
                or text.strip() == match.group()
                or re.search(r"(?:检测|定检|巡检)(?:年份|年度|日期|时间)?[：:\s]*$", before)
            ):
                years.add(int(match.group()))
    if len(years) != 1:
        return None
    year = next(iter(years))
    identity = provenance.get("identity")
    claimed = identity.get("year") if isinstance(identity, dict) else None
    if claimed is not None and (
        not re.fullmatch(r"(?:19|20)\d{2}", str(claimed)) or int(claimed) != year
    ):
        return None
    return year


def build_fact_graph(
    record: dict, *, strict: bool = False, allowed_sha256=None
) -> FactGraph:
    data = record["final_data"]
    basic = data.get("基本信息", {})
    doc = record.get("document") or {"sha256": record.get("document_sha256")}
    digest = doc.get("sha256")
    if strict and (
        record.get("status") != "success"
        or not allowed_sha256
        or digest not in allowed_sha256
    ):
        raise ValueError("successful explicitly allowed document required")
    if not digest:
        raise ValueError("document SHA256 identity required")
    bridge_raw = str(basic.get("桥梁名称") or "")
    inspection_time = (
        basic.get("检测日期") or basic.get("检测时间") or basic.get("本次检测日期")
    )
    document = {
        "id": _id("document", digest),
        **doc,
        "report_time": basic.get("报告日期"),
        "inspection_time": inspection_time,
        "inspection_year": _year(inspection_time),
        "report_year": _year(basic.get("报告日期")),
    }
    provenance, inventory = _provenance(record), _source_inventory(record)
    calls = {c["call_id"]: c for c in record.get("model_calls", []) if c.get("call_id")}
    tasks = {t.get("id", t.get("task_id")): t for t in record.get("tasks", [])}
    entities, observations, evidence, relations, derivations, defects = (
        {},
        [],
        {},
        [],
        [],
        {},
    )
    diagnostics = {
        "coarse_evidence": 0,
        "unbound_evidence": 0,
        "missing_identity": 0,
        "uncertain_relations": 0,
        "normalization_version": NORMALIZATION_VERSION,
    }

    def evidence_for(p: dict, section: str, number: str) -> str:
        ids = list(
            dict.fromkeys(
                p.get("evidence_ids")
                or ([p["source_id"]] if p.get("source_id") else [])
            )
        )
        call_ids = [p["call_id"]] if p.get("call_id") else list(p.get("call_ids", []))
        visible = set()
        visible_intervals = {}
        for task_id in p.get("task_ids", []):
            task = tasks.get(task_id, {})
            if task.get("accepted_call_id"):
                call_ids.append(task["accepted_call_id"])
                visible.update(task.get("evidence_ids", []))
                visible.update(task.get("context_evidence_ids", []))
                visible.update(task.get("accepted_extra_ids", []))
                for span in task.get("spans", []):
                    visible_intervals.setdefault(span.get("coverage_id"), []).append(
                        (span.get("start"), span.get("end"))
                    )
                # Context IDs can refer only to a rendered prefix. Without an
                # explicit interval they cannot authorize reading the whole unit.
                for source_id in task.get("accepted_extra_ids", []):
                    visible_intervals.setdefault(source_id, []).append(
                        (0, len(inventory.get(source_id, {}).get("text", "")))
                    )
        call_ids = list(dict.fromkeys(call_ids))
        valid = bool(call_ids)
        visual_sources = {source_id: inventory[source_id] for source_id in ids
                          if source_id in inventory and str(inventory[source_id].get("kind", "")).startswith("visual-")}
        upstream_call_ids = []
        for source_id, source in visual_sources.items():
            location = source.get("location", {})
            upstream = calls.get(location.get("call_id"), {})
            if strict and (
                source.get("kind") not in ("visual-transcription", "visual-observation")
                or location.get("evidence_domain") != source.get("kind")
                or upstream.get("role") != "visual-evidence"
                or upstream.get("status") != "success"
                or upstream.get("mock") or upstream.get("simulated")
                or location.get("occurrence_id") not in upstream.get("source_ranges", [])
                or location.get("call_id") not in record.get("visual_contributing_call_ids", [])
            ):
                raise ValueError("visual fact requires typed source and successful upstream occurrence evidence")
            if location.get("call_id") not in upstream_call_ids:
                upstream_call_ids.append(location.get("call_id"))
            if strict and source.get("kind") == "visual-observation" and (
                section != "病害明细" or set(p.get("quotes", {})) - {"病害类型", "病害描述"}
                or p.get("current_evidence")
                or p.get("history_evidence") or p.get("relationship_evidence")
            ):
                raise ValueError("photo observation cannot establish literal metadata or temporal evidence")
        for call_id in call_ids:
            call = calls.get(call_id, {})
            valid = (
                valid
                and call.get("status") == "success"
                and not call.get("simulated")
                and not call.get("mock")
            )
            valid = valid and bool(
                call.get("backend_request_id")
                or call.get("response_id")
                or call.get("output_sha256")
            )
            for source in call.get("source_ranges", []):
                if isinstance(source, dict):
                    source_id, start, end = (
                        source.get("source_id"),
                        source.get("start"),
                        source.get("end"),
                    )
                else:
                    parts = str(source).rsplit(":", 2)
                    source_id = parts[0]
                    start, end = (
                        (int(parts[1]), int(parts[2]))
                        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit()
                        else (None, None)
                    )
                visible.add(source_id)
                if type(start) is int and type(end) is int:
                    visible_intervals.setdefault(source_id, []).append((start, end))
        if not ids:
            # Whole mode does not invent one evidence source per extracted row.
            call_ids = [
                c["call_id"]
                for c in calls.values()
                if c.get("status") == "success"
                and not c.get("simulated")
                and not c.get("mock")
            ]
            valid = bool(call_ids)
        if strict and (
            not valid
            or (
                ids
                and (not set(ids).issubset(visible) or not set(ids).issubset(inventory))
            )
        ):
            raise ValueError(
                "evidence source must be visible in its actual successful call"
            )
        for region in p.get("source_ranges", []):
            source = inventory.get(region.get("source_id"), {})
            start, end = region.get("start"), region.get("end")
            if strict and (
                not source
                or type(start) is not int
                or type(end) is not int
                or not 0 <= start <= end <= len(source.get("text", ""))
            ):
                raise ValueError("evidence range outside source")
            if strict and not any(
                type(a) is int and type(b) is int and a <= start <= end <= b
                for a, b in visible_intervals.get(region.get("source_id"), [])
            ):
                raise ValueError("evidence range not visible in actual call")
        for quote in p.get("quotes", {}).values():
            if strict and (
                not isinstance(quote, str)
                or not quote
                or not any(
                    quote in inventory.get(s, {}).get("text", "")[a:b]
                    for s in ids
                    for a, b in visible_intervals.get(s, [])
                    if type(a) is int and type(b) is int
                )
            ):
                raise ValueError("evidence quote absent from cited source")
        actual_ranges = []
        for source_id in ids:
            text = inventory.get(source_id, {}).get("text", "")
            declared = [
                r for r in p.get("source_ranges", []) if r.get("source_id") == source_id
            ]
            for a, b in visible_intervals.get(source_id, []):
                if (
                    type(a) is not int
                    or type(b) is not int
                    or not 0 <= a <= b <= len(text)
                ):
                    continue
                for claim in declared or [{"start": a, "end": b}]:
                    x, y = claim.get("start"), claim.get("end")
                    if type(x) is not int or type(y) is not int:
                        continue
                    start, end = max(a, x), min(b, y)
                    if start < end:
                        item = {"source_id": source_id, "start": start, "end": end}
                        if item not in actual_ranges:
                            actual_ranges.append(item)
        locations = []
        for source_id in ids:
            source = inventory.get(source_id, {})
            location = source.get("location") or {
                "source_path": source.get("source_path")
            }
            if source and any(v is not None for v in location.values()):
                locations.append({"source_id": source_id, **location})
        precision = "source" if ids and locations else "document"
        identifier = _id(
            "evidence",
            digest,
            ids,
            p.get("source_key"),
            p.get("instance_id"),
            section,
            number,
        )
        evidence[identifier] = {
            "id": identifier,
            "document_id": document["id"],
            "source_ids": ids,
            "precision": precision,
            "locations": locations,
            "call_ids": call_ids,
            "binding": "bound"
            if valid and (not ids or set(ids).issubset(visible))
            else "unverified",
            "ranges": p.get("source_ranges", []),
            "visible_source_ranges": actual_ranges,
            "quotes": p.get("quotes", {}),
            "instance_id": p.get("instance_id"),
            "source_key": p.get("source_key"),
            "current_evidence": p.get("current_evidence", []),
            "history_evidence": p.get("history_evidence", []),
            "relationship_evidence": p.get("relationship_evidence", []),
            "temporal_verification": p.get("temporal_verification"),
        }
        if visual_sources:
            domains = {inventory[source_id].get("kind") if source_id in visual_sources else "native-text"
                       for source_id in ids}
            evidence[identifier].update(
                evidence_domains={source_id: inventory[source_id].get("kind")
                                  if source_id in visual_sources else "native-text" for source_id in ids},
                upstream_call_ids=upstream_call_ids,
                # An observation quote cites the derived description, never
                # literal OCR text. Mixed evidence retains its per-source map.
                quotation_basis=(next(iter(domains)) if len(domains) == 1 else "mixed-source-domains")
                if p.get("quotes") else None,
            )
        diagnostics["coarse_evidence"] += precision == "document"
        diagnostics["unbound_evidence"] += evidence[identifier]["binding"] != "bound"
        return identifier

    def observation(
        section: str,
        number: str,
        raw: Any,
        p: dict,
        *,
        field: str | None = None,
        scope: dict | None = None,
    ) -> None:
        location = (
            str(raw.get("病害部位", raw.get("部位", "")))
            if isinstance(raw, dict)
            else ""
        )
        if scope:
            location = " ".join(
                str(scope.get(k, ""))
                for k in ("location", "side", "span", "component", "position")
            ).strip()
        identity = normalize_identity(bridge_raw, location)
        if scope:
            for key in identity:
                if scope.get(key) is not None and key != "span":
                    identity[key] = scope[key]
            if "span" in scope and re.fullmatch(r"\d+", str(scope["span"])):
                identity["span"] = str(scope["span"])
        # Unknown bridge identifiers are document scoped, never global aliases.
        identity_scope = digest if not identity.get("bridge") else None
        bridge_identity = normalize_identity(identity.get("bridge", ""), "")
        bridge_id = _id(
            "entity", NORMALIZATION_VERSION, bridge_identity, identity_scope
        )
        entities.setdefault(
            bridge_id,
            {
                "id": bridge_id,
                "kind": "bridge",
                "identity": bridge_identity,
                "parent_id": None,
                "raw_bridge": identity.get("bridge"),
                "raw_location": "",
                "rule_version": NORMALIZATION_VERSION,
            },
        )
        entity_id = _id("entity", NORMALIZATION_VERSION, identity, identity_scope)
        entities[entity_id] = {
            "id": entity_id,
            "kind": "component"
            if identity.get("component")
            else "scope"
            if location
            else "bridge",
            "parent_id": bridge_id if entity_id != bridge_id else None,
            "identity": identity,
            "raw_bridge": bridge_raw,
            "raw_location": location,
            "rule_version": NORMALIZATION_VERSION,
        }
        ev = evidence_for(p, section, number)
        obs_id = _id(
            "observation",
            digest,
            section,
            number,
            p.get("source_key"),
            p.get("instance_id"),
            scope,
        )
        measurements = (
            parse_measurements(str(raw.get("病害描述", "")))
            if isinstance(raw, dict)
            else ()
        )
        explicit_defect_id = (
            str(raw.get("病害编号", "")).strip() if isinstance(raw, dict) else ""
        )
        visible_excerpts = [
            inventory.get(part["source_id"], {}).get("text", "")[
                part["start"] : part["end"]
            ]
            for part in evidence[ev]["visible_source_ranges"]
        ]
        if section in ("病害明细", "维修建议"):
            year = (
                _source_inspection_year(visible_excerpts, p)
                if evidence[ev]["binding"] == "bound"
                else None
            )
            time_basis = "inspection" if year else "unknown"
        else:
            year = _year((scope or {}).get("year")) or document["inspection_year"]
            prior_score = bool(field and ("上一次" in field or "历史" in field))
            if prior_score:
                year = _year((scope or {}).get("year"))
            time_basis = (
                "inspection"
                if year
                else "prior-cycle-unknown"
                if prior_score
                else "report-only"
            )
        anchored = (
            identity.get("bridge")
            and identity.get("side")
            and identity.get("component")
            and any(
                identity.get(k) is not None
                for k in ("span", "pier", "abutment", "component_number")
            )
        )
        identifier_pattern = (
            (
                r"(?<![A-Za-z0-9_.:/#-])"
                + re.escape(explicit_defect_id)
                + r"(?![A-Za-z0-9_.:/#-])"
            )
            if explicit_defect_id
            else None
        )
        source_anchored = False
        for excerpt in visible_excerpts:
            if not identifier_pattern or not re.search(identifier_pattern, excerpt):
                continue
            source_identity = normalize_identity(bridge_raw, excerpt)
            if all(
                identity.get(k) is None or source_identity.get(k) == identity[k]
                for k in (
                    "side",
                    "span",
                    "pier",
                    "abutment",
                    "component",
                    "component_number",
                    "position",
                )
            ):
                source_anchored = True
                break
        defect_id = None
        if (
            section == "病害明细"
            and explicit_defect_id
            and anchored
            and source_anchored
            and evidence[ev]["binding"] == "bound"
        ):
            defect_id = _id("defect", identity, explicit_defect_id)
            defects[defect_id] = {
                "id": defect_id,
                "entity_id": entity_id,
                "reported_identifier": explicit_defect_id,
                "identity": identity,
                "evidence_ids": [ev],
                "rule_version": NORMALIZATION_VERSION,
            }

        obs = {
            "id": obs_id,
            "entity_id": entity_id,
            "document_id": document["id"],
            "kind": "score"
            if field
            else ("defect" if section == "病害明细" else "recommendation"),
            "section": section,
            "number": number,
            "attribute": field,
            "rating_system": (scope or {}).get("rating_system"),
            "inspection_year": year,
            "report_year": document["report_year"],
            "time_basis": time_basis,
            "defect_identity_id": defect_id,
            "raw": raw,
            "value": raw if field else None,
            "evidence_ids": [ev],
            "measurements": [asdict(m) for m in measurements],
        }
        observations.append(obs)
        if measurements:
            derivations.append(
                {
                    "id": _id("derivation", obs_id),
                    "type": "unit-normalization",
                    "input_observation_ids": [obs_id],
                    "evidence_ids": [ev],
                    "rule_version": NORMALIZATION_VERSION,
                    "formula": "canonical_value = reported_value * unit_factor",
                    "outputs": [asdict(m) for m in measurements],
                }
            )
        if section == "病害明细":
            # Existing/new is an extracted claim, not a proved cross-document edge.
            temporal = p.get("temporal") or "unknown"
            relations.append(
                {
                    "id": _id("relation", obs_id, "history"),
                    "left": obs_id,
                    "right": None,
                    "type": "same-defect",
                    "status": "uncertain",
                    "reported_temporal": temporal,
                    "evidence_ids": [ev],
                    "reason": "no independently bound prior observation",
                }
            )
            diagnostics["uncertain_relations"] += 1
            diagnostics["missing_identity"] += any(
                identity[k] is None for k in ("side", "component")
            )

    for section in ("病害明细", "维修建议"):
        for index, row in enumerate(data.get(section, []), 1):
            observation(
                section, str(index), row, provenance.get((section, str(index)), {})
            )
    scoped_fields = set()
    original_observations = {}
    for index, item in enumerate(record.get("basic_observations", [])):
        field = item.get("field", "")
        if "评分" in field or "等级" in field:
            scoped_fields.add(field)
            p = (
                dict(item.get("evidence") or {})
                if isinstance(item.get("evidence"), dict)
                else {}
            )
            p.setdefault("call_id", item.get("call_id"))
            if p.get("source_id") and "start" in p:
                p["source_ranges"] = [dict(item["evidence"])]
            if p.get("quote"):
                p["quotes"] = {field: p["quote"]}
            if isinstance(item.get("evidence"), list):
                p["evidence_ids"] = item["evidence"]
            observation(
                "基本信息",
                f"{field}:{index}",
                item.get("value"),
                p,
                field=field,
                scope=item.get("scope"),
            )
            original_observations[sha256(_json(item).encode()).hexdigest()] = (
                observations[-1]["id"]
            )
    for index, item in enumerate(record.get("repair_evidence", [])):
        field = item.get("field", "")
        if not ("评分" in field or "等级" in field):
            continue
        refs = item.get("supersedes", [])
        if strict and (not refs or not set(refs).issubset(original_observations)):
            raise ValueError("repair must bind superseded observations")
        p = {**item.get("evidence", {}), "call_id": item.get("call_id")}
        if p.get("source_id") and "start" in p:
            p["source_ranges"] = [dict(item["evidence"])]
        observation(
            "基本信息",
            f"{field}:repair:{index}",
            item.get("value"),
            p,
            field=field,
            scope=item.get("scope"),
        )
        resolved = observations[-1]
        resolved["resolution"] = True
        derivations.append(
            {
                "id": _id("derivation", resolved["id"], "resolution"),
                "type": "conflict-resolution",
                "rule_version": "bounded-evidence-repair-v1",
                "input_observation_ids": [
                    original_observations[r] for r in refs if r in original_observations
                ],
                "output_observation_ids": [resolved["id"]],
                "evidence_ids": resolved["evidence_ids"],
                "call_id": item.get("call_id"),
            }
        )
        scoped_fields.add(field)
    for field, value in sorted(basic.items()):
        if (
            ("评分" in field or "等级" in field)
            and value is not None
            and field not in scoped_fields
        ):
            observation(
                "基本信息",
                field,
                value,
                provenance.get(("基本信息", field), {}),
                field=field,
            )
    graph = {
        "schema_version": SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "document": document,
        "entities": list(entities.values()),
        "observations": observations,
        "evidence": list(evidence.values()),
        "defect_identities": list(defects.values()),
        "relations": relations,
        "derivations": derivations,
        "diagnostics": diagnostics,
        "three_table_sha256": sha256(_json(data).encode()).hexdigest(),
    }
    return FactGraph(_json(graph))


def validate_fact_graph(record: dict) -> list[str]:
    """Rebuild derived facts to reject changes to source, tables, or provenance."""
    if "facts" not in record:
        return []
    try:
        expected = build_fact_graph(record).to_dict()
        if record["facts"] != expected:
            return [
                "facts: derived graph differs from current source/call/table evidence"
            ]
        return []
    except (TypeError, ValueError, KeyError) as exc:
        return ["facts: invalid fact graph inputs: " + str(exc)]
