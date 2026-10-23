"""Independent evidence diagnostics; never substitutes coverage for semantic recall.

Only the offline evaluator supplies optional independent annotations. The runtime
and strategy controller do not receive annotation values or score differences.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation

PROTOCOL = "semantic-evidence-diagnostics-v1"
_UNITS = {
    "mm": ("length", "1"),
    "cm": ("length", "10"),
    "m": ("length", "1000"),
    "毫米": ("length", "1"),
    "厘米": ("length", "10"),
    "米": ("length", "1000"),
    "mm2": ("area", "1"),
    "cm2": ("area", "100"),
    "m2": ("area", "1000000"),
}


def compare_quantity(left, right):
    """Compare only finite quantities with explicit matching statistic semantics."""
    try:
        a, b = _UNITS[left["unit"]], _UNITS[right["unit"]]
        if (
            a[0] != b[0]
            or not left.get("statistic")
            or left["statistic"] != right.get("statistic")
        ):
            return {"comparable": False, "equal": None}
        x, y = Decimal(str(left["value"])), Decimal(str(right["value"]))
        if not x.is_finite() or not y.is_finite():
            return {"comparable": False, "equal": None}
        x, y = x * Decimal(a[1]), y * Decimal(b[1])
        return {
            "comparable": True,
            "equal": x == y,
            "difference_base_units": str(x - y),
        }
    except (KeyError, TypeError, InvalidOperation):
        return {"comparable": False, "equal": None}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _fact_key(fact):
    # A real duplicate in another source/instance remains a distinct observation.
    return (
        fact.get("source_id"),
        str(fact.get("instance_id", "")),
        fact.get("field"),
        str(fact.get("value")),
    )


def _relation_key(relation):
    return (
        relation.get("left"),
        relation.get("right"),
        relation.get("type", "same_defect"),
    )


def _extent_covered(text, slices):
    cursor = 0
    for item in sorted(slices, key=lambda s: s.get("start", -1)):
        start, end = item.get("start"), item.get("end")
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end > len(text)
            or end <= start
            or start > cursor
            or item.get("text") != text[start:end]
        ):
            return False
        cursor = max(cursor, end)
    return cursor == len(text)


def diagnose_record(record, *, annotations=None):
    """Return additive diagnostics with unknown (null) recall absent explicit gold.

    Annotations v1 may contain parsed_source_ids, source-occurrence facts,
    relations and quantities. They are a distinct evaluation dataset, not
    generated from the three-table gold or the model's coverage declaration.
    """
    if annotations is not None and annotations.get("schema_version") != 1:
        raise ValueError("independent diagnostic annotations v1 required")
    annotations = annotations or {}
    regions = {r["source_id"]: r for r in record.get("snapshot", {}).get("regions", [])}
    tasks = {t["task_id"]: t for t in record.get("chunk_plan", {}).get("tasks", [])}
    fragments = record.get("fragments", [])
    completed = set()
    for fragment in fragments:
        task = tasks.get(fragment.get("task_id"))
        if task is not None:
            owned = [
                f"{s.get('source_id')}:{s.get('start')}:{s.get('end')}"
                for s in task.get("owned", [])
            ]
            if owned and sorted(fragment.get("covered", [])) == sorted(owned):
                completed.add(fragment["task_id"])

    slices = [s for t in tasks.values() for s in t.get("owned", [])]
    assigned = sum(
        _extent_covered(
            r.get("text", ""), [s for s in slices if s.get("source_id") == key]
        )
        for key, r in regions.items()
    )
    refs = invalid_refs = invisible = invalid_quotes = quotes = 0
    facts = list(record.get("diagnostic_facts", []))
    for fragment in fragments:
        task = tasks.get(fragment.get("task_id"), {})
        visible = task.get("owned", []) + task.get("context", [])
        visible_ids = {s.get("source_id") for s in visible}
        for source in fragment.get("basic_sources", {}).values():
            refs += 1
            invalid_refs += source not in regions
            invisible += source not in visible_ids
        for row in fragment.get("diseases", []) + fragment.get("advice", []):
            for source in [row.get("source_id"), *row.get("history_sources", [])]:
                refs += 1
                invalid_refs += source not in regions
                invisible += source not in visible_ids
            selected = [
                s
                for s in visible
                if s.get("source_id") == row.get("source_id")
                and row.get("source_key")
                == f"{s.get('source_id')}:{s.get('start')}:{s.get('end')}"
            ]
            for quote in row.get("quotes", {}).values():
                quotes += 1
                invalid_quotes += (
                    not isinstance(quote, str)
                    or not quote
                    or not any(quote in s.get("text", "") for s in selected)
                )
            facts.extend(
                {
                    "source_id": row.get("source_id"),
                    "instance_id": row.get("instance_id", ""),
                    "field": field,
                    "value": value,
                }
                for field, value in row.get("fields", {}).items()
            )
    graph = record.get("facts", {})
    if not fragments and not facts:
        graph_evidence = {e["id"]: e for e in graph.get("evidence", [])}
        for observation in graph.get("observations", []):
            bound = [
                graph_evidence[key]
                for key in observation.get("evidence_ids", [])
                if key in graph_evidence
            ]
            source_ids = {source for e in bound for source in e.get("source_ids", [])}
            # Coarse/multiple-source observations cannot acquire invented precise source IDs.
            source = next(iter(source_ids)) if len(source_ids) == 1 else None
            instance = bound[0].get("instance_id") if len(bound) == 1 else None
            raw = observation.get("raw")
            fields = (
                raw
                if isinstance(raw, dict)
                else {observation.get("attribute"): observation.get("value")}
            )
            facts.extend(
                {
                    "source_id": source,
                    "instance_id": instance,
                    "field": field,
                    "value": value,
                }
                for field, value in fields.items()
            )
    predicted = Counter(_fact_key(f) for f in facts)
    expected_facts = annotations.get("facts")
    expected = (
        Counter(_fact_key(f) for f in expected_facts)
        if expected_facts is not None
        else None
    )
    true_facts = sum((predicted & expected).values()) if expected is not None else None
    parsed_gold = annotations.get("parsed_source_ids")
    relations = record.get("relations", record.get("facts", {}).get("relations", []))
    relation_gold = annotations.get("relations")
    truth = (
        {_relation_key(r): r["status"] for r in relation_gold}
        if relation_gold is not None
        else None
    )
    predictions = [r for r in relations if r.get("status") == "confirmed"]
    confirmed = list({_relation_key(r): r for r in predictions}.values())
    true_relations = (
        sum(truth.get(_relation_key(r)) == "confirmed" for r in confirmed)
        if truth is not None
        else None
    )
    measurements = {q["id"]: q for q in record.get("diagnostic_quantities", [])}
    for observation in graph.get("observations", []):
        for index, quantity in enumerate(observation.get("measurements", [])):
            measurements[f"{observation['id']}:{index}"] = quantity
    expected_quantities = annotations.get("quantities", [])
    quantities = [
        compare_quantity(measurements[q["id"]], q)
        for q in expected_quantities
        if q["id"] in measurements
    ]
    return {
        "protocol": PROTOCOL,
        "affects_official_score": False,
        "parsed_coverage": {
            "parsed": len(regions),
            "expected": len(parsed_gold) if parsed_gold is not None else None,
            "recall": _ratio(
                len(set(regions) & set(parsed_gold)), len(set(parsed_gold))
            )
            if parsed_gold is not None
            else None,
        },
        "task_coverage": {
            "planned_tasks": len(tasks),
            "completed_tasks": len(completed),
            "fully_assigned_sources": assigned,
            "parsed_sources": len(regions),
            "claimed_complete": record.get("coverage", {}).get("complete") is True,
        },
        "fact_coverage": {
            "predicted": sum(predicted.values()),
            "expected": sum(expected.values()) if expected is not None else None,
            "true_positive": true_facts,
            "recall": _ratio(true_facts, sum(expected.values()))
            if expected is not None
            else None,
            "precision": _ratio(true_facts, sum(predicted.values()))
            if expected is not None
            else None,
        },
        "evidence": {
            "references": refs,
            "missing_references": invalid_refs,
            "invisible_references": invisible,
            "quotes": quotes,
            "invalid_quotes": invalid_quotes,
            "semantic_support": "not-established-by-substring",
        },
        "relations": {
            "predicted_confirmed": len(confirmed),
            "duplicate_predictions": len(predictions) - len(confirmed),
            "unknown": sum(
                r.get("status") in ("unknown", "uncertain") for r in relations
            ),
            "unjudged": sum(_relation_key(r) not in truth for r in confirmed)
            if truth is not None
            else len(confirmed),
            "false_merges": sum(
                truth.get(_relation_key(r)) == "rejected" for r in confirmed
            )
            if truth is not None
            else None,
            "recall": _ratio(
                true_relations, sum(v == "confirmed" for v in truth.values())
            )
            if truth is not None
            else None,
        },
        "quantities": {
            "expected": len(expected_quantities),
            "missing": len(expected_quantities) - len(quantities),
            "compared": len(quantities),
            "incomparable": sum(not q["comparable"] for q in quantities),
            "equal": sum(q["equal"] is True for q in quantities),
        },
    }
