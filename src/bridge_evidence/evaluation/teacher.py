from __future__ import annotations

import re
from typing import Any


def _lookup(state: dict[str, Any], dotted_field: str) -> Any:
    """Read a competition field from a serialized DocumentState."""

    current: Any = state
    for part in dotted_field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, dict) and "value" in current:
        return current["value"]
    return current


def semantic_signature(value: Any) -> dict[str, Any]:
    """Normalize harmless rendering differences without hiding semantics.

    The scorer should regard ``91.30`` and ``91.3`` as the same value, while
    preserving number order, left/right order and A--E grade order.  Plain
    text fields remain exact after punctuation/whitespace normalization.
    """

    if value is None:
        return {"kind": "missing"}
    rendered = str(value).strip()
    compact = re.sub(r"[\s（）()：:/／]", "", rendered)
    numbers = [round(float(item), 6) for item in re.findall(r"\d+(?:\.\d+)?", compact)]
    grades = [item.upper() for item in re.findall(r"([ABCDE])(?=级|类|$|左幅|右幅)", compact, re.I)]
    sides = [match.group(0) for match in re.finditer(r"左幅|右幅", compact)]
    if numbers or grades or sides:
        return {
            "kind": "structured",
            "numbers": numbers,
            "grades": grades,
            "sides": sides,
        }
    normalized_text = re.sub(r"[\s，,。.;；：:]", "", rendered).upper()
    return {"kind": "text", "value": normalized_text}


def semantic_equal(left: Any, right: Any) -> bool:
    """Compare facts while collapsing only redundant left/right syntax."""

    left_signature = semantic_signature(left)
    right_signature = semantic_signature(right)
    if left_signature == right_signature:
        return True
    if left_signature.get("kind") != "structured" or right_signature.get("kind") != "structured":
        return False

    def collapse(signature: dict[str, Any]) -> dict[str, Any]:
        collapsed = dict(signature)
        sides = collapsed.get("sides", [])
        numbers = collapsed.get("numbers", [])
        grades = collapsed.get("grades", [])
        if sides == ["左幅", "右幅"]:
            if len(numbers) == 2 and numbers[0] == numbers[1]:
                collapsed["numbers"] = [numbers[0]]
            if len(grades) == 2 and grades[0] == grades[1]:
                collapsed["grades"] = [grades[0]]
            if len(collapsed.get("numbers", [])) <= 1 and len(collapsed.get("grades", [])) <= 1:
                collapsed["sides"] = []
        return collapsed

    return collapse(left_signature) == collapse(right_signature)


def compare_teacher_target(state: dict[str, Any], teacher: dict[str, Any]) -> dict[str, Any]:
    """Return field-level deltas against a development-only teacher trace."""

    rows: list[dict[str, Any]] = []
    objective_weight = 0.0
    matched_weight = 0.0
    for field, target in teacher.get("targets", {}).items():
        expected = target.get("target_value")
        actual = _lookup(state, target.get("student_field", field))
        included = bool(target.get("include_in_objective", True))
        weight = float(target.get("weight", 1.0)) if included else 0.0
        matched = semantic_equal(actual, expected)
        objective_weight += weight
        if matched:
            matched_weight += weight
        rows.append({
            "field": field,
            "expected": expected,
            "actual": actual,
            "matched": matched,
            "included_in_objective": included,
            "weight": weight,
            "status": target.get("status", "resolved"),
            "reason": target.get("reason", ""),
        })
    return {
        "teacher_id": teacher.get("teacher_id"),
        "matched_fields": sum(1 for row in rows if row["matched"]),
        "total_fields": len(rows),
        "objective_score": matched_weight / objective_weight if objective_weight else 0.0,
        "differences": [row for row in rows if not row["matched"]],
        "fields": rows,
    }
