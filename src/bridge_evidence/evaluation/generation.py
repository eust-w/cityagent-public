"""Deterministic internal generation metrics, independent of SDKs and models.

This is a normalized exact-match proxy, NOT the competition's scoring function.
Gold comes from the frozen, audited SFT labels. Pass ``prediction=None`` for
generation/JSON/schema failures or truncated responses; they retain all gold FN.
This module never recovers partial JSON or removes duplicate rows.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
import math
import re
from typing import Any, Iterable
import unicodedata


METRIC_VERSION = "normalized-exact-multiset-prf-v1"
INTERPRETATION = "internal_proxy_not_official_competition_score"
SECTION_NAMES = {"basic": "基本信息", "advice": "维修建议", "defects": "病害明细"}
NORMALIZATION = {
    "text": "NFKC and collapse whitespace; retain words, numbers, units, dates and punctuation",
    "empty": "None and blank strings are absent; literal 无 is an explicit fact; numeric 0 is retained",
    "score_values": "Exact decimal spelling normalization only for entire numeric 评分 fields; no tolerance or digit extraction",
    "rows": "Unordered full semantic-row multiset; ignore 序号 only; retain unknown nonempty fields and duplicates",
    "empty_sections": "When both counts are zero P/R/F1 are null, excluded from macro averages",
    "overall": "Micro counts mix basic-field facts and complete advice/defect rows; section metrics are primary",
}
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)\Z")


def _text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _decimal_text(value: str) -> str:
    number = Decimal(value)
    if number == 0:
        return "0"
    text = format(number, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _value(field: str, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        result = _text(value)
        if not result:
            return None
        if field.endswith("评分") and _DECIMAL.fullmatch(result):
            return _decimal_text(result)
        return result
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return _decimal_text(str(value))
    raise ValueError(f"{field}: metric values must be finite string/number/null scalars")


def _facts(obj: Any, *, row: bool = False) -> tuple[tuple[str, str], ...]:
    if not isinstance(obj, dict):
        raise ValueError("Metric input sections/rows must be objects")
    facts, seen = [], set()
    for field, raw in obj.items():
        if not isinstance(field, str) or not _text(field):
            raise ValueError("Metric field names must be nonempty strings")
        name = _text(field)
        if name in seen:
            raise ValueError("Field names collide after normalization")
        seen.add(name)
        if row and name == "序号":
            continue
        value = _value(name, raw)
        if value is not None:
            facts.append((name, value))
    return tuple(sorted(facts))


def _section_items(report: Any) -> dict[str, Counter]:
    if not isinstance(report, dict):
        raise ValueError("A parsed report must be an object; use None for a failed prediction")
    if not set(SECTION_NAMES.values()).issubset(report):
        raise ValueError("A parsed report must contain all three sections")
    result = {"basic": Counter(_facts(report["基本信息"]))}
    for section in ("advice", "defects"):
        rows = report[SECTION_NAMES[section]]
        if not isinstance(rows, list):
            raise ValueError(f"{SECTION_NAMES[section]} must be an array")
        # Counter preserves multiplicity. Even an empty row occupies one row;
        # upstream schema validation decides whether such output is acceptable.
        result[section] = Counter(_facts(item, row=True) for item in rows)
    return result


def _counts(tp: int, predicted: int, gold: int) -> dict:
    if not predicted and not gold:
        precision = recall = f1 = None
    else:
        precision = tp / predicted if predicted else 0.0
        recall = tp / gold if gold else 0.0
        f1 = 2 * tp / (predicted + gold)
    return {"tp": tp, "fp": predicted - tp, "fn": gold - tp,
            "predicted_count": predicted, "gold_count": gold,
            "precision": precision, "recall": recall, "f1": f1}


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def score_report(gold: dict, prediction: dict | None) -> dict:
    """Count exact basic-field facts and complete semantic rows with multiplicity.

    Wrong nonempty field values contribute one FP and one FN. For row metrics a
    changed semantic field makes the complete row unmatched (one FP + one FN).
    Row order/序号 do not affect scores. Extra nonempty fields are retained, so
    adding them cannot create an exact match. ``None`` predicts zero facts/rows.
    Caller must apply the same completion/schema gate to every evaluated model.
    """
    expected = _section_items(gold)
    predicted = _section_items(prediction) if prediction is not None else {key: Counter() for key in SECTION_NAMES}
    sections = {}
    for section in SECTION_NAMES:
        correct = sum((expected[section] & predicted[section]).values())
        sections[section] = _counts(correct, sum(predicted[section].values()), sum(expected[section].values()))
    overall = _counts(sum(s["tp"] for s in sections.values()),
                      sum(s["predicted_count"] for s in sections.values()),
                      sum(s["gold_count"] for s in sections.values()))
    return {"metric_version": METRIC_VERSION, "interpretation": INTERPRETATION,
            "normalization": dict(NORMALIZATION), "prediction_valid": prediction is not None,
            "sections": sections, "overall": overall,
            "macro_section_f1": _mean(section["f1"] for section in sections.values())}


def aggregate_reports(items: Iterable[dict]) -> dict:
    """Aggregate ``score_report`` results without dropping failed documents.

    Section micro P/R/F1 uses pooled TP/FP/FN. ``macro_report_f1`` on a section
    averages that section's defined report F1s, including every failure with gold
    support. Neither averaging method gives credit to an empty/empty section.
    """
    scores = list(items)
    for score in scores:
        if (not isinstance(score, dict) or score.get("metric_version") != METRIC_VERSION
                or score.get("interpretation") != INTERPRETATION
                or type(score.get("prediction_valid")) is not bool):
            raise ValueError("aggregate_reports requires score_report results from this metric version")
        sections = score.get("sections")
        if not isinstance(sections, dict) or set(sections) != set(SECTION_NAMES):
            raise ValueError("Missing metric sections")
        for section in sections.values():
            keys = ("tp", "fp", "fn", "predicted_count", "gold_count")
            if (not isinstance(section, dict) or any(type(section.get(k)) is not int or section[k] < 0 for k in keys)
                    or section["tp"] + section["fp"] != section["predicted_count"]
                    or section["tp"] + section["fn"] != section["gold_count"]):
                raise ValueError("Inconsistent metric counts")
    sections = {}
    for name in SECTION_NAMES:
        pieces = [score["sections"][name] for score in scores]
        result = _counts(sum(s["tp"] for s in pieces), sum(s["predicted_count"] for s in pieces),
                         sum(s["gold_count"] for s in pieces))
        # Recompute from counts, rather than trust stale rounded report rates.
        report_f1s = [_counts(s["tp"], s["predicted_count"], s["gold_count"])["f1"] for s in pieces]
        result["macro_report_f1"] = _mean(report_f1s)
        result["macro_report_support"] = sum(value is not None for value in report_f1s)
        sections[name] = result
    overall = _counts(sum(s["tp"] for s in sections.values()),
                      sum(s["predicted_count"] for s in sections.values()),
                      sum(s["gold_count"] for s in sections.values()))
    valid = sum(score["prediction_valid"] for score in scores)
    return {"metric_version": METRIC_VERSION, "interpretation": INTERPRETATION,
            "normalization": dict(NORMALIZATION), "reports_total": len(scores),
            "reports_valid": valid, "reports_failed": len(scores) - valid,
            "valid_report_rate": valid / len(scores) if scores else None,
            "sections": sections, "overall": overall,
            "macro_section_f1": _mean(section["f1"] for section in sections.values())}
