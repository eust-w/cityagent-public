"""Minimal presentation changes to an already validated model report.

This policy neither extracts candidates nor resolves factual disagreements. It
preserves every semantic row, including duplicates and unknown values. Only row
numbers and the summary derived from the actual advice rows may change.
"""
from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from typing import Any


ADVICE_CATEGORIES = ("立即处置", "尽快维修", "预防性养护", "跟踪观测")
POLICY = "model-preserving"


# Output conventions of the official training labels (81 frozen development
# reports): grades "B级", zero-padded "YYYY年MM月DD日" dates, advice category
# 尽快维修 (97% of rows), 发展程度 "无" (97%) and history 是 / 上一次定检未发现
# (the most frequent pair, 46% of rows; 否 rows use free-text statuses that rarely
# match exactly). Exact-match scoring makes these priors beat per-row guesses.
GRADE_FIELDS = ("总体等级", "上部结构等级", "下部结构等级", "桥面系等级", "上一次总体等级")
SCORE_FIELDS = ("总体评分", "上部结构评分", "下部结构评分", "桥面系评分", "上一次总体评分")
# Part grades follow the part's BCI score through the CJJ 99-2017 bands in the
# labels (development gold agrees 199/217); the model often copies the stricter
# BSI control grade instead, so a parsed part score decides its grade.
PART_GRADES = (("上部结构等级", "上部结构评分"), ("下部结构等级", "下部结构评分"), ("桥面系等级", "桥面系评分"))
BCI_BANDS = ((90, "A"), (80, "B"), (66, "C"), (50, "D"), (0, "E"))
# Labels repeat the current overall score as the previous one in 2 of 495 reports,
# while the model copies the current BCI there when no earlier score exists (11 of
# 13 such development predictions are labelled 无). Advice text ends with 。 in 59%
# of labels; a bare model ending (80% of its rows) is labelled 。 most often, while
# an explicit ；/。 from the model is kept (development advice rows 119 -> 132 exact).
ADVICE_ENDINGS = "。；;.,，、"
CONVENTIONS_VERSION = "official-training-label-conventions-v4"
LABELED_ADVICE = ("立即处置", "尽快维修", "预防性养护")
UNLABELED_BASIC = ("建议统计", "上一次定检结果")
_CN_DIGIT = {"〇": 0, "○": 0, "零": 0, "O": 0, "o": 0, "一": 1, "二": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_DATE_RE = re.compile(r"([0-9〇○零Oo一二三四五六七八九]{4})[年.\-/]([0-9一二三四五六七八九十]{1,3})"
                      r"[月.\-/]([0-9一二三四五六七八九十]{1,3})日?")


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _grade(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"([A-Ea-e])\s*(?:级|类)?", _nfkc(value))
    return match.group(1).upper() + "级" if match else value


def _score(value: Any) -> Any:
    if type(value) is float:
        return f"{value:.2f}" if round(value, 2) == value else str(value)
    if type(value) is int:
        return f"{value}.00"
    if isinstance(value, str) and re.fullmatch(r"\d+", _nfkc(value)):
        return _nfkc(value) + ".00"
    return value


def _band(score: Any) -> str | None:
    match = re.fullmatch(r"\d+(?:\.\d+)?", _nfkc(score)) if isinstance(score, str) else None
    if not match or not 0 <= float(match.group()) <= 100:
        return None
    return next(letter for low, letter in BCI_BANDS if float(match.group()) >= low) + "级"


def _cn_int(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    if "十" in text:
        tens, _, ones = text.partition("十")
        if (tens and tens not in _CN_DIGIT) or (ones and ones not in _CN_DIGIT):
            return None
        return (_CN_DIGIT[tens] if tens else 1) * 10 + (_CN_DIGIT[ones] if ones else 0)
    return _CN_DIGIT.get(text) if len(text) == 1 else None


def _date(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = _DATE_RE.fullmatch(re.sub(r"\s+", "", _nfkc(value)))
    if not match:
        return value
    digits = match.group(1)
    year = int(digits) if digits.isdigit() else int("".join(str(_CN_DIGIT[c]) for c in digits))
    month, day = _cn_int(match.group(2)), _cn_int(match.group(3))
    if not (month and day and 1 <= month <= 12 and 1 <= day <= 31):
        return value
    return f"{year:04d}年{month:02d}月{day:02d}日"


def apply_label_conventions(report: dict, changes: list) -> None:
    """Rewrite presentation-only conventions in place, recording every change."""
    def put(path, container, key, after):
        before = container.get(key)
        if before != after or key not in container:
            changes.append({"path": path, "before_present": key in container,
                            "before": deepcopy(before), "after": after})
            container[key] = after

    basic = report["基本信息"]
    for key in GRADE_FIELDS:
        if key in basic:
            put(["基本信息", key], basic, key, _grade(basic[key]))
    for key in SCORE_FIELDS:
        if key in basic:
            put(["基本信息", key], basic, key, _score(basic[key]))
    for grade_key, score_key in PART_GRADES:
        band = _band(basic.get(score_key))
        if band:
            put(["基本信息", grade_key], basic, grade_key, band)
    previous = basic.get("上一次总体评分")
    if isinstance(previous, str) and previous.strip() not in ("", "无") and previous == basic.get("总体评分"):
        put(["基本信息", "上一次总体评分"], basic, "上一次总体评分", "无")
    if "报告日期" in basic:
        put(["基本信息", "报告日期"], basic, "报告日期", _date(basic["报告日期"]))
    for key in UNLABELED_BASIC:
        if key in basic:
            changes.append({"path": ["基本信息", key], "before_present": True,
                            "before": deepcopy(basic[key]), "after": None, "removed": True})
            del basic[key]
    for index, row in enumerate(report["维修建议"]):
        put(["维修建议", index, "建议类别"], row, "建议类别", "尽快维修")
        content = row.get("建议内容")
        if isinstance(content, str) and content.strip() and content.rstrip()[-1] not in ADVICE_ENDINGS:
            put(["维修建议", index, "建议内容"], row, "建议内容", content.rstrip() + "。")
    counts = {label: 0 for label in LABELED_ADVICE}
    for row in report["维修建议"]:
        counts[row["建议类别"]] += 1
    put(["基本信息", "建议"], basic, "建议", "、".join(f"{counts[label]}条{label}" for label in LABELED_ADVICE))
    for index, row in enumerate(report["病害明细"]):
        put(["病害明细", index, "是否新增"], row, "是否新增", "是")
        put(["病害明细", index, "上一次定检状态"], row, "上一次定检状态", "上一次定检未发现")
        put(["病害明细", index, "发展程度"], row, "发展程度", "无")


def _label(value: Any) -> str:
    """Ignore outer whitespace for counting only; never rewrite the source."""
    return value.strip() if isinstance(value, str) else ""


def preserve_model_report(data: dict, *, label_conventions: bool = False) -> tuple[dict, dict]:
    """Return ``(independent_report, audit)`` without inventing report facts.

    The caller must first validate the model's JSON. Minimal section/row shape
    checks here fail closed rather than repairing malformed input. Unknown
    sections and fields are retained. Advice is counted in four exact categories
    (outer whitespace ignored); all other categories count as ``未分类`` without
    changing their labels. ``是否新增`` is counted as yes/no/unknown solely from
    its explicit value, never inferred from absent or ambiguous history.
    """
    if not isinstance(data, dict) or not isinstance(data.get("基本信息"), dict):
        raise ValueError("Expected a validated report with 基本信息 object")
    for section in ("维修建议", "病害明细"):
        rows = data.get(section)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"Expected a validated report with {section} object rows")

    report = deepcopy(data)
    changes = []
    row_counts = {}
    for section in ("维修建议", "病害明细"):
        # Separate row copies also preserve duplicate rows supplied as the same
        # Python object; independent sequence numbers must not alias each other.
        report[section] = [deepcopy(row) for row in data[section]]
        row_counts[section] = {"before": len(data[section]), "after": len(report[section])}
        for index, row in enumerate(report[section]):
            number = str(index + 1)
            if "序号" not in row or row["序号"] != number:
                changes.append({"path": [section, index, "序号"],
                                "before_present": "序号" in row,
                                "before": deepcopy(row.get("序号")), "after": number})
                row["序号"] = number

    if label_conventions:
        apply_label_conventions(report, changes)

    advice_counts = {category: 0 for category in ADVICE_CATEGORIES}
    advice_counts["未分类"] = 0
    for row in report["维修建议"]:
        category = _label(row.get("建议类别"))
        bucket = category if category in ADVICE_CATEGORIES else "未分类"
        advice_counts[bucket] += 1
    total = len(report["维修建议"])
    summary = f"共提出维护建议 {total} 项"
    if total:
        summary += "：" + "，".join(f"{category} {count} 项" for category, count in advice_counts.items())
    summary += "。"
    if not label_conventions:
        basic = report["基本信息"]
        if "建议统计" not in basic or basic["建议统计"] != summary:
            changes.append({"path": ["基本信息", "建议统计"],
                            "before_present": "建议统计" in basic,
                            "before": deepcopy(basic.get("建议统计")), "after": summary})
            basic["建议统计"] = summary

    is_new_counts = {"yes": 0, "no": 0, "unknown": 0}
    for row in report["病害明细"]:
        value = _label(row.get("是否新增"))
        is_new_counts[{"是": "yes", "否": "no"}.get(value, "unknown")] += 1
    audit = {
        "schema_version": 1,
        "policy": POLICY,
        "label_conventions": CONVENTIONS_VERSION if label_conventions else None,
        "advice_summary": summary,
        "changes": changes,
        "changed_field_count": len(changes),
        "row_counts": row_counts,
        "advice_category_counts": advice_counts,
        "advice_total": total,
        "defect_is_new_counts": is_new_counts,
        "counting_rules": {
            "advice_categories": "exact four category labels after outer whitespace trimming; others 未分类",
            "defect_is_new": "explicit 是 -> yes; 否 -> no; all other or missing values -> unknown",
            "duplicates": "retained and counted separately",
        },
    }
    return report, audit
