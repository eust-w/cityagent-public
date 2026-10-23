from __future__ import annotations

import re
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree


CandidateChannel = Literal["native", "visual", "semantic", "rule", "human"]

ALLOWED_FIELDS = frozenset(
    {
        "bridge.name",
        "bridge.report_date",
        "scores.overall.score",
        "scores.overall.grade",
        "scores.superstructure.score",
        "scores.superstructure.grade",
        "scores.substructure.score",
        "scores.substructure.grade",
        "scores.deck.score",
        "scores.deck.grade",
        "scores.previous.score",
        "scores.previous.grade",
    }
)

_SCORE_ALIASES = {
    "scores.overall": ("BCI", "全桥技术状况指数", "总体评分", "综合评分"),
    "scores.superstructure": ("BSIs", "上部结构"),
    "scores.substructure": ("BSIx", "下部结构"),
    "scores.deck": ("BSIm", "桥面系"),
    "scores.previous": ("上一次总体", "上次总体", "上年度总体", "去年总体"),
}


def compact_text(value: str) -> str:
    return re.sub(r"[\t\r ]+", " ", value).strip()


def _side_values(text: str, value_pattern: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    suffix = re.compile(rf"({value_pattern})\s*级?\s*[（(]?\s*(左幅|右幅)\s*[）)]?", re.I)
    prefix = re.compile(rf"(左幅|右幅)\s*[:：]?\s*({value_pattern})\s*级?", re.I)
    for value, side in suffix.findall(text):
        pairs[side] = value
    for side, value in prefix.findall(text):
        pairs[side] = value
    return pairs


def _canonical_side_scores(text: str) -> str | None:
    pairs = _side_values(text, r"100(?:\.0+)?|[1-9]?\d(?:\.\d{1,3})?")
    if "左幅" not in pairs or "右幅" not in pairs:
        return None
    return f"{float(pairs['左幅']):.2f}（左幅）/{float(pairs['右幅']):.2f}（右幅）"


def _canonical_side_grades(text: str) -> str | None:
    pairs = _side_values(text, r"[ABCDE]")
    if "左幅" not in pairs or "右幅" not in pairs:
        return None
    return f"{pairs['左幅'].upper()}级（左幅）/{pairs['右幅'].upper()}级（右幅）"


def normalize_candidate(field: str, value: Any) -> str:
    text = compact_text(str(value or ""))
    if field.endswith(".score"):
        dual = _canonical_side_scores(text)
        if dual:
            return dual
        try:
            return f"{float(text):.2f}"
        except ValueError:
            return text
    if field.endswith(".grade"):
        dual = _canonical_side_grades(text)
        if dual:
            return dual
        match = re.search(r"([ABCDE])\s*级?", text, re.I)
        return f"{match.group(1).upper()}级" if match else text
    if field == "bridge.report_date":
        digits = re.findall(r"\d+", text)
        return "-".join(digits[:3])
    if field == "bridge.name":
        text = re.sub(r"(?:定期)?检测(?:评估)?报告|检测报告|定检报告|报告", "", text)
        return re.sub(r"[\s_\-—（）()·.,，。]+", "", text).lower()
    return text


def _candidate(
    field: str,
    value: str | float,
    *,
    expert: str,
    channel: CandidateChannel,
    confidence: float,
    evidence_text: str,
    region_id: str | None,
) -> dict[str, Any]:
    return {
        "field": field,
        "value": value,
        "expert": expert,
        "channel": channel,
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence_text": compact_text(evidence_text)[:500],
        "region_id": region_id,
        "weight": 1.0,
    }


def extract_field_candidates(
    text: str,
    *,
    expert: str,
    channel: CandidateChannel,
    region_id: str | None = None,
    confidence_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Extract conservative bridge-summary candidates from OCR/semantic text."""

    text = compact_text(text)
    if not text:
        return []
    candidates: list[dict[str, Any]] = []

    date_match = re.search(
        r"(?<!\d)((?:19|20)\d{2})\s*[年./-]\s*(0?[1-9]|1[0-2])\s*[月./-]\s*([0-2]?\d|3[01])\s*日?",
        text,
    )
    if date_match:
        value = f"{int(date_match.group(1))}年{int(date_match.group(2))}月{int(date_match.group(3))}日"
        candidates.append(
            _candidate(
                "bridge.report_date",
                value,
                expert=expert,
                channel=channel,
                confidence=0.91 * confidence_scale,
                evidence_text=date_match.group(0),
                region_id=region_id,
            )
        )

    name_patterns = (
        r"桥梁名称\s*[:：]?\s*([^\n；;]{2,60}?(?:大桥|人行天桥|立交桥|桥式通道|桥))(?=\s|$|检测|报告)",
        r"([^\n；;]{2,50}?(?:K\d+\+\d+(?:\.\d+)?[^\n；;]{0,20})?(?:大桥|人行天桥|立交桥|桥式通道))\s*(?:定期)?检测",
    )
    for pattern in name_patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        value = re.sub(r"^(?:桥梁名称|工程名称)\s*[:：]?", "", match.group(1)).strip()
        value = re.sub(r"(?:定期)?检测(?:评估)?报告.*$", "", value).strip(" ：:")
        if 2 <= len(value) <= 60 and value not in {"桥梁", "大桥", "人行天桥", "立交桥"}:
            candidates.append(
                _candidate(
                    "bridge.name",
                    value,
                    expert=expert,
                    channel=channel,
                    confidence=0.84 * confidence_scale,
                    evidence_text=match.group(0),
                    region_id=region_id,
                )
            )
            break

    for prefix, aliases in _SCORE_ALIASES.items():
        alias_pattern = "|".join(re.escape(alias) for alias in aliases)
        for match in re.finditer(alias_pattern, text, re.I):
            before = text[max(0, match.start() - 24) : match.start()]
            after = text[match.end() : min(len(text), match.end() + 100)]
            window = before + text[match.start() : match.end()] + after
            dual_score = _canonical_side_scores(window)
            dual_grade = _canonical_side_grades(window)
            score_match = re.search(r"(?<!\d)(100(?:\.0+)?|[1-9]?\d(?:\.\d{1,3})?)(?!\d)", after)
            if score_match is None:
                preceding = list(re.finditer(r"(?<!\d)(100(?:\.0+)?|[1-9]?\d(?:\.\d{1,3})?)(?!\d)", before))
                score_match = preceding[-1] if preceding else None
            grade_match = re.search(r"\b([ABCDE])\s*级?\b|([ABCDE])级", window, re.I)
            if dual_score:
                candidates.append(
                    _candidate(
                        f"{prefix}.score",
                        dual_score,
                        expert=expert,
                        channel=channel,
                        confidence=0.93 * confidence_scale,
                        evidence_text=window,
                        region_id=region_id,
                    )
                )
            elif score_match:
                score = float(score_match.group(1))
                if 0.0 <= score <= 100.0:
                    candidates.append(
                        _candidate(
                            f"{prefix}.score",
                            score,
                            expert=expert,
                            channel=channel,
                            confidence=0.90 * confidence_scale,
                            evidence_text=window,
                            region_id=region_id,
                        )
                    )
            if dual_grade:
                candidates.append(
                    _candidate(
                        f"{prefix}.grade",
                        dual_grade,
                        expert=expert,
                        channel=channel,
                        confidence=0.92 * confidence_scale,
                        evidence_text=window,
                        region_id=region_id,
                    )
                )
            elif grade_match:
                grade = (grade_match.group(1) or grade_match.group(2)).upper() + "级"
                candidates.append(
                    _candidate(
                        f"{prefix}.grade",
                        grade,
                        expert=expert,
                        channel=channel,
                        confidence=0.88 * confidence_scale,
                        evidence_text=window,
                        region_id=region_id,
                    )
                )
            if dual_score or dual_grade or score_match or grade_match:
                break

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (candidate["field"], normalize_candidate(candidate["field"], candidate["value"]))
        existing = unique.get(key)
        if existing is None or candidate["confidence"] > existing["confidence"]:
            unique[key] = candidate
    return list(unique.values())


def best_page_numbers(pages: Iterable[str], *, limit: int = 2) -> list[int]:
    keywords = {
        "BCI": 12,
        "BSIm": 12,
        "BSIs": 12,
        "BSIx": 12,
        "技术状况评定": 8,
        "桥梁名称": 6,
        "总体评分": 8,
        "上部结构": 3,
        "下部结构": 3,
        "桥面系": 3,
    }
    ranked: list[tuple[int, int]] = []
    for index, page in enumerate(pages, start=1):
        score = sum(weight * page.count(keyword) for keyword, weight in keywords.items())
        ranked.append((score, index))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = [index for score, index in ranked if score >= 6][:limit]
    if not selected and ranked:
        selected = [1]
    return sorted(selected)


def best_docx_page_numbers(document: Path, *, limit: int = 2) -> list[int]:
    word_namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(document) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    pages: list[list[str]] = [[]]
    for element in root.iter():
        if element.tag == f"{word_namespace}t" and element.text:
            pages[-1].append(element.text)
        elif element.tag == f"{word_namespace}lastRenderedPageBreak" or (
            element.tag == f"{word_namespace}br" and element.get(f"{word_namespace}type") == "page"
        ):
            pages.append([])
    return best_page_numbers((" ".join(page) for page in pages), limit=limit)
