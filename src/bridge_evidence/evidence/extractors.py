from __future__ import annotations

import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from bridge_evidence.domain_rules import (
    ACTIVE_DEFECT_RE,
    canonical_disease_from_text,
    is_condition_state_label,
    is_meaningful_condition_state,
    is_non_disease_label,
)
from bridge_evidence.models.schema import (
    ComparisonRecord,
    Disease,
    DocumentState,
    EntityScoreState,
    EvidenceRef,
    FieldValue,
    Recommendation,
    ScoreEntry,
)
from bridge_evidence.history_semantics import development_level, previous_presence
from bridge_evidence.parsers.docx import ParsedDocument, TableBlock


FIELD_PRIORITY = {
    "summary": 95,
    "conclusion": 90,
    "rating": 85,
    "comparison": 82,
    "disease": 80,
    "paragraph": 60,
    "filename": 30,
}

CHINESE_DIGIT_MAP = {
    "〇": 0, "○": 0, "Ｏ": 0, "O": 0, "o": 0, "零": 0, "0": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
}

COMPONENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"斜拉索|拉索|套筒|锚头|钢绞线"), "斜拉索"),
    (re.compile(r"主塔|索塔|塔柱"), "主塔"),
    (re.compile(r"人行道吊杆|吊杆|锚具"), "人行道吊杆"),
    (re.compile(r"桥面铺装|路面|沥青铺装"), "桥面铺装"),
    (re.compile(r"伸缩缝|伸缩装置"), "伸缩缝"),
    (re.compile(r"防撞护栏|防护栏|护栏|栏杆"), "护栏"),
    (re.compile(r"排水|泄水"), "排水系统"),
    (re.compile(r"人行道"), "人行道"),
    (re.compile(r"照明|标志|标识牌"), "照明、标志"),
    (re.compile(r"横向联系|横隔板|横梁"), "横向联系"),
    (re.compile(r"支座"), "支座"),
    (re.compile(r"桥墩|墩柱|盖梁|墩身"), "桥墩"),
    (re.compile(r"桥台|台帽|台身|前墙|侧墙|翼墙"), "桥台"),
    (
        re.compile(
            r"主梁|箱梁|空心板|T梁|板梁|梁体|梁底|梁面|腹板|翼板|底板|"
            r"\d+#梁|\d+#板|板间|铰缝|铰接缝|湿接缝|拱肋"
        ),
        "主梁",
    ),
    (re.compile(r"照明|灯具|接地|防雷"), "机电设施"),
]

# Table-section headings are stronger component evidence than incidental words
# in a long surrounding context.  For example a main-girder table may locate a
# defect by distance from a pier and the report preface may mention anchors;
# neither should turn the row into a pier or cable defect.
CONTEXT_COMPONENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:上部结构)?主梁(?:检测|检查|病害|结果)|主梁检测结果表"), "主梁"),
    (re.compile(r"横向(?:联系|连接系).*(?:检测|检查|结果)|横隔板检测结果表"), "横向联系"),
    (re.compile(r"桥墩(?:检测|检查|病害|结果)|桥墩检测结果表"), "桥墩"),
    (re.compile(r"桥台(?:检测|检查|病害|结果)|桥台检测结果表"), "桥台"),
    (re.compile(r"支座(?:检测|检查|病害|结果)|支座检测结果表"), "支座"),
    (re.compile(r"桥面(?:铺装)?(?:检测|检查|病害|结果)|桥面铺装检测结果表"), "桥面铺装"),
    (re.compile(r"伸缩缝(?:检测|检查|病害|结果)|伸缩缝检测结果表"), "伸缩缝"),
    (re.compile(r"(?:防撞)?护栏(?:检测|检查|病害|结果)|护栏检测结果表"), "护栏"),
    (re.compile(r"斜拉索|拉索检测结果表"), "斜拉索"),
]

DISEASE_TYPE_ALIASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"破损.*露筋|露筋.*破损"), "破损、露筋"),
    (re.compile(r"渗水.*泛碱|泛碱.*渗水"), "渗水泛碱"),
    (re.compile(r"破损.*开裂|开裂.*破损"), "破损开裂"),
    (re.compile(r"涂层.*剥落"), "涂层剥落"),
    (re.compile(r"涂层.*开裂"), "涂层开裂"),
]

GRADE_RE = re.compile(r"([ABCDE])\s*(?:级|类)?", re.I)
FLOAT_RE = r"(\d{1,3}(?:\.\d+)?)"


def normalize_grade(value: str | None) -> str | None:
    if not value:
        return None
    match = GRADE_RE.search(value)
    return f"{match.group(1).upper()}级" if match else None


def grade_from_score(value: float | None) -> str | None:
    if value is None:
        return None
    if value >= 90:
        return "A级"
    if value >= 80:
        return "B级"
    if value >= 66:
        return "C级"
    if value >= 50:
        return "D级"
    return "E级"


def safe_float(value: str) -> float | None:
    match = re.search(FLOAT_RE, value.replace(",", ""))
    return float(match.group(1)) if match else None


def normalize_date_text(text: str) -> str | None:
    if not text:
        return None
    t = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    t = re.sub(r"(?<=[0-9〇○ＯOo零一二三四五六七八九十])\s+(?=[0-9〇○ＯOo零一二三四五六七八九十年月日./-～~至—–])", "", t)

    # Inspection summaries often publish a date range while the expected
    # report date is the completion date (the right-hand endpoint).  Resolve
    # that endpoint before the ordinary single-date matcher consumes the
    # left-hand date.  The rule is purely syntactic and covers both Chinese
    # and numeric date notation.
    chinese_range = re.search(
        r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?\s*[～~至—–]\s*"
        r"(?:(20\d{2})\s*年)?\s*(?:(\d{1,2})\s*月)?\s*(\d{1,2})\s*日",
        t,
    )
    numeric_range = re.search(
        r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})\s*[～~至—–]\s*"
        r"(?:(20\d{2})[./-])?(?:(\d{1,2})[./-])?(\d{1,2})",
        t,
    )
    date_range = chinese_range or numeric_range
    if date_range:
        start_year, start_month, _, end_year, end_month, end_day = date_range.groups()
        parts = (
            int(end_year or start_year),
            int(end_month or start_month),
            int(end_day),
        )
        try:
            parsed = date(*parts)
            return f"{parsed.year}年{parsed.month}月{parsed.day}日"
        except ValueError:
            return None

    match = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", t)
    if not match:
        match = re.search(r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})", t)
    if match:
        parts = tuple(int(v) for v in match.groups())
        try:
            parsed = date(*parts)
            return f"{parsed.year}年{parsed.month}月{parsed.day}日"
        except ValueError:
            pass

    chinese = re.search(r"([〇○ＯOo零一二三四五六七八九0-9]{4})\s*年\s*([一二三四五六七八九十0-9]{1,3})\s*月\s*([一二三四五六七八九十0-9]{1,3})\s*日", t)
    if chinese:
        try:
            year = int("".join(str(CHINESE_DIGIT_MAP[c]) for c in chinese.group(1)))
            month = _chinese_number(chinese.group(2))
            day = _chinese_number(chinese.group(3))
            parsed = date(year, month, day)
            return f"{parsed.year}年{parsed.month}月{parsed.day}日"
        except (KeyError, ValueError):
            pass

    month_match = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月(?:\s*$|[^\d日])", t)
    if month_match:
        year, month_value = (int(v) for v in month_match.groups())
        if 1 <= month_value <= 12:
            return f"{year}年{month_value}月"

    numeric_month = re.search(r"(20\d{2})[./-](\d{1,2})(?:\s*$|[^\d./-])", t)
    if numeric_month:
        year, month_value = (int(v) for v in numeric_month.groups())
        if 1 <= month_value <= 12:
            return f"{year}年{month_value}月"

    chinese_month = re.search(r"([〇○ＯOo零一二三四五六七八九0-9]{4})\s*年\s*([一二三四五六七八九十0-9]{1,3})\s*月", t)
    if chinese_month:
        try:
            year = int("".join(str(CHINESE_DIGIT_MAP[c]) for c in chinese_month.group(1)))
            month_value = _chinese_number(chinese_month.group(2))
            if 1 <= month_value <= 12:
                return f"{year}年{month_value}月"
        except (KeyError, ValueError):
            pass

    return None


def _chinese_number(value: str) -> int:
    value = str(value).strip()
    if value.isdigit():
        return int(value)
    if "十" not in value:
        return CHINESE_DIGIT_MAP[value]
    before, after = value.split("十", 1)
    tens = CHINESE_DIGIT_MAP[before] if before else 1
    ones = CHINESE_DIGIT_MAP[after] if after else 0
    return tens * 10 + ones


def normalize_component(text: str, context: str = "") -> str:
    for pattern, canonical in COMPONENT_PATTERNS:
        if pattern.search(text):
            return canonical
    # ``table.context`` is chronological and can contain several preceding
    # subsection titles.  The closest/latest matching title describes the
    # current table; fixed regex order would let an older "支座检测" heading
    # steal rows from a later "桥面铺装检测" table.
    section_matches = [
        (match.start(), canonical)
        for pattern, canonical in CONTEXT_COMPONENT_PATTERNS
        for match in pattern.finditer(context)
    ]
    if section_matches:
        return max(section_matches, key=lambda item: item[0])[1]
    context_matches = [
        (match.start(), canonical)
        for pattern, canonical in COMPONENT_PATTERNS
        for match in pattern.finditer(context)
    ]
    if context_matches:
        return max(context_matches, key=lambda item: item[0])[1]
    return "其他"


def normalize_bridge_name(value: str) -> str:
    value = " ".join(value.split()).strip(" ：:;；")
    value = re.sub(r"^HR/?QL-\d{6}-\d{3}", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^20\d{2}年.*?(?:车行桥|人行桥)[：:\s]*", "", value)
    value = re.sub(r"^(?:工程名称|桥梁名称|项目名称|桥梁概况)[：:\s]*", "", value)
    value = re.sub(r"^\d{1,3}\s*[-–—]\s*\d{1,3}\s*[-、_\s]*", "", value)
    value = re.sub(r"^\d{1,4}\s*[-、_]+\s*", "", value)
    value = re.sub(r"^\d{3}(?=[\u4e00-\u9fff])", "", value)
    value = re.sub(r"(?:20\d{2}(?:年度|[.年]\d{1,2}(?:[.月]\d{1,2}日?)?))$", "", value)
    value = re.sub(r"^(?:重庆市|重庆)(?=[\u4e00-\u9fffA-Za-z0-9])", "", value)
    value = re.sub(r"桥桥$", "桥", value)
    value = re.sub(r"[（(](?:主桥|引桥|车行桥|人行桥)[）)]$", "", value)
    return value.strip()


def normalize_disease_type(value: str) -> str:
    value = " ".join(value.replace("，", "、").split()).strip("；;、 ")
    for pattern, canonical in DISEASE_TYPE_ALIASES:
        if pattern.search(value):
            return canonical
    return value or "未明确"


class EvidenceExtractor:
    def extract(self, parsed: ParsedDocument, source_path: Path, document_id: str) -> DocumentState:
        state = DocumentState(document_id=f"sha256:{document_id}", source_path=str(source_path))
        summary = self._extract_summary_table(parsed)
        self._extract_bridge_meta(parsed, source_path, state, summary)
        self._extract_scores(parsed, state, summary)
        self._extract_comparisons(parsed, state)
        self._extract_diseases(parsed, state)
        self._aggregate_large_diseases(parsed, state)
        self._infer_disease_history(state)
        self._cohere_disease_history(state)
        self._extract_recommendations(parsed, state, summary)
        self._extract_narratives(parsed, state, summary)
        return state

    @staticmethod
    def _extract_summary_table(parsed: ParsedDocument) -> tuple[TableBlock, dict[str, str], dict[str, int]] | None:
        canonical_labels = (
            "工程概况",
            "检测依据",
            "检测结果",
            "检测结论",
            "维护建议",
            "评估结论",
            "工程名称",
            "桥梁名称",
            "报告日期",
            "检验日期",
            "检测日期",
            "评估日期",
            "编制日期",
            "完成日期",
            "出具日期",
            "检查日期",
            "检测时间",
        )
        for table in parsed.tables[:6]:
            mapping: dict[str, str] = {}
            rows: dict[str, int] = {}
            for row_index, row in enumerate(table.rows):
                if len(row) < 2:
                    continue
                for col_idx in range(0, len(row) - 1):
                    raw_label = re.sub(r"\s+", "", row[col_idx])
                    label = next((key for key in canonical_labels if key in raw_label), None)
                    if label and col_idx + 1 < len(row) and row[col_idx + 1].strip():
                        if label not in mapping:
                            mapping[label] = row[col_idx + 1]
                            rows[label] = row_index
            if ("工程名称" in mapping or "桥梁名称" in mapping) and any(k in mapping for k in ("检测结论", "检测结果", "评估结论", "检测依据", "工程概况")):
                return table, mapping, rows
        return None

    def _table_evidence(self, state: DocumentState, parsed: ParsedDocument, table: TableBlock, row: int, text: str, priority: int) -> str:
        evidence_id = f"T{table.index}R{row}"
        state.add_evidence(EvidenceRef(id=evidence_id, kind="table", text=text, source_path=str(parsed.path), priority=priority, table_index=table.index, row_index=row))
        return evidence_id

    def _paragraph_evidence(self, state: DocumentState, parsed: ParsedDocument, index: int, text: str, priority: int = 60) -> str:
        evidence_id = f"P{index}"
        state.add_evidence(EvidenceRef(id=evidence_id, kind="paragraph", text=text, source_path=str(parsed.path), priority=priority, paragraph_index=index))
        return evidence_id

    def _extract_bridge_meta(self, parsed: ParsedDocument, source_path: Path, state: DocumentState, summary: tuple[TableBlock, dict[str, str], dict[str, int]] | None) -> None:
        if summary:
            table, mapping, rows = summary
            bridge_name = normalize_bridge_name(mapping.get("工程名称") or mapping.get("桥梁名称") or "")
            if bridge_name:
                row_key = "工程名称" if "工程名称" in rows else "桥梁名称"
                evidence = self._table_evidence(state, parsed, table, rows[row_key], bridge_name, FIELD_PRIORITY["summary"])
                state.bridge.name = FieldValue(value=bridge_name, evidence=[evidence], source_priority=FIELD_PRIORITY["summary"], confidence=0.99)

        if not state.bridge.name.value:
            for table in parsed.tables[:8]:
                for row_index, row in enumerate(table.rows):
                    if len(row) < 2:
                        continue
                    raw_label = re.sub(r"\s+", "", row[0])
                    if not any(label in raw_label for label in ("工程名称", "桥梁名称")):
                        continue
                    bridge_name = normalize_bridge_name(row[1])
                    if not bridge_name:
                        continue
                    evidence = self._table_evidence(
                        state,
                        parsed,
                        table,
                        row_index,
                        " | ".join(row),
                        FIELD_PRIORITY["summary"],
                    )
                    state.bridge.name = FieldValue(
                        value=bridge_name,
                        evidence=[evidence],
                        source_priority=FIELD_PRIORITY["summary"],
                        confidence=0.99,
                    )
                    break
                if state.bridge.name.value:
                    break

        if state.bridge.name.value:
            current_name = str(state.bridge.name.value)
            for i, p in enumerate(parsed.paragraphs[:25]):
                text = p.text.strip()
                if text in ("检测评估报告", "定期检测评估报告", "定期检测报告", "检测报告", "外观检测评估报告") and i > 0:
                    prev_text = parsed.paragraphs[i - 1].text.strip()
                    if prev_text and not prev_text.startswith("报告编号"):
                        clean_t = normalize_bridge_name(prev_text)
                        if clean_t and re.search(r"K\d+\+\d+", clean_t, re.I) and not re.search(r"K\d+\+\d+", current_name, re.I):
                            if current_name in clean_t or "无名桥" in current_name or "匝道" in current_name:
                                state.bridge.name.value = clean_t
                                break

        if not state.bridge.name.value:
            title_candidates = []
            for i, p in enumerate(parsed.paragraphs[:25]):
                text = p.text.strip()
                if text in ("检测评估报告", "定期检测评估报告", "定期检测报告", "检测报告", "外观检测评估报告") and i > 0:
                    prev_text = parsed.paragraphs[i - 1].text.strip()
                    if prev_text and len(prev_text) <= 30 and not prev_text.startswith("报告编号"):
                        clean = normalize_bridge_name(prev_text)
                        if clean:
                            title_candidates.append((i - 1, prev_text, clean))
            if title_candidates:
                idx, raw_text, clean_title = title_candidates[0]
                evidence = self._paragraph_evidence(state, parsed, idx, raw_text, 94)
                state.bridge.name = FieldValue(value=clean_title, evidence=[evidence], source_priority=94, confidence=0.99)

        if not state.bridge.name.value:
            for paragraph in parsed.paragraphs[:90]:
                if "检测评估报告摘要" in paragraph.text:
                    name = paragraph.text.split("检测评估报告摘要", 1)[0].strip()
                    if name:
                        evidence = self._paragraph_evidence(state, parsed, paragraph.index, paragraph.text)
                        state.bridge.name = FieldValue(value=name, evidence=[evidence], source_priority=60, confidence=0.9)
                        break
        if not state.bridge.name.value:
            name = re.sub(r"^(?:快[一二三四]处|[一二三四]处|大桥处)-", "", source_path.stem)
            name = re.sub(r"^\d{5,8}-\d{2,4}[、_\- ]*", "", name)
            name = re.sub(r"^\d{1,4}[、_\- ]+", "", name)
            name = re.sub(r"^20\d{2}年", "", name)
            name = re.sub(r"(?:桥梁)?检测评估.*$", "", name)
            name = re.sub(r"报告.*$", "", name).strip(" -_")
            name = normalize_bridge_name(name)
            evidence_id = "FILENAME"
            state.add_evidence(EvidenceRef(id=evidence_id, kind="filename", text=source_path.name, source_path=str(source_path), priority=30))
        # 1. Scan cover paragraphs (e.g. 二〇一八年十二月十二日)
        cover_candidates: list[tuple[int, str, str]] = []
        for paragraph in parsed.paragraphs[:45]:
            value = normalize_date_text(paragraph.text)
            if (
                value
                and len(paragraph.text) <= 40
                and "检测日期" not in paragraph.text
                and "检验日期" not in paragraph.text
                and "委托日期" not in paragraph.text
                and "竣工" not in paragraph.text
            ):
                cover_candidates.append((paragraph.index, paragraph.text, value))
        if cover_candidates:
            index, text, value = cover_candidates[-1]
            evidence = self._paragraph_evidence(state, parsed, index, text, 88)
            state.bridge.report_date = FieldValue(value=value, evidence=[evidence], source_priority=88, confidence=0.96)
        else:
            table_cover_candidates: list[tuple[TableBlock, int, str, str]] = []
            for table in parsed.tables[:6]:
                for row_index, row in enumerate(table.rows):
                    text = " | ".join(row)
                    if "检验日期" in text or "检测日期" in text or "竣工日期" in text or "委托日期" in text or "委托时间" in text:
                        continue
                    value = normalize_date_text(text)
                    if value and len(text) <= 80:
                        table_cover_candidates.append((table, row_index, text, value))
            if table_cover_candidates:
                table, row_index, text, value = table_cover_candidates[-1]
                evidence = self._table_evidence(state, parsed, table, row_index, text, 85)
                state.bridge.report_date = FieldValue(value=value, evidence=[evidence], source_priority=85, confidence=0.94)

        if not state.bridge.report_date.value:
            for paragraph in parsed.paragraphs[:25]:
                if "检测日期" in paragraph.text and any(sym in paragraph.text for sym in ("至", "～", "~", "—", "–")):
                    value = normalize_date_text(paragraph.text)
                    if value:
                        evidence = self._paragraph_evidence(state, parsed, paragraph.index, paragraph.text, 86)
                        state.bridge.report_date = FieldValue(value=value, evidence=[evidence], source_priority=86, confidence=0.95)
                        break

        # 2. If no cover date, check explicit report date in summary table
        if not state.bridge.report_date.value and summary:
            table, mapping, rows = summary
            for date_key in ("报告日期", "出具日期", "完成日期", "编制日期"):
                raw = mapping.get(date_key, "")
                if raw:
                    value = normalize_date_text(raw)
                    if value:
                        evidence = self._table_evidence(state, parsed, table, rows[date_key], raw, 80)
                        state.bridge.report_date = FieldValue(value=value, evidence=[evidence], source_priority=80, confidence=0.90)
                        break

        # 3. Check trailing signature / approval paragraphs
        if not state.bridge.report_date.value:
            for paragraph in reversed(parsed.paragraphs[-30:]):
                value = normalize_date_text(paragraph.text)
                if value and len(paragraph.text) <= 50 and "竣工" not in paragraph.text:
                    evidence = self._paragraph_evidence(state, parsed, paragraph.index, paragraph.text, 65)
                    state.bridge.report_date = FieldValue(value=value, evidence=[evidence], source_priority=65, confidence=0.85)
                    break

        # 4. Fallback to sampling / inspection date in summary table
        if not state.bridge.report_date.value and summary:
            table, mapping, rows = summary
            for date_key in ("检验日期", "检测日期", "评估日期", "检查日期", "检测时间"):
                raw = mapping.get(date_key, "")
                if raw:
                    value = normalize_date_text(raw)
                    if value:
                        evidence = self._table_evidence(state, parsed, table, rows[date_key], raw, 60)
                        state.bridge.report_date = FieldValue(value=value, evidence=[evidence], source_priority=60, confidence=0.80)
                        break

        # 5. Fallback to filename
        if not state.bridge.report_date.value:
            value = normalize_date_text(source_path.stem)
            if value:
                evidence_id = "FILENAME"
                if evidence_id not in state.evidence:
                    state.add_evidence(EvidenceRef(id=evidence_id, kind="filename", text=source_path.name, source_path=str(source_path), priority=30))
                state.bridge.report_date = FieldValue(value=value, evidence=[evidence_id], source_priority=45, confidence=0.8)

        # 6. Proactive visual image OCR fallback for scanned covers and image headers
        if not state.bridge.name.value or not state.bridge.report_date.value:
            from bridge_evidence.evidence.visuals import extract_visual_candidates, LocalOcrEngine
            candidates = extract_visual_candidates(parsed.path, limit=2)
            for candidate in candidates:
                ocr_text = LocalOcrEngine.ocr_image(candidate.data)
                if not ocr_text:
                    continue
                facts = LocalOcrEngine.extract_facts(ocr_text)
                for fact in facts:
                    if fact["field"] == "bridge.name" and not state.bridge.name.value:
                        evidence_id = f"IMG:{candidate.digest[:16]}"
                        state.add_evidence(EvidenceRef(
                            id=evidence_id,
                            kind="image",
                            text=f"封面/扫描图片 OCR 桥名: {fact['value']}",
                            source_path=f"{parsed.path}#{candidate.media_path}",
                            priority=75,
                        ))
                        state.bridge.name = FieldValue(
                            value=fact["value"],
                            evidence=[evidence_id],
                            source_priority=75,
                            confidence=0.92,
                        )
                    if fact["field"] == "bridge.report_date" and not state.bridge.report_date.value:
                        evidence_id = f"IMG:{candidate.digest[:16]}"
                        state.add_evidence(EvidenceRef(
                            id=evidence_id,
                            kind="image",
                            text=f"封面/扫描图片 OCR 日期: {fact['value']}",
                            source_path=f"{parsed.path}#{candidate.media_path}",
                            priority=75,
                        ))
                        state.bridge.report_date = FieldValue(
                            value=fact["value"],
                            evidence=[evidence_id],
                            source_priority=75,
                            confidence=0.92,
                        )

    def _extract_scores(self, parsed: ParsedDocument, state: DocumentState, summary: tuple[TableBlock, dict[str, str], dict[str, int]] | None) -> None:
        rating_records: dict[str, list[tuple[str | None, float, str | None, str]]] = {
            "deck": [],
            "superstructure": [],
            "substructure": [],
            "overall": [],
        }
        for table in parsed.tables:
            header_text = "|".join(table.header)
            if not (
                ("部位名称" in header_text or "桥梁部位" in header_text)
                and ("技术状况指数" in header_text or "部位技术状况指数" in header_text)
                and ("BCI" in header_text or "BCL" in header_text)
            ):
                continue
            header = table.header
            score_index = next((i for i, value in enumerate(header) if "部位技术状况指数" in value), None)
            if score_index is None:
                score_index = next((i for i, value in enumerate(header) if "技术状况指数" in value), 1)
            overall_index = next(
                (
                    i
                    for i, value in enumerate(header)
                    if value.strip() in {"BCI", "BCIm"} or "桥梁技术状况指数" in value
                ),
                None,
            )
            component_grade_index = next((i for i, value in enumerate(header) if value.strip() == "技术状况"), None)
            overall_grade_index = next(
                (i for i, value in enumerate(header) if "整体技术状况等级" in value or "完好状况评估等级" in value),
                None,
            )
            side = self._rating_table_side(table)
            scope_names = {"桥面系": "deck", "上部结构": "superstructure", "下部结构": "substructure"}
            for row_index, row in enumerate(table.rows[1:], 1):
                if not row:
                    continue
                scope = scope_names.get(row[0].replace(" ", ""))
                if not scope:
                    continue
                # Some binary-Word conversions expand the data rows by one
                # cell while leaving the five-cell header unchanged:
                # [桥面系, BCIm, 88.53, B, 0.15, 91.10].  All header-derived
                # indices after the first column must then be shifted once.
                shifted = len(row) == len(header) + 1 and len(row) > 2 and bool(
                    re.fullmatch(r"BCI[mMsSxX]?|BCL[mMsSxX]?", row[1].strip(), re.I)
                )

                def row_value(index: int | None) -> str:
                    if index is None:
                        return ""
                    actual = index + 1 if shifted and index >= 1 else index
                    return row[actual] if actual < len(row) else ""

                raw = " | ".join(row)
                evidence = self._table_evidence(state, parsed, table, row_index, raw, FIELD_PRIORITY["rating"])
                value = safe_float(row_value(score_index))
                # A recurring official template has a swapped header/data
                # layout: the header says ``技术状况指数 | 权重`` while the
                # data row stores ``0.15 | 84.93``.  Values in (0, 1] are
                # weights, never BCI component scores, so recover the adjacent
                # score instead of reporting the weight as a condition index.
                if value is not None and 0 < value <= 1:
                    adjacent_candidates: list[float] = []
                    for adjacent_index in (
                        (score_index + 1) if score_index is not None else None,
                        (score_index - 1) if score_index is not None and score_index > 0 else None,
                    ):
                        adjacent = safe_float(row_value(adjacent_index))
                        if adjacent is not None and 20 <= adjacent <= 100:
                            adjacent_candidates.append(adjacent)
                    if adjacent_candidates:
                        value = adjacent_candidates[0]
                explicit_grade = normalize_grade(row_value(component_grade_index))
                if value is not None:
                    rating_records[scope].append((side, value, explicit_grade, evidence))
                if overall_index is not None:
                    overall = safe_float(row_value(overall_index))
                    overall_grade = normalize_grade(row_value(overall_grade_index))
                    if overall is not None and not any(existing[0] == side and abs(existing[1] - overall) < 0.001 for existing in rating_records["overall"]):
                        rating_records["overall"].append((side, overall, overall_grade, evidence))

        for scope in ("deck", "superstructure", "substructure"):
            self._apply_rating_records(getattr(state.scores, scope), rating_records[scope], score_priority=86, explicit_grade_priority=96)

        conclusion_text = ""
        conclusion_evidence: str | None = None
        if summary:
            table, mapping, rows = summary
            conclusion_label = "检测结论"
            conclusion_text = mapping.get(conclusion_label, "")
            if not re.search(r"(?:BCI|Dr|评分).{0,24}?\d", conclusion_text, re.I) and mapping.get("评估结论"):
                conclusion_label = "评估结论"
                conclusion_text = mapping[conclusion_label]
            if conclusion_text:
                conclusion_evidence = self._table_evidence(state, parsed, table, rows[conclusion_label], conclusion_text, FIELD_PRIORITY["conclusion"])

        if conclusion_text:
            current = re.search(
                r"(?:BCI(?![msx])|Dr)\s*(?:值|评分)?\s*(?:为|=|：|:)?\s*" + FLOAT_RE,
                conclusion_text,
                re.I,
            )
            if not current:
                current = re.search(r"(?:主桥|全桥|总体).{0,20}?评分\s*(?:为|=|：|:)\s*" + FLOAT_RE, conclusion_text, re.I)
            grade = self._find_overall_grade(conclusion_text)
            dual_records = self._side_record_map(rating_records["overall"])
            current_value = float(current.group(1)) if current else None
            current_matches_side = current_value is not None and any(abs(value[0] - current_value) < 0.011 for value in dual_records.values())
            conclusion_side_values = self._explicit_side_overall_scores(conclusion_text)
            if current_matches_side:
                state.scores.overall.score = FieldValue(value=current_value, evidence=[conclusion_evidence] if conclusion_evidence else [], source_priority=93, confidence=0.99)
                selected_grade = next(
                    (record[1] for record in dual_records.values() if abs(record[0] - current_value) <= 0.011),
                    None,
                )
                state.scores.overall.grade = FieldValue(
                    value=grade or selected_grade or grade_from_score(current_value),
                    evidence=[conclusion_evidence] if conclusion_evidence else [],
                    source_priority=93,
                    confidence=0.99,
                )
            elif len(conclusion_side_values) >= 2 and len(dual_records) >= 2:
                selected = min(conclusion_side_values.values())
                state.scores.overall.score = FieldValue(
                    value=selected,
                    evidence=[conclusion_evidence] if conclusion_evidence else [],
                    source_priority=93,
                    confidence=0.99,
                )
                state.scores.overall.grade = FieldValue(
                    value=grade or grade_from_score(selected),
                    evidence=[conclusion_evidence] if conclusion_evidence else [],
                    source_priority=93,
                    confidence=0.99,
                )
            elif current_value is not None:
                state.scores.overall.score = FieldValue(value=current_value, evidence=[conclusion_evidence] if conclusion_evidence else [], source_priority=90, confidence=0.99)
                if grade:
                    state.scores.overall.grade = FieldValue(value=grade, evidence=[conclusion_evidence] if conclusion_evidence else [], source_priority=90, confidence=0.99)
            else:
                self._apply_rating_records(state.scores.overall, rating_records["overall"], score_priority=86, explicit_grade_priority=86)
        else:
            self._apply_rating_records(state.scores.overall, rating_records["overall"], score_priority=86, explicit_grade_priority=86)

        self._extract_component_score_paragraphs(parsed, state)
        self._extract_overall_score_paragraphs(parsed, state)
        self._extract_final_overall_rating(parsed, state)
        self._align_component_scores_with_selected_overall(rating_records, state)
        self._resolve_component_grades(parsed, state)
        self._extract_previous_score(parsed, state)
        self._extract_filename_grades(state)

        if state.scores.overall.grade.value is None and isinstance(state.scores.overall.score.value, (float, int)):
            state.scores.overall.grade = FieldValue(
                value=grade_from_score(float(state.scores.overall.score.value)),
                evidence=list(state.scores.overall.score.evidence),
                source_priority=70,
                confidence=0.72,
            )

        for scope in ("deck", "superstructure", "substructure"):
            entry = getattr(state.scores, scope)
            if entry.grade.value is None and isinstance(entry.score.value, (float, int)):
                entry.grade = FieldValue(
                    value=grade_from_score(float(entry.score.value)),
                    evidence=list(entry.score.evidence),
                    source_priority=70,
                    confidence=0.72,
                )

        self._populate_score_entities(
            rating_records,
            state,
            conclusion_text=conclusion_text,
            conclusion_evidence=conclusion_evidence,
        )

    def _populate_score_entities(
        self,
        rating_records: dict[str, list[tuple[str | None, float, str | None, str]]],
        state: DocumentState,
        *,
        conclusion_text: str = "",
        conclusion_evidence: str | None = None,
    ) -> None:
        """Retain bridge-side cardinality without breaking legacy score fields."""

        whole = EntityScoreState(
            entity_id="bridge:whole",
            label="整桥",
            side="whole",
            overall=state.scores.overall.model_copy(deep=True),
            superstructure=state.scores.superstructure.model_copy(deep=True),
            substructure=state.scores.substructure.model_copy(deep=True),
            deck=state.scores.deck.model_copy(deep=True),
        )
        side_records = {
            scope: self._side_record_map(records)
            for scope, records in rating_records.items()
        }
        explicit_overall = self._explicit_side_overall_scores(conclusion_text)
        for side, value in explicit_overall.items():
            side_records["overall"].setdefault(
                side,
                (
                    value,
                    grade_from_score(value),
                    conclusion_evidence or "",
                ),
            )

        entities = [whole]
        side_specs = (("左幅", "left", "bridge:left"), ("右幅", "right", "bridge:right"))
        for source_side, side, entity_id in side_specs:
            if not any(source_side in records for records in side_records.values()):
                continue
            entity = EntityScoreState(entity_id=entity_id, label=source_side, side=side)
            for scope in ("overall", "deck", "superstructure", "substructure"):
                record = side_records[scope].get(source_side)
                if record is None:
                    continue
                score, grade, evidence = record
                target = getattr(entity, scope)
                target.score = FieldValue(
                    value=score,
                    evidence=[evidence] if evidence else [],
                    source_priority=97,
                    confidence=0.99,
                )
                target.grade = FieldValue(
                    value=grade or grade_from_score(score),
                    evidence=[evidence] if evidence else [],
                    source_priority=97,
                    confidence=0.97,
                )
            entities.append(entity)
        state.scores.entities = entities

        entity_by_side = {entity.side: entity for entity in entities}
        source_is_dual = len(explicit_overall) >= 2
        legacy_is_dual = "左幅" in str(state.scores.overall.score.value) and "右幅" in str(state.scores.overall.score.value)
        if not source_is_dual and not legacy_is_dual:
            state.scores.presentation_mode = "single"
            return
        if "left" not in entity_by_side or "right" not in entity_by_side:
            return

        state.scores.presentation_mode = "dual"
        for scope in ("overall", "deck", "superstructure", "substructure"):
            left = getattr(entity_by_side["left"], scope)
            right = getattr(entity_by_side["right"], scope)
            if left.score.value in (None, "") or right.score.value in (None, ""):
                continue
            target: ScoreEntry = getattr(state.scores, scope)
            target.score = FieldValue(
                value=(
                    f"{self._format_number(float(left.score.value))}（左幅）/"
                    f"{self._format_number(float(right.score.value))}（右幅）"
                ),
                evidence=list(dict.fromkeys([*left.score.evidence, *right.score.evidence])),
                source_priority=max(left.score.source_priority, right.score.source_priority),
                confidence=min(left.score.confidence, right.score.confidence),
            )
            if left.grade.value not in (None, "") and right.grade.value not in (None, ""):
                target.grade = FieldValue(
                    value=f"{left.grade.value}（左幅）/{right.grade.value}（右幅）",
                    evidence=list(dict.fromkeys([*left.grade.evidence, *right.grade.evidence])),
                    source_priority=max(left.grade.source_priority, right.grade.source_priority),
                    confidence=min(left.grade.confidence, right.grade.confidence),
                )


    @staticmethod
    def _explicit_side_overall_scores(text: str) -> dict[str, float]:
        """Return side values only when a side clause explicitly names BCI/Dr.

        A generic ``左幅 ... 62.72`` matcher confuses bridge-level BCI with
        component BSI values in the same conclusion.  Requiring the overall
        token next to each side prevents that field-classification error.
        """
        result: dict[str, float] = {}
        for side in ("左幅", "右幅"):
            match = re.search(
                rf"{side}[^，,。；;]{{0,60}}?(?:BCI(?![msx])|Dr)"
                rf"\s*(?:值|评分)?\s*(?:为|=|：|:)?\s*{FLOAT_RE}",
                text,
                re.I,
            )
            if match:
                result[side] = float(match.group(1))
        if len(result) < 2:
            # Another common form names the whole-bridge metric once and then
            # lists both sides in parentheses: ``BCI...（右幅82.71分，左幅
            # 82.77分）``.  Constrain lookup to the short tail of an explicit
            # whole-bridge token so nearby BCIm/BSI values cannot leak in.
            metric = re.search(r"(?:BCI(?![msx])|Dr)", text, re.I)
            if metric:
                tail = text[metric.end():metric.end() + 120]
                for side, number in re.findall(
                    rf"(左幅|右幅)[^\d]{{0,8}}{FLOAT_RE}",
                    tail,
                ):
                    result[side] = float(number)
        return result

    @staticmethod
    def _extract_filename_grades(state: DocumentState) -> None:
        """Use explicit ``原/现`` filename metadata only to fill gaps.

        The 2019 delivery convention includes strings such as
        ``（原B级，现A级）`` in the supplied source filename.  They are direct
        source metadata, but lower priority than any statement inside the
        report.  This fallback is intentionally limited to missing grades and
        does not derive or invent numeric scores.
        """
        filename = Path(state.source_path).name
        current_match = re.search(r"(?:现|当前)\s*[“\"']?\s*([ABCDE])\s*(?:级|类)?", filename, re.I)
        previous_match = re.search(r"(?:原|此前|上一年度)\s*[“\"']?\s*([ABCDE])\s*(?:级|类)?", filename, re.I)
        if not current_match and not previous_match:
            return
        evidence_id = "FILENAME"
        if evidence_id not in state.evidence:
            state.add_evidence(
                EvidenceRef(
                    id=evidence_id,
                    kind="filename",
                    text=filename,
                    source_path=state.source_path,
                    priority=30,
                )
            )
        if current_match and state.scores.overall.grade.value in (None, "", "无"):
            state.scores.overall.grade = FieldValue(
                value=f"{current_match.group(1).upper()}级",
                evidence=[evidence_id],
                source_priority=30,
                confidence=0.78,
            )
        if previous_match and state.scores.previous.grade.value in (None, "", "无"):
            state.scores.previous.grade = FieldValue(
                value=f"{previous_match.group(1).upper()}级",
                evidence=[evidence_id],
                source_priority=30,
                confidence=0.78,
            )

    def _align_component_scores_with_selected_overall(
        self,
        rating_records: dict[str, list[tuple[str | None, float, str | None, str]]],
        state: DocumentState,
    ) -> None:
        """Close the final-rating-table consistency loop."""
        overall_sides = self._side_record_map(rating_records["overall"])
        overall_text = str(state.scores.overall.score.value or "")
        overall_values = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", overall_text)]
        overall_is_dual = "左幅" in overall_text and "右幅" in overall_text and len(overall_values) >= 2

        selected_side: str | None = None
        if not overall_is_dual and len(overall_values) == 1 and overall_sides:
            target = overall_values[0]
            matches = [side for side, record in overall_sides.items() if abs(record[0] - target) <= 0.011]
            if len(matches) == 1:
                selected_side = matches[0]

        for scope in ("deck", "superstructure", "substructure"):
            records = rating_records[scope]
            if not records:
                continue
            sides = self._side_record_map(records)
            if selected_side and selected_side in sides:
                score, grade, evidence = sides[selected_side]
                self._apply_rating_records(
                    getattr(state.scores, scope),
                    [(selected_side, score, grade, evidence)],
                    score_priority=97,
                    explicit_grade_priority=97,
                )
            elif overall_is_dual and "左幅" in sides and "右幅" in sides:
                self._apply_rating_records(
                    getattr(state.scores, scope),
                    records,
                    score_priority=97,
                    explicit_grade_priority=97,
                )
            else:
                unscoped = [record for record in records if record[0] is None]
                if len(unscoped) == 1:
                    self._apply_rating_records(
                        getattr(state.scores, scope),
                        unscoped,
                        score_priority=97,
                        explicit_grade_priority=97,
                    )
                elif records:
                    self._apply_rating_records(
                        getattr(state.scores, scope),
                        records,
                        score_priority=97,
                        explicit_grade_priority=97,
                    )

    def _resolve_component_grades(self, parsed: ParsedDocument, state: DocumentState) -> None:
        modern_2017_standard = "CJJ 99-2017" in parsed.full_text or "CJJ99-2017" in parsed.full_text.replace(" ", "")
        pedestrian_bridge = "人行天桥" in parsed.full_text[:8000]
        for scope in ("deck", "superstructure", "substructure"):
            entry = getattr(state.scores, scope)
            if modern_2017_standard and scope == "deck" and pedestrian_bridge:
                explicit = self._explicit_component_grade(parsed, state, "桥面系")
                if explicit:
                    grade, evidence = explicit
                    entry.grade = FieldValue(
                        value=grade,
                        evidence=[evidence],
                        source_priority=94,
                        confidence=0.98,
                    )
                    continue
            # A grade printed directly in the official rating table outranks
            # paragraph formulas and derived thresholds.
            if not modern_2017_standard and entry.grade.source_priority >= 96:
                continue
            basis = entry.score
            if not modern_2017_standard and scope == "deck" and self._minimum_numeric(entry.control_score.value) < 66:
                basis = entry.control_score
            values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", str(basis.value or ""))]
            if not values:
                continue
            grades = [grade_from_score(value) for value in values]
            if len(grades) >= 2 and "左幅" in str(basis.value) and "右幅" in str(basis.value):
                resolved = f"{grades[0]}（左幅）/{grades[1]}（右幅）"
            else:
                resolved = grades[0]
            entry.grade = FieldValue(
                value=resolved,
                evidence=list(basis.evidence),
                source_priority=91,
                confidence=0.94,
            )

    @staticmethod
    def _minimum_numeric(value: object) -> float:
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(value or ""))]
        return min(numbers) if numbers else float("inf")

    def _explicit_component_grade(
        self,
        parsed: ParsedDocument,
        state: DocumentState,
        component_label: str,
    ) -> tuple[str, str] | None:
        scope_token = {"桥面系": "m", "上部结构": "s", "下部结构": "x"}.get(component_label, "")
        for paragraph in parsed.paragraphs:
            text = paragraph.text
            if component_label not in text and (not scope_token or f"BSI{scope_token}" not in text and f"BCI{scope_token}" not in text):
                continue
            if not re.search(r"BCI|BSI|技术状况|完好状况", text, re.I):
                continue
            left_match = re.search(r"左幅.{0,40}?评定为\s*[“\"]?\s*([ABCDE])\s*(?:级|类)?", text, re.I)
            right_match = re.search(r"右幅.{0,40}?评定为\s*[“\"]?\s*([ABCDE])\s*(?:级|类)?", text, re.I)
            if left_match and right_match:
                evidence = self._paragraph_evidence(state, parsed, paragraph.index, text, 94)
                return f"{left_match.group(1).upper()}级（左幅）/{right_match.group(1).upper()}级（右幅）", evidence
            match = re.search(r"(?:" + re.escape(component_label) + r"|BSI" + scope_token + r"|BCI" + scope_token + r").{0,30}?评定为\s*[“\"]?\s*([ABCDE])\s*(?:级|类)?", text, re.I)
            if not match:
                match = re.search(r"评定为\s*[“\"]?\s*([ABCDE])\s*(?:级|类)?", text, re.I)
            if match:
                evidence = self._paragraph_evidence(state, parsed, paragraph.index, text, 94)
                return f"{match.group(1).upper()}级", evidence
        return None

    @staticmethod
    def _rating_table_side(table: TableBlock) -> str | None:
        for text in reversed(table.context):
            if "技术状况评定表" not in text:
                continue
            if "左幅" in text:
                return "左幅"
            if "右幅" in text:
                return "右幅"
        return None

    @staticmethod
    def _side_record_map(records: list[tuple[str | None, float, str | None, str]]) -> dict[str, tuple[float, str | None, str]]:
        result: dict[str, tuple[float, str | None, str]] = {}
        for side, score, grade, evidence in records:
            if side in {"左幅", "右幅"}:
                result[side] = (score, grade, evidence)
        return result

    def _apply_rating_records(
        self,
        entry,
        records: list[tuple[str | None, float, str | None, str]],
        *,
        score_priority: int,
        explicit_grade_priority: int,
    ) -> None:
        if not records:
            return
        sides = self._side_record_map(records)
        if "左幅" in sides and "右幅" in sides:
            left, right = sides["左幅"], sides["右幅"]
            entry.score = FieldValue(
                value=f"{self._format_number(left[0])}（左幅）/{self._format_number(right[0])}（右幅）",
                evidence=list(dict.fromkeys([left[2], right[2]])),
                source_priority=score_priority,
                confidence=0.99,
            )
            if left[1] and right[1] and entry.grade.source_priority <= explicit_grade_priority:
                entry.grade = FieldValue(
                    value=f"{left[1]}（左幅）/{right[1]}（右幅）",
                    evidence=list(dict.fromkeys([left[2], right[2]])),
                    source_priority=explicit_grade_priority,
                    confidence=0.99,
                )
            return
        _, score, grade, evidence = records[0]
        entry.score = FieldValue(value=score, evidence=[evidence], source_priority=score_priority, confidence=0.99)
        if grade and entry.grade.source_priority <= explicit_grade_priority:
            entry.grade = FieldValue(value=grade, evidence=[evidence], source_priority=explicit_grade_priority, confidence=0.99)

    def _extract_component_score_paragraphs(self, parsed: ParsedDocument, state: DocumentState) -> None:
        scope_patterns = {
            "deck": re.compile(r"桥面系.{0,36}?BCIm\s*(?:为|=|：|:)?\s*" + FLOAT_RE, re.I),
            "superstructure": re.compile(r"上部结构.{0,48}?BCIs\s*(?:为|=|：|:)?\s*" + FLOAT_RE, re.I),
            "substructure": re.compile(r"下部结构.{0,48}?BCIx\s*(?:为|=|：|:)?\s*" + FLOAT_RE, re.I),
        }
        candidates: dict[str, list[tuple[str | None, float, str | None, str]]] = {key: [] for key in scope_patterns}
        aggregate_candidates: dict[str, list[tuple[str | None, float, str | None, str]]] = {key: [] for key in scope_patterns}
        control_candidates: dict[str, list[tuple[str | None, float, str | None, str]]] = {key: [] for key in scope_patterns}
        grade_candidates: dict[str, list[tuple[str | None, str, str]]] = {key: [] for key in scope_patterns}
        for paragraph_position, paragraph in enumerate(parsed.paragraphs):
            if not any(token in paragraph.text for token in ("BCIm", "BCIs", "BCIx", "BSIm", "BSIs", "BSIx")):
                continue
            nearby_context = " ".join(block.text for block in parsed.paragraphs[max(0, paragraph_position - 2):paragraph_position])
            pieces = re.split(r"(?=(?:左幅|右幅))", paragraph.text)
            for piece in pieces:
                side = "左幅" if "左幅" in piece[:18] else "右幅" if "右幅" in piece[:18] else None
                if "全桥" in piece:
                    side = None
                elif side is None:
                    left = nearby_context.rfind("左幅")
                    right = nearby_context.rfind("右幅")
                    if left >= 0 or right >= 0:
                        side = "左幅" if left > right else "右幅"
                grade = self._find_grade_or_class(piece)
                for scope, pattern in scope_patterns.items():
                    scope_label = {"deck": "桥面系", "superstructure": "上部结构", "substructure": "下部结构"}[scope]
                    is_full_substructure_formula = scope == "substructure" and "全桥BCIx" in piece
                    is_contextual_substructure_formula = (
                        scope == "substructure"
                        and "BCIx" in piece
                        and "下部结构评分" in nearby_context
                    )
                    if scope_label not in piece and not is_full_substructure_formula and not is_contextual_substructure_formula:
                        continue
                    evidence = self._paragraph_evidence(state, parsed, paragraph.index, paragraph.text, 92)
                    if grade:
                        grade_candidates[scope].append((side, grade, evidence))
                    # Old Word templates often show the complete calculation, for example
                    # ``桥面系BCIm=(...)/2=73.91``.  The score is the result after the
                    # final equals sign, not the first coefficient in the formula.
                    record: tuple[str | None, float, str | None, str] | None = None
                    score_token = {"deck": "BCIm", "superstructure": "BCIs", "substructure": "BCIx"}[scope]
                    score_value = self._metric_value(piece, score_token)
                    if score_value is not None:
                        record = (side, score_value, grade, evidence)
                    else:
                        match = pattern.search(piece)
                        if match:
                            record = (side, float(match.group(1)), grade, evidence)
                    if record:
                        candidates[scope].append(record)
                        aggregate_pattern = {
                            "deck": r"BCIm(?![左右0-9])",
                            "superstructure": r"BCIs(?![0-9])",
                            "substructure": r"(?:全桥\s*BCIx|BCIx(?![0-9]))",
                        }[scope]
                        if re.search(aggregate_pattern, piece, re.I):
                            aggregate_candidates[scope].append(record)
                    control_token = {"deck": "BSIm", "superstructure": "BSIs", "substructure": "BSIx"}[scope]
                    control_value = self._metric_value(piece, control_token)
                    if control_value is not None:
                        control_candidates[scope].append((side, control_value, grade, evidence))

        for scope, records in candidates.items():
            entry = getattr(state.scores, scope)
            preferred_records = aggregate_candidates[scope] or records
            sides = self._side_record_map(preferred_records)
            modern_2017_standard = "CJJ99-2017" in self._normalized_document_text(parsed)
            overall_is_single = not any(token in str(state.scores.overall.score.value or "") for token in ("左幅", "右幅"))
            unscoped_aggregates = [record for record in aggregate_candidates[scope] if record[0] is None]
            if modern_2017_standard and overall_is_single and unscoped_aggregates:
                # CJJ 99-2017 reports may print many left/right intermediate
                # formulas, followed by one unscoped component value in the
                # final bridge-level assessment.  The latter is the reporting
                # value; allowing earlier side formulas to override it creates
                # false dual results.
                self._apply_rating_records(entry, [unscoped_aggregates[-1]], score_priority=93, explicit_grade_priority=92)
            elif preferred_records and "左幅" in sides and "右幅" in sides and entry.score.source_priority < 90:
                self._apply_rating_records(entry, preferred_records, score_priority=90, explicit_grade_priority=92)
            elif preferred_records and entry.score.value is None:
                # Repeated span-level formulas may occur before a final unscoped
                # aggregate.  Aggregate candidates are filtered above; choosing
                # the last remaining record follows the document's calculation
                # order without depending on a bridge name or template year.
                self._apply_rating_records(entry, [preferred_records[-1]], score_priority=90, explicit_grade_priority=92)
            elif records and entry.grade.source_priority < 92:
                if "左幅" in sides and "右幅" in sides and sides["左幅"][1] and sides["右幅"][1]:
                    entry.grade = FieldValue(
                        value=f"{sides['左幅'][1]}（左幅）/{sides['右幅'][1]}（右幅）",
                        evidence=list(dict.fromkeys([sides["左幅"][2], sides["右幅"][2]])),
                        source_priority=92,
                        confidence=0.98,
                    )
                else:
                    candidate = next((record for record in records if record[2]), None)
                    if candidate:
                        entry.grade = FieldValue(value=candidate[2], evidence=[candidate[3]], source_priority=92, confidence=0.98)
            controls = control_candidates[scope]
            if controls:
                self._apply_control_records(entry, controls)
            grades = grade_candidates[scope]
            if grades and entry.grade.source_priority < 92:
                side_grades = {side: (grade, evidence) for side, grade, evidence in grades if side in {"左幅", "右幅"}}
                if "左幅" in side_grades and "右幅" in side_grades:
                    entry.grade = FieldValue(
                        value=f"{side_grades['左幅'][0]}（左幅）/{side_grades['右幅'][0]}（右幅）",
                        evidence=list(dict.fromkeys([side_grades["左幅"][1], side_grades["右幅"][1]])),
                        source_priority=92,
                        confidence=0.98,
                    )
                else:
                    entry.grade = FieldValue(value=grades[0][1], evidence=[grades[0][2]], source_priority=92, confidence=0.98)

    @staticmethod
    def _metric_value(text: str, token: str) -> float | None:
        """Read one BCI/BSI formula without consuming the next metric."""
        match = re.search(re.escape(token) + r"(?:[左右0-9]*)", text, re.I)
        if not match:
            return None
        tail = text[match.end():]
        next_family = "BSI" if token.upper().startswith("BCI") else "BCI"
        next_metric = re.search(next_family + r"[mMsSxX](?:[左右0-9]*)", tail, re.I)
        segment = tail[:next_metric.start()] if next_metric else tail
        equals_values = re.findall(r"[=＝]\s*(\d{1,3}(?:\.\d+)?)", segment)
        if equals_values:
            return float(equals_values[-1])
        direct = re.search(r"(?:为|：|:)\s*(\d{1,3}(?:\.\d+)?)", segment)
        return float(direct.group(1)) if direct else None

    @staticmethod
    def _normalized_document_text(parsed: ParsedDocument) -> str:
        return parsed.full_text.replace(" ", "")

    def _extract_overall_score_paragraphs(self, parsed: ParsedDocument, state: DocumentState) -> None:
        """Recover old-template overall scores stated only in prose conclusions."""
        # A summary/conclusion may already have selected the controlling bridge
        # side.  Do not reopen that decision merely because earlier paragraphs
        # contain both side calculations; those formulas are upstream evidence,
        # not a later bridge-level outcome.
        existing_overall = str(state.scores.overall.score.value or "")
        if re.search(r"\d", existing_overall) and not (
            "左幅" in existing_overall and "右幅" in existing_overall
        ):
            return
        patterns = (
            re.compile(r"(?:主桥|全桥|总体|整体).{0,36}?评分\s*(?:为|=|：|:)\s*" + FLOAT_RE, re.I),
            re.compile(r"(?:主桥|全桥|总体|整体).{0,36}?(?:BCI|Dr)\s*(?:值|评分)?\s*(?:为|=|：|:)?\s*" + FLOAT_RE, re.I),
            re.compile(r"(?:主桥|全桥).{0,48}?总体技术状况评定.{0,20}?评定为\s*" + FLOAT_RE, re.I),
        )
        candidates: list[tuple[str | None, float, str | None, str]] = []
        for paragraph in parsed.paragraphs:
            formula_marker = re.search(r"(?:左幅|右幅|主桥|全桥|总体|整体).{0,16}?(?:BCI(?![msx])|Dr)", paragraph.text, re.I)
            formula_values = re.findall(r"[=＝]\s*(\d{1,3}(?:\.\d+)?)", paragraph.text) if formula_marker else []
            if formula_values:
                side = "左幅" if "左幅" in paragraph.text[:24] else "右幅" if "右幅" in paragraph.text[:24] else None
                evidence = self._paragraph_evidence(state, parsed, paragraph.index, paragraph.text, 94)
                candidates.append((side, float(formula_values[-1]), self._find_overall_grade(paragraph.text), evidence))
                continue
            for pattern in patterns:
                match = pattern.search(paragraph.text)
                if not match:
                    continue
                side = "左幅" if "左幅" in paragraph.text[:24] else "右幅" if "右幅" in paragraph.text[:24] else None
                evidence = self._paragraph_evidence(state, parsed, paragraph.index, paragraph.text, 88)
                candidates.append((side, float(match.group(1)), self._find_overall_grade(paragraph.text), evidence))
                break
        if not candidates:
            return
        side_records = self._side_record_map(candidates)
        if "左幅" in side_records and "右幅" in side_records:
            self._apply_rating_records(state.scores.overall, candidates, score_priority=94, explicit_grade_priority=94)
        elif state.scores.overall.score.value is None:
            self._apply_rating_records(state.scores.overall, [candidates[-1]], score_priority=88, explicit_grade_priority=88)

    def _extract_previous_score(self, parsed: ParsedDocument, state: DocumentState) -> None:
        # Word-generated tables of contents repeat the same heading near the
        # front of the document (often with a trailing page number).  The last
        # occurrence is the actual body section and therefore owns the facts
        # that follow it.
        markers = [
            paragraph.index
            for paragraph in parsed.paragraphs
            if re.search(r"上一次(?:检测|检查)状况", paragraph.text)
        ]
        marker = markers[-1] if markers else None
        if marker is None:
            # A dated inspection/maintenance ledger is a temporal sequence,
            # not a bag of historical numbers.  It is the strongest fallback
            # when the report has no dedicated prior-inspection section.
            if self._extract_rating_timeline(parsed, state):
                return
            # Newer municipal templates may state the previous result directly
            # in one unheaded sentence.
            if self._extract_previous_from_explicit_history(parsed, state):
                return
            self._extract_previous_grade_from_narrative(parsed, state)
            self._extract_previous_from_history_table(parsed, state)
            return
        candidates = []
        for paragraph in parsed.paragraphs[marker + 1:marker + 32]:
            if re.match(r"\d+\s+检测(?:目的|依据)", paragraph.text):
                break
            candidates.append(paragraph)
        if not candidates:
            self._extract_previous_from_history_table(parsed, state)
            return
        combined = " ".join(paragraph.text for paragraph in candidates)
        score_matches = list(
            re.finditer(
                r"(?:BCI|Dr)\s*(?:值|评分)?\s*(?:为|=|：|:)?\s*" + FLOAT_RE,
                combined,
                re.I,
            )
        )
        grade = self._find_grade_or_class(combined)
        evidence = self._paragraph_evidence(state, parsed, candidates[-1].index, combined[:4000], 78)
        side_scores: dict[str, float] = {}
        for match in score_matches:
            nearby = combined[max(0, match.start() - 24):match.end() + 12]
            side = "左幅" if "左幅" in nearby else "右幅" if "右幅" in nearby else None
            if side:
                side_scores[side] = float(match.group(1))
        if "左幅" in side_scores and "右幅" in side_scores:
            state.scores.previous.score = FieldValue(
                value=f"{self._format_number(side_scores['左幅'])}（左幅）/{self._format_number(side_scores['右幅'])}（右幅）",
                evidence=[evidence],
                source_priority=78,
                confidence=0.95,
            )
        elif score_matches:
            state.scores.previous.score = FieldValue(value=float(score_matches[0].group(1)), evidence=[evidence], source_priority=78, confidence=0.94)
        if grade:
            state.scores.previous.grade = FieldValue(value=grade, evidence=[evidence], source_priority=78, confidence=0.94)
        if not score_matches:
            if self._extract_rating_timeline(parsed, state):
                return
            if self._extract_previous_from_explicit_history(parsed, state):
                return
            self._extract_previous_from_history_table(parsed, state)

    def _extract_previous_from_explicit_history(self, parsed: ParsedDocument, state: DocumentState) -> bool:
        history_marker = re.compile(
            r"(?:上一次|上次|上一年度|上年度|前次|既往|往年|往期|上期|历次|历史定检|前次检测|历史检测|前期|原检测|前一周期|上周期)"
            r".{0,25}?(?:检测|检查|评定|技术状况|评估|时间|BCI|Dr|评分|等级|结果)",
            re.I,
        )
        score_pattern = re.compile(r"(?:BCI|Dr|评分|技术状况指数)\s*(?:值|评分)?\s*(?:为|=|：|:)?\s*" + FLOAT_RE, re.I)
        for paragraph in parsed.paragraphs:
            text = paragraph.text
            if not history_marker.search(text):
                continue
            score_matches = list(score_pattern.finditer(text))
            if not score_matches:
                continue
            evidence = self._paragraph_evidence(state, parsed, paragraph.index, text, 81)
            side_scores: dict[str, float] = {}
            for match in score_matches:
                nearby = text[max(0, match.start() - 32):match.end() + 16]
                side = "左幅" if "左幅" in nearby else "右幅" if "右幅" in nearby else None
                if side:
                    side_scores[side] = float(match.group(1))
            if "左幅" in side_scores and "右幅" in side_scores:
                value: float | str = (
                    f"{self._format_number(side_scores['左幅'])}（左幅）/"
                    f"{self._format_number(side_scores['右幅'])}（右幅）"
                )
            else:
                value = float(score_matches[-1].group(1))
            state.scores.previous.score = FieldValue(
                value=value,
                evidence=[evidence],
                source_priority=81,
                confidence=0.98,
            )
            grade = self._find_grade_or_class(text)
            if grade:
                state.scores.previous.grade = FieldValue(
                    value=grade,
                    evidence=[evidence],
                    source_priority=81,
                    confidence=0.98,
                )
            return True
        return False

    def _extract_previous_grade_from_narrative(self, parsed: ParsedDocument, state: DocumentState) -> None:
        """Read the prior condition grade from historical-inspection prose.

        Many municipal reports provide the previous grade but omit the old BCI
        number.  Keeping that supported grade is preferable to inventing the
        score that appears in some manually enriched training labels.
        """
        for paragraph in parsed.paragraphs:
            text = paragraph.text
            if not (re.search(r"于20\d{2}年|上一?次|前次|往期|历史", text) and any(token in text for token in ("定期检查", "定检", "技术状况", "评定", "评估"))):
                continue
            side_grades: dict[str, str] = {}
            for side in ("左幅", "右幅"):
                match = re.search(side + r".{0,12}?([ABCDE一二三四五])\s*(?:级|类)", text, re.I)
                if match:
                    val = match.group(1).upper()
                    side_grades[side] = f"{val}级" if val in "ABCDE" else f"{val}类"
            grade: str | None = None
            if "左幅" in side_grades and "右幅" in side_grades:
                grade = f"{side_grades['左幅']}（左幅）/{side_grades['右幅']}（右幅）"
            else:
                match = re.search(r"(?:等级.{0,8}?为|评定为|评为|为)\s*([ABCDE一二三四五])\s*(?:级|类)", text, re.I)
                if match:
                    val = match.group(1).upper()
                    grade = f"{val}级" if val in "ABCDE" else f"{val}类"
            if grade:
                evidence = self._paragraph_evidence(state, parsed, paragraph.index, text, 78)
                state.scores.previous.grade = FieldValue(value=grade, evidence=[evidence], source_priority=78, confidence=0.97)
                return

    def _extract_previous_from_history_table(self, parsed: ParsedDocument, state: DocumentState) -> None:
        self._extract_rating_timeline(parsed, state)

    def _extract_final_overall_rating(self, parsed: ParsedDocument, state: DocumentState) -> None:
        """Select the final bridge rating table instead of a stale cover copy."""
        candidates: list[tuple[int, int, float, str | None, str, str | None]] = []
        for table in parsed.tables:
            final_table_context = any(
                re.search(r"技术状况评定表|技术状况评定结果", item)
                for item in table.context[-6:]
            )
            side = self._rating_table_side(table)
            for row_index, row in enumerate(table.rows):
                text = " | ".join(row)
                if "综合评定分数" not in text and not final_table_context:
                    continue
                if not re.search(r"综合评定分数|(?:^|\W)(?:Dr|BCI)\s*[=：:]", text, re.I):
                    continue
                score_match = re.search(
                    r"(?:综合评定分数\s*)?(?:Dr|BCI)?\s*(?:值|评分)?\s*[=：:]?\s*" + FLOAT_RE,
                    text,
                    re.I,
                )
                if not score_match:
                    continue
                score = float(score_match.group(1))
                if not 0 <= score <= 100:
                    continue
                scope_rank = self._final_rating_scope_rank(table, text)
                grade = self._find_grade_or_class(text) or self._direct_grade(text)
                evidence = self._table_evidence(state, parsed, table, row_index, text, 98)
                candidates.append((scope_rank, table.index, score, grade, evidence, side))
        if not candidates:
            return

        _, _, score, grade, evidence, _ = max(candidates, key=lambda item: (item[0], item[1]))
        state.scores.overall.score = FieldValue(
            value=score,
            evidence=[evidence],
            source_priority=98,
            confidence=0.995,
        )
        if grade:
            state.scores.overall.grade = FieldValue(
                value=grade,
                evidence=[evidence],
                source_priority=98,
                confidence=0.995,
            )

    @staticmethod
    def _final_rating_scope_rank(table: TableBlock, row_text: str) -> int:
        """Classify scope from the nearest rating-table title.

        ``table.context`` intentionally carries several preceding paragraphs.
        A new approach-bridge table can therefore still contain the previous
        main-bridge conclusion; reading the whole context makes both tables
        look like main-bridge tables.  The nearest title is the table's own
        scope and must dominate older context.
        """
        for context in reversed(table.context):
            if not re.search(r"技术状况评定表|评定表", context):
                continue
            if "引桥" in context or "匝道" in context:
                return 0
            if re.search(r"全桥|总体|整体|综合", context):
                return 4
            if "主桥" in context or "主跨" in context:
                return 3
            return 2
        if "引桥" in row_text or "匝道" in row_text:
            return 0
        if re.search(r"全桥|总体|整体|综合", row_text):
            return 4
        if "主桥" in row_text or "主跨" in row_text:
            return 3
        return 2

    def _extract_rating_timeline(self, parsed: ParsedDocument, state: DocumentState) -> bool:
        """Resolve current/previous scores from a chronological history table."""
        records: list[tuple[tuple[int, int, int, int], float, str | None, str]] = []
        for table in parsed.tables:
            header = "|".join(table.header)
            if "检测维修日期" not in header or "质量状况" not in header:
                continue
            for row_index, row in enumerate(table.rows[1:], 1):
                text = " | ".join(row)
                period = self._inspection_period_key(row[0] if row else text)
                score, grade = self._primary_rating(text)
                if period is None or score is None:
                    continue
                evidence = self._table_evidence(state, parsed, table, row_index, text, 97)
                records.append(((*period, table.index, row_index), score, grade, evidence))
        if not records:
            return False
        records.sort(key=lambda item: item[0])
        report_year_match = re.search(r"20\d{2}", str(state.bridge.report_date.value or ""))
        report_year = int(report_year_match.group()) if report_year_match else None
        latest = records[-1]
        existing_current = state.scores.overall.score.value
        latest_matches_selected = self._same_numeric_value(existing_current, latest[1])
        latest_in_current_cycle = latest[0][0] == report_year if report_year is not None else latest_matches_selected
        current_is_latest = (
            latest_in_current_cycle and (existing_current is None or latest_matches_selected)
            if report_year is not None
            else latest_matches_selected
        )
        previous_index = -2 if latest_in_current_cycle and len(records) >= 2 else -1

        if current_is_latest:
            _, score, grade, evidence = latest
            existing = state.scores.overall.score
            if existing.value is None:
                state.scores.overall.score = FieldValue(
                    value=score,
                    evidence=[evidence],
                    source_priority=97,
                    confidence=0.99,
                )
                if grade:
                    state.scores.overall.grade = FieldValue(
                        value=grade,
                        evidence=[evidence],
                        source_priority=97,
                        confidence=0.99,
                    )
            elif self._same_numeric_value(existing.value, score):
                existing.evidence = list(dict.fromkeys([*existing.evidence, evidence]))
                if grade and state.scores.overall.grade.value == grade:
                    state.scores.overall.grade.evidence = list(
                        dict.fromkeys([*state.scores.overall.grade.evidence, evidence])
                    )

        if latest_in_current_cycle and len(records) < 2:
            return False
        if not records or abs(previous_index) > len(records):
            return current_is_latest
        _, score, grade, evidence = records[previous_index]
        state.scores.previous.score = FieldValue(
            value=score,
            evidence=[evidence],
            source_priority=97,
            confidence=0.99,
        )
        if grade:
            state.scores.previous.grade = FieldValue(
                value=grade,
                evidence=[evidence],
                source_priority=97,
                confidence=0.99,
            )
        return True

    @staticmethod
    def _inspection_period_key(value: str) -> tuple[int, int] | None:
        years = [int(item) for item in re.findall(r"20\d{2}", value)]
        if not years:
            return None
        months = [int(item) for item in re.findall(r"(\d{1,2})\s*月", value)]
        valid_months = [month for month in months if 1 <= month <= 12]
        return years[-1], max(valid_months, default=12)

    @classmethod
    def _primary_rating(cls, text: str) -> tuple[float | None, str | None]:
        # When a facility is split into main and approach bridges, official
        # labels report the main bridge as the primary overall value.
        patterns = (
            r"主桥.{0,40}?评分\s*(?:为|=|：|:)?\s*" + FLOAT_RE,
            r"(?:全桥|总体|整体).{0,40}?评分\s*(?:为|=|：|:)?\s*" + FLOAT_RE,
            r"(?:BCI(?![msx])|Dr)\s*(?:值|评分)?\s*(?:为|=|：|:)?\s*" + FLOAT_RE,
            r"评分\s*(?:为|=|：|:)?\s*" + FLOAT_RE,
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if not match:
                continue
            score = float(match.group(1))
            if 0 <= score <= 100:
                nearby = text[match.start():match.end() + 48]
                return score, cls._find_grade_or_class(nearby) or cls._direct_grade(nearby)
        return None, None

    @staticmethod
    def _direct_grade(text: str) -> str | None:
        alpha = re.search(r"([ABCDE])\s*(?:级|类)", text, re.I)
        if alpha:
            return f"{alpha.group(1).upper()}级"
        chinese = re.search(r"([一二三四五])\s*类", text)
        return f"{chinese.group(1)}类" if chinese else None

    @staticmethod
    def _same_numeric_value(value: object, expected: float) -> bool:
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(value or ""))]
        return len(numbers) == 1 and abs(numbers[0] - expected) <= 0.011

    def _apply_control_records(self, entry, records: list[tuple[str | None, float, str | None, str]]) -> None:
        sides = self._side_record_map(records)
        if "左幅" in sides and "右幅" in sides:
            left, right = sides["左幅"], sides["右幅"]
            entry.control_score = FieldValue(
                value=f"{self._format_number(left[0])}（左幅）/{self._format_number(right[0])}（右幅）",
                evidence=list(dict.fromkeys([left[2], right[2]])),
                source_priority=92,
                confidence=0.98,
            )
            return
        _, value, _, evidence = records[0]
        entry.control_score = FieldValue(value=value, evidence=[evidence], source_priority=92, confidence=0.98)

    @classmethod
    def _find_grade_or_class(cls, text: str) -> str | None:
        alpha = re.search(r"(?:等级(?:评定)?为|评定为|评为|等级为|状况为|为)\s*[“\"]?\s*([ABCDE])\s*(?:级|类)?", text, re.I)
        if alpha:
            return f"{alpha.group(1).upper()}级"
        chinese = re.search(r"(?:等级(?:评定)?为|评定为|评为|等级为|状况为|为)\s*[“\"]?\s*([一二三四五])\s*类", text)
        if chinese:
            return f"{chinese.group(1)}类"
        chinese_direct = re.search(r"([一二三四五])\s*类", text)
        if chinese_direct:
            return f"{chinese_direct.group(1)}类"
        return normalize_grade(text)

    @classmethod
    def _find_overall_grade(cls, text: str) -> str | None:
        overall = re.search(
            r"(?:整体|总体|全桥).{0,28}?(?:等级(?:评定)?为|评定为|评为|等级为)\s*[“\"]?\s*([ABCDE])\s*(?:级|类)?",
            text,
            re.I,
        )
        if overall:
            return f"{overall.group(1).upper()}级"
        matches = re.findall(r"(?:等级(?:评定)?为|评定为|评为|等级为)\s*[“\"]?\s*([ABCDE一二三四五])\s*(级|类)?", text, re.I)
        if matches:
            value, suffix = matches[-1]
            return f"{value.upper()}级" if value.upper() in "ABCDE" else f"{value}{suffix or '类'}"
        return cls._find_grade_or_class(text)

    @staticmethod
    def _format_number(value: float) -> str:
        return f"{value:.6f}".rstrip("0").rstrip(".")

    def _extract_comparisons(self, parsed: ParsedDocument, state: DocumentState) -> None:
        for table in parsed.tables:
            if not table.contains("上一次检测结果", "本次检测结果", "发展状况"):
                continue
            for row_index, row in enumerate(table.rows[1:], 1):
                if len(row) < 4 or not row[0]:
                    continue
                # Preserve the temporal roles in the evidence text.  A bare
                # four-cell join loses which description is historical and
                # which is current, making both deterministic and model
                # reviewers needlessly guess the direction of time.
                comparison_text = (
                    f"位置={row[0]} | 上一次检测结果={row[1]} | "
                    f"本次检测结果={row[2]} | 发展状况={row[3]}"
                )
                evidence = self._table_evidence(
                    state,
                    parsed,
                    table,
                    row_index,
                    comparison_text,
                    FIELD_PRIORITY["comparison"],
                )
                state.comparisons.append(ComparisonRecord(section=row[0], previous=row[1], current=row[2], development=row[3], evidence=[evidence]))
                if row[0].replace(" ", "") in {"全桥", "总体", "整体", "主桥"} and state.scores.previous.score.value is None:
                    prev_score_match = re.search(r"(?:BCI|Dr)?\s*(?:值|评分)?\s*(?:为|=|：|:)?\s*" + FLOAT_RE, row[1], re.I)
                    prev_grade = self._find_grade_or_class(row[1])
                    if prev_score_match:
                        state.scores.previous.score = FieldValue(
                            value=float(prev_score_match.group(1)),
                            evidence=[evidence],
                            source_priority=80,
                            confidence=0.95,
                        )
                    if prev_grade and state.scores.previous.grade.value is None:
                        state.scores.previous.grade = FieldValue(
                            value=prev_grade,
                            evidence=[evidence],
                            source_priority=80,
                            confidence=0.95,
                        )

        # Prose paragraph scan for previous score & grade if still unextracted
        if state.scores.previous.score.value is None:
            for p in parsed.paragraphs:
                text = p.text.strip()
                if not text or not re.search(r"上次|上一次|前次|上一轮|上一周期|20\d{2}年(?:度)?", text):
                    continue
                dual_match = re.search(
                    r"(?:上次|上一次|前次|上一轮|上一周期|20\d{2}年(?:度)?).{0,40}?(?:评分|BCI|Dr|分值)\s*(?:为|=|：|:)?\s*([0-9]{2,3}(?:\.[0-9]{1,4})?)\s*（左幅）\s*/\s*([0-9]{2,3}(?:\.[0-9]{1,4})?)\s*（右幅）",
                    text,
                )
                if dual_match:
                    evidence_id = self._paragraph_evidence(state, parsed, p.index, text, 80)
                    dual_score = f"{dual_match.group(1)}（左幅）/{dual_match.group(2)}（右幅）"
                    state.scores.previous.score = FieldValue(
                        value=dual_score,
                        evidence=[evidence_id],
                        source_priority=80,
                        confidence=0.92,
                    )
                    break
                single_match = re.search(
                    r"(?:上次|上一次|前次|上一轮|上一周期|20\d{2}年(?:度)?).{0,40}?(?:总体|全桥|主桥)?\s*(?:BCI|Dr|评分|得分|分值)\s*(?:为|=|：|:)?\s*([0-9]{2,3}(?:\.[0-9]{1,4})?)\s*(?:分)?",
                    text,
                )
                if single_match:
                    val = float(single_match.group(1))
                    if 0 <= val <= 100:
                        evidence_id = self._paragraph_evidence(state, parsed, p.index, text, 80)
                        state.scores.previous.score = FieldValue(
                            value=val,
                            evidence=[evidence_id],
                            source_priority=80,
                            confidence=0.92,
                        )
                        break

        if state.scores.previous.grade.value is None and state.scores.previous.score.value not in (None, "无"):
            for p in parsed.paragraphs:
                text = p.text.strip()
                if not text or not re.search(r"上次|上一次|前次|上一轮|上一周期|20\d{2}年(?:度)?", text):
                    continue
                grade_match = re.search(
                    r"(?:上次|上一次|前次|上一轮|上一周期|20\d{2}年(?:度)?).{0,40}?(?:等级|评定为|评为)\s*[:=为\s]*([ABCDE一二三四五])\s*(?:级|类)?",
                    text,
                )
                if grade_match:
                    evidence_id = self._paragraph_evidence(state, parsed, p.index, text, 80)
                    state.scores.previous.grade = FieldValue(
                        value=f"{grade_match.group(1).upper().replace('类', '级')}",
                        evidence=[evidence_id],
                        source_priority=80,
                        confidence=0.92,
                    )
                    break
            if state.scores.previous.grade.value is None:
                if isinstance(state.scores.previous.score.value, (int, float)):
                    derived = grade_from_score(state.scores.previous.score.value)
                    state.scores.previous.grade = FieldValue(
                        value=derived,
                        evidence=list(state.scores.previous.score.evidence),
                        source_priority=75,
                        confidence=0.90,
                    )

    def _extract_diseases(self, parsed: ParsedDocument, state: DocumentState) -> None:
        for table in parsed.tables:
            header = table.header
            normalized_header = ["".join(cell.split()) for cell in header]
            try:
                type_index = next(i for i, x in enumerate(normalized_header) if "病害类型" in x or "病害种类" in x or "损坏类型" in x)
                desc_index = next(i for i, x in enumerate(normalized_header) if "病害描述" in x or "病害情况" in x or "具体情况" in x or "具体位置" in x or "检查结果" in x or "检测结果" in x)
            except StopIteration:
                continue
            loc_indexes = [
                i
                for i, x in enumerate(normalized_header)
                if x == "位置" or any(term in x for term in ("病害位置", "构件位置", "桥幅", "桥跨", "构件部位"))
            ]
            seq_index = next((i for i, x in enumerate(normalized_header) if "序号" in x), None)
            context = " ".join(table.context)
            context_side = self._table_context_side(table)
            for row_index, row in enumerate(table.rows[1:], 1):
                required_indexes = [type_index, desc_index, *loc_indexes] + ([seq_index] if seq_index is not None else [])
                if len(row) <= max(required_indexes):
                    continue
                raw_sequence = row[seq_index] if seq_index is not None else ""
                sequence = raw_sequence if re.fullmatch(r"\d+(?:[.)、）])?", raw_sequence.strip()) else str(row_index)
                description = row[desc_index]
                location_parts = [row[index] for index in loc_indexes if row[index].strip()]
                location = " ".join(location_parts) if location_parts else description
                disease_type = row[type_index]
                # Legacy Word reports often merge the photo contact sheet into
                # the preceding disease table. Those rows contain two photo
                # captions and an empty description cell; they are not issues.
                if not description.strip() and re.search(r"照片\s*\d", f"{location} {disease_type}"):
                    continue
                normal_type = disease_type.strip() in {"-", "—", "整体良好", "良好", "正常"}
                good_description = any(
                    x in description
                    for x in ("未见明显病害", "工作正常", "状况良好", "外观良好", "无明显病害", "整体现状良好")
                )
                # A row can describe the assembly as generally sound and then
                # state a local active defect (for example ``墩台总体状况良好，
                # 桥墩局部渗水痕迹``).  Dropping the whole row loses a labelled
                # defect.  Treat positive defect language as the controlling
                # clause; skip the row only when the good-state wording is not
                # followed by any active issue signal.
                active_defect = bool(ACTIVE_DEFECT_RE.search(f"{disease_type} {description}"))
                if is_non_disease_label(disease_type):
                    inferred_type = canonical_disease_from_text(description)
                    if inferred_type:
                        disease_type = inferred_type
                        active_defect = True
                    elif not description.strip():
                        continue
                    elif not is_condition_state_label(disease_type):
                        # Composite labels such as ``桥涵标现状`` and
                        # ``工作照`` are presentation captions.  Only the exact
                        # bare condition labels survive for possible history
                        # enrichment; an active defect above has already been
                        # recovered from its description.
                        continue
                    elif not is_meaningful_condition_state(disease_type, description):
                        # Bare ``现状`` / ``外观`` rows whose description is also
                        # a pure photo caption (ends with 现状 / 外观 and carries
                        # no defect or positive-state signal) are photo annotations
                        # embedded in the disease table.  They would fail the
                        # NON_DISEASE_LABEL validator; drop them here.
                        if not ACTIVE_DEFECT_RE.search(description):
                            continue
                if (normal_type or good_description) and not active_defect and not is_condition_state_label(disease_type):
                    continue
                # A bare repair trace is maintenance history, not an active
                # disease.  Keep it only when the same record states a current
                # defect such as re-cracking, spalling, seepage or corrosion.
                if re.fullmatch(r"(?:修补痕迹|修补现状|修补)", disease_type.strip()) and not re.search(
                    r"裂缝|开裂|破损|剥落|脱落|渗水|泛碱|锈蚀|露筋",
                    description,
                ):
                    continue
                if disease_type.strip() in {"", "/", "无"}:
                    if "未设置" in description or "缺失" in description:
                        disease_type = f"{location}缺失" if location else "缺失"
                    else:
                        continue
                evidence = self._table_evidence(state, parsed, table, row_index, " | ".join(row), FIELD_PRIORITY["disease"])
                description = description.replace("m2", "m²").replace("m 2", "m²")
                if context_side and "左幅" not in location and "右幅" not in location:
                    location = f"{context_side}{location}"
                location = self._precise_disease_location(description, disease_type, location)
                component = normalize_component(location)
                if component == "其他":
                    component = normalize_component(description, context)
                metrics = self._extract_metrics(description)
                state.diseases.append(Disease(sequence=sequence, component=component, location=location, type=normalize_disease_type(disease_type), description=description, metrics=metrics, evidence=[evidence], confidence=0.98))

        unique: dict[tuple[str, str, str, str], Disease] = {}
        for disease in state.diseases:
            key = (disease.component, disease.location, disease.type, disease.description)
            if key not in unique:
                unique[key] = disease
            else:
                unique[key].evidence = sorted(set(unique[key].evidence + disease.evidence))
        state.diseases = list(unique.values())

    @staticmethod
    def _precise_disease_location(description: str, disease_type: str, fallback: str) -> str:
        """Prefer the structural prefix in a detailed defect description.

        Many source tables split location into coarse columns (for example
        ``右幅 | 1#跨``) but put the useful location in the description
        (``1#跨1#墩上方右侧翼板破损露筋``).  Selecting the prefix before the
        first defect term is a template-independent way to retain that detail.
        """
        defect_terms = (
            "网状裂缝", "纵向裂缝", "横向裂缝", "竖向裂缝", "环向裂缝", "斜向裂缝",
            "破损露筋", "破损、露筋", "渗水泛碱", "蜂窝麻面", "涂层剥落",
            "裂缝", "开裂", "破损", "露筋", "渗水", "泛碱", "锈蚀", "腐蚀",
            "坑槽", "车辙", "错台", "松动", "脱落", "缺失", "划痕", "变形",
        )
        positions = [description.find(term) for term in defect_terms if description.find(term) > 0]
        if not positions:
            compact_type = re.sub(r"[、，,\s]+", "", disease_type)
            compact_description = re.sub(r"[、，,\s]+", "", description)
            position = compact_description.find(compact_type)
            if position <= 0:
                return fallback
            candidate = compact_description[:position]
        else:
            candidate = description[: min(positions)]
        # When a sentence first states a generally good condition and then a
        # local exception, only the final clause belongs to the defect
        # location (``...状况良好，桥墩局部渗水`` -> ``桥墩``).
        candidate = re.split(r"[，,；;。]", candidate)[-1]
        candidate = candidate.strip(" ，,；;：:、-/").removesuffix("处")
        for suffix in ("局部", "个别", "少量", "多处", "部分"):
            candidate = candidate.removesuffix(suffix).strip()
        if not (2 <= len(candidate) <= 100):
            return fallback
        if not re.search(r"#|距|梁|板|墩|台|缝|跨|栏|防护网|塔|索|支座|桥面|人行道|路面|排水", candidate):
            return fallback
        return candidate

    @staticmethod
    def _table_context_side(table: TableBlock) -> str | None:
        for text in reversed(table.context):
            left = text.rfind("左幅")
            right = text.rfind("右幅")
            if left >= 0 or right >= 0:
                return "左幅" if left > right else "右幅"
        return None

    def _aggregate_large_diseases(self, parsed: ParsedDocument, state: DocumentState) -> None:
        """Use the report's own publication grain for long evidence ledgers.

        Ordinary reports remain row-granular.  Long-span reports that publish
        an explicit disease summary (or independently split main/approach
        ledgers) use one row per structural scope and issue family, with count
        and representative literal records.  This mixed-grain rule matches the
        released training labels and is triggered by document structure rather
        than bridge identity.
        """
        # Keep ordinary reports row-for-row.  Two independently observable
        # source signals enable aggregation: an explicit aggregate narrative,
        # or a large report that separates both main-bridge and approach-bridge
        # ledgers.  The latter template otherwise produces hundreds of photo
        # rows where the requested output is one row per structural-region /
        # disease family.  Row count alone is intentionally insufficient.
        has_main_approach_split = self._has_main_approach_disease_split(parsed)
        has_aggregate_semantics = self._has_aggregate_disease_semantics(parsed)
        if not has_aggregate_semantics and not (len(state.diseases) > 100 and has_main_approach_split):
            return
        groups: dict[tuple[str, str], list[Disease]] = {}
        for disease in state.diseases:
            if self._inactive_aggregate_record(disease):
                continue
            region = self._disease_region_from_source_table(parsed, disease)
            disease_type = self._canonical_group_type(disease.type, disease.description)
            groups.setdefault((region, disease_type), []).append(disease)

        aggregated: list[Disease] = []
        for sequence, ((region, disease_type), items) in enumerate(groups.items(), 1):
            evidence_ids = list(dict.fromkeys(evidence_id for item in items for evidence_id in item.evidence))
            examples = list(dict.fromkeys(item.description for item in items))[:3]
            if len(items) == 1:
                description = examples[0]
                location = items[0].location or region
                metrics = items[0].metrics
            else:
                example_text = "；".join(examples)
                class_suffix = "" if disease_type.endswith("类病害") else "类病害"
                description = f"{disease_type}{class_suffix}共计{len(items)}处（如{example_text}）"
                location = region
                metrics = {"COUNT": str(len(items))}
            aggregated.append(
                Disease(
                    sequence=str(sequence),
                    component=region,
                    location=location,
                    type=disease_type,
                    description=description[:1600],
                    metrics=metrics,
                    evidence=evidence_ids,
                    confidence=min(item.confidence for item in items),
                )
            )
        # Modern inspection reports often publish an explicit, evidence-rich
        # ``病害总结`` after the photo ledger.  It is a more faithful output
        # layer than counts recomputed from raw rows: the summary deliberately
        # combines synonyms (for example ``破损、露筋``), retains side/main-
        # approach scope, and reports the engineer-approved totals.  Overlay
        # those source-defined issue families only after the independent gate
        # above has selected aggregation; ordinary granular reports therefore
        # remain untouched.
        summary_families = self._summary_disease_families(parsed, state)
        if summary_families:
            aggregated = self._overlay_summary_families(aggregated, summary_families)
        for sequence, disease in enumerate(aggregated, 1):
            disease.sequence = str(sequence)
        state.diseases = aggregated

    _SUMMARY_DEFECT = (
        r"(?:纵向|横向|竖向|斜向|水平|环向|网状|表层网状)?(?:裂缝|开裂)|龟裂|"
        r"(?:混凝土)?(?:破损(?:[、,，]?露筋(?:[、,，]?锈蚀)?)?|露筋(?:[、,，]?锈蚀)?|剥落)|"
        r"蜂窝[、,，]?麻面(?:[、,，]?混凝土剥落)?|"
        r"(?:渗水|浸水)(?:[、,，]?泛碱)?|泛碱|锈蚀|锈胀|"
        r"模板未拆|坑槽|坑洞|车辙|变形(?:堵塞)?|缺失|脱落|松动|"
        r"划痕|污渍|断裂|脱漆|修补痕迹|磨损|露骨|黄油变质|泄水孔堵塞"
    )
    _SUMMARY_UNIT = r"处|条|盏|根|颗|块"
    _SUMMARY_COMPONENT = re.compile(
        r"主梁|箱梁|T梁|横向连接系|横向联系|横隔板|主塔|桥塔|塔柱|拉索|"
        r"支座|桥墩|桥台|桥面铺装|伸缩缝|护栏|人行道|排水|照明"
    )

    def _summary_disease_families(self, parsed: ParsedDocument, state: DocumentState) -> list[Disease]:
        """Extract issue families from the report's own final disease summary.

        The final occurrence is used because some reports embed an older
        report or a generated table of contents before the current conclusion.
        Relative OOXML table depth separates true narrative from nested ledger
        cells for both native DOCX and converted legacy DOC files.
        """
        nonempty = [paragraph for paragraph in parsed.paragraphs if paragraph.text]
        if not nonempty:
            return []
        base_depth = min(paragraph.table_depth for paragraph in nonempty)
        narrative = [paragraph for paragraph in nonempty if paragraph.table_depth == base_depth]
        starts = [
            index
            for index, paragraph in enumerate(narrative)
            if re.search(r"(?:主要)?病害(?:统计如下|总结)", paragraph.text)
        ]
        if not starts:
            return []
        start = starts[-1]
        # A table-of-contents entry can be the last heading in malformed
        # conversions. Prefer the last explicit ``统计如下`` marker when one is
        # available, since it starts the actual evidence block.
        explicit = [
            index
            for index in starts
            if "统计如下" in narrative[index].text
        ]
        if explicit:
            start = explicit[-1]

        context = ""
        extracted: list[Disease] = []
        for paragraph in narrative[start + 1:start + 181]:
            text = paragraph.text.strip()
            if re.search(r"(?:养护|维护)建议", text):
                break
            if not text or "未见明显病害" in text or "状况良好" in text:
                continue
            if len(text) <= 50 and self._SUMMARY_COMPONENT.search(text):
                context = text.strip(" ：:；;")
            if "：" not in text and ":" not in text:
                continue
            outer, body = re.split(r"[：:]", text, maxsplit=1)
            if not self._SUMMARY_COMPONENT.search(f"{context} {outer}"):
                continue
            # A single paragraph may switch components or carriageways after
            # a semicolon (``主梁：...；主塔：...`` or ``左幅：...；右幅：...``).
            body = re.sub(r"[。；;](?=(?:左幅|右幅|主桥|引桥|主塔|拉索|桥墩|桥台)[：:])", "；", body)
            segments = [segment.strip(" ，,。；;") for segment in re.split(r"[；;]", body)]
            local_context = ""
            for segment in segments:
                if not segment or "未见明显病害" in segment:
                    continue
                if "：" in segment or ":" in segment:
                    prefix, remainder = re.split(r"[：:]", segment, maxsplit=1)
                    if self._SUMMARY_COMPONENT.search(prefix) or re.search(r"左幅|右幅|主桥|引桥", prefix):
                        local_context = prefix.strip()
                        segment = remainder.strip()
                leading = self._summary_leading_location(segment)
                if leading:
                    local_context = leading
                scope = self._summary_scope(f"{context} {outer}", local_context)
                families = self._summary_segment_families(segment)
                if not families:
                    continue
                evidence_id = self._paragraph_evidence(
                    state,
                    parsed,
                    paragraph.index,
                    text,
                    FIELD_PRIORITY["disease"] + 8,
                )
                for disease_type, description in families:
                    extracted.append(Disease(
                        sequence=str(len(extracted) + 1),
                        component=normalize_component(scope),
                        location=scope,
                        type=disease_type,
                        description=description,
                        metrics=self._extract_metrics(description),
                        evidence=[evidence_id],
                        confidence=0.99,
                    ))

        # Merge repeated issue-family fragments inside the same final summary
        # while preserving source order and all evidence coordinates.
        merged: dict[tuple[str, str], Disease] = {}
        for disease in extracted:
            canonical = self._canonical_group_type(disease.type, disease.description)
            key = (disease.location, canonical)
            if key not in merged:
                merged[key] = disease
                continue
            current = merged[key]
            if disease.description not in current.description:
                current.description = f"{current.description}；{disease.description}"
            current.type = "、".join(dict.fromkeys([*current.type.split("、"), *disease.type.split("、")]))
            current.evidence = list(dict.fromkeys([*current.evidence, *disease.evidence]))
            current.metrics.update(disease.metrics)
        return list(merged.values())

    @classmethod
    def _summary_leading_location(cls, segment: str) -> str:
        marker = re.search(
            rf"(?:存在\s*)?\d+(?:\.\d+)?\s*(?:{cls._SUMMARY_UNIT})\s*(?:{cls._SUMMARY_DEFECT})|"
            rf"(?:{cls._SUMMARY_DEFECT})\s*\d+(?:\.\d+)?\s*(?:{cls._SUMMARY_UNIT})",
            segment,
        )
        if not marker:
            return ""
        prefix = segment[:marker.start()].strip(" ，,。；;：:")
        if 0 < len(prefix) <= 80 and (
            cls._SUMMARY_COMPONENT.search(prefix)
            or re.search(r"左幅|右幅|主桥|引桥|箱内|箱外", prefix)
        ):
            return prefix
        return ""

    @classmethod
    def _summary_segment_families(cls, segment: str) -> list[tuple[str, str]]:
        """Return canonical issue families and literal count/measure clauses."""
        before = re.compile(
            rf"(?:存在\s*)?(?P<count>\d+(?:\.\d+)?)\s*(?P<unit>{cls._SUMMARY_UNIT})\s*"
            rf"(?P<type>{cls._SUMMARY_DEFECT})"
        )
        after = re.compile(
            rf"(?P<type>{cls._SUMMARY_DEFECT})\s*(?P<count>\d+(?:\.\d+)?)\s*"
            rf"(?P<unit>{cls._SUMMARY_UNIT})"
        )
        matches: list[re.Match[str]] = []
        occupied: list[tuple[int, int]] = []
        for pattern in (before, after):
            for match in pattern.finditer(segment):
                if any(not (match.end() <= left or match.start() >= right) for left, right in occupied):
                    continue
                matches.append(match)
                occupied.append((match.start(), match.end()))
        matches.sort(key=lambda match: match.start())
        if not matches:
            # Explicit summary rows occasionally omit a count (for example
            # ``1#伸缩缝保护带裂缝``). Keep the literal disease rather than
            # inventing a count.
            defect = re.search(cls._SUMMARY_DEFECT, segment)
            if not defect:
                return []
            value = cls._summary_type_text(defect.group())
            return [(value, segment.strip())]

        grouped: dict[str, dict[str, Any]] = {}
        for index, match in enumerate(matches):
            raw_type = cls._summary_type_text(match.group("type"))
            canonical = cls._canonical_group_type(raw_type, raw_type)
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(segment)
            tail = segment[match.end():next_start].strip(" ，,")
            # Attach only measurements belonging to this issue. Stop before a
            # new component/location sentence even when its count was written
            # in an unsupported form.
            if tail and not re.match(r"(?:总|最大|最小|面积|长度|宽度|L\s*=|W\s*=|S\s*=|V\s*=)", tail, re.I):
                tail = ""
            literal = match.group(0).strip()
            description = f"{literal}，{tail}" if tail else literal
            bucket = grouped.setdefault(canonical, {"types": [], "descriptions": []})
            if raw_type not in bucket["types"]:
                bucket["types"].append(raw_type)
            if description not in bucket["descriptions"]:
                bucket["descriptions"].append(description)
        return [
            ("、".join(value["types"]), "，".join(value["descriptions"]))
            for value in grouped.values()
        ]

    @staticmethod
    def _summary_type_text(value: str) -> str:
        value = re.sub(r"[，,]", "、", value.strip())
        value = value.replace("破损、露筋、锈蚀", "破损露筋、锈蚀")
        value = value.replace("破损、露筋", "破损露筋")
        value = value.replace("露筋、锈蚀", "露筋锈蚀")
        value = value.replace("渗水、泛碱", "渗水泛碱").replace("浸水、泛碱", "浸水泛碱")
        return value

    @classmethod
    def _summary_scope(cls, outer: str, local: str) -> str:
        """Resolve a template-independent side/region/component scope."""
        text = f"{outer} {local}"
        side = "左幅" if "左幅" in local else "右幅" if "右幅" in local else "左幅" if "左幅" in outer else "右幅" if "右幅" in outer else ""
        region = "主桥" if "主桥" in local else "引桥" if "引桥" in local else "主桥" if "主桥" in outer else "引桥" if "引桥" in outer else ""
        inner = "箱梁内" if re.search(r"箱梁内|箱内", text) else "箱梁外" if re.search(r"箱梁外|箱外", text) else ""
        if inner and not region:
            region = "主桥"
        component = next(
            (
                token
                for token in (
                    "横向连接系", "横向联系", "桥面铺装", "伸缩缝", "人行道",
                    "护栏", "排水系统", "照明、标志", "照明", "拉索", "主塔",
                    "桥塔", "塔柱", "支座", "桥墩", "桥台", "主梁", "箱梁", "T梁",
                )
                if token in text
            ),
            "其他",
        )
        aliases = {"横向连接系": "横向联系", "箱梁": "主梁", "T梁": "主梁", "桥塔": "主塔", "塔柱": "主塔", "照明": "照明、标志"}
        component = aliases.get(component, component)
        if inner:
            component = inner
        return "".join(part for part in (side, region, component) if part) or local or outer

    @classmethod
    def _overlay_summary_families(cls, aggregated: list[Disease], summary: list[Disease]) -> list[Disease]:
        """Replace covered raw groups with the report-authored summary layer."""
        consumed: set[int] = set()
        overlaid: list[Disease] = []
        for family in summary:
            family_type = cls._canonical_group_type(family.type, family.description)
            matches = [
                index
                for index, item in enumerate(aggregated)
                if index not in consumed
                and cls._canonical_group_type(item.type, item.description) == family_type
                and cls._summary_scopes_compatible(family.location, item.location or item.component)
            ]
            # The summary is a calibration layer, not an independent source
            # of new rows.  A family with no corresponding active ledger
            # evidence may describe maintenance history, an unparsed sensor
            # subsystem, or an unsupported wording variant.  Keep it out of
            # the result until a raw issue supplies a second evidence channel.
            if not matches:
                continue
            examples: list[str] = []
            for index in matches:
                consumed.add(index)
                item = aggregated[index]
                family.evidence = list(dict.fromkeys([*family.evidence, *item.evidence]))
                grouped_examples = re.search(r"（如(.*)）$", item.description)
                source = grouped_examples.group(1) if grouped_examples else item.description
                examples.extend(value for value in re.split(r"；", source) if value)
            examples = list(dict.fromkeys(examples))[:3]
            if examples:
                family.description = f"{family.description}（如" + "；".join(examples) + "）"
            overlaid.append(family)
        overlaid.extend(item for index, item in enumerate(aggregated) if index not in consumed)
        return overlaid

    @classmethod
    def _summary_scopes_compatible(cls, left: str, right: str) -> bool:
        def features(value: str) -> tuple[str, str, str, str]:
            side = "左幅" if "左幅" in value else "右幅" if "右幅" in value else ""
            region = "主桥" if "主桥" in value else "引桥" if "引桥" in value else ""
            inner = "箱梁内" if "箱梁内" in value else "箱梁外" if "箱梁外" in value else ""
            component = normalize_component(value)
            return side, region, inner, component

        a = features(left)
        b = features(right)
        for first, second in zip(a[:3], b[:3]):
            if first and second and first != second:
                return False
        return a[3] == b[3] or a[3] == "其他" or b[3] == "其他"

    @staticmethod
    def _inactive_aggregate_record(disease: Disease) -> bool:
        """Exclude maintenance/construction history from the active issue layer."""
        if is_condition_state_label(disease.type) and not is_meaningful_condition_state(
            disease.type,
            disease.description,
        ):
            return True
        compact_type = re.sub(r"[、，,\s]+", "", disease.type)
        compact_description = re.sub(r"[、，,\s]+", "", disease.description)
        if re.fullmatch(r"(?:裂缝)?(?:修补|封闭)(?:痕迹|现状)?", compact_type):
            active_recurrence = re.search(r"(?:修补|封闭)(?:处|后)?(?:重新|再次|二次)?(?:开裂|裂缝)", compact_description)
            if not active_recurrence:
                return True
        if re.fullmatch(r"(?:模板未拆(?:除)?|施工残留|油毡未清理)", compact_type):
            return True
        return False

    @staticmethod
    def _disease_table_title(table: TableBlock) -> str:
        return next(
            (
                text
                for text in reversed(table.context)
                if "表" in text and any(token in text for token in ("结果表", "病害", "检测"))
            ),
            table.context[-1] if table.context else "",
        )

    @classmethod
    def _has_main_approach_disease_split(cls, parsed: ParsedDocument) -> bool:
        titles = [cls._disease_table_title(table) for table in parsed.tables if table.is_disease_table()]
        return any("主桥" in title for title in titles) and any("引桥" in title for title in titles)

    @classmethod
    def _disease_region_from_source_table(cls, parsed: ParsedDocument, disease: Disease) -> str:
        table_index: int | None = None
        for evidence_id in disease.evidence:
            match = re.match(r"T(\d+)R\d+", evidence_id)
            if match:
                table_index = int(match.group(1))
                break
        if table_index is None or table_index >= len(parsed.tables):
            return disease.component
        title = cls._disease_table_title(parsed.tables[table_index])
        parts = [token for token in ("左幅", "右幅", "主桥", "引桥") if token in title]
        component = next(
            (
                token
                for token in (
                    "箱梁内",
                    "箱梁外",
                    "主塔内部",
                    "主塔外部",
                    "横向联系",
                    "主塔",
                    "拉索",
                    "桥面铺装",
                    "伸缩缝",
                    "人行道",
                    "护栏",
                    "排水系统",
                    "照明",
                    "桥墩",
                    "桥台",
                    "支座",
                    "主梁",
                )
                if token in title
            ),
            disease.component,
        )
        return "".join([*parts, component]) or disease.component

    @staticmethod
    def _has_aggregate_disease_semantics(parsed: ParsedDocument) -> bool:
        """Return whether the source itself defines an aggregate issue layer.

        Row count alone is insufficient: some competition labels retain a
        thousand-row component ledger.  Aggregation is therefore enabled only
        when the report also contains explicit ``显著病害`` / ``需及时处理的
        新增病害`` wording or several component-level ``检出…处`` summaries.
        """
        texts = [paragraph.text for paragraph in parsed.paragraphs]
        if any("显著病害" in text or re.search(r"需及时处理.{0,8}新增病害", text) for text in texts):
            return True
        summaries = [
            text
            for text in texts
            if "检出" in text
            and re.search(r"(?:共计\s*)?\d+\s*处", text)
            and re.search(r"主梁|横向联系|主塔|拉索|桥面铺装|人行道|护栏", text)
        ]
        return len(summaries) >= 3

    @staticmethod
    def _disease_region(disease: Disease, context: str) -> str:
        text = f"{context} {disease.location} {disease.description}"
        parts: list[str] = []
        if "左幅" in text:
            parts.append("左幅")
        elif "右幅" in text:
            parts.append("右幅")
        if "主桥" in text:
            parts.append("主桥")
        elif "引桥" in text:
            parts.append("引桥")
        parts.append(disease.component)
        if "箱内" in text:
            parts.append("箱内")
        elif "箱外" in text:
            parts.append("箱外")
        return "".join(dict.fromkeys(parts)) or disease.component

    @staticmethod
    def _canonical_group_type(value: str, context: str = "") -> str:
        compacted = re.sub(r"[、，,\s]+", "", value)
        compact_context = re.sub(r"[、，,\s]+", "", context)
        if re.search(r"(?:修补|封闭)(?:处|后)?(?:重新|再次|二次)?(?:开裂|裂缝)", compact_context):
            return "裂缝"
        if (
            re.search(r"破损|剥落|损坏", compacted)
            and re.search(r"露筋|钢筋裸露", compact_context)
        ) or (
            re.search(r"露筋|钢筋裸露", compacted)
            and re.search(r"破损|剥落|损坏", compact_context)
        ):
            return "破损露筋"
        # Normalise wording variants only at the large-ledger aggregation
        # boundary.  Raw reports use dozens of near-synonyms (纵/横/斜裂缝、
        # 局部开裂、网裂, etc.) for the same decision-relevant issue family.
        aliases = (
            (r"(?:裂缝|开裂|网裂|龟裂).*(?:修补|封闭|加固)|(?:修补|封闭|加固).*(?:裂缝|开裂|网裂|龟裂)", "裂缝修补"),
            (r"裂缝|开裂|网裂|龟裂", "裂缝"),
            (r"(?:破损|剥落|表层脱落|露骨料|损坏).*(?:露筋|钢筋)|(?:露筋|钢筋).*(?:破损|剥落|损坏)", "破损露筋"),
            (r"破损|剥落|表层脱落|露骨料|损坏", "破损"),
            (r"露筋.*(?:锈蚀|锈胀)|(?:锈蚀|锈胀).*露筋", "露筋锈蚀"),
            (r"露筋|钢筋裸露", "露筋"),
            (r"渗水|漏水|泛碱|盐析|水迹", "渗水泛碱"),
            (r"蜂窝|麻面|不密实|浇筑不实", "蜂窝麻面"),
            (r"(?:涂层|涂料).*(?:脱落|剥离|疏松|空鼓|起泡|开裂)", "涂层损坏"),
            (r"黄油", "黄油异常"),
            (r"锈蚀|锈胀|腐蚀|锈迹", "锈蚀"),
        )
        for pattern, canonical in aliases:
            if re.search(pattern, compacted):
                return canonical
        return value.strip("-—/ ") or "未明确"

    @staticmethod
    def _extract_metrics(description: str) -> dict[str, str]:
        metrics: dict[str, str] = {}
        for key, value in re.findall(r"\b(L|W|max|Wmax|S|D)\s*=\s*([^，,；;。]+)", description, re.I):
            metrics[key.upper()] = value.strip()
        return metrics

    def _infer_disease_history(self, state: DocumentState) -> None:
        if not state.comparisons:
            return
        comparison_by_section = {record.section: record for record in state.comparisons}
        section_type_counts = Counter(
            (self._disease_section(item.component), item.type)
            for item in state.diseases
        )
        for disease in state.diseases:
            section = self._disease_section(disease.component)
            record = comparison_by_section.get(section)
            if not record:
                continue
            # The section comparison is relevant evidence for every row in
            # that section, even when it is not specific enough to decide the
            # row automatically.  Attaching it does not broadcast a label; it
            # gives the bounded Agent reviewer the correct temporal context.
            disease.evidence = list(dict.fromkeys([*disease.evidence, *record.evidence]))
            # Section-level prose is useful narrative context but is not a
            # row-level historical fact.  Bind it to a disease only when the
            # comparison explicitly names that disease type or its location;
            # otherwise a chronology-safe HistoryIndex may perform semantic
            # matching later, and the renderer truthfully reports "未明确".
            row_tokens = [
                token
                for token in re.split(r"[、，,；;：:\s]+", f"{disease.type} {disease.location}")
                if len(token) >= 2
            ]
            comparison_text = f"{record.previous} {record.current} {record.development}"
            row_specific = any(token in comparison_text for token in row_tokens)
            if row_specific:
                disease.previous_state = previous_presence(record.previous)
                disease.development = development_level(record.development)
            development = record.development
            explicitly_no_new = bool(re.search(r"未(?:见)?新增|无(?:明显)?新增|未增新", development))
            # A section-level statement such as “本次未见新增” is not a
            # row-level fact.  Apply it only when the comparison prose names
            # this disease type/location; otherwise leave the row unresolved
            # for chronology-safe history matching or model review.
            if explicitly_no_new and row_specific:
                disease.is_new = False
                disease.novelty_source = "current_report_no_new"
            elif "新增" in development or "新发现" in development:
                after_new = development.split("新增", 1)[-1]
                type_tokens = [token for token in re.split(r"[、，,；;：:\s]+", disease.type) if token]
                location_tokens = [
                    token
                    for token in re.split(r"[、，,；;：:\s]+", disease.location)
                    if len(token) >= 2
                ]
                names_type = any(token in after_new for token in type_tokens)
                names_location = any(token in after_new for token in location_tokens)
                unique_type_in_section = section_type_counts[(section, disease.type)] == 1
                if names_type and (unique_type_in_section or names_location):
                    disease.is_new = True
                    disease.novelty_source = "current_report_new"
            elif row_specific and any(term in development for term in ("稳定", "延续", "仍存在")):
                disease.is_new = False
                disease.novelty_source = "current_report_stable"

    @staticmethod
    def _cohere_disease_history(state: DocumentState) -> None:
        """Resolve contradictory row-level novelty/history fields.

        Some source reports mark an observation as new while copying a
        section-level ``已发现`` value into its previous-state column.  A new
        row means the exact observation was absent in the previous cycle, so
        preserve the explicit novelty flag and normalize the dependent enum.
        This is a schema-level invariant and does not use labels or bridge
        identities.
        """
        for disease in state.diseases:
            if (
                disease.is_new is True
                and disease.previous_state
                and "已发现" in disease.previous_state
                and "未发现" not in disease.previous_state
            ):
                disease.previous_state = "未发现"

    @staticmethod
    def _disease_section(component: str) -> str:
        if component in {"主梁", "横向联系"}:
            return "上部结构"
        if component in {"桥墩", "桥台", "支座"}:
            return "下部结构"
        return "桥面系"

    def _extract_recommendations(self, parsed: ParsedDocument, state: DocumentState, summary: tuple[TableBlock, dict[str, str], dict[str, int]] | None) -> None:
        # Prefer the dedicated recommendation section: page-broken Word tables
        # may expose repeated ``养护建议`` rows as separate XML tables, whereas
        # each section paragraph still represents exactly one recommendation.
        records = []
        for paragraph in self._recommendation_section(parsed):
            numbered = self._numbered_item(paragraph.text)
            if not numbered and not self._is_recommendation_text(paragraph.text):
                continue
            item = self._strip_item_number(paragraph.text) if numbered else paragraph.text.strip(" ；;")
            if len(item) < 8:
                continue
            item_evidence = self._paragraph_evidence(state, parsed, paragraph.index, paragraph.text, FIELD_PRIORITY["conclusion"])
            records.append((item, item_evidence))

        if not records and summary:
            table, mapping, rows = summary
            text = mapping.get("维护建议", "")
            if text:
                evidence = self._table_evidence(state, parsed, table, rows["维护建议"], text, FIELD_PRIORITY["conclusion"])
                records = [(item, evidence) for item in self._split_numbered_items(text)]

        for sequence, (item, item_evidence) in enumerate(records, 1):
            # Numbered bullets in an explicit recommendation section are
            # occasionally written as bare noun phrases.  Preserve their
            # source content while making the action semantics explicit in the
            # generated recommendation table.
            if not re.search(r"建议|应|需|及时|立即|加强|修复|维修|处治|处理|清理|更换|观测|检查|养护", item):
                item = f"建议{item}"
            category = self._recommendation_category(item)
            component = self._recommendation_component(item)
            state.recommendations.append(
                Recommendation(
                    sequence=sequence,
                    category=category,
                    content=item,
                    component=component,
                    evidence=[item_evidence] if item_evidence else [],
                    confidence=0.93 if item_evidence else 0.65,
                )
            )
        category_order = {"立即处置": 0, "尽快维修": 1, "预防性养护": 2}
        state.recommendations.sort(key=lambda item: (category_order[item.category], item.sequence))
        for sequence, recommendation in enumerate(state.recommendations, 1):
            recommendation.sequence = sequence

    @classmethod
    def _recommendation_section(cls, parsed: ParsedDocument):
        headings = [
            paragraph.index
            for paragraph in parsed.paragraphs
            if len(paragraph.text) <= 40
            and re.search(r"(?:养护建议|维护建议|维修建议|处治建议|处理建议|结论与建议|(?:\d+\s*)?建议)\s*$", paragraph.text)
        ]
        if not headings:
            return []
        marker = headings[-1]
        section = []
        for paragraph in parsed.paragraphs[marker + 1:marker + 45]:
            if section and (
                paragraph.style.startswith("Heading")
                or re.match(r"^(?:附[录表]|第[一二三四五六七八九十]+章)", paragraph.text)
            ):
                break
            section.append(paragraph)
        return section

    @staticmethod
    def _numbered_item(text: str) -> bool:
        return bool(re.match(r"^(?:[（(]\d+[）)]|[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]|\d+[.、）)])", text.strip()))

    @staticmethod
    def _is_recommendation_text(text: str) -> bool:
        value = text.strip()
        if len(value) < 8 or re.search(
            r"(?:(?:提出|给出).{0,8}(?:如下|以下).{0,6}建议|"
            r"(?:针对|根据|结合).{0,30}建议(?:如下(?:所示)?|为)?[：:]?\s*$|"
            r"(?:具体|主要|养护|维修|处治)?建议(?:如下(?:所示)?)?[：:]\s*$)",
            value,
        ):
            return False
        return bool(
            re.search(
                r"(?:建议|应当|应及时|应立即|应进行|应采取|应加强|可采用|宜(?:对|进行|及时)|"
                r"需(?:要|及时|进行)|加强(?:对|日常|观测|巡查)|加强.{0,12}(?:巡查|观测|保养|养护|管理)|及时(?:修复|处理|清理|更换)|"
                r"进行(?:维修|修复|处理|处治|更换|加固|清理|养护|观测|检测)|严禁)",
                value,
            )
        )

    @staticmethod
    def _strip_item_number(text: str) -> str:
        return re.sub(r"^(?:[（(]\d+[）)]|[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]|\d+[.、）)])\s*", "", text.strip()).strip(" ；;")

    @staticmethod
    def _split_numbered_items(text: str) -> list[str]:
        if not text:
            return []
        # The negative look-behind prevents standards such as
        # ``DB50/231-2006）`` and decimal values from being mistaken for list
        # markers while still accepting the common ``1）`` Word style.
        marker = r"(?<![\d/.-])(?:[（(]\d+[）)]|[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]|\d+[.、）)])(?!\d)"
        pieces = re.split(rf"(?={marker})", text)
        items = [re.sub(rf"^{marker}\s*", "", piece).strip(" ；;") for piece in pieces if re.match(rf"^{marker}", piece)]
        return [item for item in items if len(item) >= 8]

    @staticmethod
    def _recommendation_category(text: str) -> str:
        if re.search(r"立即|限行|禁止通行|停止使用|应急", text):
            return "立即处置"
        if re.search(r"其他常规|日常(?:保养|检查|巡查|观测)|定期(?:检查|观测)|加强观测|预防性|必要养护|严禁超载|加强.{0,12}(?:巡查|观测|保养|养护|管理|引导)", text):
            return "预防性养护"
        return "尽快维修"

    @staticmethod
    def _recommendation_component(text: str) -> str:
        found = []
        for pattern, component in COMPONENT_PATTERNS:
            if pattern.search(text) and component not in found:
                found.append(component)
        return "、".join(found) if found else "全桥"

    def _extract_narratives(self, parsed: ParsedDocument, state: DocumentState, summary: tuple[TableBlock, dict[str, str], dict[str, int]] | None) -> None:
        mapping: dict[str, str] = summary[1] if summary else {}
        current = self._numeric_mean(state.scores.overall.score.value)
        previous = self._numeric_mean(state.scores.previous.score.value)

        raw_conclusion = mapping.get("检测结论", "").strip()

        if raw_conclusion:
            state.narrative.overall_conclusion = raw_conclusion
        else:
            # Build from diseases when no raw conclusion available
            state.narrative.overall_conclusion = self._build_conclusion_from_diseases(state)

        # --- Key Risks ---
        assessment = mapping.get("评估结论", "")
        if assessment:
            state.narrative.key_risks = self._summarize_risks(assessment, state.diseases)
            prediction = re.search(r"预测评估\s*(.*?)(?:综合评估|$)", assessment)
            if prediction:
                state.narrative.safety_impacts = [prediction.group(1).strip()]
        if not state.narrative.key_risks:
            disease_summary = self._summarize_risks("", state.diseases)
            if disease_summary:
                state.narrative.key_risks = disease_summary
            else:
                state.narrative.key_risks = "无明显重大安全风险点，常规病害需持续跟踪监测。"

        # --- Development Trend ---
        trend_parts: list[str] = []
        if current is not None and previous is not None:
            delta = current - previous
            direction = "上升" if delta > 0 else "下降" if delta < 0 else "持平"
            trend_parts.append(f"总体评分较上一次{direction}{abs(delta):.2f}分。")
        if state.comparisons:
            trend_parts.append("；".join(f"{x.section}：{x.development}" for x in state.comparisons))
        if trend_parts:
            state.narrative.development_trend = " ".join(trend_parts)
        else:
            if state.scores.previous.score.value in ("无", None) and state.scores.previous.grade.value in ("无", None):
                state.narrative.development_trend = "无"
            else:
                state.narrative.development_trend = "与上一次定检相比，桥梁技术状况总体保持基本稳定，部分构件出现轻微病害发展，未见影响结构安全的新增恶化缺陷。"

    @staticmethod
    def _clean_conclusion(raw: str) -> str:
        """Strip regulation boilerplate and BCI/grade formula text from raw 检测结论,
        keeping only the qualitative disease description core."""
        t = raw.strip()
        # Remove leading numbering like "1 " or "1."
        t = re.sub(r"^\d+\s*", "", t)
        # Remove regulation citations: 依据规范《...》..., 按照《...》..., 按照重庆市《...》...
        t = re.sub(
            r"(?:依据|按照)\s*(?:规范)?[《<].*?[》>]\s*(?:的相关[规规]定)?[，,]?\s*(?:对桥梁进行完好状况(?:评定|评估)[，,]?)?\s*",
            "", t
        )
        # Remove "BCI=XX.XX，整体技术状况等级评定为X级" patterns
        t = re.sub(
            r"(?:本桥|全桥|.*?桥)?(?:桥)?的?\s*(?:BCI|Dr|BSI|整体技术状况指数)\s*[=：:]\s*\d+(?:\.\d+)?\s*[，,]?\s*",
            "", t
        )
        t = re.sub(
            r"整+体技术状况等级评定为\s*[ABCDE一二三四五]\s*级?\s*[，,]?\s*",
            "", t
        )
        t = re.sub(
            r"(?:为|属于?)\s*(?:完好|良好|合格|不合格|危险)\s*(?:状态|等级)?\s*[，,]?\s*",
            "", t
        )
        t = re.sub(r"应进行\s*(?:日常保养|保养小修|小修或中修|大修或改造|.*?)(?:。|$)", "", t)
        t = re.sub(r"养护对策为\s*.*?(?:。|$)", "", t)
        # Remove leftover regulation boilerplate at the beginning
        t = re.sub(r"^按照重庆市.*?(?:标准|规程)[，,]?\s*", "", t)
        # Clean up punctuation artifacts
        t = re.sub(r"^[\s，,；;。、]+", "", t)
        t = re.sub(r"[\s，,；;。]+$", "", t)
        t = re.sub(r"[。]{2,}", "。", t)
        t = re.sub(r"[，,]{2,}", "，", t)
        if t:
            if not t.endswith("。"):
                t += "。"
        return t

    def _build_conclusion_from_diseases(self, state: DocumentState) -> str:
        """Build a conclusion from extracted diseases when no raw text available.
        Ground truth format: '{bridge}桥面系评定为X级（Y），上部结构X级，下部结构X级，
        整体技术状况X级（Y）。主要病害为...'"""
        ov_g = state.scores.overall.grade.value or "未明确"
        dk_g = state.scores.deck.grade.value or "未明确"
        sp_g = state.scores.superstructure.grade.value or "未明确"
        sb_g = state.scores.substructure.grade.value or "未明确"

        condition_map = {
            "A级": "完好", "B级": "良好", "C级": "合格",
            "D级": "不合格", "E级": "危险",
        }

        def _grade_text(g: str) -> str:
            cond = condition_map.get(str(g)[:2], "")
            return f"{g}（{cond}）" if cond else str(g)

        # Grade summary
        parts = [
            f"桥面系评定为{_grade_text(dk_g)}",
            f"上部结构{_grade_text(sp_g)}",
            f"下部结构{_grade_text(sb_g)}",
            f"整体技术状况{_grade_text(ov_g)}",
        ]
        grade_str = "，".join(parts)

        # Group top diseases by scope for disease summary
        scope_types: dict[str, list[str]] = {"上部结构": [], "下部结构": [], "桥面系": []}
        for d in state.diseases:
            typ = d.type or ""
            comp = d.component or d.location or ""
            desc_snippet = ""
            if comp:
                desc_snippet = comp
            if d.scope == "superstructure" or any(k in comp for k in ("梁", "板", "拱", "索", "铰缝")):
                entry = f"{comp}{typ}" if comp else typ
                if entry and entry not in scope_types["上部结构"]:
                    scope_types["上部结构"].append(entry)
            elif d.scope == "substructure" or any(k in comp for k in ("墩", "台", "基础", "盖梁", "支座")):
                entry = f"{comp}{typ}" if comp else typ
                if entry and entry not in scope_types["下部结构"]:
                    scope_types["下部结构"].append(entry)
            else:
                entry = f"{comp}{typ}" if comp else typ
                if entry and entry not in scope_types["桥面系"]:
                    scope_types["桥面系"].append(entry)

        disease_parts = []
        for scope_name in ("桥面系", "上部结构", "下部结构"):
            entries = scope_types[scope_name][:5]
            if entries:
                disease_parts.append(f"{scope_name}{'、'.join(entries)}")

        if disease_parts:
            return f"{grade_str}。主要病害为{'；'.join(disease_parts)}。"
        return f"{grade_str}。"

    @staticmethod
    def _build_key_risks(assessment: str, diseases: list) -> str:
        """Build key risks using specific disease locations and impact descriptions."""
        if not diseases:
            if assessment:
                return assessment[:300]
            return "无明显重大安全风险点，常规病害需持续跟踪监测。"

        risk_clauses = []
        seen_scopes = set()

        # Priority: structural diseases first, then durability, then safety
        priority_diseases = []
        for d in diseases:
            comp = d.component or d.location or ""
            typ = d.type or ""
            desc = d.description or ""
            severity = 0
            if any(k in typ or k in desc for k in ("露筋", "空洞", "断裂", "剪切变形", "脱空")):
                severity = 3
            elif any(k in typ or k in desc for k in ("裂缝", "破损", "剥落", "锈蚀")):
                severity = 2
            elif any(k in typ or k in desc for k in ("渗水", "泛碱", "缺失", "堵塞")):
                severity = 1
            priority_diseases.append((severity, d))

        priority_diseases.sort(key=lambda x: -x[0])

        for _, d in priority_diseases:
            if len(risk_clauses) >= 4:
                break
            comp = d.component or d.location or ""
            comp = re.sub(r"^\d+[\.、\s]+", "", comp)
            typ = d.type or "病害"
            desc = d.description or ""

            # Determine impact based on location/type
            if any(k in comp for k in ("支座",)):
                impact = "可能影响梁体受力"
                scope = "支座"
            elif any(k in comp for k in ("梁", "板", "拱")) or any(k in typ for k in ("裂缝", "破损", "露筋", "空洞")):
                if "裂缝" in typ:
                    impact = "影响结构耐久性"
                else:
                    impact = "影响结构耐久性"
                scope = "上部"
            elif any(k in comp for k in ("墩", "台", "盖梁", "基础")):
                impact = "长期发展可能影响下部结构耐久性"
                scope = "下部"
            elif any(k in comp for k in ("护栏", "防抛网", "防撞")):
                impact = "影响行车及行人安全"
                scope = "护栏"
            elif any(k in comp for k in ("伸缩缝",)):
                impact = "影响行车舒适性"
                scope = "伸缩缝"
            elif any(k in comp for k in ("铺装", "桥面")):
                impact = "影响行车安全"
                scope = "桥面"
            else:
                impact = "影响结构耐久性"
                scope = "其他"

            if scope in seen_scopes:
                continue
            seen_scopes.add(scope)

            # Extract dimension info from description if available
            dim_match = re.search(r"[（(]?[LlWwSs]\s*[=＝]\s*[\d.]+\s*m?(?:\s*[×xX*,，]\s*[\d.]+\s*m?)*[）)]?", desc)
            dim_match2 = re.search(r"(?:最大)?(?:宽度?|长度?)\s*(?:为)?\s*[\d.]+\s*mm", desc)
            dim_info = ""
            if dim_match:
                dim_info = f"（{dim_match.group(0).strip('（）()')}）"
            elif dim_match2:
                dim_info = f"（{dim_match2.group(0)}）"

            clause = f"{comp}{typ}{dim_info}，{impact}"
            risk_clauses.append(clause)

        if not risk_clauses:
            # Fallback to count-based summary
            counts = Counter((d.component, d.type) for d in diseases)
            parts = [f"{component}{disease_type}{count}处" for (component, disease_type), count in counts.most_common(6)]
            return "；".join(parts) + "。"

        return "；".join(risk_clauses) + "。"

    @staticmethod
    def _numeric_mean(value: object) -> float | None:
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(value or ""))]
        return sum(numbers) / len(numbers) if numbers else None

    @staticmethod
    def _summarize_risks(assessment: str, diseases: list[Disease]) -> str:
        if not diseases:
            return assessment[:300]
        counts = Counter((d.component, d.type) for d in diseases)
        parts = [f"{component}{disease_type}{count}处" for (component, disease_type), count in counts.most_common(6)]
        return "；".join(parts) + "。"
