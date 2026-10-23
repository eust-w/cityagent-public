from __future__ import annotations

import re
from collections import Counter
import json
from math import sqrt
from typing import Any, Callable

from bridge_evidence.agents import (
    DecisionWriter,
    DetailReviewer,
    DiseaseNormalizer,
    GatewayClient,
    SummaryReviewer,
    VisionEvidenceReviewer,
)
from bridge_evidence.agents.decision import score_development_trend
from bridge_evidence.config import Settings
from bridge_evidence.domain_rules import (
    canonical_disease_from_text,
    is_condition_state_label,
    is_meaningful_condition_state,
    is_non_disease_label,
)
from bridge_evidence.evidence import extract_visual_candidates
from bridge_evidence.models.agent import AgentCommand, ToolDescriptor, ToolManifest, ToolResult
from bridge_evidence.models.schema import DocumentState, EvidenceRef, FieldValue
from bridge_evidence.models.validation import ConstraintEngine
from bridge_evidence.parsers.docx import ParsedDocument


FLOAT = r"\d{1,3}(?:\.\d+)?"
DECISION_DEFECT_TERMS = (
    "裂缝", "破损", "露筋", "渗水", "泛碱", "锈蚀", "腐蚀", "剥落",
    "坑槽", "车辙", "错台", "松动", "脱落", "缺失", "变形", "沉降",
)
DETAIL_LINK_TERMS = (
    "主梁", "横向联系", "桥墩", "桥台", "支座", "桥面铺装", "伸缩缝",
    "护栏", "排水", "人行道", "斜拉索", "主塔", "吊杆", "左幅", "右幅",
    *DECISION_DEFECT_TERMS,
)


class EvidenceToolRegistry:
    def __init__(
        self,
        state: DocumentState,
        parsed: ParsedDocument,
        validator: ConstraintEngine,
        *,
        client: GatewayClient | None = None,
        settings: Settings | None = None,
        prompts: dict[str, str] | None = None,
        workflow_profile: str = "full",
    ):
        self.state = state
        self.parsed = parsed
        self.validator = validator
        self.client = client
        self.settings = settings
        self.prompts = prompts or {}
        if workflow_profile not in {"full", "summary", "details"}:
            raise ValueError(f"未知 workflow_profile: {workflow_profile}")
        self.workflow_profile = workflow_profile
        self.finalized = False
        self.failed = False
        self.failure_reason: str | None = None
        self._visual_candidates = None
        self.required_actions = self._build_required_actions()
        self.completed_action_keys: set[str] = set()
        tools = [
            ToolDescriptor(name="get_validation_report", description="读取当前硬约束错误、告警与证据覆盖率"),
            ToolDescriptor(name="inspect_summary", description="读取报告开头摘要表及精确表格坐标"),
            ToolDescriptor(name="search_evidence", description="在段落和表格中检索关键词", arguments_schema={"query": "string", "limit": "1..20"}),
            ToolDescriptor(name="inspect_table", description="分页读取指定表格", arguments_schema={"table_id": "int", "start": "int", "limit": "1..30"}),
            ToolDescriptor(name="list_score_candidates", description="无偏枚举总体/部位/上一次的 BCI、BSI、左右幅及重复计算候选，不自动裁决", arguments_schema={"scope": "overall|deck|superstructure|substructure|previous"}),
            ToolDescriptor(name="profile_disease_tables", description="统计病害表规模、构件和左右幅分布"),
            ToolDescriptor(name="inspect_visual_evidence", description="仅在文本关键事实缺失时，用开放权重视觉模型交叉 OCR 最可能含表格的图片"),
            ToolDescriptor(name="review_summary", description="用开源文本模型 + VL Embedding/Reranker 根据冲突候选复核 12 个简要信息字段", read_only=False),
            ToolDescriptor(name="review_details", description="用开源文本模型依据当前与历史证据审校病害和建议字段", read_only=False),
            ToolDescriptor(name="apply_fact_patch", description="提交有证据支持的关键字段 patch", read_only=False, arguments_schema={"patches": "[{field,value,evidence_ids}]"}),
            ToolDescriptor(name="mark_unavailable", description="在穷尽候选检索后，将原报告未提供的关键字段显式标为无", read_only=False, arguments_schema={"scope": "overall"}),
            ToolDescriptor(name="normalize_diseases", description="清理照片伪行并调用领域 Agent 规范化未明确病害实体", read_only=False),
            ToolDescriptor(name="write_decision", description="调用领域 Agent 补充受约束成因和安全影响", read_only=False),
            ToolDescriptor(name="finalize", description="验证通过后结束 Agent run", read_only=False),
            ToolDescriptor(name="fail", description="明确终止无法完成的 run", read_only=False, arguments_schema={"reason": "string"}),
        ]
        if self.workflow_profile == "summary":
            allowed = {
                "get_validation_report",
                "inspect_summary",
                "search_evidence",
                "inspect_table",
                "list_score_candidates",
                "review_summary",
                "apply_fact_patch",
                "mark_unavailable",
                "finalize",
                "fail",
            }
            tools = [tool for tool in tools if tool.name in allowed]
        elif self.workflow_profile == "details":
            allowed = {
                "get_validation_report",
                "review_details",
                "finalize",
                "fail",
            }
            tools = [tool for tool in tools if tool.name in allowed]
        self.manifest = ToolManifest(tools=tools)
        self._handlers: dict[str, Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]] = {
            "get_validation_report": self._validation_report,
            "inspect_summary": self._inspect_summary,
            "search_evidence": self._search_evidence,
            "inspect_table": self._inspect_table,
            "list_score_candidates": self._list_score_candidates,
            "profile_disease_tables": self._profile_disease_tables,
            "inspect_visual_evidence": self._inspect_visual_evidence,
            "review_summary": self._review_summary,
            "review_details": self._review_details,
            "apply_fact_patch": self._apply_fact_patch,
            "mark_unavailable": self._mark_unavailable,
            "normalize_diseases": self._normalize_diseases,
            "write_decision": self._write_decision,
            "finalize": self._finalize,
            "fail": self._fail,
        }

    def execute(self, command: AgentCommand) -> ToolResult:
        handler = self._handlers.get(command.action)
        if not handler:
            return ToolResult(run_id=command.run_id, seq=command.seq, tool=command.action, success=False, error="工具未注册")
        try:
            content, evidence_ids = handler(command.arguments)
            result = ToolResult(run_id=command.run_id, seq=command.seq, tool=command.action, success=True, content=content, new_evidence_ids=evidence_ids)
            if command.action not in {"finalize", "fail"}:
                self.completed_action_keys.add(self._action_key(command.action, command.arguments))
            return result
        except Exception as exc:
            return ToolResult(run_id=command.run_id, seq=command.seq, tool=command.action, success=False, error=str(exc))

    def _validation_report(self, _: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        self.state.validation = self.validator.validate(self.state)
        return self.state.validation.model_dump(mode="json"), []

    def _inspect_summary(self, _: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        tables = []
        evidence_ids: list[str] = []
        for table in self.parsed.tables[:4]:
            if not any(any(key in cell for key in ("工程名称", "检测结论", "检测结果", "维护建议")) for row in table.rows for cell in row[:1]):
                continue
            rows = []
            for row_index, row in enumerate(table.rows[:24]):
                evidence_id = self._ensure_table_evidence(table.index, row_index, " | ".join(row), priority=95)
                evidence_ids.append(evidence_id)
                rows.append({"evidence_id": evidence_id, "cells": row})
            tables.append({"table_id": table.index, "rows": rows})
        return {"tables": tables}, evidence_ids

    def _search_evidence(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("query 不能为空")
        limit = min(20, max(1, int(arguments.get("limit") or 10)))
        try:
            pattern = re.compile(query, re.I)
        except re.error:
            pattern = re.compile(re.escape(query), re.I)
        matches: list[dict[str, Any]] = []
        evidence_ids: list[str] = []
        for paragraph in self.parsed.paragraphs:
            if paragraph.text and pattern.search(paragraph.text):
                evidence_id = self._ensure_paragraph_evidence(paragraph.index, paragraph.text, priority=80)
                evidence_ids.append(evidence_id)
                matches.append({"evidence_id": evidence_id, "kind": "paragraph", "text": paragraph.text[:1800]})
                if len(matches) >= limit:
                    return {"matches": matches}, evidence_ids
        for table in self.parsed.tables:
            for row_index, row in enumerate(table.rows):
                text = " | ".join(row)
                if pattern.search(text):
                    evidence_id = self._ensure_table_evidence(table.index, row_index, text, priority=82)
                    evidence_ids.append(evidence_id)
                    matches.append({"evidence_id": evidence_id, "kind": "table", "table_id": table.index, "row": row_index, "text": text[:1800]})
                    if len(matches) >= limit:
                        return {"matches": matches}, evidence_ids
        return {"matches": matches}, evidence_ids

    def _inspect_table(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        table_id = int(arguments.get("table_id", -1))
        if table_id < 0 or table_id >= len(self.parsed.tables):
            raise ValueError(f"table_id 越界: {table_id}")
        start = max(0, int(arguments.get("start") or 0))
        limit = min(30, max(1, int(arguments.get("limit") or 12)))
        table = self.parsed.tables[table_id]
        rows = []
        evidence_ids: list[str] = []
        for row_index in range(start, min(len(table.rows), start + limit)):
            text = " | ".join(table.rows[row_index])
            evidence_id = self._ensure_table_evidence(table.index, row_index, text, priority=82)
            evidence_ids.append(evidence_id)
            rows.append({"evidence_id": evidence_id, "cells": table.rows[row_index]})
        return {"table_id": table_id, "context": table.context[-8:], "header": table.header, "rows": rows, "total_rows": len(table.rows)}, evidence_ids

    def _list_score_candidates(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        scope = str(arguments.get("scope") or "overall")
        if scope not in {"overall", "deck", "superstructure", "substructure", "previous"}:
            raise ValueError(f"未知评分范围: {scope}")
        candidates, evidence_ids = self._score_candidate_catalog(scope)
        resolved = getattr(self.state.scores, scope).model_dump(mode="json")
        metric_kinds = sorted({str(item["metric"]) for item in candidates})
        distinct_values = sorted({float(item["score"]) for item in candidates})
        return {
            "scope": scope,
            "resolved_state": resolved,
            "candidate_metric_kinds": metric_kinds,
            "distinct_values": distinct_values,
            "semantic_conflict": len(metric_kinds) > 1 or len(distinct_values) > 1,
            "candidates": candidates[:30],
        }, evidence_ids

    def _score_candidate_catalog(self, scope: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Enumerate score evidence without deciding which metric is the answer.

        This is deliberately a recall tool, not a rule engine.  It keeps BCI,
        BSI, side-specific and repeated calculation candidates distinct so the
        model can resolve the report's final publishing convention.
        """
        metric_patterns = {
            "overall": r"(?:BCI(?![msx])|Dr)",
            "deck": r"(?:BCIm|BSIm)(?![A-Za-z0-9])",
            "superstructure": r"(?:BCIs|BSIs)(?![A-Za-z0-9])",
            "substructure": r"(?:BCIx|BSIx)(?![A-Za-z0-9])",
            "previous": r"(?:BCI(?![msx])|Dr)",
        }
        pattern = re.compile(rf"(?P<metric>{metric_patterns[scope]})", re.I)
        next_metric = re.compile(r"(?:BCI[mMsSxX]?|BSI[mMsSxX]?)(?![A-Za-z0-9])|\bDr\b", re.I)
        previous_signal = re.compile(r"(?:上一次|上次|上一年度|原评定|原等级)", re.I)
        sources: list[tuple[str, str, int | None, int | None, str]] = []
        sources.extend((paragraph.text, "paragraph", paragraph.index, None, "") for paragraph in self.parsed.paragraphs)
        for table in self.parsed.tables:
            context = " ".join(table.context[-4:])
            sources.extend((" | ".join(row), "table", table.index, row_index, context) for row_index, row in enumerate(table.rows))

        candidates: list[dict[str, Any]] = []
        evidence_ids: list[str] = []
        seen: set[tuple[str, str, float, str | None]] = set()
        if scope == "previous":
            # Expose the complete dated sequence to the Agent.  A history row
            # need not literally say ``上一次``; its temporal role comes from
            # the inspection-date column and its position in that sequence.
            for table in self.parsed.tables:
                header = " | ".join(table.header)
                if "检测维修日期" not in header or "质量状况" not in header:
                    continue
                for row_index, row in enumerate(table.rows[1:], 1):
                    text = " | ".join(row)
                    match = re.search(
                        rf"(?:主桥|全桥|总体|整体).{{0,40}}?评分\s*(?:为|=|：|:)?\s*({FLOAT})",
                        text,
                        re.I,
                    ) or re.search(
                        rf"(?:BCI(?![msx])|Dr)\s*(?:值|评分)?\s*(?:为|=|：|:)?\s*({FLOAT})",
                        text,
                        re.I,
                    )
                    date_match = re.search(r"20\d{2}(?:年[^|]{0,18})?", row[0] if row else text)
                    if not match or not date_match:
                        continue
                    score = float(match.group(1))
                    grade_match = re.search(r"([ABCDE])\s*(?:级|类)|([一二三四五])\s*类", text, re.I)
                    grade = (grade_match.group(1) or grade_match.group(2)).upper() if grade_match else None
                    evidence_id = self._ensure_table_evidence(table.index, row_index, text, priority=97)
                    evidence_ids.append(evidence_id)
                    seen.add((evidence_id, "DR", score, None))
                    candidates.append({
                        "scope": scope,
                        "metric": "Dr",
                        "metric_role": "overall_index",
                        "source_role": "dated_history_record",
                        "inspection_period": date_match.group(0),
                        "score": score,
                        "numeric_candidates": [score],
                        "grade": grade,
                        "grade_binding": grade is not None,
                        "side": None,
                        "self_contradictory": False,
                        "evidence_id": evidence_id,
                        "text": f"{header} | {text}"[:1800],
                    })
        scope_labels = {
            "deck": "桥面系",
            "superstructure": "上部结构",
            "substructure": "下部结构",
        }
        scope_metric = {
            "deck": "BCIm",
            "superstructure": "BCIs",
            "substructure": "BCIx",
        }
        for text, kind, major_index, row_index, context in sources:
            if scope == "previous" and not previous_signal.search(text):
                continue
            if (
                kind == "table"
                and scope in scope_labels
                and scope_labels[scope] in text
                and "技术状况评定表" in context
            ):
                table_values = [float(value) for value in re.findall(FLOAT, text) if 0 <= float(value) <= 100]
                if table_values:
                    table = self.parsed.tables[int(major_index)]
                    header = table.header
                    component_grade_index = next(
                        (
                            index
                            for index, value in enumerate(header)
                            if "技术状况" in "".join(value.split())
                            and "指数" not in value
                            and "整体" not in value
                            and "桥梁" not in value
                            and "完好" not in value
                        ),
                        None,
                    )
                    component_grade: str | None = None
                    if component_grade_index is not None and row_index is not None:
                        row = table.rows[int(row_index)]
                        shifted = len(row) == len(header) + 1 and len(row) > 2 and bool(
                            re.fullmatch(r"BCI[mMsSxX]?|BCL[mMsSxX]?", row[1].strip(), re.I)
                        )
                        actual_grade_index = (
                            component_grade_index + 1
                            if shifted and component_grade_index >= 1
                            else component_grade_index
                        )
                        if actual_grade_index < len(row):
                            grade_match = re.fullmatch(
                                r"\s*([ABCDE])\s*(?:级|类)?\s*",
                                row[actual_grade_index],
                                re.I,
                            )
                            if grade_match:
                                component_grade = grade_match.group(1).upper()
                    evidence_id = self._ensure_table_evidence(int(major_index), int(row_index), text, priority=90)
                    side = "左幅" if "左幅" in context else "右幅" if "右幅" in context else None
                    key = (evidence_id, scope_metric[scope].upper(), table_values[0], side)
                    if key not in seen:
                        seen.add(key)
                        evidence_ids.append(evidence_id)
                        candidates.append({
                            "scope": scope,
                            "metric": scope_metric[scope],
                            "metric_role": "component_index",
                            "source_role": "final_component_rating_table",
                            "score": table_values[0],
                            "numeric_candidates": table_values[:8],
                            "grade": component_grade,
                            "grade_binding": component_grade is not None,
                            "side": side,
                            "self_contradictory": False,
                            "evidence_id": evidence_id,
                            "text": f"{context} | {text}"[-1800:],
                        })
            for match in pattern.finditer(text):
                clause_start = max(text.rfind(mark, 0, match.start()) for mark in ("；", ";", "。", "\n")) + 1
                clause_ends = [
                    position
                    for mark in ("；", ";", "。", "\n")
                    if (position := text.find(mark, match.end())) >= 0
                ]
                clause_end = min(clause_ends) if clause_ends else min(len(text), match.end() + 180)
                clause = text[clause_start:clause_end]
                local_match_start = match.start() - clause_start
                mapped_grade: str | None = None

                # Parallel metric lists are common in published conclusions,
                # for example ``BSIm、BSIs、BSIx分别为76.36(C)、80(B)、100(A)``.
                # Bind by position instead of taking the last number in a
                # fixed-width tail (which previously captured the ``1`` in a
                # following bridge name and emitted BSIx=1.0).
                split = re.search(r"分别为", clause)
                values: list[float] = []
                score: float | None = None
                if split and local_match_start < split.start():
                    metric_matches = list(next_metric.finditer(clause[:split.start()]))
                    value_matches = list(re.finditer(
                        rf"(?P<score>{FLOAT})(?:\s*[（(]\s*(?P<grade>[ABCDE])\s*(?:级|类)?\s*[）)])?",
                        clause[split.end():],
                        re.I,
                    ))
                    metric_index = next(
                        (index for index, item in enumerate(metric_matches) if item.start() == local_match_start),
                        None,
                    )
                    values = [float(item.group("score")) for item in value_matches]
                    if metric_index is not None and metric_index < len(value_matches):
                        bound = value_matches[metric_index]
                        score = float(bound.group("score"))
                        mapped_grade = (bound.group("grade") or "").upper() or None
                if score is None:
                    body = next_metric.split(text[match.end():clause_end], maxsplit=1)[0]
                    body = re.sub(r"^\s*(?:值|评分|得分)?\s*", "", body)
                    values = [float(value) for value in re.findall(FLOAT, body) if 0 <= float(value) <= 100]
                    bound_values = [
                        float(value)
                        for value in re.findall(rf"(?:=|为|：|:)\s*({FLOAT})", body)
                        if 0 <= float(value) <= 100
                    ]
                    if bound_values:
                        score = bound_values[-1]
                    elif values:
                        score = values[0]
                if not values:
                    continue
                if score is None:
                    continue
                local = text[max(0, match.start() - 90): min(len(text), match.end() + 160)]
                tail = text[match.end():clause_end]
                grade_match = re.search(r"(?:评定为|等级为|评为|定为)?\s*[“\"]?([ABCDE一二三四五])\s*(?:级|类)", tail, re.I)
                if mapped_grade:
                    grade_match = re.match(r"([ABCDE])", mapped_grade, re.I)
                grade_binding = bool(
                    grade_match
                    and (mapped_grade is not None or not next_metric.search(tail[:grade_match.start()]))
                )
                if kind == "paragraph":
                    evidence_id = self._ensure_paragraph_evidence(int(major_index), text, priority=88)
                else:
                    evidence_id = self._ensure_table_evidence(int(major_index), int(row_index), text, priority=90)
                evidence_ids.append(evidence_id)
                prefix = text[max(0, match.start() - 120):match.start()]
                left_position = prefix.rfind("左幅")
                right_position = prefix.rfind("右幅")
                if left_position >= 0 or right_position >= 0:
                    side = "左幅" if left_position > right_position else "右幅"
                else:
                    suffix = text[match.end():match.end() + 24]
                    side = "左幅" if "左幅" in suffix else "右幅" if "右幅" in suffix else None
                metric = match.group("metric")
                key = (evidence_id, metric.upper(), score, side)
                if key in seen:
                    continue
                seen.add(key)
                final_overall_table = (
                    scope == "overall"
                    and "综合评定分数" in text
                    and "技术状况评定表" in context
                )
                candidates.append({
                    "scope": scope,
                    "metric": metric,
                    "metric_role": "control_index" if metric.upper().startswith("BSI") else "component_index",
                    "source_role": "final_overall_rating_table" if final_overall_table else "explicit_assessment_sentence" if grade_match else "metric_mention",
                    "score": score,
                    "numeric_candidates": values[-8:],
                    "grade": grade_match.group(1).upper() if grade_match and grade_binding else None,
                    "grade_binding": grade_binding,
                    "side": side,
                    "self_contradictory": False,
                    "evidence_id": evidence_id,
                    "text": f"{context} | {text}"[:1800],
                })
        # Surface an invariant violation instead of resolving it here.  The
        # same metric/value cannot coherently map to two grades inside one
        # evidence sentence.  Marking the evidence lets the Agent seek a final
        # rating table or report-local grade scale without encoding a bridge,
        # year, direction or preferred numeric answer in deterministic code.
        grade_sets: dict[tuple[str, str, float], set[str]] = {}
        for item in candidates:
            if not item.get("grade"):
                continue
            key = (str(item["evidence_id"]), str(item["metric"]).upper(), float(item["score"]))
            grade_sets.setdefault(key, set()).add(str(item["grade"]).upper())
        contradictory = {key for key, grades in grade_sets.items() if len(grades) > 1}
        for item in candidates:
            key = (str(item["evidence_id"]), str(item["metric"]).upper(), float(item["score"]))
            item["self_contradictory"] = key in contradictory
        return candidates, list(dict.fromkeys(evidence_ids))

    def _profile_disease_tables(self, _: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        profiles = []
        all_rows = 0
        for table in self.parsed.tables:
            if not table.is_disease_table():
                continue
            data = table.rows[1:]
            texts = [" | ".join(row) for row in data]
            side_counts = Counter("左幅" if "左幅" in text else "右幅" if "右幅" in text else "未标幅" for text in texts)
            all_rows += len(data)
            profiles.append({
                "table_id": table.index,
                "rows": len(data),
                "context": table.context[-4:],
                "side_counts": dict(side_counts),
            })
        return {"table_count": len(profiles), "total_rows": all_rows, "tables": profiles}, []

    def _review_summary(self, _: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        if not self.client or not self.settings:
            raise RuntimeError("模型不可用，不能执行 review_summary")
        score_catalog: dict[str, list[dict[str, Any]]] = {}
        ambiguous_scopes: set[str] = set()
        for scope in ("overall", "deck", "superstructure", "substructure", "previous"):
            rows, _ = self._score_candidate_catalog(scope)
            score_catalog[scope] = rows
            metric_kinds = {str(item["metric"]).upper() for item in rows}
            values = {float(item["score"]) for item in rows}
            side_modes = {item.get("side") for item in rows}
            if len(metric_kinds) > 1 or len(values) > 1 or len(side_modes - {None}) > 1:
                ambiguous_scopes.add(scope)
        rating_scales: list[dict[str, Any]] = []
        rating_scale_evidence_ids: list[str] = []
        for table in self.parsed.tables:
            if not table.rows:
                continue
            header_text = " | ".join(table.rows[0])
            full_text = " || ".join(" | ".join(row) for row in table.rows[:4])
            if not re.search(r"\b(?:BCI|Dr)\b", header_text, re.I):
                continue
            if not re.search(r"(?:评估|评定|技术状况)?等级|[ABCDE]\s*(?:级|类)?", full_text, re.I):
                continue
            ids: list[str] = []
            for row_index, row in enumerate(table.rows[:4]):
                text = " | ".join(row)
                evidence_id = self._ensure_table_evidence(table.index, row_index, text, priority=91)
                ids.append(evidence_id)
                rating_scale_evidence_ids.append(evidence_id)
            rating_scales.append({
                "table_id": table.index,
                "evidence_ids": ids,
                "text": full_text[:1200],
            })
        score_signal = re.compile(
            r"(?:工程名称|桥梁名称|报告日期|检验日期|检测日期|"
            r"BCI[mMsSxX]?|BCL[mMsSxX]?|BSI[mMsSxX]?|\bDr\b|"
            r"总体评分|整体评分|上一次|上次|上一年度|评定为|技术状况等级|权重)",
            re.I,
        )
        candidates: list[dict[str, str]] = []
        for evidence_id, evidence in self.state.evidence.items():
            if score_signal.search(evidence.text):
                candidates.append({"evidence_id": evidence_id, "text": evidence.text[:700]})

        # Some parsers only materialise evidence for fields already selected.
        # Add every score-bearing paragraph/table row before semantic review so
        # the model can correct a deterministic winner using a better source.
        for paragraph in self.parsed.paragraphs:
            if score_signal.search(paragraph.text):
                evidence_id = self._ensure_paragraph_evidence(paragraph.index, paragraph.text, priority=92)
                candidates.append({"evidence_id": evidence_id, "text": paragraph.text[:700]})
        for table in self.parsed.tables:
            table_probe = " | ".join([
                *table.context[-6:],
                " | ".join(table.header),
                *(" | ".join(row) for row in table.rows[:30]),
            ])
            if not score_signal.search(table_probe):
                continue
            for row_index, row in enumerate(table.rows[:30]):
                text = " | ".join(row)
                evidence_id = self._ensure_table_evidence(table.index, row_index, text, priority=90)
                candidates.append({"evidence_id": evidence_id, "text": text[:700]})

        unique = {item["evidence_id"]: item for item in candidates}
        candidates = sorted(
            unique.values(),
            key=lambda item: -self.state.evidence[item["evidence_id"]].priority,
        )
        if not candidates:
            # A malformed or unfamiliar report may not contain the expected
            # score tokens.  Online mode still performs semantic retrieval
            # over bounded document chunks instead of silently skipping the
            # competition models.
            for paragraph in self.parsed.paragraphs:
                if not paragraph.text.strip():
                    continue
                evidence_id = self._ensure_paragraph_evidence(paragraph.index, paragraph.text, priority=72)
                candidates.append({"evidence_id": evidence_id, "text": paragraph.text[:700]})
                if len(candidates) >= 80:
                    break
            if not candidates:
                for table in self.parsed.tables:
                    for row_index, row in enumerate(table.rows):
                        text = " | ".join(row)
                        if not text.strip(" |"):
                            continue
                        evidence_id = self._ensure_table_evidence(table.index, row_index, text, priority=72)
                        candidates.append({"evidence_id": evidence_id, "text": text[:700]})
                        if len(candidates) >= 80:
                            break
                    if len(candidates) >= 80:
                        break
        if not candidates:
            raise RuntimeError("报告中没有可供在线语义审校的文本证据")

        query = (
            "复核桥梁名称、报告日期、本次总体 BCI/等级、桥面系 BCIm/等级、"
            "上部结构 BCIs/等级、下部结构 BCIx/等级、上一次总体评分/等级；"
            "区分部位 BCI、BSI 控制指标、权重、自相矛盾证据和报告内分级区间"
        )
        # Embedding performs broad recall; the VL reranker then resolves the
        # tighter field semantics.  Limit the corpus deterministically to keep
        # long reports bounded without selecting bridge-specific examples.
        candidates = candidates[:160]
        documents = [f"{item['evidence_id']} {item['text']}" for item in candidates]
        vectors = self.client.embed_texts(
            [query, *documents],
            task="summary-evidence-embedding",
        )
        if len(vectors) != len(documents) + 1:
            raise RuntimeError(
                f"Embedding 返回数量异常: expected={len(documents) + 1}, actual={len(vectors)}"
            )
        query_vector = vectors[0]
        query_norm = sqrt(sum(value * value for value in query_vector)) or 1.0
        recalled: list[tuple[float, int]] = []
        for index, vector in enumerate(vectors[1:]):
            vector_norm = sqrt(sum(value * value for value in vector)) or 1.0
            cosine = sum(left * right for left, right in zip(query_vector, vector)) / (query_norm * vector_norm)
            recalled.append((cosine, index))
        recalled.sort(key=lambda item: (-item[0], item[1]))
        recall_indices = [index for _, index in recalled[: min(36, len(recalled))]]
        recall_candidates = [candidates[index] for index in recall_indices]
        recall_documents = [documents[index] for index in recall_indices]
        reranked = self.client.rerank_documents(
            query=query,
            documents=recall_documents,
            top_n=min(12, len(recall_documents)),
            task="summary-evidence-rerank",
        )
        selected = [recall_candidates[item["index"]] for item in reranked if 0 <= item["index"] < len(recall_candidates)]
        if not selected:
            raise RuntimeError("Reranker 未返回可用的证据候选")

        # Pin the evidence already supporting BCI/BSI/grade fields plus one
        # representative per metric/side.  Retrieval still ranks relevance,
        # while the model is guaranteed to see the alternatives it must
        # arbitrate instead of receiving only the deterministic winner.
        pinned_ids: list[str] = list(rating_scale_evidence_ids)
        for scope in ("overall", "deck", "superstructure", "substructure", "previous"):
            entry = getattr(self.state.scores, scope)
            pinned_ids.extend([*entry.score.evidence, *entry.control_score.evidence, *entry.grade.evidence])
            representatives: dict[tuple[str, str | None], str] = {}
            for item in score_catalog[scope]:
                representatives[(str(item["metric"]).upper(), item.get("side"))] = str(item["evidence_id"])
            pinned_ids.extend(representatives.values())
        candidate_map = {item["evidence_id"]: item for item in candidates}
        selected_map: dict[str, dict[str, str]] = {}
        for evidence_id in dict.fromkeys(pinned_ids):
            if evidence_id in candidate_map:
                selected_map.setdefault(evidence_id, candidate_map[evidence_id])
        for item in selected:
            selected_map.setdefault(item["evidence_id"], item)
        # Pinned alternatives are the bounded decision set; reranked rows add
        # surrounding context.  Put pins first so an arbitrary top-N slice can
        # never discard one side of a left/right or BCI/BSI conflict.
        selected = list(selected_map.values())[:40]
        review_evidence_ids = {item["evidence_id"] for item in selected}

        current = {
            "bridge.name": self.state.bridge.name.value,
            "bridge.report_date": self.state.bridge.report_date.value,
            "scores.overall.score": self.state.scores.overall.score.value,
            "scores.overall.grade": self.state.scores.overall.grade.value,
            "scores.deck.score": self.state.scores.deck.score.value,
            "scores.deck.grade": self.state.scores.deck.grade.value,
            "scores.superstructure.score": self.state.scores.superstructure.score.value,
            "scores.superstructure.grade": self.state.scores.superstructure.grade.value,
            "scores.substructure.score": self.state.scores.substructure.score.value,
            "scores.substructure.grade": self.state.scores.substructure.grade.value,
            "scores.previous.score": self.state.scores.previous.score.value,
            "scores.previous.grade": self.state.scores.previous.grade.value,
        }
        reviewer = SummaryReviewer(
            self.client,
            self.settings,
            self.prompts.get("review_system", "只做证据支持的字段纠错，输出严格 JSON"),
        )
        side_coverage_gaps: list[str] = []
        for scope in ("deck", "superstructure", "substructure"):
            final_role = "final_component_rating_table"
            final_table_sides = {
                item.get("side")
                for item in score_catalog[scope]
                if item.get("source_role") == final_role
                and not item.get("self_contradictory")
            }
            current_value = str(current[f"scores.{scope}.score"] or "")
            if {"左幅", "右幅"}.issubset(final_table_sides) and not all(
                side in current_value for side in ("左幅", "右幅")
            ):
                side_coverage_gaps.append(f"scores.{scope}.score")
        decision_context = {
            "control_scores": {
                scope: getattr(self.state.scores, scope).control_score.model_dump(mode="json")
                for scope in ("overall", "deck", "superstructure", "substructure", "previous")
            },
            "ambiguous_scopes": sorted(ambiguous_scopes),
            "side_coverage_gaps": side_coverage_gaps,
            "score_candidates": {
                scope: [
                    {
                        "metric": item["metric"],
                        "metric_role": item["metric_role"],
                        "source_role": item["source_role"],
                        "score": item["score"],
                        "grade": item.get("grade"),
                        "grade_binding": item.get("grade_binding", False),
                        "side": item.get("side"),
                        "self_contradictory": item.get("self_contradictory", False),
                        "evidence_id": item["evidence_id"],
                    }
                    for item in rows[-12:]
                    if item["evidence_id"] in review_evidence_ids
                ]
                for scope, rows in score_catalog.items()
                if any(item["evidence_id"] in review_evidence_ids for item in rows)
            },
            "report_local_rating_scales": rating_scales,
            "instruction": (
                "候选集合只负责召回，不代表答案；按证据语义确定信息提取口径。"
                "评分与等级必须组成内部一致的最终评定；self_contradictory 候选不能单独裁决。"
                "报告摘要、审核后结论和最终汇总表的发布口径优先于中间计算段；"
                "BSI 与等级相邻本身不足以覆盖最终部位表的 component_index。"
                "side_coverage_gaps 是结构完整性缺口，不能把只含一个方向的标量误判为完整值。"
                "最终部位表与报告内分级区间可用于识别中间段落笔误；若原文左右幅分别评定，保留左右幅。"
            ),
        }
        review = reviewer.review(current=current, evidence=selected, decision_context=decision_context)
        requested_by_field: dict[str, dict[str, Any]] = {}
        skipped = []
        contradiction_rejections: list[dict[str, Any]] = []
        contradictory_evidence_ids = {
            str(item["evidence_id"])
            for rows in score_catalog.values()
            for item in rows
            if item.get("self_contradictory")
        }
        field_targets = {
            "bridge.name": self.state.bridge.name,
            "bridge.report_date": self.state.bridge.report_date,
            "scores.overall.score": self.state.scores.overall.score,
            "scores.overall.grade": self.state.scores.overall.grade,
            "scores.deck.score": self.state.scores.deck.score,
            "scores.deck.grade": self.state.scores.deck.grade,
            "scores.superstructure.score": self.state.scores.superstructure.score,
            "scores.superstructure.grade": self.state.scores.superstructure.grade,
            "scores.substructure.score": self.state.scores.substructure.score,
            "scores.substructure.grade": self.state.scores.substructure.grade,
            "scores.previous.score": self.state.scores.previous.score,
            "scores.previous.grade": self.state.scores.previous.grade,
        }
        def collect_patches(review_result, allowed_ids: set[str], phase: str) -> None:
            for patch in review_result.patches:
                field = patch.field
                value: Any = patch.value
                if field.endswith(".grade") and re.fullmatch(r"[ABCDE]", str(value), re.I):
                    value = f"{str(value).upper()}级"
                target = field_targets[field]
                evidence_ids = [item for item in patch.evidence_ids if item in allowed_ids]
                if not evidence_ids:
                    skipped.append({"field": field, "phase": phase, "reason": "引用了未提供给审校器的证据"})
                    continue
                evidence_text = " ".join(self.state.evidence[item].text for item in evidence_ids)
                if not self._value_supported(value, evidence_text):
                    skipped.append({"field": field, "phase": phase, "reason": "候选值不受引用证据支持"})
                    continue
                if not self._has_noncontradictory_support(value, evidence_ids, contradictory_evidence_ids):
                    rejection = {"field": field, "phase": phase, "reason": "自相矛盾证据不能单独支持事实改写"}
                    skipped.append(rejection)
                    contradiction_rejections.append({
                        **rejection,
                        "rejected_value": value,
                        "rejected_evidence_ids": evidence_ids,
                    })
                    continue
                consensus = self._repeated_final_table_consensus(field, score_catalog)
                if consensus is not None and not self._same_summary_value(value, consensus["value"]):
                    skipped.append({
                        "field": field,
                        "phase": phase,
                        "reason": "模型候选与至少两张相互独立且一致的最终评定表冲突",
                        "published_value": consensus["value"],
                        "published_evidence_ids": consensus["evidence_ids"],
                    })
                    continue
                if str(target.value) == str(value):
                    continue
                max_priority = max(self.state.evidence[item].priority for item in evidence_ids)
                scope = field.split(".")[1] if field.startswith("scores.") else ""
                if not self._fact_override_allowed(target, max_priority, semantic_conflict=scope in ambiguous_scopes):
                    skipped.append({"field": field, "phase": phase, "reason": "现有高置信事实证据不弱于模型候选"})
                    continue
                requested_by_field[field] = {"field": field, "value": value, "evidence_ids": evidence_ids}

        collect_patches(review, review_evidence_ids, "primary")
        warnings = list(review.warnings)
        reflection_used = False
        if contradiction_rejections:
            reflection_evidence = [
                item for item in selected
                if item["evidence_id"] not in contradictory_evidence_ids
            ]
            reflection_ids = {item["evidence_id"] for item in reflection_evidence}
            if reflection_evidence:
                reflection_fields = list(dict.fromkeys([
                    *(item["field"] for item in contradiction_rejections),
                    *side_coverage_gaps,
                ]))
                reflection_context = {
                    **decision_context,
                    "score_candidates": {
                        scope: [item for item in rows if not item.get("self_contradictory")]
                        for scope, rows in decision_context["score_candidates"].items()
                    },
                    "rejected_primary_patches": contradiction_rejections,
                    "instruction": (
                        "主审校器尝试用 self_contradictory 证据改写字段，已被写入门拒绝。"
                        "本轮看不到矛盾证据；只可依据最终部位表、最终结论及报告内分级表提出替代 patch，"
                        "找不到充分证据就保持当前值。"
                    ),
                }
                reflection = reviewer.review(
                    current={field: current[field] for field in reflection_fields},
                    evidence=reflection_evidence,
                    decision_context=reflection_context,
                    task="summary-conflict-review",
                    instruction=(
                        f"只复核这些已确认存在结构缺口或矛盾的字段：{reflection_fields}。"
                        "side_coverage_gaps 中的当前标量已被系统判定为不完整，不能以‘数值本身正确’为由保持；"
                        "若非矛盾的最终左右幅表分别给出数值，必须返回保留左右幅的 patch。"
                        "不得处理列表外字段，也不得引用 self_contradictory 证据。"
                    ),
                )
                reflection_used = True
                warnings.extend(reflection.warnings)
                collect_patches(reflection, reflection_ids, "reflection")

        requested = list(requested_by_field.values())

        semantic_conflict_fields = {
            item["field"]
            for item in requested
            if item["field"].startswith("scores.")
            and item["field"].split(".")[1] in ambiguous_scopes
        }
        applied, used_evidence = self._apply_review_fact_patches(
            requested,
            semantic_conflict_fields=semantic_conflict_fields,
            skipped=skipped,
        )
        structural_patches = self._structural_side_completion_patches(
            score_catalog,
            side_coverage_gaps,
        )
        structural_completion_applied: list[str] = []
        if structural_patches:
            structural_completion_applied, structural_evidence = self._apply_review_fact_patches(
                structural_patches,
                semantic_conflict_fields={item["field"] for item in structural_patches},
                skipped=skipped,
                phase="structural_completion",
            )
            applied.extend(item for item in structural_completion_applied if item not in applied)
            used_evidence.extend(item for item in structural_evidence if item not in used_evidence)
        component_grade_patches = self._selected_component_grade_patches(score_catalog)
        component_grade_patches = [
            item for item in component_grade_patches
            if self._published_summary_patch_repairs_gap(field_targets[item["field"]], item["value"])
        ]
        component_grade_applied: list[str] = []
        if component_grade_patches:
            component_grade_applied, component_grade_evidence = self._apply_review_fact_patches(
                component_grade_patches,
                semantic_conflict_fields={item["field"] for item in component_grade_patches},
                skipped=skipped,
                phase="component_grade",
            )
            applied.extend(item for item in component_grade_applied if item not in applied)
            used_evidence.extend(item for item in component_grade_evidence if item not in used_evidence)
        mirrored_bsi_evidence_ids = self._bsi_mirrors_component_summary_ids(score_catalog)
        control_grade_patches = self._unambiguous_control_grade_patches(
            score_catalog,
            excluded_evidence_ids=mirrored_bsi_evidence_ids,
        )
        control_grade_applied: list[str] = []
        if control_grade_patches:
            control_grade_applied, control_grade_evidence = self._apply_review_fact_patches(
                control_grade_patches,
                semantic_conflict_fields={item["field"] for item in control_grade_patches},
                skipped=skipped,
                phase="control_grade",
            )
            applied.extend(item for item in control_grade_applied if item not in applied)
            used_evidence.extend(item for item in control_grade_evidence if item not in used_evidence)
        divergent_overall_patches = self._divergent_overall_side_patches()
        divergent_overall_applied: list[str] = []
        if divergent_overall_patches:
            divergent_overall_applied, divergent_overall_evidence = self._apply_review_fact_patches(
                divergent_overall_patches,
                semantic_conflict_fields={item["field"] for item in divergent_overall_patches},
                skipped=skipped,
                phase="divergent_overall",
            )
            applied.extend(item for item in divergent_overall_applied if item not in applied)
            used_evidence.extend(item for item in divergent_overall_evidence if item not in used_evidence)
        published_summary_patches = self._published_summary_metric_patches(score_catalog)
        published_summary_patches = [
            item for item in published_summary_patches
            if self._published_summary_patch_allowed(
                item["field"],
                field_targets[item["field"]],
                item["value"],
            )
        ]
        published_summary_applied: list[str] = []
        if published_summary_patches:
            published_summary_applied, published_summary_evidence = self._apply_review_fact_patches(
                published_summary_patches,
                semantic_conflict_fields={item["field"] for item in published_summary_patches},
                skipped=skipped,
                phase="published_summary",
            )
            applied.extend(item for item in published_summary_applied if item not in applied)
            used_evidence.extend(item for item in published_summary_evidence if item not in used_evidence)
        return {
            "reviewed_fields": len(current),
            "evidence_candidates": len(candidates),
            "embedded_evidence": len(documents),
            "recalled_evidence": len(recall_candidates),
            "reranked_evidence": len(selected),
            "ambiguous_scopes": sorted(ambiguous_scopes),
            "applied": applied,
            "skipped": skipped,
            "warnings": warnings,
            "reflection_used": reflection_used,
            "structural_completion_applied": structural_completion_applied,
            "component_grade_applied": component_grade_applied,
            "mirrored_bsi_evidence_ids": sorted(mirrored_bsi_evidence_ids),
            "control_grade_applied": control_grade_applied,
            "divergent_overall_applied": divergent_overall_applied,
            "published_summary_applied": published_summary_applied,
            "validation_passed": self.state.validation.passed,
        }, used_evidence

    def _apply_review_fact_patches(
        self,
        patches: list[dict[str, Any]],
        *,
        semantic_conflict_fields: set[str],
        skipped: list[dict[str, Any]],
        phase: str = "write_gate",
    ) -> tuple[list[str], list[str]]:
        """Apply reviewer patches independently so one bad proposal is isolated.

        The model may return several otherwise independent field proposals in
        one structured response.  The evidence write gate must still reject an
        unsupported value, but that rejection is not a network/tool failure and
        must not discard other grounded proposals or strand the Agent Loop.
        """

        applied: list[str] = []
        used_evidence: list[str] = []
        for patch in patches:
            field = str(patch.get("field") or "")
            try:
                result, evidence_ids = self._apply_fact_patches(
                    [patch],
                    semantic_conflict_fields={field} if field in semantic_conflict_fields else set(),
                )
            except ValueError as exc:
                skipped.append({
                    "field": field,
                    "phase": phase,
                    "reason": str(exc),
                })
                continue
            applied.extend(item for item in result["applied"] if item not in applied)
            used_evidence.extend(item for item in evidence_ids if item not in used_evidence)
        if not patches:
            self.state.validation = self.validator.validate(self.state)
        return applied, used_evidence

    def _get_visual_candidates(self):
        if self._visual_candidates is None:
            self._visual_candidates = extract_visual_candidates(self.parsed.path, limit=3)
        return self._visual_candidates

    def _critical_visual_gap(self) -> bool:
        always_critical = [
            self.state.bridge.name,
            self.state.bridge.report_date,
            self.state.scores.overall.score,
            self.state.scores.overall.grade,
        ]
        if any(field.value in (None, "", "无") or field.conflicts for field in always_critical):
            return True

        # Some valid reports provide only a whole-bridge score. A missing
        # component score is a visual gap only if the extracted text signals
        # that this component metric should exist; otherwise hundreds of
        # ordinary defect photographs would trigger needless OCR calls.
        text = self.parsed.full_text
        component_groups = [
            ((self.state.scores.deck.score, self.state.scores.deck.grade), ("BCIm", "桥面系评分", "桥面系等级")),
            ((self.state.scores.superstructure.score, self.state.scores.superstructure.grade), ("BCIs", "上部结构评分", "上部结构等级")),
            ((self.state.scores.substructure.score, self.state.scores.substructure.grade), ("BCIx", "下部结构评分", "下部结构等级")),
        ]
        return any(
            any(field.value in (None, "", "无") or field.conflicts for field in fields)
            and any(token in text for token in signals)
            for fields, signals in component_groups
        )

    def _inspect_visual_evidence(self, _: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        candidates = self._get_visual_candidates()
        if not candidates:
            return {"candidates": 0, "verified_images": 0, "facts": [], "warnings": ["无合格图片候选"]}, []
        reviewer = VisionEvidenceReviewer(
            self.client,
            self.settings or Settings(llm_mode="off"),
            self.prompts.get("vision_system", "只抄录图中明确可见事实，输出严格 JSON"),
        )
        evidence_ids: list[str] = []
        facts: list[dict[str, str]] = []
        warnings: list[str] = []
        for candidate in candidates:
            review = reviewer.review(candidate.data_url)
            warnings.extend(review.warnings)
            if not review.verified_facts:
                continue
            evidence_id = f"IMG:{candidate.digest[:16]}"
            fact_text = "；".join(
                f"{fact.field}={fact.value}（图中文字：{fact.evidence_quote}）"
                for fact in review.verified_facts
            )
            self.state.add_evidence(EvidenceRef(
                id=evidence_id,
                kind="image",
                text=f"双视觉模型交叉一致的 OCR 事实：{fact_text}",
                source_path=f"{self.parsed.path}#{candidate.media_path}",
                priority=68,
            ))
            evidence_ids.append(evidence_id)
            facts.extend({
                "evidence_id": evidence_id,
                "field": fact.field,
                "value": fact.value,
            } for fact in review.verified_facts)
        return {
            "candidates": len(candidates),
            "verified_images": len(evidence_ids),
            "facts": facts,
            "warnings": warnings,
        }, evidence_ids

    @staticmethod
    def _bounded_review_indices(
        items: list[Any],
        uncertain: Callable[[Any], bool],
        limit: int | None,
    ) -> list[int]:
        if not items or (limit is not None and limit <= 0):
            return []
        uncertain_indices = [index for index, item in enumerate(items) if uncertain(item)]
        if limit is None:
            # The model is used as an exception handler, so review every row
            # that the deterministic pipeline itself marked uncertain.  A hard
            # first-N/sampling cap leaves most long bridge ledgers untouched.
            return uncertain_indices
        if len(uncertain_indices) >= limit:
            # Long photo ledgers can contain hundreds of uncertain rows.  A
            # first-N slice reviews only the first pages, so sample uniformly
            # over the uncertain set while remaining deterministic.
            return [
                uncertain_indices[min(len(uncertain_indices) - 1, (offset * len(uncertain_indices)) // limit)]
                for offset in range(limit)
            ]
        selected = uncertain_indices
        selected_set = set(selected)
        remaining = [index for index in range(len(items)) if index not in selected_set]
        slots = limit - len(selected)
        if len(remaining) <= slots:
            return [*selected, *remaining]
        # Fill the rest with a deterministic, evenly spaced sample so long
        # ledgers are reviewed across the whole report rather than only at the
        # first page.  This is format-general and never depends on bridge name.
        sampled = [remaining[min(len(remaining) - 1, (offset * len(remaining)) // slots)] for offset in range(slots)]
        return list(dict.fromkeys([*selected, *sampled]))

    @staticmethod
    def _sample_index_pool(indices: list[int], limit: int) -> list[int]:
        """Uniformly sample an already-ranked pool without a first-page bias."""
        if limit <= 0 or not indices:
            return []
        if len(indices) <= limit:
            return indices
        return [
            indices[min(len(indices) - 1, (offset * len(indices)) // limit)]
            for offset in range(limit)
        ]

    def _detail_item_evidence(self, evidence_ids: list[str]) -> tuple[list[str], list[dict[str, str]]]:
        # Detail review is a semantic exception handler, not a second copy of
        # the whole report.  Four compact, row-linked snippets keep prompts
        # below gateway latency cliffs while preserving the current-row,
        # historical and reranked evidence needed for a grounded patch.
        allowed = [item for item in dict.fromkeys(evidence_ids) if item in self.state.evidence][:4]
        evidence = [
            {"evidence_id": item, "text": self.state.evidence[item].text[:420]}
            for item in allowed
        ]
        return allowed, evidence

    @staticmethod
    def _detail_relevance_tokens(text: str) -> set[str]:
        value = str(text or "")
        tokens = {term for term in DETAIL_LINK_TERMS if term in value}
        tokens.update(re.findall(r"(?:\d+[-—~～至])?\d+#?(?:跨|梁|墩|台|支座|索)", value))
        return tokens

    def _retrieve_detail_evidence(
        self,
        disease_indices: list[int],
        recommendation_indices: list[int],
    ) -> tuple[dict[int, list[str]], dict[int, list[str]], dict[str, int], list[str]]:
        """Recall and rerank cross-row evidence for the detail-review action."""
        assert self.client is not None
        signal = re.compile(
            r"上一次|上次|上一年度|历史|新增|新发现|发展|加重|加剧|稳定|"
            r"修复|修补|建议|养护|处治|维修|风险|结论|检测结果"
        )
        candidates: dict[str, dict[str, str]] = {}
        for evidence_id, evidence in self.state.evidence.items():
            if evidence_id.startswith("HISTORY:"):
                candidates[evidence_id] = {"evidence_id": evidence_id, "text": evidence.text[:700]}
                if len(candidates) >= 180:
                    break
        for evidence_id, evidence in self.state.evidence.items():
            if signal.search(evidence.text):
                candidates[evidence_id] = {"evidence_id": evidence_id, "text": evidence.text[:700]}
                if len(candidates) >= 180:
                    break
        if len(candidates) < 180:
            for paragraph in self.parsed.paragraphs:
                if not signal.search(paragraph.text):
                    continue
                evidence_id = self._ensure_paragraph_evidence(paragraph.index, paragraph.text, priority=82)
                candidates[evidence_id] = {"evidence_id": evidence_id, "text": paragraph.text[:700]}
                if len(candidates) >= 180:
                    break
        if len(candidates) < 180:
            for table in self.parsed.tables:
                for row_index, row in enumerate(table.rows):
                    text = " | ".join(row)
                    if not signal.search(text):
                        continue
                    evidence_id = self._ensure_table_evidence(table.index, row_index, text, priority=84)
                    candidates[evidence_id] = {"evidence_id": evidence_id, "text": text[:700]}
                    if len(candidates) >= 180:
                        break
                if len(candidates) >= 180:
                    break
        if not candidates:
            # The semantic stages remain observable even on terse reports.
            for paragraph in self.parsed.paragraphs:
                if not paragraph.text.strip():
                    continue
                evidence_id = self._ensure_paragraph_evidence(paragraph.index, paragraph.text, priority=70)
                candidates[evidence_id] = {"evidence_id": evidence_id, "text": paragraph.text[:700]}
                if len(candidates) >= 60:
                    break
        if not candidates:
            for table in self.parsed.tables:
                for row_index, row in enumerate(table.rows):
                    text = " | ".join(row)
                    if not text.strip(" |"):
                        continue
                    evidence_id = self._ensure_table_evidence(table.index, row_index, text, priority=70)
                    candidates[evidence_id] = {"evidence_id": evidence_id, "text": text[:700]}
                    if len(candidates) >= 60:
                        break
                if len(candidates) >= 60:
                    break
        if not candidates:
            raise RuntimeError("报告中没有可供明细语义召回的文本证据")

        disease_query = [
            f"病害{index}:{self.state.diseases[index].component}|{self.state.diseases[index].location}|{self.state.diseases[index].type}"
            for index in disease_indices
        ]
        recommendation_query = [
            f"建议{index}:{self.state.recommendations[index].component}|{self.state.recommendations[index].content[:80]}"
            for index in recommendation_indices
        ]
        query = (
            "逐行复核病害是否新增、上一次定检状态、发展程度以及建议类别和精确部位；"
            "优先历史对比表、检测结论和养护建议。\n"
            + "\n".join([*disease_query, *recommendation_query])[:6000]
        )
        candidate_list = list(candidates.values())
        documents = [f"{item['evidence_id']} {item['text']}" for item in candidate_list]
        vectors = self.client.embed_texts([query, *documents], task="detail-evidence-embedding")
        if len(vectors) != len(documents) + 1:
            raise RuntimeError(
                f"明细 Embedding 返回数量异常: expected={len(documents) + 1}, actual={len(vectors)}"
            )
        query_vector = vectors[0]
        query_norm = sqrt(sum(value * value for value in query_vector)) or 1.0
        recalled: list[tuple[float, int]] = []
        for index, vector in enumerate(vectors[1:]):
            vector_norm = sqrt(sum(value * value for value in vector)) or 1.0
            cosine = sum(left * right for left, right in zip(query_vector, vector)) / (query_norm * vector_norm)
            recalled.append((cosine, index))
        recalled.sort(key=lambda item: (-item[0], item[1]))
        recall_indices = [index for _, index in recalled[: min(36, len(recalled))]]
        recall_candidates = [candidate_list[index] for index in recall_indices]
        recall_documents = [documents[index] for index in recall_indices]
        reranked = self.client.rerank_documents(
            query=query,
            documents=recall_documents,
            top_n=min(16, len(recall_documents)),
            task="detail-evidence-rerank",
        )
        selected = [
            recall_candidates[item["index"]]
            for item in reranked
            if 0 <= item["index"] < len(recall_candidates)
        ]
        if not selected:
            raise RuntimeError("明细 Reranker 未返回可用证据候选")

        def links_for(text: str) -> list[str]:
            tokens = self._detail_relevance_tokens(text)
            if not tokens:
                return []
            return [
                item["evidence_id"]
                for item in selected
                if tokens & self._detail_relevance_tokens(item["text"])
            ][:3]

        disease_links = {
            index: links_for(
                f"{self.state.diseases[index].component} {self.state.diseases[index].location} "
                f"{self.state.diseases[index].type} {self.state.diseases[index].description}"
            )
            for index in disease_indices
        }
        recommendation_links = {
            index: links_for(
                f"{self.state.recommendations[index].component} {self.state.recommendations[index].content}"
            )
            for index in recommendation_indices
        }
        return (
            disease_links,
            recommendation_links,
            {
                "semantic_candidates": len(candidate_list),
                "semantic_recalled": len(recall_candidates),
                "semantic_reranked": len(selected),
                "semantic_linked_rows": sum(bool(value) for value in disease_links.values())
                + sum(bool(value) for value in recommendation_links.values()),
            },
            [item["evidence_id"] for item in selected],
        )

    @staticmethod
    def _numbers_supported(value: str, evidence_text: str) -> bool:
        numbers = re.findall(r"\d+(?:\.\d+)?", str(value or ""))
        evidence_numbers = re.findall(r"\d+(?:\.\d+)?", evidence_text)
        return all(
            any(abs(float(number) - float(candidate)) <= 0.011 for candidate in evidence_numbers)
            for number in numbers
        )

    @classmethod
    def _literal_text_supported(cls, value: str, evidence_text: str) -> bool:
        if not value or not cls._numbers_supported(value, evidence_text):
            return False
        normalize = lambda text: re.sub(r"[\s，,。；;：:（）()、|]", "", str(text or ""))
        rendered = normalize(value)
        evidence = normalize(evidence_text)
        return bool(rendered and rendered in evidence)

    @staticmethod
    def _category_supported(category: str, evidence_text: str) -> bool:
        if category == "立即处置":
            return bool(re.search(r"立即|立刻|紧急|封闭|限行|停止使用|交通管制|应急", evidence_text))
        if category == "预防性养护":
            return bool(re.search(r"预防性|日常养护|定期|观察|观测|监测|加强检查|保持", evidence_text))
        return bool(re.search(r"尽快|及时|修复|维修|处治|处理|更换|清理|增设|完善", evidence_text))

    def _review_details(self, _: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        if not self.client or not self.settings:
            raise RuntimeError("模型不可用，不能执行 review_details")

        # Novelty/history fields are the weakest detail dimension.  Review
        # unresolved rows first across the complete ledger, then spend any
        # remaining bounded budget on lower-confidence deterministic calls.
        # This keeps the model an exception handler while avoiding a first-18
        # blind spot on long bridge reports.
        primary_indices = [
            index
            for index, item in enumerate(self.state.diseases)
            if (
                item.is_new is None
                or item.previous_state in (None, "", "未明确", "未提及")
                or item.development in (None, "", "未明确", "未提及")
                or item.component == "其他"
                or item.type == "未明确"
                or not item.location
                or not item.description
            )
        ]
        # Formal reports currently stay below 100 uncertain rows, but use a
        # generous evidence-bounded ceiling for unusually long photo ledgers.
        # Each model request still contains only six rows, so reviewing the
        # whole ordinary ledger does not enlarge a single prompt.
        review_ceiling = 120
        disease_indices = self._sample_index_pool(primary_indices, review_ceiling)
        if len(disease_indices) < review_ceiling:
            selected_set = set(disease_indices)
            secondary_indices = [
                index
                for index, item in enumerate(self.state.diseases)
                if index not in selected_set
                and item.novelty_source in {"history_absence", "current_report_no_new"}
            ]
            disease_indices.extend(
                self._sample_index_pool(secondary_indices, review_ceiling - len(disease_indices))
            )
        recommendation_indices = self._bounded_review_indices(
            self.state.recommendations,
            lambda item: (
                item.component in {"", "全桥", "其他", "未明确"}
                or not item.content
                or (
                    item.category == "预防性养护"
                    and bool(re.search(r"尽快|及时|修复|维修|处治|处理|更换", item.content))
                )
            ),
            12,
        )
        try:
            disease_links, recommendation_links, semantic_stats, semantic_evidence_ids = self._retrieve_detail_evidence(
                disease_indices,
                recommendation_indices,
            )
        except Exception as exc:
            disease_links = {}
            recommendation_links = {}
            semantic_evidence_ids = []
            semantic_stats = {
                "semantic_candidates": 0,
                "semantic_recalled": 0,
                "semantic_reranked": 0,
                "semantic_linked_rows": 0,
                "semantic_fallback": type(exc).__name__,
            }

        disease_payloads: list[dict[str, Any]] = []
        disease_allowed: dict[int, set[str]] = {}
        for index in disease_indices:
            item = self.state.diseases[index]
            allowed, evidence = self._detail_item_evidence([*item.evidence, *disease_links.get(index, [])])
            if not allowed:
                continue
            disease_allowed[index] = set(allowed)
            disease_payloads.append({
                "index": index,
                "current": {
                    "location": item.location,
                    "disease_type": item.type,
                    "description": item.description,
                    "is_new": item.is_new,
                    "previous_state": item.previous_state,
                    "development": item.development,
                },
                "allowed_evidence_ids": allowed,
                "evidence": evidence,
            })

        recommendation_payloads: list[dict[str, Any]] = []
        recommendation_allowed: dict[int, set[str]] = {}
        for index in recommendation_indices:
            item = self.state.recommendations[index]
            allowed, evidence = self._detail_item_evidence([*item.evidence, *recommendation_links.get(index, [])])
            if not allowed:
                continue
            recommendation_allowed[index] = set(allowed)
            recommendation_payloads.append({
                "index": index,
                "current": {
                    "category": item.category,
                    "component": item.component,
                    "content": item.content,
                },
                "allowed_evidence_ids": allowed,
                "evidence": evidence,
            })

        reviewer = DetailReviewer(
            self.client,
            self.settings,
            self.prompts.get("detail_system", "只做证据支持的病害和建议字段审校，输出严格 JSON"),
        )
        reviews = []
        model_failures: list[dict[str, Any]] = []
        # Keep each typed request small.  The previous 16/8 row batches could
        # exceed the text gateway's 90-second budget on evidence-rich reports.
        # Uniform sampling above still covers the complete ledger, so reducing
        # batch width is a latency bound rather than filename-specific fitting.
        disease_batch_size = 8
        recommendation_batch_size = 4
        batch_count = max(
            1,
            (len(disease_payloads) + disease_batch_size - 1) // disease_batch_size,
            (len(recommendation_payloads) + recommendation_batch_size - 1) // recommendation_batch_size,
        )
        for batch_index in range(batch_count):
            try:
                reviews.append(reviewer.review(
                    diseases=disease_payloads[
                        batch_index * disease_batch_size:(batch_index + 1) * disease_batch_size
                    ],
                    recommendations=recommendation_payloads[
                        batch_index * recommendation_batch_size:(batch_index + 1) * recommendation_batch_size
                    ],
                ))
            except Exception as exc:
                model_failures.append({
                    "phase": "primary",
                    "batch": batch_index,
                    "error": type(exc).__name__,
                })

        # Structured models occasionally omit a single difficult temporal
        # row even though the batch prompt requires a binary decision.  Retry
        # only those omitted rows, never the whole ledger.  The retry remains
        # evidence-bounded and model-decided; it does not invent a default
        # novelty value or broadcast a section-level rule.
        returned_novelty_indices: set[int] = set()
        for review in reviews:
            for patch in review.disease_patches:
                if "is_new" not in patch.model_fields_set or patch.is_new is None:
                    continue
                allowed_ids = disease_allowed.get(patch.index, set())
                evidence_ids = [item for item in patch.evidence_ids if item in allowed_ids]
                evidence_text = " ".join(
                    self.state.evidence[item].text
                    for item in evidence_ids
                    if item in self.state.evidence
                )
                history_ids = [item for item in evidence_ids if item.startswith("HISTORY:")]
                supported_new = (
                    any(":ABSENCE:" in item for item in history_ids)
                    or bool(re.search(r"新增|新发现|本次新", evidence_text))
                )
                supported_existing = (
                    any(re.search(r":D\d+$", item) for item in history_ids)
                    or bool(re.search(r"既有|原有|上次|上一", evidence_text))
                )
                if (patch.is_new and supported_new) or (not patch.is_new and supported_existing):
                    returned_novelty_indices.add(patch.index)
        omitted_temporal_payloads = [] if model_failures else [
            payload
            for payload in disease_payloads
            if payload["current"]["is_new"] is None
            and payload["index"] not in returned_novelty_indices
            and re.search(
                r"上一次检测结果|上次|上一年度|历史病害|既有病害|新增|本次检测结果",
                " ".join(item["text"] for item in payload["evidence"]),
            )
        ]
        for offset in range(0, len(omitted_temporal_payloads), disease_batch_size):
            try:
                reviews.append(reviewer.review(
                    diseases=omitted_temporal_payloads[offset:offset + disease_batch_size],
                    recommendations=[],
                    force_temporal_decision=True,
                ))
            except Exception as exc:
                model_failures.append({
                    "phase": "temporal_retry",
                    "batch": offset // disease_batch_size,
                    "error": type(exc).__name__,
                })

        applied: list[str] = []
        skipped: list[dict[str, Any]] = []
        used_evidence: list[str] = []
        for review in reviews:
            for patch in review.disease_patches:
                if patch.index not in disease_allowed:
                    skipped.append({"kind": "disease", "index": patch.index, "reason": "行不在审校批次"})
                    continue
                evidence_ids = [item for item in patch.evidence_ids if item in disease_allowed[patch.index]]
                if not evidence_ids:
                    skipped.append({"kind": "disease", "index": patch.index, "reason": "未引用该行允许的证据"})
                    continue
                item = self.state.diseases[patch.index]
                evidence_text = " ".join(self.state.evidence[evidence_id].text for evidence_id in evidence_ids)
                history_ids = [evidence_id for evidence_id in evidence_ids if evidence_id.startswith("HISTORY:")]
                has_temporal_evidence = bool(history_ids) or bool(re.search(
                    r"上一次检测结果|上次|上一年度|历史病害|既有病害",
                    evidence_text,
                ))
                changed = False

                if patch.location and patch.location != item.location:
                    if self._literal_text_supported(patch.location, evidence_text):
                        item.location = patch.location
                        applied.append(f"diseases[{patch.index}].location")
                        changed = True
                if patch.disease_type and patch.disease_type != item.type:
                    if self._literal_text_supported(patch.disease_type, evidence_text):
                        item.type = patch.disease_type
                        applied.append(f"diseases[{patch.index}].type")
                        changed = True
                if patch.description and patch.description != item.description:
                    if self._literal_text_supported(patch.description, evidence_text):
                        item.description = patch.description
                        applied.append(f"diseases[{patch.index}].description")
                        changed = True
                replaceable_novelty = (
                    item.is_new is None
                    or item.novelty_source in {"history_absence", "current_report_no_new"}
                )
                if (
                    "is_new" in patch.model_fields_set
                    and patch.is_new is not None
                    and patch.is_new != item.is_new
                    and replaceable_novelty
                ):
                    supported_new = (
                        any(":ABSENCE:" in evidence_id for evidence_id in history_ids)
                        or bool(re.search(r"新增|新发现|本次新", evidence_text))
                    )
                    supported_existing = (
                        any(re.search(r":D\d+$", evidence_id) for evidence_id in history_ids)
                        or bool(re.search(r"既有|原有|上次|上一", evidence_text))
                    )
                    if (patch.is_new and supported_new) or (not patch.is_new and supported_existing):
                        item.is_new = patch.is_new
                        item.novelty_source = "model"
                        applied.append(f"diseases[{patch.index}].is_new")
                        coherent_previous = "未发现" if patch.is_new else "已发现"
                        if item.previous_state != coherent_previous:
                            item.previous_state = coherent_previous
                            applied.append(f"diseases[{patch.index}].previous_state")
                        if patch.is_new and item.development != "无":
                            item.development = "无"
                            applied.append(f"diseases[{patch.index}].development")
                        changed = True
                if patch.previous_state and item.previous_state in (None, "", "未明确", "未提及") and has_temporal_evidence:
                    if patch.previous_state in {"已发现", "未发现"}:
                        item.previous_state = patch.previous_state
                        applied.append(f"diseases[{patch.index}].previous_state")
                        changed = True
                if patch.development and item.development in (None, "", "未明确", "未提及") and has_temporal_evidence:
                    if patch.development in {"无", "轻微", "严重"}:
                        item.development = patch.development
                        applied.append(f"diseases[{patch.index}].development")
                        changed = True
                if changed:
                    item.confidence = max(item.confidence, 0.90)
                used_evidence.extend(evidence_ids)

            for patch in review.recommendation_patches:
                if patch.index not in recommendation_allowed:
                    skipped.append({"kind": "recommendation", "index": patch.index, "reason": "行不在审校批次"})
                    continue
                evidence_ids = [item for item in patch.evidence_ids if item in recommendation_allowed[patch.index]]
                if not evidence_ids:
                    skipped.append({"kind": "recommendation", "index": patch.index, "reason": "未引用该行允许的证据"})
                    continue
                item = self.state.recommendations[patch.index]
                evidence_text = " ".join(self.state.evidence[evidence_id].text for evidence_id in evidence_ids)
                changed = False
                if patch.category and patch.category != item.category and self._category_supported(patch.category, evidence_text):
                    item.category = patch.category
                    applied.append(f"recommendations[{patch.index}].category")
                    changed = True
                if patch.component and patch.component != item.component:
                    if self._literal_text_supported(patch.component, evidence_text):
                        item.component = patch.component
                        applied.append(f"recommendations[{patch.index}].component")
                        changed = True
                if patch.content and patch.content != item.content and self._literal_text_supported(patch.content, evidence_text):
                    item.content = patch.content
                    applied.append(f"recommendations[{patch.index}].content")
                    changed = True
                if changed:
                    item.confidence = max(item.confidence, 0.90)
                used_evidence.extend(evidence_ids)

        # ``is_new`` and ``previous_state`` encode the same binary temporal
        # relation from opposite directions.  If the evidence-gated reviewer
        # accepted one side but omitted the other, close the pair logically
        # instead of spending another unconstrained guess or leaving a blank.
        for index, item in enumerate(self.state.diseases):
            if item.is_new is not None:
                continue
            if item.previous_state == "已发现":
                item.is_new = False
            elif item.previous_state == "未发现":
                item.is_new = True
                if item.development in (None, "", "未明确", "未提及"):
                    item.development = "无"
                    applied.append(f"diseases[{index}].development")
            else:
                continue
            item.novelty_source = "model"
            item.confidence = max(item.confidence, 0.90)
            applied.append(f"diseases[{index}].is_new")

        self.state.validation = self.validator.validate(self.state)
        return {
            "reviewed_diseases": len(disease_payloads),
            "reviewed_recommendations": len(recommendation_payloads),
            "uncertain_diseases": len(disease_indices),
            "uncertain_recommendations": len(recommendation_indices),
            "unreviewed_uncertain_diseases": len(disease_indices) - len(disease_payloads),
            "unreviewed_uncertain_recommendations": len(recommendation_indices) - len(recommendation_payloads),
            "model_batches": len(reviews),
            "model_failures": model_failures,
            "applied": list(dict.fromkeys(applied)),
            "skipped": skipped,
            "validation_passed": self.state.validation.passed,
            **semantic_stats,
        }, list(dict.fromkeys([*semantic_evidence_ids, *used_evidence]))

    def _apply_fact_patch(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        patches = arguments.get("patches")
        if not isinstance(patches, list) or not patches:
            raise ValueError("patches 必须是非空列表")
        # Direct Planner patches never receive semantic-conflict privileges;
        # only the bounded SummaryReviewer path can call the internal helper
        # with conflict context after retrieval and evidence validation.
        return self._apply_fact_patches(patches)

    def _apply_fact_patches(
        self,
        patches: list[dict[str, Any]],
        *,
        semantic_conflict_fields: set[str] | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        semantic_conflict_fields = semantic_conflict_fields or set()
        field_map = {
            "bridge.name": self.state.bridge.name,
            "bridge.report_date": self.state.bridge.report_date,
            "scores.overall.score": self.state.scores.overall.score,
            "scores.overall.grade": self.state.scores.overall.grade,
            "scores.deck.score": self.state.scores.deck.score,
            "scores.deck.grade": self.state.scores.deck.grade,
            "scores.superstructure.score": self.state.scores.superstructure.score,
            "scores.superstructure.grade": self.state.scores.superstructure.grade,
            "scores.substructure.score": self.state.scores.substructure.score,
            "scores.substructure.grade": self.state.scores.substructure.grade,
            "scores.previous.score": self.state.scores.previous.score,
            "scores.previous.grade": self.state.scores.previous.grade,
        }
        prepared: list[tuple[str, FieldValue, Any, list[str], int]] = []
        used_evidence: list[str] = []
        for patch in patches:
            field_name = str(patch.get("field") or "")
            target = field_map.get(field_name)
            if target is None:
                raise ValueError(f"不允许 patch 字段: {field_name}")
            value = patch.get("value")
            evidence_ids = [str(item) for item in patch.get("evidence_ids") or []]
            if not evidence_ids or any(item not in self.state.evidence for item in evidence_ids):
                raise ValueError(f"{field_name} 引用了不存在的证据")
            texts = " ".join(self.state.evidence[item].text for item in evidence_ids)
            if not self._field_semantics_supported(field_name, value, evidence_ids):
                raise ValueError(f"{field_name} 的证据未在同一语义分句中明确指向上一次评定")
            if not self._value_supported(value, texts):
                raise ValueError(f"{field_name} 的值不受引用证据支持: {value}")
            priority = max(self.state.evidence[item].priority for item in evidence_ids)
            if not self._fact_override_allowed(
                target,
                priority,
                semantic_conflict=field_name in semantic_conflict_fields,
            ):
                raise ValueError(f"{field_name} 现有高置信事实证据不弱于候选，拒绝覆盖")
            prepared.append((field_name, target, value, evidence_ids, priority))
            used_evidence.extend(evidence_ids)
        applied = []
        for field_name, target, value, evidence_ids, priority in prepared:
            previous = target.value
            target.value = value
            target.evidence = evidence_ids
            target.source_priority = priority
            target.confidence = 0.92
            if previous not in (None, "", value):
                target.conflicts = list(dict.fromkeys(target.conflicts + [str(previous)]))
            applied.append(field_name)
        self.state.validation = self.validator.validate(self.state)
        return {"applied": applied, "validation_passed": self.state.validation.passed}, list(dict.fromkeys(used_evidence))

    def _field_semantics_supported(
        self,
        field_name: str,
        value: Any,
        evidence_ids: list[str],
    ) -> bool:
        """Require previous-cycle values to be anchored to previous-cycle text.

        A long conclusion row can mention that previous defects were repaired
        and later print the *current* BCI.  Numeric occurrence alone must not
        let a reviewer copy that current value into ``scores.previous``.  The
        value has to occur in the same sentence/semicolon clause as an
        explicit prior-cycle marker, or come from the chronology-safe history
        summary / filename's explicit original grade.
        """
        if field_name not in {"scores.previous.score", "scores.previous.grade"}:
            return True
        previous_signal = re.compile(
            r"上一次|上次|上一年度|上一轮|前次|历史(?:评分|评定|检测)?|"
            r"原(?:评定|等级|技术状况|总体|评分|BCI|Dr)",
            re.I,
        )
        for evidence_id in evidence_ids:
            evidence = self.state.evidence[evidence_id]
            if re.fullmatch(r"HISTORY:\d{4}:[^:]+", evidence_id):
                if self._value_supported(value, evidence.text):
                    return True
            if evidence_id == "FILENAME" and previous_signal.search(evidence.text):
                if self._value_supported(value, evidence.text):
                    return True
            for clause in re.split(r"[。；;！？!?\n]", evidence.text):
                if previous_signal.search(clause) and self._value_supported(value, clause):
                    return True
        return False

    def _normalize_diseases(self, _: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        retained = []
        pruned_photo_rows = 0
        repaired_descriptions = 0
        history_repaired_rows = 0
        for disease in self.state.diseases:
            evidence_text = " ".join(
                self.state.evidence[evidence_id].text
                for evidence_id in disease.evidence
                if evidence_id in self.state.evidence
            )
            if not disease.description and re.search(r"照片\s*\d", f"{disease.location} {disease.type} {evidence_text}"):
                pruned_photo_rows += 1
                continue
            if is_non_disease_label(disease.type):
                inferred_type = canonical_disease_from_text(disease.description)
                if inferred_type:
                    disease.type = inferred_type
                else:
                    historical = self._exact_history_disease(disease.evidence)
                    if historical:
                        disease.type, disease.description = historical
                        history_repaired_rows += 1
                    elif is_meaningful_condition_state(disease.type, disease.description):
                        # Official labels preserve explicit good-condition
                        # rows, but an exact historical defect takes priority
                        # when the current ``现状`` cell is only a placeholder.
                        pass
                    else:
                        pruned_photo_rows += 1
                        continue
            if not disease.description and disease.location and disease.type:
                disease.description = f"{disease.location}：{disease.type}"
                repaired_descriptions += 1
            retained.append(disease)
        self.state.diseases = retained
        before = sum(d.component == "其他" or d.type == "未明确" for d in self.state.diseases)
        if before:
            if not self.client or not self.settings:
                raise RuntimeError("模型不可用，不能规范化剩余未明确病害")
            normalizer = DiseaseNormalizer(self.client, self.settings, self.prompts.get("disease_system", "输出严格 JSON"))
            self.state.diseases = normalizer.normalize(self.state.diseases)
        after = sum(d.component == "其他" or d.type == "未明确" for d in self.state.diseases)
        self.state.validation = self.validator.validate(self.state)
        return {
            "pruned_photo_rows": pruned_photo_rows,
            "repaired_descriptions": repaired_descriptions,
            "history_repaired_rows": history_repaired_rows,
            "targets_before": before,
            "targets_after": after,
            "validation_passed": self.state.validation.passed,
        }, []

    def _exact_history_disease(self, evidence_ids: list[str]) -> tuple[str, str] | None:
        """Recover a placeholder row only from its exact prior-cycle match.

        Group and absence evidence are deliberately excluded.  The history
        matcher has already established row identity; this tool merely restores
        the explicit type/description that a photo-led current ledger replaced
        with ``现状``.  Unmatched presentation rows remain prunable.
        """
        for evidence_id in evidence_ids:
            if not evidence_id.startswith("HISTORY:") or not re.search(r":D\d+$", evidence_id):
                continue
            evidence = self.state.evidence.get(evidence_id)
            if not evidence:
                continue
            match = re.search(
                r"历史病害：部位=.*?；类型=(?P<type>.*?)；描述=(?P<description>.*?)(?:。|$)",
                evidence.text,
            )
            if not match:
                continue
            disease_type = match.group("type").strip()
            description = match.group("description").strip()
            if not disease_type or is_non_disease_label(disease_type):
                continue
            if not description or description in {"未明确", "未提及"}:
                description = disease_type
            return disease_type, description
        return None

    def _mark_unavailable(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        scope = str(arguments.get("scope") or "")
        if scope != "overall":
            raise ValueError("只允许对总体评分范围执行缺失确认")
        action_key = self._action_key("list_score_candidates", {"scope": scope})
        if action_key not in self.completed_action_keys:
            raise ValueError("必须先执行 list_score_candidates 才能确认字段缺失")
        candidate_report, _ = self._list_score_candidates({"scope": scope})
        if candidate_report["candidates"]:
            raise ValueError("仍存在总体评分候选，不能标记为无")
        evidence_id = "D-ABSENT-OVERALL"
        if evidence_id not in self.state.evidence:
            self.state.add_evidence(EvidenceRef(
                id=evidence_id,
                kind="derived",
                text="Agent 已检索全文段落和表格，原报告未提供可引用的总体评分或总体等级。",
                source_path=str(self.parsed.path),
                priority=70,
            ))
        marked: list[str] = []
        for field_name, target in (
            ("scores.overall.score", self.state.scores.overall.score),
            ("scores.overall.grade", self.state.scores.overall.grade),
        ):
            if target.value in (None, ""):
                target.value = "无"
                target.evidence = [evidence_id]
                target.source_priority = 70
                target.confidence = 1.0
                marked.append(field_name)
        self.state.validation = self.validator.validate(self.state)
        return {"marked": marked, "validation_passed": self.state.validation.passed}, [evidence_id]

    def _write_decision(self, _: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        if not self.client or not self.settings:
            raise RuntimeError("模型不可用，不能执行 write_decision")
        writer = DecisionWriter(self.client, self.settings, self.prompts.get("decision_system", "输出严格 JSON"))
        decision = writer.write(self.state)
        evidence_text = " ".join(item.text for item in self.state.evidence.values())
        state_text = " ".join([
            str(self.state.bridge.name.value or ""),
            str(self.state.bridge.report_date.value or ""),
            *[
                str(field.value or "")
                for scope in ("overall", "deck", "superstructure", "substructure", "previous")
                for field in (
                    getattr(self.state.scores, scope).score,
                    getattr(self.state.scores, scope).grade,
                )
            ],
            *[
                f"{item.component} {item.location} {item.type} {item.description}"
                for item in self.state.diseases
            ],
            *[
                f"{item.category} {item.component} {item.content}"
                for item in self.state.recommendations
            ],
        ])
        counts = Counter((item.component, item.type) for item in self.state.diseases)
        derived_counts = " ".join(f"{count}处" for count in counts.values())
        support_text = f"{evidence_text} {state_text} {derived_counts}"
        rejected: list[str] = []

        def accepted(field: str, value: str) -> bool:
            if not value:
                return False
            if self._narrative_grounded(value, support_text):
                return True
            rejected.append(field)
            return False

        no_source_comparison = "D-NO-SOURCE-COMPARISON" in self.state.evidence
        if not no_source_comparison and accepted("development_trend", decision.development_trend):
            self.state.narrative.development_trend = decision.development_trend
        if accepted("overall_conclusion", decision.overall_conclusion):
            self.state.narrative.overall_conclusion = decision.overall_conclusion
        accepted_causes = [
            value for index, value in enumerate(decision.causes)
            if accepted(f"causes[{index}]", value)
        ]
        if accepted_causes:
            self.state.narrative.causes = accepted_causes
        accepted_impacts = [
            value for index, value in enumerate(decision.safety_impacts)
            if accepted(f"safety_impacts[{index}]", value)
        ]
        if accepted_impacts:
            self.state.narrative.safety_impacts = accepted_impacts
        if accepted("key_risks", decision.key_risks):
            self.state.narrative.key_risks = decision.key_risks
        self.state.validation = self.validator.validate(self.state)
        trend_repaired = any(
            item.code == "TREND_SCORE_CONTRADICTION"
            for item in self.state.validation.errors
        )
        if trend_repaired:
            corrected_trend = score_development_trend(self.state)
            if corrected_trend:
                self.state.narrative.development_trend = corrected_trend
                self.state.validation = self.validator.validate(self.state)
        if no_source_comparison:
            # Preserve the source-level publication convention even when the
            # mandatory online decision model tries to narrate a trend from
            # external history or current-only disease counts.
            self.state.narrative.development_trend = "无"
            self.state.validation = self.validator.validate(self.state)
        return {
            "development_trend": bool(decision.development_trend),
            "overall_conclusion": bool(decision.overall_conclusion),
            "causes": len(decision.causes),
            "safety_impacts": len(decision.safety_impacts),
            "rejected_ungrounded": rejected,
            "trend_repaired": trend_repaired,
        }, []

    @classmethod
    def _narrative_grounded(cls, value: str, support_text: str) -> bool:
        if not cls._numbers_supported(value, support_text):
            return False
        normalized_support = re.sub(r"\s+", "", support_text).upper().replace("类", "级")
        grades = re.findall(r"[ABCDE一二三四五](?=[级类])", value, re.I)
        if any(f"{grade.upper()}级" not in normalized_support for grade in grades):
            return False
        return all(term not in value or term in support_text for term in DECISION_DEFECT_TERMS)

    def _finalize(self, _: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        pending = self.pending_actions()
        if pending:
            names = [f"{item['action']}({item.get('arguments') or {}})" for item in pending]
            raise ValueError(f"复杂格式的强制探索义务尚未完成: {names}")
        self.state.validation = self.validator.validate(self.state)
        if not self.state.validation.passed:
            raise ValueError("ConstraintEngine 未通过，禁止 finalize")
        self.finalized = True
        return {"validation_passed": True, "evidence_coverage": self.state.validation.evidence_coverage}, []

    def _fail(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        self.failed = True
        self.failure_reason = str(arguments.get("reason") or "Agent 主动终止")[:1000]
        return {"failure_reason": self.failure_reason}, []

    def _ensure_paragraph_evidence(self, index: int, text: str, priority: int) -> str:
        evidence_id = f"P{index}"
        if evidence_id not in self.state.evidence:
            self.state.add_evidence(EvidenceRef(id=evidence_id, kind="paragraph", text=text, source_path=str(self.parsed.path), priority=priority, paragraph_index=index))
        elif priority > self.state.evidence[evidence_id].priority:
            self.state.evidence[evidence_id].priority = priority
        return evidence_id

    def _ensure_table_evidence(self, table_index: int, row_index: int, text: str, priority: int) -> str:
        evidence_id = f"T{table_index}R{row_index}"
        if evidence_id not in self.state.evidence:
            self.state.add_evidence(EvidenceRef(id=evidence_id, kind="table", text=text, source_path=str(self.parsed.path), priority=priority, table_index=table_index, row_index=row_index))
        elif priority > self.state.evidence[evidence_id].priority:
            self.state.evidence[evidence_id].priority = priority
        return evidence_id

    @staticmethod
    def _value_supported(value: Any, evidence_text: str) -> bool:
        rendered = str(value or "")
        numbers = re.findall(r"\d+(?:\.\d+)?", rendered)
        evidence_numbers = re.findall(r"\d+(?:\.\d+)?", evidence_text)
        for number in numbers:
            try:
                if not any(abs(float(number) - float(candidate)) <= 0.011 for candidate in evidence_numbers):
                    return False
            except ValueError:
                return False
        grade_tokens = re.findall(r"[ABCDE一二三四五](?=[级类])", rendered, re.I)
        for token in grade_tokens:
            # A bare letter inside ``BCI``/``BSI`` or a scale row containing
            # every A--E label is not literal support for one selected grade.
            # Fact patches must cite a source that explicitly prints the
            # proposed grade/class; report-local scales remain reasoning
            # context, not a loophole in the write gate.
            explicit_grade = re.search(rf"(?<![A-Za-z]){re.escape(token)}\s*(?:级|类)", evidence_text, re.I)
            table_grade_cell = (
                any(label in evidence_text for label in ("技术状况", "评估等级", "评定等级"))
                and re.search(rf"(?:^|\|)\s*{re.escape(token)}\s*(?:\||$)", evidence_text, re.I)
            )
            if not explicit_grade and not table_grade_cell:
                return False
        if numbers or grade_tokens:
            return True
        normalized_value = re.sub(r"[\s（）()]", "", rendered)
        normalized_evidence = re.sub(r"[\s（）()]", "", evidence_text)
        return bool(normalized_value and normalized_value in normalized_evidence)

    def _has_noncontradictory_support(
        self,
        value: Any,
        evidence_ids: list[str],
        contradictory_evidence_ids: set[str],
    ) -> bool:
        """Fail closed when a patch derives only from inconsistent evidence."""

        if not any(item in contradictory_evidence_ids for item in evidence_ids):
            return True
        consistent_ids = [item for item in evidence_ids if item not in contradictory_evidence_ids]
        if not consistent_ids:
            return False
        consistent_text = " ".join(self.state.evidence[item].text for item in consistent_ids)
        return self._value_supported(value, consistent_text)

    @staticmethod
    def _same_summary_value(left: Any, right: Any) -> bool:
        """Compare one scalar score/grade without weakening the write gate."""

        left_grade = re.fullmatch(r"\s*([ABCDE])\s*(?:级|类)?\s*", str(left), re.I)
        right_grade = re.fullmatch(r"\s*([ABCDE])\s*(?:级|类)?\s*", str(right), re.I)
        if left_grade or right_grade:
            return bool(
                left_grade
                and right_grade
                and left_grade.group(1).upper() == right_grade.group(1).upper()
            )
        try:
            return abs(float(left) - float(right)) <= 0.011
        except (TypeError, ValueError):
            return str(left).strip() == str(right).strip()

    def _repeated_final_table_consensus(
        self,
        field: str,
        score_catalog: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        """Return a scalar value repeated by two independent final tables.

        A model may arbitrate one final table against one narrative statement,
        but a lone intermediate calculation must not override a value repeated
        in multiple published summaries.  This is deliberately structural: it
        depends on source roles and independent table coordinates, never on a
        report name, year, bridge, or expected answer.
        """

        match = re.fullmatch(
            r"scores\.(overall|deck|superstructure|substructure)\.(score|grade)",
            field,
        )
        if not match:
            return None
        scope, attribute = match.groups()
        final_role = (
            "final_overall_rating_table"
            if scope == "overall"
            else "final_component_rating_table"
        )
        candidates = [
            item
            for item in score_catalog.get(scope, [])
            if item.get("source_role") == final_role
            and not item.get("self_contradictory")
            and item.get("side") is None
            and item.get(attribute) not in (None, "")
        ]
        by_table: dict[int, dict[str, Any]] = {}
        for item in candidates:
            table_match = re.fullmatch(r"T(\d+)R\d+", str(item.get("evidence_id", "")))
            if table_match:
                by_table[int(table_match.group(1))] = item
        if len(by_table) < 2:
            return None

        rows = list(by_table.values())
        first = rows[0][attribute]
        if not all(self._same_summary_value(item[attribute], first) for item in rows[1:]):
            return None
        value: Any = first
        if attribute == "grade":
            value = f"{str(first).strip().upper().removesuffix('级').removesuffix('类')}级"
        return {
            "value": value,
            "evidence_ids": [str(item["evidence_id"]) for item in rows],
        }

    def _structural_side_completion_patches(
        self,
        score_catalog: dict[str, list[dict[str, Any]]],
        side_coverage_gaps: list[str],
    ) -> list[dict[str, Any]]:
        """Complete a left/right final-table shape the Agent already selected.

        This operator never changes metric roles.  It activates only when the
        current value already equals one side of the final published table;
        then it restores the omitted side and its explicitly paired grade.
        """

        patches: list[dict[str, Any]] = []
        for field in side_coverage_gaps:
            scope = field.split(".")[1]
            current = getattr(self.state.scores, scope).score.value
            candidates = score_catalog.get(scope, [])
            if all(side in str(current or "") for side in ("左幅", "右幅")):
                continue
            final_role = (
                "final_overall_rating_table"
                if scope == "overall"
                else "final_component_rating_table"
            )
            final_rows = [
                item for item in candidates
                if item.get("source_role") == final_role
                and not item.get("self_contradictory")
                and item.get("side") in {"左幅", "右幅"}
            ]
            sides: dict[str, dict[str, Any]] = {}
            for item in final_rows:
                side = str(item["side"])
                if side in sides and (
                    float(sides[side]["score"]) != float(item["score"])
                    or str(sides[side].get("grade")) != str(item.get("grade"))
                ):
                    sides = {}
                    break
                sides[side] = item
            if not {"左幅", "右幅"}.issubset(sides):
                continue
            component_values = [float(sides[side]["score"]) for side in ("左幅", "右幅")]
            current_values = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(current or ""))]
            if not any(
                abs(selected - published) <= 0.011
                for selected in current_values
                for published in component_values
            ):
                continue
            value = (
                f"{component_values[0]}（左幅）/"
                f"{component_values[1]}（右幅）"
            )
            evidence_ids = [
                str(sides["左幅"]["evidence_id"]),
                str(sides["右幅"]["evidence_id"]),
            ]
            for evidence_id in list(evidence_ids):
                table_match = re.fullmatch(r"T(\d+)R\d+", evidence_id)
                if table_match:
                    table_index = int(table_match.group(1))
                    table = next((item for item in self.parsed.tables if item.index == table_index), None)
                    if table is None:
                        continue
                    row_match = re.fullmatch(r"T\d+R(\d+)", evidence_id)
                    row_index = int(row_match.group(1)) if row_match else 0
                    if row_index < len(table.rows):
                        self._ensure_table_evidence(
                            table_index,
                            row_index,
                            " | ".join(table.rows[row_index]),
                            priority=98,
                        )
                    evidence_ids.append(self._ensure_table_evidence(
                        table_index,
                        0,
                        " | ".join(table.header),
                        priority=97,
                    ))
            evidence_ids = list(dict.fromkeys(evidence_ids))
            patches.append({
                "field": field,
                "value": value,
                "evidence_ids": evidence_ids,
            })
            grades = [str(sides[side].get("grade") or "") for side in ("左幅", "右幅")]
            if all(grades):
                patches.append({
                    "field": f"scores.{scope}.grade",
                    "value": f"{grades[0]}级（左幅）/{grades[1]}级（右幅）",
                    "evidence_ids": evidence_ids,
                })
        return patches

    @staticmethod
    def _published_summary_patch_repairs_gap(target: FieldValue, candidate: Any) -> bool:
        """Limit deterministic BSI publication patches to genuine repairs.

        A report may publish both component BCI and BSI.  Which one the label
        expects is a semantic choice for the reviewer, not a blanket rule.
        Deterministic code may fill an absent value or replace the known
        parser failure range ``0 < score <= 1`` with an explicit normal-scale
        score, but must not rewrite a stable, high-confidence fact.
        """

        if target.value in (None, "", "无", "未明确"):
            return True
        try:
            current_score = float(target.value)
            candidate_score = float(candidate)
        except (TypeError, ValueError):
            return False
        return 0 < current_score <= 1 < candidate_score <= 100

    def _published_summary_patch_allowed(self, field: str, target: FieldValue, candidate: Any) -> bool:
        """Keep the report's component BCI number while allowing grade review.

        CJJ 99 reports often publish a component BCI as the score and a lower
        BSI as the controlling condition grade.  The competition's component
        score field nevertheless remains BCI/BCIm; a grade arbitration must
        never cascade into replacing a stable BCI with the lower BSI.  BSI may
        fill a genuinely missing or parser-corrupted score, and its explicit
        grade remains separately reviewable.
        """

        if field.endswith(".grade"):
            return str(target.value) != str(candidate)
        return self._published_summary_patch_repairs_gap(target, candidate)

    def _bsi_mirrors_component_summary_ids(
        self,
        score_catalog: dict[str, list[dict[str, Any]]],
    ) -> set[str]:
        """Detect a summary that mislabeled all three component BCI values as BSI.

        If one evidence row's alleged BSIm/BSIs/BSIx values exactly mirror the
        three independently parsed final component-table scores, the row cannot
        establish lower control grades. It remains usable as BCI context; only
        its BSI role is excluded from automatic grade arbitration.
        """
        scopes = ("deck", "superstructure", "substructure")
        final_scores: dict[str, float] = {}
        for scope in scopes:
            values = {
                float(item["score"])
                for item in score_catalog.get(scope, [])
                if item.get("source_role") == "final_component_rating_table"
                and item.get("side") is None
                and not item.get("self_contradictory")
            }
            if len(values) != 1:
                return set()
            final_scores[scope] = next(iter(values))

        evidence_ids = {
            str(item["evidence_id"])
            for scope in scopes
            for item in score_catalog.get(scope, [])
            if item.get("metric_role") == "control_index"
            and item.get("grade_binding")
            and item.get("side") is None
        }
        mirrored: set[str] = set()
        for evidence_id in evidence_ids:
            if all(any(
                str(item.get("evidence_id")) == evidence_id
                and item.get("metric_role") == "control_index"
                and item.get("grade_binding")
                and item.get("side") is None
                and abs(float(item["score"]) - final_scores[scope]) <= 0.011
                for item in score_catalog.get(scope, [])
            ) for scope in scopes):
                mirrored.add(evidence_id)
        return mirrored

    def _unambiguous_control_grade_patches(
        self,
        score_catalog: dict[str, list[dict[str, Any]]],
        *,
        excluded_evidence_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Keep component BCI numbers while applying a unique explicit BSI grade."""
        patches: list[dict[str, Any]] = []
        for scope in ("deck", "superstructure", "substructure"):
            controls = [
                item for item in score_catalog.get(scope, [])
                if item.get("metric_role") == "control_index"
                and item.get("grade_binding")
                and item.get("grade")
                and item.get("source_role") == "explicit_assessment_sentence"
                and not item.get("self_contradictory")
                and str(item.get("evidence_id")) not in excluded_evidence_ids
            ]
            side_values = {item.get("side") for item in controls}
            selected: list[dict[str, Any]] = []
            if {"左幅", "右幅"}.issubset(side_values):
                grades: list[str] = []
                for side in ("左幅", "右幅"):
                    candidates = [item for item in controls if item.get("side") == side]
                    consensus = self._control_grade_consensus(candidates)
                    if consensus is None:
                        break
                    grade, rows = consensus
                    grades.append(grade)
                    selected.extend(rows)
                if len(grades) != 2:
                    continue
                grade_value = (
                    f"{grades[0]}级"
                    if grades[0] == grades[1]
                    else f"{grades[0]}级（左幅）/{grades[1]}级（右幅）"
                )
            else:
                scalar = [item for item in controls if item.get("side") is None]
                consensus = self._control_grade_consensus(scalar)
                if consensus is None:
                    continue
                grade, selected = consensus
                grade_value = f"{grade}级"
            target = getattr(self.state.scores, scope).grade
            if str(target.value) == grade_value:
                continue
            evidence_ids = list(dict.fromkeys(str(item["evidence_id"]) for item in selected))
            patches.append({
                "field": f"scores.{scope}.grade",
                "value": grade_value,
                "evidence_ids": evidence_ids,
            })
        return patches

    def _control_grade_consensus(
        self,
        controls: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]] | None:
        """Resolve a grade without requiring every control score to be equal.

        Output keeps the component BCI number and uses BSI only as the
        controlling grade. Thus 80/B and 85/B are not a grade conflict. When
        grades actually differ, fall back to independent score/grade evidence
        consensus so an isolated abstract cannot dominate.
        """
        usable = [item for item in controls if item.get("grade")]
        grades = {str(item["grade"]).upper() for item in usable}
        if len(grades) == 1:
            return next(iter(grades)), usable
        pair = self._control_pair_consensus(usable)
        if pair is None:
            return None
        _, grade, rows = pair
        return grade, rows

    def _control_pair_consensus(
        self,
        controls: list[dict[str, Any]],
    ) -> tuple[float, str, list[dict[str, Any]]] | None:
        """Choose a control score/grade only when independent evidence agrees.

        A report can repeat the opening abstract both as a body paragraph and
        inside the metadata table. Those are one source, not two independent
        votes. Paragraph containment therefore collapses such duplicates. If
        conflicting pairs remain, a unique pair needs at least two independent
        textual supports; otherwise the caller fails closed.
        """
        grouped: dict[tuple[float, str], list[dict[str, Any]]] = {}
        for item in controls:
            try:
                pair = (round(float(item["score"]), 3), str(item["grade"]).upper())
            except (KeyError, TypeError, ValueError):
                continue
            grouped.setdefault(pair, []).append(item)
        if not grouped:
            return None

        support_counts: dict[tuple[float, str], int] = {}
        for pair, rows in grouped.items():
            distinct_texts: list[str] = []
            for item in rows:
                evidence_id = str(item.get("evidence_id") or "")
                evidence = self.state.evidence.get(evidence_id)
                text = str(item.get("text") or (evidence.text if evidence else ""))
                normalized = re.sub(r"[\s\u3000，。；：,.;:|]", "", text).lower()
                if not normalized:
                    normalized = f"evidence:{evidence_id}"
                if any(normalized in existing or existing in normalized for existing in distinct_texts):
                    continue
                distinct_texts.append(normalized)
            support_counts[pair] = len(distinct_texts)

        if len(grouped) == 1:
            pair = next(iter(grouped))
            return pair[0], pair[1], grouped[pair]
        ranked = sorted(
            grouped,
            key=lambda pair: (support_counts[pair], len(grouped[pair])),
            reverse=True,
        )
        winner = ranked[0]
        runner_up = ranked[1]
        if support_counts[winner] < 2 or support_counts[winner] <= support_counts[runner_up]:
            return None
        return winner[0], winner[1], grouped[winner]

    def _published_summary_metric_patches(
        self,
        score_catalog: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Honor a BSI explicitly published in the report abstract conclusion.

        Training-label analysis shows component BCI is the dominant convention;
        this exception is therefore restricted to key-value abstract rows named
        ``检测结论`` rather than any intermediate calculation paragraph.
        """

        published_ids: set[str] = set()
        published_rows: list[tuple[str, str]] = []
        for table in self.parsed.tables:
            for row_index, row in enumerate(table.rows):
                if not row or row[0].strip() != "检测结论":
                    continue
                text = " | ".join(row)
                if "BSI" not in text:
                    continue
                evidence_id = self._ensure_table_evidence(
                    table.index,
                    row_index,
                    text,
                    priority=99,
                )
                published_ids.add(evidence_id)
                published_rows.append((evidence_id, text))

        # Some reports contain an early abstract that calls the three final
        # component BCI values ``BSIm/BSIs/BSIx``.  Treating that mislabeled
        # row as a control-index publication overwrites the later, independently
        # calculated BSI grades.  Exclude only rows that mirror all three final
        # component-table scores; a genuine BSI abstract remains eligible.
        mirrored_ids = self._bsi_mirrors_component_summary_ids(score_catalog)
        published_ids.difference_update(mirrored_ids)
        published_rows = [
            item for item in published_rows
            if item[0] not in mirrored_ids
        ]

        patches: list[dict[str, Any]] = []
        for scope in ("deck", "superstructure", "substructure"):
            scope_label = {
                "deck": "桥面系",
                "superstructure": "上部结构",
                "substructure": "下部结构",
            }[scope]
            controls = [
                item for item in score_catalog.get(scope, [])
                if item.get("evidence_id") in published_ids
                and item.get("metric_role") == "control_index"
                and item.get("grade_binding")
                and item.get("grade")
                and not item.get("self_contradictory")
            ]
            for evidence_id, text in published_rows:
                pattern = re.compile(
                    rf"(?:(?P<side>左幅|右幅)[^；。]{{0,45}}?)?"
                    rf"{scope_label}\s*BSI[mMsSxX]?\s*(?:值|评分)?\s*(?:为|=|：|:)?\s*"
                    rf"(?P<score>{FLOAT})[^；。]{{0,35}}?"
                    rf"(?:评定为|等级为|评为|定为)\s*(?P<grade>[ABCDE])\s*(?:级|类)",
                    re.I,
                )
                for match in pattern.finditer(text):
                    controls.append({
                        "evidence_id": evidence_id,
                        "metric_role": "control_index",
                        "score": float(match.group("score")),
                        "grade": match.group("grade").upper(),
                        "grade_binding": True,
                        "side": match.group("side"),
                        "self_contradictory": False,
                    })

            # An opening ``检测结论`` is not authoritative when two
            # independent component/final-summary sentences agree on another
            # BSI pair. Conversely, a genuinely consistent published abstract
            # remains eligible. This is evidence consensus rather than a
            # positional "last paragraph wins" rule.
            reference_controls = [
                item for item in score_catalog.get(scope, [])
                if item.get("metric_role") == "control_index"
                and item.get("grade_binding")
                and item.get("grade")
                and item.get("source_role") == "explicit_assessment_sentence"
                and not item.get("self_contradictory")
                and str(item.get("evidence_id")) not in mirrored_ids
            ]
            reference_sides = {item.get("side") for item in reference_controls}
            if {"左幅", "右幅"}.issubset(reference_sides):
                filtered: list[dict[str, Any]] = []
                for side in ("左幅", "右幅"):
                    side_reference = [item for item in reference_controls if item.get("side") == side]
                    consensus = self._control_pair_consensus(side_reference)
                    pairs = {
                        (round(float(item["score"]), 3), str(item["grade"]).upper())
                        for item in side_reference
                    }
                    if consensus is None:
                        if len(pairs) <= 1:
                            filtered.extend(item for item in controls if item.get("side") == side)
                        continue
                    score, grade, _ = consensus
                    filtered.extend(
                        item for item in controls
                        if item.get("side") == side
                        and abs(float(item["score"]) - score) <= 0.011
                        and str(item["grade"]).upper() == grade
                    )
                controls = filtered
            else:
                scalar_reference = [item for item in reference_controls if item.get("side") is None]
                consensus = self._control_pair_consensus(scalar_reference)
                pairs = {
                    (round(float(item["score"]), 3), str(item["grade"]).upper())
                    for item in scalar_reference
                }
                if consensus is None:
                    if len(pairs) > 1:
                        controls = []
                else:
                    score, grade, _ = consensus
                    controls = [
                        item for item in controls
                        if item.get("side") is None
                        and abs(float(item["score"]) - score) <= 0.011
                        and str(item["grade"]).upper() == grade
                    ]
            by_side = {str(item.get("side")): item for item in controls}
            if {"左幅", "右幅"}.issubset(by_side):
                selected = [by_side["左幅"], by_side["右幅"]]
                score_value: Any = "/".join(
                    f"{float(item['score'])}（{side}）"
                    for side, item in zip(("左幅", "右幅"), selected)
                )
                grade_values = [str(item["grade"]).upper() for item in selected]
                grade_value: Any = (
                    f"{grade_values[0]}级"
                    if grade_values[0] == grade_values[1]
                    else f"{grade_values[0]}级（左幅）/{grade_values[1]}级（右幅）"
                )
            else:
                scalar = [item for item in controls if item.get("side") is None]
                pairs = {(float(item["score"]), str(item["grade"]).upper()) for item in scalar}
                if len(pairs) != 1:
                    continue
                score, grade = next(iter(pairs))
                selected = scalar
                score_value = score
                grade_value = f"{grade}级"
            evidence_ids = list(dict.fromkeys(str(item["evidence_id"]) for item in selected))
            patches.extend([
                {"field": f"scores.{scope}.score", "value": score_value, "evidence_ids": evidence_ids},
                {"field": f"scores.{scope}.grade", "value": grade_value, "evidence_ids": evidence_ids},
            ])
        return patches

    def _divergent_overall_side_patches(self) -> list[dict[str, Any]]:
        """Preserve two published facility results when one side was selected.

        Cover abstracts often print only the conservative side while the final
        facility tables publish both left and right results.  Restore the pair
        only when the current scalar exactly equals one of those final values;
        this prevents unrelated intermediate whole-bridge tables from
        replacing an independently published facility score.
        """

        sides: dict[str, dict[str, Any]] = {}
        for table in self.parsed.tables:
            context = " ".join(table.context[-4:])
            side = "左幅" if "左幅" in context else "右幅" if "右幅" in context else None
            if side is None or not table.rows:
                continue
            header = table.header
            header_text = " | ".join(header)
            if "技术状况评定表" not in context or "BCI" not in header_text:
                continue
            overall_index = next(
                (
                    index for index, value in enumerate(header)
                    if value.strip() in {"BCI", "BCIm"} or "桥梁技术状况指数" in value
                ),
                None,
            )
            grade_index = next(
                (
                    index for index, value in enumerate(header)
                    if "整体技术状况等级" in value or "完好状况评估等级" in value
                ),
                None,
            )
            if overall_index is None or grade_index is None:
                continue
            records: list[tuple[float, str, int]] = []
            for row_index, row in enumerate(table.rows[1:], 1):
                shifted = len(row) == len(header) + 1 and len(row) > 2 and bool(
                    re.fullmatch(r"BCI[mMsSxX]?|BCL[mMsSxX]?", row[1].strip(), re.I)
                )

                def cell(index: int) -> str:
                    actual = index + 1 if shifted and index >= 1 else index
                    return row[actual] if actual < len(row) else ""

                score_match = re.search(FLOAT, cell(overall_index))
                grade_match = re.fullmatch(r"\s*([ABCDE一二三四五])\s*(?:级|类)?\s*", cell(grade_index), re.I)
                if score_match and grade_match:
                    record = (float(score_match.group(0)), grade_match.group(1).upper(), row_index)
                    if record not in records:
                        records.append(record)
            values = {(score, grade) for score, grade, _ in records}
            if len(values) != 1:
                continue
            score, grade = next(iter(values))
            row_index = min(
                index for item_score, item_grade, index in records
                if item_score == score and item_grade == grade
            )
            row_evidence = self._ensure_table_evidence(
                table.index,
                row_index,
                " | ".join(table.rows[row_index]),
                priority=99,
            )
            header_evidence = self._ensure_table_evidence(
                table.index,
                0,
                header_text,
                priority=99,
            )
            sides[side] = {
                "score": score,
                "grade": grade,
                "evidence_ids": [row_evidence, header_evidence],
            }
        if not {"左幅", "右幅"}.issubset(sides):
            return []
        current_numbers = [
            float(value)
            for value in re.findall(
                r"\d+(?:\.\d+)?",
                str(self.state.scores.overall.score.value or ""),
            )
        ]
        if not any(
            abs(current - float(sides[side]["score"])) <= 0.011
            for current in current_numbers
            for side in ("左幅", "右幅")
        ):
            return []
        evidence_ids = list(dict.fromkeys([
            *sides["左幅"]["evidence_ids"],
            *sides["右幅"]["evidence_ids"],
        ]))
        return [
            {
                "field": "scores.overall.score",
                "value": (
                    f"{sides['左幅']['score']}（左幅）/"
                    f"{sides['右幅']['score']}（右幅）"
                ),
                "evidence_ids": evidence_ids,
            },
            {
                "field": "scores.overall.grade",
                "value": f"{sides['左幅']['grade']}级（左幅）/{sides['右幅']['grade']}级（右幅）",
                "evidence_ids": evidence_ids,
            },
        ]

    def _report_local_grade(self, score: float) -> str | None:
        """Map a score with a grading scale printed inside this report.

        No external CJJ threshold is encoded here.  The operator only reads a
        horizontal or vertical range/grade table from the current document,
        which makes the consistency check portable across report standards.
        """

        range_pattern = re.compile(
            r"[\[（(]\s*(\d+(?:\.\d+)?)\s*[,，]\s*(\d+(?:\.\d+)?)\s*[\]）)]"
        )

        def contains(raw_range: str) -> bool:
            match = range_pattern.search(raw_range)
            if not match:
                return False
            lower, upper = (float(match.group(1)), float(match.group(2)))
            lower_inclusive = raw_range.lstrip().startswith("[")
            upper_inclusive = raw_range.rstrip().endswith("]")
            lower_ok = score >= lower if lower_inclusive else score > lower
            upper_ok = score <= upper if upper_inclusive else score < upper
            return lower_ok and upper_ok

        for table in self.parsed.tables:
            if len(table.rows) >= 2:
                range_row = table.rows[0]
                grade_row = table.rows[1]
                if re.search(r"\b(?:BCI|Dr)\b", " | ".join(range_row), re.I) and re.search(
                    r"等级", " | ".join(grade_row)
                ):
                    for raw_range, raw_grade in zip(range_row[1:], grade_row[1:]):
                        grade_match = re.fullmatch(r"\s*([ABCDE一二三四五])\s*(?:级|类)?\s*", raw_grade, re.I)
                        if grade_match and contains(raw_range):
                            return grade_match.group(1).upper()
            for row in table.rows:
                row_text = " | ".join(row)
                grade_match = re.search(r"(?<![A-Za-z])([ABCDE一二三四五])\s*(?:级|类)(?![A-Za-z])", row_text, re.I)
                if grade_match and contains(row_text):
                    return grade_match.group(1).upper()
        return None

    def _report_local_scale_evidence_ids(self) -> list[str]:
        evidence_ids: list[str] = []
        for table in self.parsed.tables:
            if len(table.rows) < 2:
                continue
            first_two = " || ".join(" | ".join(row) for row in table.rows[:2])
            if not re.search(r"\b(?:BCI|Dr)\b", first_two, re.I):
                continue
            if not re.search(r"(?:评估|评定)?等级|[ABCDE一二三四五]", first_two, re.I):
                continue
            for row_index, row in enumerate(table.rows[:2]):
                evidence_ids.append(self._ensure_table_evidence(
                    table.index,
                    row_index,
                    " | ".join(row),
                    priority=91,
                ))
        return list(dict.fromkeys(evidence_ids))

    def _selected_component_grade_patches(
        self,
        score_catalog: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Make the grade follow a component index already selected upstream."""

        patches: list[dict[str, Any]] = []
        scale_evidence_ids = self._report_local_scale_evidence_ids()
        for scope in ("deck", "superstructure", "substructure"):
            current_score = getattr(self.state.scores, scope).score.value
            current_grade = getattr(self.state.scores, scope).grade.value
            rows = [
                item for item in score_catalog.get(scope, [])
                if item.get("source_role") == "final_component_rating_table"
                and not item.get("self_contradictory")
            ]
            if not rows or current_score in (None, "", "无"):
                continue
            rendered = str(current_score)
            side_values = {
                side: float(value)
                for value, side in re.findall(r"(\d+(?:\.\d+)?)\s*[（(](左幅|右幅)[）)]", rendered)
            }
            selected_rows: list[dict[str, Any]] = []
            grades: list[str] = []
            sides: list[str | None]
            if {"左幅", "右幅"}.issubset(side_values):
                sides = ["左幅", "右幅"]
                for side in sides:
                    matches = [
                        item for item in rows
                        if item.get("side") == side
                        and abs(float(item["score"]) - side_values[side]) <= 0.011
                    ]
                    if not matches:
                        break
                    grade_set = {
                        str(item.get("grade") or self._report_local_grade(float(item["score"])) or "")
                        for item in matches
                    }
                    if len(grade_set) != 1 or not next(iter(grade_set)):
                        break
                    selected_rows.extend(matches)
                    grades.append(next(iter(grade_set)).upper())
                if len(grades) != 2:
                    continue
                value = f"{grades[0]}级（左幅）/{grades[1]}级（右幅）"
            else:
                numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", rendered)]
                if len(numbers) != 1:
                    continue
                matches = [item for item in rows if abs(float(item["score"]) - numbers[0]) <= 0.011]
                if not matches:
                    continue
                grade_set = {
                    str(item.get("grade") or self._report_local_grade(float(item["score"])) or "")
                    for item in matches
                }
                if len(grade_set) != 1 or not next(iter(grade_set)):
                    continue
                selected_rows = matches
                value = f"{next(iter(grade_set)).upper()}级"
            if str(current_grade) == value:
                continue
            evidence_ids = list(dict.fromkeys(
                str(item["evidence_id"]) for item in selected_rows
            ))
            for evidence_id in list(evidence_ids):
                table_match = re.fullmatch(r"T(\d+)R\d+", evidence_id)
                if not table_match:
                    continue
                table_index = int(table_match.group(1))
                table = next((item for item in self.parsed.tables if item.index == table_index), None)
                if table is not None:
                    evidence_ids.append(self._ensure_table_evidence(
                        table.index,
                        0,
                        " | ".join(table.header),
                        priority=90,
                    ))
            evidence_ids.extend(scale_evidence_ids)
            patches.append({
                "field": f"scores.{scope}.grade",
                "value": value,
                "evidence_ids": list(dict.fromkeys(evidence_ids)),
            })
        return patches

    def _metric_semantic_consistency_patches(
        self,
        score_catalog: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Reconcile component and control indices with report-local grades.

        The Agent still discovers and reviews candidates.  This final bounded
        operator enforces one schema invariant: keep the final component index
        unless a non-contradictory, grade-bound control index changes the
        published component grade.  If one side changes the metric role, use
        that role for the whole component so left/right output stays coherent.
        """

        patches: list[dict[str, Any]] = []
        for scope in ("deck", "superstructure", "substructure"):
            rows = score_catalog.get(scope, [])
            components = [
                item for item in rows
                if item.get("source_role") == "final_component_rating_table"
                and not item.get("self_contradictory")
            ]
            controls = [
                item for item in rows
                if item.get("metric_role") == "control_index"
                and item.get("grade_binding")
                and item.get("grade")
                and not item.get("self_contradictory")
            ]

            component_sides = {item.get("side") for item in components}
            control_sides = {item.get("side") for item in controls}
            if {"左幅", "右幅"}.issubset(component_sides) and {"左幅", "右幅"}.issubset(control_sides):
                sides: list[str | None] = ["左幅", "右幅"]
            elif None in component_sides and None in control_sides:
                sides = [None]
            else:
                continue

            selected_components: list[dict[str, Any]] = []
            selected_controls: list[dict[str, Any]] = []
            ambiguous = False
            for side in sides:
                side_components = [item for item in components if item.get("side") == side]
                side_controls = [item for item in controls if item.get("side") == side]
                component_pairs = {
                    (float(item["score"]), str(item.get("grade") or self._report_local_grade(float(item["score"])) or ""))
                    for item in side_components
                }
                control_pairs = {
                    (float(item["score"]), str(item["grade"]).upper())
                    for item in side_controls
                }
                if len(component_pairs) != 1 or len(control_pairs) != 1:
                    ambiguous = True
                    break
                component_score, component_grade = next(iter(component_pairs))
                control_score, control_grade = next(iter(control_pairs))
                if not component_grade:
                    ambiguous = True
                    break
                component_evidence_ids = list(dict.fromkeys(
                    str(item["evidence_id"]) for item in side_components
                ))
                for evidence_id in list(component_evidence_ids):
                    table_match = re.fullmatch(r"T(\d+)R\d+", evidence_id)
                    if table_match:
                        table_index = int(table_match.group(1))
                        table = self.parsed.tables[table_index]
                        component_evidence_ids.append(self._ensure_table_evidence(
                            table_index,
                            0,
                            " | ".join(table.header),
                            priority=90,
                        ))
                selected_components.append({
                    "score": component_score,
                    "grade": component_grade,
                    "evidence_ids": list(dict.fromkeys(component_evidence_ids)),
                })
                selected_controls.append({
                    "score": control_score,
                    "grade": control_grade,
                    "evidence_ids": list(dict.fromkeys(str(item["evidence_id"]) for item in side_controls)),
                })
            if ambiguous:
                continue

            use_control = any(
                component["grade"] != control["grade"]
                for component, control in zip(selected_components, selected_controls)
            )
            selected = selected_controls if use_control else selected_components
            evidence_ids = list(dict.fromkeys(
                evidence_id
                for item in selected
                for evidence_id in item["evidence_ids"]
            ))
            if sides == [None]:
                score_value: Any = selected[0]["score"]
                grade_value: Any = f"{selected[0]['grade']}级"
            else:
                score_value = "/".join(
                    f"{item['score']}（{side}）" for side, item in zip(sides, selected)
                )
                grade_value = "/".join(
                    f"{item['grade']}级（{side}）" for side, item in zip(sides, selected)
                )
            patches.extend([
                {"field": f"scores.{scope}.score", "value": score_value, "evidence_ids": evidence_ids},
                {"field": f"scores.{scope}.grade", "value": grade_value, "evidence_ids": evidence_ids},
            ])
        return patches

    @staticmethod
    def _fact_override_allowed(
        target: FieldValue,
        candidate_priority: int,
        *,
        semantic_conflict: bool = False,
    ) -> bool:
        """Allow repairs without letting equal evidence destabilize a fact.

        An equal-priority model quote is not new information, so it cannot
        replace a high-confidence, conflict-free extraction. Missing,
        conflicted or lower-confidence values remain repairable, while an
        actually stronger source can still supersede a stable value.
        """

        if target.value in (None, ""):
            return True
        if candidate_priority < target.source_priority:
            return False
        if (
            candidate_priority == target.source_priority
            and target.confidence >= 0.94
            and not target.conflicts
        ):
            return False
        return True

    def pending_actions(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.required_actions
            if self._action_key(item["action"], item.get("arguments") or {}) not in self.completed_action_keys
        ]

    def completed_actions(self) -> list[dict[str, Any]]:
        return [
            {"action": item["action"], "arguments": item.get("arguments") or {}}
            for item in self.required_actions
            if self._action_key(item["action"], item.get("arguments") or {}) in self.completed_action_keys
        ]

    def _build_required_actions(self) -> list[dict[str, Any]]:
        text = self.parsed.full_text
        required: list[dict[str, Any]] = []
        if self.workflow_profile == "details":
            if self.client:
                required.append({
                    "action": "review_details",
                    "arguments": {},
                    "reason": "开源文本模型依据当前与历史证据复核病害和建议字段",
                })
            return required
        if "左幅" in text and "右幅" in text:
            required.append({"action": "list_score_candidates", "arguments": {"scope": "overall"}, "reason": "报告包含左右幅"})
        if "Dr" in text and any(token in text for token in ("一类", "二类", "三类", "四类", "五类")):
            required.extend([
                {"action": "list_score_candidates", "arguments": {"scope": "overall"}, "reason": "报告使用旧规范 Dr"},
                {"action": "list_score_candidates", "arguments": {"scope": "previous"}, "reason": "旧规范需核对上次评分"},
            ])
        for scope, score_token, control_token in (
            ("deck", "BCIm", "BSIm"),
            ("superstructure", "BCIs", "BSIs"),
            ("substructure", "BCIx", "BSIx"),
        ):
            if score_token in text and control_token in text:
                required.append({
                    "action": "list_score_candidates",
                    "arguments": {"scope": scope},
                    "reason": f"{scope} 同时存在 BCI/BSI 候选，交由 Agent 证据消歧",
                })
        disease_rows = sum(max(0, len(table.rows) - 1) for table in self.parsed.tables if table.is_disease_table())
        if self.workflow_profile == "full" and disease_rows > 200:
            required.append({"action": "profile_disease_tables", "arguments": {}, "reason": "病害明细超过 200 行"})
        if self.client:
            if self.settings and self.settings.vision_model and self._critical_visual_gap() and self._get_visual_candidates():
                required.append({
                    "action": "inspect_visual_evidence",
                    "arguments": {},
                    "reason": "文本解析存在关键事实缺口，先用双视觉模型做证据化 OCR",
                })
            required.append({
                "action": "review_summary",
                "arguments": {},
                "reason": "开源文本模型与 VL 召回精排复核简要信息",
            })
            if self.workflow_profile == "full":
                required.extend([
                    {"action": "review_details", "arguments": {}, "reason": "开源文本模型依据当前与历史证据复核病害和建议字段"},
                    {"action": "normalize_diseases", "arguments": {}, "reason": "开源文本模型复核未明确病害实体"},
                    {"action": "write_decision", "arguments": {}, "reason": "开源文本模型生成受证据约束的结论与安全影响"},
                ])
        unique: dict[str, dict[str, Any]] = {}
        for item in required:
            unique[self._action_key(item["action"], item.get("arguments") or {})] = item
        return list(unique.values())

    @staticmethod
    def _action_key(action: str, arguments: dict[str, Any]) -> str:
        return json.dumps({"action": action, "arguments": arguments}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
