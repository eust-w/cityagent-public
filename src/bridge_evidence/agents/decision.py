from __future__ import annotations

from collections import Counter
import re

from pydantic import BaseModel, Field

from bridge_evidence.config import Settings
from bridge_evidence.models.schema import DocumentState

from .client import GatewayClient


class DecisionResult(BaseModel):
    development_trend: str = ""
    overall_conclusion: str = ""
    causes: list[str] = Field(default_factory=list)
    safety_impacts: list[str] = Field(default_factory=list)
    key_risks: str = ""


class DecisionWriter:
    def __init__(self, client: GatewayClient, settings: Settings, system_prompt: str):
        self.client = client
        self.settings = settings
        self.system_prompt = system_prompt

    def write(self, state: DocumentState) -> DecisionResult:
        source_comparison_available = "D-NO-SOURCE-COMPARISON" not in state.evidence
        disease_summary = Counter((d.component, d.type) for d in state.diseases)
        representative_diseases = self._representative_items(
            state.diseases,
            limit=36,
            group_key=lambda item: (item.component, item.type),
        )
        disease_details = [
            {
                "component": disease.component,
                "location": disease.location[:180],
                "type": disease.type,
                "description": disease.description[:320],
                "is_new": disease.is_new,
                "previous_state": (disease.previous_state or "")[:320],
                "development": (disease.development or "")[:240],
            }
            for disease in representative_diseases
        ]
        representative_recommendations = self._representative_items(
            state.recommendations,
            limit=24,
            group_key=lambda item: (item.category, item.component),
        )
        return self.client.complete_model(
            DecisionResult,
            system=self.system_prompt,
            user_payload={
                "task": "生成与事实表和病害明细一致的发展趋势、总体结论、主要风险、成因和安全影响，不得新增事实",
                "bridge": state.bridge.name.value,
                "scores": {
                    # Decision prose follows the competition's published
                    # summary fields.  Internal BSI/control candidates stay in
                    # the evidence graph but are intentionally hidden here so
                    # they cannot leak into the textual conclusion as if they
                    # were the reported component BCI score.
                    scope: {
                        "score": entry.score.value,
                        "grade": entry.grade.value,
                    }
                    for scope, entry in (
                        ("overall", state.scores.overall),
                        ("superstructure", state.scores.superstructure),
                        ("substructure", state.scores.substructure),
                        ("deck", state.scores.deck),
                        ("previous", state.scores.previous),
                    )
                },
                "score_entities": [
                    {
                        "entity_id": entity.entity_id,
                        "label": entity.label,
                        "side": entity.side,
                        "overall": {
                            "score": entity.overall.score.value,
                            "grade": entity.overall.grade.value,
                        },
                        "deck": {"score": entity.deck.score.value, "grade": entity.deck.grade.value},
                        "superstructure": {
                            "score": entity.superstructure.score.value,
                            "grade": entity.superstructure.grade.value,
                        },
                        "substructure": {
                            "score": entity.substructure.score.value,
                            "grade": entity.substructure.grade.value,
                        },
                    }
                    for entity in state.scores.entities
                ],
                "score_trend_control": score_development_trend(state),
                "source_comparison_available": source_comparison_available,
                "disease_summary": [{"component": c, "type": t, "count": n} for (c, t), n in disease_summary.most_common(30)],
                "disease_details": disease_details,
                "disease_coverage": {
                    "total": len(state.diseases),
                    "representative_rows": len(representative_diseases),
                    "distinct_component_types": len(disease_summary),
                },
                "recommendations": [
                    {"category": r.category, "component": r.component, "content": r.content[:320]}
                    for r in representative_recommendations
                ],
                "recommendation_coverage": {
                    "total": len(state.recommendations),
                    "representative_rows": len(representative_recommendations),
                },
                "source_conclusion": state.narrative.overall_conclusion[:1000],
                "source_development_trend": state.narrative.development_trend[:1000],
                "source_safety": state.narrative.safety_impacts[:3],
                "evidence_bound_claims": [claim.model_dump() for claim in state.narrative.claims],
                "output_schema": {
                    "development_trend": (
                        "70-160字，总—分结构：首句为趋势结论句（趋势级别＋最弱侧总体评分较上次的升降幅度）；"
                        "随后按‘新增—既有发展—已修复’递进各一句，最多列4个代表项；"
                        "句间用‘其中、此外’衔接，末句用‘综上’收束趋势判断"
                        if source_comparison_available
                        else "当前报告无上一次定检对比信息，只返回‘无’"
                    ),
                    "overall_conclusion": (
                        "70-150字，结论先行：首句概括全桥技术状况总体判断；"
                        "再按上部结构、下部结构、桥面系各一句概括主要病害类型及代表性量化证据；"
                        "末句写总体等级和养护等级作收束；不罗列病害总数或重复全部评分"
                    ),
                    "causes": ["只写定性推断"],
                    "safety_impacts": ["只覆盖已有病害"],
                    "key_risks": "30-90字，只保留2-4个最高风险病害，优先保留裂缝宽度、长度、面积、数量等原始量化证据及其影响",
                },
                "style_rules": [
                    "使用比赛标签式的压缩表达，不写报告复述或推理过程",
                    "总体趋势采用最弱侧/最小总体评分作为控制值，不对左右幅取平均",
                    "同类病害合并列举，不逐行重复位置",
                    "没有来源支持时不得写明显改善、明显劣化或安全结论",
                    "source_comparison_available=false 时不得推断时间趋势，development_trend 必须为‘无’",
                    "句式骨架：趋势=‘较上次定检，…总体（持平/缓慢劣化/改善），最弱侧总体评分（上升/下降）X.X分；其中，新增…；此外，既有…持续发展；综上，…’",
                    "句式骨架：结论=‘本桥…（总体判断）；上部结构…；下部结构…；桥面系…；总体评定为X级，养护等级…’",
                ],
            },
            temperature=self.settings.decision_temperature,
            task="decision",
        )

    @staticmethod
    def _representative_items(items, *, limit: int, group_key):
        """Cover the complete ledger without a biased first-N prompt slice."""
        if len(items) <= limit:
            return list(items)
        selected_indices: list[int] = []
        seen_groups = set()
        for index, item in enumerate(items):
            key = group_key(item)
            if key in seen_groups:
                continue
            seen_groups.add(key)
            selected_indices.append(index)
            if len(selected_indices) >= limit:
                return [items[index] for index in sorted(selected_indices)]
        selected_set = set(selected_indices)
        remaining = [index for index in range(len(items)) if index not in selected_set]
        slots = limit - len(selected_indices)
        if slots > 0 and remaining:
            selected_indices.extend(
                remaining[min(len(remaining) - 1, (offset * len(remaining)) // slots)]
                for offset in range(slots)
            )
        return [items[index] for index in sorted(set(selected_indices))]


FALLBACK_CAUSES = {
    "裂缝": "裂缝通常与混凝土收缩、温度作用、车辆反复荷载或局部受力有关，应结合裂缝宽度及变化持续观察。",
    "破损、露筋": "混凝土破损露筋多与保护层老化、施工缺陷、碰撞或水侵蚀有关，长期发展可能加速钢筋锈蚀。",
    "破损": "局部破损可能与材料老化、冲击磨耗或养护不及时有关。",
    "渗水泛碱": "渗水泛碱通常与防水失效、接缝渗漏及混凝土孔隙水迁移有关。",
    "涂层剥落": "涂层剥落多与自然老化、紫外线、温湿变化及基层附着不足有关。",
    "锈蚀": "锈蚀通常由长期潮湿、涂层失效和氯盐等环境作用引起。",
}


def score_development_trend(state: DocumentState) -> str | None:
    """Describe movement using the worst-side score as the bridge control."""

    def values(raw: object) -> tuple[list[float], dict[str, float]]:
        text = str(raw or "")
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
        sides: dict[str, float] = {}
        for match in re.finditer(r"(\d+(?:\.\d+)?)[^/\d]{0,10}(左幅|右幅)", text):
            sides[match.group(2)] = float(match.group(1))
        if not sides and len(numbers) == 2:
            sides = {"左幅": numbers[0], "右幅": numbers[1]}
        return numbers, sides

    current_numbers, _ = values(state.scores.overall.score.value)
    previous_numbers, _ = values(state.scores.previous.score.value)
    if not current_numbers or not previous_numbers:
        return None

    def movement(current: float, previous: float) -> str:
        delta = current - previous
        direction = "上升" if delta > 0.005 else "下降" if delta < -0.005 else "持平"
        return direction if direction == "持平" else f"{direction}{abs(delta):.2f}分"

    current = min(current_numbers)
    previous = min(previous_numbers)
    return f"总体评分较上一次{movement(current, previous)}。"


def apply_fallback_decision(state: DocumentState) -> None:
    absent_evidence_id = "D-ABSENT-SCORE-FIELDS"
    missing_score_fields = []
    for scope in ("overall", "superstructure", "substructure", "deck", "previous"):
        entry = getattr(state.scores, scope)
        for field_name, value in (("score", entry.score), ("grade", entry.grade)):
            if value.value in (None, ""):
                missing_score_fields.append(f"scores.{scope}.{field_name}")
                value.value = "无"
                value.evidence = [absent_evidence_id]
                value.source_priority = 1
                value.confidence = 0.35
    if missing_score_fields:
        from bridge_evidence.models.schema import EvidenceRef

        state.add_evidence(
            EvidenceRef(
                id=absent_evidence_id,
                kind="derived",
                text="确定性抽取遍历全文后，以下字段尚无可引用候选：" + "、".join(missing_score_fields),
                source_path=state.source_path,
                priority=1,
            )
        )

    types = {d.type for d in state.diseases}
    causes = []
    for disease_type, text in FALLBACK_CAUSES.items():
        if any(disease_type in value or value in disease_type for value in types):
            causes.append(text)
    state.narrative.causes = causes or ["病害成因需结合材料、环境、运营荷载和历史养护记录进一步分析。"]
    if not state.narrative.safety_impacts:
        components = "、".join(dict.fromkeys(d.component for d in state.diseases)) or "桥梁构件"
        state.narrative.safety_impacts = [f"现有病害涉及{components}；若长期不处治，可能影响结构耐久性和正常使用，应按原报告建议及时养护并持续观察。"]
    disease_counts = Counter((d.component, d.type) for d in state.diseases)
    disease_summary = "；".join(
        f"{component}{disease_type}{count}处"
        for (component, disease_type), count in disease_counts.most_common(8)
    )
    if not state.narrative.key_risks:
        state.narrative.key_risks = f"重点关注：{disease_summary}。" if disease_summary else "原报告未提取到明确病害记录，主要风险点未明确。"
    if not state.narrative.overall_conclusion:
        score = str(state.scores.overall.score.value or "无")
        grade = str(state.scores.overall.grade.value or "无")
        score_clause = "原报告未提供可核实的总体评分" if not re.search(r"\d", score) else f"总体评分为{score}分"
        grade_clause = "总体等级未明确" if grade in {"无", "未明确"} else f"总体等级为{grade}"
        disease_clause = f"主要病害包括{disease_summary}" if disease_summary else "未提取到明确病害记录"
        state.narrative.overall_conclusion = f"本次定检{score_clause}，{grade_clause}；{disease_clause}。"
    if not state.narrative.development_trend:
        score_trend = score_development_trend(state)
        if score_trend:
            state.narrative.development_trend = score_trend
        else:
            new_count = sum(d.is_new is True for d in state.diseases)
            existing_count = sum(d.is_new is False for d in state.diseases)
            if new_count or existing_count:
                state.narrative.development_trend = f"原报告未提供可核实的上次总体评分；本次识别新增病害{new_count}处、既有病害{existing_count}处。"
            else:
                state.narrative.development_trend = "原报告未提供可核实的上次总体评分，无法进行量化趋势比较。"
    state.narrative.dispositions = [r.content for r in state.recommendations]
