from __future__ import annotations

import re

from bridge_evidence.domain_rules import (
    is_condition_state_label,
    is_meaningful_condition_state,
    is_non_disease_label,
)

from .schema import DocumentState, ValidationMessage, ValidationState


KEY_FIELDS = (
    ("bridge.name", lambda s: s.bridge.name),
    ("bridge.report_date", lambda s: s.bridge.report_date),
    ("scores.overall.score", lambda s: s.scores.overall.score),
    ("scores.overall.grade", lambda s: s.scores.overall.grade),
)

EMERGENCY_EVIDENCE = re.compile(
    r"立即|封闭|限行|禁止通行|停止使用|危及结构安全|承载能力不足|垮塌|应急",
    re.I,
)


class ConstraintEngine:
    def validate(self, state: DocumentState) -> ValidationState:
        errors: list[ValidationMessage] = []
        warnings: list[ValidationMessage] = []

        for name, getter in KEY_FIELDS:
            value = getter(state)
            if value.value in (None, ""):
                errors.append(ValidationMessage(severity="error", code="MISSING_KEY_FIELD", message=f"缺少关键字段 {name}", field=name))
            elif not value.evidence:
                errors.append(ValidationMessage(severity="error", code="UNSUPPORTED_KEY_FIELD", message=f"关键字段 {name} 没有证据", field=name))

        for scope in ("overall", "superstructure", "substructure", "deck", "previous"):
            entry = getattr(state.scores, scope)
            if entry.score.value not in (None, "") and not entry.score.evidence:
                errors.append(ValidationMessage(severity="error", code="UNSUPPORTED_SCORE", message=f"{scope} 评分没有证据", field=f"scores.{scope}.score"))
            if entry.control_score.value not in (None, "") and not entry.control_score.evidence:
                errors.append(ValidationMessage(severity="error", code="UNSUPPORTED_CONTROL_SCORE", message=f"{scope} 控制评分没有证据", field=f"scores.{scope}.control_score"))
            if entry.grade.value not in (None, "") and not entry.grade.evidence:
                errors.append(ValidationMessage(severity="error", code="UNSUPPORTED_GRADE", message=f"{scope} 等级没有证据", field=f"scores.{scope}.grade"))

        evidence_ids = set(state.evidence)
        for disease in state.diseases:
            missing = set(disease.evidence) - evidence_ids
            if missing:
                errors.append(ValidationMessage(severity="error", code="BROKEN_DISEASE_EVIDENCE", message=f"病害证据不存在: {sorted(missing)}"))
            if not disease.description or not disease.type:
                errors.append(ValidationMessage(severity="error", code="INVALID_DISEASE", message="病害缺少类型或描述"))
            if is_non_disease_label(disease.type) and not is_meaningful_condition_state(
                disease.type,
                disease.description,
            ):
                errors.append(
                    ValidationMessage(
                        severity="error",
                        code="NON_DISEASE_LABEL",
                        message=f"非病害展示标签不得进入病害清单: {disease.type}",
                        field="diseases.type",
                    )
                )
            if (
                disease.is_new is True
                and disease.previous_state
                and "已发现" in disease.previous_state
                and "未发现" not in disease.previous_state
            ):
                warnings.append(
                    ValidationMessage(
                        severity="warning",
                        code="HISTORY_COHERENCE",
                        message=f"病害 {disease.sequence} 标记为新增，已将上次状态规范为未发现",
                        field="diseases.previous_state",
                    )
                )
                disease.previous_state = "未发现"
            if disease.novelty_source == "history_exact" and (
                disease.is_new is not False
                or not disease.previous_state
                or "未发现" in disease.previous_state
            ):
                warnings.append(
                    ValidationMessage(
                        severity="warning",
                        code="EXACT_HISTORY_COHERENCE",
                        message=f"病害 {disease.sequence} 已有精确历史匹配，已恢复为非新增/上次已发现",
                        field="diseases.previous_state",
                    )
                )
                disease.is_new = False
                disease.previous_state = "已发现"

        current = self._numeric_control(state.scores.overall.score.value)
        previous = self._numeric_control(state.scores.previous.score.value)
        if current is not None and previous is not None:
            delta = current - previous
            trend = state.narrative.development_trend
            has_up = bool(re.search(r"较上一次(?:上升|提高|变好)", trend))
            has_down = bool(re.search(r"较上一次(?:下降|降低|变差)", trend))
            mixed_side_movement = has_up and has_down
            contradiction = (
                not mixed_side_movement
                and (
                    (delta > 0.005 and has_down)
                    or (delta < -0.005 and has_up)
                    or (abs(delta) <= 0.005 and (has_up or has_down))
                )
            )
            if contradiction:
                errors.append(
                    ValidationMessage(
                        severity="error",
                        code="TREND_SCORE_CONTRADICTION",
                        message="发展趋势与当前/上次评分的数值方向矛盾",
                        field="narrative.development_trend",
                    )
                )

        for recommendation in state.recommendations:
            if recommendation.category == "立即处置":
                evidence_text = " ".join(state.evidence[e].text for e in recommendation.evidence if e in state.evidence)
                if not EMERGENCY_EVIDENCE.search(evidence_text):
                    warnings.append(ValidationMessage(severity="warning", code="CATASTROPHE_GUARD", message=f"建议 {recommendation.sequence} 缺少立即处置强证据，已降级", field="recommendations"))
                    recommendation.category = "尽快维修"

        self._validate_semantic_graph(state, errors, warnings)

        supported_values = 0
        total_values = 0
        for _, getter in KEY_FIELDS:
            value = getter(state)
            if value.value not in (None, ""):
                total_values += 1
                supported_values += bool(value.evidence)
        for disease in state.diseases:
            total_values += 1
            supported_values += bool(disease.evidence)
        coverage = supported_values / total_values if total_values else 0.0
        return ValidationState(errors=errors, warnings=warnings, passed=not errors, evidence_coverage=coverage)

    @staticmethod
    def _validate_semantic_graph(
        state: DocumentState,
        errors: list[ValidationMessage],
        warnings: list[ValidationMessage],
    ) -> None:
        entities = {entity.side: entity for entity in state.scores.entities}
        if state.scores.presentation_mode == "dual" and not {"left", "right"}.issubset(entities):
            errors.append(ValidationMessage(
                severity="error",
                code="DUAL_SCORE_CARDINALITY",
                message="双幅评分缺少左幅或右幅实体",
                field="scores.entities",
            ))
        grade_rank = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
        for entity in state.scores.entities:
            for scope in ("overall", "superstructure", "substructure", "deck"):
                entry = getattr(entity, scope)
                numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(entry.score.value or ""))]
                if any(number < 0 or number > 100 for number in numbers):
                    errors.append(ValidationMessage(
                        severity="error",
                        code="ENTITY_SCORE_RANGE",
                        message=f"{entity.label}{scope}评分超出0至100范围",
                        field=f"scores.entities.{entity.entity_id}.{scope}",
                    ))
                if numbers and entry.grade.value:
                    grade_match = re.search(r"([A-E])", str(entry.grade.value), re.I)
                    expected = "A" if numbers[0] >= 90 else "B" if numbers[0] >= 80 else "C" if numbers[0] >= 66 else "D" if numbers[0] >= 50 else "E"
                    if grade_match and grade_rank[grade_match.group(1).upper()] < grade_rank[expected]:
                        message = ValidationMessage(
                            severity="error" if scope == "overall" else "warning",
                            code=(
                                "ENTITY_SCORE_GRADE_CONTRADICTION"
                                if scope == "overall"
                                else "ENTITY_COMPONENT_GRADE_REVIEW"
                            ),
                            message=f"{entity.label}{scope}等级优于评分阈值推导结果，保留报告显式评定",
                            field=f"scores.entities.{entity.entity_id}.{scope}.grade",
                        )
                        (errors if scope == "overall" else warnings).append(message)
        event_ids = {event.event_id for event in state.disease_events}
        for disease in state.diseases:
            dangling = set(disease.event_ids) - event_ids
            if dangling:
                errors.append(ValidationMessage(
                    severity="error",
                    code="DANGLING_DISEASE_EVENT",
                    message=f"病害引用不存在的事件：{sorted(dangling)}",
                    field="diseases.event_ids",
                ))
        for claim in state.narrative.claims:
            if set(claim.disease_event_ids) - event_ids:
                errors.append(ValidationMessage(
                    severity="error",
                    code="UNSUPPORTED_NARRATIVE_CLAIM",
                    message="叙述声明引用了不存在的病害事件",
                    field="narrative.claims",
                ))
            if any(evidence not in state.evidence for evidence in claim.evidence):
                errors.append(ValidationMessage(
                    severity="error",
                    code="DANGLING_NARRATIVE_EVIDENCE",
                    message="叙述声明引用了不存在的证据",
                    field="narrative.claims",
                ))
        for recommendation in state.recommendations:
            if state.disease_events and recommendation.component not in {"全桥", "桥梁总体"} and not recommendation.disease_event_ids:
                warnings.append(ValidationMessage(
                    severity="warning",
                    code="UNLINKED_RECOMMENDATION",
                    message=f"建议{recommendation.sequence}未关联到病害事件",
                    field="recommendations.disease_event_ids",
                ))

    @staticmethod
    def _numeric_control(value: object) -> float | None:
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(value or ""))]
        return min(numbers) if numbers else None
