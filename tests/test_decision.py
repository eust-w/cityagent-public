from __future__ import annotations

from bridge_evidence.agents.decision import DecisionWriter, apply_fallback_decision, score_development_trend
from bridge_evidence.config import Settings
from bridge_evidence.models.schema import Disease, DocumentState, FieldValue


def test_fallback_decision_closes_missing_summary_without_inventing_scores() -> None:
    state = DocumentState(document_id="sha256:decision", source_path="report.docx")
    state.scores.overall.score = FieldValue(value=85.2, evidence=["SOURCE"], confidence=1.0)
    state.scores.overall.grade = FieldValue(value="B级", evidence=["SOURCE"], confidence=1.0)
    state.diseases = [
        Disease(
            sequence="1",
            component="主梁",
            location="1#梁跨中",
            type="裂缝",
            description="1#梁跨中裂缝",
            is_new=True,
            evidence=["SOURCE"],
        )
    ]

    apply_fallback_decision(state)

    assert "总体评分为85.2分" in state.narrative.overall_conclusion
    assert "主梁裂缝1处" in state.narrative.key_risks
    assert "新增病害1处" in state.narrative.development_trend
    assert state.scores.deck.score.value == "无"
    assert state.scores.deck.score.evidence == ["D-ABSENT-SCORE-FIELDS"]
    assert "D-ABSENT-SCORE-FIELDS" in state.evidence


def test_fallback_decision_preserves_direct_source_narratives() -> None:
    state = DocumentState(document_id="sha256:decision-source", source_path="report.docx")
    state.narrative.development_trend = "原报告趋势。"
    state.narrative.overall_conclusion = "原报告结论。"
    state.narrative.key_risks = "原报告风险。"

    apply_fallback_decision(state)

    assert state.narrative.development_trend == "原报告趋势。"
    assert state.narrative.overall_conclusion == "原报告结论。"
    assert state.narrative.key_risks == "原报告风险。"


def test_score_trend_uses_worst_dual_carriageway_as_control() -> None:
    state = DocumentState(document_id="sha256:dual-trend", source_path="dual.docx")
    state.scores.overall.score = FieldValue(value="87.91（左幅）/88.64（右幅）")
    state.scores.previous.score = FieldValue(value=88.15)

    assert score_development_trend(state) == "总体评分较上一次下降0.24分。"


class _DecisionClient:
    def __init__(self) -> None:
        self.payload = None

    def complete_model(self, response_model, *, user_payload, **_: object):
        self.payload = user_payload
        return response_model(**{
            "development_trend": "无法量化比较。",
            "overall_conclusion": "按证据评定。",
            "causes": [],
            "safety_impacts": [],
            "key_risks": "关注既有病害。",
        })


def test_decision_writer_accepts_diseases_without_history_fields() -> None:
    state = DocumentState(document_id="sha256:decision-online", source_path="report.docx")
    state.diseases = [Disease(
        sequence="1",
        component="主梁",
        location="1#梁",
        type="裂缝",
        description="1#梁裂缝",
        previous_state=None,
        development=None,
    )]
    client = _DecisionClient()

    result = DecisionWriter(client, Settings(llm_mode="off"), "system").write(state)

    assert result.overall_conclusion == "按证据评定。"
    assert client.payload["disease_details"][0]["previous_state"] == ""
    assert client.payload["disease_details"][0]["development"] == ""
