from __future__ import annotations

import unittest

from bridge_evidence.models.schema import Disease, DocumentState, EvidenceRef, FieldValue, Recommendation
from bridge_evidence.models.validation import ConstraintEngine


class ConstraintEngineTest(unittest.TestCase):
    def _state(self) -> DocumentState:
        state = DocumentState(document_id="sha256:test", source_path="test.docx")
        state.add_evidence(EvidenceRef(id="E1", kind="table", text="BCI=90.1，A级", source_path="test.docx", priority=90))
        state.bridge.name = FieldValue(value="测试桥", evidence=["E1"], source_priority=90, confidence=1)
        state.bridge.report_date = FieldValue(value="2026年1月1日", evidence=["E1"], source_priority=90, confidence=1)
        state.scores.overall.score = FieldValue(value=90.1, evidence=["E1"], source_priority=90, confidence=1)
        state.scores.overall.grade = FieldValue(value="A级", evidence=["E1"], source_priority=90, confidence=1)
        return state

    def test_valid_state_passes(self) -> None:
        result = ConstraintEngine().validate(self._state())
        self.assertTrue(result.passed)

    def test_catastrophe_guard_downgrades_unsupported_emergency(self) -> None:
        state = self._state()
        state.recommendations.append(Recommendation(sequence=1, category="立即处置", content="建议修补", evidence=["E1"]))
        result = ConstraintEngine().validate(state)
        self.assertEqual(state.recommendations[0].category, "尽快维修")
        self.assertTrue(any(item.code == "CATASTROPHE_GUARD" for item in result.warnings))

    def test_history_coherence_guard_survives_late_agent_patch(self) -> None:
        state = self._state()
        state.diseases.append(
            Disease(
                sequence="1",
                component="桥台",
                location="0#台",
                type="渗水",
                description="0#台新增渗水",
                is_new=True,
                previous_state="已发现",
                evidence=["E1"],
            )
        )

        result = ConstraintEngine().validate(state)

        self.assertEqual(state.diseases[0].previous_state, "未发现")
        self.assertTrue(any(item.code == "HISTORY_COHERENCE" for item in result.warnings))

    def test_control_score_requires_evidence(self) -> None:
        state = self._state()
        state.scores.deck.control_score = FieldValue(value=35.0)

        result = ConstraintEngine().validate(state)

        self.assertFalse(result.passed)
        self.assertTrue(any(item.code == "UNSUPPORTED_CONTROL_SCORE" for item in result.errors))

    def test_exact_history_provenance_restores_non_new_coherence(self) -> None:
        state = self._state()
        state.diseases.append(Disease(
            sequence="1",
            component="主梁",
            location="1#梁",
            type="裂缝",
            description="1#梁裂缝",
            is_new=False,
            novelty_source="history_exact",
            previous_state="未发现",
            development="无",
            evidence=["E1"],
        ))

        result = ConstraintEngine().validate(state)

        self.assertEqual(state.diseases[0].previous_state, "已发现")
        self.assertFalse(state.diseases[0].is_new)
        self.assertTrue(any(item.code == "EXACT_HISTORY_COHERENCE" for item in result.warnings))

    def test_non_disease_presentation_label_is_rejected(self) -> None:
        state = self._state()
        state.diseases.append(Disease(
            sequence="1",
            component="其他",
            location="桥梁全景",
            type="桥涵标现状",
            description="桥涵标现状照片",
            evidence=["E1"],
        ))

        result = ConstraintEngine().validate(state)

        self.assertFalse(result.passed)
        self.assertTrue(any(item.code == "NON_DISEASE_LABEL" for item in result.errors))

    def test_mixed_dual_carriageway_trend_is_not_rejected_by_mean_direction(self) -> None:
        state = self._state()
        state.scores.overall.score = FieldValue(value="87.91（左幅）/88.64（右幅）", evidence=["E1"])
        state.scores.previous.score = FieldValue(value=88.15, evidence=["E1"])
        state.narrative.development_trend = (
            "总体评分左幅较上一次下降0.24分，右幅较上一次上升0.49分。"
        )

        result = ConstraintEngine().validate(state)

        self.assertTrue(result.passed)

    def test_bare_condition_state_label_is_allowed(self) -> None:
        state = self._state()
        state.diseases.append(Disease(
            sequence="1",
            component="人行道",
            location="人行道",
            type="现状",
            description="外观良好",
            evidence=["E1"],
        ))

        result = ConstraintEngine().validate(state)

        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
