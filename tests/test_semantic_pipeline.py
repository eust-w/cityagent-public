from __future__ import annotations

from pathlib import Path

from bridge_evidence.document_ai.consensus import accepted_table_region_ids
from bridge_evidence.document_ai.schema import DocumentRegion
from bridge_evidence.evaluation.metrics import evaluation_strata
from bridge_evidence.evidence.disease_graph import DiseaseGraphBuilder, published_diseases
from bridge_evidence.evidence.extractors import EvidenceExtractor
from bridge_evidence.evidence.narrative import NarrativePlanner
from bridge_evidence.models.schema import (
    Disease,
    DocumentState,
    EntityScoreState,
    EvidenceRef,
    FieldValue,
)
from bridge_evidence.models.validation import ConstraintEngine
from bridge_evidence.orchestrator import BridgeEvidenceOrchestrator
from bridge_evidence.parsers.docx import ParsedDocument, TableBlock


def _rating_table(index: int, side: str, overall: float, deck: float) -> TableBlock:
    return TableBlock(
        index=index,
        rows=[
            ["部位名称", "技术状况指数", "权重", "BCI", "桥梁整体技术状况等级"],
            ["桥面系", str(deck), "0.15", str(overall), "B"],
            ["上部结构", "82.0", "0.40", str(overall), "B"],
            ["下部结构", "90.0", "0.45", str(overall), "B"],
        ],
        context=[f"桥梁{side}技术状况评定表"],
    )


def test_extractor_preserves_dual_carriageway_as_typed_entities() -> None:
    parsed = ParsedDocument(
        path=Path("dual.docx"),
        paragraphs=[],
        tables=[_rating_table(0, "左幅", 88.82, 91.0), _rating_table(1, "右幅", 77.94, 76.0)],
    )
    state = DocumentState(document_id="sha256:dual", source_path="dual.docx")

    EvidenceExtractor()._extract_scores(parsed, state, None)

    assert state.scores.presentation_mode == "dual"
    assert state.scores.overall.score.value == "88.82（左幅）/77.94（右幅）"
    assert {entity.side for entity in state.scores.entities} == {"whole", "left", "right"}
    right = next(entity for entity in state.scores.entities if entity.side == "right")
    assert right.deck.score.value == 76.0


def test_disease_graph_retains_atomic_events_and_links_aggregate() -> None:
    atomic = [
        Disease(sequence=str(index), component="主梁", location=f"左幅1跨{index}#梁底", type="裂缝", description="纵向裂缝", evidence=[f"E{index}"])
        for index in range(1, 4)
    ]
    aggregate = Disease(sequence="1", component="主梁", location="左幅1跨主梁", type="裂缝", description="多片梁底纵向裂缝")
    events = DiseaseGraphBuilder.build(atomic, "doc")

    DiseaseGraphBuilder.attach_aggregates([aggregate], events)

    assert len(events) == 3
    assert len(set(event.event_id for event in events)) == 3
    assert aggregate.event_ids == [event.event_id for event in events]


def test_published_diseases_restore_atomic_rows_and_inherit_history() -> None:
    atomic = [
        Disease(
            sequence=str(index),
            component="主梁",
            location=f"左幅1跨{index}#梁底",
            type="裂缝",
            description=f"左幅1跨{index}#梁底纵向裂缝",
            evidence=[f"E{index}"],
            confidence=0.98,
        )
        for index in range(1, 4)
    ]
    aggregate = Disease(
        sequence="1",
        component="主梁",
        location="左幅1跨主梁",
        type="裂缝",
        description="多片梁底纵向裂缝",
        is_new=False,
        previous_state="已发现类似病害",
        development="本次基本维持",
        confidence=0.95,
    )
    state = DocumentState(document_id="sha256:published", source_path="published.docx")
    state.diseases = [aggregate]
    state.disease_events = DiseaseGraphBuilder.build(atomic, state.document_id)
    DiseaseGraphBuilder.attach_aggregates(state.diseases, state.disease_events)

    rows = published_diseases(state)

    assert len(rows) == 3
    assert [row.location for row in rows] == [item.location for item in atomic]
    assert all(row.is_new is False for row in rows)
    assert all(row.previous_state == "已发现类似病害" for row in rows)
    assert all(len(row.event_ids) == 1 for row in rows)


def test_narrative_planner_emits_evidence_bound_mechanism_claims() -> None:
    state = DocumentState(document_id="sha256:narrative", source_path="test.docx")
    state.diseases = [Disease(
        sequence="1",
        component="支座",
        location="左幅1#墩支座",
        type="剪切变形",
        description="支座发生剪切变形",
        event_ids=["DE:1"],
        evidence=["E1"],
        confidence=0.9,
    )]

    NarrativePlanner.plan(state)

    assert any("传力" in claim.text for claim in state.narrative.claims)
    assert all(claim.disease_event_ids == ["DE:1"] for claim in state.narrative.claims)
    assert "支座" in state.narrative.key_risks


def test_narrative_planner_preserves_compact_risk_coverage_summary() -> None:
    state = DocumentState(document_id="sha256:risk-summary", source_path="test.docx")
    state.diseases = [Disease(
        sequence="1",
        component="主梁",
        location="左幅1#梁底",
        type="裂缝",
        description="主梁纵向裂缝",
        event_ids=["DE:1"],
        evidence=["E1"],
        confidence=0.98,
    )]
    state.narrative.key_risks = "主梁裂缝8处；支座剪切变形2处；桥面铺装破损3处。"

    NarrativePlanner.plan(state)

    assert state.narrative.key_risks == "主梁裂缝8处；支座剪切变形2处；桥面铺装破损3处。"
    assert any(claim.claim_type == "risk" for claim in state.narrative.claims)


def test_semantic_validator_rejects_incomplete_dual_cardinality() -> None:
    state = DocumentState(document_id="sha256:invalid", source_path="test.docx")
    state.add_evidence(EvidenceRef(id="E1", kind="table", text="测试", source_path="test.docx"))
    state.bridge.name = FieldValue(value="测试桥", evidence=["E1"])
    state.bridge.report_date = FieldValue(value="2018年1月", evidence=["E1"])
    state.scores.overall.score = FieldValue(value="88.0（左幅）/77.0（右幅）", evidence=["E1"])
    state.scores.overall.grade = FieldValue(value="B级（左幅）/C级（右幅）", evidence=["E1"])
    state.scores.presentation_mode = "dual"
    state.scores.entities = [EntityScoreState(entity_id="bridge:left", label="左幅", side="left")]

    result = ConstraintEngine().validate(state)

    assert not result.passed
    assert any(message.code == "DUAL_SCORE_CARDINALITY" for message in result.errors)


def test_component_explicit_grade_difference_is_review_warning_not_failure() -> None:
    state = DocumentState(document_id="sha256:pedestrian", source_path="test.docx")
    state.add_evidence(EvidenceRef(id="E1", kind="table", text="桥面系评定为A级", source_path="test.docx"))
    state.bridge.name = FieldValue(value="测试人行天桥", evidence=["E1"])
    state.bridge.report_date = FieldValue(value="2018年1月", evidence=["E1"])
    state.scores.overall.score = FieldValue(value=91.0, evidence=["E1"])
    state.scores.overall.grade = FieldValue(value="A级", evidence=["E1"])
    entity = EntityScoreState(entity_id="bridge:whole", label="整桥", side="whole")
    entity.deck.score = FieldValue(value=85.0, evidence=["E1"])
    entity.deck.grade = FieldValue(value="A级", evidence=["E1"])
    state.scores.entities = [entity]

    result = ConstraintEngine().validate(state)

    assert result.passed
    assert any(message.code == "ENTITY_COMPONENT_GRADE_REVIEW" for message in result.warnings)


def test_external_history_cannot_manufacture_previous_score() -> None:
    state = DocumentState(document_id="sha256:history-score", source_path="test.docx")
    state.scores.previous.score = FieldValue(value=89.05, evidence=["HISTORY"])
    state.scores.previous.grade = FieldValue(value="B级", evidence=["SOURCE"])
    source_score = FieldValue(value="无", evidence=["D-ABSENT-SCORE-FIELDS"], confidence=0.35)

    BridgeEvidenceOrchestrator._restore_absent_source_previous_score(state, source_score)

    assert state.scores.previous.score.value == "无"
    assert state.scores.previous.score.evidence == ["D-ABSENT-SCORE-FIELDS"]
    assert state.scores.previous.grade.value == "B级"


def test_table_consensus_accepts_minor_ocr_variation() -> None:
    left = DocumentRegion(
        id="ocr:table",
        kind="disease_table",
        rows=[["病害位置", "病害类型", "病害描述"], ["左幅1跨1#梁底", "裂缝", "纵向裂缝0.5m"]],
        confidence=0.94,
    )
    right = DocumentRegion(
        id="vlm:table",
        kind="disease_table",
        rows=[["病害部位", "病害类型", "病害描述"], ["左幅1跨1号梁底", "裂缝", "纵向裂缝 0.5m"]],
        confidence=0.91,
    )

    assert accepted_table_region_ids({"ocr": [left], "vlm": [right]}, min_agreement=2) == {"ocr:table"}


def test_evaluation_strata_expose_cardinality_format_and_volume() -> None:
    state = DocumentState(document_id="sha256:strata", source_path="sample.doc")
    state.converted_path = "sample.docx"
    state.scores.presentation_mode = "dual"
    state.diseases = [Disease(sequence=str(i), component="主梁", type="裂缝", description="裂缝") for i in range(12)]

    assert evaluation_strata(state) == [
        "format:.doc",
        "score_cardinality:dual",
        "disease_volume:medium",
        "conversion:converted",
    ]
