from __future__ import annotations

# Hand-constructed fixtures use synthetic bridge and organization names.

from pathlib import Path

from bridge_evidence.document_ai.schema import CandidateValue
from bridge_evidence.document_ai.student import CalibratedDocumentStudent
from bridge_evidence.document_ai.vision_candidates import best_page_numbers, extract_field_candidates, normalize_candidate


def test_visual_candidate_extractor_emits_schema_valid_bridge_fields() -> None:
    text = "桥梁名称：K0+123.45合成双幅大桥 检测日期：2019年11月20日 全桥技术状况指数BCI为88.82，B级"

    candidates = extract_field_candidates(
        text,
        expert="vision",
        channel="visual",
        region_id="IMG1",
    )

    validated = [CandidateValue.model_validate(candidate) for candidate in candidates]
    values = {candidate.field: candidate.value for candidate in validated}
    assert values["bridge.name"] == "K0+123.45合成双幅大桥"
    assert values["bridge.report_date"] == "2019年11月20日"
    assert values["scores.overall.score"] == 88.82
    assert values["scores.overall.grade"] == "B级"


def test_page_router_prioritizes_score_summary() -> None:
    pages = ["封面", "目录", "桥面系一般描述", "技术状况评定 BCI 88.82 BSIm 90.1 BSIs 80.2 BSIx 95.0"]

    assert best_page_numbers(pages, limit=2) == [4]


def test_dual_carriageway_values_are_atomic_and_strict() -> None:
    truth = "93.83（左幅）/86.56（右幅）"

    assert normalize_candidate("scores.overall.score", truth) == truth
    assert normalize_candidate("scores.overall.score", "93.83") != normalize_candidate("scores.overall.score", truth)
    assert normalize_candidate("scores.overall.grade", "A级（左幅）/B级（右幅）") == "A级（左幅）/B级（右幅）"
    assert normalize_candidate("scores.overall.grade", "B级") != normalize_candidate(
        "scores.overall.grade", "A级（左幅）/B级（右幅）"
    )


def test_calibrated_student_round_trip_and_shrinks_weak_field(tmp_path: Path) -> None:
    model = CalibratedDocumentStudent.fit(
        [("scores.overall.score", True)] * 8
        + [("scores.overall.score", False)]
        + [("bridge.name", False)] * 8
    )
    path = tmp_path / "student.json"
    model.save(path)

    restored = CalibratedDocumentStudent.load(path)

    assert restored.confidence("scores.overall.score", 0.9) > 0.8
    assert restored.confidence("bridge.name", 0.9) < 0.5
