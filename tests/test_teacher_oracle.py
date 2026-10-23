from pathlib import Path
import runpy

from bridge_evidence.evaluation.teacher import compare_teacher_target, semantic_signature
from bridge_evidence.parsers.docx import ParsedDocument, TableBlock


benchmark_convention_conflicts = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/build_official_teacher_samples.py"),
)["benchmark_convention_conflicts"]


def test_semantic_signature_normalizes_numeric_rendering_but_keeps_side_order() -> None:
    assert semantic_signature("81.360（左幅）/90（右幅）") == semantic_signature("81.36（左幅）/90.0（右幅）")
    assert semantic_signature("81.36（右幅）/90.0（左幅）") != semantic_signature("81.36（左幅）/90.0（右幅）")


def test_teacher_builder_marks_source_underdetermined_label_conventions() -> None:
    parsed = ParsedDocument(
        path=Path("sample.docx"),
        paragraphs=[],
        tables=[
            TableBlock(index=0, rows=[["工程名称", "示例桥（K1+200）"]]),
            TableBlock(
                index=1,
                rows=[["部位", "技术状况指数"], ["桥面系", "81.50"]],
                context=["左幅技术状况评定表"],
            ),
            TableBlock(
                index=2,
                rows=[["部位", "技术状况指数"], ["桥面系", "82.08"]],
                context=["右幅技术状况评定表"],
            ),
        ],
    )

    conflicts = benchmark_convention_conflicts(parsed, {
        "桥梁名称": "示例桥",
        "桥面系评分": "81.50",
    })

    assert set(conflicts) == {"bridge.name", "scores.deck.score"}


def test_compare_teacher_target_excludes_adjudication_only_field_from_objective() -> None:
    state = {
        "scores": {"overall": {"score": {"value": 91.3}, "grade": {"value": "A级"}}},
        "bridge": {"report_date": {"value": "2018年12月20日"}},
    }
    teacher = {
        "teacher_id": "teacher:test",
        "targets": {
            "scores.overall.score": {"target_value": "91.30"},
            "scores.overall.grade": {"target_value": "A级"},
            "bridge.report_date": {
                "target_value": "2018年12月31日",
                "include_in_objective": False,
                "status": "benchmark_convention_conflict",
            },
        },
    }

    result = compare_teacher_target(state, teacher)

    assert result["objective_score"] == 1.0
    assert result["matched_fields"] == 2
    assert [item["field"] for item in result["differences"]] == ["bridge.report_date"]
