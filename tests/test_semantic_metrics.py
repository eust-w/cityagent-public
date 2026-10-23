from __future__ import annotations

from pathlib import Path

import pytest

from bridge_evidence.evaluation.labels import LabelReport, canonical_header
from bridge_evidence.evaluation.metrics import semantic_proxy_scores
from bridge_evidence.models.schema import Disease, DocumentState, FieldValue, Recommendation


def test_label_header_variants_are_canonicalized_without_touching_values() -> None:
    assert canonical_header("病害位置") == "病害部位"
    assert canonical_header("病害描述（含照片编号）") == "病害描述"
    assert canonical_header("上一次定检状态（2014年）") == "上一次定检状态"


def test_semantic_proxy_rewards_exact_rows_and_penalizes_count_mismatch() -> None:
    state = DocumentState(document_id="sha256:metric", source_path="metric.docx")
    state.bridge.name = FieldValue(value="测试桥", evidence=["E"])
    state.bridge.report_date = FieldValue(value="2019年1月", evidence=["E"])
    for scope, score, grade in (
        ("overall", 88.0, "B级"),
        ("superstructure", 87.0, "B级"),
        ("substructure", 90.0, "A级"),
        ("deck", 85.0, "B级"),
    ):
        entry = getattr(state.scores, scope)
        entry.score = FieldValue(value=score, evidence=["E"])
        entry.grade = FieldValue(value=grade, evidence=["E"])
    state.scores.previous.score = FieldValue(value="无", evidence=["E"])
    state.scores.previous.grade = FieldValue(value="无", evidence=["E"])
    state.narrative.overall_conclusion = "桥梁总体状况良好。"
    state.narrative.key_risks = "主梁裂缝影响耐久性。"
    state.narrative.development_trend = "首次检测，无对比年度。"
    state.validation.passed = True
    state.recommendations = [
        Recommendation(sequence=1, category="尽快维修", content="建议及时修补主梁裂缝。", component="主梁")
    ]
    state.diseases = [
        Disease(
            sequence="1",
            component="主梁",
            location="1#梁跨中",
            type="裂缝",
            description="1#梁跨中裂缝",
            is_new=False,
            previous_state="无",
            development="无",
        )
    ]
    label = LabelReport(
        path=Path("label.docx"),
        summary={
            "桥梁名称": "测试桥",
            "报告日期": "2019年1月",
            "总体评分": "88.0",
            "总体等级": "B级",
            "上部结构评分": "87.0",
            "上部结构等级": "B级",
            "下部结构评分": "90.0",
            "下部结构等级": "A级",
            "桥面系评分": "85.0",
            "桥面系等级": "B级",
            "上一次总体评分": "无",
            "上一次总体等级": "无",
            "病害发展趋势与具体说明": "首次检测，无对比年度。",
            "总体结论": "桥梁总体状况良好。",
            "主要风险点": "主梁裂缝影响耐久性。",
            "建议": "0条立即处置、1条尽快维修、0条预防性养护",
        },
        recommendations=[{"建议类别": "尽快维修", "建议内容": "建议及时修补主梁裂缝。", "病害部位": "主梁"}],
        diseases=[
            {
                "病害部位": "1#梁跨中",
                "病害类型": "裂缝",
                "病害描述": "1#梁跨中裂缝",
                "是否新增": "否",
                "上一次定检状态": "无",
                "发展程度": "无",
            }
        ],
    )

    exact = semantic_proxy_scores(state, label)
    assert exact["summary_fact_accuracy"] == 1.0
    assert exact["summary_text_consistency"] == 1.0
    assert exact["detail_text_consistency"] == pytest.approx(1.0)

    state.diseases.append(
        Disease(sequence="2", component="桥台", location="0#桥台", type="破损", description="0#桥台破损")
    )
    with_extra = semantic_proxy_scores(state, label)
    assert with_extra["detail_text_consistency"] < exact["detail_text_consistency"]
