from __future__ import annotations

from copy import deepcopy
import json

import pytest

from bridge_evidence.evaluation.generation import aggregate_reports, score_report


def report(*, basic=None, advice=None, defects=None):
    return {"基本信息": {} if basic is None else basic,
            "维修建议": [] if advice is None else advice,
            "病害明细": [] if defects is None else defects}


def defect(description="P1裂缝", number="1"):
    return {"序号": number, "病害部位": "P1", "病害类型": "裂缝",
            "病害描述": description, "是否新增": "否", "上一次定检状态": "无", "发展程度": "无"}


def test_missing_prediction_counts_all_gold_as_false_negatives():
    gold = report(basic={"桥梁名称": "东桥", "总体评分": "80"},
                  advice=[{"序号": "1", "建议内容": "维修桥面"}], defects=[defect(), defect()])
    score = score_report(gold, None)
    assert score["prediction_valid"] is False
    assert score["overall"] == {"tp": 0, "fp": 0, "fn": 5, "predicted_count": 0,
                                "gold_count": 5, "precision": 0, "recall": 0, "f1": 0}
    aggregate = aggregate_reports([score, score])
    assert aggregate["reports_total"] == aggregate["reports_failed"] == 2
    assert aggregate["overall"]["fn"] == 10
    assert aggregate["sections"]["defects"]["f1"] == 0


def test_wrong_and_extra_basic_values_count_fp_and_fn():
    score = score_report(report(basic={"桥梁名称": "东桥", "总体评分": "80"}),
                         report(basic={"桥梁名称": "西桥", "总体评分": 80, "不存在字段": "编造"}))
    section = score["sections"]["basic"]
    assert (section["tp"], section["fp"], section["fn"]) == (1, 2, 1)
    assert section["precision"] == pytest.approx(1 / 3)
    assert section["recall"] == 0.5


def test_duplicate_rows_are_multisets_and_not_overwritten_by_location_type():
    first, second = defect("P1左侧裂缝"), defect("P1右侧裂缝", "2")
    gold = report(defects=[first, first, second])
    prediction = report(defects=[second, first])
    section = score_report(gold, prediction)["sections"]["defects"]
    assert (section["tp"], section["fp"], section["fn"]) == (2, 0, 1)
    assert section["gold_count"] == 3
    duplicate = score_report(report(defects=[first]), report(defects=[first, first]))["sections"]["defects"]
    assert (duplicate["tp"], duplicate["fp"], duplicate["fn"]) == (1, 1, 0)


def test_extra_advice_and_disease_rows_reduce_precision():
    advice = {"建议类别": "尽快维修", "病害部位": "桥面", "建议内容": "修补"}
    gold = report(advice=[advice], defects=[defect()])
    predicted = report(advice=[advice, {"建议内容": "多报"}], defects=[defect(), defect("多报")])
    for section in ("advice", "defects"):
        result = score_report(gold, predicted)["sections"][section]
        assert result["precision"] == 0.5
        assert result["recall"] == 1.0
        assert result["f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("gold,predicted", [
    ("P1裂缝", "P1锈蚀"), ("东桥1号", "西桥1号"),
    ("裂缝宽0.5 mm", "裂缝宽0.5 m"), ("2018年1月2日", "2018年12月1日"),
    ("左幅90/右幅80", "右幅90/左幅80"),
])
def test_words_dates_units_and_component_score_meaning_are_preserved(gold, predicted):
    value = score_report(report(basic={"总体评分": gold}), report(basic={"总体评分": predicted}))
    assert (value["overall"]["tp"], value["overall"]["fp"], value["overall"]["fn"]) == (0, 1, 1)


def test_literal_none_and_numeric_zero_are_not_missing_values():
    gold = report(basic={"总体评分": "0.00", "上一次定检状态": "无", "空字段": " "})
    score = score_report(gold, report(basic={"总体评分": 0, "上一次定检状态": "无", "空字段": None}))
    assert score["overall"]["tp"] == 2
    missing = score_report(gold, report(basic={"总体评分": None, "上一次定检状态": ""}))
    assert missing["overall"]["fn"] == 2


def test_row_numbers_and_order_are_ignored_but_unknown_semantic_fields_are_retained():
    left, right = defect("left", "1"), defect("right", "2")
    prediction = [dict(right, 序号="99"), dict(left, 序号="98")]
    assert score_report(report(defects=[left, right]), report(defects=prediction))["overall"]["f1"] == 1
    prediction[0]["hallucinated"] = "value"
    value = score_report(report(defects=[left, right]), report(defects=prediction))["sections"]["defects"]
    assert (value["tp"], value["fp"], value["fn"]) == (1, 1, 1)


def test_format_normalization_is_deterministic_and_does_not_mutate_inputs():
    gold = report(basic={"桥梁名称": "P1  东桥", "总体评分": "080.00"})
    prediction = report(basic={"桥梁名称": "Ｐ１\n 东桥", "总体评分": 80.0})
    before = deepcopy((gold, prediction))
    assert score_report(gold, prediction)["overall"]["f1"] == 1
    assert (gold, prediction) == before


def test_failures_stay_in_micro_and_macro_denominators():
    gold = report(basic={"桥梁名称": "东桥"}, defects=[defect()])
    aggregate = aggregate_reports([score_report(gold, gold), score_report(gold, None)])
    assert aggregate["reports_valid"] == aggregate["reports_failed"] == 1
    assert aggregate["valid_report_rate"] == 0.5
    assert aggregate["overall"]["recall"] == 0.5
    assert aggregate["overall"]["f1"] == pytest.approx(2 / 3)
    assert aggregate["sections"]["defects"]["macro_report_f1"] == 0.5
    assert aggregate["sections"]["defects"]["macro_report_support"] == 2


def test_empty_sections_give_no_free_credit_and_output_is_json_serializable():
    score = score_report(report(), report())
    assert score["macro_section_f1"] is None
    assert score["overall"]["f1"] is None
    aggregate = aggregate_reports([])
    assert aggregate["reports_total"] == 0
    assert aggregate["valid_report_rate"] is None
    assert aggregate["overall"]["f1"] is None
    json.dumps([score, aggregate], allow_nan=False)


def test_empty_prediction_row_is_counted_as_an_extra_row():
    score = score_report(report(), report(defects=[{}]))
    assert score["sections"]["defects"]["fp"] == 1
    assert score["sections"]["defects"]["f1"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, [], {}])
def test_invalid_scalar_values_fail_instead_of_corrupting_metric(value):
    with pytest.raises(ValueError):
        score_report(report(basic={"总体评分": "80"}), report(basic={"总体评分": value}))


def test_aggregate_rejects_mixed_versions_or_invalid_counts():
    score = score_report(report(basic={"桥梁名称": "桥"}), None)
    changed = deepcopy(score)
    changed["metric_version"] = "different"
    with pytest.raises(ValueError):
        aggregate_reports([changed])
    changed = deepcopy(score)
    changed["sections"]["basic"]["tp"] = 1
    with pytest.raises(ValueError):
        aggregate_reports([changed])
