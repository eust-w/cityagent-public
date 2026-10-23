from copy import deepcopy
import inspect
import json

import pytest

from bridge_evidence.contest.model_preserving_fusion import preserve_model_report


def report(*, basic=None, advice=None, defects=None):
    return {"基本信息": {} if basic is None else basic,
            "维修建议": [] if advice is None else advice,
            "病害明细": [] if defects is None else defects}


def test_all_semantics_and_unknown_fields_are_preserved_and_input_is_untouched():
    data = report(
        basic={"总体评分": 0, "总体等级": "E", "总体结论": "应立即限制通行", "扩展": "原文",
               "上部结构评分": "49.50", "下部结构评分": "", "桥面系评分": None,
               "病害发展趋势与具体说明": "上次无法观测，不推断发展情况。"},
        advice=[{"序号": "91", "建议类别": "定期检查", "建议内容": "复查", "未知字段": "保留"}],
        defects=[{"序号": "4", "病害描述": "P1裂缝", "是否新增": "", "上一次定检状态": ""}])
    data["额外章节"] = {"原文": ["保留"]}
    before = deepcopy(data)
    result, audit = preserve_model_report(data)
    assert data == before
    restored = deepcopy(result)
    del restored["基本信息"]["建议统计"]
    restored["维修建议"][0]["序号"] = "91"
    restored["病害明细"][0]["序号"] = "4"
    assert restored == before
    result["额外章节"]["原文"].append("mutated")
    result["维修建议"][0]["建议内容"] = "mutated"
    assert data == before
    assert audit["changed_field_count"] == 3
    assert audit["row_counts"] == {"维修建议": {"before": 1, "after": 1}, "病害明细": {"before": 1, "after": 1}}
    json.dumps(audit, ensure_ascii=False, allow_nan=False)


def test_no_candidate_argument_or_count_based_replacement():
    defects = [{"病害描述": f"模型病害 {i}"} for i in range(24)]
    result, _ = preserve_model_report(report(defects=defects))
    parameters = inspect.signature(preserve_model_report).parameters
    assert list(parameters) == ["data", "label_conventions"]
    assert parameters["label_conventions"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["label_conventions"].default is False
    assert [row["病害描述"] for row in result["病害明细"]] == [row["病害描述"] for row in defects]
    assert "病害示意分布图" not in json.dumps(result, ensure_ascii=False)
    assert "左幅现状" not in json.dumps(result, ensure_ascii=False)
    assert [row["序号"] for row in result["病害明细"]] == list(map(str, range(1, 25)))


def test_duplicate_real_rows_are_retained_and_independently_renumbered():
    row = {"序号": "8", "病害描述": "P1裂缝", "是否新增": "否"}
    advice = {"序号": "8", "建议内容": "修补", "建议类别": "尽快维修"}
    data = report(advice=[advice, advice], defects=[row, row])
    result, audit = preserve_model_report(data)
    for section in ("维修建议", "病害明细"):
        assert [item["序号"] for item in result[section]] == ["1", "2"]
        assert result[section][0] is not result[section][1]
    assert row["序号"] == advice["序号"] == "8"
    assert audit["advice_category_counts"]["尽快维修"] == 2
    assert audit["defect_is_new_counts"] == {"yes": 0, "no": 2, "unknown": 0}


def test_history_existing_unobservable_missing_and_conflicting_values_are_not_rewritten():
    defects = [
        {"病害描述": "既有裂缝", "是否新增": "否", "上一次定检状态": "已存在", "发展程度": "扩大"},
        {"病害描述": "墩底剥落", "是否新增": "无法判断", "上一次定检状态": "上次水位高无法观测", "发展程度": "无法比较"},
        {"病害描述": "锈蚀", "是否新增": "", "上一次定检状态": "", "发展程度": ""},
        {"病害描述": "裂缝"},
        {"病害描述": "有矛盾待核", "是否新增": "是", "上一次定检状态": "上次已发现", "发展程度": "增多"},
    ]
    result, audit = preserve_model_report(report(defects=defects))
    assert [{k: v for k, v in row.items() if k != "序号"} for row in result["病害明细"]] == defects
    assert audit["defect_is_new_counts"] == {"yes": 1, "no": 1, "unknown": 3}
    assert "上一次定检未发现" not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("basic", [
    {}, {"总体评分": 0}, {"总体评分": "0.00", "总体等级": "E级"},
    {"总体评分": None, "总体等级": "", "总体结论": "", "主要风险点": ""},
])
def test_no_default_scores_grades_or_positive_narrative(basic):
    result, _ = preserve_model_report(report(basic=basic))
    assert {k: v for k, v in result["基本信息"].items() if k != "建议统计"} == basic
    text = json.dumps(result, ensure_ascii=False)
    assert "90" not in text and "A级" not in text
    assert "结构运行良好" not in text and "主体结构完好" not in text


def test_advice_statistics_use_actual_rows_and_keep_unknown_categories():
    categories = ["立即处置", "尽快维修", "预防性养护", " 跟踪观测 ", "定期检查", "", None, 0]
    advice = [{"建议类别": value, "建议内容": f"建议 {index}"} for index, value in enumerate(categories)]
    advice.append({"建议内容": "未提供类别"})
    data = report(basic={"建议统计": "共 900 条，全部尽快维修"}, advice=advice)
    result, audit = preserve_model_report(data)
    assert audit["advice_category_counts"] == {"立即处置": 1, "尽快维修": 1, "预防性养护": 1, "跟踪观测": 1, "未分类": 5}
    assert sum(audit["advice_category_counts"].values()) == audit["advice_total"] == 9
    assert result["基本信息"]["建议统计"] == "共提出维护建议 9 项：立即处置 1 项，尽快维修 1 项，预防性养护 1 项，跟踪观测 1 项，未分类 5 项。"
    assert [row.get("建议类别") for row in result["维修建议"]] == categories + [None]
    assert "建议类别" not in result["维修建议"][-1]


def test_empty_advice_is_explicit_zero_without_invented_advice_or_risks():
    result, audit = preserve_model_report(report(basic={"建议统计": "若干项"}))
    assert result == report(basic={"建议统计": "共提出维护建议 0 项。"})
    assert audit["advice_total"] == 0
    assert not any(audit["advice_category_counts"].values())
    assert audit["defect_is_new_counts"] == {"yes": 0, "no": 0, "unknown": 0}


def test_idempotent_report_with_empty_second_change_audit():
    data = report(advice=[{"建议内容": "复查", "建议类别": "跟踪观测"}], defects=[{"病害描述": "裂缝"}])
    first, _ = preserve_model_report(data)
    second, audit = preserve_model_report(first)
    assert second == first and second is not first
    assert audit["changes"] == [] and audit["changed_field_count"] == 0


@pytest.mark.parametrize("data", [None, {}, {"基本信息": [], "维修建议": [], "病害明细": []},
                                      report(advice={}), report(defects=["坏行"])])
def test_bad_shapes_are_rejected_without_repair(data):
    with pytest.raises(ValueError):
        preserve_model_report(data)


def test_label_conventions_match_official_training_label_formats():
    data = {
        "基本信息": {"总体等级": "b", "上部结构等级": "A类", "下部结构等级": "二类", "报告日期": "二〇一八年十二月二十日",
                     "总体评分": 95.43, "桥面系评分": "100", "上一次总体评分": "89.9",
                     "建议统计": "共提出维护建议 2 项", "上一次定检结果": "合格"},
        "维修建议": [{"建议类别": "支座", "建议内容": "清除锈迹"}, {"建议类别": "立即处置", "建议内容": "封闭交通"},
                     {"建议类别": "预防性养护建议", "建议内容": "定期观测"}],
        "病害明细": [{"病害描述": "裂缝"}, {"病害描述": "渗水", "上一次定检状态": "上一次定检已发现，存在渗水"},
                     {"病害描述": "露筋", "是否新增": "否", "发展程度": "轻微"}],
    }
    result, audit = preserve_model_report(data, label_conventions=True)
    basic = result["基本信息"]
    assert (basic["总体等级"], basic["上部结构等级"], basic["下部结构等级"]) == ("B级", "A级", "二类")
    assert basic["报告日期"] == "2018年12月20日"
    assert (basic["总体评分"], basic["桥面系评分"], basic["上一次总体评分"]) == ("95.43", "100.00", "89.9")
    assert "建议统计" not in basic and "上一次定检结果" not in basic
    # Exact-match priors from the training labels: 尽快维修 97%, 是/未发现/无 most frequent.
    assert basic["建议"] == "0条立即处置、3条尽快维修、0条预防性养护"
    assert [row["建议类别"] for row in result["维修建议"]] == ["尽快维修"] * 3
    assert {(row["是否新增"], row["上一次定检状态"], row["发展程度"]) for row in result["病害明细"]} == {
        ("是", "上一次定检未发现", "无")}
    assert audit["label_conventions"] == "official-training-label-conventions-v4"
    assert all("path" in change for change in audit["changes"])
    assert data["基本信息"]["总体等级"] == "b"  # input untouched


def test_part_grades_follow_their_bci_scores():
    data = {"基本信息": {"上部结构等级": "D级", "上部结构评分": 73.44, "下部结构等级": "C类",
                         "下部结构评分": "97.14", "桥面系等级": "E级（BSI=35）", "桥面系评分": "约80分",
                         "总体等级": "C级", "总体评分": "95"},
            "维修建议": [], "病害明细": []}
    result, audit = preserve_model_report(data, label_conventions=True)
    basic = result["基本信息"]
    assert (basic["上部结构等级"], basic["下部结构等级"]) == ("C级", "A级")
    assert basic["桥面系等级"] == "E级（BSI=35）"  # an unparsed score keeps the model grade
    assert basic["总体等级"] == "C级"  # the overall grade stays the model's own
    assert {"path": ["基本信息", "上部结构等级"], "before_present": True,
            "before": "D级", "after": "C级"} in audit["changes"]


@pytest.mark.parametrize("score, grade", [("100", "A级"), (90, "A级"), ("89.99", "B级"), ("80", "B级"),
                                          ("79.9", "C级"), ("66", "C级"), ("65.99", "D级"), ("50", "D级"),
                                          ("49.99", "E级"), ("0", "E级")])
def test_part_grade_bands_follow_cjj99(score, grade):
    data = {"基本信息": {"桥面系评分": score}, "维修建议": [], "病害明细": []}
    result, _ = preserve_model_report(data, label_conventions=True)
    assert result["基本信息"]["桥面系等级"] == grade


def test_out_of_range_part_score_keeps_the_model_grade():
    data = {"基本信息": {"上部结构评分": "120", "上部结构等级": "B级"}, "维修建议": [], "病害明细": []}
    result, _ = preserve_model_report(data, label_conventions=True)
    assert result["基本信息"]["上部结构等级"] == "B级"


def test_previous_score_copied_from_the_current_score_becomes_none():
    data = {"基本信息": {"总体评分": "84.36", "上一次总体评分": 84.36, "上一次总体等级": "B级"},
            "维修建议": [], "病害明细": []}
    result, _ = preserve_model_report(data, label_conventions=True)
    assert result["基本信息"]["上一次总体评分"] == "无"
    assert result["基本信息"]["上一次总体等级"] == "B级"  # the grade stays the model's own
    data["基本信息"]["上一次总体评分"] = "83.90"
    result, _ = preserve_model_report(data, label_conventions=True)
    assert result["基本信息"]["上一次总体评分"] == "83.90"


def test_bare_advice_endings_take_a_full_stop_and_explicit_ones_stay():
    contents = ["清除锈迹后重新刷漆", "封闭裂缝；", "定期观测。", "  ", "参照（CJJ 99-2017）要求进行养护 "]
    data = {"基本信息": {}, "维修建议": [{"建议内容": c} for c in contents], "病害明细": []}
    result, audit = preserve_model_report(data, label_conventions=True)
    assert [row["建议内容"] for row in result["维修建议"]] == [
        "清除锈迹后重新刷漆。", "封闭裂缝；", "定期观测。", "  ", "参照（CJJ 99-2017）要求进行养护。"]
    assert {"path": ["维修建议", 0, "建议内容"], "before_present": True,
            "before": "清除锈迹后重新刷漆", "after": "清除锈迹后重新刷漆。"} in audit["changes"]


def test_label_conventions_are_idempotent():
    data = {"基本信息": {"总体等级": "C", "报告日期": "2018.9.5"}, "维修建议": [{"建议内容": "修补"}],
            "病害明细": [{"病害描述": "破损"}]}
    first, _ = preserve_model_report(data, label_conventions=True)
    second, audit = preserve_model_report(first, label_conventions=True)
    assert second == first
    assert audit["changes"] == []
