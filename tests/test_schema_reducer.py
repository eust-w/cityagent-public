from copy import deepcopy
import json

import pytest

from bridge_evidence.contest.schema_reducer import (
    ChunkValidationError, EvidenceReducer, UnresolvedConflictsError, parse_chunk_result,
)


def chunk(*, basic=None, advice=None, defects=None, source="P1"):
    basic = {} if basic is None else basic
    advice = [] if advice is None else advice
    defects = [] if defects is None else defects
    return {"基本信息": basic, "维修建议": advice, "病害明细": defects,
            "evidence": {"基本信息": {key: [source] for key, value in basic.items()
                                       if value is not None and value != ""},
                         "维修建议": [[source] for _ in advice],
                         "病害明细": [[source] for _ in defects]}}


def parse(data, allowed=None, primary=None):
    return parse_chunk_result(json.dumps(data, ensure_ascii=False),
                              {"P1", "P2", "T1:R1", "T1:R2"} if allowed is None else allowed,
                              primary)


def test_partial_chunk_preserves_unknown_history_zero_and_scalar_extensions():
    data = chunk(basic={"总体评分": 0, "桥梁名称": "桥甲", "额外字段": "上行幅"},
                 advice=[{"建议内容": "复查", "自定义": "原文"}],
                 defects=[{"病害描述": "裂缝", "是否新增": "无法判断", "上一次定检状态": None}])
    before = deepcopy(data)
    result = parse(data, primary={"P1"})
    assert result == before and data == before
    assert result["病害明细"][0]["是否新增"] == "无法判断"
    assert "发展程度" not in result["病害明细"][0]


def test_empty_chunk_is_valid_without_claiming_document_coverage():
    assert parse(chunk(), allowed=set(), primary=set()) == chunk()


def test_json_fence_is_accepted_but_not_explanatory_text():
    data = chunk(basic={"桥梁名称": "甲桥"})
    raw = json.dumps(data)
    assert parse_chunk_result("```json\n" + raw + "\n```", {"P1"}) == data
    for text in ("以下是结果：" + raw, raw + "完成", "```json\n" + raw, raw + "{}"):
        with pytest.raises(ChunkValidationError):
            parse_chunk_result(text, {"P1"})


@pytest.mark.parametrize("raw", [
    '{"基本信息":{},"维修建议":[],"病害明细":[',
    '{"基本信息":{},"基本信息":{},"维修建议":[],"病害明细":[],"evidence":{}}',
    '{"基本信息":{"总体评分": NaN},"维修建议":[],"病害明细":[],"evidence":{}}',
    '{"基本信息":{"总体评分": Infinity},"维修建议":[],"病害明细":[],"evidence":{}}',
    '{"基本信息":{"总体评分": 1e999},"维修建议":[],"病害明细":[],"evidence":{}}',
    '{"基本信息":{},"维修建议":[],"病害明细":[],"evidence":{},}',
    '[]', '',
])
def test_truncated_duplicate_nonfinite_and_lenient_json_are_rejected(raw):
    with pytest.raises(ChunkValidationError):
        parse_chunk_result(raw, {"P1"})


@pytest.mark.parametrize("section", ["基本信息", "维修建议", "病害明细", "evidence"])
def test_all_sections_required_even_for_empty_chunks(section):
    data = chunk()
    del data[section]
    with pytest.raises(ChunkValidationError):
        parse(data)


def test_every_fact_needs_its_own_authorized_source():
    data = chunk(basic={"桥梁名称": "甲桥", "报告日期": "2018-12-20"},
                 defects=[{"病害描述": "裂缝"}])
    for mutation in (
        lambda obj: obj["evidence"]["基本信息"].pop("报告日期"),
        lambda obj: obj["evidence"]["基本信息"].update({"桥梁名称": ["UNREAD"]}),
        lambda obj: obj["evidence"].update({"病害明细": [[]]}),
        lambda obj: obj["evidence"].update({"病害明细": []}),
        lambda obj: obj["evidence"].update({"病害明细": ["P1"]}),
        lambda obj: obj["evidence"]["基本信息"].update({"不存在的字段": ["P1"]}),
    ):
        invalid = deepcopy(data)
        mutation(invalid)
        with pytest.raises(ChunkValidationError):
            parse(invalid)


@pytest.mark.parametrize("section, path", [
    ("基本信息", 'evidence.基本信息["桥梁名称"]'),
    ("维修建议", "evidence.维修建议[0]"),
    ("病害明细", "evidence.病害明细[0]"),
])
def test_malformed_citation_error_identifies_exact_field_or_row_without_repair(section, path):
    data = chunk(basic={"桥梁名称": "甲桥"}, advice=[{"建议内容": "复查"}],
                 defects=[{"病害描述": "裂缝"}])
    if section == "基本信息":
        data["evidence"][section]["桥梁名称"] = "P1"
    else:
        data["evidence"][section] = ["P1"]
    before = deepcopy(data)
    with pytest.raises(ChunkValidationError) as error:
        parse(data)
    assert str(error.value) == "evidence_ids_must_be_string_list:" + path
    assert data == before


def test_neighbor_context_cannot_contribute_a_new_fact_without_primary_source():
    data = chunk(basic={"桥梁名称": "甲桥"}, defects=[{"病害描述": "裂缝"}], source="P2")
    with pytest.raises(ChunkValidationError, match="context_only"):
        parse(data, primary={"P1"})
    data["evidence"]["基本信息"]["桥梁名称"] = ["P2", "P1", "P1"]
    data["evidence"]["病害明细"] = [["P1", "P2"]]
    result = parse(data, primary={"P1"})
    assert result["evidence"]["基本信息"]["桥梁名称"] == ["P1", "P2"]
    with pytest.raises(ChunkValidationError, match="subset"):
        parse(data, primary={"UNREAD"})


@pytest.mark.parametrize("section, row", [
    ("维修建议", {"序号": "1"}),
    ("病害明细", {"序号": "1", "是否新增": "是"}),
    ("病害明细", {"病害描述": {"nested": "裂缝"}}),
    ("维修建议", {"建议内容": ["复查"]}),
])
def test_rows_need_scalar_content_not_only_serial_or_history(section, row):
    data = chunk()
    data[section] = [row]
    data["evidence"][section] = [["P1"]]
    with pytest.raises(ChunkValidationError):
        parse(data)


def test_overlap_dedup_preserves_distinct_source_rows_and_within_source_multiplicity():
    reducer = EvidenceReducer()
    row = {"病害部位": "P1", "病害类型": "裂缝", "病害描述": "宽 0.1 mm"}
    data = chunk(defects=[row, row], advice=[{"建议内容": "复查"}] * 2, source="T1:R1")
    reducer.add("first", parse(data))
    reducer.add("same-overlap", parse(data))
    reducer.add("shorter-overlap", parse(chunk(defects=[row], source="T1:R1")))
    reducer.add("distinct-source", parse(chunk(defects=[row], source="T1:R2")))
    result = reducer.report()
    assert len(result["病害明细"]) == 3
    assert len(result["维修建议"]) == 2
    provenance = reducer.provenance()["病害明细"]
    assert provenance[0] == {"evidence_ids": ["T1:R1"],
                             "task_ids": ["first", "same-overlap", "shorter-overlap"]}
    assert provenance[1]["task_ids"] == ["first", "same-overlap"]
    assert provenance[2]["evidence_ids"] == ["T1:R2"]


def test_larger_duplicate_multiplicity_is_retained_on_later_chunk():
    reducer = EvidenceReducer()
    row = {"病害描述": "两处相同裂缝"}
    reducer.add("first", parse(chunk(defects=[row])))
    reducer.add("second", parse(chunk(defects=[row, row, row])))
    assert len(reducer.report()["病害明细"]) == 3
    assert reducer.provenance()["病害明细"][2]["task_ids"] == ["second"]


def test_disjoint_tasks_with_shared_source_id_keep_equal_rows_when_dedup_disabled():
    reducer = EvidenceReducer(deduplicate_overlapping_sources=False)
    data = parse(chunk(defects=[{"病害描述": "相同裂缝"}], advice=[{"建议内容": "复查"}]))
    reducer.add("P1:span1", data)
    reducer.add("P1:span2", data)
    reducer.add("P1:span1", data)
    assert len(reducer.report()["病害明细"]) == 2
    assert len(reducer.report()["维修建议"]) == 2
    provenance = reducer.provenance()
    assert provenance["病害明细"] == [
        {"evidence_ids": ["P1"], "task_ids": ["P1:span1"]},
        {"evidence_ids": ["P1"], "task_ids": ["P1:span2"]},
    ]
    assert provenance["reduction_policy"] == {"deduplicate_overlapping_sources": False}
    assert EvidenceReducer().provenance()["reduction_policy"] == {"deduplicate_overlapping_sources": True}


def test_deduplication_mode_requires_explicit_boolean():
    with pytest.raises(ValueError, match="boolean"):
        EvidenceReducer(deduplicate_overlapping_sources="false")


def test_similar_diseases_and_differing_source_sets_are_never_heuristically_merged():
    reducer = EvidenceReducer()
    row = {"病害部位": "P1", "病害类型": "裂缝", "病害描述": "宽 0.1 mm"}
    reducer.add("first", parse(chunk(defects=[row])))
    reducer.add("different-description", parse(chunk(defects=[dict(row, 病害描述="宽 0.2 mm")])))
    data = chunk(defects=[row])
    data["evidence"]["病害明细"] = [["P1", "P2"]]
    reducer.add("different-source-set", parse(data))
    assert len(reducer.report()["病害明细"]) == 3


def test_conflicting_basic_values_cannot_silently_overwrite():
    reducer = EvidenceReducer()
    reducer.add("cover", parse(chunk(basic={"报告日期": "2018-12-20"})))
    reducer.add("inspection", parse(chunk(basic={"报告日期": "2018-12-16"}, source="P2")))
    conflicts = reducer.conflicts()
    assert len(conflicts) == 1 and conflicts[0]["field"] == "报告日期"
    assert conflicts[0]["candidates"] == [
        {"value": "2018-12-20", "evidence_ids": ["P1"], "task_id": "cover"},
        {"value": "2018-12-16", "evidence_ids": ["P2"], "task_id": "inspection"},
    ]
    with pytest.raises(UnresolvedConflictsError) as error:
        reducer.report()
    assert error.value.conflicts == conflicts
    conflict_id = conflicts[0]["id"]
    with pytest.raises(ValueError, match="candidate"):
        reducer.resolve(conflict_id, "2018-12-19", ["P1"], {"P1", "P2"})
    with pytest.raises(ValueError, match="unauthorized"):
        reducer.resolve(conflict_id, "2018-12-20", ["UNREAD"], {"P1", "P2"})
    reducer.resolve(conflict_id, "2018-12-20", ["P1"], {"P1", "P2"})
    assert reducer.conflicts() == []
    assert reducer.report()["基本信息"]["报告日期"] == "2018-12-20"
    assert reducer.provenance()["基本信息"]["报告日期"]["resolution"]["evidence_ids"] == ["P1"]


@pytest.mark.parametrize("unknown", ["未知", "无法判断", None])
def test_review_may_mark_conflict_unknown_without_inventing_history(unknown):
    reducer = EvidenceReducer()
    reducer.add("a", parse(chunk(basic={"上一次定检结果": "有裂缝"})))
    reducer.add("b", parse(chunk(basic={"上一次定检结果": "未见裂缝"}, source="P2")))
    conflict_id = reducer.conflicts()[0]["id"]
    reducer.resolve(conflict_id, unknown, ["P1", "P2"], {"P1", "P2"})
    assert reducer.report()["基本信息"] == {"上一次定检结果": unknown}


def test_new_value_reopens_resolved_conflict_but_corroboration_does_not():
    reducer = EvidenceReducer()
    for task, score in (("a", 90), ("b", 80)):
        reducer.add(task, parse(chunk(basic={"总体评分": score})))
    conflict_id = reducer.conflicts()[0]["id"]
    reducer.resolve(conflict_id, 90, ["P1"], {"P1"})
    reducer.add("corroboration", parse(chunk(basic={"总体评分": 90})))
    assert reducer.conflicts() == []
    reducer.add("new-score", parse(chunk(basic={"总体评分": 70})))
    assert reducer.conflicts()[0]["id"] == conflict_id
    with pytest.raises(UnresolvedConflictsError):
        reducer.report()


def test_task_retry_is_idempotent_but_replacing_accepted_output_is_rejected():
    reducer = EvidenceReducer()
    data = parse(chunk(basic={"桥梁名称": "甲桥"}, defects=[{"病害描述": "裂缝"}]))
    reducer.add("task", data)
    reducer.add("task", deepcopy(data))
    assert len(reducer.report()["病害明细"]) == 1
    replacement = deepcopy(data)
    replacement["基本信息"]["桥梁名称"] = "乙桥"
    with pytest.raises(ValueError, match="different contribution"):
        reducer.add("task", replacement)
    assert reducer.report()["基本信息"]["桥梁名称"] == "甲桥"


def test_task_retry_compares_json_types_without_python_bool_integer_equivalence():
    reducer = EvidenceReducer()
    reducer.add("task", parse(chunk(basic={"扩展字段": 1})))
    with pytest.raises(ValueError, match="different contribution"):
        reducer.add("task", parse(chunk(basic={"扩展字段": True})))


def test_reducer_owns_its_state_and_provenance_is_independent_json():
    reducer = EvidenceReducer()
    data = parse(chunk(basic={"桥梁名称": "甲桥", "总体评分": "", "上一次总体评分": None},
                       defects=[{"病害描述": "裂缝"}]))
    reducer.add("task", data)
    data["基本信息"]["桥梁名称"] = "变更"
    result = reducer.report()
    result["病害明细"].clear()
    provenance = reducer.provenance()
    provenance["基本信息"]["桥梁名称"]["candidates"][0]["value"] = "变更"
    assert reducer.report()["基本信息"] == {"桥梁名称": "甲桥", "总体评分": "", "上一次总体评分": None}
    assert len(reducer.report()["病害明细"]) == 1
    assert reducer.provenance()["基本信息"]["桥梁名称"]["value"] == "甲桥"
    json.dumps(reducer.provenance(), allow_nan=False)


def test_reducer_rejects_unvalidated_bad_shape_without_partial_mutation():
    reducer = EvidenceReducer()
    data = chunk(basic={"桥梁名称": "甲桥"}, defects=[{"病害描述": "裂缝"}])
    data["evidence"]["病害明细"] = [[]]
    with pytest.raises(ChunkValidationError):
        reducer.add("bad", data)
    assert reducer.report() == {"基本信息": {}, "维修建议": [], "病害明细": []}
