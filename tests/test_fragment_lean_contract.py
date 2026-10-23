"""Lean rows (quotes-only values, derived source_id) parse to the same evidence-bound result."""

import json
import os
from pathlib import Path

import pytest

from bridge_evidence.contest import chunked_inference as ci
from bridge_evidence.contest.candidate import validate_config
from bridge_evidence.contest.long_document import (
    LEAN_FRAGMENT_INSTRUCTION,
    merge_fragments,
    parse_fragment,
)
from test_fragment_exact_contract import encoded, recovery_task, valid_output


def lean_output():
    return {
        "covered": ["p0:0:20", "p1:0:19"],
        "基本信息": {"桥梁名称": "RSI恢复验证桥", "报告日期": "2018"},
        "basic_sources": {"桥梁名称": "p0", "报告日期": "p0"},
        "basic_quotes": {"桥梁名称": "RSI恢复验证桥", "报告日期": "2018年"},
        "维修建议": [{"source_key": "p1:0:19",
                      "quotes": {"建议内容": "建议封闭处理", "病害部位": "第一跨主梁"}}],
        "病害明细": [{"source_key": "p1:0:19",
                      "quotes": {"病害部位": "第一跨主梁", "病害类型": "裂缝", "病害描述": "出现一条裂缝"}}],
    }


def merged(raw):
    snapshot, task = recovery_task()
    plan = ci.prepare_chunk_plan(snapshot, _tokenizer(), validate_config({}))
    fragment = parse_fragment(raw, task, snapshot, "call")
    return merge_fragments(snapshot, plan, [fragment])


def _tokenizer():
    from test_fragment_prompt_contract import CharTokenizer

    return CharTokenizer()


def test_lean_rows_merge_to_the_same_tables_as_verbose_rows():
    lean_data = merged(encoded(lean_output()))["data"]
    verbose_data = merged(encoded(valid_output()))["data"]
    assert len(lean_data["维修建议"]) == len(verbose_data["维修建议"]) == 1
    assert len(lean_data["病害明细"]) == len(verbose_data["病害明细"]) == 1
    for section in ("维修建议", "病害明细"):
        for lean_row, verbose_row in zip(lean_data[section], verbose_data[section]):
            assert {k: v for k, v in lean_row.items() if v is not None} == {
                k: v for k, v in verbose_row.items() if v is not None}
    assert lean_data["基本信息"]["桥梁名称"] == "RSI恢复验证桥"


def test_lean_row_source_id_is_derived_from_owned_key():
    snapshot, task = recovery_task()
    fragment = parse_fragment(encoded(lean_output()), task, snapshot, "call")
    assert fragment.diseases[0]["source_id"] == "p1"
    assert fragment.diseases[0]["fields"]["病害描述"] == "出现一条裂缝"
    assert fragment.diseases[0]["temporal"] == "证据不足"


@pytest.mark.parametrize("issue", ["unknown-field", "forged-quote", "unowned-key", "advice-alias"])
def test_lean_row_defects_degrade_only_the_offending_part(issue):
    snapshot, task = recovery_task()
    value = lean_output()
    row = value["病害明细"][0]
    if issue == "unknown-field":
        row["quotes"]["不存在的字段"] = "裂缝"
    elif issue == "forged-quote":
        row["quotes"]["病害描述"] = "出现两条裂缝"
    elif issue == "unowned-key":
        row["source_key"] = "p9:0:3"
    else:
        value["维修建议"][0]["quotes"] = {"建议": "建议封闭处理"}
    fragment = parse_fragment(encoded(value), task, snapshot, "call")
    disease, advice = fragment.diseases[0], fragment.advice[0]
    if issue == "unknown-field":
        assert "不存在的字段" not in disease["fields"] and "不存在的字段" not in disease["quotes"]
    elif issue == "forged-quote":
        # Kept for recall, but never presented as verbatim evidence.
        assert disease["fields"]["病害描述"] == "出现两条裂缝"
        assert disease["unverified_fields"] == ["病害描述"]
    elif issue == "unowned-key":
        # Rebound to the owned slice that actually contains its verbatim quotes.
        assert disease["source_key"] == "p1:0:19"
    else:
        assert advice["fields"]["建议内容"] == "建议封闭处理"
        assert advice["quotes"] == {"建议内容": "建议封闭处理"}


def test_lean_prompt_is_fragment_only_and_resolved():
    config = validate_config({"prompt": "source-fragment-lean-v1"})
    assert ci._instruction(config) == LEAN_FRAGMENT_INSTRUCTION
    with pytest.raises(ValueError):
        validate_config({"prompt": "source-fragment-lean-v1", "workflow": "whole"})


def test_lean_rows_cut_output_tokens_on_real_tokenizer():
    directory = os.environ.get("CITYAGENT_TEST_FIXED9B_TOKENIZER_DIR")
    if directory is None:
        pytest.skip("Set CITYAGENT_TEST_FIXED9B_TOKENIZER_DIR to pinned offline fixed9B assets")
    from bridge_evidence.contest.fast_tokenizer import FastTokenizer

    tokenizer = FastTokenizer(Path(directory))
    description = "梁底出现1条横向裂缝，L=0.5m，W=0.12mm"
    verbose_row = {"source_id": "p12", "source_key": "p12:0:300", "instance_id": "d3",
                   "fields": {"序号": 3, "病害部位": "第2跨左侧主梁", "病害类型": "裂缝", "病害描述": description,
                              "是否新增": None, "上一次定检状态": None, "发展程度": None},
                   "quotes": {"病害部位": "第2跨左侧主梁", "病害类型": "裂缝", "病害描述": description},
                   "temporal": "证据不足"}
    lean_row = {"source_key": "p12:0:300",
                "quotes": {"病害部位": "第2跨左侧主梁", "病害类型": "裂缝", "病害描述": description}}
    verbose = len(tokenizer.encode(json.dumps([verbose_row] * 30, ensure_ascii=False), add_special_tokens=False))
    lean = len(tokenizer.encode(json.dumps([lean_row] * 30, ensure_ascii=False), add_special_tokens=False))
    assert lean < 0.6 * verbose, (lean, verbose)


def test_unrepaired_basic_conflicts_resolve_by_majority_then_document_order():
    from bridge_evidence.contest.long_document import resolve_basic_conflicts

    def observation(value):
        return {"field": "总体评分", "value": value, "scope": {}, "evidence": {"source_id": "p0"}}

    for values, expected in ((["82", "90", "90"], "90"), (["82", "90"], "82")):
        merged = {"data": {"基本信息": {"总体评分": None}}, "basic_ambiguities": [], "conflicts": [{
            "field": "总体评分", "values": list(dict.fromkeys(values)),
            "observations": [observation(v) for v in values], "source_id": "p0"}]}
        resolve_basic_conflicts(merged)
        assert merged["data"]["基本信息"]["总体评分"] == expected
        assert merged["conflicts"] == []
        assert merged["basic_ambiguities"][0]["reason"] == "resolved_by_majority_then_document_order"


def test_merge_skips_rows_without_publishable_content():
    snapshot, task = recovery_task()
    value = lean_output()
    value["维修建议"] = [{"source_key": "p1:0:19", "fields": {"建议类别": "尽快维修"}}]
    data = merged(encoded(value))["data"]
    assert data["维修建议"] == []
    assert [row["病害描述"] for row in data["病害明细"]] == ["出现一条裂缝"]


def test_chunked_report_may_publish_rows_without_basic_information():
    from bridge_evidence.contest.inference_core import _InvalidOutput, _parse_report

    rows_only = {"基本信息": {"桥梁名称": None}, "维修建议": [], "病害明细": [{"病害描述": "裂缝"}]}
    assert _parse_report(json.dumps(rows_only, ensure_ascii=False), require_basic=False)
    with pytest.raises(_InvalidOutput, match="meaningless_basic_information"):
        _parse_report(json.dumps(rows_only, ensure_ascii=False))
    empty = {"基本信息": {}, "维修建议": [], "病害明细": []}
    with pytest.raises(_InvalidOutput, match="meaningless_report"):
        _parse_report(json.dumps(empty), require_basic=False)


@pytest.mark.parametrize("text, value, expected", [
    ("技术状况等级为B级", "B", "B"), ("评定等级：C", "C级", "C级"), ("采用C30混凝土", "C", None), ("BCI=87", "B", None),
])
def test_one_letter_grades_need_a_grade_context(text, value, expected):
    from bridge_evidence.contest.long_document import DocumentSnapshot as Snapshot

    snapshot = Snapshot.from_text(text)
    plan = ci.prepare_chunk_plan(snapshot, _tokenizer(), validate_config({}))
    task = plan.tasks[0]
    fragment = parse_fragment(encoded({"基本信息": {"总体等级": value}}), task, snapshot, "call")
    assert fragment.basic["总体等级"] == expected
