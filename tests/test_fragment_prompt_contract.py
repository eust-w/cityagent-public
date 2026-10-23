"""Replay the real v5 synthetic-report response; never treat it as model acceptance."""

import hashlib
import json
from copy import deepcopy

import pytest

from bridge_evidence.contest.inference_core import build_prompt
from bridge_evidence.contest.long_document import (
    FRAGMENT_INSTRUCTION,
    DocumentSnapshot,
    SourceRegion,
    parse_fragment,
    plan_chunks,
    split_task,
)

# Exact raw response from the official-environment v5 synthetic short-report probe.
ACTUAL_V5_RAW = '{"covered": ["p0:0:40", "p1:0:31"], "基本信息": {"桥梁名称": "运行检查桥", "报告日期": "2026年10月10日", "总体评分": "82.5", "总体等级": "B级", "上部结构评分": "无", "上部结构等级": "无", "下部结构评分": "无", "下部结构等级": "无", "桥面系评分": "无", "桥面系等级": "无", "上一次总体评分": "无", "上一次总体等级": "无", "上一次定检结果": "无", "病害发展趋势与具体说明": "无", "总体结论": "总体评分82.5，技术状况等级B级", "主要风险点": "桥面伸缩缝有杂物", "建议统计": "0条立即处置、1条尽快维修、0条预防性养护"}, "basic_sources": {"桥梁名称": "p0:0:40", "报告日期": "p0:0:40", "总体评分": "p1:0:31", "总体等级": "p1:0:31", "总体结论": "p1:0:31", "主要风险点": "p1:0:31", "建议统计": "p1:0:31"}, "basic_quotes": {"桥梁名称": "桥梁名称：运行检查桥", "报告日期": "报告日期：2026年10月10日", "总体评分": "总体评分82.5", "总体等级": "技术状况等级B", "总体结论": "总体评分82.5，技术状况等级B", "主要风险点": "桥面伸缩缝有杂物", "建议统计": "建议清理"}, "basic_scopes": {}, "维修建议": [{"序号": "1", "建议类别": "尽快维修", "建议内容": "桥面伸缩缝有杂物，建议清理", "病害部位": "桥面伸缩缝"}], "病害明细": []}'


class CharTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "CHAT\n" + "\n".join(message["content"] for message in messages) + "\nASSISTANT:"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text))}


def source_and_task():
    snapshot = DocumentSnapshot("a" * 64, [
        SourceRegion("p0", 0, "运行检查桥定期检测报告。桥梁名称：运行检查桥。报告日期：2026年10月10日。"),
        SourceRegion("p1", 1, "总体评分82.5，技术状况等级B；桥面伸缩缝有杂物，建议清理。"),
    ])
    plan = plan_chunks(snapshot, CharTokenizer())
    assert len(plan.tasks) == 1
    return snapshot, plan.tasks[0]


def lawful_output():
    return {
        "covered": ["p0:0:40", "p1:0:31"],
        "基本信息": {"桥梁名称": "运行检查桥", "报告日期": "2026年10月10日",
                     "总体评分": "82.5", "总体等级": "B", "上一次总体评分": None,
                     "建议统计": None},
        "basic_sources": {"桥梁名称": "p0:0:40", "报告日期": "p0:0:40",
                          "总体评分": "p1:0:31", "总体等级": "p1:0:31"},
        "basic_quotes": {"桥梁名称": "桥梁名称：运行检查桥", "报告日期": "报告日期：2026年10月10日",
                         "总体评分": "总体评分82.5", "总体等级": "技术状况等级B"},
        "basic_scopes": {},
        "维修建议": [{"source_id": "p1", "source_key": "p1:0:31", "instance_id": "advice-1",
                      "fields": {"建议内容": "建议清理", "建议类别": None, "病害部位": "桥面伸缩缝"},
                      "quotes": {"建议内容": "建议清理", "病害部位": "桥面伸缩缝"}}],
        "病害明细": [{"source_id": "p1", "source_key": "p1:0:31", "instance_id": "defect-1",
                     "fields": {"病害部位": "桥面伸缩缝", "病害描述": "桥面伸缩缝有杂物"},
                     "quotes": {"病害部位": "桥面伸缩缝", "病害描述": "桥面伸缩缝有杂物"},
                     "temporal": "证据不足"}],
    }


def test_actual_v5_raw_keeps_grounded_fields_and_drops_unsupported_ones():
    assert hashlib.sha256(ACTUAL_V5_RAW.encode()).hexdigest() == "5af04ff9ca34399126a44947caa1ac305a29e355f5e6494e0f6ae6985925d6c6"
    snapshot, task = source_and_task()
    fragment = parse_fragment(ACTUAL_V5_RAW, task, snapshot, "actual-v5-regression")
    assert {k: v for k, v in fragment.basic.items() if v is not None} == {
        "桥梁名称": "运行检查桥", "报告日期": "2026年10月10日", "总体评分": "82.5",
        "总体等级": "B级", "主要风险点": "桥面伸缩缝有杂物"}
    # The grade is grounded by "等级B"; derived 建议统计 and a paraphrased 总体结论 are not.
    assert fragment.basic_evidence["总体等级"]["quote"] == "等级B"
    assert fragment.basic["建议统计"] is None
    assert fragment.basic["总体结论"] is None
    # A flat advice row is bound through its verbatim content, which becomes its evidence.
    assert fragment.advice[0]["source_key"] == "p1:0:31"
    assert fragment.advice[0]["quotes"] == {"建议内容": "桥面伸缩缝有杂物，建议清理", "病害部位": "桥面伸缩缝"}
    assert fragment.advice[0]["unverified_fields"] == ["建议类别"]


@pytest.mark.parametrize("issue", ["missing", "derived", "flat"])
def test_actual_v5_contract_mistakes_degrade_without_inventing(issue):
    snapshot, task = source_and_task()
    value, actual = lawful_output(), json.loads(ACTUAL_V5_RAW)
    if issue == "missing":
        value["基本信息"]["上部结构评分"] = actual["基本信息"]["上部结构评分"]
    elif issue == "derived":
        value["基本信息"]["建议统计"] = actual["基本信息"]["建议统计"]
        value["basic_sources"]["建议统计"] = actual["basic_sources"]["建议统计"]
        value["basic_quotes"]["建议统计"] = actual["basic_quotes"]["建议统计"]
    else:
        value["维修建议"] = actual["维修建议"]
    fragment = parse_fragment(json.dumps(value, ensure_ascii=False), task, snapshot, "fixture")
    assert fragment.basic["桥梁名称"] == "运行检查桥"
    if issue == "missing":
        assert fragment.basic["上部结构评分"] is None
    elif issue == "derived":
        assert fragment.basic["建议统计"] is None
    else:
        assert [row["fields"]["建议内容"] for row in fragment.advice] == ["桥面伸缩缝有杂物，建议清理"]


def test_complete_nested_positive_preserves_values_and_explicit_nulls():
    snapshot, task = source_and_task()
    expected = lawful_output()
    fragment = parse_fragment(json.dumps(deepcopy(expected), ensure_ascii=False), task, snapshot, "fixture")
    assert fragment.basic == expected["基本信息"]
    assert fragment.basic["总体等级"] == "B"
    assert fragment.advice[0]["fields"] == expected["维修建议"][0]["fields"]
    assert fragment.diseases[0]["fields"]["病害描述"] == "桥面伸缩缝有杂物"
    assert fragment.covered == task.owned_keys


def test_prompt_contains_parseable_complete_nested_example():
    marker = "合法输出示例：\n"
    assert marker in FRAGMENT_INSTRUCTION
    example = json.loads(FRAGMENT_INSTRUCTION.split(marker, 1)[1])
    snapshot = DocumentSnapshot.from_text("桥梁名称：示例桥；等级B；梁有裂缝，建议复查。")
    task = plan_chunks(snapshot, CharTokenizer()).tasks[0]
    fragment = parse_fragment(json.dumps(example, ensure_ascii=False), task, snapshot, "example")
    assert fragment.advice and fragment.diseases
    assert fragment.basic["总体等级"] == "B"
    assert fragment.basic["建议统计"] is None


def test_expanded_template_and_oversize_row_fit_full_8k_input_without_lost_tail():
    tokenizer = CharTokenizer()
    text = "裂缝最大宽度0.3mm；" * 2000 + "本行末尾必须保留"
    snapshot = DocumentSnapshot("b" * 64, [SourceRegion("t0r0", 0, text, kind="table-row")])
    plan = plan_chunks(snapshot, tokenizer, input_budget=8192, output_budget=4096)
    assert len(plan.tasks) > 1
    assert plan.coverage(snapshot)["complete"]
    assert "".join(part.text for task in plan.tasks for part in task.owned) == text
    for task in plan.tasks:
        ids = build_prompt(tokenizer, task.text(), max_model_len=12288, max_new_tokens=4096,
                           instruction=FRAGMENT_INSTRUCTION)
        assert len(ids) == task.prompt_tokens <= 8192
    original = plan.tasks[-1]
    children = split_task(original, snapshot)
    assert "".join(part.text for child in children for part in child.owned) == "".join(part.text for part in original.owned)
