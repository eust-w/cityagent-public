"""Offline replay of fixed9B recovery syntax/quote failures; no model acceptance claim."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from bridge_evidence.contest import chunked_inference as ci
from bridge_evidence.contest.candidate import validate_config
from bridge_evidence.contest.inference_core import BASIC_FIELDS, _parse_report, build_prompt, whole_instruction
from bridge_evidence.contest.long_document import (
    FRAGMENT_INSTRUCTION,
    LEGACY_FRAGMENT_INSTRUCTION,
    ChunkTask,
    DocumentSnapshot,
    SourceRegion,
    SourceSlice,
    merge_fragments,
    parse_fragment,
)
from test_fragment_prompt_contract import CharTokenizer

# Immutable responses from the fixed synthetic process-recovery-off-r2 probe.
# The source and both raw responses are embedded so regression tests stay offline.
RECOVERY_SOURCE = 'RSI恢复验证桥，2018年定期检测。\n第一跨主梁出现一条裂缝，建议封闭处理。'
ACTUAL_SPACED_RAW = '{"covered":["p0:0:20","p1:0:19"],"基本信息":{"桥梁名称":"RSI 恢复验证桥","报告日期":"2018","总体评分":null,"总体等级":null,"上部结构评分":null,"上部结构等级":null,"下部结构评分":null,"下部结构等级":null,"桥面系评分":null,"桥面系等级":null,"上一次总体评分":null,"上一次总体等级":null,"上一次定检结果":null,"病害发展趋势与具体说明":null,"总体结论":null,"主要风险点":null,"建议统计":null},"basic_sources":{"桥梁名称":"p0","报告日期":"p0"},"basic_quotes":{"桥梁名称":"RSI 恢复验证桥","报告日期":"2018 年"},"basic_scopes":{},"维修建议":[{"source_id":"p1","source_key":"p1:0:19","instance_id":"m1","fields":{"序号":1,"建议类别":null,"建议内容":"建议封闭处理","病害部位":"第一跨主梁"},"quotes":{"序号":null,"建议类别":null,"建议内容":"建议封闭处理","病害部位":"第一跨主梁"}}],"病害明细":[{"source_id":"p1","source_key":"p1:0:19","instance_id":"d1","fields":{"序号":1,"病害部位":"第一跨主梁","病害类型":"裂缝","病害描述":"出现一条裂缝","是否新增":null,"上一次定检状态":null,"发展程度":null},"quotes":{"序号":null,"病害部位":"第一跨主梁","病害类型":"裂缝","病害描述":"出现一条裂缝"},"temporal":"证据不足"}]}'
ACTUAL_FENCED_RAW = '```json\n{\n"covered": [\n  "p0:0:20",\n  "p1:0:19"\n],\n"基本信息": {\n  "桥梁名称": "RSI 恢复验证桥",\n  "报告日期": null,\n  "总体评分": null,\n  "总体等级": null,\n  "上部结构评分": null,\n  "上部结构等级": null,\n  "下部结构评分": null,\n  "下部结构等级": null,\n  "桥面系评分": null,\n  "桥面系等级": null,\n  "上一次总体评分": null,\n  "上一次总体等级": null,\n  "上一次定检结果": null,\n  "病害发展趋势与具体说明": null,\n  "总体结论": null,\n  "主要风险点": null,\n  "建议统计": null\n},\n"basic_sources": {\n  "桥梁名称": "p0",\n  "报告日期": null,\n  "总体评分": null,\n  "总体等级": null,\n  "上部结构评分": null,\n  "上部结构等级": null,\n  "下部结构评分": null,\n  "下部结构等级": null,\n  "桥面系评分": null,\n  "桥面系等级": null,\n  "上一次总体评分": null,\n  "上一次总体等级": null,\n  "上一次定检结果": null,\n  "病害发展趋势与具体说明": null,\n  "总体结论": null,\n  "主要风险点": null,\n  "建议统计": null\n},\n"basic_quotes": {\n  "桥梁名称": "RSI 恢复验证桥",\n  "报告日期": null,\n  "总体评分": null,\n  "总体等级": null,\n  "上部结构评分": null,\n  "上部结构等级": null,\n  "下部结构评分": null,\n  "下部结构等级": null,\n  "桥面系评分": null,\n  "桥面系等级": null,\n  "上一次总体评分": null,\n  "上一次总体等级": null,\n  "上一次定检结果": null,\n  "病害发展趋势与具体说明": null,\n  "总体结论": null,\n  "主要风险点": null,\n  "建议统计": null\n},\n"basic_scopes": {},\n"维修建议": [\n  {\n    "source_id": "p1",\n    "source_key": "p1:0:19",\n    "instance_id": "m1",\n    "fields": {\n      "序号": 1,\n      "建议类别": null,\n      "建议内容": "建议封闭处理",\n      "病害部位": "第一跨主梁"\n    },\n    "quotes": {\n      "建议内容": "建议封闭处理",\n      "病害部位": "第一跨主梁"\n    }\n  }\n],\n"病害明细": [\n  {\n    "source_id": "p1",\n    "source_key": "p1:0:19",\n    "instance_id": "d1",\n    "fields": {\n      "序号": 1,\n      "病害部位": "第一跨主梁",\n      "病害类型": "裂缝",\n      "病害描述": "出现一条裂缝",\n      "是否新增": null,\n      "上一次定检状态": null,\n      "发展程度": null\n    },\n    "quotes": {\n      "病害部位": "第一跨主梁",\n      "病害类型": "裂缝",\n      "病害描述": "出现一条裂缝"\n    },\n    "temporal": "证据不足"\n  }\n]\n}\n```'


def recovery_task():
    snapshot = DocumentSnapshot.from_text(RECOVERY_SOURCE)
    config = validate_config({})
    plan = ci.prepare_chunk_plan(snapshot, CharTokenizer(), config)
    assert len(plan.tasks) == 1
    return snapshot, plan.tasks[0]


def valid_output():
    """Synthetic lawful counterpart, independently constructed from the source."""
    return {
        "covered": ["p0:0:20", "p1:0:19"],
        "基本信息": {"桥梁名称": "RSI恢复验证桥", "报告日期": "2018", "总体评分": None},
        "basic_sources": {"桥梁名称": "p0", "报告日期": "p0"},
        "basic_quotes": {"桥梁名称": "RSI恢复验证桥", "报告日期": "2018年"},
        "basic_scopes": {},
        "维修建议": [{
            "source_id": "p1", "source_key": "p1:0:19", "instance_id": "m1",
            "fields": {"序号": 1, "建议类别": None, "建议内容": "建议封闭处理", "病害部位": "第一跨主梁"},
            "quotes": {"建议内容": "建议封闭处理", "病害部位": "第一跨主梁"},
        }],
        "病害明细": [{
            "source_id": "p1", "source_key": "p1:0:19", "instance_id": "d1",
            "fields": {"序号": 1, "病害部位": "第一跨主梁", "病害类型": "裂缝", "病害描述": "出现一条裂缝"},
            "quotes": {"病害部位": "第一跨主梁", "病害类型": "裂缝", "病害描述": "出现一条裂缝"},
            "temporal": "证据不足",
        }],
    }


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@pytest.mark.parametrize("name, expected_hash", [
    ("ACTUAL_SPACED_RAW", "7141969e0a7f2347ed85c7bf9f717087d48d0c090d5fe8d8c60dfbf272cdf577"),
    ("ACTUAL_FENCED_RAW", "e898775bb868c46d1438c4e9d7e1364bf3b643076dc58c078874bf92c766b150"),
])
def test_actual_recovery_responses_accept_boundary_spaced_quotes(name, expected_hash):
    """9B model adds spaces at CJK/ASCII boundaries; fuzzy matching resolves to exact source text."""
    raw = globals()[name]
    assert hashlib.sha256(raw.encode()).hexdigest() == expected_hash
    snapshot, task = recovery_task()
    fragment = parse_fragment(raw, task, snapshot, "actual-boundary-space-recovery")
    # spaced "RSI 恢复验证桥" must be resolved to verbatim source "RSI恢复验证桥"
    assert fragment.basic_evidence["桥梁名称"]["quote"] == "RSI恢复验证桥"
    assert fragment.basic_evidence["桥梁名称"]["quote_start"] == 0
    assert fragment.basic_evidence["桥梁名称"]["quote_end"] == len("RSI恢复验证桥")
    assert hashlib.sha256(raw.encode()).hexdigest() == expected_hash


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "```\n{}\n```", " \n```JSON\n{}\n``` \t"])
def test_complete_transport_wrapper_preserves_fields_coverage_and_provenance(wrapper):
    snapshot, task = recovery_task()
    fragment = parse_fragment(wrapper.format(encoded(valid_output())), task, snapshot, "synthetic-call")
    assert fragment.covered == task.owned_keys
    assert fragment.basic == valid_output()["基本信息"]
    assert fragment.basic_evidence["桥梁名称"]["quote"] == "RSI恢复验证桥"
    assert fragment.basic_evidence["桥梁名称"]["source_id"] == "p0"
    assert fragment.basic_evidence["桥梁名称"]["source_key"] == "p0:0:20"
    assert fragment.basic_evidence["桥梁名称"]["quote_start"] == 0
    assert fragment.basic_evidence["桥梁名称"]["quote_end"] == len("RSI恢复验证桥")
    assert fragment.call_id == "synthetic-call"
    assert fragment.advice[0]["fields"]["序号"] == 1
    assert fragment.advice[0]["fields"]["建议类别"] is None
    assert fragment.diseases[0]["temporal"] == "证据不足"


TRANSPORT_NOISE = {
    "duplicate-top": lambda raw: '{"covered":[],' + raw[1:],
    "duplicate-nested": lambda raw: raw.replace('"基本信息":{', '"基本信息":{"桥梁名称":"伪造",', 1),
    "trailing-prose": lambda raw: raw + " 完毕",
    "leading-prose": lambda raw: "结果：" + raw,
    "two-objects": lambda raw: raw + " {}",
    "two-fences": lambda raw: "```json\n" + raw + "\n```\n```json\n{}\n```",
    "incomplete-fence": lambda raw: "```json\n" + raw,
    "single-backtick-close": lambda raw: "```json\n" + raw + "`",
    "wrong-language": lambda raw: "```python\n" + raw + "\n```",
    "NaN": lambda raw: '{"unused":NaN,' + raw[1:],
    "Infinity": lambda raw: '{"unused":Infinity,' + raw[1:],
    "1e999": lambda raw: '{"unused":1e999,' + raw[1:],
    "array-wrapped": lambda raw: "[" + raw + "]",
    # Broken syntax: every self-contained flat object is salvaged individually.
    "trailing-comma": lambda raw: raw[:-1] + ",}",
}


@pytest.mark.parametrize("syntax", sorted(TRANSPORT_NOISE))
def test_fragment_decoder_salvages_transport_noise_around_one_object(syntax):
    snapshot, task = recovery_task()
    fragment = parse_fragment(TRANSPORT_NOISE[syntax](encoded(valid_output())), task, snapshot)
    # A duplicated key keeps its first value; an ungrounded forged name is then dropped.
    expected_name = None if syntax == "duplicate-nested" else "RSI恢复验证桥"
    assert fragment.basic["桥梁名称"] == expected_name
    assert [row["fields"]["病害描述"] for row in fragment.diseases] == ["出现一条裂缝"]


@pytest.mark.parametrize("raw", [
    lambda raw: "```json\n```", lambda raw: "null",
    lambda raw: '{"基本信息":{"桥梁名称":"RSI', lambda raw: "没有JSON",
])
def test_fragment_decoder_rejects_unparseable_json(raw):
    snapshot, task = recovery_task()
    with pytest.raises(ValueError):
        parse_fragment(raw(encoded(valid_output())), task, snapshot)


@pytest.mark.parametrize("issue", [
    "nonowned-basic", "nonowned-row", "wrong-slice", "row-space", "missing-field-quote",
    "nested-value", "boolean-value", "forged-basic", "unbindable-row",
])
def test_content_defects_degrade_single_items_and_keep_the_rest(issue):
    snapshot, task = recovery_task()
    value = valid_output()
    if issue == "nonowned-basic":
        value["basic_sources"]["桥梁名称"] = "unseen"
    elif issue == "nonowned-row":
        value["病害明细"][0]["source_id"] = "unseen"
    elif issue == "wrong-slice":
        value["病害明细"][0]["source_key"] = "p1:0:18"
    elif issue == "row-space":
        value["病害明细"][0]["quotes"]["病害描述"] = "出现一条 裂缝"
    elif issue == "missing-field-quote":
        value["维修建议"][0]["quotes"]["不存在的字段"] = "建议封闭处理"
    elif issue == "forged-basic":
        value["基本信息"]["桥梁名称"] = value["basic_quotes"]["桥梁名称"] = "伪造桥"
    elif issue == "unbindable-row":
        value["病害明细"][0].update(source_id="unseen", source_key="p9:0:1", quotes={"病害描述": "原文没有的病害"},
                                     fields={"病害描述": "原文没有的病害"})
    else:
        value["基本信息"]["报告日期"] = {} if issue == "nested-value" else True
    fragment = parse_fragment("```json\n" + encoded(value) + "\n```", task, snapshot)
    assert fragment.advice[0]["fields"]["建议内容"] == "建议封闭处理"
    if issue == "nonowned-basic":
        # The verbatim quote relocates the claim to the owned slice that contains it.
        assert fragment.basic["桥梁名称"] == "RSI恢复验证桥"
        assert fragment.basic_sources["桥梁名称"] == "p0"
    elif issue in ("nonowned-row", "wrong-slice"):
        assert fragment.diseases[0]["source_key"] == "p1:0:19"
    elif issue == "row-space":
        # The broken quote is discarded; the verbatim field value becomes the evidence.
        assert "unverified_fields" not in fragment.diseases[0]
        assert fragment.diseases[0]["quotes"]["病害描述"] == "出现一条裂缝"
    elif issue == "missing-field-quote":
        assert "不存在的字段" not in fragment.advice[0]["quotes"]
    elif issue == "forged-basic":
        assert fragment.basic["桥梁名称"] is None and "桥梁名称" not in fragment.basic_evidence
    elif issue == "unbindable-row":
        assert fragment.diseases == []
    else:
        assert fragment.basic["报告日期"] is None
        assert fragment.basic["桥梁名称"] == "RSI恢复验证桥"


@pytest.mark.parametrize("issue", ["name-space", "date-space", "null-quote", "partial-covered", "no-covered"])
def test_boundary_space_and_null_quotes_accepted_and_normalized(issue):
    """Boundary-space quotes and null row quotes are now accepted after fuzzy matching."""
    snapshot, task = recovery_task()
    value = valid_output()
    if issue == "name-space":
        value["基本信息"]["桥梁名称"] = value["basic_quotes"]["桥梁名称"] = "RSI 恢复验证桥"
    elif issue == "date-space":
        value["basic_quotes"]["报告日期"] = "2018 年"
    elif issue == "null-quote":
        value["维修建议"][0]["quotes"]["建议类别"] = None
    elif issue == "partial-covered":
        value["covered"] = value["covered"][:1] + ["p9"]
    elif issue == "no-covered":
        del value["covered"]
    fragment = parse_fragment("```json\n" + encoded(value) + "\n```", task, snapshot)
    assert fragment.covered == task.owned_keys
    if issue in ("name-space", "date-space"):
        # Spaced quote resolved to verbatim source text
        assert fragment.basic_evidence["桥梁名称"]["quote"] == "RSI恢复验证桥"
    if issue == "name-space":
        # Field value also normalized
        assert fragment.basic["桥梁名称"] == "RSI恢复验证桥"
    if issue == "null-quote":
        # null row quote silently skipped; advice still parsed
        assert fragment.advice[0]["fields"]["序号"] == 1


def test_fence_does_not_accept_tampered_task_source():
    snapshot, task = recovery_task()
    task = deepcopy(task)
    task.owned[0] = replace(task.owned[0], text=task.owned[0].text.replace("RSI", "XYZ"))
    with pytest.raises(ValueError, match="task source does not match snapshot"):
        parse_fragment("```json\n" + encoded(valid_output()) + "\n```", task, snapshot)


def test_exact_prompt_and_fence_preserve_visible_history_context_evidence():
    prompt = ci._instruction(validate_config({"prompt": "source-fragment-exact-v1"}))
    assert "history_evidence和relationship_evidence可来自owned或context" in prompt
    snapshot = DocumentSnapshot.from_text(
        "2018 测试桥左幅1跨梁裂缝\n2017 测试桥左幅1跨梁裂缝\n2018与2017测试桥左幅1跨梁同一裂缝"
    )
    slices = [SourceSlice(region.source_id, 0, len(region.text), region.text) for region in snapshot.regions]
    task = ChunkTask("history", slices[:1], slices[1:])
    identity = {"bridge": "测试桥", "span": "左幅", "component": "梁", "position": "1跨", "year": 2018}
    row = {
        "source_id": "p0", "source_key": slices[0].key, "instance_id": "d1",
        "fields": {"病害描述": "裂缝"}, "quotes": {"病害描述": "裂缝"},
        "identity": identity, "history_identity": {**identity, "year": 2017},
        "temporal": "已有", "history_sources": ["p1"],
        **{
            key: [{"source_key": source.key, "quote": source.text}]
            for key, source in zip(("current_evidence", "history_evidence", "relationship_evidence"), slices)
        },
    }
    value = {"covered": task.owned_keys, "基本信息": {}, "维修建议": [], "病害明细": [row]}
    fragment = parse_fragment("```json\n" + encoded(value) + "\n```", task, snapshot)
    disease = fragment.diseases[0]
    assert disease["temporal"] == "已有"
    assert disease["temporal_verification"] == "explicit_visible_evidence"
    assert disease["history_evidence"][0]["source_key"] == task.context[0].key
    assert disease["relationship_evidence"][0]["source_key"] == task.context[1].key


def test_fenced_raw_is_preserved_in_receipt_and_generation_evidence(tmp_path):
    snapshot = DocumentSnapshot.from_text(RECOVERY_SOURCE)
    raw = "```json\n" + encoded(valid_output()) + "\n```"
    config = validate_config({"input_budget": 4096, "output_budget": 1024})
    calls = []

    def synthetic_backend(ids, maximum):
        calls.append((ids, maximum))
        return {"text": raw, "finish_reason": "stop", "output_tokens": 512}

    result = ci.infer_chunked(snapshot, CharTokenizer(), synthetic_backend, config=config, cache_dir=tmp_path)
    assert len(calls) == 1
    call = result["model_calls"][0]
    digest = hashlib.sha256(raw.encode()).hexdigest()
    assert call["status"] == "success" and call["output_sha256"] == digest
    assert result["generation_evidence"]["responses"] == {call["call_id"]: raw}
    receipt = json.loads(next(tmp_path.glob("**/raw-responses/*.json")).read_text())
    assert receipt["raw"] == raw
    assert receipt["raw_sha256"] == digest


def test_exact_prompt_is_opt_in_and_preserves_existing_prompt_bytes():
    assert hashlib.sha256(LEGACY_FRAGMENT_INSTRUCTION.encode()).hexdigest() == "a5e63a27215f01eaaa85f5bd4b2f1ad60c7410b28e4f57c8eec3145d3f66b9f1"
    assert hashlib.sha256(FRAGMENT_INSTRUCTION.encode()).hexdigest() == "c63fe1823de5cf8c5c2d63b7bb21aff2872c613f4ac9e1760eef692d5b39c1d6"
    assert validate_config({})["prompt"] == "source-fragment-v1"
    assert ci._instruction(validate_config({})) == FRAGMENT_INSTRUCTION
    concise = ci._instruction(validate_config({"prompt": "source-fragment-concise-v1"}))
    assert hashlib.sha256(concise.encode()).hexdigest() == "8e0467f79f3a5289546fc98f1882d220de59de9dff5bbc5d647a73413fe841bd"
    exact = ci._instruction(validate_config({"prompt": "source-fragment-exact-v1"}))
    assert exact.startswith(concise)
    for requirement in ("UTF-8", "不得", "空格", "换行", "省略", "null", "quote"):
        assert requirement in exact[len(concise):]
    assert "RSI" not in exact and "2018" not in exact


@pytest.mark.parametrize("prompt", ["source-fragment-exact-v1", "source-fragment-exact-v2"])
@pytest.mark.parametrize("workflow", ["chunked", "bounded-react"])
def test_exact_prompt_is_available_only_to_fragment_workflows(workflow, prompt):
    config = validate_config({"workflow": workflow, "prompt": prompt})
    assert config["workflow"] == workflow


@pytest.mark.parametrize("prompt", ["source-fragment-exact-v1", "source-fragment-exact-v2"])
def test_whole_workflow_cannot_silently_ignore_exact_prompt(prompt):
    with pytest.raises(ValueError, match="fragment|chunked|whole"):
        validate_config({"workflow": "whole", "prompt": prompt})
    with pytest.raises(ValueError, match="unsupported whole prompt"):
        whole_instruction(prompt)


@pytest.mark.parametrize("prompt", ["source-fragment-exact-v1", "source-fragment-exact-v2"])
def test_exact_prompt_budget_includes_full_template_and_keeps_oversize_source_tail(prompt):
    tokenizer = CharTokenizer()
    config = validate_config({"prompt": prompt, "input_budget": 8192, "output_budget": 4096})
    source = "裂缝最大宽度0.3mm；" * 2000 + "末尾必须保留"
    snapshot = DocumentSnapshot("b" * 64, [SourceRegion("t0r0", 0, source, kind="table-row")])
    plan = ci.prepare_chunk_plan(snapshot, tokenizer, config)
    assert plan.instruction == ci._instruction(config)
    assert len(plan.tasks) > 1 and plan.coverage(snapshot)["complete"]
    assert "".join(part.text for task in plan.tasks for part in task.owned) == source
    for task in plan.tasks:
        ids = build_prompt(tokenizer, task.text(), max_model_len=12288, max_new_tokens=4096,
                           instruction=plan.instruction)
        assert len(ids) == task.prompt_tokens <= 8192


@pytest.mark.parametrize("prompt", ["source-fragment-exact-v1", "source-fragment-exact-v2"])
def test_sha_verified_fixed9b_tokenizer_budgets_offline(prompt):
    """Optional local asset verification; never download or instantiate a model."""
    import ast
    import copy
    import os
    from datetime import datetime
    from pathlib import Path

    directory = os.environ.get("CITYAGENT_TEST_FIXED9B_TOKENIZER_DIR")
    if directory is None:
        pytest.skip("Set CITYAGENT_TEST_FIXED9B_TOKENIZER_DIR to pinned offline fixed9B assets")
    source_path = Path(__file__).resolve().parents[1] / "src/bridge_evidence/contest/divide_cli.py"
    tree = ast.parse(source_path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_MergedChatTokenizer")
    namespace = dict(copy=copy, hashlib=hashlib, json=json, Path=Path, datetime=datetime)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)

    class PromptTokenizer(namespace["_MergedChatTokenizer"]):
        def __call__(self, text, *, add_special_tokens=False, truncation=False):
            assert truncation is False
            return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    # Pinned model-identity receipt from fixed9B cloud-evidence-v3; template/config
    # differ from the old source-object-review tokenizer fixture.
    tokenizer = PromptTokenizer(directory, {
        "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
        "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    })
    config = validate_config({"prompt": prompt, "input_budget": 8192, "output_budget": 4096})
    for source in (RECOVERY_SOURCE, "裂缝最大宽度0.3mm；" * 4000 + "末尾必须保留"):
        snapshot = DocumentSnapshot.from_text(source)
        plan = ci.prepare_chunk_plan(snapshot, tokenizer, config)
        assert plan.coverage(snapshot)["complete"]
        assert "".join(part.text for task in plan.tasks for part in task.owned) == source
        for task in plan.tasks:
            ids = build_prompt(tokenizer, task.text(), max_model_len=12288, max_new_tokens=4096,
                               instruction=plan.instruction)
            assert len(ids) == task.prompt_tokens <= 8192


def test_exact_v2_extracts_supported_basic_fields_without_changing_v1_bytes():
    v1 = ci._instruction(validate_config({"prompt": "source-fragment-exact-v1"}))
    assert hashlib.sha256(v1.encode()).hexdigest() == "0c66ad0e45a7b63d43fe3b215ddcd552b7498dc9b0aa928cc1df32efcae3419b"
    v2 = ci._instruction(validate_config({"prompt": "source-fragment-exact-v2"}))
    assert v2.startswith(v1)
    added = v2[len(v1):]
    for requirement in ("明确支持", "必须提取", "basic_sources", "basic_quotes", "null", "不得", "规避"):
        assert requirement in added
    assert "RSI" not in added and "2018" not in added and "恢复验证桥" not in added


def test_all_null_basic_fragment_remains_valid_but_complete_report_is_rejected():
    # Reproduce the v7 failure shape synthetically; this is not the cloud raw.
    # A fragment may cover a disease-only section of a long report, so absence
    # of basic fields is checked only after all fragments have been merged.
    snapshot = DocumentSnapshot.from_text(RECOVERY_SOURCE)
    plan = ci.prepare_chunk_plan(snapshot, CharTokenizer(), validate_config({}))
    value = valid_output()
    value["基本信息"] = dict.fromkeys(sorted(BASIC_FIELDS))
    value["basic_sources"] = {}
    value["basic_quotes"] = {}
    raw = encoded(value)
    fragment = parse_fragment(raw, plan.tasks[0], snapshot, "synthetic-all-null")
    assert len(fragment.basic) == 17 and all(v is None for v in fragment.basic.values())
    merged = merge_fragments(snapshot, plan, [fragment])
    with pytest.raises(ValueError, match="meaningless_basic_information"):
        _parse_report(encoded(merged["data"]))
    valid = parse_fragment(encoded(valid_output()), plan.tasks[0], snapshot, "synthetic-supported-basic")
    assert _parse_report(encoded(merge_fragments(snapshot, plan, [valid])["data"]))["基本信息"]["桥梁名称"]
