"""A picture interpretation cannot be laundered into literal report evidence."""
import json

import pytest

from bridge_evidence.contest.long_document import (
    ChunkPlan,
    ChunkTask,
    DocumentSnapshot,
    SourceRegion,
    SourceSlice,
    parse_fragment,
)


def setup(kind="visual-observation", text="表面有裂缝"):
    region = SourceRegion("v1", 0, text, kind, {
        "source_path": "word/document.xml#/drawing[1]", "native_anchor": "p1",
        "call_id": "visual-call", "bbox": [0, 0, 1, 1], "view_sha256": "a" * 64,
    })
    snapshot = DocumentSnapshot("b" * 64, [region], "evidence-manifest-v2+visual-evidence-v1")
    source = SourceSlice(region.source_id, 0, len(text), text)
    task = ChunkTask("task", [source], [])
    value = {"covered": task.owned_keys, "基本信息": {}, "维修建议": [], "病害明细": []}
    return snapshot, task, value


def row(task, **fields):
    return {"source_id": "v1", "source_key": task.owned_keys[0], "fields": fields,
            "quotes": fields.copy(), "temporal": "证据不足"}


def parse(snapshot, task, value):
    return parse_fragment(json.dumps(value, ensure_ascii=False), task, snapshot, "text-call")


def test_visual_appearance_preserves_domain_bbox_and_upstream_call():
    snapshot, task, value = setup()
    value["病害明细"] = [row(task, 病害描述="表面有裂缝", 病害类型="裂缝")]
    fragment = parse(snapshot, task, value)
    defect = fragment.diseases[0]
    assert defect["source_ranges"][0]["evidence_domain"] == "visual-observation"
    assert defect["source_ranges"][0]["upstream_call_id"] == "visual-call"
    assert defect["source_ranges"][0]["bbox"] == [0, 0, 1, 1]
    assert defect["fields"]["是否新增"] is None


@pytest.mark.parametrize("extra", [False, True])
def test_observation_cannot_assert_basic_information_even_if_model_wrote_it(extra):
    snapshot, task, value = setup(text="示例桥2018年评分98")
    if extra:
        value["basic_observations"] = [{"field": "总体评分", "value": 98,
             "source_id": "v1", "source_key": task.owned_keys[0], "quote": "评分98"}]
    else:
        value.update({"基本信息": {"总体评分": 98}, "basic_sources": {"总体评分": "v1"},
                      "basic_quotes": {"总体评分": "评分98"}})
    fragment = parse(snapshot, task, value)
    assert fragment.basic.get("总体评分") is None
    assert fragment.basic_evidence == {} and fragment.basic_observations == []


@pytest.mark.parametrize("field,text", [
    ("病害部位", "左幅一号墩"), ("报告日期", "去年"), ("桥梁名称", "示例桥"),
    ("病害描述", "裂缝宽0.3毫米"), ("病害描述", "裂缝宽三毫米"),
    ("病害描述", "裂缝宽半毫米"), ("病害描述", "裂缝长两米"),
    ("病害描述", "裂缝长数厘米"), ("病害描述", "裂缝宽半 mm"),
    ("是否新增", "是"), ("上一次定检状态", "已修复"),
])
def test_observation_cannot_invent_identity_measurements_or_history(field, text):
    snapshot, task, value = setup(text=text)
    value["病害明细"] = [row(task, **{field: text})]
    assert parse(snapshot, task, value).diseases == []


def test_observation_cannot_generate_advice_or_unquoted_defect():
    snapshot, task, value = setup(text="建议修补裂缝")
    value["维修建议"] = [row(task, 建议内容="建议修补裂缝")]
    assert parse(snapshot, task, value).advice == []
    value["维修建议"] = []
    value["病害明细"] = [row(task, 病害描述="裂缝")]
    value["病害明细"][0]["quotes"] = {}
    # A verbatim appearance value is bound to the observation it came from.
    defect = parse(snapshot, task, value).diseases[0]
    assert defect["quotes"] == {"病害描述": "裂缝"}
    assert defect["evidence_domain"] == "visual-observation"
    value["病害明细"] = [row(task, 病害描述="大面积剥落")]
    value["病害明细"][0]["quotes"] = {}
    assert parse(snapshot, task, value).diseases == []


def test_visual_transcription_remains_distinct_and_can_support_visible_text():
    snapshot, task, value = setup(kind="visual-transcription", text="示例桥2018年评分98")
    value.update({"基本信息": {"总体评分": 98}, "basic_sources": {"总体评分": "v1"},
                  "basic_quotes": {"总体评分": "评分98"}})
    fragment = parse(snapshot, task, value)
    assert fragment.basic_evidence["总体评分"]["evidence_domain"] == "visual-transcription"
    assert fragment.basic_evidence["总体评分"]["upstream_call_id"] == "visual-call"


def test_coverage_does_not_clear_unresolved_visual_objects():
    snapshot, task, _ = setup()
    plan = ChunkPlan(snapshot.digest, [task], 8192, 4096)
    assert plan.coverage(snapshot)["document_complete"]
    snapshot.unverified_content.append({"kind": "OLEObject"})
    assert not plan.coverage(snapshot)["document_complete"]


def test_react_cannot_bypass_observation_domain_to_resolve_basic_conflicts():
    from bridge_evidence.contest.candidate import validate_config
    from bridge_evidence.contest.chunked_inference import bounded_repair

    snapshot, _, _ = setup(text="总体评分98")
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(m["content"] for m in messages)
        def __call__(self, text, **kwargs):
            return {"input_ids": list(map(ord, text))}

    def backend(ids, maximum):
        text = "".join(map(chr, ids))
        assert '"available_sources": []' in text
        return {"text": json.dumps({"tool": "read_evidence", "arguments": {"source_ids": ["v1"]}}),
                "finish_reason": "stop", "output_tokens": 50}

    calls = []
    merged = {"data": {"基本信息": {"总体评分": "80"}},
              "conflicts": [{"field": "总体评分", "values": ["80", "85"]}],
              "contributing_call_ids": []}
    with pytest.raises(ValueError, match="visual observation cannot support basic conflict"):
        bounded_repair(snapshot, merged, Tokenizer(), backend,
                       validate_config({"input_budget": 3000, "output_budget": 500}), calls)
    assert len(calls) == 1
    assert merged["conflicts"] and merged["data"]["基本信息"]["总体评分"] == "80"
