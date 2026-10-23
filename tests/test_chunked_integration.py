import json

import pytest

from bridge_evidence.contest.candidate import validate_config
from bridge_evidence.contest.chunked_inference import (
    ChunkFailure,
    infer_chunked,
    read_evidence,
)
from bridge_evidence.contest.long_document import DocumentSnapshot


class CharTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages)

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text))}


class FragmentBackend:
    def __init__(self, length_once=False, invalid_once=False):
        self.calls = 0
        self.length_once = length_once
        self.invalid_once = invalid_once

    def __call__(self, ids, maximum):
        self.calls += 1
        if self.length_once and self.calls == 1:
            return {"text": "{", "finish_reason": "length", "output_tokens": maximum}
        if self.invalid_once and self.calls == 1:
            return {"text": "invalid", "finish_reason": "stop", "output_tokens": 2}
        text = "".join(map(chr, ids))
        payload = json.JSONDecoder().raw_decode(text[text.index('{"owned":') :])[0]
        refs = payload["owned"]
        basic = (
            {"桥梁名称": "测试桥"} if any(r["source_id"] == "p0" for r in refs) else {}
        )
        value = {
            "covered": [r["key"] for r in refs],
            "基本信息": basic,
            "basic_sources": {"桥梁名称": "p0"} if basic else {},
            "维修建议": [],
            "病害明细": [
                {
                    "source_id": r["source_id"],
                    "instance_id": r["key"],
                    "fields": {"病害部位": "梁", "病害描述": r["text"]},
                    "temporal": "证据不足",
                }
                for r in refs
            ],
        }
        return {
            "text": json.dumps(value, ensure_ascii=False),
            "finish_reason": "stop",
            "output_tokens": 100,
        }


def test_length_splits_and_failure_cost_is_retained():
    snapshot = DocumentSnapshot.from_text("测试桥\n裂缝\n裂缝")
    backend = FragmentBackend(length_once=True)
    result = infer_chunked(
        snapshot,
        CharTokenizer(),
        backend,
        config=validate_config({"input_budget": 3000, "output_budget": 500}),
    )
    assert result["coverage"]["complete"]
    assert result["model_calls"][0]["status"] == "failed"
    assert len(result["data"]["病害明细"]) == 3
    assert result["seconds"] == sum(c["seconds"] for c in result["model_calls"])


def test_invalid_json_gets_feedback_without_partial_publish():
    result = infer_chunked(
        DocumentSnapshot.from_text("测试桥"),
        CharTokenizer(),
        FragmentBackend(invalid_once=True),
        config=validate_config({"input_budget": 3000, "output_budget": 500}),
    )
    assert len(result["model_calls"]) == 2
    assert (
        result["model_calls"][0]["prompt_sha256"]
        != result["model_calls"][1]["prompt_sha256"]
    )


def test_tools_reject_unknown_sources_and_extra_parameters():
    snapshot = DocumentSnapshot.from_text("桥")
    with pytest.raises(ValueError):
        read_evidence(snapshot, {"source_ids": ["another"]})
    with pytest.raises(ValueError):
        read_evidence(snapshot, {"source_ids": ["p0"], "path": "/dataset/label"})


def test_cached_calls_retained_and_not_reissued(tmp_path):
    snapshot = DocumentSnapshot.from_text("测试桥")
    config = validate_config({"input_budget": 3000, "output_budget": 500})
    first = FragmentBackend()
    a = infer_chunked(
        snapshot, CharTokenizer(), first, config=config, cache_dir=tmp_path
    )
    second = FragmentBackend()
    b = infer_chunked(
        snapshot, CharTokenizer(), second, config=config, cache_dir=tmp_path
    )
    assert second.calls == 0
    assert a["model_calls"] == b["model_calls"]


def test_timeout_is_bounded_and_failed_report_not_success():
    def failing(ids, maximum):
        raise TimeoutError("temporary")

    with pytest.raises(ChunkFailure) as failure:
        infer_chunked(
            DocumentSnapshot.from_text("桥"),
            CharTokenizer(),
            failing,
            config=validate_config({"input_budget": 3000, "output_budget": 500}),
            max_attempts=2,
        )
    assert len(failure.value.model_calls) == 2
    assert all(c["status"] == "failed" for c in failure.value.model_calls)


def test_conflict_needs_evidence_and_react_budget_is_two():
    from bridge_evidence.contest.chunked_inference import bounded_repair

    snapshot = DocumentSnapshot.from_text("桥总体评分82")
    responses = iter(
        [
            {"tool": "read_evidence", "arguments": {"source_ids": ["p0"]}},
            {"resolved": {"总体评分": {"value": "82", "source_id": "p0"}}},
        ]
    )

    def backend(ids, maximum):
        return {
            "text": json.dumps(next(responses)),
            "finish_reason": "stop",
            "output_tokens": 20,
        }

    merged = {
        "data": {"基本信息": {}},
        "conflicts": [{"field": "总体评分", "values": ["81", "82"]}],
        "contributing_call_ids": [],
    }
    calls = []
    result = bounded_repair(
        snapshot,
        merged,
        CharTokenizer(),
        backend,
        validate_config({"input_budget": 3000, "output_budget": 500}),
        calls,
    )
    assert len(calls) == 2
    assert result["data"]["基本信息"]["总体评分"] == "82"
    assert result["conflicts"] == []
    assert len(result["tool_events"]) == 1


def test_react_rejects_illegal_tool_and_does_not_issue_second_request():
    from bridge_evidence.contest.chunked_inference import bounded_repair

    calls = []

    def backend(ids, maximum):
        return {
            "text": '{"tool":"read_label","arguments":{}}',
            "finish_reason": "stop",
            "output_tokens": 10,
        }

    with pytest.raises(ValueError, match="illegal"):
        bounded_repair(
            DocumentSnapshot.from_text("桥"),
            {"conflicts": [{"field": "评分"}]},
            CharTokenizer(),
            backend,
            validate_config({"input_budget": 3000, "output_budget": 500}),
            calls,
        )
    assert len(calls) == 1


def test_cache_modified_raw_cannot_be_reused(tmp_path):
    snapshot = DocumentSnapshot.from_text("测试桥")
    config = validate_config({"input_budget": 3000, "output_budget": 500})
    infer_chunked(
        snapshot, CharTokenizer(), FragmentBackend(), config=config, cache_dir=tmp_path
    )
    path = next(tmp_path.rglob("*.json"))
    value = json.loads(path.read_text())
    value["raw"] = value["raw"].replace("测试桥", "伪造桥")
    path.write_text(json.dumps(value))
    with pytest.raises(ChunkFailure, match="cached"):
        infer_chunked(
            snapshot,
            CharTokenizer(),
            FragmentBackend(),
            config=config,
            cache_dir=tmp_path,
        )

def test_v2_execution_record_binds_merged_output_and_source_calls(tmp_path):
    from copy import deepcopy

    from bridge_evidence.contest.docx_writer import write_extraction_docx
    from bridge_evidence.contest.execution_audit import validate_execution_record
    from bridge_evidence.contest.run_state import file_digest, json_digest
    from bridge_evidence.contest.tong_workflow import execute_workflow
    result=infer_chunked(DocumentSnapshot.from_text('测试桥\n裂缝'),CharTokenizer(),FragmentBackend(),
                         config=validate_config({'input_budget':3000,'output_budget':500}))
    output=tmp_path/'bridge.docx';write_extraction_docx(result['data'],str(output))
    workflow=execute_workflow(result,{s:lambda v:v for s in ('prepare','infer','validate','render')},engine='direct')['workflow']
    doc={'relative_path':'bridge.docx','sha256':'a'*64};fingerprint='b'*64
    record={**result,'schema_version':2,'document':doc,'run_fingerprint':fingerprint,'status':'success',
        'model_validation':{'valid':True},'workflow':workflow,'output':{'name':'bridge.docx','sha256':file_digest(output)},
        'final_data':result['data'],'final_json_sha256':json_digest(result['data']),
        'fusion':{'result_sha256':json_digest(result['data'])}}
    # This fixture deliberately exercises historical V2 compatibility, not the
    # new source-bound raw-response reconstruction / formal admission profile.
    record.pop('generation_evidence', None)
    def audit(r):return validate_execution_record(r,expected_document=doc,expected_run_fingerprint=fingerprint,output_path=output,profile='legacy')
    assert audit(record)==[]
    for key,value in [('contributing_call_ids',[]),('coverage',{'complete':True}),('final_json_sha256','c'*64)]:
        changed=deepcopy(record);changed[key]=value;assert audit(changed)
