"""The visual prompt states the same ID contract enforced by the parser."""

import json

import pytest
from test_contest_visual_evidence import document, plan_for, rebuild, response

from bridge_evidence.contest.visual_evidence import (
    build_visual_request,
    execute_visual_jobs,
)


def test_typed_request_prompt_declares_ascii_string_ids_and_global_uniqueness(tmp_path):
    _, plan = plan_for(document(tmp_path))
    request = build_visual_request(plan, plan["jobs"][0])
    prompt = request["instruction"]
    assert "JSON STRING" in prompt
    assert "1-64 ASCII characters" in prompt
    assert "[A-Za-z0-9_-]" in prompt
    assert '"t1"' in prompt and '"o1"' in prompt and '"u1"' in prompt
    assert "globally unique across all three lists" in prompt
    assert "Never use numeric IDs" in prompt
    assert f'job_id:"{request["job_id"]}"' in prompt


@pytest.mark.parametrize("invalid_id", [1, 1.0, None, "图1", "x" * 65])
def test_invalid_ids_remain_rejected_without_rewriting_raw(tmp_path, invalid_id):
    native, plan = plan_for(document(tmp_path))

    def generate(request, maximum):
        receipt = response(request, maximum)
        body = json.loads(receipt["text"])
        body["text_blocks"][0]["id"] = invalid_id
        receipt["text"] = json.dumps(body, ensure_ascii=False)
        return receipt

    result = execute_visual_jobs(plan, generate, tmp_path / "cache")
    assert result["errors"]
    assert (
        json.loads(result["raw"][0]["response"]["text"])["text_blocks"][0]["id"]
        == invalid_id
    )
    assert rebuild(native, plan, result).unverified_content == native.unverified_content


@pytest.mark.parametrize("duplicate", [False, True])
def test_ids_must_be_unique_across_text_and_observations(tmp_path, duplicate):
    native, plan = plan_for(document(tmp_path))

    def generate(request, maximum):
        receipt = response(request, maximum)
        body = json.loads(receipt["text"])
        body["observations"] = [
            {
                "id": "t1" if duplicate else "o1",
                "bbox": [0, 0, 1, 1],
                "description": "visible surface",
            }
        ]
        receipt["text"] = json.dumps(body)
        return receipt

    result = execute_visual_jobs(plan, generate, tmp_path / "cache")
    effective = rebuild(native, plan, result)
    if duplicate:
        assert (
            result["errors"]
            and effective.unverified_content == native.unverified_content
        )
        assert (
            json.loads(result["raw"][0]["response"]["text"])["observations"][0]["id"]
            == "t1"
        )
    else:
        assert not result["errors"] and not effective.unverified_content
