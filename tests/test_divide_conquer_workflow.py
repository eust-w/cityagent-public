"""Orchestration integration with synthetic client/SDK; no real inference here."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest.divide_conquer_workflow import DivideSettings, DivideWorkflow
from bridge_evidence.contest.evidence_manifest import manifest_from_text


class FakeLocalClient:
    """Return explicit fixture facts for named source units, never call a model."""

    def __init__(self, manifest, *, truncate_first=False):
        self.units = {unit["id"]: unit for unit in manifest["units"]}
        self.calls = []
        self.requests = []
        self.truncate_first = truncate_first

    def count_tokens(self, text):
        # Synthetic weights force one source unit per small chunk while still
        # budgeting context. These numbers do not represent actual model tokens.
        return 100 * len(re.findall(r"\[E\d+ chars \d+:\d+\]", text)) + len(text) // 5

    def generate(self, messages, max_tokens=None):
        payload = json.loads(messages[-1]["content"])
        self.requests.append(payload)
        call_id = f"synthetic-{len(self.calls) + 1}"
        if self.truncate_first and not self.calls:
            result = {"text": '{"基本信息":', "status": "failed", "finish_reason": "length", "call_id": call_id}
        elif "conflict" in payload:
            candidate = payload["conflict"]["candidates"][-1]
            result = {"text": json.dumps({"value": candidate["value"], "evidence_ids": candidate["evidence_ids"]}),
                      "status": "success", "finish_reason": "stop", "call_id": call_id}
        else:
            data = {"基本信息": {}, "维修建议": [], "病害明细": [],
                    "evidence": {"基本信息": {}, "维修建议": [], "病害明细": []}}
            for evidence_id in payload["primary_evidence_ids"]:
                text = self.units[evidence_id]["text"].strip()
                for prefix, field in (("桥梁名称：", "桥梁名称"), ("报告日期：", "报告日期")):
                    if text.startswith(prefix):
                        data["基本信息"][field] = text.removeprefix(prefix)
                        data["evidence"]["基本信息"][field] = [evidence_id]
                if text.startswith("建议："):
                    data["维修建议"].append({"建议类别": "尽快维修", "建议内容": text.removeprefix("建议："),
                                          "病害部位": "伸缩缝"})
                    data["evidence"]["维修建议"].append([evidence_id])
                if text.startswith("病害："):
                    data["病害明细"].append({"病害部位": "原文指定部位", "病害类型": "裂缝",
                                          "病害描述": text.removeprefix("病害："), "是否新增": "未知",
                                          "上一次定检状态": "未知", "发展程度": "未知"})
                    data["evidence"]["病害明细"].append([evidence_id])
            result = {"text": json.dumps(data, ensure_ascii=False), "status": "success",
                      "finish_reason": "stop", "call_id": call_id}
        self.calls.append({**result, "kind": "generation", "synthetic": True})
        return result


def fake_react(prompt, handlers, config):
    """Exercise the same keyword argument boundary used by the SDK adapter."""
    active = json.loads(prompt.rsplit("\n", 1)[1])
    if "task_id" in active:
        outcome = handlers["extract_chunk"](task_id=active["task_id"], extra_evidence_ids=[])
        if outcome["status"] == "needs_split":
            handlers["split_task"](task_id=active["task_id"])
        elif outcome["status"] != "extracted":
            raise AssertionError(outcome)
    else:
        handlers["review_conflict"](conflict_id=active["id"], extra_evidence_ids=[])
    handlers["finish"]()
    return {"status": "completed", "native_final_event": True, "synthetic": True}


def make_workflow(tmp_path, text=None, *, mode="divide-react", runner=fake_react,
                  truncate_first=False, chunk_tokens=200):
    manifest = manifest_from_text(text or (
        "桥梁名称：合成桥\n建议：修复伸缩缝\n病害：右幅桥台裂缝\n病害：左幅桥墩裂缝\n"))
    client = FakeLocalClient(manifest, truncate_first=truncate_first)
    workflow = DivideWorkflow(manifest, client, {}, tmp_path,
                              DivideSettings(chunk_tokens=chunk_tokens, max_tool_tokens=10000),
                              mode=mode, react_runner=runner)
    return workflow, client


@pytest.mark.parametrize("mode", ["divide-fixed", "divide-react"])
def test_whole_report_three_sections_and_complete_evidence_coverage(tmp_path, mode):
    workflow, client = make_workflow(tmp_path, mode=mode)
    result = workflow.run()
    data, receipt = result["data"], result["receipt"]
    assert data["基本信息"]["桥梁名称"] == "合成桥"
    assert len(data["维修建议"]) == 1
    assert [row["病害描述"] for row in data["病害明细"]] == ["右幅桥台裂缝", "左幅桥墩裂缝"]
    assert [row["序号"] for row in data["病害明细"]] == ["1", "2"]
    assert receipt["coverage"]["complete"] is True
    assert receipt["coverage"]["missing_spans"] == []
    assert receipt["status"] == "success"
    assert receipt["submission_ready"] is False
    assert (tmp_path / "report.json").is_file()
    completed = [task for task in receipt["tasks"] if task["status"] == "completed"]
    assert len(client.calls) == len(completed)
    assert all(task["attempts"][-1]["call_id"] in {call["call_id"] for call in client.calls} for task in completed)
    assert bool(receipt["agent_runs"]) == (mode == "divide-react")


def test_truncated_parent_is_split_and_only_complete_children_contribute(tmp_path):
    workflow, client = make_workflow(tmp_path, truncate_first=True, chunk_tokens=500)
    result = workflow.run()
    tasks = result["receipt"]["tasks"]
    parent = tasks[0]
    assert parent["status"] == "split"
    assert parent["attempts"][0]["finish_reason"] == "length"
    assert {task["id"] for task in tasks[1:]} == set(parent["children"])
    assert all(task["status"] == "completed" and task["depth"] == 1 for task in tasks[1:])
    assert result["receipt"]["coverage"]["complete"]
    assert len(result["data"]["病害明细"]) == 2
    assert parent["id"] not in {task_id for row in result["receipt"]["provenance"]["病害明细"] for task_id in row["task_ids"]}
    assert len(client.calls) == 3


def test_finish_before_extraction_rejected_even_if_sdk_claims_completion(tmp_path):
    def early_finish(prompt, handlers, config):
        with pytest.raises(ValueError, match="no complete extraction"):
            handlers["finish"]()
        return {"status": "completed", "synthetic": True}

    workflow, client = make_workflow(tmp_path, runner=early_finish)
    with pytest.raises(RuntimeError, match="did not finish"):
        workflow.run()
    assert client.calls == []
    assert not (tmp_path / "report.json").exists()
    receipt = json.loads((tmp_path / "execution.json").read_text())
    assert receipt["status"] == "failed"
    assert not receipt["coverage"]["complete"]


def test_conflicting_basic_fields_require_model_review_and_provenance(tmp_path):
    text = "桥梁名称：合成桥\n报告日期：2018-12-16\n报告日期：2018-12-20\n"
    workflow, client = make_workflow(tmp_path, text, chunk_tokens=200)
    result = workflow.run()
    assert result["data"]["基本信息"]["报告日期"] == "2018-12-20"
    reviews = result["receipt"]["reviews"]
    assert len(reviews) == 1 and reviews[0]["resolved"]
    assert client.requests[-1]["conflict"]["field"] == "报告日期"
    assert {unit["text"].strip() for unit in client.requests[-1]["evidence"]} == {
        "报告日期：2018-12-16", "报告日期：2018-12-20"}
    provenance = result["receipt"]["provenance"]["基本信息"]["报告日期"]
    assert provenance["resolution"]["value"] == "2018-12-20"


def test_sdk_error_after_worker_success_never_publishes_report(tmp_path):
    def fail_after_worker(prompt, handlers, config):
        active = json.loads(prompt.rsplit("\n", 1)[1])
        assert handlers["extract_chunk"](task_id=active["task_id"], extra_evidence_ids=[])["status"] == "extracted"
        handlers["finish"]()
        return {"status": "error", "error": "synthetic SDK final failure"}

    workflow, client = make_workflow(tmp_path, runner=fail_after_worker)
    with pytest.raises(RuntimeError, match="did not finish"):
        workflow.run()
    assert len(client.calls) == 1
    assert not (tmp_path / "report.json").exists()
    receipt = json.loads((tmp_path / "execution.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["tasks"][0]["status"] == "failed"


def test_sdk_read_evidence_keyword_contract(tmp_path):
    workflow, _ = make_workflow(tmp_path)
    evidence_id = workflow.manifest["units"][0]["id"]
    records = workflow._read_handlers()["read_evidence"](evidence_ids=[evidence_id])
    assert records == [workflow.manifest["units"][0]]


def test_new_context_reextraction_replaces_previous_contribution_once(tmp_path):
    workflow, client = make_workflow(
        tmp_path, "桥梁名称：合成桥\n病害：P1 裂缝\n历史：P1 已有同位置裂缝\n")
    historical_id = workflow.manifest["units"][-1]["id"]
    original_generate = client.generate

    def contextual_generate(messages, max_tokens=None):
        response = original_generate(messages, max_tokens=max_tokens)
        payload = json.loads(messages[-1]["content"])
        if payload["CONTEXT"]["additional"]:
            data = json.loads(response["text"])
            for row, refs in zip(data["病害明细"], data["evidence"]["病害明细"]):
                row.update(是否新增="否", 上一次定检状态="同位置已有裂缝", 发展程度="基本稳定")
                refs.append(historical_id)
            response["text"] = json.dumps(data, ensure_ascii=False)
            client.calls[-1]["text"] = response["text"]
        return response

    def refine_after_extraction(prompt, handlers, config):
        active = json.loads(prompt.rsplit("\n", 1)[1])
        extracted = handlers["extract_chunk"](task_id=active["task_id"], extra_evidence_ids=[])
        if extracted["defect_count"]:
            assert extracted["unknown_history_count"] == 1
            assert extracted["can_reextract_with_extra_evidence"]
            assert handlers["read_evidence"](evidence_ids=[historical_id])
            revised = handlers["extract_chunk"](task_id=active["task_id"], extra_evidence_ids=[historical_id])
            assert revised["status"] == "extracted" and revised["unknown_history_count"] == 0
            assert not revised["can_reextract_with_extra_evidence"]
            # Replaying the accepted evidence is idempotent, not a third model call.
            before = len(client.calls)
            handlers["extract_chunk"](task_id=active["task_id"], extra_evidence_ids=[historical_id])
            assert len(client.calls) == before
        handlers["finish"]()
        return {"status": "completed", "native_final_event": True, "synthetic": True}

    client.generate = contextual_generate
    workflow.react_runner = refine_after_extraction
    result = workflow.run()
    assert len(result["data"]["病害明细"]) == 1
    assert result["data"]["病害明细"][0]["是否新增"] == "否"
    refined = [task for task in result["receipt"]["tasks"] if len(task["attempts"]) == 2]
    assert len(refined) == 1
    assert refined[0]["accepted_call_id"] == refined[0]["attempts"][-1]["call_id"]
    assert refined[0]["accepted_extra_ids"] == [historical_id]
    provenance = result["receipt"]["provenance"]["病害明细"]
    assert len(provenance) == 1 and historical_id in provenance[0]["evidence_ids"]
    assert result["receipt"]["coverage"]["complete"]


@pytest.mark.parametrize("failure", ["length", "invalid_json"])
def test_failed_new_context_invalidates_previous_result_before_finish_or_split(tmp_path, failure):
    workflow, client = make_workflow(tmp_path)
    unit = workflow.manifest["units"][0]
    extra_id = workflow.manifest["units"][-1]["id"]
    span = {"coverage_id": unit["id"], "start": 0, "end": len(unit["text"])}
    task = workflow._new_task({"spans": [span]})
    original_generate = client.generate

    def second_attempt_fails(messages, max_tokens=None):
        if client.calls:
            result = {"text": "not a complete extraction", "call_id": "synthetic-failed-refinement",
                      "status": "failed" if failure == "length" else "success",
                      "finish_reason": failure if failure == "length" else "stop"}
            client.calls.append({**result, "kind": "generation", "synthetic": True})
            return result
        return original_generate(messages, max_tokens=max_tokens)

    def refine_and_split(prompt, handlers, config):
        assert handlers["extract_chunk"](task_id=task["id"], extra_evidence_ids=[])["status"] == "extracted"
        outcome = handlers["extract_chunk"](task_id=task["id"], extra_evidence_ids=[extra_id])
        assert outcome["status"] in {"needs_split", "retryable_error"}
        assert task["status"] == "pending"
        assert not {"result", "accepted_call_id", "accepted_extra_ids"}.intersection(task)
        with pytest.raises(ValueError, match="no complete extraction"):
            handlers["finish"]()
        handlers["split_task"](task_id=task["id"])
        handlers["finish"]()
        return {"status": "completed", "native_final_event": True, "synthetic": True}

    client.generate = second_attempt_fails
    workflow.react_runner = refine_and_split
    children = workflow._run_task(task)
    assert task["status"] == "split" and len(children) == 2
    assert all(child["status"] == "pending" for child in children)
    assert workflow.reducer.provenance()["基本信息"] == {}
    assert not workflow.checkpoint()["coverage"]["complete"]
    assert children[0]["spans"][0]["start"] == 0
    assert children[0]["spans"][0]["end"] == children[1]["spans"][0]["start"]
    assert children[1]["spans"][0]["end"] == len(unit["text"])


def test_exhausted_refinement_budget_cannot_finish_with_stale_result(tmp_path):
    workflow, client = make_workflow(tmp_path)
    workflow.settings = DivideSettings(max_extraction_attempts=1)
    unit = workflow.manifest["units"][0]
    extra_id = workflow.manifest["units"][-1]["id"]
    task = workflow._new_task({"spans": [{"coverage_id": unit["id"], "start": 0, "end": len(unit["text"])}]})

    def refine_after_budget(prompt, handlers, config):
        assert handlers["extract_chunk"](task_id=task["id"], extra_evidence_ids=[])["status"] == "extracted"
        outcome = handlers["extract_chunk"](task_id=task["id"], extra_evidence_ids=[extra_id])
        assert outcome["status"] == "needs_split" and outcome["reason"] == "attempts_exhausted"
        with pytest.raises(ValueError, match="no complete extraction"):
            handlers["finish"]()
        handlers["split_task"](task_id=task["id"])
        handlers["finish"]()
        return {"status": "completed", "native_final_event": True, "synthetic": True}

    workflow.react_runner = refine_after_budget
    children = workflow._run_task(task)
    assert len(client.calls) == 1  # Budget rejection is not a model call.
    assert len(children) == 2 and task["status"] == "split"
    assert workflow.reducer.provenance()["基本信息"] == {}


@pytest.mark.parametrize("failure", ["transport", "invalid_json", "invalid_resolution"])
def test_review_failures_consume_budget_and_never_publish_a_report(tmp_path, failure):
    workflow, client = make_workflow(tmp_path, "桥梁名称：合成桥\n报告日期：2018-12-16\n报告日期：2018-12-20\n")
    original_generate = client.generate
    review_calls = []

    def failing_review(messages, max_tokens=None):
        payload = json.loads(messages[-1]["content"])
        if "conflict" not in payload:
            return original_generate(messages, max_tokens=max_tokens)
        review_calls.append(payload)
        # Reservation must already be on disk while inference is in progress.
        saved = json.loads((tmp_path / "execution.json").read_text())
        assert len(saved["reviews"]) == 1
        current = saved["reviews"][0]
        assert not current["resolved"]
        assert len(current["attempts"]) == len(review_calls)
        assert current["attempts"][-1]["status"] == "failed"
        if failure == "transport":
            raise RuntimeError("synthetic transport failure")
        text = "invalid JSON" if failure == "invalid_json" else json.dumps({
            "value": "invented date outside candidates", "evidence_ids": [payload["evidence"][0]["id"]]})
        result = {"text": text, "call_id": f"synthetic-review-{len(review_calls)}",
                  "status": "success", "finish_reason": "stop"}
        client.calls.append({**result, "kind": "generation", "synthetic": True})
        return result

    def retry_failed_review(prompt, handlers, config):
        active = json.loads(prompt.rsplit("\n", 1)[1])
        if "task_id" in active:
            return fake_react(prompt, handlers, config)
        for _ in range(workflow.settings.max_extraction_attempts):
            with pytest.raises((RuntimeError, ValueError)):
                handlers["review_conflict"](conflict_id=active["id"], extra_evidence_ids=[])
            with pytest.raises(ValueError, match="Conflict not resolved"):
                handlers["finish"]()
        with pytest.raises(ValueError, match="Review attempts exhausted"):
            handlers["review_conflict"](conflict_id=active["id"], extra_evidence_ids=[])
        return {"status": "failed", "synthetic": True}

    client.generate = failing_review
    workflow.react_runner = retry_failed_review
    with pytest.raises(RuntimeError, match="SDK conflict review did not finish"):
        workflow.run()
    assert len(review_calls) == workflow.settings.max_extraction_attempts == 2
    assert not (tmp_path / "report.json").exists()
    receipt = json.loads((tmp_path / "execution.json").read_text())
    assert receipt["status"] == "failed" and len(receipt["reviews"]) == 1
    review = receipt["reviews"][0]
    assert not review["resolved"] and len(review["attempts"]) == 2
    assert all(attempt["status"] == "failed" and attempt["error"] for attempt in review["attempts"])
    assert bool(workflow.reducer.conflicts())


def test_interrupted_review_retains_unresolved_attempt_checkpoint(tmp_path):
    workflow, client = make_workflow(tmp_path, "桥梁名称：合成桥\n报告日期：2018-12-16\n报告日期：2018-12-20\n")
    original_generate = client.generate

    def interrupted_review(messages, max_tokens=None):
        if "conflict" in json.loads(messages[-1]["content"]):
            raise KeyboardInterrupt("synthetic interrupt")
        return original_generate(messages, max_tokens=max_tokens)

    client.generate = interrupted_review
    with pytest.raises(KeyboardInterrupt, match="synthetic interrupt"):
        workflow.run()
    assert not (tmp_path / "report.json").exists()
    receipt = json.loads((tmp_path / "execution.json").read_text())
    assert len(receipt["reviews"]) == 1
    review = receipt["reviews"][0]
    assert not review["resolved"] and len(review["attempts"]) == 1
    attempt = review["attempts"][0]
    assert attempt["status"] == "failed" and "KeyboardInterrupt" in attempt["error"]
    assert "call_id" not in attempt  # An exception is not a fabricated model response.


def test_oversized_outline_returns_small_notice_without_heading_body(tmp_path):
    workflow, client = make_workflow(tmp_path, "长标题正文" * 12000)
    workflow.manifest["units"][0]["style_id"] = "Heading1"
    result = workflow._read_handlers()["read_outline"]()
    assert result["status"] == "outline_too_large"
    assert result["unit_count"] == 1 and result["table_count"] == 0
    assert "headings" not in result and "tables" not in result
    serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    assert "长标题正文" not in serialized
    assert client.count_tokens(serialized) < workflow.settings.max_tool_tokens
    assert client.calls == []  # A bounded directory notice is not inference.
