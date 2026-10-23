"""Synthetic integration checks for persisted semantic-review recovery.

The actual workflow, journal and audited gateway run against an in-memory
transport double. No SDK, HTTP listener, model or report data is used.
"""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from bridge_evidence.contest import document_workflow as module
from test_document_workflow import FakeModel, prepare, read, source_fact, workflow


class InterruptibleModel(FakeModel):
    """Keep transport attempts distinct from cached decisions and process IDs."""

    def __init__(self, *, namespace="first", fail_on=None):
        super().__init__()
        self.namespace = namespace
        self.fail_on = fail_on
        self.requests = []
        self.wire_requests = []

    def generate(self, messages, *, max_tokens):
        response = super().generate(messages, max_tokens=max_tokens)
        identifier = f"{self.namespace}:{response['call_id']}"
        response["call_id"] = self.calls[-1]["call_id"] = identifier
        return response

    def _request(self, path, payload, *, kind):
        prepared = json.loads(payload["messages"][1]["content"])
        self.wire_requests.append(deepcopy(prepared))
        # Journal keys bind the original semantic payload. Protocol assignment
        # and catalog presentation are added to its deep copy for transport.
        prepared.pop("review_assignment", None)
        prepared.pop("draft_catalog", None)
        prepared.pop("source_catalog", None)
        self.requests.append(prepared)
        if len(self.requests) == self.fail_on:
            # A failed transport attempt consumes a real request, but yields no
            # semantic decision. Match the real client's failed-call accounting.
            self.calls.append({"kind": kind, "status": "failed",
                               "call_id": f"{self.namespace}:failed-{len(self.requests)}"})
            raise ConnectionError("synthetic network interruption")
        return super()._request(path, payload, kind=kind)


def settings(*, attempts=1, tool_tokens=4096):
    return module.DocumentSettings(max_sessions=30, max_model_calls=30,
                                   max_review_attempts=attempts, tool_tokens=tool_tokens)


def interrupted_review(wf, *, initialize=True):
    if initialize:
        read(wf)
        prepare(wf)
    semantic_hash = wf._semantic_identity(wf.workspace.snapshot())
    original = wf._review_request

    def stop_before_admission(payload):
        if payload["direction"] == "REVERSE":
            raise RuntimeError("synthetic interruption before reviewer admission")
        return original(payload)

    # A journaled decision has a complete returned SDK boundary. Interrupt
    # before admitting the next session; there is no unknown request to retry.
    wf._review_request = stop_before_admission
    try:
        with pytest.raises(RuntimeError, match="before reviewer admission"):
            wf._perform_review_draft()
    finally:
        wf._review_request = original
    assert [p["direction"] for p in wf.client.requests] == ["FORWARD"]
    first_payload = wf.client.requests[0]
    first_decision = wf.review_journal.get(semantic_hash, first_payload)
    assert first_decision["model_call_id"] == wf.client.calls[0]["call_id"]
    persisted = json.loads(wf.review_journal.path.read_text(encoding="utf-8"))
    assert len(persisted["records"]) == 1
    assert wf.review_attempts == wf.review_journal.attempt_count == 1
    assert wf.review_journal.active(semantic_hash) is not None
    assert wf.workspace.snapshot()["review"] is None
    return semantic_hash, first_decision


def assert_completed_accounting(wf, original_decision):
    receipt = wf._receipt("completed", "navigation")
    # Resuming the not-yet-admitted reverse session must not regenerate the
    # already completed forward verdict or fabricate an interrupted call.
    assert receipt["review_generation_requests"] == 2
    assert len(receipt["model_calls"]) == 2
    assert len({call["call_id"] for call in receipt["model_calls"]}) == 2
    assert [call["status"] for call in receipt["model_calls"]] == ["success", "success"]
    assert len(receipt["gateway_receipts"]) == 2
    review = wf.workspace.snapshot()["review"]
    assert review["passed"]
    assert len(review["model_call_ids"]) == 2
    assert review["model_call_ids"][0] == original_decision["model_call_id"]
    assert wf.nodes[-1]["reused_model_call_ids"] == [original_decision["model_call_id"]]
    assert wf.review_attempts == wf.review_journal.attempt_count == 1
    assert receipt["delivery"]["ready"]


def test_interrupted_review_starts_only_unadmitted_batch_in_same_attempt(tmp_path):
    client = InterruptibleModel()
    wf = workflow(tmp_path, client=client, settings=settings())
    semantic_hash, original_decision = interrupted_review(wf)
    attempt_id = wf.review_journal.active(semantic_hash)["attempt_id"]

    result = wf._perform_review_draft()

    assert result["delivery"]["ready"]
    assert [p["direction"] for p in client.requests] == ["FORWARD", "REVERSE"]
    assert wf.nodes[-1]["review_attempt_id"] == attempt_id
    assert wf.review_journal.active(semantic_hash) is None
    assert_completed_accounting(wf, original_decision)


def test_reconstructed_workflow_reuses_persisted_decision_and_request_counts(tmp_path):
    first = workflow(tmp_path, client=InterruptibleModel(), settings=settings())
    semantic_hash, original_decision = interrupted_review(first)
    attempt_id = first.review_journal.active(semantic_hash)["attempt_id"]
    next_client = InterruptibleModel(namespace="restored")
    resumed = workflow(tmp_path, client=next_client, settings=settings())

    assert resumed.review_journal.get(semantic_hash, first.client.requests[0]) == original_decision
    result = resumed._perform_review_draft()

    assert result["delivery"]["ready"]
    assert [p["direction"] for p in next_client.requests] == ["REVERSE"]
    assert resumed.nodes[-1]["review_attempt_id"] == attempt_id
    assert_completed_accounting(resumed, original_decision)


@pytest.mark.parametrize("reconstruct", [False, True], ids=["same-process", "reconstructed"])
def test_unknown_native_transport_failure_preserves_forward_verdict_and_blocks_retry(tmp_path, reconstruct):
    first = workflow(tmp_path, client=InterruptibleModel(fail_on=2), settings=settings())
    read(first)
    prepare(first)
    semantic_hash = first._semantic_identity(first.workspace.snapshot())
    with pytest.raises(RuntimeError, match="Unsettled or unknown model request"):
        first._perform_review_draft()
    original = first.review_journal.get(semantic_hash, first.client.requests[0])
    assert original is not None
    calls = deepcopy(first._model_calls())
    assert [call["status"] for call in calls] == ["success", "failed"]
    current = (workflow(tmp_path, client=InterruptibleModel(namespace="restored"), settings=settings())
               if reconstruct else first)

    with pytest.raises(RuntimeError, match="Unsettled or unknown model request"):
        current._perform_review_draft()

    assert current._model_calls() == calls
    assert current.review_journal.get(semantic_hash, first.client.requests[0]) == original
    assert current.review_journal.active(semantic_hash) is not None
    assert current.workspace.snapshot()["review"] is None
    assert not current._receipt("failed", "navigation")["delivery"]["ready"]


@pytest.mark.parametrize("changed", ["fact", "relation", "draft"])
def test_semantic_changes_invalidate_old_decisions_even_if_payload_is_unchanged(tmp_path, changed):
    client = InterruptibleModel()
    wf = workflow(tmp_path, client=client, settings=settings(attempts=2))
    read(wf)
    before = source_fact(wf, "before", 1, value="2018年裂缝宽0.10mm")
    after = source_fact(wf, "after", 1, value="2020年裂缝宽0.20mm")
    relation = {"id": "comparison", "subject_id": "after", "object_id": "before",
                "predicate": "compared_with", "status": "explicit", "depends_on": [],
                "source_refs": deepcopy(after["source_refs"])}
    # These working records are deliberately not bound to the name output.
    # Changing them leaves its forward payload byte-equivalent, so only the
    # semantic-state identity can prevent reusing the obsolete decision.
    prepare(wf, facts=[source_fact(wf), before, after], relations=[relation])
    original_hash = wf._semantic_identity(wf.workspace.snapshot())
    interrupted_review(wf, initialize=False)
    original_payload = client.requests[0]
    original_decision = wf.review_journal.get(original_hash, original_payload)
    assert original_decision is not None
    snapshot = wf.workspace.snapshot()
    update = {"base_version": snapshot["version"]}
    if changed == "fact":
        update.update(facts=[{**before, "value": "2018年裂缝宽0.15mm"}], relations=[relation])
    elif changed == "relation":
        update["relations"] = [{**relation, "predicate": "same_component_as"}]
    else:
        draft = deepcopy(snapshot["draft"])
        draft["基本信息"]["桥梁名称"] = "合成桥（原名）"
        update.update(draft=draft, draft_fact_ids=snapshot["draft_fact_ids"],
                      draft_bindings=snapshot["draft_bindings"])
    wf.workspace.apply_update(update)
    new_hash = wf._semantic_identity(wf.workspace.snapshot())
    assert new_hash != original_hash
    assert wf.review_journal.get(new_hash, original_payload) is None

    wf._perform_review_draft()

    assert [p["direction"] for p in client.requests[1:]] == ["FORWARD", "REVERSE"]
    if changed != "draft":
        assert client.requests[1] == original_payload
    assert client.wire_requests[1]["review_assignment"]["id"] != client.wire_requests[0]["review_assignment"]["id"]
    assert wf.nodes[-1].get("reused_model_call_ids", []) == []
    assert original_decision["model_call_id"] not in wf.workspace.snapshot()["review"]["model_call_ids"]
    assert wf.review_attempts == wf.review_journal.attempt_count == 2


def test_reverse_review_receives_full_cross_paragraph_sources_and_relation_context(tmp_path):
    history = "历史记录说明：" + "仅为合成背景。" * 450 + "2018年左幅1号梁裂缝宽0.10mm，历史尾部标识。"
    unrelated = "无任务相关信息的合成间隔。" * 300
    current = "2020年本次左幅同一1号梁裂缝宽0.20mm，与2018年相比增宽。"
    client = InterruptibleModel()
    wf = workflow(tmp_path, client=client, settings=settings(tool_tokens=512),
                  text=f"桥梁名称：合成桥\n{history}\n{unrelated}\n{current}\n")
    read(wf)
    before = source_fact(wf, "before", 1, value="2018年左幅1号梁裂缝宽0.10mm")
    after = source_fact(wf, "after", 3, value="2020年左幅1号梁裂缝宽0.20mm")
    relation = {"id": "comparison", "subject_id": "after", "object_id": "before",
                "predicate": "wider_than_previous", "status": "explicit", "depends_on": [],
                "source_refs": deepcopy(after["source_refs"])}
    prepare(wf, facts=[source_fact(wf), before, after], relations=[relation],
            draft={"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [],
                   "病害明细": [{"病害部位": "左幅1号梁", "病害描述": "裂缝宽0.20mm",
                              "上一次定检状态": "2018年宽0.10mm", "发展程度": "增宽"}]},
            draft_fact_ids=["name", "comparison"],
            draft_bindings={"基本信息.桥梁名称": ["name"], "病害明细.0": ["comparison"]})

    wf._perform_review_draft()

    history_unit, current_unit = wf.manifest["units"][1], wf.manifest["units"][3]
    payload = next(p for p in client.requests if p["direction"] == "REVERSE"
                   and any(s["evidence_id"] == current_unit["id"] for s in p["source_spans"]))
    assert all(s["evidence_id"] != history_unit["id"] for s in payload["source_spans"])
    output = next(item for item in payload["related_output"] if item["path"] == "病害明细.0")
    records = {payload["related_facts"][ref]["id"]: payload["related_facts"][ref]
               for ref in output["fact_refs"]}
    assert set(records) == {"before", "after", "comparison"}
    assert records["comparison"]["subject_id"] == "after"
    assert records["comparison"]["object_id"] == "before"
    assert records["comparison"]["predicate"] == "wider_than_previous"
    sources = [payload["related_evidence"][ref] for ref in output["source_refs"]]
    old_source = next(s for s in sources if s["evidence_id"] == history_unit["id"])
    assert old_source["text"] == history_unit["text"] == history + "\n"
    assert old_source["start"] == 0 and old_source["end"] == len(history_unit["text"])
    assert history_unit["text"] in old_source["rendered_context"]
    assert any(s["text"] == current_unit["text"] == current + "\n" for s in sources)
    assert wf.workspace.snapshot()["review"]["passed"]
