"""Short review acknowledgements retain exact scopes without echoing them.

Synthetic model replies use the local audited gateway and native protocol
driver; these tests do not establish SDK integration or model accuracy.
"""
from copy import deepcopy
import json

import pytest
import test_document_workflow as workflow_fixtures

from bridge_evidence.contest import document_workflow as module
from bridge_evidence.contest.document_review_checkpoint import NativeReviewCheckpoint
from bridge_evidence.contest.run_state import json_digest
from test_document_workflow import FakeModel, workflow, read, prepare


class AckModel(FakeModel):
    def __init__(self, mutate=None):
        super().__init__()
        self.mutate = mutate
        self.wire_decisions = []

    def generate(self, messages, *, max_tokens):
        # The assigned task stays in message 1 while repair/tool messages may
        # follow it. The basic double otherwise assumes the last user message
        # is always the complete task.
        response = super().generate(messages[:2], max_tokens=max_tokens)
        payload = self.payloads[-1]
        decision = json.loads(response["text"])
        decision.pop("checked_output_paths")
        decision.pop("checked_evidence_spans")
        decision.update(checked_assignment_id=payload["review_assignment"]["id"], checked_all_assigned=True)
        if self.mutate:
            self.mutate(decision, payload)
        self.wire_decisions.append(deepcopy(decision))
        response["text"] = json.dumps(decision, ensure_ascii=False)
        return response


def seeded(tmp_path, *, client=None, text=None):
    options = {"client": client or AckModel()}
    if text is not None:
        options["text"] = text
    wf = workflow(tmp_path, **options)
    read(wf)
    prepare(wf)
    return wf


def scope(wf, direction):
    if direction == "FORWARD":
        return {"direction": direction, "output_items": wf._output_items(wf.workspace.snapshot()), "source_spans": []}
    spans = [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}
             for unit in wf.manifest["units"] if unit["text"]]
    return {"direction": direction, "output_items": [], "related_output": [],
            "source_spans": spans, "source": "\n".join(unit["text"] for unit in wf.manifest["units"]),
            "document_background": {"桥梁名称": "合成桥"}}


def review_request(wf, payload):
    # Exercise the native review substage with the same frozen catalog that
    # _perform_review_draft owns in a complete review. Never fabricate receipts.
    wf._active_review_items = wf._output_items(wf.workspace.snapshot())
    return wf._review_request(payload)


@pytest.mark.parametrize("direction", ["FORWARD", "REVERSE"])
def test_compact_ack_keeps_program_scopes_separate_from_model_reply(tmp_path, direction):
    wf = seeded(tmp_path)
    payload = scope(wf, direction)
    result = review_request(wf, payload)
    assert result["passed"] is True
    assert result["assignment_confirmation"] == "compact"
    assert "checked_output_paths" not in result
    assert "checked_evidence_spans" not in result
    assert result["assigned_output_paths"] == [item["path"] for item in payload["output_items"]]
    assert result["assigned_evidence_spans"] == payload["source_spans"]
    assert result["review_assignment"]["payload_sha256"] == json_digest(payload)
    assert result["review_assignment"]["id"] == result["checked_assignment_id"]
    assert wf.client.payloads[-1]["source_spans"] == payload["source_spans"]
    assert "review_assignment" not in payload


def test_full_review_can_journal_and_finish_with_compact_decisions(tmp_path):
    wf = seeded(tmp_path)
    assert wf._perform_review_draft()["delivery"]["ready"]
    assert wf._finish()["ready"]
    assert {payload["direction"] for payload in wf.client.payloads} == {"FORWARD", "REVERSE"}
    records = json.loads(wf.review_journal.path.read_text())["records"]
    assert records
    for record in records.values():
        assert record["decision"]["assignment_confirmation"] == "compact"
        assert "checked_evidence_spans" not in record["decision"]
        assignment = record["decision"]["review_assignment"]
        assert assignment["payload_sha256"] == record["payload_sha256"]
        assert len(record["decision"]["assigned_evidence_spans"]) == assignment["source_interval_count"]


@pytest.mark.parametrize("bad", [False, None, 1, "true"])
def test_incomplete_or_non_boolean_ack_cannot_pass(tmp_path, bad):
    wf = seeded(tmp_path, client=AckModel(lambda d, p: d.update(checked_all_assigned=bad)))
    with pytest.raises(ValueError, match="current complete assignment"):
        review_request(wf, scope(wf, "REVERSE"))
    assert wf.workspace.snapshot()["review"] is None


@pytest.mark.parametrize("omit", ["checked_assignment_id", "checked_all_assigned"])
def test_partial_ack_is_rejected(tmp_path, omit):
    wf = seeded(tmp_path, client=AckModel(lambda d, p: d.pop(omit)))
    with pytest.raises(ValueError, match="current complete assignment"):
        review_request(wf, scope(wf, "REVERSE"))


def test_wrong_assignment_is_rejected(tmp_path):
    wf = seeded(tmp_path, client=AckModel(lambda d, p: d.update(checked_assignment_id="ra_" + "0" * 32)))
    with pytest.raises(ValueError, match="current complete assignment"):
        review_request(wf, scope(wf, "REVERSE"))


def test_stale_assignment_after_a_semantic_edit_is_rejected(tmp_path):
    client = AckModel()
    wf = seeded(tmp_path, client=client)
    payload = scope(wf, "REVERSE")
    old_request, _ = wf._prepare_review_request(payload)
    old_id = json.loads(old_request["messages"][1]["content"])["review_assignment"]["id"]
    wf.workspace.apply_update({"base_version": wf.workspace.snapshot()["version"],
        "questions": [{"id": "q", "question": "新增待核事项", "status": "open"}]})
    client.mutate = lambda decision, payload: decision.update(checked_assignment_id=old_id)
    with pytest.raises(ValueError, match="current complete assignment"):
        review_request(wf, payload)


def test_assignment_binds_payload_workflow_semantics_and_frozen_catalog(tmp_path):
    wf = seeded(tmp_path)
    payload = scope(wf, "REVERSE")
    initial = wf._review_assignment(payload, "catalog-a")
    assert initial == wf._review_assignment(deepcopy(payload), "catalog-a")
    changed_payload = {**payload, "source": payload["source"] + "另一段上下文"}
    assert initial["id"] != wf._review_assignment(changed_payload, "catalog-a")["id"]
    assert initial["id"] != wf._review_assignment(payload, "catalog-b")["id"]
    wf.workspace.record_read(wf.manifest["units"][0]["id"], 0, 1)
    assert initial == wf._review_assignment(payload, "catalog-a")
    state = wf.workspace.snapshot()
    wf.workspace.apply_update({"base_version": state["version"],
        "questions": [{"id": "q", "question": "构件指代是否正确", "status": "open"}]})
    assert initial["id"] != wf._review_assignment(payload, "catalog-a")["id"]
    current = wf._review_assignment(payload, "catalog-a")
    wf.identity["run_identity"]["experiment"] = "different-run"
    assert current["id"] != wf._review_assignment(payload, "catalog-a")["id"]


@pytest.mark.parametrize("direction", ["FORWARD", "REVERSE"])
def test_matching_legacy_echo_can_accompany_ack_but_conflict_cannot(tmp_path, direction):
    def include_legacy(decision, payload):
        decision.update(checked_output_paths=[i["path"] for i in payload["output_items"]],
                        checked_evidence_spans=deepcopy(payload["source_spans"]))
    good = seeded(tmp_path / "good", client=AckModel(include_legacy))
    assert review_request(good, scope(good, direction))["passed"]

    def conflict(decision, payload):
        include_legacy(decision, payload)
        if direction == "FORWARD":
            decision["checked_output_paths"] = []
        else:
            decision["checked_evidence_spans"] = []
    bad = seeded(tmp_path / "bad", client=AckModel(conflict))
    with pytest.raises(ValueError, match="account for every assigned"):
        review_request(bad, scope(bad, direction))


@pytest.mark.parametrize("category", ["unsupported_claims", "missing_items", "conflicts", "unresolved_items"])
def test_compact_ack_never_erases_findings_or_permits_pass_with_issues(tmp_path, category):
    issue = {"evidence_id": "E1", "reason": "尚无足够依据", "source_refs": []}
    wf = seeded(tmp_path / "bad", client=AckModel(lambda d, p: d.update({category: [issue]})))
    with pytest.raises(ValueError, match="cannot pass while issues remain"):
        review_request(wf, scope(wf, "REVERSE"))
    wf = seeded(tmp_path / "issues", client=AckModel(lambda d, p: d.update({"passed": False, category: [issue]})))
    result = review_request(wf, scope(wf, "REVERSE"))
    assert result[category] == [issue]
    assert not result["passed"]


def test_hundreds_of_source_intervals_do_not_need_to_fit_in_model_reply(tmp_path):
    wf = seeded(tmp_path, text="桥梁名称：合成桥\n" + "正文记录某构件。\n" * 500)
    payload = scope(wf, "REVERSE")
    result = review_request(wf, payload)
    wire = wf.client.wire_decisions[-1]
    assert len(payload["source_spans"]) >= 500
    assert result["assigned_evidence_spans"] == payload["source_spans"]
    assert len(json.dumps(wire, ensure_ascii=False)) < 300
    assert "checked_evidence_spans" not in wire
    assert wf.client.payloads[-1]["source"] == payload["source"]


def test_returned_compact_response_recovers_without_another_model_call(tmp_path, monkeypatch):
    wf = seeded(tmp_path)
    payload = scope(wf, "REVERSE")
    # The native stage must have persisted both its returned SDK boundary and
    # actual HTTP receipts before interruption. A merely started checkpoint
    # cannot authorize replay, even if a response happened to reach the client.
    def interrupt_before_decision(*args, **kwargs):
        raise RuntimeError("synthetic interrupted after response")
    with monkeypatch.context() as patch:
        patch.setattr(wf, "_native_final_decision", interrupt_before_decision)
        with pytest.raises(RuntimeError, match="after response"):
            review_request(wf, payload)
    assert len(wf.client.calls) == 1
    checkpoint = wf._partial_checkpoint(payload, json_digest(wf._active_review_items))
    assert isinstance(checkpoint, NativeReviewCheckpoint)
    state = checkpoint.load()
    assert state["status"] == "returned" and state["iterations_spent"] == 1
    assert state["model_call_ids"] == [wf.client.calls[0]["call_id"]]
    assert wf.review_sessions[state["session_index"]]["outcome"]["native_final_event"] is True
    restored = workflow(tmp_path, client=AckModel())
    result = review_request(restored, payload)
    assert result["assignment_confirmation"] == "compact"
    assert result["checked_assignment_id"] == wf.client.wire_decisions[0]["checked_assignment_id"]
    assert restored.client.calls == []


def test_rejected_native_ack_stays_bound_after_resume_without_replacement_or_reset(tmp_path):
    client = AckModel(lambda d, p: d.update(checked_assignment_id="ra_" + "0" * 32))
    wf = seeded(tmp_path, client=client)
    payload = scope(wf, "REVERSE")
    with pytest.raises(ValueError, match="current complete assignment"):
        review_request(wf, payload)
    partial = json.loads(next((tmp_path / "semantic_review" / "partial").glob("*.json")).read_text())["state"]
    assert partial["status"] == "returned"
    assert partial["iterations_spent"] == 1
    assert partial["model_call_ids"] == [client.calls[0]["call_id"]]
    original_session = deepcopy(wf.review_sessions[partial["session_index"]])
    # A new client's potentially corrected answer must not overwrite a durable
    # rejected native final or silently reset its original cost.
    restored = workflow(tmp_path, client=AckModel())
    with pytest.raises(ValueError, match="current complete assignment"):
        review_request(restored, payload)
    assert len(client.calls) == 1
    assert restored.client.calls == []
    assert restored.review_sessions[partial["session_index"]] == original_session
    assert restored.workspace.snapshot()["review"] is None
    current = json.loads(next((tmp_path / "semantic_review" / "partial").glob("*.json")).read_text())["state"]
    assert current == partial


def test_missing_native_finals_do_not_reset_per_review_iteration_budget(tmp_path, monkeypatch):
    client = AckModel()
    settings = module.DocumentSettings(review_iterations=2)
    wf = workflow(tmp_path, client=client, settings=settings)
    read(wf)
    prepare(wf)
    payload = scope(wf, "REVERSE")
    native_runner = workflow_fixtures.native_review_driver
    def missing_native_final(prompt, handlers, config):
        # Audited synthetic transport succeeds, but the SDK has no final
        # event. It is a bounded known response, never an accepted verdict.
        outcome = native_runner(prompt, handlers, config)
        return {**outcome, "status": "failed", "native_final_event": False,
                "failure_reason": "missing_or_empty_native_final_response"}
    monkeypatch.setattr(workflow_fixtures, "native_review_driver", missing_native_final)
    with pytest.raises(RuntimeError, match="iteration budget exhausted"):
        review_request(wf, payload)
    assert len(client.calls) == 2
    partial = json.loads(next((tmp_path / "semantic_review" / "partial").glob("*.json")).read_text())["state"]
    assert partial["iterations_spent"] == 2
    assert partial["status"] == "returned"
    assert partial["model_call_ids"] == [call["call_id"] for call in client.calls]
    assert wf.workspace.snapshot()["review"] is None
    restored = workflow(tmp_path, client=AckModel(), settings=settings)
    with pytest.raises(RuntimeError, match="iteration budget exhausted"):
        review_request(restored, payload)
    assert len(client.calls) == 2
    assert restored.client.calls == []
