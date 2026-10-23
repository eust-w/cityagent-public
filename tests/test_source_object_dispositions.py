"""Synthetic contracts only: fake local receipts, no SDK/model/HTTP/real reports."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest import source_object_dispositions as mod
from bridge_evidence.contest.run_state import json_digest


def digest(value):
    return json_digest(value)


def source_ref(eid="E2", start=0, end=12):
    return {"evidence_id": eid, "start": start, "end": end}


@pytest.fixture
def env(tmp_path):
    class Env:
        identity = {"model": "SYNTHETIC-OWN-MODEL-NO-HTTP", "source_sha256": digest("synthetic source"),
                    "workflow_identity": {"code": "candidate-test", "weights": "synthetic"}}
        units = {"E1": "合成例：评分公式说明正文，含完整上下文。", "E2": "合成桥甲本年度评分为95分，等级A。", "E3": "合成桥乙上年度评分为72分，等级C。"}

        def __init__(self):
            self.receipts = {}
            self.occurrences = [dict(source_sha256=self.identity["source_sha256"], locator=f"word/document.xml/p[{i}]/ole[1]",
                                     kind="ole", raw_sha256=digest("identical bytes"), native_sha256=digest("native"),
                                     cache_sha256=digest("cached image"), context_refs=[source_ref("E1", 0, len(self.units["E1"]))])
                                for i in (1, 9)]
            self.ids = [mod.occurrence_id(o) for o in self.occurrences]
            self.directory = tmp_path
            self.book = self.reload()
            self.context = dict(task={"name": "three-table-extraction", "version": "synthetic-1", "unknown_behavior": "preserve"},
                                draft={"basic": {"score": 95, "history": None}}, facts={"score": "reported, not recomputed"},
                                output_paths=["basic.score", "basic.history"],
                                read_refs=[source_ref(eid, 0, len(text)) for eid, text in self.units.items()],
                                catalog_sha256=digest("synthetic frozen draft catalog"))
            self.book.bind_context(**self.context)

        def reload(self):
            return mod.SourceObjectDispositions(self.directory, self.identity, self.occurrences, self.units, self.receipts.__getitem__)

        def proposal(self, mode="not_relied_on"):
            return {"disposition": mode, "rationale": "Synthetic assertion for independent semantic judgment, not a keyword rule.",
                    "affected_output_paths": ["basic.score"], "supporting_refs": [source_ref("E2", 0, len(self.units["E2"]))],
                    "context_refs": [source_ref("E1", 0, len(self.units["E1"]))],
                    "source_limitations": ["Original formula semantics remain unverified; score copied only from text."],
                    "origin_call_ids": ["synthetic-agent-proposal-call"]}

        def propose(self, index=0, mode="not_relied_on", **changes):
            value = self.proposal(mode)
            value.update(changes)
            return self.book.propose(self.ids[index], value)

        def fake_receipt(self, assignment, decision, call_id="synthetic-independent-1"):
            # Fixture is deliberately synthetic. Production must resolve these
            # fields from real terminal gateway receipts and original sidecars.
            request = {"model": self.identity["model"], "stream": False, "max_tokens": 256,
                       "messages": [{"role": "system", "content": "Synthetic independent reviewer."},
                                    {"role": "user", "content": json.dumps(assignment, ensure_ascii=False)}]}
            response = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)}}],
                        "usage": {"prompt_tokens": 123, "completion_tokens": 33, "total_tokens": 156}}
            audit = {"call_id": call_id, "request_id": "synthetic-dispatch-"+call_id, "status": "success", "generation_attempted": True,
                     "response_validated": True, "http_status": 200, "finish_reason": "stop", "request_kind": "generation",
                     "endpoint": "/v1/chat/completions", "model": self.identity["model"],
                     "request_sha256": digest(request), "response_sha256": digest(response)}
            self.receipts[call_id] = dict(audit=audit, request=request, response=response)
            return call_id

        def reviewed(self, pids, verdict="accepted", *, call_id="synthetic-independent-1", issues=None):
            pids = [pids] if isinstance(pids, str) else pids
            assignment = self.book.review_payload(pids)
            decision = {"assignment_id": assignment["assignment_id"], "decisions": []}
            for row in assignment["proposals"]:
                item = dict(proposal_id=row["proposal_id"], object_id=row["object_id"], verdict=verdict,
                            explanation="Synthetic reviewer decision; no real model was invoked.", **{key: [] for key in mod.ISSUES})
                if issues:
                    item.update(issues)
                decision["decisions"].append(item)
            self.fake_receipt(assignment, decision, call_id)
            return call_id, decision

        def accept_all(self):
            pids = [self.propose(i) for i in range(2)]
            call, _ = self.reviewed(pids)
            self.book.record_review(pids, call)
            return pids

    return Env()


def gates(**changes):
    result = {"audit_valid": True, "draft_valid": True, "native_sdk_observed": True, "finish_passed": True,
              "export_verified": True, "task_review_passed": True, "partial_review_passed": False}
    result.update(changes)
    return result


def test_same_bytes_at_different_occurrences_never_share_disposition(env):
    assert env.ids[0] != env.ids[1]
    assert mod.occurrence_id(deepcopy(env.occurrences[0])) == env.ids[0]
    pid = env.propose()
    call, _ = env.reviewed(pid)
    env.book.record_review(pid, call)
    cover = env.book.coverage()
    assert cover["accepted_count"] == 1
    assert cover["blocking_count"] == 1
    assert cover["objects"][1]["review_status"] == "unclassified"
    assert env.book.completion(gates())["status"] == "incomplete"


def test_unread_formula_with_independent_support_can_complete_task_not_source_semantics(env):
    env.accept_all()
    outcome = env.book.completion(gates())
    assert outcome["status"] == "completed"
    assert outcome["task_complete"] is True
    assert outcome["all_source_semantics_verified"] is False
    assert outcome["coverage"]["unverified_representation_count"] == 2
    assert len(outcome["source_limitations"]) == 2
    assert env.book.context["draft"]["basic"]["history"] is None
    assert env.book.coverage() == env.reload().coverage()


def test_batch_payload_retains_complete_original_support_task_draft_and_uncertainty(env):
    pids = [env.propose(i) for i in range(2)]
    payload = env.book.review_payload(pids)
    assert payload["draft"] == env.context["draft"]
    assert payload["task"] == env.context["task"]
    assert payload["facts"] == env.context["facts"]
    assert payload["proposals"][0]["source"][1]["text"] == env.units["E2"]
    assert payload["proposals"][0]["representation"]["state"] == "unread"
    payload["draft"]["basic"]["score"] = "corrupt copy"
    assert env.book.review_payload(pids)["draft"] == env.context["draft"]


@pytest.mark.parametrize("mode", ["used_as_evidence"])
def test_unread_or_candidate_cannot_be_used_as_evidence(env, mode):
    with pytest.raises(ValueError, match="verified semantics"):
        env.propose(mode=mode)
    representation = mod._unread()
    representation.update(state="candidate", content={"ast": "unverified symbol ?"}, semantic_uncertainties=["operator ?"])
    env.book.set_representation(env.ids[0], representation)
    with pytest.raises(ValueError, match="verified semantics"):
        env.propose(mode=mode)
    assert env.book.coverage()["accepted_count"] == 0


def test_semantics_verified_requires_independent_real_bound_receipt_and_retains_glyph_uncertainty(env):
    representation = mod._unread()
    representation.update(state="semantics_verified", content={"variable": "MDP", "subscript": "h"},
                          inspection={"original_bytes_sha256": env.occurrences[0]["raw_sha256"], "comparison": "SYNTHETIC complete character comparison"},
                          origin_call_ids=["synthetic-inspection-call"])
    with pytest.raises(ValueError, match="model call ID"):
        env.book.set_representation(env.ids[0], representation)
    assignment = env.book.representation_review_payload(env.ids[0], representation)
    decision = dict(assignment_id=assignment["assignment_id"], object_id=env.ids[0],
                    representation_sha256=digest(representation), semantics_verified=True,
                    explanation="Synthetic semantic verification; glyph metrics not certified.")
    call = env.fake_receipt(assignment, decision, "synthetic-representation-review")
    env.book.set_representation(env.ids[0], representation, verification_call_id=call)
    pid = env.propose(mode="used_as_evidence")
    assert env.book.review_payload(pid)["proposals"][0]["representation"]["glyph_fidelity"] == "unverified"
    assert env.reload().representations[env.ids[0]] == representation


def test_unknown_required_operator_blocks_semantics_even_when_model_record_is_offered(env):
    value = mod._unread()
    value.update(state="semantics_verified", content="a ? b", semantic_uncertainties=["Unknown operator affects result"],
                 inspection={"synthetic": "raw observation"})
    with pytest.raises(ValueError, match="unresolved semantic"):
        env.book.set_representation(env.ids[0], value, verification_call_id="not-issued")


@pytest.mark.parametrize("field", ["draft", "facts", "task", "catalog_sha256", "read_refs"])
def test_context_identity_changes_invalidate_old_acceptances_and_keep_history(env, field):
    pids = env.accept_all()
    old_events = env.book.history()
    changed = deepcopy(env.context)
    if field in {"draft", "facts", "task"}:
        changed[field]["changed"] = True
    elif field == "catalog_sha256":
        changed[field] = digest("different frozen catalog")
    else:
        changed[field] = changed[field][:-1]
    env.book.bind_context(**changed)
    assert env.book.coverage()["accepted_count"] == 0
    assert env.book.history()[:len(old_events)] == old_events
    with pytest.raises(ValueError, match="stale"):
        env.book.record_review(pids, "synthetic-independent-1")
    env.book.bind_context(**env.context)
    assert env.book.coverage()["accepted_count"] == 0  # A->B->A cannot resurrect earlier approval.
    assert env.reload().coverage() == env.book.coverage()


def test_representation_change_invalidates_dispositions_without_deleting_native_unknown(env):
    env.accept_all()
    before = env.book.history()
    candidate = mod._unread()
    candidate.update(state="candidate", content={"mtef_record": ["unknown glyph"]}, semantic_uncertainties=["private glyph"])
    env.book.set_representation(env.ids[0], candidate)
    assert env.book.coverage()["accepted_count"] == 0
    assert env.book.history()[:len(before)] == before
    assert env.book.representations[env.ids[0]]["semantic_uncertainties"] == ["private glyph"]


@pytest.mark.parametrize("field,value", [("supporting_refs", []), ("affected_output_paths", []), ("source_limitations", []), ("context_refs", [])])
def test_non_reliance_requires_independent_read_support_paths_context_and_visible_limitations(env, field, value):
    with pytest.raises(ValueError):
        env.propose(**{field: value})


def test_unread_ref_is_rejected_but_adjacent_delivered_pages_can_support_full_ref(env):
    env.context["read_refs"] = [source_ref("E1", 0, len(env.units["E1"])), source_ref("E2", 0, 4)]
    env.book.bind_context(**env.context)
    with pytest.raises(ValueError, match="actually been read"):
        env.propose()
    env.context["read_refs"].append(source_ref("E2", 4, len(env.units["E2"])))
    env.book.bind_context(**env.context)
    env.propose()


@pytest.mark.parametrize("ref", [source_ref("E2", True, 5), source_ref("E2", 0, 999), source_ref("missing", 0, 2), {"image_id": "cache-image"}])
def test_invalid_or_cached_image_refs_cannot_stand_in_for_independent_text(env, ref):
    with pytest.raises(ValueError):
        env.propose(supporting_refs=[ref])


def test_proposal_alone_never_auto_accepts_even_if_rationale_contains_irrelevant_keywords(env):
    env.propose(mode="outside_task", rationale="通用公式 无关 不需要 难读 忽略", supporting_refs=[], affected_output_paths=[])
    assert env.book.coverage()["objects"][0]["review_status"] == "proposed"
    assert not env.book.coverage()["object_gate_passed"]


def test_semantic_same_bridge_wrong_year_is_rejected_by_independent_decision_not_ref_rules(env):
    pid = env.propose(supporting_refs=[source_ref("E3", 0, len(env.units["E3"]))])
    call, _ = env.reviewed(pid, verdict="rejected", issues={"conflicts": ["Synthetic reviewer identified different bridge/year."]})
    env.book.record_review(pid, call)
    assert env.book.coverage()["objects"][0]["review_status"] == "rejected"
    assert not env.book.coverage()["object_gate_passed"]


def test_unresolved_relevant_cannot_complete_but_independently_reviewed_partial_can_be_limited(env):
    pids = [env.propose(0, mode="unresolved_relevant"), env.propose(1)]
    call, _ = env.reviewed(pids, verdict="unresolved", issues={"unresolved_items": ["Synthetic unique score in unread formula"]})
    env.book.record_review(pids, call)
    assert env.book.completion(gates())["status"] == "incomplete"
    limited = env.book.completion(gates(task_review_passed=False, partial_review_passed=True))
    assert limited["status"] == "completed_limited"
    assert not limited["task_complete"]
    assert not limited["formal_publish_eligible"]
    assert env.book.completion(gates(audit_valid=False, partial_review_passed=True))["status"] == "failed"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_object", "wrong_assignment", "accepted_with_issues"])
def test_exact_review_assignment_coverage_is_required(env, mutation):
    pids = [env.propose(i) for i in range(2)]
    call, decision = env.reviewed(pids)
    if mutation == "missing":
        decision["decisions"].pop()
    elif mutation == "duplicate":
        decision["decisions"][1] = deepcopy(decision["decisions"][0])
    elif mutation == "wrong_object":
        decision["decisions"][0]["object_id"] = env.ids[1]
    elif mutation == "wrong_assignment":
        decision["assignment_id"] = "sd_wrong"
    else:
        decision["decisions"][0]["missing_items"] = ["needed content remains missing"]
    env.fake_receipt(env.book.review_payload(pids), decision, call)
    with pytest.raises(ValueError):
        env.book.record_review(pids, call)
    assert env.book.coverage()["accepted_count"] == 0


@pytest.mark.parametrize("mutation", ["not_found", "pending", "tokenize", "wrong_model", "truncated", "request_sha", "response_sha", "missing_generation", "reused_proposal_call"])
def test_call_id_alone_or_invalid_terminal_ledger_cannot_approve(env, mutation):
    pid = env.propose()
    call, _ = env.reviewed(pid)
    audit = env.receipts[call]["audit"]
    if mutation == "not_found":
        del env.receipts[call]
    elif mutation == "pending":
        audit["status"] = "response_validated"
    elif mutation == "tokenize":
        audit["request_kind"] = "tokenize"
    elif mutation == "wrong_model":
        audit["model"] = "another model"
    elif mutation == "truncated":
        audit["finish_reason"] = "length"
    elif mutation.endswith("_sha"):
        audit[mutation.replace("_sha", "_sha256")] = "0" * 64
    elif mutation == "missing_generation":
        audit["generation_attempted"] = False
    else:
        env.receipts["synthetic-agent-proposal-call"] = env.receipts[call]
        call = "synthetic-agent-proposal-call"
        audit["call_id"] = call
    with pytest.raises(ValueError):
        env.book.record_review(pid, call)
    assert not env.book.coverage()["object_gate_passed"]


def test_even_valid_hashes_cannot_bind_different_assignment_request(env):
    pid = env.propose()
    call, _ = env.reviewed(pid)
    request = env.receipts[call]["request"]
    request["messages"][-1]["content"] = json.dumps({"assignment_id": "not the actual full assignment"})
    env.receipts[call]["audit"]["request_sha256"] = digest(request)
    with pytest.raises(ValueError, match="complete frozen assignment"):
        env.book.record_review(pid, call)


def test_existing_decision_cannot_be_overwritten_reproposal_keeps_rejection(env):
    pid = env.propose()
    call, _ = env.reviewed(pid, verdict="rejected")
    env.book.record_review(pid, call)
    call2, _ = env.reviewed(pid, call_id="synthetic-independent-2")
    with pytest.raises(ValueError, match="immutable"):
        env.book.record_review(pid, call2)
    new_pid = env.propose(rationale="Additional complete context now supplied to reviewer.")
    new_call, _ = env.reviewed(new_pid, call_id="synthetic-independent-3")
    env.book.record_review(new_pid, new_call)
    reviews = [e for e in env.book.history() if e["kind"] == "review"]
    assert [e["payload"]["decision"]["decisions"][0]["verdict"] for e in reviews] == ["rejected", "accepted"]
    with pytest.raises(ValueError, match="superseded"):
        env.book.review_payload(pid)


def test_repeated_same_record_is_idempotent_not_an_invented_new_call(env):
    pid = env.propose()
    call, result = env.reviewed(pid)
    env.book.record_review(pid, call)
    old = env.book.history()
    assert env.book.record_review(pid, call) == result
    assert env.book.history() == old
    assert len(env.receipts) == 1


def test_atomic_persistence_failure_does_not_expose_uncommitted_acceptance(env):
    pid = env.propose()
    call, _ = env.reviewed(pid)
    before = env.book.history()
    with patch.object(mod, "atomic_json", side_effect=OSError("synthetic disk full")):
        with pytest.raises(OSError):
            env.book.record_review(pid, call)
    assert env.book.history() == before
    assert env.book.coverage()["accepted_count"] == 0
    assert env.reload().coverage()["accepted_count"] == 0


@pytest.mark.parametrize("change", ["source", "occurrence_location", "units", "schema", "event", "delete_event"])
def test_restore_rejects_identity_or_history_tampering(env, change):
    env.accept_all()
    if change == "source":
        env.identity = {**env.identity, "source_sha256": digest("different source")}
    elif change == "occurrence_location":
        env.occurrences[0]["locator"] += "-moved"
    elif change == "units":
        env.units = {**env.units, "E1": "changed text"}
    else:
        state = json.loads(env.book.path.read_text())
        if change == "schema":
            state["header"]["schema_version"] += 1
        elif change == "event":
            state["events"][1]["payload"]["proposal"]["rationale"] = "tampered"
        else:
            state["events"].pop()
        env.book.path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        env.reload()


def test_recovery_rechecks_actual_terminal_receipt_not_just_persisted_pass_boolean(env):
    env.accept_all()
    env.receipts["synthetic-independent-1"]["audit"]["status"] = "rejected"
    with pytest.raises(ValueError, match="terminal successful"):
        env.reload()


def test_duplicate_positions_wrong_paths_or_nonfinite_json_are_rejected(env):
    with pytest.raises(ValueError, match="Duplicate occurrence"):
        mod.SourceObjectDispositions(env.directory / "dup", env.identity, [env.occurrences[0]] * 2, env.units, env.receipts.__getitem__)
    with pytest.raises(ValueError, match="unknown output path"):
        env.propose(affected_output_paths=["invented_score"])
    with pytest.raises(ValueError, match="finite JSON"):
        env.book.bind_context(**{**env.context, "draft": {"score": float("nan")}})


def test_formal_limited_failure_counts_never_remove_failed_denominator(env):
    env.accept_all()
    completed = env.book.completion(gates())
    env.book.bind_context(**{**env.context, "task": {"version": "new required calculation"}})
    limited = env.book.completion(gates(task_review_passed=False, partial_review_passed=True))
    failed = env.book.completion(gates(audit_valid=False))
    incomplete = env.book.completion(gates())
    assert mod.count_completions([completed, limited, failed, incomplete]) == {
        "attempted_reports": 4, "complete_reports": 1, "limited_reports": 1, "failed_reports": 1, "incomplete_reports": 1}
    with pytest.raises(ValueError, match="disagree"):
        mod.count_completions([{**limited, "task_complete": True}])
    with pytest.raises(ValueError, match="mix completion"):
        mod.count_completions([{**completed, "completion_scope": "legacy"}])


def test_outside_task_cannot_omit_original_occurrence_context_even_with_other_read_text(env):
    with pytest.raises(ValueError, match="complete original occurrence context"):
        env.propose(mode="outside_task", context_refs=[source_ref("E3", 0, len(env.units["E3"]))])


def test_representation_verification_record_keeps_actual_response_and_receipt_hashes(env):
    value = mod._unread()
    value.update(state="semantics_verified", content={"number": 1}, inspection={"synthetic_original": "1"})
    assignment = env.book.representation_review_payload(env.ids[0], value)
    decision = dict(assignment_id=assignment["assignment_id"], object_id=env.ids[0],
                    representation_sha256=digest(value), semantics_verified=True, explanation="SYNTHETIC digit correspondence")
    call = env.fake_receipt(assignment, decision)
    env.book.set_representation(env.ids[0], value, verification_call_id=call)
    stored = env.book.history()[-1]["payload"]["verification"]
    assert stored["decision"] == decision
    assert stored["receipt"]["response_sha256"] == digest(env.receipts[call]["response"])
    changed = {**decision, "explanation": "different response after original record"}
    env.fake_receipt(assignment, changed, call)
    with pytest.raises(ValueError, match="differs from the original"):
        env.reload()
