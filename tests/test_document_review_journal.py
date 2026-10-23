"""Synthetic persistence contracts; no reports, SDK, network or model calls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest import document_review_journal as module
from bridge_evidence.contest.run_state import json_digest


def decision(call_id="synthetic-call-a", passed=True):
    return {"model_call_id": call_id, "passed": passed,
            "checked_output_paths": ["basic.bridge_name"],
            "checked_evidence_spans": [{"evidence_id": "synthetic-E1", "start": 0, "end": 3}],
            "unsupported_claims": [], "missing_items": [], "conflicts": [], "unresolved_items": [],
            "explanation": "synthetic decision retained in full"}


class ReviewJournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "review"
        self.identity = {"workflow_identity": {"model_hash": "synthetic", "document_hash": "synthetic"},
                         "review_version": "synthetic-review-v1"}
        self.payload = {"direction": "FORWARD", "output_items": [{"path": "basic.bridge_name", "value": "示例"}]}

    def journal(self):
        return module.ReviewJournal(self.directory, self.identity)

    def seed(self):
        journal = self.journal()
        attempt = journal.begin("semantic-a")
        journal.put("semantic-a", self.payload, decision())
        return journal, attempt

    def disk(self):
        return json.loads((self.directory / "review_journal.json").read_text(encoding="utf-8"))

    def replace_disk(self, state, *, reseal=False):
        state = deepcopy(state)
        if reseal:
            state.pop("state_sha256", None)
            state["state_sha256"] = json_digest(state)
        (self.directory / "review_journal.json").write_text(json.dumps(state), encoding="utf-8")

    def test_empty_journal_returns_no_invented_decision_and_binds_identity(self):
        journal = self.journal()
        self.assertEqual(journal.attempt_count, 0)
        self.assertIsNone(journal.active("semantic-a"))
        self.assertIsNone(journal.get("semantic-a", self.payload))
        self.assertEqual(self.disk()["identity"], self.identity)
        self.assertEqual(self.disk()["identity_sha256"], json_digest(self.identity))

    def test_begin_is_durable_before_return_and_same_active_attempt_resumes_without_new_round(self):
        journal = self.journal()
        first = journal.begin("semantic-a")
        self.assertTrue(first["started_new"])
        resumed = self.journal()
        with patch.object(module, "atomic_json") as write:
            self.assertEqual(resumed.begin("semantic-a"), {"attempt_id": first["attempt_id"], "started_new": False})
            write.assert_not_called()
        self.assertEqual(resumed.attempt_count, 1)
        active = resumed.active("semantic-a")
        active["semantic_hash"] = "mutated"
        self.assertEqual(resumed.active("semantic-a")["semantic_hash"], "semantic-a")
        self.assertIsNone(resumed.active("semantic-b"))

    def test_semantic_changes_supersede_old_attempts_and_keep_all_history(self):
        journal = self.journal()
        first = journal.begin("semantic-a")
        second = journal.begin("semantic-b")
        third = journal.begin("semantic-a")
        self.assertEqual(journal.attempt_count, 3)
        self.assertEqual(len({a["attempt_id"] for a in (first, second, third)}), 3)
        self.assertEqual([row["status"] for row in self.disk()["attempts"]], ["superseded", "superseded", "active"])
        self.assertEqual(self.journal().attempt_count, 3)
        with self.assertRaisesRegex(RuntimeError, "superseded"):
            journal.finish(first["attempt_id"])

    def test_finished_decisions_survive_restart_and_new_round_reuses_only_matching_payload(self):
        journal, attempt = self.seed()
        journal.finish(attempt["attempt_id"])
        journal = self.journal()
        self.assertIsNone(journal.active("semantic-a"))
        self.assertEqual(journal.get("semantic-a", self.payload), decision())
        next_attempt = journal.begin("semantic-a")
        self.assertTrue(next_attempt["started_new"])
        self.assertNotEqual(next_attempt["attempt_id"], attempt["attempt_id"])
        self.assertEqual(journal.attempt_count, 2)
        self.assertEqual(journal.get("semantic-a", self.payload), decision())
        self.assertIsNone(journal.get("semantic-a", {**self.payload, "direction": "REVERSE"}))
        self.assertEqual(len(self.disk()["records"]), 1)

    def test_semantic_hash_and_complete_payload_both_isolate_cached_decisions(self):
        journal, _attempt = self.seed()
        different = {**self.payload, "new_field": "new context"}
        self.assertIsNone(journal.get("semantic-a", different))
        self.assertIsNone(journal.get("semantic-b", self.payload))
        journal.put("semantic-a", different, decision("synthetic-call-b"))
        journal.begin("semantic-b")
        journal.put("semantic-b", self.payload, decision("synthetic-call-c"))
        self.assertEqual(journal.get("semantic-a", self.payload)["model_call_id"], "synthetic-call-a")
        self.assertEqual(journal.get("semantic-a", different)["model_call_id"], "synthetic-call-b")
        self.assertEqual(journal.get("semantic-b", self.payload)["model_call_id"], "synthetic-call-c")
        self.assertEqual(len(self.disk()["records"]), 3)

    def test_cached_decisions_are_copies_and_original_input_mutation_cannot_change_disk(self):
        journal = self.journal()
        journal.begin("semantic-a")
        value = decision()
        journal.put("semantic-a", self.payload, value)
        value["passed"] = False
        returned = journal.get("semantic-a", self.payload)
        returned["checked_output_paths"].append("invented.path")
        self.assertEqual(journal.get("semantic-a", self.payload), decision())
        self.assertEqual(self.journal().get("semantic-a", self.payload), decision())

    def test_same_decision_put_is_idempotent_but_different_decision_cannot_overwrite(self):
        journal, attempt = self.seed()
        journal.finish(attempt["attempt_id"])
        with patch.object(module, "atomic_json") as write:
            journal.put("semantic-a", self.payload, decision())
            write.assert_not_called()
        for changed in (decision(passed=False), decision("different-call"), {**decision(), "explanation": "different"}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "overwrite"):
                journal.put("semantic-a", self.payload, changed)
        self.assertEqual(self.journal().get("semantic-a", self.payload), decision())

    def test_missing_or_wrong_model_decision_fields_are_rejected_before_persisting(self):
        journal = self.journal()
        journal.begin("semantic-a")
        bad = []
        for name in decision():
            if name != "explanation":
                value = decision()
                value.pop(name)
                bad.append(value)
        bad += [{**decision(), "model_call_id": value} for value in ("", " ", None, 3)]
        bad += [{**decision(), "passed": value} for value in (0, 1, "true", None)]
        bad += [{**decision(), name: value} for name in module._ARRAYS for value in ({}, "", None)]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                journal.put("semantic-a", self.payload, value)
        self.assertEqual(self.disk()["records"], {})

    def test_new_put_requires_matching_active_attempt_and_finish_requires_known_attempt(self):
        journal = self.journal()
        with self.assertRaisesRegex(RuntimeError, "active attempt"):
            journal.put("semantic-a", self.payload, decision())
        journal.begin("semantic-b")
        with self.assertRaisesRegex(RuntimeError, "active attempt"):
            journal.put("semantic-a", self.payload, decision())
        with self.assertRaisesRegex(ValueError, "unknown attempt_id"):
            journal.finish("not-an-attempt")

    def test_finish_is_idempotent_and_does_not_consume_another_round(self):
        journal, attempt = self.seed()
        journal.finish(attempt["attempt_id"])
        with patch.object(module, "atomic_json") as write:
            journal.finish(attempt["attempt_id"])
            write.assert_not_called()
        self.assertEqual(journal.attempt_count, 1)

    def test_identity_and_schema_changes_are_not_resumed_even_with_recomputed_state_hash(self):
        self.seed()
        original = self.disk()
        with self.assertRaisesRegex(ValueError, "identity"):
            module.ReviewJournal(self.directory, {**self.identity, "review_version": "different"})
        for key, value in (("identity", {"changed": True}), ("identity_sha256", "bad"),
                           ("schema_version", 2), ("schema_version", True)):
            with self.subTest(key=key, value=value):
                self.replace_disk({**original, key: value}, reseal=True)
                with self.assertRaises(ValueError):
                    self.journal()

    def test_decision_payload_and_attempt_tampering_are_not_accepted(self):
        self.seed()
        original = self.disk()
        key = next(iter(original["records"]))
        changes = []
        changed = deepcopy(original)
        changed["records"][key]["decision"]["passed"] = False
        changes.append(changed)
        changed = deepcopy(original)
        changed["records"][key]["payload_sha256"] = "a" * 64
        changes.append(changed)
        changed = deepcopy(original)
        changed["records"][key]["attempt_id"] = "unrecorded-attempt"
        changes.append(changed)
        changed = deepcopy(original)
        changed["active_attempt_id"] = None
        changes.append(changed)
        changed = deepcopy(original)
        changed["attempts"].append(deepcopy(changed["attempts"][0]))
        changes.append(changed)
        for changed in changes:
            for reseal in (False, True):
                with self.subTest(changed=changed, reseal=reseal):
                    self.replace_disk(changed, reseal=reseal)
                    with self.assertRaises(ValueError):
                        self.journal()

    def test_corrupt_json_is_rejected_instead_of_resetting_a_journal(self):
        journal, _attempt = self.seed()
        journal.path.write_text('{"broken":', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cannot restore"):
            self.journal()
        self.assertEqual(journal.path.read_text(encoding="utf-8"), '{"broken":')

    def test_failed_begin_put_and_finish_preserve_disk_and_memory_for_retry(self):
        journal = self.journal()
        for operation in ("begin", "put", "finish"):
            with self.subTest(operation=operation):
                before = self.disk()
                if operation == "begin":
                    action = lambda: journal.begin("semantic-a")
                elif operation == "put":
                    action = lambda: journal.put("semantic-a", self.payload, decision())
                else:
                    attempt_id = journal.active("semantic-a")["attempt_id"]
                    action = lambda: journal.finish(attempt_id)
                with patch.object(module, "atomic_json", side_effect=OSError("synthetic disk failure")):
                    with self.assertRaisesRegex(OSError, operation):
                        action()
                self.assertEqual(self.disk(), before)
                self.assertEqual(journal._state, before)
                action()
        self.assertEqual(journal.attempt_count, 1)
        self.assertIsNone(journal.active("semantic-a"))
        self.assertEqual(self.journal().get("semantic-a", self.payload), decision())


def compact_fixture(direction="REVERSE", protocol="bidirectional-semantic-review-v3"):
    identity = {"workflow": {"source": "synthetic-source", "versions": {"review": protocol}},
                "protocol": protocol}
    semantic_hash = json_digest({"draft": "synthetic-frozen-draft"})
    payload = {"direction": direction, "output_items": [{"path": "基本信息.桥梁名称", "value": "合成桥"}],
               "source_spans": [{"evidence_id": "E1", "start": 0, "end": 4}]}
    assignment = {"protocol": identity["protocol"], "workflow_sha256": json_digest(identity["workflow"]),
        "semantic_state_sha256": semantic_hash, "payload_sha256": json_digest(payload),
        "draft_catalog_sha256": json_digest(payload["output_items"]), "output_item_count": 1, "source_interval_count": 1}
    assignment["id"] = "ra_" + json_digest(assignment)[:32]
    value = {"model_call_id": "synthetic-compact-call", "passed": True, "unsupported_claims": [], "missing_items": [],
        "conflicts": [], "unresolved_items": [], "review_assignment": assignment, "assignment_confirmation": "compact",
        "checked_assignment_id": assignment["id"], "checked_all_assigned": True,
        "assigned_output_paths": ["基本信息.桥梁名称"], "assigned_evidence_spans": deepcopy(payload["source_spans"])}
    return identity, semantic_hash, payload, value


def reseal_assignment(value):
    assignment = value["review_assignment"]
    assignment["id"] = "ra_" + json_digest({k: v for k, v in assignment.items() if k != "id"})[:32]
    value["checked_assignment_id"] = assignment["id"]


@pytest.mark.parametrize("direction", ["FORWARD", "REVERSE"])
@pytest.mark.parametrize("protocol", ["bidirectional-semantic-review-v3", "bidirectional-semantic-review-v4"])
def test_compact_ack_retains_program_scopes_without_inventing_model_echo(tmp_path, direction, protocol):
    identity, semantic_hash, payload, value = compact_fixture(direction, protocol)
    journal = module.ReviewJournal(tmp_path, identity)
    attempt = journal.begin(semantic_hash)
    journal.put(semantic_hash, payload, value)
    journal.finish(attempt["attempt_id"])
    restored = module.ReviewJournal(tmp_path, identity).get(semantic_hash, payload)
    assert restored == value
    assert "checked_evidence_spans" not in restored and "checked_output_paths" not in restored
    assert restored["assigned_evidence_spans"] == payload["source_spans"]


@pytest.mark.parametrize("field,bad", [
    ("checked_assignment_id", "old-assignment"), ("checked_all_assigned", False), ("checked_all_assigned", 1),
    ("assignment_confirmation", "automatic"), ("assigned_evidence_spans", []),
    ("assigned_output_paths", ["wrong.path"]), ("checked_evidence_spans", []), ("checked_output_paths", []),
    ("missing_items", [{"evidence_id": "E1", "reason": "synthetic omission"}]),
])
def test_compact_missing_false_stale_or_contradictory_ack_is_atomic(tmp_path, field, bad):
    identity, semantic_hash, payload, value = compact_fixture()
    journal = module.ReviewJournal(tmp_path, identity)
    journal.begin(semantic_hash)
    before = journal.path.read_bytes()
    value[field] = bad
    with pytest.raises(ValueError):
        journal.put(semantic_hash, payload, value)
    assert journal.path.read_bytes() == before
    assert journal.get(semantic_hash, payload) is None


@pytest.mark.parametrize("field", ["checked_assignment_id", "checked_all_assigned", "review_assignment", "assigned_output_paths", "assigned_evidence_spans"])
def test_compact_requires_explicit_ack_and_assigned_scope_fields(tmp_path, field):
    identity, semantic_hash, payload, value = compact_fixture()
    value.pop(field)
    journal = module.ReviewJournal(tmp_path, identity)
    journal.begin(semantic_hash)
    with pytest.raises(ValueError):
        journal.put(semantic_hash, payload, value)


@pytest.mark.parametrize("field,bad", [
    ("workflow_sha256", "1" * 64), ("semantic_state_sha256", "2" * 64), ("payload_sha256", "3" * 64),
    ("draft_catalog_sha256", "4" * 64), ("output_item_count", True), ("source_interval_count", 3),
    ("protocol", "bidirectional-semantic-review-v2"),
])
def test_rehashed_ack_cannot_switch_workflow_semantics_payload_or_forward_catalog(tmp_path, field, bad):
    identity, semantic_hash, payload, value = compact_fixture("FORWARD")
    value["review_assignment"][field] = bad
    reseal_assignment(value)
    journal = module.ReviewJournal(tmp_path, identity)
    journal.begin(semantic_hash)
    with pytest.raises(ValueError):
        journal.put(semantic_hash, payload, value)


def test_v3_legacy_echo_is_still_accepted_but_never_fabricated(tmp_path):
    identity, semantic_hash, payload, value = compact_fixture()
    value["assignment_confirmation"] = "legacy"
    value.pop("checked_all_assigned")
    value.pop("checked_assignment_id")
    value["checked_output_paths"] = deepcopy(value["assigned_output_paths"])
    value["checked_evidence_spans"] = deepcopy(value["assigned_evidence_spans"])
    journal = module.ReviewJournal(tmp_path, identity)
    journal.begin(semantic_hash)
    journal.put(semantic_hash, payload, value)
    assert module.ReviewJournal(tmp_path, identity).get(semantic_hash, payload) == value
    assert "checked_assignment_id" not in value


def test_compact_and_matching_legacy_can_coexist_but_old_unbound_decision_cannot(tmp_path):
    identity, semantic_hash, payload, value = compact_fixture()
    value["checked_output_paths"] = deepcopy(value["assigned_output_paths"])
    value["checked_evidence_spans"] = deepcopy(value["assigned_evidence_spans"])
    journal = module.ReviewJournal(tmp_path, identity)
    journal.begin(semantic_hash)
    with pytest.raises(ValueError, match="scope binding"):
        journal.put(semantic_hash, payload, decision())
    journal.put(semantic_hash, payload, value)
    assert journal.get(semantic_hash, payload) == value


def test_assignment_binding_is_rechecked_on_reload_after_outer_hashes_recomputed(tmp_path):
    identity, semantic_hash, payload, value = compact_fixture()
    journal = module.ReviewJournal(tmp_path, identity)
    journal.begin(semantic_hash)
    journal.put(semantic_hash, payload, value)
    raw = json.loads(journal.path.read_text())
    record = next(iter(raw["records"].values()))
    record["decision"]["review_assignment"]["semantic_state_sha256"] = "f" * 64
    reseal_assignment(record["decision"])
    record["decision_sha256"] = json_digest(record["decision"])
    raw.pop("state_sha256")
    raw["state_sha256"] = json_digest(raw)
    journal.path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="binding differs"):
        module.ReviewJournal(tmp_path, identity)


def test_compact_current_payload_rechecks_stored_program_scopes(tmp_path):
    identity, semantic_hash, payload, value = compact_fixture()
    journal = module.ReviewJournal(tmp_path, identity)
    journal.begin(semantic_hash)
    changed = deepcopy(payload)
    changed["source_spans"][0]["end"] = 7
    value["review_assignment"]["payload_sha256"] = json_digest(changed)
    reseal_assignment(value)
    with pytest.raises(ValueError, match="scopes differ"):
        journal.put(semantic_hash, changed, value)


if __name__ == "__main__":
    unittest.main()


def test_v4_cannot_use_v3_scope_binding_or_omit_compact_assignment(tmp_path):
    identity, semantic_hash, payload, value = compact_fixture(protocol='bidirectional-semantic-review-v4')
    journal=module.ReviewJournal(tmp_path,identity)
    journal.begin(semantic_hash)
    old=deepcopy(value)
    old['review_assignment']['protocol']='bidirectional-semantic-review-v3'
    reseal_assignment(old)
    with pytest.raises(ValueError,match='binding differs'):
        journal.put(semantic_hash,payload,old)
    missing={'model_call_id':'synthetic-no-binding','passed':True,'checked_output_paths':[],
        'checked_evidence_spans':[], 'unsupported_claims':[],'missing_items':[],'conflicts':[],'unresolved_items':[]}
    with pytest.raises(ValueError,match='requires program-assigned scope binding'):
        journal.put(semantic_hash,payload,missing)
    assert journal.get(semantic_hash,payload) is None
