"""CPU-only save/handoff boundaries with the real atomic entry store."""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator
import pytest

from bridge_evidence.contest.document_entries import EntryStore, entry_schema
from bridge_evidence.contest.document_multiagent_worker import unread_primary
from bridge_evidence.contest.document_section_submission import SectionSubmission, submit_section_schema
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import atomic_json


@pytest.fixture
def setup(tmp_path):
    manifest = manifest_from_text("合成桥检测发现支座锈蚀。\n养护建议为支座除锈涂漆。")
    task = {"id": "section-test", "primary_evidence_ids": [u["id"] for u in manifest["units"]]}
    store = EntryStore(manifest, task["id"], tmp_path / "entries")
    calls, active_call = [], {"id": "accepted-model-call"}
    first = manifest["units"][0]
    store.record_read(first["id"], 0, len(first["text"]))
    entry = {"section": "病害明细", "row": {"病害部位": "支座", "病害描述": "锈蚀"},
        "status": "explicit", "source_refs": [{"evidence_id": first["id"]}],
        "context": {"year": "2018", "component": "支座"}}
    path = tmp_path / "handoff.json"

    def complete(summary, unresolved):
        calls.append({"summary": summary, "unresolved": deepcopy(unresolved)})
        state = store.snapshot()
        missing = unread_primary(manifest, task, state["reads"])
        if missing:
            return {"ok": False, "status": "incomplete_reading", "missing": missing,
                    "missing_count": len(missing), "action": "Read remaining source."}
        if not active_call["id"]:
            raise RuntimeError("Section handoff lacks an actual model call")
        atomic_json(path, {"summary": summary, "unresolved": unresolved,
            "entries_version": state["version"], "origin_call_id": active_call["id"]})
        return {"ok": True, "status": "handed_off", "document_delivery_complete": False}

    def read_rest():
        for unit in manifest["units"]:
            store.record_read(unit["id"], 0, len(unit["text"]))

    return {"manifest": manifest, "task": task, "store": store, "entry": entry,
            "calls": calls, "path": path, "read_rest": read_rest, "active_call": active_call,
            "handler": SectionSubmission(store, complete, task["id"]), "callback": complete}


def test_schema_reuses_entry_fields_allows_missing_id_and_requires_control_fields(setup):
    payload = {"entries": [setup["entry"]], "unresolved": [], "complete": True}
    validator = Draft202012Validator(submit_section_schema())
    validator.validate(payload)
    for name in ("entries", "unresolved", "complete"):
        assert list(validator.iter_errors({k: v for k, v in payload.items() if k != name}))
    assert list(Draft202012Validator(entry_schema()).iter_errors(setup["entry"]))
    assert all("id" in branch["required"] for branch in entry_schema()["anyOf"])
    # The public schema is independent per caller, including nested objects.
    changed = submit_section_schema(); changed["properties"]["entries"]["items"]["anyOf"][0]["properties"].clear()
    validator.validate(payload)


def test_auto_ids_keep_duplicate_occurrences_and_replay_is_durable_noop(setup):
    entry = setup["entry"]
    original = deepcopy(entry)
    handler, store = setup["handler"], setup["store"]
    first = handler.submit_section([entry, deepcopy(entry)], [], False)
    assert first["ok"] and first["status"] == "saved" and len(set(first["assigned_ids"])) == 2
    assert first["counts"]["total"] == 2 and not setup["calls"]
    before = store.path.read_bytes()
    restored = EntryStore(setup["manifest"], setup["task"]["id"], store.directory)
    replay = SectionSubmission(restored, setup["callback"], setup["task"]["id"]).submit_section(
        [deepcopy(entry), deepcopy(entry)], [], False)
    assert replay["assigned_ids"] == first["assigned_ids"]
    assert replay["changed"] is False and all(a["reused"] for a in replay["assignments"])
    assert store.path.read_bytes() == before and entry == original
    for row in store.snapshot()["entries"].values():
        assert row["row"] == entry["row"] and row["context"] == entry["context"]


def test_generated_ids_are_task_scoped(setup, tmp_path):
    first = setup["handler"].submit_section([setup["entry"]], [], False)
    other = EntryStore(setup["manifest"], "other-task", tmp_path / "other")
    for identifier, ranges in setup["store"].snapshot()["reads"].items():
        for start, end in ranges:
            other.record_read(identifier, start, end)
    second = SectionSubmission(other, setup["callback"], "other-task").submit_section([setup["entry"]], [], False)
    assert first["assigned_ids"] != second["assigned_ids"]


def test_old_implicit_batch_cannot_overwrite_later_explicit_correction(setup):
    handler, store = setup["handler"], setup["store"]
    original = setup["entry"]
    first = handler.submit_section([original], [], False)
    identifier = first["assigned_ids"][0]
    corrected = {**deepcopy(original), "id": identifier, "row": {"病害部位": "支座", "病害描述": "轻微锈蚀"}}
    updated = handler.submit_section([corrected], [], False)
    assert updated["ok"] and updated["changed"] and updated["counts"]["total"] == 1
    before = store.path.read_bytes()
    replay = handler.submit_section([original], [], False)
    assert replay["ok"] and not replay["changed"] and replay["assignments"][0]["reused"]
    assert store.path.read_bytes() == before
    assert store.snapshot()["entries"][identifier]["row"] == corrected["row"]


@pytest.mark.parametrize("bad, path", [
    ({"entries": "[{\"section\":"}, "entries"),
    ({"complete": "true"}, "complete"),
    ({"complete": 1}, "complete"),
    ({"unresolved": "[]"}, "unresolved"),
    ({"unresolved": [7]}, "unresolved[0]"),
])
def test_bad_control_fields_and_partial_json_never_save_or_handoff(setup, bad, path):
    handler, store = setup["handler"], setup["store"]
    good = handler.submit_section([setup["entry"]], [], False)
    before = store.path.read_bytes()
    new_entry = deepcopy(setup["entry"]); new_entry["row"]["病害描述"] = "模型准备更新"
    payload = {"entries": [new_entry], "unresolved": [], "complete": True, **bad}
    failed = handler.submit_section(**payload)
    assert not failed["ok"] and not failed["saved"] and not setup["calls"]
    assert failed["error"]["details"]["path"] == path
    assert "example" in failed["error"]["details"]
    assert store.path.read_bytes() == before and good["counts"]["total"] == 1


def test_mixed_structural_failure_cannot_partially_replace_existing_entry(setup):
    handler, store = setup["handler"], setup["store"]
    first = handler.submit_section([setup["entry"]], [], False)
    changed = {**deepcopy(setup["entry"]), "id": first["assigned_ids"][0], "row": {"病害描述": "待更正"}}
    bad = deepcopy(setup["entry"]); bad["row"]["未知字段"] = "不能导出"
    before = store.path.read_bytes()
    result = handler.submit_section([changed, bad], [], True)
    assert not result["ok"] and result["error"]["details"]["path"] == "entries[1].row.未知字段"
    assert store.path.read_bytes() == before and not setup["calls"]


def test_source_failure_in_second_entry_rolls_back_entire_batch(setup):
    second = setup["manifest"]["units"][1]
    bad = deepcopy(setup["entry"]); bad["source_refs"] = [{"evidence_id": second["id"]}]
    before = setup["store"].path.read_bytes()
    result = setup["handler"].submit_section([setup["entry"], bad], [], True)
    assert not result["ok"] and result["error"]["code"] == "SOURCE_UNREAD"
    assert result["error"]["details"]["path"] == "entries[1].source_refs[0]"
    assert setup["store"].path.read_bytes() == before and not setup["calls"]


def test_explicit_and_generated_same_id_in_one_batch_is_atomic_error(setup):
    first = setup["handler"].submit_section([setup["entry"]], [], False)
    explicit = {**deepcopy(setup["entry"]), "id": first["assigned_ids"][0]}
    before = setup["store"].path.read_bytes()
    result = setup["handler"].submit_section([setup["entry"], explicit], [], True)
    assert not result["ok"] and result["error"]["details"]["path"] == "entries[1].id"
    assert setup["store"].path.read_bytes() == before and not setup["calls"]


def test_save_then_incomplete_reading_preserves_entries_and_exact_next_action(setup):
    result = setup["handler"].submit_section([setup["entry"]], ["历史状态仍需核查"], True)
    assert not result["ok"] and result["saved"] and not result["handed_off"]
    assert result["status"] == "incomplete_reading" and result["counts"]["total"] == 1
    assert result["next_action"] == {"tool": "read_source", "arguments": {
        "evidence_id": setup["manifest"]["units"][1]["id"]}}
    assert not setup["path"].exists()
    setup["read_rest"]()
    finished = setup["handler"].submit_section([], ["历史状态仍需主Agent核查"], True)
    assert finished["ok"] and finished["status"] == "handed_off" and finished["handed_off"]
    assert not finished["changed"] and not finished["document_delivery_complete"]
    saved_handoff = json.loads(setup["path"].read_text())
    assert saved_handoff["unresolved"] == ["历史状态仍需主Agent核查"]
    assert saved_handoff["entries_version"] == setup["store"].snapshot()["version"]
    assert saved_handoff["origin_call_id"] == setup["active_call"]["id"]


def test_success_uses_mechanical_summary_and_preserves_exact_unresolved(setup):
    setup["read_rest"]()
    result = setup["handler"].submit_section([setup["entry"]], ["左右幅关系待核查"], True)
    assert result["ok"] and result["handed_off"] and not result["semantic_approval"]
    summary = setup["calls"][0]["summary"]
    assert setup["task"]["id"] in summary and "病害明细" in summary
    assert "锈蚀" not in summary and "2018" not in summary
    assert setup["calls"][0]["unresolved"] == ["左右幅关系待核查"]


def test_actual_model_call_validation_still_blocks_handoff_after_save(setup):
    setup["read_rest"](); setup["active_call"]["id"] = None
    result = setup["handler"].submit_section([setup["entry"]], [], True)
    assert not result["ok"] and result["saved"] and result["status"] == "handoff_failed"
    assert "actual model call" in result["error"]["message"]
    assert setup["store"].snapshot()["counts"]["total"] == 1 and not setup["path"].exists()
    setup["active_call"]["id"] = "next-accepted-call"
    assert setup["handler"].submit_section([], [], True)["handed_off"]


def test_zero_entries_can_handoff_only_after_callback_read_validation(setup):
    result = setup["handler"].submit_section([], ["本分工暂无可交付条目，交主Agent核查"], True)
    assert result["status"] == "incomplete_reading" and not result["handed_off"]
    setup["read_rest"]()
    result = setup["handler"].submit_section([], ["本分工暂无可交付条目，交主Agent核查"], True)
    assert result["handed_off"] and result["counts"]["total"] == 0
    assert not result["semantic_approval"]
    no_op = setup["handler"].submit_section([], [], False)
    assert not no_op["ok"] and no_op["error"]["code"] == "SECTION_SUBMISSION_EMPTY"


@pytest.mark.parametrize("receipt", [None, {"ok": True, "status": "saved"}, {"ok": False, "status": "incomplete_reading"}])
def test_unverified_callback_response_cannot_claim_handoff(setup, receipt):
    handler = SectionSubmission(setup["store"], lambda *args: receipt, setup["task"]["id"])
    result = handler.submit_section([setup["entry"]], [], True)
    assert not result["ok"] and not result["handed_off"] and result["saved"]
    assert setup["store"].snapshot()["counts"]["total"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "\ud800"])
def test_nonfinite_or_invalid_unicode_does_not_hash_or_mutate(setup, value):
    bad = deepcopy(setup["entry"]); bad["row"]["病害描述"] = value
    before = setup["store"].path.read_bytes()
    result = setup["handler"].submit_section([bad], [], True)
    assert not result["ok"] and result["error"]["details"]["path"] == "entries[0].row.病害描述"
    assert setup["store"].path.read_bytes() == before and not setup["calls"]
