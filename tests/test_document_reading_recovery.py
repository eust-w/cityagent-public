"""Synthetic protocol recovery, not evidence of model quality or SDK execution."""
from copy import deepcopy
import json

import pytest

from bridge_evidence.contest.document_reading_recovery import bounded_recovery, pending_workspace_retry, scan_workspace_retry
from bridge_evidence.contest.run_state import json_digest
from test_document_workflow import workflow


def bundle(raw, *, name="update_workspace"):
    response = {"choices": [{"message": {"tool_calls": [{"id": "native-1", "type": "function",
        "function": {"name": name, "arguments": raw}}]}}]}
    return {"audit": {"request_kind": "agent_generation", "status": "success", "call_id": "model-1",
                      "response_sha256": json_digest(response)}, "response": response}


def sdk_failure():
    return {"tool_calls": [], "sdk_tool_errors": [
        {"tool_name": "update_workspace", "agent_call_id": "model-1", "code": "SDK_TOOL_ARGUMENT_JSON", "action": "Regenerate JSON"}]}


def test_malformed_native_arguments_are_exact_unaccepted_model_proposal():
    raw = '{"update":{"base_version":4,"draft_patch":[{"op":"append_row","row":{"病害描述":"合成裂缝"}},]}}'
    original = bundle(raw)
    frozen = deepcopy(original)
    result = pending_workspace_retry(original, sdk_failure())
    assert result["arguments_text"] == raw
    assert result["arguments_text_sha256"] == json_digest(raw)
    assert result["accepted_as_workspace_state"] is False
    assert result["response_sha256"] == original["audit"]["response_sha256"]
    assert "Never replay" in result["instruction"] and "section" in result["instruction"]
    assert original == frozen
    with pytest.raises(ValueError):
        json.loads(result["arguments_text"])


@pytest.mark.parametrize("status,eligible", [("failed", True), ("success", False)])
def test_valid_handler_arguments_require_matching_failed_receipt(status, eligible):
    args = {"update": {"base_version": 0, "draft_patch": [{"op": "append_row", "row": {"病害描述": "合成"}}]}}
    original = bundle(json.dumps(args, ensure_ascii=False))
    session = {"tool_calls": [{"name": "update_workspace", "args_sha256": json_digest(args), "status": status, "agent_call_id": "model-1"}]}
    result = pending_workspace_retry(original, session)
    assert bool(result) is eligible


@pytest.mark.parametrize("change", ["wrong-tool", "no-failure", "image-purpose", "multiple-choices", "text-only"])
def test_free_text_ambiguous_or_unrelated_actions_never_become_retry(change):
    original, session = bundle('{"update":{}}'), sdk_failure()
    if change == "wrong-tool":
        original = bundle('bad json', name="read_source")
    elif change == "no-failure":
        session = {"tool_calls": [], "sdk_tool_errors": []}
    elif change == "image-purpose":
        original["audit"]["request_kind"] = "image_observation"
    elif change == "multiple-choices":
        original["response"]["choices"] *= 2
    else:
        original["response"] = {"choices": [{"message": {"content": '{"update":{}}'}}]}
    assert pending_workspace_retry(original, session) is None


def tokens(value):
    return (len(value) + 3) // 4


def test_oversized_candidate_omitted_whole_with_hash_not_partial_json():
    raw = '{"update":' + "x" * 20000
    value = {"working_notes": {"next": "合成下一步"}, "pending_workspace_retry": pending_workspace_retry(bundle(raw), sdk_failure())}
    restored = bounded_recovery(value, tokens, 500)
    retry = restored["pending_workspace_retry"]
    assert "arguments_text" not in retry and "arguments_omitted" in retry
    assert retry["arguments_text_sha256"] == json_digest(raw)
    assert restored["working_notes"] == value["working_notes"]
    assert value["pending_workspace_retry"]["arguments_text"] == raw


def test_oversized_notes_remain_available_as_whole_workspace_page():
    value = {"working_notes": {"next": "x" * 20000}, "pending_workspace_retry": None}
    result = bounded_recovery(value, tokens, 100)
    assert "working_notes" not in result
    assert result["working_notes_read"]["arguments"] == {"view": "working_notes", "cursor": 0, "limit": 1}


def test_metadata_overflow_and_invalid_counter_fail_without_truncation():
    with pytest.raises(ValueError, match="metadata"):
        bounded_recovery({"static": "x" * 5000}, tokens, 20)
    with pytest.raises(ValueError, match="nonnegative"):
        bounded_recovery({}, lambda _: -1, 20)


def test_workflow_recovers_original_failed_write_across_a_later_read_only_session(tmp_path):
    wf = workflow(tmp_path)
    first = sdk_failure()
    first["reading_model_call_ids"] = ["failed-write"]
    wf.sessions = [first, {"tool_calls": [], "sdk_tool_errors": [], "reading_model_call_ids": ["read-only"]}]
    raw = '{"update":{"base_version":0,"draft_patch":[}'
    bundles = {"failed-write": bundle(raw), "read-only": bundle('{}', name="read_source")}
    wf._terminal_model_bundle = lambda call_id: deepcopy(bundles[call_id])
    before = wf.workspace.snapshot()
    restored = wf._reading_recovery(before, {})
    assert restored["pending_workspace_retry"]["arguments_text"] == raw
    assert restored["draft_counts"] == {"basic": 0, "advice": 0, "defects": 0}
    assert restored["working_notes_are_evidence"] is False
    assert wf.workspace.snapshot() == before


def test_workflow_successful_latest_write_supersedes_earlier_failed_candidate(tmp_path):
    wf = workflow(tmp_path)
    old, accepted = bundle('bad json'), bundle('{"update":{"base_version":0,"facts":[]}}')
    args = json.loads(accepted["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    wf.sessions = [{**sdk_failure(), "reading_model_call_ids": ["old"]},
                   {"reading_model_call_ids": ["accepted"], "tool_calls": [
                       {"name": "update_workspace", "args_sha256": json_digest(args), "status": "success", "agent_call_id": "model-1"}]}]
    wf._terminal_model_bundle = lambda call_id: deepcopy({"old": old, "accepted": accepted}[call_id])
    assert wf._pending_reading_update() is None


@pytest.mark.parametrize("same_session", [True, False])
def test_successful_notes_only_update_does_not_hide_failed_content_patch(tmp_path, same_session):
    wf = workflow(tmp_path)
    failed, notes = bundle('bad json'), bundle('{"update":{"base_version":0,"working_notes":{"next":"修正病害补丁"}}}')
    args = json.loads(notes["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    notes["audit"]["call_id"] = "notes-2"
    receipt = {"name": "update_workspace", "args_sha256": json_digest(args), "status": "success", "agent_call_id": "notes-2"}
    if same_session:
        wf.sessions = [{**sdk_failure(), "reading_model_call_ids": ["failed", "notes"], "tool_calls": [receipt]}]
    else:
        wf.sessions = [{**sdk_failure(), "reading_model_call_ids": ["failed"]},
                       {"reading_model_call_ids": ["notes"], "tool_calls": [receipt]}]
    wf._terminal_model_bundle = lambda call_id: deepcopy({"failed": failed, "notes": notes}[call_id])
    result = wf._pending_reading_update()
    assert result["arguments_text"] == 'bad json'
    assert result["failure_association"] == "confirmed_failed_sdk"


def test_same_tool_error_from_other_model_call_is_explicitly_unconfirmed():
    session = sdk_failure()
    session["sdk_tool_errors"][0]["agent_call_id"] = "earlier-call"
    result = pending_workspace_retry(bundle('new bad json'), session)
    assert result["failure_association"] == "unconfirmed_sdk_association"


def test_later_unconfirmed_action_does_not_hide_confirmed_earlier_failure(tmp_path):
    wf = workflow(tmp_path)
    failed, unconfirmed = bundle('confirmed bad json'), bundle('later bad json')
    unconfirmed["audit"]["call_id"] = "later-call"
    wf.sessions = [{**sdk_failure(), "reading_model_call_ids": ["failed", "later"]}]
    wf._terminal_model_bundle = lambda call_id: deepcopy({"failed": failed, "later": unconfirmed}[call_id])
    result = wf._pending_reading_update()
    assert result["arguments_text"] == 'confirmed bad json'
    assert result["failure_association"] == "confirmed_failed_sdk"


def multiple_bundle(*raws):
    value = bundle(raws[0])
    calls = value["response"]["choices"][0]["message"]["tool_calls"]
    for index, raw in enumerate(raws[1:], 2):
        calls.append({"id": f"native-{index}", "type": "function",
                      "function": {"name": "update_workspace", "arguments": raw}})
    value["audit"]["response_sha256"] = json_digest(value["response"])
    return value


def handler_receipt(raw, status):
    return {"name": "update_workspace", "args_sha256": json_digest(json.loads(raw)),
            "status": status, "agent_call_id": "model-1"}


CONTENT_A = '{"update":{"base_version":0,"draft_patch":[{"op":"append_row","row_id":"a"}]}}'
CONTENT_B = '{"update":{"base_version":1,"draft_patch":[{"op":"append_row","row_id":"b"}]}}'
NOTES = '{"update":{"base_version":1,"working_notes":{"next":"继续病害"}}}'


def test_same_response_successful_notes_do_not_hide_failed_content(tmp_path):
    wf = workflow(tmp_path)
    value = multiple_bundle(CONTENT_A, NOTES)
    session = {"reading_model_call_ids": ["model-1"], "tool_calls": [
        handler_receipt(CONTENT_A, "failed"), handler_receipt(NOTES, "success")]}
    original = deepcopy((value, session, wf.workspace.snapshot()))
    scan = scan_workspace_retry(value, session)
    assert scan["supersedes_previous"] is False
    assert scan["candidate"]["arguments_text"] == CONTENT_A
    assert scan["candidate"]["failure_association"] == "confirmed_failed_handler"
    wf.sessions = [session]
    wf._terminal_model_bundle = lambda _: value
    assert wf._pending_reading_update() == scan["candidate"]
    assert (value, session, wf.workspace.snapshot()) == original


def test_same_response_unknown_sdk_action_does_not_hide_confirmed_failure():
    value = multiple_bundle(CONTENT_A, 'bad json B')
    session = {"tool_calls": [handler_receipt(CONTENT_A, "failed")],
               "sdk_tool_errors": [{"tool_name": "update_workspace", "agent_call_id": "model-1"}]}
    scan = scan_workspace_retry(value, session)
    assert scan["candidate"]["arguments_text"] == CONTENT_A
    assert scan["candidate"]["failure_association"] == "confirmed_failed_handler"
    assert scan["supersedes_previous"] is False


@pytest.mark.parametrize("execution_order,expected", [
    ([(CONTENT_A, "failed"), (CONTENT_B, "success")], None),
    ([(CONTENT_B, "success"), (CONTENT_A, "failed")], CONTENT_A),
])
def test_content_boundary_uses_actual_receipt_order_not_native_array(execution_order, expected):
    value = multiple_bundle(CONTENT_A, CONTENT_B)
    session = {"tool_calls": [handler_receipt(raw, status) for raw, status in execution_order]}
    scan = scan_workspace_retry(value, session)
    assert scan["supersedes_previous"] is (expected is None)
    assert (scan["candidate"]["arguments_text"] if scan["candidate"] else None) == expected


def test_same_response_latest_actual_content_failure_is_retained():
    value = multiple_bundle(CONTENT_A, CONTENT_B)
    session = {"tool_calls": [handler_receipt(CONTENT_A, "success"), handler_receipt(CONTENT_B, "failed")]}
    scan = scan_workspace_retry(value, session)
    assert scan["candidate"]["arguments_text"] == CONTENT_B
    assert scan["supersedes_previous"] is False


def test_exact_sdk_call_id_proves_failed_action_among_multiple_native_calls():
    value = multiple_bundle('bad json A', NOTES)
    session = {"tool_calls": [handler_receipt(NOTES, "success")], "sdk_tool_errors": [{
        "tool_name": "update_workspace", "agent_call_id": "model-1",
        "tool_call_id_sha256": json_digest("native-1")}]}
    scan = scan_workspace_retry(value, session)
    assert scan["candidate"]["arguments_text"] == 'bad json A'
    assert scan["candidate"]["failure_association"] == "confirmed_failed_sdk"
    assert scan["supersedes_previous"] is False


def test_sdk_error_without_unique_native_binding_stays_unknown():
    value = multiple_bundle('bad json A', 'bad json B')
    session = {"sdk_tool_errors": [{"tool_name": "update_workspace", "agent_call_id": "model-1"}]}
    scan = scan_workspace_retry(value, session)
    assert scan["candidate"]["failure_association"] == "unconfirmed_sdk_association"
    assert scan["candidate"]["arguments_text"] == 'bad json B'
    assert scan["supersedes_previous"] is False


def test_accepted_content_with_unrelated_sdk_error_does_not_become_unknown():
    value = multiple_bundle(CONTENT_A, NOTES)
    session = {"tool_calls": [handler_receipt(CONTENT_A, "success"), handler_receipt(NOTES, "success")],
               "sdk_tool_errors": [{"tool_name": "update_workspace", "agent_call_id": "older-call"}]}
    assert scan_workspace_retry(value, session) == {"candidate": None, "supersedes_previous": True}
