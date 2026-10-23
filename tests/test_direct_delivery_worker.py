"""Actual worker/store/source wiring with scripted local generation receipts.

No real SDK, model, network, GPU or service is used. SDK terminal event behavior
is covered separately; these cases exercise the worker's independent checks.
"""
from copy import deepcopy
import json
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from bridge_evidence.contest import document_multiagent_worker as worker
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import atomic_json, json_digest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Direct delivery worker tests must not call a service or subprocess")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)


def terminal_outcome(handoff):
    return {"status": "completed", "termination_kind": "section_terminal_handoff",
        "native_final_event": False, "response_text": "",
        "terminal_handoff_sha256": json_digest(handoff),
        "terminal_handoff_origin_call_id": handoff["origin_call_id"],
        "terminal_handoff_proof": {"all_batch_tool_calls_paired": True,
            "stopped_before_next_model_call": True, "sdk_stop_returned": True},
        "session_boundary": {"iteration_exited": True, "tool_callbacks_drained": True,
            "active_session_clear": True, "session_lock_released": True}}


@pytest.fixture
def section(tmp_path, monkeypatch):
    manifest = manifest_from_text("合成桥的支座存在轻微锈蚀。\n建议对支座除锈涂漆。")
    source, directory = tmp_path / "source.json", tmp_path / "section"
    atomic_json(source, manifest)
    defect = {"section": "病害明细", "row": {"病害部位": "支座", "病害描述": "轻微锈蚀"},
        "source_refs": [{"evidence_id": manifest["units"][0]["id"]}], "status": "explicit"}
    advice = {"section": "维修建议", "row": {"病害部位": "支座", "建议内容": "除锈涂漆"},
        "source_refs": [{"evidence_id": manifest["units"][1]["id"]}], "status": "explicit"}
    request = {"manifest_path": str(source), "manifest_sha256": json_digest(manifest),
        "task": {"id": "direct-test", "primary_evidence_ids": [u["id"] for u in manifest["units"]]},
        "model_config": {"enable_thinking": False}, "direct_delivery": True,
        "deadline_monotonic": time.monotonic() + 600, "max_model_calls": 20}
    client = SimpleNamespace(calls=[], model="synthetic-own", api_key="synthetic", max_model_len=262144,
        count_tokens=lambda value: max(1, len(str(value)) // 4))

    class Gateway:
        instance = None

        def __init__(self, client, **kwargs):
            self.client, self.receipts, self.url = client, [], "http://unused.invalid/v1"
            self.messages = []
            self._forward_lock = threading.Lock()
            self.recovery_prompt = kwargs["recovery_prompt"]
            Gateway.instance = self

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def generation(self, prompt=None, *, finish_reason="tool_calls"):
            ordinal = len(self.client.calls) + 1
            row = {"kind": "agent_generation", "call_id": f"call-{ordinal}", "request_id": f"request-{ordinal}",
                "status": "success", "http_status": 200, "generation_attempted": True,
                "done_received": True, "finish_reason": finish_reason, "prompt_tokens_checked": 120,
                "usage_known": True, "usage": {"prompt_tokens": 120, "completion_tokens": 15, "total_tokens": 135}}
            if prompt is not None:
                self.messages.append({"role": "user", "content": prompt})
            payload = {"messages": deepcopy(self.messages)}
            audit = {**deepcopy(row), "request_kind": "agent_generation"}
            self.on_prepared_request(payload, audit)
            self.on_input_observed(payload, {**audit, "input_delivery_confirmed": True,
                "input_delivery_evidence": "complete_response"})
            self.client.calls.append(row)
            self.receipts.append(audit)
            self.on_receipt(audit)
            return row

    monkeypatch.setattr(worker, "AuditedModelGateway", Gateway)
    observed = {}

    def run(script, *, finish_reason="tool_calls"):
        def runner(prompt, handlers, config):
            gateway = Gateway.instance
            observed.update(prompt=prompt, handlers=handlers, config=config, gateway=gateway)
            assert config["terminal_handoff_state"]() is None
            gateway.generation(prompt, finish_reason=finish_reason)
            outcome = script(handlers, gateway, config)
            if outcome is not None:
                return outcome
            current = config["terminal_handoff_state"]()
            observed["current_handoff"] = current
            if current is not None:
                return terminal_outcome(current)
            return {"status": "completed", "native_final_event": True}
        return worker.run_section(request, directory, client=client, react_runner=runner)

    return SimpleNamespace(run=run, defect=defect, advice=advice, request=request, manifest=manifest,
        directory=directory, observed=observed, client=client)


def assert_direct_guidance(text):
    assert "submit_section" in text
    assert "save_entries" not in text and "complete_section" not in text
    assert "summary" not in text
    assert "id 是你给" not in text and "手填id" not in text


def test_only_direct_submission_and_read_tools_are_exposed_with_no_id_handoff(section):
    def script(handlers, gateway, config):
        assert set(handlers) == {"submit_section", "read_entries", "read_outline", "read_source",
            "read_section", "read_table_rows", "search_in_document", "get_neighbors"}
        assert_direct_guidance(config["system_prompt"])
        assert "新条目省略id，程序分配" in config["system_prompt"]
        assert config["enable_thinking"] is False and config["max_tokens"] == 262144
        assert config["dynamic_output_budget"] is True
        result = handlers["submit_section"]([section.defect, section.advice], [], True)
        assert result["ok"] and result["handed_off"] and len(set(result["assigned_ids"])) == 2
        assert all(a["generated"] for a in result["assignments"])
        assert config["terminal_handoff_state"]() is not None
        assert config["terminal_generation_is_settled"]()
        section.observed["submitted"] = result
    result = section.run(script)
    assert result["status"] == "completed" and result["model_calls"] == 1
    assert result["entries"]["counts"] == {"基本信息": 0, "病害明细": 1, "维修建议": 1, "total": 2}
    assert result["source_delivery"]["confirmed_ranges"] == result["entries"]["reads"]
    assert result["handoff"] == json.loads((section.directory / "handoff.json").read_text())
    assert result["handoff"]["origin_call_id"] == section.client.calls[0]["call_id"]
    assert "轻微锈蚀" not in result["handoff"]["summary"]
    assert result["native_outcome"]["native_final_event"] is False
    assert result["document_delivery_complete"] is False


def test_complete_false_saves_first_batch_and_next_generation_saves_final_batch(section):
    def script(handlers, gateway, config):
        first = handlers["submit_section"]([section.defect], [], False)
        assert first["ok"] and first["saved"] and not first["handed_off"]
        assert config["terminal_handoff_state"]() is None
        assert not (section.directory / "handoff.json").exists()
        page = handlers["read_entries"]()
        assert page["ok"] and page["entries"][0]["id"] == first["assigned_ids"][0]
        gateway.generation()
        final = handlers["submit_section"]([section.advice], ["历史对比交主Agent复核"], True)
        assert final["ok"] and final["handed_off"]
        assert final["counts"]["total"] == 2
    result = section.run(script)
    assert result["status"] == "completed" and result["model_calls"] == 2
    assert result["handoff"]["unresolved"] == ["历史对比交主Agent复核"]
    assert result["handoff"]["origin_call_id"] == section.client.calls[1]["call_id"]


@pytest.mark.parametrize("recomplete", [False, True])
def test_real_content_change_after_handoff_requires_new_handoff(section, recomplete):
    def script(handlers, gateway, config):
        first = handlers["submit_section"]([section.defect], [], True)
        old = config["terminal_handoff_state"]()
        assert old is not None
        corrected = {**deepcopy(section.defect), "id": first["assigned_ids"][0]}
        corrected["row"]["病害描述"] = "模型修订后的锈蚀描述"
        changed = handlers["submit_section"]([corrected], [], False)
        assert changed["changed"] and not changed["handed_off"]
        assert config["terminal_handoff_state"]() is None
        if recomplete:
            assert handlers["submit_section"]([], [], True)["handed_off"]
            return None
        return terminal_outcome(old)  # The worker must reject this stale success.
    result = section.run(script)
    assert (result["status"] == "completed") is recomplete
    assert (result["handoff"] is not None) is recomplete
    assert result["entries"]["counts"]["total"] == 1
    assert next(iter(result["entries"]["entries"].values()))["row"]["病害描述"] == "模型修订后的锈蚀描述"


def test_identical_implicit_replay_does_not_invalidate_current_handoff(section):
    def script(handlers, gateway, config):
        first = handlers["submit_section"]([section.defect], [], True)
        before = config["terminal_handoff_state"]()
        replay = handlers["submit_section"]([deepcopy(section.defect)], [], False)
        assert replay["ok"] and not replay["changed"]
        assert replay["assigned_ids"] == first["assigned_ids"]
        assert config["terminal_handoff_state"]() == before
    assert section.run(script)["status"] == "completed"


@pytest.mark.parametrize("invalid", ["complete", "unresolved", "entry_field"])
def test_invalid_submission_keeps_saved_bytes_and_valid_handoff(section, invalid):
    def script(handlers, gateway, config):
        assert handlers["submit_section"]([section.defect], [], True)["handed_off"]
        before = (section.directory / "entries" / "entries.json").read_bytes()
        handoff = config["terminal_handoff_state"]()
        payload = {"entries": [deepcopy(section.advice)], "unresolved": [], "complete": True}
        if invalid == "complete":
            payload["complete"] = "true"
        elif invalid == "unresolved":
            payload["unresolved"] = "[]"
        else:
            payload["entries"][0]["row"]["无效字段"] = "不应写入"
        failed = handlers["submit_section"](**payload)
        assert not failed["ok"] and not failed["saved"]
        assert failed["error"]["details"]["path"]
        assert (section.directory / "entries" / "entries.json").read_bytes() == before
        assert config["terminal_handoff_state"]() == handoff
    result = section.run(script)
    assert result["status"] == "completed" and result["entries"]["counts"]["total"] == 1


def test_continuation_and_loop_recovery_use_direct_contract_without_model_summary(section):
    def script(handlers, gateway, config):
        saved = handlers["submit_section"]([section.defect], [], False)
        feedback = config["continue_after_native_exit"]({"agent_call_id": gateway.client.calls[-1]["call_id"]})
        assert_direct_guidance(feedback)
        progress = json.loads(feedback.splitlines()[1])
        assert progress["saved_entries"]["病害明细"] == 1 and progress["current_handoff"] is False
        assert progress["primary"]["fully_read_units"] == 2
        assert "saved_entry_ids" not in progress
        assert all(u["text"] not in feedback for u in section.manifest["units"])
        recovery = gateway.recovery_prompt({"call_id": "failed-previous-call", "error_code": "GENERATION_REPETITION"})
        assert_direct_guidance(recovery)
        assert saved["assigned_ids"][0] in recovery  # Accurate saved ID, not an instruction to invent one.
        assert handlers["submit_section"]([section.advice], [], True)["handed_off"]
    result = section.run(script, finish_reason="stop")
    assert result["status"] == "completed" and result["model_calls"] == 1


def test_zero_row_direct_handoff_remains_model_declaration_not_automatic_approval(section):
    def script(handlers, gateway, config):
        done = handlers["submit_section"]([], ["本分工无条目，主Agent需核查适用范围"], True)
        assert done["handed_off"] and not done["semantic_approval"]
    result = section.run(script)
    assert result["status"] == "completed" and result["entries"]["counts"]["total"] == 0
    assert result["handoff"]["unresolved"] == ["本分工无条目，主Agent需核查适用范围"]
    assert not result["document_delivery_complete"]


def test_partial_unresolved_survives_later_failure_and_parent_import(section, tmp_path):
    from bridge_evidence.contest.document_multiagent import import_candidates
    from bridge_evidence.contest.document_workspace import DocumentWorkspace

    issue = "左右幅关系待主Agent核查"
    def script(handlers, gateway, config):
        saved = handlers["submit_section"]([section.defect], [issue], False)
        assert saved["ok"] and saved["saved"] and not saved["handed_off"]
        persisted = json.loads((section.directory / "submission-issues.json").read_text())
        assert persisted["unresolved"] == [issue] and not persisted["handoff_received"]
        raise RuntimeError("synthetic later branch failure")
    result = section.run(script)
    assert result["status"] == "failed" and result["handoff"] is None
    assert result["entries"]["counts"]["total"] == 1
    assert result["pending_unresolved"] == [issue]
    workspace = DocumentWorkspace(section.manifest, tmp_path / "main", {"test": "no-model"})
    import_candidates(workspace, [result])
    assert any(q["question"] == issue for q in workspace.snapshot()["questions"].values())


def test_final_handoff_explicit_unresolved_replaces_partial_pending_union(section):
    def script(handlers, gateway, config):
        assert handlers["submit_section"]([section.defect], ["初读待确认，随后已处理"], False)["saved"]
        assert handlers["submit_section"]([section.advice], ["最终仍待主Agent核查"], True)["handed_off"]
    result = section.run(script)
    assert result["status"] == "completed"
    assert result["handoff"]["unresolved"] == result["pending_unresolved"] == ["最终仍待主Agent核查"]
    persisted = json.loads((section.directory / "submission-issues.json").read_text())
    assert persisted["unresolved"] == ["最终仍待主Agent核查"] and persisted["handoff_received"]


@pytest.mark.parametrize("new_issue", [False, True])
def test_unresolved_only_change_invalidates_handoff_but_identical_replay_does_not(section, new_issue):
    def script(handlers, gateway, config):
        assert handlers["submit_section"]([section.defect], ["既有待核查问题"], True)["handed_off"]
        old = config["terminal_handoff_state"]()
        unresolved = ["同批又发现另一未决问题"] if new_issue else ["既有待核查问题"]
        replay = handlers["submit_section"]([section.defect], unresolved, False)
        assert replay["saved"] and not replay["changed"]
        assert (config["terminal_handoff_state"]() is None) is new_issue
        return terminal_outcome(old)  # Even valid old proof cannot hide newly submitted issues.
    result = section.run(script)
    assert (result["status"] == "completed") is (not new_issue)
    if new_issue:
        assert result["handoff"] is None
        assert result["pending_unresolved"] == ["既有待核查问题", "同批又发现另一未决问题"]


def test_same_directory_restart_restores_partial_unresolved_without_restoring_handoff(section):
    issue = "重启后仍需主Agent核查左右幅"

    def first_run(handlers, gateway, config):
        saved = handlers["submit_section"]([section.defect], [issue], False)
        assert saved["saved"] and not saved["handed_off"]
        raise RuntimeError("synthetic interruption after partial save")

    first = section.run(first_run)
    assert first["status"] == "failed" and first["pending_unresolved"] == [issue]
    issue_path = section.directory / "submission-issues.json"
    issue_bytes = issue_path.read_bytes()
    saved_entries = deepcopy(first["entries"]["entries"])
    generation_count = len(section.client.calls)

    def restarted_run(handlers, gateway, config):
        assert config["terminal_handoff_state"]() is None
        assert not (section.directory / "handoff.json").exists()
        page = handlers["read_entries"]()
        assert page["ok"]
        assert {entry["id"] for entry in page["entries"]} == set(saved_entries)
        # No new submission repeats the question: the result must come from
        # loading the previous successful batch's on-disk question state.
        raise RuntimeError("synthetic second interruption before submission")

    restarted = section.run(restarted_run)
    assert len(section.client.calls) == generation_count + 1
    assert restarted["status"] == "failed" and restarted["handoff"] is None
    assert restarted["pending_unresolved"] == [issue]
    assert restarted["entries"]["entries"] == saved_entries
    assert issue_path.read_bytes() == issue_bytes


@pytest.mark.parametrize("identity_field", ["task_id", "manifest_sha256"])
def test_restart_rejects_foreign_saved_question_identity_before_generation(section, identity_field):
    def first_run(handlers, gateway, config):
        assert handlers["submit_section"]([section.defect], ["已保存但待核查的问题"], False)["saved"]
        raise RuntimeError("synthetic interruption after partial save")

    assert section.run(first_run)["status"] == "failed"
    issue_path = section.directory / "submission-issues.json"
    state = json.loads(issue_path.read_bytes())
    state[identity_field] = "different-task" if identity_field == "task_id" else "0" * 64
    atomic_json(issue_path, state)
    issue_bytes = issue_path.read_bytes()
    entry_bytes = (section.directory / "entries" / "entries.json").read_bytes()
    generation_count = len(section.client.calls)

    def must_not_run(handlers, gateway, config):
        pytest.fail("Question identity must be checked before starting the Agent")

    with pytest.raises(ValueError, match="Saved section questions belong to another task/document"):
        section.run(must_not_run)
    assert len(section.client.calls) == generation_count
    assert issue_path.read_bytes() == issue_bytes
    assert (section.directory / "entries" / "entries.json").read_bytes() == entry_bytes
