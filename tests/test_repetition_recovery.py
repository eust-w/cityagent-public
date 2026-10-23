"""Real streaming/gateway recovery with synthetic local transport only."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import threading
import time
from unittest.mock import patch

import pytest

from bridge_evidence.contest.document_history_continuity import HistoryPrefixGuard
from bridge_evidence.contest.document_model_gateway import AuditedModelGateway, VerifiedGenerationLengthError
from bridge_evidence.contest.document_multiagent import MergeWorkflow
from bridge_evidence.contest.document_multiagent_worker import run_section
from bridge_evidence.contest.document_workflow import DocumentWorkflow
from bridge_evidence.contest.local_model_client import LocalModelClient
from bridge_evidence.contest.reliable_generation import ReliableGenerationError, StreamPolicy
from bridge_evidence.contest.run_state import atomic_json, json_digest
from test_reliable_document_gateway import Clock, SSEResponse, event, packed


def stopped(receipt, response):
    assert response.closed
    return {"status": "not_confirmed", "stop_observation": {
        "response_id": receipt["response_id"], "stopped_observed": True,
        "method": "service_idle_after_disconnect", "same_service_process": True,
        "cancellation_confirmed": False,
        "samples": [{"idle": True, "after_disconnect_seconds": 0},
                    {"idle": True, "after_disconnect_seconds": .5}],
        "model_identity_sha256": "a" * 64, "process_identity_sha256": "b" * 64}}


class TokenResponse:
    status = 200
    def __init__(self, count):
        self.count = count
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return packed({"count": self.count, "max_model_len": 262144})


class StreamSequence:
    def __init__(self, modes):
        self.modes = iter(modes)
        self.requests, self.streams = [], []
    @staticmethod
    def count(payload):
        return 23 + 17 * (len(payload["messages"]) - 1)
    def open(self, request, *, timeout):
        payload = json.loads(request.data)
        self.requests.append((request.full_url, deepcopy(payload), timeout))
        if request.full_url.endswith("/tokenize"):
            return TokenResponse(self.count(payload))
        mode = next(self.modes)
        if mode == "loop":
            raw = [event({"content": "H" * 256}), event({"content": "H" * 256})]
        elif mode == "timeout":
            raw = [TimeoutError("synthetic read timeout")]
        elif mode in {"length", "bad_length_usage"}:
            count = self.count(payload)
            maximum = payload["max_tokens"]
            raw = event({"tool_calls": [{"index": 0, "id": "partial", "type": "function",
                "function": {"name": "save_entries", "arguments": '{"entries": ["UNFINISHED'}}]})
            raw += event(finish="length") + event(usage={"prompt_tokens": count,
                "completion_tokens": maximum, "total_tokens": count + maximum + (mode == "bad_length_usage")})
            raw += b"data: [DONE]\n\n"
        else:
            count = self.count(payload)
            raw = event({"content": "已完成这一步，仍需检查文件。"}) + event(finish="stop")
            raw += event(usage={"prompt_tokens": count, "completion_tokens": 10, "total_tokens": count + 10})
            raw += b"data: [DONE]\n\n"
        stream = SSEResponse(raw, Clock())
        self.streams.append(stream)
        return stream


def route(tmp_path, modes, *, observer=stopped, recovery=True):
    deadline = time.monotonic() + 60
    client = LocalModelClient("http://127.0.0.1:9", "own-model", max_model_len=262144,
        max_output_tokens=262144, dynamic_output_budget=True, timeout=900,
        enable_thinking=False, temperature=1.0)
    client.enable_reliable_generation(StreamPolicy(), deadline=deadline,
        on_cancel=observer, capture_directory=tmp_path / "raw-sse")
    client._opener = StreamSequence(modes)
    feedback = []
    def progress(receipt):
        feedback.append(deepcopy(receipt))
        return '程序准确状态：已保存版本7、条目d1；下一步只检查维修建议表，然后finish。'
    gateway = AuditedModelGateway(client, deadline=deadline, max_calls=10,
        max_output_tokens=262144, recovery_prompt=progress if recovery else None)
    payload = gateway.prepare_request({"model": "own-model", "temperature": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "合成报告"}]})
    return client, gateway, payload, feedback


def test_stopped_loop_recovers_once_with_fresh_count_and_clean_sdk_history(tmp_path):
    client, gateway, payload, feedback = route(tmp_path, ["loop", "success", "loop"])
    original = deepcopy(payload)
    notifications = []
    gateway.on_receipt = lambda audit: notifications.append(deepcopy(audit))
    guard = HistoryPrefixGuard()
    guard.observe_complete_request(payload)
    response = gateway.forward(payload)
    assert payload == original
    generations = [c for c in client.calls if c["kind"] == "agent_generation"]
    assert len(generations) == 2 and generations[0]["recovery_authorized"] is True
    assert generations[0]["cancel_status"] == "not_confirmed"
    assert not generations[0]["done_received"] and not generations[0]["usage_known"]
    assert notifications[0]["recovery_authorized"] is True
    assert notifications[0]["stop_observation"]["stopped_observed"] is True
    assert notifications[1]["recovery_of_call_id"] == generations[0]["call_id"]
    wire = [p for url, p, _ in client._opener.requests if url.endswith("/chat/completions")]
    assert wire[0]["max_tokens"] == 262144 - 23
    assert wire[1]["max_tokens"] == 262144 - 40
    assert wire[1]["messages"][:-1] == payload["messages"]
    assert "版本7" in wire[1]["messages"][-1]["content"]
    assert "H" * 100 not in json.dumps(wire[1])
    assert generations[0]["raw_sse_capture"]["path"]
    assert len(feedback) == 1 and all(s.closed for s in client._opener.streams)
    # SDK keeps the original closed history and sees only the successful
    # recovery result. The temporary wire feedback is not claimed as memory.
    later = deepcopy(payload)
    later["messages"].append(response["choices"][0]["message"])
    guard.observe_complete_request(later)
    later = gateway.prepare_request(later)
    with pytest.raises(ReliableGenerationError) as failure:
        gateway.forward(later)
    assert failure.value.code == "content_repetition"
    assert len(feedback) == 1
    assert len([c for c in client.calls if c["kind"] == "agent_generation"]) == 3
    assert not failure.value.receipt.get("recovery_authorized")


@pytest.mark.parametrize("modes,observer,recovery", [
    (["loop"], None, True), (["loop"], stopped, False), (["timeout"], None, True)])
def test_unknown_stop_unconfigured_recovery_and_nonloop_never_retry(tmp_path, modes, observer, recovery):
    client, gateway, payload, feedback = route(tmp_path, modes, observer=observer, recovery=recovery)
    with pytest.raises(ReliableGenerationError):
        gateway.forward(payload)
    assert not feedback and not getattr(client, "_repetition_recoveries_used", 0)
    assert len([c for c in client.calls if c["kind"] == "agent_generation"]) == 1


def test_second_loop_in_recovery_exits_without_third_generation(tmp_path):
    client, gateway, payload, feedback = route(tmp_path, ["loop", "loop"])
    with pytest.raises(ReliableGenerationError):
        gateway.forward(payload)
    assert len(feedback) == 1
    assert len([c for c in client.calls if c["kind"] == "agent_generation"]) == 2
    assert not gateway.receipts[-1].get("recovery_authorized")


def test_main_output_cap_is_independent_of_context_and_survives_length_recovery(tmp_path):
    client, gateway, payload, feedback = route(tmp_path, ["length", "success", "length"])
    client.generation_output_cap = 1024
    payload = gateway.prepare_request(payload)
    assert payload["max_tokens"] == 1024
    assert gateway.last_prepared_allocation["actual_input_tokens"] == 23
    accepted = []
    notifications = []
    gateway.on_response = lambda result, audit: accepted.append(deepcopy(result))
    gateway.on_receipt = lambda audit: notifications.append(deepcopy(audit))
    result = gateway.forward(payload)
    assert accepted == [result]  # No truncated tool parameters reach SDK execution.
    wire = [p for url, p, _ in client._opener.requests if url.endswith("/chat/completions")]
    assert [p["max_tokens"] for p in wire] == [1024, 1024]
    assert "UNFINISHED" not in json.dumps(wire[1])
    assert wire[1]["messages"][:-1] == payload["messages"]
    assert gateway.last_prepared_allocation["actual_input_tokens"] == 40
    assert feedback[0]["error_code"] == "output_token_limit"
    assert notifications[0]["length_usage_verified"] is True
    assert notifications[0]["finish_reason"] == "length"
    assert notifications[0]["recovery_authorized"] is True
    assert "cancellation" not in notifications[0]  # Normal terminal, not cancellation.
    workflow = object.__new__(DocumentWorkflow)
    workflow.settings = SimpleNamespace(single_document_agent=True)
    workflow.model_dispatches = []
    workflow._persist = lambda: None
    workflow._on_model_receipt(notifications[0])
    assert getattr(workflow, "_single_agent_terminal_failure", None) is None
    with pytest.raises(VerifiedGenerationLengthError):
        gateway.forward(payload)
    assert len(feedback) == 1
    assert len(accepted) == 1
    assert len(gateway.verified_length_attempts) == 2


@pytest.mark.parametrize("capped,mode", [(False, "length"), (True, "bad_length_usage")])
def test_uncapped_or_unverified_length_cannot_authorize_recovery(tmp_path, capped, mode):
    client, gateway, payload, feedback = route(tmp_path, [mode])
    if capped:
        client.generation_output_cap = 1024
        payload = gateway.prepare_request(payload)
    with pytest.raises((ValueError, ReliableGenerationError)):
        gateway.forward(payload)
    assert feedback == []
    assert not getattr(client, "_repetition_recoveries_used", 0)


def test_output_cap_is_recounted_and_cannot_override_available_context_or_gateway(tmp_path):
    from bridge_evidence.contest.document_context_budget import remaining_output
    assert remaining_output(262144, 23, 1024) == 1024
    assert remaining_output(262144, 262140, 1024) == 4
    assert remaining_output(262144, 23) == 262121
    for cap in (0, -1, True, 1.5, "1024"):
        with pytest.raises(ValueError):
            remaining_output(262144, 23, cap)
    client, gateway, payload, _ = route(tmp_path, [])
    client.generation_output_cap = 1024
    with pytest.raises(ValueError, match="max_tokens"):
        gateway.forward(payload)  # Stale uncapped SDK allocation is rejected before generation.
    assert not [c for c in client.calls if c["kind"] == "agent_generation"]


def test_primary_receipt_allows_only_explicit_stopped_loop_recovery(tmp_path):
    client, gateway, payload, _ = route(tmp_path, ["loop", "success"])
    gateway.forward(payload)
    stopped_audit = gateway.receipts[0]
    for mutation, allowed in [(lambda r: None, True),
        (lambda r: r.pop("recovery_authorized"), False),
        (lambda r: r.update(response_id="different-request"), False),
        (lambda r: r.update(error_code="wall_or_read_timeout"), False)]:
        workflow = object.__new__(DocumentWorkflow)
        workflow.settings = SimpleNamespace(single_document_agent=True)
        workflow.model_dispatches = []
        workflow._persist = lambda: None
        audit = deepcopy(stopped_audit)
        mutation(audit)
        workflow._on_model_receipt(audit)
        assert (getattr(workflow, "_single_agent_terminal_failure", None) is None) is allowed


def test_merge_recovery_uses_current_file_pending_action_without_stalled_side_effect(tmp_path):
    workflow = object.__new__(MergeWorkflow)
    state = {"version": 7, "draft_sha256": "d", "draft": {"基本信息": {}, "病害明细": [], "维修建议": []},
        "facts": {"F1": {}}, "questions": {}, "working_notes": {"text": "建议表待读"}}
    workflow.workspace = SimpleNamespace(snapshot=lambda: deepcopy(state), delivery_status=lambda: {"ready": True})
    workflow._coverage_pending = lambda state: []
    workflow.artifact_delivery = {"workspace_version": 7, "draft_sha256": "d", "file_sha256": "f",
        "structure_valid": True, "content_validated": True}
    workflow.delivery_inspections = [{"workspace_version": 7, "draft_sha256": "d", "file_sha256": "f",
        "view": view, "cursor": 0, "end": 20, "total_chars": 20} for view in ("basic", "defects")]
    workflow.model_self_check = None
    workflow.merge_continuations = []
    workflow.directory = tmp_path
    text = workflow._generation_recovery_prompt({"call_id": "failed", "error_code": "content_repetition"})
    saved = json.loads((tmp_path / "generation-recovery.json").read_bytes())
    action = saved["saved_progress"]["next_action"]
    assert action == {"tool": "inspect_delivery", "arguments": {"view": "advice", "limit": 8}}
    assert saved["saved_fact_ids"] == ["F1"] and saved["working_notes"] == state["working_notes"]
    assert not workflow.merge_continuations and "唯一一次恢复" in text


@pytest.mark.parametrize("observer_seconds", [20, 60.1])
def test_recovery_does_not_reset_deadline_or_double_count_failed_attempt(tmp_path, observer_seconds):
    clock = Clock()
    def observe(receipt, response):
        value = stopped(receipt, response)
        clock.now += observer_seconds
        return value
    with patch("bridge_evidence.contest.document_model_gateway.time.monotonic", clock):
        client, gateway, payload, feedback = route(tmp_path, ["loop", "success"], observer=observe)
        client.reliable_generation_policy = replace(client.reliable_generation_policy, clock=clock)
        if observer_seconds > 60:
            with pytest.raises(TimeoutError):
                gateway.forward(payload)
            assert not feedback and not getattr(client, "_repetition_recoveries_used", 0)
            assert len([c for c in client.calls if c["kind"] == "agent_generation"]) == 1
        else:
            gateway.forward(payload)
            assert gateway.receipts[0]["seconds"] == observer_seconds
            assert gateway.receipts[1]["seconds"] == 0
            assert gateway.receipts[1]["total_forward_seconds"] == observer_seconds
            requests = [r for r in client._opener.requests if r[0].endswith("/chat/completions")]
            assert requests[1][2] <= 60 - observer_seconds


def settled_recovery_history():
    failed = {"kind": "agent_generation", "call_id": "failed-call", "request_id": "failed-request",
        "response_id": "failed-response", "status": "failed", "http_status": 200,
        "generation_attempted": True, "done_received": False, "usage_known": False,
        "cancel_status": "not_confirmed", "error_code": "content_repetition", "recovery_authorized": True}
    failed["stop_observation"] = stopped(failed, SimpleNamespace(closed=True))["stop_observation"]
    success = {"kind": "agent_generation", "call_id": "success-call", "request_id": "success-request",
        "status": "success", "http_status": 200, "generation_attempted": True, "done_received": True,
        "usage_known": True, "finish_reason": "tool_calls", "prompt_tokens_checked": 20,
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}
    audits = [{**deepcopy(failed), "request_kind": "agent_generation", "status": "rejected"},
              {**deepcopy(success), "request_kind": "agent_generation"}]
    return [failed, success], audits


def test_main_parse_continuation_accepts_stopped_old_loop_but_not_unknown_generation():
    calls, audits = settled_recovery_history()
    workflow = object.__new__(DocumentWorkflow)
    workflow.client = SimpleNamespace(calls=calls, max_model_len=262144)
    workflow.gateway_receipts = audits
    workflow.model_dispatches = [{"status": "settled"}, {"status": "settled"}]
    workflow._active_agent_call_id = "success-call"
    workflow._single_agent_terminal_failure = None
    workflow._model_forward_lock = threading.Lock()
    assert workflow._parse_recovery_generation_is_settled() is True
    calls[0].pop("stop_observation")
    assert workflow._parse_recovery_generation_is_settled() is False


def test_child_recovery_can_save_and_complete_handoff_with_original_entries(tmp_path, monkeypatch):
    from bridge_evidence.contest import document_multiagent_worker as worker
    from test_document_multiagent import manifest_from_text
    manifest = manifest_from_text("合成桥。\n第二段检测说明。")
    unit = manifest["units"][0]
    source = tmp_path / "source.json"
    atomic_json(source, manifest)
    calls, audits = settled_recovery_history()
    client = SimpleNamespace(calls=calls, model="synthetic", api_key="synthetic", max_model_len=262144,
        count_tokens=lambda x: len(str(x)))
    request = {"manifest_path": str(source), "manifest_sha256": json_digest(manifest),
        "task": {"id": "part1", "primary_evidence_ids": [u["id"] for u in manifest["units"]]}, "model_config": {},
        "deadline_monotonic": time.monotonic() + 60, "max_model_calls": 8}
    instances = []
    class Gateway:
        def __init__(self, client, **kwargs):
            self.receipts = deepcopy(audits)
            self.url = "http://unused"
            self._forward_lock = threading.Lock()
            self.recovery_prompt = kwargs["recovery_prompt"]
            instances.append(self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(worker, "AuditedModelGateway", Gateway)
    def react(prompt, handlers, config):
        gateway = instances[-1]
        def delivered(payload, request_id):
            audit = {"request_id": request_id, "call_id": "success-call",
                     "input_delivery_confirmed": True, "http_status": 200}
            gateway.on_prepared_request(payload, audit)
            gateway.on_input_observed(payload, audit)
        delivered({"messages": [{"role": "user", "content": prompt}]}, "initial-request")
        gateway.on_receipt(audits[-1])
        saved = handlers["save_entries"]([{"id": "name", "section": "基本信息", "field": "桥梁名称",
            "value": "合成桥", "source_refs": [{"evidence_id": unit["id"]}], "status": "explicit"}])
        assert saved["ok"]
        page = handlers["read_section"]("root", max_units=1)
        assert page["continuation"]["arguments"] == {"section_id": "root", "max_units": 1}
        delivered({"messages": [{"role": "tool", "content": json.dumps(page, ensure_ascii=False)}]}, "page-request")
        handlers["search_in_document"](["合成"])
        handlers["read_outline"]()
        feedback = gateway.recovery_prompt(calls[0])
        assert '"saved_entry_ids":["name"]' in feedback
        assert '"fully_read_units":2' in feedback
        recovery = json.loads((tmp_path / "section" / "generation-recovery.json").read_bytes())
        navigation = recovery["saved_progress"]["last_source_navigation"]
        assert navigation["read_section:root"]["continuation"] == page["continuation"]
        last_page = handlers["read_section"]("root", max_units=1)
        assert last_page["continuation"] is None
        recorded = json.loads((tmp_path / "section" / "source-navigation.json").read_bytes())
        assert recorded["read_section:root"]["continuation"] is None
        assert handlers["complete_section"]("本分工已读取并保存，语义仍待主Agent核对。", [])["ok"]
        assert config["terminal_generation_is_settled"]() is True
        handoff = config["terminal_handoff_state"]()
        return {"status": "completed", "native_final_event": False, "termination_kind": "section_terminal_handoff",
            "terminal_handoff_sha256": json_digest(handoff), "terminal_handoff_origin_call_id": "success-call",
            "terminal_handoff_proof": {"all_batch_tool_calls_paired": True, "stopped_before_next_model_call": True,
                "sdk_stop_returned": True}, "session_boundary": {"iteration_exited": True, "tool_callbacks_drained": True,
                    "active_session_clear": True, "session_lock_released": True}}
    result = run_section(request, tmp_path / "section", client=client, react_runner=react)
    assert result["status"] == "completed" and result["generation_attempts_settled"] is True
    assert result["entries"]["entry_order"] == ["name"] and result["model_calls"] == 2
    assert result["entries"]["reads"][unit["id"]] == [[0, len(unit["text"])]]
