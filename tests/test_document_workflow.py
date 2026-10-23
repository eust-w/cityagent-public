"""Workflow contracts with synthetic text and deterministic local doubles.

These tests exercise acceptance, provenance and recovery, not model accuracy.
No SDK process, HTTP listener, remote model or real report is used.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import time
from unittest.mock import patch

import pytest

from bridge_evidence.contest import document_workflow as module
from bridge_evidence.contest.document_model_gateway import AuditedModelGateway
from bridge_evidence.contest.document_workspace import WorkspaceIdentityError
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import json_digest


class SyntheticNativeGateway(AuditedModelGateway):
    """Real admission/persistence with an in-memory SDK transport boundary."""
    active = []

    @property
    def url(self):
        return "http://synthetic-gateway.invalid/v1"

    def __enter__(self):
        self.active.append(self)
        return self

    def __exit__(self, *_args):
        assert self.active.pop() is self
        self._accepting_requests = False
        return False


def bind_synthetic_transport(client):
    if getattr(client, "_synthetic_transport_bound", False):
        return
    original = client._request

    def request(path, payload, *, kind):
        result, receipt = original(path, payload, kind=kind)
        result.setdefault("model", client.model)
        result.setdefault("id", "synthetic-response-" + receipt["call_id"])
        usage = result.get("usage")
        if isinstance(usage, dict) and "prompt_tokens" in usage and "completion_tokens" in usage:
            usage.setdefault("total_tokens", usage["prompt_tokens"] + usage["completion_tokens"])
        receipt.update(endpoint=path, http_status=200, input_sha256=json_digest(payload),
                       response_sha256=json_digest(result), response_id=result["id"],
                       output_sha256=json_digest(result), usage=deepcopy(usage))
        return result, receipt

    client._request = request
    client._synthetic_transport_bound = True


def native_review_driver(prompt, handlers, config):
    from bridge_evidence.contest.document_review_lookup import LOOKUP_TOOLS
    gateway = SyntheticNativeGateway.active[-1]
    messages = [{"role": "system", "content": config["system_prompt"]},
                {"role": "user", "content": prompt}]
    calls = []
    for _ in range(config["max_iterations"]):
        payload = {"model": gateway.client.model, "messages": deepcopy(messages),
                   "tools": deepcopy(LOOKUP_TOOLS), "tool_choice": "auto", "stream": False,
                   "max_tokens": config["max_tokens"], "temperature": config["temperature"],
                   "chat_template_kwargs": {"enable_thinking": config["enable_thinking"]}}
        result = gateway.forward(payload)
        choice = result["choices"][0]
        message = choice["message"]
        messages.append(deepcopy(message))
        if not message.get("tool_calls"):
            return completed(calls, profile="reviewer", response_text=message["content"])
        for tool in message["tool_calls"]:
            if len(calls) >= config["max_calls"]:
                raise ValueError("Native tool budget exhausted before another lookup")
            name = tool["function"]["name"]
            arguments = json.loads(tool["function"]["arguments"])
            call_id = config["current_agent_call_id"]()
            if name in handlers:
                result = handlers[name](**arguments)
                status = "success"
            else:
                result = {"ok": False, "error": "Unregistered readonly review tool"}
                status = "failed"
            calls.append({"name": name, "args": arguments, "args_sha256": json_digest(arguments),
                          "status": status, "result": result, "agent_call_id": call_id})
            messages.append({"role": "tool", "tool_call_id": tool["id"],
                             "content": json.dumps(result, ensure_ascii=False)})
    return {**completed(calls, profile="reviewer"), "status": "failed", "native_final_event": False,
            "failure_reason": "iteration_limit"}


def install_native_review_driver(wf, *, synthetic_transport=True):
    """Only replace the SDK runner; all review admission and evidence checks run."""
    if synthetic_transport:
        bind_synthetic_transport(wf.client)
    original = wf._review_request

    def review(payload):
        runner = wf.react_runner
        wf.react_runner = native_review_driver
        try:
            with patch.object(module, "AuditedModelGateway", SyntheticNativeGateway):
                return original(payload)
        finally:
            wf.react_runner = runner

    wf._review_request = review
    return wf


class FakeModel:
    model = "synthetic-own-model"
    base_url = "http://unused.invalid"
    max_model_len = 65536
    timeout = 180.0

    def __init__(self, decide=None):
        self.calls = []
        self.payloads = []
        self.decide = decide
        self.before_reply = None
        self.response_overrides = {}

    @staticmethod
    def count_tokens(text):
        return max(1, (len(text) + 3) // 4)

    def count_chat_tokens(self, messages):
        return self.count_tokens(json.dumps(messages, ensure_ascii=False)) + 12

    def _tokenize(self, payload):
        return self.count_chat_tokens(payload["messages"])

    def _request(self, path, payload, *, kind):
        assert path == "/v1/chat/completions"
        assert kind == "generation"
        response = self.generate(payload["messages"], max_tokens=payload["max_tokens"])
        receipt = self.calls[-1]
        result = {"choices": [{"finish_reason": response["finish_reason"],
                                "message": {"role": "assistant", "content": response["text"]}}],
                  "usage": {"prompt_tokens": self._tokenize(payload), "completion_tokens": 30}}
        return result, receipt

    def generate(self, messages, *, max_tokens):
        payload = json.loads(messages[-1]["content"])
        self.payloads.append(deepcopy(payload))
        call_id = f"review-{len(self.calls) + 1}"
        self.calls.append({"kind": "generation", "call_id": call_id, "status": "success"})
        decision = {"passed": True,
                    "checked_output_paths": [item["path"] for item in payload.get("output_items", [])],
                    "checked_evidence_spans": deepcopy(payload.get("source_spans", [])),
                    "unsupported_claims": [], "missing_items": [], "conflicts": [], "unresolved_items": []}
        if self.decide:
            decision.update(self.decide(payload))
        if self.before_reply:
            self.before_reply(payload)
        return {"status": "success", "finish_reason": "stop", "text": json.dumps(decision, ensure_ascii=False),
                "call_id": call_id, **self.response_overrides}


@pytest.fixture
def fake_gateway(monkeypatch):
    class Gateway(AuditedModelGateway):
        instances = []

        def __init__(self, client, **kwargs):
            super().__init__(client, **kwargs)
            self.instances.append(self)

        @property
        def url(self):
            return "http://fake-gateway.invalid/v1"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self._accepting_requests = False
            return False

        def observe_prompt(self, prompt=None, *, tool_calls=None):
            payload = {"model": self.client.model, "messages": [{"role": "user", "content": prompt or "continue"}],
                       "stream": False, "max_tokens": 32, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}
            def transport(path, actual, *, kind):
                identifier = f"agent-{len(self.client.calls) + 1}"
                message = {"role": "assistant", "content": "synthetic reader"}
                if tool_calls:
                    message["tool_calls"] = deepcopy(tool_calls)
                usage = {"prompt_tokens": self.client._tokenize(actual), "completion_tokens": 1}
                usage["total_tokens"] = sum(usage.values())
                result = {"id": "response-" + identifier, "model": self.client.model,
                          "choices": [{"finish_reason": "tool_calls" if tool_calls else "stop", "message": message}],
                          "usage": usage}
                receipt = {"call_id": identifier, "kind": kind, "status": "success", "http_status": 200,
                           "input_sha256": json_digest(actual), "response_sha256": json_digest(result),
                           "response_id": result["id"], "output_sha256": json_digest(result), "usage": usage}
                self.client.calls.append(receipt)
                return result, receipt
            with patch.object(self.client, "_request", transport):
                return self.forward(payload)

    monkeypatch.setattr(module, "AuditedModelGateway", Gateway)
    return Gateway


def review_tool_call():
    return {"id": "request-review", "type": "function", "function": {"name": "review_draft", "arguments": "{}"}}


def workflow(tmp_path, *, client=None, text="桥梁名称：合成桥\n本次左幅1号梁有裂缝。\n", manifest=None, settings=None, runner=None, **kwargs):
    return install_native_review_driver(module.DocumentWorkflow(
        manifest if manifest is not None else manifest_from_text(text),
        client or FakeModel(), {"model": FakeModel.model, "timeout": 180}, tmp_path,
        settings=settings or module.DocumentSettings(max_sessions=30, max_model_calls=30),
        react_runner=runner or (lambda *_: completed()), **kwargs))


def completed(tool_calls=None, **kwargs):
    from bridge_evidence.contest.tong_react_adapter import session_boundary
    boundary = session_boundary()
    return {"status": "completed", "native_final_event": True, "tool_calls": tool_calls or [],
            "session_boundary": {"iteration_exited": True, "tool_callbacks_drained": True,
                                 "active_session_clear": boundary["active_session_clear"],
                                 "session_lock_released": boundary["session_lock_available"]}, **kwargs}


def source_fact(wf, identifier="name", unit_index=0, **kwargs):
    unit = wf.manifest["units"][unit_index]
    return {"id": identifier, "value": "合成桥", "status": "explicit", "depends_on": [],
            "source_refs": [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}], **kwargs}


def read(wf, indices=None):
    indices = range(len(wf.manifest["units"])) if indices is None else indices
    for index in indices:
        unit = wf.manifest["units"][index]
        wf.workspace.record_read(unit["id"], 0, len(unit["text"]))


def prepare(wf, *, facts=None, **kwargs):
    return wf.workspace.apply_update({
        "base_version": wf.workspace.snapshot()["version"],
        "facts": facts if facts is not None else [source_fact(wf)],
        "draft": {"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [], "病害明细": []},
        "draft_fact_ids": ["name"], "draft_bindings": {"基本信息.桥梁名称": ["name"]}, **kwargs})


def ready(wf):
    read(wf)
    prepare(wf)
    result = wf._perform_review_draft()
    assert result["delivery"]["ready"]


def test_native_session_needs_read_review_and_final_finish(tmp_path, fake_gateway):
    wf = workflow(tmp_path)

    def runner(prompt, handlers, config):
        assert config["url"] == fake_gateway.instances[-1].url
        gateway = fake_gateway.instances[-1]
        if wf.workspace.delivery_status()["ready"]:
            gateway.observe_prompt(prompt)
            final = handlers["finish"]()
            assert final["ok"]
            return completed([{"name": "finish", "status": "success", "result": final}])
        assert "合成桥" in prompt
        gateway.observe_prompt(prompt, tool_calls=[review_tool_call()])
        state = handlers["read_workspace"]()
        assert len(state["reads"]) == 2
        payload = {"base_version": state["version"], "facts": [source_fact(wf)],
                   "draft_patch": [{"op": "set_basic", "field": "桥梁名称", "value": "合成桥", "fact_ids": ["name"]}]}
        handlers["update_workspace"](update_json=json.dumps(payload, ensure_ascii=False))
        queued = handlers["review_draft"]()
        assert queued["queued"] and not wf.workspace.delivery_status()["ready"]
        assert not wf.review_sessions  # No nested native reviewer.
        return completed([{"name": "review_draft", "status": "success", "result": queued}])

    wf.react_runner = runner
    result = wf.run()
    assert result["data"]["基本信息"]["桥梁名称"] == "合成桥"
    assert result["receipt"]["delivery"]["ready"]
    assert result["receipt"]["agent_generation_requests"] == 2
    assert {payload["direction"] for payload in wf.client.payloads} == {"FORWARD", "REVERSE"}
    stored = wf.workspace.snapshot()
    assert stored["review"]["draft_sha256"] == stored["draft_sha256"]
    assert stored["review"]["recorded_version"] == stored["version"]


def test_final_text_and_a_bound_draft_do_not_replace_semantic_review(tmp_path, fake_gateway):
    wf = workflow(tmp_path)

    def runner(*_):
        read(wf)
        prepare(wf)
        final = wf._finish()
        assert not final["ok"]
        return completed([{"name": "finish", "status": "success", "result": final}])

    wf.react_runner = runner
    with pytest.raises(module.DocumentWorkflowFailure) as caught:
        wf.run()
    assert not caught.value.receipt["delivery"]["ready"]
    assert caught.value.receipt["review_generation_requests"] == 0


def test_full_text_admission_does_not_claim_the_model_read_it(tmp_path, fake_gateway):
    wf = workflow(tmp_path, runner=lambda *_: completed())
    with pytest.raises(module.DocumentWorkflowFailure):
        wf.run()
    assert wf.workspace.snapshot()["reads"] == {}
    assert wf.client.calls == []


def test_full_read_receipt_requires_the_source_in_actual_sdk_messages(tmp_path, fake_gateway):
    wf = workflow(tmp_path)

    def runner(*_):
        # The request succeeded, but an SDK/request bug removed the document.
        fake_gateway.instances[-1].observe_prompt("仅有一句继续，没有报告正文")
        return completed()

    wf.react_runner = runner
    with pytest.raises(module.DocumentWorkflowFailure):
        wf.run()
    assert wf.workspace.snapshot()["reads"] == {}


def test_forward_unsupported_claim_survives_review_and_blocks_finish(tmp_path):
    issue = {"path": "基本信息.桥梁名称", "reason": "输出写另一座桥，给出的报告原文是合成桥", "source_refs": ["E000001"]}
    client = FakeModel(lambda payload: {"passed": False, "unsupported_claims": [issue]} if payload["direction"] == "FORWARD" else {})
    wf = workflow(tmp_path, client=client)
    read(wf)
    prepare(wf, draft={"基本信息": {"桥梁名称": "另一座桥"}, "维修建议": [], "病害明细": []})
    result = wf._perform_review_draft()
    forward = next(payload for payload in client.payloads if payload["direction"] == "FORWARD")
    assert forward["output_items"][0]["value"] == {"桥梁名称": "另一座桥"}
    assert forward["output_items"][0]["sources"][0]["text"] == wf.manifest["units"][0]["text"]
    assert result["review"]["unsupported_claims"] == [issue]
    assert not wf._finish()["ok"]


def test_reverse_review_sees_an_unbound_defect_even_when_agent_excluded_it(tmp_path):
    issue = {"evidence_id": "E000002", "reason": "原文的左幅1号梁裂缝未进入病害表", "source_refs": ["E000002"]}

    def decide(payload):
        if payload["direction"] == "REVERSE":
            assert "本次左幅1号梁有裂缝" in payload["source"]
            assert all(item["path"] != "病害明细.0" for item in payload["related_output"])
            return {"passed": False, "missing_items": [issue]}
        return {}

    wf = workflow(tmp_path, client=FakeModel(decide))
    read(wf, [0])
    prepare(wf, exclusions=[{"evidence_id": "E000002", "start": 0, "end": len(wf.manifest["units"][1]["text"]), "reason": "Agent误判与任务无关"}])
    result = wf._perform_review_draft()
    assert result["review"]["missing_items"] == [issue]
    assert not result["delivery"]["ready"]
    # The independent model actually received the excluded text; only then
    # may its read receipt be added, without erasing the exclusion's history.
    state = wf.workspace.snapshot()
    assert state["reads"]["E000002"] == [[0, len(wf.manifest["units"][1]["text"])]]
    assert state["exclusions"]


def test_reverse_reads_are_saved_before_review_binds_its_final_version(tmp_path):
    wf = workflow(tmp_path)
    read(wf, [0])
    prepare(wf)
    version = wf.workspace.snapshot()["version"]
    result = wf._perform_review_draft()
    state = wf.workspace.snapshot()
    assert state["version"] > version + 1
    assert state["review"]["workspace_version"] == state["version"] - 1
    assert result["delivery"]["ready"]
    assert state["review"]["checked_evidence_ids"] == ["E000001", "E000002"]


def test_long_source_is_reviewed_in_bounded_batches_without_losing_its_tail(tmp_path):
    text = "桥梁名称：合成桥\n" + "合成版式占位文字。" * 6000 + "END_OF_SYNTHETIC_SECTION"
    wf = workflow(tmp_path, text=text)
    read(wf, [0])
    prepare(wf)
    assert wf._perform_review_draft()["delivery"]["ready"]
    reverse = [payload for payload in wf.client.payloads if payload["direction"] == "REVERSE"]
    assert len(reverse) > 1
    intervals = sorted((span["start"], span["end"]) for payload in reverse for span in payload["source_spans"]
                       if span["evidence_id"] == "E000002")
    cursor = 0
    for start, end in intervals:
        assert start == cursor
        assert end > start
        cursor = end
    assert cursor == len(wf.manifest["units"][1]["text"])
    assert "END_OF_SYNTHETIC_SECTION" in reverse[-1]["source"]
    assert wf.workspace.snapshot()["reads"]["E000002"] == [[0, cursor]]


def test_relation_endpoint_evidence_reaches_both_review_directions(tmp_path):
    wf = workflow(tmp_path)
    read(wf)
    prepare(wf, facts=[source_fact(wf), source_fact(wf, "damage", 1, value="左幅1号梁裂缝")],
            relations=[{"id": "located", "subject_id": "damage", "object_id": "name", "predicate": "属于", "source_refs": [], "depends_on": []}],
            draft={"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [], "病害明细": [{"病害部位": "左幅1号梁", "病害描述": "裂缝"}]},
            draft_fact_ids=["name", "located"], draft_bindings={"基本信息.桥梁名称": ["name"], "病害明细.0": ["located"]})
    assert wf._perform_review_draft()["delivery"]["ready"]
    forward = next(payload for payload in wf.client.payloads if payload["direction"] == "FORWARD")
    damage = next(item for item in forward["output_items"] if item["path"] == "病害明细.0")
    assert {source["evidence_id"] for source in damage["sources"]} == {"E000001", "E000002"}
    reverse = next(payload for payload in wf.client.payloads if payload["direction"] == "REVERSE")
    assert "病害明细.0" in {item["path"] for item in reverse["related_output"]}


def test_same_text_draft_rebinding_during_review_cannot_reuse_old_judgement(tmp_path):
    wf = workflow(tmp_path)
    read(wf)
    prepare(wf, facts=[source_fact(wf), source_fact(wf, "alternative", 1)])

    def rebind_once(_payload):
        wf.client.before_reply = None
        state = wf.workspace.snapshot()
        wf.workspace.apply_update({"base_version": state["version"], "draft": state["draft"],
                                   "draft_fact_ids": ["alternative"], "draft_bindings": {"基本信息.桥梁名称": ["alternative"]}})

    wf.client.before_reply = rebind_once
    with pytest.raises(RuntimeError, match="Native reviewer request failed"):
        wf._perform_review_draft()
    assert wf.gateway_receipts[-1]["status"] != "success"
    assert not wf.workspace.delivery_status()["ready"]
    assert wf.workspace.snapshot()["review"] is None


def test_resolved_question_dependencies_are_retained_as_review_context(tmp_path):
    wf = workflow(tmp_path)
    read(wf)
    prepare(wf, facts=[source_fact(wf, depends_on=["identity-question"])],
            questions=[{"id": "identity-question", "question": "是否同一座桥", "status": "resolved", "reason": "已核对报告桥名和构件章节"}])
    assert wf._perform_review_draft()["delivery"]["ready"]
    forward = next(payload for payload in wf.client.payloads if payload["direction"] == "FORWARD")
    records = forward["output_items"][0]["facts"]
    assert any(item["id"] == "identity-question" and item["reason"] == "已核对报告桥名和构件章节" for item in records)


def test_edit_after_successful_review_requires_a_new_review(tmp_path):
    wf = workflow(tmp_path)
    ready(wf)
    state = wf.workspace.snapshot()
    wf.workspace.apply_update({"base_version": state["version"],
                               "facts": [{"id": "name", "value": "更正后的桥"}],
                               "draft": {"基本信息": {"桥梁名称": "更正后的桥"}, "维修建议": [], "病害明细": []},
                               "draft_fact_ids": ["name"], "draft_bindings": {"基本信息.桥梁名称": ["name"]}})
    assert not wf._finish()["ok"]
    assert wf.workspace.snapshot()["review"] is None


@pytest.mark.parametrize("direction", ["FORWARD", "REVERSE"])
def test_missing_review_assignment_receipts_do_not_count_as_semantic_pass(tmp_path, direction):
    def decide(payload):
        if payload["direction"] != direction:
            return {}
        return {"checked_output_paths": []} if direction == "FORWARD" else {"checked_evidence_spans": []}

    wf = workflow(tmp_path, client=FakeModel(decide))
    read(wf)
    prepare(wf)
    with pytest.raises(ValueError, match="account for every"):
        wf._perform_review_draft()
    assert wf.workspace.snapshot()["review"] is None
    assert not wf._finish()["ok"]


def test_truncated_review_is_not_a_conclusion_and_timeout_is_restored(tmp_path):
    client = FakeModel()
    client.response_overrides["finish_reason"] = "length"
    wf = workflow(tmp_path, client=client)
    read(wf)
    prepare(wf)
    timeout = client.timeout
    with pytest.raises(ValueError, match="complete own-model|incomplete or truncated"):
        wf._perform_review_draft()
    assert client.timeout == timeout
    assert wf.workspace.snapshot()["review"] is None


def test_last_review_reply_after_deadline_cannot_publish_a_pass(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    wf = workflow(tmp_path)
    read(wf)
    prepare(wf)

    def expire_on_reverse(payload):
        if payload["direction"] == "REVERSE":
            clock[0] = wf.deadline + 1

    wf.client.before_reply = expire_on_reverse
    with pytest.raises(TimeoutError):
        wf._perform_review_draft()
    assert not wf.workspace.delivery_status()["ready"]
    assert wf.workspace.snapshot()["review"] is None


def test_finish_called_too_early_does_not_authorize_later_completion(tmp_path, fake_gateway):
    wf = workflow(tmp_path, settings=module.DocumentSettings(max_sessions=4, max_model_calls=30))

    def runner(prompt, handlers, _config):
        if wf.workspace.delivery_status()["ready"]:
            # The later model acknowledges review but never calls finish again.
            fake_gateway.instances[-1].observe_prompt(prompt)
            return completed()
        fake_gateway.instances[-1].observe_prompt(prompt, tool_calls=[review_tool_call()])
        prepare(wf)
        premature = handlers["finish"]()
        assert not premature["ok"]
        queued = handlers["review_draft"]()
        assert queued["queued"] and not wf.workspace.delivery_status()["ready"]
        return completed([{"name": "finish", "status": "success", "result": premature},
                          {"name": "review_draft", "status": "success", "result": queued}])

    wf.react_runner = runner
    with pytest.raises(module.DocumentWorkflowFailure, match="session budget exhausted"):
        wf.run()
    assert wf.workspace.delivery_status()["ready"]
    assert not wf._finish_is_current()
    assert len(wf.review_sessions) == 2


@pytest.mark.parametrize("budget", ["time", "calls"])
def test_entry_budget_exhaustion_is_a_persisted_workflow_failure(tmp_path, fake_gateway, budget):
    wf = workflow(tmp_path, settings=module.DocumentSettings(max_sessions=1, max_model_calls=2))
    if budget == "time":
        wf.deadline = time.monotonic() - 1
    else:
        wf.client.calls.extend([{"kind": "generation", "call_id": f"previous-{i}"} for i in range(2)])
    with pytest.raises(module.DocumentWorkflowFailure) as caught:
        wf.run()
    assert caught.value.receipt["status"] == "failed"
    persisted = json.loads(wf.trace_path.read_text())
    assert persisted["status"] == "failed"
    assert not fake_gateway.instances


def test_resume_keeps_semantic_attempts_model_calls_and_cumulative_time(tmp_path):
    settings = module.DocumentSettings(max_sessions=4, max_review_attempts=1, report_timeout=100)
    original = workflow(tmp_path, settings=settings)
    ready(original)
    original.total_spent_seconds = 25
    original._persist()
    restored = workflow(tmp_path, client=FakeModel(), settings=settings)
    assert restored.review_attempts == 1
    assert restored.total_spent_seconds >= 25
    assert restored.deadline - restored.started <= 75
    assert len(restored._model_calls()) == len(original.client.calls)
    assert restored.workspace.delivery_status()["ready"]
    with pytest.raises(RuntimeError, match="review attempt budget"):
        restored._perform_review_draft()
    with pytest.raises((ValueError, WorkspaceIdentityError)):
        workflow(tmp_path, settings=replace(settings, output_tokens=2048))


def test_model_identity_mismatch_cannot_mix_reader_and_reviewer(tmp_path):
    client = FakeModel()
    client.model = "different-model"
    with pytest.raises(ValueError, match="same pinned own model"):
        workflow(tmp_path, client=client)


def test_external_run_fingerprint_is_part_of_resume_identity(tmp_path):
    original = workflow(tmp_path, run_identity={"code_sha256": "first", "adapter_sha256": "weights-v1"})
    read(original, [0])
    original._persist()
    with pytest.raises((ValueError, WorkspaceIdentityError), match="identity"):
        workflow(tmp_path, run_identity={"code_sha256": "first", "adapter_sha256": "weights-v2"})


def test_completed_checkpoint_reexports_after_budgets_without_new_inference(tmp_path, fake_gateway):
    original = workflow(tmp_path)

    def runner(prompt, handlers, _config):
        gateway = fake_gateway.instances[-1]
        if not original.workspace.delivery_status()["ready"]:
            gateway.observe_prompt(prompt, tool_calls=[review_tool_call()])
            prepare(original)
            queued = handlers["review_draft"]()
            assert queued["queued"]
            return completed([{"name": "review_draft", "status": "success", "result": queued}])
        gateway.observe_prompt(prompt)
        result = handlers["finish"]()
        assert result["ok"]
        return completed([{"name": "finish", "status": "success", "result": result}])

    original.react_runner = runner
    result = original.run()
    # Export can be retried even when no session/time remains. The accepted
    # model result and finish receipt already exist and must stay unchanged.
    original.total_spent_seconds = original.settings.report_timeout + 1
    original._persist("completed")
    gateway_count = len(fake_gateway.instances)

    def must_not_run(*_):
        pytest.fail("A completed report must not start a new native agent session")

    client = FakeModel()
    restored = workflow(tmp_path, client=client, runner=must_not_run)
    repeated = restored.run()
    assert repeated["data"] == result["data"]
    assert repeated["receipt"]["delivery"]["ready"]
    assert client.calls == []
    assert len(fake_gateway.instances) == gateway_count


@pytest.mark.parametrize("section", ["基本信息", "维修建议", "病害明细"])
def test_columns_the_exporter_cannot_preserve_are_rejected_before_review(tmp_path, section):
    wf = workflow(tmp_path)
    read(wf)
    draft = {"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [], "病害明细": []}
    bindings = {"基本信息.桥梁名称": ["name"]}
    if section == "基本信息":
        draft[section]["自拟且导出器没有的列"] = "这段不应在导出时悄悄丢失"
        bindings["基本信息.自拟且导出器没有的列"] = ["name"]
    else:
        required = "建议内容" if section == "维修建议" else "病害描述"
        draft[section].append({required: "有效内容", "自拟且导出器没有的列": "不应静默丢失"})
        bindings[f"{section}.0"] = ["name"]
    # New writes reject unsupported fields atomically. Seed an older checkpoint
    # explicitly as well: restoration must not pay for review of such a draft.
    before = wf.workspace.checkpoint_path.read_bytes()
    with pytest.raises(ValueError, match="unsupported fields"):
        prepare(wf, draft=draft, draft_bindings=bindings)
    assert wf.workspace.checkpoint_path.read_bytes() == before
    prepare(wf)
    legacy = json.loads(wf.workspace.checkpoint_path.read_text())
    legacy.update(draft=draft, draft_bindings=bindings, draft_sha256=module.json_digest(draft))
    wf.workspace.checkpoint_path.write_text(json.dumps(legacy, ensure_ascii=False))
    with pytest.raises(ValueError, match="unsupported fields"):
        wf._perform_review_draft()
    assert wf.client.calls == []
    assert wf.client.payloads == []
    assert wf.review_attempts == 0
    assert wf.workspace.snapshot()["review"] is None


def test_empty_paragraph_is_not_claimed_as_nonempty_reviewed_evidence(tmp_path):
    manifest = manifest_from_text("桥梁名称：合成桥\n本次左幅1号梁有裂缝。\n")
    blank = deepcopy(manifest["units"][0])
    blank.update(id="EEMPTY", coverage_id="EEMPTY", text="", source_path="synthetic#empty-paragraph")
    manifest["units"].insert(1, blank)
    wf = workflow(tmp_path, manifest=manifest)
    read(wf, [0, 2])
    prepare(wf)
    assert wf._perform_review_draft()["delivery"]["ready"]
    state = wf.workspace.snapshot()
    assert "EEMPTY" not in state["review"]["checked_evidence_ids"]
    assert "EEMPTY" not in state["reading_progress"]["fully_read_evidence_ids"]
    assert all(span.get("evidence_id") != "EEMPTY" for span in state["review"]["checked_read_ranges"])
    assert all(span.get("evidence_id") != "EEMPTY" for payload in wf.client.payloads for span in payload["source_spans"])


def test_saved_image_observation_reaches_forward_and_reverse_review(tmp_path):
    manifest = manifest_from_text("桥梁名称：合成桥\n照片见附图。\n")
    manifest["images"] = [{"id": "I00001", "source_path": "synthetic#image-occurrence-1", "verification": "unverified"}]
    wf = workflow(tmp_path, manifest=manifest)
    read(wf)
    observation = {"ok": True, "status": "observed", "verification": "unverified", "semantic_review_required": True,
                   "image_id": "I00001", "source_path": "synthetic#image-occurrence-1", "question": "照片显示什么病害？",
                   "backend": {"identity": "synthetic-vision", "version": "v1", "model": "synthetic-own-model"},
                   "observation": {"text": "右幅2号梁照片中有裂缝", "verification": "unverified"},
                   "judgment": {"description": "疑似裂缝，尺度待核实"}}
    # Simulate a successful image tool's persisted output/read callback. The
    # workflow test never opens an image file or invokes an OCR/vision backend.
    wf.tools.image_observations.append(deepcopy(observation))
    wf.tools.on_read("I00001", 0, 1)
    prepare(wf, facts=[source_fact(wf), {"id": "photo-damage", "value": "右幅2号梁裂缝", "status": "inferred",
                                      "source_refs": [{"image_id": "I00001"}], "depends_on": []}],
            draft={"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [],
                   "病害明细": [{"病害部位": "右幅2号梁", "病害描述": "裂缝"}]},
            draft_fact_ids=["name", "photo-damage"],
            draft_bindings={"基本信息.桥梁名称": ["name"], "病害明细.0": ["photo-damage"]})
    result = wf._perform_review_draft()
    assert result["delivery"]["ready"]
    forward = next(payload for payload in wf.client.payloads if payload["direction"] == "FORWARD")
    damage = next(item for item in forward["output_items"] if item["path"] == "病害明细.0")
    source = next(source for source in damage["sources"] if source.get("image_id") == "I00001")
    assert source["observations"] == [observation]
    reverse = next(payload for payload in wf.client.payloads if payload.get("image_observations"))
    assert reverse["direction"] == "REVERSE"
    assert reverse["image_observations"] == [observation]
    assert reverse["source_spans"] == [{"image_id": "I00001"}]
    assert any(item["path"] == "病害明细.0" for item in reverse["related_output"])
    assert "cannot independently see pixels" in reverse["limitations"]
    assert "I00001" in wf.workspace.snapshot()["review"]["checked_evidence_ids"]
    persisted = json.loads(wf.trace_path.read_text())
    assert persisted["image_observations"] == [observation]


def test_actual_sdk_context_overflow_recovers_with_navigation_and_tools(tmp_path, fake_gateway):
    wf = workflow(tmp_path, settings=module.DocumentSettings(max_sessions=6, max_model_calls=30))
    prompts = []

    def runner(prompt, handlers, _config):
        prompts.append(prompt)
        gateway = fake_gateway.instances[-1]
        if len(prompts) == 1:
            assert "桥梁名称：合成桥" in prompt
            # Admission counted the report. The actual SDK's tool definitions
            # push its complete request over context; preflight sends no model
            # request and therefore grants no read receipt.
            gateway.receipts.append({"status": "rejected", "error": "ValueError",
                                     "prompt_tokens_checked": wf.settings.max_model_len - wf.settings.output_tokens + 1})
            assert wf.workspace.snapshot()["reads"] == {}
            return completed(status="failed", native_final_event=False)
        assert "桥梁名称：合成桥" not in prompt
        assert "read_workspace" in prompt or "read_outline" in prompt
        if wf.workspace.delivery_status()["ready"]:
            gateway.observe_prompt(prompt)
            final = handlers["finish"]()
            assert final["ok"]
            return completed([{"name": "finish", "status": "success", "result": final}])
        gateway.observe_prompt(prompt, tool_calls=[review_tool_call()])
        assert wf.workspace.snapshot()["reads"] == {}
        unit = wf.manifest["units"][0]
        result = handlers["read_source"](evidence_id=unit["id"])
        assert result["ok"]
        prepare(wf)
        queued = handlers["review_draft"]()
        assert queued["queued"]
        return completed([{"name": "review_draft", "status": "success", "result": queued}])

    wf.react_runner = runner
    result = wf.run()
    assert len(prompts) == 3
    assert result["receipt"]["entry_mode"] == "navigation"
    assert result["receipt"]["agent_generation_requests"] == 2
    assert result["receipt"]["sdk_sessions"][0]["status"] == "failed"
    assert result["receipt"]["delivery"]["ready"]


def test_expired_failed_checkpoint_returns_receipt_without_token_or_model_calls(tmp_path, fake_gateway):
    original = workflow(tmp_path)
    read(original, [0])
    original.client.calls.append({"kind": "agent_generation", "call_id": "prior-failed-call", "status": "failed"})
    original.total_spent_seconds = original.settings.report_timeout + 1
    original._persist("failed")

    class MustNotCallModel(FakeModel):
        def count_tokens(self, *_):
            pytest.fail("Expired recovery must not request token counts")

        count_chat_tokens = count_tokens
        _tokenize = count_tokens
        _request = count_tokens
        generate = count_tokens

    client = MustNotCallModel()
    restored = workflow(tmp_path, client=client)
    with pytest.raises(module.DocumentWorkflowFailure) as caught:
        restored.run()
    assert caught.value.receipt["status"] == "failed"
    assert caught.value.receipt["error"] == "TimeoutError"
    assert caught.value.receipt["model_calls"] == original.client.calls
    assert client.calls == []
    assert not fake_gateway.instances
    assert restored._tools is None
    assert json.loads(restored.trace_path.read_text())["status"] == "failed"
