"""Worker callbacks through the real adapter, with a synthetic SDK and gateway.

No real model or SDK stop semantics are claimed; terminal-handoff coverage lives
in its dedicated suite. This exercises same-object continuation and safeguards.
"""
from contextlib import nullcontext
from copy import deepcopy
import json
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from bridge_evidence.contest import document_multiagent_worker as worker
from bridge_evidence.contest import document_context_budget as budget
from bridge_evidence.contest import tong_react_adapter as adapter
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import atomic_json, json_digest
from test_section_agent_adapter import fake_sdk, Tool, Parameters
from test_native_business_continuation import TextAction, ToolErrorAction


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("No network or subprocess in continuation tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)
    assert adapter._ACTIVE_SESSION is None
    adapter._REGISTRATIONS.clear()
    adapter._REGISTERED_TOOL_CLASSES.clear()
    monkeypatch.setattr(budget, "allocate_sdk_wire_requests", lambda *args: nullcontext())
    monkeypatch.setattr(worker, "load_native_api", lambda: {})
    monkeypatch.setattr(worker, "private_memory", lambda *args, **kwargs: SimpleNamespace(marker="same-memory"))


@pytest.fixture
def section(tmp_path, monkeypatch):
    manifest = manifest_from_text("合成桥存在裂缝。\n合成第二段请勿重复发送全文。")
    source, directory = tmp_path / "source.json", tmp_path / "section"
    atomic_json(source, manifest)
    first = manifest["units"][0]
    entry = {"id": "bridge", "section": "基本信息", "field": "桥梁名称", "value": "合成桥",
             "source_refs": [{"evidence_id": first["id"]}], "status": "uncertain"}
    request = {"manifest_path": str(source), "manifest_sha256": json_digest(manifest),
        "task": {"id": "part-one", "primary_evidence_ids": [unit["id"] for unit in manifest["units"]]},
        "model_config": {}, "deadline_monotonic": time.monotonic() + 600, "max_model_calls": 8}
    client = SimpleNamespace(calls=[], model="synthetic-own", api_key="synthetic", max_model_len=262144,
                             count_tokens=lambda value: len(str(value)))
    class Gateway:
        instance = None
        def __init__(self, client, **kwargs):
            self.client, self.receipts, self.url = client, [], "http://127.0.0.1:9/v1"
            self._forward_lock = threading.Lock()
            self.messages = []
            self.deadline, self.max_calls = kwargs["deadline"], kwargs["max_calls"]
            Gateway.instance = self
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def generation(self, prompt=None):
            ordinal = len(self.client.calls) + 1
            row = {"kind": "agent_generation", "call_id": f"call-{ordinal}", "request_id": f"request-{ordinal}",
                   "status": "success", "http_status": 200, "generation_attempted": True,
                   "done_received": True, "finish_reason": "stop", "prompt_tokens_checked": 120,
                   "usage": {"prompt_tokens": 120, "completion_tokens": 15, "total_tokens": 135}}
            if prompt is not None:
                self.messages.append({"role": "user", "content": prompt})
            payload = {"messages": deepcopy(self.messages)}
            audit = {**deepcopy(row), "request_kind": "agent_generation"}
            # Model the real gateway order: exact prepared input, upstream
            # acceptance, then the settled output receipt. Tool return alone
            # never becomes source coverage.
            self.on_prepared_request(payload, audit)
            self.on_input_observed(payload, {**audit, "input_delivery_confirmed": True,
                "input_delivery_evidence": "complete_response"})
            self.client.calls.append(row)
            self.receipts.append(audit)
            self.on_receipt(audit)
    monkeypatch.setattr(worker, "AuditedModelGateway", Gateway)
    seen = {"prompts": [], "configs": [], "runner_calls": 0}

    def run(mode="text", first_hook=None, complete_next=True):
        def script(registry, settings, outputs, prompt):
            seen["prompts"].append(prompt)
            gateway = Gateway.instance
            step = len(seen["prompts"])
            gateway.generation(prompt)
            if step == 1:
                assert registry["save_entries"]([entry])["ok"]
                if first_hook:
                    first_hook(registry, gateway, seen["configs"][0])
                if mode == "sdk_error":
                    yield ToolErrorAction()
                elif mode == "exception":
                    raise RuntimeError("synthetic SDK exception")
                elif mode != "no_event":
                    yield TextAction("" if mode == "empty" else "已完成", "length" if mode == "length" else "stop")
            else:
                if complete_next:
                    assert registry["complete_section"]("本范围完成", ["名称仍待父Agent核查"])["ok"]
                yield TextAction("交接状态")
        sdk, registry, settings, contexts, outputs = fake_sdk(script)
        monkeypatch.setattr(adapter, "_load_tongagents", lambda: sdk)
        monkeypatch.setattr(adapter, "_load_tool_schema_api", lambda: (Tool, Parameters))
        def runner(prompt, handlers, config):
            seen["runner_calls"] += 1
            seen["configs"].append(config)
            return adapter.run_react(prompt, handlers, config)
        result = worker.run_section(request, directory, client=client, react_runner=runner)
        return SimpleNamespace(result=result, seen=seen, settings=settings, contexts=contexts, gateway=Gateway.instance)
    return SimpleNamespace(run=run, request=request, manifest=manifest, client=client, entry=entry,
                           directory=directory, seen=seen)


@pytest.mark.parametrize("mode", ["text", "empty", "no_event"])
def test_early_native_exit_continues_the_same_agent_with_accurate_progress(section, mode):
    run = section.run(mode)
    assert run.result["status"] == "completed", run.result["native_outcome"].get("error")
    assert run.seen["runner_calls"] == 1 and len(run.contexts) == len(run.settings) == 1
    assert len(run.seen["prompts"]) == 2 and run.result["model_calls"] == 2
    feedback = run.seen["prompts"][1]
    state = json.loads(feedback.splitlines()[1])
    assert state["saved_entries"] == {"基本信息": 1, "病害明细": 0, "维修建议": 0, "total": 1}
    assert state["primary"]["fully_read_units"] == state["primary"]["total_units"] == 2
    assert state["current_handoff"] is False
    assert state["unresolved"]["saved_entry_preview"] == [{"id": "bridge", "status": "uncertain"}]
    assert state["unresolved"]["last_reported_note_count"] is None
    for unit in section.manifest["units"]:
        assert unit["text"] not in feedback
    receipt = run.result["native_outcome"]["business_continuations"]
    assert len(receipt) == 1 and receipt[0]["same_agent"] and receipt[0]["same_context"] and receipt[0]["same_memory"]
    assert run.gateway.max_calls == section.request["max_model_calls"]
    assert run.gateway.deadline == section.request["deadline_monotonic"]


def test_failed_handoff_reports_missing_primary_and_known_unresolved_count(section):
    second = section.manifest["units"][1]
    def incomplete(registry, gateway, config):
        path = section.directory / "entries/entries.json"
        state = json.loads(path.read_text())
        del state["reads"][second["id"]]
        atomic_json(path, state)
        assert registry["complete_section"]("尚未读完", ["历史年份待核对"])["ok"] is False
        # This run only inspects recovery feedback, then exhausts the shared budget.
    section.request["max_model_calls"] = 2
    run = section.run(first_hook=incomplete, complete_next=False)
    state = json.loads(run.seen["prompts"][1].splitlines()[1])
    assert state["primary"] == {"total_units": 2, "fully_read_units": 1, "missing_range_count": 1,
                                  "next_unread_ranges": [{"evidence_id": second["id"], "start": 0, "end": len(second["text"])}]}
    assert state["unresolved"]["last_complete_section_status"] == "incomplete_reading"
    assert state["unresolved"]["last_reported_note_count"] == 1
    assert run.result["status"] == "failed" and run.result["handoff"] is None
    assert run.result["model_calls"] == 2


@pytest.mark.parametrize("fault", ["unknown", "failed", "protocol", "usage_missing", "usage_unknown", "usage_different", "request", "dynamic_failure"])
def test_bad_generation_never_gets_a_continuation(section, fault):
    def corrupt(registry, gateway, config):
        call, audit = section.client.calls[-1], gateway.receipts[-1]
        if fault == "unknown": call["done_received"] = False
        elif fault == "failed": call["status"] = "failed"
        elif fault == "protocol": call["http_status"] = 500
        elif fault == "usage_missing": call.pop("usage")
        elif fault == "usage_unknown": call["usage_known"] = False
        elif fault == "usage_different": audit["usage"]["completion_tokens"] += 1
        elif fault == "request": audit["request_id"] = "different"
        elif fault == "dynamic_failure": section.client._dynamic_budget_failure = "ValueError"
    run = section.run(first_hook=corrupt)
    assert run.result["status"] == "failed"
    assert run.result["model_calls"] == 1 and len(run.seen["prompts"]) == 1
    assert run.result["native_outcome"]["business_continuations"] == []
    assert run.result["entries"]["counts"]["total"] == 1


def test_worker_parse_recovery_rejects_explicit_unknown_usage(section):
    def check(registry, gateway, config):
        assert config["parse_recovery_generation_is_settled"]() is True
        section.client.calls[-1]["usage_known"] = False
        assert config["parse_recovery_generation_is_settled"]() is False
    run = section.run(first_hook=check)
    assert run.result["status"] == "failed" and run.result["model_calls"] == 1


@pytest.mark.parametrize("mode", ["length", "sdk_error", "exception"])
def test_native_protocol_failure_does_not_continue(section, mode):
    run = section.run(mode)
    assert run.result["status"] == "failed" and run.result["model_calls"] == 1
    assert run.result["native_outcome"]["business_continuations"] == []


def test_request_budget_stops_repeated_final_messages(section):
    section.request["max_model_calls"] = 3
    run = section.run(complete_next=False)
    assert run.result["status"] == "failed" and run.result["model_calls"] == 3
    assert len(run.result["native_outcome"]["business_continuations"]) == 2
    assert len(run.contexts) == 1


def test_deadline_cannot_be_refreshed_by_early_final(section, monkeypatch):
    def expire(registry, gateway, config):
        monkeypatch.setattr(worker.time, "monotonic", lambda: section.request["deadline_monotonic"] + 1)
    run = section.run(first_hook=expire)
    assert run.result["status"] == "failed" and run.result["model_calls"] == 1
    assert run.result["native_outcome"]["business_continuations"] == []


def test_current_handoff_does_not_receive_business_feedback(section):
    def complete(registry, gateway, config):
        assert registry["complete_section"]("已交接", [])["ok"]
    run = section.run(first_hook=complete)
    assert run.result["status"] == "completed" and run.result["model_calls"] == 1
    assert run.result["native_outcome"]["business_continuations"] == []
