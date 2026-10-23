"""Offline mode propagation and actual gateway transport; no model service."""
from copy import deepcopy
import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from bridge_evidence.contest import document_multiagent as pipeline
from bridge_evidence.contest import document_multiagent_cli as cli
from bridge_evidence.contest import document_multiagent_worker as worker
from bridge_evidence.contest import tong_react_adapter as adapter
from bridge_evidence.contest.document_model_gateway import AuditedModelGateway
from bridge_evidence.contest.document_workspace import DocumentWorkspace
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.reliable_generation import ReliableGenerationError, StreamPolicy
from bridge_evidence.contest.run_state import atomic_json, json_digest
from test_document_multiagent import FakeGateway, candidate
from test_multiagent_stream_completion import ContextOpener
from test_reliable_document_gateway import Clock, SSEResponse, complete, event
from test_tong_react_adapter import TextAction
from test_section_agent_adapter import Tool, Parameters, fake_sdk


CONFIG = {"model": "own-model", "base_url": "http://127.0.0.1:9", "thinking_tokens": 4096,
          "summary": {"model": "own-summary", "base_url": "http://127.0.0.1:10"}}
SAMPLING = {"temperature": 1.0, "top_p": .95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.5, "repetition_penalty": 1.1}


@pytest.mark.parametrize("disabled", [False, True])
def test_cli_preserves_default_and_explicitly_propagates_disabled_mode(tmp_path, monkeypatch, disabled):
    source = tmp_path / "synthetic.json"
    atomic_json(source, manifest_from_text("合成原文"))
    observed = []
    def run(manifest, directory, model, **kwargs):
        observed.append(model)
        return {"status": "synthetic-stop", "complete": False, "seconds": 0, "error": None}
    monkeypatch.setattr(cli, "run_multiagent", run)
    args = ["--manifest", str(source), "--manifest-sha256", hashlib.sha256(source.read_bytes()).hexdigest(),
            "--scope", "official-training", "--output-dir", str(tmp_path / "run"), "--execute",
            "--model", "own-model", "--base-url", "http://127.0.0.1:9"]
    assert cli.main(args + (["--no-thinking"] if disabled else [])) == 2
    assert observed[0]["enable_thinking"] is (not disabled)
    assert observed[0]["thinking_tokens"] == 4096


@pytest.mark.parametrize("enabled", [True, False])
def test_dispatch_and_main_workflow_receive_same_mode_and_summary_stays_thinking(tmp_path, enabled):
    manifest = manifest_from_text("合成桥")
    observed = []
    def dispatch(tasks, **kwargs):
        rows = []
        for task in tasks:
            taskdir = tmp_path / (task["id"] + "-worker")
            taskdir.mkdir()
            kwargs["worker_command"](task, taskdir)
            request = json.loads((taskdir / "request.json").read_bytes())
            assert request["model_config"]["enable_thinking"] is enabled
            observed.append("child")
            result = candidate(tmp_path, manifest, task["id"])
            path = taskdir / "result.json"
            atomic_json(path, result)
            rows.append({"task_id": task["id"], "result_path": str(path),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "settled": True})
        return {"status": "failed", "results": rows}
    class Merger:
        artifact_delivery = None
        def __init__(self, source, client, config, directory, **kwargs):
            assert client.enable_thinking is enabled
            assert config["enable_thinking"] is enabled
            assert kwargs["settings"].enable_thinking is enabled
            assert kwargs["settings"].max_model_len == 262144
            summary = kwargs["compression_client"]
            assert summary.enable_thinking is True
            assert summary.reliable_thinking_budget is None
            assert client.reliable_thinking_budget == (4096 if enabled else None)
            observed.append("main")
            self.workspace = DocumentWorkspace(source, directory, {"model": "own-model"})
        def run(self):
            raise RuntimeError("Synthetic inspection boundary")
        def _model_calls(self):
            return []
    result = pipeline.run_multiagent(manifest, tmp_path / "run", {**CONFIG, "enable_thinking": enabled},
                                    dispatcher=dispatch, workflow_type=Merger)
    assert result["error"]["message"] == "Synthetic inspection boundary"
    assert observed == ["child", "main"]


@pytest.mark.parametrize("enabled", [True, False])
def test_section_runner_passes_pinned_mode_to_sdk(tmp_path, monkeypatch, enabled):
    manifest = manifest_from_text("合成桥")
    source = tmp_path / "source.json"
    atomic_json(source, manifest)
    request = {"manifest_path": str(source), "manifest_sha256": json_digest(manifest),
        "task": {"id": "part1", "primary_evidence_ids": [manifest["units"][0]["id"]]},
        "model_config": {"enable_thinking": enabled},
        "deadline_monotonic": time.monotonic() + 60, "max_model_calls": 8}
    client = SimpleNamespace(calls=[], model="own-model", api_key="synthetic", enable_thinking=enabled,
                             count_tokens=lambda value: len(str(value)))
    monkeypatch.setattr(worker, "AuditedModelGateway", FakeGateway)
    observed = []
    def react(prompt, handlers, config):
        observed.append(config["enable_thinking"])
        gateway = FakeGateway.instance
        gateway.on_receipt({"status": "success", "call_id": "synthetic1"})
        payload = {"messages": [{"role": "user", "content": prompt}]}
        audit = {"request_id": "synthetic-request", "call_id": "synthetic1",
                 "input_delivery_confirmed": True, "http_status": 200}
        gateway.on_prepared_request(payload, audit)
        gateway.on_input_observed(payload, audit)
        assert handlers["complete_section"]("完成合成分工", [])["status"] == "handed_off"
        return {"status": "completed", "native_final_event": True}
    result = worker.run_section(request, tmp_path / "section", client=client, react_runner=react)
    assert result["status"] == "completed"
    assert observed == [enabled]


@pytest.mark.parametrize("profile", ["reading", "section"])
def test_adapter_sets_sdk_extra_body_false_without_hiding_response(monkeypatch, profile):
    def script(registry, settings, outputs, prompt):
        outputs.append(registry["read_outline"]())
        yield TextAction("合成结束")
    sdk, _, configurations, _, _ = fake_sdk(script)
    monkeypatch.setattr(adapter, "_REGISTRATIONS", [])
    monkeypatch.setattr(adapter, "_REGISTERED_TOOL_CLASSES", [])
    monkeypatch.setattr(adapter, "_load_tongagents", lambda: sdk)
    monkeypatch.setattr(adapter, "_load_tool_schema_api", lambda: (Tool, Parameters))
    result = adapter.run_react("合成任务", {"read_outline": lambda **kwargs: {}}, {
        "profile": profile, "url": "http://127.0.0.1:9", "model": "own-model",
        "enable_thinking": False, "temperature": 1.0})
    assert result["status"] == "completed"
    assert configurations[0].llm_config.extra_body == {
        "temperature": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
    assert result["response_text"] == "合成结束"


def route(raw=None):
    deadline = time.monotonic() + 60
    client = worker.make_client({**CONFIG, "enable_thinking": False}, deadline)
    stream = SSEResponse(complete(tools=True) if raw is None else raw, Clock())
    client._opener = ContextOpener(stream)
    gateway = AuditedModelGateway(client, deadline=deadline, max_calls=3, max_output_tokens=262144)
    payload = {"model": "own-model", "messages": [{"role": "user", "content": "完整合成材料"}],
        "max_tokens": 262144 - 23, "temperature": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
    return client, stream, gateway, payload


def test_real_gateway_sends_false_with_same_sampling_and_remaining_context():
    client, stream, gateway, payload = route()
    before = deepcopy(payload)
    prepared = gateway.prepare_request(payload)
    result = gateway.forward(prepared)
    assert payload == before
    wire = client._opener.requests[-1][1]
    assert wire["chat_template_kwargs"] == {"enable_thinking": False}
    assert "thinking_token_budget" not in wire
    assert {key: wire[key] for key in SAMPLING} == SAMPLING
    assert wire["max_tokens"] == 262144 - 23
    assert wire["stream"] is True
    assert all(request[1]["chat_template_kwargs"] == {"enable_thinking": False}
               for request in client._opener.requests)
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "update_workspace"
    assert client.calls[-1]["done_received"] is True
    assert client.calls[-1]["reasoning_characters"] == 0
    assert stream.closed


@pytest.mark.parametrize("override", [None, {}, {"enable_thinking": True}, {"enable_thinking": "false"}])
def test_sdk_override_is_rejected_before_any_upstream_request(override):
    client, _, gateway, payload = route()
    if override is None:
        payload.pop("chat_template_kwargs")
    else:
        payload["chat_template_kwargs"] = override
    with pytest.raises(ValueError, match="explicitly match"):
        gateway.forward(payload)
    assert client._opener.requests == []


def test_disabled_mode_does_not_hide_unexpected_service_reasoning():
    raw = event({"reasoning_content": "synthetic unexpected channel"}) + complete()
    client, _, gateway, payload = route(raw)
    result = gateway.forward(payload)
    assert result["choices"][0]["message"]["reasoning_content"] == "synthetic unexpected channel"
    assert client.calls[-1]["reasoning_characters"] > 0


def test_disabled_mode_retains_stream_failure_and_timeout_protection():
    client, stream, gateway, payload = route(event({"content": "半截合成结果"}))
    with pytest.raises(ReliableGenerationError) as failure:
        gateway.forward(payload)
    assert failure.value.code == "stream_protocol_error"
    assert client.reliable_generation_policy.wall_seconds == 900
    assert client.reliable_generation_policy.read_idle_seconds == 60
    assert stream.closed


def test_disabled_mode_rejects_stale_budgets_and_boolean_coercion():
    client, _, _, payload = route()
    with pytest.raises(ValueError, match="cannot carry a thinking budget"):
        client.reliable_payload({**payload, "thinking_token_budget": 4096})
    for kwargs in ({"thinking_budget": 4096}, {"thinking_budget_selector": lambda _: 4096}):
        with pytest.raises(ValueError, match="cannot carry a thinking budget"):
            client.enable_reliable_generation(StreamPolicy(), **kwargs)
    for bad in (None, "false", 0, 1):
        with pytest.raises(ValueError, match="explicit boolean"):
            worker.make_client({**CONFIG, "enable_thinking": bad}, time.monotonic() + 60)


def test_default_thinking_profile_keeps_budget_and_sampling():
    client = worker.make_client(CONFIG, time.monotonic() + 60)
    payload = client.reliable_payload({"max_tokens": 262121,
        "chat_template_kwargs": {"enable_thinking": True}})
    assert client.enable_thinking is True
    assert payload["thinking_token_budget"] == 4096
    assert {key: payload[key] for key in SAMPLING} == SAMPLING
