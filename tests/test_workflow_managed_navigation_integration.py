"""Run's actual native handlers, without SDK, model requests or network.

These tests deliberately end before report delivery. They prove the main
workflow exposes, persists and restores host-owned read positions; they do not
manufacture a successful report or claim that source delivery is understanding.
"""
from copy import deepcopy
import json
import socket
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

from bridge_evidence.contest import document_workflow as workflow_module
from bridge_evidence.contest import tong_react_adapter as adapter
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import json_digest


@pytest.fixture(autouse=True)
def offline_gateway(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Navigation integration must not invoke network, a model or a subprocess")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)

    class Gateway:
        def __init__(self, *_args, **_kwargs):
            self.url = "http://unused.invalid/v1"
            self.receipts = []
            self.on_success = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        prepare_request = staticmethod(forbidden)

    monkeypatch.setattr(workflow_module, "AuditedModelGateway", Gateway)


def _workflow(directory, runner, *, resume=False):
    client = SimpleNamespace(
        model="synthetic-own-model", base_url="http://unused.invalid", calls=[],
        max_model_len=262144, timeout=900.0, dynamic_output_budget=True,
        enable_thinking=False, temperature=1.0,
        count_tokens=lambda text: max(1, (len(text) + 3) // 4),
        count_chat_tokens=lambda messages: max(1, (len(json.dumps(messages, ensure_ascii=False)) + 3) // 4),
    )
    manifest = manifest_from_text("1 检测\n合成桥名，左幅裂缝。\n裂缝第二次提及。\n2 建议\n原文建议。\n建议末段。")
    return workflow_module.DocumentWorkflow(
        manifest, client, {"model": client.model}, directory, react_runner=runner,
        explicit_operator_resume=resume,
        settings=workflow_module.DocumentSettings(
            max_model_len=262144, output_tokens=262144, tool_tokens=4096,
            single_document_agent=True, dynamic_output_budget=True,
            require_semantic_review=False, enable_thinking=False, temperature=1.0,
        ),
    )


def _native_session(handlers, config, workflow):
    return {"handlers": handlers, "tool_calls": [], "max_calls": 100,
            "tool_budget_exceeded": False, "count_tokens": config["count_tokens"],
            "max_tool_tokens": config["max_tool_tokens"],
            "on_tool_result": config.get("on_tool_result"), "workflow": workflow}


def _call(session, name, *, confirm_delivery=True, **arguments):
    generation, native = uuid4().hex, uuid4().hex
    identity = {"run_id": "offline-workflow", "session_id": "offline-session",
                "generation_call_id": generation, "native_tool_call_id": native,
                "tool_name": name, "args_sha256": json_digest(arguments)}
    # The real SDK identity hook is exercised independently. This integration
    # gives each scripted generation a trusted host identity, without a model.
    with patch.object(adapter, "current_tool_call_identity", return_value=identity):
        response = adapter._invoke_active(session, name, arguments)
    assert response["ok"], (name, arguments, response)
    if confirm_delivery:
        payload = {"messages": [{"role": "tool", "tool_call_id": native,
                                "content": json.dumps(response, ensure_ascii=False)}]}
        audit = {"request_id": generation, "request_sha256": json_digest(payload),
                 "request_kind": "agent_generation", "input_delivery_confirmed": True}
        # Simulate gateway wire observation then upstream acknowledgement,
        # through production workflow hooks, not a direct record_read call.
        session["workflow"]._on_model_request(payload, audit)
        session["workflow"]._source_input_observed(payload, audit)
    return response["result"]


def _stopped():
    return {"status": "failed", "native_final_event": False, "tool_calls": [],
            "failure_reason": "synthetic_reading_integration_stop_before_delivery"}


def test_root_run_exposes_position_free_tools_and_persists_actual_progress(tmp_path):
    captured = {}

    def runner(_prompt, handlers, config):
        captured["invoked"] = True
        captured["handlers"] = set(handlers)
        session = _native_session(handlers, config, wf)
        first = _call(session, "read_section", section_id="E000001", max_units=1, confirm_delivery=False)
        replay = _call(session, "read_section", section_id="E000001", max_units=1)
        assert replay["replayed"] and replay["spans"] == first["spans"]
        other = _call(session, "read_section", section_id="E000004", max_units=1)
        following = _call(session, "read_section", section_id="E000001", max_units=1)
        assert first["spans"][0]["evidence_id"] == "E000001"
        assert other["spans"][0]["evidence_id"] == "E000004"
        assert following["spans"][0]["evidence_id"] == "E000002"
        outline = _call(session, "read_outline", limit=1)
        assert outline["sections"][0]["id"] == "root"
        hit = _call(session, "search_in_document", query=["裂缝"], max_matches=1)
        assert hit["matches"][0]["evidence_id"] == "E000002"
        source = _call(session, "read_source", evidence_id="E000002")
        assert source["spans"][0]["text"].startswith("合成桥名")
        _call(session, "read_workspace_page", view="facts", limit=1)
        captured["source_state"] = json.loads((tmp_path / "source-navigation-managed.json").read_text())
        captured["workspace_state"] = json.loads((tmp_path / "workspace-navigation.json").read_text())
        captured["source_receipts"] = deepcopy(wf.tools.receipts)
        return _stopped()

    wf = _workflow(tmp_path, runner)
    with pytest.raises(workflow_module.DocumentWorkflowFailure, match="without accepted native delivery"):
        wf.run()
    assert captured["invoked"]
    assert {"read_section", "read_source", "read_table_rows",
            "read_workspace_page", "inspect_delivery", "finish"}.issubset(captured["handlers"])
    assert "review_draft" not in captured["handlers"]
    assert captured["source_state"] == wf.tools.navigation_state()
    objects = list(captured["source_state"]["objects"].values())
    first_chapter = next(item for item in objects if item["identity"] == {
        "tool": "read_section", "section_id": "E000001"})
    assert first_chapter["pending_calls"][0]["arguments"]["cursor"] == 2
    assert first_chapter["delivered_calls"] == 2
    assert first_chapter["complete"] is False
    assert len(captured["source_receipts"]) == 7
    assert all(receipt["status"] in {"read", "outline", "search"} for receipt in captured["source_receipts"])
    assert captured["workspace_state"] == wf.workspace_tools.managed_navigation_snapshot()
    assert wf.client.calls == []


def test_new_main_workflow_restores_each_source_and_workspace_view_independently(tmp_path):
    before = {}

    def first_runner(_prompt, handlers, config):
        session = _native_session(handlers, config, first)
        before["chapter"] = _call(session, "read_section", section_id="E000001", max_units=1)
        before["other"] = _call(session, "read_section", section_id="E000004", max_units=1)
        before["facts"] = _call(session, "read_workspace_page", view="facts", limit=1)
        before["outline"] = _call(session, "read_outline", limit=1)
        before["search"] = _call(session, "search_in_document", query=["裂缝"], max_matches=1)
        return _stopped()

    first = _workflow(tmp_path, first_runner)
    for unit in first.manifest["units"]:
        first.workspace.record_read(unit["id"], 0, len(unit["text"]))
    unit = first.manifest["units"][1]
    first.workspace.apply_update({"base_version": first.workspace.snapshot()["version"], "facts": [
        {"id": f"F{i}", "value": f"合成测试事实{i}", "status": "explicit",
         "source_refs": [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}]}
        for i in range(3)]})
    with pytest.raises(workflow_module.DocumentWorkflowFailure):
        first.run()
    assert before["facts"]["items"][0]["id"] == "F0"
    saved = json.loads((tmp_path / "source-navigation-managed.json").read_text())
    after = {}

    def resumed_runner(_prompt, handlers, config):
        session = _native_session(handlers, config, resumed)
        after["chapter"] = _call(session, "read_section", section_id="E000001", max_units=1)
        after["other"] = _call(session, "read_section", section_id="E000004", max_units=1)
        after["facts"] = _call(session, "read_workspace_page", view="facts", limit=1)
        after["outline"] = _call(session, "read_outline", limit=1)
        after["search"] = _call(session, "search_in_document", query=["裂缝"], max_matches=1)
        return _stopped()

    resumed = _workflow(tmp_path, resumed_runner, resume=True)
    with pytest.raises(workflow_module.DocumentWorkflowFailure):
        resumed.run()
    assert after["chapter"]["spans"][0]["evidence_id"] == "E000002"
    assert after["other"]["spans"][0]["evidence_id"] == "E000005"
    assert after["facts"]["items"][0]["id"] == "F1"
    assert after["outline"]["sections"][0]["id"] == "E000001"
    assert after["search"]["matches"][0]["evidence_id"] == "E000003"
    assert saved["objects"] != resumed.tools.navigation_state()["objects"]
    assert first.client.calls == resumed.client.calls == []
