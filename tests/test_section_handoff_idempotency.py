"""Exercise real run_section/store/tools with a local, deterministic SDK runner.

No remote service, network listener, real model or real report is used.
"""
from copy import deepcopy
import json
import socket
import subprocess
from types import SimpleNamespace
import time

import pytest

from bridge_evidence.contest import document_multiagent_worker as worker
from bridge_evidence.contest import tong_react_adapter as worker_tools
from bridge_evidence.contest.document_entries import EntryStore
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import atomic_json, json_digest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("No network or subprocess in handoff regression tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)


@pytest.fixture
def section(tmp_path, monkeypatch):
    manifest = manifest_from_text("合成桥存在裂缝。\n补充段落提供桥梁历史信息。")
    source = tmp_path / "source.json"
    atomic_json(source, manifest)
    directory = tmp_path / "section"
    first = manifest["units"][0]
    entry = {"id": "bridge", "section": "基本信息", "field": "桥梁名称", "value": "合成桥",
             "source_refs": [{"evidence_id": first["id"]}], "status": "explicit"}
    request = {"manifest_path": str(source), "manifest_sha256": json_digest(manifest),
        "task": {"id": "part-one", "primary_evidence_ids": [first["id"]]},
        "model_config": {}, "deadline_monotonic": time.monotonic() + 600, "max_model_calls": 8}
    client = SimpleNamespace(calls=[], model="synthetic-own", api_key="synthetic",
                             count_tokens=lambda value: len(str(value)))
    class Gateway:
        instance = None
        def __init__(self, client, **kwargs):
            self.client, self.receipts, self.url = client, [], "http://unused"
            self.messages, self.pending_tool_messages = [], []
            Gateway.instance = self
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def generation(self, prompt=None):
            call_id = "synthetic-" + str(len(self.client.calls) + 1)
            receipt = {"kind": "agent_generation", "call_id": call_id,
                       "request_id": "request-" + call_id, "status": "success", "http_status": 200,
                       "generation_attempted": True, "done_received": True}
            if prompt is not None:
                self.messages.append({"role": "user", "content": prompt})
            self.messages.extend(self.pending_tool_messages)
            self.pending_tool_messages = []
            payload = {"messages": deepcopy(self.messages)}
            self.on_prepared_request(payload, receipt)
            self.on_input_observed(payload, {**receipt, "input_delivery_confirmed": True,
                "input_delivery_evidence": "complete_response"})
            self.client.calls.append(receipt)
            self.receipts.append(deepcopy(receipt))
            self.on_receipt(receipt)
        def queue_tool_result(self, name, result, arguments):
            self.pending_tool_messages.extend([
                {"role": "assistant", "content": None, "tool_calls": [{"id": "synthetic-read", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]},
                {"role": "tool", "tool_call_id": "synthetic-read",
                    "content": json.dumps(worker_tools.wrap_tool_response(name, result), ensure_ascii=False)}])
    monkeypatch.setattr(worker, "AuditedModelGateway", Gateway)
    observed = {}
    def run(after_handoff):
        def runner(prompt, handlers, config):
            gateway = Gateway.instance
            gateway.generation(prompt)
            first_save = handlers["save_entries"]([entry])
            assert first_save["ok"] and first_save["changed"]
            assert handlers["complete_section"]("已保存本范围条目", [])["status"] == "handed_off"
            observed["handoff"] = json.loads((directory / "handoff.json").read_text())
            observed["state"] = EntryStore(manifest, request["task"]["id"], directory / "entries").snapshot()
            after_handoff(handlers, gateway)
            gateway.generation()
            return {"status": "completed", "native_final_event": True, "response_text": "分工已交接"}
        return worker.run_section(request, directory, client=client, react_runner=runner)
    return SimpleNamespace(run=run, entry=entry, observed=observed, manifest=manifest,
                           directory=directory, request=request)


def test_handoff_survives_successful_noop_save(section):
    def same(handlers, gateway):
        saved = handlers["save_entries"]([deepcopy(section.entry)])
        assert saved["ok"] is True and saved["changed"] is False
    result = section.run(same)
    assert result["status"] == "completed"
    assert result["handoff"] == section.observed["handoff"]
    assert result["entries"] == section.observed["state"]
    assert result["document_delivery_complete"] is False


@pytest.mark.parametrize("handoff_again", [False, True])
def test_actual_content_change_requires_a_new_handoff(section, handoff_again):
    def change(handlers, gateway):
        gateway.generation()
        updated = {**section.entry, "value": "合成桥（名称经模型更正）"}
        saved = handlers["save_entries"]([updated])
        assert saved["ok"] is True and saved["changed"] is True
        if handoff_again:
            assert handlers["complete_section"]("已重新核对更正条目", [])["ok"] is True
    result = section.run(change)
    assert (result["status"] == "completed") is handoff_again
    if handoff_again:
        assert result["handoff"]["entries_content_sha256"] != section.observed["handoff"]["entries_content_sha256"]
        assert result["handoff"]["origin_call_id"] != section.observed["handoff"]["origin_call_id"]
    else:
        assert result["handoff"] is None


def test_failed_validation_does_not_erase_a_current_handoff(section):
    def invalid(handlers, gateway):
        saved = handlers["save_entries"]([{**section.entry, "field": "不存在的字段"}])
        assert saved["ok"] is False and saved["status"] == "error"
    result = section.run(invalid)
    assert result["status"] == "completed"
    assert result["handoff"] == section.observed["handoff"]
    assert result["entries"] == section.observed["state"]


def test_extra_source_read_changes_ledger_version_but_preserves_handed_off_entries(section):
    def read_extra(handlers, gateway):
        extra = section.manifest["units"][1]
        assert extra["id"] not in section.observed["state"]["reads"]
        page = handlers["read_source"](evidence_id=extra["id"])
        assert page["ok"]
        # Merely returning a tool page is not a model read. Its exact wrapped
        # original text must enter the next accepted generation first.
        pending = EntryStore(section.manifest, section.request["task"]["id"], section.directory / "entries").snapshot()
        assert extra["id"] not in pending["reads"]
        gateway.queue_tool_result("read_source", page, {"evidence_id": extra["id"]})
        saved = handlers["save_entries"]([deepcopy(section.entry)])
        assert saved["ok"] is True and saved["changed"] is False
    result = section.run(read_extra)
    assert result["status"] == "completed"
    assert result["entries"]["version"] > result["handoff"]["entries_version"]
    assert json_digest(result["entries"]) != result["handoff"]["entries_sha256"]
    assert result["handoff"] == section.observed["handoff"]
    assert result["entries"]["entries"] == section.observed["state"]["entries"]


def test_content_fingerprint_rejects_changed_ledger_even_without_handler_invalidation(section):
    def externally_changed(handlers, gateway):
        store = EntryStore(section.manifest, section.request["task"]["id"], section.directory / "entries")
        assert store.save_entries([{**section.entry, "value": "另一处真实内容变化"}])["changed"] is True
        # A subsequent no-op must not cause the old handoff to match this new state.
        assert handlers["save_entries"]([{**section.entry, "value": "另一处真实内容变化"}])["changed"] is False
    result = section.run(externally_changed)
    assert result["status"] == "failed"
    assert result["handoff"] == section.observed["handoff"]
    assert worker._entry_content_sha256(result["entries"]) != result["handoff"]["entries_content_sha256"]


def test_removed_primary_reading_coverage_cannot_keep_handoff_current(section):
    extra = section.manifest["units"][1]["id"]
    section.request["task"]["primary_evidence_ids"].append(extra)
    def remove_coverage(handlers, gateway):
        # Simulate loss of an assigned source's read receipt. The saved entry
        # still has its own required source, so the ledger remains valid.
        path = section.directory / "entries/entries.json"
        state = json.loads(path.read_text())
        del state["reads"][extra]
        state["version"] += 1
        atomic_json(path, state)
    result = section.run(remove_coverage)
    assert result["status"] == "failed"
    assert result["handoff"] == section.observed["handoff"]
