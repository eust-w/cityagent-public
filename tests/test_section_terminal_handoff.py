"""Real worker/store/tool integration; the SDK runner and generation receipts are fake.

These tests prove worker acceptance/rejection, not actual SDK stop semantics.
No network, model, remote process, or real report is used.
"""
from copy import deepcopy
import json
import socket
import subprocess
import threading
from types import SimpleNamespace
import time

import pytest

from bridge_evidence.contest import document_multiagent_worker as worker
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import atomic_json, json_digest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("No network or subprocess in terminal-handoff tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)


@pytest.fixture
def section(tmp_path, monkeypatch):
    manifest = manifest_from_text("合成桥有裂缝。\n另一段仅供全文回查。")
    source, directory = tmp_path / "source.json", tmp_path / "section"
    atomic_json(source, manifest)
    entry = {"id": "bridge", "section": "基本信息", "field": "桥梁名称", "value": "合成桥",
             "source_refs": [{"evidence_id": manifest["units"][0]["id"]}], "status": "explicit"}
    request = {"manifest_path": str(source), "manifest_sha256": json_digest(manifest),
               "task": {"id": "part-one", "primary_evidence_ids": [manifest["units"][0]["id"]]},
               "model_config": {}, "deadline_monotonic": time.monotonic() + 600, "max_model_calls": 8}
    client = SimpleNamespace(calls=[], model="synthetic-own", api_key="synthetic", max_model_len=262144,
                             count_tokens=lambda value: len(str(value)))

    class Gateway:
        instance = None
        def __init__(self, client, **kwargs):
            self.client, self.receipts, self.url = client, [], "http://unused"
            self._forward_lock = threading.Lock()
            self.messages = []
            Gateway.instance = self
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def generation(self, prompt=None):
            ordinal = len(self.client.calls) + 1
            row = {"kind": "agent_generation", "call_id": f"call-{ordinal}", "request_id": f"request-{ordinal}",
                   "status": "success", "http_status": 200, "generation_attempted": True,
                   "done_received": True, "finish_reason": "tool_calls", "prompt_tokens_checked": 120,
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
    observed = {}

    def run(after_handoff=None, mutate_outcome=None, *, save_handoff=True):
        def runner(prompt, handlers, config):
            gateway = Gateway.instance
            observed.update(handlers=handlers, config=config, gateway=gateway)
            assert config["terminal_handoff_state"]() is None
            assert config["terminal_generation_is_settled"]() is False
            gateway.generation(prompt)
            assert handlers["save_entries"]([entry])["changed"] is True
            if save_handoff:
                assert handlers["complete_section"]("本分工条目已保存", [])["ok"] is True
                observed["handoff"] = config["terminal_handoff_state"]()
                assert observed["handoff"] is not None
            if after_handoff:
                after_handoff(handlers, gateway, config)
            current = config["terminal_handoff_state"]()
            observed["current_handoff"] = current
            observed["settled"] = config["terminal_generation_is_settled"]()
            # Model the documented adapter return; intentionally permit a stale
            # forged success in negative cases to verify independent worker checks.
            handoff = observed.get("handoff") or {}
            outcome = {"status": "completed", "termination_kind": "section_terminal_handoff",
                       "native_final_event": False, "response_text": "",
                       "terminal_handoff_sha256": json_digest(handoff),
                       "terminal_handoff_origin_call_id": handoff.get("origin_call_id"),
                       "terminal_handoff_proof": {"all_batch_tool_calls_paired": True,
                           "stopped_before_next_model_call": True, "sdk_stop_returned": True},
                       "session_boundary": {"iteration_exited": True, "tool_callbacks_drained": True,
                           "active_session_clear": True, "session_lock_released": True}}
            if mutate_outcome:
                mutate_outcome(outcome)
            return outcome
        return worker.run_section(request, directory, client=client, react_runner=runner)

    return SimpleNamespace(run=run, entry=entry, observed=observed, client=client, directory=directory,
                           manifest=manifest, request=request)


def test_terminal_handoff_completes_without_extra_model_generation(section):
    result = section.run()
    assert result["status"] == "completed"
    assert result["model_calls"] == 1
    assert result["handoff"]["origin_call_id"] == section.client.calls[-1]["call_id"]
    assert result["native_outcome"]["native_final_event"] is False
    assert result["document_delivery_complete"] is False
    assert section.observed["settled"] is True


def test_terminal_handoff_returns_a_copy(section):
    def mutate_copy(handlers, gateway, config):
        config["terminal_handoff_state"]()["summary"] = "mutated detached copy"
    result = section.run(mutate_copy)
    assert result["status"] == "completed"
    assert result["handoff"]["summary"] == "本分工条目已保存"


@pytest.mark.parametrize("operation", ["no_op", "failed_save", "extra_read"])
def test_no_content_change_within_batch_preserves_terminal_handoff(section, operation):
    def same_batch(handlers, gateway, config):
        if operation == "extra_read":
            assert handlers["read_source"](section.manifest["units"][1]["id"])["ok"]
        else:
            entry = deepcopy(section.entry)
            if operation == "failed_save":
                entry["field"] = "未知字段"
            saved = handlers["save_entries"]([entry])
            assert saved["ok"] is (operation == "no_op")
            if operation == "no_op":
                assert saved["changed"] is False
    result = section.run(same_batch)
    assert result["status"] == "completed"
    assert result["model_calls"] == 1
    if operation == "extra_read":
        extra_id = section.manifest["units"][1]["id"]
        assert extra_id not in result["entries"]["reads"]
        assert extra_id not in result["source_delivery"]["confirmed_ranges"]
        ledger = json.loads((section.directory / "source-delivery" / "source-delivery.json").read_text())
        pending = [page for page in ledger["pages"].values() if page["kind"] == "tool"]
        assert pending and all(page["status"] == "pending" for page in pending)


def test_content_mutation_after_handoff_in_same_batch_prevents_terminal_success(section):
    def mutation(handlers, gateway, config):
        assert handlers["save_entries"]([{**section.entry, "value": "更正桥名"}])["changed"]
        assert config["terminal_handoff_state"]() is None
    result = section.run(mutation)
    assert result["status"] == "failed"
    assert result["handoff"] is None


def test_terminal_success_rejects_an_additional_generation(section):
    def extra_generation(handlers, gateway, config):
        gateway.generation()
    result = section.run(extra_generation)
    assert result["status"] == "failed"
    assert result["model_calls"] == 2
    assert section.observed["settled"] is False


@pytest.mark.parametrize("fault", ["done", "http", "status", "finish", "request_id", "usage", "missing_usage",
                                   "inconsistent_usage", "checked_usage", "duplicate_call", "duplicate_audit",
                                   "unknown_attempt", "unmatched_gateway", "forward_busy"])
def test_terminal_success_requires_matching_settled_generation(section, fault):
    def corrupt(handlers, gateway, config):
        call, audit = section.client.calls[-1], gateway.receipts[-1]
        if fault == "done":
            call["done_received"] = False
        elif fault == "http":
            audit["http_status"] = 500
        elif fault == "status":
            audit["status"] = "rejected"
        elif fault == "finish":
            call["finish_reason"] = audit["finish_reason"] = "length"
        elif fault == "request_id":
            audit["request_id"] = "other-request"
        elif fault == "usage":
            audit["usage"]["completion_tokens"] += 1
        elif fault == "missing_usage":
            call.pop("usage")
        elif fault == "inconsistent_usage":
            call["usage"]["total_tokens"] += 1
            audit["usage"] = deepcopy(call["usage"])
        elif fault == "checked_usage":
            audit["prompt_tokens_checked"] += 1
        elif fault == "duplicate_call":
            section.client.calls.append(deepcopy(call))
        elif fault == "duplicate_audit":
            gateway.receipts.append(deepcopy(audit))
        elif fault == "unknown_attempt":
            section.client.calls.insert(0, {"kind": "agent_generation", "call_id": "unknown", "done_received": False})
        elif fault == "unmatched_gateway":
            gateway.receipts.insert(0, {"request_kind": "agent_generation", "call_id": "unmatched", "generation_attempted": True})
        elif fault == "forward_busy":
            gateway._forward_lock.acquire()
        assert config["terminal_generation_is_settled"]() is False
    result = section.run(corrupt)
    assert result["status"] == "failed"


@pytest.mark.parametrize("fault", ["hash", "origin", "native_final", "outcome_error", "failed_status",
                                   "unpaired", "post_request", "sdk_stop", "iteration", "callbacks", "active", "lock"])
def test_terminal_success_requires_adapter_proof_and_clean_boundary(section, fault):
    def corrupt(outcome):
        if fault == "hash":
            outcome["terminal_handoff_sha256"] = "0" * 64
        elif fault == "origin":
            outcome["terminal_handoff_origin_call_id"] = "another-call"
        elif fault == "native_final":
            outcome["native_final_event"] = True
        elif fault == "outcome_error":
            outcome["error"] = {"type": "RuntimeError"}
        elif fault == "failed_status":
            outcome["status"] = "failed"
        elif fault in {"unpaired", "post_request", "sdk_stop"}:
            key = {"unpaired": "all_batch_tool_calls_paired", "post_request": "stopped_before_next_model_call",
                   "sdk_stop": "sdk_stop_returned"}[fault]
            outcome["terminal_handoff_proof"][key] = False
        else:
            key = {"iteration": "iteration_exited", "callbacks": "tool_callbacks_drained",
                   "active": "active_session_clear", "lock": "session_lock_released"}[fault]
            outcome["session_boundary"][key] = False
    assert section.run(mutate_outcome=corrupt)["status"] == "failed"


def test_failed_handoff_write_does_not_expose_a_terminal_receipt(section, monkeypatch):
    original = worker.atomic_json
    def cannot_persist(path, value):
        if path.name == "handoff.json":
            raise OSError("synthetic handoff persistence failure")
        return original(path, value)
    monkeypatch.setattr(worker, "atomic_json", cannot_persist)
    def fail_write(handlers, gateway, config):
        with pytest.raises(OSError, match="handoff persistence"):
            handlers["complete_section"]("not saved", [])
        assert config["terminal_handoff_state"]() is None
    result = section.run(fail_write, save_handoff=False)
    assert result["status"] == "failed"
    assert result["handoff"] is None


@pytest.mark.parametrize("fault", ["changed", "missing", "invalid_json"])
def test_current_handoff_requires_the_persisted_receipt(section, fault):
    def change_file(handlers, gateway, config):
        path = section.directory / "handoff.json"
        if fault == "missing":
            path.unlink()
        elif fault == "invalid_json":
            path.write_text("invalid json")
        else:
            handoff = json.loads(path.read_text())
            handoff["summary"] = "unexpected overwritten receipt"
            atomic_json(path, handoff)
        assert config["terminal_handoff_state"]() is None
    assert section.run(change_file)["status"] == "failed"
