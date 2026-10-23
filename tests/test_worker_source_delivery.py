"""Worker source coverage requires final payload plus upstream input evidence."""
from copy import deepcopy
import json
import time
from types import SimpleNamespace

import pytest

from bridge_evidence.contest import document_multiagent_worker as worker
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import atomic_json, json_digest
from bridge_evidence.contest.tong_react_adapter import wrap_tool_response
from test_document_multiagent import FakeGateway


def setup_worker(tmp_path, monkeypatch, *, embedded=True):
    manifest = manifest_from_text("合成桥。\n本次检查支座存在锈蚀。")
    path = tmp_path / "source.json"
    atomic_json(path, manifest)
    request = {"manifest_path": str(path), "manifest_sha256": json_digest(manifest),
        "task": {"id": "part1", "primary_evidence_ids": [unit["id"] for unit in manifest["units"]]},
        "model_config": {"enable_thinking": False}, "deadline_monotonic": time.monotonic() + 100,
        "max_model_calls": 8}
    client = SimpleNamespace(calls=[], model="own-model", api_key="synthetic", count_tokens=len)
    monkeypatch.setattr(worker, "AuditedModelGateway", FakeGateway)
    if not embedded:
        monkeypatch.setattr(worker, "section_prompt", lambda *_args, **_kwargs: ("只读任务导航；正文请用工具。", []))
    return manifest, request, client


def checkpoint_reads(directory):
    return json.loads((directory / "entries" / "entries.json").read_text())["reads"]


def prepare(gateway, payload, request_id):
    audit = {"request_id": request_id, "call_id": "call-" + request_id,
             "http_status": 200, "input_delivery_confirmed": True,
             "input_delivery_evidence": "generated_stream_data"}
    gateway.on_prepared_request(payload, audit)
    return audit


@pytest.mark.parametrize("case", ["prepared_only", "http_400", "bad_output", "initial_missing"])
def test_initial_source_confirmed_only_after_actual_input_observation(tmp_path, monkeypatch, case):
    manifest, request, client = setup_worker(tmp_path, monkeypatch)
    directory = tmp_path / "child"

    def react(prompt, handlers, config):
        gateway = FakeGateway.instance
        assert checkpoint_reads(directory) == {}
        assert not any(name.endswith("_range") for name in handlers)
        assert "_range工具" not in config["system_prompt"]
        content = "摘要而非原文" if case == "initial_missing" else prompt
        payload = {"messages": [{"role": "user", "content": content}]}
        audit = prepare(gateway, payload, "initial")
        assert checkpoint_reads(directory) == {}
        if case in {"bad_output", "initial_missing"}:
            gateway.on_input_observed(payload, audit)
        if case in {"http_400", "bad_output"}:
            gateway.on_receipt({**audit, "status": "error",
                "http_status": 400 if case == "http_400" else 200,
                "error_code": "HTTP_400" if case == "http_400" else "content_repetition"})
        return {"status": "failed", "native_final_event": False}

    result = worker.run_section(request, directory, client=client, react_runner=react)
    confirmed = {unit["id"]: [[0, len(unit["text"])]] for unit in manifest["units"]} if case == "bad_output" else {}
    assert result["entries"]["reads"] == confirmed
    assert result["source_delivery"]["confirmed_ranges"] == confirmed
    assert result["source_delivery"]["main_agent_observed"] is False
    assert result["source_delivery"]["semantic_review"] is False
    if case == "bad_output":
        receipt = result["source_delivery"]["confirmed_requests"][0]
        assert receipt["request_id"] == "initial"
        assert receipt["evidence"]["input_delivery_evidence"] == "generated_stream_data"
    assert result["status"] == "failed"


@pytest.mark.parametrize("wire", ["absent", "summary", "original"])
def test_source_tool_fetch_is_pending_until_its_original_text_is_in_accepted_request(tmp_path, monkeypatch, wire):
    manifest, request, client = setup_worker(tmp_path, monkeypatch, embedded=False)
    directory = tmp_path / "child"
    unit = manifest["units"][0]

    def react(prompt, handlers, config):
        gateway = FakeGateway.instance
        page = handlers["read_source"](unit["id"])
        assert page["ok"] and "source_delivery_receipt" in page
        assert checkpoint_reads(directory) == {}
        entry = {"id": "name", "section": "基本信息", "field": "桥梁名称", "value": "合成桥",
                 "status": "explicit", "source_refs": [{"evidence_id": unit["id"]}]}
        assert handlers["save_entries"]([entry])["ok"] is False
        content = deepcopy(page)
        if wire == "summary":
            content = {"ok": True, "source_delivery_receipt": page["source_delivery_receipt"], "summary": "已阅读桥名"}
        payload = {"messages": ([{"role": "tool", "tool_call_id": "native-read",
            "content": json.dumps(wrap_tool_response("read_source", content), ensure_ascii=False)}]
            if wire != "absent" else [{"role": "user", "content": "继续"}])}
        audit = prepare(gateway, payload, "source-page")
        assert checkpoint_reads(directory) == {}
        gateway.on_input_observed(payload, audit)
        if wire == "original":
            assert handlers["save_entries"]([entry])["ok"] is True
        else:
            assert handlers["save_entries"]([entry])["ok"] is False
        return {"status": "failed", "native_final_event": False}

    result = worker.run_section(request, directory, client=client, react_runner=react)
    # The returned neighboring source context is original text too, so both
    # units become confirmed when the unmodified complete page reaches input.
    expected = {row["id"]: [[0, len(row["text"])]] for row in manifest["units"]} if wire == "original" else {}
    assert result["entries"]["reads"] == expected
    assert result["source_delivery"]["confirmed_ranges"] == expected
    ledger = json.loads((directory / "source-delivery" / "source-delivery.json").read_text())
    pages = [row for row in ledger["pages"].values() if row["kind"] == "tool"]
    assert pages[0]["status"] == ("confirmed" if wire == "original" else "pending")


def test_confirmed_source_ledger_survives_worker_restart_with_same_identity(tmp_path, monkeypatch):
    manifest, request, client = setup_worker(tmp_path, monkeypatch)
    directory = tmp_path / "child"

    def first(prompt, _handlers, _config):
        gateway = FakeGateway.instance
        payload = {"messages": [{"role": "user", "content": prompt}]}
        audit = prepare(gateway, payload, "initial")
        gateway.on_input_observed(payload, audit)
        return {"status": "failed", "native_final_event": False}

    initial = worker.run_section(request, directory, client=client, react_runner=first)
    recovered = worker.run_section(request, directory, client=client,
        react_runner=lambda *_args: {"status": "failed", "native_final_event": False})
    assert recovered["entries"]["reads"] == initial["entries"]["reads"]
    assert recovered["source_delivery"] == initial["source_delivery"]
    assert recovered["source_delivery"]["identity"]["agent_id"] == request["task"]["id"]
    assert recovered["source_delivery"]["identity"]["manifest_sha256"] == json_digest(manifest)
