"""Actual projected wire batches determine the protected native suffix.

All messages are synthetic; no SDK, tokenizer, network or model execution.
"""
from copy import deepcopy
import hashlib
import json
import socket
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bridge_evidence.contest import document_context_compression as native
from bridge_evidence.contest import tong_react_adapter as adapter
from bridge_evidence.contest.document_history_continuity import HistoryPrefixGuard
from test_rejected_argument_history_projection import fixture as rejected_fixture


@pytest.fixture(autouse=True)
def no_external_actions(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("No external actions")
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)


class ToolResultMessage:
    def __init__(self, row):
        self.name = row.get("name", "save_entries")
        self.tool_call_id = row["tool_call_id"]


def native_messages(wire):
    result = []
    names = {}
    for row in wire:
        if row.get("tool_calls"):
            calls = []
            for call in row["tool_calls"]:
                names[call["id"]] = call["function"]["name"]
                calls.append(SimpleNamespace(tool_call_id=call["id"],
                    tool_name=call["function"]["name"], arguments=call["function"]["arguments"]))
            content = SimpleNamespace(tool_call_messages=calls)
        elif row.get("role") == "tool":
            content = ToolResultMessage({**row, "name": names[row["tool_call_id"]]})
        else:
            content = row.get("content")
        result.append(SimpleNamespace(chunks=[SimpleNamespace(content=content)], wire=deepcopy(row)))
    return result


def pair(key, name="read_source"):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": key, "type": "function",
        "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": key, "content": "{}"}]


def rejected_batch(index, mixed=False):
    payload, state = rejected_fixture(mixed)
    message = payload["messages"][2]
    keys = {call["id"]: f"rejected-{index}-{call['id']}" for call in message["tool_calls"]}
    recovery, owned = state["parse_recovery"][0], state["argument_parse_errors"][0]
    owned["agent_call_id"] = recovery["agent_call_id"] = f"settled-{index}"
    owned_hash = adapter._hash(owned)
    root_hash = adapter._hash(keys["bad"])
    recovery.update(adapter_parse_error_sha256=owned_hash, tool_call_id_sha256=root_hash)
    for call in message["tool_calls"]:
        call["id"] = keys[call["id"]]
    for result in payload["messages"][3:-1]:
        old_key = result["tool_call_id"]
        result["tool_call_id"] = keys[old_key]
        if old_key == "good":
            continue
        value = json.loads(result["content"])
        error = value["error"]
        if old_key == "bad":
            error["adapter_parse_error_sha256"] = owned_hash
        else:
            error["blocked_by_tool_call_id_sha256"] = root_hash
        result["content"] = native.canonical(value).decode()
        proof = next(row for row in recovery["paired_batch_results"]
                     if row["tool_call_id_sha256"] == adapter._hash(old_key))
        proof.update(tool_call_id_sha256=adapter._hash(keys[old_key]), result_sha256=adapter._hash(value))
    return payload["messages"][2:], state


def history(bad_count=1, mixed=False):
    messages = [{"role": "system", "content": "Synthetic stable system"},
                {"role": "user", "content": "Synthetic original source"}]
    messages += pair("older") + pair("A") + pair("B")
    state = {"argument_parse_errors": [], "parse_recovery": [], "tool_history_projections": []}
    for index in range(bad_count):
        fragment, partial = rejected_batch(index, mixed)
        messages.extend(fragment)
        messages.append({"role": "user", "content": f"Synthetic ordinary turn {index}"})
        for key in ("argument_parse_errors", "parse_recovery"):
            state[key].extend(partial[key])
    return {"model": "synthetic", "messages": messages}, state


def admit_compression(tmp_path, guard, old_wire, state, boundary):
    """Use real durable refs and real guard; only the native messages are fake."""
    summary, progress = "Synthetic semantic note", "Synthetic exact progress"
    old_plain = [{"wire": deepcopy(row)} for row in old_wire]
    replacement = [{"text": summary}, {"text": progress}, *deepcopy(old_plain[boundary:])]
    transition = {"schema": "cityagent-native-compression-transition-v1", "index": 1,
        "old_messages_sha256": guard.snapshot()["last_messages_sha256"],
        "same_agent": True, "same_context": True, "retained_native_start": boundary,
        "summary_sha256": hashlib.sha256(summary.encode()).hexdigest(),
        "exact_progress_sha256": hashlib.sha256(progress.encode()).hexdigest(),
        "archived_history_ref": native.save(tmp_path / "old.json", old_plain),
        "replacement_memory_ref": native.save(tmp_path / "new.json", replacement),
        "summary_ref": native.save(tmp_path / "summary.json", {"text": summary}),
        "exact_progress_ref": native.save(tmp_path / "progress.json", {"text": progress})}
    guard.authorize_compression(transition, summary_text=summary, exact_progress_text=progress)
    compressed = {"model": "synthetic", "messages": [deepcopy(old_wire[0]),
        {"role": "user", "content": summary}, {"role": "user", "content": progress},
        *deepcopy(old_wire[boundary:])]}
    actual = adapter._project_rejected_tool_history(compressed, state)
    original = deepcopy(actual)
    receipt = guard.observe_complete_request(actual)
    assert actual == original, "Continuity admission must not rewrite the wire"
    assert receipt["authorized_compression"] is True
    assert not guard.snapshot()["poisoned"]
    return actual


@pytest.mark.parametrize("bad_count", [1, 2, 7])
def test_multiple_failed_native_cycles_do_not_displace_actual_wire_suffix(tmp_path, bad_count):
    payload, state = history(bad_count)
    guard = HistoryPrefixGuard()
    guard.observe_complete_request(adapter._project_rejected_tool_history(payload, state))
    assert guard.compression_required_tool_call_ids() == ("A", "B")
    old_wire = [*deepcopy(payload["messages"]), *pair("C")]
    old = native_messages(old_wire)
    default = native.retained_boundary(old)
    boundary = native.retained_boundary(old, required_tool_call_ids=guard.compression_required_tool_call_ids())
    start_a = next(i for i, row in enumerate(old_wire) if any(
        call["id"] == "A" for call in row.get("tool_calls", [])))
    assert boundary == start_a < default
    before = deepcopy(old_wire)
    actual = admit_compression(tmp_path, guard, old_wire, state, boundary)
    assert old_wire == before
    assert actual["messages"][3]["tool_calls"][0]["id"] == "A"
    assert len([row for row in actual["messages"] if row.get("role") == "user"
                and row.get("content", "").startswith("CITYAGENT_REJECTED_TOOL_HISTORY_V1")]) == bad_count


def test_mixed_batch_keeps_success_siblings_and_full_inspection_pair(tmp_path):
    payload, state = history(mixed=True)
    payload["messages"][2:2] = pair("inspection", "inspect_delivery")
    projected = adapter._project_rejected_tool_history(payload, state)
    guard = HistoryPrefixGuard(); guard.observe_complete_request(projected)
    required = guard.compression_required_tool_call_ids()
    assert "inspection" in required
    assert "rejected-0-good" in required and "rejected-0-aborted-valid" in required
    assert "rejected-0-bad" not in required
    old_wire = [*deepcopy(payload["messages"]), *pair("C")]
    boundary = native.retained_boundary(native_messages(old_wire), required_tool_call_ids=required)
    actual = admit_compression(tmp_path, guard, old_wire, state, boundary)
    for key in ("inspection", "rejected-0-good", "rejected-0-aborted-valid"):
        assert next(row for row in actual["messages"] if row.get("tool_call_id") == key) == next(
            row for row in projected["messages"] if row.get("tool_call_id") == key)


def test_already_compressed_history_does_not_require_discarded_old_calls(tmp_path):
    payload, state = history(3)
    payload["messages"] = [{"role": "system", "content": "Synthetic stable system"},
        {"role": "user", "content": "Previous summary without raw rejected history"},
        *pair("A"), *pair("B")]
    guard = HistoryPrefixGuard(); guard.observe_complete_request(payload)
    old_wire = [*deepcopy(payload["messages"]), *pair("C")]
    boundary = native.retained_boundary(native_messages(old_wire),
        required_tool_call_ids=guard.compression_required_tool_call_ids())
    actual = admit_compression(tmp_path, guard, old_wire, state, boundary)
    assert not any("CITYAGENT_REJECTED_TOOL_HISTORY_V1" in row.get("content", "") for row in actual["messages"])


@pytest.mark.parametrize("fault", ["missing_call", "missing_result", "duplicate_required", "invalid_id"])
def test_missing_or_ambiguous_native_pair_fails_closed(fault):
    messages = native_messages(pair("A") + pair("B"))
    required = ("A", "B")
    if fault == "missing_call": required = ("absent",)
    elif fault == "missing_result": messages.pop(1)
    elif fault == "duplicate_required": required = ("A", "A")
    else: required = (None,)
    with pytest.raises(ValueError):
        native.retained_boundary(messages, required_tool_call_ids=required)


def test_guard_navigation_is_read_only_and_tracks_current_observation():
    guard = HistoryPrefixGuard()
    assert guard.compression_required_tool_call_ids() == ()
    payload = {"messages": [{"role": "system", "content": "Synthetic"}, *pair("A"), *pair("B")]}
    guard.observe_complete_request(payload)
    before = deepcopy(guard.snapshot()), guard.receipts
    first = guard.compression_required_tool_call_ids()
    assert first == ("A", "B")
    assert (guard.snapshot(), guard.receipts) == before
    payload["messages"].extend(pair("C"))
    guard.observe_complete_request(payload)
    assert first == ("A", "B")
    assert guard.compression_required_tool_call_ids() == ("B", "C")


def test_manager_uses_actual_wire_navigation_before_archiving_or_summarizing(tmp_path):
    payload, state = history(4)
    guard = HistoryPrefixGuard(); guard.observe_complete_request(adapter._project_rejected_tool_history(payload, state))
    old = native_messages([*payload["messages"], *pair("C")])
    memory = SimpleNamespace(get_messages=lambda: old)
    context = SimpleNamespace(memory=memory)
    manager = native.NativeContextCompression(tmp_path, summary_client=None, exact_progress=None,
        before_model=lambda **kwargs: pytest.fail("No model operation"), refresh_progress=lambda: None,
        history_guard=guard, native_api={"synthetic": True})
    manager.pending = True
    class BoundaryObserved(Exception):
        pass
    with patch.object(native, "retained_boundary", wraps=native.retained_boundary) as boundary, \
            patch.object(native, "plain", return_value=[]), \
            patch.object(native, "save", side_effect=BoundaryObserved):
        with pytest.raises(BoundaryObserved):
            manager.before_native_call(SimpleNamespace(), [], context=context)
    assert boundary.call_args.kwargs["required_tool_call_ids"] == ("A", "B")
    assert context.memory is memory


def test_no_actual_tool_prefix_skips_without_summary_and_can_compress_after_real_cycle(tmp_path):
    payload, state = history()
    # No previously delivered successful tool cycle: only a rejected native
    # call rendered as ordinary text on this observed actual wire.
    payload["messages"] = payload["messages"][:2] + payload["messages"][8:]
    guard = HistoryPrefixGuard()
    guard.observe_complete_request(adapter._project_rejected_tool_history(payload, state))
    assert guard.compression_required_tool_call_ids() == ()
    old = native_messages(payload["messages"])
    assert native.retained_boundary(old) > 0
    memory = SimpleNamespace(get_messages=lambda: pytest.fail("Skipped compression must not touch memory"))
    context = SimpleNamespace(memory=memory)
    manager = native.NativeContextCompression(tmp_path, summary_client=None, exact_progress=None,
        before_model=lambda **kwargs: pytest.fail("No model operation"), refresh_progress=lambda: None,
        history_guard=guard)
    manager.observe_complete_request(payload, 200000)
    supplied = ["Original supplied input"]
    with patch.object(native, "load_native_api", side_effect=AssertionError("No SDK import")):
        assert manager.before_native_call(SimpleNamespace(), supplied, context=context) is supplied
    assert context.memory is memory
    assert not manager.pending and manager.transitions == []
    receipt = json.loads((tmp_path / "skipped-0001.json").read_bytes())
    assert receipt["reason"] == "no_authorizable_actual_wire_tool_prefix"
    assert receipt["summary_requested"] is receipt["memory_rebound"] is False
    assert receipt["actual_messages_sha256"] == guard.snapshot()["last_messages_sha256"]
    # A later real tool cycle re-enables normal protection and compression;
    # skipping has neither poisoned the guard nor disabled future triggers.
    next_payload = deepcopy(payload)
    next_payload["messages"].extend(pair("real"))
    guard.observe_complete_request(adapter._project_rejected_tool_history(next_payload, state))
    manager.observe_complete_request(next_payload, 200000)
    assert manager.pending and guard.compression_required_tool_call_ids() == ("real",)
