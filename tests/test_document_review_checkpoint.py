"""Synthetic checkpoint persistence only; no report, SDK, HTTP or model use."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest import document_review_checkpoint as module
from bridge_evidence.contest.run_state import json_digest


def identity():
    return {"workflow": {"model": "synthetic-own", "source": "synthetic-source"},
            "semantic_state_hash": "a" * 64, "payload_sha256": "b" * 64,
            "catalog_sha256": "c" * 64}


def state():
    return {
        "messages": [
            {"role": "system", "content": "合成审核系统提示"},
            {"role": "user", "content": '{"direction":"REVERSE"}'},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "synthetic-tool-1", "type": "function", "function": {
                    "name": "review_read_draft_page", "arguments": '{"cursor":2,"limit":2}'}}]},
            {"role": "tool", "tool_call_id": "synthetic-tool-1",
             "content": '{"items":[{"path":"病害明细.2","value":"待送内容：零、未知、左幅\\n原样保留"}],"next_cursor":4}'},
        ],
        "lookup_state": {"snapshot_sha256": "c" * 64, "seen_paths": ["病害明细.0", "病害明细.1"],
                         "calls": [{"name": "review_read_draft_page", "ok": True}],
                         "semantic_memory": {"candidates": ["独立内存内容"]}},
        "model_call_ids": ["synthetic-generation-1"],
        "pending_response": None,
        "iterations_spent": 1, "tool_calls_spent": 1, "history_reset_count": 0,
    }


def checkpoint(tmp_path, binding=None):
    return module.ReviewCheckpoint(tmp_path / "review" / "partial.json", binding or identity())


def disk(cp):
    return json.loads(cp.path.read_text(encoding="utf-8"))


def replace(cp, record, *, reseal=False, reseal_state=False):
    record = deepcopy(record)
    if reseal_state:
        record["state_sha256"] = json_digest(record["state"])
    if reseal:
        record.pop("checkpoint_sha256", None)
        record["checkpoint_sha256"] = json_digest(record)
    cp.path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def test_missing_checkpoint_is_none_without_creation_or_invented_progress(tmp_path):
    cp = checkpoint(tmp_path)
    assert cp.load() is None
    assert not cp.path.parent.exists()
    cp.clear()
    assert not cp.path.parent.exists()


def test_round_trip_preserves_complete_pending_tool_result_and_nested_lookup_state(tmp_path):
    cp = checkpoint(tmp_path)
    expected = state()
    cp.save(expected)
    assert checkpoint(tmp_path).load() == expected
    assert checkpoint(tmp_path).load()["messages"][-1]["role"] == "tool"
    record = disk(cp)
    assert record["schema_version"] == 2
    assert record["state_sha256"] == json_digest(expected)
    assert record["identity"] == identity()
    assert record["identity_sha256"] == json_digest(identity())
    assert record["checkpoint_sha256"] == json_digest({k: v for k, v in record.items()
                                                       if k != "checkpoint_sha256"})


def test_inputs_and_restored_state_are_independent_copies(tmp_path):
    binding = identity()
    cp = checkpoint(tmp_path, binding)
    binding["workflow"]["model"] = "mutated"
    value = state()
    cp.save(value)
    value["messages"][-1]["content"] = "changed"
    value["lookup_state"]["seen_paths"].append("forged")
    restored = cp.load()
    restored["lookup_state"]["calls"].clear()
    restored["model_call_ids"].append("forged")
    assert cp.load() == state()


@pytest.mark.parametrize("field", list(identity()))
def test_each_identity_component_blocks_load_overwrite_and_clear(tmp_path, field):
    cp = checkpoint(tmp_path)
    cp.save(state())
    before = cp.path.read_bytes()
    changed = identity()
    changed[field] = {"changed": True} if field == "workflow" else "d" * 64
    foreign = checkpoint(tmp_path, changed)
    for action in (foreign.load, lambda: foreign.save(state()), foreign.clear):
        with pytest.raises(ValueError, match="identity"):
            action()
        assert cp.path.read_bytes() == before


@pytest.mark.parametrize("change", ["extra_field", "missing_field", "version", "bool_version", "identity_hash",
                                    "state_hash", "envelope_hash", "messages", "lookup_state", "counter"])
def test_corruption_or_schema_change_is_not_restored(tmp_path, change):
    cp = checkpoint(tmp_path)
    cp.save(state())
    record = disk(cp)
    if change == "extra_field":
        record["other"] = 1
    elif change == "missing_field":
        del record["identity_sha256"]
    elif change == "version":
        record["schema_version"] = 1
    elif change == "bool_version":
        record["schema_version"] = True
    elif change in {"identity_hash", "state_hash", "envelope_hash"}:
        key = {"identity_hash": "identity_sha256", "state_hash": "state_sha256",
               "envelope_hash": "checkpoint_sha256"}[change]
        record[key] = "0" * 64
    elif change == "messages":
        record["state"]["messages"][-1]["content"] = "different result"
    elif change == "lookup_state":
        record["state"]["lookup_state"]["seen_paths"].append("invented path")
    else:
        record["state"]["iterations_spent"] = 42
    replace(cp, record)
    with pytest.raises(ValueError, match="Cannot restore"):
        cp.load()


@pytest.mark.parametrize("change", ["version", "state_field", "role", "boolean_counter", "duplicate_id"])
def test_invalid_schema_is_rejected_even_with_recomputed_hashes(tmp_path, change):
    cp = checkpoint(tmp_path)
    cp.save(state())
    record = disk(cp)
    if change == "version":
        record["schema_version"] = 5
    elif change == "state_field":
        record["state"]["extra"] = "not in protocol"
    elif change == "role":
        record["state"]["messages"][-1]["role"] = ["tool"]
    elif change == "boolean_counter":
        record["state"]["history_reset_count"] = False
    else:
        record["state"]["model_call_ids"] *= 2
    replace(cp, record, reseal=True, reseal_state=True)
    with pytest.raises(ValueError, match="Cannot restore"):
        cp.load()


@pytest.mark.parametrize("content", [b'{"broken":', b'\xff\xfeinvalid',
                                     b'{"schema_version":1,"schema_version":1}', b'{"a":NaN}'])
def test_invalid_json_is_not_silently_reset_or_cleared(tmp_path, content):
    cp = checkpoint(tmp_path)
    cp.path.parent.mkdir(parents=True)
    cp.path.write_bytes(content)
    for action in (cp.load, lambda: cp.save(state()), cp.clear):
        with pytest.raises(ValueError, match="Cannot restore"):
            action()
        assert cp.path.read_bytes() == content


@pytest.mark.parametrize("field", ["messages", "lookup_state", "model_call_ids", "pending_response", "iterations_spent",
                                   "tool_calls_spent", "history_reset_count"])
def test_required_state_fields_cannot_be_omitted(tmp_path, field):
    cp = checkpoint(tmp_path)
    value = state()
    del value[field]
    with pytest.raises(ValueError, match="schema fields"):
        cp.save(value)
    assert not cp.path.exists()


@pytest.mark.parametrize("field,value", [
    ("messages", []), ("messages", [{"role": "invented"}]), ("messages", ["not an object"]),
    ("messages", [{"role": []}]), ("lookup_state", []),
    ("model_call_ids", [""]), ("model_call_ids", [" "]), ("model_call_ids", [1]),
    ("model_call_ids", ["same", "same"]), ("iterations_spent", True),
    ("iterations_spent", -1), ("tool_calls_spent", 1.5), ("history_reset_count", None),
])
def test_wrong_state_types_are_rejected_before_write(tmp_path, field, value):
    cp = checkpoint(tmp_path)
    candidate = state()
    candidate[field] = value
    with pytest.raises(ValueError):
        cp.save(candidate)
    assert not cp.path.exists()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), (1, 2), {1: "coerced key"}, object()])
def test_non_json_nested_state_or_identity_is_not_coerced(tmp_path, bad):
    cp = checkpoint(tmp_path)
    value = state()
    value["lookup_state"]["invalid"] = bad
    with pytest.raises(ValueError):
        cp.save(value)
    with pytest.raises(ValueError):
        module.ReviewCheckpoint(cp.path, {"invalid": bad})
    assert not cp.path.exists()


def test_save_allows_history_reset_and_forward_progress_without_losing_ledger(tmp_path):
    cp = checkpoint(tmp_path)
    cp.save(state())
    progressed = state()
    progressed["messages"] = progressed["messages"][:2] + progressed["messages"][-1:]
    progressed["iterations_spent"] += 2
    progressed["tool_calls_spent"] += 1
    progressed["history_reset_count"] = 1
    progressed["model_call_ids"].extend(["synthetic-generation-2", "synthetic-generation-3"])
    progressed["lookup_state"]["seen_paths"].append("病害明细.2")
    cp.save(progressed)
    assert checkpoint(tmp_path).load() == progressed
    cp.save(progressed)  # Idempotent save is allowed.
    assert cp.load() == progressed


@pytest.mark.parametrize("field", ["iterations_spent", "tool_calls_spent", "history_reset_count", "model_call_ids"])
def test_spent_budget_and_prior_call_ids_cannot_be_rolled_back(tmp_path, field):
    cp = checkpoint(tmp_path)
    value = state()
    value["history_reset_count"] = 1
    cp.save(value)
    candidate = deepcopy(value)
    candidate[field] = [] if field == "model_call_ids" else 0
    with pytest.raises(ValueError, match="cannot"):
        cp.save(candidate)
    assert cp.load() == value


def test_atomic_replace_failure_keeps_previous_checkpoint_and_cleans_partial_file(tmp_path, monkeypatch):
    cp = checkpoint(tmp_path)
    cp.save(state())
    before = cp.path.read_bytes()
    value = state()
    value["iterations_spent"] += 1
    def fail_replace(*_args, **_kwargs):
        raise OSError("synthetic replace failure")
    monkeypatch.setattr("bridge_evidence.contest.run_state.os.replace", fail_replace)
    with pytest.raises(OSError, match="save could not be persisted"):
        cp.save(value)
    assert cp.path.read_bytes() == before
    assert list(cp.path.parent.iterdir()) == [cp.path]
    assert cp.load() == state()


def test_atomic_first_save_failure_does_not_publish_partial_checkpoint(tmp_path, monkeypatch):
    cp = checkpoint(tmp_path)
    def fail_replace(*_args, **_kwargs):
        raise OSError("synthetic replace failure")
    monkeypatch.setattr("bridge_evidence.contest.run_state.os.replace", fail_replace)
    with pytest.raises(OSError, match="save"):
        cp.save(state())
    assert cp.load() is None
    assert not list(cp.path.parent.iterdir())


def test_clear_is_idempotent_and_removes_only_bound_checkpoint(tmp_path):
    cp = checkpoint(tmp_path)
    cp.save(state())
    cp.clear()
    assert checkpoint(tmp_path).load() is None
    cp.clear()
    assert not cp.path.exists()


def test_clear_failure_keeps_existing_progress(tmp_path, monkeypatch):
    cp = checkpoint(tmp_path)
    cp.save(state())
    def fail_unlink(*_args, **_kwargs):
        raise OSError("synthetic unlink failure")
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(OSError, match="clear could not remove"):
        cp.clear()
    assert cp.load() == state()


def pending(*, tools=False):
    message = {"role": "assistant", "content": '{"passed":true,"unresolved_items":[]}'}
    if tools:
        message = {"role": "assistant", "content": None,
                   "tool_calls": [{"id": "synthetic-pending-tool", "type": "function", "function": {
                       "name": "review_read_draft_page", "arguments": '{"cursor":4,"limit":2}'}}]}
    return {"call_id": "synthetic-generation-1", "response": {
        "id": "synthetic-response", "model": "synthetic-own", "created": 123,
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tools else "stop", "message": message}],
        "usage": {"prompt_tokens": 250, "completion_tokens": 40, "total_tokens": 290},
        "custom_metadata": {"synthetic": True, "raw": "不得裁剪的尾部"},
    }}


@pytest.mark.parametrize("tools", [False, True])
def test_pending_final_or_tool_response_is_preserved_before_workflow_consumes_it(tmp_path, tools):
    cp = checkpoint(tmp_path)
    value = state()
    value["pending_response"] = pending(tools=tools)
    cp.save(value)
    restored = checkpoint(tmp_path).load()
    assert restored == value
    assert json_digest(restored["pending_response"]["response"]) == json_digest(value["pending_response"]["response"])
    assert restored["messages"][-1]["role"] == "tool"  # Existing unsent result is also kept.
    consumed = deepcopy(restored)
    consumed["messages"].append(consumed["pending_response"]["response"]["choices"][0]["message"])
    consumed["pending_response"] = None
    cp.save(consumed)
    assert cp.load() == consumed
    assert cp.load()["model_call_ids"] == ["synthetic-generation-1"]
    assert cp.load()["iterations_spent"] == 1


@pytest.mark.parametrize("change", [
    "missing_call_id", "extra_field", "wrong_call_id", "call_id_not_last", "no_calls", "zero_iterations",
    "response_list", "no_choices", "empty_choices", "choice_string", "no_message", "wrong_role",
    "non_string_content", "no_content_or_tools", "tools_string", "tool_string", "no_tool_id", "no_function",
    "wrong_tool_type", "no_function_name", "non_string_arguments", "duplicate_tool_id", "nonfinite_response",
])
def test_malformed_pending_response_is_rejected_without_overwriting_state(tmp_path, change):
    cp = checkpoint(tmp_path)
    cp.save(state())
    candidate = state()
    candidate["pending_response"] = pending(tools=True)
    record = candidate["pending_response"]
    response = record["response"]
    message = response["choices"][0]["message"]
    if change == "missing_call_id":
        del record["call_id"]
    elif change == "extra_field":
        record["ignored"] = "no"
    elif change == "wrong_call_id":
        record["call_id"] = "different"
    elif change == "call_id_not_last":
        candidate["model_call_ids"].append("synthetic-generation-2")
    elif change == "no_calls":
        candidate["model_call_ids"] = []
    elif change == "zero_iterations":
        candidate["iterations_spent"] = 0
    elif change == "response_list":
        record["response"] = []
    elif change == "no_choices":
        del response["choices"]
    elif change == "empty_choices":
        response["choices"] = []
    elif change == "choice_string":
        response["choices"] = ["not an object"]
    elif change == "no_message":
        del response["choices"][0]["message"]
    elif change == "wrong_role":
        message["role"] = "tool"
    elif change == "non_string_content":
        message["content"] = ["not plain text"]
    elif change == "no_content_or_tools":
        message["tool_calls"] = []
    elif change == "tools_string":
        message["tool_calls"] = "not an array"
    elif change == "tool_string":
        message["tool_calls"] = ["not an object"]
    elif change == "no_tool_id":
        del message["tool_calls"][0]["id"]
    elif change == "no_function":
        del message["tool_calls"][0]["function"]
    elif change == "wrong_tool_type":
        message["tool_calls"][0]["type"] = "unavailable"
    elif change == "no_function_name":
        del message["tool_calls"][0]["function"]["name"]
    elif change == "non_string_arguments":
        message["tool_calls"][0]["function"]["arguments"] = {"cursor": 0}
    elif change == "duplicate_tool_id":
        message["tool_calls"].append(deepcopy(message["tool_calls"][0]))
    else:
        response["custom_metadata"]["bad"] = float("nan")
    with pytest.raises(ValueError):
        cp.save(candidate)
    assert cp.load() == state()


def test_pending_response_is_not_accepted_with_resealed_invalid_structure(tmp_path):
    cp = checkpoint(tmp_path)
    cp.save(state())
    record = disk(cp)
    record["state"]["pending_response"] = pending()
    record["state"]["pending_response"]["response"]["choices"][0]["message"]["role"] = "user"
    replace(cp, record, reseal=True, reseal_state=True)
    with pytest.raises(ValueError, match="assistant"):
        cp.load()


def test_pending_response_deep_copy_does_not_mutate_after_save(tmp_path):
    cp = checkpoint(tmp_path)
    value = state()
    value["pending_response"] = pending()
    cp.save(value)
    value["pending_response"]["response"]["usage"]["prompt_tokens"] = 999
    first = cp.load()
    first["pending_response"]["response"]["choices"][0]["message"]["content"] = "changed"
    assert cp.load()["pending_response"] == pending()
