"""Synthetic SDK metadata/dispatch only; no model, network, report or real SDK."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
from typing import Any, get_type_hints

from jsonschema import Draft202012Validator
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_tong_react_adapter import TextAction, fake_sdk as legacy_sdk
from test_document_agent_adapter import SchemaTool, SchemaParameters
from bridge_evidence.contest import tong_react_adapter as m

NAMES = ("read_source_objects", "read_source_object", "propose_source_object")
CONFIG = {"url": "http://127.0.0.1:8000/v1", "model": "own-model", "max_calls": 50}


def proposal():
    return {"disposition": "outside_task", "rationale": '保留原文未知项，不据此补写；含"引号"\\换行\n。',
            "affected_output_paths": [], "supporting_refs": [],
            "context_refs": [{"evidence_id": "E000001", "start": 0, "end": 12}],
            "source_limitations": ["此提议不代表已理解原生公式。"]}


def sdk(script):
    result, registry, settings, outputs, registered = legacy_sdk(script)
    decorator = result["tool"]
    metadata = {}
    def tool(**definition):
        def decorate(function):
            decorator(**definition)(function)
            instance = type("SyntheticSourceTool", (SchemaTool,), {"info_reads": []})()
            metadata[definition["name"]] = instance
            return instance
        return decorate
    result.update(tool=tool, schema_tools=metadata)
    return result, registry, settings, outputs, registered


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    assert m._ACTIVE_SESSION is None
    m._REGISTRATIONS.clear()
    monkeypatch.setattr(m, "_load_tool_schema_api", lambda: (SchemaTool, SchemaParameters))
    yield
    assert m._ACTIVE_SESSION is None


def invoke(monkeypatch, script, handlers, **config):
    bundle = sdk(script)
    monkeypatch.setattr(m, "_load_tongagents", lambda: bundle[0])
    result = m.run_react("synthetic source-object tools", handlers, {**CONFIG, **config})
    return result, bundle


def test_native_metadata_and_lossless_dispatch_preserve_host_owned_boundary(monkeypatch):
    seen = []
    values = proposal()
    def script(registry, setting, outputs, prompt):
        outputs.append(registry["read_source_objects"]())
        outputs.append(registry["read_source_object"]("so_example"))
        outputs.append(registry["propose_source_object"]("so_example", values))
        yield TextAction()
    def handler(name):
        return lambda **args: seen.append((name, deepcopy(args))) or {"ok": True, "status": "proposed", "accepted": False}
    result, bundle = invoke(monkeypatch, script, {name: handler(name) for name in NAMES})
    actual_sdk, registry, settings, outputs, registered = bundle
    assert result["status"] == "completed" and all(x["ok"] for x in outputs)
    assert seen == [(NAMES[0], {"cursor": 0, "limit": 10}), (NAMES[1], {"object_id": "so_example"}),
                    (NAMES[2], {"object_id": "so_example", "proposal": values})]
    assert set(seen[-1][1]["proposal"]) == set(values)
    assert registered == list(NAMES) == settings[0].tool_identifier_list
    assert get_type_hints(registry[NAMES[2]]) == {"object_id": str, "proposal": dict[str, Any], "return": dict[str, Any]}
    params = inspect.signature(registry[NAMES[0]]).parameters
    assert params["cursor"].default == 0 and params["limit"].default == 10
    for name, tool in actual_sdk["schema_tools"].items():
        schemas = [getattr(tool, method)().parameters.to_json_schema()
                   for method in ("get_original_tool_info", "get_tool_info")]
        assert schemas[0] == schemas[1] == m._source_object_schema(name)
        Draft202012Validator.check_schema(schemas[0])
    schema = actual_sdk["schema_tools"][NAMES[2]].parameters.to_json_schema()
    assert set(schema["properties"]["proposal"]["required"]) == set(values)
    assert schema["properties"]["proposal"]["additionalProperties"] is False
    assert Draft202012Validator(schema).is_valid({"object_id": "so_example", "proposal": values})
    receipt = result["tool_calls"][-1]
    assert receipt["args_sha256"] == m._hash({"object_id": "so_example", "proposal": values})
    assert receipt["args"]["proposal"] == {"type": "dict", "size": 6, "sha256": m._hash(values)}
    assert values["rationale"] not in json.dumps(receipt, ensure_ascii=False)


@pytest.mark.parametrize("edit", [lambda p: p.update(origin_call_ids=["invented"]),
    lambda p: p.update(accepted=True), lambda p: p.update(representation={"state": "semantics_verified"}),
    lambda p: p.update(disposition="used_as_evidence"), lambda p: p.pop("source_limitations"),
    lambda p: p.update(rationale=" "), lambda p: p.update(supporting_refs=[{"evidence_id": "E1", "start": False, "end": 2}]),
    lambda p: p.update(context_refs=[{"evidence_id": "E1", "start": 5, "end": 3}]),
    lambda p: p.update(source_limitations=[float("nan")])])
def test_invalid_proposals_are_receipted_without_calling_controller(monkeypatch, edit):
    value, calls = proposal(), []
    edit(value)
    def script(registry, setting, outputs, prompt):
        outputs.append(registry[NAMES[2]]("so_example", value))
        yield TextAction()
    result, bundle = invoke(monkeypatch, script, {NAMES[2]: lambda **args: calls.append(args)})
    assert calls == [] and bundle[3][0]["ok"] is False
    assert result["tool_error_count"] == 1 and result["tool_calls"][0]["status"] == "failed"
    assert bundle[3][0]["error"]["code"] == "SOURCE_OBJECT_ARGUMENTS"


def test_native_schema_rejects_agent_approval_and_origin_fields():
    validator = Draft202012Validator(m._source_object_schema(NAMES[2]))
    for field in ("origin_call_ids", "accepted", "representation"):
        value = proposal(); value[field] = True
        assert not validator.is_valid({"object_id": "so_example", "proposal": value})
    assert not validator.is_valid({"object_id": "so_example", "proposal": proposal(), "accepted": True})


def test_bad_pagination_or_object_identity_does_not_reach_controller(monkeypatch):
    calls = []
    def script(registry, setting, outputs, prompt):
        for cursor, limit in ((-1, 10), (False, 10), (0, 0), (0, True), (1.5, 1)):
            outputs.append(registry[NAMES[0]](cursor, limit))
        for object_id in ("", " ", None, 5):
            outputs.append(registry[NAMES[1]](object_id))
        yield TextAction()
    result, bundle = invoke(monkeypatch, script, {name: lambda **a: calls.append(a) for name in NAMES[:2]})
    assert calls == [] and len(result["tool_calls"]) == 9
    assert all(x["ok"] is False for x in bundle[3])


def test_sdk_metadata_failure_never_starts_session(monkeypatch):
    def script(*args):
        pytest.fail("No agent should execute without exact native schema")
    bundle = sdk(script)
    monkeypatch.setattr(m, "_load_tongagents", lambda: bundle[0])
    monkeypatch.setattr(m, "_load_tool_schema_api", lambda: (_ for _ in ()).throw(ImportError("synthetic unsupported SDK")))
    result = m.run_react("preflight", {NAMES[2]: lambda **a: {}}, CONFIG)
    assert result["failure_reason"] == "tool_schema_unavailable"
    assert bundle[2] == [] and result["tool_calls"] == []


def test_new_tools_remain_session_scoped_and_do_not_change_legacy_registration(monkeypatch):
    calls = []
    def script(registry, setting, outputs, prompt):
        if prompt == "objects":
            registry[NAMES[0]]()
        else:
            outputs.append(registry[NAMES[0]]())
            outputs.append(registry["finish"]())
        yield TextAction()
    bundle = sdk(script)
    monkeypatch.setattr(m, "_load_tongagents", lambda: bundle[0])
    m.run_react("objects", {NAMES[0]: lambda **a: calls.append(a)}, CONFIG)
    result = m.run_react("legacy", {"finish": lambda: {"ready": False}}, CONFIG)
    assert calls == [{"cursor": 0, "limit": 10}]
    assert result["tool_calls"][0]["status"] == "failed"
    assert bundle[3][0]["ok"] is False and bundle[3][1]["ok"] is True
