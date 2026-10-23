"""Direct-delivery SDK contracts; synthetic SDK/memory, no model or network."""
from copy import deepcopy
import inspect
import json
import socket
import subprocess
import threading
from types import SimpleNamespace

import pytest
from jsonschema import ValidationError

from bridge_evidence.contest import tong_react_adapter as adapter
from test_section_agent_adapter import Tool, Parameters, TextAction, basic, fake_sdk
from test_partial_native_batch_recovery import Memory, ToolResultMessage, call


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("No SDK, model, subprocess or network in synthetic adapter tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)
    monkeypatch.setattr(adapter, "_load_tool_schema_api", lambda: (Tool, Parameters))
    adapter._REGISTRATIONS.clear()
    adapter._REGISTERED_TOOL_CLASSES.clear()
    assert adapter._ACTIVE_SESSION is None
    yield
    adapter._REGISTRATIONS.clear()
    adapter._REGISTERED_TOOL_CLASSES.clear()
    assert adapter._ACTIVE_SESSION is None


def entry():
    value = basic()
    del value["id"]
    return value


def arguments(name):
    if name == "submit_review":
        return {"selections": ["candidate-one"], "reviews": [{"question_id": "question-one",
            "status": "resolved", "reason": "合成模型明确核对原文后的决定"}]}
    return ({"entries": [entry()], "unresolved": [], "complete": True}
            if name == "submit_section" else
            {"operations": [{"op": "select", "candidate_id": "candidate-one"}],
             "task_reviews": [{"task_id": "task-one", "reason": "原文支持，已核对"}]})


def installed(name):
    tool = type("Synthetic" + name, (Tool,), {})()
    adapter._install_tool_schema(tool, name, adapter._normalization_schema_for_tool(name))
    return tool


def run(monkeypatch, profile, handlers, script):
    sdk, registry, settings, contexts, outputs = fake_sdk(script)
    monkeypatch.setattr(adapter, "_load_tongagents", lambda: sdk)
    result = adapter.run_react("Synthetic direct delivery", handlers, {
        "profile": profile, "url": "http://127.0.0.1:8000/v1", "model": "synthetic",
        "current_agent_call_id": lambda: "synthetic-generation"})
    return result, registry, settings, contexts, outputs


@pytest.mark.parametrize("name,profile", [("submit_section", "section"), ("review_candidates", "reading"),
                                         ("submit_review", "reading")])
def test_exact_schema_and_normalized_containers_reach_only_explicit_handler(monkeypatch, name, profile):
    seen = []
    original = arguments(name)
    for key, value in list(original.items()):
        if isinstance(value, list):
            original[key] = json.dumps(value, ensure_ascii=False)
    before = deepcopy(original)
    def script(registry, settings, outputs, prompt):
        parsed = registry[name]._normalize_and_validate_params(json.dumps(original), strict_json=True)
        outputs.append(registry[name](**parsed))
        yield TextAction()
    result, registry, settings, _, outputs = run(monkeypatch, profile, {
        name: lambda **args: seen.append(deepcopy(args)) or {"saved": True}}, script)
    assert result["status"] == "completed"
    assert settings[0].tool_identifier_list == [name]
    assert set(registry) == {name}
    assert original == before
    expected = arguments(name)
    if name == "review_candidates":
        expected["issue_reviews"] = None
    if name == "submit_review":
        expected.update(merges=None, patches=None, exclusions=None)
    assert seen == [expected]
    assert outputs[0]["ok"] is True
    assert result["argument_normalizations"][0]["native_validation_passed"] is True
    for method in ("get_original_tool_info", "get_tool_info"):
        assert getattr(type(registry[name]), method)().parameters.to_json_schema() == adapter._normalization_schema_for_tool(name)


def test_old_handler_sets_do_not_advertise_new_tools(monkeypatch):
    def script(registry, settings, outputs, prompt):
        registry["read_entries"]()
        yield TextAction()
    result, registry, settings, *_ = run(monkeypatch, "section", {"read_entries": lambda **kw: {}}, script)
    assert result["status"] == "completed"
    assert set(registry) == {"read_entries"}
    assert settings[0].tool_identifier_list == ["read_entries"]


@pytest.mark.parametrize("name,profile", [("submit_section", "reading"), ("review_candidates", "section"),
                                         ("submit_section", "reviewer"), ("review_candidates", "reviewer"),
                                         ("submit_review", "section"), ("submit_review", "reviewer")])
def test_wrong_profile_rejects_before_sdk_load(monkeypatch, name, profile):
    monkeypatch.setattr(adapter, "_load_tongagents", lambda: pytest.fail("SDK must not be loaded"))
    result = adapter.run_react("task", {name: lambda **kw: pytest.fail("No handler")}, {
        "profile": profile, "url": "http://127.0.0.1:8000/v1", "model": "synthetic"})
    assert result["status"] == "error"


@pytest.mark.parametrize("name,field,bad", [
    ("submit_section", "entries", "[{},]"), ("submit_section", "unresolved", "not JSON"),
    ("submit_section", "complete", "false"), ("submit_section", "complete", 1),
    ("review_candidates", "operations", "[{},]"), ("review_candidates", "task_reviews", "{}"),
    ("review_candidates", "issue_reviews", "[1]"),
    ("submit_review", "selections", "[\"candidate-one\","),
    ("submit_review", "merges", "{}"), ("submit_review", "reviews", "[1]"),
])
def test_bad_container_or_scalar_never_passes_native_validation(name, field, bad):
    value = arguments(name)
    value[field] = bad
    with pytest.raises((ValidationError, ValueError)):
        installed(name)._normalize_and_validate_params(value)
    details = adapter._argument_schema_diagnostics(value, adapter._normalization_schema_for_tool(name))
    assert any(issue["path"].startswith(field) for issue in details["issues"])
    assert details["examples_are_shapes_not_defaults"] is True


@pytest.mark.parametrize("name,path", [("submit_section", "entries[0].status"),
                                       ("review_candidates", "operations[0].candidate_id")])
def test_missing_business_field_is_not_inferred_and_has_precise_repair(name, path):
    value = arguments(name)
    if name == "submit_section":
        del value["entries"][0]["status"]
    else:
        del value["operations"][0]["candidate_id"]
    with pytest.raises(ValidationError):
        installed(name)._normalize_and_validate_params(value)
    details = adapter._argument_schema_diagnostics(value, adapter._normalization_schema_for_tool(name))
    assert any(issue["path"] == path and issue["actual_type"] == "missing" for issue in details["issues"])


def test_review_omitted_optional_field_has_exact_native_dispatch_identity(monkeypatch):
    value = arguments("review_candidates")
    seen = []
    def script(registry, settings, outputs, prompt):
        session = adapter._ACTIVE_SESSION
        session["native_dispatch"] = {"generation_call_id": "synthetic-generation", "native_tool_call_id": "native-review",
                                      "tool_name": "review_candidates", "arguments": json.dumps(value)}
        try:
            outputs.append(registry["review_candidates"](**value))
        finally:
            session.pop("native_dispatch")
        yield TextAction()
    def handler(**kw):
        seen.append(adapter.current_tool_call_identity())
        assert kw["issue_reviews"] is None
        return {"saved": True}
    result, *_ = run(monkeypatch, "reading", {"review_candidates": handler}, script)
    assert result["tool_calls"][0]["status"] == "success"
    assert seen[0]["native_tool_call_id"] == "native-review"


def test_submit_review_uses_its_actual_identity_and_omission_is_not_a_decision(monkeypatch):
    value = arguments("submit_review")
    seen = []
    def script(registry, settings, outputs, prompt):
        session = adapter._ACTIVE_SESSION
        session["native_dispatch"] = {"generation_call_id": "synthetic-generation",
            "native_tool_call_id": "native-submit-review", "tool_name": "submit_review",
            "arguments": json.dumps(value)}
        try:
            outputs.append(registry["submit_review"](**value))
        finally:
            session.pop("native_dispatch")
        yield TextAction()
    def handler(**kw):
        seen.append(adapter.current_tool_call_identity())
        assert kw["patches"] is None and kw["exclusions"] is None and kw["merges"] is None
        assert kw["reviews"] == value["reviews"]
        return {"saved": True}
    result, registry, *_ = run(monkeypatch, "reading", {"submit_review": handler}, script)
    assert result["tool_calls"][0]["status"] == "success"
    assert set(registry) == {"submit_review"}
    assert seen[0]["tool_name"] == "submit_review"
    assert seen[0]["native_tool_call_id"] == "native-submit-review"


def terminal_boundary(*, complete=True, status="handed_off", string_arrays=False):
    args = {"entries": [], "unresolved": [], "complete": complete}
    proposed = deepcopy(args)
    if string_arrays:
        proposed.update(entries="[]", unresolved="[]")
    calls = [call("submit_section", json.dumps(proposed), "submit-one"),
             call("read_entries", "{}", "read-two")]
    memory, receipts = Memory(calls), []
    for current in calls:
        raw = json.loads(current.arguments)
        normalized, _, _ = adapter._normalize_container_arguments(raw, adapter._normalization_schema_for_tool(current.tool_name))
        bound = inspect.signature(adapter._TOOLS[current.tool_name][0]).bind(**normalized)
        bound.apply_defaults()
        response = {"ok": True, "tool": current.tool_name, "result": {"status": status}}
        memory.add_chunk(ToolResultMessage(name=current.tool_name, content=response, tool_call_id=current.tool_call_id), "tool")
        receipts.append({"name": current.tool_name, "status": "success", "agent_call_id": "generation",
                         "args_sha256": adapter._hash(dict(bound.arguments)), "result_sha256": adapter._hash(response)})
    session = {"tool_lock": threading.RLock(), "tool_calls": receipts, "tool_budget_exceeded": False,
               "current_agent_call_id": lambda: "generation", "argument_normalizations": []}
    settings = {"terminal_handoff_state": lambda: {"origin_call_id": "generation", "content_sha": "saved-content"},
                "terminal_generation_is_settled": lambda: True}
    return SimpleNamespace(context=SimpleNamespace(memory=memory), memory=memory, calls=calls,
                           session=session, settings=settings, proposed=proposed, args=args)


def prove(boundary):
    return adapter._terminal_section_handoff(boundary.context, boundary.session, boundary.settings)


def test_completed_submission_proves_full_batch_before_termination():
    fixture = terminal_boundary()
    proof = prove(fixture)
    assert proof["terminal_handoff_proof"]["batch_tool_count"] == 2
    fixture.memory.messages.pop()
    with pytest.raises(adapter.AgentCallAssociationError, match="unpaired"):
        prove(fixture)


@pytest.mark.parametrize("complete,status", [(False, "saved"), (False, "handed_off"), (True, "saved")])
def test_partial_or_nonhandoff_submission_cannot_terminate_even_with_callback(complete, status):
    fixture = terminal_boundary(complete=complete, status=status)
    with pytest.raises(adapter.AgentCallAssociationError, match="successful completed submission"):
        prove(fixture)
    fixture.settings["terminal_handoff_state"] = lambda: None
    assert prove(fixture) is None


def test_completion_still_requires_current_state_settlement_and_exact_receipts():
    fixture = terminal_boundary()
    fixture.settings["terminal_generation_is_settled"] = lambda: False
    with pytest.raises(adapter.AgentCallAssociationError, match="settled"):
        prove(fixture)
    fixture.settings["terminal_generation_is_settled"] = lambda: True
    fixture.session["tool_calls"][1]["args_sha256"] = "different"
    with pytest.raises(adapter.AgentCallAssociationError, match="exactly proved"):
        prove(fixture)


def test_normalized_submit_terminal_needs_actual_sdk_validation_receipt():
    fixture = terminal_boundary(string_arrays=True)
    with pytest.raises(adapter.AgentCallAssociationError, match="exactly proved"):
        prove(fixture)
    schema = adapter._normalization_schema_for_tool("submit_section")
    normalized, changes, _ = adapter._normalize_container_arguments(fixture.proposed, schema)
    fixture.session["argument_normalizations"] = [{"tool_name": "submit_section", "agent_call_id": "generation",
        "original_arguments_sha256": adapter._hash(fixture.proposed), "normalized_arguments_sha256": adapter._hash(normalized),
        "schema_sha256": adapter._hash(schema), "conversions": changes, "native_validation_passed": True}]
    assert prove(fixture)["terminal_handoff_proof"]["batch_tool_count"] == 2
