"""Section profile contract tests with a synthetic SDK, never a real SDK run."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest import tong_react_adapter as module
from bridge_evidence.contest.document_entries import EntryStore, EntryStoreError, entry_schema


class TextAction:
    def model_dump(self, **kwargs):
        return {"content": "section handed off", "is_final_response": True}


class Parameters:
    def __init__(self, schema):
        self.schema = deepcopy(schema)

    def to_json_schema(self):
        return deepcopy(self.schema)


class Tool:
    parameters = Parameters({"type": "object"})

    def _normalize_and_validate_params(self, params, *, strict_json=False):
        from jsonschema import Draft202012Validator
        Draft202012Validator(type(self).parameters.to_json_schema()).validate(params)
        return deepcopy(params)

    @classmethod
    def get_original_tool_info(cls):
        return SimpleNamespace(parameters=cls.parameters)

    @classmethod
    def get_tool_info(cls):
        return SimpleNamespace(parameters=cls.parameters)


def fake_sdk(script):
    registry, settings, contexts, outputs = {}, [], [], []

    def decorator(**definition):
        def decorate(function):
            name = definition["name"]
            if name in registry:
                raise AssertionError("Do not register the same cached native tool twice")
            registered_class = type(name + "Tool", (Tool,), {"__call__": staticmethod(function)})
            instance = registered_class()
            registry[name] = instance
            return instance
        return decorate

    class ReactAgent:
        def __init__(self, *, dep_context, agent_settings, tools):
            contexts.append(dep_context)
            settings.append(agent_settings)
            self.settings = agent_settings

        def step(self, prompt):
            yield from script(registry, self.settings, outputs, prompt)

    sdk = {"ModelConfig": SimpleNamespace, "ModelProvider": SimpleNamespace(OPENAI_COMPATIBLE="local"),
           "ReactAgent": ReactAgent, "ReactAgentSetting": SimpleNamespace,
           "LLMRunContext": SimpleNamespace, "tool": decorator, "version": "synthetic-section-sdk"}
    return sdk, registry, settings, contexts, outputs


def basic(**changes):
    return {"id": "bridge-name", "section": "基本信息", "field": "桥梁名称", "value": "合成桥",
            "source_refs": [{"evidence_id": "E1"}], "status": "explicit", **changes}


def row(section="病害明细", **changes):
    values = {"病害部位": "左幅1号梁", "病害描述": "原文明示裂缝及原因"} if section == "病害明细" else {"建议内容": "原文明示维修建议"}
    return {"id": "row-one", "section": section, "row": values,
            "source_refs": [{"evidence_id": "E1", "start": 0, "end": 20}], "status": "explicit", **changes}


class SectionAgentAdapterTests(unittest.TestCase):
    config = {"profile": "section", "url": "http://127.0.0.1:8000/v1", "model": "own-model"}

    def setUp(self):
        self.assertIsNone(module._ACTIVE_SESSION)
        module._REGISTRATIONS.clear()
        module._REGISTERED_TOOL_CLASSES.clear()
        self.guards = [patch.object(socket, "create_connection", side_effect=AssertionError("No network")),
                       patch.object(socket.socket, "connect", side_effect=AssertionError("No network")),
                       patch.object(subprocess, "Popen", side_effect=AssertionError("No subprocess"))]
        for guard in self.guards:
            guard.start()
            self.addCleanup(guard.stop)

    def invoke(self, script, handlers=None, **config):
        sdk, registry, settings, contexts, outputs = fake_sdk(script)
        with patch.object(module, "_load_tongagents", return_value=sdk), \
                patch.object(module, "_load_tool_schema_api", return_value=(Tool, Parameters)):
            result = module.run_react("Read your assigned source and hand off", handlers or {
                "complete_section": lambda summary, unresolved: {"handoff": True, "unresolved": unresolved}},
                {**self.config, **config})
        return result, registry, settings, contexts, outputs

    def test_exact_section_inventory_and_handoff_are_not_document_finish(self):
        expected = {"read_outline", "read_source", "read_section", "read_table_rows", "search_in_document",
                    "get_neighbors", "save_entries", "read_entries", "complete_section", "handoff_section",
                    "submit_section", "save_basic_info", "save_defects", "save_recommendations"}
        self.assertEqual(module._SECTION_TOOLS - module._INTERNAL_RANGE_TOOLS, expected)
        self.assertEqual(module._SECTION_ENTRY_TOOLS & module._READING_TOOLS,
                         {"save_defects", "save_recommendations"})
        self.assertFalse(module._SECTION_ENTRY_TOOLS & module._REVIEW_TOOLS)

        def script(registry, settings, outputs, prompt):
            outputs.append(registry["complete_section"]("已处理所分配范围", ["历史对比依据不足"]))
            yield TextAction()

        result, registry, settings, _, outputs = self.invoke(script, {name: (lambda **args: {"saved": True}) for name in expected})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(set(settings[0].tool_identifier_list), expected)
        self.assertNotIn("finish", registry)
        self.assertNotIn("review_draft", registry)
        self.assertEqual(result["profile"], "section")
        self.assertEqual([entry["name"] for entry in result["tool_calls"]], ["complete_section"])
        self.assertNotIn("self_check", outputs[0])
        self.assertTrue(result["session_boundary"]["tool_callbacks_drained"])
        self.assertTrue(result["session_boundary"]["session_lock_released"])

    def test_profile_boundaries_reject_before_sdk_import(self):
        cases = [("section", name) for name in module._READING_TOOLS - module._SECTION_TOOLS]
        cases += [(profile, name) for profile, allowed in (("reading", module._READING_TOOLS),
                                                          ("reviewer", module._REVIEW_TOOLS))
                  for name in module._SECTION_ENTRY_TOOLS - allowed]
        cases += [("other", "read_outline")]
        for profile, name in cases:
            with self.subTest(profile=profile, name=name), patch.object(module, "_load_tongagents") as loader:
                result = module.run_react("task", {name: lambda **args: {}}, {**self.config, "profile": profile})
                self.assertEqual(result["status"], "error")
                loader.assert_not_called()

    def test_shared_entry_schema_is_exposed_by_both_native_metadata_paths(self):
        def script(registry, settings, outputs, prompt):
            outputs.append(registry["save_entries"]([basic(), row(), row("维修建议", id="advice")]))
            yield TextAction()

        observed = []
        result, registry, *_ = self.invoke(script, {"save_entries": lambda entries: observed.append(deepcopy(entries)) or {"saved_ids": [entry["id"] for entry in entries]}})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(observed, [[basic(), row(), row("维修建议", id="advice")]])
        for getter in ("get_original_tool_info", "get_tool_info"):
            schema = getattr(type(registry["save_entries"]), getter)().parameters.to_json_schema()
            self.assertEqual(schema["properties"]["entries"]["items"], entry_schema())
            self.assertFalse(schema["additionalProperties"])
        self.assertEqual(result["tool_calls"][0]["args_sha256"], module._hash({"entries": observed[0]}))

    def test_entries_preserve_unknown_and_explicit_inference_without_defaults(self):
        entries = [basic(value=None, status="uncertain"), row(status="inferred"), row("维修建议", id="c", status="conflict")]
        seen = []
        def script(registry, settings, outputs, prompt):
            registry["save_entries"](entries)
            yield TextAction()
        result, *_ = self.invoke(script, {"save_entries": lambda entries: seen.append(deepcopy(entries)) or {"saved": True}})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(seen, [entries])
        self.assertNotIn("发展程度", seen[0][1]["row"])

    def test_source_dispatches_expose_exact_schemas_and_original_handler_arguments(self):
        seen = []
        handlers = {name: (lambda _name=name, **args: seen.append((_name, args)) or {"text": "合成原文"})
                    for name in module._SECTION_SOURCE_TOOLS}
        def script(registry, settings, outputs, prompt):
            registry["read_outline"]()
            registry["read_source"]("E1")
            registry["read_section"]("S1", max_units=3)
            registry["read_table_rows"]("T1", max_rows=2)
            registry["search_in_document"](["裂缝"])
            registry["get_neighbors"]("E1")
            yield TextAction()
        result, registry, *_ = self.invoke(script, handlers)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(seen, [("read_outline", {"limit": 20, "restart": False}),
                              ("read_source", {"evidence_id": "E1", "restart": False}),
                              ("read_section", {"section_id": "S1", "restart": False, "max_units": 3}),
                              ("read_table_rows", {"table_id": "T1", "restart": False, "max_rows": 2}),
                              ("search_in_document", {"query": ["裂缝"], "scope": "default", "section_id": None,
                                                       "max_matches": 8, "restart": False}),
                              ("get_neighbors", {"evidence_id": "E1"})])
        self.assertFalse(module._INTERNAL_RANGE_TOOLS & registry.keys())
        for name in registry:
            self.assertEqual(type(registry[name]).get_tool_info().parameters.to_json_schema(), module._section_tool_schema(name))

    def test_actual_entry_store_requires_delivered_reads_then_atomic_persistence(self):
        text = "合成桥裂缝需要修补"
        with tempfile.TemporaryDirectory() as directory:
            store = EntryStore({"units": [{"id": "E1", "text": text}]}, "task-one", directory)
            def save(entries):
                try:
                    return store.save_entries(entries)
                except EntryStoreError as exc:
                    return exc.as_tool_error()
            def observer(name, args, response):
                if name == "read_source" and response["ok"]:
                    source = response["result"]
                    store.record_read(source["evidence_id"], source["start"], source["end"])
            def script(registry, settings, outputs, prompt):
                outputs.append(registry["save_entries"]([basic()]))
                registry["read_source"]("E1")
                outputs.append(registry["save_entries"]([basic()]))
                outputs.append(registry["read_entries"]())
                registry["complete_section"]("已保存实际阅读的条目", [])
                yield TextAction()
            result, _, _, _, outputs = self.invoke(script, {
                "read_source": lambda evidence_id, restart: {"evidence_id": evidence_id, "start": 0, "end": len(text), "text": text},
                "save_entries": save, "read_entries": lambda restart, limit: store.read_entries(limit=limit),
                "complete_section": lambda summary, unresolved: {"handoff": True, "document_complete": False}},
                on_tool_result=observer)
            self.assertEqual(result["status"], "completed")
            self.assertFalse(outputs[0]["ok"])
            self.assertFalse(outputs[0]["result"]["error"]["details"]["applied"])
            self.assertEqual(outputs[0]["result"]["error"]["code"], "SOURCE_UNREAD")
            self.assertTrue(outputs[1]["ok"])
            self.assertEqual(outputs[2]["result"]["entries"][0]["source_refs"], [{"evidence_id": "E1", "start": 0, "end": len(text)}])
            self.assertEqual(store.snapshot()["entries"]["bridge-name"]["value"], "合成桥")
            self.assertEqual(result["tool_error_count"], 1)

    def test_invalid_entries_are_not_executed_or_saved(self):
        invalid = [[basic(extra="not allowed")], [basic(status="default")], [basic(section="other")],
                   [basic(field="全桥技术状况评分")], [basic(source_refs=[])],
                   [basic(source_refs=[{"start": 0, "end": 1}])],
                   [basic(source_refs=[{"evidence_id": "E1", "start": 0}])],
                   [basic(source_refs=[{"evidence_id": "E1", "end": 1}])],
                   [basic(source_refs=[{"evidence_id": "E1", "start": 2, "end": 1}])],
                   [basic(source_refs=[{"evidence_id": "E1", "start": True, "end": 1}])],
                   [basic(source_refs=[{"evidence_id": "E1", "page": 1}])],
                   [basic(source_refs=[{"evidence_id": " "}])], [basic(id=" ")],
                   [basic(value=float("nan"))], [row(row={})], [row(row={"extra": "bad"})],
                   [row(row={"建议内容": "wrong section"})], [basic(value=("not JSON",))]]
        for entries in invalid:
            with self.subTest(entries=repr(entries)[:120]):
                saved = []
                def script(registry, settings, outputs, prompt):
                    outputs.append(registry["save_entries"](entries))
                    yield TextAction()
                result, _, _, _, outputs = self.invoke(script, {"save_entries": lambda entries: saved.append(entries) or {"saved": True}})
                self.assertEqual(saved, [])
                self.assertFalse(outputs[0]["ok"])
                self.assertEqual(result["tool_calls"][0]["status"], "failed")

    def test_empty_entries_are_the_store_noop_and_not_completion(self):
        seen = []
        def script(registry, settings, outputs, prompt):
            registry["save_entries"]([])
            yield TextAction()
        result, *_ = self.invoke(script, {"save_entries": lambda entries: seen.append(entries) or {"changed": False, "delivery_complete": False}})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(seen, [[]])
        self.assertNotIn("complete_section", [entry["name"] for entry in result["tool_calls"]])

    def test_section_arguments_never_ignore_extra_fields(self):
        cases = {"read_outline": {"extra": "x"}, "read_source": {"evidence_id": "E1", "extra": "x"},
                 "read_section": {"section_id": "S1", "cursor": 0, "max_units": 2, "extra": "x"},
                 "read_table_rows": {"table_id": "T1", "start": 0, "end": 1, "extra": "x"},
                 "search_in_document": {"query": "裂缝", "extra": "x"},
                 "get_neighbors": {"evidence_id": "E1", "extra": "x"},
                 "save_entries": {"entries": [basic()], "extra": "x"},
                 "read_entries": {"cursor": 0, "limit": 8, "extra": "x"},
                 "complete_section": {"summary": "done", "unresolved": [], "extra": "x"}}
        for name, args in cases.items():
            with self.subTest(tool=name):
                seen = []
                def script(registry, settings, outputs, prompt):
                    outputs.append(module._invoke(name, args))
                    yield TextAction()
                result, *_ = self.invoke(script, {name: lambda **kwargs: seen.append(kwargs) or {"saved": True}})
                self.assertEqual(seen, [])
                self.assertEqual(result["tool_calls"][0]["status"], "failed")

    def test_read_entries_defaults_match_dispatcher_and_no_arbitrary_limit_cap(self):
        calls = []
        def script(registry, settings, outputs, prompt):
            registry["read_entries"]()
            registry["read_entries"](restart=True, limit=100)
            yield TextAction()
        result, *_ = self.invoke(script, {"read_entries": lambda restart, limit: calls.append((restart, limit)) or {"entries": [], "next_cursor": None}})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(calls, [(False, 8), (True, 100)])
        self.assertEqual(result["tool_calls"][0]["args_sha256"], module._hash({"restart": False, "limit": 8}))

    def test_invalid_pagination_or_empty_handoff_cannot_call_handlers(self):
        cases = [("read_entries", {"cursor": True, "limit": 8}), ("read_entries", {"cursor": -1, "limit": 8}),
                 ("read_entries", {"cursor": 0, "limit": 0}), ("read_entries", {"cursor": 0, "limit": 1.0}),
                 ("complete_section", {"summary": "", "unresolved": []}),
                 ("complete_section", {"summary": " ", "unresolved": []}),
                 ("complete_section", {"summary": "done", "unresolved": [" "]}),
                 ("complete_section", {"summary": "done"})]
        for name, args in cases:
            with self.subTest(tool=name, args=args):
                calls = []
                def script(registry, settings, outputs, prompt):
                    outputs.append(module._invoke(name, args))
                    yield TextAction()
                result, *_ = self.invoke(script, {name: lambda **kwargs: calls.append(kwargs) or {"saved": True}})
                self.assertEqual(calls, [])
                self.assertEqual(result["tool_calls"][0]["status"], "failed")

    def test_section_memory_factory_is_section_only_and_not_compression(self):
        factory = lambda: object()
        for config in ({"profile": "reading", "initial_memory_factory": factory},
                       {"profile": "reviewer", "initial_memory_factory": factory},
                       {"initial_memory_factory": None},
                       {"initial_memory_factory": factory, "native_compression": object()}):
            with self.subTest(config=config), patch.object(module, "_load_tongagents") as loader:
                result = module.run_react("task", {"complete_section": lambda **kwargs: {}}, {**self.config, **config})
                self.assertEqual(result["status"], "error")
                loader.assert_not_called()

    def test_private_memory_factory_called_once_and_bound_to_exact_context(self):
        memory, calls = object(), []
        def factory():
            calls.append("factory")
            return memory
        def script(registry, settings, outputs, prompt):
            registry["complete_section"]("done", [])
            yield TextAction()
        result, _, _, contexts, _ = self.invoke(script, initial_memory_factory=factory)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(calls, ["factory"])
        self.assertIs(contexts[0].memory, memory)
        result, _, _, contexts, _ = self.invoke(script)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(hasattr(contexts[0], "memory"))

    def test_failed_private_memory_factory_does_not_construct_or_run_agent(self):
        for factory in (lambda: None, lambda: (_ for _ in ()).throw(ValueError("private failure"))):
            with self.subTest(factory=factory):
                result, _, settings, contexts, _ = self.invoke(lambda *args: iter([TextAction()]), initial_memory_factory=factory)
                self.assertEqual(result["status"], "error")
                self.assertEqual(contexts, [])
                self.assertEqual(settings, [])

    def test_dynamic_allocation_and_trusted_call_association_are_preserved(self):
        timeline = []
        callback = lambda payload: payload
        @contextmanager
        def allocator(url, prepare):
            self.assertIs(prepare, callback)
            timeline.append("enter")
            yield
            timeline.append("exit")
        def script(registry, settings, outputs, prompt):
            self.assertIsNone(settings.llm_config.max_tokens)
            registry["complete_section"]("done", [])
            yield TextAction()
        with patch("bridge_evidence.contest.document_context_budget.allocate_sdk_wire_requests", allocator):
            result, *_ = self.invoke(script, dynamic_output_budget=True, prepare_generation_request=callback,
                                     current_agent_call_id=lambda: "trusted-actual-call")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(timeline, ["enter", "exit"])
        self.assertEqual(result["tool_calls"][0]["agent_call_id"], "trusted-actual-call")

    def test_section_final_without_any_tool_is_a_failure(self):
        result, *_ = self.invoke(lambda *args: iter([TextAction()]))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "no_tool_invocation")

    def test_cached_section_tool_cannot_invoke_a_later_reading_handler(self):
        handoffs = []
        def script(registry, settings, outputs, prompt):
            outputs.append(registry["complete_section"]("done", []))
            if prompt == "reading":
                registry["read_source"]("E1")
            yield TextAction()
        sdk, _, settings, _, outputs = fake_sdk(script)
        with patch.object(module, "_load_tongagents", return_value=sdk), \
                patch.object(module, "_load_tool_schema_api", return_value=(Tool, Parameters)):
            first = module.run_react("section", {"complete_section": lambda **kwargs: handoffs.append("section") or {"handoff": True}}, self.config)
            second = module.run_react("reading", {"read_source": lambda **kwargs: {"text": "original"}}, {**self.config, "profile": "reading"})
        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "completed")
        self.assertEqual(handoffs, ["section"])
        self.assertTrue(outputs[0]["ok"])
        self.assertFalse(outputs[1]["ok"])
        self.assertEqual(settings[1].tool_identifier_list, ["read_source"])

    def test_sdk_worker_callback_keeps_separate_lock_and_receipts(self):
        def script(registry, settings, outputs, prompt):
            errors = []
            def call():
                try:
                    outputs.append(registry["complete_section"]("done", []))
                except Exception as exc:
                    errors.append(type(exc).__name__)
            thread = threading.Thread(target=call)
            thread.start()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            yield TextAction()
        result, *_ = self.invoke(script)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["tool_calls"]), 1)


if __name__ == "__main__":
    unittest.main()
