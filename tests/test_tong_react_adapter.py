"""Fake-SDK contract tests; these are NOT evidence of real TongAgents use."""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from types import SimpleNamespace
from copy import deepcopy
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest import tong_react_adapter as module


class TextAction:
    def __init__(self, content="done", final=True, **extra):
        self.content = content
        self.is_final_response = final
        self.extra = extra

    def model_dump(self, **kwargs):
        return {"content": self.content, "is_final_response": self.is_final_response, **self.extra}


class SchemaParameters:
    def __init__(self, schema):
        self.schema = deepcopy(schema)

    def to_json_schema(self):
        return deepcopy(self.schema)


class SchemaTool:
    parameters = SchemaParameters({"type": "object"})

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


def fake_sdk(script, *, schema_api=False):
    """A persistent registry mimics cached SDK tools across multiple sessions."""
    registry, settings, results, registrations = {}, [], [], []

    def tool(**definition):
        def decorate(function):
            name = definition["name"]
            if name in registry:
                raise AssertionError("adapter must not redecorate a cached tool")
            # Other suites intentionally reuse the legacy decorator to test
            # unavailable-schema behavior. Opt in here to the public metadata
            # contract required by managed source/delivery tools.
            instance = (type(name + "SyntheticTool", (SchemaTool,), {"__call__": staticmethod(function)})()
                        if schema_api else function)
            registry[name] = instance
            registrations.append(name)
            return instance
        return decorate

    class ReactAgent:
        def __init__(self, *, dep_context, agent_settings, tools):
            settings.append(agent_settings)
            self.settings = agent_settings

        def step(self, prompt):
            yield from script(registry, self.settings, results, prompt)

    sdk = {"ModelConfig": SimpleNamespace, "ModelProvider": SimpleNamespace(OPENAI_COMPATIBLE="local"),
           "ReactAgent": ReactAgent, "ReactAgentSetting": SimpleNamespace,
           "LLMRunContext": SimpleNamespace, "tool": tool, "version": "fake-test-sdk"}
    return sdk, registry, settings, results, registrations


class TongReactAdapterTests(unittest.TestCase):
    config = {"url": "http://127.0.0.1:8000/v1", "model": "own-model"}

    def setUp(self):
        self.assertIsNone(module._ACTIVE_SESSION)
        module._REGISTRATIONS.clear()
        module._REGISTERED_TOOL_CLASSES.clear()
        loader = patch.object(module, "_load_tool_schema_api", return_value=(SchemaTool, SchemaParameters))
        loader.start()
        self.addCleanup(loader.stop)

    def invoke(self, script, handlers=None, **config):
        sdk, registry, settings, outputs, registrations = fake_sdk(script, schema_api=True)
        with patch.object(module, "_load_tongagents", return_value=sdk):
            result = module.run_react("perform authorized task", handlers or {"read_evidence": lambda evidence_ids: {"text": "source text"}},
                                      {**self.config, **config})
        return result, registry, settings, outputs, registrations

    def test_real_dispatch_and_explicit_final_are_required(self):
        def script(registry, settings, outputs, prompt):
            outputs.append(registry["read_evidence"](["E1"]))
            yield TextAction("complete")
        result, registry, settings, outputs, registered = self.invoke(script)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["response_text"], "complete")
        self.assertEqual(result["sdk_version"], "fake-test-sdk")
        self.assertEqual(result["tool_calls"][0]["args"], {"evidence_ids": ["E1"]})
        self.assertEqual(result["tool_calls"][0]["status"], "success")
        self.assertEqual(len(result["tool_calls"][0]["result_sha256"]), 64)
        self.assertTrue(outputs[0]["ok"])
        self.assertEqual(registered, ["read_evidence"])
        self.assertEqual(settings[0].tool_identifier_list, ["read_evidence"])
        self.assertFalse(hasattr(settings[0], "tool_choice"))
        self.assertFalse(settings[0].llm_config.extra_body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(settings[0].llm_config.extra_body["temperature"], 0)
        self.assertIsNone(result["model_call_count"])
        self.assertIsNone(module._ACTIVE_SESSION)

    def test_cached_tools_do_not_reuse_previous_document_handlers(self):
        def script(registry, settings, outputs, prompt):
            outputs.append(registry["read_evidence"](["E1"]))
            yield TextAction()
        sdk, _, settings, outputs, registered = fake_sdk(script, schema_api=True)
        with patch.object(module, "_load_tongagents", return_value=sdk):
            first = module.run_react("first", {"read_evidence": lambda evidence_ids: "first-document"}, self.config)
            second = module.run_react("second", {"read_evidence": lambda evidence_ids: "second-document"}, self.config)
        self.assertEqual([first["status"], second["status"]], ["completed", "completed"])
        self.assertEqual([item["result"] for item in outputs], ["first-document", "second-document"])
        self.assertEqual(len(registered), 1)
        self.assertEqual(len(settings), 2)

    def test_whitelist_rejects_cached_tool_from_another_session(self):
        calls = []
        def script(registry, settings, outputs, prompt):
            if prompt == "first":
                registry["read_evidence"](["E1"])
            else:
                outputs.append(registry["read_evidence"](["E1"]))
                registry["finish"]()
            yield TextAction()
        sdk, _, _, outputs, _ = fake_sdk(script, schema_api=True)
        with patch.object(module, "_load_tongagents", return_value=sdk):
            module.run_react("first", {"read_evidence": lambda evidence_ids: calls.append("read")}, self.config)
            result = module.run_react("second", {"finish": lambda: {"ready": False}}, self.config)
        self.assertEqual(calls, ["read"])
        self.assertFalse(outputs[0]["ok"])
        self.assertEqual(result["tool_calls"][0]["status"], "failed")
        self.assertEqual(result["status"], "completed")  # Business readiness is caller-owned.

    def test_tools_return_structured_errors_and_agent_can_recover(self):
        def handler(evidence_ids):
            if evidence_ids == ["bad"]:
                raise ValueError("private source text that must never enter receipts")
            return {"text": "valid private source text"}
        def script(registry, settings, outputs, prompt):
            outputs.append(registry["read_evidence"](["bad"]))
            outputs.append(registry["read_evidence"](["good"]))
            yield TextAction()
        result, _, _, outputs, _ = self.invoke(script, {"read_evidence": handler})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tool_error_count"], 1)
        self.assertEqual([row["status"] for row in result["tool_calls"]], ["failed", "success"])
        self.assertFalse(outputs[0]["ok"])
        self.assertNotIn("private source text", str(result))

    def test_trusted_native_identity_matches_defaults_and_cannot_be_forged(self):
        identities = []

        def handler(**_args):
            identity = module.current_tool_call_identity()
            identities.append(deepcopy(identity))
            if identity:
                identity["native_tool_call_id"] = "mutated-copy"
                self.assertNotEqual(module.current_tool_call_identity()["native_tool_call_id"], "mutated-copy")
            return {"ok": True, "text": "synthetic source"}

        def script(registry, settings, outputs, prompt):
            session = module._ACTIVE_SESSION
            session["native_dispatch"] = {"generation_call_id": "generation-one",
                "native_tool_call_id": "native-one", "tool_name": "read_source",
                "arguments": '{"evidence_id":"E1"}'}
            outputs.append(registry["read_source"]("E1"))
            self.assertIsNone(module.current_tool_call_identity())
            session["native_dispatch"]["arguments"] = '{"evidence_id":"different-object"}'
            outputs.append(registry["read_source"]("E1"))
            session["native_dispatch"]["arguments"] = '{"evidence_id":"E1"}'
            session["native_dispatch"]["generation_call_id"] = "stale-generation"
            outputs.append(registry["read_source"]("E1"))
            session.pop("native_dispatch")
            outputs.append(registry["read_source"]("E1"))
            yield TextAction()

        result, *_ = self.invoke(script, {"read_source": handler},
            current_agent_call_id=lambda: "generation-one", run_id="run-one")
        self.assertEqual(len(identities), 2)
        self.assertEqual(identities[0]["run_id"], "run-one")
        self.assertEqual(identities[0]["native_tool_call_id"], "native-one")
        self.assertEqual(identities[0]["args_sha256"], module._hash({"evidence_id": "E1", "restart": False}))
        self.assertIsNone(identities[1], "Legacy direct calls must not invent a native ID")
        self.assertEqual([r["status"] for r in result["tool_calls"]], ["success", "failed", "failed", "success"])
        self.assertIsNone(module.current_tool_call_identity())

    def test_only_managed_navigation_relaxes_sdk_same_argument_gate(self):
        for name, args, expected in (("read_source", ("E1",), 17), ("finish", (), 1)):
            with self.subTest(name=name):
                def script(registry, settings, outputs, prompt):
                    registry[name](*args)
                    yield TextAction()
                result, _, settings, *_ = self.invoke(script, {name: lambda **_args: {"ok": True}}, max_calls=17)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(settings[0].same_tool_retry_max, expected)
                self.assertEqual(settings[0].max_tool_retry, 1)
                self.assertEqual(settings[0].max_call_cnt, 17)

    def test_explicit_handler_error_is_not_recorded_as_success(self):
        def script(registry, settings, outputs, prompt):
            outputs.append(registry["finish"]())
            yield TextAction()
        result, _, _, outputs, _ = self.invoke(script, {"finish": lambda: {"ok": False, "error": "missing evidence"}})
        self.assertEqual(result["tool_calls"][0]["status"], "failed")
        self.assertFalse(outputs[0]["ok"])

    def test_empty_missing_or_nonfinal_text_and_no_tool_are_failures(self):
        for final in (None, TextAction(""), TextAction("partial", final=False)):
            with self.subTest(final=final):
                def script(registry, settings, outputs, prompt):
                    registry["read_evidence"](["E1"])
                    if final is not None:
                        yield final
                result, *_ = self.invoke(script)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["failure_reason"], "missing_or_empty_native_final_response")
        result, *_ = self.invoke(lambda *args: iter([TextAction("guessed without evidence")]))
        self.assertEqual(result["failure_reason"], "no_tool_invocation")

    def test_tool_budget_exhaustion_prevents_unbounded_handler_calls(self):
        calls = []
        def script(registry, settings, outputs, prompt):
            for _ in range(3):
                outputs.append(registry["finish"]())
            yield TextAction()
        result, *_ = self.invoke(script, {"finish": lambda: calls.append("called")}, max_calls=2)
        self.assertEqual(calls, ["called", "called"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "tool_budget_exceeded")

    def test_sdk_failure_preserves_earlier_receipts_and_releases_session(self):
        def script(registry, settings, outputs, prompt):
            registry["read_evidence"](["E1"])
            yield TextAction("partial", final=False)
            raise RuntimeError("private API key and prompt")
        result, *_ = self.invoke(script)
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertNotIn("private API key", str(result))
        self.assertIsNone(module._ACTIVE_SESSION)

    def test_missing_sdk_fails_without_executing_any_handler(self):
        calls = []
        with patch.object(module, "_load_tongagents", side_effect=ImportError("no sdk")):
            result = module.run_react("task", {"finish": lambda: calls.append(1)}, self.config)
        self.assertEqual(result["status"], "error")
        self.assertIn("no fallback", result["error"]["message"])
        self.assertEqual(calls, [])

    def test_nonlocal_urls_and_unknown_handlers_rejected_before_sdk_load(self):
        for url in ("https://api.example.com/v1", "http://127.0.0.1.example.com/v1", "http://user:pass@localhost/v1",
                    "http://localhost/v1?api_key=secret", "ftp://127.0.0.1", "https://127.0.0.1/v1",
                    "http://localhost/v2", "http://localhost/v1/chat/completions", "http://localhost/v1?",
                    "http://localhost/v1#", " http://localhost/v1", "http://local\nhost/v1"):
            with self.subTest(url=url), patch.object(module, "_load_tongagents") as loader:
                result = module.run_react("task", {"finish": lambda: None}, {**self.config, "url": url})
                self.assertEqual(result["status"], "error")
                loader.assert_not_called()
        with patch.object(module, "_load_tongagents") as loader:
            result = module.run_react("task", {"shell": lambda: None}, self.config)
            self.assertEqual(result["status"], "error")
            loader.assert_not_called()

    def test_event_and_result_receipts_do_not_embed_report_text(self):
        secret = "a private report paragraph" * 100
        def script(registry, settings, outputs, prompt):
            registry["read_evidence"](["E1"])
            yield TextAction("done", result={"tool_response": secret})
        result, *_ = self.invoke(script, {"read_evidence": lambda evidence_ids: {"text": secret}})
        self.assertNotIn(secret[:40], str(result))
        self.assertIn("sha256", str(result["events"]))
        self.assertEqual(result["response_text"], "done")

    def test_split_tool_and_argument_validation(self):
        calls = []
        def script(registry, settings, outputs, prompt):
            outputs.append(registry["split_task"]("T1"))
            outputs.append(registry["read_evidence"]("E1"))
            yield TextAction()
        result, _, _, outputs, registered = self.invoke(script, {
            "split_task": lambda task_id: calls.append(task_id) or {"children": ["T1a", "T1b"]},
            "read_evidence": lambda evidence_ids: self.fail("invalid args must not reach handler"),
        })
        self.assertEqual(calls, ["T1"])
        self.assertTrue(outputs[0]["ok"])
        self.assertFalse(outputs[1]["ok"])
        self.assertEqual(set(registered), {"split_task", "read_evidence"})
        self.assertEqual(result["tool_error_count"], 1)

    def test_length_limited_native_final_fails(self):
        def script(registry, settings, outputs, prompt):
            registry["finish"]()
            yield TextAction("partial JSON", response_meta={"finish_reason": "length"})
        result, *_ = self.invoke(script, {"finish": lambda: {}})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "final_response_truncated")

    def test_tool_arguments_are_recorded_before_handler_mutation(self):
        def handler(evidence_ids):
            evidence_ids.append("changed")
            return {"text": "source"}
        def script(registry, settings, outputs, prompt):
            registry["read_evidence"](["E1"])
            yield TextAction()
        result, *_ = self.invoke(script, {"read_evidence": handler})
        self.assertEqual(result["tool_calls"][0]["args"], {"evidence_ids": ["E1"]})

    def test_loopback_root_and_v1_urls_are_canonicalized(self):
        def script(registry, settings, outputs, prompt):
            registry["finish"]()
            yield TextAction()
        for url, canonical in (
            ("http://localhost:8000", "http://localhost:8000/v1"),
            ("http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000/v1"),
            ("http://[::1]:8000/", "http://[::1]:8000/v1"),
        ):
            with self.subTest(url=url):
                result, _, settings, *_ = self.invoke(script, {"finish": lambda: {}}, url=url)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(settings[0].llm_config.url, canonical)

    def test_sdk_worker_thread_sees_current_session_handlers(self):
        def script(registry, settings, outputs, prompt):
            with ThreadPoolExecutor(max_workers=1) as worker:
                outputs.append(worker.submit(registry["read_evidence"], ["E1"]).result(timeout=2))
            yield TextAction()
        result, _, _, outputs, _ = self.invoke(script, {
            "read_evidence": lambda evidence_ids: {"document_id": "current-document"},
        })
        self.assertEqual(result["status"], "completed")
        self.assertEqual(outputs[0]["result"], {"document_id": "current-document"})

    def test_parallel_sdk_callbacks_cannot_race_past_tool_budget(self):
        calls = []
        def handler():
            time.sleep(0.01)
            calls.append("called")
            return {}
        def script(registry, settings, outputs, prompt):
            with ThreadPoolExecutor(max_workers=2) as worker:
                futures = [worker.submit(registry["finish"]) for _ in range(2)]
                outputs.extend(future.result(timeout=2) for future in futures)
            yield TextAction()
        result, *_ = self.invoke(script, {"finish": handler}, max_calls=1)
        self.assertEqual(calls, ["called"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "tool_budget_exceeded")

    def test_outline_and_table_pagination_are_authorized_tools(self):
        pages = []
        def read_rows(table_id, max_rows, restart):
            pages.append((table_id, max_rows, restart))
            return {"rows": ["row zero", "row one"][:max_rows]}
        def script(registry, settings, outputs, prompt):
            outputs.append(registry["read_outline"]())
            outputs.append(registry["read_table_rows"]("table-1", max_rows=1))
            yield TextAction()
        result, _, settings, outputs, registered = self.invoke(script, {
            "read_outline": lambda limit, restart: {"tables": ["table-1"]},
            "read_table_rows": read_rows,
        })
        self.assertEqual(result["status"], "completed")
        self.assertEqual(pages, [("table-1", 1, False)])
        self.assertEqual(outputs[1]["result"], {"rows": ["row zero"]})
        self.assertEqual(set(registered), {"read_outline", "read_table_rows"})
        self.assertEqual(settings[0].tool_identifier_list, ["read_outline", "read_table_rows"])
        self.assertEqual(result["tool_calls"][0]["args"], {"limit": 20, "restart": False})
        self.assertEqual(result["tool_calls"][1]["args"], {"table_id": "table-1", "max_rows": 1, "restart": False})

    def test_invalid_table_page_bounds_never_reach_handler(self):
        for start, end in ((-1, 1), (0, 0), (2, 1), (False, 1), (0, True), (0.0, 1), (0, 1.0), ("0", 1)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                module._validate_arguments("read_table_range", {"table_id": "table-1", "start": start, "end": end})

    def test_precise_range_tools_are_internal_in_every_model_profile(self):
        for profile in ("reading", "section"):
            with self.subTest(profile=profile):
                def script(registry, settings, outputs, prompt):
                    self.assertFalse(set(settings.tool_identifier_list) & module._INTERNAL_RANGE_TOOLS)
                    self.assertFalse(set(registry) & module._INTERNAL_RANGE_TOOLS)
                    outputs.append(registry["read_source"]("E1"))
                    yield TextAction()
                handlers = {"read_source": lambda **_kwargs: {"ok": True},
                    **{name: lambda **_kwargs: self.fail("precise tool exposed") for name in module._INTERNAL_RANGE_TOOLS}}
                result, _, _, _, _ = self.invoke(script, handlers, profile=profile)
                self.assertEqual(result["status"], "completed")

    def test_invalid_managed_table_options_never_reach_handler(self):
        for max_rows, restart in ((0, False), (-1, False), (True, False), (1.0, False), (1, 1), (1, "false")):
            with self.subTest(max_rows=max_rows, restart=restart):
                def script(registry, settings, outputs, prompt):
                    outputs.append(registry["read_table_rows"]("table-1", max_rows=max_rows, restart=restart))
                    yield TextAction()
                result, _, _, outputs, _ = self.invoke(script, {
                    "read_table_rows": lambda **kwargs: self.fail("invalid managed page options reached handler"),
                })
                self.assertFalse(outputs[0]["ok"])
                self.assertEqual(result["tool_calls"][0]["status"], "failed")
                self.assertEqual(result["tool_error_count"], 1)


if __name__ == "__main__":
    unittest.main()
