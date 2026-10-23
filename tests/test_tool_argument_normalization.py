"""Strict parameter representation compatibility; synthetic SDK, no model/network."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator, ValidationError
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest import tong_react_adapter as adapter
from bridge_evidence.contest import document_context_compression as native
from test_section_agent_adapter import Tool, Parameters, basic, fake_sdk, TextAction
from test_partial_native_batch_recovery import Memory, ToolResultMessage, call


class ToolArgumentNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.guards = [patch.object(socket, "create_connection", side_effect=AssertionError("No network")),
                       patch.object(socket.socket, "connect", side_effect=AssertionError("No network")),
                       patch.object(subprocess, "Popen", side_effect=AssertionError("No process")),
                       patch.object(adapter, "_load_tool_schema_api", return_value=(Tool, Parameters))]
        for guard in self.guards:
            guard.start(); self.addCleanup(guard.stop)
        adapter._REGISTRATIONS.clear(); adapter._REGISTERED_TOOL_CLASSES.clear()
        self.addCleanup(adapter._REGISTRATIONS.clear)
        self.addCleanup(adapter._REGISTERED_TOOL_CLASSES.clear)
        self.assertIsNone(adapter._ACTIVE_SESSION)

    def installed(self, name="save_entries", schema=None):
        cls = type("Synthetic" + name, (Tool,), {})
        item = cls()
        adapter._install_tool_schema(item, name, schema or adapter._section_tool_schema(name))
        return item

    def test_sdk_entry_signature_strictly_decodes_before_native_schema_without_mutating_original(self):
        original_signature = inspect.signature(Tool._normalize_and_validate_params)
        self.assertEqual(list(original_signature.parameters), ["self", "params", "strict_json"])
        item = self.installed()
        native = {"entries": json.dumps([basic(value='原文"引号"\\路径\n😀')], ensure_ascii=False)}
        before = deepcopy(native)
        parsed = item._normalize_and_validate_params(native, strict_json=True)
        self.assertEqual(native, before)
        self.assertEqual(parsed, {"entries": json.loads(before["entries"])})
        self.assertIs(type(parsed["entries"]), list)

    def test_json_array_and_object_are_not_changed_or_defaulted(self):
        for name, arguments in (("save_entries", {"entries": [basic()]}),
                                ("update_workspace", {"update": {"base_version": 8}})):
            schema = adapter._normalization_schema_for_tool(name)
            normalized, changes, issues = adapter._normalize_container_arguments(arguments, schema)
            self.assertEqual(normalized, arguments)
            self.assertIsNot(normalized, arguments)
            self.assertEqual((changes, issues), ([], []))

    def test_native_sdk_text_transport_is_strictly_decoded_before_container_fields(self):
        item = self.installed()
        arguments = {"entries": json.dumps([basic(value="PRIVATE_REPORT_VALUE")], ensure_ascii=False)}
        raw = json.dumps(arguments, ensure_ascii=False)
        session = {"handlers": {"save_entries": lambda **kwargs: None},
                   "argument_normalizations": [], "current_agent_call_id": lambda: "g"}
        with patch.object(adapter, "_ACTIVE_SESSION", session):
            parsed = item._normalize_and_validate_params(raw, strict_json=False)
        self.assertEqual(parsed, {"entries": json.loads(arguments["entries"])})
        self.assertEqual(json.loads(raw), arguments)
        audit = session["argument_normalizations"][0]
        self.assertTrue(audit["native_validation_passed"])
        self.assertEqual(audit["original_arguments_sha256"], adapter._hash(arguments))
        self.assertEqual(audit["original_arguments_text_sha256"], adapter._hash(raw))
        self.assertEqual(audit["native_parameter_representation"], "json_string")
        self.assertNotIn("PRIVATE_REPORT_VALUE", json.dumps(audit))

    def test_native_sdk_outer_transport_rejects_loose_or_nonobject_json(self):
        item = self.installed()
        invalid = ["{'entries': []}", '{"entries":[],}', '{"entries":[],"entries":[]}',
                   '{"entries":[NaN]}', '{"entries":[1e1000]}', '[]', 'null', '"{}"']
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                item._normalize_and_validate_params(raw, strict_json=False)

    def test_only_top_level_schema_containers_are_decoded_not_union_strings_or_report_values(self):
        schema = {"type": "object", "properties": {"entries": {"type": "array"},
                  "update": {"type": "object"}, "query": {"type": ["string", "array"]},
                  "text": {"type": "string"}}}
        args = {"entries": '[{"source_refs":"[]","value":"[1]"}]', "update": '{"value":"{}"}',
                "query": '["literal"]', "text": '[1]'}
        normalized, changes, issues = adapter._normalize_container_arguments(args, schema)
        self.assertEqual(normalized, {"entries": [{"source_refs": "[]", "value": "[1]"}],
                         "update": {"value": "{}"}, "query": '["literal"]', "text": '[1]'})
        self.assertEqual([item["path"] for item in changes], ["entries", "update"])
        self.assertEqual(issues, [])

    def test_single_quotes_trailing_comma_duplicate_keys_nonfinite_and_wrong_containers_rejected(self):
        item = self.installed()
        invalid = ["[{'id':'x'}]", "[{},]", '[{"id":"a","id":"b"}]', '[NaN]', '[Infinity]',
                   '[1e1000]', '{}', 'null', '"[]"', '[] trailing']
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                original = {"entries": raw}
                item._normalize_and_validate_params(original, strict_json=True)
                self.assertEqual(original["entries"], raw)

    def test_decoding_does_not_supply_required_business_fields_or_fix_nested_types(self):
        item = self.installed()
        for entry in (basic(status=None), {key: value for key, value in basic().items() if key != "status"},
                      basic(source_refs='[{"evidence_id":"E1"}]')):
            with self.subTest(entry=entry), self.assertRaises(ValidationError):
                item._normalize_and_validate_params({"entries": json.dumps([entry])}, strict_json=True)

    def test_repair_details_show_precise_missing_path_without_private_values(self):
        entry = basic(value="PRIVATE_REPORT_VALUE")
        del entry["status"]
        arguments = {"entries": json.dumps([entry])}
        details = adapter._argument_schema_diagnostics(arguments, adapter._section_tool_schema("save_entries"))
        self.assertTrue(any(row["path"] == "entries[0].status" and row["actual_type"] == "missing"
                            for row in details["issues"]))
        self.assertNotIn("PRIVATE_REPORT_VALUE", json.dumps(details))
        self.assertTrue(details["examples_are_shapes_not_defaults"])

    def test_invalid_json_feedback_has_exact_path_type_and_structural_example(self):
        details = adapter._argument_schema_diagnostics({"entries": "not JSON"}, adapter._section_tool_schema("save_entries"))
        self.assertEqual(details["issues"][0]["path"], "entries")
        self.assertEqual(details["issues"][0]["expected_type"], "array")
        self.assertTrue(any(row.get("minimal_shape_example") == [] for row in details["issues"]))

    def test_unavailable_native_prevalidation_entry_fails_closed(self):
        class MissingEntry(Tool):
            _normalize_and_validate_params = None
        with self.assertRaises(adapter.ToolSchemaUnavailableError):
            adapter._install_tool_schema(MissingEntry(), "save_entries", adapter._section_tool_schema("save_entries"))

    def test_actual_route_keeps_raw_identity_and_records_successful_conversion(self):
        seen = []
        raw = {"entries": json.dumps([basic(value="PRIVATE_REPORT_VALUE")], ensure_ascii=False)}
        before = deepcopy(raw)
        def script(registry, settings, outputs, prompt):
            args = registry["save_entries"]._normalize_and_validate_params(raw, strict_json=True)
            outputs.append(registry["save_entries"](**args))
            yield TextAction()
        sdk, _, _, _, _ = fake_sdk(script)
        with patch.object(adapter, "_load_tongagents", return_value=sdk):
            result = adapter.run_react("synthetic", {"save_entries": lambda entries: seen.extend(entries) or {"saved": True}},
                {"url": "http://127.0.0.1:8000/v1", "model": "own", "profile": "section",
                 "current_agent_call_id": lambda: "actual-synthetic-generation"})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(raw, before)
        self.assertEqual(seen, json.loads(raw["entries"]))
        audit = result["argument_normalizations"][0]
        self.assertTrue(audit["native_validation_passed"])
        self.assertEqual(audit["original_arguments_sha256"], adapter._hash(before))
        self.assertEqual(audit["normalized_arguments_sha256"], adapter._hash({"entries": seen}))
        self.assertEqual(audit["agent_call_id"], "actual-synthetic-generation")
        self.assertNotIn("PRIVATE_REPORT_VALUE", json.dumps(result["argument_normalizations"]))

    def test_failed_native_validation_records_conversion_but_never_calls_handler(self):
        raw = {"entries": '[{"id":"only-id"}]'}
        session = {"handlers": {"save_entries": lambda **kwargs: self.fail("No handler")},
                   "argument_normalizations": [], "current_agent_call_id": lambda: "g"}
        item = self.installed()
        with patch.object(adapter, "_ACTIVE_SESSION", session), self.assertRaises(ValidationError):
            item._normalize_and_validate_params(raw, strict_json=True)
        self.assertFalse(session["argument_normalizations"][0]["native_validation_passed"])

    def test_stopped_batch_association_requires_real_conversion_audit(self):
        original = {"entries": json.dumps([basic()])}
        normalized, changes, _ = adapter._normalize_container_arguments(original, adapter._section_tool_schema("save_entries"))
        calls = [call("save_entries", json.dumps(original), "saved"), call("save_entries", '{"entries":', "bad")]
        memory = Memory(calls)
        result = {"ok": True, "tool": "save_entries", "result": {"saved": 1}}
        memory.add_chunk(ToolResultMessage(name="save_entries", content=result, tool_call_id="saved"), "tool")
        native_calls, answers = native.native_tool_pairs(memory.get_messages())
        row = {"name": "save_entries", "agent_call_id": "g", "args_sha256": adapter._hash(normalized),
               "result_sha256": adapter._hash(result), "status": "success"}
        audit = {"tool_name": "save_entries", "agent_call_id": "g", "original_arguments_sha256": adapter._hash(original),
                 "normalized_arguments_sha256": adapter._hash(normalized), "schema_sha256": adapter._hash(adapter._section_tool_schema("save_entries")),
                 "conversions": changes, "native_validation_passed": True}
        error = {"agent_call_id": "g"}
        before = calls[0].arguments
        with self.assertRaises(adapter.NativeBatchRecoveryError):
            adapter._prove_stopped_native_batch(memory.get_messages(), native_calls, answers, "bad", error, [row])
        batch, proved, missing = adapter._prove_stopped_native_batch(memory.get_messages(), native_calls, answers,
            "bad", error, [row], argument_normalizations=[audit])
        self.assertEqual(proved, {"saved"}); self.assertEqual(missing, [])
        self.assertEqual(calls[0].arguments, before)
        for field in ("agent_call_id", "original_arguments_sha256", "normalized_arguments_sha256", "schema_sha256", "native_validation_passed"):
            changed = {**audit, field: None}
            with self.subTest(field=field), self.assertRaises(adapter.NativeBatchRecoveryError):
                adapter._prove_stopped_native_batch(memory.get_messages(), native_calls, answers, "bad", error, [row], argument_normalizations=[changed])

    def test_schema_failure_feedback_preserves_native_arguments_and_targets_only_missing_field(self):
        entry = basic(value="PRIVATE_REPORT_VALUE")
        del entry["status"]
        raw = json.dumps({"entries": json.dumps([entry])})
        memory = Memory([call("save_entries", raw, "bad")])
        error = {"code": "SDK_TOOL_ARGUMENT_SCHEMA", "tool_name": "save_entries", "tool_call_id_sha256": adapter._hash("bad"),
                 "agent_call_id": "g", "action": "Repair the listed paths, do not assume saved."}
        ToolJsonSchemaErrorAction = type("ToolJsonSchemaErrorAction", (), {})
        event = ToolJsonSchemaErrorAction()
        event.tool_call_id = "bad"
        api = {"messages": SimpleNamespace(ToolResultMessage=ToolResultMessage),
               "memory": SimpleNamespace(MessageRole=SimpleNamespace(TOOL="tool"))}
        with patch.object(native, "load_native_api", return_value=api):
            receipt, feedback = adapter._paired_parse_feedback(SimpleNamespace(), event, error, [],
                context=SimpleNamespace(memory=memory), tool_schema=adapter._section_tool_schema("save_entries"))
        retained = memory.messages[-1].chunks[0].content.content
        details = retained["error"]["argument_repair"]
        issue = next(item for item in details["issues"] if item["path"] == "entries[0].status")
        self.assertEqual(issue["minimal_shape_example"], {"status": "explicit"})
        self.assertTrue(details["examples_are_shapes_not_defaults"])
        self.assertEqual(memory.messages[0].chunks[0].content.tool_call_messages[0].arguments, raw)
        self.assertFalse(retained["arguments_repaired_by_host"])
        self.assertFalse(retained["error"]["handler_executed"])
        self.assertNotIn("PRIVATE_REPORT_VALUE", json.dumps(receipt) + json.dumps(retained))

    def test_same_normalized_value_from_multiple_native_calls_remains_ambiguous(self):
        original = {"entries": json.dumps([basic()])}
        normalized, changes, _ = adapter._normalize_container_arguments(original, adapter._section_tool_schema("save_entries"))
        calls = [call("save_entries", json.dumps(original), "first"), call("save_entries", json.dumps(normalized), "second"),
                 call("save_entries", '{"entries":', "bad")]
        memory = Memory(calls)
        result = {"ok": True, "tool": "save_entries", "result": {"saved": 1}}
        memory.add_chunk(ToolResultMessage(name="save_entries", content=result, tool_call_id="first"), "tool")
        native_calls, answers = native.native_tool_pairs(memory.get_messages())
        row = {"name": "save_entries", "agent_call_id": "g", "args_sha256": adapter._hash(normalized),
               "result_sha256": adapter._hash(result), "status": "success"}
        audit = {"tool_name": "save_entries", "agent_call_id": "g", "original_arguments_sha256": adapter._hash(original),
                 "normalized_arguments_sha256": adapter._hash(normalized), "schema_sha256": adapter._hash(adapter._section_tool_schema("save_entries")),
                 "conversions": changes, "native_validation_passed": True}
        with self.assertRaises(adapter.NativeBatchRecoveryError) as caught:
            adapter._prove_stopped_native_batch(memory.get_messages(), native_calls, answers, "bad", {"agent_call_id": "g"},
                [row], argument_normalizations=[audit])
        self.assertEqual(caught.exception.proof_reason, "handler_identity_ambiguous")

    def test_navigation_signatures_defaults_and_schema_use_program_owned_positions(self):
        seen = []
        def script(registry, settings, outputs, prompt):
            outputs.append(registry["read_outline"]())
            outputs.append(registry["search_in_document"](["one phrase"]))
            outputs.append(registry["read_outline"](limit=4, restart=True))
            outputs.append(registry["search_in_document"](["A", "B"], scope="document", max_matches=5))
            yield TextAction()
        sdk, _, _, _, _ = fake_sdk(script)
        def handler(name):
            return lambda **kwargs: seen.append((name, kwargs)) or {}
        with patch.object(adapter, "_load_tongagents", return_value=sdk):
            result = adapter.run_react("synthetic", {name: handler(name) for name in ("read_outline", "search_in_document")},
                {"url": "http://127.0.0.1:8000/v1", "model": "own", "profile": "section"})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(seen, [("read_outline", {"limit": 20, "restart": False}),
            ("search_in_document", {"query": ["one phrase"], "max_matches": 8, "scope": "default", "section_id": None, "restart": False}),
            ("read_outline", {"limit": 4, "restart": True}),
            ("search_in_document", {"query": ["A", "B"], "scope": "document", "section_id": None, "max_matches": 5, "restart": False})])
        self.assertEqual(inspect.signature(adapter._read_outline).parameters["limit"].default, 20)
        for name, arguments in (("read_outline", {"cursor": True}), ("read_outline", {"limit": 101}), ("read_outline", {"limit": 1.0}),
              ("search_in_document", {"query": []}), ("search_in_document", {"query": ["a"], "cursor": 0.0}), ("search_in_document", {"query": ["a"]*17}),
              ("search_in_document", {"query": ["a"], "scope": "section"}),
              ("search_in_document", {"query": ["a"], "scope": "document", "section_id": "S1"}),
              ("search_in_document", {"query": ["  "]})):
            with self.subTest(name=name, arguments=arguments), self.assertRaises(ValueError):
                adapter._validate_arguments(name, arguments)
        args = {"query": ['["literal JSON-looking phrase"]']}
        self.assertEqual(adapter._normalize_container_arguments(args, adapter._section_tool_schema("search_in_document"))[0], args)

    def test_search_native_contract_is_array_in_both_registration_orders(self):
        for order in (("reading", "section"), ("section", "reading")):
            with self.subTest(order=order):
                sdk, registry, _, _, _ = fake_sdk(lambda *args: ())
                for profile in order:
                    adapter._register_tools(sdk["tool"], ["search_in_document"], profile=profile)
                    schema = adapter._registered_tool_schema(sdk["tool"], "search_in_document")
                    self.assertEqual(schema, adapter._section_tool_schema("search_in_document"))
                    self.assertEqual(schema["properties"]["query"]["type"], "array")
                tool = registry["search_in_document"]
                for query in (["病害", "新增"], ["左幅 1号梁"], ['["literal"]']):
                    expected = {"query": query}
                    self.assertEqual(tool._normalize_and_validate_params(expected), expected)
                    self.assertEqual(tool._normalize_and_validate_params({"query": json.dumps(query)}), expected)
                for query in ("病害 新增", "病害", "[not JSON]", '["病害",]', '[["病害"]]', '[1]', '[]'):
                    with self.subTest(query=query), self.assertRaises((ValueError, ValidationError)):
                        tool._normalize_and_validate_params({"query": query})


if __name__ == "__main__":
    unittest.main()
