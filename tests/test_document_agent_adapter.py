"""Synthetic fake-SDK contracts, NOT evidence of a real SDK/model execution."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, get_type_hints
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest import tong_react_adapter as module
from bridge_evidence.contest.document_review_payloads import ReviewPayloadBudgetError
from bridge_evidence.contest.document_workspace import (
    DocumentWorkspace, WorkspaceConflictError, WorkspaceError, WorkspaceIdentityError,
)
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from test_tong_react_adapter import TextAction, fake_sdk as _legacy_fake_sdk


class SchemaParameters:
    """Metadata storage double; JSON Schema checks use jsonschema separately."""
    def __init__(self, schema):
        self.schema = deepcopy(schema)

    def to_json_schema(self):
        return deepcopy(self.schema)


class SchemaTool:
    def _normalize_and_validate_params(self, params, *, strict_json=False):
        Draft202012Validator(type(self).parameters.to_json_schema()).validate(params)
        return deepcopy(params)

    @classmethod
    def get_original_tool_info(cls):
        cls.info_reads.append("original")
        return SimpleNamespace(parameters=cls.parameters)

    @classmethod
    def get_tool_info(cls):
        cls.info_reads.append("current")
        return SimpleNamespace(parameters=cls.parameters)


def fake_sdk(script):
    """Extend only document tests with the SDK's public metadata contract."""
    sdk, registry, settings, outputs, registered = _legacy_fake_sdk(script)
    original = sdk["tool"]
    schema_tools = {}

    def tool(**definition):
        def decorate(function):
            original(**definition)(function)
            instance = type("SyntheticDocumentTool", (SchemaTool,), {"info_reads": []})()
            schema_tools[definition["name"]] = instance
            return instance
        return decorate

    sdk.update(tool=tool, schema_tools=schema_tools)
    return sdk, registry, settings, outputs, registered


class ToolErrorAction:
    def __init__(self, tool_name="update_workspace", message="Parameters must be formatted as a valid JSON!"):
        self.tool_name, self.message = tool_name, message

    def model_dump(self, **kwargs):
        return {"tool_name": self.tool_name, "message": self.message, "code": 0,
                "tool_call_id": "synthetic-sdk-call", "is_final_response": False}


class ToolJsonSchemaErrorAction(ToolErrorAction):
    pass


class DocumentAgentAdapterTests(unittest.TestCase):
    config = {"url": "http://127.0.0.1:8000/v1", "model": "own-model"}

    def setUp(self):
        self.assertIsNone(module._ACTIVE_SESSION)
        module._REGISTRATIONS.clear()
        loader = patch.object(module, "_load_tool_schema_api", return_value=(SchemaTool, SchemaParameters))
        loader.start()
        self.addCleanup(loader.stop)

    def invoke(self, script, handlers, **config):
        sdk, registry, settings, outputs, registered = fake_sdk(script)
        with patch.object(module, "_load_tongagents", return_value=sdk):
            result = module.run_react("inspect this document", handlers, {**self.config, **config})
        return result, registry, settings, outputs, registered

    def test_document_tools_are_typed_dispatchers_and_update_object_is_serialized_losslessly(self):
        calls = []
        update = {"base_version": 7, "facts": [{"id": "F1", "value": '带"引号"与\\路径\n换行😀'}],
                  "draft_patch": [], "unknown": None}
        arguments = {
            "read_outline": {"limit": 20, "restart": False},
            "read_table_rows": {"table_id": "T1", "max_rows": 2, "restart": False},
            "read_source": {"evidence_id": "E1", "restart": False},
            "read_section": {"section_id": "S1", "max_units": 3, "restart": False},
            "read_image": {"image_id": "I1", "question": "检查裂缝", "region_json": "{}", "context_refs": ["E1"]},
            "search_in_document": {"query": ["历史"], "scope": "default", "section_id": None,
                                   "max_matches": 8, "restart": False},
            "get_neighbors": {"evidence_id": "E1"},
            "read_workspace": {},
            "read_workspace_page": {"view": "facts", "limit": 5, "restart": False},
            "update_workspace": {"update": update},
            "review_draft": {},
            "finish": {},
        }

        def handler(name):
            def dispatch(**args):
                calls.append((name, args))
                return {"received": args}
            return dispatch

        def script(registry, settings, outputs, prompt):
            for name, args in arguments.items():
                outputs.append(registry[name](**args))
            yield TextAction("workspace reviewed")

        result, registry, settings, outputs, registered = self.invoke(
            script, {name: handler(name) for name in arguments}, max_calls=20,
        )
        expected = {**arguments, "update_workspace": {"update_json": module._json_bytes(update).decode("utf-8")}}
        self.assertEqual(calls, list(expected.items()))
        self.assertEqual(json.loads(dict(calls)["update_workspace"]["update_json"]), update)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(all(output["ok"] for output in outputs))
        self.assertEqual(registered, list(arguments))
        self.assertEqual(settings[0].tool_identifier_list, list(arguments))
        self.assertIsNone(result["model_call_count"])
        self.assertEqual(get_type_hints(registry["read_source"]), {
            "evidence_id": str, "restart": bool, "return": dict[str, Any],
        })
        self.assertEqual(get_type_hints(registry["read_section"])["max_units"], int)
        self.assertEqual(get_type_hints(registry["read_image"])["context_refs"], list[str])
        self.assertEqual(get_type_hints(registry["read_workspace_page"]), {
            "view": str, "limit": int, "restart": bool, "return": dict[str, Any],
        })
        self.assertEqual(get_type_hints(registry["update_workspace"]), {"update": dict[str, Any], "return": dict[str, Any]})

    def test_native_tool_metadata_exposes_nested_requirements_through_both_info_paths(self):
        seen = []
        update = {"base_version": 4, "facts": [{"id": "F1", "value": "changed"}]}

        def script(registry, settings, outputs, prompt):
            outputs.append(registry["update_workspace"](update))
            yield TextAction()

        sdk, registry, _, outputs, _ = fake_sdk(script)
        with patch.object(module, "_load_tongagents", return_value=sdk):
            result = module.run_react("synthetic update", {
                "update_workspace": lambda update_json: seen.append(json.loads(update_json)) or {},
            }, self.config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(seen, [update])  # No invented version, status or defaults.
        tool = sdk["schema_tools"]["update_workspace"]
        self.assertEqual(tool.info_reads, ["original", "current"])
        original = tool.get_original_tool_info().parameters.to_json_schema()
        current = tool.get_tool_info().parameters.to_json_schema()
        self.assertEqual(original, current)
        self.assertEqual(current, module._update_workspace_schema())
        self.assertEqual(current["required"], ["update"])
        inner = current["properties"]["update"]
        self.assertEqual(inner["required"], ["base_version"])
        self.assertEqual(inner["properties"]["base_version"]["type"], "integer")
        self.assertEqual(inner["properties"]["base_version"]["minimum"], 0)
        self.assertEqual(inner["properties"]["facts"]["items"]["required"], ["id"])
        self.assertNotIn('"default"', json.dumps(current))
        validator = Draft202012Validator(current)
        validator.check_schema(current)
        self.assertTrue(validator.is_valid({"update": update}))
        self.assertTrue(validator.is_valid({"update": {"base_version": 0,
            "draft": {"基本信息": {}, "维修建议": [], "病害明细": []},
            "draft_fact_ids": [], "draft_bindings": {}, "exclusions": []}}))
        for bad in ({}, {"facts": []}, {"base_version": -1}, {"base_version": True},
                    {"base_version": "4"}, {"base_version": 4, "facts": [{"value": "no ID"}]},
                    {"base_version": 4, "facts": [{"id": "F1", "status": "guessed"}]},
                    {"base_version": 4, "draft_patch": [{"op": "delete_everything"}]}):
            with self.subTest(bad=bad):
                self.assertFalse(validator.is_valid({"update": bad}))

    def test_metadata_preserves_real_partial_fact_updates_and_workspace_rejects_invalid_versions(self):
        with TemporaryDirectory() as directory:
            workspace = DocumentWorkspace(manifest_from_text("桥名：合成桥\n"), Path(directory), {"model": "synthetic"})
            unit = workspace.manifest["units"][0]
            workspace.record_read(unit["id"], 0, len(unit["text"]))
            initial = workspace.apply_update({"base_version": workspace.snapshot()["version"],
                "facts": [{"id": "F1", "value": "旧值", "status": "explicit",
                           "source_refs": [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}]}]})
            seen = []
            invalid = [{}, *({"base_version": value} for value in (-1, True, None, "2", 2.0, 0))]
            corrected = {"base_version": initial["version"], "facts": [{"id": "F1", "value": "合成桥"}]}

            def write(update_json):
                update = json.loads(update_json)
                seen.append(deepcopy(update))
                return {"version": workspace.apply_update(update)["version"]}

            def script(registry, settings, outputs, prompt):
                for update in invalid:
                    response = registry["update_workspace"](update)
                    outputs.append(response)
                    self.assertEqual(response["error"]["code"], "WORKSPACE_STALE")
                    self.assertEqual(workspace.snapshot(), initial)
                outputs.append(registry["update_workspace"](corrected))
                current_version = workspace.snapshot()["version"]
                # New records still need all required fields, even though the
                # schema permits a partial patch for an existing record.
                outputs.append(registry["update_workspace"]({"base_version": current_version,
                    "facts": [{"id": "F_NEW", "value": "不完整的新事实"}]}))
                yield TextAction()

            result, _, _, outputs, _ = self.invoke(script, {"update_workspace": write}, max_calls=12)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(seen[:len(invalid)], invalid)
            self.assertTrue(outputs[-2]["ok"])
            self.assertEqual(outputs[-1]["error"]["code"], "FACT_FIELDS")
            saved = workspace.snapshot()["facts"]["F1"]
            self.assertEqual(saved["value"], "合成桥")
            self.assertEqual(saved["status"], initial["facts"]["F1"]["status"])
            self.assertEqual(saved["source_refs"], initial["facts"]["F1"]["source_refs"])
            self.assertNotIn("F_NEW", workspace.snapshot()["facts"])

    def test_schema_capability_failure_stops_before_agent_creation(self):
        for mode in ("unavailable", "original_dropped", "current_dropped", "not_tool"):
            with self.subTest(mode=mode):
                module._REGISTRATIONS.clear()
                factory = _legacy_fake_sdk if mode == "not_tool" else fake_sdk
                sdk, _, settings, _, _ = factory(lambda *args: self.fail("agent must not run"))
                if mode == "unavailable":
                    failure = patch.object(module, "_load_tool_schema_api", side_effect=ImportError("unsupported"))
                elif mode.endswith("dropped"):
                    accessor = "get_original_tool_info" if mode == "original_dropped" else "get_tool_info"
                    failure = patch.object(SchemaTool, accessor,
                        return_value=SimpleNamespace(parameters=SchemaParameters({"type": "object"})))
                else:
                    failure = patch.object(module, "_load_tool_schema_api", return_value=(SchemaTool, SchemaParameters))
                with failure, patch.object(module, "_load_tongagents", return_value=sdk):
                    result = module.run_react("schema check", {"update_workspace": lambda **kwargs: {}}, self.config)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["failure_reason"], "tool_schema_unavailable")
                self.assertIn("no agent was started", result["error"]["message"])
                self.assertEqual(result["tool_calls"], [])
                self.assertEqual(settings, [])
                self.assertIsNone(module._ACTIVE_SESSION)

    def test_divide_uses_native_schemas_for_shared_read_and_finish_tools(self):
        names = ("read_outline", "read_evidence", "extract_chunk", "review_conflict", "split_task", "finish")

        def script(registry, settings, outputs, prompt):
            outputs.append(registry["finish"]())
            yield TextAction()

        sdk, registry, _, _, registered = fake_sdk(script)
        with patch.object(module, "_load_tool_schema_api", return_value=(SchemaTool, SchemaParameters)) as loader:
            with patch.object(module, "_load_tongagents", return_value=sdk):
                result = module.run_react("legacy divide", {name: lambda **kwargs: {} for name in names}, self.config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(registered, list(names))
        self.assertTrue(all(registry[name] is module._TOOLS[name][0] for name in names))
        self.assertGreater(loader.call_count, 0)
        self.assertEqual(sdk["schema_tools"]["read_outline"].parameters.to_json_schema(),
                         module._section_tool_schema("read_outline"))
        self.assertEqual(sdk["schema_tools"]["finish"].parameters.to_json_schema(),
                         module._delivery_schema("finish"))

    def test_invalid_source_section_and_workspace_pages_are_rejected_before_dispatch(self):
        cases = []
        for start, end in ((-1, 1), (0, 0), (2, 1), (False, 1), (0, True), (0.0, 1), (0, 1.0), ("0", 1)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                module._validate_arguments("read_source_range", {"evidence_id": "E1", "start": start, "end": end})
        for cursor, units in ((-1, 1), (0, 0), (0, -1), (False, 1), (0, True), (0.0, 1), (0, 1.0), ("0", 1)):
            with self.subTest(cursor=cursor, units=units), self.assertRaises(ValueError):
                module._validate_arguments("read_section_range", {"section_id": "S1", "cursor": cursor, "max_units": units})
        for cursor, limit in ((-1, 1), (0, 0), (0, -1), (False, 1), (0, True), (0.0, 1), (0, 1.0), ("0", 1), (0, "1"), (None, 1)):
            with self.subTest(cursor=cursor, limit=limit), self.assertRaises(ValueError):
                module._validate_arguments("read_workspace_page", {"view": "facts", "cursor": cursor, "limit": limit})
        for restart in (0, 1, None, "false", [], {}):
            cases.append(("read_source", {"evidence_id": "E1", "restart": restart}))
        for limit in (0, -1, True, 1.0, "1", None):
            cases.append(("read_section", {"section_id": "S1", "max_units": limit}))
            cases.append(("read_workspace_page", {"view": "facts", "limit": limit}))
        for name, args in cases:
            with self.subTest(name=name, args=args):
                def script(registry, settings, outputs, prompt):
                    outputs.append(registry[name](**args))
                    yield TextAction()

                result, _, _, outputs, _ = self.invoke(script, {
                    name: lambda **kwargs: self.fail("invalid pagination reached handler"),
                })
                self.assertFalse(outputs[0]["ok"])
                self.assertEqual(result["tool_error_count"], 1)
                self.assertEqual(result["tool_calls"][0]["error"]["type"], "ValueError")

    def test_workspace_page_requires_nonempty_string_view_and_keeps_page_version(self):
        for view in ("", " ", None, 0, False, []):
            with self.subTest(view=view):
                def script(registry, settings, outputs, prompt):
                    outputs.append(registry["read_workspace_page"](view, limit=1))
                    yield TextAction()

                result, _, _, outputs, _ = self.invoke(script, {
                    "read_workspace_page": lambda **kwargs: self.fail("invalid view reached handler"),
                })
                self.assertFalse(outputs[0]["ok"])
                self.assertEqual(result["tool_calls"][0]["error"]["type"], "ValueError")

        pages = []

        def read_page(view, limit, restart):
            cursor = 0 if restart else len(pages)
            pages.append((view, limit, restart))
            return {"version": 7, "cursor": cursor, "next_cursor": cursor + limit,
                    "items": [{"id": f"synthetic-{cursor}"}]}

        def script(registry, settings, outputs, prompt):
            outputs.append(registry["read_workspace_page"]("facts", limit=1))
            outputs.append(registry["read_workspace_page"]("facts", limit=1))
            yield TextAction()

        result, _, _, outputs, _ = self.invoke(script, {"read_workspace_page": read_page})
        self.assertEqual(pages, [("facts", 1, False), ("facts", 1, False)])
        self.assertEqual([row["result"]["version"] for row in outputs], [7, 7])
        self.assertEqual([row["result"]["next_cursor"] for row in outputs], [1, 2])
        self.assertEqual(result["tool_error_count"], 0)

    def test_context_references_require_an_array_of_nonempty_strings(self):
        for refs in ("E1", ("E1",), [1], [True], [""], ["  "], None):
            with self.subTest(refs=refs):
                def script(registry, settings, outputs, prompt):
                    outputs.append(registry["read_image"]("I1", "核对图片", "{}", refs))
                    yield TextAction()

                result, _, _, outputs, _ = self.invoke(script, {
                    "read_image": lambda **kwargs: self.fail("invalid context_refs reached handler"),
                })
                self.assertFalse(outputs[0]["ok"])
                self.assertEqual(result["tool_error_count"], 1)

    def test_update_string_is_rejected_but_region_json_still_uses_its_existing_contract(self):
        seen = []

        def update_workspace(update_json):
            seen.append(update_json)
            return json.loads(update_json)

        def script(registry, settings, outputs, prompt):
            outputs.append(registry["update_workspace"]("{invalid"))
            outputs.append(registry["update_workspace"]({}))
            outputs.append(registry["read_image"]("I1", "核对图片", "{invalid", []))
            yield TextAction()

        result, _, _, outputs, _ = self.invoke(script, {
            "update_workspace": update_workspace,
            "read_image": lambda **args: {"region_json": args["region_json"], "context_refs": args["context_refs"]},
        })
        self.assertEqual(seen, ["{}"])
        self.assertEqual([row["ok"] for row in outputs], [False, True, True])
        self.assertEqual(outputs[2]["result"], {"region_json": "{invalid", "context_refs": []})
        self.assertEqual(result["tool_error_count"], 1)
        self.assertEqual(result["tool_calls"][0]["error"]["code"], "UPDATE_OBJECT_ARGUMENTS")

    def test_nonfinite_or_python_only_updates_fail_inside_dispatch_with_receipts(self):
        cyclic = {}
        cyclic["self"] = cyclic
        updates = [{"value": float("nan")}, {"value": float("inf")}, {"value": -float("inf")},
                   {1: "nonstring key"}, {"value": (1, 2)}, {"value": {1, 2}},
                   {"value": object()}, {"value": "\ud800"}, cyclic]
        for index, update in enumerate(updates):
            with self.subTest(index=index):
                def script(registry, settings, outputs, prompt):
                    outputs.append(registry["update_workspace"](update))
                    yield TextAction()

                result, _, _, outputs, _ = self.invoke(script, {
                    "update_workspace": lambda **kwargs: self.fail("non-JSON value reached handler"),
                })
                self.assertEqual(outputs[0]["error"]["code"], "UPDATE_OBJECT_ARGUMENTS")
                self.assertEqual(result["tool_error_count"], 1)
                self.assertEqual(result["sdk_tool_error_count"], 0)
                self.assertEqual(len(result["tool_calls"]), 1)
                self.assertEqual(result["tool_calls"][0]["status"], "failed")
                self.assertIsNone(result["tool_calls"][0]["args_sha256"])
                json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
                self.assertIsNone(module._ACTIVE_SESSION)

    def test_sdk_argument_parse_failure_is_audited_without_claiming_handler_execution(self):
        event = ToolErrorAction()

        def script(registry, settings, outputs, prompt):
            yield TextAction("first reasoning", final=False)
            yield event

        result, *_ = self.invoke(script, {
            "update_workspace": lambda **kwargs: self.fail("SDK parse error must not execute handler"),
        })
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "sdk_tool_error")
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(result["tool_error_count"], 0)
        self.assertEqual(result["sdk_tool_error_count"], 1)
        error = result["sdk_tool_errors"][0]
        self.assertEqual(error["tool_name"], "update_workspace")
        self.assertEqual(error["code"], "SDK_TOOL_ARGUMENT_JSON")
        self.assertIn("object", error["action"])
        self.assertEqual(error["event_sha256"], module._hash(event.model_dump()))
        self.assertNotIn(event.message, str(result))

    def test_sdk_errors_never_echo_arbitrary_messages_or_unenabled_tool_names(self):
        private = "private report text /private/path api_key=hidden"
        cases = [
            (ToolErrorAction(private, private), None, "SDK_TOOL_ERROR"),
            (ToolErrorAction("read_source", private), None, "SDK_TOOL_ERROR"),
            (ToolJsonSchemaErrorAction("update_workspace", private), "update_workspace", "SDK_TOOL_ARGUMENT_SCHEMA"),
            (ToolErrorAction("update_workspace", "Parameters must be formatted as a valid JSON! " + private),
             "update_workspace", "SDK_TOOL_ERROR"),
        ]
        for event, name, code in cases:
            with self.subTest(code=code, name=name):
                def script(registry, settings, outputs, prompt):
                    yield event

                result, *_ = self.invoke(script, {"update_workspace": lambda update_json: {}})
                error = result["sdk_tool_errors"][0]
                self.assertEqual(error["tool_name"], name)
                self.assertEqual(error["code"], code)
                self.assertNotIn(private, str(result))
                required = {"type", "tool_name", "code", "action", "event_sha256"}
                self.assertTrue(required <= set(error))
                self.assertTrue(set(error) <= required | {"tool_call_id_sha256", "arguments_text_sha256"})
                for field in set(error) - required:
                    self.assertRegex(error[field], r"^[0-9a-f]{64}$")
                self.assertEqual(result["tool_calls"], [])

    def test_sdk_error_can_be_followed_by_native_recovery_without_a_fake_handler_error(self):
        seen = []

        def script(registry, settings, outputs, prompt):
            yield ToolErrorAction()
            outputs.append(registry["update_workspace"]({"base_version": 0, "draft_patch": []}))
            yield TextAction()

        result, _, _, outputs, _ = self.invoke(script, {
            "update_workspace": lambda update_json: seen.append(json.loads(update_json)) or {},
        })
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["sdk_tool_error_count"], 1)
        self.assertEqual(result["tool_error_count"], 0)
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(seen, [{"base_version": 0, "draft_patch": []}])
        self.assertTrue(outputs[0]["ok"])

    def test_sdk_error_survives_a_later_sdk_exception(self):
        def script(registry, settings, outputs, prompt):
            yield ToolErrorAction()
            raise RuntimeError("private SDK response")

        result, *_ = self.invoke(script, {"update_workspace": lambda update_json: {}})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["failure_reason"], "sdk_tool_error")
        self.assertEqual(result["sdk_tool_error_count"], 1)
        self.assertEqual(result["tool_error_count"], 0)
        self.assertNotIn("private SDK response", str(result))

    def test_nonobject_updates_and_blank_identifiers_are_rejected(self):
        cases = [
            ("update_workspace", {"update": "{}"}),
            ("update_workspace", {"update": " "}),
            ("update_workspace", {"update": []}),
            ("update_workspace", {"update": None}),
            ("read_source", {"evidence_id": " ", "restart": False}),
            ("read_section", {"section_id": "", "max_units": 1}),
            ("read_image", {"image_id": "I1", "question": "", "region_json": "{}", "context_refs": []}),
            ("read_image", {"image_id": "I1", "question": "核对", "region_json": {}, "context_refs": []}),
        ]
        for name, args in cases:
            with self.subTest(name=name, args=args):
                def script(registry, settings, outputs, prompt):
                    outputs.append(registry[name](**args))
                    yield TextAction()

                result, _, _, outputs, _ = self.invoke(script, {
                    name: lambda **kwargs: self.fail("invalid argument reached handler"),
                })
                self.assertFalse(outputs[0]["ok"])
                self.assertEqual(result["tool_error_count"], 1)

    def test_small_workspace_updates_and_image_questions_are_not_in_audit_receipts(self):
        update = {"源文不能作为审计键回显": "源文原句"}
        question = "源文图片中该对象是否有裂缝"

        def script(registry, settings, outputs, prompt):
            registry["update_workspace"](update)
            registry["read_image"]("I1", question, '{"label":"图中标注"}', ["E1"])
            yield TextAction("done", response_meta={"finish_reason": "stop"})

        result, *_ = self.invoke(script, {
            "update_workspace": lambda update_json: {"echo": update_json},
            "read_image": lambda **args: args,
        })
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("源文原句", str(result))
        self.assertNotIn(question, str(result))
        self.assertNotIn("图中标注", str(result))
        self.assertNotIn("源文不能作为审计键回显", str(result))
        self.assertEqual(result["tool_calls"][0]["args"]["update"]["sha256"], module._hash(update))
        self.assertEqual(result["tool_calls"][1]["args"]["context_refs"], ["E1"])
        self.assertEqual(result["events"][0]["type"], "TextAction")
        self.assertTrue(result["events"][0]["is_final_response"])
        self.assertIsNone(result["model_call_count"])

    def test_stable_document_tools_rebind_workspace_and_enforce_session_whitelists(self):
        changes = []

        def script(registry, settings, outputs, prompt):
            outputs.append(registry["read_workspace"]())
            # This cached dispatcher is unauthorized in the second session.
            outputs.append(registry["update_workspace"]({}))
            yield TextAction()

        sdk, registry, _, outputs, registered = fake_sdk(script)
        with patch.object(module, "_load_tongagents", return_value=sdk):
            first = module.run_react("first", {
                "read_workspace": lambda: {"document_id": "first"},
                "update_workspace": lambda update_json: changes.append("first") or {},
            }, self.config)
            second = module.run_react("second", {
                "read_workspace": lambda: {"document_id": "second"},
            }, self.config)
        self.assertEqual([first["status"], second["status"]], ["completed", "completed"])
        self.assertEqual([outputs[0]["result"]["document_id"], outputs[2]["result"]["document_id"]], ["first", "second"])
        self.assertFalse(outputs[3]["ok"])
        self.assertEqual(changes, ["first"])
        self.assertEqual(registered, ["read_workspace", "update_workspace"])
        self.assertEqual(registry["read_workspace"]()["error"]["type"], "NoActiveSession")

    def test_system_and_user_prompts_are_separate_only_when_explicitly_configured(self):
        seen = []

        def script(registry, settings, outputs, prompt):
            seen.append((settings.system_prompt, prompt))
            registry["read_workspace"]()
            yield TextAction()

        sdk, _, _, _, _ = fake_sdk(script)
        with patch.object(module, "_load_tongagents", return_value=sdk):
            old = module.run_react("legacy task", {"read_workspace": lambda: {}}, self.config)
            new = module.run_react("only this document task", {"read_workspace": lambda: {}}, {
                **self.config, "system_prompt": "fixed document-agent instructions",
            })
        self.assertEqual([old["status"], new["status"]], ["completed", "completed"])
        self.assertEqual(seen, [
            ("legacy task", "legacy task"),
            ("fixed document-agent instructions", "only this document task"),
        ])

    def test_invalid_system_prompt_fails_before_loading_sdk(self):
        for system in (None, "", " ", 1, False, {}):
            with self.subTest(system=system), patch.object(module, "_load_tongagents") as loader:
                result = module.run_react("task", {"finish": lambda: {}}, {
                    **self.config, "system_prompt": system,
                })
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error"]["type"], "ValueError")
                loader.assert_not_called()

    def test_real_workspace_unread_and_stale_errors_give_distinct_recovery_actions(self):
        with TemporaryDirectory() as directory:
            workspace = DocumentWorkspace(manifest_from_text("桥名：合成桥\n"), Path(directory), {"model": "synthetic"})
            unit = workspace.manifest["units"][0]
            update = {"base_version": 0, "facts": [{"id": "name", "value": "合成桥", "status": "explicit",
                       "source_refs": [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}]}]}

            def write(update_json):
                state = workspace.apply_update(json.loads(update_json))
                return {"version": state["version"]}

            def script(registry, settings, outputs, prompt):
                outputs.append(registry["update_workspace"](update))
                outputs.append(registry["read_source"](unit["id"]))
                outputs.append(registry["update_workspace"](update))
                current = registry["read_workspace"]()
                outputs.append(current)
                update["base_version"] = current["result"]["version"]
                outputs.append(registry["update_workspace"](update))
                yield TextAction()

            result, _, _, outputs, _ = self.invoke(script, {
                "update_workspace": write,
                "read_source": lambda evidence_id, restart: workspace.record_read(evidence_id, 0, len(unit["text"])) or {"read": True},
                "read_workspace": lambda: {"version": workspace.snapshot()["version"]},
            })
            self.assertEqual(outputs[0]["error"]["code"], "SOURCE_NOT_READ")
            self.assertIn("read_source", outputs[0]["error"]["action"])
            self.assertEqual(outputs[2]["error"]["code"], "WORKSPACE_STALE")
            self.assertIn("base_version", outputs[2]["error"]["fields"])
            self.assertTrue(outputs[-1]["ok"])
            self.assertEqual(workspace.snapshot()["facts"]["name"]["value"], "合成桥")
            self.assertEqual(result["tool_error_count"], 2)

    def test_real_draft_binding_failure_identifies_fields_then_accepts_repair(self):
        with TemporaryDirectory() as directory:
            workspace = DocumentWorkspace(manifest_from_text("桥名：合成桥\n"), Path(directory), {"model": "synthetic"})
            unit = workspace.manifest["units"][0]
            workspace.record_read(unit["id"], 0, len(unit["text"]))
            update = {"base_version": workspace.snapshot()["version"],
                      "facts": [{"id": "name", "value": "合成桥", "status": "explicit",
                                 "source_refs": [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}]}],
                      "draft": {"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [], "病害明细": []},
                      "draft_fact_ids": ["name"], "draft_bindings": {}}

            def script(registry, settings, outputs, prompt):
                outputs.append(registry["update_workspace"](update))
                update["draft_bindings"] = {"基本信息.桥梁名称": ["name"]}
                outputs.append(registry["update_workspace"](update))
                yield TextAction()

            result, _, _, outputs, _ = self.invoke(script, {
                "update_workspace": lambda update_json: {"version": workspace.apply_update(json.loads(update_json))["version"]},
            })
            self.assertEqual(outputs[0]["error"]["code"], "DRAFT_BINDINGS")
            self.assertEqual(outputs[0]["error"]["fields"], ["draft", "draft_fact_ids", "draft_bindings"])
            self.assertTrue(outputs[1]["ok"])
            self.assertTrue(workspace.snapshot()["draft_valid"])
            self.assertEqual(result["tool_error_count"], 1)

    def test_review_budget_feedback_keeps_safe_numbers_and_no_paths_or_source_arrays(self):
        secret = "/private/report/secret-token 原文不能外泄"
        failure = ReviewPayloadBudgetError({
            "prompt_tokens": 70000, "max_model_len": 65536, "output_tokens": 4096,
            "minimum_protocol_output_tokens": False,
            "source_spans": [{"evidence_id": secret, "start": 0, "end": 1}] * 1000,
            "related_paths": [secret] * 1000, "private": secret,
        })

        def handler():
            raise failure

        def script(registry, settings, outputs, prompt):
            outputs.append(registry["review_draft"]())
            yield TextAction()

        result, _, _, outputs, _ = self.invoke(script, {"review_draft": handler},
                                             max_tool_tokens=1024, count_tokens=lambda text: len(text.encode("utf-8")))
        error = outputs[0]["error"]
        self.assertEqual(error["code"], "REVIEW_CONTEXT_BUDGET")
        self.assertEqual(error["details"], {"prompt_tokens": 70000, "max_model_len": 65536, "output_tokens": 4096})
        self.assertIn("do not drop evidence", error["action"])
        self.assertNotIn(secret, str(outputs) + str(result))
        self.assertNotIn("source_spans", error)
        self.assertLessEqual(len(module._json_bytes(error)), module._ERROR_FEEDBACK_BYTES)
        self.assertLessEqual(len(module._json_bytes(outputs[0])), 1024)

    def test_exception_class_whitelist_never_echoes_private_or_unknown_exception_details(self):
        secret = "/private/run/secret api_key=hidden report paragraph"
        impostor = type("WorkspaceError", (ValueError,), {})
        failures = [
            (WorkspaceError("Source interval was not actually read: " + secret), "SOURCE_NOT_READ"),
            (WorkspaceError(secret), "WORKSPACE_INVALID"),
            (WorkspaceConflictError(secret), "WORKSPACE_STALE"),
            (WorkspaceIdentityError(secret), "WORKSPACE_IDENTITY"),
            (ValueError(secret), "TOOL_FAILED"),
            (RuntimeError(secret), "TOOL_FAILED"),
            (impostor(secret), "TOOL_FAILED"),
            (WorkspaceError(secret * 10000), "WORKSPACE_INVALID"),
        ]
        for failure, code in failures:
            with self.subTest(error_type=type(failure), code=code):
                def handler(update_json):
                    raise failure

                def script(registry, settings, outputs, prompt):
                    outputs.append(registry["update_workspace"]({}))
                    yield TextAction()

                result, _, _, outputs, _ = self.invoke(script, {"update_workspace": handler})
                self.assertEqual(outputs[0]["error"]["code"], code)
                self.assertNotIn(secret, str(outputs) + str(result))
                self.assertNotIn("frames", outputs[0]["error"])
                self.assertLessEqual(len(module._json_bytes(outputs[0]["error"])), module._ERROR_FEEDBACK_BYTES)
                if type(failure) is WorkspaceIdentityError:
                    self.assertFalse(outputs[0]["error"]["recoverable"])
                if type(failure) is impostor:
                    self.assertEqual(outputs[0]["error"]["type"], "ToolError")

    def test_complete_response_budget_includes_wrapper_and_page_can_be_retried_smaller(self):
        counter = lambda text: len(text.encode("utf-8"))
        source = "原文" * 200
        large = {"text": source[:200], "continuation": {"start": 200, "end": len(source)}}
        budget = counter(module._json_bytes(large).decode("utf-8"))
        self.assertGreater(budget, counter(module._json_bytes(module._tool_budget_error()).decode("utf-8")))

        def handler(section_id, max_units, restart):
            return {"text": source[:max_units], "continuation": {"start": max_units, "end": len(source)}}

        def script(registry, settings, outputs, prompt):
            outputs.append(registry["read_section"]("S1", max_units=200))
            outputs.append(registry["read_section"]("S1", max_units=10, restart=True))
            yield TextAction()

        result, _, _, outputs, _ = self.invoke(script, {"read_section": handler},
                                             max_tool_tokens=budget, count_tokens=counter)
        self.assertEqual(outputs[0]["error"]["code"], "TOOL_RESPONSE_BUDGET")
        self.assertNotIn("result", outputs[0])  # No silently shortened source.
        self.assertIn("page/span", outputs[0]["error"]["action"])
        self.assertTrue(outputs[1]["ok"])
        self.assertEqual(outputs[1]["result"]["text"], source[:10])
        self.assertEqual(outputs[1]["result"]["continuation"]["start"], 10)
        self.assertTrue(all(counter(module._json_bytes(row).decode("utf-8")) <= budget for row in outputs))
        self.assertEqual(result["tool_error_count"], 1)
        self.assertEqual(result["tool_calls"][0]["result_sha256"], module._hash(outputs[0]))

    def test_expected_error_feedback_obeys_budget_and_tokenizer_remains_in_memory(self):
        class Counter:
            def __call__(self, text):
                return len(text.encode("utf-8"))

            def __repr__(self):
                return "private-tokenizer-secret"

        counter = Counter()
        budget = counter(module._json_bytes(module._tool_budget_error()).decode("utf-8"))

        def handler():
            raise ReviewPayloadBudgetError({"prompt_tokens": 100000, "max_model_len": 65536, "output_tokens": 4096})

        def script(registry, settings, outputs, prompt):
            outputs.append(registry["review_draft"]())
            yield TextAction()

        result, _, sdk_settings, outputs, _ = self.invoke(script, {"review_draft": handler},
                                                        max_tool_tokens=budget, count_tokens=counter)
        self.assertEqual(outputs[0]["error"]["code"], "TOOL_RESPONSE_BUDGET")
        self.assertLessEqual(counter(module._json_bytes(outputs[0]).decode("utf-8")), budget)
        self.assertNotIn("private-tokenizer-secret", str(result) + str(sdk_settings))
        self.assertNotIn("count_tokens", vars(sdk_settings[0]))

    def test_invalid_tool_budget_configuration_fails_before_sdk_load(self):
        cases = [{"max_tool_tokens": 4096}, {"count_tokens": len},
                 {"max_tool_tokens": True, "count_tokens": len},
                 {"max_tool_tokens": 0, "count_tokens": len},
                 {"max_tool_tokens": 4096, "count_tokens": None},
                 {"max_tool_tokens": 1, "count_tokens": len},
                 *({"max_tool_tokens": 4096, "count_tokens": lambda text, value=value: value}
                   for value in (True, -1, 1.0, None))]
        for config in cases:
            with self.subTest(config=config), patch.object(module, "_load_tongagents") as loader:
                result = module.run_react("task", {"finish": lambda: {}}, {**self.config, **config})
                self.assertEqual(result["status"], "error")
                loader.assert_not_called()

    def test_tokenizer_failure_stops_session_without_returning_unchecked_tool_payload(self):
        calls = []
        outputs_seen = []

        def counter(text):
            calls.append(text)
            if len(calls) > 1:
                raise RuntimeError("private tokenizer endpoint/credential")
            return len(text)

        def script(registry, settings, outputs, prompt):
            outputs_seen.append(registry["read_workspace"]())
            yield TextAction()

        result, *_ = self.invoke(script, {"read_workspace": lambda: {"private-source": "not returned"}},
                                max_tool_tokens=4096, count_tokens=counter)
        self.assertEqual(result["status"], "error")
        self.assertEqual(outputs_seen, [])
        self.assertNotIn("credential", str(result))
        self.assertIsNone(module._ACTIVE_SESSION)


if __name__ == "__main__":
    unittest.main()
