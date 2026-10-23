"""Adapter-owned strict JSON failures survive SDK generic wrapping; all offline."""
from copy import deepcopy
from pathlib import Path
import json
import socket
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest import tong_react_adapter as adapter
from bridge_evidence.contest import document_context_compression as native
from test_section_agent_adapter import Tool, Parameters, basic, fake_sdk, TextAction
from test_partial_native_batch_recovery import Memory, ToolResultMessage, call


class ToolErrorAction:
    def __init__(self, message, tool_name="save_entries", tool_call_id="bad"):
        self.message, self.tool_name, self.tool_call_id = message, tool_name, tool_call_id

    def model_dump(self, **kwargs):
        return {"message": self.message, "tool_name": self.tool_name, "tool_call_id": self.tool_call_id}


class StrictArgumentParseRecoveryTests(unittest.TestCase):
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
        self.api = {"messages": SimpleNamespace(ToolResultMessage=ToolResultMessage),
                    "memory": SimpleNamespace(MessageRole=SimpleNamespace(TOOL="tool"))}
        self.assertIsNone(adapter._ACTIVE_SESSION)

    def installed(self):
        item = type("SyntheticSaveEntries", (Tool,), {})()
        adapter._install_tool_schema(item, "save_entries", adapter._section_tool_schema("save_entries"))
        return item

    def rejected(self, raw='{"entries": '):
        session = {"handlers": {"save_entries": lambda **kwargs: self.fail("No business handler")},
                   "argument_parse_errors": [], "current_agent_call_id": lambda: "generation-one"}
        with patch.object(adapter, "_ACTIVE_SESSION", session):
            with self.assertRaises(adapter._StrictToolArgumentJSONError) as caught:
                self.installed()._normalize_and_validate_params(raw, strict_json=False)
            event = ToolErrorAction(str(caught.exception))
            error = adapter._sdk_tool_error(event, event.model_dump(), {"save_entries"})
        error["agent_call_id"] = "generation-one"
        return session, event, error

    def paired(self, raw, session, event, error):
        memory = Memory([call("save_entries", raw, "bad"), call("save_entries", '{"entries":[]}', "unexecuted")])
        before = [c.arguments for c in memory.messages[0].chunks[0].content.tool_call_messages]
        with patch.object(native, "load_native_api", return_value=self.api):
            receipt, feedback = adapter._paired_parse_feedback(SimpleNamespace(), event, error, [],
                context=SimpleNamespace(memory=memory), tool_schema=adapter._section_tool_schema("save_entries"),
                argument_parse_errors=session["argument_parse_errors"])
        self.assertEqual([c.arguments for c in memory.messages[0].chunks[0].content.tool_call_messages], before)
        calls, answers = native.native_tool_pairs(memory.get_messages())
        self.assertEqual(set(calls), set(answers))
        self.assertEqual(receipt["batch_tool_count"], 2)
        self.assertFalse(memory.messages[1].chunks[0].content.content["error"]["handler_executed"])
        return receipt, feedback

    def test_truncated_text_gets_owned_receipt_exact_position_and_paired_generic_recovery(self):
        raw = '{"entries": '
        session, event, error = self.rejected(raw)
        self.assertEqual(error["code"], "SDK_TOOL_ARGUMENT_JSON")
        record = session["argument_parse_errors"][0]
        self.assertEqual(record["diagnostic"], {"path": "$", "reason": "invalid_json_syntax",
                                               "character_offset": len(raw), "line": 1, "column": len(raw)+1})
        self.assertEqual(record["arguments_text_sha256"], adapter._hash(raw))
        self.assertEqual(error["adapter_parse_error_sha256"], adapter._hash(record))
        receipt, _ = self.paired(raw, session, event, error)
        self.assertEqual(receipt["argument_repair"]["issues"], [record["diagnostic"]])
        self.assertNotIn(raw, json.dumps(record))

    def test_duplicate_keys_nonfinite_and_nonobjects_remain_strict_but_recoverable(self):
        for raw in ('{"entries":[],"entries":[]}', '{"entries":[NaN]}', '{"entries":[1e1000]}',
                    "[]", "null", "{'entries':[]}"):
            with self.subTest(raw=raw):
                state, event, error = self.rejected(raw)
                self.assertEqual(error["code"], "SDK_TOOL_ARGUMENT_JSON")
                self.paired(raw, state, event, error)

    def test_marker_requires_unique_current_owned_receipt_and_not_just_matching_words(self):
        session, event, error = self.rejected()
        for change in ("no_receipt", "other_origin", "other_tool", "other_message", "wrong_text", "duplicate_receipt"):
            with self.subTest(change=change):
                state, dumped = deepcopy(session), event.model_dump()
                if change == "no_receipt": state["argument_parse_errors"] = []
                if change == "other_origin": state["current_agent_call_id"] = lambda: "generation-two"
                if change == "other_tool": dumped["tool_name"] = "complete_section"
                if change == "other_message": dumped["message"] = "Business error " + event.message
                if change == "wrong_text": dumped["arguments"] = "{}"
                if change == "duplicate_receipt": state["argument_parse_errors"] *= 2
                with patch.object(adapter, "_ACTIVE_SESSION", state):
                    observed = adapter._sdk_tool_error(event, dumped, {"save_entries", "complete_section"})
                self.assertEqual(observed["code"], "SDK_TOOL_ERROR")

    def test_native_pair_rejects_changed_call_raw_origin_or_receipt_without_writes(self):
        raw = '{"entries": '
        session, event, error = self.rejected(raw)
        for fault in ("raw", "origin", "missing_receipt", "event_id", "message", "schema"):
            state, sdk_error, current = deepcopy(session), deepcopy(error), ToolErrorAction(event.message)
            memory = Memory([call("save_entries", raw if fault != "raw" else '{"entries":', "bad")])
            schema = adapter._section_tool_schema("save_entries")
            if fault == "origin": sdk_error["agent_call_id"] = "other"
            if fault == "missing_receipt": state["argument_parse_errors"] = []
            if fault == "event_id": current.tool_call_id = "other"
            if fault == "message": current.message = "Business error"
            if fault == "schema": schema = {"type": "object", "properties": {}}
            with self.subTest(fault=fault), patch.object(native, "load_native_api", return_value=self.api):
                with self.assertRaises(adapter.AgentCallAssociationError):
                    adapter._paired_parse_feedback(SimpleNamespace(), current, sdk_error, [],
                        context=SimpleNamespace(memory=memory), tool_schema=schema,
                        argument_parse_errors=state["argument_parse_errors"])
            self.assertEqual(len(memory.messages), 1)

    def test_business_value_error_and_similar_message_are_never_json_rejections(self):
        for message in ("Expecting value: line 1 column 13 (char 12)", "Duplicate JSON key",
                        adapter._STRICT_ARGUMENT_JSON_MARKER + "0" * 64, "Business value is invalid"):
            with self.subTest(message=message):
                event = ToolErrorAction(message)
                state = {"handlers": {"save_entries": lambda **kwargs: None},
                         "argument_parse_errors": [], "current_agent_call_id": lambda: "g"}
                with patch.object(adapter, "_ACTIVE_SESSION", state):
                    self.assertEqual(adapter._sdk_tool_error(event, event.model_dump(), {"save_entries"})["code"],
                                     "SDK_TOOL_ERROR")

    def test_full_fake_sdk_route_truncation_generic_event_same_agent_then_valid_save(self):
        raw = '{"entries": '
        memory = Memory([call("save_entries", raw, "bad")])
        state = {"steps": 0, "origin": "generation-one"}
        saved = []
        def script(registry, settings, outputs, prompt):
            state["steps"] += 1
            if state["steps"] == 1:
                try:
                    registry["save_entries"]._normalize_and_validate_params(raw, strict_json=False)
                except adapter._StrictToolArgumentJSONError as exc:
                    yield ToolErrorAction(str(exc))
                else:
                    self.fail("Truncated JSON must be rejected before handler")
            else:
                self.assertEqual(state["steps"], 2)
                calls, answers = native.native_tool_pairs(memory.get_messages())
                self.assertEqual(set(calls), set(answers))
                state["origin"] = "generation-two"
                args = registry["save_entries"]._normalize_and_validate_params(json.dumps({"entries": [basic()]}))
                outputs.append(registry["save_entries"](**args))
                yield TextAction()
        sdk, _, _, contexts, _ = fake_sdk(script)
        with patch.object(adapter, "_load_tongagents", return_value=sdk), \
                patch.object(native, "load_native_api", return_value=self.api):
            outcome = adapter.run_react("synthetic task", {"save_entries": lambda entries: saved.extend(entries) or {"saved": True}},
                {"profile": "section", "url": "http://127.0.0.1:8000/v1", "model": "own",
                 "current_agent_call_id": lambda: state["origin"], "initial_memory_factory": lambda: memory,
                 "same_agent_parse_recovery": True, "before_same_agent_continue": lambda: None})
        self.assertEqual(outcome["status"], "completed", outcome)
        self.assertEqual(outcome["step_invocations"], 2)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(outcome["parse_recovery"]), 1)
        self.assertEqual(len(outcome["argument_parse_errors"]), 1)
        self.assertEqual(outcome["sdk_tool_errors"][0]["code"], "SDK_TOOL_ARGUMENT_JSON")
        self.assertEqual(memory.messages[0].chunks[0].content.tool_call_messages[0].arguments, raw)

if __name__ == "__main__":
    unittest.main()
