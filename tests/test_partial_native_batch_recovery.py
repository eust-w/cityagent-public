"""Real dispatcher returns with synthetic SDK memory; no SDK/model execution."""
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


class ToolResultMessage:
    def __init__(self, *, name, content, tool_call_id):
        self.name, self.content, self.tool_call_id = name, deepcopy(content), tool_call_id


class Memory:
    def __init__(self, calls):
        self.messages = [SimpleNamespace(chunks=[SimpleNamespace(content=SimpleNamespace(tool_call_messages=calls))])]

    def get_messages(self):
        return self.messages[:]

    def add_chunk(self, content, role):
        self.messages.append(SimpleNamespace(chunks=[SimpleNamespace(content=content)]))


class ToolErrorAction:
    def __init__(self, tool_name, tool_call_id):
        self.tool_name, self.tool_call_id = tool_name, tool_call_id

    def model_dump(self, **kwargs):
        return {"tool_name": self.tool_name, "tool_call_id": self.tool_call_id,
                "message": "Parameters must be formatted as a valid JSON!"}


class TextAction:
    def model_dump(self, **kwargs):
        return {"is_final_response": True, "content": "synthetic section completed"}


def call(name, arguments, key):
    return SimpleNamespace(tool_name=name, arguments=arguments, tool_call_id=key)


def search_arguments(query):
    # _invoke_active is below the public dispatcher. Its receipt must contain
    # the same explicit defaults that _search_in_document passes to it.
    return {"query": query, "scope": "default", "section_id": None,
            "max_matches": 8, "restart": False}


def session(handler=None):
    return {"handlers": {"search_in_document": handler or (lambda query, **navigation: {"matches": [], "total_matches": 0})},
            "tool_calls": [], "tool_response_snapshots": {}, "max_calls": 100,
            "profile": "section", "tool_lock": __import__("threading").RLock(),
            "current_agent_call_id": lambda: "generation-one"}


class PartialNativeBatchRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.guards = [patch.object(socket, "create_connection", side_effect=AssertionError("No network")),
                       patch.object(socket.socket, "connect", side_effect=AssertionError("No network")),
                       patch.object(subprocess, "Popen", side_effect=AssertionError("No subprocess"))]
        for guard in self.guards:
            guard.start()
            self.addCleanup(guard.stop)
        self.api = {"messages": SimpleNamespace(ToolResultMessage=ToolResultMessage),
                    "memory": SimpleNamespace(MessageRole=SimpleNamespace(TOOL="tool"))}

    def fixture(self):
        calls = [call("search_in_document", '{"query":["synthetic query"]}', "valid-first"),
                 call("search_in_document", '{"query":', "bad-second")]
        memory = Memory(calls)
        state = session()
        response = adapter._invoke_active(state, "search_in_document", search_arguments(["synthetic query"]))
        error = {"code": "SDK_TOOL_ARGUMENT_JSON", "tool_name": "search_in_document",
                 "agent_call_id": "generation-one", "tool_call_id_sha256": adapter._hash("bad-second"),
                 "action": "Regenerate valid JSON without assuming the failed call was saved."}
        return calls, memory, state, response, error

    def repair(self, memory, state, error, snapshots=True):
        event = ToolErrorAction(error["tool_name"], "bad-second")
        with patch.object(native, "load_native_api", return_value=self.api):
            return adapter._paired_parse_feedback(SimpleNamespace(), event, error, state["tool_calls"],
                context=SimpleNamespace(memory=memory),
                actual_responses=state["tool_response_snapshots"] if snapshots else None)

    def refused(self, memory, state, error, reason, snapshots=True):
        before = len(memory.messages)
        with self.assertRaises(adapter.NativeBatchRecoveryError) as caught:
            self.repair(memory, state, error, snapshots)
        self.assertEqual(caught.exception.proof_reason, reason)
        self.assertEqual(len(memory.messages), before, "Full proof must precede every memory write")

    def test_executed_return_is_restored_exactly_without_handler_reexecution(self):
        calls, memory, state, response, error = self.fixture()
        state["handlers"]["search_in_document"] = lambda **kwargs: self.fail("Never reexecute a handler")
        original_arguments = [entry.arguments for entry in calls]
        receipt, _ = self.repair(memory, state, error)
        paired_calls, answers = native.native_tool_pairs(memory.get_messages())
        self.assertEqual(set(answers), set(paired_calls))
        self.assertEqual(memory.messages[1].chunks[0].content.content, response)
        self.assertEqual([entry.arguments for entry in calls], original_arguments)
        self.assertEqual(receipt["restored_actual_tool_result_count"], 1)
        self.assertFalse(receipt["restored_actual_tool_results"][0]["handler_reexecuted"])
        failed = memory.messages[2].chunks[0].content.content
        self.assertFalse(failed["error"]["handler_executed"])
        self.assertFalse(failed["error"]["saved"])
        self.assertEqual(len(state["tool_calls"]), 1)

    def test_snapshot_is_private_after_budget_and_observer_cannot_mutate_return(self):
        state = session(lambda query, **navigation: {"text": "synthetic secret"})
        state.update(count_tokens=lambda value: 100 if "text" in json.dumps(value) else 1,
                     max_tool_tokens=2, on_tool_result=lambda name, args, response: response.update(extra="mutated"))
        response = adapter._invoke_active(state, "search_in_document", search_arguments(["synthetic"]))
        snapshot = state["tool_response_snapshots"][0]
        self.assertNotIn("extra", snapshot["response"])
        self.assertEqual(adapter._hash(snapshot["response"]), state["tool_calls"][0]["result_sha256"])
        self.assertEqual(snapshot["response"]["error"]["code"], "TOOL_RESPONSE_BUDGET")
        self.assertNotIn("synthetic secret", json.dumps(state["tool_calls"]))
        self.assertEqual(response, snapshot["response"])

    def test_observer_failure_does_not_register_a_recoverable_return(self):
        state = session()
        state["on_tool_result"] = lambda *args: (_ for _ in ()).throw(RuntimeError("synthetic observer failure"))
        with self.assertRaises(RuntimeError):
            adapter._invoke_active(state, "search_in_document", search_arguments(["synthetic"]))
        self.assertEqual(len(state["tool_calls"]), 1)
        self.assertEqual(state["tool_response_snapshots"], {})

    def test_already_persisted_native_result_needs_no_private_restore(self):
        calls, memory, state, response, error = self.fixture()
        memory.add_chunk(ToolResultMessage(name=calls[0].tool_name, content=response, tool_call_id=calls[0].tool_call_id), "tool")
        receipt, _ = self.repair(memory, state, error, snapshots=False)
        self.assertEqual(receipt["restored_actual_tool_result_count"], 0)
        self.assertEqual(len(memory.messages), 3)

    def test_historical_lossy_summary_cannot_restore_missing_return(self):
        _, memory, state, _, error = self.fixture()
        self.refused(memory, state, error, "actual_return_snapshot_missing", snapshots=False)

    def test_snapshot_binding_mismatch_refuses_before_writes(self):
        for field in ("agent_call_id", "name", "args_sha256", "result_sha256", "status"):
            with self.subTest(field=field):
                _, memory, state, _, error = self.fixture()
                state["tool_response_snapshots"][0][field] = "wrong"
                self.refused(memory, state, error, "actual_return_identity_mismatch")

    def test_mutated_actual_return_refuses_before_writes(self):
        _, memory, state, _, error = self.fixture()
        state["tool_response_snapshots"][0]["response"]["result"]["total_matches"] = 1
        self.refused(memory, state, error, "actual_return_hash_mismatch")

    def test_failed_or_unknown_return_cannot_be_promoted_to_success(self):
        for status, response in [("failed", {"ok": False, "tool": "search_in_document", "error": {"code": "UNKNOWN"}}),
                                 ("unknown", {"ok": True, "tool": "search_in_document", "result": {}})]:
            with self.subTest(status=status):
                _, memory, state, _, error = self.fixture()
                row, snapshot = state["tool_calls"][0], state["tool_response_snapshots"][0]
                row.update(status=status, result_sha256=adapter._hash(response))
                snapshot.update(status=status, result_sha256=row["result_sha256"], response=response)
                self.refused(memory, state, error, "actual_return_execution_not_confirmed")

    def test_duplicate_same_name_same_arguments_is_ambiguous(self):
        _, memory, state, _, error = self.fixture()
        batch = memory.messages[0].chunks[0].content.tool_call_messages
        batch.insert(1, call("search_in_document", batch[0].arguments, "duplicate-first"))
        self.refused(memory, state, error, "handler_identity_ambiguous")

    def test_late_ambiguity_does_not_half_restore_earlier_unique_result(self):
        _, memory, state, _, error = self.fixture()
        batch = memory.messages[0].chunks[0].content.tool_call_messages
        batch[:0] = [call("search_in_document", '{"query":["other"]}', "other-first")]
        adapter._invoke_active(state, "search_in_document", search_arguments(["other"]))
        batch.insert(2, call("search_in_document", batch[1].arguments, "duplicate-valid"))
        self.refused(memory, state, error, "handler_identity_ambiguous")

    def test_wrong_generation_receipt_cannot_restore_missing_prefix(self):
        _, memory, state, _, error = self.fixture()
        error["agent_call_id"] = "generation-two"
        self.refused(memory, state, error, "pre_rejection_execution_not_fully_proved")

    def test_missing_handler_receipt_and_other_pending_batch_refuse(self):
        _, memory, state, _, error = self.fixture()
        state["tool_calls"] = []
        self.refused(memory, state, error, "pre_rejection_execution_not_fully_proved")
        _, memory, state, _, error = self.fixture()
        memory.messages.insert(0, SimpleNamespace(chunks=[SimpleNamespace(content=SimpleNamespace(
            tool_call_messages=[call("read_outline", "{}", "older-pending")]))]))
        self.refused(memory, state, error, "pending_batch_not_unique")

    def test_executed_sibling_after_observed_rejection_refuses(self):
        _, memory, state, _, error = self.fixture()
        memory.messages[0].chunks[0].content.tool_call_messages.append(call("search_in_document", '{"query":["later"]}', "later"))
        adapter._invoke_active(state, "search_in_document", search_arguments(["later"]))
        self.refused(memory, state, error, "handler_execution_after_rejection_or_duplicate")

    def test_existing_native_result_with_wrong_hash_refuses(self):
        calls, memory, state, response, error = self.fixture()
        response["result"]["total_matches"] = 42
        memory.add_chunk(ToolResultMessage(name=calls[0].tool_name, content=response, tool_call_id=calls[0].tool_call_id), "tool")
        self.refused(memory, state, error, "native_result_hash_mismatch")

    def test_unexecuted_later_sibling_gets_only_explicit_aborted_result(self):
        _, memory, state, _, error = self.fixture()
        memory.messages[0].chunks[0].content.tool_call_messages.append(call("read_outline", "{}", "later"))
        receipt, _ = self.repair(memory, state, error)
        self.assertEqual(receipt["preserved_executed_tool_count"], 1)
        aborted = memory.messages[-1].chunks[0].content.content["error"]
        self.assertEqual(aborted["code"], "SDK_TOOL_BATCH_ABORTED")
        self.assertFalse(aborted["observed_sdk_rejection"])
        self.assertFalse(aborted["handler_executed"])

    def test_same_agent_second_step_receives_real_result_and_failure_pair(self):
        # Reuse the public-schema synthetic fixture, not an installed SDK.
        from test_section_agent_adapter import fake_sdk, Tool, Parameters
        calls, memory, _, _, _ = self.fixture()
        steps, handler_calls, delivered_prompts = [], [], []
        def script(registry, settings, outputs, prompt):
            delivered_prompts.append(prompt)
            steps.append(1)
            if len(steps) == 1:
                registry["search_in_document"](["synthetic query"])
                yield ToolErrorAction("search_in_document", "bad-second")
            else:
                native_calls, answers = native.native_tool_pairs(memory.get_messages())
                self.assertEqual(set(native_calls), set(answers))
                self.assertEqual(memory.messages[1].chunks[0].content.content["result"]["text"], "private synthetic source")
                yield TextAction()
        sdk, _, _, contexts, _ = fake_sdk(script)
        with patch.object(adapter, "_load_tongagents", return_value=sdk), \
                patch.object(adapter, "_load_tool_schema_api", return_value=(Tool, Parameters)), \
                patch.object(native, "load_native_api", return_value=self.api):
            result = adapter.run_react("synthetic section task", {"search_in_document":
                lambda query, **navigation: handler_calls.append(query) or {"text": "private synthetic source"}},
                {"profile": "section", "url": "http://127.0.0.1:8000/v1", "model": "own-model",
                 "same_agent_parse_recovery": True, "initial_memory_factory": lambda: memory,
                 "current_agent_call_id": lambda: "generation-one"})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["step_invocations"], 2)
        self.assertEqual(handler_calls, [["synthetic query"]])
        self.assertEqual(len(contexts), 1)
        self.assertIs(contexts[0].memory, memory)
        self.assertEqual(result["parse_recovery"][0]["restored_actual_tool_result_count"], 1)
        self.assertNotIn("private synthetic source", json.dumps(result))
        self.assertNotEqual(delivered_prompts[0], delivered_prompts[1])

    def test_recovery_diagnostic_distinguishes_proof_failure_from_callback_identity(self):
        from test_section_agent_adapter import fake_sdk, Tool, Parameters
        calls, memory, _, _, _ = self.fixture()
        memory.messages[0].chunks[0].content.tool_call_messages.insert(1,
            call("search_in_document", calls[0].arguments, "duplicate-valid"))
        def script(registry, settings, outputs, prompt):
            registry["search_in_document"](["synthetic query"])
            yield ToolErrorAction("search_in_document", "bad-second")
        sdk, *_ = fake_sdk(script)
        with patch.object(adapter, "_load_tongagents", return_value=sdk), \
                patch.object(adapter, "_load_tool_schema_api", return_value=(Tool, Parameters)), \
                patch.object(native, "load_native_api", return_value=self.api):
            result = adapter.run_react("synthetic section task", {"search_in_document": lambda query, **navigation: {}},
                {"profile": "section", "url": "http://127.0.0.1:8000/v1", "model": "own-model",
                 "same_agent_parse_recovery": True, "initial_memory_factory": lambda: memory,
                 "current_agent_call_id": lambda: "generation-one"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["failure_reason"], "agent_call_id_invalid")  # Compatibility.
        self.assertEqual(result["recovery_proof_failure_reason"], "handler_identity_ambiguous")
        self.assertEqual(result["step_invocations"], 1)
        self.assertEqual(len(memory.messages), 1)


if __name__ == "__main__":
    unittest.main()
