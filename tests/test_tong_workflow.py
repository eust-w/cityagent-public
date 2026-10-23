"""Adapter contract tests with a fake SDK; NOT real TongAgents integration proof."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest import tong_workflow as module


def fake_sdk(*, missing_trace=False, native_exception=False, reverse_trace=False,
             wrong_envelope=False, swallow_stage_error=False, debug_supported=True):
    """A tiny graph dispatcher testing OUR contract assumptions, not SDK APIs."""
    def declare(*, name, edges):
        def decorate(function):
            function.definition = (name, edges)
            return function
        return decorate

    class Agent:
        def run(self, events, *, debug):
            if debug is not True:
                raise AssertionError("native debug was not requested")
            native = []
            nodes, edges = {}, {}
            for value in vars(type(self)).values():
                if hasattr(value, "definition"):
                    name, links = value.definition
                    nodes[name] = value
                    edges.update(links)

            def responses():
                for event in events:
                    node = edges["START"]
                    while node != "END":
                        native.append(SimpleNamespace(type="node_start", data={"node_name": node, "in_event": "private text"}))
                        try:
                            event = nodes[node](self, event)
                        except Exception:
                            native.append(SimpleNamespace(type="node_exception", data={"node_name": node, "error_info": "private text"}))
                            if not swallow_stage_error:
                                raise
                        else:
                            native.append(SimpleNamespace(type="node_end", data={"node_name": node}))
                        node = edges[node]
                    if wrong_envelope:
                        event.cityagent_execution_id = "wrong-run"
                    yield event

            def debug_events():
                if not missing_trace:
                    yield from reversed(native) if reverse_trace else native
                if native_exception:
                    yield SimpleNamespace(type="node_exception", data={"node_name": "sdk_internal"})

            return (responses(), debug_events()) if debug_supported else responses()

    return Agent, SimpleNamespace, declare, "START", "END", "fake-test-version"


class TongWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def stage(name):
            def execute(value):
                self.calls.append(name)
                result = {**value, name: True}
                if name == "infer":
                    result["model_calls"] = [{"call_id": "test-call-1", "status": "success"}]
                return result
            return execute
        self.stages = {name: stage(name) for name in module.STAGES}

    def test_direct_is_explicit_non_sdk_and_does_not_import_sdk(self):
        with patch.object(module, "_load_tongagents", side_effect=AssertionError("must not import")):
            result = module.execute_workflow({"input": "fixture"}, self.stages, engine="direct")
        self.assertEqual(self.calls, list(module.STAGES))
        self.assertTrue(result["workflow"]["non_sdk"])
        self.assertFalse(result["workflow"]["native_trace_validated"])
        self.assertEqual(result["workflow"]["native_events"], [])

    def test_default_missing_sdk_never_falls_back_or_calls_any_stage(self):
        with patch.object(module, "_load_tongagents", side_effect=ImportError("missing")):
            with self.assertRaises(module.WorkflowExecutionError) as caught:
                module.execute_workflow({}, self.stages)
        self.assertEqual(self.calls, [])
        self.assertEqual(caught.exception.workflow["engine"], "tongagents")
        self.assertEqual(caught.exception.workflow["status"], "error")

    def test_fake_graph_drives_all_nodes_and_retains_unverified_status(self):
        with patch.object(module, "_load_tongagents", return_value=fake_sdk()):
            result = module.execute_workflow({"input": "fixture"}, self.stages)
        self.assertEqual(self.calls, list(module.STAGES))
        proof = result["workflow"]
        self.assertEqual(proof["verification"], "unverified")
        self.assertTrue(proof["native_trace_validated"])
        self.assertEqual(proof["sdk_version"], "fake-test-version")
        self.assertEqual(proof["stage_records"][1]["model_call_ids"], ["test-call-1"])
        self.assertEqual(proof["stage_records"][1]["accepted_model_call_ids"], ["test-call-1"])
        self.assertEqual(len(proof["native_events"]), 8)
        self.assertNotIn("private text", str(proof))

    def test_retry_history_is_distinct_from_accepted_model_calls(self):
        self.stages["infer"] = lambda value: {**value, "model_calls": [
            {"call_id": "failed-attempt", "status": "failed"},
            {"call_id": "successful-attempt", "status": "success"},
        ]}
        with patch.object(module, "_load_tongagents", return_value=fake_sdk()):
            result = module.execute_workflow({}, self.stages)
        inference = result["workflow"]["stage_records"][1]
        self.assertEqual(inference["model_call_ids"], ["failed-attempt", "successful-attempt"])
        self.assertEqual(inference["accepted_model_call_ids"], ["successful-attempt"])

    def test_stage_failure_prevents_later_callbacks_and_collects_native_exception(self):
        for failed in ("prepare", "infer", "validate"):
            with self.subTest(failed=failed):
                self.calls.clear()
                stages = dict(self.stages)
                def fail(value):
                    raise ValueError("stage failed")
                stages[failed] = fail
                with patch.object(module, "_load_tongagents", return_value=fake_sdk()):
                    with self.assertRaises(module.WorkflowExecutionError) as caught:
                        module.execute_workflow({}, stages)
                self.assertNotIn("render", self.calls)
                self.assertEqual(caught.exception.workflow["stage_records"][-1]["stage"], failed)
                self.assertTrue(any(e["type"] == "node_exception" for e in caught.exception.workflow["native_events"]))

    def test_sdk_swallowing_failure_cannot_continue_to_render(self):
        self.stages["infer"] = lambda value: (_ for _ in ()).throw(ValueError("failed"))
        with patch.object(module, "_load_tongagents", return_value=fake_sdk(swallow_stage_error=True)):
            with self.assertRaises(module.WorkflowExecutionError):
                module.execute_workflow({}, self.stages)
        self.assertEqual(self.calls, ["prepare"])

    def test_invalid_native_evidence_never_returns_success(self):
        for options in ({"missing_trace": True}, {"reverse_trace": True},
                        {"native_exception": True}, {"wrong_envelope": True},
                        {"debug_supported": False}):
            with self.subTest(options=options):
                with patch.object(module, "_load_tongagents", return_value=fake_sdk(**options)):
                    with self.assertRaises(module.WorkflowExecutionError) as caught:
                        module.execute_workflow({}, self.stages)
                self.assertFalse(caught.exception.workflow["native_trace_validated"])

    def test_direct_stage_failure_prevents_render(self):
        self.stages["validate"] = lambda value: None
        with self.assertRaises(module.WorkflowExecutionError):
            module.execute_workflow({}, self.stages, engine="direct")
        self.assertEqual(self.calls, ["prepare", "infer"])

    def test_reject_invalid_engine_or_stage_contract_before_execution(self):
        with self.assertRaises(ValueError):
            module.execute_workflow({}, self.stages, engine="automatic")
        with self.assertRaises(ValueError):
            module.execute_workflow({}, {"prepare": lambda value: value})
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
