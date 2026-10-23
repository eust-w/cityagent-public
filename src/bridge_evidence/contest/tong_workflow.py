"""Four-stage TongAgents adapter; verification requires matching real acceptance evidence.

The implementation follows the official WorkflowAgent/debug documentation, not
the course demos that directly chain generators. Importing this module does not
import TongAgents or load a model. A missing/incompatible SDK never falls back to
``direct``. Render callbacks MUST write a private temporary artifact: callers may
publish it only after this function returns and native trace checks have passed.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

STAGES = ("prepare", "infer", "validate", "render")
ADAPTER_VERSION = "2-version-bound-acceptance"


class WorkflowExecutionError(RuntimeError):
    """Execution failed; ``workflow`` contains the evidence collected so far."""

    def __init__(self, message: str, workflow: dict):
        super().__init__(message)
        self.workflow = workflow


def _load_tongagents():
    # No substitute package, compatibility shim, or mock is used at runtime.
    agent = importlib.import_module("tongagents.base_agent").WorkflowAgent
    event = importlib.import_module("tongagents.entity").Event
    declare = importlib.import_module("tongagents.workflow.simple_workflow").node_declare
    constants = importlib.import_module("tongagents.workflow.nodes.simple_node_base")
    version = importlib.metadata.version("tongagents")
    if not version.strip():
        raise ValueError("TongAgents distribution version is empty")
    return agent, event, declare, constants.START, constants.END, version


def _native_event_projection(event: Any) -> dict:
    """Retain SDK fields, excluding document bodies and exception message text."""
    kind = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
    data = event.get("data") if isinstance(event, dict) else getattr(event, "data", None)
    if (kind not in {"node_start", "node_process", "node_end", "node_exception"}
            or not isinstance(data, dict) or not isinstance(data.get("node_name"), str)):
        raise ValueError("SDK DebugEvent contract is unverified: expected type/data.node_name")
    projected = {"node_name": data["node_name"]}
    if "timestamp" in data:
        stamp = data["timestamp"]
        if not isinstance(stamp, (str, int, float, type(None))):
            raise ValueError("SDK debug timestamp is not a supported scalar")
        projected["timestamp"] = stamp
    return {"type": kind, "data": projected}


def _validate_native_events(events: list[dict]) -> None:
    if any(event["type"] == "node_exception" for event in events):
        raise ValueError("SDK reported a native node_exception")
    actual = [(event["type"], event["data"]["node_name"]) for event in events
              if event["data"]["node_name"] in STAGES and event["type"] != "node_process"]
    expected = [(kind, stage) for stage in STAGES for kind in ("node_start", "node_end")]
    if actual != expected:
        raise ValueError("SDK native trace is unverified: require one ordered start/end pair for every stage")


def execute_workflow(
    payload: dict,
    stages: dict[str, Callable[[dict], dict]],
    *,
    engine: str = "tongagents",
    acceptance_path=None,
) -> dict:
    """Return ``{'payload': final_payload, 'workflow': execution_evidence}``.

    Each callback accepts/returns a dict, and must raise on failure. The infer
    callback may return ``model_calls`` containing ``call_id`` values; those IDs
    are linked from its stage record, never fabricated as SDK-native fields.
    The caller remains responsible for semantic/model validation and atomic
    publication. ``verification`` does not become a real-SDK certification merely
    because a trace conforms to the documented shape or unit tests pass.
    """
    if engine not in {"tongagents", "direct"}:
        raise ValueError("engine must be 'tongagents' or 'direct'")
    if not isinstance(payload, dict):
        raise TypeError("workflow payload must be a dict")
    if not isinstance(stages, dict) or set(stages) != set(STAGES) or not all(callable(stages[s]) for s in STAGES):
        raise ValueError("stages must contain exactly prepare/infer/validate/render callables")

    evidence = {
        "engine": engine, "adapter_version": ADAPTER_VERSION, "execution_id": uuid4().hex,
        "sdk_version": None, "status": "running", "non_sdk": engine == "direct",
        "verification": "unverified" if engine == "tongagents" else "non-sdk-baseline",
        "native_trace_validated": False, "native_events": [], "stage_records": [],
        "native_event_projection": "type,data.node_name,data.timestamp; document/error text omitted",
    }
    completed: list[str] = []
    stage_failed = False

    def call_stage(name: str, value: dict) -> dict:
        nonlocal stage_failed
        if stage_failed or tuple(completed) != STAGES[:STAGES.index(name)]:
            stage_failed = True
            raise RuntimeError("Workflow stage order/repetition check failed")
        record = {"stage": name, "status": "running"}
        evidence["stage_records"].append(record)
        started = time.perf_counter()
        try:
            if not isinstance(value, dict):
                raise TypeError("stage input must be a dict")
            result = stages[name](value)
            if not isinstance(result, dict):
                raise TypeError("stage output must be a dict")
            if name == "infer":
                calls = result.get("model_calls", [])
                record["model_call_ids"] = [call["call_id"] for call in calls
                    if isinstance(call, dict) and isinstance(call.get("call_id"), str) and call["call_id"]]
                record["accepted_model_call_ids"] = [call["call_id"] for call in calls
                    if isinstance(call, dict) and isinstance(call.get("call_id"), str)
                    and call["call_id"] and call.get("status") == "success"]
            record["status"] = "success"
            completed.append(name)
            return result
        except Exception as error:
            stage_failed = True
            record.update(status="error", error_type=type(error).__name__)
            raise
        finally:
            record["duration_seconds"] = round(time.perf_counter() - started, 6)

    try:
        if engine == "direct":
            result = dict(payload)
            for stage in STAGES:
                result = call_stage(stage, result)
        else:
            try:
                Agent, Event, node_declare, START, END, version = _load_tongagents()
            except Exception as error:
                raise RuntimeError("真实 TongAgents SDK 不可用或接口未验证；不会回退到 direct") from error
            evidence["sdk_version"] = version
            execution_id = evidence["execution_id"]

            def unpack(event, previous: str) -> dict:
                getter = event.get if isinstance(event, dict) else lambda key: getattr(event, key, None)
                if getter("cityagent_execution_id") != execution_id or getter("cityagent_stage") != previous:
                    raise ValueError("SDK event envelope is unverified or belongs to a different execution")
                value = getter("payload")
                if not isinstance(value, dict):
                    raise TypeError("SDK event payload must be a dict")
                return value

            def invoke(name: str, event):
                nonlocal stage_failed
                index = STAGES.index(name)
                previous = STAGES[index - 1] if index else "input"
                try:
                    value = call_stage(name, unpack(event, previous))
                    return Event(payload=value, cityagent_execution_id=execution_id, cityagent_stage=name)
                except Exception:
                    # Poison the chain even if the SDK swallows an envelope or
                    # event-construction error outside the callback itself.
                    stage_failed = True
                    raise

            class DocumentWorkflow(Agent):
                @node_declare(name="prepare", edges=[(START, "prepare"), ("prepare", "infer")])
                def prepare(self, event):
                    return invoke("prepare", event)

                @node_declare(name="infer", edges=[("infer", "validate")])
                def infer(self, event):
                    return invoke("infer", event)

                @node_declare(name="validate", edges=[("validate", "render")])
                def validate(self, event):
                    return invoke("validate", event)

                @node_declare(name="render", edges=[("render", END)])
                def render(self, event):
                    return invoke("render", event)

            workflow = DocumentWorkflow()
            initial = Event(payload=dict(payload), cityagent_execution_id=execution_id, cityagent_stage="input")
            streams = workflow.run(iter([initial]), debug=True)
            if not isinstance(streams, tuple) or len(streams) != 2:
                raise ValueError("SDK debug execution is unverified: expected result/debug iterators")
            responses, native_debug = streams
            outputs = []
            response_error = None
            try:
                for response in responses:
                    outputs.append(response)
            except Exception as error:
                response_error = error
            finally:
                # Per the official synchronous example, consume responses first,
                # then debug; even a failed response stream may have native errors.
                for event in native_debug:
                    evidence["native_events"].append(_native_event_projection(event))
            if response_error is not None:
                raise response_error
            if tuple(completed) != STAGES or len(outputs) != 1:
                raise ValueError("SDK execution is unverified: incomplete stages or unexpected result count")
            _validate_native_events(evidence["native_events"])
            result = unpack(outputs[0], "render")
            evidence["native_trace_validated"] = True
            if acceptance_path is not None:
                from .sdk_acceptance import validate_acceptance
                proof = validate_acceptance(acceptance_path)
                from .run_state import json_digest
                evidence["verification"] = "verified"
                evidence["acceptance_sha256"] = json_digest(proof)
                from pathlib import Path

                from .run_state import file_digest
                evidence["acceptance_file_sha256"] = file_digest(Path(acceptance_path))
                evidence["acceptance_runtime"] = proof["runtime"]
        evidence["status"] = "success"
        return {"payload": result, "workflow": evidence}
    except Exception as error:
        evidence.update(status="error", error_type=type(error).__name__)
        raise WorkflowExecutionError(str(error), evidence) from error
