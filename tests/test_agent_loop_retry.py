from __future__ import annotations

from types import SimpleNamespace

from bridge_evidence.models.agent import (
    AgentCommand,
    AgentObservation,
    AgentRunRecord,
    ToolDescriptor,
    ToolManifest,
    ToolResult,
    TrajectoryStep,
)
from bridge_evidence.runtime.graph import CoreAgentLoop


class _RepeatingPlanner:
    def plan(self, *, run_id, seq, observation, manifest, trajectory):
        return AgentCommand(run_id=run_id, seq=seq, action="review_details")


def _failed_step(seq: int) -> TrajectoryStep:
    command = AgentCommand(run_id="run:test", seq=seq, action="review_details")
    return TrajectoryStep(
        command=command,
        result=ToolResult(
            run_id="run:test",
            seq=seq,
            tool="review_details",
            success=False,
            error="Request timed out.",
        ),
        state_hash_before="before",
        state_hash_after="after",
    )


def _loop() -> CoreAgentLoop:
    loop = object.__new__(CoreAgentLoop)
    loop.planner = _RepeatingPlanner()
    return loop


def _graph_state(record: AgentRunRecord, step_count: int, pending=None, completed=None):
    manifest = ToolManifest(
        tools=[ToolDescriptor(name="review_details", description="review")]
    )
    return {
        "run_record": record,
        "step_count": step_count,
        "observation": AgentObservation(),
        "registry": SimpleNamespace(
            manifest=manifest,
            pending_actions=lambda: list(pending or []),
            completed_actions=lambda: list(completed or []),
        ),
    }


def test_exact_failed_tool_command_may_retry_once():
    loop = _loop()
    failed = _failed_step(1)
    fingerprint = loop._command_fingerprint(failed.command)
    record = AgentRunRecord(
        run_id="run:test",
        planner="deepseek",
        trajectory=[failed],
        command_fingerprints=[fingerprint],
    )

    planned = loop._plan(_graph_state(record, 1))["command"]

    assert planned.action == "review_details"


def test_second_failed_retry_gets_one_final_graph_level_attempt():
    loop = _loop()
    failures = [_failed_step(1), _failed_step(2)]
    fingerprint = loop._command_fingerprint(failures[0].command)
    record = AgentRunRecord(
        run_id="run:test",
        planner="deepseek",
        trajectory=failures,
        command_fingerprints=[fingerprint, fingerprint],
    )

    planned = loop._plan(_graph_state(record, 2))["command"]

    assert planned.action == "review_details"


def test_third_failed_retry_is_stopped_as_duplicate():
    loop = _loop()
    failures = [_failed_step(1), _failed_step(2), _failed_step(3)]
    fingerprint = loop._command_fingerprint(failures[0].command)
    record = AgentRunRecord(
        run_id="run:test",
        planner="deepseek",
        trajectory=failures,
        command_fingerprints=[fingerprint, fingerprint, fingerprint],
    )

    planned = loop._plan(_graph_state(record, 3))["command"]

    assert planned.action == "fail"
    assert planned.arguments["reason"] == "Planner 重复命令且无未完成动作: review_details"


def test_successful_tool_command_remains_non_repeatable():
    loop = _loop()
    command = AgentCommand(run_id="run:test", seq=1, action="review_details")
    step = TrajectoryStep(
        command=command,
        result=ToolResult(
            run_id="run:test",
            seq=1,
            tool="review_details",
            success=True,
        ),
        state_hash_before="before",
        state_hash_after="after",
    )
    fingerprint = loop._command_fingerprint(command)
    record = AgentRunRecord(
        run_id="run:test",
        planner="deepseek",
        trajectory=[step],
        command_fingerprints=[fingerprint],
    )

    planned = loop._plan(_graph_state(record, 1))["command"]

    assert planned.action == "finalize"
    assert "进入最终验证" in planned.rationale


def test_successful_duplicate_advances_to_pending_required_action():
    loop = _loop()
    command = AgentCommand(run_id="run:test", seq=1, action="review_details")
    step = TrajectoryStep(
        command=command,
        result=ToolResult(run_id="run:test", seq=1, tool="review_details", success=True),
        state_hash_before="before",
        state_hash_after="after",
    )
    fingerprint = loop._command_fingerprint(command)
    record = AgentRunRecord(
        run_id="run:test",
        planner="deepseek",
        trajectory=[step],
        command_fingerprints=[fingerprint],
    )

    planned = loop._plan(_graph_state(record, 1, pending=[{
        "action": "write_decision",
        "arguments": {},
        "reason": "complete decision",
    }]))["command"]

    assert planned.action == "write_decision"
    assert "调度器改执行" in planned.rationale


def test_last_action_is_reserved_for_finalize_after_requirements_pass():
    loop = _loop()
    record = AgentRunRecord(
        run_id="run:test",
        planner="deepseek",
        max_actions=3,
    )

    planned = loop._plan(_graph_state(record, 2))["command"]

    assert planned.action == "finalize"
    assert "最后一个 action" in planned.rationale
    assert len(record.command_fingerprints) == 1


def test_completed_required_actions_finalize_without_optional_search_loop():
    loop = _loop()
    record = AgentRunRecord(
        run_id="run:test",
        planner="deepseek",
        max_actions=16,
    )

    planned = loop._plan(_graph_state(
        record,
        8,
        completed=[{"action": "review_summary", "arguments": {}}],
    ))["command"]

    assert planned.action == "finalize"
    assert "强制证据动作已全部完成" in planned.rationale
