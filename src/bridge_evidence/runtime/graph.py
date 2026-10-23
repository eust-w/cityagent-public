from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from bridge_evidence.agents import GatewayClient
from bridge_evidence.config import Settings
from bridge_evidence.models.agent import AgentCommand, AgentObservation, AgentRunRecord, ToolResult, TrajectoryStep
from bridge_evidence.models.schema import DocumentState
from bridge_evidence.models.validation import ConstraintEngine
from bridge_evidence.parsers.docx import ParsedDocument

from .planners import DeterministicPlanner, ModelPlanner
from .registry import EvidenceToolRegistry


class GraphState(TypedDict, total=False):
    document_state: DocumentState
    parsed: ParsedDocument
    registry: EvidenceToolRegistry
    observation: AgentObservation
    command: AgentCommand
    last_result: ToolResult | None
    run_record: AgentRunRecord
    step_count: int


def _document_state_hash(state: DocumentState) -> str:
    payload = state.model_dump(mode="json", exclude={"agent_run", "metrics", "validation"})
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class CoreAgentLoop:
    # A transient gateway failure must not be treated as a planner loop.  One
    # Two graph-level retries of the exact failed tool command are allowed;
    # the gateway already performs request-level retries, while this layer
    # covers a short outage spanning one complete tool call.  A third failed
    # retry is stopped by the duplicate-command guard and global action budget.
    # Successful commands remain strictly non-repeatable.
    max_failed_command_attempts = 3

    def __init__(
        self,
        settings: Settings,
        validator: ConstraintEngine,
        *,
        client: GatewayClient | None = None,
        prompts: dict[str, str] | None = None,
        max_actions: int = 16,
        workflow_profile: str = "full",
    ):
        self.settings = settings
        self.validator = validator
        self.client = client
        self.prompts = prompts or {}
        self.max_actions = max_actions
        self.workflow_profile = workflow_profile
        if client and settings.planner_mode == "model":
            self.planner = ModelPlanner(
                client,
                settings,
                self.prompts.get(
                    "react_system",
                    "你是桥梁定检证据 Agent 的 Planner。观察状态后选择一个工具动作；不得编造数值、证据或工具。输出严格 JSON。",
                ),
            )
        else:
            self.planner = DeterministicPlanner()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("observe", self._observe)
        graph.add_node("plan", self._plan)
        graph.add_node("act", self._act)
        graph.add_edge(START, "observe")
        graph.add_edge("observe", "plan")
        graph.add_edge("plan", "act")
        graph.add_conditional_edges("act", self._route_after_act, {"continue": "observe", "end": END})
        return graph.compile()

    def run(self, state: DocumentState, parsed: ParsedDocument) -> DocumentState:
        run_id = f"run:{uuid.uuid4()}"
        registry = EvidenceToolRegistry(
            state,
            parsed,
            self.validator,
            client=self.client,
            settings=self.settings,
            prompts=self.prompts,
            workflow_profile=self.workflow_profile,
        )
        run_record = AgentRunRecord(run_id=run_id, planner=self.planner.name, max_actions=self.max_actions)
        result = self._graph.invoke(
            {
                "document_state": state,
                "parsed": parsed,
                "registry": registry,
                "run_record": run_record,
                "last_result": None,
                "step_count": 0,
            },
            config={"recursion_limit": self.max_actions * 4 + 8},
        )
        final_state = result["document_state"]
        record = result["run_record"]
        if registry.finalized:
            record.status = "completed"
            record.final_validation_passed = final_state.validation.passed
        elif registry.failed:
            record.status = "failed"
            record.failure_reason = registry.failure_reason
        else:
            record.status = "budget_exhausted"
            record.failure_reason = "Agent 达到 action 预算且未 finalize"
        final_state.agent_run = record
        return final_state

    def _observe(self, graph_state: GraphState) -> dict[str, Any]:
        state = graph_state["document_state"]
        parsed = graph_state["parsed"]
        state.validation = self.validator.validate(state)
        all_text = parsed.full_text
        signals: list[str] = []
        if "左幅" in all_text and "右幅" in all_text:
            signals.append("dual-carriageway")
        if "Dr" in all_text and re_search_old_standard(all_text):
            signals.append("legacy-Dr-grade")
        if any(
            score_token in all_text and control_token in all_text
            for score_token, control_token in (("BCIm", "BSIm"), ("BCIs", "BSIs"), ("BCIx", "BSIx"))
        ):
            signals.append("BCI-BSI-semantic-conflict")
        disease_rows = sum(max(0, len(table.rows) - 1) for table in parsed.tables if table.is_disease_table())
        if disease_rows > 200:
            signals.append("large-disease-tables")
        if state.validation.errors:
            signals.append("constraint-failed")
        observation = AgentObservation(
            key_fields={
                "bridge_name": state.bridge.name.value,
                "report_date": state.bridge.report_date.value,
                "overall_score": state.scores.overall.score.value,
                "overall_grade": state.scores.overall.grade.value,
                "score_conflicts": state.scores.overall.score.conflicts,
                "grade_conflicts": state.scores.overall.grade.conflicts,
            },
            validation_errors=[item.model_dump(mode="json") for item in state.validation.errors],
            validation_warnings=[item.model_dump(mode="json") for item in state.validation.warnings],
            format_signals=signals,
            evidence_coverage=state.validation.evidence_coverage,
            disease_count=len(state.diseases),
            recommendation_count=len(state.recommendations),
            required_actions=graph_state["registry"].pending_actions(),
            completed_actions=graph_state["registry"].completed_actions(),
            last_result=graph_state.get("last_result"),
        )
        return {"document_state": state, "observation": observation}

    def _plan(self, graph_state: GraphState) -> dict[str, Any]:
        record = graph_state["run_record"]
        seq = graph_state.get("step_count", 0) + 1
        if seq > record.max_actions:
            command = AgentCommand(
                run_id=record.run_id,
                seq=seq,
                action="fail",
                arguments={"reason": "超过 Agent action 预算"},
                rationale="预算守门",
            )
            return {"command": command}
        registry = graph_state["registry"]
        pending_actions = registry.pending_actions()
        completed_actions = registry.completed_actions()

        # Required actions are the runtime's bounded evidence contract.  Once
        # they have all succeeded and the fresh observation passes validation,
        # another open-ended search cannot improve gate eligibility; it only
        # introduces stochastic loops and extra token cost.  Finalization is
        # still a real tool action, recorded in the trajectory and rechecked
        # by ConstraintEngine.
        if completed_actions and not pending_actions and not graph_state["observation"].validation_errors:
            command = AgentCommand(
                run_id=record.run_id,
                seq=seq,
                action="finalize",
                arguments={},
                rationale="强制证据动作已全部完成且约束验证通过，进入最终门禁",
                expected_information_gain="复核并提交完整证据状态",
            )
            record.command_fingerprints.append(self._command_fingerprint(command))
            return {"command": command, "run_record": record}
        # Keep the final action as a commit slot.  A model planner may keep
        # gathering low-value evidence even after every mandatory action and
        # hard constraint has passed.  Letting that consume the last slot
        # turns an otherwise valid run into ``budget_exhausted`` and discards
        # the Word result.  The policy layer therefore performs the existing
        # finalize gate itself at the boundary; it still cannot finalize while
        # an obligation or validation error remains.
        if (
            seq == record.max_actions
            and not pending_actions
            and not graph_state["observation"].validation_errors
        ):
            command = AgentCommand(
                run_id=record.run_id,
                seq=seq,
                action="finalize",
                arguments={},
                rationale="保留最后一个 action 给最终约束验证，防止有效运行因搜索预算耗尽被丢弃",
                expected_information_gain="提交当前证据状态到 finalize 门禁",
            )
            record.command_fingerprints.append(self._command_fingerprint(command))
            return {"command": command, "run_record": record}
        command = self.planner.plan(
            run_id=record.run_id,
            seq=seq,
            observation=graph_state["observation"],
            manifest=graph_state["registry"].manifest,
            trajectory=record.trajectory,
        )
        fingerprint = self._command_fingerprint(command)
        matching_steps = [
            step
            for step in record.trajectory
            if self._command_fingerprint(step.command) == fingerprint
        ]
        retryable_failure = (
            matching_steps
            and all(not step.result.success for step in matching_steps)
            and len(matching_steps) < self.max_failed_command_attempts
        )
        if fingerprint in record.command_fingerprints and not retryable_failure:
            pending = graph_state["registry"].pending_actions()
            fallback = next(
                (
                    item for item in pending
                    if hashlib.sha256(json.dumps(
                        {"action": item["action"], "arguments": item.get("arguments") or {}},
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")).hexdigest() not in record.command_fingerprints
                ),
                None,
            )
            if fallback:
                command = AgentCommand(
                    run_id=record.run_id,
                    seq=seq,
                    action=fallback["action"],
                    arguments=fallback.get("arguments") or {},
                    rationale=f"Planner 重复 {command.action}；调度器改执行尚未完成的必需动作",
                    expected_information_gain=fallback.get("reason", "完成剩余证据动作"),
                )
                fingerprint = self._command_fingerprint(command)
            elif any(step.result.success for step in matching_steps):
                command = AgentCommand(
                    run_id=record.run_id,
                    seq=seq,
                    action="finalize",
                    arguments={},
                    rationale=f"Planner 重复已完成动作 {command.action} 且无剩余必需动作；进入最终验证",
                    expected_information_gain="验证证据覆盖、在线调用审计与输出约束",
                )
                fingerprint = self._command_fingerprint(command)
            else:
                command = AgentCommand(
                    run_id=record.run_id,
                    seq=seq,
                    action="fail",
                    arguments={"reason": f"Planner 重复命令且无未完成动作: {command.action}"},
                    rationale="重复命令守门",
                )
                fingerprint = hashlib.sha256(f"fail:{seq}".encode()).hexdigest()
        record.command_fingerprints.append(fingerprint)
        return {"command": command, "run_record": record}

    @staticmethod
    def _command_fingerprint(command: AgentCommand) -> str:
        return hashlib.sha256(
            json.dumps(
                {"action": command.action, "arguments": command.arguments},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def _act(self, graph_state: GraphState) -> dict[str, Any]:
        state = graph_state["document_state"]
        record = graph_state["run_record"]
        command = graph_state["command"]
        before = _document_state_hash(state)
        result = graph_state["registry"].execute(command)
        after = _document_state_hash(state)
        record.trajectory.append(TrajectoryStep(command=command, result=result, state_hash_before=before, state_hash_after=after))
        return {
            "document_state": state,
            "last_result": result,
            "run_record": record,
            "step_count": graph_state.get("step_count", 0) + 1,
        }

    @staticmethod
    def _route_after_act(graph_state: GraphState) -> str:
        registry = graph_state["registry"]
        record = graph_state["run_record"]
        if registry.finalized or registry.failed:
            return "end"
        if graph_state.get("step_count", 0) >= record.max_actions:
            return "end"
        return "continue"


def re_search_old_standard(text: str) -> bool:
    return any(token in text for token in ("一类", "二类", "三类", "四类", "五类"))
