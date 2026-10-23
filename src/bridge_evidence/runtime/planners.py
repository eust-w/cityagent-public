from __future__ import annotations

import json

from bridge_evidence.agents import GatewayClient
from bridge_evidence.config import Settings
from bridge_evidence.models.agent import AgentAction, AgentCommand, AgentObservation, ToolManifest, TrajectoryStep
from pydantic import BaseModel, ConfigDict, Field


class PlannedCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: AgentAction
    arguments: dict = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=500)
    expected_information_gain: str = Field(default="", max_length=300)


class DeterministicPlanner:
    name = "deterministic"

    def plan(
        self,
        *,
        run_id: str,
        seq: int,
        observation: AgentObservation,
        manifest: ToolManifest,
        trajectory: list[TrajectoryStep],
    ) -> AgentCommand:
        if not trajectory:
            return AgentCommand(
                run_id=run_id,
                seq=seq,
                action="get_validation_report",
                rationale="先读取硬约束结果，建立控制反馈",
                expected_information_gain="确认是否可以安全结束",
            )
        if observation.last_result and not observation.last_result.success:
            return AgentCommand(
                run_id=run_id,
                seq=seq,
                action="fail",
                arguments={"reason": f"工具失败且离线 Planner 无法恢复：{observation.last_result.error}"},
                rationale="离线策略不猜测失败工具的替代证据",
            )
        if observation.required_actions:
            required = observation.required_actions[0]
            return AgentCommand(
                run_id=run_id,
                seq=seq,
                action=required["action"],
                arguments=required.get("arguments") or {},
                rationale=f"完成复杂格式探索义务：{required.get('reason') or required['action']}",
                expected_information_gain="核对复杂格式候选并解除 finalize 门禁",
            )
        if not observation.validation_errors:
            return AgentCommand(
                run_id=run_id,
                seq=seq,
                action="finalize",
                rationale="关键约束已通过",
                expected_information_gain="固化已验证状态",
            )

        missing_fields = {str(item.get("field") or "") for item in observation.validation_errors}
        score_scope = None
        for field, scope in (
            ("scores.overall.score", "overall"),
            ("scores.overall.grade", "overall"),
            ("scores.previous.score", "previous"),
        ):
            if field in missing_fields:
                score_scope = scope
                break
        already_listed = any(step.command.action == "list_score_candidates" and step.command.arguments.get("scope") == score_scope for step in trajectory)
        if score_scope and not already_listed:
            return AgentCommand(
                run_id=run_id,
                seq=seq,
                action="list_score_candidates",
                arguments={"scope": score_scope},
                rationale="关键评分缺失，先列出有坐标的确定性候选",
                expected_information_gain="判断是否存在唯一可验证候选",
            )
        exhausted_overall = next(
            (
                step
                for step in reversed(trajectory)
                if step.command.action == "list_score_candidates"
                and step.command.arguments.get("scope") == "overall"
            ),
            None,
        )
        if score_scope == "overall" and exhausted_overall and not exhausted_overall.result.content.get("candidates"):
            return AgentCommand(
                run_id=run_id,
                seq=seq,
                action="mark_unavailable",
                arguments={"scope": "overall"},
                rationale="穷尽总体评分候选后确认原报告未提供该字段",
                expected_information_gain="以可审计的负证据解除缺失字段门禁",
            )
        return AgentCommand(
            run_id=run_id,
            seq=seq,
            action="fail",
            arguments={"reason": "确定性抽取后仍有关键约束错误，需要 Qwen Planner 消歧"},
            rationale="拒绝在无唯一证据时自动填值",
        )


class ModelPlanner:

    def __init__(self, client: GatewayClient, settings: Settings, system_prompt: str):
        self.client = client
        self.settings = settings
        self.system_prompt = system_prompt
        planner_model = settings.model_for_task("react-plan").lower()
        self.name = "deepseek" if "deepseek" in planner_model else "qwen"

    def plan(
        self,
        *,
        run_id: str,
        seq: int,
        observation: AgentObservation,
        manifest: ToolManifest,
        trajectory: list[TrajectoryStep],
    ) -> AgentCommand:
        # The online loop must remain model-planned even when the runtime has
        # mandatory semantic-review obligations.  The typed policy constrains
        # which actions are legal; it must not bypass the planner entirely.
        allowed_actions = [tool.name for tool in manifest.tools]
        history = [
            {
                "seq": step.command.seq,
                "action": step.command.action,
                "arguments": step.command.arguments,
                "success": step.result.success,
                "result": step.result.content,
                "error": step.result.error,
            }
            for step in trajectory[-6:]
        ]
        result = self.client.complete_model(
            PlannedCommand,
            system=self.system_prompt,
            user_payload={
                "task": "根据当前观察选择且只选择一个下一动作。你在有界 Agent Loop 中，不能直接编造事实。",
                "run_id": run_id,
                "seq": seq,
                "observation": observation.model_dump(mode="json"),
                "trajectory": history,
                "tools": [tool.model_dump(mode="json") for tool in manifest.tools],
                "rules": [
                    "只能使用 tools 中的 action",
                    "关键事实只能通过 apply_fact_patch 且必须引用已经由工具返回的 evidence_ids",
                    "验证通过时优先 finalize",
                    "required_actions 非空时必须先执行其中一个动作，禁止 finalize",
                    "成功的同一 action 和 arguments 不得重复",
                    "若 required_actions 中的同一动作刚因超时等瞬时错误失败，可用原参数重试一次",
                    "输出一个 JSON 对象，不输出解释文本",
                ],
                "output_schema": {
                    "action": "tools 中的一个 action 名称",
                    "arguments": {},
                    "rationale": "简短决策依据，不输出隐藏思维链",
                    "expected_information_gain": "该动作预期获得什么新信息",
                },
            },
            temperature=0.0,
            task="react-plan",
        )
        command = AgentCommand(run_id=run_id, seq=seq, **result.model_dump())
        if command.action not in allowed_actions:
            raise ValueError(f"Planner 选择了未注册工具: {command.action}")
        if observation.required_actions:
            required_keys = {
                json.dumps(
                    {"action": item["action"], "arguments": item.get("arguments") or {}},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for item in observation.required_actions
            }
            command_key = json.dumps(
                {"action": command.action, "arguments": command.arguments},
                ensure_ascii=False,
                sort_keys=True,
            )
            if command_key not in required_keys:
                # The model remains the planner, while the policy layer is the
                # safety controller: an out-of-manifest choice cannot abort a
                # long run or bypass mandatory evidence work.  Execute the
                # first declared obligation and record the override openly in
                # the trajectory rationale.
                required = observation.required_actions[0]
                command = AgentCommand(
                    run_id=run_id,
                    seq=seq,
                    action=required["action"],
                    arguments=required.get("arguments") or {},
                    rationale=(
                        f"策略门纠正 Planner 越界动作 {command.action}；"
                        f"执行强制义务 {required['action']}"
                    ),
                    expected_information_gain="完成未闭环的强制证据动作",
                )
        elif command.action == "finalize" and observation.validation_errors:
            raise ValueError("Planner 在硬约束仍失败时尝试 finalize")
        return command


# Backward-compatible import for older integrations; online planning is now
# provider-neutral and its run record reports the actual model family.
QwenPlanner = ModelPlanner
