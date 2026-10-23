from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bridge_evidence.config import Settings

from .client import GatewayClient


ReviewField = Literal[
    "bridge.name",
    "bridge.report_date",
    "scores.overall.score",
    "scores.overall.grade",
    "scores.deck.score",
    "scores.deck.grade",
    "scores.superstructure.score",
    "scores.superstructure.grade",
    "scores.substructure.score",
    "scores.substructure.grade",
    "scores.previous.score",
    "scores.previous.grade",
]


class ReviewPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: ReviewField
    value: str | float | int
    evidence_ids: list[str] = Field(min_length=1, max_length=4)
    reason: str = Field(default="", max_length=240)


class SummaryReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patches: list[ReviewPatch] = Field(default_factory=list, max_length=12)
    warnings: list[str] = Field(default_factory=list, max_length=12)


class SummaryReviewer:
    """Evidence-bounded Qwen reviewer for the 12 scored summary fields."""

    def __init__(self, client: GatewayClient, settings: Settings, system_prompt: str):
        self.client = client
        self.settings = settings
        self.system_prompt = system_prompt

    def review(
        self,
        *,
        current: dict[str, Any],
        evidence: list[dict[str, str]],
        decision_context: dict[str, Any] | None = None,
        task: str = "summary-review",
        instruction: str = "复核当前简要信息。只有当前值错误或缺失时才返回 patch；正确字段不要返回。",
    ) -> SummaryReview:
        return self.client.complete_model(
            SummaryReview,
            system=self.system_prompt,
            user_payload={
                "task": instruction,
                "field_semantics": {
                    "scores.overall.score": "本次全桥 BCI/Dr 总体评分，不是部位评分、权重或上一次评分",
                    "scores.deck.score": "竞赛标签口径的桥面系最终评分。默认采用报告摘要、审核后结论或最终部位表发布的 BCIm；中间计算段只有 BSIm 及其等级不足以替代 BCIm，除非摘要或最终结论明确把 BSI 作为该部位发布结果",
                    "scores.superstructure.score": "竞赛标签口径的上部结构最终评分。默认采用报告摘要、审核后结论或最终部位表发布的 BCIs；BSIs 是最不利构件控制值，不能仅因它与等级相邻就覆盖 BCIs",
                    "scores.substructure.score": "竞赛标签口径的下部结构最终评分。默认采用报告摘要、审核后结论或最终部位表发布的 BCIx；BSIx 是最不利墩台控制值，只有摘要或最终结论把它作为发布结果时才替代 BCIx",
                    "scores.*.grade": "采用报告最终发布且内部一致的对应等级。只有给定证据同时提供报告自己的分级区间时，才可用区间识别明显笔误；不得凭常识或外部阈值重算",
                    "scores.previous.*": "仅限当前检查之前最近一次定检；在检查维修记录卡中按日期和原表行序恢复时间链，不得随意选更早年份，也不得把本次值当历史值",
                },
                "hard_rules": [
                    "所有 patch 必须逐字受到给定 evidence_ids 支持",
                    "只能引用 evidence 列表中存在的 evidence_id",
                    "0 到 1 之间的小数通常是权重，不得作为 BCI/BCIm/BCIs/BCIx 评分",
                    "BCI 与 BSI 是不同候选，但不得机械规定谁永远优先；应按该报告最终发布口径裁决",
                    "报告摘要、审核后结论和最终汇总表优先于中间计算段；BSI 计算句与等级相邻不等于竞赛字段应改写为 BSI",
                    "如封面摘要与报告后部最终技术状况评定章或最新有日期记录冲突，优先最终章和当前检查日期记录；封面可能是未更新的旧结论",
                    "上一次评分必须是时间链中紧邻本次之前的有效记录；同一月份/区间有多条时，保留原表行序并用与本次记录相邻者，不得对多次评分取平均",
                    "报告分别发布主桥、引桥或左右幅时，保留原文的设施范围和发布口径，不得将不同范围的数值平均成一个伪整桥分数",
                    "候选的 grade_binding=true 表示该等级在同一分句中直接跟随并绑定该指标；同句更早出现但 grade_binding=false 的指标不得冒用后面的等级",
                    "若同一指标同一数值在左右幅被标成不同等级，该证据 self_contradictory=true，不能作为最终裁决依据；应转向最终部位评定表、最终结论和报告内分级区间",
                    "最终部位技术状况评定表中的部位指数不是整桥 BCI、权重或 BSI 最小构件值；必须按表头与行语义读取",
                    "若最终部位表已发布 component_index 和部位等级，默认保留该发布结果；只有报告摘要或最终审核结论明确采用 BSI 时才选择 control_index",
                    "decision_context.side_coverage_gaps 中列出的字段当前丢失了最终表明确给出的左右幅结构，属于不完整值，必须在证据支持下修复",
                    "原文分别评定左、右幅时必须保留左右幅输出，即使两个数值相同",
                    "左右幅、BCI/BSI 或评分/等级证据冲突时，比较检测结论、评定结论、最终汇总表和最终计算段的语义角色",
                    "不得补写证据中不存在的日期、数值或等级",
                ],
                "current": current,
                "decision_context": decision_context or {},
                "evidence": evidence,
                "output_schema": {
                    "patches": [
                        {
                            "field": "scores.deck.score",
                            "value": 84.93,
                            "evidence_ids": ["P290"],
                            "reason": "当前值是权重，证据明确给出 BCIm",
                        }
                    ],
                    "warnings": [],
                },
            },
            temperature=self.settings.fact_temperature,
            task=task,
        )
