from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bridge_evidence.config import Settings

from .client import GatewayClient


class DiseaseDetailPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    evidence_ids: list[str] = Field(min_length=1, max_length=6)
    location: str | None = Field(default=None, max_length=300)
    disease_type: str | None = Field(default=None, max_length=120)
    description: str | None = Field(default=None, max_length=800)
    is_new: bool | None = None
    previous_state: Literal["已发现", "未发现"] | None = None
    development: Literal["无", "轻微", "严重"] | None = None
    reason: str = Field(default="", max_length=240)


class RecommendationDetailPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    evidence_ids: list[str] = Field(min_length=1, max_length=6)
    category: Literal["立即处置", "尽快维修", "预防性养护"] | None = None
    component: str | None = Field(default=None, max_length=200)
    content: str | None = Field(default=None, max_length=800)
    reason: str = Field(default="", max_length=240)


class DetailReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disease_patches: list[DiseaseDetailPatch] = Field(default_factory=list, max_length=48)
    recommendation_patches: list[RecommendationDetailPatch] = Field(default_factory=list, max_length=24)
    warnings: list[str] = Field(default_factory=list, max_length=12)


class DetailReviewer:
    """Evidence-bounded semantic reviewer for disease and advice rows."""

    def __init__(self, client: GatewayClient, settings: Settings, system_prompt: str):
        self.client = client
        self.settings = settings
        self.system_prompt = system_prompt

    def review(
        self,
        *,
        diseases: list[dict],
        recommendations: list[dict],
        force_temporal_decision: bool = False,
    ) -> DetailReview:
        retry_instruction = (
            "这是遗漏行的二元时态裁决重试。diseases 中每一行都必须返回一个 patch，"
            "is_new 必须为 true 或 false，不能省略该行或返回 null；evidence_ids 必须至少引用一个"
            "明确含上一次/本次对比、历史、既有或新增语义的证据片段，不能只引用当前照片行；"
            "证据不足的保留意见写入 reason。"
            if force_temporal_decision
            else ""
        )
        user_payload = {
                "task": (
                    "只审校证据明确支持的病害明细和建议字段。正确字段不要返回 patch；"
                    "不得改变行号、合并或新增行，不得创造证据中不存在的数值、部位或病害。"
                    "对于 current.is_new=null 且输入含上一次/本次对比证据的病害，必须逐行返回"
                    "is_new=true 或 false 的 patch，不得继续返回 null；不确定性写入 reason。"
                    + retry_instruction
                ),
                "disease_field_semantics": {
                    "location": "构件、桥跨、梁号、方向、幅别等精确部位；不能只写宽泛构件名",
                    "disease_type": "裂缝、破损、露筋、渗水泛碱、锈蚀等病害类型",
                    "description": "保留原文数量、尺寸、单位、方向、照片编号和范围",
                    "is_new": "只有当前或历史证据明确支持新增/既有关系时填写",
                    "previous_state": "必须来自上一次定检证据，不能把本次描述复制为历史状态",
                    "development": "三档枚举：无、轻微、严重；必须有当前与历史证据才能判断",
                },
                "recommendation_field_semantics": {
                    "category": "立即处置只用于封闭、限行或紧急风险；普通修复为尽快维修；观察和日常养护为预防性养护",
                    "component": "建议针对的精确病害部位，不是无依据的全桥",
                    "content": "保留原报告动作、对象和条件，不新增处治措施",
                },
                "hard_rules": [
                    "每个 patch 的 evidence_ids 必须来自对应输入行 allowed_evidence_ids",
                    "没有历史证据时不得填写 previous_state 或 development",
                    "标为‘上一次检测结果’与‘本次检测结果’的同一结构分区对比表属于历史证据；先区分两列再判断",
                    "‘新增部分病害’不能机械广播给整个结构分区；应比较该行病害与上一次描述，只修改证据能支持的行",
                    "对每个 current.is_new=null 的有对比行必须作二元裁决：上次无对应而本次出现则 true；上次已有同部位同类病害则 false",
                    "裁决为 true 时 previous_state 视为未发现且 development 为无；裁决为 false 时再依据前后描述判断发展程度",
                    "previous_state 只能填已发现或未发现；development 只能填无、轻微或严重",
                    "所有数字和等级必须逐字出现在引用证据中",
                    "只返回确实需要修改且证据充分的字段",
                ],
                "diseases": diseases,
                "recommendations": recommendations,
                "force_temporal_decision": force_temporal_decision,
            }
        return self.client.complete_model(
            DetailReview,
            system=self.system_prompt,
            user_payload=user_payload,
            temperature=0.0,
            task="detail-review",
        )
