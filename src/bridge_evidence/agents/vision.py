from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bridge_evidence.config import Settings

from .client import GatewayClient


VisualField = Literal[
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


VISUAL_FIELD_ALIASES: dict[str, VisualField] = {
    "bridge.name": "bridge.name",
    "桥名": "bridge.name",
    "桥梁名称": "bridge.name",
    "工程名称": "bridge.name",
    "bridge.report_date": "bridge.report_date",
    "日期": "bridge.report_date",
    "报告日期": "bridge.report_date",
    "检测日期": "bridge.report_date",
    "scores.overall.score": "scores.overall.score",
    "BCI": "scores.overall.score",
    "DR": "scores.overall.score",
    "总体评分": "scores.overall.score",
    "scores.overall.grade": "scores.overall.grade",
    "等级": "scores.overall.grade",
    "总体等级": "scores.overall.grade",
    "scores.deck.score": "scores.deck.score",
    "BCIM": "scores.deck.score",
    "桥面系评分": "scores.deck.score",
    "scores.deck.grade": "scores.deck.grade",
    "桥面系等级": "scores.deck.grade",
    "scores.superstructure.score": "scores.superstructure.score",
    "BCIS": "scores.superstructure.score",
    "上部结构评分": "scores.superstructure.score",
    "scores.superstructure.grade": "scores.superstructure.grade",
    "上部结构等级": "scores.superstructure.grade",
    "scores.substructure.score": "scores.substructure.score",
    "BCIX": "scores.substructure.score",
    "下部结构评分": "scores.substructure.score",
    "scores.substructure.grade": "scores.substructure.grade",
    "下部结构等级": "scores.substructure.grade",
    "scores.previous.score": "scores.previous.score",
    "上次评分": "scores.previous.score",
    "上一次评分": "scores.previous.score",
    "scores.previous.grade": "scores.previous.grade",
    "上次等级": "scores.previous.grade",
    "上一次等级": "scores.previous.grade",
}


class VisualFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: VisualField
    value: str = Field(min_length=1, max_length=120)
    evidence_quote: str = Field(min_length=1, max_length=300)


class VisualRead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visible_text: str = Field(default="", max_length=4000)
    facts: list[VisualFact] = Field(default_factory=list, max_length=20)


class VerifiedVisualRead(BaseModel):
    primary_text: str = ""
    critic_text: str = ""
    verified_facts: list[VisualFact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _normalized(value: str) -> str:
    return (
        re.sub(r"[\s，,。；;：:（）()=]", "", str(value or ""))
        .upper()
        .replace("GRADE", "级")
        .replace("等级", "级")
        .replace("类", "级")
    )


def _field_alias(value: object) -> VisualField | None:
    rendered = str(value or "").strip()
    return VISUAL_FIELD_ALIASES.get(rendered) or VISUAL_FIELD_ALIASES.get(rendered.upper())


def _fact_key(fact: VisualFact) -> tuple[str, str]:
    value = _normalized(fact.value)
    if fact.field.endswith(".score"):
        number = re.search(r"\d+(?:\.\d+)?", value)
        if number:
            value = f"{float(number.group()):.4f}".rstrip("0").rstrip(".")
    elif fact.field.endswith(".grade"):
        grade = re.search(r"[ABCDE一二三四五](?=级|$)", value)
        if grade:
            value = grade.group().upper()
    return fact.field, value


class VisionEvidenceReviewer:
    def __init__(self, client: GatewayClient, settings: Settings, system_prompt: str):
        self.client = client
        self.settings = settings
        self.system_prompt = system_prompt

    def _read(self, image_data_url: str, *, model: str, task: str) -> VisualRead:
        payload = self.client.complete_vision_json(
            system=self.system_prompt,
            prompt=(
                "逐字抄录图中可见文字。只提取明确出现的桥名、日期、BCI/BCIm/BCIs/BCIx/Dr、"
                "评分和等级；区分 BSI、权重、本次和上次。照片中看不见文字时 facts 为空。"
                "不得根据桥梁外观推断病害、等级或数值。输出 JSON："
                "{\"visible_text\":\"...\",\"facts\":[{\"field\":\"...\",\"value\":\"...\","
                "\"evidence_quote\":\"图中原文\"}]}"
            ),
            image_data_urls=[image_data_url],
            temperature=0.0,
            task=task,
            model=model,
        )
        # Some OpenAI-compatible open-weight VLMs honor the JSON envelope but
        # translate schema labels to Chinese (for example ``BCI`` or ``等级``).
        # Normalize only an explicit allow-list and discard all unknown facts;
        # this is a compatibility adapter, not a semantic inference step.
        normalized_facts: list[dict[str, str]] = []
        for raw_fact in payload.get("facts") or []:
            if not isinstance(raw_fact, dict):
                continue
            field = _field_alias(raw_fact.get("field"))
            value = str(raw_fact.get("value") or "").strip()
            quote = str(raw_fact.get("evidence_quote") or "").strip()
            if not field or not value or not quote:
                continue
            normalized_facts.append({
                "field": field,
                "value": value,
                "evidence_quote": quote,
            })
        return VisualRead.model_validate({
            "visible_text": str(payload.get("visible_text") or ""),
            "facts": normalized_facts,
        })

    def review(self, image_data_url: str) -> VerifiedVisualRead:
        from bridge_evidence.evidence.visuals import LocalOcrEngine
        import base64

        if not self.client or not self.settings.vision_model:
            # Local offline OCR fallback
            try:
                if "," in image_data_url:
                    _, encoded = image_data_url.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                else:
                    image_bytes = b""
                visible_text = LocalOcrEngine.ocr_image(image_bytes) if image_bytes else ""
                raw_facts = LocalOcrEngine.extract_facts(visible_text)
                verified: list[VisualFact] = []
                for rf in raw_facts:
                    field = _field_alias(rf.get("field"))
                    val = str(rf.get("value") or "").strip()
                    quote = str(rf.get("evidence_quote") or "").strip()
                    if field and val and quote:
                        verified.append(VisualFact(field=field, value=val, evidence_quote=quote))
                return VerifiedVisualRead(
                    primary_text=visible_text,
                    critic_text=visible_text,
                    verified_facts=verified,
                    warnings=["使用本地 Apple Vision/OCR 引擎"],
                )
            except Exception as exc:
                return VerifiedVisualRead(warnings=[f"本地视觉识别失败：{exc}"])

        primary = self._read(
            image_data_url,
            model=self.settings.vision_model,
            task="visual-evidence-primary",
        )
        warnings: list[str] = []
        critic = VisualRead()
        if self.settings.vision_critic_model:
            try:
                critic = self._read(
                    image_data_url,
                    model=self.settings.vision_critic_model,
                    task="visual-evidence-critic",
                )
            except Exception as exc:
                warnings.append(f"视觉交叉审校失败：{type(exc).__name__}")

        critic_facts = {_fact_key(fact): fact for fact in critic.facts}
        verified: list[VisualFact] = []
        for fact in primary.facts:
            key = _fact_key(fact)
            if key not in critic_facts:
                continue
            # Each model's quote must appear in its own OCR transcript.  This
            # rejects JSON fields that are not grounded in the visible text.
            peer = critic_facts[key]
            if _normalized(fact.evidence_quote) not in _normalized(primary.visible_text):
                continue
            if _normalized(peer.evidence_quote) not in _normalized(critic.visible_text):
                continue
            verified.append(fact)
        return VerifiedVisualRead(
            primary_text=primary.visible_text,
            critic_text=critic.visible_text,
            verified_facts=verified,
            warnings=warnings,
        )
