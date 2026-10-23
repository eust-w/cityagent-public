from __future__ import annotations

from typing import Any

from bridge_evidence.config import Settings
from bridge_evidence.models.schema import FieldValue

from .client import GatewayClient


class FactResolver:
    def __init__(self, client: GatewayClient, settings: Settings, system_prompt: str):
        self.client = client
        self.settings = settings
        self.system_prompt = system_prompt

    def resolve(self, field_name: str, candidates: list[dict[str, Any]]) -> FieldValue:
        result = self.client.complete_json(
            system=self.system_prompt,
            user_payload={
                "task": "从候选中选择唯一事实",
                "field": field_name,
                "candidates": candidates,
                "output_schema": {"selected_index": 0, "reason": "只能引用候选"},
            },
            temperature=self.settings.fact_temperature,
            task="fact",
        )
        index = int(result["selected_index"])
        if index < 0 or index >= len(candidates):
            raise ValueError(f"FactResolver returned invalid candidate index {index}")
        selected = candidates[index]
        return FieldValue(
            value=selected["value"],
            evidence=list(selected.get("evidence") or []),
            source_priority=int(selected.get("source_priority") or 0),
            confidence=float(selected.get("confidence") or 0.8),
            conflicts=[str(c["value"]) for i, c in enumerate(candidates) if i != index],
        )
