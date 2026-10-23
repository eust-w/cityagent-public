from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bridge_evidence.config import Settings
from bridge_evidence.models.schema import Disease

from .client import GatewayClient


class DiseaseNormalizationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    component: str = Field(min_length=1, max_length=60)
    type: str = Field(min_length=1, max_length=120)


class DiseaseNormalizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[DiseaseNormalizationItem] = Field(default_factory=list, max_length=20)


class DiseaseNormalizer:
    def __init__(self, client: GatewayClient, settings: Settings, system_prompt: str):
        self.client = client
        self.settings = settings
        self.system_prompt = system_prompt

    def normalize(self, diseases: list[Disease]) -> list[Disease]:
        allowed_components = {
            "主梁", "横向联系", "桥墩", "桥台", "支座", "桥面铺装",
            "伸缩缝", "护栏", "排水系统", "人行道", "照明、标志", "其他",
        }
        targets = [(i, d) for i, d in enumerate(diseases) if d.component == "其他" or d.type == "未明确"]
        if not targets:
            return diseases
        for offset in range(0, len(targets), 20):
            batch = targets[offset:offset + 20]
            result = self.client.complete_model(
                DiseaseNormalizationResult,
                system=self.system_prompt,
                user_payload={
                    "task": "仅规范构件和病害类型",
                    "allowed_components": sorted(allowed_components),
                    "items": [{"index": i, "location": d.location, "type": d.type, "description": d.description} for i, d in batch],
                    "output_schema": {"items": [{"index": 0, "component": "主梁", "type": "裂缝"}]},
                },
                temperature=self.settings.disease_temperature,
                task="disease-normalization",
            )
            allowed_indices = {i for i, _ in batch}
            for item in result.items:
                index = item.index
                if index not in allowed_indices:
                    continue
                component = item.component
                disease_type = item.type.strip()
                if component in allowed_components:
                    diseases[index].component = component
                if disease_type:
                    diseases[index].type = disease_type
                diseases[index].confidence = min(diseases[index].confidence, 0.92)
        return diseases
