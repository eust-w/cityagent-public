from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "served-model"
    fast_model: str = "served-model"
    vision_model: str | None = None
    vision_critic_model: str | None = None
    task_models: dict[str, str] = Field(default_factory=dict)
    semantic_base_url: str = "http://127.0.0.1:8001/v1"
    embedding_model: str = "embedding-model"
    reranker_model: str = "reranker-model"
    api_key_env: str = "BRIDGE_EVIDENCE_API_KEY"
    semantic_api_key_env: str = "BRIDGE_EVIDENCE_SEMANTIC_API_KEY"
    timeout_seconds: int = Field(default=90, ge=1)
    max_retries: int = Field(default=3, ge=1, le=8)
    max_concurrency: int = Field(default=4, ge=1, le=32)
    llm_mode: Literal["off", "auto", "on"] = "off"
    planner_mode: Literal["model", "deterministic"] = "model"
    online_workflow_profile: Literal["full", "summary", "details"] = "full"
    fact_temperature: float = 0.0
    disease_temperature: float = 0.0
    decision_temperature: float = 0.1
    disease_match_threshold: float = Field(default=0.56, ge=0.0, le=1.0)
    document_ai_mode: Literal["off", "audit", "enforce"] = "enforce"
    document_expert_commands: dict[str, str] = Field(default_factory=dict)
    document_expert_timeout_seconds: int = Field(default=600, ge=1, le=3600)
    document_expert_fail_closed: bool = True
    document_consensus_min_agreement: int = Field(default=2, ge=1, le=8)
    document_consensus_min_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    document_consensus_min_margin: float = Field(default=0.15, ge=0.0, le=1.0)
    disease_dedup_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    history_label_dir: Path | None = None
    history_before_year: int | None = Field(default=None, ge=2000, le=2100)
    require_history: bool = False
    cache_dir: Path = Path(".bridge_evidence/cache")
    work_dir: Path = Path(".bridge_evidence/work")

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    @property
    def semantic_api_key(self) -> str | None:
        return os.environ.get(self.semantic_api_key_env)

    def model_for_task(self, task: str) -> str:
        return self.task_models.get(task, self.model)

    @property
    def llm_enabled(self) -> bool:
        if self.llm_mode == "off":
            return False
        missing = [
            env_name
            for env_name, value in (
                (self.api_key_env, self.api_key),
                (self.semantic_api_key_env, self.semantic_api_key),
            )
            if not value
        ]
        if missing:
            variables = ", ".join(f"${name}" for name in dict.fromkeys(missing))
            raise RuntimeError(
                f"LLM mode {self.llm_mode!r} requires {variables}; "
                "refusing to silently downgrade or skip required semantic models"
            )
        return True


def load_settings(path: Path | None = None, **overrides: object) -> Settings:
    source = path if path is not None else files("bridge_evidence.resources").joinpath("base.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    expert_commands = os.environ.get("BRIDGE_EVIDENCE_DOCUMENT_EXPERT_COMMANDS")
    env_overrides = {
        "base_url": os.environ.get("BRIDGE_EVIDENCE_BASE_URL"),
        "model": os.environ.get("BRIDGE_EVIDENCE_MODEL"),
        "fast_model": os.environ.get("BRIDGE_EVIDENCE_FAST_MODEL"),
        "vision_model": os.environ.get("BRIDGE_EVIDENCE_VISION_MODEL"),
        "vision_critic_model": os.environ.get("BRIDGE_EVIDENCE_VISION_CRITIC_MODEL"),
        "semantic_base_url": os.environ.get("BRIDGE_EVIDENCE_SEMANTIC_BASE_URL"),
        "api_key_env": os.environ.get("BRIDGE_EVIDENCE_API_KEY_ENV"),
        "semantic_api_key_env": os.environ.get("BRIDGE_EVIDENCE_SEMANTIC_API_KEY_ENV"),
        "embedding_model": os.environ.get("BRIDGE_EVIDENCE_EMBEDDING_MODEL"),
        "reranker_model": os.environ.get("BRIDGE_EVIDENCE_RERANKER_MODEL"),
        "planner_mode": os.environ.get("BRIDGE_EVIDENCE_PLANNER_MODE"),
        "online_workflow_profile": os.environ.get("BRIDGE_EVIDENCE_ONLINE_WORKFLOW_PROFILE"),
        "document_ai_mode": os.environ.get("BRIDGE_EVIDENCE_DOCUMENT_AI_MODE"),
        "document_expert_commands": json.loads(expert_commands) if expert_commands else None,
        "document_expert_timeout_seconds": os.environ.get("BRIDGE_EVIDENCE_DOCUMENT_EXPERT_TIMEOUT_SECONDS"),
        "document_expert_fail_closed": os.environ.get("BRIDGE_EVIDENCE_DOCUMENT_EXPERT_FAIL_CLOSED"),
        "document_consensus_min_agreement": os.environ.get("BRIDGE_EVIDENCE_DOCUMENT_CONSENSUS_MIN_AGREEMENT"),
        "document_consensus_min_confidence": os.environ.get("BRIDGE_EVIDENCE_DOCUMENT_CONSENSUS_MIN_CONFIDENCE"),
        "document_consensus_min_margin": os.environ.get("BRIDGE_EVIDENCE_DOCUMENT_CONSENSUS_MIN_MARGIN"),
        "disease_dedup_threshold": os.environ.get("BRIDGE_EVIDENCE_DISEASE_DEDUP_THRESHOLD"),
        "history_label_dir": os.environ.get("BRIDGE_EVIDENCE_HISTORY_LABEL_DIR"),
        "history_before_year": os.environ.get("BRIDGE_EVIDENCE_HISTORY_BEFORE_YEAR"),
        "require_history": os.environ.get("BRIDGE_EVIDENCE_REQUIRE_HISTORY"),
        "cache_dir": os.environ.get("BRIDGE_EVIDENCE_CACHE_DIR"),
        "work_dir": os.environ.get("BRIDGE_EVIDENCE_WORK_DIR"),
    }
    data.update({k: v for k, v in env_overrides.items() if v})
    data.update({k: v for k, v in overrides.items() if v is not None})
    return Settings.model_validate(data)


def load_prompts(path: Path | None = None) -> dict[str, str]:
    source = path if path is not None else files("bridge_evidence.resources").joinpath("prompts.json")
    return json.loads(source.read_text(encoding="utf-8"))
