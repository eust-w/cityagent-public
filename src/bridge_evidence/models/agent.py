from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


AgentAction = Literal[
    "get_validation_report",
    "inspect_summary",
    "search_evidence",
    "inspect_table",
    "list_score_candidates",
    "profile_disease_tables",
    "inspect_visual_evidence",
    "review_summary",
    "review_details",
    "apply_fact_patch",
    "mark_unavailable",
    "normalize_diseases",
    "write_decision",
    "finalize",
    "fail",
]


class ToolDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: AgentAction
    description: str
    read_only: bool = True
    arguments_schema: dict[str, Any] = Field(default_factory=dict)


class ToolManifest(BaseModel):
    protocol_version: str = "bridge-evidence/1.0"
    tools: list[ToolDescriptor]


class AgentCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    seq: int = Field(ge=1)
    action: AgentAction
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=500)
    expected_information_gain: str = Field(default="", max_length=300)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    seq: int = Field(ge=1)
    tool: AgentAction
    success: bool
    content: dict[str, Any] = Field(default_factory=dict)
    new_evidence_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class TrajectoryStep(BaseModel):
    command: AgentCommand
    result: ToolResult
    state_hash_before: str
    state_hash_after: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AgentObservation(BaseModel):
    key_fields: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[dict[str, Any]] = Field(default_factory=list)
    validation_warnings: list[dict[str, Any]] = Field(default_factory=list)
    format_signals: list[str] = Field(default_factory=list)
    evidence_coverage: float = 0.0
    disease_count: int = 0
    recommendation_count: int = 0
    required_actions: list[dict[str, Any]] = Field(default_factory=list)
    completed_actions: list[dict[str, Any]] = Field(default_factory=list)
    last_result: ToolResult | None = None


class AgentRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    planner: Literal["deterministic", "qwen", "deepseek"]
    status: Literal["running", "completed", "failed", "budget_exhausted"] = "running"
    max_actions: int = Field(default=16, ge=1, le=32)
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    command_fingerprints: list[str] = Field(default_factory=list)
    final_validation_passed: bool = False
    failure_reason: str | None = None
