from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .agent import AgentRunRecord


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["paragraph", "table", "filename", "model", "derived", "image"]
    text: str
    source_path: str
    priority: int = Field(default=50, ge=0, le=100)
    paragraph_index: int | None = None
    table_index: int | None = None
    row_index: int | None = None


class FieldValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | float | int | bool | None = None
    evidence: list[str] = Field(default_factory=list)
    source_priority: int = Field(default=0, ge=0, le=100)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    conflicts: list[str] = Field(default_factory=list)


class BridgeMeta(BaseModel):
    name: FieldValue = Field(default_factory=FieldValue)
    report_date: FieldValue = Field(default_factory=FieldValue)


class ScoreEntry(BaseModel):
    score: FieldValue = Field(default_factory=FieldValue)
    control_score: FieldValue = Field(default_factory=FieldValue)
    grade: FieldValue = Field(default_factory=FieldValue)


class EntityScoreState(BaseModel):
    entity_id: str
    label: str
    side: Literal["whole", "left", "right", "unknown"] = "unknown"
    overall: ScoreEntry = Field(default_factory=ScoreEntry)
    superstructure: ScoreEntry = Field(default_factory=ScoreEntry)
    substructure: ScoreEntry = Field(default_factory=ScoreEntry)
    deck: ScoreEntry = Field(default_factory=ScoreEntry)


class ScoreState(BaseModel):
    overall: ScoreEntry = Field(default_factory=ScoreEntry)
    superstructure: ScoreEntry = Field(default_factory=ScoreEntry)
    substructure: ScoreEntry = Field(default_factory=ScoreEntry)
    deck: ScoreEntry = Field(default_factory=ScoreEntry)
    previous: ScoreEntry = Field(default_factory=ScoreEntry)
    presentation_mode: Literal["single", "dual"] = "single"
    entities: list[EntityScoreState] = Field(default_factory=list)


class ComparisonRecord(BaseModel):
    section: str
    previous: str
    current: str
    development: str
    evidence: list[str] = Field(default_factory=list)


class DiseaseHistoryLink(BaseModel):
    previous_cycle: int | None = None
    previous_index: int | None = None
    relation: Literal["exact", "candidate", "group", "absence"]
    probability: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class DiseaseEvent(BaseModel):
    event_id: str
    sequence: str = ""
    component: str
    location: str = ""
    side: Literal["left", "right", "unknown"] = "unknown"
    span: str | None = None
    disease_type: str
    description: str
    measurements: dict[str, str] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    history_links: list[DiseaseHistoryLink] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Disease(BaseModel):
    sequence: str = ""
    component: str
    location: str = ""
    type: str
    description: str
    metrics: dict[str, str] = Field(default_factory=dict)
    is_new: bool | None = None
    novelty_source: Literal[
        "current_report_new",
        "current_report_no_new",
        "current_report_stable",
        "history_exact",
        "history_group",
        "history_absence",
        "no_comparison",
        "model",
    ] | None = None
    previous_state: str | None = None
    development: str | None = None
    evidence: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    history_links: list[DiseaseHistoryLink] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("description", "location", "type", mode="before")
    @classmethod
    def compact_text(cls, value: Any) -> str:
        return " ".join(str(value or "").split())


class Recommendation(BaseModel):
    sequence: int
    category: Literal["立即处置", "尽快维修", "预防性养护"]
    content: str
    component: str = "全桥"
    evidence: list[str] = Field(default_factory=list)
    disease_event_ids: list[str] = Field(default_factory=list)
    policy_basis: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class NarrativeClaim(BaseModel):
    claim_type: Literal["risk", "cause", "safety_impact"]
    text: str
    disease_event_ids: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class NarrativeState(BaseModel):
    development_trend: str = ""
    overall_conclusion: str = ""
    key_risks: str = ""
    causes: list[str] = Field(default_factory=list)
    dispositions: list[str] = Field(default_factory=list)
    safety_impacts: list[str] = Field(default_factory=list)
    claims: list[NarrativeClaim] = Field(default_factory=list)


class ValidationMessage(BaseModel):
    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    field: str | None = None


class ValidationState(BaseModel):
    errors: list[ValidationMessage] = Field(default_factory=list)
    warnings: list[ValidationMessage] = Field(default_factory=list)
    passed: bool = False
    evidence_coverage: float = 0.0


class RunMetrics(BaseModel):
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    duration_seconds: float | None = None
    llm_calls: int = 0
    chat_calls: int = 0
    vision_calls: int = 0
    embedding_calls: int = 0
    reranker_calls: int = 0
    planner_calls: int = 0
    llm_cache_hits: int = 0
    task_calls: dict[str, int] = Field(default_factory=dict)
    task_cache_hits: dict[str, int] = Field(default_factory=dict)
    task_models: dict[str, str] = Field(default_factory=dict)
    model_calls: dict[str, int] = Field(default_factory=dict)
    model_cache_hits: dict[str, int] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    online_audit_passed: bool | None = None
    online_audit_errors: list[str] = Field(default_factory=list)


class DocumentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    document_id: str
    source_path: str
    converted_path: str | None = None
    bridge: BridgeMeta = Field(default_factory=BridgeMeta)
    scores: ScoreState = Field(default_factory=ScoreState)
    comparisons: list[ComparisonRecord] = Field(default_factory=list)
    diseases: list[Disease] = Field(default_factory=list)
    disease_events: list[DiseaseEvent] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    narrative: NarrativeState = Field(default_factory=NarrativeState)
    evidence: dict[str, EvidenceRef] = Field(default_factory=dict)
    validation: ValidationState = Field(default_factory=ValidationState)
    agent_run: AgentRunRecord | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)

    def add_evidence(self, evidence: EvidenceRef) -> str:
        self.evidence[evidence.id] = evidence
        return evidence.id
