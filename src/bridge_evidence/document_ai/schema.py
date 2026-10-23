from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ExpertChannel = Literal["native", "visual", "semantic", "rule", "human"]
RegionKind = Literal[
    "paragraph",
    "table",
    "score_table",
    "disease_table",
    "recommendation_table",
    "history_table",
    "image",
    "unknown",
]


class DocumentRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: RegionKind = "unknown"
    content: str = ""
    page: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    coordinate_space: Literal["page", "image"] | None = None
    rows: list[list[str]] = Field(default_factory=list)
    paragraph_index: int | None = None
    table_index: int | None = None
    image_index: int | None = None
    image_indices: list[int] = Field(default_factory=list)
    media_path: str | None = None
    story: str | None = None
    occurrence_count: int = Field(default=1, ge=1)
    expert: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class CandidateValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    value: str | float | int | bool | None
    expert: str
    channel: ExpertChannel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str = ""
    region_id: str | None = None
    weight: float = Field(default=1.0, gt=0.0)


class ExpertOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expert: str
    channel: ExpertChannel
    regions: list[DocumentRegion] = Field(default_factory=list)
    candidates: list[CandidateValue] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CandidateConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    winner: str | None = None
    alternatives: list[str] = Field(default_factory=list)
    reason: Literal[
        "alternative_values",
        "insufficient_agreement",
        "insufficient_margin",
        "no_valid_candidate",
    ]


class ResolvedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    value: str | float | int | bool | None
    normalized_value: str
    confidence: float = Field(ge=0.0, le=1.0)
    contributors: list[str] = Field(default_factory=list)
    evidence_texts: list[str] = Field(default_factory=list)
    region_ids: list[str] = Field(default_factory=list)
    abstained: bool = False


class ConsensusResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolved: dict[str, ResolvedCandidate] = Field(default_factory=dict)
    conflicts: list[CandidateConflict] = Field(default_factory=list)


class DocumentIR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    document_id: str
    source_path: str
    regions: list[DocumentRegion] = Field(default_factory=list)
    candidates: list[CandidateValue] = Field(default_factory=list)
    resolved: dict[str, ResolvedCandidate] = Field(default_factory=dict)
    conflicts: list[CandidateConflict] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
