from __future__ import annotations

# Hand-constructed fixtures use synthetic bridge and organization names.

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bridge_evidence.config import Settings
from bridge_evidence.document_ai import (
    CandidateValue,
    ConsensusResolver,
    DocumentIR,
    DocumentRegion,
    ExpertOutput,
    MultiExpertDocumentPipeline,
    LocalImageOcrExpert,
    NativeOoxmlExpert,
    ProbabilisticDiseaseLinker,
)
from bridge_evidence.models.schema import Disease
from bridge_evidence.parsers.docx import ImageBlock, ParagraphBlock, ParsedDocument, TableBlock
from bridge_evidence.evidence.visuals import VisualCandidate
from bridge_evidence.document_ai.consensus import accepted_table_region_ids
from bridge_evidence.evidence.visuals import OcrSpan


def _parsed_document() -> ParsedDocument:
    return ParsedDocument(
        path=Path("sample.docx"),
        paragraphs=[
            ParagraphBlock(0, "桥梁名称：合成跨河大桥", "标题"),
            ParagraphBlock(1, "总体 BCI=88.50，技术状况等级为B级。", "正文"),
        ],
        tables=[
            TableBlock(
                0,
                [["构件部位", "病害类型", "病害描述"], ["主梁", "裂缝", "1#梁底裂缝"]],
                ["病害明细"],
            )
        ],
    )


def _candidate(
    expert: str,
    value: str,
    *,
    confidence: float = 0.95,
    field: str = "scores.overall.score",
) -> CandidateValue:
    return CandidateValue(
        field=field,
        value=value,
        expert=expert,
        channel="visual",
        confidence=confidence,
        evidence_text=f"BCI={value}",
        region_id=f"{expert}-region",
    )


def test_document_ir_rejects_untracked_fields() -> None:
    with pytest.raises(ValidationError):
        DocumentIR(
            document_id="doc-1",
            source_path="sample.docx",
            unexpected="not-auditable",
        )


def test_native_expert_builds_typed_regions_and_fact_candidates() -> None:
    output = NativeOoxmlExpert().analyze(
        _parsed_document(),
        Path("sample.docx"),
        "doc-1",
    )

    assert [region.kind for region in output.regions] == [
        "paragraph",
        "paragraph",
        "disease_table",
    ]
    values = {(candidate.field, str(candidate.value)) for candidate in output.candidates}
    assert ("bridge.name", "合成跨河大桥") in values
    assert ("scores.overall.score", "88.50") in values
    assert ("scores.overall.grade", "B级") in values


def test_native_expert_emits_image_regions_with_stable_source_coordinates() -> None:
    parsed = ParsedDocument(
        path=Path("image.docx"),
        paragraphs=[],
        tables=[],
        images=[
            ImageBlock(
                index=0,
                relationship_id="rId7",
                target="word/media/image1.png",
                alt_text="主梁裂缝照片",
                story="body",
            )
        ],
    )

    output = NativeOoxmlExpert().analyze(parsed, parsed.path, "doc-image")

    assert len(output.regions) == 1
    assert output.regions[0].kind == "image"
    assert output.regions[0].image_index == 0
    assert output.regions[0].content == "主梁裂缝照片"


def test_local_image_ocr_expert_emits_auditable_regions_and_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = ParsedDocument(
        path=Path("scan.docx"),
        paragraphs=[],
        tables=[],
        images=[
            ImageBlock(0, "rId7", "word/media/image1.png", "评分扫描表", "body"),
            ImageBlock(1, "rId7", "word/media/image1.png", "评分扫描表", "body"),
        ],
    )
    visual = VisualCandidate(
        media_path="word/media/image1.png",
        digest="a" * 64,
        mime_type="image/png",
        width=1200,
        height=800,
        byte_size=20_000,
        text_likelihood=0.96,
        data=b"image-data",
        occurrence_count=2,
        image_indices=(0, 1),
        stories=("body",),
        alt_texts=("评分扫描表",),
    )
    monkeypatch.setattr(
        "bridge_evidence.document_ai.experts.extract_visual_candidates",
        lambda *_args, **_kwargs: [visual],
    )
    monkeypatch.setattr(
        "bridge_evidence.document_ai.experts.LocalOcrEngine.ocr_with_layout",
        lambda *_args, **_kwargs: ("总体 BCI=88.50\n桥面系 BCIm=85.20", []),
    )

    output = LocalImageOcrExpert(limit=2).analyze(parsed, parsed.path, "doc-scan")

    assert len(output.regions) == 1
    region = output.regions[0]
    assert region.kind == "image"
    assert region.image_index == 0
    assert region.media_path == "word/media/image1.png"
    assert region.occurrence_count == 2
    values = {(candidate.field, str(candidate.value)) for candidate in output.candidates}
    assert ("scores.overall.score", "88.50") in values
    assert ("scores.deck.score", "85.20") in values


def test_local_image_ocr_expert_reuses_content_addressed_disk_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = ParsedDocument(path=Path("cache.docx"), paragraphs=[], tables=[])
    visual = VisualCandidate(
        media_path="word/media/image1.png",
        digest="b" * 64,
        mime_type="image/png",
        width=1000,
        height=700,
        byte_size=20_000,
        text_likelihood=0.95,
        data=b"same-image",
    )
    monkeypatch.setattr(
        "bridge_evidence.document_ai.experts.extract_visual_candidates",
        lambda *_args, **_kwargs: [visual],
    )
    calls = 0

    def ocr(*_args, **_kwargs) -> tuple[str, list[OcrSpan]]:
        nonlocal calls
        calls += 1
        return "总体 BCI=88.50", []

    monkeypatch.setattr(
        "bridge_evidence.document_ai.experts.LocalOcrEngine.ocr_with_layout",
        ocr,
    )
    cache_dir = tmp_path / "ocr-cache"

    LocalImageOcrExpert(cache_dir=cache_dir).analyze(parsed, parsed.path, "first")
    LocalImageOcrExpert(cache_dir=cache_dir).analyze(parsed, parsed.path, "second")

    assert calls == 1
    payload = json.loads(
        (cache_dir / f"{visual.digest}.layout-v2.json").read_text(encoding="utf-8")
    )
    assert payload == {"version": 2, "text": "总体 BCI=88.50", "spans": []}


def test_local_image_ocr_reconstructs_auditable_disease_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = ParsedDocument(path=Path("scan.docx"), paragraphs=[], tables=[])
    visual = VisualCandidate(
        media_path="word/media/disease.png",
        digest="c" * 64,
        mime_type="image/png",
        width=1200,
        height=800,
        byte_size=20_000,
        text_likelihood=0.97,
        data=b"disease-table",
    )
    spans = [
        OcrSpan("病害位置", 0.98, (0.05, 0.10, 0.22, 0.16)),
        OcrSpan("病害类型", 0.97, (0.31, 0.10, 0.47, 0.16)),
        OcrSpan("病害描述", 0.96, (0.56, 0.10, 0.78, 0.16)),
        OcrSpan("左幅1跨1#梁底", 0.95, (0.05, 0.24, 0.25, 0.30)),
        OcrSpan("裂缝", 0.94, (0.32, 0.24, 0.42, 0.30)),
        OcrSpan("纵向裂缝0.5m", 0.93, (0.56, 0.24, 0.82, 0.30)),
    ]
    monkeypatch.setattr(
        "bridge_evidence.document_ai.experts.extract_visual_candidates",
        lambda *_args, **_kwargs: [visual],
    )
    monkeypatch.setattr(
        "bridge_evidence.document_ai.experts.LocalOcrEngine.ocr_with_layout",
        lambda *_args, **_kwargs: ("\n".join(span.text for span in spans), spans),
    )

    output = LocalImageOcrExpert().analyze(parsed, parsed.path, "doc-table")

    assert [region.kind for region in output.regions] == ["image", "disease_table"]
    table = output.regions[1]
    assert table.coordinate_space == "image"
    assert table.bbox == pytest.approx((0.05, 0.10, 0.82, 0.30))
    assert table.rows[1] == ["左幅1跨1#梁底", "裂缝", "纵向裂缝0.5m"]


def test_consensus_accepts_independent_agreement_and_records_alternative() -> None:
    resolver = ConsensusResolver(
        min_agreement=2,
        min_confidence=0.70,
        min_margin=0.10,
    )

    result = resolver.resolve([
        _candidate("paddleocr-vl", "88.50"),
        _candidate("mineru", "88.500"),
        _candidate("semantic-vlm", "89.10", confidence=0.75),
    ])

    resolved = result.resolved["scores.overall.score"]
    assert str(resolved.value) == "88.50"
    assert resolved.contributors == ["mineru", "paddleocr-vl"]
    assert resolved.abstained is False
    assert result.conflicts[0].alternatives == ["89.1"]


def test_consensus_abstains_when_top_values_are_tied() -> None:
    result = ConsensusResolver(
        min_agreement=1,
        min_confidence=0.70,
        min_margin=0.20,
    ).resolve([
        _candidate("paddleocr-vl", "88.50"),
        _candidate("mineru", "89.10"),
    ])

    assert result.resolved["scores.overall.score"].abstained is True
    assert result.conflicts[0].reason == "insufficient_margin"


class _ExternalExpert:
    name = "paddleocr-vl"
    channel = "visual"
    weight = 1.0

    def analyze(self, parsed: ParsedDocument, source_path: Path, document_id: str) -> ExpertOutput:
        return ExpertOutput(
            expert=self.name,
            channel=self.channel,
            regions=[
                DocumentRegion(
                    id="visual-score",
                    kind="score_table",
                    content="桥面系 BCIm=91.25",
                    page=3,
                    confidence=0.98,
                )
            ],
            candidates=[_candidate(
                self.name,
                "91.25",
                confidence=0.98,
                field="scores.deck.score",
            ).model_copy(update={"region_id": "visual-score"})],
        )


def test_pipeline_audit_is_non_mutating_and_enforce_appends_accepted_evidence() -> None:
    parsed = _parsed_document()
    experts = [NativeOoxmlExpert(), _ExternalExpert()]
    resolver = ConsensusResolver(min_agreement=1, min_confidence=0.90, min_margin=0.10)

    audited = MultiExpertDocumentPipeline(experts, resolver, mode="audit").clean(
        parsed,
        parsed.path,
        "doc-1",
    )
    enforced = MultiExpertDocumentPipeline(experts, resolver, mode="enforce").clean(
        parsed,
        parsed.path,
        "doc-1",
    )

    assert audited.parsed == parsed
    assert len(enforced.parsed.paragraphs) == len(parsed.paragraphs) + 1
    assert enforced.parsed.paragraphs[-1].text == "桥面系 BCIm=91.25"
    assert enforced.parsed.paragraphs[-1].style == "DocumentAI:paddleocr-vl"
    assert enforced.ir.metadata["accepted_external_regions"] == 1


def test_probabilistic_linker_merges_photo_duplicates_but_preserves_other_spans() -> None:
    diseases = [
        Disease(
            sequence="1",
            component="主梁",
            location="左幅1跨1#梁底",
            type="裂缝",
            description="左幅1跨1#梁底纵向裂缝，长0.5m",
            evidence=["T1:R1"],
            confidence=0.91,
        ),
        Disease(
            sequence="2",
            component="主梁",
            location="左幅1跨1#梁底",
            type="裂缝",
            description="照片2：1#梁底纵向裂缝，长度约0.5m",
            evidence=["IMG:2"],
            confidence=0.88,
        ),
        Disease(
            sequence="3",
            component="主梁",
            location="左幅2跨1#梁底",
            type="裂缝",
            description="左幅2跨1#梁底纵向裂缝，长0.5m",
            evidence=["T1:R3"],
            confidence=0.94,
        ),
    ]

    merged = ProbabilisticDiseaseLinker(threshold=0.86).deduplicate(diseases)

    assert len(merged) == 2
    first_span = next(item for item in merged if "1跨" in item.location)
    assert first_span.evidence == ["IMG:2", "T1:R1"]
    assert any("2跨" in item.location for item in merged)


def test_probabilistic_linker_aggregates_photo_ledger_within_side_span_only() -> None:
    diseases = [
        Disease(
            sequence=str(index),
            component="主梁",
            location=f"左幅1跨{index}#梁底",
            type="裂缝",
            description=f"照片{index}：左幅1跨{index}#梁底纵向裂缝，长0.{index}m",
            evidence=[f"IMG:{index}"],
            confidence=0.90,
        )
        for index in range(1, 4)
    ]
    diseases.extend([
        Disease(
            sequence="4",
            component="主梁",
            location="右幅1跨1#梁底",
            type="裂缝",
            description="右幅1跨1#梁底纵向裂缝",
            evidence=["IMG:4"],
            confidence=0.91,
        ),
        Disease(
            sequence="5",
            component="主梁",
            location="左幅2跨1#梁底",
            type="裂缝",
            description="左幅2跨1#梁底纵向裂缝",
            evidence=["IMG:5"],
            confidence=0.91,
        ),
    ])

    merged = ProbabilisticDiseaseLinker(threshold=0.90).deduplicate(diseases)

    assert len(merged) == 3
    aggregate = next(item for item in merged if item.metrics.get("聚合数量") == "3")
    assert aggregate.location == "左幅1跨主梁"
    assert all(f"{index}#梁底" in aggregate.description for index in range(1, 4))
    assert aggregate.evidence == ["IMG:1", "IMG:2", "IMG:3"]
    assert any(item.location.startswith("右幅1跨") for item in merged)
    assert any(item.location.startswith("左幅2跨") for item in merged)


def test_document_ai_is_the_default_cleaning_boundary() -> None:
    settings = Settings(llm_mode="off")

    assert settings.document_ai_mode == "enforce"
    assert settings.document_expert_commands == {}
    assert settings.document_consensus_min_agreement == 2
def test_table_region_consensus_requires_two_distinct_experts() -> None:
    rows = [
        ["病害位置", "病害类型", "病害描述"],
        ["左幅1跨1#梁底", "裂缝", "纵向裂缝0.5m"],
    ]
    local_region = DocumentRegion(
        id="local:image-table",
        kind="disease_table",
        content="病害位置 | 病害类型 | 病害描述",
        rows=rows,
        confidence=0.95,
    )
    vlm_region = DocumentRegion(
        id="vlm:image-table",
        kind="disease_table",
        content="病害位置 | 病害类型 | 病害描述",
        rows=[list(row) for row in rows],
        confidence=0.91,
    )

    assert accepted_table_region_ids(
        {"local_image_ocr": [local_region]}, min_agreement=2
    ) == set()
    assert accepted_table_region_ids(
        {"local_image_ocr": [local_region], "vlm": [vlm_region]},
        min_agreement=2,
    ) == {"local:image-table"}


class _TableExpert:
    channel = "visual"
    weight = 1.0

    def __init__(self, name: str, confidence: float):
        self.name = name
        self.confidence = confidence

    def analyze(self, parsed: ParsedDocument, source_path: Path, document_id: str) -> ExpertOutput:
        del parsed, source_path, document_id
        rows = [
            ["病害位置", "病害类型", "病害描述"],
            ["左幅1跨1#梁底", "裂缝", "纵向裂缝0.5m"],
        ]
        return ExpertOutput(
            expert=self.name,
            channel=self.channel,
            regions=[DocumentRegion(
                id=f"{self.name}:table",
                kind="disease_table",
                content="病害位置 | 病害类型 | 病害描述",
                rows=rows,
                confidence=self.confidence,
            )],
        )


def test_pipeline_appends_table_only_after_cross_expert_consensus() -> None:
    parsed = _parsed_document()
    resolver = ConsensusResolver(min_agreement=2)
    single = MultiExpertDocumentPipeline(
        [_TableExpert("local-image-ocr", 0.95)],
        resolver,
        mode="enforce",
    ).clean(parsed, parsed.path, "single")
    agreed = MultiExpertDocumentPipeline(
        [_TableExpert("local-image-ocr", 0.95), _TableExpert("vlm", 0.91)],
        resolver,
        mode="enforce",
    ).clean(parsed, parsed.path, "agreed")

    assert len(single.parsed.tables) == len(parsed.tables)
    assert len(agreed.parsed.tables) == len(parsed.tables) + 1
    assert agreed.parsed.tables[-1].rows[1][0] == "左幅1跨1#梁底"
    assert agreed.ir.metadata["accepted_external_regions"] == 1
