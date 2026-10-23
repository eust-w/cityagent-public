from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from bridge_evidence.parsers.docx import ParagraphBlock, ParsedDocument, TableBlock

from .consensus import ConsensusResolver, accepted_table_region_ids
from .schema import DocumentIR, DocumentRegion, ExpertOutput


class DocumentExpert(Protocol):
    name: str
    channel: str
    weight: float

    def analyze(
        self,
        parsed: ParsedDocument,
        source_path: Path,
        document_id: str,
    ) -> ExpertOutput: ...


@dataclass(frozen=True)
class CleaningResult:
    parsed: ParsedDocument
    ir: DocumentIR


class MultiExpertDocumentPipeline:
    def __init__(
        self,
        experts: list[DocumentExpert],
        resolver: ConsensusResolver,
        *,
        mode: Literal["off", "audit", "enforce"] = "enforce",
        fail_closed: bool = True,
    ):
        self.experts = list(experts)
        self.resolver = resolver
        self.mode = mode
        self.fail_closed = fail_closed

    def clean(
        self,
        parsed: ParsedDocument,
        source_path: Path,
        document_id: str,
    ) -> CleaningResult:
        if self.mode == "off":
            return CleaningResult(
                parsed=parsed,
                ir=DocumentIR(
                    document_id=document_id,
                    source_path=str(source_path),
                    metadata={"mode": "off", "accepted_external_regions": 0},
                ),
            )

        outputs: list[ExpertOutput] = []
        errors: list[str] = []
        for expert in self.experts:
            try:
                output = expert.analyze(parsed, source_path, document_id)
            except Exception as exc:
                message = f"{expert.name}: {exc}"
                errors.append(message)
                if self.fail_closed:
                    raise RuntimeError(f"document expert failed closed: {message}") from exc
                continue
            output = output.model_copy(update={
                "regions": [
                    region if region.expert else region.model_copy(update={"expert": output.expert})
                    for region in output.regions
                ]
            })
            outputs.append(output)

        regions = [region for output in outputs for region in output.regions]
        candidates = [candidate for output in outputs for candidate in output.candidates]
        consensus = self.resolver.resolve(candidates)
        accepted_regions = self._accepted_external_regions(regions, candidates, consensus.resolved)
        ir = DocumentIR(
            document_id=document_id,
            source_path=str(source_path),
            regions=regions,
            candidates=candidates,
            resolved=consensus.resolved,
            conflicts=consensus.conflicts,
            metadata={
                "mode": self.mode,
                "experts": [output.expert for output in outputs],
                "expert_errors": errors,
                "accepted_external_regions": len(accepted_regions),
            },
        )
        if self.mode != "enforce" or not accepted_regions:
            return CleaningResult(parsed=parsed, ir=ir)
        return CleaningResult(parsed=self._augment(parsed, accepted_regions), ir=ir)

    def _accepted_external_regions(
        self,
        regions: list[DocumentRegion],
        candidates: list,
        resolved: dict,
    ) -> list[DocumentRegion]:
        accepted_ids: set[str] = set()
        for candidate in candidates:
            resolution = resolved.get(candidate.field)
            if (
                resolution is None
                or resolution.abstained
                or candidate.channel == "native"
                or not candidate.region_id
            ):
                continue
            if self.resolver.normalize_value(candidate.field, candidate.value) == resolution.normalized_value:
                accepted_ids.add(candidate.region_id)
        table_regions_by_expert: dict[str, list[DocumentRegion]] = {}
        for region in regions:
            if not region.rows or not region.expert or region.expert == "native-ooxml":
                continue
            table_regions_by_expert.setdefault(region.expert, []).append(region)
        accepted_ids.update(accepted_table_region_ids(
            table_regions_by_expert,
            min_agreement=self.resolver.min_agreement,
        ))
        return [
            region
            for region in regions
            if region.id in accepted_ids and region.expert != "native-ooxml"
        ]

    @staticmethod
    def _augment(parsed: ParsedDocument, regions: list[DocumentRegion]) -> ParsedDocument:
        paragraphs = list(parsed.paragraphs)
        tables = list(parsed.tables)
        existing_text = {paragraph.text for paragraph in paragraphs if paragraph.text}
        for region in regions:
            style = f"DocumentAI:{region.expert or 'external'}"
            if region.rows:
                tables.append(TableBlock(
                    index=len(tables),
                    rows=region.rows,
                    context=[style, region.content] if region.content else [style],
                ))
            elif region.content and region.content not in existing_text:
                paragraphs.append(ParagraphBlock(
                    index=len(paragraphs),
                    text=region.content,
                    style=style,
                ))
                existing_text.add(region.content)
        return ParsedDocument(
            path=parsed.path,
            paragraphs=paragraphs,
            tables=tables,
            images=list(parsed.images),
        )
