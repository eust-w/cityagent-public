from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from bridge_evidence.evidence.visuals import (
    LocalOcrEngine,
    OcrSpan,
    extract_visual_candidates,
    reconstruct_ocr_table,
)
from bridge_evidence.parsers.docx import ParsedDocument, TableBlock

from .schema import CandidateValue, DocumentRegion, ExpertOutput, RegionKind


def _table_kind(table: TableBlock) -> RegionKind:
    if table.is_disease_table():
        return "disease_table"
    text = "|".join("|".join(row) for row in table.rows)
    if any(term in text for term in ("养护建议", "维修建议", "处治建议", "处置建议")):
        return "recommendation_table"
    if any(term in text for term in ("上一次", "历次", "历史检测", "历史评分")):
        return "history_table"
    if any(term in text.upper() for term in ("BCI", "BSI", "技术状况", "总体评定")):
        return "score_table"
    return "table"


class NativeOoxmlExpert:
    name = "native-ooxml"
    channel = "native"
    weight = 1.0

    def analyze(
        self,
        parsed: ParsedDocument,
        source_path: Path,
        document_id: str,
    ) -> ExpertOutput:
        del document_id
        regions: list[DocumentRegion] = []
        candidates: list[CandidateValue] = []
        for paragraph in parsed.paragraphs:
            region_id = f"native:P{paragraph.index}"
            regions.append(DocumentRegion(
                id=region_id,
                kind="paragraph",
                content=paragraph.text,
                paragraph_index=paragraph.index,
                expert=self.name,
                confidence=1.0,
            ))
            candidates.extend(self._fact_candidates(paragraph.text, region_id))

        for table in parsed.tables:
            region_id = f"native:T{table.index}"
            content = "\n".join(" | ".join(row) for row in table.rows)
            regions.append(DocumentRegion(
                id=region_id,
                kind=_table_kind(table),
                content=content,
                rows=table.rows,
                table_index=table.index,
                expert=self.name,
                confidence=1.0,
            ))
            candidates.extend(self._fact_candidates(content, region_id))
        for image in parsed.images:
            region_id = f"native:IMG{image.index}"
            content = image.alt_text or image.target
            regions.append(DocumentRegion(
                id=region_id,
                kind="image",
                content=content,
                image_index=image.index,
                image_indices=[image.index],
                media_path=image.target,
                story=image.story,
                expert=self.name,
                confidence=1.0,
            ))
            if image.alt_text:
                candidates.extend(self._fact_candidates(image.alt_text, region_id))
        return ExpertOutput(
            expert=self.name,
            channel=self.channel,
            regions=regions,
            candidates=candidates,
        )

    def _fact_candidates(self, text: str, region_id: str) -> list[CandidateValue]:
        return [
            CandidateValue(
                field=fact["field"],
                value=fact["value"],
                expert=self.name,
                channel=self.channel,
                confidence=0.96,
                evidence_text=fact.get("evidence_quote", text),
                region_id=region_id,
                weight=self.weight,
            )
            for fact in LocalOcrEngine.extract_facts(text)
        ]


class LocalImageOcrExpert:
    name = "local-image-ocr"
    channel = "visual"
    weight = 0.95

    def __init__(
        self,
        *,
        limit: int = 2,
        min_text_likelihood: float = 0.55,
        cache_dir: Path | None = None,
    ):
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not 0.0 <= min_text_likelihood <= 1.0:
            raise ValueError("min_text_likelihood must be between 0 and 1")
        self.limit = limit
        self.min_text_likelihood = min_text_likelihood
        self._ocr_cache: dict[str, tuple[str, list[OcrSpan]]] = {}
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _read_cached_ocr(self, digest: str) -> tuple[str, list[OcrSpan]] | None:
        if digest in self._ocr_cache:
            return self._ocr_cache[digest]
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{digest}.layout-v2.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("version") != 2:
                return None
            text = str(payload.get("text", ""))
            spans = [
                OcrSpan(
                    text=str(item["text"]),
                    confidence=float(item["confidence"]),
                    bbox=tuple(float(value) for value in item["bbox"]),
                )
                for item in payload.get("spans", [])
                if len(item.get("bbox", [])) == 4
            ]
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        self._ocr_cache[digest] = (text, spans)
        return text, spans

    def _store_cached_ocr(self, digest: str, text: str, spans: list[OcrSpan]) -> None:
        self._ocr_cache[digest] = (text, spans)
        if self.cache_dir is None:
            return
        target = self.cache_dir / f"{digest}.layout-v2.json"
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{digest[:16]}-",
            suffix=".tmp",
            dir=self.cache_dir,
        )
        os.close(fd)
        staged = Path(raw_path)
        try:
            staged.write_text(json.dumps({
                "version": 2,
                "text": text,
                "spans": [
                    {
                        "text": span.text,
                        "confidence": span.confidence,
                        "bbox": list(span.bbox),
                    }
                    for span in spans
                ],
            }, ensure_ascii=False), encoding="utf-8")
            os.replace(staged, target)
        finally:
            staged.unlink(missing_ok=True)

    def analyze(
        self,
        parsed: ParsedDocument,
        source_path: Path,
        document_id: str,
    ) -> ExpertOutput:
        del document_id
        regions: list[DocumentRegion] = []
        candidates: list[CandidateValue] = []
        errors: list[str] = []
        visual_candidates = extract_visual_candidates(
            source_path,
            limit=self.limit,
            parsed=parsed,
        )
        for visual in visual_candidates:
            if visual.text_likelihood < self.min_text_likelihood:
                continue
            try:
                cached = self._read_cached_ocr(visual.digest)
                if cached is None:
                    visible_text, spans = LocalOcrEngine.ocr_with_layout(
                        visual.data,
                        suffix=Path(visual.media_path).suffix or ".png",
                    )
                    self._store_cached_ocr(visual.digest, visible_text, spans)
                else:
                    visible_text, spans = cached
            except Exception as exc:
                errors.append(f"{visual.media_path}: {exc}")
                continue
            if not visible_text.strip():
                continue

            region_id = f"{self.name}:IMG:{visual.digest[:16]}"
            confidence = round(
                min(0.98, 0.82 + 0.16 * visual.text_likelihood),
                4,
            )
            image_indices = list(visual.image_indices)
            regions.append(DocumentRegion(
                id=region_id,
                kind="image",
                content=visible_text.strip(),
                image_index=image_indices[0] if image_indices else None,
                image_indices=image_indices,
                media_path=visual.media_path,
                story=visual.stories[0] if len(visual.stories) == 1 else None,
                occurrence_count=visual.occurrence_count,
                bbox=(0.0, 0.0, 1.0, 1.0),
                coordinate_space="image",
                expert=self.name,
                confidence=confidence,
            ))
            reconstructed = reconstruct_ocr_table(spans)
            if reconstructed is not None:
                regions.append(DocumentRegion(
                    id=f"{self.name}:TABLE:{visual.digest[:16]}",
                    kind="disease_table",
                    content="\n".join(" | ".join(row) for row in reconstructed.rows),
                    rows=reconstructed.rows,
                    image_index=image_indices[0] if image_indices else None,
                    image_indices=image_indices,
                    media_path=visual.media_path,
                    story=visual.stories[0] if len(visual.stories) == 1 else None,
                    occurrence_count=visual.occurrence_count,
                    bbox=reconstructed.bbox,
                    coordinate_space="image",
                    expert=self.name,
                    confidence=round(min(confidence, reconstructed.confidence), 4),
                ))
            for fact in LocalOcrEngine.extract_facts(visible_text):
                candidates.append(CandidateValue(
                    field=fact["field"],
                    value=fact["value"],
                    expert=self.name,
                    channel=self.channel,
                    confidence=confidence,
                    evidence_text=fact.get("evidence_quote", visible_text),
                    region_id=region_id,
                    weight=self.weight,
                ))
        return ExpertOutput(
            expert=self.name,
            channel=self.channel,
            regions=regions,
            candidates=candidates,
            errors=errors,
        )


class JsonCommandExpert:
    channel = "visual"

    def __init__(
        self,
        name: str,
        command_template: str,
        *,
        timeout_seconds: int = 600,
        weight: float = 1.0,
    ):
        if not name.strip():
            raise ValueError("expert name must not be empty")
        if "{input}" not in command_template:
            raise ValueError("document expert command must contain {input}")
        self.name = name.strip()
        self.command_template = command_template
        self.timeout_seconds = timeout_seconds
        self.weight = weight

    def analyze(
        self,
        parsed: ParsedDocument,
        source_path: Path,
        document_id: str,
    ) -> ExpertOutput:
        del parsed, document_id
        command = [
            token.replace("{input}", str(source_path))
            for token in shlex.split(self.command_template)
        ]
        process = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
            check=False,
            shell=False,
        )
        if process.returncode:
            detail = process.stderr.strip() or process.stdout.strip() or f"exit code {process.returncode}"
            raise RuntimeError(f"document expert {self.name} failed: {detail}")
        try:
            payload = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"document expert {self.name} returned invalid JSON: {exc}") from exc

        region_ids: dict[str, str] = {}
        regions: list[DocumentRegion] = []
        for index, item in enumerate(payload.get("regions", [])):
            raw_id = str(item.get("id") or f"region-{index}")
            region_id = f"{self.name}:{raw_id}"
            region_ids[raw_id] = region_id
            region = DocumentRegion.model_validate({
                **item,
                "id": region_id,
                "expert": self.name,
            })
            regions.append(region)

        candidates = []
        for item in payload.get("candidates", []):
            raw_region_id = item.get("region_id")
            candidates.append(CandidateValue.model_validate({
                **item,
                "expert": self.name,
                "channel": self.channel,
                "region_id": region_ids.get(str(raw_region_id), raw_region_id),
                "weight": self.weight,
            }))
        return ExpertOutput(
            expert=self.name,
            channel=self.channel,
            regions=regions,
            candidates=candidates,
        )
