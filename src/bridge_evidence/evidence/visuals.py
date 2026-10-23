from __future__ import annotations

import base64
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from bridge_evidence.parsers.docx import ParsedDocument

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class VisualCandidate:
    media_path: str
    digest: str
    mime_type: str
    width: int
    height: int
    byte_size: int
    text_likelihood: float
    data: bytes
    occurrence_count: int = 1
    image_indices: tuple[int, ...] = ()
    stories: tuple[str, ...] = ()
    alt_texts: tuple[str, ...] = ()

    @property
    def data_url(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


@dataclass(frozen=True)
class OcrSpan:
    """OCR text with a normalized, top-left-origin image bounding box."""

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class OcrTable:
    """A table reconstructed conservatively from positioned OCR spans."""

    kind: str
    rows: list[list[str]]
    bbox: tuple[float, float, float, float]
    confidence: float


def _text_likelihood(image: Image.Image) -> float:
    sample = image.convert("L")
    sample.thumbnail((160, 160))
    histogram = sample.histogram()
    total = sum(histogram)
    if not total:
        return 0.0
    white = sum(histogram[225:]) / total
    dark = sum(histogram[:91]) / total
    # Tables/scanned pages combine a light background with dark glyphs and
    # lines.  Natural defect photos rarely occupy this region of the feature
    # space.  This is only candidate selection, never a factual decision.
    return min(1.0, 0.75 * white + 2.5 * min(dark, 0.10))


def extract_visual_candidates(
    docx_path: Path,
    *,
    limit: int = 3,
    parsed: ParsedDocument | None = None,
) -> list[VisualCandidate]:
    """Return a small, content-general set of likely OCR images from DOCX."""

    if limit <= 0 or not zipfile.is_zipfile(docx_path):
        return []
    candidates: list[VisualCandidate] = []
    seen: set[str] = set()
    occurrences: dict[str, list[object]] = {}
    if parsed is not None:
        for image in parsed.images:
            occurrences.setdefault(image.target.lstrip("/"), []).append(image)
    with zipfile.ZipFile(docx_path) as archive:
        for info in archive.infolist():
            suffix = Path(info.filename).suffix.lower()
            if not info.filename.startswith("word/media/") or suffix not in MIME_TYPES:
                continue
            if info.file_size < 8_000 or info.file_size > 12_000_000:
                continue
            data = archive.read(info.filename)
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            try:
                with Image.open(io.BytesIO(data)) as image:
                    width, height = image.size
                    if width < 260 or height < 120:
                        continue
                    likelihood = _text_likelihood(image)
            except Exception:
                continue
            image_occurrences = occurrences.get(info.filename, [])
            candidates.append(VisualCandidate(
                media_path=info.filename,
                digest=digest,
                mime_type=MIME_TYPES[suffix],
                width=width,
                height=height,
                byte_size=len(data),
                text_likelihood=likelihood,
                data=data,
                occurrence_count=max(1, len(image_occurrences)),
                image_indices=tuple(
                    int(image.index) for image in image_occurrences
                ),
                stories=tuple(sorted({
                    str(image.story) for image in image_occurrences
                })),
                alt_texts=tuple(sorted({
                    str(image.alt_text)
                    for image in image_occurrences
                    if image.alt_text
                })),
            ))

    candidates.sort(
        key=lambda item: (
            -item.text_likelihood,
            -(item.width * item.height),
            item.media_path,
        )
    )
    return candidates[:limit]


_SWIFT_OCR_SOURCE = """
import Foundation
import Vision
import AppKit

struct Span: Codable {
    let text: String
    let confidence: Float
    let bbox: [CGFloat]
}

guard CommandLine.arguments.count > 1 else { exit(1) }
let imagePath = CommandLine.arguments[1]
guard let image = NSImage(contentsOfFile: imagePath),
      let tiffData = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiffData),
      let cgImage = bitmap.cgImage else {
    exit(2)
}

let request = VNRecognizeTextRequest()
request.recognitionLanguages = ["zh-Hans", "en-US"]
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try? handler.perform([request])

guard let results = request.results else { exit(0) }
let encoder = JSONEncoder()
for observation in results {
    if let top = observation.topCandidates(1).first {
        let box = observation.boundingBox
        let topY = 1.0 - box.origin.y - box.height
        let span = Span(
            text: top.string,
            confidence: top.confidence,
            bbox: [box.origin.x, topY, box.origin.x + box.width, topY + box.height]
        )
        if let data = try? encoder.encode(span),
           let line = String(data: data, encoding: .utf8) {
            print(line)
        }
    }
}
"""


class LocalOcrEngine:
    """Local offline OCR engine using macOS Native Vision Framework or pytesseract fallback."""

    _ocr_bin: str | None = None
    _initialized: bool = False

    @classmethod
    def _get_apple_ocr_bin(cls) -> str | None:
        import os
        import platform
        import subprocess

        if platform.system() != "Darwin":
            return None
        cache_dir = Path.home() / ".cache" / "bridge_evidence"
        cache_dir.mkdir(parents=True, exist_ok=True)
        bin_path = cache_dir / "apple_ocr_v2"
        if bin_path.exists() and os.access(bin_path, os.X_OK):
            return str(bin_path)
        swift_file = cache_dir / "apple_ocr_v2.swift"
        swift_file.write_text(_SWIFT_OCR_SOURCE, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["swiftc", str(swift_file), "-o", str(bin_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0 and bin_path.exists():
                return str(bin_path)
        except Exception:
            pass
        return None

    @classmethod
    def ocr_with_layout(
        cls,
        image_data: bytes,
        suffix: str = ".png",
    ) -> tuple[str, list[OcrSpan]]:
        """Extract text and normalized image coordinates without network calls."""
        import json
        import subprocess
        import tempfile

        if not cls._initialized:
            cls._ocr_bin = cls._get_apple_ocr_bin()
            cls._initialized = True

        if cls._ocr_bin:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(image_data)
                tmp_path = tmp.name
            try:
                proc = subprocess.run(
                    [cls._ocr_bin, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if proc.returncode == 0:
                    spans = []
                    for line in proc.stdout.splitlines():
                        try:
                            payload = json.loads(line)
                            bbox = tuple(float(value) for value in payload["bbox"])
                            if len(bbox) != 4:
                                continue
                            spans.append(OcrSpan(
                                text=str(payload["text"]).strip(),
                                confidence=float(payload["confidence"]),
                                bbox=bbox,
                            ))
                        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                            continue
                    spans = [span for span in spans if span.text]
                    spans.sort(key=lambda span: (span.bbox[1], span.bbox[0]))
                    if spans:
                        return "\n".join(span.text for span in spans), spans
            except Exception:
                pass
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # Fallback to pytesseract if installed
        try:
            import pytesseract
            image = Image.open(io.BytesIO(image_data))
            payload = pytesseract.image_to_data(
                image,
                lang="chi_sim+eng",
                output_type=pytesseract.Output.DICT,
            )
            width, height = image.size
            spans = []
            for index, raw_text in enumerate(payload["text"]):
                text = str(raw_text).strip()
                confidence = float(payload["conf"][index])
                if not text or confidence < 0:
                    continue
                left = float(payload["left"][index]) / width
                top = float(payload["top"][index]) / height
                right = (float(payload["left"][index]) + float(payload["width"][index])) / width
                bottom = (float(payload["top"][index]) + float(payload["height"][index])) / height
                spans.append(OcrSpan(
                    text=text,
                    confidence=confidence / 100.0,
                    bbox=(left, top, right, bottom),
                ))
            spans.sort(key=lambda span: (span.bbox[1], span.bbox[0]))
            return "\n".join(span.text for span in spans), spans
        except Exception:
            pass

        return "", []

    @classmethod
    def ocr_image(cls, image_data: bytes, suffix: str = ".png") -> str:
        """Backward-compatible plain-text OCR facade."""

        return cls.ocr_with_layout(image_data, suffix=suffix)[0]

    @classmethod
    def extract_facts(cls, visible_text: str) -> list[dict[str, str]]:
        """Parse structured inspection facts (scores, dates, bridge name) from OCR text."""
        import re

        facts: list[dict[str, str]] = []
        lines = [line.strip() for line in visible_text.splitlines() if line.strip()]

        # 1. Overall BCI / Score
        bci_match = re.search(r"(?:总体|全桥)?\s*(?:BCI|Dr|评分|分值)\s*[:=为\s]*([0-9]{2,3}(?:\.[0-9]{1,4})?)", visible_text, re.I)
        if bci_match:
            facts.append({
                "field": "scores.overall.score",
                "value": bci_match.group(1),
                "evidence_quote": bci_match.group(0),
            })

        # 2. Overall Grade
        grade_match = re.search(r"(?:总体|全桥)?\s*(?:技术状况)?\s*(?:等级|评定为)\s*[:=为\s]*([ABCDE一二三四五])\s*(?:级|类)?", visible_text, re.I)
        if grade_match:
            facts.append({
                "field": "scores.overall.grade",
                "value": f"{grade_match.group(1).upper()}级",
                "evidence_quote": grade_match.group(0),
            })

        # 3. Deck BCIm
        bcim_match = re.search(r"(?:桥面系|BCIm)\s*[:=为\s]*([0-9]{2,3}(?:\.[0-9]{1,4})?)", visible_text, re.I)
        if bcim_match:
            facts.append({
                "field": "scores.deck.score",
                "value": bcim_match.group(1),
                "evidence_quote": bcim_match.group(0),
            })

        # 4. Superstructure BCIs
        bcis_match = re.search(r"(?:上部结构|BCIs)\s*[:=为\s]*([0-9]{2,3}(?:\.[0-9]{1,4})?)", visible_text, re.I)
        if bcis_match:
            facts.append({
                "field": "scores.superstructure.score",
                "value": bcis_match.group(1),
                "evidence_quote": bcis_match.group(0),
            })

        # 5. Substructure BCIx
        bcix_match = re.search(r"(?:下部结构|BCIx)\s*[:=为\s]*([0-9]{2,3}(?:\.[0-9]{1,4})?)", visible_text, re.I)
        if bcix_match:
            facts.append({
                "field": "scores.substructure.score",
                "value": bcix_match.group(1),
                "evidence_quote": bcix_match.group(0),
            })

        # 6. Report Date
        date_match = re.search(r"((?:20\d{2}|19\d{2})\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?)", visible_text)
        if date_match:
            facts.append({
                "field": "bridge.report_date",
                "value": date_match.group(1).replace(" ", ""),
                "evidence_quote": date_match.group(0),
            })

        # 7. Bridge Name
        name_match = re.search(r"(?:工程名称|桥梁名称|桥名)\s*[:：\s]*([^\s\n，,。；;]{2,30}(?:大桥|特大桥|中桥|小桥|天桥|通道|互通|立交|高架|桥))", visible_text)
        if name_match:
            facts.append({
                "field": "bridge.name",
                "value": name_match.group(1).strip(),
                "evidence_quote": name_match.group(0),
            })

        # 8. Previous Score & Grade
        prev_bci_match = re.search(r"(?:上次|上一次|前次|历次|20\d{2}年(?:度)?).{0,30}?(?:总体|全桥)?\s*(?:BCI|Dr|评分|分值)\s*[:=为\s]*([0-9]{2,3}(?:\.[0-9]{1,4})?)", visible_text, re.I)
        if prev_bci_match:
            facts.append({
                "field": "scores.previous.score",
                "value": prev_bci_match.group(1),
                "evidence_quote": prev_bci_match.group(0),
            })
        prev_grade_match = re.search(r"(?:上次|上一次|前次|历次|20\d{2}年(?:度)?).{0,30}?(?:技术状况)?\s*(?:等级|评定为)\s*[:=为\s]*([ABCDE一二三四五])\s*(?:级|类)?", visible_text, re.I)
        if prev_grade_match:
            facts.append({
                "field": "scores.previous.grade",
                "value": f"{prev_grade_match.group(1).upper().replace('类', '级')}级" if "级" not in prev_grade_match.group(1) else prev_grade_match.group(1),
                "evidence_quote": prev_grade_match.group(0),
            })

        return facts


def _ocr_line_groups(spans: list[OcrSpan]) -> list[list[OcrSpan]]:
    usable = [span for span in spans if span.text.strip()]
    if not usable:
        return []
    heights = sorted(max(0.001, span.bbox[3] - span.bbox[1]) for span in usable)
    tolerance = max(0.018, heights[len(heights) // 2] * 0.75)
    lines: list[list[OcrSpan]] = []
    centers: list[float] = []
    for span in sorted(
        usable,
        key=lambda item: ((item.bbox[1] + item.bbox[3]) / 2, item.bbox[0]),
    ):
        center_y = (span.bbox[1] + span.bbox[3]) / 2
        if not lines or abs(center_y - centers[-1]) > tolerance:
            lines.append([span])
            centers.append(center_y)
            continue
        lines[-1].append(span)
        count = len(lines[-1])
        centers[-1] = ((centers[-1] * (count - 1)) + center_y) / count
    return [sorted(line, key=lambda item: item.bbox[0]) for line in lines]


def reconstruct_ocr_table(spans: list[OcrSpan]) -> OcrTable | None:
    """Reconstruct a disease table only when all required headers are visible."""

    lines = _ocr_line_groups(spans)
    location_terms = ("病害位置", "构件位置", "病害部位", "构件部位")
    type_terms = ("病害类型", "病害种类", "损坏类型", "病害名称")
    description_terms = ("病害描述", "具体情况", "检查结果", "检测结果")
    header_index = -1
    header: list[OcrSpan] = []
    for index, line in enumerate(lines):
        cells = ["".join(item.text.split()) for item in line]
        has_location = any(any(term in cell for term in location_terms) for cell in cells)
        has_type = any(any(term in cell for term in type_terms) for cell in cells)
        has_description = any(any(term in cell for term in description_terms) for cell in cells)
        if has_location and has_type and has_description and len(line) >= 3:
            header_index = index
            header = line
            break
    if header_index < 0:
        return None

    column_centers = [(item.bbox[0] + item.bbox[2]) / 2 for item in header]
    rows: list[list[str]] = [[item.text.strip() for item in header]]
    used = list(header)
    for line in lines[header_index + 1 :]:
        assigned: list[list[OcrSpan]] = [[] for _ in header]
        for span in line:
            center_x = (span.bbox[0] + span.bbox[2]) / 2
            column = min(
                range(len(column_centers)),
                key=lambda item: abs(center_x - column_centers[item]),
            )
            assigned[column].append(span)
        row = ["".join(item.text.strip() for item in cell) for cell in assigned]
        if sum(bool(cell) for cell in row) < 2:
            continue
        rows.append(row)
        used.extend(item for cell in assigned for item in cell)
    if len(rows) < 2:
        return None

    bbox = (
        min(item.bbox[0] for item in used),
        min(item.bbox[1] for item in used),
        max(item.bbox[2] for item in used),
        max(item.bbox[3] for item in used),
    )
    confidence = sum(item.confidence for item in used) / len(used)
    return OcrTable(
        kind="disease_table",
        rows=rows,
        bbox=bbox,
        confidence=confidence,
    )
