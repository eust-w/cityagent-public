from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from PIL import Image, ImageChops, ImageFilter, ImageStat


OLE2_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
WORD_FIB_IDENTIFIERS = {0xA5EC, 0xA5DC}
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
V_NS = "urn:schemas-microsoft-com:vml"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
MAX_DOCX_ENTRIES = 20_000
MAX_DOCX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_DOCX_PART_BYTES = 512 * 1024 * 1024
MAX_COMPRESSION_RATIO = 2_000


@dataclass(frozen=True)
class FibBase:
    valid: bool
    identifier: int | None = None
    version: int | None = None
    encrypted: bool = False
    obfuscated: bool = False


@dataclass(frozen=True)
class LegacyDocPreflight:
    valid_ole: bool
    has_word_stream: bool
    encrypted: bool
    obfuscated: bool
    has_macros: bool
    embedded_objects: int
    stream_count: int
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentFingerprint:
    provider: str
    text_chars: int = 0
    paragraphs: int = 0
    tables: int = 0
    rows: int = 0
    cells: int = 0
    image_occurrences: int = 0
    unique_images: int = 0
    story_parts: tuple[str, ...] = ()
    external_relationships: int = 0
    has_macros: bool = False
    embedded_objects: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.provider != "unavailable"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AxisResult:
    name: str
    score: float | None
    accepted: bool = True
    requires_review: bool = False
    warnings: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DeepAuditResult:
    structure: AxisResult
    visual: AxisResult
    source_fingerprint: DocumentFingerprint | None = None
    target_fingerprint: DocumentFingerprint | None = None

    @property
    def accepted(self) -> bool:
        return self.structure.accepted and self.visual.accepted

    @property
    def requires_review(self) -> bool:
        return self.structure.requires_review or self.visual.requires_review

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.structure.warnings, *self.visual.warnings)))

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "requires_review": self.requires_review,
            "structure": self.structure.to_dict(),
            "visual": self.visual.to_dict(),
            "source_fingerprint": (
                self.source_fingerprint.to_dict() if self.source_fingerprint else None
            ),
            "target_fingerprint": (
                self.target_fingerprint.to_dict() if self.target_fingerprint else None
            ),
        }


def _visible_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and unicodedata.category(character) not in {"Cc", "Cf"}
    )


def parse_fib_base(data: bytes) -> FibBase:
    if len(data) < 12:
        return FibBase(valid=False)
    identifier = int.from_bytes(data[0:2], "little")
    version = int.from_bytes(data[2:4], "little")
    flags = int.from_bytes(data[10:12], "little")
    return FibBase(
        valid=identifier in WORD_FIB_IDENTIFIERS,
        identifier=identifier,
        version=version,
        encrypted=bool(flags & (1 << 8)),
        obfuscated=bool(flags & (1 << 15)),
    )


def preflight_legacy_doc(path: Path) -> LegacyDocPreflight:
    with path.open("rb") as stream:
        magic = stream.read(len(OLE2_MAGIC))
    if magic != OLE2_MAGIC:
        raise ValueError(f"Legacy DOC is not an OLE2 compound document: {path.name}")

    try:
        import olefile
    except ImportError:
        return LegacyDocPreflight(
            valid_ole=True,
            has_word_stream=True,
            encrypted=False,
            obfuscated=False,
            has_macros=False,
            embedded_objects=0,
            stream_count=0,
            warnings=("olefile_unavailable_deep_preflight_skipped",),
        )

    try:
        with olefile.OleFileIO(str(path)) as compound:
            entries = ["/".join(parts) for parts in compound.listdir()]
            lower_entries = [entry.lower() for entry in entries]
            has_word_stream = compound.exists("WordDocument")
            if not has_word_stream:
                raise ValueError(f"OLE2 file has no WordDocument stream: {path.name}")
            fib = parse_fib_base(compound.openstream("WordDocument").read(32))
            has_macros = any(
                "vba" in entry or "macros" in entry or "_vba_project" in entry
                for entry in lower_entries
            )
            embedded_objects = sum(
                1
                for entry in lower_entries
                if "objectpool" in entry or "ole10native" in entry
            )
            warnings = () if fib.valid else ("unrecognized_word_fib_identifier",)
            return LegacyDocPreflight(
                valid_ole=True,
                has_word_stream=True,
                encrypted=fib.encrypted,
                obfuscated=fib.obfuscated,
                has_macros=has_macros,
                embedded_objects=embedded_objects,
                stream_count=len(entries),
                warnings=warnings,
            )
    except OSError as exc:
        raise ValueError(f"Invalid OLE2 compound document {path.name}: {exc}") from exc


class _WvHtmlFingerprintParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables = 0
        self.rows = 0
        self.cells = 0
        self.images = 0
        self.paragraphs = 0
        self._ignored_depth = 0
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        elif tag == "table":
            self.tables += 1
        elif tag == "tr":
            self.rows += 1
        elif tag in {"td", "th"}:
            self.cells += 1
        elif tag == "img":
            self.images += 1
        elif tag in {"p", "div", "li"}:
            self.paragraphs += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.text.append(data)


def fingerprint_wv_html(html: str) -> DocumentFingerprint:
    parser = _WvHtmlFingerprintParser()
    parser.feed(html)
    parser.close()
    return DocumentFingerprint(
        provider="wvHtml",
        text_chars=len(_visible_text("".join(parser.text))),
        paragraphs=parser.paragraphs,
        tables=parser.tables,
        rows=parser.rows,
        cells=parser.cells,
        image_occurrences=parser.images,
    )


def fingerprint_legacy_doc(path: Path, temp_dir: Path) -> DocumentFingerprint:
    wvhtml = shutil.which("wvHtml")
    if not wvhtml:
        return DocumentFingerprint(
            provider="unavailable",
            warnings=("wvhtml_unavailable_source_structure_not_verified",),
        )
    output_name = "source-fingerprint.html"
    process = subprocess.run(
        [wvhtml, str(path), output_name],
        cwd=temp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    output_path = temp_dir / output_name
    if process.returncode or not output_path.exists():
        detail = process.stdout.decode("utf-8", errors="replace").strip()
        return DocumentFingerprint(
            provider="unavailable",
            warnings=(f"wvhtml_source_fingerprint_failed:{detail[:160]}",),
        )
    return fingerprint_wv_html(
        output_path.read_text(encoding="utf-8", errors="replace")
    )


def _story_for_part(name: str) -> str | None:
    if name == "word/document.xml":
        return "body"
    if re.fullmatch(r"word/header\d+\.xml", name):
        return "header"
    if re.fullmatch(r"word/footer\d+\.xml", name):
        return "footer"
    for story in ("footnotes", "endnotes", "comments"):
        if name == f"word/{story}.xml":
            return story
    return None


def _validate_zip_limits(archive: zipfile.ZipFile) -> None:
    entries = archive.infolist()
    if len(entries) > MAX_DOCX_ENTRIES:
        raise ValueError(f"DOCX has too many ZIP entries: {len(entries)}")
    total = sum(entry.file_size for entry in entries)
    if total > MAX_DOCX_UNCOMPRESSED_BYTES:
        raise ValueError(f"DOCX uncompressed size exceeds limit: {total}")
    for entry in entries:
        if entry.file_size > MAX_DOCX_PART_BYTES:
            raise ValueError(f"DOCX part exceeds size limit: {entry.filename}")
        compressed = max(1, entry.compress_size)
        if entry.file_size / compressed > MAX_COMPRESSION_RATIO:
            raise ValueError(f"DOCX suspicious compression ratio: {entry.filename}")


def fingerprint_docx(path: Path) -> DocumentFingerprint:
    paragraphs = tables = rows = cells = images = external = embedded = 0
    text_parts: list[str] = []
    stories: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        _validate_zip_limits(archive)
        names = set(archive.namelist())
        for name in sorted(names):
            story = _story_for_part(name)
            if story is not None:
                stories.add(story)
                root = ElementTree.fromstring(archive.read(name))
                paragraphs += sum(1 for _ in root.iter(f"{{{W_NS}}}p"))
                tables += sum(1 for _ in root.iter(f"{{{W_NS}}}tbl"))
                rows += sum(1 for _ in root.iter(f"{{{W_NS}}}tr"))
                cells += sum(1 for _ in root.iter(f"{{{W_NS}}}tc"))
                images += sum(1 for _ in root.iter(f"{{{A_NS}}}blip"))
                images += sum(1 for _ in root.iter(f"{{{V_NS}}}imagedata"))
                text_parts.extend(
                    node.text or "" for node in root.iter(f"{{{W_NS}}}t")
                )
            if name.endswith(".rels"):
                root = ElementTree.fromstring(archive.read(name))
                external += sum(
                    1
                    for relation in root.iter(f"{{{REL_NS}}}Relationship")
                    if relation.attrib.get("TargetMode") == "External"
                )
        unique_images = sum(
            1
            for name in names
            if name.startswith("word/media/") and not name.endswith("/")
        )
        embedded = sum(
            1
            for name in names
            if name.startswith("word/embeddings/") and not name.endswith("/")
        )
        has_macros = "word/vbaProject.bin" in names
    return DocumentFingerprint(
        provider="ooxml",
        text_chars=len(_visible_text("".join(text_parts))),
        paragraphs=paragraphs,
        tables=tables,
        rows=rows,
        cells=cells,
        image_occurrences=images,
        unique_images=unique_images,
        story_parts=tuple(sorted(stories)),
        external_relationships=external,
        has_macros=has_macros,
        embedded_objects=embedded,
    )


def _retention(target: int, source: int) -> float:
    if source <= 0:
        return 1.0
    return min(1.0, target / source)


def compare_structure(
    source: DocumentFingerprint,
    target: DocumentFingerprint,
) -> AxisResult:
    if not source.available:
        return AxisResult(
            name="structure",
            score=None,
            requires_review=True,
            warnings=source.warnings or ("source_structure_unavailable",),
            details={
                "source": source.to_dict(),
                "target": target.to_dict(),
            },
        )

    warnings: list[str] = []
    accepted = True
    table_score = _retention(target.tables, source.tables)
    row_score = _retention(target.rows, source.rows)
    cell_score = _retention(target.cells, source.cells)
    image_score = _retention(target.image_occurrences, source.image_occurrences)
    text_score = _retention(target.text_chars, source.text_chars)

    if source.tables and not target.tables:
        warnings.append("source_tables_missing_after_conversion")
        accepted = False
    elif source.rows and row_score < 0.5:
        warnings.append("source_table_rows_severely_reduced")
        accepted = False
    elif source.cells and cell_score < 0.8:
        warnings.append("source_table_cells_reduced")

    if source.image_occurrences and not target.image_occurrences:
        warnings.append("source_images_missing_after_conversion")
        accepted = False
    elif source.image_occurrences and image_score < 0.5:
        warnings.append("source_image_occurrences_severely_reduced")
        accepted = False
    elif source.image_occurrences and image_score < 1.0:
        warnings.append("source_image_occurrences_reduced")

    if source.text_chars >= 64 and text_score < 0.5:
        warnings.append("source_structural_text_severely_reduced")
        accepted = False
    if target.external_relationships:
        warnings.append("docx_contains_external_relationships")

    scores = [text_score]
    if source.tables:
        scores.extend([table_score, row_score, cell_score])
    if source.image_occurrences:
        scores.append(image_score)
    score = round(sum(scores) / len(scores), 4)
    if score < 0.95 and accepted:
        warnings.append("structure_retention_below_full")
    return AxisResult(
        name="structure",
        score=score,
        accepted=accepted,
        requires_review=bool(warnings) or score < 0.95,
        warnings=tuple(dict.fromkeys(warnings)),
        details={
            "source": source.to_dict(),
            "target": target.to_dict(),
        },
    )


def _page_similarity(first: Image.Image, second: Image.Image) -> float:
    size = (160, 224)
    first_gray = first.convert("L").resize(size, Image.Resampling.LANCZOS)
    second_gray = second.convert("L").resize(size, Image.Resampling.LANCZOS)
    gray_difference = ImageStat.Stat(ImageChops.difference(first_gray, second_gray)).mean[0]
    gray_score = 1.0 - gray_difference / 255.0
    first_edges = first_gray.filter(ImageFilter.FIND_EDGES)
    second_edges = second_gray.filter(ImageFilter.FIND_EDGES)
    edge_difference = ImageStat.Stat(ImageChops.difference(first_edges, second_edges)).mean[0]
    edge_score = 1.0 - edge_difference / 255.0
    return max(0.0, min(1.0, 0.75 * gray_score + 0.25 * edge_score))


def _align_page_sequences(
    source: list[Image.Image],
    target: list[Image.Image],
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Align nearby pages while preventing unrelated large sequence shifts."""
    source_count = len(source)
    target_count = len(target)
    window = abs(source_count - target_count) + 2
    gap_cost = 0.08
    infinity = float("inf")
    costs = [[infinity] * (target_count + 1) for _ in range(source_count + 1)]
    previous: list[list[tuple[int, int, str] | None]] = [
        [None] * (target_count + 1) for _ in range(source_count + 1)
    ]
    similarities: dict[tuple[int, int], float] = {}
    costs[0][0] = 0.0

    for source_index in range(source_count + 1):
        target_start = max(0, source_index - window)
        target_end = min(target_count, source_index + window)
        for target_index in range(target_start, target_end + 1):
            if source_index == 0 and target_index == 0:
                continue
            candidates: list[tuple[float, int, tuple[int, int, str]]] = []
            if source_index and target_index:
                key = (source_index - 1, target_index - 1)
                similarity = similarities.setdefault(
                    key,
                    _page_similarity(source[key[0]], target[key[1]]),
                )
                candidates.append(
                    (
                        costs[source_index - 1][target_index - 1] + 1.0 - similarity,
                        0,
                        (source_index - 1, target_index - 1, "match"),
                    )
                )
            if source_index:
                candidates.append(
                    (
                        costs[source_index - 1][target_index] + gap_cost,
                        1,
                        (source_index - 1, target_index, "skip_source"),
                    )
                )
            if target_index:
                candidates.append(
                    (
                        costs[source_index][target_index - 1] + gap_cost,
                        2,
                        (source_index, target_index - 1, "skip_target"),
                    )
                )
            best_cost, _, best_previous = min(candidates, key=lambda item: (item[0], item[1]))
            if best_cost < infinity:
                costs[source_index][target_index] = best_cost
                previous[source_index][target_index] = best_previous

    matched: list[tuple[int, int, float]] = []
    unmatched_source: list[int] = []
    unmatched_target: list[int] = []
    source_index = source_count
    target_index = target_count
    while source_index or target_index:
        step = previous[source_index][target_index]
        if step is None:
            unmatched_source.extend(range(source_index))
            unmatched_target.extend(range(target_index))
            break
        next_source, next_target, operation = step
        if operation == "match":
            key = (source_index - 1, target_index - 1)
            matched.append((key[0], key[1], similarities[key]))
        elif operation == "skip_source":
            unmatched_source.append(source_index - 1)
        else:
            unmatched_target.append(target_index - 1)
        source_index, target_index = next_source, next_target

    matched.reverse()
    unmatched_source.reverse()
    unmatched_target.reverse()
    return matched, unmatched_source, unmatched_target


def compare_page_images(
    source_pages: Iterable[Image.Image],
    target_pages: Iterable[Image.Image],
) -> AxisResult:
    source = list(source_pages)
    target = list(target_pages)
    warnings: list[str] = []
    accepted = True
    if not source or not target:
        return AxisResult(
            name="visual",
            score=None,
            accepted=False,
            requires_review=True,
            warnings=tuple(warnings or ["visual_pages_unavailable"]),
            details={"source_pages": len(source), "target_pages": len(target)},
        )

    page_pairs: list[tuple[int, int, float]]
    unmatched_source: list[int] = []
    unmatched_target: list[int] = []
    if len(source) == len(target):
        page_pairs = [
            (index, index, _page_similarity(first, second))
            for index, (first, second) in enumerate(zip(source, target))
        ]
    else:
        warnings.extend(("visual_page_count_mismatch", "visual_page_alignment_applied"))
        page_pairs, unmatched_source, unmatched_target = _align_page_sequences(source, target)

    similarities: list[float] = []
    for source_index, target_index, similarity in page_pairs:
        first = source[source_index]
        second = target[target_index]
        first_ratio = first.width / max(1, first.height)
        second_ratio = second.width / max(1, second.height)
        if abs(first_ratio - second_ratio) > 0.02:
            warnings.append("visual_page_geometry_mismatch")
            accepted = False
        similarities.append(similarity)

    if not similarities:
        return AxisResult(
            name="visual",
            score=None,
            accepted=False,
            requires_review=True,
            warnings=tuple(dict.fromkeys(warnings + ["visual_pages_unaligned"])),
            details={"source_pages": len(source), "target_pages": len(target)},
        )

    minimum_score = min(similarities)
    mean_score = sum(similarities) / len(similarities)
    if len(source) == len(target):
        score = round(minimum_score, 4)
    else:
        percentile_index = max(0, int(len(similarities) * 0.05) - 1)
        lower_percentile_score = sorted(similarities)[percentile_index]
        score = round(mean_score, 4)
        maximum_count = max(len(source), len(target))
        count_delta_limit = max(2, (maximum_count * 3 + 99) // 100)
        unmatched_ratio = (len(unmatched_source) + len(unmatched_target)) / maximum_count
        source_coverage = len(page_pairs) / len(source)
        target_coverage = len(page_pairs) / len(target)
        if (
            abs(len(source) - len(target)) > count_delta_limit
            or unmatched_ratio > 0.15
            or source_coverage < 0.90
            or target_coverage < 0.90
            or mean_score < 0.95
        ):
            warnings.append("visual_page_alignment_insufficient")
            accepted = False
        if lower_percentile_score < 0.90:
            warnings.append("visual_local_reflow_detected")
    if score < 0.90:
        warnings.append("visual_content_severely_changed")
        accepted = False
    elif score < 0.985:
        warnings.append("visual_content_changed")
    return AxisResult(
        name="visual",
        score=score,
        accepted=accepted,
        requires_review=bool(warnings) or score < 0.985,
        warnings=tuple(dict.fromkeys(warnings)),
        details={
            "source_pages": len(source),
            "target_pages": len(target),
            "page_scores": [round(value, 4) for value in similarities],
            "minimum_page_score": round(minimum_score, 4),
            "mean_page_score": round(mean_score, 4),
            "lower_percentile_page_score": round(
                sorted(similarities)[max(0, int(len(similarities) * 0.05) - 1)],
                4,
            ),
            "matched_page_pairs": [
                [source_index, target_index]
                for source_index, target_index, _ in page_pairs
            ],
            "unmatched_source_pages": unmatched_source,
            "unmatched_target_pages": unmatched_target,
        },
    )


def _render_to_pdf(source: Path, output_dir: Path, soffice: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = output_dir / "profile"
    process = subprocess.run(
        [
            soffice,
            f"-env:UserInstallation={profile.resolve().as_uri()}",
            "--headless",
            "--convert-to",
            "pdf:writer_pdf_Export",
            "--outdir",
            str(output_dir),
            str(source),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
        check=False,
    )
    output = output_dir / f"{source.stem}.pdf"
    if process.returncode or not output.exists():
        raise RuntimeError(process.stdout.strip() or f"PDF render exit code {process.returncode}")
    return output


def _pdf_pages(path: Path) -> list[Image.Image]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("pypdfium2 is unavailable") from exc
    document = pdfium.PdfDocument(str(path))
    pages: list[Image.Image] = []
    try:
        for index in range(len(document)):
            page = document[index]
            try:
                bitmap = page.render(scale=1.0)
                pages.append(bitmap.to_pil().convert("RGB"))
            finally:
                page.close()
    finally:
        document.close()
    return pages


def visual_roundtrip(
    source_doc: Path,
    converted_docx: Path,
    temp_dir: Path,
    soffice: str,
) -> AxisResult:
    try:
        source_pdf = _render_to_pdf(source_doc, temp_dir / "source-pdf", soffice)
        target_pdf = _render_to_pdf(converted_docx, temp_dir / "target-pdf", soffice)
        return compare_page_images(_pdf_pages(source_pdf), _pdf_pages(target_pdf))
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
        return AxisResult(
            name="visual",
            score=None,
            requires_review=True,
            warnings=(f"visual_diff_unavailable:{str(exc)[:180]}",),
        )


def environment_fingerprint(details: dict[str, str]) -> str:
    payload = json.dumps(details, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _command_version(command: list[str]) -> str:
    if not command or not shutil.which(command[0]) and not Path(command[0]).exists():
        return "unavailable"
    try:
        process = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    first_line = process.stdout.strip().splitlines()
    return first_line[0][:240] if process.returncode == 0 and first_line else "unavailable"


def _font_set_fingerprint() -> str:
    fc_list = shutil.which("fc-list")
    if not fc_list:
        return "unavailable"
    try:
        process = subprocess.run(
            [fc_list, "--format=%{family}:%{style}\n"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    if process.returncode:
        return "unavailable"
    lines = sorted(set(process.stdout.splitlines()))
    return hashlib.sha256(b"\n".join(lines)).hexdigest()


def _find_soffice() -> str:
    candidates = [
        shutil.which("soffice"),
        shutil.which("libreoffice"),
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "/usr/bin/soffice",
        "/usr/bin/libreoffice",
        "/usr/local/bin/soffice",
        "/usr/local/bin/libreoffice",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError("LibreOffice/soffice is required for DOC visual quality auditing")


class WordQualityGate:
    def __init__(self, soffice: str | None = None, *, visual_enabled: bool = True):
        self.soffice = soffice or _find_soffice()
        self.visual_enabled = visual_enabled
        self.environment = {
            "soffice": _command_version([self.soffice, "--version"]),
            "wvText": _command_version([shutil.which("wvText") or "wvText", "--version"]),
            "wvHtml": _command_version([shutil.which("wvHtml") or "wvHtml", "--version"]),
            "pandoc": _command_version([shutil.which("pandoc") or "pandoc", "--version"]),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "fonts": _font_set_fingerprint(),
            "docx_filter": "Office Open XML Text",
            "pdf_filter": "writer_pdf_Export",
        }
        self.cache_fingerprint = environment_fingerprint(self.environment)

    @staticmethod
    def preflight(path: Path) -> LegacyDocPreflight:
        result = preflight_legacy_doc(path)
        if result.encrypted or result.obfuscated:
            raise RuntimeError(
                f"encrypted_document: {path.name} requires a password before conversion"
            )
        return result

    def assess(
        self,
        source_doc: Path,
        converted_docx: Path,
        temp_dir: Path,
    ) -> DeepAuditResult:
        structure_dir = temp_dir / "source-structure"
        structure_dir.mkdir(parents=True, exist_ok=True)
        source_fingerprint = fingerprint_legacy_doc(source_doc, structure_dir)
        target_fingerprint = fingerprint_docx(converted_docx)
        structure = compare_structure(source_fingerprint, target_fingerprint)
        visual = (
            visual_roundtrip(
                source_doc,
                converted_docx,
                temp_dir / "visual",
                self.soffice,
            )
            if self.visual_enabled
            else AxisResult(
                name="visual",
                score=None,
                requires_review=True,
                warnings=("visual_diff_disabled",),
            )
        )
        return DeepAuditResult(
            structure=structure,
            visual=visual,
            source_fingerprint=source_fingerprint,
            target_fingerprint=target_fingerprint,
        )
