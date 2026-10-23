#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bridge_evidence.document_ai.vision_candidates import extract_field_candidates


def image_size(data: bytes) -> tuple[int, int] | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"\xff\xd8"):
        offset = 2
        while offset + 9 < len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                return struct.unpack(">HH", data[offset + 5 : offset + 9])[::-1]
            if offset + 4 > len(data):
                break
            length = struct.unpack(">H", data[offset + 2 : offset + 4])[0]
            offset += max(length + 2, 2)
    return None


def ensure_binary() -> Path:
    configured = os.environ.get("APPLE_VISION_OCR_BIN")
    if configured:
        return Path(configured)
    source = Path(__file__).with_name("apple_vision_ocr.swift")
    target = ROOT / "work" / "document_ai" / "bin" / "apple_vision_ocr"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    subprocess.run(
        ["swiftc", "-O", "-framework", "Vision", "-framework", "ImageIO", str(source), "-o", str(temporary)],
        check=True,
        capture_output=True,
        text=True,
    )
    os.replace(temporary, target)
    return target


def selected_media(document: Path, *, limit: int) -> list[tuple[str, bytes]]:
    selected: list[tuple[int, str, bytes]] = []
    with zipfile.ZipFile(document) as archive:
        for name in archive.namelist():
            if not name.startswith("word/media/") or Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            data = archive.read(name)
            dimensions = image_size(data)
            if dimensions is None:
                continue
            width, height = dimensions
            area = width * height
            if area < 600_000 or min(width, height) < 450:
                continue
            selected.append((area, name, data))
    selected.sort(key=lambda item: (-item[0], item[1]))
    return [(name, data) for _, name, data in selected[:limit]]


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apple_vision_expert.py DOCUMENT.docx")
    document = Path(sys.argv[1])
    limit = max(1, int(os.environ.get("APPLE_VISION_MAX_IMAGES", "12")))
    media = selected_media(document, limit=limit)
    regions: list[dict] = []
    candidates: list[dict] = []
    errors: list[str] = []
    if not media:
        print(json.dumps({"expert": "apple_vision", "channel": "visual", "regions": [], "candidates": [], "errors": []}, ensure_ascii=False))
        return 0

    with tempfile.TemporaryDirectory(prefix="bridge-apple-vision-") as temp_dir:
        paths: list[Path] = []
        archive_names: dict[str, str] = {}
        for index, (archive_name, data) in enumerate(media):
            path = Path(temp_dir) / f"{index:03d}{Path(archive_name).suffix.lower()}"
            path.write_bytes(data)
            paths.append(path)
            archive_names[str(path)] = archive_name
        completed = subprocess.run(
            [str(ensure_binary()), *(str(path) for path in paths)],
            check=True,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("APPLE_VISION_TIMEOUT_SECONDS", "300")),
        )
        results = json.loads(completed.stdout)
        for index, result in enumerate(results):
            if result.get("error"):
                errors.append(f"{archive_names.get(result['path'], result['path'])}: {result['error']}")
                continue
            observations = result.get("observations", [])
            content = "\n".join(item["text"] for item in observations if item.get("text")).strip()
            if not content:
                continue
            confidence = sum(float(item.get("confidence", 0.0)) for item in observations) / len(observations)
            region_id = f"AVIMG{index}"
            region = {
                "id": region_id,
                "kind": "image",
                "content": content,
                "page": None,
                "bbox": None,
                "rows": [],
                "paragraph_index": None,
                "table_index": None,
                "expert": "apple_vision",
                "confidence": confidence,
            }
            regions.append(region)
            candidates.extend(
                extract_field_candidates(
                    content,
                    expert="apple_vision",
                    channel="visual",
                    region_id=region_id,
                    confidence_scale=max(0.55, confidence),
                )
            )
    print(
        json.dumps(
            {"expert": "apple_vision", "channel": "visual", "regions": regions, "candidates": candidates, "errors": errors},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
