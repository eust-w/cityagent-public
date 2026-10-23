#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bridge_evidence.document_ai.vision_candidates import ALLOWED_FIELDS, best_docx_page_numbers


def selected_page_images(document: Path, output_dir: Path, limit: int) -> list[tuple[int, Path]]:
    subprocess.run(
        ["/opt/homebrew/bin/soffice", "--headless", "--convert-to", "pdf", "--outdir", str(output_dir), str(document)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    pdf = next(output_dir.glob("*.pdf"))
    pages = best_docx_page_numbers(document, limit=limit)
    result: list[tuple[int, Path]] = []
    for page in pages:
        prefix = output_dir / f"gemini-page-{page:04d}"
        subprocess.run(
            ["pdftoppm", "-f", str(page), "-l", str(page), "-singlefile", "-jpeg", "-jpegopt", "quality=88", "-r", "144", str(pdf), str(prefix)],
            check=True,
            capture_output=True,
            timeout=180,
        )
        result.append((page, prefix.with_suffix(".jpg")))
    return result


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: gemini_document_expert.py DOCUMENT.docx")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is required")
    model = os.environ.get("GEMINI_DOCUMENT_MODEL", "gemini-2.5-flash")
    limit = max(1, int(os.environ.get("GEMINI_DOCUMENT_MAX_PAGES", "2")))
    document = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="bridge-gemini-") as temp_dir:
        images = selected_page_images(document, Path(temp_dir), limit)
        parts: list[dict] = [{
            "text": (
                "你是城市桥梁定检报告结构化专家。只依据给出的页面提取字段，不推测缺失值。"
                "fields 仅允许使用 bridge.name、bridge.report_date、scores.overall.score/grade、"
                "scores.superstructure.score/grade、scores.substructure.score/grade、scores.deck.score/grade、"
                "scores.previous.score/grade。双幅桥的左右幅值必须合并成一个原子值，严格使用"
                "X（左幅）/Y（右幅）格式；不得只返回其中一幅。每个字段给出原文证据、页码和0到1置信度。"
            )
        }]
        for page, image in images:
            parts.append({"text": f"PAGE={page}"})
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(image.read_bytes()).decode("ascii")}})
        schema = {
            "type": "object",
            "properties": {
                "regions": {"type": "array", "items": {"type": "object", "properties": {
                    "page": {"type": "integer"}, "kind": {"type": "string"}, "content": {"type": "string"}, "confidence": {"type": "number"}
                }, "required": ["page", "kind", "content", "confidence"]}},
                "fields": {"type": "array", "items": {"type": "object", "properties": {
                    "field": {"type": "string"}, "value": {"type": ["string", "number"]}, "evidence": {"type": "string"},
                    "page": {"type": "integer"}, "confidence": {"type": "number"}
                }, "required": ["field", "value", "evidence", "page", "confidence"]}},
            },
            "required": ["regions", "fields"],
        }
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseJsonSchema": schema},
        }
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=int(os.environ.get("GEMINI_DOCUMENT_TIMEOUT_SECONDS", "180"))) as response:
            body = json.loads(response.read())
        parsed = json.loads(body["candidates"][0]["content"]["parts"][0]["text"])

    regions: list[dict] = []
    region_by_page: dict[int, str] = {}
    for index, item in enumerate(parsed.get("regions", [])):
        page = max(1, int(item["page"]))
        region_id = f"GEM-P{page}-R{index}"
        region_by_page.setdefault(page, region_id)
        regions.append({
            "id": region_id, "kind": "table" if "table" in item["kind"].lower() or "表" in item["kind"] else "paragraph",
            "content": item["content"], "page": page, "bbox": None, "rows": [], "paragraph_index": None,
            "table_index": None, "expert": "gemini_document", "confidence": max(0.0, min(1.0, float(item["confidence"])))
        })
    candidates = []
    for item in parsed.get("fields", []):
        if item.get("field") not in ALLOWED_FIELDS:
            continue
        page = max(1, int(item["page"]))
        candidates.append({
            "field": item["field"], "value": item["value"], "expert": "gemini_document", "channel": "semantic",
            "confidence": max(0.0, min(1.0, float(item["confidence"]))), "evidence_text": item["evidence"][:500],
            "region_id": region_by_page.get(page), "weight": 1.0,
        })
    print(json.dumps({"expert": "gemini_document", "channel": "semantic", "regions": regions, "candidates": candidates, "errors": []}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
