#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
candidate_module_path = ROOT / "src" / "bridge_evidence" / "document_ai" / "vision_candidates.py"
candidate_spec = importlib.util.spec_from_file_location("bridge_vision_candidates", candidate_module_path)
if candidate_spec is None or candidate_spec.loader is None:
    raise RuntimeError(f"cannot load candidate module: {candidate_module_path}")
candidate_module = importlib.util.module_from_spec(candidate_spec)
candidate_spec.loader.exec_module(candidate_module)
best_page_numbers = candidate_module.best_page_numbers
best_docx_page_numbers = candidate_module.best_docx_page_numbers
extract_field_candidates = candidate_module.extract_field_candidates


def convert_to_pdf(document: Path, output_dir: Path) -> Path:
    subprocess.run(
        ["/opt/homebrew/bin/soffice", "--headless", "--convert-to", "pdf", "--outdir", str(output_dir), str(document)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    pdfs = list(output_dir.glob("*.pdf"))
    if len(pdfs) != 1:
        raise RuntimeError(f"expected one converted PDF, found {len(pdfs)}")
    return pdfs[0]


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: paddleocr_vl_expert.py DOCUMENT.docx")
    document = Path(sys.argv[1])
    limit = max(1, int(os.environ.get("PADDLEOCR_VL_MAX_PAGES", "2")))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    from paddleocr import PaddleOCRVL

    regions: list[dict] = []
    candidates: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="bridge-paddleocr-vl-") as temp_dir:
        output_dir = Path(temp_dir)
        pdf = convert_to_pdf(document, output_dir)
        pages = best_docx_page_numbers(document, limit=limit)
        images: list[Path] = []
        for page in pages:
            prefix = output_dir / f"page-{page:04d}"
            subprocess.run(
                ["pdftoppm", "-f", str(page), "-l", str(page), "-singlefile", "-png", "-r", "144", str(pdf), str(prefix)],
                check=True,
                capture_output=True,
                timeout=180,
            )
            images.append(prefix.with_suffix(".png"))

        pipeline = PaddleOCRVL(device="cpu")
        for page, result in zip(pages, pipeline.predict([str(path) for path in images])):
            payload = result.json.get("res", result.json)
            for block_index, block in enumerate(payload.get("parsing_res_list", [])):
                content = str(block.get("block_content", "")).strip()
                if not content:
                    continue
                label = str(block.get("block_label", "unknown")).lower()
                kind = "table" if "table" in label else "paragraph" if "text" in label or "title" in label else "unknown"
                region_id = f"PVL-P{page}-B{block_index}"
                bbox = block.get("block_bbox")
                region = {
                    "id": region_id,
                    "kind": kind,
                    "content": content,
                    "page": page,
                    "bbox": [float(value) for value in bbox] if bbox is not None else None,
                    "rows": [],
                    "paragraph_index": None,
                    "table_index": None,
                    "expert": "paddleocr_vl",
                    "confidence": 0.90,
                }
                regions.append(region)
                candidates.extend(
                    extract_field_candidates(
                        content,
                        expert="paddleocr_vl",
                        channel="visual",
                        region_id=region_id,
                        confidence_scale=1.0,
                    )
                )

    print(json.dumps({"expert": "paddleocr_vl", "channel": "visual", "regions": regions, "candidates": candidates, "errors": []}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
