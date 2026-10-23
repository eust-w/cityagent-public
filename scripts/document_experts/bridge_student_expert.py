#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bridge_evidence.document_ai.student import CalibratedDocumentStudent, candidate_features
from bridge_evidence.document_ai.vision_candidates import extract_field_candidates, normalize_candidate

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def document_units(path: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    body = root.find(f"{WORD_NS}body")
    if body is None:
        return []
    units: list[tuple[str, str]] = []
    for child in body:
        if child.tag == f"{WORD_NS}p":
            text = "".join(node.text or "" for node in child.iter(f"{WORD_NS}t")).strip()
            if text:
                units.append(("paragraph", text))
        elif child.tag == f"{WORD_NS}tbl":
            rows: list[str] = []
            for row in child.iter(f"{WORD_NS}tr"):
                cells = []
                for cell in row.findall(f"{WORD_NS}tc"):
                    value = "".join(node.text or "" for node in cell.iter(f"{WORD_NS}t")).strip()
                    cells.append(value)
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                units.append(("table", "\n".join(rows)))
    return units


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: bridge_student_expert.py DOCUMENT.docx")
    model_path = Path(os.environ.get("BRIDGE_EVIDENCE_STUDENT_MODEL", ROOT / "models/document_ai/bridge_student.json"))
    model = CalibratedDocumentStudent.load(model_path)
    regions: list[dict] = []
    candidates: list[dict] = []
    units = document_units(Path(sys.argv[1]))
    for index, (kind, content) in enumerate(units):
        region_id = f"STU-{kind[0].upper()}{index}"
        extracted = extract_field_candidates(
            content,
            expert="bridge_student",
            channel="semantic",
            region_id=region_id,
        )
        if not extracted:
            continue
        regions.append({
            "id": region_id, "kind": kind, "content": content, "page": None, "bbox": None, "rows": [],
            "paragraph_index": index if kind == "paragraph" else None,
            "table_index": index if kind == "table" else None,
            "expert": "bridge_student", "confidence": 0.95,
        })
        for candidate in extracted:
            features = candidate_features(
                field=candidate["field"],
                base_confidence=float(candidate["confidence"]),
                evidence_text=candidate["evidence_text"],
                region_kind=kind,
                position=index / max(1, len(units) - 1),
            )
            candidate["confidence"] = model.confidence(candidate["field"], float(candidate["confidence"]), features)
            candidates.append(candidate)
    winners: dict[str, dict] = {}
    for candidate in candidates:
        field = candidate["field"]
        existing = winners.get(field)
        if existing is None or candidate["confidence"] > existing["confidence"]:
            winners[field] = candidate
        elif candidate["confidence"] == existing["confidence"]:
            current_dual = "左幅" in str(candidate["value"]) and "右幅" in str(candidate["value"])
            existing_dual = "左幅" in str(existing["value"]) and "右幅" in str(existing["value"])
            if current_dual and not existing_dual:
                winners[field] = candidate
    candidates = list(winners.values())
    referenced = {candidate["region_id"] for candidate in candidates}
    regions = [region for region in regions if region["id"] in referenced]
    print(json.dumps({"expert": "bridge_student", "channel": "semantic", "regions": regions, "candidates": candidates, "errors": []}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
