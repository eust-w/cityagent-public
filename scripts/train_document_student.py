#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rapidfuzz.fuzz import WRatio

from bridge_evidence.adapters import FormatAdapter
from bridge_evidence.document_ai.student import CalibratedDocumentStudent, candidate_features
from bridge_evidence.document_ai.vision_candidates import normalize_candidate
from bridge_evidence.evaluation.labels import base_stem, parse_label

FIELD_LABELS = {
    "bridge.name": "桥梁名称",
    "bridge.report_date": "报告日期",
    "scores.overall.score": "总体评分",
    "scores.overall.grade": "总体等级",
    "scores.superstructure.score": "上部结构评分",
    "scores.superstructure.grade": "上部结构等级",
    "scores.substructure.score": "下部结构评分",
    "scores.substructure.grade": "下部结构等级",
    "scores.deck.score": "桥面系评分",
    "scores.deck.grade": "桥面系等级",
    "scores.previous.score": "上一次总体评分",
    "scores.previous.grade": "上一次总体等级",
}


def candidate_correct(field: str, value: Any, truth: str) -> bool:
    if truth.strip() == "无":
        return value is None or str(value).strip() in {"", "无", "none", "null"}
    if field.endswith(".score"):
        try:
            return abs(float(value) - float(truth)) <= 0.02
        except (TypeError, ValueError):
            return False
    return normalize_candidate(field, value) == normalize_candidate(field, truth)


def label_index(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in {".doc", ".docx"}:
            result[base_stem(path)] = path
    return result


def find_label(source_path: str, labels: dict[str, Path]) -> Path | None:
    key = base_stem(Path(source_path))
    exact = labels.get(key)
    if exact:
        return exact
    matches = [(abs(len(candidate) - len(key)), path) for candidate, path in labels.items() if candidate in key or key in candidate]
    if matches:
        return min(matches)[1]
    scored = [(float(WRatio(key, candidate)), path) for candidate, path in labels.items()]
    score, path = max(scored, default=(0.0, None), key=lambda item: item[0])
    return path if score >= 55.0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an auditable bridge-document candidate student")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    labels = label_index(args.label_root)
    label_adapter = FormatAdapter(args.report.parent / "label-conversion-cache")
    train_observations: list[tuple[str, bool, list[float]]] = []
    holdout: list[tuple[str, bool, list[float]]] = []
    all_observations: list[tuple[str, bool, list[float]]] = []
    documents = paired_documents = 0
    label_errors: list[dict[str, str]] = []
    with args.corpus.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            documents += 1
            label_path = find_label(record["source_path"], labels)
            if label_path is None:
                continue
            try:
                label = parse_label(label_path, label_adapter)
            except (OSError, ValueError) as exc:
                label_errors.append({"path": str(label_path), "error": str(exc)})
                continue
            paired_documents += 1
            is_holdout = int(hashlib.sha256(record["document_id"].encode()).hexdigest()[:8], 16) % 5 == 0
            regions = record.get("regions", [])
            region_map = {
                region["id"]: (region.get("kind", "unknown"), index / max(1, len(regions) - 1))
                for index, region in enumerate(regions)
            }
            for candidate in record.get("candidates", []):
                field = candidate.get("field")
                label_key = FIELD_LABELS.get(field)
                if not label_key or label_key not in label.summary:
                    continue
                kind, position = region_map.get(candidate.get("region_id"), ("unknown", 1.0))
                features = candidate_features(
                    field=field,
                    base_confidence=float(candidate.get("confidence", 0.0)),
                    evidence_text=str(candidate.get("evidence_text", "")),
                    region_kind=kind,
                    position=position,
                )
                observation = (field, candidate_correct(field, candidate.get("value"), label.summary[label_key]), features)
                all_observations.append(observation)
                (holdout if is_holdout else train_observations).append(observation)

    validation_model = CalibratedDocumentStudent.fit_featured(train_observations)
    holdout_correct = sum(correct for _, correct, _ in holdout)
    model = CalibratedDocumentStudent.fit_featured(all_observations)
    model.save(args.output)
    report = {
        "schema_version": "1.0",
        "documents": documents,
        "paired_documents": paired_documents,
        "label_errors": label_errors,
        "train_candidates": len(train_observations),
        "holdout_candidates": len(holdout),
        "holdout_raw_accuracy": holdout_correct / len(holdout) if holdout else None,
        "validation_global_reliability": validation_model.global_reliability,
        "trained_candidates": len(all_observations),
        "model": model.to_dict(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "model"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
