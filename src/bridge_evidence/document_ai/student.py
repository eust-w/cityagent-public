from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


FEATURE_NAMES = (
    "bias",
    "base_confidence",
    "is_table",
    "early_position",
    "exact_index_code",
    "assessment_context",
    "history_context",
    "name_context",
    "date_context",
    "dual_carriageway",
    "evidence_length",
)


def candidate_features(
    *,
    field: str,
    base_confidence: float,
    evidence_text: str,
    region_kind: str,
    position: float,
) -> list[float]:
    text = evidence_text or ""
    codes = {
        "scores.overall": "BCI",
        "scores.deck": "BSIm",
        "scores.superstructure": "BSIs",
        "scores.substructure": "BSIx",
    }
    prefix = field.rsplit(".", 1)[0]
    exact_code = codes.get(prefix, "")
    return [
        1.0,
        max(0.0, min(1.0, base_confidence)),
        1.0 if "table" in region_kind else 0.0,
        1.0 - max(0.0, min(1.0, position)),
        1.0 if exact_code and exact_code.lower() in text.lower() else 0.0,
        1.0 if any(token in text for token in ("技术状况评定", "评定结果", "总体评分", "评分汇总")) else 0.0,
        1.0 if any(token in text for token in ("上一次", "上次", "往年", "历史", "对比")) else 0.0,
        1.0 if any(token in text for token in ("桥梁名称", "工程名称")) else 0.0,
        1.0 if any(token in text for token in ("报告日期", "检测日期", "检测时间")) else 0.0,
        1.0 if "左幅" in text and "右幅" in text else 0.0,
        min(1.0, len(text) / 500.0),
    ]


@dataclass(frozen=True)
class FieldCalibration:
    correct: int
    total: int
    reliability: float
    weights: tuple[float, ...] = ()


class CalibratedDocumentStudent:
    """Small auditable student that calibrates field candidates from labels."""

    schema_version = "1.0"

    def __init__(self, fields: dict[str, FieldCalibration], *, global_reliability: float = 0.75):
        self.fields = fields
        self.global_reliability = global_reliability

    @classmethod
    def fit(cls, observations: list[tuple[str, bool]]) -> "CalibratedDocumentStudent":
        counts: dict[str, list[int]] = {}
        correct_total = 0
        for field, correct in observations:
            bucket = counts.setdefault(field, [0, 0])
            bucket[1] += 1
            if correct:
                bucket[0] += 1
                correct_total += 1
        fields = {
            field: FieldCalibration(correct, total, (correct + 2.0) / (total + 3.0))
            for field, (correct, total) in counts.items()
        }
        global_reliability = (correct_total + 2.0) / (len(observations) + 3.0)
        return cls(fields, global_reliability=global_reliability)

    @classmethod
    def fit_featured(cls, observations: list[tuple[str, bool, list[float]]]) -> "CalibratedDocumentStudent":
        grouped: dict[str, list[tuple[bool, list[float]]]] = {}
        for field, correct, features in observations:
            grouped.setdefault(field, []).append((correct, features))
        fields: dict[str, FieldCalibration] = {}
        correct_total = sum(int(correct) for _, correct, _ in observations)
        for field, values in grouped.items():
            correct = sum(int(label) for label, _ in values)
            total = len(values)
            reliability = (correct + 2.0) / (total + 3.0)
            weights: tuple[float, ...] = ()
            if total >= 20 and 0 < correct < total:
                matrix = np.asarray([features for _, features in values], dtype=float)
                labels = np.asarray([float(label) for label, _ in values], dtype=float)

                def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
                    logits = np.clip(matrix @ beta, -30.0, 30.0)
                    probabilities = 1.0 / (1.0 + np.exp(-logits))
                    epsilon = 1e-9
                    loss = -np.sum(labels * np.log(probabilities + epsilon) + (1.0 - labels) * np.log(1.0 - probabilities + epsilon))
                    regularization = 0.05 * np.sum(beta[1:] ** 2)
                    gradient = matrix.T @ (probabilities - labels)
                    gradient[1:] += 0.1 * beta[1:]
                    return float(loss + regularization), gradient

                initial = np.zeros(matrix.shape[1], dtype=float)
                initial[0] = math.log(reliability / max(1e-9, 1.0 - reliability))
                fitted = minimize(objective, initial, jac=True, method="L-BFGS-B")
                if fitted.success:
                    weights = tuple(float(value) for value in fitted.x)
            fields[field] = FieldCalibration(correct, total, reliability, weights)
        global_reliability = (correct_total + 2.0) / (len(observations) + 3.0)
        return cls(fields, global_reliability=global_reliability)

    def confidence(self, field: str, base_confidence: float, features: list[float] | None = None) -> float:
        calibration = self.fields.get(field)
        reliability = calibration.reliability if calibration else self.global_reliability
        if calibration and calibration.weights and features and len(calibration.weights) == len(features):
            logit = sum(weight * feature for weight, feature in zip(calibration.weights, features))
            probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
            return max(0.05, min(0.99, 0.2 * base_confidence + 0.8 * probability))
        return max(0.05, min(0.99, 0.35 * base_confidence + 0.65 * reliability))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_type": "beta_calibrated_candidate_student",
            "global_reliability": self.global_reliability,
            "fields": {
                field: {
                    "correct": value.correct,
                    "total": value.total,
                    "reliability": value.reliability,
                    "weights": list(value.weights),
                }
                for field, value in sorted(self.fields.items())
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CalibratedDocumentStudent":
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = {
            field: FieldCalibration(
                correct=int(value["correct"]),
                total=int(value["total"]),
                reliability=float(value["reliability"]),
                weights=tuple(float(item) for item in value.get("weights", [])),
            )
            for field, value in payload.get("fields", {}).items()
        }
        return cls(fields, global_reliability=float(payload.get("global_reliability", 0.75)))
