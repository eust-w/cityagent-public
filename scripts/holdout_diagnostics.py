from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from bridge_evidence.config import load_settings
from bridge_evidence.evaluation.labels import pair_files, parse_label
from bridge_evidence.evaluation.metrics import (
    _disease_similarity,
    _recommendation_similarity,
    _text_similarity,
    semantic_proxy_scores,
)
from bridge_evidence.orchestrator import BridgeEvidenceOrchestrator


DISEASE_FIELDS = (
    "病害部位",
    "病害类型",
    "病害描述",
    "是否新增",
    "上一次定检状态",
    "发展程度",
)
RECOMMENDATION_FIELDS = ("建议类别", "病害部位", "建议内容")


def _predicted_diseases(state) -> list[dict[str, str]]:
    return [
        {
            "病害部位": item.location or item.component,
            "病害类型": item.type,
            "病害描述": item.description,
            "是否新增": "是" if item.is_new is True else "否" if item.is_new is False else "未明确",
            "__novelty_source": item.novelty_source or "unresolved",
            "上一次定检状态": item.previous_state or "未明确",
            "发展程度": item.development or "未明确",
        }
        for item in state.diseases
    ]


def _predicted_recommendations(state) -> list[dict[str, str]]:
    return [
        {"建议类别": item.category, "病害部位": item.component, "建议内容": item.content}
        for item in state.recommendations
    ]


def _assignment_diagnostics(
    predicted: list[dict[str, str]],
    expected: list[dict[str, str]],
    scorer: Callable[[dict[str, str], dict[str, str]], float],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    denominator = max(len(predicted), len(expected))
    if denominator == 0:
        return {
            "score": 1.0,
            "columns": {field: 1.0 for field in fields},
            "predicted": 0,
            "expected": 0,
            "unmatched_predicted": 0,
            "unmatched_expected": 0,
            "worst_pairs": [],
        }
    if not predicted or not expected:
        return {
            "score": 0.0,
            "columns": {field: 0.0 for field in fields},
            "predicted": len(predicted),
            "expected": len(expected),
            "unmatched_predicted": len(predicted),
            "unmatched_expected": len(expected),
            "worst_pairs": [],
        }

    scores = np.array([[scorer(left, right) for right in expected] for left in predicted], dtype=float)
    rows, columns = linear_sum_assignment(scores, maximize=True)
    column_scores = {
        field: float(sum(_text_similarity(predicted[row].get(field), expected[column].get(field)) for row, column in zip(rows, columns)) / denominator)
        for field in fields
    }
    pair_details = []
    confusions: dict[str, Counter[str]] = {
        "是否新增": Counter(),
        "发展程度": Counter(),
        "上一次定检状态类别": Counter(),
    }
    novelty_source_confusions: dict[str, Counter[str]] = {}

    def previous_class(value: str) -> str:
        if "未发现" in value or "未提及" in value:
            return "未发现"
        if "已发现" in value or "已有" in value:
            return "已发现"
        if not value or value == "未明确":
            return "未明确"
        return "其他"

    for row, column in zip(rows, columns):
        pair_score = float(scores[row, column])
        confusions["是否新增"][f"{predicted[row].get('是否新增', '')} -> {expected[column].get('是否新增', '')}"] += 1
        source = predicted[row].get("__novelty_source", "unresolved") or "unresolved"
        novelty_source_confusions.setdefault(source, Counter())[
            f"{predicted[row].get('是否新增', '')} -> {expected[column].get('是否新增', '')}"
        ] += 1
        confusions["发展程度"][f"{predicted[row].get('发展程度', '')} -> {expected[column].get('发展程度', '')}"] += 1
        confusions["上一次定检状态类别"][
            f"{previous_class(predicted[row].get('上一次定检状态', ''))} -> "
            f"{previous_class(expected[column].get('上一次定检状态', ''))}"
        ] += 1
        pair_details.append(
            {
                "score": pair_score,
                "predicted": predicted[row],
                "expected": expected[column],
                "columns": {
                    field: _text_similarity(predicted[row].get(field), expected[column].get(field))
                    for field in fields
                },
            }
        )
    pair_details.sort(key=lambda item: item["score"])
    return {
        "score": float(scores[rows, columns].sum() / denominator),
        "columns": column_scores,
        "predicted": len(predicted),
        "expected": len(expected),
        "unmatched_predicted": len(predicted) - len(rows),
        "unmatched_expected": len(expected) - len(columns),
        "confusions": {name: dict(values) for name, values in confusions.items()},
        "novelty_source_confusions": {
            name: dict(values)
            for name, values in novelty_source_confusions.items()
        },
        "worst_pairs": pair_details[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose IAIC temporal holdout errors by row and column")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--history-label-dir", type=Path)
    parser.add_argument(
        "--history-before-year",
        type=int,
        help="仅加载该年份之前的历史标签，保持严格时间外推边界。",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--skip-legacy-doc",
        action="store_true",
        help="Skip binary .doc inputs when isolating model/matcher quality from conversion reliability.",
    )
    parser.add_argument(
        "--disease-match-threshold",
        type=float,
        default=0.56,
        help="Development-only sweep for the chronology-safe one-to-one matcher.",
    )
    args = parser.parse_args()

    settings = load_settings(
        None,
        llm_mode="off",
        history_label_dir=args.history_label_dir,
        history_before_year=args.history_before_year,
        disease_match_threshold=args.disease_match_threshold,
        work_dir=args.work_dir,
        cache_dir=args.cache_dir,
    )
    runner = BridgeEvidenceOrchestrator(settings)
    pairs = pair_files(args.input_dir, args.label_dir)
    if args.skip_legacy_doc:
        pairs = [pair for pair in pairs if pair[0].suffix.lower() != ".doc"]

    def process(pair: tuple[Path, Path]):
        input_path, label_path = pair
        state = runner.process(input_path, output_dir=None, render=False)
        label = parse_label(label_path, runner.adapter)
        disease = _assignment_diagnostics(
            _predicted_diseases(state), label.diseases, _disease_similarity, DISEASE_FIELDS
        )
        recommendation = _assignment_diagnostics(
            _predicted_recommendations(state),
            label.recommendations,
            _recommendation_similarity,
            RECOMMENDATION_FIELDS,
        )
        return {
            "file": input_path.name,
            "proxy": semantic_proxy_scores(state, label),
            "disease": disease,
            "recommendation": recommendation,
        }

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="holdout-diagnostics") as pool:
            documents = list(pool.map(process, pairs))
    else:
        documents = [process(pair) for pair in pairs]

    disease_columns = Counter()
    recommendation_columns = Counter()
    disease_confusions: dict[str, Counter[str]] = {
        "是否新增": Counter(),
        "发展程度": Counter(),
        "上一次定检状态类别": Counter(),
    }
    novelty_source_confusions: dict[str, Counter[str]] = {}
    proxy_totals = Counter()
    for document in documents:
        proxy_totals.update(document["proxy"])
        disease_columns.update(document["disease"]["columns"])
        recommendation_columns.update(document["recommendation"]["columns"])
        for name, values in document["disease"].get("confusions", {}).items():
            disease_confusions[name].update(values)
        for name, values in document["disease"].get("novelty_source_confusions", {}).items():
            novelty_source_confusions.setdefault(name, Counter()).update(values)

    count = len(documents) or 1
    summary = {
        "documents": len(documents),
        "proxy": {key: value / count for key, value in proxy_totals.items()},
        "disease_columns": {field: disease_columns[field] / count for field in DISEASE_FIELDS},
        "recommendation_columns": {
            field: recommendation_columns[field] / count for field in RECOMMENDATION_FIELDS
        },
        "disease_confusions": {name: dict(values.most_common()) for name, values in disease_confusions.items()},
        "novelty_source_confusions": {
            name: dict(values.most_common())
            for name, values in sorted(novelty_source_confusions.items())
        },
        "disease_counts": {
            "predicted": sum(document["disease"]["predicted"] for document in documents),
            "expected": sum(document["disease"]["expected"] for document in documents),
        },
        "recommendation_counts": {
            "predicted": sum(document["recommendation"]["predicted"] for document in documents),
            "expected": sum(document["recommendation"]["expected"] for document in documents),
        },
    }
    documents.sort(key=lambda item: item["proxy"]["detail_text_consistency"])
    payload = {"summary": summary, "documents": documents}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
