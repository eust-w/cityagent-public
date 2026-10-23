from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from bridge_evidence.config import Settings
from bridge_evidence.evidence.disease_graph import published_diseases
from bridge_evidence.orchestrator import BridgeEvidenceOrchestrator

from .labels import pair_files, parse_label


SUMMARY_FIELDS = {
    "桥梁名称": lambda s: s.bridge.name.value,
    "报告日期": lambda s: s.bridge.report_date.value,
    "总体评分": lambda s: s.scores.overall.score.value,
    "总体等级": lambda s: s.scores.overall.grade.value,
    "上部结构评分": lambda s: s.scores.superstructure.score.value,
    "上部结构等级": lambda s: s.scores.superstructure.grade.value,
    "下部结构评分": lambda s: s.scores.substructure.score.value,
    "下部结构等级": lambda s: s.scores.substructure.grade.value,
    "桥面系评分": lambda s: s.scores.deck.score.value,
    "桥面系等级": lambda s: s.scores.deck.grade.value,
    "上一次总体评分": lambda s: s.scores.previous.score.value,
    "上一次总体等级": lambda s: s.scores.previous.grade.value,
}

SUMMARY_TEXT_FIELDS = {
    "病害发展趋势与具体说明": lambda s: s.narrative.development_trend,
    "总体结论": lambda s: s.narrative.overall_conclusion,
    "主要风险点": lambda s: s.narrative.key_risks,
    "建议": lambda s: _recommendation_count_text(s),
}

DISEASE_DETAIL_FIELDS = ("病害部位", "病害类型", "病害描述", "是否新增", "上一次定检状态", "发展程度")
RECOMMENDATION_DETAIL_FIELDS = ("建议类别", "病害部位", "建议内容")


def evaluation_strata(state) -> list[str]:
    source_suffix = Path(state.source_path).suffix.lower() or "unknown"
    disease_count = len(published_diseases(state))
    disease_volume = "large" if disease_count >= 50 else "medium" if disease_count >= 10 else "small"
    return [
        f"format:{source_suffix}",
        f"score_cardinality:{state.scores.presentation_mode}",
        f"disease_volume:{disease_volume}",
        f"conversion:{'converted' if state.converted_path else 'native'}",
    ]


def _normalize(value: Any) -> str:
    return re.sub(r"[\s，,。；;（）()]", "", str(value or "")).replace("类", "级")


def _matches(field: str, predicted: Any, expected: str) -> bool:
    if _normalize(expected) == "无":
        return predicted in (None, "", "无")
    if "评分" in field:
        predicted_values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", str(predicted or ""))]
        expected_values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", str(expected or ""))]
        return bool(predicted_values) and len(predicted_values) == len(expected_values) and all(
            abs(left - right) <= 0.011 for left, right in zip(predicted_values, expected_values)
        )
    if "等级" in field:
        grade_pattern = r"[ABCDE](?:级|类)?|[一二三四五]类"
        predicted_grades = [value.upper().replace("类", "级") for value in re.findall(grade_pattern, str(predicted or ""), re.I)]
        expected_grades = [value.upper().replace("类", "级") for value in re.findall(grade_pattern, str(expected or ""), re.I)]
        if predicted_grades and expected_grades:
            return predicted_grades == expected_grades
    return _normalize(predicted) == _normalize(expected)


def _recommendation_count_text(state) -> str:
    counts = Counter(item.category for item in state.recommendations)
    return (
        f"{counts['立即处置']}条立即处置、"
        f"{counts['尽快维修']}条尽快维修、"
        f"{counts['预防性养护']}条预防性养护"
    )


def _text_similarity(left: Any, right: Any) -> float:
    left_normalized = _normalize(left)
    right_normalized = _normalize(right)
    if not left_normalized and not right_normalized:
        return 1.0
    if not left_normalized or not right_normalized:
        return 0.0
    return fuzz.WRatio(left_normalized, right_normalized) / 100.0


def _assignment_pairs(
    predicted: list[Any],
    expected: list[Any],
    scorer: Callable[[Any, Any], float],
    *,
    matrix_limit: int = 120_000,
) -> tuple[list[tuple[int, int, float]], int]:
    """Return bounded one-to-one row matches and the penalized denominator.

    The exact Hungarian assignment is used for ordinary reports.  Very large
    photo ledgers use a deterministic greedy fallback to keep the evaluator
    bounded; this affects diagnostics only and never extraction behavior.
    """
    if not predicted and not expected:
        return [], 0
    if not predicted or not expected:
        return [], max(len(predicted), len(expected))
    denominator = max(len(predicted), len(expected))
    if len(predicted) * len(expected) <= matrix_limit:
        scores = np.array([[scorer(left, right) for right in expected] for left in predicted], dtype=float)
        rows, columns = linear_sum_assignment(scores, maximize=True)
        return [(int(row), int(column), float(scores[row, column])) for row, column in zip(rows, columns)], denominator

    # Large ledgers are usually near row-for-row.  Compare within a moving
    # window, then greedily reserve the best unused candidate.
    unused = set(range(len(expected)))
    matches: list[tuple[int, int, float]] = []
    radius = max(12, min(80, abs(len(predicted) - len(expected)) + 12))
    for index, left in enumerate(predicted):
        if not unused:
            break
        local = [candidate for candidate in range(max(0, index - radius), min(len(expected), index + radius + 1)) if candidate in unused]
        candidates = local or list(unused)[:radius]
        best = max(candidates, key=lambda candidate: scorer(left, expected[candidate]))
        matches.append((index, best, scorer(left, expected[best])))
        unused.remove(best)
    return matches, denominator


def _assignment_score(
    predicted: list[Any],
    expected: list[Any],
    scorer: Callable[[Any, Any], float],
    *,
    matrix_limit: int = 120_000,
) -> float:
    matches, denominator = _assignment_pairs(
        predicted,
        expected,
        scorer,
        matrix_limit=matrix_limit,
    )
    if denominator == 0:
        return 1.0
    return sum(score for _, _, score in matches) / denominator


def _assignment_field_scores(
    predicted: list[dict[str, str]],
    expected: list[dict[str, str]],
    scorer: Callable[[dict[str, str], dict[str, str]], float],
    fields: tuple[str, ...],
) -> dict[str, float]:
    matches, denominator = _assignment_pairs(predicted, expected, scorer)
    if denominator == 0:
        return {field: 1.0 for field in fields}
    return {
        field: sum(_text_similarity(predicted[left].get(field), expected[right].get(field)) for left, right, _ in matches) / denominator
        for field in fields
    }


def _recommendation_similarity(predicted: dict[str, str], expected: dict[str, str]) -> float:
    return (
        0.20 * _text_similarity(predicted.get("建议类别"), expected.get("建议类别"))
        + 0.25 * _text_similarity(predicted.get("病害部位"), expected.get("病害部位"))
        + 0.55 * _text_similarity(predicted.get("建议内容"), expected.get("建议内容"))
    )


def _disease_similarity(predicted: dict[str, str], expected: dict[str, str]) -> float:
    return (
        0.20 * _text_similarity(predicted.get("病害部位"), expected.get("病害部位"))
        + 0.15 * _text_similarity(predicted.get("病害类型"), expected.get("病害类型"))
        + 0.35 * _text_similarity(predicted.get("病害描述"), expected.get("病害描述"))
        + 0.10 * _text_similarity(predicted.get("是否新增"), expected.get("是否新增"))
        + 0.10 * _text_similarity(predicted.get("上一次定检状态"), expected.get("上一次定检状态"))
        + 0.10 * _text_similarity(predicted.get("发展程度"), expected.get("发展程度"))
    )


def _summary_logic_score(state) -> float:
    checks: list[bool] = []
    for scope in ("overall", "superstructure", "substructure", "deck"):
        entry = getattr(state.scores, scope)
        checks.extend([entry.score.value not in (None, ""), entry.grade.value not in (None, "")])
        numeric = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(entry.score.value or ""))]
        checks.append(bool(numeric) and all(0 <= item <= 100 for item in numeric))
        checks.append(not any(0 < item <= 1 for item in numeric))
    checks.extend([
        bool(state.narrative.overall_conclusion),
        bool(state.narrative.key_risks),
        "同上" not in state.narrative.development_trend,
        state.validation.passed,
    ])
    return sum(checks) / len(checks) if checks else 0.0


def _detail_logic_score(state) -> float:
    checks = _detail_logic_checks(state)
    total = sum(total for _, total in checks.values())
    passed = sum(passed for passed, _ in checks.values())
    return passed / total if total else 0.0


def _detail_logic_checks(state) -> dict[str, tuple[int, int]]:
    checks: dict[str, list[bool]] = {
        "recommendation_content": [],
        "recommendation_component": [],
        "recommendation_action": [],
        "recommendation_category": [],
        "disease_location": [],
        "disease_type": [],
        "disease_description": [],
        "previous_not_ditto": [],
        "development_not_ditto": [],
        "new_history_consistent": [],
    }
    action_pattern = re.compile(r"建议|应|需|及时|立即|加强|修复|维修|处治|处理|清理|更换|观测|检查|养护")
    for item in state.recommendations:
        checks["recommendation_content"].append(bool(item.content))
        checks["recommendation_component"].append(bool(item.component))
        checks["recommendation_action"].append(bool(action_pattern.search(item.content)))
        checks["recommendation_category"].append(item.category in {"立即处置", "尽快维修", "预防性养护"})
    for item in published_diseases(state):
        checks["disease_location"].append(bool(item.location or item.component))
        checks["disease_type"].append(bool(item.type))
        checks["disease_description"].append(bool(item.description))
        checks["previous_not_ditto"].append(item.previous_state != "同上")
        checks["development_not_ditto"].append(item.development != "同上")
        checks["new_history_consistent"].append(
            not (
                item.is_new is True
                and item.previous_state
                and "已发现" in item.previous_state
                and "未提及" not in item.previous_state
            )
        )
    return {name: (sum(values), len(values)) for name, values in checks.items()}


def _predicted_recommendations(state) -> list[dict[str, str]]:
    result = []
    for item in state.recommendations:
        cat = item.category if item.category in {"立即处置", "尽快维修", "预防性养护"} else "预防性养护"
        raw_comp = (item.component or "桥梁总体").strip()
        clean_comp = re.sub(r"^\d+[\.、\s]+", "", raw_comp).strip() or raw_comp
        result.append({
            "建议类别": cat,
            "建议内容": (item.content or "").strip(),
            "病害部位": clean_comp,
        })
    return result


def _predicted_diseases(state) -> list[dict[str, str]]:
    result = []
    for item in published_diseases(state):
        raw_loc = (item.location or item.component or "").strip()
        clean_loc = re.sub(r"^\d+[\.、\s]+", "", raw_loc).strip() or raw_loc
        is_new_str = "是" if item.is_new is True else "否"
        prev = str(item.previous_state or "").strip()
        history_str = prev if prev and prev != "未明确" else "未明确"
        dev = str(item.development or "").strip()
        dev_str = dev if dev and dev != "未明确" else "未明确"
        result.append({
            "病害部位": clean_loc,
            "病害类型": (item.type or "").strip(),
            "病害描述": (item.description or "").strip(),
            "是否新增": is_new_str,
            "上一次定检状态": history_str,
            "发展程度": dev_str,
        })
    return result


def detail_field_scores(state, label) -> dict[str, float]:
    recommendation = _assignment_field_scores(
        _predicted_recommendations(state),
        label.recommendations,
        _recommendation_similarity,
        RECOMMENDATION_DETAIL_FIELDS,
    )
    disease = _assignment_field_scores(
        _predicted_diseases(state),
        label.diseases,
        _disease_similarity,
        DISEASE_DETAIL_FIELDS,
    )
    return {
        **{f"建议.{field}": score for field, score in recommendation.items()},
        **{f"病害.{field}": score for field, score in disease.items()},
    }


def semantic_proxy_scores(state, label) -> dict[str, float]:
    factual = [
        1.0 if _matches(field, getter(state), label.summary.get(field, "")) else 0.0
        for field, getter in SUMMARY_FIELDS.items()
        if label.summary.get(field, "")
    ]
    text_scores = list(factual)
    for field, getter in SUMMARY_TEXT_FIELDS.items():
        expected = label.summary.get(field, "")
        if not expected:
            continue
        predicted = getter(state)
        text_scores.append(1.0 if field == "建议" and _matches(field, predicted, expected) else _text_similarity(predicted, expected))

    predicted_recommendations = _predicted_recommendations(state)
    predicted_diseases = _predicted_diseases(state)
    recommendation_score = _assignment_score(predicted_recommendations, label.recommendations, _recommendation_similarity)
    disease_score = _assignment_score(predicted_diseases, label.diseases, _disease_similarity)
    detail_consistency = 0.25 * recommendation_score + 0.75 * disease_score
    return {
        "summary_fact_accuracy": sum(factual) / len(factual) if factual else 0.0,
        "summary_text_consistency": sum(text_scores) / len(text_scores) if text_scores else 0.0,
        "detail_text_consistency": detail_consistency,
        "summary_expression_logic": _summary_logic_score(state),
        "detail_expression_logic": _detail_logic_score(state),
        "recommendation_row_similarity": recommendation_score,
        "disease_row_similarity": disease_score,
    }


def evaluate_year(
    input_dir: Path,
    label_dir: Path,
    settings: Settings,
    limit: int | None = None,
    workers: int = 1,
    runner: BridgeEvidenceOrchestrator | None = None,
) -> dict[str, Any]:
    pairs = pair_files(input_dir, label_dir)
    if limit:
        pairs = pairs[:limit]
    runner = runner or BridgeEvidenceOrchestrator(settings)
    field_totals = Counter()
    field_correct = Counter()
    failures: list[dict[str, str]] = []
    mismatches: dict[str, list[dict[str, str]]] = {field: [] for field in SUMMARY_FIELDS}
    count_outliers: list[dict[str, Any]] = []
    disease_counts = {"predicted": 0, "expected": 0}
    recommendation_counts = {"predicted": 0, "expected": 0}
    proxy_totals = Counter()
    proxy_count = 0
    detail_field_totals = Counter()
    failure_sample_types = Counter()
    detail_logic_passed = Counter()
    detail_logic_total = Counter()
    stratum_totals: dict[str, Counter] = defaultdict(Counter)
    def process_pair(pair: tuple[Path, Path]):
        input_path, label_path = pair
        try:
            state = runner.process(input_path, output_dir=None, render=False)
            label = parse_label(label_path, runner.adapter)
        except Exception as exc:
            return input_path, None, None, exc
        return input_path, state, label, None

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bridge-eval") as pool:
            results = list(pool.map(process_pair, pairs))
    else:
        results = [process_pair(pair) for pair in pairs]

    for input_path, state, label, error in results:
        if error is not None:
            failures.append({"input": str(input_path), "error": str(error)})
            continue
        assert state is not None and label is not None
        published_disease_count = len(published_diseases(state))
        document_summary_total = 0
        document_summary_correct = 0
        for field, getter in SUMMARY_FIELDS.items():
            expected = label.summary.get(field, "")
            if not expected:
                continue
            field_totals[field] += 1
            predicted = getter(state)
            matched = _matches(field, predicted, expected)
            document_summary_total += 1
            document_summary_correct += int(matched)
            field_correct[field] += matched
            if not matched and len(mismatches[field]) < 25:
                mismatches[field].append({"file": input_path.name, "predicted": str(predicted), "expected": expected})
        disease_counts["predicted"] += published_disease_count
        disease_counts["expected"] += len(label.diseases)
        recommendation_counts["predicted"] += len(state.recommendations)
        recommendation_counts["expected"] += len(label.recommendations)
        proxy = semantic_proxy_scores(state, label)
        for stratum in evaluation_strata(state):
            stratum_totals[stratum]["documents"] += 1
            stratum_totals[stratum]["summary_total"] += document_summary_total
            stratum_totals[stratum]["summary_correct"] += document_summary_correct
            stratum_totals[stratum]["disease_count_error"] += abs(published_disease_count - len(label.diseases))
        proxy_totals.update(proxy)
        proxy_count += 1
        fields = detail_field_scores(state, label)
        detail_field_totals.update(fields)
        if published_disease_count < len(label.diseases):
            failure_sample_types["病害行漏提"] += 1
        elif published_disease_count > len(label.diseases):
            failure_sample_types["病害行多提"] += 1
        if len(state.recommendations) < len(label.recommendations):
            failure_sample_types["建议行漏提"] += 1
        elif len(state.recommendations) > len(label.recommendations):
            failure_sample_types["建议行多提"] += 1
        for field, score in fields.items():
            if score < 0.5:
                failure_sample_types[f"{field}低相似"] += 1
        for check, (passed, total_checks) in _detail_logic_checks(state).items():
            detail_logic_passed[check] += passed
            detail_logic_total[check] += total_checks
        if published_disease_count > max(10, len(label.diseases) * 1.5) or published_disease_count < len(label.diseases) * 0.7:
            count_outliers.append({"file": input_path.name, "predicted_diseases": published_disease_count, "expected_diseases": len(label.diseases)})
    total = sum(field_totals.values())
    correct = sum(field_correct.values())
    return {
        "pairs": len(pairs),
        "processed": len(pairs) - len(failures),
        "summary_accuracy": correct / total if total else 0.0,
        "field_accuracy": {field: field_correct[field] / count for field, count in field_totals.items()},
        "field_counts": dict(field_totals),
        "disease_counts": disease_counts,
        "recommendation_counts": recommendation_counts,
        "five_metric_proxy": {field: proxy_totals[field] / proxy_count for field in proxy_totals} if proxy_count else {},
        "detail_field_similarity": {
            field: detail_field_totals[field] / proxy_count for field in detail_field_totals
        } if proxy_count else {},
        "failure_sample_types": dict(failure_sample_types.most_common()),
        "detail_logic_diagnostics": {
            check: detail_logic_passed[check] / total_checks if total_checks else 1.0
            for check, total_checks in detail_logic_total.items()
        },
        "stratified_metrics": {
            stratum: {
                "documents": int(values["documents"]),
                "summary_accuracy": values["summary_correct"] / values["summary_total"] if values["summary_total"] else 0.0,
                "mean_disease_count_error": values["disease_count_error"] / values["documents"] if values["documents"] else 0.0,
            }
            for stratum, values in sorted(stratum_totals.items())
        },
        "mismatches": mismatches,
        "disease_count_outliers": count_outliers,
        "failures": failures,
    }


def write_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
