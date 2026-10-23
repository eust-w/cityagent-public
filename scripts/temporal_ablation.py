from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from bridge_evidence.config import load_settings
from bridge_evidence.evaluation.metrics import evaluate_year
from bridge_evidence.orchestrator import BridgeEvidenceOrchestrator


FIVE_PROXY_FIELDS = (
    "summary_fact_accuracy",
    "summary_text_consistency",
    "detail_text_consistency",
    "summary_expression_logic",
    "detail_expression_logic",
)
PIPELINE_REVISION = "v12.6-row-scoped-novelty-provenance-20260814"


def _write_checkpoint(path: Path, years: list[int], folds: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "1.0",
        "pipeline_revision": PIPELINE_REVISION,
        "years": years,
        "folds": folds,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_checkpoint(path: Path, years: list[int]) -> dict[str, list[dict[str, Any]]]:
    empty: dict[str, list[dict[str, Any]]] = {"no_history": [], "strict_history": []}
    if not path.exists():
        return empty
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("pipeline_revision") != PIPELINE_REVISION:
        raise RuntimeError(
            "时间外推 checkpoint 代码版本不匹配: "
            f"expected={PIPELINE_REVISION}, actual={payload.get('pipeline_revision')}"
        )
    if payload.get("years") != years:
        raise RuntimeError(
            f"时间外推 checkpoint 年份不匹配: expected={years}, actual={payload.get('years')}"
        )
    raw = payload.get("folds") or {}
    return {
        variant: list(raw.get(variant) or [])
        for variant in empty
    }


def _aggregate(folds: list[dict[str, Any]]) -> dict[str, Any]:
    proxy = {
        field: {
            "mean": mean(fold["metrics"]["five_metric_proxy"].get(field, 0.0) for fold in folds),
            "worst": min(fold["metrics"]["five_metric_proxy"].get(field, 0.0) for fold in folds),
        }
        for field in FIVE_PROXY_FIELDS
    }
    detail_fields = sorted({
        field
        for fold in folds
        for field in fold["metrics"].get("detail_field_similarity", {})
    })
    details = {
        field: {
            "mean": mean(fold["metrics"]["detail_field_similarity"].get(field, 0.0) for fold in folds),
            "worst": min(fold["metrics"]["detail_field_similarity"].get(field, 0.0) for fold in folds),
        }
        for field in detail_fields
    }
    failure_types: Counter[str] = Counter()
    for fold in folds:
        failure_types.update(fold["metrics"].get("failure_sample_types", {}))
    return {
        "folds": len(folds),
        "documents": sum(fold["metrics"]["processed"] for fold in folds),
        "five_metric_proxy": proxy,
        "detail_field_similarity": details,
        "failure_sample_types": dict(failure_types.most_common()),
        "model_calls": {"chat": 0, "embedding": 0, "reranker": 0, "planner": 0},
    }


def _deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "five_metric_proxy_mean": {
            field: right["five_metric_proxy"][field]["mean"] - left["five_metric_proxy"][field]["mean"]
            for field in FIVE_PROXY_FIELDS
        },
        "detail_field_similarity_mean": {
            field: right["detail_field_similarity"][field]["mean"] - left["detail_field_similarity"][field]["mean"]
            for field in left["detail_field_similarity"].keys() & right["detail_field_similarity"].keys()
        },
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    variants = payload["aggregates"]
    lines = [
        "# IAIC 多年度严格时间外推与消融报告",
        "",
        "> 本报告中的五项数值均为本地代理指标，不是赛事官方成绩。",
        "",
        f"折年份：{', '.join(str(year) for year in payload['years'])}",
        "",
        "## 汇总",
        "",
        "| 变体 | 处理文档 | 简要事实 | 简要文本 | 详细文本 | 简要逻辑 | 详细逻辑 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"no_history": "无历史", "strict_history": "严格历史"}
    for name in ("no_history", "strict_history"):
        aggregate = variants[name]
        values = aggregate["five_metric_proxy"]
        lines.append(
            f"| {labels[name]} | {aggregate['documents']} | "
            + " | ".join(f"{values[field]['mean']:.2%}" for field in FIVE_PROXY_FIELDS)
            + " |"
        )
    lines.extend([
        "",
        "## 严格历史相对无历史的均值变化",
        "",
    ])
    for field, value in payload["ablation_delta"]["five_metric_proxy_mean"].items():
        lines.append(f"- {field}: {value:+.2%}")
    lines.extend(["", "## 严格历史详细字段（均值 / 最差折）", ""])
    for field, values in variants["strict_history"]["detail_field_similarity"].items():
        lines.append(f"- {field}: {values['mean']:.2%} / {values['worst']:.2%}")
    lines.extend(["", "## 失败样本类型（严格历史）", ""])
    for name, count in variants["strict_history"]["failure_sample_types"].items():
        lines.append(f"- {name}: {count}")
    lines.extend([
        "",
        "## 模型调用说明",
        "",
        "本轮是确定性解析与历史模块的离线消融，模型、Embedding、Reranker 和 Qwen Planner 调用均为 0；"
        "它不能替代正式在线 ReAct 运行审计。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict multi-year temporal holdout ablations")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2014, 2016, 2017, 2018])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--conversion-cache",
        type=Path,
        help="可复用的 DOC 内容哈希转换缓存；默认放在 run-root/conversion-cache",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    input_root = args.data_root / "赛题一_训练集"
    label_root = args.data_root / "赛题一_训练集标签"
    checkpoint_path = args.run_root / "temporal-checkpoint.json"
    conversion_cache = args.conversion_cache or args.run_root / "conversion-cache"
    all_folds = _load_checkpoint(checkpoint_path, args.years)

    for variant in ("no_history", "strict_history"):
        for year in args.years:
            existing = next(
                (fold for fold in all_folds[variant] if fold.get("year") == year),
                None,
            )
            if existing and not existing.get("metrics", {}).get("failures"):
                print(f"[resume] {variant} {year}", flush=True)
                continue
            if existing:
                all_folds[variant].remove(existing)
                print(f"[rerun-failed-fold] {variant} {year}", flush=True)
            print(f"[run] {variant} {year}", flush=True)
            settings = load_settings(
                None,
                llm_mode="off",
                history_label_dir=label_root if variant == "strict_history" else None,
                history_before_year=year if variant == "strict_history" else None,
                work_dir=args.run_root / variant / str(year) / "work",
                cache_dir=conversion_cache,
            )
            runner = BridgeEvidenceOrchestrator(settings)
            loaded_years = sorted({record.cycle_year for record in runner.history.records}) if runner.history else []
            if any(loaded_year >= year for loaded_year in loaded_years):
                raise RuntimeError(f"时间泄漏门禁失败：fold={year}, loaded={loaded_years}")
            metrics = evaluate_year(
                input_root / f"{year}年",
                label_root / f"{year}年",
                settings,
                workers=args.workers,
                runner=runner,
            )
            all_folds[variant].append({
                "year": year,
                "history_years_loaded": loaded_years,
                "metrics": metrics,
            })
            _write_checkpoint(checkpoint_path, args.years, all_folds)
            print(
                f"[done] {variant} {year}: processed={metrics['processed']}, "
                f"failures={len(metrics.get('failures', []))}",
                flush=True,
            )

    aggregates = {name: _aggregate(folds) for name, folds in all_folds.items()}
    payload = {
        "schema_version": "1.0",
        "pipeline_revision": PIPELINE_REVISION,
        "metric_disclaimer": "Local proxy metrics only; not official IAIC scores.",
        "years": args.years,
        "folds": all_folds,
        "aggregates": aggregates,
        "ablation_delta": _deltas(aggregates["no_history"], aggregates["strict_history"]),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "years": args.years,
        "documents": {name: value["documents"] for name, value in aggregates.items()},
        "delta": payload["ablation_delta"]["five_metric_proxy_mean"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
