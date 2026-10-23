from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from bridge_evidence.config import load_settings
from bridge_evidence.evaluation.metrics import evaluate_year, write_metrics
from bridge_evidence.orchestrator import BridgeEvidenceOrchestrator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bridge-evidence", description="桥鉴城市桥梁定检报告证据闭环智能体")
    parser.add_argument("--config", type=Path, help="JSON 配置文件")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="批量处理 Word 报告")
    run.add_argument("--input-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--json-dir", type=Path)
    run.add_argument("--model")
    run.add_argument("--base-url")
    run.add_argument("--llm-mode", choices=["off", "auto", "on"])
    run.add_argument("--history-label-dir", type=Path)
    run.add_argument("--history-before-year", type=int)
    run.add_argument("--require-history", action="store_true", default=None)
    run.add_argument("--limit", type=int)
    run.add_argument("--workers", type=int, default=1)

    single = subparsers.add_parser("single", help="处理单个报告并打印状态摘要")
    single.add_argument("--input", type=Path, required=True)
    single.add_argument("--output-dir", type=Path)
    single.add_argument("--json-dir", type=Path)
    single.add_argument("--model")
    single.add_argument("--base-url")
    single.add_argument("--llm-mode", choices=["off", "auto", "on"])
    single.add_argument("--history-label-dir", type=Path)
    single.add_argument("--history-before-year", type=int)
    single.add_argument("--require-history", action="store_true", default=None)

    evaluate = subparsers.add_parser("evaluate", help="对训练标签运行离线代理评测")
    evaluate.add_argument("--input-dir", type=Path, required=True)
    evaluate.add_argument("--label-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--llm-mode", choices=["off", "auto", "on"], default="off")
    evaluate.add_argument("--history-label-dir", type=Path)
    evaluate.add_argument("--history-before-year", type=int)
    evaluate.add_argument("--require-history", action="store_true", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    overrides = {
        "model": getattr(args, "model", None),
        "base_url": getattr(args, "base_url", None),
        "llm_mode": getattr(args, "llm_mode", None),
        "history_label_dir": getattr(args, "history_label_dir", None),
        "history_before_year": getattr(args, "history_before_year", None),
        "require_history": getattr(args, "require_history", None),
    }
    settings = load_settings(args.config, **overrides)
    settings.cache_dir = settings.cache_dir.resolve()
    settings.work_dir = settings.work_dir.resolve()

    if args.command == "run":
        runner = BridgeEvidenceOrchestrator(settings)
        states, failures = runner.process_directory(args.input_dir, args.output_dir, json_dir=args.json_dir, limit=args.limit, workers=args.workers)
        task_calls: Counter[str] = Counter()
        task_cache_hits: Counter[str] = Counter()
        model_calls: Counter[str] = Counter()
        model_cache_hits: Counter[str] = Counter()
        for state in states:
            task_calls.update(state.metrics.task_calls)
            task_cache_hits.update(state.metrics.task_cache_hits)
            model_calls.update(state.metrics.model_calls)
            model_cache_hits.update(state.metrics.model_cache_hits)
        summary = {
            "processed": len(states),
            "passed": sum(state.validation.passed for state in states),
            "failed": len(failures),
            "llm_calls": sum(state.metrics.llm_calls for state in states),
            "chat_calls": sum(state.metrics.chat_calls for state in states),
            "vision_calls": sum(state.metrics.vision_calls for state in states),
            "embedding_calls": sum(state.metrics.embedding_calls for state in states),
            "reranker_calls": sum(state.metrics.reranker_calls for state in states),
            "planner_calls": sum(state.metrics.planner_calls for state in states),
            "llm_cache_hits": sum(state.metrics.llm_cache_hits for state in states),
            "task_calls": dict(task_calls),
            "task_cache_hits": dict(task_cache_hits),
            "model_calls": dict(model_calls),
            "model_cache_hits": dict(model_cache_hits),
            "online_audit_passed": (
                None
                if settings.llm_mode == "off"
                else bool(states) and all(state.metrics.online_audit_passed for state in states)
            ),
            "failures": [{"file": str(path), "error": str(error)} for path, error in failures],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if failures:
            raise SystemExit(2)
        return

    if args.command == "single":
        runner = BridgeEvidenceOrchestrator(settings)
        state = runner.process(args.input, args.output_dir, json_dir=args.json_dir, render=bool(args.output_dir))
        print(json.dumps({
            "bridge": state.bridge.name.value,
            "report_date": state.bridge.report_date.value,
            "overall": state.scores.overall.model_dump(mode="json"),
            "diseases": len(state.diseases),
            "recommendations": len(state.recommendations),
            "validation": state.validation.model_dump(mode="json"),
            "metrics": state.metrics.model_dump(mode="json"),
        }, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate":
        metrics = evaluate_year(args.input_dir, args.label_dir, settings, args.limit, args.workers)
        if args.output:
            write_metrics(metrics, args.output)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    raise AssertionError(args.command)


if __name__ == "__main__":
    main(sys.argv[1:])
