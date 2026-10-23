#!/usr/bin/env python3
"""Summarize existing PPU API benchmark runs offline; never runs inference.

  python experiments/summarize_ppu_framework_benchmark.py /runs/vllm-fields \
    /runs/sglang-fields /runs/vllm-throughput /runs/sglang-throughput \
    --out-json /reports/comparison.json --out-md /reports/comparison.md

Output files must be new. Results are grouped by run, probe type and full request
configuration hash, excluding only the model alias (which is recorded separately).
ignore_eos=true identifies a fixed-length throughput probe. All other requests
are classified as fields probes under this experiment's protocol. Warmups and
failed requests do not enter latency/rate aggregates. Missing usage never becomes
zero tokens. Fixed-length metrics additionally require successful completion,
finish_reason=length, and completion_tokens exactly equal to max_tokens. Decode
rates are observed-SSE approximations, not exact token timing.

Bridge/date comparisons are explicitly HUMAN EXPECTATIONS, not authoritative
labels, source verification, task accuracy or F1. Length finishes cannot pass the
manual comparison, even if an apparent JSON object exists in the partial text.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any
import unicodedata


METRICS = ("ttft_observed_seconds", "end_to_end_seconds",
           "approx_completion_tokens_per_decode_second",
           "completion_tokens_per_end_to_end_second")
FIELD_ALIASES = {
    "bridge_name": {"bridge_name", "bridgeName", "桥名", "桥梁名称", "桥梁名"},
    "report_date": {"report_date", "reportDate", "报告日期", "报告时间", "报告出具日期"},
}


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def aggregate(values: list[Any]) -> dict[str, Any]:
    valid = [value for value in values if numeric(value) and value >= 0]
    return {"count": len(valid), "median": statistics.median(valid) if valid else None,
            "min": min(valid) if valid else None, "max": max(valid) if valid else None}


def normalize(value: str, field: str) -> str:
    value = re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))
    if field == "report_date":
        chinese = re.fullmatch(r"([〇零一二三四五六七八九]{4})年([一二三四五六七八九十]{1,3})月([一二三四五六七八九十]{1,3})日", value)
        if chinese:
            digits = {char: number for number, char in enumerate("零一二三四五六七八九")}
            digits["〇"] = 0

            def integer(text: str) -> int:
                if "十" in text:
                    tens, units = text.split("十")
                    return (digits[tens] if tens else 1) * 10 + (digits[units] if units else 0)
                return digits[text]

            year, month, day = chinese.groups()
            try:
                value = f'{int("".join(str(digits[char]) for char in year))}-{integer(month)}-{integer(day)}'
            except (ValueError, KeyError):
                pass
        match = re.fullmatch(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", value)
        if match:
            try:
                return date(*map(int, match.groups())).isoformat()
            except ValueError:
                pass
    return value


def json_objects(text: str) -> list[Any]:
    """Accept one complete JSON response/fence, rejecting duplicate object keys."""
    source = text.strip()
    if source.startswith("```"):
        fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", source,
                             flags=re.DOTALL | re.IGNORECASE)
        if not fence:
            return []
        source = fence.group(1)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-JSON constant: {value}")

    try:
        value = json.loads(source, object_pairs_hook=unique_object,
                           parse_constant=reject_constant)
        return [value] if isinstance(value, (dict, list)) else []
    except ValueError:
        return []


def usable_token_count(row: dict[str, Any]) -> bool:
    usage = row.get("usage")
    count = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return isinstance(count, int) and not isinstance(count, bool) and count >= 0


def fixed_length_protocol(row: dict[str, Any]) -> dict[str, Any]:
    request = row.get("request") or {}
    maximum = request.get("max_tokens")
    valid_maximum = isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0
    reasons = []
    if row.get("success") is not True:
        reasons.append("request_not_successful")
    if request.get("ignore_eos") is not True:
        reasons.append("ignore_eos_not_explicitly_true")
    if not usable_token_count(row):
        reasons.append("missing_or_invalid_completion_usage")
    if not valid_maximum:
        reasons.append("missing_or_invalid_max_tokens")
    if usable_token_count(row) and valid_maximum and row["usage"]["completion_tokens"] != maximum:
        reasons.append("completion_tokens_not_equal_to_max_tokens")
    if row.get("finish_reason") != "length":
        reasons.append("finish_reason_not_length")
    return {"sample_id": row.get("sample_id"), "repetition": row.get("repetition"),
            "protocol_valid": not reasons, "invalid_reasons": reasons,
            "completion_tokens": row["usage"]["completion_tokens"] if usable_token_count(row) else None,
            "max_tokens": maximum, "finish_reason": row.get("finish_reason")}


def configuration(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    request = row.get("request") or {}
    config = {key: value for key, value in request.items() if key != "model"}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return config, digest


def compare_fields(row: dict[str, Any], expected: dict[str, str]) -> dict[str, Any]:
    objects = json_objects(row.get("content_text") or "")
    values: dict[str, list[str]] = {field: [] for field in FIELD_ALIASES}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                for field, aliases in FIELD_ALIASES.items():
                    if key in aliases and isinstance(item, str) and item not in values[field]:
                        values[field].append(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for value in objects:
        visit(value)
    eligible = bool(row.get("success") is True and row.get("finish_reason") == "stop" and objects)
    comparisons = {}
    for field in FIELD_ALIASES:
        normalized = {normalize(value, field) for value in values[field]}
        matched = len(normalized) == 1 and normalize(expected[field], field) in normalized
        comparisons[field] = {"observed_values": values[field], "expected": expected[field],
                              "ambiguous": len(normalized) > 1,
                              "matches_human_expectation": matched if eligible else None}
    return {"sample_id": row.get("sample_id"), "repetition": row.get("repetition"),
            "eligible": eligible, "finish_reason": row.get("finish_reason"),
            "parsed_json_count": len(objects), "complete_response_json_valid": bool(objects),
            "fields": comparisons,
            "all_expected_fields_match": all(value["matches_human_expectation"] is True
                                             for value in comparisons.values()) if eligible else None,
            "comparison_basis": "human_expectation_not_verified_label",
            "quality_evaluated": False}


def read_run(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {directory}")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    journal = directory / "results.jsonl"
    warnings, rows = [], []
    if journal.exists():
        for number, line in enumerate(journal.read_text().splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{journal}:{number}: expected object")
                rows.append(row)
    else:
        warnings.append("results.jsonl missing; using saved request result.json files")
        for path in sorted(directory.glob("request-*/result.json")):
            rows.append(json.loads(path.read_text()))
    saved_results = list(directory.glob("request-*/result.json"))
    if saved_results and len(saved_results) != len(rows):
        warnings.append("Journal and per-request result counts differ; inspect interrupted writes")
    if not rows:
        warnings.append("No request receipts found; no measurements inferred from summary.json")
    # Recover a missing usage object only from that request's persisted SSE events.
    for index, row in enumerate(rows):
        event_path = directory / f"request-{index:04d}" / "events.jsonl"
        if row.get("first_output_delta_seconds") is None or not event_path.exists():
            continue
        archived_path = event_path.with_name("result.json")
        if not archived_path.exists():
            continue
        archived = json.loads(archived_path.read_text())
        if not row.get("started_at") or archived.get("started_at") != row["started_at"]:
            continue
        events = [json.loads(line) for line in event_path.read_text().splitlines() if line.strip()]
        usage_events = [event for event in events
                        if isinstance(event.get("event"), dict) and event["event"].get("usage")]
        if usage_events and row.get("usage") is None:
            row["usage"] = usage_events[-1]["event"]["usage"]
    return manifest, rows, warnings


def summarize_run(directory: Path, expected: dict[str, str]) -> dict[str, Any]:
    manifest, rows, warnings = read_run(directory)
    measured = [row for row in rows if row.get("phase") == "measured"]
    warmups = [row for row in rows if row.get("phase") == "warmup"]
    if len(measured) + len(warmups) != len(rows):
        warnings.append("Receipts with unknown phase excluded from measured statistics")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in measured:
        request = row.get("request") or {}
        mode = "fixed_length_throughput" if request.get("ignore_eos") is True else "fields"
        _, digest = configuration(row)
        grouped.setdefault((mode, digest), []).append(row)
    groups = []
    for (mode, digest), members in grouped.items():
        successful = [row for row in members if row.get("success") is True]
        config, _ = configuration(members[0])
        checks = [fixed_length_protocol(row) for row in members] if mode == "fixed_length_throughput" else []
        performance_rows = [row for row, check in zip(members, checks) if check["protocol_valid"]] if checks else successful
        # Without server token usage, a saved or inferred token rate is not credible.
        metrics = {}
        for key in METRICS:
            eligible = [row for row in performance_rows if "tokens_per" not in key or usable_token_count(row)]
            metrics[key] = aggregate([row.get(key) for row in eligible])
        groups.append({
            "probe_type": mode, "request_config_sha256": digest,
            "request_options": {key: value for key, value in config.items() if key != "messages"},
            "messages_sha256": hashlib.sha256(json.dumps(config.get("messages"), sort_keys=True,
                                                          ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
            "request_models": sorted({str(row.get("request", {}).get("model")) for row in members}),
            "response_models": sorted({str(row.get("response_model")) for row in members}),
            "sample_ids": sorted({str(row.get("sample_id")) for row in members}),
            "request_count": len(members), "success_count": len(successful),
            "failure_count": len(members) - len(successful),
            "protocol_valid_count": sum(check["protocol_valid"] for check in checks) if checks else None,
            "protocol_invalid_count": sum(not check["protocol_valid"] for check in checks) if checks else None,
            "fixed_length_protocol_checks": checks,
            "performance_request_count": len(performance_rows),
            "missing_completion_usage_count": sum(not usable_token_count(row) for row in successful),
            "finish_reasons": dict(Counter(str(row.get("finish_reason")) for row in members)),
            "length_finish_count": sum(row.get("finish_reason") == "length" for row in successful),
            "max_tokens_values": sorted({row.get("request", {}).get("max_tokens") for row in members}, key=str),
            "prompt_tokens": aggregate([row.get("usage", {}).get("prompt_tokens")
                                         for row in performance_rows if isinstance(row.get("usage"), dict)]),
            "completion_tokens": aggregate([row["usage"]["completion_tokens"]
                                             for row in performance_rows if usable_token_count(row)]),
            "metrics": metrics,
            "field_comparisons": [compare_fields(row, expected) for row in members] if mode == "fields" else [],
            "errors": [row.get("error") for row in members if row.get("success") is not True],
        })
    return {"directory": str(directory.resolve()),
            "label": manifest.get("arguments", {}).get("label", directory.name),
            "receipt_count": len(rows), "measured_request_count": len(measured),
            "warmup_request_count": len(warmups),
            "warmup_failure_count": sum(row.get("success") is not True for row in warmups),
            "warnings": warnings, "groups": groups}


def markdown(report: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    def metric(value: dict[str, Any]) -> str:
        if value["count"] == 0:
            return "—"
        return f'{value["median"]:.3f} [{value["min"]:.3f}, {value["max"]:.3f}]'

    lines = ["PPU 框架实测汇总", "", "数值为中位数 [最小值, 最大值]。排除 warmup 和失败请求；解码速度是 SSE 时间估计。",
             "固定长度探针还要求 completion_tokens 等于 max_tokens、finish_reason 为 length，且 usage 有效；协议无效请求不进入性能统计。请求成功数独立保留。", "",
             "| 运行 | 配置哈希前12位 | 探针 | 请求成功/失败 | 协议有效/无效 | TTFT 秒 | 端到端 秒 | 近似解码 tokens/s | 缺 usage | length 结束 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    fields = []
    for run in report["runs"]:
        for group in run["groups"]:
            metrics = group["metrics"]
            protocol = "—" if group["protocol_valid_count"] is None else f'{group["protocol_valid_count"]}/{group["protocol_invalid_count"]}'
            lines.append("| " + " | ".join(map(cell, [run["label"], group["request_config_sha256"][:12], group["probe_type"],
                f'{group["success_count"]}/{group["failure_count"]}',
                protocol,
                metric(metrics["ttft_observed_seconds"]), metric(metrics["end_to_end_seconds"]),
                metric(metrics["approx_completion_tokens_per_decode_second"]),
                group["missing_completion_usage_count"], group["length_finish_count"]])) + " |")
            for comparison in group["field_comparisons"]:
                values = comparison["fields"]
                fields.append([run["label"], group["request_config_sha256"][:12], comparison["sample_id"], comparison["repetition"],
                    "; ".join(values["bridge_name"]["observed_values"]) or "缺失",
                    "; ".join(values["report_date"]["observed_values"]) or "缺失",
                    "不适用" if not comparison["eligible"] else
                    ("符合人工预期" if comparison["all_expected_fields_match"] else "与人工预期有差异")])
    lines.extend(["", "字段人工预期对照", "",
                  f'预期桥名：{report["human_expectations"]["bridge_name"]}；报告日：{report["human_expectations"]["report_date"]}。',
                  "这不是已核实标签、来源审核、完整任务准确率或 F1。length 结束不能作为字段质量通过。", "",
                  "| 运行 | 配置哈希前12位 | 样本 | 重复 | 桥名输出 | 日期输出 | 人工预期对照 |",
                  "|---|---|---|---:|---|---|---|"])
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in fields)
    for run in report["runs"]:
        lines.extend(["", f'{cell(run["label"])}：测量 {run["measured_request_count"]} 次，'
                      f'warmup {run["warmup_request_count"]} 次（失败 {run["warmup_failure_count"]} 次）。'])
        for group in run["groups"]:
            lines.extend(["", f'- 完整请求配置哈希：`{group["request_config_sha256"]}`；'
                          f'输出上限：{group["max_tokens_values"]}；模型别名：{cell(group["request_models"])}。'])
        lines.extend(f"- {warning}" for warning in run["warnings"])
    lines.extend(["", "配置哈希包含完整请求（messages、seed、thinking、输出上限等），仅排除模型别名。仅比较相同请求配置、模型权重和运行条件的数据。此汇总不能自行证明设备独占或权重一致。", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--expected-bridge", default="合成异型梁桥")
    parser.add_argument("--expected-date", default="2024-05-16")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()
    expected = {"bridge_name": args.expected_bridge, "report_date": args.expected_date}
    for path in (args.out_json, args.out_md):
        if path and path.exists():
            parser.error(f"Refusing to overwrite {path}")
    report = {"schema_version": 2, "generated_at": datetime.now(timezone.utc).isoformat(),
              "human_expectations": expected, "expectations_are_verified_labels": False,
              "quality_evaluated": False, "runs": [summarize_run(path, expected) for path in args.run_dirs]}
    for path, content in ((args.out_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
                          (args.out_md, markdown(report))):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
    if not args.out_json and not args.out_md:
        print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
