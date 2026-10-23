#!/usr/bin/env python3
"""Option 3 (Divide & Conquer / Map-Reduce) 专项实验脚本。

用于在隔离目录测试超长报告的分治抽取效果：
- 测试超长输入的分治处理与输出完整性
- 5 个部位子任务独立生成
- 规约归并并做确定性清洗
- 输出完整 DOCX 与统计对比指标
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bridge_evidence.contest.doc_splitter import split_document_sections
from bridge_evidence.contest.divide_conquer import (
    PROMPT_OVERVIEW,
    PROMPT_ADVICE,
    PROMPT_DEFECT_TEMPLATE,
    get_defect_prompt,
    build_chat_prompt,
    parse_subtask_json,
)
from bridge_evidence.contest.reducer import reduce_subtask_results
from bridge_evidence.contest.table_facts import extract_table_facts
from bridge_evidence.contest.temporal_router import detect_temporal_baseline
from bridge_evidence.contest.docx_writer import write_extraction_docx


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def ensure_ppu_environment() -> None:
    profile = Path("/etc/profile.d/dockerenv.sh")
    if profile.is_file() and os.environ.get("BRIDGE_PPU_ENV_READY") != "1":
        os.execv("/bin/bash", [
            "bash", "-c",
            'set -e; . /etc/profile.d/dockerenv.sh; export BRIDGE_PPU_ENV_READY=1; exec "$@"',
            "contest", sys.executable, *sys.argv,
        ])


def main():
    parser = argparse.ArgumentParser(description="Run Option 3 Divide & Conquer Experiment")
    parser.add_argument("--model", type=str, required=True, help="Path to merged model")
    parser.add_argument("--docx", type=str, required=True, help="Path to target docx")
    parser.add_argument("--output-dir", type=str, default="/tmp/contest_option3_test", help="Directory for experiment outputs")
    parser.add_argument("--max-model-len", type=int, default=65536, help="vLLM max context length")
    args = parser.parse_args()

    ensure_ppu_environment()

    target_docx = Path(args.docx)
    if not target_docx.is_file():
        log(f"错误：输入文件不存在: {target_docx}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("【Option 3: 分治切片抽取实验启动】")
    log(f"目标文档: {target_docx.name} ({target_docx.stat().st_size} bytes)")
    log(f"隔离输出目录: {out_dir}")
    log("=" * 60)

    # 1. 结构化切片与事实提取
    log(">>> 阶段 1: 文档部位切片与确定性事实抽取...")
    t_start = time.time()
    slices = split_document_sections(target_docx)
    for k, v in slices.items():
        log(f"  切片 [{k:15s}]: {len(v):6d} 字符, {len(v.splitlines()):4d} 行")

    table_facts = extract_table_facts(target_docx)
    temporal_info = detect_temporal_baseline(target_docx, table_facts)
    log(f"  确定性表格事实抽取完成 ({len(table_facts)} 个关键字段)")
    log(f"  时态基线模式: {temporal_info.get('mode')}, 有无对比: {temporal_info.get('has_comparison')}")

    # 2. 初始化大模型推理引擎
    log(">>> 阶段 2: 加载大模型与分词器...")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.92,
        enforce_eager=True,
        trust_remote_code=True,
    )
    log("大模型加载完成！开始分治子任务批处理...")

    # 3. 构造 5 个部位子任务
    subtasks = [
        ("overview", PROMPT_OVERVIEW, slices.get("overview", ""), "基本信息", 4096),
        ("advice", PROMPT_ADVICE, slices.get("advice", ""), "维修建议", 4096),
        ("deck", get_defect_prompt("桥面系"), slices.get("deck", ""), "病害明细", 6144),
        ("superstructure", get_defect_prompt("上部结构"), slices.get("superstructure", ""), "病害明细", 12288),
        ("substructure", get_defect_prompt("下部结构与支座系统"), slices.get("substructure", ""), "病害明细", 12288),
    ]

    subtask_results = {}
    total_infer_seconds = 0.0
    subtask_metrics = []

    for name, sys_prompt, content, target_key, max_toks in subtasks:
        log(f"\n--- 正在执行子任务 [{name}] ---")
        prompt_text = build_chat_prompt(tok, sys_prompt, content)
        input_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        budget = args.max_model_len - max_toks
        if len(input_ids) > budget:
            half = budget // 2
            input_ids = input_ids[:half] + input_ids[-(budget - half):]
            log(f"  [提示] 输入超预算，已裁剪至 {len(input_ids)} tokens")
        log(f"  Prompt tokens: {len(input_ids)}, 目标 Key: {target_key}, 输出上限: {max_toks}")

        sp = SamplingParams(temperature=0.0, max_tokens=max_toks)
        t0 = time.time()
        out = llm.generate([{"prompt_token_ids": input_ids}], sp)[0]
        dt = time.time() - t0
        total_infer_seconds += dt

        raw_out = out.outputs[0].text
        finish_reason = out.outputs[0].finish_reason
        out_tokens = len(out.outputs[0].token_ids)
        log(f"  生成耗时: {dt:.2f}s | 输出 Tokens: {out_tokens} | 结束原因: {finish_reason}")

        parsed = parse_subtask_json(raw_out, target_key)
        subtask_results[name] = parsed

        count = len(parsed) if isinstance(parsed, list) else (len(parsed.keys()) if isinstance(parsed, dict) else 0)
        log(f"  解析成功: {parsed is not None} | 提取项数: {count}")

        # 保存原始输出
        with open(out_dir / f"subtask_{name}_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw_out)

        subtask_metrics.append({
            "subtask": name,
            "seconds": round(dt, 2),
            "prompt_tokens": len(input_ids),
            "output_tokens": out_tokens,
            "finish_reason": finish_reason,
            "parsed_ok": parsed is not None,
            "extracted_count": count,
        })

    # 4. 规约合流与约束清洗
    log("\n>>> 阶段 3: 执行 Reducer 规约合流与行业约束校验...")
    merged_data = reduce_subtask_results(
        overview_dict=subtask_results.get("overview"),
        advice_list=subtask_results.get("advice"),
        deck_defects=subtask_results.get("deck"),
        super_defects=subtask_results.get("superstructure"),
        sub_defects=subtask_results.get("substructure"),
        doc_path=target_docx,
        table_facts=table_facts,
        temporal_info=temporal_info,
    )

    out_docx_path = out_dir / (target_docx.stem + "_option3.docx")
    write_extraction_docx(merged_data, str(out_docx_path))

    total_time = time.time() - t_start
    total_defects = len(merged_data.get("病害明细") or [])
    total_advice = len(merged_data.get("维修建议") or [])
    total_tokens = sum(m["output_tokens"] for m in subtask_metrics)

    summary = {
        "experiment": "Option 3: Divide & Conquer",
        "target_file": target_docx.name,
        "total_wall_time_seconds": round(total_time, 2),
        "total_infer_seconds": round(total_infer_seconds, 2),
        "total_output_tokens": total_tokens,
        "total_defects_extracted": total_defects,
        "total_advice_extracted": total_advice,
        "output_docx": str(out_docx_path),
        "output_docx_size_bytes": out_docx_path.stat().st_size,
        "subtasks": subtask_metrics,
        "comparison_with_baseline": {
            "monolithic_baseline_time_seconds": 3635,
            "monolithic_baseline_tokens": 32768,
            "monolithic_finish_reason": "length (TRUNCATED)",
            "option3_total_tokens": total_tokens,
            "token_reduction_percent": round((1 - total_tokens / 32768) * 100, 2),
            "speedup_ratio": round(3635 / total_infer_seconds, 2) if total_infer_seconds > 0 else 0,
            "truncation_eliminated": all(m["finish_reason"] == "stop" for m in subtask_metrics),
        }
    }

    summary_file = out_dir / "option3_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("=" * 60)
    log("【Option 3 实验完成】")
    log(f"总推理时间: {total_infer_seconds:.2f}s (基线单体: 3635s, 提速: {summary['comparison_with_baseline']['speedup_ratio']}x)")
    log(f"总生成 Token: {total_tokens} (基线单体: 32768, 减少: {summary['comparison_with_baseline']['token_reduction_percent']}%)")
    log(f"截断彻底消除: {summary['comparison_with_baseline']['truncation_eliminated']}")
    log(f"最终交付成果: {total_defects} 条病害, {total_advice} 条建议")
    log(f"生成 DOCX: {out_docx_path} ({out_docx_path.stat().st_size} bytes)")
    log(f"实验总结已写入: {summary_file}")
    log("=" * 60)


if __name__ == "__main__":
    main()
