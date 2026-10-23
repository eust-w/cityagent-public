"""分治归约与组装引擎（Option 3 Reducer）。

负责将 5 个部位子任务的产物：
- 基本信息（概况与综合决策）
- 维修建议
- 桥面系病害
- 上部结构病害
- 下部结构与支座病害
进行无缝归约拼接、序号连续编排、时序基线闭环与行业约束校验。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bridge_evidence.contest.constraint_cleaner import clean_and_constrain


def reduce_subtask_results(
    overview_dict: dict[str, Any] | None,
    advice_list: list[dict[str, Any]] | None,
    deck_defects: list[dict[str, Any]] | None,
    super_defects: list[dict[str, Any]] | None,
    sub_defects: list[dict[str, Any]] | None,
    doc_path: Path | str,
    table_facts: dict[str, Any],
    temporal_info: dict[str, Any],
) -> dict[str, Any]:
    """将分治输出归约合流为官方标准三部分数据结构。"""
    basic = (overview_dict or {}).copy()
    advice = list(advice_list or [])

    # 1. 拼接全桥构件病害，按桥面系 -> 上部 -> 下部顺序
    raw_defects: list[dict[str, Any]] = []
    for d in (deck_defects or []):
        if isinstance(d, dict) and d.get("病害类型"):
            raw_defects.append(d)
    for d in (super_defects or []):
        if isinstance(d, dict) and d.get("病害类型"):
            raw_defects.append(d)
    for d in (sub_defects or []):
        if isinstance(d, dict) and d.get("病害类型"):
            raw_defects.append(d)

    # 2. 确定性表格事实纠偏（四级评分、等级与历次定检事实）
    for k in ["桥梁名称", "报告日期", "总体评分", "总体等级", "病害发展趋势与具体说明"]:
        if not basic.get(k) and table_facts.get(k):
            basic[k] = table_facts[k]

    for k in ["上部结构评分", "上部结构等级", "下部结构评分", "下部结构等级",
             "桥面系评分", "桥面系等级", "上一次总体评分", "上一次总体等级", "上一次定检结果"]:
        if table_facts.get(k):
            basic[k] = table_facts[k]

    # 3. 规范化连续重编序号
    for i, item in enumerate(advice, 1):
        item["序号"] = str(i)

    for i, item in enumerate(raw_defects, 1):
        item["序号"] = str(i)

    combined_data = {
        "基本信息": basic,
        "维修建议": advice,
        "病害明细": raw_defects,
    }

    # 4. 执行行业约束清洗器（互斥逻辑闭环、默认值清洗）
    final_data = clean_and_constrain(combined_data, doc_path, temporal_info)
    return final_data
