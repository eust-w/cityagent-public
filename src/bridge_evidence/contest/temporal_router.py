"""时态路由器（Temporal Baseline Router）。

负责判定检测报告是否具有“历次对比年度”，并锚定时态基线。
- 明确无对比：将所有时态字段（是否新增/上次状态/发展程度/发展趋势）闭合为规范值“否 / 无 / 无 / 无”，
  彻底消除大模型无依据乱猜历史病害的问题。
- 明确有对比：提取报告中的历次定检对比上下文，作为模型判断依据。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from docx import Document


def detect_temporal_baseline(doc_path: Path | str, facts: dict[str, Any]) -> dict[str, Any]:
    """检测报告的时态对比属性。

    返回：
      {
        "has_comparison": bool,
        "mode": "with_comparison" | "no_comparison",
        "comparison_summary": str,
      }
    """
    path = Path(doc_path)
    stem = path.stem

    # 1. 显式规则检测文件名/标题
    if "无对比年度" in stem:
        return {"has_comparison": False, "mode": "no_comparison", "comparison_summary": ""}
    if "含有对比年度" in stem or "有对比年度" in stem:
        return {"has_comparison": True, "mode": "with_comparison", "comparison_summary": facts.get("上一次定检结果", "")}

    # 2. 检查 "原A级，现B级" 样式
    if re.search(r"原[A-E]\s*[类级].*?现[A-E]\s*[类级]", stem):
        return {"has_comparison": True, "mode": "with_comparison", "comparison_summary": facts.get("上一次定检结果", "")}
    if re.search(r"现[A-E]\s*[类级]", stem) and not re.search(r"原[A-E]", stem):
        # 仅有“现X级”通常说明无前次对比
        return {"has_comparison": False, "mode": "no_comparison", "comparison_summary": ""}

    # 3. 检查已提取事实中的上一次定检信息
    if facts.get("上一次总体评分") or facts.get("上一次定检结果"):
        return {"has_comparison": True, "mode": "with_comparison", "comparison_summary": facts.get("上一次定检结果", "")}

    # 4. 扫描正文前 30 个段落中是否存在历次检测证据
    doc = Document(str(doc_path))
    has_comp_signals = False
    no_comp_signals = False
    comparison_snippets = []

    for p in doc.paragraphs[:40]:
        text = p.text.strip()
        if not text:
            continue
        if re.search(r"(?:上一次|上一年度|上期|前次|历次).{0,20}?(?:定检|检测|评定|病害|状况)", text):
            has_comp_signals = True
            comparison_snippets.append(text)
        if re.search(r"(?:初次|首次|无往年|未见往年|无前次|无对比年度)", text):
            no_comp_signals = True

    if no_comp_signals and not has_comp_signals:
        return {"has_comparison": False, "mode": "no_comparison", "comparison_summary": ""}

    if has_comp_signals:
        summary = "；".join(comparison_snippets[:3])
        return {"has_comparison": True, "mode": "with_comparison", "comparison_summary": summary}

    # 默认兜底：如果没有发现任何上一次评分、对比表或对比段落，视为无对比年度
    return {"has_comparison": False, "mode": "no_comparison", "comparison_summary": ""}


def apply_temporal_closure(data: dict[str, Any], temporal_info: dict[str, Any]) -> dict[str, Any]:
    """若属于无对比年度，强制封闭时态字段，消除模型胡编乱造。"""
    if not temporal_info.get("has_comparison", False):
        basic = data.setdefault("基本信息", {})
        basic["上一次总体评分"] = ""
        basic["上一次总体等级"] = ""
        basic["上一次定检结果"] = "无"
        basic["病害发展趋势与具体说明"] = "无"

        # 病害列表强制置为 否/无/无
        defects = data.setdefault("病害明细", [])
        for row in defects:
            if isinstance(row, dict):
                row["是否新增"] = "否"
                row["上一次定检状态"] = "无"
                row["发展程度"] = "无"

    return data
