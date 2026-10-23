"""病害表格精准切片与行抽取模块（Defect Chunker）。

定位报告核心附表中的病害检查记录表，过滤照片说明/示意图/试验计算等干扰行，
直接输出高精度的病害明细基础行，解决保头保尾截断导致病害行物理遗失的问题。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from docx import Document

from bridge_evidence.domain_rules import (
    ACTIVE_DEFECT_RE,
    canonical_disease_from_text,
    is_non_disease_label,
)


def _clean_text(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def extract_defect_rows_from_tables(doc_path: Path | str) -> list[dict[str, str]]:
    """遍历 DOCX 所有表格，直接抽取构件病害检查记录行。"""
    doc = Document(str(doc_path))
    defects: list[dict[str, str]] = []
    seen_signatures = set()

    pos_col = None
    type_col = None
    desc_col = None

    for t_idx, table in enumerate(doc.tables):
        if len(table.rows) < 1:
            continue

        prev_elem = table._element.getprevious()
        caption = _clean_text(prev_elem.text) if prev_elem is not None and hasattr(prev_elem, "text") else ""
        header_cells = [_clean_text(c.text) for c in table.rows[0].cells]
        header_str = caption + " " + "|".join(header_cells)

        # 排除计算、荷载试验、材料物理试验、扣分等无关大表
        if any(ign in header_str for ign in (
            "荷载试验", "静载", "动载", "有限元", "检算", "承载能力", "电位",
            "电阻率", "氯离子", "永久观测点", "交通量", "设备编号", "扣分",
            "DPij", "MDPi", "权重", "评价等级", "自振频率", "应变增大系数"
        )):
            pos_col = type_col = desc_col = None
            continue

        has_pos = any(k in header_str for k in ("病害位置", "构件位置", "位置", "构件部位", "部位", "桥跨", "桥幅", "构件名称"))
        has_type = any(k in header_str for k in ("病害类型", "病害种类", "损坏类型", "损坏种类", "病害名称", "种类"))
        has_desc = any(k in header_str for k in ("病害描述", "病害情况", "具体情况", "检查结果", "检测结果", "损坏评价", "情况", "说明"))

        # 支持无独立标题/表头的续表（继承上一表的列索引）
        is_continuation = (
            caption == ""
            and len(header_cells) >= 3
            and (header_cells[0] == "" or re.match(r"^\d+$", header_cells[0]))
            and type_col is not None
        )

        if not is_continuation:
            if len(table.rows) < 2 or not (has_type and (has_desc or has_pos)):
                pos_col = type_col = desc_col = None
                continue

            pos_col = None
            type_col = None
            desc_col = None
            for ci, h in enumerate(header_cells):
                if any(k in h for k in ("病害位置", "构件位置", "位置", "构件部位", "部位", "桥幅", "构件名称")) and pos_col is None:
                    pos_col = ci
                elif any(k in h for k in ("病害类型", "病害种类", "损坏类型", "损坏种类", "病害名称", "种类")) and type_col is None:
                    type_col = ci
                elif any(k in h for k in ("病害描述", "病害情况", "具体情况", "检查结果", "检测结果", "损坏评价", "情况")) and desc_col is None:
                    desc_col = ci

            start_row = 1
        else:
            start_row = 0

        if type_col is None:
            continue

        for r_idx in range(start_row, len(table.rows)):
            cells = [_clean_text(c.text) for c in table.rows[r_idx].cells]
            row_full = " ".join(cells)

            # 过滤照片行、说明行、示意图行
            if not row_full or len(row_full) < 2:
                continue
            if re.search(r"^(?:照片|\d+[）\)]\s*(?:照片|病害分布示意图|缺陷照片))", row_full):
                continue
            if "病害分布示意图" in row_full or "缺陷照片" in row_full:
                continue

            pos_val = cells[pos_col] if (pos_col is not None and len(cells) > pos_col) else ""
            type_val = cells[type_col] if len(cells) > type_col else ""
            desc_val = cells[desc_col] if (desc_col is not None and len(cells) > desc_col) else ""

            # 过滤非病害标签
            if is_non_disease_label(type_val) or is_non_disease_label(desc_val):
                continue
            if not pos_val and not desc_val and not type_val:
                continue

            # 过滤“未见明显病害 / 无病害 / 正常 / 完好”行
            if re.search(r"^(?:无|未见明显病害|无明显病害|外观良好|正常|完好|未发现病害|未见异常|未见缺损)$", type_val):
                continue
            if re.search(r"^(?:无|未见明显病害|无明显病害|外观良好|正常|完好|未发现病害|未见异常|未见缺损)$", desc_val):
                continue

            # 规范化病害类型
            norm_type = canonical_disease_from_text(type_val) or type_val
            if not norm_type and desc_val:
                norm_type = canonical_disease_from_text(desc_val) or "破损"

            if not desc_val:
                desc_val = f"{pos_val}{norm_type}" if pos_val else norm_type
            if not pos_val:
                pos_val = "全桥"

            # 防重复（同一位置、类型和描述）
            sig = (pos_val, norm_type, desc_val)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            defects.append({
                "序号": str(len(defects) + 1),
                "病害部位": pos_val,
                "病害类型": norm_type,
                "病害描述": desc_val,
                "是否新增": "是",
                "上一次定检状态": "上一次定检未发现",
                "发展程度": "无",
            })

    return defects
