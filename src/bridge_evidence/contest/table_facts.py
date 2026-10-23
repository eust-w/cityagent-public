"""确定性提取桥梁基本信息与评定表事实（无需大模型，100% 规则抽取）。

覆盖字段：
  - 桥梁名称
  - 报告日期
  - 总体评分、总体等级
  - 上部结构评分、上部结构等级
  - 下部结构评分、下部结构等级
  - 桥面系评分、桥面系等级
  - 上一次总体评分、上一次总体等级、上一次定检结果
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from docx import Document

FLOAT_RE = r"(\d{1,3}(?:\.\d+)?)"
GRADE_RE = re.compile(r"([ABCDE])\s*(?:级|类)?", re.I)


def _clean_str(text: str) -> str:
    return re.sub(r"\s+", "", text).strip()


def _normalize_grade(val: str | None) -> str:
    if not val:
        return ""
    m = GRADE_RE.search(val)
    if m:
        return f"{m.group(1).upper()}级"
    return ""


def _normalize_date(text: str) -> str:
    if not text:
        return ""
    # 匹配 2018年12月12日 / 2018.12.12 / 2018-12-12 / 二〇一八年十二月十二日
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?", text)
    if m:
        return f"{m.group(1)}年{int(m.group(2))}月{int(m.group(3))}日"
    m = re.search(r"(\d{4})[./\-](\d{1,2})[./\-](\d{1,2})", text)
    if m:
        return f"{m.group(1)}年{int(m.group(2))}月{int(m.group(3))}日"
    return ""


def _safe_float(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(FLOAT_RE, text.replace(" ", ""))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def extract_table_facts(doc_path: Path | str) -> dict[str, Any]:
    """从 DOCX 表格及段落中提取确定性基本信息字典。"""
    facts: dict[str, Any] = {
        "桥梁名称": "",
        "报告日期": "",
        "总体评分": "",
        "总体等级": "",
        "上部结构评分": "",
        "上部结构等级": "",
        "下部结构评分": "",
        "下部结构等级": "",
        "桥面系评分": "",
        "桥面系等级": "",
        "上一次总体评分": "",
        "上一次总体等级": "",
        "上一次定检结果": "",
    }

    doc = Document(str(doc_path))

    # 1. 从文件名和前几个段落预取桥梁名称
    stem = Path(doc_path).stem
    # 去除常见编号与后缀：如 "10、上跨铁路桥", "10、上跨铁路桥-含有对比年度的信息提取报告"
    clean_stem = re.sub(r"^\d+[、.\-_]\s*", "", stem)
    clean_stem = re.sub(r"[-_](?:含有|无)对比年度.*$", "", clean_stem)
    clean_stem = re.sub(r"报告.*$", "", clean_stem)
    clean_stem = re.sub(r"桥梁.*$", "", clean_stem)
    if clean_stem and not facts["桥梁名称"]:
        facts["桥梁名称"] = clean_stem.strip()

    # 2. 遍历表格寻找工程名称/桥梁名称、报告日期、技术状况评定表
    for t_idx, table in enumerate(doc.tables):
        if not table.rows:
            continue

        # 2.1 封面/概况表 (通常在前 6 个表)
        if t_idx < 8:
            for r_idx, row in enumerate(table.rows):
                cells_text = [c.text.strip() for c in row.cells]
                row_str = " | ".join(cells_text)

                # 桥名查找
                for i, c in enumerate(cells_text[:-1]):
                    c_clean = _clean_str(c)
                    if c_clean in ("桥梁名称", "工程名称", "桥梁全称"):
                        val = cells_text[i+1].strip()
                        if val and len(val) >= 2 and val not in ("桥梁名称", "工程名称"):
                            facts["桥梁名称"] = val

                # 日期查找
                for i, c in enumerate(cells_text[:-1]):
                    c_clean = _clean_str(c)
                    if any(k in c_clean for k in ("报告日期", "出具日期", "评估日期", "完成日期", "检测日期")):
                        d = _normalize_date(cells_text[i+1])
                        if d and not facts["报告日期"]:
                            facts["报告日期"] = d

                # 封面单独出现的日期行
                if not facts["报告日期"] and len(row_str) < 50:
                    d = _normalize_date(row_str)
                    if d:
                        facts["报告日期"] = d

        # 2.2 技术状况评定表（核心表）
        # 特征：表头含 '部位名称'/'桥梁部位'，且包含 '技术状况指数' 或 'BCI'
        header_text = "".join(c.text for c in table.rows[0].cells).replace(" ", "")
        if ("部位名称" in header_text or "桥梁部位" in header_text) and ("技术状况" in header_text or "BCI" in header_text):
            # 找到列位置
            header_cells = [c.text.strip().replace("\n", " ") for c in table.rows[0].cells]
            score_col = None
            grade_col = None
            overall_col = None
            overall_grade_col = None

            for ci, h in enumerate(header_cells):
                h_clean = _clean_str(h)
                if "部位技术状况指数" in h_clean:
                    score_col = ci
                elif "技术状况指数" in h_clean and score_col is None:
                    score_col = ci
                elif h_clean in ("技术状况", "等级", "评定等级", "技术状况等级") and grade_col is None:
                    grade_col = ci
                elif h_clean == "BCI" or h_clean in ("全桥BCI", "总体评分", "桥梁技术状况指数", "总体BCI"):
                    overall_col = ci
                elif "整体技术状况等级" in h_clean or "总体等级" in h_clean or "评估等级" in h_clean:
                    overall_grade_col = ci

            # 默认兜底列索引
            if score_col is None:
                score_col = 1
            if grade_col is None:
                grade_col = 2

            for row in table.rows[1:]:
                cells = [c.text.strip() for c in row.cells]
                if not cells:
                    continue
                part = _clean_str(cells[0])

                # 提取评分与等级
                val_score = _safe_float(cells[score_col]) if len(cells) > score_col else None
                val_grade = _normalize_grade(cells[grade_col]) if len(cells) > grade_col else ""

                # 有些表如果列错位，权重在 0~1 之间，寻找大于 20 的有效评分
                if val_score is not None and 0 < val_score <= 1:
                    for test_c in cells[1:]:
                        candidate = _safe_float(test_c)
                        if candidate is not None and 20 <= candidate <= 100:
                            val_score = candidate
                            break

                if "桥面" in part:
                    if val_score is not None:
                        facts["桥面系评分"] = f"{val_score:.2f}"
                    if val_grade:
                        facts["桥面系等级"] = val_grade
                elif "上部" in part:
                    if val_score is not None:
                        facts["上部结构评分"] = f"{val_score:.2f}"
                    if val_grade:
                        facts["上部结构等级"] = val_grade
                elif "下部" in part:
                    if val_score is not None:
                        facts["下部结构评分"] = f"{val_score:.2f}"
                    if val_grade:
                        facts["下部结构等级"] = val_grade

                # 提取总体评分与总体等级
                if overall_col is not None and len(cells) > overall_col:
                    ov_s = _safe_float(cells[overall_col])
                    if ov_s is not None and 20 <= ov_s <= 100 and not facts["总体评分"]:
                        facts["总体评分"] = f"{ov_s:.2f}"
                if overall_grade_col is not None and len(cells) > overall_grade_col:
                    ov_g = _normalize_grade(cells[overall_grade_col])
                    if ov_g and not facts["总体等级"]:
                        facts["总体等级"] = ov_g

        # 2.3 历次检测对比表（如 Table 8 类型的表）
        # 特征：表头含 '上一次'/'上期'/'历次'/'对比'
        if any(term in header_text for term in ("上一次", "上期", "历次", "对比", "发展状况")):
            prev_results = []
            trend_parts = []
            prev_col = None
            trend_col = None
            for ci, h in enumerate(table.rows[0].cells):
                hc = _clean_str(h.text)
                if "上一次检测结果" in hc or "上期" in hc:
                    prev_col = ci
                elif "发展状况" in hc or "发展趋势" in hc:
                    trend_col = ci

            for row in table.rows[1:]:
                cells = [c.text.strip() for c in row.cells]
                row_full = " ".join(cells)
                # 寻找上一次评分
                m_prev_s = re.search(r"(?:上期|上次|上一次|前次|上一年度).{0,15}?(?:评分|BCI|指数).{0,10}?" + FLOAT_RE, row_full)
                if m_prev_s and not facts["上一次总体评分"]:
                    facts["上一次总体评分"] = f"{float(m_prev_s.group(1)):.2f}"
                m_prev_g = re.search(r"(?:上期|上次|上一次|前次|上一年度).{0,15}?(?:等级|状况).{0,10}?([ABCDE])级?", row_full)
                if m_prev_g and not facts["上一次总体等级"]:
                    facts["上一次总体等级"] = f"{m_prev_g.group(1).upper()}级"

                part_name = cells[0].strip() if cells else ""
                if prev_col is not None and len(cells) > prev_col:
                    p_txt = cells[prev_col].strip().rstrip("；;，,")
                    if p_txt and p_txt not in ("无", "/", "-"):
                        prev_results.append(f"{part_name}：{p_txt}" if part_name else p_txt)
                if trend_col is not None and len(cells) > trend_col:
                    t_txt = cells[trend_col].strip().rstrip("；;，,")
                    if t_txt and t_txt not in ("无", "/", "-"):
                        trend_parts.append(f"{part_name}：{t_txt}" if part_name else t_txt)

            if prev_results and not facts["上一次定检结果"]:
                facts["上一次定检结果"] = "；".join(prev_results) + "。"
            if trend_parts and "病害发展趋势与具体说明" not in facts:
                facts["病害发展趋势与具体说明"] = "；".join(trend_parts) + "。"

    # 3. 如果表格里没抓全总体评分或等级，扫描全文段落里的结论句
    if not facts["总体评分"] or not facts["总体等级"] or not facts["上一次总体评分"]:
        for p in doc.paragraphs:
            txt = p.text.strip()
            if not txt:
                continue

            # 总体评分与等级
            if not facts["总体评分"]:
                m = re.search(r"(?:全桥|总体|综合|主桥|跨线[^\s，。]+桥).{0,15}?(?:评分|BCI|Dr)\s*(?:为|=|：|:)?\s*" + FLOAT_RE, txt, re.I)
                if m:
                    facts["总体评分"] = f"{float(m.group(1)):.2f}"
                else:
                    m2 = re.search(r"(?:BCI|Dr)\s*=\s*" + FLOAT_RE, txt, re.I)
                    if m2:
                        facts["总体评分"] = f"{float(m2.group(1)):.2f}"
            if not facts["总体等级"]:
                m = re.search(r"(?:全桥|总体|综合|技术状况).{0,10}?等级.{0,6}?(?:为|:|=|评定为)?\s*([ABCDE])\s*级?", txt)
                if m:
                    facts["总体等级"] = f"{m.group(1).upper()}级"

            # 上一次定检评分与等级
            if not facts["上一次总体评分"]:
                m = re.search(r"(?:上一次|上一年度|上期|前次).{0,15}?(?:评分|BCI|指数)\s*(?:为|=|：|:)?\s*" + FLOAT_RE, txt, re.I)
                if m:
                    facts["上一次总体评分"] = f"{float(m.group(1)):.2f}"
                else:
                    # 例如 "按照旧规程(DB50/231-2006)的标准...BCI=84.30，整体技术状况等级评定为B级"
                    m2 = re.search(r"(?:DB50|旧标准|原标准|上一年度).{0,30}?BCI\s*=\s*" + FLOAT_RE, txt, re.I)
                    if m2:
                        facts["上一次总体评分"] = f"{float(m2.group(1)):.2f}"
            if not facts["上一次总体等级"]:
                m = re.search(r"(?:上一次|上一年度|上期|前次|原等级).{0,15}?等级.{0,6}?(?:为|:|=|为)?\s*([ABCDE])\s*级?", txt)
                if m:
                    facts["上一次总体等级"] = f"{m.group(1).upper()}级"
                elif facts["上一次总体评分"]:
                    m2 = re.search(r"BCI=" + re.escape(facts["上一次总体评分"]) + r".{0,25}?([ABCDE])\s*级", txt)
                    if m2:
                        facts["上一次总体等级"] = f"{m2.group(1).upper()}级"

            # 报告日期兜底
            if not facts["报告日期"] and ("年" in txt and "月" in txt and "日" in txt):
                d = _normalize_date(txt)
                if d:
                    facts["报告日期"] = d

    # 4. 根据评分推导缺失的等级
    def _grade_from_score(score_str: str) -> str:
        s = _safe_float(score_str)
        if s is None:
            return ""
        if s >= 90:
            return "A级"
        elif s >= 80:
            return "B级"
        elif s >= 66:
            return "C级"
        elif s >= 50:
            return "D级"
        else:
            return "E级"

    if facts["总体评分"] and not facts["总体等级"]:
        facts["总体等级"] = _grade_from_score(facts["总体评分"])
    if facts["上部结构评分"] and not facts["上部结构等级"]:
        facts["上部结构等级"] = _grade_from_score(facts["上部结构评分"])
    if facts["下部结构评分"] and not facts["下部结构等级"]:
        facts["下部结构等级"] = _grade_from_score(facts["下部结构评分"])
    if facts["桥面系评分"] and not facts["桥面系等级"]:
        facts["桥面系等级"] = _grade_from_score(facts["桥面系评分"])
    if facts["上一次总体评分"] and not facts["上一次总体等级"]:
        facts["上一次总体等级"] = _grade_from_score(facts["上一次总体评分"])

    return facts
