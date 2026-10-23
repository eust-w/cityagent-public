"""分治切片抽取调度器（Divide & Conquer Dispatcher）。

将整篇超长检测报告按部位与职责切分成 5 个子任务独立提示，
彻底杜绝单次 32k token 耗尽截断，并大幅提升提取专注度。
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

PROMPT_OVERVIEW = (
    "你是桥梁定期检测报告分析专家。阅读给定的桥梁工程概况、技术状况评定表及结论，提取基本信息，只输出严格JSON对象：\n"
    "{\"基本信息\": {\"桥梁名称\": \"...\", \"报告日期\": \"...\", \"总体评分\": \"...\", \"总体等级\": \"...\", "
    "\"上部结构评分\": \"...\", \"上部结构等级\": \"...\", \"下部结构评分\": \"...\", \"下部结构等级\": \"...\", "
    "\"桥面系评分\": \"...\", \"桥面系等级\": \"...\", \"上一次总体评分\": \"...\", \"上一次总体等级\": \"...\", "
    "\"上一次定检结果\": \"...\", \"病害发展趋势与具体说明\": \"...\", \"总体结论\": \"...\", \"主要风险点\": \"...\", \"建议统计\": \"...\"}}\n"
    "所有信息必须来自输入内容，不得编造；封闭字段按行业规范取值。"
)

PROMPT_ADVICE = (
    "你是桥梁养护维修专家。阅读给定的桥梁养护建议章节与表格，提取建议列表，只输出严格JSON对象：\n"
    "{\"维修建议\": [{\"序号\": \"1\", \"建议类别\": \"...\", \"建议内容\": \"...\", \"病害部位\": \"...\"}]}\n"
    "建议类别通常包括：尽快维修、立即处置、预防性养护、跟踪观测。内容必须忠于原文。"
)

def get_defect_prompt(section_name: str) -> str:
    return (
        f"你是桥梁病害检测记录分析专家。阅读给定的【{section_name}】外观检查记录与表格，逐条提取属于该部位的病害明细，只输出严格JSON对象：\n"
        '{"病害明细": [{"序号": "1", "病害部位": "...", "病害类型": "...", "病害描述": "...", '
        '"是否新增": "...", "上一次定检状态": "...", "发展程度": "..."}]}\n'
        "规范要求：保留原始几何尺寸(如L=..m, S=..㎡)；是否新增必须为'是'或'否'；无前次对比时上一次定检状态与发展程度填'无'。"
    )


PROMPT_DEFECT_TEMPLATE = (
    "你是桥梁病害检测记录分析专家。阅读给定的【{section_name}】外观检查记录与表格，逐条提取属于该部位的病害明细，只输出严格JSON对象：\n"
    '{{"病害明细": [{{"序号": "1", "病害部位": "...", "病害类型": "...", "病害描述": "...", '
    '"是否新增": "...", "上一次定检状态": "...", "发展程度": "..."}}]}}\n'
    "规范要求：保留原始几何尺寸(如L=..m, S=..㎡)；是否新增必须为'是'或'否'；无前次对比时上一次定检状态与发展程度填'无'。"
)


def build_chat_prompt(tokenizer: Any, system_prompt: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_subtask_json(text: str, key: str) -> Any:
    """宽容解析子任务输出的 JSON，支持局部截断抢救。"""
    text = text.strip()
    m = re.search(rf'"{key}"\s*:\s*(\{{|\[)', text)
    if not m:
        try:
            return json.loads(text).get(key)
        except Exception:
            return None

    start = m.start(1)
    is_list = (m.group(1) == "[")

    # 尝试直接截取配对
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c in "{[":
                depth += 1
            elif c in "]}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except Exception:
                        pass
    # 若被切断，对列表逐项抢救
    if is_list:
        items = []
        for match in re.finditer(r'\{[^{}]*?\}', text[start:]):
            try:
                items.append(json.loads(match.group(0)))
            except Exception:
                pass
        return items
    return None
