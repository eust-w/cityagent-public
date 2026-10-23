from __future__ import annotations

import re


def development_level(text: str) -> str:
    """Map comparison prose to the competition's three-valued field."""
    value = str(text or "")
    if re.search(r"严重(?:发展|加剧|恶化)|显著(?:发展|加剧|恶化)|大幅(?:发展|加剧|恶化)", value):
        return "严重"
    if re.search(r"轻微(?:发展|加剧|恶化)|有所(?:发展|加重|加剧|扩大)|略有(?:发展|增加|扩大)|继续发展", value):
        return "轻微"
    return "无"


def previous_presence(text: str) -> str:
    """Reduce a previous-cycle narrative to the label field's enum."""
    value = str(text or "").strip()
    if not value or re.search(r"未发现|未见|无(?:明显)?病害|未提及|不存在", value):
        return "未发现"
    return "已发现"
