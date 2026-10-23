"""零依赖 .docx 文本提取（docx = zip + XML，不依赖 python-docx）。

赛机/官方评测机上 python-docx 不一定装，这里只用标准库。
训练时教材的 input 是整报告纯文本，这里保持一致：段落 + 表格行顺序输出。
"""
from __future__ import annotations

import re
import zipfile
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _text_of(elem) -> str:
    return "".join(t.text or "" for t in elem.iter(f"{W}t"))


def extract_docx_text(path: str) -> str:
    """按文档顺序输出段落和表格行文本。"""
    with zipfile.ZipFile(path) as z:
        with z.open("word/document.xml") as f:
            root = ET.parse(f).getroot()
    body = root.find(f"{W}body")
    lines = []
    for child in body:
        tag = child.tag
        if tag == f"{W}p":
            text = _text_of(child).strip()
            if text:
                lines.append(text)
        elif tag == f"{W}tbl":
            for row in child.iter(f"{W}tr"):
                cells = [_text_of(tc).strip() for tc in row.findall(f"{W}tc")]
                line = " ".join(c for c in cells if c)
                if line:
                    lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text)
