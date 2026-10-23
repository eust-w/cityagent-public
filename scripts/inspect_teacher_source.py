from __future__ import annotations

import argparse
import re
from pathlib import Path

from bridge_evidence.parsers.docx import DocxStructureParser


SCORE_SIGNAL = re.compile(
    r"(?:工程名称|桥梁名称|报告日期|检测日期|检验日期|"
    r"BCI[mMsSxX]?|BSI[mMsSxX]?|\bDr\b|"
    r"总体.{0,8}(?:评分|等级)|上部结构.{0,12}(?:评分|等级)|"
    r"下部结构.{0,12}(?:评分|等级)|桥面系.{0,12}(?:评分|等级)|"
    r"技术状况评定表|评定为\s*[ABCDE一二三四五]\s*(?:级|类)|"
    r"上一次|上次|上一年度|原评定|原等级)",
    re.I,
)


def _compact(text: str, limit: int = 1800) -> str:
    rendered = " ".join(text.split())
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


def inspect(path: Path) -> None:
    parsed = DocxStructureParser().parse(path)
    print(f"## SOURCE {path}")
    print(f"paragraphs={len(parsed.paragraphs)} tables={len(parsed.tables)}")
    for paragraph in parsed.paragraphs:
        if paragraph.text and SCORE_SIGNAL.search(paragraph.text):
            print(f"P{paragraph.index}\t{_compact(paragraph.text)}")
    for table in parsed.tables:
        context = " / ".join(table.context[-4:])
        for row_index, row in enumerate(table.rows):
            text = " | ".join(row)
            if SCORE_SIGNAL.search(text) or (
                SCORE_SIGNAL.search(context)
                and re.search(r"(?:上部结构|下部结构|桥面系|总体|全桥)", text)
            ):
                print(f"T{table.index}R{row_index}\t{_compact(context, 480)} || {_compact(text)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print source-only evidence used for teacher adjudication")
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.inputs:
        inspect(path)


if __name__ == "__main__":
    main()
