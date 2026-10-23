"""Read the exported DOCX back against the reviewed JSON, without inference.

This checks the current three-table delivery contract, including repeated
conclusions and content-bearing layout text. It is a content check, not a Word
rendering or professional-semantic review. XML line-break representation is
normalized and table display numbers from the explicit document-agent writer
are generated in final row order. Legacy whole/chunked exports preserve their
saved sequence values and use their existing reconstruction audit. All
other values, whitespace, units and row order stay significant. No report
content is included in the returned audit summary.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import math
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from .docx_writer import ADVICE_FIELDS, BASIC_FIELDS, DEFECT_FIELDS, numbered_delivery_rows
from .run_state import json_digest


VALIDATOR_VERSION = "document_delivery_validation_v2_program_row_numbers"
_SECTIONS = {"基本信息", "维修建议", "病害明细"}
_UNKNOWN_HISTORY = {"未知", "未提供", "无法判断", "无法确定", "不详", "unknown",
                    "不确定", "未说明", "未明确", "未提及", "无法比较"}


def _display(value) -> str:
    # python-docx converts each CR and LF to a separate Word line break. Thus a
    # CRLF pair is two breaks with the current writer, not one normalized LF.
    return "" if value is None or value == "" else str(value).replace("\r", "\n")


def _validate_values(row: dict, fields, location: str) -> None:
    if not isinstance(row, dict):
        raise ValueError(f"Delivery {location} must be an object")
    if set(row) - set(fields):
        raise ValueError(f"Delivery {location} has unsupported fields")
    for field, value in row.items():
        if not (value is None or isinstance(value, str) or type(value) is int
                or (type(value) is float and math.isfinite(value))):
            raise ValueError(f"Delivery {location}.{field} is not an exportable scalar")
        # Reject characters XML 1.0 cannot represent before invoking the writer.
        if isinstance(value, str) and any(
            not (c in "\t\n\r" or "\u0020" <= c <= "\ud7ff"
                 or "\ue000" <= c <= "\ufffd" or "\U00010000" <= c <= "\U0010ffff")
            for c in value
        ):
            raise ValueError(f"Delivery {location}.{field} contains an unexportable XML character")


def validate_delivery_data(data: dict) -> None:
    """Reject anything the current writer would silently omit/stringify.

    Missing standard fields, nulls and empty strings are allowed. This checks
    representability only; semantic completeness belongs to the model review.
    """
    if not isinstance(data, dict) or set(data) != _SECTIONS:
        raise ValueError("Delivery requires exactly the three standard sections")
    _validate_values(data["基本信息"], (field for field, _ in BASIC_FIELDS), "基本信息")
    for section, fields in (("维修建议", ADVICE_FIELDS), ("病害明细", DEFECT_FIELDS)):
        if not isinstance(data[section], list):
            raise ValueError(f"Delivery {section} must be an array")
        for index, row in enumerate(data[section]):
            _validate_values(row, fields, f"{section}[{index}]")


def _expected(data: dict, processing_note: str | None = None, *, submission_format: bool = False) -> list[tuple[str, object]]:
    basic = data["基本信息"]
    history = [_display(basic.get(field)).strip().casefold()
               for field in ("上一次总体评分", "上一次总体等级", "上一次定检结果")]
    if any(value not in {"", "无", *_UNKNOWN_HISTORY} for value in history):
        subtitle = "（有对比年度）"
    elif "无" in history and not any(value in _UNKNOWN_HISTORY for value in history):
        subtitle = "（无对比年度）"
    else:
        subtitle = "（对比年度信息待确认）"
    basic_rows = [["字段", "内容", "说明"]]
    basic_rows += [[field, _display(basic.get(field)), gloss] for field, gloss in BASIC_FIELDS
                   if _display(basic.get(field))]
    blocks: list[tuple[str, object]] = [
        ("paragraph", (_display(basic.get("桥梁名称")) or "未知桥梁") + "·信息提取报告"),
        ("paragraph", subtitle),
        ("paragraph", "1、简要信息"),
        ("paragraph", "从报告中提取桥梁定检的概要结论与关键指标，输出内容包括："),
        ("table", basic_rows),
        ("paragraph", "2、详细信息"),
        ("paragraph", "（1）详细结论"),
    ]
    blocks += [("paragraph", _display(basic.get(field)))
               for field in ("总体结论", "病害发展趋势与具体说明", "主要风险点")
               if _display(basic.get(field))]
    for section, heading, fields in (("维修建议", "（2）建议明细", ADVICE_FIELDS),
                                     ("病害明细", "（3）病害列表", DEFECT_FIELDS)):
        blocks += [("paragraph", heading), ("table", [list(fields)] + [
            [_display(row.get(field)) for field in fields]
            for row in numbered_delivery_rows(data[section])
        ])]
    if submission_format:
        if processing_note is not None:
            raise ValueError("Submission format cannot contain processing notes")
        # Remove only the two program-generated introductory blocks, not facts.
        blocks = [block for index, block in enumerate(blocks) if index not in (1, 3)]
    if processing_note is not None:
        if not isinstance(processing_note, str) or not processing_note.strip():
            raise ValueError("Processing note must be a nonempty string")
        _validate_values({"note": processing_note}, {"note"}, "processing note")
        blocks += [("paragraph", "处理范围说明"), ("paragraph", _display(processing_note))]
    return blocks


def _validate_table(table: Table, expected: list[list[str]], number: int) -> int:
    if len(table.rows) != len(expected):
        raise ValueError(f"DOCX table {number} row count differs from reviewed delivery")
    columns = len(expected[0])
    if len(table.columns) != columns:
        raise ValueError(f"DOCX table {number} column count differs from delivery schema")
    for row_index, (row, values) in enumerate(zip(table.rows, expected)):
        if len(row.cells) != columns or len(row._tr.tc_lst) != columns:
            raise ValueError(f"DOCX table {number} row {row_index} has missing/merged cells")
        for column_index, (cell, value) in enumerate(zip(row.cells, values)):
            if cell._tc.xpath("./w:tcPr/w:vMerge | ./w:tcPr/w:gridSpan") or cell.tables:
                raise ValueError(f"DOCX table {number} row {row_index} has merged/nested cells")
            if cell.text != value:
                raise ValueError(f"DOCX table {number} row {row_index} column {column_index} "
                                 "differs from reviewed delivery")
    return len(expected) * columns


def validate_delivery_docx(data: dict, path: str | Path, *, processing_note: str | None = None,
                           submission_format: bool = False) -> dict:
    """Read the exact hashed DOCX bytes and compare every expected displayed value.

    Raises ValueError on schema, file or content mismatch. Readback includes
    column labels/glosses, every duplicate row, titles, history subtitle,
    repeated narrative paragraphs, and table placement. It never edits files.
    """
    validate_delivery_data(data)
    try:
        content = Path(path).read_bytes()
        document = Document(BytesIO(content))
    except Exception as error:
        raise ValueError("Cannot read exported DOCX for delivery validation") from error
    actual = []
    for element in document.element.body:
        if element.tag == qn("w:p"):
            actual.append(("paragraph", Paragraph(element, document)))
        elif element.tag == qn("w:tbl"):
            actual.append(("table", Table(element, document)))
        elif element.tag != qn("w:sectPr"):
            raise ValueError("DOCX contains an unsupported body block")
    expected = _expected(data, processing_note, submission_format=submission_format)
    if len(actual) != len(expected):
        raise ValueError("DOCX paragraph/table block count differs from reviewed delivery")
    table_rows, cells_checked, paragraphs_checked = [], 0, 0
    for index, ((kind, block), (expected_kind, value)) in enumerate(zip(actual, expected)):
        if kind != expected_kind:
            raise ValueError(f"DOCX block {index} placement differs from delivery layout")
        if kind == "paragraph":
            if block.text != value:
                raise ValueError(f"DOCX paragraph at block {index} differs from reviewed delivery")
            paragraphs_checked += 1
        else:
            cells_checked += _validate_table(block, value, len(table_rows))
            table_rows.append(len(value) - 1)
    result = {"schema_version": 1, "validator": VALIDATOR_VERSION,
            "report_sha256": json_digest(data),
            "docx_sha256": hashlib.sha256(content).hexdigest(),
            "table_data_rows": table_rows, "cells_checked": cells_checked,
            "paragraphs_checked": paragraphs_checked}
    if processing_note is not None:
        result["processing_note_sha256"] = json_digest(processing_note)
    return result
