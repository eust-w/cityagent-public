"""Recover the three tables from a model response that run.py rejected (truncated or not strict JSON).

Used only by the fallback pass after a failed report. Each section is located by its key and parsed on
its own, so a wrong separator between sections, a cut-off tail or one broken row does not discard the
rest. Only text the model wrote is kept: complete row objects are taken verbatim, nothing is invented,
and every dropped or cut row is counted in the returned notes.
"""
from __future__ import annotations

import json
import re

from .inference_core import BASIC_FIELDS, _meaningful, _scalar, _without_trailing_commas

SECTIONS = (("维修建议", {"建议内容"}), ("病害明细", {"病害部位", "病害类型", "病害描述"}))
MAX_REPEATS = 2  # a row repeated more often back to back is a generation loop, not report content


def _value_end(text: str, start: int) -> int | None:
    """End (exclusive) of the object/array opening at text[start]; None when it never closes."""
    stack, quoted, escaped = [], False, False
    for index in range(start, len(text)):
        character = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            if not stack or (stack.pop() == "{") != (character == "}"):
                return None
            if not stack:
                return index + 1
    return None


def _load(fragment: str):
    try:
        return json.loads(_without_trailing_commas(fragment))
    except ValueError:
        return None


def _rows(text: str, start: int, content_fields: set) -> tuple[list[dict], dict]:
    """Complete, meaningful row objects of the array opening at text[start]."""
    rows, notes = [], {"dropped_rows": 0, "cut_off": False, "loop_rows": 0}
    index = start + 1
    while index < len(text):
        while index < len(text) and text[index] in " \t\r\n,":
            index += 1
        if index >= len(text):
            notes["cut_off"] = True
            break
        if text[index] == "]":
            break
        if text[index] != "{":
            notes["cut_off"] = True
            break
        end = _value_end(text, index)
        if end is None:
            notes["cut_off"] = True
            break
        row = _load(text[index:end])
        if (isinstance(row, dict) and all(_scalar(v) for v in row.values())
                and any(k in content_fields and _meaningful(v) for k, v in row.items())):
            if len(rows) >= MAX_REPEATS and all(previous == row for previous in rows[-MAX_REPEATS:]):
                notes["loop_rows"] += 1
            else:
                rows.append(row)
        else:
            notes["dropped_rows"] += 1
        index = end
    return rows, notes


def salvage_report(raw: str) -> tuple[dict, dict] | None:
    """Return (report, notes), or None when no meaningful basic information can be recovered."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    match = re.search(r'"基本信息"\s*:\s*\{', text)
    if not match:
        return None
    start = match.end() - 1
    end = _value_end(text, start)
    basic = _load(text[start:end]) if end else None
    if not isinstance(basic, dict) or any(not _scalar(v) for v in basic.values()):
        return None
    if not any(key in BASIC_FIELDS and _meaningful(value) for key, value in basic.items()):
        return None
    report, notes = {"基本信息": basic}, {}
    for section, fields in SECTIONS:
        found = next((m for m in re.finditer(rf'"{section}"\s*:\s*\[', text) if not start <= m.start() < end), None)
        if not found:
            report[section] = []
            notes[section] = {"missing": True}
            continue
        rows, section_notes = _rows(text, found.end() - 1, fields)
        report[section] = rows
        notes[section] = {"kept_rows": len(rows), **section_notes}
    return report, notes
