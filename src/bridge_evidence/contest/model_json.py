"""Decode one complete model JSON object without salvaging malformed output."""

from __future__ import annotations

import json
import math
import re


def _missing_closers(text: str) -> str | None:
    """Closing brackets that would complete *text*, or None if it is not a clean prefix."""
    stack: list[str] = []
    quoted = escaped = False
    for character in text:
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
            stack.append("}" if character == "{" else "]")
        elif character in "}]":
            if not stack or stack.pop() != character:
                return None
    if quoted or not stack:
        return None
    return "".join(reversed(stack))


def parse_json_object(text: str, *, close_unbalanced: bool = False, lenient: bool = False) -> dict:
    """Accept raw JSON or one complete optional-json Markdown fence only.

    Raw model bytes remain in their receipts. This recognizes a transport
    wrapper; it does not repair trailing commas, find objects inside prose,
    choose duplicate keys, or accept nonfinite numeric values.
    With ``close_unbalanced`` a stop-finished object missing only its final
    closing brackets is completed; no content is added.
    ``lenient`` (fragment extraction only) also accepts an unterminated fence,
    leading/trailing prose around the first object, duplicate keys (first wins)
    and nonfinite numbers (as null); evidence checks happen downstream.
    """
    if not isinstance(text, str):
        raise ValueError("model JSON must be text")  # noqa: TRY004 -- one invalid-output contract
    text = text.strip()
    if text.startswith("```"):
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text,
                              flags=re.DOTALL | re.IGNORECASE)
        if fenced is not None:
            text = fenced.group(1)
        elif lenient:
            text = text.split("\n", 1)[1] if "\n" in text else ""
            text = text.rstrip().rstrip("`").rstrip()
        else:
            raise ValueError("invalid complete JSON fence")
    if lenient and not text.startswith("{") and "{" in text:
        text = text[text.index("{"):]

    def unique_pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                if lenient:
                    continue
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            if lenient:
                return None
            raise ValueError("nonfinite JSON number")
        return number

    def reject_constant(_value):
        if lenient:
            return None
        raise ValueError("nonfinite JSON number")

    def decode(candidate):
        return json.loads(candidate, object_pairs_hook=unique_pairs, parse_float=finite_float,
                          parse_constant=reject_constant)

    try:
        value = decode(text)
    except json.JSONDecodeError:
        value = None
        if lenient:
            try:
                value, _ = json.JSONDecoder(
                    object_pairs_hook=unique_pairs, parse_float=finite_float,
                    parse_constant=reject_constant,
                ).raw_decode(text)
            except json.JSONDecodeError:
                value = None
        if value is None:
            closers = _missing_closers(text) if close_unbalanced else None
            if closers is None:
                raise
            value = decode(text + closers)
    if not isinstance(value, dict):
        raise ValueError("model JSON must be an object")  # noqa: TRY004 -- one invalid-output contract
    return value
