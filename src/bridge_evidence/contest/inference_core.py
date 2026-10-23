"""Model-only inference boundary: complete prompts, bounded retries, auditable calls.

No backend dependencies or rule-based fallback live here. A caller supplies
``generate(input_ids, max_new_tokens)`` and remains responsible for fact fusion.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

INSTRUCTION = (
    "你是桥梁定期检测报告分析专家。阅读检测报告全文，提取关键信息，输出三部分JSON："
    "基本信息（桥梁名称、报告日期、总体及各部位技术状况评分与等级、上一次定检结果、"
    "病害发展趋势与具体说明、总体结论、主要风险点、建议统计）、"
    "维修建议（逐条含序号、建议类别、建议内容、病害部位）、"
    "病害明细（逐条含序号、病害部位、病害类型、病害描述、是否新增、上一次定检状态、发展程度）。"
    "所有信息必须来自报告原文，不得编造；封闭字段按规定取值。"
)

# Optional whole-report shape clarification. Keep INSTRUCTION byte-for-byte
# unchanged for historical prompts, training alignment and baseline comparisons.
WHOLE_FLAT_INSTRUCTION = INSTRUCTION + (
    "\n只输出一个JSON对象，顶层键为“基本信息”“维修建议”“病害明细”，不要附加解释或Markdown。"
    "\n“基本信息”必须是扁平对象，以字段名直接为键，值只能是字符串、有限数字或null，"
    "不能包含嵌套对象、数组或布尔值。评分与等级分别用“总体评分”“总体等级”、"
    "“上部结构评分”“上部结构等级”等独立键，不要嵌套为“技术状况”或“评分与等级”对象。"
    "\n“维修建议”和“病害明细”必须是数组；每条记录必须是扁平对象，其字段值同样只能是"
    "字符串、有限数字或null。维修建议的字段为“序号”“建议类别”“建议内容”“病害部位”；"
    "病害明细的字段为“序号”“病害部位”“病害类型”“病害描述”“是否新增”“上一次定检状态”“发展程度”。"
    "未知字段填null，不推测、不补造事实。原文中有记录时逐条填写，确实没有记录时才用空数组[]。"
    "\n以下仅为结构示例，null不是报告事实，空数组不是省略原文记录的指令；"
    "必须根据报告真实内容填写，不得编造示例记录："
    '{"基本信息":{"桥梁名称":null,"报告日期":null,"总体评分":null,"总体等级":null,'
    '"上部结构评分":null,"上部结构等级":null,"下部结构评分":null,"下部结构等级":null,'
    '"桥面系评分":null,"桥面系等级":null,"上一次总体评分":null,"上一次总体等级":null,'
    '"上一次定检结果":null,"病害发展趋势与具体说明":null,"总体结论":null,"主要风险点":null,'
    '"建议统计":null},"维修建议":[],"病害明细":[]}'
)


def whole_instruction(prompt: str = "source-fragment-v1") -> str:
    """Resolve an opt-in whole prompt without changing legacy candidate behavior."""
    if prompt == "whole-flat-v1":
        return WHOLE_FLAT_INSTRUCTION
    if prompt in ("source-fragment-v1", "source-fragment-concise-v1"):
        return INSTRUCTION
    raise ValueError("unsupported whole prompt")


BASIC_FIELDS = frozenset({
    "桥梁名称", "报告日期", "总体评分", "总体等级", "上部结构评分", "上部结构等级",
    "下部结构评分", "下部结构等级", "桥面系评分", "桥面系等级", "上一次总体评分",
    "上一次总体等级", "上一次定检结果", "病害发展趋势与具体说明", "总体结论", "主要风险点", "建议统计",
})
_PLACEHOLDERS = frozenset({
    "", "无", "未知", "不详", "未提供", "暂无", "暂无数据", "无数据", "未提及", "不适用",
    "unknown", "none", "null", "n/a", "na", "-", "--", "...", "…", "待填写", "待补充",
})


class ContextBudgetError(ValueError):
    """The full prompt plus reserved output cannot fit; no backend was called."""

    def __init__(self, prompt_tokens: int, max_model_len: int, max_new_tokens: int):
        super().__init__(f"Full prompt ({prompt_tokens}) + output budget ({max_new_tokens}) "
                         f"exceeds model context ({max_model_len}); input was not truncated")
        self.prompt_tokens = prompt_tokens
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.model_calls: list[dict] = []
        self.seconds = 0.0


class InferenceFailure(RuntimeError):
    """All allowed backend attempts failed; records never imply rule-based success."""

    def __init__(self, model_calls: list[dict], *, raw_output: str, wall_seconds: float):
        super().__init__(f"No complete, meaningful model JSON after {len(model_calls)} backend calls")
        self.model_calls = model_calls
        self.raw_output = raw_output
        self.seconds = sum(record["seconds"] for record in model_calls)
        self.wall_seconds = wall_seconds
        last = model_calls[-1] if model_calls else {}
        self.finish_reason = last.get("finish_reason")
        self.prompt_tokens = last.get("prompt_tokens", 0)
        self.output_tokens = last.get("output_tokens")


def prompt_ids_sha256(ids: list[int]) -> str:
    """Hash the exact untruncated IDs, using canonical compact JSON (not decoded text)."""
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_prompt(tokenizer: Any, text: str, *, max_model_len: int,
                 max_new_tokens: int, instruction: str = INSTRUCTION) -> list[int]:
    for name, value in (("max_model_len", max_model_len), ("max_new_tokens", max_new_tokens)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Document text must be a nonempty string")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction must be a nonempty string")
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": instruction}, {"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Tokenizer must return a nonempty flat list of nonnegative integer token IDs")
    if len(ids) + max_new_tokens > max_model_len:
        raise ContextBudgetError(len(ids), max_model_len, max_new_tokens)
    return ids


class _InvalidOutput(ValueError):
    pass


def _without_trailing_commas(text: str) -> str:
    """Remove only syntax commas before ]/}; commas inside JSON strings stay intact."""
    out = []
    quoted = escaped = False
    for index, character in enumerate(text):
        if quoted:
            out.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == ",":
            following = index + 1
            while following < len(text) and text[following].isspace():
                following += 1
            if following < len(text) and text[following] in "]}":
                continue
        out.append(character)
    return "".join(out)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidOutput("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise _InvalidOutput("nonfinite_json_number")


def _scalar(value):
    return (value is None or isinstance(value, str) or type(value) is int
            or (type(value) is float and math.isfinite(value)))


def _meaningful(value):
    if isinstance(value, str):
        cleaned = value.strip().casefold()
        return cleaned not in _PLACEHOLDERS and any(character.isalnum() for character in cleaned)
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _parse_report(raw: str, *, require_basic: bool = True) -> dict:
    text = raw.strip()
    if not text:
        raise _InvalidOutput("empty_output")
    if text.startswith("```"):
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if not fenced:
            raise _InvalidOutput("invalid_json_fence")
        text = fenced.group(1)
    try:
        data = json.loads(_without_trailing_commas(text), object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except _InvalidOutput:
        raise
    except (ValueError, TypeError, RecursionError) as error:
        raise _InvalidOutput("invalid_json") from error
    if not isinstance(data, dict):
        raise _InvalidOutput("report_must_be_object")
    if not {"基本信息", "维修建议", "病害明细"}.issubset(data):
        raise _InvalidOutput("missing_report_sections")
    basic = data["基本信息"]
    if not isinstance(basic, dict) or any(not _scalar(value) for value in basic.values()):
        raise _InvalidOutput("invalid_basic_shape")
    has_basic = any(key in BASIC_FIELDS and _meaningful(value) for key, value in basic.items())
    if require_basic and not has_basic:
        raise _InvalidOutput("meaningless_basic_information")
    for section, content_fields in (
        ("维修建议", {"建议内容"}),
        ("病害明细", {"病害部位", "病害类型", "病害描述"}),
    ):
        rows = data[section]
        if not isinstance(rows, list):
            raise _InvalidOutput("invalid_section_shape:" + section)
        for row in rows:
            if not isinstance(row, dict) or any(not _scalar(value) for value in row.values()):
                raise _InvalidOutput("invalid_row_shape:" + section)
            if not any(key in content_fields and _meaningful(value) for key, value in row.items()):
                raise _InvalidOutput("meaningless_row:" + section)
    if not has_basic and not (data["维修建议"] or data["病害明细"]):
        raise _InvalidOutput("meaningless_report")
    return data


def infer_document(text: str, tokenizer: Any, generate: Callable[[list[int], int], dict], *,
                   max_model_len: int, max_new_tokens: int, max_attempts: int = 2,
                   instruction: str = INSTRUCTION) -> dict:
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    wall_start = time.perf_counter()
    ids = build_prompt(tokenizer, text, max_model_len=max_model_len,
                       max_new_tokens=max_new_tokens, instruction=instruction)
    fingerprint = prompt_ids_sha256(ids)
    calls = []
    raw_output = ""
    for attempt in range(1, max_attempts + 1):
        record = {
            "call_id": str(uuid4()), "attempt": attempt,
            "utc": datetime.now(timezone.utc).isoformat(), "prompt_sha256": fingerprint,
            "output_sha256": None, "prompt_tokens": len(ids), "output_tokens": None,
            "finish_reason": None, "seconds": 0.0, "latency_seconds": 0.0,
            "status": "failed", "errors": [],
        }
        calls.append(record)
        call_start = time.perf_counter()
        try:
            # Isolate the recorded prompt from an accidental backend list mutation.
            response = generate(ids.copy(), max_new_tokens)
        except Exception as error:
            record["call_id"] = getattr(error, "call_id", record["call_id"])
            record["errors"].append("backend_exception:" + type(error).__name__)
            continue
        finally:
            record["seconds"] = time.perf_counter() - call_start
            record["latency_seconds"] = record["seconds"]
        try:
            if not isinstance(response, dict):
                raise _InvalidOutput("backend_response_must_be_object")
            record["call_id"] = response.get("call_id", record["call_id"])
            finish = response.get("finish_reason")
            if isinstance(finish, str):
                record["finish_reason"] = finish
            count = response.get("output_tokens")
            if type(count) is int:
                record["output_tokens"] = count
            raw = response.get("text")
            if isinstance(raw, str):
                raw_output = raw
                record["output_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            else:
                raise _InvalidOutput("backend_text_must_be_string")
            if type(count) is not int or not 0 < count <= max_new_tokens:
                raise _InvalidOutput("invalid_output_token_count")
            if finish != "stop":
                raise _InvalidOutput("incomplete_generation:" + str(record["finish_reason"]))
            data = _parse_report(raw)
        except _InvalidOutput as error:
            record["errors"].append(str(error))
            continue
        except UnicodeError:
            # A malformed backend string still belongs to this recorded attempt.
            record["errors"].append("invalid_unicode_output")
            continue
        record["status"] = "success"
        return {"data": data, "model_calls": calls, "raw_output": raw_output,
                "seconds": sum(call["seconds"] for call in calls),
                "wall_seconds": time.perf_counter() - wall_start,
                "finish_reason": record["finish_reason"], "prompt_tokens": len(ids),
                "output_tokens": record["output_tokens"]}
    raise InferenceFailure(calls, raw_output=raw_output, wall_seconds=time.perf_counter() - wall_start)
