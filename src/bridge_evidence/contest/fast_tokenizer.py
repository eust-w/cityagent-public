"""CPU tokenizer from tokenizer.json + chat template, without importing transformers.

Importing transformers costs ~35 s per process on the contest PPU. This mirrors the
HF fast-tokenizer surface the pipeline uses (call, encode, decode, eos_token_id,
apply_chat_template) with HF's own Jinja environment; unsupported options raise so
``load_tokenizer`` can fall back to AutoTokenizer instead of silently diverging.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any


def _compile_template(source: str):
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env.from_string(source)


def _token_text(value):
    if isinstance(value, dict):
        value = value.get("content")
    return value if isinstance(value, str) and value else None


class FastTokenizer:
    def __init__(self, path: str | Path):
        import tokenizers

        path = Path(path)
        self._tok = tokenizers.Tokenizer.from_file(str(path / "tokenizer.json"))
        self._tok.no_truncation()
        self._tok.no_padding()
        config_file = path / "tokenizer_config.json"
        config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.is_file() else {}
        if config.get("clean_up_tokenization_spaces"):
            raise ValueError("clean_up_tokenization_spaces=True is not mirrored")
        template_file = path / "chat_template.jinja"
        source = (template_file.read_text(encoding="utf-8") if template_file.is_file()
                  else config.get("chat_template"))
        if not isinstance(source, str) or not source:
            raise ValueError("chat template missing")
        self._template = _compile_template(source)
        self.special_tokens_map = {
            key: text for key in ("bos_token", "eos_token", "unk_token", "pad_token")
            if (text := _token_text(config.get(key)))
        }
        self.eos_token = self.special_tokens_map.get("eos_token")
        self.pad_token = self.special_tokens_map.get("pad_token")
        self.eos_token_id = self._tok.token_to_id(self.eos_token) if self.eos_token else None
        self.pad_token_id = self._tok.token_to_id(self.pad_token) if self.pad_token else None

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs: Any) -> list[int]:
        if kwargs.get("truncation") or kwargs.get("max_length") or kwargs.get("return_tensors"):
            raise NotImplementedError("truncation/max_length/return_tensors are not mirrored")
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs: Any) -> dict:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens, **kwargs)}

    def decode(self, ids, skip_special_tokens: bool = False, **_kwargs: Any) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, conversation, tools=None, documents=None,
                            add_generation_prompt=False, tokenize=True, **kwargs: Any):
        for unsupported in ("chat_template", "continue_final_message", "return_dict",
                            "return_assistant_tokens_mask", "return_tensors"):
            if kwargs.pop(unsupported, None):
                raise NotImplementedError(f"{unsupported} is not mirrored")
        for ignored in ("padding", "truncation", "max_length", "tokenizer_kwargs"):
            kwargs.pop(ignored, None)
        if tools is not None and not all(isinstance(tool, dict) for tool in tools):
            raise NotImplementedError("only JSON-schema tools are mirrored")
        prompt = self._template.render(
            messages=conversation, tools=tools, documents=documents,
            add_generation_prompt=add_generation_prompt,
            **{**self.special_tokens_map, **kwargs},
        )
        return self.encode(prompt, add_special_tokens=False) if tokenize else prompt


def load_tokenizer(path: str | Path, *, allow_slow_fallback: bool = True):
    try:
        return FastTokenizer(path)
    except Exception as error:
        if not allow_slow_fallback:
            raise
        warnings.warn(f"FastTokenizer unavailable ({error}); using AutoTokenizer", stacklevel=2)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)


_PARITY_TEXTS = [
    "RSI恢复验证桥，2018年定期检测。",
    "总体等级：B；上部结构评分：88.15；下部结构评分：90.20。",
    "第1跨 K0+123 处主梁出现一条 0.15mm 裂缝，建议封闭处理。",
    "  leading spaces\n\n\tTabs and\r\nCRLF  ",
    "特殊符号 <|im_end|> 与 <|endoftext|> 出现在正文中",
    "全角，。！？（）【】１２３ＡＢＣ　表情😀",
]
_PARITY_TOOL = {"type": "function", "function": {
    "name": "read_evidence", "description": "读取证据",
    "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}
_PARITY_CHATS = [
    ([{"role": "system", "content": "你是桥梁定期检测报告分析专家。"},
      {"role": "user", "content": "桥梁名称：测试桥，总体等级：B"}], None),
    ([{"role": "user", "content": "列出病害"},
      {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
          "name": "read_evidence", "arguments": {"key": "p0:0:20"}}}]},
      {"role": "tool", "content": "第一跨主梁出现一条裂缝"},
      {"role": "assistant", "content": "{\"病害\": 1}"},
      {"role": "user", "content": "继续"}], [_PARITY_TOOL]),
]


def verify_parity(path: str | Path) -> dict:
    """Compare against AutoTokenizer on prompts, tools, ids, decode and EOS; for the target host."""
    from transformers import AutoTokenizer

    fast = FastTokenizer(path)
    hf = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
    mismatches = []

    def check(name, left, right):
        if left != right:
            mismatches.append({"check": name, "fast": repr(left)[:200], "hf": repr(right)[:200]})

    check("eos_token_id", fast.eos_token_id, hf.eos_token_id)
    for index, (messages, tools) in enumerate(_PARITY_CHATS):
        for generation in (False, True):
            for thinking in (False, True):
                options = dict(tools=tools, add_generation_prompt=generation, enable_thinking=thinking)
                left = fast.apply_chat_template(messages, tokenize=False, **options)
                right = hf.apply_chat_template(messages, tokenize=False, **options)
                check(f"chat[{index}] generation={generation} thinking={thinking}", left, right)
                check(f"chat-ids[{index}] generation={generation} thinking={thinking}",
                      fast(left, add_special_tokens=False)["input_ids"],
                      hf(right, add_special_tokens=False, truncation=False)["input_ids"])
    for index, text in enumerate(_PARITY_TEXTS):
        for special in (False, True):
            ids = fast(text, add_special_tokens=special)["input_ids"]
            check(f"ids[{index}] special={special}", ids,
                  hf(text, add_special_tokens=special, truncation=False)["input_ids"])
            check(f"decode[{index}]", fast.decode(ids), hf.decode(ids))
            check(f"decode-prefix[{index}]", fast.decode(ids[:3]), hf.decode(ids[:3]))
    tail = [fast.eos_token_id, fast.pad_token_id]
    check("decode-special", fast.decode(tail), hf.decode(tail))
    check("decode-skip-special", fast.decode(tail, skip_special_tokens=True),
          hf.decode(tail, skip_special_tokens=True))
    return {"passed": not mismatches, "mismatches": mismatches,
            "transformers": __import__("transformers").__version__}


if __name__ == "__main__":
    result = verify_parity(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
