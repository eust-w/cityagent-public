"""FastTokenizer must reproduce HF prompt bytes and token IDs for the pinned 9B tokenizer.

Reference values were produced by transformers 4.57.6 AutoTokenizer on the same
SHA-pinned assets; run ``python -m bridge_evidence.contest.fast_tokenizer DIR`` on the
target host to re-verify against its installed transformers.
"""

import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from bridge_evidence.contest.fast_tokenizer import FastTokenizer, load_tokenizer

PINNED = {
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
}
MESSAGES = [
    {"role": "system", "content": "你是桥梁定期检测报告分析专家。"},
    {"role": "user", "content": "RSI恢复验证桥，2018年定期检测。"},
]


def pinned_tokenizer():
    directory = os.environ.get("CITYAGENT_TEST_FIXED9B_TOKENIZER_DIR")
    if directory is None:
        pytest.skip("Set CITYAGENT_TEST_FIXED9B_TOKENIZER_DIR to pinned offline fixed9B assets")
    path = Path(directory)
    for name, digest in PINNED.items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest, name
    return FastTokenizer(path)


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_prompt_ids_decode_and_eos_match_hf_reference():
    tokenizer = pinned_tokenizer()
    prompt = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    assert sha256(prompt) == "c0c71b5bb85b5090b5f80038fa265fccb1d3ea32b7407fd9aa24bd3f648a30ec"
    assert len(ids) == 38
    assert sha256(json.dumps(ids)) == "d4054ce8066004fd46fc04578b3aa8b948129669bed4cb0cc5dc60c28fd0923b"
    assert tokenizer.eos_token_id == 248046
    assert tokenizer.decode([248046]) == "<|im_end|>"
    assert tokenizer.decode([248046], skip_special_tokens=True) == ""
    assert tokenizer.encode("第1跨 K0+123 处", add_special_tokens=False) == [
        96025, 16, 97852, 710, 15, 10, 16, 17, 18, 220, 96117]


def test_unmirrored_options_raise_instead_of_diverging():
    tokenizer = pinned_tokenizer()
    with pytest.raises(NotImplementedError):
        tokenizer("桥梁", truncation=True, max_length=1)
    with pytest.raises(NotImplementedError):
        tokenizer.apply_chat_template(MESSAGES, tokenize=False, continue_final_message=True)
    with pytest.raises(NotImplementedError):
        tokenizer.apply_chat_template(MESSAGES, tokenize=False, tools=[lambda key: key])


def test_missing_assets_fall_back_to_autotokenizer(tmp_path, monkeypatch):
    sentinel = object()
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: sentinel)))
    with pytest.warns(UserWarning, match="FastTokenizer unavailable"):
        assert load_tokenizer(tmp_path) is sentinel
    with pytest.raises(Exception):
        load_tokenizer(tmp_path, allow_slow_fallback=False)
