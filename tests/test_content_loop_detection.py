"""Offline ordinary-content loop guards, including real-SSE replay support.

Run the optional archived-data check with CITYAGENT_REPLAY_SSE_DIR pointing to
the frozen v9 readback. No inference, network or changes to archived evidence.
"""
import hashlib
import os
from pathlib import Path

import pytest

from bridge_evidence.contest.reliable_generation import (
    StreamPolicy, ReliableGenerationError, _Aggregate, _ContentLoop, stream_chat,
)
from test_reliable_generation import Client, Clock, MODEL, Opener, Response, complete, event


def aggregate_fragments(text, fragment_size):
    aggregate = _Aggregate(StreamPolicy())
    for start in range(0, len(text), fragment_size):
        # SSE-level fragments and arbitrary transport byte boundaries both vary.
        raw = event({"content": text[start:start + fragment_size]})
        for offset in range(0, len(raw), 17):
            aggregate.feed(raw[offset:offset + 17])
    return aggregate


@pytest.mark.parametrize("size", [1, 7, 127, 511, 512, 2000])
def test_single_letter_loop_crosses_sse_and_utf8_transport_boundaries(size):
    prefix = "已保存桥梁基本信息。"
    with pytest.raises(_ContentLoop) as caught:
        aggregate_fragments(prefix + "H" * 800 + "不会读到这里", size)
    detail = caught.value.details
    assert detail["reason"] == "single_character_run"
    assert detail["content_start_char"] == len(prefix)
    assert detail["content_end_char"] == len(prefix) + 512
    assert detail["repeated_characters"] == 512
    assert detail["repeated_unit_sha256"] == hashlib.sha256(b"H").hexdigest()


@pytest.mark.parametrize("unit", ["Ha", "ABH", "哈好", "H ", "abc012", "abcdefghijklmnopqrstuvwxzyABCDEF"])
def test_short_exact_loop_needs_no_four_distinct_character_rule(unit):
    with pytest.raises(_ContentLoop) as caught:
        aggregate_fragments(unit * 1100, 19)
    detail = caught.value.details
    assert detail["reason"] == "short_cycle"
    assert detail["repeated_characters"] >= 1024
    assert detail["repeat_count"] >= 32
    assert detail["period_chars"] == len(unit)
    assert detail["content_start_char"] == 0
    assert detail["content_end_char"] == max(1024, len(unit) * 32)


@pytest.mark.parametrize("text", [
    "H" * 511, "AB" * 511,
    "H" * 511 + "间断" + "H" * 511,
    "AB" * 400 + "停止" + "AB" * 400,
    "0" * 6000, "0123456789" * 700, "0.00 1.00 " * 700,
    "。" * 6000, "-_= .,:;()" * 700, " \n\t" * 3000,
    "| 构件 | 病害 | 数量 |\n| 支座 | 锈蚀 | 1 |\n" * 300,
    "构件\t病害\t数量\n支座\t锈蚀\t1\n" * 300,
    "构件 病害 数量\n" * 300,
    "\n".join(f"{i}号支座，锈蚀，数量1，评分100.00。" for i in range(1500)),
])
def test_normal_long_tables_headers_numbers_punctuation_and_threshold_boundaries(text):
    aggregate = aggregate_fragments(text, 151)
    assert aggregate.content_characters == len(text)
    assert "".join(aggregate.content) == text
    assert aggregate.policy.max_content_chars is None


def test_transport_receipt_precedes_cancel_and_does_not_claim_server_stopped():
    clock = Clock()
    response = Response([event({"content": "H" * 300}), event({"content": "H" * 300}), complete()], clock,
                        advances=[2, 3], marker_status="cancellation_unknown")
    original_close = response.close
    def close():
        if not response.closed:
            clock.advance(0.25)
        original_close()
    response.close = close
    client = Client(Opener(response, clock))
    callbacks = []
    def cancel(receipt, exact_response):
        assert exact_response is response and response.closed
        callbacks.append(receipt)
        assert receipt["response_id"] == "chatcmpl-test"
        assert receipt["repetition_detection"]["content_end_char"] == 512
        assert receipt["repetition_detection"]["detected_after_seconds"] == 5
        assert receipt["connection_close_seconds"] == 0.25
        assert receipt["content_characters"] == 600  # full consumed SSE delta
        clock.advance(1.5)
        return {"status": "not_confirmed", "stop_observation": {"status": "unknown", "response_id": "chatcmpl-test"}}
    with pytest.raises(ReliableGenerationError) as caught:
        stream_chat(client, {"model": MODEL, "messages": [], "max_tokens": 2000}, deadline=1000,
                    policy=StreamPolicy(clock=clock), on_cancel=cancel)
    receipt = caught.value.receipt
    assert caught.value.code == "content_repetition"
    assert len(callbacks) == 1 and len(response.chunks) == 1
    assert receipt["seconds"] == 6.75
    assert receipt["cancellation_observation_seconds"] == 1.5
    assert receipt["repetition_detection"]["event_index"] == 2
    assert receipt["cancel_status"] != "cancellation_confirmed"
    assert receipt["stop_observation"] == {"status": "unknown", "response_id": "chatcmpl-test"}
    assert receipt["usage_known"] is False
    assert receipt["content_repetition_policy"]["single_char_repeat"] == 512


def test_guard_does_not_concatenate_distinct_requests_or_mutate_tool_arguments():
    for _ in range(2):
        aggregate = _Aggregate(StreamPolicy())
        aggregate.feed(complete({"content": "H" * 300, "tool_calls": [{"index": 0, "id": "a",
            "function": {"name": "save_entries", "arguments": "H" * 6000}}]}, finish="tool_calls"))
        result = aggregate.result(MODEL)
        assert result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == "H" * 6000


@pytest.mark.parametrize("field,value", [
    ("content_single_char_repeat", 511), ("content_cycle_min_chars", 1023),
    ("content_cycle_repeat_count", 31), ("content_cycle_max_period", 65),
    ("content_single_char_repeat", True),
])
def test_content_guard_thresholds_cannot_be_accidentally_aggressive(field, value):
    with pytest.raises(ValueError):
        StreamPolicy(**{field: value})


def test_archived_v9_sse_contains_one_detectable_H_loop_and_normal_streams_are_untouched():
    directory = os.environ.get("CITYAGENT_REPLAY_SSE_DIR")
    if not directory:
        pytest.skip("Optional read-only replay requires CITYAGENT_REPLAY_SSE_DIR")
    paths = sorted(Path(directory).rglob("*.sse"))
    assert len(paths) == 108
    failures, completed = [], []
    for path in paths:
        aggregate = _Aggregate(StreamPolicy())
        try:
            with path.open("rb") as handle:
                while raw := handle.read(8192):
                    aggregate.feed(raw)
            aggregate.feed(b"", final=True)
            aggregate.result(MODEL)
            completed.append(path)
        except _ContentLoop as failure:
            failures.append((path, failure.details))
    assert len(completed) == 107
    assert len(failures) == 1
    path, detail = failures[0]
    assert path.name == "e33a089631284b4ea30ffe4b3319de63.sse"
    assert detail["reason"] == "single_character_run"
    assert detail["repeated_unit_sha256"] == hashlib.sha256(b"H").hexdigest()
    assert detail["repeated_characters"] == 512
