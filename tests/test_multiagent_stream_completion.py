"""A long ordinary answer must reach the SDK; real stream failures still stop."""
import time
import json

import pytest

from bridge_evidence.contest.document_model_gateway import AuditedModelGateway
from bridge_evidence.contest.document_multiagent_worker import make_client
from bridge_evidence.contest.reliable_generation import ReliableGenerationError
from test_reliable_document_gateway import Clock, JSONResponse, Opener, SSEResponse, event, packed


class ContextResponse(JSONResponse):
    def read(self):
        return packed({"count": 23, "max_model_len": 262144})


class ContextOpener(Opener):
    def open(self, request, *, timeout):
        if request.full_url.endswith("/tokenize"):
            self.requests.append((request.full_url, json.loads(request.data), timeout))
            return ContextResponse()
        return super().open(request, timeout=timeout)


def route(raw):
    deadline = time.monotonic() + 60
    client = make_client({"base_url": "http://127.0.0.1:9", "model": "own-model",
                          "thinking_tokens": 4096}, deadline)
    stream = SSEResponse(raw, Clock())
    client._opener = ContextOpener(stream)
    gateway = AuditedModelGateway(client, deadline=deadline, max_calls=3,
                                  max_output_tokens=262144)
    payload = {"model": "own-model", "messages": [{"role": "user", "content": "合成材料"}],
               "max_tokens": 262144 - 23, "temperature": 1.0,
               "chat_template_kwargs": {"enable_thinking": True}}
    return client, stream, gateway, payload


def test_real_multiagent_client_accepts_long_text_with_done_and_actual_usage():
    text = "\n".join(f"合成章节{i}：记录已读取的对象和仍待保存的条目，随后继续完成文件交付。" for i in range(180))
    assert len(text) > 4096
    raw = (event({"content": text}) + event(finish="stop")
           + event(usage={"prompt_tokens": 23, "completion_tokens": 5000, "total_tokens": 5023})
           + b"data: [DONE]\n\n")
    client, stream, gateway, payload = route([raw[i:i+1024] for i in range(0, len(raw), 1024)])
    result = gateway.forward(payload)
    assert result["choices"][0]["message"]["content"] == text
    assert client.calls[-1]["done_received"] is True
    assert client.calls[-1]["usage_known"] is True
    assert client.calls[-1]["content_characters"] == len(text)
    assert client.reliable_generation_policy.max_content_chars is None
    assert client.reliable_generation_policy.wall_seconds == 900
    assert client._opener.requests[-1][1]["max_tokens"] == 262144 - 23
    assert stream.closed


def test_no_character_cap_does_not_disable_reasoning_loop_protection():
    repeated = "This repeated reasoning never changes its current decision.\n" * 8
    client, stream, gateway, payload = route(event({"reasoning_content": repeated}))
    with pytest.raises(ReliableGenerationError) as failure:
        gateway.forward(payload)
    assert failure.value.code == "reasoning_repetition"
    assert failure.value.receipt["usage_known"] is False
    assert stream.closed
    assert sum(c.get("kind") == "agent_generation" for c in client.calls) == 1


def test_long_text_without_done_is_never_accepted_as_complete():
    text = "\n".join(f"第{i}项为合成检查记录，保留原始位置与对应检测年份。" for i in range(300))
    client, stream, gateway, payload = route(event({"content": text}))
    with pytest.raises(ReliableGenerationError) as failure:
        gateway.forward(payload)
    assert failure.value.code == "stream_protocol_error"
    assert client.calls[-1]["content_characters"] > 4096
    assert client.calls[-1]["done_received"] is False
    assert stream.closed
