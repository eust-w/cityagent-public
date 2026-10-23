"""Synthetic SSE through the real gateway/client/transport; no model service."""
from copy import deepcopy
import hashlib
from http.client import HTTPConnection
import json
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from bridge_evidence.contest.document_model_gateway import AuditedModelGateway
from bridge_evidence.contest.local_model_client import LocalModelClient
from bridge_evidence.contest.reliable_generation import ReliableGenerationError, StreamPolicy


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def event(delta=None, finish=None, usage=None):
    value = {"id": "synthetic-response", "model": "own-model", "choices": [] if usage else [
        {"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        value["usage"] = usage
    return b"data: " + packed(value) + b"\n\n"


def complete(*, tools=False, finish="stop"):
    if tools:
        raw = event({"tool_calls": [{"index": 0, "id": "call-synthetic", "function": {
            "name": "update_", "arguments": '{"text":"'}}]})
        raw += event({"tool_calls": [{"index": 0, "function": {
            "name": "workspace", "arguments": "重复表头保持原文内容" * 100 + '"}'}}]})
        finish = "tool_calls"
    else:
        raw = event({"content": "合成结果"})
    return raw + event(finish=finish) + event(usage={
        "prompt_tokens": 23, "completion_tokens": 3, "total_tokens": 26}) + b"data: [DONE]\n\n"


class Clock:
    def __init__(self):
        self.now = 100.0
    def __call__(self):
        return self.now


class JSONResponse:
    status = 200
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        return False
    def read(self):
        return packed({"count": 23})


class SSEResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}
    def __init__(self, raw, clock, advances=()):
        self.clock = clock
        self.chunks = [raw] if isinstance(raw, bytes) else list(raw)
        self.advances = list(advances)
        self.closed = False
        self.timeouts = []
    def set_read_timeout(self, seconds):
        self.timeouts.append(seconds)
    def read1(self, _size):
        if self.advances:
            self.clock.now += self.advances.pop(0)
        value = self.chunks.pop(0) if self.chunks else b""
        if isinstance(value, BaseException):
            raise value
        return value
    def close(self):
        self.closed = True


class Opener:
    def __init__(self, stream):
        self.stream = stream
        self.requests = []
    def open(self, request, *, timeout):
        payload = json.loads(request.data)
        self.requests.append((request.full_url, payload, timeout))
        return JSONResponse() if request.full_url.endswith("/tokenize") else self.stream


class ReliableDocumentGatewayTests(unittest.TestCase):
    def setup_route(self, raw=None, *, timeout=900, duration=50, advances=(), cancel=None):
        self.clock = Clock()
        self.stream = SSEResponse(complete() if raw is None else raw, self.clock, advances)
        self.client = LocalModelClient("http://127.0.0.1:9", "own-model", timeout=timeout,
            max_model_len=512, max_output_tokens=16, enable_thinking=True, temperature=1)
        self.client._opener = Opener(self.stream)
        self.client.enable_reliable_generation(StreamPolicy(clock=self.clock), deadline=100 + duration,
                                               on_cancel=cancel)
        self.payload = {"model": "own-model", "stream": False, "max_tokens": 16,
            "temperature": 1, "chat_template_kwargs": {"enable_thinking": True, "custom_flag": "kept"},
            "messages": [{"role": "user", "content": "完整合成文档"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "previous", "type": "function",
                    "function": {"name": "read_source", "arguments": '{"start":0,"end":100}'}}]},
                {"role": "tool", "tool_call_id": "previous", "content": "完整合成证据"}],
            "tools": [{"type": "function", "function": {"name": "update_workspace", "parameters": {
                "type": "object", "properties": {"text": {"type": "string"}}}}}], "tool_choice": "auto"}
        self.responses = []
        self.gateway = AuditedModelGateway(self.client, deadline=100 + duration, max_calls=10,
            max_output_tokens=16, on_response=lambda result, receipt: self.responses.append(deepcopy(result)))

    def clocked(self):
        return patch("bridge_evidence.contest.local_model_client.time.monotonic", new=self.clock)

    def test_real_route_profiles_wire_and_preserves_complete_tool_inputs_and_output(self):
        self.setup_route(complete(tools=True))
        original = deepcopy(self.payload)
        with self.clocked():
            result = self.gateway.forward(self.payload)
        requests = self.client._opener.requests
        self.assertEqual(len(requests), 2)
        self.assertEqual([r["kind"] for r in self.client.calls], ["tokenize", "agent_generation"])
        wire = requests[-1][1]
        self.assertEqual(wire["messages"], original["messages"])
        self.assertEqual(wire["tools"], original["tools"])
        self.assertEqual(wire["tool_choice"], original["tool_choice"])
        self.assertEqual(self.payload, original)
        self.assertTrue(wire["stream"])
        self.assertEqual(wire["stream_options"], {"include_usage": True})
        self.assertEqual(wire["repetition_penalty"], 1.1)
        self.assertEqual(wire["presence_penalty"], 1.5)
        self.assertEqual(wire["chat_template_kwargs"], {"enable_thinking": True, "custom_flag": "kept"})
        self.assertEqual(requests[0][1]["messages"], wire["messages"])
        self.assertEqual(requests[0][1]["tools"], wire["tools"])
        self.assertEqual(requests[0][1]["chat_template_kwargs"], wire["chat_template_kwargs"])
        self.assertEqual(self.gateway.receipts[-1]["request_sha256"], hashlib.sha256(packed(wire)).hexdigest())
        self.assertEqual(self.client.calls[-1]["input_sha256"], self.gateway.receipts[-1]["request_sha256"])
        output = result["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(output["function"]["name"], "update_workspace")
        self.assertEqual(json.loads(output["function"]["arguments"])["text"], "重复表头保持原文内容" * 100)
        self.assertEqual(self.responses, [result])
        self.assertTrue(self.stream.closed)
        self.assertTrue(self.client.calls[-1]["done_received"])
        self.assertTrue(self.client.calls[-1]["usage_known"])
        self.assertNotIn("完整合成证据", str(self.client.calls))

    def test_reasoning_loop_fails_once_and_reports_unknown_cancel_without_retry(self):
        repeated = "This repeated reasoning never changes its current decision.\n" * 8
        self.setup_route(event({"reasoning_content": repeated}))
        with self.clocked(), self.assertRaises(ReliableGenerationError) as caught:
            self.gateway.forward(self.payload)
        receipt = caught.exception.receipt
        self.assertEqual(receipt["error_code"], "reasoning_repetition")
        self.assertIsNone(receipt["usage"])
        self.assertFalse(receipt["usage_known"])
        self.assertNotEqual(receipt["cancel_status"], "cancellation_confirmed")
        self.assertEqual(self.responses, [])
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(self.gateway.receipts[-1]["call_id"], receipt["call_id"])
        self.assertEqual(self.gateway.receipts[-1]["cancel_status"], receipt["cancel_status"])
        self.assertTrue(self.stream.closed)
        # Cross-request admission belongs to the report/batch coordinator.
        # This transport must preserve unknown termination and never retry it.
        from bridge_evidence.contest.generation_stop_observer import generation_stopped
        self.assertFalse(generation_stopped(receipt))
        self.assertFalse(receipt["automatic_retry"])
        self.assertEqual(len(self.client._opener.requests), 2)

    def test_even_unverified_callback_confirmed_does_not_enable_retry(self):
        self.setup_route([TimeoutError("synthetic idle")], cancel=lambda *_: {"status": "cancellation_confirmed"})
        with self.clocked(), self.assertRaises(ReliableGenerationError) as caught:
            self.gateway.forward(self.payload)
        self.assertEqual(caught.exception.receipt["cancel_status"], "unknown")
        self.assertEqual(len(self.client._opener.requests), 2)
        from bridge_evidence.contest.generation_stop_observer import generation_stopped
        self.assertFalse(generation_stopped(caught.exception.receipt))
        self.assertFalse(caught.exception.receipt["automatic_retry"])

    def test_wall_limit_counts_prefill_and_cannot_reset_with_stream_activity(self):
        raw = event({"reasoning_content": "nonrepeating reasoning"})
        self.setup_route([raw, raw], duration=3, advances=[2, 2])
        with self.clocked(), self.assertRaises(ReliableGenerationError) as caught:
            self.gateway.forward(self.payload)
        self.assertEqual(caught.exception.code, "wall_or_read_timeout")
        self.assertEqual(self.stream.timeouts, [3, 1])
        self.assertEqual(caught.exception.receipt["wall_seconds_limit"], 3)
        self.assertEqual(self.client.timeout, 900)

    def test_request_ceiling_is_900_even_when_client_and_report_are_longer(self):
        self.setup_route(timeout=2000, duration=2000)
        with self.clocked():
            self.gateway.forward(self.payload)
        self.assertEqual(self.client.calls[-1]["wall_seconds_limit"], 900)
        self.assertEqual(self.client._opener.requests[-1][2],
                         self.client.reliable_generation_policy.read_idle_seconds)
        self.assertEqual(self.client.timeout, 2000)

    def test_complete_length_response_is_not_delivered_to_sdk(self):
        self.setup_route(complete(finish="length"))
        with self.clocked(), self.assertRaisesRegex(ValueError, "incomplete or truncated"):
            self.gateway.forward(self.payload)
        self.assertEqual(self.responses, [])
        self.assertEqual(self.client.calls[-1]["finish_reason"], "length")
        self.assertTrue(self.client.calls[-1]["usage_known"])
        self.assertEqual(self.client.calls[-1]["status"], "failed")

    def test_actual_sdk_http_route_returns_complete_json_while_upstream_is_sse(self):
        self.setup_route(complete(tools=True))
        with self.clocked(), self.gateway:
            target = urlsplit(self.gateway.url)
            connection = HTTPConnection(target.hostname, target.port, timeout=2)
            try:
                connection.request("POST", "/v1/chat/completions", packed(self.payload), {"Content-Type": "application/json"})
                response = connection.getresponse()
                result = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Content-Type"], "application/json")
                self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
            finally:
                connection.close()
        self.assertTrue(self.client._opener.requests[-1][1]["stream"])
        with self.clocked(), self.assertRaisesRegex(RuntimeError, "Closed SDK gateway"):
            self.gateway.forward(self.payload)
        self.assertEqual(len(self.client._opener.requests), 2)

    def test_reliable_cpu_tokenizer_is_preserved_and_freshly_checked(self):
        self.setup_route()
        calls = []
        class Tokenizer:
            def apply_chat_template(_self, messages, **kwargs):
                calls.append((deepcopy(messages), deepcopy(kwargs)))
                return "synthetic complete template"
            def encode(_self, _prompt, **kwargs):
                return list(range(23))
        self.client.chat_tokenizer = Tokenizer()
        with self.clocked():
            self.gateway.forward(self.payload)
        self.assertEqual(calls[0][0], self.payload["messages"])
        self.assertTrue(calls[0][1]["enable_thinking"])
        self.assertEqual(calls[0][1]["custom_flag"], "kept")
        self.assertEqual(calls[0][1]["tools"], self.payload["tools"])
        self.assertTrue(self.client.calls[0]["token_counts_match"])
        self.assertEqual(self.client.calls[0]["local_token_count"], 23)
        self.assertEqual(self.client._opener.requests[0][1]["prompt"], "synthetic complete template")

    def test_zero_or_expired_global_deadline_cannot_be_treated_as_unbounded(self):
        self.setup_route()
        self.client.reliable_generation_deadline = 0
        with self.clocked(), self.assertRaises(ReliableGenerationError) as caught:
            self.client._request("/v1/chat/completions", self.payload, kind="generation")
        self.assertFalse(caught.exception.receipt["generation_attempted"])
        self.assertEqual(self.client._opener.requests, [])
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0]["status"], "failed")

    def test_explicit_profile_validation_and_reenable_preserve_unknown_receipts(self):
        default = LocalModelClient("http://127.0.0.1:9", "own-model")
        self.assertEqual(default.timeout, 600)
        self.assertIsNone(default.reliable_generation_policy)
        default.enable_reliable_generation(StreamPolicy())
        with self.assertRaisesRegex(ValueError, "temperature=1"):
            default.reliable_payload({"messages": []})
        no_thinking = LocalModelClient("http://127.0.0.1:9", "own-model", temperature=1,
                                      enable_thinking=False)
        no_thinking.enable_reliable_generation(StreamPolicy())
        self.assertEqual(no_thinking.reliable_payload({"messages": [],
            "chat_template_kwargs": {"enable_thinking": False}})["chat_template_kwargs"],
            {"enable_thinking": False})
        self.setup_route([TimeoutError("synthetic idle")])
        with self.clocked(), self.assertRaises(ReliableGenerationError):
            self.gateway.forward(self.payload)
        recorded = deepcopy(self.client.calls)
        self.client.enable_reliable_generation(StreamPolicy(clock=self.clock), deadline=150)
        self.assertEqual(self.client.calls, recorded)
        self.assertNotEqual(self.client.calls[-1]["cancel_status"], "cancellation_confirmed")
        fresh = LocalModelClient("http://127.0.0.1:9", "own-model", enable_thinking=True, temperature=1)
        fresh.enable_reliable_generation(StreamPolicy())
        fresh.temperature = 0
        with self.assertRaisesRegex(ValueError, "temperature=1"):
            fresh.reliable_payload(self.payload)


if __name__ == "__main__":
    unittest.main()
