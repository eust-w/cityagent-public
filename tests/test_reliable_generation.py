"""Offline transport boundaries; no SDK, socket, subprocess or model call."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

PATH = Path(__file__).resolve().parents[1] / "src/bridge_evidence/contest/reliable_generation.py"
SPEC = importlib.util.spec_from_file_location("reliable_generation_under_test", PATH)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)
MODEL = "cityagent-qwen36-27b-epoch-002"
USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}


def event(delta=None, finish=None, *, choices=True, usage=None, model=MODEL):
    obj = {"id": "chatcmpl-test", "model": model, "created": 123,
           "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}] if choices else []}
    if usage is not None:
        obj["usage"] = usage
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\r\n\r\n").encode()


def complete(delta=None, finish="stop"):
    return event(delta or {"content": "完成"}) + event(finish=finish) + event(choices=False, usage=USAGE) + b"data: [DONE]\r\n\r\n"


class Clock:
    def __init__(self): self.value = 100.0
    def __call__(self): return self.value
    def advance(self, value): self.value += value


class Response:
    status = 200
    headers = {"Content-Type": "text/event-stream; charset=utf-8"}
    wire_attempt_id = "wire-001"
    def __init__(self, chunks, clock, advances=None, marker_status="cancellation_confirmed"):
        self.chunks = list(chunks)
        self.clock = clock
        self.advances = list(advances or [])
        self.closed = False
        self.timeouts = []
        self.completed = []
        self.cancelled = []
        self.marker_status = marker_status
    def geturl(self): return "http://127.0.0.1:18110/v1/chat/completions"
    def set_read_timeout(self, seconds): self.timeouts.append(seconds)
    def read1(self, size):
        if self.advances: self.clock.advance(self.advances.pop(0))
        item = self.chunks.pop(0) if self.chunks else b""
        if isinstance(item, BaseException): raise item
        return item
    def close(self): self.closed = True
    def mark_stream_completed(self, value, receipt):
        self.completed.append((value, dict(receipt)))
        return {"attempt_id": self.wire_attempt_id,
                "aggregate_response_path": "/controlled/fake-aggregate.json",
                "aggregated_response_canonical_sha256": receipt["output_sha256"]}
    def mark_cancelled(self, settlement):
        if not self.closed: raise AssertionError("marker must follow close")
        self.cancelled.append(settlement)
        return {"status": self.marker_status, "cancellation_receipt": {"path": "/controlled/fake.json", "sha256": "a" * 64}}


class Opener:
    def __init__(self, response, clock, advance=0):
        self.response = response; self.clock = clock; self.advance = advance; self.calls = []
    def open(self, request, *, timeout):
        self.calls.append((request, timeout))
        self.clock.advance(self.advance)
        if isinstance(self.response, BaseException): raise self.response
        return self.response


class Client:
    model = MODEL
    api_key = "sdk-placeholder"
    base_url = "http://127.0.0.1:18110"
    def __init__(self, opener): self._opener = opener; self.calls = []


class ReliableGenerationTests(unittest.TestCase):
    def test_failed_disconnect_signals_eof_with_another_socket_reference(self):
        from types import SimpleNamespace
        left, right = socket.socketpair()
        extra_reference = left.makefile('rb')
        response = SimpleNamespace(fp=SimpleNamespace(raw=SimpleNamespace(_sock=left)),
                                   close=left.close)
        try:
            m._disconnect_failed_response(response)
            right.settimeout(.5)
            self.assertEqual(right.recv(1), b'')
        finally:
            extra_reference.close()
            left.close()
            right.close()

    def setUp(self):
        self.clock = Clock()
        self.policy = m.StreamPolicy(clock=self.clock)
        self.payload = {"model": MODEL, "messages": [{"role": "user", "content": "只读原文"}], "max_tokens": 262000,
                        "temperature": 1, "repetition_penalty": 1.1, "presence_penalty": 1.5,
                        "chat_template_kwargs": {"enable_thinking": True}}
        self.guard = patch.object(socket, "create_connection", side_effect=AssertionError("offline network forbidden"))
        self.guard.start(); self.addCleanup(self.guard.stop)

    def invoke(self, chunks, *, advances=None, deadline=700, policy=None, on_cancel=None, response=None, open_advance=0, on_chunk=None):
        response = response or Response(chunks, self.clock, advances)
        client = Client(Opener(response, self.clock, open_advance))
        self.last_client = client; self.last_response = response
        return m.stream_chat(client, self.payload, deadline=deadline, policy=policy or self.policy, on_cancel=on_cancel, on_chunk=on_chunk)

    def failure(self, chunks, **kw):
        with self.assertRaises(m.ReliableGenerationError) as caught:
            self.invoke(chunks, **kw)
        exc = caught.exception
        self.assertTrue(exc.unknown_usage)
        self.assertIsNone(exc.receipt["usage"])
        self.assertEqual(len(self.last_client._opener.calls), 1)
        self.assertEqual(self.last_client.calls, [])
        if isinstance(self.last_response, Response): self.assertTrue(self.last_response.closed)
        return exc

    def test_full_fragmented_utf8_reasoning_content_tools_and_usage(self):
        raw = (b": keepalive\r\n\r\n" + event({"role": "assistant", "reasoning_content": "推理跨字节"})
               + event({"content": "桥梁名称", "tool_calls": [
                   {"index": 1, "id": "call-b", "type": "function", "function": {"name": "read_", "arguments": "{"}},
                   {"index": 0, "id": "call-a", "function": {"name": "save", "arguments": "{\"桥\": "}}]})
               + event({"tool_calls": [
                   {"index": 0, "function": {"arguments": "BAD JSON}"}},
                   {"index": 1, "function": {"name": "source", "arguments": "}"}}]})
               + event(finish="tool_calls") + event(choices=False, usage=USAGE) + b"data: [DONE]\r\n\r\n")
        captured = []
        result, receipt = self.invoke([raw[i:i+3] for i in range(0, len(raw), 3)], on_chunk=captured.append)
        msg = result["choices"][0]["message"]
        self.assertEqual(msg["reasoning_content"], "推理跨字节")
        self.assertEqual(msg["content"], "桥梁名称")
        self.assertEqual([t["id"] for t in msg["tool_calls"]], ["call-a", "call-b"])
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], '{"桥": BAD JSON}')
        self.assertEqual(msg["tool_calls"][1]["function"], {"name": "read_source", "arguments": "{}"})
        self.assertEqual(receipt["usage"], USAGE)
        self.assertEqual(receipt["raw_sse_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(receipt["output_sha256"], hashlib.sha256(m._bytes(result)).hexdigest())
        self.assertEqual(receipt["wire_attempt_id"], "wire-001")
        self.assertEqual(b"".join(captured), raw)
        self.assertEqual(len(self.last_response.completed), 1)
        self.assertTrue(self.last_response.closed)
        self.assertEqual(self.last_client.calls, [])

    def test_prepare_changes_only_transport_flags_and_strict_one_choice(self):
        prepared = m.prepare_stream_payload(self.payload)
        self.assertEqual({k:v for k,v in prepared.items() if k not in {"stream", "stream_options"}}, self.payload)
        self.assertEqual(prepared["stream_options"], {"include_usage": True})
        self.assertNotIn("stream", self.payload)
        for n in (True, 2, "1"):
            with self.subTest(n=n), self.assertRaises(ValueError): m.prepare_stream_payload({"n": n})

    def test_missing_final_evidence_or_malformed_tail_fails_closed(self):
        cases = [event({"content":"ok"}) + event(finish="stop") + event(choices=False, usage=USAGE),
                 event({"content":"ok"}) + event(choices=False, usage=USAGE) + b"data: [DONE]\n\n",
                 event({"content":"ok"}) + event(finish="stop") + b"data: [DONE]\n\n",
                 complete() + b"data: {", complete() + b"\xff", b"data: badJSON\n\n",
                 complete() + event({"content":"after"})]
        for raw in cases:
            with self.subTest(size=len(raw)):
                exc = self.failure([raw])
                self.assertEqual(exc.code, "stream_protocol_error")

    def test_actual_wall_time_cannot_reset_with_fresh_bytes(self):
        raw = event({"reasoning_content": "first"})
        seen = []
        exc = self.failure([raw, raw, raw], advances=[5, 5, 5], deadline=112, on_chunk=seen.append)
        self.assertEqual(exc.code, "wall_or_read_timeout")
        self.assertEqual(self.last_client._opener.calls[0][1], 12)
        self.assertEqual(self.last_response.timeouts, [12, 7, 2])
        self.assertEqual(exc.receipt["received_bytes"], 3*len(raw))
        self.assertEqual(exc.receipt["raw_sse_sha256"], hashlib.sha256(raw*3).hexdigest())
        self.assertEqual(exc.receipt["seconds"], 15)
        self.assertEqual(len(seen), 3)

    def test_initial_open_and_read_idle_have_actual_deadlines(self):
        exc = self.failure([TimeoutError("idle")], deadline=800)
        self.assertEqual(exc.code, "wall_or_read_timeout")
        self.assertEqual(self.last_client._opener.calls[0][1], 60)
        self.assertEqual(self.last_response.timeouts, [60])
        exc = self.failure([complete()], deadline=110, open_advance=11)
        self.assertEqual(exc.code, "wall_or_read_timeout")
        self.assertEqual(self.last_response.timeouts, [])
        self.assertEqual(self.last_client._opener.calls[0][1], 10)

    def test_reasoning_repeated_lines_and_short_cycles_cancel(self):
        patterns = [("This reasoning cannot yet conclude its next logical step.\n" * 8),
                    ("我们需要再次确认相同问题并继续重复当前推理步骤因为还没有确认结束。" * 10)]
        for text in patterns:
            with self.subTest(kind=len(text)):
                exc = self.failure([event({"reasoning_content": text})])
                self.assertEqual(exc.code, "reasoning_repetition")
                self.assertNotIn(text, str(exc))

    def test_repeated_table_headers_and_tool_argument_text_are_not_reasoning_loops(self):
        repeat = "构件 | 病害 | 数量\n支座 | 锈蚀 | 1\n" * 100
        result, _ = self.invoke([complete({"content": repeat, "tool_calls": [
            {"index":0,"id":"save1","function":{"name":"save","arguments":repeat}}]}, finish="tool_calls")])
        msg = result["choices"][0]["message"]
        self.assertEqual(msg["content"], repeat)
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], repeat)

    def test_cancellation_callback_uses_closed_exact_response_and_marker_acceptance(self):
        calls = []
        def confirmed(receipt, response):
            self.assertTrue(response.closed)
            self.assertEqual(receipt["wire_attempt_id"], response.wire_attempt_id)
            calls.append((receipt, response))
            return {"status":"cancellation_confirmed","wire_attempt_id":response.wire_attempt_id}
        exc = self.failure([TimeoutError("idle")], on_cancel=confirmed)
        self.assertEqual(exc.cancel_status, "cancellation_confirmed")
        self.assertEqual(len(calls), 1)
        response = Response([TimeoutError("idle")], self.clock, marker_status="cancellation_unknown")
        exc = self.failure([], response=response, on_cancel=confirmed)
        self.assertEqual(exc.cancel_status, "unknown")
        self.assertEqual(len(response.cancelled), 1)

    def test_unknown_or_failed_cancel_never_claims_confirmed(self):
        for callback in (None, lambda *_: {"status":"unknown"}, lambda *_: (_ for _ in ()).throw(RuntimeError("private diagnostic"))):
            with self.subTest(callback=callback is None):
                response = Response([TimeoutError("idle")], self.clock, marker_status="cancellation_unknown")
                exc = self.failure([], response=response, on_cancel=callback)
                self.assertNotEqual(exc.cancel_status, "cancellation_confirmed")
                self.assertNotIn("private diagnostic", str(exc))

    def test_http_error_wrong_identity_redirect_and_unenforceable_timeout(self):
        response = HTTPError("http://127.0.0.1:18110/v1/chat/completions",400,"bad",{},None)
        exc = self.failure([], response=response)
        self.assertEqual(exc.receipt["http_status"], 400)
        response.close()
        exc = self.failure([event({"content":"wrong"},model="other") + event(finish="stop",model="other")
                            + event(choices=False,usage=USAGE,model="other") + b"data: [DONE]\n\n"])
        self.assertEqual(exc.code, "stream_protocol_error")
        response = Response([complete()], self.clock)
        response.geturl = lambda: "http://outside.invalid"
        self.assertEqual(self.failure([],response=response).code,"stream_protocol_error")
        response = Response([complete()], self.clock); response.set_read_timeout = None
        self.assertEqual(self.failure([],response=response).code,"stream_protocol_error")

    def test_usage_and_tool_identity_are_exact_not_guessed(self):
        raw = (event({"tool_calls":[{"index":0,"id":"original","function":{"name":"save","arguments":"{"}}]})
               + event({"tool_calls":[{"index":0,"id":"changed","function":{"arguments":"}"}}]}))
        self.assertEqual(self.failure([raw]).code,"stream_protocol_error")
        invalid = {"prompt_tokens":100,"completion_tokens":20,"total_tokens":121}
        raw = event({"content":"x"}) + event(finish="stop") + event(choices=False,usage=invalid) + b"data: [DONE]\n\n"
        self.assertEqual(self.failure([raw]).code,"stream_protocol_error")

    def test_length_is_transport_complete_and_preserves_truthful_finish(self):
        result, receipt = self.invoke([complete(finish="length")])
        self.assertEqual(result["choices"][0]["finish_reason"], "length")
        self.assertEqual(receipt["finish_reason"], "length")
        self.assertTrue(receipt["usage_known"])
        self.assertNotIn("quality_passed", receipt)


if __name__ == "__main__": unittest.main()
