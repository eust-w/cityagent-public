"""Real loopback HTTP with synthetic responses; NOT SDK/GPU/model validation."""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import traceback
import unittest
from urllib.parse import urlsplit
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest import document_model_gateway as module
from bridge_evidence.contest.local_model_client import LocalModelClient
from bridge_evidence.contest.run_state import atomic_json


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sdk_payload():
    tool_call = {"id": "synthetic-tool-call", "type": "function", "function": {
        "name": "read_source", "arguments": '{"evidence_id":"E1","start":0,"end":3}',
    }}
    return {
        "model": "own-model", "stream": False, "max_tokens": 16, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": "fixed synthetic instructions"},
            {"role": "user", "content": "private synthetic report"},
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": tool_call["id"], "content": '{"text":"示例证据"}'},
        ],
        "tools": [{"type": "function", "function": {
            "name": "read_source", "description": "read source text",
            "parameters": {"type": "object", "properties": {
                "evidence_id": {"type": "string"}, "start": {"type": "integer"},
                "end": {"type": "integer"},
            }, "required": ["evidence_id", "start", "end"]},
        }}],
        "tool_choice": "auto", "parallel_tool_calls": False,
    }


def completion(*, prompt_tokens=23, finish_reason="tool_calls"):
    return {
        "id": "synthetic-upstream-response", "model": "own-model",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "next-synthetic-tool", "type": "function", "function": {
                "name": "read_workspace", "arguments": "{}",
            }}],
        }, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 7,
                  "total_tokens": prompt_tokens + 7},
    }


@contextmanager
def synthetic_upstream(response=None, *, token_count=23, generation_status=200):
    """Two endpoints only; never loads a model or calls another service."""
    requests = []
    response = completion() if response is None else response

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            if self.path == "/tokenize":
                body = {"count": len(payload["prompt"]) if "prompt" in payload else token_count}
            elif self.path == "/v1/chat/completions":
                body = response
            else:
                self.send_error(404)
                return
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(generation_status if self.path == "/v1/chat/completions" else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def post(gateway, payload, *, path="/v1/chat/completions"):
    target = urlsplit(gateway.url)
    connection = HTTPConnection(target.hostname, target.port, timeout=3)
    try:
        connection.request("POST", path, json.dumps(payload, ensure_ascii=False).encode(),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


class SyntheticChatTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        return self.render(messages, kwargs["tools"])

    def encode(self, text, **_kwargs):
        return list(range(len(text)))

    @staticmethod
    def render(messages, tools):
        return json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)


class DocumentModelGatewayTests(unittest.TestCase):
    def gateway(self, client, **kwargs):
        return module.AuditedModelGateway(client, **{
            "deadline": time.monotonic() + 20, "max_calls": 5, "max_output_tokens": 16, **kwargs,
        })

    def test_loopback_preserves_tools_messages_and_actual_http_receipts(self):
        payload = sdk_payload()
        expected = completion()
        with synthetic_upstream(expected) as (url, requests):
            client = LocalModelClient(url, "own-model")
            original_timeout = client.timeout
            with self.gateway(client) as gateway:
                status, response = post(gateway, payload)
            self.assertEqual(status, 200)
            self.assertEqual(response, expected)
            self.assertEqual(requests[-1], ("/v1/chat/completions", payload))
            self.assertEqual(requests[0], ("/tokenize", {
                "model": "own-model", "messages": payload["messages"],
                "add_generation_prompt": True, "tools": payload["tools"],
                "chat_template_kwargs": {"enable_thinking": False},
            }))
            self.assertEqual([row["kind"] for row in client.calls], ["tokenize", "agent_generation"])
            audit = gateway.receipts[0]
            self.assertEqual(audit["call_id"], client.calls[1]["call_id"])
            self.assertEqual(audit["request_sha256"], digest(payload))
            self.assertEqual(audit["response_sha256"], digest(expected))
            self.assertEqual(audit["usage"], expected["usage"])
            self.assertEqual(audit["finish_reason"], "tool_calls")
            self.assertEqual(audit["status"], "success")
            self.assertEqual(client.timeout, original_timeout)
            self.assertNotIn("private synthetic report", str(gateway.receipts))
            self.assertNotIn("示例证据", str(client.calls))
            self.assertFalse(gateway.thread.is_alive())

    def test_direct_review_forward_uses_generation_kind_and_counts_once_without_proxy_server(self):
        payload = sdk_payload()
        expected = completion(finish_reason="stop")
        expected["choices"][0]["message"] = {"role": "assistant", "content": '{"review":"ok"}'}
        with synthetic_upstream(expected) as (url, requests):
            client = LocalModelClient(url, "own-model", timeout=180)
            gateway = self.gateway(client, max_calls=1, request_kind="generation")
            response = gateway.forward(payload)
            self.assertEqual(response, expected)
            self.assertIsNone(gateway.server)
            self.assertIsNone(gateway.thread)
            self.assertEqual([path for path, _ in requests], ["/tokenize", "/v1/chat/completions"])
            self.assertEqual(requests[-1][1], payload)
            self.assertEqual([row["kind"] for row in client.calls], ["tokenize", "generation"])
            self.assertEqual(gateway.receipts[0]["call_id"], client.calls[-1]["call_id"])
            self.assertEqual(gateway.receipts[0]["status"], "success")
            self.assertEqual(client.calls[-1]["usage"], expected["usage"])
            self.assertEqual(client.timeout, 180)

    def test_dispatch_is_persisted_before_generation_and_terminal_ids_match(self):
        for kind in ("agent_generation", "generation"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, \
                    synthetic_upstream() as (url, requests):
                client = LocalModelClient(url, "own-model")
                path = Path(directory) / "pending.json"
                dispatches = []

                def persist_dispatch(record):
                    self.assertEqual([p for p, _ in requests], ["/tokenize"])
                    atomic_json(path, record)
                    dispatches.append(copy.deepcopy(record))

                gateway = self.gateway(client, request_kind=kind, on_dispatch=persist_dispatch)
                real_request = client._request

                def checked_request(endpoint, payload, *, kind):
                    if kind in {"generation", "agent_generation"}:
                        self.assertEqual(json.loads(path.read_text()), dispatches[0])
                    return real_request(endpoint, payload, kind=kind)

                with patch.object(client, "_request", side_effect=checked_request):
                    gateway.forward(sdk_payload())
                self.assertEqual(len(dispatches), 1)
                dispatch = dispatches[0]
                self.assertEqual(dispatch["request_kind"], kind)
                self.assertEqual(dispatch["request_sha256"], digest(sdk_payload()))
                self.assertEqual(dispatch["max_tokens"], 16)
                self.assertEqual(dispatch["prompt_tokens_checked"], 23)
                self.assertEqual(datetime.fromisoformat(dispatch["started_at"]).utcoffset(), timezone.utc.utcoffset(None))
                self.assertEqual(gateway.receipts[0]["request_id"], dispatch["request_id"])
                self.assertEqual(client.calls[-1]["request_id"], dispatch["request_id"])
                self.assertTrue(gateway.receipts[0]["generation_attempted"])
                self.assertEqual(gateway.receipts[0]["status"], "success")
                self.assertNotIn("request_id", client.calls[0])  # Tokenization is not generation.
                self.assertNotIn("private synthetic report", str(dispatch))

    def test_http_failure_retains_dispatch_id_in_terminal_and_failed_http_receipts(self):
        dispatches, terminals = [], []
        with synthetic_upstream(generation_status=503) as (url, requests):
            client = LocalModelClient(url, "own-model", timeout=180)
            gateway = self.gateway(client, on_dispatch=lambda record: dispatches.append(record))
            gateway.on_receipt = lambda audit: terminals.append(copy.deepcopy(audit))
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                gateway.forward(sdk_payload())
            self.assertEqual([path for path, _ in requests], ["/tokenize", "/v1/chat/completions"])
            self.assertEqual(len(dispatches), 1)
            audit = terminals[0]
            self.assertEqual(audit["request_id"], dispatches[0]["request_id"])
            self.assertEqual(audit["request_id"], client.calls[-1]["request_id"])
            self.assertEqual(audit["call_id"], client.calls[-1]["call_id"])
            self.assertEqual(audit["http_status"], 503)
            self.assertEqual(audit["failure_stage"], "generation_http")
            self.assertTrue(audit["generation_attempted"])
            self.assertEqual(audit["status"], "rejected")
            self.assertEqual(client.calls[-1]["status"], "failed")
            self.assertEqual(client.timeout, 180)

    def test_dispatch_callback_failure_prevents_generation_and_retains_intent_id(self):
        dispatched = []
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model", timeout=180)

            def cannot_persist(record):
                dispatched.append(record)
                raise OSError("synthetic disk failure")

            gateway = self.gateway(client, on_dispatch=cannot_persist)
            with self.assertRaisesRegex(RuntimeError, "Could not persist.*dispatch"):
                gateway.forward(sdk_payload())
            self.assertEqual([path for path, _ in requests], ["/tokenize"])
            self.assertEqual([row["kind"] for row in client.calls], ["tokenize"])
            audit = gateway.receipts[0]
            self.assertEqual(audit["request_id"], dispatched[0]["request_id"])
            self.assertFalse(audit["generation_attempted"])
            self.assertNotIn("call_id", audit)
            self.assertEqual(audit["failure_stage"], "dispatch_callback")
            self.assertEqual(audit["dispatch_callback_error"], "OSError")
            self.assertEqual(client.timeout, 180)

    def test_context_rejection_never_creates_a_generation_dispatch(self):
        dispatched = []
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model", max_model_len=38)
            gateway = self.gateway(client, on_dispatch=lambda record: dispatched.append(record))
            with self.assertRaisesRegex(ValueError, "exceed context"):
                gateway.forward(sdk_payload())
            self.assertEqual(dispatched, [])
            self.assertEqual([path for path, _ in requests], ["/tokenize"])
            self.assertNotIn("request_id", gateway.receipts[0])

    def test_deadline_expiring_in_dispatch_prevents_generation(self):
        dispatched = []
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model", timeout=180)
            gateway = self.gateway(client)

            def persist_and_expire(record):
                dispatched.append(record)
                gateway.deadline = time.monotonic() - 1

            gateway.on_dispatch = persist_and_expire
            with self.assertRaises(TimeoutError):
                gateway.forward(sdk_payload())
            self.assertEqual([path for path, _ in requests], ["/tokenize"])
            audit = gateway.receipts[0]
            self.assertEqual(audit["request_id"], dispatched[0]["request_id"])
            self.assertEqual(audit["failure_stage"], "before_generation")
            self.assertFalse(audit["generation_attempted"])
            self.assertEqual(client.timeout, 180)

    def test_request_kind_only_accepts_generation_and_agent_generation(self):
        client = LocalModelClient("http://127.0.0.1:1", "own-model")
        for kind in ("tokenize", "models", "", None, [], True):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.gateway(client, request_kind=kind)
        self.assertEqual(client.calls, [])
        self.assertEqual(self.gateway(client).request_kind, "agent_generation")

    def test_receipt_callback_observes_appended_audit_released_lock_and_restored_timeout(self):
        with synthetic_upstream() as (url, _requests):
            client = LocalModelClient(url, "own-model", timeout=180)
            gateway = self.gateway(client)
            saved = []

            def persist(audit):
                self.assertIs(audit, gateway.receipts[-1])
                self.assertFalse(gateway._forward_lock.locked())
                self.assertEqual(client.timeout, 180)
                saved.append(audit)

            gateway.on_receipt = persist
            gateway.forward(sdk_payload())
            with self.assertRaises(ValueError):
                gateway.forward({**sdk_payload(), "model": "wrong"})
            self.assertEqual([audit["status"] for audit in saved], ["success", "rejected"])

    def test_receipt_callback_failure_preserves_the_original_request_error(self):
        client = LocalModelClient("http://127.0.0.1:1", "own-model")
        gateway = self.gateway(client)

        def cannot_persist(_audit):
            raise OSError("synthetic disk failure")

        gateway.on_receipt = cannot_persist
        with self.assertRaisesRegex(ValueError, "pinned own model"):
            gateway.forward({**sdk_payload(), "model": "wrong"})
        self.assertEqual(gateway.receipts[0]["error"], "ValueError")
        self.assertEqual(gateway.receipts[0]["receipt_callback_error"], "OSError")
        self.assertEqual(gateway.receipts[0]["status"], "rejected")
        self.assertEqual(client.calls, [])

    def test_receipt_callback_failure_after_success_stops_with_an_explicit_audit(self):
        with synthetic_upstream() as (url, _requests):
            client = LocalModelClient(url, "own-model", timeout=180)
            gateway = self.gateway(client)

            def cannot_persist(_audit):
                raise OSError("synthetic disk failure")

            gateway.on_receipt = cannot_persist
            with self.assertRaisesRegex(RuntimeError, "Could not persist"):
                gateway.forward(sdk_payload())
            self.assertEqual(gateway.receipts[0]["status"], "rejected")
            self.assertEqual(gateway.receipts[0]["failure_stage"], "receipt_callback")
            self.assertEqual(gateway.receipts[0]["receipt_callback_error"], "OSError")
            self.assertEqual(client.calls[-1]["status"], "success")
            self.assertFalse(gateway._forward_lock.locked())
            self.assertEqual(client.timeout, 180)

    def test_unpinned_model_stream_and_invalid_output_limits_never_reach_upstream(self):
        invalid = [
            {"model": "another-model"}, {"stream": True}, {"stream": "false"},
            {"messages": []}, {"messages": "not an array"},
        ] + [{"max_tokens": value} for value in (0, 17, True, 1.5, "16", None)]
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model")
            with self.gateway(client) as gateway:
                for fields in invalid:
                    with self.subTest(fields=fields):
                        status, response = post(gateway, {**sdk_payload(), **fields})
                        self.assertEqual(status, 400)
                        self.assertEqual(response["error"]["type"], "ValueError")
            self.assertEqual(requests, [])
            self.assertEqual(client.calls, [])
            self.assertEqual(len(gateway.receipts), len(invalid))
            self.assertTrue(all(row["status"] == "rejected" for row in gateway.receipts))

    def test_context_overflow_does_not_generate_or_truncate_and_exact_fit_succeeds(self):
        payload = sdk_payload()
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model", max_model_len=38)
            with self.gateway(client) as gateway:
                status, _ = post(gateway, payload)
                self.assertEqual(status, 400)  # 23 + 16 > 38.
                self.assertEqual([path for path, _ in requests], ["/tokenize"])
                self.assertEqual(requests[0][1]["messages"], payload["messages"])
                self.assertNotIn("call_id", gateway.receipts[0])
                status, _ = post(gateway, {**payload, "max_tokens": 15})
                self.assertEqual(status, 200)
            self.assertEqual([row["kind"] for row in client.calls], ["tokenize", "tokenize", "agent_generation"])

    def test_prompt_usage_mismatch_rejects_actual_generation_and_marks_its_receipt_failed(self):
        with synthetic_upstream(completion(prompt_tokens=24)) as (url, requests):
            client = LocalModelClient(url, "own-model")
            with self.gateway(client) as gateway:
                status, _ = post(gateway, sdk_payload())
            self.assertEqual(status, 400)
            self.assertEqual([path for path, _ in requests], ["/tokenize", "/v1/chat/completions"])
            self.assertEqual(client.calls[-1]["status"], "failed")
            self.assertEqual(gateway.receipts[0]["call_id"], client.calls[-1]["call_id"])
            self.assertEqual(gateway.receipts[0]["usage"]["prompt_tokens"], 24)
            self.assertEqual(gateway.receipts[0]["status"], "rejected")

    def test_missing_usage_and_truncated_completion_are_not_accepted(self):
        missing_usage = completion()
        missing_usage.pop("usage")
        for response in (missing_usage, completion(finish_reason="length")):
            with self.subTest(response=response), synthetic_upstream(response) as (url, requests):
                client = LocalModelClient(url, "own-model")
                with self.gateway(client) as gateway:
                    status, returned = post(gateway, sdk_payload())
                self.assertEqual(status, 400)
                self.assertEqual(returned["error"]["type"],
                    "VerifiedGenerationLengthError" if response["choices"][0]["finish_reason"] == "length" else "ValueError")
                self.assertEqual(len(requests), 2)
                self.assertEqual(client.calls[-1]["status"], "failed")
                self.assertEqual(gateway.receipts[0]["status"], "rejected")

    def test_total_request_budget_blocks_next_request_before_tokenization(self):
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model")
            with self.gateway(client, max_calls=1) as gateway:
                self.assertEqual(post(gateway, sdk_payload())[0], 200)
                status, response = post(gateway, sdk_payload())
            self.assertEqual(status, 400)
            self.assertEqual(response["error"]["type"], "RuntimeError")
            self.assertEqual(len(requests), 2)
            self.assertEqual([row["status"] for row in gateway.receipts], ["success", "rejected"])
            self.assertNotIn("call_id", gateway.receipts[1])

    def test_concurrent_requests_cannot_overbook_the_report_budget(self):
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model")
            with self.gateway(client, max_calls=1) as gateway:
                ready = threading.Barrier(2)

                def send():
                    ready.wait(timeout=2)
                    return post(gateway, sdk_payload())[0]

                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(send) for _ in range(2)]
                    statuses = sorted(future.result(timeout=4) for future in futures)
            self.assertEqual(statuses, [200, 400])
            self.assertEqual([path for path, _ in requests], ["/tokenize", "/v1/chat/completions"])
            self.assertEqual(sorted(row["status"] for row in gateway.receipts), ["rejected", "success"])
            self.assertEqual(sum("call_id" in row for row in gateway.receipts), 1)

    def test_success_callback_receives_validated_request_and_actual_generation_receipt(self):
        callback_calls = []
        with synthetic_upstream() as (url, _requests):
            client = LocalModelClient(url, "own-model")
            with self.gateway(client) as gateway:
                gateway.on_success = lambda payload, receipt: callback_calls.append((payload, receipt))
                self.assertEqual(post(gateway, sdk_payload())[0], 200)
            self.assertEqual(len(callback_calls), 1)
            payload, receipt = callback_calls[0]
            self.assertEqual(payload, sdk_payload())
            self.assertIs(receipt, client.calls[-1])
            self.assertEqual(receipt["status"], "success")
            self.assertEqual(receipt["finish_reason"], "tool_calls")
            self.assertEqual(receipt["prompt_tokens_checked"], 23)

    def test_response_callback_runs_before_success_and_sdk_return_with_bound_full_response(self):
        # Invalid JSON inside assistant text is SDK/workflow parsing territory;
        # it must survive here for diagnosis rather than being auto-repaired.
        expected = completion(finish_reason="stop")
        expected["choices"][0]["message"] = {"role": "assistant", "content": '{"incomplete":'}
        expected["custom_metadata"] = {"nested": ["完整合成原始响应", 0, None]}
        for kind in ("generation", "agent_generation"):
            with self.subTest(kind=kind), synthetic_upstream(expected) as (url, requests):
                client = LocalModelClient(url, "own-model")
                events, saved = [], []

                def persist_response(result, audit):
                    events.append("response")
                    self.assertEqual(len(requests), 2)
                    self.assertEqual(client.calls[-1]["kind"], kind)
                    self.assertEqual(client.calls[-1]["status"], "success")
                    self.assertEqual(audit["call_id"], client.calls[-1]["call_id"])
                    self.assertEqual(audit["request_id"], client.calls[-1]["request_id"])
                    self.assertEqual(audit["request_sha256"], digest(sdk_payload()))
                    self.assertEqual(audit["response_sha256"], digest(result))
                    self.assertEqual(audit["status"], "response_validated")
                    self.assertTrue(audit["response_validated"])
                    self.assertTrue(audit["generation_attempted"])
                    self.assertEqual(audit["http_status"], 200)
                    saved.append((result, audit))

                with self.gateway(client, request_kind=kind, on_response=persist_response) as gateway:
                    gateway.on_success = lambda *_args: events.append("success")
                    gateway.on_receipt = lambda _audit: events.append("receipt")
                    status, result = post(gateway, sdk_payload())
                    events.append("returned")
                self.assertEqual(status, 200)
                self.assertEqual(result, expected)
                self.assertEqual(saved[0][0], expected)
                self.assertEqual(events, ["response", "success", "receipt", "returned"])
                self.assertTrue(gateway.receipts[0]["response_callback_completed"])
                self.assertEqual([call["kind"] for call in client.calls], ["tokenize", kind])

    def test_response_callback_receives_deep_copies_and_cannot_change_delivery_or_receipts(self):
        expected = completion()
        with synthetic_upstream(expected) as (url, _requests):
            client = LocalModelClient(url, "own-model")
            retained = []

            def mutate(result, audit):
                retained.append((result, audit))
                result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "mutated"
                result["usage"]["prompt_tokens"] = 999
                audit["usage"]["prompt_tokens"] = 1000
                audit["request_id"] = "forged"
                audit["call_id"] = "forged"
                audit["response_sha256"] = "forged"

            gateway = self.gateway(client)
            gateway.on_response = mutate
            result = gateway.forward(sdk_payload())
            self.assertEqual(result, expected)
            self.assertEqual(client.calls[-1]["usage"], expected["usage"])
            self.assertEqual(gateway.receipts[0]["usage"], expected["usage"])
            self.assertEqual(gateway.receipts[0]["response_sha256"], digest(expected))
            self.assertNotEqual(gateway.receipts[0]["request_id"], "forged")
            self.assertNotEqual(gateway.receipts[0]["call_id"], "forged")
            retained[0][0]["usage"].clear()
            retained[0][1]["usage"].clear()
            self.assertEqual(result, expected)
            self.assertEqual(gateway.receipts[0]["usage"], expected["usage"])

    def test_response_callback_is_not_called_for_invalid_response_or_http_failure(self):
        malformed = completion()
        malformed["choices"] = []
        for response, status in ((completion(prompt_tokens=24), 200),
                                 (completion(finish_reason="length"), 200),
                                 (malformed, 200), (completion(), 503)):
            with self.subTest(response=response, status=status), synthetic_upstream(
                    response, generation_status=status) as (url, requests):
                client = LocalModelClient(url, "own-model")
                calls = []
                gateway = self.gateway(client, on_response=lambda *args: calls.append(args))
                with self.assertRaises(Exception):
                    gateway.forward(sdk_payload())
                self.assertEqual(calls, [])
                self.assertEqual([path for path, _ in requests], ["/tokenize", "/v1/chat/completions"])
                self.assertNotIn("response_callback_completed", gateway.receipts[0])

    def test_response_callback_failure_blocks_sdk_return_preserves_http_success_and_does_not_retry(self):
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model", timeout=180)
            successes, terminals = [], []

            def cannot_persist(_result, _audit):
                raise OSError("synthetic-secret-response-text /private/path/sidecar")

            with self.gateway(client, on_response=cannot_persist) as gateway:
                gateway.on_success = lambda *args: successes.append(args)
                gateway.on_receipt = lambda audit: terminals.append(copy.deepcopy(audit))
                status, response = post(gateway, sdk_payload())
            self.assertEqual(status, 400)
            self.assertEqual(response["error"]["type"], "RuntimeError")
            self.assertEqual(successes, [])
            self.assertEqual([path for path, _ in requests], ["/tokenize", "/v1/chat/completions"])
            self.assertEqual([call["kind"] for call in client.calls], ["tokenize", "agent_generation"])
            self.assertEqual(client.calls[-1]["status"], "success")
            self.assertEqual(client.timeout, 180)
            self.assertEqual(len(terminals), 1)
            audit = gateway.receipts[0]
            self.assertEqual(audit["status"], "rejected")
            self.assertEqual(audit["failure_stage"], "response_callback")
            self.assertEqual(audit["response_callback_error"], "OSError")
            self.assertTrue(audit["generation_attempted"])
            self.assertTrue(audit["response_validated"])
            self.assertEqual(audit["http_status"], 200)
            self.assertEqual(audit["response_sha256"], digest(completion()))
            self.assertEqual(audit["call_id"], client.calls[-1]["call_id"])
            self.assertEqual(audit["request_id"], client.calls[-1]["request_id"])
            self.assertNotIn("synthetic-secret", str(response) + str(audit))
            self.assertNotIn("/private/path", str(response) + str(audit))

    def test_direct_response_callback_failure_does_not_expose_callback_exception_chain(self):
        with synthetic_upstream() as (url, _requests):
            client = LocalModelClient(url, "own-model")
            def fail(_result, _audit):
                raise RuntimeError("sensitive-original-report-text")
            gateway = self.gateway(client, on_response=fail)
            try:
                gateway.forward(sdk_payload())
            except RuntimeError as exc:
                self.assertEqual(str(exc), "Could not persist the validated model response")
                formatted = "".join(traceback.format_exception(exc))
                self.assertNotIn("sensitive-original-report-text", formatted)
                self.assertIsNone(exc.__cause__)
                self.assertTrue(exc.__suppress_context__)
            else:
                self.fail("Persistence failure must abort forward")

    def test_persisted_response_stays_available_if_later_success_hook_fails(self):
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model")
            saved = []
            gateway = self.gateway(client, on_response=lambda result, audit: saved.append((result, audit)))
            def fail_later(_payload, _receipt):
                raise RuntimeError("synthetic later workflow failure")
            gateway.on_success = fail_later
            with self.assertRaisesRegex(RuntimeError, "later workflow failure"):
                gateway.forward(sdk_payload())
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0][0], completion())
            self.assertEqual(saved[0][1]["status"], "response_validated")
            self.assertEqual(gateway.receipts[0]["status"], "rejected")
            self.assertTrue(gateway.receipts[0]["response_callback_completed"])
            self.assertEqual(client.calls[-1]["status"], "success")
            self.assertEqual(len(requests), 2)

    def test_invalid_usage_or_truncation_does_not_invoke_success_callback(self):
        for response in (completion(prompt_tokens=24), completion(finish_reason="length")):
            with self.subTest(response=response), synthetic_upstream(response) as (url, _requests):
                client = LocalModelClient(url, "own-model")
                callback_calls = []
                with self.gateway(client) as gateway:
                    gateway.on_success = lambda *args: callback_calls.append(args)
                    self.assertEqual(post(gateway, sdk_payload())[0], 400)
                self.assertEqual(callback_calls, [])

    def test_callback_failure_rejects_gateway_receipt_and_restores_client_state(self):
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model", timeout=180)
            with self.gateway(client) as gateway:
                def fail_callback(_payload, _receipt):
                    raise RuntimeError("synthetic workflow bookkeeping failure")

                gateway.on_success = fail_callback
                status, response = post(gateway, sdk_payload())
                self.assertEqual(status, 400)
                self.assertEqual(response["error"]["type"], "RuntimeError")
                self.assertEqual(gateway.receipts[0]["status"], "rejected")
                self.assertEqual(gateway.receipts[0]["error"], "RuntimeError")
                # The model's real successful HTTP response remains distinguishable
                # from the rejected workflow operation; it was not rolled back.
                self.assertEqual(client.calls[-1]["status"], "success")
                self.assertEqual(client.timeout, 180)
                gateway.on_success = None
                self.assertEqual(post(gateway, sdk_payload())[0], 200)
            self.assertEqual(len(requests), 4)

    def test_before_request_refusal_has_no_http_and_does_not_consume_native_budget(self):
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model")
            checks = []

            def exhausted_workflow_budget():
                checks.append(len(requests))
                raise RuntimeError("synthetic total workflow generation budget exhausted")

            with self.gateway(client, max_calls=1, before_request=exhausted_workflow_budget) as gateway:
                status, response = post(gateway, sdk_payload())
                self.assertEqual(status, 400)
                self.assertEqual(response["error"]["type"], "RuntimeError")
                self.assertEqual(checks, [0])
                self.assertEqual(requests, [])
                self.assertEqual(client.calls, [])
                gateway.before_request = lambda: checks.append(len(requests))
                self.assertEqual(post(gateway, sdk_payload())[0], 200)
                self.assertEqual(post(gateway, sdk_payload())[0], 400)
            self.assertEqual(checks, [0, 0])
            self.assertEqual(len(requests), 2)

    def test_before_request_expiration_prevents_http_and_invalid_requests_skip_hook(self):
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model", timeout=180)
            checks = []
            with self.gateway(client) as gateway:
                def expire():
                    checks.append("checked")
                    gateway.deadline = time.monotonic() - 1

                gateway.before_request = expire
                self.assertEqual(post(gateway, {**sdk_payload(), "model": "other"})[0], 400)
                self.assertEqual(checks, [])
                status, response = post(gateway, sdk_payload())
            self.assertEqual(status, 400)
            self.assertEqual(response["error"]["type"], "TimeoutError")
            self.assertEqual(checks, ["checked"])
            self.assertEqual(requests, [])
            self.assertEqual(client.timeout, 180)

    def test_expired_deadline_prevents_even_tokenization(self):
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model")
            with self.gateway(client, deadline=time.monotonic() - 1) as gateway:
                status, response = post(gateway, sdk_payload())
            self.assertEqual(status, 400)
            self.assertEqual(response["error"]["type"], "TimeoutError")
            self.assertEqual(requests, [])
            self.assertEqual(client.calls, [])
            self.assertEqual(gateway.receipts[0]["status"], "rejected")

    def test_deadline_that_expires_during_tokenization_prevents_generation(self):
        client = LocalModelClient("http://127.0.0.1:1", "own-model")
        gateway = self.gateway(client)

        def count_and_expire(_payload):
            gateway.deadline = time.monotonic() - 1
            return 23

        with patch.object(gateway, "count_request", side_effect=count_and_expire), \
                patch.object(client, "_request") as generate:
            with self.assertRaises(TimeoutError):
                gateway.forward(sdk_payload())
            generate.assert_not_called()
        self.assertEqual(gateway.receipts[0]["prompt_tokens_checked"], 23)

    def test_tokenization_timeout_is_bounded_by_remaining_report_time(self):
        client = LocalModelClient("http://127.0.0.1:1", "own-model", timeout=180)
        gateway = self.gateway(client, deadline=time.monotonic() + 2)
        observed = []

        def inspect_timeout(_payload):
            observed.append(client.timeout)
            raise TimeoutError("synthetic tokenization failure")

        with patch.object(gateway, "count_request", side_effect=inspect_timeout):
            with self.assertRaises(TimeoutError):
                gateway.forward(sdk_payload())
        self.assertEqual(len(observed), 1)
        self.assertGreater(observed[0], 0)
        self.assertLessEqual(observed[0], 2)
        self.assertEqual(client.timeout, 180)

    def test_generation_timeout_uses_time_remaining_after_tokenization(self):
        client = LocalModelClient("http://127.0.0.1:1", "own-model", timeout=180)
        gateway = self.gateway(client, deadline=time.monotonic() + 2)
        observed = []

        def count(_payload):
            observed.append(("count", client.timeout))
            gateway.deadline = time.monotonic() + 0.5
            return 23

        def generate(_endpoint, _payload, **_kwargs):
            observed.append(("generate", client.timeout))
            return completion(), {"call_id": "synthetic-not-http", "status": "success"}

        with patch.object(gateway, "count_request", side_effect=count), \
                patch.object(client, "_request", side_effect=generate):
            gateway.forward(sdk_payload())
        self.assertEqual([kind for kind, _ in observed], ["count", "generate"])
        self.assertLessEqual(observed[0][1], 2)
        self.assertLessEqual(observed[1][1], 0.5)
        self.assertGreater(observed[1][1], 0)
        self.assertEqual(client.timeout, 180)

    def test_generation_returning_after_deadline_is_rejected_without_success_callback(self):
        client = LocalModelClient("http://127.0.0.1:1", "own-model", timeout=180)
        gateway = self.gateway(client)
        receipt = {"call_id": "synthetic-not-http", "status": "success"}
        callbacks = []
        gateway.on_success = lambda *args: callbacks.append(args)

        def late_generation(_endpoint, _payload, **_kwargs):
            gateway.deadline = time.monotonic() - 1
            return completion(), receipt

        with patch.object(gateway, "count_request", return_value=23), \
                patch.object(client, "_request", side_effect=late_generation):
            with self.assertRaises(TimeoutError):
                gateway.forward(sdk_payload())
        self.assertEqual(callbacks, [])
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(gateway.receipts[0]["call_id"], receipt["call_id"])
        self.assertEqual(gateway.receipts[0]["status"], "rejected")
        self.assertEqual(client.timeout, 180)

    def test_waiting_for_another_request_respects_deadline_without_http(self):
        client = LocalModelClient("http://127.0.0.1:1", "own-model")
        gateway = self.gateway(client, deadline=time.monotonic() + 0.02)
        gateway._forward_lock.acquire()
        try:
            with patch.object(client, "_request") as upstream:
                with self.assertRaises(TimeoutError):
                    gateway.forward(sdk_payload())
                upstream.assert_not_called()
        finally:
            gateway._forward_lock.release()
        self.assertEqual(gateway.receipts[0]["status"], "rejected")

    def test_concurrent_forward_serializes_count_generation_and_restores_timeout(self):
        client = LocalModelClient("http://127.0.0.1:1", "own-model", timeout=180)
        gateway = self.gateway(client)
        first_count = threading.Event()
        second_started = threading.Event()
        release_first = threading.Event()
        second_count = threading.Event()
        order = []

        def count(_payload):
            order.append("count")
            if not first_count.is_set():
                first_count.set()
                if not release_first.wait(timeout=2):
                    raise TimeoutError("test did not release the first request")
            else:
                second_count.set()
            return 23

        def generate(_endpoint, _payload, **_kwargs):
            order.append("generate")
            return completion(), {"call_id": "synthetic-not-http", "status": "success"}

        def send_second():
            second_started.set()
            return gateway.forward(sdk_payload())

        with patch.object(gateway, "count_request", side_effect=count), \
                patch.object(client, "_request", side_effect=generate), \
                ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(gateway.forward, sdk_payload())
            self.assertTrue(first_count.wait(timeout=2))
            second = pool.submit(send_second)
            try:
                self.assertTrue(second_started.wait(timeout=2))
                self.assertFalse(second_count.wait(timeout=0.03))
            finally:
                release_first.set()
            self.assertEqual(first.result(timeout=2), completion())
            self.assertEqual(second.result(timeout=2), completion())
        self.assertEqual(order, ["count", "generate", "count", "generate"])
        self.assertEqual(client.timeout, 180)

    def test_local_chat_template_receives_full_tool_schema_and_history(self):
        payload = sdk_payload()
        tokenizer = SyntheticChatTokenizer()
        rendered = tokenizer.render(payload["messages"], payload["tools"])
        with synthetic_upstream(completion(prompt_tokens=len(rendered))) as (url, requests):
            client = LocalModelClient(url, "own-model", max_model_len=10000, chat_tokenizer=tokenizer)
            with self.gateway(client) as gateway:
                status, _ = post(gateway, payload)
            self.assertEqual(status, 200)
            self.assertEqual(tokenizer.calls, [(payload["messages"], {
                "tokenize": False, "add_generation_prompt": True,
                "enable_thinking": False, "tools": payload["tools"],
            })])
            self.assertEqual(requests[0][1], {
                "model": "own-model", "prompt": rendered, "add_special_tokens": False,
            })
            self.assertEqual(gateway.receipts[0]["prompt_tokens_checked"], len(rendered))
            self.assertEqual(requests[-1][1], payload)

    def test_wrong_path_and_oversized_body_do_not_reach_model(self):
        with synthetic_upstream() as (url, requests):
            client = LocalModelClient(url, "own-model")
            with self.gateway(client, max_request_bytes=100) as gateway:
                self.assertEqual(post(gateway, sdk_payload())[0], 400)
                self.assertEqual(post(gateway, {}, path="/v1/embeddings")[0], 400)
            self.assertEqual(requests, [])
            self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
