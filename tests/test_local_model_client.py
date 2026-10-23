"""Local HTTP stubs, not GPU/model validation."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest.local_model_client import LocalModelClient


_DEFAULT_USAGE = object()


@contextmanager
def server(*, prompt_count=19, finish="stop", text='{"桥梁名称":"合成桥"}', error_path=None,
           malformed_path=None, redirect=None, usage=True, usage_payload=_DEFAULT_USAGE):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            payload = json.loads(raw) if raw else None
            requests.append((self.path, payload, dict(self.headers)))
            if redirect:
                self.send_response(302)
                self.send_header("Location", redirect)
                self.end_headers()
                return
            if self.path == error_path:
                self.send_response(500)
                self.end_headers()
                return
            if self.path == "/tokenize":
                data = {"count": len(payload["prompt"]) if "prompt" in payload else prompt_count}
            elif self.path == "/v1/models":
                data = {"data": [{"id": "own-model"}]}
            else:
                data = {"id": "real-stub-response-id", "model": "own-model", "choices": [{
                    "message": {"content": text}, "finish_reason": finish}]}
                if usage:
                    data["usage"] = ({"prompt_tokens": prompt_count, "completion_tokens": 7,
                                      "total_tokens": prompt_count + 7}
                                     if usage_payload is _DEFAULT_USAGE else usage_payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"not json" if self.path == malformed_path else json.dumps(data).encode())

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", requests
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_template_preflight_generation_and_real_receipts():
    with server() as (url, requests):
        client = LocalModelClient(url + "/v1", "own-model", max_output_tokens=20)
        messages = [{"role": "user", "content": "合成报告"}]
        result = client.generate(messages)
        assert result["status"] == "success"
        assert json.loads(result["text"]) == {"桥梁名称": "合成桥"}
        assert [r[0] for r in requests] == ["/tokenize", "/v1/chat/completions"]
        tokenize = requests[0][1]
        assert tokenize == {"model": "own-model", "messages": messages, "add_generation_prompt": True,
                            "chat_template_kwargs": {"enable_thinking": False}}
        generate = requests[1][1]
        assert generate["messages"] == messages
        assert generate["max_tokens"] == 20
        assert generate["temperature"] == 0 and generate["stream"] is False
        assert generate["chat_template_kwargs"] == {"enable_thinking": False}
        assert result["usage"] == {"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26}
        assert result["output_sha256"] == hashlib.sha256(result["text"].encode()).hexdigest()
        assert result["call_id"] == client.calls[-1]["call_id"]
        assert result["response_id"] == "real-stub-response-id"
        assert result["seconds"] >= 0
        assert "合成报告" not in json.dumps(client.calls, ensure_ascii=False)
        assert "local-cityagent" not in json.dumps(client.calls)


def test_count_cache_and_models():
    with server() as (url, requests):
        client = LocalModelClient(url, "own-model")
        assert client.models() == ["own-model"]
        assert client.count_tokens("abc") == 3
        assert client.count_tokens("abc") == 3
        assert len(requests) == len(client.calls) == 2
        assert requests[1][1] == {"model": "own-model", "prompt": "abc", "add_special_tokens": False}
        # Check eviction without performing 1024 HTTP requests.
        for index in range(1023):
            client._token_cache[f"synthetic-key-{index}"] = index
        client.count_tokens("different")
        assert len(client._token_cache) == 1024
        assert hashlib.sha256(b"abc").hexdigest() not in client._token_cache


def test_length_is_failed_and_usage_is_not_invented():
    with server(finish="length", text='{"unfinished":', usage=False) as (url, requests):
        client = LocalModelClient(url, "own-model")
        result = client.generate([{"role": "user", "content": "whole report"}])
        assert result["status"] == "failed"
        assert result["finish_reason"] == "length"
        assert result["error"] == "output_truncated"
        assert result["usage"] is None
        assert client.calls[-1]["status"] == "failed"


def test_budget_overflow_stops_before_generation_and_equality_fits():
    with server(prompt_count=100) as (url, requests):
        messages = [{"role": "user", "content": "entire report is kept"}]
        client = LocalModelClient(url, "own-model", max_model_len=110, max_output_tokens=11)
        with pytest.raises(ValueError, match="input was not truncated"):
            client.generate(messages)
        assert [r[0] for r in requests] == ["/tokenize"]
        assert requests[0][1]["messages"] == messages
        assert client.generate(messages, max_tokens=10)["status"] == "success"


@pytest.mark.parametrize("url", [
    "https://localhost:8000", "http://example.com", "http://127.0.0.2", "http://0.0.0.0",
    "http://localhost.evil:8000", "http://user@localhost:8000", "http://localhost/?key=secret",
    "http://localhost/?", "http://localhost/#", "http://localhost/#x", "http://localhost/other",
    "http://localhost:bad", " http://localhost", "http://local\nhost", "http://localhost:8000\\@example.com",
])
def test_external_or_ambiguous_urls_rejected(url):
    with pytest.raises(ValueError):
        LocalModelClient(url, "own-model")


@pytest.mark.parametrize("url", ["http://localhost:8000", "http://127.0.0.1:8000/v1/", "http://[::1]:8000"])
def test_loopback_forms_accepted_without_network(url):
    LocalModelClient(url, "own-model")


def test_redirect_is_never_followed():
    with server() as (destination, destination_requests):
        with server(redirect=destination + "/v1/models") as (url, requests):
            client = LocalModelClient(url, "own-model")
            with pytest.raises(RuntimeError, match="HTTP 302"):
                client.models()
            assert destination_requests == []
            assert len(requests) == len(client.calls) == 1
            assert client.calls[0]["status"] == "failed"


@pytest.mark.parametrize("failure", ["http", "json"])
def test_http_and_json_failures_have_failed_receipts(failure):
    options = {"error_path" if failure == "http" else "malformed_path": "/v1/chat/completions"}
    with server(**options) as (url, requests):
        client = LocalModelClient(url, "own-model")
        with pytest.raises(RuntimeError):
            client.generate([{"role": "user", "content": "report"}])
        assert len(client.calls) == 2
        assert client.calls[-1]["kind"] == "generation"
        assert client.calls[-1]["status"] == "failed"


def test_environment_proxy_is_not_used():
    with server() as (url, requests):
        with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1", "no_proxy": "", "NO_PROXY": ""}):
            assert LocalModelClient(url, "own-model").models() == ["own-model"]
        assert len(requests) == 1


@pytest.mark.parametrize("kwargs", [{"timeout": 0}, {"timeout": float("nan")}, {"max_model_len": True}, {"max_output_tokens": 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LocalModelClient("http://localhost:8000", "own-model", **kwargs)


class FakeChatTokenizer:
    def __init__(self, *, extra_tokens=0):
        self.extra_tokens = extra_tokens
        self.template_calls = []
        self.encode_calls = []
        self.rendered = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        self.rendered = "".join(f'<{row["role"]}>{row["content"]}</{row["role"]}>' for row in messages)
        self.rendered += "<assistant><think></think>"
        return self.rendered

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        # The HTTP stub deliberately uses one token per character.
        return list(range(len(text) + self.extra_tokens))


def test_injected_chat_tokenizer_renders_full_prompt_and_verifies_server_count():
    tokenizer = FakeChatTokenizer()
    messages = [{"role": "system", "content": "只按原文抽取"},
                {"role": "user", "content": "完整报告：桥名与病害\n末尾内容不能截断"}]
    expected_count = len(FakeChatTokenizer().apply_chat_template(messages))
    with server(prompt_count=expected_count) as (url, requests):
        client = LocalModelClient(url, "own-model", max_output_tokens=20, chat_tokenizer=tokenizer)
        result = client.generate(messages)
        assert tokenizer.template_calls == [(messages, {
            "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
        })]
        assert tokenizer.encode_calls == [(tokenizer.rendered, {"add_special_tokens": False})]
        assert [row[0] for row in requests] == ["/tokenize", "/v1/chat/completions"]
        assert requests[0][1] == {
            "model": "own-model", "prompt": tokenizer.rendered, "add_special_tokens": False,
        }
        assert all(row["content"] in requests[0][1]["prompt"] for row in messages)
        assert requests[1][1] == {
            "model": "own-model", "messages": messages, "max_tokens": 20,
            "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": False},
        }
        assert len(requests) == len(client.calls) == 2
        receipt = client.calls[0]
        assert receipt["status"] == "success" and receipt["token_counts_match"] is True
        assert receipt["local_token_count"] == receipt["token_count"] == len(tokenizer.rendered)
        assert receipt["rendered_prompt_sha256"] == hashlib.sha256(tokenizer.rendered.encode()).hexdigest()
        assert result["prompt_tokens_checked"] == len(tokenizer.rendered)
        assert result["chat_tokenization_mode"] == receipt["chat_tokenization_mode"] == "local_template_server_prompt"
        assert result["usage"] == {"prompt_tokens": expected_count, "completion_tokens": 7,
                                    "total_tokens": expected_count + 7}
        assert result["prompt_tokens_usage_match"] is True
        assert "完整报告" not in json.dumps(client.calls, ensure_ascii=False)


def test_injected_tokenizer_count_mismatch_fails_closed_with_one_real_receipt():
    tokenizer = FakeChatTokenizer(extra_tokens=1)
    with server() as (url, requests):
        client = LocalModelClient(url, "own-model", chat_tokenizer=tokenizer)
        with pytest.raises(RuntimeError, match="Local and server chat token counts differ"):
            client.generate([{"role": "user", "content": "must never generate after mismatch"}])
        assert [row[0] for row in requests] == ["/tokenize"]
        assert len(client.calls) == 1
        receipt = client.calls[0]
        assert receipt["kind"] == "tokenize" and receipt["status"] == "failed"
        assert receipt["token_counts_match"] is False
        assert receipt["local_token_count"] == receipt["token_count"] + 1
        assert receipt["http_status"] == 200


def test_injected_tokenizer_full_budget_refuses_overflow_and_allows_exact_fit():
    tokenizer = FakeChatTokenizer()
    messages = [{"role": "user", "content": "all source text " * 50 + "最后一段"}]
    rendered = tokenizer.apply_chat_template(messages)
    tokenizer.template_calls.clear()
    with server(prompt_count=len(rendered)) as (url, requests):
        client = LocalModelClient(url, "own-model", chat_tokenizer=tokenizer,
                                  max_model_len=len(rendered) + 10, max_output_tokens=11)
        with pytest.raises(ValueError, match="input was not truncated"):
            client.generate(messages)
        assert [row[0] for row in requests] == ["/tokenize"]
        assert requests[0][1]["prompt"] == rendered
        assert len(client.calls) == 1 and client.calls[0]["status"] == "success"
        assert client.generate(messages, max_tokens=10)["status"] == "success"
        assert [row[0] for row in requests] == ["/tokenize", "/tokenize", "/v1/chat/completions"]
        assert len(client.calls) == 3
        assert requests[-1][1]["messages"] == messages


@pytest.mark.parametrize("failure", ["render", "encode"])
def test_injected_tokenizer_local_failure_does_not_invent_http_receipts(failure):
    class BrokenTokenizer(FakeChatTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if failure == "render":
                raise ValueError("synthetic template failure")
            return super().apply_chat_template(messages, **kwargs)

        def encode(self, text, **kwargs):
            raise ValueError("synthetic encode failure")

    with server() as (url, requests):
        client = LocalModelClient(url, "own-model", chat_tokenizer=BrokenTokenizer())
        with pytest.raises(ValueError, match="synthetic"):
            client.generate([{"role": "user", "content": "source"}])
        assert requests == client.calls == []


def test_injected_tokenizer_does_not_change_plain_text_token_counting():
    tokenizer = FakeChatTokenizer()
    with server() as (url, requests):
        client = LocalModelClient(url, "own-model", chat_tokenizer=tokenizer)
        assert client.count_tokens("evidence text") == len("evidence text")
        assert client.count_tokens("evidence text") == len("evidence text")
        assert tokenizer.template_calls == tokenizer.encode_calls == []
        assert len(requests) == len(client.calls) == 1
        assert requests[0][1] == {"model": "own-model", "prompt": "evidence text", "add_special_tokens": False}


@pytest.mark.parametrize("tokenizer", [object(), {"encode": lambda value: []}])
def test_injected_tokenizer_requires_both_standard_methods(tokenizer):
    with pytest.raises(ValueError, match="apply_chat_template and encode"):
        LocalModelClient("http://localhost:8000", "own-model", chat_tokenizer=tokenizer)


@pytest.mark.parametrize("injected", [False, True])
def test_public_chat_count_issues_only_tokenization_request(injected):
    tokenizer = FakeChatTokenizer() if injected else None
    messages = [{"role": "user", "content": "CPU-only chat tokenization preflight"}]
    with server(prompt_count=37) as (url, requests):
        client = LocalModelClient(url, "own-model", chat_tokenizer=tokenizer)
        count = client.count_chat_tokens(messages)
        assert [row[0] for row in requests] == ["/tokenize"]
        assert len(client.calls) == 1 and client.calls[0]["kind"] == "tokenize"
        if injected:
            assert count == len(tokenizer.rendered)
            assert "messages" not in requests[0][1]
        else:
            assert count == 37
            assert requests[0][1] == {"model": "own-model", "messages": messages,
                                       "add_generation_prompt": True,
                                       "chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("usage_payload", [None, [], "invalid", {}, {"completion_tokens": 7},
                                          {"prompt_tokens": None}, {"prompt_tokens": True},
                                          {"prompt_tokens": False}, {"prompt_tokens": -1},
                                          {"prompt_tokens": 1.5}, {"prompt_tokens": "20"}])
def test_injected_tokenizer_rejects_invalid_generation_usage_and_preserves_receipt(usage_payload):
    output = "synthetic model output retained only as a hash"
    with server(usage_payload=usage_payload, text=output) as (url, requests):
        client = LocalModelClient(url, "own-model", chat_tokenizer=FakeChatTokenizer())
        with pytest.raises(RuntimeError, match=r"usage\.prompt_tokens must be a nonnegative integer"):
            client.generate([{"role": "user", "content": "full source"}])
        assert [row[0] for row in requests] == ["/tokenize", "/v1/chat/completions"]
        assert len(client.calls) == 2 and client.calls[0]["status"] == "success"
        receipt = client.calls[-1]
        assert receipt["kind"] == "generation" and receipt["status"] == "failed"
        assert receipt["usage"] == usage_payload
        assert receipt["finish_reason"] == "stop"
        assert receipt["output_sha256"] == hashlib.sha256(output.encode()).hexdigest()
        assert receipt["http_status"] == 200 and receipt["response_id"] == "real-stub-response-id"
        assert receipt["prompt_tokens_usage_match"] is False
        assert output not in json.dumps(client.calls)


def test_injected_tokenizer_rejects_missing_generation_usage_without_inventing_it():
    with server(usage=False, text="complete response") as (url, requests):
        client = LocalModelClient(url, "own-model", chat_tokenizer=FakeChatTokenizer())
        with pytest.raises(RuntimeError, match=r"usage\.prompt_tokens"):
            client.generate([{"role": "user", "content": "full source"}])
        assert len(requests) == len(client.calls) == 2
        receipt = client.calls[-1]
        assert receipt["status"] == "failed" and receipt["usage"] is None
        assert receipt["finish_reason"] == "stop"
        assert receipt["output_sha256"] == hashlib.sha256(b"complete response").hexdigest()


@pytest.mark.parametrize("finish", ["stop", "length"])
@pytest.mark.parametrize("reported_tokens", [0, 10000])
def test_injected_tokenizer_rejects_generation_count_mismatch_and_preserves_original_usage(finish, reported_tokens):
    usage = {"prompt_tokens": reported_tokens, "completion_tokens": 3,
             "total_tokens": reported_tokens + 3, "server_detail": "retained"}
    with server(usage_payload=usage, finish=finish, text="model reply") as (url, requests):
        tokenizer = FakeChatTokenizer()
        client = LocalModelClient(url, "own-model", chat_tokenizer=tokenizer)
        with pytest.raises(RuntimeError, match=r"usage\.prompt_tokens differs from checked prompt count"):
            client.generate([{"role": "user", "content": "full source"}])
        receipt = client.calls[-1]
        assert len(requests) == len(client.calls) == 2
        assert receipt["status"] == "failed" and receipt["usage"] == usage
        assert receipt["prompt_tokens_checked"] == len(tokenizer.rendered)
        assert receipt["finish_reason"] == finish
        assert receipt["output_sha256"] == hashlib.sha256(b"model reply").hexdigest()
        assert receipt["prompt_tokens_usage_match"] is False


@pytest.mark.parametrize("usage_payload", [None, {"prompt_tokens": True}, {"prompt_tokens": 10000}])
def test_default_vllm_path_preserves_usage_without_new_validation(usage_payload):
    with server(usage_payload=usage_payload) as (url, requests):
        client = LocalModelClient(url, "own-model")
        result = client.generate([{"role": "user", "content": "full source"}])
        assert result["status"] == "success" and result["usage"] == usage_payload
        assert "prompt_tokens_usage_match" not in result
        assert len(requests) == len(client.calls) == 2
