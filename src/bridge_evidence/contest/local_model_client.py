"""Audited, loopback-only HTTP access to the contest's own vLLM model.

This module deliberately uses only the standard library so it can also run in
the isolated TongAgents environment. It never truncates prompts or follows
redirects. ``calls`` contains one receipt per actual HTTP request; cache hits
and rejected context budgets are not presented as model calls.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True, allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_temperature(value: Any) -> int | float:
    if type(value) not in (int, float) or not 0 <= value <= 2 or not math.isfinite(value):
        raise ValueError("temperature must be a finite number between 0 and 2")
    return value


def text_generation_parameters(*, max_tokens: int, temperature: int | float,
                               enable_thinking: bool) -> dict:
    if type(enable_thinking) is not bool:
        raise ValueError("enable_thinking must be a boolean")
    return {"max_tokens": _positive_int("max_tokens", max_tokens),
            "temperature": validate_temperature(temperature),
            "chat_template_kwargs": {"enable_thinking": enable_thinking}}


def validate_text_request_policy(payload: dict, *, temperature: int | float,
                                 enable_thinking: bool) -> None:
    chat_kwargs = payload.get("chat_template_kwargs")
    enabled = chat_kwargs.get("enable_thinking") if isinstance(chat_kwargs, dict) else None
    if type(enabled) is not bool or type(enable_thinking) is not bool or enabled is not enable_thinking:
        raise ValueError("Text request must explicitly match the own-model client's boolean thinking setting")
    actual = validate_temperature(payload.get("temperature"))
    if actual != validate_temperature(temperature):
        raise ValueError("Text request temperature must match the own-model client's pinned setting")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Reject even loopback redirects: the configured origin is the only
        # endpoint authorized to receive document contents and credentials.
        raise HTTPError(req.full_url, code, "Redirects are disabled", headers, fp)


class LocalModelClient:
    """Call a local vLLM server and retain content hashes, not document text.

    ``base_url`` accepts a root URL or its ``/v1`` spelling. ``models()``
    returns the served model IDs. ``generate()`` returns unsuccessful length
    completions with ``status='failed'``; transport/schema failures raise
    ``RuntimeError`` after recording the failed request. Context overflow
    raises ``ValueError`` before any generation request.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "local-cityagent",
                 max_model_len: int = 65536, timeout: float = 600,
                 max_output_tokens: int = 4096,
                 receipt_sink: Callable[[dict], None] | None = None, *,
                 chat_tokenizer: Any | None = None,
                 enable_thinking: bool = False, temperature: int | float = 0,
                 dynamic_output_budget: bool = False):
        if not isinstance(base_url, str) or base_url != base_url.strip():
            raise ValueError("base_url must be an HTTP loopback URL")
        try:
            parsed = urlsplit(base_url)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Invalid local model URL") from exc
        if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or "?" in base_url or "#" in base_url
                or parsed.path.rstrip("/") not in {"", "/v1"}
                or any(ord(character) < 33 for character in base_url)):
            raise ValueError("Only HTTP localhost, 127.0.0.1 or [::1] without credentials/query is allowed")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty served model ID")
        if not isinstance(api_key, str) or "\n" in api_key or "\r" in api_key:
            raise ValueError("api_key must be a single-line string")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if type(enable_thinking) is not bool:
            raise ValueError("enable_thinking must be a boolean")
        if chat_tokenizer is not None and any(
                not callable(getattr(chat_tokenizer, method, None))
                for method in ("apply_chat_template", "encode")):
            raise ValueError("chat_tokenizer must provide apply_chat_template and encode")
        host = "[::1]" if parsed.hostname == "::1" else parsed.hostname
        self.base_url = f"http://{host}" + (f":{port}" if port is not None else "")
        self.model = model
        self.api_key = api_key
        self.max_model_len = _positive_int("max_model_len", max_model_len)
        self.max_output_tokens = _positive_int("max_output_tokens", max_output_tokens)
        self.timeout = timeout
        self.reliable_generation_policy = None
        self.reliable_generation_deadline = None
        self.reliable_generation_on_cancel = None
        self.reliable_generation_capture_directory = None
        self.reliable_generation_diagnostic_token_ids = False
        self.reliable_generation_raw_guard_factory = None
        self.reliable_generation_native_repetition_detection = False
        self.reliable_thinking_budget = None
        self.reliable_thinking_budget_selector = None
        self.chat_tokenizer = chat_tokenizer
        self.enable_thinking = enable_thinking
        self.temperature = validate_temperature(temperature)
        if type(dynamic_output_budget) is not bool:
            raise ValueError("dynamic_output_budget must be a boolean")
        if dynamic_output_budget and self.max_output_tokens != self.max_model_len:
            raise ValueError("Dynamic output ceiling must equal the complete context")
        self.dynamic_output_budget = dynamic_output_budget
        # Independent of context capacity. A workflow can opt in for short
        # tool decisions without changing child/summary clients or input.
        self.generation_output_cap = None
        self._dynamic_budget_failure = None
        self.calls: list[dict[str, Any]] = []
        self.receipt_sink = receipt_sink
        self.request_context: dict[str, str] = {}
        self.batch_admission = None
        self.model_catalog: list[dict[str, Any]] = []
        self._token_cache: OrderedDict[str, int] = OrderedDict()
        # Environment HTTP proxies must never receive report content.
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def enable_reliable_generation(self, policy, *, deadline=None, on_cancel=None, thinking_budget=None,
                                   thinking_budget_selector=None, capture_directory=None,
                                   diagnostic_token_ids=False, raw_guard_tokenizer_path=None,
                                   raw_guard_tokenizer_sha256=None, native_repetition_detection=False):
        """Enable observed SSE transport without changing the SDK response shape."""
        from .raw_sse_capture import capture_directory as validated_capture_directory
        capture_path = validated_capture_directory(capture_directory)
        if type(diagnostic_token_ids) is not bool:
            raise ValueError("diagnostic_token_ids must be a boolean")
        if diagnostic_token_ids and capture_path is None:
            raise ValueError("diagnostic_token_ids requires an explicit capture_directory")
        if type(native_repetition_detection) is not bool:
            raise ValueError("native_repetition_detection must be a boolean")
        if (raw_guard_tokenizer_path is None) != (raw_guard_tokenizer_sha256 is None):
            raise ValueError("Raw guard requires both tokenizer path and pinned SHA256")
        raw_guard_factory = None
        if raw_guard_tokenizer_path is not None:
            if capture_path is None:
                raise ValueError("Raw generation guard requires an explicit capture_directory")
            from .raw_generation_guard import RawGenerationGuardFactory
            raw_guard_factory = RawGenerationGuardFactory(raw_guard_tokenizer_path, raw_guard_tokenizer_sha256)
        if thinking_budget is not None and (type(thinking_budget) is not int or thinking_budget <= 0):
            raise ValueError("thinking_budget must be a verified positive integer")
        if thinking_budget_selector is not None and not callable(thinking_budget_selector):
            raise ValueError("thinking_budget_selector must be callable")
        if thinking_budget is not None and thinking_budget_selector is not None:
            raise ValueError("Fixed and selected thinking budgets cannot both be configured")
        if not self.enable_thinking and (thinking_budget is not None or thinking_budget_selector is not None):
            raise ValueError("Disabled thinking cannot carry a thinking budget")
        self.reliable_generation_policy = policy
        self.reliable_generation_deadline = deadline
        self.reliable_generation_on_cancel = on_cancel
        self.reliable_generation_capture_directory = capture_path
        self.reliable_generation_diagnostic_token_ids = diagnostic_token_ids
        self.reliable_generation_raw_guard_factory = raw_guard_factory
        self.reliable_generation_native_repetition_detection = native_repetition_detection
        self.reliable_thinking_budget = thinking_budget
        self.reliable_thinking_budget_selector = thinking_budget_selector

    def reliable_payload(self, payload: dict) -> dict:
        from copy import deepcopy
        value = deepcopy(payload)
        if self.reliable_generation_policy is None:
            return value
        if self.temperature != 1.0 or type(self.enable_thinking) is not bool:
            raise ValueError("Reliable Qwen profile requires explicit temperature=1 and boolean thinking mode")
        value.update(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                     presence_penalty=1.5, repetition_penalty=1.1)
        if self.reliable_generation_diagnostic_token_ids or self.reliable_generation_raw_guard_factory is not None:
            # Original generated IDs expose text withheld by the server's tool
            # parser. The observer never rewrites parsed SDK tool arguments.
            value["return_token_ids"] = True
        if self.reliable_generation_native_repetition_detection:
            from .raw_generation_guard import NATIVE_REPETITION_POLICY
            value["repetition_detection"] = dict(NATIVE_REPETITION_POLICY)
        if not self.enable_thinking:
            if ("thinking_token_budget" in value or self.reliable_thinking_budget is not None
                    or self.reliable_thinking_budget_selector is not None):
                raise ValueError("Disabled thinking cannot carry a thinking budget")
            return value
        selector = self.reliable_thinking_budget_selector
        budget = None
        if selector is not None:
            budget = selector(deepcopy(value))
            if type(budget) is not int or not 1 <= budget <= 8192:
                raise ValueError("Selected thinking budget must be an integer from 1 to the verified ceiling 8192")
        elif self.reliable_thinking_budget is not None:
            budget = self.reliable_thinking_budget
            # File inspection ordinarily only needs a short next operation;
            # extraction and conflict handling retain a larger reasoning budget.
            for message in reversed(value.get("messages", [])):
                if message.get("role") == "assistant" and message.get("tool_calls"):
                    names = [c.get("function", {}).get("name", "") for c in message["tool_calls"]]
                    if any(name.endswith("inspect_delivery") for name in names):
                        budget = min(budget, 4096)
                    break
        if budget is not None:
            maximum = value.get("max_tokens")
            if type(maximum) is int and maximum > 0:
                budget = min(budget, max(1, maximum - 128))
            value["thinking_token_budget"] = budget
        return value

    def configure_batch_admission(self, config, *, pool, agent_id, deadline):
        from .document_batch_client import BatchClientAdmission
        self.batch_admission = BatchClientAdmission(config, pool=pool, agent_id=agent_id, deadline=deadline)

    def generation_slot(self, *, deadline=None, is_valid=None):
        return (self.batch_admission.slot(deadline=deadline, is_valid=is_valid)
                if self.batch_admission is not None else nullcontext())

    def _request(self, endpoint: str, payload: dict | None, *, kind: str) -> tuple[dict, dict]:
        if endpoint != "/v1/chat/completions" or self.batch_admission is None:
            return self._request_unadmitted(endpoint, payload, kind=kind)
        before = len(self.calls)
        try:
            with self.generation_slot():
                # Reliable streaming marks this at its actual opener boundary,
                # after raw-guard and payload preparation have succeeded.
                if self.reliable_generation_policy is None:
                    self.batch_admission.transport_started()
                try:
                    return self._request_unadmitted(endpoint, payload, kind=kind)
                finally:
                    current = [row for row in self.calls[before:] if row.get("endpoint") == endpoint]
                    if len(current) == 1:
                        self.batch_admission.observe(current[0])
        finally:
            # The lease settles on context exit, including failed streams. Its
            # final disposition must reach the same durable call record.
            for receipt in self.calls[before:]:
                if receipt.get("endpoint") == endpoint:
                    self._emit_receipt(receipt)

    def _request_unadmitted(self, endpoint: str, payload: dict | None, *, kind: str) -> tuple[dict, dict]:
        if endpoint == "/v1/chat/completions" and self.reliable_generation_policy is not None:
            from .reliable_generation import stream_chat, ReliableGenerationError
            receipt = None
            capture = None

            def reserve_stream_call(event):
                nonlocal receipt
                # Keep the actual receipt object so BaseException interruption
                # still reaches the journal with the same reserved identity.
                receipt = event
                if self.request_context.get("reservation_id"):
                    receipt["reservation_id"] = self.request_context["reservation_id"]
                self._emit_receipt(receipt)

            if self.reliable_generation_capture_directory is not None:
                from .raw_sse_capture import RawSSECapture, MAX_CAPTURE_BYTES
                configured = self.reliable_generation_policy
                limit = (configured.get("max_response_bytes", MAX_CAPTURE_BYTES) if type(configured) is dict
                         else getattr(configured, "max_response_bytes", MAX_CAPTURE_BYTES))
                capture = RawSSECapture(self.reliable_generation_capture_directory,
                    max_bytes=limit if type(limit) is int and limit > 0 else MAX_CAPTURE_BYTES)
            deadline = min(time.monotonic() + self.timeout,
                           self.reliable_generation_deadline
                           if self.reliable_generation_deadline is not None else float("inf"))
            if self.batch_admission is not None:
                deadline = min(deadline, self.batch_admission.active["deadline"])
            try:
                policy = self.reliable_generation_policy
                tool_limit = getattr(self, "reliable_tool_content_limit", None)
                if tool_limit is not None and payload.get("tools") and payload.get("tool_choice") != "none":
                    from dataclasses import replace
                    from .reliable_generation import StreamPolicy
                    policy = StreamPolicy(**policy) if type(policy) is dict else policy
                    policy = replace(policy, max_content_chars=tool_limit)
                result, receipt = stream_chat(self, self.reliable_payload(payload), deadline=deadline,
                    policy=policy, on_cancel=self.reliable_generation_on_cancel,
                    on_chunk=capture.observe if capture is not None else None,
                    on_dispatch=reserve_stream_call)
                return result, receipt
            except ReliableGenerationError as exc:
                receipt = exc.receipt
                raise
            finally:
                if receipt is not None:
                    receipt.update(kind=kind, endpoint=endpoint, model=self.model)
                    receipt.setdefault("call_id", str(uuid4()))
                    if capture is not None:
                        try:
                            receipt["raw_sse_capture"] = capture.finalize(receipt)
                        except Exception as failure:
                            # Evidence errors must never replace the transport
                            # response/error or alter its cancellation/usage.
                            receipt["raw_sse_capture"] = {"schema": "cityagent-raw-sse-capture-v1",
                                "scope": "consumer_observed_bytes", "status": "failed", "complete": False,
                                "path": None, "sha256": None, "bytes": None,
                                "errors": [{"operation": "finalize", "type": type(failure).__name__}]}
                    self.calls.append(receipt)
                    self._emit_receipt(receipt)

        body = _json_bytes(payload) if payload is not None else None
        receipt: dict[str, Any] = {
            "call_id": str(uuid4()), "kind": kind, "endpoint": endpoint,
            "method": "POST" if body is not None else "GET", "model": self.model,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input_sha256": _sha(body or b""), "output_sha256": None,
            "seconds": 0.0, "status": "running", "http_status": None,
            "usage": None, "finish_reason": None,
        }
        if self.request_context.get("reservation_id"):
            receipt["reservation_id"] = self.request_context["reservation_id"]
        request = Request(self.base_url + endpoint, data=body, headers={
            "Content-Type": "application/json", "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }, method=receipt["method"])
        start = time.monotonic()
        # The caller's synchronous journal reserves this identity before send.
        # A sink failure therefore cannot produce an unrecorded HTTP request.
        self._emit_receipt(receipt)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                receipt["http_status"] = response.status
                raw = response.read()
            receipt["response_sha256"] = _sha(raw)
            receipt["output_sha256"] = receipt["response_sha256"]
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("Response must be a JSON object")
            receipt["status"] = "success"
            return result, receipt
        except Exception as exc:
            receipt["status"] = "failed"
            if isinstance(exc, HTTPError):
                receipt["http_status"] = exc.code
                # Error bodies may echo source text. Keep only a bounded hash
                # and structural decoder coordinates, never the raw message.
                try:
                    error_body = exc.read(16385)
                    receipt["http_error_body_sha256"] = _sha(error_body)
                    receipt["http_error_body_bytes_read"] = len(error_body)
                    receipt["http_error_body_truncated"] = len(error_body) > 16384
                    if len(error_body) <= 16384:
                        parsed = json.loads(error_body)
                        detail = parsed.get("error", parsed) if isinstance(parsed, dict) else {}
                        if isinstance(detail, dict):
                            diagnostic = {}
                            if detail.get("type") in {"BadRequestError", "ValidationError", "JSONDecodeError"}:
                                diagnostic["type"] = detail["type"]
                            if type(detail.get("code")) is int and 100 <= detail["code"] <= 599:
                                diagnostic["code"] = detail["code"]
                            location = re.search(r"line (\d+) column (\d+) \(char (\d+)\)", str(detail.get("message", "")))
                            if location is not None:
                                diagnostic["json_location"] = dict(zip(("line", "column", "character_offset"),
                                                                       map(int, location.groups())))
                            if diagnostic:
                                receipt["http_error_diagnostic"] = diagnostic
                except Exception:
                    receipt["http_error_diagnostic_unavailable"] = True
                exc.close()
            receipt["error"] = f"{type(exc).__name__}: local HTTP request failed"
            raise RuntimeError(f"Local model request {endpoint} failed ({type(exc).__name__}, "
                               f"HTTP {receipt['http_status']})") from exc
        except BaseException:
            receipt.update(status="interrupted", error="request_outcome_unknown")
            raise
        finally:
            receipt["seconds"] = time.monotonic() - start
            self.calls.append(receipt)
            self._emit_receipt(receipt)

    def _emit_receipt(self, receipt: dict) -> None:
        if self.receipt_sink is not None:
            self.receipt_sink(json.loads(_json_bytes(receipt)))

    def _bad_response(self, receipt: dict, message: str) -> RuntimeError:
        receipt["status"] = "failed"
        receipt["error"] = message
        self._emit_receipt(receipt)
        return RuntimeError(message)

    def _tokenize(self, payload: dict, *, local_token_count: int | None = None,
                  rendered_prompt_sha256: str | None = None) -> int:
        result, receipt = self._request("/tokenize", payload, kind="tokenize")
        if local_token_count is not None:
            receipt.update(chat_tokenization_mode="local_template_server_prompt",
                           local_token_count=local_token_count,
                           rendered_prompt_sha256=rendered_prompt_sha256)
        count = result.get("count")
        if type(count) is not int or count < 0:
            raise self._bad_response(receipt, "Tokenizer response must contain a nonnegative integer count")
        receipt["token_count"] = count
        if local_token_count is not None:
            receipt["token_counts_match"] = count == local_token_count
            if count != local_token_count:
                raise self._bad_response(receipt, "Local and server chat token counts differ "
                                         f"(local={local_token_count}, server={count})")
        self._emit_receipt(receipt)
        return count

    def count_tokens(self, text: str) -> int:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        key = _sha(text.encode("utf-8"))
        if key in self._token_cache:
            self._token_cache.move_to_end(key)
            return self._token_cache[key]
        count = self._tokenize({"model": self.model, "prompt": text, "add_special_tokens": False})
        self._token_cache[key] = count
        if len(self._token_cache) > 1024:
            self._token_cache.popitem(last=False)
        return count

    def models(self) -> list[str]:
        result, receipt = self._request("/v1/models", None, kind="models")
        data = result.get("data")
        if (not isinstance(data, list) or any(not isinstance(item, dict)
                or not isinstance(item.get("id"), str) or not item["id"] for item in data)):
            raise self._bad_response(receipt, "Model list must contain objects with nonempty string IDs")
        self.model_catalog = data
        return [item["id"] for item in data]

    def count_chat_tokens(self, messages: list[dict]) -> int:
        """Count the full chat prompt without issuing a generation request.

        An injected CPU tokenizer renders the template and must agree with the
        server's count for that exact string. No prompt is truncated or cached.
        """
        if self.dynamic_output_budget:
            return self.count_complete_request({"model": self.model, "messages": messages, "stream": False,
                **text_generation_parameters(max_tokens=self.max_output_tokens, temperature=self.temperature,
                                             enable_thinking=self.enable_thinking)})
        if not isinstance(messages, list) or not messages or any(not isinstance(item, dict) for item in messages):
            raise ValueError("messages must be a nonempty list of message objects")
        if self.chat_tokenizer is None:
            prompt_tokens = self._tokenize({
                "model": self.model, "messages": messages, "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            })
        else:
            # Some compatible servers tokenize raw prompts but do not apply
            # chat templates in /tokenize. Render the full template on CPU,
            # then require agreement with the server before allowing generation.
            rendered = self.chat_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
            if not isinstance(rendered, str) or not rendered:
                raise ValueError("Chat template must render a nonempty string")
            token_ids = self.chat_tokenizer.encode(rendered, add_special_tokens=False)
            if (not isinstance(token_ids, (list, tuple))
                    or any(type(token) is not int or token < 0 for token in token_ids)):
                raise ValueError("Chat tokenizer must encode a sequence of nonnegative integer token IDs")
            prompt_tokens = self._tokenize({
                "model": self.model, "prompt": rendered, "add_special_tokens": False,
            }, local_token_count=len(token_ids), rendered_prompt_sha256=_sha(rendered.encode("utf-8")))
        return prompt_tokens

    def count_complete_request(self, payload: dict) -> int:
        """Fresh server rendering of exact messages/tools/template/history.

        The backend returns both count and its actual context limit. Optional
        CPU parity checks the same rendered token IDs; it never replaces the
        backend count. No complete-request count uses the raw-text cache.
        """
        from .document_context_budget import tokenize_chat_payload, request_input_hash
        if payload.get("model") != self.model:
            raise ValueError("Complete request must use the pinned own model")
        validate_text_request_policy(payload, temperature=self.temperature, enable_thinking=self.enable_thinking)
        tokenize = tokenize_chat_payload(payload)
        result, receipt = self._request("/tokenize", tokenize, kind="tokenize")
        count = result.get("count")
        receipt.update(complete_request_input_sha256=request_input_hash(payload),
                       chat_tokenization_mode="installed_server_complete_chat_template")
        if (type(count) is not int or count < 0 or type(result.get("max_model_len")) is not int
                or result["max_model_len"] != self.max_model_len):
            raise self._bad_response(receipt, "Complete request tokenizer count/context differs from pinned policy")
        receipt["token_count"] = count
        receipt["server_max_model_len"] = result["max_model_len"]
        if self.chat_tokenizer is not None:
            args = dict(tokenize.get("chat_template_kwargs") or {})
            args.update(tokenize=False, add_generation_prompt=tokenize["add_generation_prompt"], tools=tokenize.get("tools"))
            if tokenize["continue_final_message"]:
                args["continue_final_message"] = True
            if tokenize.get("chat_template") is not None:
                args["chat_template"] = tokenize["chat_template"]
            rendered = self.chat_tokenizer.apply_chat_template(tokenize["messages"], **args)
            ids = self.chat_tokenizer.encode(rendered, add_special_tokens=tokenize["add_special_tokens"])
            receipt.update(local_token_count=len(ids), token_counts_match=len(ids) == count,
                           rendered_prompt_sha256=_sha(rendered.encode()))
            # The installed tokenize endpoint returns complete token IDs, so
            # matching length alone is not presented as template parity.
            if result.get("tokens") != list(ids):
                raise self._bad_response(receipt, "Local model template/token IDs differ from installed server")
        self._emit_receipt(receipt)
        return count

    def prepare_generation_request(self, payload: dict) -> tuple[dict, dict]:
        from copy import deepcopy
        from .document_context_budget import POLICY, CAPPED_POLICY, remaining_output, request_input_hash
        if not self.dynamic_output_budget:
            raise ValueError("Dynamic allocation must be explicitly enabled")
        if self._dynamic_budget_failure is not None:
            raise RuntimeError("Prior dynamic context admission failed; no compression/retry permitted")
        # Allocation preserves the already constructed SDK request. The
        # gateway applies the configured sampling profile before its fresh
        # admission and upstream transport; direct generate() profiles first.
        prepared = deepcopy(payload)
        try:
            count = self.count_complete_request(prepared)
            maximum = remaining_output(self.max_model_len, count, self.generation_output_cap)
        except Exception as exc:
            self._dynamic_budget_failure = type(exc).__name__
            raise
        prepared.pop("max_completion_tokens", None)
        prepared["max_tokens"] = maximum
        return prepared, {"policy": CAPPED_POLICY if self.generation_output_cap is not None else POLICY,
                          "output_cap": self.generation_output_cap, "max_model_len": self.max_model_len,
                          "actual_input_tokens": count, "max_tokens": maximum,
                          "complete_request_input_sha256": request_input_hash(prepared)}

    def generate(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        payload = {"model": self.model, "messages": messages, "stream": False,
                   **text_generation_parameters(max_tokens=self.max_output_tokens, temperature=self.temperature,
                                                enable_thinking=self.enable_thinking)}
        payload = self.reliable_payload(payload)
        if self.dynamic_output_budget:
            payload, allocation = self.prepare_generation_request(payload)
            output_budget, prompt_tokens = payload["max_tokens"], allocation["actual_input_tokens"]
            if max_tokens is not None and max_tokens != output_budget:
                raise ValueError("Requested output differs from exact remaining-context allocation")
        else:
            output_budget = _positive_int("max_tokens", self.max_output_tokens if max_tokens is None else max_tokens)
            prompt_tokens = self.count_chat_tokens(messages)
            payload["max_tokens"] = output_budget
        if prompt_tokens + output_budget > self.max_model_len:
            raise ValueError(f"Full prompt ({prompt_tokens}) + output budget ({output_budget}) "
                             f"exceeds model context ({self.max_model_len}); input was not truncated")
        if self.dynamic_output_budget:
            try:
                checked_again = self.count_complete_request(payload)
                from .document_context_budget import remaining_output
                if checked_again != prompt_tokens or payload["max_tokens"] != remaining_output(self.max_model_len, checked_again, self.generation_output_cap):
                    raise ValueError("Complete direct request changed after dynamic allocation")
            except Exception as exc:
                self._dynamic_budget_failure = type(exc).__name__
                raise
        result, receipt = self._request("/v1/chat/completions", payload, kind="generation")
        if self.dynamic_output_budget:
            receipt["output_allocation"] = allocation
        receipt["prompt_tokens_checked"] = prompt_tokens
        if self.chat_tokenizer is not None:
            receipt["chat_tokenization_mode"] = "local_template_server_prompt"
        receipt["max_tokens"] = output_budget
        receipt["response_id"] = result.get("id")
        receipt["response_model"] = result.get("model")
        receipt["usage"] = result.get("usage")  # Preserve server usage; never estimate it.
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise self._bad_response(receipt, "Generation response must contain exactly one choice")
        choice = choices[0]
        receipt["finish_reason"] = choice.get("finish_reason")
        message = choice.get("message")
        if isinstance(message, dict):
            receipt["final_content_type"] = ("missing" if "content" not in message
                                             else "null" if message["content"] is None
                                             else type(message["content"]).__name__)
            receipt["reasoning_channels"] = {
                key: ({"sha256": _sha(value.encode("utf-8")), "characters": len(value)}
                      if isinstance(value, str) else {"type": type(value).__name__})
                for key in ("reasoning", "reasoning_content") if key in message
                for value in (message[key],)
            }
        if receipt["finish_reason"] == "length":
            receipt["output_truncated"] = True
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise self._bad_response(receipt, "Generation response must contain text content")
        text = message["content"]
        receipt["output_sha256"] = _sha(text.encode("utf-8"))
        if self.chat_tokenizer is not None or self.dynamic_output_budget:
            usage = receipt["usage"]
            reported_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            receipt["prompt_tokens_usage_match"] = (
                type(reported_tokens) is int and reported_tokens >= 0 and reported_tokens == prompt_tokens)
            if type(reported_tokens) is not int or reported_tokens < 0:
                raise self._bad_response(receipt, "Generation usage.prompt_tokens must be a nonnegative integer")
            if reported_tokens != prompt_tokens:
                raise self._bad_response(receipt, "Generation usage.prompt_tokens differs from checked prompt count "
                                         f"(reported={reported_tokens}, checked={prompt_tokens})")
        if self.dynamic_output_budget:
            usage = receipt["usage"]
            completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
            total = usage.get("total_tokens") if isinstance(usage, dict) else None
            if (result.get("model") != self.model or type(completion) is not int
                    or not 0 <= completion <= output_budget or type(total) is not int
                    or total != prompt_tokens + completion or total > self.max_model_len):
                raise self._bad_response(receipt, "Dynamic generation model/usage exceeds exact admitted context")
        if receipt["finish_reason"] != "stop" or not text.strip():
            receipt["status"] = "failed"
            receipt["error"] = ("output_truncated" if receipt["finish_reason"] == "length"
                                else "incomplete_or_empty_output")
        self._emit_receipt(receipt)
        return {**receipt, "text": text}
