"""Bounded SSE transport. Never retries, interprets tool arguments or runs tools.

Private generated text is returned to the caller, never printed. The caller owns
wire persistence, final fresh tokenization and backend cancellation confirmation.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import codecs
import hashlib
import json
import math
import re
import socket
import time
from urllib.parse import urlsplit
from urllib.request import Request
from uuid import uuid4


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class StreamPolicy:
    wall_seconds: float = 900
    read_idle_seconds: float = 60
    read_bytes: int = 16384
    max_event_bytes: int = 4 * 1024 * 1024
    max_response_bytes: int = 32 * 1024 * 1024
    reasoning_repeat_count: int = 8
    reasoning_repeat_min_chars: int = 256
    content_single_char_repeat: int = 512
    content_cycle_repeat_count: int = 32
    content_cycle_min_chars: int = 1024
    content_cycle_max_period: int = 32
    max_content_chars: int | None = None
    clock: object = time.monotonic

    def __post_init__(self):
        for name in ("wall_seconds", "read_idle_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Stream time limits must be positive finite numbers")
        if self.wall_seconds > 900:
            raise ValueError("Independent stream wall limit cannot exceed 900 seconds")
        for name in ("read_bytes", "max_event_bytes", "max_response_bytes", "reasoning_repeat_count", "reasoning_repeat_min_chars",
                     "content_single_char_repeat", "content_cycle_repeat_count", "content_cycle_min_chars", "content_cycle_max_period"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Stream byte/repetition limits must be positive integers")
        if self.reasoning_repeat_count < 4 or self.reasoning_repeat_min_chars < 128:
            raise ValueError("Reasoning repetition detection must remain conservative")
        if (self.content_single_char_repeat < 512 or self.content_cycle_repeat_count < 32
                or self.content_cycle_min_chars < 1024 or not 2 <= self.content_cycle_max_period <= 64):
            raise ValueError("Content repetition detection must remain conservative and bounded")
        if self.max_content_chars is not None and (type(self.max_content_chars) is not int or self.max_content_chars <= 0):
            raise ValueError("max_content_chars must be a positive integer or None")
        if not callable(self.clock):
            raise ValueError("A monotonic clock is required")


class ReliableGenerationError(RuntimeError):
    def __init__(self, code, receipt):
        super().__init__("Reliable generation failed (" + code + ")")
        self.code = code
        self.receipt = receipt
        self.unknown_usage = receipt.get("usage_known") is not True
        self.cancel_status = receipt.get("cancel_status", "not_confirmed")


class _ProtocolError(Exception):
    def __init__(self, code, **details):
        super().__init__(code)
        self.code, self.details = code, details


class _ReasoningLoop(Exception):
    pass


class _ContentBudgetExceeded(Exception):
    pass


class _ContentLoop(Exception):
    def __init__(self, details):
        super().__init__(details["reason"])
        self.details = details


class _RawGenerationLoop(_ContentLoop):
    pass


class _ToolArgumentContract(_ContentLoop):
    pass


class _NativeRepetition(_ContentLoop):
    pass


class _ContentLoopDetector:
    """Exact, incremental letter-bearing loops; not a prose length limit.

    Pure numbers, whitespace and punctuation are deliberately excluded. Short
    cycles containing line/tab/table boundaries are also excluded so ordinary
    long tables and repeated headers do not become cancellation evidence. This
    detector is intentionally narrow: it does not judge facts or relevance.
    State survives SSE/read boundaries and is bounded by the maximum period.
    """
    def __init__(self, policy):
        self.policy = policy
        self.position = 0
        self.previous = None
        self.same_character_count = 0
        self.tail = ""
        self.period_matches = [0] * (policy.content_cycle_max_period + 1)

    def _raise(self, reason, unit, repeated_characters):
        raise _ContentLoop({"channel": "content", "reason": reason,
            "content_start_char": self.position - repeated_characters,
            "content_end_char": self.position, "offset_unit": "unicode_code_points_zero_based_end_exclusive",
            "period_chars": len(unit), "repeated_characters": repeated_characters,
            "repeat_count": repeated_characters // len(unit),
            "repeated_unit_sha256": _sha(unit.encode("utf-8"))})

    def feed(self, text):
        for character in text:
            self.position += 1
            self.same_character_count = self.same_character_count + 1 if character == self.previous else 1
            self.previous = character
            if character.isalpha() and self.same_character_count >= self.policy.content_single_char_repeat:
                self._raise("single_character_run", character, self.same_character_count)
            for width in range(2, min(len(self.tail), self.policy.content_cycle_max_period) + 1):
                matched = self.period_matches[width] + 1 if character == self.tail[-width] else 0
                self.period_matches[width] = matched
                repeated = matched + width
                if repeated < max(self.policy.content_cycle_min_chars, width * self.policy.content_cycle_repeat_count):
                    continue
                unit = (self.tail + character)[-width:]
                if any(c.isalpha() for c in unit) and not any(c in "\r\n\t|" for c in unit):
                    self._raise("short_cycle", unit, repeated)
            self.tail = (self.tail + character)[-self.policy.content_cycle_max_period:]


class _Aggregate:
    def __init__(self, policy, *, raw_generation_guard=None):
        self.policy = policy
        self.raw_generation_guard = raw_generation_guard
        self.native_stop_reason = None
        self.native_stop_observation = None
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.buffer = ""
        self.data = []
        self.event_bytes = 0
        self.done = False
        self.finish = None
        self.usage = None
        self.identity = {}
        self.content = []
        self.content_characters = 0
        self.content_loop_detector = _ContentLoopDetector(policy)
        self.reasoning = []
        self.reasoning_tail = ""
        self.refusal = []
        self.tools = {}
        self.events = 0
        self.choice_seen = False

    def _reasoning_fragment(self, text):
        self.reasoning.append(text)
        self.reasoning_tail = (self.reasoning_tail + text)[-32768:]
        tail = self.reasoning_tail
        # This heuristic inspects reasoning only. Ordinary prose has a separate
        # conservative detector; structured tool argument strings are untouched.
        lines = [re.sub(r"\s+", " ", s).strip() for s in tail.splitlines() if s.strip()]
        n = self.policy.reasoning_repeat_count
        if len(lines) >= n and len(lines[-1]) >= 32 and len(set(lines[-n:])) == 1:
            if sum(map(len, lines[-n:])) >= self.policy.reasoning_repeat_min_chars:
                raise _ReasoningLoop("repeated_reasoning_lines")
        compact = re.sub(r"\s+", " ", tail).rstrip()
        # Repeated short reasoning paragraphs/cycles, including a generator
        # that omits newlines. A suffix must contain >=256 repeated characters.
        for width in range(16, min(512, len(compact) // n) + 1):
            if width * n < self.policy.reasoning_repeat_min_chars:
                continue
            unit = compact[-width:]
            if len(set(unit)) >= 4 and compact.endswith(unit * n):
                raise _ReasoningLoop("repeated_reasoning_cycle")

    def _event(self):
        if not self.data:
            return
        raw = "\n".join(self.data)
        self.data = []
        self.event_bytes = 0
        if self.done:
            raise _ProtocolError("data_after_done")
        if raw.strip() == "[DONE]":
            self.done = True
            return
        chunk = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(_ProtocolError("non_finite_json")))
        if type(chunk) is not dict or "error" in chunk:
            raise _ProtocolError("invalid_or_error_event")
        self.events += 1
        for key in ("id", "model", "created", "system_fingerprint"):
            if chunk.get(key) is not None:
                if key in self.identity and self.identity[key] != chunk[key]:
                    raise _ProtocolError("response_identity_changed")
                self.identity[key] = chunk[key]
        usage = chunk.get("usage")
        if usage is not None:
            if type(usage) is not dict or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
                raise _ProtocolError("invalid_usage")
            if usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]:
                raise _ProtocolError("inconsistent_usage")
            if self.usage is not None and (usage["prompt_tokens"] != self.usage["prompt_tokens"] or usage["completion_tokens"] < self.usage["completion_tokens"]):
                raise _ProtocolError("usage_decreased_or_prompt_changed")
            self.usage = deepcopy(usage)
        choices = chunk.get("choices")
        if type(choices) is not list or len(choices) > 1:
            raise _ProtocolError("exactly_one_choice_required")
        if not choices:
            if usage is None:
                raise _ProtocolError("empty_event_without_usage")
            return
        choice = choices[0]
        if type(choice) is not dict or type(choice.get("index")) is not int or choice["index"] != 0:
            raise _ProtocolError("choice_index_invalid")
        self.choice_seen = True
        delta = choice.get("delta")
        if type(delta) is not dict:
            raise _ProtocolError("delta_invalid")
        stop_reason = choice.get("stop_reason")
        if stop_reason is not None:
            self.native_stop_reason = stop_reason
        if stop_reason == "repetition_detected" or choice.get("finish_reason") == "repetition":
            if (stop_reason == "repetition_detected" and self.identity.get("id")
                    and choice.get("finish_reason") in {"stop", "tool_calls", "repetition"}):
                self.finish = choice["finish_reason"]
                self.native_stop_observation = {"response_id": self.identity["id"], "choice_index": 0,
                    "stop_reason": stop_reason, "finish_reason": choice["finish_reason"],
                    "terminal_observed": True, "method": "native_terminal_event"}
        # Server tool parsing can emit empty deltas while withholding thousands
        # of generated argument tokens. Observe the original IDs first.
        if self.raw_generation_guard is not None:
            ids = choice.get("token_ids")
            if ids is not None:
                try:
                    alarm = self.raw_generation_guard.feed(ids)
                except ValueError as exc:
                    raise _ProtocolError("raw_generation_token_ids_invalid") from exc
                if alarm is not None:
                    raise (_ToolArgumentContract(alarm) if alarm.get("kind") == "tool_argument_contract" else _RawGenerationLoop(alarm))
            elif any(delta.get(k) for k in ("content", "reasoning_content", "reasoning", "refusal", "tool_calls")):
                raise _ProtocolError("raw_generation_token_ids_missing")
        if stop_reason == "repetition_detected" or choice.get("finish_reason") == "repetition":
            raise _NativeRepetition({"channel": "native_token_patterns", "reason": "repetition_detected",
                "stop_reason": stop_reason, "finish_reason": choice.get("finish_reason")})
        if self.finish is not None and (delta or choice.get("finish_reason") not in (None, self.finish)):
            raise _ProtocolError("delta_after_finish")
        if delta.get("role") not in (None, "assistant"):
            raise _ProtocolError("delta_role_invalid")
        for name, target in (("content", self.content), ("refusal", self.refusal)):
            text = delta.get(name)
            if text is not None:
                if type(text) is not str:
                    raise _ProtocolError("text_delta_invalid")
                target.append(text)
                if name == "content":
                    # Count ordinary assistant text only, in Unicode code
                    # points. Reasoning, refusals and tool argument JSON are
                    # separate channels and do not consume this optional cap.
                    self.content_characters += len(text)
                    self.content_loop_detector.feed(text)
                    limit = self.policy.max_content_chars
                    if limit is not None and self.content_characters > limit:
                        raise _ContentBudgetExceeded("ordinary_content_character_limit")
        reasoning = delta.get("reasoning_content", delta.get("reasoning"))
        if delta.get("reasoning_content") is not None and delta.get("reasoning") is not None and delta["reasoning_content"] != delta["reasoning"]:
            raise _ProtocolError("conflicting_reasoning_aliases")
        if reasoning is not None:
            if type(reasoning) is not str:
                raise _ProtocolError("reasoning_delta_invalid")
            self._reasoning_fragment(reasoning)
        tool_calls = delta.get("tool_calls")
        if tool_calls is not None:
            if type(tool_calls) is not list:
                raise _ProtocolError("tool_delta_invalid")
            for tool in tool_calls:
                if type(tool) is not dict or type(tool.get("index")) is not int or tool["index"] < 0:
                    raise _ProtocolError("tool_index_invalid")
                item = self.tools.setdefault(tool["index"], {"id": None, "type": "function", "name": [], "arguments": []})
                if tool.get("id") is not None:
                    if type(tool["id"]) is not str or not tool["id"] or item["id"] not in (None, tool["id"]):
                        raise _ProtocolError("tool_id_changed")
                    item["id"] = tool["id"]
                if tool.get("type") not in (None, "function"):
                    raise _ProtocolError("tool_type_invalid")
                function = tool.get("function", {})
                if type(function) is not dict:
                    raise _ProtocolError("tool_function_invalid")
                for key in ("name", "arguments"):
                    if function.get(key) is not None:
                        if type(function[key]) is not str:
                            raise _ProtocolError("tool_function_fragment_invalid")
                        item[key].append(function[key])
        finish = choice.get("finish_reason")
        if finish is not None:
            if finish not in {"stop", "tool_calls", "length", "content_filter", "function_call"}:
                raise _ProtocolError("finish_reason_invalid")
            if self.finish not in (None, finish):
                raise _ProtocolError("finish_reason_changed")
            self.finish = finish

    def feed(self, raw, *, final=False):
        self.buffer += self.decoder.decode(raw, final=final)
        if len(self.buffer.encode("utf-8")) > self.policy.max_event_bytes:
            raise _ProtocolError("event_byte_limit")
        while True:
            found = re.search(r"\r\n|\n|\r", self.buffer)
            if found is None or (found.group() == "\r" and found.end() == len(self.buffer) and not final):
                break
            line, self.buffer = self.buffer[:found.start()], self.buffer[found.end():]
            if line.startswith("\ufeff") and self.events == 0 and not self.data:
                line = line[1:]
            if not line:
                self._event()
            elif line.startswith(":"):
                continue
            else:
                name, sep, value = line.partition(":")
                if name == "data":
                    value = value[1:] if sep and value.startswith(" ") else value
                    self.data.append(value)
                    self.event_bytes += len(value.encode("utf-8"))
                    if self.event_bytes > self.policy.max_event_bytes:
                        raise _ProtocolError("event_byte_limit")
        if final:
            if self.buffer:
                raise _ProtocolError("unterminated_sse_line")
            if self.data:
                raise _ProtocolError("unterminated_sse_event")

    def result(self, expected_model):
        if not self.done or self.finish is None or self.usage is None or not self.choice_seen:
            raise _ProtocolError("missing_done_finish_or_usage")
        if self.identity.get("model") != expected_model or type(self.identity.get("id")) is not str or not self.identity["id"]:
            raise _ProtocolError("missing_or_wrong_response_identity")
        if self.raw_generation_guard is not None:
            try:
                self.raw_generation_guard.validate_complete(self.usage)
            except ValueError as exc:
                raise _ProtocolError("raw_generation_incomplete_token_coverage") from exc
        if sorted(self.tools) != list(range(len(self.tools))):
            raise _ProtocolError("tool_index_gap")
        message = {"role": "assistant", "content": "".join(self.content) if self.content else None}
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        if self.refusal:
            message["refusal"] = "".join(self.refusal)
        if self.tools:
            tools = []
            for index in sorted(self.tools):
                item = self.tools[index]
                name = "".join(item["name"])
                if not item["id"] or not name:
                    raise _ProtocolError("incomplete_tool_call", tool_index=index,
                        missing_fields=(["id"] if not item["id"] else [])
                            + (["function.name"] if not name else []))
                tools.append({"id": item["id"], "type": "function", "function": {"name": name, "arguments": "".join(item["arguments"])}})
            message["tool_calls"] = tools
        if self.finish == "tool_calls" and not self.tools:
            raise _ProtocolError("missing_finished_tool_calls")
        return {**self.identity, "object": "chat.completion", "choices": [{"index": 0, "finish_reason": self.finish, "message": message}], "usage": deepcopy(self.usage)}


def prepare_stream_payload(payload):
    """Call before final fresh tokenize/allocation; only transport flags change."""
    if type(payload) is not dict:
        raise ValueError("Chat payload must be a dict")
    prepared = deepcopy(payload)
    if type(prepared.get("n", 1)) is not int or prepared.get("n", 1) != 1:
        raise ValueError("Exactly one streamed choice is supported")
    prepared["stream"] = True
    options = prepared.get("stream_options", {})
    if type(options) is not dict:
        raise ValueError("stream_options must be a dict")
    prepared["stream_options"] = {**options, "include_usage": True}
    _bytes(prepared)
    return prepared


def _read_timeout(response, seconds):
    setter = getattr(response, "set_read_timeout", None)
    if callable(setter):
        setter(seconds)
        return
    # urllib HTTPResponse carries its genuine socket here. A capture wrapper
    # must expose set_read_timeout; do not pretend a Python clock aborts read().
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is None or not callable(getattr(sock, "settimeout", None)):
        raise _ProtocolError("read_timeout_unavailable")
    sock.settimeout(seconds)


def _disconnect_failed_response(response):
    """Signal EOF even when urllib still has another reference to the socket.

    This only closes transport. The independent observer must still prove that
    generation stopped; a successful shutdown is never an abort ACK.
    """
    sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    if sock is not None and callable(getattr(sock, "shutdown", None)):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # Already disconnected; close and observe the server below.
    response.close()


def stream_chat(client, payload, *, deadline, policy=None, on_chunk=None, on_cancel=None,
                on_dispatch=None):
    """Return a complete OpenAI JSON response and transport receipt, or raise.

    on_chunk(bytes) is raw-byte capture only; never printed here. on_cancel is
    called as on_cancel(receipt, response) once after closing a failed connection
    and may return a root-owned
    settlement dict. Its absence never implies server cancellation confirmed.
    Length/content_filter finish responses are returned truthfully for the
    existing caller's quality/length gate; transport completion is not quality.
    on_dispatch(receipt) synchronously reserves the call before any transport.
    If it fails, no model request is sent.
    """
    policy = StreamPolicy() if policy is None else StreamPolicy(**policy) if type(policy) is dict else policy
    if not isinstance(policy, StreamPolicy) or type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("Explicit monotonic deadline and StreamPolicy required")
    if any(observer is not None and not callable(observer)
           for observer in (on_chunk, on_cancel, on_dispatch)):
        raise ValueError("Stream observers must be callable")
    prepared = prepare_stream_payload(payload)
    if prepared.get("model") != client.model:
        raise ValueError("Stream request must use the pinned own model")
    base = urlsplit(client.base_url)
    if base.scheme != "http" or base.hostname not in {"127.0.0.1", "localhost", "::1"} or base.username or base.password or base.path not in ("", "/") or base.query or base.fragment:
        raise ValueError("Only normalized loopback own-model base URL allowed")
    body = _bytes(prepared)
    start = policy.clock()
    wall_deadline = min(float(deadline), start + policy.wall_seconds)
    receipt = {"call_id": str(uuid4()), "kind": "generation", "endpoint": "/v1/chat/completions", "method": "POST", "model": client.model,
        "started_at": datetime.now(timezone.utc).isoformat(), "input_sha256": _sha(body), "source_payload_sha256": _sha(_bytes(payload)),
        "status": "failed", "http_status": None, "usage": None, "usage_known": False, "finish_reason": None, "done_received": False,
        "raw_sse_sha256": None, "response_sha256": None, "output_sha256": None, "seconds": 0.0, "chunks": 0, "received_bytes": 0,
        "wall_seconds_limit": min(policy.wall_seconds, max(0.0, float(deadline) - start)), "read_idle_seconds": policy.read_idle_seconds,
        "max_content_chars": policy.max_content_chars, "thinking_token_budget": prepared.get("thinking_token_budget"),
        "content_repetition_policy": {"single_char_repeat": policy.content_single_char_repeat,
            "cycle_repeat_count": policy.content_cycle_repeat_count, "cycle_min_chars": policy.content_cycle_min_chars,
            "cycle_max_period": policy.content_cycle_max_period},
        "automatic_retry": False, "cancel_status": "not_needed_before_dispatch", "generation_attempted": False}
    receipt.update(connection_close_seconds=None, cancellation_observation_seconds=None)
    guard_factory = getattr(client, "reliable_generation_raw_guard_factory", None)
    raw_guard = guard_factory() if guard_factory is not None else None
    if getattr(client, "reliable_tool_argument_contract", False) and payload.get("tools"):
        if raw_guard is None or not hasattr(raw_guard, "configure_tool_parameters"):
            raise ValueError("Tool argument contract protection requires the verified raw-token observer")
        raw_guard.configure_tool_parameters(payload["tools"])
    aggregate = _Aggregate(policy, raw_generation_guard=raw_guard)
    response = None
    raw_hash = hashlib.sha256()
    close_error = None
    def remaining():
        seconds = wall_deadline - policy.clock()
        if seconds <= 0:
            raise TimeoutError("generation_wall_deadline")
        return min(policy.read_idle_seconds, seconds)
    try:
        timeout = remaining()
        request = Request(client.base_url + "/v1/chat/completions", data=body, method="POST", headers={
            "Content-Type": "application/json", "Accept": "text/event-stream", "Authorization": "Bearer " + client.api_key})
        receipt["status"] = "running"
        if on_dispatch is not None:
            on_dispatch(receipt)
        timeout = remaining()
        admission = getattr(client, "batch_admission", None)
        if admission is not None:
            admission.transport_started()
        receipt["generation_attempted"] = True
        receipt["cancel_status"] = "not_confirmed"
        response = client._opener.open(request, timeout=timeout)
        receipt["wire_attempt_id"] = getattr(response, "wire_attempt_id", None)
        receipt["http_status"] = getattr(response, "status", None)
        if receipt["http_status"] != 200:
            raise _ProtocolError("non_200_stream_response")
        geturl = getattr(response, "geturl", None)
        if callable(geturl) and geturl() != request.full_url:
            raise _ProtocolError("redirected_stream_response")
        headers = getattr(response, "headers", None)
        if headers is not None and headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
            raise _ProtocolError("not_event_stream")
        while not aggregate.done:
            _read_timeout(response, remaining())
            reader = getattr(response, "read1", None)
            raw = reader(policy.read_bytes) if callable(reader) else response.read(policy.read_bytes)
            if not isinstance(raw, bytes):
                raise _ProtocolError("stream_read_not_bytes")
            if not raw:
                aggregate.feed(b"", final=True)
                remaining()
                break
            raw_hash.update(raw)
            receipt["chunks"] += 1
            receipt["received_bytes"] += len(raw)
            if receipt["received_bytes"] > policy.max_response_bytes:
                raise _ProtocolError("response_byte_limit")
            if on_chunk is not None:
                on_chunk(raw)
            remaining()
            aggregate.feed(raw)
        # Validate all bytes already received, including UTF-8 or an incomplete
        # extra event after DONE. The delimiter does not excuse malformed tails.
        aggregate.feed(b"", final=True)
        result = aggregate.result(client.model)
        receipt.update(status="success", usage=deepcopy(result["usage"]), usage_known=True, finish_reason=aggregate.finish,
            done_received=True, response_sha256=_sha(_bytes(result)), output_sha256=_sha(_bytes(result)), raw_sse_sha256=raw_hash.hexdigest(),
            cancel_status="not_needed_stream_completed", reasoning_characters=sum(map(len, aggregate.reasoning)),
            content_characters=aggregate.content_characters, tool_call_count=len(aggregate.tools), response_id=aggregate.identity.get("id"))
        marker = getattr(response, "mark_stream_completed", None)
        if callable(marker):
            marked = marker(result, receipt)
            if isinstance(marked, dict):
                for key in ("attempt_id", "response_body_sha256", "response_body_bytes", "aggregate_response_path",
                            "aggregate_response_sha256", "aggregated_response_canonical_sha256"):
                    if key in marked:
                        receipt[key] = deepcopy(marked[key])
        return result, receipt
    except Exception as exc:
        failure_detected_at = policy.clock()
        code = ("incomplete_tool_call" if isinstance(exc, _ProtocolError) and exc.code == "incomplete_tool_call"
                else "tool_argument_contract" if isinstance(exc, _ToolArgumentContract) else "raw_repetition" if isinstance(exc, _RawGenerationLoop) else "native_repetition" if isinstance(exc, _NativeRepetition)
                else "content_repetition" if isinstance(exc, _ContentLoop) else "content_budget_exceeded" if isinstance(exc, _ContentBudgetExceeded) else "reasoning_repetition" if isinstance(exc, _ReasoningLoop) else "wall_or_read_timeout" if isinstance(exc, (TimeoutError, socket.timeout))
                else "stream_protocol_error" if isinstance(exc, (_ProtocolError, UnicodeError, json.JSONDecodeError)) else "stream_transport_error")
        receipt.update(status="failed", error_code=code, error_type=type(exc).__name__, usage=None, usage_known=False,
            done_received=aggregate.done, finish_reason=aggregate.finish, observed_usage=deepcopy(aggregate.usage), raw_sse_sha256=raw_hash.hexdigest(),
            content_characters=aggregate.content_characters, reasoning_characters=sum(map(len, aggregate.reasoning)), tool_call_count=len(aggregate.tools),
            response_id=aggregate.identity.get("id"), detected_after_seconds=max(0.0, failure_detected_at - start))
        if isinstance(exc, _ProtocolError):
            receipt["protocol_error"] = {"reason": exc.code, "details": deepcopy(exc.details)}
        if code == "incomplete_tool_call":
            # result() reaches this precise check only after the complete SSE
            # tail, DONE, usage shape, pinned identity, index order and raw-token
            # coverage have passed. The gateway must still bind observed usage
            # to its exact admitted prompt and output allocation before retry.
            # This is an ended response, never a client cancellation ACK.
            receipt["rejected_tool_terminal_candidate"] = {
                "method": "complete_sse_rejected_tool_envelope",
                "response_id": aggregate.identity.get("id"),
                "model": aggregate.identity.get("model"),
                "done_received": aggregate.done, "finish_reason": aggregate.finish,
                "observed_usage": deepcopy(aggregate.usage),
                "raw_sse_sha256": raw_hash.hexdigest(),
                "complete_tail_validated": True,
                "cancellation_confirmed": False}
        if isinstance(exc, _ContentLoop):
            receipt["repetition_detection"] = {**deepcopy(exc.details), "event_index": aggregate.events,
                "detected_after_seconds": receipt["detected_after_seconds"],
                "detected_at": datetime.now(timezone.utc).isoformat()}
        if isinstance(exc, _ToolArgumentContract):
            receipt["tool_argument_detection"] = deepcopy(exc.details)
        if isinstance(exc, _ContentBudgetExceeded):
            receipt["content_budget_detection"] = {"reason": str(exc), "channel": "ordinary_content",
                "limit_characters": policy.max_content_chars, "observed_characters": aggregate.content_characters,
                "event_index": aggregate.events, "tool_arguments_counted": False}
        if aggregate.raw_generation_guard is not None:
            receipt["raw_generation_observation"] = aggregate.raw_generation_guard.evidence()
        if aggregate.native_stop_reason is not None:
            receipt["native_stop_reason"] = aggregate.native_stop_reason
        if (aggregate.native_stop_observation is not None
                and aggregate.identity.get("model") == client.model):
            receipt["native_stop_observation"] = deepcopy(aggregate.native_stop_observation)
        if receipt["http_status"] is None and type(getattr(exc, "code", None)) is int:
            receipt["http_status"] = exc.code
        if response is not None:
            try: _disconnect_failed_response(response)
            except Exception as failure: close_error = type(failure).__name__
            finally: receipt["connection_close_seconds"] = max(0.0, policy.clock() - failure_detected_at)
        receipt["connection_closed"] = response is not None and close_error is None
        if close_error is not None: receipt["close_error_type"] = close_error
        if code == "incomplete_tool_call":
            # This exact failure is discovered only after a complete valid SSE
            # tail. Do not poll whole-service idle while other reports run.
            # Admission/usage validation and retry remain gateway duties.
            receipt.update(cancel_status="not_requested_response_ended", cancel_attempted=False)
        elif receipt["generation_attempted"]:
            receipt["cancel_status"] = "not_confirmed"
            if on_cancel is not None:
                try:
                    observation_started_at = policy.clock()
                    try:
                        settlement = on_cancel(deepcopy(receipt), response)
                    finally:
                        receipt["cancellation_observation_seconds"] = max(0.0, policy.clock() - observation_started_at)
                    _bytes(settlement)
                    if type(settlement) is not dict or settlement.get("status") not in {
                            "confirmed", "cancellation_confirmed", "unknown", "cancellation_unknown", "not_confirmed"}:
                        raise ValueError("Cancellation callback must provide explicit settlement status")
                    receipt["cancellation_settlement"] = settlement
                    receipt["cancel_status"] = settlement["status"]
                    if type(settlement.get("stop_observation")) is dict:
                        receipt["stop_observation"] = deepcopy(settlement["stop_observation"])
                except Exception as failure:
                    receipt["cancel_status"] = "unknown"
                    receipt["cancel_error_type"] = type(failure).__name__
            marker = getattr(response, "mark_cancelled", None) if response is not None else None
            if callable(marker):
                try:
                    marked = marker(deepcopy(receipt.get("cancellation_settlement", {"status": receipt["cancel_status"]})))
                    # Root confirmation is only usable if the actual capture
                    # accepted its identity/attempt/raw-observation proof.
                    if isinstance(marked, dict):
                        receipt["cancel_status"] = ("cancellation_confirmed"
                            if marked.get("status") == "cancellation_confirmed" else "unknown")
                        if "cancellation_receipt" in marked:
                            receipt["cancellation_receipt"] = deepcopy(marked["cancellation_receipt"])
                    elif receipt["cancel_status"] in {"confirmed", "cancellation_confirmed"}:
                        receipt["cancel_status"] = "unknown"
                except Exception as failure:
                    receipt["cancel_status"] = "unknown";receipt["cancel_marker_error_type"] = type(failure).__name__
            elif receipt["cancel_status"] in {"confirmed", "cancellation_confirmed"}:
                # Without the wire marker, cancellation remains unverified.
                receipt["cancel_status"] = "unknown"
        raise ReliableGenerationError(code, receipt) from exc
    except BaseException:
        receipt.update(status="interrupted", error="request_outcome_unknown", usage=None,
                       usage_known=False, timing_complete=False,
                       done_received=aggregate.done, finish_reason=aggregate.finish,
                       raw_sse_sha256=raw_hash.hexdigest())
        raise
    finally:
        if aggregate.raw_generation_guard is not None:
            receipt["raw_generation_observation"] = aggregate.raw_generation_guard.evidence()
        if aggregate.native_stop_reason is not None:
            receipt["native_stop_reason"] = aggregate.native_stop_reason
        if (aggregate.native_stop_observation is not None
                and aggregate.identity.get("model") == client.model):
            receipt["native_stop_observation"] = deepcopy(aggregate.native_stop_observation)
        receipt["native_repetition_policy"] = deepcopy(prepared.get("repetition_detection"))
        receipt["seconds"] = max(0.0, policy.clock() - start)
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        if response is not None and close_error is None:
            try: response.close()
            except Exception as failure:
                receipt["close_error_type"] = type(failure).__name__
