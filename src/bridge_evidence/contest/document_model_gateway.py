"""Audit native SDK requests while preserving the own model's HTTP protocol.

The SDK speaks to an ephemeral loopback server. Requests are forwarded intact
to the already configured LocalModelClient, with real token/budget checks and
receipts. This does not load a model or approximate request counts from events.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
import base64
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any
from uuid import uuid4

from .local_model_client import validate_text_request_policy


def _bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True, allow_nan=False).encode()


class VerifiedGenerationLengthError(ValueError):
    """Terminal HTTP200 length with checked usage, never executable output.

    ``response`` exists solely for a failed sidecar, not SDK tool execution or a
    verdict. Reconcile ``attempt`` with durable request/client/gateway/dispatch
    receipts before scheduling smaller independent work.
    """
    def __init__(self, attempt: dict, response: dict):
        super().__init__("SDK generation incomplete or truncated (verified length)")
        self.attempt = deepcopy(attempt)
        self.response = deepcopy(response)


def verified_incomplete_tool_terminal(receipt, *, model, prompt_tokens, max_tokens, max_model_len):
    """Validate an ended rejected envelope, independently of cancellation.

    A missing function name or ID never becomes executable. This predicate only
    permits the existing bounded retry after exact transport and usage checks.
    """
    proof = receipt.get("rejected_tool_terminal_candidate") or {}
    error = receipt.get("protocol_error") or {}
    details = error.get("details") or {}
    usage = receipt.get("observed_usage")
    maximum = max_tokens
    prompt = prompt_tokens
    if (receipt.get("error_code") != "incomplete_tool_call"
            or error.get("reason") != "incomplete_tool_call"
            or type(details.get("tool_index")) is not int or details["tool_index"] < 0
            or not isinstance(details.get("missing_fields"), list)
            or not details["missing_fields"]
            or any(not isinstance(field, str) for field in details["missing_fields"])
            or not set(details["missing_fields"]) <= {"id", "function.name"}
            or receipt.get("http_status") != 200
            or receipt.get("done_received") is not True
            or receipt.get("finish_reason") not in {"stop", "tool_calls", "function_call"}
            or not receipt.get("response_id")
            or proof.get("method") != "complete_sse_rejected_tool_envelope"
            or proof.get("response_id") != receipt["response_id"]
            or proof.get("model") != model
            or proof.get("done_received") is not True
            or proof.get("finish_reason") != receipt["finish_reason"]
            or proof.get("complete_tail_validated") is not True
            or proof.get("cancellation_confirmed") is not False
            or not AuditedModelGateway._hash(receipt.get("raw_sse_sha256"))
            or proof.get("raw_sse_sha256") != receipt["raw_sse_sha256"]
            or not isinstance(usage, dict) or usage != proof.get("observed_usage")
            or any(type(usage.get(key)) is not int or usage[key] < 0
                   for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            or type(prompt) is not int or prompt != usage["prompt_tokens"]
            or type(maximum) is not int or not 0 <= usage["completion_tokens"] <= maximum
            or usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]
            or usage["total_tokens"] > max_model_len):
        return False
    return True


class AuditedModelGateway(AbstractContextManager):
    """A report-local boundary for native SDK and direct semantic-review calls."""

    def __init__(self, client, *, deadline: float, max_calls: int,
                 max_output_tokens: int, max_request_bytes: int = 16 * 1024 * 1024,
                 before_request: Callable[[], None] | None = None,
                 request_kind: str = "agent_generation",
                 on_dispatch: Callable[[dict], None] | None = None,
                 on_response: Callable[[dict, dict], None] | None = None,
                 request_admission: Callable | None = None,
                 admission_identity: dict | None = None,
                 forward_lock=None,
                 generation_purpose: str | None = None,
                 on_prepared_request: Callable[[dict, dict], None] | None = None,
                 on_input_observed: Callable[[dict, dict], None] | None = None,
                 recovery_prompt: Callable[[dict], str] | None = None):
        if not isinstance(request_kind, str) or request_kind not in {"generation", "agent_generation"}:
            raise ValueError("request_kind must be generation or agent_generation")
        self.client = client
        self.deadline = deadline
        self.max_calls = max_calls
        self.max_output_tokens = max_output_tokens
        self.dynamic_output_budget = getattr(client, "dynamic_output_budget", False)
        if self.dynamic_output_budget and max_output_tokens != client.max_model_len:
            raise ValueError("Dynamic gateway output ceiling must equal model context")
        self.max_request_bytes = max_request_bytes
        self.request_kind = request_kind
        self.receipts: list[dict] = []
        self.lock = threading.Lock()
        # LocalModelClient.timeout is mutable. Keep one native SDK request
        # active for this report, including its preflight and success hook.
        self._forward_lock = forward_lock if forward_lock is not None else threading.Lock()
        if request_admission is not None and (not callable(request_admission)
                or not isinstance(admission_identity, dict)
                or set(admission_identity) != {"processor_identity_sha256", "service_identity_sha256"}
                or any(not self._hash(value) for value in admission_identity.values())):
            raise ValueError("Exact request admission requires pinned processor and service identities")
        if generation_purpose not in {None, "image_observation", "source_object_review"}:
            raise ValueError("Unsupported generation purpose")
        self.request_admission = request_admission
        self.admission_identity = deepcopy(admission_identity)
        self.generation_purpose = generation_purpose
        self.on_prepared_request = on_prepared_request
        self.on_input_observed = on_input_observed
        self.server = None
        self.thread = None
        self._accepting_requests = True
        # Inert, failed terminal evidence only. Never returned to the SDK.
        self.verified_length_attempts: list[dict] = []
        self._issued = 0
        self.before_request = before_request
        self.on_dispatch = on_dispatch
        # Synchronous response persistence before SDK parsing or success hooks.
        # Both arguments are copies; callbacks cannot alter the SDK result,
        # receipt accounting or the hashes binding the actual HTTP response.
        self.on_response = on_response
        self.on_success = None
        self.on_receipt = None
        if recovery_prompt is not None and not callable(recovery_prompt):
            raise ValueError("Loop recovery requires an exact progress prompt callback")
        self.recovery_prompt = recovery_prompt
        self.last_prepared_allocation = None

    @staticmethod
    def _hash(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    @staticmethod
    def _multimodal(payload):
        return any(isinstance(message, dict) and isinstance(message.get("content"), list)
                   and any(not isinstance(part, dict) or part.get("type") != "text"
                           for part in message["content"])
                   for message in payload.get("messages", []))

    def count_request(self, payload: dict) -> int:
        if self.dynamic_output_budget:
            return self.client.count_complete_request(payload)
        if self._multimodal(payload):
            raise ValueError("Multimodal input requires exact native processor admission; text tokenization is insufficient")
        validate_text_request_policy(payload, temperature=getattr(self.client, "temperature", 0),
                                     enable_thinking=getattr(self.client, "enable_thinking", False))
        chat_kwargs = payload["chat_template_kwargs"]
        tokenizer = getattr(self.client, "chat_tokenizer", None)
        if tokenizer is not None:
            template_kwargs = deepcopy(chat_kwargs)
            template_kwargs.update(tokenize=False, add_generation_prompt=True, tools=payload.get("tools"))
            prompt = tokenizer.apply_chat_template(
                payload["messages"], **template_kwargs,
            )
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("Chat template must render a nonempty string")
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if (not isinstance(token_ids, (list, tuple))
                    or any(type(token) is not int or token < 0 for token in token_ids)):
                raise ValueError("Chat tokenizer must encode a sequence of nonnegative integer token IDs")
            # The serving tokenizer counts the rendered prompt, also on
            # engines whose /tokenize endpoint doesn't accept chat objects.
            # Do not reuse raw-text counts for a newly rendered SDK template.
            return self.client._tokenize(
                {"model": self.client.model, "prompt": prompt, "add_special_tokens": False},
                local_token_count=len(token_ids),
                rendered_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        tokenize = {"model": self.client.model, "messages": payload["messages"],
                    "add_generation_prompt": True,
                    "chat_template_kwargs": deepcopy(chat_kwargs)}
        if "tools" in payload:
            tokenize["tools"] = payload["tools"]
        return self.client._tokenize(tokenize)

    def prepare_request(self, payload: dict) -> dict:
        """Allocate a final SDK/direct request before HTTP; no input rewrite."""
        if not self.dynamic_output_budget:
            return deepcopy(payload)
        if self.request_admission is not None:
            raise ValueError("Dynamic multimodal allocation is unsupported; native gate unchanged")
        original_timeout = self.client.timeout
        try:
            self.client.timeout = min(original_timeout, self._remaining())
            prepared, allocation = self.client.prepare_generation_request(payload)
            self.last_prepared_allocation = deepcopy(allocation)
            self._remaining()
            return prepared
        except Exception as exc:
            self.client._dynamic_budget_failure = type(exc).__name__
            audit = {"status": "rejected", "error": type(exc).__name__,
                     "request_kind": self.request_kind, "request_sha256": hashlib.sha256(_bytes(payload)).hexdigest(),
                     "generation_attempted": False, "failure_stage": "dynamic_wire_allocation",
                     "output_budget_policy": "complete_request_remaining_context_v1"}
            with self.lock:
                self.receipts.append(audit)
            if self.on_receipt is not None:
                self.on_receipt(deepcopy(audit))
            raise
        finally:
            self.client.timeout = original_timeout

    def _admit_request(self, payload, audit):
        if self.request_admission is None:
            return self.count_request(payload)
        # A private copy prevents a provider from changing the request later
        # sent to the model. Both its return and input are checked before HTTP.
        counted_payload = deepcopy(payload)
        admission = self.request_admission(counted_payload, deadline=self.deadline)
        if hashlib.sha256(_bytes(counted_payload)).hexdigest() != audit["request_sha256"]:
            raise ValueError("Native processor admission changed its input request")
        if not isinstance(admission, dict):
            raise ValueError("Exact native processor admission must return a receipt")
        admission = deepcopy(admission)
        tokens = admission.get("expanded_prompt_tokens")
        maximum = payload.get("max_tokens", payload.get("max_completion_tokens"))
        images = [part for message in payload["messages"] if isinstance(message, dict)
                  and isinstance(message.get("content"), list) for part in message["content"]
                  if isinstance(part, dict) and part.get("type") == "image_url"]
        try:
            if len(images) != 1:
                raise ValueError("Expected one embedded PNG")
            url = images[0]["image_url"]["url"]
            prefix = "data:image/png;base64,"
            if not isinstance(url, str) or not url.startswith(prefix):
                raise ValueError("Expected embedded PNG data")
            image_sha256 = hashlib.sha256(base64.b64decode(url[len(prefix):], validate=True)).hexdigest()
        except (KeyError, TypeError, ValueError):
            raise ValueError("Native observation admission requires one embedded PNG image") from None
        if (admission.get("status") != "processor_counted"
                or admission.get("method") != "installed_vllm_native_processor"
                or admission.get("request_sha256") != audit["request_sha256"]
                or type(tokens) is not int or tokens <= 0
                or type(admission.get("max_model_len")) is not int
                or admission["max_model_len"] != self.client.max_model_len
                or type(admission.get("reserved_output_tokens")) is not int
                or admission["reserved_output_tokens"] != maximum
                or not self._hash(admission.get("final_prompt_token_ids_sha256"))
                or admission.get("image_payload_sha256") != image_sha256
                or any(admission.get(key) != value for key, value in self.admission_identity.items())):
            raise ValueError("Native processor receipt does not bind this exact request and serving identity")
        audit["admission"] = admission
        self._remaining()
        return tokens

    def forward(self, payload: dict, *, request_alive=None) -> dict:
        payload = self.client.reliable_payload(payload) if getattr(self.client, "reliable_generation_policy", None) is not None else deepcopy(payload)
        started = time.monotonic()
        audit = {"status": "rejected", "started_monotonic": started,
                 "request_kind": self.request_kind,
                 "request_sha256": hashlib.sha256(_bytes(payload)).hexdigest()}
        if self.generation_purpose is not None:
            audit["purpose"] = self.generation_purpose
        acquired = False
        try:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or not self._forward_lock.acquire(timeout=remaining):
                raise TimeoutError("Report deadline exceeded while waiting for the SDK request")
            acquired = True
            if not self._accepting_requests:
                raise RuntimeError("Closed SDK gateway cannot issue a late generation")
            batch_admission = getattr(self.client, "batch_admission", None)
            if batch_admission is None:
                self._active_request_deadline = min(self.deadline, time.monotonic() + self.client.timeout)
            from .reliable_generation import ReliableGenerationError
            from .generation_stop_observer import generation_stopped
            for attempt in range(2):
                audit["recovery_index"] = attempt
                try:
                    if batch_admission is None:
                        return self._forward_serial(deepcopy(payload), audit)
                    with self.client.generation_slot(deadline=self.deadline,
                            is_valid=lambda: self._accepting_requests and (request_alive is None or request_alive())) as slot:
                        self._active_request_deadline = min(self.deadline, slot["deadline"],
                                                            time.monotonic() + self.client.timeout)
                        audit["batch_queue_seconds"] = slot["queue_seconds"]
                        issued_before = self._issued
                        try:
                            return self._forward_serial(deepcopy(payload), audit)
                        finally:
                            if not slot["transport_entered"]:
                                # Preserve any persisted dispatch intent, but
                                # settle it as unsent when stream preflight or
                                # a vanished caller prevented the actual open.
                                audit.update(generation_attempted=False, batch_dispatch_state="not_sent")
                                with self.lock:
                                    self._issued = issued_before
                except (ReliableGenerationError, VerifiedGenerationLengthError) as exc:
                    if isinstance(exc, VerifiedGenerationLengthError):
                        # HTTP has already ended with validated usage/model.
                        # Never execute or replay the truncated output. Only
                        # an explicitly capped workflow may retry smaller work.
                        if getattr(self.client, "generation_output_cap", None) is None:
                            raise
                        failed_receipt = next(row for row in self.client.calls
                                              if row.get("call_id") == exc.attempt["call_id"])
                        failed_receipt.update(error_code="output_token_limit", length_usage_verified=True)
                        code, stopped = "output_token_limit", True
                    else:
                        failed_receipt, code = exc.receipt, exc.code
                        stopped = (self._verified_incomplete_tool_terminal(failed_receipt, audit)
                            if code == "incomplete_tool_call" else generation_stopped(failed_receipt))
                    audit.update(error_code=code,
                                 response_id=failed_receipt.get("response_id"),
                                 finish_reason=failed_receipt.get("finish_reason"),
                                 cancel_status=failed_receipt.get("cancel_status"),
                                 stop_observation=deepcopy(failed_receipt.get("stop_observation")),
                                 native_stop_observation=deepcopy(failed_receipt.get("native_stop_observation")))
                    if "done_received" in failed_receipt:
                        audit["done_received"] = failed_receipt["done_received"]
                    if (attempt or self.recovery_prompt is None
                            or code not in {"content_repetition", "reasoning_repetition", "raw_repetition", "native_repetition", "content_budget_exceeded", "tool_argument_contract", "output_token_limit", "incomplete_tool_call"}
                            or getattr(self.client, "_repetition_recoveries_used", 0) >= 1
                            or not stopped):
                        if code == "incomplete_tool_call":
                            audit["recovery_blocked"] = {
                                "reason": ("same_request_recovery_exhausted" if attempt else
                                    "agent_recovery_allowance_exhausted" if getattr(self.client, "_repetition_recoveries_used", 0) >= 1 else
                                    "ended_response_proof_missing_or_invalid" if not stopped else
                                    "progress_feedback_unavailable"),
                                "protocol_error": deepcopy(failed_receipt.get("protocol_error")),
                                "failed_output_executed": False, "limit_per_agent": 1,
                                "recoveries_used": getattr(self.client, "_repetition_recoveries_used", 0)}
                        raise
                    self._remaining()
                    feedback = self.recovery_prompt(deepcopy(failed_receipt))
                    if not isinstance(feedback, str) or not feedback.strip():
                        raise ValueError("Loop recovery lacks accurate saved progress")
                    # One recovery across this Agent's client lifetime, not
                    # one per forward() call. The failed aggregate never goes
                    # into SDK memory or tool execution. Only this temporary
                    # wire attempt receives the exact progress feedback.
                    self.client._repetition_recoveries_used = 1
                    failed_receipt["recovery_authorized"] = True
                    audit.update(status="rejected", error=type(exc).__name__,
                                 seconds=time.monotonic() - audit["started_monotonic"],
                                 recovery_authorized=True,
                                 failed_generation=deepcopy(failed_receipt))
                    if code not in {"output_token_limit", "incomplete_tool_call"}:
                        audit["cancellation"] = deepcopy(failed_receipt)
                    with self.lock:
                        self.receipts.append(deepcopy(audit))
                    if self.on_receipt is not None:
                        self.on_receipt(deepcopy(audit))
                    failed_call_id = failed_receipt.get("call_id")
                    payload = deepcopy(payload)
                    payload["messages"].append({"role": "user", "content": feedback})
                    audit.clear()
                    audit.update(status="rejected", started_monotonic=time.monotonic(),
                                 request_kind=self.request_kind,
                                 recovery_of_call_id=failed_call_id,
                                 recovery_feedback_sha256=hashlib.sha256(feedback.encode()).hexdigest(),
                                 recovery_history_policy="temporary_wire_feedback_original_sdk_history_unchanged",
                                 request_sha256=hashlib.sha256(_bytes(payload)).hexdigest())
                    if self.generation_purpose is not None:
                        audit["purpose"] = self.generation_purpose
                    # Recompute complete input and remaining output; keep the
                    # original SDK request deadline including stop observation.
                    payload = self.prepare_request(payload)
        except Exception as exc:
            if self.dynamic_output_budget and audit.get("generation_attempted") is not True:
                # A rejected complete input cannot be silently retried after
                # SDK compression, even via a new report session gateway.
                self.client._dynamic_budget_failure = type(exc).__name__
            audit["error"] = type(exc).__name__
            raise
        finally:
            audit["seconds"] = time.monotonic() - audit["started_monotonic"]
            audit["total_forward_seconds"] = time.monotonic() - started
            with self.lock:
                self.receipts.append(audit)
            if acquired:
                self._active_request_deadline = None
                self._forward_lock.release()
            if self.on_receipt is not None:
                try:
                    self.on_receipt(audit)
                except Exception as exc:
                    # Preserve a rejected request's original exception. If an
                    # otherwise successful request cannot be persisted, stop
                    # explicitly; its completed HTTP receipt remains intact.
                    audit["receipt_callback_error"] = type(exc).__name__
                    if audit["status"] == "success":
                        audit.update(status="rejected", error="RuntimeError",
                                     failure_stage="receipt_callback")
                        raise RuntimeError("Could not persist the model request receipt") from exc

    def _verified_incomplete_tool_terminal(self, receipt, audit):
        if not verified_incomplete_tool_terminal(receipt, model=self.client.model,
                prompt_tokens=audit.get("prompt_tokens_checked"), max_tokens=audit.get("max_tokens"),
                max_model_len=self.client.max_model_len):
            return False
        receipt.update(usage=deepcopy(receipt["observed_usage"]), usage_known=True,
            prompt_tokens_checked=audit["prompt_tokens_checked"], max_tokens=audit["max_tokens"],
            terminal_verified_by="gateway_exact_request_admission",
            rejected_tool_output_executed=False)
        return True

    def _remaining(self) -> float:
        active = getattr(self, "_active_request_deadline", None)
        remaining = min(self.deadline, active or self.deadline) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Report deadline exceeded")
        return remaining

    def _observe_input(self, payload, audit, receipt, result=None):
        """Generation proves input delivery, independently of usable output.

        A dispatch intent or an HTTP 200 header alone is insufficient. Failed
        streams qualify only when the client actually received generated data.
        This hook runs before output validation so a truncated/invalid answer
        cannot erase evidence that the request's source text reached the model.
        """
        if receipt.get("http_status") != 200:
            return
        choices = result.get("choices") if isinstance(result, dict) else None
        response_observed = (isinstance(choices, list) and bool(choices)
            and any(isinstance(choice, dict) and isinstance(choice.get("message"), dict)
                    and choice.get("finish_reason") is not None for choice in choices))
        stream_observed = any(type(receipt.get(key)) is int and receipt[key] > 0
            for key in ("content_characters", "reasoning_characters", "tool_call_count"))
        raw = receipt.get("raw_generation_observation") or {}
        raw_observed = (isinstance(raw, dict)
            and type(raw.get("generated_tokens_observed")) is int
            and raw["generated_tokens_observed"] > 0
            and self._hash(raw.get("tokenizer_sha256"))
            and self._hash(raw.get("decoded_text_sha256")))
        # Raw generated IDs can arrive while the server's tool parser still
        # withholds every delta. Their validated receipt proves input delivery,
        # never output completeness, source understanding or safe execution.
        if not (response_observed or stream_observed or raw_observed):
            return
        audit.update(input_delivery_confirmed=True,
            input_delivery_evidence=("complete_response" if response_observed else
                "generated_stream_data" if stream_observed else "raw_generated_stream_data"))
        receipt["input_delivery_confirmed"] = True
        if self.on_input_observed is not None:
            try:
                self.on_input_observed(deepcopy(payload), deepcopy(audit))
            except Exception as exc:
                audit.update(failure_stage="input_observed_callback",
                    input_observed_callback_error=type(exc).__name__)
                raise RuntimeError("Could not persist confirmed model input delivery") from None

    def _forward_serial(self, payload: dict, audit: dict) -> dict:
        original_timeout = self.client.timeout
        try:
            if self.dynamic_output_budget and getattr(self.client, "_dynamic_budget_failure", None) is not None:
                raise RuntimeError("Prior dynamic admission failed; no fallback generation permitted")
            if not isinstance(payload, dict) or payload.get("model") != self.client.model:
                raise ValueError("SDK request must use the pinned own model")
            if len(_bytes(payload)) > self.max_request_bytes:
                raise ValueError("Complete model request exceeds the byte budget; no request sent")
            if payload.get("stream") not in (False, None):
                raise ValueError("Audited document runs require complete non-streaming responses")
            if getattr(self.client, "reliable_generation_policy", None) is not None:
                from .reliable_generation import prepare_stream_payload
                payload = prepare_stream_payload(payload)
                audit["request_sha256"] = hashlib.sha256(_bytes(payload)).hexdigest()
            if not isinstance(payload.get("messages"), list) or not payload["messages"]:
                raise ValueError("SDK request must contain chat messages")
            if self.dynamic_output_budget and ("max_completion_tokens" in payload or "max_tokens" not in payload):
                raise ValueError("Dynamic request must have one explicit canonical max_tokens; no default or alias")
            maximum = payload.get("max_tokens", payload.get("max_completion_tokens"))
            if type(maximum) is not int or not 0 < maximum <= self.max_output_tokens:
                raise ValueError("SDK output budget exceeds the report's pinned limit")
            # Image observation has its own exact native admission policy.
            # Text count and generation must use the configured policy intact.
            if self.request_admission is None:
                validate_text_request_policy(payload, temperature=getattr(self.client, "temperature", 0),
                                             enable_thinking=getattr(self.client, "enable_thinking", False))
            with self.lock:
                if self._issued >= self.max_calls:
                    raise RuntimeError("Report SDK request budget exhausted")
            self._remaining()
            # The workflow owns the combined SDK/tool generation budget.
            # A denied preflight produces no HTTP request or claimed model call.
            if self.before_request is not None:
                self.before_request()
            self.client.timeout = min(original_timeout, self._remaining())
            with self.lock:
                self._issued += 1
            prompt_tokens = self._admit_request(payload, audit)
            audit["prompt_tokens_checked"] = prompt_tokens
            if self.dynamic_output_budget:
                from .document_context_budget import POLICY, CAPPED_POLICY, remaining_output, request_input_hash
                cap = getattr(self.client, "generation_output_cap", None)
                expected = remaining_output(self.client.max_model_len, prompt_tokens, cap)
                if maximum != expected:
                    raise ValueError("Actual max_tokens differs from exact complete-request remaining context")
                audit["output_allocation"] = {"policy": CAPPED_POLICY if cap is not None else POLICY,
                    "output_cap": cap, "max_model_len": self.client.max_model_len,
                    "actual_input_tokens": prompt_tokens, "max_tokens": maximum,
                    "complete_request_input_sha256": request_input_hash(payload)}
            if prompt_tokens + maximum > self.client.max_model_len:
                raise ValueError("Complete SDK prompt and reserved output exceed context; no truncation performed")
            self.client.timeout = min(original_timeout, self._remaining())
            # This intent is persisted synchronously by the workflow before
            # HTTP generation. It is not proof that the server received it,
            # nor an upstream idempotency key. Recovery can reserve unresolved
            # intents without confusing tokenization with generation.
            dispatch = {
                "request_id": str(uuid4()), "request_sha256": audit["request_sha256"],
                "request_kind": self.request_kind, "started_at": datetime.now(timezone.utc).isoformat(),
                "model": self.client.model, "endpoint": "/v1/chat/completions",
                "max_tokens": maximum, "prompt_tokens_checked": prompt_tokens,
            }
            if self.dynamic_output_budget:
                dispatch["output_allocation"] = deepcopy(audit["output_allocation"])
            audit.update(dispatch, generation_attempted=False)
            if self.generation_purpose is not None:
                dispatch["purpose"] = self.generation_purpose
                audit["purpose"] = self.generation_purpose
            if self.on_prepared_request is not None:
                try:
                    self.on_prepared_request(deepcopy(payload), deepcopy(dispatch))
                except Exception as exc:
                    audit.update(failure_stage="request_callback", request_callback_error=type(exc).__name__)
                    raise RuntimeError("Could not persist the complete model request") from None
            if self.on_dispatch is not None:
                try:
                    self.on_dispatch(dict(dispatch))
                except Exception as exc:
                    audit.update(failure_stage="dispatch_callback", dispatch_callback_error=type(exc).__name__)
                    raise RuntimeError("Could not persist the model request dispatch") from exc
            try:
                # Persistence time uses the same report deadline, rather than
                # giving generation a fresh copy of the earlier time budget.
                self.client.timeout = min(original_timeout, self._remaining())
            except TimeoutError:
                audit["failure_stage"] = "before_generation"
                raise
            before_calls = len(self.client.calls)
            audit["generation_attempted"] = True
            try:
                result, receipt = self.client._request(
                    "/v1/chat/completions", payload, kind=self.request_kind)
            except Exception:
                audit["failure_stage"] = "generation_http"
                # _request retains actual failed HTTP receipts in its finally
                # block even when it cannot return one to this caller.
                matching = [call for call in self.client.calls[before_calls:]
                            if call.get("kind") == self.request_kind
                            and call.get("endpoint") == dispatch["endpoint"]
                            and call.get("input_sha256") == dispatch["request_sha256"]]
                if len(matching) == 1:
                    failed = matching[0]
                    failed["request_id"] = dispatch["request_id"]
                    if self.generation_purpose is not None:
                        failed["purpose"] = self.generation_purpose
                    audit.update(call_id=failed["call_id"], http_status=failed.get("http_status"))
                    self._observe_input(payload, audit, failed)
                raise
            receipt["request_id"] = dispatch["request_id"]
            if self.generation_purpose is not None:
                receipt["purpose"] = self.generation_purpose
            audit.update(call_id=receipt["call_id"], usage=result.get("usage"),
                         http_status=receipt.get("http_status"),
                         response_sha256=hashlib.sha256(_bytes(result)).hexdigest())
            receipt.update(usage=result.get("usage"), prompt_tokens_checked=prompt_tokens, max_tokens=maximum)
            self._observe_input(payload, audit, receipt, result)
            if self.dynamic_output_budget:
                receipt["output_allocation"] = deepcopy(audit["output_allocation"])
            try:
                self._remaining()
            except TimeoutError:
                receipt["status"] = "failed"
                receipt["error"] = "SDK generation returned after the report deadline"
                raise
            usage = result.get("usage")
            reported = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            if type(reported) is not int or reported != prompt_tokens:
                receipt["status"] = "failed"
                receipt["error"] = "SDK prompt token count differs from actual server usage"
                raise ValueError("SDK prompt token count differs from actual server usage")
            if self.request_admission is not None or self.dynamic_output_budget:
                completion = usage.get("completion_tokens")
                total = usage.get("total_tokens")
                if (result.get("model") != self.client.model
                        or type(completion) is not int or not 0 <= completion <= maximum
                        or type(total) is not int or total != prompt_tokens + completion
                        or total > self.client.max_model_len):
                    receipt.update(status="failed", error="Multimodal response model or complete usage differs from admission")
                    raise ValueError("Multimodal response model or complete usage differs from admission")
            choices = result.get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                receipt["status"] = "failed"
                receipt["error"] = "Own model must return one complete SDK choice"
                raise ValueError("Own model must return one complete SDK choice")
            reason = choices[0].get("finish_reason")
            audit["finish_reason"] = reason
            receipt.update(prompt_tokens_checked=prompt_tokens, finish_reason=reason)
            if reason not in {"stop", "tool_calls", "function_call"}:
                receipt["status"] = "failed"
                receipt["error"] = "SDK generation incomplete or truncated"
                completion = usage.get("completion_tokens")
                total = usage.get("total_tokens")
                response_id = result.get("id")
                # No unknown HTTP, deadline/usage/model error or other finish
                # reason may masquerade as this narrowly verified trigger.
                if (reason == "length" and receipt.get("http_status") == 200
                        and result.get("model") == self.client.model
                        and type(completion) is int and 0 <= completion <= maximum
                        and type(total) is int and total == prompt_tokens + completion
                        and total <= self.client.max_model_len
                        and isinstance(response_id, str) and response_id.strip()
                        and self._hash(receipt.get("response_sha256"))):
                    audit.update(length_usage_verified=True, response_id=response_id)
                    receipt["response_id"] = response_id
                    error = VerifiedGenerationLengthError({
                        "call_id": audit["call_id"], "request_id": audit["request_id"],
                        "response_id": response_id, "request_sha256": audit["request_sha256"],
                        "gateway_canonical_response_sha256": audit["response_sha256"],
                        "client_http_response_sha256": receipt["response_sha256"],
                        "usage": deepcopy(usage), "prompt_tokens_checked": prompt_tokens,
                    }, result)
                    self.verified_length_attempts.append({"attempt": deepcopy(error.attempt), "response": deepcopy(error.response)})
                    raise error
                raise ValueError("SDK generation incomplete or truncated")
            if self.request_admission is not None and reason != "stop":
                receipt.update(status="failed", error="Multimodal observation did not complete")
                raise ValueError("Multimodal observation did not complete")
            audit["response_validated"] = True
            if self.on_response is not None:
                try:
                    # This is a pre-delivery validation milestone, not the
                    # terminal gateway receipt. Persistence or later hooks can
                    # still fail while the upstream HTTP success remains real.
                    response_audit = {**deepcopy(audit), "status": "response_validated"}
                    self.on_response(deepcopy(result), response_audit)
                    audit["response_callback_completed"] = True
                except Exception as exc:
                    audit.update(failure_stage="response_callback",
                                 response_callback_error=type(exc).__name__)
                    # A filesystem/backend exception can contain report text
                    # or local paths. Keep both the direct and chained error
                    # presented to callers free of that callback's message.
                    raise RuntimeError("Could not persist the validated model response") from None
            if self.on_success is not None:
                self.on_success(payload, receipt)
            audit["status"] = "success"
            return result
        finally:
            self.client.timeout = original_timeout

    def __enter__(self):
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                try:
                    if self.path != "/v1/chat/completions":
                        raise ValueError("Only the pinned chat endpoint is available")
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= gateway.max_request_bytes:
                        raise ValueError("SDK request size is outside the report limit")
                    payload = json.loads(self.rfile.read(length))
                    def request_alive():
                        import select
                        import socket
                        try:
                            if select.select([self.connection], [], [], 0)[0]:
                                return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) != b""
                            return True
                        except (BlockingIOError, InterruptedError):
                            return True
                        except OSError:
                            return False
                    result = gateway.forward(payload, request_alive=request_alive)
                    status = 200
                except Exception as exc:
                    status = 400
                    result = {"error": {"type": type(exc).__name__,
                                        "message": "Audited own-model request rejected; inspect report receipts"}}
                body = _bytes(result)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        return self

    def __exit__(self, *_args):
        # An already accepted daemon HTTP handler may reach forward() late.
        # It must never dispatch after this gateway has handed off its stage.
        self._accepting_requests = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=5)
        return False
