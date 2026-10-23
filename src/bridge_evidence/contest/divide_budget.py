"""Optional report-wide budgets at the divide workflow's real HTTP boundary."""
from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
import sys
import threading
import time
from uuid import uuid4

from .document_model_gateway import AuditedModelGateway
from .local_model_client import LocalModelClient
from .run_state import atomic_json, json_digest


GENERATIONS = {"generation", "agent_generation"}


class DivideBudget:
    def __init__(self, directory: Path, *, started: float, report_timeout=None, max_model_calls=None):
        if report_timeout is not None and (type(report_timeout) not in (int, float)
                or not math.isfinite(report_timeout) or report_timeout <= 0):
            raise ValueError("Divide report timeout must be positive and finite")
        if max_model_calls is not None and (type(max_model_calls) is not int or max_model_calls <= 0):
            raise ValueError("Divide maximum model calls must be a positive integer")
        self.directory = Path(directory)
        self.started = started
        self.report_timeout = report_timeout
        self.max_model_calls = max_model_calls
        self.deadline = started + report_timeout if report_timeout is not None else None
        self.generation_count = 0
        self.dispatches = []
        self.failures = []
        self.gateway_receipts = []
        self.client = None
        self.lock = threading.RLock()

    def check(self, *, generation=False, stage="check"):
        code = None
        if self.deadline is not None and time.monotonic() >= self.deadline:
            code = "REPORT_DEADLINE_EXCEEDED"
        elif generation and self.max_model_calls is not None and self.generation_count >= self.max_model_calls:
            code = "REPORT_GENERATION_BUDGET_EXHAUSTED"
        if code:
            self.failures.append({"code": code, "stage": stage})
            if code == "REPORT_DEADLINE_EXCEEDED":
                raise TimeoutError("Divide report deadline exceeded")
            raise RuntimeError("Divide report generation budget exhausted")

    def snapshot(self):
        unresolved = sum(row.get("generation_attempted") is None for row in self.dispatches)
        return {"schema_version": 1, "report_timeout": self.report_timeout,
                "max_model_calls": self.max_model_calls,
                "generation_requests": self.generation_count - unresolved,
                "generation_allowance_spent": self.generation_count,
                "unresolved_generation_reservations": unresolved,
                "seconds": time.monotonic() - self.started, "dispatches": self.dispatches,
                "failures": self.failures, "gateway_receipts": self.gateway_receipts,
                "model_calls": self.client.calls if self.client is not None else [],
                "counting_scope": "generation_requests counts known HTTP attempts including failures; unresolved reservations may have reached upstream and are included only in generation_allowance_spent; tokenizer/identity HTTP excluded",
                "dispatch_semantics": "generation_attempted=null reserves allowance before HTTP; after a crash its outcome is unknown, not proof of delivery or non-delivery; no exactly-once guarantee",
                "timing_scope": "document entry before conversion through publication checks",
                "resume_supported": False}

    def persist(self):
        with self.lock:
            atomic_json(self.directory / "divide_budget.json", self.snapshot())

    def persist_preserving_error(self):
        active_error = sys.exc_info()[0] is not None
        try:
            self.persist()
        except Exception as exc:
            self.failures.append({"code": "BUDGET_RECEIPT_PERSISTENCE_FAILED", "error_type": type(exc).__name__})
            if not active_error:
                raise

    def wrap(self, client):
        self.client = client
        return BudgetedDivideClient(client, self)

    def gateway(self, client, *, output_tokens, enabled=True):
        if not enabled:
            return nullcontext(None)
        # The gateway requires a finite lock timeout. With only a call cap,
        # its technical lock horizon stays below Python's platform maximum;
        # no report deadline is declared or enforced by this budget.
        deadline = self.deadline if self.deadline is not None else time.monotonic() + threading.TIMEOUT_MAX / 2
        gateway = AuditedModelGateway(client, deadline=deadline, max_calls=sys.maxsize,
                                      max_output_tokens=output_tokens,
                                      before_request=lambda: self.check(generation=True, stage="sdk_preflight"))
        # Gateway admissions include tokenization rejects; only our actual
        # HTTP boundary spends the combined generation allowance.
        self.gateway_receipts = gateway.receipts
        gateway.on_receipt = lambda _: self.persist_preserving_error()
        return gateway

    def request(self, proxy, endpoint, payload, *, kind):
        with self.lock:
            is_generation = kind in GENERATIONS
            self.check(generation=is_generation, stage="before_http")
            dispatch = None
            if is_generation:
                dispatch = {"request_id": str(uuid4()), "kind": kind, "endpoint": endpoint,
                            "request_sha256": json_digest(payload), "generation_attempted": False,
                            "status": "dispatch_intent"}
                self.dispatches.append(dispatch)
                self.persist()
            raw = self.client
            original_timeout = raw.timeout
            call_start = len(raw.calls)
            try:
                self.check(generation=is_generation, stage="before_dispatch")
                if is_generation:
                    self.generation_count += 1
                    dispatch.update(generation_attempted=None, status="dispatch_pending")
                    # Durable intent BEFORE sending. A crash from this point
                    # leaves an unresolved reservation, never a claim that no
                    # request was made or that the server definitely saw it.
                    self.persist()
                # Persistence uses the same deadline. The exact last allowed
                # generation was already reserved; do not reject it again.
                self.check(stage="after_dispatch_persistence")
                raw.timeout = min(proxy.timeout, self.deadline - time.monotonic()) if self.deadline is not None else proxy.timeout
                if is_generation:
                    dispatch.update(generation_attempted=True, status="pending")
                result, receipt = raw._request(endpoint, payload, kind=kind)
                if is_generation:
                    # Preserve upstream HTTP success even when the deadline
                    # rejects this late result. Usage is real server data.
                    receipt.update(usage=result.get("usage"), response_id=result.get("id"),
                                   response_model=result.get("model"))
                    choices = result.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        receipt["finish_reason"] = choices[0].get("finish_reason")
                self.check(stage="after_http")
                if dispatch is not None:
                    dispatch.update(status="returned", call_id=receipt["call_id"], usage=receipt.get("usage"))
                return result, receipt
            except Exception as exc:
                if dispatch is not None:
                    if dispatch["generation_attempted"] is None:
                        # This process knows it never called the transport.
                        # If this correction also fails to persist, the older
                        # durable pending record remains explicitly unknown.
                        self.generation_count -= 1
                        dispatch.update(generation_attempted=False, status="failed_before_http")
                    else:
                        dispatch["status"] = "failed"
                    dispatch["error_type"] = type(exc).__name__
                self.failures.append({"code": "HTTP_OR_BUDGET_FAILURE", "stage": "request",
                                      "kind": kind, "error_type": type(exc).__name__})
                raise
            finally:
                raw.timeout = original_timeout
                if dispatch is not None:
                    matching = [row for row in raw.calls[call_start:] if row.get("kind") == kind
                                and row.get("endpoint") == endpoint
                                and row.get("input_sha256") == dispatch["request_sha256"]]
                    if len(matching) == 1:
                        receipt = matching[0]
                        dispatch.update(call_id=receipt["call_id"], http_status=receipt.get("http_status"),
                                        http_receipt_status=receipt.get("status"), usage=receipt.get("usage"))
                        receipt["divide_request_id"] = dispatch["request_id"]
                self.persist_preserving_error()


class BudgetedDivideClient(LocalModelClient):
    """Reuse client generation/parsing semantics while guarding every HTTP call."""

    def __init__(self, client, budget):
        self._delegate = client
        self._budget = budget
        self.timeout = client.timeout

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    @property
    def receipt_sink(self):
        return self._delegate.receipt_sink

    @receipt_sink.setter
    def receipt_sink(self, callback):
        # HTTP reservations are emitted by the underlying transport, so the
        # durable workflow journal must be installed there, not on the proxy.
        self._delegate.receipt_sink = callback

    @property
    def request_context(self):
        return self._delegate.request_context

    @request_context.setter
    def request_context(self, context):
        self._delegate.request_context = context

    def _request(self, endpoint, payload, *, kind):
        return self._budget.request(self, endpoint, payload, kind=kind)

    def count_tokens(self, text):
        self._budget.check(stage="before_token_count")
        return super().count_tokens(text)

    def generate(self, messages, max_tokens=None):
        try:
            self._budget.check(generation=True, stage="direct_preflight")
            result = super().generate(messages, max_tokens=max_tokens)
            usage = result.get("usage")
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            if type(prompt_tokens) is not int or prompt_tokens != result.get("prompt_tokens_checked"):
                for receipt in self.calls:
                    if receipt.get("call_id") == result["call_id"]:
                        receipt.update(status="failed", error="Generation usage differs from checked prompt count")
                self._budget.failures.append({"code": "PROMPT_USAGE_MISMATCH", "call_id": result["call_id"]})
                raise ValueError("Generation usage differs from checked prompt count")
            self._budget.check(stage="after_generation_validation")
            return result
        finally:
            self._budget.persist_preserving_error()
