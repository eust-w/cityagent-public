"""Synthetic loopback HTTP only; no real model, processor, SDK or training data.

Import via PYTHONPATH=<candidate>/code/src. This test deliberately never edits
sys.path or imports the current repository's implementation or test helpers.
"""
from contextlib import contextmanager
from copy import deepcopy
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bridge_evidence.contest import document_model_gateway as module
from bridge_evidence.contest.local_model_client import LocalModelClient


MODEL = "synthetic-own-model-alias"
IDENTITY = {"processor_identity_sha256": "a" * 64,
            "service_identity_sha256": "b" * 64}


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(packed(value)).hexdigest()


def image_payload():
    # Deliberately synthetic bytes; the stub does not parse an image or claim
    # that 662 is a processor result from a real model.
    return {"model": MODEL, "max_tokens": 512, "stream": False,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Read visible synthetic marks only."},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,c3ludGhldGljLWltYWdl"}},
            ]}]}


def admission(payload, *, deadline):
    assert isinstance(deadline, (int, float))
    return {"status": "processor_counted", "method": "installed_vllm_native_processor",
            "request_sha256": digest(payload), "expanded_prompt_tokens": 662,
            "max_model_len": 65536, "reserved_output_tokens": 512,
            "final_prompt_token_ids_sha256": "c" * 64,
            "image_payload_sha256": hashlib.sha256(base64.b64decode(
                payload["messages"][0]["content"][1]["image_url"]["url"].split(",", 1)[1],
                validate=True)).hexdigest(), **IDENTITY}


def completion(prompt=662, tokens=7, reason="stop"):
    return {"id": "synthetic-http-response", "model": MODEL,
            "choices": [{"index": 0, "finish_reason": reason,
                         "message": {"role": "assistant", "content": "Synthetic visible marks."}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": tokens,
                      "total_tokens": prompt + tokens}}


@contextmanager
def fake_native_alias(response=None, status=200):
    """An in-process HTTP fixture, not a served model or external API."""
    calls = []
    response = completion() if response is None else response

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, value))
            if self.path == "/tokenize":
                value, code = {"count": 65}, 200
            elif self.path == "/v1/chat/completions":
                value, code = response, status
            else:
                self.send_error(404)
                return
            body = packed(value)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield LocalModelClient(f"http://127.0.0.1:{server.server_port}", MODEL,
                               timeout=120, max_model_len=65536), calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


class DocumentMultimodalGatewayTests(unittest.TestCase):
    def gateway(self, client, **changes):
        options = dict(deadline=time.monotonic() + 10, max_calls=4,
                       max_output_tokens=512, request_kind="generation",
                       generation_purpose="image_observation",
                       request_admission=admission, admission_identity=IDENTITY)
        options.update(changes)
        return module.AuditedModelGateway(client, **options)

    def test_images_never_fall_back_to_text_tokenizer_or_tokenize_http(self):
        for local_template in (False, True):
            with self.subTest(local_template=local_template), fake_native_alias() as (client, calls):
                tokenizer = Mock() if local_template else None
                client.chat_tokenizer = tokenizer
                gateway = self.gateway(client, request_admission=None, admission_identity=None)
                with self.assertRaisesRegex(ValueError, "Multimodal.*native processor"):
                    gateway.forward(image_payload())
                self.assertEqual(calls, [])
                self.assertEqual(client.calls, [])
                if tokenizer is not None:
                    tokenizer.apply_chat_template.assert_not_called()

    def test_native_662_is_used_not_template_65_with_bound_audit_and_purpose(self):
        payload = image_payload()
        dispatched, persisted = [], []
        provider = Mock(side_effect=admission)
        with fake_native_alias() as (client, calls):
            gateway = self.gateway(client, request_admission=provider,
                                   on_dispatch=lambda record: dispatched.append(deepcopy(record)),
                                   on_response=lambda value, audit: persisted.append((value, audit)))
            result = gateway.forward(payload)
            self.assertEqual(result, completion())
            self.assertEqual(calls, [("/v1/chat/completions", payload)])
            self.assertEqual(provider.call_args.kwargs["deadline"], gateway.deadline)
            audit, call = gateway.receipts[0], client.calls[0]
            self.assertEqual(audit["admission"], admission(payload, deadline=gateway.deadline))
            self.assertEqual(audit["prompt_tokens_checked"], 662)
            self.assertEqual(call["prompt_tokens_checked"], 662)
            self.assertEqual(call["usage"], result["usage"])
            self.assertEqual(audit["request_sha256"], digest(payload))
            self.assertEqual(audit["response_sha256"], digest(result))
            self.assertEqual(audit["call_id"], call["call_id"])
            self.assertEqual(dispatched[0]["request_id"], call["request_id"])
            self.assertEqual({row["purpose"] for row in (audit, call, dispatched[0])}, {"image_observation"})
            self.assertEqual(persisted[0][1]["admission"], audit["admission"])
            self.assertEqual(client.timeout, 120)

    def test_text_requests_keep_existing_template_counting_behavior(self):
        payload = image_payload()
        payload["messages"] = [{"role": "user", "content": "Synthetic text."}]
        with fake_native_alias(completion(prompt=65)) as (client, calls):
            gateway = self.gateway(client, request_admission=None, admission_identity=None,
                                   generation_purpose=None)
            gateway.forward(payload)
            self.assertEqual([path for path, _ in calls], ["/tokenize", "/v1/chat/completions"])
            self.assertEqual(gateway.receipts[0]["prompt_tokens_checked"], 65)

    def test_stale_request_hash_rejects_without_any_http(self):
        def stale(payload, *, deadline):
            previous = deepcopy(payload)
            previous["messages"][0]["content"][0]["text"] = "Different question."
            return admission(previous, deadline=deadline)
        with fake_native_alias() as (client, calls):
            gateway = self.gateway(client, request_admission=stale)
            with self.assertRaisesRegex(ValueError, "exact request"):
                gateway.forward(image_payload())
            self.assertEqual((calls, client.calls), ([], []))

    def test_stale_processor_service_or_image_identity_rejects_without_generation(self):
        for identity_key in (*IDENTITY, "image_payload_sha256"):
            def stale(payload, *, deadline):
                result = admission(payload, deadline=deadline)
                result[identity_key] = "e" * 64
                return result
            with self.subTest(identity=identity_key), fake_native_alias() as (client, calls):
                gateway = self.gateway(client, request_admission=stale)
                with self.assertRaises(ValueError):
                    gateway.forward(image_payload())
                self.assertEqual((calls, client.calls), ([], []))

    def test_unusable_native_count_cannot_become_a_budget(self):
        for count in (0, -1, True, 662.0):
            def bad_count(payload, *, deadline):
                return {**admission(payload, deadline=deadline), "expanded_prompt_tokens": count}
            with self.subTest(count=count), fake_native_alias() as (client, calls):
                with self.assertRaises(ValueError):
                    self.gateway(client, request_admission=bad_count).forward(image_payload())
                self.assertEqual((calls, client.calls), ([], []))

    def test_output_reservation_must_match_admission_and_gateway_limit(self):
        payload = image_payload()
        payload["max_tokens"] = 128  # Provider still reserves 512: wrong binding.
        with fake_native_alias() as (client, calls):
            with self.assertRaises(ValueError):
                self.gateway(client).forward(payload)
            oversized = image_payload()
            oversized["max_tokens"] = 513
            provider = Mock(side_effect=admission)
            with self.assertRaisesRegex(ValueError, "output budget"):
                self.gateway(client, request_admission=provider).forward(oversized)
            provider.assert_not_called()
            self.assertEqual((calls, client.calls), ([], []))

    def test_expanded_context_overflow_is_not_hidden_by_small_template_count(self):
        def too_long(payload, *, deadline):
            return {**admission(payload, deadline=deadline), "expanded_prompt_tokens": 65100}
        with fake_native_alias() as (client, calls):
            gateway = self.gateway(client, request_admission=too_long)
            with self.assertRaisesRegex(ValueError, "reserved output exceed context"):
                gateway.forward(image_payload())
            self.assertEqual(gateway.receipts[0]["prompt_tokens_checked"], 65100)
            self.assertEqual((calls, client.calls), ([], []))

    def test_provider_mutating_its_private_copy_is_rejected_and_caller_unchanged(self):
        payload = image_payload()
        original = deepcopy(payload)
        def mutate(value, *, deadline):
            result = admission(value, deadline=deadline)
            value["messages"][0]["content"].pop()
            return result
        with fake_native_alias() as (client, calls):
            with self.assertRaisesRegex(ValueError, "changed its input"):
                self.gateway(client, request_admission=mutate).forward(payload)
            self.assertEqual(payload, original)
            self.assertEqual((calls, client.calls), ([], []))

    def test_native_count_uses_whole_deadline_and_cannot_leave_time_for_generation(self):
        clock = [100.0]
        def expensive_count(payload, *, deadline):
            self.assertEqual(deadline, 101.0)
            clock[0] = 102.0
            return admission(payload, deadline=deadline)
        with fake_native_alias() as (client, calls):
            gateway = self.gateway(client, deadline=101.0, request_admission=expensive_count)
            with patch.object(module, "time", SimpleNamespace(monotonic=lambda: clock[0])), \
                    self.assertRaises(TimeoutError):
                gateway.forward(image_payload())
            self.assertEqual((calls, client.calls), ([], []))
            self.assertEqual(gateway.receipts[0]["admission"]["expanded_prompt_tokens"], 662)
            self.assertEqual(client.timeout, 120)

    def test_actual_sent_prompt_usage_mismatch_retains_failed_http_call(self):
        with fake_native_alias(completion(prompt=65)) as (client, calls):
            gateway = self.gateway(client)
            with self.assertRaisesRegex(ValueError, "actual server usage"):
                gateway.forward(image_payload())
            self.assertEqual(len(calls), 1)
            call, audit = client.calls[0], gateway.receipts[0]
            self.assertEqual(call["status"], "failed")
            self.assertEqual(call["http_status"], 200)
            self.assertEqual(audit["call_id"], call["call_id"])
            self.assertTrue(audit["generation_attempted"])
            self.assertEqual(call["usage"]["prompt_tokens"], 65)
            self.assertEqual(client.timeout, 120)

    def test_native_response_requires_consistent_complete_usage_and_model(self):
        responses = []
        wrong_total = completion(); wrong_total["usage"]["total_tokens"] -= 1
        missing_completion = completion(); del missing_completion["usage"]["completion_tokens"]
        wrong_model = completion(); wrong_model["model"] = "synthetic-base-fallback"
        responses.extend((wrong_total, missing_completion, wrong_model, completion(tokens=513)))
        for result in responses:
            with self.subTest(response=result), fake_native_alias(result) as (client, calls):
                with self.assertRaisesRegex(ValueError, "complete usage"):
                    self.gateway(client).forward(image_payload())
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(client.calls[0]["status"], "failed")
                self.assertEqual(client.calls[0]["usage"], result["usage"])

    def test_truncation_or_tool_call_is_not_a_completed_image_observation(self):
        for reason in ("length", "tool_calls"):
            with self.subTest(reason=reason), fake_native_alias(completion(reason=reason)) as (client, calls):
                gateway = self.gateway(client)
                with self.assertRaises(ValueError):
                    gateway.forward(image_payload())
                self.assertEqual(len(calls), 1)
                self.assertEqual(client.calls[0]["status"], "failed")
                self.assertEqual(client.calls[0]["finish_reason"], reason)
                self.assertEqual(gateway.receipts[0]["finish_reason"], reason)

    def test_upstream_http_failure_keeps_sent_image_purpose_and_request_binding(self):
        with fake_native_alias({"error": "synthetic unavailable"}, status=503) as (client, calls):
            gateway = self.gateway(client)
            with self.assertRaises(RuntimeError):
                gateway.forward(image_payload())
            self.assertEqual(len(calls), 1)
            call, audit = client.calls[0], gateway.receipts[0]
            self.assertEqual(call["status"], "failed")
            self.assertEqual(call["http_status"], 503)
            self.assertEqual(audit["failure_stage"], "generation_http")
            self.assertEqual(audit["call_id"], call["call_id"])
            self.assertEqual(audit["request_id"], call["request_id"])
            self.assertEqual(call["input_sha256"], digest(image_payload()))
            self.assertEqual(call["purpose"], "image_observation")
            self.assertEqual(client.timeout, 120)

    def test_two_gateways_share_lock_wait_counts_toward_deadline_and_restore_timeout(self):
        entered, release = threading.Event(), threading.Event()
        errors = []
        def first_count(payload, *, deadline):
            entered.set()
            if not release.wait(2):
                raise AssertionError("Synthetic count release was not signalled")
            return admission(payload, deadline=deadline)
        with fake_native_alias() as (client, calls):
            shared = threading.Lock()
            first = self.gateway(client, forward_lock=shared, request_admission=first_count)
            def run_first():
                try:
                    first.forward(image_payload())
                except BaseException as exc:
                    errors.append(exc)
            worker = threading.Thread(target=run_first)
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                in_flight_timeout = client.timeout
                self.assertLess(in_flight_timeout, 120)
                second_count = Mock(side_effect=admission)
                second = self.gateway(client, deadline=time.monotonic() + 0.06,
                                      forward_lock=shared, request_admission=second_count)
                with self.assertRaisesRegex(TimeoutError, "waiting"):
                    second.forward(image_payload())
                self.assertGreaterEqual(second.receipts[0]["seconds"], 0.04)
                second_count.assert_not_called()
                self.assertEqual(client.timeout, in_flight_timeout)
                self.assertEqual((calls, client.calls), ([], []))
            finally:
                release.set()
                worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(calls), 1)
            self.assertEqual(client.timeout, 120)
            # Waiting rejection releases neither somebody else's lock nor
            # corrupts the timeout that a later independent request restores.
            third = self.gateway(client, forward_lock=shared)
            third.forward(image_payload())
            self.assertEqual(len(calls), 2)
            self.assertEqual(client.timeout, 120)

    def test_direct_forward_byte_limit_precedes_admission_and_all_http(self):
        payload = image_payload()
        with fake_native_alias() as (client, calls):
            provider = Mock(side_effect=admission)
            gateway = self.gateway(client, max_request_bytes=len(packed(payload)) - 1,
                                   request_admission=provider)
            with self.assertRaisesRegex(ValueError, "byte budget"):
                gateway.forward(payload)
            provider.assert_not_called()
            self.assertEqual((calls, client.calls), ([], []))
            self.assertFalse(gateway.receipts[0].get("generation_attempted", False))
            self.assertEqual(client.timeout, 120)

    def test_persisting_dispatch_cannot_reset_expired_whole_deadline(self):
        clock = [100.0]
        dispatches = []
        def slow_persistence(record):
            dispatches.append(deepcopy(record))
            clock[0] = 102.0
        with fake_native_alias() as (client, calls):
            gateway = self.gateway(client, deadline=101.0, on_dispatch=slow_persistence)
            with patch.object(module, "time", SimpleNamespace(monotonic=lambda: clock[0])), \
                    self.assertRaises(TimeoutError):
                gateway.forward(image_payload())
            self.assertEqual(len(dispatches), 1)
            self.assertEqual((calls, client.calls), ([], []))
            self.assertFalse(gateway.receipts[0]["generation_attempted"])
            self.assertEqual(gateway.receipts[0]["failure_stage"], "before_generation")
            self.assertEqual(client.timeout, 120)


if __name__ == "__main__":
    unittest.main()
