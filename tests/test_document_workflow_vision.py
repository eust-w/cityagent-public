"""Cross-module synthetic vision integration, never a real model/processor.

The HTTP responder and CPU admission are deliberate fakes; workflow, gateway,
image backend, tools, workspace, journal and sidecar persistence are real.
Only generated PNG/ZIP/text fixtures are used. Set candidate/code/src through
PYTHONPATH; do not import the current repository's implementation or helpers.
"""
import base64
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image
from bridge_evidence.contest import document_workflow as workflow
from bridge_evidence.contest.document_vision_backend import MODEL, OwnModelVisionBackend
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.local_model_client import LocalModelClient
from bridge_evidence.contest.run_state import json_digest
from test_document_workflow import install_native_review_driver


NATIVE_IDENTITY = {"model": MODEL, "max_model_len": 65536,
                   "processor_identity_sha256": "a" * 64, "service_identity_sha256": "b" * 64}


class FakeAdmission:
    """Constant synthetic count, with actual request/image hashes; no processor."""
    identity = NATIVE_IDENTITY

    def __init__(self):
        self.calls = []

    def __call__(self, payload, *, deadline):
        self.calls.append((deepcopy(payload), deadline))
        image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
        blob = base64.b64decode(image_url.split(",", 1)[1])
        return {"status": "processor_counted", "method": "installed_vllm_native_processor",
                "request_sha256": json_digest(payload), "expanded_prompt_tokens": 662,
                "max_model_len": 65536, "reserved_output_tokens": 512,
                "final_prompt_token_ids_sha256": "c" * 64,
                "image_payload_sha256": hashlib.sha256(blob).hexdigest(), **NATIVE_IDENTITY}


class LegacyOCR:
    """Synthetic legacy text observations; intentionally no model call."""
    model, version, cache_key = "synthetic-ocr", "synthetic-v1", "synthetic-legacy-ocr"

    def __call__(self, _blob, *, question, **_kwargs):
        return {"text": "合成旧OCR文字", "question": question, "verification": "unverified"}


class Factory:
    def __init__(self, legacy=False):
        self.identity = {"backend": "synthetic-contract-factory", "native": NATIVE_IDENTITY,
                         "legacy_ocr_fixture": legacy}
        self.admission = FakeAdmission()
        self.legacy, self.arguments = legacy, None

    def __call__(self, **arguments):
        self.arguments = arguments
        backend = LegacyOCR() if self.legacy else OwnModelVisionBackend(
            arguments["generate_observation"],
            identity={**NATIVE_IDENTITY, "model_identity_sha256": "d" * 64},
            document_sha256=arguments["manifest"]["document_sha256"],
            deadline=arguments["deadline"], output_tokens=512)
        self.backend = backend
        return backend, self.admission


@contextmanager
def synthetic_http(*, invalid_image=False):
    """Loopback protocol double: no proxy, remote address, weights or SDK."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            if self.path == "/tokenize":
                response = {"count": 65}
            elif self.path == "/v1/chat/completions":
                image = isinstance(payload["messages"][-1]["content"], list)
                if image:
                    content = ({"wrong_schema": "synthetic invalid observation"} if invalid_image else
                               {"visible_content": ["合成灰色方块"], "readable_text": [],
                                "uncertainties": ["合成测试，不代表真实图像识别"]})
                else:
                    assignment = json.loads(payload["messages"][1]["content"])["review_assignment"]
                    content = {"passed": True, "checked_assignment_id": assignment["id"],
                               "checked_all_assigned": True, "unsupported_claims": [],
                               "missing_items": [], "conflicts": [], "unresolved_items": []}
                prompt = 662 if image else 65
                response = {"id": "synthetic-response", "model": MODEL,
                            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                                "role": "assistant", "content": json.dumps(content, ensure_ascii=False)}}],
                            "usage": {"prompt_tokens": prompt, "completion_tokens": 7,
                                      "total_tokens": prompt + 7}}
            else:
                self.send_error(404)
                return
            body = json.dumps(response, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield LocalModelClient(f"http://127.0.0.1:{server.server_port}", MODEL,
                               max_model_len=65536, timeout=120), requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


class DocumentWorkflowVisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        blob = io.BytesIO()
        Image.new("RGB", (8, 6), (90, 90, 90)).save(blob, format="PNG")
        self.png = blob.getvalue()
        self.source = self.root / "synthetic.docx"
        # Minimal ZIP fixture exercises the real image extraction/hash path;
        # manifest text is synthetic and is not parsed from this tiny archive.
        with zipfile.ZipFile(self.source, "w") as archive:
            archive.writestr("word/media/image1.png", self.png)
        self.manifest = manifest_from_text("桥梁名称：合成桥。\n")
        self.manifest.update(source_kind="docx", document_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.image = {"id": "I000001", "source_path": "word/document.xml#synthetic-image-1",
                      "target": "media/image1.png", "relationship_id": "rIdSynthetic"}
        self.manifest["images"] = [self.image]

    def make(self, client, *, max_calls=30, legacy=False):
        factory = Factory(legacy=legacy)
        wf = workflow.DocumentWorkflow(
            deepcopy(self.manifest), client, {"model": MODEL}, self.root / "work",
            settings=workflow.DocumentSettings(max_model_calls=max_calls, max_sessions=30,
                                               report_timeout=60, max_review_attempts=3),
            source_path=self.source, vision_backend_factory=factory,
            react_runner=lambda *_args: (_ for _ in ()).throw(AssertionError("No SDK runner in these tests")))
        return install_native_review_driver(wf, synthetic_transport=False), factory

    def observe(self, wf, question="直接观察合成图一"):
        return wf.tools.read_image(self.image["id"], question, "{}", [])

    def seed_draft(self, wf, source_ref=None):
        unit = wf.manifest["units"][0]
        wf.workspace.record_read(unit["id"], 0, len(unit["text"]))
        ref = source_ref or {"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}
        wf.workspace.apply_update({"base_version": wf.workspace.snapshot()["version"],
            "facts": [{"id": "Fname", "value": "合成桥", "status": "explicit", "source_refs": [ref]}],
            "draft": {"基本信息": {"桥梁名称": "合成桥"}, "病害明细": [], "维修建议": []},
            "draft_fact_ids": ["Fname"], "draft_bindings": {"基本信息.桥梁名称": ["Fname"]}})

    @staticmethod
    def generations(requests):
        return [payload for path, payload in requests if path == "/v1/chat/completions"]

    def test_factory_image_tool_persists_real_http_dispatch_request_response_and_observation(self):
        with synthetic_http() as (client, requests):
            wf, factory = self.make(client)
            result = self.observe(wf)
            self.assertTrue(result["ok"], result)
            self.assertEqual(factory.arguments["deadline"], wf.deadline)
            self.assertEqual(wf.identity["vision_backend"], factory.identity)
            self.assertTrue(result["backend"]["requires_observation_id"])
            self.assertEqual(len(factory.admission.calls), 1)
            self.assertEqual(len(self.generations(requests)), 1)
            call = next(row for row in client.calls if row["kind"] == "generation")
            dispatch = wf.model_dispatches[0]
            self.assertEqual((dispatch["status"], dispatch["call_id"]), ("settled", call["call_id"]))
            self.assertEqual(call["purpose"], "image_observation")
            self.assertEqual(call["prompt_tokens_checked"], 662)
            sent = json.loads((wf.directory / "model_requests" / (dispatch["request_id"] + ".json")).read_text())
            responses = list((wf.directory / "model_responses").glob("*.json"))
            self.assertEqual(len(responses), 1)
            response = json.loads(responses[0].read_text())
            self.assertEqual(sent["request_sha256"], json_digest(self.generations(requests)[0]))
            self.assertEqual(sent["request"], self.generations(requests)[0])
            self.assertEqual(response["call_id"], call["call_id"])
            self.assertEqual(response["response_sha256"], json_digest(response["response"]))
            saved = wf.workspace.load_image_observation(self.image["id"], result["observation_id"])
            self.assertEqual(saved["observation_sha256"], result["observation_sha256"])
            self.assertEqual(wf.workspace.snapshot()["image_reads"], [self.image["id"]])
            self.assertFalse(wf.workspace.delivery_status()["ready"])

    def test_transient_terminal_persistence_failure_cannot_leave_successful_dispatch(self):
        with synthetic_http() as (client, requests):
            wf, _ = self.make(client)
            original = wf._persist
            failed = []
            def fail_terminal_once(*args, **kwargs):
                if not failed and wf.gateway_receipts and wf.gateway_receipts[-1].get("status") == "success":
                    failed.append(True)
                    raise OSError("synthetic terminal disk failure")
                return original(*args, **kwargs)
            with patch.object(wf, "_persist", side_effect=fail_terminal_once):
                result = wf._guard(lambda: self.observe(wf))()
            self.assertFalse(result["ok"])
            self.assertEqual(len(self.generations(requests)), 1)
            self.assertEqual(wf.workspace.snapshot()["image_reads"], [])
            trace = json.loads(wf.trace_path.read_text())
            self.assertEqual(trace["gateway_receipts"][-1]["failure_stage"], "receipt_callback")
            self.assertEqual(trace["model_dispatches"][-1]["receipt_status"], "rejected")
            self.assertEqual(trace["model_dispatches"][-1]["status"], "settled")
            resumed, _ = self.make(client)
            self.assertEqual(resumed.model_dispatches[-1]["receipt_status"], "rejected")

    def test_exact_offline_bpe_rejects_long_image_reply_before_generation_then_recovers(self):
        tokenizer_dir = os.environ.get("CITYAGENT_TEST_TOKENIZER_DIR")
        if not tokenizer_dir:
            self.skipTest("Set CITYAGENT_TEST_TOKENIZER_DIR for the optional offline tokenizer test")
        tokenizer_path = Path(tokenizer_dir) / "tokenizer.json"
        if not tokenizer_path.is_file():
            self.skipTest("Explicit offline tokenizer fixture unavailable")
        self.assertEqual(hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                         "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        synthetic_text = manifest_from_text("桥梁名称：合成桥。\n" + "检查未见异常。" * 700)
        self.manifest["units"] = synthetic_text["units"]
        with synthetic_http() as (client, requests):
            wf, factory = self.make(client)
            wf._count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
            wf._tool_configuration["count_tokens"] = wf._count_tokens
            context_id = self.manifest["units"][-1]["id"]
            denied = wf.tools.read_image(self.image["id"], "核对合成图片", "{}", [context_id])
            self.assertFalse(denied["ok"])
            self.assertIn("before generation", denied["error"]["message"])
            self.assertEqual(len(self.generations(requests)), 0)
            self.assertEqual(factory.admission.calls, [])
            self.assertEqual(wf.workspace.snapshot()["image_reads"], [])
            self.assertEqual(wf.tools._judgment_cache, {})
            recovered = wf.tools.read_image(self.image["id"], "核对合成图片", "{}", [])
            self.assertTrue(recovered["ok"], recovered)
            self.assertEqual(len(self.generations(requests)), 1)
            self.assertLessEqual(wf.tools.receipts[-1]["result_tokens"], 4096)

    def test_images_share_report_generation_budget_and_expired_deadline_cannot_send_again(self):
        with synthetic_http() as (client, requests):
            wf, factory = self.make(client, max_calls=1)
            self.assertTrue(self.observe(wf)["ok"])
            denied = self.observe(wf, "另一项合成观察")
            self.assertFalse(denied["ok"])
            self.assertEqual(denied["error"]["code"], "VISION_CALLBACK_FAILED")
            with self.assertRaisesRegex(RuntimeError, "model request budget exhausted"):
                wf._generate_observation(factory.admission.calls[0][0], provenance={}, deadline=wf.deadline)
            self.assertEqual(len(factory.admission.calls), 1)
            self.assertEqual(len(self.generations(requests)), 1)
            with patch.object(workflow, "time", SimpleNamespace(monotonic=lambda: wf.deadline + 1)):
                with self.assertRaises(TimeoutError):
                    wf._generate_observation(factory.admission.calls[0][0], provenance={}, deadline=wf.deadline)
            self.assertEqual(len(self.generations(requests)), 1)

    def test_image_and_review_calls_have_separate_purpose_totals_but_one_shared_limit(self):
        with synthetic_http() as (client, requests):
            wf, _ = self.make(client, max_calls=2)
            self.assertTrue(self.observe(wf)["ok"])
            self.seed_draft(wf)
            wf._active_review_items = wf._output_items(wf.workspace.snapshot())
            decision = wf._review_request({"direction": "FORWARD", "output_items": wf._active_review_items, "source_spans": []})
            self.assertTrue(decision["passed"])
            self.assertFalse(self.observe(wf, "第三个请求必须被拒绝")["ok"])
            receipt = wf._receipt("synthetic-partial", "navigation")
            self.assertEqual(receipt["image_generation_requests"], 1)
            self.assertEqual(receipt["review_generation_requests"], 1)
            self.assertEqual(receipt["agent_generation_requests"], 0)
            self.assertEqual(len(self.generations(requests)), 2)
            self.assertTrue(all(row["status"] == "settled" for row in receipt["model_dispatches"]))

    def test_invalid_observation_schema_keeps_sent_call_but_grants_no_image_read(self):
        with synthetic_http(invalid_image=True) as (client, requests):
            wf, _ = self.make(client)
            denied = self.observe(wf)
            self.assertFalse(denied["ok"])
            snapshot = wf.workspace.snapshot()
            self.assertEqual(snapshot["image_reads"], [])
            self.assertEqual(snapshot["image_observations"], {})
            self.assertEqual(wf.tools.image_observations, [])
            self.assertEqual(len(self.generations(requests)), 1)
            self.assertEqual(wf._receipt("failed", "navigation")["image_generation_requests"], 1)
            self.assertEqual(len(list((wf.directory / "model_responses").glob("*.json"))), 1)

    def test_new_observation_changes_semantic_identity_and_invalidates_journal_review_and_finish(self):
        with synthetic_http() as (client, requests):
            wf, _ = self.make(client)
            first = self.observe(wf)
            self.assertTrue(first["ok"], first)
            self.seed_draft(wf)
            old_semantic = wf._semantic_identity(wf.workspace.snapshot())
            payload = {"direction": "FORWARD", "output_items": wf._output_items(wf.workspace.snapshot()), "source_spans": []}
            wf._perform_review_draft()
            self.assertTrue(wf.workspace.delivery_status()["ready"])
            self.assertTrue(wf._finish()["ok"])
            self.assertTrue(wf._finish_is_current())
            self.assertIsNotNone(wf.review_journal.get(old_semantic, payload))
            before = len(self.generations(requests))
            second = self.observe(wf, "同一张图的另一个问题")
            self.assertTrue(second["ok"], second)
            self.assertNotEqual(first["observation_id"], second["observation_id"])
            current = wf.workspace.snapshot()
            new_semantic = wf._semantic_identity(current)
            self.assertNotEqual(old_semantic, new_semantic)
            self.assertIsNone(current["review"])
            self.assertFalse(wf._finish_is_current())
            self.assertFalse(wf._finish()["ok"])
            self.assertIsNone(wf.review_journal.get(new_semantic, payload))
            self.assertIsNotNone(wf.review_journal.get(old_semantic, payload))
            self.assertEqual(len(self.generations(requests)), before + 1)

    def test_specific_observation_reference_sends_only_that_immutable_observation_to_forward_reviewer(self):
        with synthetic_http() as (client, requests):
            wf, _ = self.make(client)
            first, second = self.observe(wf), self.observe(wf, "新的观察问题")
            self.assertTrue(first["ok"] and second["ok"], (first, second))
            self.seed_draft(wf, {"image_id": self.image["id"], "observation_id": first["observation_id"]})
            items = wf._output_items(wf.workspace.snapshot())
            observations = items[0]["sources"][0]["observations"]
            self.assertEqual([row["observation_id"] for row in observations], [first["observation_id"]])
            wf._active_review_items = items
            wf._review_request({"direction": "FORWARD", "output_items": items, "source_spans": []})
            reviewer = json.loads(self.generations(requests)[-1]["messages"][1]["content"])
            self.assertIn(first["observation_id"], json.dumps(reviewer))
            self.assertNotIn(second["observation_id"], json.dumps(reviewer))

    def test_legacy_bare_image_reference_freezes_delivered_ids_and_cannot_read_later_observation(self):
        with synthetic_http() as (client, requests):
            wf, _ = self.make(client, legacy=True)
            first = self.observe(wf)
            self.assertTrue(first["ok"], first)
            self.seed_draft(wf, {"image_id": self.image["id"]})
            frozen = deepcopy(wf.workspace.snapshot()["facts"]["Fname"]["source_refs"][0])
            self.assertEqual(frozen["observation_ids"], [first["observation_id"]])
            second = self.observe(wf, "稍后的旧OCR问题")
            self.assertTrue(second["ok"], second)
            self.assertNotEqual(first["observation_id"], second["observation_id"])
            snapshot = wf.workspace.snapshot()
            self.assertEqual(snapshot["facts"]["Fname"]["source_refs"][0], frozen)
            observations = wf._output_items(snapshot)[0]["sources"][0]["observations"]
            self.assertEqual([row["observation_id"] for row in observations], [first["observation_id"]])
            self.assertEqual(len(self.generations(requests)), 0)


if __name__ == "__main__":
    unittest.main()
