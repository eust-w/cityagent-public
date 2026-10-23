"""Endpoint identity regression: read-only factory and synthetic audit only.

No native processor, HTTP, model, weights, remote host, or real source report.
The admission stub proves interface binding only, not a live process identity.
"""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock, patch
import unittest

import test_document_vision_runtime as fixture
from bridge_evidence.contest.document_vision_runtime import OwnModelVisionFactory
from bridge_evidence.contest.document_workflow import DocumentSettings, DocumentWorkflow
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.local_model_client import LocalModelClient


class DocumentVisionEndpointTests(unittest.TestCase):
    def setUp(self):
        fixture.DocumentVisionRuntimeTests.setUp(self)
        self.audit = json.loads(Path(self.value["path_audit"]).read_text())
        self.native_calls = Mock(side_effect=AssertionError("Native processor execution forbidden"))
        active = patch.object(fixture.StubNativeAdmission, "__call__", self.native_calls)
        active.start()
        self.addCleanup(active.stop)
        self.addCleanup(self.assert_no_execution)

    def assert_no_execution(self):
        self.native_calls.assert_not_called()
        self.no_network.assert_not_called()
        self.no_subprocess.assert_not_called()
        self.no_popen.assert_not_called()

    def audit_argv(self, *arguments):
        self.audit["service_process"]["cmdline"] = ["/synthetic/vllm", "serve", "/synthetic/model", *arguments]
        Path(self.value["path_audit"]).write_text(json.dumps(self.audit))

    def factory(self, url):
        return OwnModelVisionFactory(self.config, model=fixture.MODEL,
            model_identity=deepcopy(fixture.MODEL_IDENTITY), base_url=url)

    def test_explicit_ipv4_binding_accepts_same_port_and_records_canonical_endpoint(self):
        for url, canonical in (("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
                               ("http://localhost:8000/", "http://localhost:8000")):
            with self.subTest(url=url):
                factory = self.factory(url)
                self.assertEqual(factory.identity["endpoint"], canonical)
                factory.verify_endpoint(canonical)
        self.assertFalse((self.root / ".unused-vision-preflight").exists())

    def test_explicit_ipv6_binding_accepts_ipv6_or_localhost_same_port(self):
        self.audit_argv("--host", "::1", "--port", "8000")
        for url in ("http://[::1]:8000", "http://localhost:8000"):
            with self.subTest(url=url):
                self.assertEqual(self.factory(url).identity["endpoint"], url)
        with self.assertRaises(ValueError):
            self.factory("http://127.0.0.1:8000")

    def test_ipv4_wildcard_maps_only_to_local_ipv4_endpoint(self):
        self.audit_argv("--host", "0.0.0.0", "--port", "8000")
        self.assertEqual(self.factory("http://127.0.0.1:8000").identity["endpoint"], "http://127.0.0.1:8000")
        with self.assertRaises(ValueError):
            self.factory("http://[::1]:8000")

    def test_ipv6_wildcard_maps_only_to_local_ipv6_endpoint(self):
        self.audit_argv("--host", "::", "--port", "8000")
        self.assertEqual(self.factory("http://[::1]:8000").identity["endpoint"], "http://[::1]:8000")
        with self.assertRaises(ValueError):
            self.factory("http://127.0.0.1:8000")

    def test_wrong_or_implicit_client_port_cannot_use_another_local_service(self):
        for url in ("http://127.0.0.1:8001", "http://localhost:8001", "http://127.0.0.1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.factory(url)

    def test_audit_host_and_port_must_each_be_explicit_unique_and_valid(self):
        cases = [("--host", "127.0.0.1"), ("--port", "8000"), (),
                 ("--host", "127.0.0.1", "--host", "127.0.0.1", "--port", "8000"),
                 ("--host", "127.0.0.1", "--port", "8000", "--port", "8000"),
                 ("--host", "127.0.0.1", "--port", "0"),
                 ("--host", "127.0.0.1", "--port", "65536"),
                 ("--host", "192.0.2.10", "--port", "8000")]
        for args in cases:
            with self.subTest(argv=args), self.assertRaises(ValueError):
                self.audit_argv(*args)
                self.factory("http://127.0.0.1:8000")

    def test_credentials_scheme_query_fragment_and_control_characters_are_rejected(self):
        urls = ["https://127.0.0.1:8000", "http://user:secret@127.0.0.1:8000",
                "http://127.0.0.1:8000?x=1", "http://127.0.0.1:8000?",
                "http://127.0.0.1:8000#x", "http://127.0.0.1:8000#",
                "http://127.0.0.1:8000\n", " http://127.0.0.1:8000"]
        for url in urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.factory(url)

    def test_factory_receives_normalized_root_not_extra_proxy_or_v1_paths(self):
        for path in ("/v1", "/proxy", "//", "/../", "/%2e/"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.factory("http://127.0.0.1:8000" + path)

    def test_remote_and_nonloopback_hosts_are_never_generation_endpoints(self):
        for host in ("192.0.2.10", "example.invalid", "127.0.0.1.example.invalid", "0.0.0.0", "[::]"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                self.factory("http://" + host + ":8000")

    def test_factory_rejects_changed_client_at_actual_workflow_image_dispatch(self):
        factory = self.factory("http://127.0.0.1:8000")
        with self.assertRaises(ValueError):
            factory.verify_endpoint("http://127.0.0.1:8001")
        factory.verify_endpoint("http://127.0.0.1:8000")
        client = LocalModelClient("http://127.0.0.1:8001", fixture.MODEL, max_model_len=65536)
        payload = {"model": fixture.MODEL, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Synthetic observation"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,c3ludGhldGlj"}}]}],
            "max_tokens": 512, "temperature": 0, "stream": False, "n": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
        with self.assertRaisesRegex(ValueError, "(?i)endpoint"):
            workflow = DocumentWorkflow(manifest_from_text("Synthetic source"), client,
                {"model": fixture.MODEL}, self.root / "work", settings=DocumentSettings(report_timeout=30),
                vision_backend_factory=factory)
            workflow._generate_observation(payload, provenance={}, deadline=workflow.deadline)
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
