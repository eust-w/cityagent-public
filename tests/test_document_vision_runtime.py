"""Factory/CLI contract tests with a named admission stub; no processor/model.

Use candidate/code/src in PYTHONPATH. These tests make no sys.path edits, use
only temporary synthetic files, and never claim to validate the real native
processor constructor (covered by its own audit tests).
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bridge_evidence.contest import document_cli as cli
from bridge_evidence.contest import document_multimodal_admission as native
from bridge_evidence.contest import document_vision_runtime as runtime
from bridge_evidence.contest.document_vision_backend import MODEL, OwnModelVisionBackend
from bridge_evidence.contest.document_workflow import DocumentSettings
from bridge_evidence.contest.run_state import file_digest, json_digest


NATIVE_IDENTITY = {"model": MODEL, "max_model_len": 65536,
                   "processor_identity_sha256": "a" * 64,
                   "service_identity_sha256": "b" * 64}
MODEL_IDENTITY = {"adapter": {"weights_sha256": "c" * 64, "config_sha256": "d" * 64},
                  "deployment": {"kind": "dynamic_lora", "adapter_path": "/synthetic/adapter",
                                 "runtime_weights_checksum_verified": False}}


class StubNativeAdmission:
    """In-memory identity only: never creates dirs or executes the processor."""
    instances = []
    next_identity = None

    def __init__(self, directory, **options):
        self.directory, self.options = Path(directory), deepcopy(options)
        self.identity = deepcopy(self.next_identity or NATIVE_IDENTITY)
        self.verify_calls = 0
        self.instances.append(self)

    def verify(self):
        self.verify_calls += 1

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("The processor must never be called in factory/CLI contract tests")


class DocumentVisionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.config = self.root / "vision.json"
        self.value = {"processor_python": str(self.root / "synthetic-python"),
                      "path_audit": str(self.root / "synthetic-audit.json"),
                      "runtime_environment": {"PATH": "/synthetic/bin", "LD_LIBRARY_PATH": "/synthetic/lib"}}
        self.config.write_text(json.dumps(self.value))
        Path(self.value["path_audit"]).write_text(json.dumps({"service_process": {
            "cmdline": ["vllm", "serve", "/synthetic/model", "--host", "127.0.0.1", "--port", "8000"]}}))
        StubNativeAdmission.instances = []
        StubNativeAdmission.next_identity = None
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(native, "NativeMultimodalAdmission", StubNativeAdmission))
        self.no_subprocess = stack.enter_context(patch.object(
            subprocess, "run", side_effect=AssertionError("No native subprocess permitted")))
        self.no_popen = stack.enter_context(patch.object(
            subprocess, "Popen", side_effect=AssertionError("No process launch permitted")))
        self.no_network = stack.enter_context(patch.object(
            socket, "create_connection", side_effect=AssertionError("No HTTP/model connection permitted")))

    def factory(self, **changes):
        options = dict(model=MODEL, model_identity=deepcopy(MODEL_IDENTITY), base_url="http://127.0.0.1:8000")
        options.update(changes)
        return runtime.OwnModelVisionFactory(self.config, **options)

    def args(self, **changes):
        values = dict(document_vision="own-model", document_vision_config=self.config,
                      document_ocr="none", served_model=MODEL,
                      document_model_identity=deepcopy(MODEL_IDENTITY), resume=False,
                      work_dir=self.root / "work", result_dir=self.root / "result",
                      model_timeout=120, document_total_report_timeout=None)
        values["vllm_url"] = "http://127.0.0.1:8000"
        values.update(changes)
        return SimpleNamespace(**values)

    def test_factory_preflight_reads_identity_without_calling_backend_or_processor(self):
        before = set(self.root.iterdir())
        factory = self.factory()
        factory.verify()
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(len(StubNativeAdmission.instances), 1)
        probe = StubNativeAdmission.instances[0]
        self.assertGreaterEqual(probe.verify_calls, 2)
        self.assertFalse(probe.directory.exists())
        self.assertEqual(factory.identity["status"], "configured_not_executed")
        self.assertFalse(factory.identity["external_vision_api"])
        self.assertFalse(factory.identity["real_generation_verified"])
        self.assertEqual(factory.identity["configuration_sha256"], file_digest(self.config))
        self.no_subprocess.assert_not_called()
        self.no_popen.assert_not_called()
        self.no_network.assert_not_called()

    def test_wrong_model_or_missing_deployment_identity_refuses_construction(self):
        for changes in ({"model": "another-model"}, {"model_identity": {}}, {"model_identity": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.factory(**changes)
        self.assertEqual(StubNativeAdmission.instances, [])

    def test_model_and_configuration_bytes_are_independently_bound(self):
        first = self.factory()
        changed_model = deepcopy(MODEL_IDENTITY)
        changed_model["adapter"]["weights_sha256"] = "e" * 64
        second = self.factory(model_identity=changed_model)
        self.assertNotEqual(first.identity["model_identity_sha256"], second.identity["model_identity_sha256"])
        self.assertEqual(first.identity["configuration_sha256"], second.identity["configuration_sha256"])
        self.value["runtime_environment"]["PATH"] = "/another/synthetic/bin"
        self.config.write_text(json.dumps(self.value))
        third = self.factory()
        self.assertNotEqual(first.identity["configuration_sha256"], third.identity["configuration_sha256"])
        self.assertEqual(first.identity["model_identity_sha256"], third.identity["model_identity_sha256"])

    def test_malformed_or_implicit_configuration_rejected_before_admission(self):
        relative = deepcopy(self.value); relative["path_audit"] = "relative.json"
        extra = {**self.value, "external_api": "not-allowed"}
        missing_env = deepcopy(self.value); del missing_env["runtime_environment"]["LD_LIBRARY_PATH"]
        empty_env = deepcopy(self.value); empty_env["runtime_environment"]["PATH"] = ""
        raws = [json.dumps(value) for value in (relative, extra, missing_env, empty_env)]
        raws.extend(('{"processor_python":"/a","processor_python":"/b"}', '{"value":NaN}'))
        for raw in raws:
            with self.subTest(raw=raw):
                self.config.write_text(raw)
                with self.assertRaises(ValueError):
                    self.factory()
        self.assertEqual(StubNativeAdmission.instances, [])

    def test_configuration_edit_stops_existing_factory_before_backend_creation(self):
        factory = self.factory()
        self.config.write_text(json.dumps(self.value, indent=2))  # Even same values, different bound bytes.
        generate = Mock()
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            factory(generate_observation=generate, deadline=time.monotonic() + 10,
                    directory=self.root / "worker", manifest={"document_sha256": "f" * 64})
        self.assertEqual(len(StubNativeAdmission.instances), 1)
        generate.assert_not_called()
        self.assertFalse((self.root / "worker").exists())

    def test_preflight_native_identity_change_stops_existing_factory(self):
        factory = self.factory()
        StubNativeAdmission.instances[0].identity["service_identity_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "Native processor identity changed"):
            factory.verify()

    def test_worker_native_identity_must_equal_preflight_before_backend_creation(self):
        factory = self.factory()
        StubNativeAdmission.next_identity = {**NATIVE_IDENTITY, "processor_identity_sha256": "e" * 64}
        with self.assertRaisesRegex(ValueError, "worker identity differs"):
            factory(generate_observation=Mock(), deadline=time.monotonic() + 10,
                    directory=self.root / "worker", manifest={"document_sha256": "f" * 64})
        self.assertEqual(len(StubNativeAdmission.instances), 2)
        self.assertFalse((self.root / "worker").exists())

    def test_created_backend_binds_callback_document_deadline_and_own_model_without_running(self):
        factory = self.factory()
        generate = Mock(side_effect=AssertionError("No generation during construction"))
        deadline = time.monotonic() + 10
        backend, admission = factory(generate_observation=generate, deadline=deadline,
                                     directory=self.root / "worker", manifest={"document_sha256": "f" * 64})
        self.assertIsInstance(backend, OwnModelVisionBackend)
        self.assertIs(backend._generate, generate)
        self.assertEqual(backend.document_sha256, "f" * 64)
        self.assertEqual(backend.deadline, deadline)
        self.assertEqual(backend.output_tokens, 512)
        self.assertEqual(backend._identity["model_identity_sha256"], json_digest(MODEL_IDENTITY))
        self.assertEqual(admission.directory, self.root / "worker/admission")
        generate.assert_not_called()
        self.no_network.assert_not_called()

    def test_cli_flags_require_explicit_selection_and_keep_vision_disabled_by_default(self):
        parser = argparse.ArgumentParser()
        cli.add_arguments(parser)
        defaults = parser.parse_args([])
        self.assertEqual(defaults.document_vision, "none")
        self.assertIsNone(defaults.document_vision_config)
        chosen = parser.parse_args(["--document-vision", "own-model", "--document-vision-config", str(self.config)])
        self.assertEqual(chosen.document_vision_config, self.config)
        with patch.object(cli, "_make_ocr_backend") as ocr, \
                self.assertRaisesRegex(ValueError, "explicit"):
            cli._make_vision_backend(self.args(document_vision="none"))
        ocr.assert_not_called()
        self.assertEqual(StubNativeAdmission.instances, [])

    def test_cli_rejects_missing_config_or_mixed_cpu_ocr_before_any_construction(self):
        for args in (self.args(document_vision_config=None), self.args(document_ocr="tesseract")):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "CPU OCR disabled"):
                cli._make_vision_backend(args)
        self.assertEqual(StubNativeAdmission.instances, [])

    def test_cli_factory_identity_is_copied_and_disabled_mode_keeps_existing_ocr(self):
        backend, factory, identity = cli._make_vision_backend(self.args())
        self.assertIsNone(backend)
        identity["native_admission"]["service_identity_sha256"] = "e" * 64
        self.assertEqual(factory.identity["native_admission"], NATIVE_IDENTITY)
        ocr = object()
        with patch.object(cli, "_make_ocr_backend", return_value=(ocr, {"backend": "tesseract"})) as create:
            actual = cli._make_vision_backend(self.args(document_vision="none", document_vision_config=None))
        self.assertEqual(actual, (ocr, None, {"backend": "tesseract"}))
        create.assert_called_once()

    def worker_setup(self):
        args = self.args()
        args.result_dir.mkdir()
        original = self.root / "synthetic-source.json"
        original.write_text('{"synthetic":"source identity only, no report or labels"}')
        source_hash = hashlib.sha256(original.read_bytes()).hexdigest()
        _, _, args.document_vision_identity = cli._make_vision_backend(args)
        settings = DocumentSettings(output_tokens=4096, session_iterations=8, session_calls=12)
        return args, original, source_hash, settings

    def test_cli_worker_refuses_changed_preflight_identity_before_client_or_workflow(self):
        args, original, source_hash, settings = self.worker_setup()
        StubNativeAdmission.next_identity = {**NATIVE_IDENTITY, "service_identity_sha256": "e" * 64}
        with patch.object(cli.divide_cli, "_make_client") as make_client, \
                patch.object(cli, "_workflow_types") as workflow_types:
            outcome = cli._run_document("synthetic", {"document_sha256": "f" * 64}, None,
                                        original, args, settings, "9" * 64, source_hash, Mock())
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("Image backend identity changed", outcome["error"])
        make_client.assert_not_called()
        workflow_types.assert_not_called()
        self.assertTrue(list((args.work_dir / "documents/synthetic/attempts").glob("*.failure.json")))

    def test_cli_passes_bound_factory_into_workflow_and_verifies_it_after_run(self):
        args, original, source_hash, settings = self.worker_setup()
        constructions = []
        configurations = []
        class FakeWorkflow:
            def __init__(self, *_args, **kwargs):
                constructions.append(kwargs)
                configurations.append(_args[2])
            def run(self):
                return {"data": {"synthetic": True}, "receipt": {"status": "completed"}}
        client = SimpleNamespace(base_url="http://127.0.0.1:1", api_key="synthetic", calls=[])
        with patch.object(cli.divide_cli, "_make_client", return_value=client), \
                patch.object(cli, "_workflow_types", return_value=(object, FakeWorkflow)), \
                patch.object(cli, "_bind_source"), patch.object(cli, "_publish", return_value="8" * 64):
            outcome = cli._run_document("synthetic", {"document_sha256": "f" * 64}, None,
                                        original, args, settings, "9" * 64, source_hash, Mock())
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(len(constructions), 1)
        self.assertEqual(configurations[0]["enable_thinking"], settings.enable_thinking)
        self.assertEqual(configurations[0]["temperature"], settings.temperature)
        options = constructions[0]
        self.assertIsNone(options["vision_backend"])
        self.assertIsInstance(options["vision_backend_factory"], runtime.OwnModelVisionFactory)
        self.assertEqual(options["vision_backend_factory"].identity, args.document_vision_identity)
        self.assertEqual(options["run_identity"], {"run_fingerprint": "9" * 64, "source_sha256": source_hash})
        self.assertGreaterEqual(StubNativeAdmission.instances[-1].verify_calls, 2)
        self.no_network.assert_not_called()
        self.no_subprocess.assert_not_called()


if __name__ == "__main__":
    unittest.main()
