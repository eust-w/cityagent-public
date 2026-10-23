"""Endpoint plumbing with the real APIBackend and explicitly synthetic transport.

These tests do not load a model, run cloud inference or certify RSI acceptance.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from bridge_evidence.contest.backends import APIBackend

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("benchmark_rsi_process_recovery", "benchmark_rsi_recovery",
           "probe_rsi_parallel_sdk", "propose_rsi_strategy")


def driver(name):
    spec = importlib.util.spec_from_file_location("endpoint_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def required_arguments(name, tmp_path):
    names = {
        "benchmark_rsi_process_recovery": ("output-dir", "model-manifest", "bundle", "sandbox-root"),
        "benchmark_rsi_recovery": ("output-dir", "model-manifest", "bundle", "sandbox-root"),
        "probe_rsi_parallel_sdk": ("output-dir", "model-manifest", "sdk-acceptance"),
        "propose_rsi_strategy": ("parent-bundle", "model-manifest", "sandbox-root", "records-dir", "output-dir"),
    }[name]
    return [value for name in names for value in ("--" + name, str(tmp_path / name))]


@pytest.mark.parametrize("name", SCRIPTS)
@pytest.mark.parametrize("explicit", [None, "http://127.0.0.1:8011/v1/"])
def test_cli_endpoint_reaches_real_api_backend_transport(name, explicit, tmp_path, monkeypatch):
    module = driver(name)
    arguments = required_arguments(name, tmp_path)
    if explicit:
        arguments += ["--url", explicit]
    args = module.build_parser().parse_args(arguments)
    backend = module.create_backend(args, SimpleNamespace(served_model="fixed-test-lora"), object())
    assert type(backend) is APIBackend
    expected = "http://127.0.0.1:8011/v1" if explicit else "http://127.0.0.1:8000/v1"
    requests = []

    class FixtureOpener:
        def open(self, request, *, timeout):
            requests.append((request.full_url, json.loads(request.data), timeout))
            return io.BytesIO(json.dumps({"id": "test-transport-only", "choices": [
                {"text": "synthetic fixture", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}}).encode())

    # Only the wire transport is synthetic; CLI parsing, factory, APIBackend's
    # request construction, dispatch and response parsing all run unchanged.
    from bridge_evidence.contest import backends
    monkeypatch.setattr(backends.urllib.request, "build_opener", lambda *args: FixtureOpener())
    result = backend([11, 12], 4)
    assert backend.url == args.url == expected
    assert requests[0][0] == expected + "/completions"
    assert requests[0][1] == {"model": "fixed-test-lora", "prompt": [11, 12], "max_tokens": 4,
                              "temperature": 0.0, "seed": 17, "stream": False}
    assert requests[0][2] == (120 if name == "propose_rsi_strategy" else 180)
    assert result["backend_request_id"] == "test-transport-only"


@pytest.mark.parametrize("name", SCRIPTS)
def test_cli_rejects_non_loopback_before_loading_models(name, tmp_path):
    with pytest.raises(SystemExit):
        driver(name).build_parser().parse_args(required_arguments(name, tmp_path) + ["--url", "https://remote.invalid/v1"])


def test_process_runtime_preserves_non_default_endpoint(tmp_path, monkeypatch):
    module = driver("benchmark_rsi_process_recovery")
    model = SimpleNamespace(tokenizer="fixed-tokenizer", served_model="fixed-test-lora")
    model.validate = lambda: model
    model.identity = lambda: {"test_only": True}
    monkeypatch.setattr(module.ModelManifest, "load", lambda path: model)
    token = object()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: token)))
    args = SimpleNamespace(model_manifest=tmp_path / "model.json", url="http://127.0.0.1:8011/v1/")
    tokenizer, backend, identity = module.runtime(args)
    assert tokenizer is token
    assert type(backend) is APIBackend
    assert backend.url == "http://127.0.0.1:8011/v1"
    assert identity == {"test_only": True}


def test_process_supervisor_forwards_endpoint_to_its_real_child_command(tmp_path, monkeypatch):
    module = driver("benchmark_rsi_process_recovery")
    args = module.build_parser().parse_args(required_arguments("benchmark_rsi_process_recovery", tmp_path)
                                            + ["--url", "http://127.0.0.1:8011/v1"])
    commands = []

    class StopBeforeLaunch(Exception):
        pass

    def observe(command, **kwargs):
        commands.append(command)
        raise StopBeforeLaunch

    monkeypatch.setattr(module.subprocess, "Popen", observe)
    with pytest.raises(StopBeforeLaunch):
        module.run_process_recovery(args)
    assert commands[0][commands[0].index("--url") + 1] == args.url


def test_parallel_probe_binds_endpoint_through_the_actual_run_parser(tmp_path):
    module = driver("probe_rsi_parallel_sdk")
    spec = importlib.util.spec_from_file_location("endpoint_contest_entry", ROOT / "run.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    args = module.build_parser().parse_args(required_arguments("probe_rsi_parallel_sdk", tmp_path)
                                            + ["--url", "http://127.0.0.1:8011/v1/"])
    backend = module.create_backend(args, SimpleNamespace(served_model="test-only"), None)
    runtime = module.create_workflow_runtime(entry, args, tmp_path / "inputs", tmp_path / "work", backend)
    assert runtime.backend_url == backend.url == "http://127.0.0.1:8011/v1"
    assert runtime.sdk_acceptance == args.sdk_acceptance
    assert runtime.engine == "tongagents"


def test_all_factories_send_to_an_actual_non_default_loopback_server(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            raw = json.dumps({"id": "test-local-http-only", "choices": [{
                "text": "synthetic response, not model evidence", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    # An ephemeral port cannot accidentally call an existing 8000/8011 service.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        for name in SCRIPTS:
            module = driver(name)
            args = module.build_parser().parse_args(required_arguments(name, tmp_path) + ["--url", endpoint])
            backend = module.create_backend(args, SimpleNamespace(served_model="test-only"), object())
            assert backend([1, 2], 3)["backend_request_id"] == "test-local-http-only"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert len(requests) == 4
    assert all(path == "/v1/completions" and payload["model"] == "test-only" for path, payload in requests)


@pytest.mark.parametrize("changed_endpoint", [False, True])
def test_process_resume_checks_endpoint_before_checkpoint_or_model_reuse(tmp_path, monkeypatch, changed_endpoint):
    module = driver("benchmark_rsi_process_recovery")
    from bridge_evidence.contest.rsi import StrategyBundle
    from bridge_evidence.contest.rsi_acceptance import RECOVERY_TEXT

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps(StrategyBundle.baseline().as_dict()))
    output = tmp_path / "run"
    output.mkdir()
    (output / "source.txt").write_text(RECOVERY_TEXT)
    args = SimpleNamespace(output_dir=output, worker_stage="crash-window", bundle=bundle,
                           mode="off", sandbox_root=None, url="http://127.0.0.1:8000/v1")
    entered = []

    def stop_before_inference(*args, **kwargs):
        entered.append(True)
        raise InterruptedError("test stops before inference")

    monkeypatch.setattr(module.ci, "infer_chunked", stop_before_inference)
    # worker temporarily installs its real crash-window hook; restore it after
    # the test even though no inference or parser invocation occurs here.
    monkeypatch.setattr(module.ci, "parse_fragment", module.ci.parse_fragment)

    def runtime(arguments):
        return object(), APIBackend(arguments.url, "test-only", None), {"test_only": True}

    with pytest.raises(InterruptedError):
        module.worker(args, runtime_factory=runtime)
    original = json.loads((output / "crash-window.identity.json").read_text())
    assert original["url"] == "http://127.0.0.1:8000/v1"
    args.worker_stage = "resume"
    if changed_endpoint:
        args.url = "http://127.0.0.1:8011/v1"
        with pytest.raises(ValueError, match="endpoint changed before checkpoint recovery"):
            module.worker(args, runtime_factory=runtime)
        assert len(entered) == 1
    else:
        with pytest.raises(InterruptedError):
            module.worker(args, runtime_factory=runtime)
        assert len(entered) == 2
    assert not (output / "result.json").exists()
