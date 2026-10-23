"""Real loopback HTTP with synthetic fixture replies, never model acceptance."""

import hashlib
import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from bridge_evidence.contest.meta_optimization import CloudBudget
from bridge_evidence.contest.timing import TimingLedger

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def probe(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "fixed_model_vision_probe", ROOT / "scripts/probe_fixed_model_vision.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    budget_path = workspace / "budget.sqlite"
    monkeypatch.setattr(module, "WORKSPACE", workspace)
    monkeypatch.setattr(module, "ORIGINAL_BUDGET", budget_path)
    budget = CloudBudget(budget_path)
    old = budget.start("previous-real-experiment", 100)
    budget.finish(old, 12.5)
    base = workspace / "Qwen3.6-27B"
    base.mkdir()
    (base / "config.json").write_text('{}')
    (base / "tokenizer.json").write_text('{"test_fixture":true}')
    adapter = workspace / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture-weights")
    adapter_sha = hashlib.sha256(b"fixture-weights").hexdigest()
    monkeypatch.setattr(module, "FIXED_ADAPTER_SHA256", adapter_sha)
    manifest = workspace / "manifest.json"
    manifest.write_text(json.dumps({
        "base": str(base), "tokenizer": str(base), "adapter": str(adapter),
        "adapter_sha256": adapter_sha, "served_model": module.FIXED_ALIAS,
    }))
    args = SimpleNamespace(
        model_manifest=manifest, output_dir=workspace / "result", budget_ledger=budget_path,
        api_url="http://127.0.0.1:8000/v1", limit_seconds=10, request_timeout=3,
        prepared_identity=None, prepared_identity_sha256=None,
    )
    return module, args, budget


@pytest.fixture
def server():
    running = []

    def start(module, *, answers=None, transform=None, served_alias=None, delay=0,
              redirect=None, post_status=200):
        state = {"requests": [], "get_requests": 0}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                state["get_requests"] += 1
                assert self.path == "/v1/models"
                self.respond({"data": [{"id": served_alias or module.FIXED_ALIAS}]})

            def do_POST(self):
                assert self.path == "/v1/chat/completions"
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                state["requests"].append(request)
                index = len(state["requests"]) - 1
                if delay:
                    time.sleep(delay)
                if redirect:
                    self.send_response(302)
                    self.send_header("Location", redirect)
                    self.end_headers()
                    return
                answer = (answers[index] if answers else
                          ({"left": "red", "right": "blue"} if index == 0
                           else {"left": "blue", "right": "red"}))
                response = {
                    "id": f"fixture-response-{index}", "model": module.FIXED_ALIAS,
                    "usage": {"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35},
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": json.dumps(answer),
                    }}],
                }
                if transform:
                    response = transform(response, index)
                self.respond(response, status=post_status)

            def respond(self, response, status=200):
                payload = json.dumps(response).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        running.append((http, thread))
        return f"http://127.0.0.1:{http.server_port}/v1", state

    yield start
    for http, thread in running:
        http.shutdown()
        http.server_close()
        thread.join(timeout=3)


def assert_budget_settled(budget):
    with budget.connect() as db:
        rows = db.execute('SELECT phase,seconds,status FROM jobs ORDER BY start').fetchall()
    assert rows[0] == ("previous-real-experiment", 12.5, "complete")
    assert len(rows) == 2
    assert rows[1][0] == "fixed-model-vision-capability"
    assert rows[1][1] > 0
    assert rows[1][2] != "running"


def test_two_real_http_requests_same_prompt_different_pixels_and_durable_receipts(probe, server):
    module, args, budget = probe
    args.api_url, state = server(module)
    result = module.run_probe(args)
    assert result['passed'] is True
    assert result['formal_release_evidence'] is False
    assert result['production_multimodal_verified'] is False
    assert result['rsi_enabled'] is False
    assert len(state['requests']) == len(result['calls']) == 2
    first, second = state['requests']
    assert first['messages'][0]['content'][0] == second['messages'][0]['content'][0]
    assert first['messages'][0]['content'][1] != second['messages'][0]['content'][1]
    assert first['model'] == second['model'] == module.FIXED_ALIAS
    assert first['seed'] == 17 and first['temperature'] == 0 and first['max_tokens'] == 256
    assert first['chat_template_kwargs'] == {'enable_thinking': False}
    assert 'expected' not in json.dumps(first)
    assert result['model_identity']['files']['adapter']['adapter_model.safetensors']
    persisted = json.loads((args.output_dir / 'vision-probe.json').read_text())
    assert persisted == result
    ledger = TimingLedger(args.output_dir / 'timing.sqlite')
    assert ledger.summary()['calls'] == 2 and ledger.summary()['failures'] == 0
    assert len({call['call_id'] for call in result['calls']}) == 2
    assert len({call['backend_request_id'] for call in result['calls']}) == 2
    assert_budget_settled(budget)


@pytest.mark.parametrize('mutation,error', [
    (lambda r, i: {**r, 'model': 'base-model-fallback'}, 'different model alias'),
    (lambda r, i: {**r, 'usage': {}}, 'actual usage'),
    (lambda r, i: {**r, 'usage': {'prompt_tokens': True, 'completion_tokens': 1, 'total_tokens': 2}}, 'actual usage'),
    (lambda r, i: {**r, 'id': ''}, 'request ID'),
    (lambda r, i: {**r, 'id': 'duplicate'}, 'request ID'),
])
def test_invalid_backend_receipt_never_passes_and_settles_budget(probe, server, mutation, error):
    module, args, budget = probe
    args.api_url, state = server(module, transform=mutation)
    result = module.run_probe(args)
    assert result['passed'] is False and error in result['error']
    assert len(state['requests']) <= 2
    assert TimingLedger(args.output_dir / 'timing.sqlite').summary()['failures'] == 1
    assert_budget_settled(budget)


@pytest.mark.parametrize('answers', [
    [{'left': 'red', 'right': 'blue'}, {'left': 'red', 'right': 'blue'}],
    [{'left': 'blue', 'right': 'red'}, {'left': 'blue', 'right': 'red'}],
    [{'left': 'red', 'right': 'blue'}, {'left': 'blue', 'right': 'red', 'extra': 'field'}],
])
def test_ignored_pixels_or_only_one_matching_image_never_passes(probe, server, answers):
    module, args, budget = probe
    args.api_url, state = server(module, answers=answers)
    result = module.run_probe(args)
    assert result['passed'] is False and result['status'] == 'failed'
    assert len(state['requests']) == 2
    assert any(not call['capability_match'] for call in result['calls'])
    assert_budget_settled(budget)


def test_active_budget_refuses_probe_without_creating_output_or_recovering_job(probe, server):
    module, args, budget = probe
    running = budget.start('do-not-interrupt', 100)
    args.api_url, state = server(module)
    with pytest.raises(RuntimeError, match='already running'):
        module.run_probe(args)
    assert not args.output_dir.exists() and not state['requests'] and state['get_requests'] == 0
    with budget.connect() as db:
        assert db.execute('SELECT status,seconds FROM jobs WHERE id=?', (running,)).fetchone() == ('running', None)


@pytest.mark.parametrize('field,value,error', [
    ('api_url', 'https://127.0.0.1/v1', 'loopback'),
    ('api_url', 'http://example.org/v1', 'loopback'),
    ('api_url', 'http://localhost/v1', 'loopback'),
    ('api_url', 'http://127.0.0.1/v1?forward=external', 'loopback'),
    ('request_timeout', 181, 'at most 180'),
    ('limit_seconds', 901, '1..900'),
])
def test_invalid_inputs_rejected_before_lease(probe, field, value, error):
    module, args, budget = probe
    setattr(args, field, value)
    with pytest.raises(ValueError, match=error):
        module.run_probe(args)
    with budget.connect() as db:
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0] == 1


def test_wrong_manifest_alias_or_adapter_rejected_before_lease(probe):
    module, args, budget = probe
    manifest = json.loads(args.model_manifest.read_text())
    manifest['served_model'] = 'other-alias'
    args.model_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='no model fallback'):
        module.run_probe(args)
    with budget.connect() as db:
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0] == 1


def test_new_budget_file_is_never_initialized(probe):
    module, args, budget = probe
    args.budget_ledger = args.budget_ledger.parent / 'reset-budget.sqlite'
    with pytest.raises(ValueError, match='original cumulative'):
        module.run_probe(args)
    assert not args.budget_ledger.exists()


def test_outside_workspace_output_rejected_before_lease(probe, tmp_path):
    module, args, budget = probe
    args.output_dir = tmp_path / 'outside'
    with pytest.raises(ValueError, match='/workspace'):
        module.run_probe(args)


def test_unserved_alias_makes_zero_model_calls(probe, server):
    module, args, budget = probe
    args.api_url, state = server(module, served_alias='base-without-adapter')
    result = module.run_probe(args)
    assert result['passed'] is False and 'not served' in result['error']
    assert not state['requests']
    assert_budget_settled(budget)


def test_external_redirect_never_followed_and_budget_is_settled(probe, server):
    module, args, budget = probe
    args.api_url, state = server(module, redirect='http://example.invalid/collect')
    result = module.run_probe(args)
    assert result['passed'] is False and 'redirects are forbidden' in result['error']
    assert len(state['requests']) == 1
    assert_budget_settled(budget)


def test_real_http_timeout_settles_call_and_budget(probe, server):
    module, args, budget = probe
    args.request_timeout = 0.05
    args.api_url, state = server(module, delay=0.2)
    result = module.run_probe(args)
    assert result['passed'] is False and 'TimeoutError' in result['error']
    assert len(state['requests']) == 1
    assert_budget_settled(budget)


def test_http_failure_keeps_bounded_backend_error_and_settles_budget(probe, server):
    module, args, budget = probe
    args.api_url, state = server(
        module, transform=lambda r, i: {"error": {"message": "image input unsupported"}},
        post_status=400,
    )
    result = module.run_probe(args)
    assert result['passed'] is False and 'HTTP 400' in result['error']
    assert len(state['requests']) == 1
    assert result['calls'][0]['backend_error'] == {
        'status': 400, 'body': json.dumps({"error": {"message": "image input unsupported"}}),
    }
    assert_budget_settled(budget)


def test_identity_work_is_inside_actual_cloud_wall_deadline(probe, server, monkeypatch):
    module, args, budget = probe
    args.limit_seconds = 1
    args.api_url, state = server(module)

    def slow_identity(self):
        time.sleep(2)
        raise AssertionError("deadline did not interrupt identity work")

    monkeypatch.setattr(module.ModelManifest, 'identity', slow_identity)
    result = module.run_probe(args)
    assert result['passed'] is False and 'deadline exceeded' in result['error']
    assert 0.9 <= result['wall_seconds'] < 1.8
    assert not state['requests'] and not state['get_requests']
    assert_budget_settled(budget)


def _manifest_locked_fixture(args, *, alias="cityagent-qwen35-9b-base"):
    manifest = json.loads(args.model_manifest.read_text())
    manifest["served_model"] = alias
    args.model_manifest.write_text(json.dumps(manifest))
    tokenizer = Path(manifest["tokenizer"])
    (tokenizer / "tokenizer_config.json").write_text('{"test_fixture":true}')
    (tokenizer / "preprocessor_config.json").write_text('{"test_fixture":true}')
    args.model_policy = "manifest-locked"
    return manifest


def test_explicit_manifest_policy_uses_declared_alias_and_bound_processor(probe, server):
    from bridge_evidence.contest.vision_capability import validate_vision_capability

    module, args, budget = probe
    manifest = _manifest_locked_fixture(args)
    alias = manifest["served_model"]
    args.api_url, state = server(
        module, served_alias=alias,
        transform=lambda response, _: {**response, "model": alias},
    )
    result = module.run_probe(args)
    assert result["passed"] is True
    assert result["model_policy"] == "manifest-locked"
    assert all(request["model"] == alias for request in state["requests"])
    assert result["model_identity"]["files"]["tokenizer"]["preprocessor_config.json"]
    proof = validate_vision_capability(args.output_dir / "vision-probe.json", result["model_identity"])
    assert proof["mode"] == "same-model"
    assert_budget_settled(budget)


def test_manifest_policy_probes_explicit_base_without_inventing_adapter_or_merge(probe, server):
    module, args, budget = probe
    manifest = _manifest_locked_fixture(args)
    manifest.pop("adapter")
    manifest.pop("adapter_sha256")
    manifest["weight_kind"] = "base"
    # Inventory fixture only; these bytes are never loaded as a real model.
    (Path(manifest["base"]) / "model.safetensors").write_bytes(b"synthetic-base-weights")
    args.model_manifest.write_text(json.dumps(manifest))
    alias = manifest["served_model"]
    args.api_url, state = server(
        module, served_alias=alias,
        transform=lambda response, _: {**response, "model": alias},
    )
    result = module.run_probe(args)
    assert result["passed"] is True
    assert result["model_identity"]["manifest"]["weight_kind"] == "base"
    assert set(result["model_identity"]["files"]) == {"base", "tokenizer"}
    assert result["model_identity"]["manifest"]["adapter"] is None
    assert result["model_identity"]["manifest"]["merged"] is None
    assert len(state["requests"]) == 2
    assert_budget_settled(budget)


@pytest.mark.parametrize("alias", [None, "", " wrong alias ", "x" * 257])
def test_manifest_policy_requires_explicit_valid_alias_before_lease(probe, alias):
    module, args, budget = probe
    _manifest_locked_fixture(args, alias=alias)
    with pytest.raises(ValueError, match="explicit served model alias"):
        module.run_probe(args)
    with budget.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


@pytest.mark.parametrize("name", ["tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"])
def test_manifest_policy_requires_bound_tokenizer_and_processor_files(probe, name):
    module, args, budget = probe
    manifest = _manifest_locked_fixture(args)
    (Path(manifest["tokenizer"]) / name).unlink()
    with pytest.raises(ValueError, match="tokenizer|processor"):
        module.run_probe(args)
    with budget.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_manifest_policy_never_accepts_response_from_other_model(probe, server):
    module, args, budget = probe
    manifest = _manifest_locked_fixture(args)
    args.api_url, state = server(module, served_alias=manifest["served_model"])
    result = module.run_probe(args)
    assert result["passed"] is False
    assert "different model alias" in result["error"]
    assert len(state["requests"]) == 1
    assert_budget_settled(budget)


def test_unknown_model_policy_cannot_enable_generic_probe(probe):
    module, args, budget = probe
    args.model_policy = "arbitrary-fallback"
    with pytest.raises(ValueError, match="model policy"):
        module.run_probe(args)
    with budget.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


@pytest.mark.parametrize("fence", ["```json\n{}\n```", "```\n{}\n```", " \n```JSON\n{}\n``` \n"])
def test_complete_fenced_object_has_same_probe_and_independent_validation(probe, server, fence):
    from bridge_evidence.contest.vision_capability import validate_vision_capability

    module, args, budget = probe

    def fenced_response(response, index):
        message = response["choices"][0]["message"]
        message["content"] = fence.format(message["content"])
        return response

    args.api_url, state = server(module, transform=fenced_response)
    result = module.run_probe(args)
    assert result["passed"] is True
    assert all(call["content"].strip().startswith("```") for call in result["calls"])
    assert validate_vision_capability(
        args.output_dir / "vision-probe.json", result["model_identity"]
    )["mode"] == "same-model"
    assert len(state["requests"]) == 2
    assert_budget_settled(budget)


@pytest.mark.parametrize("content", [
    'Explanation: {"left":"red","right":"blue"}',
    '```json\n{"left":"red","right":"blue"}\n``` trailing text',
    '```json\n{"left":"red","right":"blue"}\n```\n```json\n{}\n```',
    '{"left":"wrong","left":"red","right":"blue"}',
    '```json\n{"left":"red","right":"blue",}\n```',
])
def test_probe_rejects_wrapped_junk_and_ambiguous_json(probe, server, content):
    module, args, budget = probe

    def malformed_first(response, index):
        if index == 0:
            response["choices"][0]["message"]["content"] = content
        return response

    args.api_url, _state = server(module, transform=malformed_first)
    result = module.run_probe(args)
    assert result["passed"] is False
    assert result["calls"][0]["capability_match"] is False
    assert_budget_settled(budget)
