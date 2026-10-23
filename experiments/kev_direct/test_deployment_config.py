"""Synthetic configuration checks; never start services or query a real GPU."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import deployment


def configure_deployment(monkeypatch, tmp_path):
    settings = {
        'EXPECTED_HOSTNAME': 'synthetic-authorized-host',
        'SDK_PYTHON': str(tmp_path / 'sdk-python'),
        'INFERENCE_PYTHON': str(tmp_path / 'inference-python'),
        'KEV_PYTHON': str(tmp_path / 'kev-python'),
        'MAIN_MODEL_PATH': str(tmp_path / 'model'),
        'TOKENIZER_PATH': str(tmp_path / 'model' / 'tokenizer.json'),
        'LAUNCH_CONFIG': str(tmp_path / 'launch.json'),
        'MODEL_MANIFEST': str(tmp_path / 'model-manifest.json'),
        'MODEL_VERIFICATION': str(tmp_path / 'verified.json'),
        'GPU': '2', 'MAIN_PORT': '23001', 'MAIN_MODEL': 'synthetic-main',
        'SUMMARY_PORT': '23002', 'SUMMARY_MODEL': 'synthetic-summary',
        'INPUT_ROOT': str(tmp_path), 'KEV_ROOT': str(tmp_path),
        'DOCUMENT_IDS': 'synthetic-a,synthetic-b',
    }
    Path(settings['MAIN_MODEL_PATH']).mkdir()
    for name in ('SDK_PYTHON', 'INFERENCE_PYTHON', 'KEV_PYTHON', 'TOKENIZER_PATH'):
        Path(settings[name]).touch()
    for name in ('config.json', 'tokenizer_config.json'):
        (Path(settings['MAIN_MODEL_PATH']) / name).write_text('{}')
    (Path(settings['MAIN_MODEL_PATH']) / 'model.safetensors').write_bytes(b'synthetic-weights-not-for-inference')
    manifest = deployment.ModelManifest(base=settings['MAIN_MODEL_PATH'],
                                        tokenizer=settings['MAIN_MODEL_PATH'], weight_kind='base')
    Path(settings['MODEL_MANIFEST']).write_text(json.dumps(manifest.as_dict()))
    from bridge_evidence.contest import model_identity
    settings['MODEL_VERIFICATION_SHA256'] = model_identity.prepare_identity(manifest, settings['MODEL_VERIFICATION'])
    # Only this synthetic fixture substitutes mount detection. No real model or
    # production read-only gate is bypassed by the experiment implementation.
    monkeypatch.setattr(deployment, 'readonly_mount', lambda path: True)
    monkeypatch.setattr(model_identity, 'readonly_mount', lambda path: True)
    argv = [settings['INFERENCE_PYTHON'], '-m', 'vllm.entrypoints.openai.api_server',
            '--model', settings['MAIN_MODEL_PATH'], '--port', '22001', '--served-model-name', 'synthetic-template',
            '--host', '127.0.0.1']
    Path(settings['LAUNCH_CONFIG']).write_text(json.dumps({'argv': argv}))
    for key, value in settings.items():
        monkeypatch.setenv('CITYAGENT_KEV_' + key, value)
    monkeypatch.setattr(deployment.os, 'getuid', lambda: 0)
    monkeypatch.setattr(deployment.os, 'uname', lambda: SimpleNamespace(nodename=settings['EXPECTED_HOSTNAME']))
    return settings


@pytest.mark.parametrize('value', [None, True, False, '', ' 1', '-1', '0', '65536', '3.0', 3.5, '１２'])
def test_port_requires_explicit_valid_integer(value):
    with pytest.raises(ValueError, match='Port'):
        deployment.port_number(value)


def test_service_settings_accept_explicit_aliases_without_private_defaults():
    assert deployment.service_settings(1, 'custom-main', 65535, 'custom-summary') == {
        'main_port': 1, 'main_model': 'custom-main', 'summary_port': 65535, 'summary_model': 'custom-summary'}
    with pytest.raises(ValueError, match='distinct'):
        deployment.service_settings(1, 'custom-main', 1, 'custom-summary')
    for value in (None, '', ' bad', 'bad\nname'):
        with pytest.raises(ValueError, match='Model'):
            deployment.service_settings(1, value, 2, 'custom-summary')


@pytest.mark.parametrize('value', [None, True, '-1', '1,2', '2.5'])
def test_gpu_must_identify_one_device(value):
    with pytest.raises(ValueError, match='GPU'):
        deployment.gpu_number(value)


def test_launch_identity_is_explicit_and_template_is_not_modified(monkeypatch, tmp_path):
    settings = configure_deployment(monkeypatch, tmp_path)
    before = Path(settings['LAUNCH_CONFIG']).read_bytes()
    result = deployment.deployment_settings(require_inputs=True)
    argv = result['launch_argv']
    assert result['GPU'] == 2
    assert argv[argv.index('--port') + 1] == '23001'
    assert argv[argv.index('--served-model-name') + 1] == 'synthetic-main'
    assert argv[argv.index('--model') + 1] == settings['MAIN_MODEL_PATH']
    assert Path(settings['LAUNCH_CONFIG']).read_bytes() == before


@pytest.mark.parametrize('change', ['python', 'model', 'module', 'duplicate-port', 'equals-model'])
def test_launch_identity_mismatch_is_rejected(monkeypatch, tmp_path, change):
    settings = configure_deployment(monkeypatch, tmp_path)
    path = Path(settings['LAUNCH_CONFIG'])
    launch = json.loads(path.read_text())
    argv = launch['argv']
    if change == 'python':
        argv[0] = '/synthetic/other-python'
    elif change == 'model':
        argv[argv.index('--model') + 1] = '/synthetic/other-model'
    elif change == 'module':
        argv[2] = 'synthetic.other_module'
    elif change == 'duplicate-port':
        argv.extend(['--port', '23003'])
    else:
        argv.append('--model=/synthetic/other-model')
    path.write_text(json.dumps(launch))
    with pytest.raises(ValueError, match='Launch'):
        deployment.deployment_settings()


@pytest.mark.parametrize('change', ['missing-host', 'public-host', 'hostname', 'duplicate-host',
                                     'equals-host', 'python-command', 'duplicate-module', 'python-option'])
def test_launch_rejects_nonlocal_binding_and_ambiguous_python_mode(monkeypatch, tmp_path, change):
    settings = configure_deployment(monkeypatch, tmp_path)
    path = Path(settings['LAUNCH_CONFIG'])
    launch = json.loads(path.read_text())
    argv = launch['argv']
    if change == 'missing-host':
        del argv[-2:]
    elif change == 'public-host':
        argv[-1] = '0.0.0.0'
    elif change == 'hostname':
        argv[-1] = 'localhost'
    elif change == 'duplicate-host':
        argv.extend(['--host', '127.0.0.1'])
    elif change == 'equals-host':
        argv.append('--host=127.0.0.1')
    elif change == 'python-command':
        argv[1:1] = ['-c', 'pass']
    elif change == 'duplicate-module':
        argv.extend(['-m', 'vllm.entrypoints.openai.api_server'])
    else:
        argv.insert(1, '-I')
    path.write_text(json.dumps(launch))
    with pytest.raises(ValueError, match='Launch'):
        deployment.deployment_settings()


@pytest.mark.parametrize('raw', [b'', b'{}', b'{"verified":true}', b'{"schema_version":1,"identity":{},"inventory":{}}'])
def test_empty_or_self_declared_verification_is_rejected(monkeypatch, tmp_path, raw):
    settings = configure_deployment(monkeypatch, tmp_path)
    Path(settings['MODEL_VERIFICATION']).write_bytes(raw)
    monkeypatch.setenv('CITYAGENT_KEV_MODEL_VERIFICATION_SHA256', hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match='MODEL_VERIFICATION'):
        deployment.deployment_settings()


def test_mutable_model_mount_does_not_fall_back_to_weight_rehash(monkeypatch, tmp_path):
    configure_deployment(monkeypatch, tmp_path)
    monkeypatch.setattr(deployment, 'readonly_mount', lambda path: False)
    monkeypatch.setattr(deployment, 'load_identity', lambda *args, **kwargs: pytest.fail('Mutable mount reached large-weight identity loading'))
    with pytest.raises(ValueError, match='read-only'):
        deployment.deployment_settings()


@pytest.mark.parametrize('change', ['model-path', 'quantization', 'tokenizer-content', 'verification-content', 'manifest-content'])
def test_replay_binds_content_and_launch_not_only_service_alias(monkeypatch, tmp_path, change):
    settings = configure_deployment(monkeypatch, tmp_path)
    before = deployment.deployment_settings()
    config = {'deployment_identity': before['identity']}
    if change == 'model-path':
        other = tmp_path / 'other-model'
        other.mkdir()
        monkeypatch.setenv('CITYAGENT_KEV_MAIN_MODEL_PATH', str(other))
        path = Path(settings['LAUNCH_CONFIG'])
        launch = json.loads(path.read_text())
        launch['argv'][launch['argv'].index('--model') + 1] = str(other)
        path.write_text(json.dumps(launch))
    elif change == 'quantization':
        path = Path(settings['LAUNCH_CONFIG'])
        launch = json.loads(path.read_text())
        launch['argv'].extend(['--quantization', 'synthetic-quantization'])
        path.write_text(json.dumps(launch))
    elif change == 'tokenizer-content':
        Path(settings['TOKENIZER_PATH']).write_text('{"synthetic":"changed"}')
    elif change == 'verification-content':
        path = Path(settings['MODEL_VERIFICATION'])
        receipt = json.loads(path.read_text())
        receipt['synthetic_revision'] = 2
        raw = json.dumps(receipt).encode()
        path.write_bytes(raw)
        monkeypatch.setenv('CITYAGENT_KEV_MODEL_VERIFICATION_SHA256', hashlib.sha256(raw).hexdigest())
    else:
        path = Path(settings['MODEL_MANIFEST'])
        manifest = json.loads(path.read_text())
        manifest['served_model'] = settings['MAIN_MODEL']
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        deployment.require_same_deployment(config, deployment.deployment_settings())


def test_prepared_identity_reuses_inventory_without_reading_weight_bytes(monkeypatch, tmp_path):
    settings = configure_deployment(monkeypatch, tmp_path)
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.suffix == '.safetensors':
            pytest.fail('Fast verification read weight bytes')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    result = deployment.deployment_settings()
    assert result['identity']['model_verification_sha256'] == settings['MODEL_VERIFICATION_SHA256']
    deployment.require_same_deployment({'deployment_identity': result['identity']}, result)


@pytest.mark.parametrize('runner_name', ['three_doc_runner', 'single_review_runner'])
def test_legacy_replay_is_rejected_before_services_or_subprocesses(monkeypatch, tmp_path, runner_name):
    import importlib
    configure_deployment(monkeypatch, tmp_path)
    (tmp_path / 'freeze.json').write_text('{"files":{}}')
    (tmp_path / 'config.json').write_text('{}')
    runner = importlib.import_module(runner_name)
    def forbidden(*args, **kwargs):
        pytest.fail('Legacy replay reached process or service operations')
    monkeypatch.setattr(runner, 'services', forbidden)
    monkeypatch.setattr(runner.subprocess, 'run', forbidden)
    monkeypatch.setattr(runner.subprocess, 'Popen', forbidden)
    args = [tmp_path, True] if runner_name == 'three_doc_runner' else [tmp_path, tmp_path, 'synthetic-a']
    with pytest.raises(ValueError, match='Legacy replay directory'):
        runner.run(*args)


@pytest.mark.parametrize('runner_name', ['three_doc_runner', 'single_review_runner'])
def test_unconfigured_runner_cannot_start_or_stop_processes(monkeypatch, tmp_path, runner_name):
    import importlib
    runner = importlib.import_module(runner_name)
    for key in list(deployment.os.environ):
        if key.startswith('CITYAGENT_KEV_'):
            monkeypatch.delenv(key)
    def forbidden(*args, **kwargs):
        pytest.fail('Unconfigured experiment reached process or service operations')
    monkeypatch.setattr(runner.subprocess, 'run', forbidden)
    monkeypatch.setattr(runner.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(runner.subprocess, 'check_output', forbidden)
    monkeypatch.setattr(runner.os, 'killpg', forbidden)
    monkeypatch.setattr(runner, 'services', forbidden)
    args = [tmp_path] if runner_name == 'three_doc_runner' else [tmp_path, tmp_path, 'synthetic-a']
    with pytest.raises(ValueError, match='CITYAGENT_KEV_EXPECTED_HOSTNAME'):
        runner.run(*args)


def test_unconfigured_batch_and_merge_fail_before_service_access(monkeypatch, tmp_path):
    import batch
    import merge
    def forbidden(*args, **kwargs):
        pytest.fail('Unconfigured experiment reached service operations')
    monkeypatch.setattr(merge, 'snapshot', forbidden)
    monkeypatch.setattr(batch.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='Port'):
        batch.extract({}, tmp_path / 'extract')
    with pytest.raises(ValueError, match='Port'):
        merge.run(None, None, tmp_path / 'merge', None, None)
    assert not (tmp_path / 'extract').exists()
    assert not (tmp_path / 'merge').exists()


@pytest.mark.parametrize('wrong_role', ['main', 'summary'])
def test_merge_rejects_unserved_explicit_model_before_model_calls(monkeypatch, tmp_path, wrong_role):
    import merge
    source = tmp_path / 'source'
    source.mkdir()
    manifest = {'document_sha256': 'synthetic-sha', 'units': []}
    identity = {'document_sha256': 'synthetic-sha', 'manifest_sha256': merge.json_digest(manifest)}
    (source / 'source-projection.json').write_text(json.dumps(manifest))
    (source / 'materials.json').write_text('{}')
    candidates = tmp_path / 'candidates.json'
    candidates.write_text(json.dumps({**identity, 'subagent_calls': 0}))
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(identity))
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_text('{}')
    selected = []
    def snapshot(port):
        selected.append(port)
        role = 'main' if port == 23001 else 'summary'
        served = 'synthetic-unrelated' if role == wrong_role else 'synthetic-' + role
        return {'process': {'pid': port}, 'models': {'data': [{'id': served}]}}
    monkeypatch.setattr(merge, 'snapshot', snapshot)
    def forbidden(*args, **kwargs):
        pytest.fail('Model mismatch reached model client creation')
    monkeypatch.setattr(merge, 'make_client', forbidden)
    monkeypatch.setenv('CITYAGENT_STOP_OBSERVER_SERVER_PID', 'synthetic-before')
    monkeypatch.setenv('CITYAGENT_SUMMARY_STOP_OBSERVER_SERVER_PID', 'synthetic-before')
    with pytest.raises(AssertionError, match='Explicit ' + wrong_role + ' model'):
        merge.run(source, candidates, tmp_path / 'output', tokenizer, plan,
                  main_port=23001, main_model='synthetic-main',
                  summary_port=23002, summary_model='synthetic-summary')
    assert selected == [23001, 23002]
