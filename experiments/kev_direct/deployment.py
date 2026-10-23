"""Explicit operator configuration for optional, host-local Kev experiments."""
import json
import os
from pathlib import Path
import re

from bridge_evidence.contest.backends import ModelManifest
from bridge_evidence.contest.model_identity import load_identity, readonly_mount
from bridge_evidence.contest.run_state import file_digest, json_digest


def port_number(value):
    if isinstance(value, bool) or not str(value).isascii() or not str(value).isdigit():
        raise ValueError('Port must be an integer in 1..65535')
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError('Port must be an integer in 1..65535')
    return port


def gpu_number(value):
    if isinstance(value, bool) or not str(value).isascii() or not str(value).isdigit():
        raise ValueError('GPU must be an explicit nonnegative device index')
    return int(value)


def model_name(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError('Model must be an explicit nonempty served model name')
    return value


def service_settings(main_port, main_model, summary_port, summary_model):
    result = {'main_port': port_number(main_port), 'main_model': model_name(main_model),
              'summary_port': port_number(summary_port), 'summary_model': model_name(summary_model)}
    if result['main_port'] == result['summary_port']:
        raise ValueError('Main and summary services must use distinct explicitly configured ports')
    return result


def existing_path(value, *, directory=False):
    if not isinstance(value, (str, Path)) or not str(value) or '\0' in str(value):
        raise ValueError('Configuration paths must be existing absolute paths of the required type')
    path = Path(value)
    if not path.is_absolute() or not (path.is_dir() if directory else path.is_file()):
        raise ValueError('Configuration paths must be existing absolute paths of the required type')
    return str(path)


def launch_command(settings):
    """Validate the supplied launch identity before changing only port and alias."""
    payload = json.loads(Path(settings['LAUNCH_CONFIG']).read_bytes())
    argv = payload.get('argv') if isinstance(payload, dict) else None
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg for arg in argv):
        raise ValueError('Launch configuration must contain a nonempty argv string list')
    if argv[0] != settings['INFERENCE_PYTHON']:
        raise ValueError('Launch Python does not match explicitly configured inference Python')
    module = 'vllm.entrypoints.openai.api_server'
    if argv[:3] != [settings['INFERENCE_PYTHON'], '-m', module] or argv.count('-m') != 1 or argv.count(module) != 1 or '-c' in argv:
        raise ValueError('Launch prefix must be exactly [inference_python, -m, vllm.entrypoints.openai.api_server]')
    for flag in ('--model', '--port', '--served-model-name', '--host'):
        if argv.count(flag) != 1 or any(arg.startswith(flag + '=') for arg in argv):
            raise ValueError('Launch configuration requires one explicit ' + flag)
        if argv.index(flag) + 1 == len(argv) or argv[argv.index(flag) + 1].startswith('--'):
            raise ValueError('Launch configuration requires one explicit ' + flag)
    if argv[argv.index('--model') + 1] != settings['MAIN_MODEL_PATH']:
        raise ValueError('Launch model path does not match explicitly configured model path')
    if argv[argv.index('--host') + 1] != '127.0.0.1':
        raise ValueError('Launch requires the numeric loopback host 127.0.0.1')
    argv = list(argv)
    argv[argv.index('--port') + 1] = str(settings['MAIN_PORT'])
    argv[argv.index('--served-model-name') + 1] = settings['MAIN_MODEL']
    return argv


def deployment_identity(settings):
    """Bind a prepared model receipt without rehashing large mutable weights.

    Fast verification requires the existing kernel-read-only preparation
    protocol. An empty marker or self-declared verified flag is not a receipt.
    """
    digest = settings['MODEL_VERIFICATION_SHA256']
    if re.fullmatch('[0-9a-f]{64}', digest) is None:
        raise ValueError('MODEL_VERIFICATION_SHA256 must pin a SHA256 preparation receipt')
    manifest = ModelManifest.load(settings['MODEL_MANIFEST'])
    main_root = Path(settings['MAIN_MODEL_PATH']).resolve()
    if Path(manifest.merged or manifest.base).resolve() != main_root:
        raise ValueError('Prepared model manifest does not bind MAIN_MODEL_PATH')
    if manifest.served_model is not None and manifest.served_model != settings['MAIN_MODEL']:
        raise ValueError('Prepared model manifest served_model differs from MAIN_MODEL')
    raw = Path(settings['MODEL_VERIFICATION']).read_bytes()
    try:
        receipt = json.loads(raw)
        if (not isinstance(receipt, dict) or type(receipt.get('schema_version')) is not int
                or receipt['schema_version'] != 1 or not isinstance(receipt.get('identity'), dict)
                or not isinstance(receipt.get('inventory'), dict) or not receipt['inventory']):
            raise ValueError('missing schema, identity or inventory')
        files = receipt['identity']['files']
        role = 'merged' if manifest.merged else 'base'
        if not any(Path(name).suffix in ('.safetensors', '.bin') for name in files[role]):
            raise ValueError('prepared model has no weight inventory')
        tokenizer = Path(settings['TOKENIZER_PATH']).resolve()
        tokenizer_name = tokenizer.relative_to(Path(manifest.tokenizer).resolve()).as_posix()
        if tokenizer_name not in files['tokenizer']:
            raise ValueError('tokenizer is not bound by prepared manifest')
        for group in files.values():
            if not isinstance(group, dict) or not group or any(not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None for value in group.values()):
                raise ValueError('invalid prepared content digests')
        for group in receipt['inventory'].values():
            if not isinstance(group, dict) or not group:
                raise ValueError('empty prepared inventory')
            for stamp in group.values():
                if not readonly_mount(stamp['path']):
                    raise ValueError('Kev fast identity requires kernel read-only model/tokenizer mounts; mutable files are not reused')
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError('Invalid MODEL_VERIFICATION preparation receipt: ' + str(exc)) from exc
    identity = load_identity(manifest, settings['MODEL_VERIFICATION'], expected_sha256=digest)
    return {
        'schema_version': 'kev-deployment-v1',
        'main_model_path': str(main_root),
        'model_manifest_path': str(Path(settings['MODEL_MANIFEST']).resolve()),
        'model_manifest_sha256': file_digest(Path(settings['MODEL_MANIFEST'])),
        'model_verification_sha256': digest,
        'prepared_model_identity_sha256': json_digest(identity),
        'launch_config_sha256': file_digest(Path(settings['LAUNCH_CONFIG'])),
        'launch_argv': settings['launch_argv'],
        'tokenizer_path': str(tokenizer), 'tokenizer_sha256': file_digest(tokenizer),
    }


def require_same_deployment(config, settings):
    previous = config.get('deployment_identity')
    if not isinstance(previous, dict) or previous.get('schema_version') != 'kev-deployment-v1':
        raise ValueError('Legacy replay directory lacks deployment identity; prepare a new baseline instead of reusing it')
    if previous != settings['identity']:
        raise ValueError('Replay deployment identity changed: model, launch, verification or tokenizer differs')


def deployment_settings(*, require_inputs=False):
    """Keep host and ownership gates mandatory; never infer deployment defaults."""
    names = ['EXPECTED_HOSTNAME', 'SDK_PYTHON', 'INFERENCE_PYTHON', 'MAIN_MODEL_PATH',
             'TOKENIZER_PATH', 'LAUNCH_CONFIG', 'MODEL_MANIFEST', 'MODEL_VERIFICATION',
             'MODEL_VERIFICATION_SHA256', 'GPU',
             'MAIN_PORT', 'MAIN_MODEL', 'SUMMARY_PORT', 'SUMMARY_MODEL']
    if require_inputs:
        names += ['INPUT_ROOT', 'DOCUMENT_IDS', 'KEV_ROOT', 'KEV_PYTHON']
    values = {}
    for name in names:
        value = os.environ.get('CITYAGENT_KEV_' + name, '').strip()
        if not value:
            raise ValueError('Set CITYAGENT_KEV_' + name + ' explicitly before this experiment')
        values[name] = value
    if os.getuid() != 0 or os.uname().nodename != values['EXPECTED_HOSTNAME']:
        raise ValueError('Run only on the explicitly selected host with the required process permissions')
    for name in ('SDK_PYTHON', 'INFERENCE_PYTHON', 'TOKENIZER_PATH', 'LAUNCH_CONFIG',
                 'MODEL_MANIFEST', 'MODEL_VERIFICATION', 'KEV_PYTHON', 'MAIN_MODEL_PATH', 'INPUT_ROOT', 'KEV_ROOT'):
        if name in values:
            try:
                values[name] = existing_path(values[name], directory=name in ('MAIN_MODEL_PATH', 'INPUT_ROOT', 'KEV_ROOT'))
            except ValueError as exc:
                raise ValueError('CITYAGENT_KEV_' + name + ' must be an existing absolute path of the required type') from exc
    service = service_settings(values['MAIN_PORT'], values['MAIN_MODEL'], values['SUMMARY_PORT'], values['SUMMARY_MODEL'])
    values.update({key.upper(): value for key, value in service.items()})
    values['GPU'] = gpu_number(values['GPU'])
    if require_inputs:
        ids = values['DOCUMENT_IDS'].split(',')
        if not ids or any(not item or item in ('.', '..') or '/' in item or '\\' in item for item in ids):
            raise ValueError('CITYAGENT_KEV_DOCUMENT_IDS must list safe comma-separated input directory names')
        if len(ids) != len(set(ids)):
            raise ValueError('CITYAGENT_KEV_DOCUMENT_IDS must not contain duplicates')
        values['document_ids'] = ids
    values['launch_argv'] = launch_command(values)
    values['identity'] = deployment_identity(values)
    return values
