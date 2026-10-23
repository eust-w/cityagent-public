"""Run identity must change with the bytes that determine inference and packaging."""
from pathlib import Path
import json

import pytest

from bridge_evidence.contest.run_state import (
    atomic_json, bind_work_directory, build_run_identity, document_identity,
    file_digest, json_digest, read_bound_text, record_name, validate_output_names,
)
from scripts.package_submission import RUNTIME_FILES


@pytest.fixture
def identity_inputs(tmp_path):
    code = tmp_path / 'code'
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    (model / 'model.safetensors').write_bytes(b'weights-v1')
    (model / 'tokenizer.json').write_text('{"tokenizer": 1}')
    for name in RUNTIME_FILES:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(name)
    return model, code, {'documents': [{'relative_path': 'bridge.docx', 'sha256': 'a' * 64}]}


def test_identity_covers_entire_submission_runtime(identity_inputs):
    identity = build_run_identity(*identity_inputs)
    assert set(RUNTIME_FILES).issubset(identity['code_files'])
    assert set(identity['model_files']) == {'config.json', 'model.safetensors', 'tokenizer.json'}


@pytest.mark.parametrize('changed', ['model.safetensors', 'tokenizer.json', 'run.py', 'README.md', 'LICENSE', 'NOTICE'])
def test_actual_content_change_invalidates_identity(identity_inputs, changed):
    model, code, options = identity_inputs
    before = json_digest(build_run_identity(model, code, options))
    target = (model if (model / changed).exists() else code) / changed
    target.write_bytes(target.read_bytes() + b'v2')
    assert json_digest(build_run_identity(model, code, options)) != before


def test_work_directory_requires_explicit_resume_and_identical_inputs(tmp_path, identity_inputs):
    identity = build_run_identity(*identity_inputs)
    work = tmp_path / 'work'
    first = bind_work_directory(work, identity, resume=False)
    with pytest.raises(FileExistsError):
        bind_work_directory(work, identity, resume=False)
    assert bind_work_directory(work, identity, resume=True) == first
    identity['options']['documents'][0]['sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='Run identity changed'):
        bind_work_directory(work, identity, resume=True)
    assert json.loads((work / 'run_manifest.json').read_text())['run_fingerprint'] == first


def test_untracked_directory_is_never_adopted(tmp_path):
    (tmp_path / 'old-record.json').write_text('{}')
    with pytest.raises(FileExistsError, match='untracked'):
        bind_work_directory(tmp_path, {}, resume=True)


def test_record_key_changes_when_same_filename_has_different_bytes(tmp_path):
    report = tmp_path / 'bridge.docx'
    report.write_bytes(b'first report')
    first = record_name(document_identity(report, tmp_path))
    report.write_bytes(b'second report')
    assert record_name(document_identity(report, tmp_path)) != first


def test_flat_output_rejects_collisions_including_case_and_extensions():
    with pytest.raises(ValueError, match='same output'):
        validate_output_names([Path('2018/Bridge.docx'), Path('2019/bridge.doc')])


def test_external_text_cache_requires_report_and_text_hashes(tmp_path):
    report = tmp_path / 'bridge.doc'
    report.write_bytes(b'original report')
    text = tmp_path / 'bridge.txt'
    text.write_text('bridge evidence', encoding='utf-8')
    sidecar = tmp_path / 'bridge.txt.source.json'
    with pytest.raises(FileNotFoundError):
        read_bound_text(report, tmp_path)
    metadata = {'document_sha256': file_digest(report), 'text_sha256': file_digest(text),
                'parser': 'contest-libreoffice-utf8-v1'}
    atomic_json(sidecar, metadata)
    assert read_bound_text(report, tmp_path) == 'bridge evidence'
    text.write_text('wrong bridge', encoding='utf-8')
    with pytest.raises(ValueError, match='hashes'):
        read_bound_text(report, tmp_path)
    text.write_text('bridge evidence', encoding='utf-8')
    report.write_bytes(b'changed report')
    with pytest.raises(ValueError, match='hashes'):
        read_bound_text(report, tmp_path)


def test_failed_record_serialization_preserves_previous_record(tmp_path):
    target = tmp_path / 'record.json'
    atomic_json(target, {'value': 'original'})
    with pytest.raises(ValueError):
        atomic_json(target, {'value': float('nan')})
    assert json.loads(target.read_text()) == {'value': 'original'}
    assert list(tmp_path.iterdir()) == [target]
