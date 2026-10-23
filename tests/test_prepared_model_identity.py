import json

import pytest

from bridge_evidence.contest.backends import ModelManifest


def model(tmp_path):
    root = tmp_path / 'model'
    root.mkdir()
    (root / 'config.json').write_text('{}')
    (root / 'weights.safetensors').write_bytes(b'weights')
    return ModelManifest(str(root), str(root), merged=str(root))


def test_prepared_identity_requires_pinned_digest(tmp_path):
    from bridge_evidence.contest.model_identity import load_identity, prepare_identity
    manifest = model(tmp_path)
    receipt = tmp_path / 'identity.json'
    prepare_identity(manifest, receipt)
    with pytest.raises(ValueError, match='pinned'):
        load_identity(manifest, receipt, expected_sha256=None)


def test_mutable_files_are_rehashed_and_changes_rejected(tmp_path):
    from bridge_evidence.contest.model_identity import load_identity, prepare_identity
    manifest = model(tmp_path)
    receipt = tmp_path / 'identity.json'
    digest = prepare_identity(manifest, receipt)
    assert load_identity(manifest, receipt, expected_sha256=digest) == manifest.identity()
    (tmp_path / 'model' / 'weights.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        load_identity(manifest, receipt, expected_sha256=digest)


def test_readonly_mount_fast_path_still_checks_metadata_and_inventory(tmp_path, monkeypatch):
    from bridge_evidence.contest import model_identity
    manifest = model(tmp_path)
    receipt = tmp_path / 'identity.json'
    digest = model_identity.prepare_identity(manifest, receipt)
    expected = manifest.identity()
    # Simulates a kernel read-only mount for this unit test only.
    monkeypatch.setattr(model_identity, 'readonly_mount', lambda path: True)
    original_hash = model_identity.file_digest
    def guarded_hash(path):
        assert str(path) == str(receipt), 'weight bytes reread on readonly mount'
        return original_hash(path)
    monkeypatch.setattr(model_identity, 'file_digest', guarded_hash)
    assert model_identity.load_identity(manifest, receipt, expected_sha256=digest) == expected
    (tmp_path / 'model' / 'extra.json').write_text('{}')
    with pytest.raises(ValueError, match='inventory'):
        model_identity.load_identity(manifest, receipt, expected_sha256=digest)


def test_receipt_tamper_rejected_before_reuse(tmp_path):
    from bridge_evidence.contest.model_identity import load_identity, prepare_identity
    manifest = model(tmp_path)
    receipt = tmp_path / 'identity.json'
    digest = prepare_identity(manifest, receipt)
    value = json.loads(receipt.read_text())
    value['identity']['files']['base']['weights.safetensors'] = '0' * 64
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='digest'):
        load_identity(manifest, receipt, expected_sha256=digest)


def test_tokenizer_chat_template_is_part_of_fixed_identity(tmp_path):
    from bridge_evidence.contest.model_identity import load_identity, prepare_identity
    manifest = model(tmp_path)
    template = tmp_path / 'model' / 'chat_template.jinja'
    template.write_text('original template')
    original = manifest.identity()
    receipt = tmp_path / 'identity.json'
    digest = prepare_identity(manifest, receipt)
    template.write_text('different template')
    assert manifest.identity() != original
    with pytest.raises(ValueError, match='changed'):
        load_identity(manifest, receipt, expected_sha256=digest)
