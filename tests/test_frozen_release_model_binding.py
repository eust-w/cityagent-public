"""Frozen publication evidence must identify current model bytes, not paths."""

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from bridge_evidence.contest.backends import ModelManifest
from bridge_evidence.contest.model_identity import prepare_identity
from bridge_evidence.contest.rsi import StrategyBundle
from bridge_evidence.contest.run_state import file_digest


@pytest.fixture
def entry():
    spec = importlib.util.spec_from_file_location(
        "contest_entry_release_binding", Path(__file__).resolve().parents[1] / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("changed_file", ["adapter", "tokenizer"])
@pytest.mark.parametrize("prepared", [False, True])
def test_frozen_release_rejects_same_path_changed_model_before_run_identity(
    tmp_path, monkeypatch, entry, changed_file, prepared
):
    base, adapter, tokenizer = [tmp_path / name for name in ("base", "adapter", "tokenizer")]
    for folder in (base, adapter, tokenizer):
        folder.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"original-base")
    (adapter / "adapter_config.json").write_text("{}")
    adapter_file = adapter / "adapter_model.safetensors"
    adapter_file.write_bytes(b"original-adapter")
    tokenizer_file = tokenizer / "tokenizer.json"
    tokenizer_file.write_text('{"version":1}')
    manifest = ModelManifest(str(base), str(tokenizer), adapter=str(adapter))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest)))
    qualified = {"model_identity": manifest.identity()}
    manifest_hash = file_digest(manifest_path)
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"model_manifest_sha256": manifest_hash}))
    target = adapter_file if changed_file == "adapter" else tokenizer_file
    target.write_bytes(b"replacement-bytes-at-identical-path")
    assert file_digest(manifest_path) == manifest_hash
    assert manifest.identity() != qualified["model_identity"]

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(StrategyBundle.baseline().as_dict()))
    sandbox = tmp_path / "sandbox"
    sandbox.with_suffix(".identity.json").write_text('{"digest":"synthetic-sandbox"}')
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    argv = [
        "run.py", "--input-dir", str(input_dir), "--model-manifest", str(manifest_path),
        "--rsi-mode", "frozen", "--rsi-bundle", str(bundle_path),
        "--rsi-sandbox-root", str(sandbox), "--rsi-release", str(release),
    ]
    if prepared:
        # Even a valid new preparation receipt cannot inherit old model approval.
        receipt = tmp_path / "prepared.json"
        digest = prepare_identity(manifest, receipt)
        argv.extend(["--prepared-model-identity", str(receipt),
                     "--prepared-model-identity-sha256", digest])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr("bridge_evidence.contest.rsi_sandbox.verify_root", lambda *_: None)
    monkeypatch.setattr("bridge_evidence.contest.rsi_acceptance.verify_release", lambda *_, **__: qualified)
    monkeypatch.setattr(entry, "ensure_ppu_environment", lambda: None)
    monkeypatch.setattr(entry, "preflight", lambda *_: [])
    monkeypatch.setattr(entry, "prepare_model", lambda *_: None)

    def forbidden_identity(*args, **kwargs):
        pytest.fail("changed model reached run identity instead of failing closed")

    monkeypatch.setattr(entry, "build_run_identity", forbidden_identity)
    with pytest.raises(SystemExit) as raised:
        entry._main_locked()
    assert raised.value.code == 2


def test_release_model_binding_accepts_identical_identity_and_no_release(entry):
    identity = {"manifest": {"base": "/base"}, "files": {"base": {"model.bin": "a" * 64}}}
    entry._validate_release_model_identity(None, identity)
    entry._validate_release_model_identity({"model_identity": identity}, identity)


@pytest.mark.parametrize("release", [{}, {"model_identity": None}, {"model_identity": {}}])
def test_release_model_binding_rejects_missing_complete_identity(entry, release):
    with pytest.raises(ValueError, match="model identity"):
        entry._validate_release_model_identity(release, None)
