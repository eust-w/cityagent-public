"""Material admission cannot dispatch a mismatched or changed source bundle."""
import hashlib
import json
from pathlib import Path

import pytest

from bridge_evidence.contest import document_material_run as material_run
from bridge_evidence.contest import document_multiagent_cli as cli
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import atomic_json


def inputs(tmp_path):
    source = tmp_path / "source.docx"
    source.write_bytes(b"synthetic-source-bytes-for-admission-test")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = manifest_from_text("合成桥", source_sha256=digest)
    materials = {"docx_sha256": digest, "packets": []}
    return manifest, materials, source


def test_run_copies_materials_and_source_before_worker_dispatch(tmp_path, monkeypatch):
    manifest, materials, source = inputs(tmp_path)
    class StubIndex:
        def save(self, path):
            atomic_json(path, {"test": "no-model"})
    def load(m, material, path):
        assert m == manifest and material == materials
        assert Path(path).read_bytes() == source.read_bytes()
        return StubIndex()
    monkeypatch.setattr(material_run, "load_material_index", load)
    index, request = material_run.prepare_material_run(manifest, materials, source, tmp_path / "run")
    assert Path(request["materials_path"]).parent == tmp_path / "run" / "materials"
    material_run.load_worker_material_index(manifest, request)
    original = source.read_bytes()
    source.write_bytes(b"sidecar-original-changed-later")
    assert Path(request["materials_source_docx"]).read_bytes() == original
    Path(request["materials_path"]).write_text("{}")
    with pytest.raises(ValueError, match="changed after dispatch"):
        material_run.load_worker_material_index(manifest, request)


def test_foreign_source_is_rejected_before_artifact_creation(tmp_path):
    manifest, materials, source = inputs(tmp_path)
    source.write_bytes(b"different-document")
    with pytest.raises(ValueError, match="differs"):
        material_run.prepare_material_run(manifest, materials, source, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_partial_worker_binding_cannot_fall_back_silently():
    with pytest.raises(ValueError, match="Incomplete"):
        material_run.load_worker_material_index({}, {"materials_source_docx": "somewhere"})
    assert material_run.load_worker_material_index({}, {}) is None


@pytest.mark.parametrize("flags", [
    ["--materials", "missing.json"],
    ["--materials-sha256", "a" * 64],
    ["--materials-source-docx", "missing.docx"],
])
def test_incomplete_cli_material_selection_never_runs(tmp_path, monkeypatch, flags):
    manifest, _, _ = inputs(tmp_path)
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    monkeypatch.setattr(cli, "run_multiagent", lambda *a, **kw: pytest.fail("Unexpected model dispatch"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--manifest", str(path), "--manifest-sha256", hashlib.sha256(path.read_bytes()).hexdigest(),
                  "--scope", "official-training", "--output-dir", str(tmp_path / "out"), *flags])
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()
