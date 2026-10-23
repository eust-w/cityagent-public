"""Synthetic training-text manifests only; no real reports, labels or servers."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest import training_evidence_batch as batch


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source_batch(tmp_path):
    root = tmp_path / "inputs"
    first, second = root / "a.json", root / "nested/b.json"
    entries = []
    for index, path in enumerate((first, second), 1):
        digest = write_json(path, {"evidence_id": f"TRAIN-{index}", "training_line": index,
                                   "text": f"合成桥{index}"})
        entries.append({"path": path.relative_to(root).as_posix(), "sha256": digest})
    path = root / "manifest.json"
    raw = {"schema_version": 1, "scope": batch.SCOPE, "entries": entries}
    write_json(path, raw)
    return path, raw, first, second


def test_order_scope_and_every_source_bound(source_batch):
    path, raw, first, second = source_batch
    raw["entries"].reverse()
    write_json(path, raw)
    rows, identity = batch.load_training_evidence_batch(path)
    assert [row[0] for row in rows] == ["TRAIN-2", "TRAIN-1"]
    assert [row[2] for row in rows] == [second, first]
    assert identity["manifest_sha256"] == batch.file_digest(path)
    assert identity["scope"] == "official_training_text_only"
    assert "image coverage" in identity["limitations"][1]
    for (_, manifest, original), entry in zip(rows, identity["entries"]):
        assert manifest["source_json_sha256"] == entry["sha256"] == batch.file_digest(original)
    batch.verify_training_evidence_batch(identity)


def test_unlisted_files_are_never_discovered(source_batch):
    path, _, _, _ = source_batch
    (path.parent / "unlisted.json").write_text("not valid JSON; must never be opened")
    rows, identity = batch.load_training_evidence_batch(path)
    assert len(rows) == len(identity["entries"]) == 2


@pytest.mark.parametrize("change", ["labels", "extra", "empty", "scope", "boolean_version", "string_version",
                                    "entry_extra", "entry_labels", "entry_hash", "duplicate_path", "duplicate_id"])
def test_rejects_non_whitelist_or_ambiguous_batch(source_batch, change):
    path, raw, _, second = source_batch
    if change in {"labels", "extra"}:
        raw[change] = {}
    elif change == "empty":
        raw["entries"] = []
    elif change == "scope":
        raw["scope"] = "unspecified"
    elif change == "boolean_version":
        raw["schema_version"] = True
    elif change == "string_version":
        raw["schema_version"] = "1"
    elif change in {"entry_extra", "entry_labels"}:
        raw["entries"][0][change] = None
    elif change == "entry_hash":
        raw["entries"][0]["sha256"] = "invalid"
    elif change == "duplicate_path":
        raw["entries"].append(dict(raw["entries"][0]))
    else:
        raw["entries"][1]["sha256"] = write_json(second, {"evidence_id": "train-1", "training_line": 2, "text": "合成另一桥"})
    write_json(path, raw)
    with pytest.raises(ValueError):
        batch.load_training_evidence_batch(path)


@pytest.mark.parametrize("relative", ["../a.json", "/a.json", "./a.json", "nested/../a.json", "nested//b.json", "nested\\b.json", "."])
def test_rejects_escape_or_noncanonical_paths(source_batch, relative):
    path, raw, *_ = source_batch
    raw["entries"][0]["path"] = relative
    write_json(path, raw)
    with pytest.raises(ValueError, match="canonical"):
        batch.load_training_evidence_batch(path)


@pytest.mark.parametrize("which", ["source", "source_parent", "manifest", "manifest_parent"])
def test_rejects_symlinks(source_batch, which):
    path, _, first, second = source_batch
    if which == "source":
        first.unlink()
        first.symlink_to(second)
    elif which == "source_parent":
        directory = second.parent
        moved = directory.with_name("real-nested")
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
    elif which == "manifest":
        alias = path.with_name("alias.json")
        alias.symlink_to(path)
        path = alias
    else:
        alias = path.parent.with_name("alias-inputs")
        alias.symlink_to(path.parent, target_is_directory=True)
        path = alias / path.name
    with pytest.raises(ValueError, match="symlink"):
        batch.load_training_evidence_batch(path)


@pytest.mark.parametrize("root_kind", ["manifest_parent", "manifest_file", "source_parent", "source_file"])
def test_forbids_work_result_overlap(source_batch, root_kind):
    path, _, _, second = source_batch
    root = {"manifest_parent": path.parent, "manifest_file": path,
            "source_parent": second.parent, "source_file": second}[root_kind]
    with pytest.raises(ValueError, match="overlap"):
        batch.load_training_evidence_batch(path, forbidden_roots=(root,))


def test_manifest_duplicate_json_keys_rejected(source_batch):
    path, raw, *_ = source_batch
    path.write_text(json.dumps(raw)[:-1] + ', "scope": "official_training_text_only"}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        batch.load_training_evidence_batch(path)


@pytest.mark.parametrize("change", ["labels", "text_hash", "unexpected_field", "unsafe_id"])
def test_existing_training_parser_still_validates_each_input(source_batch, change):
    path, raw, first, _ = source_batch
    record = json.loads(first.read_text())
    if change == "text_hash":
        record["text_sha256"] = "0" * 64
    elif change == "unsafe_id":
        record["evidence_id"] = "bad\\name"
    else:
        record[change] = {}
    raw["entries"][0]["sha256"] = write_json(first, record)
    write_json(path, raw)
    with pytest.raises(ValueError):
        batch.load_training_evidence_batch(path)


def test_hash_tampering_before_and_after_load(source_batch):
    path, _, first, _ = source_batch
    _, identity = batch.load_training_evidence_batch(path)
    first.write_bytes(first.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        batch.load_training_evidence_batch(path)
    with pytest.raises(ValueError, match="source changed"):
        batch.verify_training_evidence_batch(identity)


def test_manifest_tampering_is_detected_even_when_semantics_unchanged(source_batch):
    path, _, first, _ = source_batch
    _, identity = batch.load_training_evidence_batch(path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest changed"):
        batch.verify_training_evidence_batch(identity, original=first)


def test_worker_verifies_only_own_source_plus_manifest(source_batch, monkeypatch):
    path, _, first, second = source_batch
    _, identity = batch.load_training_evidence_batch(path)
    digests = []
    actual = batch.file_digest
    def tracked(file):
        digests.append(file)
        return actual(file)
    monkeypatch.setattr(batch, "file_digest", tracked)
    second.unlink()
    batch.verify_training_evidence_batch(identity, original=first)
    assert digests == [path, first]
    with pytest.raises(ValueError, match="uniquely belong"):
        batch.verify_training_evidence_batch(identity, original=path)


def test_mutation_during_legacy_parser_is_rejected(source_batch, monkeypatch):
    from bridge_evidence.contest import divide_cli
    path, _, first, _ = source_batch
    parse = divide_cli._training_manifest
    def mutating(original):
        result = parse(original)
        original.write_bytes(original.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(divide_cli, "_training_manifest", mutating)
    with pytest.raises(ValueError, match="changed while parsing"):
        batch.load_training_evidence_batch(path)
