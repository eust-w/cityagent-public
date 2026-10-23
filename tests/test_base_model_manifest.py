"""An unadapted baseline must be explicit and bound to its actual artifacts."""

import json
from collections import Counter
from pathlib import Path

import pytest

from bridge_evidence.contest.backends import ModelManifest


@pytest.fixture
def base_manifest_data(tmp_path):
    base = tmp_path / "base"
    tokenizer = tmp_path / "tokenizer"
    base.mkdir()
    tokenizer.mkdir()
    (base / "config.json").write_text('{"model_type":"qwen3_5"}')
    (base / "model.safetensors").write_bytes(b"fixed-base-weights")
    (tokenizer / "tokenizer_config.json").write_text('{}')
    (tokenizer / "tokenizer.json").write_text('{"version":"1.0"}')
    return {"base": str(base), "tokenizer": str(tokenizer),
            "served_model": "cityagent-qwen35-9b-base"}


def test_implicit_base_remains_rejected(base_manifest_data):
    with pytest.raises(ValueError, match="adapter or explicit merged"):
        ModelManifest(**base_manifest_data).validate()


def test_explicit_base_loads_and_binds_true_deployment(tmp_path, base_manifest_data):
    data = dict(base_manifest_data, weight_kind="base")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    model = ModelManifest.load(path).validate()
    identity = model.identity()
    assert model.adapter is None and model.merged is None
    assert identity["manifest"]["weight_kind"] == "base"
    assert set(identity["files"]) == {"base", "tokenizer"}
    assert "model.safetensors" in identity["files"]["base"]
    assert "tokenizer.json" in identity["files"]["tokenizer"]


@pytest.mark.parametrize("changed", ["base", "tokenizer"])
def test_explicit_base_identity_changes_with_artifact_bytes(base_manifest_data, changed):
    from pathlib import Path

    model = ModelManifest(**base_manifest_data, weight_kind="base")
    original = model.identity()
    artifact = (Path(model.base) / "model.safetensors" if changed == "base"
                else Path(model.tokenizer) / "tokenizer.json")
    artifact.write_bytes(b"replacement-bytes")
    assert model.identity() != original


def test_legacy_manifest_identity_shape_is_unchanged(base_manifest_data):
    model = ModelManifest(**base_manifest_data, merged=base_manifest_data["base"])
    expected = dict(base_manifest_data, adapter=None, merged=base_manifest_data["base"],
                    schema_version=1, adapter_sha256=None, eligibility_evidence=None)
    assert model.identity()["manifest"] == expected
    assert "weight_kind" not in model.identity()["manifest"]
    assert model.identity() != ModelManifest(**base_manifest_data, weight_kind="base").identity()


def test_legacy_native_lora_identity_shape_is_unchanged(tmp_path, base_manifest_data):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path":"base"}')
    (adapter / "adapter_model.safetensors").write_bytes(b"fixed-adapter")
    model = ModelManifest(**base_manifest_data, adapter=str(adapter))
    expected = dict(base_manifest_data, adapter=str(adapter), merged=None,
                    schema_version=1, adapter_sha256=None, eligibility_evidence=None)
    assert model.identity()["manifest"] == expected
    assert "adapter_model.safetensors" in model.identity()["files"]["adapter"]


@pytest.mark.parametrize("field,value", [
    ("adapter", "/adapter"), ("adapter", ""),
    ("merged", "/merged"), ("merged", ""),
    ("adapter_sha256", "a" * 64), ("adapter_sha256", ""),
])
def test_explicit_base_rejects_finetuned_claims(base_manifest_data, field, value):
    data = dict(base_manifest_data, weight_kind="base", **{field: value})
    with pytest.raises(ValueError, match="base.*adapter|base.*merged|base.*fine-tuned"):
        ModelManifest(**data).validate()


@pytest.mark.parametrize("kind", ["", "merged", "native-lora", 0, [], {}])
def test_unknown_weight_kind_is_rejected(base_manifest_data, kind):
    with pytest.raises(ValueError, match="weight_kind"):
        ModelManifest(**base_manifest_data, weight_kind=kind).validate()


@pytest.mark.parametrize("field,value", [
    ("base", 1), ("tokenizer", []), ("served_model", 9),
    ("eligibility_evidence", {}), ("schema_version", True),
    ("schema_version", "1"),
])
def test_manifest_field_types_are_validated(base_manifest_data, field, value):
    data = dict(base_manifest_data, weight_kind="base")
    data[field] = value
    with pytest.raises(ValueError):
        ModelManifest(**data).validate()


@pytest.mark.parametrize("missing", ["base/config.json", "tokenizer/tokenizer_config.json",
                                     "tokenizer/tokenizer.json"])
def test_explicit_base_requires_actual_config_and_tokenizer(tmp_path, base_manifest_data, missing):
    (tmp_path / missing).unlink()
    with pytest.raises(ValueError, match="config|tokenizer"):
        ModelManifest(**base_manifest_data, weight_kind="base").validate()


@pytest.mark.parametrize("invalid", ["base/config.json", "tokenizer/tokenizer_config.json"])
def test_explicit_base_requires_config_objects(tmp_path, base_manifest_data, invalid):
    (tmp_path / invalid).write_text('[]')
    with pytest.raises(ValueError, match="object"):
        ModelManifest(**base_manifest_data, weight_kind="base").validate()


def test_explicit_base_rejects_nonexistent_tokenizer(base_manifest_data):
    base_manifest_data["tokenizer"] += "-absent"
    with pytest.raises(ValueError, match="tokenizer directory missing"):
        ModelManifest(**base_manifest_data, weight_kind="base").validate()


@pytest.mark.parametrize("base_mode", [False, True])
def test_prepared_identity_roundtrip_preserves_manifest_mode(tmp_path, base_manifest_data, base_mode):
    from bridge_evidence.contest.model_identity import load_identity, prepare_identity

    mode = {"weight_kind": "base"} if base_mode else {"merged": base_manifest_data["base"]}
    model = ModelManifest(**base_manifest_data, **mode)
    receipt = tmp_path / "prepared.json"
    digest = prepare_identity(model, receipt)
    assert load_identity(model, receipt, expected_sha256=digest) == model.identity()


@pytest.mark.parametrize("state", ["missing", "empty"])
def test_explicit_base_requires_nonempty_weights(tmp_path, base_manifest_data, state):
    weight = tmp_path / "base/model.safetensors"
    if state == "missing":
        weight.unlink()
    else:
        weight.write_bytes(b"")
    with pytest.raises(ValueError, match="weight"):
        ModelManifest(**base_manifest_data, weight_kind="base").validate()


@pytest.mark.parametrize("index_name,weight_name", [
    ("model.safetensors.index.json", "model-00001-of-00001.safetensors"),
    ("pytorch_model.bin.index.json", "pytorch_model-00001-of-00001.bin"),
])
def test_explicit_base_accepts_bound_nonempty_shards(tmp_path, base_manifest_data, index_name, weight_name):
    root = tmp_path / "base"
    (root / "model.safetensors").unlink()
    (root / weight_name).write_bytes(b"test-inventory-bytes-not-load-evidence")
    (root / index_name).write_text(json.dumps({"weight_map": {"layer.weight": weight_name}}))
    identity = ModelManifest(**base_manifest_data, weight_kind="base").identity()
    assert {index_name, weight_name} <= identity["files"]["base"].keys()


@pytest.mark.parametrize("index_name,suffix", [
    ("model.safetensors.index.json", ".safetensors"),
    ("pytorch_model.bin.index.json", ".bin"),
])
def test_repeated_shard_references_validate_each_file_once(
    tmp_path, base_manifest_data, monkeypatch, index_name, suffix,
):
    root = tmp_path / "base"
    shards = [root / f"weights-{number}{suffix}" for number in range(4)]
    for shard in shards:
        shard.write_bytes(b"synthetic-shard")
    mapping = {f"layer.{number}.weight": shards[number % len(shards)].name
               for number in range(982)}
    (root / index_name).write_text(json.dumps({"weight_map": mapping}))
    resolved_shards = Counter()
    original_resolve = Path.resolve

    def record_resolve(path, *args, **kwargs):
        if path in shards:
            resolved_shards[path] += 1
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", record_resolve)
    ModelManifest(**base_manifest_data, weight_kind="base").validate()
    assert resolved_shards == Counter({shard: 1 for shard in shards})


@pytest.mark.parametrize("invalid", [
    None, 1, [], {}, "", "config.json", "sub\\weights.safetensors",
    "../outside.safetensors", "ABSOLUTE", "link.safetensors",
    "missing.safetensors", "empty.safetensors",
])
def test_duplicate_shards_do_not_skip_later_invalid_references(
    tmp_path, base_manifest_data, invalid,
):
    root = tmp_path / "base"
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    (root / "link.safetensors").symlink_to(outside)
    (root / "empty.safetensors").write_bytes(b"")
    if invalid == "ABSOLUTE":
        invalid = str(outside)
    mapping = {"layer.0": "model.safetensors", "layer.1": "model.safetensors",
               "layer.2": invalid}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError, match="weight"):
        ModelManifest(**base_manifest_data, weight_kind="base").validate()


@pytest.mark.parametrize("state", ["missing", "empty", "outside"])
def test_shards_are_rechecked_on_each_validation(tmp_path, base_manifest_data, state):
    root = tmp_path / "base"
    shard = root / "weights.safetensors"
    shard.write_bytes(b"synthetic-shard")
    mapping = {"layer.0": shard.name, "layer.1": shard.name}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    model = ModelManifest(**base_manifest_data, weight_kind="base")
    model.validate()

    shard.unlink()
    if state == "empty":
        shard.write_bytes(b"")
    elif state == "outside":
        outside = tmp_path / "outside.safetensors"
        outside.write_bytes(b"outside")
        shard.symlink_to(outside)
    with pytest.raises(ValueError, match="weight"):
        model.validate()


@pytest.mark.parametrize("state", ["missing", "empty"])
def test_index_cannot_hide_missing_or_empty_shard(tmp_path, base_manifest_data, state):
    root = tmp_path / "base"
    # Even an otherwise valid standalone file must not excuse a broken index.
    shard = "model-00001-of-00001.safetensors"
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": shard}}))
    if state == "empty":
        (root / shard).write_bytes(b"")
    with pytest.raises(ValueError, match="weight"):
        ModelManifest(**base_manifest_data, weight_kind="base").validate()


@pytest.mark.parametrize("mapping", [None, {}, [], {"layer.weight": None},
                                     {"layer.weight": 1}, {"layer.weight": "config.json"}])
def test_explicit_base_rejects_invalid_weight_index(tmp_path, base_manifest_data, mapping):
    (tmp_path / "base/model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError, match="weight"):
        ModelManifest(**base_manifest_data, weight_kind="base").validate()


@pytest.mark.parametrize("reference", ["../outside.safetensors", "ABSOLUTE", "link.safetensors"])
def test_weight_index_rejects_paths_outside_base(tmp_path, base_manifest_data, reference):
    root = tmp_path / "base"
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"not-in-declared-root")
    if reference == "ABSOLUTE":
        reference = str(outside)
    if reference == "link.safetensors":
        (root / reference).symlink_to(outside)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": reference}}))
    with pytest.raises(ValueError, match="weight"):
        ModelManifest(**base_manifest_data, weight_kind="base").validate()


def test_standalone_weight_symlink_cannot_escape_base(tmp_path, base_manifest_data):
    weight = tmp_path / "base/model.safetensors"
    weight.unlink()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    weight.symlink_to(outside)
    with pytest.raises(ValueError, match="weight"):
        ModelManifest(**base_manifest_data, weight_kind="base").validate()


def test_unindexed_shard_is_not_a_complete_model(tmp_path, base_manifest_data):
    root = tmp_path / "base"
    (root / "model.safetensors").rename(root / "model-00001-of-00002.safetensors")
    with pytest.raises(ValueError, match="weight"):
        ModelManifest(**base_manifest_data, weight_kind="base").validate()


def test_complete_index_does_not_excuse_escaping_standalone_weights(tmp_path, base_manifest_data):
    root = tmp_path / "base"
    shard = "model-00001-of-00001.safetensors"
    (root / shard).write_bytes(b"synthetic-shard")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": shard}}))
    (root / "model.safetensors").unlink()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    (root / "model.safetensors").symlink_to(outside)
    with pytest.raises(ValueError, match="weight"):
        ModelManifest(**base_manifest_data, weight_kind="base").validate()
