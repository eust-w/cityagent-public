"""CLI boundary tests using synthetic services; these do not verify the real SDK."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest import divide_cli as cli

spec = importlib.util.spec_from_file_location("divide_cli_test_entry", ROOT / "run.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic-weight-fixture")
    source = tmp_path / "training.json"
    source.write_text(json.dumps({"evidence_id": "TRAIN-1", "training_line": 1,
                                  "text": "桥梁名称：合成桥"}, ensure_ascii=False))
    args = entry.build_parser().parse_args([
        "--workflow", "divide-react", "--training-evidence-json", str(source),
        "--served-adapter", str(adapter), "--served-model", "synthetic-lora",
        "--result-dir", str(tmp_path / "result"), "--work-dir", str(tmp_path / "work"),
    ])
    model_catalog = [{"id": "synthetic-lora", "parent": "synthetic-base", "root": str(adapter)}]
    calls = {"models": 0, "workflows": 0, "writes": 0}

    class SyntheticClient:
        base_url, api_key = "http://127.0.0.1:8000", "synthetic-local-key"

        def __init__(self, *args, **kwargs):
            self.calls = []
            self.model_catalog = model_catalog

        def models(self):
            calls["models"] += 1
            return [item["id"] for item in self.model_catalog]

    class SyntheticWorkflow:
        def __init__(self, manifest, client, config, directory, **kwargs):
            calls["workflows"] += 1
            calls["manifest"] = manifest
            calls["react_config"] = config
            calls["settings"] = kwargs["settings"]

        def run(self):
            return {"data": {"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [], "病害明细": []},
                    "receipt": {"status": "success", "synthetic": True}}

    def synthetic_writer(data, path):
        calls["writes"] += 1
        Path(path).write_bytes(b"synthetic-docx-fixture")

    monkeypatch.setattr(cli, "LocalModelClient", SyntheticClient)
    monkeypatch.setattr(cli, "DivideWorkflow", SyntheticWorkflow)
    monkeypatch.setattr(cli, "write_extraction_docx", synthetic_writer)
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda name: "synthetic-sdk")
    return args, source, model_catalog, calls


def run(args):
    cli.run_divide(args, code_root=ROOT,
                   convert_doc=lambda *_: pytest.fail("Unexpected document conversion"))


@pytest.mark.parametrize("both", [False, True])
def test_requires_one_explicit_input_without_dataset_discovery(boundary, both):
    args, source, _, calls = boundary
    if both:
        args.input_dir = source.parent
    else:
        args.training_evidence_json = None
    with pytest.raises(ValueError, match="exactly one explicit"):
        run(args)
    assert calls["models"] == calls["workflows"] == 0


@pytest.mark.parametrize("field,value", [("root", "/other/lora"), ("parent", None),
                                          ("id", "other-lora")])
def test_requested_model_must_bind_to_local_lora_weights(boundary, field, value):
    args, _, catalog, calls = boundary
    catalog[0][field] = value
    with pytest.raises(ValueError, match="not served|does not bind"):
        run(args)
    assert calls["workflows"] == 0
    assert not args.work_dir.exists()


def test_existing_results_are_never_overwritten(boundary):
    args, _, _, calls = boundary
    args.result_dir.mkdir()
    existing = args.result_dir / "TRAIN-1.docx"
    existing.write_bytes(b"previous-result")
    with pytest.raises(FileExistsError, match="empty result directory"):
        run(args)
    assert existing.read_bytes() == b"previous-result"
    assert calls["models"] == calls["writes"] == 0


def test_existing_untracked_work_is_never_reused(boundary):
    args, _, _, calls = boundary
    args.work_dir.mkdir()
    evidence = args.work_dir / "prior-evidence.json"
    evidence.write_text("prior evidence")
    with pytest.raises(FileExistsError, match="untracked nonempty"):
        run(args)
    assert evidence.read_text() == "prior evidence"
    assert calls["workflows"] == 0


@pytest.mark.parametrize("extra", [{"output": "gold label"}, {"labels": []},
                                   {"evidence_id": "../escape"}, {"training_line": True},
                                   {"text_sha256": "0" * 64}])
def test_training_evidence_rejects_labels_and_invalid_identity(boundary, extra):
    args, source, _, calls = boundary
    content = json.loads(source.read_text())
    source.write_text(json.dumps({**content, **extra}))
    with pytest.raises(ValueError):
        run(args)
    assert calls["workflows"] == 0


def test_check_does_not_create_artifacts_or_claim_actual_sdk_execution(boundary, capsys):
    args, _, _, calls = boundary
    args.check = True
    run(args)
    assert calls["models"] == 1
    assert calls["workflows"] == calls["writes"] == 0
    assert not args.result_dir.exists() and not args.work_dir.exists()
    assert "real SDK/model execution is not yet verified" in capsys.readouterr().out


def test_failed_workflow_has_no_published_word(boundary, monkeypatch):
    args, _, _, calls = boundary

    class FailedWorkflow:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            raise RuntimeError("synthetic incomplete evidence")

    monkeypatch.setattr(cli, "DivideWorkflow", FailedWorkflow)
    with pytest.raises(RuntimeError, match="1 divide document"):
        run(args)
    assert calls["writes"] == 0
    assert list(args.result_dir.glob("*.docx")) == []
    summary = json.loads((args.work_dir / "summary.json").read_text())
    assert len(summary["failures"]) == 1 and not summary["submission_ready"]


def test_training_source_changed_during_execution_is_not_published(boundary, monkeypatch):
    args, source, _, calls = boundary
    original = cli.DivideWorkflow

    class MutatingWorkflow(original):
        def run(self):
            result = super().run()
            source.write_text(source.read_text() + "\n")
            return result

    monkeypatch.setattr(cli, "DivideWorkflow", MutatingWorkflow)
    with pytest.raises(RuntimeError, match="1 divide document"):
        run(args)
    assert calls["writes"] == 0
    assert list(args.result_dir.glob("*.docx")) == []
    summary = json.loads((args.work_dir / "summary.json").read_text())
    assert "Training evidence input changed" in summary["failures"][0]["error"]


def test_writer_failure_cleans_pending_file_and_does_not_publish(boundary, monkeypatch):
    args, _, _, _ = boundary

    def failed_writer(data, path):
        Path(path).write_bytes(b"partial-word")
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(cli, "write_extraction_docx", failed_writer)
    with pytest.raises(RuntimeError, match="1 divide document"):
        run(args)
    assert list(args.result_dir.glob("*.docx")) == []
    assert list(args.result_dir.glob(".pending-*")) == []


def test_success_publishes_bound_result_and_keeps_development_status(boundary):
    args, _, _, calls = boundary
    args.max_attempts = 3
    run(args)
    output = args.result_dir / "TRAIN-1.docx"
    receipt = json.loads((args.work_dir / "documents/TRAIN-1/execution.json").read_text())
    identity = json.loads((args.work_dir / "run_manifest.json").read_text())
    summary = json.loads((args.work_dir / "summary.json").read_text())
    assert output.read_bytes() == b"synthetic-docx-fixture"
    assert receipt["output"]["sha256"] == cli.file_digest(output)
    assert receipt["run_fingerprint"] == identity["run_fingerprint"]
    assert identity["identity"]["served_model"] == "synthetic-lora"
    assert calls["react_config"]["model"] == "synthetic-lora"
    assert calls["settings"].max_extraction_attempts == 3
    assert "api_key" not in identity["identity"]
    assert summary["failures"] == [] and summary["submission_ready"] is False


def test_resume_identity_excludes_volatile_preflight_receipts(boundary, monkeypatch):
    from uuid import uuid4

    args, _, _, calls = boundary
    models = cli.LocalModelClient.models

    def with_receipt(client):
        result = models(client)
        client.calls.append({"kind": "models", "endpoint": "/v1/models",
                             "call_id": uuid4().hex, "seconds": float(calls["models"]),
                             "status": "completed"})
        return result

    monkeypatch.setattr(cli.LocalModelClient, "models", with_receipt)
    run(args)
    args.resume = True
    run(args)
    summary = json.loads((args.work_dir / "summary.json").read_text())
    assert calls["workflows"] == calls["writes"] == 1
    assert summary["reused_reports"] == ["TRAIN-1"]
    prior, resumed = summary["invocations"]
    assert prior["preflight_calls"][0]["call_id"] != resumed["preflight_calls"][0]["call_id"]


def test_active_report_owner_cannot_be_recovered_by_another_worker(boundary):
    args, _, _, calls = boundary
    directory = args.work_dir / "documents" / "TRAIN-1"
    name, manifest = cli._training_manifest(args.training_evidence_json)
    with cli.WorkLock(directory):
        with pytest.raises(RuntimeError, match="active owner"):
            cli._run_document(name, manifest, None, args, cli.DivideSettings(), "fixture",
                              cli.file_digest(args.training_evidence_json), lambda *_: None)
    assert calls["workflows"] == calls["writes"] == 0


def test_resume_rejects_changed_dynamic_output_budget(boundary):
    args, _, _, calls = boundary
    run(args)
    args.resume = True
    args.document_dynamic_output_budget = True
    with pytest.raises(ValueError, match="Run identity changed"):
        run(args)
    assert calls["workflows"] == calls["writes"] == 1


def test_divide_main_branches_before_official_test_discovery(boundary, monkeypatch):
    args, source, _, _ = boundary
    seen = []
    monkeypatch.setattr(sys, "argv", ["run.py", "--workflow", "divide-react",
                                      "--training-evidence-json", str(source)])
    monkeypatch.setattr(entry, "select_input_dir", lambda: pytest.fail("Official dataset auto-discovery"))
    monkeypatch.setattr(cli, "run_divide", lambda args, **kwargs: seen.append(args.workflow))
    entry.main()
    assert seen == ["divide-react"]


@pytest.mark.parametrize("cli_value,expected", [(None, 1800), ("2400", 2400)])
def test_model_timeout_reaches_worker_agent_and_run_identity(boundary, monkeypatch, cli_value, expected):
    args, _, _, calls = boundary
    parsed = entry.build_parser().parse_args([] if cli_value is None else ["--model-timeout", cli_value])
    args.model_timeout = parsed.model_timeout
    original_client = cli.LocalModelClient
    constructor = {}

    class CapturingClient(original_client):
        def __init__(self, *positional, **kwargs):
            constructor.update(kwargs)
            super().__init__(*positional, **kwargs)

    monkeypatch.setattr(cli, "LocalModelClient", CapturingClient)
    run(args)
    identity = json.loads((args.work_dir / "run_manifest.json").read_text())["identity"]
    assert args.model_timeout == expected
    assert constructor["timeout"] == expected
    assert calls["react_config"]["timeout"] == expected
    assert identity["model_timeout"] == expected
    # Waiting longer must not silently alter the already chosen token budgets.
    assert constructor["max_output_tokens"] == args.chunk_output_tokens
    assert constructor["max_model_len"] == args.max_model_len
    assert calls["react_config"]["max_tokens"] == 1024
    assert identity["settings"]["output_tokens"] == args.chunk_output_tokens


@pytest.mark.parametrize("timeout", [0, -1, -3600])
def test_nonpositive_model_timeout_fails_before_model_contact_or_artifacts(boundary, timeout):
    args, _, _, calls = boundary
    args.model_timeout = entry.build_parser().parse_args(["--model-timeout", str(timeout)]).model_timeout
    with pytest.raises(ValueError, match="Model timeout must be positive"):
        run(args)
    assert calls["models"] == calls["workflows"] == calls["writes"] == 0
    assert not args.work_dir.exists() and not args.result_dir.exists()


def test_training_manifest_hash_binds_the_same_bytes_that_were_parsed(tmp_path, monkeypatch):
    import hashlib

    source = tmp_path / "input-evidence.json"
    original_text = "桥梁名称：合成桥\n病害：右幅裂缝"
    original_bytes = (json.dumps({"evidence_id": "TRAIN-SNAPSHOT", "training_line": 3,
                                  "text": original_text}, ensure_ascii=False, indent=2)
                      + "\r\n").encode("utf-8")
    changed_bytes = json.dumps({"evidence_id": "TRAIN-CHANGED", "training_line": 4,
                                "text": "随后被替换的另一个报告"}, ensure_ascii=False).encode("utf-8")
    source.write_bytes(original_bytes)
    real_read_bytes = Path.read_bytes
    reads = []

    def replace_after_read(path):
        data = real_read_bytes(path)
        if path == source:
            reads.append(data)
            # Simulate a concurrent replacement between parse and hash. The
            # helper must identify the snapshot it read, not later file bytes.
            source.write_bytes(changed_bytes)
        return data

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    name, manifest = cli._training_manifest(source)
    assert reads == [original_bytes]
    assert name == "TRAIN-SNAPSHOT"
    assert "".join(unit["text"] for unit in manifest["units"]) == original_text
    assert manifest["source_json_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert manifest["source_json_sha256"] != hashlib.sha256(changed_bytes).hexdigest()
    assert manifest["text_sha256"] == hashlib.sha256(original_text.encode("utf-8")).hexdigest()


def _synthetic_merged_export(boundary, tmp_path, mode="merged_bf16"):
    args, _, catalog, _ = boundary
    adapter_config = args.served_adapter / "adapter_config.json"
    adapter_config.write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    adapter = {"path": str(args.served_adapter.resolve()), "rank": 16, "alpha": 32, "scale": 2.0,
               "weights_sha256": cli.file_digest(args.served_adapter / "adapter_model.safetensors"),
               "config_sha256": cli.file_digest(adapter_config)}

    def create(path, export_mode, parent=None):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel

        path.mkdir()
        config = {"model_type": "synthetic"}
        if export_mode == "w8a8_int8":
            config["quantization_config"] = {"quant_method": "w8a8_int8", "activation_scheme": "dynamic",
                                              "is_dynamic": True}
        (path / "config.json").write_text(json.dumps(config))
        Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")).save(str(path / "tokenizer.json"))
        (path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "</s>"}))
        (path / "chat_template.jinja").write_text(
            "{% for message in messages %}{{ message.content|tojson }}{{ eos_token }}{% endfor %}"
            "{% if add_generation_prompt %}assistant:{% endif %}"
            "{% if enable_thinking %}thinking{% endif %}"
        )
        (path / "model-1.safetensors").write_bytes(b"synthetic-checksum-only-fixture")
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer.weight": "model-1.safetensors"}}))
        manifest = {"schema_version": 1, "status": "completed", "mode": export_mode,
                    "lora_coverage": {"expected": 256, "merged": 256}, "output_tensor_count": 1,
                    "adapter": adapter if export_mode == "merged_bf16" else None,
                    "artifacts": {p.name: {"sha256": cli.file_digest(p), "bytes": p.stat().st_size}
                                  for p in path.iterdir()}}
        if parent:
            manifest.update(original_adapter=adapter, source=str(parent),
                            merged_manifest_sha256=cli.file_digest(parent / "conversion_manifest.json"),
                            source_config_sha256=cli.file_digest(parent / "config.json"),
                            source_index_sha256=cli.file_digest(parent / "model.safetensors.index.json"),
                            quantization={"quantized_matrices_expected": 400, "quantized_matrices": 400})
        (path / "conversion_manifest.json").write_text(json.dumps(manifest))
        return path

    parent = create(tmp_path / "parent-bf16", "merged_bf16") if mode == "w8a8_int8" else None
    root = create(tmp_path / "merged-model", mode, parent)
    args.served_merged_model = root
    catalog[:] = [{"id": args.served_model, "root": str(root), "parent": None}]
    return root, parent


@pytest.mark.parametrize("mode", ["merged_bf16", "w8a8_int8"])
def test_explicit_merged_export_binds_actual_artifacts_and_adapter(boundary, tmp_path, mode):
    args, _, catalog, _ = boundary
    root, parent = _synthetic_merged_export(boundary, tmp_path, mode)
    run(args)
    identity = json.loads((args.work_dir / "run_manifest.json").read_text())["identity"]
    model = identity["model_identity"]
    assert model["kind"] == "merged_export" and model["mode"] == mode
    assert model["binding_source"] == "model_catalog.root"
    assert model["exported_artifacts_full_checksum_verified"] is True
    assert model["verified_artifacts"]["model-1.safetensors"]["sha256"] == cli.file_digest(root / "model-1.safetensors")
    assert model["original_adapter"]["weights_sha256"] == identity["adapter_sha256"]
    assert identity["model_catalog"] == catalog
    assert identity["base_weights_full_checksum_verified"] is False
    assert bool(model["parent"]) == bool(parent)


def test_sglang_merged_alias_requires_real_model_info_binding(boundary, tmp_path, monkeypatch):
    args, _, catalog, _ = boundary
    root, _ = _synthetic_merged_export(boundary, tmp_path)
    catalog[0]["root"] = args.served_model
    calls = []
    original = cli.LocalModelClient

    class WithModelInfo(original):
        def _request(self, endpoint, payload, *, kind):
            calls.append((endpoint, payload, kind))
            receipt = {"kind": kind, "status": "success", "endpoint": endpoint}
            self.calls.append(receipt)
            return {"model_path": str(root), "tokenizer_path": str(root)}, receipt

    monkeypatch.setattr(cli, "LocalModelClient", WithModelInfo)
    run(args)
    identity = json.loads((args.work_dir / "run_manifest.json").read_text())["identity"]
    assert calls == [("/model_info", None, "model_identity")]
    assert identity["model_catalog"][0]["root"] == args.served_model
    assert identity["model_identity"]["model_info"]["model_path"] == str(root)
    assert identity["model_identity"]["binding_source"] == "model_catalog.alias+model_info.model_path"
    assert identity["preflight_model_calls"][0]["kind"] == "model_identity"
    assert identity["chat_tokenization_mode"] == "verified_local_template+server_prompt_count"


def test_verified_chat_renderer_preserves_unicode_and_disables_padding_and_truncation(boundary, tmp_path):
    from tokenizers import Tokenizer

    root, _ = _synthetic_merged_export(boundary, tmp_path)
    tokenizer_path = root / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_padding(length=24)
    tokenizer.enable_truncation(max_length=1)
    tokenizer.save(str(tokenizer_path))
    hashes = {name: cli.file_digest(root / name) for name in (
        "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}
    renderer = cli._MergedChatTokenizer(root, hashes)
    messages = [{"role": "user", "content": "桥梁 <>&'"}]
    prompt = renderer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    assert prompt == '"桥梁 <>&\'"</s>assistant:'
    assert renderer.tokenizer.padding is None and renderer.tokenizer.truncation is None
    assert renderer.apply_chat_template(messages, tokenize=True) == renderer.encode(prompt)
    assert messages == [{"role": "user", "content": "桥梁 <>&'"}]
    (root / "chat_template.jinja").write_text("a changed template")
    with pytest.raises(ValueError, match="Verified chat tokenizer changed"):
        cli._MergedChatTokenizer(root, hashes)


@pytest.mark.parametrize("change,match", [
    ("pending", "completed"), ("coverage", "coverage"), ("adapter", "adapter identity"),
    ("same_size_weight_mutation", "checksum mismatch"), ("index_missing_shard", "index/shard"),
    ("unsafe_artifact", "Unsafe"), ("catalog_path", "does not bind"),
    ("selected_lora", "base-model"), ("bf16_quantization", "unexpectedly"),
    ("bf16_nested_quantization", "unexpectedly"),
])
def test_merged_identity_rejects_inconsistent_export_before_execution(boundary, tmp_path, change, match):
    args, _, catalog, calls = boundary
    root, _ = _synthetic_merged_export(boundary, tmp_path)
    path = root / "conversion_manifest.json"
    manifest = json.loads(path.read_text())
    if change == "pending":
        manifest["status"] = "running"
    elif change == "coverage":
        manifest["lora_coverage"]["merged"] = 255
    elif change == "adapter":
        manifest["adapter"]["weights_sha256"] = "0" * 64
    elif change == "same_size_weight_mutation":
        p = root / "model-1.safetensors"
        p.write_bytes(p.read_bytes().upper())
    elif change == "index_missing_shard":
        p = root / "model.safetensors.index.json"
        p.write_text(json.dumps({"weight_map": {"layer.weight": "absent.safetensors"}}))
        manifest["artifacts"][p.name] = {"sha256": cli.file_digest(p), "bytes": p.stat().st_size}
    elif change == "unsafe_artifact":
        manifest["artifacts"]["../escape"] = {"sha256": "0" * 64, "bytes": 0}
    elif change == "catalog_path":
        catalog[0]["root"] = "/different/merged/model"
    elif change == "selected_lora":
        catalog[0]["parent"] = "other-base"
    elif change in {"bf16_quantization", "bf16_nested_quantization"}:
        p = root / "config.json"
        config = {"quantization_config": {"quant_method": "w8a8_int8"}}
        p.write_text(json.dumps({"text_config": config} if change == "bf16_nested_quantization" else config))
        manifest["artifacts"][p.name] = {"sha256": cli.file_digest(p), "bytes": p.stat().st_size}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        run(args)
    assert calls["workflows"] == 0 and not args.work_dir.exists()


@pytest.mark.parametrize("served_path", [None, "relative/model", "/different/model"])
def test_sglang_alias_with_unbound_model_info_fails_closed(boundary, tmp_path, monkeypatch, served_path):
    args, _, catalog, calls = boundary
    _synthetic_merged_export(boundary, tmp_path)
    catalog[0]["root"] = args.served_model
    monkeypatch.setattr(cli.LocalModelClient, "_request", lambda *_args, **_kwargs: ({"model_path": served_path}, {}), raising=False)
    with pytest.raises(ValueError, match="model_info does not bind"):
        run(args)
    assert calls["workflows"] == 0


def test_int8_export_rejects_broken_bf16_provenance_chain(boundary, tmp_path):
    args, _, _, calls = boundary
    _, parent = _synthetic_merged_export(boundary, tmp_path, "w8a8_int8")
    p = parent / "conversion_manifest.json"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError, match="parent merged manifest checksum"):
        run(args)
    assert calls["workflows"] == 0


def test_sglang_alias_rejects_a_different_served_tokenizer(boundary, tmp_path, monkeypatch):
    args, _, catalog, calls = boundary
    root, _ = _synthetic_merged_export(boundary, tmp_path)
    catalog[0]["root"] = args.served_model
    monkeypatch.setattr(cli.LocalModelClient, "_request", lambda *_args, **_kwargs: (
        {"model_path": str(root), "tokenizer_path": "/different/tokenizer"}, {}), raising=False)
    with pytest.raises(ValueError, match="model_info does not bind"):
        run(args)
    assert calls["workflows"] == 0


# These helpers must remain module-level: a real spawn worker imports them,
# rather than inheriting pytest's parent-process monkeypatches.
_PROCESS_FIXTURE = {}


class _ProcessSyntheticClient:
    base_url, api_key = "http://127.0.0.1:8000", "synthetic-worker-key"

    def __init__(self, *args, **kwargs):
        from uuid import uuid4

        self.client_id = uuid4().hex
        self.calls = []


class _ProcessSyntheticWorkflow:
    def __init__(self, manifest, client, config, directory, **kwargs):
        import os

        self.directory, self.name, self.client = Path(directory), Path(directory).name, client
        self.observation = {
            "name": self.name, "pid": os.getpid(), "client_id": client.client_id,
            "prior_calls": list(client.calls), "prior_config_owner": config.get("fixture_owner"),
            "directory": str(directory), "manifest_name": manifest["fixture_name"],
            "cwd": str(Path.cwd()),
            "sdk_environment": {key: os.environ.get(key) for key in (
                "LOG_TRACE_LOG_DIR", "LOG_TRACE_LOG_FILE", "LOG_TRACE_DB_PATH", "LOG_TRACE_BACKEND")},
            "max_tokens": config["max_tokens"], "model": config["model"],
        }
        config["fixture_owner"] = self.name
        client.calls.append({"document": self.name, "client_id": client.client_id})

    def run(self):
        import time

        self.observation["entered_ns"] = time.monotonic_ns()
        barrier = Path(_PROCESS_FIXTURE["barrier"])
        if _PROCESS_FIXTURE["concurrent"] or _PROCESS_FIXTURE.get("isolated"):
            Path("default_sql.db").write_text(self.name)
        if _PROCESS_FIXTURE["concurrent"]:
            (barrier / (self.name + ".entered")).write_text(str(self.observation["pid"]))
            deadline = time.monotonic() + 15
            expected = {name + ".entered" for name in
                        _PROCESS_FIXTURE.get("barrier_names", _PROCESS_FIXTURE["names"])}
            while not expected.issubset({path.name for path in barrier.glob("*.entered")}):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Synthetic documents did not overlap in distinct spawn workers")
                time.sleep(0.01)
        self.observation["leaving_ns"] = time.monotonic_ns()
        self.observation["model_calls"] = list(self.client.calls)
        (self.directory / "synthetic-observation.json").write_text(json.dumps(self.observation))
        if self.name == _PROCESS_FIXTURE.get("fail_name"):
            raise RuntimeError("synthetic report failure")
        return {
            "data": {"基本信息": {"桥梁名称": self.name}, "维修建议": [], "病害明细": []},
            "receipt": {"status": "success", "synthetic": True, **self.observation},
        }


def _process_synthetic_writer(data, path):
    Path(path).write_text(json.dumps(data, ensure_ascii=False))


def _process_synthetic_manifest(path):
    return {"document_sha256": cli.file_digest(path), "fixture_name": Path(path).stem}


def _process_synthetic_convert(source, directory):
    import os

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "synthetic-conversion.json").write_text(json.dumps({
        "source": str(source), "directory": str(directory), "pid": os.getpid(),
    }))
    target = directory / (source.stem + ".docx")
    target.write_bytes(source.read_bytes())
    return target


def _initialize_process_boundary(config):
    global _PROCESS_FIXTURE

    _PROCESS_FIXTURE = config
    cli.LocalModelClient = _ProcessSyntheticClient
    cli.DivideWorkflow = _ProcessSyntheticWorkflow
    cli.write_extraction_docx = _process_synthetic_writer
    cli.build_manifest = _process_synthetic_manifest


def _prepare_document_process_test(boundary, tmp_path, monkeypatch, *, workers=2, fail_name=None):
    args, _, _, calls = boundary
    args.training_evidence_json = None
    args.input_dir = tmp_path / "reports"
    args.input_dir.mkdir()
    # Identical contents deliberately exercise conversion-cache isolation.
    names = ["report-a", "report-b"]
    for name in names:
        (args.input_dir / (name + ".doc")).write_bytes(b"identical synthetic original document")
    args.document_workers = workers
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    config = {"names": names, "barrier": str(barrier), "concurrent": workers > 1, "fail_name": fail_name}
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/synthetic/soffice")
    return args, calls, config


def _install_real_spawn_pool(monkeypatch, config):
    from concurrent.futures import ProcessPoolExecutor

    pools = []

    def spawn_pool(**kwargs):
        assert kwargs["max_tasks_per_child"] == 1
        pools.append({"workers": kwargs["max_workers"],
                      "start_method": kwargs["mp_context"].get_start_method()})
        return ProcessPoolExecutor(**kwargs, initializer=_initialize_process_boundary, initargs=(config,))

    monkeypatch.setattr(cli, "ProcessPoolExecutor", spawn_pool)
    return pools


def _document_observations(args):
    return [json.loads(path.read_text()) for path in sorted(
        (args.work_dir / "documents").glob("*/synthetic-observation.json"))]


def test_spawn_documents_overlap_and_keep_clients_configs_and_conversion_isolated(boundary, tmp_path, monkeypatch):
    import math
    import os

    args, calls, config = _prepare_document_process_test(boundary, tmp_path, monkeypatch)
    pools = _install_real_spawn_pool(monkeypatch, config)
    parent_cwd = Path.cwd()
    parent_environment = dict(os.environ)
    cli.run_divide(args, code_root=ROOT, convert_doc=_process_synthetic_convert)
    assert Path.cwd() == parent_cwd
    assert dict(os.environ) == parent_environment

    observations = _document_observations(args)
    assert pools == [{"workers": 2, "start_method": "spawn"}]
    assert len(observations) == 2
    assert len({row["pid"] for row in observations}) == 2
    assert all(row["pid"] != os.getpid() for row in observations)
    assert max(row["entered_ns"] for row in observations) <= min(row["leaving_ns"] for row in observations)
    assert len({row["client_id"] for row in observations}) == 2
    for row in observations:
        assert row["prior_calls"] == []
        assert row["prior_config_owner"] is None
        assert row["manifest_name"] == row["name"]
        assert row["model_calls"] == [{"document": row["name"], "client_id": row["client_id"]}]
        assert row["max_tokens"] == 1024 and row["model"] == args.served_model
        directory = args.work_dir / "documents" / row["name"]
        assert row["cwd"] == str(directory)
        assert (directory / "default_sql.db").read_text() == row["name"]
        assert row["sdk_environment"] == {
            "LOG_TRACE_LOG_DIR": str(directory / "sdk-logs"), "LOG_TRACE_LOG_FILE": "tongagents.log",
            "LOG_TRACE_DB_PATH": str(directory / "sdk-logs/traces.db"), "LOG_TRACE_BACKEND": "local"}
        conversion = json.loads((directory / "converted/synthetic-conversion.json").read_text())
        assert conversion["directory"] == str(directory / "converted")
        assert conversion["pid"] == row["pid"]
        receipt = json.loads((directory / "execution.json").read_text())
        assert receipt["model_calls"] == row["model_calls"]
        assert (args.result_dir / (row["name"] + ".docx")).is_file()
    # The parent fake performs only preflight; worker fakes were installed in
    # independently imported interpreters by the process initializer.
    assert calls["models"] == 1 and calls["workflows"] == calls["writes"] == 0
    summary = json.loads((args.work_dir / "summary.json").read_text())
    assert summary["completed_reports"] == summary["reports"] == 2
    assert summary["failures"] == []
    assert [row["name"] for row in summary["documents"]] == config["names"]
    assert {row["worker_pid"] for row in summary["documents"]} == {row["pid"] for row in observations}
    assert summary["document_seconds_sum"] == pytest.approx(sum(row["seconds"] for row in summary["documents"]))
    assert summary["completed_reports_per_hour"] == pytest.approx(2 * 3600 / summary["seconds"])
    assert json.loads((args.result_dir / "infer_time.json").read_text())["infer_time"] == math.ceil(summary["seconds"])
    identity = json.loads((args.work_dir / "run_manifest.json").read_text())["identity"]
    assert identity["document_workers"] == 2 and identity["document_worker_backend"] == "spawn-processes"


def test_spawn_report_failure_preserves_other_output_and_counts_only_completed(boundary, tmp_path, monkeypatch):
    args, _, config = _prepare_document_process_test(boundary, tmp_path, monkeypatch, fail_name="report-a")
    _install_real_spawn_pool(monkeypatch, config)
    with pytest.raises(RuntimeError, match="1 divide document"):
        cli.run_divide(args, code_root=ROOT, convert_doc=_process_synthetic_convert)

    observations = _document_observations(args)
    assert len(observations) == 2 and len({row["pid"] for row in observations}) == 2
    assert not (args.result_dir / "report-a.docx").exists()
    assert (args.result_dir / "report-b.docx").is_file()
    assert list(args.result_dir.glob(".pending-*")) == []
    summary = json.loads((args.work_dir / "summary.json").read_text())
    assert summary["reports"] == 2 and summary["completed_reports"] == 1
    assert summary["failures"] == [{"name": "report-a", "error": "RuntimeError: synthetic report failure"}]
    assert [row["status"] for row in summary["documents"]] == ["failed", "completed"]
    assert summary["completed_reports_per_hour"] == pytest.approx(3600 / summary["seconds"])
    assert summary["document_seconds_sum"] == pytest.approx(sum(row["seconds"] for row in summary["documents"]))
    failed = json.loads((args.work_dir / "documents/report-a/document_summary.json").read_text())
    assert failed["status"] == "failed" and "synthetic report failure" in failed["error"]


def test_default_worker_is_serial_but_each_report_still_gets_fresh_state(boundary, tmp_path, monkeypatch):
    import os

    defaults = entry.build_parser().parse_args([])
    assert defaults.document_workers == 1
    assert defaults.isolate_documents is False
    args, _, config = _prepare_document_process_test(boundary, tmp_path, monkeypatch,
                                                    workers=defaults.document_workers)
    monkeypatch.setattr(sys.modules[__name__], "_PROCESS_FIXTURE", config)
    monkeypatch.setattr(cli, "LocalModelClient", _ProcessSyntheticClient)
    monkeypatch.setattr(cli, "DivideWorkflow", _ProcessSyntheticWorkflow)
    monkeypatch.setattr(cli, "write_extraction_docx", _process_synthetic_writer)
    monkeypatch.setattr(cli, "build_manifest", _process_synthetic_manifest)
    monkeypatch.setattr(cli, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("Default serial run spawned a pool"))
    # Keep a synthetic catalog for the separate parent preflight client.
    monkeypatch.setattr(_ProcessSyntheticClient, "models", lambda self: [args.served_model], raising=False)
    monkeypatch.setattr(_ProcessSyntheticClient, "model_catalog", [{
        "id": args.served_model, "parent": "synthetic-base", "root": str(args.served_adapter.resolve()),
    }], raising=False)
    cli.run_divide(args, code_root=ROOT, convert_doc=_process_synthetic_convert)

    observations = _document_observations(args)
    assert len(observations) == 2 and {row["pid"] for row in observations} == {os.getpid()}
    assert observations[0]["leaving_ns"] <= observations[1]["entered_ns"]
    assert len({row["client_id"] for row in observations}) == 2
    assert all(row["prior_calls"] == [] and row["prior_config_owner"] is None for row in observations)
    assert all(row["model_calls"] == [{"document": row["name"], "client_id": row["client_id"]}]
               for row in observations)
    identity = json.loads((args.work_dir / "run_manifest.json").read_text())["identity"]
    assert identity["document_workers"] == 1 and identity["document_worker_backend"] == "serial"
    assert all(row["cwd"] == str(Path.cwd()) for row in observations)


@pytest.mark.parametrize("workers", [1, 2])
def test_explicit_isolation_uses_fresh_sdk_process_for_every_report(boundary, tmp_path, monkeypatch, workers):
    import os

    args, _, config = _prepare_document_process_test(boundary, tmp_path, monkeypatch, workers=workers)
    args.isolate_documents = entry.build_parser().parse_args(["--isolate-documents"]).isolate_documents
    config["isolated"] = True
    config["barrier_names"] = list(config["names"])
    config["names"].append("report-c")
    (args.input_dir / "report-c.doc").write_bytes(b"third synthetic original")
    pools = _install_real_spawn_pool(monkeypatch, config)
    parent_cwd, parent_environment = Path.cwd(), dict(os.environ)
    cli.run_divide(args, code_root=ROOT, convert_doc=_process_synthetic_convert)
    assert Path.cwd() == parent_cwd and dict(os.environ) == parent_environment
    assert pools == [{"workers": workers, "start_method": "spawn"}]
    rows = _document_observations(args)
    assert len(rows) == len({row["pid"] for row in rows}) == 3
    assert all(row["pid"] != os.getpid() for row in rows)
    for row in rows:
        directory = args.work_dir / "documents" / row["name"]
        assert row["cwd"] == str(directory)
        assert (directory / "default_sql.db").read_text() == row["name"]
        assert row["sdk_environment"]["LOG_TRACE_DB_PATH"] == str(directory / "sdk-logs/traces.db")
        assert row["prior_calls"] == [] and row["prior_config_owner"] is None
    if workers == 1:
        assert all(a["leaving_ns"] <= b["entered_ns"] for a, b in zip(rows, rows[1:]))
    identity = json.loads((args.work_dir / "run_manifest.json").read_text())["identity"]
    summary = json.loads((args.work_dir / "summary.json").read_text())
    for record in (identity, summary):
        assert record["isolate_documents"] is True
        assert record["effective_document_workers"] == workers
        assert record["document_worker_backend"] == "spawn-processes"
    assert summary["completed_reports"] == summary["reports"] == 3


def test_actual_backend_is_serial_when_only_one_report_without_explicit_isolation(boundary):
    args, _, _, _ = boundary
    args.document_workers = 4
    run(args)
    identity = json.loads((args.work_dir / "run_manifest.json").read_text())["identity"]
    summary = json.loads((args.work_dir / "summary.json").read_text())
    for record in (identity, summary):
        assert record["document_workers"] == 4
        assert record["effective_document_workers"] == 1
        assert record["isolate_documents"] is False
        assert record["document_worker_backend"] == "serial"


def test_batch_wall_time_includes_conversion_workflow_retries_and_rendering(boundary, tmp_path, monkeypatch):
    args, _, _, _ = boundary
    args.training_evidence_json = None
    args.input_dir = tmp_path / "timed-reports"
    args.input_dir.mkdir()
    source = args.input_dir / "one.doc"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/synthetic/soffice")
    clock = [0.0]
    monkeypatch.setattr(cli.time, "perf_counter", lambda: clock[0])
    original_workflow = cli.DivideWorkflow
    original_writer = cli.write_extraction_docx

    def convert(path, directory):
        clock[0] += 2
        return path

    def manifest(path):
        clock[0] += 3
        return {"document_sha256": cli.file_digest(path)}

    class RetryingWorkflow(original_workflow):
        def run(self):
            clock[0] += 11  # Includes all attempts, not only the accepted call.
            return super().run()

    def render(data, path):
        clock[0] += 5
        original_writer(data, path)

    monkeypatch.setattr(cli, "build_manifest", manifest)
    monkeypatch.setattr(cli, "DivideWorkflow", RetryingWorkflow)
    monkeypatch.setattr(cli, "write_extraction_docx", render)
    cli.run_divide(args, code_root=ROOT, convert_doc=convert)
    summary = json.loads((args.work_dir / "summary.json").read_text())
    assert summary["seconds"] == 21
    assert summary["completed_reports_per_hour"] == pytest.approx(3600 / 21)
    assert json.loads((args.result_dir / "infer_time.json").read_text())["infer_time"] == 21


def test_process_pool_start_failure_is_recorded_as_incomplete_batch(boundary, tmp_path, monkeypatch):
    args, _, _ = _prepare_document_process_test(boundary, tmp_path, monkeypatch)

    def unavailable_pool(**kwargs):
        raise OSError("synthetic process startup failure")

    monkeypatch.setattr(cli, "ProcessPoolExecutor", unavailable_pool)
    with pytest.raises(RuntimeError, match="2 divide document"):
        cli.run_divide(args, code_root=ROOT, convert_doc=_process_synthetic_convert)
    summary = json.loads((args.work_dir / "summary.json").read_text())
    assert summary["completed_reports"] == 0 and len(summary["failures"]) == 2
    assert summary["completed_reports_per_hour"] == 0
    assert not list(args.result_dir.glob("*.docx"))


@pytest.mark.parametrize("workers", [1, 8])
def test_document_worker_boundary_values_pass_preflight(boundary, workers):
    args, _, _, calls = boundary
    args.document_workers = entry.build_parser().parse_args(["--document-workers", str(workers)]).document_workers
    args.check = True
    run(args)
    assert calls["models"] == 1 and calls["workflows"] == 0
    assert not args.work_dir.exists() and not args.result_dir.exists()


@pytest.mark.parametrize("workers", [0, -1, 9, True, 1.5])
def test_invalid_document_worker_count_fails_before_contact_or_artifacts(boundary, workers):
    args, _, _, calls = boundary
    args.document_workers = workers
    with pytest.raises(ValueError, match="workers.*between 1 and 8"):
        run(args)
    assert calls["models"] == calls["workflows"] == calls["writes"] == 0
    assert not args.work_dir.exists() and not args.result_dir.exists()


def test_output_name_collision_fails_before_starting_document_workers(boundary, tmp_path, monkeypatch):
    args, calls, _ = _prepare_document_process_test(boundary, tmp_path, monkeypatch)
    (args.input_dir / "REPORT-A.docx").write_bytes(b"another synthetic original")
    monkeypatch.setattr(cli, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("Collision started document workers"))
    with pytest.raises(ValueError, match="same output filename"):
        cli.run_divide(args, code_root=ROOT, convert_doc=_process_synthetic_convert)
    assert calls["workflows"] == calls["writes"] == 0
    assert not args.work_dir.exists() and not args.result_dir.exists()
