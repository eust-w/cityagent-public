"""Explicit synthetic training batches; never open a dataset or model service."""
from concurrent.futures import Future
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest import divide_cli as cli

spec = importlib.util.spec_from_file_location("divide_training_batch_entry", ROOT / "run.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


@pytest.fixture
def batch_boundary(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    originals = []
    for index in (1, 2):
        source = inputs / f"source-{index}.json"
        source.write_text(json.dumps({"evidence_id": f"TRAIN-{index}", "training_line": index,
                                      "text": f"桥梁名称：合成桥{index}"}, ensure_ascii=False))
        originals.append(source)
    batch = inputs / "manifest.json"
    batch.write_text(json.dumps({"schema_version": 1, "scope": "official_training_text_only",
                                 "entries": [{"path": source.name, "sha256": cli.file_digest(source)}
                                             for source in originals]}))
    # An unrelated invalid file must not be discovered or parsed.
    (inputs / "unlisted.json").write_text("not a training input")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic-weights")
    args = entry.build_parser().parse_args([
        "--workflow", "divide-react", "--training-evidence-manifest", str(batch),
        "--served-adapter", str(adapter), "--served-model", "synthetic-lora",
        "--work-dir", str(tmp_path / "work"), "--result-dir", str(tmp_path / "result"),
    ])
    calls = {"clients": 0, "models": 0, "workflows": [], "writes": 0}

    class Client:
        base_url, api_key = "http://synthetic.invalid", "synthetic-key"

        def __init__(self, *args, **kwargs):
            calls["clients"] += 1
            self.calls = []
            self.model_catalog = [{"id": "synthetic-lora", "parent": "synthetic-base", "root": str(adapter)}]

        def models(self):
            calls["models"] += 1
            return ["synthetic-lora"]

    class Workflow:
        def __init__(self, manifest, client, config, directory, **kwargs):
            self.name = manifest["training_source"]["evidence_id"]
            calls["workflows"].append(self.name)

        def run(self):
            if calls.get("on_run"):
                calls["on_run"](self.name)
            return {"data": {"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [], "病害明细": []},
                    "receipt": {"status": "success", "synthetic": True}}

    def writer(data, path):
        calls["writes"] += 1
        Path(path).write_bytes(b"synthetic-docx")
        if calls.get("on_write"):
            calls["on_write"]()

    monkeypatch.setattr(cli, "LocalModelClient", Client)
    monkeypatch.setattr(cli, "DivideWorkflow", Workflow)
    monkeypatch.setattr(cli, "write_extraction_docx", writer)
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda _: "synthetic-sdk")
    return args, batch, originals, calls


def run(args):
    cli.run_divide(args, code_root=ROOT,
                   convert_doc=lambda *_: pytest.fail("Training JSON must not be converted as Word"))


def read_summary(args):
    return json.loads((args.work_dir / "summary.json").read_text())


@pytest.mark.parametrize("extra", ["training_evidence_json", "input_dir"])
def test_batch_excludes_other_input_modes_before_client(batch_boundary, extra):
    args, _, originals, calls = batch_boundary
    setattr(args, extra, originals[0] if extra == "training_evidence_json" else originals[0].parent)
    with pytest.raises(ValueError, match="exactly one explicit"):
        run(args)
    assert calls["clients"] == 0


@pytest.mark.parametrize("limit", [0, 1, 20])
def test_batch_cannot_be_silently_truncated(batch_boundary, limit):
    args, _, _, calls = batch_boundary
    args.limit = limit
    with pytest.raises(ValueError, match="limit cannot truncate"):
        run(args)
    assert calls["clients"] == 0


@pytest.mark.parametrize("kind", ["manifest", "manifest_parent", "source", "work"])
def test_batch_symlinks_are_rejected_before_resolution(batch_boundary, tmp_path, kind):
    args, batch, originals, calls = batch_boundary
    if kind == "manifest":
        alias = tmp_path / "linked.json"
        alias.symlink_to(batch)
        args.training_evidence_manifest = alias
    elif kind == "manifest_parent":
        alias = tmp_path / "linked-inputs"
        alias.symlink_to(batch.parent, target_is_directory=True)
        args.training_evidence_manifest = alias / batch.name
    elif kind == "source":
        target = tmp_path / "source.json"
        originals[0].rename(target)
        originals[0].symlink_to(target)
    else:
        target = tmp_path / "actual-work"
        target.mkdir()
        args.work_dir.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run(args)
    assert calls["clients"] == 0


@pytest.mark.parametrize("kind", ["source_in_work", "work_in_result", "result_in_work"])
def test_batch_rejects_overlapping_input_and_output_roots(batch_boundary, kind):
    args, batch, _, calls = batch_boundary
    if kind == "source_in_work":
        args.work_dir = batch.parent
    elif kind == "work_in_result":
        args.work_dir = args.result_dir / "nested"
    else:
        args.result_dir = args.work_dir / "nested"
    with pytest.raises(ValueError, match="overlap"):
        run(args)
    assert calls["clients"] == 0


def test_batch_binds_exact_order_all_sources_and_text_limitations(batch_boundary):
    args, batch, originals, calls = batch_boundary
    raw = json.loads(batch.read_text())
    raw["entries"].reverse()
    batch.write_text(json.dumps(raw))
    run(args)
    identity = json.loads((args.work_dir / "run_manifest.json").read_text())["identity"]
    assert identity["sources"] == {str(path): cli.file_digest(path) for path in originals}
    batch_identity = identity["training_evidence_batch"]
    assert batch_identity["manifest_sha256"] == cli.file_digest(batch)
    assert [row["evidence_id"] for row in batch_identity["entries"]] == ["TRAIN-2", "TRAIN-1"]
    assert calls["workflows"] == ["TRAIN-2", "TRAIN-1"]
    assert {p.name for p in args.result_dir.glob("*.docx")} == {"TRAIN-1.docx", "TRAIN-2.docx"}
    summary = read_summary(args)
    assert summary["reports"] == summary["completed_reports"] == 2
    assert summary["input_scope"] == "official_training_text_only"
    assert "image coverage" in summary["input_limitations"][1]
    assert summary["training_evidence_manifest_sha256"] == cli.file_digest(batch)
    assert summary["submission_ready"] is False
    # Per-worker additions belong to the copied namespace, not the caller.
    assert not hasattr(args, "divide_training_originals")


def test_check_parses_full_batch_without_running_or_creating_outputs(batch_boundary, capsys):
    args, _, _, calls = batch_boundary
    args.check = True
    run(args)
    assert calls["models"] == 1 and calls["workflows"] == []
    assert not args.work_dir.exists() and not args.result_dir.exists()
    assert "2 reports; original images/layout are not covered" in capsys.readouterr().out


def test_each_worker_keeps_eight_arguments_and_own_original(batch_boundary, monkeypatch):
    args, _, originals, calls = batch_boundary
    args.document_workers = 8
    jobs, pools = [], []
    worker = cli._run_document

    class SyntheticPool:
        def __init__(self, **kwargs):
            pools.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *job):
            jobs.append(job)
            assert function is worker and len(job) == 8
            future = Future()
            try:
                future.set_result(function(*job))
            except Exception as exc:
                future.set_exception(exc)
            return future

    monkeypatch.setattr(cli, "ProcessPoolExecutor", SyntheticPool)
    run(args)
    assert pools[0]["max_workers"] == 2
    assert pools[0]["max_tasks_per_child"] == 1
    assert pools[0]["mp_context"].get_start_method() == "spawn"
    for job, original in zip(jobs, originals):
        name, manifest, source, worker_args, _, _, digest, _ = job
        assert source is None and manifest["source_json_sha256"] == digest == cli.file_digest(original)
        assert worker_args.divide_training_originals[name] == original
    assert read_summary(args)["completed_reports"] == 2


@pytest.mark.parametrize("kind", ["manifest", "own_source"])
def test_worker_rechecks_before_creating_its_client(batch_boundary, monkeypatch, kind):
    args, batch, originals, calls = batch_boundary
    bind = cli.bind_work_directory

    def mutate_after_binding(*positional, **kwargs):
        fingerprint = bind(*positional, **kwargs)
        target = batch if kind == "manifest" else originals[0]
        target.write_bytes(target.read_bytes() + b"\n")
        return fingerprint

    monkeypatch.setattr(cli, "bind_work_directory", mutate_after_binding)
    with pytest.raises(RuntimeError, match="2 divide document"):
        run(args)
    assert "TRAIN-1" not in calls["workflows"]
    assert calls["clients"] == (1 if kind == "manifest" else 2)
    assert not (args.result_dir / "TRAIN-1.docx").exists()
    summary = read_summary(args)
    assert summary["reports"] == len(summary["failures"]) == 2
    assert summary["completed_reports"] == summary["completed_reports_per_hour"] == 0


@pytest.mark.parametrize("stage", ["workflow", "writer"])
def test_mutation_before_publication_never_exposes_that_output(batch_boundary, stage):
    args, _, originals, calls = batch_boundary

    def mutate(*_):
        originals[0].write_bytes(originals[0].read_bytes() + b"\n")

    calls["on_run" if stage == "workflow" else "on_write"] = mutate
    with pytest.raises(RuntimeError, match="2 divide document"):
        run(args)
    assert not (args.result_dir / "TRAIN-1.docx").exists()
    assert list(args.result_dir.glob(".pending-*")) == []
    assert read_summary(args)["completed_reports"] == 0


def test_later_source_drift_invalidates_prior_completion_and_preserves_denominator(batch_boundary):
    args, _, originals, calls = batch_boundary

    def change_finished_source(name):
        if name == "TRAIN-2":
            originals[0].write_bytes(originals[0].read_bytes() + b"\n")

    calls["on_run"] = change_finished_source
    with pytest.raises(RuntimeError, match="2 divide document"):
        run(args)
    # Keep earlier artifacts as evidence, but never score a drifted batch as success.
    assert (args.result_dir / "TRAIN-1.docx").exists()
    summary = read_summary(args)
    assert summary["reports"] == len(summary["failures"]) == 2
    assert summary["completed_reports"] == summary["completed_reports_per_hour"] == 0
    assert all(row["completed_before_input_revalidation"] for row in summary["documents"])
    persisted = json.loads((args.work_dir / "documents/TRAIN-1/document_summary.json").read_text())
    assert persisted["status"] == "failed"


def test_one_failed_report_stays_in_batch_denominator(batch_boundary):
    args, _, _, calls = batch_boundary

    def fail_second(name):
        if name == "TRAIN-2":
            raise RuntimeError("synthetic extraction failure")

    calls["on_run"] = fail_second
    with pytest.raises(RuntimeError, match="1 divide document"):
        run(args)
    summary = read_summary(args)
    assert summary["reports"] == 2 and summary["completed_reports"] == 1
    assert len(summary["failures"]) == 1
    assert summary["completed_reports_per_hour"] == 3600 / summary["seconds"]


def test_legacy_single_json_namespace_needs_no_batch_attribute(batch_boundary):
    args, _, originals, _ = batch_boundary
    del args.training_evidence_manifest
    args.training_evidence_json = originals[0]
    run(args)
    identity = json.loads((args.work_dir / "run_manifest.json").read_text())["identity"]
    assert "training_evidence_batch" not in identity
    assert "input_scope" not in read_summary(args)
    assert read_summary(args)["completed_reports"] == 1


def test_batch_main_dispatch_never_discovers_official_test_inputs(batch_boundary, monkeypatch):
    _, batch, _, _ = batch_boundary
    seen = []
    monkeypatch.setattr(sys, "argv", ["run.py", "--workflow", "divide-react",
                                     "--training-evidence-manifest", str(batch)])
    monkeypatch.setattr(entry, "select_input_dir", lambda: pytest.fail("Unexpected dataset discovery"))
    monkeypatch.setattr(cli, "run_divide", lambda args, **kwargs: seen.append(args.training_evidence_manifest))
    entry.main()
    assert seen == [batch]
