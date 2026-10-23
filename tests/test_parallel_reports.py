"""Four parallel reports: pipeline capacity, proof coverage and largest-first schedule."""
import importlib.util
import json
from pathlib import Path

import pytest

from bridge_evidence.contest import rsi_pipeline
from bridge_evidence.contest.rsi_pipeline import pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_runs_four_reports_in_flight_and_rejects_three():
    items = list(range(9))
    results = dict(pipeline(items, lambda x: x * 10, lambda i, v: v + i, workers=4))
    assert results == {i: i * 11 for i in items}
    with pytest.raises(ValueError):
        list(pipeline(items, lambda x: x, lambda i, v: v, workers=3))


def test_parallel_proof_must_cover_the_requested_worker_count(tmp_path, monkeypatch):
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text("{}")
    outputs = []
    for i in range(2):
        out = tmp_path / f"out{i}.docx"
        out.write_bytes(b"x%d" % i)
        outputs.append(out)
    monkeypatch.setattr(rsi_pipeline, "parallel_runtime_digest", lambda: "runtime")
    monkeypatch.setattr("bridge_evidence.contest.execution_audit._native_errors", lambda workflow: [])
    proof = tmp_path / "parallel.json"
    proof.write_text(json.dumps({
        "schema_version": 1, "url": "http://127.0.0.1:8011/v1", "overlap": True,
        "sdk_acceptance_sha256": rsi_pipeline.file_digest(acceptance),
        "runtime_sha256": rsi_pipeline.file_digest(ROOT / "run.py"), "protected_sha256": "runtime",
        "receipts": [{"document_sha256": f"d{i}", "workflow": {}, "model_call_ids": ["c"],
                      "output_path": str(out), "output_sha256": rsi_pipeline.file_digest(out)}
                     for i, out in enumerate(outputs)],
    }))
    url = "http://127.0.0.1:8011/v1"
    assert rsi_pipeline.verify_parallel_sdk(proof, acceptance, url)
    assert rsi_pipeline.verify_parallel_sdk(proof, acceptance, url, workers=2)
    assert not rsi_pipeline.verify_parallel_sdk(proof, acceptance, url, workers=4)


def test_largest_first_schedule_orders_by_size_then_name(tmp_path):
    spec = importlib.util.spec_from_file_location("contest_run", ROOT / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    paths = []
    for name, size in (("b.doc", 10), ("a.doc", 30), ("c.doc", 30), ("d.doc", 5)):
        path = tmp_path / name
        path.write_bytes(b"x" * size)
        paths.append(path)
    assert [p.name for p in run.order_reports(paths, "largest-first")] == ["a.doc", "c.doc", "b.doc", "d.doc"]
    assert [p.name for p in run.order_reports(paths)] == ["a.doc", "b.doc", "c.doc", "d.doc"]


def _load_run():
    spec = importlib.util.spec_from_file_location("contest_run", ROOT / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    return run


def test_longest_first_schedule_orders_by_main_text_not_photo_bytes(tmp_path):
    from docx import Document

    run = _load_run()
    paths = []
    for name, paragraphs, padding in (("short-heavy.docx", 2, 400_000), ("long-light.docx", 40, 0),
                                      ("middle.docx", 10, 0)):
        document = Document()
        for i in range(paragraphs):
            document.add_paragraph(f"第{i}段病害记录：主梁腹板竖向裂缝。")
        path = tmp_path / name
        document.save(path)
        if padding:  # an embedded photo makes the file large without adding text
            import os
            import zipfile

            with zipfile.ZipFile(path, "a", zipfile.ZIP_STORED) as archive:
                archive.writestr("word/media/photo.bin", os.urandom(padding))
        paths.append(path)
    assert paths[0].stat().st_size > paths[1].stat().st_size
    assert run.report_text_size(paths[1]) > run.report_text_size(paths[2]) > run.report_text_size(paths[0])
    assert [p.name for p in run.order_reports(paths, "longest-first")] == [
        "long-light.docx", "middle.docx", "short-heavy.docx"]


def test_word97_text_size_reads_fib_ccp_text(tmp_path, monkeypatch):
    import io
    import sys
    import types

    run = _load_run()
    fib = bytearray(512)
    fib[0:2] = b"\xec\xa5"
    fib[32:34] = (14).to_bytes(2, "little")  # csw
    fib[62:64] = (22).to_bytes(2, "little")  # cslw
    fib[76:80] = (123456).to_bytes(4, "little")  # FibRgLw97.ccpText

    class Ole:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def openstream(self, name):
            assert name == "WordDocument"
            return io.BytesIO(bytes(fib))

    monkeypatch.setitem(sys.modules, "olefile", types.SimpleNamespace(OleFileIO=Ole))
    path = tmp_path / "report.doc"
    path.write_bytes(b"ole")
    assert run.report_text_size(path) == 123456


def test_unreadable_report_falls_back_to_bytes(tmp_path):
    run = _load_run()
    broken, small = tmp_path / "broken.doc", tmp_path / "small.doc"
    broken.write_bytes(b"not an ole file" * 100)
    small.write_bytes(b"x")
    assert run.report_text_size(broken) is None
    assert [p.name for p in run.order_reports([small, broken], "longest-first")] == ["broken.doc", "small.doc"]


def test_four_workers_prepare_four_reports_ahead():
    import threading
    import time

    started, gate = [], threading.Event()

    def prepare(item):
        started.append(item)
        if len(started) == 4:
            gate.set()
        gate.wait(timeout=5)  # all four first preparations must run concurrently
        return item

    t0 = time.monotonic()
    results = dict(pipeline(list(range(6)), prepare, lambda i, v: v, workers=4))
    assert results == {i: i for i in range(6)}
    assert sorted(started[:4]) == [0, 1, 2, 3]
    assert time.monotonic() - t0 < 4  # no preparation waited for the 5 s gate timeout


def test_whole_doc_preparation_runs_both_conversions_side_by_side(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    from bridge_evidence.contest.run_state import file_digest

    run = _load_run()
    source = tmp_path / "report.doc"
    source.write_bytes(b"legacy word bytes")
    barrier = threading.Barrier(2, timeout=5)  # sequential conversions would break it

    def to_docx(path, cache_dir):
        barrier.wait()
        target = cache_dir / "k" / "report.docx"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"converted docx")
        (target.parent / "conversion.json").write_text(json.dumps({
            "source_sha256": file_digest(path), "output_sha256": file_digest(target),
            "conversion": "docx", "converter": "LibreOffice test"}))
        return target

    def to_text(path, cache_dir):
        barrier.wait()
        return "报告正文"

    monkeypatch.setattr(run, "convert_doc_to_docx", to_docx)
    monkeypatch.setattr(run, "convert_doc_to_text", to_text)
    args = SimpleNamespace(workflow="whole", text_dir=None, work_dir=tmp_path / "work",
                           visual_evidence="off", visual_display_transforms=False)
    prepared = run.prepare_document(args, source, tokenizer=None)
    assert prepared["text"] == "报告正文"
    assert prepared["source_evidence"]["conversion"]["conversion"] == "docx"
