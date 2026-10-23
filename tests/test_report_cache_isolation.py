"""Independent reports must not share checkpoints or generation call receipts."""

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from threading import Barrier

import pytest
from docx import Document
from test_chunked_integration import CharTokenizer, FragmentBackend

import run
from bridge_evidence.contest.candidate import validate_config
from bridge_evidence.contest.run_state import json_digest


def setup_reports(tmp_path, names):
    inputs = tmp_path / "input"
    inputs.mkdir()
    seed = tmp_path / "seed.docx"
    doc = Document()
    doc.add_paragraph("测试桥")
    doc.save(seed)
    paths, arguments = [], []
    for index, name in enumerate(names):
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(seed.read_bytes())
        result = tmp_path / f"result-{index}"
        result.mkdir()
        args = run.build_parser().parse_args([
            "--engine", "direct", "--workflow", "chunked",
            "--input-dir", str(inputs), "--result-dir", str(result),
            "--work-dir", str(tmp_path / "work"),
        ])
        args.candidate = validate_config({"workflow": "chunked", "input_budget": 4000,
                                          "output_budget": 500})
        paths.append(path)
        arguments.append(args)
    return paths, arguments


def process(path, args, backend, **kwargs):
    return run.process_document(args, path, CharTokenizer(), backend,
                                fingerprint="f" * 64, **kwargs)


@pytest.mark.parametrize("names", [("one.docx", "two.docx"),
                                   ("north/bridge.docx", "south/bridge.docx")])
def test_same_bytes_different_relative_reports_have_independent_receipts(tmp_path, names):
    paths, args = setup_reports(tmp_path, names)
    backend = FragmentBackend()
    records = [process(path, options, backend) for path, options in zip(paths, args)]
    assert backend.calls == 2
    assert records[0]["document"]["sha256"] == records[1]["document"]["sha256"]
    assert records[0]["model_calls"][0]["call_id"] != records[1]["model_calls"][0]["call_id"]
    for record in records:
        path = tmp_path / "work" / "task_cache" / "reports" / json_digest(record["document"])
        assert len(list(path.glob("*/checkpoint.sqlite"))) == 1


def test_same_bytes_parallel_reports_keep_independent_checkpoints(tmp_path):
    paths, args = setup_reports(tmp_path, ("north/bridge.docx", "south/bridge.docx"))
    barrier = Barrier(2, timeout=2)
    backends = [FragmentBackend(), FragmentBackend()]

    def execute(index):
        def generate(ids, maximum):
            barrier.wait()
            return backends[index](ids, maximum)
        return process(paths[index], args[index], generate)

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(execute, range(2)))
    assert [backend.calls for backend in backends] == [1, 1]
    ids = [record["model_calls"][0]["call_id"] for record in records]
    assert len(set(ids)) == 2
    checkpoints = list((tmp_path / "work" / "task_cache").rglob("checkpoint.sqlite"))
    assert len(checkpoints) == 2


def test_render_failure_resumes_only_its_original_report_cache(tmp_path):
    paths, args = setup_reports(tmp_path, ("one.docx", "two.docx"))
    backend = FragmentBackend()
    first = process(paths[0], args[0], backend)

    def fail_render(*args):
        raise OSError("controlled renderer failure")

    with pytest.raises(run.DocumentProcessingError) as caught:
        process(paths[1], args[1], backend, writer=fail_render)
    failed_id = caught.value.record["model_calls"][0]["call_id"]
    assert failed_id != first["model_calls"][0]["call_id"]
    resume = copy(args[1])
    resume.resume = True

    def no_new_generation(*args):
        pytest.fail("completed original report task should be reused")

    recovered = process(paths[1], resume, no_new_generation)
    assert backend.calls == 2
    assert recovered["model_calls"][0]["call_id"] == failed_id
    assert recovered["generation_evidence"] == caught.value.record["generation_evidence"]
