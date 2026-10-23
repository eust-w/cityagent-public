"""Synthetic recovery and filesystem tests; not real model/SDK acceptance."""
from __future__ import annotations

import errno
import json
from pathlib import Path
from uuid import uuid4

import pytest
from test_divide_cli import ROOT
from test_divide_cli import boundary as _boundary
from test_divide_conquer_workflow import FakeLocalClient, fake_react

from bridge_evidence.contest import divide_cli as cli
from bridge_evidence.contest.divide_conquer_workflow import (
    DivideSettings,
    DivideWorkflow,
)
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.rsi_runtime import WorkLock


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    return _boundary.__wrapped__(tmp_path, monkeypatch)


class RecoveryClient(FakeLocalClient):
    def generate(self, messages, max_tokens=None):
        result = super().generate(messages, max_tokens=max_tokens)
        result["call_id"] = uuid4().hex
        self.calls[-1]["call_id"] = result["call_id"]
        self.calls[-1]["seconds"] = 1.0
        return result


def make(tmp_path, *, resume=False, text=None):
    manifest = manifest_from_text(text or "桥梁名称：合成桥\n病害：右幅桥台裂缝\n病害：左幅桥墩裂缝\n")
    client = RecoveryClient(manifest)
    workflow = DivideWorkflow(manifest, client, {}, tmp_path,
                              DivideSettings(chunk_tokens=200, max_tool_tokens=10000),
                              mode="divide-fixed", resume=resume)
    return workflow, client


def test_resume_does_not_regenerate_completed_tasks_or_duplicate_calls(tmp_path):
    workflow, client = make(tmp_path)
    generate = client.generate

    def interrupt_second(messages, max_tokens=None):
        if client.calls:
            raise KeyboardInterrupt("process interrupted")
        return generate(messages, max_tokens=max_tokens)

    client.generate = interrupt_second
    with pytest.raises(KeyboardInterrupt):
        workflow.run()
    saved = json.loads((tmp_path / "execution.json").read_text())
    first = next(task for task in saved["tasks"] if task["status"] == "completed")
    resumed, second_client = make(tmp_path, resume=True)
    result = resumed.run()
    assert first["id"] not in {request["task_id"] for request in second_client.requests}
    calls = result["receipt"]["model_calls"]
    assert len(calls) == len({call["call_id"] for call in calls}) == 3
    assert result["receipt"]["coverage"]["complete"]
    assert len(result["data"]["病害明细"]) == 2
    assert result["receipt"]["recovery"]["resumed"]
    interrupted = [attempt for task in result["receipt"]["tasks"] for attempt in task["attempts"]
                   if attempt.get("interrupted")]
    assert len(interrupted) == 1 and "call_id" not in interrupted[0]


def test_response_durable_before_task_checkpoint_can_be_recovered(tmp_path):
    workflow, client = make(tmp_path)
    checkpoint = workflow.checkpoint

    def interrupt_after_response(*args, **kwargs):
        if any(task["status"] == "extracted" for task in workflow.tasks):
            raise KeyboardInterrupt("receipt persisted before checkpoint")
        return checkpoint(*args, **kwargs)

    workflow.checkpoint = interrupt_after_response
    with pytest.raises(KeyboardInterrupt):
        workflow.run()
    resumed, second_client = make(tmp_path, resume=True)
    result = resumed.run()
    assert len(second_client.requests) == 2
    assert len(result["receipt"]["model_calls"]) == 3
    assert result["receipt"]["coverage"]["complete"]


@pytest.mark.parametrize("change", ["source", "settings", "response"])
def test_resume_rejects_changed_identity_or_corrupt_cached_response(tmp_path, change):
    workflow, _ = make(tmp_path)
    workflow.run()
    resumed, client = make(tmp_path, resume=True,
                           text="桥梁名称：另外一座桥" if change == "source" else None)
    if change == "settings":
        resumed.settings = DivideSettings(chunk_tokens=300)
    if change == "response":
        response = next(tmp_path.glob("task-*-attempt-*.json"))
        data = json.loads(response.read_text())
        data["text"] = "tampered response"
        response.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity|response|checkpoint"):
        resumed.run()
    assert client.calls == []


def test_second_owner_cannot_recover_running_workflow(tmp_path):
    workflow, client = make(tmp_path, resume=True)
    with WorkLock(tmp_path):
        with pytest.raises(RuntimeError, match="active owner"):
            workflow.run()
    assert client.calls == []


def test_divide_publication_uses_mount_compatible_helper(boundary, monkeypatch):
    args, _, _, _ = boundary

    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "mount does not support hardlinks")

    monkeypatch.setattr(cli.os, "link", unsupported)
    cli.run_divide(args, code_root=ROOT, convert_doc=lambda *_: pytest.fail("conversion"))
    assert (args.result_dir / "TRAIN-1.docx").is_file()
    receipt = json.loads((args.work_dir / "documents/TRAIN-1/execution.json").read_text())
    assert receipt["publication"]["mode"] == "locked-replace"


def test_cli_resume_reuses_published_bound_artifact(boundary):
    args, _, _, calls = boundary
    cli.run_divide(args, code_root=ROOT, convert_doc=lambda *_: pytest.fail("conversion"))
    args.resume = True
    cli.run_divide(args, code_root=ROOT, convert_doc=lambda *_: pytest.fail("conversion"))
    assert calls["workflows"] == calls["writes"] == 1
    summary = json.loads((args.work_dir / "summary.json").read_text())
    assert summary["reused_reports"] == ["TRAIN-1"]
    assert summary["seconds"] >= summary["invocations"][-1]["seconds"]


def test_cli_resume_rejects_modified_published_artifact(boundary):
    args, _, _, calls = boundary
    cli.run_divide(args, code_root=ROOT, convert_doc=lambda *_: pytest.fail("conversion"))
    (args.result_dir / "TRAIN-1.docx").write_bytes(b"corrupt")
    args.resume = True
    with pytest.raises(RuntimeError, match="1 divide document"):
        cli.run_divide(args, code_root=ROOT, convert_doc=lambda *_: pytest.fail("conversion"))
    assert calls["workflows"] == calls["writes"] == 1


def test_conflict_response_reused_but_sdk_finish_is_not_fabricated(tmp_path):
    manifest = manifest_from_text("桥梁名称：合成桥\n报告日期：2018-12-16\n报告日期：2018-12-20\n")
    client = RecoveryClient(manifest)

    def interrupted_finish(prompt, handlers, config):
        active = json.loads(prompt.rsplit("\n", 1)[1])
        if "task_id" in active:
            return fake_react(prompt, handlers, config)
        handlers["review_conflict"](conflict_id=active["id"], extra_evidence_ids=[])
        raise KeyboardInterrupt("SDK finish not delivered")

    settings = DivideSettings(chunk_tokens=200, max_tool_tokens=10000)
    workflow = DivideWorkflow(manifest, client, {}, tmp_path, settings,
                              mode="divide-react", react_runner=interrupted_finish)
    with pytest.raises(KeyboardInterrupt):
        workflow.run()
    second = RecoveryClient(manifest)
    sdk_calls = []

    def finishing(prompt, handlers, config):
        sdk_calls.append(prompt)
        return fake_react(prompt, handlers, config)

    resumed = DivideWorkflow(manifest, second, {}, tmp_path, settings,
                             mode="divide-react", react_runner=finishing, resume=True)
    result = resumed.run()
    assert second.calls == []
    assert len(sdk_calls) == 1
    assert result["data"]["基本信息"]["报告日期"] == "2018-12-20"
    assert len(result["receipt"]["reviews"][0]["attempts"]) == 1


def test_post_publication_receipt_failure_rolls_back_only_own_output(boundary, monkeypatch):
    args, _, _, _ = boundary
    original = cli.atomic_json

    def fail_publication_receipt(path, data):
        if Path(path).name == "execution.json" and "publication" in data:
            raise OSError("receipt disk failure")
        return original(path, data)

    monkeypatch.setattr(cli, "atomic_json", fail_publication_receipt)
    with pytest.raises(RuntimeError, match="1 divide document"):
        cli.run_divide(args, code_root=ROOT, convert_doc=lambda *_: pytest.fail("conversion"))
    assert not (args.result_dir / "TRAIN-1.docx").exists()


def test_prepare_overlap_uses_isolated_conversion_directories(tmp_path, monkeypatch):
    from threading import Barrier
    from types import SimpleNamespace

    sources = [tmp_path / "one.doc", tmp_path / "two.doc"]
    for source in sources:
        source.write_bytes(source.name.encode())
    args = SimpleNamespace(work_dir=tmp_path / "work", result_dir=tmp_path / "results", resume=False)
    identity = {"sources": {str(source.resolve()): cli.file_digest(source) for source in sources}}
    barrier = Barrier(2)
    directories = []

    def convert(source, directory):
        directories.append(directory)
        barrier.wait(timeout=3)
        return source

    monkeypatch.setattr(cli, "build_manifest", lambda source: {"source": source.name})
    results = list(cli._prepared_reports([(s.stem, None, s) for s in sources], args, identity, convert))
    assert all(error is None for _, error in results)
    assert len(set(directories)) == 2


def test_local_http_receipt_sink_reserves_before_network_and_records_interrupt():
    from bridge_evidence.contest.local_model_client import LocalModelClient

    events = []
    client = LocalModelClient("http://127.0.0.1:8000", "model",
                              receipt_sink=lambda event: events.append(json.loads(json.dumps(event))))

    class InterruptedOpener:
        def open(self, request, timeout=None):
            assert len(events) == 1 and events[0]["status"] == "running"
            raise KeyboardInterrupt("request outcome unknown")

    client._opener = InterruptedOpener()
    with pytest.raises(KeyboardInterrupt):
        client.count_tokens("source")
    assert len(events) == 2
    assert events[0]["call_id"] == events[1]["call_id"]
    assert events[-1]["status"] == "interrupted"
    assert events[-1]["output_sha256"] is None


def test_local_receipt_sink_records_post_response_schema_failure():
    from bridge_evidence.contest.local_model_client import LocalModelClient

    events = []
    client = LocalModelClient("http://127.0.0.1:8000", "model",
                              receipt_sink=lambda event: events.append(json.loads(json.dumps(event))))

    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return b'{"count":"invalid"}'

    class Opener:
        def open(self, *args, **kwargs):
            return Response()

    client._opener = Opener()
    with pytest.raises(RuntimeError, match="nonnegative integer"):
        client.count_tokens("source")
    assert events[-1]["status"] == "failed"
    assert len({event["call_id"] for event in events}) == 1


def test_success_writes_bound_fact_graph_outside_submission_directory(tmp_path):
    workflow, _ = make(tmp_path)
    result = workflow.run()
    reference = result["receipt"]["fact_graph"]
    artifact = tmp_path / reference["path"]
    assert cli.file_digest(artifact) == reference["sha256"]
    graph = json.loads(artifact.read_text())
    assert graph["document"]["sha256"] == workflow.manifest["document_sha256"]
    assert graph["observations"]


def test_durable_call_reservation_survives_without_model_return(tmp_path):
    workflow, client = make(tmp_path)

    def vanished_model(messages, max_tokens=None):
        client.receipt_sink({"call_id": "actual-reserved-call", "kind": "generation", "status": "running",
                             "reservation_id": client.request_context["reservation_id"], "seconds": 0.0})
        raise KeyboardInterrupt("killed before any response")

    client.generate = vanished_model
    with pytest.raises(KeyboardInterrupt):
        workflow.run()
    resumed, _ = make(tmp_path, resume=True)
    result = resumed.run()
    lost = next(call for call in result["receipt"]["model_calls"] if call["call_id"] == "actual-reserved-call")
    assert lost["status"] == "interrupted" and not lost["timing_complete"]
    assert not lost.get("output_sha256")


def test_real_loopback_http_receipts_remain_unique_when_resuming(tmp_path):
    from test_local_model_client import server

    from bridge_evidence.contest.local_model_client import LocalModelClient
    from bridge_evidence.contest.rsi_runtime import Dispatcher

    manifest = manifest_from_text("桥梁名称：合成桥")
    evidence_id = manifest["units"][0]["id"]
    text = json.dumps({"基本信息": {"桥梁名称": "合成桥"}, "病害明细": [], "维修建议": [],
                       "evidence": {"基本信息": {"桥梁名称": [evidence_id]}, "病害明细": [], "维修建议": []}})
    with server(text=text) as (url, requests):
        client = LocalModelClient(url, "fixture-model")
        with Dispatcher(client) as dispatcher:
            result = DivideWorkflow(manifest, client, {}, tmp_path, mode="divide-fixed", dispatcher=dispatcher).run()
        calls = result["receipt"]["model_calls"]
        assert len(calls) == len(requests)
        assert all(call["status"] == "success" for call in calls)
        assert len({call["call_id"] for call in calls}) == len(calls)
        fresh = LocalModelClient(url, "fixture-model")
        with Dispatcher(fresh) as dispatcher:
            resumed = DivideWorkflow(manifest, fresh, {}, tmp_path, mode="divide-fixed",
                                     dispatcher=dispatcher, resume=True).run()
        assert fresh.calls == []
        assert resumed["receipt"]["model_calls"] == calls
        assert resumed["data"] == result["data"]
