"""Fresh-run total budgets using fake clocks and synthetic documents only.

No SDK, HTTP listener, remote model or real report is used. Request doubles
below record the timeout an actual gateway would pass to its upstream client.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_document_cli import boundary, read, run
from bridge_evidence.contest import document_cli as cli
from bridge_evidence.contest import document_workflow as workflow_module
from bridge_evidence.contest import document_model_gateway as gateway_module
from bridge_evidence.contest.evidence_manifest import manifest_from_text


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    perf_counter = monotonic

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(cli, "time", clock)
    monkeypatch.setattr(workflow_module, "time", clock)
    monkeypatch.setattr(gateway_module, "time", clock)
    return clock


def capture_workflow(boundary, monkeypatch, clock):
    _, _, calls, workflow, _ = boundary
    settings_type, _ = cli._workflow_types()

    class CapturingWorkflow(workflow):
        def __init__(self, *args, external_deadline=None, **kwargs):
            calls["external_deadline"] = external_deadline
            calls["workflow_started"] = clock.now
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cli, "_workflow_types", lambda: (settings_type, CapturingWorkflow))


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf"), True, "10"])
def test_invalid_total_budget_rejected_before_client(boundary, value):
    args, _, calls, *_ = boundary
    args.document_total_report_timeout = value
    with pytest.raises(ValueError, match="positive and finite"):
        run(boundary)
    assert calls["clients"] == calls["workflows"] == calls["writes"] == 0
    assert not args.work_dir.exists()


def test_total_budget_rejects_resume_before_client(boundary):
    args, _, calls, *_ = boundary
    args.document_total_report_timeout = 10
    args.resume = True
    with pytest.raises(ValueError, match="fresh-run only"):
        run(boundary)
    assert calls["clients"] == calls["workflows"] == calls["writes"] == 0


def test_absent_flag_preserves_old_identity_and_resume(boundary):
    args, _, calls, *_ = boundary
    assert args.document_total_report_timeout is None
    assert run(boundary)["completed_reports"] == 1
    identity = read(args.work_dir / "run_manifest.json")["identity"]
    assert "document_total_report_timeout" not in identity
    assert "document_total_report_timeout" not in calls["run_identity"]
    # The old Workflow double does not accept external_deadline: the default
    # path therefore verifies the keyword is absent, not merely passed as None.
    args.resume = True
    assert run(boundary)["reused_reports"] == 1
    assert calls["workflows"] == 1


def structural_input(boundary, monkeypatch):
    # These tests inject a converter and a fake clock; no external binary runs.
    # Make the dependency-discovery seam explicit on hosts without LibreOffice.
    original_which = cli.shutil.which
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/fake/soffice" if name == "soffice" else original_which(name))
    args = boundary[0]
    args.training_evidence_json = None
    args.input_dir = args.work_dir.parent / "originals"
    args.input_dir.mkdir()
    source = args.input_dir / "synthetic.doc"
    source.write_bytes(b"synthetic conversion source")
    return source


def converter(clock, seconds):
    def convert(source, directory):
        from docx import Document
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (source.stem + ".docx")
        document = Document()
        document.add_paragraph("桥梁名称：合成桥")
        document.save(target)
        clock.advance(seconds)
        return target
    return convert


def test_preparation_and_workflow_share_one_bound_deadline(boundary, monkeypatch, clock):
    args, code_root, calls, *_ = boundary
    source = structural_input(boundary, monkeypatch)
    args.document_total_report_timeout = 10
    capture_workflow(boundary, monkeypatch, clock)
    original_build = cli.build_manifest

    def build(path):
        result = original_build(path)
        clock.advance(3)
        return result

    monkeypatch.setattr(cli, "build_manifest", build)
    result = cli.run_document_agent(args, code_root=code_root, convert_doc=converter(clock, 4))
    assert result["completed_reports"] == 1
    assert calls["workflow_started"] == 1007
    assert calls["external_deadline"] == 1010
    assert calls["external_deadline"] - calls["workflow_started"] == 3
    identity = read(args.work_dir / "run_manifest.json")["identity"]
    assert identity["document_total_report_timeout"] == 10
    assert calls["run_identity"]["document_total_report_timeout"] == 10
    assert "external_deadline" not in calls["run_identity"]
    assert read(calls["directory"] / "source_binding.json")["original_source_sha256"] == cli.file_digest(source)


def test_conversion_at_deadline_never_starts_workflow(boundary, monkeypatch, clock):
    args, code_root, calls, *_ = boundary
    structural_input(boundary, monkeypatch)
    args.document_total_report_timeout = 10

    def unexpected_manifest(_):
        pytest.fail("expired conversion must stop before manifest parsing")

    monkeypatch.setattr(cli, "build_manifest", unexpected_manifest)
    with pytest.raises(RuntimeError, match="1 document-agent report"):
        cli.run_document_agent(args, code_root=code_root, convert_doc=converter(clock, 10))
    assert calls["workflows"] == calls["writes"] == 0
    summary = read(args.work_dir / "summary.json")
    assert "after_conversion" in summary["failures"][0]["error"]
    assert not list(args.result_dir.glob("*.docx"))


@pytest.mark.parametrize("stage", ["write", "readback", "before_link"])
def test_expired_export_never_publishes_and_removes_pending(boundary, monkeypatch, clock, stage):
    args, _, calls, *_ = boundary
    args.document_total_report_timeout = 10
    capture_workflow(boundary, monkeypatch, clock)
    if stage == "write":
        original = cli.write_extraction_docx

        def late_write(*args):
            original(*args)
            clock.advance(10)

        monkeypatch.setattr(cli, "write_extraction_docx", late_write)
    elif stage == "readback":
        original = cli.validate_delivery_docx

        def late_readback(*args):
            result = original(*args)
            clock.advance(10)
            return result

        monkeypatch.setattr(cli, "validate_delivery_docx", late_readback)
    else:
        original = cli.file_digest

        def late_digest(path):
            digest = original(path)
            if path.name.startswith(".pending-"):
                clock.advance(10)
            return digest

        monkeypatch.setattr(cli, "file_digest", late_digest)
    with pytest.raises(RuntimeError, match="1 document-agent report"):
        run(boundary)
    assert calls["workflows"] == calls["writes"] == 1
    assert not list(args.result_dir.glob("*.docx"))
    assert not list(args.result_dir.glob(".pending-*"))
    failure = read(args.work_dir / "summary.json")["failures"][0]
    assert "Document total report deadline exhausted" in failure["error"]
    marker = calls["directory"] / "publication.json"
    assert not marker.exists() or read(marker)["status"] == "prepared"


def test_workflow_result_at_deadline_never_starts_export(boundary, monkeypatch, clock):
    args, _, calls, workflow, _ = boundary
    args.document_total_report_timeout = 10
    capture_workflow(boundary, monkeypatch, clock)
    original = workflow.run

    def late_run(self):
        result = original(self)
        clock.advance(10)
        return result

    monkeypatch.setattr(workflow, "run", late_run)
    with pytest.raises(RuntimeError, match="1 document-agent report"):
        run(boundary)
    assert calls["writes"] == 0
    assert "after_workflow" in read(args.work_dir / "summary.json")["failures"][0]["error"]


def synthetic_workflow(tmp_path, clock, *, external_deadline=None, report_timeout=20):
    client = SimpleNamespace(model="synthetic", base_url="http://unused.invalid",
                             max_model_len=65536, timeout=180.0, calls=[])
    return workflow_module.DocumentWorkflow(
        manifest_from_text("合成内容"), client, {"model": "synthetic"}, tmp_path,
        settings=workflow_module.DocumentSettings(report_timeout=report_timeout),
        external_deadline=external_deadline)


@pytest.mark.parametrize("external,expected", [(None, 1020), (1010, 1010), (1030, 1020)])
def test_external_deadline_only_caps_existing_budget(tmp_path, clock, external, expected):
    wf = synthetic_workflow(tmp_path, clock, external_deadline=external)
    assert wf.deadline == expected
    assert "external_deadline" not in wf.identity


def test_workflow_capped_deadline_reaches_real_gateway_request_timeout(tmp_path, clock):
    # Emulate seven seconds already spent in source preparation. The real
    # gateway runs its count/request stages against an in-memory client only.
    clock.advance(7)
    wf = synthetic_workflow(tmp_path, clock, external_deadline=1010)
    timeouts = []

    def tokenize(payload):
        timeouts.append(("tokenize", wf.client.timeout))
        clock.advance(2)
        return 3

    def request(path, payload, *, kind):
        timeouts.append(("generation", wf.client.timeout))
        receipt = {"call_id": "synthetic-1", "kind": kind, "status": "success"}
        wf.client.calls.append(receipt)
        return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1}}, receipt

    wf.client._tokenize = tokenize
    wf.client._request = request
    gateway = gateway_module.AuditedModelGateway(wf.client, deadline=wf.deadline,
        max_calls=1, max_output_tokens=4, request_kind="generation")
    gateway.forward({"model": "synthetic", "messages": [{"role": "user", "content": "合成问题"}],
                     "max_tokens": 4, "stream": False, "temperature": 0,
                     "chat_template_kwargs": {"enable_thinking": False}})
    assert timeouts == [("tokenize", 3), ("generation", 1)]
    assert wf.client.timeout == 180
    assert len(wf.client.calls) == 1


@pytest.mark.parametrize("deadline", [0, -1, float("nan"), float("inf"), True, "1010"])
def test_invalid_external_deadline_rejected_without_artifacts(tmp_path, clock, deadline):
    directory = tmp_path / "not-created"
    with pytest.raises(ValueError, match="positive finite"):
        synthetic_workflow(directory, clock, external_deadline=deadline)
    assert not directory.exists()
