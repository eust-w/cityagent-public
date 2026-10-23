"""Both inference families remain explicitly selectable after the branch merge."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def entry():
    path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("merged_contest_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merged_defaults_keep_existing_inference_and_workspace(entry):
    args = entry.build_parser().parse_args([])
    assert args.workflow == "whole"
    assert args.rsi_mode == "off"
    assert args.visual_evidence == "off"
    assert args.document_vision == "none"
    assert args.document_ocr == "none"
    assert args.result_dir == Path("/workspace/result")
    assert args.work_dir == Path("/workspace/work/contest_run")


@pytest.mark.parametrize("workflow", ["whole", "chunked", "bounded-react"])
def test_existing_workflows_keep_the_locked_executor(entry, monkeypatch, workflow):
    from bridge_evidence.contest import divide_cli, document_cli, rsi_runtime

    calls = []
    monkeypatch.setattr(sys, "argv", ["run.py", "--workflow", workflow])
    monkeypatch.setattr(rsi_runtime, "WorkLock", lambda path: nullcontext())
    monkeypatch.setattr(entry, "_main_locked", lambda: calls.append("existing"))
    monkeypatch.setattr(divide_cli, "run_divide", lambda *a, **k: calls.append("divide"))
    monkeypatch.setattr(document_cli, "run_document_agent", lambda *a, **k: calls.append("document"))
    entry.main()
    assert calls == ["existing"]


@pytest.mark.parametrize("workflow", ["divide-fixed", "divide-react", "document-agent"])
def test_explicit_agent_route_preserves_the_shared_converter(entry, monkeypatch, workflow):
    from bridge_evidence.contest import divide_cli, document_cli

    calls = []

    def capture(args, *, code_root, convert_doc):
        calls.append((args.workflow, code_root, convert_doc))

    monkeypatch.setattr(sys, "argv", ["run.py", "--workflow", workflow])
    monkeypatch.setattr(entry, "_main_locked", lambda: pytest.fail("wrong executor"))
    monkeypatch.setattr(divide_cli, "run_divide", capture)
    monkeypatch.setattr(document_cli, "run_document_agent", capture)
    entry.main()
    assert calls == [(workflow, entry.ROOT, entry.convert_doc_to_docx)]


@pytest.mark.parametrize("option", [
    ["--rsi-mode", "frozen"],
    ["--rsi-bundle", "bundle.json"],
    ["--rsi-release", "release.json"],
    ["--candidate-config", "candidate.json"],
])
def test_document_route_cannot_silently_ignore_existing_strategy(entry, monkeypatch, option):
    from bridge_evidence.contest import document_cli

    monkeypatch.setattr(sys, "argv", ["run.py", "--workflow", "document-agent", *option])
    monkeypatch.setattr(document_cli, "run_document_agent", lambda *a, **k: pytest.fail("must reject before dispatch"))
    with pytest.raises(ValueError, match="document-agent requires RSI off"):
        entry.main()
