"""CPU-only CLI wiring checks; no SDK execution, model requests or reports."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bridge_evidence.contest import document_cli as cli
from bridge_evidence.contest.document_workflow import DocumentSettings


def parser():
    result = argparse.ArgumentParser()
    cli.add_arguments(result)
    return result


def test_review_batch_defaults_preserve_existing_settings_and_quality_gates():
    args = parser().parse_args([])
    assert args.document_review_batch_items == 0
    assert cli._settings(args) == DocumentSettings()


@pytest.mark.parametrize("items", [0, 1, 4])
def test_review_batch_flag_only_changes_existing_batch_setting(items):
    args = parser().parse_args(["--document-review-batch-items", str(items)])
    assert cli._settings(args) == replace(DocumentSettings(), review_batch_items=items)


@pytest.mark.parametrize("value", [-1, 1.5, True, False, "2", None, []])
def test_review_batch_settings_reject_invalid_programmatic_values(value):
    args = parser().parse_args([])
    args.document_review_batch_items = value
    with pytest.raises(ValueError, match="document-review-batch-items.*nonnegative integer"):
        cli._settings(args)


@pytest.mark.parametrize("value", ["1.5", "true", "invalid"])
def test_review_batch_cli_rejects_noninteger_tokens(value):
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["--document-review-batch-items", value])
    assert error.value.code == 2


def test_review_batch_cli_rejects_negative_before_workflow_start():
    args = parser().parse_args(["--document-review-batch-items", "-1"])
    with pytest.raises(ValueError, match="document-review-batch-items.*nonnegative integer"):
        cli._settings(args)


def test_run_entrypoint_passes_batch_setting_to_document_workflow(monkeypatch):
    spec = importlib.util.spec_from_file_location("review_batch_contest_entry", ROOT / "run.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    received = []

    def capture(args, **kwargs):
        received.append(cli._settings(args))

    monkeypatch.setattr(cli, "run_document_agent", capture)
    monkeypatch.setattr(sys, "argv", ["run.py", "--workflow", "document-agent",
                                     "--document-review-batch-items", "3"])
    entry.main()
    assert received == [replace(DocumentSettings(), review_batch_items=3)]
