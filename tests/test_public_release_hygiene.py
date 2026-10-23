"""Exercise publication exclusions with real Git and case-sensitive rules."""
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def public_ignore_repository(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "core.ignorecase", "false"], check=True)
    source = Path(__file__).resolve().parents[1] / ".gitignore"
    (tmp_path / ".gitignore").write_bytes(source.read_bytes())
    return tmp_path


def _ignored(repository: Path, paths: list[str]) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(repository), "-c", "core.excludesFile=/dev/null",
         "check-ignore", "--no-index", "-z", "--stdin"],
        input="\0".join(paths) + "\0", capture_output=True, text=True,
    )
    assert result.returncode in (0, 1), result.stderr
    return set(filter(None, result.stdout.split("\0")))


def test_documented_outputs_and_local_configurations_are_ignored(public_ignore_repository):
    paths = [
        "outputs/evidence.json", "outputs-json/report.json",
        "outputs-online/report.json", "outputs-online-json/report.json",
        "output/receipt.json", "output-json/report.json",
        "result/receipt.json", "results/receipt.json",
        "result-json/report.json", "results-online-json/report.json",
        "my-config.json", "my-online-config.json", "evaluation.json",
        ".env", ".env.local", ".bridge_evidence/work/events.jsonl",
    ]
    assert _ignored(public_ignore_repository, paths) == set(paths)


def test_private_key_names_and_case_variant_report_images_are_ignored(public_ignore_repository):
    paths = [
        "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
        "id_ecdsa_sk", "id_ed25519_sk", ".ssh/config",
        "local/id_ed25519", "local/identity.PEM", "local/identity.Key",
        "reports/input.DOC", "reports/input.DoCx", "reports/input.PDF",
        "reports/image.JPG", "reports/image.JpEg", "reports/image.PNG",
        "reports/image.GIF", "reports/image.BMP", "reports/image.TIF",
        "reports/image.TIFF", "reports/image.WEBP", "reports/image.HEIC",
        "reports/image.HEIF", "reports/drawing.EMF", "reports/drawing.WmF",
    ]
    assert _ignored(public_ignore_repository, paths) == set(paths)


def test_public_configuration_and_synthetic_text_fixtures_remain_trackable(public_ignore_repository):
    paths = [
        ".env.example", "configs/base.json", "configs/prompts.json",
        "configs/contest/chunked-baseline.json",
        "tests/fixtures/synthetic_managed_handoff.json",
        "examples/offline_demo.py", "docs/usage.md",
        "examples/config.json", "tests/fixtures/synthetic-report.txt",
    ]
    assert not _ignored(public_ignore_repository, paths)
