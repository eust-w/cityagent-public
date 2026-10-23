from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import pytest

from bridge_evidence.config import Settings, load_prompts, load_settings


@pytest.fixture(autouse=True)
def isolated_config_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for name in os.environ:
        if name.startswith("BRIDGE_EVIDENCE_"):
            monkeypatch.delenv(name, raising=False)


def test_public_defaults_are_offline_and_do_not_use_private_gateways() -> None:
    for settings in (Settings(), load_settings()):
        assert settings.llm_mode == "off"
        assert not settings.llm_enabled
        assert settings.base_url == "http://127.0.0.1:8000/v1"
        assert settings.semantic_base_url == "http://127.0.0.1:8001/v1"
        assert settings.vision_model is None
        assert settings.vision_critic_model is None
        assert settings.api_key_env == "BRIDGE_EVIDENCE_API_KEY"
        assert settings.semantic_api_key_env == "BRIDGE_EVIDENCE_SEMANTIC_API_KEY"


def test_packaged_resources_are_independent_of_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_settings().llm_mode == "off"
    assert load_prompts()["fact_system"]
    assert files("bridge_evidence.resources").joinpath("base.json").is_file()


def test_explicit_configuration_and_overrides_remain_supported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "custom.json"
    config.write_text(json.dumps({"base_url": "http://127.0.0.1:9100/v1", "model": "local-test"}), encoding="utf-8")
    monkeypatch.setenv("BRIDGE_EVIDENCE_MODEL", "environment-test")
    settings = load_settings(config, model="argument-test")
    assert settings.model == "argument-test"
    assert settings.base_url == "http://127.0.0.1:9100/v1"
    assert settings.llm_mode == "off"
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text('{"fact_system":"explicit"}', encoding="utf-8")
    assert load_prompts(prompt_file) == {"fact_system": "explicit"}


def test_online_mode_requires_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(llm_mode="on", model="local-model")
    with pytest.raises(RuntimeError, match="BRIDGE_EVIDENCE_API_KEY"):
        _ = settings.llm_enabled
    monkeypatch.setenv("BRIDGE_EVIDENCE_API_KEY", "test-only-local-placeholder")
    with pytest.raises(RuntimeError, match="BRIDGE_EVIDENCE_SEMANTIC_API_KEY"):
        _ = settings.llm_enabled
    monkeypatch.setenv("BRIDGE_EVIDENCE_SEMANTIC_API_KEY", "test-only-local-placeholder")
    assert settings.llm_enabled


def test_checked_in_config_matches_packaged_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    assert json.loads((root / "configs/base.json").read_text()) == json.loads(
        files("bridge_evidence.resources").joinpath("base.json").read_text(encoding="utf-8")
    )
    assert json.loads((root / "configs/prompts.json").read_text()) == load_prompts()


def test_fictional_offline_demo_runs_the_real_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "demo"
    # The fixed demo must not inherit a user's external document-expert command.
    monkeypatch.setenv("BRIDGE_EVIDENCE_DOCUMENT_EXPERT_COMMANDS", "not-json")
    completed = subprocess.run(
        [sys.executable, str(root / "examples/offline_demo.py"), "--output-dir", str(destination)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads((destination / "json/synthetic-report.json").read_text(encoding="utf-8"))
    assert data["validation"]["passed"] is True
    assert data["bridge"]["name"]["value"] == "云样桥"
    assert data["metrics"]["llm_calls"] == 0
    assert data["metrics"]["chat_calls"] == 0
    assert len(data["diseases"]) == 1
    assert len(data["recommendations"]) == 1
    assert (destination / "result/synthetic-report.docx").is_file()
    assert "Synthetic smoke test only" in completed.stdout


def test_offline_cli_succeeds_when_network_is_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import socket

    from bridge_evidence.cli import main

    def deny_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("The offline demonstration attempted network access")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "synthetic.docx"
    runpy.run_path(str(root / "examples/offline_demo.py"))["create_report"](source)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"cache_dir": str(tmp_path / "cache"), "work_dir": str(tmp_path / "work")}), encoding="utf-8")
    main(["--config", str(config), "single", "--input", str(source),
          "--output-dir", str(tmp_path / "result"), "--llm-mode", "off"])
    assert (tmp_path / "result/synthetic.docx").is_file()
