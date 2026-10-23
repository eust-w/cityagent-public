"""Guard configuration reaches the 27B Agents without changing the 4B model."""
import hashlib
import json
import time

import pytest

from bridge_evidence.contest import document_multiagent_cli as cli
from bridge_evidence.contest import document_multiagent_worker as worker
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import atomic_json


def cli_arguments(tmp_path):
    source = tmp_path / "source.json"
    atomic_json(source, manifest_from_text("合成桥梁报告"))
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(json.dumps({"model": {"type": "BPE", "vocab": {"a": 0}},
                                    "decoder": {"type": "ByteLevel"}, "added_tokens": []}))
    digest = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    args = ["--manifest", str(source), "--manifest-sha256", hashlib.sha256(source.read_bytes()).hexdigest(),
            "--scope", "official-training", "--output-dir", str(tmp_path / "run"), "--execute",
            "--model", "own-27b", "--base-url", "http://127.0.0.1:9", "--no-thinking"]
    return args, tokenizer, digest


def test_cli_forwards_verified_guard_profile_without_changing_sampling(tmp_path, monkeypatch):
    args, tokenizer, digest = cli_arguments(tmp_path)
    captured = []
    monkeypatch.setattr(cli, "run_multiagent", lambda manifest, directory, model, **kwargs:
                        captured.append(model) or {"status": "synthetic", "complete": False, "seconds": 0, "error": None})
    assert cli.main(args + ["--capture-raw-sse", "--raw-guard-tokenizer", str(tokenizer),
                           "--raw-guard-tokenizer-sha256", digest, "--native-repetition-detection"]) == 2
    assert captured == [{"model": "own-27b", "base_url": "http://127.0.0.1:9",
                         "api_key_env": "CITYAGENT_5090_VLLM_API_KEY", "thinking_tokens": 4096,
                         "enable_thinking": False, "capture_raw_sse": True,
                         "raw_guard_tokenizer_path": str(tokenizer.resolve()),
                         "raw_guard_tokenizer_sha256": digest, "native_repetition_detection": True}]


@pytest.mark.parametrize("case", ["no_hash", "hash_only", "wrong_hash", "no_capture", "missing_file", "unsupported_tokenizer"])
def test_invalid_guard_configuration_never_dispatches_agents(tmp_path, monkeypatch, case):
    args, tokenizer, digest = cli_arguments(tmp_path)
    if case == "unsupported_tokenizer":
        tokenizer.write_text('{"decoder":{"type":"WordPiece"},"model":{"type":"BPE"}}')
        digest = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    flags = ["--capture-raw-sse"]
    if case != "hash_only":
        flags += ["--raw-guard-tokenizer", str(tokenizer)]
    if case != "no_hash":
        flags += ["--raw-guard-tokenizer-sha256", "0" * 64 if case == "wrong_hash" else digest]
    if case == "no_capture":
        flags.remove("--capture-raw-sse")
    if case == "missing_file":
        tokenizer.unlink()
    monkeypatch.setattr(cli, "run_multiagent", lambda *a, **kw: pytest.fail("No agent may start on invalid guard configuration"))
    with pytest.raises(SystemExit) as error:
        cli.main(args + flags)
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_shared_main_child_factory_keeps_27b_tokenizer_out_of_4b(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(worker.LocalModelClient, "enable_reliable_generation",
                        lambda self, policy, **kwargs: captured.append((self, policy, kwargs)))
    config = {"model": "own-27b", "base_url": "http://127.0.0.1:9", "enable_thinking": False,
              "capture_raw_sse": True, "raw_guard_tokenizer_path": "/synthetic/27b/tokenizer.json",
              "raw_guard_tokenizer_sha256": "a" * 64, "native_repetition_detection": True,
              "summary": {"model": "own-4b", "base_url": "http://127.0.0.1:10"}}
    for name, summary in (("main", False), ("child", False), ("summary", True)):
        client = worker.make_client(config, time.monotonic() + 1200, summary=summary,
                                    capture_directory=tmp_path / name / "raw-sse")
        _, policy, kwargs = captured[-1]
        assert client.max_model_len == 262144 and client.dynamic_output_budget
        assert client.timeout == 900 and policy.wall_seconds == 900
        assert client.enable_thinking is summary
        assert kwargs["raw_guard_tokenizer_path"] == (None if summary else config["raw_guard_tokenizer_path"])
        assert kwargs["raw_guard_tokenizer_sha256"] == (None if summary else config["raw_guard_tokenizer_sha256"])
        assert kwargs["native_repetition_detection"] is (not summary)
