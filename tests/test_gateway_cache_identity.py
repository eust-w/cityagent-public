from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bridge_evidence.agents.client import GatewayClient
from bridge_evidence.agents.json_cache import JsonCache
from bridge_evidence.config import Settings


@pytest.fixture
def fake_services(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    monkeypatch.setenv("BRIDGE_EVIDENCE_API_KEY", "fake-chat-credential")
    monkeypatch.setenv("BRIDGE_EVIDENCE_SEMANTIC_API_KEY", "fake-semantic-credential")

    def fake_openai(*, base_url: str, **kwargs: object) -> SimpleNamespace:
        version = 2 if "service-b" in base_url else 1

        def chat(**kwargs: object) -> SimpleNamespace:
            calls.append(("chat", base_url))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"service_version": version})))])

        def embeddings(**kwargs: object) -> SimpleNamespace:
            calls.append(("embedding", base_url))
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[float(version)])])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=chat)),
                               embeddings=SimpleNamespace(create=embeddings))

    def rerank(url: str, **kwargs: object) -> SimpleNamespace:
        calls.append(("rerank", url))
        version = 2 if "service-b" in url else 1
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"results": [{"index": 0, "score": version / 10, "document": "example"}]})

    monkeypatch.setattr("bridge_evidence.agents.client.OpenAI", fake_openai)
    monkeypatch.setattr("bridge_evidence.agents.client.instructor.from_openai", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr("bridge_evidence.agents.client.httpx.post", rerank)
    return calls


def _client(directory: Path, *, chat: str = "http://service-a.test/v1", semantic: str = "http://service-a.test/v1") -> GatewayClient:
    return GatewayClient(Settings(cache_dir=directory, base_url=chat, semantic_base_url=semantic,
                                  model="same-model", embedding_model="same-model", reranker_model="same-model"))


def _request(client: GatewayClient, kind: str) -> object:
    if kind == "chat":
        return client.complete_json(system="fixture", user_payload={"question": "same input"}, temperature=0, task="fixture-chat")
    if kind == "embedding":
        return client.embed_texts(["same input"])
    return client.rerank_documents(query="same input", documents=["example"])


@pytest.mark.parametrize("kind", ["chat", "embedding", "rerank"])
def test_service_changes_do_not_reuse_same_model_request_cache(tmp_path: Path, fake_services: list[tuple[str, str]], kind: str) -> None:
    first = _client(tmp_path)
    assert isinstance(first.cache, JsonCache)
    answer_a = _request(first, kind)
    same_service = _client(tmp_path)
    assert _request(same_service, kind) == answer_a
    assert same_service.metrics.cache_hits == 1
    assert len(fake_services) == 1

    options = {"chat" if kind == "chat" else "semantic": "http://service-b.test/v1"}
    second = _client(tmp_path, **options)
    answer_b = _request(second, kind)
    assert answer_b != answer_a
    assert second.metrics.cache_hits == 0
    assert len(fake_services) == 2
    assert _request(_client(tmp_path, **options), kind) == answer_b
    assert len(fake_services) == 2


def test_cache_keys_bind_both_endpoints_without_storing_url_or_credentials(tmp_path: Path, fake_services: list[tuple[str, str]]) -> None:
    import sqlite3

    first = _client(tmp_path)
    _request(first, "chat")
    changed_semantic = _client(tmp_path, semantic="http://service-b.test/v1")
    _request(changed_semantic, "chat")
    assert len(fake_services) == 2
    with sqlite3.connect(first.cache.path) as connection:
        keys = [row[0] for row in connection.execute("SELECT key FROM entries")]
    assert len(set(keys)) == 2
    assert all("service-a.test" not in key and "service-b.test" not in key and "credential" not in key for key in keys)


def test_static_payload_fingerprint_interface_remains_available() -> None:
    assert GatewayClient._fingerprint("same", {"x": 1}) == GatewayClient._fingerprint("same", {"x": 1})
