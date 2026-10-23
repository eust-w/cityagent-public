from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from threading import Lock
from typing import Any, TypeVar

import instructor
import httpx
from openai import OpenAI
from pydantic import BaseModel

from bridge_evidence.config import Settings
from bridge_evidence.agents.json_cache import JsonCache


ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


_QWEN_TASK_MAX_TOKENS = {
    "react-plan": 1024,
    "summary-review": 3072,
    "detail-review": 4096,
    "disease-normalization": 2048,
    "decision": 2048,
}


@dataclass
class ClientMetrics:
    calls: int = 0
    chat_calls: int = 0
    vision_calls: int = 0
    embedding_calls: int = 0
    reranker_calls: int = 0
    planner_calls: int = 0
    cache_hits: int = 0
    task_calls: dict[str, int] = field(default_factory=dict)
    task_cache_hits: dict[str, int] = field(default_factory=dict)
    task_models: dict[str, str] = field(default_factory=dict)
    model_calls: dict[str, int] = field(default_factory=dict)
    model_cache_hits: dict[str, int] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0


class GatewayClient:
    """OpenAI-compatible gateway client with typed Instructor responses and disk cache."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.api_key:
            raise RuntimeError(f"Missing API key in ${settings.api_key_env}")
        cache_dir = settings.cache_dir / "llm"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = JsonCache(cache_dir)
        self.metrics = ClientMetrics()
        self._metrics_lock = Lock()
        self.raw = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )
        if not settings.semantic_api_key:
            raise RuntimeError(f"Missing semantic API key in ${settings.semantic_api_key_env}")
        self.semantic_raw = OpenAI(
            api_key=settings.semantic_api_key,
            base_url=settings.semantic_base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )
        self.typed = instructor.from_openai(self.raw, mode=instructor.Mode.JSON)

    def complete_json(self, *, system: str, user_payload: dict[str, Any], temperature: float, task: str) -> dict[str, Any]:
        messages = self._messages(system, user_payload)
        model = self.settings.model_for_task(task)
        fingerprint = self._cache_fingerprint(
            task,
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            },
        )
        cached = self.cache.get(fingerprint)
        if cached is not None:
            self._record_cache_hit(task, model)
            return cached["parsed"]

        completion = self.raw.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or ""
        parsed = self._parse_json_content(content)
        self._record_completion(completion, kind="chat", task=task, model=model)
        self.cache.set(fingerprint, {"parsed": parsed})
        return parsed

    def complete_model(
        self,
        response_model: type[ResponseModelT],
        *,
        system: str,
        user_payload: dict[str, Any],
        temperature: float,
        task: str,
    ) -> ResponseModelT:
        model = self.settings.model_for_task(task)
        transport_options = self._typed_transport_options(model, task)
        response_schema = response_model.model_json_schema()
        if "kimi" in model.lower():
            user_payload = {
                **user_payload,
                "response_contract": "只输出一个满足 response_json_schema 的 JSON 对象；不得使用 Markdown 解释。",
                "response_json_schema": response_schema,
            }
        messages = self._messages(system, user_payload)
        fingerprint = self._cache_fingerprint(
            task,
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "schema": response_schema,
                "transport_options": transport_options,
            },
        )
        cached = self.cache.get(fingerprint)
        if cached is not None:
            self._record_cache_hit(task, model)
            return response_model.model_validate(cached["parsed"])

        if "kimi" in model.lower():
            # Kimi gateways occasionally wrap valid JSON in a Markdown fence.
            # Parse that transport decoration ourselves, then apply the exact
            # same Pydantic schema before recording or caching the response.
            retry_messages = list(messages)
            last_validation_error: ValueError | None = None
            for attempt in range(self.settings.max_retries):
                completion = self.raw.chat.completions.create(
                    model=model,
                    messages=retry_messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = completion.choices[0].message.content or ""
                self._record_completion(completion, kind="chat", task=task, model=model)
                try:
                    result = response_model.model_validate(self._parse_json_content(content))
                except ValueError as exc:
                    last_validation_error = exc
                    with self._metrics_lock:
                        self.metrics.retries += 1
                    if attempt + 1 >= self.settings.max_retries:
                        raise
                    retry_messages.extend([
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "task": "上一个 JSON 未通过类型校验。只修正 JSON，不解释。",
                                    "validation_error": str(exc)[:1800],
                                    "response_json_schema": response_schema,
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    ])
                    continue
                self.cache.set(fingerprint, {"parsed": result.model_dump(mode="json")})
                return result
            assert last_validation_error is not None
            raise last_validation_error
        else:
            result, completion = self.typed.chat.completions.create_with_completion(
                response_model=response_model,
                model=model,
                messages=messages,
                temperature=temperature,
                max_retries=self.settings.max_retries,
                **transport_options,
            )
        self._record_completion(completion, kind="chat", task=task, model=model)
        self.cache.set(fingerprint, {"parsed": result.model_dump(mode="json")})
        return result

    @staticmethod
    def _typed_transport_options(model: str, task: str) -> dict[str, Any]:
        if "qwen" not in model.lower():
            return {}
        return {
            "max_tokens": _QWEN_TASK_MAX_TOKENS.get(task, 2048),
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }

    def complete_vision_json(
        self,
        *,
        system: str,
        prompt: str,
        image_data_urls: list[str],
        temperature: float,
        task: str,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Call the configured open-weight vision model on bounded images."""

        model = model or self.settings.vision_model
        if not model:
            raise RuntimeError("vision_model 未配置")
        if not image_data_urls:
            raise ValueError("视觉调用至少需要一张图片")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_data_urls)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        fingerprint = self._cache_fingerprint(task, {"model": model, "messages": messages, "temperature": temperature})
        cached = self.cache.get(fingerprint)
        if cached is not None:
            self._record_cache_hit(task, model)
            return cached["parsed"]
        completion = self.raw.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        content_text = completion.choices[0].message.content or ""
        parsed = self._parse_json_content(content_text)
        self._record_completion(completion, kind="vision", task=task, model=model)
        self.cache.set(fingerprint, {"parsed": parsed})
        return parsed

    def embed_texts(self, texts: list[str], *, task: str = "embedding") -> list[list[float]]:
        """Embed text with the competition-provided multimodal embedding model."""
        if not texts:
            return []
        fingerprint = self._cache_fingerprint(task, {"model": self.settings.embedding_model, "texts": texts})
        cached = self.cache.get(fingerprint)
        if cached is not None:
            self._record_cache_hit(task, self.settings.embedding_model)
            return cached["vectors"]
        response = self.semantic_raw.embeddings.create(model=self.settings.embedding_model, input=texts)
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        self._record_completion(response, kind="embedding", task=task, model=self.settings.embedding_model)
        self.cache.set(fingerprint, {"vectors": vectors})
        return vectors

    def rerank_documents(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int = 20,
        task: str = "rerank",
    ) -> list[dict[str, Any]]:
        """Rerank evidence with the competition-provided Qwen VL reranker."""
        if not documents:
            return []
        top_n = min(max(1, top_n), len(documents))
        payload = {
            "model": self.settings.reranker_model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        fingerprint = self._cache_fingerprint(task, payload)
        cached = self.cache.get(fingerprint)
        if cached is not None:
            self._record_cache_hit(task, self.settings.reranker_model)
            return cached["results"]
        response = httpx.post(
            f"{self.settings.semantic_base_url.rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {self.settings.semantic_api_key}"},
            json=payload,
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        raw_results = body if isinstance(body, list) else body.get("results", body.get("data", []))
        results = [
            {
                "index": int(item.get("index", 0)),
                "score": float(item.get("score", item.get("relevance_score", 0.0))),
                "document": str(item.get("document", "")),
            }
            for item in raw_results[:top_n]
        ]
        self._record_request(kind="reranker", task=task, model=self.settings.reranker_model)
        self.cache.set(fingerprint, {"results": results})
        return results

    @staticmethod
    def _messages(system: str, user_payload: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
        ]

    @staticmethod
    def _fingerprint(task: str, payload: dict[str, Any]) -> str:
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return f"{task}:{digest}"

    def _cache_fingerprint(self, task: str, payload: dict[str, Any]) -> str:
        # The same model alias can resolve to different weights on two servers.
        # Bind service addresses to every cached request, including embeddings
        # and reranking, without storing addresses or credentials as cache keys.
        return self._fingerprint(task, {
            "cache_schema": "service-bound-json-v1",
            "services": {
                "chat": self.settings.base_url,
                "semantic": self.settings.semantic_base_url,
            },
            "request": payload,
        })

    def _record_cache_hit(self, task: str, model: str) -> None:
        with self._metrics_lock:
            self.metrics.cache_hits += 1
            self.metrics.task_cache_hits[task] = self.metrics.task_cache_hits.get(task, 0) + 1
            self.metrics.task_models[task] = model
            self.metrics.model_cache_hits[model] = self.metrics.model_cache_hits.get(model, 0) + 1

    def _record_request(self, *, kind: str, task: str, model: str) -> None:
        with self._metrics_lock:
            self.metrics.calls += 1
            self.metrics.task_calls[task] = self.metrics.task_calls.get(task, 0) + 1
            self.metrics.task_models[task] = model
            self.metrics.model_calls[model] = self.metrics.model_calls.get(model, 0) + 1
            if kind == "chat":
                self.metrics.chat_calls += 1
                if task == "react-plan":
                    self.metrics.planner_calls += 1
            elif kind == "vision":
                self.metrics.chat_calls += 1
                self.metrics.vision_calls += 1
            elif kind == "embedding":
                self.metrics.embedding_calls += 1
            elif kind == "reranker":
                self.metrics.reranker_calls += 1
            else:
                raise ValueError(f"unknown client metric kind: {kind}")

    def _record_completion(self, completion: Any, *, kind: str, task: str, model: str) -> None:
        usage = getattr(completion, "usage", None)
        self._record_request(kind=kind, task=task, model=model)
        with self._metrics_lock:
            self.metrics.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.metrics.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("Model output must be a JSON object")
        return value
