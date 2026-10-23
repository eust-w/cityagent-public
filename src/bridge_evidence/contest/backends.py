"""Local vendor engine and loopback OpenAI-compatible engines; no external API."""

from __future__ import annotations

import base64
import binascii
import contextvars
import hashlib
import io
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from .run_state import MODEL_IDENTITY_SUFFIXES, file_digest, file_digests, json_digest


@dataclass
class ModelManifest:
    base: str
    tokenizer: str
    adapter: str | None = None
    merged: str | None = None
    served_model: str | None = None
    schema_version: int = 1
    adapter_sha256: str | None = None
    eligibility_evidence: str | None = None
    # Omission retains the original fine-tuned-only manifest contract. A new
    # unadapted baseline must be declared explicitly, never disguised as merged.
    weight_kind: str | None = None

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError("model manifest must be object")
        return cls(**data)

    def validate(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unknown model manifest version")
        if self.weight_kind is not None and self.weight_kind != "base":
            raise ValueError("unknown model manifest weight_kind")
        if self.weight_kind == "base" and any(
            value is not None
            for value in (self.adapter, self.merged, self.adapter_sha256)
        ):
            raise ValueError("base weight_kind cannot declare fine-tuned adapter or merged weights")
        for label in ("base", "tokenizer", "adapter", "merged", "served_model",
                      "adapter_sha256", "eligibility_evidence"):
            value = getattr(self, label)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{label} must be a nonempty string")
        for label, path in (("base", self.base), ("tokenizer", self.tokenizer)):
            if not path or not Path(path).is_dir():
                raise ValueError(f"{label} directory missing")
        if not (Path(self.base) / "config.json").is_file():
            raise ValueError("base model config missing")
        if self.weight_kind == "base":
            for label, config_path in (
                ("base", Path(self.base) / "config.json"),
                ("tokenizer", Path(self.tokenizer) / "tokenizer_config.json"),
            ):
                if not config_path.is_file():
                    raise ValueError(f"{label} config missing")
                if not isinstance(json.loads(config_path.read_text()), dict):
                    raise ValueError(f"{label} config must be an object")
            tokenizer_root = Path(self.tokenizer)
            if not (
                any((tokenizer_root / name).is_file() for name in (
                    "tokenizer.json", "tokenizer.model", "spiece.model", "vocab.txt",
                ))
                or all((tokenizer_root / name).is_file() for name in ("vocab.json", "merges.txt"))
            ):
                raise ValueError("tokenizer artifact missing")
            self._validate_base_weights()
        if self.adapter:
            root = Path(self.adapter)
            config = json.loads((root / "adapter_config.json").read_text())
            expected = config.get("base_model_name_or_path")
            if expected and Path(expected).name != Path(self.base).name:
                raise ValueError(
                    "adapter base model mismatch; retrain for a different base"
                )
            digest = file_digest(root / "adapter_model.safetensors")
            if self.adapter_sha256 and digest != self.adapter_sha256:
                raise ValueError("adapter SHA256 mismatch")
        if self.merged and (
            not (Path(self.merged) / "config.json").is_file()
            or (Path(self.merged) / ".merge_incomplete").exists()
        ):
            raise ValueError("merged weights not explicitly prepared")
        if self.weight_kind != "base" and not self.adapter and not self.merged:
            raise ValueError("fine-tuned adapter or explicit merged model required")
        return self

    def _validate_base_weights(self):
        """Check a complete, root-bound HF weight inventory, not loadability.

        Tensor decoding stays with the actual inference service; this check
        neither loads pickle checkpoints nor claims that nonempty bytes are
        valid model tensors. ``identity`` separately hashes those exact bytes.
        """
        root = Path(self.base).resolve(strict=True)

        def artifact(name, suffix):
            if (not isinstance(name, str) or not name or "\\" in name
                    or Path(name).is_absolute() or ".." in Path(name).parts
                    or Path(name).suffix != suffix):
                raise ValueError("invalid base weight artifact path")
            try:
                path = (root / name).resolve(strict=True)
                path.relative_to(root)
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError("base weight artifact must be a nonempty file")
            except (OSError, RuntimeError, ValueError) as error:
                raise ValueError("base weight artifact missing, empty or outside model root") from error
            return path

        indexed = False
        for name, suffix in (
            ("model.safetensors.index.json", ".safetensors"),
            ("pytorch_model.bin.index.json", ".bin"),
        ):
            index = root / name
            if not index.exists() and not index.is_symlink():
                continue
            indexed = True
            try:
                value = json.loads(artifact(name, ".json").read_text())
            except (UnicodeError, ValueError) as error:
                raise ValueError("invalid base weight index") from error
            mapping = value.get("weight_map") if isinstance(value, dict) else None
            if (not isinstance(mapping, dict) or not mapping
                    or any(not isinstance(key, str) or not key for key in mapping)):
                raise ValueError("base weight index requires a nonempty weight_map")
            validated_shards = set()
            for shard in mapping.values():
                if not isinstance(shard, str) or shard not in validated_shards:
                    artifact(shard, suffix)
                    validated_shards.add(shard)
        standalone = [name for name in ("model.safetensors", "pytorch_model.bin")
                      if (root / name).exists() or (root / name).is_symlink()]
        if not indexed and not standalone:
            raise ValueError("base weights missing; provide standalone weights or a complete shard index")
        for name in standalone:
            artifact(name, Path(name).suffix)

    def as_dict(self):
        """Canonical identity fields; legacy manifests retain their exact shape."""
        value = asdict(self)
        if self.weight_kind is None:
            value.pop("weight_kind")
        return value

    def identity(self):
        self.validate()
        # Actual model bytes, including adapter and tokenizer, bind every invocation.
        files = {}
        roots = {}
        for label, folder in (
            ("base", self.base),
            ("adapter", self.adapter),
            ("tokenizer", self.tokenizer),
            ("merged", self.merged),
        ):
            if folder:
                resolved = str(Path(folder).resolve())
                if resolved in roots:
                    files[label] = roots[resolved]
                    continue
                paths = [
                    p for p in sorted(Path(folder).rglob("*"))
                    if p.is_file() and p.suffix in MODEL_IDENTITY_SUFFIXES
                ]
                files[label] = dict(zip(
                    (p.relative_to(folder).as_posix() for p in paths),
                    file_digests(paths, file_digest),
                ))
                roots[resolved] = files[label]
        return {"manifest": self.as_dict(), "files": files}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("inference endpoint redirects are not allowed")


def validate_visual_request(value):
    """Only bounded in-memory raster pixels may cross the visual API boundary."""
    if (not isinstance(value, dict)
            or set(value) != {"kind", "job_id", "image", "instruction"}
            or value["kind"] != "visual-evidence-v1"
            or not isinstance(value["job_id"], str)
            or len(value["job_id"]) != 64
            or any(c not in "0123456789abcdef" for c in value["job_id"])
            or not isinstance(value["instruction"], str)
            or not 0 < len(value["instruction"].encode()) <= 64 * 1024):
        raise ValueError("invalid bounded visual request")
    image = value["image"]
    if (not isinstance(image, dict)
            or set(image) != {"mime_type", "sha256", "base64"}
            or image["mime_type"] not in {"image/png", "image/jpeg"}
            or not isinstance(image["base64"], str)
            or not 0 < len(image["base64"]) <= 24 * 1024 * 1024):
        raise ValueError("invalid embedded raster image")
    try:
        blob = base64.b64decode(image["base64"], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("invalid image base64") from error
    if (not 0 < len(blob) <= 16 * 1024 * 1024
            or hashlib.sha256(blob).hexdigest() != image["sha256"]):
        raise ValueError("image content hash or size mismatch")
    from PIL import Image

    try:
        with Image.open(io.BytesIO(blob)) as raster:
            if (raster.format != {"image/png": "PNG", "image/jpeg": "JPEG"}[image["mime_type"]]
                    or not 0 < raster.width * raster.height <= 16_000_000
                    or getattr(raster, "n_frames", 1) != 1):
                raise ValueError("unsupported image format, size or animation")
            raster.verify()
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError("image decoding failed") from error
    return value


def normalize_api_url(url):
    """Validate a loopback service address without connecting or loading a model."""
    if not isinstance(url, str) or not url:
        raise ValueError("explicit local loopback inference endpoint required")
    try:
        parsed = urlsplit(url)
        port = parsed.port
        allowed = (
            parsed.scheme == "http"
            and parsed.hostname in ("127.0.0.1", "localhost", "::1")
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
            and (port is None or port > 0)
        )
    except ValueError:
        allowed = False
    if not allowed:
        raise ValueError("only local loopback inference endpoints are allowed")
    return url.rstrip("/")


class APIBackend:
    def __init__(
        self,
        url,
        model,
        tokenizer,
        *,
        kind="vllm-api",
        timeout=7200,
        workers=1,
        seed=17,
    ):
        url = normalize_api_url(url)
        if (
            kind not in ("vllm-api", "sglang-api")
            or workers not in (1, 2, 4, 8)
            or timeout <= 0
        ):
            raise ValueError("invalid backend configuration")
        if type(seed) is not int or seed < 0:
            raise ValueError("invalid seed")
        self.seed = seed
        self.url, self.model, self.tokenizer, self.kind, self.timeout, self.workers = (
            url.rstrip("/"),
            model,
            tokenizer,
            kind,
            timeout,
            workers,
        )

    def _request(self, route, value=None):
        data = json.dumps(value).encode() if value is not None else None
        headers = {"Content-Type": "application/json"}
        if os.environ.get("CITYAGENT_LOCAL_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["CITYAGENT_LOCAL_API_KEY"]
        request = urllib.request.Request(
            self.url + "/" + route, data=data, headers=headers
        )
        from .rsi_runtime import REPORT_DEADLINE

        deadline = REPORT_DEADLINE.get()
        timeout = self.timeout
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("report deadline exhausted before backend request")
            timeout = min(timeout, remaining)
        with urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect()).open(
            request, timeout=timeout
        ) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("inference response exceeds receipt limit")
            return json.loads(raw)

    def probe(self):
        models = self._request("models")
        if self.model not in [m["id"] for m in models.get("data", [])]:
            raise ValueError("configured base/adapter alias is not served")
        return {
            "kind": self.kind,
            "models": models,
            "optimizations": "server-owned; require deployment evidence",
        }

    def __call__(self, ids, maximum):
        if isinstance(ids, dict):
            return self.visual(ids, maximum)
        # The completions endpoint accepts exact token IDs; no second chat template.
        result = self._request(
            "completions",
            {
                "model": self.model,
                "prompt": ids,
                "max_tokens": maximum,
                "temperature": 0.0,
                "seed": self.seed,
                "stream": False,
            },
        )
        choice = result["choices"][0]
        return {
            "text": choice["text"],
            "finish_reason": choice["finish_reason"],
            "output_tokens": result["usage"]["completion_tokens"],
            "backend_request_id": result.get("id"),
        }

    def visual(self, request, maximum):
        request = validate_visual_request(request)
        if type(maximum) is not int or not 1 <= maximum <= 4096:
            raise ValueError("visual output budget must be 1..4096 tokens")
        result = self._request("chat/completions", {
            "model": self.model, "max_tokens": maximum, "temperature": 0.0,
            "seed": self.seed, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": request["instruction"]},
                {"type": "image_url", "image_url": {"url": (
                    "data:" + request["image"]["mime_type"] + ";base64,"
                    + request["image"]["base64"])}},
            ]}],
        })
        try:
            usage, choices = result["usage"], result["choices"]
            if (result["model"] != self.model
                    or not isinstance(result.get("id"), str) or not result["id"]
                    or len(choices) != 1 or choices[0]["index"] != 0
                    or choices[0]["message"]["role"] != "assistant"
                    or not isinstance(choices[0]["message"]["content"], str)
                    or any(type(usage[k]) is not int or usage[k] <= 0
                           for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
                    or usage["completion_tokens"] > maximum):
                raise ValueError("invalid visual model identity, content or usage")
            return {"text": choices[0]["message"]["content"],
                    "finish_reason": choices[0]["finish_reason"],
                    "prompt_tokens": usage["prompt_tokens"],
                    "output_tokens": usage["completion_tokens"],
                    "backend_request_id": result["id"]}
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("invalid visual backend response") from error

    def batch(self, requests):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda r: self(*r), requests))


class LocalVLLMBackend:
    def __init__(self, manifest, *, max_model_len=65536, parameters=None, seed=17):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        manifest.validate()
        self.params = SamplingParams
        self.seed = seed
        options = {
            "dtype": "bfloat16",
            "max_model_len": max_model_len,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": True,
            "trust_remote_code": True,
            **(parameters or {}),
        }
        self.llm = LLM(
            model=manifest.merged or manifest.base,
            enable_lora=bool(manifest.adapter and not manifest.merged),
            max_lora_rank=16,
            **options,
        )
        # Vendor request IDs may restart at zero when an actual engine is
        # reloaded. Scope their original value to this live engine instance.
        self.backend_instance_id = uuid4().hex
        self.lora = (
            LoRARequest("cityagent", 1, manifest.adapter)
            if manifest.adapter and not manifest.merged
            else None
        )

    def batch(self, requests):
        outputs = self.llm.generate(
            [{"prompt_token_ids": ids} for ids, _ in requests],
            [
                self.params(temperature=0, max_tokens=maximum, seed=self.seed)
                for _, maximum in requests
            ],
            lora_request=self.lora,
        )
        return [
            {
                "text": o.outputs[0].text,
                "finish_reason": o.outputs[0].finish_reason,
                "output_tokens": len(o.outputs[0].token_ids),
                "backend_request_id": o.request_id,
                "backend_instance_id": self.backend_instance_id,
            }
            for o in outputs
        ]

    def __call__(self, ids, maximum):
        return self.batch([(ids, maximum)])[0]


class AccountedBackend:
    def __init__(self, backend, ledger, role="extraction"):
        self.backend, self.ledger, self.role = backend, ledger, role

    def __call__(self, ids, maximum):
        role = ("visual-evidence" if isinstance(ids, dict)
                and ids.get("kind") == "visual-evidence-v1" else self.role)
        return self.call_with_role(ids, maximum, role=role)

    @staticmethod
    def _response_receipt(response, prompt):
        if not isinstance(response, dict) or not isinstance(response.get("text"), str):
            raise ValueError("backend response must contain text")
        receipt = {
            "prompt_sha256": json_digest(prompt),
            "backend_request_id": response.get("backend_request_id"),
            "output_sha256": hashlib.sha256(response["text"].encode()).hexdigest(),
            "output_tokens": response.get("output_tokens"),
            "finish_reason": response.get("finish_reason"),
        }
        if response.get("backend_instance_id") is not None:
            receipt["backend_instance_id"] = response["backend_instance_id"]
        return receipt

    @staticmethod
    def _annotate_error(error, call_id, seconds):
        try:
            error.call_id = call_id
            error.latency_seconds = seconds
        except AttributeError:
            pass

    def _finish_failure(self, call_id, prompt, seconds, error, **extra):
        # A later sibling failure (or post-commit error) must never overwrite
        # an already finalized receipt. Only this caller's reserved ID is read.
        with self.ledger.connect() as db:
            row = db.execute("SELECT seconds FROM calls WHERE id=?", (call_id,)).fetchone()
        if row and row[0] is not None:
            return
        self.ledger.finish(
            call_id, seconds,
            status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            receipt={"prompt_sha256": json_digest(prompt),
                     "error_type": type(error).__name__, "outcome": "unknown", **extra},
        )

    def call_with_role(self, ids, maximum, *, role):
        from .rsi_runtime import REQUEST_CALL_ID

        call_id = REQUEST_CALL_ID.get() or uuid4().hex
        self.ledger.claim(call_id, role=role)
        start = time.perf_counter()
        try:
            response = self.backend(ids, maximum)
            receipt = self._response_receipt(response, ids)
            response["call_id"] = call_id
            response["latency_seconds"] = time.perf_counter() - start
            self.ledger.finish(
                call_id,
                response["latency_seconds"],
                status="success"
                if response.get("finish_reason") == "stop"
                else "failed",
                receipt=receipt,
            )
            return response
        except BaseException as error:
            elapsed = time.perf_counter() - start
            self._finish_failure(call_id, ids, elapsed, error)
            self._annotate_error(error, call_id, elapsed)
            raise

    def batch(self, requests):
        # Keep individual IDs and receipts. API requests overlap; local vLLM batches are tracked together.
        from .rsi_runtime import REQUEST_CALL_ID, REQUEST_CALL_IDS

        reserved = REQUEST_CALL_IDS.get()
        if reserved is not None and (len(reserved) != len(requests)
                                     or len(set(reserved)) != len(reserved)):
            raise ValueError("batch reservation identities do not match requests")
        ids = list(reserved) if reserved is not None else [uuid4().hex for _ in requests]
        if isinstance(self.backend, APIBackend):

            def execute(item):
                call_id, request, context = item
                def generate():
                    token = REQUEST_CALL_ID.set(call_id)
                    try:
                        return self(*request)
                    except Exception as error:
                        return error
                    finally:
                        REQUEST_CALL_ID.reset(token)
                return context.run(generate)

            with ThreadPoolExecutor(max_workers=self.backend.workers) as pool:
                return list(pool.map(execute, [(call_id, request, contextvars.copy_context())
                                               for call_id, request in zip(ids, requests)]))
        if not ids:
            return []
        self.ledger.claim_many(ids, role=self.role)
        start = time.perf_counter()
        try:
            responses = self.backend.batch(requests)
            elapsed = time.perf_counter() - start
            if len(responses) != len(ids):
                raise ValueError("backend batch receipt count mismatch")
            outcomes = []
            for call_id, response, request in zip(ids, responses, requests):
                try:
                    if isinstance(response, BaseException):
                        raise response
                    receipt = self._response_receipt(response, request[0])
                except BaseException as error:
                    self._finish_failure(call_id, request[0], elapsed / len(ids), error,
                                         batch_id=ids[0], batch_wall_seconds=elapsed)
                    self._annotate_error(error, call_id, elapsed / len(ids))
                    outcomes.append(error)
                    continue
                response["call_id"] = call_id
                response["latency_seconds"] = elapsed / len(ids)
                self.ledger.finish(
                    call_id,
                    elapsed / len(ids),
                    status="success"
                    if response.get("finish_reason") == "stop"
                    else "failed",
                    receipt={**receipt, "batch_id": ids[0], "batch_wall_seconds": elapsed},
                )
                outcomes.append(response)
            return outcomes
        except BaseException as error:
            elapsed = time.perf_counter() - start
            settlement_errors = []
            for call_id, request in zip(ids, requests):
                try:
                    self._finish_failure(call_id, request[0], elapsed / len(ids), error,
                                         batch_id=ids[0], batch_wall_seconds=elapsed)
                except Exception as settlement_error:
                    # Try all owned reservations even if one journal write fails.
                    settlement_errors.append(settlement_error)
            if settlement_errors:
                raise RuntimeError("batch call settlement failed; durable recovery required") from settlement_errors[0]
            raise


def dependency_fingerprint():
    import importlib

    result = {}
    for name, module in (
        ("torch", "torch"),
        ("vllm", "vllm"),
        ("tongagents", "tongagents"),
        ("python-docx", "docx"),
    ):
        try:
            imported = importlib.import_module(module)
            result[name] = {
                "version": importlib.metadata.version(name),
                "imported": True,
            }
            if name == "torch":
                result[name]["devices"] = [
                    imported.cuda.get_device_properties(i).total_memory
                    for i in range(imported.cuda.device_count())
                ]
        except Exception as error:
            result[name] = {"imported": False, "error_type": type(error).__name__}
    return result
