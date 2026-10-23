"""Exact native CPU processor admission, without HTTP or model generation.

Only the frozen helper is launched. Its complete input/output artifacts remain
on disk for both success and failure. The caller still owns generation, usage
equivalence, shared report budgets and actual image-quality acceptance.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time
from uuid import uuid4

VERSION = "native-multimodal-admission-v3"
HELPER_SHA256 = "e4e036fe2500d24af149871159a819f872dd093cdab778f03feee9c8c65727e1"
HELPER_VERSION = "cityagent-own-vision-processor-admission-v3"
NATIVE_METHOD = "installed_vllm_renderer_and_InputPreprocessor.preprocess_to_mm_processor.apply"
MODEL = "cityagent-qwen36-27b-epoch-002"
BASE_MODEL = Path("/model/Qwen3.6-27B")
_METADATA = ("config.json", "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json",
             "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "added_tokens.json", "special_tokens_map.json", "vocab.json", "merges.txt")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_LIMIT = 64 * 1024 * 1024


def _packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _digest(value):
    return _sha(_packed(value))


def _valid_sha(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


class MultimodalAdmissionError(ValueError):
    def __init__(self, code, attempt_directory=None):
        self.code = code
        self.attempt_directory = str(attempt_directory) if attempt_directory is not None else None
        super().__init__(code)


def _fail(code):
    raise MultimodalAdmissionError(code)


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _fail("MULTIMODAL_DEADLINE_EXCEEDED")
    return remaining


def _read(path, limit=_LIMIT):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        _fail("MULTIMODAL_ARTIFACT_INVALID")
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        _fail("MULTIMODAL_ARTIFACT_INVALID")
    return value


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail("MULTIMODAL_DUPLICATE_JSON_KEY")
        value[key] = item
    return value


def _load(path):
    try:
        return json.loads(_read(path), object_pairs_hook=_unique,
                          parse_constant=lambda _: _fail("MULTIMODAL_NONFINITE_JSON"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, MultimodalAdmissionError):
            raise
        _fail("MULTIMODAL_JSON_INVALID")


def _save(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("xb") as stream:
        stream.write(_packed(value) + b"\n")
    pending.replace(path)


def _metadata(base_model=None):
    rows = []
    for name in _METADATA:
        path = (BASE_MODEL if base_model is None else base_model) / name
        if path.is_symlink():
            _fail("MULTIMODAL_METADATA_SYMLINK")
        exists = path.is_file()
        rows.append({"path": str(path), "exists": exists, "sha256": _sha(_read(path)) if exists else None})
    return rows


def _binding(value):
    keys = {"schema_version", "catalog_binding_source", "base_model_root", "base_model_id", "model",
            "adapter_path", "adapter_weights_sha256", "adapter_config_sha256"}
    if (not isinstance(value, dict) or set(value) != keys or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1 or value.get("catalog_binding_source") != "model_catalog.id+root+parent"
            or value.get("model") != MODEL
            or not isinstance(value.get("base_model_id"), str) or not value["base_model_id"]
            or "\x00" in value["base_model_id"] or value["base_model_id"].startswith("--")
            or any(not _valid_sha(value.get(k)) for k in ("adapter_weights_sha256", "adapter_config_sha256"))):
        _fail("MULTIMODAL_MODEL_BINDING_INVALID")
    for key in ("base_model_root", "adapter_path"):
        name = value.get(key)
        if (not isinstance(name, str) or "\x00" in name or not Path(name).is_absolute()
                or str(Path(name)) != name or ".." in Path(name).parts
                or Path(name).is_symlink() or not Path(name).is_dir()):
            _fail("MULTIMODAL_MODEL_BINDING_INVALID")
    return deepcopy(value)


def _lora_mapping(argv, binding):
    if argv.count("--enable-lora") != 1 or argv.count("--lora-modules") != 1:
        _fail("MULTIMODAL_SERVICE_LORA_BINDING_INVALID")
    modules = []
    for item in argv[argv.index("--lora-modules") + 1:]:
        if item.startswith("--"):
            break
        try:
            if item.startswith("{"):
                module = json.loads(item, object_pairs_hook=_unique)
                if (not isinstance(module, dict) or not {"name", "path"} <= set(module)
                        or not set(module) <= {"name", "path", "base_model_name"}):
                    raise ValueError("Invalid LoRA module")
            else:
                name, path = item.split("=", 1)
                module = {"name": name, "path": path}
            if any(not isinstance(module.get(k), str) or not module[k] for k in ("name", "path")):
                raise ValueError("Invalid LoRA module")
            modules.append(module)
        except (ValueError, TypeError):
            _fail("MULTIMODAL_SERVICE_LORA_BINDING_INVALID")
    own = [module for module in modules if module["name"] == binding["model"]]
    if (len(own) != 1 or own[0]["path"] != binding["adapter_path"]
            or ("base_model_name" in own[0] and own[0]["base_model_name"] != binding["base_model_id"])):
        _fail("MULTIMODAL_SERVICE_LORA_BINDING_INVALID")


def _adapter_hash(path, deadline=None):
    # Large adapter weights are hashed only by the parent, never the CPU helper.
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        _fail("MULTIMODAL_ADAPTER_IDENTITY_CHANGED")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
            if deadline is not None:
                _remaining(deadline)
    return h.hexdigest()


def _server_policy(argv, base_model):
    # Only explicit portable binding uses this parser. Preserve the actual argv;
    # never translate a module entry into a fictitious `serve` command.
    protected = ("--model", "--served-model-name", "--max-model-len", "--dtype", "--enable-lora", "--lora-modules")
    if any(item.startswith(tuple(flag + "=" for flag in protected)) for item in argv):
        _fail("MULTIMODAL_SERVER_POLICY_INVALID")
    if len(argv) >= 3 and argv[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]:
        if not Path(argv[0]).is_absolute() or "serve" in argv or argv.count("-m") != 1:
            _fail("MULTIMODAL_SERVER_POLICY_INVALID")
        entrypoint, model_flag = "vllm_python_module_api_server", "--model"
    else:
        if argv.count("serve") != 1 or "-m" in argv or "--model" in argv:
            _fail("MULTIMODAL_SERVER_POLICY_INVALID")
        index = argv.index("serve")
        if index not in (1, 2) or Path(argv[index - 1]).name != "vllm":
            _fail("MULTIMODAL_SERVER_POLICY_INVALID")
        entrypoint, model_flag = "vllm_serve", "serve"
    if (argv.count(model_flag) != 1 or argv.index(model_flag) + 1 >= len(argv)
            or argv[argv.index(model_flag) + 1] != str(base_model)):
        _fail("MULTIMODAL_SERVER_POLICY_INVALID")
    trust_flags = [item for item in argv if item.startswith(("--trust-remote-code", "--no-trust-remote-code"))]
    if trust_flags not in ([], ["--trust-remote-code"]):
        _fail("MULTIMODAL_SERVER_POLICY_INVALID")
    if trust_flags:
        index = argv.index("--trust-remote-code")
        if index + 1 < len(argv) and argv[index + 1].lower() in {"true", "false", "0", "1"}:
            _fail("MULTIMODAL_SERVER_POLICY_INVALID")
    return {"schema_version": 1, "entrypoint": entrypoint, "trust_remote_code": bool(trust_flags)}


def _request(payload):
    required = {"model", "messages", "max_tokens", "temperature", "stream", "n", "chat_template_kwargs"}
    if (not isinstance(payload, dict) or set(payload) != required or payload.get("model") != MODEL
            or type(payload.get("max_tokens")) is not int or payload["max_tokens"] != 512
            or type(payload.get("n")) is not int or payload["n"] != 1
            or type(payload.get("temperature")) not in (int, float) or payload["temperature"] != 0
            or payload.get("stream") is not False or payload.get("chat_template_kwargs") != {"enable_thinking": False}
            or payload["chat_template_kwargs"].get("enable_thinking") is not False):
        _fail("MULTIMODAL_REQUEST_UNSUPPORTED")
    messages = payload["messages"]
    if (not isinstance(messages, list) or len(messages) != 1 or not isinstance(messages[0], dict)
            or set(messages[0]) != {"role", "content"} or messages[0]["role"] != "user"):
        _fail("MULTIMODAL_REQUEST_UNSUPPORTED")
    content = messages[0]["content"]
    if (not isinstance(content, list) or len(content) != 2 or any(not isinstance(x, dict) for x in content)
            or set(content[0]) != {"type", "text"} or content[0]["type"] != "text"
            or not isinstance(content[0]["text"], str) or not content[0]["text"].strip()
            or set(content[1]) != {"type", "image_url"} or content[1]["type"] != "image_url"
            or not isinstance(content[1]["image_url"], dict) or set(content[1]["image_url"]) != {"url"}):
        _fail("MULTIMODAL_REQUEST_UNSUPPORTED")
    url = content[1]["image_url"]["url"]
    if not isinstance(url, str) or not url.startswith("data:image/png;base64,"):
        _fail("MULTIMODAL_IMAGE_UNSUPPORTED")
    try:
        image = base64.b64decode(url.split(",", 1)[1], validate=True)
    except ValueError:
        _fail("MULTIMODAL_IMAGE_INVALID")
    if len(image) < 24 or image[:8] != b"\x89PNG\r\n\x1a\n" or image[12:16] != b"IHDR":
        _fail("MULTIMODAL_IMAGE_INVALID")
    width, height = struct.unpack(">II", image[16:24])
    if width <= 0 or height <= 0:
        _fail("MULTIMODAL_IMAGE_INVALID")
    return image, {"blob_sha256": _sha(image), "blob_bytes": len(image), "width": width, "height": height,
                   "mime_type": "image/png", "raw_bytes_unchanged": True,
                   "scope": "exact_payload_png_not_a_claim_about_original_docx_bytes"}


class NativeMultimodalAdmission:
    def __init__(self, directory, *, processor_python, path_audit, runtime_environment, helper_path=None,
                 model_binding=None, modelinfo_cache=None):
        self.directory = Path(directory).absolute()
        if self.directory.is_symlink():
            _fail("MULTIMODAL_DIRECTORY_INVALID")
        self.directory = self.directory.resolve()
        helper_path = helper_path or Path(__file__).with_name("document_native_processor.py")
        self.helper_path, self.path_audit = Path(helper_path).absolute(), Path(path_audit).absolute()
        self.processor_python = Path(processor_python).absolute()
        self._python_resolved = self.processor_python.resolve(strict=True)
        self._python_sha = _sha(_read(self._python_resolved))
        self._helper_sha = _sha(_read(self.helper_path))
        if self._helper_sha != HELPER_SHA256:
            _fail("MULTIMODAL_HELPER_IDENTITY_CHANGED")
        self._audit_bytes = _read(self.path_audit)
        self._audit = _load(self.path_audit)
        self._audit_sha = _sha(self._audit_bytes)
        self._binding = _binding(model_binding) if model_binding is not None else None
        self._base_model = Path(self._binding["base_model_root"]) if self._binding else BASE_MODEL
        self._server_policy = None
        self._validate_audit()
        self._cache_binding, self._cache_path, self._cache_bytes = None, None, None
        if modelinfo_cache is not None:
            from .document_native_processor import modelinfo_cache_binding
            if (self._binding is None or not isinstance(modelinfo_cache, dict)
                    or set(modelinfo_cache) != {"path", "sha256"}
                    or not isinstance(modelinfo_cache["path"], str)):
                _fail("MULTIMODAL_MODELINFO_CACHE_INVALID")
            self._cache_path = Path(modelinfo_cache["path"])
            try:
                self._cache_binding = modelinfo_cache_binding(self._cache_path, modelinfo_cache["sha256"],
                                                             self._audit, self._base_model)
            except (ValueError, KeyError, TypeError, OSError):
                _fail("MULTIMODAL_MODELINFO_CACHE_INVALID")
            self._cache_bytes = _read(self._cache_path)
        if (not isinstance(runtime_environment, dict) or set(runtime_environment) != {"PATH", "LD_LIBRARY_PATH"}
                or any(not isinstance(v, str) or "\x00" in v for v in runtime_environment.values())
                or not runtime_environment["PATH"]):
            _fail("MULTIMODAL_EXPLICIT_ENV_REQUIRED")
        self._environment = deepcopy(runtime_environment)
        self._metadata = _metadata(self._base_model)
        self._verify_adapter(None)
        processor_identity = {"version": VERSION, "audit_sha256": self._audit_sha,
            "helper_sha256": self._helper_sha, "python_path": str(self._python_resolved),
            "python_sha256": self._python_sha, "runtime_environment": self._environment,
            "model_metadata_files": self._metadata}
        if self._binding is not None:
            processor_identity["model_binding"] = self._binding
            processor_identity["server_policy"] = self._server_policy
        if self._cache_binding is not None:
            processor_identity["modelinfo_cache"] = self._cache_binding
        self._identity = {"model": MODEL, "max_model_len": 65536,
            "processor_identity_sha256": _digest(processor_identity),
            "service_identity_sha256": _digest(self._audit["service_process"])}
        if self._binding is not None:
            self._identity["model_binding"] = deepcopy(self._binding)
            self._identity["server_policy"] = deepcopy(self._server_policy)
        if self._cache_binding is not None:
            self._identity["modelinfo_cache"] = deepcopy(self._cache_binding)

    @property
    def identity(self):
        return deepcopy(self._identity)

    def verify(self):
        """Read-only artifact verification; does not run the native processor."""
        self._verify_fixed(None)

    def _validate_audit(self):
        audit = self._audit
        if not isinstance(audit, dict):
            _fail("MULTIMODAL_AUDIT_INVALID")
        service, packages, files = (audit.get(k) for k in ("service_process", "packages", "source_index"))
        if (not isinstance(service, dict) or type(service.get("pid")) is not int or service["pid"] <= 0
                or not _valid_sha(service.get("cmdline_sha256"))
                or not isinstance(service.get("cmdline"), list) or not service["cmdline"]
                or any(not isinstance(x, str) or "\x00" in x for x in service["cmdline"])
                or not isinstance(packages, dict) or not {"vllm", "transformers", "pillow"} <= set(packages)
                or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in packages.items())
                or not isinstance(files, list) or not files):
            _fail("MULTIMODAL_AUDIT_INVALID")
        argv = [x for x in service["cmdline"] if x]
        if _sha(b"\0".join(x.encode() for x in argv) + b"\0") != service["cmdline_sha256"]:
            _fail("MULTIMODAL_SERVICE_IDENTITY_INVALID")
        base_alias = self._binding["base_model_id"] if self._binding else MODEL
        if self._binding is not None:
            self._server_policy = _server_policy(argv, self._base_model)
            model_flag = "--model" if self._server_policy["entrypoint"] == "vllm_python_module_api_server" else "serve"
        else:
            model_flag = "serve"
        for flag, value in ((model_flag, str(self._base_model)), ("--served-model-name", base_alias),
                            ("--max-model-len", "65536"), ("--dtype", "bfloat16")):
            if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv) or argv[argv.index(flag) + 1] != value:
                _fail("MULTIMODAL_SERVICE_IDENTITY_INVALID")
        if self._binding is None and "--trust-remote-code" not in argv:
            _fail("MULTIMODAL_SERVICE_IDENTITY_INVALID")
        if self._binding is not None:
            _lora_mapping(argv, self._binding)
        paths = []
        for row in files:
            if (not isinstance(row, dict) or not isinstance(row.get("path"), str)
                    or not Path(row["path"]).is_absolute() or Path(row["path"]).suffix not in {".py", ".json", ".jinja", ".txt"}
                    or not _valid_sha(row.get("sha256")) or type(row.get("bytes")) is not int or row["bytes"] < 0):
                _fail("MULTIMODAL_AUDIT_SOURCE_INVALID")
            paths.append(row["path"])
        if len(set(paths)) != len(paths):
            _fail("MULTIMODAL_AUDIT_SOURCE_INVALID")

    def _verify_adapter(self, deadline):
        if self._binding is not None:
            for filename, key in (("adapter_model.safetensors", "adapter_weights_sha256"),
                                  ("adapter_config.json", "adapter_config_sha256")):
                if _adapter_hash(Path(self._binding["adapter_path"]) / filename, deadline) != self._binding[key]:
                    _fail("MULTIMODAL_ADAPTER_IDENTITY_CHANGED")

    def _verify_fixed(self, deadline):
        if deadline is not None:
            _remaining(deadline)
        if (_sha(_read(self.helper_path)) != self._helper_sha or _sha(_read(self.path_audit)) != self._audit_sha
                or self.processor_python.resolve(strict=True) != self._python_resolved
                or _sha(_read(self._python_resolved)) != self._python_sha
                or _metadata(self._base_model) != self._metadata):
            _fail("MULTIMODAL_FIXED_IDENTITY_CHANGED")
        self._verify_adapter(deadline)
        if self._cache_binding is not None:
            from .document_native_processor import modelinfo_cache_binding
            try:
                current = modelinfo_cache_binding(self._cache_path, self._cache_binding["sha256"],
                                                 self._audit, self._base_model)
            except (ValueError, KeyError, TypeError, OSError):
                _fail("MULTIMODAL_MODELINFO_CACHE_CHANGED")
            if current != self._cache_binding or _read(self._cache_path) != self._cache_bytes:
                _fail("MULTIMODAL_MODELINFO_CACHE_CHANGED")
        for row in self._audit["source_index"]:
            data = _read(row["path"])
            if len(data) != row["bytes"] or _sha(data) != row["sha256"]:
                _fail("MULTIMODAL_NATIVE_SOURCE_CHANGED")
            if deadline is not None:
                _remaining(deadline)
        if deadline is not None:
            _remaining(deadline)

    def _run(self, argv, attempt, deadline):
        environment = {**self._environment, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_USE_MODELSCOPE": "0",
                       "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false"}
        with (attempt / "processor.log").open("xb") as output:
            _remaining(deadline)
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                       cwd=str(attempt), env=environment, start_new_session=True)
            try:
                return process.wait(timeout=_remaining(deadline))
            except BaseException as exc:
                # New session belongs only to this helper and any helper children.
                # Never look up or signal the model-service PID from the audit.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass  # Cleanup must not replace the original failure.
                if isinstance(exc, (subprocess.TimeoutExpired, MultimodalAdmissionError)):
                    _fail("MULTIMODAL_PROCESSOR_TIMEOUT")
                raise

    def _validate_summary(self, attempt, payload, image, source, bound, deadline):
        path = attempt / "native" / "summary.json"
        summary = _load(path)
        if (not isinstance(summary, dict) or summary.get("schema_version") != 1
                or any(type(summary.get(k)) is not int for k in ("schema_version", "generation_calls", "image_bytes",
                                                                "max_model_len", "max_output_tokens", "processor_apply_calls"))
                or summary.get("helper") != HELPER_VERSION or summary.get("status") != "processor_counted"
                or summary.get("method") != NATIVE_METHOD or summary.get("generation_calls") != 0
                or summary.get("server_usage_equivalence_verified") is not False
                or summary.get("request_sha256") != _digest(payload)
                or summary.get("messages_sha256") != _digest(payload["messages"])
                or summary.get("source_image_sha256") != _sha(image)
                or summary.get("image_bytes") != len(image) or summary.get("image_dimensions") != [source["width"], source["height"]]
                or summary.get("max_model_len") != 65536 or summary.get("max_output_tokens") != 512
                or summary.get("fits_context") is not True or summary.get("bound_input_files") != bound
                or summary.get("processor_apply_calls") != 1):
            _fail("MULTIMODAL_NATIVE_SUMMARY_INVALID")
        identity = summary.get("identity")
        service, expected_sources = self._audit["service_process"], self._audit["source_index"]
        expected_files = [{k: row[k] for k in ("path", "sha256", "bytes")} for row in expected_sources]
        if (not isinstance(identity, dict) or identity.get("service_pid") != service["pid"]
                or identity.get("service_argv") != [x for x in service["cmdline"] if x]
                or identity.get("service_cmdline_sha256") != service["cmdline_sha256"]
                or summary.get("service_cmdline_sha256") != service["cmdline_sha256"]
                or identity.get("package_versions") != self._audit["packages"]
                or identity.get("verified_audited_files") != expected_files
                or identity.get("model_metadata_files") != self._metadata):
            _fail("MULTIMODAL_NATIVE_IDENTITY_CHANGED")
        if self._binding is not None and identity.get("model_binding") != self._binding:
            _fail("MULTIMODAL_NATIVE_IDENTITY_CHANGED")
        if self._binding is not None and identity.get("server_policy") != self._server_policy:
            _fail("MULTIMODAL_NATIVE_IDENTITY_CHANGED")
        if self._cache_binding is not None:
            cache_binding = deepcopy(self._cache_binding)
            cache_binding["input_path"] = str(attempt / "modelinfo-cache.json")
            cache_path = attempt / "native/cache/vllm/modelinfos"
            from .document_native_processor import MODELINFO_CACHE_FILENAME
            cache_path = cache_path / MODELINFO_CACHE_FILENAME
            expected_cache = {"binding": cache_binding, "isolated_path": str(cache_path),
                              "sha256": cache_binding["sha256"], "official_module_hash_verified": True,
                              "subprocess_fallback_allowed": False}
            if (summary.get("modelinfo_cache") != expected_cache
                    or _sha(_read(cache_path)) != cache_binding["sha256"]):
                _fail("MULTIMODAL_NATIVE_CACHE_IDENTITY_CHANGED")
        for row in expected_files:
            data = _read(row["path"])
            if _sha(data) != row["sha256"] or len(data) != row["bytes"]:
                _fail("MULTIMODAL_NATIVE_SOURCE_CHANGED")
            _remaining(deadline)
        original = _load(attempt / "native" / "original-template-prompt-token-ids.json")
        final = _load(attempt / "native" / "expanded-prompt-token-ids.json")
        for tokens, count_key, hash_key in ((original, "original_template_prompt_tokens", "original_template_prompt_token_ids_sha256"),
                                           (final, "expanded_prompt_tokens", "final_prompt_token_ids_sha256")):
            if (not isinstance(tokens, list) or not tokens or any(type(x) is not int or x < 0 for x in tokens)
                    or type(summary.get(count_key)) is not int or len(tokens) != summary[count_key]
                    or summary.get(hash_key) != _digest(tokens)):
                _fail("MULTIMODAL_TOKEN_PROOF_INVALID")
        ranges = summary.get("image_placeholder_ranges")
        if (len(final) <= len(original) or len(final) + 512 > 65536
                or not isinstance(ranges, list) or len(ranges) != 1 or not isinstance(ranges[0], dict)
                or set(ranges[0]) != {"offset", "length"}
                or any(type(ranges[0].get(k)) is not int for k in ("offset", "length"))
                or ranges[0]["offset"] < 0 or ranges[0]["length"] <= 1
                or ranges[0]["offset"] + ranges[0]["length"] > len(final)):
            _fail("MULTIMODAL_EXPANSION_INVALID")
        objects = summary.get("native_objects")
        if (not isinstance(objects, dict) or set(objects) != {"renderer", "tokenizer", "input_preprocessor",
                "multimodal_processor", "hf_processor", "image_processor"}
                or not isinstance(objects.get("multimodal_processor"), dict)
                or objects["multimodal_processor"].get("class") != "vllm.model_executor.models.qwen3_vl.Qwen3VLMultiModalProcessor"):
            _fail("MULTIMODAL_NATIVE_OBJECT_INVALID")
        audited = {row["path"]: row["sha256"] for row in expected_files}
        for row in objects.values():
            if (not isinstance(row, dict) or not isinstance(row.get("class"), str) or not row["class"]
                    or not isinstance(row.get("source_file"), str) or not Path(row["source_file"]).is_absolute()
                    or Path(row["source_file"]).suffix != ".py" or not _valid_sha(row.get("source_sha256"))
                    or _sha(_read(row["source_file"])) != row["source_sha256"]
                    or row["source_file"] not in audited
                    or audited[row["source_file"]] != row["source_sha256"]):
                _fail("MULTIMODAL_NATIVE_OBJECT_INVALID")
            _remaining(deadline)
        tensors = summary.get("cpu_tensors")
        if (not isinstance(tensors, list) or not tensors or any(not isinstance(row, dict)
                or row.get("device") != "cpu" or not isinstance(row.get("path"), str)
                or not isinstance(row.get("dtype"), str) or not isinstance(row.get("shape"), list)
                or any(type(x) is not int or x <= 0 for x in row["shape"]) for row in tensors)):
            _fail("MULTIMODAL_CPU_PROOF_INVALID")
        return summary, path

    def __call__(self, payload, *, deadline):
        self.directory.mkdir(parents=True, exist_ok=True)
        attempt = self.directory / ("admission-" + uuid4().hex)
        attempt.mkdir(mode=0o700)
        started = time.monotonic()
        record = {"schema_version": 1, "version": VERSION, "status": "failed",
                  "started_at": datetime.now(timezone.utc).isoformat(), "generation_calls": 0,
                  "identity": self.identity, "request_sha256": None, "processor_returncode": None}
        try:
            if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0:
                _fail("MULTIMODAL_DEADLINE_INVALID")
            _remaining(deadline)
            original = _packed(payload)
            if len(original) > 16 * 1024 * 1024:
                _fail("MULTIMODAL_REQUEST_BYTES")
            request = json.loads(original)
            image, source = _request(request)
            record["request_sha256"] = _sha(original)
            self._verify_fixed(deadline)
            inputs = {"request.json": original, "image.png": image, "source.json": _packed(source),
                      "path-audit.json": self._audit_bytes}
            if self._binding is not None:
                inputs["model-binding.json"] = _packed(self._binding)
                inputs["server-policy.json"] = _packed(self._server_policy)
            if self._cache_binding is not None:
                inputs["modelinfo-cache.json"] = self._cache_bytes
            for name, data in inputs.items():
                (attempt / name).write_bytes(data)
                _remaining(deadline)
            bound = {str(attempt / name): _sha(data) for name, data in inputs.items()}
            bound[str(self.helper_path)] = self._helper_sha
            if self._cache_binding is not None:
                bound.update({row["path"]: row["sha256"] for row in self._cache_binding["source_files"]})
            argv = [str(self.processor_python), "-I", "-B", str(self.helper_path),
                    "--request", str(attempt / "request.json"), "--output", str(attempt / "native"),
                    "--source", str(attempt / "source.json"), "--image", str(attempt / "image.png"),
                    "--path-audit", str(attempt / "path-audit.json"), "--base-model", str(self._base_model)]
            if self._binding is not None:
                argv.extend(["--model-binding", str(attempt / "model-binding.json"),
                             "--server-policy", str(attempt / "server-policy.json")])
            if self._cache_binding is not None:
                argv.extend(["--modelinfo-cache", str(attempt / "modelinfo-cache.json"),
                             "--modelinfo-cache-sha256", self._cache_binding["sha256"]])
            _save(attempt / "launch.json", {"argv": argv, "helper_sha256": self._helper_sha,
                "runtime_environment": self._environment, "generation_requests_allowed": 0,
                "remaining_seconds_at_launch": _remaining(deadline)})
            returncode = self._run(argv, attempt, deadline)
            record["processor_returncode"] = returncode
            self._verify_fixed(deadline)
            if _packed(payload) != original or any(_sha(_read(path)) != sha for path, sha in bound.items()):
                _fail("MULTIMODAL_REQUEST_OR_INPUT_CHANGED")
            if returncode != 0:
                _fail("MULTIMODAL_NATIVE_PROCESSOR_FAILED")
            summary, summary_path = self._validate_summary(attempt, request, image, source, bound, deadline)
            self._verify_fixed(deadline)
            if _packed(payload) != original:
                _fail("MULTIMODAL_REQUEST_OR_INPUT_CHANGED")
            _remaining(deadline)
            admission = {"status": "processor_counted", "method": "installed_vllm_native_processor",
                "request_sha256": _sha(original), "image_payload_sha256": _sha(image),
                "expanded_prompt_tokens": summary["expanded_prompt_tokens"], "max_model_len": 65536,
                "reserved_output_tokens": 512, "final_prompt_token_ids_sha256": summary["final_prompt_token_ids_sha256"],
                "processor_identity_sha256": self.identity["processor_identity_sha256"],
                "service_identity_sha256": self.identity["service_identity_sha256"],
                "native_summary": {"path": str(summary_path), "sha256": _sha(_read(summary_path))},
                "native_identity_sha256": _digest(summary["identity"]),
                "native_objects_sha256": _digest(summary["native_objects"]),
                "image_placeholder_ranges": summary["image_placeholder_ranges"],
                "server_usage_equivalence_verified": False, "generation_calls": 0}
            if self._cache_binding is not None:
                admission["modelinfo_cache"] = deepcopy(summary["modelinfo_cache"])
            _save(attempt / "admission.json", admission)
            _remaining(deadline)
            record.update(status="admitted", admission_sha256=_digest(admission))
            return admission
        except Exception as exc:
            code = exc.code if isinstance(exc, MultimodalAdmissionError) else "MULTIMODAL_ADMISSION_FAILED"
            record.update(status="failed", error_code=code, error_type=type(exc).__name__)
            raise MultimodalAdmissionError(code, attempt) from None
        finally:
            record.update(finished_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
            try:
                _save(attempt / "summary.json", record)
                if record["status"] == "admitted":
                    _remaining(deadline)
            except MultimodalAdmissionError:
                record.update(status="failed", error_code="MULTIMODAL_DEADLINE_EXCEEDED")
                try:
                    _save(attempt / "summary.json", record)
                except Exception:
                    pass
                raise MultimodalAdmissionError("MULTIMODAL_DEADLINE_EXCEEDED", attempt) from None
            except Exception:
                if record["status"] == "admitted":
                    raise MultimodalAdmissionError("MULTIMODAL_RECEIPT_PERSIST_FAILED", attempt) from None
