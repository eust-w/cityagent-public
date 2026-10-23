"""CLI boundary for the local-server divide workflows (no model loading here)."""
from __future__ import annotations

from contextlib import nullcontext
import copy
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import shutil
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .artifact_publish import publish_no_replace
from .divide_budget import DivideBudget
from .divide_conquer_workflow import VERSION, DivideSettings, DivideWorkflow
from .docx_writer import write_extraction_docx
from .evidence_manifest import build_manifest, manifest_from_text
from .local_model_client import LocalModelClient
from .rsi_runtime import Dispatcher, WorkLock
from .run_state import (
    atomic_json,
    bind_work_directory,
    file_digest,
    json_digest,
    validate_output_names,
)
from .training_evidence_batch import load_training_evidence_batch, verify_training_evidence_batch


def add_arguments(parser, *, include_workflow=True):
    if include_workflow:
        parser.add_argument("--workflow", choices=("whole", "divide-fixed", "divide-react", "document-agent"), default="whole",
                            help="whole 保留旧基线；divide-react 使用真实 TongAgents 分块和动态回查")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model", default="cityagent-qwen36-27b-epoch-002")
    parser.add_argument("--served-adapter", type=Path, default=Path("/workspace/work/qwen36_bridge_lora_epoch002"))
    parser.add_argument("--served-merged-model", type=Path,
                        help="显式使用含 completed conversion_manifest.json 的合并权重；仍核验原 LoRA 身份")
    parser.add_argument("--chunk-tokens", type=int, default=6000)
    parser.add_argument("--chunk-output-tokens", type=int, default=4096)
    parser.add_argument("--agent-max-iterations", type=int, default=6)
    parser.add_argument("--agent-max-calls", type=int, default=8)
    parser.add_argument("--model-timeout", type=int, default=1800,
                        help="分治模式单次模型请求等待秒数，不改变 token 上限")
    parser.add_argument("--divide-report-timeout", type=float,
                        help="分治整篇累计秒数，从转换前开始；默认不新增整篇时间限制")
    parser.add_argument("--divide-max-model-calls", type=int,
                        help="分治整篇真实生成请求总上限，含 SDK、抽取、复核及失败请求；默认不新增限制")
    parser.add_argument("--agent-output-tokens", type=int, default=1024,
                        help="分治 SDK 每次生成输出上限；严格对照时可显式设 4096")
    parser.add_argument("--review-output-tokens", type=int, default=1024,
                        help="分治冲突复核输出上限，同时不超过 --chunk-output-tokens")
    parser.add_argument("--document-workers", type=int, default=1,
                        help="分治模式同时处理报告数（1–8，默认串行）；独立 CPU 进程共享一个模型服务")
    parser.add_argument("--isolate-documents", action="store_true",
                        help="每篇报告均用新的独立进程，包括单并发；用于统一并发对照的 SDK 状态隔离")
    parser.add_argument("--training-evidence-json", type=Path,
                        help="开发专用：只含训练来源原文的 JSON，不能含目标标签；与 --input-dir 互斥")


def _training_manifest(path: Path) -> tuple[str, dict]:
    source_bytes = path.read_bytes()
    raw = json.loads(source_bytes)
    allowed = {"evidence_id", "source", "training_line", "text", "source_sha256", "text_sha256"}
    if (not isinstance(raw, dict) or set(raw) - allowed or not isinstance(raw.get("text"), str)
            or not raw["text"].strip() or type(raw.get("training_line")) is not int
            or raw["training_line"] < 1):
        raise ValueError("Training evidence must contain only identified input text, never output/labels")
    name = raw.get("evidence_id")
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Unsafe training evidence ID")
    if raw.get("text_sha256") is not None and raw["text_sha256"] != hashlib.sha256(raw["text"].encode()).hexdigest():
        raise ValueError("Training text_sha256 does not match input text")
    manifest = manifest_from_text(raw["text"])
    manifest["training_source"] = {k: v for k, v in raw.items() if k != "text"}
    manifest["source_json_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    return name, manifest


class _MergedChatTokenizer:
    """Render the verified deployed template without importing torch/model code.

    Uses the same Rust tokenizer + sandboxed Jinja configuration independently
    checked on the PPU SDK environment. It does not substitute a new template.
    Only the text worker's template is rendered here; SDK tool requests still
    go through the serving engine's tool parser and template normalization.
    """

    def __init__(self, directory, expected_hashes):
        from jinja2 import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        snapshots = {}
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            raw = (Path(directory) / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected_hashes.get(name):
                raise ValueError(f"Verified chat tokenizer changed: {name}")
            snapshots[name] = raw
        template_source = snapshots["chat_template.jinja"].decode("utf-8")
        if "{% generation" in template_source or "{%- generation" in template_source:
            raise ValueError("CPU chat renderer does not support generation tracking blocks")
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                    extensions=["jinja2.ext.loopcontrols"])

        def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            # HF's override preserves Unicode and does not HTML-escape values.
            return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                              separators=separators, sort_keys=sort_keys)

        def raise_exception(message):
            raise TemplateError(message)

        environment.filters["tojson"] = tojson
        environment.globals["raise_exception"] = raise_exception
        environment.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.template = environment.from_string(template_source)
        self.tokenizer = Tokenizer.from_str(snapshots["tokenizer.json"].decode("utf-8"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        config = json.loads(snapshots["tokenizer_config.json"])
        self.special_tokens = {
            key: (value.get("content") if isinstance(value, dict) else value)
            for key, value in config.items() if key.endswith("_token")
        }

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False, tools=None):
        normalized = copy.deepcopy(messages)
        for message in normalized:
            for call in message.get("tool_calls") or []:
                function = call.get("function", call)
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = json.loads(function["arguments"])
        prompt = self.template.render(**{**self.special_tokens, "messages": normalized,
                                         "tools": tools, "add_generation_prompt": add_generation_prompt,
                                         "enable_thinking": enable_thinking})
        return self.encode(prompt, add_special_tokens=False) if tokenize else prompt

    def encode(self, prompt, *, add_special_tokens=False):
        return self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens).ids


def _make_client(args):
    optional = {}
    if getattr(args, "chat_tokenizer_directory", None) is not None:
        optional["chat_tokenizer"] = _MergedChatTokenizer(args.chat_tokenizer_directory,
                                                        args.chat_tokenizer_hashes)
    return LocalModelClient(args.vllm_url, args.served_model,
                            api_key=os.environ.get("PPU_VLLM_API_KEY", "local-cityagent"),
                            max_model_len=args.max_model_len, max_output_tokens=args.chunk_output_tokens,
                            timeout=args.model_timeout,
                            enable_thinking=getattr(args, "document_enable_thinking", False),
                            temperature=getattr(args, "document_temperature", 0),
                            dynamic_output_budget=getattr(args, "document_dynamic_output_budget", False), **optional)


def _merged_identity(args, client, adapter_sha256):
    """Bind a merged export to disk bytes, its adapter, and server metadata.

    Hash exported artifacts once in the parent, before the measured document
    batch. This proves file identity, not merge arithmetic or live GPU values.
    """
    root = args.served_merged_model.resolve(strict=True)
    manifest_path = root / "conversion_manifest.json"
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    adapter_config = args.served_adapter / "adapter_config.json"
    adapter_config_sha = file_digest(adapter_config)
    adapter_settings = json.loads(adapter_config.read_bytes())

    def validate_manifest(record, expected_mode=None):
        mode = record.get("mode")
        if (record.get("schema_version") != 1 or record.get("status") != "completed"
                or mode not in {"merged_bf16", "w8a8_int8"} or (expected_mode and mode != expected_mode)):
            raise ValueError("Merged model requires a completed supported conversion manifest")
        if record.get("lora_coverage") != {"expected": 256, "merged": 256}:
            raise ValueError("Merged model LoRA coverage must be complete (256/256)")
        adapter = record.get("adapter" if mode == "merged_bf16" else "original_adapter")
        if (not isinstance(adapter, dict) or adapter.get("weights_sha256") != adapter_sha256
                or adapter.get("config_sha256") != adapter_config_sha
                or adapter.get("path") != str(args.served_adapter.resolve())
                or adapter.get("rank") != adapter_settings.get("r")
                or adapter.get("alpha") != adapter_settings.get("lora_alpha")):
            raise ValueError("Merged model original adapter identity does not match requested LoRA")
        if (type(adapter.get("rank")) is not int or adapter["rank"] < 1
                or adapter.get("scale") != adapter["alpha"] / adapter["rank"]):
            raise ValueError("Merged model adapter scaling is inconsistent")
        return mode, adapter

    mode, adapter = validate_manifest(manifest)
    artifacts = manifest.get("artifacts")
    required = {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"}
    if not isinstance(artifacts, dict) or not required <= artifacts.keys():
        raise ValueError("Merged model manifest must bind config, index and tokenizer artifacts")
    verified_artifacts, parsed_artifacts = {}, {}
    for name, metadata in artifacts.items():
        if (not isinstance(name, str) or Path(name).name != name or name in {".", ".."}
                or not isinstance(metadata, dict)):
            raise ValueError("Unsafe merged model artifact")
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != metadata.get("bytes"):
            raise ValueError(f"Merged model artifact size/type mismatch: {name}")
        if name in {"config.json", "model.safetensors.index.json"}:
            # Parse the same snapshot whose checksum is verified.
            snapshot = path.read_bytes()
            digest = hashlib.sha256(snapshot).hexdigest()
            parsed_artifacts[name] = json.loads(snapshot)
        else:
            digest = file_digest(path)
        if digest != metadata.get("sha256"):
            raise ValueError(f"Merged model artifact checksum mismatch: {name}")
        verified_artifacts[name] = {"sha256": digest, "bytes": metadata["bytes"]}
    index = parsed_artifacts["model.safetensors.index.json"]
    mapping = index.get("weight_map")
    if (not isinstance(mapping, dict) or not mapping
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items())
            or manifest.get("output_tensor_count") != len(mapping)):
        raise ValueError("Merged model tensor index/count mismatch")
    shards = {name for name in artifacts if name.endswith(".safetensors")}
    if set(mapping.values()) != shards or {p.name for p in root.glob("*.safetensors")} != shards:
        raise ValueError("Merged model index/shard coverage mismatch")
    config = parsed_artifacts["config.json"]
    quantization = config.get("quantization_config")
    parent_identity = None
    if mode == "merged_bf16":
        if quantization or config.get("text_config", {}).get("quantization_config"):
            raise ValueError("BF16 merged model unexpectedly declares quantization")
    else:
        if (not isinstance(quantization, dict) or quantization.get("quant_method") != "w8a8_int8"
                or quantization.get("activation_scheme") != "dynamic" or quantization.get("is_dynamic") is not True):
            raise ValueError("INT8 merged model quantization config mismatch")
        counts = manifest.get("quantization", {})
        if counts.get("quantized_matrices_expected") != 400 or counts.get("quantized_matrices") != 400:
            raise ValueError("INT8 merged model quantized matrix coverage must be 400/400")
        source = Path(manifest.get("source", ""))
        if not source.is_absolute():
            raise ValueError("INT8 parent merged model path must be absolute")
        parent_raw = (source / "conversion_manifest.json").read_bytes()
        parent_sha = hashlib.sha256(parent_raw).hexdigest()
        if parent_sha != manifest.get("merged_manifest_sha256"):
            raise ValueError("INT8 parent merged manifest checksum mismatch")
        parent = json.loads(parent_raw)
        _, parent_adapter = validate_manifest(parent, "merged_bf16")
        if parent_adapter != adapter:
            raise ValueError("INT8 parent adapter provenance mismatch")
        for name, source_key in (("config.json", "source_config_sha256"),
                                 ("model.safetensors.index.json", "source_index_sha256")):
            expected = parent.get("artifacts", {}).get(name, {}).get("sha256")
            if not expected or manifest.get(source_key) != expected or file_digest(source / name) != expected:
                raise ValueError("INT8 parent config/index provenance mismatch")
        parent_identity = {"path": str(source), "manifest_sha256": parent_sha,
                           "validation_scope": "manifest, adapter, config and index; parent weight shards not rehashed"}
    entries = [entry for entry in client.model_catalog if entry.get("id") == args.served_model]
    if len(entries) != 1 or entries[0].get("parent"):
        raise ValueError("Merged model must be served as one unambiguous base-model entry")
    entry = entries[0]
    served_root = entry.get("root")
    model_info = None
    if isinstance(served_root, str) and Path(served_root).is_absolute() and Path(served_root).resolve() == root:
        binding_source = "model_catalog.root"
    elif served_root == args.served_model and len([item for item in client.model_catalog if not item.get("parent")]) == 1:
        # SGLang's base ModelCard uses the served alias as root. Its separate
        # model_info response carries the actual model_path; never invent root.
        model_info, _ = client._request("/model_info", None, kind="model_identity")
        served_path = model_info.get("model_path")
        tokenizer_path = model_info.get("tokenizer_path")
        if (not isinstance(served_path, str) or not Path(served_path).is_absolute()
                or Path(served_path).resolve() != root
                or not isinstance(tokenizer_path, str) or not Path(tokenizer_path).is_absolute()
                or Path(tokenizer_path).resolve() != root
                or model_info.get("served_model_name", args.served_model) != args.served_model):
            raise ValueError("Served model_info does not bind the requested merged model path")
        binding_source = "model_catalog.alias+model_info.model_path"
    else:
        raise ValueError("Served model catalog does not bind the requested merged model path")
    if file_digest(manifest_path) != manifest_sha:
        raise RuntimeError("Merged conversion manifest changed during identity verification")
    return {"kind": "merged_export", "path": str(root), "mode": mode,
            "manifest_sha256": manifest_sha, "lora_coverage": manifest["lora_coverage"],
            "original_adapter": adapter, "verified_artifacts": verified_artifacts,
            "exported_artifacts_full_checksum_verified": True, "parent": parent_identity,
            "binding_source": binding_source, "model_info": model_info,
            "served_tokenizer_path_verified": model_info is not None,
            "validation_scope": "exported file checksums and provenance + server-declared path; not live GPU tensor values or merge arithmetic"}


def _original_input(name, source, args):
    if source is not None:
        return source
    if getattr(args, "divide_training_batch_identity", None) is not None:
        return args.divide_training_originals[name]
    return args.training_evidence_json


def _verify_training_batch(args, original=None):
    identity = getattr(args, "divide_training_batch_identity", None)
    if identity is not None:
        verify_training_evidence_batch(identity, original=original)


def _run_document(name, manifest, source, args, settings, fingerprint, source_sha256, convert_doc):
    # A parent may die while a spawned worker remains alive. Its report lock
    # prevents a new owner from recovering that still-running report.
    with WorkLock(args.work_dir / "documents" / name):
        return _run_owned_document(name, manifest, source, args, settings, fingerprint,
                                   source_sha256, convert_doc)


def _run_owned_document(name, manifest, source, args, settings, fingerprint, source_sha256, convert_doc):
    """One isolated report; called directly or in a spawned CPU worker.

    TongAgents keeps a process-global tool registry and active session. Using
    processes preserves its session lock while allowing different reports to
    contact the same model server concurrently. No worker loads model weights.
    """
    budget_started = time.monotonic()
    start = time.perf_counter()
    directory = args.work_dir / "documents" / name
    original_cwd = Path.cwd()
    isolated_cwd = multiprocessing.current_process().name != "MainProcess"
    sdk_environment = {
        "LOG_TRACE_LOG_DIR": str(directory / "sdk-logs"),
        "LOG_TRACE_LOG_FILE": "tongagents.log",
        "LOG_TRACE_DB_PATH": str(directory / "sdk-logs" / "traces.db"),
        "LOG_TRACE_BACKEND": "local",
    }
    previous_environment = {key: os.environ.get(key) for key in sdk_environment}
    summary = {"name": name, "status": "failed", "worker_pid": os.getpid(),
               "worker_cwd": str(directory if isolated_cwd else original_cwd)}
    budget = None
    print(f"START {name}: worker_pid={os.getpid()}", flush=True)
    try:
        report_timeout = getattr(args, "divide_report_timeout", None)
        max_model_calls = getattr(args, "divide_max_model_calls", None)
        if report_timeout is not None or max_model_calls is not None:
            budget = DivideBudget(directory, started=budget_started, report_timeout=report_timeout,
                                  max_model_calls=max_model_calls)
            budget.check(stage="document_entry")
        directory.mkdir(parents=True, exist_ok=True)
        original = _original_input(name, source, args)
        _verify_training_batch(args, original)
        output = args.result_dir / (name + ".docx")
        if output.exists() or output.is_symlink():
            if not args.resume:
                raise FileExistsError("Unexpected existing output")
            _reuse_publication(output, directory, fingerprint)
            summary.update(status="completed", output=output.name, reused=True)
            return summary
        if isolated_cwd:
            # SDK relative stores (for example default_sql.db) must not be
            # shared by concurrent reports. Parent/serial execution keeps cwd.
            os.chdir(directory)
            # Official SDK env overrides YAML defaults. Configure before its
            # first lazy import, including its otherwise user-global trace DB.
            os.environ.update(sdk_environment)
        client = _make_client(args)
        if budget is not None:
            client = budget.wrap(client)
        agent_output_tokens = getattr(args, "agent_output_tokens", 1024)
        react_config = {"url": client.base_url + "/v1", "model": args.served_model,
                        "api_key": client.api_key, "max_tokens": agent_output_tokens, "timeout": args.model_timeout,
                        "max_iterations": args.agent_max_iterations, "max_calls": args.agent_max_calls,
                        "enable_thinking": getattr(client, "enable_thinking", False),
                        "temperature": getattr(client, "temperature", 0)}
        if budget is not None:
            budget.check(stage="before_conversion")
        cached, bound = directory / "manifest.json", directory / "workflow_identity.json"
        if manifest is None and args.resume and cached.is_file() and bound.is_file():
            manifest = json.loads(cached.read_text())
            expected = json.loads(bound.read_text())["manifest_sha256"]
            if (json_digest(manifest) != expected
                    or manifest.get("original_document_sha256", manifest.get("source_json_sha256")) != source_sha256):
                raise ValueError("Cached manifest identity changed")
        if source is not None and manifest is None:
            target = convert_doc(source, directory / "converted") if source.suffix.lower() == ".doc" else source
            manifest = build_manifest(target)
            manifest["original_document_sha256"] = source_sha256
        if budget is not None:
            budget.check(stage="after_conversion")
        workflow_options = {}
        if getattr(args, "review_output_tokens", 1024) != 1024:
            workflow_options["review_output_tokens"] = args.review_output_tokens
        gateway_context = (budget.gateway(client, output_tokens=agent_output_tokens,
                                         enabled=args.workflow == "divide-react")
                           if budget is not None else nullcontext(None))
        with gateway_context as gateway, Dispatcher(client, capacity=1) as dispatcher:
            if gateway is not None:
                react_config["url"] = gateway.url
            dispatcher.begin_foreground()
            try:
                result = DivideWorkflow(manifest, client, react_config, directory,
                                        settings=settings, mode=args.workflow, resume=args.resume,
                                        run_fingerprint=fingerprint, dispatcher=dispatcher,
                                        **workflow_options).run()
            finally:
                dispatcher.end_foreground()
        if budget is not None:
            budget.check(stage="before_export")
        _verify_training_batch(args, original)
        if file_digest(original) != source_sha256:
            label = "Input" if source is not None else "Training evidence input"
            raise RuntimeError(f"{label} changed during execution")
        pending = args.result_dir / (".pending-" + uuid4().hex + ".docx")
        published = None
        try:
            write_extraction_docx(result["data"], str(pending))
            if budget is not None:
                budget.check(stage="after_export")
            receipt = result["receipt"]
            receipt.update(run_fingerprint=fingerprint,
                           output={"name": output.name, "sha256": file_digest(pending)},
                           report_sha256=json_digest(result["data"]))
            # Rendering may take time: recheck the bound batch immediately
            # before publishing, without rereading other workers' inputs.
            _verify_training_batch(args, original)
            if budget is not None:
                budget.persist()
                receipt["divide_budget"] = budget.snapshot()
            # Preserve provenance and audit before exposing the completed file.
            atomic_json(directory / "execution.json", receipt)
            if budget is not None:
                budget.check(stage="before_publish")
            published = publish_no_replace(pending, output, args.work_dir / "publish-locks")
            receipt["publication"] = published
            atomic_json(directory / "execution.json", receipt)
        except BaseException:
            if published is not None and output.exists():
                stat = output.stat()
                if (stat.st_dev, stat.st_ino) == (published["device"], published["inode"]):
                    output.unlink()
            raise
        finally:
            pending.unlink(missing_ok=True)
        summary.update(status="completed", output=output.name)
        print(f'DONE {name}: defects={len(result["data"]["病害明细"])} '
              f'advice={len(result["data"]["维修建议"])} coverage=complete', flush=True)
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED {name}: {summary['error']}", flush=True)
    finally:
        if budget is not None:
            try:
                budget.persist()
            except Exception as exc:
                summary.setdefault("error", f"{type(exc).__name__}: budget receipt persistence failed")
                summary["status"] = "failed"
                summary["budget_receipt_error"] = type(exc).__name__
            summary["divide_budget"] = budget.snapshot()
        if isolated_cwd:
            os.chdir(original_cwd)
            for key, value in previous_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        summary["seconds"] = time.perf_counter() - start
        atomic_json(directory / "document_summary.json", summary)
    return summary


def run_divide(args, *, code_root: Path, convert_doc: callable) -> None:
    batch_started = time.perf_counter()
    # Spawned workers change into their private document directory; every path
    # sent to them must retain its original meaning. Do not mutate caller args.
    args = copy.copy(args)
    batch_path = getattr(args, "training_evidence_manifest", None)
    if batch_path is not None:
        # Resolving first would erase evidence of an input/output symlink.
        for path in (batch_path, args.work_dir, args.result_dir):
            path = Path(path).absolute()
            if any(part.is_symlink() for part in (path, *path.parents)):
                raise ValueError("Training batch paths must not contain symlinks")
    for field, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, field, value.resolve())
    code_root = code_root.resolve()
    if args.text_dir:
        raise ValueError("Divide DOCX mode requires the structural original; use --training-evidence-json for text diagnostics")
    batch_path = getattr(args, "training_evidence_manifest", None)
    if sum(bool(value) for value in (args.input_dir, args.training_evidence_json, batch_path)) != 1:
        raise ValueError("Specify exactly one explicit --input-dir, --training-evidence-json or --training-evidence-manifest; no dataset auto-discovery")
    if batch_path is not None and args.limit is not None:
        raise ValueError("--limit cannot truncate an explicit training evidence manifest")
    if args.workflow == "divide-react" and args.engine != "tongagents":
        raise ValueError("divide-react requires --engine tongagents")
    if args.fusion_policy != "model-preserving":
        raise ValueError("Divide workflow does not permit legacy rule overrides")
    settings = DivideSettings(chunk_tokens=args.chunk_tokens, output_tokens=args.chunk_output_tokens,
                              max_extraction_attempts=args.max_attempts)
    if any(type(value) is not int or value < 1 for value in asdict(settings).values()):
        raise ValueError("Divide settings must be positive integers")
    if not 0 < args.chunk_tokens + args.chunk_output_tokens < args.max_model_len:
        raise ValueError("Chunk input/output must leave room within model context")
    if args.agent_max_calls < 1 or args.agent_max_iterations < 1:
        raise ValueError("Agent budgets must be positive")
    if args.model_timeout < 1:
        raise ValueError("Model timeout must be positive")
    report_timeout = getattr(args, "divide_report_timeout", None)
    max_model_calls = getattr(args, "divide_max_model_calls", None)
    if report_timeout is not None and (type(report_timeout) not in (int, float)
            or not math.isfinite(report_timeout) or report_timeout <= 0):
        raise ValueError("Divide report timeout must be positive and finite")
    if max_model_calls is not None and (type(max_model_calls) is not int or max_model_calls < 1):
        raise ValueError("Divide maximum model calls must be a positive integer")
    agent_output_tokens = getattr(args, "agent_output_tokens", 1024)
    review_output_tokens = getattr(args, "review_output_tokens", 1024)
    if any(type(value) is not int or not 0 < value < args.max_model_len
           for value in (agent_output_tokens, review_output_tokens)):
        raise ValueError("Divide agent/review output tokens must be positive integers within context")
    if type(args.document_workers) is not int or not 1 <= args.document_workers <= 8:
        raise ValueError("Document workers must be an integer between 1 and 8")
    if args.resume and (report_timeout is not None or max_model_calls is not None):
        raise ValueError("Explicit divide-budget resume is not yet supported; original checkpoints are preserved")
    if not args.resume and args.result_dir.exists() and any(args.result_dir.iterdir()):
        raise FileExistsError("Use an empty result directory for divide experiments")
    batch_identity = None
    if batch_path is not None:
        if (args.work_dir == args.result_dir or args.work_dir in args.result_dir.parents
                or args.result_dir in args.work_dir.parents):
            raise ValueError("Training batch work/result paths must not overlap")
        rows, batch_identity = load_training_evidence_batch(
            batch_path, forbidden_roots=(args.work_dir, args.result_dir))
        args.divide_training_batch_identity = batch_identity
        args.divide_training_originals = {name: original for name, _, original in rows}
    client = _make_client(args)
    if args.served_model not in client.models():
        raise ValueError("Requested model is not served by local vLLM")
    adapter_file = args.served_adapter / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise ValueError("Local served adapter weights are required for the execution identity")
    adapter_sha256 = file_digest(adapter_file)
    if args.served_merged_model:
        model_identity = _merged_identity(args, client, adapter_sha256)
        if model_identity["binding_source"] == "model_catalog.alias+model_info.model_path":
            # This installed SGLang only accepts prompt strings at /tokenize.
            # Its model_info binding verified both model and tokenizer paths.
            tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
            if not all(name in model_identity["verified_artifacts"] for name in tokenizer_files):
                raise ValueError("SGLang local chat rendering requires verified tokenizer/template artifacts")
            args.chat_tokenizer_directory = args.served_merged_model
            args.chat_tokenizer_hashes = {name: model_identity["verified_artifacts"][name]["sha256"]
                                         for name in tokenizer_files}
            # Catch missing CPU dependencies/templates before scheduling work.
            _MergedChatTokenizer(args.chat_tokenizer_directory, args.chat_tokenizer_hashes)
    else:
        if not any(item.get("id") == args.served_model and item.get("parent")
                   and item.get("root") == str(args.served_adapter.resolve()) for item in client.model_catalog):
            raise ValueError("Served model catalog does not bind this model ID to the requested LoRA path")
        model_identity = {"kind": "dynamic_lora", "adapter_path": str(args.served_adapter.resolve())}
    sdk_version = None
    if args.workflow == "divide-react":
        sdk_version = importlib.metadata.version("tongagents")
    source_files = []
    if batch_identity is not None:
        prepared = [(name, manifest, None) for name, manifest, _ in rows]
        source_files = [original for _, _, original in rows]
    elif args.training_evidence_json:
        name, manifest = _training_manifest(args.training_evidence_json)
        prepared = [(name, manifest, None)]
        source_files = [args.training_evidence_json]
    else:
        if not args.input_dir.is_dir():
            raise ValueError("Input directory does not exist")
        source_files = sorted(p for p in args.input_dir.rglob("*") if p.is_file()
                              and p.suffix.lower() in (".doc", ".docx") and not p.name.startswith("~$"))
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("limit must be positive")
            source_files = source_files[:args.limit]
        if not source_files:
            raise ValueError("Input directory has no reports")
        if any(p.is_symlink() for p in source_files):
            raise ValueError("Symlink inputs are unsupported")
        validate_output_names(source_files)
        if any(p.suffix.lower() == ".doc" for p in source_files) and not shutil.which("soffice"):
            raise ValueError("LibreOffice is required for .doc conversion")
        prepared = [(p.stem, None, p) for p in source_files]
    worker_count = min(args.document_workers, len(prepared))
    spawn_documents = args.isolate_documents or worker_count > 1
    worker_backend = "spawn-processes" if spawn_documents else "serial"
    code_files = [code_root / "run.py", *sorted((code_root / "src/bridge_evidence/contest").glob("*.py"))]
    identity = {"schema_version": 2, "workflow_version": VERSION, "workflow": args.workflow,
                "code": {p.relative_to(code_root).as_posix(): file_digest(p) for p in code_files},
                "sources": {str(p.resolve()): file_digest(p) for p in source_files},
                "adapter_sha256": adapter_sha256, "served_model": args.served_model,
                "model_identity": model_identity, "preflight_model_calls": [{k: v for k, v in call.items()
                    if k in {"kind", "endpoint", "input_sha256", "response_model"}} for call in client.calls],
                "chat_tokenization_mode": "verified_local_template+server_prompt_count"
                    if getattr(args, "chat_tokenizer_directory", None) else "server_chat_template",
                "model_catalog": client.model_catalog,
                "endpoint": client.base_url, "sdk_version": sdk_version,
                "settings": asdict(settings), "max_model_len": args.max_model_len,
                "text_request_policy": {"temperature": getattr(client, "temperature", 0),
                                        "enable_thinking": getattr(client, "enable_thinking", False),
                                        "dynamic_output_budget": getattr(args, "document_dynamic_output_budget", False)},
                "agent_max_iterations": args.agent_max_iterations, "agent_max_calls": args.agent_max_calls,
                "model_timeout": args.model_timeout,
                "result_dir": str(args.result_dir.resolve()),
                "document_workers": args.document_workers,
                "effective_document_workers": worker_count,
                "isolate_documents": args.isolate_documents,
                "document_worker_backend": worker_backend,
                "base_weights_full_checksum_verified": False}
    if report_timeout is not None or max_model_calls is not None:
        identity["divide_budget"] = {"report_timeout": report_timeout, "max_model_calls": max_model_calls}
    if agent_output_tokens != 1024 or review_output_tokens != 1024:
        identity["divide_output_tokens"] = {"agent": agent_output_tokens,
                                           "review": min(review_output_tokens, args.chunk_output_tokens)}
    if args.training_evidence_json and manifest["source_json_sha256"] != identity["sources"][str(args.training_evidence_json.resolve())]:
        raise RuntimeError("Training evidence changed before run identity was bound")
    if batch_identity is not None:
        identity["training_evidence_batch"] = batch_identity
        _verify_training_batch(args)
        for _, manifest, original in rows:
            if manifest["source_json_sha256"] != identity["sources"][str(original)]:
                raise RuntimeError("Training batch evidence changed before run identity was bound")
    if args.check:
        if batch_identity is not None:
            print(f"Explicit training text diagnostic: {len(prepared)} reports; original images/layout are not covered.", flush=True)
        print("Divide preflight passed; real SDK/model execution is not yet verified.", flush=True)
        return
    with WorkLock(args.work_dir):
        fingerprint = bind_work_directory(args.work_dir, identity, resume=args.resume)
        _run_bound(args, prepared, client, settings, identity, fingerprint, convert_doc, batch_started)


def _prepared_reports(prepared, args, identity, convert_doc):
    """Keep at most two CPU preparations ahead, in independent converter dirs.

    SDK sessions/model work remain serial. Real parallel SDK acceptance is not
    implied by CPU prefetch or by the dispatcher being reusable across reports.
    """
    def prepare(item):
        name, manifest, source = item
        if args.resume:
            if (args.result_dir / (name + ".docx")).exists():
                return item  # Output is hash-checked by the owning executor.
            directory = args.work_dir / "documents" / name
            cached, bound = directory / "manifest.json", directory / "workflow_identity.json"
            if cached.is_file() and bound.is_file():
                manifest = json.loads(cached.read_text())
                expected = json.loads(bound.read_text())["manifest_sha256"]
                source_digest = identity["sources"][str(_original_input(name, source, args).resolve())]
                if (json_digest(manifest) != expected
                        or manifest.get("original_document_sha256", manifest.get("source_json_sha256")) != source_digest):
                    raise ValueError("Cached manifest identity changed")
                return name, manifest, source
        if source is not None:
            converted = args.work_dir / "converted" / name
            target = convert_doc(source, converted) if source.suffix.lower() == ".doc" else source
            manifest = build_manifest(target)
            manifest["original_document_sha256"] = identity["sources"][str(source.resolve())]
            if file_digest(source) != manifest["original_document_sha256"]:
                raise RuntimeError("Input changed during preparation")
        return name, manifest, source

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="divide-prepare") as pool:
        remaining, pending = iter(prepared), deque()
        for _ in range(2):
            item = next(remaining, None)
            if item is not None:
                pending.append((item, pool.submit(prepare, item)))
        while pending:
            original, future = pending.popleft()
            item = next(remaining, None)
            if item is not None:
                pending.append((item, pool.submit(prepare, item)))
            try:
                yield future.result(), None
            except Exception as exc:
                yield original, exc


def _reuse_publication(output, directory, fingerprint):
    path = directory / "execution.json"
    if not path.is_file():
        raise ValueError("Published output has no bound execution receipt")
    receipt = json.loads(path.read_text())
    if (receipt.get("status") != "success" or receipt.get("run_fingerprint") != fingerprint
            or receipt.get("output", {}).get("name") != output.name
            or output.is_symlink() or file_digest(output) != receipt["output"].get("sha256")):
        raise ValueError("Published output or execution identity changed")
    graph = receipt.get("fact_graph")
    if graph:
        if (Path(graph["path"]).name != graph["path"]
                or file_digest(directory / graph["path"]) != graph["sha256"]):
            raise ValueError("Published fact graph changed")
    return receipt


def _run_bound(args, prepared, client, settings, identity, fingerprint, convert_doc, batch_started):
    args.result_dir.mkdir(parents=True, exist_ok=True)
    worker_count = identity["effective_document_workers"]
    spawn_documents = identity["document_worker_backend"] == "spawn-processes"
    batch_identity = getattr(args, "divide_training_batch_identity", None)
    path = args.work_dir / "summary.json"
    previous = json.loads(path.read_text()) if args.resume and path.exists() else {}
    invocations = previous.get("invocations", [])
    for old in invocations:
        if old.get("status") == "running":
            old.update(status="interrupted", timing_complete=False)
    current = {"id": uuid4().hex, "status": "running", "seconds": 0.0,
               "timing_complete": False, "preflight_calls": client.calls.copy()}
    invocations.append(current)
    outcomes, batch_error = {}, None

    def save():
        current["seconds"] = time.perf_counter() - batch_started
        total = sum(item["seconds"] for item in invocations)
        documents = [outcomes[name] for name, _, _ in prepared if name in outcomes]
        failures = [{"name": d["name"], "error": d["error"]}
                    for d in documents if d["status"] != "completed"]
        completed = sum(d["status"] == "completed" for d in documents)
        summary = {
            "workflow": args.workflow, "run_fingerprint": fingerprint,
            "reports": len(prepared), "failures": failures, "seconds": total,
            "completed_reports": completed, "document_workers": args.document_workers,
            "effective_document_workers": worker_count, "documents": documents,
            "isolate_documents": args.isolate_documents,
            "document_worker_backend": identity["document_worker_backend"],
            "document_seconds_sum": sum(d["seconds"] for d in documents if d["seconds"] is not None),
            "completed_reports_per_hour": completed * 3600 / total if total > 0 else 0,
            "timing_kind": "batch_wall_seconds_across_invocations",
            "timing_complete": all(item["timing_complete"] for item in invocations),
            "timing_scope": "batch wall time: preflight, worker startup, conversion, workflow/retries, rendering and publication",
            "invocations": invocations,
            "reused_reports": [d["name"] for d in documents if d.get("reused")],
            "submission_ready": False, "evaluation": "development; no platform score implied",
        }
        if batch_identity is not None:
            summary.update(input_scope=batch_identity["scope"], input_limitations=batch_identity["limitations"],
                           training_evidence_manifest_sha256=batch_identity["manifest_sha256"])
            if batch_error is not None:
                summary["batch_identity_error"] = batch_error
        atomic_json(path, summary)
        atomic_json(args.result_dir / "infer_time.json", {"infer_time": math.ceil(total)})

    def job_for(item):
        name, manifest, source = item
        return (name, manifest, source, args, settings, fingerprint,
                identity["sources"][str(_original_input(name, source, args).resolve())], convert_doc)

    def record_worker_failure(name, exc):
        outcomes[name] = {"name": name, "status": "failed",
                          "error": f"Worker failure: {type(exc).__name__}: {exc}", "seconds": None}
        print(f"FAILED {name}: {outcomes[name]['error']}", flush=True)

    save()
    try:
        if not spawn_documents:
            # Preserve bounded CPU prefetch on the original default path.
            # Explicit report budgets begin before conversion inside the worker.
            has_budget = (getattr(args, "divide_report_timeout", None) is not None
                          or getattr(args, "divide_max_model_calls", None) is not None)
            items = (((item, None) for item in prepared) if has_budget
                     else _prepared_reports(prepared, args, identity, convert_doc))
            for item, preparation_error in items:
                try:
                    if preparation_error is not None:
                        raise preparation_error
                    outcomes[item[0]] = _run_document(*job_for(item))
                except Exception as exc:
                    record_worker_failure(item[0], exc)
                finally:
                    save()
        else:
            # No SDK/session objects cross this boundary. Each process serves
            # exactly one report and acquires its own durable report lock.
            try:
                with ProcessPoolExecutor(max_workers=worker_count,
                                         mp_context=multiprocessing.get_context("spawn"),
                                         max_tasks_per_child=1) as pool:
                    futures = {pool.submit(_run_document, *job_for(item)): item[0] for item in prepared}
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            outcomes[name] = future.result()
                        except Exception as exc:
                            record_worker_failure(name, exc)
                        finally:
                            save()
            except Exception as exc:
                for name, _, _ in prepared:
                    if name not in outcomes:
                        record_worker_failure(name, exc)
        if batch_identity is not None:
            try:
                _verify_training_batch(args)
            except Exception as exc:
                batch_error = f"Batch input identity changed: {type(exc).__name__}: {exc}"
                for row in outcomes.values():
                    if row["status"] == "completed":
                        row.update(status="failed", completed_before_input_revalidation=True,
                                   error=batch_error)
                        atomic_json(args.work_dir / "documents" / row["name"] / "document_summary.json", row)
        current["status"] = "failed" if any(d["status"] != "completed" for d in outcomes.values()) else "completed"
    except BaseException:
        current["status"] = "interrupted"
        raise
    finally:
        current["timing_complete"] = True
        save()
    failures = [d for d in outcomes.values() if d["status"] != "completed"]
    if failures:
        raise RuntimeError(f"{len(failures)} divide document(s) failed; see work directory")
