#!/usr/bin/env python3
"""Offline, processor-only admission for the prepared one-image PPU probe.

No HTTP, engine construction, model weight loading or generation. The caller
owns idle checks and the whole-experiment deadline. Importing this module does
not import vLLM/torch; actual installed CPU execution remains a separate gate.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone

VERSION = "cityagent-own-vision-processor-admission-v3"
MODEL = "cityagent-qwen36-27b-epoch-002"
MAX_MODEL_LEN = 65536
MAX_OUTPUT = 512
LEGACY_VLLM_VERSION = "0.15.0+ppu2.0.0.post1"
MODERN_VLLM_VERSION = "0.19.1"
MODELINFO_CACHE_SHA256 = "67f48c551f2719d94d77a4e8a3f21d640c0bc20fa455b773575775b0a8c8f071"
MODELINFO_CACHE_BYTES = 760
MODELINFO_CACHE_FILENAME = "vllm-model_executor-models-qwen3_5-Qwen3_5ForConditionalGeneration.json"
# Public source has no bound historical deployment. These are deliberately
# unusable placeholders, not measured provenance or a valid admission receipt.
MODELINFO_ORIGIN = {
    "service_pid": 0,
    "service_cmdline_sha256": "0" * 64,
    "path": "/__cityagent_unconfigured_runtime__/cache/vllm/modelinfos/" + MODELINFO_CACHE_FILENAME,
    "base_model_root": "/__cityagent_unconfigured_runtime__/models/Qwen3.6-27B",
}
MODELINFO_SOURCE_ROOT = Path("/__cityagent_unconfigured_runtime__/venv/lib/python3.12/site-packages")
MODELINFO_SOURCE_PINS = {
    "vllm/model_executor/models/registry.py": "3efaece3f48dc76d8bd6bd6825830f778365f8b2976ef4d1b42de0c9d59620c3",
    "vllm/model_executor/models/qwen3_5.py": "5ce2151170c38c2a2718e3a9386598c0660d30db5ab8844fd284e08633ebc904",
    "vllm/utils/hashing.py": "89b4bd2bd246a5bfdcf00d766ea38a8182376aecfd6b6b7d5c47d96b1559243e",
}
CPU_METADATA_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
CPU_METADATA_PYTHON = Path("/__cityagent_unconfigured_runtime__/cache/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12")
CPU_METADATA_ALIAS_DIRECTORY = CPU_METADATA_PYTHON.parent.parent.parent / "cpython-3.12-linux-x86_64-gnu"
CPU_METADATA_ALIAS_DIRECTORY_TARGET = CPU_METADATA_PYTHON.parent.parent
CPU_METADATA_ARGV_PYTHON = CPU_METADATA_ALIAS_DIRECTORY / "bin/python3.12"
CPU_METADATA_PLATFORM = CPU_METADATA_PYTHON.parent.parent / "lib/python3.12/platform.py"
CPU_METADATA_CPUINFO = MODELINFO_SOURCE_ROOT / "cpuinfo/cpuinfo.py"
CPU_METADATA_BINARIES = {"file": Path("/usr/bin/file"), "uname": Path("/usr/bin/uname")}
CPU_METADATA_FILE_PINS = {
    str(CPU_METADATA_PLATFORM): "f5ff83f15c841792be42709a6710dc8d7a9f72791c1ae4cad719b1fb0abe0a05",
    str(CPU_METADATA_PLATFORM.parent / "subprocess.py"): "85d29b2bf0249f5436838298c9a60ee93508b1102e9ac43b001f8a7e7ae8f375",
    str(CPU_METADATA_CPUINFO): "1c7c8394350da2f13743327786f8a23359e0c8e0b8883ee2ea81a2cf68939959",
    str(MODELINFO_SOURCE_ROOT / "vllm/usage/usage_lib.py"): "f34d7cdbd3044672ceb2f229a0f6f9a65643b39b56aa8eadcd7e8a4deffd48c4",
    str(MODELINFO_SOURCE_ROOT / "vllm/config/parallel.py"): "308996eda88b1cd0904dc6918d3d2140dfa4620c8a80e73705cf209d443c4478",
    "/usr/bin/file": "ffa64f607f77d57cb3e2b650825868367a06eb91cdc5959c4e2f62ceb885b18a",
    "/usr/bin/uname": "37df0311d0e24169abfd166bc6018d40b87306f7ff64d9eec256c8331ac26347",
    str(CPU_METADATA_PYTHON): "0886fb76ceec8ff568d23079b5da6342bb41675e216e11525d4d6a9e65988d10",
    "/usr/lib/x86_64-linux-gnu/libmagic.so.1.0.0": "1a3f509b0f2a9604faa622ba41f9faeb066d7cc5725205d3276f2722e0d4b074",
    "/usr/lib/file/magic.mgc": "5d82f8f7172ab254d0241a7a558fe3a7a9d88cbb4bc6c8828c6bd48fd80cd228",
}
CV2_IMPORT_SOURCE_PINS = {
    "cv2/__init__.py": "69192904dead3ada37c8b305ee850ac7474924730881a9d5e0b2390944eb5aae",
    "cv2/config.py": "974e2d4096ee1a9a9a341df1bf33e16973683bf1ac733006de27ff2f23bc584d",
    "cv2/config-3.py": "9a7aadf724b822001f5e963b01fd4e375d45e2cbbe328d2ca7b42a440c083a1c",
    "cv2/load_config_py3.py": "9c22766454a9ac2f6869d37830b2efc699e7b8c3125484e9976c6e677c066029",
}
CV2_IMPORT_LD = str(MODELINFO_SOURCE_ROOT / "cv2/../../lib64") + ":"
CV2_IMPORT_ENV_KEYS = ("PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "MAGIC")


def require_signature(target, names, label):
    actual = tuple(inspect.signature(target).parameters)
    if actual != names:
        raise ValueError(f"Unsupported native {label} signature: {actual}")


def native_interface_profile(preprocessor_type, version):
    """Select only the two audited installed interfaces, before construction."""
    if version == LEGACY_VLLM_VERSION:
        require_signature(preprocessor_type,
                          ("model_config", "observability_config", "mm_registry", "mm_processor_cache"),
                          "PPU InputPreprocessor")
        require_signature(preprocessor_type.preprocess,
                          ("self", "prompt", "tokenization_kwargs", "mm_uuids"), "PPU preprocess")
        return "ppu-0.15.0-model-config"
    if version == MODERN_VLLM_VERSION:
        require_signature(preprocessor_type, ("vllm_config", "renderer", "mm_registry"),
                          "0.19.1 InputPreprocessor")
        require_signature(preprocessor_type.preprocess, ("self", "prompt", "tokenization_kwargs"),
                          "0.19.1 preprocess")
        return "vllm-0.19.1-config-chatparams"
    raise ValueError(f"Unsupported native processor version: {version}")


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def load_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"Non-finite JSON: {value}")))


def modelinfo_cache_binding(path, expected_sha256, audit, base_model):
    """Bind historical cache bytes and the separately audited current consumer.

    Cache origin is immutable provenance, not a required live service PID.
    main() verifies the consumer through installed_identity before cache use;
    NativeMultimodalAdmission independently validates its audited model/LoRA.
    """
    if (MODELINFO_ORIGIN["service_pid"] <= 0
            or MODELINFO_ORIGIN["service_cmdline_sha256"] == "0" * 64):
        raise ValueError("Historical modelinfo cache replay is unconfigured in public source")
    path = Path(path)
    service = audit["service_process"]
    if (audit["packages"]["vllm"] != MODERN_VLLM_VERSION
            or str(base_model) != MODELINFO_ORIGIN["base_model_root"]
            or expected_sha256 != MODELINFO_CACHE_SHA256
            or not path.is_absolute() or ".." in path.parts or path.is_symlink()
            or str(path) == MODELINFO_ORIGIN["path"] or not path.is_file()
            or path.stat().st_size != MODELINFO_CACHE_BYTES or file_digest(path) != MODELINFO_CACHE_SHA256):
        raise ValueError("Invalid pinned modern modelinfo cache/version/source binding")
    value = load_json(path)
    if (set(value) != {"hash", "modelinfo"}
            or value["hash"] != "3ababf511be73b387dc3d1ed7c923915"
            or value["modelinfo"].get("architecture") != "Qwen3_5ForConditionalGeneration"):
        raise ValueError("Invalid official modelinfo cache content")
    sources = []
    for relative, digest in MODELINFO_SOURCE_PINS.items():
        source = MODELINFO_SOURCE_ROOT / relative
        if source.is_symlink() or not source.is_file() or file_digest(source) != digest:
            raise ValueError("Pinned official modelinfo source changed")
        sources.append({"path": str(source), "sha256": digest, "bytes": source.stat().st_size})
    return {"schema_version": 1, "version": MODERN_VLLM_VERSION,
            "input_path": str(path), "sha256": MODELINFO_CACHE_SHA256, "bytes": MODELINFO_CACHE_BYTES,
            "origin": dict(MODELINFO_ORIGIN),
            "consumer_service": {"pid": service["pid"], "cmdline_sha256": service["cmdline_sha256"],
                                 "audit_service_sha256": hashlib.sha256(packed(service)).hexdigest()},
            "module_official_hash": value["hash"],
            "architecture": value["modelinfo"]["architecture"], "source_files": sources}


def verify_cpu_metadata_policy():
    """Pin the two observed stdlib metadata commands, never execute them here."""
    if (os.environ.get("PATH") != CPU_METADATA_PATH
            or os.environ.get("LD_LIBRARY_PATH") != ""
            or any(os.environ.get(key) for key in ("LD_PRELOAD", "LD_AUDIT", "MAGIC"))
            or Path(sys.executable).resolve(strict=True) != CPU_METADATA_PYTHON):
        raise ValueError("CPU metadata command environment or interpreter differs from the fixed source")
    rows = []
    for name, expected in CPU_METADATA_FILE_PINS.items():
        path = Path(name)
        before = path.stat()
        if path.is_symlink() or not path.is_file() or file_digest(path) != expected:
            raise ValueError("Pinned CPU metadata source/binary changed: " + name)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("CPU metadata file changed while hashing: " + name)
        rows.append({"path": name, "sha256": expected, "bytes": after.st_size})
    for name, expected in CPU_METADATA_BINARIES.items():
        resolved = shutil.which(name, path=CPU_METADATA_PATH)
        if resolved != str(expected) or Path(resolved).resolve(strict=True) != expected:
            raise ValueError("CPU metadata PATH resolves a different executable")
    if (not CPU_METADATA_ALIAS_DIRECTORY.is_symlink()
            or os.readlink(CPU_METADATA_ALIAS_DIRECTORY) != str(CPU_METADATA_ALIAS_DIRECTORY_TARGET)
            or CPU_METADATA_ARGV_PYTHON.is_symlink()
            or CPU_METADATA_ARGV_PYTHON.resolve(strict=True) != CPU_METADATA_PYTHON
            or file_digest(CPU_METADATA_ARGV_PYTHON) != CPU_METADATA_FILE_PINS[str(CPU_METADATA_PYTHON)]):
        raise ValueError("Observed CPU metadata Python alias changed")
    return rows


def _cv2_env_value(key, value):
    if key in {"PATH", "LD_LIBRARY_PATH"}:
        return value
    return {"present": value is not None, "nonempty": bool(value),
            "sha256": None if value is None else digest(value.encode())}


def _cv2_import_snapshot():
    return {"environment": {key: _cv2_env_value(key, os.environ.get(key)) for key in CV2_IMPORT_ENV_KEYS},
            "sys_executable": sys.executable,
            "interpreter_realpath": str(Path(sys.executable).resolve(strict=True))}


def _verify_cv2_import_sources():
    if (MODELINFO_SOURCE_ROOT / "cv2/config-3.12.py").exists() or (
            MODELINFO_SOURCE_ROOT / "cv2/config-3.12.py").is_symlink():
        raise ValueError("Unobserved version-specific OpenCV import configuration")
    rows = []
    for relative, expected in CV2_IMPORT_SOURCE_PINS.items():
        path = MODELINFO_SOURCE_ROOT / relative
        if path.is_symlink() or not path.is_file() or file_digest(path) != expected:
            raise ValueError("Pinned OpenCV import source changed: " + relative)
        rows.append({"path": str(path), "sha256": expected, "bytes": path.stat().st_size})
    return rows


def _begin_cv2_import_environment(receipt):
    """Observe one modern cached import window; never relax metadata command policy."""
    receipt.update(schema_version=1, status="checking_initial_policy", before=_cv2_import_snapshot(),
                   restricted_mutations=[], restored=False)
    state = {"receipt": receipt, "active": False, "policy_files": verify_cpu_metadata_policy()}
    receipt["source_files"] = _verify_cv2_import_sources()
    state["active"] = True

    def observe(event, args):
        if not state["active"] or event not in {"os.putenv", "os.unsetenv"}:
            return
        key = os.fsdecode(args[0])
        if key not in CV2_IMPORT_ENV_KEYS:
            return
        value = os.fsdecode(args[1]) if event == "os.putenv" else None
        origin = None
        frame = sys._getframe(1)
        try:
            while frame is not None:
                if (frame.f_code.co_filename == str(MODELINFO_SOURCE_ROOT / "cv2/__init__.py")
                        and frame.f_code.co_name == "bootstrap" and frame.f_lineno == 147):
                    origin = {"path": frame.f_code.co_filename, "function": "bootstrap", "line": 147}
                    break
                frame = frame.f_back
        finally:
            del frame
        receipt["restricted_mutations"].append({"event": event, "key": key,
            "before": _cv2_env_value(key, os.environ.get(key)),
            "requested_after": _cv2_env_value(key, value), "origin": origin})

    sys.addaudithook(observe)
    receipt["status"] = "observing_imports"
    return state


def _finish_cv2_import_environment(state):
    state["active"] = False
    receipt = state["receipt"]
    receipt.update(status="checking_imported_environment", observed=_cv2_import_snapshot())
    before, observed = receipt["before"], receipt["observed"]
    if (observed["sys_executable"] != before["sys_executable"]
            or observed["interpreter_realpath"] != before["interpreter_realpath"]
            or any(observed["environment"][key] != before["environment"][key]
                   for key in CV2_IMPORT_ENV_KEYS if key != "LD_LIBRARY_PATH")):
        raise ValueError("Unrelated controlled environment/interpreter change during native imports")
    if _verify_cv2_import_sources() != receipt["source_files"]:
        raise ValueError("OpenCV import source identity changed during imports")
    changes = receipt["restricted_mutations"]
    observed_ld = observed["environment"]["LD_LIBRARY_PATH"]
    if observed_ld == before["environment"]["LD_LIBRARY_PATH"] and not changes:
        receipt["status"] = "no_change"
    else:
        expected = {"event": "os.putenv", "key": "LD_LIBRARY_PATH", "before": "",
                    "requested_after": CV2_IMPORT_LD,
                    "origin": {"path": str(MODELINFO_SOURCE_ROOT / "cv2/__init__.py"),
                               "function": "bootstrap", "line": 147}}
        cv2 = sys.modules.get("cv2")
        if (before["environment"]["LD_LIBRARY_PATH"] != "" or observed_ld != CV2_IMPORT_LD
                or changes != [expected] or getattr(cv2, "__file__", None) != str(
                    MODELINFO_SOURCE_ROOT / "cv2/__init__.py")):
            raise ValueError("Unknown OpenCV import environment mutation; no restoration performed")
        os.environ["LD_LIBRARY_PATH"] = before["environment"]["LD_LIBRARY_PATH"]
        receipt.update(status="restored_pinned_opencv_ld", restored=True)
    receipt["after"] = _cv2_import_snapshot()
    if verify_cpu_metadata_policy() != state["policy_files"] or receipt["after"] != before:
        raise ValueError("Original CPU metadata environment/identity not restored")
    receipt["original_policy_reverified"] = True


def _cpu_metadata_callsite(command):
    expected_function = "_syscmd_file" if command == "file" else "from_subprocess"
    platform_frame = cpuinfo_frame = False
    frame = sys._getframe(1)
    try:
        while frame is not None:
            code = frame.f_code
            platform_frame |= code.co_filename == str(CPU_METADATA_PLATFORM) and code.co_name == expected_function
            cpuinfo_frame |= code.co_filename == str(CPU_METADATA_CPUINFO) and code.co_name == "DataSource"
            frame = frame.f_back
    finally:
        del frame
    return platform_frame and cpuinfo_frame


def cpu_metadata_spawn_receipt(event, args):
    """Recognize only source-pinned cpuinfo import metadata; not a generic spawn permission."""
    if event != "subprocess.Popen" or not isinstance(args, tuple) or len(args) != 4:
        return None
    executable, argv, cwd, environment = args
    if type(executable) is not str or type(argv) is not list or cwd is not None:
        return None
    if executable == "file" and argv == ["file", "-b", str(CPU_METADATA_ARGV_PYTHON)]:
        if type(environment) is not dict or environment != dict(os.environ, LC_ALL="C"):
            return None
    elif executable == "uname" and argv == ["uname", "-p"]:
        if environment is not None:
            return None
    else:
        return None
    if not _cpu_metadata_callsite(executable):
        return None
    verify_cpu_metadata_policy()
    return {"event": event, "action": "allowed_metadata_spawn", "command": executable,
            "argv": list(argv), "cwd": None,
            "environment": "current_environment_with_LC_ALL_C" if executable == "file" else "inherited",
            "binary_path": str(CPU_METADATA_BINARIES[executable]),
            "binary_sha256": CPU_METADATA_FILE_PINS[str(CPU_METADATA_BINARIES[executable])],
            "target_python_argv_path": str(CPU_METADATA_ARGV_PYTHON) if executable == "file" else None,
            "target_python_realpath": str(CPU_METADATA_PYTHON),
            "target_python_sha256": CPU_METADATA_FILE_PINS[str(CPU_METADATA_PYTHON)],
            "callsite": "pinned_cpuinfo_DataSource_and_stdlib_platform",
            "permission_only_not_process_completion": True}


def cpu_metadata_spawn_guard(events):
    def guard(event, args):
        if event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork", "os.forkpty"}:
            receipt = cpu_metadata_spawn_receipt(event, args)
            if receipt is None:
                raise RuntimeError("Pinned-cache CPU path forbids subprocess fallback")
            events.append(receipt)
    return guard


def seed_modelinfo_cache(output, binding, metadata_events=None):
    """Copy original bytes into this run's cache, retaining the official registry path."""
    target_root = Path(output) / "cache/vllm"
    if os.environ.get("VLLM_CACHE_ROOT") != str(target_root):
        raise ValueError("Modelinfo cache destination is not this run's isolated cache")

    verify_cpu_metadata_policy()
    sys.addaudithook(cpu_metadata_spawn_guard(metadata_events if metadata_events is not None else []))
    from vllm.utils.hashing import safe_hash
    if Path(inspect.getsourcefile(safe_hash)) != MODELINFO_SOURCE_ROOT / "vllm/utils/hashing.py":
        raise ValueError("Official hash function was not imported from the pinned installed source")
    module = MODELINFO_SOURCE_ROOT / "vllm/model_executor/models/qwen3_5.py"
    if safe_hash(module.read_bytes(), usedforsecurity=False).hexdigest() != binding["module_official_hash"]:
        raise ValueError("Official architecture module hash differs from pinned cache")
    destination = target_root / "modelinfos" / MODELINFO_CACHE_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=False)
    with destination.open("xb") as stream:
        stream.write(Path(binding["input_path"]).read_bytes())
    if file_digest(destination) != binding["sha256"]:
        raise ValueError("Modelinfo cache copy changed original bytes")
    return {"binding": binding, "isolated_path": str(destination), "sha256": file_digest(destination),
            "official_module_hash_verified": True, "subprocess_fallback_allowed": False}


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_bytes(packed(value) + b"\n")
    tmp.replace(path)


def validate_request(request, source, image):
    """This narrow helper accepts exactly the prepared one-image request."""
    required = {"model", "messages", "max_tokens", "temperature", "stream",
                "n", "chat_template_kwargs"}
    if not isinstance(request, dict) or set(request) != required:
        raise ValueError("Unsupported final request fields")
    if (type(request["max_tokens"]) is not int or request["max_tokens"] != MAX_OUTPUT
            or type(request["n"]) is not int or request["n"] != 1
            or type(request["temperature"]) not in (int, float)
            or request["temperature"] != 0 or request["stream"] is not False
            or request["chat_template_kwargs"] != {"enable_thinking": False}
            or request["chat_template_kwargs"].get("enable_thinking") is not False
            or not isinstance(request["model"], str) or not request["model"]):
        raise ValueError("Unsupported sampling/template settings")
    messages = request["messages"]
    if (not isinstance(messages, list) or len(messages) != 1
            or not isinstance(messages[0], dict)
            or set(messages[0]) != {"role", "content"}
            or messages[0]["role"] != "user"):
        raise ValueError("Expected one user message")
    content = messages[0]["content"]
    if (not isinstance(content, list) or len(content) != 2
            or not all(isinstance(item, dict) for item in content)
            or set(content[0]) != {"type", "text"}
            or content[0]["type"] != "text"
            or not isinstance(content[0]["text"], str) or not content[0]["text"].strip()
            or set(content[1]) != {"type", "image_url"}
            or content[1]["type"] != "image_url"
            or not isinstance(content[1]["image_url"], dict)
            or set(content[1]["image_url"]) != {"url"}):
        raise ValueError("Expected text then one original image, without extra controls")
    url = content[1]["image_url"]["url"]
    prefix = "data:image/png;base64,"
    if not isinstance(url, str) or not url.startswith(prefix):
        raise ValueError("Only embedded PNG data is allowed; no remote URL")
    if base64.b64decode(url[len(prefix):], validate=True) != image:
        raise ValueError("Request image differs from original source bytes")
    if len(image) < 24 or image[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Expected original PNG")
    width, height = struct.unpack(">II", image[16:24])
    if (source.get("blob_sha256") != digest(image)
            or source.get("blob_bytes") != len(image)
            or (width, height) != (source.get("width"), source.get("height"))
            or source.get("mime_type") != "image/png"
            or source.get("raw_bytes_unchanged") is not True):
        raise ValueError("Source metadata does not bind original image")
    return {"request_sha256": digest(packed(request)),
            "messages_sha256": digest(packed(messages)),
            "source_image_sha256": digest(image), "image_bytes": len(image),
            "image_dimensions": [width, height]}


def validate_binding(value, base_model, request):
    keys = {"schema_version", "catalog_binding_source", "base_model_root", "base_model_id", "model",
            "adapter_path", "adapter_weights_sha256", "adapter_config_sha256"}
    if (not isinstance(value, dict) or set(value) != keys or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1 or value.get("catalog_binding_source") != "model_catalog.id+root+parent"
            or value.get("model") != MODEL or request["model"] != MODEL
            or value.get("base_model_root") != str(base_model)
            or not isinstance(value.get("base_model_id"), str) or not value["base_model_id"]
            or "\x00" in value["base_model_id"] or value["base_model_id"].startswith("--")
            or any(not isinstance(value.get(k), str) or re.fullmatch(r"[0-9a-f]{64}", value[k]) is None
                   for k in ("adapter_weights_sha256", "adapter_config_sha256"))):
        raise ValueError("Invalid catalog model binding")
    for key in ("base_model_root", "adapter_path"):
        name = value.get(key)
        if (not isinstance(name, str) or "\x00" in name or not Path(name).is_absolute()
                or str(Path(name)) != name or ".." in Path(name).parts
                or Path(name).is_symlink() or not Path(name).is_dir()):
            raise ValueError("Invalid catalog model binding path")
    return dict(value)


def validate_lora_mapping(argv, binding):
    if argv.count("--enable-lora") != 1 or argv.count("--lora-modules") != 1:
        raise ValueError("Missing or ambiguous audited LoRA module binding")
    modules = []
    for item in argv[argv.index("--lora-modules") + 1:]:
        if item.startswith("--"):
            break
        if item.startswith("{"):
            module = json.loads(item, object_pairs_hook=unique_object)
            if (not isinstance(module, dict) or not {"name", "path"} <= set(module)
                    or not set(module) <= {"name", "path", "base_model_name"}):
                raise ValueError("Invalid audited LoRA module")
        else:
            name, path = item.split("=", 1)
            module = {"name": name, "path": path}
        if any(not isinstance(module.get(k), str) or not module[k] for k in ("name", "path")):
            raise ValueError("Invalid audited LoRA module")
        modules.append(module)
    own = [module for module in modules if module["name"] == binding["model"]]
    if (len(own) != 1 or own[0]["path"] != binding["adapter_path"]
            or ("base_model_name" in own[0] and own[0]["base_model_name"] != binding["base_model_id"])):
        raise ValueError("Audited LoRA module differs from catalog binding")


def server_policy_from_argv(argv, base_model):
    protected = ("--model", "--served-model-name", "--max-model-len", "--dtype", "--enable-lora", "--lora-modules")
    if any(item.startswith(tuple(flag + "=" for flag in protected)) for item in argv):
        raise ValueError("Invalid portable server policy")
    if len(argv) >= 3 and argv[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]:
        if not Path(argv[0]).is_absolute() or "serve" in argv or argv.count("-m") != 1:
            raise ValueError("Invalid portable server policy")
        entrypoint, model_flag = "vllm_python_module_api_server", "--model"
    else:
        if argv.count("serve") != 1 or "-m" in argv or "--model" in argv:
            raise ValueError("Invalid portable server policy")
        index = argv.index("serve")
        if index not in (1, 2) or Path(argv[index - 1]).name != "vllm":
            raise ValueError("Invalid portable server policy")
        entrypoint, model_flag = "vllm_serve", "serve"
    if (argv.count(model_flag) != 1 or argv.index(model_flag) + 1 >= len(argv)
            or argv[argv.index(model_flag) + 1] != str(base_model)):
        raise ValueError("Invalid portable server policy")
    trust_flags = [item for item in argv if item.startswith(("--trust-remote-code", "--no-trust-remote-code"))]
    if trust_flags not in ([], ["--trust-remote-code"]):
        raise ValueError("Invalid portable server policy")
    if trust_flags:
        index = argv.index("--trust-remote-code")
        if index + 1 < len(argv) and argv[index + 1].lower() in {"true", "false", "0", "1"}:
            raise ValueError("Invalid portable server policy")
    return {"schema_version": 1, "entrypoint": entrypoint, "trust_remote_code": bool(trust_flags)}


def installed_identity(audit, base_model, request, model_binding=None, server_policy=None):
    service = audit["service_process"]
    cmdline = Path(f"/proc/{int(service['pid'])}/cmdline").read_bytes()
    cmd_sha = digest(cmdline)
    if cmd_sha != service["cmdline_sha256"]:
        raise ValueError("Service command line changed since source audit")
    argv = [value.decode() for value in cmdline.split(b"\0") if value]
    if argv != [value for value in service["cmdline"] if value]:
        raise ValueError("Audited service argv does not match raw process identity")
    binding = validate_binding(model_binding, base_model, request) if model_binding is not None else None
    base_alias = binding["base_model_id"] if binding else request["model"]
    if binding is not None:
        actual_policy = server_policy_from_argv(argv, base_model)
        if (not isinstance(server_policy, dict) or set(server_policy) != {"schema_version", "entrypoint", "trust_remote_code"}
                or type(server_policy.get("schema_version")) is not int
                or type(server_policy.get("trust_remote_code")) is not bool or server_policy != actual_policy):
            raise ValueError("Bound server policy differs from actual process")
        for flag, value in (("--served-model-name", base_alias), ("--max-model-len", str(MAX_MODEL_LEN)),
                            ("--dtype", "bfloat16")):
            if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv) or argv[argv.index(flag) + 1] != value:
                raise ValueError("Unsupported or ambiguous model/server settings")
        validate_lora_mapping(argv, binding)
        # CPU-only metadata check. Adapter weights are checked by the parent;
        # neither this check nor the native processor may open parameter shards.
        config = Path(binding["adapter_path"]) / "adapter_config.json"
        if config.is_symlink() or file_digest(config) != binding["adapter_config_sha256"]:
            raise ValueError("Adapter config differs from catalog-bound local identity")
    else:
        if server_policy is not None:
            raise ValueError("Server policy requires explicit portable model binding")
        if (argv[argv.index("serve") + 1] != str(base_model)
                or argv[argv.index("--served-model-name") + 1] != base_alias
                or argv[argv.index("--max-model-len") + 1] != str(MAX_MODEL_LEN)
                or argv[argv.index("--dtype") + 1] != "bfloat16"
                or "--trust-remote-code" not in argv):
            raise ValueError("Unsupported model/server settings")
    versions = {name: importlib.metadata.version(name) for name in audit["packages"]}
    if versions != audit["packages"]:
        raise ValueError("Installed package versions differ from source audit")
    files = []
    for entry in audit["source_index"]:
        path = Path(entry["path"])
        actual = file_digest(path)
        if actual != entry["sha256"]:
            raise ValueError(f"Audited processor/source/config changed: {path}")
        files.append({"path": str(path), "sha256": actual, "bytes": path.stat().st_size})
    # Only tokenizer/processor/config files: never model parameter shards.
    additional = []
    for name in ("config.json", "generation_config.json", "preprocessor_config.json",
                 "video_preprocessor_config.json", "processor_config.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja", "added_tokens.json",
                 "special_tokens_map.json", "vocab.json", "merges.txt"):
        path = base_model / name
        additional.append({"path": str(path), "exists": path.is_file(),
                           "sha256": file_digest(path) if path.is_file() else None})
    identity = {"service_pid": service["pid"], "service_argv": argv,
            "service_cmdline_sha256": cmd_sha, "package_versions": versions,
            "verified_audited_files": files, "model_metadata_files": additional}
    if binding is not None:
        identity["model_binding"] = binding
        identity["server_policy"] = dict(server_policy)
    return identity


def configure_offline(output):
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_USE_MODELSCOPE": "0",
                       "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false"})
    for key, subdir in {"HF_HOME": "hf", "HF_HUB_CACHE": "hf/hub",
                        "VLLM_CACHE_ROOT": "vllm", "XDG_CACHE_HOME": "xdg",
                        "TORCH_EXTENSIONS_DIR": "torch_extensions"}.items():
        os.environ[key] = str(output / "cache" / subdir)
    sys.dont_write_bytecode = True

    def guard(event, args):
        if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
            raise RuntimeError("Offline processor forbids DNS")
        if event in {"socket.connect", "socket.sendto"}:
            address = args[-1]
            if isinstance(address, tuple):
                raise RuntimeError("Offline processor forbids IP network access")
        if event == "open" and isinstance(args[0], (str, bytes)):
            name = os.fsdecode(args[0]).lower()
            if name.endswith((".safetensors", ".pt", ".pth", ".gguf")) or (
                    name.endswith(".bin") and ("/model/" in name or "pytorch_model" in name)):
                raise RuntimeError("Processor admission forbids model weight access")
    sys.addaudithook(guard)


def token_ids(value, label):
    if not isinstance(value, list) or not value or any(type(v) is not int or v < 0 for v in value):
        raise ValueError(f"Invalid {label} token IDs")
    return value


async def expanded_pipeline(preprocessor, request, interface="ppu-0.15.0-model-config"):
    """Use installed renderer plus the generation path's InputPreprocessor."""
    if interface == "vllm-0.19.1-config-chatparams":
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        require_signature(preprocessor.renderer.render_messages_async, ("messages", "params"),
                          "0.19.1 renderer")
        # Use the serving protocol's own defaults and ChatParams builder. Do not
        # invent template kwargs or use an OpenAI serving/engine object.
        chat_request = ChatCompletionRequest(**request)
        require_signature(chat_request.build_chat_params,
                          ("default_template", "default_template_content_format"), "0.19.1 ChatParams builder")
        params = chat_request.build_chat_params(None, "auto")
        _, prompt = await preprocessor.renderer.render_messages_async(request["messages"], params)
    elif interface == "ppu-0.15.0-model-config":
        require_signature(preprocessor.renderer.render_messages_async,
                          ("messages", "chat_template_content_format", "kwargs"), "PPU renderer")
        _, prompt = await preprocessor.renderer.render_messages_async(
            request["messages"], chat_template_content_format="auto", tokenize=False,
            chat_template=None, add_generation_prompt=True, continue_final_message=False,
            tools=None, documents=None, reasoning_effort=None,
            **request["chat_template_kwargs"])
    else:
        raise ValueError("Unsupported native pipeline interface")
    prompt = dict(prompt)
    if "prompt_token_ids" not in prompt:
        # Same underlying tokenizer and add_special_tokens=False as serving.py's
        # asynchronous tokenizer wrapper; no string-length/token-ratio estimate.
        encoded = preprocessor.renderer.get_tokenizer()(prompt["prompt"], add_special_tokens=False)
        prompt["prompt_token_ids"] = encoded.input_ids
    original = list(token_ids(prompt["prompt_token_ids"], "original template"))
    if set(prompt.get("multi_modal_data", {})) != {"image"}:
        raise ValueError("Renderer did not produce exactly image multimodal data")
    if interface == "vllm-0.19.1-config-chatparams":
        require_signature(preprocessor.renderer.get_mm_processor, (), "0.19.1 MM accessor")
        mm_processor = preprocessor.renderer.get_mm_processor()
        require_signature(mm_processor.apply, ("inputs", "timing_ctx"), "0.19.1 MM apply")
    else:
        require_signature(preprocessor._get_mm_processor, (), "PPU MM accessor")
        mm_processor = preprocessor._get_mm_processor()
    real_apply = mm_processor.apply
    applied = []

    def audited_apply(*args, **kwargs):
        result = real_apply(*args, **kwargs)
        applied.append(list(token_ids(result["prompt_token_ids"], "processor apply")))
        return result

    mm_processor.apply = audited_apply
    try:
        if interface == "vllm-0.19.1-config-chatparams":
            # 0.19.1 consumes multi_modal_uuids from the prompt through its
            # renderer. The older external mm_uuids keyword no longer exists.
            result = preprocessor.preprocess(prompt, tokenization_kwargs={})
        else:
            result = preprocessor.preprocess(prompt, tokenization_kwargs={},
                                             mm_uuids=prompt.get("multi_modal_uuids"))
    finally:
        mm_processor.apply = real_apply
    final = list(token_ids(result.get("prompt_token_ids"), "expanded"))
    if len(applied) != 1 or applied[0] != final or result.get("type") != "multimodal":
        raise ValueError("Native multimodal processor apply was not the final token source")
    placeholders = result.get("mm_placeholders", {})
    if set(placeholders) != {"image"} or len(placeholders["image"]) != 1:
        raise ValueError("Expected exactly one processed image placeholder")
    ranges = []
    for item in placeholders["image"]:
        offset, length = item.offset, item.length
        if type(offset) is not int or type(length) is not int or offset < 0 or length <= 1 or offset + length > len(final):
            raise ValueError("Invalid/unexpanded image placeholder range")
        ranges.append({"offset": offset, "length": length})
    if len(final) <= len(original):
        raise ValueError("Image placeholder was not expanded beyond template token count")
    return original, final, result, mm_processor, ranges


def object_source(obj):
    cls = type(obj)
    path = inspect.getsourcefile(cls)
    return {"class": cls.__module__ + "." + cls.__qualname__, "source_file": path,
            "source_sha256": file_digest(path) if path else None}


def cpu_tensor_metadata(value, torch, path="mm_kwargs"):
    """Record small shapes, never pixel arrays; refuse accelerator tensors."""
    found = []
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"Processor allocated non-CPU tensor at {path}")
        row = {"path": path, "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        if path.endswith("image_grid_thw.data"):
            row["values"] = value.tolist()
        found.append(row)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(cpu_tensor_metadata(item, torch, path + "." + str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(cpu_tensor_metadata(item, torch, path + f"[{index}]"))
    elif hasattr(value, "data"):
        found.extend(cpu_tensor_metadata(value.data, torch, path + ".data"))
    return found


def execute_cpu(request, base_model, base_model_id=None, trust_remote_code=True, *,
                restore_pinned_cv2_ld=False, import_environment_receipt=None):
    if (type(restore_pinned_cv2_ld) is not bool or import_environment_receipt is not None
            and (not restore_pinned_cv2_ld or type(import_environment_receipt) is not dict)):
        raise ValueError("Invalid internal import environment restoration controls")
    state = None
    if restore_pinned_cv2_ld:
        if importlib.metadata.version("vllm") != MODERN_VLLM_VERSION:
            raise ValueError("OpenCV import restoration is only for the pinned modern cache route")
        receipt = import_environment_receipt if import_environment_receipt is not None else {}
        state = _begin_cv2_import_environment(receipt)
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_default_device("cpu")
        from vllm.config import ModelConfig
        from vllm.inputs.preprocess import InputPreprocessor
    except BaseException:
        if state is not None:
            state["active"] = False
            state["receipt"].update(status="import_failed_no_restoration", observed=_cv2_import_snapshot())
        raise
    if state is not None:
        _finish_cv2_import_environment(state)
    version = importlib.metadata.version("vllm")
    interface = native_interface_profile(InputPreprocessor, version)

    # No EngineArgs/create_engine/LLM/model factory/weight loader is called.
    with torch.device("cpu"):
        config = ModelConfig(model=str(base_model), tokenizer=str(base_model),
                             served_model_name=base_model_id if base_model_id is not None else request["model"],
                             trust_remote_code=trust_remote_code,
                             dtype="bfloat16", max_model_len=MAX_MODEL_LEN)
        if config.max_model_len != MAX_MODEL_LEN or not config.is_multimodal_model:
            raise ValueError("Native config is not the expected multimodal context")
        if interface == "vllm-0.19.1-config-chatparams":
            from vllm.config import DeviceConfig, VllmConfig
            native_config = VllmConfig(model_config=config, device_config=DeviceConfig(device="cpu"))
            if native_config.device_config.device.type != "cpu" or native_config.model_config is not config:
                raise ValueError("Native processor config is not the bound CPU model config")
            preprocessor = InputPreprocessor(native_config)
        else:
            preprocessor = InputPreprocessor(config)
        original, final, result, processor, ranges = asyncio.run(expanded_pipeline(preprocessor, request, interface))
        tensors = cpu_tensor_metadata(result["mm_kwargs"], torch)
        if not tensors:
            raise ValueError("No CPU multimodal processor tensors found")
        identities = {"renderer": object_source(preprocessor.renderer),
                      "tokenizer": object_source(preprocessor.renderer.get_tokenizer()),
                      "input_preprocessor": object_source(preprocessor),
                      "multimodal_processor": object_source(processor),
                      "hf_processor": object_source(processor.info.get_hf_processor()),
                      "image_processor": object_source(processor.info.get_image_processor())}
    if identities["multimodal_processor"]["class"] != "vllm.model_executor.models.qwen3_vl.Qwen3VLMultiModalProcessor":
        raise ValueError("Unexpected native processor implementation")
    native = {"native_objects": identities, "processor_apply_calls": 1,
                             "processor_interface": {"version": version, "profile": interface,
                                                     "fallback": False, "device": "cpu"},
                             "image_placeholder_ranges": ranges, "cpu_tensors": tensors,
              "torch_version": torch.__version__, "mm_hashes": result.get("mm_hashes")}
    if state is not None:
        native["import_environment_restoration"] = state["receipt"]
    return original, final, native


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=root / "source.json")
    parser.add_argument("--image", type=Path, default=root / "source-image.png")
    parser.add_argument("--path-audit", type=Path, default=root / "tokenize-image-path-audit.json")
    parser.add_argument("--base-model", type=Path, default=Path("/model/Qwen3.6-27B"))
    parser.add_argument("--model-binding", type=Path)
    parser.add_argument("--server-policy", type=Path)
    parser.add_argument("--modelinfo-cache", type=Path)
    parser.add_argument("--modelinfo-cache-sha256")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    summary = {"schema_version": 1, "helper": VERSION, "status": "failed",
               "started_at": datetime.now(timezone.utc).isoformat(),
               "expanded_prompt_tokens": None, "max_output_tokens": MAX_OUTPUT,
               "max_model_len": MAX_MODEL_LEN, "fits_context": False,
               "server_usage_equivalence_verified": False, "generation_calls": 0,
               "request_sha256": None, "source_image_sha256": None,
               "service_cmdline_sha256": None,
               "original_template_prompt_token_ids_sha256": None,
               "final_prompt_token_ids_sha256": None,
               "limitations": ["CPU processor execution alone does not verify generation usage or semantic image understanding",
                               "No model weights are opened; runtime LoRA weight application is not checked",
                               "Caller must enforce idle state and remaining whole-experiment deadline"]}
    try:
        request, source, audit = load_json(args.request), load_json(args.source), load_json(args.path_audit)
        binding = load_json(args.model_binding) if args.model_binding is not None else None
        policy = load_json(args.server_policy) if args.server_policy is not None else None
        if (args.model_binding is None) != (args.server_policy is None):
            raise ValueError("Portable model binding and audited server policy must be provided together")
        image = args.image.read_bytes()
        summary.update(validate_request(request, source, image))
        bound_inputs = {str(path): file_digest(path) for path in
                        (args.request, args.source, args.image, args.path_audit, Path(__file__))}
        if args.model_binding is not None:
            bound_inputs[str(args.model_binding)] = file_digest(args.model_binding)
            bound_inputs[str(args.server_policy)] = file_digest(args.server_policy)
        # Read the current process/model/LoRA/policy identity independently of
        # the historical process that originally produced the official cache.
        identity = installed_identity(audit, args.base_model, request, binding, policy)
        summary["service_cmdline_sha256"] = identity["service_cmdline_sha256"]
        summary["identity"] = identity
        cache_binding = None
        if (args.modelinfo_cache is None) != (args.modelinfo_cache_sha256 is None):
            raise ValueError("Explicit cache input and fixed SHA must be provided together")
        if args.modelinfo_cache is not None:
            if binding is None:
                raise ValueError("Pinned modern cache requires explicit portable model binding")
            cache_binding = modelinfo_cache_binding(args.modelinfo_cache, args.modelinfo_cache_sha256, audit, args.base_model)
            bound_inputs[str(args.modelinfo_cache)] = cache_binding["sha256"]
            bound_inputs.update({row["path"]: row["sha256"] for row in cache_binding["source_files"]})
        summary["bound_input_files"] = bound_inputs
        configure_offline(args.output)
        if cache_binding is not None:
            summary["cpu_metadata_policy_files"] = verify_cpu_metadata_policy()
            summary["allowed_metadata_commands"] = []
            summary["modelinfo_cache"] = seed_modelinfo_cache(args.output, cache_binding, summary["allowed_metadata_commands"])
        if binding is None:
            original, final, native = execute_cpu(request, args.base_model)
        elif cache_binding is not None:
            summary["import_environment_restoration"] = {}
            original, final, native = execute_cpu(request, args.base_model, binding["base_model_id"],
                policy["trust_remote_code"], restore_pinned_cv2_ld=True,
                import_environment_receipt=summary["import_environment_restoration"])
        else:
            original, final, native = execute_cpu(request, args.base_model, binding["base_model_id"], policy["trust_remote_code"])
        if installed_identity(audit, args.base_model, request, binding, policy) != identity:
            raise ValueError("Service/processor/config identity changed during CPU stage")
        if any(file_digest(path) != sha for path, sha in bound_inputs.items()):
            raise ValueError("Bound request/source/audit/helper changed during CPU stage")
        if cache_binding is not None:
            if verify_cpu_metadata_policy() != summary["cpu_metadata_policy_files"]:
                raise ValueError("CPU metadata policy files changed during CPU stage")
            if (modelinfo_cache_binding(args.modelinfo_cache, args.modelinfo_cache_sha256, audit, args.base_model) != cache_binding
                    or file_digest(summary["modelinfo_cache"]["isolated_path"]) != cache_binding["sha256"]):
                raise ValueError("Bound official cache/source changed during CPU stage")
        save(args.output / "original-template-prompt-token-ids.json", original)
        save(args.output / "expanded-prompt-token-ids.json", final)
        summary.update(native)
        summary.update({"status": "processor_counted", "original_template_prompt_tokens": len(original),
                        "expanded_prompt_tokens": len(final), "fits_context": len(final) + MAX_OUTPUT <= MAX_MODEL_LEN,
                        "original_template_prompt_token_ids_sha256": digest(packed(original)),
                        "final_prompt_token_ids_sha256": digest(packed(final)),
                        "method": "installed_vllm_renderer_and_InputPreprocessor.preprocess_to_mm_processor.apply"})
    except Exception as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        (args.output / "error.txt").write_text(traceback.format_exc())
    finally:
        summary["wall_seconds"] = time.monotonic() - start
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "expanded_prompt_tokens", "fits_context", "wall_seconds")}))
    return 0 if summary["status"] == "processor_counted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
