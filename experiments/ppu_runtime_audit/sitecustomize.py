"""Opt-in, metadata-only SGLang W8A8 loading audit.

Enable only for the INT8 server by prepending this directory to PYTHONPATH and
setting CITYAGENT_INT8_AUDIT_PATH to an audit output directory. Each process
writes w8a8-load.<PID>.jsonl. Python's normal site import applies the hook to
spawned workers too. With the environment variable unset, no hook is installed.

There is no eager SGLang/Torch import, model loading, tensor copy, item(), CPU
transfer, or change to kernel selection/math. Records mean that the original
post-load method returned; consumers must still check counts, shapes and dtype.
Hook errors go to stderr and, when possible, an audit_error record. Errors while
auditing an actual W8A8 layer propagate instead of pretending auditing succeeded.
"""

import os as _os


_TARGET_MODULE = "sglang.srt.layers.quantization.w8a8_int8"
_PATCH_MARKER = "_cityagent_int8_load_audit_installed"


def _append_record(directory, record):
    import fcntl
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": _os.getpid(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd = _os.open(root / f"w8a8-load.{_os.getpid()}.jsonl", _os.O_WRONLY | _os.O_CREAT | _os.O_APPEND, 0o600)
    try:
        # Separate descriptors/threads in this PID cannot interleave JSON lines.
        fcntl.flock(fd, fcntl.LOCK_EX)
        remaining = memoryview(encoded)
        while remaining:
            written = _os.write(fd, remaining)
            if written <= 0:
                raise OSError("Audit JSONL write made no progress")
            remaining = remaining[written:]
    finally:
        _os.close(fd)


def _report_error(directory, stage, exc, prefix=None):
    import sys
    import traceback

    message = f"CITYAGENT_INT8_AUDIT_ERROR pid={_os.getpid()} stage={stage} prefix={prefix!r}: {type(exc).__name__}: {exc}"
    print(message, file=sys.stderr, flush=True)
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        _append_record(directory, {
            "event": "audit_error", "stage": stage, "prefix": prefix,
            "error_type": type(exc).__name__, "error": str(exc), "traceback": trace,
        })
    except Exception as write_error:
        print(f"CITYAGENT_INT8_AUDIT_ERROR cannot persist error record: {write_error}", file=sys.stderr, flush=True)


def _class_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _tensor_metadata(value):
    # dtype/shape are tensor metadata, not tensor computations or host transfers.
    return {"dtype": str(value.dtype), "shape": [int(dimension) for dimension in value.shape]}


def _patch_module(module, directory):
    import functools
    import inspect
    import weakref

    if getattr(module, _PATCH_MARKER, False):
        return
    config_class = module.W8A8Int8Config
    method_class = module.W8A8Int8LinearMethod
    original_get = config_class.get_quant_method
    original_process = method_class.process_weights_after_loading
    if list(inspect.signature(original_get).parameters) != ["self", "layer", "prefix"]:
        raise RuntimeError("Unexpected W8A8Int8Config.get_quant_method signature")
    if list(inspect.signature(original_process).parameters) != ["self", "layer"]:
        raise RuntimeError("Unexpected W8A8Int8LinearMethod.process_weights_after_loading signature")
    prefixes = weakref.WeakKeyDictionary()

    @functools.wraps(original_get)
    def get_quant_method(self, layer, prefix):
        try:
            method = original_get(self, layer, prefix)
            if isinstance(method, method_class):
                if not isinstance(prefix, str) or not prefix:
                    raise RuntimeError("Missing layer prefix for selected W8A8 method")
                prefixes[layer] = (prefix, id(method))
            return method
        except Exception as exc:
            _report_error(directory, "get_quant_method", exc, prefix)
            raise

    @functools.wraps(original_process)
    def process_weights_after_loading(self, layer):
        prefix = None
        try:
            association = prefixes.get(layer)
            prefix = association[0] if association is not None else None
        except Exception as exc:
            _report_error(directory, "lookup_layer_prefix", exc)
            raise
        try:
            # Preserve the original method and return value exactly. Log only
            # after it returns; a loader exception cannot produce a success row.
            result = original_process(self, layer)
        except Exception as exc:
            _report_error(directory, "original_process_weights_after_loading", exc, prefix)
            raise
        try:
            if association is None or association[1] != id(self):
                raise RuntimeError("W8A8 post-load method lacks its captured layer/prefix association")
            _append_record(directory, {
                "event": "weights_postprocessed",
                "prefix": prefix,
                "layer_class": _class_name(layer),
                "quant_method_class": _class_name(self),
                "weight": _tensor_metadata(layer.weight),
                "weight_scale": _tensor_metadata(layer.weight_scale),
            })
        except Exception as exc:
            _report_error(directory, "record_postprocessed_weights", exc, prefix)
            raise
        return result

    config_class.get_quant_method = get_quant_method
    method_class.process_weights_after_loading = process_weights_after_loading
    setattr(module, _PATCH_MARKER, True)


def _activate(directory):
    import importlib.abc
    import importlib.machinery
    from pathlib import Path
    import sys

    # Resolve once so spawned workers with changed cwd use the same directory.
    directory = str(Path(directory).expanduser().absolute())
    _os.environ["CITYAGENT_INT8_AUDIT_PATH"] = directory

    class AuditLoader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            create = getattr(self.original, "create_module", None)
            return create(spec) if create is not None else None

        def exec_module(self, module):
            self.original.exec_module(module)
            try:
                _patch_module(module, directory)
            except Exception as exc:
                _report_error(directory, "install_module_wrappers", exc)
                raise

        def __getattr__(self, name):
            return getattr(self.original, name)

    class AuditFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET_MODULE:
                return None
            # The installed target is a normal filesystem Python module. Do
            # not import it eagerly or intercept any other module/package.
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                exc = ImportError(f"Cannot locate {_TARGET_MODULE} for audit")
                _report_error(directory, "find_target_module", exc)
                raise exc
            if not hasattr(spec.loader, "exec_module"):
                exc = ImportError("Target module loader lacks exec_module")
                _report_error(directory, "wrap_target_loader", exc)
                raise exc
            spec.loader = AuditLoader(spec.loader)
            return spec

    if _TARGET_MODULE in sys.modules:
        _patch_module(sys.modules[_TARGET_MODULE], directory)
    else:
        sys.meta_path.insert(0, AuditFinder())


if _os.environ.get("CITYAGENT_INT8_AUDIT_PATH"):
    try:
        _activate(_os.environ["CITYAGENT_INT8_AUDIT_PATH"])
    except Exception as _error:
        # Python reports exceptions from sitecustomize but may continue startup;
        # explicitly identify the failure. No successful audit is implied.
        _report_error(_os.environ["CITYAGENT_INT8_AUDIT_PATH"], "sitecustomize_initialization", _error)
        raise
