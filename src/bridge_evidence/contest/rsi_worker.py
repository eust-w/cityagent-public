"""Trusted single-use worker. Loaded only inside the prepared Linux sandbox."""


def _install_stdlib_map(path="/stdlib.marshal"):
    """Load only the parent-attested trusted runtime, before candidate execution."""
    import _frozen_importlib
    import _frozen_importlib_external
    import marshal
    import sys

    with open(path, "rb") as source:
        encoded = source.read(64 * 1024 * 1024 + 1)
    if len(encoded) > 64 * 1024 * 1024:
        raise ValueError("trusted stdlib code map exceeds limit")
    payload = marshal.loads(encoded)
    del encoded
    expected = {
        "schema_version": 1,
        "python_version": sys.version,
        "cache_tag": sys.implementation.cache_tag,
        "marshal_version": marshal.version,
        "magic_number": _frozen_importlib_external.MAGIC_NUMBER,
    }
    if (
        type(payload) is not dict
        or set(payload) != {*expected, "modules"}
        or any(payload[key] != value for key, value in expected.items())
        or type(payload["modules"]) is not dict
        or not 0 < len(payload["modules"]) <= 10000
    ):
        raise ValueError("trusted stdlib identity mismatch")
    modules = payload["modules"]
    del payload
    code_type = type((lambda: None).__code__)
    prefix = f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/"
    for name, entry in modules.items():
        if (
            type(name) is not str
            or not name
            or len(name) > 256
            or any(not part or any(not (c.isalnum() or c in "_-") for c in part)
                   for part in name.split("."))
            or type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not bool
            or type(entry[1]) is not code_type
        ):
            raise ValueError("invalid trusted stdlib module")
        filename = prefix + name.replace(".", "/") + ("/__init__.py" if entry[0] else ".py")
        if entry[1].co_filename != filename:
            raise ValueError("trusted stdlib source path mismatch")
    # Capture trusted primitives locally; none are exposed in the candidate scope.
    trusted_exec = exec
    module_spec = _frozen_importlib.ModuleSpec

    class MemoryStdlibFinder:
        def find_spec(self, fullname, path=None, target=None):
            entry = modules.get(fullname)
            if entry is None:
                return None
            spec = module_spec(fullname, self, is_package=entry[0])
            spec.origin = entry[1].co_filename
            spec.has_location = True
            if entry[0]:
                spec.submodule_search_locations = [spec.origin.rsplit("/", 1)[0]]
            return spec

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            trusted_exec(modules[module.__name__][1], module.__dict__)

    finder = MemoryStdlibFinder()
    priority = [_frozen_importlib.BuiltinImporter, _frozen_importlib.FrozenImporter]
    sys.meta_path[:] = [*priority, finder, *(item for item in sys.meta_path if item not in priority)]
    return finder


# Bound the code-map allocation as well as the later strategy execution.
import resource

resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
_install_stdlib_map()

import ast
import ctypes
import json
import os
import socket
import sys

sys.path.insert(0, "/")
from trusted_helpers import base_action, default_candidates, default_search


def strict_json(line):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    return json.loads(
        line,
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def install_filter():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0):
        raise RuntimeError("no_new_privs failed")
    sec = ctypes.CDLL("/lib/libseccomp.so.2", use_errno=True)
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_syscall_resolve_name.restype = ctypes.c_int
    sec.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    # Default EPERM; dependencies must already be loaded before candidate execution.
    context = sec.seccomp_init(0x00050001)
    if not context:
        raise RuntimeError("seccomp init failed")
    try:
        for name in (
            "read",
            "write",
            "close",
            "fstat",
            "newfstatat",
            "lseek",
            "brk",
            "mmap",
            "munmap",
            "mprotect",
            "mremap",
            "madvise",
            "futex",
            "clock_gettime",
            "rt_sigaction",
            "rt_sigprocmask",
            "rt_sigreturn",
            "sigaltstack",
            "getpid",
            "gettid",
            "getrandom",
            "exit",
            "exit_group",
        ):
            number = sec.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and sec.seccomp_rule_add(context, 0x7FFF0000, number, 0):
                raise RuntimeError("seccomp rule failed")
        if sec.seccomp_load(context):
            raise RuntimeError("seccomp load failed")
    finally:
        sec.seccomp_release(context)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    first = strict_json(sys.stdin.buffer.readline(262145))
    if not isinstance(first, dict) or len(first["source"].encode()) > 65536:
        raise ValueError("source limit")
    # Compilation is non-executing. Module execution occurs after filter installation.
    code = compile(ast.parse(first["source"]), "<strategy>", "exec")
    builtins = {
        "len": len,
        "min": min,
        "max": max,
        "abs": abs,
        "sorted": sorted,
        "range": lambda n: range(min(max(n, 0), 128)),
        "enumerate": enumerate,
    }
    scope = {
        "__builtins__": builtins,
        "base_action": base_action,
        "default_candidates": default_candidates,
        "default_search": default_search,
        "get": lambda d, k, default=None: d.get(k, default),
        "copy_dict": lambda d: dict(d),
    }
    probe_libc = ctypes.CDLL(None, use_errno=True)
    install_filter()
    exec(code, scope, scope)
    print('{"ready":true}', flush=True)
    request = strict_json(sys.stdin.buffer.readline(262145))
    probe = request["args"][0] if request.get("function") == "probe" else None
    if probe:
        try:
            if probe == "read":
                os.open("/etc/passwd", os.O_RDONLY)
            elif probe == "write":
                os.open("/escape", os.O_WRONLY | os.O_CREAT, 0o600)
            elif probe == "network":
                socket.socket()
            elif probe == "fork":
                os.fork()
            elif probe == "exec":
                os.execve("/bin/python", ["python"], {})
            elif probe == "labels":
                os.open("/labels", os.O_RDONLY)
            elif probe == "memory":
                try:
                    bytearray(512 * 1024 * 1024)
                except MemoryError:
                    print('{"blocked":true}', flush=True)
                    return
            elif probe == "flood":
                while True:
                    pass
            elif probe == "mount":
                if (
                    probe_libc.mount(None, b"/", None, 0, None) == -1
                    and ctypes.get_errno() == 1
                ):
                    print('{"blocked":true}', flush=True)
                    return
            elif probe == "ptrace":
                if probe_libc.ptrace(0, 0, 0, 0) == -1 and ctypes.get_errno() == 1:
                    print('{"blocked":true}', flush=True)
                    return
            else:
                raise ValueError("unknown probe")
        except PermissionError:
            print('{"blocked":true}', flush=True)
            return
        raise RuntimeError("sandbox probe escaped")
    function = request["function"]
    if function not in ("choose_action", "propose_candidates", "update_search"):
        raise ValueError("unknown ABI function")
    result = scope[function](*request["args"])
    output = json.dumps(result, allow_nan=False, separators=(",", ":"))
    if len(output.encode()) > 16384:
        raise ValueError("action size limit")
    print(output, flush=True)


if __name__ == "__main__":
    main()
