"""Prepared minimal-root sandbox and prewarmed one-shot workers; fail closed."""

from __future__ import annotations

import _frozen_importlib_external
import ast
import json
import marshal
import math
import os
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import sysconfig
import threading
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from . import rsi
from .inference_core import BASIC_FIELDS
from .rsi import base_action, validate_action
from .run_state import atomic_json, file_digest, json_digest


def _stage_stdlib(stdlib, root):
    """Collapse pure stdlib files into Python's relocatable zip landmark."""
    stdlib, root = Path(stdlib).resolve(), Path(root)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    destination = root / "lib" / ("python" + version)
    destination.mkdir(parents=True, exist_ok=True)
    archive = root / "lib" / ("python" + version.replace(".", "") + ".zip")
    ignored = {
        "site-packages", "dist-packages", "__pycache__", "test", "tests",
        "tkinter", "ensurepip", "idlelib",
    }
    modules = {}
    # ZIP_STORED avoids depending on a native zlib extension during encodings boot.
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as zipped:
        for directory, dirs, files in os.walk(stdlib, followlinks=False):
            # Build archives/Makefiles are not interpreter runtime imports.
            dirs[:] = sorted(
                name for name in dirs
                if name not in ignored and not name.startswith("config-")
            )
            for name in dirs:
                if (Path(directory) / name).is_symlink():
                    raise ValueError("stdlib directory symlink unsupported")
            for name in sorted(files):
                if name in ignored:
                    continue
                source = Path(directory) / name
                if source.is_symlink() and not source.resolve().is_relative_to(stdlib):
                    raise ValueError("stdlib symlink escapes selected runtime")
                if not source.is_file():
                    raise ValueError("unsupported stdlib entry")
                relative = source.relative_to(stdlib)
                if source.suffix == ".so":
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                else:
                    contents = source.read_bytes()
                    zipped.writestr(relative.as_posix(), contents)
                    if source.suffix == ".py":
                        package = relative.name == "__init__.py"
                        parts = relative.parts[:-1] if package else (*relative.parts[:-1], relative.stem)
                        name = ".".join(parts)
                        if not name or name in modules:
                            raise ValueError("invalid duplicate stdlib module")
                        filename = f"/lib/python{version}/{relative.as_posix()}"
                        modules[name] = (package, compile(
                            contents, filename, "exec", dont_inherit=True, optimize=0
                        ))
    payload = {
        "schema_version": 1,
        "python_version": sys.version,
        "cache_tag": sys.implementation.cache_tag,
        "marshal_version": marshal.version,
        "magic_number": _frozen_importlib_external.MAGIC_NUMBER,
        "modules": modules,
    }
    encoded = marshal.dumps(payload)
    if len(encoded) > 64 * 1024 * 1024:
        raise ValueError("trusted stdlib code map exceeds 64 MiB")
    with (root / "stdlib.marshal").open("xb") as output:
        output.write(encoded)
    return destination


def prepare_root(root):
    root = Path(root).resolve()
    if sys.platform != "linux" or not str(root).startswith("/workspace/"):
        raise RuntimeError("sandbox preparation requires official Linux /workspace")
    root.mkdir(parents=True, exist_ok=False)
    (root / "bin").mkdir()
    (root / "lib").mkdir()
    interpreter = Path(sys.executable).resolve()
    shutil.copy2(interpreter, root / "bin/python")
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    destination = _stage_stdlib(stdlib, root)
    # Capture native dependencies of Python and every included extension.
    binaries = [interpreter, *destination.rglob("*.so")]
    copied = set()
    for binary in binaries:
        result = subprocess.run(
            ["ldd", str(binary)], capture_output=True, text=True, check=False
        )
        for line in result.stdout.splitlines():
            for item in line.split():
                if item.startswith("/") and Path(item).is_file() and item not in copied:
                    target = root / Path(item).relative_to("/")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
                    copied.add(item)
    seccomp = next(
        (
            p
            for p in (
                Path("/lib/x86_64-linux-gnu/libseccomp.so.2"),
                Path("/usr/lib/x86_64-linux-gnu/libseccomp.so.2"),
                Path("/lib/aarch64-linux-gnu/libseccomp.so.2"),
            )
            if p.is_file()
        ),
        None,
    )
    if seccomp is None:
        raise RuntimeError("libseccomp unavailable")
    shutil.copy2(seccomp, root / "lib/libseccomp.so.2")
    worker = Path(__file__).with_name("rsi_worker.py")
    shutil.copy2(worker, root / "worker.py")
    tree = ast.parse(Path(rsi.__file__).read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("base_action", "default_candidates", "default_search")
    ]
    helper = (
        "DEFAULT = "
        + repr(rsi.DEFAULT)
        + "\n"
        + "\n".join(ast.unparse(n) for n in functions)
    )
    (root / "trusted_helpers.py").write_text(helper)
    # Reject symlinks that would escape the minimal root.
    for path in root.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise RuntimeError("sandbox runtime has escaping symlink")
    files = {
        str(p.relative_to(root)): file_digest(p) for p in root.rglob("*") if p.is_file()
    }
    atomic_json(
        root.parent / (root.name + ".identity.json"),
        {"schema_version": 1, "files": files, "digest": json_digest(files)},
    )
    return root


def _runtime_metadata(root, identity_path, expected_files):
    """Kernel metadata and entry set, including links and empty directories."""
    if root.resolve() != root or not root.is_dir() or identity_path.is_symlink():
        raise ValueError("sandbox runtime path changed")
    metadata, actual_files = {}, set()

    def record(path, info):
        if not (
            stat.S_ISREG(info.st_mode)
            or stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
        ):
            raise ValueError("unsupported sandbox runtime entry")
        link = None
        is_file = stat.S_ISREG(info.st_mode)
        if stat.S_ISLNK(info.st_mode):
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("sandbox runtime symlink escapes root")
            link = os.readlink(path)
            is_file = path.is_file()
        if path not in (root, identity_path) and is_file:
            actual_files.add(str(path.relative_to(root)))
        metadata[str(path)] = (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            link,
        )

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW

    def walk(directory, fd):
        # DirEntry.stat reuses one nofollow lookup; never follow a replaced dir.
        with os.scandir(fd) as entries:
            for entry in entries:
                path = directory / entry.name
                info = entry.stat(follow_symlinks=False)
                record(path, info)
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(entry.name, flags, dir_fd=fd)
                    try:
                        opened = os.fstat(child)
                        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                            raise ValueError("sandbox directory changed during verification")
                        walk(path, child)
                    finally:
                        os.close(child)

    try:
        record(identity_path, identity_path.lstat())
        fd = os.open(root, flags)
        try:
            record(root, os.fstat(fd))
            walk(root, fd)
        finally:
            os.close(fd)
    except OSError as error:
        raise ValueError("sandbox runtime changed during verification") from error
    if actual_files != set(expected_files):
        raise ValueError("sandbox runtime file set changed")
    return metadata


@dataclass(frozen=True)
class _RuntimeAttestation:
    """One prewarm round only; never persisted or reused by a subsequent round."""

    root: Path
    identity_path: Path
    identity: dict
    metadata: dict

    def check_unchanged(self):
        if (
            _runtime_metadata(self.root, self.identity_path, self.identity["files"])
            != self.metadata
        ):
            raise ValueError("sandbox runtime changed after full verification")


def _verify_runtime(root):
    root = Path(root).resolve()
    identity_path = root.parent / (root.name + ".identity.json")
    if identity_path.is_symlink():
        raise ValueError("sandbox identity symlink forbidden")
    # Include identity-file metadata before reading it, not just after parsing.
    identity_stat = identity_path.stat()
    identity = json.loads(identity_path.read_text())
    files = identity.get("files")
    if not isinstance(files, dict) or identity.get("digest") != json_digest(files):
        raise ValueError("sandbox runtime identity changed")
    for name in files:
        path = Path(name)
        if (
            not isinstance(name, str)
            or path.is_absolute()
            or not path.parts
            or any(p in ("..", ".") for p in path.parts)
        ):
            raise ValueError("sandbox runtime path escapes root")
    before = _runtime_metadata(root, identity_path, files)
    after_read = identity_path.stat()
    if (
        identity_stat.st_dev,
        identity_stat.st_ino,
        identity_stat.st_size,
        identity_stat.st_mtime_ns,
        identity_stat.st_ctime_ns,
    ) != (
        after_read.st_dev,
        after_read.st_ino,
        after_read.st_size,
        after_read.st_mtime_ns,
        after_read.st_ctime_ns,
    ):
        raise ValueError("sandbox identity changed during verification")
    if any(file_digest(root / name) != digest for name, digest in files.items()):
        raise ValueError("sandbox runtime changed")
    verified = _RuntimeAttestation(root, identity_path, identity, before)
    verified.check_unchanged()
    return verified


class _WorkerLifetime:
    """Bound one Popen-owned session independently of the prewarm thread.

    Every reap and signal uses this lock. An unreaped child retains its PID even
    after exit; after poll/close releases it, no callback may signal that PID.
    """

    def __init__(self, process, born):
        self.process = process
        self.lock = threading.Lock()
        self.closed = False
        self.expired = False
        self.timer = threading.Timer(
            max(0, born + 3 - time.monotonic()), self._expire
        )
        self.timer.daemon = True
        try:
            self.timer.start()
        except Exception:
            self.close()
            process.wait(timeout=3)
            raise

    def _kill_locked(self):
        # Do not poll here: reaping would release the PID before killpg().
        if self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _expire(self):
        with self.lock:
            if not self.closed and self.process.returncode is None:
                self.expired = True
                self._kill_locked()

    def poll(self):
        with self.lock:
            result = self.process.poll()
            if result is not None:
                self.timer.cancel()
            return result

    def close(self):
        """Cancel and signal only; process waiting remains on the cleanup path."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.timer.cancel()
            self._kill_locked()


class SandboxWorker:
    def __init__(self, root, bundle, *, probe=None, _verified_runtime=None):
        if sys.platform != "linux" or not shutil.which("unshare"):
            raise RuntimeError("Linux namespace sandbox unavailable")
        self.root = Path(root).resolve()
        verified = _verified_runtime or _verify_runtime(self.root)
        if not isinstance(verified, _RuntimeAttestation) or verified.root != self.root:
            raise ValueError("invalid sandbox runtime attestation")
        verified.check_unchanged()  # Cheap mutation check immediately before process creation.
        self.identity = verified.identity
        launcher = Path(__file__).with_name("rsi_namespace.py")
        # Start conservatively before Popen: startup/readiness never extends life.
        self.born = time.monotonic()
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(launcher),
                "--outer",
                str(self.root),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin"},
            start_new_session=True,
            bufsize=0,
        )
        self._lifetime = _WorkerLifetime(self.process, self.born)
        self.selector = None
        self.buffer = b""
        try:
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.process.stdout, selectors.EVENT_READ)
            self.send({"source": bundle.source, "probe": probe})
            if self.read(3) != {"ready": True}:
                raise RuntimeError("sandbox readiness failed")
            self.kernel_proof = self.verify_process()
            verified.check_unchanged()
        except Exception:
            self.close()
            raise

    def verify_process(self):
        # Inspect the actual child from the host, not a worker-supplied status flag.
        children = (
            Path(f"/proc/{self.process.pid}/task/{self.process.pid}/children")
            .read_text()
            .split()
        )
        if len(children) != 1:
            raise RuntimeError("sandbox worker process identity unavailable")
        pid = int(children[0])
        status = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                status[key] = value.strip()
        if (
            status.get("NoNewPrivs") != "1"
            or status.get("Seccomp") != "2"
            or int(status.get("CapEff", "1"), 16) != 0
        ):
            raise RuntimeError("sandbox kernel protections missing")
        for namespace in ("user", "mnt", "net", "pid", "ipc", "uts"):
            if os.readlink(f"/proc/{pid}/ns/{namespace}") == os.readlink(
                f"/proc/self/ns/{namespace}"
            ):
                raise RuntimeError("sandbox namespace isolation missing")
        if Path(f"/proc/{pid}/root").resolve() != self.root:
            raise RuntimeError("sandbox root identity mismatch")
        mounts = Path(f"/proc/{pid}/mountinfo").read_text().splitlines()
        roots = [
            line.split()[5].split(",") for line in mounts if line.split()[4] == "/"
        ]
        if not roots or not all(
            {"ro", "nosuid", "nodev"}.issubset(flags) for flags in roots
        ):
            raise RuntimeError("sandbox readonly root missing")
        return {
            "pid": pid,
            "seccomp": 2,
            "no_new_privs": 1,
            "capabilities": 0,
            "namespaces": 6,
            "readonly_root": True,
        }

    def send(self, value):
        data = json.dumps(value, allow_nan=False).encode() + b"\n"
        if len(data) > 262144:
            raise ValueError("sandbox state limit")
        self.process.stdin.write(data)

    def read(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError("sandbox deadline")
            data = os.read(self.process.stdout.fileno(), 16385)
            if not data:
                raise RuntimeError("sandbox exited before reply")
            self.buffer += data
            if len(self.buffer) > 16384:
                raise ValueError("sandbox reply size")
        line, self.buffer = self.buffer.split(b"\n", 1)

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate sandbox key")
                result[key] = value
            return result

        return json.loads(
            line,
            object_pairs_hook=pairs,
            parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)),
        )

    def invoke(self, function, args, timeout=0.005):
        self.send({"function": function, "args": args})
        return self.read(timeout)

    def poll(self):
        return self._lifetime.poll()

    def terminate(self):
        self._lifetime.close()

    def close(self):
        self.terminate()
        self.process.wait(timeout=3)
        if self.selector is not None:
            self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()


def policy_state(value):
    """Only structural, numeric execution state may enter proposed code or its logs."""
    if not isinstance(value, dict) or value.get("phase") not in {
        "schedule",
        "failure",
        "merge",
    }:
        raise ValueError("invalid policy phase")
    result = {"phase": value["phase"]}
    for key in (
        "batch_size",
        "remaining_calls",
        "remaining_seconds",
        "attempt",
        "depth",
        "max_depth",
        "max_attempts",
        "react_calls",
    ):
        item = value.get(key)
        if type(item) in (int, float) and math.isfinite(item) and 0 <= item <= 1e9:
            result[key] = item
    if type(value.get("allow_repair")) is bool:
        result["allow_repair"] = value["allow_repair"]
    if value.get("error_kind") in {
        "length",
        "format",
        "temporary",
        "oom",
        "coverage",
        "conflict",
    }:
        result["error_kind"] = value["error_kind"]
    for group, keys in {
        "structure": (
            "owned_regions",
            "context_regions",
            "table_rows",
            "characters",
            "years",
        ),
        "coverage": ("completed_tasks", "pending_tasks", "total_tasks"),
        "conflicts": ("count",),
    }.items():
        source = value.get(group, {})
        if isinstance(source, dict):
            clean = {
                key: source[key]
                for key in keys
                if type(source.get(key)) in (int, float)
                and math.isfinite(source[key])
                and 0 <= source[key] <= 1e9
            }
            if group == "conflicts" and isinstance(source.get("fields"), list):
                clean["fields"] = [
                    v
                    for v in source["fields"]
                    if isinstance(v, str) and v in BASIC_FIELDS
                ][:32]
            if clean:
                result[group] = clean
    return result


class PolicyController:
    """Prewarm bounded one-shot workers; no foreground wait for replenishment."""

    def __init__(self, bundle, root=None, *, pool_size=2):
        if type(pool_size) is not int or not 1 <= pool_size <= 4:
            raise ValueError("bounded worker pool required")
        self.bundle, self.root = bundle, root
        self.pool_size = pool_size
        self.pool = deque()
        self.retired = deque()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.ready = threading.Event()
        self.decisions = []
        self.error = None
        self.thread = None
        if root:
            self.thread = threading.Thread(target=self._prepare, daemon=True)
            self.thread.start()

    @staticmethod
    def _expired(worker):
        return (
            time.monotonic() - worker.born >= 2.5 or worker.poll() is not None
        )

    def _prepare(self):
        while not self.stop.is_set():
            with self.lock:
                expired = [w for w in self.pool if self._expired(w)]
                self.pool = deque(w for w in self.pool if w not in expired)
                retired = [*self.retired, *expired]
                self.retired.clear()
                needed = len(self.pool) < self.pool_size
                self.ready.set() if len(
                    self.pool
                ) == self.pool_size else self.ready.clear()
            for worker in retired:
                worker.close()
            if needed:
                round_workers = []
                try:
                    # A second full content read used to consume the first worker's
                    # entire lifetime. Validate once, then start this bounded round.
                    verified = _verify_runtime(self.root)
                    with self.lock:
                        expired = [w for w in self.pool if self._expired(w)]
                        self.pool = deque(w for w in self.pool if w not in expired)
                        self.retired.extend(expired)
                        count = self.pool_size - len(self.pool)
                    for _ in range(count):
                        if self.stop.is_set():
                            break
                        round_workers.append(
                            SandboxWorker(
                                self.root, self.bundle, _verified_runtime=verified
                            )
                        )
                    verified.check_unchanged()
                except Exception as error:
                    for worker in round_workers:
                        worker.close()
                    self.error = type(error).__name__
                    self.ready.clear()
                    return
                with self.lock:
                    if self.stop.is_set():
                        self.retired.extend(round_workers)
                    else:
                        self.pool.extend(round_workers)
                        if len(self.pool) == self.pool_size:
                            self.ready.set()
                continue
            self.wake.wait(0.05)
            self.wake.clear()

    def wait_ready(self, timeout=2.5):
        """Offline/prepare-stage only. choose() never waits on this event."""
        if not self.root:
            return False
        return self.ready.wait(min(max(timeout, 0), 2.5))

    def choose(self, state):
        started = time.perf_counter()
        state = policy_state(state)
        worker, discarded = None, []
        with self.lock:
            for _ in range(min(self.pool_size, len(self.pool))):
                if time.perf_counter() - started >= 0.005:
                    break
                candidate = self.pool.popleft()
                if self._expired(candidate):
                    discarded.append(candidate)
                else:
                    worker = candidate
                    break
            self.retired.extend(discarded)
            self.ready.clear()
        reason = (
            "sandbox-unavailable"
            if not self.root
            else ("worker-expired" if discarded else "worker-not-ready")
        )
        action = None
        if worker:
            if self._expired(worker):
                reason = "worker-expired"
            else:
                try:
                    remaining = 0.005 - (time.perf_counter() - started)
                    if remaining <= 0:
                        raise TimeoutError("controller deadline")
                    action = validate_action(
                        worker.invoke("choose_action", [state], timeout=remaining),
                        state,
                    )
                    if time.perf_counter() - started > 0.005:
                        raise TimeoutError("late policy action")
                    reason = None
                except TimeoutError:
                    reason, action = "decision-timeout", None
                except ValueError:
                    reason, action = "invalid-action", None
                except Exception:
                    reason, action = "sandbox-error", None
            # A one-shot worker is never reused, including on timeout/invalid action.
            if reason:
                worker.terminate()
            with self.lock:
                self.retired.append(worker)
        self.wake.set()
        if action is None:
            action = base_action(state)
        validate_action(action, state)
        self.decisions.append(
            {
                "state": state,
                "state_sha256": json_digest(state),
                "action": action,
                "bundle_sha256": self.bundle.digest,
                "fallback": reason is not None,
                "fallback_reason": reason,
                "seconds": time.perf_counter() - started,
                **(
                    {
                        "kernel_proof": worker.kernel_proof,
                        "sandbox_runtime_digest": worker.identity["digest"],
                    }
                    if reason is None
                    else {}
                ),
            }
        )
        return action

    def telemetry(self):
        total = len(self.decisions)
        executed = sum(d["fallback"] is False for d in self.decisions)
        reasons = {}
        for decision in self.decisions:
            if decision["fallback"]:
                key = decision["fallback_reason"]
                reasons[key] = reasons.get(key, 0) + 1
        return {
            "decisions": total,
            "executed_decisions": executed,
            "execution_rate": executed / total if total else 0.0,
            "fallback_reasons": reasons,
        }

    def close(self):
        self.stop.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=4)
        with self.lock:
            for worker in [*self.pool, *self.retired]:
                worker.close()
            self.pool.clear()
            self.retired.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def verify_root(root):
    root = Path(root).resolve()
    identity = json.loads((root.parent / (root.name + ".identity.json")).read_text())
    proof = json.loads((root.parent / (root.name + ".acceptance.json")).read_text())
    if (
        proof.get("passed") is not True
        or proof.get("runtime_digest") != identity["digest"]
        or proof.get("worker_sha256")
        != file_digest(Path(__file__).with_name("rsi_worker.py"))
        or proof.get("launcher_sha256")
        != file_digest(Path(__file__).with_name("rsi_namespace.py"))
        or {p["probe"] for p in proof["probes"] if p["blocked"] is True}
        != {
            "read",
            "write",
            "network",
            "fork",
            "exec",
            "labels",
            "memory",
            "mount",
            "ptrace",
            "flood",
        }
    ):
        raise ValueError("real sandbox acceptance missing or stale")
    if any(
        file_digest(root / name) != digest for name, digest in identity["files"].items()
    ):
        raise ValueError("sandbox runtime changed after acceptance")
    return identity["digest"]


def sandbox_call(root, bundle, function, arguments, *, budget=None):
    """Account the actual isolated CPU experiment without overlapping a PPU lease."""
    job = budget.start("rsi-isolated-" + function, 10) if budget else None
    started, status = time.perf_counter(), "failed"
    try:
        worker = SandboxWorker(root, bundle)
        try:
            value = worker.invoke(function, arguments, timeout=1)
            status = "complete"
            return value
        finally:
            worker.close()
    finally:
        if job:
            budget.finish(job, time.perf_counter() - started, status)
