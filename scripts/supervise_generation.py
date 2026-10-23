#!/usr/bin/env python3
"""Persistently supervise one fresh generation job; never launch distributed training.

Usage: python scripts/supervise_generation.py --request /absolute/request.json
The request contains name, python, script, args, out_dir, log_dir, gpus,
runtime_limit_seconds, and required_sha256 (absolute file paths to SHA256 values).

Advisory GPU locks serialize cooperating supervisors. Non-cooperating GPU users
are checked before launch and at exit; they are never signalled by this script.
Only the child session/process group created here may receive TERM/KILL.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from uuid import uuid4


LOCK_ROOT = Path("/tmp/cityagent-generation-gpu-locks")
QUERY_TIMEOUT = 15
METRICS_INTERVAL = 10
_REQUIRED = {"name", "python", "script", "args", "out_dir", "log_dir", "gpus",
             "runtime_limit_seconds", "required_sha256"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def error_text(error):
    return f"{type(error).__name__}: {error}"


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path, interrupted=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            if interrupted is not None and interrupted["signal"] is not None:
                raise InterruptedError("Interrupted while checking required hashes")
            digest.update(block)
        after = os.fstat(handle.fileno())
    current = Path(path).stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise RuntimeError(f"Required file changed while hashing: {path}")
    return digest.hexdigest()


def validate_request(value):
    if not isinstance(value, dict) or set(value) != _REQUIRED:
        raise ValueError(f"Request must contain exactly: {', '.join(sorted(_REQUIRED))}")
    request = dict(value)
    if not isinstance(request["name"], str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", request["name"]):
        raise ValueError("Invalid run name")
    for key in ("python", "script", "out_dir", "log_dir"):
        path = request[key]
        if not isinstance(path, str) or not Path(path).is_absolute() or "\x00" in path:
            raise ValueError(f"{key} must be an absolute filesystem path")
        # Preserve the venv's python symlink; resolving it can change the environment.
        request[key] = os.path.abspath(path)
    if not Path(request["python"]).is_file() or not os.access(request["python"], os.X_OK):
        raise ValueError("python must be an existing executable")
    if not Path(request["script"]).is_file() or Path(request["script"]).suffix != ".py":
        raise ValueError("script must be an existing Python source file")
    args = request["args"]
    if not isinstance(args, list) or any(not isinstance(arg, str) or "\x00" in arg for arg in args):
        raise ValueError("args must be a list of strings")
    if any(arg == "--" or arg == "--out-dir" or arg.startswith("--out-dir=") for arg in args):
        raise ValueError("The supervisor exclusively sets --out-dir")
    gpus = request["gpus"]
    if isinstance(gpus, str):
        pieces = gpus.split(",")
        if any(not re.fullmatch(r"\d+", item) for item in pieces):
            raise ValueError("gpus must contain distinct nonnegative physical GPU indexes")
        gpus = list(map(int, pieces))
    if (not isinstance(gpus, list) or not gpus or any(type(gpu) is not int or not 0 <= gpu < 1024 for gpu in gpus)
            or len(set(gpus)) != len(gpus)):
        raise ValueError("gpus must contain distinct nonnegative physical GPU indexes")
    request["gpus"] = gpus
    limit = request["runtime_limit_seconds"]
    if type(limit) not in (int, float) or not math.isfinite(limit) or not 0 < limit <= 86400:
        raise ValueError("runtime_limit_seconds must be positive and at most 86400")
    required = request["required_sha256"]
    if not isinstance(required, dict) or not required:
        raise ValueError("required_sha256 must be a nonempty absolute-path/hash mapping")
    for path, digest in required.items():
        if (not isinstance(path, str) or not Path(path).is_absolute() or not Path(path).is_file()
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Invalid required file or SHA256")
    if Path(request["script"]).resolve() not in {Path(path).resolve() for path in required}:
        raise ValueError("required_sha256 must include the child script")
    out, logs = Path(request["out_dir"]).resolve(), Path(request["log_dir"]).resolve()
    if out == logs or out in logs.parents:
        raise ValueError("log_dir cannot be inside the child-managed output directory")
    return request


def artifact_paths(request):
    prefix = Path(request["log_dir"]) / request["name"]
    return {key: Path(str(prefix) + suffix) for key, suffix in {
        "stdout": ".stdout.log", "metrics": ".metrics.jsonl",
        "started": ".started.json", "receipt": ".receipt.json",
    }.items()}


def write_new_json(path, value):
    """Atomic no-replace publication; a failed write never leaves partial JSON."""
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def resource_locks(request, lock_root=LOCK_ROOT):
    """Acquire every lock before GPU queries; descriptors never pass to the child."""
    lock_root = Path(lock_root)
    lock_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if lock_root.is_symlink():
        raise ValueError("Lock root cannot be a symlink")
    names = [f"gpu-{index:04d}.lock" for index in request["gpus"]]
    names += ["output-" + json_sha(str(Path(request["out_dir"]).resolve())) + ".lock",
              "logs-" + json_sha(str(Path(request["log_dir"]).resolve() / request["name"])) + ".lock"]
    with ExitStack() as stack:
        for name in sorted(names):
            descriptor = os.open(lock_root / name, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
            handle = stack.enter_context(os.fdopen(descriptor, "r+"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Resource is already locked: {name}") from error
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"supervisor_pid": os.getpid(), "name": request["name"], "utc": utc_now()}))
            handle.flush()
        yield


class NvidiaSmi:
    def _query(self, kind, fields, uuids=None):
        command = ["nvidia-smi", f"--query-{kind}=" + fields, "--format=csv,noheader,nounits"]
        if uuids:
            command += ["--id=" + ",".join(uuids)]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=QUERY_TIMEOUT)
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))

    def devices(self):
        result = {}
        for row in self._query("gpu", "index,uuid"):
            if len(row) != 2 or not row[1].strip().startswith("GPU-"):
                raise RuntimeError("Unrecognized nvidia-smi GPU identity output")
            index = int(row[0])
            if index in result:
                raise RuntimeError("Duplicate GPU index in nvidia-smi output")
            result[index] = row[1].strip()
        return result

    def processes(self, uuids):
        selected = set(uuids)
        result = []
        for row in self._query("compute-apps", "gpu_uuid,pid,process_name,used_gpu_memory"):
            if len(row) != 4:
                raise RuntimeError("Unrecognized nvidia-smi process output")
            if row[0].strip() in selected:
                result.append({"gpu_uuid": row[0].strip(), "pid": int(row[1]),
                               "process_name": row[2].strip(), "used_memory_mib": row[3].strip()})
        return result

    def metrics(self, uuids):
        rows = self._query("gpu", "index,uuid,utilization.gpu,memory.used,power.draw", uuids)
        result = []
        for row in rows:
            if len(row) != 5 or row[1].strip() not in uuids:
                raise RuntimeError("Unrecognized requested-GPU metrics output")
            result.append(dict(zip(("index", "uuid", "utilization_percent", "memory_mib", "power_watts"),
                                   (value.strip() for value in row))))
        return result


def process_identity(pid):
    result = {"pid": pid}
    try:
        result.update(pgid=os.getpgid(pid), sid=os.getsid(pid))
        path = Path(f"/proc/{pid}/stat")
        if path.exists():
            fields = path.read_text().rsplit(")", 1)[1].split()
            result.update(state=fields[0], parent_pid=int(fields[1]), start_time_ticks=int(fields[19]))
    except (OSError, ValueError, IndexError) as error:
        result["stat_error"] = error_text(error)
    return result


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def cleanup_owned_group(process, record, *, term_seconds=15, kill_seconds=10):
    errors = record.setdefault("cleanup_errors", [])

    def probe():
        process.poll()
        return group_exists(process.pid)

    def send(sig):
        try:
            os.killpg(process.pid, sig)
            record.setdefault("cleanup_signals", []).append(sig.name)
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(error_text(error))

    def wait_group(seconds):
        deadline = time.monotonic() + seconds
        while probe():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(0.1, remaining))
        return False

    try:
        if probe():
            send(signal.SIGTERM)
            if wait_group(term_seconds):
                send(signal.SIGKILL)
                wait_group(kill_seconds)
    except Exception as error:
        errors.append(error_text(error))
    finally:
        # A process can disappear between poll() and killpg(0); always reap the
        # session leader explicitly, even when its group has already vanished.
        try:
            process.wait(timeout=kill_seconds)
        except Exception as error:
            errors.append(error_text(error))
        try:
            record["final_owned_group_exists"] = probe()
        except Exception as error:
            errors.append(error_text(error))
            record["final_owned_group_exists"] = None
        record["child_exit_code"] = process.returncode


@contextmanager
def interrupt_signals():
    state = {"signal": None}
    originals = {}

    def handle(signum, _frame):
        if state["signal"] is None:
            state["signal"] = signum

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            originals[signum] = signal.signal(signum, handle)
        yield state
    finally:
        for signum, original in originals.items():
            signal.signal(signum, original)


def exit_code(record):
    if record.get("interrupted_signal") is not None:
        return 128 + record["interrupted_signal"]
    if record.get("timed_out"):
        return 124
    if record.get("controller_error") or record.get("cleanup_errors") or record.get("finalization_errors"):
        return 1
    if record.get("final_owned_group_exists") is not False or record.get("final_requested_processes") != []:
        return 1
    child = record.get("child_exit_code")
    return child if type(child) is int and child >= 0 else (128 - child if type(child) is int else 1)


def supervise(value, *, gpu=None, lock_root=LOCK_ROOT, metrics_interval=METRICS_INTERVAL,
              cleanup=cleanup_owned_group):
    request = validate_request(value)
    gpu = gpu or NvidiaSmi()
    paths = artifact_paths(request)
    out = Path(request["out_dir"])
    with interrupt_signals() as interrupted, resource_locks(request, lock_root):
        if any(os.path.lexists(path) for path in [out, *paths.values()]):
            raise FileExistsError("Output/log/receipt already exists; select fresh paths")
        Path(request["log_dir"]).mkdir(parents=True, exist_ok=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        record = {"schema_version": 1, "name": request["name"], "request": request,
                  "request_sha256": json_sha(request), "supervisor": process_identity(os.getpid()),
                  "controller_sha256": file_sha(Path(__file__)), "started_utc": utc_now(),
                  "child_exit_code": None, "final_owned_group_exists": False,
                  "final_requested_processes": None, "timed_out": False}
        process, uuids = None, []
        started = time.monotonic()
        try:
            # Reserve log files before any child starts. Existing files are never overwritten.
            with paths["stdout"].open("xb") as stdout, paths["metrics"].open("x", encoding="utf-8") as metrics:
                verified = {}
                for path, expected in request["required_sha256"].items():
                    actual = file_sha(path, interrupted)
                    if actual != expected:
                        raise ValueError(f"Required SHA256 mismatch: {path}")
                    verified[path] = actual
                record["verified_sha256"] = verified
                devices = gpu.devices()
                if any(index not in devices for index in request["gpus"]):
                    raise ValueError("A requested GPU index is unavailable")
                uuids = [devices[index] for index in request["gpus"]]
                record["gpu_devices"] = [{"index": index, "uuid": devices[index]} for index in request["gpus"]]
                before = gpu.processes(uuids)
                record["initial_requested_processes"] = before
                if before:
                    raise RuntimeError("Requested GPUs already have compute processes; refusing overlap")
                if interrupted["signal"] is not None:
                    raise InterruptedError("Interrupted before child launch")
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=",".join(uuids), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
                command = [request["python"], request["script"], *request["args"], "--out-dir", request["out_dir"]]
                record.update(command=command, environment_overrides={key: env[key] for key in
                              ("CUDA_VISIBLE_DEVICES", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM")})
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout,
                    stderr=subprocess.STDOUT, env=env, start_new_session=True, close_fds=True)
                child_started = time.monotonic()
                record.update(child=process_identity(process.pid), owned_pgid=process.pid, child_started_utc=utc_now())
                write_new_json(paths["started"], {**record, "status": "running"})
                print(json.dumps({"status": "started", "name": request["name"], "pid": process.pid,
                                  "started_receipt": str(paths["started"])}), flush=True)
                deadline = child_started + request["runtime_limit_seconds"]
                next_metrics = child_started
                while process.poll() is None:
                    now = time.monotonic()
                    if interrupted["signal"] is not None:
                        break
                    if now >= deadline:
                        record["timed_out"] = True
                        break
                    if now >= next_metrics:
                        sample = {"utc": utc_now(), "elapsed_seconds": now - child_started}
                        try:
                            sample["gpus"] = gpu.metrics(uuids)
                        except Exception as error:
                            sample["telemetry_error"] = error_text(error)
                        metrics.write(json.dumps(sample, allow_nan=False) + "\n")
                        metrics.flush()
                        next_metrics = now + metrics_interval
                    remaining = min(deadline - time.monotonic(), next_metrics - time.monotonic(), 0.2)
                    if remaining > 0:
                        try:
                            process.wait(timeout=remaining)
                        except subprocess.TimeoutExpired:
                            pass
        except (Exception, KeyboardInterrupt) as error:
            record["controller_error"] = error_text(error)
            if isinstance(error, KeyboardInterrupt):
                interrupted["signal"] = signal.SIGINT
        finally:
            try:
                if process is not None:
                    cleanup(process, record)
            except (Exception, KeyboardInterrupt) as error:
                record.setdefault("cleanup_errors", []).append(error_text(error))
                record["final_owned_group_exists"] = None
            finally:
                record["interrupted_signal"] = interrupted["signal"]
                record["wall_seconds"] = time.monotonic() - started
                record["ended_utc"] = utc_now()
                if process is not None:
                    record["child_exit_code"] = process.returncode
                try:
                    if uuids:
                        record["final_requested_processes"] = gpu.processes(uuids)
                except Exception as error:
                    record["final_gpu_query_error"] = error_text(error)
                finally:
                    try:
                        record["summary_exists"] = (out / "summary.json").is_file()
                    except OSError as error:
                        record.setdefault("finalization_errors", []).append(error_text(error))
                        record["summary_exists"] = None
                    record["exit_code"] = exit_code(record)
                    record["status"] = "completed" if record["exit_code"] == 0 else "failed"
                    try:
                        write_new_json(paths["receipt"], record)
                    except Exception as error:
                        record["receipt_write_error"] = error_text(error)
                        record["exit_code"] = 1
                        record["status"] = "failed"
                        print(json.dumps(record), file=sys.stderr, flush=True)
        print(json.dumps({"status": record["status"], "name": request["name"], "exit_code": record["exit_code"],
                          "receipt": str(paths["receipt"])}), flush=True)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        return supervise(request)["exit_code"]
    except Exception as error:
        print(json.dumps({"status": "rejected", "error": error_text(error)}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
