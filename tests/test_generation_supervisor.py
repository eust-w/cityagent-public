"""Local synthetic child processes and mock GPU queries; never runs a GPU workload."""
import contextlib
import functools
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/supervise_generation.py"
SPEC = importlib.util.spec_from_file_location("generation_supervisor", SCRIPT)
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


class MockGpu:
    def __init__(self, busy=False, telemetry_error=False):
        self.busy, self.telemetry_error = busy, telemetry_error
        self.queries = []

    def devices(self):
        self.queries.append("devices")
        return {0: "GPU-synthetic-0", 1: "GPU-synthetic-1"}

    def processes(self, uuids):
        self.queries.append(("processes", list(uuids)))
        return [{"gpu_uuid": uuids[0], "pid": 999999}] if self.busy else []

    def metrics(self, uuids):
        self.queries.append(("metrics", list(uuids)))
        if self.telemetry_error:
            raise subprocess.TimeoutExpired("mock-nvidia-smi", 15)
        return [{"uuid": uuid, "memory_mib": "0"} for uuid in uuids]


NORMAL_CHILD = '''import argparse, json, os, sys, time
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('--out-dir'); args = p.parse_args()
out = Path(args.out_dir); out.mkdir()
time.sleep(.06)
(out/'summary.json').write_text(json.dumps({'cuda':os.environ['CUDA_VISIBLE_DEVICES'],
    'pid':os.getpid(),'pgid':os.getpgrp(),'stdin':sys.stdin.read()}))
print('synthetic generation completed',flush=True)
'''


class GenerationSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="generation-supervisor-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.child = self.root / "synthetic_child.py"
        self.child.write_text(NORMAL_CHILD)
        self.request = {
            "name": "synthetic", "python": sys.executable, "script": str(self.child), "args": [],
            "out_dir": str(self.root / "new-output"), "log_dir": str(self.root / "logs"),
            "gpus": [1, 0], "runtime_limit_seconds": 3,
            "required_sha256": {str(self.child): supervisor.file_sha(self.child)},
        }
        self.locks = self.root / "locks"
        self.cleanup = functools.partial(supervisor.cleanup_owned_group, term_seconds=.2, kill_seconds=2)

    def change_child(self, code):
        self.child.write_text(code)
        self.request["required_sha256"][str(self.child)] = supervisor.file_sha(self.child)

    def run_supervisor(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return supervisor.supervise(self.request, gpu=kwargs.pop("gpu", MockGpu()), lock_root=self.locks,
                                        metrics_interval=.02, cleanup=kwargs.pop("cleanup", self.cleanup), **kwargs)

    def test_success_records_pid_hashes_uuid_visibility_devnull_and_new_receipts(self):
        gpu = MockGpu()
        record = self.run_supervisor(gpu=gpu)
        self.assertEqual(record["exit_code"], 0)
        self.assertEqual(record["child_exit_code"], 0)
        self.assertFalse(record["final_owned_group_exists"])
        self.assertEqual(record["final_requested_processes"], [])
        self.assertTrue(record["summary_exists"])
        summary = json.loads((Path(self.request["out_dir"]) / "summary.json").read_text())
        self.assertEqual(summary["cuda"], "GPU-synthetic-1,GPU-synthetic-0")
        self.assertEqual(summary["stdin"], "")
        self.assertEqual(summary["pid"], summary["pgid"])
        self.assertEqual(summary["pid"], record["owned_pgid"])
        self.assertEqual(record["verified_sha256"], self.request["required_sha256"])
        paths = supervisor.artifact_paths(self.request)
        self.assertEqual(json.loads(paths["receipt"].read_text()), record)
        self.assertEqual(json.loads(paths["started"].read_text())["status"], "running")
        self.assertIn("synthetic generation completed", paths["stdout"].read_text())
        self.assertTrue(paths["metrics"].read_text().strip())

    def test_existing_output_or_log_is_rejected_without_gpu_query_or_overwrite(self):
        output = Path(self.request["out_dir"])
        output.mkdir()
        gpu = MockGpu()
        with self.assertRaises(FileExistsError):
            self.run_supervisor(gpu=gpu)
        self.assertEqual(gpu.queries, [])
        output.rmdir()
        logs = Path(self.request["log_dir"]); logs.mkdir()
        stdout = supervisor.artifact_paths(self.request)["stdout"]
        stdout.write_bytes(b"previous evidence")
        with self.assertRaises(FileExistsError):
            self.run_supervisor(gpu=gpu)
        self.assertEqual(stdout.read_bytes(), b"previous evidence")
        self.assertEqual(gpu.queries, [])

    def test_wrong_hash_prevents_child_and_preserves_failure_receipt(self):
        self.request["required_sha256"][str(self.child)] = "0" * 64
        gpu = MockGpu()
        record = self.run_supervisor(gpu=gpu)
        self.assertEqual(record["exit_code"], 1)
        self.assertIn("SHA256 mismatch", record["controller_error"])
        self.assertNotIn("child", record)
        self.assertEqual(gpu.queries, [])
        self.assertFalse(Path(self.request["out_dir"]).exists())
        self.assertTrue(supervisor.artifact_paths(self.request)["receipt"].is_file())

    def test_locks_are_acquired_before_gpu_queries_and_held_through_final_query(self):
        gpu = MockGpu()
        with supervisor.resource_locks(self.request, self.locks):
            with self.assertRaisesRegex(RuntimeError, "already locked"):
                self.run_supervisor(gpu=gpu)
        self.assertEqual(gpu.queries, [])
        original = gpu.processes
        def check_lock(uuids):
            with self.assertRaisesRegex(RuntimeError, "already locked"):
                with supervisor.resource_locks(self.request, self.locks):
                    pass
            return original(uuids)
        gpu.processes = check_lock
        self.assertEqual(self.run_supervisor(gpu=gpu)["exit_code"], 0)
        with supervisor.resource_locks(self.request, self.locks):
            pass  # All locks released only after final receipt publication.

    def test_busy_requested_gpu_prevents_child(self):
        record = self.run_supervisor(gpu=MockGpu(busy=True))
        self.assertEqual(record["exit_code"], 1)
        self.assertIn("refusing overlap", record["controller_error"])
        self.assertNotIn("child", record)
        self.assertFalse(Path(self.request["out_dir"]).exists())

    def test_timeout_is_124_even_when_term_handler_exits_zero(self):
        self.change_child("import signal,time\nsignal.signal(signal.SIGTERM, lambda *a: exit(0))\ntime.sleep(30)\n")
        self.request["runtime_limit_seconds"] = .15
        record = self.run_supervisor()
        self.assertTrue(record["timed_out"])
        self.assertEqual(record["exit_code"], 124)
        self.assertEqual(record["child_exit_code"], 0)
        self.assertFalse(record["final_owned_group_exists"])

    def test_sigterm_is_recorded_and_only_owned_child_group_is_stopped(self):
        self.change_child("import time\ntime.sleep(30)\n")
        sentinel = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
        try:
            gpu = MockGpu()
            def interrupt(_uuids):
                os.kill(os.getpid(), signal.SIGTERM)
                return []
            gpu.metrics = interrupt
            record = self.run_supervisor(gpu=gpu)
            self.assertEqual(record["interrupted_signal"], signal.SIGTERM)
            self.assertEqual(record["exit_code"], 143)
            self.assertIsInstance(record["child_exit_code"], int)
            self.assertFalse(record["final_owned_group_exists"])
            self.assertIsNone(sentinel.poll(), "unrelated process must never be signalled")
        finally:
            sentinel.terminate(); sentinel.wait(timeout=3)

    def test_orphan_worker_is_killed_even_after_leader_exits(self):
        self.change_child('import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c",'
                          '"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"])\n'
                          'time.sleep(.1)\n')
        record = self.run_supervisor()
        self.assertEqual(record["child_exit_code"], 0)
        self.assertIn("SIGKILL", record["cleanup_signals"])
        self.assertFalse(record["final_owned_group_exists"])

    def test_telemetry_failure_does_not_abort_valid_child(self):
        record = self.run_supervisor(gpu=MockGpu(telemetry_error=True))
        self.assertEqual(record["exit_code"], 0)
        rows = [json.loads(line) for line in supervisor.artifact_paths(self.request)["metrics"].read_text().splitlines()]
        self.assertTrue(any("telemetry_error" in row for row in rows))

    def test_cleanup_exception_still_writes_failed_receipt(self):
        def failing_cleanup(process, record):
            self.cleanup(process, record)
            raise RuntimeError("synthetic cleanup bookkeeping failure")
        record = self.run_supervisor(cleanup=failing_cleanup)
        self.assertEqual(record["exit_code"], 1)
        self.assertEqual(record["child_exit_code"], 0)
        self.assertTrue(record["cleanup_errors"])
        self.assertEqual(json.loads(supervisor.artifact_paths(self.request)["receipt"].read_text()), record)

    def test_nvidia_queries_are_bounded_and_ignore_other_gpu_processes(self):
        nvidia = supervisor.NvidiaSmi()
        output = "GPU-other, 444, unrelated-job, 1024\n"
        completed = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
        with patch.object(supervisor.subprocess, "run", return_value=completed) as run:
            self.assertEqual(nvidia.processes(["GPU-selected"]), [])
        self.assertEqual(run.call_args.kwargs["timeout"], 15)
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_request_cannot_override_output_or_omit_script_hash(self):
        for args in (["--out-dir", "/tmp/other"], ["--out-dir=/tmp/other"], ["--"]):
            with self.subTest(args=args):
                request = {**self.request, "args": args}
                with self.assertRaisesRegex(ValueError, "exclusively sets"):
                    supervisor.validate_request(request)
        with self.assertRaises(ValueError):
            supervisor.validate_request({**self.request, "required_sha256": {}})


if __name__ == "__main__":
    unittest.main()
