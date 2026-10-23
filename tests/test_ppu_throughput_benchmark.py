"""Synthetic HTTP throughput tests; no real dataset, inference, or accelerator."""

import contextlib
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "ppu_throughput_benchmark.py"
SPEC = importlib.util.spec_from_file_location("ppu_throughput_benchmark", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class MockServer(ThreadingHTTPServer):
    request_queue_size = 32


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.requests.append(request)
            self.server.active += 1
            self.server.peak = max(self.server.peak, self.server.active)
        try:
            text = request["messages"][0]["content"]
            # Make overlapping requests and client dispatch wait observable.
            time.sleep(.015)
            if text == "HTTP_ERROR":
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'{"error":"synthetic failure"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def event(value):
                self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())
                self.wfile.flush()

            event({"choices": [{"index": 0, "delta": {"content": "synthetic"}}]})
            finish = "length" if request.get("ignore_eos") or text == "TRUNCATED" else "stop"
            event({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            if text != "MISSING_USAGE":
                count = request["max_tokens"] - (1 if text == "SHORT" else 0)
                event({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": count}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            with self.server.lock:
                self.server.active -= 1


class PPUThroughputBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = MockServer(("127.0.0.1", 0), Handler)
        cls.server.lock = threading.Lock()
        cls.server.requests, cls.server.active, cls.server.peak = [], 0, 0
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ppu-throughput-mock-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_cli(self, name, prompts, concurrency=2, fixed=True, extra=()):
        samples = self.root / f"{name}-samples.json"
        samples.write_text(json.dumps({"dataset_split": "train", "source": "synthetic mock only",
            "samples": [{"id": f"sample-{index:02d}", "prompt": prompt}
                        for index, prompt in enumerate(prompts)]}))
        output = self.root / name
        argv = [str(SCRIPT), "--samples", str(samples), "--endpoint",
                f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                "--model", "mock", "--concurrency", str(concurrency), "--max-tokens", "256",
                "--out-dir", str(output), "--label", name, *extra]
        if fixed:
            argv.append("--ignore-eos")
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            code = bench.main()
        report = json.loads((output / "summary.json").read_text())
        rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
        return code, report, rows, output

    def test_all_sixteen_samples_run_at_every_cap_and_rates_use_whole_batch_wall_time(self):
        for cap in (1, 2, 4, 8):
            with self.subTest(concurrency=cap):
                code, report, rows, output = self.run_cli(f"cap-{cap}", [f"synthetic-{i}" for i in range(16)], cap)
                self.assertEqual(code, 0)
                summary = report["summary"]
                self.assertTrue(summary["valid_for_comparison"])
                self.assertEqual(summary["expected_request_count"], 16)
                self.assertEqual(summary["finished_request_count"], 16)
                self.assertEqual(summary["protocol_valid_count"], 16)
                self.assertEqual(summary["tokens"]["completion_tokens"]["known_successful_tokens"], 4096)
                self.assertAlmostEqual(summary["tokens"]["completion_tokens"]["aggregate_tokens_per_second"],
                                       4096 / report["measurement_window_seconds"])
                self.assertLessEqual(report["peak_observed_client_inflight"], cap)
                self.assertEqual(report["peak_observed_client_inflight"], cap)
                self.assertEqual(summary["finish_reasons"], {"length": 16})
                batch = report["batches"][0]
                self.assertTrue(batch["sample_id_closure_valid"])
                self.assertEqual(batch["finished_sample_ids"], [f"sample-{i:02d}" for i in range(16)])
                self.assertGreaterEqual(batch["summary"]["wall_seconds"],
                                        max(row["batch_arrival_to_completion_seconds"] for row in rows))
                later = next(row for row in rows if row["sample_index"] == 15)
                self.assertGreater(later["client_dispatch_wait_seconds"], 0)
                self.assertGreaterEqual(later["batch_arrival_to_completion_seconds"], later["end_to_end_seconds"])
                for index in range(16):
                    self.assertTrue((output / "batch-0000" / f"sample-{index:04d}" / "response.sse").exists())
                    self.assertTrue((output / "batch-0000" / f"sample-{index:04d}" / "throughput-result.json").exists())

    def test_failures_short_output_missing_usage_keep_full_denominator_and_invalidate_comparison(self):
        code, report, rows, _ = self.run_cli("invalid", ["good", "HTTP_ERROR", "SHORT", "MISSING_USAGE"])
        self.assertEqual(code, 1)
        summary = report["summary"]
        self.assertEqual(summary["finished_request_count"], 4)
        self.assertEqual((summary["success_count"], summary["failure_count"]), (3, 1))
        self.assertEqual((summary["protocol_valid_count"], summary["protocol_invalid_count"]), (1, 3))
        self.assertTrue(summary["batch_complete"])
        self.assertFalse(summary["valid_for_comparison"])
        self.assertIsNone(summary["tokens"]["completion_tokens"]["aggregate_tokens_per_second"])
        self.assertEqual(summary["tokens"]["completion_tokens"]["successful_requests_missing_usage"], 1)
        self.assertEqual(summary["latencies"]["end_to_end_seconds"]["count"], 1)
        self.assertEqual(summary["all_request_arrival_to_completion_seconds"]["count"], 4)
        self.assertEqual(report["comparison_status"], "invalid_or_partial")
        failure = next(row for row in rows if row["success"] is False)
        self.assertEqual(failure["http_status"], 500)
        self.assertIn("synthetic failure", failure["error"]["response_body"])

    def test_warmup_full_list_is_excluded_from_measured_window_and_tokens(self):
        code, report, rows, _ = self.run_cli("warmup", ["first", "second"], extra=("--warmup", "1"))
        self.assertEqual(code, 0)
        self.assertEqual(report["warmup_batches"], 1)
        self.assertEqual(len(rows), 4)
        self.assertEqual(report["summary"]["finished_request_count"], 2)
        self.assertEqual(report["summary"]["tokens"]["completion_tokens"]["known_successful_tokens"], 512)
        warmup, measured = report["batches"]
        self.assertEqual(warmup["phase"], "warmup")
        self.assertEqual(measured["phase"], "measured")
        self.assertGreaterEqual(datetime.fromisoformat(report["measurement_started_at"]),
                                datetime.fromisoformat(warmup["ended_at"]))
        self.assertGreaterEqual(report["measurement_window_seconds"], measured["summary"]["wall_seconds"])

    def test_normal_eos_does_not_claim_task_quality_and_flags_length_finishes(self):
        code, report, rows, _ = self.run_cli("normal", ["good", "TRUNCATED"], fixed=False)
        self.assertEqual(code, 0)
        self.assertEqual(report["summary"]["probe_type"], "normal_eos_generation")
        self.assertEqual(report["summary"]["successful_stop_count"], 1)
        self.assertEqual(report["summary"]["successful_length_count"], 1)
        self.assertFalse(report["quality_evaluated"])
        self.assertTrue(all("ignore_eos" not in row["request"] for row in rows))

    def test_incomplete_or_usage_missing_receipts_never_get_primary_rates(self):
        row = {"success": True, "request": {"max_tokens": 256, "ignore_eos": True},
               "finish_reason": "length", "usage": {"prompt_tokens": 10, "completion_tokens": 256}}
        partial = bench.summarize_rows([row], 5.0, 16, True)
        self.assertFalse(partial["batch_complete"])
        self.assertEqual(partial["unrecorded_request_count"], 15)
        self.assertFalse(partial["valid_for_comparison"])
        self.assertIsNone(partial["tokens"]["completion_tokens"]["aggregate_tokens_per_second"])
        for usage in (None, {"prompt_tokens": 10, "completion_tokens": True}):
            result = bench.summarize_rows([{**row, "usage": usage}], 5.0, 1, True)
            self.assertFalse(result["valid_for_comparison"])
            self.assertIsNone(result["tokens"]["completion_tokens"]["aggregate_tokens_per_second"])

    def test_identity_closure_and_warmup_failure_both_gate_comparison_status(self):
        original = bench.base.run_request
        for scenario in ("wrong_identity", "failed_warmup"):
            def injected(args, sample, directory, phase, repetition):
                row = original(args, sample, directory, phase, repetition)
                if scenario == "wrong_identity":
                    row["sample_id"] = "incorrect-id"
                elif phase == "warmup":
                    row["success"] = False
                return row

            with self.subTest(scenario=scenario), patch.object(bench.base, "run_request", injected):
                extra = ("--warmup", "1") if scenario == "failed_warmup" else ()
                code, report, _, _ = self.run_cli(scenario, ["good"], extra=extra)
            self.assertEqual(code, 1)
            self.assertEqual(report["comparison_status"], "invalid_or_partial")
            self.assertFalse(report["summary"]["valid_for_comparison"])
            self.assertIsNone(report["summary"]["tokens"]["completion_tokens"]["aggregate_tokens_per_second"])
            if scenario == "wrong_identity":
                self.assertFalse(report["batches"][0]["sample_id_closure_valid"])
            else:
                self.assertEqual(report["warmup_invalid_batch_count"], 1)


if __name__ == "__main__":
    unittest.main()
