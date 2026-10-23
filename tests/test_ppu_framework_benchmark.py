"""Synthetic local HTTP checks; no models, training corpus, or accelerator needed."""

import contextlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "ppu_framework_benchmark.py"
SPEC = importlib.util.spec_from_file_location("ppu_framework_benchmark", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(request)
        if self.path == "/http-error":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"unsupported option"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(event):
            self.wfile.write(("data: " + json.dumps(event, ensure_ascii=False)
                              + "\r\n\r\n").encode("utf-8"))
            self.wfile.flush()

        emit({"model": "mock", "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        for content in ("桥梁", "检查"):
            emit({"choices": [{"index": 0, "delta": {"content": content}}]})
        if self.path == "/cut-stream":
            return
        if self.path != "/no-finish":
            finish = "length" if request.get("ignore_eos") else "stop"
            emit({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        emit({"choices": [], "usage": {"prompt_tokens": 9,
                                       "completion_tokens": 4, "total_tokens": 13}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class PPUFrameworkBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), MockHandler)
        cls.server.requests = []
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ppu-benchmark-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.samples = self.root / "samples.json"
        self.samples.write_text(json.dumps({
            "dataset_split": "train", "source": "synthetic protocol mock only",
            "samples": [{"id": "mock", "prompt": "Synthetic protocol check"}],
        }), encoding="utf-8")

    def run_cli(self, directory, route="ok", *extra):
        out = self.root / directory
        argv = [str(SCRIPT), "--samples", str(self.samples), "--endpoint",
                f"http://127.0.0.1:{self.server.server_port}/{route}",
                "--model", "mock", "--out-dir", str(out), *extra]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            status = bench.main()
        records = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
        summary = json.loads((out / "summary.json").read_text())
        return status, records, summary, out

    def test_usage_tail_does_not_extend_decode_span_or_ttft(self):
        status, rows, _, out = self.run_cli("success")
        self.assertEqual(status, 0)
        row = rows[0]
        events = [json.loads(line) for line in
                  (out / "request-0000" / "events.jsonl").read_text().splitlines()]
        self.assertTrue(row["success"])
        self.assertEqual(row["content_text"], "桥梁检查")
        self.assertEqual(row["usage"]["completion_tokens"], 4)
        self.assertEqual(row["output_delta_count"], 2)
        # Role metadata, finish packet and trailing usage must not be output tokens.
        first, last = events[1]["elapsed_seconds"], events[2]["elapsed_seconds"]
        self.assertEqual(row["ttft_observed_seconds"], first)
        self.assertEqual(row["last_output_delta_seconds"], last)
        self.assertGreater(events[-1]["elapsed_seconds"], last)
        self.assertEqual(row["observed_output_span_seconds"], last - first)
        self.assertAlmostEqual(row["approx_completion_tokens_per_decode_second"],
                               3 / (last - first))

    def test_http_error_and_incomplete_stream_are_failures(self):
        for route in ("http-error", "cut-stream", "no-finish"):
            with self.subTest(route=route):
                status, rows, summary, _ = self.run_cli(route, route)
                row = rows[0]
                self.assertEqual(status, 1)
                self.assertFalse(row["success"])
                self.assertEqual(summary["successful_requests"], 0)
                self.assertEqual(summary["failed_requests"], 1)
                self.assertEqual(summary["metrics"]["end_to_end_seconds"]["count"], 0)
                if route == "http-error":
                    self.assertEqual(row["http_status"], 400)
                    self.assertIn("unsupported option", row["error"]["response_body"])
                else:
                    self.assertEqual(row["content_text"], "桥梁检查")
                    self.assertIn("Incomplete stream", row["error"]["message"])

    def test_warmup_is_excluded_from_measured_statistics(self):
        status, rows, summary, _ = self.run_cli("warmup", "ok", "--warmup", "1")
        self.assertEqual(status, 0)
        self.assertEqual([row["phase"] for row in rows], ["warmup", "measured"])
        self.assertEqual(summary["warmup_requests"], 1)
        self.assertEqual(summary["measured_requests"], 1)
        self.assertEqual(summary["successful_requests"], 1)
        for metric, aggregate in summary["metrics"].items():
            self.assertEqual(aggregate["count"], 1)
            self.assertEqual(aggregate["median"], rows[1][metric])

    def test_ignore_eos_requires_explicit_flag_and_length_is_not_quality_pass(self):
        status, rows, _, _ = self.run_cli("normal")
        self.assertEqual(status, 0)
        self.assertNotIn("ignore_eos", rows[0]["request"])
        self.assertFalse(rows[0]["generation_hit_limit"])
        status, rows, summary, _ = self.run_cli("fixed", "ok", "--ignore-eos")
        self.assertEqual(status, 0)
        self.assertIs(rows[0]["request"]["ignore_eos"], True)
        self.assertEqual(rows[0]["finish_reason"], "length")
        self.assertTrue(rows[0]["generation_hit_limit"])
        self.assertEqual(summary["generation_hit_limit_count"], 1)
        self.assertFalse(rows[0]["quality_evaluated"])


if __name__ == "__main__":
    unittest.main()
