"""Strict offline comparison fixtures; no server, real dataset, or model."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "summarize_ppu_throughput.py"
SPEC = importlib.util.spec_from_file_location("summarize_ppu_throughput", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def fixture(root, name, *, label="sglang", concurrency=4, walls=(2.0,), seed=0, model="same-model",
            reverse=False, prompt_delta=0, output_tokens=256, max_tokens=256):
    directory = root / name
    directory.mkdir()
    samples = [{"id": f"s-{i:02d}", "messages": [{"role": "user", "content": f"synthetic-{i}"}]} for i in range(16)]
    if reverse:
        samples.reverse()
    frozen = {"dataset_split": "train", "source": "synthetic-fixture", "samples": samples}
    manifest = {"arguments": {"label": label, "concurrency": concurrency, "max_tokens": max_tokens,
                              "ignore_eos": True, "repetitions": len(walls), "model": model},
                "sample_ids_in_order": [sample["id"] for sample in samples],
                "frozen_samples_sha256": summary.digest(frozen), "script_sha256": "measurement-code",
                "base_script_sha256": "base-code"}
    batches, rows = [], []
    for repetition, wall in enumerate(walls):
        batch_id = f"batch-{repetition:04d}"
        batches.append({"batch_id": batch_id, "phase": "measured", "concurrency": concurrency,
                        "monotonic_start_seconds": 100.0, "monotonic_end_seconds": 100.0 + wall,
                        "peak_observed_client_inflight": concurrency, "summary": {"wall_seconds": wall}})
        for index, sample in enumerate(samples):
            request = {"model": model, "messages": sample["messages"], "temperature": 0,
                       "seed": seed, "max_tokens": max_tokens, "ignore_eos": True,
                       "chat_template_kwargs": {"enable_thinking": False}, "stream": True}
            rows.append({"batch_id": batch_id, "phase": "measured", "sample_index": index,
                         "sample_id": sample["id"], "request": request,
                         "request_sha256": summary.digest(request),
                         "messages_sha256": summary.digest(request["messages"]),
                         "success": True, "done_received": True, "finish_reason": "length",
                         "usage": {"prompt_tokens": 10 + int(sample["id"][2:]) + prompt_delta,
                                   "completion_tokens": output_tokens},
                         "ttft_observed_seconds": .1, "end_to_end_seconds": .5,
                         "client_dispatch_wait_seconds": .01,
                         "batch_arrival_to_completion_seconds": .75})
    write(directory / "manifest.json", manifest)
    write(directory / "frozen-samples.json", frozen)
    write(directory / "summary.json", {"batches": batches})
    (directory / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return directory


class PPUThroughputSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ppu-throughput-summary-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_repeat_medians_use_each_round_rate_and_exclude_32_token_warmup(self):
        repeated = fixture(self.root, "sg-repeat", walls=(2.0, 8.0))
        other = fixture(self.root, "vllm", label="vllm", concurrency=1, walls=(4.0,))
        warmup = fixture(self.root, "warmup", max_tokens=32, output_tokens=32)
        report = summary.compare([repeated, other, warmup])
        self.assertTrue(report["cross_round_comparison_valid"])
        self.assertTrue(report["runs"][2]["excluded"])
        group = next(group for group in report["repeat_groups"] if group["label"] == "sglang")
        self.assertEqual(group["valid_round_count"], 2)
        self.assertEqual(group["aggregate_completion_tokens_per_second"],
                         {"count": 2, "median": 1280.0, "min": 512.0, "max": 2048.0})
        self.assertNotEqual(group["aggregate_completion_tokens_per_second"]["median"], 8192 / 10)
        self.assertEqual(group["whole_round_wall_seconds"]["median"], 5.0)
        self.assertTrue(all(row["completion_tokens"] == 4096 for row in report["runs"][0]["rounds"]))
        self.assertIn("固定生成 256 tokens", summary.markdown(report))

    def test_cross_round_request_model_input_order_and_prompt_counts_must_match(self):
        baseline = fixture(self.root, "baseline")
        cases = {"seed": {"seed": 7}, "model": {"model": "different-model"},
                 "order": {"reverse": True}, "prompt-count": {"prompt_delta": 1}}
        for name, changes in cases.items():
            with self.subTest(name=name):
                candidate = fixture(self.root, name, **changes)
                report = summary.compare([baseline, candidate])
                self.assertFalse(report["cross_round_comparison_valid"])
                self.assertFalse(report["ranking_permitted"])
                row = report["runs"][1]["rounds"][0]
                self.assertFalse(row["valid_for_comparison"])
                self.assertIsNone(row["aggregate_completion_tokens_per_second"])
                self.assertTrue(any("cross_round" in reason for reason in row["invalid_reasons"]))

    def test_short_output_duplicate_ids_bad_hash_or_wrong_wall_never_pass(self):
        cases = ("short", "duplicate-id", "bad-hash", "bad-wall", "missing-usage")
        for name in cases:
            with self.subTest(name=name):
                directory = fixture(self.root, name, output_tokens=255 if name == "short" else 256)
                journal = directory / "results.jsonl"
                rows = [json.loads(line) for line in journal.read_text().splitlines()]
                if name == "duplicate-id":
                    rows[-1]["sample_id"] = rows[0]["sample_id"]
                elif name == "bad-hash":
                    rows[0]["request_sha256"] = "fake"
                elif name == "bad-wall":
                    report = summary.load_json(directory / "summary.json")
                    report["batches"][0]["summary"]["wall_seconds"] = 1.0
                    write(directory / "summary.json", report)
                elif name == "missing-usage":
                    rows[0]["usage"] = None
                journal.write_text("\n".join(json.dumps(row) for row in rows))
                report = summary.compare([directory])
                self.assertFalse(report["cross_round_comparison_valid"])
                self.assertIsNone(report["runs"][0]["rounds"][0]["aggregate_completion_tokens_per_second"])

    def test_interrupted_run_without_final_summary_is_not_a_comparable_round(self):
        baseline = fixture(self.root, "baseline")
        interrupted = fixture(self.root, "interrupted")
        (interrupted / "summary.json").unlink()
        report = summary.compare([baseline, interrupted])
        self.assertFalse(report["cross_round_comparison_valid"])
        self.assertEqual(report["runs"][1]["rounds"], [])
        self.assertTrue(report["runs"][1]["errors"])


if __name__ == "__main__":
    unittest.main()
