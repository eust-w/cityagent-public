"""Synthetic-only checks for benchmark selection; no model/data dependencies."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_lora.py"
SPEC = importlib.util.spec_from_file_location("benchmark_lora", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class CharTokenizer:
    eos_token = "!"

    def apply_chat_template(self, messages, **kwargs):
        return "S" + messages[0]["content"] + "U" + messages[1]["content"] + "A"

    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": list(map(ord, text))}


def row(index, year="2019", input_length=3, answer_length=2):
    content = {"instruction": "I", "input": str(index) * input_length, "output": "X" * answer_length}
    return {**content, "year": year, "line": index + 1,
            "sample_sha256": bench.digest(content), "prompt_sha256": bench.digest(content["input"])}


class BenchmarkPreparationTests(unittest.TestCase):
    def test_activation_offload_is_opt_in_and_rejects_qlora(self):
        common = ["--model", "/unused", "--sft", "/unused", "--out-dir", "/unused"]
        for mode in ("qlora", "bf16-sharded"):
            baseline = bench.parser().parse_args(common + ["--mode", mode])
            self.assertFalse(baseline.activation_offload)
        incompatible = bench.parser().parse_args(common + ["--mode", "qlora", "--activation-offload"])
        with self.assertRaisesRegex(ValueError, "only supported with --mode bf16-sharded"):
            bench.validate_paths_and_args(incompatible)

    def test_activation_offload_accepts_fsdp_without_changing_sample_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            model, sft = base / "model", base / "data.jsonl"
            model.mkdir()
            config = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}
            (model / "config.json").write_text(json.dumps(config))
            sft.write_text("{}")
            common = ["--model", str(model), "--sft", str(sft), "--out-dir", str(base / "new-run"),
                      "--mode", "bf16-sharded", "--length-bucket", "32k"]
            baseline = bench.parser().parse_args(common)
            offload = bench.parser().parse_args(common + ["--activation-offload"])
            self.assertEqual(bench.validate_paths_and_args(offload), config)
            self.assertEqual(bench.length_bounds(offload), bench.length_bounds(baseline))
            before, after = vars(baseline).copy(), vars(offload).copy()
            before.pop("activation_offload")
            after.pop("activation_offload")
            self.assertEqual(before, after)

    def test_loading_audit_does_not_broadly_ignore_mtp_or_missing_main_layers(self):
        result = bench.audit_checkpoint_keys({"main", "mtp.fc.weight", "mtp.unrecognized.weight"},
                                             {"main", "visual.missing.weight"})
        self.assertEqual(result["unused_auxiliary_mtp_keys"], ["mtp.fc.weight"])
        self.assertEqual(result["unexpected_keys"], ["mtp.unrecognized.weight"])
        self.assertEqual(result["missing_keys"], ["visual.missing.weight"])

    def test_complete_answer_and_prompt_are_preserved(self):
        source = row(1, input_length=90, answer_length=70)
        item = bench.tokenize_row(source, CharTokenizer())
        self.assertEqual(item["input_ids"], list(map(ord, "SIU" + "1" * 90 + "A" + "X" * 70 + "!")))
        self.assertEqual(item["labels"][:item["prompt_tokens"]], [-100] * item["prompt_tokens"])
        self.assertEqual(item["labels"][item["prompt_tokens"]:], list(map(ord, "X" * 70 + "!")))

    def test_skips_whole_overlong_row_and_never_trains_holdout(self):
        rows = [row(1), row(2, year="2018"), row(3, answer_length=100), row(4, input_length=100)]
        train, holdout, audit = bench.prepare_rows(rows, CharTokenizer(), min_length=1, max_length=20,
                                                   max_samples=64, eval_samples=4, seed=42)
        self.assertEqual([item["year"] for item in train], ["2019"])
        self.assertEqual([item["year"] for item in holdout], ["2018"])
        self.assertEqual(sum(item["status"] == "overlong_full_sequence" for item in audit), 2)
        self.assertFalse(any(key in item for item in audit for key in ("input", "output", "input_ids", "labels")))

    def test_selection_is_stable_across_file_order(self):
        rows = [row(index) for index in range(1, 10)] + [row(11, "2018"), row(12, "2018")]
        kwargs = dict(min_length=1, max_length=100, max_samples=4, eval_samples=1, seed=42)
        first = bench.prepare_rows(rows, CharTokenizer(), **kwargs)
        second = bench.prepare_rows(rows[::-1], CharTokenizer(), **kwargs)
        for a, b in zip(first[:2], second[:2]):
            self.assertEqual([r["sample_sha256"] for r in a], [r["sample_sha256"] for r in b])

    def test_world_sizes_process_identical_global_batches_with_no_extra_padding(self):
        schedule = bench.make_schedule(7, 6, 8)
        for step in schedule:
            for world in (1, 2, 4, 8):
                split = [bench.rank_indices(step, rank, world) for rank in range(world)]
                reconstructed = [split[i % world][i // world] for i in range(8)]
                self.assertEqual(reconstructed, step)
        with self.assertRaises(ValueError):
            bench.rank_indices(schedule[0], 0, 3)

    def test_ambiguous_year_and_duplicate_prompt_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.jsonl"
            data = {"instruction": "i", "input": "synthetic", "output": "x", "meta": {"year": None}}
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "meta.year"):
                bench.load_rows(path)
            data["meta"]["year"] = "2018年"
            other = {**data, "meta": {"year": "2019年"}, "output": "different"}
            path.write_text(json.dumps(data) + "\n" + json.dumps(other))
            with self.assertRaisesRegex(ValueError, "duplicate prompt"):
                bench.load_rows(path)

    def test_output_reuse_is_rejected_before_model_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            model, out, sft = base / "model", base / "old-run", base / "data.jsonl"
            model.mkdir()
            out.mkdir()
            sft.write_text("{}")
            args = bench.parser().parse_args(["--model", str(model), "--sft", str(sft),
                                             "--out-dir", str(out), "--mode", "qlora"])
            with self.assertRaises(FileExistsError):
                bench.validate_paths_and_args(args)

    def test_bucket_rejects_empty_or_conflicting_selection(self):
        args = bench.parser().parse_args(["--model", "/m", "--sft", "/s", "--out-dir", "/o",
                                         "--mode", "qlora", "--length-bucket", "16k", "--max-length", "8192"])
        with self.assertRaises(ValueError):
            bench.length_bounds(args)
        with self.assertRaisesRegex(ValueError, "No complete non-2018"):
            bench.prepare_rows([row(1)], CharTokenizer(), min_length=6144, max_length=8192,
                               max_samples=64, eval_samples=0, seed=42)


if __name__ == "__main__":
    unittest.main()
