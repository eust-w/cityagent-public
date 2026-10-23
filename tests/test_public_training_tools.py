"""Public training CLI tests use only synthetic text and lightweight fake tokenizers."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location("public_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load_script("train_lora")
stats = load_script("token_stats")


class SyntheticTokenizer:
    eos_token = "<E>"
    eos_token_id = 9
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return "S:" + messages[0]["content"] + "|U:" + messages[1]["content"] + "|A:"

    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["add_special_tokens"] is False
        return {"input_ids": list(text.encode("utf-8"))}


class PublicTrainingToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}))
        self.sft = self.root / "synthetic.jsonl"
        self.write_rows([self.row("synthetic report A", "2020"), self.row("synthetic report B", "2018")])
        self.output = self.root / "new-run"
        self.argv = ["--model", str(self.model), "--sft", str(self.sft), "--out-dir", str(self.output)]
        tokenizer = SyntheticTokenizer()
        self.fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer))

    def row(self, text, year="2020"):
        return {"instruction": "extract", "input": text, "output": "synthetic answer", "meta": {"year": year}}

    def write_rows(self, rows):
        self.sft.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def test_help_and_import_require_no_third_party_packages(self):
        for filename in (
            "train_lora.py",
            "token_stats.py",
            "merge_lora_v3.py",
        ):
            with self.subTest(filename=filename):
                script = ROOT / "scripts" / filename
                completed = subprocess.run([sys.executable, "-S", str(script), "--help"],
                                           cwd=self.root, text=True, capture_output=True, timeout=10, check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout)
                imported = subprocess.run([
                    sys.executable, "-S", "-c",
                    ("import importlib.util,sys; s=importlib.util.spec_from_file_location('x',sys.argv[1]); "
                    "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                    "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"), str(script)
                ], cwd=self.root, text=True, capture_output=True, timeout=10, check=False)
                self.assertEqual(imported.returncode, 0, imported.stderr)
        self.assertFalse(self.output.exists())

    def test_default_preparation_does_not_load_weights_or_train(self):
        with patch.dict(sys.modules, {"transformers": self.fake_transformers}), patch.object(train, "run_training") as runtime:
            train.main(self.argv)
        runtime.assert_not_called()
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertFalse(manifest["execute_requested"])
        self.assertEqual(manifest["selected_training_samples"], 1)
        self.assertEqual({row["status"] for row in manifest["audit"]}, {"selected", "held_out"})
        self.assertIsNone(manifest["resumed_checkpoint"])
        self.assertFalse(manifest["truncation"])
        self.assertFalse(manifest["quality_validated"])
        self.assertFalse(manifest["ppu_runtime_validated"])
        self.assertFalse((self.output / ".training_incomplete").exists())
        self.assertNotIn("synthetic report A", json.dumps(manifest))
        self.assertNotIn("synthetic answer", json.dumps(manifest))

    def test_existing_output_is_rejected_before_optional_dependencies(self):
        self.output.mkdir()
        with patch.dict(sys.modules, {"transformers": None}), self.assertRaises(SystemExit):
            train.main(self.argv)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_execute_failure_keeps_incomplete_receipt_without_success(self):
        with patch.dict(sys.modules, {"transformers": self.fake_transformers}), \
                patch.object(train, "run_training", side_effect=RuntimeError("synthetic failure")) as runtime, \
                self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            train.main(self.argv + ["--execute"])
        runtime.assert_called_once()
        self.assertTrue((self.output / ".training_incomplete").is_file())
        self.assertTrue((self.output / "manifest.json").is_file())
        self.assertFalse((self.output / "summary.json").exists())
        self.assertFalse((self.output / "adapter").exists())

    def test_modes_require_explicit_small_step_budget_and_new_paths(self):
        for mode in ("smoke", "benchmark"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "max-steps"):
                    train.validate_args(train.build_parser().parse_args(self.argv + ["--mode", mode]))
                with self.assertRaisesRegex(ValueError, "max-steps"):
                    train.validate_args(train.build_parser().parse_args(self.argv + ["--mode", mode, "--max-steps", "101"]))
                train.validate_args(train.build_parser().parse_args(self.argv + ["--mode", mode, "--max-steps", "2"]))
        args = train.build_parser().parse_args(self.argv)
        args.out_dir = self.model / "new-output"
        with self.assertRaisesRegex(ValueError, "overlap"):
            train.validate_args(args)

    def test_model_class_never_automatically_falls_back(self):
        with self.assertRaisesRegex(ValueError, "model-class mismatch"):
            train.validate_args(train.build_parser().parse_args(self.argv + ["--model-class", "causal-lm"]))
        (self.model / "config.json").write_text(json.dumps({"architectures": ["UnknownModel"]}))
        with self.assertRaisesRegex(ValueError, "Unsupported architecture"):
            train.validate_args(train.build_parser().parse_args(self.argv))

    def test_tied_embedding_aliases_are_explicitly_unsupported(self):
        (self.model / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {"tie_word_embeddings": True}}))
        with self.assertRaisesRegex(ValueError, "Tied embeddings"):
            train.validate_args(train.build_parser().parse_args(self.argv))

    def test_nonfinite_loss_or_adapter_cannot_be_published_as_completed(self):
        key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        finite = SimpleNamespace(requires_grad=True, finite=True)
        bad = SimpleNamespace(requires_grad=True, finite=False)
        self.assertEqual(train.validate_training_result(0.5, [(key, finite)], lambda p: p.finite), 1)
        for loss in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "loss must be finite"):
                train.validate_training_result(loss, [(key, finite)], lambda p: p.finite)
        with self.assertRaisesRegex(ValueError, "finite LoRA"):
            train.validate_training_result(0.5, [(key, bad)], lambda p: p.finite)
        with self.assertRaisesRegex(ValueError, "finite LoRA"):
            train.validate_training_result(0.5, [("lm_head.weight", finite)], lambda p: p.finite)
        with self.assertRaisesRegex(ValueError, "No trainable"):
            train.validate_training_result(0.5, [], lambda p: p.finite)

    def test_overlong_training_is_recorded_without_truncation_or_holdout_leakage(self):
        self.write_rows([self.row("short"), self.row("z" * 1000), self.row("held out", "2018")])
        args = train.build_parser().parse_args(self.argv + ["--max-length", "100"])
        selected, audit = train.prepare_data(args, SyntheticTokenizer())
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["year"], "2020")
        self.assertEqual({row["status"] for row in audit}, {"selected", "held_out", "overlong_full_sequence"})
        self.assertTrue(next(row for row in audit if row["status"] == "overlong_full_sequence")["tokens"] > 1000)
        self.assertEqual(selected[0]["input_ids"][-3:], list(b"<E>"))
        self.assertEqual(selected[0]["labels"][:selected[0]["prompt_tokens"]], [-100] * selected[0]["prompt_tokens"])

    def test_all_heldout_or_oversized_samples_refuse_training(self):
        self.write_rows([self.row("held out", "2018")])
        args = train.build_parser().parse_args(self.argv)
        with self.assertRaisesRegex(ValueError, "No complete non-holdout"):
            train.prepare_data(args, SyntheticTokenizer())
        self.assertFalse(self.output.exists())

    def test_duplicate_prompts_and_missing_year_fail_closed(self):
        self.write_rows([self.row("same", "2018"), self.row("same", "2020")])
        with self.assertRaisesRegex(ValueError, "duplicate prompt"):
            train.prepare_data(train.build_parser().parse_args(self.argv), SyntheticTokenizer())
        self.write_rows([{"instruction": "i", "input": "synthetic", "output": "answer", "bridge_entity": "fictional", "split": "train"}])
        with self.assertRaisesRegex(ValueError, "meta.year"):
            train.prepare_data(train.build_parser().parse_args(self.argv), SyntheticTokenizer())

    def test_zero_pad_id_and_prompt_masks_are_retained(self):
        actual = train.pad_batch([{"input_ids": [1, 2], "labels": [-100, 2]},
                                  {"input_ids": [3], "labels": [3]}], 0)
        self.assertEqual(actual["input_ids"], [[1, 2], [3, 0]])
        self.assertEqual(actual["labels"], [[-100, 2], [3, -100]])
        self.assertEqual(actual["attention_mask"], [[1, 1], [1, 0]])

    def test_token_stats_accept_schema_without_year_and_use_real_eos(self):
        self.write_rows([{"instruction": "i", "input": "abc", "output": "de"}])
        rows = stats.read_samples(self.sft)
        result = stats.summarize(rows, SyntheticTokenizer(), [100, 1, 100])
        self.assertEqual(result["answer_including_eos"]["min"], 5)
        self.assertEqual(result["coverage"], [{"max_length": 1, "complete_samples": 0, "total_samples": 1},
                                             {"max_length": 100, "complete_samples": 1, "total_samples": 1}])
        self.assertIsNone(result["speed_estimate"])
        self.assertNotIn("abc", json.dumps(result))
        for invalid in ([], [0], [-1]):
            with self.assertRaises(ValueError):
                stats.summarize(rows, SyntheticTokenizer(), invalid)

    def test_token_stats_never_overwrite_input_or_existing_output(self):
        output = self.root / "lengths.json"
        argv = ["--model", str(self.model), "--sft", str(self.sft), "--output", str(output)]
        with patch.dict(sys.modules, {"transformers": self.fake_transformers}):
            stats.main(argv)
            first = output.read_bytes()
            with self.assertRaises(SystemExit):
                stats.main(argv)
        self.assertEqual(output.read_bytes(), first)
        result = json.loads(first)
        self.assertEqual(len(result["sft_sha256"]), 64)
        self.assertNotIn("synthetic report", first.decode())


if __name__ == "__main__":
    unittest.main()
