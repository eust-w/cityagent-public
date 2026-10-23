"""Synthetic-only invariants for epoch coverage, distributed tail loss and export."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import types
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_lora_nvidia.py"
SPEC = importlib.util.spec_from_file_location("train_lora_nvidia", SCRIPT)
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


class CharTokenizer:
    eos_token = "!"

    def apply_chat_template(self, messages, **kwargs):
        return "S" + messages[0]["content"] + "U" + messages[1]["content"] + "A"

    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": list(map(ord, text))}


class FakeParameter:
    def __init__(self, shape=(2, 3), trainable=True):
        self.shape, self.requires_grad = shape, trainable
        self.gathers = self.cpu_copies = 0

    def detach(self):
        return self

    def full_tensor(self):
        self.gathers += 1
        return self

    def to(self, device):
        assert device == "cpu"
        self.cpu_copies += 1
        return self

    def contiguous(self):
        return self


class FakeModel:
    def __init__(self, pairs):
        self.pairs = pairs

    def named_parameters(self):
        return iter(self.pairs)


class FormalTrainingTests(unittest.TestCase):
    def test_350_rows_each_once_with_six_active_tail_and_two_zero_weight_slots(self):
        schedules = [train.epoch_schedule(350, 8, 42, epoch) for epoch in (1, 2)]
        for batches in schedules:
            active = [slot["index"] for batch in batches for slot in batch if slot["weight"]]
            self.assertEqual(len(batches), 44)
            self.assertEqual(sorted(active), list(range(350)))
            self.assertEqual(sum(slot["weight"] for slot in batches[-1]), 6)
            self.assertEqual(sum(1 - slot["weight"] for batch in batches for slot in batch), 2)
            self.assertTrue(all(len(batch) == 8 for batch in batches))
        self.assertNotEqual(schedules[0], schedules[1])
        self.assertEqual(schedules[0], train.epoch_schedule(350, 8, 42, 1))

    def test_tail_token_mean_and_gradient_are_independent_of_world_size(self):
        slots = train.epoch_schedule(6, 8, 42, 1)[0]
        answers = [3, 17, 2, 11, 29, 5]
        gradients = [0.2, -0.8, 1.1, 0.05, 0.3, -0.1]
        global_answers = sum(answers)
        reference = sum(a * g for a, g in zip(answers, gradients)) / global_answers
        for world in (1, 2, 4, 8):
            rank_gradients, counted_samples, counted_answers = [], 0, 0
            for rank in range(world):
                local = train.rank_indices(slots, rank, world)
                rank_gradients.append(sum(gradients[s["index"]] * answers[s["index"]]
                                          * s["weight"] * world / global_answers for s in local))
                counted_samples += sum(s["weight"] for s in local)
                counted_answers += sum(answers[s["index"]] * s["weight"] for s in local)
            self.assertAlmostEqual(sum(rank_gradients) / world, reference)
            self.assertEqual(counted_samples, 6)
            self.assertEqual(counted_answers, global_answers)

    def test_one_row_and_smaller_than_world_still_have_nonzero_denominator(self):
        batch = train.epoch_schedule(1, 8, 0, 1)[0]
        self.assertEqual(sum(s["weight"] for s in batch), 1)
        self.assertEqual(sum(not s["weight"] for s in batch), 7)
        for rank in range(8):
            self.assertEqual(len(train.rank_indices(batch, rank, 8)), 1)

    def test_positive_lr_for_every_update_and_cosine_decay_after_warmup(self):
        rates = [train.learning_rate_for_step(step, 88, 1e-4, 0.03) for step in range(1, 89)]
        self.assertAlmostEqual(rates[0], 1e-4 / 3)
        self.assertAlmostEqual(rates[2], 1e-4)
        self.assertTrue(all(math.isfinite(rate) and 0 < rate <= 1e-4 for rate in rates))
        self.assertTrue(all(a >= b for a, b in zip(rates[3:], rates[4:])))
        self.assertEqual(train.learning_rate_for_step(1, 1, 1e-4, 0.03), 1e-4)

    def test_default_limits_do_not_reuse_benchmark_subset(self):
        args = train.parser().parse_args(["--model", "/m", "--sft", "/s", "--out-dir", "/o"])
        self.assertIsNone(args.max_samples)
        self.assertIsNone(args.eval_samples)
        self.assertIsNone(args.max_steps)
        self.assertEqual((args.epochs, args.global_batch_size, args.min_length, args.max_length), (2, 8, 1, 32768))
        self.assertTrue(args.activation_offload)

    def test_full_holdout_and_overlong_rows_are_isolated_without_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft.jsonl"
            rows = [{"instruction": "i", "input": f"case{index}", "output": "answer",
                     "meta": {"year": "2019" if index < 70 else "2018"}} for index in range(80)]
            rows.append({"instruction": "i", "input": "overlong", "output": "X" * 300, "meta": {"year": "2019"}})
            rows.append({"instruction": "i", "input": "heldoutoverlong", "output": "X" * 300, "meta": {"year": "2018"}})
            path.write_text("\n".join(json.dumps(row) for row in rows))
            args = types.SimpleNamespace(sft=path, min_length=1, max_length=100,
                                         max_samples=None, eval_samples=None, seed=42)
            actual_train, holdout, audit = train.prepare_data(args, CharTokenizer())
            self.assertEqual((len(actual_train), len(holdout)), (70, 10))
            self.assertEqual({row["year"] for row in actual_train}, {"2019"})
            self.assertEqual({row["year"] for row in holdout}, {"2018"})
            self.assertEqual(sum(row["status"] == "overlong_full_sequence" for row in audit), 2)
            self.assertFalse(any("input_ids" in row or "input" in row or "output" in row for row in audit))

    def test_adapter_export_gathers_only_trainable_keys_and_retains_wrapper_names(self):
        canonical = "base_model.model.layers.0.q_proj.lora_A.default.weight"
        wrapped = canonical.replace("layers.0.", "layers.0._checkpoint_wrapped_module._checkpoint_wrapped_module.")
        adapter, frozen = FakeParameter(), FakeParameter((1000, 1000), trainable=False)
        model = FakeModel([(wrapped, adapter), ("base_model.lm_head.weight", frozen)])
        state = train.gather_adapter_cpu(model, rank=0, expected={canonical: [2, 3]})
        self.assertEqual(list(state), [wrapped])
        self.assertEqual((adapter.gathers, adapter.cpu_copies, frozen.gathers), (1, 1, 0))
        self.assertEqual(train.gather_adapter_cpu(model, rank=1, expected={canonical: [2, 3]}), {})
        self.assertEqual((adapter.gathers, adapter.cpu_copies), (2, 1))

    def test_adapter_inventory_rejects_missing_shapes_collisions_and_non_lora(self):
        key = "base_model.model.q_proj.lora_B.default.weight"
        model = FakeModel([(key, FakeParameter())])
        with self.assertRaisesRegex(RuntimeError, "keys/shapes"):
            train.adapter_parameter_map(model, {key: [3, 2]})
        with self.assertRaisesRegex(RuntimeError, "collision"):
            train.adapter_parameter_map(FakeModel([(key, FakeParameter()),
                (key.replace("q_proj.", "q_proj._checkpoint_wrapped_module."), FakeParameter())]))
        with self.assertRaisesRegex(RuntimeError, "non-LoRA"):
            train.adapter_parameter_map(FakeModel([("base_model.lm_head.weight", FakeParameter())]))

    def test_output_reuse_and_invalid_lr_fail_without_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, sft, out = root / "model", root / "sft.jsonl", root / "out"
            model.mkdir(); sft.write_text("{}"); out.mkdir()
            args = train.parser().parse_args(["--model", str(model), "--sft", str(sft), "--out-dir", str(out)])
            with self.assertRaises(FileExistsError):
                train.validate_args(args)
            args.out_dir = root / "new"
            args.learning_rate = float("nan")
            with self.assertRaisesRegex(ValueError, "learning rate"):
                train.validate_args(args)


if __name__ == "__main__":
    unittest.main()
