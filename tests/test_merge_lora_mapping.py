"""Synthetic tensor geometries test strict legacy/current PEFT mapping without torch."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/merge_lora_v3.py"
SPEC = importlib.util.spec_from_file_location("public_merge_mapping", SCRIPT)
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)

try:
    import torch
except ImportError:
    torch = None


class MergeMappingTests(unittest.TestCase):
    def setUp(self):
        self.mount = "model.language_model."
        self.target = self.mount + "layers.0.self_attn.q_proj.weight"
        self.parameters = {self.target: SimpleNamespace(shape=(4, 6))}

    def state(self, layout="model"):
        prefix = "base_model.model." + layout + ".layers.0.self_attn.q_proj.lora_"
        return {prefix + "A.weight": SimpleNamespace(shape=(2, 6)),
                prefix + "B.weight": SimpleNamespace(shape=(4, 2))}

    def plan(self, state):
        return merge.plan_adapter_merge(state, self.parameters, self.mount, 2)

    def test_legacy_and_complete_model_prefixes_map_to_identical_base_parameter(self):
        for layout in ("model", "model.language_model"):
            with self.subTest(layout=layout):
                result = self.plan(self.state(layout))
                self.assertEqual(set(result), {self.target})
                self.assertEqual(set(result[self.target]), {"A", "B"})

    def test_cli_requires_explicit_model_adapter_and_output(self):
        with self.assertRaises(SystemExit):
            merge.build_parser().parse_args([])
        args = merge.build_parser().parse_args([
            "--base-model", "/example/base", "--adapter", "/example/adapter", "--output", "/example/new"])
        self.assertEqual((args.base_model, args.adapter, args.output),
                         (Path("/example/base"), Path("/example/adapter"), Path("/example/new")))

    def test_duplicate_canonical_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "重复"):
            self.plan({**self.state(), **self.state("model.language_model")})

    def test_mixed_layout_cannot_join_unrelated_halves(self):
        old = self.state()
        current = self.state("model.language_model")
        mixed = {key: tensor for key, tensor in old.items() if ".lora_A." in key}
        mixed.update({key: tensor for key, tensor in current.items() if ".lora_B." in key})
        with self.assertRaisesRegex(ValueError, "混合"):
            self.plan(mixed)

    def test_unknown_extra_tensors_and_non_text_targets_are_never_ignored(self):
        for key in ("base_model.model.visual.layers.0.q_proj.lora_A.weight",
                    "base_model.model.model.layers.0.self_attn.q_proj.bias",
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight.extra",
                    "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "不支持"):
                self.plan({**self.state(), key: SimpleNamespace(shape=(2, 6))})

    def test_missing_pair_target_and_rank_mismatch_are_rejected(self):
        state = self.state()
        state.pop(next(key for key in state if ".lora_B." in key))
        with self.assertRaisesRegex(ValueError, "配对"):
            self.plan(state)
        with self.assertRaisesRegex(ValueError, "找不到"):
            merge.plan_adapter_merge(self.state(), {}, self.mount, 2)
        with self.assertRaisesRegex(ValueError, "形状"):
            merge.plan_adapter_merge(self.state(), self.parameters, self.mount, 3)
        with self.assertRaisesRegex(ValueError, "为空"):
            self.plan({})


@unittest.skipIf(torch is None, "Optional CPU tensor environment required")
class MergeNumericalTests(unittest.TestCase):
    def test_both_layouts_produce_the_same_matrix_update(self):
        mount = "model.language_model."
        target = mount + "layers.0.mlp.down_proj.weight"
        original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        a, b = torch.ones(2, 4), torch.ones(3, 2)
        for layout in ("model", "model.language_model"):
            with self.subTest(layout=layout):
                prefix = "base_model.model." + layout + ".layers.0.mlp.down_proj.lora_"
                state = {prefix + "A.weight": a, prefix + "B.weight": b}
                parameters = {target: original.clone()}
                pairs = merge.plan_adapter_merge(state, parameters, mount, 2)
                merge.apply_adapter_merge(pairs, parameters, 2.0, torch)
                self.assertTrue(torch.equal(parameters[target], original + 4))

    def test_nan_adapter_and_final_bf16_overflow_are_rejected(self):
        parameters = {"weight": torch.ones((1, 1), dtype=torch.bfloat16)}
        with self.assertRaisesRegex(ValueError, "有限"):
            merge.apply_adapter_merge({"weight": {"A": torch.tensor([[float("nan")]]), "B": torch.ones(1, 1)}},
                                      parameters, 1.0, torch)
        parameters["weight"] = torch.full((1, 1), torch.finfo(torch.bfloat16).max, dtype=torch.bfloat16)
        pair = {"A": torch.tensor([[1e19]]), "B": torch.tensor([[1e19]])}
        with self.assertRaisesRegex(ValueError, "底座参数不是有限值"):
            merge.apply_adapter_merge({"weight": pair}, parameters, 1.0, torch)


if __name__ == "__main__":
    unittest.main()
