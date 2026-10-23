"""Synthetic CPU checks for the PPU weight exporter; no model or dataset access."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    torch = None

prepare = None
if torch is not None:
    SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "prepare_ppu_quantization.py"
    SPEC = importlib.util.spec_from_file_location("prepare_ppu_quantization", SCRIPT)
    prepare = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(prepare)


class QuantizationCommandHelpTests(unittest.TestCase):
    def without_tensor_dependencies(self, arguments):
        script = Path(__file__).resolve().parents[1] / "experiments" / "prepare_ppu_quantization.py"
        # Isolated stdlib-only interpreter: deny these imports even on machines
        # where optional tensor frameworks happen to be installed.
        program = """
import builtins, runpy, sys
original_import = builtins.__import__
def deny_tensor_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'safetensors'}:
        print('TENSOR_IMPORT_ATTEMPT', file=sys.stderr)
        raise ModuleNotFoundError('optional tensor dependency deliberately unavailable', name=name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = deny_tensor_import
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        return subprocess.run([sys.executable, "-I", "-S", "-c", program, str(script), *arguments],
                              text=True, capture_output=True, timeout=15)

    def test_help_does_not_import_torch_or_safetensors(self):
        result = self.without_tensor_dependencies(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--bf16-output", result.stdout)
        self.assertIn("--from-merged", result.stdout)
        self.assertNotIn("TENSOR_IMPORT_ATTEMPT", result.stderr)

    def test_execution_without_dependencies_has_actionable_error_and_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.without_tensor_dependencies([
                "--base", str(root / "base"), "--adapter", str(root / "adapter"),
                "--bf16-output", str(root / "output"), "--bf16-only",
            ])
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("Optional export dependencies", result.stderr)
            self.assertIn("torch", result.stderr)
            self.assertIn("safetensors", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((root / "output").exists())


def make_fixture(root):
    """Use the real 64-layer naming layout, with tiny 4x4 CPU matrices."""
    base, adapter = root / "base", root / "adapter"
    base.mkdir()
    adapter.mkdir()
    tensors, lora = {}, {}
    for layer in range(64):
        suffixes = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        if layer % 4 == 3:
            suffixes += ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
        else:
            suffixes += ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"]
        for suffix in suffixes:
            name = f"model.language_model.layers.{layer}.{suffix}.weight"
            tensors[name] = (torch.arange(16).reshape(4, 4) / 13 - 0.5).to(torch.bfloat16)
            if prepare.LORA_TARGET.fullmatch(name):
                for side, shape in [("A", (16, 4)), ("B", (4, 16))]:
                    key = prepare.LORA_PREFIX + name[:-7] + f".lora_{side}.weight"
                    lora[key] = torch.full(shape, 0.03125, dtype=torch.float32)
        if layer % 4 != 3:
            for suffix in ("in_proj_a", "in_proj_b"):
                tensors[f"model.language_model.layers.{layer}.linear_attn.{suffix}.weight"] = torch.ones(2, 4, dtype=torch.bfloat16)
    tensors["lm_head.weight"] = torch.ones(8, 4, dtype=torch.bfloat16)
    tensors["model.visual.test.weight"] = torch.ones(4, 4, dtype=torch.bfloat16)
    tensors["model.language_model.layers.0.linear_attn.A_log"] = torch.ones(4, dtype=torch.float32)
    save_file(tensors, str(base / "model.safetensors"), metadata={"format": "pt", "custom": "preserve"})
    (base / prepare.INDEX).write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values()),
                     "custom": "index-preserve"},
        "weight_map": {name: "model.safetensors" for name in tensors},
    }))
    (base / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"num_hidden_layers": 64, "hidden_size": 5120},
    }))
    (base / "tokenizer.json").write_text('{"test":1}')
    save_file(lora, str(adapter / "adapter_model.safetensors"), metadata={"format": "pt"})
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 16, "lora_alpha": 32, "bias": "none",
    }))
    return base, adapter, tensors, lora


@unittest.skipUnless(torch is not None, "CPU exporter checks require torch and safetensors")
class PreparePPUQuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.base, cls.adapter, cls.original, cls.lora = make_fixture(cls.root)
        cls.source_hashes = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in (cls.base, cls.adapter) for path in directory.iterdir()
        }
        cls.args = types.SimpleNamespace(threads=2, row_chunk=2, shard_bytes=4096)
        cls.bf16, cls.int8 = cls.root / "bf16", cls.root / "int8"
        with contextlib.redirect_stdout(io.StringIO()):
            prepare.export(cls.base, cls.bf16, "merged_bf16", cls.args, cls.adapter)
            prepare.export(cls.bf16, cls.int8, "w8a8_int8", cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def test_full_layout_merge_quantization_and_source_integrity(self):
        merged_values = {}
        for directory in (self.bf16, self.int8):
            manifest = json.loads((directory / prepare.MANIFEST).read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["lora_coverage"], {"expected": 256, "merged": 256})
            self.assertFalse(manifest["ppu_runtime_validated"])
            self.assertFalse(manifest["quality_validated"])
            index = json.loads((directory / prepare.INDEX).read_text())
            self.assertEqual(index["metadata"]["custom"], "index-preserve")
            self.assertEqual((directory / "tokenizer.json").read_bytes(), (self.base / "tokenizer.json").read_bytes())
            for filename, receipt in manifest["artifacts"].items():
                self.assertEqual(hashlib.sha256((directory / filename).read_bytes()).hexdigest(), receipt["sha256"])
            loaded = {}
            for filename in set(index["weight_map"].values()):
                with safe_open(str(directory / filename), framework="pt", device="cpu") as weights:
                    self.assertEqual(weights.metadata(), {"format": "pt", "custom": "preserve"})
                    for name in weights.keys():
                        loaded[name] = weights.get_tensor(name)
            if directory == self.bf16:
                self.assertEqual(set(loaded), set(self.original))
                self.assertEqual(len(manifest["tensor_statistics"]), 256)
                for name, value in loaded.items():
                    expected = self.original[name]
                    # 2 * B@A = 2 * 16 * (1/32)^2 = 1/32 exactly.
                    if prepare.LORA_TARGET.fullmatch(name):
                        expected = (expected.float() + 0.03125).to(torch.bfloat16)
                    self.assertTrue(torch.equal(value, expected), name)
                merged_values = loaded
            else:
                self.assertEqual(manifest["quantization"]["quantized_matrices"], 400)
                self.assertEqual(len(loaded), len(self.original) + 400)
                self.assertEqual(len(manifest["tensor_statistics"]), 400)
                for name, value in loaded.items():
                    if name.endswith(".weight_scale"):
                        self.assertEqual(value.dtype, torch.float32)
                        self.assertEqual(tuple(value.shape), (4, 1))
                        self.assertTrue(value.gt(0).all())
                    elif prepare.QUANT_TARGET.fullmatch(name):
                        self.assertEqual(value.dtype, torch.int8)
                        scale = loaded[name[:-7] + ".weight_scale"]
                        error = (merged_values[name].float() - value.float() * scale).abs()
                        self.assertTrue((error <= scale / 2 + 1e-6).all(), name)
                    else:
                        self.assertTrue(torch.equal(value, self.original[name]), name)
        for path, digest in self.source_hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_writer_official_reader_metadata_and_oversized_tensor(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            writer = prepare.ShardWriter(directory, 32, {"custom": "index"})
            values = {
                "big": torch.arange(100, dtype=torch.float32),
                "bf16": torch.ones(4, dtype=torch.bfloat16),
                "bool": torch.tensor([True, False]),
                "int8": torch.tensor([-127, 0, 127], dtype=torch.int8),
                "scalar": torch.tensor(3.0),
            }
            with contextlib.redirect_stdout(io.StringIO()):
                for name, value in values.items():
                    writer.add(name, value, {"format": "pt", "extra": "retained"})
                writer.finish()
            self.assertTrue(writer.artifacts["model-00001.safetensors"]["oversized_single_tensor"])
            for name, filename in writer.weight_map.items():
                with safe_open(str(directory / filename), framework="pt", device="cpu") as weights:
                    self.assertTrue(torch.equal(weights.get_tensor(name), values[name]))
                    self.assertEqual(weights.metadata(), {"format": "pt", "extra": "retained"})

    def test_unknown_or_incomplete_lora_is_rejected(self):
        headers = prepare.inspect_source(self.base)[2]
        expected = {name for name in self.original if prepare.LORA_TARGET.fullmatch(name)}
        with tempfile.TemporaryDirectory() as td:
            adapter = Path(td)
            shutil.copy2(self.adapter / "adapter_config.json", adapter / "adapter_config.json")
            bad = dict(self.lora)
            bad["unknown.weight"] = torch.zeros(1)
            save_file(bad, str(adapter / "adapter_model.safetensors"))
            with self.assertRaisesRegex(ValueError, "Unknown adapter tensor"):
                prepare.inspect_adapter(adapter, expected, headers)
            bad = dict(self.lora)
            bad.pop(next(iter(bad)))
            save_file(bad, str(adapter / "adapter_model.safetensors"))
            with self.assertRaisesRegex(ValueError, "256/256"):
                prepare.inspect_adapter(adapter, expected, headers)

    def test_tampered_merged_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            changed = Path(td) / "bf16"
            shutil.copytree(self.bf16, changed)
            (changed / "tokenizer.json").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                prepare.verify_merged(changed)

    def test_failure_receipt_stays_partial(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "failed"
            with mock.patch.object(prepare, "merge_matrix", side_effect=RuntimeError("injected CPU failure")):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "injected CPU failure"):
                        prepare.export(self.base, output, "merged_bf16", self.args, self.adapter)
            receipt = json.loads((output / prepare.MANIFEST).read_text())
            self.assertEqual(receipt["status"], "partial")
            self.assertNotIn("completed_at", receipt)
            self.assertLess(receipt["lora_coverage"]["merged"], 256)
            self.assertEqual(receipt["error"]["type"], "RuntimeError")

    def test_zero_quantization_nonfinite_rejection_and_fused_ignore_names(self):
        q, scale, stats = prepare.quantize_matrix(torch.zeros(3, 4, dtype=torch.bfloat16), 2)
        self.assertTrue(q.eq(0).all())
        self.assertTrue(scale.gt(0).all())
        self.assertEqual(stats["max_abs_error"], 0)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            prepare.quantize_matrix(torch.full((2, 2), float("nan"), dtype=torch.bfloat16), 1)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            prepare.merge_matrix(torch.zeros(2, 2, dtype=torch.bfloat16),
                                 torch.full((16, 2), float("nan")), torch.ones(2, 16), 1)
        ignored = lambda name: any(re.match(pattern[3:], name) if pattern.startswith("re:")
                                   else pattern == name for pattern in prepare.IGNORED_LAYERS)
        for name in ("lm_head", "visual.blocks.0.attn.qkv_proj", "model.layers.0.linear_attn.in_proj_a",
                     "model.layers.0.linear_attn.in_proj_b", "model.layers.0.linear_attn.conv1d"):
            self.assertTrue(ignored(name), name)
        for name in ("model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.gate_proj",
                     "model.layers.0.mlp.up_proj", "model.layers.3.qkv_proj", "model.layers.3.q_proj",
                     "model.layers.3.k_proj", "model.layers.3.v_proj", "model.layers.3.o_proj",
                     "model.layers.0.linear_attn.in_proj_qkv", "model.layers.0.linear_attn.in_proj_z"):
            self.assertFalse(ignored(name), name)


if __name__ == "__main__":
    unittest.main()
