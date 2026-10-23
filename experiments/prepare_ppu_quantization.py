#!/usr/bin/env python3
"""CPU-only, bounded-memory export of the audited Qwen 27B + epoch-002 LoRA.

This produces experimental weights, not a claim of PPU runtime compatibility.
No dataset is opened. Existing source/output directories are never overwritten.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
from datetime import datetime, timezone

# Optional tensor frameworks are loaded only when export operations begin.
# Importing this module and showing --help work with the Python standard library.
torch = None
safe_open = None


MANIFEST = "conversion_manifest.json"
INDEX = "model.safetensors.index.json"
LORA_PREFIX = "base_model.model."
LORA_TARGET = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:mlp\.(?:gate_proj|up_proj|down_proj)|self_attn\.(?:q_proj|k_proj|v_proj|o_proj))\.weight$"
)
QUANT_TARGET = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:mlp\.(?:gate_proj|up_proj|down_proj)|self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj))\.weight$"
)
# Runtime names, after model.language_model -> model and self_attn removal.
# None of these patterns matches qkv_proj/gate_up_proj or any fused shard.
IGNORED_LAYERS = [
    "lm_head", "re:^visual\\.",
    "re:.*\\.linear_attn\\.(?:conv1d|in_proj_a|in_proj_b)$",
]
DTYPES = {}


def load_tensor_dependencies():
    """Initialize the CPU exporter without changing vendor inference environments."""
    global torch, safe_open, DTYPES
    if torch is not None and safe_open is not None:
        return
    try:
        import torch as tensor_runtime
        from safetensors import safe_open as open_tensors
    except ImportError as error:
        raise RuntimeError(
            "Optional export dependencies are unavailable: torch and safetensors are required. "
            "Install compatible CPU packages in a separate export environment; "
            "do not replace vendor PPU dependencies. See docs/quantization.md."
        ) from error
    torch, safe_open = tensor_runtime, open_tensors
    DTYPES = {
        torch.bool: "BOOL", torch.uint8: "U8", torch.int8: "I8",
        torch.int16: "I16", torch.int32: "I32", torch.int64: "I64",
        torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32",
        torch.float64: "F64",
    }


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value):
    temp = path.with_name(path.name + ".partial")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def read_header(path: Path):
    with path.open("rb") as f:
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        length = struct.unpack("<Q", length_bytes)[0]
        if length > 100 * 1024 * 1024:
            raise ValueError(f"Unexpected safetensors header size: {path}")
        return json.loads(f.read(length))


def inspect_source(base: Path):
    config = json.loads((base / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("Only the audited Qwen3_5ForConditionalGeneration layout is supported")
    text = config["text_config"]
    if text.get("num_hidden_layers") != 64 or text.get("hidden_size") != 5120:
        raise ValueError("Expected the audited 64-layer, hidden_size=5120 model config")
    if config.get("quantization_config") or text.get("quantization_config"):
        raise ValueError("Source must be unquantized BF16")
    index = json.loads((base / INDEX).read_text())
    weight_map = index["weight_map"]
    headers, source_metadata = {}, {}
    for filename in sorted(set(weight_map.values())):
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"Unsafe or unsupported shard name: {filename}")
        header = read_header(base / filename)
        source_metadata[filename] = header.pop("__metadata__", None)
        for name, info in header.items():
            if name in headers or weight_map.get(name) != filename:
                raise ValueError(f"Duplicate/unindexed tensor: {name}")
            headers[name] = info
    if set(headers) != set(weight_map):
        raise ValueError("Index and shard tensor names differ")
    quant_names = {k for k in headers if QUANT_TARGET.fullmatch(k)}
    lora_names = {k for k in headers if LORA_TARGET.fullmatch(k)}
    if len(quant_names) != 400 or len(lora_names) != 256:
        raise ValueError(f"Expected 400 quantization / 256 LoRA targets, got {len(quant_names)}/{len(lora_names)}")
    for name in quant_names:
        info = headers[name]
        if info["dtype"] != "BF16" or len(info["shape"]) != 2:
            raise ValueError(f"Expected BF16 matrix: {name}")
    return config, index, headers, source_metadata, quant_names, lora_names


def inspect_adapter(adapter: Path, expected_names, headers):
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    cfg = json.loads(config_path.read_text())
    if (cfg.get("peft_type") != "LORA" or cfg.get("r") != 16
            or cfg.get("lora_alpha") != 32 or cfg.get("bias") != "none"):
        raise ValueError("Expected the audited LoRA r=16, alpha=32, bias=none adapter")
    for key in ("use_dora", "use_rslora", "fan_in_fan_out", "lora_bias",
                "rank_pattern", "alpha_pattern", "modules_to_save", "layer_replication",
                "target_parameters", "alora_invocation_tokens", "use_bdlora"):
        if cfg.get(key):
            raise ValueError(f"Unsupported adapter option: {key}")
    header = read_header(weights_path)
    header.pop("__metadata__", None)
    pairs = {}
    for name, info in header.items():
        if not name.startswith(LORA_PREFIX):
            raise ValueError(f"Unknown adapter tensor: {name}")
        stripped = name[len(LORA_PREFIX):]
        match = re.fullmatch(r"(.+)\.lora_([AB])\.weight", stripped)
        if not match:
            raise ValueError(f"Unknown adapter tensor: {name}")
        target, side = match[1] + ".weight", match[2]
        if target not in expected_names or side in pairs.setdefault(target, {}):
            raise ValueError(f"Unexpected/duplicate adapter target: {name}")
        rows, cols = headers[target]["shape"]
        expected = [16, cols] if side == "A" else [rows, 16]
        if info["shape"] != expected or info["dtype"] != "F32":
            raise ValueError(f"Unexpected adapter shape/dtype: {name}")
        pairs[target][side] = name
    if set(pairs) != expected_names or any(set(p) != {"A", "B"} for p in pairs.values()):
        raise ValueError("Adapter coverage must be exactly 256/256 A+B pairs")
    return pairs, {"config_sha256": sha256(config_path), "weights_sha256": sha256(weights_path),
                   "path": str(adapter), "rank": 16, "alpha": 32, "scale": 2.0}


class ShardWriter:
    """At most one bounded shard plus a single oversized tensor in memory.

    Write safetensors payloads directly from CPU storage, avoiding save_file's
    possible extra Python bytes copies of an entire shard.
    """
    def __init__(self, directory, max_bytes, index_metadata):
        load_tensor_dependencies()
        self.directory, self.max_bytes = directory, max_bytes
        self.index_metadata = dict(index_metadata)
        self.pending, self.pending_bytes, self.metadata = [], 0, None
        self.weight_map, self.artifacts = {}, {}
        self.number, self.total_bytes = 0, 0

    def add(self, name, tensor, metadata):
        if name in self.weight_map or any(k == name for k, _ in self.pending):
            raise ValueError(f"Duplicate output tensor: {name}")
        if tensor.device.type != "cpu" or tensor.dtype not in DTYPES or not tensor.is_contiguous():
            raise ValueError(f"Unsupported output tensor storage: {name}")
        size = tensor.numel() * tensor.element_size()
        if self.pending and (self.pending_bytes + size > self.max_bytes or metadata != self.metadata):
            self.flush()
        self.metadata = metadata
        self.pending.append((name, tensor))
        self.pending_bytes += size
        if self.pending_bytes >= self.max_bytes:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        self.number += 1
        filename = f"model-{self.number:05d}.safetensors"
        target = self.directory / filename
        header, offset = {}, 0
        if self.metadata is not None:
            header["__metadata__"] = self.metadata
        for name, tensor in self.pending:
            end = offset + tensor.numel() * tensor.element_size()
            header[name] = {"dtype": DTYPES[tensor.dtype], "shape": list(tensor.shape),
                            "data_offsets": [offset, end]}
            offset = end
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * ((-len(encoded)) % 8)
        h = hashlib.sha256()
        with target.with_suffix(".partial").open("xb") as f:
            for block in (struct.pack("<Q", len(encoded)), encoded):
                f.write(block)
                h.update(block)
            for _, tensor in self.pending:
                payload = memoryview(tensor.detach().reshape(-1).view(torch.uint8).numpy()).cast("B")
                f.write(payload)
                h.update(payload)
                del payload
            f.flush()
            os.fsync(f.fileno())
        target.with_suffix(".partial").rename(target)
        for name, _ in self.pending:
            self.weight_map[name] = filename
        self.artifacts[filename] = {"sha256": h.hexdigest(), "bytes": target.stat().st_size,
                                    "tensor_bytes": offset, "tensor_count": len(self.pending),
                                    "oversized_single_tensor": len(self.pending) == 1 and offset > self.max_bytes}
        self.total_bytes += offset
        self.pending.clear()
        self.pending_bytes = 0
        gc.collect()
        print(json.dumps({"event": "shard_written", "file": str(target), "tensor_bytes": offset}), flush=True)

    def finish(self):
        self.flush()
        metadata = dict(self.index_metadata)
        metadata["total_size"] = self.total_bytes
        write_json(self.directory / INDEX, {"metadata": metadata, "weight_map": self.weight_map})


def merge_matrix(weight, a, b, row_chunk):
    load_tensor_dependencies()
    merged = torch.empty_like(weight, device="cpu")
    stats = {"max_abs_bf16_rounding": 0.0, "sum_abs_bf16_rounding": 0.0,
             "max_abs_lora_delta": 0.0, "elements": weight.numel()}
    for start in range(0, weight.shape[0], row_chunk):
        stop = min(start + row_chunk, weight.shape[0])
        exact = weight[start:stop].float()
        delta = torch.mm(b[start:stop], a).mul_(2.0)
        if not torch.isfinite(delta).all():
            raise ValueError("Non-finite LoRA delta")
        stats["max_abs_lora_delta"] = max(stats["max_abs_lora_delta"], delta.abs().max().item())
        exact.add_(delta)
        rounded = exact.to(torch.bfloat16)
        if not torch.isfinite(rounded).all():
            raise ValueError("Non-finite merged BF16 weight")
        difference = (exact - rounded.float()).abs()
        stats["max_abs_bf16_rounding"] = max(stats["max_abs_bf16_rounding"], difference.max().item())
        stats["sum_abs_bf16_rounding"] += difference.sum(dtype=torch.float64).item()
        merged[start:stop].copy_(rounded)
    stats["mean_abs_bf16_rounding"] = stats.pop("sum_abs_bf16_rounding") / stats["elements"]
    return merged, stats


def quantize_matrix(weight, row_chunk):
    load_tensor_dependencies()
    quantized = torch.empty(weight.shape, dtype=torch.int8, device="cpu")
    scales = torch.empty((weight.shape[0], 1), dtype=torch.float32, device="cpu")
    stats = {"max_abs_error": 0.0, "sum_abs_error": 0.0, "elements": weight.numel()}
    for start in range(0, weight.shape[0], row_chunk):
        stop = min(start + row_chunk, weight.shape[0])
        values = weight[start:stop].float()
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite quantization input")
        scale = values.abs().amax(dim=1, keepdim=True).clamp_min_(1e-10).div_(127.0)
        q = (values / scale).round().clamp_(-127, 127).to(torch.int8)
        quantized[start:stop].copy_(q)
        scales[start:stop].copy_(scale)
        error = (values - q.float() * scale).abs()
        stats["max_abs_error"] = max(stats["max_abs_error"], error.max().item())
        stats["sum_abs_error"] += error.sum(dtype=torch.float64).item()
    stats["mean_abs_error"] = stats.pop("sum_abs_error") / stats["elements"]
    return quantized, scales, stats


def copy_auxiliary(source, output):
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(p.startswith(".") for p in relative.parts):
            continue
        if path.is_dir():
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing auxiliary symlink: {path}")
        if path.name in (INDEX, MANIFEST) or path.suffix in (".safetensors", ".bin", ".pt", ".pth", ".ckpt"):
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def verify_merged(base):
    manifest = json.loads((base / MANIFEST).read_text())
    if (manifest.get("status") != "completed" or manifest.get("mode") != "merged_bf16"
            or manifest.get("lora_coverage") != {"expected": 256, "merged": 256}):
        raise ValueError("--from-merged requires a completed audited BF16 export")
    for filename, info in manifest["artifacts"].items():
        path = base / filename
        if path.resolve().parent != base.resolve() and not path.resolve().is_relative_to(base.resolve()):
            raise ValueError(f"Unsafe manifest artifact: {filename}")
        if not path.is_file() or sha256(path) != info["sha256"]:
            raise ValueError(f"Merged artifact hash mismatch: {filename}")
    return manifest


def export(source, output, mode, args, adapter=None):
    load_tensor_dependencies()
    config, index, headers, source_metadata, quant_names, lora_names = inspect_source(source)
    pairs, adapter_receipt = ({}, None)
    parent_manifest = None
    if mode == "merged_bf16":
        pairs, adapter_receipt = inspect_adapter(adapter, lora_names, headers)
    else:
        parent_manifest = verify_merged(source)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "status": "partial", "mode": mode, "started_at": now(),
        "source": str(source), "source_config_sha256": sha256(source / "config.json"),
        "source_index_sha256": sha256(source / INDEX), "adapter": adapter_receipt,
        "source_safetensors_metadata": source_metadata, "threads": args.threads,
        "row_chunk": args.row_chunk, "max_shard_bytes": args.shard_bytes,
        "artifacts": {}, "tensor_statistics": {},
        "lora_coverage": {"expected": 256, "merged": 0},
        "ppu_runtime_validated": False, "quality_validated": False,
        "limitations": ["Offline export is not runtime or accuracy validation.",
                       "Merged BF16 rounds the LoRA sum; dynamic LoRA is a separate baseline."],
    }
    if mode == "w8a8_int8":
        manifest.update({"merged_manifest_sha256": sha256(source / MANIFEST),
                         "original_adapter": parent_manifest["adapter"],
                         "lora_coverage": parent_manifest["lora_coverage"],
                         "quantization": {"scheme": "symmetric per-output-channel W8 / dynamic per-token A8",
                                          "quantized_matrices_expected": 400, "quantized_matrices": 0,
                                          "source_values": "completed merged BF16 checkpoint",
                                          "calibration_data_used": False, "ignored_layers": IGNORED_LAYERS}})
    write_json(output / MANIFEST, manifest)
    writer = ShardWriter(output, args.shard_bytes, index.get("metadata", {}))
    try:
        copy_auxiliary(source, output)
        adapter_context = safe_open(str(adapter / "adapter_model.safetensors"), framework="pt", device="cpu") if adapter else None
        if adapter_context:
            adapter_context.__enter__()
        try:
            for filename in sorted(set(index["weight_map"].values())):
                with safe_open(str(source / filename), framework="pt", device="cpu") as weights:
                    for name in sorted(weights.keys()):
                        value = weights.get_tensor(name)
                        if mode == "merged_bf16" and name in pairs:
                            pair = pairs[name]
                            value, stats = merge_matrix(value, adapter_context.get_tensor(pair["A"]),
                                                        adapter_context.get_tensor(pair["B"]), args.row_chunk)
                            manifest["tensor_statistics"][name] = stats
                            manifest["lora_coverage"]["merged"] += 1
                        if mode == "w8a8_int8" and name in quant_names:
                            value, scale, stats = quantize_matrix(value, args.row_chunk)
                            manifest["tensor_statistics"][name] = stats
                            manifest["quantization"]["quantized_matrices"] += 1
                            writer.add(name, value, source_metadata[filename])
                            writer.add(name[:-7] + ".weight_scale", scale, source_metadata[filename])
                            del scale
                        else:
                            writer.add(name, value, source_metadata[filename])
                        del value
                writer.flush()
                manifest["artifacts"] = writer.artifacts
                write_json(output / MANIFEST, manifest)
        finally:
            if adapter_context:
                adapter_context.__exit__(None, None, None)
        if manifest["lora_coverage"]["merged"] != 256:
            raise ValueError("Incomplete LoRA coverage")
        if mode == "w8a8_int8":
            if manifest["quantization"]["quantized_matrices"] != 400:
                raise ValueError("Incomplete quantization coverage")
            config["quantization_config"] = {"quant_method": "w8a8_int8", "activation_scheme": "dynamic",
                                               "is_dynamic": True, "ignored_layers": IGNORED_LAYERS}
            write_json(output / "config.json", config)
        writer.finish()
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != MANIFEST:
                relative = str(path.relative_to(output))
                if relative not in writer.artifacts:
                    writer.artifacts[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        manifest["artifacts"] = writer.artifacts
        manifest.update({"status": "completed", "completed_at": now(),
                         "output_tensor_count": len(writer.weight_map), "output_tensor_bytes": writer.total_bytes})
        write_json(output / MANIFEST, manifest)
        print(json.dumps({"event": "completed", "mode": mode, "directory": str(output)}), flush=True)
    except BaseException as exc:
        manifest["status"] = "partial"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc), "at": now()}
        manifest["artifacts"] = writer.artifacts
        write_json(output / MANIFEST, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--bf16-output", type=Path)
    parser.add_argument("--int8-output", type=Path)
    parser.add_argument("--bf16-only", action="store_true")
    parser.add_argument("--from-merged", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--row-chunk", type=int, default=512)
    parser.add_argument("--shard-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()
    if min(args.threads, args.row_chunk, args.shard_bytes) <= 0:
        parser.error("Thread/chunk/shard limits must be positive")
    if args.from_merged:
        if args.base or args.adapter or args.bf16_output or args.bf16_only or not args.int8_output:
            parser.error("--from-merged requires only --int8-output plus resource options")
        sources, outputs = [args.from_merged], [args.int8_output]
    else:
        if not (args.base and args.adapter and args.bf16_output):
            parser.error("Provide --base, --adapter and --bf16-output")
        if args.bf16_only == bool(args.int8_output):
            parser.error("Choose --bf16-only or provide --int8-output")
        sources = [args.base, args.adapter]
        outputs = [p for p in (args.bf16_output, args.int8_output) if p]
    resolved = [p.resolve() for p in outputs]
    for out in resolved:
        if out.exists() or any(out.is_relative_to(src.resolve()) for src in sources):
            parser.error(f"Output must be a fresh directory outside every source: {out}")
    if len(resolved) != len(set(resolved)) or any(a != b and a.is_relative_to(b) for a in resolved for b in resolved):
        parser.error("Outputs must be distinct non-nested directories")
    try:
        load_tensor_dependencies()
    except RuntimeError as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.from_merged:
        export(args.from_merged.resolve(), args.int8_output.resolve(), "w8a8_int8", args)
    else:
        export(args.base.resolve(), args.bf16_output.resolve(), "merged_bf16", args, args.adapter.resolve())
        if not args.bf16_only:
            export(args.bf16_output.resolve(), args.int8_output.resolve(), "w8a8_int8", args)


if __name__ == "__main__":
    main()
