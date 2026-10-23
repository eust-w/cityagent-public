#!/usr/bin/env python3
"""Bounded, reproducible Qwen3.5 LoRA hardware benchmark (never resumes/saves weights).

Examples (each output directory must be NEW):
  python scripts/benchmark_lora.py --model /models/Qwen --sft /data/new/sft_train.jsonl \
      --out-dir /runs/qlora8k --mode qlora --length-bucket 8k
  torchrun --standalone --nproc_per_node=4 scripts/benchmark_lora.py \
      --model /models/Qwen --sft /data/new/sft_train.jsonl --out-dir /runs/bf16-4gpu \
      --mode bf16-sharded --length-bucket 8k

Uses Accelerate FSDP2 (BF16) or DDP (NF4 QLoRA), and Liger fused linear CE.
Optional --activation-offload uses FSDP checkpoint-input CPU offload; it is off by default.
Full prompts and answers are retained; overlong rows are skipped, never truncated.
2018 rows are held out. Their teacher-forced loss is a numerical check, not quality.
Only hashes/lengths/metrics are written; neither SFT text nor checkpoints are copied.
Heavy dependencies are imported only by main/runtime functions.

API references:
  https://huggingface.co/docs/accelerate/en/usage_guides/fsdp
  https://huggingface.co/docs/accelerate/en/concept_guides/fsdp1_vs_fsdp2
  https://huggingface.co/docs/peft/en/developer_guides/quantization
  https://github.com/linkedin/Liger-Kernel/blob/v0.8.2/src/liger_kernel/transformers/model/qwen3_5.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time
from functools import partial
from datetime import datetime, timezone
from typing import Any


BUCKETS = {"8k": (6144, 8192), "16k": (12288, 16384), "32k": (24576, 32768)}
TEXT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# HF's Qwen3_5PreTrainedModel ignores ^mtp.*. Enumerate the known auxiliary head
# instead of silently inheriting that broad exclusion. Main causal CE does not use MTP.
UNUSED_MTP_KEYS = {
    "mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight", "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight", "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.self_attn.k_norm.weight", "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight", "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.v_proj.weight",
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def audit_checkpoint_keys(checkpoint_keys: set[str], model_keys: set[str]) -> dict:
    extra = checkpoint_keys - model_keys
    unused = extra & UNUSED_MTP_KEYS
    return {"checkpoint_key_count": len(checkpoint_keys), "model_key_count": len(model_keys),
            "unused_auxiliary_mtp_keys": sorted(unused),
            "unused_mtp_explanation": "Official Qwen3.5 conditional forward implements main next-token CE, not the auxiliary MTP training head.",
            "unexpected_keys": sorted(extra - UNUSED_MTP_KEYS),
            "missing_keys": sorted(model_keys - checkpoint_keys)}


def load_rows(path: Path) -> list[dict]:
    """Validate explicit SFT input; ambiguous years and duplicate prompts fail closed."""
    rows, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("instruction", "input", "output")
            ):
                raise ValueError(f"SFT line {line_no}: nonempty instruction/input/output required")
            meta = row.get("meta")
            year = str(meta.get("year", "")) if isinstance(meta, dict) else ""
            if not re.fullmatch(r"(?:19|20)\d{2}年?", year):
                raise ValueError(f"SFT line {line_no}: explicit unambiguous meta.year required")
            content = {key: row[key] for key in ("instruction", "input", "output")}
            prompt_hash = digest({key: row[key] for key in ("instruction", "input")})
            if prompt_hash in seen:
                raise ValueError(f"SFT line {line_no}: duplicate prompt (possible split leakage)")
            seen.add(prompt_hash)
            rows.append({**content, "year": year.removesuffix("年"), "line": line_no,
                         "sample_sha256": digest(content), "prompt_sha256": prompt_hash})
    if not rows:
        raise ValueError("SFT is empty")
    return rows


def tokenize_row(row: dict, tokenizer: Any) -> dict:
    """Qwen chat prompt + every answer token + EOS; no truncation or head/tail slicing."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": row["instruction"]},
         {"role": "user", "content": row["input"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not tokenizer.eos_token:
        raise ValueError("Tokenizer has no EOS token")
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    answer_ids = tokenizer(row["output"] + tokenizer.eos_token,
                           add_special_tokens=False, truncation=False)["input_ids"]
    if not prompt_ids or not answer_ids:
        raise ValueError(f"Empty token sequence at SFT line {row['line']}")
    ids = prompt_ids + answer_ids
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + answer_ids,
            "sample_sha256": row["sample_sha256"], "prompt_sha256": row["prompt_sha256"],
            "token_sha256": digest(ids), "line": row["line"], "year": row["year"],
            "tokens": len(ids), "prompt_tokens": len(prompt_ids),
            "answer_tokens": len(answer_ids)}


def prepare_rows(rows: list[dict], tokenizer: Any, *, min_length: int, max_length: int,
                 max_samples: int, eval_samples: int, seed: int) -> tuple[list, list, list]:
    candidates = {"train": [], "holdout": []}
    audit = []
    for row in rows:
        sample = tokenize_row(row, tokenizer)
        split = "holdout" if row["year"] == "2018" else "train"
        record = {key: value for key, value in sample.items() if key not in ("input_ids", "labels")}
        record["split"] = split
        if sample["tokens"] > max_length:
            record["status"] = "overlong_full_sequence"
        elif sample["tokens"] < min_length:
            record["status"] = "below_length_bucket"
        else:
            record["status"] = "eligible"
            candidates[split].append(sample)
        audit.append(record)
    # Stable across file ordering, world sizes and modes; hashes identify the exact content.
    for split, limit in (("train", max_samples), ("holdout", eval_samples)):
        candidates[split].sort(key=lambda item: item["sample_sha256"])
        random.Random(f"{seed}:{split}").shuffle(candidates[split])
        candidates[split] = candidates[split][:limit]
    selected = {sample["sample_sha256"] for group in candidates.values() for sample in group}
    for record in audit:
        if record["status"] == "eligible":
            record["status"] = "selected" if record["sample_sha256"] in selected else "sample_limit"
    if not candidates["train"]:
        raise ValueError("No complete non-2018 training samples fit this length range")
    if eval_samples and not candidates["holdout"]:
        raise ValueError("No complete 2018 holdout samples fit this length range; change range or explicitly --eval-samples 0")
    return candidates["train"], candidates["holdout"], audit


def make_schedule(sample_count: int, max_steps: int, global_batch_size: int) -> list[list[int]]:
    if min(sample_count, max_steps, global_batch_size) <= 0:
        raise ValueError("schedule dimensions must be positive")
    return [[(step * global_batch_size + offset) % sample_count
             for offset in range(global_batch_size)] for step in range(max_steps)]


def rank_indices(step_indices: list[int], rank: int, world_size: int) -> list[int]:
    if world_size < 1 or not 0 <= rank < world_size or len(step_indices) % world_size:
        raise ValueError("global batch must be divisible by world size")
    return step_indices[rank::world_size]


def length_bounds(args: argparse.Namespace) -> tuple[int, int]:
    if args.length_bucket:
        bounds = BUCKETS[args.length_bucket]
        if args.max_length is not None and args.max_length != bounds[1]:
            raise ValueError("--max-length conflicts with --length-bucket")
        if args.min_length is not None and args.min_length != bounds[0]:
            raise ValueError("--min-length conflicts with --length-bucket")
        return bounds
    bounds = (args.min_length or 1, args.max_length or 8192)
    if not 1 <= bounds[0] <= bounds[1]:
        raise ValueError("Require 1 <= min-length <= max-length")
    return bounds


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--sft", type=Path, required=True)
    result.add_argument("--out-dir", type=Path, required=True)
    result.add_argument("--mode", choices=("qlora", "bf16-sharded"), required=True)
    result.add_argument("--activation-offload", action="store_true",
                        help="Offload checkpoint inputs to CPU (bf16-sharded only; default: disabled)")
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--max-steps", type=int, default=6)
    result.add_argument("--warmup-steps", type=int, default=2,
                        help="Exclude initial steps from steady-state throughput; LR is constant")
    result.add_argument("--global-batch-size", type=int, default=8)
    result.add_argument("--max-length", type=int)
    result.add_argument("--min-length", type=int)
    result.add_argument("--length-bucket", choices=BUCKETS)
    result.add_argument("--max-samples", type=int, default=64)
    result.add_argument("--eval-samples", type=int, default=4)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--lora-rank", type=int, default=16)
    result.add_argument("--lora-alpha", type=int, default=32)
    result.add_argument("--lora-dropout", type=float, default=0.05)
    result.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    return result


def versions() -> dict:
    import platform
    values = {"python": platform.python_version()}
    for package in ("torch", "transformers", "accelerate", "peft", "bitsandbytes",
                    "liger-kernel", "triton", "flash-linear-attention", "causal-conv1d"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = None
    return values


def validate_paths_and_args(args: argparse.Namespace) -> dict:
    if args.activation_offload and args.mode != "bf16-sharded":
        raise ValueError("--activation-offload is only supported with --mode bf16-sharded")
    if not args.model.is_dir() or not args.sft.is_file():
        raise ValueError("--model must be a local model directory and --sft an existing JSONL file")
    if args.out_dir.exists():
        raise FileExistsError(f"Output already exists; choose a NEW directory: {args.out_dir}")
    if min(args.max_steps, args.max_samples, args.global_batch_size, args.lora_rank, args.lora_alpha) < 1:
        raise ValueError("steps/samples/batch/LoRA dimensions must be positive")
    if not 0 <= args.warmup_steps < args.max_steps or args.eval_samples < 0:
        raise ValueError("Require 0 <= warmup-steps < max-steps and eval-samples >= 0")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 <= args.lora_dropout < 1:
        raise ValueError("Invalid learning rate or LoRA dropout")
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    expected = ["Qwen3_5ForConditionalGeneration"]
    if config.get("model_type") != "qwen3_5" or config.get("architectures") != expected:
        raise ValueError(f"This benchmark requires explicit full Qwen3_5ForConditionalGeneration; got {config.get('architectures')}")
    return config


def run_training(args: argparse.Namespace, train: list, holdout: list, schedule: list,
                 manifest: dict) -> None:
    import torch
    import transformers
    from accelerate import Accelerator, FullyShardedDataParallelPlugin
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin, set_seed
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
    from packaging.version import Version
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    for package, minimum in (("torch", "2.9"), ("transformers", "5.18"),
                             ("accelerate", "1.15"), ("peft", "0.21.2"), ("liger-kernel", "0.8.4")):
        if Version(importlib.metadata.version(package)) < Version(minimum):
            raise RuntimeError(f"Require {package}>={minimum} for the verified API combination")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    sharded = args.mode == "bf16-sharded"
    if sharded and int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise ValueError("bf16-sharded requires torchrun with multiple GPUs")
    # Initialize Accelerate BEFORE from_pretrained so HF can load on rank 0 only.
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2, reshard_after_forward=True, cpu_ram_efficient_loading=True,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"],
        activation_checkpointing=True,
        activation_checkpointing_offload=args.activation_offload,
    ) if sharded else None
    accelerator = Accelerator(mixed_precision="bf16", fsdp_plugin=plugin,
                              gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=1),
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
    rank, world = accelerator.process_index, accelerator.num_processes
    rank_indices(schedule[0], rank, world)
    if sharded and accelerator.distributed_type.name != "FSDP":
        raise RuntimeError("FSDP was requested but not activated")
    if accelerator.gradient_accumulation_steps != 1:
        raise RuntimeError("Unexpected extra gradient-accumulation loss scaling")
    if world > 1:
        fingerprints = [None] * world
        fingerprint = {key: manifest[key] for key in ("sft_sha256", "config_sha256", "script_sha256", "schedule_sha256")}
        torch.distributed.all_gather_object(fingerprints, fingerprint)
        if any(value != fingerprint for value in fingerprints):
            raise RuntimeError("Ranks disagree on input/config/script/sample schedule")
    set_seed(args.seed)
    # Output reservation is collective so an existing path fails every rank cleanly.
    status = [None]
    if rank == 0:
        try:
            args.out_dir.mkdir(parents=True, exist_ok=False)
            write_json(args.out_dir / "manifest.json", manifest)
        except Exception as error:
            status[0] = f"{type(error).__name__}: {error}"
    if world > 1:
        torch.distributed.broadcast_object_list(status, src=0)
    if status[0]:
        raise RuntimeError(status[0])
    accelerator.wait_for_everyone()
    runtime = {"rank": rank, "world_size": world, "versions": versions(),
               "gpu": torch.cuda.get_device_name(accelerator.device), "cuda": torch.version.cuda,
               "distributed_type": accelerator.distributed_type.name,
               "gradient_sync": "every_microbatch", "per_device_batch_size": 1,
               "microbatches_per_optimizer_step": args.global_batch_size // world}
    write_json(args.out_dir / f"runtime-rank{rank}.json", runtime)

    def record_loading_resources(phase, loaded_model):
        torch.cuda.synchronize()
        groups = {}
        for parameter in loaded_model.parameters():
            local = parameter.detach()
            if hasattr(local, "to_local"):
                local = local.to_local()
            key = (str(local.device), str(local.dtype), type(parameter).__name__)
            group = groups.setdefault(key, {"device": key[0], "dtype": key[1], "parameter_class": key[2],
                                            "tensor_count": 0, "tensor_numel": 0, "tensor_bytes": 0})
            group["tensor_count"] += 1
            group["tensor_numel"] += local.numel()
            group["tensor_bytes"] += local.numel() * local.element_size()
        write_json(args.out_dir / f"loading-resource-rank{rank}-{phase}.json", {
            "phase": phase, "rank": rank, "parameter_tensors": list(groups.values()),
            "numel_interpretation": "DTensor counts use local shards. Params4bit counts describe packed tensors, not original dense parameter counts. CPU/meta tensor bytes are geometry, not measured RSS.",
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()})

    # Patch only the loss: no architecture dropping, alternative targets, or full logits.
    apply_liger_kernel_to_qwen3_5(rope=False, rms_norm=False, swiglu=False,
                                  cross_entropy=False, fused_linear_cross_entropy=True)
    model_class = getattr(transformers, "Qwen3_5ForConditionalGeneration")
    kwargs = dict(local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
                  attn_implementation=args.attn_implementation, output_loading_info=True)
    if not sharded:
        kwargs.update(device_map={"": accelerator.local_process_index},
                      quantization_config=transformers.BitsAndBytesConfig(
                          load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model, loading = model_class.from_pretrained(str(args.model), **kwargs)
    record_loading_resources("after-from-pretrained", model)
    index_path = args.model / "model.safetensors.index.json"
    if index_path.is_file():
        checkpoint_keys = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"])
    else:
        from safetensors import safe_open
        with safe_open(str(args.model / "model.safetensors"), framework="pt", device="cpu") as checkpoint:
            checkpoint_keys = set(checkpoint.keys())
    # Quantized state_dict adds absmax/quantization buffers that are not checkpoint
    # parameters. HF loading info covers buffers; compare original parameter names.
    key_audit = audit_checkpoint_keys(checkpoint_keys,
                                    {name for name, _ in model.named_parameters(remove_duplicate=False)})
    problems = {key: loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading.get(key)}
    write_json(args.out_dir / f"loading-rank{rank}.json", {
        "class": type(model).__name__, "problems": problems, "checkpoint_key_audit": key_audit})
    if problems or key_audit["missing_keys"] or key_audit["unexpected_keys"]:
        raise RuntimeError("Model keys did not load strictly; see loading metadata")
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    if not sharded:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
        # PEFT casts all nonquantized weights to FP32. At 248k vocab, these two
        # frozen matrices alone would waste ~5 GiB compared with the BF16 base.
        model.get_input_embeddings().to(dtype=torch.bfloat16)
        model.get_output_embeddings().to(dtype=torch.bfloat16)
    # Restrict adapters to the language layers, keeping the visual tower frozen and loaded.
    target_pattern = r"model\.language_model\.layers\.\d+\..*\.(:TARGETS:)".replace(
        ":TARGETS:", "|".join(TEXT_TARGETS))
    targeted = [name for name, _ in model.named_modules() if re.fullmatch(target_pattern, name)]
    if not targeted:
        raise RuntimeError("No expected language projection modules matched")
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_pattern, task_type="CAUSAL_LM"))
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if rank == 0:
        write_json(args.out_dir / "adapter-config.json", {
            "trainable_parameters": trainable, "target_modules": targeted,
            "text_projection_suffixes": TEXT_TARGETS,
            "note": "Same seven projection suffixes as original LoRA; hybrid linear-attention projections are not added."})
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=0.0)
    record_loading_resources("before-prepare", model)
    model, optimizer = accelerator.prepare(model, optimizer)
    record_loading_resources("after-prepare", model)
    head = None
    if sharded:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
        from torch.distributed.fsdp import FSDPModule
        checkpoint_wrappers = 0
        for module in model.modules():
            if isinstance(module, CheckpointWrapper):
                # Accelerate 1.15 defaults to False. LoRA dropout requires the
                # recomputation to use the SAME masks as the original forward.
                module.checkpoint_fn = partial(module.checkpoint_fn, preserve_rng_state=True)
                checkpoint_wrappers += 1
        if not checkpoint_wrappers:
            raise RuntimeError("Expected FSDP activation-checkpoint wrappers were not installed")
        head = model.get_output_embeddings()
        if not isinstance(head, FSDPModule) or any(parameter.requires_grad for parameter in head.parameters()):
            raise RuntimeError("Expected a frozen FSDP2 output head for fused CE")
        write_json(args.out_dir / f"fsdp-rank{rank}.json", {
            "activation_checkpoint_wrappers": checkpoint_wrappers, "preserve_rng_state": True,
            "activation_checkpointing_offload": args.activation_offload,
            "frozen_lm_head": "Explicit unshard before fused CE; reshard after backward/evaluation."})
    model.train()
    step_records = []
    log_path = args.out_dir / f"steps-rank{rank}.jsonl"

    def batch(sample):
        return {"input_ids": torch.tensor([sample["input_ids"]], device=accelerator.device),
                "labels": torch.tensor([sample["labels"]], device=accelerator.device),
                "attention_mask": torch.ones((1, sample["tokens"]), dtype=torch.long, device=accelerator.device)}

    def scalar_loss(inputs):
        if head is not None:
            # Accelerate 1.15 shards lm_head separately, while Liger accesses
            # head.weight directly and never calls head.forward (hence no FSDP
            # pre-forward hook). Use the public FSDP2 weight lifecycle explicitly.
            head.unshard()
        with accelerator.autocast():
            output = model(**inputs, use_cache=False, skip_logits=True)
        if output.logits is not None:
            raise RuntimeError("Fused loss did not suppress logits; refusing a different memory benchmark")
        return output.loss

    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        for step, indices in enumerate(schedule, 1):
            local = rank_indices(indices, rank, world)
            global_answers = sum(train[index]["answer_tokens"] for index in indices)
            optimizer.zero_grad(set_to_none=True)
            accelerator.wait_for_everyone()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss_sum = torch.zeros((), dtype=torch.float64, device=accelerator.device)
            local_tokens = local_answers = 0
            for index in local:
                sample = train[index]
                inputs = batch(sample)
                loss = scalar_loss(inputs)
                # DDP/FSDP average across ranks. This scaling gives the full global
                # minibatch's token-mean CE, independent of GPU count/accumulation.
                accelerator.backward(loss * (sample["answer_tokens"] * world / global_answers))
                if head is not None:
                    head.reshard()
                loss_sum += loss.detach().double() * sample["answer_tokens"]
                local_tokens += int(inputs["attention_mask"].sum().item())
                local_answers += int(inputs["labels"][:, 1:].ne(-100).sum().item())
                del inputs, loss
            grad_square = torch.zeros((), dtype=torch.float64, device=accelerator.device)
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    gradient = parameter.grad.detach()
                    if hasattr(gradient, "to_local"):
                        gradient = gradient.to_local()
                    grad_square += gradient.float().square().sum().double()
            if world > 1:
                torch.distributed.all_reduce(grad_square)
                if not sharded:  # DDP has a complete, replicated averaged gradient.
                    grad_square /= world
            gradient_l2 = math.sqrt(grad_square.item())
            if not math.isfinite(gradient_l2) or gradient_l2 <= 0:
                raise RuntimeError("Adapter gradients are non-finite or all zero")
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            local_record = {"step": step, "warmup": step <= args.warmup_steps,
                            "seconds": seconds, "nonpad_tokens": local_tokens,
                            "answer_tokens": local_answers, "loss_sum": loss_sum.item(),
                            "global_gradient_l2": gradient_l2,
                            "sample_sha256": [train[index]["sample_sha256"] for index in local],
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            log.write(json.dumps(local_record, allow_nan=False) + "\n")
            gathered = [None] * world
            if world > 1:
                torch.distributed.all_gather_object(gathered, local_record)
            else:
                gathered[0] = local_record
            answers = sum(record["answer_tokens"] for record in gathered)
            record = {"step": step, "warmup": step <= args.warmup_steps,
                      "seconds": max(record["seconds"] for record in gathered),
                      "nonpad_tokens": sum(record["nonpad_tokens"] for record in gathered),
                      "answer_tokens": answers,
                      "global_gradient_l2": gradient_l2,
                      "token_mean_loss": sum(record["loss_sum"] for record in gathered) / answers,
                      "peak_allocated_bytes_per_rank": [record["peak_allocated_bytes"] for record in gathered],
                      "peak_reserved_bytes_per_rank": [record["peak_reserved_bytes"] for record in gathered]}
            if not math.isfinite(record["token_mean_loss"]):
                raise RuntimeError("Non-finite loss; benchmark failed")
            step_records.append(record)
            if rank == 0:
                print(json.dumps(record, allow_nan=False), flush=True)
    # All FSDP ranks execute the same held-out rows, preserving collective order.
    # Duplicate rank evaluation is counted once, and dropout is disabled.
    model.eval()
    eval_records = []
    with torch.no_grad():
        for sample in holdout:
            loss = scalar_loss(batch(sample)).item()
            if head is not None:
                head.reshard()
            if not math.isfinite(loss):
                raise RuntimeError("Non-finite held-out teacher-forced loss")
            eval_records.append({"sample_sha256": sample["sample_sha256"],
                                 "answer_tokens": sample["answer_tokens"], "loss": loss})
    accelerator.wait_for_everyone()
    if rank == 0:
        measured = [record for record in step_records if not record["warmup"]]
        elapsed = sum(record["seconds"] for record in measured)
        summary = {"status": "completed", "mode": args.mode, "world_size": world,
                   "manifest_sha256": file_digest(args.out_dir / "manifest.json"),
                   "steps": step_records, "measured_steps": len(measured),
                   "measured_seconds": elapsed,
                   "nonpad_tokens_per_second": sum(record["nonpad_tokens"] for record in measured) / elapsed,
                   "answer_tokens_per_second": sum(record["answer_tokens"] for record in measured) / elapsed,
                   "median_step_seconds": statistics.median(record["seconds"] for record in measured),
                   "heldout_teacher_forced": {"year": "2018", "samples": eval_records,
                       "token_mean_loss": (sum(record["loss"] * record["answer_tokens"] for record in eval_records)
                                           / sum(record["answer_tokens"] for record in eval_records)) if eval_records else None,
                       "interpretation": "Numerical sanity check only; not a report-quality or task-score evaluation."},
                   "timing_scope": "GPU-synchronized optimizer steps including forward, backward, gradient synchronization, gradient-norm validation and its all-reduce, and optimizer; excludes loading, tokenization, between-step logging and holdout evaluation.",
                   "weights_saved": False, "resumed_checkpoint": None}
        summary["unused_auxiliary_mtp_keys"] = key_audit["unused_auxiliary_mtp_keys"]
        write_json(args.out_dir / "summary.json", summary)
    accelerator.end_training()


def main() -> None:
    args = parser().parse_args()
    config = validate_paths_and_args(args)
    lower, upper = length_bounds(args)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.global_batch_size % world:
        raise ValueError("--global-batch-size must be divisible by WORLD_SIZE")
    if args.prepare_only and world != 1:
        raise ValueError("Run --prepare-only without torchrun")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    rows = load_rows(args.sft)
    train, holdout, audit = prepare_rows(rows, tokenizer, min_length=lower, max_length=upper,
                                       max_samples=args.max_samples, eval_samples=args.eval_samples, seed=args.seed)
    schedule = make_schedule(len(train), args.max_steps, args.global_batch_size)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "sft_sha256": file_digest(args.sft), "config_sha256": file_digest(args.model / "config.json"),
                "script_sha256": file_digest(Path(__file__)), "model_architectures": config["architectures"],
                "versions": versions(), "length_bounds_inclusive": [lower, upper],
                "objective": "Global-batch answer-token-mean causal CE, including EOS; all prompt tokens masked.",
                "prompt_format": "Tokenizer chat template, add_generation_prompt=True, enable_thinking=False; entire output plus tokenizer EOS.",
                "train_samples": len(train), "holdout_samples": len(holdout), "audit": audit,
                "schedule_sha256": digest([[train[index]["token_sha256"] for index in step] for step in schedule]),
                "schedule": [[train[index]["sample_sha256"] for index in step] for step in schedule],
                "repeated_training_examples": max(0, args.max_steps * args.global_batch_size - len(train))}
    if args.prepare_only:
        args.out_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.out_dir / "manifest.json", manifest)
        print(json.dumps({"status": "prepared", "train_samples": len(train), "holdout_samples": len(holdout),
                          "manifest": str(args.out_dir / "manifest.json")}), flush=True)
    else:
        run_training(args, train, holdout, schedule, manifest)


if __name__ == "__main__":
    main()
