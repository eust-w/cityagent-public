#!/usr/bin/env python3
"""Fresh Qwen3.5 LoRA training with the independently audited NVIDIA FSDP2 path.

Full sequences only; 2018 is held out; overlong rows are isolated in the manifest.
Every epoch visits each selected training row once. Tail padding participates in
collectives with zero loss weight and is excluded from sample/token accounting.
Only LoRA tensors are gathered for portable PEFT checkpoints, one tensor at a time.
This script never restores optimizer state or resumes an existing output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import partial
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import time


# Keep the benchmark file unchanged. Its pure preparation/audit functions remain
# shared, while the verified runtime setup below is copied into this trainer.
_HELPER_PATH = Path(__file__).with_name("benchmark_lora.py")
_SPEC = importlib.util.spec_from_file_location("_nvidia_benchmark_helpers", _HELPER_PATH)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)
digest, file_digest, write_json = bench.digest, bench.file_digest, bench.write_json
versions, audit_checkpoint_keys = bench.versions, bench.audit_checkpoint_keys
rank_indices, TEXT_TARGETS = bench.rank_indices, bench.TEXT_TARGETS


def epoch_schedule(sample_count: int, global_batch_size: int, seed: int, epoch: int) -> list[list[dict]]:
    if min(sample_count, global_batch_size, epoch) < 1:
        raise ValueError("sample count, batch size and epoch must be positive")
    indices = list(range(sample_count))
    random.Random(seed + epoch).shuffle(indices)
    batches = []
    for start in range(0, sample_count, global_batch_size):
        active = indices[start:start + global_batch_size]
        batch = [{"index": index, "weight": 1} for index in active]
        # Retain real labels for finite CE, then multiply the entire loss by zero.
        batch += [{"index": active[0], "weight": 0}
                  for _ in range(global_batch_size - len(batch))]
        batches.append(batch)
    return batches


def learning_rate_for_step(step: int, total_steps: int, base_lr: float, warmup_ratio: float) -> float:
    if not 1 <= step <= total_steps or not 0 <= warmup_ratio < 1:
        raise ValueError("Invalid scheduler step or warmup ratio")
    warmup = min(total_steps, math.ceil(total_steps * warmup_ratio))
    if warmup and step <= warmup:
        return base_lr * step / warmup
    # Apply a positive LR to every planned optimizer update. The conceptual
    # scheduler value after the final update is zero (as with HF scheduler.step).
    progress = (step - warmup - 1) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def canonical_adapter_name(name: str) -> str:
    parts = [part for part in name.split(".")
             if part not in {"_checkpoint_wrapped_module", "_offload_wrapped_module"}]
    if parts and parts[0] == "module":
        parts.pop(0)
    return ".".join(parts)


def adapter_parameter_map(model, expected: dict | None = None) -> dict:
    parameters = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        name = canonical_adapter_name(name)
        if not re.search(r"\.lora_[AB]\.default\.weight$", name):
            raise RuntimeError(f"Refusing non-LoRA trainable tensor: {name}")
        if name in parameters:
            raise RuntimeError(f"Adapter key collision: {name}")
        parameters[name] = parameter
    geometry = {name: list(parameter.shape) for name, parameter in parameters.items()}
    if not parameters or (expected is not None and geometry != expected):
        raise RuntimeError("Adapter keys/shapes differ from the pre-FSDP inventory")
    return dict(sorted(parameters.items()))


def gather_adapter_cpu(model, rank: int, expected: dict) -> dict:
    """All ranks call full_tensor in identical order; retain CPU state only on rank 0."""
    state = {}
    adapter_parameter_map(model, expected)
    # PEFT 0.21 filters against named_modules, so keep wrapper names until after
    # get_peft_model_state_dict has selected the current model's adapter tensors.
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        value = parameter.detach()
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        if rank == 0:
            state[name] = value.to(device="cpu").contiguous()
        del value
    return state


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--sft", type=Path, required=True)
    result.add_argument("--out-dir", type=Path, required=True)
    result.add_argument("--mode", choices=("bf16-sharded",), default="bf16-sharded")
    result.add_argument("--activation-offload", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--max-steps", type=int, help="Stop early and save a partial-epoch checkpoint (tiny preflight)")
    result.add_argument("--global-batch-size", type=int, default=8)
    result.add_argument("--max-length", type=int, default=32768)
    result.add_argument("--min-length", type=int, default=1)
    result.add_argument("--max-samples", type=int, help="Training limit for synthetic preflight; default: all eligible")
    result.add_argument("--eval-samples", type=int, help="Holdout limit for synthetic preflight; default: all eligible")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--warmup-ratio", type=float, default=0.03)
    result.add_argument("--max-grad-norm", type=float, default=1.0)
    result.add_argument("--lora-rank", type=int, default=16)
    result.add_argument("--lora-alpha", type=int, default=32)
    result.add_argument("--lora-dropout", type=float, default=0.05)
    result.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    return result


def validate_args(args) -> dict:
    if not args.model.is_dir() or not args.sft.is_file():
        raise ValueError("Explicit local model directory and SFT file are required")
    if args.out_dir.exists():
        raise FileExistsError("Choose a NEW output directory; automatic resume is forbidden")
    if min(args.epochs, args.global_batch_size, args.lora_rank, args.lora_alpha) < 1:
        raise ValueError("Epochs/batch/LoRA dimensions must be positive")
    for name in ("max_steps", "max_samples"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive when supplied")
    if args.eval_samples is not None and args.eval_samples < 0:
        raise ValueError("eval_samples cannot be negative")
    if not 1 <= args.min_length <= args.max_length:
        raise ValueError("Require 1 <= min-length <= max-length")
    if not (math.isfinite(args.learning_rate) and args.learning_rate > 0
            and math.isfinite(args.max_grad_norm) and args.max_grad_norm > 0
            and 0 <= args.warmup_ratio < 1 and 0 <= args.lora_dropout < 1):
        raise ValueError("Invalid learning rate, gradient norm, warmup or dropout")
    config = json.loads((args.model / "config.json").read_text())
    if config.get("model_type") != "qwen3_5" or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("Require the complete Qwen3_5ForConditionalGeneration architecture")
    return config


def prepare_data(args, tokenizer):
    rows = bench.load_rows(args.sft)
    return bench.prepare_rows(
        rows, tokenizer, min_length=args.min_length, max_length=args.max_length,
        max_samples=args.max_samples if args.max_samples is not None else len(rows),
        eval_samples=args.eval_samples if args.eval_samples is not None else len(rows), seed=args.seed)


def run_training(args: argparse.Namespace, train: list, holdout: list, schedule: list,
                 manifest: dict, tokenizer) -> None:
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
    rank_indices(schedule[0][0], rank, world)
    if sharded and accelerator.distributed_type.name != "FSDP":
        raise RuntimeError("FSDP was requested but not activated")
    if accelerator.gradient_accumulation_steps != 1:
        raise RuntimeError("Unexpected extra gradient-accumulation loss scaling")
    if world > 1:
        fingerprints = [None] * world
        fingerprint = {key: manifest[key] for key in ("sft_sha256", "config_sha256", "script_sha256", "helper_sha256", "schedule_sha256")}
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
    expected_adapter_geometry = {name: list(p.shape) for name, p in adapter_parameter_map(model).items()}
    model, optimizer = accelerator.prepare(model, optimizer)
    adapter_parameter_map(model, expected_adapter_geometry)
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

    def emit(event, **values):
        if rank == 0:
            record = {"event": event, "utc": datetime.now(timezone.utc).isoformat(), **values}
            text = json.dumps(record, allow_nan=False)
            with (args.out_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")
                handle.flush()
            print(text, flush=True)

    def batch(sample):
        return {"input_ids": torch.tensor([sample["input_ids"]], device=accelerator.device),
                "labels": torch.tensor([sample["labels"]], device=accelerator.device),
                "attention_mask": torch.ones((1, sample["tokens"]), dtype=torch.long,
                                             device=accelerator.device)}

    def scalar_loss(sample):
        if head is not None:
            head.unshard()
        with accelerator.autocast():
            output = model(**batch(sample), use_cache=False, skip_logits=True)
        if output.logits is not None:
            raise RuntimeError("Fused CE unexpectedly produced complete logits")
        return output.loss

    def evaluate(epoch, step):
        model.eval()
        records = []
        emit("evaluation_started", epoch=epoch, step=step, samples=len(holdout))
        with torch.no_grad():
            # Every rank uses the same row to preserve the FSDP collective order.
            for index, sample in enumerate(holdout, 1):
                loss = scalar_loss(sample).item()
                if head is not None:
                    head.reshard()
                if not math.isfinite(loss):
                    raise RuntimeError("Non-finite held-out teacher-forced loss")
                record = {"sample_sha256": sample["sample_sha256"],
                          "answer_tokens": sample["answer_tokens"], "loss": loss}
                records.append(record)
                emit("evaluation_sample", epoch=epoch, step=step, index=index, **record)
        result = {"year": "2018", "samples": records,
                  "token_mean_loss": (sum(row["loss"] * row["answer_tokens"] for row in records)
                                      / sum(row["answer_tokens"] for row in records)) if records else None,
                  "interpretation": "Teacher-forced loss; not generated-report quality or task score."}
        emit("evaluation_completed", epoch=epoch, step=step,
             samples=len(records), token_mean_loss=result["token_mean_loss"])
        return result

    def save_adapter(epoch, step, epoch_complete, evaluation):
        import copy
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        label = f"epoch-{epoch:03d}" if epoch_complete else f"epoch-{epoch:03d}-partial-step-{step:06d}"
        final_path = args.out_dir / label
        pending_path = args.out_dir / (label + ".incomplete")
        emit("save_started", epoch=epoch, step=step, checkpoint=label)
        accelerator.wait_for_everyone()
        # This is deliberately not accelerator.get_state_dict/full model state_dict.
        # Only a single adapter tensor is materialized on GPUs at any one time.
        cpu_state = gather_adapter_cpu(model, rank, expected_adapter_geometry)
        status = [None]
        checkpoint_record = None
        if rank == 0:
            try:
                if final_path.exists():
                    raise FileExistsError(final_path)
                pending_path.mkdir(exist_ok=False)
                peft_model = accelerator.unwrap_model(model)
                filtered = get_peft_model_state_dict(
                    peft_model, state_dict=cpu_state, adapter_name="default", save_embedding_layers=False)
                exported = {}
                for name, value in filtered.items():
                    key = canonical_adapter_name(name)
                    if key in exported:
                        raise RuntimeError(f"Serialized adapter key collision: {key}")
                    exported[key] = value
                expected = {name.replace(".default.", "."): shape
                            for name, shape in expected_adapter_geometry.items()}
                actual = {name: list(value.shape) for name, value in exported.items()}
                if actual != expected:
                    raise RuntimeError("Exported adapter keys/shapes differ from pre-FSDP inventory")
                tensor_records = {}
                for name, value in exported.items():
                    if value.device.type != "cpu" or not torch.isfinite(value).all().item():
                        raise RuntimeError("Adapter export must contain finite CPU tensors")
                    tensor_records[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                save_file(exported, str(pending_path / "adapter_model.safetensors"), metadata={"format": "pt"})
                peft_config = copy.deepcopy(peft_model.peft_config["default"])
                peft_config.inference_mode = True
                peft_config.base_model_name_or_path = str(args.model.resolve())
                peft_config.save_pretrained(pending_path)
                tokenizer.save_pretrained(pending_path)
                checkpoint_record = {
                    "epoch": epoch, "step": step, "epoch_complete": epoch_complete,
                    "checkpoint": label, "adapter_tensor_count": len(exported),
                    "adapter_tensors": tensor_records,
                    "adapter_sha256": file_digest(pending_path / "adapter_model.safetensors"),
                    "manifest_sha256": file_digest(args.out_dir / "manifest.json"),
                    "heldout_teacher_forced": evaluation,
                    "export": "Adapter-only DTensor full_tensor, one tensor per collective; rank 0 CPU safetensors.",
                }
                write_json(pending_path / "manifest.json", manifest)
                write_json(pending_path / "summary.json", checkpoint_record)
                pending_path.rename(final_path)
            except Exception as error:
                status[0] = f"{type(error).__name__}: {error}"
        del cpu_state
        if world > 1:
            torch.distributed.broadcast_object_list(status, src=0)
        if status[0]:
            raise RuntimeError(f"Adapter export failed: {status[0]}")
        accelerator.wait_for_everyone()
        emit("save_completed", epoch=epoch, step=step, checkpoint=label)
        return checkpoint_record

    total_steps = sum(len(epoch) for epoch in schedule)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    step_records, checkpoint_records = [], []
    global_step = 0
    emit("training_started", train_samples=len(train), holdout_samples=len(holdout),
         epochs=args.epochs, planned_optimizer_steps=total_steps, world_size=world,
         global_batch_size=args.global_batch_size, warmup_steps=math.ceil(total_steps * args.warmup_ratio))
    try:
        with (args.out_dir / f"steps-rank{rank}.jsonl").open("x", encoding="utf-8", buffering=1) as log:
            for epoch, batches in enumerate(schedule, 1):
                model.train()
                emit("epoch_started", epoch=epoch, batches=len(batches))
                completed_batches = 0
                for batch_index, slots in enumerate(batches, 1):
                    if global_step >= total_steps:
                        break
                    global_step += 1
                    local = rank_indices(slots, rank, world)
                    global_answers = sum(train[slot["index"]]["answer_tokens"] * slot["weight"] for slot in slots)
                    lr = learning_rate_for_step(global_step, total_steps, args.learning_rate, args.warmup_ratio)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    accelerator.wait_for_everyone()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    loss_sum = torch.zeros((), dtype=torch.float64, device=accelerator.device)
                    local_tokens = local_answers = local_samples = 0
                    active_hashes, padding_hashes = [], []
                    for slot in local:
                        sample, weight = train[slot["index"]], slot["weight"]
                        loss = scalar_loss(sample)
                        if not torch.isfinite(loss).item():
                            raise RuntimeError("Non-finite training loss")
                        # All ranks backpropagate, including dummy slots. DDP/FSDP
                        # averages gradients: multiply by world to obtain the true
                        # global answer-token mean over ONLY the active samples.
                        accelerator.backward(loss * (sample["answer_tokens"] * weight * world / global_answers))
                        if head is not None:
                            head.reshard()
                        loss_sum += loss.detach().double() * sample["answer_tokens"] * weight
                        local_tokens += sample["tokens"] * weight
                        local_answers += sample["answer_tokens"] * weight
                        local_samples += weight
                        (active_hashes if weight else padding_hashes).append(sample["sample_sha256"])
                        del loss
                    grad_square = torch.zeros((), dtype=torch.float64, device=accelerator.device)
                    for parameter in model.parameters():
                        if parameter.requires_grad and parameter.grad is not None:
                            gradient = parameter.grad.detach()
                            if hasattr(gradient, "to_local"):
                                gradient = gradient.to_local()
                            grad_square += gradient.float().square().sum().double()
                    if world > 1:
                        torch.distributed.all_reduce(grad_square)
                    gradient_l2 = math.sqrt(grad_square.item())
                    if not math.isfinite(gradient_l2) or gradient_l2 <= 0:
                        raise RuntimeError("Adapter gradients are non-finite or all zero")
                    clip_scale = min(1.0, args.max_grad_norm / (gradient_l2 + 1e-6))
                    with torch.no_grad():
                        for parameter in model.parameters():
                            if parameter.requires_grad and parameter.grad is not None:
                                parameter.grad.mul_(clip_scale)
                    optimizer.step()
                    torch.cuda.synchronize()
                    local_record = {
                        "step": global_step, "epoch": epoch, "batch_in_epoch": batch_index,
                        "seconds": time.perf_counter() - start, "learning_rate": lr,
                        "nonpad_tokens": local_tokens, "answer_tokens": local_answers,
                        "training_samples": local_samples, "padding_slots": len(local) - local_samples,
                        "sample_sha256": active_hashes, "zero_weight_padding_sha256": padding_hashes,
                        "loss_sum": loss_sum.item(), "global_gradient_l2_before_clip": gradient_l2,
                        "gradient_clip_scale": clip_scale,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    }
                    log.write(json.dumps(local_record, allow_nan=False) + "\n")
                    log.flush()
                    gathered = [None] * world
                    torch.distributed.all_gather_object(gathered, local_record)
                    answers = sum(value["answer_tokens"] for value in gathered)
                    record = {
                        "step": global_step, "epoch": epoch, "batch_in_epoch": batch_index,
                        "seconds": max(value["seconds"] for value in gathered), "learning_rate": lr,
                        "training_samples": sum(value["training_samples"] for value in gathered),
                        "padding_slots": sum(value["padding_slots"] for value in gathered),
                        "nonpad_tokens": sum(value["nonpad_tokens"] for value in gathered),
                        "answer_tokens": answers,
                        "token_mean_loss": sum(value["loss_sum"] for value in gathered) / answers,
                        "global_gradient_l2_before_clip": gradient_l2, "gradient_clip_scale": clip_scale,
                        "peak_allocated_bytes_per_rank": [value["peak_allocated_bytes"] for value in gathered],
                    }
                    step_records.append(record)
                    completed_batches += 1
                    emit("optimizer_step", **record)
                epoch_complete = completed_batches == len(batches)
                evaluation = evaluate(epoch, global_step)
                checkpoint = save_adapter(epoch, global_step, epoch_complete, evaluation)
                if rank == 0:
                    checkpoint_records.append(checkpoint)
                emit("epoch_completed" if epoch_complete else "epoch_stopped_early",
                     epoch=epoch, step=global_step, completed_batches=completed_batches)
                if global_step >= total_steps:
                    break
        if rank == 0:
            write_json(args.out_dir / "summary.json", {
                "status": "completed" if args.max_steps is None or global_step == sum(map(len, schedule)) else "max_steps_reached",
                "optimizer_steps": global_step, "epochs_requested": args.epochs,
                "training_sample_visits": sum(record["training_samples"] for record in step_records),
                "zero_weight_padding_slots": sum(record["padding_slots"] for record in step_records),
                "steps": step_records, "checkpoints": checkpoint_records,
                "manifest_sha256": file_digest(args.out_dir / "manifest.json"),
                "weights_saved": "LoRA adapters only", "resumed_checkpoint": None,
            })
        emit("training_completed", optimizer_steps=global_step)
        accelerator.wait_for_everyone()
    finally:
        accelerator.end_training()


def main():
    args = parser().parse_args()
    config = validate_args(args)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.global_batch_size % world:
        raise ValueError("Global batch size must be divisible by WORLD_SIZE")
    if args.prepare_only and world != 1:
        raise ValueError("Run prepare-only without torchrun")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    train, holdout, audit = prepare_data(args, tokenizer)
    schedule = [epoch_schedule(len(train), args.global_batch_size, args.seed, epoch)
                for epoch in range(1, args.epochs + 1)]
    hashed_schedule = [[[{"sample_sha256": train[slot["index"]]["sample_sha256"],
                          "token_sha256": train[slot["index"]]["token_sha256"], "weight": slot["weight"]}
                         for slot in batch] for batch in epoch] for epoch in schedule]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sft_sha256": file_digest(args.sft), "config_sha256": file_digest(args.model / "config.json"),
        "script_sha256": file_digest(Path(__file__)), "helper_sha256": file_digest(_HELPER_PATH),
        "model_architectures": config["architectures"], "versions": versions(),
        "objective": "Global-batch answer-token-mean causal CE including EOS; prompts masked; zero-weight padding excluded.",
        "prompt_format": "Unchanged benchmark tokenizer chat template; full input and output; enable_thinking=False.",
        "length_bounds_inclusive": [args.min_length, args.max_length],
        "train_samples": len(train), "holdout_samples": len(holdout),
        "audit_counts": dict(Counter(f"{row['split']}:{row['status']}" for row in audit)),
        "audit": audit, "schedule": hashed_schedule, "schedule_sha256": digest(hashed_schedule),
        "scheduler": "Per-step LR, ceil(steps*warmup_ratio) linear warmup then cosine to zero; no hidden scheduler stepping.",
        "resumed_checkpoint": None,
    }
    if args.prepare_only:
        args.out_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.out_dir / "manifest.json", manifest)
        print(json.dumps({"status": "prepared", "train_samples": len(train), "holdout_samples": len(holdout),
                          "audit_counts": manifest["audit_counts"]}), flush=True)
    else:
        run_training(args, train, holdout, schedule, manifest, tokenizer)


if __name__ == "__main__":
    main()
