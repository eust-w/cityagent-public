#!/usr/bin/env python3
"""Explicit single-device LoRA experiments using the historical HF Trainer path.

By default this only prepares a manifest using a local tokenizer. Training needs
--execute. Full sequences are retained; overlong rows are recorded and excluded.
Existing outputs and automatic resume are forbidden. This public implementation
has not been revalidated on PPU hardware; see docs/training.md.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import time
from datetime import UTC, datetime
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "_legacy_training_helpers", Path(__file__).with_name("benchmark_lora.py"))
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

TARGET_PATTERN = (
    r"(?:model\.language_model|model)\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
)
ARCHITECTURES = {
    "image-text-to-text": {"Qwen3_5ForConditionalGeneration"},
    "causal-lm": {"Qwen3_5ForCausalLM", "Qwen3ForCausalLM"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "smoke", "benchmark"), default="train")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute", action="store_true", help="Explicitly load weights and train")
    execution.add_argument("--prepare-only", action="store_true", help="Prepare only (the default)")
    parser.add_argument("--model-class", choices=tuple(ARCHITECTURES), default="image-text-to-text")
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0",
                        help="Expose one device with CUDA_VISIBLE_DEVICES before --execute")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-samples", type=int, help="Optional deterministic training subset")
    parser.add_argument("--max-steps", type=int, help="Required for smoke/benchmark, between 1 and 100")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--holdout-year", default="2018")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-liger-kernel", action="store_true", help="Opt-in; requires a compatible installed kernel")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args) -> dict:
    if not args.model.is_dir() or not args.sft.is_file():
        raise ValueError("Explicit local model directory and SFT file are required")
    output, model, sft = (path.resolve() for path in (args.out_dir, args.model, args.sft))
    if output.exists():
        raise FileExistsError("Choose a NEW output directory; automatic resume is forbidden")
    if output == model or model in output.parents or output in model.parents or output in sft.parents:
        raise ValueError("Output must not overlap model inputs or contain the SFT input")
    if min(args.max_length, args.epochs, args.gradient_accumulation_steps, args.lora_rank, args.lora_alpha) < 1:
        raise ValueError("Length, epochs, accumulation and LoRA dimensions must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.mode in {"smoke", "benchmark"} and (args.max_steps is None or args.max_steps > 100):
        raise ValueError("smoke/benchmark require explicit --max-steps between 1 and 100")
    if not re.fullmatch(r"(?:19|20)\d{2}", args.holdout_year):
        raise ValueError("holdout-year must be an explicit four-digit year")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 <= args.lora_dropout < 1:
        raise ValueError("Invalid learning rate or LoRA dropout")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This entry point is single-device; use train_lora_nvidia.py for FSDP2")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures") if isinstance(config, dict) else None
    if (not isinstance(architectures, list) or len(architectures) != 1
            or architectures[0] not in ARCHITECTURES[args.model_class]):
        raise ValueError("Unsupported architecture or model-class mismatch; automatic fallback is forbidden")
    if config.get("tie_word_embeddings") or config.get("text_config", {}).get("tie_word_embeddings"):
        raise ValueError("Tied embeddings are not supported by this entry point's exact checkpoint-key audit")
    return config


def prepare_data(args, tokenizer):
    """Keep every selected prompt/answer intact, with explicit exclusion receipts."""
    rows = bench.load_rows(args.sft)
    training, audit = [], []
    for row in rows:
        sample = bench.tokenize_row(row, tokenizer)
        record = {key: value for key, value in sample.items() if key not in {"input_ids", "labels"}}
        record["split"] = "holdout" if row["year"] == args.holdout_year else "train"
        if record["split"] == "holdout":
            record["status"] = "held_out"
        elif sample["tokens"] > args.max_length:
            record["status"] = "overlong_full_sequence"
        else:
            record["status"] = "eligible"
            training.append(sample)
        audit.append(record)
    training.sort(key=lambda row: row["sample_sha256"])
    random.Random(args.seed).shuffle(training)
    limit = args.max_samples
    if limit is None and args.mode in {"smoke", "benchmark"}:
        limit = 8
    if limit is not None:
        training = training[:limit]
    selected = {row["sample_sha256"] for row in training}
    for row in audit:
        if row["status"] == "eligible":
            row["status"] = "selected" if row["sample_sha256"] in selected else "sample_limit"
    if not training:
        raise ValueError("No complete non-holdout training samples fit the configured limits")
    return training, audit


def pad_batch(batch, pad_token_id):
    """Pure collator core, including correct handling of a valid pad token ID 0."""
    if not batch or pad_token_id is None:
        raise ValueError("Nonempty batch and an explicit pad token ID are required")
    if any(not row["input_ids"] or len(row["input_ids"]) != len(row["labels"]) for row in batch):
        raise ValueError("Input IDs and labels must have matching nonzero lengths")
    width = max(len(row["input_ids"]) for row in batch)
    return {
        "input_ids": [row["input_ids"] + [pad_token_id] * (width - len(row["input_ids"])) for row in batch],
        "labels": [row["labels"] + [-100] * (width - len(row["labels"])) for row in batch],
        "attention_mask": [[1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"])) for row in batch],
    }


def checkpoint_keys(model_dir):
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        return set(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    from safetensors import safe_open
    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as saved:
        return set(saved.keys())


def validate_training_result(loss, named_parameters, is_finite):
    """Never publish a numerically invalid adapter as a completed experiment."""
    if type(loss) not in (int, float) or not math.isfinite(loss):
        raise ValueError("Training loss must be finite")
    tensors = 0
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if not re.search(r"\.lora_[AB]\.default\.weight$", name) or not is_finite(parameter):
            raise ValueError("Trainable tensors must contain only finite LoRA A/B weights")
        tensors += 1
    if tensors == 0:
        raise ValueError("No trainable LoRA tensors to verify")
    return tensors


def run_training(args, training, tokenizer):
    """Only main's explicit --execute branch may call this heavy runtime path."""
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Expose exactly one BF16-capable CUDA-compatible device before --execute")
    transformers.set_seed(args.seed)
    runtime_start = time.monotonic()
    model_class = (transformers.AutoModelForImageTextToText if args.model_class == "image-text-to-text"
                   else transformers.AutoModelForCausalLM)
    model, loading = model_class.from_pretrained(
        str(args.model), dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=False, local_files_only=True, output_loading_info=True,
    )
    problems = {key: loading[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading.get(key)}
    audit = bench.audit_checkpoint_keys(
        checkpoint_keys(args.model), {name for name, _ in model.named_parameters(remove_duplicate=False)})
    if problems or audit["missing_keys"] or audit["unexpected_keys"]:
        raise RuntimeError("Base loading is incomplete; refusing training")
    targets = [name for name, _ in model.named_modules() if re.fullmatch(TARGET_PATTERN, name)]
    if not targets:
        raise RuntimeError("No supported text-layer LoRA targets found")
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=TARGET_PATTERN, bias="none", task_type="CAUSAL_LM"))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not re.search(r"\.lora_[AB]\.default\.weight$", name) for name in trainable):
        raise RuntimeError("Only LoRA A/B tensors may be trainable")
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate(batch):
        return {key: torch.tensor(value, dtype=torch.long) for key, value in pad_batch(batch, pad_id).items()}

    arguments = transformers.TrainingArguments(
        output_dir=str(args.out_dir / "trainer"), per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs, max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=1, save_strategy="no", bf16=True, report_to=[],
        dataloader_num_workers=0, use_liger_kernel=args.use_liger_kernel,
        seed=args.seed, remove_unused_columns=False,
    )
    trainer = transformers.Trainer(
        model=model, args=arguments,
        train_dataset=[{key: row[key] for key in ("input_ids", "labels")} for row in training],
        data_collator=collate)
    torch.cuda.synchronize()
    preparation_seconds = time.monotonic() - runtime_start
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=None)
    torch.cuda.synchronize()
    training_seconds = time.monotonic() - started
    verified_tensors = validate_training_result(
        result.training_loss, model.named_parameters(),
        lambda parameter: parameter.is_floating_point() and torch.isfinite(parameter).all().item())
    saved = args.mode != "benchmark"
    adapter_hash = None
    if saved:
        pending = args.out_dir / "adapter.incomplete"
        pending.mkdir(exist_ok=False)
        model.save_pretrained(str(pending), safe_serialization=True)
        tokenizer.save_pretrained(str(pending))
        adapter_hash = bench.file_digest(pending / "adapter_model.safetensors")
        pending.rename(args.out_dir / "adapter")
    summary = {
        "status": "completed", "mode": args.mode, "global_step": trainer.state.global_step,
        "selected_training_samples": len(training), "max_steps": args.max_steps,
        "model_preparation_seconds": preparation_seconds, "training_loop_seconds": training_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "trainer_metrics": result.metrics,
        "weights_saved": "LoRA adapter only" if saved else False, "resumed_checkpoint": None,
        "versions": bench.versions(), "base_key_audit": audit,
        "finite_adapter_tensors": verified_tensors, "adapter_sha256": adapter_hash,
        "quality_validated": False, "ppu_runtime_validated": False,
        "interpretation": "Observed local training loop; no corpus-speed extrapolation or generation-quality claim.",
    }
    bench.write_json(args.out_dir / "summary.json", summary)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = validate_args(args)
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=False, local_files_only=True)
    training, audit = prepare_data(args, tokenizer)
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(UTC).isoformat(),
        "mode": args.mode, "execute_requested": args.execute,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "script_sha256": bench.file_digest(Path(__file__)), "helper_sha256": bench.file_digest(Path(bench.__file__)),
        "sft_sha256": bench.file_digest(args.sft), "config_sha256": bench.file_digest(args.model / "config.json"),
        "architectures": config["architectures"], "selected_training_samples": len(training), "audit": audit,
        "split_protocol": "year-holdout", "holdout_year": args.holdout_year,
        "truncation": False, "resumed_checkpoint": None,
        "quality_validated": False, "ppu_runtime_validated": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    bench.write_json(args.out_dir / "manifest.json", manifest)
    if not args.execute:
        print(json.dumps({"status": "prepared", "train_samples": len(training), "weights_loaded": False}))
        return
    incomplete = args.out_dir / ".training_incomplete"
    incomplete.write_text("Explicit training started; success requires summary.json and removal of this marker.\n")
    run_training(args, training, tokenizer)
    incomplete.unlink()


if __name__ == "__main__":
    main()
