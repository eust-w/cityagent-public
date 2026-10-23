#!/usr/bin/env python3
"""Frozen-SFT, model-only generation evaluation; never silently drop failed reports.

Run in an isolated NVIDIA inference process. The training environment and snapshot
are read-only. Labels are used only for scoring after generation and for a CPU
prompt-prefix assertion; the backend receives prompt IDs, never answer IDs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest.inference_core import (
    ContextBudgetError, InferenceFailure, build_prompt, infer_document, prompt_ids_sha256,
)
from bridge_evidence.evaluation.generation import aggregate_reports, score_report

_SPEC = importlib.util.spec_from_file_location("generation_benchmark", Path(__file__).with_name("benchmark_lora.py"))
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def select_samples(manifest, rows, *, partition, shard_index=0, shards=1, limit=None,
                   sample_sha256=None):
    if not 0 <= shard_index < shards or (limit is not None and limit < 1):
        raise ValueError("Invalid shard or sample limit")
    by_hash = {row["sample_sha256"]: row for row in rows}
    if len(by_hash) != len(rows):
        raise ValueError("Duplicate source samples")
    samples = manifest["samples"]
    expected = {row["sample_sha256"] for row in rows if row["year"] == "2018"}
    if len(samples) != len(expected) or {s["sample_sha256"] for s in samples} != expected:
        raise ValueError("Manifest must contain the entire internal 2018 holdout exactly once")
    selected = []
    for sample in samples:
        if sample.get("partition") not in {"diagnostic", "final_acceptance"}:
            raise ValueError("Every held-out sample must belong to a known partition")
        row = by_hash[sample["sample_sha256"]]
        if sample["prompt_content_sha256"] != row["prompt_sha256"] or sample["line"] != row["line"]:
            raise ValueError("Manifest/source binding mismatch")
        if sample["partition"] == partition:
            selected.append((sample, row))
    # Prompt-only ordering cannot select samples based on their expected answers.
    selected.sort(key=lambda pair: pair[0]["prompt_content_sha256"])
    selected = selected[shard_index::shards]
    if sample_sha256 is not None:
        requested = set(sample_sha256)
        if not requested or len(requested) != len(sample_sha256):
            raise ValueError("Explicit sample IDs must be distinct and nonempty")
        if not requested.issubset({sample["sample_sha256"] for sample, _ in selected}):
            raise ValueError("Explicit samples must belong to the selected partition and shard")
        selected = [(sample, row) for sample, row in selected if sample["sample_sha256"] in requested]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("Empty evaluation selection")
    return selected


def output_response(token_ids, tokenizer, *, eos_token_ids, max_new_tokens):
    ids = list(token_ids)
    stopped = bool(ids) and ids[-1] in set(eos_token_ids)
    return {"text": tokenizer.decode(ids, skip_special_tokens=True),
            "output_tokens": len(ids),
            "finish_reason": "stop" if stopped else (
                "length" if len(ids) >= max_new_tokens else "stopped_without_eos")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("diagnostic", "final_acceptance"), required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-sha256", action="append",
                        help="Explicit subset of the selected frozen partition/shard; repeat for multiple cases")
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-gib", type=int, default=24)
    parser.add_argument("--sample-timeout-seconds", type=int, default=1200)
    parser.add_argument("--prefill-chunk-size", type=int, default=0,
                        help="Optional full-context prefill blocks; 0 keeps the original full prefill")
    parser.add_argument("--prefill-min-tokens", type=int, default=32768,
                        help="Apply chunking only at/above this prompt length; shorter calls remain unchanged")
    args = parser.parse_args()
    if min(args.max_model_len, args.max_new_tokens, args.gpu_memory_gib, args.sample_timeout_seconds) < 1:
        parser.error("Budgets must be positive")
    if args.prefill_chunk_size < 0 or args.prefill_min_tokens < 1:
        parser.error("Prefill chunk size must be nonnegative and threshold positive")
    config = json.loads((args.model / "config.json").read_text())
    native_context = config.get("text_config", {}).get("max_position_embeddings")
    if type(native_context) is not int or args.max_model_len > native_context:
        parser.error("Requested context exceeds the base model's declared native context")
    manifest = json.loads(args.selection_manifest.read_text())
    if bench.file_digest(args.sft) != manifest["sft_sha256"]:
        raise ValueError("Frozen SFT hash mismatch")
    rows = bench.load_rows(args.sft)
    selected = select_samples(manifest, rows, partition=args.partition,
                              shards=args.shards, shard_index=args.shard_index, limit=args.limit,
                              sample_sha256=args.sample_sha256)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def emit(event, **data):
        record = {"event": event, "utc": datetime.now(timezone.utc).isoformat(), **data}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with (args.out_dir / "events.jsonl").open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    import torch
    import transformers
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors import safe_open
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA BF16 hardware required")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.model), local_files_only=True,
                                                           trust_remote_code=False)
    frozen = {"sft_sha256": manifest["sft_sha256"],
              "selection_manifest_sha256": bench.file_digest(args.selection_manifest),
              "runner_sha256": bench.file_digest(Path(__file__)),
              "helper_sha256": bench.file_digest(Path(bench.__file__)),
              "inference_core_sha256": bench.file_digest(Path(sys.modules[infer_document.__module__].__file__)),
              "metrics_sha256": bench.file_digest(Path(sys.modules[score_report.__module__].__file__)),
              "base_config_sha256": bench.file_digest(args.model / "config.json"),
              "base_index_sha256": bench.file_digest(args.model / "model.safetensors.index.json"),
              "adapter_sha256": bench.file_digest(args.adapter / "adapter_model.safetensors"),
              "adapter_config_sha256": bench.file_digest(args.adapter / "adapter_config.json"),
              "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "selected_sample_sha256": [sample["sample_sha256"] for sample, _ in selected],
              "versions": bench.versions(), "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "interpretation": "Internal training-data holdout; not official competition test or score.",
              "generation": "greedy, no thinking, full prompt, one attempt, no rule fusion"}
    bench.write_json(args.out_dir / "manifest.json", frozen)
    # Validate all prompt prefixes on CPU before allocating weights.
    prepared = []
    for sample, row in selected:
        tokenized = bench.tokenize_row(row, tokenizer)
        if tokenized["prompt_tokens"] != sample["prompt_tokens"]:
            raise ValueError("Tokenizer differs from frozen sample metadata")
        if tokenized["token_sha256"] != sample["full_sequence_token_sha256"]:
            raise ValueError("Token IDs differ from the frozen training tokenization")
        expected_ids = tokenized["input_ids"][:tokenized["prompt_tokens"]]
        try:
            ids = build_prompt(tokenizer, row["input"], instruction=row["instruction"],
                               max_model_len=args.max_model_len, max_new_tokens=args.max_new_tokens)
            if ids != expected_ids:
                raise ValueError("Inference prompt differs from the training prefix")
            budget_error = None
        except ContextBudgetError as error:
            budget_error = str(error)
        prepared.append((sample, row, prompt_ids_sha256(expected_ids), budget_error))
        del tokenized, expected_ids
    emit("loading_started", samples=len(selected), budget_failures=sum(x[3] is not None for x in prepared))
    model, loading = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map="balanced",
        max_memory={i: f"{args.gpu_memory_gib}GiB" for i in range(torch.cuda.device_count())},
        output_loading_info=True,
    )
    keys = set(json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"])
    key_audit = bench.audit_checkpoint_keys(keys, {name for name, _ in model.named_parameters(remove_duplicate=False)})
    problems = {k: loading[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(k)}
    device_map = {k: str(v) for k, v in model.hf_device_map.items()}
    if problems or key_audit["missing_keys"] or key_audit["unexpected_keys"]:
        raise RuntimeError(f"Base model loading failure: {problems}; {key_audit}")
    if set(device_map.values()) & {"cpu", "disk"}:
        raise RuntimeError("Evaluation requires all base parameters on GPUs")
    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False,
                                     local_files_only=True, autocast_adapter_dtype=True,
                                     low_cpu_mem_usage=False, torch_device="cpu")
    actual = get_peft_model_state_dict(model, save_embedding_layers=False)
    adapter_audit = {"tensors": 0, "numel": 0, "exact": True}
    with safe_open(str(args.adapter / "adapter_model.safetensors"), framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(actual):
            raise RuntimeError("Reloaded adapter tensor names differ")
        for name in saved.keys():
            expected = saved.get_tensor(name)
            value = actual[name].detach().cpu()
            if value.dtype != expected.dtype or not torch.equal(value, expected):
                raise RuntimeError("Reloaded adapter differs: " + name)
            adapter_audit["tensors"] += 1
            adapter_audit["numel"] += value.numel()
    del actual
    model.eval()
    model.config.use_cache = True
    model.config.text_config.use_cache = True
    bench.write_json(args.out_dir / "reload-verification.json", {
        "base_key_audit": key_audit, "loading_problems": problems,
        "device_map": device_map, "adapter": adapter_audit,
        "adapter_sha256": frozen["adapter_sha256"],
    })
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if not eos:
        raise ValueError("Generation EOS is missing")
    input_device = model.get_input_embeddings().weight.device
    emit("loading_completed", adapter=adapter_audit, gpu_memory_bytes=[torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())])
    active_sample = None

    def generate(ids, max_new_tokens):
        started_generate = time.monotonic()
        last_progress = [started_generate]

        class Progress(transformers.StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                now = time.monotonic()
                if now - last_progress[0] >= 30:
                    emit("generation_progress", sample_sha256=active_sample,
                         output_tokens=input_ids.shape[-1] - len(ids), seconds=now - started_generate)
                    last_progress[0] = now
                return False

        inputs = torch.tensor([ids], dtype=torch.long, device=input_device)
        prefill = {}
        if args.prefill_chunk_size and len(ids) >= args.prefill_min_tokens:
            # Transformers 5.18 chunking slices positions as [batch, length].
            # Qwen's automatic multimodal positions are 3-D, so explicitly use
            # the equivalent 2-D positions for these text-only report inputs.
            # Keep the full input_ids and attention_mask: KV and recurrent state
            # carry every earlier token through successive prefill blocks.
            prefill = {"prefill_chunk_size": args.prefill_chunk_size,
                       "position_ids": torch.arange(len(ids), device=input_device)[None, :]}
            emit("chunked_prefill_started", sample_sha256=active_sample,
                 prompt_tokens=len(ids), chunk_size=args.prefill_chunk_size)
        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                logits_to_keep=1,
                eos_token_id=eos, pad_token_id=tokenizer.pad_token_id or eos[0],
                max_time=args.sample_timeout_seconds,
                stopping_criteria=transformers.StoppingCriteriaList([Progress()]),
                **prefill,
            )
        generated = output[0, len(ids):].tolist()
        return output_response(generated, tokenizer, eos_token_ids=eos, max_new_tokens=max_new_tokens)

    scores = []
    for index, (sample, row, prompt_hash, budget_error) in enumerate(prepared, 1):
        active_sample = sample["sample_sha256"]
        emit("sample_started", index=index, sample_sha256=active_sample, prompt_tokens=sample["prompt_tokens"])
        record = {**sample, "prompt_ids_sha256": prompt_hash, "prediction": None,
                  "raw_output": "", "model_calls": [], "status": "failed",
                  "prefill_chunk_size": args.prefill_chunk_size if (
                      args.prefill_chunk_size and sample["prompt_tokens"] >= args.prefill_min_tokens) else 0}
        if budget_error:
            record["error"] = "context_budget: " + budget_error
        else:
            try:
                result = infer_document(row["input"], tokenizer, generate, instruction=row["instruction"],
                                        max_model_len=args.max_model_len, max_new_tokens=args.max_new_tokens,
                                        max_attempts=1)
                if any(call["prompt_sha256"] != prompt_hash for call in result["model_calls"]):
                    raise RuntimeError("Backend prompt fingerprint mismatch")
                record.update(result)
                record["prediction"] = record.pop("data")
                record["status"] = "success"
            except InferenceFailure as error:
                record.update(error=type(error).__name__, model_calls=error.model_calls,
                              raw_output=error.raw_output, seconds=error.seconds,
                              wall_seconds=error.wall_seconds, finish_reason=error.finish_reason,
                              output_tokens=error.output_tokens)
        gold = json.loads(row["output"])
        score_report(gold, None)  # Invalid labels are fatal, never hidden as model failures.
        try:
            score = score_report(gold, record["prediction"])
        except ValueError as error:
            record.update(error="metric_schema: " + str(error), prediction=None, status="failed")
            score = score_report(gold, None)
        record["unknown_top_level_sections"] = sorted(
            set(record["prediction"] or {}) - {"基本信息", "维修建议", "病害明细"})
        scores.append(score)
        record["score"] = score
        bench.write_json(args.out_dir / (active_sample + ".json"), record)
        emit("sample_completed", index=index, sample_sha256=active_sample, status=record["status"],
             finish_reason=record.get("finish_reason"), output_tokens=record.get("output_tokens"),
             seconds=record.get("seconds"), overall=score["overall"])
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    summary = {"status": "completed", "samples": len(scores), "partition": args.partition,
               "wall_seconds": time.monotonic() - started, "metrics": aggregate_reports(scores),
               "manifest_sha256": bench.file_digest(args.out_dir / "manifest.json")}
    bench.write_json(args.out_dir / "summary.json", summary)
    emit("evaluation_completed", **summary)


if __name__ == "__main__":
    main()
