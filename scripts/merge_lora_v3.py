#!/usr/bin/env python3
"""将显式指定的 LoRA 合并到完整多模态底座：W' = W + (alpha/r) * (B @ A)。

底座、adapter 与全新输出目录均须由用户显式提供，只读取本地模型代码。
不训练，不覆盖已有模型，未完成的输出不能用于推理。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path

ADAPTER_KEY = re.compile(
    r"^base_model\.model\.(?P<layout>model(?:\.language_model)?)\."
    r"(?P<module>layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj)))\.lora_(?P<side>[AB])\.weight$"
)

def log(*args):
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check", action="store_true", help="仅检查路径及 LoRA 配置，不加载或写入模型")
    return parser


def validate_paths(args) -> dict:
    base, adapter, output = (p.resolve() for p in (args.base_model, args.adapter, args.output))
    for source in (base, adapter):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"输出目录不能与模型输入目录重叠：{output} / {source}")
    if output == Path("/model") or Path("/model") in output.parents:
        raise ValueError("赛事 /model 是只读挂载；请将合并模型输出到 /workspace 下")
    if output.exists():
        raise ValueError(f"输出目录已存在，不覆盖：{output}。已有完整合并模型可直接交给 run.py；重建请另选空路径")
    if not (base / "config.json").is_file():
        raise ValueError(f"底座 config.json 不存在：{base}")
    index = base / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards or any(not (base / shard).is_file() for shard in shards):
            raise ValueError(f"底座权重分片缺失：{base}")
    elif not (base / "model.safetensors").is_file():
        raise ValueError(f"底座权重缺失：{base}")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file():
            raise ValueError(f"外置 LoRA 文件缺失：{adapter / name}；请将微调权重放入 work 目录，或指定 --adapter")
    config = json.loads((adapter / "adapter_config.json").read_text())
    unsupported = ("use_dora", "use_rslora", "fan_in_fan_out", "rank_pattern", "alpha_pattern",
                   "modules_to_save", "lora_bias", "layer_replication", "target_parameters",
                   "alora_invocation_tokens", "use_bdlora")
    if config.get("peft_type") != "LORA" or any(config.get(key) for key in unsupported):
        raise ValueError("此合并工具只支持普通 LoRA、统一 rank/alpha")
    rank, alpha = config.get("r"), config.get("lora_alpha")
    if (config.get("bias", "none") != "none" or type(rank) is not int or rank <= 0
            or type(alpha) not in (float, int) or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("LoRA 配置不支持：要求 bias=none、r > 0 且包含 lora_alpha")
    return config


def plan_adapter_merge(state, parameters, mount, rank):
    """Map either documented PEFT layout once; reject ambiguous/unknown tensors.

    This pure shape/name check validates every pair before any base weight is
    changed. It accepts old text adapters and complete-multimodal trainer output,
    without prefix guessing or silently ignoring unsupported keys.
    """
    if not state:
        raise ValueError("LoRA 权重为空")
    pairs, layouts = {}, set()
    for key, tensor in state.items():
        match = ADAPTER_KEY.fullmatch(key)
        if not match:
            raise ValueError(f"不支持的 LoRA 键：{key}")
        layouts.add(match["layout"])
        target, side = mount + match["module"] + ".weight", match["side"]
        pair = pairs.setdefault(target, {})
        if side in pair:
            raise ValueError(f"重复 LoRA 映射：{target} / {side}")
        pair[side] = tensor
    if len(layouts) != 1:
        raise ValueError("混合 LoRA 命名前缀不明确；每个 adapter 必须使用单一布局")
    for target, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"LoRA A/B 配对不完整：{target}")
        parameter = parameters.get(target)
        if parameter is None:
            raise ValueError(f"底座中找不到 LoRA 对应参数：{target}")
        shape = tuple(parameter.shape)
        if (len(shape) != 2 or tuple(pair["A"].shape) != (rank, shape[1])
                or tuple(pair["B"].shape) != (shape[0], rank)):
            raise ValueError(f"LoRA rank 或矩阵形状不匹配：{target}")
    return pairs


def apply_adapter_merge(pairs, parameters, scaling, torch):
    """Apply checked pairs; a numeric failure occurs before any output is saved."""
    for pair in pairs.values():
        for value in pair.values():
            if not value.is_floating_point() or not torch.isfinite(value).all().item():
                raise ValueError("LoRA 权重必须是有限浮点数")
    with torch.no_grad():
        for name, pair in pairs.items():
            parameter = parameters[name]
            a, b = pair["A"].to(parameter.device), pair["B"].to(parameter.device)
            delta = (b.float() @ a.float()) * scaling
            if not torch.isfinite(delta).all().item():
                raise ValueError(f"LoRA 合并结果不是有限值：{name}")
            parameter.add_(delta.to(parameter.dtype))
            if not torch.isfinite(parameter).all().item():
                raise ValueError(f"LoRA 合并后的底座参数不是有限值：{name}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = validate_paths(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.check:
        log("路径与 LoRA 配置检查通过；尚未加载模型或验证输出空间")
        return

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    scaling = config["lora_alpha"] / config["r"]
    log("loading FULL multimodal base (bf16) ...", args.base_model)
    t0 = time.time()
    full = AutoModelForImageTextToText.from_pretrained(
        str(args.base_model), dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=False, local_files_only=True,
    )
    log(f"loaded in {time.time()-t0:.0f}s, class={type(full).__name__}")

    params = dict(full.named_parameters())
    candidates = [name for name in params if name.endswith("layers.0.mlp.down_proj.weight")]
    if len(candidates) != 1:
        raise ValueError(f"无法唯一确定文本层：{candidates}")
    mount = candidates[0].removesuffix("layers.0.mlp.down_proj.weight")
    log("text tower mounted at:", mount)

    state = load_file(str(args.adapter / "adapter_model.safetensors"))
    pairs = plan_adapter_merge(state, params, mount, config["r"])
    log(f"target modules to merge: {len(pairs)}")
    apply_adapter_merge(pairs, params, scaling, torch)

    args.output.mkdir(parents=True, exist_ok=False)
    pending = args.output / ".merge_incomplete"
    pending.write_text("Model save in progress; do not use for inference.\n")
    log("saving to", args.output)
    full.save_pretrained(str(args.output), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=False, local_files_only=True)
    tokenizer.save_pretrained(str(args.output))
    for name in ("preprocessor_config.json", "video_preprocessor_config.json", "configuration.json",
                 "generation_config.json", "chat_template.jinja", "vocab.json", "merges.txt"):
        source = args.base_model / name
        if source.exists():
            shutil.copy2(source, args.output / name)
    pending.unlink()
    total = sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
    log(f"=== MERGE V3 DONE === {len(pairs)} modules merged, total {total / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
