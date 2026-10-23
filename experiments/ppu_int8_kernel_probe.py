#!/usr/bin/env python3
"""Opt-in compatibility/numerical probe for the installed SGLang W8A8 path.

Without --run-gpu this script uses only the standard library and prints a plan.
Run each backend in a fresh process: SGLang reads USE_KUNLUN during import.
This is neither a model-accuracy evaluation nor a performance benchmark. It
does not load a model/checkpoint, change backend configuration, or retry using
another backend. Native-library output remains on stdout/stderr; use tee when
running to retain it alongside the structured --output report.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch


# (N, K), single-device shapes from installed qwen3_5.py and qwen2_moe.py.
# Full attention doubles Q for the output gate: (24 * 2 + 4 + 4) * 256.
# Linear attention uses K=16*128 and V=48*128, with separate qkv and z.
SHAPE_ROLES = {
    (17408, 5120): "unfused_ffn_gate_or_up",
    (5120, 17408): "ffn_down",
    (34816, 5120): "fused_ffn_gate_up",
    (14336, 5120): "full_attention_fused_q_gate_k_v",
    (10240, 5120): "linear_attention_fused_q_k_v",
    (6144, 5120): "linear_attention_z",
    (5120, 6144): "full_attention_o_or_linear_attention_out",
}
SHAPES = tuple(SHAPE_ROLES)
OUTPUT_PARTITIONS = {
    (34816, 5120): [17408, 17408],
    (14336, 5120): [12288, 1024, 1024],
    (10240, 5120): [2048, 2048, 6144],
}
ENV_DEFAULTS = {"USE_KUNLUN": "1", "SGL_DEEPGEMM_DENSE": "1", "USE_ACEXT_CUDA": "1"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-gpu", action="store_true", help="Explicitly allocate GPU tensors and call real kernels.")
    p.add_argument("--m", type=int, nargs="+", default=[1], choices=[1, 16], help="Token batch sizes; use --m 1 16 for both.")
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reference-chunk", type=int, default=256, help="CPU reference output-channel chunk size.")
    p.add_argument("--output", type=Path, help="Save final structured report, including tracebacks on failure.")
    return p


def selected_backend(env: dict) -> str:
    if int(env["USE_KUNLUN"]):
        return "kunlun.int8_gemm_static_config"
    if int(env["SGL_DEEPGEMM_DENSE"]):
        return "deepgemm.w8a8_int8_quant_matmul"
    if int(env["USE_ACEXT_CUDA"]):
        return "acext.int8_gemm"
    return "sgl_kernel.int8_scaled_mm"


def cpu_reference(torch, xq, xs, weights, ws, chunk: int):
    """Exact CPU INT64 dot products, FP32 scaling, then BF16 output rounding."""
    x64 = xq.to(device="cpu", dtype=torch.int64)
    xs = xs.to(device="cpu", dtype=torch.float32).reshape(xq.shape[0], 1)
    ws = ws.to(device="cpu", dtype=torch.float32).reshape(-1)
    ref = torch.empty((xq.shape[0], weights.shape[0]), dtype=torch.bfloat16)
    for start in range(0, weights.shape[0], chunk):
        end = min(start + chunk, weights.shape[0])
        dot = x64 @ weights[start:end].to(dtype=torch.int64).t()
        scaled = dot.to(torch.float32) * xs
        scaled = scaled * ws[start:end].view(1, -1)
        ref[:, start:end] = scaled.to(torch.bfloat16)
    return ref


def check_activation_quantization(torch, original, xq, scales) -> dict:
    """Independently verify quantization, not just GEMM with captured integers.

    Installed int8_kernel.py uses max(abs(x), 1e-10)/127 and nearest integer
    rounding. Check against the original BF16 values on CPU; never call the
    installed quantizer to build this oracle. The slack is eight FP32 epsilons
    for division/reciprocal/scale rounding, not an additional INT8 rounding step.
    Both tie directions satisfy the 0.5-step bound (libdevice.round uses ties
    away from zero, unlike torch.round's ties-to-even).
    """
    if original.ndim != 2 or xq.shape != original.shape or scales.shape != (original.shape[0], 1):
        return {"passed": False, "error": "Unexpected activation/scales shapes"}
    if xq.dtype != torch.int8 or scales.dtype != torch.float32:
        return {"passed": False, "error": "Expected INT8 activation and FP32 scales"}
    x = original.detach().to(device="cpu", dtype=torch.float32)
    q = xq.detach().to(device="cpu", dtype=torch.int16)
    s = scales.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(s).all()) or not bool((s > 0).all()):
        return {"passed": False, "error": "Nonfinite activation/scales or nonpositive scale"}

    absmax = x.abs().amax(dim=1, keepdim=True)
    clamped_absmax = absmax.clamp_min(1e-10)
    expected_scale = clamped_absmax / 127.0
    slack = 8 * torch.finfo(torch.float32).eps
    scale_relative_error = (s.to(torch.float64) / expected_scale.to(torch.float64) - 1).abs()
    x64, s64 = x.to(torch.float64), s.to(torch.float64)
    # This reference uses original input and the independently computed peak.
    normalized = x64 * 127.0 / clamped_absmax.to(torch.float64)
    normalized_error = (normalized - q.to(torch.float64)).abs()
    rounding_limit = 0.5 + slack * 127.0
    reconstruction_error = (x64 - q.to(torch.float64) * s64).abs()
    reconstruction_limit = 0.5 * s64 + slack * clamped_absmax.to(torch.float64)
    saturation = q.abs() == 127
    false_saturation = saturation & (normalized.abs() < 126.5 - slack * 127.0)
    saturation_required = absmax[:, 0] >= 1e-10
    missing_peak_saturation = saturation_required & ~saturation.any(dim=1)
    zero_rows = absmax[:, 0] == 0
    nonzero_codes_for_zero_rows = zero_rows & (q != 0).any(dim=1)
    invalid_range = (q < -127) | (q > 127)
    checks = {
        "symmetric_int8_range": not bool(invalid_range.any()),
        "scale_matches_input_peak": bool((scale_relative_error <= slack).all()),
        "nearest_rounding_bound": bool((normalized_error <= rounding_limit).all()),
        "reconstruction_bound": bool((reconstruction_error <= reconstruction_limit).all()),
        "saturation_consistent_with_input": not bool(false_saturation.any() or missing_peak_saturation.any()),
        "zero_rows_produce_zero_codes": not bool(nonzero_codes_for_zero_rows.any()),
    }
    token_reports = []
    for row in range(x.shape[0]):
        token_reports.append({
            "token": row,
            "input_absmax": float(absmax[row, 0]),
            "scale": float(s[row, 0]),
            "expected_scale": float(expected_scale[row, 0]),
            "scale_relative_error": float(scale_relative_error[row, 0]),
            "max_abs_reconstruction_error": float(reconstruction_error[row].max()),
            "abs_reconstruction_error_bound": float(reconstruction_limit[row, 0]),
            "max_error_in_quantization_steps": float(normalized_error[row].max()),
            "saturated_elements": int(saturation[row].sum()),
            "invalid_saturation_elements": int(false_saturation[row].sum()),
        })
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "dtype": str(xq.dtype),
        "shape": list(xq.shape),
        "scale_min": float(s.min()),
        "scale_max": float(s.max()),
        "scale_relative_tolerance": slack,
        "rounding_error_bound_in_steps": rounding_limit,
        "symmetric_code_min": int(q.min()),
        "symmetric_code_max": int(q.max()),
        "invalid_range_elements": int(invalid_range.sum()),
        "tokens": token_reports,
    }


def compare(torch, actual, reference) -> dict:
    if actual.dtype != torch.bfloat16 or actual.shape != reference.shape:
        raise RuntimeError(f"Unexpected output: dtype={actual.dtype}, shape={tuple(actual.shape)}")
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(reference).all()):
        raise RuntimeError("Output/reference contains NaN or infinity")
    # Monotone BF16 bit ordering, with both signs of zero mapped to zero.
    def ordered(tensor):
        bits = tensor.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -32768 - bits, bits)

    ulps = (ordered(actual) - ordered(reference)).abs()
    delta = actual.to(torch.float64) - reference.to(torch.float64)
    reference_norm = float(torch.linalg.vector_norm(reference.to(torch.float64)))
    error_norm = float(torch.linalg.vector_norm(delta))
    relative_l2 = error_norm / reference_norm if reference_norm else (0.0 if error_norm == 0 else None)
    max_ulp = int(ulps.max())
    worst_flat = int(ulps.reshape(-1).argmax())
    row, col = divmod(worst_flat, reference.shape[1])
    ok = max_ulp <= 1 and relative_l2 is not None and relative_l2 <= 1e-3
    return {
        "passed": ok,
        "max_abs_error": float(delta.abs().max()),
        "relative_l2": relative_l2,
        "max_bfloat16_ulp": max_ulp,
        "exact_fraction": float((ulps == 0).to(torch.float64).mean()),
        "ulp_counts": {"0": int((ulps == 0).sum()), "1": int((ulps == 1).sum()), ">1": int((ulps > 1).sum())},
        "worst_ulp_element": {"row": row, "column": col, "actual": float(actual[row, col]), "reference": float(reference[row, col])},
    }


def run_case(torch, module, args, report: dict, m: int, n: int, k: int, case_seed: int) -> None:
    partitions = OUTPUT_PARTITIONS.get((n, k), [n])
    case = {"m": m, "n": n, "k": k, "role": SHAPE_ROLES[(n, k)], "output_partitions": partitions, "seed": case_seed, "status": "running", "observed_backend_calls": []}
    report["cases"].append(case)
    started = time.perf_counter()
    try:
        generator = torch.Generator(device="cpu").manual_seed(case_seed)
        weights = torch.randint(-127, 128, (n, k), dtype=torch.int8, generator=generator, device="cpu")
        # Nonconstant channel/token scales expose incorrect broadcast axes.
        weight_scales = torch.rand((n, 1), generator=generator, device="cpu") * 0.008 + 0.002
        input_cpu = torch.randn((m, k), generator=generator, device="cpu")
        input_cpu *= torch.linspace(0.25, 2.0, m, device="cpu").view(m, 1)
        if m == 16:
            # Exercise the installed absmax floor as well as ordinary rows.
            input_cpu[0].zero_()
            input_cpu[1] *= 1e-12
        input_cpu = input_cpu.to(torch.bfloat16)

        config = module.W8A8Int8Config({"quant_method": "w8a8_int8", "activation_scheme": "dynamic"})
        method = module.W8A8Int8LinearMethod(config)
        layer = torch.nn.Module()
        # Exercise the installed checkpoint layout and post-load transpose too.
        method.create_weights(layer, k, partitions, k, n, torch.bfloat16, weight_loader=lambda *a, **kw: None)
        with torch.no_grad():
            layer.weight.copy_(weights)
            layer.weight_scale.copy_(weight_scales)
        layer = layer.to(args.device)
        method.process_weights_after_loading(layer)
        x = input_cpu.to(args.device)
        case["loaded_weight"] = {"shape": list(layer.weight.shape), "dtype": str(layer.weight.dtype), "device": str(layer.weight.device), "stride": list(layer.weight.stride())}
        if layer.weight.dtype != torch.int8 or layer.weight.device.type != "cuda":
            raise RuntimeError("Expected a real device INT8 weight, not a CPU/unquantized path")

        captured = {}
        original_quantizer = module.per_token_quant_int8

        def capture_quantizer(*a, **kw):
            result = original_quantizer(*a, **kw)
            if captured:
                raise RuntimeError("Unexpected repeated activation quantization")
            captured["xq"], captured["xs"] = result
            return result

        def trace_call(name, function):
            def invoke(*a, **kw):
                event = {"backend": name, "completed": False}
                case["observed_backend_calls"].append(event)
                print(f"PROBE_BACKEND {name} M={m} N={n} K={k}", file=sys.stderr, flush=True)
                value = function(*a, **kw)
                event["completed"] = True
                return value
            return invoke

        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "per_token_quant_int8", capture_quantizer))
            # These wrappers only observe/delegate; none substitutes kernel math.
            if hasattr(method, "Int8Ops"):
                original = method.Int8Ops.int8_gemm_static_config
                method.Int8Ops = SimpleNamespace(int8_gemm_static_config=trace_call("kunlun.int8_gemm_static_config", original))
            for attr, name in (("w8a8_int8_quant_matmul", "deepgemm.w8a8_int8_quant_matmul"), ("acext_int8_gemm", "acext.int8_gemm"), ("int8_scaled_mm", "sgl_kernel.int8_scaled_mm")):
                function = getattr(module, attr, None)
                if function is not None:
                    stack.enter_context(patch.object(module, attr, trace_call(name, function)))
            torch.cuda.synchronize()
            apply_started = time.perf_counter()
            with torch.inference_mode():
                output = method.apply(layer, x, bias=None)
            torch.cuda.synchronize()
            case["apply_and_sync_seconds_including_first_call_overheads"] = time.perf_counter() - apply_started

        calls = case["observed_backend_calls"]
        if len(calls) != 1 or calls[0] != {"backend": report["expected_backend"], "completed": True}:
            raise RuntimeError(f"Backend route mismatch: expected {report['expected_backend']}, observed {calls}")
        if not captured or captured["xq"].dtype != torch.int8:
            raise RuntimeError("Did not capture the real apply's INT8 activation")
        if captured["xq"].device.type != "cuda" or output.device.type != "cuda":
            raise RuntimeError("Activation/output unexpectedly left the GPU")
        xq = captured["xq"].detach().cpu()
        xs = captured["xs"].detach().cpu()
        if xq.shape != (m, k) or xs.shape != (m, 1) or not bool(torch.isfinite(xs).all()) or not bool((xs > 0).all()):
            raise RuntimeError("Unexpected activation quantizer shape/scales")
        actual = output.detach().cpu()
        case["activation_quantization"] = check_activation_quantization(torch, input_cpu, xq, xs)
        reference_started = time.perf_counter()
        reference = cpu_reference(torch, xq, xs, weights, weight_scales, args.reference_chunk)
        case["cpu_integer_reference_seconds"] = time.perf_counter() - reference_started
        case["numerical"] = compare(torch, actual, reference)
        case["status"] = "passed" if case["numerical"]["passed"] and case["activation_quantization"]["passed"] else "failed"
    except Exception as exc:
        case["status"] = "failed"
        case["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        case["wall_seconds"] = time.perf_counter() - started


def run_gpu(args, report: dict) -> None:
    import torch
    import sglang.srt.layers.quantization.w8a8_int8 as module

    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A real CUDA-compatible PPU/GPU is required; no CPU fallback")
    torch.cuda.set_device(args.device)
    if not module._is_cuda or module._is_cpu:
        raise RuntimeError("Installed SGLang selected a non-CUDA platform; refusing fallback")
    report["packages"] = {}
    for name in ("torch", "sglang", "sgl-kernel", "kunlun_extops", "deep_gemm", "acext"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    report["device"] = {"name": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()), "device": args.device}
    report["implementation"] = {"file": module.__file__, "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), "apply_signature": str(inspect.signature(module.W8A8Int8LinearMethod.apply)), "apply_source": inspect.getsource(module.W8A8Int8LinearMethod.apply)}
    if bool(module.USE_KUNLUN) != bool(int(report["backend_environment"]["USE_KUNLUN"])):
        raise RuntimeError("USE_KUNLUN differs from imported module; use a fresh process")
    for m in dict.fromkeys(args.m):
        for n, k in SHAPES:
            # The same shape/seed produces the same input in separate backend runs.
            run_case(torch, module, args, report, m, n, k, args.seed + m + n + k)
            if report["cases"][-1]["status"] != "passed":
                raise RuntimeError("Numerical gate failed; no retry or backend fallback")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.reference_chunk < 1:
        parser().error("--reference-chunk must be positive")
    env = {key: os.environ.get(key, default) for key, default in ENV_DEFAULTS.items()}
    report = {
        "status": "planned",
        "purpose": "Dense INT8 compatibility/numerical gate only; not model accuracy or throughput",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_execution_authorized_by_flag": args.run_gpu,
        "backend_environment": env,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "planned_shapes": [{"m": m, "n": n, "k": k, "role": SHAPE_ROLES[(n, k)], "output_partitions": OUTPUT_PARTITIONS.get((n, k), [n])} for m in dict.fromkeys(args.m) for n, k in SHAPES],
        "seed": args.seed,
        "reference": "CPU INT64 dot(actual captured x_q, original INT8 weight); FP32 multiply token then channel scales; round BF16",
        "gate": {"max_bfloat16_ulp": 1, "max_relative_l2": 1e-3, "finite_only": True, "observed_backend_must_match": True},
        "activation_gate": {"source": "installed int8_kernel.py per_token_quant_int8: scale=max(abs(x),1e-10)/127", "independent_cpu_check": True, "nearest_rounding_bound_steps": 0.5, "floating_point_slack": "8 * FP32 epsilon", "require_finite_positive_scales": True, "require_symmetric_range": [-127, 127], "check_saturation": True},
        "synthetic_activation_rows": "M=1: random BF16; M=16: one zero row, one tiny row below the 1e-10 absmax floor, fourteen random BF16 rows",
        "limitations": ["Synthetic dense layers only; no attention/whole-model/adapter/checkpoint validation", "Provider-internal dispatch is not profiled; observed calls identify the SGLang selected backend entry", "First-call wall time includes initialization/JIT; no performance claims"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    exit_code = 0
    started = time.perf_counter()
    try:
        report["expected_backend"] = selected_backend(env)
        if args.run_gpu:
            run_gpu(args, report)
            report["status"] = "passed"
        else:
            report["next_step"] = "Add --run-gpu only when GPU execution is authorized; test backends in separate processes."
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        traceback.print_exc(file=sys.stderr)
        exit_code = 1
    report["wall_seconds"] = time.perf_counter() - started
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
