"""Frozen configuration validation and measured replacement gates."""

import math

DEFAULT = {
    "schema_version": 1,
    "workflow": "chunked",
    "input_budget": 8192,
    "output_budget": 4096,
    "batch_size": 1,
    "overlap": 1,
    "seed": 17,
    "prompt": "source-fragment-v1",
    "engine_parameters": {},
}
ALLOWED_PARAMETERS = {
    "enable_prefix_caching",
    "enforce_eager",
    "gpu_memory_utilization",
}


def validate_config(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT):
        raise ValueError("unknown candidate fields")
    config = {**DEFAULT, **value}
    if config["schema_version"] != 1 or config["workflow"] not in (
        "whole",
        "chunked",
        "bounded-react",
    ):
        raise ValueError("invalid candidate workflow/version")
    for key in ("input_budget", "output_budget", "batch_size", "overlap", "seed"):
        if type(config[key]) is not int or config[key] < (
            0 if key in ("overlap", "seed") else 1
        ):
            raise ValueError("invalid " + key)
    if config["batch_size"] not in (1, 2, 4, 8) or config["prompt"] not in (
        "source-fragment-v1",
        "source-fragment-concise-v1",
        "source-fragment-exact-v1",
        "source-fragment-exact-v2",
        "source-fragment-lean-v1",
        "whole-flat-v1",
    ):
        raise ValueError("unsupported batch/prompt")
    if config["prompt"] == "whole-flat-v1" and config["workflow"] != "whole":
        raise ValueError("whole-flat-v1 requires the whole workflow")
    if config["prompt"] in ("source-fragment-exact-v1", "source-fragment-exact-v2",
                            "source-fragment-lean-v1") and config["workflow"] == "whole":
        raise ValueError(config["prompt"] + " requires a fragment workflow")
    params = config["engine_parameters"]
    if not isinstance(params, dict) or set(params) - ALLOWED_PARAMETERS:
        raise ValueError("unsupported engine parameter")
    for key in ("enable_prefix_caching", "enforce_eager"):
        if key in params and type(params[key]) is not bool:
            raise ValueError("engine flag must be bool")
    memory = params.get("gpu_memory_utilization", 0.9)
    if type(memory) not in (int, float) or not 0 < memory < 1:
        raise ValueError("invalid memory utilization")
    return config


def select_candidate(baseline, candidates):
    eligible = []
    for item in candidates:
        if (
            item.get("conditions") != baseline.get("conditions")
            or item.get("repeated") is not True
        ):
            continue
        timing = item.get("wall_seconds")
        failures = item.get("failures")
        if (
            type(timing) not in (int, float)
            or not math.isfinite(timing)
            or timing <= 0
            or type(failures) is not int
            or failures < 0
        ):
            continue
        metrics = item.get("f1", [])
        if (
            len(metrics) != 3
            or len(baseline.get("f1", [])) != 3
            or any(
                type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
                for x in metrics
            )
        ):
            continue
        if (
            any(x < y for x, y in zip(metrics, baseline["f1"]))
            or item.get("wall_seconds", float("inf")) > baseline["wall_seconds"] * 0.8
            or item.get("failures", float("inf")) > baseline["failures"]
        ):
            continue
        eligible.append(item)
    return min(eligible, key=lambda c: c["wall_seconds"]) if eligible else baseline
