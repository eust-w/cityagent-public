"""Version-bound real integration evidence. Mock traces never grant acceptance."""

import importlib.metadata
import json
from functools import lru_cache
from pathlib import Path

from .execution_audit import _native_errors
from .run_state import file_digest, json_digest


@lru_cache(maxsize=1)
def runtime_identity():
    distribution = importlib.metadata.distribution("tongagents")
    sdk_files = {
        str(p): file_digest(distribution.locate_file(p))
        for p in distribution.files or []
        if str(p).endswith((".py", ".so"))
    }
    if not sdk_files:
        raise ValueError("installed SDK source fingerprint unavailable")
    return {
        "sdk_version": distribution.version,
        "sdk_sha256": json_digest(sdk_files),
        "adapter_sha256": file_digest(Path(__file__).with_name("tong_workflow.py")),
    }


def validate_acceptance(path):
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") != 1 or value.get("runtime") != runtime_identity():
        raise ValueError("SDK acceptance does not match installed SDK/adapter")
    workflow = value.get("workflow", {})
    if (
        workflow.get("engine") != "tongagents"
        or workflow.get("non_sdk") is not False
        or workflow.get("status") != "success"
        or workflow.get("sdk_version") != value["runtime"]["sdk_version"]
        or _native_errors(workflow)
    ):
        raise ValueError("SDK acceptance lacks real native stage evidence")
    if workflow.get("mock") or workflow.get("simulated"):
        raise ValueError("simulated SDK cannot be accepted")
    calls = value.get("model_calls", [])
    if (
        not calls
        or any(c.get("mock") or c.get("simulated") for c in calls)
        or not any(
            c.get("status") == "success"
            and c.get("finish_reason") == "stop"
            and c.get("output_tokens", 0) > 0
            for c in calls
        )
    ):
        raise ValueError("SDK acceptance lacks completed model request")
    for label in ("input", "output"):
        item = value[label]
        if file_digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("SDK acceptance artifact changed")
    if len(workflow.get("stage_records", [])) != 4 or any(
        s.get("status") != "success" for s in workflow["stage_records"]
    ):
        raise ValueError("SDK acceptance incomplete stages")
    return value
