"""Fail-closed checks for content-bound inference receipts.

These checks establish consistency of recorded evidence, not cryptographic
attestation of an SDK/backend. ``verification=verified`` is reserved for an
integration that has passed a real SDK acceptance run; mock tests cannot grant it.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

STAGES = ("prepare", "infer", "validate", "render")
PROFILES = ("contest-sdk", "legacy")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _seconds(value: Any, *, positive: bool = True) -> bool:
    return (type(value) in (int, float) and (type(value) is int or math.isfinite(value))
            and (value > 0 if positive else value >= 0))


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    except UnicodeError:
        return None


def _relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or value == "." or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and path.as_posix() == value


def _native_errors(workflow: dict) -> list[str]:
    errors = []
    events = workflow.get("native_events")
    if not isinstance(events, list) or not events:
        return ["workflow.native_events: missing native SDK events"]
    transitions = []
    active = None
    for index, event in enumerate(events):
        if not isinstance(event, dict) or not isinstance(event.get("data"), dict):
            errors.append(f"native_events[{index}]: invalid event/data")
            continue
        kind, stage = event.get("type"), event["data"].get("node_name")
        if kind not in ("node_start", "node_end", "node_process") or stage not in (*STAGES, "__end__"):
            errors.append(f"native_events[{index}]: failed/unknown native event")
            continue
        if kind == "node_process":
            if active != stage:
                errors.append(f"native_events[{index}]: process event outside its active node")
        else:
            transitions.append((kind, stage))
            if kind == "node_start":
                if active is not None:
                    errors.append(f"native_events[{index}]: overlapping node lifecycle")
                active = stage
            else:
                if active != stage:
                    errors.append(f"native_events[{index}]: end without matching start")
                active = None
    # Official debug examples include the native __end__ control node. Its pair
    # may follow the stages but cannot replace or interrupt an application node.
    ordered_stages = list(STAGES)
    if any(stage == "__end__" for _, stage in transitions):
        ordered_stages.append("__end__")
    expected = [(kind, stage) for stage in ordered_stages for kind in ("node_start", "node_end")]
    if transitions != expected:
        errors.append("workflow.native_events: require prepare/infer/validate/render start/end in order")
    return errors


def validate_execution_record(
    record: Any, *, expected_document: dict, expected_run_fingerprint: str,
    output_path: Path, profile: str = "contest-sdk",
    require_strong: bool = False, source_path: Path | None = None,
    expected_model_identity_sha256: str | None = None,
) -> list[str]:
    """Return violations; an empty list means this receipt satisfies the profile.

    ``expected_document`` must be calculated from the current input, and the run
    fingerprint must come from the expected model/config/code/prompt identity.
    Failed retry attempts may precede a valid final answer. They never count as
    accepted calls, and must retain their failure details.
    """
    errors: list[str] = []
    if profile not in PROFILES:
        return [f"unknown audit profile: {profile}"]
    if not isinstance(record, dict):
        return ["record must be an object"]
    if type(record.get("schema_version")) is not int or record["schema_version"] not in (1, 2, 3):
        errors.append("schema_version must be 1, 2 or 3; legacy filename caches are unsupported")
    if (not isinstance(expected_document, dict) or not _relative(expected_document.get("relative_path"))
            or not _sha(expected_document.get("sha256"))):
        errors.append("expected_document must contain a safe relative path and input SHA256")
    if record.get("document") != expected_document:
        errors.append("document: input path/SHA256 mismatch")
    if not _sha(expected_run_fingerprint) or record.get("run_fingerprint") != expected_run_fingerprint:
        errors.append("run_fingerprint: model/config/code/prompt identity mismatch")
    if record.get("status") != "success":
        errors.append("status must be success")
    if not _seconds(record.get("seconds")):
        errors.append("seconds must be finite and positive")
    validation = record.get("model_validation")
    if (not isinstance(validation, dict) or validation.get("valid") is not True
            or validation.get("errors") or validation.get("truncated") is True):
        errors.append("model_validation: complete valid model output required")

    calls = record.get("model_calls")
    if not isinstance(calls, list) or not calls:
        errors.append("model_calls: at least one successful actual model call required")
        calls = []
    call_ids, accepted_ids = [], []
    for index, call in enumerate(calls):
        prefix = f"model_calls[{index}]"
        if not isinstance(call, dict):
            errors.append(prefix + ": invalid call")
            continue
        call_id = call.get("call_id")
        if not isinstance(call_id, str) or not call_id.strip() or call_id in call_ids:
            errors.append(prefix + ": call_id must be nonempty and unique")
        else:
            call_ids.append(call_id)
        if not _sha(call.get("prompt_sha256")):
            errors.append(prefix + ": prompt SHA256 required")
        if type(call.get("prompt_tokens")) is not int or call["prompt_tokens"] <= 0:
            errors.append(prefix + ": positive actual prompt_tokens required")
        if not _seconds(call.get("seconds")):
            errors.append(prefix + ": positive finite seconds required")
        if call.get("simulated") is True or call.get("mock") is True:
            errors.append(prefix + ": simulated model calls are not evidence")
        if call.get("status") == "success":
            accepted_ids.append(call_id)
            if (call.get("finish_reason") != "stop" or call.get("errors")
                    or call.get("truncated") is True):
                errors.append(prefix + ": successful call must be complete, stop, and error-free")
            if not _sha(call.get("output_sha256")):
                errors.append(prefix + ": output SHA256 required")
            if type(call.get("output_tokens")) is not int or call["output_tokens"] <= 0:
                errors.append(prefix + ": positive actual output_tokens required")
        elif call.get("status") == "failed":
            if not isinstance(call.get("errors"), list) or not call["errors"]:
                errors.append(prefix + ": failed retry must retain failure details")
        else:
            errors.append(prefix + ": status must be success or failed")
    if record.get("schema_version") == 1 and (not accepted_ids or not calls or not isinstance(calls[-1], dict) or calls[-1].get("status") != "success"):
        errors.append("model_calls: final attempt must be a successful complete model call")
    if "raw_output" in record and record.get("schema_version") == 1:
        raw_digest = _text_sha(record["raw_output"])
        if (raw_digest is None or not calls or not isinstance(calls[-1], dict)
                or raw_digest != calls[-1].get("output_sha256")):
            errors.append("raw_output: saved model response does not match final call output SHA256")

    if record.get("schema_version") == 1 and (
        "visual_evidence" in record
        or any(isinstance(call, dict) and call.get("role") == "visual-evidence" for call in calls)
    ):
        errors.append("visual evidence requires a V2/V3 source-bound record")
    if record.get("schema_version") in (2, 3):
        errors.extend(_v2_errors(record, accepted_ids))
    if record.get("schema_version") == 3:
        errors.extend(_v3_errors(record))
    if require_strong or "generation_evidence" in record or "visual_evidence" in record:
        errors.extend(_reconstruction_errors(
            record, output_path, source_path=source_path, require_strong=require_strong,
            expected_model_identity_sha256=expected_model_identity_sha256))
    if "facts" in record or "facts_sha256" in record:
        from .facts import validate_fact_graph
        from .run_state import json_digest

        try:
            facts = record["facts"]
            if (json_digest(facts) != record.get("facts_sha256")
                    or facts["document"]["sha256"] != expected_document["sha256"]):
                errors.append("facts: content/document identity mismatch")
            errors.extend(validate_fact_graph(record))
        except (KeyError, TypeError, ValueError):
            errors.append("facts: invalid evidence graph")

    workflow = record.get("workflow")
    if not isinstance(workflow, dict):
        workflow = {}
    if workflow.get("status") != "success":
        errors.append("workflow.status must be success")
    if workflow.get("mock") is True or workflow.get("simulated") is True:
        errors.append("workflow: simulated execution is not evidence")
    stages = workflow.get("stage_records")
    if not isinstance(stages, list) or len(stages) != len(STAGES):
        errors.append("workflow.stage_records: four successful stages required")
        stages = []
    for expected_stage, stage in zip(STAGES, stages):
        if (not isinstance(stage, dict) or stage.get("stage") != expected_stage
                or stage.get("status") != "success"
                or not _seconds(stage.get("duration_seconds"), positive=False)):
            errors.append(f"workflow.stage_records: invalid {expected_stage} stage")
            continue
        if expected_stage == "infer":
            if stage.get("model_call_ids") != call_ids:
                errors.append("workflow.infer: model_call_ids must match all actual attempts")
            if stage.get("accepted_model_call_ids") != accepted_ids:
                errors.append("workflow.infer: accepted_model_call_ids must match successful calls")
    if profile == "contest-sdk":
        if workflow.get("engine") != "tongagents" or workflow.get("non_sdk") is not False:
            errors.append("workflow: contest-sdk requires real tongagents execution")
        version = workflow.get("sdk_version")
        if not isinstance(version, str) or not version.strip() or not any(c.isdigit() for c in version):
            errors.append("workflow.sdk_version: installed SDK version required")
        if workflow.get("verification") != "verified":
            errors.append("workflow.verification: real SDK integration is unverified")
        if workflow.get("native_trace_validated") is not True:
            errors.append("workflow.native_trace_validated must be true")
        errors.extend(_native_errors(workflow))
    elif workflow.get("engine") != "direct" or workflow.get("non_sdk") is not True:
        errors.append("workflow: legacy is reserved for explicit direct baselines")

    output = record.get("output")
    output_path = Path(output_path)
    relative = expected_document.get("relative_path") if isinstance(expected_document, dict) else None
    expected_name = PurePosixPath(relative).stem + ".docx" if _relative(relative) else ""
    if (not isinstance(output, dict) or output.get("name") != output_path.name
            or output_path.name != expected_name or not _sha(output.get("sha256"))):
        errors.append("output: report name/SHA256 does not match the input")
    else:
        try:
            if output_path.is_symlink() or _file_sha(output_path) != output["sha256"]:
                errors.append("output: report bytes do not match recorded SHA256")
        except OSError:
            errors.append("output: report is missing or unreadable")
    return errors


def audit_execution_records(
    input_dir: Path, result_dir: Path, records_dir: Path, expected_run_fingerprint: str,
    profile: str = "contest-sdk", *, require_strong: bool = False,
) -> dict[str, Any]:
    """Audit one receipt per current input; formal callers require fresh derivation and timing."""
    errors, details = [], []
    input_dir, result_dir, records_dir = map(Path, (input_dir, result_dir, records_dir))
    inputs = sorted(p for p in input_dir.rglob("*") if p.is_file()
                    and p.suffix.lower() in (".doc", ".docx") and not p.name.startswith("~$")) if input_dir.is_dir() else []
    if not inputs:
        errors.append("input_dir: explicit nonempty report directory required")
    if profile not in PROFILES:
        errors.append(f"unknown audit profile: {profile}")
    if not _sha(expected_run_fingerprint):
        errors.append("expected_run_fingerprint must be a SHA256 digest")
    if not records_dir.is_dir() or records_dir.is_symlink():
        errors.append("records_dir: a real execution records directory is required")
    records = {}
    record_files = sorted(records_dir.glob("*.json")) if records_dir.is_dir() else []
    for path in record_files:
        try:
            if path.is_symlink():
                raise ValueError("symlink records are unsupported")
            record = json.loads(path.read_text(encoding="utf-8"))
            document = record.get("document") if isinstance(record, dict) else None
            if (not isinstance(document, dict) or not _relative(document.get("relative_path"))
                    or not _sha(document.get("sha256"))):
                raise ValueError("missing content-bound document identity")
            key = (document["relative_path"], document["sha256"])
            if key in records:
                raise ValueError("duplicate receipt for the same document")
            records[key] = record
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"records/{path.name}: {exc}")
    expected_model_identity_sha256 = None
    if any("visual_evidence" in record for record in records.values()):
        try:
            expected_model_identity_sha256 = _model_identity_from_manifest(
                records_dir.parent / "run_manifest.json", expected_run_fingerprint)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append("visual model identity: " + str(exc))
    used, names = set(), set()
    for path in inputs:
        name = path.stem + ".docx"
        if name.casefold() in names:
            errors.append(f"input_dir: output filename collision: {name}")
        names.add(name.casefold())
        try:
            if path.is_symlink():
                raise ValueError("symlink inputs are unsupported")
            document = {"relative_path": path.relative_to(input_dir).as_posix(), "sha256": _file_sha(path)}
            key = (document["relative_path"], document["sha256"])
            used.add(key)
            issues = validate_execution_record(records.get(key), expected_document=document,
                expected_run_fingerprint=expected_run_fingerprint, output_path=result_dir / name, profile=profile,
                require_strong=require_strong, source_path=path if require_strong else None,
                expected_model_identity_sha256=expected_model_identity_sha256)
        except (OSError, ValueError) as exc:
            issues = [str(exc)]
        details.append({"input": path.name, "passed": not issues, "errors": issues})
        errors.extend(f"{path.name}: {issue}" for issue in issues)
    for path, digest in sorted(records.keys() - used):
        errors.append(f"records: stale/unexpected input receipt: {path} ({digest[:12]})")
    timing = audit_timing_records(result_dir, records_dir) if require_strong else None
    if timing:
        errors.extend(timing["errors"])
    return {"passed": not errors, "profile": profile, "sdk_passed": not errors and profile == "contest-sdk",
            "expected_inputs": len(inputs), "record_files": len(record_files), "documents": details,
            "run_fingerprint": expected_run_fingerprint, "errors": errors, "timing": timing}


def audit_timing_records(result_dir: Path, records_dir: Path) -> dict[str, Any]:
    """Bind the delivered total to every durable request, including failed attempts."""
    import math
    import sqlite3

    from .rsi_acceptance import audit_call_ledger
    from .run_state import file_digest

    result_dir, records_dir = Path(result_dir), Path(records_dir)
    result = {"passed": False, "errors": []}
    try:
        paths = sorted(records_dir.glob("*.json"))
        if any(path.is_symlink() for path in paths):
            raise ValueError("timing records cannot be symlinked")
        records = [json.loads(path.read_text()) for path in paths]
        ledger_path = records_dir.parent / "timing.sqlite"
        audited = audit_call_ledger(ledger_path, records)
        for record in records:
            measured = record.get("seconds")
            total = sum(call["seconds"] for call in record["model_calls"])
            if not _seconds(measured) or not math.isclose(measured, total, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError("report generation time differs from its unique calls")
        timing_path = result_dir / "infer_time.json"
        if timing_path.is_symlink():
            raise ValueError("official timing cannot be symlinked")
        seconds = json.loads(timing_path.read_text())["infer_time"]
        total = audited["generation_seconds"]
        # run.py exports the conservative integer ceiling; the evaluation
        # protocol retains the exact float. Neither may undercount a request.
        if not _seconds(seconds) or not (
            math.isclose(seconds, total, rel_tol=1e-9, abs_tol=1e-6)
            or seconds == math.ceil(total)
        ):
            raise ValueError("official generation time differs from actual unique-call ledger")
        result.update(audited, timing_sha256=file_digest(ledger_path),
                      timing_json_sha256=file_digest(timing_path), official_seconds=seconds)
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        result["errors"].append("timing: " + str(exc))
    result["passed"] = not result["errors"]
    return result


def _model_identity_from_manifest(path, expected_run_fingerprint):
    """Use the externally selected run, never a model hash supplied by a report."""
    from .run_state import json_digest

    if path.is_symlink() or not path.is_file():
        raise ValueError("actual run manifest required for visual evidence")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if (manifest.get("run_fingerprint") != expected_run_fingerprint
            or json_digest(identity) != expected_run_fingerprint):
        raise ValueError("run manifest differs from expected run identity")
    options = identity["options"]
    model = options.get("model_manifest", options.get("model"))
    if not isinstance(model, dict) or not model:
        raise ValueError("fixed model manifest identity missing from run")
    return json_digest(model)


def _visual_call_bindings(record):
    """Check typed upstream participation; actual source/probe checks are separate."""
    visual_calls = [call for call in record.get("model_calls", [])
                    if isinstance(call, dict) and call.get("role") == "visual-evidence"]
    visual = record.get("visual_evidence")
    if visual is None:
        regions = (record.get("snapshot") or {}).get("regions", [])
        if (visual_calls or record.get("visual_contributing_call_ids")
                or any(str(region.get("kind", "")).startswith("visual-") for region in regions)):
            raise ValueError("visual role/source requires upstream visual evidence")
        return set()
    if not isinstance(visual, dict) or record.get("schema_version") not in (2, 3):
        raise ValueError("visual evidence requires a V2/V3 envelope")
    for key in ("calls", "raw", "evidence", "errors"):
        if not isinstance(visual.get(key), list):
            raise TypeError("visual envelope missing " + key)
    if visual_calls != visual["calls"]:
        raise ValueError("visual upstream calls differ from main model ledger receipts")
    identity = visual["plan"]["identity"]
    if (visual.get("plan_sha256") != visual["plan"].get("plan_sha256")
            or visual.get("model_identity_sha256") != identity["model_identity_sha256"]
            or not _sha(visual.get("model_identity_sha256"))
            or visual.get("trial_id") != record.get("run_fingerprint")
            or identity["config"]["trial_id"] != record.get("run_fingerprint")):
        raise ValueError("visual model/config/trial identity mismatch")
    successful = {call["call_id"] for call in visual_calls if call.get("status") == "success"}
    contributors = record.get("visual_contributing_call_ids")
    if (not isinstance(contributors, list) or len(set(contributors)) != len(contributors)
            or set(contributors) != successful
            or successful.intersection(record.get("contributing_call_ids", []))):
        raise ValueError("visual contributors must exactly identify successful upstream calls")
    evidence_ids = [item["call_id"] for item in visual["evidence"]]
    if len(set(evidence_ids)) != len(evidence_ids) or set(evidence_ids) != successful:
        raise ValueError("successful visual calls must supply parsed evidence")
    regions = (record.get("snapshot") or {}).get("regions", [])
    participating = {region.get("location", {}).get("call_id") for region in regions
                     if str(region.get("kind", "")).startswith("visual-")}
    if participating != successful:
        raise ValueError("every successful visual call must contribute actual source regions")
    responses = record.get("generation_evidence", {}).get("responses", {})
    for item in visual["raw"]:
        response = item.get("response", {})
        if response.get("call_id") in successful and responses.get(response["call_id"]) != response.get("text"):
            raise ValueError("visual raw response differs from generation evidence")
    return successful


def _rebuild_visual_source(record, native_snapshot, *, expected_model_identity_sha256=None,
                           verify_capability=False):
    from .run_state import json_digest
    from .visual_evidence import rebuild_visual_snapshot

    _visual_call_bindings(record)
    visual = record["visual_evidence"]
    identity = visual["plan"]["identity"]
    prepared = Path(record["source_evidence"]["prepared_docx_path"])
    if Path(identity["source_path"]).resolve() != prepared.resolve():
        raise ValueError("visual source path differs from prepared document")
    if verify_capability:
        if not _sha(expected_model_identity_sha256) or identity["model_identity_sha256"] != expected_model_identity_sha256:
            raise ValueError("visual source lacks independently bound fixed model identity")
        from .vision_capability import validate_vision_capability

        path = Path(visual["capability_path"])
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("actual bounded vision capability proof required")
        proof = json.loads(path.read_text(encoding="utf-8"))
        model = proof["model_identity"]
        if json_digest(model) != expected_model_identity_sha256:
            raise ValueError("vision capability model differs from expected fixed model")
        if validate_vision_capability(path, model) != visual.get("capability"):
            raise ValueError("vision capability summary differs from actual proof/ledger")
    _visual_shared_budget(record, native_snapshot=native_snapshot)
    return rebuild_visual_snapshot(native_snapshot, visual["plan"], visual["evidence"],
                                   visual["calls"], visual["raw"])


def _visual_shared_budget(record, *, native_snapshot=None):
    """Bind the frozen native/visual task counts; derived text cannot add budget."""
    from .chunked_inference import _validate_prefix_budget
    from .long_document import source_metadata
    from .run_state import json_digest

    visual = record.get("visual_evidence")
    shared = record.get("shared_budget")
    if visual is None:
        if shared is not None:
            raise ValueError("shared visual budget requires source evidence")
        return None
    jobs = visual["plan"]["jobs"]
    if not jobs:
        if shared is not None:
            raise ValueError("empty visual job plan cannot grant a prefix budget")
        return None
    if not isinstance(shared, dict):
        raise TypeError("visual requests require one shared report budget")
    prefix = _validate_prefix_budget(shared.get("prefix"))
    if prefix is None:
        raise ValueError("visual prefix budget missing")
    _, native_plan = _snapshot_and_plan({**record, "chunk_plan": visual["native_chunk_plan"]})
    native_digest = native_plan.digest
    if (native_digest != visual.get("native_plan_sha256")
            or native_plan.snapshot_sha256 != visual["plan"]["identity"]["native_snapshot_sha256"]
            or prefix["initial_text_tasks"] != len(native_plan.tasks)
            or prefix["initial_visual_jobs"] != len(jobs)
            or prefix["used_calls"] != len(visual["calls"])
            or prefix["identity"] != json_digest({"run": record["run_fingerprint"],
                "document": record["document"], "visual_plan": visual["plan"]["plan_sha256"],
                "native_plan": native_digest})):
        raise ValueError("visual prefix identity/counts differ from frozen source plans")
    effective = record["chunk_plan"]
    if any(visual["native_chunk_plan"][key] != effective[key]
           for key in ("input_budget", "output_budget", "instruction")):
        raise ValueError("visual native and effective plan budgets/templates differ")
    if native_snapshot is not None:
        if native_plan.snapshot_sha256 != native_snapshot.digest or not native_plan.coverage(native_snapshot)["complete"]:
            raise ValueError("visual initial native task coverage changed")
        sources = native_snapshot.sources()
        task_ids = set()
        for task in native_plan.tasks:
            if task.task_id in task_ids or task.task_id != json_digest(task.owned_keys)[:24]:
                raise ValueError("visual native task identity changed")
            task_ids.add(task.task_id)
            for span in [*task.owned, *task.context]:
                source = sources.get(span.source_id)
                if (source is None or type(span.start) is not int or type(span.end) is not int
                        or not (0 <= span.start < span.end <= len(source.text)
                                or (source.text == "" and span.start == span.end == 0))
                        or span.text != source.text[span.start:span.end]
                        or json_digest(span.metadata) != json_digest(source_metadata(source))):
                    raise ValueError("visual native plan source text/metadata changed")
    calls = record["model_calls"]
    text_count = sum(call.get("role") != "visual-evidence" for call in calls)
    if (type(shared.get("text_issued_calls")) is not int or shared["text_issued_calls"] != text_count
            or type(shared.get("total_issued_calls")) is not int or shared["total_issued_calls"] != len(calls)
            or shared["total_issued_calls"] != shared["text_issued_calls"] + prefix["used_calls"]
            or shared.get("total_call_limit") != prefix["total_call_limit"]
            or shared["total_issued_calls"] > prefix["total_call_limit"]):
        raise ValueError("shared visual/text call reservations differ from actual unique attempts")
    text_elapsed, total_elapsed = shared.get("text_elapsed_seconds"), shared.get("total_elapsed_seconds")
    visual_elapsed = visual.get("spent_seconds")
    if (not _seconds(text_elapsed, positive=False) or not _seconds(total_elapsed)
            or not _seconds(visual_elapsed, positive=False)
            or not math.isclose(total_elapsed, prefix["elapsed_seconds"] + text_elapsed, abs_tol=1e-6)
            or total_elapsed > 1200 or prefix["elapsed_seconds"] < visual_elapsed
            or visual_elapsed + 1e-6 < sum(call["seconds"] for call in visual["calls"])):
        raise ValueError("shared visual/text elapsed budget changed or exhausted")
    return shared


def _v2_errors(record, accepted_ids):
    from .long_document import (
        ChunkPlan,
        ChunkTask,
        DocumentSnapshot,
        SourceRegion,
        SourceSlice,
    )
    from .run_state import json_digest
    errors = []
    try:
        snapshot_values = dict(record['snapshot'])
        snapshot_values['regions'] = [SourceRegion(**r) for r in snapshot_values['regions']]
        snapshot = DocumentSnapshot(**snapshot_values)
        value = record['chunk_plan']
        plan = ChunkPlan(value['snapshot_sha256'],[ChunkTask(t['task_id'],
            [SourceSlice(**s) for s in t['owned']],[SourceSlice(**s) for s in t['context']],t['prompt_tokens'])
            for t in value['tasks']],value['input_budget'],value['output_budget'],value['instruction'])
        coverage = plan.coverage(snapshot)
        recorded_coverage = record.get('coverage')
        coverage_matches = recorded_coverage == coverage
        if snapshot.parser in ('ordered-word-xml-v1', 'bound-text-v1') and isinstance(recorded_coverage, dict):
            legacy_keys = ('complete', 'missing', 'unexpected', 'sources')
            coverage_matches = coverage_matches or recorded_coverage == {key: coverage[key] for key in legacy_keys}
        if (snapshot.digest != record['snapshot_sha256'] or plan.digest != record['plan_sha256']
                or not coverage['complete'] or not coverage_matches):
            errors.append('v2: source coverage/snapshot/plan mismatch')
        contributors = record['contributing_call_ids']
        if not contributors or len(set(contributors)) != len(contributors) or not set(contributors).issubset(accepted_ids):
            errors.append('v2: contributing calls must be unique successful calls')
        calls = {c['call_id']:c for c in record['model_calls']}
        expected = {t.task_id:t.owned_keys for t in plan.tasks}
        fragments = record['fragments']
        if len(fragments) != len(expected) or {f['task_id'] for f in fragments} != set(expected):
            errors.append('v2: fragment coverage incomplete')
        for fragment in fragments:
            call = calls[fragment['call_id']]
            if (fragment['call_id'] not in contributors or call.get('role') != 'extraction'
                    or call.get('task_id') != fragment['task_id'] or fragment['covered'] != expected[fragment['task_id']]
                    or call.get('source_ranges') != fragment['covered']):
                errors.append('v2: fragment source/call binding mismatch')
        _visual_call_bindings(record)
        _visual_shared_budget(record)
        for call in calls.values():
            if call.get('role') not in ('extraction','react-tool-selection','react-repair','visual-evidence'):
                errors.append('v2: invalid call role')
        if sum(c.get('role','').startswith('react-') for c in calls.values()) > 2:
            errors.append('v2: ReAct request budget exceeded')
        if record.get('conflicts'):
            errors.append('v2: unresolved conflicts')
        if json_digest(record['final_data']) != record['final_json_sha256']:
            errors.append('v2: final merged artifact hash mismatch')
        if record.get('fusion',{}).get('result_sha256') != record['final_json_sha256']:
            errors.append('v2: rendered data differs from merged artifact')
    except (KeyError,TypeError,ValueError,AttributeError):
        errors.append('v2: missing or invalid extraction evidence')
    return errors


def _snapshot_and_plan(record):
    from .long_document import (
        ChunkPlan,
        ChunkTask,
        DocumentSnapshot,
        SourceRegion,
        SourceSlice,
    )

    value = dict(record["snapshot"])
    value["regions"] = [SourceRegion(**r) for r in value["regions"]]
    snapshot = DocumentSnapshot(**value)
    value = record["chunk_plan"]
    plan = ChunkPlan(value["snapshot_sha256"], [ChunkTask(
        t["task_id"], [SourceSlice(**s) for s in t["owned"]],
        [SourceSlice(**s) for s in t["context"]], t["prompt_tokens"])
        for t in value["tasks"]], value["input_budget"], value["output_budget"], value["instruction"])
    return snapshot, plan


def rebuild_chunked_result(value, *, source_snapshot=None):
    """Reparse actual response bytes, merge, then replay evidence-only repairs.

    This proves derivation consistency. Callers must separately bind response
    hashes to the actual backend ledger and source bytes to the input file.
    It accepts either an inference result or an execution-record envelope.
    """
    from dataclasses import asdict

    from .long_document import merge_fragments, parse_fragment, resolve_basic_conflicts, source_metadata
    from .run_state import json_digest

    snapshot, plan = _snapshot_and_plan(value)
    if snapshot.digest != value.get("snapshot_sha256") or plan.digest != value.get("plan_sha256"):
        raise ValueError("source snapshot/plan content identity mismatch")
    visual_ids = _visual_call_bindings(value)
    if "visual_evidence" in value:
        if source_snapshot is None:
            from .long_document import DocumentSnapshot

            native = DocumentSnapshot.from_docx(Path(value["source_evidence"]["prepared_docx_path"]))
            source_snapshot = _rebuild_visual_source(value, native)
        if json_digest(asdict(source_snapshot)) != json_digest(asdict(snapshot)):
            raise ValueError("visual source snapshot differs from actual source reconstruction")
    evidence = value.get("generation_evidence", {})
    if evidence.get("schema_version") != 1 or not isinstance(evidence.get("responses"), dict):
        raise ValueError("fresh generation response evidence required")
    responses = evidence["responses"]
    calls = {call["call_id"]: call for call in value["model_calls"]}
    if len(calls) != len(value["model_calls"]) or set(responses) - set(calls):
        raise ValueError("duplicate/unknown response call identity")
    accepted = {key for key, call in calls.items() if call.get("status") == "success"}
    if not accepted.issubset(responses):
        raise ValueError("actual successful response bytes missing")
    for key, raw in responses.items():
        if (not isinstance(raw, str) or len(raw.encode("utf-8")) > 8 * 1024 * 1024
                or _text_sha(raw) != calls[key].get("output_sha256")
                or calls[key].get("mock") or calls[key].get("simulated")):
            raise ValueError("response bytes differ from actual call receipt")
    sources, tasks = snapshot.sources(), {task.task_id: task for task in plan.tasks}
    if len(sources) != len(snapshot.regions) or len(tasks) != len(plan.tasks):
        raise ValueError("duplicate source/task identity")
    for task in plan.tasks:
        for span in [*task.owned, *task.context]:
            source = sources.get(span.source_id)
            if (source is None or type(span.start) is not int or type(span.end) is not int
                    or not (0 <= span.start < span.end <= len(source.text)
                            or (source.text == "" and span.start == span.end == 0))
                    or span.text != source.text[span.start:span.end]
                    or json_digest(span.metadata) != json_digest(source_metadata(source))):
                raise ValueError("task source text/metadata differs from snapshot")
    fragments = []
    for stored in value["fragments"]:
        call = calls[stored["call_id"]]
        task = tasks[stored["task_id"]]
        if (call["call_id"] not in accepted or call.get("role") != "extraction"
                or call.get("task_id") != task.task_id or call.get("source_ranges") != task.owned_keys):
            raise ValueError("fragment does not bind its successful extraction call")
        parsed = parse_fragment(responses[call["call_id"]], task, snapshot, call["call_id"])
        if json_digest(asdict(parsed)) != json_digest(stored):
            raise ValueError("fragment differs from reparsed actual response")
        fragments.append(parsed)
    merged = merge_fragments(snapshot, plan, fragments)
    repaired = value.get("repair_evidence") or any(
        call.get("role", "").startswith("react-") for call in calls.values())
    if merged["conflicts"] and repaired:
        _rebuild_repair(snapshot, merged, calls, responses, value)
    elif merged["conflicts"]:
        resolve_basic_conflicts(merged)
    elif repaired:
        raise ValueError("repair has no corresponding extraction conflict")
    used = set(merged["contributing_call_ids"])
    used.update(call["call_id"] for call in calls.values() if call.get("role") == "react-tool-selection")
    used.update(visual_ids)
    if accepted != used:
        raise ValueError("successful generation does not participate in reconstruction")
    for key in ("coverage", "provenance", "conflicts", "contributing_call_ids", "basic_observations", "basic_ambiguities"):
        if json_digest(value.get(key)) != json_digest(merged[key]):
            raise ValueError("reconstructed " + key + " differs from extraction evidence")
    if json_digest(value.get("repair_evidence", [])) != json_digest(merged.get("repair_evidence", [])):
        raise ValueError("repair evidence differs from actual response/source derivation")
    return merged


def _rebuild_repair(snapshot, merged, calls, responses, value):
    from .chunked_inference import _repair_support, read_evidence
    from .run_state import json_digest

    selections = [c for c in calls.values() if c.get("role") == "react-tool-selection"]
    repairs = [c for c in calls.values() if c.get("role") == "react-repair"]
    if len(selections) != 1 or len(repairs) != 1 or any(c.get("status") != "success" for c in selections + repairs):
        raise ValueError("two successful actual repair responses required")
    selection_call, repair_call = selections[0], repairs[0]
    selection = json.loads(responses[selection_call["call_id"]])
    if not isinstance(selection, dict) or set(selection) != {"tool", "arguments"} or selection["tool"] != "read_evidence":
        raise ValueError("illegal repair tool")
    evidence = read_evidence(snapshot, selection["arguments"])
    events = [{"tool": "read_evidence", "arguments": selection["arguments"],
               "status": "success", "call_id": selection_call["call_id"]}]
    if value.get("tool_events") != events or repair_call.get("source_ranges") != [
        f"{entry['source_id']}:0:{len(entry['text'])}" for entry in evidence]:
        raise ValueError("repair tool evidence differs from selected source ranges")
    response = json.loads(responses[repair_call["call_id"]])
    resolved = response.get("resolved") if isinstance(response, dict) else None
    fields = {conflict["field"]: conflict for conflict in merged["conflicts"]}
    if not isinstance(resolved, dict) or set(resolved) != set(fields):
        raise ValueError("repair did not resolve exactly the actual conflicts")
    allowed = {entry["source_id"] for entry in evidence}
    reconstructed = []
    for field, resolution in resolved.items():
        observations = fields[field]["observations"]
        scopes = {json_digest(item.get("scope", {})) for item in observations}
        if len(scopes) != 1 or not isinstance(resolution, dict) or resolution.get("source_id") not in allowed:
            raise ValueError("repair source/scope is unsupported")
        fixed = resolution.get("value")
        if fixed is None or isinstance(fixed, (dict, list)):
            raise ValueError("repair value must be scalar")
        scope = observations[0].get("scope", {})
        source = snapshot.sources()[resolution["source_id"]]
        if source.kind == "visual-observation":
            raise ValueError("photo observation cannot repair basic metadata conflicts")
        start, support = _repair_support(source.text, field, fixed, scope)
        text = str(fixed).strip()
        reconstructed.append({"field": field, "value": fixed, "scope": scope,
            "support_quote": support, "support_rule": "same-claim-field-value-scope-v1",
            "evidence": {"source_id": source.source_id,
                "source_key": f"{source.source_id}:{start}:{start + len(text)}",
                "start": start, "end": start + len(text), "text": text,
                "source_text_sha256": _text_sha(source.text), "document_sha256": snapshot.document_sha256,
                "snapshot_sha256": snapshot.digest, "location": source.location},
            "call_id": repair_call["call_id"], "supersedes": [json_digest(item) for item in observations]})
        merged["data"]["基本信息"][field] = fixed
    merged.update(repair_evidence=reconstructed, conflicts=[], tool_events=events)
    merged["contributing_call_ids"].append(repair_call["call_id"])


def _source_errors(record, source_path, *, require_strong,
                   expected_model_identity_sha256=None, reconstructed_snapshots=None):
    from dataclasses import asdict
    from tempfile import TemporaryDirectory

    from .long_document import DocumentSnapshot
    from .run_state import json_digest

    binding = record.get("source_evidence", {})
    if binding.get("schema_version") != 1 or binding.get("source_sha256") != record["document"]["sha256"]:
        return ["source: fresh original/prepared document binding required"]
    prepared = Path(binding["prepared_docx_path"])
    if (prepared.is_symlink() or not prepared.is_file() or prepared.suffix.lower() != ".docx"
            or _file_sha(prepared) != binding["prepared_docx_sha256"]):
        return ["source: prepared document bytes changed"]
    native_snapshot = DocumentSnapshot.from_docx(prepared)
    snapshot = native_snapshot
    if "visual_evidence" in record:
        snapshot = _rebuild_visual_source(record, native_snapshot,
            expected_model_identity_sha256=expected_model_identity_sha256, verify_capability=True)
    if reconstructed_snapshots is not None:
        reconstructed_snapshots.append(snapshot)
    if record.get("schema_version") in (2, 3):
        if snapshot.digest != record["snapshot_sha256"] or json_digest(asdict(snapshot)) != json_digest(record["snapshot"]):
            return ["source: snapshot differs from actual prepared document"]
        if require_strong and (not record.get("coverage", {}).get("document_complete")
                               or snapshot.unverified_content):
            return ["source: full document coverage is not verified"]
    if not require_strong:
        return []
    if source_path is None:
        return ["source: independent original source_path required"]
    source = Path(source_path)
    if source.is_symlink() or not source.is_file() or _file_sha(source) != binding["source_sha256"]:
        return ["source: original source bytes changed"]
    if source.suffix.lower() == ".docx":
        if binding["source_sha256"] != binding["prepared_docx_sha256"] or binding.get("conversion") is not None:
            return ["source: DOCX identity conversion mismatch"]
        if record.get("schema_version") == 1:
            from .text_extract import extract_docx_text

            if _text_sha(extract_docx_text(str(source))) != record.get("input_text_sha256"):
                return ["source: whole input text is not independently bound to DOCX"]
    elif source.suffix.lower() == ".doc":
        conversion = binding.get("conversion") or {}
        if (conversion.get("source_sha256") != binding["source_sha256"]
                or conversion.get("output_sha256") != binding["prepared_docx_sha256"]
                or conversion.get("conversion") != "docx"
                or not str(conversion.get("converter", "")).startswith("LibreOffice")):
            return ["source: genuine legacy conversion chain required"]
        # Fresh conversion is an offline admission check, never the online
        # recovery path. Container ZIP metadata may vary: compare all parsed
        # source structure rather than requiring equal DOC and DOCX hashes.
        with TemporaryDirectory(prefix=".source-audit-", dir=prepared.parent) as directory:
            replay = _reconvert_doc(source, Path(directory), conversion["converter"])
            left, right = asdict(native_snapshot), asdict(DocumentSnapshot.from_docx(replay))
            left.pop("document_sha256")
            right.pop("document_sha256")
            if json_digest(left) != json_digest(right):
                return ["source: fresh legacy conversion does not reproduce snapshot"]
            if "visual_evidence" in record and _visual_inventory_signature(prepared) != _visual_inventory_signature(replay):
                return ["source: fresh legacy conversion does not reproduce visual media/views"]
            if record.get("schema_version") == 1:
                text = _reconvert_doc_text(source, Path(directory), conversion["converter"])
                if _text_sha(text) != record.get("input_text_sha256"):
                    return ["source: whole input text is not independently bound to DOC conversion"]
    else:
        return ["source: unsupported input document type"]
    if _file_sha(source) != binding["source_sha256"] or _file_sha(prepared) != binding["prepared_docx_sha256"]:
        return ["source: document changed during independent verification"]
    return []


def _visual_inventory_signature(path):
    """Ignore ZIP container identity, retain each media/view/source occurrence."""
    from copy import deepcopy

    from .run_state import json_digest
    from .visual_inventory import inventory_docx

    occurrences = deepcopy(inventory_docx(path)["occurrences"])
    for occurrence in occurrences:
        occurrence.pop("id", None)
        occurrence.pop("document_sha256", None)
        for parent in occurrence.get("parent_objects", []):
            parent.pop("id", None)
    return json_digest(occurrences)


def _reconvert_doc(source, directory, expected_version):
    import subprocess

    version = subprocess.check_output(["soffice", "--version"], text=True, timeout=30).strip()
    if version != expected_version:
        raise ValueError("source: converter version changed")
    subprocess.run(["soffice", "-env:UserInstallation=" + (directory / "profile").resolve().as_uri(),
                    "--headless", "--convert-to", "docx", "--outdir", str(directory), str(source.resolve())],
                   check=True, timeout=300, capture_output=True)
    output = directory / (source.stem + ".docx")
    if not output.is_file() or not output.stat().st_size:
        raise ValueError("source: actual conversion did not produce DOCX")
    return output


def _reconvert_doc_text(source, directory, expected_version):
    import subprocess

    version = subprocess.check_output(["soffice", "--version"], text=True, timeout=30).strip()
    if version != expected_version:
        raise ValueError("source: converter version changed")
    subprocess.run(["soffice", "-env:UserInstallation=" + (directory / "text-profile").resolve().as_uri(),
                    "--headless", "--convert-to", "txt:Text (encoded):UTF8", "--outdir", str(directory), str(source.resolve())],
                   check=True, timeout=300, capture_output=True)
    return (directory / (source.stem + ".txt")).read_text(encoding="utf-8")


def _reconstruction_errors(record, output_path, *, source_path, require_strong,
                           expected_model_identity_sha256=None):
    from subprocess import SubprocessError
    from tempfile import TemporaryDirectory
    from xml.etree.ElementTree import ParseError
    from zipfile import BadZipFile, ZipFile

    from .docx_writer import write_extraction_docx
    from .inference_core import _parse_report
    from .model_preserving_fusion import preserve_model_report
    from .run_state import json_digest

    errors, snapshots = [], []
    try:
        errors.extend(_source_errors(record, source_path, require_strong=require_strong,
            expected_model_identity_sha256=expected_model_identity_sha256, reconstructed_snapshots=snapshots))
    except (OSError, KeyError, TypeError, ValueError, AttributeError, BadZipFile, SubprocessError, ParseError) as exc:
        errors.append("source: invalid original/conversion/snapshot evidence: " + str(exc))
    try:
        if record.get("schema_version") in (2, 3):
            model = rebuild_chunked_result(record, source_snapshot=snapshots[0] if snapshots else None)["data"]
            _parse_report(json.dumps(model, ensure_ascii=False), require_basic=False)
        elif record.get("schema_version") == 1:
            # Historical receipts still receive their ordinary V1 audit by
            # default. Fresh whole-mode admission additionally reconstructs
            # its actual final response and model-preserving projection.
            raw = record["raw_output"]
            call = record["model_calls"][-1]
            if call.get("status") != "success" or _text_sha(raw) != call["output_sha256"]:
                raise ValueError("actual final whole response is missing")
            model = _parse_report(raw)
        else:
            raise ValueError("unsupported reconstruction schema")
        if json_digest(model) != record.get("model_json_sha256"):
            raise ValueError("model projection differs from actual generation")
        if record.get("fusion", {}).get("mode") != "model_preserving":
            raise ValueError("formal reconstruction requires model-preserving fusion")
        final, fusion = preserve_model_report(
            model, label_conventions=bool(record.get("fusion", {}).get("label_conventions")))
        expected_fusion = {**fusion, "mode": "model_preserving", "result_sha256": json_digest(final)}
        if json_digest(record.get("final_data")) != json_digest(final) or record.get("fusion") != expected_fusion:
            raise ValueError("final projection differs from reconstructed generation/fusion")
        output = Path(output_path)
        with TemporaryDirectory(prefix=".render-audit-", dir=output.parent) as directory:
            expected = Path(directory) / "expected.docx"
            write_extraction_docx(final, str(expected))
            with ZipFile(expected) as reference, ZipFile(output) as actual:
                if reference.read("word/document.xml") != actual.read("word/document.xml"):
                    raise ValueError("delivered DOCX differs from reconstructed projection")
    except (OSError, KeyError, TypeError, ValueError, AttributeError, BadZipFile, RecursionError) as exc:
        errors.append("reconstruction: " + str(exc))
    return errors


def _v3_errors(record):
    from .rsi import StrategyBundle, validate_action
    from .run_state import json_digest
    errors = []
    try:
        evidence = record['rsi']
        value = evidence['bundle']
        bundle = StrategyBundle(value['source'], value['config'], value['parent_sha256'], value['abi'])
        if bundle.digest != evidence['bundle_sha256'] or evidence['mode'] not in ('frozen', 'experimental-online'):
            errors.append('v3: frozen strategy identity mismatch')
        initial = evidence['initial_tasks']
        issued = evidence['issued_calls']
        shared = _visual_shared_budget(record)
        call_limit = shared['total_call_limit'] - shared['prefix']['used_calls'] if shared else 4 * initial + 2
        recorded_calls = shared['text_issued_calls'] if shared else len(record['model_calls'])
        if (type(initial) is not int or initial < 1 or type(issued) is not int
                or not 0 < issued <= call_limit or recorded_calls > issued
                or not _seconds(evidence['spent_seconds']) or evidence['spent_seconds'] > 1200):
            errors.append('v3: report budget exceeded')
        if shared and (issued != shared['text_issued_calls']
                       or not math.isclose(evidence['spent_seconds'], shared['text_elapsed_seconds'], abs_tol=1.0)):
            errors.append('v3: text strategy budget differs from shared report execution')
        if not isinstance(evidence['decisions'], list) or not evidence['decisions']:
            errors.append('v3: missing strategy decisions')
        for decision in evidence['decisions']:
            if (decision['bundle_sha256'] != bundle.digest or decision['state_sha256'] != json_digest(decision['state'])
                    or type(decision['fallback']) is not bool or not _seconds(decision['seconds'], positive=False)):
                errors.append('v3: strategy decision identity mismatch')
            validate_action(decision['action'], decision['state'])
        if record.get('rsi_bundle_sha256') != bundle.digest:
            errors.append('v3: report-start strategy mismatch')
    except (KeyError, TypeError, ValueError, AttributeError):
        errors.append('v3: invalid strategy evidence')
    return errors
