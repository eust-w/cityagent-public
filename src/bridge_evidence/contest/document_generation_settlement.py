"""Observe later backend quiescence without rewriting an unknown generation.

This permits a new parent aggregation after stopped children, not a retry or
an assertion that an old request was cancelled. Missing DONE/usage stays
unknown in its original record. Only metadata GETs are made by this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Callable
from urllib import error, parse, request


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")
_SAMPLE = re.compile(r"^(vllm:num_requests_running|vllm:num_requests_waiting)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?\s*$")
_LABEL = re.compile(r'\s*([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"\s*')


class _Refused(RuntimeError):
    pass


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _save(path, value):
    name = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".quiescence-", delete=False) as stream:
            name = stream.name
            stream.write(_encoded(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        name = None
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def _ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _read(path, refs):
    if path.is_symlink() or not path.is_file():
        raise _Refused("TRUSTED_ARTIFACT_MISSING")
    raw = path.read_bytes()
    refs.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError):
        raise _Refused("TRUSTED_ARTIFACT_INVALID_JSON") from None


def _process_stopped(pid):
    # PID reuse is conservative: any current process/group with this numeric
    # identity blocks admission. No signal is sent and no process is killed.
    for probe in (lambda: os.kill(pid, 0), lambda: os.killpg(pid, 0)):
        try:
            probe()
            return False
        except ProcessLookupError:
            continue
        except (PermissionError, OSError):
            return False
    return True


def _trusted_children(dispatch, directory, model, manifest_sha256, deadline):
    refs, pids, unknowns = [], [], []
    stored = _read(directory / "dispatch-summary.json", refs)
    claim = _read(directory / "dispatch-claim.json", refs)
    if (_encoded(stored) != _encoded(dispatch) or not isinstance(claim, dict) or
            dispatch.get("schema") != "cityagent-native-subagent-dispatch-v1" or
            any(dispatch.get(key) != claim.get(key) for key in ("run_id", "model", "task_ids", "deadline_monotonic")) or
            dispatch.get("model") != model or dispatch.get("deadline_monotonic") != deadline):
        raise _Refused("DISPATCH_IDENTITY_MISMATCH")
    rows, task_ids = dispatch.get("results"), dispatch.get("task_ids")
    if (not isinstance(rows, list) or not rows or not isinstance(task_ids, list) or
            len(set(task_ids)) != len(task_ids) or len(rows) != len(task_ids) or
            [row.get("task_id") for row in rows] != task_ids):
        raise _Refused("DISPATCH_TASKS_MISMATCH")
    native_path = directory / "native-background-processes.json"
    if dispatch.get("native_registry_path") != str(native_path):
        raise _Refused("NATIVE_REGISTRY_IDENTITY_MISMATCH")
    _read(native_path, refs)
    for row in rows:
        task_id = row["task_id"]
        if (not isinstance(task_id, str) or not _ID.fullmatch(task_id) or row.get("association_error") is not None or
                any(row.get(key) is not True for key in ("process_settled", "runner_done", "native_terminal", "native_result_matches")) or
                row.get("native_status") not in {"completed", "failed", "stopped", "timeout"} or
                type(row.get("returncode")) is not int or type(row.get("pid")) is not int or row["pid"] <= 0):
            raise _Refused("CHILD_NOT_PROVEN_STOPPED_OR_ASSOCIATED")
        task_directory = directory / "tasks" / task_id
        result_path = task_directory / "result.json"
        if row.get("task_directory") != str(task_directory) or row.get("result_path") != str(result_path):
            raise _Refused("CHILD_ARTIFACT_PATH_MISMATCH")
        receipt = _read(task_directory / "process-receipt.json", refs)
        if (not isinstance(receipt, dict) or any(receipt.get(key) != row.get(key) for key in
                ("phase", "pid", "returncode", "process_settled", "generation_attempts_settled", "result_path", "sha256", "assignment_ref"))):
            raise _Refused("CHILD_PROCESS_RECEIPT_MISMATCH")
        assignment = _read(task_directory / "assignment.json", refs)
        if (not isinstance(assignment, dict) or assignment.get("id", assignment.get("task_id")) != task_id or
                ("id" in assignment and "task_id" in assignment and assignment["id"] != assignment["task_id"]) or
                row.get("assignment_ref") != refs[-1]):
            raise _Refused("CHILD_ASSIGNMENT_MISMATCH")
        result = _read(result_path, refs)
        if (refs[-1]["sha256"] != row.get("sha256") or not isinstance(result, dict) or
                result.get("task_id") != task_id or result.get("manifest_sha256") != manifest_sha256 or
                result.get("status") not in {"completed", "failed"} or
                type(result.get("generation_attempts_settled")) is not bool):
            raise _Refused("CHILD_RESULT_IDENTITY_MISMATCH")
        calls = result.get("unknown_generation_calls")
        if (not isinstance(calls, list) or any(not isinstance(call, str) or not call for call in calls) or
                (result["generation_attempts_settled"] and calls) or
                (not result["generation_attempts_settled"] and not calls)):
            raise _Refused("CHILD_UNKNOWN_CALLS_MISMATCH")
        unknowns.extend({"task_id": task_id, "call_id": call} for call in calls)
        if not _process_stopped(row["pid"]):
            raise _Refused("OWNED_CHILD_OR_PROCESS_GROUP_STILL_EXISTS")
        pids.append(row["pid"])
    return refs, pids, unknowns


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch(url, *, timeout, headers):
    started = time.monotonic()
    opener = request.build_opener(_NoRedirect())
    with opener.open(request.Request(url, headers=headers, method="GET"), timeout=timeout) as response:
        chunks, size = [], 0
        while True:
            if time.monotonic() - started >= timeout:
                raise TimeoutError("Metadata GET deadline exceeded")
            chunk = response.read1(65536)
            if not chunk:
                break
            size += len(chunk)
            if size > 16 * 1024 * 1024:
                raise _Refused("METADATA_RESPONSE_TOO_LARGE")
            chunks.append(chunk)
        return response.status, b"".join(chunks)


def _model_identity(raw, model, expected_parent, expected_max_model_len):
    try:
        payload = json.loads(raw)
        rows = payload["data"]
        matches = [row for row in rows if row.get("id") == model]
    except (KeyError, TypeError, ValueError, UnicodeError, AttributeError):
        raise _Refused("MODELS_RESPONSE_INVALID") from None
    if len(matches) != 1:
        raise _Refused("MODEL_NOT_UNIQUE")
    row = matches[0]
    if ("parent" not in row or (expected_parent is not None and row["parent"] != expected_parent) or
            (row["parent"] is not None and (not isinstance(row["parent"], str) or not row["parent"]))):
        raise _Refused("MODEL_PARENT_OR_CONTEXT_CHANGED")
    # A LoRA card may omit max_model_len, or display a value unrelated to the
    # serving base context. Match its actual parent card, as model preflight
    # does; never attribute the adapter's local field to backend capacity.
    base = row
    if row["parent"] is not None:
        parents = [card for card in rows if card.get("id") == row["parent"]]
        if len(parents) != 1:
            raise _Refused("MODEL_BASE_NOT_UNIQUE")
        base = parents[0]
    if type(base.get("max_model_len")) is not int or base["max_model_len"] != expected_max_model_len:
        raise _Refused("MODEL_PARENT_OR_CONTEXT_CHANGED")
    # ModelCard created/permission metadata can be generated per GET. Bind the
    # actual model/root/parent/context identity; retain full raw cards separately.
    selected_identity = {key: row[key] for key in ("id", "parent", "root", "max_model_len") if key in row}
    base_identity = {key: base[key] for key in ("id", "parent", "root", "max_model_len") if key in base}
    return {"model": model, "parent": row["parent"], "max_model_len": base["max_model_len"],
            **({"root": row["root"]} if "root" in row else {}),
            "selected_identity_sha256": hashlib.sha256(_encoded(selected_identity)).hexdigest(),
            "base_identity_sha256": hashlib.sha256(_encoded(base_identity)).hexdigest(),
            "selected": selected_identity, "base": base_identity}


def _labels(raw):
    result, position = {}, 0
    while position < len(raw):
        match = _LABEL.match(raw, position)
        if match is None or match[1] in result:
            raise _Refused("METRICS_LABELS_INVALID")
        try:
            result[match[1]] = json.loads('"' + match[2] + '"')
        except ValueError:
            raise _Refused("METRICS_LABELS_INVALID") from None
        position = match.end()
        if position == len(raw):
            break
        if raw[position] != ",":
            raise _Refused("METRICS_LABELS_INVALID")
        position += 1
        if position == len(raw):
            raise _Refused("METRICS_LABELS_INVALID")
    return result


def _gauges(raw):
    values = {name: [] for name in _GAUGES}
    try:
        text = raw.decode("utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not any(line.startswith(name) and (len(line) == len(name) or line[len(name)] in "{ \t") for name in _GAUGES):
                continue
            match = _SAMPLE.fullmatch(line)
            if match is None:
                raise _Refused("METRICS_SAMPLE_INVALID")
            value = float(match[3])
            if not math.isfinite(value) or value < 0:
                raise _Refused("METRICS_VALUE_INVALID")
            values[match[1]].append({"labels": _labels(match[2] or ""), "value": value})
    except (UnicodeError, ValueError):
        raise _Refused("METRICS_SAMPLE_INVALID") from None
    identities = []
    for samples in values.values():
        if not samples:
            raise _Refused("METRICS_REQUIRED_GAUGE_MISSING")
        keys = [_encoded(sample["labels"]) for sample in samples]
        if len(set(keys)) != len(keys):
            raise _Refused("METRICS_DUPLICATE_ENGINE_SAMPLE")
        identities.append(set(keys))
    if identities[0] != identities[1]:
        raise _Refused("METRICS_ENGINE_COVERAGE_MISMATCH")
    return values, all(sample["value"] == 0 for samples in values.values() for sample in samples)


def observe_generation_quiescence(dispatch: dict, *, run_directory: Path | str, base_url: str,
        model: str, manifest_sha256: str, deadline: float, expected_max_model_len: int = 262144,
        expected_parent=None, api_key: str | None = None, fetch: Callable | None = None,
        clock: Callable = time.monotonic, sleep: Callable = time.sleep,
        max_wait_seconds: float = 60, poll_interval: float = .25) -> dict:
    """Read metadata until two consecutive idle samples or a bounded refusal.

    Injected ``fetch(url, *, timeout, headers)`` returns ``(http_status, bytes)``.
    Its contract, like the default GET implementation, must honor the timeout.
    Busy samples reset the consecutive count. Invalid metadata fails closed.
    No generation, tokenize, reset, restart, cancellation or retry is issued.
    """
    if (type(deadline) not in (int, float) or not math.isfinite(deadline) or
            type(max_wait_seconds) not in (int, float) or not 0 < max_wait_seconds <= 60 or
            type(poll_interval) not in (int, float) or not 0 < poll_interval <= 1 or
            type(expected_max_model_len) is not int or expected_max_model_len <= 0 or
            not isinstance(model, str) or not model or not isinstance(manifest_sha256, str) or
            not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256)):
        raise ValueError("Invalid bounded quiescence configuration")
    parsed = parse.urlsplit(base_url)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or
            parsed.query or parsed.fragment or parsed.path.rstrip("/") not in {"", "/v1"}):
        raise ValueError("Expected the pinned backend origin or /v1 URL")
    origin = parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    directory = Path(run_directory).absolute()
    output = directory / "generation-quiescence"
    output.mkdir(exist_ok=False)
    started = clock()
    stop_at = min(deadline, started + max_wait_seconds)
    receipt = {"schema": "cityagent-generation-quiescence-v1", "outcome": "unknown",
        "may_continue_aggregation": False, "prior_requests_reclassified": False,
        "cancellation_confirmed": False, "model_retry_permitted": False,
        "historical_done_and_usage": "unchanged; missing values remain unknown",
        "base_url": origin, "model": model, "shared_deadline_monotonic": deadline,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "started_at_monotonic": started,
        "observation_deadline_monotonic": stop_at, "observations": [], "unknown_calls": [],
        "unknown_call_ids": [], "input_refs": [], "error_code": None}
    receipt_path = output / "receipt.json"
    getter = fetch or _fetch
    headers = {"Accept": "application/json, text/plain"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    try:
        refs, pids, unknowns = _trusted_children(dispatch, directory, model, manifest_sha256, deadline)
        receipt.update(input_refs=refs, unknown_calls=unknowns,
                       unknown_call_ids=list(dict.fromkeys(item["call_id"] for item in unknowns)))
        consecutive, identity = 0, None
        while clock() < stop_at:
            sample = {"index": len(receipt["observations"]), "at_monotonic": clock(),
                      "at_utc": datetime.now(timezone.utc).isoformat(), "requests": []}
            receipt["observations"].append(sample)
            bodies = []
            for kind, endpoint in (("models", "/v1/models"), ("metrics", "/metrics")):
                if clock() >= stop_at:
                    raise _Refused("OBSERVATION_DEADLINE_EXCEEDED")
                item = {"kind": kind, "started_at_monotonic": clock(), "http_status": None}
                sample["requests"].append(item)
                try:
                    status, body = getter(origin + endpoint, timeout=min(5.0, stop_at - clock()), headers=dict(headers))
                except error.HTTPError as exc:
                    item["http_status"] = exc.code
                    raise _Refused("METADATA_HTTP_ERROR") from None
                except (OSError, TimeoutError):
                    raise _Refused("METADATA_FETCH_ERROR") from None
                item.update(http_status=status, completed_at_monotonic=clock())
                if type(status) is not int or status != 200 or not isinstance(body, bytes):
                    raise _Refused("METADATA_HTTP_OR_BODY_INVALID")
                raw_path = output / f"sample-{sample['index']:03d}-{kind}.raw"
                raw_path.write_bytes(body)
                item["body_ref"] = _ref(raw_path)
                bodies.append(body)
                if clock() >= stop_at:
                    raise _Refused("OBSERVATION_DEADLINE_EXCEEDED")
            current_identity = _model_identity(bodies[0], model, expected_parent, expected_max_model_len)
            if identity is not None and current_identity != identity:
                raise _Refused("MODEL_IDENTITY_CHANGED_DURING_OBSERVATION")
            identity = current_identity
            values, idle = _gauges(bodies[1])
            consecutive = consecutive + 1 if idle else 0
            sample.update(model_identity=identity, gauges=values, all_engines_idle=idle, consecutive_idle_samples=consecutive)
            _save(receipt_path, receipt)
            if consecutive >= 2:
                if any(not _process_stopped(pid) for pid in pids):
                    raise _Refused("OWNED_CHILD_REAPPEARED")
                if any(_ref(Path(ref["path"])) != ref for ref in refs):
                    raise _Refused("TRUSTED_INPUT_CHANGED_DURING_OBSERVATION")
                if clock() >= stop_at:
                    raise _Refused("OBSERVATION_DEADLINE_EXCEEDED")
                receipt.update(outcome="quiescent_observed", may_continue_aggregation=True,
                               model_identity=identity, consecutive_idle_samples=consecutive)
                break
            sleep(min(poll_interval, max(0, stop_at - clock())))
        if receipt["outcome"] != "quiescent_observed":
            raise _Refused("OBSERVATION_DEADLINE_EXCEEDED")
    except _Refused as exc:
        receipt["error_code"] = str(exc)
    except Exception as exc:
        receipt.update(error_code="QUIESCENCE_CHECK_ERROR", error_type=type(exc).__name__)
    receipt.update(finished_at_monotonic=clock(), finished_at_utc=datetime.now(timezone.utc).isoformat())
    _save(receipt_path, receipt)
    return {**receipt, "receipt_ref": _ref(receipt_path)}
