"""External RSI gates. A candidate cannot redefine its baseline or metrics."""

import math


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def replacement_errors(baseline, receipts):
    errors = []
    if len(receipts) != 2:
        return ["two independent complete receipts required"]
    if (
        not baseline.get("comparison_identity")
        or not baseline.get("audit_passed")
        or baseline.get("completed") != baseline.get("denominator")
        or not isinstance(baseline.get("denominator"), int)
        or baseline["denominator"] < 1
    ):
        return ["invalid complete anchor"]
    ids = [r.get("run_id") for r in receipts]
    if not all(ids) or len(set(ids)) != 2 or baseline.get("run_id") in ids:
        errors.append("receipts must be independent runs")
    bundles = {r.get("bundle_sha256") for r in receipts}
    if len(bundles) != 1 or None in bundles:
        errors.append("frozen strategy identity mismatch")
    for receipt in receipts:
        if (
            receipt.get("protocol") != "frozen-rsi-v3"
            or receipt.get("comparison_identity") != baseline["comparison_identity"]
            or receipt.get("denominator") != baseline["denominator"]
            or receipt.get("completed") != baseline["denominator"]
            or receipt.get("cache_reused") is not False
            or receipt.get("audit_passed") is not True
        ):
            errors.append("incomplete/mixed/stale/cached receipt")
        metrics, reference = receipt.get("f1", []), baseline.get("f1", [])
        if (
            len(metrics) != 3
            or len(reference) != 3
            or any(not _number(x) or x > 1 for x in [*metrics, *reference])
            or any(x < y for x, y in zip(metrics, reference))
        ):
            errors.append("three-table quality regression")
        for key in ("failures", "missing_sources", "conflicts"):
            values = receipt.get(key), baseline.get(key)
            if (
                any(type(v) is not int or v < 0 for v in values)
                or values[0] > values[1]
            ):
                errors.append(key + " regression or missing evidence")
        wall, anchor = receipt.get("wall_seconds"), baseline.get("wall_seconds")
        if (
            not _number(wall)
            or not _number(anchor)
            or anchor <= 0
            or wall <= 0
            or wall > 0.8 * anchor
        ):
            errors.append("less than 20 percent whole-batch speedup")
        for key in ("p50", "p95"):
            value, reference = (
                receipt.get("latency", {}).get(key),
                baseline.get("latency", {}).get(key),
            )
            if not _number(value) or not _number(reference) or value > reference:
                errors.append("normal latency " + key + " regression/missing")
        cases = receipt.get("recovery", {}).get("cases", {})
        anchors = baseline.get("recovery", {}).get("cases", {})
        if not anchors or set(cases) != set(anchors):
            errors.append("matching fault recovery suite required")
        else:
            for case in anchors:
                for key in ("start_seconds", "finish_seconds"):
                    value, reference = cases[case].get(key), anchors[case].get(key)
                    if (
                        not _number(value)
                        or not _number(reference)
                        or value > reference
                    ):
                        errors.append("recovery regression: " + case + "/" + key)
    return errors


def backend_request_key(receipt):
    """Scope vendor counters by actual local engine instance; API IDs stay global."""
    import json

    request_id = receipt.get("backend_request_id")
    instance = receipt.get("backend_instance_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("ledger lacks actual backend request identity")
    if instance is not None and (not isinstance(instance, str) or not instance.strip()):
        raise ValueError("invalid actual backend instance identity")
    return json.dumps([instance, request_id], ensure_ascii=False, separators=(",", ":"))


def audit_call_ledger(path, records):
    """Bind every recorded generation to unique durable backend evidence."""
    import json
    import sqlite3
    from pathlib import Path

    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("actual timing ledger required")
    try:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
            rows = db.execute(
                "SELECT id,role,seconds,status,receipt FROM calls"
            ).fetchall()
            spans = dict(
                db.execute(
                    "SELECT kind,SUM(seconds) FROM spans GROUP BY kind"
                ).fetchall()
            )
    except sqlite3.Error as error:
        raise ValueError("invalid actual timing ledger") from error
    ledger = {row[0]: row[1:] for row in rows}
    calls = [
        (record, call) for record in records for call in record.get("model_calls", [])
    ]
    ids = [call["call_id"] for _, call in calls]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("missing or duplicate generation call IDs")
    if set(ids) != set(ledger):
        raise ValueError("model call set differs from actual timing ledger")
    backend_ids, backend_keys = [], []
    for record, call in calls:
        role, seconds, status, encoded = ledger[call["call_id"]]
        if (
            role != call.get("role", "extraction")
            or not _number(seconds)
            or seconds <= 0
            or status not in ("success", "failed", "interrupted")
            or (call.get("status") == "success" and status != "success")
        ):
            raise ValueError("ledger role/status/duration mismatch")
        measured = call.get("seconds")
        if not _number(measured) or measured + 1e-6 < seconds:
            raise ValueError("ledger generation time understated")
        # V1 wraps backend bookkeeping in its timer; V2/V3 retain exact backend latency.
        if record.get("schema_version") in (2, 3) and not math.isclose(
            measured, seconds, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("ledger generation duration changed")
        try:
            receipt = json.loads(encoded or "{}")
        except (TypeError, ValueError) as error:
            raise ValueError("invalid ledger backend receipt") from error
        if receipt.get("mock") or receipt.get("simulated"):
            raise ValueError("mock ledger receipt cannot qualify")
        if not receipt.get("prompt_sha256") or receipt["prompt_sha256"] != call.get(
            "prompt_sha256"
        ):
            raise ValueError("ledger receipt mismatch: prompt_sha256")
        if status == "success" or call.get("output_sha256") is not None:
            for field in (
                "prompt_sha256",
                "output_sha256",
                "output_tokens",
                "finish_reason",
            ):
                if receipt.get(field) is None or receipt[field] != call.get(field):
                    raise ValueError("ledger receipt mismatch: " + field)
            backend_keys.append(backend_request_key(receipt))
            backend_ids.append(receipt["backend_request_id"])
    if len(backend_keys) != len(set(backend_keys)):
        raise ValueError("duplicate backend receipt identity in ledger")
    return {
        "call_ids": ids,
        "backend_request_ids": backend_ids,
        "backend_request_keys": backend_keys,
        "generation_seconds": sum(row[2] for row in rows),
        "failures": sum(row[3] in ("failed", "interrupted") for row in rows),
        "spans": spans,
    }


def audit_strategy_execution(records, bundle_digest, runtime_digest):
    """Require exercised isolated decisions bound to the actual frozen runtime."""
    from .rsi import validate_action
    from .rsi_sandbox import policy_state
    from .run_state import json_digest

    if not isinstance(runtime_digest, str) or len(runtime_digest) != 64:
        raise ValueError("frozen strategy isolation identity missing")
    executed = total = 0
    for record in records:
        for decision in record.get("rsi", {}).get("decisions", []):
            total += 1
            if decision.get("fallback") is True:
                continue  # Fixed base rules remain valid; they are not evolved decisions.
            if decision.get("fallback") is not False:
                raise ValueError("invalid strategy execution/fallback status")
            state = policy_state(decision["state"])
            if (
                state != decision["state"]
                or json_digest(state) != decision.get("state_sha256")
                or decision.get("bundle_sha256") != bundle_digest
            ):
                raise ValueError("strategy state or bundle identity changed")
            validate_action(decision["action"], state)
            proof = decision.get("kernel_proof", {})
            if (
                decision.get("sandbox_runtime_digest") != runtime_digest
                or decision.get("fallback_reason") is not None
                or type(proof.get("pid")) is not int
                or proof["pid"] <= 0
                or proof.get("seccomp") != 2
                or proof.get("no_new_privs") != 1
                or proof.get("capabilities") != 0
                or proof.get("namespaces") != 6
                or proof.get("readonly_root") is not True
            ):
                raise ValueError("strategy lacks matching actual isolation evidence")
            executed += 1
    if not executed:
        raise ValueError(
            "frozen strategy never executed; fallback cannot qualify evolution"
        )
    return {
        "executed_decisions": executed,
        "fallback_decisions": total - executed,
        "execution_rate": executed / total,
        "sandbox_runtime_digest": runtime_digest,
    }


def audit_summary_timing(result, ledger, records):
    """Use durable batch spans and per-report timing, not mutable summary metrics."""
    timing = result.get("timing", {})
    if (
        timing.get("calls") != len(ledger["call_ids"])
        or timing.get("failures") != ledger["failures"]
        or not _number(timing.get("generation_seconds"))
        or not math.isclose(
            timing["generation_seconds"],
            ledger["generation_seconds"],
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("evaluation timing differs from actual ledger")
    wall = ledger["spans"].get("batch_wall")
    summary_wall = result.get("wall_seconds")
    # The span closes immediately before assembling evaluation.json. One second
    # permits JSON bookkeeping, never a fabricated baseline slowdown.
    if (
        not _number(wall)
        or wall <= 0
        or not _number(summary_wall)
        or not 0 <= summary_wall - wall <= 1
    ):
        raise ValueError("whole-batch wall time differs from its durable span")
    values = [r.get("report_wall_seconds") for r in records]
    if not values or any(not _number(v) or v <= 0 for v in values):
        raise ValueError("actual per-report latency evidence required")
    values.sort()
    latencies = {}
    for name, quantile in (("p50", 0.5), ("p95", 0.95)):
        position = (len(values) - 1) * quantile
        low = int(position)
        high = min(low + 1, len(values) - 1)
        expected = values[low] + (values[high] - values[low]) * (position - low)
        summary = result.get("latency", {}).get(name)
        if not _number(summary) or not math.isclose(
            summary, expected, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError("evaluation latency differs from actual report receipts")
        latencies[name] = expected
    return {"wall_seconds": wall, "latency": latencies}


PROTECTED_RUNTIME_FILES = (
    "run.py",
    "README.md",
    "requirements-contest.txt",
    "THIRD_PARTY_NOTICES.md",
    "scripts/merge_lora_v3.py",
    "src/bridge_evidence/__init__.py",
    "src/bridge_evidence/domain_rules.py",
    "src/bridge_evidence/evaluation/__init__.py",
    "src/bridge_evidence/evaluation/generation.py",
)
RECOVERY_KINDS = ("length", "format", "timeout", "oom", "interrupt")
RECOVERY_TEXT = (
    "RSI恢复验证桥，2018年定期检测。\n第一跨主梁出现一条裂缝，建议封闭处理。"
)


def audit_protected_source(declared, source_root):
    """Require the complete canonical runtime, including newly added modules."""
    from pathlib import Path

    from .run_state import file_digest, json_digest

    root = Path(source_root)
    paths = [root / name for name in PROTECTED_RUNTIME_FILES]
    paths += sorted((root / "src/bridge_evidence/contest").rglob("*.py"))
    if any(
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        for path in paths
    ):
        raise ValueError("canonical protected source missing or symlinked")
    expected = {p.relative_to(root).as_posix(): file_digest(p) for p in paths}
    if declared != expected:
        raise ValueError("canonical protected source set or content changed")
    return json_digest(expected)


def comparison_identity(identity, *, source_root):
    """Rebuild immutable comparison conditions; candidate configuration is excluded."""
    from pathlib import Path

    from bridge_evidence.evaluation.generation import METRIC_VERSION

    from .backends import normalize_api_url
    from .run_state import file_digest, json_digest

    options = identity["options"]
    endpoint = normalize_api_url(options.get("url"))
    evaluator = Path(source_root) / "scripts/evaluate_multitask.py"
    if (
        not evaluator.is_file()
        or evaluator.is_symlink()
        or options.get("evaluator_sha256") != file_digest(evaluator)
    ):
        raise ValueError("matching evaluator source evidence required")
    samples = options.get("manifest", {}).get("samples", [])
    if (
        options.get("protocol") != "complete-multitask-report-v2"
        or options.get("metric_version") != METRIC_VERSION
        or not samples
        or len({s["id"] for s in samples}) != len(samples)
        or options["manifest"].get("metric_denominator", len(samples)) != len(samples)
        or any(not s.get("report_sha256") or not s.get("gold_sha256") for s in samples)
        or not options.get("model", {}).get("files")
        or not isinstance(options.get("hardware"), str)
        or not options["hardware"].strip()
        or type(options.get("warmup")) is not int
        or options["warmup"] < 0
        or options.get("backend") not in ("vllm-api", "sglang-api")
        or options.get("engine") not in ("direct", "tongagents")
    ):
        raise ValueError("incomplete immutable comparison conditions")
    return json_digest(
        {
            "protocol": "rsi-comparison-v2",
            "protected_sha256": json_digest(identity["code_files"]),
            "evaluator_sha256": options["evaluator_sha256"],
            "model": options["model"],
            "manifest": options["manifest"],
            "metric_version": METRIC_VERSION,
            "hardware": options["hardware"],
            "warmup": options["warmup"],
            "backend": options["backend"],
            "url": endpoint,
            "engine": options["engine"],
            "sandbox_digest": options.get("sandbox_digest"),
            "sdk_acceptance_sha256": options.get("sdk_acceptance_sha256"),
            "visual_evidence": options.get("visual_evidence", {"mode": "off"}),
        }
    )


def raw_request_call(request, *, role="extraction"):
    """Reconstruct a call from its actual input IDs and unmodified backend response."""
    import hashlib

    from .run_state import json_digest

    prompt, response = request.get("prompt_ids"), request.get("response", {})
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(type(v) is not int or v < 0 for v in prompt)
        or not isinstance(response.get("text"), str)
        or type(response.get("output_tokens")) is not int
        or response["output_tokens"] < 0
        or not isinstance(response.get("finish_reason"), str)
        or not isinstance(response.get("call_id"), str)
        or not response["call_id"]
    ):
        raise ValueError("actual raw generation evidence required")
    backend_request_key(response)
    return {
        "call_id": response["call_id"],
        "role": role,
        "status": "success" if response["finish_reason"] == "stop" else "failed",
        "seconds": response.get("latency_seconds"),
        "prompt_sha256": json_digest(prompt),
        "output_sha256": hashlib.sha256(response["text"].encode()).hexdigest(),
        "output_tokens": response["output_tokens"],
        "finish_reason": response["finish_reason"],
    }


def audit_warmup(directory, count):
    import json
    from pathlib import Path

    from .run_state import file_digest

    directory = Path(directory)
    path = directory / "warmup.json"
    value = json.loads(path.read_text())
    if value.get("protocol") != "actual-warmup-v1" or value.get("requests") != count:
        raise ValueError("actual matching warmup evidence required")
    requests = value.get("generations", [])
    if len(requests) != count:
        raise ValueError("warmup model request count mismatch")
    ledger_path = directory / "warmup.sqlite"
    if value.get("timing_sha256") != file_digest(ledger_path):
        raise ValueError("warmup ledger changed")
    if not count:
        import sqlite3

        with sqlite3.connect(f"file:{ledger_path.resolve()}?mode=ro", uri=True) as db:
            if db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]:
                raise ValueError("unexpected unreported warmup calls")
        return {
            "call_ids": [],
            "backend_request_keys": [],
            "warmup_sha256": file_digest(path),
        }
    if any(
        r["response"].get("finish_reason") not in ("stop", "length") for r in requests
    ):
        raise ValueError("warmup generation failed")
    audited = audit_call_ledger(
        ledger_path,
        [
            {
                "schema_version": 3,
                "model_calls": [raw_request_call(r, role="warmup") for r in requests],
            }
        ],
    )
    return {**audited, "warmup_sha256": file_digest(path)}


def audit_recovery_evidence(path, *, identity, bundle_digest, source_root):
    """Rebuild fault latencies from bound events, raw requests, cases and SQLite."""
    import json
    import sqlite3
    from pathlib import Path

    from .backends import normalize_api_url
    from .long_document import DocumentSnapshot
    from .run_state import file_digest, json_digest

    path = Path(path)
    recovery = json.loads(path.read_text())
    options = identity["options"]
    mode = "frozen" if bundle_digest else "off"
    if (
        recovery.get("protocol") != "rsi-recovery-v2"
        or recovery.get("mode") != mode
        or recovery.get("bundle_sha256") != bundle_digest
        or recovery.get("model") != options.get("model")
        or recovery.get("config") != options.get("config")
        or recovery.get("sandbox_digest") != options.get("sandbox_digest")
        or recovery.get("protected_sha256")
        != json_digest(identity.get("code_files", {}))
        or set(recovery.get("cases", {})) != set(RECOVERY_KINDS)
    ):
        raise ValueError("matching current candidate recovery evidence required")
    if normalize_api_url(recovery.get("url")) != normalize_api_url(options.get("url")):
        raise ValueError("matching recovery service endpoint required")
    driver = Path(source_root) / "scripts/benchmark_rsi_recovery.py"
    if not driver.is_file() or recovery.get("driver_sha256") != file_digest(driver):
        raise ValueError("recovery driver changed")
    artifacts = recovery.get("artifacts", {})
    expected = {
        "timing.sqlite",
        *(
            kind + suffix
            for kind in RECOVERY_KINDS
            for suffix in (".json", ".events.json")
        ),
    }
    if set(artifacts) != expected:
        raise ValueError("complete recovery artifact set required")
    for name, digest in artifacts.items():
        target = path.parent / name
        if not target.is_file() or target.is_symlink() or file_digest(target) != digest:
            raise ValueError("recovery artifact changed: " + name)
    ledger_path = path.parent / "timing.sqlite"
    with sqlite3.connect(f"file:{ledger_path.resolve()}?mode=ro", uri=True) as db:
        starts = dict(db.execute("SELECT id,start FROM calls").fetchall())
        spans = {
            name: (kind, seconds)
            for name, kind, seconds in db.execute(
                "SELECT id,kind,seconds FROM spans"
            ).fetchall()
        }
    all_calls, cases = [], {}
    for kind in RECOVERY_KINDS:
        value = json.loads((path.parent / (kind + ".json")).read_text())
        journal = json.loads((path.parent / (kind + ".events.json")).read_text())
        events, requests = journal.get("events", []), journal.get("requests", [])
        if journal.get("case") != kind or not events or not requests:
            raise ValueError("actual recovery case events required")
        clocks = [(e.get("monotonic_ns"), e.get("wall_ns")) for e in events]
        if (
            any(
                type(a) is not int or type(b) is not int or min(a, b) <= 0
                for a, b in clocks
            )
            or any(a[0] > b[0] or a[1] > b[1] for a, b in zip(clocks, clocks[1:]))
            or any(
                abs((a - clocks[0][0]) - (b - clocks[0][1])) > 100_000_000
                for a, b in clocks
            )
        ):
            raise ValueError("recovery monotonic/wall clock evidence invalid")
        marks = {}
        for name in ("started", "fault", "resume", "complete"):
            matches = [e for e in events if e.get("event") == name]
            if len(matches) != 1:
                raise ValueError("unique recovery clock boundaries required")
            marks[name] = matches[0]["monotonic_ns"]
        if (
            not marks["started"]
            <= marks["fault"]
            <= marks["resume"]
            < marks["complete"]
        ):
            raise ValueError("recovery event order invalid")
        durations = {
            "start_seconds": (marks["resume"] - marks["fault"]) / 1e9,
            "finish_seconds": (marks["complete"] - marks["fault"]) / 1e9,
            "total_seconds": (marks["complete"] - marks["started"]) / 1e9,
        }
        for name, measured in durations.items():
            declared = recovery["cases"][kind].get(name)
            if (
                not _number(declared)
                or not math.isclose(declared, measured, abs_tol=1e-9)
                or spans.get(kind + ":" + name) != ("recovery_" + name, measured)
            ):
                raise ValueError("recovery duration differs from durable events/spans")
        case_calls = []
        for index, request in enumerate(requests):
            call = raw_request_call(request)
            case_calls.append(call)
            boundaries = [
                [
                    e
                    for e in events
                    if e.get("event") == name and e.get("request_index") == index
                ]
                for name in ("call_start", "call_finish")
            ]
            if any(len(items) != 1 for items in boundaries):
                raise ValueError("recovery call boundaries missing")
            begin, end = (items[0] for items in boundaries)
            ledger_start = starts.get(call["call_id"])
            if (
                ledger_start is None
                or not _number(call["seconds"])
                or end["monotonic_ns"] - begin["monotonic_ns"] < call["seconds"] * 1e9
                or not begin["wall_ns"] / 1e9 - 0.001
                <= ledger_start
                <= end["wall_ns"] / 1e9
                or not marks["started"]
                <= begin["monotonic_ns"]
                < end["monotonic_ns"]
                <= marks["complete"]
            ):
                raise ValueError("recovery call clock differs from actual ledger")
            if index == 0 and kind in ("length", "format", "interrupt"):
                if end["monotonic_ns"] > marks["fault"]:
                    raise ValueError("recovery starts before actual fault observation")
            elif begin["monotonic_ns"] < marks["resume"]:
                raise ValueError("recovery retry predates its resume boundary")
        if len(
            [e for e in events if e.get("event") in ("call_start", "call_finish")]
        ) != 2 * len(requests):
            raise ValueError("unbound recovery call event")
        before_retry = [
            e
            for e in events
            if e.get("event") == "call_start" and e["monotonic_ns"] < marks["resume"]
        ]
        if len(before_retry) != (1 if kind in ("format", "length", "interrupt") else 0):
            raise ValueError(
                "recovery injection/request order differs from fixed suite"
            )
        if (
            next(e for e in events if e.get("event") == "fault").get("injection")
            != kind
        ):
            raise ValueError("recovery fault kind changed")
        retry_calls = [
            e
            for e in events
            if e.get("event") == "call_start" and e["monotonic_ns"] >= marks["resume"]
        ]
        if not retry_calls or (
            kind in ("format", "length", "interrupt") and len(requests) < 2
        ):
            raise ValueError("actual recovery retry missing")
        # Rebuild all semantic fragments from raw successful generations. Controlled
        # injection modifies the failed response only, not any merged fragment.
        from .execution_audit import rebuild_chunked_result

        rebuilt = rebuild_chunked_result(value)
        if (
            value.get("snapshot_sha256")
            != DocumentSnapshot.from_text(RECOVERY_TEXT).digest
            or value.get("data") != rebuilt["data"]
            or not rebuilt["coverage"]["complete"]
            or rebuilt["conflicts"]
            or recovery["cases"][kind].get("final_sha256")
            != json_digest(rebuilt["data"])
        ):
            raise ValueError("recovery case final evidence mismatch")
        raw_by_id = {c["call_id"]: c for c in case_calls}
        recorded_calls = value.get("model_calls", [])
        if not recorded_calls or recorded_calls[0].get("status") != "failed":
            raise ValueError("recovery injection failure evidence missing")
        initial = recorded_calls[0]
        expected_error = {
            "length": "incomplete_generation:length",
            "format": "JSONDecodeError:",
            "timeout": "TimeoutError:controlled timeout injection",
            "oom": "OSError:controlled out of memory injection",
            "interrupt": "interrupted before durable model response",
        }[kind]
        if not any(expected_error in str(error) for error in initial.get("errors", [])):
            raise ValueError("recovery failure differs from fixed controlled injection")
        record_ids = {call["call_id"] for call in recorded_calls}
        permitted_synthetic = (
            {initial["call_id"]} if kind in ("timeout", "oom") else set()
        )
        if record_ids != set(raw_by_id) | permitted_synthetic:
            raise ValueError(
                "recovery case calls differ from actual ledger and injected fault"
            )
        for call in recorded_calls:
            if call.get("status") == "success":
                actual = raw_by_id.get(call["call_id"], {})
                if any(
                    call.get(field) != actual.get(field)
                    for field in (
                        "prompt_sha256",
                        "output_sha256",
                        "output_tokens",
                        "finish_reason",
                    )
                ):
                    raise ValueError(
                        "recovery successful fragment lacks actual request"
                    )
        if bundle_digest:
            audit_strategy_execution(
                [value], bundle_digest, options.get("sandbox_digest")
            )
        all_calls.extend(case_calls)
        cases[kind] = {**recovery["cases"][kind], **durations}
    ledger = audit_call_ledger(
        ledger_path, [{"schema_version": 3, "model_calls": all_calls}]
    )
    if recovery.get("actual_model_call_ids") != ledger["call_ids"]:
        raise ValueError("recovery actual call set changed")
    return {
        **recovery,
        "cases": cases,
        "path": str(path.resolve()),
        "generation_ledger": ledger,
    }


def audit_independent_receipts(receipts):
    ledgers = []
    for receipt in receipts:
        ledgers.extend(
            [
                receipt.get("generation_ledger", {}),
                receipt.get("warmup_ledger", {}),
                receipt.get("recovery", {}).get("generation_ledger", {}),
            ]
        )
    for key in ("call_ids", "backend_request_keys"):
        values = [v for ledger in ledgers for v in ledger.get(key, [])]
        if len(values) != len(set(values)):
            raise ValueError("independent runs reused actual model receipts")


def _audited_evaluation(directory, *, source_root):
    """Rebuild complete evaluation evidence; recovery admission is a separate gate."""
    import json
    from pathlib import Path

    from .execution_audit import validate_execution_record
    from .run_state import document_identity, file_digest, json_digest, record_name

    directory, source_root = Path(directory), Path(source_root)
    result = json.loads((directory / "evaluation.json").read_text())
    manifest = json.loads((directory / "run_manifest.json").read_text())
    identity = manifest["identity"]
    if json_digest(identity) != manifest["run_fingerprint"]:
        raise ValueError("run identity mismatch")
    protected = audit_protected_source(identity["code_files"], source_root)
    conditions = comparison_identity(identity, source_root=source_root)
    if (
        result.get("comparison_identity") != conditions
        or result.get("conditions") != conditions
    ):
        raise ValueError(
            "evaluation comparison identity differs from immutable evidence"
        )
    if result.get("protected_sha256") != protected:
        raise ValueError("protected execution source changed")
    if result.get("cache_reused") is not False or not result.get("run_id"):
        raise ValueError("fresh uniquely identified complete evaluation required")
    samples = identity["options"]["manifest"]["samples"]
    if result.get("denominator") != len(samples) or result.get("completed") != len(
        samples
    ):
        raise ValueError("incomplete evaluation denominator")
    if set(r["id"] for r in result["reports"]) != {s["id"] for s in samples}:
        raise ValueError("report receipt set mismatch")
    profile = (
        "contest-sdk" if identity["options"]["engine"] == "tongagents" else "legacy"
    )
    missing = conflicts = 0
    records = []
    bundle = result.get("bundle_sha256")
    if bool(bundle) != (identity["options"].get("rsi_mode") == "frozen"):
        raise ValueError("frozen receipt strategy mode mismatch")
    if bundle:
        from .rsi import StrategyBundle

        value = identity["options"].get("rsi_bundle", {})
        bound = StrategyBundle(
            **{key: value[key] for key in ("source", "config", "parent_sha256", "abi")}
        )
        if (
            value != bound.as_dict()
            or bound.digest != bundle
            or bound.config != identity["options"].get("config")
        ):
            raise ValueError("frozen strategy differs from bound runtime configuration")
    for case in samples:
        report = Path(case["report"])
        doc = document_identity(report, report.parent)
        record = json.loads((directory / "records" / record_name(doc)).read_text())
        records.append(record)
        if case.get("report_sha256") != doc["sha256"]:
            raise ValueError("frozen report source changed")
        if record["status"] != "success":
            raise ValueError(
                "failed report cannot be a submission-ready complete receipt"
            )
        errors = validate_execution_record(
            record,
            expected_document=doc,
            expected_run_fingerprint=manifest["run_fingerprint"],
            output_path=directory / "result" / (report.stem + ".docx"),
            profile=profile,
            require_strong=True,
            source_path=report,
            expected_model_identity_sha256=json_digest(identity["options"]["model"]),
        )
        if errors:
            raise ValueError(
                "execution receipt failed independent audit: " + "; ".join(errors)
            )
        if bundle and record.get("rsi_bundle_sha256") != bundle:
            raise ValueError("mixed strategy in frozen receipt")
        missing += len(record.get("coverage", {}).get("missing", []))
        conflicts += len(record.get("conflicts", []))
    ledger = audit_call_ledger(directory / "timing.sqlite", records)
    warmup = audit_warmup(directory, identity["options"]["warmup"])
    actual_timing = audit_summary_timing(result, ledger, records)
    strategy = (
        audit_strategy_execution(
            records, bundle, identity["options"].get("sandbox_digest")
        )
        if bundle
        else None
    )
    # Re-score source-bound gold and final artifacts; never trust candidate-supplied F1.
    from bridge_evidence.evaluation.generation import (
        METRIC_VERSION,
        aggregate_reports,
        score_report,
    )

    scores = []
    for case in samples:
        if file_digest(Path(case["gold"])) != case["gold_sha256"]:
            raise ValueError("frozen scoring labels changed")
        doc = document_identity(Path(case["report"]), Path(case["report"]).parent)
        record = json.loads((directory / "records" / record_name(doc)).read_text())
        scores.append(
            score_report(
                json.loads(Path(case["gold"]).read_text()), record["final_data"]
            )
        )
    metrics = aggregate_reports(scores)
    f1 = [metrics["sections"][s]["f1"] for s in ("basic", "advice", "defects")]
    if result["metric_version"] != METRIC_VERSION or result["f1"] != f1:
        raise ValueError("metric evidence differs from independently scored artifacts")
    return {
        **result,
        **actual_timing,
        "f1": f1,
        "status": "complete",
        "failures": metrics["reports_failed"],
        "missing_sources": missing,
        "conflicts": conflicts,
        "audit_passed": True,
        "generation_ledger": ledger,
        "comparison_identity": conditions,
        "model_identity": identity["options"]["model"],
        "warmup_ledger": warmup,
        "strategy_execution": strategy,
        "timing_sha256": file_digest(directory / "timing.sqlite"),
        "evaluation_sha256": file_digest(directory / "evaluation.json"),
    }, identity


_AUDIT_SEAL = object()


class AuditedEvaluationReceipt(dict):
    """In-process proof of completed trusted auditing, not a persisted flag.

    JSON serialization remains ordinary evidence storage. Reloaded dictionaries
    must pass the auditor again before optimizer selection can use them.
    """

    __slots__ = ("_audit_seal", "_audit_digest")

    def __init__(self, value, *, _seal):
        from .run_state import json_digest

        if _seal is not _AUDIT_SEAL:
            raise ValueError("only the trusted evaluator can seal a receipt")
        super().__init__(value)
        self._audit_seal = _seal
        self._audit_digest = json_digest(value)


def is_audited_evaluation(value):
    from .run_state import json_digest

    if (
        type(value) is not AuditedEvaluationReceipt
        or value._audit_seal is not _AUDIT_SEAL
    ):
        return False
    try:
        return value._audit_digest == json_digest(value)
    except (TypeError, ValueError):
        return False


def audited_optimizer_trial(directory, *, source_root):
    """Audit a search trial without claiming recovery or formal release acceptance.

    This entrypoint shares all semantic/source/ledger checks with the formal
    gate. It does not accept a recovery-skipping flag on the release path.
    """
    result, _ = _audited_evaluation(directory, source_root=source_root)
    audit_independent_receipts([result])
    return AuditedEvaluationReceipt(
        {**result, "status": "complete", "audit_scope": "optimizer-trial-only"},
        _seal=_AUDIT_SEAL,
    )


def load_receipt(directory, recovery_path, *, source_root):
    """Formal admission always rebuilds both the evaluation and recovery suite."""
    from pathlib import Path

    from .run_state import file_digest

    result, identity = _audited_evaluation(directory, source_root=source_root)
    recovery = audit_recovery_evidence(
        recovery_path,
        identity=identity,
        bundle_digest=result.get("bundle_sha256"),
        source_root=source_root,
    )
    result = {
        **result,
        "recovery": recovery,
        "recovery_sha256": file_digest(Path(recovery_path)),
        "audit_scope": "formal-rsi-receipt",
    }
    audit_independent_receipts([result])
    return AuditedEvaluationReceipt(result, _seal=_AUDIT_SEAL)


def verify_release(path, bundle_sha256, *, source_root):
    """Independently rebuild both receipts; a release flag alone is never sufficient."""
    import json
    from pathlib import Path

    from .run_state import file_digest

    path = Path(path)
    release = json.loads(path.read_text())
    if (
        release.get("schema_version") != 1
        or release.get("bundle_sha256") != bundle_sha256
    ):
        raise ValueError("release strategy identity mismatch")
    anchor_path = Path(release["anchor"])
    run_paths = [Path(p) for p in release["runs"]]
    if len(run_paths) != 2 or file_digest(anchor_path) != release["anchor_sha256"]:
        raise ValueError("release anchor evidence changed")
    if [file_digest(p) for p in run_paths] != release["receipts_sha256"]:
        raise ValueError("release evaluation evidence changed")
    anchor = load_receipt(
        anchor_path.parent,
        release["baseline"]["recovery"]["path"],
        source_root=source_root,
    )
    receipts = [
        load_receipt(p.parent, evidence["recovery"]["path"], source_root=source_root)
        for p, evidence in zip(run_paths, release["metrics"])
    ]
    errors = replacement_errors(anchor, receipts)
    if (
        release.get("comparison_identity") != anchor["comparison_identity"]
        or release.get("protected_sha256") != anchor["protected_sha256"]
    ):
        errors.append("release comparison/source identity changed")
    for rebuilt, snapshot in zip(
        [anchor, *receipts], [release["baseline"], *release["metrics"]]
    ):
        for key in ("recovery_sha256", "timing_sha256", "evaluation_sha256"):
            if rebuilt.get(key) != snapshot.get(key):
                errors.append("release evidence snapshot changed: " + key)
        if rebuilt.get("warmup_ledger") != snapshot.get("warmup_ledger"):
            errors.append("release warmup snapshot changed")
    try:
        audit_independent_receipts([anchor, *receipts])
    except ValueError as error:
        errors.append(str(error))
    if errors or any(r["bundle_sha256"] != bundle_sha256 for r in receipts):
        raise ValueError("release rejected: " + "; ".join(errors))
    return {
        "path": str(path.resolve()),
        "sha256": file_digest(path),
        "bundle_sha256": bundle_sha256,
        "model_identity": anchor["model_identity"],
    }
