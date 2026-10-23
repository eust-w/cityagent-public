"""The external acceptance gate must rebuild evidence, not trust summary flags."""

import importlib
import json

import pytest

from bridge_evidence.contest.run_state import json_digest
from bridge_evidence.contest.timing import TimingLedger


def acceptance():
    return importlib.import_module("bridge_evidence.contest.rsi_acceptance")


def ledger_fixture(tmp_path):
    path = tmp_path / "timing.sqlite"
    ledger = TimingLedger(path)
    call = {
        "call_id": "call1",
        "role": "extraction",
        "status": "success",
        "seconds": 0.5,
        "prompt_sha256": "a" * 64,
        "output_sha256": "b" * 64,
        "output_tokens": 12,
        "finish_reason": "stop",
    }
    receipt = {
        key: call[key]
        for key in ("prompt_sha256", "output_sha256", "output_tokens", "finish_reason")
    }
    receipt["backend_request_id"] = "backend-1"
    ledger.start("call1", role="extraction")
    ledger.finish("call1", 0.5, status="success", receipt=receipt)
    return path, ledger, call


def test_missing_actual_ledger_cannot_be_replaced_by_verified_flags(tmp_path):
    with pytest.raises(ValueError, match="ledger"):
        acceptance().audit_call_ledger(
            tmp_path / "missing.sqlite", [{"schema_version": 3, "model_calls": []}]
        )
    assert not (tmp_path / "missing.sqlite").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_sha256", "c" * 64),
        ("output_sha256", "c" * 64),
        ("seconds", 0.01),
        ("output_tokens", 1),
        ("role", "proposal"),
    ],
)
def test_real_ledger_binds_each_generation_and_timing(tmp_path, field, value):
    path, _, call = ledger_fixture(tmp_path)
    changed = {**call, field: value}
    with pytest.raises(ValueError, match="ledger"):
        acceptance().audit_call_ledger(
            path, [{"schema_version": 3, "model_calls": [changed]}]
        )


def test_parser_retry_preserves_backend_success_and_frozen_exact_seconds(tmp_path):
    path, _, call = ledger_fixture(tmp_path)
    call["status"] = "failed"
    call["errors"] = ["invalid JSON"]
    result = acceptance().audit_call_ledger(
        path, [{"schema_version": 3, "model_calls": [call]}]
    )
    assert result["call_ids"] == ["call1"]
    assert result["generation_seconds"] == 0.5


def test_extra_or_duplicate_model_receipt_cannot_be_hidden(tmp_path):
    path, ledger, call = ledger_fixture(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        acceptance().audit_call_ledger(
            path, [{"schema_version": 3, "model_calls": [call, call]}]
        )
    ledger.start("extra", role="extraction")
    ledger.finish("extra", 0.1, status="failed")
    with pytest.raises(ValueError, match="ledger"):
        acceptance().audit_call_ledger(
            path, [{"schema_version": 3, "model_calls": [call]}]
        )


def test_full_fallback_cannot_qualify_as_exercised_frozen_policy():
    with pytest.raises(ValueError, match="never executed"):
        acceptance().audit_strategy_execution(
            [{"rsi": {"decisions": [{"fallback": True}]}}], "a" * 64, "b" * 64
        )


def test_boolean_false_without_real_kernel_proof_is_not_strategy_evidence():
    from bridge_evidence.contest.rsi import base_action

    state = {"phase": "schedule", "remaining_calls": 1, "remaining_seconds": 10}
    # Missing runtime/kernel evidence must fail even when action/state are valid.
    decision = {
        "fallback": False,
        "fallback_reason": None,
        "bundle_sha256": "a" * 64,
        "state": state,
        "state_sha256": json_digest(state),
        "action": base_action(state),
    }
    with pytest.raises(ValueError):
        acceptance().audit_strategy_execution(
            [{"rsi": {"decisions": [decision]}}], "a" * 64, "b" * 64
        )


def test_readonly_namespace_proof_must_match_bound_runtime_digest():
    from bridge_evidence.contest.rsi import base_action
    from bridge_evidence.contest.rsi_sandbox import policy_state

    state = policy_state(
        {"phase": "schedule", "remaining_calls": 1, "remaining_seconds": 10}
    )
    decision = {
        "fallback": False,
        "fallback_reason": None,
        "bundle_sha256": "a" * 64,
        "state": state,
        "state_sha256": json_digest(state),
        "action": base_action(state),
        "sandbox_runtime_digest": "b" * 64,
        "kernel_proof": {
            "pid": 2,
            "seccomp": 2,
            "no_new_privs": 1,
            "capabilities": 0,
            "namespaces": 6,
            "readonly_root": True,
        },
    }
    result = acceptance().audit_strategy_execution(
        [{"rsi": {"decisions": [decision]}}], "a" * 64, "b" * 64
    )
    assert result["executed_decisions"] == 1
    with pytest.raises(ValueError, match="isolation"):
        acceptance().audit_strategy_execution(
            [{"rsi": {"decisions": [decision]}}], "a" * 64, "c" * 64
        )


def test_summary_latency_and_wall_must_match_actual_execution_evidence():
    ledger = {
        "call_ids": ["c"],
        "generation_seconds": 2.5,
        "failures": 0,
        "spans": {"batch_wall": 10},
    }
    result = {
        "timing": {"calls": 1, "generation_seconds": 2.5, "failures": 0},
        "wall_seconds": 10.01,
        "latency": {"p50": 3.0, "p95": 3.9},
    }
    records = [{"report_wall_seconds": 2}, {"report_wall_seconds": 4}]
    checked = acceptance().audit_summary_timing(result, ledger, records)
    assert checked["wall_seconds"] == 10
    forged = {**result, "latency": {"p50": 0.1, "p95": 0.2}}
    with pytest.raises(ValueError, match="latency"):
        acceptance().audit_summary_timing(forged, ledger, records)
    with pytest.raises(ValueError, match="wall"):
        acceptance().audit_summary_timing(
            {**result, "wall_seconds": 20}, ledger, records
        )


def test_failed_backend_attempt_requires_prompt_binding_too(tmp_path):
    ledger = TimingLedger(tmp_path / "timing.sqlite")
    ledger.start("failed", role="extraction")
    ledger.finish("failed", 0.1, status="failed")
    call = {
        "call_id": "failed",
        "role": "extraction",
        "status": "failed",
        "seconds": 0.1,
        "prompt_sha256": "a" * 64,
        "output_sha256": None,
        "errors": ["timeout"],
    }
    with pytest.raises(ValueError, match="prompt_sha256"):
        acceptance().audit_call_ledger(
            tmp_path / "timing.sqlite", [{"schema_version": 3, "model_calls": [call]}]
        )
    with ledger.connect() as connection:
        connection.execute(
            "UPDATE calls SET receipt=? WHERE id=?",
            (
                '{"prompt_sha256":"'
                + "a" * 64
                + '","error_type":"TimeoutError","outcome":"unknown"}',
                "failed",
            ),
        )
    assert (
        acceptance().audit_call_ledger(
            tmp_path / "timing.sqlite", [{"schema_version": 3, "model_calls": [call]}]
        )["failures"]
        == 1
    )


def test_local_backend_request_identity_is_scoped_to_actual_engine_instance(tmp_path):
    first = {"backend_request_id": "0", "backend_instance_id": "instance-one"}
    second = {"backend_request_id": "0", "backend_instance_id": "instance-two"}
    assert acceptance().backend_request_key(first) != acceptance().backend_request_key(
        second
    )
    assert acceptance().backend_request_key(
        {"backend_request_id": "api-id"}
    ) == acceptance().backend_request_key(
        {"backend_request_id": "api-id", "backend_instance_id": None}
    )
    with pytest.raises(ValueError, match="instance"):
        acceptance().backend_request_key(
            {"backend_request_id": "0", "backend_instance_id": ""}
        )


def test_same_raw_vendor_id_across_local_instances_is_not_receipt_reuse(tmp_path):
    path, ledger, first = ledger_fixture(tmp_path)
    with ledger.connect() as db:
        raw = json.loads(
            db.execute("SELECT receipt FROM calls WHERE id=?", ("call1",)).fetchone()[0]
        )
        raw.update(backend_request_id="0", backend_instance_id="engine-a")
        db.execute("UPDATE calls SET receipt=? WHERE id=?", (json.dumps(raw), "call1"))
    second = {**first, "call_id": "call2"}
    ledger.start("call2", role="extraction")
    ledger.finish(
        "call2",
        0.5,
        status="success",
        receipt={**raw, "backend_instance_id": "engine-b"},
    )
    audited = acceptance().audit_call_ledger(
        path, [{"schema_version": 3, "model_calls": [first, second]}]
    )
    assert audited["backend_request_ids"] == ["0", "0"]
    assert len(set(audited["backend_request_keys"])) == 2


def test_comparison_identity_is_rebuilt_from_bound_inputs_and_model(tmp_path):
    from bridge_evidence.evaluation.generation import METRIC_VERSION

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "evaluate_multitask.py").write_text("trusted evaluator")
    from bridge_evidence.contest.run_state import file_digest

    options = {
        "protocol": "complete-multitask-report-v2",
        "metric_version": METRIC_VERSION,
        "hardware": "fixed-device",
        "warmup": 1,
        "backend": "vllm-api",
        "url": "http://127.0.0.1:8000/v1",
        "engine": "direct",
        "model": {"files": {"base": {"weights": "a" * 64}}},
        "manifest": {
            "samples": [
                {"id": "a", "report_sha256": "b" * 64, "gold_sha256": "c" * 64}
            ],
            "metric_denominator": 1,
        },
        "evaluator_sha256": file_digest(scripts / "evaluate_multitask.py"),
    }
    identity = {"code_files": {"run.py": "d" * 64}, "options": options}
    before = acceptance().comparison_identity(identity, source_root=tmp_path)
    options["manifest"]["samples"][0]["report_sha256"] = "e" * 64
    assert acceptance().comparison_identity(identity, source_root=tmp_path) != before
    options["model"]["files"]["base"]["weights"] = "f" * 64
    assert acceptance().comparison_identity(identity, source_root=tmp_path) != before
    options["evaluator_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="evaluator"):
        acceptance().comparison_identity(identity, source_root=tmp_path)


def test_canonical_protected_source_set_cannot_omit_or_add_files(tmp_path):
    from bridge_evidence.contest.run_state import file_digest

    names = acceptance().PROTECTED_RUNTIME_FILES
    for name in (*names, "src/bridge_evidence/contest/one.py"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(name)
    files = {
        str(p.relative_to(tmp_path)): file_digest(p)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert acceptance().audit_protected_source(files, tmp_path) == json_digest(files)
    incomplete = dict(files)
    incomplete.pop("run.py")
    with pytest.raises(ValueError, match="protected"):
        acceptance().audit_protected_source(incomplete, tmp_path)
    with pytest.raises(ValueError, match="protected"):
        acceptance().audit_protected_source({**files, "../secret": "a" * 64}, tmp_path)


def test_legacy_recovery_or_wrong_candidate_cannot_qualify(tmp_path):
    source = tmp_path / "recovery.json"
    source.write_text(
        json.dumps(
            {
                "protocol": "rsi-recovery-v1",
                "cases": {},
                "actual_model_call_ids": ["invented"],
            }
        )
    )
    with pytest.raises(ValueError, match="recovery"):
        acceptance().audit_recovery_evidence(
            source,
            identity={"options": {}},
            bundle_digest="a" * 64,
            source_root=tmp_path,
        )


def test_independence_includes_recovery_and_warmup_actual_calls():
    entry = {
        "generation_ledger": {"call_ids": ["eval"], "backend_request_keys": ["eval-r"]},
        "recovery": {
            "generation_ledger": {
                "call_ids": ["recover"],
                "backend_request_keys": ["recover-r"],
            }
        },
        "warmup_ledger": {"call_ids": ["warmup"], "backend_request_keys": ["warmup-r"]},
    }
    replay = {
        "generation_ledger": {
            "call_ids": ["another"],
            "backend_request_keys": ["another-r"],
        },
        "recovery": {
            "generation_ledger": {
                "call_ids": ["recover"],
                "backend_request_keys": ["recover-r"],
            }
        },
    }
    with pytest.raises(ValueError, match="independent"):
        acceptance().audit_independent_receipts([entry, replay])


def recovery_artifacts(tmp_path):
    """Deterministic unit fixture only; does not certify real PPU inference."""
    import importlib.util
    from pathlib import Path

    from bridge_evidence.contest.backends import AccountedBackend
    from bridge_evidence.contest.rsi import StrategyBundle
    from bridge_evidence.contest.run_state import atomic_json, file_digest

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "recovery_driver_fixture", root / "scripts/benchmark_rsi_recovery.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(m["content"] for m in messages)

        def __call__(self, text, **kwargs):
            return {"input_ids": list(map(ord, text))}

    class Backend:
        def __init__(self):
            self.count = 0

        def __call__(self, ids, maximum):
            self.count += 1
            if maximum == 1:
                return {
                    "text": "{",
                    "output_tokens": 1,
                    "finish_reason": "length",
                    "backend_request_id": "fixture-" + str(self.count),
                }
            prompt = "".join(map(chr, ids))
            payload = json.JSONDecoder().raw_decode(
                prompt[prompt.index('{"owned":') :]
            )[0]
            owned = payload["owned"]
            basic = (
                {"桥梁名称": "RSI恢复验证桥"}
                if any(r["source_id"] == "p0" for r in owned)
                else {}
            )
            value = {
                "covered": [r["key"] for r in owned],
                "基本信息": basic,
                "basic_sources": {"桥梁名称": "p0"} if basic else {},
                "维修建议": [],
                "病害明细": [
                    {
                        "source_id": r["source_id"],
                        "instance_id": r["key"],
                        "fields": {"病害部位": "梁", "病害描述": r["text"]},
                        "temporal": "证据不足",
                    }
                    for r in owned
                ],
            }
            return {
                "text": json.dumps(value, ensure_ascii=False),
                "output_tokens": 100,
                "finish_reason": "stop",
                "backend_request_id": "fixture-" + str(self.count),
            }

    bundle = StrategyBundle.baseline()
    ledger = TimingLedger(tmp_path / "timing.sqlite")
    cases, actual = driver.run_cases(
        tmp_path, bundle, AccountedBackend(Backend(), ledger), Tokenizer(), ledger
    )
    identity = {
        "code_files": {},
        "options": {
            "model": {"test-only": True},
            "url": "http://127.0.0.1:8000/v1",
            "config": bundle.config,
            "sandbox_digest": None,
        },
    }
    recovery = {
        "protocol": "rsi-recovery-v2",
        "mode": "off",
        "bundle_sha256": None,
        "model": identity["options"]["model"],
        "url": identity["options"]["url"],
        "config": bundle.config,
        "sandbox_digest": None,
        "protected_sha256": json_digest({}),
        "cases": cases,
        "actual_model_call_ids": actual,
        "driver_sha256": file_digest(root / "scripts/benchmark_rsi_recovery.py"),
        "artifacts": {
            name: file_digest(tmp_path / name)
            for name in [
                "timing.sqlite",
                *(
                    kind + suffix
                    for kind in acceptance().RECOVERY_KINDS
                    for suffix in (".json", ".events.json")
                ),
            ]
        },
    }
    path = tmp_path / "recovery.json"
    atomic_json(path, recovery)
    return path, recovery, identity, root


def test_recovery_metrics_rebuild_from_actual_events_cases_and_ledger(tmp_path):
    path, recovery, identity, root = recovery_artifacts(tmp_path)
    audited = acceptance().audit_recovery_evidence(
        path, identity=identity, bundle_digest=None, source_root=root
    )
    assert audited["cases"] == recovery["cases"]
    assert audited["generation_ledger"]["call_ids"] == recovery["actual_model_call_ids"]
    # Changing a displayed scalar alone never changes a trusted duration.
    recovery["cases"]["timeout"]["finish_seconds"] = 0
    path.write_text(json.dumps(recovery))
    with pytest.raises(ValueError, match="duration"):
        acceptance().audit_recovery_evidence(
            path, identity=identity, bundle_digest=None, source_root=root
        )


@pytest.mark.parametrize(
    "tamper", ["bundle", "mode", "model", "call_id", "case", "ledger"]
)
def test_recovery_other_candidate_or_changed_raw_artifacts_rejected(tmp_path, tamper):
    path, recovery, identity, root = recovery_artifacts(tmp_path)
    if tamper in ("bundle", "mode", "model"):
        key = "bundle_sha256" if tamper == "bundle" else tamper
        recovery[key] = "not-the-tested-candidate"
    elif tamper == "call_id":
        recovery["actual_model_call_ids"][0] = "invented"
    elif tamper == "case":
        case_path = tmp_path / "timeout.json"
        case = json.loads(case_path.read_text())
        case["data"]["病害明细"] = []
        case_path.write_text(json.dumps(case))
        from bridge_evidence.contest.run_state import file_digest

        recovery["artifacts"]["timeout.json"] = file_digest(case_path)
    else:
        ledger = TimingLedger(tmp_path / "timing.sqlite")
        with ledger.connect() as db:
            db.execute("UPDATE calls SET seconds=0")
        from bridge_evidence.contest.run_state import file_digest

        recovery["artifacts"]["timing.sqlite"] = file_digest(tmp_path / "timing.sqlite")
    path.write_text(json.dumps(recovery))
    with pytest.raises(ValueError):
        acceptance().audit_recovery_evidence(
            path, identity=identity, bundle_digest=None, source_root=root
        )


def test_fault_clock_starts_at_observation_not_first_generation(tmp_path):
    path, _, _, _ = recovery_artifacts(tmp_path)
    for kind in ("format", "length", "interrupt"):
        events = json.loads((path.parent / (kind + ".events.json")).read_text())[
            "events"
        ]
        fault = next(e["monotonic_ns"] for e in events if e["event"] == "fault")
        first_completed = next(
            e["monotonic_ns"] for e in events if e["event"] == "call_finish"
        )
        assert fault >= first_completed, (
            "model first-generation speed must not hide recovery latency"
        )


def test_faster_first_generation_cannot_mask_slower_recovery_start(
    tmp_path, monkeypatch
):
    import importlib.util
    from pathlib import Path

    from bridge_evidence.contest.rsi import StrategyBundle

    script = Path(__file__).resolve().parents[1] / "scripts/benchmark_rsi_recovery.py"
    spec = importlib.util.spec_from_file_location("recovery_clock_test", script)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    def scenario(name, first_generation, recovery_delay):
        class Clock:
            now = 1_000_000_000

            def monotonic_ns(self):
                return self.now

            def time_ns(self):
                return self.now + 1_000_000_000_000_000_000

            def advance(self, seconds):
                self.now += int(seconds * 1e9)

        clock, attempted = Clock(), set()
        monkeypatch.setattr(driver, "time", clock)

        def infer(snapshot, tokenizer, generate, *, cache_dir, **kwargs):
            if cache_dir not in attempted:
                attempted.add(cache_dir)
                try:
                    generate([1], 4096)
                except (TimeoutError, OSError):
                    pass
                except KeyboardInterrupt:
                    clock.advance(recovery_delay)
                    raise
                clock.advance(recovery_delay)
            generate([2], 4096)
            return {"coverage": {"complete": True}, "conflicts": [], "data": {}}

        count = 0

        def generate(ids, maximum):
            nonlocal count
            count += 1
            clock.advance(first_generation if ids == [1] else 0.01)
            return {"call_id": str(count), "text": "{}"}

        monkeypatch.setattr(driver, "infer_chunked", infer)
        output = tmp_path / name
        output.mkdir()
        return driver.run_cases(
            output,
            StrategyBundle.baseline(),
            generate,
            None,
            TimingLedger(output / "timing.sqlite"),
        )[0]

    anchor = scenario("anchor", first_generation=20, recovery_delay=0.1)
    candidate = scenario("candidate", first_generation=10, recovery_delay=5)
    # Total completion improves, but the recovery gate must still reject all three.
    for kind in ("length", "format", "interrupt"):
        assert candidate[kind]["total_seconds"] < anchor[kind]["total_seconds"]
        assert candidate[kind]["start_seconds"] == 5
        assert anchor[kind]["start_seconds"] == 0.1
        assert candidate[kind]["finish_seconds"] > anchor[kind]["finish_seconds"]


def optimizer_trial_artifacts(tmp_path):
    """Unit artifacts with real parsing/ledger checks, not real model acceptance."""
    from pathlib import Path

    from test_execution_reconstruction import reconstructed

    from bridge_evidence.contest.run_state import atomic_json, file_digest, record_name
    from bridge_evidence.evaluation.generation import (
        METRIC_VERSION,
        aggregate_reports,
        score_report,
    )

    source, _, record = reconstructed.__wrapped__(tmp_path)
    code = tmp_path / "code"
    for name in (
        *acceptance().PROTECTED_RUNTIME_FILES,
        "src/bridge_evidence/contest/fixture.py",
        "scripts/evaluate_multitask.py",
    ):
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# isolated identity fixture")
    files = {
        str(p.relative_to(code)): file_digest(p)
        for p in code.rglob("*")
        if p.is_file() and str(p.relative_to(code)) != "scripts/evaluate_multitask.py"
    }
    gold = tmp_path / "gold.json"
    atomic_json(gold, record["final_data"])
    sample = {
        "id": "one",
        "report": str(source),
        "report_sha256": file_digest(source),
        "gold": str(gold),
        "gold_sha256": file_digest(gold),
    }
    identity = {
        "code_files": files,
        "options": {
            "model": {"files": {"base": {"weight": "a" * 64}}},
            "rsi_mode": "off",
            "manifest": {"samples": [sample], "metric_denominator": 1},
            "engine": "direct",
            "metric_version": METRIC_VERSION,
            "protocol": "complete-multitask-report-v2",
            "hardware": "unit-fixture",
            "backend": "vllm-api",
            "url": "http://127.0.0.1:8000/v1",
            "warmup": 0,
            "evaluator_sha256": file_digest(code / "scripts/evaluate_multitask.py"),
        },
    }
    fingerprint = json_digest(identity)
    atomic_json(
        tmp_path / "run_manifest.json",
        {"identity": identity, "run_fingerprint": fingerprint},
    )
    record.update(run_fingerprint=fingerprint, report_wall_seconds=1.0)
    records = tmp_path / "records"
    records.mkdir()
    atomic_json(records / record_name(record["document"]), record)
    ledger = TimingLedger(tmp_path / "timing.sqlite")
    for call in record["model_calls"]:
        ledger.start(call["call_id"], role=call.get("role", "extraction"))
        ledger.finish(
            call["call_id"],
            call["seconds"],
            status="success",
            receipt={**call, "backend_request_id": "unit-" + call["call_id"]},
        )
    ledger.span("batch-wall", "batch_wall", 2.0)
    TimingLedger(tmp_path / "warmup.sqlite")
    atomic_json(
        tmp_path / "warmup.json",
        {
            "protocol": "actual-warmup-v1",
            "requests": 0,
            "generations": [],
            "timing_sha256": file_digest(tmp_path / "warmup.sqlite"),
        },
    )
    metrics = aggregate_reports(
        [score_report(record["final_data"], record["final_data"])]
    )
    conditions = acceptance().comparison_identity(identity, source_root=code)
    summary = {
        "run_id": "unit-trial",
        "denominator": 1,
        "completed": 1,
        "reports": [{"id": "one"}],
        "comparison_identity": conditions,
        "conditions": conditions,
        "protected_sha256": json_digest(files),
        "bundle_sha256": None,
        "cache_reused": False,
        "metric_version": METRIC_VERSION,
        "f1": [metrics["sections"][s]["f1"] for s in ("basic", "advice", "defects")],
        "timing": ledger.summary(),
        "wall_seconds": 2.01,
        "latency": {"p50": 1.0, "p95": 1.0},
    }
    atomic_json(tmp_path / "evaluation.json", summary)
    return Path(tmp_path), code, summary, records / record_name(record["document"])


def test_optimizer_trial_uses_full_audit_without_claiming_recovery(tmp_path):
    directory, code, _, _ = optimizer_trial_artifacts(tmp_path)
    result = acceptance().audited_optimizer_trial(directory, source_root=code)
    assert result["audit_passed"] is True
    assert result["audit_scope"] == "optimizer-trial-only"
    assert result["status"] == "complete"
    assert result["generation_ledger"]["call_ids"]
    assert "recovery" not in result
    # The release entrypoint still independently requires the fault suite.
    with pytest.raises((OSError, ValueError)):
        acceptance().load_receipt(
            directory, tmp_path / "missing-recovery.json", source_root=code
        )


@pytest.mark.parametrize("tamper", ["f1", "cache", "raw", "ledger"])
def test_optimizer_trial_cannot_certify_self_declared_or_changed_evidence(
    tmp_path, tamper
):
    directory, code, summary, record_path = optimizer_trial_artifacts(tmp_path)
    if tamper == "f1":
        summary["f1"] = [0, 0, 0]
    elif tamper == "cache":
        summary["cache_reused"] = True
    elif tamper == "raw":
        record = json.loads(record_path.read_text())
        record.pop("generation_evidence")
        record_path.write_text(json.dumps(record))
    else:
        (directory / "timing.sqlite").unlink()
    (directory / "evaluation.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        acceptance().audited_optimizer_trial(directory, source_root=code)


def test_optimizer_audit_identity_cannot_be_replaced_by_flags_or_mutated(tmp_path):
    directory, code, _, _ = optimizer_trial_artifacts(tmp_path)
    receipt = acceptance().audited_optimizer_trial(directory, source_root=code)
    assert acceptance().is_audited_evaluation(receipt)
    assert not acceptance().is_audited_evaluation(dict(receipt))
    assert not acceptance().is_audited_evaluation(json.loads(json.dumps(receipt)))
    receipt["wall_seconds"] = 0.0001
    assert not acceptance().is_audited_evaluation(receipt)
