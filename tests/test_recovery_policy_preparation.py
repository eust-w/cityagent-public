"""Preparation may warm a policy; recovery must retain its actual delay."""

import importlib.util
from pathlib import Path

import pytest

from bridge_evidence.contest.rsi import StrategyBundle
from bridge_evidence.contest.timing import TimingLedger


def driver_module():
    source = Path(__file__).resolve().parents[1] / "scripts/benchmark_rsi_recovery.py"
    spec = importlib.util.spec_from_file_location("recovery_prepare_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_preparation_is_counted_and_interrupt_reuses_controller(
    monkeypatch, tmp_path
):
    driver = driver_module()

    class Clock:
        now = 1_000_000_000

        def monotonic_ns(self):
            return self.now

        def time_ns(self):
            return self.now + 1_000_000_000_000_000_000

        def advance(self, seconds):
            self.now += int(seconds * 1e9)

    clock, policies, attempted, used = Clock(), [], set(), {}
    monkeypatch.setattr(driver, "time", clock)

    class Policy:
        error = None

        def __init__(self):
            self.waits = 0
            self.closed = False
            policies.append(self)

        def wait_ready(self, timeout):
            self.waits += 1
            clock.advance(2)
            return True

        def close(self):
            self.closed = True

    def infer(snapshot, tokenizer, generate, *, cache_dir, policy, **kwargs):
        assert policy.waits == 1, "controller must be warmed before the first fault"
        assert used.setdefault(cache_dir.name, policy) is policy
        if cache_dir not in attempted:
            attempted.add(cache_dir)
            try:
                generate([1], 4096)
            except (TimeoutError, OSError):
                pass
            except KeyboardInterrupt:
                clock.advance(0.5)
                raise
            clock.advance(0.5)
        generate([2], 4096)
        return {"coverage": {"complete": True}, "conflicts": [], "data": {}}

    calls = []

    def generate(ids, maximum):
        calls.append(ids)
        clock.advance(1)
        return {"call_id": str(len(calls)), "text": "{}"}

    monkeypatch.setattr(driver, "infer_chunked", infer)
    ledger = TimingLedger(tmp_path / "timing.sqlite")
    cases, _ = driver.run_cases(
        tmp_path, StrategyBundle.baseline(), generate, None, ledger,
        policy_factory=Policy,
    )
    assert len(policies) == len(driver.RECOVERY_KINDS)
    assert all(policy.waits == 1 and policy.closed for policy in policies)
    for kind, case in cases.items():
        assert case["start_seconds"] == 0.5
        assert case["finish_seconds"] == 1.5
        assert case["total_seconds"] == (3.5 if kind in ("timeout", "oom") else 4.5)


def test_failed_policy_preparation_never_starts_fault_or_model(monkeypatch, tmp_path):
    driver = driver_module()

    class Policy:
        error = "sandbox-preparation-failed"
        closed = False

        def wait_ready(self, timeout):
            return False

        def close(self):
            self.closed = True

    policy = Policy()
    monkeypatch.setattr(
        driver, "infer_chunked", lambda *args, **kwargs: pytest.fail("fault was started")
    )
    with pytest.raises(RuntimeError, match="policy preparation"):
        driver.run_cases(
            tmp_path, StrategyBundle.baseline(), None, None,
            TimingLedger(tmp_path / "timing.sqlite"), policy_factory=lambda: policy,
        )
    assert policy.closed
