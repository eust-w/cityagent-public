"""Real process death with a fake model; this is not PPU/model acceptance."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _recovery_worker(root, mode):
    # The child runs this file directly, independently of pytest's import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from bridge_evidence.contest import chunked_inference as ci
    from bridge_evidence.contest.backends import AccountedBackend
    from bridge_evidence.contest.candidate import validate_config
    from bridge_evidence.contest.long_document import DocumentSnapshot
    from bridge_evidence.contest.rsi_runtime import WorkLock
    from bridge_evidence.contest.run_state import atomic_json
    from bridge_evidence.contest.timing import TimingLedger

    class CharTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(message["content"] for message in messages)

        def __call__(self, text, **kwargs):
            return {"input_ids": list(map(ord, text))}

    class FakeBackend:
        def __call__(self, ids, maximum):
            counter = root / "model_calls.json"
            count = json.loads(counter.read_text()) if counter.exists() else 0
            atomic_json(counter, count + 1)
            if mode != "crash-window":
                raise AssertionError("resume must not invoke even the fake model")
            prompt = "".join(map(chr, ids))
            payload = json.JSONDecoder().raw_decode(prompt[prompt.index('{"owned":'):])[0]
            refs = payload["owned"]
            output = {
                "covered": [ref["key"] for ref in refs],
                "基本信息": {"桥梁名称": "测试桥"},
                "basic_sources": {"桥梁名称": "p0"},
                "维修建议": [],
                "病害明细": [
                    {
                        "source_id": ref["source_id"],
                        "instance_id": ref["key"],
                        "fields": {"病害部位": "梁", "病害描述": ref["text"]},
                        "temporal": "证据不足",
                    }
                    for ref in refs
                ],
            }
            return {
                "text": json.dumps(output, ensure_ascii=False),
                "finish_reason": "stop",
                "output_tokens": 100,
                "backend_request_id": "fake-process-recovery-1",
            }

        def batch(self, requests):
            return [self(*request) for request in requests]

    if mode == "crash-window":

        def pause_before_parse(raw, task, snapshot, call_id):
            # infer_chunked has synchronously committed its raw receipt and
            # checkpoint here. SIGKILL prevents all Python exception/finally paths.
            atomic_json(root / "ready.json", {"pid": os.getpid(), "call_id": call_id})
            time.sleep(30)
            raise TimeoutError("parent did not kill its test worker in time")

        ci.parse_fragment = pause_before_parse

    with WorkLock(root):
        result = ci.infer_chunked(
            DocumentSnapshot.from_text("测试桥"),
            CharTokenizer(),
            AccountedBackend(FakeBackend(), TimingLedger(root / "ledger.sqlite")),
            config=validate_config({"input_budget": 3000, "output_budget": 500}),
            cache_dir=root / "cache",
            max_attempts=1,
        )
        atomic_json(root / (mode + ".json"), result)


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL and flock require POSIX")
def test_sigkill_after_raw_receipt_recovers_without_another_model_call(tmp_path):
    from bridge_evidence.contest.rsi_runtime import TaskJournal, WorkLock
    from bridge_evidence.contest.timing import TimingLedger

    children = []

    def launch(mode, log):
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(tmp_path), mode],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        children.append(child)
        return child

    log_path = tmp_path / "child.log"
    try:
        with log_path.open("w") as log:
            child = launch("crash-window", log)
            deadline = time.monotonic() + 20
            ready_path = tmp_path / "ready.json"
            while not ready_path.exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready_path.exists(), log_path.read_text()
            ready = json.loads(ready_path.read_text())
            assert ready["pid"] == child.pid
            assert child.poll() is None
            with pytest.raises(RuntimeError, match="active owner"):
                with WorkLock(tmp_path):
                    pass

            checkpoint = next((tmp_path / "cache").glob("*/checkpoint.sqlite"))
            with TaskJournal(checkpoint, checkpoint.parent.name) as journal:
                before = journal.load()
            assert len(before["calls"]) == 1
            call = before["calls"][0]
            assert call["call_id"] == ready["call_id"]
            assert call["response_validation"] == "pending"
            assert not before["fragments"]
            assert len(before["pending"]) == 1
            raw_path = checkpoint.parent / call["raw_receipt"]["path"]
            raw_receipt = json.loads(raw_path.read_text())
            ledger = TimingLedger(tmp_path / "ledger.sqlite")
            ledger_before = ledger.summary()
            assert ledger_before["calls"] == 1
            assert ledger_before["failures"] == 0
            assert not ledger_before["incomplete_call_ids"]

            child.kill()
            assert child.wait(timeout=10) == -signal.SIGKILL
            # The kernel releases the dead owner's lock without recovery scans.
            with WorkLock(tmp_path):
                pass

            for mode in ("resume-raw", "resume-fragment"):
                resumed = launch(mode, log)
                assert resumed.wait(timeout=20) == 0, log_path.read_text()
                result = json.loads((tmp_path / (mode + ".json")).read_text())
                assert result["data"]["基本信息"]["桥梁名称"] == "测试桥"
                assert result["data"]["病害明细"][0]["病害描述"] == "测试桥"
                assert len(result["fragments"]) == 1
                assert len(result["model_calls"]) == 1
                final_call = result["model_calls"][0]
                assert final_call["call_id"] == ready["call_id"]
                assert final_call["status"] == "success"
                assert final_call["errors"] == []
                assert result["generation_evidence"]["responses"] == {
                    ready["call_id"]: raw_receipt["raw"]
                }
                assert hashlib.sha256(raw_receipt["raw"].encode()).hexdigest() == final_call["output_sha256"]
                assert json.loads((tmp_path / "model_calls.json").read_text()) == 1
                assert ledger.summary() == ledger_before
                with TaskJournal(checkpoint, checkpoint.parent.name) as journal:
                    state = journal.load()
                assert state["issued"] == 1
                assert [value["count"] for value in state["attempts"].values()] == [1]
                assert not state["pending"]
                assert len(state["fragments"]) == 1
    finally:
        # Only handles created above are signalled; never kill a process by name.
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)


if __name__ == "__main__":
    assert len(sys.argv) == 4 and sys.argv[1] == "--worker"
    _recovery_worker(Path(sys.argv[2]), sys.argv[3])
