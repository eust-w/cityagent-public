import pytest
from bridge_evidence.contest.backends import APIBackend, ModelManifest
from bridge_evidence.contest.long_document import (
    DocumentSnapshot,
    parse_fragment,
    ChunkTask,
    SourceSlice,
)


def test_api_rejects_external_endpoint():
    with pytest.raises(ValueError, match="local"):
        APIBackend("https://example.com/v1", "model", None)


def test_manifest_requires_adapter_matches_base(tmp_path):
    import json

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "another"})
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="base"):
        ModelManifest(str(base), str(base), str(adapter)).validate()


def test_unowned_source_cannot_claim_coverage():
    snapshot = DocumentSnapshot.from_text("梁\n墩")
    s = SourceSlice("p0", 0, 2, "梁\n")
    task = ChunkTask("task", [s], [])
    import json

    raw = json.dumps(
        {
            "covered": [s.key],
            "基本信息": {"桥梁名称": "桥"},
            "basic_sources": {"桥梁名称": "p1"},
            "维修建议": [],
            "病害明细": [],
        }
    )
    fragment = parse_fragment(raw, task, snapshot)
    assert fragment.basic["桥梁名称"] is None and fragment.basic_sources == {}


def test_accounted_api_batch_preserves_success_when_one_request_times_out(tmp_path):
    from bridge_evidence.contest.backends import AccountedBackend
    from bridge_evidence.contest.timing import TimingLedger

    class Backend(APIBackend):
        def __call__(self, ids, maximum):
            if ids == [2]:
                raise TimeoutError("temporary")
            return {"text": "{}", "finish_reason": "stop", "output_tokens": 2}

    ledger = TimingLedger(tmp_path / "time.sqlite")
    backend = AccountedBackend(
        Backend("http://127.0.0.1:8000/v1", "m", None, workers=2), ledger
    )
    responses = backend.batch([([1], 10), ([2], 10)])
    assert responses[0]["finish_reason"] == "stop"
    assert isinstance(responses[1], TimeoutError)
    assert responses[0]["call_id"] != responses[1].call_id
    assert ledger.summary()["calls"] == 2
    assert ledger.summary()["failures"] == 1


def test_repair_role_is_bound_to_unique_ledger_call(tmp_path):
    from bridge_evidence.contest.backends import AccountedBackend
    from bridge_evidence.contest.timing import TimingLedger

    ledger = TimingLedger(tmp_path / "time.sqlite")
    backend = AccountedBackend(
        lambda ids, maximum: {"text": "{}", "finish_reason": "stop"}, ledger
    )
    response = backend.call_with_role([1], 10, role="react-repair")
    with ledger.connect() as db:
        assert (
            db.execute(
                "SELECT role FROM calls WHERE id=?", (response["call_id"],)
            ).fetchone()[0]
            == "react-repair"
        )


def test_accounted_latency_equals_ledger_receipt(tmp_path):
    from bridge_evidence.contest.backends import AccountedBackend
    from bridge_evidence.contest.timing import TimingLedger
    from bridge_evidence.contest.chunked_inference import _complete, _receipt

    ledger = TimingLedger(tmp_path / "time.sqlite")
    backend = AccountedBackend(
        lambda ids, maximum: {
            "text": "{}",
            "finish_reason": "stop",
            "output_tokens": 2,
        },
        ledger,
    )
    response = backend([1], 10)
    call = _receipt([1], "extraction")
    _complete(call, response, 1000, 10)
    with ledger.connect() as db:
        measured = db.execute(
            "SELECT seconds FROM calls WHERE id=?", (response["call_id"],)
        ).fetchone()[0]
    assert call["seconds"] == measured == response["latency_seconds"]
