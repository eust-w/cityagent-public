"""Raw response evidence only; fake HTTP transport, no SDK/network/model."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
from urllib.error import HTTPError

import pytest

from bridge_evidence.contest import raw_sse_capture as capture_module
from bridge_evidence.contest.local_model_client import LocalModelClient
from bridge_evidence.contest.reliable_generation import StreamPolicy, ReliableGenerationError, _Aggregate
from test_reliable_generation import MODEL, Clock, Response, Opener, complete, event


@pytest.fixture(autouse=True)
def no_external_actions(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("No network or subprocess")
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)


def client(chunks, directory=None, *, policy=None, response=None, diagnostic_token_ids=False):
    clock = Clock()
    value = LocalModelClient("http://127.0.0.1:18110", MODEL,
        api_key="HEADER_SECRET_NOT_IN_CAPTURE", enable_thinking=True, temperature=1)
    value.enable_reliable_generation(policy or StreamPolicy(clock=clock), capture_directory=directory,
                                    diagnostic_token_ids=diagnostic_token_ids)
    value._opener = Opener(response or Response(chunks, clock), clock)
    return value


def request(value):
    return value._request("/v1/chat/completions", {"model": MODEL,
        "messages": [{"role": "user", "content": "Synthetic task"}], "max_tokens": 1000,
        "temperature": 1, "chat_template_kwargs": {"enable_thinking": True}}, kind="agent_generation")


def saved(receipt):
    metadata = receipt["raw_sse_capture"]
    raw = Path(metadata["path"]).read_bytes()
    assert metadata["bytes"] == len(raw)
    assert metadata["sha256"] == hashlib.sha256(raw).hexdigest()
    assert stat.S_IMODE(Path(metadata["path"]).stat().st_mode) == 0o600
    assert metadata["call_id"] == receipt["call_id"]
    return raw, metadata


def test_fragmented_bytes_replay_to_exact_same_aggregate_and_payload_is_unchanged(tmp_path):
    raw = complete({"content": "Synthetic unicode 跨字节", "reasoning_content": "Synthetic note"})
    chunks = [raw[i:i+3] for i in range(0, len(raw), 3)]
    enabled = client(chunks, tmp_path)
    disabled = client(chunks)
    result, receipt = request(enabled)
    original_result, original_receipt = request(disabled)
    captured, metadata = saved(receipt)
    assert captured == raw and result == original_result
    replay = _Aggregate(StreamPolicy()); replay.feed(captured, final=True)
    assert replay.result(MODEL) == result
    assert enabled._opener.calls[0][0].data == disabled._opener.calls[0][0].data
    assert metadata["status"] == "complete" and metadata["complete"] is True
    assert metadata["observed_all_received_bytes"] is True
    assert metadata["scope"] == "consumer_observed_bytes"
    assert metadata["http_body_eof_proven"] is False
    assert b"HEADER_SECRET_NOT_IN_CAPTURE" not in captured
    assert "raw_sse_capture" not in original_receipt


@pytest.mark.parametrize("tail", [b"", b"data: [DONE]\n\ndata: {"])
def test_protocol_failure_preserves_raw_prefix_without_promoting_done_or_usage(tmp_path, tail):
    raw = event({"content": "Synthetic unfinished output"}) + tail
    value = client([raw], tmp_path)
    with pytest.raises(ReliableGenerationError) as caught:
        request(value)
    receipt = caught.value.receipt
    captured, metadata = saved(receipt)
    assert captured == raw and value.calls[-1] is receipt
    assert receipt["usage"] is None and receipt["usage_known"] is False
    assert metadata["status"] == "partial" and metadata["complete"] is False
    assert metadata["observed_all_received_bytes"] is True
    assert metadata["done_received"] is bool(tail)
    assert metadata["usage_known"] is False


def test_disconnect_preserves_bytes_and_original_transport_error(tmp_path):
    raw = event({"content": "Synthetic partial"})
    value = client([raw, OSError("Private exception detail must not be copied")], tmp_path)
    with pytest.raises(ReliableGenerationError) as caught:
        request(value)
    captured, metadata = saved(caught.value.receipt)
    assert captured == raw
    assert caught.value.code == "stream_transport_error"
    assert metadata["complete"] is False and metadata["errors"] == []


def test_http_400_is_empty_sse_not_started_not_an_error_body_capture(tmp_path):
    body = b'{"error":"Private HTTP error body"}'
    failure = HTTPError("http://127.0.0.1:18110/v1/chat/completions", 400,
        "Synthetic failure", {}, io.BytesIO(body))
    value = client([], tmp_path, response=failure)
    with pytest.raises(ReliableGenerationError) as caught:
        request(value)
    captured, metadata = saved(caught.value.receipt)
    assert captured == b"" and metadata["status"] == "not_started"
    assert metadata["http_status"] == 400
    assert metadata["done_received"] is metadata["usage_known"] is metadata["complete"] is False
    assert body not in captured


def test_stream_limit_chunk_not_delivered_to_observer_is_explicit_partial_gap(tmp_path):
    first = event({"content": "Synthetic partial"})
    last = complete()
    clock = Clock()
    policy = StreamPolicy(clock=clock, max_response_bytes=len(first)+1)
    value = client([first, last], tmp_path, policy=policy)
    with pytest.raises(ReliableGenerationError) as caught:
        request(value)
    captured, metadata = saved(caught.value.receipt)
    assert captured == first
    assert metadata["received_bytes"] == len(first)+len(last)
    assert metadata["observer_bytes"] == len(first)
    assert metadata["observed_all_received_bytes"] is metadata["complete"] is False
    assert metadata["status"] == "partial"


def test_directory_failure_is_explicit_without_changing_success_or_cancelling(tmp_path):
    destination = tmp_path / "occupied"
    destination.write_text("Unrelated file")
    value = client([complete()], destination)
    result, receipt = request(value)
    assert result["choices"][0]["finish_reason"] == "stop"
    assert receipt["status"] == "success" and receipt["usage_known"] is True
    metadata = receipt["raw_sse_capture"]
    assert metadata["status"] == "failed" and metadata["path"] is None
    assert metadata["errors"] == [{"operation": "persist", "type": "FileExistsError"}]
    assert value._opener.response.cancelled == []
    assert destination.read_text() == "Unrelated file"


def test_partial_disk_write_failure_preserves_actual_file_hash_and_service_success(tmp_path, monkeypatch):
    real_write = os.write
    counts = []
    def broken(fd, raw):
        counts.append(len(raw))
        if len(counts) == 1:
            return real_write(fd, raw[:5])
        raise OSError("Private source text and key must not enter metadata")
    monkeypatch.setattr(capture_module.os, "write", broken)
    value = client([complete()], tmp_path)
    result, receipt = request(value)
    captured, metadata = saved(receipt)
    assert len(captured) == 5
    assert receipt["status"] == "success" and result["choices"]
    assert metadata["status"] == "failed" and metadata["complete"] is False
    assert metadata["observed_all_received_bytes"] is False
    assert metadata["errors"] == [{"operation": "persist", "type": "OSError"}]
    assert "Private source" not in json.dumps(metadata)


def test_finalizer_failure_cannot_mask_original_generation_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_module.RawSSECapture, "finalize",
        lambda *args: (_ for _ in ()).throw(OSError("Private failure")))
    value = client([b"data: invalid\n\n"], tmp_path)
    with pytest.raises(ReliableGenerationError) as caught:
        request(value)
    assert caught.value.code == "stream_protocol_error"
    assert caught.value.receipt["raw_sse_capture"]["status"] == "failed"
    assert "Private failure" not in json.dumps(caught.value.receipt["raw_sse_capture"])


def test_capture_bound_is_explicit_and_does_not_raise_into_transport(tmp_path):
    observer = capture_module.RawSSECapture(tmp_path, max_bytes=10)
    observer.observe(b"x" * 25)
    receipt = {"call_id": "synthetic", "received_bytes": 25, "raw_sse_sha256": hashlib.sha256(b"x"*25).hexdigest(),
        "status": "success", "done_received": True, "usage_known": True}
    metadata = observer.finalize(receipt)
    assert Path(metadata["path"]).read_bytes() == b"x"*10
    assert metadata["bytes"] == metadata["limit_bytes"] == 10
    assert metadata["status"] == "failed" and metadata["complete"] is False
    assert metadata["errors"] == [{"operation": "observe", "type": "BufferError"}]
    assert observer.finalize(receipt) == metadata
    metadata["errors"].clear()
    assert observer.finalize(receipt)["errors"]


def test_parallel_requests_get_distinct_private_files_and_exact_content(tmp_path):
    def run(index):
        raw = complete({"content": f"Synthetic response {index}"})
        value = client([raw[:9], raw[9:]], tmp_path)
        result, receipt = request(value)
        captured, metadata = saved(receipt)
        assert captured == raw and result["choices"][0]["message"]["content"] == f"Synthetic response {index}"
        return metadata["path"], receipt["call_id"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(8)))
    assert len({path for path, _ in results}) == len({call_id for _, call_id in results}) == 8
    assert len(list(tmp_path.glob("*.sse"))) == 8


@pytest.mark.parametrize("bad", ["", b"bytes", 1, False])
def test_invalid_capture_path_is_rejected_at_configuration(bad):
    with pytest.raises((TypeError, ValueError)):
        client([], bad)


@pytest.mark.parametrize("bad", [None, 0, 1, "true", []])
def test_diagnostic_token_ids_requires_boolean_and_capture_directory(tmp_path, bad):
    with pytest.raises(ValueError, match="must be a boolean"):
        client([], tmp_path, diagnostic_token_ids=bad)
    with pytest.raises(ValueError, match="requires an explicit capture_directory"):
        client([], diagnostic_token_ids=True)


def test_diagnostic_token_ids_opt_in_is_idempotent_and_not_a_tokenize_field(tmp_path):
    from bridge_evidence.contest.document_context_budget import tokenize_chat_payload
    value = client([], tmp_path, diagnostic_token_ids=True)
    original = {"model": MODEL, "messages": [{"role": "user", "content": "Synthetic"}],
                "max_tokens": 1000, "chat_template_kwargs": {"enable_thinking": True}}
    prepared = value.reliable_payload(original)
    assert prepared["return_token_ids"] is True
    assert value.reliable_payload(prepared) == prepared
    assert "return_token_ids" not in original
    assert "return_token_ids" not in tokenize_chat_payload(prepared)
    default = client([])
    assert "return_token_ids" not in default.reliable_payload(original)


def test_token_ids_survive_raw_capture_without_changing_aggregated_content_or_tools(tmp_path):
    delta = {"content": "Synthetic response", "tool_calls": [{"index": 0,
        "id": "synthetic-call", "type": "function",
        "function": {"name": "synthetic_tool", "arguments": '{"id":1}'}}]}
    normal = complete(delta, finish="tool_calls")
    # Installed vLLM returns token IDs as response metadata, independently of
    # parser delta content. Preserve both metadata positions verbatim.
    obj = json.loads(event(delta).decode().split("data: ", 1)[1])
    obj["token_ids"] = [11, 12]
    obj["choices"][0]["token_ids"] = [21, 22]
    with_ids = ("data: " + json.dumps(obj) + "\r\n\r\n").encode() + normal[len(event(delta)):]
    opted = client([with_ids[:7], with_ids[7:]], tmp_path, diagnostic_token_ids=True)
    plain = client([normal])
    actual, receipt = request(opted)
    expected, _ = request(plain)
    raw, metadata = saved(receipt)
    assert raw == with_ids and b'"token_ids": [11, 12]' in raw
    assert actual == expected
    assert actual["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"id":1}'
    assert "token_ids" not in actual and "token_ids" not in actual["choices"][0]
    assert metadata["complete"] is True
    assert json.loads(opted._opener.calls[0][0].data)["return_token_ids"] is True
    assert "return_token_ids" not in json.loads(plain._opener.calls[0][0].data)
