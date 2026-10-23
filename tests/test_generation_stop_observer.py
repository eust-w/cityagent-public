import json
from types import SimpleNamespace

import pytest

from bridge_evidence.contest.generation_stop_observer import GenerationStopObserver, generation_stopped


class Clock:
    now = 0.0
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


def observer(tmp_path, values, *, process=None, model="base"):
    clock = Clock()
    client = SimpleNamespace(base_url="http://127.0.0.1:18110", model="own",
        api_key="test", max_model_len=262144, timeout=900)
    values = iter(values)
    last = 1
    def fetch(url, **kwargs):
        nonlocal last
        if url.endswith("/models"):
            return 200, json.dumps({"data": [{"id": "own", "parent": "base", "root": "/weights"},
                {"id": "base", "parent": None, "max_model_len": 262144}]}).encode()
        last = next(values, last)
        return 200, (f'vllm:num_requests_running{{model_name="{model}",engine="0"}} {last}\n'
                     f'vllm:num_requests_waiting{{model_name="{model}",engine="0"}} 0\n').encode()
    result = GenerationStopObserver(client, tmp_path, deadline=100, server_pid=123,
        fetch=fetch, clock=clock, sleep=clock.sleep,
        process_identity=process or (lambda *_: {"pid": 123, "starttime_ticks": "123", "cmdline_sha256": "abc"}))
    return result, clock


def receipt(**kwargs):
    return {"call_id": "local-call", "response_id": "chatcmpl-exact", "connection_closed": True,
            "detected_after_seconds": 42, "wall_seconds_limit": 900, **kwargs}


def test_observed_idle_does_not_claim_abort_ack_or_usage(tmp_path):
    callback, clock = observer(tmp_path, [1, 0, 0])
    original = receipt()
    result = callback(original, None)
    assert result["status"] == "not_confirmed"
    assert result["stop_observation"]["seconds"] == 1
    assert generation_stopped({**original, "stop_observation": result["stop_observation"]})
    assert result["stop_observation"]["cancellation_confirmed"] is False
    assert "usage" not in result and "done_received" not in result
    assert len(list(tmp_path.rglob("observation.json"))) == 1
    assert not generation_stopped({**original, "response_id": "another", "stop_observation": result["stop_observation"]})


@pytest.mark.parametrize("values", [[1], [0, 1], [1, 0, 1]])
def test_busy_or_single_idle_cannot_authorize_recovery(tmp_path, values):
    callback, clock = observer(tmp_path, values)
    result = callback(receipt(), None)
    assert not generation_stopped({**receipt(), "stop_observation": result["stop_observation"]})
    assert clock.now <= 10


def test_process_replacement_is_not_stopped_request(tmp_path):
    counter = 0
    def process(*_):
        nonlocal counter
        counter += 1
        return {"pid": 123, "starttime_ticks": str(counter)}
    callback, _ = observer(tmp_path, [0], process=process)
    result = callback(receipt(), None)
    assert not result["stop_observation"]["stopped_observed"]
    assert "process changed" in result["stop_observation"]["reason"]


def test_wrong_model_gauges_are_rejected(tmp_path):
    callback, _ = observer(tmp_path, [0], model="wrong")
    result = callback(receipt(), None)
    assert not result["stop_observation"]["stopped_observed"]


@pytest.mark.parametrize("change", [{"response_id": None}, {"connection_closed": False},
                                    {"detected_after_seconds": 900}])
def test_missing_identity_or_expired_request_no_retry(tmp_path, change):
    callback, clock = observer(tmp_path, [0])
    result = callback(receipt(**change), None)
    assert not result["stop_observation"]["stopped_observed"]
    assert clock.now == 0


def native_receipt(**overrides):
    return receipt(http_status=200, error_code="native_repetition", finish_reason="stop",
        native_stop_observation={"response_id": "chatcmpl-exact", "choice_index": 0,
            "stop_reason": "repetition_detected", "finish_reason": "stop",
            "terminal_observed": True, "method": "native_terminal_event"}, **overrides)


def test_native_response_terminal_proof_is_request_specific_while_peer_still_runs(tmp_path):
    callback, clock = observer(tmp_path, [1])
    callback.fetch = lambda *args, **kwargs: pytest.fail("Known native stop must not wait for unrelated requests")
    stopped = callback(native_receipt(), None)
    assert stopped["stop_observation"]["method"] == "native_terminal_event"
    assert stopped["stop_observation"]["samples"] == [] and clock.now == 0
    value = {**native_receipt(), "stop_observation": stopped["stop_observation"]}
    assert generation_stopped(value)
    assert "cancel_status" not in value  # Native end is not a client abort ACK.


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(error_code="reasoning_repetition"),
    lambda r: r.update(http_status=500),
    lambda r: r.update(response_id="other-request"),
    lambda r: r.update(finish_reason=None),
    lambda r: r["native_stop_observation"].update(choice_index=1),
    lambda r: r["native_stop_observation"].update(choice_index=False),
    lambda r: r["native_stop_observation"].update(terminal_observed=False),
    lambda r: r["native_stop_observation"].update(stop_reason=None),
    lambda r: r["native_stop_observation"].update(finish_reason="length"),
    lambda r: r["native_stop_observation"].update(method="configured_repetition_detection"),
])
def test_native_configuration_or_wrong_response_does_not_authorize_recovery(mutation):
    value = native_receipt()
    mutation(value)
    assert not generation_stopped(value)
