import base64
import hashlib
import io
import json

import pytest
from PIL import Image

from bridge_evidence.contest.backends import AccountedBackend, APIBackend
from bridge_evidence.contest.run_state import json_digest
from bridge_evidence.contest.timing import TimingLedger


def request():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, format="PNG")
    blob = buffer.getvalue()
    return {"kind": "visual-evidence-v1", "job_id": "a" * 64,
            "image": {"mime_type": "image/png", "sha256": hashlib.sha256(blob).hexdigest(),
                      "base64": base64.b64encode(blob).decode()},
            "instruction": "Transcribe the visible image; report unreadable regions."}


def response(**changes):
    return {"model": "fixed-alias", "id": "vision-1",
            "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102},
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "{}"}}], **changes}


def test_visual_request_uses_same_alias_and_actual_usage(monkeypatch, tmp_path):
    api = APIBackend("http://127.0.0.1:8000/v1", "fixed-alias", None)
    sent = []
    monkeypatch.setattr(api, "_request", lambda route, value: sent.append((route, value)) or response())
    ledger = TimingLedger(tmp_path / "timing.sqlite")
    value = request()
    actual = AccountedBackend(api, ledger)(value, 512)
    route, body = sent[0]
    assert route == "chat/completions"
    assert body["model"] == "fixed-alias"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["seed"] == 17 and body["temperature"] == 0
    assert body["messages"][0]["content"][1]["image_url"]["url"].endswith(value["image"]["base64"])
    assert actual["prompt_tokens"] == 100
    with ledger.connect() as db:
        role, receipt = db.execute("SELECT role,receipt FROM calls").fetchone()
    assert role == "visual-evidence"
    assert json.loads(receipt)["prompt_sha256"] == json_digest(value)


@pytest.mark.parametrize("change", [
    lambda r: r.update(kind="arbitrary"),
    lambda r: r.update(path="/dataset/labels.json"),
    lambda r: r["image"].update(sha256="b" * 64),
    lambda r: r["image"].update(base64="https://example.com/image.png"),
    lambda r: r["image"].update(mime_type="image/jpeg"),
    lambda r: r["image"].update(url="file:///model/weights"),
    lambda r: r.update(instruction=""),
])
def test_invalid_visual_request_never_reaches_model(monkeypatch, change):
    api = APIBackend("http://127.0.0.1:8000/v1", "fixed-alias", None)
    monkeypatch.setattr(api, "_request", lambda *a: pytest.fail("invalid image reached server"))
    value = request()
    change(value)
    with pytest.raises(ValueError):
        api(value, 512)


@pytest.mark.parametrize("bad", [response(model="different"), response(id=""),
    response(usage={"prompt_tokens": 0, "completion_tokens": 2, "total_tokens": 2}),
    response(choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": None}}])])
def test_invalid_visual_response_retains_failed_ledger(monkeypatch, tmp_path, bad):
    api = APIBackend("http://127.0.0.1:8000/v1", "fixed-alias", None)
    monkeypatch.setattr(api, "_request", lambda *a: bad)
    ledger = TimingLedger(tmp_path / "timing.sqlite")
    with pytest.raises(ValueError):
        AccountedBackend(api, ledger)(request(), 512)
    assert ledger.summary()["failures"] == 1
