"""Receipt reconstruction fixtures; these do not establish actual model capability."""
import base64
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import pytest

from bridge_evidence.contest.run_state import atomic_json, file_digest, json_digest
from bridge_evidence.contest.timing import TimingLedger
from bridge_evidence.contest.vision_capability import validate_vision_capability


@pytest.fixture
def evidence(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts/probe_fixed_model_vision.py"
    spec = importlib.util.spec_from_file_location("capability_probe_fixture", source)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    model = {"manifest": {"served_model": "test-fixed-model"}, "files": {"base": {"weights": "f" * 64}}}
    alias = model["manifest"]["served_model"]
    ledger = TimingLedger(tmp_path / "timing.sqlite")
    proof = {"protocol": "fixed-model-vision-capability-v1", "model_identity": model,
             "model_identity_sha256": json_digest(model), "served_model": alias,
             "probe_source_sha256": file_digest(source), "seed": 17, "temperature": 0.0,
             "response_parser_sha256": file_digest(source.parents[1] / "src/bridge_evidence/contest/model_json.py"),
             "max_tokens": 256, "prompt_sha256": probe.hashlib.sha256(probe.PROMPT.encode()).hexdigest(),
             "passed": True, "calls": []}
    for i in range(2):
        pixels = probe.synthetic_png(swapped=bool(i))
        (tmp_path / f"synthetic-{i}.png").write_bytes(pixels)
        payload = {"model": alias, "temperature": 0.0, "seed": 17, "max_tokens": 256,
                   "stream": False, "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "user", "content": [{"type": "text", "text": probe.PROMPT},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(pixels).decode()}}]}]}
        answer = {"left": "blue", "right": "red"} if i else {"left": "red", "right": "blue"}
        content = json.dumps(answer)
        response = {"id": "test-vision-" + str(i), "model": alias,
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}
        call_id = uuid4().hex
        call = {"call_id": call_id, "image_sha256": file_digest(tmp_path / f"synthetic-{i}.png"),
                "request_sha256": json_digest(payload), "response_sha256": json_digest(response),
                "backend_request_id": response["id"], "usage": response["usage"],
                "finish_reason": "stop", "content": content, "seconds": 0.5, "status": "success"}
        ledger.claim(call_id, role="synthetic-vision-probe")
        ledger.finish(call_id, 0.5, status="success", receipt=call)
        proof["calls"].append(call)
        atomic_json(tmp_path / f"response-{call_id}.json", response)
    path = tmp_path / "vision-probe.json"
    atomic_json(path, proof)
    return path, model, proof, ledger


def test_capability_reconstructs_receipts_not_passed_claim(evidence):
    path, model, proof, _ = evidence
    proof["passed"] = False
    atomic_json(path, proof)
    assert validate_vision_capability(path, model)["mode"] == "same-model"
    assert "RSI admission" in validate_vision_capability(path, model)["scope"]


@pytest.mark.parametrize("mutation", ["model", "raw", "pixels", "ledger", "duplicate", "driver", "parser", "request", "mock"])
def test_forged_or_stale_capability_rejected(evidence, mutation):
    path, model, proof, ledger = evidence
    if mutation == "model":
        model = {**model, "files": {"base": {"weights": "changed"}}}
    elif mutation == "raw":
        p = path.parent / f"response-{proof['calls'][1]['call_id']}.json"
        raw = json.loads(p.read_text()); raw["choices"][0]["message"]["content"] = proof["calls"][0]["content"]
        atomic_json(p, raw)
    elif mutation == "pixels":
        (path.parent / "synthetic-1.png").write_bytes((path.parent / "synthetic-0.png").read_bytes())
    elif mutation == "ledger":
        with ledger.connect() as db:
            db.execute("UPDATE calls SET seconds=0.01")
    elif mutation == "duplicate":
        proof["calls"][1] = proof["calls"][0]
    elif mutation == "driver":
        proof["probe_source_sha256"] = "a" * 64
    elif mutation == "parser":
        proof["response_parser_sha256"] = "a" * 64
    elif mutation == "request":
        proof["calls"][0]["request_sha256"] = "b" * 64
    else:
        proof["calls"][0]["mock"] = True
    atomic_json(path, proof)
    with pytest.raises(ValueError):
        validate_vision_capability(path, model)
