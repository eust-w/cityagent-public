"""Reconstruct the bounded fixed-model pixel probe; never grant RSI admission."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

from .model_json import parse_json_object
from .run_state import file_digest, json_digest

PROBE_PROMPT = (
    "Look at the attached image. Identify the fill color of the rectangle on "
    "the left and the fill color of the rectangle on the right. "
    'Return exactly one JSON object with keys "left" and "right", whose values '
    "are lowercase English color names. Do not include explanations or markdown."
)
IMAGE_HASHES = (
    "3d3ff8c9fa73b82f698468d1aa226d7ca003ae509fb2ecfa367eb7b2210688f2",
    "919e8632a0c72efad726fce74c9ecce73ae6ca75df4ba08e697218ae7cc34450",
)


def _read(path, maximum=4 * 1024 * 1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("bounded regular capability artifact required")
    return path.read_bytes()


def validate_vision_capability(path, model_identity):
    """Bind both actual responses and their unique durable receipts to this model.

    A successful return authorizes only the experimental raster input path.
    It says nothing about report semantics, complete coverage or RSI promotion.
    """
    path = Path(path)
    try:
        proof = json.loads(_read(path))
        model_sha = json_digest(model_identity)
        alias = model_identity["manifest"]["served_model"]
        driver = Path(__file__).resolve().parents[3] / "scripts/probe_fixed_model_vision.py"
        if (proof.get("protocol") != "fixed-model-vision-capability-v1"
                or proof.get("model_identity") != model_identity
                or proof.get("model_identity_sha256") != model_sha
                or proof.get("served_model") != alias
                or proof.get("probe_source_sha256") != file_digest(driver)
                or proof.get("response_parser_sha256") != file_digest(Path(__file__).with_name("model_json.py"))
                or proof.get("prompt_sha256") != hashlib.sha256(PROBE_PROMPT.encode()).hexdigest()
                or proof.get("seed") != 17 or proof.get("temperature") != 0.0
                or proof.get("max_tokens") != 256
                or not isinstance(proof.get("calls"), list) or len(proof["calls"]) != 2):
            raise ValueError("capability protocol, model or probe identity changed")
        ledger_path = path.parent / "timing.sqlite"
        if ledger_path.is_symlink() or not ledger_path.is_file():
            raise ValueError("actual capability ledger required")
        with sqlite3.connect(ledger_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            ledger = {row[0]: row[1:] for row in db.execute(
                "SELECT id,role,seconds,status,receipt FROM calls")}
        ids = [call["call_id"] for call in proof["calls"]]
        if len(set(ids)) != 2 or set(ids) != set(ledger):
            raise ValueError("capability call IDs differ from unique actual ledger")
        backend_ids = set()
        for index, call in enumerate(proof["calls"]):
            if not re.fullmatch(r"[a-f0-9]{32}", call["call_id"]):
                raise ValueError("invalid capability call identity")
            if call.get("mock") or call.get("simulated"):
                raise ValueError("simulated capability is not model evidence")
            pixels = _read(path.parent / f"synthetic-{index}.png", 65536)
            image_sha = hashlib.sha256(pixels).hexdigest()
            if image_sha != IMAGE_HASHES[index] or call.get("image_sha256") != image_sha:
                raise ValueError("capability pixels differ from fixed contrast pair")
            payload = {
                "model": alias, "temperature": 0.0, "seed": 17,
                "max_tokens": 256, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": PROBE_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(pixels).decode()}},
                ]}],
            }
            if json_digest(payload) != call.get("request_sha256"):
                raise ValueError("capability request does not bind actual pixels/template")
            response = json.loads(_read(path.parent / f"response-{call['call_id']}.json"))
            if json_digest(response) != call.get("response_sha256"):
                raise ValueError("capability raw response changed")
            usage, choices = response["usage"], response["choices"]
            if (response.get("model") != alias or len(choices) != 1
                    or choices[0]["index"] != 0 or choices[0]["finish_reason"] != "stop"
                    or choices[0]["message"]["role"] != "assistant"
                    or not isinstance(response.get("id"), str) or not response["id"]
                    or response["id"] in backend_ids
                    or any(type(usage[k]) is not int or usage[k] <= 0
                           for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
                    or usage["completion_tokens"] > 256
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
                raise ValueError("capability actual model response or usage invalid")
            expected = {"left": "red", "right": "blue"} if index == 0 else {"left": "blue", "right": "red"}
            if parse_json_object(choices[0]["message"]["content"]) != expected:
                raise ValueError("capability did not follow both actual image inputs")
            backend_ids.add(response["id"])
            role, seconds, status, encoded = ledger[call["call_id"]]
            if (role != "synthetic-vision-probe" or status != "success"
                    or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
                    or call.get("seconds") != seconds or call.get("status") != "success"
                    or call.get("backend_request_id") != response["id"]
                    or call.get("usage") != usage or call.get("finish_reason") != "stop"
                    or call.get("content") != choices[0]["message"]["content"]
                    or json.loads(encoded) != call):
                raise ValueError("capability ledger and actual response disagree")
        return {"protocol": proof["protocol"], "probe_sha256": file_digest(path),
                "model_identity_sha256": model_sha, "probe_source_sha256": file_digest(driver),
                "ledger_sha256": file_digest(ledger_path), "mode": "same-model",
                "scope": "synthetic raster capability; not report quality or RSI admission"}
    except (OSError, KeyError, TypeError, sqlite3.Error, json.JSONDecodeError) as error:
        raise ValueError("invalid fixed-model visual capability evidence") from error
