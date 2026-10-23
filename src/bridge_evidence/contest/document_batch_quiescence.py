"""Read-only, service-wide admission proof after a batch pool becomes unknown.

Later scheduler idle can release broker capacity. It does not establish any
earlier request's cancellation, completion, DONE event, or token usage.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from ipaddress import ip_address
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from .document_generation_settlement import _fetch, _gauges, _model_identity
from .generation_stop_observer import _process_identity
from .run_state import atomic_json, json_digest


_PID_ENV = {"main": "CITYAGENT_STOP_OBSERVER_SERVER_PID",
            "summary": "CITYAGENT_SUMMARY_STOP_OBSERVER_SERVER_PID"}
_KEY_ENV = {"main": "CITYAGENT_5090_VLLM_API_KEY",
            "summary": "CITYAGENT_SUMMARY_VLLM_API_KEY"}


class BatchServiceQuiescence:
    """Pin existing processes once, then observe only metadata GET endpoints.

    The injected fetch must honor its timeout, like the shared default fetch.
    All hooks exist for local deterministic tests; no model client is created.
    """

    def __init__(self, model_config, directory, *, fetch=_fetch,
                 clock=time.monotonic, sleep=time.sleep, process_identity=_process_identity):
        self.directory = Path(directory)
        self.fetch, self.clock, self.sleep = fetch, clock, sleep
        self.process_identity = process_identity
        self._profiles = {}
        for pool in _PID_ENV:
            pinned = {"pool": pool, "pid_environment": _PID_ENV[pool]}
            try:
                profile = model_config if pool == "main" else model_config.get("summary")
                if not isinstance(profile, dict):
                    raise ValueError("Missing service profile")
                url = urlsplit(profile["base_url"])
                if (url.scheme not in {"http", "https"} or not url.netloc
                        or url.username or url.password or url.query or url.fragment
                        or url.path.rstrip("/") not in {"", "/v1"}):
                    raise ValueError("Expected the configured service origin or /v1 URL")
                # A local /proc PID cannot attest the process behind a remote URL.
                if url.hostname != "localhost" and not ip_address(url.hostname).is_loopback:
                    raise ValueError("Local service PID proof requires a loopback origin")
                port = url.port or (443 if url.scheme == "https" else 80)
                model = profile["model"]
                if not isinstance(model, str) or not model:
                    raise ValueError("Missing served model name")
                context = profile.get("max_model_len", 262144)
                if type(context) is not int or context != 262144:
                    raise ValueError("Expected context capacity 262144")
                pid_text = os.environ.get(_PID_ENV[pool], "")
                if not pid_text.isdecimal() or int(pid_text) <= 0:
                    raise ValueError("Explicit verified service PID is required")
                pid = int(pid_text)
                identity = deepcopy(process_identity(pid, port))
                if not isinstance(identity, dict) or identity.get("pid") != pid:
                    raise ValueError("Pinned process identity does not match configured PID")
                pinned.update(origin=urlunsplit((url.scheme, url.netloc, "", "", "")),
                    port=port, model=model, server_pid=pid, process_identity=identity,
                    process_identity_sha256=json_digest(identity),
                    api_key=os.environ.get(profile.get("api_key_env", _KEY_ENV[pool]), "local-cityagent"))
            except Exception as exc:
                pinned.update(pin_error=str(exc), pin_error_type=type(exc).__name__)
            self._profiles[pool] = pinned

    def __call__(self, pool, deadline, snapshot):
        started = self.clock()
        directory = self.directory / uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        proof = {"schema": "cityagent-batch-service-quiescence-v1", "pool": pool,
            "confirmed_idle": False, "method": "later_whole_service_idle",
            "cancellation_confirmed": False, "prior_requests_reclassified": False,
            "request_completion_known": False, "usage_known": False,
            "historical_done_and_usage": "Unchanged; missing values remain unknown.",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_monotonic": started, "samples": [], "same_service_process": False}
        try:
            if type(deadline) not in (int, float) or not math.isfinite(deadline):
                raise ValueError("Invalid shared monotonic deadline")
            limit = min(deadline, started + 10)
            proof.update(shared_deadline_monotonic=deadline, observation_deadline_monotonic=limit)
            atomic_json(directory / "broker-snapshot.json", snapshot)
            proof["snapshot_ref"] = self._file_ref(directory / "broker-snapshot.json")
            rows = snapshot.get("requests")
            state = snapshot.get("pools", {}).get(pool, {})
            if (not isinstance(rows, list) or type(state.get("active")) is not int
                    or state["active"] != 0 or type(state.get("unknown")) is not int
                    or state["unknown"] <= 0
                    or any(row.get("pool") == pool and row.get("state") == "active" for row in rows)
                    or sum(row.get("pool") == pool and row.get("state") == "unknown"
                           for row in rows) != state["unknown"]):
                raise ValueError("Pool must have unknown requests and no active requests")
            profile = self._profiles.get(pool)
            if profile is not None:
                proof["service"] = {key: deepcopy(value) for key, value in profile.items() if key != "api_key"}
            if profile is None or profile.get("pin_error"):
                raise ValueError((profile or {}).get("pin_error", "Unknown model pool"))
            identity_hash = engine_hash = None
            consecutive = 0
            while self.clock() < limit:
                self._same_process(profile)
                sample = {"started_after_seconds": self.clock() - started, "idle": False}
                proof["samples"].append(sample)
                raw_bodies = {}
                for endpoint in ("/v1/models", "/metrics"):
                    timeout = min(2, limit - self.clock())
                    if timeout <= 0:
                        raise TimeoutError("Service observation deadline exceeded")
                    status, raw = self.fetch(profile["origin"] + endpoint, timeout=timeout,
                        headers={"Authorization": "Bearer " + profile["api_key"]})
                    # Keep the exact response before parsing or rejecting it.
                    name = f"sample-{len(proof['samples']):02d}-{endpoint.rsplit('/', 1)[-1]}.txt"
                    (directory / name).write_bytes(raw)
                    sample[endpoint] = {"http_status": status, **self._file_ref(directory / name)}
                    if self.clock() >= limit:
                        raise TimeoutError("Metadata response arrived after observation deadline")
                    if status != 200:
                        raise ValueError("Service observation HTTP failure")
                    raw_bodies[endpoint] = raw
                identity = _model_identity(raw_bodies["/v1/models"], profile["model"], None, 262144)
                current_hash = json_digest(identity)
                if identity_hash is not None and current_hash != identity_hash:
                    raise ValueError("Serving model identity changed")
                gauges, idle = _gauges(raw_bodies["/metrics"])
                if any(row["labels"].get("model_name") != identity["base"]["id"]
                       for values in gauges.values() for row in values):
                    raise ValueError("Scheduler gauges belong to another model")
                current_engines = json_digest(sorted(json_digest(row["labels"])
                    for row in next(iter(gauges.values()))))
                if engine_hash is not None and current_engines != engine_hash:
                    raise ValueError("Serving engine identities changed")
                self._same_process(profile)
                if self.clock() >= limit:
                    raise TimeoutError("Service observation deadline exceeded")
                identity_hash, engine_hash = current_hash, current_engines
                consecutive = consecutive + 1 if idle else 0
                sample.update(idle=idle, gauges=gauges, model_identity=identity,
                    completed_after_seconds=self.clock() - started, consecutive_idle=consecutive)
                if consecutive >= 2:
                    if sample["started_after_seconds"] - proof["samples"][-2]["completed_after_seconds"] < .5:
                        raise ValueError("Idle samples must be separated by at least 0.5 seconds")
                    proof.update(confirmed_idle=True, same_service_process=True,
                        model_identity_sha256=identity_hash, engine_identity_sha256=engine_hash,
                        process_identity_sha256=profile["process_identity_sha256"])
                    break
                if limit - self.clock() <= .5:
                    break
                self.sleep(.5)
            if not proof["confirmed_idle"]:
                proof["reason"] = "No two consecutive idle observations before the bounded deadline"
        except Exception as exc:
            proof.update(confirmed_idle=False, reason=str(exc), error_type=type(exc).__name__)
        proof.update(seconds=max(0, self.clock() - started),
                     finished_at_utc=datetime.now(timezone.utc).isoformat())
        if ("observation_deadline_monotonic" in proof
                and self.clock() >= proof["observation_deadline_monotonic"]):
            proof.update(confirmed_idle=False, reason="Service observation deadline exceeded")
        atomic_json(directory / "proof.json", proof)
        # Disk persistence is part of admission proof, never publish a late success.
        if proof["confirmed_idle"] and self.clock() >= proof["observation_deadline_monotonic"]:
            proof.update(confirmed_idle=False, reason="Proof persisted after observation deadline")
            atomic_json(directory / "proof.json", proof)
        return {"confirmed_idle": proof["confirmed_idle"], "proof": proof,
                "proof_ref": self._file_ref(directory / "proof.json")}

    def _same_process(self, profile):
        if self.process_identity(profile["server_pid"], profile["port"]) != profile["process_identity"]:
            raise ValueError("Serving process changed since batch construction")

    @staticmethod
    def _file_ref(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
