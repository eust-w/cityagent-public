"""Bounded, read-only observation after a cancelled HTTP stream.

This is not an abort acknowledgement. Two idle scheduler observations on the
same pinned service establish later quiescence; DONE and usage remain unknown.
No retry is authorized here. The gateway separately limits recovery to once.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from .document_generation_settlement import _fetch, _gauges, _model_identity
from .run_state import atomic_json


def _native_terminal_observed(receipt: dict) -> bool:
    native = receipt.get("native_stop_observation") or {}
    # A terminal repetition event identifies this response even while another
    # section is still generating. This is a server stop, not a client abort
    # acknowledgement, and must come from the matching SSE response.
    return bool(receipt.get("error_code") in {"native_repetition", "raw_repetition"}
            and receipt.get("http_status") == 200
            and receipt.get("response_id")
            and native.get("response_id") == receipt["response_id"]
            and native.get("method") == "native_terminal_event"
            and type(native.get("choice_index")) is int
            and native["choice_index"] == 0
            and native.get("terminal_observed") is True
            and native.get("stop_reason") == "repetition_detected"
            and native.get("finish_reason") in {"stop", "tool_calls", "repetition"}
            and receipt.get("finish_reason") == native["finish_reason"])


def generation_stopped(receipt: dict) -> bool:
    if _native_terminal_observed(receipt):
        return True
    if receipt.get("cancel_status") == "cancellation_confirmed":
        return True
    observed = receipt.get("stop_observation") or {}
    samples = observed.get("samples") or []
    return bool(receipt.get("response_id") and
        observed.get("response_id") == receipt["response_id"] and
        observed.get("stopped_observed") is True and
        observed.get("method") == "service_idle_after_disconnect" and
        observed.get("same_service_process") is True and
        observed.get("cancellation_confirmed") is False and
        len(samples) >= 2 and all(s.get("idle") is True for s in samples[-2:]) and
        samples[-1].get("after_disconnect_seconds", 0) - samples[-2].get("after_disconnect_seconds", 0) >= .5 and
        observed.get("model_identity_sha256") and observed.get("process_identity_sha256"))


def _process_identity(pid: int, port: int) -> dict:
    root = Path("/proc") / str(pid)
    command = (root / "cmdline").read_bytes()
    argv = command.decode().split("\0")
    if ("vllm.entrypoints.openai.api_server" not in argv or
            "--port" not in argv or argv[argv.index("--port") + 1] != str(port)):
        raise ValueError("Stop observer process does not own the configured service profile")
    return {"pid": pid, "starttime_ticks": (root / "stat").read_text().rsplit(")", 1)[1].split()[19],
            "cmdline_sha256": hashlib.sha256(command).hexdigest()}


class GenerationStopObserver:
    def __init__(self, client, directory, *, deadline, server_pid=None,
                 fetch=_fetch, clock=time.monotonic, sleep=time.sleep,
                 process_identity=_process_identity, max_wait_seconds=10):
        self.client, self.directory, self.deadline = client, Path(directory), deadline
        self.fetch, self.clock, self.sleep = fetch, clock, sleep
        self.process_identity = process_identity
        self.server_pid = server_pid
        self.port = urlsplit(client.base_url).port or 80
        self.max_wait_seconds = min(10, max_wait_seconds)
        self.pinned_process = None
        self.pin_error = None
        try:
            if type(server_pid) is not int or server_pid <= 0:
                raise ValueError("Explicit verified service PID is required")
            self.pinned_process = process_identity(server_pid, self.port)
        except (OSError, ValueError, IndexError) as exc:
            self.pin_error = type(exc).__name__

    def __call__(self, receipt, response):
        start = self.clock()
        if _native_terminal_observed(receipt):
            # The server already ended this exact response. Waiting for all
            # other sections to become idle adds no proof and delays recovery.
            observed = {"method": "native_terminal_event", "response_id": receipt["response_id"],
                "stopped_observed": True, "cancellation_confirmed": False,
                "request_specific_abort_ack_available": False, "samples": [], "seconds": 0,
                "meaning": "Matching server repetition terminal event; no client abort acknowledgement or scheduler poll."}
            directory = self.directory / hashlib.sha256(str(receipt.get("call_id")).encode()).hexdigest()
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(directory / "observation.json", observed)
            return {"status": "not_confirmed", "stop_observation": observed}
        # LocalModelClient.timeout is reduced to the enclosing request's actual
        # remaining limit before dispatch. Do not extend it while observing.
        remaining_request = max(0, min(self.client.timeout, receipt.get("wall_seconds_limit", 900))
                                - receipt.get("detected_after_seconds", 0))
        limit = min(self.deadline, start + remaining_request, start + self.max_wait_seconds)
        observed = {"method": "service_idle_after_disconnect", "response_id": receipt.get("response_id"),
            "stopped_observed": False, "cancellation_confirmed": False,
            "same_service_process": False, "samples": [], "started_at": datetime.now(timezone.utc).isoformat(),
            "request_specific_abort_ack_available": False,
            "meaning": "Later whole-service idle observation only; original completion and usage remain unknown."}
        directory = self.directory / hashlib.sha256(str(receipt.get("call_id")).encode()).hexdigest()
        directory.mkdir(parents=True, exist_ok=False)
        try:
            if self.pinned_process is None or not receipt.get("response_id") or receipt.get("connection_closed") is not True:
                raise ValueError("Missing pinned service identity, response ID or closed stream")
            identity_hash = None
            consecutive = 0
            while self.clock() < limit:
                if self.process_identity(self.server_pid, self.port) != self.pinned_process:
                    raise ValueError("Serving process changed")
                sample = {"after_disconnect_seconds": self.clock() - start}
                raws = {}
                for path in ("/v1/models", "/metrics"):
                    timeout = min(2, limit - self.clock())
                    if timeout <= 0:
                        raise TimeoutError("Stop observation deadline")
                    status, raw = self.fetch(self.client.base_url + path, timeout=timeout,
                        headers={"Authorization": "Bearer " + self.client.api_key})
                    if status != 200:
                        raise ValueError("Stop observation HTTP failure")
                    raws[path] = raw
                identity = _model_identity(raws["/v1/models"], self.client.model, None, self.client.max_model_len)
                current_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                if identity_hash is not None and current_hash != identity_hash:
                    raise ValueError("Serving model identity changed")
                identity_hash = current_hash
                gauges, idle = _gauges(raws["/metrics"])
                # Ensure the advertised base model covers the observed engines.
                if any(row["labels"].get("model_name") != identity["base"]["id"] for rows in gauges.values() for row in rows):
                    raise ValueError("Scheduler gauges belong to another model")
                if self.process_identity(self.server_pid, self.port) != self.pinned_process:
                    raise ValueError("Serving process changed during observation")
                sample.update(idle=idle, gauges=gauges, after_disconnect_seconds=self.clock() - start)
                for path, raw in raws.items():
                    name = f"{len(observed['samples'])}-{path.rsplit('/', 1)[-1]}.txt"
                    (directory / name).write_bytes(raw)
                    sample[path] = {"path": str(directory / name), "sha256": hashlib.sha256(raw).hexdigest()}
                observed["samples"].append(sample)
                consecutive = consecutive + 1 if idle else 0
                if consecutive >= 2:
                    observed.update(stopped_observed=True, same_service_process=True,
                        model_identity_sha256=identity_hash,
                        process_identity_sha256=hashlib.sha256(json.dumps(self.pinned_process, sort_keys=True).encode()).hexdigest())
                    break
                if limit - self.clock() < .5:
                    break
                self.sleep(.5)
            if not observed["stopped_observed"]:
                observed["reason"] = "No two consecutive idle observations before the bounded deadline"
        except Exception as exc:
            observed.update(reason=str(exc), error_type=type(exc).__name__)
        observed["seconds"] = max(0, self.clock() - start)
        observed["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(directory / "observation.json", observed)
        return {"status": "not_confirmed", "stop_observation": observed}


def configured_stop_observer(client, directory, deadline):
    """An absent verification PID disables recovery, never guesses ownership."""
    value = os.environ.get("CITYAGENT_STOP_OBSERVER_SERVER_PID", "")
    return GenerationStopObserver(client, directory, deadline=deadline,
        server_pid=int(value) if value.isdecimal() else None)
