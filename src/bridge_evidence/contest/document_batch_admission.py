"""Local, batch-scoped generation admission shared by independent processes.

Deadlines are absolute ``time.monotonic()`` values on this host. A lease is
permission to send exactly one generation, not proof that it was sent/stopped.
Unknown requests retain occupancy and pause their pool. Only explicit settlement
or a trusted observer's later ``confirmed_idle: true`` can remove that barrier.
The broker never inspects PIDs, kills workers, or controls the model service.
"""
from __future__ import annotations

from copy import deepcopy
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4


class AdmissionError(RuntimeError):
    pass


class AdmissionDeadlineExceeded(AdmissionError):
    pass


class AdmissionUnavailable(AdmissionError):
    pass


class AdmissionBlocked(AdmissionError):
    """No verified drain is available; this generation was never admitted."""


class _AdmissionHTTPServer(ThreadingHTTPServer):
    # Set before TCPServer.server_activate() calls listen(). The stdlib default
    # of five is too small for simultaneous worker starts plus queued polling.
    request_queue_size = 128
    daemon_threads = True
    block_on_close = False


def _deadline(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("deadline must be a finite absolute monotonic timestamp")
    return float(value)


def _name(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{field} must be a nonempty string of at most 512 characters")
    return value


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


class BatchAdmissionBroker:
    """Own one localhost HTTP broker in the batch parent process.

    ``quiescence_observer(pool, deadline, snapshot)`` runs outside the lock in a
    daemon thread, only after this pool has no active requests. It must bound its
    own I/O by deadline and return a JSON dict with ``confirmed_idle is True``
    only for verified idle of the bound service. Ordinary truthy values, errors,
    and late responses never release leases. Its raw proof is journalled.
    """

    def __init__(self, directory, limits, document_ids, deadline, quiescence_observer=None,
                 *, observer_max_attempts=3, observer_retry_interval=1.0, observer_timeout=900.0,
                 settlement_deadline=None):
        self.directory = Path(directory)
        self.deadline = _deadline(deadline)
        self.settlement_deadline = _deadline(self.deadline if settlement_deadline is None else settlement_deadline)
        if self.settlement_deadline < self.deadline:
            raise ValueError("settlement_deadline cannot precede the generation deadline")
        if self.deadline <= time.monotonic():
            raise AdmissionDeadlineExceeded("Batch deadline already expired")
        if not isinstance(limits, dict) or not limits or any(
                not isinstance(k, str) or not k or type(v) is not int or v < 1 for k, v in limits.items()):
            raise ValueError("limits must map pool names to positive integer capacities")
        self.limits = dict(limits)
        self.document_ids = list(document_ids)
        if not self.document_ids or len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("document_ids must be a nonempty unique list")
        for value in self.document_ids:
            _name(value, "document_id")
        if quiescence_observer is not None and not callable(quiescence_observer):
            raise ValueError("quiescence_observer must be callable")
        if type(observer_max_attempts) is not int or observer_max_attempts < 1:
            raise ValueError("observer_max_attempts must be positive")
        if (not math.isfinite(observer_retry_interval) or not math.isfinite(observer_timeout)
                or observer_retry_interval < 0 or observer_timeout <= 0):
            raise ValueError("observer timing must be nonnegative interval and positive timeout")
        self.observer = quiescence_observer
        self.observer_max_attempts = observer_max_attempts
        self.observer_retry_interval = observer_retry_interval
        self.observer_timeout = observer_timeout
        self.batch_id = uuid4().hex
        self._tokens = {doc: secrets.token_urlsafe(32) for doc in self.document_ids}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._requests = {}
        self._stopped_documents = set()
        self._pools = {name: {"next_document": 0, "observing": False, "observer_attempts": 0,
                             "next_observation": 0.0, "recovery_deadline": None,
                             "observation_id": None} for name in self.limits}
        self._closed = False
        self._seq = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        # Exclusive create is also the restart guard: never forget prior unknown
        # requests by silently opening a fresh in-memory broker over an old run.
        self._journal = (self.directory / "admission-events.jsonl").open("x", encoding="utf-8")
        broker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.connection.settimeout(5)
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 262144:
                        raise ValueError("Invalid admission request size")
                    body = json.loads(self.rfile.read(size))
                    result = broker._dispatch(self.path, body)
                    status = 200
                except AdmissionDeadlineExceeded as exc:
                    result, status = {"error": str(exc), "error_type": "deadline"}, 408
                except (ValueError, TypeError, KeyError, AdmissionError) as exc:
                    result, status = {"error": str(exc), "error_type": "invalid"}, 400
                except Exception as exc:
                    result, status = {"error": type(exc).__name__, "error_type": "unavailable"}, 503
                data = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    # Loss of the HTTP client cannot prove an upstream stop.
                    pass

        try:
            self._server = _AdmissionHTTPServer(("127.0.0.1", 0), Handler)
            self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
            with self._lock:
                self._event("started", limits=self.limits, document_ids=self.document_ids,
                            deadline=self.deadline, settlement_deadline=self.settlement_deadline,
                            observer_enabled=self.observer is not None,
                            request_queue_size=self._server.request_queue_size)
            self._http_thread = threading.Thread(target=self._server.serve_forever,
                kwargs={"poll_interval": 0.05}, daemon=True, name="batch-admission-http")
            self._monitor_thread = threading.Thread(target=self._monitor, daemon=True,
                name="batch-admission-monitor")
            self._http_thread.start()
            self._monitor_thread.start()
        except BaseException:
            self._journal.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def config_for(self, document_id):
        if document_id not in self._tokens:
            raise ValueError("Document does not belong to this batch")
        return {"url": self.url, "batch_id": self.batch_id, "document_id": document_id,
                "auth_token": self._tokens[document_id], "deadline": self.deadline}

    def _event(self, kind, **values):
        self._seq += 1
        event = {"sequence": self._seq, "event": kind, "batch_id": self.batch_id,
                 "monotonic": time.monotonic(), "wall_time": time.time(), **_json_copy(values)}
        self._journal.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self._journal.flush()
        os.fsync(self._journal.fileno())

    def _pool_rows(self, pool, states):
        return [row for row in self._requests.values() if row["pool"] == pool and row["state"] in states]

    def _blocked_reason(self, name):
        if not self._pool_rows(name, {"unknown"}):
            return None
        pool = self._pools[name]
        if self.observer is None:
            return "quiescence_observer_unavailable"
        if time.monotonic() >= pool["recovery_deadline"]:
            return "quiescence_deadline_elapsed"
        if pool["observer_attempts"] >= self.observer_max_attempts and not pool["observing"]:
            return "quiescence_attempts_exhausted"
        return None

    def snapshot(self):
        with self._lock:
            pools = {}
            for name, capacity in self.limits.items():
                active = self._pool_rows(name, {"active"})
                unknown = self._pool_rows(name, {"unknown"})
                pending = self._pool_rows(name, {"pending"})
                pools[name] = {"capacity": capacity, "active": len(active), "unknown": len(unknown),
                    "pending": len(pending), "occupied": len(active) + len(unknown), "blocked": bool(unknown),
                    "blocked_reason": self._blocked_reason(name),
                    "recovery_deadline": self._pools[name]["recovery_deadline"],
                    "observing": self._pools[name]["observing"],
                    "observer_attempts": self._pools[name]["observer_attempts"]}
            return deepcopy({"batch_id": self.batch_id, "deadline": self.deadline,
                             "settlement_deadline": self.settlement_deadline, "closed": self._closed,
                             "sequence": self._seq, "pools": pools, "requests": list(self._requests.values()),
                             "stopped_documents": sorted(self._stopped_documents)})

    def _transition(self, row, state, evidence=None):
        previous = row["state"]
        first_unknown = state == "unknown" and not self._pool_rows(row["pool"], {"unknown"})
        row.update(state=state, updated_at=time.monotonic())
        if evidence is not None:
            row["evidence"] = _json_copy(evidence)
        if state == "active":
            row["granted_at"] = row["updated_at"]
            row["queue_wait_seconds"] = row["granted_at"] - row["enqueued_at"]
        if first_unknown:
            pool = self._pools[row["pool"]]
            pool["observer_attempts"] = 0
            pool["next_observation"] = 0.0
            # A bounded drain belongs to the batch, not the expired document.
            # Additional unknown leases never extend this recovery window.
            pool["recovery_deadline"] = min(self.settlement_deadline,
                                             row["updated_at"] + self.observer_timeout)
        self._event("transition", previous_state=previous, request=row)

    def _schedule(self):
        now = time.monotonic()
        for row in self._requests.values():
            if row["deadline"] <= now:
                if row["state"] == "pending":
                    self._transition(row, "expired", {"reason": "queue_deadline_elapsed", "not_sent": True})
                elif row["state"] == "active":
                    self._transition(row, "unknown", {"reason": "lease_deadline_elapsed", "stop_confirmed": False})
        for name in self.limits:
            pool = self._pools[name]
            if pool["observing"] and not self._pool_rows(name, {"unknown"}):
                # Late request-bound completion supersedes a slower idle probe.
                # Its eventual response must not affect subsequently admitted work.
                pool["observing"] = False
                pool["observation_id"] = None
                self._event("observation_superseded", pool=name, reason="all_unknown_requests_settled")
            reason = self._blocked_reason(name)
            if reason:
                for row in self._pool_rows(name, {"pending"}):
                    self._transition(row, "blocked", {"reason": reason, "not_sent": True,
                                                     "unknown_requests_retained": True})
        if self._closed or now >= self.deadline:
            return
        for name, capacity in self.limits.items():
            if self._pool_rows(name, {"unknown"}) or self._pools[name]["observing"]:
                continue
            pool = self._pools[name]
            while len(self._pool_rows(name, {"active"})) < capacity:
                chosen = None
                for offset in range(len(self.document_ids)):
                    index = (pool["next_document"] + offset) % len(self.document_ids)
                    doc = self.document_ids[index]
                    chosen = next((r for r in self._requests.values()
                        if r["pool"] == name and r["document_id"] == doc and r["state"] == "pending"
                        and r["deadline"] > time.monotonic()), None)
                    if chosen is not None:
                        pool["next_document"] = (index + 1) % len(self.document_ids)
                        break
                if chosen is None:
                    break
                self._transition(chosen, "active")

    def _dispatch(self, path, body):
        if not isinstance(body, dict):
            raise ValueError("Admission request must be an object")
        with self._lock:
            doc = body.get("document_id")
            if (body.get("batch_id") != self.batch_id or doc not in self._tokens
                    or not isinstance(body.get("auth_token"), str)
                    or not secrets.compare_digest(body["auth_token"], self._tokens[doc])):
                raise AdmissionError("Admission batch/document identity mismatch")
            if self._closed:
                raise AdmissionUnavailable("Admission broker is closed")
            if path == "/snapshot":
                return self.snapshot()
            if path == "/acquire":
                if doc in self._stopped_documents:
                    raise AdmissionError("Document worker has stopped; new admission is disabled")
                request_id = _name(body.get("request_id"), "request_id")
                pool = body.get("pool")
                if pool not in self.limits:
                    raise ValueError("Unknown admission pool")
                agent_id = _name(body.get("agent_id"), "agent_id")
                deadline = min(_deadline(body.get("deadline")), self.deadline)
                if request_id in self._requests:
                    row = self._requests[request_id]
                    if any(row[k] != v for k, v in {"document_id": doc, "pool": pool,
                            "agent_id": agent_id, "deadline": deadline}.items()):
                        raise AdmissionError("Repeated request ID changed its identity")
                else:
                    if deadline <= time.monotonic():
                        raise AdmissionDeadlineExceeded("Generation expired before admission")
                    row = {"request_id": request_id, "lease_id": uuid4().hex, "batch_id": self.batch_id,
                        "document_id": doc, "agent_id": agent_id, "pool": pool, "deadline": deadline,
                        "state": "pending", "enqueued_at": time.monotonic()}
                    self._requests[request_id] = row
                    self._event("enqueued", request=row)
                self._schedule()
                return deepcopy(row)
            row = self._requests.get(body.get("request_id"))
            if row is None or row["document_id"] != doc:
                raise AdmissionError("Unknown request or foreign document")
            if path == "/cancel_unsent":
                # Used only while acquire has not returned: this client has never
                # been given permission to send. Do not use for generation aborts.
                if row["state"] in {"pending", "active"}:
                    self._transition(row, "not_sent", {"reason": "acquire_did_not_return", "not_sent": True})
            elif path == "/settle":
                if body.get("lease_id") != row["lease_id"]:
                    raise AdmissionError("Lease identity mismatch")
                state, evidence = body.get("state"), body.get("evidence", {})
                if state not in {"complete", "not_sent", "unknown"} or not isinstance(evidence, dict):
                    raise ValueError("Settlement requires complete/not_sent/unknown and object evidence")
                evidence = _json_copy(evidence)
                self._settle(row, state, evidence)
            else:
                raise ValueError("Unknown admission operation")
            self._schedule()
            return deepcopy(row)

    def _settle(self, row, state, evidence):
        """Keep the original request association even after idle-proof fallback."""
        original_state = row.get("settlement_state")
        if original_state == state and row["settlement_evidence"] == evidence:
            return  # Idempotent even if later observation changed current state.
        if state == "complete" and row.get("completion_evidence") == evidence:
            return
        if row["state"] not in {"active", "unknown", "quiescent"}:
            if row["state"] == state:
                raise AdmissionError("Repeated settlement changed its evidence")
            raise AdmissionError(f"Cannot settle {row['state']} lease as {state}")
        if row["state"] in {"unknown", "quiescent"} and state == "not_sent":
            raise AdmissionError("Unknown generation cannot be reclassified as not sent")
        if original_state is not None and state != "complete":
            raise AdmissionError("Repeated settlement changed its evidence")
        if original_state is None:
            row["settlement_state"] = state
            row["settlement_evidence"] = deepcopy(evidence)
            row["settlement_recorded_at"] = time.monotonic()
            self._event("settlement_recorded", request=row)
        if state == "complete":
            row["completion_evidence"] = deepcopy(evidence)
        if row["state"] == "quiescent" and state == "unknown":
            # Deadline/parent failure may have preceded the client's callback.
            # Attach its true call ID without undoing already verified later idle.
            return
        if row["state"] != state:
            self._transition(row, state, evidence)

    def _monitor(self):
        while not self._stop.wait(0.02):
            with self._lock:
                if self._closed:
                    return
                self._schedule()
                for name, pool in self._pools.items():
                    unknown = self._pool_rows(name, {"unknown"})
                    if (pool["observing"] and pool["recovery_deadline"] is not None
                            and time.monotonic() >= pool["recovery_deadline"]):
                        pool["observing"] = False
                        pool["observation_id"] = None
                        self._event("observation_timed_out", pool=name,
                                    deadline=pool["recovery_deadline"], unknown_requests_retained=True)
                    if (not self.observer or not unknown or self._pool_rows(name, {"active"})
                            or pool["observing"] or pool["observer_attempts"] >= self.observer_max_attempts
                            or time.monotonic() < pool["next_observation"]):
                        continue
                    deadline = pool["recovery_deadline"]
                    if deadline <= time.monotonic():
                        continue
                    pool["observing"] = True
                    pool["observation_id"] = uuid4().hex
                    pool["observer_attempts"] += 1
                    self._event("observation_started", pool=name, deadline=deadline,
                                attempt=pool["observer_attempts"])
                    threading.Thread(target=self._observe,
                                     args=(name, deadline, self.snapshot(), pool["observation_id"]),
                                     daemon=True, name=f"batch-idle-{name}").start()

    def _observe(self, pool_name, deadline, snapshot, observation_id):
        try:
            evidence = _json_copy(self.observer(pool_name, deadline, snapshot))
        except Exception as exc:
            evidence = {"confirmed_idle": False, "error": type(exc).__name__, "message": str(exc)}
        with self._lock:
            if self._closed:
                return
            pool = self._pools[pool_name]
            if pool["observation_id"] != observation_id:
                return
            pool["observing"] = False
            pool["next_observation"] = time.monotonic() + self.observer_retry_interval
            confirmed = (isinstance(evidence, dict) and evidence.get("confirmed_idle") is True
                         and time.monotonic() < deadline and not self._pool_rows(pool_name, {"active"}))
            self._event("observation_finished", pool=pool_name, confirmed_idle=confirmed, evidence=evidence)
            if confirmed:
                for row in self._pool_rows(pool_name, {"unknown"}):
                    self._transition(row, "quiescent", {"later_service_idle": evidence,
                        "request_completion_known": False, "usage_known": False})
            self._schedule()

    def mark_document_unknown(self, document_id, evidence=None):
        """Parent reports worker failure; never treat its PID exit as a stop."""
        if document_id not in self.document_ids:
            raise ValueError("Document does not belong to this batch")
        if evidence is not None and not isinstance(evidence, dict):
            raise ValueError("Failure evidence must be an object")
        with self._lock:
            if self._closed:
                return self.snapshot()
            proof = {"reason": "document_worker_unavailable", "stop_confirmed": False,
                     "worker_evidence": _json_copy(evidence or {})}
            if document_id not in self._stopped_documents:
                self._stopped_documents.add(document_id)
                self._event("document_stopped", document_id=document_id, evidence=proof)
            for row in self._requests.values():
                if row["document_id"] == document_id:
                    if row["state"] == "active":
                        self._transition(row, "unknown", proof)
                    elif row["state"] == "pending":
                        self._transition(row, "not_sent", {"reason": "document_worker_unavailable_before_admission",
                                                           "not_sent": True})
            self._schedule()
            return self.snapshot()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            for row in self._requests.values():
                if row["state"] == "pending":
                    self._transition(row, "not_sent", {"reason": "broker_closed_before_admission", "not_sent": True})
                elif row["state"] == "active":
                    self._transition(row, "unknown", {"reason": "broker_closed_unsettled", "stop_confirmed": False})
            self._event("closed", snapshot=self.snapshot())
        self._server.shutdown()
        self._server.server_close()
        self._http_thread.join(timeout=1)
        self._monitor_thread.join(timeout=1)
        with self._lock:
            self._journal.close()


class AdmissionClient:
    """JSON-only configuration can be passed unchanged into spawned workers."""

    def __init__(self, config):
        self.config = _json_copy(config)
        if not self.config.get("url", "").startswith("http://127.0.0.1:"):
            raise ValueError("Admission broker must be on localhost")
        self._opener = build_opener(ProxyHandler({}))
        self.transport_events = []

    def _rpc(self, path, values, *, timeout=5):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("RPC timeout must be finite and positive")
        body = {**self.config, **values}
        encoded = json.dumps(body, allow_nan=False).encode()
        started = time.monotonic()
        deadline = started + timeout
        last_error = None
        attempts = 0
        # Only these local control operations have durable idempotent handlers.
        # This never retries model generation or creates a new request/lease ID.
        retryable = path in {"/acquire", "/settle", "/cancel_unsent", "/snapshot"}
        for attempt in range(1, 5 if retryable else 2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            attempts = attempt
            request = Request(self.config["url"] + path, data=encoded,
                              headers={"Content-Type": "application/json"}, method="POST")
            try:
                with self._opener.open(request, timeout=remaining) as response:
                    return json.load(response)
            except HTTPError as exc:
                # Identity, deadline, state, and journal errors are application
                # decisions, not transport failures. Never retry them blindly.
                try:
                    payload = json.load(exc)
                except Exception:
                    payload = {"error": f"Admission HTTP {exc.code}"}
                kind = AdmissionDeadlineExceeded if payload.get("error_type") == "deadline" else AdmissionError
                raise kind(payload["error"]) from exc
            except (OSError, URLError, HTTPException) as exc:
                last_error = exc
                self.transport_events.append({"path": path, "attempt": attempt,
                    "request_id": values.get("request_id"), "error_type": type(exc).__name__,
                    "elapsed_seconds": time.monotonic() - started})
                del self.transport_events[:-128]
                remaining = deadline - time.monotonic()
                if not retryable or attempt == 4 or remaining <= 0:
                    break
                time.sleep(min(0.01 * (2 ** (attempt - 1)), remaining))
        raise AdmissionUnavailable(f"Admission broker unavailable: path={path}, attempts={attempts}, "
            f"error={type(last_error).__name__}, elapsed={time.monotonic() - started:.3f}s") from last_error

    def acquire(self, pool, agent_id, deadline):
        deadline = min(_deadline(deadline), _deadline(self.config["deadline"]))
        request_id = uuid4().hex
        values = {"request_id": request_id, "pool": pool, "agent_id": agent_id, "deadline": deadline}
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AdmissionDeadlineExceeded("Generation deadline elapsed in admission queue")
                row = self._rpc("/acquire", values, timeout=min(5, remaining))
                if row["state"] == "active":
                    if time.monotonic() >= row["deadline"]:
                        raise AdmissionDeadlineExceeded("Generation expired before lease was received")
                    return row
                if row["state"] == "blocked":
                    raise AdmissionBlocked("Generation admission blocked: " + row.get("evidence", {}).get("reason", "unknown request"))
                if row["state"] != "pending":
                    raise AdmissionDeadlineExceeded(f"Request not admitted: {row['state']}")
                time.sleep(min(0.025, max(0, deadline - time.monotonic())))
        except BaseException:
            try:
                self._rpc("/cancel_unsent", {"request_id": request_id}, timeout=0.5)
            except Exception:
                # Server-side state remains durable; never guess it was freed.
                pass
            raise

    def settle(self, lease, state="complete", evidence=None):
        if (lease.get("batch_id") != self.config["batch_id"]
                or lease.get("document_id") != self.config["document_id"]):
            raise AdmissionError("Cannot settle a foreign lease")
        return self._rpc("/settle", {"request_id": lease["request_id"], "lease_id": lease["lease_id"],
                                     "state": state, "evidence": {} if evidence is None else evidence})

    def snapshot(self, *, timeout=5):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Snapshot timeout must be finite and positive")
        return self._rpc("/snapshot", {}, timeout=timeout)
