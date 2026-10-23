"""Optional bounded evidence of consumed SSE bytes, never request headers.

The observer buffers bytes only. Disk I/O runs after transport finalization so
capture failures cannot become stream failures or trigger backend cancellation.
A process killed before finalization has no durable capture guarantee.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import threading
from uuid import uuid4


MAX_CAPTURE_BYTES = 32 * 1024 * 1024


def capture_directory(value):
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
        raise ValueError("capture_directory must be a nonempty filesystem path or None")
    return Path(value).expanduser().absolute()


class RawSSECapture:
    def __init__(self, directory, *, max_bytes=MAX_CAPTURE_BYTES):
        self.directory = capture_directory(directory)
        if self.directory is None or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Raw SSE capture needs a directory and positive byte limit")
        self.limit = min(max_bytes, MAX_CAPTURE_BYTES)
        self.capture_id = uuid4().hex
        self.path = self.directory / (self.capture_id + ".sse")
        self._buffer = bytearray()
        self._observed_bytes = 0
        self._observed_hash = hashlib.sha256()
        self._errors = []
        self._lock = threading.Lock()
        self._final = None

    def _error(self, operation, exc):
        if len(self._errors) < 8:
            self._errors.append({"operation": operation, "type": type(exc).__name__})

    def observe(self, raw):
        """Never raise into stream_chat, including on allocation failure."""
        with self._lock:
            try:
                if self._final is not None:
                    raise RuntimeError("Capture already finalized")
                if type(raw) is not bytes:
                    raise TypeError("Capture requires original bytes")
                self._observed_bytes += len(raw)
                self._observed_hash.update(raw)
                available = self.limit - len(self._buffer)
                if available > 0:
                    self._buffer.extend(raw[:available])
                if len(raw) > available:
                    raise BufferError("Capture byte limit")
            except Exception as exc:
                self._error("observe", exc)

    def finalize(self, stream_receipt):
        """Persist raw bytes and return metadata; preserve every stream outcome."""
        with self._lock:
            if self._final is not None:
                return deepcopy(self._final)
            fd = None
            created = False
            written = 0
            persisted_hash = hashlib.sha256()
            try:
                self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(self.path, flags, 0o600)
                created = True
                view = memoryview(self._buffer)
                try:
                    while written < len(view):
                        count = os.write(fd, view[written:])
                        if type(count) is not int or count <= 0:
                            raise OSError("Raw capture write did not progress")
                        persisted_hash.update(view[written:written + count])
                        written += count
                    os.fsync(fd)
                finally:
                    view.release()
            except Exception as exc:
                self._error("persist", exc)
            finally:
                if fd is not None:
                    try:
                        os.close(fd)
                    except Exception as exc:
                        self._error("close", exc)

            # Hash the actual saved bytes after close, not a synthesized JSON
            # response or just the memory buffer. Verification is also bounded.
            actual_hash = None
            actual_bytes = None
            if created:
                try:
                    digest = hashlib.sha256()
                    actual_bytes = 0
                    with self.path.open("rb") as saved:
                        while True:
                            raw = saved.read(min(1024 * 1024, self.limit + 1 - actual_bytes))
                            if not raw:
                                break
                            actual_bytes += len(raw)
                            digest.update(raw)
                            if actual_bytes > self.limit:
                                raise OSError("Raw capture verification exceeds bound")
                    actual_hash = digest.hexdigest()
                    if actual_bytes != written or actual_hash != persisted_hash.hexdigest():
                        raise OSError("Raw capture changed during finalization")
                except Exception as exc:
                    self._error("verify", exc)

            received = stream_receipt.get("received_bytes")
            stream_hash = stream_receipt.get("raw_sse_sha256")
            matched = (actual_bytes == received == self._observed_bytes
                       and actual_hash == stream_hash == self._observed_hash.hexdigest())
            complete = (not self._errors and matched and stream_receipt.get("status") == "success"
                        and stream_receipt.get("done_received") is True
                        and stream_receipt.get("usage_known") is True)
            status = ("failed" if self._errors else "not_started" if received == 0
                      else "complete" if complete else "partial")
            self._final = {"schema": "cityagent-raw-sse-capture-v1", "capture_id": self.capture_id,
                "call_id": stream_receipt.get("call_id"), "scope": "consumer_observed_bytes",
                "path": str(self.path) if created else None, "attempted_path": str(self.path),
                "sha256": actual_hash, "bytes": actual_bytes, "limit_bytes": self.limit,
                "observer_bytes": self._observed_bytes, "received_bytes": received,
                "observed_all_received_bytes": matched, "status": status, "complete": complete,
                "stream_status": stream_receipt.get("status"),
                "done_received": stream_receipt.get("done_received") is True,
                "usage_known": stream_receipt.get("usage_known") is True,
                "http_status": stream_receipt.get("http_status"), "errors": deepcopy(self._errors),
                "http_body_eof_proven": False, "durable_before_finalize": False}
            self._buffer.clear()
            return deepcopy(self._final)
