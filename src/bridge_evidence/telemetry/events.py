from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import structlog


class EventLogger:
    """Thread-safe structlog JSONL sink used by CLI and batch runs."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stream = self.path.open("a", encoding="utf-8")
        self._logger = structlog.wrap_logger(
            structlog.PrintLogger(self._stream),
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.JSONRenderer(
                    serializer=lambda value, **_: json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                ),
            ],
        )

    def emit(self, event: str, **payload: Any) -> None:
        with self._lock:
            self._logger.info(event, **payload)
            self._stream.flush()
