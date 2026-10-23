from __future__ import annotations

import json
from pathlib import Path

from bridge_evidence.models.schema import DocumentState


def write_json_result(state: DocumentState, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
