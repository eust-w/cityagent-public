#!/usr/bin/env python3
"""Render audited DocumentState JSON files back to canonical DOCX results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridge_evidence.models.schema import DocumentState
from bridge_evidence.rendering import DocxReportRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="批量重渲染 IAIC 状态文件")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    states = sorted(args.state_dir.glob("*.json"))
    if not states:
        raise SystemExit("状态目录没有 JSON 文件")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = DocxReportRenderer()
    rendered = 0
    for path in states:
        state = DocumentState.model_validate_json(path.read_text(encoding="utf-8"))
        # Source inputs may be legacy .doc files, while the competition result
        # contract always requires a .docx output with the same stem.
        output_name = Path(state.source_path).with_suffix(".docx").name
        renderer.render(state, args.output_dir / output_name)
        rendered += 1
    print(json.dumps({"states": len(states), "rendered": rendered}, ensure_ascii=False))


if __name__ == "__main__":
    main()
