from __future__ import annotations

import argparse
from pathlib import Path

from bridge_evidence.adapters import FormatAdapter
from bridge_evidence.config import Settings
from bridge_evidence.document_ai import (
    ConsensusResolver,
    JsonCommandExpert,
    MultiExpertDocumentPipeline,
    NativeOoxmlExpert,
)
from bridge_evidence.parsers import DocxStructureParser


def _expert_mapping(values: list[str]) -> dict[str, str]:
    experts: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--expert must use NAME=COMMAND format")
        name, command = value.split("=", 1)
        if not name.strip() or not command.strip():
            raise ValueError("--expert name and command must not be empty")
        experts[name.strip()] = command.strip()
    return experts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auditable multi-expert DocumentIR teacher records")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path(".bridge_evidence/teacher-work"))
    parser.add_argument("--expert", action="append", default=[], help="NAME=COMMAND containing {input}")
    parser.add_argument("--min-agreement", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--min-margin", type=float, default=0.15)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--continue-on-expert-error", action="store_true")
    args = parser.parse_args()

    commands = _expert_mapping(args.expert)
    settings = Settings(
        llm_mode="off",
        work_dir=args.work_dir,
        cache_dir=args.work_dir / "cache",
        document_expert_commands=commands,
        document_expert_timeout_seconds=args.timeout_seconds,
        document_consensus_min_agreement=args.min_agreement,
        document_consensus_min_confidence=args.min_confidence,
        document_consensus_min_margin=args.min_margin,
        document_expert_fail_closed=not args.continue_on_expert_error,
    )
    adapter = FormatAdapter(settings.cache_dir)
    structure_parser = DocxStructureParser()
    experts = [NativeOoxmlExpert(), *[
        JsonCommandExpert(name, command, timeout_seconds=args.timeout_seconds)
        for name, command in sorted(commands.items())
    ]]
    pipeline = MultiExpertDocumentPipeline(
        experts,
        ConsensusResolver(
            min_agreement=args.min_agreement,
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
        ),
        mode="audit",
        fail_closed=not args.continue_on_expert_error,
    )

    files = sorted(
        path
        for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".doc", ".docx"}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for source_path in files:
            adapted = adapter.adapt(source_path)
            parsed = structure_parser.parse(adapted.docx_path)
            result = pipeline.clean(parsed, adapted.docx_path, adapted.sha256)
            stream.write(result.ir.model_dump_json())
            stream.write("\n")


if __name__ == "__main__":
    main()
