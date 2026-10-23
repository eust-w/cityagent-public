from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridge_evidence.evaluation.teacher import compare_teacher_target


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a DocumentState JSON with a teacher trace")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--scope",
        choices=("all", "current-summary"),
        default="all",
        help="current-summary excludes previous-history fields and benchmark-convention conflicts",
    )
    args = parser.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    teacher = json.loads(args.teacher.read_text(encoding="utf-8"))
    if args.scope == "current-summary":
        teacher["targets"] = {
            field: target
            for field, target in teacher.get("targets", {}).items()
            if not field.startswith("scores.previous.")
            and target.get("status") != "benchmark_convention_conflict"
        }
    result = compare_teacher_target(state, teacher)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if not result["differences"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
