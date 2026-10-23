"""Prepare one explicit official training DOCX in a separate offline process."""
import argparse
import json
from pathlib import Path

from .prepare_document import PreparationError, prepare_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="One official training .docx, never a test-set file")
    parser.add_argument("output", type=Path, help="New or empty output directory")
    parser.add_argument("--training-root", required=True, type=Path, help="Explicit authorized official training source root")
    parser.add_argument("--python", dest="python_executable", help="Existing MinerU 4.0.12 / DocVortex 0.5.16 Python executable")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--character-budget", type=int, default=6000)
    args = parser.parse_args()
    try:
        result = prepare_document(args.source, args.output, training_root=args.training_root,
            python_executable=args.python_executable, expected_sha256=args.expected_sha256,
            timeout=args.timeout, character_budget=args.character_budget)
    except (PreparationError, ValueError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
