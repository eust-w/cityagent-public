from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridge_evidence.audit import render_audit_markdown, run_submission_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit IAIC Agent traces, DOCX results, and submission package")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--reference-result-dir", type=Path)
    parser.add_argument("--require-online", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument(
        "--workflow-profile",
        choices=("full", "summary", "details"),
        default="full",
        help="Online workflow profile whose network evidence must be present.",
    )
    parser.add_argument(
        "--planner-mode",
        choices=("model", "deterministic"),
        default="model",
        help="Planner implementation expected in each state file.",
    )
    args = parser.parse_args()

    report = run_submission_audit(
        input_dir=args.input_dir,
        result_dir=args.result_dir,
        state_dir=args.state_dir,
        require_online=args.require_online,
        online_workflow_profile=args.workflow_profile,
        planner_mode=args.planner_mode,
        package_root=args.package_root,
        archive=args.archive,
        reference_result_dir=args.reference_result_dir,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_markdown.write_text(render_audit_markdown(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "errors": len(report["errors"])}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
