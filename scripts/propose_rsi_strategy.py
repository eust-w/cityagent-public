#!/usr/bin/env python3
"""Generate a bounded Python patch in an explicit idle experiment, then replay and shadow."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.backends import (
    AccountedBackend,
    APIBackend,
    ModelManifest,
    normalize_api_url,
)
from bridge_evidence.contest.long_document import (
    ChunkTask,
    DocumentSnapshot,
    SourceSlice,
)
from bridge_evidence.contest.rsi import StrategyBundle
from bridge_evidence.contest.rsi_experiments import (
    propose_patch,
    replay,
    search_history,
    shadow,
    write_ready,
)
from bridge_evidence.contest.rsi_sandbox import verify_root
from bridge_evidence.contest.run_state import atomic_json, file_digest
from bridge_evidence.contest.timing import TimingLedger


def create_backend(args, model, tokenizer):
    return APIBackend(args.url, model.served_model, tokenizer, timeout=120)


def build_parser():
    parser = argparse.ArgumentParser()
    for name in (
        "parent-bundle",
        "model-manifest",
        "sandbox-root",
        "records-dir",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reports-manifest", type=Path)
    parser.add_argument("--search-history", type=Path)
    parser.add_argument(
        "--target",
        choices=("choose_action", "propose_candidates", "update_search"),
        default="choose_action",
    )
    parser.add_argument("--url", type=normalize_api_url, default="http://127.0.0.1:8000/v1")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not str(args.output_dir.resolve()).startswith("/workspace/"):
        parser.error("official /workspace required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    verify_root(args.sandbox_root)
    parent = StrategyBundle.load(args.parent_bundle)
    model = ModelManifest.load(args.model_manifest).validate()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model.tokenizer, local_files_only=True, trust_remote_code=True
    )
    backend = create_backend(args, model, tokenizer)
    generate = AccountedBackend(backend, TimingLedger(args.output_dir / "timing.sqlite"))
    paths = sorted(args.records_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)[
        -5:
    ]
    records = [json.loads(p.read_text()) for p in paths]
    if not records:
        raise ValueError("actual generation-only history required")
    candidate, calls = propose_patch(
        parent,
        records,
        tokenizer,
        generate,
        target=args.target,
        outcomes=search_history(json.loads(args.search_history.read_text()))
        if args.search_history
        else (),
    )
    evidence = replay(candidate, args.sandbox_root, records=records)
    atomic_json(args.output_dir / "candidate.bundle.json", candidate.as_dict())
    atomic_json(
        args.output_dir / "code-evidence.json",
        {
            "url": backend.url,
            "target": args.target,
            "parent_sha256": parent.digest,
            "candidate_sha256": candidate.digest,
            "calls": calls,
            "replay": evidence,
            "optimizer_verified": False,
            "model_manifest_sha256": file_digest(args.model_manifest),
        },
    )
    if args.target == "choose_action":
        record = next(
            (
                r
                for r in reversed(records)
                if len(r.get("chunk_plan", {}).get("tasks", [])) >= 2
            ),
            None,
        )
        if record is None:
            if not args.reports_manifest:
                raise ValueError(
                    "approved source manifest required for whole-report shadow"
                )
            from bridge_evidence.contest.chunked_inference import _instruction
            from bridge_evidence.contest.long_document import (
                plan_chunks,
                split_task,
            )

            source_view = [
                {"report": v["report"], "sha256": v["report_sha256"]}
                for v in json.loads(args.reports_manifest.read_text())["samples"]
            ]
            completed = {
                r["document"]["sha256"] for r in records if r.get("status") == "success"
            }
            source = next(v for v in source_view if v["sha256"] in completed)
            report_path = Path(source["report"])
            if file_digest(report_path) != source["sha256"]:
                raise ValueError("approved shadow report changed")
            if report_path.suffix.lower() == ".doc":
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "contest_entry", ROOT / "run.py"
                )
                entry = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(entry)
                report_path = entry.convert_doc_to_docx(
                    report_path, args.output_dir / "conversion"
                )
            snapshot = DocumentSnapshot.from_docx(report_path)
            plan = plan_chunks(
                snapshot,
                tokenizer,
                input_budget=parent.config["input_budget"],
                output_budget=parent.config["output_budget"],
                overlap=parent.config["overlap"],
                instruction=_instruction(parent.config),
            )
            tasks = plan.tasks
            if len(tasks) == 1:
                tasks = split_task(tasks[0], snapshot)
            shadow_calls = shadow(
                parent,
                candidate,
                snapshot,
                tasks,
                tokenizer,
                generate,
                sandbox_root=args.sandbox_root,
            )
            write_ready(
                args.output_dir / "ready",
                candidate,
                calls,
                evidence,
                shadow_calls,
                ledger_path=generate.ledger.path,
            )
            print(
                json.dumps(
                    {
                        "candidate_sha256": candidate.digest,
                        "target": args.target,
                        "verified": False,
                    }
                )
            )
            return
        from bridge_evidence.contest.long_document import SourceRegion

        snap = record["snapshot"]
        snapshot = DocumentSnapshot(
            snap["document_sha256"],
            [SourceRegion(**r) for r in snap["regions"]],
            snap["parser"],
        )
        if snapshot.digest != record["snapshot_sha256"]:
            raise ValueError("shadow source snapshot changed")
        tasks = [
            ChunkTask(
                t["task_id"],
                [SourceSlice(**s) for s in t["owned"]],
                [SourceSlice(**s) for s in t["context"]],
                t["prompt_tokens"],
            )
            for t in record["chunk_plan"]["tasks"]
        ]
        shadow_calls = shadow(
            parent,
            candidate,
            snapshot,
            tasks,
            tokenizer,
            generate,
            sandbox_root=args.sandbox_root,
        )
        write_ready(
            args.output_dir / "ready",
            candidate,
            calls,
            evidence,
            shadow_calls,
            ledger_path=generate.ledger.path,
        )
    print(
        json.dumps(
            {
                "candidate_sha256": candidate.digest,
                "target": args.target,
                "verified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
