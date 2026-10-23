"""Check or execute an explicit, exact-hash official-training document batch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .document_multiagent import MultiAgentSettings
from .document_multiagent_cli import add_execution_arguments, validate_execution_arguments, model_configuration
from .document_multiagent_batch import BatchSettings, prepare_batch, run_batch
from .run_state import atomic_json


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--batch", type=Path, required=True)
    cli.add_argument("--batch-sha256", required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Validate all input hashes/indices/plans without model calls")
    mode.add_argument("--execute", action="store_true")
    cli.add_argument("--document-workers", type=int, default=3)
    cli.add_argument("--global-main-model-inflight", type=int, default=2)
    cli.add_argument("--global-summary-model-inflight", type=int, default=1)
    add_execution_arguments(cli)
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    validate_execution_arguments(args, cli)
    try:
        settings = MultiAgentSettings(tasks=args.tasks, workers=args.workers, max_model_calls=args.max_model_calls,
            report_timeout=args.report_timeout, compression_trigger_tokens=args.compression_trigger_tokens,
            direct_delivery=args.direct_delivery, section_prompt_variant=args.section_prompt_variant, speed_priority=args.speed_priority)
        batch = BatchSettings(args.document_workers, args.global_main_model_inflight, args.global_summary_model_inflight)
        prepared = prepare_batch(args.batch, args.batch_sha256, settings=settings)
    except (OSError, ValueError, KeyError, TypeError) as error:
        cli.error(f"Batch preflight failed; no model dispatched: {error}")
    if args.check:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output_dir / "input-identity.json", prepared)
        print(json.dumps({"status": "inputs_validated", "documents": len(prepared["documents"]),
            "tasks": sum(len(doc["plan"]["tasks"]) for doc in prepared["documents"]), "model_calls": 0}))
        return 0
    model = model_configuration(args, cli)
    from .document_batch_quiescence import BatchServiceQuiescence
    result = run_batch(prepared, args.output_dir, model, settings=settings, batch_settings=batch,
                       quiescence_observer_factory=lambda directory: BatchServiceQuiescence(model, directory / "service-settlement"))
    print(json.dumps({key: result[key] for key in ("status", "seconds", "new_complete_reports",
        "reused_complete_reports", "new_complete_reports_per_hour")}, ensure_ascii=False))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
