"""Explicit, hash-pinned official-training experiment; no directory discovery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .document_multiagent import MultiAgentSettings, run_multiagent
from .document_task_plan import build_document_task_plan
from .run_state import atomic_json


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--manifest-sha256", required=True, help="SHA256 of the exact input JSON file")
    result.add_argument("--scope", choices=["official-training"], required=True)
    result.add_argument("--materials", type=Path, help="Optional repaired MinerU materials JSON; omit for the legacy material path")
    result.add_argument("--materials-sha256", help="SHA256 of the exact prepared materials JSON file")
    result.add_argument("--materials-source-docx", type=Path, help="Original official-training DOCX for reference verification")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--execute", action="store_true", help="Otherwise only save the structural reading plan")
    add_execution_arguments(result)
    return result


def add_execution_arguments(result):
    result.add_argument("--ner-index", type=Path, help="Optional source-bound UIE candidate index; experimental material admission only")
    result.add_argument("--ner-index-sha256", help="SHA256 of the exact NER index JSON file")
    result.add_argument("--speed-priority", action="store_true", help="Experimental short workflow accepting recorded content gaps; requires direct-delivery")
    result.add_argument("--direct-delivery", action="store_true",
                        help="Submit section entries with handoff; seed merger with evidence and local review tools")
    result.add_argument("--base-url")
    result.add_argument("--model")
    result.add_argument("--api-key-env", default="CITYAGENT_5090_VLLM_API_KEY")
    result.add_argument("--summary-base-url")
    result.add_argument("--summary-model")
    result.add_argument("--summary-api-key-env", default="CITYAGENT_SUMMARY_VLLM_API_KEY")
    result.add_argument("--tasks", type=int, default=4)
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--max-model-calls", type=int, default=256)
    result.add_argument("--report-timeout", type=float, default=3420)
    result.add_argument("--thinking-tokens", type=int, default=4096)
    result.add_argument("--no-thinking", action="store_true",
                        help="Disable thinking explicitly for the main model and every section Agent; the summary model is unchanged")
    result.add_argument("--compression-trigger-tokens", type=int, default=196608)
    result.add_argument("--capture-raw-sse", action="store_true",
                        help="Save private raw response streams for transport/parser diagnosis")
    result.add_argument("--diagnostic-token-ids", action="store_true",
                        help="Request generated token IDs from a verified supporting server; requires raw SSE capture")
    result.add_argument("--raw-guard-tokenizer", type=Path,
                        help="Verified 27B ByteLevel BPE tokenizer.json for protection before server tool parsing")
    result.add_argument("--raw-guard-tokenizer-sha256",
                        help="SHA256 of the exact tokenizer file used by the raw generation guard")
    result.add_argument("--native-repetition-detection", action="store_true",
                        help="Enable the verified server token repetition detector (pattern 1..64, count 16)")
    from .document_section_prompts import SECTION_PROMPT_VARIANTS
    result.add_argument("--section-prompt-variant", default="scoped_v1", choices=SECTION_PROMPT_VARIANTS)


def prepare_training_input(manifest_path, manifest_sha256, *, materials_path=None,
                           materials_sha256=None, source_docx=None):
    """Pure preflight shared by single and batch entry points; no model calls."""
    raw = Path(manifest_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("Manifest file SHA256 differs; no planning or inference started")
    manifest = json.loads(raw)
    materials = material_index = None
    material_args = (materials_path, materials_sha256, source_docx)
    if any(material_args) and not all(material_args):
        raise ValueError("Provide --materials, --materials-sha256 and --materials-source-docx together")
    if materials_path:
        from .document_material_run import load_material_index
        material_raw = Path(materials_path).read_bytes()
        if hashlib.sha256(material_raw).hexdigest() != materials_sha256:
            raise ValueError("MinerU materials file SHA256 differs")
        materials = json.loads(material_raw)
        material_index = load_material_index(manifest, materials, source_docx)
    if manifest.get("images") or manifest.get("unverified_content"):
        raise ValueError("The initial experiment supports the explicitly prepared training text/table projection only")
    return manifest, materials, material_index


def validate_execution_arguments(args, cli):
    if bool(args.ner_index) != bool(args.ner_index_sha256):
        cli.error("Provide both --ner-index and --ner-index-sha256")
    if args.diagnostic_token_ids and not args.capture_raw_sse:
        cli.error("--diagnostic-token-ids requires --capture-raw-sse")
    if bool(args.raw_guard_tokenizer) != bool(args.raw_guard_tokenizer_sha256):
        cli.error("Provide both --raw-guard-tokenizer and --raw-guard-tokenizer-sha256")
    if args.raw_guard_tokenizer:
        if not args.capture_raw_sse:
            cli.error("--raw-guard-tokenizer requires --capture-raw-sse")
        try:
            tokenizer_raw = args.raw_guard_tokenizer.read_bytes()
        except OSError as exc:
            cli.error(f"Cannot read raw guard tokenizer: {exc}")
        if hashlib.sha256(tokenizer_raw).hexdigest() != args.raw_guard_tokenizer_sha256:
            cli.error("Raw guard tokenizer SHA256 differs; no planning or inference started")
        from .raw_generation_guard import RawGenerationGuardFactory
        try:
            RawGenerationGuardFactory(args.raw_guard_tokenizer, args.raw_guard_tokenizer_sha256)
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
            cli.error(f"Unsupported raw guard tokenizer; no inference started: {exc}")
    if args.thinking_tokens <= 0:
        cli.error("thinking-tokens must be positive; output remains dynamically allocated")


def model_configuration(args, cli):
    if not args.model or not args.base_url:
        cli.error("--execute requires --model and --base-url")
    if bool(args.summary_base_url) != bool(args.summary_model):
        cli.error("Provide both summary model and URL")
    model = {"model": args.model, "base_url": args.base_url,
             "api_key_env": args.api_key_env, "thinking_tokens": args.thinking_tokens,
             "enable_thinking": not args.no_thinking}
    if args.ner_index:
        model.update(ner_index_path=str(args.ner_index.resolve()), ner_index_sha256=args.ner_index_sha256)
    if args.capture_raw_sse:
        model["capture_raw_sse"] = True
    if args.diagnostic_token_ids:
        model["diagnostic_token_ids"] = True
    if args.raw_guard_tokenizer:
        model["raw_guard_tokenizer_path"] = str(args.raw_guard_tokenizer.resolve())
        model["raw_guard_tokenizer_sha256"] = args.raw_guard_tokenizer_sha256
    if args.native_repetition_detection:
        model["native_repetition_detection"] = True
    if args.summary_model:
        model["summary"] = {"model": args.summary_model, "base_url": args.summary_base_url,
                            "api_key_env": args.summary_api_key_env}
    return model


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    validate_execution_arguments(args, cli)
    try:
        manifest, materials, material_index = prepare_training_input(args.manifest, args.manifest_sha256,
            materials_path=args.materials, materials_sha256=args.materials_sha256,
            source_docx=args.materials_source_docx)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        cli.error(f"Invalid training input; no inference started: {exc}")
    settings = MultiAgentSettings(tasks=args.tasks, workers=args.workers,
        max_model_calls=args.max_model_calls, report_timeout=args.report_timeout,
        compression_trigger_tokens=args.compression_trigger_tokens, direct_delivery=args.direct_delivery,
        section_prompt_variant=args.section_prompt_variant, speed_priority=args.speed_priority)
    scope = {"scope": args.scope, "manifest_path": str(args.manifest.resolve()),
             "manifest_file_sha256": args.manifest_sha256, "scope_is_operator_assertion": True}
    if material_index is not None:
        scope.update(materials_path=str(args.materials.resolve()), materials_file_sha256=args.materials_sha256,
                     materials_source_docx=str(args.materials_source_docx.resolve()),
                     material_strategy="repaired_mineru_with_canonical_evidence")
    if not args.execute:
        plan = build_document_task_plan(manifest, target_tasks=settings.tasks, max_tasks=settings.tasks,
                                       **({"material_index": material_index} if material_index is not None else {}))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output_dir / "scope.json", scope)
        atomic_json(args.output_dir / "plan.json", plan)
        print(json.dumps({"status": "planned", "tasks": len(plan["tasks"]), "model_calls": 0}))
        return 0
    model = model_configuration(args, cli)
    material_options = ({"materials": materials, "materials_source_path": args.materials_source_docx}
                        if materials is not None else {})
    result = run_multiagent(manifest, args.output_dir, model, settings=settings, **material_options)
    atomic_json(args.output_dir / "scope.json", scope)
    print(json.dumps({key: result[key] for key in ("status", "complete", "seconds", "error")}, ensure_ascii=False))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
