#!/usr/bin/env python3
"""Build and audit the IAIC competition submission package.

Creates a standard package root with `code/`, `design/`, `result/`,
audits all contents for secrets and correctness, and produces both
.tar.gz and .zip archives.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUNTIME_FILES = (
    "run.py",
    "requirements-contest.txt",
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "scripts/merge_lora_v3.py",
    "src/bridge_evidence/__init__.py",
    "src/bridge_evidence/domain_rules.py",
    "src/bridge_evidence/evaluation/__init__.py",
    "src/bridge_evidence/evaluation/generation.py",
    "src/bridge_evidence/contest/__init__.py",
    "src/bridge_evidence/contest/text_extract.py",
    "src/bridge_evidence/contest/docx_writer.py",
    "src/bridge_evidence/contest/table_facts.py",
    "src/bridge_evidence/contest/facts.py",
    "src/bridge_evidence/contest/semantic_diagnostics.py",
    "src/bridge_evidence/contest/model_identity.py",
    "src/bridge_evidence/contest/temporal_router.py",
    "src/bridge_evidence/contest/defect_chunker.py",
    "src/bridge_evidence/contest/constraint_cleaner.py",
    "src/bridge_evidence/contest/inference_core.py",
    "src/bridge_evidence/contest/model_preserving_fusion.py",
    "src/bridge_evidence/contest/tong_workflow.py",
    "src/bridge_evidence/contest/execution_audit.py",
    "src/bridge_evidence/contest/run_state.py",
    "src/bridge_evidence/contest/long_document.py",
    "src/bridge_evidence/contest/chunked_inference.py",
    "src/bridge_evidence/contest/backends.py",
    "src/bridge_evidence/contest/candidate.py",
    "src/bridge_evidence/contest/timing.py",
    "src/bridge_evidence/contest/rsi.py",
    "src/bridge_evidence/contest/rsi_runtime.py",
    "src/bridge_evidence/contest/rsi_pipeline.py",
    "src/bridge_evidence/contest/rsi_sandbox.py",
    "src/bridge_evidence/contest/rsi_worker.py",
    "src/bridge_evidence/contest/rsi_namespace.py",
    "src/bridge_evidence/contest/rsi_evolution.py",
    "src/bridge_evidence/contest/rsi_experiments.py",
    "src/bridge_evidence/contest/rsi_acceptance.py",
    "src/bridge_evidence/contest/sdk_acceptance.py",
    "src/bridge_evidence/contest/artifact_publish.py",
    "src/bridge_evidence/contest/divide_cli.py",
    "src/bridge_evidence/contest/divide_conquer_workflow.py",
    "src/bridge_evidence/contest/divide_budget.py",
    "src/bridge_evidence/contest/training_evidence_batch.py",
    "src/bridge_evidence/contest/document_cli.py",
    "src/bridge_evidence/contest/document_delivery_validation.py",
    "src/bridge_evidence/contest/document_model_gateway.py",
    "src/bridge_evidence/contest/document_multimodal_admission.py",
    # Resolved by filename and executed as the native admission subprocess.
    "src/bridge_evidence/contest/document_native_processor.py",
    "src/bridge_evidence/contest/document_reading_recovery.py",
    "src/bridge_evidence/contest/document_review_checkpoint.py",
    "src/bridge_evidence/contest/document_review_journal.py",
    "src/bridge_evidence/contest/document_review_lookup.py",
    "src/bridge_evidence/contest/document_review_payloads.py",
    "src/bridge_evidence/contest/document_source_objects.py",
    "src/bridge_evidence/contest/document_tools.py",
    "src/bridge_evidence/contest/document_vision_backend.py",
    "src/bridge_evidence/contest/document_image_representation.py",
    "src/bridge_evidence/contest/document_vision_runtime.py",
    "src/bridge_evidence/contest/document_workflow.py",
    "src/bridge_evidence/contest/document_workspace.py",
    "src/bridge_evidence/contest/document_workspace_tools.py",
    "src/bridge_evidence/contest/source_object_catalog.py",
    "src/bridge_evidence/contest/source_object_dispositions.py",
    "src/bridge_evidence/contest/source_object_review.py",
    "src/bridge_evidence/contest/evidence_manifest.py",
    "src/bridge_evidence/contest/local_model_client.py",
    "src/bridge_evidence/contest/schema_reducer.py",
    "src/bridge_evidence/contest/tong_react_adapter.py",
)


def copy_code(src_root: Path, dst_code: Path) -> None:
    """只复制推理所需文件；模型、训练代码、缓存和旧压缩包均留在提交目录外。"""
    missing = [name for name in RUNTIME_FILES if not (src_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少比赛运行文件：{', '.join(missing)}")
    if dst_code.exists() and any(dst_code.iterdir()):
        raise ValueError(f"代码目标目录必须为空，避免残留权重或旧文件：{dst_code}")
    dst_code.mkdir(parents=True, exist_ok=True)
    # This is the same protected Python namespace bound by build_run_identity.
    # Include nested packages and subprocess workers, without copying reports,
    # experiment scripts, caches, or arbitrary files from the source tree.
    protected_contest = sorted(
        p.relative_to(src_root).as_posix()
        for p in (src_root / "src/bridge_evidence/contest").rglob("*.py")
    )
    audit_tools = [
        name
        for name in (
            "scripts/evaluate_multitask.py",
            "scripts/benchmark_rsi_recovery.py",
            # validate_vision_capability binds this exact helper's source hash.
            "scripts/probe_fixed_model_vision.py",
        )
        if (src_root / name).is_file()
    ]
    selected = tuple(dict.fromkeys([*RUNTIME_FILES, *protected_contest, *audit_tools]))
    for item in selected:
        relative = Path(item)
        if any(src_root.joinpath(*relative.parts[:index]).is_symlink()
               for index in range(1, len(relative.parts) + 1)):
            raise ValueError(f"运行源码不能通过符号链接引用其他内容：{item}")
    for item in selected:
        src = src_root / item
        dst = dst_code / item
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_design(design_file: Path, dst_design: Path) -> None:
    if (
        design_file is None
        or not Path(design_file).is_file()
        or Path(design_file).is_symlink()
    ):
        raise ValueError("必须通过 --design-file 明确指定唯一最终方案书")
    if Path(design_file).suffix.lower() not in (".docx", ".pdf", ".md"):
        raise ValueError("方案书必须为 docx/pdf/md")
    dst_design.mkdir(parents=True, exist_ok=False)
    shutil.copy2(design_file, dst_design / Path(design_file).name)


def copy_result(result_src: Path, dst_result: Path) -> None:
    """官方格式：result/ = 与测试集同名的 doc 文件 + infer_time.json"""
    dst_result.mkdir(parents=True, exist_ok=True)
    for docx in sorted(result_src.glob("*.docx")):
        shutil.copy2(docx, dst_result / docx.name)
    infer_time = result_src / "infer_time.json"
    if infer_time.exists():
        shutil.copy2(infer_time, dst_result / infer_time.name)


def build_tar_gz(package_root: Path, output_tar: Path) -> None:
    with tarfile.open(output_tar, "x:gz") as tar:
        for top_dir in ["code", "design", "result"]:
            dir_path = package_root / top_dir
            if dir_path.exists():
                tar.add(dir_path, arcname=top_dir)


def build_zip(package_root: Path, output_zip: Path) -> None:
    with zipfile.ZipFile(output_zip, "x", zipfile.ZIP_DEFLATED) as zf:
        for top_dir in ["code", "design", "result"]:
            dir_path = package_root / top_dir
            if not dir_path.exists():
                continue
            zf.write(dir_path, arcname=top_dir + "/")
            for file_path in sorted(dir_path.rglob("*")):
                if file_path.is_file():
                    arcname = file_path.relative_to(package_root).as_posix()
                    zf.write(file_path, arcname=arcname)


def audit_run_manifest(
    src_root: Path, package_root: Path, records_dir: Path, expected_run_fingerprint: str
) -> tuple[dict, list[str]]:
    """Bind the packaged source to the exact model/config/code identity used to run."""
    from bridge_evidence.contest.run_state import file_digest, json_digest

    errors = []
    manifest_path = records_dir.parent / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = manifest["identity"]
        if (
            type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
            or manifest.get("run_fingerprint") != expected_run_fingerprint
            or json_digest(identity) != expected_run_fingerprint
        ):
            errors.append("run_manifest: fingerprint/identity mismatch")
        if not isinstance(identity, dict) or not identity.get("model_files"):
            errors.append("run_manifest: model content hashes are missing")
        options = identity.get("options", {})
        if options.get("rsi_mode") == "experimental-online":
            errors.append("run_manifest: mixed experimental RSI cannot be submitted")
        if options.get("rsi_mode") == "frozen":
            from bridge_evidence.contest.rsi_acceptance import verify_release

            release = options.get("rsi_release")
            if not release:
                errors.append(
                    "run_manifest: frozen RSI lacks independently qualified release"
                )
            else:
                from bridge_evidence.contest.run_state import file_digest

                if file_digest(Path(release["path"])) != release["sha256"]:
                    errors.append("run_manifest: release evidence changed")
                verify_release(
                    release["path"], release["bundle_sha256"], source_root=src_root
                )
        bundle = options.get("rsi_bundle")
        if bundle:
            from bridge_evidence.contest.rsi import StrategyBundle
            from bridge_evidence.contest.run_state import atomic_json

            strategy = StrategyBundle(
                bundle["source"],
                bundle["config"],
                bundle["parent_sha256"],
                bundle["abi"],
            )
            atomic_json(
                package_root / "code" / "frozen-rsi-bundle.json", strategy.as_dict()
            )
        code_files = identity.get("code_files") if isinstance(identity, dict) else None
        if not isinstance(code_files, dict) or not code_files:
            raise ValueError("missing source code hashes")
        for name in RUNTIME_FILES:
            if name not in code_files:
                errors.append(
                    f"run_manifest: runtime file has no recorded hash: {name}"
                )
        for name, digest in code_files.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                errors.append("run_manifest: unsafe source path")
                continue
            source = src_root / relative
            if source.is_symlink() or file_digest(source) != digest:
                errors.append(f"run_manifest: source changed since inference: {name}")
            packaged = package_root / "code" / relative
            if packaged.exists() and (
                packaged.is_symlink() or file_digest(packaged) != digest
            ):
                errors.append(f"run_manifest: packaged source hash mismatch: {name}")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"run_manifest: missing/invalid identity: {exc}")
    return {
        "path": str(manifest_path),
        "run_fingerprint": expected_run_fingerprint,
        "passed": not errors,
    }, errors


def package_submission(
    *,
    src_root: Path,
    input_dir: Path,
    result_dir: Path,
    state_dir: Path,
    output_dir: Path,
    run_fingerprint: str,
    audit_profile: str = "contest-sdk",
    design_file: Path | None = None,
    smoke_request: Path | None = None,
) -> dict:
    """Create a fresh, reviewable staging directory; publish only after all audits."""
    from bridge_evidence.audit import audit_archive, audit_results, audit_secrets
    from bridge_evidence.contest.execution_audit import audit_execution_records

    if audit_profile not in ("contest-sdk", "legacy"):
        raise ValueError("unknown audit profile")
    # Never delete or reuse another package, even when its previous audit failed.
    output_dir.mkdir(parents=True, exist_ok=False)
    package_root = output_dir / "package_root"
    package_root.mkdir()
    report = {
        "passed": False,
        "profile": audit_profile,
        "sdk_passed": False,
        "errors": [],
    }
    pending = []
    published = []
    try:
        copy_code(src_root, package_root / "code")
        copy_design(design_file, package_root / "design")
        copy_result(result_dir, package_root / "result")
        from bridge_evidence.contest.run_state import file_digest

        report["artifact_binding"] = {
            "design_sha256": file_digest(design_file),
            "records": {p.name: file_digest(p) for p in state_dir.glob("*.json")},
            "timing_json_sha256": file_digest(result_dir / "infer_time.json"),
            "run_fingerprint": run_fingerprint,
        }
        errors = report["errors"]
        execution = audit_execution_records(
            input_dir,
            package_root / "result",
            state_dir,
            run_fingerprint,
            profile=audit_profile,
            require_strong=audit_profile == "contest-sdk",
        )
        report["execution"] = execution
        if (execution.get("timing") or {}).get("passed"):
            report["artifact_binding"]["timing_sha256"] = execution["timing"][
                "timing_sha256"
            ]
        report["expected_inputs"] = execution["expected_inputs"]
        errors.extend(execution["errors"])
        expected_names = (
            {
                p.stem + ".docx"
                for p in input_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in (".doc", ".docx")
                and not p.name.startswith("~$")
            }
            if input_dir.is_dir()
            else set()
        )
        # Check the original directory too: copying must not hide extra/invalid files.
        report["source_results"], issues = audit_results(result_dir, expected_names)
        errors.extend(issues)
        report["results"], issues = audit_results(
            package_root / "result", expected_names
        )
        errors.extend(issues)
        for directory in (result_dir, package_root / "result"):
            try:
                seconds = json.loads(
                    (directory / "infer_time.json").read_text(encoding="utf-8")
                )["infer_time"]
                if (
                    type(seconds) not in (int, float)
                    or not math.isfinite(seconds)
                    or seconds <= 0
                ):
                    raise ValueError("finite positive seconds required")
            except (OSError, ValueError, TypeError, KeyError, OverflowError):
                errors.append(
                    "infer_time.json: finite positive numeric seconds required"
                )
        report["run_manifest"], issues = audit_run_manifest(
            src_root, package_root, state_dir, run_fingerprint
        )
        errors.extend(issues)
        manifest = json.loads((state_dir.parent / "run_manifest.json").read_text())
        options = manifest["identity"].get("options", {})
        if audit_profile == "contest-sdk":
            if smoke_request is None:
                errors.append(
                    "contest-sdk requires a real staged smoke request; --help is not acceptance"
                )
            acceptance_sha = options.get("sdk_acceptance_sha256")
            if not isinstance(acceptance_sha, str) or len(acceptance_sha) != 64:
                errors.append(
                    "contest-sdk package requires version-bound real acceptance evidence"
                )
            for receipt in state_dir.glob("*.json"):
                workflow = json.loads(receipt.read_text()).get("workflow", {})
                if workflow.get("acceptance_file_sha256") != acceptance_sha:
                    errors.append(
                        "SDK acceptance does not match frozen run identity: "
                        + receipt.name
                    )
        active_config = options.get("candidate", options.get("config"))
        active_model = options.get("model_manifest", options.get("model"))
        from bridge_evidence.contest.run_state import atomic_json, json_digest

        if active_config:
            atomic_json(package_root / "code" / "active-config.json", active_config)
        if (
            active_model
            and isinstance(active_model, dict)
            and "manifest" in active_model
        ):
            atomic_json(
                package_root / "code" / "active-model-manifest.json",
                active_model["manifest"],
            )
        report["artifact_binding"]["active_config_sha256"] = json_digest(active_config)
        report["artifact_binding"]["active_model_sha256"] = json_digest(active_model)
        report["secrets"], issues = audit_secrets(
            [state_dir, state_dir.parent / "run_manifest.json", package_root]
        )
        errors.extend(issues)
        if errors:
            return report
        if audit_profile == "contest-sdk":
            # Run fresh short inference from exactly the directory archived below.
            # A prewritten 'verified' JSON cannot replace this operator-run check.
            import importlib.util

            request = json.loads(Path(smoke_request).read_text())
            allowed = {
                "input_dir",
                "python",
                "sdk_acceptance",
                "backend_url",
                "timeout",
                "budget_ledger",
                "idle_window",
                "rsi_sandbox_root",
                "rsi_release",
                "model_manifest",
                "sdk_parallel_acceptance",
            }
            if not isinstance(request, dict) or set(request) - allowed:
                raise ValueError("unknown staged smoke request fields")
            if not all(
                request.get(key)
                for key in (
                    "input_dir",
                    "sdk_acceptance",
                    "budget_ledger",
                    "idle_window",
                )
            ):
                raise ValueError(
                    "smoke request requires short inputs, SDK evidence, original shared budget and idle window"
                )
            allowed_inputs = {
                file_digest(p)
                for p in input_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in (".doc", ".docx")
                and not p.name.startswith("~$")
            }
            smoke_inputs = [
                p
                for p in Path(request["input_dir"]).rglob("*")
                if p.is_file()
                and p.suffix.lower() in (".doc", ".docx")
                and not p.name.startswith("~$")
            ]
            if not smoke_inputs or any(
                file_digest(p) not in allowed_inputs for p in smoke_inputs
            ):
                raise ValueError(
                    "smoke inputs must be approved report bytes from this submission"
                )
            request.setdefault(
                "backend_url", options.get("backend_url", "http://127.0.0.1:8000/v1")
            )
            exact_model = request.pop("model_manifest", None)
            if options.get("rsi_mode") == "frozen" and not exact_model:
                raise ValueError(
                    "frozen smoke requires original byte-bound model_manifest"
                )
            if exact_model:
                from bridge_evidence.contest.backends import ModelManifest

                if ModelManifest.load(exact_model).as_dict() != active_model["manifest"]:
                    raise ValueError("smoke model manifest differs from submission")
                # Frozen release pins exact file bytes, not equivalent JSON formatting.
                shutil.copyfile(
                    exact_model, package_root / "code" / "active-model-manifest.json"
                )
            spec = importlib.util.spec_from_file_location(
                "staged_smoke", Path(__file__).with_name("smoke_submission.py")
            )
            smoke = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(smoke)
            smoke_report = smoke.run_smoke(
                code_dir=package_root / "code",
                output_dir=output_dir / "smoke",
                model_manifest=package_root / "code" / "active-model-manifest.json",
                candidate_config=package_root / "code" / "active-config.json",
                backend=options.get("backend", "vllm-api"),
                expected_identity=manifest["identity"],
                rsi_bundle=(package_root / "code" / "frozen-rsi-bundle.json")
                if options.get("rsi_mode") == "frozen"
                else None,
                **request,
            )
            report["staged_smoke"] = smoke_report
            report["artifact_binding"]["smoke_sha256"] = file_digest(
                output_dir / "smoke/smoke-evidence.json"
            )
            if not smoke_report.get("formal_accepted"):
                errors.extend(
                    "staged smoke: " + error for error in smoke_report["errors"]
                )
                return report

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Legacy packages are explicitly baselines and never claim SDK acceptance.
        label = "legacy_baseline" if audit_profile == "legacy" else "contest_sdk"
        stem = f"IAIC_蜗牛战队_{label}_{stamp}"
        archives = [
            (output_dir / (stem + ".tar.gz"), build_tar_gz),
            (output_dir / (stem + ".zip"), build_zip),
        ]
        report["archives"] = {}
        for final, build in archives:
            temporary = final.with_name(final.name + ".pending")
            pending.append(temporary)
            build(package_root, temporary)
            section, issues = audit_archive(temporary)
            report["archives"][final.name] = section
            errors.extend(issues)
        if errors:
            return report
        for (final, _), temporary in zip(archives, pending):
            # Hard-link publication is atomic and refuses to replace an existing file.
            from bridge_evidence.contest.artifact_publish import publish_no_replace

            publish_no_replace(temporary, final, output_dir / "publication_locks")
            published.append(final)
        report["passed"] = True
        report["sdk_passed"] = audit_profile == "contest-sdk"
        report["archive_paths"] = [str(path.resolve()) for path in published]
        return report
    except Exception as exc:
        report["errors"].append(f"package build failed: {type(exc).__name__}: {exc}")
        for path in published:
            path.unlink(missing_ok=True)
        return report
    finally:
        for path in pending:
            path.unlink(missing_ok=True)
        # Keep staging and the audit outside the archive for inspection.
        try:
            with (output_dir / "audit.json").open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        except Exception:
            # Publication requires a durable audit result as well as valid bytes.
            for path in published:
                path.unlink(missing_ok=True)
            raise


def main() -> None:

    parser = argparse.ArgumentParser(description="Package IAIC submission")
    parser.add_argument(
        "--design-file", type=Path, required=True, help="唯一最终方案书"
    )
    parser.add_argument("--src-root", type=Path, default=Path("."))
    parser.add_argument("--result-dir", type=Path, default=Path("output/result"))
    parser.add_argument(
        "--state-dir", type=Path, required=True, help="本次 work-dir/records"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="显式指定本次输入目录"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="必须是尚不存在的新目录"
    )
    parser.add_argument(
        "--run-fingerprint",
        required=True,
        help="本次 run_manifest.json 的 SHA256 identity",
    )
    parser.add_argument(
        "--audit-profile",
        choices=("contest-sdk", "legacy"),
        default="contest-sdk",
        help="legacy 仅允许新格式内部 baseline；不代表 SDK 通过",
    )
    parser.add_argument(
        "--smoke-request",
        type=Path,
        help="正式包必需：短样本、SDK、共享预算与空闲窗口的 JSON；打包目录内实机执行",
    )
    args = parser.parse_args()

    report = package_submission(
        src_root=args.src_root,
        input_dir=args.input_dir,
        result_dir=args.result_dir,
        state_dir=args.state_dir,
        output_dir=args.output_dir,
        run_fingerprint=args.run_fingerprint,
        audit_profile=args.audit_profile,
        design_file=args.design_file,
        smoke_request=args.smoke_request,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
