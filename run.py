#!/usr/bin/env python3
"""赛题一·比赛推理入口（在赛事环境内运行，全程离线，不使用任何外部 API）。

用法：
  # 我们赛机跑结果（合并模型已在 /workspace/work 下）
  python run.py \
      --model /workspace/work/qwen36_bridge_merged_full

  # 模型准备与正式推理解耦：先显式准备模型清单或合并权重
  python run.py --check
  python run.py

输出（对齐官方提交格式）：
  result/<与测试集同名>.docx   每份报告一个信息提取报告
  result/infer_time.json       {"infer_time": 总推理秒数}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bridge_evidence.contest.artifact_publish import publish_no_replace
from bridge_evidence.contest.chunked_inference import _instruction as fragment_instruction
from bridge_evidence.contest.constraint_cleaner import (
    clean_and_constrain,
    extract_advice_from_docx,
)
from bridge_evidence.contest.defect_chunker import extract_defect_rows_from_tables
from bridge_evidence.contest.execution_audit import validate_execution_record
from bridge_evidence.contest.inference_core import (
    InferenceFailure, infer_document, whole_instruction,
)
from bridge_evidence.contest.model_preserving_fusion import preserve_model_report
from bridge_evidence.contest.run_state import (
    atomic_json,
    bind_work_directory,
    build_run_identity,
    document_identity,
    file_digest,
    json_digest,
    read_bound_text,
    record_name,
    validate_output_names,
)
from bridge_evidence.contest.table_facts import extract_table_facts
from bridge_evidence.contest.temporal_router import detect_temporal_baseline
from bridge_evidence.contest.text_extract import extract_docx_text
from bridge_evidence.contest.tong_workflow import execute_workflow


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _validate_release_model_identity(release_identity, manifest_identity):
    """Bind a qualified frozen release to actual current model/tokenizer bytes."""
    if release_identity is None:
        return
    qualified = release_identity.get("model_identity") if isinstance(release_identity, dict) else None
    if (
        not isinstance(qualified, dict)
        or not isinstance(qualified.get("manifest"), dict)
        or not qualified["manifest"]
        or not isinstance(qualified.get("files"), dict)
        or not qualified["files"]
        or qualified != manifest_identity
    ):
        raise ValueError("qualified release model identity differs from current model contents")


VISUAL_CONFIG = {"max_tokens": 2048}


def add_visual_arguments(parser):
    parser.add_argument("--visual-evidence", choices=("off", "same-model"), default="off")
    parser.add_argument("--visual-capability", type=Path,
                        help="Actual fixed-model raster capability proof; no RSI admission")
    parser.add_argument("--visual-display-transforms", action="store_true",
                        help="Explicitly allow bound source-pixel crop/quarter-turn/flip views with same-model visual evidence")


def validate_visual_options(args):
    mode = getattr(args, "visual_evidence", "off")
    proof = getattr(args, "visual_capability", None)
    transforms = getattr(args, "visual_display_transforms", False)
    if type(transforms) is not bool:
        raise ValueError("visual display transforms must be an explicit Boolean")
    if transforms and mode != "same-model":
        raise ValueError("visual display transforms require same-model visual evidence")
    if mode == "off":
        if proof:
            raise ValueError("visual capability supplied while visual evidence is off")
        return
    if (mode != "same-model" or getattr(args, "workflow", None) not in ("chunked", "bounded-react")
            or args.backend not in ("vllm-api", "sglang-api") or not proof):
        raise ValueError("visual evidence requires chunked API inference and an explicit actual capability proof")
    if getattr(args, "text_dir", None):
        raise ValueError("visual evidence requires the original document, not external extracted text")


def visual_config(args):
    """Keep legacy default identity while binding an explicitly enabled view policy."""
    return {**VISUAL_CONFIG, **({"allow_display_transforms": True}
                              if getattr(args, "visual_display_transforms", False) else {})}


def configure_visual_runtime(args, model_identity):
    """Validate real probe artifacts once, before any report/model request."""
    validate_visual_options(args)
    args.model_identity = model_identity
    args.visual_model_identity_sha256 = json_digest(model_identity) if model_identity else None
    if getattr(args, "visual_evidence", "off") == "off":
        args.visual_capability_binding = None
        return {"mode": "off"}
    if not isinstance(model_identity, dict) or not model_identity.get("files"):
        raise ValueError("actual unchanged model identity required for visual evidence")
    from bridge_evidence.contest.vision_capability import validate_vision_capability

    args.visual_capability_binding = validate_vision_capability(args.visual_capability, model_identity)
    return {"mode": "same-model", "config": visual_config(args),
            "capability": args.visual_capability_binding}


def visual_run_nonce(args):
    """Separate fresh visual trials while preserving the current run on resume."""
    if getattr(args, "visual_evidence", "off") == "off":
        return None
    manifest = args.work_dir / "run_manifest.json"
    if getattr(args, "resume", False) and manifest.is_file():
        nonce = json.loads(manifest.read_text())["identity"]["options"].get("visual_trial_nonce")
        if not isinstance(nonce, str) or len(nonce) != 32 or any(c not in "0123456789abcdef" for c in nonce):
            raise ValueError("resumed visual run has no bound trial nonce")
        return nonce
    return uuid.uuid4().hex


def _convert_cached(path: Path, cache_dir: Path, suffix: str, conversion: str) -> Path:
    """Bind LibreOffice cache to source bytes, converter version and output bytes."""
    import subprocess

    source_sha = file_digest(path)
    version = subprocess.check_output(
        ["soffice", "--version"], text=True, timeout=30
    ).strip()
    key = json_digest(
        {"source_sha256": source_sha, "converter": version, "conversion": conversion}
    )
    directory = cache_dir / key
    target = directory / (path.stem + suffix)
    metadata = directory / "conversion.json"
    if target.is_file() and metadata.is_file():
        info = json.loads(metadata.read_text())
        if info.get("output_sha256") == file_digest(target):
            return target
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / ("attempt-" + uuid.uuid4().hex)
    attempt.mkdir()
    try:
        subprocess.run(
            [
                "soffice",
                "-env:UserInstallation=" + (attempt / "profile").resolve().as_uri(),
                "--headless",
                "--convert-to",
                conversion,
                "--outdir",
                str(attempt),
                str(path),
            ],
            check=True,
            timeout=300,
            capture_output=True,
        )
        produced = attempt / target.name
        if not produced.is_file() or not produced.stat().st_size:
            raise RuntimeError("LibreOffice did not produce a nonempty output")
        if source_sha != file_digest(path):
            raise RuntimeError("Report changed during document conversion")
        os.replace(produced, target)
        atomic_json(
            metadata,
            {
                "source_sha256": source_sha,
                "converter": version,
                "conversion": conversion,
                "output_sha256": file_digest(target),
            },
        )
        return target
    finally:
        shutil.rmtree(attempt)


def convert_doc_to_docx(path: Path, cache_dir: Path) -> Path:
    return _convert_cached(path, cache_dir, ".docx", "docx")


def convert_doc_to_text(path: Path, cache_dir: Path) -> str:
    return _convert_cached(
        path, cache_dir, ".txt", "txt:Text (encoded):UTF8"
    ).read_text(encoding="utf-8")


def report_text_size(path: Path) -> int | None:
    """Main-text characters read without conversion: Word 97 FIB ccpText or DOCX <w:t> text."""
    import re
    import zipfile

    try:
        if path.suffix.lower() == ".doc":
            import olefile

            with olefile.OleFileIO(str(path)) as ole:
                fib = ole.openstream("WordDocument").read(512)
            if fib[:2] != b"\xec\xa5":
                return None
            # FibBase (32 bytes), csw + fibRgW, cslw, then FibRgLw97 whose 4th long is ccpText.
            rg_lw = 34 + 2 * int.from_bytes(fib[32:34], "little") + 2
            return int.from_bytes(fib[rg_lw + 12:rg_lw + 16], "little")
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", "replace")
        return sum(len(text) for text in re.findall(r"<w:t(?:\s[^>]*)?>([^<]*)</w:t>", xml))
    except Exception:  # a scheduling hint must never fail the run
        return None


def order_reports(files: list[Path], schedule: str = "name") -> list[Path]:
    """Name order, or largest first so parallel workers do not end on one long tail.

    ``largest-first`` uses file bytes; ``longest-first`` uses main-text characters,
    since photos dominate the bytes while the text drives prompt and output length.
    """
    if schedule == "largest-first":
        return sorted(files, key=lambda p: (-p.stat().st_size, p.name))
    if schedule == "longest-first":
        def size(path):
            chars = report_text_size(path)
            return chars if chars is not None else path.stat().st_size // 16

        return sorted(files, key=lambda p: (-size(p), p.name))
    return sorted(files)


def select_input_dir(dataset_root: Path = Path("/dataset")) -> Path:
    """只自动选择测试目录，禁止默认递归混入同一挂载下的训练报告和标签。"""
    contest_test = dataset_root / "决赛数据集1/赛题1/线上测试集"
    if contest_test.is_dir():
        return contest_test
    candidates = [
        p
        for p in dataset_root.rglob("*")
        if p.is_dir() and p.name in {"线上测试集", "初赛测试集", "测试集"}
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        reports = [
            p
            for p in dataset_root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".doc", ".docx"}
        ]
        if reports and all(p.parent == dataset_root for p in reports):
            return dataset_root
    raise ValueError(
        f"无法唯一定位测试集：{dataset_root}。请用 --input-dir 明确指定测试报告目录，不能默认扫描整个数据挂载"
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add_visual_arguments(ap)
    from bridge_evidence.contest.divide_cli import add_arguments
    from bridge_evidence.contest.document_cli import add_arguments as add_document_arguments

    add_arguments(ap, include_workflow=False)
    add_document_arguments(ap)
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="默认选择 /dataset 下的线上测试集，不包含训练报告和标签",
    )
    ap.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/workspace/result"),
        help="默认写入与 code/ 并列的 result/，不依赖启动时的工作目录",
    )
    ap.add_argument(
        "--work-dir", type=Path, default=Path("/workspace/work/contest_run")
    )
    ap.add_argument("--model-manifest", type=Path)
    ap.add_argument("--prepared-model-identity", type=Path)
    ap.add_argument("--prepared-model-identity-sha256",
                    help="Operator-pinned preparation receipt; mutable weights are still rehashed")
    ap.add_argument("--candidate-config", type=Path)
    ap.add_argument(
        "--rsi-release",
        type=Path,
        help="independently qualified dual-run release evidence",
    )
    ap.add_argument(
        "--rsi-mode", choices=("off", "frozen", "experimental-online"), default="off"
    )
    ap.add_argument("--rsi-bundle", type=Path)
    ap.add_argument("--rsi-sandbox-root", type=Path)
    ap.add_argument("--rsi-development-manifest", type=Path)
    ap.add_argument("--rsi-ready-dir", type=Path)
    ap.add_argument(
        "--report-workers",
        type=int,
        choices=(1, 2, 4),
        default=None,
        help="Two independent reports; SDK concurrency requires actual parallel acceptance",
    )
    ap.add_argument("--sdk-parallel-acceptance", type=Path)
    ap.add_argument(
        "--backend",
        choices=("local-vllm", "vllm-api", "sglang-api"),
        default="local-vllm",
    )
    ap.add_argument("--backend-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument(
        "--workflow", choices=("whole", "chunked", "bounded-react", "divide-fixed", "divide-react", "document-agent"), default="whole"
    )
    ap.add_argument("--sdk-acceptance", type=Path)
    ap.add_argument(
        "--text-dir",
        type=Path,
        default=None,
        help="预提取 UTF-8 文本；每份必须附 .txt.source.json，绑定原件和文本 SHA-256",
    )
    ap.add_argument("--model", default="/path/to/merged-model")
    ap.add_argument(
        "--base-model",
        type=Path,
        default=Path("/path/to/base-model"),
        help="原生 LoRA 的只读底座",
    )
    ap.add_argument(
        "--adapter",
        type=Path,
        default=Path("/path/to/adapter"),
        help="原生 LoRA adapter，始终留在提交目录外",
    )
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument(
        "--engine",
        choices=("tongagents", "direct"),
        default="tongagents",
        help="正式默认 TongAgents；direct 仅用于非 SDK 开发基线，不通过 SDK 提交审计",
    )
    ap.add_argument(
        "--schedule",
        choices=("name", "largest-first", "longest-first"),
        default="name",
        help="报告处理顺序：按文件名、文件字节从大到小，或正文字数从多到少（并行时减少长报告拖尾）",
    )
    ap.add_argument(
        "--label-conventions",
        choices=("auto", "on", "off"),
        default="auto",
        help="官方标注约定规范化：auto 对非 whole 流程开启；on 对所有流程开启",
    )
    ap.add_argument(
        "--fusion-policy",
        choices=("model-preserving", "legacy"),
        default="model-preserving",
        help="默认保留模型语义，仅统一序号及建议计数；legacy 用于复现历史规则覆盖基线",
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        choices=range(1, 4),
        default=2,
        help="每报告真实模型调用最多 1–3 次；失败不回退为规则成功",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="仅复用输入、权重、代码、配置、输出与执行记录全部匹配的成功结果",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--check",
        action="store_true",
        help="只检查模型文件、Python 依赖及文档转换工具，不加载大模型、不生成结果",
    )
    return ap


def preflight(args, files: list[Path]) -> list[str]:
    import importlib.util
    import shutil

    errors = []
    if not args.input_dir.is_dir():
        errors.append(f"输入目录不存在：{args.input_dir}")
    elif not files:
        errors.append(f"输入目录没有 .doc/.docx：{args.input_dir}")
    if not 0 < args.max_new_tokens < args.max_model_len:
        errors.append("必须满足 0 < max-new-tokens < max-model-len")
    model = Path(args.model)
    if not model.exists():
        errors.append(
            f"模型未准备：{model}；请显式运行 scripts/merge_lora_v3.py 或使用已验证原生 LoRA 模型清单"
        )
    elif (model / ".merge_incomplete").exists():
        errors.append(f"合并模型尚未保存完成：{model}；请另选空的 --model 路径重建")
    elif not (model / "config.json").is_file():
        errors.append(f"模型目录不完整：{model}；请另选空的 --model 路径重建")
    else:
        index = model / "model.safetensors.index.json"
        try:
            if index.exists():
                shards = set(json.loads(index.read_text())["weight_map"].values())
                if not shards or any(not (model / shard).is_file() for shard in shards):
                    errors.append(f"模型权重分片缺失：{model}")
            elif not (model / "model.safetensors").is_file():
                errors.append(f"模型权重缺失：{model}")
        except (OSError, ValueError, KeyError, TypeError) as e:
            errors.append(f"无法读取模型索引：{e}")
    if (
        not getattr(args, "model_manifest", None)
        and model.resolve() == args.base_model.resolve()
    ):
        errors.append(
            "指定的是原始底座，会遗漏微调权重；请先合并外置 LoRA，再用 --model 指向合并模型"
        )
    try:
        validate_output_names(files)
    except ValueError as error:
        errors.append(str(error))
    packages = ["transformers", "docx"]
    if getattr(args, "backend", "local-vllm") == "local-vllm":
        packages += ["torch", "vllm", "safetensors"]
    if args.engine == "tongagents":
        packages.append("tongagents")
    for package in packages:
        if importlib.util.find_spec(package) is None:
            errors.append(f"缺少 Python 依赖：{package}")
    if args.text_dir:
        for path in files:
            try:
                read_bound_text(path, args.text_dir)
            except (OSError, ValueError) as error:
                errors.append(f"预提取文本缺少有效来源绑定：{path.name}: {error}")
    needs_soffice = any(p.suffix.lower() == ".doc" for p in files)
    if needs_soffice and not shutil.which("soffice"):
        errors.append(
            "输入含旧版 .doc，文本和结构化表格转换需要 LibreOffice（soffice）"
        )
    if not args.resume and args.result_dir.exists() and any(args.result_dir.iterdir()):
        errors.append("结果目录必须为空；续跑必须显式 --resume 并通过内容哈希校验")
    return errors


def prepare_model(args) -> None:
    """Formal execution never implicitly creates/merges large weights."""
    if not Path(args.model).exists():
        raise FileNotFoundError(
            "Prepare weights explicitly before inference, or select native LoRA with --model-manifest"
        )


def ensure_ppu_environment() -> None:
    """普通 SSH 会话也能直接启动；驱动路径须在 Python 进程启动前生效。"""
    profile = Path("/etc/profile.d/dockerenv.sh")
    if profile.is_file() and os.environ.get("BRIDGE_PPU_ENV_READY") != "1":
        os.execv(
            "/bin/bash",
            [
                "bash",
                "-c",
                'set -e; . /etc/profile.d/dockerenv.sh; export BRIDGE_PPU_ENV_READY=1; exec "$@"',
                "contest",
                sys.executable,
                *sys.argv,
            ],
        )


def _fuse_existing_success_path(
    data: dict,
    target_docx: Path,
    table_facts: dict,
    table_defects: list,
    table_advice: list,
    temporal_info: dict,
) -> dict:
    """Keep the historical successful-output fusion as an isolated baseline variable."""
    basic = data["基本信息"]
    for key in [
        "桥梁名称",
        "报告日期",
        "总体评分",
        "总体等级",
        "病害发展趋势与具体说明",
    ]:
        if not basic.get(key) and table_facts.get(key):
            basic[key] = table_facts[key]
    for key in [
        "上部结构评分",
        "上部结构等级",
        "下部结构评分",
        "下部结构等级",
        "桥面系评分",
        "桥面系等级",
        "上一次总体评分",
        "上一次总体等级",
        "上一次定检结果",
    ]:
        if table_facts.get(key):
            basic[key] = table_facts[key]
    for section, extracted in (("病害明细", table_defects), ("维修建议", table_advice)):
        model_rows = data[section]
        if extracted and (len(extracted) > len(model_rows) or len(model_rows) < 2):
            data[section] = extracted
    return clean_and_constrain(data, target_docx, temporal_info)


class DocumentProcessingError(RuntimeError):
    def __init__(self, record: dict):
        super().__init__(record["error"])
        self.record = record


def bind_source_evidence(path, target):
    """Bind parsed DOCX bytes to the report and its content-bound conversion."""
    source_sha, prepared_sha = file_digest(path), file_digest(target)
    conversion = None
    if path.suffix.lower() == ".doc":
        try:
            conversion = json.loads((target.parent / "conversion.json").read_text())
        except (OSError, ValueError) as error:
            raise ValueError("DOC conversion evidence missing") from error
        if (not isinstance(conversion, dict)
                or conversion.get("source_sha256") != source_sha
                or conversion.get("output_sha256") != prepared_sha
                or conversion.get("conversion") != "docx"
                or not isinstance(conversion.get("converter"), str)
                or not conversion["converter"]):
            raise ValueError("DOC conversion evidence does not match source")
    elif source_sha != prepared_sha:
        raise ValueError("DOCX preparation changed the report bytes")
    return {"schema_version": 1, "source_sha256": source_sha,
            "prepared_docx_path": str(target.resolve()),
            "prepared_docx_sha256": prepared_sha, "conversion": conversion}


def prepare_document(args, path, tokenizer):
    """CPU-only preparation shared by prefetch and synchronous execution.

    Structured workflows convert legacy Word once and build their exact plan in
    the preparation pool. Whole-report prompts retain the historical text path.
    """
    if (getattr(args, "visual_evidence", "off") != "off"
            or getattr(args, "visual_display_transforms", False)):
        validate_visual_options(args)
    before = file_digest(path)
    whole_doc_text = None
    if (path.suffix.lower() == ".doc" and getattr(args, "workflow", "whole") == "whole"
            and not getattr(args, "text_dir", None)):
        # The DOCX and TXT exports are independent LibreOffice runs; run them side by side.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            docx_job = pool.submit(convert_doc_to_docx, path, args.work_dir / "converted_docx")
            text_job = pool.submit(convert_doc_to_text, path, args.work_dir / "converted_txt")
            target, whole_doc_text = docx_job.result(), text_job.result()
    else:
        target = (convert_doc_to_docx(path, args.work_dir / "converted_docx")
                  if path.suffix.lower() == ".doc" else path)
    result = {"document_sha256": before, "target_docx": str(target),
              "source_evidence": bind_source_evidence(path, target)}
    if getattr(args, "workflow", "whole") != "whole":
        from bridge_evidence.contest.chunked_inference import prepare_chunk_plan
        from bridge_evidence.contest.long_document import DocumentSnapshot

        snapshot = DocumentSnapshot.from_docx(target)
        # Interrupted reports load their dynamic checkpoint before any planning.
        plan = (None if getattr(args, "resume", False)
                else prepare_chunk_plan(snapshot, tokenizer, args.candidate))
        if getattr(args, "visual_evidence", "off") != "off":
            validate_visual_options(args)
            from bridge_evidence.contest.visual_evidence import plan_visual_jobs

            result["visual_plan"] = plan_visual_jobs(
                target, snapshot, args.visual_model_identity_sha256,
                {**visual_config(args), "trial_id": args.visual_trial_id})
        result.update(snapshot=snapshot,
                      chunk_plan=plan,
                      prepared_config_sha256=json_digest(args.candidate))
        text = (read_bound_text(path, args.text_dir) if args.text_dir
                else "\n".join(region.text for region in snapshot.regions))
    elif whole_doc_text is not None:
        text = whole_doc_text
    else:
        text = (read_bound_text(path, args.text_dir) if args.text_dir else
                extract_docx_text(str(path)) if path.suffix.lower() == ".docx"
                else convert_doc_to_text(path, args.work_dir / "converted_txt"))
    if file_digest(path) != before:
        raise RuntimeError("Report changed during pipeline preparation")
    result["text"] = text
    return result


def infer_with_visual_evidence(args, prepared, tokenizer, generate, *, fingerprint,
                               document, record, policy=None):
    """Run optional pixels and text with one fixed report budget and one backend."""
    from dataclasses import asdict

    from bridge_evidence.contest.chunked_inference import (
        infer_chunked,
        prepare_chunk_plan,
    )
    from bridge_evidence.contest.rsi_runtime import REPORT_DEADLINE
    from bridge_evidence.contest.visual_evidence import (
        execute_visual_jobs,
        rebuild_visual_snapshot,
    )

    started = time.monotonic()
    inherited = REPORT_DEADLINE.get()
    deadline = min(inherited, started + 1200) if inherited is not None else started + 1200
    deadline_token = REPORT_DEADLINE.set(deadline)
    try:
        configure_visual_runtime(args, args.model_identity)
        if args.visual_trial_id != fingerprint:
            raise ValueError("visual evidence trial differs from the active run")
        native = prepared["snapshot"]
        plan = prepared["visual_plan"]
        if plan["identity"]["config"]["trial_id"] != fingerprint:
            raise ValueError("prepared visual plan belongs to another trial")
        if plan["identity"]["model_identity_sha256"] != args.visual_model_identity_sha256:
            raise ValueError("prepared visual plan belongs to another model identity")
        if plan["identity"]["config"]["max_tokens"] != VISUAL_CONFIG["max_tokens"]:
            raise ValueError("prepared visual plan output budget changed")
        if (plan["identity"]["config"].get("allow_display_transforms", False)
                != getattr(args, "visual_display_transforms", False)):
            raise ValueError("prepared visual plan display transform policy changed")
        native_plan = prepared.get("chunk_plan")
        if native_plan is None or prepared.get("prepared_config_sha256") != json_digest(args.candidate):
            native_plan = prepare_chunk_plan(native, tokenizer, args.candidate)
        text_tasks, visual_jobs = len(native_plan.tasks), len(plan["jobs"])
        total_limit = 4 * (text_tasks + visual_jobs) + 2
        cache_root = args.work_dir / "task_cache" / "reports" / json_digest(document)
        timing_path = cache_root / fingerprint / "visual-phase-timing.json"
        timing_identity = json_digest({"run": fingerprint, "document": document,
                                       "plan": plan["plan_sha256"]})
        previous_overhead = 0.0
        if timing_path.is_file():
            timing = json.loads(timing_path.read_text())
            previous_overhead = timing.get("overhead_seconds")
            if (timing.get("identity") != timing_identity
                    or type(previous_overhead) not in (int, float)
                    or not math.isfinite(previous_overhead) or previous_overhead < 0):
                raise ValueError("visual phase timing identity changed")
        REPORT_DEADLINE.set(min(REPORT_DEADLINE.get(), started + 1200 - previous_overhead))
        core_started = time.monotonic()

        def before_visual(job):
            prior = job["prior_calls"]
            if not isinstance(prior, list) or len(prior) + 1 > total_limit:
                raise RuntimeError("shared report call budget exhausted")
            spent = job["spent_seconds"]
            if type(spent) not in (int, float) or not math.isfinite(spent) or spent < 0:
                raise ValueError("invalid cumulative visual timing")
            # Core includes this attempt's elapsed time. Deduct history once.
            history = max(0.0, spent - (time.monotonic() - core_started))
            REPORT_DEADLINE.set(min(REPORT_DEADLINE.get(), started + 1200 - history - previous_overhead))
            if time.monotonic() >= REPORT_DEADLINE.get():
                raise TimeoutError("shared report deadline exhausted before visual call")

        visual = execute_visual_jobs(
            plan, generate, cache_root / fingerprint / "visual", budget=before_visual)
        core_finished = time.monotonic()
        calls = visual["calls"]
        raw_responses = {
            raw["response"]["call_id"]: raw["response"]["text"]
            for raw in visual["raw"]
            if isinstance(raw.get("response", {}).get("text"), str)
            and raw["response"].get("call_id")
        }
        envelope = {"plan": plan, **visual,
                    "native_chunk_plan": asdict(native_plan),
                    "native_plan_sha256": native_plan.digest,
                    "capability": args.visual_capability_binding,
                    "capability_path": str(args.visual_capability.resolve()),
                    "model_identity_sha256": args.visual_model_identity_sha256,
                    "trial_id": fingerprint}
        record["schema_version"] = 3 if policy else 2
        record["visual_evidence"] = envelope
        record["visual_config"] = plan["identity"]["config"]
        record["visual_contributing_call_ids"] = list(dict.fromkeys(
            item["call_id"] for item in visual["evidence"]))
        record["model_calls"] = calls
        record["seconds"] = sum(call.get("seconds") or 0.0 for call in calls)
        record["generation_evidence"] = {"schema_version": 1, "responses": raw_responses}
        if visual.get("recovery_blocked"):
            raise RuntimeError("visual recovery has a reserved call without durable response")
        spent = visual["spent_seconds"]
        if type(spent) not in (int, float) or not math.isfinite(spent) or spent < 0:
            raise ValueError("invalid cumulative visual timing")
        if spent + core_started - started >= 1200:
            raise TimeoutError("shared report deadline exhausted by visual history")
        effective = rebuild_visual_snapshot(native, plan, visual["evidence"], calls, visual["raw"])
        overhead = previous_overhead + (core_started - started) + (time.monotonic() - core_finished)
        atomic_json(timing_path, {"schema_version": 1, "identity": timing_identity,
                                  "overhead_seconds": overhead})
        elapsed = spent + overhead
        REPORT_DEADLINE.set(min(REPORT_DEADLINE.get(), time.monotonic() + 1200 - elapsed))
        if time.monotonic() >= REPORT_DEADLINE.get():
            raise TimeoutError("shared report deadline exhausted before text inference")
        prefix = {
            "identity": json_digest({"run": fingerprint, "document": document,
                                     "visual_plan": plan["plan_sha256"],
                                     "native_plan": native_plan.digest}),
            "initial_text_tasks": text_tasks,
            "initial_visual_jobs": visual_jobs,
            "total_call_limit": total_limit,
            "used_calls": len(calls), "elapsed_seconds": elapsed,
        } if visual_jobs else None
        result = infer_chunked(
            effective, tokenizer, generate, config=args.candidate,
            max_attempts=args.max_attempts, cache_dir=cache_root,
            policy=policy, prepared_plan=None, prefix_budget=prefix)
        text_calls = result["model_calls"]
        if set(c["call_id"] for c in calls) & set(c["call_id"] for c in text_calls):
            raise ValueError("visual and text model call IDs collide")
        result["model_calls"] = calls + text_calls
        result["seconds"] = sum(call.get("seconds") or 0.0 for call in result["model_calls"])
        result["generation_evidence"]["responses"] = {
            **raw_responses, **result["generation_evidence"]["responses"]}
        for key in ("visual_evidence", "visual_config", "visual_contributing_call_ids"):
            result[key] = record[key]
        return result
    finally:
        REPORT_DEADLINE.reset(deadline_token)


def process_document(
    args,
    path: Path,
    tokenizer,
    generate,
    *,
    fingerprint: str,
    document: dict | None = None,
    workflow_executor=execute_workflow,
    writer=None,
    expected_text_sha256: str | None = None,
    prepared=None,
    policy=None,
) -> dict:
    """Publish only after model validation and complete workflow verification."""
    if writer is None:
        from bridge_evidence.contest.docx_writer import write_extraction_docx

        writer = write_extraction_docx
    current_document = document_identity(path, args.input_dir)
    if document is not None and document != current_document:
        raise ValueError(
            "Report changed since run identity was created; use a fresh run"
        )
    document = current_document
    bound_text = None
    if args.text_dir:
        if expected_text_sha256 is None:
            raise ValueError("External text must be bound to the frozen run identity")
        bound_text = read_bound_text(path, args.text_dir)
        if (
            hashlib.sha256(bound_text.encode("utf-8")).hexdigest()
            != expected_text_sha256
        ):
            raise ValueError(
                "Pre-extracted text changed since run identity was created; use a fresh run"
            )
    out = args.result_dir / (path.stem + ".docx")
    record_path = args.work_dir / "records" / record_name(document)
    profile = "contest-sdk" if args.engine == "tongagents" else "legacy"
    if args.resume and record_path.is_file() and out.is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        errors = validate_execution_record(
            previous,
            expected_document=document,
            expected_run_fingerprint=fingerprint,
            output_path=out,
            profile=profile,
            **({"source_path": path,
                "expected_model_identity_sha256": args.visual_model_identity_sha256}
               if getattr(args, "visual_evidence", "off") != "off" else {}),
        )
        if (
            expected_text_sha256 is not None
            and previous.get("input_text_sha256") != expected_text_sha256
        ):
            errors.append("Resumed output does not match the frozen extracted text")
        if not errors:
            return previous
    if out.exists():
        raise FileExistsError(
            f"Existing output has no matching verified execution record: {out}; use a fresh result directory"
        )
    pending_dir = args.result_dir / (".pending-" + uuid.uuid4().hex)
    pending_dir.mkdir(parents=True)
    pending = pending_dir / out.name
    payload = {"document": document, "model_calls": []}
    report_started = time.perf_counter()
    record = {
        "schema_version": 1,
        "document": document,
        "run_fingerprint": fingerprint,
        "status": "failed",
        "model_calls": [],
        "model_validation": {"valid": False},
        "workflow": {
            "engine": args.engine,
            "verification": "unverified",
            "native_events": [],
        },
        "seconds": 0.0,
        "utc": datetime.now(UTC).isoformat(),
    }

    def prepare(value):
        if file_digest(path) != document["sha256"]:
            raise RuntimeError("Report changed before extraction")
        if prepared is not None:
            if prepared["document_sha256"] != document["sha256"]:
                raise ValueError("Prepared document no longer matches source")
            value.update(prepared)
            record["input_text_sha256"] = hashlib.sha256(
                prepared["text"].encode()
            ).hexdigest()
            return value
        if getattr(args, "workflow", "whole") != "whole":
            value.update(prepare_document(args, path, tokenizer))
            record["input_text_sha256"] = hashlib.sha256(value["text"].encode()).hexdigest()
            return value
        target = (
            convert_doc_to_docx(path, args.work_dir / "converted_docx")
            if path.suffix.lower() == ".doc"
            else path
        )
        if args.text_dir:
            text = bound_text
        elif path.suffix.lower() == ".docx":
            text = extract_docx_text(str(path))
        else:
            text = convert_doc_to_text(path, args.work_dir / "converted_txt")
        if file_digest(path) != document["sha256"]:
            raise RuntimeError("Report changed during extraction")
        value.update(text=text, target_docx=str(target))
        record["input_text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return value

    def infer(value):
        try:
            record["source_evidence"] = value.get("source_evidence") or bind_source_evidence(
                path, Path(value["target_docx"]))
            if getattr(args, "workflow", "whole") != "whole":
                from bridge_evidence.contest.chunked_inference import infer_chunked
                from bridge_evidence.contest.long_document import DocumentSnapshot

                snapshot = value.get("snapshot") or DocumentSnapshot.from_docx(Path(value["target_docx"]))
                if snapshot.document_sha256 != file_digest(Path(value["target_docx"])):
                    raise ValueError("Prepared document content changed before inference")
                prepared_plan = (value.get("chunk_plan")
                                 if value.get("prepared_config_sha256") == json_digest(args.candidate)
                                 else None)
                if getattr(args, "visual_evidence", "off") != "off":
                    result = infer_with_visual_evidence(
                        args, value, tokenizer, generate, fingerprint=fingerprint,
                        document=document, record=record, policy=policy)
                else:
                    result = infer_chunked(
                        snapshot,
                        tokenizer,
                        generate,
                        config=args.candidate,
                        max_attempts=args.max_attempts,
                        # Independent reports never share receipt/checkpoint state.
                        cache_dir=args.work_dir / "task_cache" / "reports" / json_digest(document),
                        policy=policy,
                        prepared_plan=prepared_plan,
                    )
                record["schema_version"] = 3 if policy else 2
                if policy:
                    record["rsi"] = result["rsi"]
                    record["rsi"]["mode"] = getattr(args, "rsi_mode", "frozen")
                    record["rsi_bundle_sha256"] = policy.bundle.digest
                for key in (
                    "snapshot",
                    "chunk_plan",
                    "fragments",
                    "coverage",
                    "provenance",
                    "conflicts",
                    "contributing_call_ids",
                    "snapshot_sha256",
                    "plan_sha256",
                ):
                    record[key] = result[key]
                if "shared_budget" in result:
                    record["shared_budget"] = result["shared_budget"]
                record["tool_events"] = result.get("tool_events", [])
                record["basic_observations"] = result.get("basic_observations", [])
                record["basic_ambiguities"] = result.get("basic_ambiguities", [])
                record["repair_evidence"] = result.get("repair_evidence", [])
                if "generation_evidence" in result:
                    record["generation_evidence"] = result["generation_evidence"]
            else:
                result = infer_document(
                    value["text"],
                    tokenizer,
                    generate,
                    max_model_len=args.max_model_len,
                    max_new_tokens=args.max_new_tokens,
                    max_attempts=args.max_attempts,
                    instruction=whole_instruction(
                        (getattr(args, "candidate", None) or {}).get("prompt", "source-fragment-v1")
                    ),
                )
        except Exception as error:
            if hasattr(error, "partial_evidence"):
                record["schema_version"] = 2
                record.update(error.partial_evidence)
            visual_calls = record.get("visual_evidence", {}).get("calls", [])
            record["model_calls"] = visual_calls + getattr(error, "model_calls", [])
            record["seconds"] = sum(call.get("seconds") or 0.0 for call in record["model_calls"])
            if isinstance(error, InferenceFailure):
                # Capture before the SDK wraps this exception. These remain
                # failed diagnostics and never enter fusion, scoring or rendering.
                try:
                    raw_bytes = error.raw_output.encode("utf-8")
                except UnicodeError:
                    # Reversible ASCII representation permits a durable failure
                    # receipt even when the backend returned invalid Unicode.
                    escaped = json.dumps(error.raw_output, ensure_ascii=True)
                    record["raw_output_json"] = escaped
                    record["raw_output_json_sha256"] = hashlib.sha256(escaped.encode("ascii")).hexdigest()
                else:
                    digest = hashlib.sha256(raw_bytes).hexdigest()
                    matches = [call for call in error.model_calls if call.get("output_sha256") == digest]
                    if matches:
                        record["raw_output"] = error.raw_output
                        record["raw_output_sha256"] = digest
                        record["raw_output_call_id"] = matches[-1]["call_id"]
            raise
        value.update(result)
        record["model_calls"] = result["model_calls"]
        record["seconds"] = result["seconds"]
        record["raw_output"] = result["raw_output"]
        record["model_json_sha256"] = json_digest(result["data"])
        return value

    def validate(value):
        # Postprocessing runs only after a complete meaningful model result.
        policy = getattr(args, "fusion_policy", "model-preserving")
        if policy == "legacy":
            target = Path(value["target_docx"])
            facts = extract_table_facts(target)
            temporal = detect_temporal_baseline(target, facts)
            defects = extract_defect_rows_from_tables(target)
            advice = extract_advice_from_docx(target)
            value["data"] = _fuse_existing_success_path(
                value["data"], target, facts, defects, advice, temporal
            )
            fusion_audit = {"mode": "legacy_success_path"}
        elif policy == "model-preserving":
            conventions = getattr(args, "label_conventions", "auto")
            value["data"], fusion_audit = preserve_model_report(
                value["data"], label_conventions=conventions == "on" or (
                    conventions == "auto" and getattr(args, "workflow", "whole") != "whole"))
            fusion_audit = {**fusion_audit, "mode": "model_preserving"}
        else:
            raise ValueError("Unknown fusion policy")
        record["model_validation"] = {
            "valid": True,
            "errors": [],
            "truncated": False,
            "scope": "complete_model_json_schema_and_" + fusion_audit["mode"],
        }
        record["fusion"] = {**fusion_audit, "result_sha256": json_digest(value["data"])}
        record["final_data"] = value["data"]
        record["final_json_sha256"] = json_digest(value["data"])
        from bridge_evidence.contest.facts import build_fact_graph

        record["facts"] = build_fact_graph(record).to_dict()
        record["facts_sha256"] = json_digest(record["facts"])
        return value

    def render(value):
        writer(value["data"], str(pending))
        if not pending.is_file() or not pending.stat().st_size:
            raise RuntimeError("Renderer did not produce a nonempty document")
        return value

    published_identity = None
    try:
        result = workflow_executor(
            payload,
            {
                "prepare": prepare,
                "infer": infer,
                "validate": validate,
                "render": render,
            },
            engine=args.engine,
            **(
                {"acceptance_path": args.sdk_acceptance}
                if getattr(args, "sdk_acceptance", None)
                else {}
            ),
        )
        record["workflow"] = result["workflow"]
        record["output"] = {"name": out.name, "sha256": file_digest(pending)}
        record["status"] = "success"
        record["report_wall_seconds"] = time.perf_counter() - report_started
        record["submission_ready"] = args.engine == "tongagents" and (
            getattr(args, "rsi_mode", "off") == "off"
            or (
                getattr(args, "rsi_mode", "off") == "frozen"
                and bool(getattr(args, "rsi_release_identity", None))
            )
        )
        errors = validate_execution_record(
            record,
            expected_document=document,
            expected_run_fingerprint=fingerprint,
            output_path=pending,
            profile=profile,
            **({"source_path": path,
                "expected_model_identity_sha256": args.visual_model_identity_sha256}
               if getattr(args, "visual_evidence", "off") != "off" else {}),
        )
        if errors:
            raise RuntimeError("Execution audit rejected output: " + "; ".join(errors))
        if file_digest(path) != document["sha256"]:
            raise RuntimeError("Report changed during inference")
        publication = publish_no_replace(
            pending, out, args.work_dir / "publication_locks"
        )
        record["publication"] = publication
        published_identity = (publication["device"], publication["inode"])
    except Exception as error:
        record["status"] = "failed"
        record["submission_ready"] = False
        record["error"] = f"{type(error).__name__}: {error}"
        if hasattr(error, "workflow"):
            record["workflow"] = error.workflow
    finally:
        record["report_wall_seconds"] = time.perf_counter() - report_started
        try:
            shutil.rmtree(pending_dir)
            if record_path.exists():
                history = (
                    record_path.parent
                    / "history"
                    / (record_path.stem + "-" + uuid.uuid4().hex + ".json")
                )
                history.parent.mkdir(parents=True, exist_ok=True)
                os.replace(record_path, history)
            atomic_json(record_path, record)
        except Exception:
            if published_identity is not None and out.exists():
                current = out.lstat()
                if (
                    current.st_dev,
                    current.st_ino,
                ) == published_identity and file_digest(out) == record["output"][
                    "sha256"
                ]:
                    out.unlink()  # Only this invocation's inode, never a replacement.
            raise
    if record["status"] != "success":
        raise DocumentProcessingError(record)
    return record


def main() -> None:
    # Lock before bind/recover; parallel document workers never recover the ledger.
    from bridge_evidence.contest.rsi_runtime import WorkLock

    args = build_parser().parse_args()
    if args.workflow == "document-agent":
        if args.rsi_mode != "off" or args.rsi_bundle or args.rsi_release or args.candidate_config:
            raise ValueError("document-agent requires RSI off and its own explicit settings")
        validate_visual_options(args)
        from bridge_evidence.contest.document_cli import run_document_agent
        run_document_agent(args, code_root=ROOT, convert_doc=convert_doc_to_docx)
        return
    if args.workflow in ("divide-fixed", "divide-react"):
        validate_visual_options(args)
        if args.rsi_mode != "off" or args.rsi_bundle or args.rsi_release or args.candidate_config:
            raise ValueError("upstream divide workflows require RSI off and their own explicit settings")
        from bridge_evidence.contest.divide_cli import run_divide
        run_divide(args, code_root=ROOT, convert_doc=convert_doc_to_docx)
        return
    with WorkLock(args.work_dir):
        _main_locked()


def _main_locked() -> None:
    wall_start = time.perf_counter()
    parser = build_parser()
    args = parser.parse_args()
    if bool(args.prepared_model_identity) != bool(args.prepared_model_identity_sha256):
        parser.error("prepared model identity requires both receipt and pinned SHA256")
    if args.prepared_model_identity and not args.model_manifest:
        parser.error("prepared identity requires --model-manifest")
    args.strategy_bundle = None
    args.rsi_release_identity = None
    if args.report_workers is None:
        args.report_workers = (
            2
            if args.backend != "local-vllm"
            and (args.engine == "direct" or args.sdk_parallel_acceptance)
            else 1
        )
    if args.rsi_mode != "off":
        from bridge_evidence.contest.rsi import StrategyBundle

        if not args.rsi_bundle or not args.rsi_sandbox_root:
            parser.error(
                "RSI requires an explicit bundle and prepared Linux sandbox root"
            )
        args.strategy_bundle = StrategyBundle.load(args.rsi_bundle)
        from bridge_evidence.contest.rsi_sandbox import verify_root

        verify_root(args.rsi_sandbox_root)
        if args.workflow == "whole" and not args.candidate_config:
            args.workflow = args.strategy_bundle.config["workflow"]
        if args.rsi_mode == "experimental-online" and not args.rsi_development_manifest:
            parser.error("online evolution requires an approved development manifest")
    from bridge_evidence.contest.backends import ModelManifest
    from bridge_evidence.contest.candidate import validate_config

    try:
        args.candidate = validate_config(
            json.loads(args.candidate_config.read_text())
            if args.candidate_config
            else (
                args.strategy_bundle.config
                if args.strategy_bundle
                else {"workflow": args.workflow}
            )
        )
        if args.candidate_config:
            args.workflow = args.candidate["workflow"]
        args.candidate["workflow"] = args.workflow
        validate_visual_options(args)
        if args.rsi_mode != "off" and args.workflow == "whole":
            parser.error("RSI requires chunked or bounded-react workflow")
        if (
            args.workflow != "whole"
            and args.candidate["input_budget"] + args.candidate["output_budget"]
            > args.max_model_len
        ):
            parser.error("candidate input and output budgets exceed model context")
        if args.backend != "local-vllm" and args.candidate["engine_parameters"]:
            parser.error(
                "API server engine parameters require an explicit verified server deployment"
            )
        manifest = (
            ModelManifest.load(args.model_manifest).validate()
            if args.model_manifest
            else None
        )
        if manifest:
            args.model = manifest.merged or manifest.base
            args.base_model = Path(manifest.base)
            args.adapter = Path(manifest.adapter) if manifest.adapter else None
        elif args.backend != "local-vllm":
            parser.error(
                "API backend requires --model-manifest binding its base, adapter and served alias"
            )
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    if args.rsi_release:
        if args.rsi_mode != "frozen":
            parser.error("release evidence requires frozen RSI")
        from bridge_evidence.contest.rsi_acceptance import verify_release

        try:
            args.rsi_release_identity = verify_release(
                args.rsi_release, args.strategy_bundle.digest, source_root=ROOT
            )
            release_value = json.loads(args.rsi_release.read_text())
            if (
                not args.model_manifest
                or file_digest(args.model_manifest)
                != release_value["model_manifest_sha256"]
            ):
                raise ValueError("qualified release model manifest changed")
        except (ValueError, OSError, KeyError, TypeError) as error:
            parser.error(str(error))
    if args.strategy_bundle and args.candidate != args.strategy_bundle.config:
        parser.error("RSI configuration must match its frozen strategy bundle")
    if args.report_workers > 1 and args.backend == "local-vllm":
        parser.error("parallel reports require the globally dispatched API backend")
    if args.report_workers > 1 and args.engine == "tongagents":
        from bridge_evidence.contest.rsi_pipeline import verify_parallel_sdk

        if not args.sdk_parallel_acceptance or not verify_parallel_sdk(
            args.sdk_parallel_acceptance, args.sdk_acceptance, args.backend_url,
            workers=args.report_workers,
        ):
            parser.error(
                "parallel SDK reports require content-bound real parallel acceptance"
            )
    if args.input_dir is None:
        try:
            args.input_dir = select_input_dir()
        except ValueError as error:
            parser.error(str(error))
    if not args.check:
        ensure_ppu_environment()
    files = sorted(
        p
        for p in args.input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (".docx", ".doc")
        and not p.name.startswith("~$")
    )
    files = order_reports(files, args.schedule)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("limit must be positive")
        files = files[: args.limit]
    if args.rsi_mode == "experimental-online":
        from bridge_evidence.contest.rsi_pipeline import approved_reports

        approved_reports(args.rsi_development_manifest, files)
    errors = preflight(args, files)
    if errors:
        for error in errors:
            log("检查失败：", error)
        raise SystemExit(1)
    if args.check:
        log("文件与依赖检查通过；尚未验证真实 SDK 编排、模型加载和推理")
        return
    prepare_model(args)
    documents = [document_identity(path, args.input_dir) for path in files]
    converter_version = None
    if any(path.suffix.lower() == ".doc" for path in files):
        import subprocess

        converter_version = subprocess.check_output(
            ["soffice", "--version"], text=True, timeout=30
        ).strip()
    log("校验模型、tokenizer 和运行代码内容哈希")
    if manifest and args.prepared_model_identity:
        from bridge_evidence.contest.model_identity import load_identity

        manifest_identity = load_identity(manifest, args.prepared_model_identity,
                                         expected_sha256=args.prepared_model_identity_sha256)
    else:
        manifest_identity = manifest.identity() if manifest else None
    try:
        _validate_release_model_identity(args.rsi_release_identity, manifest_identity)
    except ValueError as error:
        parser.error(str(error))
    visual_conditions = configure_visual_runtime(args, manifest_identity)
    identity = build_run_identity(
        Path(args.model),
        ROOT,
        {
            "engine": args.engine,
            "visual_evidence": visual_conditions,
            "visual_trial_nonce": visual_run_nonce(args),
            "rsi_mode": args.rsi_mode,
            "rsi_release": args.rsi_release_identity,
            "rsi_bundle": args.strategy_bundle.as_dict()
            if args.strategy_bundle
            else None,
            "rsi_sandbox_identity": json.loads(
                (
                    args.rsi_sandbox_root.parent
                    / (args.rsi_sandbox_root.name + ".identity.json")
                ).read_text()
            )["digest"]
            if args.rsi_sandbox_root
            else None,
            "report_workers": args.report_workers,
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "max_attempts": args.max_attempts,
            "enable_thinking": False,
            "temperature": 0.0,
            "instruction": whole_instruction(args.candidate["prompt"])
            if args.workflow == "whole" else fragment_instruction(args.candidate),
            "fusion": args.fusion_policy,
            "documents": documents,
            "text_sources": {
                document["relative_path"]: hashlib.sha256(
                    read_bound_text(path, args.text_dir).encode()
                ).hexdigest()
                for path, document in zip(files, documents)
            }
            if args.text_dir
            else None,
            "converter_version": converter_version,
            "workflow": args.workflow,
            "candidate": args.candidate,
            "backend": args.backend,
            "backend_url": args.backend_url,
            "model_manifest": manifest_identity,
            "sdk_acceptance_sha256": file_digest(args.sdk_acceptance)
            if args.sdk_acceptance
            else None,
        },
        model_files=(
            manifest_identity["files"]["merged" if manifest.merged else "base"]
            if manifest
            else None
        ),
    )
    fingerprint = bind_work_directory(args.work_dir, identity, resume=args.resume)
    args.visual_trial_id = fingerprint
    args.result_dir.mkdir(parents=True, exist_ok=True)
    log("run_fingerprint:", fingerprint)
    from uuid import uuid4

    from bridge_evidence.contest.backends import (
        AccountedBackend,
        APIBackend,
        LocalVLLMBackend,
    )
    from bridge_evidence.contest.fast_tokenizer import load_tokenizer
    from bridge_evidence.contest.timing import TimingLedger

    ledger = TimingLedger(args.work_dir / "timing.sqlite")
    ledger.recover()
    load_start = time.perf_counter()
    tok = load_tokenizer(
        manifest.tokenizer if manifest else args.model,
        allow_slow_fallback=True,
    )
    if args.backend == "local-vllm":
        if manifest is None:
            manifest = ModelManifest(
                str(args.base_model), args.model, merged=args.model
            )
        backend = LocalVLLMBackend(
            manifest,
            max_model_len=args.max_model_len,
            parameters=args.candidate["engine_parameters"],
            seed=args.candidate["seed"],
        )
    else:
        if not manifest.served_model:
            parser.error("model manifest must include served_model for native LoRA API")
        backend = APIBackend(
            args.backend_url,
            manifest.served_model,
            tok,
            kind=args.backend,
            workers=args.candidate["batch_size"],
            seed=args.candidate["seed"],
        )
        atomic_json(args.work_dir / "backend_probe.json", backend.probe())
    ledger.span(uuid4().hex, "model_load", time.perf_counter() - load_start)
    generate = AccountedBackend(backend, ledger)

    import threading
    from concurrent.futures import ThreadPoolExecutor
    from copy import copy

    from bridge_evidence.contest.rsi_pipeline import pipeline
    from bridge_evidence.contest.rsi_runtime import Dispatcher
    from bridge_evidence.contest.rsi_sandbox import PolicyController

    background = ThreadPoolExecutor(max_workers=1)
    validation_future = None
    policy_pool, policy_lock = {}, threading.Lock()
    coordinator, evolution = None, None
    if args.rsi_mode == "experimental-online":
        from bridge_evidence.contest.rsi_evolution import (
            EvolutionStore,
            OnlineCoordinator,
        )

        evolution = EvolutionStore(
            args.work_dir / "evolution.sqlite",
            json_digest(manifest_identity),
            file_digest(args.rsi_development_manifest),
        )
        evolution.add(args.strategy_bundle, generation=0)
        coordinator = OnlineCoordinator(args.strategy_bundle, evolution)

    def policy_for(index, bundle):
        if bundle is None:
            return None
        key = (index, bundle.digest)
        with policy_lock:
            if key not in policy_pool:
                prepared_key = (-1, bundle.digest)
                policy_pool[key] = (
                    policy_pool.pop(prepared_key)
                    if index >= 0 and prepared_key in policy_pool
                    else PolicyController(bundle, args.rsi_sandbox_root)
                )
            return policy_pool[key]

    for index in range(min(2, len(files))):
        policy_for(index, args.strategy_bundle)

    def prepare_ahead(path):
        if args.resume:
            document = document_identity(path, args.input_dir)
            if ((args.work_dir / "records" / record_name(document)).is_file()
                    and (args.result_dir / (path.stem + ".docx")).is_file()):
                # process_document validates the existing full receipt before use.
                return None
        return prepare_document(args, path, tok)

    total_seconds, failures = 0.0, 0
    capacity = args.candidate["batch_size"] if args.backend != "local-vllm" else 1
    with Dispatcher(generate, capacity=capacity) as dispatcher:
        dispatcher.begin_foreground()

        def process(index, prepared):
            bundle = coordinator.pin() if coordinator else args.strategy_bundle
            if bundle:
                from bridge_evidence.contest.rsi import StrategyBundle

                pin_dir = args.work_dir / "rsi-report-pins"
                pin_dir.mkdir(exist_ok=True)
                pin_path = pin_dir / record_name(documents[index])
                if pin_path.is_file():
                    pinned = json.loads(pin_path.read_text())
                    if pinned["run_fingerprint"] != fingerprint:
                        raise ValueError("report-start RSI identity changed")
                    value = pinned["bundle"]
                    bundle = StrategyBundle(
                        value["source"],
                        value["config"],
                        value["parent_sha256"],
                        value["abi"],
                    )
                else:
                    atomic_json(
                        pin_path,
                        {"run_fingerprint": fingerprint, "bundle": bundle.as_dict()},
                    )
            runtime = copy(args)
            if bundle:
                runtime.candidate = bundle.config
            return process_document(
                runtime,
                files[index],
                tok,
                generate if args.backend == "local-vllm" else dispatcher,
                fingerprint=fingerprint,
                document=documents[index],
                expected_text_sha256=(identity["options"]["text_sources"] or {}).get(
                    documents[index]["relative_path"]
                ),
                prepared=prepared,
                policy=policy_for(index, bundle),
            )

        try:
            for index, response in pipeline(
                files, prepare_ahead, process, workers=args.report_workers
            ):
                if isinstance(response, Exception):
                    failures += 1
                    if coordinator:
                        coordinator.rollback(type(response).__name__)
                    log(
                        f"[{index + 1}/{len(files)}] {files[index].name} 失败：{response}"
                    )
                else:
                    total_seconds += response["seconds"]
                    log(
                        f"[{index + 1}/{len(files)}] {files[index].name} 完成，模型推理 {response['seconds']:.2f}s"
                    )
                with policy_lock:
                    retired = [
                        policy_pool.pop(key)
                        for key in list(policy_pool)
                        if key[0] == index
                    ]
                for policy in retired:
                    background.submit(policy.close)
                if (
                    coordinator
                    and coordinator.report_done()
                    and args.rsi_ready_dir
                    and (validation_future is None or validation_future.done())
                ):
                    # CPU validation only; foreground does not wait for its result.
                    from bridge_evidence.contest.rsi_experiments import load_ready

                    def check_ready():
                        try:
                            candidate, evidence = load_ready(
                                args.rsi_ready_dir, coordinator.active
                            )
                            generation = (
                                next(
                                    h["generation"]
                                    for h in evolution.history()
                                    if h["sha256"] == candidate.parent_sha256
                                )
                                + 1
                            )
                            warm = policy_for(-1, candidate)
                            deadline = time.monotonic() + 3
                            while time.monotonic() < deadline and not warm.error:
                                with warm.lock:
                                    if warm.pool:
                                        break
                                time.sleep(0.01)
                            with warm.lock:
                                ready = bool(warm.pool)
                            if ready:
                                evolution.add(candidate, generation)
                                coordinator.offer(
                                    candidate,
                                    evidence={**evidence, "worker_ready": True},
                                )
                        except (
                            OSError,
                            ValueError,
                            KeyError,
                            RuntimeError,
                            StopIteration,
                        ):
                            return

                    validation_future = background.submit(check_ready)
        finally:
            dispatcher.end_foreground()
    background.shutdown(wait=True)
    for policy in policy_pool.values():
        policy.close()
    if evolution:
        evolution.close()
    ledger.span(uuid4().hex, "batch_wall", time.perf_counter() - wall_start)
    timing = ledger.summary()
    atomic_json(args.work_dir / "timing_summary.json", timing)
    # Wall clock of the whole batch: concurrent requests overlap and are counted once.
    infer_seconds = timing["spans"].get("batch_wall") or timing["generation_seconds"]
    atomic_json(
        args.result_dir / "infer_time.json",
        {"infer_time": math.ceil(infer_seconds)},
    )
    if failures:
        log(f"FAILED: {failures} 份；有效记录保存在 {args.work_dir / 'records'}")
        raise SystemExit(1)
    log(
        f"DONE: {len(files)} 份，模型总推理时间 {total_seconds:.2f}s；engine={args.engine}"
    )


if __name__ == "__main__":
    main()
