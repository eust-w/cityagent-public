"""Content-bound run identities and atomic records; no filename-only resume."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import uuid
from pathlib import Path

MODEL_IDENTITY_SUFFIXES = frozenset({
    ".safetensors", ".bin", ".json", ".model", ".txt", ".tiktoken", ".py", ".jinja", ".jinja2"
})

def json_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    path = Path(path)
    before = path.stat()
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise RuntimeError(f"File changed while fingerprinting: {path}")
    return result.hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with pending.open("x", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def document_identity(path: Path, input_dir: Path) -> dict:
    return {
        "relative_path": Path(path)
        .resolve()
        .relative_to(Path(input_dir).resolve())
        .as_posix(),
        "sha256": file_digest(path),
    }


def record_name(document: dict) -> str:
    return json_digest(document) + ".json"


def validate_output_names(files: list[Path]) -> None:
    seen = {}
    for path in files:
        name = (path.stem + ".docx").casefold()
        if name in seen:
            raise ValueError(
                f"Reports map to the same output filename: {seen[name]} and {path}"
            )
        seen[name] = path


def file_digests(paths, digest=None) -> list[str]:
    """Hash files concurrently in input order; hashlib releases the GIL on large reads."""
    digest = digest or file_digest
    paths = list(paths)
    if len(paths) < 2:
        return [digest(p) for p in paths]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as pool:
        return list(pool.map(digest, paths))


def build_run_identity(
    model_dir: Path, code_root: Path, options: dict, *, model_files: dict | None = None
) -> dict:
    """Hash actual model bytes once per invocation, including tokenizer/config files."""
    model_dir, code_root = Path(model_dir).resolve(), Path(code_root).resolve()
    model_paths = sorted(
        p
        for p in model_dir.rglob("*")
        if p.is_file()
        and p.suffix
        in MODEL_IDENTITY_SUFFIXES
    )
    if not model_paths or not (model_dir / "config.json").is_file():
        raise ValueError("Missing model content for run identity")
    if model_files is None:
        model_files = dict(zip(
            (p.relative_to(model_dir).as_posix() for p in model_paths),
            file_digests(model_paths),
        ))
    elif not model_files or any(
        not isinstance(k, str) or not isinstance(v, str) or len(v) != 64
        for k, v in model_files.items()
    ):
        raise ValueError("Actual precomputed model hashes required")
    code_paths = [
        code_root / name
        for name in (
            "run.py",
            "README.md",
            "LICENSE",
            "NOTICE",
            "requirements-contest.txt",
            "THIRD_PARTY_NOTICES.md",
            "scripts/merge_lora_v3.py",
            "src/bridge_evidence/__init__.py",
            "src/bridge_evidence/domain_rules.py",
            "src/bridge_evidence/evaluation/__init__.py",
            "src/bridge_evidence/evaluation/generation.py",
        )
    ]
    code_paths += sorted((code_root / "src/bridge_evidence/contest").rglob("*.py"))
    code_files = {
        p.relative_to(code_root).as_posix(): file_digest(p)
        for p in code_paths
        if p.is_file()
    }
    packages = {}
    for name in ("torch", "transformers", "vllm", "tongagents", "python-docx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": 1,
        "model_files": model_files,
        "code_files": code_files,
        "packages": packages,
        "options": options,
    }


def bind_work_directory(work_dir: Path, identity: dict, *, resume: bool) -> str:
    work_dir = Path(work_dir)
    manifest = work_dir / "run_manifest.json"
    fingerprint = json_digest(identity)
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(
                "Work directory already belongs to a run; use a fresh directory or explicit --resume"
            )
        if (
            previous.get("run_fingerprint") != fingerprint
            or previous.get("identity") != identity
        ):
            raise ValueError(
                "Run identity changed: model, tokenizer, code, inputs or options differ; use a fresh directory"
            )
    else:
        if work_dir.exists() and any(
            p.name != ".execution.lock" for p in work_dir.iterdir()
        ):
            raise FileExistsError(
                "Refusing an untracked nonempty work directory; use a fresh directory"
            )
        atomic_json(
            manifest,
            {"schema_version": 1, "run_fingerprint": fingerprint, "identity": identity},
        )
    return fingerprint


def read_bound_text(path: Path, text_dir: Path) -> str:
    """External text requires a sidecar binding its bytes to the original report."""
    text_path = Path(text_dir) / (Path(path).stem + ".txt")
    sidecar = text_path.with_suffix(".txt.source.json")
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    data = text_path.read_bytes()
    if (
        info.get("document_sha256") != file_digest(path)
        or info.get("text_sha256") != hashlib.sha256(data).hexdigest()
    ):
        raise ValueError("Pre-extracted text does not match its report/content hashes")
    if info.get("parser") != "contest-libreoffice-utf8-v1":
        raise ValueError(
            "Unrecognized pre-extracted text parser; regenerate using the aligned extractor"
        )
    return data.decode("utf-8")
