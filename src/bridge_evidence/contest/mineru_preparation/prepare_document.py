"""Launch one offline native DOCX parser in a separate Python process.

The caller's interpreter, environment and inference services remain untouched.
MinerU is optional and must already exist in the explicitly selected runtime.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


class PreparationError(RuntimeError):
    """Preparation failed; available diagnostics remain in the new output dir."""


def validate_paths(source, output, training_root, expected_sha256=None, *, input_scope="official-training"):
    if input_scope not in {"official-training", "official-test"}:
        raise ValueError("Explicit official-training or official-test scope required")
    root = Path(training_root).expanduser().resolve(strict=True)
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if not root.is_dir() or not source.is_relative_to(root):
        raise ValueError("Input must be inside the explicitly supplied source root")
    if not source.is_file() or source.suffix.lower() != ".docx":
        raise ValueError("Expected an official training DOCX; legacy DOC conversion is separate")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Output directory must be empty; existing results are never overwritten")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        if digest != expected_sha256:
            raise ValueError("Official training DOCX differs from expected_sha256")
    return source, output, root, digest


def parser_environment(output, base=None):
    env = dict(os.environ if base is None else base)
    env.update(MINERU_HOME=str(Path(output) / "runtime"), HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", MINERU_MODEL_SOURCE="local", CUDA_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", ACCELERATE_USE_CPU="true")
    # Only this child process receives the package path. No in-process sys.path
    # edits or top-level sidecar imports can contaminate a serving interpreter.
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_document(source, output, *, training_root=None, source_root=None, input_scope="official-training",
                     python_executable=None, expected_sha256=None, timeout=300, character_budget=6000):
    """Prepare one user-designated official training DOCX into an empty dir.

    ``training_root`` is an explicit scope supplied by the caller, not a claim
    that arbitrary local files have been independently certified as official.
    No corpus scan, test input, model inference or dependency installation runs.
    """
    if source_root is not None and training_root is not None:
        raise ValueError("Supply only one source root")
    if input_scope == "official-test" and source_root is None:
        raise ValueError("Test inference requires explicit source_root, not a training alias")
    source, output, root, digest = validate_paths(source, output, source_root or training_root,
        expected_sha256, input_scope=input_scope)
    if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    if type(character_budget) is not int or character_budget <= 0:
        raise ValueError("character_budget must be a positive integer; it is not a model token budget")
    executable = str(Path(python_executable or sys.executable).expanduser().absolute())
    if not Path(executable).is_file() or not os.access(executable, os.X_OK):
        raise ValueError("python_executable must be an existing executable in the prepared MinerU runtime")
    output.mkdir(parents=True, exist_ok=True)
    command = [executable, "-m", "bridge_evidence.contest.mineru_preparation._worker",
        str(source), str(output), "--source-root" if source_root is not None else "--training-root", str(root), "--expected-sha256", digest,
        "--character-budget", str(character_budget)]
    if input_scope != "official-training":
        command += ["--input-scope", input_scope]
    audit = {"source": str(source), "source_root": str(root), "docx_sha256": digest,
        "command": command, "separate_process": True, "input_scope": input_scope,
        "timeout_seconds": timeout}
    try:
        process = subprocess.run(command, env=parser_environment(output), cwd=str(output),
            capture_output=True, text=True, timeout=timeout, check=False)
        (output / "parser.stdout.log").write_text(process.stdout, encoding="utf-8")
        (output / "parser.stderr.log").write_text(process.stderr, encoding="utf-8")
        audit["returncode"] = process.returncode
        if process.returncode != 0:
            raise PreparationError("MinerU preparation failed; see failure.json and parser.stderr.log")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "prepared" or manifest.get("docx_sha256") != digest:
            raise PreparationError("Parser did not produce a prepared manifest for this input SHA")
        for name in ("document.json", "document.md", "materials.json", "parsed-input.docx"):
            if not (output / name).is_file():
                raise PreparationError("Parser omitted required artifact: " + name)
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise PreparationError("Official training source changed while parsing")
    except Exception as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            for key, value in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                (output / ("parser." + key + ".log")).write_bytes(
                    value if isinstance(value, bytes) else (value or "").encode())
        audit.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        _write_json(output / "launch.json", audit)
        if not (output / "failure.json").exists():
            _write_json(output / "failure.json", audit)
        if isinstance(exc, PreparationError):
            raise
        raise PreparationError("Preparation failed; diagnostics preserved in " + str(output)) from exc
    audit["status"] = "prepared"
    _write_json(output / "launch.json", audit)
    return manifest
