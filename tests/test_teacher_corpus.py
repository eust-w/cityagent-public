import hashlib
import json
from pathlib import Path

import pytest

from bridge_evidence.evaluation.teacher import semantic_equal


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ROOT = ROOT / "artifacts" / "v13-teacher-oracle"


def test_teacher_manifest_sources_are_hash_pinned_and_outside_runtime() -> None:
    manifest_path = TEACHER_ROOT / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("development-only teacher corpus is intentionally absent from the submission package")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["samples"]) >= 8
    assert {sample["split"] for sample in manifest["samples"]} == {
        "dev_teacher",
        "locked_audit",
        "temporal_holdout",
    }
    for sample in manifest["samples"]:
        teacher_path = TEACHER_ROOT / sample["teacher"]
        teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
        source = ROOT / teacher["source"]["path"]
        assert source.is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == teacher["source"]["sha256"]
        assert teacher["annotation_protocol"]["runtime_import_forbidden"] is True


def test_runtime_does_not_reference_teacher_artifacts() -> None:
    runtime_root = ROOT / "src"
    assert runtime_root.is_dir(), "Runtime source directory must exist for this audit"
    runtime_files = sorted(runtime_root.rglob("*.py"))
    assert runtime_files, "Runtime source audit must inspect at least one Python file"
    forbidden = ("v13-teacher-oracle", ".teacher.json", "teacher_id")
    inspected = 0
    for path in runtime_files:
        if path.as_posix().endswith("bridge_evidence/evaluation/teacher.py"):
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
        inspected += 1
    assert inspected, "Runtime source audit must not skip every file"


def test_teacher_comparison_collapses_only_identical_side_rendering() -> None:
    assert semantic_equal("D级", "D级（左幅）/D级（右幅）")
    assert semantic_equal(85, "85（左幅）/85（右幅）")
    assert not semantic_equal("B级", "B级（左幅）/C级（右幅）")
    assert not semantic_equal(85, "85（左幅）/84（右幅）")
