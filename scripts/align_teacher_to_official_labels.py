from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bridge_evidence.evaluation.labels import pair_files, parse_label


FIELD_TO_LABEL = {
    "bridge.name": "桥梁名称",
    "scores.overall.score": "总体评分",
    "scores.overall.grade": "总体等级",
    "scores.superstructure.score": "上部结构评分",
    "scores.superstructure.grade": "上部结构等级",
    "scores.substructure.score": "下部结构评分",
    "scores.substructure.grade": "下部结构等级",
    "scores.deck.score": "桥面系评分",
    "scores.deck.grade": "桥面系等级",
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_adjudications(path: Path | None, sample_names: set[str]) -> dict[str, dict[str, str]]:
    """Load optional, user-owned adjudications; no corpus identities are bundled."""
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Adjudications must map teacher filenames to field/reason objects")
    for name, fields in payload.items():
        if name not in sample_names or not isinstance(fields, dict):
            raise ValueError("Adjudication must name a manifest sample and a field/reason object")
        for field, reason in fields.items():
            if field not in FIELD_TO_LABEL or not isinstance(reason, str) or not reason.strip():
                raise ValueError("Adjudication requires a supported target field and a nonempty reason")
    return payload


def _record_path(path: Path, project_root: Path) -> str:
    return str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Align source-first teacher traces with released training labels")
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Base for manifest source/label paths (default: current directory)")
    parser.add_argument("--label-root", type=Path, help="Labels grouped by source parent name; alternatively provide each sample's label path")
    parser.add_argument("--adjudications", type=Path, help="Optional JSON: {teacher_filename: {target_field: reason}}; default: no sample-specific adjudications")
    args = parser.parse_args()

    teacher_root = args.teacher_root.resolve()
    project_root = args.project_root.resolve()
    label_root = args.label_root.resolve() if args.label_root else None
    manifest = json.loads((teacher_root / "manifest.json").read_text(encoding="utf-8"))
    try:
        adjudications = load_adjudications(args.adjudications, {sample["teacher"] for sample in manifest["samples"]})
        if label_root is None and any(not sample.get("label") for sample in manifest["samples"]):
            raise ValueError("Provide each manifest sample's label path or an explicit --label-root")
    except ValueError as exc:
        parser.error(str(exc))
    pair_cache: dict[tuple[Path, Path], dict[Path, Path]] = {}
    updated: list[dict[str, Any]] = []

    for sample in manifest["samples"]:
        teacher_path = teacher_root / sample["teacher"]
        if not teacher_path.resolve().is_relative_to(teacher_root):
            raise ValueError("Teacher output must remain within --teacher-root")
        teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
        source_path = (project_root / teacher["source"]["path"]).resolve()
        if sample.get("label"):
            label_path = (project_root / sample["label"]).resolve()
        else:
            assert label_root is not None
            label_dir = label_root / source_path.parent.name
            cache_key = (source_path.parent, label_dir)
            if cache_key not in pair_cache:
                pair_cache[cache_key] = {source.resolve(): label.resolve() for source, label in pair_files(source_path.parent, label_dir)}
            label_path = pair_cache[cache_key].get(source_path)
        if label_path is None:
            raise RuntimeError(f"No released label paired with {source_path}")
        label = parse_label(label_path)

        for field, label_key in FIELD_TO_LABEL.items():
            if field not in teacher["targets"]:
                continue
            target = teacher["targets"][field]
            source_target = target.get("source_target_value", target.get("target_value"))
            target["source_target_value"] = source_target
            target["target_value"] = label.summary[label_key]
            target["status"] = "official_label_resolved"
            target["reason"] = (
                f"官方训练标签“{label_key}”定义竞赛评分口径；"
                f"source-first 审定值保存在 source_target_value，二者不一致时不把样本特例写入运行时。"
            )
        for field, reason in adjudications.get(sample["teacher"], {}).items():
            if field not in teacher["targets"]:
                raise ValueError("Adjudication names a field missing from the teacher targets")
            teacher["targets"][field]["status"] = "benchmark_convention_conflict"
            teacher["targets"][field]["reason"] = reason

        teacher["annotation_protocol"]["benchmark_label_checked_after_source_adjudication"] = True
        teacher["benchmark_label"] = {
            "path": _record_path(label_path, project_root),
            "sha256": sha256(label_path),
            "role": "released_training_label_and_final_teacher_target",
        }
        teacher_path.write_text(json.dumps(teacher, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        updated.append({"teacher": sample["teacher"], "label": str(label_path), "targets": len(teacher["targets"])})

    print(json.dumps({"updated": updated}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
