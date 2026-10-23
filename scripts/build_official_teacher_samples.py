from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from bridge_evidence.evaluation.labels import pair_files, parse_label
from bridge_evidence.parsers.docx import DocxStructureParser


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

FIELD_TERMS = {
    "bridge.name": ("桥梁名称", "工程名称"),
    "scores.overall.score": ("总体", "全桥", "综合评定", "BCI", "Dr"),
    "scores.overall.grade": ("总体", "全桥", "综合评定", "BCI", "Dr"),
    "scores.superstructure.score": ("上部结构", "BCIs", "BSIs"),
    "scores.superstructure.grade": ("上部结构", "BCIs", "BSIs"),
    "scores.substructure.score": ("下部结构", "BCIx", "BSIx"),
    "scores.substructure.grade": ("下部结构", "BCIx", "BSIx"),
    "scores.deck.score": ("桥面系", "BCIm", "BSIm"),
    "scores.deck.grade": ("桥面系", "BCIm", "BSIm"),
}

SCORE_SCOPE_LABELS = {
    "scores.overall.score": ("总体", "整体"),
    "scores.deck.score": ("桥面系",),
    "scores.superstructure.score": ("上部结构",),
    "scores.substructure.score": ("下部结构",),
}


def _sha256(path: Path) -> str:
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


def _compact(value: Any) -> str:
    return re.sub(r"[\s（）()：:/／，,。.;；]", "", str(value)).upper()


def _signals(value: Any) -> list[str]:
    rendered = str(value)
    signals = re.findall(r"\d+(?:\.\d+)?|[ABCDE](?=级|类)|[\u4e00-\u9fff]{2,}", rendered, re.I)
    return [signal for signal in signals if signal not in {"左幅", "右幅", "无"}]


def _evidence_for(parsed: Any, field: str, expected: Any) -> list[str]:
    """Locate source support without reading a student prediction.

    The evidence locator is deliberately recall-oriented.  It records stable
    paragraph/table coordinates for human review; it does not decide which
    competing metric is correct and is never imported by runtime code.
    """

    terms = FIELD_TERMS[field]
    expected_compact = _compact(expected)
    numeric_or_grade = [
        token.upper()
        for token in re.findall(r"\d+(?:\.\d+)?|[ABCDE](?=级|类)", str(expected), re.I)
    ]
    rows: list[tuple[int, str]] = []

    def consider(identifier: str, text: str) -> None:
        compact = _compact(text)
        term_hit = any(_compact(term) in compact for term in terms)
        exact_hit = bool(expected_compact and expected_compact != "无" and expected_compact in compact)
        signal_hits = sum(token in compact for token in numeric_or_grade)
        if field == "bridge.name":
            score = 100 if exact_hit else (20 if term_hit else 0)
        else:
            score = (40 if term_hit else 0) + signal_hits * 20 + (40 if exact_hit else 0)
        if score:
            rows.append((score, identifier))

    for paragraph in parsed.paragraphs:
        if paragraph.text:
            consider(f"P{paragraph.index}", paragraph.text)
    for table in parsed.tables:
        for row_index, row in enumerate(table.rows):
            consider(f"T{table.index}R{row_index}", " | ".join(row))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [identifier for _, identifier in rows[:6]]


def _pair_map(source_dir: Path, label_dir: Path) -> dict[Path, Path]:
    return {source.resolve(): label.resolve() for source, label in pair_files(source_dir, label_dir)}


def _source_name_has_optional_chainage(parsed: Any, expected: Any) -> bool:
    """Detect a label-only choice to omit a source-published chainage suffix."""

    expected_text = str(expected).strip()
    if not expected_text or re.search(r"[KkL][A-Za-z]?\d*\+\d", expected_text):
        return False
    escaped = re.escape(expected_text)
    suffix = rf"{escaped}\s*[（(]\s*[KkL][A-Za-z]?\d*\+\d+(?:\.\d+)?\s*[）)]"
    for table in parsed.tables:
        for row in table.rows:
            text = " | ".join(row)
            if any(term in text for term in ("工程名称", "桥梁名称")) and re.search(suffix, text):
                return True
    return False


def _published_side_scores(parsed: Any, field: str) -> dict[str, set[float]]:
    """Collect direction-specific values only from final rating tables."""

    labels = SCORE_SCOPE_LABELS[field]
    scores: dict[str, set[float]] = {"左幅": set(), "右幅": set()}
    for table in parsed.tables:
        context = " ".join([*table.context[-6:], " | ".join(table.header)])
        if "技术状况评定表" not in context:
            continue
        left_position = context.rfind("左幅")
        right_position = context.rfind("右幅")
        if left_position < 0 and right_position < 0:
            continue
        side = "左幅" if left_position > right_position else "右幅"
        for row in table.rows:
            row_label = "".join(row[:1])
            if not any(label in row_label for label in labels):
                continue
            for cell in row[1:]:
                match = re.search(r"(?<!\d)(\d{1,3}(?:\.\d+)?)(?!\d)", cell)
                if match and 0 <= float(match.group(1)) <= 100:
                    scores[side].add(float(match.group(1)))
                    break
    return scores


def benchmark_convention_conflicts(parsed: Any, label_summary: dict[str, Any]) -> dict[str, str]:
    """Identify released-label choices that the source cannot determine.

    These annotations affect development scoring only. They are derived before
    a student prediction exists, never imported by runtime code, and never
    included in the submission package.
    """

    conflicts: dict[str, str] = {}
    bridge_name = label_summary.get("桥梁名称", "")
    if _source_name_has_optional_chainage(parsed, bridge_name):
        conflicts["bridge.name"] = (
            "源报告在工程名称/桥梁名称字段发布了含桩号的全名，而释放标签省略桩号；"
            "同一标签语料中又大量保留桩号，无法由源报告稳定推断该格式选择。"
        )
    for field, label_key in FIELD_TO_LABEL.items():
        if field not in SCORE_SCOPE_LABELS:
            continue
        expected = str(label_summary.get(label_key, ""))
        if not re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", expected):
            continue
        side_scores = _published_side_scores(parsed, field)
        if not side_scores["左幅"] or not side_scores["右幅"]:
            continue
        if len(side_scores["左幅"] | side_scores["右幅"]) <= 1:
            continue
        conflicts[field] = (
            "源报告最终技术状况评定表分别发布左右幅且数值不同，释放标签却压成单值；"
            "同类释放标签同时存在保留双幅的样本，单值/双值格式无法由报告结构唯一决定。"
        )
    return conflicts


def build_teacher(
    project_root: Path,
    teacher_root: Path,
    sample: dict[str, Any],
    *,
    label_root: Path | None = None,
    adjudications: dict[str, dict[str, str]] | None = None,
) -> Path:
    source_path = (project_root / sample["source"]).resolve()
    output_path = teacher_root / sample["teacher"]
    if not output_path.resolve().is_relative_to(teacher_root.resolve()):
        raise ValueError("Teacher output must remain within the manifest directory")
    if sample.get("label"):
        label_path = (project_root / sample["label"]).resolve()
    elif label_root is not None:
        label_path = _pair_map(source_path.parent, label_root / source_path.parent.name).get(source_path)
    else:
        raise ValueError("Provide the manifest sample's label path or an explicit --label-root")
    if label_path is None:
        raise RuntimeError(f"No released label paired with {source_path}")

    parsed = DocxStructureParser().parse(source_path)
    label = parse_label(label_path)
    targets: dict[str, Any] = {}
    for field, label_key in FIELD_TO_LABEL.items():
        expected = label.summary.get(label_key, "无")
        evidence = _evidence_for(parsed, field, expected)
        targets[field] = {
            "target_value": expected,
            "evidence": evidence,
            "status": "official_label_resolved",
            "confidence": 1.0,
            "reason": (
                f"官方训练标签“{label_key}”作为竞赛教师目标；"
                "evidence 仅定位源报告中的候选依据，不向运行时暴露标签。"
            ),
        }
    for field, reason in benchmark_convention_conflicts(parsed, label.summary).items():
        targets[field]["status"] = "benchmark_convention_conflict"
        targets[field]["reason"] = reason
    for field, reason in (adjudications or {}).get(sample["teacher"], {}).items():
        targets[field]["status"] = "benchmark_convention_conflict"
        targets[field]["reason"] = reason

    source_hash = _sha256(source_path)
    teacher = {
        "schema_version": "1.0",
        "teacher_id": f"teacher:{source_hash[:8]}:official-summary-v1",
        "purpose": "development-only official-label teacher trace; forbidden as runtime or submission input",
        "source": {
            "path": _record_path(source_path, project_root),
            "sha256": source_hash,
        },
        "annotation_protocol": {
            "source_evidence_reviewed": True,
            "student_output_hidden_during_teacher_construction": True,
            "benchmark_label_is_final_teacher_target": True,
            "runtime_import_forbidden": True,
        },
        "targets": targets,
        "benchmark_label": {
            "path": _record_path(label_path, project_root),
            "sha256": _sha256(label_path),
            "role": "released_training_label_and_final_teacher_target",
        },
    }
    output_path.write_text(json.dumps(teacher, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hash-pinned official-label teacher traces")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Base for manifest source/label paths (default: current directory)")
    parser.add_argument("--label-root", type=Path, help="Labels grouped by source parent name; alternatively provide each sample's label path")
    parser.add_argument("--adjudications", type=Path, help="Optional JSON: {teacher_filename: {target_field: reason}}; default: no sample-specific adjudications")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    label_root = args.label_root.resolve() if args.label_root else None
    manifest_path = args.manifest.resolve()
    teacher_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        adjudications = load_adjudications(args.adjudications, {sample["teacher"] for sample in manifest["samples"]})
        if label_root is None and any("source" in sample and not sample.get("label") for sample in manifest["samples"]):
            raise ValueError("Provide each manifest sample's label path or an explicit --label-root")
    except ValueError as exc:
        parser.error(str(exc))
    built: list[str] = []
    for sample in manifest["samples"]:
        output_path = teacher_root / sample["teacher"]
        if "source" not in sample or (output_path.exists() and not args.force):
            continue
        build_teacher(project_root, teacher_root, sample, label_root=label_root, adjudications=adjudications)
        built.append(sample["teacher"])
    print(json.dumps({"built": built, "count": len(built)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
