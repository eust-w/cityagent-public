from __future__ import annotations

from pathlib import Path

import pytest

from bridge_evidence.config import Settings
from bridge_evidence.models.schema import Disease, DocumentState, FieldValue
from bridge_evidence.orchestrator import BridgeEvidenceOrchestrator
from bridge_evidence.parsers.docx import ParagraphBlock, ParsedDocument


def _complete_state() -> DocumentState:
    state = DocumentState(document_id="sha256:auto", source_path="auto.docx")
    fields = [
        state.bridge.name,
        state.bridge.report_date,
        state.scores.overall.score,
        state.scores.overall.grade,
        state.scores.deck.score,
        state.scores.deck.grade,
        state.scores.superstructure.score,
        state.scores.superstructure.grade,
        state.scores.substructure.score,
        state.scores.substructure.grade,
    ]
    values = ["示例桥", "2019年1月1日", 91.2, "A级", 88.0, "B级", 93.0, "A级", 95.0, "A级"]
    for field, value in zip(fields, values):
        field.value = value
        field.source_priority = 90
        field.confidence = 0.99
    return state


def test_auto_review_skips_complete_high_confidence_facts() -> None:
    state = _complete_state()
    parsed = ParsedDocument(path=Path("auto.docx"), paragraphs=[], tables=[])

    assert not BridgeEvidenceOrchestrator._should_escalate_to_online_review(state, parsed)


def test_auto_review_escalates_missing_or_conflicting_facts() -> None:
    state = _complete_state()
    state.scores.deck.score.value = "无"
    parsed = ParsedDocument(path=Path("auto.docx"), paragraphs=[], tables=[])

    assert BridgeEvidenceOrchestrator._should_escalate_to_online_review(state, parsed)

    state = _complete_state()
    state.scores.overall.grade.conflicts = ["P10"]
    assert BridgeEvidenceOrchestrator._should_escalate_to_online_review(state, parsed)


def test_result_filename_normalizes_legacy_doc_to_docx() -> None:
    assert BridgeEvidenceOrchestrator.result_filename(Path("旧格式报告.doc")) == "旧格式报告.docx"
    assert BridgeEvidenceOrchestrator.result_filename(Path("新格式报告.docx")) == "新格式报告.docx"


def test_no_source_comparison_closes_dependent_detail_fields() -> None:
    state = _complete_state()
    state.diseases = [Disease(
        sequence="1",
        component="主梁",
        location="1#梁",
        type="裂缝",
        description="1#梁裂缝",
    )]

    BridgeEvidenceOrchestrator._apply_no_comparison_convention(state)

    disease = state.diseases[0]
    assert disease.is_new is False
    assert disease.novelty_source == "no_comparison"
    assert disease.previous_state == "无"
    assert disease.development == "无"
    assert state.scores.previous.score.value == "无"
    assert state.scores.previous.grade.value == "无"
    assert state.narrative.development_trend == "无"


def test_source_comparison_detection_ignores_generic_history_but_recovers_prior_result_prose() -> None:
    state = _complete_state()
    state.scores.previous.score = FieldValue()
    state.scores.previous.grade = FieldValue()
    generic = ParsedDocument(
        path=Path("generic.docx"),
        paragraphs=[ParagraphBlock(0, "该桥建成于2014年并于2018年完成本次检测。", "正文")],
        tables=[],
    )
    temporal = ParsedDocument(
        path=Path("temporal.docx"),
        paragraphs=[ParagraphBlock(0, "上一次定期检查总体等级为B级。", "正文")],
        tables=[],
    )

    assert not BridgeEvidenceOrchestrator._source_has_temporal_comparison(state, generic)
    assert BridgeEvidenceOrchestrator._source_has_temporal_comparison(state, temporal)


def test_source_comparison_mode_uses_explicit_current_only_metadata() -> None:
    current_only = _complete_state()
    current_only.source_path = "测试桥报告（现A级）.docx"
    comparative = _complete_state()
    comparative.source_path = "测试桥报告（原B级，现A级）.docx"
    parsed = ParsedDocument(path=Path("sample.docx"), paragraphs=[], tables=[])

    assert BridgeEvidenceOrchestrator._source_declares_no_comparison(current_only, parsed)
    assert not BridgeEvidenceOrchestrator._source_declares_no_comparison(comparative, parsed)


def test_explicit_no_history_declaration_overrides_toc_comparison_false_positive() -> None:
    state = _complete_state()
    state.source_path = "测试桥报告（现A级）.docx"
    state.scores.previous.score = FieldValue()
    state.scores.previous.grade = FieldValue()
    parsed = ParsedDocument(
        path=Path("sample.docx"),
        paragraphs=[
            ParagraphBlock(0, "上一次检测状况 3  检测结果 4", "目录"),
            ParagraphBlock(1, "本桥无往年检测报告，故无上一次检测报告。", "正文"),
        ],
        tables=[],
    )

    # The broad temporal detector deliberately recovers non-standard prose,
    # so a TOC can still be a hit.  The explicit current-only declaration is
    # the authoritative mode selector used by process().
    assert BridgeEvidenceOrchestrator._source_has_temporal_comparison(state, parsed)
    assert BridgeEvidenceOrchestrator._source_declares_no_comparison(state, parsed)


def test_no_history_declaration_does_not_join_unrelated_report_phrases() -> None:
    state = _complete_state()
    state.source_path = "普通报告.docx"
    parsed = ParsedDocument(
        path=Path("sample.docx"),
        paragraphs=[
            ParagraphBlock(0, "本次未发现明显病害。", "正文"),
            ParagraphBlock(1, "上一次定检记录显示总体状况稳定。", "正文"),
        ],
        tables=[],
    )

    assert not BridgeEvidenceOrchestrator._source_declares_no_comparison(state, parsed)


def test_online_directory_run_requires_single_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRIDGE_EVIDENCE_SEMANTIC_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("BRIDGE_EVIDENCE_API_KEY", "test-only-digua-placeholder")
    runner = BridgeEvidenceOrchestrator(Settings(
        llm_mode="on",
        work_dir=tmp_path / "work",
        cache_dir=tmp_path / "cache",
    ))

    with pytest.raises(ValueError, match="workers=1"):
        runner.process_directory(tmp_path, tmp_path / "result", workers=2)


def test_directory_run_rejects_non_positive_workers(tmp_path: Path) -> None:
    runner = BridgeEvidenceOrchestrator(Settings(
        llm_mode="off",
        work_dir=tmp_path / "work",
        cache_dir=tmp_path / "cache",
    ))

    with pytest.raises(ValueError, match="大于等于 1"):
        runner.process_directory(tmp_path, tmp_path / "result", workers=0)


def test_doc_conversion_cache_is_shared_across_run_directories(tmp_path: Path) -> None:
    cache_dir = tmp_path / "shared-cache"
    first = BridgeEvidenceOrchestrator(Settings(
        llm_mode="off",
        work_dir=tmp_path / "run-a",
        cache_dir=cache_dir,
    ))
    second = BridgeEvidenceOrchestrator(Settings(
        llm_mode="off",
        work_dir=tmp_path / "run-b",
        cache_dir=cache_dir,
    ))

    assert first.adapter.convert_dir == cache_dir / "converted"
    assert second.adapter.convert_dir == first.adapter.convert_dir
