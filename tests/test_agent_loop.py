from __future__ import annotations

from pathlib import Path

from bridge_evidence.models.agent import AgentCommand
from bridge_evidence.models.schema import Disease, DocumentState, EvidenceRef, FieldValue
from bridge_evidence.models.validation import ConstraintEngine
from bridge_evidence.parsers.docx import ParsedDocument
from bridge_evidence.runtime.registry import EvidenceToolRegistry


def _registry(*, complete: bool) -> tuple[DocumentState, EvidenceToolRegistry]:
    state = DocumentState(document_id="sha256:test", source_path="test.docx")
    state.add_evidence(EvidenceRef(id="E1", kind="paragraph", text="测试桥 BCI=90.1，A级", source_path="test.docx", priority=90))
    state.bridge.name = FieldValue(value="测试桥", evidence=["E1"], source_priority=90, confidence=1)
    if complete:
        state.bridge.report_date = FieldValue(value="2026年1月1日", evidence=["E1"], source_priority=90, confidence=1)
        state.scores.overall.score = FieldValue(value=90.1, evidence=["E1"], source_priority=90, confidence=1)
        state.scores.overall.grade = FieldValue(value="A级", evidence=["E1"], source_priority=90, confidence=1)
    parsed = ParsedDocument(path=Path("test.docx"), paragraphs=[], tables=[])
    return state, EvidenceToolRegistry(state, parsed, ConstraintEngine())


def test_fact_patch_rejects_unknown_evidence() -> None:
    state, registry = _registry(complete=True)
    result = registry.execute(AgentCommand(
        run_id="run:test",
        seq=1,
        action="apply_fact_patch",
        arguments={"patches": [{"field": "scores.overall.score", "value": 70.0, "evidence_ids": ["MISSING"]}]},
    ))
    assert not result.success
    assert state.scores.overall.score.value == 90.1


def test_previous_patch_rejects_current_score_from_long_conclusion() -> None:
    state, registry = _registry(complete=True)
    state.scores.previous.score = FieldValue(
        value=86.05,
        evidence=["HISTORY:2014:bridge"],
        source_priority=76,
        confidence=0.98,
    )
    state.add_evidence(EvidenceRef(
        id="HISTORY:2014:bridge",
        kind="derived",
        text="2014年历史标签：总体评分=86.05，总体等级=B级。",
        source_path="history.docx",
        priority=76,
    ))
    state.add_evidence(EvidenceRef(
        id="T0R9",
        kind="table",
        text="上一次所检测的病害已修复；本次技术状况指数BCI为94.8，评定为A级。",
        source_path="test.docx",
        priority=90,
    ))

    result = registry.execute(AgentCommand(
        run_id="run:test",
        seq=1,
        action="apply_fact_patch",
        arguments={"patches": [{
            "field": "scores.previous.score",
            "value": 94.8,
            "evidence_ids": ["T0R9"],
        }]},
    ))

    assert result.success is False
    assert "同一语义分句" in (result.error or "")
    assert state.scores.previous.score.value == 86.05


def test_previous_patch_accepts_explicit_prior_clause() -> None:
    state, registry = _registry(complete=True)
    state.add_evidence(EvidenceRef(
        id="P1",
        kind="paragraph",
        text="上一次总体评分为86.05，评定为B级；本次总体评分为94.8，评定为A级。",
        source_path="test.docx",
        priority=90,
    ))

    result = registry.execute(AgentCommand(
        run_id="run:test",
        seq=1,
        action="apply_fact_patch",
        arguments={"patches": [{
            "field": "scores.previous.score",
            "value": 86.05,
            "evidence_ids": ["P1"],
        }]},
    ))

    assert result.success
    assert state.scores.previous.score.value == 86.05


def test_finalize_is_a_constraint_gate() -> None:
    _, registry = _registry(complete=False)
    result = registry.execute(AgentCommand(run_id="run:test", seq=1, action="finalize"))
    assert not result.success
    assert not registry.finalized


def test_finalize_succeeds_only_after_validation() -> None:
    _, registry = _registry(complete=True)
    result = registry.execute(AgentCommand(run_id="run:test", seq=1, action="finalize"))
    assert result.success
    assert registry.finalized


def test_mark_unavailable_requires_exhaustive_score_search() -> None:
    state, registry = _registry(complete=True)
    state.scores.overall.score = FieldValue()
    state.scores.overall.grade = FieldValue()
    premature = registry.execute(AgentCommand(
        run_id="run:test",
        seq=1,
        action="mark_unavailable",
        arguments={"scope": "overall"},
    ))
    assert not premature.success

    searched = registry.execute(AgentCommand(
        run_id="run:test",
        seq=2,
        action="list_score_candidates",
        arguments={"scope": "overall"},
    ))
    marked = registry.execute(AgentCommand(
        run_id="run:test",
        seq=3,
        action="mark_unavailable",
        arguments={"scope": "overall"},
    ))
    assert searched.success and searched.content["candidates"] == []
    assert marked.success
    assert state.scores.overall.score.value == "无"
    assert state.scores.overall.score.evidence == ["D-ABSENT-OVERALL"]


def test_normalize_diseases_can_prune_photo_rows_without_model() -> None:
    state, registry = _registry(complete=True)
    state.add_evidence(EvidenceRef(
        id="T7R7",
        kind="table",
        text="照片2.1-1 桥面整体照 | 照片2.1-2 栏杆良好 | ",
        source_path="test.docx",
        priority=80,
        table_index=7,
        row_index=7,
    ))
    state.diseases.append(Disease(
        sequence="7",
        component="桥面铺装",
        location="照片2.1-1 桥面整体照",
        type="照片2.1-2 栏杆良好",
        description="",
        evidence=["T7R7"],
    ))

    result = registry.execute(AgentCommand(run_id="run:test", seq=1, action="normalize_diseases"))

    assert result.success
    assert result.content["pruned_photo_rows"] == 1
    assert state.diseases == []


def test_normalize_diseases_prunes_nonempty_presentation_label() -> None:
    state, registry = _registry(complete=True)
    state.diseases.append(Disease(
        sequence="8",
        component="其他",
        location="桥梁全景",
        type="工作照",
        description="检测人员现场工作照片",
        evidence=["E1"],
    ))

    result = registry.execute(AgentCommand(run_id="run:test", seq=1, action="normalize_diseases"))

    assert result.success
    assert result.content["pruned_photo_rows"] == 1
    assert state.diseases == []


def test_normalize_diseases_preserves_bare_condition_state_row() -> None:
    state, registry = _registry(complete=True)
    state.diseases.append(Disease(
        sequence="8",
        component="人行道",
        location="人行道",
        type="现状",
        description="外观良好",
        evidence=["E1"],
    ))

    result = registry.execute(AgentCommand(run_id="run:test", seq=1, action="normalize_diseases"))

    assert result.success
    assert result.content["pruned_photo_rows"] == 0
    assert [(item.type, item.description) for item in state.diseases] == [("现状", "外观良好")]


def test_normalize_diseases_prunes_duplicated_status_caption() -> None:
    state, registry = _registry(complete=True)
    history_evidence_id = "HISTORY:2018:bridge:D8"
    state.add_evidence(EvidenceRef(
        id=history_evidence_id,
        kind="derived",
        text="2018年历史病害：部位=JS-8；类型=现状现状；描述=JS-8距下锚头0.1m处套筒现状。",
        source_path="history.docx",
        priority=76,
    ))
    state.diseases.append(Disease(
        sequence="8",
        component="拉索",
        location="JS-8",
        type="现状现状",
        description="JS-8距下锚头0.1m处套筒现状",
        evidence=["E1", history_evidence_id],
    ))

    result = registry.execute(AgentCommand(run_id="run:test", seq=1, action="normalize_diseases"))

    assert result.success
    assert result.content["pruned_photo_rows"] == 1
    assert result.content["history_repaired_rows"] == 0
    assert state.diseases == []


def test_normalize_diseases_recovers_real_defect_from_presentation_label() -> None:
    state, registry = _registry(complete=True)
    state.diseases.append(Disease(
        sequence="9",
        component="主梁",
        location="梁底",
        type="修复现状",
        description="梁底修补处再次开裂",
        evidence=["E1"],
    ))

    result = registry.execute(AgentCommand(run_id="run:test", seq=1, action="normalize_diseases"))

    assert result.success
    assert state.diseases[0].type == "裂缝"


def test_normalize_diseases_recovers_placeholder_from_exact_history() -> None:
    state, registry = _registry(complete=True)
    evidence_id = "HISTORY:2014:bridge:D3"
    state.add_evidence(EvidenceRef(
        id=evidence_id,
        kind="derived",
        text="2014年历史病害：部位=左幅2#伸缩缝；类型=局部堵塞；描述=左幅2#伸缩缝局部堵塞。",
        source_path="history.docx",
        priority=76,
    ))
    state.diseases.append(Disease(
        sequence="2",
        component="伸缩缝",
        location="左幅2#伸缩缝",
        type="现状",
        description="左幅2#伸缩缝现状",
        is_new=False,
        novelty_source="history_exact",
        previous_state="已发现",
        development="无",
        evidence=["E1", evidence_id],
    ))

    result = registry.execute(AgentCommand(run_id="run:test", seq=1, action="normalize_diseases"))

    assert result.success
    assert result.content["history_repaired_rows"] == 1
    assert result.content["pruned_photo_rows"] == 0
    assert state.diseases[0].type == "局部堵塞"
    assert state.diseases[0].description == "左幅2#伸缩缝局部堵塞"
