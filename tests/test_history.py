from __future__ import annotations

# Hand-constructed fixtures use synthetic bridge and organization names.

from dataclasses import replace
from pathlib import Path

import pytest

import bridge_evidence.history.index as history_module
from bridge_evidence.evaluation.labels import LabelReport
from bridge_evidence.history.index import (
    HistoryIndex,
    HistoryRecord,
    _name_from_label_path,
)
from bridge_evidence.history_semantics import development_level, previous_presence
from bridge_evidence.models.schema import Disease, DocumentState, FieldValue
from bridge_evidence.parsers.docx import ParagraphBlock, ParsedDocument


def _record(
    year: int,
    *,
    name: str = "K10+000测试桥",
    score: str = "80.30",
    grade: str = "C级",
) -> HistoryRecord:
    label = LabelReport(
        path=Path(f"/released-labels/{year}年/{name}-信息提取报告.docx"),
        summary={"桥梁名称": name, "报告日期": f"{year}年6月", "总体评分": score, "总体等级": grade},
        recommendations=[],
        diseases=[],
    )
    return HistoryRecord(
        label=label,
        bridge_name=name,
        cycle_year=year,
        report_date=f"{year}年6月",
        overall_score=score,
        overall_grade=grade,
        diseases=[
            Disease(
                sequence="1",
                component="主梁",
                location="左幅1#梁跨中",
                type="裂缝",
                description="左幅1#梁跨中纵向裂缝，长1.0m",
                confidence=1.0,
            )
        ],
    )


def _state() -> DocumentState:
    state = DocumentState(document_id="sha256:history", source_path="current.docx")
    state.bridge.name = FieldValue(value="K10+000测试桥", evidence=["SOURCE"], confidence=1.0)
    state.bridge.report_date = FieldValue(value="2014年8月", evidence=["SOURCE"], confidence=1.0)
    state.scores.overall.score = FieldValue(value=85.3, evidence=["SOURCE"], confidence=1.0)
    state.diseases = [
        Disease(
            sequence="1",
            component="主梁",
            location="左幅1#梁跨中",
            type="裂缝",
            description="左幅1#梁跨中纵向裂缝，长1.2m",
            evidence=["SOURCE"],
            confidence=1.0,
        ),
        Disease(
            sequence="2",
            component="桥台",
            location="0#桥台",
            type="破损",
            description="0#桥台局部破损",
            evidence=["SOURCE"],
            confidence=1.0,
        ),
    ]
    return state


def _parsed_with_prior(year: int = 2012) -> ParsedDocument:
    return ParsedDocument(
        path=Path("current.docx"),
        paragraphs=[ParagraphBlock(index=0, text=f"本桥上次检测时间为{year}年12月，本次与上次病害进行对比。", style="正文")],
        tables=[],
    )


def test_history_index_never_reads_current_or_future_cycle() -> None:
    index = HistoryIndex(
        [
            _record(2012, score="80.30", grade="C级"),
            _record(2014, score="99.99", grade="A级"),
            _record(2016, score="98.00", grade="A级"),
        ]
    )
    state = _state()
    parsed = _parsed_with_prior()

    result = index.enrich(state, parsed, Path("/inputs/2014年/current.docx"))

    assert result["matched"] is True
    assert result["history_year"] == 2012
    assert state.scores.previous.score.value == 80.30
    assert state.scores.previous.grade.value == "C级"
    assert all("/2014年/" not in evidence.source_path for evidence in state.evidence.values())
    assert all("/2016年/" not in evidence.source_path for evidence in state.evidence.values())


def test_history_index_preserves_dual_carriageway_score() -> None:
    index = HistoryIndex([_record(2012, score="84.57（左幅）/77.90（右幅）", grade="B级（左幅）/C级（右幅）")])
    state = _state()

    result = index.enrich(state, _parsed_with_prior(), Path("/inputs/2014年/current.docx"))

    assert result["matched"] is True
    assert state.scores.previous.score.value == "84.57（左幅）/77.90（右幅）"
    assert state.scores.previous.grade.value == "B级（左幅）/C级（右幅）"


def test_history_enrichment_uses_global_disease_matching_and_stable_evidence_id() -> None:
    index = HistoryIndex([_record(2012)])
    parsed = _parsed_with_prior()
    first = _state()
    second = _state()

    first_result = index.enrich(first, parsed, Path("/inputs/2014年/current.docx"))
    second_result = index.enrich(second, parsed, Path("/inputs/2014年/current.docx"))

    assert first_result["evidence_id"] == second_result["evidence_id"]
    assert first.diseases[0].is_new is False
    assert first.diseases[0].previous_state == "已发现"
    assert first.diseases[1].is_new is None
    assert first.diseases[1].previous_state == "未明确"
    assert any(evidence_id.startswith("HISTORY:") and ":D0" in evidence_id for evidence_id in first.diseases[0].evidence)
    assert not any(":ABSENCE:" in evidence_id for evidence_id in first.diseases[1].evidence)
    assert "总体评分上升5.00分" in first.narrative.development_trend


def test_history_semantics_use_competition_enums_without_treating_novelty_as_growth() -> None:
    assert previous_presence("上次未发现该病害") == "未发现"
    assert previous_presence("上次有纵向裂缝2条") == "已发现"
    assert development_level("新增部分病害，整体趋于稳定") == "无"
    assert development_level("原病害有所发展") == "轻微"
    assert development_level("裂缝严重加剧") == "严重"


def test_history_reuses_aggregate_family_during_large_granularity_shift() -> None:
    historical = replace(
        _record(2012),
        diseases=[
            Disease(
                sequence="1",
                component="主梁",
                location="右幅主梁",
                type="裂缝、破损露筋、渗水泛碱",
                description="右幅主梁存在裂缝、破损露筋及渗水泛碱",
            )
        ],
    )
    state = _state()
    state.diseases = [
        Disease(
            sequence=str(index + 1),
            component="主梁",
            location=f"右幅{index + 1}#梁",
            type="纵向裂缝",
            description=f"右幅{index + 1}#梁纵向裂缝",
        )
        for index in range(20)
    ]

    result = HistoryIndex([historical]).enrich(
        state,
        _parsed_with_prior(),
        Path("/inputs/2014年/current.docx"),
    )

    assert result["disease_group_matches"] >= 19
    assert all(item.is_new is False for item in state.diseases)
    assert all(item.previous_state == "已发现" for item in state.diseases)


def test_history_never_overwrites_direct_comparison_fields() -> None:
    index = HistoryIndex([_record(2012)])
    state = _state()
    state.diseases[0].is_new = True
    state.diseases[0].previous_state = "原报告直接记载的上次状态"
    state.diseases[0].development = "原报告直接记载的发展程度"
    state.narrative.development_trend = "原报告直接给出的总体趋势。"
    parsed = _parsed_with_prior()

    index.enrich(state, parsed, Path("/inputs/2014年/current.docx"))

    assert state.diseases[0].is_new is True
    assert state.diseases[0].previous_state == "原报告直接记载的上次状态"
    assert state.diseases[0].development == "原报告直接记载的发展程度"
    assert state.narrative.development_trend == "原报告直接给出的总体趋势。"


def test_explicit_novelty_outranks_fuzzy_history_row_match() -> None:
    index = HistoryIndex([_record(2012)])
    state = _state()
    state.diseases[0].is_new = True

    index.enrich(state, _parsed_with_prior(), Path("/inputs/2014年/current.docx"))

    assert state.diseases[0].is_new is True
    assert state.diseases[0].previous_state == "未发现"


def test_novel_row_cannot_keep_descriptive_already_found_state() -> None:
    index = HistoryIndex([_record(2012)])
    state = _state()
    state.diseases[0].is_new = True
    state.diseases[0].previous_state = "上一次定检已发现类似病害"

    index.enrich(state, _parsed_with_prior(), Path("/inputs/2014年/current.docx"))

    assert state.diseases[0].previous_state == "未发现"


def test_explicit_existing_flag_keeps_unmatched_history_fields_coherent() -> None:
    index = HistoryIndex([_record(2012)])
    state = _state()
    state.diseases[1].is_new = False

    index.enrich(state, _parsed_with_prior(), Path("/inputs/2014年/current.docx"))

    assert state.diseases[1].is_new is False
    assert state.diseases[1].previous_state == "已发现"


def test_history_route_code_conflict_is_rejected() -> None:
    index = HistoryIndex([_record(2012, name="K20+000测试桥")])
    state = _state()
    parsed = _parsed_with_prior()

    result = index.enrich(state, parsed, Path("/inputs/2014年/current.docx"))

    assert result["matched"] is False
    assert state.scores.previous.score.value is None


def test_history_low_similarity_candidate_is_rejected_without_error() -> None:
    index = HistoryIndex([_record(2012, name="完全不同的跨江大桥")])
    state = _state()
    parsed = _parsed_with_prior()

    result = index.enrich(state, parsed, Path("/inputs/2014年/current.docx"))

    assert result["matched"] is False
    assert state.evidence == {}


def test_history_accepts_distinctive_name_when_only_current_name_has_route_code() -> None:
    index = HistoryIndex([_record(2012, name="合成林中桥", score="80.30")])
    state = _state()
    state.bridge.name.value = "合成林中桥（K8+150）"
    parsed = _parsed_with_prior()

    result = index.enrich(state, parsed, Path("/inputs/2014年/current.docx"))

    assert result["matched"] is True
    assert state.scores.previous.score.value == 80.30


def test_history_rejects_conflicting_ramp_and_structure_identity() -> None:
    parsed = _parsed_with_prior()
    state = _state()
    state.bridge.name.value = "合成山立交E匝道"
    ramp_result = HistoryIndex([_record(2012, name="合成山立交D匝道")]).enrich(
        state, parsed, Path("/inputs/2014年/current.docx")
    )
    assert ramp_result["matched"] is False

    state = _state()
    state.bridge.name.value = "合成园区立交1号立交桥"
    type_result = HistoryIndex([_record(2013, name="合成园区立交1号旱桥")]).enrich(
        state, _parsed_with_prior(2013), Path("/inputs/2014年/current.docx")
    )
    assert type_result["matched"] is False


def test_history_requires_comparison_evidence_from_current_report() -> None:
    index = HistoryIndex([_record(2012)])
    state = _state()
    parsed = ParsedDocument(path=Path("current.docx"), paragraphs=[], tables=[])

    result = index.enrich(state, parsed, Path("/inputs/2014年/current.docx"))

    assert result["matched"] is False
    assert result["reason"] == "当前报告无历史对比信号"
    assert state.scores.previous.score.value is None


def test_explicit_prior_year_beats_newer_history_record() -> None:
    index = HistoryIndex([_record(2012, score="74.34"), _record(2013, score="90.78")])
    state = _state()

    result = index.enrich(state, _parsed_with_prior(2012), Path("/inputs/2014年/current.docx"))

    assert result["matched"] is True
    assert result["history_year"] == 2012
    assert state.scores.previous.score.value == 74.34


def test_prior_year_can_match_report_year_when_folder_cycle_is_earlier() -> None:
    historical = replace(_record(2012, score="97.25"), report_date="2013年10月")
    index = HistoryIndex([historical])
    state = _state()

    result = index.enrich(state, _parsed_with_prior(2013), Path("/inputs/2014年/current.docx"))

    assert result["matched"] is True
    assert state.scores.previous.score.value == 97.25


def test_missing_label_bridge_name_can_be_recovered_from_filename() -> None:
    path = Path("/labels/2012年/201310-027合成学校立交桥-无对比年度的信息提取报告.docx")
    assert _name_from_label_path(path) == "合成学校立交桥"


def test_history_loader_does_not_parse_current_or_future_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = []
    for year in (2012, 2013, 2014, 2016):
        directory = tmp_path / f"{year}年"
        directory.mkdir()
        path = directory / f"测试桥-{year}-信息提取报告.docx"
        path.write_bytes(b"not-read-by-test")
        paths.append(path)
    parsed_paths: list[Path] = []

    def fake_parse_label(path: Path, _adapter) -> LabelReport:
        parsed_paths.append(path)
        year = int(path.parent.name[:4])
        return LabelReport(
            path=path,
            summary={"桥梁名称": f"测试桥{year}", "报告日期": f"{year}年", "总体评分": "80", "总体等级": "B级"},
            recommendations=[],
            diseases=[],
        )

    monkeypatch.setattr(history_module, "parse_label", fake_parse_label)

    index = HistoryIndex.from_directory(tmp_path, object(), before_year=2014)

    assert {record.cycle_year for record in index.records} == {2012, 2013}
    assert parsed_paths == paths[:2]
