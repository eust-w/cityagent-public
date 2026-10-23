from __future__ import annotations

# Hand-constructed fixtures use synthetic bridge and organization names.

from pathlib import Path

from bridge_evidence.evidence.extractors import EvidenceExtractor, normalize_bridge_name, normalize_component, normalize_date_text
from bridge_evidence.models.schema import ComparisonRecord, Disease, DocumentState, FieldValue
from bridge_evidence.parsers.docx import ParagraphBlock, ParsedDocument, TableBlock


def test_normalize_date_supports_cover_month_and_chinese_month() -> None:
    assert normalize_date_text("2016年8月") == "2016年8月"
    assert normalize_date_text("二零一三年十月") == "2013年10月"
    assert normalize_date_text("二○一五年五月") == "2015年5月"


def test_normalize_date_supports_numeric_cover_date() -> None:
    assert normalize_date_text("2016.11.22") == "2016年11月22日"


def test_normalize_date_uses_inspection_range_completion_date() -> None:
    assert normalize_date_text("2014/7/28～7/29") == "2014年7月29日"
    assert normalize_date_text("2014年7月28日至7月29日") == "2014年7月29日"
    assert normalize_date_text("2014.12.31~2015.1.2") == "2015年1月2日"


def test_bridge_name_strips_project_prefix_but_keeps_bridge_identifier() -> None:
    value = "2016年结构设施检测项目（一标段）：车行桥 K1+600旱桥"
    assert normalize_bridge_name(value) == "K1+600旱桥"


def test_bridge_name_strips_legacy_catalogue_number_and_date_suffix() -> None:
    assert normalize_bridge_name("12-025合成甲桥立交DA1匝道桥") == "合成甲桥立交DA1匝道桥"
    assert normalize_bridge_name("025合成甲桥立交DA1匝道桥") == "合成甲桥立交DA1匝道桥"
    assert normalize_bridge_name("101-合成乙村大桥2014.11.22") == "合成乙村大桥"
    assert normalize_bridge_name("HRQL-202001-001合成跨江大桥") == "合成跨江大桥"
    # The city prefix is a normalization rule; the bridge name is synthetic.
    assert normalize_bridge_name("重庆市合成路无名桥(K33+500)") == "合成路无名桥(K33+500)"
    assert normalize_bridge_name("合成路扩宽改造工程1标合成河桥桥") == "合成路扩宽改造工程1标合成河桥"


def test_bridge_meta_recovers_authoritative_name_when_summary_schema_is_broken() -> None:
    parsed = ParsedDocument(
        path=Path("合成河大桥（K26+123.4）检测报告.docx"),
        paragraphs=[],
        tables=[TableBlock(index=0, rows=[
            ["工程名称", "合成河大桥", "委托日期", "2017.8"],
            ["检测 结论", "状态良好"],
        ])],
    )
    state = DocumentState(document_id="sha256:broken-summary", source_path=str(parsed.path))

    EvidenceExtractor()._extract_bridge_meta(parsed, parsed.path, state, None)

    assert state.bridge.name.value == "合成河大桥"
    assert state.bridge.name.evidence == ["T0R0"]
    assert state.bridge.name.source_priority == 95


def test_filename_grade_metadata_only_fills_missing_values() -> None:
    state = DocumentState(
        document_id="sha256:filename-grades",
        source_path="示例桥报告（原B级，现A级）.docx",
    )
    state.scores.overall.grade = FieldValue(
        value="B级",
        evidence=["P1"],
        source_priority=90,
        confidence=0.99,
    )

    EvidenceExtractor()._extract_filename_grades(state)

    assert state.scores.overall.grade.value == "B级"
    assert state.scores.previous.grade.value == "B级"
    assert state.scores.previous.grade.evidence == ["FILENAME"]


def test_report_date_prefers_cover_issue_month_over_inspection_date() -> None:
    parsed = ParsedDocument(
        path=Path("cover-date.docx"),
        paragraphs=[],
        tables=[
            TableBlock(index=0, rows=[["报告编号：synthetic2015-00001"]]),
            TableBlock(index=1, rows=[["合成水库大桥"], ["检测评估报告"]]),
            TableBlock(index=2, rows=[["合成检测机构"], ["二○一五年五月"]]),
            TableBlock(
                index=3,
                rows=[
                    ["工程名称", "合成水库大桥", "委托日期", "2014.11"],
                    ["委托单位", "某单位", "检测日期", "2014.11.14"],
                    ["检测结果", "总体状况良好"],
                ],
            ),
        ],
    )
    state = DocumentState(document_id="sha256:cover-date", source_path="cover-date.docx")

    EvidenceExtractor()._extract_bridge_meta(parsed, Path("cover-date.docx"), state, None)

    assert state.bridge.report_date.value == "2015年5月"
    assert state.bridge.report_date.evidence == ["T2R1"]


def test_disease_table_accepts_legacy_and_modern_headers() -> None:
    legacy = TableBlock(index=0, rows=[["位置", "病害种类", "具体位置"]])
    modern = TableBlock(index=1, rows=[["序号", "病害位置", "病害类型", "病害描述"]])
    unrelated = TableBlock(index=2, rows=[["构件名称", "损坏类型", "技术状况指数"]])
    location_in_description = TableBlock(index=3, rows=[["序号", "病害 类型", "病害描述", "照片编号"]])
    assert legacy.is_disease_table()
    assert modern.is_disease_table()
    assert location_in_description.is_disease_table()
    assert not unrelated.is_disease_table()


def test_photo_contact_sheet_rows_are_not_extracted_as_diseases() -> None:
    parsed = ParsedDocument(
        path=Path("photo-rows.docx"),
        paragraphs=[],
        tables=[
            TableBlock(
                index=0,
                rows=[
                    ["位置", "病害种类", "具体位置"],
                    ["车行道", "磨损", "靠近1#缝处桥面铺装磨损"],
                    ["照片2.1-1 桥面整体照", "照片2.1-2 栏杆良好", ""],
                    ["2#伸缩缝", "现状", ""],
                    ["全桥", "全景", ""],
                ],
            )
        ],
    )
    state = DocumentState(document_id="sha256:photo-rows", source_path="photo-rows.docx")

    EvidenceExtractor()._extract_diseases(parsed, state)

    assert len(state.diseases) == 1
    assert state.diseases[0].type == "磨损"


def test_local_defect_survives_mixed_good_state_wording() -> None:
    parsed = ParsedDocument(
        path=Path("mixed-condition.docx"),
        paragraphs=[],
        tables=[
            TableBlock(
                index=0,
                rows=[
                    ["序号", "位置", "病害种类", "病害情况"],
                    ["1", "墩台", "渗水痕迹", "墩、台总体外观状况良好，桥墩局部渗水痕迹"],
                ],
            )
        ],
    )
    state = DocumentState(document_id="sha256:mixed", source_path="mixed-condition.docx")

    EvidenceExtractor()._extract_diseases(parsed, state)

    assert len(state.diseases) == 1
    assert state.diseases[0].type == "渗水痕迹"
    assert state.diseases[0].location == "桥墩"
    assert "桥墩局部渗水痕迹" in state.diseases[0].description


def test_disease_location_prefers_precise_structural_description_prefix() -> None:
    parsed = ParsedDocument(
        path=Path("precise-location.docx"),
        paragraphs=[],
        tables=[
            TableBlock(
                index=0,
                rows=[
                    ["桥幅", "桥跨", "病害类型", "病害描述"],
                    ["右幅", "1#跨", "破损露筋", "1#跨1#墩上方右侧翼板破损露筋，L=0.3m"],
                ],
            )
        ],
    )
    state = DocumentState(document_id="sha256:location", source_path="location.docx")

    EvidenceExtractor()._extract_diseases(parsed, state)

    assert state.diseases[0].location == "1#跨1#墩上方右侧翼板"


def test_table_section_heading_beats_incidental_anchor_context_for_component() -> None:
    assert normalize_component(
        "左幅距2-1#墩8m处",
        "预应力锚头检查 表6.1-1 上部结构主梁检测结果表",
    ) == "主梁"


def test_nearest_table_section_heading_wins_over_older_context() -> None:
    assert normalize_component(
        "距1#缝36m处",
        "6.2 支座检测结果 表6.2-1 支座检测结果表 6.3 桥面铺装检测结果 表6.3-1 桥面铺装检测结果表",
    ) == "桥面铺装"


def test_section_comparison_is_not_copied_to_unrelated_disease_rows() -> None:
    state = DocumentState(document_id="sha256:history-scope", source_path="history.docx")
    state.comparisons.append(ComparisonRecord(
        section="上部结构",
        previous="主梁存在露筋",
        current="主梁仍有露筋",
        development="新增部分病害",
    ))
    state.diseases.append(Disease(sequence="1", component="主梁", location="2#梁底", type="渗水", description="2#梁底渗水"))

    EvidenceExtractor()._infer_disease_history(state)

    assert state.diseases[0].previous_state is None
    assert state.diseases[0].development is None
    assert state.diseases[0].is_new is None


def test_section_comparison_is_attached_as_temporal_review_evidence() -> None:
    state = DocumentState(document_id="sha256:history-context", source_path="history.docx")
    state.comparisons.append(ComparisonRecord(
        section="上部结构",
        previous="主梁存在露筋",
        current="主梁新增渗水",
        development="新增部分病害",
        evidence=["T7R2"],
    ))
    state.diseases.append(
        Disease(sequence="1", component="主梁", location="2#梁底", type="渗水", description="2#梁底渗水")
    )

    EvidenceExtractor()._infer_disease_history(state)

    assert "T7R2" in state.diseases[0].evidence


def test_section_level_no_new_is_not_broadcast_to_unrelated_rows() -> None:
    state = DocumentState(document_id="sha256:history-no-new-scope", source_path="history.docx")
    state.comparisons.append(
        ComparisonRecord(
            section="下部结构",
            previous="桥墩存在露筋",
            current="桥墩总体稳定",
            development="本次未见新增病害",
        )
    )
    state.diseases.append(
        Disease(sequence="1", component="桥台", location="0#台侧墙", type="渗水", description="0#台侧墙渗水")
    )

    EvidenceExtractor()._infer_disease_history(state)

    assert state.diseases[0].is_new is None
    assert state.diseases[0].novelty_source is None


def test_row_specific_no_new_is_retained_as_direct_report_evidence() -> None:
    state = DocumentState(document_id="sha256:history-no-new-row", source_path="history.docx")
    state.comparisons.append(
        ComparisonRecord(
            section="下部结构",
            previous="0#台侧墙已有渗水",
            current="0#台侧墙仍有渗水",
            development="0#台侧墙渗水未见新增",
        )
    )
    state.diseases.append(
        Disease(sequence="1", component="桥台", location="0#台侧墙", type="渗水", description="0#台侧墙渗水")
    )

    EvidenceExtractor()._infer_disease_history(state)

    assert state.diseases[0].is_new is False
    assert state.diseases[0].novelty_source == "current_report_no_new"


def test_generic_new_type_is_not_broadcast_across_same_type_rows() -> None:
    state = DocumentState(document_id="sha256:history-new-scope", source_path="history.docx")
    state.comparisons.append(
        ComparisonRecord(
            section="上部结构",
            previous="主梁存在裂缝",
            current="本次仍有裂缝",
            development="本次新增裂缝",
        )
    )
    state.diseases.extend([
        Disease(sequence="1", component="主梁", location="1#梁底", type="裂缝", description="1#梁底裂缝"),
        Disease(sequence="2", component="主梁", location="2#梁底", type="裂缝", description="2#梁底裂缝"),
    ])

    EvidenceExtractor()._infer_disease_history(state)

    assert all(item.is_new is None for item in state.diseases)


def test_new_type_with_precise_location_marks_only_named_row() -> None:
    state = DocumentState(document_id="sha256:history-new-location", source_path="history.docx")
    state.comparisons.append(
        ComparisonRecord(
            section="上部结构",
            previous="主梁存在裂缝",
            current="本次仍有裂缝",
            development="本次新增1#梁底裂缝",
        )
    )
    state.diseases.extend([
        Disease(sequence="1", component="主梁", location="1#梁底", type="裂缝", description="1#梁底裂缝"),
        Disease(sequence="2", component="主梁", location="2#梁底", type="裂缝", description="2#梁底裂缝"),
    ])

    EvidenceExtractor()._infer_disease_history(state)

    assert state.diseases[0].is_new is True
    assert state.diseases[0].novelty_source == "current_report_new"
    assert state.diseases[1].is_new is None


def test_new_disease_normalizes_contradictory_previous_state() -> None:
    state = DocumentState(document_id="sha256:history-coherence", source_path="history.docx")
    state.diseases.append(
        Disease(
            sequence="1",
            component="桥台",
            location="0#台侧墙",
            type="渗水",
            description="0#台侧墙新增渗水",
            is_new=True,
            previous_state="已发现",
        )
    )

    EvidenceExtractor()._cohere_disease_history(state)

    assert state.diseases[0].is_new is True
    assert state.diseases[0].previous_state == "未发现"


def test_aggregate_layer_drops_inactive_repairs_and_construction_residue() -> None:
    extractor = EvidenceExtractor()
    assert extractor._inactive_aggregate_record(
        Disease(sequence="1", component="主梁", location="梁底", type="裂缝修补", description="梁底裂缝修补现状")
    )
    assert extractor._inactive_aggregate_record(
        Disease(sequence="2", component="主梁", location="箱内", type="模板未拆", description="箱内模板未拆")
    )
    assert not extractor._inactive_aggregate_record(
        Disease(sequence="3", component="主梁", location="梁底", type="裂缝修补", description="梁底修补处再次开裂")
    )


def test_component_formula_uses_final_result_and_unscoped_aggregate() -> None:
    parsed = ParsedDocument(
        path=Path("formula.docx"),
        paragraphs=[
            ParagraphBlock(0, "左幅桥面系BCIm左=72.82", ""),
            ParagraphBlock(1, "右幅桥面系BCIm右=74.99", ""),
            ParagraphBlock(2, "桥面系BCIm=（BCIm左+BCIm右）/2=73.91", ""),
            ParagraphBlock(3, "上部结构评分BCIs=（BCIs1+BCIs2+BCIs3）/3=71.81", ""),
            ParagraphBlock(4, "全桥BCIx=（BCIqt0+BCIqd1）/2=92.88", ""),
        ],
        tables=[],
    )
    state = DocumentState(document_id="sha256:test", source_path="formula.docx")
    extractor = EvidenceExtractor()
    extractor._extract_component_score_paragraphs(parsed, state)

    assert state.scores.deck.score.value == 73.91
    assert state.scores.superstructure.score.value == 71.81
    assert state.scores.substructure.score.value == 92.88


def test_cjj_2017_single_bridge_prefers_final_unscoped_component_score() -> None:
    parsed = ParsedDocument(
        path=Path("modern.docx"),
        paragraphs=[
            ParagraphBlock(0, "《城市桥梁养护技术标准》CJJ 99-2017", ""),
            ParagraphBlock(1, "左幅桥面系BCIm左=72.82", ""),
            ParagraphBlock(2, "右幅桥面系BCIm右=74.99", ""),
            ParagraphBlock(3, "全桥桥面系BCIm=88.53", ""),
        ],
        tables=[],
    )
    state = DocumentState(document_id="sha256:modern", source_path="modern.docx")

    EvidenceExtractor()._extract_component_score_paragraphs(parsed, state)

    assert state.scores.deck.score.value == 88.53


def test_cjj_2017_pedestrian_deck_uses_explicit_component_grade() -> None:
    parsed = ParsedDocument(
        path=Path("pedestrian.docx"),
        paragraphs=[
            ParagraphBlock(0, "人行天桥定期检测，依据CJJ 99-2017。", ""),
            ParagraphBlock(1, "桥面系BCIm为95.25，BSIm为85，桥面系评定为B级。", ""),
        ],
        tables=[],
    )
    state = DocumentState(document_id="sha256:pedestrian", source_path="pedestrian.docx")
    state.scores.deck.score.value = 95.25
    state.scores.deck.control_score.value = 85.0

    EvidenceExtractor()._resolve_component_grades(parsed, state)

    assert state.scores.deck.grade.value == "B级"


def test_cjj_2017_vehicle_deck_defaults_to_bci_grade_without_critical_summary() -> None:
    """Formula-only control grades do not replace the published BCI grade."""
    parsed = ParsedDocument(
        path=Path("vehicle.docx"),
        paragraphs=[
            ParagraphBlock(0, "城市车行桥定期检测，依据CJJ 99-2017。", ""),
            ParagraphBlock(1, "桥面系BCIm为85.48，BSIm为51.61，桥面系评定为D级。", ""),
        ],
        tables=[],
    )
    state = DocumentState(document_id="sha256:vehicle", source_path="vehicle.docx")
    state.scores.deck.score.value = 85.48
    state.scores.deck.control_score.value = 51.61

    EvidenceExtractor()._resolve_component_grades(parsed, state)

    assert state.scores.deck.grade.value == "B级"


def test_side_overall_parser_does_not_treat_component_bsi_as_bci() -> None:
    text = (
        "左幅桥面系BSI=62.72，评定为D级；右幅桥面系BSI=62.72，评定为D级；"
        "该桥BCI=91.30，整体技术状况评定为A级。"
    )

    assert EvidenceExtractor._explicit_side_overall_scores(text) == {}


def test_side_overall_parser_preserves_explicit_dual_bci() -> None:
    text = "左幅全桥BCI=93.83，右幅全桥BCI=86.56。"

    assert EvidenceExtractor._explicit_side_overall_scores(text) == {
        "左幅": 93.83,
        "右幅": 86.56,
    }


def test_side_overall_parser_reads_parenthesized_values_after_bci() -> None:
    text = "审核后的桥梁技术状况指数BCI在80分以上（右幅82.71分，左幅82.77分）。"

    assert EvidenceExtractor._explicit_side_overall_scores(text) == {
        "左幅": 82.77,
        "右幅": 82.71,
    }


def test_final_rating_table_components_follow_selected_overall_side() -> None:
    state = DocumentState(document_id="sha256:align", source_path="align.docx")
    state.scores.overall.score = FieldValue(value=85.03)
    records = {
        "overall": [
            ("左幅", 88.11, None, "O-L"),
            ("右幅", 85.03, None, "O-R"),
        ],
        "deck": [
            ("左幅", 96.43, None, "D-L"),
            ("右幅", 80.20, None, "D-R"),
        ],
        "superstructure": [
            ("左幅", 75.0, None, "S-L"),
            ("右幅", 70.0, None, "S-R"),
        ],
        "substructure": [
            ("左幅", 97.0, None, "X-L"),
            ("右幅", 100.0, None, "X-R"),
        ],
    }

    EvidenceExtractor()._align_component_scores_with_selected_overall(records, state)

    assert state.scores.deck.score.value == 80.20
    assert state.scores.superstructure.score.value == 70.0
    assert state.scores.substructure.score.value == 100.0


def test_final_rating_table_preserves_components_when_overall_is_dual() -> None:
    state = DocumentState(document_id="sha256:dual", source_path="dual.docx")
    state.scores.overall.score = FieldValue(value="91.0（左幅）/89.0（右幅）")
    records = {
        "overall": [("左幅", 91.0, None, "O-L"), ("右幅", 89.0, None, "O-R")],
        "deck": [("左幅", 88.0, None, "D-L"), ("右幅", 77.0, None, "D-R")],
        "superstructure": [],
        "substructure": [],
    }

    EvidenceExtractor()._align_component_scores_with_selected_overall(records, state)

    assert state.scores.deck.score.value == "88（左幅）/77（右幅）"


def test_side_formula_paragraphs_do_not_reopen_selected_overall_score() -> None:
    parsed = ParsedDocument(
        path=Path("selected.docx"),
        paragraphs=[
            ParagraphBlock(0, "左幅整体BCI=93.68", ""),
            ParagraphBlock(1, "右幅整体BCI=91.82", ""),
        ],
        tables=[],
    )
    state = DocumentState(document_id="sha256:selected", source_path="selected.docx")
    state.scores.overall.score = FieldValue(value=91.82, source_priority=90)

    EvidenceExtractor()._extract_overall_score_paragraphs(parsed, state)

    assert state.scores.overall.score.value == 91.82


def test_summary_selected_overall_drives_same_side_rating_table_values() -> None:
    parsed = ParsedDocument(
        path=Path("selected-summary.docx"),
        paragraphs=[],
        tables=[
            TableBlock(
                index=0,
                rows=[
                    ["工程名称", "示例桥"],
                    ["检测结论", "示例桥的BCI=85.03，整体技术状况等级评定为B级。"],
                ],
            ),
            TableBlock(
                index=1,
                rows=[
                    ["部位名称", "技术状况指数", "权重", "BCI", "桥梁整体技术状况等级"],
                    ["桥面系", "96.43", "0.15", "88.11", "B"],
                    ["上部结构", "75.00", "0.40", "88.11", "B"],
                    ["下部结构", "97.00", "0.45", "88.11", "B"],
                ],
                context=["表8.4-1 示例桥左幅技术状况评定表"],
            ),
            TableBlock(
                index=2,
                rows=[
                    ["部位名称", "技术状况指数", "权重", "BCI", "桥梁整体技术状况等级"],
                    ["桥面系", "80.20", "0.15", "85.03", "B"],
                    ["上部结构", "70.00", "0.40", "85.03", "B"],
                    ["下部结构", "100.00", "0.45", "85.03", "B"],
                ],
                context=["表8.4-2 示例桥右幅技术状况评定表"],
            ),
        ],
    )
    state = DocumentState(document_id="sha256:selected-summary", source_path="selected-summary.docx")
    extractor = EvidenceExtractor()
    summary = extractor._extract_summary_table(parsed)

    extractor._extract_scores(parsed, state, summary)

    assert state.scores.overall.score.value == 85.03
    assert state.scores.deck.score.value == 80.20
    assert state.scores.superstructure.score.value == 70.0
    assert state.scores.substructure.score.value == 100.0


def test_legacy_shifted_rating_rows_use_score_grade_and_overall_columns() -> None:
    parsed = ParsedDocument(
        path=Path("shifted-rating.docx"),
        paragraphs=[],
        tables=[
            TableBlock(
                index=0,
                rows=[
                    ["部位名称", "技术状况指数", "技术状况", "权重", "BCI"],
                    ["桥面系", "BCIm", "88.53", "B", "0.15", "91.10"],
                    ["上部结构", "BCIs", "91.44", "A", "0.40", "91.10"],
                    ["下部结构", "BCIx", "91.61", "A", "0.45", "91.10"],
                ],
                context=["表6.4-1 桥梁技术状况评定表"],
            )
        ],
    )
    state = DocumentState(document_id="sha256:shifted", source_path="shifted-rating.docx")

    EvidenceExtractor()._extract_scores(parsed, state, None)

    assert state.scores.deck.score.value == 88.53
    assert state.scores.deck.grade.value == "B级"
    assert state.scores.overall.score.value == 91.1
    assert state.scores.overall.grade.value == "A级"


def test_swapped_weight_and_condition_index_row_rejects_weight_as_score() -> None:
    parsed = ParsedDocument(
        path=Path("swapped-weight.docx"),
        paragraphs=[],
        tables=[
            TableBlock(
                index=0,
                rows=[
                    ["部位名称", "技术状况指数", "权重", "BCI", "桥梁整体技术状况等级"],
                    ["桥面系", "0.15", "84.93", "88.88", "B"],
                    ["上部结构", "0.4", "79.89", "88.88", "B"],
                    ["下部结构", "0.45", "98.19", "88.88", "B"],
                ],
            )
        ],
    )
    state = DocumentState(document_id="sha256:swapped", source_path="swapped-weight.docx")

    EvidenceExtractor()._extract_scores(parsed, state, None)

    assert state.scores.deck.score.value == 84.93
    assert state.scores.superstructure.score.value == 79.89
    assert state.scores.substructure.score.value == 98.19
    assert state.scores.overall.score.value == 88.88


def test_main_beam_location_outranks_stale_cable_table_context() -> None:
    assert normalize_component("4#板距4#墩0.5m处梁底", "斜拉索病害表") == "主梁"
    assert normalize_component("空心板板间铰接缝底部", "拉索检查") == "主梁"
    assert normalize_component("距2#台6m处右幅4#板纵向裂缝", "预应力锚头检查") == "主梁"
    assert normalize_component("右幅第2跨1#梁右侧裂缝", "斜拉索检查") == "主梁"


def test_numbered_recommendation_accepts_chinese_closing_parenthesis() -> None:
    text = "1）加强对桥梁进行日常巡查和养护。"
    assert EvidenceExtractor._numbered_item(text)
    assert EvidenceExtractor._strip_item_number(text) == "加强对桥梁进行日常巡查和养护。"
    combined = "1）按照DB50/231-2006）要求巡查。 2）及时修复主梁裂缝。"
    assert EvidenceExtractor._split_numbered_items(combined) == [
        "按照DB50/231-2006）要求巡查。",
        "及时修复主梁裂缝。",
    ]


def test_previous_score_accepts_explicit_unheaded_history_sentence() -> None:
    parsed = ParsedDocument(
        path=Path("history.docx"),
        paragraphs=[
            ParagraphBlock(0, "本次检查全桥BCI=91.20，评定为A级。", ""),
            ParagraphBlock(1, "上次检测时间是2014年11月，桥整体技术状况指数BCI=83.88，整体技术状况等级评定为B级。", ""),
        ],
        tables=[],
    )
    state = DocumentState(document_id="sha256:history", source_path="history.docx")

    EvidenceExtractor()._extract_previous_score(parsed, state)

    assert state.scores.previous.score.value == 83.88
    assert state.scores.previous.grade.value == "B级"


def test_dedicated_previous_section_outranks_older_archive_ledger() -> None:
    parsed = ParsedDocument(
        path=Path("history-priority.docx"),
        paragraphs=[
            ParagraphBlock(0, "1.2 上一次检测状况", "Heading 2"),
            ParagraphBlock(1, "该桥于2014年11月完成上一次检查。", ""),
            ParagraphBlock(2, "桥梁BCI=87.74，整体技术状况等级评定为B级。", ""),
            ParagraphBlock(3, "2 检测目的", "Heading 1"),
        ],
        tables=[TableBlock(index=0, rows=[
            ["检测维修日期", "检测维修内容", "单位", "质量状况"],
            ["2014年11月至2015年5月", "外观检查", "甲", "桥整体技术状况指数BCI=80.25，评定为B级。"],
        ])],
    )
    state = DocumentState(document_id="sha256:history-priority", source_path="history-priority.docx")

    EvidenceExtractor()._extract_previous_score(parsed, state)

    assert state.scores.previous.score.value == 87.74
    assert state.scores.previous.grade.value == "B级"


def test_recommendation_section_accepts_unnumbered_action_paragraphs() -> None:
    parsed = ParsedDocument(
        path=Path("recommendations.docx"),
        paragraphs=[
            ParagraphBlock(0, "5.2 养护建议", "Heading 2"),
            ParagraphBlock(1, "针对上述病害提出以下几点建议。", ""),
            ParagraphBlock(2, "建议立即修补主梁裂缝。", ""),
            ParagraphBlock(3, "可采用耐久性材料处理桥面破损。", ""),
            ParagraphBlock(4, "加强日常巡查和定期观测。", ""),
            ParagraphBlock(5, "加强桥梁日常观测巡查、清洁保养工作。", ""),
            ParagraphBlock(6, "附表1 病害照片", ""),
        ],
        tables=[],
    )
    state = DocumentState(document_id="sha256:recommendations", source_path="recommendations.docx")

    EvidenceExtractor()._extract_recommendations(parsed, state, None)

    assert [item.content for item in state.recommendations] == [
        "建议立即修补主梁裂缝。",
        "可采用耐久性材料处理桥面破损。",
        "加强日常巡查和定期观测。",
        "加强桥梁日常观测巡查、清洁保养工作。",
    ]


def test_recommendation_section_rejects_intro_sentence() -> None:
    parsed = ParsedDocument(
        path=Path("recommendation-intro.docx"),
        paragraphs=[
            ParagraphBlock(0, "5.2 养护建议", "Heading 2"),
            ParagraphBlock(1, "针对桥梁现有病害，建议如下：", ""),
            ParagraphBlock(2, "建议及时修补主梁裂缝。", ""),
        ],
        tables=[],
    )
    state = DocumentState(document_id="sha256:recommendation-intro", source_path="recommendation-intro.docx")

    EvidenceExtractor()._extract_recommendations(parsed, state, None)

    assert [item.content for item in state.recommendations] == ["建议及时修补主梁裂缝。"]


def test_numbered_bare_recommendation_gets_explicit_action_semantics() -> None:
    parsed = ParsedDocument(
        path=Path("bare-recommendation.docx"),
        paragraphs=[
            ParagraphBlock(0, "5.2 养护建议", "Heading 2"),
            ParagraphBlock(1, "1）主梁裂缝封闭及表面防护。", ""),
        ],
        tables=[],
    )
    state = DocumentState(document_id="sha256:bare-recommendation", source_path="bare-recommendation.docx")

    EvidenceExtractor()._extract_recommendations(parsed, state, None)

    assert [item.content for item in state.recommendations] == ["建议主梁裂缝封闭及表面防护。"]


def test_large_photo_ledger_uses_evidence_triggered_issue_family_grain() -> None:
    parsed = ParsedDocument(
        path=Path("large.docx"),
        paragraphs=[ParagraphBlock(0, "本次检测新发现24处显著病害。", "")],
        tables=[],
    )
    state = DocumentState(document_id="sha256:large", source_path="large.docx")
    state.diseases = [
        Disease(
            sequence=str(index + 1),
            component="主梁",
            location=f"第{index + 1}处",
            type="纵向裂缝" if index % 2 else "横向开裂",
            description=f"第{index + 1}处裂缝",
            evidence=[f"T1R{index + 1}"],
        )
        for index in range(501)
    ]

    EvidenceExtractor()._aggregate_large_diseases(parsed, state)

    assert len(state.diseases) == 1
    assert state.diseases[0].type == "裂缝"
    assert len(state.diseases[0].evidence) == 501
    assert state.diseases[0].metrics == {"COUNT": "501"}
    assert "裂缝类病害共计501处" in state.diseases[0].description


def test_large_component_ledger_stays_granular_without_source_aggregate_semantics() -> None:
    parsed = ParsedDocument(path=Path("large-granular.docx"), paragraphs=[], tables=[])
    state = DocumentState(document_id="sha256:large-granular", source_path="large-granular.docx")
    state.diseases = [
        Disease(
            sequence=str(index + 1),
            component="斜拉索",
            location=f"第{index + 1}根",
            type="裂缝",
            description=f"第{index + 1}根拉索裂缝",
            evidence=[f"T1R{index + 1}"],
        )
        for index in range(501)
    ]

    EvidenceExtractor()._aggregate_large_diseases(parsed, state)

    assert len(state.diseases) == 501


def test_main_and_approach_large_ledgers_aggregate_by_exact_table_region() -> None:
    parsed = ParsedDocument(
        path=Path("main-approach.docx"),
        paragraphs=[],
        tables=[
            TableBlock(
                index=0,
                rows=[["序号", "病害位置", "病害类型", "病害描述"]],
                context=["表6.1-1 左幅主桥箱梁内检测结果表"],
            ),
            TableBlock(
                index=1,
                rows=[["序号", "病害位置", "病害类型", "病害描述"]],
                context=["表6.1-2 右幅引桥主梁检测结果表"],
            ),
        ],
    )
    state = DocumentState(document_id="sha256:main-approach", source_path="main-approach.docx")
    state.diseases = [
        Disease(
            sequence=str(index + 1),
            component="主梁",
            location=f"第{index + 1}处",
            type="裂缝",
            description=f"第{index + 1}处裂缝",
            evidence=[f"T{index % 2}R{index + 1}"],
        )
        for index in range(102)
    ]

    EvidenceExtractor()._aggregate_large_diseases(parsed, state)

    assert {item.location for item in state.diseases} == {"左幅主桥箱梁内", "右幅引桥主梁"}


def test_large_ledger_uses_final_source_disease_summary_as_issue_layer() -> None:
    parsed = ParsedDocument(
        path=Path("summary-overlay.docx"),
        paragraphs=[
            ParagraphBlock(0, "12.1 病害总结", "Heading 2", table_depth=0),
            ParagraphBlock(1, "通过本次检测，主要病害统计如下：", "", table_depth=0),
            ParagraphBlock(
                2,
                "主桥主梁：左幅箱梁内存在11处破损、露筋，总面积为1.0m2；存在3处渗水、泛碱。",
                "",
                table_depth=0,
            ),
            ParagraphBlock(3, "12.2 养护建议", "Heading 2", table_depth=0),
        ],
        tables=[
            TableBlock(
                index=0,
                rows=[["序号", "病害位置", "病害类型", "病害描述"]],
                context=["表6.1-1 左幅主桥箱梁内检测结果表"],
            ),
            TableBlock(
                index=1,
                rows=[["序号", "病害位置", "病害类型", "病害描述"]],
                context=["表6.1-2 右幅引桥主梁检测结果表"],
            ),
        ],
    )
    state = DocumentState(document_id="sha256:summary-overlay", source_path="summary-overlay.docx")
    state.diseases = [
        Disease(
            sequence=str(index + 1),
            component="主梁",
            location=f"第{index + 1}处",
            type="破损露筋" if index % 2 else "渗水泛碱",
            description=f"第{index + 1}处" + ("破损露筋" if index % 2 else "渗水泛碱"),
            evidence=[f"T{0 if index < 51 else 1}R{index + 1}"],
        )
        for index in range(102)
    ]

    EvidenceExtractor()._aggregate_large_diseases(parsed, state)

    summary_rows = [item for item in state.diseases if item.location == "左幅主桥箱梁内"]
    assert {item.type for item in summary_rows} == {"破损露筋", "渗水泛碱"}
    assert any("11处破损、露筋" in item.description for item in summary_rows)
    assert any("3处渗水、泛碱" in item.description for item in summary_rows)
    assert all("P2" in item.evidence for item in summary_rows)


def test_final_rating_and_chronological_history_override_stale_cover_summary() -> None:
    parsed = ParsedDocument(
        path=Path("timeline.docx"),
        paragraphs=[ParagraphBlock(0, "2019年11月20日", "")],
        tables=[
            TableBlock(index=0, rows=[
                ["工程名称", "示例桥"],
                ["检测结论", "主桥评分为74.8，技术等级评定为二类。"],
            ]),
            TableBlock(
                index=1,
                rows=[["项目", "权重"], ["综合评定分数 Dr=76.6", "等级", "二类"]],
                context=["桥梁技术状况评定结果", "技术状况评定表（主桥）"],
            ),
            TableBlock(
                index=2,
                rows=[["项目", "权重"], ["综合评定分数Dr=74", "等级", "二类"]],
                context=["技术状况评定表（引桥）"],
            ),
            TableBlock(index=3, rows=[
                ["检测维修日期", "检测维修内容", "单位", "质量状况"],
                ["2016年8月", "外观检测", "甲", "主桥评分为73.4，技术等级评定为二类；引桥评分为75.9。"],
                ["2018年12月", "外观检测", "甲", "主桥评分为77.6，技术等级评定为二类；引桥评分为70.6。"],
                ["2018年11月至12月", "外观检测", "乙", "主桥评分为74.8，技术等级评定为二类；引桥评分为71.8。"],
                ["2019年9月至10月", "外观检测", "乙", "主桥评分为76.6，技术等级评定为二类；引桥评分为74。"],
            ]),
        ],
    )

    state = EvidenceExtractor().extract(parsed, parsed.path, "timeline")

    assert state.scores.overall.score.value == 76.6
    assert state.scores.overall.grade.value == "二类"
    assert state.scores.previous.score.value == 74.8
    assert state.scores.previous.grade.value == "二类"
    assert state.narrative.development_trend.startswith("总体评分较上一次上升1.80分")


def test_bare_condition_state_row_is_preserved_but_photo_caption_is_not() -> None:
    parsed = ParsedDocument(
        path=Path("condition-state.docx"),
        paragraphs=[],
        tables=[TableBlock(index=0, rows=[
            ["序号", "病害位置", "病害类型", "病害描述"],
            ["1", "人行道", "现状", "外观良好"],
            ["2", "桥涵标", "桥涵标现状", "现场照片"],
        ])],
    )
    state = DocumentState(document_id="sha256:condition-state", source_path=str(parsed.path))

    EvidenceExtractor()._extract_diseases(parsed, state)

    assert [(item.location, item.type, item.description) for item in state.diseases] == [
        ("人行道", "现状", "外观良好")
    ]


def test_bare_repair_trace_is_not_active_disease_but_recracking_is_kept() -> None:
    parsed = ParsedDocument(
        path=Path("repair-traces.docx"),
        paragraphs=[],
        tables=[
            TableBlock(
                index=0,
                rows=[
                    ["序号", "病害位置", "病害类型", "病害描述"],
                    ["1", "路缘石", "修补痕迹", "距伸缩缝5m处修补痕迹"],
                    ["2", "主梁", "修补痕迹", "既有修补处再次开裂"],
                ],
            )
        ],
    )
    state = DocumentState(document_id="sha256:repair-traces", source_path="repair-traces.docx")

    EvidenceExtractor()._extract_diseases(parsed, state)

    assert len(state.diseases) == 1
    assert state.diseases[0].description == "既有修补处再次开裂"


def test_narrative_trend_supports_dual_carriageway_score_values() -> None:
    parsed = ParsedDocument(path=Path("dual.docx"), paragraphs=[], tables=[])
    state = DocumentState(document_id="sha256:dual", source_path="dual.docx")
    state.scores.previous.score = FieldValue(value="54.67（左幅）/86.9（右幅）")
    state.scores.overall.score = FieldValue(value="60.0（左幅）/90.0（右幅）")

    EvidenceExtractor()._extract_narratives(parsed, state, None)

    assert state.narrative.development_trend == "总体评分较上一次上升4.22分。"


def test_normalize_date_supports_spaced_and_fullwidth_digits() -> None:
    assert normalize_date_text("2 0 1 8 年 3 月 5 日") == "2018年3月5日"
    assert normalize_date_text("２０１８年１１月２２日") == "2018年11月22日"
    assert normalize_date_text("二〇一八年三月五日") == "2018年3月5日"
    assert normalize_date_text("二○一八年十一月") == "2018年11月"
    assert normalize_date_text("2018.05") == "2018年5月"
    assert normalize_date_text("2018-11") == "2018年11月"


def test_extract_summary_table_multi_column_and_expanded_labels() -> None:
    parsed = ParsedDocument(
        path=Path("summary.docx"),
        paragraphs=[],
        tables=[
            TableBlock(
                index=0,
                rows=[
                    ["工程名称", "合成跨河大桥", "报告日期", "2018年11月"],
                    ["检测依据", "CJJ 99-2017", "检测结论", "总体技术状况评定为B级，BCI=88.5"],
                ],
            )
        ],
    )
    state = DocumentState(document_id="sha256:summary", source_path="summary.docx")
    extractor = EvidenceExtractor()
    summary = extractor._extract_summary_table(parsed)
    assert summary is not None
    table, mapping, rows = summary
    assert mapping["工程名称"] == "合成跨河大桥"
    assert mapping["报告日期"] == "2018年11月"
    extractor._extract_bridge_meta(parsed, Path("summary.docx"), state, summary)
    assert state.bridge.name.value == "合成跨河大桥"
    assert state.bridge.report_date.value == "2018年11月"


def test_previous_score_from_comparison_table() -> None:
    parsed = ParsedDocument(
        path=Path("comparison.docx"),
        paragraphs=[],
        tables=[
            TableBlock(
                index=0,
                rows=[
                    ["部位", "上一次检测结果", "本次检测结果", "发展状况"],
                    ["全桥", "BCI=85.20，B级", "BCI=88.50，B级", "持平"],
                    ["上部结构", "良好", "良好", "稳定"],
                ],
            )
        ],
    )
    state = DocumentState(document_id="sha256:comp", source_path="comparison.docx")
    extractor = EvidenceExtractor()
    extractor._extract_comparisons(parsed, state)
    assert state.scores.previous.score.value == 85.20
    assert state.scores.previous.grade.value == "B级"
    assert len(state.comparisons) == 2
