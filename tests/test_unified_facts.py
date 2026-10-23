"""Evidence/identity invariants independent of model outputs and scoring labels."""

from copy import deepcopy

import pytest


def record(year=2020, width="最大宽度0.3mm", digest="a" * 64):
    return {
        "status": "success",
        "document": {"sha256": digest, "relative_path": "桥.docx"},
        "final_data": {
            "基本信息": {
                "桥梁名称": "测试桥",
                "报告日期": f"{year + 1}年1月",
                "检测日期": f"{year}年12月",
                "总体评分": "88",
            },
            "病害明细": [
                {"病害部位": "左幅1跨3号桥墩", "病害类型": "裂缝", "病害描述": width}
            ],
            "维修建议": [],
        },
        "provenance": [
            {"section": "病害明细", "number": 1, "source_id": "p0", "call_id": "call-1"}
        ],
        "snapshot": {
            "regions": [
                {
                    "source_id": "p0",
                    "text": width,
                    "location": {"source_path": "word/document.xml#/p[1]"},
                }
            ]
        },
        "model_calls": [
            {
                "call_id": "call-1",
                "status": "success",
                "backend_request_id": "req-1",
                "source_ranges": ["p0:0:" + str(len(width))],
            }
        ],
    }


def test_fact_graph_is_detached_versioned_and_preserves_duplicate_source_instances():
    from bridge_evidence.contest.facts import build_fact_graph

    source = record()
    source["final_data"]["病害明细"] *= 2
    source["provenance"].append(
        {**source["provenance"][0], "number": 2, "source_id": "p1"}
    )
    graph = build_fact_graph(source)
    data = graph.to_dict()
    defects = [o for o in data["observations"] if o["kind"] == "defect"]
    assert len(defects) == 2 and defects[0]["id"] != defects[1]["id"]
    assert data["schema_version"] == "cityagent-facts-v1"
    assert data["document"]["inspection_time"] == "2020年12月"
    assert data["document"]["report_time"] == "2021年1月"
    data["observations"].clear()
    source["final_data"]["病害明细"].clear()
    assert len(graph.to_dict()["observations"]) == 3


def test_identity_preserves_missing_fields_and_conflicting_numbers():
    from bridge_evidence.contest.facts import normalize_identity

    left = normalize_identity("测试桥", "左幅第1跨P3墩")
    assert left["span"] == "1" and left["pier"] == "3" and left["side"] == "left"
    assert normalize_identity("测试桥", "梁")["span"] is None
    assert normalize_identity("测试桥", "右幅1跨3号桥墩")["side"] != left["side"]
    assert (
        normalize_identity("测试桥1号", "")["bridge"]
        != normalize_identity("测试桥2号", "")["bridge"]
    )


def test_measurements_normalize_ranges_units_and_do_not_compare_statistics():
    from bridge_evidence.contest.facts import parse_measurements

    first = parse_measurements("最大宽度0.03cm")[0]
    second = parse_measurements("宽度0.2～0.4mm")[0]
    assert (
        first.value == pytest.approx(0.3)
        and first.unit == "mm"
        and first.statistic == "maximum"
    )
    assert second.lower == 0.2 and second.upper == 0.4 and second.value is None


def test_strict_facts_reject_unbound_or_out_of_call_sources():
    from bridge_evidence.contest.facts import build_fact_graph

    source = record()
    build_fact_graph(source, strict=True, allowed_sha256={"a" * 64})
    for change in ("no-call", "wrong-source", "simulated"):
        bad = deepcopy(source)
        if change == "no-call":
            bad["model_calls"] = []
        elif change == "wrong-source":
            bad["model_calls"][0]["source_ranges"] = ["p9:0:5"]
        else:
            bad["model_calls"][0]["simulated"] = True
        with pytest.raises(ValueError, match="evidence|call|source"):
            build_fact_graph(bad, strict=True, allowed_sha256={"a" * 64})


def test_whole_evidence_has_no_invented_fine_location():
    from bridge_evidence.contest.facts import build_fact_graph

    source = record()
    source.pop("provenance")
    graph = build_fact_graph(source).to_dict()
    assert all(
        e["precision"] == "document" and not e["source_ids"] for e in graph["evidence"]
    )
    assert all(r["status"] == "uncertain" for r in graph["relations"])


def test_qa_measurement_query_requires_confirmed_identity_for_trend(tmp_path):
    from bridge_evidence.contest.report_qa import ReportIndex

    index = ReportIndex(tmp_path / "facts.sqlite")
    index.ingest(record(), allowed_sha256={"a" * 64})
    index.ingest(record(2021, "最大宽度0.05cm", "b" * 64), allowed_sha256={"b" * 64})
    answer = index.measurements(
        bridge="测试桥", attribute="width", minimum=0.4, unit="mm"
    )
    assert answer["count"] == 1 and answer["measurements"][0]["value"] == 0.5
    comparison = index.compare_measurements("测试桥", [2020, 2021], attribute="width")
    assert comparison["status"] == "insufficient-evidence"
    assert comparison["reason"] == "same-defect relation not confirmed"
    scores = index.scores(bridge="测试桥", year=2020)
    assert scores["count"] == 1 and scores["facts"][0]["value"] == "88"
    assert index.query(bridge="' OR 1=1 --")["count"] == 0


def test_matching_and_aggregate_fallback_reject_explicit_identity_conflicts():
    from bridge_evidence.evidence.disease_graph import DiseaseGraphBuilder
    from bridge_evidence.matching.disease_graph import DiseaseMatcher
    from bridge_evidence.models.schema import Disease

    left = Disease(
        component="支座", location="左幅1跨1号支座", type="裂缝", description="纵向裂缝"
    )
    right = left.model_copy(update={"location": "左幅9跨8号支座"})
    assert not DiseaseMatcher.compatible(left, right)
    assert DiseaseMatcher.group_score(left, right) == 0
    events = DiseaseGraphBuilder.build([right], "doc")
    DiseaseGraphBuilder.attach_aggregates([left], events)
    assert left.event_ids == []


def test_no_fuzzy_match_does_not_prove_historical_absence():
    from pathlib import Path

    from test_history import _parsed_with_prior, _record, _state

    from bridge_evidence.history.index import HistoryIndex

    state = _state()
    HistoryIndex([_record(2012)]).enrich(
        state, _parsed_with_prior(), Path("/inputs/2014年/current.docx")
    )
    assert state.diseases[1].is_new is None
    assert state.diseases[1].previous_state in (None, "未明确")
    assert not any(
        link.relation == "absence" for link in state.diseases[1].history_links
    )


def test_explicit_defect_id_supported_by_source_enables_comparison(tmp_path):
    from bridge_evidence.contest.report_qa import ReportIndex

    index = ReportIndex(tmp_path / "facts.sqlite")
    for year, width, digest in [(2020, "0.3", "a" * 64), (2021, "0.5", "b" * 64)]:
        source = record(
            year, f"{year}年检测 左幅1跨3号桥墩 病害编号 LF-001 最大宽度{width}mm", digest
        )
        source["final_data"]["病害明细"][0]["病害编号"] = "LF-001"
        index.ingest(source, allowed_sha256={digest}, strict=True)
    answer = index.compare_measurements(
        "测试桥", [2020, 2021], attribute="width", statistic="maximum"
    )
    assert answer["status"] == "supported"
    assert answer["changes"][0]["delta"] == pytest.approx(0.2)
    assert answer["changes"][0]["relation"]["status"] == "confirmed"
    assert len(answer["changes"][0]["relation"]["evidence_ids"]) == 2
    assert index.count(bridge="测试桥", basis="defect-identities")["count"] == 1
    assert index.count(bridge="测试桥", basis="source-occurrences")["count"] == 2


def test_matching_number_without_same_location_remains_uncertain(tmp_path):
    from bridge_evidence.contest.report_qa import ReportIndex

    index = ReportIndex(tmp_path / "facts.sqlite")
    for year, location, digest in [
        (2020, "左幅1跨3号桥墩", "a" * 64),
        (2021, "左幅2跨3号桥墩", "b" * 64),
    ]:
        source = record(year, "病害编号 LF-001 最大宽度0.3mm", digest)
        source["final_data"]["病害明细"][0].update(病害编号="LF-001", 病害部位=location)
        index.ingest(source, allowed_sha256={digest}, strict=True)
    assert (
        index.compare_measurements("测试桥", [2020, 2021], attribute="width")["status"]
        == "insufficient-evidence"
    )


def test_unbound_defect_identifier_cannot_support_identity(tmp_path):
    from bridge_evidence.contest.report_qa import ReportIndex

    index = ReportIndex(tmp_path / "facts.sqlite")
    source = record()
    source["final_data"]["病害明细"][0]["病害编号"] = "LF-001"
    index.ingest(source, allowed_sha256={"a" * 64})
    assert (
        index.count(bridge="测试桥", basis="defect-identities")["status"]
        == "insufficient-evidence"
    )


def test_relation_rejects_explicit_scope_conflict_without_equating_unknowns():
    from bridge_evidence.contest.facts import assess_relation

    assert assess_relation({"span": "1"}, {"span": "2"})["status"] == "rejected"
    assert assess_relation({"span": None}, {"span": None})["status"] == "uncertain"


def test_scores_do_not_assign_previous_score_to_current_inspection():
    from bridge_evidence.contest.facts import build_fact_graph

    source = record()
    source["final_data"]["基本信息"]["上一次总体评分"] = "90"
    scores = [
        o
        for o in build_fact_graph(source).to_dict()["observations"]
        if o["kind"] == "score"
    ]
    previous = next(o for o in scores if o["attribute"] == "上一次总体评分")
    assert previous["inspection_year"] is None
    assert previous["time_basis"] == "prior-cycle-unknown"


def test_strict_rejects_fabricated_source_range_and_quote():
    from bridge_evidence.contest.facts import build_fact_graph

    for patch in (
        {"source_ranges": [{"source_id": "p0", "start": 0, "end": 100000}]},
        {"quotes": {"病害描述": "fabricated"}},
    ):
        source = record()
        source["provenance"][0].update(patch)
        with pytest.raises(ValueError, match="range|quote"):
            build_fact_graph(source, strict=True, allowed_sha256={"a" * 64})


def test_facts_audit_detects_changed_final_tables():
    from bridge_evidence.contest.facts import build_fact_graph, validate_fact_graph

    source = record()
    source["facts"] = build_fact_graph(source).to_dict()
    assert validate_fact_graph(source) == []
    source["final_data"]["病害明细"][0]["病害描述"] = "篡改"
    assert validate_fact_graph(source)


def test_actual_index_ledger_required_and_opened_readonly(tmp_path):
    from bridge_evidence.contest.report_qa import validate_index_ledger
    from bridge_evidence.contest.timing import TimingLedger

    source = record()
    source["model_calls"][0]["prompt_sha256"] = "c" * 64
    path = tmp_path / "timing.sqlite"
    ledger = TimingLedger(path)
    ledger.start("call-1", role="extraction")
    with pytest.raises(ValueError, match="ledger"):
        validate_index_ledger(source, path)
    ledger.finish(
        "call-1",
        0.2,
        status="success",
        receipt={"prompt_sha256": "c" * 64, "backend_request_id": "req-1"},
    )
    validate_index_ledger(source, path)
    source["model_calls"][0]["prompt_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="ledger"):
        validate_index_ledger(source, path)
    with pytest.raises(ValueError):
        validate_index_ledger(source, tmp_path / "missing.sqlite")
    assert not (tmp_path / "missing.sqlite").exists()


def test_divide_provenance_adapter_binds_task_source_and_real_call():
    from bridge_evidence.contest.facts import build_fact_graph

    source = record()
    source["provenance"] = {
        "基本信息": {},
        "病害明细": [{"evidence_ids": ["E1"], "task_ids": ["task-1"]}],
        "维修建议": [],
    }
    source["evidence_manifest"] = {
        "units": [
            {
                "id": "E1",
                "text": "最大宽度0.3mm",
                "source_path": "word/document.xml#/p[1]",
            }
        ]
    }
    source["tasks"] = [
        {
            "id": "task-1",
            "accepted_call_id": "call-1",
            "evidence_ids": ["E1"],
            "spans": [{"coverage_id": "E1", "start": 0, "end": 10}],
        }
    ]
    source["model_calls"][0].pop("source_ranges")
    assert (
        build_fact_graph(source, strict=True, allowed_sha256={"a" * 64}).to_dict()[
            "evidence"
        ][0]["binding"]
        == "bound"
    )


def test_defect_identity_needs_location_supported_by_its_source(tmp_path):
    from bridge_evidence.contest.report_qa import ReportIndex

    index = ReportIndex(tmp_path / "facts.sqlite")
    source = record(width="右幅9跨8号桥墩 病害编号 LF-001 最大宽度0.3mm")
    source["final_data"]["病害明细"][0]["病害编号"] = "LF-001"
    index.ingest(source, allowed_sha256={"a" * 64})
    assert (
        index.count(bridge="测试桥", basis="defect-identities")["status"]
        == "insufficient-evidence"
    )


def test_call_partial_source_cannot_support_quote_from_unseen_tail():
    from bridge_evidence.contest.facts import build_fact_graph

    source = record(width="前半段；尾部隐藏内容")
    source["model_calls"][0]["source_ranges"] = ["p0:0:3"]
    source["provenance"][0]["source_ranges"] = [
        {"source_id": "p0", "start": 4, "end": 10}
    ]
    source["provenance"][0]["quotes"] = {"病害描述": "尾部隐藏内容"}
    with pytest.raises(ValueError, match="range|visible"):
        build_fact_graph(source, strict=True, allowed_sha256={"a" * 64})


def test_graph_rebuild_stable_after_sorted_json_roundtrip():
    import json

    from bridge_evidence.contest.facts import build_fact_graph, validate_fact_graph

    source = record()
    source["final_data"]["基本信息"].update(总体等级="B", 上部结构评分="90")
    source["facts"] = build_fact_graph(source).to_dict()
    assert validate_fact_graph(json.loads(json.dumps(source, sort_keys=True))) == []


def test_repair_preserves_original_facts_and_appends_derivation():
    from bridge_evidence.contest.facts import build_fact_graph
    from bridge_evidence.contest.run_state import json_digest

    source = record()
    old = {
        "field": "总体评分",
        "value": "80",
        "scope": {"year": 2020},
        "evidence": {"source_id": "p0", "start": 0, "end": 3},
        "call_id": "call-1",
    }
    source["basic_observations"] = [old]
    source["repair_evidence"] = [
        {
            "field": "总体评分",
            "value": "88",
            "scope": {"year": 2020},
            "evidence": {"source_id": "p0", "start": 0, "end": 3},
            "call_id": "repair",
            "supersedes": [json_digest(old)],
        }
    ]
    facts = build_fact_graph(source).to_dict()
    values = [obs["value"] for obs in facts["observations"] if obs["kind"] == "score"]
    assert values == ["80", "88"]
    derivation = next(
        d for d in facts["derivations"] if d["type"] == "conflict-resolution"
    )
    assert len(derivation["input_observation_ids"]) == 1
    assert len(derivation["output_observation_ids"]) == 1


def test_index_reingest_is_digest_idempotent_across_filename_change(tmp_path):
    from bridge_evidence.contest.report_qa import ReportIndex

    index = ReportIndex(tmp_path / "facts.sqlite")
    source = record()
    index.ingest(source, allowed_sha256={"a" * 64})
    source["document"]["relative_path"] = "renamed.docx"
    index.ingest(source, allowed_sha256={"a" * 64})
    assert index.query(bridge="测试桥")["count"] == 1
    assert index.measurements(bridge="测试桥")["count"] == 1


def test_index_migrates_legacy_fact_lookup_without_losing_rows(tmp_path):
    import json
    import sqlite3

    from bridge_evidence.contest.report_qa import ReportIndex

    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE facts (id TEXT PRIMARY KEY, document TEXT, bridge TEXT, year INTEGER, component TEXT, disease TEXT, source TEXT, fields TEXT)"
        )
        db.execute(
            "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?)",
            (
                "legacy",
                json.dumps({"sha256": "a" * 64}),
                "桥",
                2020,
                "梁",
                "裂缝",
                "{}",
                "{}",
            ),
        )
    index = ReportIndex(path)
    assert index.query(bridge="桥")["count"] == 1
    with index.connect() as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(facts)")}
        assert "document_sha256" in columns


def test_measurement_endpoint_units_are_not_silently_truncated():
    from bridge_evidence.contest.facts import parse_measurements

    value = parse_measurements("宽度0.2mm～0.04cm")[0]
    assert value.value is None
    assert value.lower == pytest.approx(0.2) and value.upper == pytest.approx(0.4)


def test_aggregate_span_range_is_not_misidentified_as_one_span():
    from bridge_evidence.contest.facts import normalize_identity

    assert normalize_identity("桥", "左幅1～3跨支座")["span"] is None


def test_component_entities_have_explicit_bridge_parent_and_missing_bridge_not_shared():
    from bridge_evidence.contest.facts import build_fact_graph

    source = record()
    facts = build_fact_graph(source).to_dict()
    component = next(e for e in facts["entities"] if e["kind"] == "component")
    parent = next(e for e in facts["entities"] if e["id"] == component["parent_id"])
    assert parent["kind"] == "bridge"
    source["final_data"]["基本信息"]["桥梁名称"] = None
    first = build_fact_graph(source).to_dict()
    source["document"]["sha256"] = "b" * 64
    second = build_fact_graph(source).to_dict()
    assert {e["id"] for e in first["entities"]}.isdisjoint(
        e["id"] for e in second["entities"]
    )


def test_unseen_source_tail_cannot_create_cross_report_defect_identity(tmp_path):
    from bridge_evidence.contest.facts import build_fact_graph
    from bridge_evidence.contest.report_qa import ReportIndex

    index = ReportIndex(tmp_path / "facts.sqlite")
    for year, digest in ((2020, "a" * 64), (2021, "b" * 64)):
        visible = "最大宽度0.3mm"
        source = record(year, visible, digest)
        source["snapshot"]["regions"][0]["text"] += "；左幅1跨3号桥墩 病害编号 LF-001"
        source["final_data"]["病害明细"][0]["病害编号"] = "LF-001"
        source["provenance"][0]["source_ranges"] = [
            {"source_id": "p0", "start": 0, "end": len(visible)}
        ]
        graph = build_fact_graph(source, strict=True, allowed_sha256={digest}).to_dict()
        assert graph["defect_identities"] == []
        index.ingest(source, allowed_sha256={digest}, strict=True)
    assert (
        index.compare_measurements("测试桥", [2020, 2021], attribute="width")["status"]
        == "insufficient-evidence"
    )


@pytest.mark.parametrize(
    "actual_identifier", ["LF-0010", "XLF-001", "LF-001.1", "LF-001-2"]
)
def test_defect_identifier_is_complete_token_not_source_substring(actual_identifier):
    from bridge_evidence.contest.facts import build_fact_graph

    source = record(width=f"左幅1跨3号桥墩 病害编号 {actual_identifier} 最大宽度0.3mm")
    source["final_data"]["病害明细"][0]["病害编号"] = "LF-001"
    graph = build_fact_graph(source, strict=True, allowed_sha256={"a" * 64}).to_dict()
    assert graph["defect_identities"] == []


def test_historical_defects_and_recommendations_keep_source_years_in_qa(tmp_path):
    from bridge_evidence.contest.facts import build_fact_graph
    from bridge_evidence.contest.report_qa import ReportIndex

    source = record()
    source["final_data"]["病害明细"] = []
    source["provenance"] = []
    source["snapshot"]["regions"] = []
    source["model_calls"][0]["source_ranges"] = []
    for section in ("病害明细", "维修建议"):
        for number, year in enumerate((2017, 2020), 1):
            source_id = f"p{len(source['provenance'])}"
            text = f"{year}年检测 左幅1跨3号桥墩 裂缝最大宽度0.3mm 建议复查"
            row = (
                {"病害部位": "左幅1跨3号桥墩", "病害描述": "裂缝最大宽度0.3mm"}
                if section == "病害明细"
                else {"建议内容": "复查"}
            )
            source["final_data"][section].append(row)
            source["provenance"].append({
                "section": section, "number": number, "source_id": source_id,
                "call_id": "call-1", "identity": {"year": year},
            })
            source["snapshot"]["regions"].append({
                "source_id": source_id, "text": text,
                "location": {"source_path": source_id},
            })
            source["model_calls"][0]["source_ranges"].append(f"{source_id}:0:{len(text)}")
    original = deepcopy(source["final_data"])
    graph = build_fact_graph(source, strict=True, allowed_sha256={"a" * 64}).to_dict()
    assert [o["inspection_year"] for o in graph["observations"] if o["kind"] != "score"] == [2017, 2020, 2017, 2020]
    assert source["final_data"] == original
    index = ReportIndex(tmp_path / "history.sqlite")
    index.ingest(source, allowed_sha256={"a" * 64}, strict=True)
    assert index.query(bridge="测试桥", year=2017)["count"] == 2
    assert index.query(bridge="测试桥", year=2020)["count"] == 2
    assert index.query(bridge="测试桥", year=2021)["count"] == 0
    assert index.measurements(bridge="测试桥", year=2017)["count"] == 1


@pytest.mark.parametrize(
    "visible,claimed,tail",
    [
        ("2017年检测 最大宽度0.3mm", 2018, ""),
        ("2017年与2020年检测 最大宽度0.3mm", 2017, ""),
        ("最大宽度0.3mm", None, ""),
        ("最大宽度0.3mm", 2017, "；2017年检测"),
        ("2017年检测 最大宽度0.3mm", "2017或2020", ""),
        ("病害编号 LF-2017 最大宽度0.3mm", 2017, ""),
        ("累计2017处 最大宽度0.3mm", 2017, ""),
        ("报告日期：2017年 最大宽度0.3mm", 2017, ""),
    ],
)
def test_uncertain_row_year_is_not_current_inspection_or_report_year(
    tmp_path, visible, claimed, tail
):
    from bridge_evidence.contest.facts import build_fact_graph
    from bridge_evidence.contest.report_qa import ReportIndex

    source = record(width=visible)
    source["snapshot"]["regions"][0]["text"] += tail
    source["provenance"][0]["identity"] = {"year": claimed}
    source["final_data"]["维修建议"] = [{"建议内容": "复查"}]
    source["provenance"].append({**source["provenance"][0], "section": "维修建议"})
    graph = build_fact_graph(source, strict=True, allowed_sha256={"a" * 64}).to_dict()
    rows = [o for o in graph["observations"] if o["kind"] != "score"]
    assert all(o["inspection_year"] is None and o["time_basis"] == "unknown" for o in rows)
    index = ReportIndex(tmp_path / "uncertain.sqlite")
    index.ingest(source, allowed_sha256={"a" * 64}, strict=True)
    assert index.query(bridge="测试桥")["count"] == 2
    for year in (2017, 2018, 2020, 2021):
        assert index.query(bridge="测试桥", year=year)["count"] == 0
        assert index.measurements(bridge="测试桥", year=year)["count"] == 0


def test_identity_count_does_not_reassign_unknown_year_from_same_report(tmp_path):
    from bridge_evidence.contest.report_qa import ReportIndex

    first = "左幅1跨3号桥墩 病害编号 LF-001 最大宽度0.3mm"
    second = "2021年检测 左幅1跨3号桥墩 病害编号 LF-002 最大宽度0.5mm"
    source = record(width=first)
    source["final_data"]["病害明细"][0]["病害编号"] = "LF-001"
    source["final_data"]["病害明细"].append({
        "病害部位": "左幅1跨3号桥墩", "病害编号": "LF-002", "病害描述": second,
    })
    source["snapshot"]["regions"].append({
        "source_id": "p1", "text": second, "location": {"source_path": "p1"},
    })
    source["provenance"].append({
        "section": "病害明细", "number": 2, "source_id": "p1", "call_id": "call-1",
    })
    source["model_calls"][0]["source_ranges"].append(f"p1:0:{len(second)}")
    index = ReportIndex(tmp_path / "counts.sqlite")
    index.ingest(source, allowed_sha256={"a" * 64}, strict=True)
    assert index.count(bridge="测试桥", basis="defect-identities")["count"] == 2
    assert index.count(bridge="测试桥", year=2021, basis="defect-identities")["count"] == 1
