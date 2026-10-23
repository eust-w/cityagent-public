"""Supplemental diagnostics never alter the official three-table score."""

import importlib

import pytest


def diagnostic_module():
    return importlib.import_module("bridge_evidence.contest.semantic_diagnostics")


def test_covered_sources_do_not_establish_fact_recall_or_parser_completeness():
    result = diagnostic_module().diagnose_record(
        {
            "snapshot": {"regions": [{"source_id": "r1", "text": "裂缝0.3mm"}]},
            "coverage": {"complete": True},
            "fragments": [],
            "chunk_plan": {
                "tasks": [
                    {
                        "task_id": "a",
                        "owned": [
                            {
                                "source_id": "r1",
                                "start": 0,
                                "end": 7,
                                "text": "裂缝0.3mm",
                            }
                        ],
                    }
                ]
            },
        }
    )
    assert result["protocol"] == "semantic-evidence-diagnostics-v1"
    assert result["parsed_coverage"]["recall"] is None
    assert result["fact_coverage"]["recall"] is None
    assert result["task_coverage"]["claimed_complete"] is True
    assert result["task_coverage"]["completed_tasks"] == 0


def test_evidence_must_be_visible_to_the_specific_call_and_quote_must_match():
    row = {
        "source_id": "r2",
        "source_key": "r2:0:8",
        "instance_id": "1",
        "fields": {"病害": "裂缝"},
        "quotes": {"病害": "裂缝"},
        "history_sources": ["r3"],
    }
    record = {
        "snapshot": {
            "regions": [
                {"source_id": "r1", "text": "裂缝"},
                {"source_id": "r2", "text": "空白"},
                {"source_id": "r3", "text": "2016年裂缝"},
            ]
        },
        "chunk_plan": {
            "tasks": [
                {
                    "task_id": "t",
                    "owned": [
                        {"source_id": "r1", "start": 0, "end": 2, "text": "裂缝"}
                    ],
                    "context": [],
                }
            ]
        },
        "fragments": [
            {"task_id": "t", "call_id": "c", "diseases": [row], "advice": []}
        ],
    }
    report = diagnostic_module().diagnose_record(record)
    assert report["evidence"]["invisible_references"] == 2
    assert report["evidence"]["invalid_quotes"] == 1
    assert report["evidence"]["semantic_support"] == "not-established-by-substring"


def test_relation_gold_is_independent_unknown_never_counts_as_confirmed_match():
    result = diagnostic_module().diagnose_record(
        {
            "relations": [
                {"left": "a", "right": "b", "status": "confirmed"},
                {"left": "b", "right": "c", "status": "unknown"},
            ]
        },
        annotations={
            "schema_version": 1,
            "relations": [
                {"left": "a", "right": "b", "status": "rejected"},
                {"left": "b", "right": "c", "status": "confirmed"},
            ],
        },
    )
    assert result["relations"]["false_merges"] == 1
    assert result["relations"]["unknown"] == 1
    assert result["relations"]["recall"] == 0


def test_facts_have_source_occurrence_identity_not_text_deduplication():
    record = {
        "diagnostic_facts": [
            {"source_id": "a", "instance_id": "1", "field": "病害", "value": "裂缝"}
        ]
    }
    gold = {
        "schema_version": 1,
        "facts": [
            {"source_id": "a", "instance_id": "1", "field": "病害", "value": "裂缝"},
            {"source_id": "b", "instance_id": "1", "field": "病害", "value": "裂缝"},
        ],
    }
    result = diagnostic_module().diagnose_record(record, annotations=gold)
    assert result["fact_coverage"]["recall"] == 0.5
    assert result["fact_coverage"]["expected"] == 2


def test_numeric_conversion_requires_compatible_statistic_and_units():
    compare = diagnostic_module().compare_quantity
    assert (
        compare(
            {"value": "0.3", "unit": "mm", "statistic": "max"},
            {"value": "0.03", "unit": "cm", "statistic": "max"},
        )["equal"]
        is True
    )
    assert (
        compare(
            {"value": "0.3", "unit": "mm", "statistic": "max"},
            {"value": "0.03", "unit": "cm", "statistic": "mean"},
        )["comparable"]
        is False
    )
    assert (
        compare({"value": "nan", "unit": "mm"}, {"value": 0, "unit": "mm"})[
            "comparable"
        ]
        is False
    )


def test_manifest_strict_entity_split_rejects_alias_leakage_and_missing_evidence():
    prepare = importlib.import_module("scripts.prepare_multitask_manifest")
    cases = [
        {"id": "1", "report_sha256": "a" * 64},
        {"id": "2", "report_sha256": "b" * 64},
    ]
    registry = {
        "schema_version": 1,
        "independent_verified": True,
        "evidence": "review-v1",
        "reports": {
            "a" * 64: {
                "bridge_entity": "bridge-1",
                "split": "train",
                "evidence": "source-a",
            },
            "b" * 64: {
                "bridge_entity": "bridge-1",
                "split": "evaluation",
                "evidence": "source-b",
            },
        },
    }
    with pytest.raises(ValueError, match="multiple splits"):
        prepare.bind_bridge_entities(cases, registry, strict=True)
    registry["reports"]["b" * 64]["bridge_entity"] = "bridge-2"
    bound = prepare.bind_bridge_entities(cases, registry, strict=True)
    assert bound[0]["entity_verified"] is True
    assert bound[1]["split"] == "evaluation"
    del registry["reports"]["b" * 64]["evidence"]
    with pytest.raises(ValueError, match="verification evidence"):
        prepare.bind_bridge_entities(cases, registry, strict=True)


def test_quantity_diagnostics_use_actual_predictions_not_annotation_prediction():
    report = diagnostic_module().diagnose_record(
        {
            "diagnostic_quantities": [
                {"id": "width-1", "value": 0.3, "unit": "mm", "statistic": "maximum"}
            ]
        },
        annotations={
            "schema_version": 1,
            "quantities": [
                {"id": "width-1", "value": 0.03, "unit": "cm", "statistic": "maximum"}
            ],
        },
    )
    assert report["quantities"]["equal"] == 1
    assert report["quantities"]["expected"] == 1
    assert report["quantities"]["missing"] == 0


def test_graph_relations_and_observation_facts_preserve_independent_occurrences():
    graph = {
        "observations": [
            {"id": "o1", "raw": {"病害": "裂缝"}, "evidence_ids": ["e1"]},
            {"id": "o2", "raw": {"病害": "裂缝"}, "evidence_ids": ["e2"]},
        ],
        "evidence": [
            {"id": "e1", "source_ids": ["a"], "instance_id": "1"},
            {"id": "e2", "source_ids": ["b"], "instance_id": "1"},
        ],
        "relations": [
            {"left": "o1", "right": None, "type": "same-defect", "status": "uncertain"}
        ],
    }
    result = diagnostic_module().diagnose_record(
        {"facts": graph},
        annotations={
            "schema_version": 1,
            "facts": [
                {
                    "source_id": "a",
                    "instance_id": "1",
                    "field": "病害",
                    "value": "裂缝",
                },
                {
                    "source_id": "b",
                    "instance_id": "1",
                    "field": "病害",
                    "value": "裂缝",
                },
            ],
        },
    )
    assert result["fact_coverage"]["recall"] == 1
    assert result["relations"]["unknown"] == 1


def test_fragment_declaration_alone_does_not_complete_owned_task():
    record = {
        "chunk_plan": {
            "tasks": [
                {
                    "task_id": "t",
                    "owned": [{"source_id": "p", "start": 0, "end": 3, "text": "abc"}],
                }
            ]
        },
        "fragments": [{"task_id": "t", "covered": [], "diseases": [], "advice": []}],
    }
    assert (
        diagnostic_module().diagnose_record(record)["task_coverage"]["completed_tasks"]
        == 0
    )


def test_duplicate_relation_receipts_do_not_inflate_recall():
    relation = {"left": "a", "right": "b", "status": "confirmed"}
    report = diagnostic_module().diagnose_record(
        {"relations": [relation, relation]},
        annotations={"schema_version": 1, "relations": [relation]},
    )
    assert report["relations"]["recall"] == 1
    assert report["relations"]["duplicate_predictions"] == 1
