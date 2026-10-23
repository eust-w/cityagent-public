import pytest

from bridge_evidence.contest.long_document import DocumentSnapshot
from bridge_evidence.contest.training_revisions import apply_revision, entity_partition


def test_original_label_immutable_and_requires_evidence():
    original = {"基本信息": {"总体评分": "88.15"}, "维修建议": [], "病害明细": []}
    snapshot = DocumentSnapshot.from_text("总体评分85.91")
    revision = {
        "document_sha256": snapshot.document_sha256,
        "status": "approved",
        "adjudication": "同对象正文评分",
        "changes": [
            {
                "section": "基本信息",
                "field": "总体评分",
                "value": "85.91",
                "source_id": "p0",
                "quote": "85.91",
            }
        ],
    }
    fixed = apply_revision(original, snapshot, revision)
    assert original["基本信息"]["总体评分"] == "88.15"
    assert fixed["基本信息"]["总体评分"] == "85.91"
    with pytest.raises(ValueError):
        apply_revision(original, snapshot, {**revision, "status": "pending"})


def test_split_bridge_entity_before_chunks():
    rows = [
        {"bridge_entity": "A", "entity_verified": True, "report": "1"},
        {"bridge_entity": "A", "entity_verified": True, "report": "2"},
    ]
    result = entity_partition(rows, seed=17)
    assert result[0]["split"] == result[1]["split"]
    with pytest.raises(ValueError):
        entity_partition([{"bridge_entity": "A", "entity_verified": False}])


def test_subtasks_require_adjudicated_complete_coverage():
    from test_contest_optimization import Tokenizer, chunk_input_budget

    from bridge_evidence.contest.long_document import plan_chunks
    from bridge_evidence.contest.training_revisions import build_subtasks

    snapshot = DocumentSnapshot.from_text("桥")
    plan = plan_chunks(snapshot, Tokenizer(), input_budget=chunk_input_budget(snapshot), output_budget=100)
    task = plan.tasks[0]
    target = {
        "covered": task.owned_keys,
        "基本信息": {"桥梁名称": "桥"},
        "basic_sources": {"桥梁名称": "p0"},
        "维修建议": [],
        "病害明细": [],
    }
    annotation = {
        "task_id": task.task_id,
        "status": "approved",
        "covered": task.owned_keys,
        "adjudication": "完整来源核对",
        "target": target,
    }
    rows = build_subtasks(snapshot, plan, [annotation])
    assert len(rows) == 1
    assert rows[0]["task_id"] == task.task_id
    with pytest.raises(ValueError):
        build_subtasks(snapshot, plan, [{**annotation, "covered": []}])
    assert build_subtasks(snapshot, plan, [{**annotation, "status": "pending"}]) == []
