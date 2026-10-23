import pytest

from bridge_evidence.contest.meta_optimization import CloudBudget, candidate_configs
from bridge_evidence.contest.report_qa import ReportIndex


def test_search_fixed_seed_bounded_and_weights_unchanged():
    a = candidate_configs({"length_failures": 2, "format_failures": 1}, seed=17)
    assert a == candidate_configs({"length_failures": 2, "format_failures": 1}, seed=17)
    assert len(a) <= 8
    assert all("model" not in c for c in a)


def test_budget_does_not_start_when_total_exhausted(tmp_path):
    budget = CloudBudget(tmp_path / "budget.sqlite", maximum_seconds=1)
    job = budget.start("probe", 1)
    budget.finish(job, 1)
    with pytest.raises(RuntimeError, match="budget"):
        budget.start("new", 1)


def test_index_requires_allowed_sources_and_retains_duplicate_rows(tmp_path):
    index = ReportIndex(tmp_path / "facts.sqlite")
    record = {
        "status": "success",
        "document": {"relative_path": "桥.docx", "sha256": "a" * 64},
        "final_data": {
            "基本信息": {"桥梁名称": "桥", "报告日期": "2018-01-01"},
            "维修建议": [],
            "病害明细": [
                {"病害部位": "梁", "病害类型": "裂缝"},
                {"病害部位": "梁", "病害类型": "裂缝"},
            ],
        },
        "provenance": [
            {"section": "病害明细", "number": 1, "source_id": "t0r1"},
            {"section": "病害明细", "number": 2, "source_id": "t0r2"},
        ],
    }
    with pytest.raises(ValueError, match="allowed"):
        index.ingest(record, allowed_sha256=set())
    index.ingest(record, allowed_sha256={"a" * 64})
    answer = index.query(bridge="桥", component="梁", disease="裂缝")
    assert answer["count"] == 2
    assert len(answer["sources"]) == 2
    # Report date alone does not date individual historical defect occurrences.
    assert index.query(bridge="桥", year=2018, component="梁", disease="裂缝")["count"] == 0
    assert index.query(bridge="不存在")["status"] == "insufficient-evidence"


def test_meta_budget_single_job_recovery_and_duplicate_finish(tmp_path):
    budget = CloudBudget(tmp_path / "budget.sqlite", maximum_seconds=100)
    budget.start("probe", 20)
    with pytest.raises(RuntimeError, match="already running"):
        budget.start("second", 10)
    budget.recover()
    assert budget.remaining() < 100
    other = budget.start("probe", 20)
    budget.finish(other, 2)
    budget.finish(other, 2)
    with pytest.raises(ValueError):
        budget.finish(other, 3)


def test_qa_comparison_requires_evidence_for_each_year(tmp_path):
    index = ReportIndex(tmp_path / "qa.sqlite")
    assert index.compare("桥", [2017, 2018])["status"] == "insufficient-evidence"
    with pytest.raises(ValueError):
        index.compare("桥", [2018])


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), -1, True])
def test_cloud_budget_rejects_invalid_duration(tmp_path, seconds):
    budget = CloudBudget(tmp_path / "budget.sqlite", maximum_seconds=100)
    job = budget.start("probe", 20)
    with pytest.raises(ValueError):
        budget.finish(job, seconds)


def test_gold_differences_select_only_generic_search_parameters():
    from bridge_evidence.contest.meta_optimization import diagnose_baseline

    errors = diagnose_baseline(
        {
            "reports": [{"error": "ContextBudgetError"}, {"error": "invalid JSON"}],
            "final_metrics": {"sections": {"defects": {"fn": 3, "fp": 2}}},
        }
    )
    assert errors == {
        "length_failures": 1,
        "format_failures": 1,
        "gold_false_negatives": 3,
        "gold_false_positives": 2,
    }
    configs = candidate_configs(errors, seed=17)
    assert any(c["output_budget"] == 6144 for c in configs)
    assert any(c["prompt"] == "source-fragment-concise-v1" for c in configs)
