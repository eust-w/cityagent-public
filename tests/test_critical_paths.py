from pathlib import Path
import pytest
from bridge_evidence.contest.table_facts import extract_table_facts
from bridge_evidence.contest.temporal_router import detect_temporal_baseline
from bridge_evidence.contest.defect_chunker import extract_defect_rows_from_tables
from bridge_evidence.contest.constraint_cleaner import clean_and_constrain, extract_advice_from_docx


DEMO_DOCX = Path("demo/10、上跨铁路桥.docx")


@pytest.mark.skipif(not DEMO_DOCX.exists(), reason="demo docx not found")
def test_table_facts_extraction():
    facts = extract_table_facts(DEMO_DOCX)
    assert facts["桥梁名称"] == "上跨铁路桥"
    assert facts["总体评分"] == "93.33"
    assert facts["总体等级"] == "A级"
    assert facts["上部结构评分"] == "85.00"
    assert facts["上部结构等级"] == "B级"
    assert facts["下部结构评分"] == "98.88"
    assert facts["下部结构等级"] == "A级"
    assert facts["桥面系评分"] == "98.89"
    assert facts["桥面系等级"] == "A级"
    assert facts["上一次总体评分"] == "84.30"
    assert facts["上一次总体等级"] == "B级"


@pytest.mark.skipif(not DEMO_DOCX.exists(), reason="demo docx not found")
def test_temporal_router():
    facts = extract_table_facts(DEMO_DOCX)
    temporal = detect_temporal_baseline(DEMO_DOCX, facts)
    assert temporal["has_comparison"] is True
    assert temporal["mode"] == "with_comparison"


@pytest.mark.skipif(not DEMO_DOCX.exists(), reason="demo docx not found")
def test_defect_chunker():
    defects = extract_defect_rows_from_tables(DEMO_DOCX)
    assert len(defects) == 7
    # 验证没有照片干扰行
    for d in defects:
        assert not d["病害部位"].startswith("照片")
        assert not d["病害描述"].startswith("照片")


@pytest.mark.skipif(not DEMO_DOCX.exists(), reason="demo docx not found")
def test_constraint_cleaner():
    facts = extract_table_facts(DEMO_DOCX)
    temporal = detect_temporal_baseline(DEMO_DOCX, facts)
    defects = extract_defect_rows_from_tables(DEMO_DOCX)
    advice = extract_advice_from_docx(DEMO_DOCX)

    data = {
        "基本信息": facts,
        "维修建议": advice,
        "病害明细": defects,
    }
    cleaned = clean_and_constrain(data, DEMO_DOCX, temporal)
    assert len(cleaned["维修建议"]) >= 4
    assert len(cleaned["病害明细"]) == 7
    assert cleaned["基本信息"]["总体结论"] != ""
    assert cleaned["基本信息"]["主要风险点"] != ""
