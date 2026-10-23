import copy
import hashlib
import json

import pytest

from bridge_evidence.contest.document_ner_materials import build_index, load_index, pin_index, project_prompt, admitted_primary_complete
from bridge_evidence.contest.document_source_delivery import SourceDeliveryLedger
from bridge_evidence.contest.evidence_manifest import manifest_from_text


def fixture():
    m = manifest_from_text("\n".join([f"封面行{i}" for i in range(16)] + [
        "本次未见渗水，梯道底部另有渗水。", "常规仪器介绍", "一般检测方法",
        "上一次2015年评分97.98，本次95.43。", "在长期作用下对结构耐久性有一定影响。",
        "建议对其他常规缺陷进行必要养护维修。", "二〇一八年十二月二十日", "右幅89.96，左幅89.78。", "无关附录"]))
    # Numeric scores protected by the containing table below, not model guesses.
    m["units"][-2].update(table_id="table-1", kind="table_row", row_index=1, cells=[])
    m["tables"] = [{"id": "table-1", "header_evidence_ids": [m["units"][19]["id"]]}]
    u = m["units"][16]
    mentions = [{"evidence_id": u["id"], "type": "病害类型", "start": 4, "end": 6, "text": "渗水"}]
    return m, mentions


def test_context_protection_and_non_fact_entities():
    m, mentions = fixture()
    index = build_index(m, mentions)
    selected = set(index["selected_evidence_ids"])
    for n in (16, 19, 20, 21, 22, 23):
        assert m["units"][n]["id"] in selected
    assert m["units"][-1]["id"] not in selected
    assert index["mentions"] == mentions
    assert "confirmed_fact" in index["selection_reasons"][mentions[0]["evidence_id"]][0]
    assert "未见渗水" in m["units"][16]["text"]


def test_no_candidates_falls_back_to_all_units():
    m, _ = fixture()
    assert len(build_index(m, [])["selected_evidence_ids"]) == len(m["units"])


@pytest.mark.parametrize("change", ["document", "text", "selection", "manifest", "file"])
def test_index_cannot_cross_documents_or_be_silently_changed(tmp_path, change):
    m, mentions = fixture()
    index = build_index(m, mentions)
    if change == "document": index["document_sha256"] = "0" * 64
    if change == "text": index["mentions"][0]["text"] = "裂缝"
    if change == "selection": index["selected_evidence_ids"] = []
    if change == "manifest": m["units"][0]["text"] += "changed"
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index, ensure_ascii=False))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if change == "file": sha = "0" * 64
    with pytest.raises(ValueError): load_index(m, path, sha)


def test_projection_and_read_receipts_are_exact_and_pinning_is_local(tmp_path):
    m, mentions = fixture()
    index = build_index(m, mentions)
    task = {"primary_evidence_ids": [u["id"] for u in m["units"]]}
    document = {"materials": [{"units": [{"evidence_id": u["id"], "text": u["text"]} for u in m["units"]]}],
        "assignment_brief": {"input_provision": {}}}
    ranges = [{"evidence_id": u["id"], "start": 0, "end": len(u["text"])} for u in m["units"]]
    original = copy.deepcopy(document)
    prompt, actual = project_prompt(m, task, "source\n" + json.dumps(document, ensure_ascii=False), ranges, index)
    assert document == original
    delivered = {r["evidence_id"] for r in actual}
    assert delivered == set(index["selected_evidence_ids"])
    assert m["units"][-1]["text"] not in prompt
    brief = json.loads(prompt.split("\n", 1)[1])["assignment_brief"]["input_provision"]
    assert brief["all_primary_units_embedded"] is False
    assert "source_reads_needed_for_unembedded_ids" not in brief
    assert admitted_primary_complete(m, task, actual, index) is True
    assert admitted_primary_complete(m, task, actual[1:], index) is False
    ledger = SourceDeliveryLedger(m, tmp_path / "delivery", run_id="r", agent_id="a")
    ledger.register_initial_message(prompt, actual)
    assert ledger.confirmed_ranges() == {}
    assert pin_index(m, {}, tmp_path) == {}
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index, ensure_ascii=False))
    cfg = {"ner_index_path": str(path), "ner_index_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    pinned = pin_index(m, cfg, tmp_path)
    path.unlink()
    assert load_index(m, pinned["ner_index_path"], pinned["ner_index_sha256"]) == index


def test_context_is_one_hop_not_recursive():
    m = manifest_from_text("\n".join([f"资料条目{i}" for i in range(30)] + ["桥台渗水"]))
    for prev, u in zip(m["units"], m["units"][1:]):
        u["context"] = {"preceding_evidence_ids": [prev["id"]]}
    last = m["units"][-1]
    index = build_index(m, [{"evidence_id": last["id"], "type": "病害类型", "start": 2, "end": 4, "text": "渗水"}])
    assert m["units"][20]["id"] not in index["selected_evidence_ids"]


def test_entire_nested_table_and_merged_cell_origins_are_preserved():
    m, mentions = fixture()
    row = m["units"][23]
    m["tables"].append({"id": "nested", "parent_table_id": "table-1", "header_evidence_ids": []})
    m["units"][-1].update(table_id="nested", cells=[{"merged_from_source": "origin"}])
    m["units"][18]["cells"] = [{"source_path": "origin"}]
    index = build_index(m, mentions)
    assert row["id"] in index["selected_evidence_ids"]
    assert m["units"][-1]["id"] in index["selected_evidence_ids"]
    assert m["units"][18]["id"] in index["selected_evidence_ids"]
