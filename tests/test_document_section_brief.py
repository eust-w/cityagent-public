"""Offline scope, source inventory, budget and worker selection contracts."""
import copy
import json

import pytest

from bridge_evidence.contest.document_multiagent_worker import section_prompt
from bridge_evidence.contest.document_section_brief import build_section_brief, briefed_section_prompt
from bridge_evidence.contest.document_section_input import SectionInputBudgetError, compact_section_prompt
from bridge_evidence.contest.document_source_delivery import SourceDeliveryLedger
from bridge_evidence.contest.document_task_plan import build_document_task_plan
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import json_digest


def data(prompt):
    return json.loads(prompt.split("\n", 1)[1])


def task(identifier, ids, *, context=()):
    return {"id": identifier, "primary_evidence_ids": list(ids),
            "context_evidence_ids": list(context), "headings": [], "table_ids": [],
            "material_section_refs": []}


def plan(manifest, tasks):
    return {"document_sha256": manifest["document_sha256"],
            "manifest_sha256": json_digest(manifest), "tasks": copy.deepcopy(tasks)}


class Materials:
    """Exact source fixture whose front matter is absent from task.headings."""
    def __init__(self, manifest):
        self.manifest = copy.deepcopy(manifest)
        units = manifest["units"]
        self.sections = [{"id": f"#/sections/S{i}", "title": title, "parent": None,
            "region": "front_matter" if i < 2 else "content"}
            for i, title in enumerate(["报告前置内容", "目录", "检查结果", "评分"])]
        self.packets = [{"id": f"#/packets/P{i}", "section_id": section["id"]}
                        for i, section in enumerate(self.sections)]
        self.spans = [{"evidence_id": u["id"], "start": 0, "end": len(u["text"]), "text": u["text"]}
                      for u in units]

    def validate_manifest(self, manifest):
        if manifest != self.manifest:
            raise ValueError("Different manifest")

    def section(self, ref, *, include_descendants, include_context):
        return {"spans": [self.spans[int(ref.rsplit("S", 1)[1])]]}

    def packet(self, ref):
        return {"spans": [self.spans[int(ref.rsplit("P", 1)[1])]], "context": [],
                "unresolved": [{"code": "ambiguous_xml_text_mapping", "ref": "#/texts/duplicate"}]}


@pytest.fixture
def source():
    manifest = manifest_from_text("摘要：某桥支座锈蚀，建议除锈\n目录：检查结果、评分\n检查结果：支座锈蚀\n评分章没有给出分数")
    ids = [u["id"] for u in manifest["units"]]
    tasks = [task("overview", ids[:2]), task("findings", ids[2:3], context=ids[:1]), task("ratings", ids[3:])]
    return manifest, tasks, plan(manifest, tasks)


def test_actual_first_input_scope_includes_summary_and_toc_without_turning_them_into_tasks(source):
    manifest, tasks, partition = source
    prompt, exposed = briefed_section_prompt(manifest, tasks[0], material_index=Materials(manifest), task_plan=partition)
    document = data(prompt); brief = document["assignment_brief"]
    assert next(iter(document)) == "assignment_brief"
    assert [s["title"] for s in brief["scope"]["sections"]] == ["报告前置内容", "目录"]
    assert [t["task_id"] for t in brief["other_tasks"]] == ["findings", "ratings"]
    assert brief["input_provision"]["primary_fully_embedded_units"] == 2
    assert brief["input_provision"]["all_primary_units_embedded"] is True
    assert "摘要" in brief["work_policy"]["summary"]
    assert "不是要求逐章访问" in brief["work_policy"]["navigation"]
    rows = [u for g in document["materials"] for u in g["units"]]
    assert [(r["evidence_id"], r["text"]) for r in rows] == [(u["id"], u["text"]) for u in manifest["units"][:2]]
    assert len(exposed) == 2
    assert "field_assignments" not in prompt and "95.43" not in prompt
    assert "不自动意味着业务缺证" in brief["work_policy"]["technical_diagnostics"]
    assert document["unresolved_material_references"]


def test_context_is_described_without_adding_it_to_primary_or_duplicating_source(source):
    manifest, tasks, partition = source
    prompt, exposed = briefed_section_prompt(manifest, tasks[1], task_plan=partition)
    document = data(prompt); inventory = document["assignment_brief"]["input_provision"]
    assert inventory["primary_total_units"] == inventory["primary_fully_embedded_units"] == 1
    assert inventory["context_embedded_units"] == 1
    assert len(document["source_units"]) == len(exposed) == 2
    assert {r["role"] for r in document["source_units"]} == {"PRIMARY", "CONTEXT"}
    assert inventory["meaning"] == "embedded_in_this_request_not_delivery_or_understanding"


def test_speed_projection_keeps_every_source_and_table_but_removes_alias_ambiguity(source):
    from bridge_evidence.contest.document_section_brief import speed_section_input
    manifest,tasks,partition=source
    prompt,exposed=briefed_section_prompt(manifest,tasks[0],material_index=Materials(manifest),task_plan=partition)
    original=data(prompt); wire=data(speed_section_input(prompt))
    assert [r for g in original['materials'] for r in g['units']]==[r for g in wire['materials'] for r in g['units']]
    assert wire['tables']==original['tables']
    assert wire['task']==original['task']
    assert wire['assignment_brief']['input_provision']==original['assignment_brief']['input_provision']
    assert '#/sections/' not in json.dumps(wire) and '#/packets/' not in json.dumps(wire)
    assert '避免重复提交' in wire['assignment_brief']['work_policy']['summary']
    assert wire['material_mapping_diagnostic_count']==len(original['unresolved_material_references'])
    assert data(prompt)==original and exposed


@pytest.mark.parametrize("mutation", ["document", "manifest", "current", "duplicate_task", "overlap", "omitted"])
def test_foreign_or_inconsistent_plan_is_rejected_before_scope_claims(source, mutation):
    manifest, tasks, partition = source
    if mutation == "document":
        partition["document_sha256"] = "0" * 64
    elif mutation == "manifest":
        partition["manifest_sha256"] = "0" * 64
    elif mutation == "current":
        partition["tasks"][0]["extra"] = "changed task"
    elif mutation == "duplicate_task":
        partition["tasks"][1]["id"] = tasks[0]["id"]
    elif mutation == "overlap":
        partition["tasks"][1]["primary_evidence_ids"].append(tasks[0]["primary_evidence_ids"][0])
    else:
        partition["tasks"].pop()
    with pytest.raises(ValueError):
        briefed_section_prompt(manifest, tasks[0], task_plan=partition)


def test_partial_gapped_source_inventory_does_not_claim_complete():
    manifest = manifest_from_text("abcdef\n短文\n")
    ids = [u["id"] for u in manifest["units"]]
    current = task("one", ids)
    spans = [{"evidence_id": ids[0], "start": 0, "end": 2},
             {"evidence_id": ids[0], "start": 3, "end": len(manifest["units"][0]["text"])},
             {"evidence_id": ids[1], "start": 0, "end": len(manifest["units"][1]["text"])}]
    brief = build_section_brief(manifest, current, spans)
    inventory = brief["input_provision"]
    assert inventory["primary_fully_embedded_units"] == 1
    assert inventory["unembedded_or_partial_primary_ids"] == [ids[0]]
    assert inventory["source_reads_needed_for_unembedded_ids"] == [ids[0]]
    spans.append({"evidence_id": ids[0], "start": 2, "end": 3})
    assert build_section_brief(manifest, current, spans)["input_provision"]["all_primary_units_embedded"]


def test_legacy_plan_heading_keeps_canonical_newline_without_inventing_fields():
    manifest = manifest_from_text("1 检查结果\n原文只有位置\n2 评分\n原文未记载分数")
    partition = build_document_task_plan(manifest, target_tasks=2, max_tasks=2)
    for current in partition["tasks"]:
        prompt, _ = briefed_section_prompt(manifest, current, task_plan=partition)
        brief = data(prompt)["assignment_brief"]
        assert brief["input_provision"]["all_primary_units_embedded"]
        assert "field_assignments" not in brief and "评分" not in brief.get("assigned_fields", [])
    bad = copy.deepcopy(partition["tasks"][0])
    bad["headings"][0]["title"] = "编造的标题"
    with pytest.raises(ValueError, match="heading"):
        briefed_section_prompt(manifest, bad)


@pytest.mark.parametrize("span", [{"start": -1, "end": 1}, {"start": 0, "end": 999},
                                 {"start": False, "end": 1}])
def test_invalid_embedded_span_rejected(source, span):
    manifest, tasks, _ = source
    with pytest.raises(ValueError, match="Embedded source"):
        build_section_brief(manifest, tasks[0], [{"evidence_id": tasks[0]["primary_evidence_ids"][0], **span}])


def test_total_budget_includes_brief_and_inventory_tracks_actual_admission():
    manifest = manifest_from_text("完整长段" * 3000 + "\n必须保留的短段")
    ids = [u["id"] for u in manifest["units"]]
    current = task("bounded", ids)
    prompt, exposed = briefed_section_prompt(manifest, current, 4000)
    document = data(prompt); brief = document["assignment_brief"]
    assert len(prompt) <= 4000
    assert len(exposed) == len(document["source_units"]) == 1
    assert document["source_units"][0]["text"] == "必须保留的短段"
    assert brief["input_provision"]["unembedded_or_partial_primary_ids"] == [ids[0]]
    assert brief["input_provision"]["primary_fully_embedded_units"] == 1
    with pytest.raises(SectionInputBudgetError):
        briefed_section_prompt(manifest, current, 100)


def test_exact_budget_reuses_complete_input_and_exposure(source):
    manifest, tasks, partition = source
    prompt, exposed = briefed_section_prompt(manifest, tasks[0], task_plan=partition)
    bounded, actual = briefed_section_prompt(manifest, tasks[0], len(prompt), task_plan=partition)
    assert (bounded, actual) == (prompt, exposed)
    shorter, less = briefed_section_prompt(manifest, tasks[0], len(prompt) - 1, task_plan=partition)
    assert len(shorter) <= len(prompt) - 1 and len(less) < len(exposed)


def test_worker_old_path_is_byte_identical_and_direct_path_uses_brief(source):
    manifest, tasks, partition = source
    assert section_prompt(manifest, tasks[0]) == compact_section_prompt(manifest, tasks[0])
    prompt, exposed = section_prompt(manifest, tasks[0], direct_delivery=True, task_plan=partition)
    assert data(prompt)["assignment_brief"]["other_tasks_map_available"]
    assert len(exposed) == 2


def test_brief_does_not_grant_delivery_or_facts_and_exact_initial_message_can_be_confirmed(source, tmp_path):
    manifest, tasks, partition = source
    prompt, exposed = briefed_section_prompt(manifest, tasks[0], task_plan=partition)
    ledger = SourceDeliveryLedger(manifest, tmp_path / "ledger", run_id="test", agent_id=tasks[0]["id"])
    ledger.register_initial_message(prompt, exposed, role="user")
    assert not ledger.confirmed_ranges()
    payload = {"messages": [{"role": "user", "content": prompt}]}
    ledger.observe_request(payload, {"request_id": "request", "call_id": "generation"})
    assert not ledger.confirmed_ranges()
    ledger.confirm_request("request", evidence={"http_status": 200, "input_delivery_confirmed": True,
                                               "request_id": "request", "call_id": "generation"})
    assert set(ledger.confirmed_ranges()) == set(tasks[0]["primary_evidence_ids"])
