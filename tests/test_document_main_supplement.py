"""CPU-only main supplementation, source binding and replay regressions."""
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from bridge_evidence.contest import document_candidate_review as engine
from bridge_evidence.contest.document_main_supplement import MainSupplementTools, SUPPLEMENT_SCHEMAS, validate_main_delivery
from bridge_evidence.contest.document_material_index import DocumentMaterialIndex
from bridge_evidence.contest.document_material_refs import normalize_material_refs
from bridge_evidence.contest.document_workspace import DocumentWorkspace
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import json_digest
from test_document_material_index import make_material_case
from test_document_candidate_review import case


@pytest.fixture
def owner(tmp_path):
    manifest = manifest_from_text("左幅支座轻微锈蚀\n建议除锈涂漆\n右幅梁体裂缝\n")
    workspace = DocumentWorkspace(manifest, tmp_path / "main", {"model": "synthetic"})
    for unit in manifest["units"]:
        workspace.record_read(unit["id"], 0, len(unit["text"]))
    workspace.apply_update({"base_version": workspace.snapshot()["version"],
        "draft": {"基本信息": {}, "病害明细": [], "维修建议": []}, "draft_fact_ids": [], "draft_bindings": {}})
    return workspace


def tool(owner, name="save_defects"):
    return MainSupplementTools(owner, {}, [], name)


def entry(owner, index=0, name="save_defects", **changes):
    result = {"source_refs": [{"evidence_id": owner.manifest["units"][index]["id"]}],
        "status": "explicit", "病害描述" if name == "save_defects" else "建议内容":
        "轻微锈蚀" if name == "save_defects" else "除锈涂漆"}
    result.update(changes)
    return result


def native(monkeypatch, name, payload, call="call-1"):
    invocation = {"tool_name": name, "run_id": "run", "session_id": "session", "generation_call_id": "generation",
        "native_tool_call_id": call, "args_sha256": json_digest(payload)}
    monkeypatch.setattr(engine, "_current_invocation", lambda: invocation)
    return invocation


def test_missing_defects_and_advice_are_saved_with_facts_and_bindings(owner):
    defects, advice = tool(owner), tool(owner, "save_recommendations")
    first = defects.call(entries=[entry(owner, 病害部位="左幅支座", context={"year": "2018"})])
    second = advice.call(entries=[entry(owner, 1, "save_recommendations", 建议类别="尽快维修")])
    assert first["ok"] and second["ok"] and first["added_count"] == second["added_count"] == 1
    state = owner.snapshot()
    assert state["draft"]["病害明细"] == [{"病害描述": "轻微锈蚀", "病害部位": "左幅支座"}]
    assert state["draft"]["维修建议"] == [{"建议内容": "除锈涂漆", "建议类别": "尽快维修"}]
    assert "尽快维修1条" in state["draft"]["基本信息"]["建议统计"]
    for section in ("病害明细", "维修建议"):
        identifier = state["draft_bindings"][section + ".0"][0]
        assert identifier.startswith("MSUP_") and state["facts"][identifier]["value"] == state["draft"][section][0]
    assert all("id" not in item for item in first["assignments"])
    assert defects.handoff == advice.handoff == {}  # Original candidate inventory is frozen.


def test_repeated_content_is_noop_across_calls_and_within_batch(owner):
    t = tool(owner)
    assert t.call(entries=[entry(owner), entry(owner)])["added_count"] == 1
    before = (owner.directory / "workspace.json").read_bytes()
    result = t.call(entries=[entry(owner)])
    assert result["ok"] and not result["changed"] and result["added_count"] == 0
    assert (owner.directory / "workspace.json").read_bytes() == before
    assert result["assignments"] == [{"entry_index": 0, "status": "already_saved"}]


def test_similar_claims_are_not_automatically_semantically_deduplicated(owner):
    assert tool(owner).call(entries=[entry(owner), entry(owner, 病害描述="轻微锈蚀，范围不详")])["ok"]
    assert len(owner.snapshot()["draft"]["病害明细"]) == 2


@pytest.mark.parametrize("name,candidate", [
    ("save_defects", "left_rust"), ("save_recommendations", "left_paint")])
def test_exact_existing_child_is_noop_ignoring_only_program_row_number(case, name, candidate):
    owner = case.workspace
    state = owner.snapshot()
    fact = state["facts"][case.ids[candidate]]
    incoming = {**deepcopy(fact["value"]), "序号": 999,
        "source_refs": [{"evidence_id": ref["evidence_id"]} for ref in fact["source_refs"]],
        "status": fact["status"]}
    before = owner.checkpoint_path.read_bytes()
    t = MainSupplementTools(owner, case.handoff, case.summary, name)
    result = t.call(entries=[incoming])
    assert result["ok"] and not result["changed"] and result["added_count"] == 0
    assert result["assignments"] == [{"entry_index": 0, "status": "already_saved"}]
    # This also preserves any pre-existing advice statistics and all bindings.
    assert owner.checkpoint_path.read_bytes() == before
    assert t.handoff == case.handoff


def seed_child(owner, name="save_defects", **fact_changes):
    incoming = entry(owner, name=name)
    value = {key: val for key, val in incoming.items() if key not in {"source_refs", "status"}}
    fact = {"id": "child_saved", "value": value, "source_refs": incoming["source_refs"],
        "status": incoming["status"], **fact_changes}
    owner.apply_update({"base_version": owner.snapshot()["version"], "facts": [fact],
        "draft_patch": [{"op": "append_row", "section": "病害明细" if name == "save_defects" else "维修建议",
            "row": value, "fact_ids": [fact["id"]]}]})
    return incoming


@pytest.mark.parametrize("difference", ["refs", "status", "value", "context", "dependencies", "fact_value"])
def test_child_entry_differences_are_not_automatically_merged(owner, difference):
    if difference == "dependencies":
        owner.apply_update({"base_version": owner.snapshot()["version"], "facts": [{
            "id": "dependency", "value": "读取过的依据", "source_refs": entry(owner)["source_refs"], "status": "explicit"}]})
    incoming = seed_child(owner, **({"depends_on": ["dependency"]} if difference == "dependencies" else {}))
    if difference == "refs": incoming["source_refs"] = entry(owner, 2)["source_refs"]
    elif difference == "status": incoming["status"] = "inferred"
    elif difference == "value": incoming["病害描述"] += "，范围不详"
    elif difference == "context": incoming["context"] = {"year": "2018"}
    elif difference == "fact_value":
        owner.apply_update({"base_version": owner.snapshot()["version"], "facts": [{
            "id": "child_saved", "value": {"病害描述": "已经修改的候选事实"}}]})
    before = owner.snapshot()
    result = tool(owner).call(entries=[incoming])
    assert result["ok"] and result["added_count"] == 1
    state = owner.snapshot()
    assert state["draft"]["病害明细"][0] == before["draft"]["病害明细"][0]
    assert state["facts"]["child_saved"] == before["facts"]["child_saved"]
    assert state["draft_bindings"]["病害明细.0"] == ["child_saved"]


@pytest.mark.parametrize("context", [None, {"year": "2019"}, {"year": "2018"}])
def test_child_context_must_match_exactly(owner, context):
    incoming = seed_child(owner, context={"year": "2018"})
    if context is not None:
        incoming["context"] = context
    result = tool(owner).call(entries=[incoming])
    assert result["ok"]
    assert result["added_count"] == (0 if context == {"year": "2018"} else 1)


def test_child_context_retained_only_in_handoff_is_not_assumed_absent(owner):
    incoming = seed_child(owner)
    t = MainSupplementTools(owner, {"child_saved": {"model_context": {"year": "2018"}}}, [], "save_defects")
    assert t.call(entries=[incoming])["added_count"] == 1


@pytest.mark.parametrize("problem", ["invalid_fact", "conflict", "missing_binding", "unlisted_binding", "merged_binding"])
def test_exact_noop_requires_intact_valid_single_fact_binding(owner, problem):
    incoming = seed_child(owner)
    t = tool(owner)
    state = owner.snapshot()
    normalized = owner._normalize_record("facts", {"id": "incoming", "value": {"病害描述": "轻微锈蚀"},
        "source_refs": incoming["source_refs"], "status": "explicit"}, deepcopy(state))
    if problem == "invalid_fact": state["facts"]["child_saved"]["valid"] = False
    elif problem == "conflict":
        state["facts"]["child_saved"]["status"] = normalized["status"] = "conflict"
    elif problem == "missing_binding": state["draft_bindings"].pop("病害明细.0")
    elif problem == "unlisted_binding": state["draft_fact_ids"].remove("child_saved")
    else: state["draft_bindings"]["病害明细.0"] = ["child_saved", "another_candidate"]
    assert not t._already_saved_row(state, normalized)


def test_exact_child_advice_noop_preserves_derived_statistics(owner):
    from bridge_evidence.contest.document_main_review_tools import derived_advice_statistics
    incoming = seed_child(owner, "save_recommendations")
    state = owner.snapshot()
    update = {"base_version": state["version"]}
    derived_advice_statistics(state, update)
    owner.apply_update(update)
    before = owner.checkpoint_path.read_bytes()
    assert "未分类/其他1条" in owner.snapshot()["draft"]["基本信息"]["建议统计"]
    result = tool(owner, "save_recommendations").call(entries=[incoming])
    assert result["ok"] and result["added_count"] == 0
    assert owner.checkpoint_path.read_bytes() == before


@pytest.mark.parametrize("kind,path", [
    ("id", "entries[1].id"), ("unknown_field", "entries[1].建议内容"),
    ("missing", "entries[1].病害描述"), ("blank", "entries[1].病害描述"),
    ("unknown_source", "entries[1].source_refs[0].evidence_id"),
    ("range", "entries[1].source_refs[0]"),
    ("malformed_container", "entries"), ("too_many", "entries"),
    ("control", "entries[1].病害部位"),
])
def test_rejection_precise_path_and_all_or_nothing(owner, kind, path):
    assert tool(owner).call(entries=[entry(owner)])["ok"]
    payload = {"entries": [entry(owner, 2, 病害描述="裂缝"), entry(owner)]}
    bad = payload["entries"][1]
    if kind == "id": bad["id"] = "user-chosen"
    elif kind == "unknown_field": bad["建议内容"] = "错类"
    elif kind == "missing": del bad["病害描述"]
    elif kind == "blank": bad["病害描述"] = " "
    elif kind == "unknown_source": bad["source_refs"] = [{"evidence_id": "E999999"}]
    elif kind == "range": bad["source_refs"][0].update(start=0, end=999999)
    elif kind == "malformed_container": payload["entries"] = '[{"病害描述":'
    elif kind == "too_many": payload["entries"] *= 3
    elif kind == "control": bad["病害部位"] = "支座\x00"
    before = (owner.directory / "workspace.json").read_bytes()
    response = tool(owner).call(**payload)
    assert not response["ok"] and response["error"]["details"]["applied"] is False
    assert response["error"]["details"]["path"] == path
    assert "entries" in response["error"]["details"]["example"]
    assert (owner.directory / "workspace.json").read_bytes() == before


def test_partially_read_unit_requires_real_exact_range(tmp_path):
    manifest = manifest_from_text("支座轻微锈蚀，其他材料还没读")
    owner = DocumentWorkspace(manifest, tmp_path, {"model": "synthetic"})
    unit = manifest["units"][0]
    owner.record_read(unit["id"], 0, 6)
    before = (owner.directory / "workspace.json").read_bytes()
    result = tool(owner).call(entries=[entry(owner)])
    assert not result["ok"] and result["error"]["code"] == "SUPPLEMENT_SOURCE_UNREAD"
    assert result["error"]["details"]["path"] == "entries[0].source_refs[0]"
    assert (owner.directory / "workspace.json").read_bytes() == before
    row = entry(owner)
    row["source_refs"][0].update(start=0, end=6)
    assert tool(owner).call(entries=[row])["ok"]


def test_native_replay_preserves_exact_receipt_even_after_another_save(owner, monkeypatch):
    payload = {"entries": [entry(owner)]}
    invocation = native(monkeypatch, "save_defects", payload)
    t = tool(owner)
    first = t.call(**payload)
    monkeypatch.setattr(engine, "_current_invocation", lambda: None)
    assert tool(owner, "save_recommendations").call(entries=[entry(owner, 1, "save_recommendations")])["ok"]
    before = (owner.directory / "workspace.json").read_bytes()
    monkeypatch.setattr(engine, "_current_invocation", lambda: invocation)
    repeated = t.call(**payload)
    assert repeated == {**first, "replayed": True}
    assert (owner.directory / "workspace.json").read_bytes() == before
    assert not t.call(entries=[entry(owner, 病害描述="修改参数")])["ok"]


def test_post_commit_journal_failure_recovers_proven_commit(owner, monkeypatch):
    payload = {"entries": [entry(owner)]}
    native(monkeypatch, "save_defects", payload)
    real_atomic = engine.atomic_json
    def fail_final(path, value):
        if any(r.get("phase") == "complete" for r in value.get("calls", {}).values()):
            raise OSError("simulated journal failure after workspace commit")
        return real_atomic(path, value)
    monkeypatch.setattr(engine, "atomic_json", fail_final)
    t = tool(owner)
    assert not t.can_replay_saved_call(payload)
    assert t.call(**payload)["error"]["details"]["applied"] is None
    assert len(owner.snapshot()["draft"]["病害明细"]) == 1
    before = (owner.directory / "workspace.json").read_bytes()
    journal_before = t.path.read_bytes()
    assert t.can_replay_saved_call(payload)
    assert t.path.read_bytes() == journal_before  # Predicate cannot finish the journal itself.
    assert not t.can_replay_saved_call({"entries": [entry(owner, 病害描述="另一个参数")]})
    assert owner.checkpoint_path.read_bytes() == before
    monkeypatch.setattr(engine, "atomic_json", real_atomic)
    result = t.call(**payload)
    assert result["ok"] and result["replayed"] and (owner.directory / "workspace.json").read_bytes() == before


def test_replay_saved_predicate_rejects_uncommitted_intent_changed_identity_and_direct_call(owner, monkeypatch):
    payload = {"entries": [entry(owner)]}
    invocation = native(monkeypatch, "save_defects", payload)
    t = tool(owner)
    real_apply = owner.apply_update
    def fail_before_commit(update):
        raise OSError("simulated failure before atomic save")
    monkeypatch.setattr(owner, "apply_update", fail_before_commit)
    assert t.call(**payload)["error"]["details"]["applied"] is None
    before = t.path.read_bytes()
    assert not t.can_replay_saved_call(payload)
    assert t.path.read_bytes() == before
    monkeypatch.setattr(owner, "apply_update", real_apply)
    assert t.call(**payload)["ok"]
    assert t.can_replay_saved_call(payload)
    changed = deepcopy(invocation); changed["generation_call_id"] = "another-generation"
    monkeypatch.setattr(engine, "_current_invocation", lambda: changed)
    assert not t.can_replay_saved_call(payload)
    changed = deepcopy(invocation); changed["native_tool_call_id"] = "another-native-call"
    assert not t.can_replay_saved_call(payload)
    changed = deepcopy(invocation); changed.pop("args_sha256")
    assert not t.can_replay_saved_call(payload)
    changed = deepcopy(invocation); changed["tool_name"] = "save_recommendations"
    assert not t.can_replay_saved_call(payload)
    monkeypatch.setattr(engine, "_current_invocation", lambda: None)
    assert not t.can_replay_saved_call(payload)


def test_deleted_supplement_not_implicitly_restored(owner):
    assert tool(owner).call(entries=[entry(owner)])["ok"]
    state = owner.snapshot()
    owner.apply_update({"base_version": state["version"], "draft_patch": [
        {"op": "remove_row", "section": "病害明细", "index": 0}]})
    before = (owner.directory / "workspace.json").read_bytes()
    result = tool(owner).call(entries=[entry(owner)])
    assert not result["ok"] and result["error"]["code"] == "SUPPLEMENT_PREVIOUSLY_UNBOUND"
    assert (owner.directory / "workspace.json").read_bytes() == before


def test_material_alias_preserves_origin_and_rejects_foreign_document(tmp_path):
    case = make_material_case(tmp_path / "source")
    index = DocumentMaterialIndex(case["manifest"], case["materials"], source_path=case["source_path"])
    owner = DocumentWorkspace(case["manifest"], tmp_path / "main", {"model": "synthetic"}, material_index=index)
    alias = index.qualify_ref(case["refs"]["table"] + "/r1/c1")
    refs, provenance = normalize_material_refs([{"evidence_id": alias}], owner.manifest, index)
    for ref in refs:
        owner.record_read(ref["evidence_id"], ref["start"], ref["end"])
    row = {"病害描述": "轻微锈蚀", "source_refs": [{"evidence_id": alias}], "status": "explicit"}
    result = tool(owner).call(entries=[row])
    assert result["ok"]
    state = owner.snapshot()
    identifier = state["draft_bindings"]["病害明细.0"][0]
    assert state["facts"][identifier]["source_refs"] == refs
    assert state["material_provenance"]["facts"][identifier] == provenance
    canonical = deepcopy(row); canonical["source_refs"] = refs
    assert not tool(owner).call(entries=[canonical])["changed"]
    assert owner.snapshot()["material_provenance"]["facts"][identifier] == provenance
    before = (owner.directory / "workspace.json").read_bytes()
    row["source_refs"][0]["evidence_id"] = "mineru:" + "0" * 64 + case["refs"]["unique"]
    result = tool(owner).call(entries=[row])
    assert not result["ok"] and result["error"]["details"]["path"] == "entries[0].source_refs[0].evidence_id"
    assert (owner.directory / "workspace.json").read_bytes() == before


def test_new_save_changes_review_identity_and_keeps_unknowns_unknown(owner):
    from bridge_evidence.contest.document_speed_workflow import SpeedMergeWorkflow
    wf = object.__new__(SpeedMergeWorkflow)
    wf.workspace = owner
    wf.directory = owner.directory
    wf.quick_review_receipt = {"semantic_state_sha256": wf._semantic_identity(owner.snapshot())}
    assert wf._quick_current()
    assert tool(owner).call(entries=[entry(owner)])["ok"]
    assert not wf._quick_current()
    assert "是否新增" not in owner.snapshot()["draft"]["病害明细"][0]


def test_schema_real_arrays_no_ids_and_small_batches(owner):
    for name, schema in SUPPLEMENT_SCHEMAS.items():
        assert schema["properties"]["entries"]["maxItems"] == 5
        assert "id" not in schema["properties"]["entries"]["items"]["properties"]
        assert Draft202012Validator(schema).is_valid({"entries": [entry(owner, name=name)]})


def test_existing_children_facts_rows_and_inventory_preserved(case):
    before = case.workspace.snapshot()
    t = MainSupplementTools(case.workspace, case.handoff, case.summary, "save_defects")
    result = t.call(entries=[entry(case.workspace, 4, 病害描述="模型补充的裂缝记录")])
    assert result["ok"] and result["added_count"] == 1
    state = case.workspace.snapshot()
    assert state["draft"]["病害明细"][:-1] == before["draft"]["病害明细"]
    assert state["draft"]["维修建议"] == before["draft"]["维修建议"]
    assert {key: state["facts"][key] for key in before["facts"]} == before["facts"]
    assert t.handoff == case.handoff and t.children == case.summary
    for path, identifiers in before["draft_bindings"].items():
        assert state["draft_bindings"][path] == identifiers


def test_common_spelling_alias_normalized_but_conflict_rejected(owner):
    t = tool(owner)
    result = t.call(entries=[entry(owner, development_degree="轻微")])
    assert result["ok"] and owner.snapshot()["draft"]["病害明细"][0]["发展程度"] == "轻微"
    before = owner.checkpoint_path.read_bytes()
    result = t.call(entries=[entry(owner, development_degree="轻微", 发展程度="明显")])
    assert not result["ok"] and result["error"]["details"]["path"] == "entries[0].development_degree"
    assert owner.checkpoint_path.read_bytes() == before


def test_main_delivery_child_full_read_does_not_authorize_main_partial(owner):
    unit = owner.manifest["units"][0]
    before = owner.checkpoint_path.read_bytes()
    row = entry(owner)
    partial = {unit["id"]: [[0, 2]]}
    # The workspace already includes a child's full reading coverage.
    assert owner.snapshot()["reads"][unit["id"]] == [[0, len(unit["text"])]]
    result = validate_main_delivery([row], owner, partial)
    assert not result["ok"] and result["error"]["code"] == "MAIN_SOURCE_NOT_DELIVERED"
    assert result["error"]["details"]["path"] == "entries[0].source_refs[0]"
    assert owner.checkpoint_path.read_bytes() == before
    # Only real confirmed main ranges may authorize a whole-unit reference.
    full = {unit["id"]: [[0, 2], [2, len(unit["text"])]]}
    assert validate_main_delivery([row], owner, full) == {"ok": True}
    assert tool(owner).call(entries=[row])["ok"]


def test_main_delivery_exact_partial_range_and_unavailable_ledger(owner):
    unit = owner.manifest["units"][0]
    row = entry(owner)
    row["source_refs"][0].update(start=0, end=2)
    assert validate_main_delivery([row], owner, {unit["id"]: [[0, 2]]})["ok"]
    result = validate_main_delivery([row], owner, None)
    assert not result["ok"] and result["error"]["code"] == "MAIN_SOURCE_DELIVERY_UNAVAILABLE"
    assert not validate_main_delivery('[{"病害描述":', owner, {})["ok"]


def test_main_delivery_material_alias_checks_only_exact_canonical_span(tmp_path):
    case = make_material_case(tmp_path / "source")
    index = DocumentMaterialIndex(case["manifest"], case["materials"], source_path=case["source_path"])
    owner = DocumentWorkspace(case["manifest"], tmp_path / "main", {"model": "synthetic"}, material_index=index)
    alias = index.qualify_ref(case["refs"]["table"] + "/r1/c1")
    refs, _ = normalize_material_refs([{"evidence_id": alias}], owner.manifest, index)
    row = {"病害描述": "轻微锈蚀", "source_refs": [{"evidence_id": alias}], "status": "explicit"}
    ranges = {}
    for ref in refs:
        ranges.setdefault(ref["evidence_id"], []).append([ref["start"], ref["end"]])
    assert validate_main_delivery([row], owner, ranges)["ok"]
    result = validate_main_delivery([row], owner, {})
    assert not result["ok"] and result["error"]["details"]["path"] == "entries[0].source_refs[0]"
    row["source_refs"][0]["evidence_id"] = "mineru:" + "0" * 64 + case["refs"]["unique"]
    result = validate_main_delivery([entry(owner), row], owner, {})
    # A valid but undelivered earlier entry fails before any later entry.
    assert not result["ok"] and result["error"]["details"]["path"] == "entries[0].source_refs[0]"
    result = validate_main_delivery([row], owner, ranges)
    assert not result["ok"] and result["error"]["details"]["path"] == "entries[0].source_refs[0].evidence_id"
