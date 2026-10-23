"""Synthetic provenance/persistence tests; no model or report data."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import multiprocessing
import os

import pytest

from bridge_evidence.contest import document_entries as module
from bridge_evidence.contest.document_entries import EntryStore, EntryStoreError, entry_schema
from bridge_evidence.contest.docx_writer import ADVICE_FIELDS, BASIC_FIELDS, DEFECT_FIELDS
from bridge_evidence.contest.evidence_manifest import manifest_from_text


def manifest(text="桥名示例\n2018年度\n左幅梁裂缝"):
    return manifest_from_text(text)


def store(tmp_path):
    source = manifest()
    result = EntryStore(source, "defects-1", tmp_path / "task")
    for unit in source["units"]:
        result.record_read(unit["id"], 0, len(unit["text"]))
    return result


def basic(entry_id="name", evidence_id="E000001", value="桥名示例"):
    return {"id": entry_id, "section": "基本信息", "field": "桥梁名称", "value": value,
            "status": "explicit", "source_refs": [{"evidence_id": evidence_id}]}


def defect(entry_id="crack", evidence_id="E000003"):
    return {"id": entry_id, "section": "病害明细", "row": {"病害描述": "裂缝", "是否新增": None},
            "status": "uncertain", "source_refs": [{"evidence_id": evidence_id}]}


def advice(entry_id="repair", evidence_id="E000003"):
    return {"id": entry_id, "section": "维修建议", "row": {"建议内容": "修补"},
            "status": "inferred", "source_refs": [{"evidence_id": evidence_id}]}


def test_batch_persists_rows_provenance_and_counts_without_completeness(tmp_path):
    ledger = store(tmp_path)
    batch = [basic(), defect(), advice()]
    original = deepcopy(batch)
    receipt = ledger.save_entries(batch)
    assert receipt["ok"] is True and receipt["changed"] is True
    assert receipt["counts"] == {"基本信息": 1, "病害明细": 1, "维修建议": 1, "total": 3}
    assert receipt["delivery_complete"] is False
    snapshot = ledger.snapshot()
    assert batch == original
    assert snapshot["entries"]["crack"]["source_refs"] == [{"evidence_id": "E000003", "start": 0, "end": 5}]
    assert snapshot["entries"]["crack"]["row"] == original[1]["row"]
    assert snapshot["reads"]["E000003"] == [[0, 5]]
    assert snapshot["identity"]["task_id"] == "defects-1"
    assert "units" not in snapshot
    reloaded = EntryStore(manifest(), "defects-1", ledger.directory)
    assert reloaded.snapshot() == snapshot
    snapshot["entries"]["crack"]["row"]["病害描述"] = "调用方改动"
    assert reloaded.snapshot()["entries"]["crack"]["row"]["病害描述"] == "裂缝"


def test_stable_upsert_and_identical_retry_do_not_duplicate_rows(tmp_path):
    ledger = store(tmp_path)
    first = ledger.save_entries([defect()])
    retry = ledger.save_entries([defect()])
    assert retry["changed"] is False and retry["version"] == first["version"]
    corrected = defect()
    corrected["row"]["病害描述"] = "模型修订的原值"
    last = ledger.save_entries([corrected])
    assert last["counts"]["病害明细"] == 1 and last["version"] == first["version"] + 1
    assert ledger.snapshot()["entry_order"] == ["crack"]
    assert ledger.read_entries()["entries"][0] == ledger.snapshot()["entries"]["crack"]


def test_equivalent_whole_and_explicit_references_are_idempotent(tmp_path):
    ledger = store(tmp_path)
    first = ledger.save_entries([basic()])
    equivalent = basic()
    equivalent["source_refs"][0].update(start=0, end=5)
    assert ledger.save_entries([equivalent])["version"] == first["version"]


def test_upsert_preserves_explicit_numeric_representation(tmp_path):
    ledger = store(tmp_path)
    ledger.save_entries([basic(value=0)])
    receipt = ledger.save_entries([basic(value=0.0)])
    assert receipt["changed"]
    assert type(ledger.snapshot()["entries"]["name"]["value"]) is float


def test_empty_result_is_not_complete_delivery(tmp_path):
    ledger = EntryStore(manifest(), "empty", tmp_path)
    receipt = ledger.save_entries([])
    assert receipt["ok"] and not receipt["changed"]
    assert receipt["counts"]["total"] == 0 and receipt["delivery_complete"] is False
    assert ledger.snapshot()["entries"] == {}


def test_any_invalid_entry_rejects_entire_batch_and_keeps_old_bytes(tmp_path):
    ledger = store(tmp_path)
    ledger.save_entries([basic()])
    before = ledger.path.read_bytes()
    bad = advice()
    bad["row"]["非导出字段"] = "不能悄悄丢失"
    with pytest.raises(EntryStoreError) as caught:
        ledger.save_entries([defect(), bad])
    error = caught.value.as_tool_error()
    assert error["ok"] is False and error["error"]["code"] == "OUTPUT_FIELD"
    assert error["error"]["details"]["path"] == "entries[1].row.非导出字段"
    assert error["error"]["details"]["entry_id"] == "repair"
    assert error["error"]["details"]["applied"] is False
    assert ledger.path.read_bytes() == before
    assert set(ledger.snapshot()["entries"]) == {"name"}


@pytest.mark.parametrize("ref,code", [
    ({"evidence_id": "E000001", "start": 0}, "SOURCE_FIELDS"),
    ({"evidence_id": "E000001", "end": 4}, "SOURCE_FIELDS"),
    ({"evidence_id": "E000001", "start": None, "end": 4}, "SOURCE_RANGE"),
    ({"evidence_id": "E000001", "start": True, "end": 4}, "SOURCE_RANGE"),
    ({"evidence_id": "E000001", "start": -1, "end": 4}, "SOURCE_RANGE"),
    ({"evidence_id": "E000001", "start": 3, "end": 2}, "SOURCE_RANGE"),
    ({"evidence_id": "E000001", "start": 0, "end": 6}, "SOURCE_RANGE"),
    ({"evidence_id": "E000001", "start": 0, "end": 0}, "SOURCE_EMPTY"),
    ({"evidence_id": "other-document"}, "SOURCE_ID"),
])
def test_explicit_bad_source_ranges_never_clamp(tmp_path, ref, code):
    ledger = store(tmp_path)
    before = ledger.path.read_bytes()
    entry = basic()
    entry["source_refs"] = [ref]
    with pytest.raises(EntryStoreError) as caught:
        ledger.save_entries([entry])
    assert caught.value.code == code
    assert caught.value.details["path"].startswith("entries[0].source_refs[0]")
    if code == "SOURCE_RANGE":
        assert caught.value.details["actual_length"] == 5
    assert ledger.path.read_bytes() == before


def test_whole_unit_requires_complete_actually_delivered_coverage(tmp_path):
    ledger = EntryStore(manifest(), "parts", tmp_path)
    ledger.record_read("E000001", 0, 1)
    ledger.record_read("E000001", 2, 5)
    entry = basic()
    with pytest.raises(EntryStoreError, match="entire reference"):
        ledger.save_entries([entry])
    explicit = deepcopy(entry)
    explicit["source_refs"][0].update(start=2, end=4)
    assert ledger.save_entries([explicit])["ok"]
    ledger.record_read("E000001", 1, 2)
    assert ledger.snapshot()["reads"]["E000001"] == [[0, 5]]
    assert ledger.save_entries([entry])["ok"]


def test_global_cross_paragraph_reference_is_legal_only_when_read(tmp_path):
    ledger = EntryStore(manifest(), "one-task", tmp_path)
    ledger.record_read("E000003", 0, 5)
    entry = defect()
    entry["source_refs"].append({"evidence_id": "E000001"})
    with pytest.raises(EntryStoreError) as caught:
        ledger.save_entries([entry])
    assert caught.value.code == "SOURCE_UNREAD"
    assert caught.value.details["path"] == "entries[0].source_refs[1]"
    ledger.record_read("E000001", 0, 5)
    assert ledger.save_entries([entry])["ok"]
    assert len(ledger.snapshot()["entries"]["crack"]["source_refs"]) == 2


@pytest.mark.parametrize("missing", ["id", "section", "value", "field", "source_refs", "status"])
def test_missing_required_entry_fields_rejected(tmp_path, missing):
    ledger = store(tmp_path)
    entry = basic()
    del entry[missing]
    with pytest.raises(EntryStoreError):
        ledger.save_entries([entry])
    assert ledger.snapshot()["entries"] == {}


@pytest.mark.parametrize("value", [True, {}, [], float("nan"), float("inf"), "bad\x00XML"])
def test_values_exporter_would_drop_or_stringify_are_refused(tmp_path, value):
    ledger = store(tmp_path)
    with pytest.raises(EntryStoreError) as caught:
        ledger.save_entries([basic(value=value)])
    assert caught.value.code == "OUTPUT_VALUE"


@pytest.mark.parametrize("value", [None, "未知", "", 0, 0.0])
def test_unknown_and_zero_values_are_preserved_never_filled(tmp_path, value):
    ledger = store(tmp_path)
    entry = basic(value=value)
    entry["status"] = "uncertain"
    ledger.save_entries([entry])
    saved = ledger.snapshot()["entries"]["name"]
    assert saved["value"] == value and type(saved["value"]) is type(value)
    assert "总体评分" not in saved
    assert "来源没有提供" not in json.dumps(saved, ensure_ascii=False)


@pytest.mark.parametrize("status", module.ENTRY_STATUSES)
def test_all_statuses_need_real_source_provenance(tmp_path, status):
    ledger = store(tmp_path)
    entry = basic()
    entry.update(status=status, source_refs=[])
    with pytest.raises(EntryStoreError) as caught:
        ledger.save_entries([entry])
    assert caught.value.code == "SOURCE_REFS"


def test_duplicate_batch_ids_and_changed_identity_do_not_replace_old_entry(tmp_path):
    ledger = store(tmp_path)
    ledger.save_entries([basic()])
    before = ledger.path.read_bytes()
    with pytest.raises(EntryStoreError) as caught:
        ledger.save_entries([defect(), defect()])
    assert caught.value.code == "ENTRY_DUPLICATE"
    changed = basic()
    changed["field"] = "总体评分"
    with pytest.raises(EntryStoreError) as caught:
        ledger.save_entries([changed])
    assert caught.value.code == "ENTRY_IDENTITY"
    assert ledger.path.read_bytes() == before


def test_different_document_or_task_cannot_adopt_existing_directory(tmp_path):
    ledger = store(tmp_path)
    ledger.save_entries([basic()])
    before = ledger.path.read_bytes()
    for source, task in [(manifest("另一个文档"), "defects-1"), (manifest(), "other-task")]:
        with pytest.raises(EntryStoreError) as caught:
            EntryStore(source, task, ledger.directory)
        assert caught.value.code == "STORE_IDENTITY"
    assert ledger.path.read_bytes() == before


def test_concurrent_tasks_have_independent_local_ids_and_provenance(tmp_path):
    first = EntryStore(manifest(), "A", tmp_path / "A")
    second = EntryStore(manifest(), "B", tmp_path / "B")
    for ledger in (first, second):
        ledger.record_read("E000001", 0, 5)
    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(ledger.save_entries, [basic(value=value)])
                   for ledger, value in ((first, "甲模型值"), (second, "乙模型值"))]
        assert all(future.result()["ok"] for future in futures)
    assert first.snapshot()["entries"]["name"]["value"] == "甲模型值"
    assert second.snapshot()["entries"]["name"]["value"] == "乙模型值"
    assert first.identity["task_sha256"] != second.identity["task_sha256"]
    assert first.identity["manifest_sha256"] == second.identity["manifest_sha256"]


def test_two_instances_of_same_task_serialize_upserts(tmp_path):
    first = store(tmp_path)
    second = EntryStore(manifest(), "defects-1", first.directory)
    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(first.save_entries, [basic()]), executor.submit(second.save_entries, [defect()])]
        assert all(future.result()["ok"] for future in futures)
    assert first.snapshot()["counts"]["total"] == 2


def test_filesystem_replace_failure_preserves_old_batch(tmp_path, monkeypatch):
    ledger = store(tmp_path)
    ledger.save_entries([basic()])
    before = ledger.path.read_bytes()

    def fail_replace(*args):
        raise OSError("synthetic disk failure before commit")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(EntryStoreError) as caught:
        ledger.save_entries([defect(), advice()])
    assert caught.value.code == "STORE_WRITE" and caught.value.details["applied"] is False
    assert ledger.path.read_bytes() == before
    assert list(ledger.snapshot()["entries"]) == ["name"]
    assert not list(ledger.directory.glob("*.tmp"))


def test_uncertain_filesystem_ack_does_not_claim_success_or_partial_save(tmp_path, monkeypatch):
    ledger = store(tmp_path)
    ledger.save_entries([basic()])
    real_replace = os.replace

    def replace_then_error(*args):
        real_replace(*args)
        raise OSError("synthetic uncertain acknowledgement")

    monkeypatch.setattr(os, "replace", replace_then_error)
    with pytest.raises(EntryStoreError) as caught:
        ledger.save_entries([defect(), advice()])
    assert caught.value.details["applied"] is None
    assert caught.value.as_tool_error()["ok"] is False
    assert set(ledger.snapshot()["entries"]) == {"name", "crack", "repair"}


def _crash_before_replace(directory):
    ledger = EntryStore(manifest(), "defects-1", directory)
    os.replace = lambda *args: os._exit(73)
    ledger.save_entries([defect(), advice()])


def test_real_writer_process_crash_before_atomic_rename_keeps_last_complete_state(tmp_path):
    ledger = store(tmp_path)
    ledger.save_entries([basic()])
    before = ledger.path.read_bytes()
    child = multiprocessing.get_context("fork").Process(target=_crash_before_replace, args=(ledger.directory,))
    child.start()
    child.join(5)
    if child.is_alive():
        child.terminate()
        child.join()
        pytest.fail("Synthetic crash writer did not exit")
    assert child.exitcode == 73
    assert ledger.path.read_bytes() == before
    reopened = EntryStore(manifest(), "defects-1", ledger.directory)
    assert set(reopened.snapshot()["entries"]) == {"name"}
    assert reopened.save_entries([defect(), advice()])["counts"]["total"] == 3


def test_paging_keeps_full_rows_and_validation_is_pure(tmp_path):
    ledger = store(tmp_path)
    long = defect("long")
    long["row"]["病害描述"] = "模型内容" * 3000
    original_bytes = ledger.path.read_bytes()
    normalized = ledger.validate_entries([long])
    assert ledger.path.read_bytes() == original_bytes
    assert normalized[0]["source_refs"][0]["end"] == 5
    ledger.save_entries([basic(), long, advice()])
    first = ledger.read_entries(0, 1)
    second = ledger.read_entries(first["next_cursor"], 1)
    third = ledger.read_entries(second["next_cursor"], 8)
    assert first["entries"][0]["id"] == "name"
    assert second["entries"][0]["row"] == long["row"]
    assert third["next_cursor"] is None
    assert [entry["id"] for entry in third["entries"]] == ["repair"]


def test_schema_uses_exporter_fields_and_is_not_mutable_shared_state():
    schema = entry_schema()
    variants = {variant["properties"]["section"]["enum"][0]: variant for variant in schema["anyOf"]}
    assert variants["基本信息"]["properties"]["field"]["enum"] == [name for name, _ in BASIC_FIELDS]
    for section, fields in (("病害明细", DEFECT_FIELDS), ("维修建议", ADVICE_FIELDS)):
        assert list(variants[section]["properties"]["row"]["properties"]) == fields
        assert variants[section]["properties"]["row"]["additionalProperties"] is False
    assert all(variant["additionalProperties"] is False for variant in schema["anyOf"])
    schema["anyOf"].clear()
    assert len(entry_schema()["anyOf"]) == 3


def test_tampered_persisted_read_provenance_fails_closed(tmp_path):
    ledger = store(tmp_path)
    ledger.save_entries([basic()])
    state = json.loads(ledger.path.read_text())
    state["reads"]["E000001"] = [[0, 3]]
    ledger.path.write_text(json.dumps(state, ensure_ascii=False))
    with pytest.raises(EntryStoreError) as caught:
        ledger.snapshot()
    assert caught.value.code == "STORE_STATE"
