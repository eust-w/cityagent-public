"""Frozen omission-review retrieval contracts with synthetic data only.

These tools expose evidence and candidate draft entries; none of these checks
treats literal matches, citation IDs or successful reads as semantic proof.
"""
from copy import deepcopy
import json

import pytest

from bridge_evidence.contest.document_review_lookup import FrozenDraftLookup, LOOKUP_TOOLS
from bridge_evidence.contest.run_state import json_digest


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def draft_items(count=7):
    return [{"path": f"病害明细.{index}",
             "value": {"病害描述": f"左幅{index + 1}号梁裂缝：" + "合成说明。" * 30},
             "facts": [{"id": f"F{index}-{j}", "value": f"2018年左幅{index + 1}号梁宽{j}mm",
                        "source_refs": [{"evidence_id": f"E{index}"}]} for j in range(4)],
             "sources": [{"evidence_id": f"E{index}", "start": j * 10, "end": (j + 1) * 10,
                          "text": f"第{j}段合成原文", "context": {"year": "2018"}} for j in range(4)]}
            for index in range(count)]


def lookup(items=None, *, budget=1024, read_source=None):
    # One synthetic token per character makes envelope accounting exact and
    # transparent. Production supplies the pinned model's real tokenizer.
    return FrozenDraftLookup(draft_items() if items is None else items,
                             count_tokens=len, max_tool_tokens=budget, read_source=read_source)


def collect_pages(reader, name, **arguments):
    cursor, result, snapshot = 0, [], None
    for _ in range(100):
        page = reader.invoke(name, {**arguments, "cursor": cursor, "limit": 100})
        assert page["ok"]
        assert len(encoded(page)) <= reader.max_tool_tokens
        assert page["start"] == cursor
        assert page["end"] - page["start"] == len(page["items"])
        if snapshot is None:
            snapshot = page["snapshot_sha256"]
        assert page["snapshot_sha256"] == snapshot == reader.snapshot_sha256
        result.extend(page["items"])
        if page["next_cursor"] is None:
            assert page["end"] == page["total"]
            return result
        assert page["next_cursor"] > cursor
        cursor = page["next_cursor"]
    pytest.fail("Pagination did not finish")


def test_entire_frozen_draft_is_accessible_beyond_current_citation_candidates():
    items = draft_items(17)
    reader = lookup(items, budget=800)
    # An item supported by E16 remains discoverable while reviewing E0.
    found = reader.invoke("review_search_draft", {"query": "左幅17号梁", "cursor": 0, "limit": 100})
    assert [item["path"] for item in found["items"]] == ["病害明细.16"]
    assert reader.seen_paths == {"病害明细.16"}
    assert not reader.all_values_seen
    values = collect_pages(reader, "review_read_draft_page")
    assert values == [{"path": item["path"], "value": item["value"]} for item in items]
    assert reader.all_values_seen
    assert len([call for call in reader.calls if call["name"] == "review_read_draft_page"]) > 1


@pytest.mark.parametrize("view", ["facts", "sources"])
def test_item_fact_and_source_pages_are_complete_and_do_not_claim_draft_value_coverage(view):
    items = draft_items(1)
    # Force multiple pages with complete records, preserving original order.
    items[0][view] *= 8
    reader = lookup(items, budget=650)
    result = collect_pages(reader, "review_read_draft_item", path=items[0]["path"], view=view)
    assert result == items[0][view]
    assert reader.seen_paths == set()
    assert not reader.all_values_seen


def test_input_and_returned_pages_cannot_mutate_frozen_snapshot():
    original = draft_items(1)
    frozen = deepcopy(original)
    reader = lookup(original)
    digest = reader.snapshot_sha256
    original[0]["value"]["病害描述"] = "外部改写"
    original[0]["facts"][0]["source_refs"][0]["evidence_id"] = "E_CHANGED"
    original[0]["sources"][0]["text"] = "外部源文改写"
    original.append(draft_items(2)[1])
    full = reader.metadata(full=True)
    full["items"][0]["value"]["病害描述"] = "返回值改写"
    for view in ("facts", "sources"):
        page = reader.invoke("review_read_draft_item", {"path": frozen[0]["path"], "view": view, "cursor": 0, "limit": 1})
        page["items"][0]["mutated"] = "返回记录改写"
    values = reader.invoke("review_read_draft_page", {"cursor": 0, "limit": 1})
    values["items"][0]["value"]["病害描述"] = "返回页改写"
    assert reader.items == frozen
    assert reader.snapshot_sha256 == digest == json_digest(frozen)
    assert reader.metadata(full=True)["items"] == [{"path": frozen[0]["path"], "value": frozen[0]["value"]}]


def test_search_semantics_are_budgeted_before_page_growth_instead_of_discarding_results():
    items = draft_items(4)
    probe = lookup(items, budget=10000)
    without_semantics = probe.invoke("review_read_draft_page", {"cursor": 0, "limit": 2})
    # Exactly two ordinary rows fit; the search envelope's extra disclaimer
    # means only one search row fits. This reproduced the former all-error bug.
    reader = lookup(items, budget=len(encoded(without_semantics)))
    page = reader.invoke("review_search_draft", {"query": "裂缝", "cursor": 0, "limit": 2})
    assert page["ok"]
    assert len(page["items"]) == 1
    assert page["next_cursor"] == 1 and page["total"] == 4
    assert "literal substring" in page["search_semantics"]
    assert len(encoded(page)) <= reader.max_tool_tokens
    assert reader.seen_paths == {page["items"][0]["path"]}
    assert not reader.all_values_seen
    assert collect_pages(reader, "review_search_draft", query="裂缝") == [
        {"path": item["path"], "value": item["value"]} for item in items]
    assert reader.all_values_seen


def test_empty_literal_search_is_not_an_omission_or_completion_verdict():
    reader = lookup()
    result = reader.invoke("review_search_draft", {"query": "混凝土开裂的同义表述", "cursor": 0, "limit": 10})
    assert result["ok"] and result["items"] == [] and result["total"] == 0
    assert "no result is not an omission verdict" in result["search_semantics"]
    assert "missing_items" not in result
    assert reader.seen_paths == set() and not reader.all_values_seen
    policy = reader.metadata()
    assert policy["linked_candidates_are_not_complete_draft"]
    assert "inspect all draft values" in policy["policy"]
    assert "unresolved_items" in policy["policy"]


@pytest.mark.parametrize("tool,extra,section", [
    ("review_read_draft_page", {}, "value"),
    ("review_search_draft", {"query": "巨大"}, "value"),
    ("review_read_draft_item", {"path": "病害明细.0", "view": "facts"}, "facts"),
    ("review_read_draft_item", {"path": "病害明细.0", "view": "sources"}, "sources"),
])
def test_indivisible_oversized_record_is_explicitly_unresolved_without_truncation(tool, extra, section):
    items = draft_items(1)
    private = "巨大但不可截断的合成文本" * 500
    if section == "value":
        items[0][section] = {"病害描述": private}
    else:
        items[0][section][0]["value" if section == "facts" else "text"] = private
    reader = lookup(items, budget=600)
    result = reader.invoke(tool, {**extra, "cursor": 0, "limit": 100})
    assert not result["ok"] and result["code"] == "lookup_record_too_large"
    assert result["unresolved"] is True
    assert "unresolved" in result["next_action"]
    assert "No record was shortened" in result["next_action"]
    assert "items" not in result and private not in encoded(result)
    assert len(encoded(result)) <= reader.max_tool_tokens
    assert reader.seen_paths == set() and not reader.all_values_seen


def test_only_returned_value_rows_are_counted_as_seen_not_requested_or_metadata_rows():
    reader = lookup(budget=600)
    reader.metadata(full=True)  # Preparing metadata is not evidence of delivery.
    assert reader.seen_paths == set()
    first = reader.invoke("review_read_draft_page", {"cursor": 0, "limit": 100})
    assert 0 < len(first["items"]) < first["total"]
    assert reader.seen_paths == {row["path"] for row in first["items"]}
    unchanged = reader.seen_paths.copy()
    reader.invoke("review_read_draft_page", {"cursor": 999, "limit": 100})
    reader.invoke("review_search_draft", {"query": "未提及事项", "cursor": 0, "limit": 100})
    reader.invoke("review_read_draft_page", {"cursor": first["total"], "limit": 100})
    assert reader.seen_paths == unchanged and not reader.all_values_seen


@pytest.mark.parametrize("name,arguments", [
    ("shell /private/secret-report", {}), ([], {}), ("review_read_draft_page", None),
    ("review_read_draft_page", []), ("review_read_draft_page", {"cursor": 0}),
    ("review_read_draft_page", {"cursor": 0, "limit": 1, "extra": "/private/secret-report"}),
    *[("review_read_draft_page", {"cursor": cursor, "limit": limit})
      for cursor, limit in [(-1, 1), (False, 1), (0.0, 1), ("0", 1), (99, 1), (0, 0), (0, 101), (0, True), (0, 1.5)]],
    *[("review_search_draft", {"query": query, "cursor": 0, "limit": 1}) for query in ["", " ", True, [], "x" * 513]],
    ("review_read_draft_item", {"path": "/private/secret-report", "view": "sources", "cursor": 0, "limit": 1}),
    ("review_read_draft_item", {"path": [], "view": "facts", "cursor": 0, "limit": 1}),
    ("review_read_draft_item", {"path": "病害明细.0", "view": "history", "cursor": 0, "limit": 1}),
    *[("review_read_source", {"evidence_id": "E0", "start": start, "end": end})
      for start, end in [(-1, 1), (False, 1), (0, True), (0.0, 1), (0, "1"), (2, 1), (0, 0)]],
    *[("review_read_source", {"evidence_id": identifier, "start": 0, "end": 1}) for identifier in ["", " ", [], True]],
    ("review_read_source", {"evidence_id": "E0", "start": 0, "end": 1, "path": "/private/secret-report"}),
    ("review_read_draft_page", {"cursor": 0, "limit": float("nan")}),
    ("review_read_draft_page", {"cursor": object(), "limit": 1}),
])
def test_unauthorized_tools_and_invalid_arguments_are_rejected_before_source_dispatch(name, arguments):
    source_calls = []
    reader = lookup(read_source=lambda **args: source_calls.append(args) or {"ok": True, "text": "secret-report"})
    result = reader.invoke(name, arguments)
    assert not result["ok"] and result["unresolved"]
    assert "do not infer absence" in result["next_action"]
    assert source_calls == [] and reader.seen_paths == set()
    assert "secret-report" not in encoded(result) + encoded(reader.calls)
    assert len(encoded(result)) <= reader.max_tool_tokens


@pytest.mark.parametrize("failure", ["exception", "error_result", "invalid_result", "oversized"])
def test_source_backend_errors_and_oversized_responses_never_leak_raw_text(failure):
    secret = "/private/report secret-api-key 原文"

    def backend(**arguments):
        if failure == "exception":
            raise RuntimeError(secret)
        if failure == "error_result":
            return {"ok": False, "error": {"message": secret, "stack": secret}}
        if failure == "invalid_result":
            return [secret]
        return {"ok": True, "text": secret * 100}

    reader = lookup(read_source=backend, budget=600)
    result = reader.invoke("review_read_source", {"evidence_id": "E0", "start": 0, "end": 1})
    assert not result["ok"] and result["unresolved"]
    assert secret not in encoded(result) + encoded(reader.calls)
    assert reader.seen_paths == set()


def test_valid_original_source_range_is_forwarded_exactly_and_receipts_are_hash_only():
    calls = []
    source = {"ok": True, "span": {"evidence_id": "E0", "start": 3, "end": 8, "text": "合成原文段"},
              "continuation": {"start": 8, "end": 20}}

    def backend(**arguments):
        calls.append(arguments)
        return source

    reader = lookup(read_source=backend)
    arguments = {"evidence_id": "E0", "start": 3, "end": 8}
    result = reader.invoke("review_read_source", arguments)
    assert calls == [arguments]
    assert result == source and result is not source
    assert reader.calls[-1]["arguments_sha256"] == json_digest(arguments)
    assert reader.calls[-1]["result_sha256"] == json_digest(result)
    assert "合成原文段" not in encoded(reader.calls)
    source["span"]["text"] = "外部改写"
    assert result["span"]["text"] == "合成原文段"
    assert reader.seen_paths == set()


def test_unavailable_source_backend_is_explicitly_unresolved():
    result = lookup().invoke("review_read_source", {"evidence_id": "E0", "start": 0, "end": 1})
    assert not result["ok"] and result["unresolved"]


@pytest.mark.parametrize("budget", [0, -1, True, 1.0, None, 1])
def test_unusable_tool_budgets_are_rejected_before_review(budget):
    with pytest.raises(ValueError):
        lookup(budget=budget)


@pytest.mark.parametrize("count", [None, True, -1, 1.0])
def test_invalid_token_counts_cannot_disable_budget_checks(count):
    with pytest.raises(ValueError, match="count_tokens"):
        FrozenDraftLookup(draft_items(), count_tokens=lambda text: count, max_tool_tokens=1000)


def test_duplicate_frozen_paths_are_rejected_and_tool_schemas_forbid_extra_fields():
    item = draft_items(1)[0]
    with pytest.raises(ValueError, match="unique"):
        lookup([item, deepcopy(item)])
    assert {tool["function"]["name"] for tool in LOOKUP_TOOLS} == {
        "review_read_draft_page", "review_search_draft", "review_read_draft_item",
        "review_read_source", "review_read_source_catalog", "review_search_source",
        "review_read_source_section", "review_save_notes",
    }
    for tool in LOOKUP_TOOLS:
        parameters = tool["function"]["parameters"]
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])
        if "limit" in parameters["properties"]:
            assert parameters["properties"]["limit"]["maximum"] == 100


def test_save_notes_is_audited_bounded_working_memory_without_echo_or_coverage_claims():
    reader = lookup(budget=600)
    original_items = deepcopy(reader.items)
    notes = {"pending_source": ["E0:0:10"], "checked_paths": ["病害明细.0"],
             "reason": "待核跨段对象关系，不代表原文结论", "remaining": ["核对左右幅"]}
    result = reader.save_notes(encoded(notes))
    assert result == {"ok": True, "snapshot_sha256": reader.snapshot_sha256, "notes_sha256": json_digest(notes)}
    assert len(encoded(result)) <= reader.max_tool_tokens
    assert reader.notes == notes
    assert "notes" not in result and notes["reason"] not in encoded(reader.calls)
    assert reader.calls[-1]["name"] == "review_save_notes"
    assert reader.calls[-1]["arguments_sha256"] == json_digest({"notes_json": encoded(notes)})
    assert reader.calls[-1]["result_sha256"] == json_digest(result)
    assert reader.items == original_items
    assert reader.seen_paths == set() and not reader.all_values_seen
    # The caller receives exact note values, but cannot mutate the stored copy.
    returned = reader.notes
    returned["remaining"].append("外部改写")
    assert reader.notes == notes
    replaced = {"remaining": ["最后一项待核"]}
    assert reader.invoke("review_save_notes", {"notes_json": encoded(replaced)})["ok"]
    assert reader.notes == replaced and len(reader.calls) == 2


def test_notes_budget_includes_snapshot_envelope_and_failed_save_keeps_prior_notes():
    notes = {"pending": "待核事项" * 100}
    # The object alone fits exactly, but its complete snapshot envelope does not.
    budget = len(encoded(notes))
    reader = lookup(budget=budget)
    previous = {"pending": "先前已保存待查问题"}
    assert reader.save_notes(encoded(previous))["ok"]
    result = reader.save_notes(encoded(notes))
    assert not result["ok"] and result["code"] == "review_notes_too_large"
    assert "Shorten" in result["next_action"]
    assert result["unresolved"] and reader.notes == previous
    assert "待核事项" not in encoded(result)
    assert len(encoded(result)) <= budget
    assert reader.calls[-1]["ok"] is False


@pytest.mark.parametrize("notes_json", [None, {}, [], "", "null", "[]", "1", "true", "{invalid",
                                        '{"pending":1,"pending":2}', '{"nested":{"x":1,"x":2}}',
                                        '{"value":NaN}', '{"value":Infinity}', '{"value":1e9999}'])
def test_invalid_notes_do_not_change_previous_state_or_echo_the_input(notes_json):
    reader = lookup()
    reader.save_notes('{"pending":"already-saved"}')
    result = reader.save_notes(notes_json)
    assert not result["ok"] and result["code"] == "invalid_review_notes"
    assert result["unresolved"]
    assert reader.notes == {"pending": "already-saved"}
    assert reader.calls[-1]["ok"] is False


def test_saving_notes_does_not_replace_original_evidence_lookup():
    reader = lookup()
    reader.save_notes('{"working_hypothesis":"宽度可能是2mm，需要回读，尚非结论"}')
    result = reader.invoke("review_read_draft_item", {"path": "病害明细.0", "view": "sources", "cursor": 0, "limit": 1})
    assert result["ok"]
    assert result["items"] == draft_items()[0]["sources"][:1]
    assert "working_hypothesis" not in encoded(result)
    assert reader.notes["working_hypothesis"].endswith("尚非结论")


def checkpoint_reader():
    reader = lookup()
    reader.invoke("review_read_draft_page", {"cursor": 0, "limit": 1})
    reader.save_notes('{"checked":["病害明细.0"],"remaining":["E1待核"],"reason":"合成原由"}')
    reader.invoke("unknown /private/source", {})
    return reader


def reseal(state):
    state["state_sha256"] = json_digest({key: value for key, value in state.items() if key != "state_sha256"})
    return state


def test_lookup_state_round_trips_as_json_with_snapshot_calls_and_notes_isolated():
    reader = checkpoint_reader()
    state = reader.export_state()
    transported = json.loads(encoded(state))
    restored = lookup()
    restored.restore_state(transported)
    assert restored.snapshot_sha256 == reader.snapshot_sha256
    assert restored.seen_paths == reader.seen_paths == {"病害明细.0"}
    assert restored.notes == reader.notes and restored.calls == reader.calls
    assert restored.export_state() == state
    assert not restored.all_values_seen
    transported["notes"]["remaining"].append("不能污染恢复值")
    transported["calls"][0]["name"] = "不能污染恢复回执"
    transported["seen_paths"].append("病害明细.1")
    state["notes"]["reason"] = "不能污染原值"
    assert restored.notes == reader.notes
    assert restored.calls == reader.calls
    assert restored.seen_paths == reader.seen_paths
    # Resume at the remaining row, preserving the completed local lookups.
    result = restored.invoke("review_read_draft_page", {"cursor": 1, "limit": 1})
    assert result["items"][0]["path"] == "病害明细.1"
    assert restored.seen_paths == {"病害明细.0", "病害明细.1"}


def test_empty_lookup_state_can_be_restored_and_export_does_not_call_tokenizer_or_backend():
    reader = lookup(read_source=lambda **args: pytest.fail("export must not read source"))
    initial = reader.export_state()
    assert initial["notes"] is None and initial["seen_paths"] == [] and initial["calls"] == []
    restored = lookup()
    restored.restore_state(initial)
    assert restored.export_state() == initial
    reader.save_notes('{"pending":"save after model deadline"}')
    reader.count_tokens = lambda text: pytest.fail("export must not call a model tokenizer")
    exported = reader.export_state()
    assert exported["notes"] == reader.notes
    assert "read_source" not in exported and "count_tokens" not in exported


@pytest.mark.parametrize("change", [
    lambda state: state.update(schema_version=True),
    lambda state: state.update(schema_version=2),
    lambda state: state.update(snapshot_sha256="0" * 64),
    lambda state: state.update(seen_paths=["/private/unknown-source"]),
    lambda state: state.update(seen_paths=["病害明细.0", "病害明细.0"]),
    lambda state: state.update(seen_paths={"病害明细.0": True}),
    lambda state: state.update(notes=[]),
    lambda state: state.update(notes={"private": "private-secret" * 1000}),
    lambda state: state.update(calls={}),
    lambda state: state["calls"][0].update(name="/private/source-token"),
    lambda state: state["calls"][0].update(ok=1),
    lambda state: state["calls"][0].update(result_sha256="not-a-digest"),
    lambda state: state["calls"][0].update(arguments_sha256="A" * 64),
    lambda state: state["calls"][0].update(arguments_sha256=None),
    lambda state: state["calls"][0].update(name="invalid_lookup"),
    lambda state: state["calls"][0].update(raw_source="private-secret"),
    lambda state: state.update(execute="private-secret"),
    lambda state: state.pop("notes"),
])
def test_restoration_rejects_invalid_resealed_fields_atomically_without_echo(change):
    reader = checkpoint_reader()
    before = reader.export_state()
    damaged = deepcopy(before)
    change(damaged)
    reseal(damaged)
    with pytest.raises(ValueError, match="^Invalid or incompatible frozen lookup checkpoint$") as exc:
        reader.restore_state(damaged)
    assert "private" not in str(exc.value)
    assert reader.export_state() == before


def test_state_checksum_and_changed_frozen_draft_are_rejected():
    reader = checkpoint_reader()
    state = reader.export_state()
    damaged = deepcopy(state)
    damaged["notes"]["reason"] = "未重新校验的改写"
    with pytest.raises(ValueError):
        reader.restore_state(damaged)
    changed_items = draft_items()
    changed_items[0]["value"]["病害描述"] += "新版本"
    different = lookup(changed_items)
    with pytest.raises(ValueError):
        different.restore_state(state)
    assert different.notes is None and different.calls == []


@pytest.mark.parametrize("untrusted", [lambda: None, object(), {1: "non-string key"},
                                       ("tuple",), float("nan"), float("inf")])
def test_restore_rejects_non_json_objects_and_never_invokes_them(untrusted):
    reader = checkpoint_reader()
    state = reader.export_state()
    before = deepcopy(state)
    state["notes"] = {"untrusted": untrusted}
    with pytest.raises(ValueError, match="^Invalid or incompatible frozen lookup checkpoint$"):
        reader.restore_state(state)
    assert reader.export_state() == before


def test_python_looking_note_text_is_only_data_and_never_an_execution_instruction():
    reader = lookup()
    notes = {"quoted_source": "__import__('os').system('not-an-executable-action')"}
    assert reader.save_notes(encoded(notes))["ok"]
    restored = lookup()
    restored.restore_state(reader.export_state())
    assert restored.notes == notes
    assert restored.seen_paths == set()
