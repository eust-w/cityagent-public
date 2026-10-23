"""Synthetic lossless reverse-review packing; no model or tokenizer service."""
from copy import deepcopy
import json

import pytest

from bridge_evidence.contest.document_review_payloads import ReviewPayloadBudgetError, reverse_payloads
from bridge_evidence.contest.evidence_manifest import manifest_from_text


PROMPT = "核对原文遗漏；根据关联事实、原始证据和上下文判断。"


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def count_tokens(text):
    # Exact character accounting intentionally makes all tests deterministic.
    return len(text)


def count_chat_tokens(messages):
    return len(compact(messages)) + 20


def pack(manifest, items=(), background=None, **kwargs):
    options = dict(count_chat_tokens=count_chat_tokens, count_tokens=count_tokens,
                   max_model_len=65536, output_tokens=600, source_tokens=10000,
                   review_prompt=PROMPT)
    options.update(kwargs)
    return reverse_payloads(manifest, list(items), background or {}, **options)


def source(manifest, index=0, start=0, end=None):
    unit = manifest["units"][index]
    if end is None:
        end = len(unit["text"])
    return {"evidence_id": unit["id"], "start": start, "end": end,
            "source_path": unit["source_path"], "text": unit["text"][start:end],
            "context": deepcopy(unit["context"])}


def output(path, sources, facts=None, value=None):
    return {"path": path, "value": value or {"病害描述": "原文观察"},
            "facts": facts or [], "sources": sources}


def assert_coverage(manifest, payloads):
    actual = [span for payload in payloads for span in payload["source_spans"]]
    assert all(span["end"] > span["start"] for span in actual)
    for unit in manifest["units"]:
        intervals = [(span["start"], span["end"]) for span in actual if span["evidence_id"] == unit["id"]]
        cursor = 0
        for start, end in intervals:
            assert start == cursor, "source coverage must have neither gaps nor repeated primary characters"
            cursor = end
        assert cursor == len(unit["text"])


def assert_budget(payload, maximum, output_tokens):
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": compact(payload)}]
    assert count_chat_tokens(messages) + output_tokens <= maximum
    minimal_reply = {"passed": True, "checked_output_paths": [],
                     "checked_evidence_spans": payload["source_spans"],
                     "unsupported_claims": [], "missing_items": [], "conflicts": [], "unresolved_items": []}
    assert count_tokens(compact(minimal_reply)) <= output_tokens


def test_long_original_is_covered_once_and_complete_payloads_fit():
    manifest = manifest_from_text("合成报告标题\n" + "长段落原文。" * 700 + "最终位置标记")
    payloads = list(pack(manifest, max_model_len=2000, output_tokens=400, source_tokens=700))
    assert len(payloads) > 3
    assert_coverage(manifest, payloads)
    assert "最终位置标记" in payloads[-1]["source"]
    assert all(not payload["related_output"] for payload in payloads)
    for payload in payloads:
        assert_budget(payload, 2000, 400)


def test_full_request_overflow_splits_same_unit_and_reselects_by_character_overlap():
    manifest = manifest_from_text("左右幅分段证据" * 12)
    size = len(manifest["units"][0]["text"])
    items = [output(f"病害明细.{i}", [source(manifest, start=i * 12, end=(i + 1) * 12)],
                    value={"病害描述": str(i) + "原文已有详细描述" * 30})
             for i in range(size // 12)]
    payloads = list(pack(manifest, items, max_model_len=2200, output_tokens=400))
    assert len(payloads) > 1
    assert_coverage(manifest, payloads)
    for payload in payloads:
        expected = {item["path"] for item in items
                    if any(ref["start"] < span["end"] and span["start"] < ref["end"]
                           for ref in item["sources"] for span in payload["source_spans"])}
        assert {item["path"] for item in payload["related_output"]} == expected
        assert len(expected) < len(items)
        assert_budget(payload, 2200, 400)


def test_cross_section_facts_sources_questions_and_image_observations_are_lossless():
    manifest = manifest_from_text("上次检测：左幅2号梁裂缝。\n" + "分隔章节。" * 200 + "\n本次该裂缝变化见上次记录。")
    previous, current = source(manifest, 0), source(manifest, 2)
    image = {"image_id": "I001", "observations": [{"text": "疑似裂缝", "uncertain": True}],
             "verification": "unverified"}
    facts = [{"id": "old", "value": "上次裂缝", "source_refs": [previous]},
             {"id": "comparison", "subject_id": "old", "object_id": "current", "status": "uncertain"},
             {"id": "question", "status": "resolved", "reason": "同一左幅构件，按跨节证据关联"}]
    item = output("病害明细.0", [current, previous, image], facts)
    payloads = list(pack(manifest, [item], source_tokens=600))
    current_payload = next(payload for payload in payloads
                           if any(span["evidence_id"] == current["evidence_id"] for span in payload["source_spans"]))
    related = current_payload["related_output"][0]
    assert [current_payload["related_facts"][key] for key in related["fact_refs"]] == facts
    assert [current_payload["related_evidence"][key] for key in related["source_refs"]] == [current, previous, image]
    assert previous["evidence_id"] not in {span["evidence_id"] for span in current_payload["source_spans"]}
    assert_coverage(manifest, payloads)


def test_exact_dedup_does_not_merge_equal_text_at_distinct_source_occurrences():
    manifest = manifest_from_text("裂缝\n裂缝\n")
    first, second = source(manifest, 0), source(manifest, 1)
    fact = {"id": "same", "value": "裂缝", "source_refs": [first, second]}
    items = [output("病害明细.0", [first, second, first], [fact, fact]),
             output("病害明细.1", [deepcopy(first), deepcopy(second)], [deepcopy(fact)])]
    payload = list(pack(manifest, items))[0]
    assert len(payload["related_facts"]) == 1
    assert len(payload["related_evidence"]) == 2
    assert list(payload["related_evidence"].values()) == [first, second]
    assert payload["related_output"][0]["fact_refs"] == payload["related_output"][1]["fact_refs"]
    assert payload["related_output"][0]["source_refs"] == payload["related_output"][1]["source_refs"]


def test_single_character_shared_context_overflow_is_explicit_not_partial():
    manifest = manifest_from_text("桥")
    items = [output(f"病害明细.{i}", [source(manifest)], value={"病害描述": "细节" * 100}) for i in range(10)]
    with pytest.raises(ReviewPayloadBudgetError, match="complete related outputs/evidence") as caught:
        list(pack(manifest, items, max_model_len=1800, output_tokens=400))
    details = caught.value.details
    assert details["source_spans"] == [{"evidence_id": "E000001", "start": 0, "end": 1}]
    assert details["related_paths"] == [item["path"] for item in items]
    assert details["prompt_tokens"] + details["output_tokens"] > details["max_model_len"]


def test_echo_protocol_budget_splits_many_short_units_without_claiming_empty_units():
    manifest = manifest_from_text("甲\n" * 20)
    manifest["units"].append({**deepcopy(manifest["units"][-1]), "id": "EMPTY", "text": ""})
    payloads = list(pack(manifest, output_tokens=300))
    assert len(payloads) > 1
    assert_coverage(manifest, payloads)
    for payload in payloads:
        assert_budget(payload, 65536, 300)
        assert all(span["evidence_id"] != "EMPTY" for span in payload["source_spans"])


@pytest.mark.parametrize("start,end", [(0, 0), (False, 1), (0, 100), (-1, 1)])
def test_invalid_text_reference_is_rejected_instead_of_silently_dropping_the_output(start, end):
    manifest = manifest_from_text("原文")
    ref = {**source(manifest), "start": start, "end": end}
    with pytest.raises(ValueError, match="invalid primary character interval"):
        list(pack(manifest, [output("病害明细.0", [ref])]))


def test_caller_input_and_different_yields_have_no_mutable_aliases():
    manifest = manifest_from_text("第一项原文\n第二项原文\n")
    items = [output("病害明细.0", [source(manifest, 0), source(manifest, 1)], [{"id": "F1", "value": ["观察"]}])]
    background = {"桥梁名称": "合成桥"}
    original = deepcopy((manifest, items, background))
    payloads = list(pack(manifest, items, background, source_tokens=1))
    assert len(payloads) == 2
    payloads[0]["related_facts"]["F000001"]["value"].append("不应回写")
    payloads[0]["related_evidence"]["S000001"]["text"] = "改写"
    payloads[0]["document_background"]["桥梁名称"] = "改写"
    assert (manifest, items, background) == original
    assert payloads[1]["related_facts"]["F000001"]["value"] == ["观察"]
    assert payloads[1]["related_evidence"]["S000001"]["text"] != "改写"


def test_source_only_target_cannot_veto_a_request_that_fits_the_real_chat_budget():
    manifest = manifest_from_text("原文元信息比单字目标长。")
    payloads = list(pack(manifest, source_tokens=1))
    assert len(payloads) == 1
    assert_coverage(manifest, payloads)
    assert_budget(payloads[0], 65536, 600)


def test_even_one_source_span_must_have_room_for_a_complete_protocol_reply():
    manifest = manifest_from_text("桥")
    with pytest.raises(ReviewPayloadBudgetError) as caught:
        list(pack(manifest, output_tokens=20))
    assert caught.value.details["minimum_protocol_output_tokens"] > 20
    assert caught.value.details["prompt_tokens"] + 20 < 65536


def compact_success_reply(identifier="ra_" + "0" * 32):
    return {"passed": True, "checked_assignment_id": identifier,
            "checked_all_assigned": True, "unsupported_claims": [],
            "missing_items": [], "conflicts": [], "unresolved_items": []}


def test_compact_ack_avoids_echo_splits_without_dropping_source_ranges():
    # The input fits as one assignment. Only legacy's repeated span output
    # exceeds the output allowance, causing unnecessary additional batches.
    manifest = manifest_from_text("甲\n" * 20 + "\n" + "尾部证据")
    legacy = list(pack(manifest, output_tokens=300))
    packed = list(pack(manifest, output_tokens=300, acknowledgement_mode="compact"))
    assert len(legacy) > 1
    assert len(packed) == 1
    assert_coverage(manifest, packed)
    expected = [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}
                for unit in manifest["units"] if unit["text"]]
    assert packed[0]["source_spans"] == expected
    assert "尾部证据" in packed[0]["source"]
    assert count_tokens(compact(compact_success_reply())) <= 300
    # No placeholder acknowledgement becomes actual source/assignment data.
    assert "checked_assignment_id" not in packed[0]
    assert "assignment_id" not in packed[0]


def test_compact_mode_changes_only_output_admission_not_input_payload():
    manifest = manifest_from_text("上次左幅有裂缝。\n本次变化需对照。")
    refs = [source(manifest, 0), source(manifest, 1)]
    facts = [{"id": "f1", "value": "按上下文判断", "source_refs": deepcopy(refs)}]
    items = [output("病害明细.0", refs, facts)]
    background = {"桥梁名称": "合成桥", "上一次定检结果": "上次左幅有裂缝"}
    before = deepcopy((manifest, items, background))
    legacy = list(pack(manifest, items, background, output_tokens=2000))
    captured = []

    def counted(messages):
        captured.append(deepcopy(messages))
        return count_chat_tokens(messages)

    packed = list(pack(manifest, items, background, output_tokens=2000,
                       acknowledgement_mode="compact", count_chat_tokens=counted))
    assert packed == legacy
    assert (manifest, items, background) == before
    assert json.loads(captured[-1][-1]["content"]) == packed[0]
    related = packed[0]["related_output"][0]
    assert [packed[0]["related_evidence"][key] for key in related["source_refs"]] == refs
    assert [packed[0]["related_facts"][key] for key in related["fact_refs"]] == facts
    assert packed[0]["document_background"] == background


def test_compact_ack_still_splits_full_prompt_and_preserves_all_characters():
    manifest = manifest_from_text("长段落原文。" * 300)
    payloads = list(pack(manifest, max_model_len=1400, output_tokens=300,
                         source_tokens=10000, acknowledgement_mode="compact"))
    assert len(payloads) > 1
    assert_coverage(manifest, payloads)
    for payload in payloads:
        messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": compact(payload)}]
        assert count_chat_tokens(messages) + 300 <= 1400


def test_compact_ack_enforces_complete_reply_at_exact_output_boundary():
    manifest = manifest_from_text("桥")
    required = count_tokens(compact(compact_success_reply()))
    with pytest.raises(ReviewPayloadBudgetError) as caught:
        list(pack(manifest, output_tokens=required - 1, acknowledgement_mode="compact",
                  assignment_id=lambda _: "ra_" + "0" * 32))
    assert caught.value.details["minimum_protocol_output_tokens"] == required
    packed = list(pack(manifest, output_tokens=required, acknowledgement_mode="compact",
                       assignment_id=lambda _: "ra_" + "0" * 32))
    assert len(packed) == 1
    assert_coverage(manifest, packed)


def test_explicit_legacy_ack_is_identical_to_default():
    manifest = manifest_from_text("甲\n" * 20)
    assert list(pack(manifest, output_tokens=300)) == list(
        pack(manifest, output_tokens=300, acknowledgement_mode="legacy"))


@pytest.mark.parametrize("mode", [None, "", "echo", "COMPACT", 1, True, [], {}])
def test_unknown_ack_mode_rejected_before_counting_or_packing(mode):
    def unexpected_count(*_):
        pytest.fail("invalid mode must fail before tokenizer work")

    with pytest.raises(ValueError, match="acknowledgement_mode"):
        list(pack(manifest_from_text("原文"), acknowledgement_mode=mode,
                  count_tokens=unexpected_count, count_chat_tokens=unexpected_count))


def test_compact_without_assignment_callback_reserves_explicit_extra_headroom():
    required = count_tokens(compact(compact_success_reply())) + 35
    with pytest.raises(ReviewPayloadBudgetError) as caught:
        list(pack(manifest_from_text("桥"), output_tokens=required - 1,
                  acknowledgement_mode="compact"))
    assert caught.value.details["minimum_protocol_output_tokens"] == required
    assert len(list(pack(manifest_from_text("桥"), output_tokens=required,
                         acknowledgement_mode="compact"))) == 1


def test_actual_assignment_id_token_cost_controls_output_admission():
    manifest = manifest_from_text("桥")
    numeric, alphabetic = "ra_" + "0" * 32, "ra_" + "a" * 32

    def variable_token_count(text):
        # Synthetic nonuniform token cost: equal-width IDs need not cost the same.
        return len(text) + (32 if alphabetic in text else 0)

    required = variable_token_count(compact(compact_success_reply(alphabetic)))
    numeric_cost = variable_token_count(compact(compact_success_reply(numeric)))
    assert numeric_cost < required
    assert len(list(pack(manifest, output_tokens=numeric_cost, acknowledgement_mode="compact",
                         assignment_id=lambda _: numeric, count_tokens=variable_token_count))) == 1
    with pytest.raises(ReviewPayloadBudgetError) as caught:
        list(pack(manifest, output_tokens=numeric_cost, acknowledgement_mode="compact",
                  assignment_id=lambda _: alphabetic, count_tokens=variable_token_count))
    assert caught.value.details["minimum_protocol_output_tokens"] == required
    assert len(list(pack(manifest, output_tokens=required, acknowledgement_mode="compact",
                         assignment_id=lambda _: alphabetic, count_tokens=variable_token_count))) == 1


def test_assignment_callback_sees_complete_payload_without_mutating_source_delivery():
    manifest = manifest_from_text("原文不能被预算回调改写。")
    expected = list(pack(manifest, acknowledgement_mode="compact"))[0]
    observed = []

    def identifier(payload):
        observed.append(deepcopy(payload))
        payload["source"] = "不得影响交付"
        payload["source_spans"].clear()
        return "ra_" + "d5" * 16

    actual = list(pack(manifest, acknowledgement_mode="compact", assignment_id=identifier))
    assert observed == [expected]
    assert actual == [expected]


@pytest.mark.parametrize("identifier", [None, 1, "", "ra_" + "0" * 31,
                                          "ra_" + "0" * 33, "ra_" + "A" * 32,
                                          "ra_" + "g" * 32, "ra_" + "0" * 32 + "\n"])
def test_invalid_actual_assignment_id_rejected(identifier):
    with pytest.raises(ValueError, match="assignment_id must return"):
        list(pack(manifest_from_text("原文"), acknowledgement_mode="compact",
                  assignment_id=lambda _: identifier))


def test_noncallable_assignment_factory_rejected_only_for_compact():
    with pytest.raises(ValueError, match="assignment_id must be a callable"):
        list(pack(manifest_from_text("原文"), acknowledgement_mode="compact", assignment_id="ra_"))
    assert list(pack(manifest_from_text("原文"), assignment_id="unused")) == list(
        pack(manifest_from_text("原文")))
