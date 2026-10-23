"""Child liveness is counted at complete generation/tool-batch boundaries."""
from copy import deepcopy
import json

import pytest

from bridge_evidence.contest import document_multiagent_worker as worker
from test_document_multiagent import FakeGateway
from test_worker_source_delivery import prepare, setup_worker


def name_entry(manifest):
    return {"id": "name", "section": "基本信息", "field": "桥梁名称", "value": "合成桥",
        "status": "explicit", "source_refs": [{"evidence_id": manifest["units"][0]["id"]}]}


def accepted(gateway, payload, request_id):
    audit = prepare(gateway, payload, request_id)
    gateway.on_input_observed(payload, audit)
    gateway.on_receipt({**audit, "status": "success"})


def generation(gateway, identifier):
    gateway.on_receipt({"status": "success", "call_id": identifier})


def test_warns_once_after_six_then_stops_after_twelve_preserving_entries(tmp_path, monkeypatch):
    manifest, request, client = setup_worker(tmp_path, monkeypatch)
    request["max_model_calls"] = 100
    warnings = []

    def react(prompt, handlers, config):
        gateway = FakeGateway.instance
        accepted(gateway, {"messages": [{"role": "user", "content": prompt}]}, "initial")
        assert handlers["save_entries"]([name_entry(manifest)])["changed"] is True
        assert config["before_native_model"]() is None
        for index in range(1, 13):
            generation(gateway, "round-" + str(index))
            # Different empty searches are not new evidence or new pagination.
            assert handlers["search_in_document"]("missing-" + str(index))["ok"]
            feedback = config["before_native_model"]()
            if feedback is not None:
                warnings.append(feedback)
                assert index == 6
                assert '"saved_entry_ids":["name"]' in feedback
                assert '"tool":"complete_section"' in feedback
                assert config["before_native_model"]() is None  # same call boundary
        pytest.fail("Twelfth unchanged round must stop before another model request")

    result = worker.run_section(request, tmp_path / "child", client=client, react_runner=react)
    assert result["error"]["type"] == "SectionProgressStalled"
    assert len(warnings) == 1
    assert result["entries"]["entry_order"] == ["name"]
    guard = result["progress_guard"]
    assert guard["guard"]["unchanged"] == 12
    assert [row["action"] for row in guard["events"]] == ["warn", "stop"]
    assert guard["blocked"]["main_agent_takeover_required"] is True
    assert guard["blocked"]["semantic_review"] is False
    assert result["handoff"] is None and result["status"] == "failed"
    assert json.loads((tmp_path / "child" / "section-progress.json").read_text()) == guard


def test_a_later_save_in_the_same_tool_batch_counts_before_guard(tmp_path, monkeypatch):
    manifest, request, client = setup_worker(tmp_path, monkeypatch)

    def react(prompt, handlers, config):
        gateway = FakeGateway.instance
        accepted(gateway, {"messages": [{"role": "user", "content": prompt}]}, "initial")
        assert config["before_native_model"]() is None
        for index in range(1, 6):
            generation(gateway, str(index))
            assert config["before_native_model"]() is None
        generation(gateway, "sixth")
        handlers["search_in_document"]("absent")  # no progress, first tool
        assert handlers["save_entries"]([name_entry(manifest)])["ok"]  # second tool
        assert config["before_native_model"]() is None
        assert handlers["complete_section"]("原文已读，条目已保存，交主Agent审核。", [])["ok"]
        # Existing successful handoff takes priority over liveness feedback.
        assert config["before_native_model"]() is None
        return {"status": "failed", "native_final_event": False}

    result = worker.run_section(request, tmp_path / "child", client=client, react_runner=react)
    assert result["error"] is None
    assert result["progress_guard"]["guard"]["unchanged"] == 0
    assert result["progress_guard"]["events"] == []
    assert result["handoff"] is not None


def test_repeated_entry_pages_and_noop_saves_do_not_reset_guard(tmp_path, monkeypatch):
    manifest, request, client = setup_worker(tmp_path, monkeypatch)

    def react(prompt, handlers, config):
        gateway = FakeGateway.instance
        accepted(gateway, {"messages": [{"role": "user", "content": prompt}]}, "initial")
        assert handlers["save_entries"]([name_entry(manifest)])["ok"]
        assert handlers["read_entries"]()["entries"]
        assert config["before_native_model"]() is None
        for index in range(1, 7):
            generation(gateway, str(index))
            assert handlers["read_entries"](restart=True)["entries"]
            assert handlers["save_entries"]([name_entry(manifest)])["changed"] is False
            feedback = config["before_native_model"]()
            assert (feedback is not None) == (index == 6)
        return {"status": "failed", "native_final_event": False}

    result = worker.run_section(request, tmp_path / "child", client=client, react_runner=react)
    assert result["error"] is None
    assert result["progress_guard"]["guard"]["unchanged"] == 6
    assert len(result["progress_guard"]["entry_page_items"]) == 1


def test_new_pending_source_page_allows_next_request_without_confirming_reads(tmp_path, monkeypatch):
    manifest, request, client = setup_worker(tmp_path, monkeypatch, embedded=False)

    def react(_prompt, handlers, config):
        gateway = FakeGateway.instance
        assert config["before_native_model"]() is None
        for index in range(1, 6):
            generation(gateway, str(index))
            assert config["before_native_model"]() is None
        generation(gateway, "sixth")
        page = handlers["read_source"](manifest["units"][0]["id"])
        assert page["ok"]
        assert config["before_native_model"]() is None  # the new page needs its next request
        assert json.loads((tmp_path / "child" / "entries" / "entries.json").read_text())["reads"] == {}
        accepted(gateway, {"messages": [{"role": "tool", "content": json.dumps(page, ensure_ascii=False)}]}, "source")
        assert config["before_native_model"]() is None
        return {"status": "failed", "native_final_event": False}

    result = worker.run_section(request, tmp_path / "child", client=client, react_runner=react)
    assert result["error"] is None
    assert result["progress_guard"]["guard"]["unchanged"] == 0
    assert result["entries"]["reads"]


def test_guard_restores_unchanged_count_without_replaying_warning(tmp_path, monkeypatch):
    _manifest, request, client = setup_worker(tmp_path, monkeypatch, embedded=False)
    directory = tmp_path / "child"

    def first(_prompt, _handlers, config):
        for index in range(1, 7):
            generation(FakeGateway.instance, str(index))
            feedback = config["before_native_model"]()
            assert (feedback is not None) == (index == 6)
        return {"status": "failed", "native_final_event": False}

    initial = worker.run_section(request, directory, client=client, react_runner=first)
    assert initial["error"] is None
    assert initial["progress_guard"]["guard"]["unchanged"] == 6

    def resumed(_prompt, _handlers, config):
        assert config["before_native_model"]() is None  # no generation yet
        generation(FakeGateway.instance, "7")
        assert config["before_native_model"]() is None
        return {"status": "failed", "native_final_event": False}

    result = worker.run_section(request, directory, client=client, react_runner=resumed)
    assert result["error"] is None
    assert result["progress_guard"]["guard"]["unchanged"] == 7
    assert [row["action"] for row in result["progress_guard"]["events"]] == ["warn"]


def test_fingerprint_ignores_versions_notes_and_replayed_outline_pages():
    state = {"entries": {"name": {"value": "合成桥"}}, "entry_order": ["name"], "version": 1}
    page = {"kind": "tool", "status": "confirmed", "page": {
        "sections": [{"id": "s1", "title": "桥梁概述"}], "tables": []}}
    delivery = {"pages": {"p1": page}}
    first = worker._section_progress_fingerprint(state, delivery, {"E1": [[0, 8]]}, {"entry-hash"})
    state.update(version=100, notes=["继续", "完成了"])
    delivery["pages"]["p2"] = deepcopy(page)
    assert worker._section_progress_fingerprint(state, delivery, {"E1": [[0, 8]]}, {"entry-hash"}) == first
    delivery["pages"]["p3"] = {"kind": "tool", "status": "pending", "page": {
        "sections": [{"id": "s2", "title": "尚未实际发送的目录"}]}}
    pending = worker._section_progress_fingerprint(state, delivery, {"E1": [[0, 8]]}, {"entry-hash"})
    assert pending != first  # a real new outline page allows the next request
    delivery["pages"]["p3"]["status"] = "confirmed"
    assert worker._section_progress_fingerprint(state, delivery, {"E1": [[0, 8]]}, {"entry-hash"}) == pending


def test_twelfth_round_with_new_pending_page_is_not_stopped_and_replay_is_bounded(tmp_path, monkeypatch):
    manifest, request, client = setup_worker(tmp_path, monkeypatch, embedded=False)
    warnings = []

    def react(_prompt, handlers, config):
        gateway = FakeGateway.instance
        for index in range(1, 12):
            generation(gateway, str(index))
            feedback = config["before_native_model"]()
            if feedback is not None:
                warnings.append(feedback)
        assert len(warnings) == 1
        generation(gateway, "12")
        original = handlers["read_source"](manifest["units"][0]["id"])
        assert original["ok"]
        assert config["before_native_model"]() is None
        disk = json.loads((tmp_path / "child" / "section-progress.json").read_text())
        assert disk["guard"]["unchanged"] == 0
        # It is still only a pending tool reply, never model-seen original input.
        assert json.loads((tmp_path / "child" / "entries" / "entries.json").read_text())["reads"] == {}
        for index in range(13, 25):
            generation(gateway, str(index))
            repeated = handlers["read_source"](manifest["units"][0]["id"])
            assert repeated["source_delivery_receipt"] == original["source_delivery_receipt"]
            feedback = config["before_native_model"]()
            if feedback is not None:
                warnings.append(feedback)
        pytest.fail("Repeated pending page must eventually stop")

    result = worker.run_section(request, tmp_path / "child", client=client, react_runner=react)
    assert result["error"]["type"] == "SectionProgressStalled"
    assert result["progress_guard"]["guard"]["unchanged"] == 12
    assert len(warnings) == 2  # exactly once per stalled period
    assert result["entries"]["reads"] == {}
    assert result["source_delivery"]["confirmed_ranges"] == {}
    assert result["progress_guard"]["blocked"]["saved_progress"]["primary"]["missing_range_count"] == 2


def test_exploration_fingerprint_unions_ranges_independently_of_page_receipts():
    state = {"entries": {}, "entry_order": []}
    def page(start, end):
        return {"kind": "tool", "status": "pending", "page": {},
                "fragments": [{"evidence_id": "E1", "start": start, "end": end}]}
    delivery = {"pages": {"random-one": page(0, 20)}}
    expected = worker._section_progress_fingerprint(state, delivery, {}, set())
    delivery["pages"].update({"random-two": page(0, 10), "random-three": page(8, 20)})
    assert worker._section_progress_fingerprint(state, delivery, {}, set()) == expected
    delivery["pages"]["new-characters"] = page(19, 21)
    assert worker._section_progress_fingerprint(state, delivery, {}, set()) != expected


@pytest.mark.parametrize("failure", ["http_400", "bad_output"])
def test_failed_generation_attempts_count_without_authorizing_tool_origin(tmp_path, monkeypatch, failure):
    _manifest, request, client = setup_worker(tmp_path, monkeypatch, embedded=False)
    warnings = []

    def react(_prompt, _handlers, config):
        gateway = FakeGateway.instance
        for index in range(1, 13):
            payload = {"messages": [{"role": "user", "content": "继续既有任务"}]}
            audit = prepare(gateway, payload, "failed-" + str(index))
            if failure == "bad_output":
                gateway.on_input_observed(payload, audit)
            audit.update(status="error", generation_attempted=True,
                http_status=400 if failure == "http_400" else 200)
            gateway.receipts.append(audit)
            gateway.on_receipt(audit)
            assert config["current_agent_call_id"]() is None
            feedback = config["before_native_model"]()
            if feedback is not None:
                warnings.append(feedback)
                assert index == 6
        pytest.fail("Failed actual attempts must not loop past the stop boundary")

    result = worker.run_section(request, tmp_path / "child", client=client, react_runner=react)
    assert result["error"]["type"] == "SectionProgressStalled"
    assert len(warnings) == 1
    assert result["progress_guard"]["guard"]["unchanged"] == 12
    assert result["progress_guard"]["guard"]["generation_id"] == "failed-12"
    assert result["progress_guard"]["blocked"]["generation_call_id"] is None
    assert result["entries"]["reads"] == {}


def test_preflight_and_tokenization_failures_do_not_count_as_generations(tmp_path, monkeypatch):
    _manifest, request, client = setup_worker(tmp_path, monkeypatch, embedded=False)

    def react(_prompt, _handlers, config):
        gateway = FakeGateway.instance
        for index in range(15):
            audit = {"request_id": "preflight-" + str(index), "status": "rejected",
                     "generation_attempted": False, "failure_stage": "dynamic_wire_allocation"}
            gateway.receipts.append(audit)
            gateway.on_receipt(audit)
            assert config["before_native_model"]() is None
        audit = {"request_id": "actual-attempt", "status": "error", "generation_attempted": True}
        gateway.receipts.append(audit)
        gateway.on_receipt(audit)
        assert config["before_native_model"]() is None
        return {"status": "failed", "native_final_event": False}

    result = worker.run_section(request, tmp_path / "child", client=client, react_runner=react)
    assert result["error"] is None
    assert result["progress_guard"]["events"] == []
    assert result["progress_guard"]["guard"]["unchanged"] == 1
    assert result["progress_guard"]["guard"]["generation_id"] == "actual-attempt"
