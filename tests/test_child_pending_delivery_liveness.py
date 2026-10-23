"""A pending tool page must reach the next request before a liveness stop."""
from copy import deepcopy

import pytest

from bridge_evidence.contest.document_entries import EntryStore
from bridge_evidence.contest.document_merge_progress import ConsecutiveProgressGuard
from bridge_evidence.contest.document_multiagent_worker import _section_progress_fingerprint
from bridge_evidence.contest.document_source_delivery import SourceDeliveryLedger
from bridge_evidence.contest.document_tools import DocumentTools
from bridge_evidence.contest.evidence_manifest import manifest_from_text


@pytest.mark.parametrize("tool", ["read_source", "read_outline"])
def test_new_child_pending_page_at_eleven_unchanged_rounds_can_reach_next_request(tmp_path, tool):
    manifest = manifest_from_text("1 本次检测\n这是刚请求的新原文，尚未进入下一轮模型请求。")
    store = EntryStore(manifest, "child", tmp_path / "entries")
    ledger = SourceDeliveryLedger(manifest, tmp_path / "source", run_id="run", agent_id="child",
                                  on_confirmed=store.record_read)
    tools = DocumentTools(manifest, count_tokens=len, max_tool_tokens=4000, delivery_ledger=ledger)
    guard = ConsecutiveProgressGuard()

    def fingerprint():
        return _section_progress_fingerprint(store.snapshot(), ledger.state(), ledger.confirmed_ranges(), set())

    guard.observe(None, fingerprint())
    for index in range(1, 12):
        guard.observe(str(index), fingerprint())
    assert guard.state["unchanged"] == 11
    handler = tools.automatic_handlers()[tool]
    page = handler("E000002") if tool == "read_source" else handler()
    assert page["ok"] is True
    assert ledger.confirmed_ranges() == {} and store.snapshot()["reads"] == {}
    # The one new pending page is navigation progress, not semantic approval.
    assert guard.observe("12", fingerprint()) is None
    assert guard.state["unchanged"] == 0
    # Pending-page retries/new receipt IDs are not another useful action.
    duplicate = deepcopy(page)
    duplicate.pop("source_delivery_receipt")
    ledger.prepare_page(duplicate, tool=tool)
    assert guard.observe("13", fingerprint()) is None
    assert guard.state["unchanged"] == 1
