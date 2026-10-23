"""Exercise the actual native dispatcher against divide handlers, without a model."""
from copy import deepcopy

import pytest

from bridge_evidence.contest import tong_react_adapter as adapter
from bridge_evidence.contest.divide_conquer_workflow import DivideSettings, DivideWorkflow
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from test_tong_react_adapter import SchemaParameters, SchemaTool, TextAction, fake_sdk


@pytest.fixture
def workflow(tmp_path):
    manifest = manifest_from_text("桥梁\n裂缝甲\n裂缝乙\n锈蚀丙")
    manifest["tables"] = [{"id": "T1", "row_count": 3, "header_evidence_ids": []}]
    for index, unit in enumerate(manifest["units"][1:]):
        unit.update(table_id="T1", row_index=index)

    class Client:
        calls = []

        def count_tokens(self, text):
            return len(text)

    return DivideWorkflow(manifest, Client(), {}, tmp_path,
                          settings=DivideSettings(max_tool_tokens=20000))


def test_native_managed_read_tools_reach_divide_handlers_without_signature_drift(workflow, monkeypatch):
    def script(registry, settings, outputs, prompt):
        outputs.append(registry["read_outline"]())
        outputs.append(registry["read_table_rows"]("T1", max_rows=1))
        outputs.append(registry["read_table_rows"]("T1", max_rows=1))
        outputs.append(registry["search_in_document"](["裂缝", "锈蚀"], max_matches=1))
        outputs.append(registry["search_in_document"](["裂缝", "锈蚀"], max_matches=1))
        yield TextAction()

    sdk, _, _, outputs, _ = fake_sdk(script, schema_api=True)
    monkeypatch.setattr(adapter, "_REGISTRATIONS", [])
    monkeypatch.setattr(adapter, "_REGISTERED_TOOL_CLASSES", [])
    monkeypatch.setattr(adapter, "_load_tongagents", lambda: sdk)
    monkeypatch.setattr(adapter, "_load_tool_schema_api", lambda: (SchemaTool, SchemaParameters))
    result = adapter.run_react("read synthetic source", workflow._read_handlers(),
                              {"url": "http://127.0.0.1:8000/v1", "model": "synthetic"})
    assert result["status"] == "completed"
    assert all(item["ok"] for item in outputs)
    assert outputs[1]["result"]["start"] == 0 and outputs[2]["result"]["start"] == 1
    assert outputs[3]["result"]["matches"][0]["id"] != outputs[4]["result"]["matches"][0]["id"]
    assert all(row["status"] == "success" for row in result["tool_calls"])
    assert workflow.client.calls == []


def test_old_direct_ranges_and_search_remain_available(workflow):
    handlers = workflow._read_handlers()
    rows = handlers["read_table_rows"]("T1", 1, 3)
    assert [row["row_index"] for row in rows] == [1, 2]
    matches = handlers["search_in_document"]("裂缝")
    assert len(matches) == 2 and all(row["preview_only"] for row in matches)
    assert len(handlers["read_outline"]()["tables"]) == 1


def test_failed_page_does_not_advance_source_cursor_and_restart_is_exact(workflow, monkeypatch):
    handler = workflow._read_handlers()["read_table_rows"]
    original = workflow._bounded_units

    def too_large(ids):
        raise ValueError("over budget")

    monkeypatch.setattr(workflow, "_bounded_units", too_large)
    with pytest.raises(ValueError, match="over budget"):
        handler("T1", max_rows=1)
    monkeypatch.setattr(workflow, "_bounded_units", original)
    first = handler("T1", max_rows=1)
    assert first["start"] == 0
    assert handler("T1", max_rows=1)["start"] == 1
    assert handler("T1", max_rows=1, restart=True) == first
    assert handler("T1", max_rows=2)["complete"]
    assert handler("T1", max_rows=1)["units"] == []


@pytest.mark.parametrize("name,args", [
    ("read_outline", {"limit": True}), ("read_outline", {"restart": 1}),
    ("read_table_rows", {"table_id": "T1", "max_rows": True}),
    ("read_table_rows", {"table_id": "T1", "start": 2, "end": 1}),
    ("search_in_document", {"query": []}),
    ("search_in_document", {"query": [" "]}),
    ("search_in_document", {"query": ["裂缝"], "max_matches": True}),
    ("search_in_document", {"query": ["裂缝"], "scope": "section"}),
    ("search_in_document", {"query": ["裂缝"], "scope": "task"}),
])
def test_managed_parameters_fail_closed_without_mutating_evidence(workflow, name, args):
    before = deepcopy(workflow.manifest)
    with pytest.raises(ValueError):
        workflow._read_handlers()[name](**args)
    assert workflow.manifest == before
    assert workflow.client.calls == []
