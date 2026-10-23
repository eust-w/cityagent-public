"""Managed-tool recovery cannot ask the model for removed cursor arguments."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from bridge_evidence.contest import tong_react_adapter as adapter
from bridge_evidence.contest.document_workspace import DocumentWorkspace
from bridge_evidence.contest.document_workspace_tools import WorkspaceTools
from bridge_evidence.contest.evidence_manifest import manifest_from_text


@pytest.mark.parametrize('message,code', [
    ('Workspace version changed; discard collected fragments and restart the view', 'WORKSPACE_PAGE_STALE'),
    ('Fragment continuation requires expected_version; restart the item at item_start=0', 'WORKSPACE_FRAGMENT_RANGE'),
    ('item_start exceeds the current record', 'WORKSPACE_FRAGMENT_RANGE'),
    ('Unknown workspace view; use a summary counts key', 'WORKSPACE_VIEW'),
])
def test_workspace_recovery_only_requests_model_visible_fields(message, code):
    repair = adapter._tool_error(ValueError(message), 'read_workspace_page', 'handler')
    assert repair['code'] == code
    assert repair['fields'] == ['view', 'limit', 'restart']
    for removed in ('cursor', 'item_start', 'expected_version'):
        assert removed not in json.dumps(repair)
    assert 'restart' in repair['action']


def test_unknown_managed_view_can_recover_without_a_model_cursor(tmp_path):
    workspace = DocumentWorkspace(manifest_from_text('合成源'), tmp_path, {'model': 'synthetic'})
    pages = WorkspaceTools(workspace, count_tokens=len, native_response_budget=True)
    session = {'handlers': {'read_workspace_page': pages.read_page_managed}, 'tool_calls': [],
               'max_calls': 20, 'count_tokens': len, 'max_tool_tokens': 4096}
    before = workspace.snapshot()
    failed = adapter._invoke_active(session, 'read_workspace_page', {'view': 'draft', 'limit': 4, 'restart': False})
    assert failed['error']['code'] == 'WORKSPACE_VIEW'
    assert failed['error']['fields'] == ['view', 'limit', 'restart']
    assert workspace.snapshot() == before
    good = adapter._invoke_active(session, 'read_workspace_page', {'view': 'draft_defects', 'limit': 4, 'restart': False})
    assert good['ok'] and good['result']['items'] == []


@pytest.mark.parametrize('name,field', [('read_source', 'evidence_id'), ('read_section', 'section_id'),
                                      ('read_table_rows', 'table_id'), ('get_neighbors', 'evidence_id')])
def test_program_source_range_failure_does_not_reintroduce_model_offsets(name, field):
    repair = adapter._tool_error(ValueError('Invalid character interval for E1'), name, 'handler')
    assert repair['code'] == 'SOURCE_RANGE' and repair['fields'] == [field]
    assert 'restart=true' in repair['action']
    assert 'start/end' not in repair['action'] and 'read_source_range' not in repair['action']
