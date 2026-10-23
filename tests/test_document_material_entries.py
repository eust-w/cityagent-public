"""Material aliases preserve exact canonical evidence and atomic host provenance."""
from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator

from test_document_material_index import make_material_case
from bridge_evidence.contest.document_entries import EntryStore, EntryStoreError, entry_schema
from bridge_evidence.contest.document_material_index import DocumentMaterialIndex
from bridge_evidence.contest.document_material_refs import normalize_material_refs
from bridge_evidence.contest.document_merge_progress import build_handoff_evidence, child_fact_id, MergeWorkspaceTools
from bridge_evidence.contest.document_multiagent import import_candidates
from bridge_evidence.contest.document_workspace import DocumentWorkspace, WorkspaceError, WorkspaceIdentityError, WorkspaceRepairError
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.tong_react_adapter import _update_workspace_schema, _tool_error


@pytest.fixture
def material(tmp_path):
    case = make_material_case(tmp_path / 'source')
    index = DocumentMaterialIndex(case['manifest'], case['materials'], source_path=case['source_path'])
    return case, index


def make_store(material, path):
    case, index = material
    return EntryStore(case['manifest'], 'child', path, material_index=index)


def entry(ref, identifier='bearing'):
    return {'id': identifier, 'section': '病害明细', 'row': {'病害描述': '轻微锈蚀'},
            'status': 'explicit', 'source_refs': [{'evidence_id': ref}],
            'context': {'year': '2018', 'component': '支座'}}


def deliver(owner, index, ref):
    """Simulate only the confirmed source-input callback, not a tool invocation."""
    for span in index.resolve(ref, include_context=False)['spans']:
        owner.record_read(span['evidence_id'], span['start'], span['end'])


def patch(workspace, **changes):
    return workspace.apply_update({'base_version': workspace.snapshot()['version'], **changes})


def fact(ref, identifier='bearing'):
    return {'id': identifier, 'value': {'病害描述': '轻微锈蚀'}, 'status': 'explicit',
            'source_refs': [{'evidence_id': ref}]}


def test_entry_alias_exact_cell_does_not_require_reading_other_same_text_cell(material, tmp_path):
    case, index = material
    alias = index.qualify_ref(case['refs']['table'] + '/r1/c1')
    store = make_store(material, tmp_path / 'child')
    before = store.path.read_bytes()
    with pytest.raises(EntryStoreError) as failure:
        store.save_entries([entry(alias)])
    assert failure.value.code == 'SOURCE_UNREAD'
    assert store.path.read_bytes() == before
    deliver(store, index, alias)
    original = entry(alias)
    saved = store.save_entries([original])
    snapshot = store.snapshot()
    expected, origins = normalize_material_refs(original['source_refs'], case['manifest'], index)
    assert snapshot['entries']['bearing']['source_refs'] == expected
    assert snapshot['material_provenance'] == {'bearing': origins}
    assert snapshot['entries']['bearing']['row'] == original['row']
    assert original['source_refs'] == [{'evidence_id': alias}]
    assert saved['changed'] and not saved['delivery_complete']
    restored = make_store(material, tmp_path / 'child')
    assert restored.snapshot() == snapshot
    canonical = deepcopy(original); canonical['source_refs'] = expected
    assert restored.save_entries([canonical])['changed'] is False
    assert restored.save_entries([original])['changed'] is False
    assert restored.snapshot() == snapshot


@pytest.mark.parametrize('bad_kind', ['foreign', 'ambiguous', 'offsets', 'unread'])
def test_entry_mixed_batch_failure_preserves_bytes_and_precise_error(material, tmp_path, bad_kind):
    case, index = material
    store = make_store(material, tmp_path / 'child')
    good = index.qualify_ref(case['refs']['unique']); deliver(store, index, good)
    bad = entry(index.qualify_ref(case['refs']['table']+'/r1/c1'), 'bad')
    if bad_kind == 'foreign':
        bad['source_refs'][0]['evidence_id'] = 'mineru:' + '0'*64 + case['refs']['unique']
    elif bad_kind == 'ambiguous':
        bad['source_refs'][0]['evidence_id'] = index.qualify_ref(case['refs']['ambiguous'])
    elif bad_kind == 'offsets':
        bad['source_refs'][0].update(start=0, end=1)
    before = store.path.read_bytes()
    with pytest.raises(EntryStoreError) as failure:
        store.save_entries([entry(good, 'good'), bad])
    error = failure.value.as_tool_error()['error']
    assert error['details']['path'].startswith('entries[1].source_refs[0]')
    assert error['details']['applied'] is False
    assert store.path.read_bytes() == before
    if bad_kind != 'unread':
        assert 'example' in error['details']


def test_canonical_id_only_workspace_still_requires_full_original_read(tmp_path):
    manifest = manifest_from_text('完整的原文单元')
    workspace = DocumentWorkspace(manifest, tmp_path, {'model': 'synthetic'})
    evidence_id = manifest['units'][0]['id']
    workspace.record_read(evidence_id, 0, 2)
    with pytest.raises(WorkspaceError, match='not actually read'):
        patch(workspace, facts=[fact(evidence_id)])
    workspace.record_read(evidence_id, 2, len(manifest['units'][0]['text']))
    patch(workspace, facts=[fact(evidence_id)])
    assert workspace.snapshot()['facts']['bearing']['source_refs'] == [
        {'evidence_id': evidence_id, 'start': 0, 'end': len(manifest['units'][0]['text'])}]


def test_workspace_alias_and_draft_saved_together_preserving_provenance(material, tmp_path):
    case, index = material
    workspace = DocumentWorkspace(case['manifest'], tmp_path / 'main', {'model': 'synthetic'}, material_index=index)
    alias = index.qualify_ref(case['refs']['table']+'/r1/c1')
    deliver(workspace, index, alias)
    patch(workspace, facts=[fact(alias)], draft_patch=[{'op': 'append_row', 'section': '病害明细',
        'row': {'病害描述': '轻微锈蚀'}, 'fact_ids': ['bearing']}])
    snapshot = workspace.snapshot()
    expected, origins = normalize_material_refs([{'evidence_id': alias}], case['manifest'], index)
    assert snapshot['facts']['bearing']['source_refs'] == expected
    assert snapshot['material_provenance']['facts']['bearing'] == origins
    assert snapshot['draft']['病害明细'] == [{'病害描述': '轻微锈蚀'}]
    assert snapshot['draft_bindings']['病害明细.0'] == ['bearing']
    assert DocumentWorkspace(case['manifest'], workspace.directory, {'model': 'synthetic'},
        material_index=index).snapshot() == snapshot
    patch(workspace, facts=[{'id': 'bearing', 'value': {'病害描述': '模型修订'}}])
    assert workspace.snapshot()['material_provenance']['facts']['bearing'] == origins
    replacement = index.qualify_ref(case['refs']['table']+'/r1/c2')
    deliver(workspace, index, replacement)
    patch(workspace, facts=[{'id': 'bearing', 'source_refs': [{'evidence_id': replacement}]}])
    origins2 = workspace.snapshot()['material_provenance']['facts']['bearing']
    assert len(origins2) == 1 and origins2[0]['source_ref']['evidence_id'] == replacement
    patch(workspace, retract_fact_ids=['bearing'])
    assert not workspace.snapshot()['material_provenance']['facts']


def test_workspace_bad_alias_rolls_back_prior_batch_changes_with_actionable_sdk_error(material, tmp_path):
    case, index = material
    workspace = DocumentWorkspace(case['manifest'], tmp_path / 'main', {'model': 'synthetic'}, material_index=index)
    good = index.qualify_ref(case['refs']['unique']); deliver(workspace, index, good)
    before = workspace.checkpoint_path.read_bytes()
    with pytest.raises(WorkspaceRepairError) as failure:
        patch(workspace, facts=[fact(good, 'good'), fact('mineru:'+'0'*64+case['refs']['unique'], 'bad')])
    assert workspace.checkpoint_path.read_bytes() == before
    error = _tool_error(failure.value, 'update_workspace', 'execution')
    assert error['details']['path'] == 'facts[1].source_refs[0].evidence_id'
    assert error['details']['applied'] is False
    assert 'source_refs' in error['fields']
    assert 'Foreign document' in error['details']['reason']


@pytest.mark.parametrize('target', ['entry', 'workspace'])
def test_persisted_alias_tampering_is_rejected(material, tmp_path, target):
    case, index = material
    alias = index.qualify_ref(case['refs']['unique'])
    if target == 'entry':
        owner = make_store(material, tmp_path / 'child'); deliver(owner, index, alias)
        owner.save_entries([entry(alias)])
        path = owner.path
    else:
        owner = DocumentWorkspace(case['manifest'], tmp_path / 'main', {'model': 'synthetic'}, material_index=index)
        deliver(owner, index, alias); patch(owner, facts=[fact(alias)])
        path = owner.checkpoint_path
    state = json.loads(path.read_text())
    provenance = state['material_provenance'] if target == 'entry' else state['material_provenance']['facts']
    provenance['bearing'][0]['source_ref']['evidence_id'] = 'mineru:'+'0'*64+case['refs']['unique']
    path.write_text(json.dumps(state))
    with pytest.raises((EntryStoreError, WorkspaceIdentityError)):
        owner.snapshot()


def test_child_to_main_keeps_origins_and_compact_verbatim_context(material, tmp_path):
    case, index = material
    alias = index.qualify_ref(case['refs']['table']+'/r1/c1')
    paragraph = index.qualify_ref(case['refs']['unique'])
    store = make_store(material, tmp_path / 'child')
    deliver(store, index, alias); deliver(store, index, paragraph)
    value = entry(alias); value['source_refs'].append({'evidence_id': paragraph})
    store.save_entries([value])
    result = {'task_id': 'child', 'entries': store.snapshot(), 'handoff': {'unresolved': ['历史关系待判断']}}
    ledger = build_handoff_evidence(case['manifest'], [result], material_index=index)
    key = child_fact_id('child', 'bearing'); evidence = ledger[key]
    material_evidence = evidence['material_evidence']
    assert len(material_evidence) == 2
    roles = {row['role'] for row in material_evidence[0]['context']}
    assert {'same_row', 'header'} <= roles
    assert any('2018' in quote['text'] for quote in evidence['source_excerpts'])
    assert evidence['semantic_approval'] is False
    assert evidence['unresolved'] == ['历史关系待判断']
    workspace = DocumentWorkspace(case['manifest'], tmp_path / 'main', {'model': 'synthetic'}, material_index=index)
    imported = import_candidates(workspace, [result])
    state = workspace.snapshot()
    assert imported['semantic_approval'] is False
    assert state['material_provenance']['facts'][key] == store.snapshot()['material_provenance']['bearing']
    assert state['draft_bindings']['病害明细.0'] == [key]
    tools = MergeWorkspaceTools(workspace, count_tokens=len, max_tool_tokens=40000, handoff_evidence=ledger)
    assert tools.read_page('facts', 0, 1)['items'][0]['handoff_evidence'] == evidence
    for item in material_evidence:
        resolved = index.resolve(item['source_ref']['evidence_id'])
        assert all(span in state['facts'][key]['source_refs'] for span in item['canonical_refs'])
        assert resolved['ok']


def test_foreign_child_origin_rejected_before_main_read_or_fact_mutation(material, tmp_path):
    case, index = material
    alias = index.qualify_ref(case['refs']['unique'])
    store = make_store(material, tmp_path / 'child'); deliver(store, index, alias)
    store.save_entries([entry(alias)])
    result = {'task_id': 'child', 'entries': store.snapshot()}
    result['entries']['material_provenance']['bearing'][0]['identity']['document_sha256'] = '0'*64
    workspace = DocumentWorkspace(case['manifest'], tmp_path / 'main', {'model': 'synthetic'}, material_index=index)
    before = workspace.checkpoint_path.read_bytes()
    with pytest.raises(ValueError, match='provenance'):
        import_candidates(workspace, [result])
    with pytest.raises(ValueError, match='provenance'):
        build_handoff_evidence(case['manifest'], [result], material_index=index)
    assert workspace.checkpoint_path.read_bytes() == before


@pytest.mark.parametrize('ref', [
    {'evidence_id': 'E000001'}, {'evidence_id': 'mineru:'+'a'*64+'#/texts/1'},
    {'evidence_id': 'E000001', 'start': 0, 'end': 2}, {'image_id': 'I000001'}])
def test_sdk_workspace_schema_accepts_id_only_without_business_defaults(ref):
    update = {'update': {'base_version': 0, 'facts': [{**fact('unused'), 'source_refs': [ref]}]}}
    Draft202012Validator(_update_workspace_schema()).validate(update)
    if 'image_id' not in ref:
        value = entry('unused'); value['source_refs'] = [ref]
        Draft202012Validator(entry_schema()).validate(value)


@pytest.mark.parametrize('ref', [
    {'evidence_id': 'E000001', 'start': 0}, {'evidence_id': 'E000001', 'end': 2}, {}])
def test_sdk_workspace_schema_rejects_incomplete_offsets(ref):
    update = {'update': {'base_version': 0, 'facts': [{**fact('unused'), 'source_refs': [ref]}]}}
    assert list(Draft202012Validator(_update_workspace_schema()).iter_errors(update))
