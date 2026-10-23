from copy import deepcopy
from types import SimpleNamespace
import json

from test_document_candidate_review import case
from bridge_evidence.contest.document_speed_workflow import SpeedMergeWorkflow
from bridge_evidence.contest.document_main_review_tools import SCHEMAS
from bridge_evidence.contest.document_main_supplement import SUPPLEMENT_SCHEMAS
from bridge_evidence.contest import tong_react_adapter
from bridge_evidence.contest import document_candidate_review as engine
from bridge_evidence.contest.run_state import json_digest


def workflow(case, *, failed=False, empty=False):
    if empty:
        state = case.workspace.snapshot()
        draft = deepcopy(state['draft'])
        draft['病害明细'] = []
        draft['维修建议'] = []
        bindings = {k: v for k, v in state['draft_bindings'].items() if k.startswith('基本信息.')}
        case.workspace.apply_update(dict(base_version=state['version'], draft=draft,
            draft_bindings=bindings, draft_fact_ids=sorted({i for values in bindings.values() for i in values})))
    wf = object.__new__(SpeedMergeWorkflow)
    wf.workspace, wf.handoff_evidence = case.workspace, case.handoff
    wf.manifest = case.manifest
    wf.child_summary = deepcopy(case.summary)
    if failed:
        wf.child_summary[-1]['status'] = 'failed'
    wf.directory = case.workspace.directory
    wf._prepare_review_word = lambda payload, result: result
    return wf


def test_gap_only_triggers_for_empty_category_or_failed_child(case):
    wf = workflow(case)
    assert not wf._gap_active()
    assert 'save_defects' in tong_react_adapter._READING_TOOLS
    assert 'save_recommendations' in tong_react_adapter._READING_TOOLS
    wf.child_summary[-1]['status'] = 'failed'
    del wf.main_gap_repair
    (wf.directory / 'main-gap-repair.json').unlink()
    assert wf._gap_active()
    assert wf._gap_state()['failed_tasks'] == [wf.child_summary[-1]['task_id']]


def test_empty_tables_may_finish_with_model_explanation_no_host_facts(case):
    wf = workflow(case, empty=True)
    handlers = wf._additional_reading_handlers()
    before = case.workspace.snapshot()
    rejected = handlers['complete_review'](basis='快速审核')
    assert rejected['error']['code'] == 'GAP_CHECK_REQUIRED'
    assert case.workspace.snapshot() == before
    checks = {'病害明细': '本轮可见片段不支持确定病害，保留未知',
              '维修建议': '该片段没有明确措施，不据此推断全报告没有建议'}
    accepted = handlers['complete_review'](basis='已核对交接证据', gap_checks=checks)
    assert accepted['ok'] and accepted['model_gap_checks'] == checks
    assert not wf._gap_active() and wf._quick_current()
    assert case.workspace.snapshot()['draft']['病害明细'] == []
    assert case.workspace.snapshot()['facts'] == before['facts']
    assert 'gap_checks' not in SCHEMAS['finish_report']['properties']


def test_invalid_save_attempts_do_not_exhaust_permission_and_closed_repair_cannot_reopen(case):
    wf = workflow(case, failed=True)
    wf.source_delivery = SimpleNamespace(confirmed_ranges=lambda: case.workspace.snapshot()['reads'])
    handlers = wf._additional_reading_handlers()
    before = case.workspace.snapshot()
    # Audit every rejected attempt, without making a later correction illegal.
    for _ in range(3):
        result = handlers['save_defects'](entries=[])
        assert result['error']['code'] == 'SUPPLEMENT_ARGUMENTS'
        assert result['save_progress']['outcome'] == 'not_saved'
    assert wf._gap_state()['save_attempts']['save_defects'] == 3
    assert case.workspace.snapshot() == before
    assert handlers['complete_review'](basis='保留已有成果',
        gap_checks={'病害明细': '补存失败，保留现有条目', '维修建议': '已核对现有建议'})['ok']
    assert handlers['save_recommendations'](entries=[])['error']['code'] == 'GAP_REPAIR_CLOSED'


def test_source_budget_enforced_inside_same_tool_batch_and_persisted(case):
    wf = workflow(case, failed=True)
    calls = []
    names = ('read_source', 'read_table_rows', 'search_in_document')
    wf._tools = SimpleNamespace(automatic_handlers=lambda: {
        name: lambda **kw: calls.append(kw) or {'ok': True} for name in names})
    handlers = wf._additional_reading_handlers()
    for _ in range(3):
        assert handlers['search_in_document'](query='支座')['ok']
    assert handlers['read_source'](evidence_id='E000003')['error']['code'] == 'GAP_READ_BUDGET'
    assert len(calls) == 3
    del wf.main_gap_repair
    assert wf._gap_state()['source_attempts'] == 3


def test_gap_stage_does_not_use_candidate_only_action_before_or_after_save(case):
    wf = workflow(case, failed=True)
    wf.artifact_delivery, wf.delivery_inspections = None, []
    wf.settings = SimpleNamespace(max_model_calls=16)
    wf._model_calls = lambda: []
    review = wf._main_tools()['complete_review']

    def obsolete_action():
        raise AssertionError('Candidate-only action cannot describe active supplementation')

    review.next_action = obsolete_action
    wf.source_delivery = SimpleNamespace(confirmed_ranges=lambda: case.workspace.snapshot()['reads'])
    handlers = wf._additional_reading_handlers()
    before = wf._speed_progress()
    assert before['stage'] == 'review_edits'
    assert before['action'] == before['gap_repair']['instruction']
    saved = handlers['save_defects'](entries=[{'病害描述': '右幅裂缝',
        'source_refs': [{'evidence_id': 'E000005'}], 'status': 'explicit'}])
    assert saved['ok'] and saved['added_count'] == 1
    after = wf._speed_progress()
    assert after['action'] == after['gap_repair']['instruction']
    assert after['gap_repair']['current_rows']['病害明细'] == before['gap_repair']['current_rows']['病害明细'] + 1
    actual_rows = after['gap_repair']['current_entries']['sections']['病害明细']
    assert actual_rows['rows'][-1]['value'] == {'病害描述': '右幅裂缝'}
    assert actual_rows['rows'][-1]['source_evidence_ids'] == ['E000005']
    assert actual_rows['values_complete']
    assert not after['gap_repair']['closed']  # Existing rows are not a coverage verdict.
    wf._gap_state()['closed'] = True
    review.next_action = lambda: 'candidate-only-review'
    closed = wf._speed_progress()
    assert closed['action'] == 'candidate-only-review'
    assert 'current_entries' not in closed['gap_repair']


def test_current_saved_entries_are_readonly_and_never_candidate_numbers(case):
    before = case.workspace.snapshot()
    view = SpeedMergeWorkflow._gap_saved_entries(before)
    assert view['reference_kind'] == 'snapshot_only_not_tool_candidate'
    for section, values in view['sections'].items():
        assert values['total_rows'] == len(before['draft'][section])
        assert values['values_complete'] and values['omitted_rows'] == 0
        assert [r['value'] for r in values['rows']] == before['draft'][section]
        assert all('candidate' not in r for r in values['rows'])
    view['sections']['病害明细']['rows'][0]['value']['病害描述'] = '不应覆盖工作区'
    assert case.workspace.snapshot() == before


def test_saved_entry_view_marks_long_values_and_omitted_rows_per_section(case):
    snapshot = case.workspace.snapshot()
    for section in ('病害明细', '维修建议'):
        snapshot['draft'][section] = [{'内容': '长' * 1600} for _ in range(12)]
    view = SpeedMergeWorkflow._gap_saved_entries(snapshot)
    for values in view['sections'].values():
        assert values['total_rows'] == 12
        assert values['shown_rows'] > 0
        assert values['shown_rows'] + values['omitted_rows'] == 12
        assert values['omitted_rows'] > 0 and not values['values_complete']
        assert all(r['value_truncated'] and 'value' not in r for r in values['rows'])
        assert sum(len(r['value_excerpt']) for r in values['rows']) == 3000


def test_saved_entry_view_marks_row_limit_and_reference_omissions(case):
    snapshot = case.workspace.snapshot()
    for section in ('病害明细', '维修建议'):
        snapshot['draft'][section] = [{'内容': f'条目{i}'} for i in range(12)]
        snapshot['draft_bindings'][f'{section}.0'] = ['view_fixture_fact']
    snapshot['facts']['view_fixture_fact'] = {'source_refs': [
        {'evidence_id': f'E{i:06}'} for i in range(9)]}
    view = SpeedMergeWorkflow._gap_saved_entries(snapshot)
    for values in view['sections'].values():
        assert values['shown_rows'] == 10 and values['omitted_rows'] == 2
        assert not values['values_complete']
        assert len(values['rows'][0]['source_evidence_ids']) == 8
        assert values['rows'][0]['omitted_source_id_count'] == 1


def test_wire_selection_keeps_registered_sdk_schemas_unchanged(case):
    wf = workflow(case, failed=True)
    wf._finish_is_current = lambda: False
    wf._speed_progress = lambda: {'stage': 'review_edits', 'budget': {'wrap_up': True}}
    names = ['complete_review', 'finish_report', 'search_in_document', *SUPPLEMENT_SCHEMAS]
    payload = {'messages': [], 'tools': [{'type': 'function', 'function': {'name': n}} for n in names]}
    result = wf._before_continuous_request(payload)
    tools = {t['function']['name']: t['function'] for t in result['tools']}
    assert set(tools) == set(names) - {'finish_report'}
    assert all(t in payload['tools'] for t in result['tools'])
    assert SUPPLEMENT_SCHEMAS['save_defects']['properties']['entries']['maxItems'] == 5
    assert payload['tools'][3]['function'] == {'name': 'save_defects'}


def test_source_table_reader_advertised_only_with_real_tables(case):
    wf = workflow(case, failed=True)
    wf._finish_is_current = lambda: False
    wf._speed_progress = lambda: {'stage': 'review_edits', 'budget': {'wrap_up': True}}
    names = ['complete_review', 'read_source', 'read_table_rows', 'search_in_document']
    payload = {'messages': [], 'tools': [{'type': 'function', 'function': {'name': n}} for n in names]}
    assert 'read_table_rows' not in {t['function']['name'] for t in wf._before_continuous_request(payload)['tools']}
    wf.manifest = {**case.manifest, 'tables': [{'id': 'T000001'}]}
    assert 'read_table_rows' in {t['function']['name'] for t in wf._before_continuous_request(payload)['tools']}


def test_main_actual_delivery_required_then_saves_and_invalidates_review(case):
    wf = workflow(case, failed=True)
    wf.source_delivery = SimpleNamespace(confirmed_ranges=lambda: {})
    handlers = wf._additional_reading_handlers()
    ref = {'evidence_id': 'E000005'}
    entries = [{'病害描述': '右幅裂缝', 'source_refs': [ref], 'status': 'explicit'}]
    before = case.workspace.snapshot()
    rejected = handlers['save_defects'](entries=entries)
    assert rejected['error']['code'] == 'MAIN_SOURCE_NOT_DELIVERED'
    assert rejected['error']['details']['path'] == 'entries[0].source_refs[0]'
    assert case.workspace.snapshot() == before
    wf.source_delivery.confirmed_ranges = lambda: before['reads']
    wf.quick_review_receipt = {'semantic_state_sha256': wf._semantic_identity(before)}
    accepted = handlers['save_defects'](entries=entries)
    assert accepted['ok'] and accepted['added_count'] == 1
    assert len(case.workspace.snapshot()['draft']['病害明细']) == len(before['draft']['病害明细']) + 1
    assert not wf._quick_current()


def test_gap_entry_grants_new_source_only_after_exact_final_input_ack(tmp_path):
    from bridge_evidence.contest.document_entries import EntryStore
    from bridge_evidence.contest.document_merge_progress import build_handoff_evidence
    from bridge_evidence.contest.document_multiagent import import_candidates
    from bridge_evidence.contest.document_source_delivery import SourceDeliveryLedger
    from bridge_evidence.contest.document_workspace import DocumentWorkspace
    from bridge_evidence.contest.evidence_manifest import manifest_from_text

    manifest = manifest_from_text('桥梁甲\n支座锈蚀\n支座锈蚀应除锈涂漆\n' + '未提供的过长原文' * 2000)
    store = EntryStore(manifest, 'child', tmp_path / 'child')
    store.record_read('E000001', 0, len(manifest['units'][0]['text']))
    store.save_entries([{'id': 'name', 'section': '基本信息', 'field': '桥梁名称',
        'value': '桥梁甲', 'status': 'explicit', 'source_refs': [{'evidence_id': 'E000001'}]}])
    children = [{'task_id': 'child', 'status': 'complete', 'entries': store.snapshot(),
        'handoff': {'unresolved': []}}]
    ids = [u['id'] for u in manifest['units']]
    workspace = DocumentWorkspace(manifest, tmp_path / 'main', {'model': 'offline'})
    import_candidates(workspace, children, [{'id': 'child', 'primary_evidence_ids': ids, 'headings': []}])
    wf = object.__new__(SpeedMergeWorkflow)
    wf.workspace, wf.manifest, wf.directory = workspace, manifest, workspace.directory
    wf.handoff_evidence = build_handoff_evidence(manifest, children)
    wf.child_summary = [{'task_id': 'child', 'status': 'complete', 'primary_evidence_ids': ids}]
    wf.direct_delivery = True
    wf.source_delivery = SourceDeliveryLedger(manifest, tmp_path / 'delivery',
        run_id='offline-main', agent_id='main', on_confirmed=workspace.record_read)
    mode, prompt = wf._entry()
    assert wf.handoff_bundle['gap_material']['added_evidence_ids'] == ['E000002', 'E000003']
    projected = json.loads((wf.directory / 'initial-main-review-bundle.json').read_bytes())
    assert projected['review_order'] == wf._gap_progress()['instruction']
    assert projected['review_order'] in prompt
    wf._register_initial_source(prompt, mode, 0, False)
    assert wf.source_delivery.confirmed_ranges() == {}
    handlers = wf._additional_reading_handlers()
    defect = [{'病害描述': '支座锈蚀', 'source_refs': [{'evidence_id': 'E000002'}], 'status': 'explicit'}]
    assert handlers['save_defects'](entries=defect)['error']['code'] == 'MAIN_SOURCE_NOT_DELIVERED'
    wf.source_delivery.observe_request({'messages': [{'role': 'user', 'content': prompt}]}, {'request_id': 'actual-input'})
    assert wf.source_delivery.confirmed_ranges() == {}
    wf.source_delivery.confirm_request('actual-input', evidence={'request_id': 'actual-input', 'input_delivery_confirmed': True})
    assert set(wf.source_delivery.confirmed_ranges()) == {'E000001', 'E000002', 'E000003'}
    assert handlers['save_defects'](entries=defect)['ok']
    advice = [{'建议内容': '原文未送达，不能作为依据', 'source_refs': [{'evidence_id': 'E000004'}], 'status': 'explicit'}]
    assert handlers['save_recommendations'](entries=advice)['error']['code'] == 'MAIN_SOURCE_NOT_DELIVERED'
    advice[0].update(建议内容='支座除锈涂漆', source_refs=[{'evidence_id': 'E000003'}])
    assert handlers['save_recommendations'](entries=advice)['ok']


def test_sdk_complete_review_forwards_model_gap_checks(monkeypatch):
    received = []
    monkeypatch.setattr(tong_react_adapter, '_invoke', lambda name, payload: received.append((name, payload)))
    checks = {'病害明细': '已补存原文明示病害', '维修建议': '仍未解决'}
    tong_react_adapter._complete_review('核对完成', gap_checks=checks)
    assert received[-1] == ('complete_review', {'basis': '核对完成', 'notes': '', 'gap_checks': checks})
    tong_react_adapter._complete_review('已有候选无补漏触发')
    assert 'gap_checks' not in received[-1][1]


def test_committed_second_save_replays_receipt_without_new_budget(case, monkeypatch):
    wf = workflow(case, failed=True)
    wf.source_delivery = SimpleNamespace(confirmed_ranges=lambda: case.workspace.snapshot()['reads'])
    handlers = wf._additional_reading_handlers()
    assert not handlers['save_defects'](entries=[])['ok']
    payload = {'entries': [{'病害描述': '右幅裂缝',
        'source_refs': [{'evidence_id': 'E000005'}], 'status': 'explicit'}]}
    invocation = {'tool_name': 'save_defects', 'run_id': 'r', 'session_id': 's',
        'generation_call_id': 'g', 'native_tool_call_id': 'save2', 'args_sha256': json_digest(payload)}
    monkeypatch.setattr(engine, '_current_invocation', lambda: invocation)
    atomic = engine.atomic_json
    def fail_final_journal(path, data):
        if path.name == 'candidate-review.json' and any(r['phase'] == 'complete' for r in data['calls'].values()):
            raise OSError('synthetic post-commit journal failure')
        return atomic(path, data)
    monkeypatch.setattr(engine, 'atomic_json', fail_final_journal)
    first = handlers['save_defects'](**payload)
    assert first['error']['details']['applied'] is None
    saved = case.workspace.snapshot()
    assert wf._gap_state()['save_attempts']['save_defects'] == 2
    monkeypatch.setattr(engine, 'atomic_json', atomic)
    again = handlers['save_defects'](**payload)
    assert again['ok'] and again['replayed']
    assert case.workspace.snapshot() == saved
    assert wf._gap_state()['save_attempts']['save_defects'] == 2
