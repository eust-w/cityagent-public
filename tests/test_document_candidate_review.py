"""Local model decisions retain exact child evidence and atomic workspace saves."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from bridge_evidence.contest import document_candidate_review as review_module
from bridge_evidence.contest.document_candidate_review import CandidateReviewTools, review_candidates_schema
from bridge_evidence.contest.document_entries import EntryStore
from bridge_evidence.contest.document_merge_progress import build_handoff_evidence, child_fact_id
from bridge_evidence.contest.document_multiagent import import_candidates, coverage_question_id
from bridge_evidence.contest.document_workspace import DocumentWorkspace
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import json_digest


@pytest.fixture
def case(tmp_path):
    manifest = manifest_from_text('桥梁甲\n桥梁乙\n左幅支座锈蚀\n同一左幅支座轻微锈蚀\n右幅裂缝\n除锈涂漆\n2018年\n')
    units = manifest['units']

    def entry(identifier, section, unit, **values):
        return {'id': identifier, 'section': section, 'status': 'explicit',
            'source_refs': [{'evidence_id': units[unit]['id']}], **values}

    specs = {
        'left': [entry('name', '基本信息', 0, field='桥梁名称', value='桥梁甲'),
            entry('date', '基本信息', 6, field='报告日期', value='2018年'),
            entry('rust', '病害明细', 2, row={'序号': 1, '病害部位': '左幅支座', '病害描述': '锈蚀'}),
            entry('crack', '病害明细', 4, row={'序号': 2, '病害部位': '右幅梁体', '病害描述': '裂缝'}),
            entry('paint', '维修建议', 5, row={'建议内容': '除锈涂漆', '病害部位': '左幅支座'})],
        'right': [entry('name', '基本信息', 1, field='桥梁名称', value='桥梁乙'),
            entry('rust', '病害明细', 3, row={'序号': 3, '病害部位': '左幅支座', '病害描述': '轻微锈蚀'})],
    }
    results, tasks = [], []
    for task, entries in specs.items():
        store = EntryStore(manifest, task, tmp_path / task)
        for unit in units:
            store.record_read(unit['id'], 0, len(unit['text']))
        store.save_entries(entries)
        results.append({'task_id': task, 'status': 'complete', 'entries': store.snapshot(),
            'handoff': {'unresolved': ['历史状态未确定'] if task == 'left' else []}})
        tasks.append({'id': task, 'primary_evidence_ids': [u['id'] for u in units], 'headings': []})
    workspace = DocumentWorkspace(manifest, tmp_path / 'main', {'model': 'synthetic'})
    import_candidates(workspace, results, tasks)
    handoff = build_handoff_evidence(manifest, results)
    summary = [{'task_id': task['id'], 'status': 'complete'} for task in tasks]
    ids = {task + '_' + entry['id']: child_fact_id(task, entry['id'])
        for task, entries in specs.items() for entry in entries}
    return SimpleNamespace(workspace=workspace, handoff=handoff, summary=summary, ids=ids,
        tools=CandidateReviewTools(workspace, handoff, summary), manifest=manifest,
        issue='Q_' + json_digest(['left', 0, '历史状态未确定'])[:24],
        basic_issue='Q_basic_' + json_digest('桥梁名称')[:24])


def run(case, operations, task_reviews=None, issue_reviews=None):
    return case.tools.review_candidates(operations, task_reviews or [], issue_reviews)


def update(workspace, **changes):
    return workspace.apply_update({'base_version': workspace.snapshot()['version'], **changes})


def row_for(state, candidate_id, section='病害明细'):
    return next(row for i, row in enumerate(state['draft'][section])
        if candidate_id in state['draft_bindings'][f'{section}.{i}'])


def native(monkeypatch, payload, call='call-1'):
    identity = {'tool_name': 'review_candidates', 'run_id': 'run', 'session_id': 'session',
        'generation_call_id': 'generation', 'native_tool_call_id': call,
        'args_sha256': json_digest(payload)}
    monkeypatch.setattr(review_module, '_current_invocation', lambda: identity)
    return identity


def test_select_is_explicit_and_does_not_auto_resolve_conflict_or_coverage(case):
    before = case.workspace.snapshot()
    assert '桥梁名称' not in before['draft']['基本信息']
    response = run(case, [{'op': 'select', 'candidate_id': case.ids['right_name']}])
    state = case.workspace.snapshot()
    assert response['ok'] and response['changed'] and not response['semantic_approval']
    assert state['draft']['基本信息']['桥梁名称'] == '桥梁乙'
    assert state['draft_bindings']['基本信息.桥梁名称'] == [case.ids['right_name']]
    assert state['facts'] == before['facts']
    assert state['questions'] == before['questions']
    assert state['questions'][case.basic_issue]['status'] == 'open'
    assert state['questions'][coverage_question_id('right')]['status'] == 'open'


def test_merge_keeps_requested_later_row_and_all_original_evidence(case):
    before = case.workspace.snapshot()
    operation = {'op': 'merge', 'keep_id': case.ids['right_rust'], 'merge_ids': [case.ids['left_rust']],
        'reason': '两段记载同一左幅支座，合并保留轻微程度。', 'changes': {'病害描述': '轻微锈蚀，历史状态未知'}}
    response = run(case, [operation])
    state = case.workspace.snapshot()
    assert response['ok']
    assert state['draft']['病害明细'] == [row_for(before, case.ids['left_crack']),
        {**row_for(before, case.ids['right_rust']), '病害描述': '轻微锈蚀，历史状态未知'}]
    assert state['draft_bindings']['病害明细.0'] == [case.ids['left_crack']]
    assert state['draft_bindings']['病害明细.1'] == [case.ids['right_rust'], case.ids['left_rust']]
    assert state['facts'] == before['facts']
    assert state['draft']['维修建议'] == before['draft']['维修建议']
    refs = [r for cid in state['draft_bindings']['病害明细.1'] for r in state['facts'][cid]['source_refs']]
    assert refs == case.handoff[case.ids['right_rust']]['source_refs'] + case.handoff[case.ids['left_rust']]['source_refs']
    version = state['version']
    assert run(case, [operation])['changed'] is False
    assert case.workspace.snapshot()['version'] == version


def test_patch_changes_only_named_fields_and_preserves_original_candidates(case):
    before = case.workspace.snapshot()
    response = run(case, [
        {'op': 'patch', 'candidate_id': case.ids['left_date'], 'changes': {'value': '2018'}},
        {'op': 'patch', 'candidate_id': case.ids['left_rust'], 'changes': {'是否新增': '未知'}}])
    state = case.workspace.snapshot()
    assert response['ok'] and state['version'] == before['version'] + 1
    assert state['draft']['基本信息']['报告日期'] == '2018'
    assert row_for(state, case.ids['left_rust']) == {**row_for(before, case.ids['left_rust']), '是否新增': '未知'}
    assert state['draft_bindings'] == before['draft_bindings']
    assert state['facts'] == before['facts']


def test_drop_replay_never_deletes_next_row_and_records_explicit_reason(case):
    before = case.workspace.snapshot()
    operation = {'op': 'drop', 'candidate_id': case.ids['left_rust'], 'reason': '该项为模型判定的重复候选'}
    assert run(case, [operation])['ok']
    state = case.workspace.snapshot()
    assert state['draft']['病害明细'] == [row_for(before, case.ids['left_crack']), row_for(before, case.ids['right_rust'])]
    assert state['draft_bindings']['病害明细.0'] == [case.ids['left_crack']]
    assert state['facts'] == before['facts']
    again = run(case, [operation])
    assert again['ok'] and not again['changed']
    assert case.workspace.snapshot() == state
    journal = json.loads(case.tools.path.read_text())
    assert all(row['payload']['operations'][0]['reason'] == operation['reason'] for row in journal['calls'].values())


def test_review_updates_only_explicit_questions_in_same_atomic_save(case):
    before = case.workspace.snapshot()
    response = run(case, [{'op': 'select', 'candidate_id': case.ids['left_name']}],
        [{'task_id': 'left', 'reason': '已读该分工原文；基本字段、病害和建议候选均已覆盖。'}],
        [{'question_id': case.basic_issue, 'status': 'resolved', 'reason': '按原文桥梁全名选择甲。'},
         {'question_id': case.issue, 'status': 'unknown', 'reason': '历史对比没有足够依据，保持未知。'}])
    state = case.workspace.snapshot()
    assert response['ok'] and state['version'] == before['version'] + 1
    assert state['questions'][coverage_question_id('left')]['status'] == 'resolved'
    assert state['questions'][coverage_question_id('right')]['status'] == 'open'
    assert state['questions'][case.basic_issue]['status'] == 'resolved'
    assert state['questions'][case.issue]['status'] == 'unknown'
    assert state['facts'] == before['facts']
    assert not response['semantic_approval'] and response['file_inspection_still_required']


@pytest.mark.parametrize('bad,expected', [
    ('unknown', 'CANDIDATE_UNKNOWN'), ('cross_section', 'REVIEW_SECTION'),
    ('repeat_id', 'REVIEW_OVERLAP'), ('same_basic', 'REVIEW_OVERLAP'),
    ('wrong_column', 'REVIEW_FIELDS'), ('boolean', 'REVIEW_ARGUMENTS'),
    ('unknown_task', 'TASK_UNKNOWN'), ('unknown_question', 'QUESTION_UNKNOWN'),
    ('blank_reason', 'REVIEW_REASON'), ('coverage_as_issue', 'REVIEW_COVERAGE'),
    ('double_question', 'REVIEW_OVERLAP'), ('duplicate_merge_ids', 'REVIEW_ARGUMENTS')])
def test_invalid_batch_rolls_back_even_after_valid_first_decision(case, bad, expected):
    operations = [{'op': 'patch', 'candidate_id': case.ids['left_date'], 'changes': {'value': '2018'}}]
    tasks, issues = [], []
    if bad == 'unknown':
        operations.append({'op': 'select', 'candidate_id': 'foreign-id'})
    elif bad == 'cross_section':
        operations.append({'op': 'merge', 'keep_id': case.ids['left_rust'],
            'merge_ids': [case.ids['left_paint']], 'reason': '模型决定'})
    elif bad == 'repeat_id':
        operations.append(deepcopy(operations[0]))
    elif bad == 'same_basic':
        operations += [{'op': 'select', 'candidate_id': case.ids[k]} for k in ('left_name', 'right_name')]
    elif bad == 'wrong_column':
        operations.append({'op': 'patch', 'candidate_id': case.ids['left_rust'], 'changes': {'建议内容': '错列'}})
    elif bad == 'boolean':
        operations[0]['changes']['value'] = True
    elif bad == 'unknown_task':
        tasks = [{'task_id': 'foreign', 'reason': '已读'}]
    elif bad == 'unknown_question':
        issues = [{'question_id': 'foreign', 'status': 'resolved', 'reason': '已读'}]
    elif bad == 'blank_reason':
        tasks = [{'task_id': 'left', 'reason': '  '}]
    elif bad == 'coverage_as_issue':
        issues = [{'question_id': coverage_question_id('left'), 'status': 'resolved', 'reason': '已读'}]
    elif bad == 'double_question':
        issues = [{'question_id': case.issue, 'status': 'unknown', 'reason': '不足'}] * 2
    else:
        operations.append({'op': 'merge', 'keep_id': case.ids['left_rust'],
            'merge_ids': [case.ids['right_rust']] * 2, 'reason': '模型决定'})
    before = case.workspace.checkpoint_path.read_bytes()
    response = run(case, operations, tasks, issues)
    assert not response['ok'] and response['error']['code'] == expected
    assert response['error']['details']['applied'] is False
    assert case.workspace.checkpoint_path.read_bytes() == before


def test_merged_member_cannot_implicitly_edit_or_drop_retained_row(case):
    run(case, [{'op': 'merge', 'keep_id': case.ids['left_rust'], 'merge_ids': [case.ids['right_rust']],
        'reason': '显式合并'}])
    before = case.workspace.checkpoint_path.read_bytes()
    for op in ({'op': 'drop', 'candidate_id': case.ids['right_rust'], 'reason': '删除'},
               {'op': 'patch', 'candidate_id': case.ids['right_rust'], 'changes': {'病害描述': '修改'}}):
        response = run(case, [op])
        assert response['error']['code'] == 'CANDIDATE_MERGED_MEMBER'
        assert case.workspace.checkpoint_path.read_bytes() == before


def test_same_fact_bound_twice_is_ambiguous_not_a_row_index_guess(case):
    row = deepcopy(case.workspace.snapshot()['draft']['病害明细'][0])
    update(case.workspace, draft_patch=[{'op': 'append_row', 'section': '病害明细',
        'row': row, 'fact_ids': [case.ids['left_rust']]}])
    before = case.workspace.checkpoint_path.read_bytes()
    response = run(case, [{'op': 'drop', 'candidate_id': case.ids['left_rust'], 'reason': '删除'}])
    assert response['error']['code'] == 'CANDIDATE_AMBIGUOUS'
    assert case.workspace.checkpoint_path.read_bytes() == before


def test_changed_candidate_cannot_be_overwritten_from_stale_handoff(case):
    update(case.workspace, facts=[{'id': case.ids['left_name'], 'value': '已由另一次审核修正的名称'}])
    before = case.workspace.checkpoint_path.read_bytes()
    response = run(case, [{'op': 'select', 'candidate_id': case.ids['left_name']}])
    assert response['error']['code'] == 'CANDIDATE_CHANGED'
    assert case.workspace.checkpoint_path.read_bytes() == before


def test_optimistic_conflict_preserves_concurrent_workspace_update(case, monkeypatch):
    original = case.workspace.apply_update
    captured = {}
    def concurrent(update_arg):
        original({'base_version': case.workspace.snapshot()['version'], 'working_notes': {'decision': 'concurrent'}})
        captured['bytes'] = case.workspace.checkpoint_path.read_bytes()
        return original(update_arg)
    monkeypatch.setattr(case.workspace, 'apply_update', concurrent)
    response = run(case, [{'op': 'select', 'candidate_id': case.ids['left_name']}])
    assert response['error']['code'] == 'REVIEW_WORKSPACE_REJECTED'
    assert response['error']['details']['applied'] is False
    assert case.workspace.checkpoint_path.read_bytes() == captured['bytes']


def test_trusted_native_replay_is_exact_receipt_even_after_later_changes(case, monkeypatch):
    payload = {'operations': [{'op': 'drop', 'candidate_id': case.ids['left_rust'], 'reason': '重复'}],
        'task_reviews': [], 'issue_reviews': None}
    native(monkeypatch, payload)
    first = case.tools.review_candidates(**payload)
    assert first['ok']
    update(case.workspace, working_notes={'later': 'new state'})
    before = case.workspace.checkpoint_path.read_bytes()
    restored = CandidateReviewTools(case.workspace, case.handoff, case.summary)
    second = restored.review_candidates(**payload)
    assert second == {**first, 'replayed': True}
    assert case.workspace.checkpoint_path.read_bytes() == before
    changed = deepcopy(payload); changed['operations'][0]['candidate_id'] = case.ids['left_crack']
    assert restored.review_candidates(**changed)['error']['code'] == 'REVIEW_REPLAY_CONFLICT'
    assert case.workspace.checkpoint_path.read_bytes() == before


def test_postcommit_audit_failure_recovers_by_exact_workspace_history(case, monkeypatch):
    payload = {'operations': [{'op': 'drop', 'candidate_id': case.ids['left_rust'], 'reason': '重复'}],
        'task_reviews': [], 'issue_reviews': None}
    native(monkeypatch, payload)
    original = review_module.atomic_json
    count = 0
    def fail_second(path, data):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError('simulated audit disk failure after atomic workspace commit')
        return original(path, data)
    monkeypatch.setattr(review_module, 'atomic_json', fail_second)
    response = case.tools.review_candidates(**payload)
    assert response['error']['code'] == 'REVIEW_SAVE_UNCONFIRMED'
    assert response['error']['details']['applied'] is None
    state = case.workspace.snapshot()
    assert len(state['draft']['病害明细']) == 2
    monkeypatch.setattr(review_module, 'atomic_json', original)
    restored = CandidateReviewTools(case.workspace, case.handoff, case.summary)
    replay = restored.review_candidates(**payload)
    assert replay['ok'] and replay['replayed']
    assert case.workspace.snapshot() == state


def test_empty_review_does_not_create_or_mark_complete_an_empty_draft(tmp_path):
    workspace = DocumentWorkspace(manifest_from_text('原文'), tmp_path, {'model': 'synthetic'})
    before = workspace.checkpoint_path.read_bytes()
    result = CandidateReviewTools(workspace, {}, []).review_candidates([], [])
    assert result['ok'] and not result['changed'] and not result['semantic_approval']
    assert workspace.checkpoint_path.read_bytes() == before
    assert workspace.snapshot()['draft'] is None


def test_empty_basic_without_binding_is_preserved_by_noop_review(tmp_path):
    workspace = DocumentWorkspace(manifest_from_text('原文'), tmp_path, {'model': 'synthetic'})
    update(workspace, draft={'基本信息': {'总体评分': None}, '病害明细': [], '维修建议': []},
        draft_fact_ids=[], draft_bindings={})
    before = workspace.checkpoint_path.read_bytes()
    result = CandidateReviewTools(workspace, {}, []).review_candidates([], [])
    assert result['ok'] and not result['changed']
    assert workspace.checkpoint_path.read_bytes() == before


def test_unconfirmed_journal_does_not_claim_failed_batch_was_never_saved(case, monkeypatch):
    payload = {'operations': [{'op': 'drop', 'candidate_id': case.ids['left_rust'], 'reason': '重复'}],
        'task_reviews': [], 'issue_reviews': None}
    identity = native(monkeypatch, payload)
    version = case.workspace.snapshot()['version']
    key = 'native_' + json_digest([identity['run_id'], identity['session_id'], identity['native_tool_call_id']])
    review_module.atomic_json(case.tools.path, {'schema_version': 1, 'identity': case.tools.identity,
        'calls': {key: {'phase': 'pending', 'invocation': identity, 'payload': payload,
            'base_version': version, 'update_sha256': 'unproven-update', 'response': {}}}})
    update(case.workspace, working_notes={'another': 'save'})
    before = case.workspace.checkpoint_path.read_bytes()
    result = case.tools.review_candidates(**payload)
    assert result['error']['code'] == 'REVIEW_SAVE_UNCONFIRMED'
    assert result['error']['details']['applied'] is None
    assert case.workspace.checkpoint_path.read_bytes() == before


def test_foreign_journal_inventory_rejected_without_workspace_change(case):
    assert run(case, [])['ok']
    modified = deepcopy(case.handoff)
    modified[case.ids['left_name']]['candidate_value'] = 'different inventory'
    foreign = CandidateReviewTools(case.workspace, modified, case.summary)
    before = case.workspace.checkpoint_path.read_bytes()
    response = foreign.review_candidates([], [])
    assert response['error']['code'] == 'REVIEW_IDENTITY'
    assert case.workspace.checkpoint_path.read_bytes() == before


def test_schema_rejects_model_supplied_mechanical_metadata():
    schema = Draft202012Validator(review_candidates_schema())
    minimal = {'operations': [{'op': 'select', 'candidate_id': 'existing'}], 'task_reviews': []}
    assert schema.is_valid(minimal) and schema.is_valid({**minimal, 'issue_reviews': None})
    for field in ('base_version', 'source_refs', 'row_index', 'draft_bindings'):
        assert not schema.is_valid({**minimal, field: 1})
        value = deepcopy(minimal); value['operations'][0][field] = 1
        assert not schema.is_valid(value)
