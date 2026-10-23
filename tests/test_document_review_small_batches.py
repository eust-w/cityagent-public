"""Small review transactions preserve earlier work and explicit model decisions."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from test_document_candidate_review import case, row_for, update
from test_document_review_submission import native, payload
from bridge_evidence.contest import document_candidate_review as engine_module
from bridge_evidence.contest.document_delivery_artifacts import DocumentDeliveryArtifacts
from bridge_evidence.contest.document_merge_progress import next_merge_action
from bridge_evidence.contest.document_multiagent import MergeWorkflow, coverage_question_id
from bridge_evidence.contest.document_review_submission import ReviewSubmissionTools, submit_review_schema
from bridge_evidence.contest.document_workflow import DocumentWorkflow
from bridge_evidence.contest.tong_react_adapter import wrap_tool_response


COLLECTIONS = ('selections', 'merges', 'patches', 'exclusions', 'reviews')


@pytest.fixture
def facade(case):
    return ReviewSubmissionTools(case.workspace, case.handoff, case.summary)


def empty_accepted():
    return {key: [] for key in COLLECTIONS}


def expected_accepted(args):
    return {
        'selections': args.get('selections') or [],
        'merges': [{'keep_id': row['keep_id'], 'other_ids': row['other_ids']}
            for row in args.get('merges') or []],
        'patches': [row['candidate_id'] for row in args.get('patches') or []],
        'exclusions': [row['candidate_id'] for row in args.get('exclusions') or []],
        'reviews': [row['question_id'] for row in args.get('reviews') or []],
    }


def assert_saved(response, args, version, *, changed=True):
    assert response['ok'], response
    receipt = response['batch_receipt']
    assert receipt['status'] == ('saved' if changed else 'unchanged')
    assert receipt['action_count'] == sum(len(args.get(key) or []) for key in COLLECTIONS)
    assert receipt['accepted'] == expected_accepted(args)
    assert receipt['workspace_version'] == version
    assert response['semantic_approval'] is False
    assert response['file_inspection_still_required'] is True


def assert_rejected(response, count, path, version):
    assert not response['ok'], response
    assert response['error']['details']['path'] == path
    assert response['error']['details']['applied'] is False
    receipt = response['batch_receipt']
    assert receipt['status'] == 'rejected'
    assert receipt['action_count'] == count
    assert receipt['accepted'] == empty_accepted()
    assert receipt['workspace_version'] == version


def test_successive_small_batches_reject_only_bad_batch_and_retry_only_that_batch(case, facade, monkeypatch, tmp_path):
    initial = case.workspace.snapshot()
    original_apply = case.workspace.apply_update
    applied = []

    def capture(changes):
        applied.append(deepcopy(changes))
        return original_apply(changes)

    monkeypatch.setattr(case.workspace, 'apply_update', capture)
    batch_one = payload(selections=[case.ids['left_name'], case.ids['left_date']])
    native(monkeypatch, batch_one, call='batch-1')
    first = facade.submit_review(**batch_one)
    assert_saved(first, batch_one, initial['version'] + 1)
    batch_two = payload(
        merges=[{'keep_id': case.ids['right_rust'], 'other_ids': [case.ids['left_rust']],
            'reason': '两个章节记录同一支座，保留轻微程度'}],
        patches=[{'candidate_id': case.ids['left_date'], 'changes': {'value': '2018'}}],
        reviews=[{'question_id': coverage_question_id('left'), 'status': 'unknown',
            'reason': '还没有足够历史资料'}])
    native(monkeypatch, batch_two, call='batch-2')
    second = facade.submit_review(**batch_two)
    saved_state = case.workspace.snapshot()
    assert_saved(second, batch_two, initial['version'] + 2)
    assert len(applied) == 2

    # A real existing Word/JSON/status bundle must survive a later rejected batch.
    manager = DocumentDeliveryArtifacts(tmp_path / 'delivery', 'small-batch-test')
    artifact = manager.export(saved_state, incomplete=True, reason='尚未完成全部自查')
    assert artifact['export_error'] is None
    artifact_bytes = {key: Path(artifact[key]).read_bytes()
        for key in ('docx_path', 'json_path', 'status_path')}
    checkpoint = case.workspace.checkpoint_path.read_bytes()
    journal = facade.path.read_bytes()
    failed_batch = payload(patches=[
        {'candidate_id': case.ids['left_date'], 'changes': {'value': '2018年'}},
        {'candidate_id': 'not-a-real-candidate', 'changes': {'建议内容': '除锈后涂漆'}}])
    native(monkeypatch, failed_batch, call='batch-3-rejected')
    failed = facade.submit_review(**failed_batch)
    assert failed['error']['code'] == 'CANDIDATE_UNKNOWN'
    assert_rejected(failed, 2, 'patches[1].candidate_id', saved_state['version'])
    assert len(applied) == 2  # Even the first valid edit in this batch was not applied.
    assert case.workspace.checkpoint_path.read_bytes() == checkpoint
    assert facade.path.read_bytes() == journal
    for key, content in artifact_bytes.items():
        assert Path(artifact[key]).read_bytes() == content

    # Only correct and resubmit the rejected batch; do not reconstruct batches 1/2.
    corrected = deepcopy(failed_batch)
    corrected['patches'][1]['candidate_id'] = case.ids['left_paint']
    native(monkeypatch, corrected, call='batch-3-corrected')
    third = facade.submit_review(**corrected)
    final = case.workspace.snapshot()
    assert_saved(third, corrected, saved_state['version'] + 1)
    assert len(applied) == 3
    assert final['draft']['基本信息']['桥梁名称'] == '桥梁甲'
    assert final['draft']['基本信息']['报告日期'] == '2018年'
    assert row_for(final, case.ids['right_rust'])['病害描述'] == '轻微锈蚀'
    assert final['draft_bindings']['病害明细.1'] == [case.ids['right_rust'], case.ids['left_rust']]
    assert final['draft']['维修建议'][0]['建议内容'] == '除锈后涂漆'
    assert final['facts'] == initial['facts']
    assert final['questions'] == saved_state['questions']
    for key, content in artifact_bytes.items():
        assert Path(artifact[key]).read_bytes() == content
    calls = json.loads(facade.path.read_text())['calls']
    assert len(calls) == 3
    assert all(row['payload'] != failed_batch for row in calls.values())


def test_native_merge_replay_preserves_original_receipt_version_and_does_not_merge_again(case, facade, monkeypatch):
    args = payload(merges=[{'keep_id': case.ids['right_rust'],
        'other_ids': [case.ids['left_rust']], 'reason': '模型确认同一对象'}])
    native(monkeypatch, args, call='merge-once')
    first = facade.submit_review(**args)
    assert_saved(first, args, case.workspace.snapshot()['version'])
    update(case.workspace, working_notes={'later_progress': '此后已继续阅读'})
    later = case.workspace.snapshot()
    checkpoint = case.workspace.checkpoint_path.read_bytes()
    calls_before = json.loads(facade.path.read_text())['calls']
    restored = ReviewSubmissionTools(case.workspace, case.handoff, case.summary)
    replay = restored.submit_review(**args)
    assert replay['ok'] and replay['replayed']
    assert replay['batch_receipt'] == first['batch_receipt']
    assert replay['batch_receipt']['workspace_version'] < later['version']
    assert case.workspace.checkpoint_path.read_bytes() == checkpoint
    assert json.loads(facade.path.read_text())['calls'] == calls_before
    assert case.workspace.snapshot()['draft_bindings']['病害明细.1'] == [case.ids['right_rust'], case.ids['left_rust']]


def test_individually_legal_collections_cannot_exceed_combined_five_action_limit(case, facade):
    args = payload(selections=[case.ids['left_name'], case.ids['left_date']], reviews=[
        {'question_id': q, 'status': 'unknown', 'reason': '等待明确证据'}
        for q in (coverage_question_id('left'), coverage_question_id('right'), case.basic_issue, case.issue)])
    # The schema bounds each collection; the host must additionally bound the sum.
    assert Draft202012Validator(submit_review_schema()).is_valid(args)
    checkpoint = case.workspace.checkpoint_path.read_bytes()
    response = facade.submit_review(**args)
    assert response['error']['code'] == 'REVIEW_BATCH_LIMIT'
    # Schema/budget rejection happens before opening the transaction snapshot;
    # its receipt must not invent a current workspace version.
    assert_rejected(response, 6, 'reviews[3]', None)
    assert case.workspace.checkpoint_path.read_bytes() == checkpoint
    assert not facade.path.exists()


@pytest.mark.parametrize('collection', COLLECTIONS)
def test_each_collection_rejects_sixth_item_before_planning_or_writing(case, facade, collection):
    examples = {
        'selections': case.ids['left_name'],
        'merges': {'keep_id': case.ids['left_rust'], 'other_ids': [case.ids['right_rust']], 'reason': '合并'},
        'patches': {'candidate_id': case.ids['left_date'], 'changes': {'value': '2018'}},
        'exclusions': {'candidate_id': case.ids['left_crack'], 'reason': '排除'},
        'reviews': {'question_id': case.issue, 'status': 'unknown', 'reason': '未查明'},
    }
    args = payload(**{collection: [deepcopy(examples[collection]) for _ in range(6)]})
    assert submit_review_schema()['properties'][collection]['maxItems'] == 5
    assert not Draft202012Validator(submit_review_schema()).is_valid(args)
    checkpoint = case.workspace.checkpoint_path.read_bytes()
    response = facade.submit_review(**args)
    assert response['error']['code'] == 'REVIEW_BATCH_LIMIT'
    assert_rejected(response, 6, f'{collection}[5]', None)
    assert case.workspace.checkpoint_path.read_bytes() == checkpoint
    assert not facade.path.exists()


def test_a_large_explicit_merge_group_is_one_action_and_retains_every_binding(case):
    # Synthetic child candidates for the same quoted defect. Six other_ids are
    # one explicit merge group, never implicitly split into several transactions.
    base = case.workspace.snapshot()['facts'][case.ids['left_rust']]
    extra_ids = [f'child_extra_duplicate_{i}' for i in range(5)]
    facts, patches = [], []
    for cid in extra_ids:
        facts.append({'id': cid, 'value': deepcopy(base['value']), 'status': 'explicit',
            'source_refs': deepcopy(base['source_refs'])})
        patches.append({'op': 'append_row', 'section': '病害明细',
            'row': deepcopy(base['value']), 'fact_ids': [cid]})
        case.handoff[cid] = {**deepcopy(case.handoff[case.ids['left_rust']]),
            'candidate_value': deepcopy(base['value']), 'source_refs': deepcopy(base['source_refs'])}
    update(case.workspace, facts=facts, draft_patch=patches)
    facade = ReviewSubmissionTools(case.workspace, case.handoff, case.summary)
    before = case.workspace.snapshot()
    others = [case.ids['left_rust'], *extra_ids]
    args = payload(selections=[case.ids['left_name']],
        merges=[{'keep_id': case.ids['right_rust'], 'other_ids': others, 'reason': '模型明确合并同一支座的多段记载'}],
        reviews=[{'question_id': q, 'status': 'unknown', 'reason': '保持未决，不根据合并推断完成'}
            for q in (coverage_question_id('left'), coverage_question_id('right'), case.issue)])
    assert Draft202012Validator(submit_review_schema()).is_valid(args)
    response = facade.submit_review(**args)
    after = case.workspace.snapshot()
    assert_saved(response, args, before['version'] + 1)
    assert response['batch_receipt']['action_count'] == 5
    assert len(after['draft']['病害明细']) == 2
    assert after['draft_bindings']['病害明细.1'] == [case.ids['right_rust'], *others]
    assert after['facts'] == before['facts']
    assert after['draft']['维修建议'] == before['draft']['维修建议']


def test_unknown_review_receipt_is_saved_decision_not_completion_and_noop_is_unchanged(case, facade):
    before = case.workspace.snapshot()
    args = payload(reviews=[{'question_id': coverage_question_id('left'),
        'status': 'unknown', 'reason': '没有确认历史关系的证据'}])
    first = facade.submit_review(**args)
    state = case.workspace.snapshot()
    assert_saved(first, args, before['version'] + 1)
    assert state['questions'][coverage_question_id('left')]['status'] == 'unknown'
    assert state['questions'][coverage_question_id('right')] == before['questions'][coverage_question_id('right')]
    merge = object.__new__(MergeWorkflow)
    merge.child_summary = case.summary
    assert merge._coverage_pending(state)
    again = facade.submit_review(**args)
    assert_saved(again, args, state['version'], changed=False)
    assert case.workspace.snapshot() == state
    empty = payload()
    assert_saved(facade.submit_review(**empty), empty, state['version'], changed=False)


def test_postcommit_journal_failure_reports_unconfirmed_not_rejected_then_exact_replay_recovers(case, facade, monkeypatch):
    args = payload(exclusions=[{'candidate_id': case.ids['left_rust'], 'reason': '明确排除重复'}])
    native(monkeypatch, args, call='postcommit-failure')
    original = engine_module.atomic_json
    writes = 0

    def fail_second(path, value):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError('simulated audit write failure after the workspace commit')
        return original(path, value)

    before = case.workspace.snapshot()
    monkeypatch.setattr(engine_module, 'atomic_json', fail_second)
    response = facade.submit_review(**args)
    assert response['error']['code'] == 'REVIEW_SAVE_UNCONFIRMED'
    assert response['error']['details']['applied'] is None
    receipt = response['batch_receipt']
    assert receipt['status'] == 'unconfirmed'
    assert receipt['action_count'] == 1 and receipt['accepted'] == empty_accepted()
    assert receipt.get('workspace_version') is None
    committed = case.workspace.snapshot()
    assert committed['version'] == before['version'] + 1
    assert len(committed['draft']['病害明细']) == 2
    monkeypatch.setattr(engine_module, 'atomic_json', original)
    replay = ReviewSubmissionTools(case.workspace, case.handoff, case.summary).submit_review(**args)
    assert replay['replayed']
    assert_saved(replay, args, committed['version'])
    assert case.workspace.snapshot() == committed


@pytest.mark.parametrize('last_batch_fails', [False, True])
def test_main_continuation_carries_actual_wrapped_receipt_and_only_pending_review(case, facade, monkeypatch, last_batch_fails):
    solved = [case.basic_issue, case.issue, coverage_question_id('left')]
    args = payload(selections=[case.ids['left_name']], reviews=[
        {'question_id': q, 'status': 'resolved', 'reason': '已由模型明确核对本项候选和对应原文'}
        for q in solved])
    response = facade.submit_review(**args)
    assert response['ok']
    pending = coverage_question_id('right')
    if last_batch_fails:
        args = payload(reviews=[
            {'question_id': pending, 'status': 'resolved', 'reason': '准备确认此分工'},
            {'question_id': 'Q_foreign', 'status': 'resolved', 'reason': '误写的问题编号'}])
        response = facade.submit_review(**args)
        assert response['error']['code'] == 'QUESTION_UNKNOWN'
        assert response['batch_receipt']['status'] == 'rejected'
    # Use the production envelope, including its failed-handler result nesting.
    wrapped = wrap_tool_response('submit_review', response)
    assert wrapped['ok'] is (not last_batch_fails)
    merge = object.__new__(MergeWorkflow)
    merge.direct_delivery = True
    merge.workspace, merge.child_summary = case.workspace, case.summary
    monkeypatch.setattr(DocumentWorkflow, '_observe_native_tool', lambda *_: None)
    merge._observe_native_tool('submit_review', args, wrapped)
    state = case.workspace.snapshot()
    progress = next_merge_action(state, case.workspace.delivery_status(),
        merge._coverage_pending(state), None, [])
    continuation = merge._review_action(progress)
    assert continuation['last_review_batch'] == response['batch_receipt']
    assert continuation['next_action'] == {'tool': 'submit_review',
        'required_question_id': pending, 'model_must_supply_status_and_reason': True}
    assert continuation['pending_coverage'] == [pending]
    assert continuation['unresolved_question_ids'] == [pending]
    assert all(state['questions'][q]['status'] == 'resolved' for q in solved)
    assert state['questions'][pending]['status'] == 'open'
    assert state['draft']['基本信息']['桥梁名称'] == '桥梁甲'
    # The retained real receipt is a snapshot, not a shared mutable wrapper.
    wrapped['result']['batch_receipt']['accepted']['reviews'].append('mutated-after-observation')
    assert 'mutated-after-observation' not in merge._last_review_receipt['accepted']['reviews']


@pytest.mark.parametrize('previous_stage', ['resolve_candidates', 'inspect_current_file'])
def test_unconfirmed_real_receipt_blocks_new_review_or_finish_even_at_file_stage(case, facade, monkeypatch, previous_stage):
    # Match the production main workflow policy: model review is explicit in
    # questions and final file self-check, without a second source-review gate.
    case.workspace.require_semantic_review = False
    if previous_stage == 'inspect_current_file':
        response = facade.submit_review(selections=[case.ids['left_name']], reviews=[
            {'question_id': q, 'status': 'resolved', 'reason': '模型已逐项核对候选与原文'}
            for q in case.workspace.snapshot()['questions']])
        assert response['ok']
    args = payload(exclusions=[{'candidate_id': case.ids['left_rust'], 'reason': '明确排除重复'}])
    native(monkeypatch, args, call='uncertain-save-before-continuation')
    original = engine_module.atomic_json
    writes = 0

    def fail_after_commit(path, data):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError('workspace saved but final journal confirmation failed')
        return original(path, data)

    monkeypatch.setattr(engine_module, 'atomic_json', fail_after_commit)
    response = facade.submit_review(**args)
    assert response['batch_receipt']['status'] == 'unconfirmed'
    assert response['error']['details']['applied'] is None
    monkeypatch.setattr(engine_module, 'atomic_json', original)
    checkpoint = case.workspace.checkpoint_path.read_bytes()
    merge = object.__new__(MergeWorkflow)
    merge.direct_delivery = True
    merge.workspace, merge.child_summary = case.workspace, case.summary
    monkeypatch.setattr(DocumentWorkflow, '_observe_native_tool', lambda *_: None)
    merge._observe_native_tool('submit_review', args, wrap_tool_response('submit_review', response))
    state = case.workspace.snapshot()
    progress = next_merge_action(state, case.workspace.delivery_status(),
        merge._coverage_pending(state), None, [])
    assert progress['stage'] == previous_stage
    continuation = merge._review_action(progress)
    assert continuation['stage'] == 'verify_review_save'
    assert continuation['last_review_batch'] == response['batch_receipt']
    assert continuation['next_action'] == {'tool': 'read_workspace', 'arguments': {}}
    assert '不等于未保存' in continuation['instruction']
    assert '不要作为新批次重交' in continuation['instruction']
    assert '原事务日志' in continuation['instruction']
    assert case.workspace.checkpoint_path.read_bytes() == checkpoint


@pytest.mark.parametrize('last_receipt_state', ['unconfirmed', 'later_saved', 'absent_after_restart'])
def test_actual_finish_gate_blocks_pending_transaction_until_original_journal_is_confirmed(case, facade, monkeypatch, last_receipt_state):
    # Resolve all model coverage decisions first so that the transaction gate,
    # not another outstanding question, must block both real finish entrances.
    prepared = facade.submit_review(selections=[case.ids['left_name']], reviews=[
        {'question_id': q, 'status': 'resolved', 'reason': '已明确审核当前候选与对应原文'}
        for q in case.workspace.snapshot()['questions']])
    assert prepared['ok']
    pending_args = payload(exclusions=[{'candidate_id': case.ids['left_rust'], 'reason': '排除明确重复候选'}])
    native(monkeypatch, pending_args, call='pending-before-finish')
    original_write = engine_module.atomic_json
    writes = 0

    def fail_after_save(path, data):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError('postcommit journal confirmation failed')
        return original_write(path, data)

    monkeypatch.setattr(engine_module, 'atomic_json', fail_after_save)
    uncertain = facade.submit_review(**pending_args)
    assert uncertain['batch_receipt']['status'] == 'unconfirmed'
    monkeypatch.setattr(engine_module, 'atomic_json', original_write)
    assert any(row['phase'] == 'pending' for row in facade._journal()['calls'].values())
    merge = object.__new__(MergeWorkflow)
    merge.direct_delivery = True
    merge.workspace, merge.child_summary = case.workspace, case.summary
    merge.candidate_review = ReviewSubmissionTools(case.workspace, case.handoff, case.summary)
    monkeypatch.setattr(DocumentWorkflow, '_observe_native_tool', lambda *_: None)
    if last_receipt_state == 'unconfirmed':
        merge._observe_native_tool('submit_review', pending_args, wrap_tool_response('submit_review', uncertain))
    elif last_receipt_state == 'later_saved':
        later_args = payload(patches=[{'candidate_id': case.ids['left_date'], 'changes': {'value': '2018'}}])
        native(monkeypatch, later_args, call='unrelated-later-success')
        later = merge.candidate_review.submit_review(**later_args)
        assert later['batch_receipt']['status'] == 'saved'
        merge._observe_native_tool('submit_review', later_args, wrap_tool_response('submit_review', later))
        assert merge._last_review_receipt['status'] == 'saved'
    else:
        assert not hasattr(merge, '_last_review_receipt')
    assert merge._coverage_pending(case.workspace.snapshot()) == []
    parent_calls = {'finish': [], 'finish_is_current': 0}

    def parent_finish(self, self_check=None):
        parent_calls['finish'].append(deepcopy(self_check))
        return {'ok': True, 'parent_checks_executed': True}

    def parent_is_current(self):
        parent_calls['finish_is_current'] += 1
        return True

    monkeypatch.setattr(DocumentWorkflow, '_finish', parent_finish)
    monkeypatch.setattr(DocumentWorkflow, '_finish_is_current', parent_is_current)
    self_check = {'inspected_export': True, 'complete': True,
        'basis': '模型声称完成当前文件核对', 'remaining_issues': []}
    checkpoint = case.workspace.checkpoint_path.read_bytes()
    blocked = merge._finish(self_check=self_check)
    assert blocked['ok'] is False
    assert blocked['stage'] == 'verify_review_save'
    assert blocked['last_review_batch']['status'] == 'unconfirmed'
    assert blocked['next_action'] == {'tool': 'read_workspace', 'arguments': {}}
    assert merge._finish_is_current() is False
    assert parent_calls == {'finish': [], 'finish_is_current': 0}
    assert case.workspace.checkpoint_path.read_bytes() == checkpoint

    # Replaying the exact original native transaction confirms its saved history;
    # a different successful batch or an in-memory reset cannot substitute this.
    native(monkeypatch, pending_args, call='pending-before-finish')
    recovered = merge.candidate_review.submit_review(**pending_args)
    assert recovered['ok'] and recovered['replayed']
    assert recovered['batch_receipt']['status'] == 'saved'
    assert all(row['phase'] == 'complete' for row in merge.candidate_review._journal()['calls'].values())
    merge._observe_native_tool('submit_review', pending_args, wrap_tool_response('submit_review', recovered))
    assert case.workspace.checkpoint_path.read_bytes() == checkpoint
    assert merge._finish(self_check=self_check) == {'ok': True, 'parent_checks_executed': True}
    assert merge._finish_is_current() is True
    assert parent_calls == {'finish': [self_check], 'finish_is_current': 1}
