"""Simple review submissions keep the old atomic semantics and real SDK identity."""
from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator

from test_document_candidate_review import case, row_for, update
from bridge_evidence.contest import document_candidate_review as engine_module
from bridge_evidence.contest.document_candidate_review import CandidateReviewTools, review_candidates_schema
from bridge_evidence.contest.document_multiagent import coverage_question_id, MergeWorkflow
from bridge_evidence.contest.document_review_submission import ReviewSubmissionTools, submit_review_schema
from bridge_evidence.contest.run_state import json_digest


@pytest.fixture
def facade(case):
    return ReviewSubmissionTools(case.workspace, case.handoff, case.summary)


def payload(**kwargs):
    return {**dict.fromkeys(('selections', 'merges', 'patches', 'exclusions', 'reviews')), **kwargs}


def native(monkeypatch, args, *, tool='submit_review', call='call-1'):
    identity = {'tool_name': tool, 'run_id': 'run', 'session_id': 'session',
        'generation_call_id': 'generation', 'native_tool_call_id': call, 'args_sha256': json_digest(args)}
    monkeypatch.setattr(engine_module, '_current_invocation', lambda: identity)
    return identity


def test_all_arrays_optional_and_noop_does_not_change_workspace(case, facade):
    before = case.workspace.checkpoint_path.read_bytes()
    result = facade.submit_review()
    assert result['ok'] and not result['changed'] and not result['semantic_approval']
    assert result['reviews_recorded'] == 0
    assert case.workspace.checkpoint_path.read_bytes() == before
    assert facade.submit_review(**{k: [] for k in payload()})['changed'] is False


def test_explicit_mixed_batch_saves_once_and_preserves_all_original_facts(case, facade, monkeypatch):
    before = case.workspace.snapshot()
    original = case.workspace.apply_update
    updates = []
    def capture(update_arg):
        updates.append(deepcopy(update_arg))
        return original(update_arg)
    monkeypatch.setattr(case.workspace, 'apply_update', capture)
    result = facade.submit_review(
        selections=[case.ids['left_name']],
        merges=[{'keep_id': case.ids['right_rust'], 'other_ids': [case.ids['left_rust']],
            'reason': '根据相同部位及上下文合并', 'changes': {'病害描述': '支座轻微锈蚀，原报告原因另见描述'}}],
        patches=[{'candidate_id': case.ids['left_date'], 'changes': {'value': '2018'}}],
        exclusions=[{'candidate_id': case.ids['left_crack'], 'reason': '模型确认该行不应进入本次交付'}],
        reviews=[{'question_id': coverage_question_id('left'), 'status': 'resolved', 'reason': '三类候选均按本分工原文审阅'}])
    state = case.workspace.snapshot()
    assert result['ok'], result
    assert len(updates) == 1 and state['version'] == before['version'] + 1
    assert state['facts'] == before['facts']
    assert state['draft']['基本信息']['桥梁名称'] == '桥梁甲'
    assert state['draft']['基本信息']['报告日期'] == '2018'
    assert len(state['draft']['病害明细']) == 1
    assert state['draft_bindings']['病害明细.0'] == [case.ids['right_rust'], case.ids['left_rust']]
    assert state['draft']['维修建议'] == before['draft']['维修建议']
    assert state['questions'][coverage_question_id('right')] == before['questions'][coverage_question_id('right')]
    assert state['questions'][case.issue] == before['questions'][case.issue]
    assert result['reviews_recorded'] == 1 and result['task_reviews_recorded'] == 1
    # The two remaining decisions are their own atomic batch; successful edits
    # from the first batch do not need to be generated or submitted again.
    followup = facade.submit_review(reviews=[
        {'question_id': coverage_question_id('right'), 'status': 'unknown', 'reason': '历史关系仍然无法判断'},
        {'question_id': case.basic_issue, 'status': 'resolved', 'reason': '桥名选择甲'}])
    final = case.workspace.snapshot()
    assert followup['ok'], followup
    assert len(updates) == 2 and final['version'] == state['version'] + 1
    assert final['draft'] == state['draft'] and final['facts'] == before['facts']
    assert final['draft_bindings'] == state['draft_bindings']
    assert final['questions'][coverage_question_id('right')]['status'] == 'unknown'
    assert final['questions'][case.issue] == before['questions'][case.issue]
    assert followup['reviews_recorded'] == 2 and followup['task_reviews_recorded'] == 1
    assert followup['review_results'][0]['coverage_task_id'] == 'right'


def test_unknown_coverage_remains_pending_and_no_omitted_reviews_are_filled(case, facade):
    before = case.workspace.snapshot()
    result = facade.submit_review(reviews=[{'question_id': coverage_question_id('left'),
        'status': 'unknown', 'reason': '原文缺少所需的历史状态'}])
    assert result['ok']
    state = case.workspace.snapshot()
    assert state['questions'][coverage_question_id('left')]['status'] == 'unknown'
    assert state['questions'][coverage_question_id('right')] == before['questions'][coverage_question_id('right')]
    assert state['draft'] == before['draft'] and state['facts'] == before['facts']
    merge = object.__new__(MergeWorkflow)
    merge.child_summary = case.summary
    assert merge._coverage_pending(state)


@pytest.mark.parametrize('failure,path,code', [
    ('unknown_candidate', 'patches[0].candidate_id', 'CANDIDATE_UNKNOWN'),
    ('whitespace_candidate', 'exclusions[0].candidate_id', 'CANDIDATE_UNKNOWN'),
    ('empty_candidate', 'selections[1]', 'REVIEW_ARGUMENTS'),
    ('missing_keep', 'merges[0].keep_id', 'REVIEW_ARGUMENTS'),
    ('unknown_merge_member', 'merges[0].other_ids[0]', 'CANDIDATE_UNKNOWN'),
    ('cross_section', 'merges[0]', 'REVIEW_SECTION'),
    ('duplicate_candidate', 'patches[0].candidate_id', 'REVIEW_OVERLAP'),
    ('wrong_field', 'patches[0].changes', 'REVIEW_FIELDS'),
    ('unknown_question', 'reviews[0].question_id', 'QUESTION_UNKNOWN'),
    ('duplicate_question', 'reviews[1].question_id', 'REVIEW_OVERLAP'),
    ('blank_reason', 'reviews[0].reason', 'REVIEW_REASON'),
    ('wrong_status', 'reviews[0].status', 'REVIEW_ARGUMENTS')])
def test_bad_last_decision_rejects_whole_batch_with_new_schema_feedback(case, facade, failure, path, code):
    args = payload(selections=[case.ids['left_name']])
    if failure == 'unknown_candidate':
        args['patches'] = [{'candidate_id': 'foreign', 'changes': {'value': 'wrong'}}]
    elif failure == 'whitespace_candidate':
        args['exclusions'] = [{'candidate_id': '  ', 'reason': 'explicit'}]
    elif failure == 'empty_candidate':
        args['selections'].append('')
    elif failure == 'missing_keep':
        args['merges'] = [{'other_ids': [case.ids['left_rust']], 'reason': 'explicit'}]
    elif failure in ('unknown_merge_member', 'cross_section'):
        args['merges'] = [{'keep_id': case.ids['left_rust'],
            'other_ids': ['foreign' if failure == 'unknown_merge_member' else case.ids['left_paint']], 'reason': 'explicit'}]
    elif failure == 'duplicate_candidate':
        args['patches'] = [{'candidate_id': case.ids['left_name'], 'changes': {'value': '甲'}}]
    elif failure == 'wrong_field':
        args['patches'] = [{'candidate_id': case.ids['left_rust'], 'changes': {'建议内容': 'wrong section'}}]
    else:
        args['reviews'] = [{'question_id': case.issue, 'status': 'unknown', 'reason': '明确不知道'}]
        if failure == 'unknown_question': args['reviews'][0]['question_id'] = 'foreign'
        if failure == 'duplicate_question': args['reviews'] *= 2
        if failure == 'blank_reason': args['reviews'][0]['reason'] = '  '
        if failure == 'wrong_status': args['reviews'][0]['status'] = 'all_done'
    before = case.workspace.checkpoint_path.read_bytes()
    result = facade.submit_review(**args)
    assert not result['ok'], result
    error = result['error']
    assert error['code'] == code and error['details']['path'] == path
    assert error['details']['applied'] is False
    assert 'operations' not in error['details']['example'] and 'task_reviews' not in error['details']['example']
    assert Draft202012Validator(submit_review_schema()).is_valid(error['details']['example'])
    assert case.workspace.checkpoint_path.read_bytes() == before


def test_existing_but_unmapped_coverage_question_is_not_guessed(case, facade):
    unknown = coverage_question_id('foreign-task')
    update(case.workspace, questions=[{'id': unknown, 'question': '像是覆盖检查，但不是本次分工', 'status': 'open'}])
    before = case.workspace.checkpoint_path.read_bytes()
    result = facade.submit_review(reviews=[{'question_id': unknown, 'status': 'resolved', 'reason': '尝试提交'}])
    assert result['error']['code'] == 'REVIEW_COVERAGE'
    assert result['error']['details']['path'] == 'reviews[0].question_id'
    assert case.workspace.checkpoint_path.read_bytes() == before


@pytest.mark.parametrize('origin', ['legacy', 'new'])
def test_explicitly_excluded_basic_candidate_is_not_restored(case, facade, origin):
    assert facade.submit_review(selections=[case.ids['left_name']])['ok']
    if origin == 'legacy':
        result = case.tools.review_candidates([{'op': 'drop', 'candidate_id': case.ids['left_name'], 'reason': '显式排除'}], [])
    else:
        result = facade.submit_review(exclusions=[{'candidate_id': case.ids['left_name'], 'reason': '显式排除'}])
    assert result['ok']
    restored = ReviewSubmissionTools(case.workspace, case.handoff, case.summary)
    before = case.workspace.checkpoint_path.read_bytes()
    for args in ({'selections': [case.ids['left_name']]},
                 {'patches': [{'candidate_id': case.ids['left_name'], 'changes': {'value': '重建'}}]}):
        result = restored.submit_review(**args)
        assert result['error']['code'] == 'CANDIDATE_EXCLUDED'
        assert case.workspace.checkpoint_path.read_bytes() == before


def test_exclusion_repeat_is_noop_and_cannot_delete_next_row(case, facade):
    args = {'exclusions': [{'candidate_id': case.ids['left_rust'], 'reason': '明确重复'}]}
    assert facade.submit_review(**args)['ok']
    state = case.workspace.snapshot()
    result = facade.submit_review(**args)
    assert result['ok'] and not result['changed']
    assert case.workspace.snapshot() == state
    assert row_for(state, case.ids['left_crack'])['病害描述'] == '裂缝'


def test_real_native_identity_and_original_payload_are_preserved_for_replay(case, facade, monkeypatch):
    args = payload(exclusions=[{'candidate_id': case.ids['left_rust'], 'reason': '明确排除'}])
    identity = native(monkeypatch, args)
    result = facade.submit_review(**args)
    assert result['ok']
    journal = json.loads(facade.path.read_text())
    row = next(iter(journal['calls'].values()))
    assert row['tool_name'] == 'submit_review' and row['invocation'] == identity
    assert row['payload'] == args and row['submitted_args_sha256'] == json_digest(args)
    assert row['payload']['selections'] is None and 'operations' not in row['payload']
    update(case.workspace, working_notes={'new': 'later state'})
    before = case.workspace.checkpoint_path.read_bytes()
    restored = ReviewSubmissionTools(case.workspace, case.handoff, case.summary)
    assert restored.submit_review(**args) == {**result, 'replayed': True}
    changed = deepcopy(args); changed['exclusions'][0]['reason'] = 'different args'
    assert restored.submit_review(**changed)['error']['code'] == 'REVIEW_REPLAY_CONFLICT'
    assert case.workspace.checkpoint_path.read_bytes() == before


def test_cross_tool_same_native_call_id_cannot_reuse_or_overwrite_receipt(case, facade, monkeypatch):
    old_args = {'operations': [{'op': 'select', 'candidate_id': case.ids['left_name']}],
        'task_reviews': [], 'issue_reviews': None}
    native(monkeypatch, old_args, tool='review_candidates')
    assert case.tools.review_candidates(**old_args)['ok']
    new_args = payload(selections=[case.ids['right_name']])
    native(monkeypatch, new_args)
    before = case.workspace.checkpoint_path.read_bytes()
    result = facade.submit_review(**new_args)
    assert result['error']['code'] == 'REVIEW_REPLAY_CONFLICT'
    assert case.workspace.checkpoint_path.read_bytes() == before


def test_wrong_native_tool_name_is_rejected_not_rewritten(case, facade, monkeypatch):
    args = payload(selections=[case.ids['left_name']])
    native(monkeypatch, args, tool='review_candidates')
    before = case.workspace.checkpoint_path.read_bytes()
    result = facade.submit_review(**args)
    assert result['error']['code'] == 'REVIEW_INVOCATION'
    assert case.workspace.checkpoint_path.read_bytes() == before


def test_crash_after_commit_recovers_original_submission_without_redropping(case, facade, monkeypatch):
    args = payload(exclusions=[{'candidate_id': case.ids['left_rust'], 'reason': '明确排除'}])
    native(monkeypatch, args)
    original = engine_module.atomic_json
    count = 0
    def fail_second(path, data):
        nonlocal count
        count += 1
        if count == 2: raise OSError('postcommit journal failure')
        return original(path, data)
    monkeypatch.setattr(engine_module, 'atomic_json', fail_second)
    result = facade.submit_review(**args)
    assert result['error']['code'] == 'REVIEW_SAVE_UNCONFIRMED'
    assert result['error']['details']['applied'] is None
    state = case.workspace.snapshot()
    monkeypatch.setattr(engine_module, 'atomic_json', original)
    restored = ReviewSubmissionTools(case.workspace, case.handoff, case.summary)
    result = restored.submit_review(**args)
    assert result['ok'] and result['replayed']
    assert case.workspace.snapshot() == state


def test_schema_has_no_operation_union_and_legacy_contract_is_unchanged():
    schema = submit_review_schema()
    encoded = json.dumps(schema)
    assert 'oneOf' not in encoded and 'task_reviews' not in encoded and 'issue_reviews' not in encoded
    validator = Draft202012Validator(schema)
    assert validator.is_valid({}) and validator.is_valid(payload())
    assert not validator.is_valid({'operations': []})
    assert not validator.is_valid({'selections': ['x'], 'base_version': 1})
    legacy = Draft202012Validator(review_candidates_schema())
    assert not legacy.is_valid({})
    assert legacy.is_valid({'operations': [], 'task_reviews': []})
    assert not legacy.is_valid({'selections': ['x']})
