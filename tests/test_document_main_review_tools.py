from copy import deepcopy
import json
import pytest
from test_document_candidate_review import case, row_for
from bridge_evidence.contest.document_main_review_tools import MainReviewTools, SCHEMAS
from bridge_evidence.contest.document_handoff_bundle import build_handoff_bundle, validate_handoff_bundle
from bridge_evidence.contest import document_candidate_review as engine
from bridge_evidence.contest.run_state import json_digest


def tool(case, name):
    return MainReviewTools(case.workspace, case.handoff, case.summary, name)


def n(case, key):
    return tool(case, 'complete_review').numbers[case.ids[key]]


def test_split_main_operations_keep_evidence_and_atomic_batches(case):
    before = case.workspace.snapshot()
    assert tool(case, 'select_basic_info').call(selections={'桥梁名称': n(case, 'left_name'), '报告日期': n(case, 'left_date')})['ok']
    merger = tool(case, 'merge_duplicate_entries')
    assert merger.call(groups=[{'keep': n(case, 'left_rust'), 'others': [n(case, 'right_rust')], 'reason': '同一病害'}])['ok']
    state = case.workspace.snapshot()
    assert len(state['draft']['病害明细']) == 2
    refs = next(v for k,v in state['draft_bindings'].items() if k.startswith('病害明细.') and case.ids['left_rust'] in v)
    assert set(refs) == {case.ids['left_rust'], case.ids['right_rust']}
    assert tool(case, 'revise_defects').call(items=[{'candidate': n(case, 'left_rust'), '是否新增': '未知'}])['ok']
    assert tool(case, 'revise_recommendations').call(items=[{'candidate': n(case, 'left_paint'), '建议内容': '除锈后涂漆'}])['ok']
    assert tool(case, 'revise_basic_info').call(items=[{'candidate': n(case, 'left_date'), 'value': '2018'}])['ok']
    saved = case.workspace.snapshot()
    bad = tool(case, 'revise_defects').call(items=[{'candidate': n(case, 'left_rust'), '是否新增': '是'}, {'candidate': 99999, '是否新增': '否'}])
    assert not bad['ok'] and bad['error']['details']['path'] == 'items[1].candidate'
    assert case.workspace.snapshot() == saved
    assert saved['facts'] == before['facts']
    assert tool(case, 'complete_review').call(basis='已审核附带依据', notes='历史未明确')['ok']
    assert all(q['status'] != 'resolved' for q in case.workspace.snapshot()['questions'].values())


def test_mixed_operations_and_wrong_sections_rejected_without_saving(case):
    before = case.workspace.snapshot()
    invalid = [
        ('select_basic_info', {'selections': {'报告日期': n(case, 'left_name')}}),
        ('revise_defects', {'items': [{'candidate': n(case, 'left_paint'), '病害描述': '无'}]}),
        ('exclude_entries', {'items': [{'candidate': n(case, 'left_rust'), 'reason': '重复'}, {'candidate': n(case, 'left_paint'), 'reason': '重复'}]}),
        ('complete_review', {'basis': '审核', 'patches': []}),
        ('revise_defects', {'items': [{'candidate': n(case, 'left_rust'), '建议内容': '不属于病害'}]}),
        ('select_basic_info', {'selections': {'桥梁名称': True}}),
    ]
    for name, payload in invalid:
        result = tool(case, name).call(**payload)
        assert not result['ok'], (name, result)
        assert result['error']['details']['applied'] is False
        assert 'operations' not in result['error']['details']['example']
        assert case.workspace.snapshot() == before


def test_excluded_candidates_cannot_be_accidentally_restored(case):
    assert tool(case, 'exclude_entries').call(items=[{'candidate': n(case, 'left_rust'), 'reason': '重复'}])['ok']
    result = tool(case, 'revise_defects').call(items=[{'candidate': n(case, 'left_rust'), '病害描述': '恢复'}])
    assert not result['ok'] and result['error']['code'] == 'CANDIDATE_EXCLUDED'


def test_actual_invocation_replay_and_original_payload_preserved(case, monkeypatch):
    t = tool(case, 'select_basic_info')
    payload = {'selections': {'桥梁名称': n(case, 'left_name')}}
    invocation = {'tool_name': t.name, 'run_id': 'r', 'session_id': 's', 'generation_call_id': 'g', 'native_tool_call_id': 'c', 'args_sha256': json_digest(payload)}
    monkeypatch.setattr(engine, '_current_invocation', lambda: invocation)
    first = t.call(**payload)
    assert first['ok']
    state = case.workspace.snapshot()
    again = t.call(**payload)
    assert again['ok'] and again['replayed']
    assert case.workspace.snapshot() == state
    journal = json.loads(t.path.read_text())
    assert next(iter(journal['calls'].values()))['payload'] == payload
    assert not t.call(selections={'桥梁名称': n(case, 'right_name')})['ok']


def test_map_identity_and_original_evidence_projection(case):
    t = tool(case, 'complete_review')
    original = build_handoff_bundle(case.manifest, case.handoff, case.summary, case.workspace.snapshot())
    spans = validate_handoff_bundle(case.manifest, original)
    wire = t.project_bundle(original)
    assert wire['source_map'] == list(original['source_map'].values()) and spans
    assert all(type(r['candidate']) is int and 'fact_id' not in r for r in wire['candidates'])
    projected = wire['candidates'] + [r for group in wire['basic_candidates_by_field'].values() for r in group]
    assert {r['candidate']: r['value'] for r in projected} == {t.numbers[r['fact_id']]: r['value'] for r in original['candidates']}
    assert original['candidates'][0]['fact_id']
    with pytest.raises(ValueError, match='another document'):
        MainReviewTools(case.workspace, {}, [], 'complete_review')


def test_complete_review_receipt_invalidates_only_on_actual_change(case):
    from bridge_evidence.contest.document_speed_workflow import SpeedMergeWorkflow
    wf = object.__new__(SpeedMergeWorkflow)
    wf.workspace, wf.handoff_evidence, wf.child_summary = case.workspace, case.handoff, case.summary
    wf.directory = case.workspace.directory
    # This unit covers review invalidation; real Word/wire binding has separate tests.
    wf._prepare_review_word = lambda payload, result: result
    handlers = wf._additional_reading_handlers()
    assert handlers['select_basic_info'](selections={'桥梁名称': n(case, 'left_name')})['ok']
    assert handlers['complete_review'](basis='已核对', notes='未补历史')['ok']
    assert wf._quick_current()
    assert handlers['select_basic_info'](selections={'桥梁名称': n(case, 'left_name')})['ok']
    assert wf._quick_current()
    assert handlers['revise_basic_info'](items=[{'candidate': n(case, 'left_name'), 'value': '桥梁甲（原名）'}])['ok']
    assert not wf._quick_current()


def test_statistics_follow_model_categories_and_current_rows(case):
    before=deepcopy(case.workspace.snapshot()['facts'])
    reviewer=tool(case,'revise_recommendations')
    r=reviewer.call(items=[{'candidate':n(case,'left_paint'),'建议类别':'尽快维修'}])
    assert r['ok'] and r['advice_statistics']['counts']['尽快维修']==1
    state=case.workspace.snapshot()
    assert '尽快维修1条' in state['draft']['基本信息']['建议统计']
    assert state['draft_bindings']['基本信息.建议统计']==[case.ids['left_paint']]
    assert state['facts']==before
    assert reviewer.call(items=[{'candidate':n(case,'left_paint'),'建议类别':None}])['ok']
    assert '未分类/其他1条' in case.workspace.snapshot()['draft']['基本信息']['建议']
    assert tool(case,'exclude_entries').call(items=[{'candidate':n(case,'left_paint'),'reason':'本用例排除建议'}])['ok']
    assert '建议统计' not in case.workspace.snapshot()['draft']['基本信息']
    assert case.workspace.snapshot()['facts']==before


def test_review_does_not_automatically_merge_similar_or_different_locations(case):
    before=deepcopy(case.workspace.snapshot()['draft']['病害明细'])
    assert tool(case,'complete_review').call(basis='模型判断当前需要保留独立条目',notes='历史状态未知')['ok']
    assert case.workspace.snapshot()['draft']['病害明细']==before


def test_next_action_uses_committed_changes_not_failed_attempts(case):
    reviewer=tool(case,'complete_review')
    assert '当前先' in reviewer.next_action()
    assert tool(case,'select_basic_info').call(selections={'桥梁名称':n(case,'left_name')})['ok']
    assert '当前只处理' in reviewer.next_action()
    assert not tool(case,'revise_defects').call(items=[{'candidate':9999,'病害类型':'轻微'}])['ok']
    assert '当前只处理' in reviewer.next_action()
    r=tool(case,'revise_defects').call(items=[{'candidate':n(case,'left_rust'),'病害类型':'锈蚀'}])
    assert r['ok'] and r['draft_counts']=={k:len(v)for k,v in case.workspace.snapshot()['draft'].items()}
    assert '接着处理' in reviewer.next_action()
    assert tool(case,'revise_recommendations').call(items=[{'candidate':n(case,'left_paint'),'建议类别':'尽快维修'}])['ok']
    assert '均已有修改成功回执' in reviewer.next_action()


@pytest.mark.parametrize('mode', ['current', 'no_pages', 'missing_page_in_request', 'changed_draft', 'wrong_text', 'omitted_notes'])
def test_flat_finish_still_binds_actual_model_call_and_current_file(tmp_path, mode):
    from test_managed_file_self_check import Harness
    from bridge_evidence.contest.document_speed_workflow import SpeedMergeWorkflow
    h = Harness(tmp_path)
    h.wf.__class__ = SpeedMergeWorkflow
    h.wf.direct_delivery = True
    h.wf.child_summary = []
    h.wf.quick_review_receipt = {'semantic_state_sha256': h.wf._semantic_identity(h.wf.workspace.snapshot())}
    if mode != 'no_pages':
        h.read_all()
    if mode == 'changed_draft':
        h.revise()
        h.wf.quick_review_receipt = {'semantic_state_sha256': h.wf._semantic_identity(h.wf.workspace.snapshot())}
    args = {'basis': '核对三表完成', **({} if mode == 'omitted_notes' else {'notes': ''})}
    h.call('finish_report', args, messages=[] if mode == 'missing_page_in_request' else None)
    if mode in ('missing_page_in_request', 'wrong_text'):
        with pytest.raises(ValueError):
            h.wf._finish_report('伪造的正文' if mode == 'wrong_text' else args['basis'])
    else:
        result = h.wf._finish_report(args['basis'])
        assert result['ok'] == (mode in ('current', 'omitted_notes'))
        if result['ok']:
            assert h.wf.model_self_check['model_statement']['basis'] == args['basis']
            assert h.wf.model_self_check['host_semantic_verdict'] is None
            assert h.wf.model_self_check['verified_file_binding']['file_sha256'] == result['file_sha256']
