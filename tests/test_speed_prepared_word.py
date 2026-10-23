"""Program-prepared Word still needs a subsequent real model file check."""
from copy import deepcopy
import json
import pytest

from test_managed_file_self_check import Harness, offline
from bridge_evidence.contest.document_speed_workflow import SpeedMergeWorkflow
from bridge_evidence.contest.tong_react_adapter import wrap_tool_response
from bridge_evidence.contest.run_state import json_digest


def prepare(h):
    h.wf.__class__ = SpeedMergeWorkflow
    h.wf.child_summary = []
    h.wf.quick_review_receipt = {'semantic_state_sha256': h.wf._semantic_identity(h.wf.workspace.snapshot())}
    args = {'basis': '合成审核', 'notes': ''}
    message = h.call('complete_review', args)
    result = h.wf._prepare_review_word(args, {'ok': True})
    envelope = wrap_tool_response('complete_review', result)
    h.wf._observe_native_tool('complete_review', args, envelope)
    h.history += [message, {'role': 'tool', 'tool_call_id': message['tool_calls'][0]['id'],
        'content': json.dumps(envelope, ensure_ascii=False)}]
    return result


def finish(h, messages=None):
    args = {'basis': '模型已检查三张实际Word表', 'notes': ''}
    h.call('finish_report', args, messages=messages)
    return h.wf._finish_report(**args)


def test_prepared_pages_finish_only_after_seen_in_next_model_request(tmp_path):
    h = Harness(tmp_path)
    result = prepare(h)
    assert len(result['actual_word_pages']) == 3
    assert h.wf._artifact_read_coverage(h.wf.workspace.snapshot())
    assert h.wf.model_self_check is None and not h.wf._finish_is_current()
    assert finish(h)['ok']
    assert h.wf._finish_is_current()
    assert len(h.wf.model_self_check['actual_inspection_receipt_sha256s']) == 3


def test_sdk_default_notes_preserves_actual_review_origin_and_word(tmp_path, monkeypatch):
    from bridge_evidence.contest import tong_react_adapter as adapter
    h = Harness(tmp_path)
    h.wf.__class__ = SpeedMergeWorkflow
    h.wf.child_summary = []
    args = {'basis': '模型核对完成', 'gap_checks': {'病害明细': '此片段未涉及', '维修建议': '此片段未涉及'}}
    h.call('complete_review', args)
    monkeypatch.setattr(adapter, '_invoke', lambda name, payload: h.wf._prepare_review_word(payload, {'ok': True}))
    result = adapter._complete_review(**args)
    assert len(result['actual_word_pages']) == 3
    with pytest.raises(ValueError, match='No unique actual'):
        h.wf._source_proposal_origin('complete_review', {**args, 'notes': '',
            'gap_checks': {'病害明细': '已补存', '维修建议': '未涉及'}})


def test_sdk_exact_object_string_conversion_preserves_review_origin(tmp_path, monkeypatch):
    from bridge_evidence.contest import tong_react_adapter as adapter
    from bridge_evidence.contest.document_main_review_tools import SCHEMAS
    h = Harness(tmp_path)
    h.wf.__class__ = SpeedMergeWorkflow
    h.wf.child_summary = []
    checks = {'病害明细': '本段未涉及', '维修建议': '本段未涉及'}
    raw = {'basis': '模型已核对', 'gap_checks': json.dumps(checks, ensure_ascii=False)}
    h.call('complete_review', raw)
    parsed, changes, issues = adapter._normalize_container_arguments(raw, SCHEMAS['complete_review'])
    assert len(changes) == 1 and not issues
    monkeypatch.setattr(adapter, '_invoke', lambda name, payload: h.wf._prepare_review_word(payload, {'ok': True}))
    assert len(adapter._complete_review(**parsed)['actual_word_pages']) == 3
    for bad in ('{"病害明细":', '{"病害明细":"a","病害明细":"b","维修建议":"c"}', '[]'):
        h.call('complete_review', {'basis': raw['basis'], 'gap_checks': bad})
        with pytest.raises(ValueError, match='No unique actual'):
            h.wf._source_proposal_origin('complete_review', {'basis': raw['basis'], 'notes': '', 'gap_checks': checks})


def test_long_finish_explanation_preserved_and_delivered_without_rewrite(tmp_path):
    from jsonschema import Draft202012Validator
    from bridge_evidence.contest.document_main_review_tools import SCHEMAS
    h = Harness(tmp_path)
    prepare(h)
    args = {'basis': ''.join(chr(0x4e00+i) for i in range(929)), 'notes': ''}
    Draft202012Validator(SCHEMAS['finish_report']).validate(args)
    h.call('finish_report', args)
    assert h.wf._finish_report(**args)['ok']
    assert h.wf._finish_is_current()
    assert h.wf.model_self_check['model_statement']['basis'] == args['basis']


def test_unseen_or_tampered_pages_cannot_prove_file_review(tmp_path):
    h = Harness(tmp_path)
    prepare(h)
    with pytest.raises(ValueError, match='not actually received'):
        finish(h, messages=[{'role': 'user', 'content': '没有文件内容'}])
    corrupted = deepcopy(h.history)
    body = json.loads(corrupted[-1]['content'])
    body['result']['actual_word_pages'] = body['result']['actual_word_pages'][:2]
    corrupted[-1]['content'] = json.dumps(body)
    with pytest.raises(ValueError, match='not actually received'):
        finish(h, messages=corrupted)
    assert finish(h)['ok']


def test_edit_invalidates_automatically_prepared_word(tmp_path):
    h = Harness(tmp_path)
    prepare(h)
    h.revise()
    assert not h.wf._artifact_read_coverage(h.wf.workspace.snapshot())
    assert not finish(h)['ok']


def test_long_file_retains_paged_continuation(tmp_path):
    h = Harness(tmp_path, long=True)
    result = prepare(h)
    assert any(p['next_cursor'] is not None for p in result['actual_word_pages'])
    assert not finish(h)['ok']
    h.read_all()
    assert finish(h)['ok']


def test_compact_page_hash_without_actual_content_is_not_evidence(tmp_path):
    h = Harness(tmp_path)
    result = prepare(h)
    assert 'inspection_proof' not in result['actual_word_pages'][0]
    assert 'receipt' not in result['actual_word_pages'][0]
    corrupted = deepcopy(h.history)
    body = json.loads(corrupted[-1]['content'])
    body['result']['actual_word_pages'][0]['text'] = '伪造或省略文件内容'
    corrupted[-1]['content'] = json.dumps(body)
    with pytest.raises(ValueError, match='not actually received'):
        finish(h, messages=corrupted)
    assert finish(h)['ok']


def test_terminal_finish_validation_uses_durable_origin_after_session_cleanup(tmp_path):
    h = Harness(tmp_path)
    prepare(h)
    assert finish(h)['ok']
    state = h.wf._terminal_delivery_state()
    assert state
    h.wf._active_agent_call_id = None  # Exactly what DocumentWorkflow.run finally does.
    assert h.wf._terminal_delivery_state() is None
    outcome = {'status': 'completed', 'termination_kind': 'document_terminal_finish',
        'session_boundary': {k: True for k in ('iteration_exited', 'tool_callbacks_drained',
            'active_session_clear', 'session_lock_released')},
        'reading_model_call_ids': [state['origin_call_id']],
        'terminal_handoff_origin_call_id': state['origin_call_id'],
        'terminal_handoff_sha256': json_digest(state),
        'terminal_handoff_proof': {k: True for k in ('all_batch_tool_calls_paired',
            'stopped_before_next_model_call', 'sdk_stop_returned')}}
    h.wf._validate_single_agent_native_delivery(outcome)
    for key in ('terminal_handoff_sha256', 'terminal_handoff_origin_call_id'):
        bad = {**outcome, key: 'foreign'}
        with pytest.raises(ValueError, match='current, fully drained'):
            h.wf._validate_single_agent_native_delivery(bad)
