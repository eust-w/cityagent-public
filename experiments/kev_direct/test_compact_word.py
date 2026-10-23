"""Real file checks still require model receipt, including adaptive pages."""
from copy import deepcopy
import json
import pytest

from compact_workflow import CompactReviewMixin
from bridge_evidence.contest.document_speed_workflow import SpeedMergeWorkflow
from bridge_evidence.contest.tong_react_adapter import wrap_tool_response
from test_managed_file_self_check import Harness, offline


class CompactHarnessWorkflow(CompactReviewMixin,SpeedMergeWorkflow):
    pass


def test_unresolved_details_share_existing_lookup_allowance(tmp_path):
    from unittest.mock import patch
    h=Harness(tmp_path)
    h.wf.__class__=CompactHarnessWorkflow
    h.wf.compact_review=True
    source={'type':'function','function':{'name':'read_source'}}
    workspace={'type':'function','function':{'name':'read_workspace_page'}}
    payload={'tools':[source,workspace]}
    with patch.object(SpeedMergeWorkflow,'_before_continuous_request',return_value={'tools':[source]}):
        result=h.wf._before_continuous_request(payload)
        assert result['tools']==[source,workspace]
        assert payload['tools']==[source,workspace]
    with patch.object(SpeedMergeWorkflow,'_before_continuous_request',return_value={'tools':[]}):
        assert not h.wf._before_continuous_request(payload)['tools']
    with patch.object(SpeedMergeWorkflow,'_observe_native_tool'):
        h.wf._observe_native_tool('read_workspace_page',{}, {})
    assert h.wf._speed_lookup_attempts==1


def prepare(h):
    h.wf.__class__=CompactHarnessWorkflow
    h.wf.compact_review=True
    h.wf.child_summary=[]
    h.wf.quick_review_receipt={'semantic_state_sha256':h.wf._semantic_identity(h.wf.workspace.snapshot())}
    args={'basis':'合成审核','notes':''}
    message=h.call('complete_review',args)
    result=h.wf._prepare_review_word(args,{'ok':True})
    envelope=wrap_tool_response('complete_review',result)
    h.wf._observe_native_tool('complete_review',args,envelope)
    h.history += [message,{'role':'tool','tool_call_id':message['tool_calls'][0]['id'],'content':json.dumps(envelope,ensure_ascii=False)}]
    return result


def finish(h,messages=None):
    args={'basis':'已核对实际文件','notes':''}
    h.call('finish_report',args,messages=messages)
    return h.wf._finish_report(**args)


def test_larger_actual_pages_do_not_finish_before_model_self_check(tmp_path):
    h=Harness(tmp_path,long=True)
    result=prepare(h)
    assert len(result['actual_word_pages'])==3
    assert all(p['next_cursor'] is None for p in result['actual_word_pages'])
    assert not h.wf._finish_is_current()
    assert not h.wf.model_self_check
    assert finish(h)['ok']
    assert len(h.wf.model_self_check['actual_inspection_receipt_sha256s'])==3


def test_adaptive_pages_tampering_and_old_file_are_rejected(tmp_path):
    h=Harness(tmp_path);prepare(h)
    bad=deepcopy(h.history)
    body=json.loads(bad[-1]['content']);body['result']['actual_word_pages'][0]['text']='伪造'
    bad[-1]['content']=json.dumps(body,ensure_ascii=False)
    with pytest.raises(ValueError,match='not actually received'):finish(h,bad)
    h.revise()
    assert not finish(h)['ok']


def test_word_page_budget_can_shrink_without_claiming_full_read(tmp_path,monkeypatch):
    import test_managed_file_self_check as fixture_module
    # Expand both the immutable synthetic source and its saved value together.
    monkeypatch.setattr(fixture_module,'range',lambda n:range(n*10),raising=False)
    h=Harness(tmp_path,long=True)
    result=prepare(h)
    assert any(p['next_cursor'] is not None for p in result['actual_word_pages'])
    assert not finish(h)['ok']
    h.read_all()
    assert finish(h)['ok']
