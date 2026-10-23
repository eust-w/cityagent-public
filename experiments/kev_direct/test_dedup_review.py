"""Check policy isolation and preserve the existing file-inspection contract."""
from types import SimpleNamespace
from unittest.mock import patch

from dedup_workflow import DedupReviewMixin, PROMPT
from bridge_evidence.contest.document_speed_workflow import SpeedMergeWorkflow
import pytest


class Base:
    def _speed_progress(self):
        return {'stage':'review_edits','action':'仅快速处理一批',
                'budget':{'wrap_up':True,'quality_policy':'省略次要细节'}}

    def _before_continuous_request(self, payload):
        from copy import deepcopy
        return deepcopy(payload)


class Harness(DedupReviewMixin, Base):
    settings = SimpleNamespace(report_timeout=900, max_model_calls=32)
    workspace = SimpleNamespace(snapshot=lambda: {'draft':{'基本信息':{},'病害明细':[{}]*9,'维修建议':[{}]*3}})

    def _model_calls(self):
        return [{'kind':'agent_generation','seconds':120,'generation_attempted':True}]


def test_override_removes_early_wrapup_without_using_counts_as_quality():
    h=Harness();h.deduplicate_review=True
    p=h._speed_progress()
    assert not p['budget']['wrap_up']
    assert p['budget']['model_requests_remaining']==31
    assert p['current_saved_rows']=={'基本信息':0,'病害明细':9,'维修建议':3}
    assert p['counts_are_not_quality']
    assert '已保存一批不代表该类全部完成' in p['action']
    h.deduplicate_review=False
    assert h._speed_progress()['budget']['wrap_up']


def test_actual_tool_schema_and_input_payload_are_not_changed():
    from copy import deepcopy
    from bridge_evidence.contest.tong_react_adapter import _select_generation_tools
    h=Harness();h.deduplicate_review=True
    payload={'messages':[{'role':'user','content':'核对当前候选'}],
        'max_tokens':1024,'tools':[{'type':'function','function':{'name':'complete_review',
        'description':'快速审核','parameters':{'type':'object','required':['basis']}}}]}
    original=deepcopy(payload)
    result=_select_generation_tools(payload,h._before_continuous_request)
    assert result==original
    assert payload==original


def test_dedup_changes_do_not_replace_real_word_inspection(tmp_path):
    from test_managed_file_self_check import Harness as FileHarness
    from test_compact_word import CompactHarnessWorkflow, prepare, finish
    class Reviewed(DedupReviewMixin,CompactHarnessWorkflow):
        pass
    h=FileHarness(tmp_path,long=True)
    prepare(h)
    h.wf.__class__=Reviewed
    h.wf.deduplicate_review=True
    assert not h.wf._finish_is_current()
    assert finish(h)['ok']
    assert len(h.wf.model_self_check['actual_inspection_receipt_sha256s'])==3


def test_training_split_and_weight_training_are_distinct():
    from single_review_runner import validate_training_replay
    c={'training':False,'document_sha256':'same','subagent_calls':0}
    p={'official_training_only':True,'document_sha256':'same'}
    d={'document_sha256':'same'}
    validate_training_replay(c,p,d)
    with pytest.raises(AssertionError,match='official training'):
        validate_training_replay(c,{**p,'official_training_only':False},d)
    with pytest.raises(AssertionError,match='pretrained'):
        validate_training_replay({**c,'training':True},p,d)


def test_public_deployment_requires_explicit_configuration(monkeypatch):
    from three_doc_runner import deployment_settings
    monkeypatch.delenv('CITYAGENT_KEV_EXPECTED_HOSTNAME', raising=False)
    with pytest.raises(ValueError, match='CITYAGENT_KEV_EXPECTED_HOSTNAME'):
        deployment_settings()


def test_public_deployment_keeps_host_and_input_path_guards(monkeypatch, tmp_path):
    import three_doc_runner as runner
    from test_deployment_config import configure_deployment
    settings = configure_deployment(monkeypatch, tmp_path)
    monkeypatch.setattr(runner.os, 'uname', lambda: SimpleNamespace(nodename='wrong-host'))
    with pytest.raises(ValueError, match='explicitly selected host'):
        runner.deployment_settings(require_inputs=True)
    monkeypatch.setattr(runner.os, 'uname', lambda: SimpleNamespace(nodename=settings['EXPECTED_HOSTNAME']))
    assert runner.deployment_settings(require_inputs=True)['document_ids'] == ['synthetic-a', 'synthetic-b']
    for invalid in ('../private', '/outside', 'a\\b', 'a,a', 'a,'):
        monkeypatch.setenv('CITYAGENT_KEV_DOCUMENT_IDS', invalid)
        with pytest.raises(ValueError, match='DOCUMENT_IDS'):
            runner.deployment_settings(require_inputs=True)
    monkeypatch.setenv('CITYAGENT_KEV_DOCUMENT_IDS', 'synthetic-a')
    monkeypatch.setenv('CITYAGENT_KEV_SDK_PYTHON', 'relative/python')
    with pytest.raises(ValueError, match='existing absolute path'):
        runner.deployment_settings(require_inputs=True)
