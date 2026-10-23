"""Review may diagnose incomplete synthetic drafts; publication stays strict.

No real report, label, SDK server, remote model, or generated basic information.
"""
from copy import deepcopy

import pytest

from test_document_workflow import FakeModel, workflow, read, prepare, source_fact


def partial(wf, *, row=None):
    read(wf)
    prepare(wf, facts=[source_fact(wf, 'damage', 1, value='左幅1号梁裂缝')],
        draft={'基本信息': {}, '维修建议': [], '病害明细': [row or {'病害描述':'左幅1号梁裂缝'}]},
        draft_fact_ids=['damage'], draft_bindings={'病害明细.0':['damage']})


def test_partial_draft_reaches_both_actual_review_directions_without_program_filling_basic(tmp_path):
    client=FakeModel()
    wf=workflow(tmp_path,client=client)
    partial(wf)
    draft=deepcopy(wf.workspace.snapshot()['draft'])
    wf._perform_review_draft()
    assert {p['direction'] for p in client.payloads}=={'FORWARD','REVERSE'}
    assert len(client.calls)>=2 and wf.review_attempts==1
    assert wf.workspace.snapshot()['draft']==draft
    assert wf.workspace.snapshot()['draft']['基本信息']=={}
    reverse=[p for p in client.payloads if p['direction']=='REVERSE']
    actual={(s['evidence_id'],s['start'],s['end']) for p in reverse for s in p['source_spans']}
    assert actual=={(u['id'],0,len(u['text'])) for u in wf.manifest['units']}


def test_incorrect_fake_pass_does_not_allow_finish_or_full_report_delivery(tmp_path):
    client=FakeModel()  # Deliberately says passed even though basic data is absent.
    wf=workflow(tmp_path,client=client)
    partial(wf)
    wf._perform_review_draft()
    assert wf.workspace.snapshot()['review']['passed'] is True
    with pytest.raises(ValueError, match='meaningless_basic_information'):
        wf._finish()
    assert wf.last_finish is None
    assert not list(tmp_path.rglob('*.docx'))
    assert wf.workspace.snapshot()['draft']['基本信息']=={}


def test_unsupported_column_still_rejected_before_semantic_dispatch(tmp_path, monkeypatch):
    client=FakeModel()
    wf=workflow(tmp_path,client=client)
    partial(wf)
    state=wf.workspace.snapshot()
    state['draft']['病害明细'][0]['unrecognized_extra']='synthetic'
    # Simulate an invalid inherited/in-memory snapshot to test D3's own gate,
    # independent of the normal atomic update gate that rejects this earlier.
    monkeypatch.setattr(wf.workspace, 'snapshot', lambda: deepcopy(state))
    with pytest.raises(ValueError,match='unsupported fields'):
        wf._perform_review_draft()
    assert client.calls==[] and wf.review_attempts==0
    assert wf.workspace.snapshot()['review'] is None
