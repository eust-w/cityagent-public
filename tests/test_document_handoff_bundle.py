"""Offline compact merge handoff: no model, filesystem delivery or semantic judge."""
from copy import deepcopy
import json

import pytest

from bridge_evidence.contest.document_handoff_bundle import (
    build_handoff_bundle, serialize_handoff_bundle, validate_handoff_bundle)
from bridge_evidence.contest.document_source_delivery import SourceDeliveryLedger
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import json_digest
from test_document_material_index import make_material_case


def _case():
    manifest=manifest_from_text('2018年检测\n左幅支座轻微锈蚀\n建议除锈涂漆')
    ref={'evidence_id':'E000002','start':0,'end':len(manifest['units'][1]['text'])}
    quote={**ref,'text':manifest['units'][1]['text'],'source_path':'unused/full/xml/path'}
    evidence={}
    facts={}
    for i in range(3):
        eid=f'child_{i}'
        value='轻微锈蚀' if i!=2 else '严重锈蚀'
        evidence[eid]={'task_id':f'task-{i%2}','entry_id':f'entry-{i}',
            'section':'基本信息','field':'风险点','candidate_value':value,
            'source_refs':[ref.copy()],'source_excerpts':[deepcopy(quote)],'table_headers':[],
            'model_context':{'year':'2018','position':'左幅'},'unresolved':['是否存在同一位置的重复条目？']}
        facts[eid]={'id':eid,'value':value,'source_refs':[ref.copy()],'status':'explicit','valid':True}
    questions={'Q_basic_1':{'question':'核对基本字段“风险点”的多个候选并决定字段归属：'+json.dumps(
        [[key,row['candidate_value']] for key,row in evidence.items()],ensure_ascii=False),'status':'open','valid':True},
        'Q_free':{'question':'核查新增和加重的区别，不得据报告年份猜测。','status':'open','valid':True}}
    workspace={'identity':{'manifest_sha256':json_digest(manifest)},'version':7,'facts':facts,'questions':questions}
    children=[{'task_id':'task-0','status':'completed','entry_count':2,'headings':['检测结果'],
        'primary_evidence_ids':['E000001','E000002'],'handoff':{'unresolved':['是否存在同一位置的重复条目？']}},
        {'task_id':'task-1','status':'failed','entry_count':1,'headings':['建议'],
        'primary_evidence_ids':['E000003'],'handoff':None}]
    return manifest,evidence,children,workspace


def test_deduplicates_only_source_not_business_candidates_and_keeps_context():
    manifest,evidence,children,workspace=_case()
    bundle=build_handoff_bundle(manifest,evidence,children,workspace)
    assert len(bundle['candidates'])==3 and len(bundle['source_map'])==1
    assert [c['value'] for c in bundle['candidates']]==['轻微锈蚀','轻微锈蚀','严重锈蚀']
    assert all(c['model_context']=={'year':'2018','position':'左幅'} for c in bundle['candidates'])
    assert all(c['source_keys']==bundle['candidates'][0]['source_keys'] for c in bundle['candidates'])
    assert bundle['children'][1]['status']=='failed'
    assert bundle['children'][0]['scope']['first_source']=='E000001'
    assert len(bundle['issues'])==3  # two questions, one shared exact child issue
    assert bundle['issues']['Q_basic_1']['value']['candidate_ids']==list(evidence)
    assert '严重锈蚀' not in bundle['issues']['Q_basic_1']['value']['question']
    assert bundle['issues']['Q_free']['value']['question']==workspace['questions']['Q_free']['question']
    assert 'unused/full/xml' not in serialize_handoff_bundle(bundle)
    assert not bundle['remaining']['candidate_ids']
    assert validate_handoff_bundle(manifest,bundle)[0]['text']==manifest['units'][1]['text']


def test_no_year_or_position_is_inferred_from_original_source():
    manifest,evidence,children,workspace=_case()
    for value in evidence.values():value['model_context']={}
    bundle=build_handoff_bundle(manifest,evidence,children,workspace)
    assert all(c.get('model_context',{})=={} for c in bundle['candidates'])


def test_current_workspace_fact_overrides_stale_child_value_and_context():
    manifest,evidence,children,workspace=_case()
    workspace['facts']['child_0'].update(value='未知',status='uncertain',source_refs=[{'evidence_id':'E000003','start':0,'end':6}])
    bundle=build_handoff_bundle(manifest,evidence,children,workspace)
    candidate=bundle['candidates'][0]
    assert candidate['value']=='未知' and candidate['status']=='uncertain'
    assert candidate.get('model_context',{})=={} and candidate['context_missing']
    assert bundle['source_map'][candidate['source_keys'][0]]['text']=='建议除锈涂漆'


def test_huge_candidate_is_navigation_only_no_sentence_or_context_clipping():
    manifest,evidence,children,workspace=_case()
    huge='完整的很长业务记录。'*1000
    evidence['child_0']['candidate_value']=huge
    workspace['facts']['child_0']['value']=huge
    bundle=build_handoff_bundle(manifest,evidence,children,workspace,max_characters=5000)
    assert len(serialize_handoff_bundle(bundle))<=5000
    first=next(c for c in bundle['candidates'] if c['fact_id']=='child_0')
    assert first['navigation_only'] and 'value' not in first
    assert 'child_0' in bundle['remaining']['candidate_ids']
    assert first['read']=={'tool':'read_workspace_page','arguments':{'view':'facts'}}
    assert any(c.get('value')=='严重锈蚀' for c in bundle['candidates'])
    assert '完整的很长业务记录' not in serialize_handoff_bundle(bundle)


def test_huge_original_quote_not_partially_exposed():
    manifest,evidence,children,workspace=_case()
    manifest['units'][1]['text']='不能截成半句。'*3000
    workspace['identity']['manifest_sha256']=json_digest(manifest)
    for key in evidence:
        evidence[key]['source_refs'][0]['end']=len(manifest['units'][1]['text'])
        evidence[key]['source_excerpts'][0].update(text=manifest['units'][1]['text'],end=len(manifest['units'][1]['text']))
        workspace['facts'][key]['source_refs']=deepcopy(evidence[key]['source_refs'])
    bundle=build_handoff_bundle(manifest,evidence,children,workspace,max_characters=5000)
    assert not bundle['source_map'] and not validate_handoff_bundle(manifest,bundle)
    assert set(bundle['remaining']['candidate_ids'])==set(evidence)


def test_large_unresolved_question_remains_explicit_and_fail_tiny_budget():
    manifest,evidence,children,workspace=_case()
    workspace['questions']['Q_huge']={'question':'未决问题完整保留。'*1000,'status':'open','valid':True}
    bundle=build_handoff_bundle(manifest,evidence,children,workspace,max_characters=5000)
    assert 'Q_huge' in bundle['remaining']['issue_ids'] and 'Q_huge' not in bundle['issues']
    assert len(serialize_handoff_bundle(bundle))<=5000
    with pytest.raises(ValueError,match='mandatory'):
        build_handoff_bundle(manifest,evidence,children,workspace,max_characters=10)


@pytest.mark.parametrize('mutation',['text','range','foreign_workspace'])
def test_rejects_unreliable_source_before_building_bundle(mutation):
    manifest,evidence,children,workspace=_case()
    if mutation=='text':evidence['child_0']['source_excerpts'][0]['text']='无依据补写'
    elif mutation=='range':evidence['child_0']['source_refs'][0]['end']=10000
    else:workspace['identity']['manifest_sha256']='0'*64
    with pytest.raises(ValueError):build_handoff_bundle(manifest,evidence,children,workspace)


@pytest.mark.parametrize('mutation',['drop_candidate','drop_issue','source_text','source_key','foreign_manifest'])
def test_validation_detects_corrupt_inventory_and_exposed_source(mutation):
    manifest,evidence,children,workspace=_case()
    bundle=build_handoff_bundle(manifest,evidence,children,workspace)
    if mutation=='drop_candidate':bundle['candidates'].pop()
    elif mutation=='drop_issue':bundle['issues'].pop('Q_free')
    elif mutation=='source_text':next(iter(bundle['source_map'].values()))['text']='假的'
    elif mutation=='source_key':bundle['candidates'][0]['source_keys'].append('missing')
    else:manifest=manifest_from_text('其他报告')
    with pytest.raises(ValueError):validate_handoff_bundle(manifest,bundle)


def test_exposed_ranges_require_actual_main_upstream_receipt(tmp_path):
    manifest,evidence,children,workspace=_case()
    bundle=build_handoff_bundle(manifest,evidence,children,workspace)
    text='主Agent请核对以下候选：'+serialize_handoff_bundle(bundle)
    spans=validate_handoff_bundle(manifest,bundle)
    ledger=SourceDeliveryLedger(manifest,tmp_path,run_id='r',agent_id='main')
    ledger.register_initial_message(text,spans,role='user')
    assert not ledger.confirmed_ranges()
    ledger.observe_request({'messages':[{'role':'user','content':text}]},{'request_id':'real'})
    assert not ledger.confirmed_ranges()
    ledger.confirm_request('real',evidence={'input_delivery_confirmed':True,'request_id':'real'})
    assert ledger.confirmed_ranges()=={'E000002':[[0,len(manifest['units'][1]['text'])]]}


def test_canonical_headers_and_material_context_shared_without_cell_text_duplication(tmp_path):
    case=make_material_case(tmp_path)
    manifest=case['manifest']
    row=next(u for u in manifest['units'] if u.get('table_id')=='T00001' and u.get('row_index')==1)
    ref={'evidence_id':row['id'],'start':0,'end':len(row['text'])}
    evidence={f'c{i}':{'task_id':'task','entry_id':str(i),'section':'病害明细','candidate_value':{'病害描述':'支座轻微锈蚀'},
        'source_refs':[ref],'source_excerpts':[{**ref,'text':row['text']}],'table_headers':[],
        'model_context':{},'unresolved':[]} for i in range(2)}
    header=next(u for u in manifest['units'] if u.get('table_id')=='T00001' and u.get('row_index')==0)
    span={'evidence_id':header['id'],'start':0,'end':len(header['text']),'text':header['text'],'role':'header_candidate'}
    for e in evidence.values():
        e['material_evidence']=[{'identity':{'document_sha256':manifest['document_sha256'],'manifest_sha256':json_digest(manifest)},
            'source_ref':{'evidence_id':'mineru:'+manifest['document_sha256']+'#/tables/50/r1'},
            'context':[span],'source_metadata':{'header_kind':'first_row_candidate','source_provenance':{'full':'xml/path'},
                'cells':[{'text':row['text']}],'table_ref':'#/tables/50'}}]
    bundle=build_handoff_bundle(manifest,evidence,[],{'facts':{},'questions':{}})
    assert len(bundle['source_map'])==2
    assert all(c['header_keys']==bundle['candidates'][0]['header_keys'] for c in bundle['candidates'])
    assert bundle['candidates'][0]['material_context'][0]['context'][0]['role']=='header_candidate'
    assert 'source_provenance' not in serialize_handoff_bundle(bundle)
    assert 'cells' not in serialize_handoff_bundle(bundle)
    assert len(validate_handoff_bundle(manifest,bundle))==2


def test_coverage_question_exposes_exact_structured_scope_and_stable_id():
    manifest,evidence,children,workspace=_case()
    key='Q_coverage_'+json_digest('task-0')[:24]
    scope={'task_id':'task-0','first_source':'E000001','last_source':'E000002','saved_counts':{'病害明细':2}}
    workspace['questions'][key]={'question':'核对分工范围的三类交付是否齐全：'+json.dumps(scope,ensure_ascii=False),
        'status':'open','valid':True}
    bundle=build_handoff_bundle(manifest,evidence,children,workspace)
    assert bundle['issues'][key]['value']['coverage_scope']==scope
    assert bundle['children'][0]['coverage_question_id']==key


def test_current_output_paths_show_selected_fields_and_existing_rows_without_draft_copy():
    manifest,evidence,children,workspace=_case()
    workspace.update(draft={'基本信息':{'风险点':'轻微锈蚀','备注':None},'病害明细':[{'描述':'已保存明细'}],'维修建议':[]},
        draft_bindings={'基本信息.风险点':['child_0'],'病害明细.0':['child_1','child_2']},draft_valid=True)
    bundle=build_handoff_bundle(manifest,evidence,children,workspace)
    assert bundle['candidates'][0]['current_output_paths']==['基本信息.风险点']
    assert bundle['candidates'][1]['current_output_paths']==['病害明细.0']
    assert bundle['current_output']['path_bindings']=={'基本信息.风险点':['child_0'],
        '基本信息.备注':[],'病害明细.0':['child_1','child_2']}
    assert bundle['current_output']['empty_paths']==['基本信息.备注']
    assert '已保存明细' not in serialize_handoff_bundle(bundle)
    workspace['draft_bindings']['病害明细.99']=['child_0']
    with pytest.raises(ValueError,match='absent from the current draft'):
        build_handoff_bundle(manifest,evidence,children,workspace)


def test_failed_child_without_saved_entries_preserves_partial_handoff_issue():
    manifest=manifest_from_text('官方训练原文')
    pending='评分来源仍不确定；尚未保存条目。'
    task='failed-child'
    key='Q_'+json_digest([task,0,pending])[:24]
    workspace={'facts':{},'questions':{key:{'id':key,'question':pending,'status':'open','valid':True}}}
    children=[{'task_id':task,'status':'failed','entry_count':0,'handoff':None,
        'pending_unresolved':[pending],'primary_evidence_ids':['E000001']}]
    bundle=build_handoff_bundle(manifest,{},children,workspace)
    assert bundle['candidate_count']==0 and not bundle['candidates']
    assert bundle['children'][0]['status']=='failed'
    assert bundle['children'][0]['issue_ids']==[key]
    assert bundle['issues'][key]['value']['question']==pending
    assert bundle['issues'][key]['kind']=='workspace_question'
    assert bundle['issue_count']==1
    assert not validate_handoff_bundle(manifest,bundle)
