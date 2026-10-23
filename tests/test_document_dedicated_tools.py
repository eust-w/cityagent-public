from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator
from test_document_section_submission import setup
from bridge_evidence.contest.document_dedicated_tools import DedicatedSectionTools, dedicated_schema, SAVE_SECTIONS
from bridge_evidence.contest.tong_react_adapter import _normalize_container_arguments, _section_tool_schema
from bridge_evidence.contest.reliable_generation import _Aggregate, StreamPolicy, _ContentBudgetExceeded


def flat(entry):
    return {**entry['row'], **{k:v for k,v in entry.items() if k not in ('row','section')}}


def test_fixed_shapes_and_strict_array_conversion():
    for name in SAVE_SECTIONS:
        schema=dedicated_schema(name)
        assert schema['required']==['entries']
        item=schema['properties']['entries']['items']
        assert 'anyOf' not in item and 'section' not in item['properties'] and 'row' not in item['properties']
        normalized,changes,issues=_normalize_container_arguments({'entries':'[]'},schema)
        assert normalized=={'entries':[]} and changes and not issues
        # Syntax repair is never guessed.
        normalized,changes,issues=_normalize_container_arguments({'entries':'[{"field":'},schema)
        assert not changes and issues
    assert _section_tool_schema('complete_section')['required']==['summary','unresolved']


def test_save_all_three_types_handoff_and_readback(setup):
    setup['read_rest']();h=DedicatedSectionTools(setup['handler']);entry=flat(setup['entry'])
    assert h.save('save_defects',[entry])['ok']
    common={k:deepcopy(entry[k]) for k in ('source_refs','status')}
    assert h.save('save_basic_info',[dict(common,field='桥梁名称',value='合成桥')])['ok']
    assert h.save('save_recommendations',[dict(common,建议内容='除锈涂漆')])['ok']
    state=setup['store'].snapshot()
    assert state['counts']['total']==3
    assert {r['section']for r in state['entries'].values()}==set(SAVE_SECTIONS.values())
    done=h.complete([])
    assert done['handed_off'] and done['counts']==state['counts']
    assert setup['calls'][0]['unresolved']==[]


def test_failure_retains_previous_batch_and_cannot_finish_silently(setup):
    setup['read_rest']();h=DedicatedSectionTools(setup['handler']);entry=flat(setup['entry'])
    first=h.save('save_defects',[entry]);assert first['ok']
    before=setup['store'].path.read_bytes()
    bad=deepcopy(entry);bad['source_refs']=[{'evidence_id':'foreign'}]
    assert not h.save('save_defects',[bad])['ok']
    assert setup['store'].path.read_bytes()==before
    assert not h.complete([])['ok'] and not setup['calls']
    assert h.save('save_defects',[entry])['changed'] is False
    assert h.complete([])['handed_off']


def test_explicit_gap_preserves_failure_in_handoff(setup):
    setup['read_rest']();h=DedicatedSectionTools(setup['handler'])
    assert not h.save('save_defects',[{'病害描述':'x'}])['ok']
    result=h.complete(['未保存的病害待主Agent处理'])
    assert result['handed_off']
    assert any('保存失败' in x for x in setup['calls'][0]['unresolved'])


def test_small_batches_detailed_path_atomicity_and_replay(setup):
    h=DedicatedSectionTools(setup['handler']);row=flat(setup['entry'])
    good=h.save('save_defects',[row,row]);assert good['counts']['total']==2
    assert not h.save('save_defects',[row,row])['changed']
    before=setup['store'].path.read_bytes()
    bad=deepcopy(row);bad.pop('status')
    error=h.save('save_defects',[row,bad]);assert not error['ok']
    assert error['error']['details']['path']=='entries[1].status'
    assert setup['store'].path.read_bytes()==before
    assert h.save('save_defects',[row]*9)['ok']
    assert not h.save('save_recommendations',[row])['ok']


def test_large_complete_batch_keeps_all_entries_and_atomic_corrections(setup):
    h = DedicatedSectionTools(setup['handler'])
    row = flat(setup['entry'])
    entries = [{**row, '病害描述': f'已读原文的合成条目{i}'} for i in range(10)]
    schema = dedicated_schema('save_defects')
    Draft202012Validator(schema).validate({'entries': entries})
    result = h.save('save_defects', entries)
    assert result['ok'] and result['saved_indices'] == list(range(10))
    assert result['counts']['total'] == 10
    saved = setup['store'].path.read_bytes()
    assert not h.save('save_defects', entries)['changed']
    corrections = [{**entry, 'id': assignment['id']} for entry, assignment in zip(entries, result['assignments'])]
    corrections[0]['病害描述'] = '修改'
    corrections[-1]['source_refs'] = [{'evidence_id': 'foreign'}]
    rejected = h.save('save_defects', corrections)
    assert not rejected['saved'] and rejected['error']['details']['atomic_corrections']
    assert setup['store'].path.read_bytes() == saved


def sse(delta,finish=None,usage=None):
    v={'id':'r','model':'m','created':1,'choices':[{'index':0,'delta':delta,'finish_reason':finish}]}
    if usage:v['usage']=usage
    return ('data: '+json.dumps(v,ensure_ascii=False)+'\n\n').encode()


def test_prose_budget_accumulates_across_fragments_and_keeps_long_tool_data():
    agg=_Aggregate(StreamPolicy(max_content_chars=2048))
    raw=sse({'content':''.join(chr(0x4e00+i) for i in range(1200))})+sse({'content':''.join(chr(0x5300+i) for i in range(1200))})
    with pytest.raises(_ContentBudgetExceeded):
        for b in raw:agg.feed(bytes([b]))
    assert agg.content_characters==2400
    normal=_Aggregate(StreamPolicy(max_content_chars=2048))
    # Legitimate long tables and repeated headers are tool arguments, not prose.
    args=json.dumps({'entries':[{'病害描述':'桥台表头 2018 1.00，；'*1000}]},ensure_ascii=False)
    for i in range(0,len(args),17):
        normal.feed(sse({'tool_calls':[{'index':0,'id':'t','type':'function','function':{'name':'save_defects' if i==0 else '', 'arguments':args[i:i+17]}}]}))
    assert normal.content_characters==0 and len(normal.tools)==1


def test_new_good_rows_survive_bad_sibling_with_specific_feedback(setup):
    setup['read_rest']();h=DedicatedSectionTools(setup['handler']);good=flat(setup['entry'])
    bad=deepcopy(good);bad['source_refs']=[]
    r=h.save('save_defects',[good,bad])
    assert not r['ok'] and r['saved'] and r['saved_indices']==[0]
    assert r['counts']['total']==1
    assert r['error']['details']['rejected'][0]['path']=='entries[1].source_refs'
    assert not h.complete([])['ok']
    assert h.complete(['第二项没有出处，已放弃该项'])['handed_off']


def test_mixed_explicit_corrections_remain_atomic(setup):
    h=DedicatedSectionTools(setup['handler']);row=flat(setup['entry'])
    first=h.save('save_defects',[row]);ident=first['assignments'][0]['id']
    update={**row,'id':ident,'病害描述':'更正'};bad={**row,'source_refs':[]}
    before=setup['store'].path.read_bytes();r=h.save('save_defects',[update,bad])
    assert not r['saved'] and r['error']['details']['atomic_corrections']
    assert setup['store'].path.read_bytes()==before


def test_handoff_plain_text_is_preserved_without_parsing_json(setup):
    setup['read_rest']();h=DedicatedSectionTools(setup['handler'])
    note='["未决说明"】这是一段普通文本，不是要修复的参数JSON'
    assert h.handoff(note)['handed_off']
    assert setup['calls'][0]['unresolved']==[note]
    schema=_section_tool_schema('handoff_section')
    Draft202012Validator(schema).validate({'notes':note})
    Draft202012Validator(schema).validate({})


def test_known_aliases_preserve_values_and_conflicts_never_overwrite(setup):
    setup['read_rest']();h=DedicatedSectionTools(setup['handler'])
    common={k:deepcopy(flat(setup['entry'])[k]) for k in ('source_refs','status')}
    advice={**common,'建议内容':'除锈涂漆','源部件':'左幅支座'}
    original=deepcopy(advice)
    r=h.save('save_recommendations',[advice]);assert r['ok'] and r['field_normalizations']
    assert advice==original
    saved=setup['store'].snapshot()['entries'][r['assignments'][0]['id']]
    assert saved['row']=={'建议内容':'除锈涂漆','病害部位':'左幅支座'}
    r=h.save('save_defects',[{**common,'病害描述':'锈蚀','development_degree':'未见发展'}])
    assert r['ok']
    before=setup['store'].path.read_bytes()
    r=h.save('save_recommendations',[{**advice,'病害部位':'右幅支座'}])
    assert r['error']['details']['path']=='entries[0].源部件'
    assert r['error']['details']['rejected'][0]['reason']=='conflicting_alias'
    assert setup['store'].path.read_bytes()==before


def test_unknown_field_lists_name_allowed_fields_and_keeps_good_sibling(setup):
    h=DedicatedSectionTools(setup['handler']);good=flat(setup['entry'])
    bad={**good,'state':'显式'}
    r=h.save('save_defects',[good,bad])
    issue=r['error']['details']['rejected'][0]
    assert r['saved_indices']==[0] and issue['entry_index']==1
    assert issue['path']=='entries[1].state' and issue['unexpected_fields']==['state']
    assert 'status' in issue['allowed_fields'] and issue['message']
    assert setup['store'].snapshot()['counts']['total']==1


def test_invalid_reference_feedback_uses_actual_batch_index(setup):
    h=DedicatedSectionTools(setup['handler']);good=flat(setup['entry'])
    bad={**good,'source_refs':[{'evidence_id':'mineru:wrong#/texts/made-up'}]}
    r=h.save('save_defects',[good,bad])
    issue=r['error']['details']['rejected'][0]
    assert issue['entry_index']==1 and issue['path'].startswith('entries[1].source_refs')
    assert issue['message'] and r['saved_indices']==[0]
