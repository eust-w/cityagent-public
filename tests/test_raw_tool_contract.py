import json
import pytest
from bridge_evidence.contest.raw_tool_contract import RawToolContractGuard
from bridge_evidence.contest.document_main_review_tools import SCHEMAS


def guard():
    return RawToolContractGuard([{'function': {'name': k, 'parameters': v}} for k, v in SCHEMAS.items()])


def stream(name, parameter, text):
    return '<tool_call>\n<function='+name+'>\n<parameter='+parameter+'>\n'+text+'\n</parameter>\n</function>\n</tool_call>'


@pytest.mark.parametrize('stride', [1, 7, 32, 1024])
def test_unfinished_plain_or_json_string_stopped_across_fragments(stride):
    for name, parameter, body in [('revise_defects', 'items', '[{"candidate":1,"病害描述":"'+''.join(chr(0x4e00+i) for i in range(1800)))]:
        g = guard(); text = stream(name, parameter, body)
        alarm = None
        for i in range(0, len(text), stride):
            alarm = g.feed(text[i:i+stride])
            if alarm:
                break
        assert alarm and alarm['kind'] == 'tool_argument_contract'
        assert alarm['parameter'] == parameter
        assert i < len(text)


@pytest.mark.parametrize('ascii', [False, True])
def test_long_legal_batches_not_capped_as_ordinary_prose(ascii):
    rows = [{'candidate': i+1, '病害描述': '甲乙丙丁戊己庚辛壬癸'*120, '病害部位': '不同位置'+str(i)} for i in range(5)]
    text = stream('revise_defects', 'items', json.dumps(rows, ensure_ascii=ascii))
    assert len(text) > 2048
    g = guard()
    for i in range(0, len(text), 3):
        assert g.feed(text[i:i+3]) is None


def test_surrogate_pairs_and_escaped_quotes_count_as_codepoints():
    text = stream('revise_defects', 'items', json.dumps([{'candidate': 1, '病害描述': '\U0001f600'*1198+'"\\'}]))
    g = guard()
    for c in text:
        assert g.feed(c) is None


def test_multiple_calls_reset_and_do_not_mix_parameters():
    g = guard()
    text = stream('complete_review', 'basis', '核对'*390) + stream('complete_review', 'notes', '未知'*590)
    for c in text:
        assert g.feed(c) is None


def test_raw_contract_cancels_before_parsed_tool_arguments_and_no_partial_result():
    from bridge_evidence.contest.raw_generation_guard import RawGenerationGuard
    from bridge_evidence.contest import reliable_generation as transport
    from test_reliable_generation import Clock, Response, Client, Opener, MODEL
    clock = Clock()
    body = stream('revise_defects', 'items', '[{"candidate":1,"病害描述":"'+''.join(chr(0x4e00+i) for i in range(1800)))
    tokens = {i: c.encode() for i, c in enumerate(dict.fromkeys(body))}
    reverse = {b.decode(): i for i,b in tokens.items()}
    events=[]
    for start in range(0,len(body),16):
        events.append(('data: '+json.dumps({'id':'call','model':MODEL,'choices':[{'index':0,'delta':{},'token_ids':[reverse[c] for c in body[start:start+16]],'finish_reason':None}]})+'\n\n').encode())
    response=Response(events,clock)
    client=Client(Opener(response,clock))
    client.reliable_generation_raw_guard_factory=lambda: RawGenerationGuard(tokens,'0'*64)
    client.reliable_tool_argument_contract=True
    settlements=[]
    def cancel(receipt, connection):
        assert connection.closed
        settlements.append(receipt)
        return {'status':'not_confirmed','generation_stopped':True,'evidence':'synthetic idle observation; not per-request ACK'}
    payload={'model':MODEL,'messages':[{'role':'user','content':'合成验证'}],'max_tokens':9999,
        'tools':[{'type':'function','function':{'name':k,'parameters':v}} for k,v in SCHEMAS.items()]}
    with pytest.raises(transport.ReliableGenerationError) as error:
        transport.stream_chat(client,payload,deadline=999,policy=transport.StreamPolicy(clock=clock),on_cancel=cancel)
    assert error.value.code=='tool_argument_contract'
    assert response.closed and len(settlements)==1 and response.chunks
    assert settlements[0]['tool_call_count']==0


@pytest.mark.parametrize('name', ['complete_review', 'finish_report'])
@pytest.mark.parametrize('encoded', [False, True])
def test_review_explanation_length_is_not_a_cancellation_condition(name, encoded):
    from jsonschema import Draft202012Validator
    basis = ''.join(chr(0x4e00+i) for i in range(929))
    notes = ''.join(chr(0x6000+i) for i in range(1600))
    payload = {'basis': basis, 'notes': notes}
    Draft202012Validator(SCHEMAS[name]).validate(payload)
    g = guard()
    for key, value in payload.items():
        body = stream(name, key, json.dumps(value) if encoded else value)
        for start in range(0, len(body), 32):
            assert g.feed(body[start:start+32]) is None
    for invalid in ({}, {'basis': None}, {'basis': ''}, {'basis': basis, 'notes': []}):
        assert list(Draft202012Validator(SCHEMAS[name]).iter_errors(invalid))


@pytest.mark.parametrize('inner_ascii', [False, True])
def test_encoded_container_keeps_per_field_limits_not_whole_batch(inner_ascii):
    # Supported by the adapter's lossless top-level JSON container conversion.
    data = [{'candidate': i+1, '病害描述': '长表格文字'*220} for i in range(5)]
    encoded = json.dumps(json.dumps(data, ensure_ascii=inner_ascii), ensure_ascii=True)
    g = guard()
    for c in stream('revise_defects', 'items', encoded):
        assert g.feed(c) is None
    bad = json.dumps(json.dumps([{'candidate': 1, '病害描述': '超限'*700}]))
    g = guard(); alarm = None
    for c in stream('revise_defects', 'items', bad):
        alarm = g.feed(c)
        if alarm:
            break
    assert alarm and alarm['observed_string_characters'] == 1201


def test_encoded_basic_field_map_and_emoji():
    g = guard()
    for text in [stream('select_basic_info', 'selections', json.dumps(json.dumps({'桥梁名称':1,'总体评分':2},ensure_ascii=False))),
                 stream('revise_defects','items',json.dumps(json.dumps([{'candidate':1,'病害描述':'\U0001f600'*1200}],ensure_ascii=False)))]:
        for c in text:
            assert g.feed(c) is None
