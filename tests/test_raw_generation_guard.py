"""CPU-only protection tests. Synthetic SSE and optional archived request replay."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from bridge_evidence.contest.local_model_client import LocalModelClient
from bridge_evidence.contest.raw_generation_guard import (
    DualWindowGuard, RawGenerationGuardFactory, NATIVE_REPETITION_POLICY,
)
from bridge_evidence.contest.reliable_generation import (
    _Aggregate, _ProtocolError, _RawGenerationLoop, StreamPolicy,
    ReliableGenerationError, stream_chat,
)
from test_reliable_generation import Client, Clock, MODEL, Opener, Response


def tokenizer_file(tmp_path):
    values = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
    chars = values[:]
    missing = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte); chars.append(256 + missing); missing += 1
    data = {'model': {'type': 'BPE', 'vocab': {chr(c): b for b,c in zip(values,chars)}},
            'decoder': {'type': 'ByteLevel'}, 'added_tokens': []}
    path = tmp_path/'tokenizer.json'
    path.write_text(json.dumps(data))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def wire(ids=None, *, delta=None, finish=None, stop=None, usage=None, prompt_ids=None):
    choice = {'index': 0, 'delta': delta or {}, 'finish_reason': finish}
    if ids is not None: choice['token_ids'] = ids
    if stop is not None: choice['stop_reason'] = stop
    obj = {'id': 'chatcmpl-test', 'model': MODEL, 'created': 123,
           'choices': [choice] if usage is None else [], **({'usage':usage} if usage else {})}
    if prompt_ids is not None: obj['prompt_token_ids'] = prompt_ids
    return ('data: '+json.dumps(obj)+'\n\n').encode()


def finish(n, reason='stop'):
    return (wire(finish=reason) + wire(usage={'prompt_tokens': 7,'completion_tokens':n,'total_tokens':7+n})
            + b'data: [DONE]\n\n')


def factory(tmp_path):
    return RawGenerationGuardFactory(*tokenizer_file(tmp_path))


@pytest.mark.parametrize('chunk',[1,17,101,16000])
def test_hidden_tool_parameter_loop_is_seen_before_parsed_arguments(tmp_path, chunk):
    prefix = '<tool_call><function=save_entries><parameter=entries>[{"id":"'
    generated = (prefix+'H'*1600).encode()
    raw = wire(list(generated[:37]), delta={'tool_calls':[{'index':0,'id':'t',
        'function':{'name':'save_entries','arguments':'{"entries": '}}]})
    raw += b''.join(wire(list(generated[i:i+31])) for i in range(37,len(generated),31))
    agg = _Aggregate(StreamPolicy(), raw_generation_guard=factory(tmp_path)())
    with pytest.raises(_RawGenerationLoop) as caught:
        for i in range(0,len(raw),chunk): agg.feed(raw[i:i+chunk])
    assert caught.value.details['channel'] == 'raw_generated_tokens'
    assert caught.value.details['character_offset'] < 1300
    assert 'tail' not in caught.value.details
    assert agg.content_characters == 0
    assert ''.join(agg.tools[0]['arguments']) == '{"entries": '


@pytest.mark.parametrize('text', [
    json.dumps({'entries':[{'id':str(i),'value':'支座锈蚀','score':100.0} for i in range(300)]},ensure_ascii=False),
    json.dumps({'entries':[{'columns':['构件','病害','数量'], 'row':['支座','锈蚀',1]} for i in range(300)]},ensure_ascii=False),
    json.dumps({'entries':[{'values':[0,0,0,100.0], 'punctuation':'-- / :'} for _ in range(300)]}),
])
def test_normal_long_structured_tables_complete_containers_without_alarm(tmp_path,text):
    rawguard=factory(tmp_path)()
    for b in text.encode(): assert rawguard.feed([b]) is None
    rawguard.validate_complete({'completion_tokens':len(text.encode())})
    assert rawguard.evidence()['decoded_text_sha256'] == hashlib.sha256(text.encode()).hexdigest()


def test_utf8_tokens_are_incremental_and_each_request_has_fresh_state(tmp_path):
    create=factory(tmp_path); a=create(); b=create()
    text='桥梁汉字\U0001f680'
    for byte in text.encode(): assert a.feed([byte]) is None
    assert a.evidence()['decoded_characters_observed'] == len(text)
    assert a.evidence()['pending_utf8_bytes'] == 0
    assert b.evidence()['generated_tokens_observed'] == 0
    assert b.feed([0xe6]) is None
    assert b.evidence()['pending_utf8_bytes'] == 1
    with pytest.raises(ValueError): b.validate_complete({'completion_tokens':1})


def test_prompt_ids_are_not_generated_and_incomplete_ids_fail_closed(tmp_path):
    raw = wire([65],delta={'content':'A'},prompt_ids=[72]*5000)+finish(1)
    agg=_Aggregate(StreamPolicy(),raw_generation_guard=factory(tmp_path)())
    agg.feed(raw);agg.feed(b'',final=True)
    assert agg.result(MODEL)['choices'][0]['message']['content']=='A'
    assert agg.raw_generation_guard.generated_tokens==1
    for raw in [wire(delta={'content':'missing'})+finish(1),
                wire([65],delta={'content':'A'})+finish(2), wire([999999])+finish(1)]:
        agg=_Aggregate(StreamPolicy(),raw_generation_guard=factory(tmp_path)())
        with pytest.raises(_ProtocolError):
            agg.feed(raw);agg.feed(b'',final=True);agg.result(MODEL)


def test_native_terminal_repetition_is_failure_with_bound_response_evidence():
    clock=Clock()
    response=Response([wire(delta={'tool_calls':[{'index':0,'id':'t','function':{
        'name':'save_entries','arguments':'{"entries": '}}]}),
        wire(finish='stop',stop='repetition_detected'), finish(20)],clock,marker_status='unknown')
    client=Client(Opener(response,clock));seen=[]
    with pytest.raises(ReliableGenerationError) as caught:
        stream_chat(client,{'model':MODEL},deadline=700,policy=StreamPolicy(clock=clock),
            on_cancel=lambda receipt,response: seen.append(receipt) or {'status':'not_confirmed'})
    assert caught.value.code=='native_repetition'
    proof=seen[0]['native_stop_observation']
    assert proof=={'response_id':'chatcmpl-test','choice_index':0,'stop_reason':'repetition_detected',
        'finish_reason':'stop','terminal_observed':True,'method':'native_terminal_event'}
    assert caught.value.receipt['cancel_status']=='unknown'
    assert response.closed and not response.completed


def test_raw_alarm_cancels_without_executing_half_argument_and_preserves_capture(tmp_path):
    clock=Clock(); raw=wire(list(b'{"entries": ["'+b'H'*1400))
    response=Response([raw,finish(1414)],clock,advances=[2],marker_status='unknown')
    client=Client(Opener(response,clock));client.reliable_generation_raw_guard_factory=factory(tmp_path)
    captures=[];seen=[]
    with pytest.raises(ReliableGenerationError) as caught:
        stream_chat(client,{'model':MODEL},deadline=700,policy=StreamPolicy(clock=clock),on_chunk=captures.append,
            on_cancel=lambda receipt,response: seen.append(receipt) or {'status':'not_confirmed'})
    receipt=caught.value.receipt
    assert caught.value.code=='raw_repetition'
    assert receipt['detected_after_seconds']==2
    assert receipt['raw_generation_observation']['generated_tokens_observed']==1414
    assert seen[0]['raw_generation_observation']['policy']['slow_chars']==1024
    assert captures==[raw] and not response.completed
    assert response.closed and receipt['cancel_status']=='unknown'


def test_client_configuration_pins_tokenizer_requests_ids_and_native_parameters(tmp_path):
    path,sha=tokenizer_file(tmp_path)
    client=LocalModelClient(base_url='http://127.0.0.1:18110',model=MODEL,enable_thinking=False,temperature=1.0)
    client.enable_reliable_generation(StreamPolicy(),capture_directory=tmp_path/'capture',
        raw_guard_tokenizer_path=path,raw_guard_tokenizer_sha256=sha,native_repetition_detection=True)
    payload=client.reliable_payload({'model':MODEL,'chat_template_kwargs':{'enable_thinking':False}})
    assert payload['return_token_ids'] is True
    assert payload['repetition_detection']==NATIVE_REPETITION_POLICY
    assert payload['repetition_penalty']==1.1
    assert payload['chat_template_kwargs']=={'enable_thinking':False}
    assert client.reliable_payload(payload)==payload
    with pytest.raises(ValueError):
        client.enable_reliable_generation(StreamPolicy(),capture_directory=tmp_path/'capture',
            raw_guard_tokenizer_path=path,raw_guard_tokenizer_sha256='0'*64)
    with pytest.raises(ValueError):
        client.enable_reliable_generation(StreamPolicy(),raw_guard_tokenizer_path=path,raw_guard_tokenizer_sha256=sha)


def test_dual_window_policy_negative_boundary_is_not_semantic_judgement():
    # Changing nonsense need not trigger a repetition detector. Preserve this
    # limitation rather than treating a lack of alarm as task correctness.
    text=' '.join(f'unrelatedword{i:x}chemical{(i*174731):x}' for i in range(2000))
    assert DualWindowGuard().feed(text) is None


@pytest.mark.skipif(not os.environ.get('CITYAGENT_RAW_GUARD_ARCHIVE'),reason='optional frozen request11 evidence')
def test_archived_request11_decodes_full_text_but_dual_window_is_negative():
    root=Path(os.environ['CITYAGENT_RAW_GUARD_ARCHIVE'])
    tokenizer=Path(os.environ['CITYAGENT_RAW_GUARD_TOKENIZER'])
    sha='5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
    agg=_Aggregate(StreamPolicy(),raw_generation_guard=RawGenerationGuardFactory(tokenizer,sha)())
    raw=(root/'request11.sse').read_bytes()
    for i in range(0,len(raw),16384):agg.feed(raw[i:i+16384])
    agg.feed(b'',final=True);result=agg.result(MODEL)
    evidence=agg.raw_generation_guard.evidence()
    assert evidence['generated_tokens_observed']==14315
    assert evidence['decoded_characters_observed']==54349
    assert evidence['decoded_text_sha256']=='b5f429fb6330277d457e7a89594ca2660c2a86a9d3dc9e8fd29093140b032ae3'
    assert result['choices'][0]['message']['tool_calls'][0]['function']['arguments']=='{"entries": '


def test_same_event_raw_alarm_retains_native_terminal_proof(tmp_path):
    from bridge_evidence.contest.generation_stop_observer import generation_stopped
    clock=Clock()
    response=Response([wire([72]*1200,finish='stop',stop='repetition_detected')],clock,
                      marker_status='unknown')
    client=Client(Opener(response,clock));client.reliable_generation_raw_guard_factory=factory(tmp_path)
    seen=[]
    with pytest.raises(ReliableGenerationError) as caught:
        stream_chat(client,{'model':MODEL},deadline=700,policy=StreamPolicy(clock=clock),
            on_cancel=lambda receipt,response: seen.append(receipt) or {'status':'not_confirmed'})
    receipt=caught.value.receipt
    assert caught.value.code=='raw_repetition'
    assert receipt['finish_reason']=='stop'
    assert seen[0]['native_stop_observation']['response_id']=='chatcmpl-test'
    assert generation_stopped(receipt)
    assert receipt['cancel_status']=='unknown'  # terminal server event is distinct from abort acknowledgement
    assert response.closed and not response.completed


def test_raw_guard_rejects_wrong_tokenizer_type_and_missing_pinned_hash(tmp_path):
    path,sha=tokenizer_file(tmp_path)
    data=json.loads(path.read_text());data['decoder']['type']='WordPiece';path.write_text(json.dumps(data))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='ByteLevel BPE'):RawGenerationGuardFactory(path,sha)
    with pytest.raises(ValueError,match='SHA256'):RawGenerationGuardFactory(path,None)
