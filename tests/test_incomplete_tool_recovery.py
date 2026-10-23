"""Replay synthetic SSE through isolated real client/gateway; no model service."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bridge_evidence.contest.document_model_gateway import AuditedModelGateway, verified_incomplete_tool_terminal
from bridge_evidence.contest.document_workflow import DocumentWorkflow
from bridge_evidence.contest.local_model_client import LocalModelClient
from bridge_evidence.contest.reliable_generation import ReliableGenerationError, StreamPolicy


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()


def event(delta=None, finish=None, usage=None, *, model='own-model'):
    value={'id':'synthetic-response', 'model':model,
        'choices': [] if usage is not None else [{'index':0, 'delta':delta or {}, 'finish_reason':finish}]}
    if usage is not None:value['usage']=usage
    return b'data: '+packed(value)+b'\n\n'


def stream(*,name='save_defects', identifier='tool-1', done=True, usage=True,
           prompt=23, completion=3, total=None, finish='tool_calls', model='own-model', arguments='{"entries":[{"text":"synthetic"}]}',
           mixed=False):
    tool={'index':0, 'function':{'arguments':arguments}}
    if identifier is not None:tool['id']=identifier
    if name is not None:tool['function']['name']=name
    rows=[tool]
    if mixed:
        rows=[{'index':0,'id':'valid-part','function':{'name':'save_defects','arguments':'{"entries":[]}' }},
              {**tool,'index':1}]
    raw=event({'tool_calls':rows},model=model)+event(finish=finish,model=model)
    if usage:raw+=event(usage={'prompt_tokens':prompt,'completion_tokens':completion,'total_tokens':prompt+completion if total is None else total},model=model)
    if done:raw+=b'data: [DONE]\n\n'
    return raw


class Clock:
    now=100.
    def __call__(self):return self.now


class Response:
    status=200
    headers={'Content-Type':'text/event-stream'}
    def __init__(self,raw):self.chunks=[raw];self.closed=False
    def set_read_timeout(self,seconds):pass
    def read1(self,size):return self.chunks.pop(0) if self.chunks else b''
    def close(self):self.closed=True


class JSONResponse:
    status=200
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return packed({'count':23})


class Opener:
    def __init__(self,streams):self.responses=list(streams);self.requests=[]
    def open(self,request,*,timeout):
        body=json.loads(request.data)
        self.requests.append((request.full_url,body))
        if request.full_url.endswith('/tokenize'):return JSONResponse()
        if not self.responses:raise AssertionError('Unexpected extra generation')
        return self.responses.pop(0)


class Tests(unittest.TestCase):
    def setup_route(self,streams,*,max_calls=6,cancel=None):
        self.clock=Clock()
        self.client=LocalModelClient('http://127.0.0.1:9','own-model',timeout=900,
            max_model_len=512,max_output_tokens=16,enable_thinking=False,temperature=1)
        self.client._opener=Opener([Response(s) for s in streams])
        self.client.enable_reliable_generation(StreamPolicy(clock=self.clock),deadline=200.,on_cancel=cancel)
        self.saved={'entries':['already saved'], 'cursor':17, 'version':9}
        self.initial_saved=deepcopy(self.saved)
        self.feedback_calls=[]
        self.responses=[]
        def feedback(receipt):
            self.feedback_calls.append(deepcopy(receipt))
            return 'The previous tool envelope was rejected and executed nothing. Saved version=9; submit one pending group.'
        self.gateway=AuditedModelGateway(self.client,deadline=200.,max_calls=max_calls,
            max_output_tokens=16,recovery_prompt=feedback,
            on_response=lambda response,audit:self.responses.append(deepcopy(response)))
        self.payload={'model':'own-model','max_tokens':16,'temperature':1.0,
            'chat_template_kwargs':{'enable_thinking':False},'messages':[
            {'role':'user','content':'synthetic assigned source and saved progress'},
            {'role':'assistant','content':None,'tool_calls':[{'id':'previous','type':'function',
                'function':{'name':'read_source','arguments':'{}'}}]},
            {'role':'tool','tool_call_id':'previous','content':'synthetic earlier evidence'}],
            'tools':[{'type':'function','function':{'name':'save_defects','parameters':
                {'type':'object','properties':{'entries':{'type':'array'}}}}}], 'tool_choice':'auto'}
        self.original_payload=deepcopy(self.payload)
        self.executed=[]

    def run_forward(self):
        with patch('bridge_evidence.contest.local_model_client.time.monotonic',new=self.clock):
            response=self.gateway.forward(self.payload)
        # Explicit fake business dispatch is downstream of the real gateway;
        # no partial aggregate reaches this boundary on a failed response.
        for tool in response['choices'][0]['message'].get('tool_calls',[]):
            self.executed.append((tool['function']['name'],json.loads(tool['function']['arguments'])))
        return response

    def connect_actual_workflow(self, *, maximum=16, prior=0):
        # Real callbacks and budget gate, isolated from disk rendering/model IO.
        wf = object.__new__(DocumentWorkflow)
        wf.client = self.client
        wf.compression_client = None
        wf.native_compression = None
        wf.settings = SimpleNamespace(single_document_agent=True, dynamic_output_budget=False,
                                      max_model_calls=maximum)
        wf.model_calls_before_resume = [
            {'kind':'agent_generation','call_id':'old-'+str(i),'status':'success'} for i in range(prior)]
        wf.model_dispatches = []
        wf.gateway_receipts = self.gateway.receipts
        wf.deadline = 200.
        wf.source_objects = None
        wf._single_agent_terminal_failure = None
        wf._persist = lambda: None
        self.gateway.before_request = wf._check_budget
        self.gateway.on_dispatch = wf._on_dispatch
        self.gateway.on_receipt = wf._on_model_receipt
        self.workflow = wf
        return wf

    def generations(self):return [c for c in self.client.calls if c.get('kind')=='agent_generation']

    def test_valid_tool_uses_one_call_no_recovery(self):
        self.setup_route([stream()]);self.run_forward()
        self.assertEqual(len(self.generations()),1)
        self.assertEqual(len(self.executed),1)
        self.assertEqual(self.feedback_calls,[])

    def test_missing_name_done_usage_retries_once_with_saved_progress(self):
        self.setup_route([stream(name=None),stream()]);self.run_forward()
        failed,valid=self.generations()
        self.assertEqual(failed['error_code'],'incomplete_tool_call')
        self.assertEqual(failed['protocol_error'],{'reason':'incomplete_tool_call',
            'details':{'tool_index':0,'missing_fields':['function.name']}})
        self.assertEqual(failed['usage'],failed['observed_usage'])
        self.assertTrue(failed['usage_known'])
        self.assertEqual(failed['status'],'failed')
        self.assertEqual(len(self.executed),1)
        self.assertEqual(len(self.responses),1)
        self.assertEqual(len(self.feedback_calls),1)
        self.assertEqual(self.saved,self.initial_saved)
        self.assertEqual(self.payload,self.original_payload)
        self.assertEqual(self.client._repetition_recoveries_used,1)
        wire=[body for url,body in self.client._opener.requests if url.endswith('/chat/completions')]
        self.assertEqual(wire[1]['messages'][:-1],wire[0]['messages'])
        self.assertEqual(wire[1]['messages'][-1]['role'],'user')
        self.assertNotIn('tool-1',json.dumps(wire[1]['messages']))
        self.assertEqual(self.gateway._issued,2)

    def test_missing_id_done_uses_same_narrow_recovery(self):
        self.setup_route([stream(identifier=None),stream()]);self.run_forward()
        self.assertEqual(self.generations()[0]['protocol_error']['details']['missing_fields'],['id'])
        self.assertEqual(len(self.executed),1)

    def test_multiple_tool_batch_with_one_missing_name_executes_nothing_from_bad_batch(self):
        self.setup_route([stream(name=None,mixed=True),stream()]);self.run_forward()
        self.assertEqual(len(self.executed),1)
        self.assertEqual(self.generations()[0]['tool_call_count'],2)
        self.assertEqual(len(self.responses),1)

    def test_without_done_unknown_cancel_never_retries(self):
        self.setup_route([stream(name=None,done=False),stream()])
        with self.assertRaises(ReliableGenerationError):self.run_forward()
        self.assertEqual(len(self.generations()),1)
        self.assertEqual(self.feedback_calls,[])
        self.assertEqual(self.executed,[])
        self.assertFalse(self.generations()[0]['usage_known'])
        self.assertIsNone(self.generations()[0]['usage'])

    def test_without_done_even_unverified_confirm_callback_never_retries(self):
        self.setup_route([stream(name=None,done=False),stream()],
            cancel=lambda *args:{'status':'cancellation_confirmed'})
        with self.assertRaises(ReliableGenerationError):self.run_forward()
        self.assertEqual(len(self.generations()),1)
        self.assertNotEqual(self.generations()[0]['cancel_status'],'cancellation_confirmed')

    def test_missing_usage_is_not_promoted_or_retried(self):
        self.setup_route([stream(name=None,usage=False),stream()])
        with self.assertRaises(ReliableGenerationError):self.run_forward()
        self.assertEqual(len(self.generations()),1)
        self.assertFalse(self.generations()[0]['usage_known'])

    def test_wrong_prompt_usage_rejected_without_retry(self):
        self.setup_route([stream(name=None,prompt=24),stream()])
        with self.assertRaises(ReliableGenerationError):self.run_forward()
        self.assertEqual(len(self.generations()),1)
        self.assertEqual(self.feedback_calls,[])
        self.assertFalse(self.generations()[0]['usage_known'])
        self.assertEqual(self.generations()[0]['observed_usage']['prompt_tokens'],24)

    def test_wrong_model_never_retried(self):
        self.setup_route([stream(name=None,model='other-model'),stream()])
        with self.assertRaises(ReliableGenerationError):self.run_forward()
        self.assertEqual(len(self.generations()),1)
        self.assertEqual(self.generations()[0]['protocol_error']['reason'],'missing_or_wrong_response_identity')

    def test_malformed_arguments_are_not_guessed_or_completed_by_transport(self):
        self.setup_route([stream(arguments='{"entries":')])
        with patch('bridge_evidence.contest.local_model_client.time.monotonic',new=self.clock):
            response=self.gateway.forward(self.payload)
        tool=response['choices'][0]['message']['tool_calls'][0]
        self.assertEqual(tool['function']['arguments'],'{"entries":')
        with self.assertRaises(json.JSONDecodeError):json.loads(tool['function']['arguments'])
        self.assertEqual(self.feedback_calls,[])
        self.assertEqual(self.executed,[])

    def test_missing_name_and_truncated_arguments_never_execute_partial_parameters(self):
        self.setup_route([stream(name=None,arguments='{"entries":'),stream()]);self.run_forward()
        self.assertEqual(len(self.executed),1)
        self.assertEqual(self.executed[0][0],'save_defects')
        self.assertEqual(self.payload,self.original_payload)

    def test_failed_retry_counts_toward_original_call_budget(self):
        self.setup_route([stream(name=None),stream()],max_calls=1)
        with self.assertRaises(RuntimeError):self.run_forward()
        self.assertEqual(len(self.generations()),1)
        self.assertEqual(self.executed,[])
        self.assertEqual(self.gateway._issued,1)

    def test_second_incomplete_failure_ends_child_retaining_progress_for_main(self):
        self.setup_route([stream(name=None),stream(name=None),stream()])
        with self.assertRaises(ReliableGenerationError):self.run_forward()
        self.assertEqual(len(self.generations()),2)
        self.assertEqual(self.executed,[])
        self.assertEqual(self.saved,self.initial_saved)
        self.assertEqual(len(self.feedback_calls),1)
        self.assertEqual(self.gateway.receipts[-1]['recovery_blocked']['reason'],
                         'same_request_recovery_exhausted')

    def test_recovery_allowance_is_per_agent_client_not_per_forward(self):
        self.setup_route([stream(name=None),stream(),stream(name=None),stream()])
        self.run_forward()
        with self.assertRaises(ReliableGenerationError):self.run_forward()
        self.assertEqual(len(self.generations()),3)
        self.assertEqual(len(self.feedback_calls),1)

    def test_source_payload_and_fixed_sampling_are_preserved(self):
        self.setup_route([stream(name=None),stream()]);self.run_forward()
        wire=[body for url,body in self.client._opener.requests if url.endswith('/chat/completions')]
        for body in wire:
            self.assertFalse(body['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(body['repetition_penalty'],1.1)
            self.assertEqual(body['presence_penalty'],1.5)
            self.assertEqual(body['tools'],self.original_payload['tools'])


    def test_real_workflow_receipt_callback_allows_exact_recovery(self):
        self.setup_route([stream(name=None), stream()])
        wf = self.connect_actual_workflow()
        self.run_forward()
        self.assertIsNone(wf._single_agent_terminal_failure)
        self.assertEqual(len(self.generations()), 2)
        self.assertEqual(len(self.executed), 1)
        self.assertTrue(all(row['status']=='settled' for row in wf.model_dispatches))
        self.assertEqual(wf._active_agent_call_id, self.generations()[-1]['call_id'])
        failed = self.gateway.receipts[0]
        self.assertTrue(failed['recovery_authorized'])
        # No malformed output reaches the response hook or active SDK history.
        self.assertEqual(len(self.responses),1)

    def test_real_workflow_refuses_unproven_protocol_receipt(self):
        self.setup_route([stream(name=None),stream()]);self.run_forward()
        wf = self.connect_actual_workflow()
        receipt=deepcopy(self.gateway.receipts[0])
        receipt['failed_generation']['done_received']=False
        wf._on_model_receipt(receipt)
        self.assertIsNotNone(wf._single_agent_terminal_failure)
        with self.assertRaisesRegex(RuntimeError,'progress saved'):
            wf._check_budget()

    def test_cumulative_main_sixteen_cap_preserved_at_last_slot(self):
        self.setup_route([stream(name=None),stream()], max_calls=16)
        wf = self.connect_actual_workflow(prior=15,maximum=16)
        with self.assertRaisesRegex(RuntimeError,'cumulative model request budget'):
            self.run_forward()
        self.assertEqual(len(wf._model_calls()),17)  # 15 old + tokenizer + failed generation
        self.assertEqual(len(self.generations()),1)
        self.assertEqual(len([c for c in wf._model_calls() if c.get('kind')=='agent_generation']),16)
        self.assertEqual(self.executed,[])

    def test_cumulative_main_permits_final_two_slots_only(self):
        self.setup_route([stream(name=None),stream()],max_calls=16)
        wf = self.connect_actual_workflow(prior=14,maximum=16)
        self.run_forward()
        self.assertEqual(len(self.generations()),2)
        with patch('bridge_evidence.contest.local_model_client.time.monotonic',new=self.clock):
            with self.assertRaisesRegex(RuntimeError,'cumulative model request budget'):
                wf._check_budget()

    def test_prior_loop_recovery_consumes_same_protocol_allowance(self):
        self.setup_route([stream(name=None),stream()])
        self.client._repetition_recoveries_used=1
        with self.assertRaises(ReliableGenerationError):self.run_forward()
        self.assertEqual(len(self.generations()),1)
        self.assertEqual(self.feedback_calls,[])
        self.assertEqual(self.gateway.receipts[-1]['recovery_blocked']['reason'],
                         'agent_recovery_allowance_exhausted')

    def test_complete_rejected_response_does_not_wait_for_other_reports_idle(self):
        cancel=[]
        self.setup_route([stream(name=None),stream()],cancel=lambda *args:cancel.append(args))
        self.run_forward()
        failed=self.generations()[0]
        self.assertEqual(cancel,[])
        self.assertEqual(failed['cancel_status'],'not_requested_response_ended')
        self.assertFalse(failed['cancel_attempted'])
        self.assertNotIn('cancellation_settlement',failed)
        self.assertNotIn('cancellation',self.gateway.receipts[0])

    def test_completion_over_exact_output_cap_never_retries(self):
        self.setup_route([stream(name=None,completion=17),stream()])
        with self.assertRaises(ReliableGenerationError):self.run_forward()
        self.assertEqual(len(self.generations()),1)
        self.assertFalse(self.generations()[0]['usage_known'])
        self.assertEqual(self.generations()[0]['observed_usage']['completion_tokens'],17)

    def test_invalid_total_usage_never_retries(self):
        self.setup_route([stream(name=None,total=99),stream()])
        with self.assertRaises(ReliableGenerationError):self.run_forward()
        self.assertEqual(len(self.generations()),1)
        self.assertFalse(self.generations()[0]['usage_known'])

    def test_usage_total_over_context_is_not_terminal_proof(self):
        self.setup_route([stream(name=None),stream()]);self.run_forward()
        row=self.generations()[0]
        self.assertFalse(verified_incomplete_tool_terminal(row,model='own-model',
            prompt_tokens=23,max_tokens=16,max_model_len=25))

    def test_length_is_not_promoted_to_incomplete_tool_recovery(self):
        self.setup_route([stream(name=None,finish='length'),stream()])
        with self.assertRaises(ReliableGenerationError):self.run_forward()
        self.assertEqual(len(self.generations()),1)
        self.assertFalse(self.generations()[0]['usage_known'])

    def test_response_identity_change_never_retries(self):
        raw=stream(name=None).replace(b'"id":"synthetic-response"',
                b'"id":"changed-response"',1)
        self.setup_route([raw,stream()])
        with self.assertRaises(ReliableGenerationError):self.run_forward()
        self.assertEqual(self.generations()[0]['protocol_error']['reason'],'response_identity_changed')
        self.assertEqual(len(self.generations()),1)

    def test_no_done_late_idle_does_not_authorize_protocol_retry(self):
        # A genuine later idle observation may prove quiescence, but cannot
        # turn missing DONE into complete valid protocol/usage.
        def stopped(receipt,response):
            return {'status':'not_confirmed','stop_observation':{
              'response_id':receipt['response_id'],'stopped_observed':True,
              'method':'service_idle_after_disconnect','same_service_process':True,
              'cancellation_confirmed':False,'model_identity_sha256':'model',
              'process_identity_sha256':'process','samples':[
                 {'idle':True,'after_disconnect_seconds':1},
                 {'idle':True,'after_disconnect_seconds':2}]}}
        self.setup_route([stream(name=None,done=False),stream()],cancel=stopped)
        with self.assertRaises(ReliableGenerationError):self.run_forward()
        self.assertEqual(len(self.generations()),1)
        self.assertIsNone(self.generations()[0]['usage'])

    def test_recovery_cannot_reset_enclosing_request_deadline(self):
        self.setup_route([stream(name=None),stream()])
        original=self.gateway.recovery_prompt
        def expensive_feedback(receipt):
            text=original(receipt);self.clock.now=201.;return text
        self.gateway.recovery_prompt=expensive_feedback
        with self.assertRaises(TimeoutError):self.run_forward()
        self.assertEqual(len(self.generations()),1)
        self.assertEqual(self.executed,[])


    def test_actual_child_failure_result_keeps_saved_entries_for_main(self):
        from bridge_evidence.contest import document_multiagent_worker as worker
        from bridge_evidence.contest.document_entries import EntryStore
        from bridge_evidence.contest.evidence_manifest import manifest_from_text
        from bridge_evidence.contest.run_state import atomic_json, json_digest
        self.setup_route([stream(name=None),stream(name=None),stream()])
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td)/'child'
            manifest=manifest_from_text('synthetic source bridge report')
            source=Path(td)/'source.json';atomic_json(source,manifest)
            unit=manifest['units'][0]
            store=EntryStore(manifest,'part',directory/'entries')
            store.record_read(unit['id'],0,len(unit['text']))
            store.save_entries([{'id':'saved-name','section':'基本信息','field':'桥梁名称',
                'value':'synthetic bridge','source_refs':[{'evidence_id':unit['id']}],
                'status':'explicit'}])
            before=store.snapshot()
            request={'manifest_path':str(source),'manifest_sha256':json_digest(manifest),
                'task':{'id':'part','primary_evidence_ids':[unit['id']]},
                'model_config':{'enable_thinking':False},'deadline_monotonic':200.,
                'max_model_calls':6}
            gateways=[]
            def gateway_factory(*args,**kwargs):
                gateway=AuditedModelGateway(*args,**kwargs);gateways.append(gateway);return gateway
            def react(prompt,handlers,config):
                # Request goes through real gateway and real worker callbacks.
                gateways[-1].forward(self.payload)
                self.fail('two rejected envelopes cannot become a completed child')
            with patch('bridge_evidence.contest.local_model_client.time.monotonic',new=self.clock), \
                 patch.object(worker,'AuditedModelGateway',side_effect=gateway_factory):
                result=worker.run_section(request,directory,client=self.client,react_runner=react)
            persisted=json.loads((directory/'result.json').read_text())
            self.assertEqual(persisted,result)
            self.assertEqual(result['status'],'failed')
            self.assertFalse(result['document_delivery_complete'])
            self.assertTrue(result['generation_attempts_settled'])
            self.assertEqual(result['model_calls'],2)
            self.assertEqual(result['entries']['entries'],before['entries'])
            self.assertEqual(result['entries']['reads'],before['reads'])
            self.assertEqual(result['entries']['version'],before['version'])
            self.assertEqual(result['error']['type'],'ReliableGenerationError')
            recovery=json.loads((directory/'generation-recovery.json').read_text())
            self.assertFalse(recovery['failed_output_executed'])
            self.assertEqual(recovery['additional_recovery_limit'],0)
            self.assertEqual(gateways[-1].receipts[-1]['recovery_blocked']['reason'],
                             'same_request_recovery_exhausted')
            self.assertEqual(len(self.generations()),2)


if __name__=='__main__':unittest.main(verbosity=2)
