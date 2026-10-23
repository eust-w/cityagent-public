"""Synthetic source-object integration: real local HTTP/gateway/workspace only.

The responder fabricates decisions and native tool calls; it is not TongAgents
or a real model. Generated DOCX/PNG fixtures contain no report/test-set data.
Set PYTHONPATH=code/src. No remote request, processor or weights are used.
"""
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time
import zipfile

from PIL import Image
import pytest

from bridge_evidence.contest import document_workflow as workflow
from bridge_evidence.contest.document_model_gateway import AuditedModelGateway
from bridge_evidence.contest.evidence_manifest import build_manifest
from bridge_evidence.contest.local_model_client import LocalModelClient
from bridge_evidence.contest.run_state import json_digest
from bridge_evidence.contest.source_object_review import SourceObjectReview, SourceObjectReviewError
from test_document_source_objects import synthetic_docx
from test_document_workflow import completed, install_native_review_driver

MODEL = 'SYNTHETIC-OWN-MODEL-SOURCE-WORKFLOW'
ISSUES = ('unsupported_claims', 'missing_items', 'conflicts', 'unresolved_items')


class SyntheticVision:
    model, version, cache_key = 'synthetic-pixel-fixture', 'v1', 'synthetic-pixel-fixture-v1'

    def __call__(self, _blob, *, question, **_kwargs):
        return {'text': '合成缓存图观察，未确认原生公式字形。', 'question': question,
                'verification': 'unverified'}


@contextmanager
def local_responder():
    requests = []
    controls = {'source_ack_only': False, 'sse': False, 'omit_done': False}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path, payload))
            if self.path == '/tokenize':
                response = {'count': 65}  # Synthetic count; not a BPE assertion.
            elif self.path == '/v1/chat/completions':
                assignment = json.loads(payload['messages'][-1]['content'])
                reason = 'stop'
                if 'synthetic_agent' in assignment:
                    agent = assignment['synthetic_agent']
                    reason = agent.get('finish_reason', 'tool_calls')
                    message = {'role': 'assistant', 'content': agent.get('content'),
                               'tool_calls': agent.get('tool_calls', [])}
                elif assignment.get('purpose') == 'source_object_disposition_review':
                    decision = {'assignment_id': assignment['assignment_id']}
                    if not controls['source_ack_only']:
                        decision['decisions'] = [
                            {'proposal_id': row['proposal_id'], 'object_id': row['object_id'],
                             'verdict': 'accepted', 'explanation': 'Synthetic independent task nonreliance decision.',
                             **{key: [] for key in ISSUES}} for row in assignment['proposals']]
                    message = {'role': 'assistant', 'content': json.dumps(decision, ensure_ascii=False)}
                else:
                    decision = {'passed': True, 'checked_assignment_id': assignment['review_assignment']['id'],
                                'checked_all_assigned': True, **{key: [] for key in ISSUES}}
                    message = {'role': 'assistant', 'content': json.dumps(decision, ensure_ascii=False)}
                response = {'id': f'synthetic-local-response-{len(requests)}', 'model': MODEL,
                            'choices': [{'index': 0, 'finish_reason': reason, 'message': message}],
                            'usage': {'prompt_tokens': 65, 'completion_tokens': 11, 'total_tokens': 76}}
            else:
                self.send_error(404)
                return
            if self.path == '/v1/chat/completions' and payload.get('stream') is True:
                delta = {'role': 'assistant'}
                if message.get('content') is not None:
                    delta['content'] = message['content']
                if message.get('tool_calls'):
                    delta['tool_calls'] = [{**tool, 'index': index}
                                           for index, tool in enumerate(message['tool_calls'])]
                def event(choices, usage=None):
                    value = {'id': response['id'], 'model': MODEL, 'choices': choices}
                    if usage is not None:
                        value['usage'] = usage
                    return ('data: ' + json.dumps(value, ensure_ascii=False) + '\n\n').encode()
                blob = event([{'index': 0, 'delta': delta, 'finish_reason': None}])
                blob += event([{'index': 0, 'delta': {}, 'finish_reason': reason}])
                blob += event([], response['usage'])
                if not controls['omit_done']:
                    blob += b'data: [DONE]\n\n'
                content_type = 'text/event-stream'
            else:
                blob = json.dumps(response, ensure_ascii=False).encode()
                content_type = 'application/json'
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests, controls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class Environment:
    def __init__(self, root, url, requests, controls):
        self.root, self.url, self.requests, self.controls = root, url, requests, controls
        self.source = root / 'synthetic.docx'
        synthetic_docx(self.source, count=1, text='桥梁名称：合成桥；总体评分95分。原生公式字形仍然未知。')
        with zipfile.ZipFile(self.source) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        png = io.BytesIO()
        Image.new('RGB', (8, 8), (127, 127, 127)).save(png, format='PNG')
        files['word/media/cache.png'] = png.getvalue()
        # A distant ordinary paragraph lets reverse review genuinely extend reads.
        tail = ''.join(f'<w:p><w:r><w:t>合成附记{i}：只记录既有检查结果。</w:t></w:r></w:p>' for i in range(4))
        files['word/document.xml'] = files['word/document.xml'].replace(b'</w:body>', tail.encode() + b'</w:body>')
        with zipfile.ZipFile(self.source, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, blob in files.items():
                archive.writestr(name, blob)
        self.manifest = build_manifest(self.source)
        self.settings = workflow.DocumentSettings(max_model_calls=30, max_sessions=30,
            max_review_attempts=4, report_timeout=60, temperature=1.0, enable_thinking=True)
        self.wf = self.make()

    def make(self, runner=None):
        client = LocalModelClient(self.url, MODEL, timeout=30, temperature=1.0, enable_thinking=True)
        if self.controls['sse']:
            from bridge_evidence.contest.reliable_generation import StreamPolicy
            client.enable_reliable_generation(StreamPolicy(wall_seconds=10, read_idle_seconds=2),
                                              deadline=time.monotonic() + 60)
        return install_native_review_driver(workflow.DocumentWorkflow(deepcopy(self.manifest), client, {'model': MODEL}, self.root / 'work',
            settings=self.settings, source_path=self.source, vision_backend=SyntheticVision(),
            react_runner=runner or (lambda *_: (_ for _ in ()).throw(AssertionError('No real SDK in fixture')))),
            synthetic_transport=False)

    def seed(self, *, read_all=True, images=True):
        wf = self.wf
        refs = wf.source_objects.catalog['occurrences'][0]['context_refs']
        wanted = {r['evidence_id'] for r in refs}
        for unit in self.manifest['units']:
            if read_all or unit['id'] in wanted:
                assert wf.tools.read_source(unit['id'], 0, len(unit['text']))['ok']
        if images:
            for image in self.manifest['images']:
                assert wf.tools.read_image(image['id'], '观察合成缓存图', '{}', [])['ok']
        unit = self.manifest['units'][0]
        ref = {'evidence_id': unit['id'], 'start': 0, 'end': len(unit['text'])}
        wf.workspace.apply_update({'base_version': wf.workspace.snapshot()['version'],
            'facts': [{'id': 'Fbridge', 'value': '合成桥', 'status': 'explicit', 'source_refs': [ref]},
                      {'id': 'Fscore', 'value': 95, 'status': 'explicit', 'source_refs': [ref]}],
            'draft': {'基本信息': {'桥梁名称': '合成桥', '总体评分': 95}, '病害明细': [], '维修建议': []},
            'draft_fact_ids': ['Fbridge', 'Fscore'],
            'draft_bindings': {'基本信息.桥梁名称': ['Fbridge'], '基本信息.总体评分': ['Fscore']}})
        wf._source_status()

    def arguments(self, suffix='', *, index=0):
        entry = self.wf.source_objects.catalog['entries'][index]
        occurrence = entry['occurrence']
        oid = entry['object_id']
        return {'object_id': oid, 'proposal': {'disposition': 'not_relied_on',
            'rationale': '合成提案：正文独立记载95分，未根据未知公式重算。' + suffix,
            'affected_output_paths': ['基本信息.总体评分'],
            'supporting_refs': deepcopy(occurrence['context_refs']),
            'context_refs': deepcopy(occurrence['context_refs']),
            'source_limitations': ['原生公式仍未解码；缓存图不证明公式语义。']}}

    def origin(self, arguments, *, free_text=False, finish_reason='tool_calls', duplicate=False, output_tokens=None, tool_name='propose_source_object'):
        function = {'id': 'synthetic-tool-call', 'type': 'function', 'function': {
            'name': tool_name, 'arguments': json.dumps(arguments, ensure_ascii=False)}}
        agent = {'finish_reason': finish_reason, 'tool_calls': [] if free_text else [function],
                 'content': json.dumps(arguments) if free_text else None}
        if duplicate:
            agent['tool_calls'].append(deepcopy(function))
        wf = self.wf
        gateway = AuditedModelGateway(wf.client, deadline=wf.deadline,
            max_calls=10, max_output_tokens=wf.settings.output_tokens, before_request=wf._check_budget,
            forward_lock=wf._model_forward_lock, on_dispatch=wf._on_dispatch,
            on_prepared_request=wf._on_model_request, on_response=wf._on_model_response)
        gateway.receipts = wf.gateway_receipts
        gateway.on_receipt = wf._on_model_receipt
        gateway.forward({'model': MODEL, 'messages': [{'role': 'user', 'content': json.dumps({'synthetic_agent': agent})}],
                         'max_tokens': output_tokens or wf.settings.output_tokens, 'stream': False,
                         'temperature': wf.settings.temperature,
                         'chat_template_kwargs': {'enable_thinking': wf.settings.enable_thinking}})
        return wf.gateway_receipts[-1]['call_id']

    def propose(self, arguments=None, *, handler=None):
        arguments = arguments or self.arguments()
        self.origin(arguments)
        call = handler or self.wf._source_handler(self.wf.source_objects.propose_source_object)
        result = call(**arguments)
        assert result['ok'], result
        return result

    def propose_all(self, suffix='', *, handler=None):
        return [self.propose(self.arguments(suffix, index=index), handler=handler)
                for index in range(len(self.wf.source_objects.catalog['entries']))]

    def review_objects(self):
        wf = self.wf
        wf._source_status()
        return SourceObjectReview(wf.source_objects.ledger, model=MODEL, max_model_len=65536,
            output_tokens=wf.settings.output_tokens, history_reserve_tokens=wf.settings.history_reserve_tokens,
            count_tokens=wf._count_chat_tokens, generate_review=wf._generate_source_review).review_pending(
                wf.source_objects.pending_proposal_ids())

    def generations(self):
        return [payload for path, payload in self.requests if path == '/v1/chat/completions']

    def source_calls(self):
        return [row for row in self.wf._model_calls() if row.get('purpose') == 'source_object_review']

    def complete(self):
        def runner(_prompt, handlers, _config):
            if self.wf.workspace.delivery_status()['ready']:
                self.origin({}, tool_name='finish')
                final = handlers['finish']()
                assert final['ready']
                return completed([{'name': 'finish', 'status': 'success', 'result': final}])
            self.seed()
            self.propose_all(handler=handlers['propose_source_object'])
            self.origin({}, tool_name='review_draft')
            queued = handlers['review_draft']()
            assert queued['queued'] and not self.wf.review_sessions
            return completed([{'name': 'review_draft', 'status': 'success', 'result': queued}])
        self.wf.react_runner = runner  # Synthetic SDK outcome, not actual TongAgents evidence.
        return self.wf.run()


@pytest.fixture
def env(tmp_path):
    with local_responder() as (url, requests, controls):
        yield Environment(tmp_path, url, requests, controls)


def test_complete_inventory_never_automatically_accepts_or_decodes_unknown_objects(env):
    env.seed()
    source = env.wf._source_status()
    assert source['catalog_ready'] and source['inventory_count'] == 2 and not source['passed']
    assert source['processing_scope']['unverified_content'] == env.manifest['unverified_content']
    assert source['coverage']['unverified_representation_count'] == 2
    assert not env.wf._finish()['ready'] and not env.generations()


def test_only_current_exact_native_terminal_call_can_authorize_a_proposal(env):
    env.seed()
    wf, arguments = env.wf, env.arguments()
    submit = wf._source_handler(wf.source_objects.propose_source_object)
    assert not submit(**arguments)['ok']  # No model response exists.
    env.origin(arguments, free_text=True, finish_reason='stop')
    assert not submit(**arguments)['ok']  # JSON-looking prose is not a native function call.
    env.origin(arguments, duplicate=True)
    assert not submit(**arguments)['ok']  # Duplicate exact matches are ambiguous.
    first = env.origin(arguments)
    altered = deepcopy(arguments)
    altered['proposal']['rationale'] += 'changed after generation'
    assert not submit(**altered)['ok']
    assert submit(**arguments)['ok']
    second = env.origin(arguments)
    assert first != second and submit(**arguments)['ok']
    latest = list(wf.source_objects.ledger.proposals.values())[-1]
    assert latest['proposal']['origin_call_ids'] == [second]
    # A new current response cannot borrow a matching historical response.
    env.origin(env.arguments('new current arguments'))
    assert not submit(**arguments)['ok']
    with pytest.raises(ValueError, match='truncated'):
        env.origin(arguments, finish_reason='length')
    assert wf._active_agent_call_id is None and not submit(**arguments)['ok']


def test_terminal_sidecar_bytes_and_resume_cannot_be_replaced_by_a_call_id(env):
    env.seed()
    arguments = env.arguments()
    cid = env.origin(arguments)
    path = env.wf.directory / 'model_responses' / (json_digest({'call_id': cid}) + '.json')
    record = json.loads(path.read_text())
    record['response']['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{}'
    path.write_text(json.dumps(record))
    submit = env.wf._source_handler(env.wf.source_objects.propose_source_object)
    assert not submit(**arguments)['ok'] and not env.wf.source_objects.ledger.proposals


def test_rejected_new_agent_preflight_cannot_leave_old_proposal_authorization(env):
    env.seed()
    arguments = env.arguments()
    env.origin(arguments)
    before = len(env.generations())
    with pytest.raises(ValueError, match='output budget'):
        env.origin(env.arguments('invalid new request'), output_tokens=env.wf.settings.output_tokens + 1)
    assert len(env.generations()) == before
    assert env.wf._active_agent_call_id is None
    submit = env.wf._source_handler(env.wf.source_objects.propose_source_object)
    assert not submit(**arguments)['ok'] and not env.wf.source_objects.ledger.proposals


def test_shared_gateway_review_finish_and_completed_resume_keep_unknown_formula(env):
    result = env.complete()
    assert result['receipt']['status'] == 'completed'
    assert len(env.source_calls()) == 1
    source = result['receipt']['source_objects']
    assert source['passed'] and not source['processing_scope']['all_source_semantics_verified']
    assert source['processing_scope']['unverified_content'] == env.manifest['unverified_content']
    row = source['coverage']['objects'][0]
    assert row['representation_state'] == 'unread' and row['source_limitations']
    source_call = env.source_calls()[0]
    bundle = env.wf._terminal_model_bundle(source_call['call_id'])
    assert {'request_workspace_version', 'request_semantic_state_sha256'} <= set(bundle)
    narrow = env.wf.source_objects.receipt_lookup(source_call['call_id'])
    assert set(narrow) == {'audit', 'request', 'response'}
    assert narrow == {key: bundle[key] for key in narrow}
    assignment = json.loads(bundle['request']['messages'][-1]['content'])
    assert bundle['audit']['request_kind'] == 'generation'
    assert assignment['purpose'] == 'source_object_disposition_review'
    assert assignment['draft'] == env.wf.workspace.snapshot()['draft']
    assert assignment['proposals'][0]['source']
    assert source_call['call_id'] not in assignment['proposals'][0]['proposal']['origin_call_ids']
    finish = result['receipt']['accepted_finish']
    assert finish['source_object_attestation_sha256'] == source['attestation_sha256']
    assert finish['source_object_processing_scope_sha256'] == json_digest(source['processing_scope'])
    assert env.wf.workspace.snapshot()['review']['source_object_processing_scope_sha256'] == finish['source_object_processing_scope_sha256']
    count = len(env.generations())
    env.wf = env.make()
    assert env.wf._active_agent_call_id is None
    restored = env.wf.run()
    assert restored['data'] == result['data'] and len(env.generations()) == count
    # Resume replays real review sidecars; deleting one closes the gate.
    response_path = env.wf.directory / 'model_responses' / (json_digest({'call_id': source_call['call_id']}) + '.json')
    response_path.unlink()
    with pytest.raises(ValueError):
        env.make()
    assert len(env.generations()) == count


def test_source_review_shares_cumulative_budget_and_deadline_and_rejects_ack_only(env):
    env.seed()
    env.propose()
    env.controls['source_ack_only'] = True
    with pytest.raises(SourceObjectReviewError, match='REVIEW_RECEIPT_REJECTED'):
        env.review_objects()
    assert len(env.source_calls()) == 1 and not env.wf._source_status()['passed']
    env.controls['source_ack_only'] = False
    # Existing actual Agent + review calls consume the same request allowance.
    from dataclasses import replace
    env.wf.settings = replace(env.wf.settings, max_model_calls=2)
    before = len(env.generations())
    with pytest.raises(SourceObjectReviewError, match='REVIEW_GENERATION_FAILED'):
        env.review_objects()
    assert len(env.generations()) == before
    env.wf.settings = replace(env.wf.settings, max_model_calls=30)
    env.wf.deadline = time.monotonic() - 1
    with pytest.raises(SourceObjectReviewError, match='REVIEW_TOKENIZATION_FAILED'):
        env.review_objects()
    assert len(env.generations()) == before


def test_reverse_new_reads_invalidate_proposals_and_require_fresh_model_origin(env):
    env.seed(read_all=False)
    assert not env.wf.workspace.snapshot()['reading_progress']['text_coverage_complete']
    env.propose()
    old_scope = env.wf._source_status()['scope_sha256']
    env.wf._perform_review_draft()
    current = env.wf._source_status()
    assert current['scope_sha256'] != old_scope and not current['passed']
    assert env.wf.workspace.snapshot()['reading_progress']['text_coverage_complete']
    assert current['coverage']['objects'][0]['proposal_id'] is None
    assert not env.source_calls() and not env.wf.workspace.snapshot()['review']['passed']
    env.propose_all('after complete reverse reading')
    env.wf._perform_review_draft()
    assert len(env.source_calls()) == 1 and env.wf._finish()['ready']


def test_new_source_attestation_invalidates_review_and_finish_without_draft_change(env):
    env.complete()
    prior = env.wf.workspace.snapshot()
    previous_finish = deepcopy(env.wf.last_finish)
    env.propose(env.arguments('reconsider task reliance'))
    status = env.wf.workspace.delivery_status()
    assert status['draft_sha256'] == prior['draft_sha256'] and status['workspace_version'] == prior['version']
    assert 'source_object_review_stale' in status['blockers']
    assert not env.wf._finish_is_current() and not env.wf._finish()['ready']
    env.wf._perform_review_draft()
    assert env.wf.workspace.delivery_status()['ready'] and not env.wf._finish_is_current()
    assert env.wf._finish()['ready'] and env.wf.last_finish != previous_finish


def test_accepted_nonreliance_does_not_bypass_normal_image_gate(env):
    env.seed(images=False)
    env.propose_all()
    env.review_objects()
    source = env.wf._source_status()
    assert source['passed'] and not source['processing_scope']['all_source_semantics_verified']
    status = env.wf.workspace.delivery_status()
    assert 'unverified_images' in status['blockers']
    assert status['reading_progress']['unverified_image_ids']
    assert status['reading_progress']['unverified_content'] == env.manifest['unverified_content']
    assert not env.wf._finish()['ready']


def test_complete_sse_source_chain_retains_bound_done_and_unknown_formula(env):
    env.controls['sse'] = True
    env.wf = env.make()
    result = env.complete()
    source = result['receipt']['source_objects']
    assert source['passed'] and not source['processing_scope']['all_source_semantics_verified']
    assert all(row['representation_state'] == 'unread' for row in source['coverage']['objects'])
    assert all(request['stream'] is True for request in env.generations())
    for call in env.source_calls():
        bundle = env.wf.source_objects.receipt_lookup(call['call_id'])
        proof = bundle['audit']['stream_transport_receipt']
        assert set(bundle) == {'audit', 'request', 'response'}
        assert proof['done_received'] is True and proof['usage_known'] is True
        assert proof['raw_sse_sha256'] == call['raw_sse_sha256']
        assert proof['input_sha256'] == json_digest(bundle['request'])
        assert proof['response_sha256'] == json_digest(bundle['response'])
    before = len(env.generations())
    env.wf = env.make()
    assert env.wf.run()['data'] == result['data']
    assert len(env.generations()) == before


@pytest.mark.parametrize('stage', ['proposal', 'review'])
def test_sse_without_done_cannot_authorize_source_or_resume_generation(env, stage):
    from bridge_evidence.contest.reliable_generation import ReliableGenerationError
    env.controls['sse'] = True
    env.wf = env.make()
    env.seed()
    if stage == 'review':
        env.propose_all()
    before = len(env.generations())
    env.controls['omit_done'] = True
    if stage == 'proposal':
        with pytest.raises(ReliableGenerationError):
            env.propose()
        assert not env.wf.source_objects.ledger.proposals
    else:
        with pytest.raises(SourceObjectReviewError, match='REVIEW_GENERATION_FAILED'):
            env.review_objects()
    assert len(env.generations()) == before + 1
    call = env.wf.client.calls[-1]
    assert call['http_status'] == 200 and call['done_received'] is False
    assert call['finish_reason'] in {'stop', 'tool_calls'}  # A finish without DONE is insufficient.
    assert env.wf.model_dispatches[-1]['status'] == 'outcome_unknown'
    assert not env.wf._source_status()['passed']
    env.wf = env.make()
    with pytest.raises(RuntimeError, match='unresolved|unknown|unsettled'):
        env.origin(env.arguments())
    assert len(env.generations()) == before + 1


@pytest.mark.parametrize(('field', 'invalid'), [
    ('done_received', False), ('usage_known', False), ('raw_sse_sha256', None),
    ('response_sha256', '0' * 64), ('request_id', 'other-dispatch'),
])
def test_sse_source_transport_proof_cannot_be_replaced_by_stream_true(env, field, invalid):
    env.controls['sse'] = True
    env.wf = env.make()
    env.seed()
    arguments = env.arguments()
    call_id = env.origin(arguments)
    actual = next(call for call in env.wf.client.calls if call.get('call_id') == call_id)
    actual[field] = invalid
    result = env.wf._source_handler(env.wf.source_objects.propose_source_object)(**arguments)
    assert not result['ok'] and not env.wf.source_objects.ledger.proposals
