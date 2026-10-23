"""Synthetic local contracts only. No real report/model/SDK/HTTP was used."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from unittest.mock import patch

import pytest
from docx import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bridge_evidence.contest import document_source_objects as mod
from bridge_evidence.contest.evidence_manifest import build_manifest
from bridge_evidence.contest.run_state import json_digest
from bridge_evidence.contest.source_object_dispositions import ISSUES
from bridge_evidence.contest.tong_react_adapter import wrap_tool_response

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
NS = (f'xmlns:w="{W}" xmlns:r="{R}" xmlns:v="urn:schemas-microsoft-com:vml" '
      'xmlns:o="urn:schemas-microsoft-com:office:office"')
OBJECT = ('<w:object><v:shape id="shape1"><v:imagedata r:id="rImg"/></v:shape>'
          '<o:OLEObject Type="Embed" ProgID="Equation.DSMT4" ShapeID="shape1" r:id="rOle"/></w:object>')


def synthetic_docx(path, count=2, text='合成说明：原报告文字记载评分95分；公式字形未恢复。'):
    Document().save(path)
    with zipfile.ZipFile(path) as z:
        files = {item.filename: z.read(item) for item in z.infolist()}
    body = ''.join('<w:p><w:r><w:t>' + text + '</w:t>' + OBJECT + '</w:r></w:p>' for _ in range(count))
    files['word/document.xml'] = f'<w:document {NS}><w:body>{body}</w:body></w:document>'.encode()
    files['word/_rels/document.xml.rels'] = (f'<Relationships xmlns="{REL}"><Relationship Id="rOle" '
        f'Type="{R}/oleObject" Target="embeddings/native.bin"/><Relationship Id="rImg" '
        f'Type="{R}/image" Target="media/cache.png"/></Relationships>').encode()
    types = files['[Content_Types].xml'].decode().replace('</Types>',
        '<Default Extension="bin" ContentType="application/vnd.openxmlformats-officedocument.oleObject"/>'
        '<Default Extension="png" ContentType="image/png"/></Types>')
    files['[Content_Types].xml'] = types.encode()
    files['word/embeddings/native.bin'] = b'SYNTHETIC OLE: no semantics decoded'
    files['word/media/cache.png'] = b'SYNTHETIC separate image cache bytes'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return build_manifest(path)


@pytest.fixture
def env(tmp_path):
    class Env:
        model = 'SYNTHETIC-OWN-MODEL-NO-HTTP'

        def __init__(self):
            self.source = tmp_path / 'synthetic.docx'
            self.manifest = synthetic_docx(self.source)
            self.receipts = {}
            self.origin_calls = []
            self.work = tmp_path / 'controller'
            self.identity = {'model': self.model, 'workflow': 'synthetic-v7', 'source': json_digest(self.manifest)}
            self.count_tokens = len  # exact fake wire tokenizer, intentionally NOT BPE evidence.
            self.controller = self.new()
            self.ids = [e['object_id'] for e in self.controller.catalog['entries']]
            draft = {'基本信息': {'桥梁名称': '合成桥', '总体评分': 95}, '维修建议': [], '病害明细': []}
            self.state = {'version': 1, 'facts_version': 1,
                'facts': {'F1': {'id': 'F1', 'valid': True, 'value': 95}}, 'relations': {}, 'questions': {},
                'draft': draft, 'draft_sha256': json_digest(draft), 'draft_valid': True,
                'draft_fact_ids': ['F1'], 'draft_bindings': {'基本信息.总体评分': ['F1'], '基本信息.桥梁名称': ['F1']},
                'reads': {u['id']: [[0, len(u['text'])]] for u in self.manifest['units'] if u['text']},
                'image_reads': [], 'image_observations': {}, 'image_observation_index_sha256': json_digest({}),
                'exclusions': [], 'review': None, 'history': []}
            self.controller.bind_state(self.state)

        def new(self, **kwargs):
            return mod.DocumentSourceObjects(self.manifest, self.source, self.work, self.identity,
                self.receipts.__getitem__, self.count_tokens, kwargs.pop('max_tool_tokens', 100000),
                generate_origin_call_id=kwargs.pop('generate_origin_call_id', self.origin), **kwargs)

        def bundle(self, request, response, call_id, kind='agent_generation'):
            return {'request': request, 'response': response, 'audit': {
                'call_id': call_id, 'request_id': 'synthetic-dispatch-' + call_id, 'status': 'success',
                'request_kind': kind, 'generation_attempted': True, 'response_validated': True,
                'http_status': 200, 'finish_reason': 'stop', 'endpoint': '/v1/chat/completions',
                'model': self.model, 'request_sha256': json_digest(request), 'response_sha256': json_digest(response)}}

        def origin(self, name, arguments):
            # Represents the TRUSTED root callback's exact actual-SDK-call matching;
            # it is a fake receipt only, never proof of a real Agent execution.
            self.origin_calls.append((name, deepcopy(arguments)))
            call_id = 'synthetic-origin-' + json_digest(arguments)[:20]
            request = {'model': self.model, 'stream': False, 'messages': [{'role': 'user', 'content': 'Synthetic task'}]}
            response = {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'synthetic-tool', 'type': 'function', 'function': {'name': name,
                                'arguments': json.dumps(arguments, ensure_ascii=False)}}]}}]}
            self.receipts[call_id] = self.bundle(request, response, call_id)
            return call_id

        def proposal(self, index=0, mode='not_relied_on'):
            refs = deepcopy(self.controller.catalog['occurrences'][index]['context_refs'])
            return {'disposition': mode, 'rationale': 'Synthetic proposed scope; model judgment required.',
                'affected_output_paths': ['基本信息.总体评分'] if mode == 'not_relied_on' else [],
                'supporting_refs': deepcopy(refs) if mode == 'not_relied_on' else [], 'context_refs': refs,
                'source_limitations': ['The original formula remains unverified; no score is recomputed.']}

        def propose(self, index=0, **changes):
            value = self.proposal(index)
            value.update(changes)
            return self.controller.propose_source_object(self.ids[index], value)

        def review(self, ids=None, verdict='accepted'):
            ids = ids or self.controller.pending_proposal_ids()
            payload = self.controller.ledger.review_payload(ids)
            decision = {'assignment_id': payload['assignment_id'], 'decisions': [
                {'proposal_id': row['proposal_id'], 'object_id': row['object_id'], 'verdict': verdict,
                 'explanation': 'Synthetic independent result, not actual model evidence.', **{k: [] for k in ISSUES}}
                for row in payload['proposals']]}
            request = {'model': self.model, 'stream': False, 'messages': [
                {'role': 'system', 'content': 'Synthetic independent object reviewer'},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]}
            response = {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant',
                         'content': json.dumps(decision, ensure_ascii=False)}}]}
            call_id = 'synthetic-review-' + json_digest(payload)[:20]
            self.receipts[call_id] = self.bundle(request, response, call_id, 'generation')
            self.controller.ledger.record_review(ids, call_id)
            return call_id

        def accept_all(self):
            for i in range(len(self.ids)):
                assert self.propose(i)['ok'] is True
            self.review()
            assert self.controller.coverage()['passed']

    return Env()


def test_real_synthetic_docx_catalog_retains_complete_separate_occurrences(env):
    original = deepcopy(env.manifest)
    result = env.controller.coverage()
    assert result['catalog_ready'] and result['inventory_count'] == 4
    assert env.ids[0] != env.ids[1]
    assert not result['passed']
    assert result['processing_scope']['unverified_content'] == env.manifest['unverified_content']
    assert env.manifest == original
    assert env.controller.catalog['occurrences'][0]['native_sha256'] == env.controller.catalog['occurrences'][1]['native_sha256']


def test_one_accepted_occurrence_does_not_unlock_same_bytes_other_position(env):
    assert env.propose()['ok']
    env.review()
    result = env.controller.coverage()
    assert not result['passed'] and result['coverage']['accepted_count'] == 1
    assert result['coverage']['objects'][1]['review_status'] == 'unclassified'


def test_all_accepted_retains_unread_representation_original_unknown_and_scope(env):
    original = deepcopy(env.manifest['unverified_content'])
    env.accept_all()
    scope = env.controller.coverage()['processing_scope']
    assert not scope['all_source_semantics_verified']
    assert scope['unverified_content'] == original
    assert all(row['representation_state'] == 'unread' and row['source_limitations'] for row in scope['objects'])
    assert not env.controller.pending_proposal_ids()


def test_outside_task_requires_real_independent_acceptance_without_invented_output_paths(env):
    for i in range(len(env.ids)):
        result = env.controller.propose_source_object(env.ids[i], env.proposal(i, 'outside_task'))
        assert result['ok'] and result['accepted'] is False
    assert not env.controller.coverage()['passed']
    env.review()
    assert env.controller.coverage()['passed']


@pytest.mark.parametrize('verdict', ['rejected', 'unresolved'])
def test_unaccepted_review_never_unlocks(env, verdict):
    env.propose(0); env.propose(1)
    env.review(verdict=verdict)
    assert not env.controller.coverage()['passed']
    assert not env.controller.pending_proposal_ids()  # must revise, not overwrite a finished decision.


def test_origin_is_host_owned_exact_arguments_and_duplicates_idempotent(env):
    proposal = env.proposal()
    assert env.controller.propose_source_object(env.ids[0], proposal)['ok']
    history = len(env.controller.ledger.history())
    assert env.controller.propose_source_object(env.ids[0], proposal)['ok']
    assert len(env.controller.ledger.history()) == history
    assert env.origin_calls[0] == ('propose_source_object', {'object_id': env.ids[0], 'proposal': proposal})
    stored = next(iter(env.controller.ledger.proposals.values()))['proposal']
    assert len(stored['origin_call_ids']) == 1 and stored['origin_call_ids'][0].startswith('synthetic-origin-')
    assert 'origin_call_ids' not in proposal
    proposal['origin_call_ids'] = ['agent-fabricated']
    result = env.controller.propose_source_object(env.ids[1], proposal)
    assert not result['ok'] and len(env.controller.ledger.proposals) == 1


@pytest.mark.parametrize('change', ['missing', 'failed', 'model', 'kind', 'request_hash', 'response_hash'])
def test_origin_receipt_failure_never_appends(env, change):
    def origin(name, arguments):
        cid = env.origin(name, arguments)
        bundle = env.receipts[cid]
        if change == 'missing': del env.receipts[cid]
        elif change == 'failed': bundle['audit']['status'] = 'failed'
        elif change == 'model': bundle['audit']['model'] = 'other-model'
        elif change == 'kind': bundle['audit']['request_kind'] = 'generation'
        elif change == 'request_hash': bundle['request']['extra'] = 'tampered'
        elif change == 'response_hash': bundle['response']['extra'] = 'tampered'
        return cid
    env.controller.generate_origin_call_id = origin
    assert not env.propose()['ok']
    assert not env.controller.ledger.proposals


def test_origin_callback_error_fails_closed_without_leaking_underlying_payload(env):
    def bad(*_):
        raise RuntimeError('SENSITIVE-RAW-MODEL-STRING')
    env.controller.generate_origin_call_id = bad
    result = env.propose()
    assert not result['ok'] and 'SENSITIVE' not in json.dumps(result)
    assert not env.controller.ledger.proposals


def test_unread_full_context_is_not_replaced_by_exclusions_or_reading_progress(env):
    state = deepcopy(env.state)
    state['reads'] = {}
    state['reading_progress'] = {'read_chars': 99999, 'unread_ranges': []}
    state['exclusions'] = [{'reason': 'synthetic claim that everything is irrelevant'}]
    env.controller.bind_state(state)
    assert not env.propose()['ok']
    assert not env.controller.ledger.proposals


def test_split_actual_reads_merge_but_gap_blocks_context(env):
    state = deepcopy(env.state)
    for eid, ranges in state['reads'].items():
        n = ranges[0][1]
        state['reads'][eid] = [[3, n], [0, 3]]
    scope = env.controller.bind_state(state)['scope_sha256']
    assert scope == env.controller.bind_state(env.state)['scope_sha256']
    for eid, ranges in state['reads'].items():
        n = max(r[1] for r in ranges)
        state['reads'][eid] = [[0, 2], [3, n]]
    env.controller.bind_state(state)
    assert not env.propose()['ok']


@pytest.mark.parametrize('field', ['facts', 'relations', 'questions', 'draft_bindings', 'exclusions', 'image_observations'])
def test_semantic_context_changes_invalidate_and_reversion_does_not_resurrect(env, field):
    env.accept_all()
    before = env.controller.coverage()['scope_sha256']
    state = deepcopy(env.state)
    if field == 'exclusions': state[field].append({'reason': 'changed'})
    else: state[field]['changed'] = {'valid': False, 'value': 'changed'}
    if field == 'image_observations': state['image_observation_index_sha256'] = json_digest(state[field])
    result = env.controller.bind_state(state)
    assert not result['passed'] and result['scope_sha256'] != before
    assert not env.controller.bind_state(env.state)['passed']
    assert len(env.controller.ledger.reviews) == len(env.ids)


def test_draft_value_change_and_read_extension_invalidate(env):
    env.accept_all()
    state = deepcopy(env.state)
    state['draft']['基本信息']['总体评分'] = 80
    state['draft_sha256'] = json_digest(state['draft'])
    assert not env.controller.bind_state(state)['passed']
    state = deepcopy(env.state)
    state['reads'][next(iter(state['reads']))] = [[0, 1]]
    assert not env.controller.bind_state(state)['passed']


def test_review_history_workspace_version_only_does_not_invalidate(env):
    env.accept_all()
    before = env.controller.coverage()
    state = deepcopy(env.state)
    state['version'] = 900
    state['review'] = {'passed': True, 'model_call_ids': ['synthetic-elsewhere']}
    state['history'] = [{'event': 'review'}]
    assert env.controller.bind_state(state) == before


def test_invalid_snapshot_closes_old_passed_gate(env):
    env.accept_all()
    state = deepcopy(env.state)
    state['draft']['基本信息']['总体评分'] = 71  # intentionally stale draft hash
    with pytest.raises(ValueError, match='draft hash'):
        env.controller.bind_state(state)
    assert not env.controller.coverage()['passed']


def test_resume_rechecks_accepted_and_origin_receipts(env):
    env.accept_all()
    expected = env.controller.coverage()
    other = env.new()
    assert other.bind_state(env.state) == expected
    origin_id = next(iter(env.controller.ledger.proposals.values()))['proposal']['origin_call_ids'][0]
    env.receipts[origin_id]['audit']['status'] = 'failed'
    with pytest.raises(ValueError, match='Origin'):
        env.new()


def test_resume_changed_workflow_and_catalog_file_tampering_rejected(env):
    env.identity['workflow'] = 'different'
    with pytest.raises(ValueError, match='identity differs'):
        env.new()
    env.identity['workflow'] = 'synthetic-v7'
    path = env.work / 'source_object_catalog.json'
    value = json.loads(path.read_text())
    value['catalog']['entries'][0]['blocking'] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='identity differs'):
        env.new()


def test_read_tools_are_bounded_metadata_not_actual_source_delivery(env):
    before = deepcopy(env.state['reads'])
    first = env.controller.read_source_objects(0, 1)
    assert first['ok'] and first['next_cursor'] == 1 and first['has_more']
    assert first['objects'][0]['locator'] == env.controller.catalog['occurrences'][0]['locator']
    assert first['objects'][0]['kind'] == 'ole'
    second = env.controller.read_source_objects(1, 1)
    assert second['objects'][0]['object_id'] == env.ids[1] and second['has_more']
    final = env.controller.read_source_objects(len(env.ids) - 1, 1)
    assert final['objects'][0]['object_id'] == env.ids[-1] and not final['has_more']
    detail = env.controller.read_source_object(env.ids[0])
    assert detail['ok'] and detail['entry']['inventory_record'] == env.manifest['unverified_content'][0]
    assert env.state['reads'] == before and not env.controller.ledger.proposals
    assert not env.controller.read_source_objects(True, 1)['ok']
    assert not env.controller.read_source_objects(0, False)['ok']
    assert not env.controller.read_source_object('invented-id')['ok']


def test_native_wrapper_budget_blocks_full_detail_without_truncating_or_granting(env):
    c = env.controller
    c.native_response_budget = True
    detail = c.read_source_object(env.ids[0])
    raw_len = len(mod._compact(detail))
    wrapped_len = len(mod._compact(wrap_tool_response('read_source_object', detail)))
    assert wrapped_len > raw_len
    c.max_tool_tokens = raw_len
    assert c.read_source_object(env.ids[0])['status'] == 'budget_exceeded'
    c.max_tool_tokens = wrapped_len
    assert c.read_source_object(env.ids[0]) == detail
    assert not c.ledger.proposals


def test_write_ack_native_budget_preflight_prevents_write_and_origin_callback(env):
    c = env.controller
    c.native_response_budget = True
    response = {'ok': True, 'object_id': env.ids[0], 'scope_sha256': c.ledger.scope_sha256,
                'status': 'proposed', 'accepted': False,
                'next_action': 'Independent review is required. Read the object to inspect the stored proposal.'}
    raw_len = len(mod._compact(response))
    c.max_tool_tokens = raw_len
    assert env.propose()['status'] == 'budget_exceeded'
    assert not c.ledger.proposals and not env.origin_calls
    c.max_tool_tokens = len(mod._compact(wrap_tool_response('propose_source_object', response)))
    assert env.propose() == response
    assert len(c.ledger.proposals) == 1


def test_native_pagination_pages_fit_and_cover_every_occurrence(env):
    c = env.controller
    c.native_response_budget = True
    page = c.read_source_objects(0, 1)
    final_page = c.read_source_objects(1, 1)
    c.max_tool_tokens = max(len(mod._compact(wrap_tool_response('read_source_objects', p))) for p in (page, final_page))
    bounded = c.read_source_objects(0, 2)
    assert bounded['ok'] and len(bounded['objects']) == 1 and bounded['has_more']
    assert c.read_source_objects(bounded['next_cursor'], 2)['objects'][0]['object_id'] == env.ids[1]


def test_incomplete_native_inventory_blocks_ledger_and_proposals(tmp_path):
    source = tmp_path / 'empty-context.docx'
    manifest = synthetic_docx(source, count=1, text='')
    c = mod.DocumentSourceObjects(manifest, source, tmp_path / 'work', {'model': 'synthetic'},
        lambda _: None, len, 100000, generate_origin_call_id=lambda *_: 'not-real')
    assert not c.catalog['ledger_ready'] and c.ledger is None
    assert 'source_object_inventory_incomplete' in c.coverage()['blockers']
    assert not c.propose_source_object(c.catalog['entries'][0]['object_id'], {
        'disposition': 'outside_task', 'rationale': 'guessed', 'affected_output_paths': [],
        'supporting_refs': [], 'context_refs': [], 'source_limitations': ['unknown']})['ok']


def test_falsely_complete_catalog_and_missing_manifest_occurrences_rejected(env):
    bad = env.controller.catalog
    bad['occurrences'].pop()
    bad['catalog_sha256'] = json_digest({k: v for k, v in bad.items() if k != 'catalog_sha256'})
    with patch.object(mod, 'build_source_object_catalog', return_value=bad), pytest.raises(ValueError, match='complete ledger'):
        env.new()


@pytest.mark.parametrize('mode', ['used_as_evidence', 'ignore', 'semantics_verified'])
def test_agent_cannot_upgrade_object_representation_or_ignore_whole_report(env, mode):
    assert not env.propose(disposition=mode)['ok']
    assert not env.controller.ledger.proposals


def test_zero_length_workspace_reads_prove_no_text_and_do_not_break_binding(env):
    state = deepcopy(env.state)
    for ranges in state['reads'].values():
        ranges.append([0, 0])
    original = env.controller.coverage()['scope_sha256']
    assert env.controller.bind_state(state)['scope_sha256'] == original
    state['reads'] = {eid: [[0, 0]] for eid in state['reads']}
    env.controller.bind_state(state)
    assert not env.propose()['ok']


def test_draft_object_key_order_does_not_invalidate_on_atomic_json_restore(env):
    env.accept_all()
    before = env.controller.coverage()
    state = deepcopy(env.state)
    state['draft']['基本信息'] = dict(reversed(list(state['draft']['基本信息'].items())))
    assert env.controller.bind_state(state) == before
