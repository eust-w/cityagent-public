"""Candidate controller for native-object task dispositions; no model/network I/O.

The source catalog verifies byte inventory, not meaning. A trusted caller supplies
workspace snapshots and resolves original model receipts. Only independent model
review in SourceObjectDispositions can accept relevance/nonreliance. Normal image
gates and text reading remain the existing workflow's responsibility. Single worker.

The origin callback MUST find the unique terminal Agent generation whose actually
returned SDK tool call has this exact name/arguments. The controller checks its
terminal receipt and byte hashes; it intentionally does not guess SDK content-JSON
syntax. Do not implement that callback by returning the last model call blindly.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Callable

from .document_delivery_validation import validate_delivery_data
from .run_state import atomic_json, json_digest
from .source_object_catalog import build_source_object_catalog
from .source_object_dispositions import SourceObjectDispositions, _json as strict_json, validate_source_transport
from .tong_react_adapter import wrap_tool_response

VERSION = 'document-source-objects-candidate-v1'
TOOLS = ('read_source_objects', 'read_source_object', 'propose_source_object')
PROPOSAL_FIELDS = {'disposition', 'rationale', 'affected_output_paths', 'supporting_refs',
                   'context_refs', 'source_limitations'}
MODES = {'not_relied_on', 'outside_task', 'unresolved_relevant', 'unclassified'}
TASK = {'name': 'three-table-faithful-extraction', 'version': VERSION,
        'sections': ['基本信息', '维修建议', '病害明细'],
        'scope': 'Judge each original occurrence against the complete frozen three-table draft; '
                 'unverified objects cannot supply evidence or be treated as understood.',
        'unknown_behavior': 'Retain source limitations. Unknown calculations or unique relevant information remain blocking.'}
# Deliberately exclude workspace version, history and review attestations: review
# recording itself must not invalidate its frozen semantic inputs. Reads are bound
# separately and any newly delivered interval invalidates older dispositions.
SEMANTIC_FIELDS = ('facts', 'relations', 'questions', 'facts_version', 'draft_fact_ids',
                   'draft_bindings', 'draft_valid', 'draft_sha256', 'exclusions',
                   'image_reads', 'image_observations', 'image_observation_index_sha256')


def _compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class SourceObjectToolError(ValueError):
    pass


class DocumentSourceObjects:
    def __init__(self, manifest: dict, source_path: Path, directory: Path,
                 workflow_identity: dict, receipt_lookup: Callable[[str], dict],
                 count_tokens: Callable[[str], int], max_tool_tokens: int = 4096, *,
                 generate_origin_call_id: Callable[[str, dict], str],
                 native_response_budget: bool = False):
        if type(max_tool_tokens) is not int or max_tool_tokens <= 0:
            raise ValueError('max_tool_tokens must be a positive integer')
        if type(native_response_budget) is not bool:
            raise ValueError('native_response_budget must be boolean')
        if not all(callable(f) for f in (receipt_lookup, count_tokens, generate_origin_call_id)):
            raise ValueError('Trusted receipt/origin callbacks and a tokenizer are required')
        self._manifest = strict_json(manifest)
        self.directory = Path(directory)
        self.receipt_lookup = receipt_lookup
        self.generate_origin_call_id = generate_origin_call_id
        self.count_tokens = count_tokens
        self.max_tool_tokens = max_tool_tokens
        self.native_response_budget = native_response_budget
        self._bound = False
        self._draft_ready = False
        self._state_semantics = None
        for name in TOOLS:
            if self._tokens(self._budget_error(), name) > max_tool_tokens:
                raise ValueError('max_tool_tokens cannot fit a minimal source-object budget error')
        self._catalog = build_source_object_catalog(Path(source_path), self._manifest)
        self._check_catalog()
        self.identity = {'controller_version': VERSION, 'workflow_identity': strict_json(workflow_identity),
                         'model': workflow_identity.get('model'),
                         'source_sha256': self._manifest['document_sha256'],
                         'source_catalog_sha256': self._catalog['catalog_sha256']}
        catalog_record = {'identity': self.identity, 'catalog': self._catalog}
        self.directory.mkdir(parents=True, exist_ok=True)
        self._save_or_match(self.directory / 'source_object_catalog.json', catalog_record)
        self.units = {u['id']: u['text'] for u in self._manifest['units']}
        self.ledger = None
        if self._catalog['ledger_ready']:
            self.ledger = SourceObjectDispositions(self.directory, self.identity,
                self._catalog['occurrences'], self.units, receipt_lookup)
            # The core ledger validates independent-review receipts during replay.
            # This controller also revalidates real origin-call receipts on resume.
            for row in self.ledger.proposals.values():
                for call_id in row['proposal']['origin_call_ids']:
                    self._origin_receipt(call_id)

    @staticmethod
    def _save_or_match(path, value):
        if path.exists():
            try:
                existing = strict_json(json.loads(path.read_text(encoding='utf-8')))
            except Exception:
                raise ValueError('Stored source-object catalog is invalid') from None
            if existing != value:
                raise ValueError('Stored source-object catalog or identity differs')
        else:
            atomic_json(path, value)

    def _check_catalog(self):
        c = self._catalog
        body = {k: v for k, v in c.items() if k != 'catalog_sha256'}
        if c.get('catalog_sha256') != json_digest(body):
            raise ValueError('Source-object catalog checksum differs')
        if (c.get('source_sha256') != self._manifest.get('document_sha256')
                or c.get('manifest_sha256') != json_digest(self._manifest)
                or c.get('inventory_sha256') != json_digest(self._manifest.get('unverified_content', []))
                or c.get('inventory_count') != len(self._manifest.get('unverified_content', []))):
            raise ValueError('Source-object catalog does not bind the complete manifest inventory')
        if c.get('ledger_ready') is True and (
                c.get('text_inventory_verified_against_source') is not True
                or c.get('unlisted_xml_occurrences') or c.get('unexpected_inventory_locators')
                or len(c.get('entries', [])) != c['inventory_count']
                or len(c.get('occurrences', [])) != c['inventory_count']
                or any(e.get('blocking') is not False for e in c['entries'])):
            raise ValueError('Source-object catalog falsely claims a complete ledger inventory')
        if type(c.get('ledger_ready')) is not bool:
            raise ValueError('Source-object catalog requires an explicit readiness state')

    @property
    def catalog(self):
        return deepcopy(self._catalog)

    @staticmethod
    def _budget_error():
        return {'ok': False, 'status': 'budget_exceeded', 'error': {
            'type': 'SourceObjectToolBudgetExceeded', 'recoverable': True,
            'message': 'Response cannot fit. Increase the tool budget; no object was truncated or accepted.'}}

    def _tokens(self, value, name):
        wire = wrap_tool_response(name, value) if self.native_response_budget else value
        count = self.count_tokens(_compact(wire))
        if type(count) is not int or count < 0:
            raise ValueError('count_tokens must return a non-negative integer')
        return count

    def _fits(self, value, name):
        return self._tokens(value, name) <= self.max_tool_tokens

    def _bounded(self, value, name):
        return deepcopy(value) if self._fits(value, name) else self._budget_error()

    def _error(self, exc, name):
        return self._bounded({'ok': False, 'status': 'error', 'error': {
            'type': type(exc).__name__, 'message': str(exc), 'recoverable': True}}, name)

    def _read_refs(self, reads):
        if type(reads) is not dict:
            raise ValueError('Workspace reads must be the trusted actual-delivery interval map')
        refs = []
        for eid in sorted(reads):
            intervals = reads[eid]
            if eid not in self.units or type(intervals) is not list:
                raise ValueError('Workspace read ledger contains an unknown unit or invalid intervals')
            merged = []
            for interval in intervals:
                if (type(interval) is not list or len(interval) != 2
                        or any(type(x) is not int for x in interval)
                        or not 0 <= interval[0] <= interval[1] <= len(self.units[eid])):
                    raise ValueError('Workspace read interval is outside the immutable text unit')
            for start, end in sorted(intervals):
                if start == end:  # Existing source tools can deliver empty units; they prove no text.
                    continue
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            refs.extend({'evidence_id': eid, 'start': start, 'end': end} for start, end in merged)
        return refs

    @staticmethod
    def _output_paths(draft):
        if not draft:
            return []
        validate_delivery_data(draft)
        return ([f'基本信息.{key}' for key, value in sorted(draft['基本信息'].items()) if value not in (None, '')]
                + [f'{name}.{i}' for name in ('维修建议', '病害明细') for i in range(len(draft[name]))])

    def bind_state(self, state: dict) -> dict:
        """Bind a trusted workspace snapshot, never Agent-supplied state.

        Call at guard/gate boundaries with an already obtained trusted snapshot;
        this controller never reenters the workspace or invokes model callbacks.
        New reads intentionally change the scope: freeze after required source
        reading, then propose/review. Review/version-only changes are ignored.
        """
        self._bound = False  # Any invalid new snapshot must close the prior gate.
        self._draft_ready = False
        state = strict_json(state)
        required = {'draft', 'reads', *SEMANTIC_FIELDS}
        if type(state) is not dict or required - set(state):
            raise ValueError('Workspace snapshot lacks source-object semantic identity fields')
        if any(type(state[name]) is not dict for name in ('facts', 'relations', 'questions', 'draft_bindings', 'image_observations')):
            raise ValueError('Workspace semantic collections must be objects')
        if (type(state['draft_valid']) is not bool or type(state['facts_version']) is not int
                or state['facts_version'] < 0):
            raise ValueError('Workspace semantic validity/version fields are malformed')
        draft = state['draft']
        if draft is not None and type(draft) is not dict:
            raise ValueError('Workspace draft must be an object or null')
        if state['draft_sha256'] != (json_digest(draft) if draft is not None else None):
            raise ValueError('Workspace draft hash differs from draft')
        if state['image_observation_index_sha256'] != json_digest(state['image_observations']):
            raise ValueError('Workspace image observation index hash differs')
        paths = self._output_paths(draft)
        refs = self._read_refs(state['reads'])
        facts = {key: deepcopy(state[key]) for key in SEMANTIC_FIELDS}
        catalog_sha = json_digest({'draft': draft, 'output_paths': paths, 'semantic_state': facts})
        if self.ledger is not None:
            self.ledger.bind_context(task=TASK, draft=draft or {}, facts=facts,
                                     output_paths=paths, read_refs=refs, catalog_sha256=catalog_sha)
        self._draft_ready = state['draft_valid'] and draft is not None
        self._state_semantics = json_digest({'facts': facts, 'draft': draft, 'read_refs': refs})
        self._bound = True
        return self.coverage()

    def coverage(self):
        """Trusted object gate only. It cannot certify whole-report publication."""
        current = self.ledger.coverage() if self.ledger is not None and self._bound else None
        rows = current['objects'] if current else []
        blockers = []
        if not self._catalog['ledger_ready']:
            blockers.append('source_object_inventory_incomplete')
        if not self._bound:
            blockers.append('source_object_context_unbound')
        elif not self._draft_ready:
            blockers.append('source_object_draft_not_ready')
        if current is not None:
            for row in rows:
                if not row['object_gate_passed'] or row['disposition'] not in {'not_relied_on', 'outside_task'}:
                    blockers.append('source_object_unresolved:' + row['object_id'])
        passed = not blockers and current is not None
        scope = current['scope_sha256'] if current else None
        inputs = [deepcopy(self.ledger.proposals[r['proposal_id']]) for r in rows if r['proposal_id']] if current else []
        occurrences = {e['object_id']: e['occurrence'] for e in self._catalog['entries'] if e.get('object_id')}
        scope_rows = [{**deepcopy(row), 'locator': occurrences[row['object_id']]['locator'],
                       'kind': occurrences[row['object_id']]['kind']} for row in rows]
        processing = {'task': deepcopy(TASK), 'source_catalog_sha256': self._catalog['catalog_sha256'],
                      'scope_sha256': scope, 'object_gate_passed': passed,
                      'all_source_semantics_verified': current['all_source_semantics_verified'] if current else False,
                      'unverified_content': deepcopy(self._manifest.get('unverified_content', [])),
                      'objects': scope_rows,
                      'limitation': 'An accepted disposition limits task reliance; it does not decode or verify the original object. Normal image/text/review/export gates still apply.'}
        attestation = {'catalog_sha256': self._catalog['catalog_sha256'], 'scope_sha256': scope,
                       'objects': rows, 'passed': passed, 'blockers': blockers}
        return {'passed': passed, 'blockers': blockers, 'catalog_ready': self._catalog['ledger_ready'],
                'scope_sha256': scope, 'source_catalog_sha256': self._catalog['catalog_sha256'],
                'semantic_input_sha256': json_digest({'scope': scope, 'proposals': inputs}),
                'attestation_sha256': json_digest(attestation),
                'inventory_count': self._catalog['inventory_count'],
                'coverage': deepcopy(current), 'processing_scope': processing}

    def pending_proposal_ids(self):
        if self.ledger is None or not self._bound:
            return []
        return [row['proposal_id'] for row in self.ledger.coverage()['objects']
                if row['proposal_id'] and row['review_status'] == 'proposed']

    def _origin_receipt(self, call_id):
        if type(call_id) is not str or not call_id.strip():
            raise ValueError('Trusted origin callback did not return a real call ID')
        try:
            bundle = strict_json(self.receipt_lookup(call_id))
            audit, request, response = (bundle[k] for k in ('audit', 'request', 'response'))
        except Exception:
            raise ValueError('Trusted Agent origin receipt is unavailable') from None
        if (any(type(v) is not dict for v in (audit, request, response))
                or audit.get('call_id') != call_id or audit.get('status') != 'success'
                or audit.get('request_kind') != 'agent_generation'
                or audit.get('generation_attempted') is not True or audit.get('response_validated') is not True
                or audit.get('http_status') != 200 or audit.get('finish_reason') not in {'stop', 'tool_calls'}
                or audit.get('endpoint') != '/v1/chat/completions'
                or type(audit.get('request_id')) is not str or not audit['request_id']
                or audit.get('model') != self.identity['model'] or request.get('model') != self.identity['model']
                or audit.get('request_sha256') != json_digest(request)
                or audit.get('response_sha256') != json_digest(response)):
            raise ValueError('Origin must be a terminal successful own-model Agent generation receipt')
        validate_source_transport(audit, request, response)
        return call_id

    def read_source_objects(self, cursor: int = 0, limit: int = 10):
        name = 'read_source_objects'
        try:
            entries = self._catalog['entries']
            if type(cursor) is not int or not 0 <= cursor <= len(entries) or type(limit) is not int or limit <= 0:
                raise SourceObjectToolError('Require integer cursor in catalog bounds and positive integer limit')
            value = {'ok': True, 'catalog_sha256': self._catalog['catalog_sha256'], 'cursor': cursor,
                     'total_objects': len(entries), 'next_cursor': None, 'has_more': False,
                     'scope': 'Metadata only; use read_source_object and read_source for complete original context.', 'objects': []}
            for entry in entries[cursor:cursor + limit]:
                occurrence = entry.get('occurrence') or {}
                original = entry.get('inventory_record') or {}
                row = {key: deepcopy(entry.get(key)) for key in ('object_id', 'blocking', 'diagnostics')}
                row.update(locator=occurrence.get('locator', original.get('source_path')),
                           kind=occurrence.get('kind', original.get('kind')))
                candidate = deepcopy(value)
                candidate['objects'].append(row)
                end = cursor + len(candidate['objects'])
                candidate.update(next_cursor=end if end < len(entries) else None, has_more=end < len(entries))
                if not self._fits(candidate, name):
                    if not value['objects']:
                        return self._budget_error()
                    break
                value = candidate
            return self._bounded(value, name)
        except Exception as exc:
            return self._error(exc, name)

    def read_source_object(self, object_id: str):
        name = 'read_source_object'
        try:
            entries = [e for e in self._catalog['entries'] if e.get('object_id') == object_id]
            if type(object_id) is not str or not object_id or len(entries) != 1:
                raise SourceObjectToolError('Unknown source object occurrence')
            entry = entries[0]
            current = None
            if self._bound and self.ledger is not None:
                current = next(r for r in self.ledger.coverage()['objects'] if r['object_id'] == object_id)
            result = {'ok': True, 'catalog_sha256': self._catalog['catalog_sha256'], 'entry': deepcopy(entry),
                      'scope_sha256': self.ledger.scope_sha256 if self._bound and self.ledger else None,
                      'representation': deepcopy(self.ledger.representations[object_id]) if self.ledger else None,
                      'current': current, 'proposal': None, 'review': None,
                      'instruction': 'Metadata is not decoded formula evidence. Read every context/support reference via source tools before proposing; independent review decides relevance.'}
            if current and current['proposal_id']:
                pid = current['proposal_id']
                result['proposal'] = deepcopy(self.ledger.proposals[pid])
                result['review'] = deepcopy(self.ledger.reviews.get(pid))
            return self._bounded(result, name)
        except Exception as exc:
            return self._error(exc, name)

    def propose_source_object(self, object_id: str, proposal: dict):
        name = 'propose_source_object'
        try:
            value = strict_json(proposal)
            if type(value) is not dict or set(value) != PROPOSAL_FIELDS:
                raise SourceObjectToolError('Proposal requires exactly disposition/rationale/affected_output_paths/supporting_refs/context_refs/source_limitations; origin IDs are host-owned')
            if value['disposition'] not in MODES:
                raise SourceObjectToolError('Only not_relied_on, outside_task, unresolved_relevant or unclassified are supported')
            if self.ledger is None or not self._bound or not self._draft_ready:
                raise SourceObjectToolError('Complete catalog and a current valid draft context are required')
            if object_id not in self.ledger.objects:
                raise SourceObjectToolError('Unknown source object occurrence')
            # This small acknowledgement has no unpredictable proposal-ID budget.
            # The immutable ID is discoverable via read_source_object/host ledger.
            response = {'ok': True, 'object_id': object_id, 'scope_sha256': self.ledger.scope_sha256,
                        'status': 'proposed', 'accepted': False,
                        'next_action': 'Independent review is required. Read the object to inspect the stored proposal.'}
            if not self._fits(response, name):
                return self._budget_error()
            arguments = {'object_id': object_id, 'proposal': deepcopy(value)}
            try:
                call_id = self.generate_origin_call_id(name, deepcopy(arguments))
            except Exception:
                raise SourceObjectToolError('No unique real Agent generation matches this exact tool request') from None
            call_id = self._origin_receipt(call_id)
            stored = {**value, 'origin_call_ids': [call_id]}
            # Preserve idempotence for delivery retry of the same real tool call.
            latest = [r for r in self.ledger.proposals.values() if r['object_id'] == object_id
                      and r['scope_sha256'] == self.ledger.scope_sha256]
            if not latest or latest[-1]['proposal'] != stored:
                self.ledger.propose(object_id, stored)
            return deepcopy(response)
        except Exception as exc:
            return self._error(exc, name)

    def handlers(self):
        return {name: getattr(self, name) for name in TOOLS}
