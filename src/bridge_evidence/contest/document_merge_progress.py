"""Mechanical merge progress and verbatim child evidence, never semantic verdicts."""
from __future__ import annotations
from copy import deepcopy
from .document_workspace_tools import WorkspaceTools
from .run_state import json_digest
from .document_delivery_artifacts import artifact_matches_snapshot, artifact_page_matches


class MergeProgressStalled(RuntimeError):
    pass


class ConsecutiveProgressGuard:
    """Check complete model/tool rounds, not each member of a tool batch.

    One warning after six unchanged rounds; stop after six more. This is a
    liveness limit only. It never determines whether bridge facts are correct.
    """
    def __init__(self, state=None, *, warning_after=6, stop_after=12):
        self.state = deepcopy(state or {})
        self.warning_after, self.stop_after = warning_after, stop_after

    def observe(self, generation_id, fingerprint):
        if generation_id is None:
            self.state.setdefault('fingerprint', fingerprint)
            return None
        if generation_id == self.state.get('generation_id'):
            return None
        unchanged = (self.state.get('unchanged', 0) + 1
                     if self.state.get('fingerprint') == fingerprint else 0)
        self.state.update(generation_id=generation_id, fingerprint=fingerprint,
                          unchanged=unchanged)
        if unchanged >= self.stop_after:
            return 'stop'
        if unchanged == self.warning_after:
            return 'warn'
        return None


def child_fact_id(task_id, entry_id):
    return 'child_' + json_digest({'task': task_id, 'entry': entry_id})[:24]


def build_handoff_evidence(manifest, results, *, material_index=None):
    from .document_material_refs import validate_entry_material_provenance
    units = {u['id']: u for u in manifest['units']}
    tables = {t['id']: t for t in manifest.get('tables', [])}
    ledger = {}
    def excerpt(ref):
        u = units[ref['evidence_id']]
        start, end = ref.get('start', 0), ref.get('end', len(u['text']))
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(u['text']):
            raise ValueError('Handoff evidence has an invalid original-source range')
        return {'evidence_id': u['id'], 'start': start, 'end': end, 'text': u['text'][start:end],
                'source_path': u.get('source_path'),
                'source_metadata': {k: deepcopy(u[k]) for k in
                    ('context', 'cells', 'table_id', 'row_index', 'segment_index') if k in u}}
    for result in results:
        state = result['entries']; entries = state.get('entries', {})
        origins = validate_entry_material_provenance(state, manifest, material_index)
        for entry in entries.values() if isinstance(entries, dict) else entries:
            refs = deepcopy(entry['source_refs']); quotes = [excerpt(r) for r in refs]
            header_ids = dict.fromkeys(h for r in refs for h in tables.get(
                units[r['evidence_id']].get('table_id'), {}).get('header_evidence_ids', []))
            headers = [excerpt({'evidence_id': h}) for h in header_ids if h in units and units[h]['text']]
            material_evidence = []
            for origin in origins.get(entry['id'], []):
                resolved = material_index.resolve(origin['source_ref']['evidence_id'], include_context=True)
                context = []
                for span in resolved.get('context', []):
                    if span.get('start') == span.get('end'):
                        continue  # Empty cells contain no quotation, not missing facts.
                    quote = excerpt(span)
                    if span.get('text') != quote['text']:
                        raise ValueError('Material handoff context differs from immutable original text')
                    quote['role'] = span.get('role', 'context')
                    if quote not in context:
                        context.append(quote)
                material_evidence.append({**deepcopy(origin), 'context': context,
                    'source_metadata': deepcopy(resolved.get('metadata', {})),
                    'context_is_supplemental_original_text': True,
                    'context_is_not_an_additional_model_claim': True})
            ledger[child_fact_id(result['task_id'], entry['id'])] = {
                'task_id': result['task_id'], 'entry_id': entry['id'], 'section': entry['section'],
                'field': entry.get('field'), 'candidate_value': deepcopy(entry.get('value') if entry['section'] == '基本信息' else entry['row']),
                'source_refs': refs, 'source_excerpts': quotes, 'table_headers': headers,
                **({'material_evidence': material_evidence} if material_evidence else {}),
                'model_context': deepcopy(entry.get('context', {})),
                'unresolved': deepcopy((result.get('handoff') or {
                    'unresolved': result.get('pending_unresolved', [])}).get('unresolved', [])),
                'semantic_approval': False, 'excerpts_are_verbatim_not_model_summary': True}
    return ledger


class MergeWorkspaceTools(WorkspaceTools):
    """Existing bounded/fragmented fact views include their original handoff evidence."""
    def __init__(self, *args, handoff_evidence, **kwargs):
        super().__init__(*args, **kwargs)
        self.handoff_evidence = deepcopy(handoff_evidence)

    def _state(self):
        state, delivery = super()._state()
        for identifier, fact in state.get('facts', {}).items():
            evidence = self.handoff_evidence.get(identifier)
            if (evidence and fact.get('value') == evidence['candidate_value']
                    and fact.get('source_refs') == evidence['source_refs']):
                fact['handoff_evidence'] = deepcopy(evidence)
        return state, delivery


def inspection_progress(state, artifact, pages):
    artifact = artifact or {}
    current = (artifact_matches_snapshot(state, artifact)
               and artifact.get('structure_valid') is True and artifact.get('content_validated') is True
               and artifact.get('requested_failure_marker') is not True)
    pending, complete = [], []
    for view in ('basic', 'advice', 'defects'):
        valid = [p for p in pages if current and p.get('view') == view
                 and artifact_page_matches(p, artifact)]
        valid = [p for p in valid if all(type(p.get(k)) is int for k in ('cursor', 'end', 'total_chars'))
                 and 0 <= p['cursor'] <= p['end'] <= p['total_chars']]
        totals = {p['total_chars'] for p in valid}
        cursor = 0
        if len(totals) == 1:
            for p in sorted(valid, key=lambda p: (p['cursor'], p['end'])):
                if p['cursor'] > cursor: break
                cursor = max(cursor, p['end'])
            if cursor == next(iter(totals)):
                complete.append(view); continue
        pending.append({'view': view, 'cursor': cursor if len(totals) == 1 else 0, 'limit': 8})
    return {'current_export': current, 'completed_views': complete, 'pending_reads': pending}


def semantic_state_fingerprint(state, inspection):
    # Reading version counters, notes, no-op saves, repeated reads and ordinary
    # declarations cannot masquerade as successful draft/file progress.
    records = {}
    for collection in ('facts', 'relations', 'questions'):
        records[collection] = {k: {f: v[f] for f in ('id', 'value', 'source_refs', 'status', 'valid',
            'reason', 'question', 'depends_on', 'subject_id', 'object_id') if f in v}
            for k, v in state.get(collection, {}).items()}
    return json_digest({'records': records, 'draft': state.get('draft'), 'bindings': state.get('draft_bindings'),
        'reads': state.get('reads'), 'inspections': inspection})


def next_blocker_action(state, delivery, pending_coverage=()):
    """One concrete mechanical next step shared by finish and continuation.

    An unknown answer may be a valid semantic outcome. Only delivery blockers
    and the explicit child coverage checks require further action here.
    """
    blockers = list(delivery.get('blockers', []))
    if delivery.get('ready') and not pending_coverage:
        return None
    questions = state.get('questions', {})
    required = set(pending_coverage)
    required.update(item.split(':', 1)[1] for item in blockers
                    if item.startswith(('unresolved_question:', 'invalidated_record:')))
    candidates = [key for key, q in questions.items()
        if key in required or q.get('valid') is not True or q.get('status') not in ('resolved', 'unknown')]
    result = {'stage': 'resolve_candidates', 'blockers': blockers[:8],
              'remaining_count': len(blockers), 'semantic_approval': False}
    missing_question = next((key for key in pending_coverage if key not in questions), None)
    if missing_question is not None:
        result.update(target_question_id=missing_question,
            next_action={'tool': 'update_workspace', 'required_question_id': missing_question},
            instruction='所列子任务覆盖核对记录缺失。请在工作区恢复该问题，结合该子任务原文和已保存条目审核三类交付，记录真实结论和理由；不得仅根据行数声称完成。')
        return result
    if candidates:
        target = sorted(candidates)[0]
        question = deepcopy(questions[target])
        for key in ('question', 'reason'):
            if isinstance(question.get(key), str):
                question[key] = question[key][:2400]
        result.update(target_question_id=target, target_question=question,
            next_action={'tool': 'read_workspace_page', 'arguments': {'view': 'unresolved_questions', 'limit': 4},
                         'target_question_id': target},
            instruction='只处理所列未决问题。优先核对facts.handoff_evidence的原文、表头和年份位置；证据不足才回查。局部修改草稿，再如实更新问题状态和理由。')
        return result
    unread = delivery.get('reading_progress', {}).get('unread_ranges', [])
    if unread:
        result.update(stage='complete_source_delivery', remaining_count=len(unread),
            missing_ranges_preview=deepcopy(unread[:4]),
            next_action={'tool': 'read_source', 'arguments': {'evidence_id': unread[0]['evidence_id']}},
            instruction='以下原文区间尚无模型收到内容的成功记录。调用所列普通阅读工具，程序管理位置和续页；无需猜字符偏移。读到不代表已抽取或已审核，仍由你判断是否需要补充草稿。')
        return result
    invalid = [b.split(':', 1)[1] for b in blockers if b.startswith('invalidated_record:')]
    if invalid:
        target = invalid[0]
        view = 'relations' if target in state.get('relations', {}) else 'facts'
        result.update(target_record_id=target,
            target_record=deepcopy(state.get(view, {}).get(target)),
            next_action={'tool': 'read_workspace_page', 'arguments': {'view': view, 'limit': 4}},
            instruction='该记录因依赖变化失效。只核对并修正所列记录及其草稿关联，不重复已经有效的文件检查。')
        return result
    view = ('draft_invalidations' if 'draft_invalidated' in blockers else
            'unbound_facts' if 'draft_missing' in blockers else
            'unverified_image_ids' if 'unverified_images' in blockers else
            'unverified_content' if 'unverified_manifest_content' in blockers else 'blockers')
    result.update(next_action={'tool': 'read_workspace_page', 'arguments': {'view': view, 'limit': 4}},
        instruction='仅查看所列阻塞的具体内容并修复；已有草稿和成功工具回执保留，程序不替你判断事实。')
    return result


def next_merge_action(state, delivery, pending_coverage, artifact, pages, self_check=None):
    """Return one actionable stage from receipts, not from guessed report contents."""
    inspection = inspection_progress(state, artifact, pages)
    unresolved = [k for k, q in state.get('questions', {}).items()
                  if q.get('valid') is not True or q.get('status') != 'resolved']
    blocker = next_blocker_action(state, delivery, pending_coverage)
    if blocker:
        stage, action, instruction = blocker['stage'], blocker['next_action'], blocker['instruction']
    elif inspection['pending_reads']:
        stage = 'inspect_current_file'
        action = {'tool': 'inspect_delivery', 'arguments': {
            key: value for key, value in inspection['pending_reads'][0].items() if key != 'cursor'}}
        instruction = ('只需继续读取下列当前文件尚未读完的部分，再由你判断是否正确、是否遗漏。'
            '已列为completed_views且仍绑定当前版本的部分无需重读；发现问题再局部修改。全部核对后提交finish。')
    else:
        stage = 'submit_file_self_check'
        action = {'tool': 'finish', 'program_binds_inspected_file': True}
        instruction = ('当前Word三部分的实际读取回执齐全。只需由你提交finish的self_check，'
            '如实填写inspected_export、complete、basis和remaining_issues；程序不替你判断内容正确。')
    return {**(blocker or {}), 'stage': stage, 'workspace_version': state['version'], 'draft_sha256': state['draft_sha256'],
        'next_action': action, 'instruction': instruction, 'pending_coverage': pending_coverage,
        'unresolved_question_ids': unresolved, 'blockers': deepcopy(delivery.get('blockers', [])[:8]),
        'file_inspection': inspection, 'self_check_submitted': bool(self_check),
        'model_reported_remaining_issues': deepcopy((self_check or {}).get('remaining_issues', [])),
        'self_check_accepted': False, 'semantic_approval': False,
        'saved_counts_diagnostic_only': {k: len(v) for k, v in (state.get('draft') or {}).items()}}
