"""Typed, homogeneous main-agent decisions over the atomic review journal.

Numbers address candidates only in this document/run. The original evidence,
source text and business values are not rewritten or selected by the program.
"""
from copy import deepcopy
import json
import re
from jsonschema import Draft202012Validator

from .document_candidate_review import CandidateReviewError
from .document_review_submission import ReviewSubmissionTools
from .docx_writer import ADVICE_FIELDS, BASIC_FIELDS, DEFECT_FIELDS
from .run_state import atomic_json, json_digest


def obj(properties, required):
    return dict(type='object', additionalProperties=False, properties=properties, required=required)


def arr(items, limit=5):
    return dict(type='array', minItems=1, maxItems=limit, items=items)


REF = dict(type='integer', minimum=1, description='本次文档候选的 candidate 整数编号。')
REASON = dict(type='string', minLength=1, maxLength=240)
VALUE = dict(type=['string', 'number', 'null'], maxLength=1200)
SCHEMAS = {
    'select_basic_info': obj({'selections': {**obj({k: REF for k,_ in BASIC_FIELDS}, []), 'minProperties': 1, 'maxProperties': 24}}, ['selections']),
    'merge_duplicate_entries': obj({'groups': arr(obj({'keep': REF, 'others': {
        **arr(REF, 24), 'uniqueItems': True}, 'reason': REASON}, ['keep', 'others', 'reason']))}, ['groups']),
    'revise_basic_info': obj({'items': arr(obj({'candidate': REF, 'value': VALUE}, ['candidate', 'value']))}, ['items']),
    'revise_defects': obj({'items': arr({**obj({'candidate': REF, **{k: VALUE for k in DEFECT_FIELDS}}, ['candidate']), 'minProperties': 2})}, ['items']),
    'revise_recommendations': obj({'items': arr({**obj({'candidate': REF, **{k: VALUE for k in ADVICE_FIELDS}}, ['candidate']), 'minProperties': 2})}, ['items']),
    'exclude_entries': obj({'items': arr(obj({'candidate': REF, 'reason': REASON}, ['candidate', 'reason']))}, ['items']),
    # Review explanations are not delivery cells. Length alone must not abort
    # an otherwise valid finish. Request output budgets and repetition guards
    # still bound generation; preserve the model's complete explanation.
    'complete_review': obj({'basis': dict(type='string', minLength=1),
        'notes': dict(type='string', default='')}, ['basis']),
}
SCHEMAS['finish_report'] = deepcopy(SCHEMAS['complete_review'])
SCHEMAS['complete_review']['properties']['gap_checks'] = obj({
    section: dict(type='string', minLength=1,
        description='模型根据本轮实际证据对该类补漏的结论；可说明补存、原文未涉及或仍未解决，不按行数推断。')
    for section in ('病害明细', '维修建议')}, ['病害明细', '维修建议'])
DESCRIPTIONS = {
    'finish_report': '确认已读回当前Word并完成快速自查和交付。basis写自查依据，notes写剩余缺项。只有真实当前文件三表均已送入此请求才接受；程序绑定文件版本。',
    'select_basic_info': '选择已有基本信息候选；不同字段可合批，同一字段只能选一个。selections是字段名到候选整数编号的对象，同一字段只能出现一次。不重写内容。',
    'merge_duplicate_entries': '仅合并模型判断为重复的明细，每组保留keep并合入others；同批只能同类病害或同类建议，不同时改字段。',
    'revise_basic_info': '仅修改基本信息值；每项candidate/value。保留原始候选及依据。',
    'revise_defects': '仅修改病害明细；每项candidate及需要改变的病害列，不重写未变字段。',
    'revise_recommendations': '仅修改维修建议；每项candidate及需要改变的建议列，不重写未变字段。',
    'exclude_entries': '仅排除明确不保留的候选；每项candidate/reason，同批只处理同一类。',
    'complete_review': '完成主Agent快速审核，不包含任何编辑。basis简述已审内容；程序提示补漏时，gap_checks分别写病害明细和维修建议的证据判断，合理空表可保留，未处理不能说没有。notes记录剩余缺项。随后必须读取实际Word并finish。',
}
EXAMPLES = {
    'select_basic_info': {'selections': {'桥梁名称': 1}},
    'merge_duplicate_entries': {'groups': [{'keep': 2, 'others': [3], 'reason': '两处依据描述同一构件同一病害'}]},
    'revise_basic_info': {'items': [{'candidate': 1, 'value': '经原文确认的值'}]},
    'revise_defects': {'items': [{'candidate': 2, '病害描述': '经原文核对的简短描述'}]},
    'revise_recommendations': {'items': [{'candidate': 3, '建议内容': '经原文核对的简短措施'}]},
    'exclude_entries': {'items': [{'candidate': 2, 'reason': '不保留的明确理由'}]},
    'complete_review': {'basis': '已快速核对交接候选及附带依据', 'notes': '仍存在的具体缺项；无则空串'},
}


def derived_advice_statistics(state, update):
    """Count final rows, never assign their categories or certify completeness.

    Included in the same atomic transaction, bound to the current advice facts.
    Original model statistics remain in candidate facts/the journal for audit.
    With no rows remove summary fields rather than claiming no source advice.
    """
    draft = deepcopy(update.get('draft', state['draft']))
    bindings = deepcopy(update.get('draft_bindings', state['draft_bindings']))
    before = deepcopy((draft, bindings))
    rows = draft['维修建议']
    ids = list(dict.fromkeys(cid for i in range(len(rows))
        for cid in bindings.get(f'维修建议.{i}', [])))
    counts = {category: sum(r.get('建议类别') == category for r in rows)
        for category in ('立即处置', '尽快维修', '预防性养护')}
    counts['未分类/其他'] = len(rows) - sum(counts.values())
    value = '按当前交付明细统计：' + '、'.join(f'{k}{v}条' for k, v in counts.items())
    for field in ('建议', '建议统计'):
        if ids:
            draft['基本信息'][field] = value
            bindings['基本信息.' + field] = ids.copy()
        else:
            draft['基本信息'].pop(field, None)
            bindings.pop('基本信息.' + field, None)
    if (draft, bindings) != before:
        update.update(draft=draft, draft_bindings=bindings,
            draft_fact_ids=sorted({cid for bound in bindings.values() for cid in bound}))
    return {'basis': 'current_delivery_rows_not_source_completeness',
        'counts': counts, 'row_count': len(rows), 'categories_inferred_by_program': False}


class MainReviewTools(ReviewSubmissionTools):
    def __init__(self, workspace, handoff_evidence, child_summary, name):
        super().__init__(workspace, handoff_evidence, child_summary)
        if name not in SCHEMAS or name == 'finish_report':
            raise ValueError('Unknown main review tool')
        self.name = name
        self.ids = {i: key for i, key in enumerate(sorted(self.handoff), 1)}
        self.numbers = {key: i for i, key in self.ids.items()}
        record = {'identity': self.identity, 'candidates': {str(k): v for k, v in self.ids.items()}}
        path = self.directory / 'main-review-candidate-map.json'
        with self._lock():
            if path.exists():
                if json.loads(path.read_text()) != record:
                    raise ValueError('Candidate number map belongs to another document/run')
            else:
                atomic_json(path, record)

    def _schema_check(self, payload):
        try:
            json_digest(payload)
        except (ValueError, TypeError):
            raise CandidateReviewError('MAIN_REVIEW_ARGUMENTS', 'Arguments must be finite JSON', 'arguments')
        error = next(Draft202012Validator(SCHEMAS[self.name]).iter_errors(payload), None)
        if error:
            path = ''.join(f'[{p}]' if isinstance(p, int) else ('.' if i else '') + p
                for i, p in enumerate(error.absolute_path)) or 'arguments'
            if error.validator == 'required':
                missing = next(k for k in error.validator_value if k not in error.instance)
                path = (path + '.' if path != 'arguments' else '') + missing
            raise CandidateReviewError('MAIN_REVIEW_ARGUMENTS', error.message, path)
        for text in [payload.get('basis', 'ok'), *(r.get('reason', 'ok') for r in payload.get('groups', payload.get('items', [])))]:
            if not text.strip():
                raise CandidateReviewError('MAIN_REVIEW_REASON', 'Review reason cannot be blank', 'basis/reason')

    def _excluded_candidates(self, state):
        excluded = super()._excluded_candidates(state)
        for row in self._journal()['calls'].values():
            if row.get('phase') != 'complete' and not self._committed(state, row):
                continue
            # Accepted journal receipts contain original IDs, never model-guessed IDs.
            excluded.update(row.get('response', {}).get('batch_receipt', {}).get('accepted', {}).get('exclusions', []))
        return excluded

    def _plan(self, state, payload):
        routed = {k: [] for k in ('selections', 'merges', 'patches', 'exclusions', 'reviews')}
        origins, sections = {}, set()
        expected = {'select_basic_info': '基本信息', 'revise_basic_info': '基本信息',
            'revise_defects': '病害明细', 'revise_recommendations': '维修建议'}.get(self.name)

        def identifier(number, path):
            if type(number) is not int or number not in self.ids:
                raise CandidateReviewError('CANDIDATE_UNKNOWN', 'Use a candidate integer from this document inventory', path)
            key = self.ids[number]
            section = self._candidate(key, state, path)['section']
            sections.add(section)
            if expected and section != expected:
                raise CandidateReviewError('CANDIDATE_SECTION', f'This tool accepts only {expected}', path)
            return key

        if self.name == 'select_basic_info':
            for field, number in payload['selections'].items():
                path = 'selections.' + field
                key = identifier(number, path)
                if self.handoff[key].get('field') != field:
                    choices = [self.numbers[k] for k, item in self.handoff.items()
                        if item['section'] == '基本信息' and item.get('field') == field]
                    raise CandidateReviewError('CANDIDATE_FIELD', f'This field has candidate choices {choices}; select one using its evidence', path)
                i = len(routed['selections'])
                routed['selections'].append(key)
                origins[f'selections[{i}]'] = path
        elif self.name == 'merge_duplicate_entries':
            for i, group in enumerate(payload['groups']):
                routed['merges'].append({'keep_id': identifier(group['keep'], f'groups[{i}].keep'),
                    'other_ids': [identifier(n, f'groups[{i}].others[{j}]') for j, n in enumerate(group['others'])], 'reason': group['reason']})
                origins[f'merges[{i}]'] = f'groups[{i}]'
        elif self.name.startswith('revise_'):
            for i, item in enumerate(payload['items']):
                routed['patches'].append({'candidate_id': identifier(item['candidate'], f'items[{i}].candidate'),
                    'changes': {k: deepcopy(v) for k, v in item.items() if k != 'candidate'}})
                origins[f'patches[{i}]'] = f'items[{i}]'
        elif self.name == 'exclude_entries':
            for i, item in enumerate(payload['items']):
                routed['exclusions'].append({'candidate_id': identifier(item['candidate'], f'items[{i}].candidate'), 'reason': item['reason']})
                origins[f'exclusions[{i}]'] = f'items[{i}]'
        elif self.name == 'complete_review':
            routed['reviews'] = [{'question_id': key, 'status': 'unknown', 'reason': '速度优先，接受暂不解决：' + payload['basis']}
                for key, row in state['questions'].items() if row.get('status') != 'resolved' or not row.get('valid')]
        if self.name in {'merge_duplicate_entries', 'exclude_entries'} and len(sections) > 1:
            raise CandidateReviewError('MIXED_REVIEW_BATCH', 'Batch only one section; submit disease and advice decisions separately', 'groups' if 'groups' in payload else 'items')
        try:
            update, response = super()._plan(state, routed)
        except CandidateReviewError as exc:
            for old, new in origins.items():
                if exc.path == old or exc.path.startswith(old + '.'):
                    suffix = exc.path[len(old):].replace('.changes', '').replace('.candidate_id', '.candidate').replace('.keep_id', '.keep').replace('.other_ids', '.others')
                    exc.path = new + suffix
                    break
            raise
        response['advice_statistics'] = derived_advice_statistics(state, update)
        changed = len(update) > 1
        response.update(changed=changed, version=state['version'] + int(changed))
        response['draft_counts'] = {k: len(v) for k, v in update.get('draft', state['draft']).items()}
        response['batch_receipt'].update(status='saved' if changed else 'unchanged',
            workspace_version=response['version'])
        response['batch_receipt'].update(limit=24 if self.name == 'select_basic_info' else 5,
            instruction='已保存本批决定，勿重复提交。继续其他同类修改；审核结束后complete_review，再读取实际Word并finish。')
        response.update(review_completed=self.name == 'complete_review', exhaustive_quality_verified=False)
        if self.name == 'complete_review' and 'gap_checks' in payload:
            response['model_gap_checks'] = deepcopy(payload['gap_checks'])
        return update, response

    def call(self, **payload):
        response = self._execute_review(payload, tool_name=self.name)
        if not response.get('ok'):
            details = response['error'].setdefault('details', {})
            if details.get('applied') is False:
                details['action'] = '本批未保存；先前成功批次保留。只修本批指出的字段，仍用当前专用工具。'
            details['example'] = deepcopy(EXAMPLES[self.name])
        return self._public_response(response)

    def _public_response(self, response):
        # Only structural identifier fields are projected. Never alter reason,
        # business value or source strings, even if they resemble identifiers.
        result = deepcopy(response)
        for op in result.get('operations', []):
            if 'candidate_ids' in op:
                op['candidate_ids'] = [self.numbers.get(k, k) for k in op['candidate_ids']]
        accepted = result.get('batch_receipt', {}).get('accepted', {})
        for key in ('selections', 'patches', 'exclusions'):
            if key in accepted:
                accepted[key] = [self.numbers.get(k, k) for k in accepted[key]]
        for group in accepted.get('merges', []):
            group['keep_id'] = self.numbers.get(group['keep_id'], group['keep_id'])
            group['other_ids'] = [self.numbers.get(k, k) for k in group['other_ids']]
        return result

    def project_bundle(self, bundle):
        out = deepcopy(bundle)
        for row in out['candidates']:
            row['candidate'] = self.numbers[row.pop('fact_id')]
        for values in out.get('current_output', {}).get('path_bindings', {}).values():
            values[:] = [self.numbers.get(k, k) for k in values]
        for key in ('candidate_ids', 'navigation_only_candidate_ids'):
            if key in out.get('remaining', {}):
                out['remaining'][key] = [self.numbers.get(k, k) for k in out['remaining'][key]]
        for issue in out.get('issues', {}).values():
            for key in ('candidate_ids', 'depends_on'):
                if isinstance(issue.get(key), list):
                    issue[key] = [self.numbers.get(k, k) for k in issue[key]]
        basic = {}
        for row in out['candidates']:
            if row['section'] == '基本信息':
                basic.setdefault(row['field'], []).append(row)
        out['candidates'] = [r for r in out['candidates'] if r['section'] != '基本信息']
        out = {'basic_candidates_by_field': basic, **out}
        out['candidate_reference'] = 'candidate是本次文档的整数编号；专用审核工具只传该整数。原始编号和完整来源由程序保存。'
        # Exact position strings are a navigation index, not duplicate verdicts.
        groups = {}
        for row in out['candidates']:
            value = row.get('value')
            if isinstance(value, dict) and value.get('病害部位'):
                groups.setdefault(row['section'], {}).setdefault(str(value['病害部位']), []).append(row['candidate'])
        out['position_navigation'] = groups
        # The complete handoff bundle stays on disk. Exhaustive E-ID lists
        # and storage hashes are bookkeeping, not additional source content.
        for child in out.get('children', []):
            child.pop('primary_evidence_ids', None)
        for row in [*out['candidates'], *(r for values in basic.values() for r in values)]:
            row['table_headers'] = [{key: deepcopy(source[key]) for key in
                ('evidence_id', 'start', 'end', 'text', 'header_kind') if key in source}
                for ref in row.get('header_keys', [])
                for source in [out.get('source_map', {}).get(ref, {})] if source]
        # Range keys are a host index, not valid read_source IDs. Do not invite
        # the model to copy E:start:end into an evidence_id argument.
        source_map = out['source_map']
        for row in [*out['candidates'], *(r for values in basic.values() for r in values)]:
            row['source_refs'] = [{k: source_map[ref][k] for k in ('evidence_id', 'start', 'end')
                                   if k in source_map[ref]} for ref in row.pop('source_keys', [])]
            row.pop('header_keys', None)
        out['source_map'] = list(source_map.values())
        out['source_navigation'] = ('source_map为逐字原文列表，条目中的source_refs可定位对应段落。'
            'table_headers为已提供的表头或候选，候选语义仍由你判断。'
            '仅有缺口才read_source，evidence_id只传E编号，不拼接字符范围。')
        out['review_order'] = '先核对病害重复和类型/描述矛盾，再核对同对象同措施的建议；对照附带证据，执行必要合并和局部修正，成功后complete_review。相同位置不代表相同病害，不同位置文字也可能描述同一对象；判断由模型完成。'
        return out

    def next_action(self):
        """Focus the next invocation using committed edits, not prose claims.

        This is guidance, not a mandatory edit quota. A model may retain
        uncertain items and complete_review with a reason instead of inventing
        a merge to advance a program stage.
        """
        committed = [r for r in self._journal()['calls'].values() if r.get('phase') == 'complete']
        if not any(r.get('tool_name') == 'select_basic_info' for r in committed):
            return '当前先从basic_candidates_by_field选择基本信息，用select_basic_info直接提交。此轮不展开病害或建议分析；没有可选基本信息才跳过。不要写普通正文。'
        edited = set()
        for call in committed:
            accepted = call.get('response', {}).get('batch_receipt', {}).get('accepted', {})
            ids = accepted.get('patches', []) + accepted.get('exclusions', [])
            ids += [group['keep_id'] for group in accepted.get('merges', [])]
            edited.update(self.handoff[cid]['section'] for cid in ids if cid in self.handoff)
        if '病害明细' not in edited:
            return '当前只处理一批已能确定的病害重复或类型/描述矛盾，直接merge_duplicate_entries或revise_defects，不写分析。不能确定的保留；若病害无需修改，可直接处理建议或complete_review说明。'
        if '维修建议' not in edited:
            return '接着处理一批同对象同措施的重复建议，直接merge_duplicate_entries；必要时revise_recommendations。不写分析。病害若还有明确未处理问题仍可局部补改；无需修改时complete_review说明。'
        return '病害、建议均已有修改成功回执。若还有明确未完成的修改，直接提交那一批；否则立即complete_review，再读Word并finish_report。不复述已保存的修改。'
