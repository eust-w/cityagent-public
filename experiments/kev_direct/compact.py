"""Compact, document-bound requests; original provenance remains in the ledger.

This module proposes literal choices and organizes source material. It never
decides whether a finding is true, duplicated, new, or worse.
"""
from collections import Counter
from copy import deepcopy
import json


def grouped_questions(items, state_builder, maximum=8):
    """Share background across independent questions without changing targets."""
    from lean import classification_batches
    result = []
    for pack in classification_batches([(c, g) for c, g, _ in items], 'grouped', limit=maximum):
        questions = {}
        metadata = []
        by_id = {c['id']: qs for c, g, qs in items}
        for index, (candidate, group) in enumerate(pack):
            target = f'source_fragments[{index}].source_fragment'
            for name, question in by_id[candidate['id']].items():
                q = deepcopy(question)
                q['instructions'] = ('Judge ONLY ' + target + '. '
                    + q['instructions'].replace('source_fragment', target))
                questions[f'{index}_{name}'] = q
            metadata.append((candidate, group))
        result.append((state_builder(pack), questions, metadata))
    return result


def literal_field_options(field, pool):
    """Equal scalar options share a value, retaining every original reference.

    These are alternatives, not selected facts. No score is assigned a role by
    this function. The main model must select a role using the source context.
    """
    values = {}
    for value, candidate, group in pool:
        item = values.setdefault(value, {'field': field, 'value': value,
            'source_refs': [], 'status': 'uncertain', 'context': {
                'candidate_source': 'Literal alternative; NOT a selected or verified basic value'}})
        for ref in candidate['source_refs']:
            if ref not in item['source_refs']:
                item['source_refs'].append(deepcopy(ref))
    return list(values.values())


def compact_review_bundle(manifest, evidence, numbers, snapshot, unresolved):
    """All candidate values first; each canonical source unit appears once.

    Complete units preserve table rows and local context without repeating
    parser/XML provenance per candidate. Every exposed unit is registered with
    the actual request. This never marks a source as semantically reviewed.
    """
    units = {u['id']: u for u in manifest['units']}
    from bridge_evidence.contest.run_state import json_digest
    manifest_digest = json_digest(manifest)
    sources, rows, basic, contexts, context_numbers = {}, [], {}, {}, {}

    def source(ref):
        if ref.get('evidence_id') not in units:
            raise ValueError('Source belongs to another document or namespace')
        unit = units[ref['evidence_id']]
        a, b = ref.get('start', 0), ref.get('end', len(unit['text']))
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(unit['text']):
            raise ValueError('Invalid source interval')
        if 'text' in ref and ref['text'] != unit['text'][a:b]:
            raise ValueError('Source excerpt differs from immutable source')
        eid = unit['id']
        sources[eid] = {k: deepcopy(unit[k]) for k in
            ('id', 'text', 'kind', 'table_id', 'row_index', 'parent_table_id') if k in unit}
        layout, offset = [], 0
        for index,cell in enumerate(unit.get('cells', [])):
            if (cell.get('column_index',index)!=index or cell.get('column_span',1)!=1
                    or cell.get('vertical_merge') or ' | ' in cell['text']):
                layout.append({'column':cell.get('column_index',index),'start':offset,
                    'end':offset+len(cell['text']),'column_span':cell.get('column_span',1),
                    'merge':cell.get('vertical_merge')})
            offset += len(cell['text'])+3
        if layout:sources[eid]['cell_layout']=layout
        sources[eid]['evidence_id'] = eid
        sources[eid].pop('id', None)
        return {'evidence_id': eid, 'start': a, 'end': b}

    for fact_id, item in sorted(evidence.items(), key=lambda pair: numbers[pair[0]]):
        fact = snapshot.get('facts', {}).get(fact_id)
        value = fact['value'] if fact is not None else item['candidate_value']
        refs = fact['source_refs'] if fact is not None else item.get('source_refs', [])
        row = {'candidate': numbers[fact_id], 'value': deepcopy(value),
               'source_evidence_ids': list(dict.fromkeys(source(r)['evidence_id'] for r in refs))}
        context_ids = []
        for quote in item.get('source_excerpts', []) + item.get('table_headers', []):
            eid = source(quote)['evidence_id']
            if eid not in context_ids: context_ids.append(eid)
        for material in item.get('material_evidence', []):
            identity = material.get('identity', {})
            if identity.get('document_sha256') != manifest['document_sha256']:
                raise ValueError('Material context belongs to another document')
            if identity.get('manifest_sha256') != manifest_digest:
                raise ValueError('Material context belongs to another manifest')
            for quote in material.get('context', []):
                if quote.get('start') == quote.get('end'): continue
                eid = source(quote)['evidence_id']
                if eid not in context_ids: context_ids.append(eid)
        if context_ids:
            key = tuple(sorted(context_ids))
            if key not in context_numbers:
                number = str(len(context_numbers)+1)
                context_numbers[key] = number
                contexts[number] = list(key)
            row['context_group'] = context_numbers[key]
        if item['section'] == '基本信息':
            basic.setdefault(item['field'], []).append(row)
        else:
            row['section'] = item['section']
            rows.append(row)
    issue_counts = Counter(s.split(':', 1)[0] for s in unresolved)
    bundle = {'schema': 'cityagent-kev-compact-review-v1',
        'document_sha256': manifest['document_sha256'],
        'basic_candidates_by_field': basic, 'candidates': rows,
        'source_map': list(sources.values()), 'context_groups': contexts,
        'unresolved': {'count': len(unresolved), 'categories': dict(issue_counts),
            'details': {'tool': 'read_workspace_page', 'arguments': {'view': 'questions'}},
            'meaning': 'Full unresolved records persist; omitted details are NOT resolved.'},
        'candidate_count': len(evidence), 'all_candidate_values_included': True,
        'meaning': 'Literal candidates are unverified. Source text and header candidates are not judgments. '
                   'All alternatives require model selection; numbers identify this document only.'}
    ranges = [{'evidence_id': eid, 'start': 0, 'end': len(units[eid]['text'])}
              for eid in sources if units[eid]['text']]
    if len(basic) + len(rows) == 0 and evidence:
        raise ValueError('Candidate projection is empty')
    return bundle, ranges


def serialized_size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
