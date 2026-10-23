"""Audit measured work and literal-source preservation; not a quality scorer."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import statistics


def review(root):
    root=Path(root);out=root/'r3'
    if not (out/'full-done.json').exists():raise RuntimeError('Full A/B not complete')
    plan=json.loads((root/'r2/mineru-plan-v3.json').read_bytes())
    items={c['id']:c for g in plan['groups'] for c in g['candidates']}
    fixtures=json.loads((out/'classification-fixtures.json').read_bytes())['cases']
    result={'scope':'Official training C1, Kev extraction only; no main Agent or Word timing',
        'document_sha256':plan['document_sha256'],'source_fragments':len(items),'main_agent_calls':0,'runs':{}}
    for name in ('baseline','joint','grouped'):
        folder=out/f'full-{name}';a=json.loads((folder/'candidates.json').read_bytes())
        calls=json.loads((folder/'calls.json').read_bytes());by_kind=defaultdict(Counter);ids=[]
        for call in calls:
            if 'error' in call:raise RuntimeError('Failed call in completed experiment')
            t=by_kind[call['kind']];t['requests']+=1
            t['questions']+=len(call['response']['answers'])
            t['input_tokens']+=call['response']['usage']['input_tokens']
            if call['kind']=='classify':ids.append(call['identity'])
            if call['kind']=='classify_group':ids.extend(call['identity'])
        literal_failures=[]
        for e in a['entries']:
            ref=e['source_refs'][0]
            source=items.get(f"{ref['evidence_id']}:{ref['start']}:{ref['end']}")
            value=e.get('value') if e['section']=='基本信息' else e['row'].get('病害描述',e['row'].get('建议内容'))
            if source is None or value not in source['text']:literal_failures.append(e['id'])
        p=json.loads((out/f'pilot-{name}/regression.json').read_bytes())
        run={k:a[k] for k in ('total_seconds','load_seconds','input_tokens','decision_requests','stage_seconds','classification_counts','peak_gpu_allocated_bytes')}
        run.update(extraction_seconds=a['total_seconds']-a['load_seconds'],
            questions=sum(t['questions'] for t in by_kind.values()),by_kind=dict(by_kind),
            all_655_fragments_classified_once=Counter(ids)==Counter(items.keys()),
            classified_fragments=len(ids),literal_source_failures=literal_failures,
            entry_counts=Counter(e['section'] for e in a['entries']),
            selected_checks={'correct':p['correct'],'total':p['total'],'not_document_accuracy':True},
            unresolved=len(a['unresolved']),
            primary_current_location_rows={ref:any(e['section']=='病害明细' and e['source_refs'][0]['evidence_id']==ref for e in a['entries'])
                for ref in ('E000229','E000230','E000240','E000241','E000263','E000274')},
            major_action_location_rows={ref:any(e['section']=='维修建议' and e['source_refs'][0]['evidence_id']==ref for e in a['entries'])
                for ref in ('E000565','E000566','E000567','E000568')},
            selected_current_defect_refs={f['id']:sum(e['section']=='病害明细' and e['source_refs'][0]==f['source_refs'][0] for e in a['entries'])
                for f in fixtures if f['gold']=='defect' and f['id'].startswith('E')},
            selected_action_presence={f['id']:any(e['section']=='维修建议' and f['text'].strip('；。;') in e['row']['建议内容'] for e in a['entries'])
                for f in fixtures if f['gold']=='advice'},
            basic_values={e['field']:e['value'] for e in a['entries'] if e['section']=='基本信息'})
        result['runs'][name]=run
    base=result['runs']['baseline']
    for name,a in result['runs'].items():
        a['versus_baseline']={
            'extraction_speedup':base['extraction_seconds']/a['extraction_seconds'],
            'total_speedup':base['total_seconds']/a['total_seconds'],
            'requests_reduction_fraction':1-a['decision_requests']/base['decision_requests'],
            'questions_reduction_fraction':1-a['questions']/base['questions'],
            'input_reduction_fraction':1-a['input_tokens']/base['input_tokens']}
    confirmations=[]
    for folder in (out/'full-grouped',out/'confirm-grouped-1',out/'confirm-grouped-2'):
        p=folder/'candidates.json'
        if not p.exists():continue
        a=json.loads(p.read_bytes());calls=json.loads((folder/'calls.json').read_bytes())
        ids=[identity for c in calls if c['kind']=='classify_group' for identity in c['identity']]
        confirmations.append({'name':folder.name,
            **{k:a[k] for k in ('total_seconds','load_seconds','decision_requests','decision_questions','input_tokens')},
            'extraction_seconds':a['total_seconds']-a['load_seconds'],
            'all_655_fragments_classified_once':Counter(ids)==Counter(items.keys()),
            'primary_current_location_rows':{ref:any(e['section']=='病害明细' and e['source_refs'][0]['evidence_id']==ref for e in a['entries'])
                for ref in ('E000229','E000230','E000240','E000241','E000263','E000274')},
            'major_action_location_rows':{ref:any(e['section']=='维修建议' and e['source_refs'][0]['evidence_id']==ref for e in a['entries'])
                for ref in ('E000565','E000566','E000567','E000568')}})
    result['grouped_confirmations']=confirmations
    result['grouped_summary']={k:{'median':statistics.median(a[k] for a in confirmations),
        'min':min(a[k] for a in confirmations),'max':max(a[k] for a in confirmations)}
        for k in ('total_seconds','extraction_seconds','decision_requests','decision_questions','input_tokens')}
    result['limitations']=[
        'No main Agent invocation, final Word, end-to-end report throughput or downstream handling of uncertain fragments tested.',
        'Selected checks and literal-source validation do not establish document completeness or correctness.',
        'Manual reading found retained duplicated findings, headings/absent-condition errors, wrong basic fields and historical associations.',
        'All source fragments were classified; uncertain classifications remain unresolved, not successful extraction.',
        'No semantic deduplication implemented in this experiment. Only combined classification and shared material context changed.']
    (out/'decision-reduction-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    print(json.dumps(review(sys.argv[1]),ensure_ascii=False,indent=2))
