"""Reuse repaired MinerU packets and their existing canonical-reference mapping."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from bridge_evidence.contest.document_material_run import load_material_index
from bridge_evidence.contest.run_state import json_digest


def prepare(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'source-projection.json').read_bytes())
    materials = json.loads((folder/'materials.json').read_bytes())
    index = load_material_index(manifest, materials, folder/'source.docx')
    groups, seen, diagnostics = [], set(), []
    for packet in materials['packets']:
        if packet.get('region') in {'toc', 'references', 'footer', 'header'}:
            diagnostics.append({'packet': packet['packet_id'], 'reason': 'non_content_region'})
            continue
        resolved = index.packet(packet['packet_id'])
        if resolved['unresolved']:
            diagnostics.append({'packet': packet['packet_id'], 'reason': 'unresolved_alias',
                                'details': resolved['unresolved']})
        candidates = []
        for span in resolved['spans']:
            # Transport deduplication only: one exact original interval, not
            # a claim that similar findings at different positions are equal.
            identity = (span['evidence_id'], span['start'], span['end'])
            if identity in seen:
                continue
            seen.add(identity)
            text = span['text']
            # Keep each existing short source unit intact. Long units can be
            # continued at punctuation boundaries; ranges remain exact.
            intervals = [(0, len(text))] if len(text) <= 420 else [
                (m.start(), m.end()) for m in re.finditer(r'[^；;。\n]+[；;。\n]?', text)]
            for a,b in intervals:
                quote = text[a:b]
                if len(quote.strip()) < 6:
                    diagnostics.append({'evidence_id': span['evidence_id'],
                                        'reason': 'short_unit_deferred', 'characters': b-a})
                    continue
                candidates.append({'id': f"{span['evidence_id']}:{span['start']+a}:{span['start']+b}",
                    'text': quote, 'source_refs': [{'evidence_id': span['evidence_id'],
                        'start': span['start']+a, 'end': span['start']+b}]})
        current, chars = [], 0
        def emit():
            if current:
                groups.append({'packet': packet['packet_id'], 'headings': packet.get('headings', []),
                    'table': packet.get('table'), 'context': deepcopy(resolved['context']),
                    'candidates': deepcopy(current)})
        for candidate in candidates:
            if current and (chars + len(candidate['text']) > 1400 or len(current) >= 12):
                emit(); current=[]; chars=0
            current.append(candidate); chars += len(candidate['text'])
        emit()
    return {'schema': 'cityagent-kev-mineru-plan-v1', 'training': False,
        'official_training_only': True, 'document_sha256': manifest['document_sha256'],
        'manifest_sha256': json_digest(manifest), 'material_identity': index.identity,
        'source_folder': str(folder), 'groups': groups, 'diagnostics': diagnostics,
        'policy': 'Original MinerU organization; exact canonical spans; no NER and no child Agent. Deferred or unselected material remains available to the main Agent.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-folder',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(); result=prepare(args.source_folder)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'groups':len(result['groups']), 'candidates':sum(len(g['candidates']) for g in result['groups']),
                      'diagnostics':len(result['diagnostics'])}))
