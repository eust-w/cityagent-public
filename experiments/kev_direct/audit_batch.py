"""Verify the three-training-document experiment from actual saved receipts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    root = Path(root)
    result = read(root/'result.json')
    config = read(root/'config.json')
    assert result['complete'] and result['extraction']['model_loads'] == 1
    suffix = result.get('service_snapshot_suffix', '')
    before, after = read(root/('services-before'+suffix+'.json')), read(root/('services-after'+suffix+'.json'))
    assert all(after.get(port) == identity for port, identity in before.items())
    service_dir = Path(result.get('service_directory', str(root/'service-9b')))
    stop = read(service_dir/'stop.json')
    assert stop['idle_before_stop'] and stop['returncode'] is not None
    model = config['main_model']
    service = read(service_dir/'ready.json')['models']['data'][0]
    assert service['id'] == model and service['max_model_len'] == 262144
    records = []
    for index, doc in enumerate(config['documents'], 1):
        doc_id = doc['id']
        folder = Path(doc['source_folder'])
        manifest = read(folder/'source-projection.json')
        plan = read(Path(doc['plan']))
        candidate = read(root/'extraction'/doc_id/'candidates.json')
        assert sha(folder/'source.docx') == doc['document_sha256']
        assert candidate['document_sha256'] == plan['document_sha256'] == manifest['document_sha256'] == doc['document_sha256']
        assert candidate['resident_session']['document_index'] == index
        assert candidate['resident_session']['model_loaded_this_document'] == (index == 1)
        expected = Counter(c['id'] for g in plan['groups'] for c in g['candidates'])
        classified = Counter()
        decisions = sorted((root/'extraction'/doc_id).glob('response-*.json'))
        for path in decisions:
            call = read(path)
            assert not call.get('error'), str(path)
            if call['kind'] == 'classify_group':
                classified.update(call['identity'])
        assert classified == expected and len(decisions) == candidate['decision_requests']
        directory = root/'main'/doc_id
        merged = read(directory/'result.json')
        workflow = read(directory/'merge/document_workflow.json')
        assert merged['complete'] and merged['main_model'] == model and merged['subagent_calls'] == 0
        delivery = merged['delivery']
        word = Path(delivery['docx_path'])
        assert sha(word) == delivery['file_sha256'] and delivery['complete'] and not delivery['incomplete']
        check = workflow['model_self_check']
        assert workflow['accepted_finish'] and workflow['actual_finish_origin']
        assert check['model_statement']['inspected_export'] and check['model_statement']['complete']
        assert check['verified_file_binding']['file_sha256'] == delivery['file_sha256']
        assert check['actual_inspection_receipt_sha256s']
        request_files = sorted((directory/'merge/model_requests').glob('*.json'))
        wire = []
        for path in request_files:
            request = read(path)['request']
            assert request['model'] == model and request['chat_template_kwargs']['enable_thinking'] is False
            assert request['max_tokens'] == 1024
            wire.append({key: request.get(key) for key in (
                'model', 'chat_template_kwargs', 'max_tokens', 'temperature', 'top_p',
                'top_k', 'presence_penalty', 'repetition_penalty', 'min_p')})
        assert wire and len(wire) <= 16
        sessions = workflow['sessions']
        tools = [tool for session in sessions for tool in session.get('tool_calls', [])]
        tool_names = Counter(call['name'] for call in tools)
        stream_reasoning_characters = 0
        finish_reasons = Counter()
        sse_files = sorted((directory/'merge/raw-sse').glob('*.sse'))
        assert len(sse_files) == len(wire)
        for path in sse_files:
            text = path.read_text()
            assert 'data: [DONE]' in text
            for line in text.splitlines():
                if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                    continue
                event = json.loads(line[5:])
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    stream_reasoning_characters += sum(len(delta.get(key) or '') for key in ('reasoning', 'reasoning_content'))
                    if choice.get('finish_reason'):
                        finish_reasons[choice['finish_reason']] += 1
        assert stream_reasoning_characters == 0
        with zipfile.ZipFile(word) as archive:
            xml = ET.fromstring(archive.read('word/document.xml'))
        ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        word_text = ''.join(node.text or '' for node in xml.findall('.//w:t', ns))
        assert word_text.strip()
        word_rows = [len(table.findall('w:tr', ns)) for table in xml.findall('.//w:tbl', ns)]
        records.append({
            'id':doc_id, 'document_sha256':doc['document_sha256'],
            'classified_fragments':sum(classified.values()), 'all_fragments_classified_once':all(n == 1 for n in classified.values()),
            'resident_session':candidate['resident_session'], 'kev_load_seconds':candidate['load_seconds'],
            'kev_seconds':candidate['total_seconds'],
            'kev_stage_seconds':next(item['seconds'] for item in result['extraction']['records'] if item['id'] == doc_id),
            'main_stage_seconds':next(item['seconds'] for item in result['main']['records'] if item['id'] == doc_id),
            'kev_input_tokens':candidate['input_tokens'],
            'kev_requests':candidate['decision_requests'], 'kev_questions':candidate['decision_questions'],
            'kev_unresolved':len(candidate['unresolved']), 'main_seconds':merged['main_seconds'],
            'main_calls':merged['main_calls'], 'main_input_tokens':merged['main_input_tokens'],
            'main_output_tokens':merged['main_output_tokens'], 'tool_calls':len(tools), 'tool_names':dict(tool_names),
            'tool_statuses':dict(Counter(tool['status'] for tool in tools)),
            'sdk_tool_error_count':sum(session.get('sdk_tool_error_count', 0) for session in sessions),
            'tool_error_count':sum(session.get('tool_error_count', 0) for session in sessions),
            'argument_parse_errors':sum(len(session.get('argument_parse_errors', [])) for session in sessions),
            'parse_recoveries':sum(len(session.get('parse_recovery', [])) for session in sessions),
            'stream_reasoning_characters':stream_reasoning_characters, 'finish_reasons':dict(finish_reasons),
            'wire_configurations':list({json.dumps(item, sort_keys=True):item for item in wire}.values()),
            'draft_counts':merged['draft_counts'], 'word_path':str(word), 'word_sha256':sha(word),
            'word_bytes':word.stat().st_size, 'word_characters':len(word_text), 'word_table_rows_with_headers':word_rows,
            'actual_current_word_self_check':True, 'remaining_issues':check['model_statement']['remaining_issues'],
            'exhaustive_quality_verified':False})
    return {'complete':True, 'documents':records, 'service_context':262144,
            'preexisting_services_unchanged':True, 'owned_service_stopped':True,
            'processing_stage_seconds':result['processing_stage_seconds'], 'elapsed_seconds':result['elapsed_seconds'],
            'service_startup_seconds':result['service_startup_seconds'], 'kev_model_loads':1,
            'environment_recovery':result.get('environment_recovery', False),
            'timing_note':'processing_stage_seconds includes one Kev weight load and per-document SDK process startup; elapsed_seconds additionally includes service startup, preflight, the failed SDK-environment attempt and manual repair wait.',
            'schedule':result['schedule'], 'exhaustive_quality_verified':False}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    result = audit(args.root)
    (Path(args.root)/'verified-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))
