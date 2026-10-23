"""Explicit training-document batches; one resident Kev, unchanged fast merger."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from deployment import existing_path, gpu_number, service_settings


def validate_config(config):
    config = dict(config)
    config.update(service_settings(config.get('main_port'), config.get('main_model'),
                                   config.get('summary_port'), config.get('summary_model')))
    config['gpu'] = gpu_number(config.get('gpu'))
    config['kev_root'] = existing_path(config.get('kev_root', ''), directory=True)
    config['tokenizer'] = existing_path(config.get('tokenizer', ''))
    ids = [doc['id'] for doc in config['documents']]
    if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i or Path(i).name != i or i in ('.', '..') or '\\' in i for i in ids):
        raise ValueError('Document IDs must be unique safe directory names')
    for doc in config['documents']:
        existing_path(doc['plan'])
        existing_path(doc['source_folder'], directory=True)
    return config


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def extract(config, output):
    config = validate_config(config)
    import lean
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    session, records = {}, []
    try:
        for doc in config['documents']:
            t = time.monotonic()
            result = lean.run(config['kev_root'], doc['plan'], output/doc['id'],
                              gpu=config['gpu'], workers=8, cuda_graphs=True,
                              strategy=config.get('kev_strategy','grouped'), session=session)
            records.append({'id': doc['id'], 'document_sha256': result['document_sha256'],
                            'seconds': time.monotonic()-t, 'load_seconds': result['load_seconds'],
                            'input_tokens': result['input_tokens'],
                            'decision_requests': result['decision_requests'],
                            'decision_questions': result['decision_questions'],
                            'entry_counts': {s: sum(e['section']==s for e in result['entries'])
                                             for s in ('基本信息','病害明细','维修建议')},
                            'unresolved': len(result['unresolved'])})
            save(output/'progress.json', {'completed': len(records), 'records': records})
    finally:
        if session and not session.get('closed'):
            session['server'].close()
            session['closed'] = True
    result = {'complete': len(records)==len(config['documents']), 'records': records,
              'seconds': time.monotonic()-started, 'model_loads': sum(r['load_seconds']>0 for r in records),
              'load_seconds': session.get('load_seconds',0), 'subagent_calls': 0,
              'note': 'Includes one model load; no main Agent or Word in this stage.'}
    save(output/'summary.json', result)
    return result


def main_stage(config, output, preflight=False):
    config = validate_config(config)
    output.mkdir(exist_ok=False)
    records = []
    started = time.monotonic()
    for doc in config['documents']:
        directory = output/doc['id']
        cmd = [sys.executable, '-B', str(Path(__file__).with_name('merge.py')),
               '--source-folder', doc['source_folder'], '--plan', doc['plan'],
               '--candidates', str(Path(config['extraction_output'])/doc['id']/'candidates.json'),
               '--output', str(directory), '--tokenizer', config['tokenizer'],
               '--main-port', str(config['main_port']), '--main-model', config['main_model'],
               '--summary-port', str(config['summary_port']), '--summary-model', config['summary_model']]
        if preflight:
            cmd.append('--preflight')
        if config.get('compact_review'):
            cmd.append('--compact-review')
        t = time.monotonic()
        with (output/(doc['id']+'.stdout.log')).open('xb') as stdout, \
                (output/(doc['id']+'.stderr.log')).open('xb') as stderr:
            proc = subprocess.run(cmd, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL)
        result_path = directory/('preflight.json' if preflight else 'result.json')
        result = json.loads(result_path.read_bytes()) if result_path.exists() else {}
        records.append({'id':doc['id'], 'returncode':proc.returncode,
                        'seconds':time.monotonic()-t, 'result':result})
        save(output/'progress.json', {'completed': len(records), 'records': records})
        if proc.returncode or not result.get('preflight_passed' if preflight else 'complete'):
            # Preserve failures; do not hide extra attempts in throughput data.
            break
    result = {'complete': len(records)==len(config['documents']) and all(
                  not r['returncode'] and r['result'].get('preflight_passed' if preflight else 'complete')
                  for r in records), 'preflight':preflight, 'seconds':time.monotonic()-started,
              'records':records, 'main_model':config['main_model'], 'subagent_calls':0}
    save(output/'summary.json', result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['extract','preflight','main'],required=True)
    args=p.parse_args();config=json.loads(Path(args.config).read_bytes())
    config=validate_config(config)
    result=(extract(config,Path(args.output)) if args.stage=='extract' else
            main_stage(config,Path(args.output),args.stage=='preflight'))
    print(json.dumps(result,ensure_ascii=False))
    if not result['complete']:sys.exit(1)
