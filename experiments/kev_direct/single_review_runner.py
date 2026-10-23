"""One official-training main review, reusing immutable Kev candidates.

Starts only an owned main service on an explicitly selected idle GPU, and stops it only after idle.
No Kev extraction, reference-answer input, or additional documents are run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from three_doc_runner import save, fetch, idle, gpu_used, services, deployment_settings
from deployment import service_settings, require_same_deployment


def validate_training_replay(candidate, plan, document):
    # `training` describes weight training, not whether a source is in the
    # official training split. Preserve both independent assertions.
    assert candidate['training'] is False, 'Replay must use pretrained extraction, not training'
    assert plan['official_training_only'] is True, 'Source must be explicitly official training'
    assert candidate['document_sha256']==plan['document_sha256']==document['document_sha256']
    assert candidate['subagent_calls']==0


def execute(root, name, argv, env):
    started=time.monotonic()
    with (root/(name+'.stdout.log')).open('xb') as out, (root/(name+'.stderr.log')).open('xb') as err:
        process=subprocess.run(argv,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL)
    record={'returncode':process.returncode,'seconds':time.monotonic()-started,'argv':argv}
    save(root/(name+'-execution.json'),record)
    if process.returncode:raise RuntimeError(name+' failed; original logs preserved')
    return record


def run(root, baseline, document_id):
    deployment=deployment_settings()
    gpu,port=deployment['GPU'],deployment['MAIN_PORT']
    root,baseline=Path(root).resolve(),Path(baseline).resolve()
    started=time.monotonic()
    freeze=json.loads((root/'freeze.json').read_bytes())
    for name,digest in freeze['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    config=json.loads((baseline/'config.json').read_bytes())
    require_same_deployment(config, deployment)
    expected=service_settings(config.get('main_port'),config.get('main_model'),
                              config.get('summary_port'),config.get('summary_model'))
    assert all(expected[key]==deployment[key.upper()] for key in expected),'Replay service identity differs from the baseline configuration'
    assert config['tokenizer']==deployment['TOKENIZER_PATH'],'Replay tokenizer differs from baseline configuration'
    matches=[d for d in config['documents'] if d['id']==document_id]
    assert len(matches)==1
    doc=matches[0]
    candidate=baseline/'extraction'/document_id/'candidates.json'
    data=json.loads(candidate.read_bytes())
    validate_training_replay(data,json.loads(Path(doc['plan']).read_bytes()),doc)
    save(root/'reused-extraction.json',{'document_id':document_id,
        'document_sha256':data['document_sha256'],'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
        'candidate_path':str(candidate),'original_kev_seconds':data['total_seconds'],
        'new_kev_requests':0,'reference_answers_provided':False})
    before=services();save(root/'services-before.json',before)
    assert gpu_used(gpu)<100 and str(port) not in before,'Selected GPU or port occupied; do not interrupt other services'
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
    sdk=deployment['SDK_PYTHON']
    inference=deployment['INFERENCE_PYTHON']
    code=root/'code'
    env=dict(os.environ,PYTHONPATH=str(code/'src'),TOKENIZERS_PARALLELISM='false')
    execute(root,'sdk-preflight',[sdk,'-B','-c',
        "import importlib.metadata,jsonschema,sys;from copy import deepcopy;"
        "from bridge_evidence.contest.document_speed_workflow import SpeedMergeWorkflow;"
        "from bridge_evidence.contest.tong_react_adapter import _select_generation_tools;"
        "assert importlib.metadata.version('tongagents')=='2.7.20';"
        "sys.path.insert(0,"+repr(str(code/'experiments/kev_direct'))+");"
        "from dedup_workflow import DedupReviewMixin;"
        "base=type('Base',(),{'_before_continuous_request':lambda self,p:deepcopy(p)});"
        "h=type('Harness',(DedupReviewMixin,base),{'deduplicate_review':True})();"
        "payload={'messages':[{'role':'user','content':'check'}],'max_tokens':1024,"
        "'tools':[{'type':'function','function':{'name':'complete_review',"
        "'description':'registered','parameters':{'type':'object'}}}]};"
        "assert _select_generation_tools(payload,h._before_continuous_request)==payload"],env)
    argv=list(deployment['launch_argv'])
    service_env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',
        MAX_JOBS='4',VLLM_CACHE_ROOT=str(root/'service-cache'))
    service_env['PATH']=str(Path(inference).parent)+os.pathsep+service_env.get('PATH','')
    folder=root/'service-main';folder.mkdir(exist_ok=False)
    service_started=time.monotonic()
    with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
        process=subprocess.Popen(argv,env=service_env,stdout=out,stderr=err,
            stdin=subprocess.DEVNULL,start_new_session=True)
    ticks=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    save(folder/'launch.json',{'pid':process.pid,'start_ticks':ticks,'argv':argv,'gpu':gpu,'epoch':time.time()})
    save(root/'status.json',{'stage':'main-service-startup','document_id':document_id})
    try:
        until=time.monotonic()+600
        while time.monotonic()<until:
            if process.poll() is not None:raise RuntimeError('Owned main service startup failed')
            try:
                models=fetch(port,'/v1/models')
                if models['data'][0]['id']==config['main_model'] and idle(port):break
            except Exception:pass
            time.sleep(2)
        else:raise TimeoutError('Owned main service startup exceeded 600 seconds')
        assert models['data'][0]['max_model_len']==262144
        startup=time.monotonic()-service_started
        save(folder/'ready.json',{'startup_seconds':startup,'models':models,'epoch':time.time()})
        command=[sdk,'-B',str(code/'experiments/kev_direct/merge.py'),
            '--source-folder',doc['source_folder'],'--plan',doc['plan'],
            '--candidates',str(candidate),'--tokenizer',config['tokenizer'],
            '--main-port',str(port),'--main-model',config['main_model'],
            '--summary-port',str(config['summary_port']),'--summary-model',config['summary_model'],
            '--compact-review','--deduplicate-review']
        execute(root,'main-preflight',command+['--output',str(root/'preflight'),'--preflight'],env)
        assert json.loads((root/'preflight/preflight.json').read_bytes())['model_requests']==0
        save(root/'status.json',{'stage':'main-review','document_id':document_id,'epoch':time.time()})
        actual=execute(root,'main',command+['--output',str(root/'main')],env)
        result=json.loads((root/'main/result.json').read_bytes())
        save(root/'result.json',{'complete':result['complete'],'document_id':document_id,
            'main':result,'main_process_seconds':actual['seconds'],'service_startup_seconds':startup,
            'new_kev_requests':0,'reference_answers_provided':False,
            'cold_attempt_seconds':time.monotonic()-started,
            'comparison':'Main-review-only replay using exact same baseline candidates; no full pipeline rerun.'})
        save(root/'status.json',{'stage':'completed' if result['complete'] else 'failed','document_id':document_id})
    finally:
        if process.poll() is None:
            first=idle(port)
            time.sleep(1)
            second=idle(port)
            if first and second:
                current=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
                assert current==ticks and services()[str(port)]['pid']==process.pid
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=40)
                except subprocess.TimeoutExpired:pass
                save(folder/'stop.json',{'pid':process.pid,'returncode':process.poll(),
                    'two_idle_checks':True,'epoch':time.time()})
            else:save(folder/'stop-blocked.json',{'reason':'Owned service busy; no interruption sent'})
        after=services();save(root/'services-after.json',after)
        assert all(after.get(port)==identity for port,identity in before.items()),'Pre-existing service changed'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--document-id',required=True)
    a=p.parse_args()
    try:run(a.root,a.baseline,a.document_id)
    except Exception as exc:
        if Path(a.root).is_dir():
            save(Path(a.root)/'runner-error.json',{'type':type(exc).__name__,'message':str(exc),'epoch':time.time()})
            save(Path(a.root)/'status.json',{'stage':'failed','error_type':type(exc).__name__})
        raise
