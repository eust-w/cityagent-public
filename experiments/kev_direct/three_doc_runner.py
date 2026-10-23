"""Own an explicitly selected GPU in two phases; never alter any pre-existing model service."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from deployment import deployment_settings, gpu_number, require_same_deployment


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2))


def fetch(port,path):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=10) as f:
        text=f.read().decode()
    return json.loads(text) if path=='/v1/models' else text


def idle(port):
    lines=[s for s in fetch(port,'/metrics').splitlines()
           if s.startswith(('vllm:num_requests_running','vllm:num_requests_waiting'))]
    return bool(lines) and all(float(s.rsplit(' ',1)[1])==0 for s in lines)


def gpu_used(gpu):
    gpu = gpu_number(gpu)
    return int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
                                       '--format=csv,noheader,nounits'],text=True).strip())


def services():
    result={}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            raw=(p/'cmdline').read_bytes();args=raw.decode().split('\0')
            if 'vllm.entrypoints.openai.api_server' not in args:continue
            result[args[args.index('--port')+1]]={'pid':int(p.name),
                'start_ticks':(p/'stat').read_text().rsplit(')',1)[1].split()[19],
                'argv_sha256':hashlib.sha256(raw).hexdigest()}
        except (OSError,ValueError,IndexError,UnicodeError):pass
    return result


def run(root,reuse_extraction=False,compact=False):
    deployment=deployment_settings(require_inputs=True)
    root=Path(root).resolve();started=time.monotonic();started_epoch=time.time()
    freeze=json.loads((root/'freeze.json').read_bytes())
    for name,digest in freeze['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    if reuse_extraction:
        require_same_deployment(json.loads((root/'config.json').read_bytes()), deployment)
    kev_root=Path(deployment['KEV_ROOT'])
    gpu,port=deployment['GPU'],deployment['MAIN_PORT']
    input_root=Path(deployment['INPUT_ROOT'])
    sdk=deployment['SDK_PYTHON']
    inference=deployment['INFERENCE_PYTHON']
    code=root/'code/experiments/kev_direct'
    env=dict(os.environ,PYTHONPATH=str(root/'code/src'),TOKENIZERS_PARALLELISM='false')
    suffix='-002' if reuse_extraction else ''
    before=services();save(root/('services-before'+suffix+'.json'),before)
    assert str(port) not in before and gpu_used(gpu)<100,'Selected GPU or port is occupied; no existing process will be stopped'
    docs=[]

    def execute(name,cmd):
        start=time.monotonic()
        with (root/(name+'.stdout.log')).open('xb') as stdout, (root/(name+'.stderr.log')).open('xb') as stderr:
            proc=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
        receipt={'returncode':proc.returncode,'seconds':time.monotonic()-start,'argv':cmd}
        save(root/(name+'-execution.json'),receipt)
        if proc.returncode:raise RuntimeError(name+' failed; original logs preserved')
        return receipt

    execute('sdk-preflight'+suffix,[sdk,'-B','-c',
        "import importlib.metadata,json;import jsonschema,tongagents;"
        "from bridge_evidence.contest.document_speed_workflow import SpeedMergeWorkflow;"
        "assert importlib.metadata.version('tongagents')=='2.7.20';"
        "print(json.dumps({'imports_passed':True,'tongagents':'2.7.20','generation_requests':0}))"])
    if not reuse_extraction:
        (root/'plans').mkdir(exist_ok=False)
    for doc_id in deployment['document_ids']:
        folder=input_root/doc_id
        manifest=json.loads((folder/'source-projection.json').read_bytes())
        assert hashlib.sha256((folder/'source.docx').read_bytes()).hexdigest()==manifest['document_sha256']
        plan=root/'plans'/(doc_id+'.json')
        if not reuse_extraction:
            execute('prepare-'+doc_id,[sdk,'-B',str(code/'lean.py'),'--prepare','--source-folder',str(folder),'--output',str(plan)])
        prepared=json.loads(plan.read_bytes())
        assert prepared['document_sha256']==manifest['document_sha256']
        docs.append({'id':doc_id,'source_folder':str(folder),'plan':str(plan),
                     'document_sha256':manifest['document_sha256'],
                     'fragments':sum(len(g['candidates']) for g in prepared['groups'])})
    config={'documents':docs,'kev_root':str(kev_root),'gpu':gpu,'main_port':port,
            'main_model':deployment['MAIN_MODEL'],
            'summary_port':deployment['SUMMARY_PORT'],'summary_model':deployment['SUMMARY_MODEL'],
            'tokenizer':deployment['TOKENIZER_PATH'],
            'deployment_identity':deployment['identity'],
            'extraction_output':str(root/'extraction')}
    if compact:
        config.update(kev_strategy='compact',compact_review=True)
    if reuse_extraction:
        assert json.loads((root/'config.json').read_bytes())==config
        extraction=json.loads((root/'extraction/summary.json').read_bytes())
        assert extraction['complete'] and extraction['model_loads']==1
        for doc in docs:
            candidate=root/'extraction'/doc['id']/'candidates.json'
            assert json.loads(candidate.read_bytes())['document_sha256']==doc['document_sha256']
        save(root/'reused-extraction.json',{'candidate_sha256':{
            d['id']:hashlib.sha256((root/'extraction'/d['id']/'candidates.json').read_bytes()).hexdigest()
            for d in docs},'note':'Completed extraction reused; no second Kev load or extraction.'})
    else:
        save(root/'config.json',config)
        save(root/'status.json',{'stage':'kev-extraction','documents':docs,'started_epoch':started_epoch})
        execute('extract',[deployment['KEV_PYTHON'],'-B',str(code/'batch.py'),'--stage','extract',
                           '--config',str(root/'config.json'),'--output',str(root/'extraction')])
    for _ in range(30):
        if gpu_used(gpu)<100:break
        time.sleep(1)
    assert gpu_used(gpu)<100,'Kev process exited but the selected GPU has not been released'
    cmd=list(deployment['launch_argv'])
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
    service_env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',MAX_JOBS='4',
                     VLLM_CACHE_ROOT=str(root/'service-cache'))
    service_env['PATH']=str(Path(inference).parent)+os.pathsep+service_env.get('PATH','')
    service_started=time.monotonic();service_dir=root/('service-main'+suffix);service_dir.mkdir(exist_ok=False)
    with (service_dir/'stdout.log').open('xb') as stdout,(service_dir/'stderr.log').open('xb') as stderr:
        proc=subprocess.Popen(cmd,env=service_env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
    ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    save(service_dir/'launch.json',{'pid':proc.pid,'start_ticks':ticks,'argv':cmd,'gpu':gpu,'epoch':time.time()})
    save(root/'status.json',{'stage':'main-service-startup','started_epoch':started_epoch})
    try:
        until=time.monotonic()+600
        while time.monotonic()<until:
            if proc.poll() is not None:raise RuntimeError('Owned main service startup failed')
            try:
                models=fetch(port,'/v1/models')
                if models['data'][0]['id']==config['main_model'] and idle(port):break
            except Exception:pass
            time.sleep(2)
        else:raise TimeoutError('Owned main service startup exceeded 600 seconds')
        startup_seconds=time.monotonic()-service_started
        assert models['data'][0]['max_model_len']==262144
        save(service_dir/'ready.json',{'models':models,'startup_seconds':startup_seconds,'epoch':time.time()})
        execute('main-preflight'+suffix,[sdk,'-B',str(code/'batch.py'),'--stage','preflight',
                 '--config',str(root/'config.json'),'--output',str(root/('main-preflight'+suffix))])
        save(root/'status.json',{'stage':'main-review','started_epoch':started_epoch})
        execute('main',[sdk,'-B',str(code/'batch.py'),'--stage','main',
                 '--config',str(root/'config.json'),'--output',str(root/'main')])
        extraction=json.loads((root/'extraction/summary.json').read_bytes())
        main=json.loads((root/'main/summary.json').read_bytes())
        assert main['complete'] and extraction['complete'] and extraction['model_loads']==1
        result={'complete':True,'documents':docs,'extraction':extraction,'main':main,
                'service_startup_seconds':startup_seconds,'current_attempt_seconds':time.monotonic()-started,
                'elapsed_seconds':time.time()-json.loads((root/'launch.json').read_bytes())['epoch'],
                'processing_stage_seconds':extraction['seconds']+main['seconds'],
                'kev_load_seconds':extraction['load_seconds'],'model':config['main_model'],'subagent_calls':0,
                'gpu':gpu,'schedule':'same GPU, all Kev extraction then all main Agent reviews',
                'service_directory':str(service_dir),'service_snapshot_suffix':suffix,
                'environment_recovery':reuse_extraction,
                'not_a_simultaneously_resident_pipeline':True,'exhaustive_quality_verified':False}
        save(root/'result.json',result)
        save(root/'status.json',{'stage':'completed','seconds':result['elapsed_seconds']})
    finally:
        if proc.poll() is None:
            # Stop only this owned process after actual idle evidence.
            if idle(port):
                current=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
                assert current==ticks and services()[str(port)]['pid']==proc.pid
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=40)
                except subprocess.TimeoutExpired:pass
                save(service_dir/'stop.json',{'pid':proc.pid,'returncode':proc.poll(),'idle_before_stop':True,'epoch':time.time()})
            else:
                save(service_dir/'stop-blocked.json',{'reason':'service not idle; no interrupt sent','epoch':time.time()})
        after=services();save(root/('services-after'+suffix+'.json'),after)
        assert all(after.get(port)==identity for port,identity in before.items()),'A pre-existing service identity changed'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--reuse-extraction',action='store_true')
    p.add_argument('--compact',action='store_true')
    a=p.parse_args()
    try:run(a.root,a.reuse_extraction,a.compact)
    except Exception as exc:
        if Path(a.root).is_dir():
            save(Path(a.root)/'runner-error.json',{'type':type(exc).__name__,'message':str(exc),'epoch':time.time()})
            save(Path(a.root)/'status.json',{'stage':'failed','error_type':type(exc).__name__})
        raise
