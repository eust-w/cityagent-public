"""Serial, exclusive-GPU6 A/B runner; no HTTP requests or main Agent calls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root, phase):
    root=Path(root);out=root/'r3';code=out/'code';runs=[]
    plan=root/'r2/mineru-plan-v3.json'
    baseline=root/'r2/code/experiments/kev_direct/refine.py'
    assert sha(baseline)=='eae1650a1e1bb685ff5be6e00e421e06ec6a71e3bb02d0ef556f80d446349348'
    assert sha(plan)=='d47b219a972d3d354b3da837221394f0a2becee7d56094f7543bc54a986da264'
    gpu=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(gpu.strip())>128:raise RuntimeError('GPU6 occupied; do not disrupt existing processes')
    freeze={'phase':phase,'start_epoch':time.time(),'gpu6_memory_before_mib':int(gpu.strip()),
        'source_hash':json.loads(plan.read_bytes())['document_sha256'],'plan_sha256':sha(plan),
        'files':{p.name:sha(p) for p in code.glob('*.py')},'main_agent_calls':0,
        'models':'Same pretrained Kev-4B BF16; CUDA Graph on; 8 request workers; only official training C1'}
    (out/f'{phase}-freeze.json').write_text(json.dumps(freeze,indent=2))
    variants=[('baseline',baseline,[]),('joint',code/'lean.py',['--strategy','joint']),
        ('grouped',code/'lean.py',['--strategy','grouped'])]
    if phase=='full':variants=[variants[2],variants[0],variants[1]]
    for name,driver,extra in variants:
        dest=out/f'{phase}-{name}'
        if dest.exists():raise FileExistsError(dest)
        argv=[str(root/'venv/bin/python'),'-B',str(driver),'--model-root',str(root),
            '--plan',str(plan),'--output',str(dest),'--gpu','6','--workers','8','--cuda-graphs',*extra]
        if phase=='pilot':argv+=['--fixtures',str(out/'classification-fixtures.json')]
        started=time.time()
        with (out/f'{phase}-{name}.log').open('x') as log:
            p=subprocess.run(argv,cwd=code,stdout=log,stderr=subprocess.STDOUT,timeout=360)
        run={'name':name,'argv':argv,'driver_sha256':sha(driver),'returncode':p.returncode,
            'started_epoch':started,'ended_epoch':time.time(),'process_seconds':time.time()-started}
        path=dest/('regression.json' if phase=='pilot' else 'candidates.json')
        if path.exists():
            result=json.loads(path.read_bytes())
            run['result']={k:v for k,v in result.items() if k not in ('entries','reads','cases')}
        runs.append(run)
        (out/f'{phase}-ab.json').write_text(json.dumps({'freeze':freeze,'runs':runs},ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in run.items() if k!='argv'},ensure_ascii=False),flush=True)
        if p.returncode:raise RuntimeError(f'{name} failed; full log preserved')
    (out/f'{phase}-done.json').write_text(json.dumps({'runs':len(runs),'ended_epoch':time.time()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',choices=['pilot','full'],required=True)
    a=p.parse_args();run(a.root,a.phase)
