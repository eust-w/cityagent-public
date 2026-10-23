"""Prove capture AND replay on the same declared regressions, not a flag."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.root);output=Path(a.output);output.mkdir(exist_ok=False)
os.environ.update(CUDA_VISIBLE_DEVICES='6',HF_HUB_CACHE=str(root/'hf-cache'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    KEV_PREFIX_CACHE='1',KEV_PREFIX_MAX_TOKENS='2048')
sys.path.insert(0,str(root/'kev'))
import torch
from kev.api import SystemOneRequest
from kev.checkpoint import Checkpoint
from kev.serve import Server
from runtime import load_graph_gpu
from refine import focused_state,classify_questions,classified
torch.set_num_threads(4)
d=json.loads((root/'model-download.json').read_text())
ck=Checkpoint(next(m['path'] for m in d['models'] if m['repo']=='jaredpalmer/kev-4b'))
started=time.monotonic();server=None;result={'success':False,'cuda_graphs_requested':True}
try:
    tok,model,deployment=load_graph_gpu(ck)
    result['deployment']=deployment
    replay_counts={'captured_replays':0,'eager_passes':0}
    original_replay=model.graphs._replay
    def counted_replay(key,body,rows):
        replay_counts['captured_replays' if key in model.graphs.graphs else 'eager_passes']+=1
        return original_replay(key,body,rows)
    model.graphs._replay=counted_replay
    server=Server(ck,tok,model,'cuda')
    fixtures=json.loads((root/'single-focus-fixtures.json').read_text())['cases']
    reqs=[SystemOneRequest(state=focused_state({'text':c['text']},{'headings':[]}),
        questions=classify_questions()) for c in fixtures]
    records=[]
    for iteration in range(3):
        t=time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses=list(pool.map(server.answer,reqs))
        server.wait_idle()
        correct=sum(classified(r['answers'])==c['gold'] for c,r in zip(fixtures,responses))
        records.append({'iteration':iteration,'seconds':time.monotonic()-t,
            'correct':correct,'total':len(fixtures),'responses':responses,
            'graphs':model.graphs.stats(),'batches':server.batches,
            'batched_requests':server.batched_requests,'replay_counts':dict(replay_counts)})
        result['records']=records
        (output/'progress.json').write_text(json.dumps(records[-1],ensure_ascii=False,indent=2))
    result.update(graphs=model.graphs.stats(),failed_buckets={str(k):v[0] for k,v in model.graphs.failed.items()},
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        replay_counts=replay_counts,
        success=model.graphs.captures>0 and replay_counts['captured_replays']>0 and not model.graphs.failed
            and all(r['correct']==len(fixtures) for r in records))
except Exception as exc:
    import traceback
    result['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
finally:
    if server is not None:server.close()
    result['seconds']=time.monotonic()-started
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','failed_buckets')},ensure_ascii=False))
