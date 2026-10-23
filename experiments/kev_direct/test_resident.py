"""Exercise document isolation and loader ownership without a GPU."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import lean
import runtime


class ResidentSessionTest(unittest.TestCase):
    def test_two_documents_share_one_load_and_keep_independent_results(self):
        closed=[]
        class Request:
            def __init__(self,state,questions):self.state,self.questions=state,questions
            def model_dump_json(self,**kw):return json.dumps({'state':self.state,'questions':self.questions},**kw)
        class Server:
            batches=0;batched_requests=0
            def __init__(self,*args):pass
            def answer(self,request):
                answers={k:{'choice':'other' if 'other' in q['criteria'] else 'unknown'}
                         for k,q in request.questions.items()}
                return {'answers':answers,'usage':{'input_tokens':1}}
            def close(self):closed.append(True)
        modules={n:ModuleType(n) for n in ('torch','kev','kev.api','kev.checkpoint','kev.serve','kev.model')}
        modules['torch'].set_num_threads=lambda _:None
        modules['torch'].cuda=SimpleNamespace(max_memory_allocated=lambda:0)
        modules['kev.api'].SystemOneRequest=Request
        modules['kev.checkpoint'].Checkpoint=lambda p:p
        modules['kev.serve'].Server=Server
        modules['kev.model'].rows_per_pass=lambda **kw:None
        modules['kev'].model=modules['kev.model']
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,modules), \
                patch.object(runtime,'load_graph_gpu',return_value=(None,SimpleNamespace(graphs=None),{})) as load, \
                patch.dict(lean.os.environ),patch.object(sys,'path',list(sys.path)):
            root=Path(tmp)
            (root/'model-download.json').write_text(json.dumps({'complete':True,'models':[
                {'repo':'jaredpalmer/kev-4b','path':tmp}]}))
            session={};results=[]
            for index in range(2):
                plan={'document_sha256':f'doc-{index}','manifest_sha256':f'manifest-{index}',
                      'groups':[{'headings':[],'candidates':[{'id':'E000001:0:3','text':'正常。',
                      'source_unit':'正常。','source_refs':[{'evidence_id':'E000001','start':0,'end':3}]}]}]}
                path=root/f'plan-{index}.json';path.write_text(json.dumps(plan))
                results.append(lean.run(root,path,root/f'out-{index}',workers=2,cuda_graphs=True,
                                        strategy='grouped',session=session))
            self.assertEqual(load.call_count,1)
            self.assertEqual(closed,[])
            self.assertEqual(session['documents'],2)
            self.assertEqual([r['document_sha256'] for r in results],['doc-0','doc-1'])
            self.assertEqual(results[1]['load_seconds'],0)
            self.assertEqual([r['resident_session']['document_index'] for r in results],[1,2])
            with self.assertRaisesRegex(ValueError,'identity changed'):
                lean.run(root,path,root/'wrong-gpu',gpu=5,cuda_graphs=True,session=session)
            session['server'].close();session['closed']=True
            self.assertEqual(closed,[True])


if __name__=='__main__':unittest.main()
