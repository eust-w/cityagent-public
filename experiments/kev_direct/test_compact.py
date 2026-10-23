"""Structural regressions, not a substitute for real model quality checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from compact import compact_review_bundle, grouped_questions, literal_field_options
import lean
import runtime


class CompactMaterialTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {'document_sha256':'document-a','units':[
            {'id':'E000001','text':'甲桥，评分95.43。'},
            {'id':'E000002','text':'左幅支座锈蚀。','kind':'table_row','table_id':'T1','row_index':1}]}
        self.refs = [{'evidence_id':'E000001','start':0,'end':len(self.manifest['units'][0]['text'])}]

    def evidence(self):
        return {'basic':{'section':'基本信息','field':'桥梁名称','candidate_value':'甲桥','source_refs':self.refs},
            **{f'defect-{i}':{'section':'病害明细','candidate_value':{'病害描述':'左幅支座锈蚀。'},
                'source_refs':[{'evidence_id':'E000002'}]} for i in range(100)}}

    def test_all_values_and_basics_survive_many_unresolved_questions(self):
        evidence = self.evidence();numbers={k:i for i,k in enumerate(evidence,1)}
        b,ranges = compact_review_bundle(self.manifest,evidence,numbers,{},['分类不确定: '+str(i) for i in range(200)])
        self.assertEqual(len(b['candidates']),100)
        self.assertEqual(b['basic_candidates_by_field']['桥梁名称'][0]['value'],'甲桥')
        self.assertEqual(len(b['source_map']),2)
        self.assertEqual(len(ranges),2)
        self.assertEqual(b['unresolved']['count'],200)
        self.assertNotIn('resolved',b['unresolved'])
        self.assertEqual({r['candidate'] for r in b['candidates']}|{b['basic_candidates_by_field']['桥梁名称'][0]['candidate']},set(numbers.values()))

    def test_forged_quote_and_foreign_namespace_rejected(self):
        e=self.evidence();e['basic']['source_excerpts']=[{**self.refs[0],'text':'伪造'}]
        with self.assertRaisesRegex(ValueError,'immutable'):compact_review_bundle(self.manifest,e,{k:i for i,k in enumerate(e,1)},{},[])
        e=self.evidence();e['basic']['source_refs']=[{'evidence_id':'mineru_1'}]
        with self.assertRaisesRegex(ValueError,'namespace'):compact_review_bundle(self.manifest,e,{k:i for i,k in enumerate(e,1)},{},[])

    def test_foreign_document_context_rejected(self):
        e=self.evidence();e['basic']['material_evidence']=[{'identity':{'document_sha256':'other-document'},'context':[]}]
        with self.assertRaisesRegex(ValueError,'another document'):compact_review_bundle(self.manifest,e,{k:i for i,k in enumerate(e,1)},{},[])

    def test_literal_equal_value_retains_all_origins_as_unselected_option(self):
        c={'source_refs':self.refs}
        other={'source_refs':[{'evidence_id':'E000002'}]}
        options=literal_field_options('总体评分',[('95.43',c,{}),('95.43',other,{})])
        self.assertEqual(len(options),1)
        self.assertEqual(len(options[0]['source_refs']),2)
        self.assertEqual(options[0]['status'],'uncertain')

    def test_field_questions_are_targeted_and_do_not_cross_packets(self):
        items=[]
        for i in range(3):
            c={'id':str(i),'text':str(i),'source_unit':str(i),'source_refs':[{'evidence_id':f'E{i}'}]}
            g={'packet':'P1' if i<2 else 'P2','headings':[]}
            items.append((c,g,{'part':{'type':'choice','instructions':'Select source_fragment part','criteria':{'unknown':'unknown'}}}))
        packs=grouped_questions(items,lean.grouped_state)
        self.assertEqual([len(p[2]) for p in packs],[2,1])
        self.assertIn('source_fragments[1].source_fragment',packs[0][1]['1_part']['instructions'])


class CompactDriverTest(unittest.TestCase):
    def test_compact_driver_defers_global_selection_and_batches_history(self):
        calls=[];closed=[]
        class Request:
            def __init__(self,state,questions):self.state,self.questions=state,questions
            def model_dump_json(self,**kw):return json.dumps({'state':self.state,'questions':self.questions},**kw)
        class Server:
            batches=0;batched_requests=0
            def __init__(self,*args):pass
            def answer(self,request):
                calls.append(request)
                answers={}
                for key,q in request.questions.items():
                    opts=q['criteria']
                    if 'comparison' in opts: choice='comparison'
                    elif 'defect' in opts and 'advice' in opts:
                        text=request.state['source_fragments'][key]['source_fragment']
                        choice='basic' if '甲桥' in text else 'history' if '上次' in text else 'defect'
                    elif 'positive' in opts: choice='positive'
                    elif key.endswith('applies'):choice='yes'
                    else:choice=next(k for k in opts if k!='unknown')
                    answers[key]={'choice':choice}
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
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,modules),patch.dict(lean.os.environ),\
                patch.object(sys,'path',list(sys.path)),patch.object(runtime,'load_graph_gpu',return_value=(None,SimpleNamespace(graphs=None),{})):
            root=Path(tmp)
            (root/'model-download.json').write_text(json.dumps({'complete':True,'models':[{'repo':'jaredpalmer/kev-4b','path':tmp}]}))
            candidates=[]
            for i,text in enumerate(['甲桥，评分95.43。','左幅支座锈蚀。','右幅护栏锈蚀。','上次支座锈蚀 | 本次支座锈蚀 | 无明显发展']):
                candidates.append({'id':str(i),'text':text,'source_unit':text,
                    'source_refs':[{'evidence_id':f'E{i}','start':0,'end':len(text)}],
                    **({'table_id':'T1','unit_kind':'table_row'} if i==3 else {})})
            plan={'document_sha256':'doc','manifest_sha256':'manifest','groups':[{'packet':'P1','headings':[],'candidates':candidates}]}
            (root/'plan.json').write_text(json.dumps(plan))
            r=lean.run(root,root/'plan.json',root/'out',workers=2,cuda_graphs=True,strategy='compact')
            recorded=json.loads((root/'out/calls.json').read_bytes())
            kinds=[c['kind'] for c in recorded]
            self.assertNotIn('global_basic',kinds)
            self.assertIn('row_fields_group',kinds)
            self.assertIn('history_match_group',kinds)
            self.assertIn('history_fields_group',kinds)
            self.assertTrue(any(e.get('field')=='桥梁名称' and e['value']=='甲桥' for e in r['entries']))
            self.assertEqual(sum(r['classification_counts'].values()),4)
            self.assertEqual(len([e for e in r['entries'] if e['section']=='病害明细']),2)
            self.assertEqual(closed,[True])


if __name__=='__main__':unittest.main()
