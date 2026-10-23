"""Exercise the complete extraction driver without weights or GPU services."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import refine
import runtime


class ExtractionDriverTest(unittest.TestCase):
    def test_global_field_selection_keeps_executor_and_closes_server(self):
        closed = []

        class Request:
            def __init__(self, state, questions):
                self.state, self.questions = state, questions

            def model_dump_json(self, **kwargs):
                return json.dumps({'state': self.state, 'questions': self.questions}, **kwargs)

        class Server:
            batches = 0
            batched_requests = 0

            def __init__(self, *args):
                pass

            def answer(self, request):
                self.batches += 1
                self.batched_requests += 1
                answers = {}
                for key, question in request.questions.items():
                    options = question['criteria']
                    choice = ('other' if 'other' in options else 'unknown' if 'unknown' in options
                              else 'no' if 'no' in options else next(iter(options)))
                    answers[key] = {'choice': choice}
                return {'answers': answers, 'usage': {'input_tokens': 1}}

            def close(self):
                closed.append(True)

        modules = {name: ModuleType(name) for name in
                   ('torch', 'kev', 'kev.api', 'kev.checkpoint', 'kev.serve', 'kev.model')}
        modules['torch'].set_num_threads = lambda _: None
        modules['torch'].cuda = SimpleNamespace(max_memory_allocated=lambda: 0)
        modules['kev.api'].SystemOneRequest = Request
        modules['kev.checkpoint'].Checkpoint = lambda path: path
        modules['kev.serve'].Server = Server
        modules['kev.model'].rows_per_pass = lambda **kw: None
        modules['kev'].model = modules['kev.model']
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), \
                patch.object(runtime, 'load_graph_gpu', return_value=(None, SimpleNamespace(graphs=None), {})), \
                patch.dict(refine.os.environ), patch.object(sys, 'path', list(sys.path)):
            root = Path(tmp)
            (root/'model-download.json').write_text(json.dumps({
                'complete': True, 'models': [{'repo': 'jaredpalmer/kev-4b', 'path': str(root)}]}))
            plan = {'document_sha256': 'training', 'manifest_sha256': 'manifest',
                    'groups': [{'headings': [], 'candidates': [{
                        'id': 'E000001:0:14', 'text': '甲桥，评分95.43，A级。',
                        'source_unit': '甲桥，评分95.43，A级。',
                        'source_refs': [{'evidence_id': 'E000001', 'start': 0, 'end': 14}]}]}]}
            (root/'plan.json').write_text(json.dumps(plan))
            result = refine.run(root, root/'plan.json', root/'output', workers=2, cuda_graphs=True)
            calls = json.loads((root/'output/calls.json').read_bytes())
            self.assertGreater(sum(c['kind'] == 'global_basic' for c in calls), 1)
            self.assertEqual(len(calls), result['decision_requests'])
            self.assertEqual(len(calls), len(list((root/'output').glob('response-*.json'))))
            self.assertTrue((root/'output/candidates.json').exists())
            self.assertEqual(result['entries'], [])
        self.assertEqual(closed, [True])


if __name__ == '__main__':
    unittest.main()
