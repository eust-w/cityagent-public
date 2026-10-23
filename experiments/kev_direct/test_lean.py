"""Verify that compact requests preserve targets and source/context isolation."""
import unittest
from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from lean import classification_batches, grouped_state, joint_question
import test_refine_run
import lean
import runtime


class GroupedRequestsTest(unittest.TestCase):
    def setUp(self):
        self.items=[]
        for i in range(12):
            c={'id':f'E{i}:0:3','text':f'片段{i}','source_unit':f'原文{i}',
                'source_refs':[{'evidence_id':f'E{i}','start':0,'end':3}],
                'prior_context':['同一段较长背景 '*20], 'table_id':'T1',
                'table_headers':['部位 | 病害 | 现状']}
            self.items.append((c,{'packet':'P1','headings':['本次检测']}))

    def test_every_fragment_has_one_target_and_no_drop(self):
        packs=classification_batches(self.items,'grouped')
        self.assertEqual([c['id'] for p in packs for c,g in p],
            [c['id'] for c,g in self.items])
        for pack in packs:
            state=grouped_state(pack)
            self.assertEqual(len(state['source_fragments']),len(pack))
            for i,(c,g) in enumerate(pack):
                target=state['source_fragments'][str(i)]
                self.assertEqual(target['source_fragment'],c['text'])
                self.assertEqual(state['context_not_additional_findings']['original_units'][target['original_unit_id']],c['source_unit'])

    def test_different_packet_or_table_never_share_state(self):
        items=deepcopy(self.items[:3])
        items[1][1]['packet']='P2'
        items[2][0]['table_id']='T2'
        self.assertEqual(list(map(len,classification_batches(items,'grouped'))),[1,1,1])

    def test_conflicting_source_unit_is_rejected(self):
        items=deepcopy(self.items[:2])
        items[1][0]['source_refs'][0]['evidence_id']='E0'
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            grouped_state(items)

    def test_shared_background_appears_once(self):
        state=grouped_state(self.items[:4])
        self.assertEqual(len(state['context_not_additional_findings']['prior_original_context']),1)
        self.assertEqual(len(state['table_headers_or_candidates']),1)

    def test_questions_point_to_individual_fragment(self):
        for i in range(4):
            target=f'source_fragments[{i}].source_fragment'
            self.assertIn(target,joint_question(target)['instructions'])
            self.assertIn('uncertain',joint_question(target)['criteria'])


class LeanDriverTest(test_refine_run.ExtractionDriverTest):
    def setUp(self):
        self.original=test_refine_run.refine
        test_refine_run.refine=lean

    def tearDown(self):
        test_refine_run.refine=self.original


def synthetic_run(root, text, *, category='basic', table=False, condition='positive',
                  strategy='joint', selected_fields=None, fixtures=False):
    """Run the complete driver with in-memory classifier decisions, never weights."""
    selected_fields = selected_fields or {}
    closed = []
    replayed = []
    graph = (SimpleNamespace(graphs={'captured': True},
        _replay=lambda key, body, rows: replayed.append(key)) if fixtures else None)
    model = SimpleNamespace(graphs=graph)

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
            if model.graphs is not None:
                # Exercise the driver's real replay instrumentation with both
                # captured and eager keys, instead of fabricating its counters.
                for key in ('captured', 'uncaptured', 'captured'):
                    model.graphs._replay(key, None, None)
            answers = {}
            for key, question in request.questions.items():
                options = question['criteria']
                if 'comparison' in options:
                    choice = 'comparison'
                elif 'defect' in options:
                    choice = category
                elif 'positive' in options:
                    choice = condition
                elif request.state.get('requested_field') in selected_fields:
                    expected = selected_fields[request.state['requested_field']]
                    choice = next((k for k, value in options.items()
                        if isinstance(value, dict) and value.get('value') == expected), 'unknown')
                else:
                    choice = 'unknown' if 'unknown' in options else 'no' if 'no' in options else next(iter(options))
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
    modules['kev.model'].rows_per_pass = lambda **kwargs: None
    modules['kev'].model = modules['kev.model']
    candidate = {'id': 'E1:0:' + str(len(text)), 'text': text, 'source_unit': text,
        'source_refs': [{'evidence_id': 'E1', 'start': 0, 'end': len(text)}]}
    if table:
        candidate.update(table_id='T1', unit_kind='table_row')
    plan = {'document_sha256': 'synthetic', 'manifest_sha256': 'synthetic',
        'groups': [{'packet': 'P1', 'headings': [], 'candidates': [candidate]}]}
    (root / 'model-download.json').write_text(json.dumps({'complete': True,
        'models': [{'repo': 'jaredpalmer/kev-4b', 'path': str(root)}]}))
    (root / 'plan.json').write_text(json.dumps(plan))
    fixture_path = None
    if fixtures:
        fixture_path = root / 'fixtures.json'
        fixture_path.write_text(json.dumps({'cases': [{**candidate, 'gold': category}]}))
    with patch.dict(sys.modules, modules), patch.dict(lean.os.environ), \
            patch.object(sys, 'path', list(sys.path)), \
            patch.object(runtime, 'load_graph_gpu', return_value=(None, model, {})):
        result = lean.run(root, root / 'plan.json', root / 'output', workers=1,
                          cuda_graphs=True, strategy=strategy, fixtures=fixture_path)
    assert closed == [True]
    if fixtures:
        assert replayed == ['captured', 'uncaptured', 'captured']
    return result


@pytest.mark.parametrize('strategy', ['joint', 'compact'])
def test_comparison_table_keeps_a_model_confirmed_current_defect(tmp_path, strategy):
    text = '左幅支座 | 上次无病害 | 本次锈蚀 | 新增'
    result = synthetic_run(tmp_path, text, category='defect', table=True, strategy=strategy)
    rows = [entry for entry in result['entries'] if entry['section'] == '病害明细']
    assert len(rows) == 1 and rows[0]['row']['病害描述'] == text
    assert rows[0]['source_refs'] == [{'evidence_id': 'E1', 'start': 0, 'end': len(text)}]
    # The double declines history association: keep the current finding and
    # its unresolved comparison instead of silently losing both.
    assert '是否新增' not in rows[0]['row']
    assert result['unresolved'] == ['Historical comparison not attached to ' + rows[0]['id']]


@pytest.mark.parametrize('category,condition', [('history', 'positive'), ('defect', 'absent')])
def test_comparison_context_never_promotes_history_or_absent_condition(tmp_path, category, condition):
    result = synthetic_run(tmp_path, '左幅支座 | 上次锈蚀 | 本次完好', category=category,
                           condition=condition, table=True)
    assert not any(entry['section'] == '病害明细' for entry in result['entries'])


@pytest.mark.parametrize('strategy', ['joint', 'compact'])
@pytest.mark.parametrize('score,grade', [('95', '合格'), ('0', '不合格'), ('100', 'A级'), ('95.43', '五类')])
def test_global_scalar_candidates_keep_integer_scores_and_literal_grades(tmp_path, strategy, score, grade):
    text = f'甲桥，总体评分{score}，总体等级{grade}。'
    selected = {'总体评分': score, '总体等级': grade}
    result = synthetic_run(tmp_path, text, strategy=strategy, selected_fields=selected)
    for field, expected in selected.items():
        entries = [entry for entry in result['entries'] if entry.get('field') == field]
        assert [entry['value'] for entry in entries] == [expected]
        assert all(entry['source_refs'] == [{'evidence_id': 'E1', 'start': 0, 'end': len(text)}]
                   for entry in entries)


def test_fixture_driver_records_actual_captured_and_eager_replays(tmp_path):
    result = synthetic_run(tmp_path, '甲桥，总体评分95。', fixtures=True)
    assert result['replay_counts'] == {'captured_replays': 2, 'eager_passes': 1}
    assert result['correct'] == result['total'] == result['decision_requests'] == 1
    assert json.loads((tmp_path / 'output' / 'regression.json').read_text()) == result


if __name__=='__main__':unittest.main()
