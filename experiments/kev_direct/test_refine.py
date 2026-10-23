"""Structural checks; factual regression decisions are evaluated by real Kev."""
import unittest

from refine import classified, focused_state, history_questions, spans, values


class FocusedContractTests(unittest.TestCase):
    def test_slices_reconstruct_original_and_retain_short_numbers(self):
        text = '桥名：甲桥，评分95.43；A级。\n0#台渗水；'
        ranges = spans(text)
        self.assertEqual(''.join(text[a:b] for a,b in ranges).replace('\n',''), text.replace('\n',''))
        self.assertIn('A级。', [text[a:b] for a,b in ranges])

    def test_decimal_point_does_not_split_score(self):
        self.assertEqual(spans('BCI=95.43，A级。'),[(0,13)])

    def test_action_keeps_object_and_comma_linked_steps(self):
        text='支座锈蚀，可先将锈迹清除，再涂刷油漆修复。'
        self.assertEqual(spans(text),[(0,len(text))])

    def test_multiline_summary_table_is_not_one_mixed_object(self):
        text='结论 | 桥台渗水；\n支座锈蚀。'
        self.assertGreater(len(spans(text,table=True)),1)

    def test_table_row_retains_previous_current_column_binding(self):
        text = '桥面系 | 无明显病害 | 左护栏锈蚀 | 新增部分病害'
        self.assertEqual(spans(text,table=True),[(0,len(text))])

    def test_current_fragment_isolated_from_context(self):
        state=focused_state({'text':'未见明显病害','source_unit':'裂缝；未见明显病害'}, {'headings':['现状']})
        self.assertEqual(state['source_fragment'],'未见明显病害')
        self.assertNotIn('source_fragments',state)

    def test_positive_admission_can_reject_wrong_category(self):
        self.assertEqual(classified({'category':{'choice':'advice'},'advice_admit':{'choice':'no'}}),'other')

    def test_field_value_options_never_generate_values(self):
        text='下部结构BCIx=94.75，A级。'
        for field in ['下部结构评分','下部结构等级']:
            for key,value in values(text,field).items():
                if key!='unknown': self.assertIn(value,text)
        self.assertIn('94.75',values(text,'下部结构评分').values())
        self.assertNotIn(text,values(text,'下部结构评分').values())

    def test_history_is_model_choice_not_copied_current_default(self):
        q=history_questions('桥面系 | 无明显病害 | 左护栏锈蚀 | 新增部分病害')
        self.assertIn('unknown',q['是否新增']['criteria'])
        self.assertIn('无明显病害',q['上一次定检状态']['criteria'].values())
        self.assertIn('applies',q)


if __name__=='__main__':unittest.main()
