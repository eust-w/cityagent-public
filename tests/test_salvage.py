import json
import unittest

from bridge_evidence.contest.inference_core import _parse_report
from bridge_evidence.contest.salvage import salvage_report

BASIC = {"桥梁名称": "某桥", "总体评分": "88.15", "总体等级": "B级"}
ADVICE = [{"序号": "1", "建议类别": "尽快维修", "建议内容": "修补破损。"}]
DEFECTS = [{"序号": str(i), "病害部位": f"{i}#梁", "病害类型": "裂缝", "病害描述": f"{i}#梁裂缝"} for i in range(1, 5)]


def dump(report):
    return json.dumps(report, ensure_ascii=False)


class SalvageTests(unittest.TestCase):
    def test_complete_output_is_returned_unchanged(self):
        report = {"基本信息": BASIC, "维修建议": ADVICE, "病害明细": DEFECTS}
        data, notes = salvage_report(dump(report))
        self.assertEqual(data, report)
        self.assertEqual(notes["病害明细"], {"kept_rows": 4, "dropped_rows": 0, "cut_off": False, "loop_rows": 0})
        self.assertEqual(_parse_report(dump(data)), report)

    def test_cut_off_inside_a_defect_row_keeps_the_complete_rows(self):
        text = dump({"基本信息": BASIC, "维修建议": ADVICE, "病害明细": DEFECTS})
        cut = text[: text.index('"3#梁裂缝"') + 4]
        data, notes = salvage_report(cut)
        self.assertEqual(data["病害明细"], DEFECTS[:2])
        self.assertTrue(notes["病害明细"]["cut_off"])
        self.assertEqual(data["维修建议"], ADVICE)

    def test_wrong_separator_between_sections(self):
        # The real failure: an extra "{" between 基本信息 and 维修建议 made the whole response invalid JSON.
        text = '{"基本信息": ' + dump(BASIC) + ', {"维修建议": ' + dump(ADVICE) + ', "病害明细": ' + dump(DEFECTS) + "}"
        with self.assertRaises(Exception):
            json.loads(text)
        data, _ = salvage_report(text)
        self.assertEqual(data, {"基本信息": BASIC, "维修建议": ADVICE, "病害明细": DEFECTS})

    def test_one_broken_row_is_dropped_and_counted(self):
        rows = [dump(row) for row in DEFECTS]
        rows[1] = rows[1].replace('"病害类型": "裂缝"', '"病害类型" "裂缝"')
        text = '{"基本信息": ' + dump(BASIC) + ', "维修建议": [], "病害明细": [' + ", ".join(rows) + "]}"
        data, notes = salvage_report(text)
        self.assertEqual(len(data["病害明细"]), 3)
        self.assertEqual(notes["病害明细"]["dropped_rows"], 1)

    def test_generation_loop_rows_are_not_repeated(self):
        looped = DEFECTS[:1] * 6
        data, notes = salvage_report(dump({"基本信息": BASIC, "维修建议": [], "病害明细": looped}))
        self.assertEqual(len(data["病害明细"]), 2)
        self.assertEqual(notes["病害明细"]["loop_rows"], 4)

    def test_fenced_and_missing_section(self):
        text = "```json\n" + dump({"基本信息": BASIC, "病害明细": DEFECTS}) + "\n```"
        data, notes = salvage_report(text)
        self.assertEqual(data["维修建议"], [])
        self.assertTrue(notes["维修建议"]["missing"])

    def test_nothing_recoverable(self):
        self.assertIsNone(salvage_report('{"基本信息": {"桥梁名称": "某'))
        self.assertIsNone(salvage_report('{"基本信息": {"桥梁名称": "无"}, "病害明细": []}'))
        self.assertIsNone(salvage_report(""))


if __name__ == "__main__":
    unittest.main()
