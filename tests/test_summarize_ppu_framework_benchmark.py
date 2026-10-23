"""Offline summary checks using synthetic receipts, not model inference."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "summarize_ppu_framework_benchmark.py"
SPEC = importlib.util.spec_from_file_location("summarize_ppu_framework_benchmark", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)
EXPECTED = {"bridge_name": "合成异型梁桥", "report_date": "2024-05-16"}


def receipt(**changes):
    row = {"phase": "measured", "success": True, "messages_sha256": "prompt-hash",
           "sample_id": "sample", "repetition": 0, "finish_reason": "stop",
           "request": {"model": "alias-a", "max_tokens": 256, "seed": 0,
                       "messages": [{"role": "user", "content": "synthetic"}],
                       "chat_template_kwargs": {"enable_thinking": False}},
           "usage": {"prompt_tokens": 123, "completion_tokens": 40},
           "ttft_observed_seconds": 1.0, "end_to_end_seconds": 4.0,
           "approx_completion_tokens_per_decode_second": 10.0,
           "completion_tokens_per_end_to_end_second": 8.0,
           "content_text": '{"桥梁名称":"合成异型梁桥","报告日期":"2024-05-16"}'}
    row.update(changes)
    return row


class PPUBenchmarkSummaryTests(unittest.TestCase):
    def test_groups_separate_probe_types_and_exclude_warmup_failure_missing_usage(self):
        rows = [receipt(), receipt(end_to_end_seconds=8.0,
                                   approx_completion_tokens_per_decode_second=12.0),
                receipt(end_to_end_seconds=12.0, usage=None,
                        approx_completion_tokens_per_decode_second=999.0),
                receipt(success=False, end_to_end_seconds=100.0, error={"type": "HTTPError"}),
                receipt(phase="warmup", end_to_end_seconds=1000.0),
                receipt(request={"ignore_eos": True, "max_tokens": 256}, finish_reason="length")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            result = summary.summarize_run(root, EXPECTED)
        self.assertEqual(result["measured_request_count"], 5)
        self.assertEqual(result["warmup_request_count"], 1)
        groups = {group["probe_type"]: group for group in result["groups"]}
        fields, throughput = groups["fields"], groups["fixed_length_throughput"]
        self.assertEqual((fields["success_count"], fields["failure_count"]), (3, 1))
        self.assertEqual(fields["missing_completion_usage_count"], 1)
        self.assertEqual(fields["metrics"]["end_to_end_seconds"],
                         {"count": 3, "median": 8.0, "min": 4.0, "max": 12.0})
        self.assertEqual(fields["metrics"]["approx_completion_tokens_per_decode_second"],
                         {"count": 2, "median": 11.0, "min": 10.0, "max": 12.0})
        self.assertEqual(throughput["length_finish_count"], 1)
        self.assertEqual(throughput["field_comparisons"], [])

    def test_manual_comparison_accepts_chinese_date_but_not_bridge_variant(self):
        row = receipt(content_text='```json\n{"桥梁名称":"合成异型梁桥",'
                      '"报告日期":"二〇二四年五月十六日"}\n```')
        result = summary.compare_fields(row, EXPECTED)
        self.assertTrue(result["all_expected_fields_match"])
        self.assertFalse(result["quality_evaluated"])
        self.assertEqual(result["comparison_basis"], "human_expectation_not_verified_label")
        row["content_text"] = row["content_text"].replace("异型", "异形")
        self.assertFalse(summary.compare_fields(row, EXPECTED)["all_expected_fields_match"])
        self.assertEqual(summary.normalize("二〇一八年二二月一日", "report_date"),
                         "二〇一八年二二月一日")

    def test_length_failed_and_conflicting_outputs_cannot_match_all_expectations(self):
        for row in (receipt(finish_reason="length"), receipt(success=False),
                    receipt(content_text="invalid JSON")):
            with self.subTest(row=row):
                result = summary.compare_fields(row, EXPECTED)
                self.assertFalse(result["eligible"])
                self.assertIsNone(result["all_expected_fields_match"])
        row = receipt(content_text='{"bridge_name":"合成异型梁桥",'
                      '"report_date":"2024-05-16","报告日期":"2024-05-17"}')
        result = summary.compare_fields(row, EXPECTED)
        self.assertTrue(result["fields"]["report_date"]["ambiguous"])
        self.assertFalse(result["all_expected_fields_match"])

    def test_only_complete_json_response_or_single_fence_without_duplicate_keys_is_accepted(self):
        valid = receipt()["content_text"]
        invalid = [valid + '\n{"报告日期":"2024-05-17"}',
                   '{"broken_outer":' + valid, "prefix " + valid, valid + " trailing text",
                   '```json\n' + valid + '\n```\n```json\n{}\n```',
                   '{"桥梁名称":"wrong","桥梁名称":"合成异型梁桥","报告日期":"2024-05-16"}',
                   '{"result":' + valid + ',"unused":{"x":1,"x":2}}']
        for text in invalid:
            with self.subTest(text=text):
                result = summary.compare_fields(receipt(content_text=text), EXPECTED)
                self.assertFalse(result["eligible"])
                self.assertFalse(result["complete_response_json_valid"])
                self.assertIsNone(result["all_expected_fields_match"])
        for text in (valid, "\n  " + valid + "\n", "```json\n" + valid + "\n```"):
            with self.subTest(text=text):
                self.assertTrue(summary.compare_fields(receipt(content_text=text), EXPECTED)["all_expected_fields_match"])

    def test_fixed_length_protocol_invalid_requests_do_not_enter_performance_medians(self):
        def fixed(count=256, **changes):
            values = {"request": {**receipt()["request"], "ignore_eos": True},
                      "finish_reason": "length",
                      "usage": {"prompt_tokens": 123, "completion_tokens": count}}
            values.update(changes)
            return receipt(**values)

        rows = [fixed(end_to_end_seconds=2, approx_completion_tokens_per_decode_second=10),
                fixed(end_to_end_seconds=4, approx_completion_tokens_per_decode_second=20),
                fixed(8, end_to_end_seconds=100), fixed(300, end_to_end_seconds=200),
                fixed(finish_reason="stop", end_to_end_seconds=300),
                fixed(usage=None, end_to_end_seconds=400), fixed(success=False, end_to_end_seconds=500)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            group = summary.summarize_run(root, EXPECTED)["groups"][0]
        self.assertEqual((group["success_count"], group["failure_count"]), (6, 1))
        self.assertEqual((group["protocol_valid_count"], group["protocol_invalid_count"]), (2, 5))
        self.assertEqual(group["performance_request_count"], 2)
        self.assertEqual(group["metrics"]["end_to_end_seconds"],
                         {"count": 2, "median": 3.0, "min": 2, "max": 4})
        self.assertEqual(group["metrics"]["approx_completion_tokens_per_decode_second"]["median"], 15)
        self.assertEqual(group["completion_tokens"]["min"], 256)
        self.assertEqual(group["missing_completion_usage_count"], 1)
        for count in (True, "256", -1, None):
            self.assertFalse(summary.fixed_length_protocol(fixed(count))["protocol_valid"])

    def test_configuration_hash_excludes_only_model_alias_and_separates_changed_options(self):
        base = receipt()
        request = base["request"]
        _, original = summary.configuration(base)
        alternate_alias = receipt(request={**request, "model": "alias-b"})
        self.assertEqual(original, summary.configuration(alternate_alias)[1])
        modifications = [{"max_tokens": 128}, {"seed": 1}, {"ignore_eos": True},
                         {"chat_template_kwargs": {"enable_thinking": True}},
                         {"messages": [{"role": "user", "content": "different"}]},
                         {"stream_options": {"include_usage": False}}]
        changed = [receipt(request={**request, **modification}) for modification in modifications]
        for row in changed:
            self.assertNotEqual(original, summary.configuration(row)[1])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results.jsonl").write_text("\n".join(json.dumps(row) for row in [base, alternate_alias, *changed]))
            groups = summary.summarize_run(root, EXPECTED)["groups"]
        self.assertEqual(len(groups), 1 + len(changed))
        base_group = next(group for group in groups if group["request_config_sha256"] == original)
        self.assertEqual(base_group["request_count"], 2)
        self.assertEqual(base_group["request_models"], ["alias-a", "alias-b"])
        self.assertEqual(base_group["max_tokens_values"], [256])

    def test_missing_receipts_do_not_promote_summary_totals_to_actual_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text('{"successful_requests": 999}')
            result = summary.summarize_run(root, EXPECTED)
        self.assertEqual(result["receipt_count"], 0)
        self.assertEqual(result["groups"], [])
        self.assertTrue(any("No request receipts" in warning for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
