"""Explicit fake-backend tests; no model inference or GPU is used here."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_evidence.contest.inference_core import (
    ContextBudgetError, INSTRUCTION, InferenceFailure, build_prompt, infer_document, prompt_ids_sha256,
)


class FakeTokenizer:
    """One character equals one token, solely for deterministic synthetic tests."""
    def __init__(self):
        self.templates = []
        self.encodings = []

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        return "SYSTEM:" + messages[0]["content"] + "|USER:" + messages[1]["content"] + "|ASSISTANT:"

    def __call__(self, text, **kwargs):
        self.encodings.append((text, kwargs))
        return {"input_ids": [ord(character) for character in text]}


class FakeBackend:
    """Returns supplied synthetic responses; calls are never presented as real model runs."""
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, ids, max_new_tokens):
        self.calls.append((ids.copy(), max_new_tokens))
        response = self.responses[len(self.calls) - 1]
        if isinstance(response, Exception):
            raise response
        return response


def report(**changes):
    return {"基本信息": {"桥梁名称": "合成测试桥"}, "维修建议": [], "病害明细": [], **changes}


def response(data=None, *, raw=None, finish="stop", output_tokens=24):
    return {"text": raw if raw is not None else json.dumps(report() if data is None else data, ensure_ascii=False),
            "finish_reason": finish, "output_tokens": output_tokens}


def call(backend, *, tokenizer=None, max_attempts=2, **kwargs):
    return infer_document("合成报告全文", tokenizer or FakeTokenizer(), backend,
                          max_model_len=5000, max_new_tokens=200, max_attempts=max_attempts, **kwargs)


class InferenceCoreTests(unittest.TestCase):
    def test_instruction_matches_rebuilt_training_data(self):
        from scripts.build_sft_dataset import INSTRUCTION as TRAINING_INSTRUCTION
        self.assertEqual(INSTRUCTION, TRAINING_INSTRUCTION)

    def test_template_matches_training_flags_and_keeps_complete_document(self):
        tokenizer = FakeTokenizer()
        text = "开头" + "正文" * 300 + "末尾"
        ids = build_prompt(tokenizer, text, max_model_len=5000, max_new_tokens=200)
        messages, flags = tokenizer.templates[0]
        self.assertEqual(messages, [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": text}])
        self.assertEqual(flags, {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})
        encoded_text, flags = tokenizer.encodings[0]
        self.assertEqual(flags, {"add_special_tokens": False, "truncation": False})
        self.assertEqual(ids, list(map(ord, encoded_text)))
        self.assertIn(text, encoded_text)

    def test_context_overflow_never_calls_backend_and_boundary_fits(self):
        tokenizer, backend = FakeTokenizer(), FakeBackend(response())
        expected = build_prompt(tokenizer, "合成报告", max_model_len=5000, max_new_tokens=10)
        self.assertEqual(build_prompt(FakeTokenizer(), "合成报告", max_model_len=len(expected) + 10,
                                      max_new_tokens=10), expected)
        with self.assertRaises(ContextBudgetError) as failure:
            infer_document("合成报告", tokenizer, backend, max_model_len=len(expected) + 9, max_new_tokens=10)
        self.assertEqual(backend.calls, [])
        self.assertEqual(failure.exception.model_calls, [])
        self.assertEqual(failure.exception.prompt_tokens, len(expected))
        self.assertEqual(failure.exception.seconds, 0)

    def test_success_keeps_empty_arrays_and_exact_backend_token_count(self):
        backend = FakeBackend(response(output_tokens=37))
        result = call(backend)
        self.assertEqual(result["data"], report())
        self.assertEqual(result["output_tokens"], 37)
        self.assertEqual(result["finish_reason"], "stop")
        record = result["model_calls"][0]
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["errors"], [])
        self.assertEqual(record["prompt_sha256"], prompt_ids_sha256(backend.calls[0][0]))
        self.assertEqual(record["output_sha256"], hashlib.sha256(result["raw_output"].encode()).hexdigest())
        self.assertEqual(record["seconds"], record["latency_seconds"])
        self.assertEqual(result["seconds"], record["seconds"])
        self.assertGreaterEqual(result["wall_seconds"], result["seconds"])

    def test_invalid_then_valid_records_both_calls_without_rule_completion(self):
        backend = FakeBackend(response(raw="not JSON"), response())
        result = call(backend)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual([row["status"] for row in result["model_calls"]], ["failed", "success"])
        self.assertEqual(result["model_calls"][0]["errors"], ["invalid_json"])
        self.assertNotEqual(result["model_calls"][0]["call_id"], result["model_calls"][1]["call_id"])
        self.assertEqual(backend.calls[0], backend.calls[1])
        self.assertEqual(result["seconds"], sum(row["seconds"] for row in result["model_calls"]))

    def test_length_finish_is_rejected_even_if_returned_json_is_valid(self):
        backend = FakeBackend(response(finish="length"), response(finish="length"))
        with self.assertRaises(InferenceFailure) as failure:
            call(backend)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(len(failure.exception.model_calls), 2)
        self.assertTrue(all(row["status"] == "failed" for row in failure.exception.model_calls))
        self.assertEqual(failure.exception.finish_reason, "length")
        self.assertEqual(failure.exception.seconds, sum(row["seconds"] for row in failure.exception.model_calls))

    def test_backend_exception_and_bad_json_are_bounded_and_both_audited(self):
        backend = FakeBackend(RuntimeError("synthetic failure with private document text"), response(raw="{"))
        with self.assertRaises(InferenceFailure) as failure:
            call(backend)
        records = failure.exception.model_calls
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(records[0]["errors"], ["backend_exception:RuntimeError"])
        self.assertIsNone(records[0]["output_tokens"])
        self.assertIsNone(records[0]["output_sha256"])
        self.assertNotIn("private document", json.dumps(records))
        self.assertEqual(records[1]["errors"], ["invalid_json"])

    def test_only_complete_fences_and_trailing_commas_are_tolerated(self):
        raw = '```json\n{"基本信息":{"桥梁名称":"测试,}桥",},"维修建议":[],"病害明细":[],}\n```'
        self.assertEqual(call(FakeBackend(response(raw=raw)))["data"]["基本信息"]["桥梁名称"], "测试,}桥")
        valid = json.dumps(report(), ensure_ascii=False)
        for raw in ("说明：" + valid, valid + "还没生成完", valid[:-1], "```json\n" + valid,
                    '{"基本信息":{"桥梁名称":"桥"},"维修建议":[],"病害明细":[{"病害类型":"裂缝"}'):
            with self.subTest(raw=raw):
                with self.assertRaises(InferenceFailure):
                    call(FakeBackend(response(raw=raw)), max_attempts=1)

    def test_missing_sections_wrong_shapes_and_meaningless_rows_fail(self):
        bad = [None, [], {}, {"基本信息": {"桥梁名称": "桥"}},
               report(基本信息=[]), report(基本信息={}), report(基本信息={"桥梁名称": "未知"}),
               report(基本信息={"桥梁名称": "???"}), report(基本信息={"other": "bridge"}),
               report(基本信息={"总体评分": True}), report(基本信息={"总体评分": {"value": 90}}),
               report(维修建议={}), report(病害明细="无"), report(维修建议=[{}]), report(病害明细=["裂缝"])]
        for data in bad:
            with self.subTest(data=data):
                backend = FakeBackend(response(raw=json.dumps(data, ensure_ascii=False)))
                with self.assertRaises(InferenceFailure):
                    call(backend, max_attempts=1)
                self.assertEqual(len(backend.calls), 1)

    def test_nonempty_rows_and_zero_numeric_basic_field_are_valid(self):
        data = report(基本信息={"总体评分": 0}, 维修建议=[{"建议内容": "修补伸缩缝"}],
                      病害明细=[{"病害类型": "裂缝", "病害描述": "纵向裂缝"}])
        self.assertEqual(call(FakeBackend(response(data)))["data"], data)

    def test_missing_nonpositive_or_fabricated_output_counts_fail(self):
        for count in (None, 0, -1, True, "24", 24.0, 201):
            with self.subTest(count=count):
                with self.assertRaises(InferenceFailure) as failure:
                    call(FakeBackend(response(output_tokens=count)), max_attempts=1)
                self.assertEqual(failure.exception.model_calls[0]["errors"], ["invalid_output_token_count"])

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        for raw in ('{"基本信息":{"桥梁名称":"桥","桥梁名称":"另桥"},"维修建议":[],"病害明细":[]}',
                    '{"基本信息":{"总体评分":NaN},"维修建议":[],"病害明细":[]}'):
            with self.assertRaises(InferenceFailure):
                call(FakeBackend(response(raw=raw)), max_attempts=1)

    def test_malformed_backend_unicode_is_a_recorded_failure(self):
        with self.assertRaises(InferenceFailure) as failure:
            call(FakeBackend(response(raw="\ud800")), max_attempts=1)
        self.assertEqual(failure.exception.model_calls[0]["errors"], ["invalid_unicode_output"])
        self.assertEqual(failure.exception.model_calls[0]["output_tokens"], 24)
        self.assertEqual(failure.exception.model_calls[0]["finish_reason"], "stop")

    def test_caller_instruction_and_backend_mutation_do_not_change_retry_ids(self):
        seen = []
        def mutating_fake(ids, max_new_tokens):
            seen.append(ids.copy())
            ids.clear()
            return response(raw="bad") if len(seen) == 1 else response()
        tokenizer = FakeTokenizer()
        result = call(mutating_fake, tokenizer=tokenizer, instruction="合成自定义指令")
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(tokenizer.templates[0][0][0]["content"], "合成自定义指令")
        self.assertEqual(result["prompt_tokens"], len(seen[0]))


if __name__ == "__main__":
    unittest.main()
