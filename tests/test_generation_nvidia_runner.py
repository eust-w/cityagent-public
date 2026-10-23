"""Synthetic CPU checks for generation cohort selection and completion gates.

These tests do not load torch, a model, SDK, training data, or an online test set.
They do not establish that a particular PEFT/GPU installation can load a model.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _script("evaluate_generation_nvidia")
manifest_builder = _script("prepare_generation_manifest")


def cohort():
    """All 81 synthetic holdout cases plus two training cases, with four lengths."""
    rows, metadata = [], []
    for index in range(83):
        content = {"instruction": "Read source only", "input": f"Synthetic report {index}",
                   "output": f"GOLD-MUST-NOT-BE-A-PROMPT-{index}"}
        row = {**content, "year": "2018" if index < 81 else "2017", "line": index + 1,
               "sample_sha256": runner.bench.digest(content),
               "prompt_sha256": runner.bench.digest({k: content[k] for k in ("instruction", "input")})}
        prompt_tokens = (500, 9000, 20000, 50000)[index % 4]
        rows.append(row)
        metadata.append({k: row[k] for k in ("sample_sha256", "prompt_sha256", "line", "year")}
                        | {"prompt_tokens": prompt_tokens, "answer_tokens": 100,
                           "tokens": prompt_tokens + 100,
                           "token_sha256": runner.bench.digest([index]),
                           "status": "overlong_full_sequence" if prompt_tokens > 32768 else "sample_limit"})
    manifest = manifest_builder.build_manifest(rows, metadata, sft_sha256="a" * 64)
    return manifest, rows


class CharacterTokenizer:
    eos_token = "\U000f0000"
    eos_ids = (ord(eos_token), ord(eos_token) + 1)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        return {"input_ids": list(map(ord, text))}

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join(chr(token) for token in ids if token not in self.eos_ids)


PREDICTION = {"基本信息": {"桥梁名称": "合成桥"}, "维修建议": [], "病害明细": []}
VALID_JSON = json.dumps(PREDICTION, ensure_ascii=False)


class SelectionTests(unittest.TestCase):
    def test_explicit_probe_subset_cannot_escape_its_frozen_partition(self):
        manifest, rows = cohort()
        diagnostic = [s["sample_sha256"] for s in manifest["samples"] if s["partition"] == "diagnostic"]
        acceptance = next(s["sample_sha256"] for s in manifest["samples"] if s["partition"] == "final_acceptance")
        selected = runner.select_samples(manifest, rows, partition="diagnostic", sample_sha256=diagnostic[:2])
        self.assertEqual({s["sample_sha256"] for s, _ in selected}, set(diagnostic[:2]))
        for ids in ([], [diagnostic[0], diagnostic[0]], [acceptance], ["f" * 64]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                runner.select_samples(manifest, rows, partition="diagnostic", sample_sha256=ids)

    def test_full_81_cohort_is_disjoint_and_exhaustive_across_partitions_and_shards(self):
        manifest, rows = cohort()
        expected = {row["sample_sha256"] for row in rows if row["year"] == "2018"}
        selected = []
        for partition in ("diagnostic", "final_acceptance"):
            for shard in range(3):
                selected.extend(runner.select_samples(manifest, rows, partition=partition,
                                                      shard_index=shard, shards=3))
        counts = Counter(sample["sample_sha256"] for sample, _ in selected)
        self.assertEqual(len(counts), 81)
        self.assertEqual(set(counts), expected)
        self.assertEqual(set(counts.values()), {1})
        self.assertTrue(any(sample["prompt_tokens"] > 32768 for sample, _ in selected))

    def test_selection_order_is_invariant_to_source_and_manifest_order(self):
        manifest, rows = cohort()
        first = runner.select_samples(manifest, rows, partition="final_acceptance")
        changed = deepcopy(manifest)
        changed["samples"].reverse()
        second = runner.select_samples(changed, rows[::-1], partition="final_acceptance")
        self.assertEqual([s["prompt_content_sha256"] for s, _ in first],
                         [s["prompt_content_sha256"] for s, _ in second])

    def test_smoke_limit_applies_within_each_disjoint_shard(self):
        manifest, rows = cohort()
        limited_ids = []
        for shard in range(3):
            options = dict(partition="final_acceptance", shard_index=shard, shards=3)
            full = runner.select_samples(manifest, rows, **options)
            limited = runner.select_samples(manifest, rows, limit=2, **options)
            self.assertEqual(limited, full[:2])
            limited_ids.extend(s["sample_sha256"] for s, _ in limited)
        self.assertEqual(len(set(limited_ids)), 6)

    def test_missing_duplicate_or_training_substitution_in_manifest_is_rejected(self):
        manifest, rows = cohort()
        variants = []
        missing = deepcopy(manifest)
        missing["samples"].pop()
        variants.append(missing)
        duplicate = deepcopy(manifest)
        duplicate["samples"][-1] = deepcopy(duplicate["samples"][0])
        variants.append(duplicate)
        training = deepcopy(manifest)
        training["samples"][-1]["sample_sha256"] = rows[-1]["sample_sha256"]
        variants.append(training)
        for bad in variants:
            with self.subTest(bad=bad["samples"][-1]["sample_sha256"]), self.assertRaises(ValueError):
                runner.select_samples(bad, rows, partition="diagnostic")

    def test_binding_errors_even_in_unselected_partition_are_rejected(self):
        manifest, rows = cohort()
        target = next(i for i, sample in enumerate(manifest["samples"])
                      if sample["partition"] == "final_acceptance")
        for key, value in (("line", 9999), ("prompt_content_sha256", "b" * 64)):
            bad = deepcopy(manifest)
            bad["samples"][target][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.select_samples(bad, rows, partition="diagnostic")

    def test_unknown_partition_cannot_silently_remove_a_frozen_case(self):
        manifest, rows = cohort()
        manifest["samples"][0]["partition"] = "final_acceptance_typo"
        with self.assertRaises(ValueError):
            runner.select_samples(manifest, rows, partition="diagnostic")

    def test_duplicate_source_and_invalid_or_empty_shards_are_rejected(self):
        manifest, rows = cohort()
        with self.assertRaises(ValueError):
            runner.select_samples(manifest, rows + rows[:1], partition="diagnostic")
        for options in ({"shards": 0}, {"shards": -1}, {"shard_index": -1},
                        {"shards": 2, "shard_index": 2}, {"limit": 0}, {"limit": -1},
                        {"shards": 100, "shard_index": 99}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                runner.select_samples(manifest, rows, partition="diagnostic", **options)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def response(self, ids, budget):
        return runner.output_response(ids, self.tokenizer, eos_token_ids=self.tokenizer.eos_ids,
                                      max_new_tokens=budget)

    def infer(self, ids, budget):
        return runner.infer_document("Synthetic source", self.tokenizer,
                                     lambda _prompt, cap: self.response(ids, cap),
                                     instruction="Extract", max_model_len=10000,
                                     max_new_tokens=budget, max_attempts=1)

    def test_only_actual_configured_final_eos_counts_as_stop(self):
        for eos in self.tokenizer.eos_ids:
            ids = list(map(ord, VALID_JSON)) + [eos]
            result = self.response(ids, len(ids))
            self.assertEqual(result["text"], VALID_JSON)
            self.assertEqual(result["output_tokens"], len(ids))
            self.assertEqual(result["finish_reason"], "stop")
            self.assertEqual(self.infer(ids, len(ids))["data"], PREDICTION)
        response = self.response([self.tokenizer.eos_ids[0], ord("x")], 20)
        self.assertEqual(response["finish_reason"], "stopped_without_eos")

    def test_even_complete_json_is_rejected_without_eos_at_timeout_or_token_limit(self):
        ids = list(map(ord, VALID_JSON))
        for budget, expected_reason in ((len(ids), "length"), (len(ids) + 20, "stopped_without_eos")):
            with self.subTest(budget=budget):
                self.assertEqual(self.response(ids, budget)["finish_reason"], expected_reason)
                with self.assertRaises(runner.InferenceFailure) as caught:
                    self.infer(ids, budget)
                self.assertEqual(caught.exception.model_calls[0]["status"], "failed")
                self.assertIn("incomplete_generation:" + expected_reason, caught.exception.model_calls[0]["errors"])

    def test_eos_does_not_rescue_empty_partial_json_or_excess_token_count(self):
        for text in ("", VALID_JSON[:-1]):
            ids = list(map(ord, text)) + [self.tokenizer.eos_ids[0]]
            with self.subTest(text=text), self.assertRaises(runner.InferenceFailure):
                self.infer(ids, 1000)
        ids = list(map(ord, VALID_JSON)) + [self.tokenizer.eos_ids[0]]
        with self.assertRaises(runner.InferenceFailure) as caught:
            self.infer(ids, len(ids) - 1)
        self.assertIn("invalid_output_token_count", caught.exception.model_calls[0]["errors"])

    def test_training_prefix_and_backend_prompt_have_no_gold_answer_tokens(self):
        content = {"instruction": "Use source", "input": "Synthetic full source",
                   "output": "GOLD_SENTINEL_NEVER_SEND_TO_MODEL"}
        row = {**content, "line": 1, "year": "2018", "sample_sha256": runner.bench.digest(content),
               "prompt_sha256": runner.bench.digest({k: content[k] for k in ("instruction", "input")})}
        training = runner.bench.tokenize_row(row, self.tokenizer)
        captured = []

        def generate(prompt, budget):
            captured.append(prompt)
            return self.response(list(map(ord, VALID_JSON)) + [self.tokenizer.eos_ids[0]], budget)

        runner.infer_document(row["input"], self.tokenizer, generate, instruction=row["instruction"],
                              max_model_len=10000, max_new_tokens=1000, max_attempts=1)
        self.assertEqual(captured, [training["input_ids"][:training["prompt_tokens"]]])
        prompt = self.tokenizer.decode(captured[0], skip_special_tokens=True)
        self.assertIn(row["input"], prompt)
        self.assertIn(row["instruction"], prompt)
        self.assertNotIn("GOLD_SENTINEL", prompt)
        self.assertIn("GOLD_SENTINEL", self.tokenizer.decode(training["input_ids"], skip_special_tokens=True))


if __name__ == "__main__":
    unittest.main()
