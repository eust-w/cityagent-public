"""Synthetic checks for frozen, answer-independent generation partitions."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_generation_manifest.py"
SPEC = importlib.util.spec_from_file_location("prepare_generation_manifest", SCRIPT)
generation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generation)


def fixtures(lengths, *, year="2018", answer="gold"):
    rows, metadata = [], []
    for index, prompt_tokens in enumerate(lengths, 1):
        content = {"instruction": "extract", "input": f"report-{index}", "output": answer}
        row = {**content, "year": year, "line": index,
               "sample_sha256": generation.bench.digest(content),
               "prompt_sha256": generation.bench.digest({key: content[key] for key in ("instruction", "input")})}
        rows.append(row)
        metadata.append({key: row[key] for key in ("sample_sha256", "prompt_sha256", "year", "line")}
                        | {"prompt_tokens": prompt_tokens, "answer_tokens": len(answer),
                           "tokens": prompt_tokens + len(answer),
                           "token_sha256": generation.bench.digest([index, prompt_tokens, len(answer)]),
                           "split": "holdout" if year == "2018" else "train",
                           "status": "overlong_full_sequence"})
    return rows, metadata


def build(rows, metadata, **kwargs):
    return generation.build_manifest(rows, metadata, sft_sha256="a" * 64, **kwargs)


class GenerationManifestTests(unittest.TestCase):
    def test_prompt_bucket_boundaries_are_exhaustive(self):
        lengths = [1, 8191, 8192, 16383, 16384, 32768, 32769, 260000]
        expected = ["lt_8k"] * 2 + ["8k_to_16k"] * 2 + ["16k_to_32k"] * 2 + ["gt_32k"] * 2
        self.assertEqual([generation.prompt_length_bucket(n) for n in lengths], expected)
        for invalid in (0, -1, True, 8192.0, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                generation.prompt_length_bucket(invalid)

    def test_top_two_per_bucket_keep_all_cases_regardless_of_benchmark_status(self):
        rows, metadata = fixtures([100] * 3 + [9000] * 3 + [20000] * 3 + [160000] * 3)
        manifest = build(rows, metadata)
        self.assertEqual(manifest["partition_counts"], {"diagnostic": 8, "final_acceptance": 4})
        self.assertEqual(manifest["sample_count"], 12)
        self.assertTrue(all(sample["year"] == "2018" for sample in manifest["samples"]))
        for bucket in generation.BUCKETS:
            group = [sample for sample in manifest["samples"] if sample["prompt_length_bucket"] == bucket["name"]]
            self.assertEqual([sample["partition"] for sample in group], ["diagnostic", "diagnostic", "final_acceptance"])
            self.assertEqual([s["prompt_content_sha256"] for s in group], sorted(s["prompt_content_sha256"] for s in group))
        for index, item in enumerate(metadata):
            item["status"] = ["selected", "sample_limit", "below_length_bucket"][index % 3]
        self.assertEqual(manifest, build(rows, metadata))
        self.assertEqual(manifest, build(rows[::-1], metadata[::-1]))
        self.assertFalse(any(key in sample for sample in manifest["samples"]
                             for key in ("instruction", "input", "output", "input_ids", "labels", "prompt_sha256")))

    def test_gold_answers_full_lengths_and_seed_cannot_select_diagnostics(self):
        lengths = [12000] * 6 + [40000] * 6
        first = build(*fixtures(lengths))
        second = build(*fixtures(lengths, answer="different answer" * 5000), seed=123)
        self.assertEqual({s["prompt_content_sha256"]: s["partition"] for s in first["samples"]},
                         {s["prompt_content_sha256"]: s["partition"] for s in second["samples"]})
        self.assertEqual(first["partition_prompt_content_sha256"], second["partition_prompt_content_sha256"])
        self.assertNotEqual(first["cohort_sha256"], second["cohort_sha256"])

    def test_non_2018_rows_never_become_generation_samples(self):
        rows, metadata = fixtures([11000, 160000, 12000])
        rows[1]["year"] = metadata[1]["year"] = "2017"
        metadata[1]["split"] = "train"
        manifest = build(rows, metadata)
        self.assertEqual(manifest["source_row_count"], 3)
        self.assertEqual(manifest["sample_count"], 2)
        self.assertEqual(manifest["bucket_counts"]["gt_32k"], 0)
        self.assertNotIn(rows[1]["sample_sha256"], [s["sample_sha256"] for s in manifest["samples"]])

    def test_missing_extra_duplicate_or_mismatched_metadata_fails(self):
        rows, metadata = fixtures([12000, 40000])
        cases = [metadata[:1], metadata + metadata[:1]]
        other_rows, other_metadata = fixtures([12000, 40000, 100])
        cases.append(other_metadata)
        for key, value in (("prompt_sha256", "b" * 64), ("line", True), ("year", "2017"),
                           ("split", "train"), ("tokens", 12), ("prompt_tokens", True),
                           ("token_sha256", "not-a-sha")):
            changed = copy.deepcopy(metadata)
            changed[0][key] = value
            cases.append(changed)
        for index, bad in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                build(rows, bad)
        changed_rows = copy.deepcopy(rows)
        changed_rows[0]["output"] = "changed without changing its hash"
        with self.assertRaisesRegex(ValueError, "sample hash"):
            build(changed_rows, metadata)
        with self.assertRaisesRegex(ValueError, "duplicate SFT"):
            build(rows + rows[:1], metadata)
        with self.assertRaisesRegex(ValueError, "no 2018"):
            build(*fixtures([12000], year="2017"))

    def test_cli_binds_source_hash_asserts_count_and_never_overwrites(self):
        rows, metadata = fixtures([12000, 160000, 14000])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sft, tokens, output = base / "sft.jsonl", base / "tokens.json", base / "manifest.json"
            sft.write_text("\n".join(json.dumps({**{key: row[key] for key in ("instruction", "input", "output")},
                                                  "meta": {"year": row["year"]}}) for row in rows))
            tokens.write_text(json.dumps({"sft_sha256": "b" * 64, "audit": metadata}))
            args = ["--sft", str(sft), "--token-manifest", str(tokens), "--output", str(output)]
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                generation.main(args)
            self.assertFalse(output.exists())
            tokens.write_text(json.dumps({"sft_sha256": generation.bench.file_digest(sft), "audit": metadata}))
            with self.assertRaisesRegex(ValueError, "expected count"):
                generation.main(args + ["--expected-holdout-count", "81"])
            self.assertFalse(output.exists())
            manifest = generation.main(args + ["--expected-holdout-count", "3"])
            self.assertEqual(json.loads(output.read_text()), manifest)
            self.assertEqual(manifest["sample_count"], 3)
            with self.assertRaises(FileExistsError):
                generation.main(args)


if __name__ == "__main__":
    unittest.main()
