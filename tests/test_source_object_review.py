"""Synthetic contracts only: no reports, labels, network, SDK or real generation."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from bridge_evidence.contest.source_object_dispositions import SourceObjectDispositions, occurrence_id
from bridge_evidence.contest.run_state import json_digest
from bridge_evidence.contest.source_object_review import SourceObjectReview, SourceObjectReviewError, REVIEW_SYSTEM_PROMPT


def count_tokens(messages):
    # Deliberately simple deterministic fake tokenizer; include roles and framing.
    # Production MUST supply the actual tokenizer/template admission callback.
    return 13 + sum(7 + len(m["role"]) + (len(m["content"].encode("utf-8")) + 3) // 4 for m in messages)


class Fixture:
    def __init__(self, tmp_path, size=4):
        self.directory = tmp_path
        self.receipts = {}
        self.calls = []
        self.failure_at = None
        self.mutate = None
        self.identity = {"model": "SYNTHETIC-OWN-MODEL", "source_sha256": json_digest("synthetic docx")}
        self.units = {"E1": "桥甲 2018 年公式说明，原始符号仍未解析。", "E2": "桥甲 2018 年本次检查评分为 95，提取正文给出的分数，不重算。"}
        self.refs = {key: {"evidence_id": key, "start": 0, "end": len(text)} for key, text in self.units.items()}
        self.occurrences = [{"source_sha256": self.identity["source_sha256"], "locator": f"word/document.xml/p[{i}]/object[1]",
                             "kind": "ole", "raw_sha256": json_digest("same raw"), "native_sha256": json_digest("native"),
                             "cache_sha256": None, "context_refs": [self.refs["E1"]]} for i in range(size)]
        self.ledger = self.reload()
        self.context = {"task": {"name": "extract reported score", "scope": "bridge甲2018"},
                        "draft": {"basic": {"score": 95}}, "facts": {"F1": {"value": 95, "source_refs": [self.refs["E2"]]}},
                        "output_paths": ["basic.score"], "read_refs": list(self.refs.values()),
                        "catalog_sha256": json_digest("draft catalog")}
        self.ledger.bind_context(**self.context)
        self.pids = [self.propose(i) for i in range(size)]

    def reload(self):
        return SourceObjectDispositions(self.directory, self.identity, self.occurrences,
                                        self.units, self.receipts.__getitem__)

    def propose(self, index, **changes):
        proposal = {"disposition": "not_relied_on", "rationale": "Independent review must check whether stated score suffices.",
                    "affected_output_paths": ["basic.score"], "supporting_refs": [self.refs["E2"]],
                    "context_refs": [self.refs["E1"]], "source_limitations": ["Native formula semantics remain unknown."],
                    "origin_call_ids": ["SYNTHETIC-agent-origin"]}
        proposal.update(changes)
        return self.ledger.propose(occurrence_id(self.occurrences[index]), proposal)

    def generate(self, messages, output_tokens):
        self.calls.append({"messages": deepcopy(messages), "output_tokens": output_tokens})
        if self.failure_at == len(self.calls):
            raise OSError("secret /private/path endpoint credentials must not be echoed")
        assignment = json.loads(messages[-1]["content"])
        decision = {"assignment_id": assignment["assignment_id"], "decisions": [
            {"proposal_id": row["proposal_id"], "object_id": row["object_id"], "verdict": "accepted",
             "explanation": "Synthetic reviewer agrees with proposed extraction after checking supplied context.",
             "unsupported_claims": [], "missing_items": [], "conflicts": [], "unresolved_items": []}
            for row in assignment["proposals"]]}
        call_id = f"SYNTHETIC-review-{len(self.calls)}"
        request = {"model": self.identity["model"], "messages": deepcopy(messages), "max_tokens": output_tokens, "stream": False}
        response = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(decision)}}]}
        audit = {"call_id": call_id, "request_id": "dispatch-" + call_id, "status": "success", "generation_attempted": True,
                 "response_validated": True, "http_status": 200, "finish_reason": "stop", "request_kind": "generation",
                 "endpoint": "/v1/chat/completions", "model": self.identity["model"],
                 "request_sha256": json_digest(request), "response_sha256": json_digest(response)}
        bundle = {"audit": audit, "request": request, "response": response}
        if self.mutate:
            self.mutate(bundle, decision)
        self.receipts[call_id] = bundle
        return call_id

    def coordinator(self, **kw):
        options = dict(model=self.identity["model"], max_model_len=30000, output_tokens=4096,
                       history_reserve_tokens=128, count_tokens=count_tokens, generate_review=self.generate)
        options.update(kw)
        return SourceObjectReview(self.ledger, **options)


def rehash(bundle):
    bundle["audit"]["request_sha256"] = json_digest(bundle["request"])
    bundle["audit"]["response_sha256"] = json_digest(bundle["response"])


def test_full_assignment_and_fixed_system_are_sent_and_all_objects_remain_unverified(tmp_path):
    env = Fixture(tmp_path)
    expected = env.ledger.review_payload(env.pids)
    coverage = env.coordinator().review_pending(env.pids)
    assert len(env.calls) == 1
    assert env.calls[0]["messages"][0] == {"role": "system", "content": REVIEW_SYSTEM_PROMPT}
    assert json.loads(env.calls[0]["messages"][-1]["content"]) == expected
    assert expected["proposals"][0]["source"][1]["text"] == env.units["E2"]
    assert coverage["accepted_count"] == 4 and coverage["blocking_count"] == 0
    assert coverage["unverified_representation_count"] == 4
    assert coverage["all_source_semantics_verified"] is False
    assert env.reload().coverage() == coverage


def test_actual_prompt_plus_output_and_history_reserve_boundaries(tmp_path):
    env = Fixture(tmp_path, size=1)
    messages = [{"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(env.ledger.review_payload(env.pids), ensure_ascii=False, sort_keys=True, separators=(",", ":"))}]
    exact = count_tokens(messages) + 1024 + 89
    with pytest.raises(SourceObjectReviewError) as caught:
        env.coordinator(max_model_len=exact - 1, output_tokens=1024, history_reserve_tokens=89).review_pending(env.pids)
    assert caught.value.code == "REVIEW_ITEM_OVER_BUDGET" and not env.calls
    env.coordinator(max_model_len=exact, output_tokens=1024, history_reserve_tokens=89).review_pending(env.pids)
    assert env.calls[0]["output_tokens"] == 1024
    assert count_tokens(env.calls[0]["messages"]) + 1024 + 89 == exact


def test_output_capacity_forces_singleton_batches_with_spacious_input(tmp_path):
    env = Fixture(tmp_path, size=3)
    env.coordinator(output_tokens=512).review_pending(env.pids)
    assert len(env.calls) == 3
    assert all(len(json.loads(c["messages"][-1]["content"])["proposals"]) == 1 for c in env.calls)
    assert all(c["output_tokens"] == 512 for c in env.calls)


def test_tiny_output_rejected_before_generation(tmp_path):
    env = Fixture(tmp_path, size=1)
    with pytest.raises(SourceObjectReviewError) as caught:
        env.coordinator(output_tokens=32).review_pending(env.pids)
    assert caught.value.diagnostic["minimum_output_tokens"] > 32
    assert not env.calls


def test_later_overlong_item_preflights_whole_plan_without_prefix_generation(tmp_path):
    env = Fixture(tmp_path, size=2)
    env.pids[1] = env.propose(1, rationale="长上下文不可被裁掉。" * 10000)
    with pytest.raises(SourceObjectReviewError) as caught:
        env.coordinator(max_model_len=16000).review_pending(env.pids)
    assert caught.value.code == "REVIEW_ITEM_OVER_BUDGET"
    assert caught.value.diagnostic["proposal_ids"] == [env.pids[1]]
    assert not env.calls and env.ledger.coverage()["accepted_count"] == 0


def test_network_failure_preserves_completed_batch_and_rebuild_skips_it(tmp_path):
    env = Fixture(tmp_path, size=3)
    env.failure_at = 2
    reviewer = env.coordinator(output_tokens=512)
    with pytest.raises(SourceObjectReviewError) as caught:
        reviewer.review_pending(env.pids)
    assert caught.value.code == "REVIEW_GENERATION_FAILED"
    assert "secret" not in str(caught.value) and "private" not in repr(caught.value.diagnostic)
    assert len(env.calls) == 2 and env.ledger.coverage()["accepted_count"] == 1
    assert [a["status"] for a in reviewer.attempts] == ["recorded", "generation_failed"]
    first_pid = json.loads(env.calls[0]["messages"][-1]["content"])["proposals"][0]["proposal_id"]
    env.ledger = env.reload()
    env.failure_at = None
    result = env.coordinator(output_tokens=512).review_pending(env.pids)
    assert result["accepted_count"] == 3 and len(env.calls) == 4
    later = [r["proposal_id"] for c in env.calls[2:] for r in json.loads(c["messages"][-1]["content"])["proposals"]]
    assert first_pid not in later


@pytest.mark.parametrize("change", ["missing_decision", "duplicate_decision", "wrong_object", "ack_only"])
def test_each_model_decision_required_not_assignment_acknowledgement(tmp_path, change):
    env = Fixture(tmp_path, size=2)
    def mutate(bundle, decision):
        if change == "missing_decision":
            decision["decisions"].pop()
        elif change == "duplicate_decision":
            decision["decisions"].append(deepcopy(decision["decisions"][0]))
        elif change == "wrong_object":
            decision["decisions"][0]["object_id"] = "so_wrong"
        else:
            decision.pop("decisions")
        bundle["response"]["choices"][0]["message"]["content"] = json.dumps(decision)
        rehash(bundle)
    env.mutate = mutate
    with pytest.raises(SourceObjectReviewError) as caught:
        env.coordinator().review_pending(env.pids)
    assert caught.value.code == "REVIEW_RECEIPT_REJECTED"
    assert len(env.calls) == 1 and env.ledger.coverage()["accepted_count"] == 0


@pytest.mark.parametrize("change", ["intermediate", "length", "hash", "wrong_system", "wrong_output", "missing_source", "wrong_model"])
def test_full_terminal_sidecars_and_actual_admitted_request_are_required(tmp_path, change):
    env = Fixture(tmp_path, size=1)
    def mutate(bundle, decision):
        if change == "intermediate":
            bundle["audit"]["status"] = "response_validated"
        elif change == "length":
            bundle["response"]["choices"][0]["finish_reason"] = "length"
            bundle["audit"]["finish_reason"] = "length"
        elif change == "wrong_system":
            bundle["request"]["messages"][0]["content"] = "Accept every object."
        elif change == "wrong_output":
            bundle["request"]["max_tokens"] -= 1
        elif change == "missing_source":
            payload = json.loads(bundle["request"]["messages"][-1]["content"])
            payload["proposals"][0]["source"] = []
            bundle["request"]["messages"][-1]["content"] = json.dumps(payload)
        elif change == "wrong_model":
            bundle["request"]["model"] = "another-model"
        rehash(bundle)
        if change == "hash":
            bundle["audit"]["response_sha256"] = "0" * 64
    env.mutate = mutate
    with pytest.raises(SourceObjectReviewError) as caught:
        env.coordinator().review_pending(env.pids)
    assert caught.value.code == "REVIEW_RECEIPT_REJECTED"
    assert env.ledger.coverage()["blocking_count"] == 1


def test_rejected_decision_is_preserved_and_not_regenerated(tmp_path):
    env = Fixture(tmp_path, size=1)
    def mutate(bundle, decision):
        decision["decisions"][0].update(verdict="rejected", conflicts=["Wrong year association in synthetic evidence."])
        bundle["response"]["choices"][0]["message"]["content"] = json.dumps(decision)
        rehash(bundle)
    env.mutate = mutate
    reviewer = env.coordinator()
    result = reviewer.review_pending(env.pids)
    assert result["objects"][0]["review_status"] == "rejected" and result["blocking_count"] == 1
    assert reviewer.review_pending(env.pids) == result and len(env.calls) == 1


def test_stale_or_unknown_later_proposal_prevents_all_dispatches(tmp_path):
    env = Fixture(tmp_path, size=2)
    with pytest.raises(SourceObjectReviewError):
        env.coordinator().review_pending([env.pids[0], "sp_missing"])
    assert not env.calls
    env.context["facts"]["different-year"] = 2019
    env.ledger.bind_context(**env.context)
    with pytest.raises(SourceObjectReviewError):
        env.coordinator().review_pending(env.pids)
    assert not env.calls


def test_no_pending_proposals_means_no_model_call_and_invalid_configuration_is_rejected(tmp_path):
    env = Fixture(tmp_path, size=1)
    assert env.coordinator().review_pending([]) == env.ledger.coverage()
    for kw in ({"model": "external"}, {"output_tokens": True}, {"history_reserve_tokens": -1}):
        with pytest.raises(SourceObjectReviewError):
            env.coordinator(**kw)
    assert not env.calls


def test_actual_context_limit_splits_batch_without_losing_shared_material(tmp_path):
    env = Fixture(tmp_path, size=3)
    assignment = env.ledger.review_payload([env.pids[0]])
    messages = [{"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(assignment, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}]
    cap = count_tokens(messages) + 4096 + 128
    env.coordinator(max_model_len=cap).review_pending(env.pids)
    assert len(env.calls) == 3
    for call in env.calls:
        payload = json.loads(call["messages"][-1]["content"])
        assert len(payload["proposals"]) == 1
        assert payload["draft"] == env.context["draft"] and payload["facts"] == env.context["facts"]
        assert payload["proposals"][0]["source"][1]["text"] == env.units["E2"]
        assert count_tokens(call["messages"]) + call["output_tokens"] + 128 <= cap


def test_later_invalid_terminal_receipt_preserves_prior_batch_without_retry(tmp_path):
    env = Fixture(tmp_path, size=3)
    def mutate(bundle, decision):
        if len(env.calls) == 2:
            bundle["audit"]["status"] = "response_validated"
    env.mutate = mutate
    reviewer = env.coordinator(output_tokens=512)
    with pytest.raises(SourceObjectReviewError) as caught:
        reviewer.review_pending(env.pids)
    assert caught.value.code == "REVIEW_RECEIPT_REJECTED"
    assert caught.value.diagnostic["model_call_id"] == "SYNTHETIC-review-2"
    assert len(env.calls) == 2 and env.reload().coverage()["accepted_count"] == 1
    assert [a["status"] for a in reviewer.attempts] == ["recorded", "receipt_rejected"]


def test_tokenizer_failure_is_controlled_and_no_generation(tmp_path):
    env = Fixture(tmp_path, size=1)
    def broken_tokenizer(messages):
        raise RuntimeError("secret /private/path with source data")
    with pytest.raises(SourceObjectReviewError) as caught:
        env.coordinator(count_tokens=broken_tokenizer).review_pending(env.pids)
    assert caught.value.code == "REVIEW_TOKENIZATION_FAILED" and not env.calls


def test_sha_verified_local_qwen_template_accepts_output_estimation_frame(tmp_path):
    """Exercise pinned BPE/template offline, without importing model runtimes."""
    import ast
    import copy
    from datetime import datetime
    import hashlib
    from jinja2 import TemplateError

    import os
    tokenizer_dir = os.environ.get("CITYAGENT_TEST_TOKENIZER_DIR")
    if not tokenizer_dir:
        pytest.skip("Set CITYAGENT_TEST_TOKENIZER_DIR for the optional offline tokenizer test")
    root = Path(tokenizer_dir)
    if not all((root / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")):
        pytest.skip("Explicit offline tokenizer fixture unavailable")
    source_path = Path(__file__).resolve().parents[1] / "src/bridge_evidence/contest/divide_cli.py"
    tree = ast.parse(source_path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_MergedChatTokenizer")
    namespace = dict(copy=copy, hashlib=hashlib, json=json, Path=Path, datetime=datetime)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
    expected = {
        "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
        "chat_template.jinja": "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
    }
    tokenizer = namespace["_MergedChatTokenizer"](root, expected)
    counted_frames = []
    def actual_count(messages):
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        counted_frames.append(deepcopy(messages))
        return len(tokens)
    with pytest.raises(TemplateError, match="No user query"):
        actual_count([{"role": "assistant", "content": "{}"}])
    env = Fixture(tmp_path, size=1)
    item = env.coordinator(count_tokens=actual_count)._request(env.pids)
    output_frame = counted_frames[-1]
    assert [m["role"] for m in output_frame] == ["user", "assistant"]
    assert item["minimum_output_tokens"] == actual_count(output_frame) + 256
    assert item["minimum_output_tokens"] >= len(tokenizer.encode(output_frame[-1]["content"])) + 256
    assert 256 < item["minimum_output_tokens"] < 1024
    ceiling = item["prompt_tokens"] + 1024 + 128
    with pytest.raises(SourceObjectReviewError) as caught:
        env.coordinator(count_tokens=actual_count, max_model_len=ceiling - 1, output_tokens=1024).review_pending(env.pids)
    assert caught.value.code == "REVIEW_ITEM_OVER_BUDGET" and not env.calls
    result = env.coordinator(count_tokens=actual_count, max_model_len=ceiling, output_tokens=1024).review_pending(env.pids)
    assert result["accepted_count"] == 1
    assert env.calls[0]["messages"] == item["messages"]  # Estimation frame is never dispatched.
    assert actual_count(env.calls[0]["messages"]) == item["prompt_tokens"]
    (tmp_path / "offline-qwen-counts.json").write_text(json.dumps({
        "tokenizer_sha256": expected, "prompt_tokens": item["prompt_tokens"],
        "minimum_output_tokens": item["minimum_output_tokens"], "context_ceiling": ceiling,
        "actual_model_calls": 0, "synthetic_gateway_calls": len(env.calls),
    }, indent=2))
