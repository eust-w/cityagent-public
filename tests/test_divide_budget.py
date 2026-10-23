"""Shared budgets against a synthetic transport; no model or remote connection."""
from collections import deque
from contextlib import contextmanager
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest import divide_budget as module
from bridge_evidence.contest import divide_cli as cli
from bridge_evidence.contest.divide_conquer_workflow import DivideSettings, DivideWorkflow
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.local_model_client import LocalModelClient
from test_divide_training_batch import batch_boundary, run, read_summary


MESSAGES = [{"role": "user", "content": "synthetic request"}]


class Transport:
    def __init__(self):
        self.requests = []
        self.failures = deque()
        self.after_response = None
        self.prompt_usage = 7
        self.finish_reason = "stop"

    @contextmanager
    def open(self, request, timeout):
        payload = json.loads(request.data) if request.data is not None else None
        self.requests.append({"url": request.full_url, "payload": payload, "timeout": timeout})
        if request.full_url.endswith("/tokenize"):
            result = {"count": 7}
        elif request.full_url.endswith("/v1/models"):
            result = {"data": [{"id": "synthetic-lora", "root": "/synthetic", "parent": "synthetic-base"}]}
        else:
            if self.failures and self.failures.popleft():
                raise HTTPError(request.full_url, 503, "synthetic failure", {}, io.BytesIO(b""))
            result = {"id": "synthetic-response", "model": "synthetic-lora",
                      "choices": [{"message": {"role": "assistant", "content": "synthetic result"},
                                   "finish_reason": self.finish_reason}],
                      "usage": {"prompt_tokens": self.prompt_usage, "completion_tokens": 3, "total_tokens": 10}}
        transport = self

        class Response:
            status = 200

            def read(self):
                if transport.after_response:
                    transport.after_response(request.full_url)
                return json.dumps(result).encode()

        yield Response()


@pytest.fixture
def bounded(tmp_path, monkeypatch):
    now = [module.time.monotonic()]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    transport = Transport()
    raw = LocalModelClient("http://127.0.0.1:8765", "synthetic-lora", timeout=90, max_output_tokens=16)
    raw._opener = transport
    budget = module.DivideBudget(tmp_path / "budget", started=now[0], report_timeout=30, max_model_calls=2)
    client = budget.wrap(raw)
    return budget, client, raw, transport, now


def payload(client, maximum=16):
    return {"model": client.model, "messages": MESSAGES, "max_tokens": maximum, "stream": False,
            "temperature": client.temperature,
            "chat_template_kwargs": {"enable_thinking": client.enable_thinking}}


def test_budget_proxy_preserves_durable_transport_receipts_and_reservation_context(bounded):
    _, client, raw, _, _ = bounded
    snapshots = []
    client.receipt_sink = lambda receipt: snapshots.append(dict(receipt))
    client.request_context = {"reservation_id": "workflow-reservation"}
    client.generate(MESSAGES)
    assert raw.receipt_sink is client.receipt_sink
    assert raw.request_context == {"reservation_id": "workflow-reservation"}
    assert snapshots
    by_call = {}
    for receipt in snapshots:
        by_call.setdefault(receipt["call_id"], []).append(receipt)
        assert receipt["reservation_id"] == "workflow-reservation"
    assert all(rows[0]["status"] == "running" and rows[-1]["status"] == "success"
               for rows in by_call.values())


def test_native_and_direct_share_actual_http_allowance_without_double_count(bounded):
    budget, client, raw, transport, _ = bounded
    gateway = budget.gateway(client, output_tokens=16)
    gateway.forward(payload(client))
    client.generate(MESSAGES)
    assert budget.generation_count == 2
    assert [r["kind"] for r in raw.calls] == ["tokenize", "agent_generation", "tokenize", "generation"]
    before = len(transport.requests)
    with pytest.raises(RuntimeError, match="generation budget exhausted"):
        gateway.forward(payload(client))
    with pytest.raises(RuntimeError, match="generation budget exhausted"):
        client.generate(MESSAGES)
    assert len(transport.requests) == before
    snapshot = json.loads((budget.directory / "divide_budget.json").read_text())
    assert snapshot["generation_requests"] == 2
    assert len({row["call_id"] for row in snapshot["dispatches"]}) == 2
    assert all(row["usage"]["completion_tokens"] == 3 for row in snapshot["dispatches"])


def test_tokenizer_and_identity_http_do_not_spend_generations_even_after_last_call(bounded):
    budget, client, raw, _, _ = bounded
    budget.max_model_calls = 1
    client.count_tokens("synthetic input")
    client.models()
    assert budget.generation_count == 0
    client.generate(MESSAGES)
    budget.check(stage="before_publish")
    client.count_tokens("another synthetic input")
    client.models()
    assert budget.generation_count == 1
    assert len(raw.calls) == 6


def test_failed_http_counts_and_preserves_real_failed_receipt(bounded):
    budget, client, raw, transport, _ = bounded
    transport.failures.extend([True, False])
    with pytest.raises(RuntimeError, match="HTTP 503"):
        client.generate(MESSAGES)
    assert budget.generation_count == 1
    assert raw.calls[-1]["status"] == "failed" and raw.calls[-1]["http_status"] == 503
    assert budget.dispatches[-1]["call_id"] == raw.calls[-1]["call_id"]
    budget.gateway(client, output_tokens=16).forward(payload(client))
    assert budget.generation_count == 2
    with pytest.raises(RuntimeError, match="generation budget exhausted"):
        client.generate(MESSAGES)


def test_context_rejection_does_not_consume_generation_or_prevent_retry(bounded):
    budget, client, raw, _, _ = bounded
    gateway = budget.gateway(client, output_tokens=16)
    raw.max_model_len = 20
    with pytest.raises(ValueError, match="exceed context"):
        gateway.forward(payload(client))
    assert budget.generation_count == 0
    gateway.forward(payload(client, maximum=10))
    assert budget.generation_count == 1


@pytest.mark.parametrize("native", [False, True])
def test_late_success_is_rejected_without_relabelling_http_success(bounded, native):
    budget, client, raw, transport, now = bounded
    transport.after_response = lambda url: now.__setitem__(0, now[0] + (31 if url.endswith("completions") else 0))
    with pytest.raises(TimeoutError, match="deadline"):
        (budget.gateway(client, output_tokens=16).forward(payload(client)) if native else client.generate(MESSAGES))
    actual = raw.calls[-1]
    assert actual["status"] == "success" and actual["http_status"] == 200
    assert actual["usage"]["completion_tokens"] == 3
    assert budget.dispatches[-1]["status"] == "failed"
    assert budget.dispatches[-1]["call_id"] == actual["call_id"]
    assert any(row["code"] == "REPORT_DEADLINE_EXCEEDED" for row in budget.failures)


def test_tokenization_time_uses_same_deadline_and_prevents_generation(bounded):
    budget, client, raw, transport, now = bounded
    transport.after_response = lambda _: now.__setitem__(0, now[0] + 31)
    with pytest.raises(TimeoutError):
        client.generate(MESSAGES)
    assert budget.generation_count == 0
    assert [row["kind"] for row in raw.calls] == ["tokenize"]


def test_request_timeout_is_remaining_report_time_and_restored(bounded):
    budget, client, raw, transport, now = bounded
    now[0] += 27
    client.generate(MESSAGES)
    assert [row["timeout"] for row in transport.requests] == [3, 3]
    assert raw.timeout == client.timeout == 90


def test_missing_usage_rejected_with_actual_call_id(bounded):
    budget, client, raw, transport, _ = bounded
    transport.prompt_usage = None
    with pytest.raises(ValueError, match="usage differs"):
        client.generate(MESSAGES)
    assert budget.generation_count == 1
    assert budget.failures[-1]["call_id"] == raw.calls[-1]["call_id"]


def test_receipt_io_failure_cannot_hide_original_http_failure(bounded, monkeypatch):
    budget, client, raw, transport, _ = bounded
    transport.failures.append(True)
    persist = budget.persist

    def fail_after_dispatch():
        if any(row["kind"] == "generation" for row in raw.calls):
            raise OSError("synthetic receipt write failed")
        persist()

    monkeypatch.setattr(budget, "persist", fail_after_dispatch)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        client.generate(MESSAGES)
    assert budget.generation_count == 1
    assert any(row["code"] == "BUDGET_RECEIPT_PERSISTENCE_FAILED" for row in budget.failures)


def test_pending_intent_is_durable_before_generation_transport(bounded, monkeypatch):
    budget, client, raw, _, _ = bounded
    request = raw._request
    seen = []

    def inspect_pending(endpoint, payload, *, kind):
        if kind == "generation":
            snapshot = json.loads((budget.directory / "divide_budget.json").read_text())
            seen.append(snapshot)
            assert snapshot["generation_requests"] == 0
            assert snapshot["generation_allowance_spent"] == 1
            assert snapshot["unresolved_generation_reservations"] == 1
            pending = snapshot["dispatches"][-1]
            assert pending["status"] == "dispatch_pending" and pending["generation_attempted"] is None
            assert "not proof of delivery or non-delivery" in snapshot["dispatch_semantics"]
        return request(endpoint, payload, kind=kind)

    monkeypatch.setattr(raw, "_request", inspect_pending)
    client.generate(MESSAGES)
    assert len(seen) == 1
    completed = json.loads((budget.directory / "divide_budget.json").read_text())
    assert completed["generation_requests"] == completed["generation_allowance_spent"] == 1
    assert completed["unresolved_generation_reservations"] == 0
    assert completed["dispatches"][-1]["generation_attempted"] is True


@pytest.mark.parametrize("failure", ["persist_error", "deadline"])
def test_pending_persistence_failure_or_timeout_prevents_http_and_releases_reservation(bounded, monkeypatch, failure):
    budget, client, raw, transport, now = bounded
    persist = budget.persist

    def fail_pending():
        if budget.dispatches and budget.dispatches[-1]["status"] == "dispatch_pending":
            if failure == "persist_error":
                raise OSError("synthetic pending intent persistence failed")
            persist()
            now[0] += 31
        else:
            persist()

    monkeypatch.setattr(budget, "persist", fail_pending)
    with pytest.raises(OSError if failure == "persist_error" else TimeoutError):
        client.generate(MESSAGES)
    assert [row["kind"] for row in raw.calls] == ["tokenize"]
    assert len(transport.requests) == 1
    assert budget.generation_count == 0
    snapshot = json.loads((budget.directory / "divide_budget.json").read_text())
    assert snapshot["generation_requests"] == snapshot["generation_allowance_spent"] == 0
    assert snapshot["unresolved_generation_reservations"] == 0
    assert snapshot["dispatches"][-1]["status"] == "failed_before_http"
    assert snapshot["dispatches"][-1]["generation_attempted"] is False


@pytest.mark.parametrize("option", ["timeout_only", "calls_only"])
def test_each_optional_budget_is_effective_independently(bounded, option):
    budget, client, _, transport, now = bounded
    if option == "timeout_only":
        budget.max_model_calls = None
        for _ in range(3):
            client.generate(MESSAGES)
        now[0] += 31
        with pytest.raises(TimeoutError):
            client.generate(MESSAGES)
    else:
        budget.report_timeout = budget.deadline = None
        budget.max_model_calls = 1
        # Also exercise the native gateway's finite platform lock horizon.
        budget.gateway(client, output_tokens=16).forward(payload(client))
        now[0] += 10000
        with pytest.raises(RuntimeError, match="generation budget exhausted"):
            client.generate(MESSAGES)
        budget.check(stage="before_publish")


def test_cli_budget_and_output_flags_preserve_old_defaults():
    from test_divide_training_batch import entry
    args = entry.build_parser().parse_args([])
    assert args.divide_report_timeout is args.divide_max_model_calls is None
    assert args.agent_output_tokens == args.review_output_tokens == 1024


@pytest.mark.parametrize("timeout,maximum", [(30, None), (None, 2), (30, 2)])
def test_cli_explicit_budget_resume_is_rejected_without_resetting_any_budget(batch_boundary, timeout, maximum):
    args, _, _, calls = batch_boundary
    args.resume = True
    args.divide_report_timeout, args.divide_max_model_calls = timeout, maximum
    prior = args.work_dir / "documents" / "TRAIN-1" / "divide_budget.json"
    prior.parent.mkdir(parents=True)
    data = b'{"generation_allowance_spent":2,"unresolved_generation_reservations":1}'
    prior.write_bytes(data)
    with pytest.raises(ValueError, match="resume is not yet supported"):
        run(args)
    assert calls["clients"] == 0
    assert prior.read_bytes() == data


def test_cli_native_loopback_and_direct_extraction_use_one_budget(batch_boundary, monkeypatch):
    args, _, _, _ = batch_boundary
    args.divide_report_timeout = 60
    args.divide_max_model_calls = 2
    args.agent_output_tokens = args.review_output_tokens = 4096
    preflight_client = cli._make_client
    transports = []
    seen = []

    def client_factory(config):
        if not transports:
            transports.append(None)
            return preflight_client(config)
        raw = LocalModelClient("http://127.0.0.1:8765", config.served_model, max_output_tokens=4096)
        raw._opener = Transport()
        transports.append(raw._opener)
        return raw

    class Workflow:
        def __init__(self, manifest, client, config, directory, **options):
            self.client, self.config = client, config
            seen.append(options)

        def run(self):
            request = Request(self.config["url"] + "/chat/completions",
                              data=json.dumps(payload(self.client, self.config["max_tokens"])).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=5) as response:
                assert response.status == 200
            self.client.generate(MESSAGES, max_tokens=4096)
            return {"data": {"基本信息": {"桥梁名称": "合成桥"}, "病害明细": [], "维修建议": []},
                    "receipt": {"status": "success"}}

    monkeypatch.setattr(cli, "_make_client", client_factory)
    monkeypatch.setattr(cli, "DivideWorkflow", Workflow)
    run(args)
    assert all(item["review_output_tokens"] == 4096 for item in seen)
    for row in read_summary(args)["documents"]:
        assert row["divide_budget"]["generation_requests"] == 2
        assert [r["kind"] for r in row["divide_budget"]["model_calls"]] == ["tokenize", "agent_generation", "tokenize", "generation"]
    identity = json.loads((args.work_dir / "run_manifest.json").read_text())["identity"]
    assert identity["divide_budget"] == {"report_timeout": 60, "max_model_calls": 2}
    assert identity["divide_output_tokens"] == {"agent": 4096, "review": 4096}


@pytest.mark.parametrize("stage", ["conversion", "export"])
def test_cli_deadline_includes_conversion_and_export_before_publication(batch_boundary, monkeypatch, stage):
    args, _, _, calls = batch_boundary
    now = [module.time.monotonic()]
    clock = SimpleNamespace(monotonic=lambda: now[0], perf_counter=lambda: now[0])
    monkeypatch.setattr(module, "time", clock)
    monkeypatch.setattr(cli, "time", clock)
    args.divide_report_timeout = 10
    # The synthetic workflow does not need SDK; retain the same CLI budget.
    args.workflow = "divide-fixed"
    factory = cli._make_client

    def with_timeout(config):
        client = factory(config)
        client.timeout = 90
        return client

    monkeypatch.setattr(cli, "_make_client", with_timeout)
    if stage == "export":
        calls["on_write"] = lambda: now.__setitem__(0, now[0] + 11)
    else:
        inputs = args.work_dir.parent / "word-inputs"
        inputs.mkdir()
        (inputs / "synthetic.doc").write_bytes(b"synthetic doc placeholder")
        args.training_evidence_manifest = None
        args.input_dir = inputs
        monkeypatch.setattr(cli.shutil, "which", lambda _: "synthetic-soffice")

        def convert(source, directory):
            now[0] += 11
            return source

        monkeypatch.setattr(cli, "build_manifest", lambda _: manifest_from_text("synthetic"))
    with pytest.raises(RuntimeError, match="divide document"):
        if stage == "conversion":
            cli.run_divide(args, code_root=ROOT, convert_doc=convert)
        else:
            run(args)
    assert not list(args.result_dir.glob("*.docx"))
    assert not list(args.result_dir.glob(".pending-*"))
    summary = read_summary(args)
    assert summary["completed_reports"] == 0
    assert all(any(item["code"] == "REPORT_DEADLINE_EXCEEDED" for item in row["divide_budget"]["failures"])
               for row in summary["documents"])
    if stage == "conversion":
        assert calls["workflows"] == []


@pytest.mark.parametrize("limit,expected", [(1024, 1024), (4096, 4096), (9000, 4096)])
def test_conflict_review_output_cap_is_parameterized_without_algorithm_change(tmp_path, limit, expected):
    manifest = manifest_from_text("桥梁名称：合成桥\n报告日期：2018\n报告日期：2019")
    seen = []

    class Client:
        calls = []

        def count_tokens(self, text):
            return 1

        def generate(self, messages, max_tokens):
            seen.append(max_tokens)
            return {"status": "success", "finish_reason": "stop", "call_id": "synthetic-review",
                    "text": json.dumps({"value": "未知", "evidence_ids": []})}

    workflow = DivideWorkflow(manifest, Client(), {}, tmp_path, DivideSettings(output_tokens=4096),
                              mode="divide-fixed", review_output_tokens=limit)
    # A synthetic reducer confirms only transport/output configuration here.
    workflow.reducer.resolve = lambda *args: None
    workflow._review({"id": "synthetic-conflict", "field": "报告日期", "candidates": [
        {"value": "2018", "evidence_ids": [manifest["units"][1]["id"]]},
        {"value": "2019", "evidence_ids": [manifest["units"][2]["id"]]}]})
    assert seen == [expected]
