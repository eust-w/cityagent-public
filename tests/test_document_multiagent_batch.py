"""Local spawned fake workers and actual DOCX validators; never a model service."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import pytest

from bridge_evidence.contest import document_multiagent_batch as batch
from bridge_evidence.contest.document_batch_admission import AdmissionClient
from bridge_evidence.contest.document_delivery_artifacts import DocumentDeliveryArtifacts
from bridge_evidence.contest.document_multiagent import MultiAgentSettings
from bridge_evidence.contest.document_multiagent_batch_cli import main as batch_main
from bridge_evidence.contest.run_state import atomic_json, json_digest
from test_document_material_index import make_material_case


def runtime(tmp_path, count=3):
    case = make_material_case(tmp_path / "inputs")
    paths = {"source_docx": case["source_path"], "official_original": case["source_path"]}
    for name, content in (("manifest", case["manifest"]), ("materials", case["materials"]),
                          ("full_source_inventory", case["manifest"]), ("excluded_content", {})):
        path = tmp_path / "inputs" / (name + ".json")
        atomic_json(path, content)
        paths[name] = path
    documents = [{"document_id": f"doc-{index}", "title": "同名报告/不是目录", "scope": "official-training",
        "runtime_conversion_required": False, "evaluation_outputs_are_not_model_inputs": True,
        **{name: {"path": str(path.relative_to(tmp_path)), "sha256": batch.file_sha(path)}
           for name, path in paths.items()}} for index in range(count)]
    path = tmp_path / "batch.runtime.json"
    atomic_json(path, {"schema": batch.RUNTIME_SCHEMA, "status": "ready", "scope": "official-training",
                       "documents": documents})
    return path


def prepared(path, settings=None):
    return batch.prepare_batch(path, batch.file_sha(path), settings=settings or MultiAgentSettings())


def _word_result(document, directory):
    directory.mkdir(parents=True, exist_ok=False)
    merge = directory / "merge"
    merge.mkdir()
    draft = {"基本信息": {"桥梁名称": document["document_id"]}, "病害明细": [], "维修建议": []}
    snapshot = {"version": 1, "draft": draft, "draft_sha256": json_digest(draft),
                "identity": {"manifest_sha256": document["manifest_identity_sha256"],
                             "test_only": "synthetic native receipts; no model called"}}
    atomic_json(merge / "workspace" / "workspace.json", snapshot)
    manager = DocumentDeliveryArtifacts(merge / "delivery_artifacts", document["document_key"])
    receipt = manager.export(snapshot, incomplete=False)
    origin = {"call_id": "cpu-call", "request_id": "cpu-request", "request_sha256": "1" * 64,
              "response_sha256": "2" * 64}
    from bridge_evidence.contest.document_workflow import DocumentWorkflow
    statement = {"semantic_state_sha256": DocumentWorkflow._semantic_identity(snapshot), "call_id": origin["call_id"], "model_statement": {"inspected_export": True,
        "complete": True, "remaining_issues": [], "summary": "synthetic fixture"},
        "verified_file_binding": {"workspace_version": 1, "draft_sha256": snapshot["draft_sha256"],
            "file_sha256": receipt["file_sha256"]}}
    receipt.update(complete=True, incomplete=False, phase="completed", actual_finish_origin=origin,
                   agent_file_self_check={"performed": True, "statement": statement})
    atomic_json(Path(receipt["status_path"]), receipt)
    atomic_json(merge / "document_workflow.json", {"actual_finish_origin": origin,
        "artifact_delivery": receipt, "accepted_finish": {"file_sha256": receipt["file_sha256"],
                            "model_self_check_sha256": json_digest(statement)}, "model_self_check": statement})
    result = {"complete": True, "delivery": receipt, "partial_docx": receipt["docx_path"],
              "artifact_selection": {"current_workspace_verified": True}}
    atomic_json(directory / "result.json", result)
    return result


def _fake_worker(document, directory, model, settings):
    mode = model.get("test_mode")
    client = AdmissionClient(model["batch_admission"])
    lease = client.acquire("main", "same-child-task-id", time.monotonic() + settings.report_timeout)
    if mode == "crash" and document["position"] == 0:
        os._exit(17)
    if mode == "hang":
        time.sleep(30)
    time.sleep(0.2)
    client.settle(lease, "complete", {"done": True, "synthetic": True})
    if mode == "one_failure" and document["position"] == 1:
        raise RuntimeError("intentional independent document failure")
    result = _word_result(document, directory)
    if mode == "corrupt":
        Path(result["delivery"]["docx_path"]).write_bytes(b"wrong actual Word")
    return result


def test_all_input_checks_are_pure_and_design_is_never_executable(tmp_path, monkeypatch):
    path = runtime(tmp_path, 4)
    monkeypatch.setattr(batch, "run_multiagent", lambda *a, **kw: pytest.fail("model dispatch"))
    value = prepared(path)
    assert len(value["documents"]) == 4 and value["model_calls"] == 0
    assert len({d["document_key"] for d in value["documents"]}) == 4
    raw = json.loads(path.read_bytes())
    raw.update(schema="cityagent-multiagent-batch-design-v1", status="design_inputs_validated_not_executed")
    atomic_json(path, raw)
    with pytest.raises(ValueError, match="design"):
        prepared(path)


@pytest.mark.parametrize("field", list(batch.FILE_FIELDS))
def test_any_file_hash_failure_rejects_entire_batch(tmp_path, field):
    path = runtime(tmp_path)
    raw = json.loads(path.read_bytes())
    raw["documents"][-1][field]["sha256"] = "0" * 64
    atomic_json(path, raw)
    with pytest.raises(ValueError, match="SHA256 differs"):
        prepared(path)


def test_material_identity_is_checked_after_exact_file_hash(tmp_path):
    path = runtime(tmp_path)
    raw = json.loads(path.read_bytes())
    materials_path = tmp_path / raw["documents"][0]["materials"]["path"]
    materials = json.loads(materials_path.read_bytes())
    materials["docx_sha256"] = "b" * 64
    atomic_json(materials_path, materials)
    for document in raw["documents"]:
        document["materials"]["sha256"] = batch.file_sha(materials_path)
    atomic_json(path, raw)
    with pytest.raises(ValueError):
        prepared(path)


def test_check_cli_no_model_and_same_settings(tmp_path, capsys):
    path = runtime(tmp_path, 1)
    assert batch_main(["--batch", str(path), "--batch-sha256", batch.file_sha(path), "--check",
                      "--output-dir", str(tmp_path / "check"), "--no-thinking", "--direct-delivery"]) == 0
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0
    assert not (tmp_path / "check" / "admission-events.jsonl").exists()


@pytest.mark.parametrize("count", [1, 3, 4])
def test_spawn_workers_isolate_same_name_and_evidence_ids_and_share_global_cap(tmp_path, count):
    path = runtime(tmp_path, count)
    result = batch.run_batch(prepared(path), tmp_path / "run", {"model": "cpu-only"}, worker=_fake_worker,
                             batch_settings=batch.BatchSettings(settlement_grace_seconds=0.2))
    assert result["complete"] and result["new_complete_reports"] == count
    assert result["reused_complete_reports"] == 0 and not result["resume_supported"]
    rows = result["documents"]
    assert len({row["pid"] for row in rows}) == count
    assert len({row["run_directory"] for row in rows}) == count
    assert all(row["pid"] != os.getpid() for row in rows)
    for row in rows:
        receipt = row["publication"]["source_receipt"]
        assert json.loads(Path(receipt["json_path"]).read_bytes())["基本信息"]["桥梁名称"] == row["document_id"]
        assert batch.file_sha(row["publication"]["path"]) == receipt["file_sha256"]
    requests = result["admission"]["requests"]
    assert {row["document_id"] for row in requests} == {row["document_key"] for row in rows}
    assert all(row["state"] == "complete" for row in requests)
    # Intervals come from actual broker grants/settlements, not worker claims.
    events = [json.loads(line) for line in (tmp_path / "run" / "admission-events.jsonl").read_text().splitlines()]
    active, peak = set(), 0
    for event in events:
        if event.get("event") == "transition":
            state = event.get("state", event.get("to"))
            request = event.get("request", {})
            identifier = event.get("request_id", request.get("request_id"))
            if state == "active":
                active.add(identifier)
            elif state in {"complete", "not_sent"}:
                active.discard(identifier)
            peak = max(peak, len(active))
    # Also compare the authoritative lease intervals for all requests.
    intervals = []
    for request in requests:
        intervals.extend([(request["granted_at"], 1), (request["updated_at"], -1)])
    current = peak = 0
    for _, delta in sorted(intervals):
        current += delta
        peak = max(peak, current)
    assert peak <= 2
    if count >= 3:
        assert peak == 2


def test_failure_does_not_cancel_other_documents_and_throughput_counts_only_new_complete(tmp_path):
    path = runtime(tmp_path, 3)
    result = batch.run_batch(prepared(path), tmp_path / "run", {"test_mode": "one_failure"},
        worker=_fake_worker, batch_settings=batch.BatchSettings(settlement_grace_seconds=0.2))
    assert not result["complete"] and result["new_complete_reports"] == 2
    assert result["documents"][1]["error"]["message"] == "intentional independent document failure"
    assert result["new_complete_reports_per_hour"] == 2 * 3600 / result["seconds"]


def test_corrupt_word_not_counted_complete(tmp_path):
    path = runtime(tmp_path, 1)
    result = batch.run_batch(prepared(path), tmp_path / "run", {"test_mode": "corrupt"}, worker=_fake_worker,
                            batch_settings=batch.BatchSettings(settlement_grace_seconds=0.1))
    assert result["new_complete_reports"] == 0 and not result["complete"]
    assert "changed" in result["documents"][0]["publication"]["error"]


def test_explicit_reuse_checks_word_and_does_not_count_as_new(tmp_path):
    path = runtime(tmp_path, 1)
    settings = batch.BatchSettings(settlement_grace_seconds=0.1)
    first = batch.run_batch(prepared(path), tmp_path / "first", {}, worker=_fake_worker, batch_settings=settings)
    raw = json.loads(path.read_bytes())
    record = first["documents"][0]["batch_record_path"]
    raw["documents"][0]["reuse_result"] = {"path": record, "sha256": batch.file_sha(record)}
    atomic_json(path, raw)
    second = batch.run_batch(prepared(path), tmp_path / "second", {}, worker=_fake_worker, batch_settings=settings)
    assert second["complete"] and second["reused_complete_reports"] == 1
    assert second["new_complete_reports"] == 0 and second["new_complete_reports_per_hour"] == 0
    assert not second["admission"]["requests"]
    Path(first["documents"][0]["publication"]["path"]).write_bytes(b"stale")
    with pytest.raises(ValueError, match="changed"):
        batch.run_batch(prepared(path), tmp_path / "third", {}, worker=_fake_worker, batch_settings=settings)


def test_changed_input_after_preflight_rejected_before_output_or_process(tmp_path):
    path = runtime(tmp_path)
    ready = prepared(path)
    Path(ready["documents"][0]["materials"]["path"]).write_text("{}")
    with pytest.raises(ValueError):
        batch.run_batch(ready, tmp_path / "never-created", {}, worker=_fake_worker)
    assert not (tmp_path / "never-created").exists()


def test_crashed_document_unknown_not_refunded(tmp_path):
    path = runtime(tmp_path, 1)
    result = batch.run_batch(prepared(path), tmp_path / "run", {"test_mode": "crash"}, worker=_fake_worker,
                             batch_settings=batch.BatchSettings(settlement_grace_seconds=0.1))
    assert not result["complete"] and result["new_complete_reports"] == 0
    assert result["admission"]["pools"]["main"]["unknown"] == 1
    assert result["documents"][0]["service_cancellation_confirmed"] is False


def test_default_worker_calls_real_pipeline(tmp_path, monkeypatch):
    path = runtime(tmp_path, 1)
    document = prepared(path)["documents"][0]
    received = {}
    def actual(manifest, directory, model, **kwargs):
        received.update(manifest=manifest, directory=directory, model=model, kwargs=kwargs)
        return {"test_only": True}
    monkeypatch.setattr(batch, "run_multiagent", actual)
    config = {"batch_admission": {"test": "binding"}}
    assert batch._default_worker(document, tmp_path / "out", config, MultiAgentSettings()) == {"test_only": True}
    assert received["model"] == config
    assert received["kwargs"]["materials"]["docx_sha256"] == document["document_sha256"]
    assert Path(received["kwargs"]["materials_source_path"]) == Path(document["source_docx"]["path"])


@pytest.mark.parametrize("change", ["scope", "design_status", "duplicate", "empty", "conversion"])
def test_runtime_contract_rejects_ambiguous_or_unprepared_lists(tmp_path, change):
    path = runtime(tmp_path, 2)
    value = json.loads(path.read_bytes())
    if change == "scope":
        value["documents"][1]["scope"] = "unknown"
    elif change == "design_status":
        value["status"] = "planned"
    elif change == "duplicate":
        value["documents"][1]["document_id"] = value["documents"][0]["document_id"]
    elif change == "empty":
        value["documents"] = []
    else:
        value["documents"][1]["runtime_conversion_required"] = True
    atomic_json(path, value)
    with pytest.raises(ValueError):
        prepared(path)


def _late_child(config, output_path):
    time.sleep(1.2)
    try:
        lease = AdmissionClient(config).acquire("main", "late-orphan-child", time.monotonic() + 2)
    except Exception as error:
        atomic_json(Path(output_path), {"admitted": False, "error": type(error).__name__})
    else:
        atomic_json(Path(output_path), {"admitted": True, "lease": lease})


def _parent_hangs_worker(document, directory, model, settings):
    import multiprocessing
    client = AdmissionClient(model["batch_admission"])
    client.acquire("main", "crashing-parent", time.monotonic() + settings.report_timeout)
    atomic_json(Path(model["started_file"]), {"pid": os.getpid(), "deadline": time.monotonic() + settings.report_timeout})
    child = multiprocessing.get_context("spawn").Process(target=_late_child,
        args=(model["batch_admission"], model["late_file"]))
    child.start()
    time.sleep(30)


def test_supervisor_deadline_preserves_unknown_and_late_child_cannot_send(tmp_path):
    path = runtime(tmp_path, 1)
    settings = MultiAgentSettings(report_timeout=1.0)
    result = batch.run_batch(prepared(path, settings), tmp_path / "run",
        {"started_file": str(tmp_path / "started.json"), "late_file": str(tmp_path / "late.json")},
        settings=settings, worker=_parent_hangs_worker,
        batch_settings=batch.BatchSettings(settlement_grace_seconds=0.1))
    assert (tmp_path / "started.json").exists(), "test must reach actual acquired lease before forcing deadline"
    assert not result["complete"] and result["admission"]["pools"]["main"]["unknown"] == 1
    assert result["documents"][0]["service_cancellation_confirmed"] is False
    until = time.monotonic() + 5
    while not (tmp_path / "late.json").exists() and time.monotonic() < until:
        time.sleep(0.03)
    assert json.loads((tmp_path / "late.json").read_bytes())["admitted"] is False
    assert len(result["admission"]["requests"]) == 1


@pytest.mark.parametrize("mutation", ["workspace_semantics", "finish", "foreign_manifest"])
def test_current_delivery_rejects_changed_semantics_or_finish_binding(tmp_path, mutation):
    path = runtime(tmp_path, 1)
    document = prepared(path)["documents"][0]
    directory = tmp_path / "run"
    result = _word_result(document, directory)
    if mutation == "finish":
        file = directory / "merge" / "document_workflow.json"
        value = json.loads(file.read_bytes());value["actual_finish_origin"]["request_id"] = "foreign"
    else:
        file = directory / "merge" / "workspace" / "workspace.json"
        value = json.loads(file.read_bytes())
        if mutation == "workspace_semantics":
            value["facts"] = {"new": "changed after Word review"}
        else:
            value["identity"]["manifest_sha256"] = "a" * 64
    atomic_json(file, value)
    with pytest.raises(ValueError):
        batch.verify_current_delivery(directory, result,
            manifest_identity_sha256=document["manifest_identity_sha256"])
