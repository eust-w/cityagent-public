"""Publication order with real spawned workers and the real local broker."""
import json
from pathlib import Path
import time

from bridge_evidence.contest import document_multiagent_batch as batch
from bridge_evidence.contest.document_batch_admission import AdmissionClient
from bridge_evidence.contest.run_state import atomic_json
from test_document_multiagent_batch import runtime, prepared, _word_result


def _publication_worker(document, directory, model, settings):
    client = AdmissionClient(model["batch_admission"])
    # The second report is a CPU-only sibling; it does not contend with an
    # intentionally unknown request in the first report.
    lease = client.acquire("main", "publication-worker", time.monotonic() + settings.report_timeout) if document["position"] == 0 else None
    if model.get("overlap"):
        marker = Path(model["overlap"])
        if document["position"] == 1:
            marker.write_text("started")
            time.sleep(0.8)
            marker.write_text("finished")
        else:
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists(), "sibling must start to exercise overlap"
    if lease is not None and not model.get("unknown"):
        client.settle(lease, "complete", {"synthetic_cpu_only": True})
    return _word_result(document, directory)


def test_unknown_request_never_publishes_success_and_does_not_cancel_sibling(tmp_path):
    source = runtime(tmp_path, 2)
    callbacks = []
    def publish(document, record):
        callbacks.append((document["document_id"], record["complete"]))
        if record["complete"]:
            (tmp_path / (document["document_id"] + ".published")).write_text("final")
    result = batch.run_batch(prepared(source), tmp_path / "run", {"unknown": True},
        worker=_publication_worker, batch_settings=batch.BatchSettings(settlement_grace_seconds=0.1),
        on_document_complete=publish)
    first, second = result["documents"]
    assert first["status"] == "unsettled_generation" and not first["complete"]
    assert not first["generation_settlement"]["settled"]
    assert not (Path(first["run_directory"]) / "delivery" / "report.docx").exists()
    assert not (tmp_path / "doc-0.published").exists()
    assert (tmp_path / "doc-1.published").exists() and second["complete"]
    assert sorted(callbacks) == [("doc-0", False), ("doc-1", True)]
    assert json.loads(Path(first["batch_record_path"]).read_bytes())["complete"] is False


def test_settled_callback_does_not_wait_for_running_sibling(tmp_path):
    source = runtime(tmp_path, 2)
    marker = tmp_path / "sibling-state"
    callbacks = []
    def publish(document, record):
        assert record["generation_settlement"]["settled"]
        if document["document_id"] == "doc-0":
            assert marker.read_text() == "started"
        callbacks.append(document["document_id"])
    result = batch.run_batch(prepared(source), tmp_path / "run", {"overlap": str(marker)},
        worker=_publication_worker, batch_settings=batch.BatchSettings(settlement_grace_seconds=0.1),
        on_document_complete=publish)
    assert result["complete"] and callbacks == ["doc-0", "doc-1"]


def test_late_idle_settlement_precedes_publication(tmp_path):
    source = runtime(tmp_path, 1)
    idle = tmp_path / "idle-proof-recorded"
    callbacks = []
    def observe(pool, deadline, snapshot):
        time.sleep(0.1)
        idle.write_text("synthetic CPU observer; no real service")
        return {"confirmed_idle": True, "synthetic_cpu_only": True}
    def publish(document, record):
        assert idle.exists()
        assert record["generation_settlement"]["settled"]
        assert record["generation_settlement"]["requests"][0]["state"] == "quiescent"
        callbacks.append(record["complete"])
    result = batch.run_batch(prepared(source), tmp_path / "run", {"unknown": True},
        worker=_publication_worker, batch_settings=batch.BatchSettings(settlement_grace_seconds=0.5),
        quiescence_observer=observe, on_document_complete=publish)
    assert result["complete"] and callbacks == [True]


def test_reused_report_receives_one_settled_callback_without_new_model_request(tmp_path):
    source = runtime(tmp_path, 1)
    settings = batch.BatchSettings(settlement_grace_seconds=0.1)
    first = batch.run_batch(prepared(source), tmp_path / "first", {}, worker=_publication_worker,
                            batch_settings=settings)
    value = json.loads(source.read_bytes())
    record_path = first["documents"][0]["batch_record_path"]
    value["documents"][0]["reuse_result"] = {"path": record_path, "sha256": batch.file_sha(record_path)}
    atomic_json(source, value)
    callbacks = []
    second = batch.run_batch(prepared(source), tmp_path / "second", {}, worker=_publication_worker,
        batch_settings=settings, on_document_complete=lambda doc, record: callbacks.append(record))
    assert second["complete"] and second["reused_complete_reports"] == 1
    assert not second["admission"]["requests"]
    assert len(callbacks) == 1 and callbacks[0]["status"] == "reused_complete"
    assert callbacks[0]["generation_settlement"]["settled"]
    assert callbacks[0]["generation_settlement"]["reused_prior_result"]


def test_callback_failure_is_recorded_without_cancelling_sibling(tmp_path):
    source = runtime(tmp_path, 2)
    callbacks = []
    def publish(document, record):
        callbacks.append(document["document_id"])
        if document["document_id"] == "doc-0":
            raise OSError("synthetic publication failure")
    result = batch.run_batch(prepared(source), tmp_path / "run", {}, worker=_publication_worker,
        batch_settings=batch.BatchSettings(settlement_grace_seconds=0.1), on_document_complete=publish)
    assert set(callbacks) == {"doc-0", "doc-1"}
    assert result["documents"][0]["status"] == "publication_failed"
    assert result["documents"][1]["complete"]
    assert result["new_complete_reports"] == 1 and not result["complete"]


def test_staged_bytes_changed_before_commit_are_not_published(tmp_path):
    source = runtime(tmp_path, 1)
    document = prepared(source)["documents"][0]
    directory = tmp_path / "run"
    result = _word_result(document, directory)
    staged = batch._stage_delivery(directory, result, document)
    assert staged["verified_complete"]
    Path(staged["path"]).write_bytes(b"changed after validation")
    publication = batch._commit_delivery({"run_directory": str(directory), "staged_delivery": staged})
    assert not publication["verified_complete"] and "changed" in publication["error"]
    assert not (directory / "delivery" / "report.docx").exists()
