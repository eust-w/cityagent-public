"""Result links must identify saved current content, not an old candidate dump."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from docx import Document
import pytest

from bridge_evidence.contest import document_multiagent as pipeline
from bridge_evidence.contest.document_delivery_artifacts import DocumentDeliveryArtifacts
from bridge_evidence.contest.document_workspace import DocumentWorkspace
from bridge_evidence.contest.docx_writer import BASIC_FIELDS
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import atomic_json, json_digest


def draft(basic, defects, advice):
    return {"基本信息": {field: "合成内容" + str(i) for i, (field, _) in enumerate(BASIC_FIELDS[:basic])},
            "病害明细": [{"病害描述": "候选病害" + str(i)} for i in range(defects)],
            "维修建议": [{"建议内容": "候选建议" + str(i)} for i in range(advice)]}


def rows(path):
    return [len(table.rows) - 1 for table in Document(path).tables]


@pytest.mark.parametrize("mode", ["failed", "stale", "corrupt", "unexportable",
    "completed", "completed_corrupt", "completed_pending", "reading_only_version"])
def test_result_selects_current_verified_word_or_explicit_candidate_fallback(tmp_path, monkeypatch, mode):
    # Reproduces v11: 3 basic fields / 19 defects / 15 suggestions in the
    # initial candidates versus 17 / 4 / 5 currently saved. Counts are not
    # semantic pass criteria. Word renders suggestions before defects.
    manifest = manifest_from_text("合成桥的检测记录。")
    candidates, current = draft(3, 19, 15), draft(17, 4, 5)
    unit = manifest["units"][0]

    def dispatch(tasks, **kwargs):
        task = tasks[0]["id"]
        entries = {}
        for section, values in candidates.items():
            for index, (field, value) in enumerate(values.items() if isinstance(values, dict) else enumerate(values)):
                identifier = section + str(index)
                entry = {"id": identifier, "section": section, "status": "explicit",
                         "source_refs": [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}]}
                entry.update({"field": field, "value": value} if section == "基本信息" else {"row": value})
                entries[identifier] = entry
        child = {"task_id": task, "manifest_sha256": json_digest(manifest), "status": "completed",
                 "model_calls": 1, "generation_attempts_settled": True, "handoff": {"unresolved": []},
                 "entries": {"identity": {"task_id": task, "manifest_sha256": json_digest(manifest)},
                             "reads": {unit["id"]: [[0, len(unit["text"])]]}, "entries": entries}}
        path = tmp_path / "child.json"
        atomic_json(path, child)
        return {"status": "completed", "results": [{"task_id": task, "result_path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "settled": True}]}

    client = SimpleNamespace(base_url="http://unused", model="synthetic", api_key="synthetic")
    monkeypatch.setattr(pipeline, "make_client", lambda *args, **kwargs: client)

    class Merger:
        artifact_delivery = None
        original_path = None

        def __init__(self, source, client, config, directory, **kwargs):
            self.workspace = DocumentWorkspace(source, directory, {"model": "synthetic"})
            self.artifact_manager = DocumentDeliveryArtifacts(directory / "delivery_artifacts", "synthetic-C1")

        def run(self):
            snapshot = self.workspace.snapshot()
            snapshot.update(draft=deepcopy(current), draft_sha256=json_digest(current))
            self.workspace = SimpleNamespace(snapshot=lambda: deepcopy(snapshot))
            export_snapshot = deepcopy(snapshot)
            if mode == "stale":
                export_snapshot.update(draft=deepcopy(candidates), draft_sha256=json_digest(candidates))
            success = mode.startswith("completed")
            self.artifact_delivery = self.artifact_manager.export(export_snapshot, incomplete=not success)
            self.original_path = self.artifact_delivery["docx_path"]
            Merger.original_path = self.original_path
            if success and mode != "completed_pending":
                self.artifact_delivery.update(complete=True, incomplete=False, phase="completed")
                atomic_json(Path(self.artifact_delivery["status_path"]), self.artifact_delivery)
            if mode == "reading_only_version":
                snapshot["version"] += 1
                snapshot["workspace_version"] = snapshot["version"]
            if mode in {"corrupt", "completed_corrupt", "unexportable"}:
                Path(self.original_path).write_bytes(b"not the saved Word file")
            if mode == "unexportable":
                def fail_export(*args, **kwargs):
                    raise OSError("Synthetic export failure")
                self.artifact_manager.export = fail_export
            if success:
                return {"status": "completed"}
            raise RuntimeError("Synthetic unresolved evidence")

        def _model_calls(self):
            return []

    result = pipeline.run_multiagent(manifest, tmp_path / "run", {"model": "synthetic"},
                                    dispatcher=dispatch, workflow_type=Merger)
    selected = Path(result["partial_docx"])
    candidate = Path(result["candidate_partial_docx"])
    assert candidate.name == "candidate_partial.docx"
    assert rows(candidate) == [3, 15, 19]
    assert not (tmp_path / "run" / "partial.docx").exists()
    assert (tmp_path / "run" / "candidate_partial.json").is_file()
    assert json.loads((tmp_path / "run" / "result.json").read_text()) == result
    assert result["complete"] is (mode == "completed")
    if mode == "unexportable":
        assert selected == candidate
        assert result["artifact_selection"]["source"] == "candidate_fallback"
        assert result["artifact_selection"]["current_workspace_verified"] is False
        assert result["delivery"] is None
    else:
        assert selected != candidate
        assert rows(selected) == [17, 5, 4]
        assert result["artifact_selection"]["current_workspace_verified"] is True
        assert result["delivery"]["file_sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
        assert result["delivery"]["draft_sha256"] == json_digest(current)
        if mode == "completed":
            assert selected.name == "report.docx"
            assert result["artifact_selection"]["source"] == "completed_report"
        else:
            assert selected.name == "partial.docx"
            assert result["artifact_selection"]["source"] == "current_workspace_partial"
            assert result["delivery"]["complete"] is False
    if mode in {"failed", "reading_only_version", "completed"}:
        assert str(selected) == Merger.original_path
        assert not result["artifact_selection"]["validation_errors"]
    else:
        assert result["artifact_selection"]["validation_errors"]
    if mode in {"completed_corrupt", "completed_pending"}:
        assert result["error"]["type"] == "DeliveryArtifactValidationError"
