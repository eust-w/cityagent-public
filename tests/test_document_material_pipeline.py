"""Run-local materials through a child, import, real Word and file self-check.

Model decisions are scripted here; this verifies wiring and file provenance,
not report accuracy or GPU performance.
"""
import socket
import time
from types import SimpleNamespace

from docx import Document

from bridge_evidence.contest import document_multiagent_worker as worker
from bridge_evidence.contest.document_material_run import prepare_material_run
from bridge_evidence.contest.document_multiagent import import_candidates
from bridge_evidence.contest.document_task_plan import build_document_task_plan
from bridge_evidence.contest.document_workflow import DocumentSettings, DocumentWorkflow
from bridge_evidence.contest.run_state import atomic_json, json_digest
from test_document_material_index import make_material_case
from test_document_multiagent import FakeGateway
from test_managed_file_self_check import Harness


def test_materials_reach_child_saved_entries_main_and_actual_word(tmp_path, monkeypatch):
    def blocked(*a, **kw):
        raise AssertionError("This integration gate must not call a model service")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    case = make_material_case(tmp_path / "input")
    manifest = case["manifest"]
    index, material_request = prepare_material_run(manifest, case["materials"], case["source_path"], tmp_path / "run")
    source = tmp_path / "run" / "canonical.json"
    atomic_json(source, manifest)
    task = build_document_task_plan(manifest, target_tasks=1, max_tasks=1, material_index=index)["tasks"][0]
    request = {"manifest_path": str(source), "manifest_sha256": json_digest(manifest), "task": task,
               "model_config": {"enable_thinking": False}, "deadline_monotonic": time.monotonic() + 120,
               "max_model_calls": 8, **material_request}
    client = SimpleNamespace(calls=[], model="synthetic-own", api_key="synthetic",
                             count_tokens=lambda text: max(1, len(str(text)) // 4))
    monkeypatch.setattr(worker, "AuditedModelGateway", FakeGateway)
    def react(prompt, handlers, config):
        assert "桥梁甲，2018年检测。" in prompt
        assert "document-section-input-v1" not in prompt  # The old complete bundle is not appended.
        gateway = FakeGateway.instance
        gateway.on_receipt({"status": "success", "call_id": "synthetic-material-call"})
        payload = {"messages": [{"role": "user", "content": prompt}]}
        audit = {"request_id": "synthetic-material-request", "call_id": "synthetic-material-call",
                 "http_status": 200, "input_delivery_confirmed": True}
        gateway.on_prepared_request(payload, audit)
        gateway.on_input_observed(payload, audit)
        saved = handlers["save_entries"]([{"id": "bridge", "section": "基本信息", "field": "桥梁名称",
            "value": "桥梁甲", "status": "explicit", "source_refs": [{"evidence_id": index.qualify_ref(case["refs"]["unique"])}]}])
        assert saved["ok"], saved
        assert handlers["complete_section"]("合成测试已保存", ["合成fixture仅验证流程，不代表事实质量"])["status"] == "handed_off"
        return {"status": "completed", "native_final_event": True}
    result = worker.run_section(request, tmp_path / "run" / "child", client=client, react_runner=react)
    assert result["status"] == "completed", result.get("error")
    assert result["entries"]["entries"]["bridge"]["source_refs"][0]["evidence_id"].startswith("E")
    # Relocate/delete the original parser inputs; subsequent work uses the pinned run copy.
    case["source_path"].unlink()
    main_client = SimpleNamespace(model="synthetic-own", base_url="http://unused.invalid", calls=[],
        max_model_len=262144, timeout=900, dynamic_output_budget=True, enable_thinking=False,
        temperature=1.0, count_tokens=lambda text: max(1, len(str(text)) // 4))
    wf = DocumentWorkflow(manifest, main_client, {"model": main_client.model}, tmp_path / "run" / "merge",
        material_index=index, settings=DocumentSettings(single_document_agent=True, require_semantic_review=False,
            max_model_len=262144, output_tokens=262144, dynamic_output_budget=True, temperature=1.0))
    import_candidates(wf.workspace, [result], tasks=[task])
    state = wf.workspace.snapshot()
    questions = [{"id": q["id"], "status": "resolved", "reason": "合成fixture的模拟审核，仅验证保存/导出链路"}
                 for q in state["questions"].values()]
    wf.workspace.apply_update({"base_version": state["version"], "questions": questions,
        "exclusions": [{"evidence_id": u["id"], "start": 0, "end": len(u["text"]),
                        "reason": "合成fixture其余原文只作为格式测试背景"} for u in manifest["units"] if u["text"]]})
    assert wf.workspace.delivery_status()["ready"]
    harness = Harness.__new__(Harness)
    harness.wf, harness.client, harness.history = wf, main_client, [{"role": "user", "content": "合成文件自查"}]
    harness.read_all()
    finish = harness.finish()
    assert finish["ok"] and wf._finish_is_current()
    assert len(wf.model_self_check["actual_inspection_receipt_sha256s"]) == 3
    paths = list((tmp_path / "run" / "merge").rglob("*.docx"))
    assert paths and any("桥梁甲" in cell.text for path in paths for table in Document(path).tables
                         for row in table.rows for cell in row.cells)
