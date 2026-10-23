"""Actual synthetic Word pages and durable wire provenance, without a model.

These tests do not infer factual quality. They verify that omitting mechanical
bindings never lets an old file, another document, or an unseen page pass.
"""
from copy import deepcopy
import json
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import pytest

from bridge_evidence.contest.document_workflow import DocumentSettings, DocumentWorkflow
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import json_digest
from bridge_evidence.contest.tong_react_adapter import wrap_tool_response


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("No network, model, SDK process, or external document in this test")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)


class Harness:
    def __init__(self, root, *, long=False, extra_source=False):
        name = "合成桥" + ("".join(f"合成名称补充{i}。" for i in range(220)) if long else "")
        self.client = SimpleNamespace(model="synthetic-own", base_url="http://unused.invalid", calls=[],
            max_model_len=65536, timeout=180, dynamic_output_budget=True, enable_thinking=False,
            temperature=0, count_tokens=lambda text: max(1, (len(text) + 3) // 4))
        source = name + ("\n单独检索的背景文字" if extra_source else "")
        self.wf = DocumentWorkflow(manifest_from_text(source), self.client,
            {"model":self.client.model}, root,
            settings=DocumentSettings(single_document_agent=True, require_semantic_review=False,
                dynamic_output_budget=True, output_tokens=65536))
        unit = self.wf.manifest["units"][0]
        self.wf.workspace.record_read(unit["id"], 0, len(unit["text"]))
        exclusions = [{"evidence_id": item["id"], "start": 0, "end": len(item["text"]),
            "reason": "合成测试的背景文字，不含交付条目"} for item in self.wf.manifest["units"][1:]]
        self.wf.workspace.apply_update({"base_version":self.wf.workspace.snapshot()["version"],
            "facts":[{"id":"name", "value":name, "status":"explicit",
                "source_refs":[{"evidence_id":unit["id"], "start":0, "end":len(unit["text"])}]}],
            "draft":{"基本信息":{"桥梁名称":name},"维修建议":[],"病害明细":[]},
            "draft_fact_ids":["name"],"draft_bindings":{"基本信息.桥梁名称":["name"]},
            "exclusions": exclusions})
        assert self.wf.workspace.delivery_status()["ready"]
        self.history = [{"role":"user", "content":"合成文档自查"}]

    def call(self, name, arguments, *, messages=None):
        index = len(self.client.calls) + 1
        call_id, request_id = f"call-{index}", f"request-{index}"
        request = {"model":self.client.model, "messages":deepcopy(self.history if messages is None else messages)}
        message = {"role":"assistant", "content":None,"tool_calls":[{
            "id":f"tool-{index}", "type":"function",
            "function":{"name":name, "arguments":json.dumps(arguments, ensure_ascii=False)}}]}
        response = {"choices":[{"message":message, "finish_reason":"tool_calls"}]}
        audit = {"status":"success", "request_kind":"agent_generation", "request_id":request_id,
            "call_id":call_id, "request_sha256":json_digest(request), "response_sha256":json_digest(response)}
        self.wf._on_model_request(request, audit)
        self.wf._on_model_response(response, audit)
        self.wf.gateway_receipts.append(audit)
        self.client.calls.append({"status":"success", "kind":"agent_generation", "http_status":200,
            "call_id":call_id,"request_id":request_id,"input_sha256":json_digest(request)})
        self.wf._on_model_receipt(audit)
        return message

    def inspect(self, view, *, limit=4, restart=False, defaults_explicit=False, legacy_cursor=None):
        arguments = {"view":view}
        if defaults_explicit or limit != 4:
            arguments["limit"] = limit
        if defaults_explicit or restart:
            arguments["restart"] = restart
        if legacy_cursor is not None:
            arguments = {"view":view,"cursor":legacy_cursor,"limit":limit}
        message = self.call("inspect_delivery", arguments)
        result = (self.wf._inspect_delivery(view, legacy_cursor, limit) if legacy_cursor is not None
                  else self.wf._inspect_delivery_managed(view, limit=limit, restart=restart))
        envelope = wrap_tool_response("inspect_delivery", result)
        self.wf._observe_native_tool("inspect_delivery", arguments, envelope)
        self.history += [message, {"role":"tool", "tool_call_id":message["tool_calls"][0]["id"],
            "content":json.dumps(envelope, ensure_ascii=False)}]
        return result

    def read_all(self):
        for view in ("basic","advice","defects"):
            while self.inspect(view)["next_cursor"] is not None:
                pass

    def finish(self, statement=None, *, messages=None):
        statement = deepcopy(statement) if statement is not None else {
            "inspected_export":True,"complete":True,"basis":"合成模型检查三表及原文后提出的公开结论。",
            "remaining_issues":[]}
        self.call("finish", {"self_check":statement}, messages=messages)
        return self.wf._finish(statement)

    def revise(self):
        self.wf.workspace.apply_update({"base_version":self.wf.workspace.snapshot()["version"],
            "facts":[{"id":"name","value":"合成桥修订", "status":"explicit",
                "source_refs":self.wf.workspace.snapshot()["facts"]["name"]["source_refs"]}],
            "draft_patch":[{"op":"set_basic","field":"桥梁名称","value":"合成桥修订","fact_ids":["name"]}]})
        assert self.wf.workspace.delivery_status()["ready"]


def test_short_statement_binds_actual_pages_and_keeps_original_model_statement(tmp_path):
    h = Harness(tmp_path)
    h.read_all()
    result = h.finish()
    assert result["ok"] and h.wf._finish_is_current()
    check = h.wf.model_self_check
    assert set(check["model_statement"]) == {"inspected_export","complete","basis","remaining_issues"}
    assert check["verified_file_binding"] == {key:result[key] for key in ("workspace_version","draft_sha256","file_sha256")}
    assert len(check["actual_inspection_receipt_sha256s"]) == 3
    assert check["host_semantic_verdict"] is None


@pytest.mark.parametrize("field,value", [("workspace_version", 0), ("draft_sha256", "wrong"), ("file_sha256", "wrong")])
def test_explicit_wrong_legacy_binding_is_rejected_not_corrected(tmp_path, field, value):
    h = Harness(tmp_path)
    h.read_all()
    statement = {"inspected_export":True,"complete":True,"basis":"检查完成", "remaining_issues":[],
        **{key:h.wf.artifact_delivery[key] for key in ("workspace_version","draft_sha256","file_sha256")}}
    statement[field] = value
    with pytest.raises(ValueError, match="Explicit self_check." + field):
        h.finish(statement)
    assert h.wf.last_finish is None


def test_matching_legacy_binding_and_explicit_cursor_still_work(tmp_path):
    h = Harness(tmp_path)
    for view in ("basic","advice","defects"):
        h.inspect(view, legacy_cursor=0)
    statement = {"inspected_export":True,"complete":True,"basis":"检查完成", "remaining_issues":[],
        **{key:h.wf.artifact_delivery[key] for key in ("workspace_version","draft_sha256","file_sha256")}}
    assert h.finish(statement)["ok"]


def test_original_receipt_without_body_hash_still_requires_complete_word_page(tmp_path):
    h = Harness(tmp_path)
    h.read_all()
    for message in h.history:
        if message["role"] != "tool":
            continue
        envelope = json.loads(message["content"])
        proof = envelope["result"]["inspection_proof"]
        old_hash = proof.pop("receipt_sha256")
        proof.pop("tool_result_sha256")
        proof.pop("managed_restart")
        proof["receipt_sha256"] = json_digest(proof)
        index = next(i for i, stored in enumerate(h.wf.delivery_inspections) if stored["receipt_sha256"] == old_hash)
        h.wf.delivery_inspections[index] = deepcopy(proof)
        message["content"] = json.dumps(envelope)
    assert h.finish()["ok"]
    forged = deepcopy(h.history)
    tool = next(message for message in forged if message["role"] == "tool")
    envelope = json.loads(tool["content"])
    envelope["result"]["text"] = "only copied metadata"
    tool["content"] = json.dumps(envelope)
    with pytest.raises(ValueError, match="has not actually received"):
        h.finish(messages=forged)


def test_old_check_does_not_rebind_after_change_and_new_pages_enable_finish(tmp_path):
    h = Harness(tmp_path)
    h.read_all()
    old_messages, old_file = deepcopy(h.history), h.wf.artifact_delivery["file_sha256"]
    h.revise()
    assert h.finish(messages=old_messages)["ok"] is False
    first = h.inspect("basic")
    assert first["cursor"] == 0 and first["receipt"]["file_sha256"] != old_file
    h.inspect("advice")
    h.inspect("defects")
    with pytest.raises(ValueError, match="has not actually received"):
        h.finish(messages=old_messages)  # Host has new pages; that request does not.
    assert h.finish()["ok"]


def test_missing_page_or_same_generation_read_cannot_finish(tmp_path):
    h = Harness(tmp_path)
    h.inspect("basic")
    h.inspect("advice")
    assert h.finish()["ok"] is False
    h.inspect("defects")
    h.wf.delivery_inspections[-1]["call_id"] = f"call-{len(h.client.calls) + 1}"
    with pytest.raises(ValueError, match="has not actually received"):
        h.finish()


@pytest.mark.parametrize("tamper", ["missing", "summary_only", "text", "proof", "wrong_role"])
def test_full_tool_payload_must_be_in_actual_finish_request(tmp_path, tamper):
    h = Harness(tmp_path)
    h.read_all()
    messages = deepcopy(h.history)
    target = next(m for m in messages if m["role"] == "tool")
    envelope = json.loads(target["content"])
    if tamper == "missing":
        messages.remove(target)
    elif tamper == "summary_only":
        target["content"] = json.dumps(envelope["result"]["inspection_proof"])
    elif tamper == "text":
        envelope["result"]["text"] = "原文已压缩，不再包含实际文件正文"
        target["content"] = json.dumps(envelope)
    elif tamper == "proof":
        envelope["result"]["inspection_proof"]["end"] = 0
        target["content"] = json.dumps(envelope)
    else:
        target["role"] = "user"
    with pytest.raises(ValueError, match="has not actually received"):
        h.finish(messages=messages)


def test_changed_word_bytes_are_detected_even_with_valid_request_receipts(tmp_path):
    h = Harness(tmp_path)
    h.read_all()
    Path(h.wf.artifact_delivery["docx_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Word has changed"):
        h.finish()


def test_other_document_receipts_do_not_grant_reading_or_finish(tmp_path):
    one, two = Harness(tmp_path / "one"), Harness(tmp_path / "two")
    one.read_all()
    assert two.finish(messages=one.history)["ok"] is False
    two.read_all()
    with pytest.raises(ValueError, match="has not actually received"):
        two.finish(messages=one.history)


def test_managed_pages_use_per_view_receipts_not_copied_positions(tmp_path):
    h = Harness(tmp_path, long=True)
    first = h.inspect("basic", limit=1)
    second = h.inspect("basic", limit=1, defaults_explicit=True)
    assert first["cursor"] == 0 and second["cursor"] == first["end"] > 0
    assert h.inspect("advice", limit=1)["cursor"] == 0
    assert h.inspect("basic", limit=1)["cursor"] == second["end"]
    restarted = h.inspect("basic", limit=1, restart=True)
    assert restarted["cursor"] == 0
    assert h.inspect("basic", limit=1)["cursor"] == restarted["end"]
    h.revise()
    assert h.inspect("basic", limit=1)["cursor"] == 0


def test_managed_paging_does_not_skip_a_missing_prefix(tmp_path):
    h = Harness(tmp_path, long=True)
    h.inspect("basic", limit=1, legacy_cursor=512)
    assert h.inspect("basic", limit=1)["cursor"] == 0


def test_failed_or_unobserved_tool_return_never_moves_managed_cursor(tmp_path):
    h = Harness(tmp_path, long=True)
    h.call("inspect_delivery", {"view":"basic", "limit":1})
    h.wf._inspect_delivery_managed("basic", limit=1)
    assert not h.wf.delivery_inspections
    assert h.inspect("basic", limit=1)["cursor"] == 0


@pytest.mark.parametrize("semantic", ["inspected_export", "complete", "remaining_issues"])
def test_omitting_hashes_never_supplies_models_semantic_decision(tmp_path, semantic):
    h = Harness(tmp_path)
    h.read_all()
    statement = {"inspected_export":True,"complete":True,"basis":"有未解决事项", "remaining_issues":[]}
    statement[semantic] = ["需要模型继续判断"] if semantic == "remaining_issues" else False
    assert h.finish(statement)["ok"] is False
    assert h.wf.last_finish is None


def test_reading_only_version_and_notes_keep_file_and_accepted_check_current(tmp_path):
    h = Harness(tmp_path, extra_source=True)
    h.read_all()
    assert h.finish()["ok"]
    artifact = deepcopy(h.wf.artifact_delivery)
    check = deepcopy(h.wf.model_self_check)
    extra = h.wf.manifest["units"][1]
    h.wf.workspace.record_read(extra["id"], 0, len(extra["text"]))
    h.wf.workspace.apply_update({"base_version": h.wf.workspace.snapshot()["version"],
        "working_notes": {"next": "仅更新导航笔记，没有改变事实或交付内容"}})
    state = h.wf.workspace.snapshot()
    assert state["version"] > artifact["workspace_version"]
    assert h.wf._finish_is_current()
    assert h.wf.model_self_check == check
    assert h.wf.artifact_manager.matches_snapshot(state, artifact)
    assert h.wf._artifact_read_coverage(state)
    # Re-reading an unchanged Word keeps the original export identity.
    page = h.inspect("basic", restart=True)
    assert page["inspection_proof"]["read_workspace_version"] == state["version"]
    assert page["inspection_proof"]["workspace_version"] == artifact["workspace_version"]
    assert h.wf.artifact_delivery == artifact
    assert h.finish()["ok"]


@pytest.mark.parametrize("change", ["fact", "source_refs", "binding", "relation", "question", "exclusion"])
def test_evidence_changes_expire_model_check_without_reexporting_identical_word(tmp_path, change):
    h = Harness(tmp_path)
    h.read_all()
    assert h.finish()["ok"]
    artifact = deepcopy(h.wf.artifact_delivery)
    old_semantic = h.wf.model_self_check["semantic_state_sha256"]
    before = h.wf.workspace.snapshot()
    ref = deepcopy(before["facts"]["name"]["source_refs"])
    update = {"base_version": before["version"]}
    if change in {"fact", "source_refs"}:
        update["facts"] = [{"id": "name", "value": "合成桥",
            "status": "inferred" if change == "fact" else "explicit",
            "source_refs": ref if change == "fact" else [{**ref[0], "end": 2}]}]
        update["draft_patch"] = [{"op": "bind_output", "path": "基本信息.桥梁名称", "fact_ids": ["name"]}]
    elif change == "binding":
        update["facts"] = [{"id": "new-name-evidence", "value": "合成桥", "status": "explicit", "source_refs": ref}]
        update["draft_patch"] = [{"op": "bind_output", "path": "基本信息.桥梁名称", "fact_ids": ["new-name-evidence"]}]
    elif change == "relation":
        update["relations"] = [{"id": "same-bridge", "value": "指向同一桥梁", "depends_on": ["name"]}]
    elif change == "question":
        update["questions"] = [{"id": "scope", "question": "桥名适用范围", "status": "resolved", "reason": "模型核对后确认"}]
    else:
        update["exclusions"] = [{**ref[0], "reason": "模型新增的范围判断，必须重新自查"}]
    h.wf.workspace.apply_update(update)
    current = h.wf.workspace.snapshot()
    assert current["draft_sha256"] == before["draft_sha256"]
    assert h.wf.workspace.delivery_status()["ready"]
    assert h.wf.artifact_manager.matches_snapshot(current, artifact)
    assert h.wf._artifact_read_coverage(current)
    assert not h.wf._finish_is_current()
    # A fresh actual model decision may use the unchanged Word pages. The
    # model still decides semantics; the host supplies no replacement verdict.
    assert h.finish()["ok"]
    assert h.wf._finish_is_current()
    assert h.wf.artifact_delivery == artifact
    assert h.wf.model_self_check["semantic_state_sha256"] != old_semantic


def test_prepared_finish_cannot_rebind_after_same_batch_changes_semantic_state(tmp_path):
    h = Harness(tmp_path)
    h.read_all()
    statement = {"inspected_export": True, "complete": True,
        "basis": "请求发出时完成检查", "remaining_issues": []}
    h.call("finish", {"self_check": statement})
    h.wf.workspace.apply_update({"base_version": h.wf.workspace.snapshot()["version"],
        "questions": [{"id": "new-question", "status": "resolved", "reason": "请求后才完成的新判断"}]})
    with pytest.raises(ValueError, match="predates the current facts"):
        h.wf._finish(statement)
    assert h.wf.last_finish is None


def test_old_explicit_audit_version_and_missing_receipts_still_reject_after_read_only_update(tmp_path):
    h = Harness(tmp_path, extra_source=True)
    h.read_all()
    old = h.wf.workspace.snapshot()["version"]
    extra = h.wf.manifest["units"][1]
    h.wf.workspace.record_read(extra["id"], 0, len(extra["text"]))
    assert h.wf._artifact_read_coverage(h.wf.workspace.snapshot())
    with pytest.raises(ValueError, match="Explicit self_check.workspace_version"):
        h.finish({"inspected_export": True, "complete": True, "basis": "旧显式版本不得自动改写",
            "remaining_issues": [], "workspace_version": old})
    with pytest.raises(ValueError, match="has not actually received"):
        h.finish(messages=[{"role": "user", "content": "只有摘要说已检查三个表，没有真实页回执"}])


@pytest.mark.parametrize("change", ["scope", "document", "contract", "workspace_identity", "draft"])
def test_word_content_identity_includes_document_scope_and_export_contract(tmp_path, monkeypatch, change):
    import bridge_evidence.contest.document_delivery_artifacts as artifacts
    h = Harness(tmp_path)
    h.read_all()
    state, record = h.wf.workspace.snapshot(), deepcopy(h.wf.artifact_delivery)
    assert artifacts.artifact_matches_snapshot(state, record)
    if change == "scope":
        record["scope"]["new-processing-restriction"] = True
    elif change == "document":
        record["document_id"] += "-other"
    elif change == "contract":
        monkeypatch.setattr(artifacts, "VALIDATOR_VERSION", "different-contract")
    elif change == "workspace_identity":
        state["identity"] = {"different": "document-workspace"}
    else:
        state["draft"]["基本信息"]["桥梁名称"] = "不同桥"
        state["draft_sha256"] = json_digest(state["draft"])
    assert not artifacts.artifact_matches_snapshot(state, record)


def test_source_read_is_pending_until_actual_input_observed_even_if_output_will_fail(tmp_path):
    h = Harness(tmp_path, extra_source=True)
    extra = h.wf.manifest["units"][1]
    result = h.wf.tools.read_source(extra["id"])
    assert result["ok"] and result["source_delivery_receipt"]
    assert extra["id"] not in h.wf.workspace.snapshot()["reads"]
    payload = {"messages": [{"role": "tool", "content": json.dumps(
        wrap_tool_response("read_source", result), ensure_ascii=False)}]}
    audit = {"request_kind": "agent_generation", "request_id": "source-failed-output",
        "request_sha256": json_digest(payload)}
    h.wf._on_model_request(payload, audit)
    assert extra["id"] not in h.wf.workspace.snapshot()["reads"]
    # This proof comes from the gateway before SDK output/argument validation.
    # Neither a usable response nor successful tool execution is needed here.
    h.wf._source_input_observed(payload, {**audit, "http_status": 200,
        "input_delivery_confirmed": True, "input_delivery_evidence": "generated_stream_data"})
    assert h.wf.workspace.snapshot()["reads"][extra["id"]] == [[0, len(extra["text"])]]
    assert h.wf.source_delivery.is_confirmed(result["source_delivery_receipt"])


def test_source_headers_dispatch_or_summary_do_not_confirm_original_text(tmp_path):
    h = Harness(tmp_path, extra_source=True)
    extra = h.wf.manifest["units"][1]
    result = h.wf.tools.read_source(extra["id"])
    payload = {"messages": [{"role": "user", "content": "摘要声称读完了背景文字"}]}
    audit = {"request_kind": "agent_generation", "request_id": "source-summary-only",
        "request_sha256": json_digest(payload), "http_status": 200}
    h.wf._on_model_request(payload, audit)
    with pytest.raises(ValueError, match="upstream generation evidence"):
        h.wf._source_input_observed(payload, audit)
    h.wf._source_input_observed(payload, {**audit, "input_delivery_confirmed": True})
    assert extra["id"] not in h.wf.workspace.snapshot()["reads"]
    assert not h.wf.source_delivery.is_confirmed(result["source_delivery_receipt"])


def test_initial_full_message_uses_exact_registered_prompt_and_request_identity(tmp_path):
    h = Harness(tmp_path, extra_source=True)
    extra = h.wf.manifest["units"][1]
    prompt = "程序生成的初始原文：" + extra["text"] + "\n当前准确状态：尚未提取"
    h.wf.source_delivery.register_initial_message(prompt,
        [{"evidence_id": extra["id"], "start": 0, "end": len(extra["text"])}])
    payload = {"messages": [{"role": "user", "content": prompt}]}
    audit = {"request_kind": "agent_generation", "request_id": "source-initial",
        "request_sha256": json_digest(payload), "input_delivery_confirmed": True}
    with pytest.raises(ValueError, match="final payload was not observed"):
        h.wf._source_input_observed(payload, audit)
    h.wf._on_model_request(payload, audit)
    with pytest.raises(ValueError, match="differs from prepared"):
        h.wf._source_input_observed({"messages": []}, audit)
    h.wf._source_input_observed(payload, audit)
    version = h.wf.workspace.snapshot()["version"]
    h.wf._source_input_observed(payload, audit)
    assert h.wf.workspace.snapshot()["version"] == version
    assert h.wf.workspace.snapshot()["reads"][extra["id"]] == [[0, len(extra["text"])]]
