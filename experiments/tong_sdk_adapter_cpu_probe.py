"""Exercise the real SDK and CityAgent adapter with scripted local responses.

Only synthetic documents are used. The sole HTTP endpoint is this process's
ephemeral loopback server; connections to any other address/port are rejected.
"""
from copy import deepcopy
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace


def run_probe(kind, directory):
    from bridge_evidence.contest import tong_react_adapter as adapter
    from bridge_evidence.contest.document_tools import DocumentTools
    from bridge_evidence.contest.document_read_navigation import DocumentReadNavigation
    from bridge_evidence.contest.document_workspace_tools import WorkspaceTools, ManagedEntriesPager
    from bridge_evidence.contest.evidence_manifest import manifest_from_text
    from bridge_evidence.contest.run_state import json_digest

    requests, executions, identities, hook_calls = [], [], [], []
    current = {"id": None}
    workflow = None
    source_ledger = None
    stop_kind = kind in {"stop", "stop_section", "stop_merge"}
    prompt = "Read the synthetic object in three generations, then finish."
    warning = "CPU_PROBE_EXACT_NEXT_ACTION: continue the remaining source page."
    if kind in {"source", "source_section", "failure"} or stop_kind:
        from bridge_evidence.contest.document_source_delivery import SourceDeliveryLedger
        manifest = manifest_from_text("合成分页测试。" * 2000)
        source_ledger = SourceDeliveryLedger(manifest, directory / "source-ledger",
            run_id="cpu-probe-run", agent_id=kind)
        source = DocumentTools(manifest, delivery_ledger=source_ledger,
            count_tokens=len, max_tool_tokens=2200, native_response_budget=True)
        unit = manifest["units"][0]
        prompt += "\nInitial original source: " + unit["text"][:32]
        source_ledger.register_initial_message(prompt, [{"evidence_id": unit["id"],
            "start": 0, "end": 32}], role="user")
        navigation = DocumentReadNavigation(source)
        tool_name, arguments = "read_section", {"section_id": "root", "max_units": 1}
        actual = navigation.read_section
        if kind == "failure":
            actual = lambda **_kwargs: {"ok": False, "error": "Synthetic bounded reading failure"}
    elif kind == "workspace":
        workspace_directory = directory / "workspace"
        class Workspace:
            directory = workspace_directory
            state = {"identity": {"document": "synthetic"}, "version": 3, "facts_version": 0,
                "facts": {}, "relations": {}, "questions": {f"Q{i}": {"id": f"Q{i}",
                "question": f"Synthetic question {i}", "status": "open", "valid": True} for i in range(4)},
                "draft": {}, "draft_fact_ids": [], "draft_bindings": {}, "draft_sha256": "x",
                "draft_valid": True, "reads": {}, "image_reads": {}, "exclusions": [], "review": {},
                "reading_progress": {}, "working_notes": {}, "history": []}

            def snapshot(self):
                return deepcopy(self.state)

            def delivery_status(self):
                return {"ready": False, "workspace_version": self.state["version"], "draft_sha256": "x", "blockers": []}
        workspace = Workspace()
        pages = WorkspaceTools(workspace, count_tokens=len, max_tool_tokens=4000, native_response_budget=True)
        tool_name, arguments = "read_workspace_page", {"view": "questions", "limit": 1}
        actual = pages.read_page_managed
    elif kind == "entries":
        from bridge_evidence.contest.document_entries import EntryStore
        manifest = manifest_from_text("合成桥裂缝。")
        store = EntryStore(manifest, "synthetic-section", directory / "entries")
        unit = manifest["units"][0]
        store.record_read(unit["id"], 0, len(unit["text"]))
        store.save_entries([{"id": f"entry-{i}", "section": "病害明细",
            "row": {"病害描述": f"合成病害{i}"}, "status": "explicit",
            "source_refs": [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}]}
            for i in range(4)])
        entries = ManagedEntriesPager(store, count_tokens=len, max_tool_tokens=4000)
        tool_name, arguments = "read_entries", {"limit": 1}
        actual = entries.read_entries
    elif kind == "word":
        from bridge_evidence.contest.document_workflow import DocumentSettings, DocumentWorkflow
        name = "合成桥" + "".join(f"长名称片段{i}。" for i in range(300))
        client = SimpleNamespace(model="scripted-cpu-only", base_url="http://unused.invalid", calls=[],
            max_model_len=262144, timeout=900, dynamic_output_budget=True, enable_thinking=False,
            temperature=0, count_tokens=lambda text: max(1, (len(text) + 3) // 4))
        workflow = DocumentWorkflow(manifest_from_text(name), client, {"model": client.model}, directory / "word",
            settings=DocumentSettings(single_document_agent=True, require_semantic_review=False,
                dynamic_output_budget=True, max_model_len=262144, output_tokens=262144))
        unit = workflow.manifest["units"][0]
        workflow.workspace.record_read(unit["id"], 0, len(unit["text"]))
        workflow.workspace.apply_update({"base_version": workflow.workspace.snapshot()["version"],
            "facts": [{"id": "name", "value": name, "status": "explicit",
                "source_refs": [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}]}],
            "draft": {"基本信息": {"桥梁名称": name}, "维修建议": [], "病害明细": []},
            "draft_fact_ids": ["name"], "draft_bindings": {"基本信息.桥梁名称": ["name"]}})
        tool_name, arguments = "inspect_delivery", {"view": "basic", "limit": 1}
        actual = workflow._inspect_delivery_managed
    else:
        raise ValueError(kind)

    def handler(**kwargs):
        identities.append(adapter.current_tool_call_identity())
        answer = actual(**kwargs)
        executions.append(deepcopy(answer))
        return answer

    def before_model():
        hook_calls.append(len(requests))
        if stop_kind and len(hook_calls) == 3:
            from bridge_evidence.contest.document_multiagent_worker import SectionProgressStalled
            from bridge_evidence.contest.document_merge_progress import MergeProgressStalled
            failure = {"stop": RuntimeError, "stop_section": SectionProgressStalled,
                       "stop_merge": MergeProgressStalled}[kind]
            raise failure("CPU_PROBE_STOP_BEFORE_THIRD_GENERATION")
        return warning if len(hook_calls) == 2 else None

    class Scripted(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            step = len(requests)
            current["id"] = f"cpu-{kind}-generation-{step}"
            if step <= 3:
                calls = [{"id": f"native-{step}-a", "type": "function", "function": {
                    "name": tool_name, "arguments": json.dumps(arguments)}}]
                if step == 1 and kind not in {"failure", "word"}:
                    calls.append({**deepcopy(calls[0]), "id": "native-1-b"})
                message, finish = {"role": "assistant", "content": None, "tool_calls": calls}, "tool_calls"
            else:
                message, finish = {"role": "assistant", "content": "CPU probe completed."}, "stop"
            response = {"id": f"probe-{step}", "object": "chat.completion", "created": int(time.time()),
                "model": "scripted-cpu-only", "choices": [{"index": 0, "message": message,
                "finish_reason": finish}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            if source_ledger is not None:
                # This scripted upstream has actually received this wire body;
                # confirmation is independent of argument parsing by the SDK.
                request_id = f"request-{step}"
                source_ledger.observe_request(payload, {"request_id": request_id})
                source_ledger.confirm_request(request_id, evidence={"kind": "upstream_response",
                    "request_id": request_id, "response_id": response["id"]})
            if workflow is not None:
                audit = {"status": "success", "request_kind": "agent_generation", "http_status": 200,
                    "request_id": f"request-{step}", "call_id": current["id"],
                    "request_sha256": json_digest(payload), "response_sha256": json_digest(response)}
                workflow._on_model_request(payload, audit)
                workflow._on_model_response(response, audit)
                workflow.gateway_receipts.append(audit)
                workflow.client.calls.append({"status": "success", "kind": "agent_generation", "http_status": 200,
                    "call_id": current["id"], "request_id": f"request-{step}", "input_sha256": json_digest(payload)})
                workflow._on_model_receipt(audit)
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Scripted)
    port = server.server_address[1]
    real_connect = socket.socket.connect

    def only_scripted_server(sock, address):
        if not isinstance(address, tuple) or address[:2] != ("127.0.0.1", port):
            raise RuntimeError("CPU probe forbids non-scripted network connections")
        return real_connect(sock, address)

    socket.socket.connect = only_scripted_server
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config = {"url": f"http://127.0.0.1:{port}/v1", "model": "scripted-cpu-only", "run_id": "cpu-probe-run",
            "max_calls": 16, "max_iterations": 10, "max_history_len": 40, "max_tokens": 256,
            "timeout": 10, "current_agent_call_id": lambda: current["id"], "before_native_model": before_model}
        if kind in {"source_section", "entries"}:
            config["profile"] = "section"
        if workflow is not None:
            config["on_tool_result"] = workflow._observe_native_tool
        result = adapter.run_react(prompt,
            {tool_name: handler, "read_source_range": lambda **_kwargs: {"ok": False}}, config)
    finally:
        socket.socket.connect = real_connect
        server.shutdown()
        server.server_close()
    assert adapter.current_tool_call_identity() is None
    advertised = sorted({t["function"]["name"] for p in requests for t in p.get("tools", [])})
    warnings = [sum(warning in json.dumps(m, ensure_ascii=False) for m in p.get("messages", [])) for p in requests]
    native_ids = [i["native_tool_call_id"] if i else None for i in identities]
    outcome = {"kind": kind, "real_sdk": True, "scripted_model": True, "gpu_requests": 0,
        "result": result, "actual_requests": requests, "handler_results": executions,
        "trusted_identities": identities, "advertised_tools": advertised, "warning_counts": warnings,
        "before_native_hook_calls": hook_calls, "native_ids": native_ids,
        "source_delivery": source_ledger.state() if source_ledger is not None else None}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"adapter-{kind}.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2, default=str))
    assert "read_source_range" not in advertised
    assert all(identities), "The real SDK did not supply trusted call identities"
    if source_ledger is not None:
        initial = [p for p in source_ledger.state()["pages"].values() if p["kind"] == "initial"]
        assert len(initial) == 1 and initial[0]["status"] == "confirmed", initial
    if stop_kind:
        assert len(requests) == 2 and result["status"] != "completed", result
    else:
        assert result["status"] == "completed", result
        assert len(requests) == 4 and warnings == [0, 1, 1, 1], warnings
        if kind == "failure":
            # Business error envelopes are visible to the model. They must not
            # permanently ban a managed read whose state can later be repaired.
            # The outer workflow bounds stalled generations; native exception
            # retry behavior is covered by the independent SDK settings probe.
            assert len(executions) == 3, executions
        elif kind in {"source", "source_section", "workspace", "entries"}:
            assert len(executions) == 4 and executions[1].get("replayed") is True, executions
            assert executions[2].get("replayed") is False and executions[3].get("replayed") is False
            assert native_ids == ["native-1-a", "native-1-b", "native-2-a", "native-3-a"]
            if kind == "workspace":
                assert [p["start"] for p in executions] == [0, 0, 1, 2]
            elif kind == "entries":
                assert [p["entries"][0]["id"] for p in executions] == ["entry-0", "entry-0", "entry-1", "entry-2"]
                assert len({p["version"] for p in executions}) == 1
            else:
                ranges = [[(s["start"], s["end"]) for s in p["spans"]] for p in executions]
                assert ranges[0] == ranges[1] and ranges[0] != ranges[2] != ranges[3]
                assert all(p["status"] == "confirmed" for p in source_ledger.state()["pages"].values())
        elif kind == "word":
            assert len(executions) == 3 and [p["cursor"] for p in executions] == [0, 512, 1024]
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["source", "source_section", "workspace", "entries", "word", "failure", "stop",
                                         "stop_section", "stop_merge"], required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    run_probe(args.kind, args.directory)
