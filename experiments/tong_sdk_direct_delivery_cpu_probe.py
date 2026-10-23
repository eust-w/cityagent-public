"""Real TongAgents + scripted loopback responses; no GPU/model service calls.

This checks native schema/dispatch/recovery/termination. Handlers deliberately
use synthetic records; it is not evidence of report quality or Word completion.
"""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
from pathlib import Path
import socket
import threading
import time


KINDS = ("submit_batch", "submit_partial", "submit_repair", "submit_incomplete",
         "review_normalized", "review_repair")


def run_probe(kind, directory):
    from bridge_evidence.contest import tong_react_adapter as adapter

    requests, executions, identities = [], [], []
    current = {"id": None, "settled": False, "handoff": None}
    is_section = kind.startswith("submit_")
    entry = {"section": "基本信息", "field": "桥梁名称", "value": "合成桥",
             "source_refs": [{"evidence_id": "E-synthetic"}], "status": "explicit"}
    good_submit = {"entries": [entry], "unresolved": [], "complete": True}
    good_review = {"operations": [{"op": "select", "candidate_id": "candidate-one"}],
                   "task_reviews": [{"task_id": "task-one", "reason": "Synthetic source checked"}]}

    def proposal(name, args, step, suffix="a"):
        return {"id": f"native-{step}-{suffix}", "type": "function", "function": {
            "name": name, "arguments": json.dumps(args, ensure_ascii=False)}}

    def scripted_calls(step):
        if kind == "submit_batch" and step == 1:
            args = {**good_submit, "entries": json.dumps(good_submit["entries"]), "unresolved": "[]"}
            return [proposal("submit_section", args, step), proposal("read_entries", {}, step, "b")]
        if kind == "submit_partial" and step <= 2:
            args = {**good_submit, "complete": False} if step == 1 else {"entries": [], "unresolved": [], "complete": True}
            return [proposal("submit_section", args, step)]
        if kind == "submit_repair" and step <= 2:
            args = deepcopy(good_submit)
            if step == 1:
                del args["entries"][0]["status"]
                args["entries"] = json.dumps(args["entries"])
            return [proposal("submit_section", args, step)]
        if kind == "submit_incomplete" and step == 1:
            return [proposal("submit_section", good_submit, step)]
        if kind == "review_normalized" and step == 1:
            return [proposal("review_candidates", {key: json.dumps(value) for key, value in good_review.items()}, step)]
        if kind == "review_repair" and step <= 2:
            args = deepcopy(good_review)
            if step == 1:
                del args["operations"][0]["candidate_id"]
                args["operations"] = json.dumps(args["operations"])
            return [proposal("review_candidates", args, step)]
        return []

    def submit_handler(entries, unresolved, complete=True):
        identities.append(adapter.current_tool_call_identity())
        executions.append({"name": "submit_section", "generation": current["id"],
                           "arguments": deepcopy({"entries": entries, "unresolved": unresolved, "complete": complete})})
        if kind == "submit_incomplete":
            return {"ok": False, "saved": True, "status": "incomplete_reading",
                    "error": {"code": "SOURCE_UNREAD", "message": "Synthetic source remains unread"}}
        if complete:
            current["handoff"] = {"origin_call_id": current["id"], "entries_content_sha256": "synthetic-saved-state",
                                  "document_delivery_complete": False}
            return {"ok": True, "status": "handed_off", "saved": True, "document_delivery_complete": False}
        return {"ok": True, "status": "saved", "saved": True, "document_delivery_complete": False}

    def read_handler(**kwargs):
        identities.append(adapter.current_tool_call_identity())
        executions.append({"name": "read_entries", "generation": current["id"], "arguments": deepcopy(kwargs)})
        return {"entries": [{"id": "synthetic-saved"}]}

    def review_handler(operations, task_reviews, issue_reviews=None):
        identities.append(adapter.current_tool_call_identity())
        executions.append({"name": "review_candidates", "generation": current["id"], "arguments": deepcopy({
            "operations": operations, "task_reviews": task_reviews, "issue_reviews": issue_reviews})})
        return {"ok": True, "saved": True, "document_delivery_complete": False}

    class Scripted(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            step = len(requests)
            current.update(id=f"cpu-{kind}-{step}", settled=False)
            calls = scripted_calls(step)
            message = ({"role": "assistant", "content": None, "tool_calls": calls} if calls else
                       {"role": "assistant", "content": "Scripted CPU probe finished; not document completion."})
            response = {"id": f"response-{step}", "object": "chat.completion", "created": int(time.time()),
                "model": "scripted-cpu-only", "choices": [{"index": 0, "message": message,
                "finish_reason": "tool_calls" if calls else "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            # The scripted generation is already complete. Mark it settled
            # before sending bytes, so a fast client cannot race this flag.
            current["settled"] = True
            self.wfile.write(body)
            self.wfile.flush()

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
        config = {"url": f"http://127.0.0.1:{port}/v1", "model": "scripted-cpu-only",
            "profile": "section" if is_section else "reading", "run_id": f"cpu-{kind}",
            "max_calls": 12, "max_iterations": 8, "max_history_len": 30, "max_tokens": 256,
            "timeout": 10, "current_agent_call_id": lambda: current["id"],
            "same_agent_parse_recovery": True, "parse_recovery_generation_is_settled": lambda: current["settled"]}
        if is_section:
            config.update(terminal_handoff_state=lambda: deepcopy(current["handoff"]),
                          terminal_generation_is_settled=lambda: current["settled"])
        handlers = {"submit_section": submit_handler} if is_section else {"review_candidates": review_handler}
        if kind == "submit_batch":
            handlers["read_entries"] = read_handler
        result = adapter.run_react("Execute only the scripted synthetic CPU test.", handlers, config)
    finally:
        socket.socket.connect = real_connect
        server.shutdown()
        server.server_close()

    advertised = sorted({t["function"]["name"] for p in requests for t in p.get("tools", [])})
    outcome = {"kind": kind, "real_sdk": True, "scripted_model": True, "gpu_requests": 0,
        "result": result, "actual_requests": requests, "handler_executions": executions,
        "trusted_identities": identities, "advertised_tools": advertised}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"direct-{kind}.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2))
    assert result["status"] == "completed", result
    assert advertised == sorted(handlers), advertised
    assert all(identities), identities
    for request in requests:
        for tool in request.get("tools", []):
            name = tool["function"]["name"]
            assert tool["function"]["parameters"] == adapter._normalization_schema_for_tool(name)
    expected_requests = {"submit_batch": 1, "submit_partial": 2, "submit_repair": 2,
                         "submit_incomplete": 2, "review_normalized": 2, "review_repair": 3}[kind]
    assert len(requests) == expected_requests, len(requests)
    expected_handlers = 2 if kind in {"submit_batch", "submit_partial"} else 1
    assert len(executions) == expected_handlers, executions
    if kind in {"submit_batch", "submit_partial", "submit_repair"}:
        assert result["termination_kind"] == "section_terminal_handoff", result
        assert result["native_final_event"] is False
        assert result["terminal_handoff_proof"]["batch_tool_count"] == (2 if kind == "submit_batch" else 1)
        assert result["terminal_handoff_proof"]["stopped_before_next_model_call"] is True
    else:
        assert result.get("termination_kind") != "section_terminal_handoff"
    if kind.endswith("repair"):
        assert len(result["sdk_tool_errors"]) == len(result["parse_recovery"]) == 1, result
        assert executions[0]["generation"].endswith("-2"), executions
        repair = result["parse_recovery"][0]
        assert repair["generation_terminal_verified"] is True
        expected = "entries[0].status" if is_section else "operations[0].candidate_id"
        assert any(i["path"] == expected for i in repair["argument_repair"]["issues"]), repair
        # These are the real next-generation messages received by the scripted
        # endpoint, not a prompt constructed separately for the assertion.
        assert expected in json.dumps(requests[1]["messages"])
    else:
        assert not result["sdk_tool_errors"] and not result["parse_recovery"], result
    if kind == "review_normalized":
        assert executions[0]["arguments"] == {**good_review, "issue_reviews": None}
    if kind == "submit_partial":
        assert executions[0]["arguments"]["complete"] is False
        assert executions[1]["arguments"]["complete"] is True
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    run_probe(args.kind, args.directory)
