"""CPU-only real TongAgents probe using an ephemeral loopback scripted server.

No external model endpoint is accepted. Each case runs in a fresh interpreter
so TongAgents' global tool registry cannot leak between configurations.
"""
from __future__ import annotations

import argparse
import inspect
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import threading
import time


def one_case(case):
    from tongagents.agents.llm.base import ModelConfig, ModelProvider
    from tongagents.agents.llm_agent import LLMRunContext, ReactAgent, ReactAgentSetting
    from tongagents.tools.tool_manager import tool

    requests, executions, native_batches, call_arguments = [], [], [], []

    @tool(name="read_probe_page", description="Read the next program-owned page.",
          retry_times=case["decorator_retry"])
    def read_probe_page(object_id: str):
        executions.append({"object_id": object_id, "page": len(executions) + 1})
        if case.get("failure"):
            raise ValueError("Synthetic probe failure")
        return {"ok": True, "page": len(executions), "complete": len(executions) >= 3}

    cls = type(read_probe_page)
    original_call = cls.call

    def recorded_call(self, *args, **kwargs):
        call_arguments.append({"args": list(args), "kwargs": kwargs})
        return original_call(self, *args, **kwargs)

    cls.call = recorded_call

    class Scripted(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            step = len(requests)
            if step <= 3:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"probe_call_{step}", "type": "function", "function": {
                        "name": "read_probe_page", "arguments": '{"object_id":"same-object"}'}}]}
                finish = "tool_calls"
            else:
                message, finish = {"role": "assistant", "content": "Probe completed."}, "stop"
            body = json.dumps({"id": f"probe-{step}", "object": "chat.completion", "created": int(time.time()),
                "model": "scripted-cpu-only", "choices": [{"index": 0, "message": message,
                "finish_reason": finish}], "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                "total_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Scripted)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class ObservedAgent(ReactAgent):
        def _tool_call(self, tool_calls):
            native_batches.append([{"tool_call_id": c.tool_call_id, "tool_name": c.tool_name,
                                    "arguments": c.arguments} for c in tool_calls])
            return super()._tool_call(tool_calls)

    events, error = [], None
    try:
        settings = ReactAgentSetting(llm_config=ModelConfig(model_name="scripted-cpu-only",
            model_provider=ModelProvider.OPENAI_COMPATIBLE,
            url=f"http://127.0.0.1:{server.server_address[1]}/v1", api_key="probe-no-secret",
            temperature=0, max_tokens=128, timeout=10, stream=False),
            system_prompt="Use the supplied tools and end.", max_iterations=8, max_call_cnt=12,
            max_tool_retry=case["max_tool_retry"], same_tool_retry_max=case["same_tool_retry_max"],
            max_history_len=20, tool_call_memory=True, tool_identifier_list=["read_probe_page"])
        agent = ObservedAgent(dep_context=LLMRunContext(), agent_settings=settings, tools=None)
        for event in agent.step("Read three pages of same-object, then finish."):
            dump = event.model_dump() if hasattr(event, "model_dump") else str(event)
            events.append({"type": type(event).__name__, "value": dump})
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        server.shutdown()
        server.server_close()
    return {"case": case, "sdk_version": version("tongagents"),
        "real_sdk": True, "scripted_model": True, "gpu_requests": 0,
        "tool_call_signature": str(inspect.signature(original_call)),
        "model_requests": len(requests), "executions": executions, "native_batches": native_batches,
        "tool_call_arguments": call_arguments, "events": events, "error": error,
        "tool_replies": [m for request in requests for m in request.get("messages", [])
                         if m.get("role") == "tool"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.case_json:
        result = one_case(json.loads(args.case_json))
    else:
        cases = []
        for name, d, m, s in [("baseline", 1, 1, 1), ("decorator_only", 8, 1, 1),
                             ("max_tool_only", 1, 8, 1), ("same_tool_only", 1, 1, 8),
                             ("both_agent", 1, 8, 8), ("all", 8, 8, 8)]:
            for failure in (False, True):
                cases.append({"name": name + ("_failure" if failure else "_success"),
                    "decorator_retry": d, "max_tool_retry": m, "same_tool_retry_max": s,
                    "failure": failure})
        result = {"schema_version": 1, "cases": []}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for case in cases:
            path = args.output.parent / (case["name"] + ".json")
            process = subprocess.run([sys.executable, "-B", __file__, "--case-json", json.dumps(case),
                                      "--output", str(path)], capture_output=True, text=True, timeout=60)
            record = json.loads(path.read_text()) if path.exists() else {"case": case}
            record["process_returncode"] = process.returncode
            record["process_stderr_tail"] = process.stderr[-3000:]
            result["cases"].append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
