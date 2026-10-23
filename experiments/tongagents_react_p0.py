"""Protocol smoke: TongAgents ReactAgent + evidence tool, with a configurable model.

This deliberately uses synthetic evidence. It is a framework smoke test, not a
competition evaluation and does not read the official test set.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tongagents.agents.llm.base import ModelConfig
from tongagents.agents.llm_agent import LLMRunContext, ReactAgent, ReactAgentSetting
from tongagents.tools.tool_manager import tool


EVIDENCE = {
    "EVIDENCE-001": {
        "source_id": "EVIDENCE-001",
        "bridge": "演示桥",
        "year": 2024,
        "component": "桥面铺装",
        "text": "检查记录：桥面铺装在 K12+340 处出现纵向裂缝，宽度约 0.20 mm。2023 年记录同一位置无裂缝。",
        "allowed_conclusion": "2024 年该位置存在纵向裂缝；相对 2023 年同位置记录，可标记为新增候选。",
    }
}

_ACTIVE_TOOL_CALLS: list[dict[str, Any]] | None = None


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return repr(value)


def _event_record(event: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(event).__name__}
    for attr in (
        "content",
        "name",
        "tool_name",
        "args",
        "arguments",
        "result",
        "role",
        "action",
        "status",
        "message",
    ):
        if hasattr(event, attr):
            value = getattr(event, attr)
            if not callable(value):
                result[attr] = _jsonable(value)
    if hasattr(event, "model_dump"):
        try:
            result["model_dump"] = event.model_dump(mode="json")
        except Exception:
            pass
    return result


def _last_text(events: list[Any]) -> str:
    for event in reversed(events):
        content = getattr(event, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _parse_json(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


@tool(
    name="read_evidence",
    description="Read the authorized evidence block by source_id. Call this before making a decision.",
)
def read_evidence(source_id: str) -> dict[str, Any]:
    if _ACTIVE_TOOL_CALLS is not None:
        _ACTIVE_TOOL_CALLS.append(
            {
                "source_id": source_id,
                "time": datetime.now(timezone.utc).isoformat(),
            }
        )
    return EVIDENCE.get(
        source_id,
        {"source_id": source_id, "error": "source_id_not_authorized"},
    )


def run_case(
    *,
    requested_source_id: str,
    model: str,
    url: str,
    api_key: str,
    max_iterations: int,
    max_calls: int,
) -> dict[str, Any]:
    global _ACTIVE_TOOL_CALLS
    tool_calls: list[dict[str, Any]] = []
    _ACTIVE_TOOL_CALLS = tool_calls

    system_prompt = (
        "你是一个桥梁报告证据审查 Agent。只能依据工具返回的证据作答。\n"
        f'先调用 read_evidence(source_id="{requested_source_id}")，不得凭提示中的事实直接回答。\n'
        "工具返回后输出一行严格 JSON，不要 Markdown，字段为："
        '{"decision":"new|not_new|unknown", "evidence_ids":["..."], "reason":"..."}\n'
        "如果工具返回 error 或没有证据，decision 必须是 unknown，evidence_ids 必须为空。"
    )
    settings = ReactAgentSetting(
        llm_config=ModelConfig(
            model_name=model,
            url=url,
            api_key=api_key,
            temperature=0.0,
            max_tokens=300,
            timeout=90,
            stream=False,
        ),
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        max_call_cnt=max_calls,
        max_tool_retry=1,
        same_tool_retry_max=1,
        max_history_len=5,
        tool_call_memory=True,
    )
    result: dict[str, Any] = {
        "requested_source_id": requested_source_id,
        "max_iterations": max_iterations,
        "max_call_cnt": max_calls,
    }
    try:
        agent = ReactAgent(
            dep_context=LLMRunContext(),
            agent_settings=settings,
            tools=[read_evidence],
        )
        events = list(
            agent.step(
                f"请审查 {requested_source_id}，判断这条病害相对上一年度是否属于新增。"
            )
        )
        text = _last_text(events)
        result.update(
            status="completed",
            event_count=len(events),
            events=[_event_record(event) for event in events],
            tool_calls=tool_calls,
            response_text=text,
            response_json=_parse_json(text),
        )
    except Exception as exc:  # preserve failure evidence for audit
        result.update(
            status="error",
            tool_calls=tool_calls,
            error={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=12),
            },
        )
    finally:
        _ACTIVE_TOOL_CALLS = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Model endpoint; use the local own-model bridge for valid experiments")
    parser.add_argument("--model", required=True, help="Model identifier; record the own Qwen base/adapter identity")
    parser.add_argument("--api-key-env", default="DIGUA_OPENAI_API_KEY")
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"missing API key environment variable: {args.api_key_env}")

    started = datetime.now(timezone.utc)
    success = run_case(
        requested_source_id="EVIDENCE-001",
        model=args.model,
        url=args.url,
        api_key=api_key,
        max_iterations=args.max_iterations,
        max_calls=args.max_calls,
    )
    unauthorized = run_case(
        requested_source_id="EVIDENCE-404",
        model=args.model,
        url=args.url,
        api_key=api_key,
        max_iterations=args.max_iterations,
        max_calls=args.max_calls,
    )
    success_json = success.get("response_json") or {}
    bad_json = unauthorized.get("response_json") or {}
    result = {
        "experiment": "tongagents-react-p0",
        "classification": "sdk_protocol_smoke_only",
        "started_at": started.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "sdk": "tongagents-2.7.20",
        "model": args.model,
        "url": args.url,
        "synthetic_evidence_only": True,
        "official_test_set_accessed": False,
        "cases": {"authorized": success, "unauthorized": unauthorized},
        "acceptance": {
            "authorized_completed": success.get("status") == "completed",
            "authorized_tool_called": bool(success.get("tool_calls")),
            "authorized_decision": success_json.get("decision"),
            "authorized_evidence_ids": success_json.get("evidence_ids"),
            "unauthorized_completed": unauthorized.get("status") == "completed",
            "unauthorized_tool_called": bool(unauthorized.get("tool_calls")),
            "unauthorized_decision": bad_json.get("decision"),
            "unauthorized_evidence_ids": bad_json.get("evidence_ids"),
        },
    }
    result["passed"] = all(
        (
            result["acceptance"]["authorized_completed"],
            result["acceptance"]["authorized_tool_called"],
            result["acceptance"]["authorized_decision"] == "new",
            result["acceptance"]["authorized_evidence_ids"] == ["EVIDENCE-001"],
            result["acceptance"]["unauthorized_completed"],
            result["acceptance"]["unauthorized_tool_called"],
            result["acceptance"]["unauthorized_decision"] == "unknown",
            result["acceptance"]["unauthorized_evidence_ids"] == [],
        )
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
