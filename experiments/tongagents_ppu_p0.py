#!/usr/bin/env python3
"""Run the smallest real-model TongAgents loop against a PPU vLLM endpoint.

The PPU model worker is expected to expose an OpenAI-compatible endpoint.  This
client leaves model-runtime imports to the PPU worker, which owns model loading
and Qwen's native XML-to-OpenAI tool-call conversion.  The only input evidence
is the hand-written synthetic block supplied through ``--evidence-json``.
Use ``--write-synthetic-evidence PATH`` to create it without model access.
No competition report, training label, or online test set is read.

A run is successful only when all of the following are true:

* TongAgents completes without an SDK exception;
* the decorated ``read_evidence`` function was actually invoked with the
  authorized evidence ID;
* the evidence bytes match the hand-written synthetic fixture hash;
* the final model response parses as an object containing the same evidence ID,
  and its bridge name and normalized report date match the report's cover.

A tool-call count alone is never treated as success.  On SDK errors or failed
validation this script writes a failure receipt (when ``--output`` is given)
and re-raises the exception, preserving a non-zero exit code and traceback.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


_TOOL_TRACE: list[dict[str, Any]] = []
# This entire source is invented for a protocol smoke test, not a real report.
EVIDENCE_ID = "SYNTHETIC-P0-001"
SYNTHETIC_TEXT = "桥梁名称：合成异型梁桥。报告日期：2024年5月16日。此文本为人工构造测试资料。"
EVIDENCE_SHA256 = hashlib.sha256(SYNTHETIC_TEXT.encode("utf-8")).hexdigest()
EXPECTED_BRIDGE_NAME = "合成异型梁桥"
EXPECTED_REPORT_DATE = "2024-05-16"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    """Keep event receipts JSON serializable without dumping large evidence."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _event_value(value: Any, *, max_text: int = 4096) -> Any:
    """Summarize oversized SDK event fields while retaining an audit hash."""
    value = _jsonable(value)
    if isinstance(value, str) and len(value) > max_text:
        return {
            "truncated": True,
            "chars": len(value),
            "sha256": _sha256_text(value),
            "preview": value[:512],
        }
    if isinstance(value, dict):
        # Tool results may contain the complete 18k-character evidence text.
        return {key: _event_value(item, max_text=max_text) for key, item in value.items()}
    if isinstance(value, list):
        return [_event_value(item, max_text=max_text) for item in value]
    return value


def _event_record(event: Any) -> dict[str, Any]:
    """Project SDK events without hiding their type or useful payload fields."""
    row: dict[str, Any] = {"type": type(event).__name__}
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
                row[attr] = _event_value(value)
    model_dump = getattr(event, "model_dump", None)
    if callable(model_dump):
        try:
            row["model_dump"] = _event_value(model_dump(mode="json"))
        except Exception as exc:  # A receipt must retain event shape even if dumping fails.
            row["model_dump_error"] = {"type": type(exc).__name__, "message": str(exc)}
    return row


def _last_text(events: list[Any]) -> str:
    for event in reversed(events):
        content = getattr(event, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from a final response with optional model residue.

    The parser does not invent fields and does not convert ordinary text to a
    tool call.  It only accepts an object that is syntactically valid JSON.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.S | re.I)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(candidate)
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    for source in candidates:
        try:
            value, _ = decoder.raw_decode(source.lstrip())
        except (json.JSONDecodeError, TypeError, ValueError):
            value = None
        if isinstance(value, dict):
            found.append(value)
    # Thinking/stop-token residue can precede the final object.  Try each
    # opening brace, but still require the JSON decoder to consume an object.
    for match in re.finditer(r"\{", candidate):
        try:
            value, _ = decoder.raw_decode(candidate[match.start():])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(value, dict):
            found.append(value)
    # Prefer the last object: Qwen may leave a schema/example object in a
    # preceding thinking block before emitting the final answer.
    return found[-1] if found else None


def validate_evidence_payload(raw: Any) -> dict[str, Any]:
    """Validate and sanitize the one authorized evidence block.

    The source is hand-written for this smoke test. Only the evidence
    identity, source metadata, and input text are retained.
    """
    if not isinstance(raw, dict):
        raise ValueError("evidence JSON must be an object")
    evidence_id = raw.get("evidence_id") or raw.get("source_id")
    text = raw.get("text")
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError("evidence JSON requires a non-empty evidence_id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("evidence JSON requires non-empty text")
    if any(key in raw for key in ("output", "label", "target", "answer", "gold")):
        raise ValueError("evidence JSON must not contain a gold output/label field")
    if evidence_id != EVIDENCE_ID or raw.get("sample_kind") != "synthetic":
        raise ValueError("this P0 requires the synthetic fixture identity and sample_kind=synthetic")
    if _sha256_text(text) != EVIDENCE_SHA256:
        raise ValueError("synthetic evidence SHA-256 does not match the frozen fixture")
    sanitized: dict[str, Any] = {
        "evidence_id": evidence_id.strip(),
        "source": "hand-written-synthetic-fixture",
        "text": text,
    }
    sanitized["sample_kind"] = "synthetic"
    if raw.get("source_sha256") is not None:
        source_sha = raw["source_sha256"]
        if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_sha):
            raise ValueError("source_sha256 must be a 64-character hex digest")
        sanitized["source_sha256"] = source_sha.lower()
    return sanitized


def normalize_report_date(value: Any) -> str | None:
    """Normalize the supported report-date spellings for this one-sample P0."""
    if not isinstance(value, str):
        return None
    compact = re.sub(r"\s+", "", value)
    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", compact)
    if match:
        parts = [int(part) for part in match.groups()]
    else:
        match = re.fullmatch(r"([0-9〇○零一二三四五六七八九]{4})年([0-9零一二三四五六七八九十]{1,3})月([0-9零一二三四五六七八九十]{1,3})日", compact)
        if not match:
            return None
        digits = {char: str(number) for number, char in enumerate("零一二三四五六七八九")}
        digits.update({"〇": "0", "○": "0"})

        def number(part: str, *, year: bool = False) -> int:
            if part.isascii() and part.isdigit():
                return int(part)
            if year:
                return int("".join(digits[char] for char in part))
            if "十" in part:
                if part.count("十") != 1:
                    raise ValueError("invalid Chinese date numeral")
                tens, ones = part.split("十")
                if len(tens) > 1 or len(ones) > 1:
                    raise ValueError("invalid Chinese date numeral")
                return 10 * (int(digits[tens]) if tens else 1) + (int(digits[ones]) if ones else 0)
            if len(part) != 1:
                raise ValueError("invalid Chinese date numeral")
            return int(digits[part])

        try:
            year, month, day = match.groups()
            parts = [number(year, year=True), number(month), number(day)]
        except (KeyError, ValueError):
            return None
    try:
        return date(*parts).isoformat()
    except ValueError:
        return None


def validate_final_response(response: Any) -> dict[str, Any]:
    """Compare final fields to evaluation-only values from the frozen cover."""
    parsed = response if isinstance(response, dict) else {}
    bridge_name = parsed.get("bridge_name")
    report_date = normalize_report_date(parsed.get("report_date"))
    bridge_matches = isinstance(bridge_name, str) and re.sub(r"\s+", "", bridge_name) == EXPECTED_BRIDGE_NAME
    checks = {
        "final_response_json_valid": isinstance(response, dict),
        "status_matches": parsed.get("status") == "evidence_loaded",
        "evidence_id_matches": parsed.get("evidence_id") == EVIDENCE_ID,
        "bridge_name_matches_source": bridge_matches,
        "report_date_matches_source": report_date == EXPECTED_REPORT_DATE,
    }
    return {
        **checks,
        "final_response_matches_evidence": all(checks.values()),
        "bridge_name": bridge_name,
        "report_date": parsed.get("report_date"),
        "report_date_normalized": report_date,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=False, help="Local vLLM OpenAI base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=False, help="vLLM served model name (including the LoRA name, if used)")
    parser.add_argument("--evidence-json", type=Path, required=False, help="Hand-written synthetic one-block evidence JSON")
    parser.add_argument("--output", type=Path, help="Write the complete JSON receipt to this path")
    parser.add_argument("--api-key", default=os.environ.get("PPU_VLLM_API_KEY", "local-ppu-p0"))
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Pass enable_thinking=false to vLLM; omit to keep the model's default")
    parser.add_argument("--write-synthetic-evidence", type=Path, help="Write the synthetic input to a new file; no SDK or model call")
    parser.add_argument("--self-test", action="store_true", help="Run local pure-function checks without importing TongAgents")
    return parser


def _run_self_test() -> None:
    parsed = parse_json_object("thinking residue {\"status\":\"evidence_loaded\",\"bridge_name\":\"桥\"}")
    assert parsed == {"status": "evidence_loaded", "bridge_name": "桥"}
    parsed = parse_json_object("<think>{\"example\":true}</think>{\"status\":\"evidence_loaded\"}")
    assert parsed == {"status": "evidence_loaded"}
    assert parse_json_object("not JSON") is None
    valid = {"status": "evidence_loaded", "evidence_id": EVIDENCE_ID,
             "bridge_name": EXPECTED_BRIDGE_NAME, "report_date": EXPECTED_REPORT_DATE}
    for spelling in ("2024-05-16", "2024年5月16日", "二〇二四年五月十六日", "二零二四年五月十六日"):
        assert validate_final_response({**valid, "report_date": spelling})["final_response_matches_evidence"]
    for wrong in (
        {**valid, "bridge_name": "不存在的桥"},
        {**valid, "bridge_name": ""},
        {**valid, "report_date": "2024年5月15日"},  # Inspection date is not report date.
        {**valid, "report_date": "2024-02-30"},
        {**valid, "report_date": "二〇二四年十三月十六日"},
        {**valid, "evidence_id": "SYNTHETIC-P0-OTHER"},
    ):
        assert not validate_final_response(wrong)["final_response_matches_evidence"]
    assert validate_evidence_payload({"evidence_id": EVIDENCE_ID, "sample_kind": "synthetic",
                                      "text": SYNTHETIC_TEXT})["text"] == SYNTHETIC_TEXT
    for raw in (
        {"evidence_id": EVIDENCE_ID, "sample_kind": "synthetic", "text": "arbitrary replacement"},
        {"evidence_id": EVIDENCE_ID, "sample_kind": "unknown", "text": "证据"},
        {"evidence_id": "SYNTHETIC-P0-OTHER", "sample_kind": "synthetic", "text": "证据"},
        {"evidence_id": EVIDENCE_ID, "sample_kind": "synthetic", "text": "证据", "output": "gold"},
    ):
        try:
            validate_evidence_payload(raw)
        except ValueError:
            pass
        else:
            raise AssertionError("unverified evidence must be rejected")


def _write_receipt(result: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.url or not args.model or args.evidence_json is None:
        raise ValueError("--url, --model and --evidence-json are required unless --self-test is used")
    if args.max_iterations < 1 or args.max_calls < 1 or args.max_tokens < 1 or args.timeout < 1:
        raise ValueError("iteration/call/token/timeout limits must be positive")

    evidence = validate_evidence_payload(json.loads(args.evidence_json.read_text(encoding="utf-8")))
    evidence_id = evidence["evidence_id"]
    _TOOL_TRACE.clear()
    result: dict[str, Any] = {
        "experiment": "tongagents-ppu-p0",
        "classification": "own_model_synthetic_protocol_smoke",
        "official_test_set_used": False,
        "external_model_used": False,
        "started_at": _utc_now(),
        "endpoint": {"url": args.url, "model": args.model},
        "evidence": {
            "evidence_id": evidence_id,
            "source": evidence["source"],
            "sample_kind": evidence["sample_kind"],
            "text_chars": len(evidence["text"]),
            "text_sha256": _sha256_text(evidence["text"]),
            "source_sha256": evidence.get("source_sha256"),
        },
        "sdk": {"name": "TongAgents", "version": None, "agent": "ReactAgent"},
        "run_settings": {
            "model_provider": "OPENAI_COMPATIBLE",
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "max_iterations": args.max_iterations,
            "max_call_cnt": args.max_calls,
            "stream": False,
            "enable_thinking": False if args.disable_thinking else "model_default",
        },
        "status": "error",
    }

    # Lazy imports keep --self-test useful on a laptop and make the absence of
    # the official wheel an explicit failure rather than a hidden fallback.
    try:
        import importlib
        llm_base = importlib.import_module("tongagents.agents.llm.base")
        ModelConfig = llm_base.ModelConfig
        ModelProvider = llm_base.ModelProvider
        llm_agent = importlib.import_module("tongagents.agents.llm_agent")
        LLMRunContext = llm_agent.LLMRunContext
        ReactAgent = llm_agent.ReactAgent
        ReactAgentSetting = llm_agent.ReactAgentSetting
        tool = importlib.import_module("tongagents.tools.tool_manager").tool
        result["sdk"]["version"] = importlib.metadata.version("tongagents")
    except Exception:
        result["finished_at"] = _utc_now()
        result["error"] = {"type": "TongAgentsImportError", "traceback": traceback.format_exc(limit=20)}
        _write_receipt(result, args.output)
        raise

    tool_trace = _TOOL_TRACE

    @tool(name="read_evidence", description="读取已授权的桥梁报告证据。调用后才能依据原文作答。")
    def read_evidence(evidence_id: str) -> dict[str, Any]:
        started = _utc_now()
        requested_evidence_id = evidence_id
        authorized = requested_evidence_id == evidence["evidence_id"]
        payload = evidence if authorized else {
            "evidence_id": requested_evidence_id,
            "error": "source_id_not_authorized",
        }
        trace = {
            "name": "read_evidence",
            "requested_evidence_id": requested_evidence_id,
            "authorized": authorized,
            "started_at": started,
            "finished_at": _utc_now(),
            "result_sha256": _sha256_json(payload),
            "result_text_chars": len(payload.get("text", "")) if isinstance(payload, dict) else 0,
        }
        tool_trace.append(trace)
        return payload

    system_prompt = (
        "你是桥梁检测报告证据审查 Agent。所有事实必须来自 read_evidence 工具返回的证据。\n"
        f'请先调用 read_evidence(evidence_id="{evidence_id}")，不要凭空回答。工具返回后，'
        "只输出严格 JSON：{\"status\":\"evidence_loaded\",\"evidence_id\":\"...\","
        "\"bridge_name\":\"...\",\"report_date\":\"...\"}。找不到字段时填空字符串。"
    )
    settings = ReactAgentSetting(
        llm_config=ModelConfig(
            model_name=args.model,
            model_provider=ModelProvider.OPENAI_COMPATIBLE,
            url=args.url,
            api_key=args.api_key,
            temperature=0.0,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            stream=False,
            **({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
               if args.disable_thinking else {}),
        ),
        system_prompt=system_prompt,
        max_iterations=args.max_iterations,
        max_call_cnt=args.max_calls,
        max_tool_retry=1,
        same_tool_retry_max=1,
        max_history_len=5,
        tool_call_memory=True,
        tool_choice=["read_evidence"],
        tool_identifier_list=["read_evidence"],
    )

    try:
        agent = ReactAgent(dep_context=LLMRunContext(), agent_settings=settings, tools=None)
        # TongAgents 2.7.20 sends its initial forced tool-selection request
        # with only the step input, before adding system_prompt to the loop.
        # Include the task here too so that first request has the authorized
        # evidence ID and output schema.  No evaluation answers are supplied.
        events = list(agent.step(system_prompt))
        event_rows = [_event_record(event) for event in events]
        response_text = _last_text(events)
        response_json = parse_json_object(response_text)
        final_checks = validate_final_response(response_json)
        authorized_calls = [
            trace for trace in tool_trace
            if trace.get("name") == "read_evidence"
            and trace.get("authorized") is True
            and trace.get("requested_evidence_id") == evidence_id
        ]
        validation = {
            "tool_trace_present": bool(tool_trace),
            "authorized_tool_call_present": bool(authorized_calls),
            "tool_call_matches_evidence": bool(authorized_calls),
            "synthetic_evidence_verified": True,
            **final_checks,
            "passed": bool(authorized_calls) and final_checks["final_response_matches_evidence"],
        }
        result.update({
            "event_count": len(events),
            "events": event_rows,
            "tool_trace": list(tool_trace),
            "response_text": response_text,
            "response_json": response_json,
            "validation": validation,
            "status": "completed" if validation["passed"] else "failed_validation",
        })
        if not validation["passed"]:
            raise RuntimeError("TongAgents run did not produce an authorized evidence trace and valid bridge/date JSON")
    except Exception as exc:
        result.setdefault("tool_trace", list(tool_trace))
        result.setdefault("event_count", 0)
        result["finished_at"] = _utc_now()
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=24),
        }
        _write_receipt(result, args.output)
        # Preserve the SDK exception/validation failure for shell and supervisor.
        raise

    result["finished_at"] = _utc_now()
    _write_receipt(result, args.output)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.write_synthetic_evidence is not None:
        with args.write_synthetic_evidence.open("x", encoding="utf-8") as handle:
            json.dump({"evidence_id": EVIDENCE_ID, "sample_kind": "synthetic",
                       "text": SYNTHETIC_TEXT}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return 0
    if args.self_test:
        _run_self_test()
        print(json.dumps({"self_test": "passed"}, ensure_ascii=False))
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
