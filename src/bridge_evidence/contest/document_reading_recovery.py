"""Recover a model's unsuccessful workspace proposal without executing it.

The caller supplies a verified terminal model bundle. This module never fixes
JSON, changes a source offset, supplies a missing field or writes a fact. The
next native Agent must decide whether and how to generate a new small update.
"""
from __future__ import annotations

from copy import deepcopy
import json

from .run_state import json_digest


def scan_workspace_retry(bundle: dict, session: dict) -> dict:
    """Select unresolved content work and a proven accepted-content boundary.

    Notes-only writes never resolve content failures. Handler receipt order is
    actual execution order; native tool_calls array order is not. SDK errors
    have no shared execution ordinal, so a confirmed SDK failure is retained
    conservatively even alongside an accepted content write in this response.
    No returned candidate is executed and its raw arguments remain unchanged.
    """
    empty = {"candidate": None, "supersedes_previous": False}
    audit = bundle["audit"]
    if audit.get("request_kind") != "agent_generation" or audit.get("status") != "success":
        return empty
    choices = bundle["response"].get("choices", [])
    # Production accepts a single choice. Do not select among ambiguous ones.
    if len(choices) != 1:
        return empty
    calls = choices[0].get("message", {}).get("tool_calls") or []
    workspace_calls = [row for row in calls if row.get("type") == "function"
                       and row.get("function", {}).get("name") == "update_workspace"]
    sdk_errors = [row for row in session.get("sdk_tool_errors", [])
                  if row.get("tool_name") == "update_workspace"]
    proposals = []
    confirmed_sdk = None
    unconfirmed = None
    for call in reversed(calls):
        function = call.get("function", {})
        if call.get("type") != "function" or function.get("name") != "update_workspace":
            continue
        raw = function.get("arguments")
        if not isinstance(raw, str):
            continue
        parsed = None
        try:
            parsed = json.loads(raw)
            args_sha = json_digest(parsed) if isinstance(parsed, dict) else None
        except (ValueError, TypeError):
            args_sha = None
        update = parsed.get("update") if isinstance(parsed, dict) else None
        if (isinstance(parsed, dict) and set(parsed) == {"update"}
                and isinstance(update, dict) and set(update) <= {"base_version", "working_notes"}):
            continue
        failed_sdk = any(row.get("agent_call_id") == audit.get("call_id") and (
            (row.get("tool_call_id_sha256") == json_digest(call.get("id"))
             if row.get("tool_call_id_sha256") is not None else
             row.get("arguments_text_sha256") == json_digest(raw)
             if row.get("arguments_text_sha256") is not None else len(workspace_calls) == 1))
            for row in sdk_errors)
        candidate = {
            "tool": "update_workspace", "call_id": audit["call_id"],
            "response_sha256": audit["response_sha256"],
            "arguments_text": raw, "arguments_text_sha256": json_digest(raw),
            "accepted_as_workspace_state": False,
            "failure_association": ("confirmed_failed_sdk" if failed_sdk else "unconfirmed_sdk_association"),
            "instruction": "This is a prior unaccepted MODEL proposal, not evidence or a saved fact. "
                           "Only confirmed associations prove this exact action failed; otherwise its execution is unknown. "
                           "Never replay it automatically. Read the current workspace and needed source, "
                           "then generate one small valid native update with current base_version, "
                           "complete patch fields (including section for a row), and source-local intervals.",
        }
        proposals.append((args_sha, candidate))
        has_handler_receipt = any(row.get("name") == "update_workspace" and args_sha is not None
            and row.get("args_sha256") == args_sha and row.get("agent_call_id") == audit.get("call_id")
            for row in session.get("tool_calls", []))
        if failed_sdk and confirmed_sdk is None:
            confirmed_sdk = candidate
        elif sdk_errors and not has_handler_receipt and unconfirmed is None:
            unconfirmed = candidate
    if confirmed_sdk is not None:
        return {"candidate": confirmed_sdk, "supersedes_previous": False}
    for row in reversed(session.get("tool_calls", [])):
        if row.get("name") != "update_workspace" or row.get("agent_call_id") != audit.get("call_id"):
            continue
        matches = [candidate for args_sha, candidate in proposals
                   if args_sha is not None and args_sha == row.get("args_sha256")]
        if not matches:
            continue
        if row.get("status") == "failed":
            candidate = dict(matches[0], failure_association="confirmed_failed_handler")
            return {"candidate": candidate, "supersedes_previous": False}
        if row.get("status") == "success":
            # Unknown native actions are retained as unknown, but must not
            # resurrect failures older than this actual accepted content write.
            return {"candidate": unconfirmed, "supersedes_previous": True}
    return {"candidate": unconfirmed, "supersedes_previous": False}


def pending_workspace_retry(bundle: dict, session: dict) -> dict | None:
    """Compatibility accessor; workflow also consumes the accepted boundary."""
    return scan_workspace_retry(bundle, session)["candidate"]


def bounded_recovery(value: dict, count_tokens, max_tokens: int) -> dict:
    """Keep entire notes/candidate or explicit reading pointers, never fragments."""
    result = deepcopy(value)

    def fits():
        count = count_tokens(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        if type(count) is not int or count < 0:
            raise ValueError("Recovery token counter must return a nonnegative integer")
        return count <= max_tokens

    if fits():
        return result
    candidate = result.get("pending_workspace_retry")
    if isinstance(candidate, dict) and "arguments_text" in candidate:
        del candidate["arguments_text"]
        candidate["arguments_omitted"] = "Whole prior proposal exceeds recovery budget; no partial proposal supplied. Re-read the saved workspace/source and generate a smaller update."
    if fits():
        return result
    if "working_notes" in result:
        del result["working_notes"]
        result["working_notes_read"] = {"tool": "read_workspace_page", "arguments": {
            "view": "working_notes", "cursor": 0, "limit": 1}}
    if fits():
        return result
    raise ValueError("Fixed recovery metadata exceeds the reserved history budget")


LENGTH_RECOVERY_VERSION = "verified-reading-length-smaller-task-v1"
LENGTH_RECOVERY_INSTRUCTION = (
    "上一次阅读调用已经由实际 HTTP 和终态回执确认 finish_reason=length，回答未完成。"
    "这不是已执行的工具操作，也不是已保存的事实；不提供、拼接、修补或重放截断内容。"
    "先 read_workspace 核对当前版本、已保存事实和 working_notes，必要时分页或回读原文。"
    "把尚未完成的工作拆小：下一次只生成一个小事实块或一行 draft_patch，并带最新 base_version。"
    "已成功保存的内容保留，不重复追加；先确认本次写入成功，再保存简短 working_notes 和下一步。"
    "后续由你决定继续拆分与阅读。输出、上下文、累计模型调用、会话和总时间预算均未增加。"
)


def reading_length_attempt(audit: dict, calls: list[dict], dispatches: list[dict],
                           request_record: dict, workflow_identity_sha256: str,
                           session_index: int) -> dict:
    """Verify terminal metadata only; never consume a truncated response body.

    Gateway response hashes cover canonical parsed JSON; client hashes cover
    original HTTP bytes. They are separately retained, never equated. Identity,
    call/request uniqueness, exact request hash, usage and finish bind them.
    This function is called only for a potential confirmed Agent length reply;
    incomplete or conflicting provenance fails closed instead of inferring it.
    """
    def require(condition):
        if not condition:
            raise ValueError("Reading length attempt lacks unique terminal/request provenance")

    def sha(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    require(type(session_index) is int and session_index >= 0)
    require(audit.get("request_kind") == "agent_generation"
            and audit.get("status") == "rejected"
            and audit.get("generation_attempted") is True
            and audit.get("http_status") == 200 and audit.get("finish_reason") == "length")
    call_id, request_id = audit.get("call_id"), audit.get("request_id")
    require(isinstance(call_id, str) and bool(call_id) and isinstance(request_id, str) and bool(request_id))
    matched_calls = [c for c in calls if c.get("call_id") == call_id or c.get("request_id") == request_id]
    matched_dispatches = [d for d in dispatches if d.get("request_id") == request_id or d.get("call_id") == call_id]
    require(len(matched_calls) == len(matched_dispatches) == 1)
    call, dispatch = matched_calls[0], matched_dispatches[0]
    require(call.get("call_id") == call_id and call.get("request_id") == request_id
            and call.get("kind") == "agent_generation" and call.get("status") == "failed"
            and call.get("http_status") == 200 and call.get("finish_reason") == "length"
            and call.get("endpoint") == "/v1/chat/completions")
    require(dispatch.get("call_id") == call_id and dispatch.get("request_id") == request_id
            and dispatch.get("status") == "settled" and dispatch.get("receipt_status") == "rejected"
            and dispatch.get("generation_attempted") is True
            and dispatch.get("request_kind") == "agent_generation"
            and dispatch.get("endpoint") == "/v1/chat/completions")
    request = request_record.get("request")
    request_sha = audit.get("request_sha256")
    require(isinstance(request, dict) and sha(request_sha)
            and json_digest(request) == request_sha
            and request_record.get("workflow_identity_sha256") == workflow_identity_sha256
            and request_record.get("request_id") == request_id
            and request_record.get("request_sha256") == request_sha
            and call.get("input_sha256") == request_sha and dispatch.get("request_sha256") == request_sha)
    maximum = request.get("max_tokens", request.get("max_completion_tokens"))
    require(type(maximum) is int and maximum > 0 and dispatch.get("max_tokens") == maximum
            and request.get("stream") in (None, False)
            and request.get("model") == call.get("model") == dispatch.get("model"))
    usage = audit.get("usage")
    require(isinstance(usage, dict) and usage == call.get("usage"))
    require(all(type(usage.get(key)) is int and usage[key] >= 0
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")))
    require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            and usage["completion_tokens"] <= maximum
            and audit.get("prompt_tokens_checked") == call.get("prompt_tokens_checked")
            == dispatch.get("prompt_tokens_checked") == usage["prompt_tokens"])
    require(sha(audit.get("response_sha256")) and sha(call.get("response_sha256")))
    return {
        "protocol": LENGTH_RECOVERY_VERSION, "session_index": session_index,
        "call_id": call_id, "request_id": request_id, "request_sha256": request_sha,
        "workflow_identity_sha256": workflow_identity_sha256,
        "gateway_canonical_response_sha256": audit["response_sha256"],
        "client_http_response_sha256": call["response_sha256"],
        "http_status": 200, "finish_reason": "length", "generation_attempted": True,
        "gateway_status": "rejected", "client_status": "failed", "usage": deepcopy(usage),
        "max_tokens": maximum, "response_content_supplied": False,
        "accepted_as_workspace_state": False,
    }
