"""Experimental reasoning allocations from structured, current host state.

This is a scheduling policy, not a claim about semantic difficulty or evidence
quality. It performs no I/O, creates no facts and does not change max_tokens.
The caller must supply the actual workspace snapshot and still verify that a
successful finish is current before using its smaller final-message budget.
"""
from __future__ import annotations

import json


MAX_VERIFIED_THINKING_BUDGET = 8192
POLICY_LABEL = "document-progress-reasoning-v7-experimental"


def _has_valid_conflict(snapshot):
    version = snapshot.get("version")
    if type(version) is not int or version < 0:
        return False
    for collection in ("facts", "relations", "questions"):
        records = snapshot.get(collection, {})
        if type(records) is not dict:
            continue
        for key, record in records.items():
            if (type(key) is not str or not key or type(record) is not dict
                    or record.get("id") != key or record.get("status") != "conflict"
                    or record.get("valid") is not True or record.get("invalidated_by") != []):
                continue
            record_version = record.get("record_version")
            if type(record_version) is int and 0 < record_version <= version:
                return True
    return False


def _successful_finish(messages, index, calls, snapshot):
    # Never infer successful delivery from a function name or prose. The
    # assistant's self-check and its matching actual tool result must agree
    # with the current snapshot. The worker additionally checks its durable
    # _finish_is_current binding; this pure helper cannot prove provenance.
    if len(calls) != 1 or calls[0]["function"]["name"] != "finish":
        return False
    later = messages[index + 1:]
    if any(type(message) is not dict or message.get("role") != "tool" for message in later):
        return False
    matched = [m for m in later if m.get("tool_call_id") == calls[0]["id"]]
    if len(matched) != 1 or type(matched[0].get("content")) is not str:
        return False
    try:
        result = json.loads(matched[0]["content"])
        arguments = json.loads(calls[0]["function"].get("arguments", ""))
    except (ValueError, TypeError):
        return False
    if type(result) is not dict or type(arguments) is not dict:
        return False
    if "result" in result:
        # The real TongAgents adapter sends this exact native envelope. Do
        # not guess arbitrary nested data keys or accept failed wrappers.
        if (result.get("ok") is not True or result.get("tool") != "finish"
                or result.get("error") or type(result["result"]) is not dict):
            return False
        result = result["result"]
    self_check = arguments.get("self_check")
    if (type(self_check) is not dict or self_check.get("complete") is not True
            or self_check.get("inspected_export") is not True or self_check.get("remaining_issues") != []
            or result.get("ok") is not True or result.get("ready") is not True):
        return False
    version, draft = snapshot.get("version"), snapshot.get("draft_sha256")
    if type(version) is not int or version < 0 or type(draft) is not str or len(draft) != 64:
        return False
    if (type(result.get("workspace_version")) is not int or result["workspace_version"] != version
            or result.get("draft_sha256") != draft
            or type(self_check.get("workspace_version")) is not int or self_check["workspace_version"] != version
            or self_check.get("draft_sha256") != draft):
        return False
    file_sha = result.get("file_sha256")
    return (type(file_sha) is str and len(file_sha) == 64 and self_check.get("file_sha256") == file_sha)


def reasoning_budget_decision(payload, workspace_snapshot):
    """Return an allocation and public structural reason without mutating input.

    Malformed or unmatched tool history gets the conservative 8192 allocation;
    it never authorizes the 2048 successful-finish case. Invalidated conflicts
    and language that mentions conflict are not treated as current conflicts.
    """
    if type(payload) is not dict or type(payload.get("messages")) is not list:
        raise ValueError("Reasoning selection needs actual structured messages")
    if type(workspace_snapshot) is not dict:
        raise ValueError("Reasoning selection needs an actual workspace snapshot")

    def decision(budget, reason):
        return {"budget": budget, "reason": reason, "policy_label": POLICY_LABEL}

    if _has_valid_conflict(workspace_snapshot):
        return decision(8192, "valid_current_workspace_conflict")
    messages = payload["messages"]
    latest = None
    has_tool_result = False
    for index, message in enumerate(messages):
        if type(message) is not dict:
            return decision(8192, "unverified_tool_history")
        has_tool_result |= message.get("role") == "tool"
        calls = message.get("tool_calls")
        if calls is None or calls == []:
            continue
        if message.get("role") != "assistant" or type(calls) is not list:
            return decision(8192, "unverified_tool_history")
        for call in calls:
            if (type(call) is not dict or call.get("type") != "function"
                    or type(call.get("id")) is not str or not call["id"]
                    or type(call.get("function")) is not dict
                    or type(call["function"].get("name")) is not str or not call["function"]["name"]):
                return decision(8192, "unverified_tool_history")
        latest = (index, calls)
    if latest is None:
        return decision(8192, "unmatched_tool_history" if has_tool_result else "initial_source_reading")
    index, calls = latest
    if any(call["function"]["name"] == "inspect_delivery" for call in calls):
        return decision(8192, "actual_file_self_check")
    if _successful_finish(messages, index, calls, workspace_snapshot):
        return decision(2048, "matched_current_successful_finish")
    return decision(4096, "ordinary_document_step")


def choose_reasoning_budget(payload, workspace_snapshot):
    """Selector adapter: return a positive int at or below the verified 8192 cap."""
    return reasoning_budget_decision(payload, workspace_snapshot)["budget"]
