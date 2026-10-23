#!/usr/bin/env python3
"""Offline, single-arm audit of explicitly selected training-text document runs.

No SDK, model, network, input discovery, or inference labels are used here.
Required: --batch-manifest --work-dir --result-dir --code-root --supervisor-status.
Optional scoring manifest: schema_version=1, never_include_in_model_inputs=true,
entries=[{evidence_id,path,sha256}]. Label paths are relative to that manifest.
Optional semantic manifest: schema_version=1, independent=true, entries containing
evidence_id, input_json_sha256, original_report_sha256, report_sha256 (canonical JSON),
reviewer, coverage="complete_supplied_text", findings=[{category:"A|B|C|D",...}],
omissions=[], accepted=bool. Accepted B findings require marked_as_inference=true
AND allowed_by_output_contract=true. Unknown original-report identity stays null.

Semantic acceptance is an externally supplied, hash-bound review, not generated
by this program. It does not establish original Word/image coverage. F1 is only
the repository's normalized exact-match training-label proxy. No arm comparison
or speedup claim is made. Missing evidence fails closed without dropping samples.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.document_delivery_validation import validate_delivery_docx
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.evaluation.generation import aggregate_reports, score_report

SCOPE = "official_training_text_only"
GENERATION = {"generation", "agent_generation"}
REVIEW_PROTOCOLS = {"bidirectional-semantic-review-v2", "bidirectional-semantic-review-v3", "bidirectional-semantic-review-v4"}
LIMITATIONS = ["Training provenance is declared by the explicit input manifest, not independently certified.",
              "Supplied extracted text only: original DOCX layout, embedded objects and images are not verified.",
              "CLI completion and DOCX fidelity are not independent semantic acceptance.",
              "Hash/receipt consistency is not cryptographic attestation of server execution or live weights.",
              "No matching two-arm frozen protocol has been verified; no speedup is reported."]


class Invalid(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise Invalid(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe_file(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink_rejected")
    require(path.is_file(), "file_missing")
    return path


def file_digest(path):
    return hashlib.sha256(safe_file(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    def nonfinite(_):
        raise Invalid("nonfinite_json")
    return json.loads(safe_file(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique, parse_constant=nonfinite)


def relative_file(root, name):
    require(isinstance(name, str) and bool(name) and "\\" not in name, "unsafe_relative_path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name
            and name != ".", "unsafe_relative_path")
    return Path(root) / name


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def number(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def named(rows, key):
    require(isinstance(rows, list), "records_not_array")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get(key), str) and row[key]
                and row[key] not in result, "record_identity_invalid")
        result[row[key]] = row
    return result


def error_code(exc):
    # Never dump label/source text or arbitrary parsing exceptions into output.
    return str(exc) if isinstance(exc, Invalid) else type(exc).__name__


def optional_json(path):
    try:
        return read_json(path), None
    except Exception as exc:
        return None, error_code(exc)


def selection(path, run_identity):
    raw = read_json(path)
    require(isinstance(raw, dict) and set(raw) == {"schema_version", "scope", "entries"}
            and type(raw["schema_version"]) is int and raw["schema_version"] == 1
            and raw["scope"] == SCOPE, "invalid_training_batch")
    entries = named(raw["entries"], "path")
    require(bool(entries), "empty_selection")
    require(len({p.casefold() for p in entries}) == len(entries), "ambiguous_source_paths")
    recorded = run_identity.get("training_evidence_batch", {})
    try:
        by_path = named(recorded.get("entries", []), "path")
    except Exception:
        by_path = {}  # Corrupt run metadata must not erase the explicit selection.
    rows, seen = [], set()
    for entry in entries.values():
        require(set(entry) == {"path", "sha256"} and sha(entry["sha256"]), "invalid_selection_entry")
        source = relative_file(Path(path).parent, entry["path"])
        prior = by_path.get(entry["path"], {})
        row = {"path": entry["path"], "source_sha256": entry["sha256"],
               "evidence_id": prior.get("evidence_id"), "source_path": source, "errors": []}
        try:
            require(file_digest(source) == entry["sha256"], "source_hash_mismatch")
            item = read_json(source)
            require(isinstance(item, dict) and not set(item) - {
                "evidence_id", "source", "training_line", "text", "source_sha256", "text_sha256"},
                "source_contains_non_input_fields")
            name = item.get("evidence_id")
            require(isinstance(name, str) and name not in {"", ".", ".."}
                    and "/" not in name and "\\" not in name, "unsafe_evidence_id")
            recorded_name = row["evidence_id"]
            row["evidence_id"] = name
            require(recorded_name in (None, name), "recorded_source_id_mismatch")
            require(type(item.get("training_line")) is int and item["training_line"] > 0
                    and isinstance(item.get("text"), str) and item["text"].strip(), "invalid_training_text")
            require(item.get("text_sha256") in (None, hashlib.sha256(item["text"].encode()).hexdigest()),
                    "text_hash_mismatch")
            manifest = manifest_from_text(item["text"])
            manifest.update(training_source={k: v for k, v in item.items() if k != "text"},
                            source_json_sha256=entry["sha256"])
            row["manifest"] = manifest
            row["original_report_sha256"] = item.get("source_sha256")
        except Exception as exc:
            row["errors"].append(error_code(exc))
        name = row["evidence_id"]
        if name is not None:
            require(isinstance(name, str) and name not in {"", ".", ".."} and "/" not in name
                    and "\\" not in name and name.casefold() not in seen, "ambiguous_evidence_id")
            seen.add(name.casefold())
        rows.append(row)
    return rows, file_digest(path)


def run_gate(record, rows, batch_sha, code_root):
    require(record.get("schema_version") == 1 and record.get("run_fingerprint") == digest(record["identity"]),
            "run_fingerprint_mismatch")
    identity = record["identity"]
    require(identity.get("workflow") == "document-agent" and identity.get("engine") == "tongagents",
            "not_document_agent_run")
    require(identity.get("versions", {}).get("review_version") in REVIEW_PROTOCOLS, "unsupported_review_protocol")
    code = identity.get("code", {})
    required = {"run.py", *["src/bridge_evidence/contest/" + name for name in (
        "document_workflow.py", "document_workspace.py", "document_delivery_validation.py", "tong_react_adapter.py")]}
    require(isinstance(code, dict) and required <= code.keys(), "incomplete_code_identity")
    for name, expected in code.items():
        require(sha(expected) and file_digest(relative_file(code_root, name)) == expected, "code_hash_mismatch")
    bound = identity.get("training_evidence_batch", {})
    require(bound.get("scope") == SCOPE and bound.get("manifest_sha256") == batch_sha, "batch_binding_mismatch")
    expected = [(r["path"], r["source_sha256"], r["evidence_id"]) for r in rows]
    entries = bound.get("entries", [])
    require([(r.get("path"), r.get("sha256"), r.get("evidence_id")) for r in entries] == expected,
            "selected_sources_mismatch")
    require(identity.get("sources") == {r["source_path"]: r["sha256"] for r in entries}, "run_sources_mismatch")


def generation_ledger(records):
    """Deduplicate receipts, never turn SDK sessions or dispatch intents into calls."""
    calls, dispatches, gateways, issues = {}, {}, {}, []
    for record in records:
        for field, key, target in (("model_calls", "call_id", calls), ("model_dispatches", "request_id", dispatches),
                                   ("gateway_receipts", "request_id", gateways)):
            items = record.get(field)
            if not isinstance(items, list):
                issues.append("receipt_ledger_incomplete:" + field)
                continue
            for item in items:
                if not isinstance(item, dict):
                    issues.append("malformed_receipt:" + field)
                    continue
                if field == "model_calls" and item.get("kind") not in GENERATION:
                    continue
                if field == "model_calls" and not (item.get("endpoint") == "/v1/chat/completions"
                        and item.get("status") in {"success", "failed"} and sha(item.get("input_sha256"))):
                    issues.append("generation_receipt_not_http_attempt")
                    continue
                ident = item.get(key)
                if (field == "gateway_receipts" and ident is None
                        and item.get("generation_attempted") is not True and item.get("status") == "failed"):
                    continue  # Rejected preflight; no actual generation dispatch.
                if not isinstance(ident, str) or not ident:
                    issues.append("receipt_identity_missing")
                    continue
                # Pre-generation failures may have no dispatch/request id.
                if ident in target and target[ident] != item:
                    issues.append("conflicting_receipt:" + field)
                target[ident] = item
    unknown = []
    for ident, dispatch in dispatches.items():
        gateway = gateways.get(ident)
        if (dispatch.get("status") != "settled" or not gateway
                or gateway.get("generation_attempted") not in (True, False)):
            unknown.append(ident)
        elif gateway.get("generation_attempted") is True and gateway.get("call_id") not in calls:
            unknown.append(ident)
    for call in calls.values():
        request = call.get("request_id")
        if request not in dispatches or request not in gateways:
            issues.append("generation_without_dispatch_receipt")
    return {"known_http_attempts": len(calls), "known_http_200_receipts": sum(
                c.get("http_status") == 200 for c in calls.values()),
            "dispatch_intents": len(dispatches), "outcome_unknown_dispatches": len(unknown),
            "outcome_unknown_request_ids": unknown,
            "complete": bool(records) and not unknown and not issues,
            "issues": sorted(set(issues)), "calls": calls, "dispatches": dispatches, "gateways": gateways}


def successful_call(call_id, kind, ledger, directory, workflow_identity):
    call = ledger["calls"].get(call_id, {})
    dispatch = ledger["dispatches"].get(call.get("request_id"), {})
    gateway = ledger["gateways"].get(call.get("request_id"), {})
    require(call.get("kind") == kind and call.get("status") == "success"
            and call.get("http_status") == 200 and call.get("endpoint") == "/v1/chat/completions"
            and gateway.get("status") == "success" and gateway.get("generation_attempted") is True
            and gateway.get("call_id") == call_id and dispatch.get("status") == "settled"
            and dispatch.get("request_sha256") == gateway.get("request_sha256") == call.get("input_sha256")
            and sha(call.get("input_sha256")), "successful_generation_receipt_missing")
    require(gateway.get("finish_reason") in {"stop", "tool_calls", "function_call"}
            and call.get("finish_reason") == gateway["finish_reason"]
            and isinstance(gateway.get("usage"), dict)
            and type(gateway.get("prompt_tokens_checked")) is int
            and gateway["usage"].get("prompt_tokens") == gateway["prompt_tokens_checked"]
            and call.get("usage") == gateway["usage"], "generation_not_validated")
    sidecar = read_json(directory / "model_responses" / (digest({"call_id": call_id}) + ".json"))
    require(sidecar.get("workflow_identity_sha256") == digest(workflow_identity)
            and sidecar.get("call_id") == call_id and sidecar.get("request_id") == call["request_id"]
            and sidecar.get("request_sha256") == call["input_sha256"]
            and sidecar.get("response_sha256") == gateway.get("response_sha256") == digest(sidecar.get("response")),
            "model_response_binding_mismatch")


def covered(ranges, start, end):
    cursor = start
    for a, b in sorted(ranges):
        if a > cursor:
            break
        cursor = max(cursor, b)
    return cursor >= end


def workspace_gate(state, manifest, identity, report, trace, execution, ledger, directory):
    report_sha = digest(report)
    require(state.get("identity") == {"manifest_sha256": digest(manifest), "run_identity": identity},
            "workspace_identity_mismatch")
    require(state.get("draft") == report and state.get("draft_sha256") == report_sha
            and state.get("draft_valid") is True, "workspace_draft_invalid")
    version, review = state["version"], state.get("review", {})
    require(review and review.get("passed") is True and review.get("draft_sha256") == report_sha
            and review.get("facts_version") == state["facts_version"]
            and review.get("recorded_version") == version
            and review.get("workspace_version") == version - 1
            and review.get("protocol") in REVIEW_PROTOCOLS
            and review["protocol"] == identity.get("versions", {}).get("review")
            and all(review.get(k) == [] for k in ("unsupported_claims", "missing_items", "conflicts", "unresolved_items")),
            "review_missing_stale_or_unresolved")
    units = named(manifest["units"], "id")
    require(not manifest.get("images") and not manifest.get("unverified_content"), "not_verified_text_only_manifest")
    reads, excluded, required, checked_ranges = state.get("reads", {}), {}, set(), []
    require(set(reads) <= units.keys(), "read_unknown_evidence")
    for exclusion in state.get("exclusions", []):
        ident = exclusion.get("evidence_id")
        require(ident in units and isinstance(exclusion.get("reason"), str) and exclusion["reason"].strip(),
                "exclusion_invalid")
        excluded.setdefault(ident, []).append([exclusion["start"], exclusion["end"]])
    for ident, unit in units.items():
        intervals = reads.get(ident, [])
        for start, end in intervals + excluded.get(ident, []):
            require(type(start) is int and type(end) is int and 0 <= start <= end <= len(unit["text"]), "read_range_invalid")
        require(covered(intervals + excluded.get(ident, []), 0, len(unit["text"])), "source_coverage_incomplete")
        if unit["text"] and not covered(excluded.get(ident, []), 0, len(unit["text"])):
            required.add(ident)
    records = {**state["facts"], **state["relations"], **state["questions"]}
    pending, contributing = list(state.get("draft_fact_ids", [])), set()
    while pending:
        ident = pending.pop()
        if ident in contributing:
            continue
        require(ident in records, "draft_dependency_missing")
        contributing.add(ident)
        item = records[ident]
        pending.extend(item.get("depends_on", []))
        pending.extend(item[k] for k in ("subject_id", "object_id") if item.get(k))
    for item in records.values():
        require(item.get("valid") is True, "invalidated_record")
        for ref in item.get("source_refs", []):
            ident = ref.get("evidence_id")
            require(ident in units and type(ref.get("start")) is int and type(ref.get("end")) is int
                    and 0 <= ref["start"] < ref["end"] <= len(units[ident]["text"])
                    and covered(reads.get(ident, []), ref["start"], ref["end"]), "fact_source_not_read")
            if item.get("id") in contributing:
                required.add(ident)
    require(all(q.get("status") in {"resolved", "unknown"} and q.get("reason")
                for q in state["questions"].values()), "unresolved_question")
    checked = review.get("checked_evidence_ids", [])
    require(required <= set(checked) <= units.keys(), "reverse_review_incomplete")
    for ident in checked:
        require(any(a < b for a, b in reads.get(ident, [])), "review_of_unread_source")
        checked_ranges += [{"evidence_id": ident, "start": a, "end": b} for a, b in reads[ident] if a < b]
    require(review.get("checked_read_ranges") == checked_ranges, "review_read_ranges_mismatch")
    finish = {"workspace_version": version, "draft_sha256": report_sha}
    require(trace.get("accepted_finish") == finish and execution.get("workspace_version") == version
            and execution.get("draft_sha256") == report_sha and execution.get("final_data") == report,
            "finish_binding_stale")
    for receipt in (trace, execution):
        delivery = receipt.get("delivery", {})
        require(receipt.get("status") == "completed" and delivery.get("ready") is True
                and delivery.get("blockers") == [] and delivery.get("workspace_version") == version
                and delivery.get("draft_sha256") == report_sha
                and delivery.get("facts_version") == state["facts_version"], "workflow_not_deliverable")
    sessions = trace.get("sessions", [])
    require(sessions and sessions[-1].get("status") == "completed"
            and sessions[-1].get("native_final_event") is True
            and any(t.get("name") == "finish" and t.get("status") == "success" for t in sessions[-1].get("tool_calls", [])),
            "native_finish_receipt_missing")
    require(ledger["complete"], "generation_accounting_incomplete")
    call_ids = review.get("model_call_ids", [])
    require(isinstance(call_ids, list) and bool(call_ids), "review_calls_missing")
    for call_id in call_ids:
        successful_call(call_id, "generation", ledger, directory, identity)
    controller = [key for key, value in ledger["calls"].items()
                  if value.get("kind") == "agent_generation" and value.get("status") == "success"]
    require(bool(controller), "native_controller_generation_missing")
    successful_call(controller[-1], "agent_generation", ledger, directory, identity)


def document_gate(row, directory, output, run, summary_row, ledger):
    identity, fingerprint = run["identity"], run["run_fingerprint"]
    manifest, report = read_json(directory / "source_manifest.json"), read_json(directory / "report.json")
    source_binding = read_json(directory / "source_binding.json")
    execution, trace = read_json(directory / "execution.json"), read_json(directory / "document_workflow.json")
    publication, terminal = read_json(directory / "publication.json"), read_json(directory / "document_summary.json")
    require(manifest == row["manifest"] and source_binding == {
        "run_fingerprint": fingerprint, "original_source_sha256": row["source_sha256"],
        "manifest_sha256": digest(manifest), "document": None}, "source_binding_mismatch")
    require(terminal == summary_row and terminal.get("name") == row["evidence_id"]
            and terminal.get("status") == "completed" and terminal.get("execution_sha256") == file_digest(directory / "execution.json"),
            "terminal_execution_mismatch")
    require(publication == {"run_fingerprint": fingerprint, "status": "committed",
                            "execution_sha256": file_digest(directory / "execution.json")}, "publication_not_committed")
    require(execution.get("cli_status") == "completed" and execution.get("run_fingerprint") == fingerprint
            and execution.get("source_sha256") == row["source_sha256"]
            and execution.get("source_binding_sha256") == file_digest(directory / "source_binding.json")
            and execution.get("report_sha256") == digest(report)
            and execution.get("output") == {"name": output.name, "sha256": file_digest(output)}
            and terminal.get("output") == output.name, "execution_hash_binding_mismatch")
    workflow_id = execution["identity"]
    require(trace.get("identity") == workflow_id and workflow_id.get("source") == digest(manifest)
            and workflow_id.get("run_identity", {}).get("run_fingerprint") == fingerprint
            and workflow_id["run_identity"].get("source_sha256") == row["source_sha256"]
            and workflow_id.get("settings") == identity.get("settings")
            and workflow_id.get("model") == identity.get("served_model"), "workflow_identity_mismatch")
    require(workflow_id.get("versions", {}).get("review") == identity.get("versions", {}).get("review_version"),
            "review_protocol_identity_mismatch")
    for key in ("model_calls", "model_dispatches", "gateway_receipts"):
        require(trace.get(key) == execution.get(key), "workflow_receipts_mismatch")
    require(trace.get("sessions") == execution.get("sdk_sessions"), "sdk_sessions_mismatch")
    for seconds, limit in ((execution.get("seconds"), identity.get("settings", {}).get("report_timeout")),
                           (terminal.get("seconds"), identity.get("document_total_report_timeout"))):
        require(number(seconds), "report_timing_missing")
        require(limit is None or (number(limit, True) and seconds <= limit), "report_finished_after_deadline")
    workspace_gate(read_json(directory / "workspace/workspace.json"), manifest, workflow_id, report, trace, execution, ledger, directory)
    return report, execution.get("delivery_validation")


def wall_time(supervisor, summary, work, fingerprint):
    try:
        require(supervisor.get("status") in {"completed", "failed"} and type(supervisor.get("returncode")) is int
                and supervisor.get("child_may_still_be_running") is not True, "supervisor_not_terminal")
        require(summary and summary.get("run_fingerprint") == fingerprint, "summary_identity_missing")
        # A terminal status must bind THIS summary, rather than merely coexist.
        require((supervisor.get("run_fingerprint") == fingerprint
                 and supervisor.get("summary_sha256") == file_digest(work / "summary.json"))
                or supervisor.get("summary") == summary, "supervisor_summary_binding_missing")
        started = datetime.fromisoformat(supervisor["started_at"])
        finished = datetime.fromisoformat(supervisor["finished_at"])
        batch_start = datetime.fromisoformat(summary["started_at"])
        require(started.tzinfo is not None and finished.tzinfo is not None and batch_start.tzinfo is not None,
                "wall_clock_timezone_missing")
        seconds = supervisor.get("wall_seconds")
        require(number(seconds, True) and started <= batch_start <= finished
                and abs((finished - started).total_seconds() - seconds) <= max(2, seconds * .01), "wall_clock_inconsistent")
        require(summary.get("cumulative_timing_complete") is True and summary.get("attempts_with_unknown_wall_time") == []
                and summary.get("reused_reports") == 0 and summary.get("cumulative_seconds") == summary.get("seconds")
                and number(summary.get("seconds")) and summary["seconds"] <= seconds + 1, "incomplete_cumulative_timing")
        attempt_id = summary.get("attempt_id")
        relative_file(work / "attempts", attempt_id + ".summary.json")
        attempts = list((work / "attempts").glob("*.summary.json"))
        preflights = list((work / "attempts").glob("*.preflight.json"))
        require(len(attempts) == len(preflights) == 1 and attempts[0].name == attempt_id + ".summary.json"
                and preflights[0].name == attempt_id + ".preflight.json"
                and read_json(attempts[0]) == summary
                and read_json(preflights[0]).get("run_fingerprint") == fingerprint, "attempt_wall_history_incomplete")
        return {"status": "known", "seconds": seconds,
                "scope": "supervisor wall time including preflight, workers, model/tool retries and publication"}
    except Exception as exc:
        return {"status": "unknown", "seconds": None, "reason": error_code(exc)}


def labels(path, rows, forbidden_paths):
    if path is None:
        return None, {"status": "not_scored", "reason": "no_scoring_manifest"}
    try:
        manifest = read_json(path)
        require(manifest.get("schema_version") == 1 and manifest.get("never_include_in_model_inputs") is True,
                "labels_not_explicitly_separated")
        entries = named(manifest.get("entries"), "evidence_id")
        require(set(entries) == {r["evidence_id"] for r in rows}, "scoring_selection_mismatch")
        result = {}
        for name, entry in entries.items():
            require(set(entry) == {"evidence_id", "path", "sha256"} and sha(entry["sha256"]), "invalid_scoring_entry")
            target = relative_file(Path(path).parent, entry["path"])
            require(target.absolute() not in forbidden_paths, "label_is_model_input_or_artifact")
            require(file_digest(target) == entry["sha256"], "label_hash_mismatch")
            gold = read_json(target)
            score_report(gold, None)
            result[name] = gold
        return result, {"status": "scored", "manifest_sha256": file_digest(path)}
    except Exception as exc:
        return None, {"status": "invalid", "reason": error_code(exc)}


def semantic_reviews(path):
    if path is None:
        return {}, "not_provided"
    try:
        manifest = read_json(path)
        require(manifest.get("schema_version") == 1 and manifest.get("independent") is True, "semantic_review_not_independent")
        return named(manifest.get("entries"), "evidence_id"), None
    except Exception as exc:
        return {}, error_code(exc)


def semantic_gate(review, row, report):
    if review is None:
        return {"status": "not_reviewed"}
    try:
        require(report is not None, "prediction_not_valid")
        require(review.get("input_json_sha256") == row["source_sha256"]
                and "original_report_sha256" in review
                and review["original_report_sha256"] == row.get("original_report_sha256")
                and review.get("report_sha256") == digest(report),
                "semantic_source_or_report_mismatch")
        require(isinstance(review.get("reviewer"), str) and review["reviewer"].strip()
                and review.get("coverage") == "complete_supplied_text"
                and type(review.get("accepted")) is bool
                and isinstance(review.get("findings"), list) and isinstance(review.get("omissions"), list),
                "semantic_review_incomplete")
        counts = {k: 0 for k in "ABCD"}
        for finding in review["findings"]:
            require(isinstance(finding, dict) and finding.get("category") in counts, "invalid_semantic_category")
            counts[finding["category"]] += 1
        clear = not counts["C"] and not counts["D"] and not review["omissions"] and all(
            f["category"] != "B" or (f.get("marked_as_inference") is True
                                     and f.get("allowed_by_output_contract") is True) for f in review["findings"])
        require(not review["accepted"] or clear, "semantic_acceptance_conflicts_with_findings")
        return {"status": "accepted" if review["accepted"] else "rejected", "categories": counts,
                "omissions": len(review["omissions"]), "scope": "supplied_text_only",
                "independence_and_completeness": "external_declarations_not_independently_authenticated"}
    except Exception as exc:
        return {"status": "invalid", "reason": error_code(exc)}


def summarize(*, batch_manifest, work_dir, result_dir, code_root, supervisor_status,
              scoring_manifest=None, semantic_manifest=None):
    batch_manifest, work, result = Path(batch_manifest), Path(work_dir), Path(result_dir)
    run, run_error = optional_json(work / "run_manifest.json")
    if run is not None and not isinstance(run, dict):
        run, run_error = None, "run_manifest_not_object"
    run = run or {}
    input_identity = run.get("identity", {})
    rows, batch_sha = selection(batch_manifest, input_identity if isinstance(input_identity, dict) else {})
    global_errors = ["run_manifest:" + run_error] if run_error else []
    try:
        run_gate(run, rows, batch_sha, code_root)
    except Exception as exc:
        global_errors.append(error_code(exc))
    summary, summary_error = optional_json(work / "summary.json")
    supervisor, supervisor_error = optional_json(supervisor_status)
    summaries = {}
    try:
        require(summary and summary.get("workflow") == "document-agent"
                and summary.get("run_fingerprint") == run.get("run_fingerprint")
                and summary.get("training_evidence_manifest_sha256") == batch_sha, "batch_summary_binding_invalid")
        summaries = named(summary.get("documents"), "name")
        require(set(summaries) == {r["evidence_id"] for r in rows} and summary.get("reports") == len(rows),
                "summary_drops_selected_reports")
        for field, status in (("completed_reports", "completed"), ("failed_reports", "failed"), ("unfinished_reports", "unfinished")):
            require(summary.get(field) == sum(v.get("status") == status for v in summaries.values()), "summary_counts_mismatch")
        require(supervisor and supervisor.get("status") in {"completed", "failed", "preflight_failed"}
                and supervisor.get("child_may_still_be_running") is not True, "supervisor_not_terminal")
        if supervisor["status"] == "completed":
            require(supervisor.get("returncode") == 0 and summary["completed_reports"] == len(rows), "supervisor_false_completion")
    except Exception as exc:
        global_errors.append(error_code(exc))
    external, external_error = semantic_reviews(semantic_manifest)
    documents, predictions = [], {}
    for row in rows:
        name = row["evidence_id"]
        errors = global_errors + row["errors"]
        record = {"evidence_id": name, "selected_path": row["path"], "input_json_sha256": row["source_sha256"],
                  "original_report_sha256": row.get("original_report_sha256"),
                  "cli_reported_status": summaries.get(name, {}).get("status", "missing"),
                  "internal_cli_completion": False, "content_fidelity": "not_verified", "errors": list(errors)}
        receipts = []
        directory = work / "documents" / name if name is not None else None
        if directory is not None:
            for filename in ("execution.json", "document_workflow.json"):
                value, _ = optional_json(directory / filename)
                if isinstance(value, dict):
                    receipts.append(value)
            for failure in sorted((directory / "attempts").glob("*.failure.json")):
                value, err = optional_json(failure)
                if err:
                    record["errors"].append("failure_receipt_unreadable")
                elif not isinstance(value, dict):
                    record["errors"].append("failure_receipt_not_object")
                elif value.get("run_fingerprint") == run.get("run_fingerprint"):
                    receipt = value.get("workflow_receipt", {"model_calls": value.get("model_calls", [])})
                    if isinstance(receipt, dict):
                        receipts.append(receipt)
                    else:
                        record["errors"].append("failure_workflow_receipt_invalid")
        ledger = generation_ledger(receipts)
        record["generations"] = {k: v for k, v in ledger.items() if k not in {"calls", "dispatches", "gateways"}}
        record["generations"]["http_attempt_receipts"] = [{k: c.get(k) for k in (
            "call_id", "request_id", "kind", "status", "http_status")} for c in ledger["calls"].values()]
        prediction = None
        if not record["errors"]:
            try:
                require(summaries.get(name, {}).get("status") == "completed", "report_not_completed")
                candidate, expected_readback = document_gate(row, directory, result / (name + ".docx"), run, summaries[name], ledger)
                record.update(internal_cli_completion=True, content_fidelity="invalid")
                readback = validate_delivery_docx(candidate, result / (name + ".docx"))
                require(expected_readback == readback, "delivery_readback_receipt_mismatch")
                prediction = candidate
                record.update(content_fidelity="verified", report_sha256=digest(prediction),
                              output_sha256=readback["docx_sha256"])
            except Exception as exc:
                record["errors"].append(error_code(exc))
        record["external_semantic_acceptance"] = semantic_gate(external.get(name), row, prediction)
        predictions[name] = prediction
        documents.append(record)
    forbidden = {r["source_path"].absolute() for r in rows} | {batch_manifest.absolute(), Path(supervisor_status).absolute()}
    # Explicit labels may not point into run/code trees or at inference input files.
    if scoring_manifest is not None:
        scoring_raw, _ = optional_json(scoring_manifest)
        raw_entries = scoring_raw.get("entries", []) if isinstance(scoring_raw, dict) else []
        for entry in raw_entries if isinstance(raw_entries, list) else []:
            try:
                target = relative_file(Path(scoring_manifest).parent, entry["path"]).absolute()
                if any(target == p.absolute() or p.absolute() in target.parents for p in (work, result, Path(code_root))):
                    forbidden.add(target)
            except (KeyError, TypeError, Invalid):
                pass
    gold, scoring = labels(scoring_manifest, rows, forbidden)
    scores = []
    if gold is not None:
        for row, document in zip(rows, documents):
            score = score_report(gold[row["evidence_id"]], predictions[row["evidence_id"]])
            document["score"] = score
            scores.append(score)
    timing = wall_time(supervisor or {}, summary, work, run.get("run_fingerprint"))
    if global_errors or any(r["errors"] for r in rows):
        timing = {"status": "unknown", "seconds": None, "reason": "run_or_source_identity_not_verified"}
    complete = sum(d["internal_cli_completion"] for d in documents)
    semantic_complete = external_error is None and all(d["external_semantic_acceptance"]["status"] in {
        "accepted", "rejected"} or not d["internal_cli_completion"] for d in documents)
    accepted = sum(d["external_semantic_acceptance"]["status"] == "accepted" for d in documents)
    wall = timing["seconds"]
    return {"schema_version": 1, "tool": "document-agent-offline-single-arm-v1", "scope": SCOPE,
            "batch_manifest_sha256": batch_sha, "run_fingerprint": run.get("run_fingerprint"),
            "selected_reports": len(rows), "internal_completed_reports": complete,
            "failed_missing_or_invalid_reports": sum(predictions[r["evidence_id"]] is None for r in rows),
            "content_fidelity_verified_reports": sum(d["content_fidelity"] == "verified" for d in documents),
            "external_semantic_review": {"status": "complete_for_valid_outputs" if semantic_complete else "not_fully_reviewed",
                "manifest_error": external_error, "manifest_sha256": file_digest(semantic_manifest)
                    if semantic_manifest is not None and external_error is None else None,
                "accepted_reports": accepted if semantic_complete else None},
            "timing": timing, "throughput": {"internally_completed_reports_per_hour": complete * 3600 / wall if wall else None,
                "accepted_reports_per_hour": accepted * 3600 / wall if wall and semantic_complete else None},
            "generations": {"known_http_attempts": sum(d["generations"]["known_http_attempts"] for d in documents),
                "outcome_unknown_dispatches": sum(d["generations"]["outcome_unknown_dispatches"] for d in documents),
                "complete": all(d["generations"]["complete"] for d in documents),
                "note": "Dispatch intents reserve budget; unresolved intents are not proof of HTTP receipt. SDK sessions are not generation calls."},
            "scoring": scoring, "metrics": aggregate_reports(scores) if gold is not None else None,
            "run_errors": global_errors, "documents": documents, "limitations": LIMITATIONS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("batch-manifest", "work-dir", "result-dir", "code-root", "supervisor-status"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("scoring-manifest", "semantic-manifest", "out-json"):
        parser.add_argument("--" + field, type=Path)
    args = vars(parser.parse_args(argv))
    output = args.pop("out_json")
    try:
        result = summarize(**args)
        status = 0
    except Exception as exc:
        result, status = {"schema_version": 1, "status": "invalid_selection", "error": error_code(exc)}, 2
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is None:
        print(rendered, end="")
    else:
        # Never overwrite evidence, runtime files, or a previous evaluation.
        with output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
