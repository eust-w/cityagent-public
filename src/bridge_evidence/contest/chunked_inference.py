"""Complete extraction with split-on-length retries, exact coverage and bounded repair."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from concurrent.futures import FIRST_COMPLETED, wait
from copy import deepcopy
from dataclasses import asdict
from uuid import uuid4

from .inference_core import (
    ContextBudgetError,
    InferenceFailure,
    _parse_report,
    build_prompt,
    prompt_ids_sha256,
)
from .long_document import (
    EXACT_FRAGMENT_INSTRUCTION,
    EXACT_FRAGMENT_V2_INSTRUCTION,
    FRAGMENT_INSTRUCTION,
    LEAN_FRAGMENT_INSTRUCTION,
    ChunkPlan,
    ChunkTask,
    ExtractionFragment,
    SourceSlice,
    _identity_in_text,
    merge_fragments,
    parse_fragment,
    resolve_basic_conflicts,
    plan_chunks,
    source_metadata,
    split_task,
)
from .run_state import atomic_json, json_digest


class ChunkFailure(InferenceFailure):
    def __str__(self):
        return super().__str__() + "; " + getattr(self, "reason", "")


def _receipt(ids, role, task=None):
    return {
        "call_id": uuid4().hex,
        "reserved_at_wall": time.time(),
        "role": role,
        "task_id": task.task_id if task else None,
        "source_ranges": task.owned_keys if task else [],
        "prompt_sha256": prompt_ids_sha256(ids),
        "prompt_tokens": len(ids),
        "output_sha256": None,
        "output_tokens": None,
        "finish_reason": None,
        "seconds": 0.0,
        "latency_seconds": 0.0,
        "status": "failed",
        "errors": [],
    }


def _raw_receipt_path(directory, call_id):
    if directory is None:
        return None
    return (
        directory
        / "raw-responses"
        / (hashlib.sha256(str(call_id).encode()).hexdigest() + ".json")
    )


def _raw_limit(config):
    return min(8 * 1024 * 1024, max(65536, config["output_budget"] * 32))


def _complete(call, response, seconds, maximum):
    call["seconds"] = call["latency_seconds"] = seconds
    if not isinstance(response, dict) or not isinstance(response.get("text"), str):
        raise ValueError("invalid backend response")
    measured = response.get("latency_seconds", seconds)
    if (
        type(measured) not in (int, float)
        or not math.isfinite(measured)
        or measured < 0
    ):
        raise ValueError("invalid backend latency")
    call["seconds"] = call["latency_seconds"] = measured
    call["call_id"] = response.get("call_id", call["call_id"])
    call["finish_reason"] = response.get("finish_reason")
    call["output_tokens"] = response.get("output_tokens")
    call["output_sha256"] = hashlib.sha256(response["text"].encode()).hexdigest()
    if (
        type(call["output_tokens"]) is not int
        or not 0 < call["output_tokens"] <= maximum
    ):
        raise ValueError("invalid output token count")
    if call["finish_reason"] != "stop":
        raise ValueError("incomplete_generation:" + str(call["finish_reason"]))
    return response["text"]


def read_evidence(snapshot, arguments):
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {"source_ids"}
        or not isinstance(arguments["source_ids"], list)
        or not 1 <= len(arguments["source_ids"]) <= 8
    ):
        raise ValueError("illegal evidence tool arguments")
    sources = snapshot.sources()
    if any(
        not isinstance(ref, str) or ref not in sources
        for ref in arguments["source_ids"]
    ):
        raise ValueError("source outside current report")
    return [
        {"source_id": ref, "text": sources[ref].text, "location": sources[ref].location}
        for ref in arguments["source_ids"]
    ]


def _repair_support(text, field, value, scope):
    """Bind field, value and inherited identity within one explicit source claim.

    A nearby number or a matching scope elsewhere in a paragraph is insufficient.
    Missing/ambiguous scope fails closed instead of silently inheriting old labels.
    """
    quote = str(value).strip()
    if not quote:
        raise ValueError("repair value lacks exact source support")
    numeric = re.fullmatch(r"[+-]?\d+(?:\.\d+)?", quote) is not None
    ending = r"(?!\d|\.\d)" if numeric else ""
    pattern = re.compile(
        re.escape(field)
        + r"\s*(?:[:：=为是]\s*)?(?P<value>"
        + re.escape(quote)
        + ")"
        + ending
    )
    candidates = []
    for segment in re.finditer(r"[^\n;；。!?！？，,]+", text):
        claim = segment.group()
        if claim.count(field) != 1 or any(
            not _identity_in_text(item, claim) for item in scope.values()
        ):
            continue
        if "year" in scope:
            years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", claim))
            if years != {str(scope["year"])}:
                continue
        if "span" in scope:
            sides = {
                "左幅" if side in ("左幅", "左线") else "右幅"
                for side in re.findall(r"左幅|右幅|左线|右线", claim)
            }
            expected = (
                str(scope["span"]).replace("左线", "左幅").replace("右线", "右幅")
            )
            if expected in ("左幅", "右幅") and sides != {expected}:
                continue
        match = pattern.search(claim)
        if match is None:
            continue
        # Current and historical score labels cannot match by suffix alone.
        prefix = claim[: match.start()]
        if not field.startswith(("上一次", "上次", "历史", "前次")) and re.search(
            r"(?:上一次|上次|历史|前次)\s*$", prefix
        ):
            continue
        candidates.append((segment.start() + match.start("value"), claim))
    if len(candidates) != 1:
        raise ValueError("repair field/value/scope lacks unambiguous source support")
    return candidates[0]


def bounded_repair(
    snapshot,
    result,
    tokenizer,
    generate,
    config,
    calls,
    *,
    checkpoint=None,
    persist=None,
    before_request=None,
):
    """At most two additional requests: choose readonly evidence, then resolve conflicts."""
    if not result["conflicts"]:
        return result
    scoped_conflicts = {}
    for conflict in result["conflicts"]:
        observations = conflict.get("observations", [])
        scopes = {json_digest(item.get("scope", {})) for item in observations}
        if len(scopes) > 1 or conflict["field"] in scoped_conflicts:
            raise ValueError("repair cannot collapse different observation scopes")
        scoped_conflicts[conflict["field"]] = observations
    instruction = '仅输出 JSON 工具调用 {"tool":"read_evidence","arguments":{"source_ids":[最多8个当前报告来源ID]}}。只能使用当前报告来源。'
    text = json.dumps(
        {
            "conflicts": result["conflicts"],
            "available_sources": [
                r.source_id for r in snapshot.regions
                if r.kind != "visual-observation"
            ],
        },
        ensure_ascii=False,
    )
    tool_events = []
    checkpoint = checkpoint if checkpoint is not None else {}
    persist = persist or (lambda: None)
    identity = json_digest(
        {
            "snapshot": snapshot.digest,
            "conflicts": result["conflicts"],
            "config": config,
        }
    )
    if checkpoint.get("identity", identity) != identity:
        raise ValueError("repair checkpoint identity changed")
    checkpoint["identity"] = identity
    responses = checkpoint.setdefault("responses", {})
    requests = checkpoint.setdefault("requests", {})

    def guard():
        from .rsi_runtime import REPORT_DEADLINE

        deadline = REPORT_DEADLINE.get()
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("repair report deadline exhausted")

    def request(text, instruction, role):
        if role in responses:
            stored = responses[role]
            call = calls[requests[role]]
            if (
                hashlib.sha256(stored["raw"].encode()).hexdigest()
                != call["output_sha256"]
            ):
                raise ValueError("repair checkpoint response changed")
            return stored["raw"], call
        if role in requests:
            raise RuntimeError(
                "interrupted repair cannot exceed its two-call reservation"
            )
        if len(requests) >= 2:
            raise RuntimeError("repair request budget exhausted")
        guard()
        ids = build_prompt(
            tokenizer,
            text,
            max_model_len=config["input_budget"] + config["output_budget"],
            max_new_tokens=config["output_budget"],
            instruction=instruction,
        )
        guard()
        if before_request:
            before_request(role)
        call = _receipt(ids, role)
        if role == "react-repair":
            call["source_ranges"] = [
                f"{entry['source_id']}:0:{len(entry['text'])}" for entry in evidence
            ]
        requests[role] = len(calls)
        calls.append(call)
        persist()  # Durable request reservation precedes the backend side effect.
        start = time.perf_counter()
        from .rsi_runtime import REQUEST_CALL_ID

        call_token = REQUEST_CALL_ID.set(call["call_id"])
        response_returned = False
        try:
            response = (
                generate.call_with_role(ids, config["output_budget"], role=role)
                if hasattr(generate, "call_with_role")
                else generate(ids, config["output_budget"])
            )
            response_returned = True
            raw = _complete(
                call, response, time.perf_counter() - start, config["output_budget"]
            )
            responses[role] = {"raw": raw}
            persist()  # Recovery reuses even a response interrupted before validation.
            return raw, call
        except Exception as error:
            measured = getattr(error, "latency_seconds", None)
            if (
                type(measured) not in (int, float)
                or not math.isfinite(measured)
                or measured < 0
            ):
                measured = (
                    call["seconds"]
                    if response_returned
                    else time.perf_counter() - start
                )
            call["seconds"] = call["latency_seconds"] = measured
            call["errors"].append(str(error))
            persist()
            raise
        finally:
            REQUEST_CALL_ID.reset(call_token)

    raw, call = request(text, instruction, "react-tool-selection")
    selection = json.loads(raw)
    if set(selection) != {"tool", "arguments"} or selection["tool"] != "read_evidence":
        call["errors"].append("illegal tool")
        raise ValueError("illegal tool")
    try:
        evidence = read_evidence(snapshot, selection["arguments"])
        if any(snapshot.sources()[entry["source_id"]].kind == "visual-observation"
               for entry in evidence):
            raise ValueError("visual observation cannot support basic conflict repair")
    except ValueError as error:
        call["errors"].append(str(error))
        raise
    call["status"] = "success"
    persist()
    tool_events.append(
        {
            "tool": "read_evidence",
            "arguments": selection["arguments"],
            "status": "success",
            "call_id": call["call_id"],
        }
    )
    raw, call = request(
        json.dumps(
            {"conflicts": result["conflicts"], "evidence": evidence}, ensure_ascii=False
        ),
        '仅根据证据解决冲突。输出 JSON {"resolved":{字段:{"value":值,"source_id":来源ID}}}。证据不足不输出该字段。',
        "react-repair",
    )
    resolved = json.loads(raw).get("resolved")
    fields = {c["field"] for c in result["conflicts"]}
    if not isinstance(resolved, dict) or set(resolved) != fields:
        call["errors"].append("unresolved conflicts")
        raise ValueError("unresolved conflicts")
    refs = {e["source_id"] for e in evidence}
    for key, value in resolved.items():
        if (
            not isinstance(value, dict)
            or value.get("source_id") not in refs
            or isinstance(value.get("value"), (dict, list))
            or value.get("value") is None
        ):
            call["errors"].append("invalid repair source")
            raise ValueError("invalid repair source")
    repairs = []
    sources = snapshot.sources()
    for key, value in resolved.items():
        region = sources[value["source_id"]]
        if region.kind == "visual-observation":
            raise ValueError("visual observation cannot support basic conflict repair")
        quote = str(value["value"]).strip()
        observations = scoped_conflicts[key]
        scope = observations[0].get("scope", {}) if observations else {}
        try:
            start, support_quote = _repair_support(
                region.text, key, value["value"], scope
            )
        except ValueError as error:
            call["errors"].append(str(error))
            persist()
            raise
        repairs.append(
            {
                "field": key,
                "value": value["value"],
                "scope": scope,
                "support_quote": support_quote,
                "support_rule": "same-claim-field-value-scope-v1",
                "evidence": {
                    "source_id": region.source_id,
                    "source_key": f"{region.source_id}:{start}:{start + len(quote)}",
                    "start": start,
                    "end": start + len(quote),
                    "text": quote,
                    "source_text_sha256": hashlib.sha256(
                        region.text.encode()
                    ).hexdigest(),
                    "document_sha256": snapshot.document_sha256,
                    "snapshot_sha256": snapshot.digest,
                    "location": region.location,
                },
                "call_id": call["call_id"],
                "supersedes": [json_digest(item) for item in observations],
            }
        )
    for key, value in resolved.items():
        result["data"]["基本信息"][key] = value["value"]
    result["repair_evidence"] = [*result.get("repair_evidence", []), *repairs]
    call["status"] = "success"
    persist()
    result["contributing_call_ids"].append(call["call_id"])
    result["tool_events"] = tool_events
    result["conflicts"] = []
    return result


def _instruction(config):
    if config["prompt"] == "source-fragment-lean-v1":
        return LEAN_FRAGMENT_INSTRUCTION
    if config["prompt"] == "source-fragment-exact-v2":
        return EXACT_FRAGMENT_V2_INSTRUCTION
    if config["prompt"] == "source-fragment-exact-v1":
        return EXACT_FRAGMENT_INSTRUCTION
    instruction = FRAGMENT_INSTRUCTION
    if config["prompt"] == "source-fragment-concise-v1":
        instruction += "\n语义字段简洁；可引用的描述放 quotes，避免重复解释。"
    return instruction


def prepare_chunk_plan(snapshot, tokenizer, config):
    """CPU preparation shared by the prefetcher and the trusted executor."""
    return plan_chunks(
        snapshot,
        tokenizer,
        input_budget=config["input_budget"],
        output_budget=config["output_budget"],
        overlap=config["overlap"],
        instruction=_instruction(config),
    )


def _restore_task(value):
    return ChunkTask(
        value["task_id"],
        [SourceSlice(**v) for v in value["owned"]],
        [SourceSlice(**v) for v in value["context"]],
        value["prompt_tokens"],
    )


def _validate_plan(plan, snapshot, config, instruction):
    if (
        not isinstance(plan, ChunkPlan)
        or plan.snapshot_sha256 != snapshot.digest
        or plan.input_budget != config["input_budget"]
        or plan.output_budget != config["output_budget"]
        or plan.instruction != instruction
        or not plan.coverage(snapshot)["complete"]
    ):
        raise ValueError("prepared/checkpoint plan identity or coverage changed")
    sources = snapshot.sources()
    seen = set()
    for task in plan.tasks:
        if task.task_id in seen or task.task_id != json_digest(task.owned_keys)[:24]:
            raise ValueError("prepared/checkpoint plan task identity changed")
        seen.add(task.task_id)
        for part in [*task.owned, *task.context]:
            region = sources.get(part.source_id)
            if (
                region is None
                or not 0 <= part.start <= part.end <= len(region.text)
                or part.text != region.text[part.start : part.end]
                or part.metadata != source_metadata(region)
            ):
                raise ValueError("prepared/checkpoint plan source content changed")


def _validate_prefix_budget(value):
    """Validate the trusted visual executor's predeclared shared report budget.

    Usage is cumulative for the visual phase only. Text checkpoint time and
    reservations remain separate so recovery neither resets nor double counts
    earlier visual work. No generated policy action can supply these fields.
    """
    if value is None:
        return None
    fields = {
        "identity", "initial_text_tasks", "initial_visual_jobs",
        "total_call_limit", "used_calls", "elapsed_seconds",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid visual prefix budget fields")
    result = dict(value)
    if (not isinstance(result["identity"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", result["identity"])):
        raise ValueError("visual prefix plan SHA256 required")
    if any(type(result[key]) is not int for key in (
        "initial_text_tasks", "initial_visual_jobs", "total_call_limit", "used_calls",
    )):
        raise ValueError("visual prefix task and call counts must be integers")
    if (result["initial_text_tasks"] < 0 or result["initial_visual_jobs"] < 1
            or not 0 <= result["used_calls"] <= result["initial_visual_jobs"]
            or result["total_call_limit"] != 4 * (
                result["initial_text_tasks"] + result["initial_visual_jobs"]
            ) + 2):
        raise ValueError("visual prefix call budget differs from frozen initial plan")
    elapsed = result["elapsed_seconds"]
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
            or not 0 <= elapsed <= 1200):
        raise ValueError("invalid visual prefix elapsed budget")
    return result


def _prefix_budget_identity(value):
    return {key: field for key, field in value.items()
            if key not in ("used_calls", "elapsed_seconds")}


def infer_chunked(
    snapshot,
    tokenizer,
    generate,
    *,
    config,
    max_attempts=2,
    cache_dir=None,
    policy=None,
    prepared_plan=None,
    prefix_budget=None,
):
    start = time.perf_counter()
    started_at_wall = time.time()
    prefix_budget = _validate_prefix_budget(prefix_budget)
    prefix_calls = prefix_budget["used_calls"] if prefix_budget else 0
    prefix_spent = prefix_budget["elapsed_seconds"] if prefix_budget else 0.0
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("positive integer attempt budget required")
    max_attempts = min(max_attempts, 2)
    trusted_budget = {
        "version": "chunked-trusted-budget-v1",
        "report_seconds": 1200,
        "calls_per_initial_task": 4,
        "extra_calls": 2,
        "max_attempts": max_attempts,
        "max_split_depth": 8,
    }
    instruction = _instruction(config)
    # Plan identity is persisted, so resume locates its graph without tokenizing again.
    cache_key = json_digest(
        {
            "schema": 3,
            "trusted_budget": trusted_budget,
            "snapshot": snapshot.digest,
            "config": config,
            "instruction": instruction,
            "strategy": policy.bundle.digest if policy else None,
            **({"prefix_budget": _prefix_budget_identity(prefix_budget)}
               if prefix_budget else {}),
        }
    )
    cache_root = cache_dir
    journal = saved = None
    previous_prefix = None
    if cache_root:
        cache_dir = cache_root / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True)
        from .rsi_runtime import TaskJournal

        journal = TaskJournal(cache_dir / "checkpoint.sqlite", cache_key)
        saved = journal.load()
        if saved and saved.get("trusted_budget") != trusted_budget:
            journal.close()
            raise ValueError("checkpoint trusted budget version or limits changed")
        if saved and prefix_budget:
            previous_prefix = saved.get("prefix_budget")
            try:
                previous_prefix = _validate_prefix_budget(previous_prefix)
                if (previous_prefix is None
                        or _prefix_budget_identity(previous_prefix)
                        != _prefix_budget_identity(prefix_budget)
                        or prefix_calls < previous_prefix["used_calls"]
                        or prefix_spent < previous_prefix["elapsed_seconds"]):
                    raise ValueError("visual prefix budget reset or identity changed")
            except BaseException:
                journal.close()
                raise
    if saved:
        plan = ChunkPlan(
            snapshot.digest,
            [_restore_task(t) for t in saved["tasks"]],
            config["input_budget"],
            config["output_budget"],
            instruction,
        )
    else:
        plan = (
            deepcopy(prepared_plan)
            if prepared_plan is not None
            else prepare_chunk_plan(snapshot, tokenizer, config)
        )
        # Older checkpoints predate equal trusted budgets for RSI off/frozen.
        # Preserve them on disk, but never migrate/reuse their responses here.
    try:
        _validate_plan(plan, snapshot, config, instruction)
        if prepared_plan is not None:
            _validate_plan(prepared_plan, snapshot, config, instruction)
            if saved and saved.get("initial_plan_sha256") not in (
                None,
                prepared_plan.digest,
            ):
                raise ValueError("prepared plan differs from checkpoint initial plan")
    except BaseException:
        if journal:
            journal.close()
        raise
    initial_plan_sha256 = (
        saved.get("initial_plan_sha256", plan.digest) if saved else plan.digest
    )
    pending = list(plan.tasks)
    depths, issued, spent = {}, 0, 0.0
    effective_batch, initial_tasks = config["batch_size"], len(plan.tasks)
    decision_start = len(policy.decisions) if policy else 0
    previous_decisions = []
    fragments, calls, attempts = [], [], {}
    generation_responses = {}
    repair = {}
    if saved:
        by_id = {t.task_id: t for t in plan.tasks}
        try:
            for fragment in saved["fragments"]:
                cached = json.loads(
                    (cache_dir / (fragment["task_id"] + ".json")).read_text()
                )
                task = by_id[fragment["task_id"]]
                if (
                    hashlib.sha256(cached["raw"].encode()).hexdigest()
                    != cached["call"]["output_sha256"]
                    or asdict(
                        parse_fragment(
                            cached["raw"], task, snapshot, cached["call"]["call_id"]
                        )
                    )
                    != fragment
                ):
                    raise ValueError("checkpoint fragment changed")
                generation_responses[cached["call"]["call_id"]] = cached["raw"]
            pending = [by_id[key] for key in saved["pending"]]
            fragments = [ExtractionFragment(**f) for f in saved["fragments"]]
            calls, attempts = saved["calls"], saved["attempts"]
            depths, issued, spent = saved["depths"], saved["issued"], saved["spent"]
            initial_tasks = saved["initial_tasks"]
            effective_batch = saved.get("effective_batch", config["batch_size"])
            previous_decisions = saved.get("decisions", [])
            repair = saved.get("repair", {})
            complete_ids = [f.task_id for f in fragments]
            pending_ids = [t.task_id for t in pending]
            if (
                len(set(complete_ids)) != len(complete_ids)
                or len(set(pending_ids)) != len(pending_ids)
                or set(complete_ids) & set(pending_ids)
                or set(complete_ids) | set(pending_ids) != set(by_id)
            ):
                raise ValueError("checkpoint task ownership changed")
            if (
                type(initial_tasks) is not int
                or initial_tasks < 1
                or type(issued) is not int
                or issued < len(calls)
                or issued > (
                    prefix_budget["total_call_limit"] - prefix_calls
                    if prefix_budget else
                    trusted_budget["calls_per_initial_task"] * initial_tasks
                    + trusted_budget["extra_calls"]
                )
                or not math.isfinite(spent)
                or spent < 0
                or any(
                    type(depth) is not int
                    or not 0 <= depth <= trusted_budget["max_split_depth"]
                    for depth in depths.values()
                )
                or any(
                    type(attempt.get("count")) is not int
                    or not 0 <= attempt["count"] <= max_attempts
                    for attempt in attempts.values()
                )
            ):
                raise ValueError("checkpoint budgets changed")
        except Exception as error:
            journal.close()
            failure = ChunkFailure(
                saved.get("calls", []),
                raw_output="",
                wall_seconds=time.perf_counter() - start,
            )
            failure.reason = "cached checkpoint invalid: " + str(error)
            raise failure from error
    from .rsi_runtime import (
        REPORT_DEADLINE,
        REQUEST_CALL_ID,
        REQUEST_CALL_IDS,
        REQUEST_PRIORITY,
    )

    priority_token = REQUEST_PRIORITY.set("recovery" if saved else "extraction")
    inherited_deadline = REPORT_DEADLINE.get()
    own_deadline = (
        time.monotonic()
        + trusted_budget["report_seconds"]
        - prefix_spent
        - spent
        - (time.perf_counter() - start)
    )
    deadline_token = REPORT_DEADLINE.set(
        min(own_deadline, inherited_deadline)
        if inherited_deadline is not None
        else own_deadline
    )
    inflight = []
    futures = {}
    dirty_calls = set()
    # The outer work-directory owner recovers the global ledger once. Report workers
    # only reconcile their own reserved IDs, never mark another worker interrupted.
    ledger = getattr(generate, "ledger", None)
    if ledger is None:
        ledger = getattr(getattr(generate, "backend", None), "ledger", None)
    for i, call in enumerate(calls):
        if (
            call.get("seconds", 0) == 0
            and call.get("status") != "success"
            and not call.get("errors")
            and not (
                _raw_receipt_path(cache_dir, call["call_id"])
                and _raw_receipt_path(cache_dir, call["call_id"]).is_file()
            )
        ):
            elapsed = max(0, time.time() - call.get("reserved_at_wall", time.time()))
            if ledger:
                with ledger.connect() as db:
                    receipt = db.execute(
                        "SELECT seconds,status FROM calls WHERE id=?",
                        (call["call_id"],),
                    ).fetchone()
                if receipt and receipt[1] == "running":
                    journal.close()
                    if deadline_token is not None:
                        REPORT_DEADLINE.reset(deadline_token)
                    REQUEST_PRIORITY.reset(priority_token)
                    raise RuntimeError(
                        "active ledger request must be recovered by the work owner"
                    )
                if receipt and receipt[0] is not None:
                    elapsed = receipt[0]
                    call["timing_source"] = "durable-ledger"
            call.setdefault("timing_source", "interruption-wall-upper-bound")
            call["seconds"] = call["latency_seconds"] = elapsed
            call["errors"] = ["interrupted before durable model response"]
            dirty_calls.add(i)
    if saved and dirty_calls:
        if prefix_budget:
            # Visual cache verification happens before this text executor is
            # resumed. Its new elapsed time is already in prefix_spent, and this
            # invocation's setup is already measured by perf_counter. Charge
            # only the remaining checkpoint-to-entry interruption gap here.
            interruption_gap = max(0, started_at_wall - saved.get(
                "saved_at_wall", started_at_wall,
            ))
            visual_recovery = prefix_spent - previous_prefix["elapsed_seconds"]
            interrupted_wall = max(0, interruption_gap - visual_recovery)
        else:
            interrupted_wall = max(0, time.time() - saved.get("saved_at_wall", time.time()))
        spent += interrupted_wall
        REPORT_DEADLINE.set(REPORT_DEADLINE.get() - interrupted_wall)
    persisted_fragments = len(fragments) if saved else 0
    persisted_decisions = len(previous_decisions) if saved else 0
    plan_dirty = not saved
    fatal_error = None

    def persist():
        nonlocal persisted_fragments, persisted_decisions, plan_dirty
        if not journal:
            return
        decisions = previous_decisions + (
            policy.decisions[decision_start:] if policy else []
        )
        records = {
            "calls": {i: calls[i] for i in dirty_calls},
            "fragments": {
                i: asdict(fragments[i])
                for i in range(persisted_fragments, len(fragments))
            },
            "decisions": {
                i: decisions[i] for i in range(persisted_decisions, len(decisions))
            },
        }
        journal.update(
            fields={
                "trusted_budget": trusted_budget,
                **({"prefix_budget": prefix_budget} if prefix_budget else {}),
                "pending": [t.task_id for t in [*inflight, *pending]],
                "attempts": attempts,
                "depths": depths,
                "issued": issued,
                "spent": spent + time.perf_counter() - start,
                "saved_at_wall": time.time(),
                "initial_tasks": initial_tasks,
                "initial_plan_sha256": initial_plan_sha256,
                "effective_batch": effective_batch,
                "repair": repair,
            },
            records=records,
            replace={"tasks": [asdict(t) for t in plan.tasks]} if plan_dirty else None,
        )
        dirty_calls.clear()
        persisted_fragments, persisted_decisions = len(fragments), len(decisions)
        plan_dirty = False

    def state(phase, task=None, **extra):
        active_task = task or (pending[0] if pending else None)
        owned = active_task.owned if active_task else []
        context = active_task.context if active_task else []
        return {
            "phase": phase,
            "batch_size": effective_batch,
            "remaining_calls": (
                total_call_limit() - prefix_calls - issued
            ),
            "remaining_seconds": max(
                0, REPORT_DEADLINE.get() - time.monotonic()
                if prefix_budget else trusted_budget["report_seconds"] - spent
                - (time.perf_counter() - start),
            ),
            "max_attempts": max_attempts,
            "max_depth": trusted_budget["max_split_depth"],
            "depth": depths.get(task.task_id, 0) if task else 0,
            "structure": {
                "owned_regions": len(owned),
                "context_regions": len(context),
                "table_rows": sum(p.metadata.get("kind") == "table_row" for p in owned),
                "characters": sum(len(p.text) for p in owned),
                "years": len(
                    {str(y) for p in owned for y in p.metadata.get("years", [])}
                ),
            },
            "coverage": {
                "completed_tasks": len(fragments),
                "pending_tasks": len(pending) + len(inflight),
                "total_tasks": len(plan.tasks),
            },
            **extra,
        }

    def select(phase, task=None, **extra):
        if policy:
            from .rsi import validate_action

            trusted_state = state(phase, task, **extra)
            return validate_action(policy.choose(trusted_state), trusted_state)
        return None

    def check_budget(count=1):
        deadline = REPORT_DEADLINE.get()
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("report deadline exhausted")
        if prefix_calls + issued + count > total_call_limit():
            raise RuntimeError("report call budget exhausted")

    def total_call_limit():
        return (prefix_budget["total_call_limit"] if prefix_budget else
                trusted_budget["calls_per_initial_task"] * initial_tasks
                + trusted_budget["extra_calls"])

    def shared_budget_evidence():
        text_spent = spent + time.perf_counter() - start
        return {
            "prefix": prefix_budget,
            "total_call_limit": total_call_limit(),
            "text_issued_calls": issued,
            "total_issued_calls": prefix_calls + issued,
            "text_elapsed_seconds": text_spent,
            "total_elapsed_seconds": prefix_spent + text_spent,
        }

    def split(task):
        nonlocal plan_dirty
        if depths.get(task.task_id, 0) >= trusted_budget["max_split_depth"]:
            raise RuntimeError("report split depth exhausted")
        children = split_task(task, snapshot)
        index = plan.tasks.index(task)
        plan.tasks[index : index + 1] = children
        for child in children:
            depths[child.task_id] = depths.get(task.task_id, 0) + 1
            child.prompt_tokens = len(
                tokenizer(child.text(), add_special_tokens=False)["input_ids"]
            )
        pending.extend(children)
        plan_dirty = True

    def capture_raw(task, call, response, elapsed):
        """Durable unvalidated backend output; never a successful fragment cache."""
        if (
            not cache_dir
            or not isinstance(response, dict)
            or not isinstance(response.get("text"), str)
        ):
            return
        backend_id = response.get("call_id", call["call_id"])
        if isinstance(backend_id, str) and backend_id and len(backend_id) <= 256:
            call["call_id"] = backend_id
        encoded = response["text"].encode()
        limit = _raw_limit(config)
        raw = encoded[:limit].decode("utf-8", errors="ignore")
        measured = response.get("latency_seconds", elapsed)
        if (
            type(measured) not in (int, float)
            or not math.isfinite(measured)
            or measured < 0
        ):
            measured = elapsed
        call["seconds"] = call["latency_seconds"] = measured
        call["output_sha256"] = hashlib.sha256(encoded).hexdigest()
        call["response_validation"] = "pending"
        stored = {
            "schema_version": 1,
            "cache_identity": cache_key,
            "call_id": call["call_id"],
            "task_id": task.task_id,
            "prompt_sha256": call["prompt_sha256"],
            "source_ranges": task.owned_keys,
            "raw": raw,
            "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "output_sha256": call["output_sha256"],
            "truncated": len(encoded) > limit,
            "finish_reason": response.get("finish_reason")[:128]
            if isinstance(response.get("finish_reason"), str)
            else None,
            "output_tokens": response.get("output_tokens")
            if type(response.get("output_tokens")) is int
            else None,
            "latency_seconds": measured,
        }
        path = _raw_receipt_path(cache_dir, call["call_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, stored)
        call["raw_receipt"] = {
            "path": path.relative_to(cache_dir).as_posix(),
            "sha256": json_digest(stored),
            "truncated": stored["truncated"],
        }
        persist()

    def resume_raw(task):
        for index in range(len(calls) - 1, -1, -1):
            call = calls[index]
            if call.get("task_id") != task.task_id:
                continue
            path = _raw_receipt_path(cache_dir, call["call_id"])
            if (
                call.get("response_validation") in ("failed", "success")
                or path is None
                or not path.is_file()
            ):
                return None
            if path.stat().st_size > 2 * _raw_limit(config) + 65536:
                raise ValueError("raw response receipt exceeds bound")
            stored = json.loads(path.read_text())
            digest = json_digest(stored)
            if (
                stored.get("schema_version") != 1
                or stored.get("cache_identity") != cache_key
                or stored.get("call_id") != call["call_id"]
                or stored.get("task_id") != task.task_id
                or stored.get("prompt_sha256") != call["prompt_sha256"]
                or stored.get("source_ranges") != task.owned_keys
                or not isinstance(stored.get("raw"), str)
                or len(stored["raw"].encode()) > _raw_limit(config)
                or hashlib.sha256(stored["raw"].encode()).hexdigest()
                != stored.get("raw_sha256")
                or stored.get("truncated") is not False
                or stored.get("raw_sha256") != stored.get("output_sha256")
                or call.get("output_sha256") not in (None, stored["output_sha256"])
                or call.get("raw_receipt", {}).get("sha256", digest) != digest
            ):
                raise ValueError(
                    "raw response receipt identity/hash mismatch or truncated"
                )
            if ledger:
                with ledger.connect() as db:
                    entry = db.execute(
                        "SELECT status,seconds,receipt FROM calls WHERE id=?",
                        (call["call_id"],),
                    ).fetchone()
                recorded = json.loads(entry[2]) if entry and entry[2] else {}
                if (
                    not entry
                    or entry[0] not in ("success", "failed")
                    or entry[1] != stored["latency_seconds"]
                    or recorded.get("output_sha256") != stored["output_sha256"]
                    or recorded.get("prompt_sha256") != call["prompt_sha256"]
                ):
                    raise ValueError(
                        "raw response does not match actual backend ledger"
                    )
            call["raw_receipt"] = {
                "path": path.relative_to(cache_dir).as_posix(),
                "sha256": digest,
                "truncated": False,
            }
            call["response_validation"] = "pending"
            call["errors"] = []
            response = {
                "text": stored["raw"],
                "call_id": stored["call_id"],
                "finish_reason": stored["finish_reason"],
                "output_tokens": stored["output_tokens"],
                "latency_seconds": stored["latency_seconds"],
            }
            return index, call, response
        return None

    last_call_by_task = {call.get("task_id"): call for call in calls}

    def has_local_result(task):
        if not cache_dir:
            return False
        if (cache_dir / (task.task_id + ".json")).is_file():
            return True
        call = last_call_by_task.get(task.task_id)
        return bool(
            call
            and call.get("response_validation") not in ("failed", "success")
            and _raw_receipt_path(cache_dir, call["call_id"]).is_file()
        )

    local_ready = (
        {task.task_id for task in pending if has_local_result(task)} if saved else set()
    )

    def prepare(task):
        nonlocal issued
        cache = cache_dir / (task.task_id + ".json") if cache_dir else None
        if cache and cache.is_file():
            stored = json.loads(cache.read_text())
            if (
                stored["call"].get("output_sha256")
                != hashlib.sha256(stored["raw"].encode()).hexdigest()
                or stored["call"].get("status") != "success"
                or stored["call"].get("source_ranges") != task.owned_keys
            ):
                raise ValueError("cached receipt/content mismatch")
            fragments.append(
                parse_fragment(stored["raw"], task, snapshot, stored["call"]["call_id"])
            )
            generation_responses[stored["call"]["call_id"]] = stored["raw"]
            existing = {c["call_id"]: i for i, c in enumerate(calls)}
            for call in stored["calls"]:
                if call["call_id"] not in existing:
                    dirty_calls.add(len(calls))
                    existing[call["call_id"]] = len(calls)
                    calls.append(call)
                else:
                    i = existing[call["call_id"]]
                    if calls[i] != call:
                        # A response file can be durable before its checkpoint delta.
                        # Only replace an uncompleted matching request reservation.
                        old = calls[i]
                        if (
                            old.get("output_sha256")
                            not in (None, call.get("output_sha256"))
                            or old.get("response_validation") not in (None, "pending")
                            or old.get("prompt_sha256") != call.get("prompt_sha256")
                            or old.get("source_ranges") != call.get("source_ranges")
                        ):
                            raise ValueError("conflicting cached receipt")
                        calls[i] = call
                        dirty_calls.add(i)
            return None
        replay = resume_raw(task)
        if replay is not None:
            index, call, response = replay
            complete(
                task,
                call,
                cache,
                index,
                response,
                response["latency_seconds"],
                replayed=True,
            )
            return None
        if attempts.get(task.task_id, {}).get("count", 0) >= max_attempts:
            raise RuntimeError("task retry budget already exhausted")
        feedback = attempts.get(task.task_id, {}).get("error")
        if feedback:
            feedback_ids = tokenizer(feedback, add_special_tokens=False)["input_ids"][
                :256
            ]
            feedback = (
                tokenizer.decode(feedback_ids)
                if hasattr(tokenizer, "decode")
                else feedback[:256]
            )
        task_instruction = instruction + (
            "\n上次格式错误：" + feedback + "。必须修正。" if feedback else ""
        )
        try:
            ids = build_prompt(
                tokenizer,
                task.text(),
                max_model_len=config["input_budget"] + config["output_budget"],
                max_new_tokens=config["output_budget"],
                instruction=task_instruction,
            )
        except ContextBudgetError:
            action = select("failure", task, error_kind="length", attempt=0)
            if (action and action["action"] != "split") or depths.get(
                task.task_id, 0
            ) >= 8:
                raise
            split(task)
            return None
        check_budget()
        call = _receipt(ids, "extraction", task)
        index = len(calls)
        calls.append(call)
        dirty_calls.add(index)
        issued += 1
        attempts.setdefault(task.task_id, {"count": 0})["count"] += 1
        return ids, call, cache, index

    def complete(task, call, cache, index, response, elapsed, *, replayed=False):
        nonlocal effective_batch, fatal_error
        dirty_calls.add(index)
        processed = False
        try:
            if isinstance(response, BaseException):
                call["call_id"] = getattr(response, "call_id", call["call_id"])
                raise response
            if not replayed:
                capture_raw(task, call, response, elapsed)
            if call.get("raw_receipt", {}).get("truncated"):
                raise ValueError("raw response exceeds bounded receipt size")
            raw = _complete(call, response, elapsed, config["output_budget"])
            if len(raw.encode()) > _raw_limit(config):
                raise ValueError("raw response exceeds bounded evidence size")
            if any(c is not call and c["call_id"] == call["call_id"] for c in calls):
                raise ValueError("duplicate backend receipt")
            frag = parse_fragment(raw, task, snapshot, call["call_id"])
            accepted_call = {
                **call,
                "status": "success",
                "response_validation": "success",
            }
            if cache:
                try:
                    atomic_json(
                        cache,
                        {
                            "raw": raw,
                            "call": accepted_call,
                            "calls": [
                                accepted_call if c is call else c
                                for c in calls
                                if c.get("task_id") == task.task_id
                            ],
                        },
                    )
                except BaseException as error:
                    # The raw response is already durable. A storage failure is
                    # not a model validation failure: keep this task pending so
                    # recovery validates the same receipt without a new attempt.
                    fatal_error = error
                    if task not in pending:
                        pending.append(task)
                    processed = True
                    return
            call.update(accepted_call)
            fragments.append(frag)
            generation_responses[call["call_id"]] = raw
            processed = True
        except Exception as error:
            processed = True
            call["response_validation"] = "failed"
            measured = (
                getattr(response, "latency_seconds", elapsed)
                if isinstance(response, Exception)
                else response.get("latency_seconds", elapsed)
                if isinstance(response, dict)
                else elapsed
            )
            if (
                type(measured) not in (int, float)
                or not math.isfinite(measured)
                or measured < 0
            ):
                measured = elapsed
            call["seconds"] = call["latency_seconds"] = measured
            call["errors"].append(type(error).__name__ + ":" + str(error))
            attempt = attempts[task.task_id]
            attempt["error"] = str(error)
            oom = (
                "out of memory" in str(error).lower()
                or "outofmemory" in str(error).lower()
            )
            if oom:
                effective_batch = 1
            kind = (
                "length"
                if call.get("finish_reason") == "length"
                else "temporary"
                if oom or isinstance(error, (TimeoutError, ConnectionError, OSError))
                else "format"
            )
            action = select("failure", task, error_kind=kind, attempt=attempt["count"])
            try:
                if action and action["action"] == "fail":
                    raise error
                if (action and action["action"] == "split") or (
                    not policy and kind == "length" and depths.get(task.task_id, 0) < 8
                ):
                    split(task)
                elif (
                    isinstance(
                        error, (TimeoutError, ConnectionError, OSError, ValueError)
                    )
                    and attempt["count"] < max_attempts
                ):
                    pending.append(task)
                else:
                    raise error
            except Exception as failure:
                fatal_error = failure
                pending.append(task)
        finally:
            dirty_calls.add(index)
            if processed and task in inflight:
                inflight.remove(task)
            persist()

    try:
        persist()
        while pending or futures:
            pending.sort(key=lambda t: t.prompt_tokens)
            if pending and not fatal_error:
                try:
                    # Validation of an already completed response requires no model
                    # budget. Do it before selecting or reserving another request.
                    while pending and not fatal_error:
                        local_task = next(
                            (task for task in pending if task.task_id in local_ready),
                            None,
                        )
                        if local_task is None:
                            break
                        local_ready.discard(local_task.task_id)
                        pending.remove(local_task)
                        inflight.append(local_task)
                        if prepare(local_task) is not None:
                            raise RuntimeError(
                                "local response disappeared before validation"
                            )
                        if local_task in inflight:
                            inflight.remove(local_task)
                        persist()
                    if fatal_error:
                        raise fatal_error
                    if not pending:
                        continue
                    decision = select("schedule")
                    if decision and decision["action"] == "fail":
                        raise TimeoutError("RSI report budget exhausted")
                    size = decision["batch_size"] if decision else effective_batch
                    if hasattr(generate, "submit"):
                        # Keep a bounded window, refill immediately after each durable receipt.
                        while pending and len(futures) < size and not fatal_error:
                            task = pending.pop(0)
                            inflight.append(task)
                            prepared = prepare(task)
                            if prepared is None:
                                if task in inflight:
                                    inflight.remove(task)
                                persist()
                                continue
                            ids, call, cache, index = prepared
                            persist()
                            submitted = time.perf_counter()
                            call_token = REQUEST_CALL_ID.set(call["call_id"])
                            try:
                                future = generate.submit(
                                    (ids, config["output_budget"]),
                                    priority="recovery" if saved else "extraction",
                                )
                            finally:
                                REQUEST_CALL_ID.reset(call_token)
                            futures[future] = task, call, cache, index, submitted
                    else:
                        # Local batch engines retain their supported batch API; API dispatcher streams.
                        batch, requests = [], []
                        for _ in range(min(size, len(pending))):
                            if fatal_error or not pending:
                                break
                            task = pending.pop(0)
                            inflight.append(task)
                            prepared = prepare(task)
                            if prepared is None:
                                if task in inflight:
                                    inflight.remove(task)
                                persist()
                                continue
                            ids, call, cache, index = prepared
                            batch.append((task, call, cache, index))
                            requests.append((ids, config["output_budget"]))
                        persist()
                        submitted = time.perf_counter()
                        if requests:
                            if hasattr(generate, "completed"):
                                values = generate.completed(requests)
                            else:
                                try:
                                    if hasattr(generate, "batch"):
                                        batch_token = REQUEST_CALL_IDS.set(
                                            tuple(item[1]["call_id"] for item in batch)
                                        )
                                        try:
                                            responses = generate.batch(requests)
                                        finally:
                                            REQUEST_CALL_IDS.reset(batch_token)
                                    else:
                                        responses = []
                                        for request, item in zip(requests, batch):
                                            call_token = REQUEST_CALL_ID.set(
                                                item[1]["call_id"]
                                            )
                                            try:
                                                check_budget(0)
                                                responses.append(generate(*request))
                                            except Exception as error:
                                                responses.append(error)
                                            finally:
                                                REQUEST_CALL_ID.reset(call_token)
                                    if len(responses) != len(requests):
                                        raise ValueError(
                                            "batch response count mismatch"
                                        )
                                except Exception as error:
                                    responses = [error] * len(requests)
                                values = enumerate(responses)
                            for i, response in values:
                                complete(
                                    *batch[i], response, time.perf_counter() - submitted
                                )
                except Exception as error:
                    fatal_error = error
                    for task in list(inflight):
                        if not any(info[0] is task for info in futures.values()):
                            inflight.remove(task)
                            if task not in pending:
                                pending.append(task)
                    persist()
            if futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    task, call, cache, index, submitted = futures.pop(future)
                    try:
                        response = future.result()
                    except Exception as error:
                        response = error
                    complete(
                        task,
                        call,
                        cache,
                        index,
                        response,
                        time.perf_counter() - submitted,
                    )
            if fatal_error and not futures:
                raise fatal_error
        merged = merge_fragments(snapshot, plan, fragments)
        if config["workflow"] == "bounded-react" and merged["conflicts"]:
            if not repair.get("reserved"):
                action = select(
                    "merge",
                    allow_repair=True,
                    react_calls=0,
                    conflicts={
                        "count": len(merged["conflicts"]),
                        "fields": [c["field"] for c in merged["conflicts"]],
                    },
                )
                if action and action["action"] != "repair":
                    raise ValueError("RSI repair budget unavailable")
                check_budget(2)
                issued += 2
                repair["reserved"] = 2
                persist()  # Both slots reserved atomically before either request starts.

            def persist_repair():
                dirty_calls.update(repair.get("requests", {}).values())
                persist()

            def check_repair_request(role):
                check_budget(0)
                if repair.get("reserved") != 2 or len(repair.get("requests", {})) >= 2:
                    raise RuntimeError("repair reservation exhausted")

            merged = bounded_repair(
                snapshot,
                merged,
                tokenizer,
                generate,
                config,
                calls,
                checkpoint=repair,
                persist=persist_repair,
                before_request=check_repair_request,
            )
        if merged["conflicts"]:
            resolve_basic_conflicts(merged)
        _parse_report(json.dumps(merged["data"], ensure_ascii=False), require_basic=False)
        for role, stored in repair.get("responses", {}).items():
            repair_call = calls[repair["requests"][role]]
            if repair_call.get("status") == "success":
                generation_responses[repair_call["call_id"]] = stored["raw"]
        accepted = {c["call_id"]: c for c in calls if c.get("status") == "success"}
        if set(generation_responses) != set(accepted):
            raise ValueError(
                "accepted model responses missing from generation evidence"
            )
        for call_id, raw in generation_responses.items():
            if (
                not isinstance(raw, str)
                or len(raw.encode()) > _raw_limit(config)
                or hashlib.sha256(raw.encode()).hexdigest()
                != accepted[call_id]["output_sha256"]
            ):
                raise ValueError(
                    "generation evidence differs from actual bounded response"
                )
        return {
            **merged,
            "model_calls": calls,
            "raw_output": json.dumps(merged["data"], ensure_ascii=False),
            "generation_evidence": {
                "schema_version": 1,
                "responses": generation_responses,
            },
            "seconds": sum(c["seconds"] for c in calls),
            "wall_seconds": time.perf_counter() - start,
            "snapshot": asdict(snapshot),
            "chunk_plan": asdict(plan),
            "fragments": [asdict(f) for f in fragments],
            "snapshot_sha256": snapshot.digest,
            "plan_sha256": plan.digest,
            **({"shared_budget": shared_budget_evidence()} if prefix_budget else {}),
            **(
                {
                    "rsi": {
                        "bundle": policy.bundle.as_dict(),
                        "bundle_sha256": policy.bundle.digest,
                        "decisions": previous_decisions
                        + policy.decisions[decision_start:],
                        "issued_calls": issued,
                        "initial_tasks": initial_tasks,
                        "spent_seconds": spent + time.perf_counter() - start,
                        "mode": "frozen",
                        "checkpoint_identity": cache_key,
                        **(
                            {"controller": policy.telemetry()}
                            if hasattr(policy, "telemetry")
                            else {}
                        ),
                    }
                }
                if policy
                else {}
            ),
        }
    except Exception as error:
        failure = ChunkFailure(
            calls, raw_output="", wall_seconds=time.perf_counter() - start
        )
        failure.reason = f"{type(error).__name__}: {error}"
        failure.partial_evidence = {
            "snapshot": asdict(snapshot),
            "chunk_plan": asdict(plan),
            "fragments": [asdict(f) for f in fragments],
            "snapshot_sha256": snapshot.digest,
            "plan_sha256": plan.digest,
            "coverage": {
                **plan.coverage(snapshot),
                "complete": False,
                "completed_tasks": [f.task_id for f in fragments],
            },
            "contributing_call_ids": [f.call_id for f in fragments if f.call_id],
            **({"shared_budget": shared_budget_evidence()} if prefix_budget else {}),
        }
        raise failure from error
    finally:
        if journal:
            journal.close()
        if deadline_token is not None:
            REPORT_DEADLINE.reset(deadline_token)
        REQUEST_PRIORITY.reset(priority_token)
