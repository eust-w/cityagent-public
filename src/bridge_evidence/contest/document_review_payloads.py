"""Lossless, budgeted reverse-review requests; no semantic judgements.

The primary source ranges partition the report. Context and evidence tables may
repeat across requests, but never count as additional primary coverage. Each
related output keeps its entire supplied fact/source closure, even when those
sources are in another section of the report.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Callable, Iterator

from .evidence_manifest import make_chunks, render_spans


def _compact(value: Any, *, canonical: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=canonical, allow_nan=False)


class ReviewPayloadBudgetError(ValueError):
    """An indivisible source assignment cannot fit without losing evidence.

    The caller can expose ``details`` to the agent/operator. Retrying unchanged
    evidence will not fix this error. No subset of the related outputs has been
    presented as the complete set.
    """

    def __init__(self, details: dict):
        self.details = deepcopy(details)
        super().__init__(
            "Reverse review cannot fit a single-character source assignment "
            "with its complete related outputs/evidence. Repair the source "
            "bindings where appropriate, or use an explicit grouped-coverage "
            "review protocol; evidence has not been truncated. " + _compact(details))


def _intersects(source: dict, spans: list[dict]) -> bool:
    """Half-open primary interval overlap, never ID-only or text matching."""
    return any(source.get("evidence_id") == span["evidence_id"]
               and source["start"] < span["end"] and span["start"] < source["end"]
               for span in spans if source.get("evidence_id") is not None)


def _make_payload(manifest: dict, items: list[dict], background: dict,
                  spans: list[dict]) -> dict:
    selected = [item for item in items if any(_intersects(source, spans) for source in item["sources"])]
    related_output = []
    fact_table: dict[str, dict] = {}
    source_table: dict[str, dict] = {}
    fact_keys: dict[str, str] = {}
    source_keys: dict[str, str] = {}

    def intern(values: list[dict], table: dict, keys: dict, prefix: str) -> list[str]:
        refs = []
        for value in values:
            canonical = _compact(value, canonical=True)
            if canonical not in keys:
                key = f"{prefix}{len(table) + 1:06d}"
                keys[canonical] = key
                table[key] = deepcopy(value)
            key = keys[canonical]
            if key not in refs:
                refs.append(key)
        return refs

    for item in selected:
        related_output.append({
            "path": item["path"], "value": deepcopy(item["value"]),
            "fact_refs": intern(item["facts"], fact_table, fact_keys, "F"),
            "source_refs": intern(item["sources"], source_table, source_keys, "S"),
        })
    primary = [{"coverage_id": span["evidence_id"], "start": span["start"], "end": span["end"]}
               for span in spans]
    return {
        "direction": "REVERSE", "output_items": [],
        "related_output": related_output,
        "related_facts": fact_table, "related_evidence": source_table,
        "source_spans": deepcopy(spans), "source": render_spans(manifest, primary)["text"],
        "document_background": deepcopy(background),
    }


def _split(spans: list[dict]) -> tuple[list[dict], list[dict]]:
    """Bisect primary characters, preserving order without overlap or gaps."""
    remaining = sum(span["end"] - span["start"] for span in spans) // 2
    left, right = [], []
    for span in spans:
        length = span["end"] - span["start"]
        if remaining >= length:
            left.append(span.copy())
            remaining -= length
        elif remaining:
            boundary = span["start"] + remaining
            left.append({**span, "end": boundary})
            right.append({**span, "start": boundary})
            remaining = 0
        else:
            right.append(span.copy())
    return left, right


def reverse_payloads(manifest: dict, output_items: list[dict], background: dict, *,
                     count_chat_tokens: Callable[[list[dict]], int],
                     max_model_len: int, output_tokens: int, source_tokens: int,
                     count_tokens: Callable[[str], int], review_prompt: str,
                     acknowledgement_mode: str = "legacy",
                     assignment_id: Callable[[dict], str] | None = None,
                     include_citation_gaps: bool = False,
                     project_payload: Callable[[dict], dict] | None = None) -> Iterator[dict]:
    """Yield complete reverse-review assignments within actual chat budgets.

    ``output_items`` uses DocumentWorkflow._output_items' path/value/facts/sources
    format. Related items are chosen only by primary text interval overlap. Their
    complete closures are retained in exact-deduplicated ``related_facts`` and
    ``related_evidence`` tables, addressed by each item's fact_refs/source_refs.
    Images in an associated closure survive; independent image review belongs to
    the workflow. Empty text units do not claim primary review coverage.

    ``source_tokens`` is the initial source chunk target, not a limit on the
    combined evidence closure. An oversized fixed render context can bypass that
    target, but can never bypass the final chat limit. Oversize requests split
    primary ranges and recompute associations. A single-character assignment
    that still exceeds the budget raises ReviewPayloadBudgetError. No facts,
    sources, outputs, or background values are silently shortened.

    The standard success JSON is also counted against output_tokens. Legacy
    acknowledgement echoes all assigned source spans. Compact acknowledgement
    budgets an assignment ID and an explicit all-assigned flag. When supplied,
    ``assignment_id(payload)`` returns the real ``ra_`` plus 32 lowercase hex
    characters; its complete success JSON is counted exactly. The workflow
    owns this ID and must validate it. Without that callback, a 35-character
    placeholder plus 35 extra tokens provides conservative headroom for ID
    variation, not an exact count or a bound for arbitrary tokenizers.
    The budgeting ID is never inserted into the returned source payload.
    Both modes keep the full source ranges and related evidence in the request.
    Neither promises that substantive findings fit the same output budget.
    """
    if type(include_citation_gaps) is not bool:
        raise ValueError("include_citation_gaps must be a boolean")
    if acknowledgement_mode not in ("legacy", "compact"):
        raise ValueError("acknowledgement_mode must be 'legacy' or 'compact'")
    if acknowledgement_mode == "compact" and assignment_id is not None and not callable(assignment_id):
        raise ValueError("assignment_id must be a callable or None")
    for name, value in (("max_model_len", max_model_len), ("output_tokens", output_tokens),
                        ("source_tokens", source_tokens)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if output_tokens >= max_model_len:
        raise ValueError("Reserved output must leave input context")
    units = {unit["id"]: unit for unit in manifest["units"]}
    if len(units) != len(manifest["units"]):
        raise ValueError("Duplicate source evidence IDs")
    paths: set[str] = set()
    for item in output_items:
        if not isinstance(item.get("path"), str) or item["path"] in paths:
            raise ValueError("Output items need distinct string paths")
        paths.add(item["path"])
        for source in item["sources"]:
            if "evidence_id" not in source:
                if "image_id" not in source:
                    raise ValueError("An output source needs a text or image evidence ID")
                continue
            identifier, start, end = source["evidence_id"], source.get("start"), source.get("end")
            if (identifier not in units or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(units[identifier]["text"])):
                raise ValueError("Output source has an invalid primary character interval")

    # make_chunks may reject a header/context larger than its source-only target.
    # Retry from complete units in that specific case: the final budget below
    # measures the actual prompt and is the authority, never a source-only guess.
    try:
        chunks = make_chunks(manifest, count_tokens, source_tokens)
        initial = [[{"evidence_id": span["coverage_id"], "start": span["start"], "end": span["end"]}
                    for span in chunk["spans"] if span["start"] < span["end"]]
                   for chunk in chunks]
    except ValueError as exc:
        if not str(exc).startswith(("Evidence header/context or one character exceeds max_tokens",
                                    "Evidence metadata exceeds max_tokens")):
            raise
        initial = [[{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}]
                   for unit in manifest["units"] if unit["text"]]

    pending = list(reversed([spans for spans in initial if spans]))
    while pending:
        spans = pending.pop()
        payload = _make_payload(manifest, output_items, background, spans)
        if project_payload is not None:
            payload = project_payload(payload)
        messages = [{"role": "system", "content": review_prompt},
                    {"role": "user", "content": _compact(payload)}]
        prompt_tokens = count_chat_tokens(messages)
        protocol_reserve = 0
        if acknowledgement_mode == "compact":
            if assignment_id is None:
                identifier = "ra_" + "0" * 32
                protocol_reserve = 35
            else:
                identifier = assignment_id(deepcopy(payload))
                if not isinstance(identifier, str) or re.fullmatch(r"ra_[0-9a-f]{32}", identifier) is None:
                    raise ValueError("assignment_id must return 'ra_' plus 32 lowercase hex characters")
            protocol_reply = {
                "passed": True, "checked_assignment_id": identifier,
                "checked_all_assigned": True,
            }
        else:
            protocol_reply = {
                "passed": True, "checked_output_paths": [], "checked_evidence_spans": spans,
            }
        protocol_reply.update(unsupported_claims=[], missing_items=[], conflicts=[], unresolved_items=[])
        if include_citation_gaps:
            protocol_reply["citation_gaps"] = []
        protocol_tokens = count_tokens(_compact(protocol_reply)) + protocol_reserve
        if prompt_tokens + output_tokens <= max_model_len and protocol_tokens <= output_tokens:
            yield payload
            continue
        if sum(span["end"] - span["start"] for span in spans) == 1:
            raise ReviewPayloadBudgetError({
                "source_spans": spans, "related_paths": [item["path"] for item in payload["related_output"]],
                "prompt_tokens": prompt_tokens, "max_model_len": max_model_len,
                "output_tokens": output_tokens, "minimum_protocol_output_tokens": protocol_tokens,
            })
        left, right = _split(spans)
        pending.extend((right, left))
