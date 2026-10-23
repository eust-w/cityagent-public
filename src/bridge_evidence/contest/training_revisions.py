"""Evidence-adjudicated overlays; originals stay immutable and entities split first."""

import hashlib
import json
from copy import deepcopy

from .run_state import json_digest


def apply_revision(original, snapshot, revision):
    if (
        revision.get("document_sha256") != snapshot.document_sha256
        or revision.get("status") != "approved"
        or not revision.get("adjudication")
        or not isinstance(revision.get("changes"), list)
    ):
        raise ValueError("approved, source-bound adjudication required")
    result = deepcopy(original)
    sources = snapshot.sources()
    for change in revision["changes"]:
        source = sources.get(change.get("source_id"))
        quote = change.get("quote")
        if (
            source is None
            or not isinstance(quote, str)
            or not quote
            or quote not in source.text
        ):
            raise ValueError("revision requires exact source evidence")
        section = change["section"]
        if section not in ("基本信息", "维修建议", "病害明细") or not isinstance(
            change.get("field"), str
        ):
            raise ValueError("invalid revision target")
        target = (
            result[section]
            if section == "基本信息"
            else result[section][change["row_index"]]
        )
        if isinstance(change.get("value"), (list, dict)):
            raise ValueError("revision values must be scalar")
        target[change["field"]] = change.get("value")
    return result


def entity_partition(reports, seed=17, train_fraction=0.8):
    if not 0 < train_fraction < 1:
        raise ValueError("invalid split ratio")
    output = []
    for report in reports:
        if report.get("entity_verified") is not True or not report.get("bridge_entity"):
            raise ValueError(
                "verified bridge entity required before splitting/chunking"
            )
        value = int(
            hashlib.sha256(f"{seed}:{report['bridge_entity']}".encode()).hexdigest(), 16
        ) / (2**256)
        output.append(
            {**report, "split": "train" if value < train_fraction else "regression"}
        )
    return output


def build_subtasks(snapshot, plan, annotations):
    """Only explicitly approved, complete source-level targets become training rows."""
    approved = {a["task_id"]: a for a in annotations if a.get("status") == "approved"}
    rows = []
    for task in plan.tasks:
        annotation = approved.get(task.task_id)
        if not annotation:
            continue
        if sorted(annotation.get("covered", [])) != sorted(
            task.owned_keys
        ) or not annotation.get("adjudication"):
            raise ValueError("training target lacks complete owned-source adjudication")
        from .long_document import parse_fragment

        raw = json.dumps(annotation["target"], ensure_ascii=False)
        parse_fragment(raw, task, snapshot)
        rows.append(
            {
                "instruction": plan.instruction,
                "input": task.text(),
                "output": raw,
                "document_sha256": snapshot.document_sha256,
                "snapshot_sha256": snapshot.digest,
                "task_id": task.task_id,
                "revision_sha256": json_digest(annotation),
                "categories": annotation.get("categories", []),
            }
        )
    return rows
