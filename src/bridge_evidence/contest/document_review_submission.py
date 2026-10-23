"""A simple model-facing review submission over the existing atomic engine.

The model decides which candidates to select, merge, edit or exclude, and the
status/reason of each named question. This facade only routes those decisions
using current host state; it never fills missing reviews or infers completeness.
"""
from __future__ import annotations

from copy import deepcopy
import re

from jsonschema import Draft202012Validator

from .docx_writer import ADVICE_FIELDS, DEFECT_FIELDS
from .document_candidate_review import CandidateReviewError, CandidateReviewTools
from .run_state import json_digest


REVIEW_BATCH_LIMIT = 5
REVIEW_COLLECTIONS = ("selections", "merges", "patches", "exclusions", "reviews")
REVIEW_BATCH_POLICY = (
    f"Submit one related group of at most {REVIEW_BATCH_LIMIT} decisions across all five collections. "
    "One merge group counts as one decision; do not split its members. Keep dependent edits and "
    "their review together. Wait for its receipt before the next batch. Omitted decisions are not reviewed."
)
REVIEW_REJECTED_ACTION = (
    "This entire submission saved zero decisions; previously successful batches remain saved. "
    "Correct the indicated error and resubmit all still-needed decisions from THIS failed batch, "
    "or regroup them into related batches of at most 5. Do not resubmit earlier successful batches."
)


def review_action_count(payload):
    if not isinstance(payload, dict) or any(payload.get(k) is not None
            and not isinstance(payload[k], list) for k in REVIEW_COLLECTIONS):
        return None  # Malformed JSON/containers do not prove a batch size.
    return sum(len(payload.get(k) or []) for k in REVIEW_COLLECTIONS)


def rejected_review_receipt(payload=None, *, version=None, unconfirmed=False):
    return {"status": "unconfirmed" if unconfirmed else "rejected",
        "action_count": review_action_count(payload), "limit": REVIEW_BATCH_LIMIT,
        "accepted": {k: [] for k in REVIEW_COLLECTIONS},
        "workspace_version": None if unconfirmed else version,
        "applied": None if unconfirmed else False,
        "instruction": ("The transaction outcome is unconfirmed. Inspect saved state and recover the exact "
            "original invocation from its journal; do not blindly submit these edits as a new batch."
            if unconfirmed else REVIEW_REJECTED_ACTION)}


def submit_review_schema(limit=REVIEW_BATCH_LIMIT):
    identifier = {"type": "string", "minLength": 1}
    reason = {"type": "string", "minLength": 1,
        "description": "Your explicit review conclusion and reason from the supplied evidence."}
    changes = {"type": "object", "minProperties": 1, "additionalProperties": False,
        "description": "Basic field: {value: reviewed value}; detail row: only changed delivery columns. No indexes, versions, fact bindings or source offsets.",
        "properties": {field: {"type": ["string", "number", "null"]}
            for field in sorted({"value", *ADVICE_FIELDS, *DEFECT_FIELDS})}}

    def obj(properties, required):
        return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": required}

    def array(items, description):
        return {"type": ["array", "null"], "default": None, "maxItems": limit,
            "description": description + " Omit or pass null/[] when there are no such decisions.", "items": items}

    return {"type": "object", "description": REVIEW_BATCH_POLICY,
        "additionalProperties": False, "properties": {
        "selections": array(identifier, "Existing basic-field candidate IDs explicitly selected by you; already-bound detail rows need no selection."),
        "merges": array(obj({"keep_id": identifier, "other_ids": {"type": "array", "minItems": 1, "items": identifier},
            "reason": reason, "changes": changes}, ["keep_id", "other_ids", "reason"]),
            "Explicitly merge same-section detail rows into keep_id, retaining all source bindings; changes are optional."),
        "patches": array(obj({"candidate_id": identifier, "changes": changes}, ["candidate_id", "changes"]),
            "Change only specified fields of an existing candidate's output; original source candidates stay preserved."),
        "exclusions": array(obj({"candidate_id": identifier, "reason": reason}, ["candidate_id", "reason"]),
            "Explicitly remove the named output candidate with a reason, without erasing its original facts."),
        "reviews": array(obj({"question_id": identifier, "status": {"type": "string", "enum": ["resolved", "unknown"]},
            "reason": reason}, ["question_id", "status", "reason"]),
            "Use the actual question ID for every review, including task-coverage questions. Only supplied statuses are recorded; unknown coverage remains pending.")}}


def _error(code, message, path):
    raise CandidateReviewError(code, message, path)


class ReviewSubmissionTools(CandidateReviewTools):
    """Same constructor and transaction store as CandidateReviewTools."""

    @staticmethod
    def _schema_check(payload):
        try:
            json_digest(payload)
        except (TypeError, ValueError):
            _error("REVIEW_ARGUMENTS", "Submission must be finite JSON", "arguments")
        count = review_action_count(payload)
        if count is not None and count > REVIEW_BATCH_LIMIT:
            position = 0
            for collection in REVIEW_COLLECTIONS:
                for index, _ in enumerate(payload.get(collection) or []):
                    position += 1
                    if position > REVIEW_BATCH_LIMIT:
                        _error("REVIEW_BATCH_LIMIT", REVIEW_BATCH_POLICY, f"{collection}[{index}]")
        error = next(Draft202012Validator(submit_review_schema()).iter_errors(payload), None)
        if error:
            path = "".join(f"[{p}]" if isinstance(p, int) else ("." if i else "") + p
                for i, p in enumerate(error.absolute_path)) or "arguments"
            if error.validator == "required":
                missing = [key for key in error.validator_value if key not in error.instance]
                if missing:
                    path += ("." if path != "arguments" else "") + missing[0]
            _error("REVIEW_ARGUMENTS", "Submission does not match submit_review schema", path)
        for collection in ("merges", "exclusions", "reviews"):
            for i, row in enumerate(payload.get(collection) or []):
                if not row["reason"].strip():
                    _error("REVIEW_REASON", "Provide your explicit nonblank review reason", f"{collection}[{i}].reason")

    @staticmethod
    def _normalized(payload):
        return {key: deepcopy(payload.get(key) or [])
            for key in REVIEW_COLLECTIONS}

    def _excluded_candidates(self, state):
        """Only recorded explicit decisions count, never an unbound row count."""
        excluded = set()
        for row in self._journal()["calls"].values():
            if row.get("phase") != "complete" and not self._committed(state, row):
                continue
            payload = row["payload"]
            for operation in payload.get("operations") or []:
                if operation.get("op") == "drop":
                    excluded.add(operation["candidate_id"])
            for item in payload.get("exclusions") or []:
                excluded.add(item["candidate_id"])
        return excluded

    def _plan(self, state, payload):
        supplied = self._normalized(payload)
        operations, origins = [], []
        excluded = self._excluded_candidates(state)
        seen = set()

        def candidate(identifier, path, *, excluding=False):
            if not identifier.strip():
                _error("CANDIDATE_UNKNOWN", "Candidate ID cannot be empty or whitespace", path)
            if identifier in seen:
                _error("REVIEW_OVERLAP", "Candidate appears in more than one decision; submit one combined decision", path)
            seen.add(identifier)
            self._candidate(identifier, state, path)
            if not excluding and identifier in excluded:
                _error("CANDIDATE_EXCLUDED", "This candidate was explicitly excluded; submit_review cannot restore it. Inspect the saved decision and use update_workspace for an intentional correction.", path)

        def add(operation, origin):
            operations.append(operation)
            origins.append(origin)

        for i, identifier in enumerate(supplied["selections"]):
            path = f"selections[{i}]"
            candidate(identifier, path)
            add({"op": "select", "candidate_id": identifier}, path)
        for i, row in enumerate(supplied["merges"]):
            path = f"merges[{i}]"
            candidate(row["keep_id"], path + ".keep_id")
            for j, identifier in enumerate(row["other_ids"]):
                candidate(identifier, f"{path}.other_ids[{j}]")
            add({"op": "merge", "keep_id": row["keep_id"], "merge_ids": row["other_ids"],
                "reason": row["reason"], **({"changes": row["changes"]} if "changes" in row else {})}, path)
        for i, row in enumerate(supplied["patches"]):
            path = f"patches[{i}]"
            candidate(row["candidate_id"], path + ".candidate_id")
            add({"op": "patch", **row}, path)
        for i, row in enumerate(supplied["exclusions"]):
            path = f"exclusions[{i}]"
            candidate(row["candidate_id"], path + ".candidate_id", excluding=True)
            add({"op": "drop", **row}, path)

        # The old planner still owns output-row/section/field semantics, source
        # identity and binding preservation. It does not save anything itself.
        try:
            update, response = super()._plan(state, {"operations": operations, "task_reviews": [], "issue_reviews": None})
        except CandidateReviewError as exc:
            match = re.match(r"operations\[(\d+)\](.*)", exc.path)
            if match and int(match[1]) < len(origins):
                exc.path = origins[int(match[1])] + match[2]
            elif exc.path == "operations":
                exc.path = "patches/merges"
            raise

        # Route only exact existing coverage IDs backed by the frozen child
        # inventory. Do not interpret question prose or infer a task from text.
        coverage = {"Q_coverage_" + json_digest(task_id)[:24]: task_id for task_id in self.task_ids}
        questions, outcomes, seen_questions = [], [], set()
        for i, row in enumerate(supplied["reviews"]):
            identifier = row["question_id"]
            path = f"reviews[{i}].question_id"
            if not identifier.strip() or identifier not in state["questions"]:
                _error("QUESTION_UNKNOWN", "Use a nonempty question ID from the current workspace", path)
            if identifier in seen_questions:
                _error("REVIEW_OVERLAP", "This question is reviewed twice in one batch", path)
            seen_questions.add(identifier)
            if identifier.startswith("Q_coverage_") and identifier not in coverage:
                _error("REVIEW_COVERAGE", "Coverage question has no exact task mapping in this run", path)
            current = state["questions"][identifier]
            if current.get("status") != row["status"] or current.get("reason") != row["reason"] or current.get("valid") is not True:
                questions.append({"id": identifier, "status": row["status"], "reason": row["reason"]})
            outcomes.append({"question_id": identifier, "status": row["status"],
                **({"coverage_task_id": coverage[identifier]} if identifier in coverage else {})})
        if questions:
            update["questions"] = questions
        changed = len(update) > 1
        response.update(changed=changed, version=state["version"] + int(changed),
            reviews_recorded=len(outcomes), review_results=outcomes,
            task_reviews_recorded=sum("coverage_task_id" in row for row in outcomes),
            issue_reviews_recorded=sum("coverage_task_id" not in row for row in outcomes))
        # Store the exact receipt in the transaction journal. A replay returns
        # the original save version, even after later batches advance state.
        response["batch_receipt"] = {
            "status": "saved" if changed else "unchanged", "limit": REVIEW_BATCH_LIMIT,
            "action_count": review_action_count(payload), "workspace_version": response["version"],
            "accepted": {"selections": deepcopy(supplied["selections"]),
                "merges": [{"keep_id": r["keep_id"], "other_ids": deepcopy(r["other_ids"])} for r in supplied["merges"]],
                "patches": [r["candidate_id"] for r in supplied["patches"]],
                "exclusions": [r["candidate_id"] for r in supplied["exclusions"]],
                "reviews": [r["question_id"] for r in supplied["reviews"]]},
            "applied": True, "semantic_approval": False,
            "instruction": "These decisions were accepted; do not resubmit them unless intentionally revising. "
                "Continue with the next related unresolved group. Accepted unknown reviews are not resolved coverage. "
                "After edits are complete inspect the current Word; current file receipts decide which views remain."}
        return update, response

    def submit_review(self, selections=None, merges=None, patches=None, exclusions=None, reviews=None):
        # Keep the exact default-expanded submission for trusted replay, rather
        # than replacing its original shape with legacy operations/task IDs.
        payload = {"selections": selections, "merges": merges, "patches": patches,
            "exclusions": exclusions, "reviews": reviews}
        response = self._execute_review(payload, tool_name="submit_review")
        if not response.get("ok"):
            details = response["error"].setdefault("details", {})
            receipt = rejected_review_receipt(payload, version=details.get("current_version"),
                unconfirmed=details.get("applied") is None)
            response["batch_receipt"] = receipt
            # Preserve any specific recovery/identity instructions alongside
            # the atomic-batch explanation, not a misleading "retry one item".
            old_action = details.get("action", "")
            details["action"] = receipt["instruction"]
            if old_action and not old_action.startswith("Correct only this explicit decision"):
                details["action"] += " " + old_action
            details["example"] = {"patches": [{"candidate_id": "<existing candidate ID>",
                "changes": {"value": "<reviewed basic-field value>"}}],
                "reviews": [{"question_id": "<existing question ID>", "status": "unknown",
                    "reason": "<your explicit review reason>"}]}
        return response
