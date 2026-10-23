"""Model-directed local delivery edits using stable child candidate IDs.

This layer translates explicit decisions; it does not decide that candidates
are duplicates, choose facts, or declare semantic review complete. Original
facts stay unchanged. Every merged output retains all contributing bindings.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from .docx_writer import ADVICE_FIELDS, BASIC_FIELDS, DEFECT_FIELDS
from .document_delivery_validation import validate_delivery_data
from .document_workspace import WorkspaceError
from .run_state import atomic_json, json_digest


_DETAILS = ("维修建议", "病害明细")
_BASIC = {field for field, _ in BASIC_FIELDS}
_FIELDS = {"维修建议": set(ADVICE_FIELDS), "病害明细": set(DEFECT_FIELDS)}


def review_candidates_schema():
    identifier = {"type": "string", "minLength": 1}
    reason = {"type": "string", "minLength": 1,
              "description": "Your explicit review decision and its evidence-based reason; not a program verdict."}
    changes = {"type": "object", "minProperties": 1, "additionalProperties": False,
        "description": "Basic candidate: {value: new scalar}. Detail candidate: only changed delivery-column values. No fact IDs, source ranges, row indexes or versions.",
        "properties": {field: {"type": ["string", "number", "null"]}
                       for field in sorted({"value", *ADVICE_FIELDS, *DEFECT_FIELDS})}}
    variants = []
    for op, fields, required in (
        ("select", {"candidate_id": identifier}, ["candidate_id"]),
        ("patch", {"candidate_id": identifier, "changes": changes}, ["candidate_id", "changes"]),
        ("drop", {"candidate_id": identifier, "reason": reason}, ["candidate_id", "reason"]),
        ("merge", {"keep_id": identifier, "merge_ids": {"type": "array", "minItems": 1,
            "items": identifier, "uniqueItems": True}, "changes": changes, "reason": reason},
            ["keep_id", "merge_ids", "reason"]),
    ):
        variants.append({"type": "object", "additionalProperties": False,
            "required": ["op", *required], "properties": {"op": {"type": "string", "enum": [op]}, **fields}})
    return {"type": "object", "additionalProperties": False,
        "required": ["operations", "task_reviews"], "properties": {
            "operations": {"type": "array", "items": {"oneOf": variants}},
            "task_reviews": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                "required": ["task_id", "reason"], "properties": {"task_id": identifier, "reason": reason}}},
            "issue_reviews": {"type": ["array", "null"], "default": None, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["question_id", "status", "reason"], "properties": {
                    "question_id": identifier, "status": {"enum": ["resolved", "unknown"]}, "reason": reason}}}}}


class CandidateReviewError(ValueError):
    def __init__(self, code, message, path, *, action="Correct only this explicit decision and retry; nothing in the batch was saved."):
        super().__init__(message)
        self.code, self.path, self.action = code, path, action


def _fail(code, message, path, **kwargs):
    raise CandidateReviewError(code, message, path, **kwargs)


def _unique(values):
    return list(dict.fromkeys(values))


def _current_invocation():
    from .tong_react_adapter import current_tool_call_identity
    return current_tool_call_identity()


class CandidateReviewTools:
    def __init__(self, workspace, handoff_evidence, child_summary):
        self.workspace = workspace
        self.handoff = deepcopy(handoff_evidence)
        self.children = deepcopy(child_summary)
        self.task_ids = {row["task_id"] for row in self.children}
        if len(self.task_ids) != len(self.children):
            raise ValueError("Child summary contains ambiguous duplicate task IDs")
        self.directory = Path(workspace.directory)
        self.path = self.directory / "candidate-review.json"
        self.identity = {"workspace": workspace.snapshot()["identity"],
            "handoff_sha256": json_digest(self.handoff), "child_summary_sha256": json_digest(self.children)}

    @contextmanager
    def _lock(self):
        with (self.directory / "candidate-review.lock").open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _journal(self):
        if not self.path.exists():
            return {"schema_version": 1, "identity": deepcopy(self.identity), "calls": {}}
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1 or state.get("identity") != self.identity or not isinstance(state.get("calls"), dict):
            _fail("REVIEW_IDENTITY", "Review journal belongs to another workspace or candidate inventory", "journal",
                  action="Stop and inspect the run identity; do not overwrite the journal.")
        return state

    @staticmethod
    def _schema_check(payload):
        try:
            json_digest(payload)  # Reject NaN/Infinity/non-JSON before any writes.
        except (TypeError, ValueError):
            _fail("REVIEW_ARGUMENTS", "Arguments must be finite JSON", "arguments")
        error = next(Draft202012Validator(review_candidates_schema()).iter_errors(payload), None)
        if error:
            path = "".join(f"[{p}]" if isinstance(p, int) else ("." if i else "") + p
                           for i, p in enumerate(error.absolute_path)) or "arguments"
            _fail("REVIEW_ARGUMENTS", "Arguments do not match the local review tool schema", path)
        for collection in ("task_reviews", "issue_reviews"):
            for i, row in enumerate(payload.get(collection) or []):
                if not row["reason"].strip():
                    _fail("REVIEW_REASON", "Review reason cannot be blank", f"{collection}[{i}].reason")
        for i, op in enumerate(payload["operations"]):
            if "reason" in op and not op["reason"].strip():
                _fail("REVIEW_REASON", "Merge/drop requires an explicit model reason", f"operations[{i}].reason")

    def _candidate(self, identifier, state, path):
        evidence, fact = self.handoff.get(identifier), state["facts"].get(identifier)
        if not evidence or not fact:
            _fail("CANDIDATE_UNKNOWN", "Use an existing child candidate ID from the current inventory", path)
        if fact.get("valid") is not True:
            _fail("CANDIDATE_INVALID", "Candidate evidence is invalidated; use update_workspace to repair its evidence first", path)
        if (fact.get("value") != evidence["candidate_value"] or fact.get("source_refs") != evidence["source_refs"]):
            _fail("CANDIDATE_CHANGED", "Candidate facts differ from the original handoff; do not silently reuse stale evidence", path,
                  action="Inspect the changed fact and use update_workspace for this non-original candidate.")
        section, field = evidence["section"], evidence.get("field")
        if section not in (*_DETAILS, "基本信息") or (section == "基本信息" and field not in _BASIC):
            _fail("CANDIDATE_SECTION", "Candidate has no unambiguous exportable section/field", path)
        return evidence

    def _plan(self, state, payload):
        draft = deepcopy(state.get("draft") or {"基本信息": {}, "维修建议": [], "病害明细": []})
        original_draft = deepcopy(draft)
        bindings = deepcopy(state.get("draft_bindings", {}))
        rows = {section: [{"row": value, "bindings": bindings.get(f"{section}.{i}", [])}
                         for i, value in enumerate(draft[section])] for section in _DETAILS}
        basic_bindings = {field: bindings["基本信息." + field] for field in draft["基本信息"]
                          if "基本信息." + field in bindings}
        seen_ids, seen_targets, outcomes = set(), set(), []

        def locate(identifier, section, path):
            matches = [row for row in rows[section] if identifier in row["bindings"]]
            if len(matches) > 1:
                _fail("CANDIDATE_AMBIGUOUS", "Candidate is bound to multiple output rows; use update_workspace", path)
            return matches[0] if matches else None

        def owner(row, section):
            return next((identifier for identifier in row["bindings"] if self.handoff.get(identifier, {}).get("section") == section), None)

        def reserve(target, path):
            if target in seen_targets:
                _fail("REVIEW_OVERLAP", "Two operations edit the same output field/row; combine them into one decision", path)
            seen_targets.add(target)

        def change(value, changes, section, path):
            if section == "基本信息":
                if set(changes) != {"value"}:
                    _fail("REVIEW_FIELDS", "Basic candidate changes must contain only value", path)
                return deepcopy(changes["value"])
            if set(changes) - _FIELDS[section]:
                _fail("REVIEW_FIELDS", "Changes include columns from another section or unsupported columns", path)
            return {**deepcopy(value), **deepcopy(changes)}

        for index, operation in enumerate(payload["operations"]):
            path, kind = f"operations[{index}]", operation["op"]
            identifier = operation.get("candidate_id", operation.get("keep_id"))
            ids = [identifier, *operation.get("merge_ids", [])]
            if len(set(ids)) != len(ids) or seen_ids.intersection(ids):
                _fail("REVIEW_OVERLAP", "A candidate cannot be edited twice in one batch", path)
            seen_ids.update(ids)
            evidence = [self._candidate(cid, state, path) for cid in ids]
            section, field = evidence[0]["section"], evidence[0].get("field")
            if kind == "select" and section != "基本信息":
                _fail("REVIEW_SECTION", "select applies only to a basic-field candidate", path)
            if kind == "merge" and (section not in _DETAILS or any(e["section"] != section for e in evidence)):
                _fail("REVIEW_SECTION", "Merge only explicitly selected detail candidates from the same section", path)
            if section == "基本信息":
                reserve((section, field), path)
                if kind == "drop":
                    bound = basic_bindings.get(field, [])
                    if identifier in bound:
                        other_candidates = [cid for cid in bound if cid != identifier and cid in self.handoff]
                        if other_candidates:
                            _fail("CANDIDATE_AMBIGUOUS", "Basic field has multiple candidate bindings; select the intended candidate or use update_workspace", path)
                        draft[section].pop(field, None); basic_bindings.pop(field, None)
                else:
                    draft[section][field] = (change(evidence[0]["candidate_value"], operation["changes"], section, path + ".changes")
                                             if kind == "patch" else deepcopy(evidence[0]["candidate_value"]))
                    basic_bindings[field] = [identifier]
            else:
                targets = [locate(cid, section, path) for cid in ids]
                if kind == "drop" and targets[0] is None:
                    outcomes.append({"op": kind, "candidate_ids": ids, "status": "already_unbound"})
                    continue
                if any(row is None for row in targets):
                    _fail("CANDIDATE_UNBOUND", "Candidate has no current output row; do not recreate a removed candidate implicitly", path)
                keep = targets[0]
                if owner(keep, section) != identifier:
                    _fail("CANDIDATE_MERGED_MEMBER", "Use the retained candidate ID to edit an already merged row", path)
                for cid, row in zip(ids, targets):
                    if row is not keep and owner(row, section) != cid:
                        _fail("CANDIDATE_MERGED_MEMBER", "A merge member is part of another retained row; select that row's retained ID", path)
                distinct = []
                for row in targets:
                    if not any(row is prior for prior in distinct):
                        distinct.append(row)
                for row in distinct:
                    reserve((section, id(row)), path)
                if kind == "drop":
                    rows[section] = [row for row in rows[section] if row is not keep]
                elif kind == "patch":
                    keep["row"] = change(keep["row"], operation["changes"], section, path + ".changes")
                elif kind == "merge":
                    keep["bindings"] = _unique(cid for row in distinct for cid in row["bindings"])
                    if "changes" in operation:
                        keep["row"] = change(keep["row"], operation["changes"], section, path + ".changes")
                    rows[section] = [row for row in rows[section] if row is keep or not any(row is gone for gone in distinct)]
            outcomes.append({"op": kind, "candidate_ids": ids, "status": "applied_as_requested"})

        draft.update({section: [row["row"] for row in rows[section]] for section in _DETAILS})
        bindings = {"基本信息." + field: ids for field, ids in basic_bindings.items()}
        bindings.update({f"{section}.{i}": row["bindings"] for section in _DETAILS
                         for i, row in enumerate(rows[section])})
        try:
            validate_delivery_data(draft)
        except ValueError:
            _fail("REVIEW_VALUE", "Requested changes contain an unexportable value", "operations")
        questions, seen_questions = [], set()

        def question(identifier, status, reason, path):
            if identifier in seen_questions:
                _fail("REVIEW_OVERLAP", "Question was reviewed twice in one batch", path)
            seen_questions.add(identifier)
            current = state["questions"].get(identifier)
            if current is None:
                _fail("QUESTION_UNKNOWN", "Review target does not exist in this workspace", path)
            if current.get("status") != status or current.get("reason") != reason or current.get("valid") is not True:
                questions.append({"id": identifier, "status": status, "reason": reason})

        for index, review in enumerate(payload["task_reviews"]):
            if review["task_id"] not in self.task_ids:
                _fail("TASK_UNKNOWN", "Use a task ID from this report's child summary", f"task_reviews[{index}].task_id")
            question("Q_coverage_" + json_digest(review["task_id"])[:24], "resolved", review["reason"], f"task_reviews[{index}]")
        for index, review in enumerate(payload.get("issue_reviews") or []):
            if review["question_id"].startswith("Q_coverage_"):
                _fail("REVIEW_COVERAGE", "Use task_reviews for an explicit task-coverage decision", f"issue_reviews[{index}].question_id")
            question(review["question_id"], review["status"], review["reason"], f"issue_reviews[{index}]")
        update = {"base_version": state["version"]}
        if draft != original_draft or bindings != state.get("draft_bindings", {}):
            update.update(draft=draft, draft_bindings=bindings,
                          draft_fact_ids=sorted({cid for ids in bindings.values() for cid in ids}))
        if questions:
            update["questions"] = questions
        changed = len(update) > 1
        response = {"ok": True, "changed": changed, "version": state["version"] + int(changed),
            "operations": outcomes, "task_reviews_recorded": len(payload["task_reviews"]),
            "issue_reviews_recorded": len(payload.get("issue_reviews") or []),
            "draft_counts": {"基本信息": len(draft["基本信息"]), **{section: len(draft[section]) for section in _DETAILS}},
            "original_candidate_facts_preserved": True, "file_inspection_still_required": True,
            "semantic_approval": False, "replayed": False}
        return update, response

    @staticmethod
    def _committed(state, row):
        return any(event.get("event") == "update" and event.get("version") == row["base_version"] + 1
                   and json_digest(event.get("update")) == row["update_sha256"] for event in state.get("history", []))

    def review_candidates(self, operations, task_reviews, issue_reviews=None):
        payload = {"operations": operations, "task_reviews": task_reviews, "issue_reviews": issue_reviews}
        return self._execute_review(payload, tool_name="review_candidates")

    def _execute_review(self, payload, *, tool_name):
        """Reuse the atomic journal for an explicitly named native entrypoint.

        Subclasses may translate another schema inside _plan, under this same
        lock. The invocation and original arguments are never rewritten.
        """
        version = None
        try:
            self._schema_check(payload)
            with self._lock():
                state, journal = self.workspace.snapshot(), self._journal()
                version = state["version"]
                invocation = _current_invocation()
                if invocation is not None and (invocation.get("tool_name") != tool_name
                        or not all(invocation.get(k) for k in ("run_id", "session_id", "generation_call_id", "native_tool_call_id", "args_sha256"))):
                    _fail("REVIEW_INVOCATION", "Trusted tool-call identity is incomplete", "invocation")
                call_key = ("native_" + json_digest([invocation["run_id"], invocation["session_id"], invocation["native_tool_call_id"]])
                            if invocation else ("direct_" if tool_name == "review_candidates" else f"direct_{tool_name}_")
                            + json_digest([payload, version]))
                previous = journal["calls"].get(call_key)
                if previous:
                    previous_tool = previous.get("tool_name", (previous.get("invocation") or {}).get("tool_name", "review_candidates"))
                    if previous_tool != tool_name or previous["invocation"] != invocation or previous["payload"] != payload:
                        _fail("REVIEW_REPLAY_CONFLICT", "A replay changed the original call arguments/identity", "invocation")
                    if previous["phase"] == "complete" or self._committed(state, previous):
                        previous["phase"] = "complete"
                        atomic_json(self.path, journal)
                        return {**deepcopy(previous["response"]), "replayed": True}
                    if version != previous["base_version"]:
                        _fail("REVIEW_SAVE_UNCONFIRMED", "Prior review outcome cannot be proven from the workspace history", "journal",
                              action="Inspect the saved workspace and journal; do not apply the same edits again.")
                update, response = self._plan(state, payload)
                row = {"phase": "pending", "invocation": invocation, "payload": deepcopy(payload),
                       "tool_name": tool_name, "submitted_args_sha256": json_digest(payload),
                       "base_version": version, "update_sha256": json_digest(update), "response": response}
                journal["calls"][call_key] = row
                # Write intent before the only atomic workspace mutation. A
                # crash after commit is recovered by its exact history event.
                atomic_json(self.path, journal)
                if response["changed"]:
                    self.workspace.apply_update(update)
                row["phase"] = "complete"
                atomic_json(self.path, journal)
                return deepcopy(response)
        except CandidateReviewError as exc:
            return {"ok": False, "error": {"code": exc.code, "message": str(exc), "details": {
                "applied": None if exc.code == "REVIEW_SAVE_UNCONFIRMED" else False,
                "current_version": version, "path": exc.path, "action": exc.action,
                "example": {"operations": [{"op": "patch", "candidate_id": "<existing child candidate ID>",
                    "changes": {"value": "<explicit reviewed basic value>"}}], "task_reviews": []}}}}
        except WorkspaceError as exc:
            # Workspace validation and optimistic-version checks run before
            # its atomic save. Never overwrite a concurrent valid update.
            return {"ok": False, "error": {"code": "REVIEW_WORKSPACE_REJECTED", "message": str(exc),
                "details": {"applied": False, "current_version": version, "path": "workspace",
                    "error_type": type(exc).__name__,
                    "action": "Read the current workspace, correct the indicated conflict, and submit a new review invocation."}}}
        except Exception as exc:
            # Do not claim a failed post-rename audit write proves no save.
            return {"ok": False, "error": {"code": "REVIEW_SAVE_UNCONFIRMED", "message": "Review transaction needs state inspection",
                "details": {"applied": None, "current_version": version, "path": "workspace",
                    "error_type": type(exc).__name__, "action": "Inspect current workspace; replay the exact original invocation to recover a proven commit, or use update_workspace after inspection."}}}
