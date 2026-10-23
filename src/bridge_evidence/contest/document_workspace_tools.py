"""Token-bounded views of mutable report state, without audit-history prompts.

Each response belongs to one workspace version. Page positions are stable
within that version; callers must restart a view if the returned version
changes. These tools only present state and authoritative workspace blockers;
they make no semantic decisions and never shorten a record's contents.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Callable
from threading import RLock

from .tong_react_adapter import wrap_tool_response
from .managed_navigation_journal import ManagedNavigationJournal


_FINDINGS = ("unsupported_claims", "citation_gaps", "missing_items", "conflicts", "unresolved_items")
_SUMMARY_TOOLS = ("read_workspace", "update_workspace", "review_draft")
_RESPONSE_TOOLS = (*_SUMMARY_TOOLS, "read_workspace_page", "finish")
_STATE_FIELDS = (
    "version", "facts_version", "facts", "relations", "questions", "draft",
    "draft_fact_ids", "draft_bindings", "draft_sha256", "draft_valid", "reads",
    "image_reads", "exclusions", "review", "reading_progress", "working_notes",
    "delivery_progress", "update_effect",
)
_PROGRESS_VIEWS = (
    "unread_ranges", "unverified_image_ids", "unverified_content", "review_required_evidence_ids",
)
_VIEWS = (
    "facts", "relations", "questions", "unresolved_questions", "unbound_facts", "unbound_relations",
    "draft_basic", "draft_advice", "draft_defects",
    "draft_bindings", "draft_fact_ids", "reads", "image_reads", "exclusions", "review_findings",
    "blockers", "working_notes", "draft_invalidations", *_PROGRESS_VIEWS,
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _current_call_identity() -> dict | None:
    # Imported at call time so only the adapter's trusted dispatch context can
    # supply identity. Models cannot add these fields to a tool invocation.
    from .tong_react_adapter import current_tool_call_identity
    return current_tool_call_identity()


class _ManagedNavigation:
    """Host-owned, identity-bound delivery positions, never audit conclusions."""

    def __init__(self, owner: Any, state: dict | None = None, on_progress: Callable | None = None):
        snapshot = owner.snapshot()
        identity = {"document": snapshot.get("identity", {}),
                    "directory": str(getattr(owner, "directory", ""))}
        if not identity["document"] and not identity["directory"]:
            identity["instance"] = id(owner)
        self.identity = hashlib.sha256(_json(identity).encode()).hexdigest()
        self.lock = RLock()
        self.on_progress = on_progress
        self.state = {"schema_version": 1, "identity": self.identity, "views": {}}
        if state:
            if (state.get("schema_version") != 1 or state.get("identity") != self.identity
                    or not isinstance(state.get("views"), dict)):
                raise ValueError("Navigation state belongs to a different document/workspace")
            for key, position in state["views"].items():
                if (not isinstance(key, str) or not isinstance(position, dict)
                        or any(type(position.get(field)) is not int or position[field] < 0
                               for field in ("version", "cursor", "item_start"))
                        or type(position.get("complete")) is not bool):
                    raise ValueError("Invalid saved navigation position")
            self.state = deepcopy(state)
        # Older checkpoints have positions only. They resume normally; pages
        # emitted before this journal existed are not invented or reconstructed.
        self.state["journal"] = ManagedNavigationJournal(self.state.get("journal")).snapshot()

    def snapshot(self) -> dict:
        return deepcopy(self.state)

    def replay(self, view: str, identity: dict | None) -> dict | None:
        return ManagedNavigationJournal(self.state["journal"]).replay(identity, [self.identity, view])

    def commit(self, view: str, position: dict, *, identity: dict | None = None,
               response: dict | None = None) -> None:
        state = deepcopy(self.state)
        state["views"][view] = deepcopy(position)
        if identity is not None:
            journal = ManagedNavigationJournal(state["journal"])
            journal.record(identity, [self.identity, view], response)
            state["journal"] = journal.snapshot()
        # A failed durable write must not advance the in-memory pointer.
        if self.on_progress is not None:
            self.on_progress(deepcopy(state))
        self.state = state


def _navigation_notice(result: dict, previous_version: int | None) -> dict:
    version = result.get("version")
    changed = previous_version is not None and previous_version != version
    notice = {"managed": True, "version_changed": changed,
              "read_complete": not bool(result.get("continuation")),
              "read_is_not_semantic_review": True}
    if changed:
        notice.update(previous_version=previous_version,
                      action="Version changed; discard old fragments and use this restarted view.")
    return notice


class WorkspaceTools:
    def __init__(self, workspace: Any, *, count_tokens: Callable[[str], int], max_tool_tokens: int = 4096,
                 native_response_budget: bool = False):
        if type(max_tool_tokens) is not int or max_tool_tokens <= 0:
            raise ValueError("max_tool_tokens must be a positive integer")
        if type(native_response_budget) is not bool:
            raise ValueError("native_response_budget must be a boolean")
        self.workspace = workspace
        self.count_tokens = count_tokens
        self.max_tool_tokens = max_tool_tokens
        self.native_response_budget = native_response_budget
        self._managed_navigation = None
        for name in _RESPONSE_TOOLS if native_response_budget else ("read_workspace",):
            if self._tokens(self._budget_error(), name) > max_tool_tokens:
                raise ValueError("max_tool_tokens cannot fit a minimal budget error")

    def configure_managed_navigation(self, state: dict | None = None,
                                     on_progress: Callable[[dict], None] | None = None) -> None:
        """Load durable host positions; use a separate state per workspace."""
        self._managed_navigation = _ManagedNavigation(self.workspace, state, on_progress)

    def managed_navigation_snapshot(self) -> dict:
        if self._managed_navigation is None:
            self.configure_managed_navigation()
        return self._managed_navigation.snapshot()

    def read_page_managed(self, view: str, limit: int = 8, restart: bool = False) -> dict:
        """Continue this view without copying offsets, including large fragments.

        Success only records bytes presented in this response. It neither marks
        a fact verified nor carries fragment offsets across workspace versions.
        """
        name = "read_workspace_page"
        try:
            if view not in _VIEWS or not isinstance(view, str):
                raise ValueError("Unknown workspace view; use a summary counts key")
            if type(limit) is not int or limit <= 0 or type(restart) is not bool:
                raise ValueError("limit must be a positive integer and restart a boolean")
            if self._managed_navigation is None:
                self.configure_managed_navigation()
            navigation = self._managed_navigation
            with navigation.lock:
                identity = _current_call_identity()
                replay = navigation.replay(view, identity)
                if replay is not None:
                    return self._bounded({**replay, "replayed": True}, name)
                previous = navigation.state["views"].get(view)
                previous_version = previous.get("version") if previous else None

                def expose(result):
                    result = deepcopy(result)
                    if result.get("ok") is True:
                        # Reserve this marker in the original page budget; a
                        # replay changes false to the shorter true without
                        # moving positions or relabelling an old version.
                        result["replayed"] = False
                        result["navigation"] = _navigation_notice(result, previous_version)
                        if result.get("continuation"):
                            result["continuation"] = {"tool": name, "version": result["version"],
                                "arguments": {"view": view, "limit": limit, "restart": False}}
                    # An obsolete low-level restart must never ask the model to
                    # copy a cursor or expected_version back into the new tool.
                    if result.get("restart"):
                        result["restart"] = {"tool": name,
                            "arguments": {"view": view, "limit": limit, "restart": True}}
                    return result

                # The low-level implementation sizes its complete response
                # using the actual managed envelope, including navigation.
                inner = WorkspaceTools(self.workspace,
                    count_tokens=lambda text: self._tokens(expose(json.loads(text)), name),
                    max_tool_tokens=self.max_tool_tokens)
                # Keep the caller's read-only state projection (for example,
                # MergeWorkspaceTools attaches each candidate's source text).
                # It must run before sizing/fragmentation, not after admission.
                inner._state = self._state
                for _ in range(3):
                    version = self.workspace.snapshot()["version"]
                    current = (previous if previous and previous_version == version and not restart else
                               {"version": version, "cursor": 0, "item_start": 0, "complete": False})
                    if current["complete"]:
                        output = self._bounded(expose({"ok": True, "status": "already_read",
                            "version": version, "view": view, "items": [], "continuation": None,
                            "action": "This view is already read at this version; set restart=true to reread."}), name)
                        if output.get("ok") is True:
                            navigation.commit(view, current, identity=identity, response=output)
                        return output
                    result = inner.read_page(view, current["cursor"], limit,
                        item_start=current["item_start"], expected_version=version)
                    if result.get("status") == "version_conflict":
                        continue
                    output = self._bounded(expose(result), name)
                    if result.get("ok") is not True or output.get("ok") is not True:
                        return output
                    following = (result.get("continuation") or {}).get("arguments")
                    position = {"version": result["version"], "complete": following is None,
                        "cursor": (following["cursor"] if following else
                                   result.get("end", result.get("item_index", current["cursor"]) + 1)),
                        "item_start": following.get("item_start", 0) if following else 0}
                    navigation.commit(view, position, identity=identity, response=output)
                    return output
                return self._error(RuntimeError("Workspace keeps changing; retry this view"), name)
        except Exception as exc:
            return self._error(exc, name)

    def _tokens(self, value: Any, tool_name: str) -> int:
        if self.native_response_budget:
            value = wrap_tool_response(tool_name, value)
        count = self.count_tokens(_json(value))
        if type(count) is not int or count < 0:
            raise ValueError("count_tokens must return a non-negative integer")
        return count

    @staticmethod
    def _budget_error(**details: Any) -> dict:
        return {"ok": False, "status": "budget_exceeded", "error": {
            "type": "WorkspaceToolBudgetExceeded", "recoverable": True,
            "message": "Response cannot fit; increase the tool budget. No record was truncated.",
        }, **details}

    def _bounded(self, result: dict, tool_name: str) -> dict:
        if self._tokens(result, tool_name) <= self.max_tool_tokens:
            return deepcopy(result)
        return self._budget_error()

    def _error(self, exc: Exception, tool_name: str) -> dict:
        return self._bounded({"ok": False, "status": "error", "error": {
            "type": type(exc).__name__, "message": str(exc), "recoverable": True,
        }}, tool_name)

    def _state(self) -> tuple[dict, dict]:
        # Do not recompute delivery logic here. Read its authoritative result,
        # and guard against mixing it with a concurrently updated snapshot.
        for _ in range(3):
            state = self.workspace.snapshot()
            delivery = self.workspace.delivery_status()
            if state["version"] == delivery["workspace_version"]:
                state["delivery_progress"] = self._delivery_progress(state)
                # An intervening read may advance the workspace. Preserve the
                # exact update_version; never label a read as a new draft save.
                state["update_effect"] = next((deepcopy(event.get("update_effect"))
                    for event in reversed(state.get("history", [])) if event.get("event") == "update"), None)
                return state, delivery
        raise RuntimeError("Workspace changed during reading; retry the current version")

    @staticmethod
    def _binding_closure(state: dict) -> tuple[set[str], set[str]]:
        """Actual output bindings and their valid evidence dependencies.

        Unused draft_fact_ids, stale/nonexistent output paths, questions and
        working notes do not create output bindings. Valid here describes the
        workspace's structural state, never a model's factual correctness.
        """
        draft = state.get("draft") or {}
        paths = {"基本信息." + field for field, value in draft.get("基本信息", {}).items()
                 if value not in (None, "", [], {})}
        for section in ("维修建议", "病害明细"):
            paths.update(f"{section}.{index}" for index in range(len(draft.get(section, []))))
        records = {**state.get("facts", {}), **state.get("relations", {}), **state.get("questions", {})}
        output_records = {**state.get("facts", {}), **state.get("relations", {})}
        direct = {identifier for path, identifiers in state.get("draft_bindings", {}).items() if path in paths
                  for identifier in identifiers
                  if identifier in output_records and output_records[identifier].get("valid") is True}
        reached = set(direct)
        pending = list(direct)
        while pending:
            record = records[pending.pop()]
            # Same dependency contract as DocumentWorkspace._dependencies,
            # including relation endpoints even if depends_on omitted them.
            dependencies = set(record.get("depends_on", []))
            dependencies.update(record[key] for key in ("subject_id", "object_id")
                                if isinstance(record.get(key), str) and record[key])
            for identifier in dependencies - reached:
                if identifier in records and records[identifier].get("valid") is True:
                    reached.add(identifier)
                    pending.append(identifier)
        return direct, reached

    @staticmethod
    def _delivery_progress(state: dict) -> dict:
        direct, supporting = WorkspaceTools._binding_closure(state)
        counts = {}
        for collection in ("facts", "relations"):
            records = state.get(collection, {})
            valid = {identifier for identifier, record in records.items() if record.get("valid") is True}
            invalid = sum(record.get("valid") is False for record in records.values())
            counts[collection] = {"valid": len(valid), "invalid": invalid,
                                  "directly_bound": len(valid & direct), "supporting_output": len(valid & supporting),
                                  "unbound_candidates": len(valid - supporting)}
            unclassified = len(records) - len(valid) - invalid
            if unclassified:
                counts[collection]["unclassified"] = unclassified
        draft = state.get("draft") or {}
        return {"version": state["version"], **counts,
                "row_counts": {section: len(draft.get(section, [])) for section in ("维修建议", "病害明细")},
                "basic_field_count": len(draft.get("基本信息", {})),
                "diagnostic_only": True, "unbound_candidates_are_omissions": False,
                "candidate_views": ["unbound_facts", "unbound_relations"]}

    @staticmethod
    def _draft_invalidations(state: dict) -> tuple[list[dict], bool]:
        """Reconstruct diagnostic confirmations from host-owned journal metadata.

        No values are copied from old drafts. Explicit edits acknowledge only
        their own path; full draft submission acknowledges the whole draft.
        Neither action here changes draft_valid or any acceptance decision.
        """
        history = state.get("history", [])
        complete = (len(history) == state["version"] and
                    all(event.get("version") == index + 1 for index, event in enumerate(history)))
        pending = []
        for event in history:
            if event.get("event") != "update":
                continue
            if "draft_impact" not in event:
                complete = False
            for impact in event.get("draft_impact", []):
                pending.append({**deepcopy(impact), "update_version": event["version"],
                                "original_path": impact["path"]})
            update = event.get("update", {})
            if "draft" in update:
                pending = []
                continue
            for operation in update.get("draft_patch", []):
                op = operation["op"]
                if op == "set_basic":
                    target = "基本信息." + operation["field"]
                    pending = [item for item in pending if item["path"] != target]
                elif op == "replace_row":
                    target = operation["section"] + "." + str(operation["index"])
                    pending = [item for item in pending if item["path"] != target]
                elif op == "bind_output":
                    pending = [item for item in pending if item["path"] != operation["path"]]
                elif op == "remove_row":
                    prefix, index = operation["section"] + ".", operation["index"]
                    retained = []
                    for item in pending:
                        path = item["path"]
                        if isinstance(path, str) and path.startswith(prefix):
                            old_index = int(path[len(prefix):])
                            if old_index == index:
                                continue
                            if old_index > index:
                                item["path"] = prefix + str(old_index - 1)
                        retained.append(item)
                    pending = retained
        return pending, complete

    def _with_draft_feedback(self, payload: dict, state: dict, tool_name: str) -> dict:
        pending, complete = self._draft_invalidations(state)
        feedback = {"version": state["version"], "draft_valid": state.get("draft_valid", False),
                    "pending_count": len(pending), "metadata_complete": complete,
                    "diagnostic_only": True, "details_view": "draft_invalidations",
                    "pagination": {"tool": "read_workspace_page", "view": "draft_invalidations",
                                   **({"restart": False} if self._managed_navigation is not None else {"cursor": 0}),
                                   "limit": 2},
                    "action": "先read_workspace取得当前version，再核对受影响事实和草稿；自行用set_basic/replace_row显式重写并绑定，或提交完整draft/ids/bindings。仅补source_refs也需重新核对。无关patch不清除本提示；它不改变原draft_valid或审核判断。"}
        if not complete:
            feedback["warning"] = "因果metadata或历史版本不完整；计数只是已知部分，不能据0认定草稿已重新核对。"
        result = {**payload, "draft_invalidation_feedback": feedback}
        # A few COMPLETE metadata items help the common case. Large identifiers
        # or cause sets remain in the existing token-bounded paged/fragment view.
        for item in pending[:2]:
            candidate = deepcopy(result)
            candidate["draft_invalidation_feedback"].setdefault("preview", []).append(item)
            if self._tokens(candidate, tool_name) > self.max_tool_tokens:
                break
            result = candidate
        return result

    def invalid_draft_response(self) -> dict:
        """Preserve the original review rejection, add bounded diagnostics only."""
        try:
            state, _delivery = self._state()
            payload = {"ok": False, "error": "Prepare a valid source-bound draft before review",
                       "version": state["version"]}
            return self._bounded(self._with_draft_feedback(payload, state, "review_draft"), "review_draft")
        except Exception as exc:
            return self._error(exc, "review_draft")

    @staticmethod
    def _delivery(delivery: dict) -> dict:
        return {"ready": delivery["ready"], "workspace_version": delivery["workspace_version"],
                "draft_sha256": delivery.get("draft_sha256"), "blocker_count": len(delivery["blockers"]),
                "blockers_view": "blockers"}

    def bounded_delivery(self) -> dict:
        """Standalone status, or a complete ``finish`` tool payload.

        Native mode counts the finish envelope. If a caller embeds/adds fields
        to this status, that different final payload needs its own admission.
        """
        try:
            return self._bounded(self._delivery(self.workspace.delivery_status()), "finish")
        except Exception as exc:
            return self._error(exc, "finish")

    @staticmethod
    def _items(state: dict, delivery: dict, view: str) -> list:
        if view == "unresolved_questions":
            return [deepcopy(record) for _, record in sorted(state.get("questions", {}).items())
                    if record.get("valid") is not True or record.get("status") != "resolved"]
        if view == "draft_invalidations":
            return WorkspaceTools._draft_invalidations(state)[0]
        if view == "working_notes":
            notes = state.get("working_notes", {})
            # One logical memo. Transport fragments, when needed, are explicitly
            # marked JSON text and never exposed as complete note records.
            return [deepcopy(notes)] if notes else []
        if view in {"facts", "relations", "questions"}:
            records = state.get(view, {})
            return [deepcopy(records[key]) for key in sorted(records)]
        if view in {"unbound_facts", "unbound_relations"}:
            records = state.get("facts" if view == "unbound_facts" else "relations", {})
            supporting = WorkspaceTools._binding_closure(state)[1]
            return [deepcopy(records[key]) for key in sorted(records)
                    if records[key].get("valid") is True and key not in supporting]
        draft = state.get("draft") or {}
        if view == "draft_basic":
            return [{"field": key, "value": value} for key, value in sorted(draft.get("基本信息", {}).items())]
        if view in {"draft_advice", "draft_defects"}:
            return deepcopy(draft.get("维修建议" if view == "draft_advice" else "病害明细", []))
        if view == "draft_bindings":
            return [{"path": key, "record_ids": value} for key, value in sorted(state.get(view, {}).items())]
        if view == "reads":
            return [{"evidence_id": key, "ranges": value} for key, value in sorted(state.get(view, {}).items())]
        if view in {"draft_fact_ids", "image_reads", "exclusions"}:
            return deepcopy(state.get(view, []))
        if view == "review_findings":
            review = state.get("review") or {}
            return [{"category": category, "index": index, "finding": finding}
                    for category in _FINDINGS for index, finding in enumerate(review.get(category, []))]
        if view == "blockers":
            return deepcopy(delivery["blockers"])
        if view in _PROGRESS_VIEWS:
            return deepcopy(state.get("reading_progress", {}).get(view, []))
        raise ValueError(f"Unknown workspace view: {view}. Allowed views: {', '.join(_VIEWS)}")

    def summary(self, *, tool_name: str = "read_workspace") -> dict:
        """Full selected state if it fits, otherwise explicit paging.

        The host selects a fixed caller name for summaries returned by updates
        and review. This option is not a model-controlled SDK argument. Native
        budgeting requires the adapter's same tokenizer and limit; the returned
        payload remains unwrapped for direct consumers.
        """
        if type(tool_name) is not str or tool_name not in _SUMMARY_TOOLS:
            raise ValueError("Summary caller must be read_workspace, update_workspace or review_draft")
        try:
            state, delivery = self._state()
            result = {key: deepcopy(state[key]) for key in _STATE_FIELDS if key in state}
            result.update(ok=True, mode="full", ready=delivery["ready"], blockers=deepcopy(delivery["blockers"]),
                          delivery=self._delivery(delivery), audit_fields_omitted=["identity", "history"])
            unknown = sorted(set(state) - set(_STATE_FIELDS) - {"identity", "history", "schema_version", "workspace_version"})
            if unknown:
                result["unexposed_fields"] = unknown
            result = self._with_draft_feedback(result, state, tool_name)
            if self._tokens(result, tool_name) <= self.max_tool_tokens:
                return result
            counts = {view: len(self._items(state, delivery, view)) for view in _VIEWS}
            blockers = delivery["blockers"]
            # Record-specific blocker text remains available in the blockers
            # view. Grouping only groups existing labels; it creates no verdict.
            blocker_counts: dict[str, int] = {}
            for blocker in blockers:
                kind = blocker.split(":", 1)[0]
                blocker_counts[kind] = blocker_counts.get(kind, 0) + 1
            result = {"ok": True, "mode": "paged", "version": state["version"],
                      "facts_version": state["facts_version"], "draft_sha256": state.get("draft_sha256"),
                      "draft_valid": state.get("draft_valid", False), "ready": delivery["ready"],
                      "delivery_progress": deepcopy(state["delivery_progress"]),
                      "update_effect": deepcopy(state["update_effect"]),
                      "delivery": self._delivery(delivery),
                      "blockers": {"total": len(blockers), "by_kind": blocker_counts, "view": "blockers"},
                      "counts": counts, "pagination": {
                          "tool": "read_workspace_page", "view": "Use a key from counts",
                          **({"restart": False} if self._managed_navigation is not None else {"cursor": 0}),
                          "limit": 20,
                          "consistency": ("Program resumes the selected view; a changed version restarts it explicitly"
                                          if self._managed_navigation is not None else
                                          "Restart the view if a page returns a different version"),
                      }, "audit_fields_omitted": ["identity", "history"],
                      "reading_progress": {
                          "text_coverage_complete": state.get("reading_progress", {}).get("text_coverage_complete"),
                          "counts": {key: len(value) for key, value in state.get("reading_progress", {}).items()
                                     if isinstance(value, list)},
                      }}
            notes = state.get("working_notes", {})
            result["working_notes_ref"] = {
                "present": bool(notes), "sha256": hashlib.sha256(_json(notes).encode("utf-8")).hexdigest(),
                "view": "working_notes", "limit": 1, "is_evidence": False,
                **({"restart": False} if self._managed_navigation is not None else {"cursor": 0}),
            }
            if unknown:
                result["unexposed_fields"] = unknown
            review = state.get("review")
            result["review"] = None
            if review is not None:
                result["review"] = {key: review[key] for key in (
                    "passed", "draft_sha256", "facts_version", "recorded_version", "protocol",
                ) if key in review}
                result["review"].update(issue_counts={key: len(review.get(key, [])) for key in _FINDINGS},
                                        findings_view="review_findings")
            result = self._with_draft_feedback(result, state, tool_name)
            with_notes = {**result, "working_notes": deepcopy(notes)}
            if notes and self._tokens(with_notes, tool_name) <= self.max_tool_tokens:
                result = with_notes
            # Preserve numeric progress and diagnostics under small historical
            # budgets. These optional instructions duplicate counts/view
            # navigation; never drop evidence records or invent completeness.
            if self._tokens(result, tool_name) > self.max_tool_tokens:
                result["delivery_progress"].pop("candidate_views", None)
            if self._tokens(result, tool_name) > self.max_tool_tokens and result["update_effect"] is None:
                result.pop("update_effect")
            if self._tokens(result, tool_name) > self.max_tool_tokens:
                result["draft_invalidation_feedback"].pop("action", None)
            if self._tokens(result, tool_name) > self.max_tool_tokens:
                # The view names and positions already live in counts and the
                # main pagination block. Keep all progress/validity counters,
                # source blockers and note hashes; remove only duplicate help.
                result["pagination"].pop("consistency", None)
                result["pagination"].pop("view", None)
                result["draft_invalidation_feedback"].pop("pagination", None)
                result.pop("audit_fields_omitted", None)
            # Compact diagnostics may now leave room for the entire memo.
            # Never truncate it to make an inline representation fit.
            with_notes = {**result, "working_notes": deepcopy(notes)}
            if notes and self._tokens(with_notes, tool_name) <= self.max_tool_tokens:
                result = with_notes
            return self._bounded(result, tool_name)
        except Exception as exc:
            return self._error(exc, tool_name)

    def _item_fragment(self, item: Any, *, view: str, cursor: int, limit: int,
                       version: int, total: int, item_start: int) -> dict:
        text = _json(item)
        if item_start >= len(text):
            raise ValueError("item_start exceeds the canonical JSON character interval")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

        def fragment(end: int) -> dict:
            following = ({"view": view, "cursor": cursor, "limit": limit,
                          "item_start": end, "expected_version": version}
                         if end < len(text) else
                         {"view": view, "cursor": cursor + 1, "limit": limit,
                          "item_start": 0, "expected_version": version}
                         if cursor + 1 < total else None)
            return {"ok": True, "mode": "canonical_json_fragment", "version": version,
                    "view": view, "item_index": cursor, "item_sha256": digest,
                    "item_start": item_start, "item_end": end, "item_total_chars": len(text),
                    "fragment": text[item_start:end], "fragment_is_complete_record": False,
                    "continuation": ({"tool": "read_workspace_page", "version": version,
                                      "arguments": following} if following else None)}

        # Count the same native envelope as dispatch, including JSON escaping of
        # this JSON-text fragment. No assumption that characters equal tokens.
        best = None
        low, high = item_start + 1, len(text)
        while low <= high:
            middle = (low + high) // 2
            candidate = fragment(middle)
            if self._tokens(candidate, "read_workspace_page") <= self.max_tool_tokens:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        if best is None:
            # Even a one-character protocol may not fit a very small budget.
            # Retrying smaller page sizes cannot fix that; preserve the item.
            return self._bounded(self._budget_error(
                version=version, view=view, item_index=cursor, item_start=item_start,
                required_tokens=self._tokens(fragment(item_start + 1), "read_workspace_page"),
                max_tool_tokens=self.max_tool_tokens,
            ), "read_workspace_page")
        return self._bounded(best, "read_workspace_page")

    def read_page(self, view: str, cursor: int, limit: int, item_start: int = 0,
                  expected_version: int | None = None) -> dict:
        """Read whole records, or lossless version-bound canonical JSON fragments.

        Fragment offsets count Unicode characters in sorted, compact JSON, not
        source evidence or JSON bytes. Concatenate every fragment for one item
        and check its SHA256 before parsing; a fragment is not a fact or memo.
        """
        tool_name = "read_workspace_page"
        try:
            if not isinstance(view, str) or view not in _VIEWS:
                raise ValueError("Unknown workspace view; use a summary counts key")
            if type(cursor) is not int or cursor < 0 or type(limit) is not int or limit <= 0:
                raise ValueError("cursor must be an integer >= 0 and limit an integer > 0; booleans are not accepted")
            if type(item_start) is not int or item_start < 0:
                raise ValueError("item_start must be an integer >= 0; booleans are not accepted")
            if expected_version is not None and (type(expected_version) is not int or expected_version < 0):
                raise ValueError("expected_version must be an integer >= 0 or null")
            if item_start and expected_version is None:
                raise ValueError("Fragment continuation requires expected_version; restart the item at item_start=0")
            state, delivery = self._state()
            items = self._items(state, delivery, view)
            total, version = len(items), state["version"]
            if expected_version is not None and expected_version != version:
                return self._bounded({"ok": False, "status": "version_conflict", "version": version,
                    "expected_version": expected_version, "error": {"type": "ValueError", "recoverable": True,
                    "message": "Workspace version changed; discard collected fragments and restart the view"},
                    "restart": {"tool": "read_workspace_page", "arguments": {
                        "view": view, "cursor": 0, "limit": limit, "item_start": 0, "expected_version": version}}}, tool_name)
            if cursor > total:
                raise ValueError(f"cursor exceeds the view length ({total})")
            if item_start:
                if cursor == total:
                    raise ValueError("Fragment continuation requires an existing item")
                return self._item_fragment(items[cursor], view=view, cursor=cursor, limit=limit,
                                           version=version, total=total, item_start=item_start)

            def page(end: int) -> dict:
                return {"ok": True, "version": version, "view": view,
                        "items": items[cursor:end], "start": cursor, "end": end, "total": total,
                        "continuation": ({"tool": "read_workspace_page", "version": version,
                                          "arguments": {"view": view, "cursor": end, "limit": limit,
                                                        **({"expected_version": version} if expected_version is not None else {})}}
                                         if end < total else None)}

            result = page(cursor)
            for end in range(cursor + 1, min(total, cursor + limit) + 1):
                candidate = page(end)
                required = self._tokens(candidate, tool_name)
                if required > self.max_tool_tokens:
                    if end == cursor + 1:
                        return self._item_fragment(items[cursor], view=view, cursor=cursor, limit=limit,
                                                   version=version, total=total, item_start=0)
                    break
                result = candidate
            return self._bounded(result, tool_name)
        except Exception as exc:
            return self._error(exc, tool_name)


class ManagedEntriesPager:
    """Continue one child's saved entries, with no model-supplied cursor.

    EntryStore keeps complete entries atomic; an entry too large for the tool
    budget produces an explicit failure with no progress. Workspace record
    fragmentation remains in WorkspaceTools and is not silently applied here.
    """

    def __init__(self, store: Any, *, count_tokens: Callable[[str], int],
                 max_tool_tokens: int = 4096, native_response_budget: bool = True,
                 state: dict | None = None, on_progress: Callable[[dict], None] | None = None):
        if type(max_tool_tokens) is not int or max_tool_tokens <= 0:
            raise ValueError("max_tool_tokens must be a positive integer")
        self.store, self.count_tokens = store, count_tokens
        self.max_tool_tokens, self.native_response_budget = max_tool_tokens, native_response_budget
        self.navigation = _ManagedNavigation(store, state, on_progress)

    def managed_navigation_snapshot(self) -> dict:
        return self.navigation.snapshot()

    def _fits(self, value: dict) -> bool:
        if self.native_response_budget:
            value = wrap_tool_response("read_entries", value)
        tokens = self.count_tokens(_json(value))
        if type(tokens) is not int or tokens < 0:
            raise ValueError("count_tokens must return a non-negative integer")
        return tokens <= self.max_tool_tokens

    def read_entries(self, limit: int = 8, restart: bool = False) -> dict:
        try:
            if type(limit) is not int or limit <= 0 or type(restart) is not bool:
                raise ValueError("limit must be a positive integer and restart a boolean")
            with self.navigation.lock:
                identity = _current_call_identity()
                replay = self.navigation.replay("entries", identity)
                if replay is not None:
                    result = {**replay, "replayed": True}
                    return result if self._fits(result) else WorkspaceTools._budget_error()
                previous = self.navigation.state["views"].get("entries")
                previous_version = previous.get("version") if previous else None
                for _ in range(3):
                    state = self.store.snapshot()
                    version = state["version"]
                    current = (previous if previous and previous_version == version and not restart else
                               {"version": version, "cursor": 0, "item_start": 0, "complete": False})
                    if current["complete"]:
                        result = {"ok": True, "status": "already_read", "version": version,
                            "entries": [], "counts": state["counts"], "next_cursor": None,
                            "replayed": False,
                            "delivery_complete": False, "continuation": None,
                            "action": "Saved entries already read at this version; set restart=true to reread."}
                        result["navigation"] = _navigation_notice(result, previous_version)
                        if not self._fits(result):
                            return WorkspaceTools._budget_error()
                        self.navigation.commit("entries", current, identity=identity, response=result)
                        return result
                    changed = False
                    remaining = max(1, state["counts"]["total"] - current["cursor"])
                    for page_limit in range(min(limit, remaining), 0, -1):
                        result = self.store.read_entries(cursor=current["cursor"], limit=page_limit)
                        if result.get("ok") is not True:
                            return result if self._fits(result) else WorkspaceTools._budget_error()
                        if result["version"] != version:
                            changed = True
                            break
                        following = result.get("next_cursor")
                        result["replayed"] = False
                        result["continuation"] = ({"tool": "read_entries", "version": version,
                            "arguments": {"limit": limit, "restart": False}} if following is not None else None)
                        result["navigation"] = _navigation_notice(result, previous_version)
                        if not self._fits(result):
                            continue
                        self.navigation.commit("entries", {"version": version,
                            "cursor": following if following is not None else current["cursor"] + len(result["entries"]),
                            "item_start": 0, "complete": following is None}, identity=identity, response=result)
                        return result
                    if not changed:
                        return WorkspaceTools._budget_error()
                raise RuntimeError("Saved entries keep changing; retry reading entries")
        except Exception as exc:
            result = {"ok": False, "status": "error", "error": {"type": type(exc).__name__,
                "message": str(exc), "recoverable": True, **getattr(exc, "details", {})}}
            return result if self._fits(result) else WorkspaceTools._budget_error()
