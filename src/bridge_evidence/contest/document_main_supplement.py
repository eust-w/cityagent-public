"""Small, provenance-checked additions to the main Agent's current draft.

The model supplies the business values and evidence. The host supplies IDs,
bindings and the atomic transaction. This never changes the frozen child
candidate inventory or decides that source text entails an entry.
"""
from copy import deepcopy

from jsonschema import Draft202012Validator

from .document_candidate_review import CandidateReviewError, CandidateReviewTools
from .document_dedicated_tools import dedicated_schema, example, normalize_fields
from .document_entries import _scalar
from .document_main_review_tools import derived_advice_statistics
from .document_material_refs import MaterialRefNormalizationError, normalize_material_refs
from .document_workspace import WorkspaceError, _covers
from .run_state import json_digest


SUPPLEMENT_SECTIONS = {"save_defects": "病害明细", "save_recommendations": "维修建议"}
SUPPLEMENT_LIMIT = 5


def supplement_schema(name):
    if name not in SUPPLEMENT_SECTIONS:
        raise ValueError("Unknown main supplement tool")
    schema = dedicated_schema(name)
    entries = schema["properties"]["entries"]
    entries["maxItems"] = SUPPLEMENT_LIMIT
    entries["description"] = (
        "Only new evidence-backed entries, at most 5 per batch. Copy source_refs "
        "from supplied material or actual reading tools. Do not supply IDs, versions "
        "or previously saved entries; the host handles these."
    )
    entries["items"]["properties"].pop("id", None)
    return schema


SUPPLEMENT_SCHEMAS = {name: supplement_schema(name) for name in SUPPLEMENT_SECTIONS}
SUPPLEMENT_DESCRIPTIONS = {
    "save_defects": "主Agent补存原文有依据但子Agent遗漏的病害。只交新增条目，每批最多5条；程序生成编号并与事实、证据一起原子保存，不要重交已保存病害。",
    "save_recommendations": "主Agent补存原文有依据但子Agent遗漏的维修建议。只交新增条目，每批最多5条；程序生成编号并与事实、证据一起原子保存，不要重交已保存建议。",
}


def validate_main_delivery(entries, workspace, confirmed_ranges):
    """Check main-Agent source delivery, separately from union child reads.

    ``confirmed_ranges`` must come from the main workflow's SourceDeliveryLedger.
    This function never grants a reading receipt or changes saved state. The
    supplementation transaction still validates its own schema and provenance.
    """
    def failure(code, message, path):
        return {"ok": False, "error": {"code": code, "message": message, "details": {
            "applied": False, "path": path,
            "action": "本批未保存；先前成果保留。只回读该引用缺少的原文，待实际请求确认投递后重交；不能以子Agent读过代替主Agent已收到。",
            "example": {"source_refs": [{"evidence_id": "<主Agent实际收到的原文编号>"}]}}}}

    if not isinstance(confirmed_ranges, dict):
        return failure("MAIN_SOURCE_DELIVERY_UNAVAILABLE", "Main-Agent confirmed source delivery is unavailable", "source_delivery")
    if not isinstance(entries, list) or not entries:
        return failure("SUPPLEMENT_ARGUMENTS", "entries must be a nonempty JSON array", "entries")
    manifest = workspace.manifest
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return failure("SUPPLEMENT_ARGUMENTS", "Each entry must be an object", f"entries[{i}]")
        refs = entry.get("source_refs")
        if not isinstance(refs, list) or not refs:
            return failure("SUPPLEMENT_ARGUMENTS", "Provide nonempty source_refs", f"entries[{i}].source_refs")
        for j, ref in enumerate(refs):
            path = f"entries[{i}].source_refs[{j}]"
            if not isinstance(ref, dict) or not isinstance(ref.get("evidence_id"), str):
                return failure("SUPPLEMENT_SOURCE", "Use an exact source ID from this document", path + ".evidence_id")
            try:
                canonical, _ = normalize_material_refs([ref], manifest, workspace.material_index,
                    path=f"entries[{i}].source_refs")
            except MaterialRefNormalizationError as exc:
                suffix = exc.path[len(f"entries[{i}].source_refs[0]"):]
                return failure(exc.code, str(exc), path + suffix)
            for span in canonical:
                identifier = span.get("evidence_id")
                if identifier not in workspace._units:
                    return failure("SUPPLEMENT_SOURCE", "Use an exact source ID from this document", path + ".evidence_id")
                try:
                    workspace._interval(identifier, span.get("start"), span.get("end"), nonempty=True)
                except WorkspaceError as exc:
                    return failure("SUPPLEMENT_SOURCE_RANGE", str(exc), path)
                if not _covers(confirmed_ranges.get(identifier, []), span["start"], span["end"]):
                    return failure("MAIN_SOURCE_NOT_DELIVERED",
                        "This source interval has not been confirmed in the main Agent's actual input. "
                        "Child reading coverage is insufficient. Read the missing source, or copy the actual "
                        "delivered start/end; an E ID without offsets means the entire unit.", path)
    return {"ok": True}


def _path(parts):
    return "".join(f"[{p}]" if isinstance(p, int) else ("." if i else "") + p
                   for i, p in enumerate(parts)) or "arguments"


class MainSupplementTools(CandidateReviewTools):
    def __init__(self, workspace, handoff_evidence, child_summary, name):
        super().__init__(workspace, handoff_evidence, child_summary)
        if name not in SUPPLEMENT_SECTIONS:
            raise ValueError("Unknown main supplement tool")
        self.name, self.section = name, SUPPLEMENT_SECTIONS[name]

    def can_replay_saved_call(self, payload):
        """Prove an exact native replay without executing or altering its journal.

        A workflow may exempt this already-saved operation from a new-operation
        budget. Pending intent alone is insufficient: either the receipt was
        finalized or the exact workspace history proves its atomic commit.
        """
        from .document_candidate_review import _current_invocation
        try:
            invocation = _current_invocation()
            if (not isinstance(invocation, dict) or invocation.get("tool_name") != self.name
                    or not all(invocation.get(key) for key in (
                        "run_id", "session_id", "generation_call_id", "native_tool_call_id", "args_sha256"))
                    or not self.path.exists()):
                return False
            key = "native_" + json_digest([
                invocation["run_id"], invocation["session_id"], invocation["native_tool_call_id"]])
            with self._lock():
                previous = self._journal()["calls"].get(key)
                if not previous:
                    return False
                digest = json_digest(payload)
                if (previous.get("tool_name") != self.name or previous.get("invocation") != invocation
                        or json_digest(previous.get("payload")) != digest
                        or previous.get("submitted_args_sha256") != digest):
                    return False
                return bool(previous.get("phase") == "complete"
                            or self._committed(self.workspace.snapshot(), previous))
        except (CandidateReviewError, OSError, ValueError, TypeError, KeyError):
            return False

    def _normalized(self, payload):
        result = deepcopy(payload)
        if isinstance(result, dict) and isinstance(result.get("entries"), list):
            for i, incoming in enumerate(result["entries"]):
                row, _, error = normalize_fields(self.name, incoming, i)
                if error:
                    raise CandidateReviewError("SUPPLEMENT_ARGUMENTS", error["message"], error["path"])
                result["entries"][i] = row
        return result

    def _schema_check(self, payload):
        try:
            json_digest(payload)
        except (TypeError, ValueError):
            raise CandidateReviewError("SUPPLEMENT_ARGUMENTS", "Arguments must be finite JSON", "arguments")
        error = next(Draft202012Validator(SUPPLEMENT_SCHEMAS[self.name]).iter_errors(self._normalized(payload)), None)
        if error:
            parts = list(error.absolute_path)
            if error.validator == "required":
                parts.append(next(k for k in error.validator_value if k not in error.instance))
            elif error.validator == "additionalProperties" and isinstance(error.instance, dict):
                parts.extend(sorted(set(error.instance) - set(error.schema.get("properties", {})))[:1])
            raise CandidateReviewError("SUPPLEMENT_ARGUMENTS", error.message, _path(parts))
        for i, entry in enumerate(self._normalized(payload)["entries"]):
            field = "病害描述" if self.name == "save_defects" else "建议内容"
            if not isinstance(entry[field], str) or not entry[field].strip():
                raise CandidateReviewError("SUPPLEMENT_CONTENT", "Provide a nonblank model-extracted description", f"entries[{i}].{field}")
            for key, value in entry.items():
                if key not in {"source_refs", "context", "status"} and not _scalar(value):
                    raise CandidateReviewError("SUPPLEMENT_VALUE", "Use an exportable finite scalar without XML control characters", f"entries[{i}].{key}")

    def _refs(self, incoming, state, path):
        """Resolve exact addresses and check reads, without granting new reads."""
        refs = []
        for i, ref in enumerate(incoming):
            here = f"{path}[{i}]"
            try:
                canonical, _ = normalize_material_refs([ref], self.workspace.manifest,
                    self.workspace.material_index, path=path)
            except MaterialRefNormalizationError as exc:
                suffix = exc.path[len(path + "[0]"):]
                raise CandidateReviewError(exc.code, str(exc), here + suffix) from exc
            for resolved in canonical:
                identifier = resolved.get("evidence_id")
                if identifier not in self.workspace._units:
                    raise CandidateReviewError("SUPPLEMENT_SOURCE", "Use an exact source ID from this document", here + ".evidence_id")
                try:
                    self.workspace._interval(identifier, resolved.get("start"), resolved.get("end"), nonempty=True)
                except WorkspaceError as exc:
                    raise CandidateReviewError("SUPPLEMENT_SOURCE_RANGE", str(exc), here) from exc
                if not _covers(state["reads"].get(identifier, []), resolved["start"], resolved["end"]):
                    raise CandidateReviewError("SUPPLEMENT_SOURCE_UNREAD",
                        "The cited interval has not been delivered. An E ID without offsets means the entire unit; "
                        "read it first or copy the exact start/end returned by the actual reading tool. Never guess offsets.", here)
                if resolved not in refs:
                    refs.append(resolved)
        return refs

    @staticmethod
    def _business_value(value):
        return {key: val for key, val in value.items() if key != "序号"} if isinstance(value, dict) else value

    def _saved_record_key(self, record):
        """Compare exact normalized claims; only host metadata is irrelevant."""
        refs, _ = normalize_material_refs(record["source_refs"], self.workspace.manifest,
                                         self.workspace.material_index)
        result = {key: val for key, val in record.items()
                  if key not in {"id", "record_version", "valid", "invalidated_by"}}
        result["value"] = self._business_value(record["value"])
        # Ordering and repeated addresses are mechanical, but different spans
        # remain different. Never merge adjacent/overlapping citation ranges.
        result["source_refs"] = sorted({(ref["evidence_id"], ref["start"], ref["end"]) for ref in refs})
        return json_digest(result)

    def _already_saved_row(self, state, normalized):
        """Recognize an intact bound row, without semantic deduplication."""
        if normalized["status"] == "conflict":
            return False
        incoming_value = json_digest(self._business_value(normalized["value"]))
        incoming_record = self._saved_record_key(normalized)
        for index, row in enumerate((state.get("draft") or {}).get(self.section, [])):
            if json_digest(self._business_value(row)) != incoming_value:
                continue
            bound = state["draft_bindings"].get(f"{self.section}.{index}", [])
            # Multiple-fact merged rows need the model's explicit review.
            if len(bound) != 1 or bound[0] not in state["draft_fact_ids"]:
                continue
            fact = state["facts"].get(bound[0])
            if not fact or fact.get("valid") is not True or fact.get("status") == "conflict":
                continue
            # Older importers kept child context only in the handoff. Do not
            # mistake an absent copied context for proof that it never existed.
            handoff = self.handoff.get(bound[0], {})
            if handoff.get("model_context") and "context" not in fact:
                continue
            if self._saved_record_key(fact) == incoming_record:
                return True
        return False

    def _plan(self, state, payload):
        proposed = deepcopy(state)
        facts, patches, assignments = [], [], []
        for i, entry in enumerate(self._normalized(payload)["entries"]):
            refs = self._refs(entry["source_refs"], state, f"entries[{i}].source_refs")
            value = {key: deepcopy(val) for key, val in entry.items()
                     if key not in {"source_refs", "status", "context"}}
            canonical = {"section": self.section, "value": value, "source_refs": refs,
                         "status": entry["status"]}
            if "context" in entry:
                canonical["context"] = deepcopy(entry["context"])
            identifier = "MSUP_" + json_digest([state["identity"], canonical])
            fact = {key: deepcopy(val) for key, val in canonical.items() if key != "section"}
            fact.update(id=identifier, source_refs=deepcopy(entry["source_refs"]))
            try:
                normalized = self.workspace._normalize_record("facts", fact, proposed, path=f"entries[{i}]")
            except WorkspaceError as exc:
                raise CandidateReviewError("SUPPLEMENT_SOURCE", str(exc), f"entries[{i}].source_refs") from exc
            existing = proposed["facts"].get(identifier)
            if existing is not None:
                fields = ("value", "source_refs", "status", "context")
                if not existing.get("valid") or any(existing.get(k) != normalized.get(k) for k in fields):
                    raise CandidateReviewError("SUPPLEMENT_ID_CONFLICT", "Saved supplement identity differs; inspect saved state", f"entries[{i}]")
                if not any(identifier in bound for bound in proposed["draft_bindings"].values()) and not any(
                        identifier in patch["fact_ids"] for patch in patches):
                    raise CandidateReviewError("SUPPLEMENT_PREVIOUSLY_UNBOUND",
                        "This exact entry was saved then removed; this new-entry tool cannot silently restore it", f"entries[{i}]")
                assignments.append({"entry_index": i, "status": "already_saved"})
                continue
            if self._already_saved_row(proposed, normalized):
                assignments.append({"entry_index": i, "status": "already_saved"})
                continue
            proposed["facts"][identifier] = normalized
            facts.append(fact)
            patches.append({"op": "append_row", "section": self.section,
                            "row": value, "fact_ids": [identifier]})
            assignments.append({"entry_index": i, "status": "saved"})
        update = {"base_version": state["version"]}
        if patches:
            # Reuse the workspace's row/binding validation on an in-memory copy.
            # Its result and the new facts are then saved in one transaction.
            self.workspace._patch_draft(patches, proposed)
            update.update(facts=facts, draft=proposed["draft"],
                draft_bindings=proposed["draft_bindings"], draft_fact_ids=proposed["draft_fact_ids"])
            statistics = derived_advice_statistics(state, update)
        else:
            statistics = None
        changed = bool(patches)
        response = {"ok": True, "changed": changed, "saved": changed,
            "version": state["version"] + int(changed), "assignments": assignments,
            "added_count": len(patches), "section": self.section,
            "draft_counts": {k: len(v) for k, v in update.get("draft", state["draft"]).items()},
            "original_candidate_facts_preserved": True, "child_inventory_changed": False,
            "file_inspection_still_required": True, "semantic_approval": False, "replayed": False,
            "batch_receipt": {"status": "saved" if changed else "unchanged", "limit": SUPPLEMENT_LIMIT,
                "action_count": len(payload["entries"]), "applied": True,
                "workspace_version": state["version"] + int(changed),
                "instruction": "本批已保存，勿重复提交。新增条目已进入当前草稿；原子Agent候选库存保持不变。继续必要补漏，再完成主审核并读取当前Word自查。"}}
        if statistics is not None:
            response["advice_statistics"] = statistics
        return update, response

    def call(self, **payload):
        response = self._execute_review(payload, tool_name=self.name)
        if not response.get("ok"):
            details = response["error"].setdefault("details", {})
            details["example"] = example(self.name)
            if details.get("applied") is False:
                details["action"] = "本批没有保存，先前成功批次保留。只修正本批所指字段后重交，不补写半截JSON，也不猜业务值或证据范围。"
        return response

    def save_defects(self, entries):
        if self.name != "save_defects":
            raise ValueError("Wrong supplement section")
        return self.call(entries=entries)

    def save_recommendations(self, entries):
        if self.name != "save_recommendations":
            raise ValueError("Wrong supplement section")
        return self.call(entries=entries)
