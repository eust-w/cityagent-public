"""A short TongAgents merge path, retaining real Word inspection/finish binding."""
from copy import deepcopy
import json

from .document_multiagent import MergeWorkflow
from .document_speed import MERGE_PROMPT, budget_receipt, constrain_tools
from .document_merge_progress import inspection_progress
from .document_delivery_validation import validate_delivery_data
from .run_state import atomic_json, json_digest
from .document_main_review_tools import MainReviewTools, SCHEMAS
from .document_main_supplement import (MainSupplementTools, SUPPLEMENT_SCHEMAS,
    validate_main_delivery)


class SpeedMergeWorkflow(MergeWorkflow):
    # Main Agent reviews existing entries in small edits; context capacity and
    # child extraction output allowances remain independent.
    main_request_output_cap = 1024
    @staticmethod
    def _validate_delivery_schema(draft):
        # Missing advice text is a recorded quality gap in this experiment.
        # The legacy one-shot parser rejects it as "meaningless_row" after
        # Word was already inspected, trapping finish in an unrelated retry.
        # Keep every exportability check; never invent the absent text.
        validate_delivery_data(draft)

    def _reader_system_prompt(self):
        return MERGE_PROMPT

    def _quick_current(self):
        record = getattr(self, "quick_review_receipt", None)
        return bool(record and record["semantic_state_sha256"] == self._semantic_identity(self.workspace.snapshot()))

    def _coverage_pending(self, snapshot):
        # Unknown is deliberately accepted, without rewriting it as resolved.
        return [] if self._quick_current() else super()._coverage_pending(snapshot) or ["quick_review_required"]

    def _main_tools(self):
        if not hasattr(self, "main_review_tools"):
            self.main_review_tools = {name: MainReviewTools(self.workspace, self.handoff_evidence, self.child_summary, name) for name in SCHEMAS if name != "finish_report"}
            self.candidate_review = self.main_review_tools["complete_review"]
            self.main_review_tools.update({name: MainSupplementTools(self.workspace,
                self.handoff_evidence, self.child_summary, name) for name in SUPPLEMENT_SCHEMAS})
        return self.main_review_tools

    def _gap_state(self):
        if not hasattr(self, "main_gap_repair"):
            state = self.workspace.snapshot()
            children = getattr(self, "child_summary", [])
            identity = json_digest({"workspace": state["identity"], "children": children})
            path = self.directory / "main-gap-repair.json"
            if path.exists():
                record = json.loads(path.read_bytes())
                if record["identity"] != identity:
                    raise ValueError("Main gap repair belongs to another document/run")
            else:
                empty = [s for s in ("病害明细", "维修建议") if not state["draft"][s]]
                failed = [c["task_id"] for c in children if c.get("status") not in {"complete", "completed"}]
                record = {"identity": identity, "triggered": bool(children and (empty or failed)),
                    "initial_empty_sections": empty, "failed_tasks": failed, "closed": False,
                    "source_attempts": 0, "save_attempts": {name: 0 for name in SUPPLEMENT_SCHEMAS},
                    "limits": {"source_attempts": 3, "entries_per_batch": 5}}
            # Saving a smaller corrected batch must not consume the permission
            # to save the next one. The main workflow already owns finite
            # model/tool/time budgets. Preserve old counters as audit history,
            # including on resume, without reopening a closed repair stage.
            record["limits"].pop("save_attempts_per_section", None)
            record["limits"]["save_calls"] = "shared_main_model_tool_time_budget"
            record.setdefault("last_save", {})
            self.main_gap_repair = record
            self._persist_gap_state()
        return self.main_gap_repair

    def _persist_gap_state(self):
        atomic_json(self.directory / "main-gap-repair.json", self.main_gap_repair)

    def _gap_active(self):
        if not getattr(self, "child_summary", []) and not hasattr(self, "main_gap_repair"):
            return False
        record = self._gap_state()
        return record["triggered"] and not record["closed"]

    def _gap_progress(self):
        record = self._gap_state()
        snapshot = self.workspace.snapshot()
        return {**deepcopy(record), "current_rows": {s: len(snapshot["draft"][s])
                    for s in ("病害明细", "维修建议")},
            **({"current_entries": self._gap_saved_entries(snapshot)}
               if record["triggered"] and not record["closed"] else {}),
            "instruction": "对照source_map与current_entries补漏：病害按对象和异常表现，建议按对象和动作分别核对。共用E编号、写入总体结论都不表示该段其余病害已入明细；不能凭空声称已归并。新增前查看实际已保存内容，勿重复提交。save_attempts仅记录调用次数，不是两次保存上限；在当前主Agent剩余调用和时间内可继续小批保存。本工具整批原子保存：applied:false表示整批未保存；只改报错字段后重交本批，或分条继续全部未保存条目，不能只补坏的一条而丢掉同批其余条目。只有回执确认saved或already_saved才跳过；成功批次保留，不重交、不重读整份材料。last_save来自实际工具回执，未确认提交不能当作已保存。原文措施不要求量化标准或施工细则，未知列省略；建议出处必须支持动作，不能只引病害句。仅缺原文才定点查阅，每响应至多一次，遵守剩余额度。补存后再处理明确重复/冲突；complete_review分别对照原句说明两类结论，未处理写未解决，有行不代表补齐。"}

    def _gap_save_receipt(self, name, payload, response):
        """Expose the actual save outcome, not an inferred semantic verdict.

        SDK rejections never reach this handler; their native failed-tool
        results remain the authority. A failed transaction may also have an
        unknown commit status, which must not be relabelled as "not saved".
        """
        entries = payload.get("entries")
        ok = response.get("ok") is True
        error = response.get("error", {})
        details = error.get("details", {})
        assignments = response.get("assignments", []) if ok else []
        receipt = {
            "attempt": self._gap_state()["save_attempts"][name],
            "outcome": ("saved" if response.get("added_count", 0) else "unchanged") if ok else
                "not_saved" if details.get("applied") is False else "commit_unconfirmed",
            "workspace_version": self.workspace.snapshot()["version"],
            "operation_workspace_version": response.get("version"),
            "batch_applied": True if ok else details.get("applied"),
            "submitted_count": len(entries) if isinstance(entries, list) else None,
            "saved_indices": [item["entry_index"] for item in assignments if item["status"] == "saved"],
            "already_saved_indices": [item["entry_index"] for item in assignments if item["status"] == "already_saved"],
            "replayed": response.get("replayed") is True,
            "save_calls_limited_by": "shared_main_model_tool_time_budget",
            "next_action": ("本批回执已确认；只继续尚未保存的条目，保存次数不是两次上限。" if ok else
                "本批整批未保存，已有成功批次保留；只改报错字段后重交本批，或分条继续全部未保存条目，不要漏掉本批其余条目。" if details.get("applied") is False else
                "本批提交结果未确认；先核对当前已保存条目和原调用回执，不猜测成功或盲目重交。"),
        }
        if not ok:
            receipt["error"] = {"code": error.get("code"), "path": details.get("path"),
                "message": error.get("message"),
                **({"example": deepcopy(details["example"])} if "example" in details else {})}
        self._gap_state()["last_save"][name] = receipt
        self._persist_gap_state()
        return {**response, "save_progress": deepcopy(receipt)}

    @staticmethod
    def _gap_saved_entries(snapshot):
        """A bounded view of committed business rows, never a coverage verdict.

        Supplement rows are not members of the frozen candidate inventory.
        Do not expose snapshot positions as editable candidate identifiers.
        """
        result = {"reference_kind": "snapshot_only_not_tool_candidate", "sections": {}}
        for section in ("病害明细", "维修建议"):
            remaining = 3000
            rows, shown = snapshot["draft"][section], []
            for index, row in enumerate(rows[:10]):
                text = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                allowance = min(800, remaining)
                if allowance <= 0:
                    break
                entry = {"value": deepcopy(row)} if len(text) <= allowance else {
                    "value_excerpt": text[:allowance], "value_truncated": True}
                remaining -= min(len(text), allowance)
                identifiers = sorted({ref["evidence_id"]
                    for fid in snapshot["draft_bindings"].get(f"{section}.{index}", [])
                    for ref in snapshot["facts"][fid].get("source_refs", [])
                    if ref.get("evidence_id")})
                entry.update(source_evidence_ids=identifiers[:8],
                    omitted_source_id_count=max(0, len(identifiers) - 8))
                shown.append(entry)
            result["sections"][section] = {"rows": shown, "total_rows": len(rows),
                "shown_rows": len(shown), "omitted_rows": len(rows) - len(shown),
                "values_complete": len(shown) == len(rows) and not any(
                    entry.get("value_truncated") for entry in shown)}
        return result

    @staticmethod
    def _gap_error(code, path, message):
        return {"ok": False, "error": {"code": code, "message": message,
            "details": {"path": path, "applied": False,
                "action": "已有成果保留；按准确剩余额度继续，未解决项写入gap_checks和notes，不无限重试。"}}}

    def _entry(self):
        mode, prompt = super()._entry()
        if self.direct_delivery:
            if self._gap_active():
                from .document_gap_material import add_gap_material
                from .document_handoff_bundle import validate_handoff_bundle
                # Missing saved rows have no candidate-bound excerpts. Supply
                # a bounded packet from the actual assigned ranges as well;
                # source presence is still not semantic review or a read ACK.
                self.handoff_bundle = add_gap_material(self.manifest,
                    self.handoff_bundle, self.child_summary, self._gap_state()["failed_tasks"])
                self._handoff_source_ranges = validate_handoff_bundle(self.manifest, self.handoff_bundle)
                atomic_json(self.directory / "initial-handoff-bundle.json", self.handoff_bundle)
                atomic_json(self.directory / "initial-handoff-source-ranges.json", self._handoff_source_ranges)
            bundle = self._main_tools()["complete_review"].project_bundle(self.handoff_bundle)
            if self._gap_active():
                # Candidate-only review order must not override the current
                # source-backed supplementation task in the initial input.
                bundle["review_order"] = self._gap_progress()["instruction"]
            atomic_json(self.directory / "initial-main-review-bundle.json", bundle)
            prompt = "以下为候选与逐字证据。candidate是本次文档整数编号；正确条目保留，只按问题类型调用专用修改工具。\n" + json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
        return mode, prompt

    def _additional_reading_handlers(self):
        def handler(name):
            def call(**payload):
                if name in SUPPLEMENT_SCHEMAS:
                    # Recover an already-proven commit before applying a new
                    # work allowance. Replaying a receipt is not a new save.
                    supplement = self._main_tools()[name]
                    if supplement.can_replay_saved_call(payload):
                        return self._gap_save_receipt(name, payload, supplement.call(**payload))
                    record = self._gap_state()
                    if not self._gap_active():
                        return self._gap_error("GAP_REPAIR_CLOSED", "entries", "本轮补漏阶段已关闭；已有成果保持不变。")
                    record["save_attempts"][name] += 1
                    self._persist_gap_state()
                    delivery = getattr(self, "source_delivery", None)
                    delivered = validate_main_delivery(payload.get("entries"), self.workspace,
                        delivery.confirmed_ranges() if delivery is not None else None)
                    if not delivered.get("ok"):
                        return self._gap_save_receipt(name, payload, delivered)
                if name == "complete_review" and self._gap_active():
                    checks = payload.get("gap_checks")
                    if not isinstance(checks, dict) or any(not isinstance(checks.get(s), str) or not checks[s].strip()
                            for s in ("病害明细", "维修建议")):
                        return self._gap_error("GAP_CHECK_REQUIRED", "gap_checks",
                            '分别说明两类补漏结论，例如{"病害明细":"已根据原文补存主要病害","维修建议":"证据不足，仍未解决"}；不要求非空表。')
                result = self._main_tools()[name].call(**payload)
                if name in SUPPLEMENT_SCHEMAS:
                    result = self._gap_save_receipt(name, payload, result)
                if result.get("ok") and name == "complete_review":
                    record = self._gap_state()
                    record.update(closed=True, model_checks=deepcopy(payload.get("gap_checks", {})),
                        review_workspace_version=self.workspace.snapshot()["version"])
                    self._persist_gap_state()
                    self.quick_review_receipt = {"semantic_state_sha256": self._semantic_identity(self.workspace.snapshot()),
                        "basis": payload["basis"], "remaining_issues": [payload["notes"]] if payload.get("notes") else [],
                        "actual_tool_receipt": deepcopy(result), "exhaustive_quality_verified": False}
                    atomic_json(self.directory / "quick-review.json", self.quick_review_receipt)
                    # Export/read actual bytes only after model review succeeds.
                    # The next model call sees these pages in this tool result;
                    # preparing pages is not a model self-check or finish.
                    result = self._prepare_review_word(payload, result)
                return result
            return call
        self._main_tools()
        result = {**{name: handler(name) for name in self._main_tools()}, "finish_report": self._finish_report}
        # These are the same managed readers initialized by the parent. Count
        # before dispatch, so several calls in one response cannot bypass the
        # allowance; rejected attempts do not grant a fresh budget.
        def reader(name, function):
            def call(**payload):
                if self._gap_active():
                    record = self._gap_state()
                    if record["source_attempts"] >= 3:
                        return self._gap_error("GAP_READ_BUDGET", "arguments", "三次定点查阅额度已用完。")
                    record["source_attempts"] += 1
                    self._persist_gap_state()
                return function(**payload)
            return call
        if hasattr(self, "tools"):
            automatic = self.tools.automatic_handlers()
            for name in ("read_source", "read_table_rows", "search_in_document"):
                result[name] = reader(name, automatic[name])
        return result

    def _prepare_review_word(self, payload, result):
        origin = self._source_proposal_origin("complete_review", payload)
        # Keep hashes, paths and the complete execution receipt on disk. The
        # model needs the actual table content and pagination, not three copies
        # of internal file-binding metadata.
        ledger_path = self.directory / "prepared-word-pages.json"
        ledger = self._prepared_word_ledger()
        limit = 4
        while True:
            pages = [self._inspect_delivery_page(view, 0, limit, origin, managed_restart=True)
                     for view in ("basic", "advice", "defects")]
            prepared = {"ok": True, "review_completed": True,
                "workspace_version": self.workspace.snapshot()["version"],
                "exhaustive_quality_verified": False,
                "actual_word_pages": [self._compact_word_page(p) for p in pages],
                "next_action": "检查以下实际Word内容；如有next_cursor，用inspect_delivery继续，读完后finish_report。内容有问题则局部修改并重新审核。"}
            if self._count_tokens(json.dumps(prepared, ensure_ascii=False)) + 64 <= self.settings.tool_tokens:
                for page in pages:
                    ledger["pages"][page["inspection_proof"]["receipt_sha256"]] = page
                atomic_json(ledger_path, ledger)
                return prepared
            if limit == 1:
                return {**result, "next_action": "审核已保存；实际文件较长，请inspect_delivery分页读取。"}
            limit = max(1, limit // 2)

    def _prepared_word_ledger(self):
        path = self.directory / "prepared-word-pages.json"
        ledger = json.loads(path.read_bytes()) if path.exists() else {
            "identity": json_digest(self.identity), "pages": {}}
        if ledger["identity"] != json_digest(self.identity):
            raise ValueError("Prepared Word pages belong to another workflow")
        return ledger

    @staticmethod
    def _compact_word_page(page):
        return {"inspection_receipt": page["inspection_proof"]["receipt_sha256"],
            **{k: deepcopy(page[k]) for k in ("section", "headers", "text", "cursor", "end",
                "total_chars", "next_cursor", "complete_page", "row_count")}}

    def _artifact_tool_payloads(self, envelope):
        if (type(envelope) is dict and envelope.get("ok") is True
                and envelope.get("tool") == "complete_review"
                and type(envelope.get("result")) is dict
                and envelope["result"].get("ok") is True):
            supplied = envelope["result"].get("actual_word_pages", [])
            if type(supplied) is not list:
                return []
            ledger = self._prepared_word_ledger()["pages"]
            return [page for public in supplied if type(public) is dict
                    for page in [ledger.get(public.get("inspection_receipt"))]
                    if page and public == self._compact_word_page(page)]
        return super()._artifact_tool_payloads(envelope)

    def _speed_progress(self):
        state = self.workspace.snapshot()
        pages = inspection_progress(state, self.artifact_delivery, self.delivery_inspections)
        if not self._quick_current():
            stage = "review_edits"
            if self._gap_active():
                # The candidate-only action does not account for supplement
                # saves. Its old "disease edits first / may finish" action
                # cannot describe this active gap-repair stage accurately.
                action = self._gap_progress()["instruction"]
            else:
                action = self._main_tools()["complete_review"].next_action()
        elif not self._artifact_read_coverage(state):
            stage, action = "inspect_delivery", "读取当前Word尚未检查的basic/advice/defects页；已有有效页不重读。"
        else:
            stage, action = "finish_report", "文件已读回；只提交快速自查和仍有的缺项，然后结束。"
        return {"stage": stage, "action": action, "file_inspection": pages,
            "gap_repair": self._gap_progress(),
            "single_request_output_tokens": self.main_request_output_cap,
            "output_instruction": "不要先写分析正文；直接提交当前工具的小批参数，写不完留待下一次。",
            "budget": budget_receipt(self._model_calls(), soft_seconds=90, max_calls=self.settings.max_model_calls)}

    def _before_native_model(self):
        self._check_budget()
        if self._finish_is_current():
            return None
        progress = self._speed_progress()
        atomic_json(self.directory / "speed-progress.json", progress)
        return "程序当前阶段与准确预算：" + json.dumps(progress, ensure_ascii=False, separators=(",", ":"))

    def _before_continuous_request(self, payload):
        if hasattr(self, "client"):
            self.client.generation_output_cap = self.main_request_output_cap
            self.client.reliable_tool_content_limit = 2048
            self.client.reliable_tool_argument_contract = True
        if self._finish_is_current():
            result = deepcopy(payload)
            result.pop("tools", None)
            result["tool_choice"] = "none"
            return result
        progress = self._speed_progress()
        stage = progress["stage"]
        # Every stage still executes through native TongAgents tool dispatch.
        allowed = set(SCHEMAS) - {"finish_report"} if stage == "review_edits" else {stage}
        if stage == "review_edits" and self._gap_active():
            record = self._gap_state()
            allowed |= set(SUPPLEMENT_SCHEMAS)
            if record["source_attempts"] < 3:
                allowed |= {"read_source", "read_table_rows", "search_in_document"}
        elif stage == "review_edits" and not progress["budget"]["wrap_up"]:
            reads = getattr(self, "_speed_lookup_attempts", 0)
            if reads < 1:
                allowed |= {"read_source", "read_table_rows", "search_in_document"}
        # The exported Word views are not source table identifiers. An input
        # with paragraphs only cannot support a source-table read at all.
        if not getattr(self, "manifest", {}).get("tables"):
            allowed.discard("read_table_rows")
        # SDK tool selection permits a registered subset, never rewritten
        # schemas. Shared child save schemas are compatible supersets; the
        # main handler enforces its smaller batch/no-ID contract atomically.
        return constrain_tools(payload, allowed, named=stage if len(allowed) == 1 else None)

    def _observe_native_tool(self, name, arguments, response):
        # Count rejected attempts too; an invalid search must not reopen the
        # one-lookup allowance indefinitely.
        if name in {"read_source", "read_table_rows", "search_in_document"}:
            self._speed_lookup_attempts = getattr(self, "_speed_lookup_attempts", 0) + 1
        super()._observe_native_tool(name, arguments, response)

    def _generation_recovery_prompt(self, receipt):
        progress = self._speed_progress()
        atomic_json(self.directory / "generation-recovery.json", {
            "failed_call_id": receipt.get("call_id"), "error_code": receipt.get("error_code"),
            "saved_progress": progress, "failed_output_executed": False, "additional_recovery_limit": 0})
        reason = ("上一请求已正常结束，但解析得到的工具调用缺少 function.name 或 id（具体路径见协议回执），整批参数未执行、未加入历史。必须选择本次 tools 列表里的精确工具名，以原生工具调用提交完整参数；不要改成普通正文或自定义 <工具名> 标签。若使用模型原生 XML 工具格式，函数起始应为 <function=工具名>，不要写成 <工具名>；不要照抄上次残缺参数。只提交准确进度中当前缺少的一小组操作。"
                  if receipt.get("error_code") == "incomplete_tool_call" else
                  "上一请求已达到单轮输出上限并结束，残缺工具参数未执行、未加入历史；缩小到一组修改。"
                  if receipt.get("error_code") == "output_token_limit" else
                  "上一请求触发重复、正文或工具字段长度保护，服务停止观测已记录；异常输出未执行、未加入历史。")
        if receipt.get("error_code") == "incomplete_tool_call":
            reason += "本次协议错误：" + json.dumps(receipt.get("protocol_error"), ensure_ascii=False)
        return (reason +
            "已有草稿和文件保留。这是唯一一次恢复；不要写长正文，只调用当前需要的工具："
            + json.dumps(progress, ensure_ascii=False))

    def _continue_after_native_exit(self, exit_state):
        self._check_budget()
        if self._finish_is_current():
            return None if exit_state.get("native_final_event") and exit_state.get("response_text", "").strip() else "文件已交付，请简短结束。"
        return "仅继续当前阶段：" + json.dumps(self._speed_progress(), ensure_ascii=False)

    def _native_finish_tool_name(self):
        return "finish_report"

    def _terminal_delivery_state(self, *, require_active=True):
        if not self._finish_is_current():
            return None
        if require_active and self.last_finish_origin["call_id"] != self._active_agent_call_id:
            return None
        return {"origin_call_id": self.last_finish_origin["call_id"],
            "finish": deepcopy(self.last_finish), "finish_origin": deepcopy(self.last_finish_origin)}

    def _native_exit_continuation_config(self):
        return {**super()._native_exit_continuation_config(),
            "terminal_completion_tool": "finish_report",
            "terminal_handoff_state": self._terminal_delivery_state,
            "terminal_generation_is_settled": self._parse_recovery_generation_is_settled}

    def _native_delivery_completed(self, outcome):
        if outcome.get("termination_kind") == "document_terminal_finish":
            return outcome.get("status") == "completed"
        return super()._native_delivery_completed(outcome)

    def _validate_single_agent_native_delivery(self, outcome):
        if outcome.get("termination_kind") != "document_terminal_finish":
            return super()._validate_single_agent_native_delivery(outcome)
        self._require_stage_boundary(outcome)
        # The SDK has released its active call by this point. Validate against
        # the durable origin and the returned final call, not a live-session ID.
        state = self._terminal_delivery_state(require_active=False)
        calls = outcome.get("reading_model_call_ids", [])
        proof = outcome.get("terminal_handoff_proof", {})
        if (not state or not calls or calls[-1] != state["origin_call_id"]
                or outcome.get("terminal_handoff_origin_call_id") != state["origin_call_id"]
                or outcome.get("terminal_handoff_sha256") != json_digest(state)
                or not all(proof.get(k) is True for k in (
                    "all_batch_tool_calls_paired", "stopped_before_next_model_call", "sdk_stop_returned"))):
            raise ValueError("Document completion lacks a current, fully drained native finish proof")
        audit = self._terminal_model_bundle(state["origin_call_id"])["audit"]
        if any(self.last_finish_origin[k] != audit[k] for k in
               ("request_id", "request_sha256", "response_sha256")):
            raise ValueError("Program completion differs from actual finish response")

    def _finish_report(self, basis, notes=""):
        # This tool explicitly attests actual-file review/completion; these
        # booleans encode its declared meaning, not a host content verdict.
        self._flat_finish_arguments = {"basis": basis, "notes": notes}
        try:
            return self._finish(self_check={"inspected_export": True, "complete": True,
                "basis": basis, "remaining_issues": [notes] if notes else []})
        finally:
            self._flat_finish_arguments = None

    def _finish_statement_origin(self, self_check):
        flat = getattr(self, "_flat_finish_arguments", None)
        if flat is None:
            return super()._finish_statement_origin(self_check)
        expected = {"inspected_export": True, "complete": True, "basis": flat["basis"],
            "remaining_issues": [flat["notes"]] if flat["notes"] else []}
        if self_check != expected:
            raise ValueError("Expanded finish statement differs from the actual flat tool arguments")
        return self._source_proposal_origin("finish_report", flat)

    def _finish(self, self_check=None):
        result = super()._finish(self_check=self_check)
        if result.get("ok"):
            state = self.workspace.snapshot()
            atomic_json(self.directory / "quality-gaps.json", {
                "mode": "speed_v1", "delivery_complete": True, "exhaustive_quality_verified": False,
                "review": deepcopy(self.quick_review_receipt), "file_self_check": deepcopy(self.model_self_check),
                "unread_source_ranges": self.workspace.snapshot()["reading_progress"]["unread_ranges"],
                "questions": state["questions"], "children": deepcopy(self.child_summary)})
            atomic_json(self.directory / "main-gap-repair.json", self._gap_state())
        return result
