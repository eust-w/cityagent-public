"""Two-phase original-source delivery, separate from extraction and review.

Tool output is an outbox item. Only the final wire request plus trusted upstream
generation evidence confirms delivery. Neither a source id nor a model summary
can manufacture original-text coverage.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
from uuid import uuid4

from .run_state import atomic_json


RECEIPT_FIELD = "source_delivery_receipt"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _union(ranges):
    result = []
    for start, end in sorted(ranges):
        if start == end:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


class SourceDeliveryLedger:
    """One immutable document/manifest, run and Agent per durable directory."""

    def __init__(self, manifest, directory, *, run_id, agent_id, on_confirmed=None):
        if not isinstance(run_id, str) or not run_id or not isinstance(agent_id, str) or not agent_id:
            raise ValueError("Source delivery requires stable run and Agent identities")
        self.manifest = deepcopy(manifest)
        self._units = {unit["id"]: unit["text"] for unit in manifest["units"]}
        self.identity = {"document_sha256": manifest["document_sha256"],
            "manifest_sha256": _digest(manifest), "run_id": run_id, "agent_id": agent_id}
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "source-delivery.json"
        self.on_confirmed = on_confirmed
        self._lock = threading.RLock()
        self._state = {"schema_version": 1, "identity": self.identity, "pages": {}, "requests": {}}
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state.get("schema_version") != 1 or state.get("identity") != self.identity:
                raise ValueError("Source delivery ledger identity differs from document/manifest/run/Agent")
            self._state = state
        else:
            self._persist()
        self._apply_confirmed()

    def _persist(self):
        try:
            atomic_json(self.path, self._state)
        except Exception:
            # A failed durable write cannot leave an in-memory confirmation
            # that navigation would trust. Reconcile with the actual disk
            # after an uncertain rename outcome; a new instance sees the same.
            if self.path.exists():
                self._state = json.loads(self.path.read_text())
            raise

    def state(self):
        with self._lock:
            return deepcopy(self._state)

    def _range(self, value):
        identifier, start, end = value.get("evidence_id"), value.get("start"), value.get("end")
        source = self._units.get(identifier)
        if (source is None or type(start) is not int or type(end) is not int
                or not 0 <= start <= end <= len(source)):
            raise ValueError("Source delivery range does not address this immutable manifest")
        text = source[start:end]
        if "text" in value and value["text"] != text:
            raise ValueError("Delivered source text differs from its exact original range")
        return {"evidence_id": identifier, "start": start, "end": end, "text": text}

    def _fragments(self, result):
        fragments = {}
        # Only source-return containers, never arbitrary model-authored facts,
        # summaries or metadata with an incidental evidence_id.
        for name in ("spans", "context", "matches"):
            for item in result.get(name, []):
                if not isinstance(item, dict) or "text" not in item:
                    continue
                if not all(key in item for key in ("evidence_id", "start", "end")):
                    continue
                span = self._range(item)
                if span["start"] != span["end"]:
                    fragments[(span["evidence_id"], span["start"], span["end"])] = span
        return list(fragments.values())

    def reserve_receipt_id(self):
        """Allocate before token budgeting; never estimate a cheaper dummy id."""
        return _digest([self.identity, uuid4().hex])

    def prepare_page(self, result, *, tool, arguments=None, receipt_id=None):
        """Persist a successful final bounded tool page before returning it."""
        if not result.get("ok"):
            return deepcopy(result)
        with self._lock:
            old = result.get(RECEIPT_FIELD)
            if old is not None:
                if old not in self._state["pages"]:
                    raise ValueError("Page receipt is not owned by this source ledger")
                return deepcopy(result)
            page = deepcopy(result)
            page.setdefault("replayed", False)
            fragments = self._fragments(page)
            identifier = receipt_id or self.reserve_receipt_id()
            if (not isinstance(identifier, str) or len(identifier) != 64
                    or any(c not in "0123456789abcdef" for c in identifier)
                    or identifier in self._state["pages"]):
                raise ValueError("Invalid or reused source delivery receipt reservation")
            page[RECEIPT_FIELD] = identifier
            self._state["pages"][identifier] = {"kind": "tool", "status": "pending",
                "tool": tool, "arguments": deepcopy(arguments), "page": page,
                "fragments": fragments, "confirmed_requests": [], "applied": False}
            self._persist()
            return deepcopy(page)

    def register_initial_message(self, content, ranges, *, role="user"):
        """Register program-built initial source; not a delivery confirmation."""
        if not isinstance(content, str) or role not in {"user", "system", "tool"}:
            raise ValueError("Initial source message must be exact text with an explicit role")
        fragments = [self._range(value) for value in ranges]
        for span in fragments:
            text = span["text"]
            # Prompts may embed original text inside JSON, including newlines.
            if text and text not in content and json.dumps(text, ensure_ascii=False)[1:-1] not in content:
                raise ValueError("Initial source range is absent from its registered message")
        identifier = _digest([self.identity, "initial", role, content, fragments])
        with self._lock:
            if identifier not in self._state["pages"]:
                self._state["pages"][identifier] = {"kind": "initial", "status": "pending",
                    "role": role, "content": content, "content_sha256": _digest(content), "fragments": fragments,
                    "confirmed_requests": [], "applied": False}
                self._persist()
        return identifier

    def pending_initial_messages(self):
        """Exact initial inputs for recovery; summaries cannot replace them."""
        with self._lock:
            delivered = self.confirmed_ranges()
            messages, seen = [], set()
            for page in self._state["pages"].values():
                if page["kind"] != "initial" or page["status"] == "confirmed":
                    continue
                fragments = page["fragments"]
                if all(any(a <= span["start"] <= span["end"] <= b
                           for a, b in delivered.get(span["evidence_id"], []))
                       or span["start"] == span["end"] for span in fragments):
                    continue
                key = _digest(sorted((span["evidence_id"], span["start"], span["end"]) for span in fragments))
                if key not in seen:
                    messages.append({"role": page["role"], "content": page["content"]})
                    seen.add(key)
            return messages

    @staticmethod
    def _tool_results(value):
        if not isinstance(value, dict):
            return []
        if RECEIPT_FIELD in value:
            return [value]
        result = value.get("result")
        return [result] if isinstance(result, dict) and RECEIPT_FIELD in result else []

    def observe_request(self, payload, audit):
        """Record only original pages actually present after all compression."""
        request_id = audit.get("request_id") if isinstance(audit, dict) else audit
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Actual request requires a stable request_id")
        with self._lock:
            observed = {}
            complete_pages = []
            for message in payload.get("messages", []):
                role, content = message.get("role"), message.get("content")
                if not isinstance(content, str):
                    continue
                for identifier, page in self._state["pages"].items():
                    if (page["kind"] == "initial" and page["role"] == role
                            and page["content_sha256"] == _digest(content)):
                        observed[identifier] = deepcopy(page["fragments"])
                        complete_pages.append(identifier)
                if role != "tool":
                    continue
                try:
                    decoded = json.loads(content)
                except (ValueError, TypeError):
                    continue
                for result in self._tool_results(decoded):
                    identifier = result[RECEIPT_FIELD]
                    page = self._state["pages"].get(identifier) if isinstance(identifier, str) else None
                    if page is None or page["kind"] != "tool" or not result.get("ok"):
                        continue
                    # The id alone cannot authorize coverage. Recollect the
                    # exact source actually carried by this final request.
                    present = self._fragments(result)
                    present = [s for s in present if any(
                        s["evidence_id"] == allowed["evidence_id"]
                        and allowed["start"] <= s["start"] <= s["end"] <= allowed["end"]
                        for allowed in page["fragments"])]
                    # A page stripped by compression must remain pending. Even
                    # navigation-only pages require their real reply, not id.
                    expected = {k: v for k, v in page["page"].items() if k not in {"replayed"}}
                    actual = {k: v for k, v in result.items() if k not in {"replayed"}}
                    if actual == expected:
                        observed[identifier] = present
                        complete_pages.append(identifier)
                    elif present:
                        # A deliberately shortened final request grants only
                        # text it actually retained; the full page stays pending.
                        observed[identifier] = present
            row = {"status": "prepared", "payload_sha256": _digest(payload), "pages": observed,
                   "complete_pages": sorted(set(complete_pages)), "applied": False}
            previous = self._state["requests"].get(request_id)
            if previous is not None:
                if previous["payload_sha256"] != row["payload_sha256"]:
                    raise ValueError("Actual request_id reused for a different payload")
                return deepcopy(previous)
            self._state["requests"][request_id] = row
            self._persist()
            return deepcopy(row)

    def confirm_request(self, request_id, *, evidence):
        """Trusted upstream acknowledgement, independent of output correctness."""
        valid = (isinstance(evidence, dict) and (
            evidence.get("input_delivery_confirmed") is True
            or (evidence.get("kind") == "upstream_response" and bool(evidence.get("response_id")))
            or (evidence.get("kind") == "generation_observed" and any(
                type(evidence.get(key)) is int and evidence[key] > 0
                for key in ("generated_tokens", "generated_characters")))))
        if not valid or evidence.get("request_id", request_id) != request_id:
            raise ValueError("Source delivery lacks matching upstream generation evidence")
        with self._lock:
            request = self._state["requests"].get(request_id)
            if request is None:
                raise ValueError("Cannot confirm a request whose final payload was not observed")
            request.update(status="confirmed", evidence=deepcopy(evidence))
            for identifier in request["complete_pages"]:
                page = self._state["pages"][identifier]
                page["status"] = "confirmed"
                if request_id not in page["confirmed_requests"]:
                    page["confirmed_requests"].append(request_id)
            self._persist()
            self._apply_confirmed()
            return {"ok": True, "confirmed_pages": len(request["pages"]),
                    "semantic_review": False, "request_id": request_id}

    def reject_request(self, request_id, reason):
        with self._lock:
            row = self._state["requests"].get(request_id)
            if row is None:
                return
            # A later output failure cannot undo known input delivery.
            if row["status"] != "confirmed":
                row.update(status="unconfirmed", failure_reason=str(reason))
                self._persist()

    def _apply_confirmed(self):
        if self.on_confirmed is None:
            return
        applied = {}
        for previous in self._state["requests"].values():
            if previous["status"] == "confirmed" and previous.get("applied"):
                for spans in previous["pages"].values():
                    for span in spans:
                        applied.setdefault(span["evidence_id"], []).append([span["start"], span["end"]])
        applied = {key: _union(ranges) for key, ranges in applied.items()}
        for request in self._state["requests"].values():
            if request["status"] != "confirmed" or request.get("applied"):
                continue
            for spans in request["pages"].values():
                for span in spans:
                    identifier, cursor, end = span["evidence_id"], span["start"], span["end"]
                    # Historical messages often repeat in hundreds of actual
                    # requests. Keep every request receipt, but apply coverage
                    # only for newly confirmed intervals.
                    for old_start, old_end in applied.get(identifier, []):
                        if old_end <= cursor:
                            continue
                        if old_start >= end:
                            break
                        if old_start > cursor:
                            self.on_confirmed(identifier, cursor, old_start)
                        cursor = max(cursor, old_end)
                    if cursor < end:
                        self.on_confirmed(identifier, cursor, end)
                    applied[identifier] = _union([*applied.get(identifier, []), [span["start"], end]])
            request["applied"] = True
            self._persist()

    def is_confirmed(self, receipt_id):
        with self._lock:
            return self._state["pages"].get(receipt_id, {}).get("status") == "confirmed"

    def discard_unpublished_page(self, receipt_id):
        """Rollback a newly prepared page rejected by final wire budgeting."""
        with self._lock:
            page = self._state["pages"].get(receipt_id)
            if page is None:
                return
            if page["status"] != "pending" or any(receipt_id in row["pages"] for row in self._state["requests"].values()):
                raise ValueError("Cannot discard a page already observed in a model request")
            del self._state["pages"][receipt_id]
            self._persist()

    def pending_page(self, receipt_id):
        with self._lock:
            page = self._state["pages"].get(receipt_id)
            if page is None:
                raise ValueError("Unknown persisted source delivery receipt")
            return deepcopy(page.get("page")) if page["status"] != "confirmed" else None

    def confirmed_ranges(self):
        result = {}
        with self._lock:
            for request in self._state["requests"].values():
                if request["status"] == "confirmed":
                    for spans in request["pages"].values():
                        for span in spans:
                            result.setdefault(span["evidence_id"], []).append([span["start"], span["end"]])
        return {key: _union(value) for key, value in result.items() if _union(value)}
