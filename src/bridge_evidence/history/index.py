from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from rapidfuzz import fuzz

from bridge_evidence.adapters import FormatAdapter
from bridge_evidence.agents import GatewayClient
from bridge_evidence.evaluation.labels import LabelReport, parse_label
from bridge_evidence.evidence.extractors import (
    normalize_component,
    normalize_disease_type,
)
from bridge_evidence.history_semantics import development_level
from bridge_evidence.matching import DiseaseMatcher
from bridge_evidence.models.schema import (
    Disease,
    DocumentState,
    EvidenceRef,
    FieldValue,
)
from bridge_evidence.parsers.docx import ParsedDocument


def _identity(value: str) -> str:
    value = str(value or "").upper().replace("Ⅰ", "I").replace("Ⅱ", "II").replace("Ⅲ", "III").replace("Ⅳ", "IV")
    value = value.replace("Ⅴ", "V").replace("Ⅵ", "VI").replace("Ⅶ", "VII").replace("Ⅷ", "VIII")
    value = value.replace("号", "#").replace("＃", "#")
    value = re.sub(r"^(?:重庆市)?(?:渝遂高速|渝黔路|上界路|成渝高速|机场路)", "", value)
    return re.sub(r"[\s_-—–（）().,，。·]", "", value)


def _route_codes(value: str) -> set[str]:
    normalized = _identity(value)
    return set(re.findall(r"K\d+\+\d+(?:\.\d+)?", normalized, re.I))


def _ramp_ids(value: str) -> set[str]:
    normalized = _identity(value)
    return set(re.findall(r"([A-Z](?:-\d+)?)匝道", normalized, re.I))


def _structure_types(value: str) -> set[str]:
    normalized = _identity(value)
    result: set[str] = set()
    aliases = (
        ("人行天桥", "人行天桥"),
        ("桥式通道", "通道"),
        ("人行通道", "通道"),
        ("旱桥", "旱桥"),
        ("立交桥", "立交桥"),
        ("跨线桥", "跨线桥"),
        ("大桥", "大桥"),
        ("中桥", "中桥"),
        ("小桥", "小桥"),
    )
    for token, canonical in aliases:
        if token in normalized:
            result.add(canonical)
    return result


def _semantic_name(value: str) -> str:
    normalized = _identity(value)
    normalized = re.sub(r"K\d+\+\d+(?:\.\d+)?", "", normalized, flags=re.I)
    normalized = re.sub(r"(?:桥梁)?(?:检测|定检|评估|检查)(?:评估)?报告\d*$", "", normalized)
    normalized = re.sub(r"报告\d*$", "", normalized)
    normalized = re.sub(r"立交(?=[A-Z0-9IVX#]+匝道)", "", normalized)
    normalized = normalized.replace("#", "")
    return normalized.strip("-")


def _name_from_label_path(path: Path) -> str:
    value = path.stem
    value = re.sub(r"[-—_ ]*(?:含有?|有|无)?对比年度的?信息提取报告.*$", "", value)
    value = re.sub(r"[-—_ ]*信息提取报告.*$", "", value)
    value = re.sub(r"^\d{6}[-—_]\d+", "", value)
    value = re.sub(r"^\d+[、.\-—_]\s*", "", value)
    value = re.sub(r"(?:桥梁)?(?:检测|定检|评估|检查)(?:评估)?报告\d*$", "", value)
    value = re.sub(r"报告\d*$", "", value)
    return value.strip(" _-—")


def _cycle_year(path: Path, parsed: ParsedDocument, report_date: object) -> int | None:
    for parent in path.parents:
        match = re.fullmatch(r"(20\d{2})年", parent.name)
        if match:
            return int(match.group(1))
    patterns = (
        r"本次检测时间为\s*(20\d{2})年",
        r"本次检查时间为\s*(20\d{2})年",
        r"检测日期\s*[|:：]?\s*(20\d{2})",
    )
    for pattern in patterns:
        match = re.search(pattern, parsed.full_text)
        if match:
            return int(match.group(1))
    match = re.search(r"20\d{2}", str(report_date or ""))
    return int(match.group()) if match else None


def _prior_cycle(parsed: ParsedDocument, current_year: int) -> tuple[int | None, bool]:
    """Return the prior inspection cycle explicitly evidenced by the report.

    A released historical label is external context, not a licence to invent a
    comparison.  It is eligible only when the current report itself contains a
    historical-comparison signal.  If a year is explicit, that exact temporal
    edge is enforced instead of selecting the newest record in the index.
    """

    text = parsed.full_text
    comparison_signal = bool(
        re.search(
            r"上(?:一|次|回|年度)|前次|历次|最近两次|"
            r"(?:对比|比较)\s*20\d{2}年|与\s*20\d{2}年(?:度)?(?:检测|检查|定检|对比)",
            text,
        )
    )
    candidates: list[int] = []
    patterns = (
        r"(?:上次|上一次|前次|上一年度).{0,30}?(?:检测|检查|定检).{0,20}?(20\d{2})年",
        r"(?:对比|比较|与)\s*(20\d{2})年(?:度)?",
        r"(20\d{2})年(?:度)?.{0,20}?(?:与本次|和本次|至本次|与\s*20\d{2}年).{0,12}?(?:对比|比较)",
    )
    for pattern in patterns:
        candidates.extend(int(match) for match in re.findall(pattern, text) if int(match) < current_year)
    return (max(candidates) if candidates else None), comparison_signal


def _numeric(value: object) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else None


def _history_score_value(value: object) -> float | str | None:
    """Preserve left/right historical scores instead of forcing one float."""
    numbers = re.findall(r"\d+(?:\.\d+)?", str(value or ""))
    if not numbers:
        return None
    if len(numbers) == 1:
        return float(numbers[0])
    return str(value).strip()


def _record_years(record: "HistoryRecord") -> set[int]:
    years = {record.cycle_year}
    report_year = re.search(r"20\d{2}", record.report_date)
    if report_year:
        years.add(int(report_year.group()))
    return years


@dataclass(frozen=True)
class HistoryRecord:
    label: LabelReport
    bridge_name: str
    cycle_year: int
    report_date: str
    overall_score: str
    overall_grade: str
    diseases: list[Disease]

    @property
    def descriptor(self) -> str:
        return f"{self.bridge_name} | {self.cycle_year}年 | {self.report_date}"


class HistoryIndex:
    """Chronology-safe index over released training labels.

    Records from the current or a future inspection cycle are never eligible.
    This makes the same implementation safe for grouped/temporal holdout
    evaluation and prevents answer leakage from the label being evaluated.
    """

    def __init__(
        self,
        records: list[HistoryRecord],
        client: GatewayClient | None = None,
        *,
        disease_match_threshold: float = 0.56,
    ):
        self.records = records
        self.client = client
        self.matcher = DiseaseMatcher(disease_match_threshold)

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        adapter: FormatAdapter,
        *,
        client: GatewayClient | None = None,
        disease_match_threshold: float = 0.56,
        before_year: int | None = None,
    ) -> "HistoryIndex":
        records: list[HistoryRecord] = []
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in {".doc", ".docx"}:
                continue
            year_match = re.fullmatch(r"(20\d{2})年", path.parent.name)
            if not year_match:
                continue
            # Temporal holdout must not even parse the current/future labels.
            # Filtering before parse_label makes the no-leakage boundary
            # observable and stronger than filtering candidates after loading.
            if before_year is not None and int(year_match.group(1)) >= before_year:
                continue
            try:
                label = parse_label(path, adapter)
            except Exception:
                continue
            bridge_name = label.summary.get("桥梁名称", "").strip() or _name_from_label_path(path)
            if not bridge_name:
                continue
            diseases = []
            for row in label.diseases:
                location = row.get("病害部位", "")
                description = row.get("病害描述", "")
                component = normalize_component(location)
                if component == "其他":
                    component = normalize_component(description)
                is_new_text = row.get("是否新增", "")
                diseases.append(
                    Disease(
                        sequence=row.get("序号", ""),
                        component=component,
                        location=location,
                        type=normalize_disease_type(row.get("病害类型", "")),
                        description=description,
                        is_new=True if is_new_text == "是" else False if is_new_text == "否" else None,
                        previous_state=row.get("上一次定检状态") or None,
                        development=row.get("发展程度") or None,
                        confidence=1.0,
                    )
                )
            records.append(
                HistoryRecord(
                    label=label,
                    bridge_name=bridge_name,
                    cycle_year=int(year_match.group(1)),
                    report_date=label.summary.get("报告日期", ""),
                    overall_score=label.summary.get("总体评分", ""),
                    overall_grade=label.summary.get("总体等级", ""),
                    diseases=diseases,
                )
            )
        return cls(records, client, disease_match_threshold=disease_match_threshold)

    def enrich(self, state: DocumentState, parsed: ParsedDocument, source_path: Path) -> dict[str, object]:
        from bridge_evidence.models.schema import DiseaseHistoryLink

        current_year = _cycle_year(source_path, parsed, state.bridge.report_date.value)
        if current_year is None or not state.bridge.name.value:
            return {"matched": False, "reason": "缺少桥名或检测年份"}
        prior_year, has_comparison_signal = _prior_cycle(parsed, current_year)
        if not has_comparison_signal:
            has_comparison_signal = bool(re.search(r"(?:^|[（(，,、\s])原\s*[A-E一二三四五]", source_path.stem, re.I))
        if not has_comparison_signal:
            return {"matched": False, "reason": "当前报告无历史对比信号", "current_year": current_year}
        eligible = [record for record in self.records if record.cycle_year < current_year]
        match = self._match_record(str(state.bridge.name.value), eligible, prior_year=prior_year)
        if match is None:
            return {
                "matched": False,
                "reason": "无高置信历史桥梁匹配",
                "current_year": current_year,
                "prior_year": prior_year,
            }
        record, similarity = match
        source_digest = sha256(str(record.label.path.resolve()).encode("utf-8")).hexdigest()[:12]
        evidence_id = f"HISTORY:{record.cycle_year}:{source_digest}"
        evidence_text = (
            f"训练集已发布历史标签：{record.bridge_name}，{record.cycle_year}年检测，"
            f"总体评分 {record.overall_score}，总体等级 {record.overall_grade}。"
        )
        state.add_evidence(
            EvidenceRef(
                id=evidence_id,
                kind="derived",
                text=evidence_text,
                source_path=str(record.label.path),
                priority=76,
            )
        )
        historical_score = _history_score_value(record.overall_score)
        if state.scores.previous.score.value in (None, "") and historical_score is not None:
            state.scores.previous.score = FieldValue(
                value=historical_score,
                evidence=[evidence_id],
                source_priority=76,
                confidence=min(0.98, similarity),
            )
        if state.scores.previous.grade.value in (None, "") and record.overall_grade:
            state.scores.previous.grade = FieldValue(
                value=record.overall_grade,
                evidence=[evidence_id],
                source_priority=76,
                confidence=min(0.98, similarity),
            )

        matches, new_indices, _ = self.matcher.match(state.diseases, record.diseases)
        exact_pairs = {(item.current_index, item.previous_index) for item in matches}
        for current_index, current in enumerate(state.diseases):
            probabilistic_links = []
            for previous_index, previous in enumerate(record.diseases):
                if not self.matcher.compatible(current, previous):
                    continue
                probability = self.matcher.score(current, previous)
                if probability < self.matcher.threshold:
                    continue
                probabilistic_links.append(DiseaseHistoryLink(
                    previous_cycle=record.cycle_year,
                    previous_index=previous_index,
                    relation="exact" if (current_index, previous_index) in exact_pairs else "candidate",
                    probability=round(probability, 6),
                    evidence=[evidence_id],
                ))
            current.history_links = sorted(
                probabilistic_links,
                key=lambda item: item.probability,
                reverse=True,
            )[:3]
        for item in matches:
            current = state.diseases[item.current_index]
            previous = record.diseases[item.previous_index]
            disease_evidence_id = f"{evidence_id}:D{item.previous_index}"
            if disease_evidence_id not in state.evidence:
                state.add_evidence(
                    EvidenceRef(
                        id=disease_evidence_id,
                        kind="derived",
                        text=(
                            f"{record.cycle_year}年历史病害：部位={previous.location or previous.component}；"
                            f"类型={previous.type}；描述={previous.description or '未明确'}。"
                        ),
                        source_path=str(record.label.path),
                        priority=76,
                    )
                )
            current.evidence = list(dict.fromkeys([*current.evidence, disease_evidence_id]))
            if current.is_new is None:
                current.is_new = False
                current.novelty_source = "history_exact"
            if not current.previous_state or current.previous_state in {"未明确", "未提及"}:
                # Explicit current-report novelty evidence outranks a fuzzy
                # cross-cycle row match.  Keep the three fields coherent.
                current.previous_state = "未发现" if current.is_new is True else "已发现"
            if not current.development or current.development in {"未明确", "未提及"}:
                current.development = self._development(current, previous)
        grouped_indices: set[int] = set()
        # A prior report may publish one aggregate issue-family row while the
        # current report exposes dozens of photo-ledger observations.  Strict
        # one-to-one assignment would call almost every observation "new".
        # Activate reusable family matching only under an observable
        # granularity shift, and never override explicit current-report flags.
        granular_shift = bool(record.diseases) and len(state.diseases) >= max(20, len(record.diseases) * 2)
        if granular_shift:
            for index in new_indices:
                current = state.diseases[index]
                if current.is_new is not None:
                    continue
                group_match = self.matcher.best_group_match(current, record.diseases)
                if group_match is None:
                    continue
                previous_index, group_score = group_match
                previous = record.diseases[previous_index]
                disease_evidence_id = f"{evidence_id}:GROUP:{previous_index}"
                if disease_evidence_id not in state.evidence:
                    state.add_evidence(
                        EvidenceRef(
                            id=disease_evidence_id,
                            kind="derived",
                            text=(
                                f"{record.cycle_year}年历史病害族：部位={previous.location or previous.component}；"
                                f"类型={previous.type}；描述={previous.description or '未明确'}。"
                            ),
                            source_path=str(record.label.path),
                            priority=72,
                        )
                    )
                current.evidence = list(dict.fromkeys([*current.evidence, disease_evidence_id]))
                current.is_new = False
                current.novelty_source = "history_group"
                if not current.previous_state or current.previous_state in {"未明确", "未提及"}:
                    current.previous_state = "已发现"
                if not current.development or current.development in {"未明确", "未提及"}:
                    current.development = self._development(current, previous)
                current.confidence = max(current.confidence, min(0.92, group_score))
                current.history_links.append(DiseaseHistoryLink(
                    previous_cycle=record.cycle_year,
                    previous_index=previous_index,
                    relation="group",
                    probability=round(group_score, 6),
                    evidence=[disease_evidence_id],
                ))
                grouped_indices.add(index)
        for index in new_indices:
            if index in grouped_indices:
                continue
            current = state.diseases[index]
            # No fuzzy candidate is an uncertain association, not evidence of
            # physical absence or novelty. Preserve explicit current-report facts.
            if current.previous_state in (None, "", "未提及", "未明确"):
                current.previous_state = ("已发现" if current.is_new is False else
                                          "未发现" if current.is_new is True else "未明确")
            if current.development in (None, "", "未提及"):
                current.development = "无" if current.is_new is not None else "未明确"

        # Keep enum fields internally coherent after combining direct report
        # evidence with historical matches.  At row level, "新增" means that
        # exact observation was absent in the prior cycle even when its wider
        # disease family existed elsewhere on the component.
        for current in state.diseases:
            if (
                current.is_new is True
                and current.previous_state
                and "已发现" in current.previous_state
                and "未发现" not in current.previous_state
            ):
                current.previous_state = "未发现"

        current_score = _numeric(state.scores.overall.score.value)
        previous_score = _numeric(state.scores.previous.score.value)
        if current_score is not None and previous_score is not None and not state.narrative.development_trend:
            delta = current_score - previous_score
            direction = "上升" if delta > 0 else "下降" if delta < 0 else "持平"
            state.narrative.development_trend = (
                f"与{record.cycle_year}年上一次定检相比，总体评分{direction}{abs(delta):.2f}分；"
                f"本次历史候选匹配{len(matches)}处，未建立历史关联{len(new_indices)}处。"
            )
        return {
            "matched": True,
            "bridge": record.bridge_name,
            "history_year": record.cycle_year,
            "explicit_prior_year": prior_year,
            "similarity": similarity,
            "disease_matches": len(matches),
            "disease_group_matches": len(grouped_indices),
            "new_diseases": sum(item.is_new is True for item in state.diseases),
            "evidence_id": evidence_id,
        }

    def _match_record(
        self,
        bridge_name: str,
        records: list[HistoryRecord],
        prior_year: int | None = None,
    ) -> tuple[HistoryRecord, float] | None:
        current_identity = _identity(bridge_name)
        current_semantic = _semantic_name(bridge_name)
        current_codes = _route_codes(bridge_name)
        current_ramps = _ramp_ids(bridge_name)
        current_types = _structure_types(bridge_name)
        scored: list[tuple[float, HistoryRecord]] = []
        for record in records:
            record_codes = _route_codes(record.bridge_name)
            if current_codes and record_codes and current_codes.isdisjoint(record_codes):
                continue
            record_ramps = _ramp_ids(record.bridge_name)
            if current_ramps and record_ramps and current_ramps.isdisjoint(record_ramps):
                continue
            record_types = _structure_types(record.bridge_name)
            if current_types and record_types and current_types.isdisjoint(record_types):
                continue
            candidate_identity = _identity(record.bridge_name)
            candidate_semantic = _semantic_name(record.bridge_name)
            if current_identity == candidate_identity:
                score = 1.0
            elif (
                current_semantic == candidate_semantic
                and len(current_semantic) >= 4
                and (current_semantic != "无名桥" or bool(current_codes & record_codes))
            ):
                score = 0.99
            elif current_identity in candidate_identity or candidate_identity in current_identity:
                score = 0.96 * min(len(current_identity), len(candidate_identity)) / max(len(current_identity), len(candidate_identity)) + 0.04
            else:
                score = fuzz.WRatio(current_identity, candidate_identity) / 100.0
            scored.append((score, record))
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], item[1].cycle_year), reverse=True)
        best_score = scored[0][0]
        candidates = [item for item in scored if item[0] >= max(0.72, best_score - 0.08)][:20]
        if not candidates:
            return None

        # Use the released embedding/reranker models only for genuinely
        # ambiguous names. Exact identities stay deterministic and free.
        if self.client and best_score < 0.995 and (len(candidates) > 1 or best_score < 0.90):
            query = f"当前桥梁：{bridge_name}"
            documents = [item[1].descriptor for item in candidates]
            vectors = self.client.embed_texts([query, *documents], task="history-bridge-embedding")
            if len(vectors) == len(documents) + 1:
                query_vector = vectors[0]
                combined = []
                query_norm = sum(value * value for value in query_vector) ** 0.5 or 1.0
                for (fuzzy_score, record), vector in zip(candidates, vectors[1:]):
                    norm = sum(value * value for value in vector) ** 0.5 or 1.0
                    cosine = sum(left * right for left, right in zip(query_vector, vector)) / (query_norm * norm)
                    combined.append((0.55 * fuzzy_score + 0.45 * max(0.0, cosine), record))
                combined.sort(key=lambda item: (item[0], item[1].cycle_year), reverse=True)
                candidates = combined[:10]
            reranked = self.client.rerank_documents(
                query=f"判断与“{bridge_name}”是否同一座桥，桩号、立交编号和方向必须一致",
                documents=[item[1].descriptor for item in candidates],
                top_n=1,
                task="history-bridge-rerank",
            )
            if reranked:
                selected_index = int(reranked[0].get("index", -1))
                if 0 <= selected_index < len(candidates):
                    selected = candidates[selected_index]
                    candidates = [(max(selected[0], min(1.0, reranked[0]["score"])), selected[1])]

        score, record = candidates[0]
        if score < 0.90:
            return None
        # Prefer the matching prior year cycle if evidenced, or the most recent eligible cycle among equivalent identities.
        equivalent = [item for item in scored if item[0] >= max(0.94, score - 0.01)]
        if equivalent:
            if prior_year is not None:
                py_matches = [item for item in equivalent if prior_year in _record_years(item[1])]
                if py_matches:
                    score, record = max(py_matches, key=lambda item: item[1].cycle_year)
                else:
                    score, record = max(equivalent, key=lambda item: item[1].cycle_year)
            else:
                score, record = max(equivalent, key=lambda item: item[1].cycle_year)
        return record, score

    @staticmethod
    def _development(current: Disease, previous: Disease) -> str:
        comparison_text = f"{current.development or ''} {current.description}"
        return development_level(comparison_text)
