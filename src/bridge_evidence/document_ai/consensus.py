from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation

from rapidfuzz import fuzz

from .schema import (
    CandidateConflict,
    CandidateValue,
    ConsensusResult,
    DocumentRegion,
    ResolvedCandidate,
)


class ConsensusResolver:
    def __init__(
        self,
        *,
        min_agreement: int = 2,
        min_confidence: float = 0.80,
        min_margin: float = 0.15,
    ):
        if min_agreement < 1:
            raise ValueError("min_agreement must be at least 1")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if not 0.0 <= min_margin <= 1.0:
            raise ValueError("min_margin must be between 0 and 1")
        self.min_agreement = min_agreement
        self.min_confidence = min_confidence
        self.min_margin = min_margin

    @staticmethod
    def normalize_value(field: str, value: object) -> str:
        text = " ".join(str(value if value is not None else "").split())
        if not text:
            return ""
        if field.endswith(".score"):
            match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
            if match:
                try:
                    decimal = Decimal(match.group(0)).normalize()
                    normalized = format(decimal, "f")
                    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized
                except InvalidOperation:
                    pass
        if field.endswith(".grade"):
            match = re.search(r"\b([A-E])\b|([A-E])\s*(?:级|类|GRADE)", text.upper())
            if match:
                return f"{next(group for group in match.groups() if group)}级"
            chinese = re.search(r"([一二三四五])\s*(?:级|类)", text)
            if chinese:
                return f"{chinese.group(1)}级"
        if "date" in field:
            return re.sub(r"[\s./-]+", "", text)
        return re.sub(r"\s+", "", text).casefold()

    def resolve(self, candidates: list[CandidateValue]) -> ConsensusResult:
        by_field: dict[str, list[CandidateValue]] = defaultdict(list)
        for candidate in candidates:
            if candidate.confidence >= self.min_confidence:
                by_field[candidate.field].append(candidate)

        result = ConsensusResult()
        for field in sorted(by_field):
            grouped: dict[str, dict[str, CandidateValue]] = defaultdict(dict)
            for candidate in by_field[field]:
                normalized = self.normalize_value(field, candidate.value)
                if not normalized:
                    continue
                previous = grouped[normalized].get(candidate.expert)
                if previous is None or candidate.confidence > previous.confidence:
                    grouped[normalized][candidate.expert] = candidate

            if not grouped:
                result.conflicts.append(CandidateConflict(
                    field=field,
                    reason="no_valid_candidate",
                ))
                continue

            ranked = sorted(
                grouped.items(),
                key=lambda item: (
                    sum(candidate.confidence * candidate.weight for candidate in item[1].values()),
                    len(item[1]),
                    item[0],
                ),
                reverse=True,
            )
            winner_value, winner_by_expert = ranked[0]
            winner_candidates = list(winner_by_expert.values())
            winner_score = sum(candidate.confidence * candidate.weight for candidate in winner_candidates)
            runner_score = (
                sum(candidate.confidence * candidate.weight for candidate in ranked[1][1].values())
                if len(ranked) > 1
                else 0.0
            )
            margin = (winner_score - runner_score) / winner_score if winner_score else 0.0
            enough_agreement = len(winner_by_expert) >= self.min_agreement
            enough_margin = margin >= self.min_margin
            abstained = not enough_agreement or not enough_margin
            representative = max(winner_candidates, key=lambda candidate: candidate.confidence)
            confidence = min(
                1.0,
                sum(candidate.confidence for candidate in winner_candidates) / len(winner_candidates),
            )
            resolved = ResolvedCandidate(
                field=field,
                value=representative.value,
                normalized_value=winner_value,
                confidence=confidence,
                contributors=sorted(winner_by_expert),
                evidence_texts=sorted({
                    candidate.evidence_text
                    for candidate in winner_candidates
                    if candidate.evidence_text
                }),
                region_ids=sorted({
                    candidate.region_id
                    for candidate in winner_candidates
                    if candidate.region_id
                }),
                abstained=abstained,
            )
            result.resolved[field] = resolved

            alternatives = sorted(value for value, _ in ranked[1:])
            if abstained:
                reason = "insufficient_agreement" if not enough_agreement else "insufficient_margin"
                result.conflicts.append(CandidateConflict(
                    field=field,
                    winner=winner_value,
                    alternatives=alternatives,
                    reason=reason,
                ))
            elif alternatives:
                result.conflicts.append(CandidateConflict(
                    field=field,
                    winner=winner_value,
                    alternatives=alternatives,
                    reason="alternative_values",
                ))
        return result


def _normalized_table_signature(
    region: DocumentRegion,
) -> tuple[str, tuple[tuple[str, ...], ...]] | None:
    if not region.rows:
        return None
    rows = tuple(
        tuple("".join(str(cell).split()).casefold() for cell in row)
        for row in region.rows
    )
    if not any(any(cell for cell in row) for row in rows):
        return None
    return str(region.kind), rows


def accepted_table_region_ids(
    regions_by_expert: dict[str, list[DocumentRegion]],
    *,
    min_agreement: int = 2,
) -> set[str]:
    """Select one table copy only after distinct experts agree on all rows."""

    grouped: list[list[tuple[str, DocumentRegion]]] = []
    for expert, regions in regions_by_expert.items():
        for region in regions:
            signature = _normalized_table_signature(region)
            if signature is None:
                continue
            best_group = None
            best_score = 0.0
            for group in grouped:
                if any(existing_expert == expert for existing_expert, _ in group):
                    continue
                score = _table_region_similarity(region, group[0][1])
                if score > best_score:
                    best_score = score
                    best_group = group
            if best_group is not None and best_score >= 0.90:
                best_group.append((expert, region))
            else:
                grouped.append([(expert, region)])

    accepted: set[str] = set()
    required = max(2, min_agreement)
    for matches in grouped:
        if len({expert for expert, _ in matches}) < required:
            continue
        representative = max(matches, key=lambda item: item[1].confidence)[1]
        accepted.add(representative.id)
    return accepted


def _table_region_similarity(left: DocumentRegion, right: DocumentRegion) -> float:
    if left.kind != right.kind or not left.rows or not right.rows:
        return 0.0
    if abs(len(left.rows) - len(right.rows)) > max(1, round(max(len(left.rows), len(right.rows)) * 0.2)):
        return 0.0
    left_header = "|".join("".join(str(cell).split()) for cell in left.rows[0])
    right_header = "|".join("".join(str(cell).split()) for cell in right.rows[0])
    header_score = fuzz.WRatio(left_header, right_header) / 100.0
    if header_score < 0.86:
        return 0.0
    right_rows = ["|".join("".join(str(cell).split()) for cell in row) for row in right.rows[1:]]
    unused = set(range(len(right_rows)))
    row_scores = []
    for row in left.rows[1:]:
        if not unused:
            break
        value = "|".join("".join(str(cell).split()) for cell in row)
        match = max(unused, key=lambda index: fuzz.WRatio(value, right_rows[index]))
        row_scores.append(fuzz.WRatio(value, right_rows[match]) / 100.0)
        unused.remove(match)
    denominator = max(len(left.rows) - 1, len(right.rows) - 1, 1)
    body_score = sum(row_scores) / denominator
    return 0.35 * header_score + 0.65 * body_score
