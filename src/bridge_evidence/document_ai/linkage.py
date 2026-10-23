from __future__ import annotations

import re

from rapidfuzz import fuzz

from bridge_evidence.models.schema import Disease


def _compact(value: str) -> str:
    value = re.sub(r"照片\s*\d+\s*[:：]?", "", str(value or ""))
    return re.sub(r"[\s,，。；;:：、()（）]+", "", value).casefold()


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return fuzz.WRatio(_compact(left), _compact(right)) / 100.0


def _side_and_span(location: str) -> tuple[str | None, str | None]:
    side_match = re.search(r"(左幅|右幅)", location)
    span_match = re.search(r"(\d+)\s*(?:#|号)?跨", location)
    return (
        side_match.group(1) if side_match else None,
        span_match.group(1) if span_match else None,
    )


def _is_photo_ledger_record(disease: Disease) -> bool:
    return (
        bool(re.search(r"(?:照片|相片|附图)\s*\d+", disease.description))
        or any(str(evidence).upper().startswith("IMG:") for evidence in disease.evidence)
    )


def _photo_aggregation_key(
    disease: Disease,
) -> tuple[str, str, str, str] | None:
    side, span = _side_and_span(disease.location)
    if not side or not span or not _is_photo_ledger_record(disease):
        return None
    component = _compact(disease.component)
    disease_type = _compact(disease.type)
    if not component or not disease_type:
        return None
    return side, span, component, disease_type


class ProbabilisticDiseaseLinker:
    def __init__(self, *, threshold: float = 0.90):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold

    def match_probability(self, left: Disease, right: Disease) -> float:
        left_side, left_span = _side_and_span(left.location)
        right_side, right_span = _side_and_span(right.location)
        if left_side and right_side and left_side != right_side:
            return 0.0
        if left_span and right_span and left_span != right_span:
            return 0.0

        component = _similarity(left.component, right.component)
        disease_type = _similarity(left.type, right.type)
        location = _similarity(left.location, right.location)
        description = _similarity(left.description, right.description)
        if component < 0.80 or disease_type < 0.85 or location < 0.80:
            return 0.0
        return (
            0.20 * component
            + 0.25 * disease_type
            + 0.35 * location
            + 0.20 * description
        )

    def deduplicate(self, diseases: list[Disease]) -> list[Disease]:
        working = self._aggregate_photo_ledger([
            item.model_copy(deep=True) for item in diseases
        ])
        if len(working) < 2:
            return working
        parents = list(range(len(working)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for left in range(len(working)):
            for right in range(left + 1, len(working)):
                if self.match_probability(working[left], working[right]) >= self.threshold:
                    union(left, right)

        groups: dict[int, list[Disease]] = {}
        for index, disease in enumerate(working):
            groups.setdefault(find(index), []).append(disease)

        merged: list[Disease] = []
        for group in groups.values():
            representative = max(
                group,
                key=lambda item: (item.confidence, len(item.description), len(item.evidence)),
            ).model_copy(deep=True)
            representative.evidence = sorted({evidence for item in group for evidence in item.evidence})
            representative.metrics = {
                key: value
                for item in group
                for key, value in item.metrics.items()
            }
            representative.confidence = max(item.confidence for item in group)
            merged.append(representative)
        return sorted(merged, key=lambda item: _sequence_key(item.sequence))

    @staticmethod
    def _aggregate_photo_ledger(diseases: list[Disease]) -> list[Disease]:
        groups: dict[tuple[str, str, str, str], list[int]] = {}
        for index, disease in enumerate(diseases):
            key = _photo_aggregation_key(disease)
            if key is not None:
                groups.setdefault(key, []).append(index)

        consumed: set[int] = set()
        result: list[Disease] = []
        for index, disease in enumerate(diseases):
            if index in consumed:
                continue
            key = _photo_aggregation_key(disease)
            group_indices = groups.get(key, []) if key is not None else []
            if len(group_indices) < 3:
                result.append(disease)
                continue

            group = [diseases[group_index] for group_index in group_indices]
            consumed.update(group_indices)
            representative = max(
                group,
                key=lambda item: (
                    item.confidence,
                    len(item.description),
                    len(item.evidence),
                ),
            ).model_copy(deep=True)
            side, span, _, _ = key
            representative.sequence = min(
                (item.sequence for item in group),
                key=_sequence_key,
            )
            representative.location = f"{side}{span}跨{representative.component}"
            details: list[str] = []
            seen_details: set[str] = set()
            for item in sorted(group, key=lambda value: _sequence_key(value.sequence)):
                detail = re.sub(
                    r"^(?:照片|相片|附图)\s*\d+\s*[:：]?",
                    "",
                    item.description,
                ).strip()
                normalized = _compact(detail)
                if detail and normalized not in seen_details:
                    details.append(detail)
                    seen_details.add(normalized)
            prefix = (
                f"{side}{span}跨{representative.component}"
                f"{representative.type}共{len(group)}处"
            )
            representative.description = (
                f"{prefix}：" + "；".join(details)
                if details
                else prefix
            )
            representative.evidence = sorted({
                evidence
                for item in group
                for evidence in item.evidence
            })
            representative.metrics = {
                key: value
                for item in group
                for key, value in item.metrics.items()
            }
            representative.metrics["聚合数量"] = str(len(group))
            representative.confidence = max(item.confidence for item in group)
            result.append(representative)
        return result


def _sequence_key(sequence: str) -> tuple[int, str]:
    match = re.search(r"\d+", str(sequence))
    return (int(match.group(0)) if match else 10**9, str(sequence))
