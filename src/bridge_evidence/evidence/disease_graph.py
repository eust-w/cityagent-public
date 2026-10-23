from __future__ import annotations

import re
from hashlib import sha256

from bridge_evidence.contest.facts import identity_conflicts, normalize_identity
from bridge_evidence.models.schema import Disease, DiseaseEvent, DocumentState

_FAMILIES = (
    ("裂缝", re.compile(r"裂缝|开裂|网裂|龟裂")),
    ("破损", re.compile(r"破损|剥落|掉角|缺损|坑槽")),
    ("露筋", re.compile(r"露筋|钢筋裸露")),
    ("锈蚀", re.compile(r"锈蚀|腐蚀|锈胀")),
    ("渗水", re.compile(r"渗水|漏水|水迹|泛碱")),
    ("变形", re.compile(r"变形|位移|错位|脱空")),
    ("缺失", re.compile(r"缺失|脱落|松动")),
)


def _side(text: str) -> str:
    if "左幅" in text or "左线" in text:
        return "left"
    if "右幅" in text or "右线" in text:
        return "right"
    return "unknown"


def _span(text: str) -> str | None:
    match = re.search(r"(?:第\s*)?(\d+)\s*[#号]?\s*跨", text)
    return f"{match.group(1)}跨" if match else None


def _family(text: str) -> str:
    return next((name for name, pattern in _FAMILIES if pattern.search(text)), text.strip() or "未明确")


def published_diseases(state: DocumentState) -> list[Disease]:
    """Return atomic report rows while inheriting aggregate history semantics."""
    if not state.disease_events:
        return list(state.diseases)

    parents_by_event: dict[str, list[Disease]] = {}
    for disease in state.diseases:
        for event_id in disease.event_ids:
            parents_by_event.setdefault(event_id, []).append(disease)

    result: list[Disease] = []
    for index, event in enumerate(state.disease_events, start=1):
        candidates = parents_by_event.get(event.event_id, [])
        parent = max(
            candidates,
            key=lambda disease: (
                len(set(event.evidence) & set(disease.evidence)),
                disease.confidence,
                -len(disease.event_ids),
            ),
            default=None,
        )
        result.append(Disease(
            sequence=str(index),
            component=event.component,
            location=event.location,
            type=event.disease_type,
            description=event.description,
            metrics=dict(event.measurements),
            is_new=parent.is_new if parent else None,
            novelty_source=parent.novelty_source if parent else None,
            previous_state=parent.previous_state if parent else None,
            development=parent.development if parent else None,
            evidence=list(event.evidence),
            event_ids=[event.event_id],
            history_links=list(event.history_links or (parent.history_links if parent else [])),
            confidence=event.confidence,
        ))
    return result


class DiseaseGraphBuilder:
    """Preserve atomic observations and link them to published aggregate rows."""

    @staticmethod
    def build(diseases: list[Disease], document_id: str) -> list[DiseaseEvent]:
        events = []
        for index, disease in enumerate(diseases):
            identity = "|".join((
                document_id,
                str(index),
                disease.sequence,
                disease.component,
                disease.location,
                disease.type,
                disease.description,
            ))
            events.append(DiseaseEvent(
                event_id=f"DE:{sha256(identity.encode('utf-8')).hexdigest()[:20]}",
                sequence=disease.sequence,
                component=disease.component,
                location=disease.location,
                side=_side(f"{disease.location}{disease.description}"),
                span=_span(f"{disease.location}{disease.description}"),
                disease_type=disease.type,
                description=disease.description,
                measurements=dict(disease.metrics),
                evidence=list(disease.evidence),
                history_links=list(disease.history_links),
                confidence=disease.confidence,
            ))
        return events

    @staticmethod
    def attach_aggregates(diseases: list[Disease], events: list[DiseaseEvent]) -> None:
        for disease in diseases:
            side = _side(f"{disease.location}{disease.description}")
            span = _span(f"{disease.location}{disease.description}")
            family = _family(f"{disease.type}{disease.description}")
            compatible = [
                event
                for event in events
                if event.component == disease.component
                and _family(f"{event.disease_type}{event.description}") == family
                and (side == "unknown" or event.side == "unknown" or event.side == side)
                and (span is None or event.span is None or event.span == span)
                and not identity_conflicts(
                    normalize_identity('', f'{disease.location} {disease.description}', disease.component),
                    normalize_identity('', f'{event.location} {event.description}', event.component),
                )
            ]
            disease.event_ids = list(dict.fromkeys(event.event_id for event in compatible))

    @staticmethod
    def propagate_history(state: DocumentState) -> None:
        links_by_event = {
            event_id: list(disease.history_links)
            for disease in state.diseases
            for event_id in disease.event_ids
        }
        for event in state.disease_events:
            if event.event_id in links_by_event:
                event.history_links = links_by_event[event.event_id]

    @staticmethod
    def link_recommendations(state: DocumentState) -> None:
        for recommendation in state.recommendations:
            text = f"{recommendation.component}{recommendation.content}"
            family = _family(text)
            matches = [
                event
                for event in state.disease_events
                if (
                    recommendation.component in {"全桥", "桥梁总体", event.component}
                    or event.component in text
                )
                and (family == _family(event.disease_type) or family == text.strip())
            ]
            if not matches:
                matches = [
                    event
                    for event in state.disease_events
                    if event.component in text or recommendation.component == event.component
                ]
            recommendation.disease_event_ids = list(dict.fromkeys(event.event_id for event in matches))
            if matches:
                recommendation.policy_basis = "基于病害构件、类型和处置动作的证据关联"
