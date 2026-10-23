from __future__ import annotations

import re
from dataclasses import dataclass

import networkx as nx
import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from bridge_evidence.contest.facts import identity_conflicts, normalize_identity
from bridge_evidence.models.schema import Disease


def _normalize(text: str) -> str:
    value = str(text or "").lower()
    value = value.replace("号", "#").replace("＃", "#")
    value = value.replace("左线", "左幅").replace("右线", "右幅")
    return re.sub(r"[\s，,。；;：:（）()×xX]", "", value)


def _similarity(left: str, right: str, *, scorer=fuzz.WRatio) -> float:
    left, right = _normalize(left), _normalize(right)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return scorer(left, right) / 100.0


def _side(text: str) -> str | None:
    normalized = _normalize(text)
    if "左幅" in normalized:
        return "左幅"
    if "右幅" in normalized:
        return "右幅"
    return None


_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("裂缝", re.compile(r"裂缝|开裂|网裂|龟裂|裂痕")),
    ("破损", re.compile(r"破损|剥落|掉角|缺损|坑槽|坑洞")),
    ("露筋", re.compile(r"露筋|钢筋裸露")),
    ("锈蚀", re.compile(r"锈蚀|锈胀|腐蚀|锈迹")),
    ("渗水", re.compile(r"渗水|漏水|水迹")),
    ("泛碱", re.compile(r"泛碱|盐析")),
    ("蜂窝麻面", re.compile(r"蜂窝|麻面|不密实")),
    # Cable sheathing reports alternate between 裂痕/划痕/刮伤 while
    # referring to the same surface-damage family.  Keep 裂痕 in the crack
    # family too; component compatibility prevents cross-structure matches.
    ("表面损伤", re.compile(r"裂痕|划痕|刮伤|擦伤")),
    ("缺失", re.compile(r"缺失|缺损")),
    ("松动", re.compile(r"松动")),
    ("变形", re.compile(r"变形|错位|位移")),
    ("积水", re.compile(r"积水")),
    ("污渍", re.compile(r"污渍|污染")),
    ("堵塞", re.compile(r"堵塞")),
    ("脱落", re.compile(r"脱落|脱漆")),
    ("老化", re.compile(r"老化|劣化")),
    ("空洞", re.compile(r"空洞")),
    ("黄油异常", re.compile(r"黄油")),
)


def _families(text: str) -> set[str]:
    return {name for name, pattern in _FAMILY_PATTERNS if pattern.search(str(text or ""))}


def _family_overlap(left: str, right: str) -> float:
    left_families = _families(left)
    right_families = _families(right)
    if not left_families or not right_families:
        return 0.0
    # Historical rows are often aggregate families while current rows are
    # individual photo-ledger observations.  The overlap coefficient preserves
    # that subset relation; Jaccard would incorrectly punish the aggregate.
    return len(left_families & right_families) / min(len(left_families), len(right_families))


@dataclass(frozen=True)
class DiseaseMatch:
    current_index: int
    previous_index: int
    score: float


class DiseaseMatcher:
    """Build a disease bipartite graph and solve a global one-to-one assignment.

    RapidFuzz supplies domain-tolerant string similarities; SciPy's Hungarian
    algorithm prevents a locally attractive edge from consuming the only good
    partner of a second disease.  NetworkX is retained as the explicit graph
    representation for diagnostics and complexity-science metrics.
    """

    def __init__(self, threshold: float = 0.56):
        self.threshold = threshold

    @staticmethod
    def compatible(current: Disease, previous: Disease) -> bool:
        if current.component != previous.component and current.component != "其他" and previous.component != "其他":
            return False
        current_side = _side(f"{current.location}{current.description}")
        previous_side = _side(f"{previous.location}{previous.description}")
        if current_side and previous_side and current_side != previous_side:
            return False
        left = normalize_identity('', f'{current.location} {current.description}', current.component)
        right = normalize_identity('', f'{previous.location} {previous.description}', previous.component)
        return not identity_conflicts(left, right)

    @staticmethod
    def score(current: Disease, previous: Disease) -> float:
        if not DiseaseMatcher.compatible(current, previous):
            return 0.0
        location = _similarity(
            f"{current.component}{current.location}",
            f"{previous.component}{previous.location}",
        )
        disease_type = max(
            _similarity(current.type, previous.type, scorer=fuzz.token_set_ratio),
            _family_overlap(current.type, previous.type),
        )
        metrics = _similarity(
            "|".join(f"{key}={value}" for key, value in sorted(current.metrics.items())),
            "|".join(f"{key}={value}" for key, value in sorted(previous.metrics.items())),
        )
        description = _similarity(current.description, previous.description, scorer=fuzz.token_set_ratio)
        return 0.35 * location + 0.30 * disease_type + 0.20 * metrics + 0.15 * description

    @classmethod
    def group_score(cls, current: Disease, previous: Disease) -> float:
        """Score a granular row against a reusable historical issue family.

        This is deliberately separate from the one-to-one score.  It is used
        only when the current ledger is much more granular than the prior
        published issue layer, where a single historical family may explain
        several current observations.
        """
        if not cls.compatible(current, previous):
            return 0.0
        type_score = max(
            _family_overlap(current.type, previous.type),
            _similarity(current.type, previous.type, scorer=fuzz.token_set_ratio),
        )
        if type_score < 0.72:
            return 0.0
        component_score = 1.0 if current.component == previous.component else 0.62
        location_score = _similarity(
            f"{current.component}{current.location}",
            f"{previous.component}{previous.location}",
        )
        description_score = max(
            _family_overlap(current.description, f"{previous.type}{previous.description}"),
            _similarity(current.description, previous.description, scorer=fuzz.token_set_ratio),
        )
        return 0.50 * type_score + 0.25 * component_score + 0.15 * location_score + 0.10 * description_score

    def best_group_match(self, current: Disease, previous: list[Disease]) -> tuple[int, float] | None:
        scored = [(index, self.group_score(current, item)) for index, item in enumerate(previous)]
        if not scored:
            return None
        index, score = max(scored, key=lambda item: item[1])
        return (index, score) if score >= 0.78 else None

    def build_graph(self, current: list[Disease], previous: list[Disease]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from((f"c:{index}", {"partition": "current", "disease": disease}) for index, disease in enumerate(current))
        graph.add_nodes_from((f"p:{index}", {"partition": "previous", "disease": disease}) for index, disease in enumerate(previous))
        for current_index, left in enumerate(current):
            for previous_index, right in enumerate(previous):
                if not self.compatible(left, right):
                    continue
                value = self.score(left, right)
                if value >= self.threshold:
                    graph.add_edge(f"c:{current_index}", f"p:{previous_index}", weight=value)
        return graph

    def match(self, current: list[Disease], previous: list[Disease]) -> tuple[list[DiseaseMatch], list[int], list[int]]:
        if not current or not previous:
            return [], list(range(len(current))), list(range(len(previous)))

        current_count = len(current)
        previous_count = len(previous)
        size = current_count + previous_count
        forbidden = 1_000_000.0
        costs = np.full((size, size), forbidden, dtype=float)
        scores = np.zeros((current_count, previous_count), dtype=float)

        for current_index, left in enumerate(current):
            for previous_index, right in enumerate(previous):
                if not self.compatible(left, right):
                    continue
                value = self.score(left, right)
                scores[current_index, previous_index] = value
                costs[current_index, previous_index] = 1.0 - value

        # One unmatched current disease and one unmatched previous disease cost
        # together exactly ``1 - threshold``.  Consequently a real edge is
        # selected only when its score clears the configured threshold.
        unmatched_cost = (1.0 - self.threshold) / 2.0
        for current_index in range(current_count):
            costs[current_index, previous_count + current_index] = unmatched_cost
        for previous_index in range(previous_count):
            costs[current_count + previous_index, previous_index] = unmatched_cost
        costs[current_count:, previous_count:] = 0.0

        row_indices, column_indices = linear_sum_assignment(costs)
        matches: list[DiseaseMatch] = []
        used_current: set[int] = set()
        used_previous: set[int] = set()
        for row, column in zip(row_indices.tolist(), column_indices.tolist()):
            if row >= current_count or column >= previous_count:
                continue
            value = float(scores[row, column])
            if value < self.threshold:
                continue
            matches.append(DiseaseMatch(row, column, value))
            used_current.add(row)
            used_previous.add(column)

        matches.sort(key=lambda item: item.current_index)
        new_current = [index for index in range(current_count) if index not in used_current]
        disappeared_previous = [index for index in range(previous_count) if index not in used_previous]
        return matches, new_current, disappeared_previous

    def diagnostics(self, current: list[Disease], previous: list[Disease]) -> dict[str, float | int]:
        graph = self.build_graph(current, previous)
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        return {
            "nodes": node_count,
            "edges": edge_count,
            "components": nx.number_connected_components(graph) if node_count else 0,
            "density": nx.density(graph) if node_count > 1 else 0.0,
        }
