from __future__ import annotations

import unittest
from unittest.mock import patch

from bridge_evidence.matching import DiseaseMatcher
from bridge_evidence.models.schema import Disease


class DiseaseMatcherTest(unittest.TestCase):
    def test_matches_same_location_and_type(self) -> None:
        current = [Disease(component="主梁", location="1#梁跨中", type="裂缝", description="1#梁跨中裂缝，L=0.5m", metrics={"L": "0.5m"})]
        previous = [Disease(component="主梁", location="1号梁跨中", type="裂缝", description="1号梁跨中存在裂缝，L=0.4m", metrics={"L": "0.4m"})]
        matches, new, missing = DiseaseMatcher(threshold=0.45).match(current, previous)
        self.assertEqual(len(matches), 1)
        self.assertEqual(new, [])
        self.assertEqual(missing, [])

    def test_different_components_do_not_match(self) -> None:
        current = [Disease(component="主梁", location="跨中", type="裂缝", description="裂缝")]
        previous = [Disease(component="桥台", location="跨中", type="裂缝", description="裂缝")]
        matches, new, missing = DiseaseMatcher(threshold=0.1).match(current, previous)
        self.assertEqual(matches, [])
        self.assertEqual(new, [0])
        self.assertEqual(missing, [0])

    def test_uses_global_assignment_instead_of_greedy_edges(self) -> None:
        current = [
            Disease(component="主梁", type="裂缝", description="c0"),
            Disease(component="主梁", type="裂缝", description="c1"),
        ]
        previous = [
            Disease(component="主梁", type="裂缝", description="p0"),
            Disease(component="主梁", type="裂缝", description="p1"),
        ]
        scores = {
            ("c0", "p0"): 0.90,
            ("c0", "p1"): 0.80,
            ("c1", "p0"): 0.85,
            ("c1", "p1"): 0.10,
        }
        with patch.object(DiseaseMatcher, "score", side_effect=lambda left, right: scores[(left.description, right.description)]):
            matches, new, missing = DiseaseMatcher(threshold=0.50).match(current, previous)
        self.assertEqual([(item.current_index, item.previous_index) for item in matches], [(0, 1), (1, 0)])
        self.assertEqual(new, [])
        self.assertEqual(missing, [])

    def test_exposes_graph_diagnostics(self) -> None:
        current = [Disease(component="主梁", location="1#梁", type="裂缝", description="裂缝")]
        previous = [Disease(component="主梁", location="1号梁", type="裂缝", description="裂缝")]
        diagnostics = DiseaseMatcher(threshold=0.20).diagnostics(current, previous)
        self.assertEqual(diagnostics["nodes"], 2)
        self.assertEqual(diagnostics["edges"], 1)

    def test_group_match_maps_granular_family_to_compound_historical_row(self) -> None:
        current = Disease(component="主梁", location="右幅2#梁", type="纵向裂缝", description="梁底纵向裂缝")
        previous = Disease(
            component="主梁",
            location="右幅主梁",
            type="裂缝、破损露筋、渗水泛碱",
            description="右幅主梁存在裂缝、破损露筋及渗水泛碱",
        )

        group_match = DiseaseMatcher().best_group_match(current, [previous])

        self.assertIsNotNone(group_match)
        assert group_match is not None
        self.assertEqual(group_match[0], 0)
        self.assertGreaterEqual(group_match[1], 0.78)

    def test_group_match_recognizes_cable_sheath_wording_variants(self) -> None:
        current = Disease(component="斜拉索", location="左幅NS12索", type="划痕", description="PE保护套表面划痕")
        previous = Disease(component="斜拉索", location="左幅斜拉索索体", type="PE保护套裂痕", description="索体裂痕共计429处")

        self.assertIsNotNone(DiseaseMatcher().best_group_match(current, [previous]))


if __name__ == "__main__":
    unittest.main()
