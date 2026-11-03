"""Checks for seed grouping and robust, equivalent predicate selection."""
from __future__ import annotations

import unittest

from finite_state_optimizer import fit_one_bit_state, geometry_predicates
from oracle_state_seed_analysis import evaluate_forward, grouped_rows, max_margin_equivalent_fit


class SeedAnalysisTests(unittest.TestCase):
    def test_repeated_seed_visits_are_one_training_row_per_prefix(self):
        def pair(prefix, y, baseline, scan):
            return {"prefix_id": prefix, "public_own_xy": [-4, y],
                    "baseline_outcome": {"action_id": "base", "official_return": baseline},
                    "scan_outcome": {"action_id": "scan", "official_return": scan}}
        visits = {10: [pair("south", -1.8, 4, 8), pair("north", -1.2, 4, 0)],
                  11: [pair("south", -1.8, 2, 4), pair("north", -1.2, 2, 2)]}
        rows = grouped_rows(visits, [10, 11], "state")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["returns"], {"base": 3, "scan": 6})

    def test_equivalent_threshold_prefers_more_clearance(self):
        predicates = geometry_predicates([{"name": "P1", "center": [-4, -1.5], "radius": 0.55}])
        rows = [{"sample_id": name, "base_state_id": "state", "own_xy": [-4, y],
                 "returns": {"base": base, "scan": scan}}
                for name, y, base, scan in [("a", -1.801, 0, 4), ("b", -1.77558, 0, 4),
                                            ("c", -1.254, 4, 0), ("d", -1.207, 4, 0)]]
        fit = fit_one_bit_state(rows, predicates, "base", min_leaf_samples=2,
                                complexity_penalty=0.1)
        deployment, equivalent = max_margin_equivalent_fit(fit, rows, predicates)
        self.assertEqual(fit["predicate"]["id"], "P1:y:-0.5r")
        self.assertEqual(deployment["predicate"]["id"], "P1:y:+0.0r")
        self.assertEqual(len(equivalent), 2)

    def test_forward_evaluation_uses_frozen_actions(self):
        plan = {"training_bundle_sha256": "training", "forward_game_seeds": [12, 13],
                "prefix_ids": ["north", "south"], "baseline_action_id": "base",
                "zero_bit_action_id": "scan", "one_bit_fit": {
                    "base_state_id": "state", "predicate": {"axis": "y", "threshold": -1.5},
                    "leaves": {"low": {"action_id": "scan"},
                               "high": {"action_id": "base"}}}}
        def pair(prefix, y, base, scan):
            return {"prefix_id": prefix, "public_own_xy": [-4, y],
                    "baseline_outcome": {"official_return": base},
                    "scan_outcome": {"action_id": "scan", "official_return": scan}}
        visits = {12: [pair("north", -1.2, 4, 0), pair("south", -1.8, 0, 4)],
                  13: [pair("north", -1.2, 4, 4), pair("south", -1.8, 0, 2)]}
        result = evaluate_forward(plan, {"base_state_id": "state",
                                         "baseline_action_id": "base"}, visits)
        self.assertEqual(result["mean_zero_bit_advantage"], 0.5)
        self.assertEqual(result["mean_one_bit_advantage"], 1.5)


if __name__ == "__main__":
    unittest.main()
