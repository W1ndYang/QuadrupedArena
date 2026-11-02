"""Behavioral checks for automatic local refinement and backward induction."""
from __future__ import annotations

import unittest

from finite_state_optimizer import (fit_one_bit_state, geometry_predicates,
                                    refined_state_key, solve_empirical_horizon)


BASE = "CAPTURE"
SCAN = "SCAN"


class FiniteStateOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.predicates = geometry_predicates([{"name": "P1", "center": [-4.0, -1.5],
                                                 "radius": .55}])
        self.rows = [
            {"sample_id": "west", "base_state_id": "root", "own_xy": [-4.3276, -1.1427],
             "returns": {BASE: 0.0, SCAN: 2.62}},
            {"sample_id": "other1", "base_state_id": "root", "own_xy": [-4.022, -1.6173],
             "returns": {BASE: 4.0}},
            {"sample_id": "other2", "base_state_id": "root", "own_xy": [-4.1556, -1.2394],
             "returns": {BASE: 4.0}},
        ]

    def test_one_bit_is_selected_from_public_map_geometry(self):
        fit = fit_one_bit_state(self.rows, self.predicates, BASE)
        self.assertEqual(fit["selected_bits"], 1)
        self.assertEqual(fit["predicate"]["id"], "P1:x:-0.5r")
        self.assertEqual(fit["predicate"]["threshold"], -4.275)
        self.assertEqual(fit["leaves"]["low"]["action_id"], SCAN)
        self.assertEqual(fit["leaves"]["high"]["action_id"], BASE)
        self.assertGreater(fit["selected_advantage"], .8)
        self.assertEqual(refined_state_key("root", [-4.338, -1.138], fit),
                         "root|P1:x:-0.5r:low")

    def test_equal_return_needs_no_extra_state_bit(self):
        rows = [{**item, "returns": {BASE: 4.0, SCAN: 4.0}} for item in self.rows]
        fit = fit_one_bit_state(rows, self.predicates, BASE)
        self.assertEqual(fit["selected_bits"], 0)
        self.assertIsNone(fit["predicate"])
        self.assertEqual(fit["leaves"]["all"]["action_id"], BASE)

    def test_complexity_penalty_and_support_control_refinement(self):
        cheap = fit_one_bit_state(self.rows, self.predicates, BASE, complexity_penalty=.5)
        costly = fit_one_bit_state(self.rows, self.predicates, BASE, complexity_penalty=1.0)
        unsupported = fit_one_bit_state(self.rows, self.predicates, BASE, min_leaf_samples=2)
        self.assertEqual(cheap["selected_bits"], 1)
        self.assertEqual(costly["selected_bits"], 0)
        self.assertEqual(unsupported["selected_bits"], 0)

    def test_dp_propagates_delayed_score_and_rejects_blocked_action(self):
        states = [{"step": 4, "state_key": "west", "sample_ids": ["w"]},
                  {"step": 4, "state_key": "other", "sample_ids": ["o1", "o2"]},
                  {"step": 10, "state_key": "p2", "sample_ids": ["w"]}]
        edges = [
            {"step": 10, "state_key": "p2", "sample_id": "w", "action_id": BASE,
             "reward": 2.1, "selectable": True, "terminal": True},
            {"step": 10, "state_key": "p2", "sample_id": "w", "action_id": SCAN,
             "reward": 2.1, "selectable": True, "terminal": True},
            {"step": 10, "state_key": "p2", "sample_id": "w", "action_id": "BLOCKED",
             "reward": 10.0, "selectable": False, "terminal": True},
            {"step": 4, "state_key": "west", "sample_id": "w", "action_id": BASE,
             "reward": 0.0, "selectable": True, "terminal": True},
            {"step": 4, "state_key": "west", "sample_id": "w", "action_id": SCAN,
             "reward": .52, "selectable": True, "terminal": False,
             "next_step": 10, "next_state_key": "p2"},
            *[{"step": 4, "state_key": "other", "sample_id": sample, "action_id": BASE,
               "reward": 4.0, "selectable": True, "terminal": True} for sample in ("o1", "o2")],
        ]
        solved = solve_empirical_horizon(states, edges, BASE)
        policy = {(row["step"], row["state_key"]): row for row in solved["values"]}
        self.assertEqual(policy[(10, "p2")]["action_id"], BASE)
        self.assertEqual(policy[(4, "west")]["action_id"], SCAN)
        self.assertAlmostEqual(policy[(4, "west")]["official_return"], 2.62)
        self.assertEqual(policy[(4, "other")]["action_id"], BASE)
        self.assertTrue(any(x["reason"] == "missing_or_blocked_sample" for x in
                            solved["action_diagnostics"] if x["action_id"] == "BLOCKED"))

    def test_unknown_successor_is_not_assigned_zero(self):
        states = [{"step": 4, "state_key": "root", "sample_ids": ["one"]}]
        edges = [{"step": 4, "state_key": "root", "sample_id": "one", "action_id": BASE,
                  "reward": 1.0, "selectable": True, "terminal": True},
                 {"step": 4, "state_key": "root", "sample_id": "one", "action_id": SCAN,
                  "reward": 5.0, "selectable": True, "terminal": False,
                  "next_step": 10, "next_state_key": "missing"}]
        solved = solve_empirical_horizon(states, edges, BASE)
        self.assertEqual(solved["values"][0]["action_id"], BASE)
        self.assertTrue(any(x.get("reason") == "unexpanded_successor" for x in solved["action_diagnostics"]))


if __name__ == "__main__":
    unittest.main()
