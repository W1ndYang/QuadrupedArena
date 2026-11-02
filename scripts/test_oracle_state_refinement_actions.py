"""Contract checks for frozen paired state-refinement capture."""
from __future__ import annotations

import unittest

from oracle_state_refinement_actions import compare_pairs, evaluation_task


class StateRefinementActionTests(unittest.TestCase):
    def setUp(self):
        self.discovery = {
            "sha256": "source", "scan_action_id": "SCAN",
            "fit": {"baseline_action_id": "BASE", "base_state_id": "root",
                    "predicate": {"threshold": -4.275}},
            "tied": [{"predicate": {"id": "P1:x:-0.5r"}},
                     {"predicate": {"id": "P1:y:+0.5r"}}],
            "plans": [{"prefix_id": "decisive", "discovery_task_key": "key",
                       "prefix_action_ids": ["P"] * 4, "expected_boundaries": [{}] * 5,
                       "own_xy": [-4.02, -1.20], "score_at_8s": 4.0,
                       "predicate_west_like": {"P1:x:-0.5r": False, "P1:y:+0.5r": True},
                       "predicate_disagreement": True}]}
        self.config = {"continuation_action_id": "BASE", "discovery_synthetic_seed": 5,
                       "game_seed": 7, "source_sha256": "current", "pixel_noise_std": 1,
                       "miss_probability": 0}

    def outcomes(self, baseline, scan, *, selectable=True):
        return [{"prefix_id": "decisive", "action_id": "BASE", "local_bin": "p1_other_exit",
                 "selectable": True, "official_return": baseline},
                {"prefix_id": "decisive", "action_id": "SCAN", "local_bin": "p1_other_exit",
                 "selectable": selectable, "official_return": scan if selectable else None}]

    def test_frozen_x_y_predictions_and_full_game_tie(self):
        task = evaluation_task(self.config, self.discovery, self.discovery["plans"][0], "SCAN")
        self.assertEqual(task["max_steps"], 20)
        self.assertEqual(task["target_state_id"], "root")
        self.assertEqual(set(task["local_policy_table"].values()), {"SCAN"})
        result = compare_pairs(self.discovery, self.outcomes(2.0, 2.0))
        decisive = result["decisive_pair"]
        self.assertEqual(decisive["empirical_winner_with_baseline_tie_break"], "BASE")
        self.assertEqual(decisive["prediction_correct"],
                         {"P1:x:-0.5r": True, "P1:y:+0.5r": False})

    def test_blocked_action_keeps_diagnostic_but_has_no_policy_value(self):
        result = compare_pairs(self.discovery, self.outcomes(2.0, 9.0, selectable=False))
        decisive = result["decisive_pair"]
        self.assertIsNone(decisive["empirical_winner_with_baseline_tie_break"])
        self.assertIsNone(decisive["scan_minus_baseline_official_return"])
        self.assertEqual(result["selectable_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
