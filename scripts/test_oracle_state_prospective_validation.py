"""Prospective matched-pair comparison checks."""
from __future__ import annotations

import unittest

from oracle_state_prospective_validation import compare_prospective


class ProspectiveComparisonTests(unittest.TestCase):
    def test_frozen_policy_uses_preoutcome_prediction_and_official_return(self):
        frozen = {"synthetic_seed": 9, "baseline_action_id": "BASE", "scan_action_id": "SCAN"}
        states = [
            {"prefix_id": "high", "own_xy": [-4.0, -1.2], "eligible_exact_root": True,
             "frozen_predicted_action_id": "SCAN", "old_x_predicted_action_id": "BASE"},
            {"prefix_id": "low", "own_xy": [-4.1, -1.3], "eligible_exact_root": True,
             "frozen_predicted_action_id": "BASE", "old_x_predicted_action_id": "BASE"},
            {"prefix_id": "off_root", "own_xy": [-3, -3], "eligible_exact_root": False,
             "frozen_predicted_action_id": "BASE", "old_x_predicted_action_id": "BASE"},
        ]
        outcomes = []
        for prefix_id, baseline, scan in (("high", 0.0, 2.0), ("low", 4.0, 2.0)):
            for action, value in (("BASE", baseline), ("SCAN", scan)):
                outcomes.append({"prefix_id": prefix_id, "action_id": action,
                                 "local_bin": "p1_other_exit", "selectable": True,
                                 "official_return": value})
        result = compare_prospective(frozen, states, outcomes)
        self.assertEqual(result["selectable_pairs"], 2)
        self.assertEqual([p["frozen_policy_advantage"] for p in result["pairs"]], [2.0, 0.0])
        self.assertEqual([p["all_scan_advantage"] for p in result["pairs"]], [2.0, -2.0])
        self.assertEqual(result["mean_frozen_policy_advantage"], 1.0)
        self.assertEqual(result["mean_all_scan_advantage"], 0.0)


if __name__ == "__main__":
    unittest.main()
