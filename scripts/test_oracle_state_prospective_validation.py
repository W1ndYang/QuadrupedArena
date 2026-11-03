"""Prospective matched-pair comparison checks."""
from __future__ import annotations

import unittest

from oracle_state_prospective_validation import (NEW_GAME_VERSION, compare_prospective,
                                                discovery_task)


class ProspectiveComparisonTests(unittest.TestCase):
    def test_frozen_policy_uses_preoutcome_prediction_and_official_return(self):
        frozen = {"version": "oracle_state_prospective_validation_v1", "synthetic_seed": 9,
                  "baseline_action_id": "BASE", "scan_action_id": "SCAN"}
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

    def test_new_game_task_freezes_policy_but_uses_new_game_and_vision_seed(self):
        config = {"game_seed": 2026102108, "source_sha256": "code", "pixel_noise_std": 1.0,
                  "miss_probability": 0.0, "continuation_action_id": "BASE"}
        frozen = {"version": NEW_GAME_VERSION, "synthetic_seed": 20261102,
                  "fit": {"predicate": {"id": "P1:y:+0.5r"}}}
        task = discovery_task(config, frozen, {"prefix_id": "new", "prefix_action_ids": ["P"] * 4})
        self.assertEqual(task["game_seed"], 2026102108)
        self.assertEqual(task["synthetic_seed"], 20261102)
        self.assertEqual(task["experiment_version"], NEW_GAME_VERSION)


if __name__ == "__main__":
    unittest.main()
