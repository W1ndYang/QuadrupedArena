"""Prospective matched-pair comparison checks."""
from __future__ import annotations

import json
import unittest

from oracle_state_prospective_validation import (NEW_GAME_VERSION, compare_prospective,
                                                discovery_task, paired_task, state_from_evidence,
                                                state_signature)


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

    def test_seed_fingerprint_transfer_requires_identical_observable_fields(self):
        fields = {"target.active": "P2", "A.region_id": "target_P1", "A.hp": 3}
        def canonical(rule, hp=3):
            return json.dumps(["schema", rule, "limited", "valid",
                               [["A.hp", hp], ["A.region_id", "target_P1"], ["target.active", "P2"]]])
        frozen = {"root_state_id": "old-id", "root_state_signature": state_signature(canonical("old")),
                  "allow_rule_fingerprint_transfer": True, "baseline_action_id": "BASE",
                  "fit": {"selected_bits": 1, "predicate": {"axis": "y", "threshold": -1.225},
                          "leaves": {"low": {"action_id": "BASE"}, "high": {"action_id": "SCAN"}}},
                  "prior_fit": {"selected_bits": 1, "predicate": {"axis": "x", "threshold": -4.275},
                                "leaves": {"low": {"action_id": "SCAN"},
                                           "high": {"action_id": "BASE"}}}}
        def boundary(hp=3):
            return {"limited": {"status": "valid", "state_id": "new-id",
                                "canonical_state": canonical("new", hp),
                                "fields": {**fields, "A.hp": hp}},
                    "finished": None, "score": 4.0, "collapsed": False,
                    "replay_reference_digest": "digest", "skill_status": {}, "visual_memory": {}}
        frames = [{"simulation_seconds": 8.0, "own_xy": [-4.02, -1.20]}]
        good = state_from_evidence("prefix", "key", [boundary()] * 5, frames, frozen)
        self.assertFalse(good["eligible_exact_root"])
        self.assertTrue(good["eligible_transfer_root"])
        self.assertEqual(good["frozen_predicted_action_id"], "SCAN")
        bad = state_from_evidence("prefix", "key", [boundary(2)] * 5, frames, frozen)
        self.assertFalse(bad["eligible_transfer_root"])
        self.assertEqual(bad["frozen_predicted_action_id"], "BASE")
        frozen.update(version="oracle_state_new_game_validation_v2", synthetic_seed=20261102,
                      scan_action_id="SCAN")
        task = paired_task({"game_seed": 2026102108, "source_sha256": "code",
                            "pixel_noise_std": 1.0, "miss_probability": 0.0,
                            "continuation_action_id": "BASE"}, frozen, good, ["P"] * 4, "SCAN")
        self.assertEqual(task["target_state_id"], "new-id")
        self.assertEqual(set(task["local_policy_table"].values()), {"SCAN"})


if __name__ == "__main__":
    unittest.main()
