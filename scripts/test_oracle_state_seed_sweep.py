"""Checks for seed-stratified prospective reporting."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from oracle_state_seed_sweep import VERSION, aggregate_completed, seed_plan


class SeedSweepTests(unittest.TestCase):
    def test_seed_plan_uses_distinct_unseen_game_and_vision_streams(self):
        plan = seed_plan({"game_seed": 100, "training_synthetic_seeds": [1, 2],
                          "test_synthetic_seeds": [8, 9, 10, 11, 12]})
        self.assertEqual(plan, [{"game_seed": 102, "synthetic_seed": 8},
                                {"game_seed": 103, "synthetic_seed": 8},
                                {"game_seed": 104, "synthetic_seed": 8}])

    def test_aggregation_keeps_seed_and_prefix_groups(self):
        plan = [{"game_seed": 102, "synthetic_seed": 10},
                {"game_seed": 103, "synthetic_seed": 11}]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for item, gain in zip(plan, (2.0, 0.0)):
                folder = root / f"game_{item['game_seed']}"
                folder.mkdir()
                (folder / "prospective_validation.json").write_text(json.dumps({
                    "version": VERSION, "synthetic_seed": item["synthetic_seed"],
                    "discovered_prefixes": 4, "eligible_exact_root": 0,
                    "eligible_compatible_root": 1, "selectable_pairs": 1,
                    "mean_frozen_policy_advantage": gain, "mean_all_scan_advantage": gain,
                    "pairs": [{"prefix_id": "same", "public_own_xy": [-4, -1.2],
                               "both_actions_selectable": True,
                               "frozen_policy_advantage": gain, "all_scan_advantage": gain}]}))
            result = aggregate_completed(root, plan)
            self.assertEqual(result["paired_seed_prefix_visits"], 2)
            self.assertEqual(result["prefix_results"][0]["paired_seeds"], 2)
            self.assertEqual(result["mean_frozen_policy_advantage"], 1.0)


if __name__ == "__main__":
    unittest.main()
