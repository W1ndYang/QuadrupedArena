"""Decision and executability checks for the 20 s P2 intervention."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from oracle_p2_second_stage import (branch_outcome, branch_task, candidate_catalog,
                                    choose_training_action)
from oracle_pilot import DEFAULT_CONFIG, resolve_config
from oracle_pilot_runtime import task_action_id


class P2SecondStageTests(unittest.TestCase):
    def test_frozen_prefix_and_single_target_action(self):
        task = {"mode": "oracle_second_stage", "target_step": 10,
                "prefix_action_ids": [f"P{i}" for i in range(10)],
                "candidate_action_id": "COVER", "expected_state_id": "root",
                "continuation_action_id": "BASE"}
        self.assertEqual(task_action_id(task, 9, "root"), ("P9", "verified_prefix"))
        self.assertEqual(task_action_id(task, 10, "root"), ("COVER", "second_stage_candidate"))
        self.assertEqual(task_action_id(task, 10, "other"), ("BASE", "second_stage_fallback"))
        self.assertEqual(task_action_id(task, 11, "root"), ("BASE", "second_stage_fallback"))

    def test_formal_shortlist_and_task_boundary(self):
        config = resolve_config(DEFAULT_CONFIG)
        candidates = candidate_catalog(config)
        self.assertEqual(len(candidates), 7)
        self.assertEqual(candidates[0], config["continuation_action_id"])
        sample = {"sample_id": "train", "prefix_id": "prefix", "synthetic_seed": 4,
                  "prefix_action_ids": ["a"] * 10, "expected_boundaries": [{}] * 11,
                  "target_state_id": "root"}
        task = branch_task(config, {"sha256": "frozen"}, sample, candidates[1])
        self.assertEqual((task["target_step"], task["max_steps"]), (10, 20))
        self.assertEqual(task["candidate_action_id"], candidates[1])
        self.assertEqual(task["frozen_full_game_bundle_sha256"], "frozen")

    def test_score_tie_keeps_baseline_and_blocked_is_ineligible(self):
        baseline = {"action_id": "BASE", "selectable": True, "final_official_score": 6.62,
                    "terminal_seconds_diagnostic": 22.1}
        tied = {"action_id": "TIED", "selectable": True, "final_official_score": 6.62,
                "terminal_seconds_diagnostic": 30.0}
        blocked = {"action_id": "BLOCKED", "selectable": False, "final_official_score": None,
                   "terminal_seconds_diagnostic": 40.0}
        choice = choose_training_action([baseline, tied, blocked], ["BASE", "TIED", "BLOCKED"], "BASE")
        self.assertEqual(choice["selected_action_id"], "BASE")
        self.assertEqual(choice["training_advantage"], 0.0)
        improved = {**tied, "final_official_score": 6.7}
        choice = choose_training_action([baseline, improved, blocked], ["BASE", "TIED", "BLOCKED"], "BASE")
        self.assertEqual(choice["selected_action_id"], "TIED")

    def test_accepted_but_blocked_remains_recorded_without_value(self):
        with TemporaryDirectory() as tmp:
            raw = Path(tmp)
            boundaries = [{"simulation_seconds": float(i * 2),
                           "limited": {"state_id": "root"},
                           "score": 4.52} for i in range(11)]
            decisions = [{} for _ in range(10)] + [{"action_id": "COVER",
                "source": "second_stage_candidate", "validator_status": "accepted",
                "execution_status": "accepted_but_blocked", "skill": "TAKE_COVER",
                "compiled_action": {"skill": "HOLD"},
                "compiled_skill": {"state": "NO_SAFE_COVER"}}]
            (raw / "boundaries.json").write_text(json.dumps(boundaries))
            (raw / "decisions.json").write_text(json.dumps(decisions))
            record = {"task_key": "key", "raw_dir": str(raw), "status": "passed",
                      "task": {"candidate_action_id": "COVER"},
                      "result": {"stop_reason": "terminal", "terminal_reason": "time_limit",
                                 "official_score": 10.0, "evaluation_delta_score": 5.48,
                                 "simulation_seconds": 40.0}, "wall_seconds": 1.0}
            sample = {"sample_id": "sample", "prefix_id": "prefix",
                      "target_state_id": "root", "score_at_20s": 4.52}
            item = branch_outcome(record, sample)
            self.assertEqual(item["status"], "accepted_but_blocked")
            self.assertFalse(item["selectable"])
            self.assertIsNone(item["final_official_score"])


if __name__ == "__main__":
    unittest.main()
