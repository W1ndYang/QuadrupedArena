"""The late probe changes exactly the 10 s decision window."""
from __future__ import annotations

import unittest

from oracle_late_decision_probe import late_task
from oracle_pilot import DEFAULT_CONFIG, resolve_config
from oracle_pilot_runtime import task_action_id


class LateDecisionProbeTests(unittest.TestCase):
    def test_step_five_uses_scan_and_other_windows_use_baseline(self):
        config = resolve_config(DEFAULT_CONFIG)
        state = "limited-state-at-ten"
        context = {"game_seed": 2026102112, "synthetic_seed": 20261101,
                   "prefix_id": "prefix", "baseline_task_key": "a" * 20,
                   "prefix_action_ids": ["action-0", "action-1", "action-2", "action-3"],
                   "expected_boundaries": [{}] * 5, "target_state_id": state}
        task = late_task(config, context, "scan", "bundle-sha")
        baseline = config["continuation_action_id"]
        self.assertEqual(task_action_id(task, 4, state), (baseline, "oracle_fallback"))
        self.assertEqual(task_action_id(task, 5, state), ("scan", "oracle_table"))
        self.assertEqual(task_action_id(task, 5, "different-state"),
                         (baseline, "oracle_fallback"))
        self.assertEqual(task_action_id(task, 6, state), (baseline, "oracle_fallback"))


if __name__ == "__main__":
    unittest.main()
