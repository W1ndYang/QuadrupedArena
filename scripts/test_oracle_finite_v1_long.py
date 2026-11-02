"""CPU contracts for delayed score training, fixed continuation, and held-out data."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from oracle_finite_v1_long import outcome_from_record, solve_rollouts, task_for
from oracle_pilot import DEFAULT_CONFIG, atomic_json, resolve_config


class LongOracleTests(unittest.TestCase):
    def test_delayed_score_selects_formal_action_using_three_samples(self):
        actions = ["BASE", "COVER", "BLOCKED"]
        samples = [{"sample_id": x} for x in ("s1", "s2", "s3")]
        outcomes = []
        for sample in samples:
            for action in actions:
                outcomes.append({"sample_id": sample["sample_id"], "initial_action_id": action,
                                 "executable": action != "BLOCKED",
                                 "official_score_delta": 0.0 if action == "BASE" else
                                                         1.0 if action == "COVER" else None,
                                 "survival_seconds": 16.0 if action == "BASE" else 20.0,
                                 "terminal_reason": "quadruped_eliminated" if action == "BASE" else None})
        policy = solve_rollouts(outcomes, actions, samples, "BASE")
        self.assertEqual(policy["selected_action_id"], "COVER")
        self.assertEqual(policy["training_mean_advantage"], 1.0)
        self.assertFalse(policy["action_values"][2]["common_executable"])

    def test_equal_score_keeps_baseline_despite_small_survival_difference(self):
        samples = [{"sample_id": x} for x in ("s1", "s2", "s3")]
        outcomes = [{"sample_id": sample["sample_id"], "initial_action_id": action,
                     "executable": True, "official_score_delta": 1.0,
                     "survival_seconds": 20.0 if action == "COVER" else 19.9,
                     "terminal_reason": None}
                    for sample in samples for action in ("BASE", "COVER")]
        policy = solve_rollouts(outcomes, ["BASE", "COVER"], samples, "BASE")
        self.assertEqual(policy["selected_action_id"], "BASE")
        self.assertEqual(policy["training_mean_advantage"], 0.0)

    def test_candidate_task_uses_only_root_r2l_id_then_fixed_continuation(self):
        config = resolve_config(DEFAULT_CONFIG)
        plan = config["prefix_library"]["prefixes"][0]
        source = {"prefix_id": plan["prefix_id"], "prefix_action_ids": plan["prefix_action_ids"],
                  "expected_boundaries": [{"state_id": "root"}] * 5}
        inputs = {"prefixes": {plan["prefix_id"]: source}, "group": {"state_id": "root"},
                  "corrected_samples_sha256": "samples"}
        prior = {"bundle_sha256": "prior"}
        sample = {"sample_id": "s1", "prefix_id": plan["prefix_id"],
                  "synthetic_seed": config["discovery_synthetic_seed"]}
        action = next(x["action_id"] for x in config["action_catalog"]
                      if x["skill"] == "TAKE_COVER" and x["params"] == {"risk": "safe"})
        task = task_for(config, inputs, prior, sample, action)
        self.assertEqual(task["mode"], "oracle_eval_prefix")
        self.assertEqual(task["max_steps"], 10)
        self.assertEqual(task["policy_table"], {"4": {"root": action}})
        self.assertNotIn("candidate_action_id", task)

    def test_rollout_keeps_each_formal_transition_and_terminal_reason(self):
        with TemporaryDirectory() as tmp:
            raw = Path(tmp)
            boundaries = [{"limited": {"state_id": f"state_{step}", "canonical_state": f"canon_{step}"},
                           "score": float(step), "finished": None} for step in range(6)]
            boundaries[-1]["finished"] = "quadruped_eliminated"
            decisions = [{"action_id": f"prefix_{step}"} for step in range(4)] + [
                {"action_id": "FORMAL", "validator_status": "accepted",
                 "execution_status": "started", "delta_score": 1.0}]
            atomic_json(raw / "boundaries.json", boundaries)
            atomic_json(raw / "decisions.json", decisions)
            record = {"raw_dir": str(raw), "task_key": "task", "task": {"sample_id": "s1", "prefix_id": "p1"},
                      "status": "passed", "wall_seconds": 1.0,
                      "result": {"status": "passed", "stop_reason": "terminal",
                                 "official_score": 5.0, "evaluation_delta_score": 1.0,
                                 "simulation_seconds": 10.0, "terminal_reason": "quadruped_eliminated"}}
            outcome = outcome_from_record(record, "FORMAL")
            self.assertEqual(outcome["official_score_delta"], 1.0)
            self.assertEqual(outcome["transitions"][0]["next_state_id"], "state_5")
            self.assertTrue(outcome["transitions"][0]["terminal"])


if __name__ == "__main__":
    unittest.main()
