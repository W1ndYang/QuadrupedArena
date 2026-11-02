"""CPU checks for frozen full-duration task construction and runtime limits."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from adaptive_motion import load_config
from oracle_local_p1_full_game import full_game_task, replay_overlap
from oracle_local_state_v1 import p1_west_boundary
from oracle_pilot import DEFAULT_CONFIG, resolve_config
from oracle_pilot_discovery import boundary_evidence
from oracle_pilot_runtime import run
from semantic_skills import SemanticSkills
from vision_core import CameraRig, VisionMatch


class FullGameTests(unittest.TestCase):
    def test_full_game_task_freezes_policy_and_prefix(self):
        original = {"mode": "oracle_local_eval_prefix", "max_steps": 10,
                    "source_sha256": "old", "prefix_action_ids": ["A", "B", "C", "D"],
                    "local_policy_table": {"p1_west_edge": "SCAN"}}
        snapshot = copy.deepcopy(original)
        task = full_game_task(original, {"source_sha256": "current"}, "prior-hash")
        self.assertEqual(original, snapshot)
        self.assertEqual(task["max_steps"], 20)
        self.assertEqual(task["prefix_action_ids"], snapshot["prefix_action_ids"])
        self.assertEqual(task["local_policy_table"], snapshot["local_policy_table"])
        self.assertEqual(task["frozen_prior_bundle_sha256"], "prior-hash")

    def test_prior_overlap_exposes_post_decision_divergence(self):
        before = [{"simulation_seconds": 8.0, "score": 4.0, "hp": [3, 3],
                   "finished": None, "collapsed": False, "replay_reference_digest": "r",
                   "skill_status": {}, "visual_memory": {},
                   "limited": {"state_id": "root"}},
                  {"simulation_seconds": 10.0, "score": 4.0, "hp": [3, 3],
                   "finished": None, "collapsed": False, "replay_reference_digest": "next",
                   "skill_status": {}, "visual_memory": {},
                   "limited": {"state_id": "next"}}]
        decisions = [{"action_id": "SCAN", "source": "local_table:p1_west_edge",
                      "validator_status": "accepted", "execution_status": "started",
                      "compiled_action": {"skill": "SCAN"}, "delta_score": 0.0}]
        self.assertTrue(replay_overlap(before, copy.deepcopy(before), decisions,
                                       copy.deepcopy(decisions))["matched"])
        changed = copy.deepcopy(before)
        changed[1]["replay_reference_digest"] = "different"
        self.assertFalse(replay_overlap(before, changed, decisions, decisions)["matched"])

    def test_cpu_episode_reaches_formal_terminal_with_twenty_step_limit(self):
        config = resolve_config(DEFAULT_CONFIG)
        cfg = load_config("configs/trial_rotating.json")
        cfg["seed"] = config["game_seed"]
        prefix = config["prefix_library"]["prefixes"][0]["prefix_action_ids"]
        scan = next(item["action_id"] for item in config["action_catalog"] if
                    item["skill"] == "OBSERVE" and
                    item["params"] == {"direction": "scan", "risk": "safe"})
        common = {"game_seed": config["game_seed"], "synthetic_seed": config["discovery_synthetic_seed"],
                  "pixel_noise_std": 1.0, "miss_probability": 0.0,
                  "continuation_action_id": config["continuation_action_id"],
                  "prefix_action_ids": prefix}

        def episode(root, task):
            match, engine, rig = VisionMatch(cfg), SemanticSkills(cfg), CameraRig()
            image = Image.new("RGB", (16, 16), "black")
            counter = 0

            def capture():
                nonlocal counter
                state = match.states[-1]
                name = f"frames/{counter:03d}.png"
                (root / "frames").mkdir(parents=True, exist_ok=True)
                image.save(root / name)
                counter += 1
                return {"file": name, "simulation_seconds": state["t"],
                        "own_xy": state["xy"][0], "body_yaw_degrees": 0.,
                        "camera_pan_degrees": rig.pan, "camera_world_yaw_degrees": 0.,
                        "camera_eye": [*state["xy"][0], .6], "horizontal_fov_degrees": 90.}

            with patch("oracle_pilot_runtime.observe_current_scene", return_value={
                "status": "not_emitted", "reason": "no_visible_foot", "sighting": None,
                "candidates": [], "source": "simulator_geometry_truth"}):
                return run(root, task, cfg, match, engine, rig, None, None, capture)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = episode(root / "prefix", {**common, "mode": "discover", "max_steps": 4})
            self.assertEqual(first["status"], "passed")
            previous = json.loads((root / "prefix/boundaries.json").read_text())
            expected = [{**boundary_evidence(item), "collapsed": item["collapsed"]}
                        for item in previous]
            task = {**common, "mode": "oracle_local_eval_prefix", "max_steps": 20,
                    "target_state_id": previous[4]["limited"]["state_id"],
                    "expected_boundaries": expected,
                    "feature_threshold_x": p1_west_boundary(cfg),
                    "local_policy_table": {"p1_west_edge": scan,
                                           "p1_other_exit": config["continuation_action_id"]}}
            result = episode(root / "full", task)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["stop_reason"], "terminal")
            self.assertIn(result["terminal_reason"], ("quadruped_eliminated", "time_limit"))
            self.assertLessEqual(result["simulation_seconds"], 40.0 + 1e-8)
            self.assertEqual(result["evaluation_delta_score"], round(
                result["official_score"] - previous[4]["score"], 6))


if __name__ == "__main__":
    unittest.main()
