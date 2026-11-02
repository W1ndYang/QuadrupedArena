"""One-bit public-state contract and formal-action execution checks."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from adaptive_motion import load_config
from oracle_local_state_v1 import VERSION, p1_exit_state, p1_west_boundary
from oracle_pilot import DEFAULT_CONFIG, resolve_config
from oracle_pilot_discovery import boundary_evidence
from oracle_pilot_runtime import run, strategy_action_executable, task_action_id
from semantic_skills import SemanticSkills
from vision_core import CameraRig, VisionMatch


class LocalP1Tests(unittest.TestCase):
    def setUp(self):
        self.config = resolve_config(DEFAULT_CONFIG)
        self.cfg = load_config("configs/trial_rotating.json")
        self.cfg["seed"] = self.config["game_seed"]

    def test_feature_is_one_map_derived_public_bit(self):
        threshold = p1_west_boundary(self.cfg)
        self.assertEqual(threshold, -4.275)
        west = p1_exit_state("root", 4, [-4.3276, -1.14], self.cfg, "root")
        east = p1_exit_state("root", 4, [-4.1556, -1.24], self.cfg, "root")
        heldout = p1_exit_state("root", 4, [-4.3380, -1.14], self.cfg, "root")
        self.assertEqual(west["bin"], "p1_west_edge")
        self.assertEqual(east["bin"], "p1_other_exit")
        self.assertEqual(heldout["bin"], west["bin"])
        self.assertEqual(west["version"], VERSION)
        self.assertNotIn("public_own_xy", west)
        self.assertIsNone(p1_exit_state("another", 4, [-4.4, -1.1], self.cfg, "root"))
        self.assertIsNone(p1_exit_state("root", 5, [-4.4, -1.1], self.cfg, "root"))

    def test_blocked_hold_is_not_a_strategy_action(self):
        blocked = {"skill": "CAPTURE_POINT", "validator_status": "accepted",
                   "execution_status": "started", "compiled_action": {"skill": "HOLD"},
                   "compiled_skill": {"state": "TARGET_INACTIVE"}}
        self.assertFalse(strategy_action_executable(blocked))
        aim = {**blocked, "skill": "AIM_AT_TRACK", "compiled_skill": {"state": "TARGET_STALE"}}
        self.assertFalse(strategy_action_executable(aim))
        observe = {**blocked, "skill": "OBSERVE", "compiled_skill": {"state": "RUNNING"}}
        self.assertTrue(strategy_action_executable(observe))
        arrived = {**blocked, "skill": "TAKE_COVER", "compiled_skill": {"state": "AT_DESTINATION"}}
        self.assertTrue(strategy_action_executable(arrived))

    def test_selection_uses_public_position_only_at_target_boundary(self):
        task = {"mode": "oracle_local_eval_prefix", "prefix_action_ids": ["P"] * 4,
                "target_state_id": "root", "local_policy_table": {"p1_west_edge": "SCAN",
                                                                  "p1_other_exit": "BASE"},
                "continuation_action_id": "BASE"}
        self.assertEqual(task_action_id(task, 4, "root", [-4.33, -1.14], self.cfg),
                         ("SCAN", "local_table:p1_west_edge"))
        self.assertEqual(task_action_id(task, 4, "root", [-4.15, -1.24], self.cfg),
                         ("BASE", "local_table:p1_other_exit"))
        self.assertEqual(task_action_id(task, 4, "other", [-4.33, -1.14], self.cfg),
                         ("BASE", "local_fallback"))
        self.assertEqual(task_action_id(task, 5, "root", [-4.33, -1.14], self.cfg),
                         ("BASE", "local_fallback"))

    def test_cpu_four_window_prefix_enters_local_policy_mode(self):
        plan = self.config["prefix_library"]["prefixes"][0]
        base = {"game_seed": self.config["game_seed"],
                "synthetic_seed": self.config["discovery_synthetic_seed"],
                "pixel_noise_std": 1.0, "miss_probability": 0.0,
                "continuation_action_id": self.config["continuation_action_id"],
                "prefix_action_ids": plan["prefix_action_ids"]}
        scan = next(x["action_id"] for x in self.config["action_catalog"]
                    if x["skill"] == "OBSERVE" and x["params"] == {"direction": "scan", "risk": "safe"})

        def episode(root, task):
            match, engine, rig = VisionMatch(self.cfg), SemanticSkills(self.cfg), CameraRig()
            image = Image.new("RGB", (32, 32), "black")
            count = 0

            def capture():
                nonlocal count
                state = match.states[-1]
                name = f"frames/{count:03d}.png"
                (root / "frames").mkdir(parents=True, exist_ok=True)
                image.save(root / name)
                count += 1
                return {"file": name, "simulation_seconds": state["t"],
                        "own_xy": state["xy"][0], "body_yaw_degrees": 0.,
                        "camera_pan_degrees": rig.pan, "camera_world_yaw_degrees": 0.,
                        "camera_eye": [*state["xy"][0], .6], "horizontal_fov_degrees": 90.}

            with patch("oracle_pilot_runtime.observe_current_scene", return_value={
                "status": "not_emitted", "reason": "no_visible_foot", "sighting": None,
                "candidates": [], "source": "simulator_geometry_truth"}):
                return run(root, task, self.cfg, match, engine, rig, None, None, capture)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = episode(root / "source", {**base, "mode": "discover", "max_steps": 4})
            self.assertEqual(source["status"], "passed")
            boundaries = json.loads((root / "source/boundaries.json").read_text())
            expected = [{**boundary_evidence(item), "collapsed": item["collapsed"]} for item in boundaries]
            target = boundaries[4]["limited"]["state_id"]
            task = {**base, "mode": "oracle_local_eval_prefix", "max_steps": 6,
                    "target_state_id": target, "expected_boundaries": expected,
                    "feature_threshold_x": p1_west_boundary(self.cfg),
                    "local_policy_table": {"p1_west_edge": scan,
                                           "p1_other_exit": self.config["continuation_action_id"]}}
            result = episode(root / "local", task)
            local_boundaries = json.loads((root / "local/boundaries.json").read_text())
            decisions = json.loads((root / "local/decisions.json").read_text())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["boundaries"], 7)
            self.assertEqual(local_boundaries[4]["local_policy_state"]["version"], VERSION)
            self.assertTrue(decisions[4]["source"].startswith("local_table:"))
            self.assertEqual(decisions[5]["source"], "local_fallback")


if __name__ == "__main__":
    unittest.main()
