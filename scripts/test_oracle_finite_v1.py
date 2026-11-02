"""Small CPU checks for empirical transitions and finite-horizon action choice."""
from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from adaptive_motion import load_config
from oracle_finite_v1 import select_expansion, solve_model
from oracle_pilot import DEFAULT_CONFIG, resolve_config
from oracle_pilot_discovery import boundary_evidence
from oracle_pilot_runtime import run, task_action_id
from semantic_skills import SemanticSkills
from vision_core import CameraRig, VisionMatch


class FiniteOracleTests(unittest.TestCase):
    def test_backward_induction_keeps_hidden_sample_outcomes(self):
        actions = [{"action_id": "A"}, {"action_id": "B"}, {"action_id": "C"}]
        transitions = []
        for sample_id, reward in (("s1", 0.0), ("s2", 1.0), ("s3", 2.0)):
            transitions.extend([
                {"decision_step": 4, "state_id": "root", "action_id": "A",
                 "sample_id": sample_id, "available": True, "reward_delta_score": reward,
                 "next_state_id": "child", "terminal": False},
                {"decision_step": 4, "state_id": "root", "action_id": "B",
                 "sample_id": sample_id, "available": True, "reward_delta_score": 2.0,
                 "next_state_id": "other", "terminal": False},
                {"decision_step": 4, "state_id": "root", "action_id": "C",
                 "sample_id": sample_id, "available": sample_id != "s2",
                 "reward_delta_score": 9.0 if sample_id != "s2" else None,
                 "next_state_id": "other", "terminal": False},
            ])
        transitions.extend([
            {"decision_step": 5, "state_id": "child", "action_id": "A",
             "sample_id": "s1", "available": True, "reward_delta_score": 3.0,
             "next_state_id": "last", "terminal": False},
            {"decision_step": 5, "state_id": "child", "action_id": "B",
             "sample_id": "s1", "available": True, "reward_delta_score": 0.0,
             "next_state_id": "last", "terminal": False},
            {"decision_step": 5, "state_id": "child", "action_id": "C",
             "sample_id": "s1", "available": False, "reward_delta_score": None,
             "next_state_id": None, "terminal": None},
        ])
        model = {"action_catalog": actions, "transitions": transitions,
                 "sample_counts": {"4:root": 3, "5:child": 1},
                 "root_decision_step": 4, "root_state_id": "root"}
        policy = solve_model(model, baseline_action_id="B")
        self.assertEqual(policy["policy_table"], {"5": {"child": "A"}, "4": {"root": "A"}})
        self.assertEqual(policy["root_estimated_value"], 4.0)
        self.assertEqual(policy["baseline_estimated_value"], 2.0)
        self.assertEqual(policy["model_value_advantage"], 2.0)
        self.assertFalse(next(x for x in policy["action_diagnostics"]
                              if x["decision_step"] == 4 and x["action_id"] == "C")["common_available"])
        self.assertEqual(next(x for x in policy["action_diagnostics"]
                              if x["decision_step"] == 4 and x["action_id"] == "A")["reward_range"], [0.0, 2.0])

    def test_expansion_uses_common_action_and_policy_only_reads_state_id(self):
        actions = [{"action_id": "A"}, {"action_id": "B"}]
        samples = [{"sample_id": "s1"}, {"sample_id": "s2"}, {"sample_id": "s3"}]
        outcomes = []
        for sample in samples:
            outcomes.append({"sample_id": sample["sample_id"], "action_id": "A",
                             "available": True, "reward_delta_score": 1.0,
                             "next_state_id": "child", "next_state_status": "valid", "terminal": False,
                             "task_key": "source" if sample["sample_id"] == "s1" else "other"})
            outcomes.append({"sample_id": sample["sample_id"], "action_id": "B",
                             "available": sample["sample_id"] != "s3", "reward_delta_score": 9.0,
                             "next_state_id": "other", "next_state_status": "valid",
                             "terminal": False, "task_key": "other"})
        expansion = select_expansion(outcomes, samples, actions)
        self.assertEqual((expansion["source_action_id"], expansion["source_task_key"]), ("A", "source"))
        task = {"mode": "oracle_eval_prefix", "prefix_action_ids": ["P"] * 4,
                "policy_table": {"4": {"root": "A"}, "5": {"child": "B"}},
                "continuation_action_id": "fallback"}
        self.assertEqual(task_action_id(task, 4, "root"), ("A", "oracle_table"))
        self.assertEqual(task_action_id(task, 5, "child"), ("B", "oracle_table"))
        self.assertEqual(task_action_id(task, 5, "unknown"), ("fallback", "oracle_fallback"))

    def test_policy_executes_after_strict_four_window_prefix(self):
        config = resolve_config(DEFAULT_CONFIG)
        cfg = load_config("configs/trial_rotating.json")
        cfg["seed"] = config["game_seed"]
        plan = config["prefix_library"]["prefixes"][0]
        base = {"game_seed": config["game_seed"], "synthetic_seed": config["discovery_synthetic_seed"],
                "pixel_noise_std": 1.0, "miss_probability": 0.0,
                "continuation_action_id": config["continuation_action_id"],
                "prefix_action_ids": plan["prefix_action_ids"]}

        def episode(root, task):
            match, engine, rig = VisionMatch(cfg), SemanticSkills(cfg), CameraRig()
            image = Image.new("RGB", (32, 32), "black")
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
            source = episode(root / "source", {**base, "mode": "discover", "max_steps": 4})
            self.assertEqual(source["status"], "passed")
            boundaries = json.loads((root / "source/boundaries.json").read_text())
            expected = [{**boundary_evidence(item), "collapsed": item["collapsed"]} for item in boundaries]
            policy = {"4": {boundaries[4]["limited"]["state_id"]: config["continuation_action_id"]}}
            result = episode(root / "eval", {**base, "mode": "oracle_eval_prefix", "max_steps": 6,
                                             "expected_boundaries": expected,
                                             "policy_table": policy})
            decisions = json.loads((root / "eval/decisions.json").read_text())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["boundaries"], 7)
            self.assertEqual(decisions[4]["source"], "oracle_table")
            self.assertEqual(decisions[5]["source"], "oracle_fallback")
            self.assertIn("evaluation_delta_score", result)


if __name__ == "__main__":
    unittest.main()
