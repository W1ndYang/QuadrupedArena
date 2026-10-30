"""CPU contracts for Pilot replay, action isolation, analysis and packaging."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from adaptive_motion import load_config
from oracle_pilot import analyze_actions, bootstrap_ci, package, resolve_config
from oracle_pilot_runtime import run, task_action_id
from oracle_state_v1 import action_catalog
from semantic_skills import SemanticSkills
from vision_core import CameraRig, VisionMatch


class PilotTests(unittest.TestCase):
    def test_config_uses_actual_44_templates(self):
        config = resolve_config(Path("configs/oracle_coarse_pilot.json"))
        self.assertEqual(len(config["action_catalog"]), 44)
        self.assertEqual(len({x["action_id"] for x in config["action_catalog"]}), 44)
        self.assertEqual(config["continuation_action_id"],
                         next(x["action_id"] for x in config["action_catalog"] if
                              x["skill"] == "CAPTURE_POINT" and x["params"] ==
                              {"target": "active", "risk": "fast"}))
        self.assertFalse(set(config["training_synthetic_seeds"]) & set(config["test_synthetic_seeds"]))

    def test_policy_selection_only_uses_limited_id(self):
        task = {"mode": "eval_policy", "policy_table": {"limited_A": "action_A"},
                "continuation_action_id": "fallback"}
        self.assertEqual(task_action_id(task, 3, "limited_A"), ("action_A", "table"))
        self.assertEqual(task_action_id(task, 3, "other"), ("fallback", "fallback"))
        self.assertIsNone(bootstrap_ci([1]))
        self.assertEqual(bootstrap_ci([2, 2, 2]), [2, 2])

    def test_fresh_cpu_prefix_replay_matches_limited_reference_and_score(self):
        config = resolve_config(Path("configs/oracle_coarse_pilot.json"))
        cfg = load_config("configs/trial_rotating.json")
        cfg["seed"] = config["game_seed"]
        task = {"mode": "preflight", "game_seed": cfg["seed"],
                "synthetic_seed": config["training_synthetic_seeds"][0],
                "pixel_noise_std": 1.0, "miss_probability": 0.0,
                "continuation_action_id": config["continuation_action_id"], "max_steps": 1}

        def one(root):
            match, engine, rig = VisionMatch(cfg), SemanticSkills(cfg), CameraRig()
            image = Image.new("RGB", (768, 432), "black")
            count = 0
            def capture():
                nonlocal count
                state = match.states[-1]
                name = f"frames/{count:03d}_{state['t']:.2f}.png"
                (root / "frames").mkdir(parents=True, exist_ok=True)
                image.save(root / name)
                count += 1
                return {"file": name, "simulation_seconds": state["t"],
                        "own_xy": state["xy"][0], "body_yaw_degrees": 0.,
                        "camera_pan_degrees": rig.pan, "camera_world_yaw_degrees": 0.,
                        "camera_eye": [state["xy"][0][0], state["xy"][0][1], .6],
                        "horizontal_fov_degrees": 90.}
            with patch("oracle_pilot_runtime.observe_current_scene", return_value={
                "status": "not_emitted", "reason": "no_visible_foot", "sighting": None,
                "candidates": [], "source": "simulator_geometry_truth"}):
                result = run(root, task, cfg, match, engine, rig, None, None, capture)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["boundaries"], 2)
            self.assertEqual(result["decisions"], 1)
            return json.loads((root / "boundaries.json").read_text())

        with tempfile.TemporaryDirectory() as tmp:
            a = one(Path(tmp) / "first")
            b = one(Path(tmp) / "second")
            self.assertEqual([(x["limited"]["state_id"], x["reference_digest_1e6"], x["score"])
                              for x in a],
                             [(x["limited"]["state_id"], x["reference_digest_1e6"], x["score"])
                              for x in b])

    def test_common_action_and_archive_are_computed_from_completed_branches(self):
        config = resolve_config(Path("configs/oracle_coarse_pilot.json"))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "tasks").mkdir()
            for sample, values in {"x1": [("A", 1.0), ("B", 2.0)],
                                   "x2": [("A", 3.0), ("B", 1.0)]}.items():
                for action, score in values:
                    key = sample + action
                    raw = output / "raw" / key
                    raw.mkdir(parents=True)
                    (raw / "decisions.json").write_text(json.dumps([{
                        "validator_status": "accepted", "execution_status": "completed"}]))
                    (output / "tasks" / f"{key}.json").write_text(json.dumps({
                        "task_key": key, "raw_dir": str(raw), "status": "passed",
                        "task": {"mode": "branch", "source_sha256": config["source_sha256"],
                                 "sample_id": sample, "target_step": 0,
                                 "expected_state_id": "state_s", "candidate_action_id": action},
                        "result": {"official_delta_score": score}}))
            summary = analyze_actions(output, config)
            self.assertEqual(summary[0]["common_action"], "A")
            self.assertEqual(summary[0]["mean_regret"], .5)
            self.assertFalse(json.loads((output / "analysis" / "policy.json").read_text())["table"])
            archive = package(output)
            self.assertTrue(archive.exists())
            self.assertGreater(json.loads((output / "manifest.json").read_text())["file_count"], 0)


if __name__ == "__main__":
    unittest.main()
