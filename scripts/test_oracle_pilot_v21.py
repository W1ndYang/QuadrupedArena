"""CPU contracts for angle equivalence, offline evidence, and replay gates."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from adaptive_motion import load_config
from oracle_pilot import DEFAULT_CONFIG, resolve_config
from oracle_pilot_discovery import (boundary_evidence, reference_diff,
                                    reference_summaries, strict_replay_reference_digest)
from oracle_pilot_runtime import run
from oracle_pilot_v21 import (ANALYSIS_VERSION, package_artifacts, replay_validate,
                              sha256_file)
from oracle_state_v1 import capture_reference
from semantic_skills import SemanticSkills
from vision_core import CameraRig, VisionMatch


class PilotV21Tests(unittest.TestCase):
    def setUp(self):
        self.config = resolve_config(DEFAULT_CONFIG)
        self.cfg = load_config("configs/trial_rotating.json")
        self.cfg["seed"] = self.config["game_seed"]
        match, engine, rig = VisionMatch(self.cfg), SemanticSkills(self.cfg), CameraRig()
        self.reference = asdict(capture_reference(match, engine, rig))

    def changed(self, actor, key, value):
        result = json.loads(json.dumps(self.reference))
        result[actor][key] = value
        return result

    def test_periodic_yaw_equivalence_and_strict_replay_digest(self):
        base = self.changed("a", "yaw", math.pi / 2)
        alias = self.changed("a", "yaw", -3 * math.pi / 2)
        self.assertEqual(reference_summaries(base)["meaningful_state_digest"],
                         reference_summaries(alias)["meaningful_state_digest"])
        self.assertEqual(reference_summaries(base)["dynamics_digest"],
                         reference_summaries(alias)["dynamics_digest"])
        self.assertFalse(reference_diff(base, alias)["qualifies"])
        self.assertNotIn("a.yaw", reference_diff(base, alias)["categories"]["physical_motion"])
        self.assertNotEqual(strict_replay_reference_digest(base), strict_replay_reference_digest(alias))
        for theta in (0.0, 0.37, math.pi / 2, math.pi):
            first = self.changed("b", "yaw", theta)
            for k in (-2, -1, 1, 2):
                other = self.changed("b", "yaw", theta + 2 * math.pi * k)
                self.assertEqual(reference_summaries(first)["meaningful_state_digest"],
                                 reference_summaries(other)["meaningful_state_digest"])
                self.assertFalse(reference_diff(first, other)["qualifies"])
        self.assertEqual(reference_summaries(self.changed("a", "yaw", math.pi))["dynamics_digest"],
                         reference_summaries(self.changed("a", "yaw", -math.pi))["dynamics_digest"])

    def test_real_angle_motion_and_nonangle_fields_still_differ(self):
        base = self.changed("a", "yaw", 0.0)
        direction = self.changed("a", "yaw", 0.1)
        diff = reference_diff(base, direction)
        self.assertTrue(diff["qualifies"])
        self.assertIn("a.yaw", diff["categories"]["physical_motion"])
        self.assertAlmostEqual(diff["values"]["a.yaw"]["minimal_circular_delta_rad"], 0.1)
        position = json.loads(json.dumps(base))
        position["a"]["xy"][0] += 0.01
        speed = self.changed("a", "signed_v", 0.1)
        self.assertTrue(reference_diff(base, position)["qualifies"])
        self.assertTrue(reference_diff(base, speed)["qualifies"])
        time_shift = json.loads(json.dumps(base))
        time_shift["t"] += 0.02
        self.assertEqual(reference_summaries(base)["meaningful_state_digest"],
                         reference_summaries(time_shift)["meaningful_state_digest"])
        self.assertIn("t", reference_diff(base, time_shift)["categories"]["metadata"])
        self.assertNotEqual(strict_replay_reference_digest(base), strict_replay_reference_digest(time_shift))

    def test_replay_validate_only_schedules_four_unique_prefixes(self):
        library = self.config["prefix_library"]
        plans = [library["prefixes"][i] for i in (0, 1, 6, 10)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis = root / "analysis"
            analysis.mkdir()
            selected_samples = [{"sample_id": f"sample_{i}",
                                 "prefix_id": plans[i % 4]["prefix_id"],
                                 "target_step": 3 if i % 2 else 4,
                                 "state_id": f"state_{i // 3}",
                                 "canonical_state": f"canonical_{i // 3}"}
                                for i in range(6)]
            samples = {"qualifying_group_count": 4,
                       "original_discovery_source_sha256": "original_source",
                       "analysis_source_sha256": self.config["source_sha256"],
                       "input_zip_sha256": "input_sha",
                       "groups": [{"state_id": f"state_{i}", "canonical_state": f"canonical_{i}",
                                   "samples": selected_samples[i * 3:(i + 1) * 3]}
                                  for i in range(2)]}
            (analysis / "samples_corrected.json").write_text(json.dumps(samples))
            expected = [{"state_id": f"state_{i}", "canonical_state": f"canonical_{i}",
                         "replay_reference_digest": f"digest_{i}", "score": 0.0,
                         "skill_status": {}, "visual_memory": {}, "finished": None,
                         "collapsed": False} for i in range(5)]
            plan = {"analysis_version": ANALYSIS_VERSION, "input_zip_sha256": "input_sha",
                    "original_source_sha256": "original_source",
                    "analysis_source_sha256": self.config["source_sha256"],
                    "corrected_samples_sha256": sha256_file(analysis / "samples_corrected.json"),
                    "prefix_library_hash": library["hash"], "game_seed": self.config["game_seed"],
                    "synthetic_seed": self.config["discovery_synthetic_seed"],
                    "prefixes": [{"prefix_id": p["prefix_id"], "prefix_action_ids": p["prefix_action_ids"],
                                  "prefix_actions": p["actions"], "expected_boundaries": expected}
                                 for p in plans],
                    "selected_samples": selected_samples}
            (analysis / "replay_plan.json").write_text(json.dumps(plan))
            tasks = []
            def fake_run_task(output, task, **_):
                tasks.append(task)
                return {"task_key": str(len(tasks)), "status": "passed", "wall_seconds": 1.0,
                        "result": {"status": "passed", "stop_reason": "planned_boundary", "boundaries": 5}}
            with patch("oracle_pilot_v21.run_task", side_effect=fake_run_task):
                result = replay_validate(analysis, root / "replay", self.config)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["passed_prefixes"], 4)
            self.assertEqual(result["covered_samples"], 6)
            self.assertEqual(len({t["prefix_id"] for t in tasks}), 4)
            self.assertTrue(all(t["mode"] == "replay_validate" and "candidate_action_id" not in t and
                                len(t["expected_boundaries"]) == 5 and
                                t["corrected_samples_sha256"] == plan["corrected_samples_sha256"]
                                for t in tasks))

    def test_cpu_four_window_replay_and_earliest_mismatch(self):
        plan = self.config["prefix_library"]["prefixes"][0]
        base = {"game_seed": self.config["game_seed"],
                "synthetic_seed": self.config["discovery_synthetic_seed"],
                "pixel_noise_std": 1.0, "miss_probability": 0.0,
                "continuation_action_id": self.config["continuation_action_id"],
                "prefix_action_ids": plan["prefix_action_ids"], "max_steps": 4}
        def one(root, task):
            match, engine, rig = VisionMatch(self.cfg), SemanticSkills(self.cfg), CameraRig()
            image = Image.new("RGB", (32, 32), "black")
            count = 0
            def capture():
                nonlocal count
                state = match.states[-1]
                filename = f"frames/{count:03d}.png"
                (root / "frames").mkdir(parents=True, exist_ok=True)
                image.save(root / filename)
                count += 1
                return {"file": filename, "simulation_seconds": state["t"],
                        "own_xy": state["xy"][0], "body_yaw_degrees": 0.,
                        "camera_pan_degrees": rig.pan, "camera_world_yaw_degrees": 0.,
                        "camera_eye": [state["xy"][0][0], state["xy"][0][1], .6],
                        "horizontal_fov_degrees": 90.}
            with patch("oracle_pilot_runtime.observe_current_scene", return_value={
                "status": "not_emitted", "reason": "no_visible_foot", "sighting": None,
                "candidates": [], "source": "simulator_geometry_truth"}):
                result = run(root, task, self.cfg, match, engine, rig, None, None, capture)
            return result, json.loads((root / "boundaries.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, boundaries = one(root / "source", {**base, "mode": "discover"})
            self.assertEqual(source["status"], "passed")
            expected = []
            for boundary in boundaries:
                item = boundary_evidence(boundary)
                item["collapsed"] = boundary["collapsed"]
                expected.append(item)
            good, _ = one(root / "good", {**base, "mode": "replay_validate",
                                         "expected_boundaries": expected})
            self.assertEqual(good["status"], "passed")
            self.assertEqual(good["decisions"], 4)
            self.assertEqual(good["boundaries"], 5)
            bad_expected = json.loads(json.dumps(expected))
            bad_expected[2]["score"] = 999.0
            bad, _ = one(root / "bad", {**base, "mode": "replay_validate",
                                       "expected_boundaries": bad_expected})
            self.assertEqual(bad["stop_reason"], "replay_mismatch")
            self.assertEqual(bad["replay_mismatch_detail"]["boundary_step"], 2)
            self.assertEqual(bad["decisions"], 2)


if __name__ == "__main__":
    unittest.main()
