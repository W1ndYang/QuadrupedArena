"""Pure Python checks for the formal-scene diagnostic decision boundary."""

import argparse
import copy
import math
import unittest

from formal_geometry_diagnostic import (
    _scenarios, classify_visibility, diagnostic_mode_error, project_local_candidate,
    record_is_structurally_same_frame,
)
from adaptive_motion import load_config
from llm_trial import PROJECT
from tactical_logic import block


class FormalGeometryDiagnosticTests(unittest.TestCase):
    INTRINSICS = (12.0, 24.0, 13.5)
    CLIPPING = (0.04, 100.0)

    def test_reuses_g1a_projection_and_image_coordinates(self):
        center = project_local_candidate((0, 0, -2), self.INTRINSICS, self.CLIPPING)
        right = project_local_candidate((0.5, 0, -2), self.INTRINSICS, self.CLIPPING)
        above = project_local_candidate((0, 0.5, -2), self.INTRINSICS, self.CLIPPING)
        self.assertEqual(center["status"], "in_frame")
        self.assertEqual(center["pixel"], [384.0, 216.0])
        self.assertEqual(right["pixel"], [480.0, 216.0])
        self.assertEqual(above["pixel"], [384.0, 120.0])

    def test_projection_rejects_behind_clipping_edges_and_invalid(self):
        cases = [((0, 0, 0), "behind_camera"), ((0, 0, 1), "behind_camera"),
                 ((0, 0, -0.01), "out_of_clipping"), ((0, 0, -101), "out_of_clipping"),
                 ((2, 0, -2), "out_of_frame"), ((-2, 0, -2), "in_frame"),
                 ((0, -1.125, -2), "out_of_frame"),
                 ((math.nan, 0, -2), "invalid"), ((0, 0, -math.inf), "invalid")]
        for point, status in cases:
            with self.subTest(point=point):
                self.assertEqual(project_local_candidate(point, self.INTRINSICS, self.CLIPPING)["status"], status)
        self.assertEqual(project_local_candidate((0, 0, -2), (0, 24, 13.5), self.CLIPPING)["status"], "invalid")
        self.assertEqual(project_local_candidate((0, 0, -2), self.INTRINSICS, (2, 1))["status"], "invalid")

    def test_visibility_is_conservative(self):
        foot = "/World/Actors/B/left_ankle_roll_link"
        self.assertEqual(classify_visibility(3, None, {"hit": False})["status"], "unknown")
        self.assertEqual(classify_visibility(3, None, {"error": "no query"})["status"], "unknown")
        cover = (.5, "High2", [2, 1, 0.5])
        self.assertEqual(classify_visibility(3, cover, {"hit": False})["status"], "occluded")
        self.assertEqual(classify_visibility(3, None, {"hit": True, "collision": "/World/Covers/High2/Collider", "distance": 1.5})["status"], "occluded")
        hit = {"hit": True, "collision": foot + "/visual0", "distance": 2.99}
        self.assertEqual(classify_visibility(3, None, hit, foot, False)["status"], "unknown")
        self.assertEqual(classify_visibility(3, None, hit, foot, True)["status"], "visible_candidate")
        self.assertEqual(classify_visibility(3, None, {"hit": True, "collision": foot + "/visual0"}, foot, True)["status"], "unknown")
        self.assertEqual(classify_visibility(3, None, {"hit": True, "collision": "", "distance": 1})["status"], "unknown")
        self.assertEqual(classify_visibility(3, None, {"hit": True, "collision": foot + "/visual0", "distance": 1}, foot, True)["status"], "unknown")

    def test_deterministic_formal_cover_layout(self):
        cfg = load_config(PROJECT / "configs/trial_rotating.json")
        state = {"t": 0.0, "xy": [list(cfg["start_a"]), list(cfg["start_b"])],
                 "yaw": [0.0, math.pi], "speed": [0.0, 0.0]}
        scenes = _scenarios(state, cfg)
        self.assertEqual([name for name, _ in scenes], ["expected_clear", "expected_high2_occluded"])
        for name, sample in scenes:
            x, y = sample["xy"][0]
            yaw = sample["yaw"][0]
            eye = (x + 0.36 * math.cos(yaw), y + 0.36 * math.sin(yaw), 0.6)
            hit = block(eye, (*cfg["start_b"], 0.035), cfg["arena"]["covers"])
            if name == "expected_clear":
                self.assertIsNone(hit)
            else:
                self.assertIsNotNone(hit)
                self.assertEqual(hit[1], "High2")

    def test_candidate_record_has_source_and_conservative_status(self):
        candidate = {"link_prim": "/World/Actors/B/right_ankle_roll_link",
                     "source": "g1_visual_mesh_vertex", "local_vertex_m": [0, 0, -0.04],
                     "world_m": [5, 2.8, 0.02], "verification": "mesh_vertex_candidate_unverified",
                     "projection": {"status": "in_frame", "pixel": [400, 250]},
                     "visibility": classify_visibility(3, None, {"hit": False})}
        self.assertEqual(candidate["visibility"]["status"], "unknown")
        self.assertNotIn("sighting", candidate)
        self.assertTrue(candidate["link_prim"].endswith("right_ankle_roll_link"))

    def test_record_frame_identity(self):
        record = {"frame_index": 1, "simulation_seconds": 0.0,
                  "frame": {"frame_index": 1, "simulation_seconds": 0.0,
                            "rgb_file": "frames/frame_001_rgb.png"},
                  "camera": {"world_transform_rows": [[1, 0, 0, 0]]},
                  "feet": [{"frame_index": 1}, {"frame_index": 1}]}
        self.assertTrue(record_is_structurally_same_frame(record))
        bad = copy.deepcopy(record)
        bad["feet"][0]["frame_index"] = 0
        self.assertFalse(record_is_structurally_same_frame(bad))
        bad = copy.deepcopy(record)
        bad["frame"]["simulation_seconds"] = 0.5
        self.assertFalse(record_is_structurally_same_frame(bad))

    def test_diagnostic_mode_is_explicit_and_isolated(self):
        values = dict(geometry_diagnostic=True, vision_trial=True, vision_smoke=False,
                      semantic_skills=False, deepseek=False, qwen_vision=None,
                      assessment=False, g1b1_visible_spawn=False)
        self.assertIsNone(diagnostic_mode_error(argparse.Namespace(**values)))
        for key, value in (("vision_trial", False), ("vision_smoke", True),
                           ("semantic_skills", True), ("deepseek", True),
                           ("qwen_vision", "max"), ("assessment", True),
                           ("g1b1_visible_spawn", True)):
            with self.subTest(flag=key):
                changed = dict(values, **{key: value})
                self.assertIsNotNone(diagnostic_mode_error(argparse.Namespace(**changed)))
        self.assertIsNone(diagnostic_mode_error(argparse.Namespace(**dict(values, geometry_diagnostic=False))))


if __name__ == "__main__":
    unittest.main()
