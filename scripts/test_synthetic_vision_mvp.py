"""Isaac-free contract tests for the simulator-derived sighting MVP."""

import copy
import unittest

from adaptive_motion import load_config
from llm_trial import PROJECT
from semantic_skills import SemanticSkills
from vision_core import VisionMatch, public_packet
from synthetic_vision_mvp import (
    build_noisy_sighting, handoff_to_skill, visibility_from_ray,
)


def frame():
    return {"file": "frames/000_00.00.png", "simulation_seconds": 0.0,
            "camera_eye": [4.8, -0.64, 0.6], "camera_world_yaw_degrees": 90.0,
            "body_yaw_degrees": 90.0, "camera_pan_degrees": 0.0,
            "horizontal_fov_degrees": 90.0}


def candidate(pixel=(400.0, 250.0), visible=True):
    return {"link_prim": "/World/Actors/B/left_ankle_roll_link",
            "world_m": [5.0, 2.8, 0.035],
            "projection": {"pixel": list(pixel), "in_frame": visible, "depth_m": 3.5},
            "visibility": {"status": "visible" if visible else "occluded"}}


class SyntheticVisionMvpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(PROJECT / "configs/trial_rotating.json")

    def packet(self):
        match = VisionMatch(self.cfg)
        return public_packet(0, match.states[-1], [frame()], self.cfg["start_b"])

    def test_visible_segment_and_cover_or_physx_occlusion(self):
        link = "/World/Actors/B/left_ankle_roll_link"
        self.assertEqual(visibility_from_ray(3.5, None, {"hit": False}, link)["status"], "visible")
        self.assertEqual(visibility_from_ray(3.5, (0.4, "High2", [1, 2, 0.5]),
                                             {"hit": False}, link)["status"], "occluded")
        self.assertEqual(visibility_from_ray(3.5, None, {"hit": True,
            "collision": "/World/Covers/High2/Collider", "distance": 2.0}, link)["status"], "occluded")
        self.assertEqual(visibility_from_ray(3.5, None, {"hit": True,
            "collision": link + "/visual0", "distance": 3.49}, link)["status"], "visible")
        self.assertEqual(visibility_from_ray(3.5, None, {"hit": True,
            "collision": link + "/visual0", "distance": 3.3}, link)["status"], "unknown")
        self.assertEqual(visibility_from_ray(3.5, None, {"hit": True,
            "collision": "/World/Actors/B/pelvis", "distance": 3.0}, link)["status"], "unknown")
        self.assertEqual(visibility_from_ray(3.5, None, {"hit": True,
            "collision": "/World/Floor/Collider", "distance": 3.48}, link)["status"], "unknown")
        self.assertEqual(visibility_from_ray(3.5, None, {"error": "unavailable"}, link)["status"], "unknown")

    def test_schema_and_skill_handoff(self):
        packet = self.packet()
        record = build_noisy_sighting([candidate()], packet, frame(), 768, 432,
                                      seed=7, pixel_noise_std=0, miss_probability=0)
        self.assertEqual(record["sighting"], {"time": 0.0, "feet": [400.0, 250.0]})
        self.assertEqual(set(record["sighting"]), {"time", "feet"})
        self.assertEqual(record["source"], "simulator_geometry_truth")
        self.assertFalse(record["same_frame_verified"])
        engine = SemanticSkills(self.cfg)
        original = copy.deepcopy(engine.memory.track)
        request = {"observation_id": 0, "skill": "OBSERVE",
                   "params": {"sighting": record["sighting"]}}
        prepared = engine.validate(request, packet)
        self.assertEqual(engine.memory.track, original)
        self.assertEqual(prepared.track["frame"], frame()["file"])
        handed = handoff_to_skill(record, engine, packet)
        self.assertEqual(handed["skill_handoff"]["status"], "accepted_and_committed")
        self.assertEqual(engine.memory.track["seen_at"], 0.0)
        self.assertEqual(engine.memory.track["feet"], [400.0, 250.0])
        self.assertEqual(engine.sequence, 1)

    def test_behind_outside_occluded_and_unknown_do_not_emit(self):
        packet = self.packet()
        for item in (candidate(visible=False),
                     {**candidate(), "projection": {"in_frame": False, "pixel": None}},
                     {**candidate(), "visibility": {"status": "unknown"}}):
            record = build_noisy_sighting([item], packet, frame(), 768, 432)
            self.assertIsNone(record["sighting"])
            self.assertEqual(record["reason"], "no_visible_foot")
            self.assertEqual(handoff_to_skill(record, SemanticSkills(self.cfg), packet)
                             ["skill_handoff"]["status"], "not_attempted_no_sighting")

    def test_noise_seed_bounds_and_miss(self):
        packet = self.packet()
        a = build_noisy_sighting([candidate()], packet, frame(), 768, 432, seed=91,
                                 pixel_noise_std=2.0)
        b = build_noisy_sighting([candidate()], packet, frame(), 768, 432, seed=91,
                                 pixel_noise_std=2.0)
        self.assertEqual(a["sighting"], b["sighting"])
        self.assertNotEqual(a["sighting"]["feet"], [400.0, 250.0])
        missed = build_noisy_sighting([candidate()], packet, frame(), 768, 432,
                                      miss_probability=1.0)
        self.assertIsNone(missed["sighting"])
        self.assertEqual(missed["reason"], "synthetic_miss")
        out = build_noisy_sighting([candidate((0.1, 0.1))], packet, frame(), 768, 432,
                                   seed=1, pixel_noise_std=1000000.0)
        self.assertIsNone(out["sighting"])
        self.assertEqual(out["reason"], "noisy_pixel_out_of_frame")

    def test_packet_time_match_and_validator_rejection(self):
        packet = self.packet()
        packet["simulation_seconds"] = 0.5
        with self.assertRaisesRegex(ValueError, "time differ"):
            build_noisy_sighting([candidate()], packet, frame(), 768, 432)
        packet["simulation_seconds"] = 0.0
        record = build_noisy_sighting([candidate((400.0, 1.0))], packet, frame(), 768, 432,
                                      pixel_noise_std=0)
        rejected = handoff_to_skill(record, SemanticSkills(self.cfg), packet)
        self.assertEqual(rejected["skill_handoff"]["status"], "rejected_by_validator")

    def test_invalid_noise_parameters(self):
        packet = self.packet()
        for kwargs in ({"seed": True}, {"pixel_noise_std": -1},
                       {"miss_probability": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_noisy_sighting([candidate()], packet, frame(), 768, 432, **kwargs)


if __name__ == "__main__":
    unittest.main()
