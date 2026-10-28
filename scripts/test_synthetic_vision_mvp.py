"""Isaac-free contract tests for the simulator-derived sighting MVP."""

import copy
from pathlib import Path
import unittest

from adaptive_motion import load_config
from llm_trial import PROJECT
from semantic_skills import SemanticSkills
from vision_core import VisionMatch, public_packet
from synthetic_vision_mvp import (
    build_noisy_sighting, handoff_to_skill, run_validation_cases,
    validation_pose, visibility_from_ray,
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

    def test_three_fresh_cycles_and_memory_boundary(self):
        import copy
        from tactical_logic import block
        from trial_rules import covers_at

        engine = SemanticSkills(self.cfg)
        original = VisionMatch(self.cfg).states[-1]
        calls = []

        def sample(index, name):
            pose = validation_pose(index, original, self.cfg)
            state = pose["state"]
            state["t"] = float(index)
            f = {**frame(), "file": f"frames/{index:03d}.png",
                 "simulation_seconds": float(index)}
            packet = public_packet(index, state, [f], self.cfg["start_b"])
            xy = pose["authored_actor_xy"]
            cover = block([*xy[0], .6], [*xy[1], .035], covers_at(self.cfg, state["t"]))
            pixel = (400.0 - 60.0 * (self.cfg["start_b"][0] - xy[1][0]), 250.0)
            c = candidate(pixel)
            c["world_m"] = [*xy[1], .035]
            c["visibility"] = visibility_from_ray(3.5, cover, {"hit": False}, c["link_prim"])
            c["raycast"] = {"hit": False}
            synthetic = build_noisy_sighting([c], packet, f, 768, 432,
                                              seed=7 + index, pixel_noise_std=0)
            calls.append((index, name, copy.deepcopy(xy), f["file"]))
            return dict(frame=f, packet=packet, synthetic=synthetic,
                        authored_actor_xy=xy, stage_actor_root_link_world_m={"B": [*xy[1], 1.0]})

        progress = []
        result = run_validation_cases(engine, sample, lambda records: progress.append(len(records)))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(progress, [1, 1, 2, 2, 3, 3])
        self.assertEqual([r["status"] for r in result["cases"]], ["passed"] * 3)
        self.assertEqual([r["simulation_seconds"] for r in result["cases"]], [0., 1., 2.])
        self.assertEqual([r["observation_id"] for r in result["cases"]], [0, 1, 2])
        self.assertEqual(result["cases"][1]["synthetic"]["skill_handoff"]["status"],
                         "not_attempted_no_sighting")
        self.assertEqual(result["cases"][1]["memory_before"], result["cases"][1]["memory_after"])
        self.assertEqual(result["cases"][1]["memory_after"]["seen_at"], 0.)
        self.assertEqual(result["cases"][2]["memory_after"]["seen_at"], 2.)
        self.assertNotEqual(result["cases"][0]["synthetic"]["raw_pixel"],
                            result["cases"][2]["synthetic"]["raw_pixel"])
        self.assertNotEqual(calls[1][2][1], calls[2][2][1])
        self.assertEqual(engine.sequence, 2)

    def test_validation_unknown_outside_and_rejected_stay_partial(self):
        for mode in ("unknown", "outside", "rejected"):
            with self.subTest(mode=mode):
                engine = SemanticSkills(self.cfg)

                def sample(index, name):
                    f = {**frame(), "file": f"frames/{index:03d}.png",
                         "simulation_seconds": float(index)}
                    state = copy.deepcopy(VisionMatch(self.cfg).states[-1])
                    state["t"] = float(index)
                    packet = public_packet(index, state, [f], self.cfg["start_b"])
                    c = candidate((400., 1.) if mode == "rejected" and index == 0 else (400., 250.))
                    c["raycast"] = {"hit": False}
                    if index == 1:
                        c["visibility"] = {"status": "occluded"}
                    if index == 0 and mode == "unknown":
                        c["visibility"] = {"status": "unknown"}
                    if index == 0 and mode == "outside":
                        c["projection"]["in_frame"] = False
                    synthetic = build_noisy_sighting([c], packet, f, 768, 432,
                                                      pixel_noise_std=0)
                    return dict(frame=f, packet=packet, synthetic=synthetic,
                                authored_actor_xy=[[4.8, -1.0], [5.0 - .4 * (index == 2),
                                                               2.8 - .4 * (index == 2)]])

                result = run_validation_cases(engine, sample)
                self.assertEqual(result["status"], "partial")
                self.assertEqual(result["cases"][0]["status"], "partial")
                self.assertEqual(len(result["cases"]), 3)
                if mode == "rejected":
                    self.assertEqual(result["cases"][0]["synthetic"]["skill_handoff"]["status"],
                                     "rejected_by_validator")
                else:
                    self.assertEqual(result["cases"][0]["synthetic"]["skill_handoff"]["status"],
                                     "not_attempted_no_sighting")

    def test_formal_modes_keep_one_shot_and_avoid_blocked_step(self):
        source = Path(__file__).with_name("vision_trial.py").read_text()
        self.assertIn("if args.synthetic_vision_mvp:", source)
        self.assertIn("if args.synthetic_vision_mvp_validation:", source)
        self.assertLess(source.index("if args.synthetic_vision_mvp_validation:",
                                     source.index("def capture():") if "def capture():" in source else
                                     source.index("def capture(state=None):")),
                        source.index("window=[capture()]"))
        self.assertNotIn("rep.orchestrator.step(", source)


if __name__ == "__main__":
    unittest.main()
