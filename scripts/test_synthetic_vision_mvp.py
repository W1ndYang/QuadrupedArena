"""Isaac-free contract tests for the simulator-derived sighting MVP."""

import copy
import math
from pathlib import Path
import unittest

from adaptive_motion import load_config
from llm_trial import PROJECT
from semantic_skills import SemanticSkills
from vision_core import VisionMatch, public_packet
from synthetic_vision_mvp import (
    build_noisy_sighting, build_window_context, handoff_to_skill, run_validation_cases,
    sample_visibility, smoke_observation, smoke_request, summarize_skill_runtime_smoke,
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
        self.assertEqual(sample_visibility(missed), "visible")
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

    def test_runtime_smoke_recomputes_and_executes_native_skills(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["seed"] = 2026102107
        match = VisionMatch(cfg)
        engine = SemanticSkills(cfg)
        records = []
        calls = []
        for index in range(4):
            state = match.states[-1]
            f = {**frame(), "file": f"frames/{index:03d}_{state['t']:05.2f}.png",
                 "simulation_seconds": state["t"],
                 "camera_eye": [state["xy"][0][0] + .36, state["xy"][0][1], .6],
                 "camera_world_yaw_degrees": 0.}
            packet = public_packet(index, state, [f], cfg["start_b"])
            def sample(current_packet, current_frame):
                calls.append((current_packet["observation_id"],
                              current_frame["simulation_seconds"],
                              tuple(match.states[-1]["xy"][0])))
                current_a_x = match.states[-1]["xy"][0][0]
                c = candidate((400.0 - 5.0 * (current_a_x - cfg["start_a"][0]), 250.0))
                if index == 1:
                    c["visibility"] = {"status": "occluded"}
                if index == 2:
                    c["visibility"] = {"status": "unknown"}
                if index == 3:
                    c["projection"]["in_frame"] = False
                return build_noisy_sighting([c], current_packet, current_frame, 768, 432,
                                            seed=index, pixel_noise_std=0)
            synthetic = smoke_observation(packet, f, sample)
            before = copy.deepcopy(engine.memory.track)
            record = {"observation_id": index, "simulation_seconds": state["t"],
                      "frame": f, "synthetic": synthetic}
            if index == 3:
                records.append(record)
                break
            request = smoke_request(index, packet, synthetic["sighting"])
            self.assertEqual(set(request), {"observation_id", "skill", "params", "reason"})
            self.assertEqual("sighting" in request["params"], index == 0)
            engine.validate(request, packet)
            action, route, camera, status = engine.commit(request, packet)
            match.compile(action)
            self.assertIn(action["skill"], ("MOVE", "HOLD", "SCAN"))
            match.apply(action, route)
            before_xy = list(match.states[-1]["xy"][0])
            for step in range(100):
                own = public_packet(index, match.states[-1], [], cfg["start_b"])
                interruption = engine.tick(own)
                if interruption:
                    match.apply(*interruption)
                else:
                    for key in ("attack", "facing"):
                        match.active[key] = engine.action[key]
                match.advance(cfg["dt"])
            moved = math.dist(before_xy, match.states[-1]["xy"][0]) > .02
            record.update(action_request=request, action_status="executed",
                          actual_state_change=moved,
                          sighting_committed=index == 0 and engine.memory.track["feet"] == synthetic["sighting"]["feet"],
                          memory_check_passed=(engine.memory.track == before if index else
                                               engine.memory.track["seen_at"] == 0.0))
            if index == 1:
                self.assertEqual(engine.memory.track["seen_at"], before["seen_at"])
            records.append(record)
        self.assertEqual(len(calls), 4)
        self.assertEqual([item[1] for item in calls], [0., 2., 4., 6.])
        self.assertNotEqual(calls[0][2], calls[1][2])
        self.assertNotEqual(records[0]["synthetic"]["raw_pixel"],
                            records[1]["synthetic"]["candidates"][0]["projection"]["pixel"])
        summary = summarize_skill_runtime_smoke(records)
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["checks"]["actual_state_change"])
        self.assertEqual(summary["checks"]["sighting_commits"], 1)

    def test_runtime_smoke_does_not_pass_on_request_acceptance_alone(self):
        records = [{"observation_id": index, "simulation_seconds": 2. * index,
                    "frame": {"file": f"frames/{index:03d}.png"},
                    **({"action_request": {"skill": "MOVE_TO"}, "action_status": "executed",
                        "actual_state_change": False, "sighting_committed": index == 0,
                        "memory_check_passed": True} if index < 3 else {})}
                   for index in range(4)]
        result = summarize_skill_runtime_smoke(records)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["checks"]["actual_state_change"])

    def test_runtime_smoke_parser_isolated_from_diagnostic_poses(self):
        source = Path(__file__).with_name("vision_trial.py").read_text()
        self.assertIn("--synthetic-vision-skill-runtime-smoke", source)
        self.assertIn("sum((args.synthetic_vision_mvp, args.synthetic_vision_mvp_validation,", source)
        self.assertNotIn("args.synthetic_vision_skill_runtime_smoke", source.split("if args.g1b1_visible_spawn or", 1)[1].split("\n", 1)[0])
        self.assertNotIn("rep.orchestrator.step(", source)

    def test_half_second_windows_keep_55_sighting_at_its_own_time(self):
        engine = SemanticSkills(self.cfg)
        samples = []
        decisions = []
        for sample_id in range(17):
            t = sample_id * .5
            f = {**frame(), "file": f"frames/{sample_id:03d}_{t:05.2f}.png",
                 "simulation_seconds": t}
            state = copy.deepcopy(VisionMatch(self.cfg).states[-1]);state["t"] = t
            packet = public_packet(sample_id, state, [f], self.cfg["start_b"])
            c = candidate((400.0 + sample_id, 250.0))
            if sample_id != 11:
                c["visibility"] = {"status": "unknown"}
            synthetic = build_noisy_sighting([c], packet, f, 768, 432,
                                              seed=sample_id, pixel_noise_std=0)
            before = copy.deepcopy(engine.memory.track)
            if synthetic["sighting"] is not None:
                engine.observe_sighting(synthetic["sighting"], packet)
            sample = dict(observation_id=sample_id, simulation_seconds=t,
                          frame_file=f["file"],packet_file=f"observations/frames/{sample_id:03d}.json",
                          synthetic_file=f"synthetic_vision_frames/{sample_id:03d}.json",
                          status=sample_visibility(synthetic),reason=synthetic["reason"],
                          seed=sample_id,sighting=synthetic["sighting"],
                          ingest_status="accepted" if synthetic["sighting"] else "not_attempted_no_sighting",
                          memory_check_passed=True,
                          visibility_statuses=[c["visibility"]["status"]])
            samples.append(sample)
            if sample_id == 0:
                bootstrap = build_window_context(0, samples)
                decisions.append(bootstrap)
            elif sample_id % 4 == 0:
                context = build_window_context(sample_id // 4, samples[-4:])
                decisions.append(context)
                if sample_id == 12:
                    self.assertEqual(context["last_valid_sighting"]["simulation_seconds"], 5.5)
                    self.assertEqual(context["frames"][-1]["simulation_seconds"], 6.0)
                    self.assertIsNone(context["frames"][-1]["sighting"])
                    self.assertEqual(engine.memory.track["seen_at"], 5.5)
                    self.assertEqual(engine.sequence, 0)
                    decision_packet = public_packet(3, state, [f], self.cfg["start_b"])
                    request = smoke_request(3, decision_packet, None)
                    engine.validate(request, decision_packet)
                    engine.commit(request, decision_packet)
                    self.assertEqual(engine.memory.track["seen_at"], 5.5)
                    self.assertNotIn("sighting", request["params"])
            elif sample_id > 11:
                self.assertEqual(engine.memory.track, before)
        self.assertEqual([x["end_simulation_seconds"] for x in decisions], [0., 2., 4., 6., 8.])
        self.assertEqual([[f["simulation_seconds"] for f in x["frames"]] for x in decisions[1:]],
                         [[.5, 1., 1.5, 2.], [2.5, 3., 3.5, 4.],
                          [4.5, 5., 5.5, 6.], [6.5, 7., 7.5, 8.]])
        self.assertEqual(len({s["observation_id"] for s in samples}), 17)
        self.assertEqual(engine.sequence, 1)  # Only the explicit 6.0 s request decided.
        request = smoke_request(2, public_packet(2, state, [f], self.cfg["start_b"]), None)
        self.assertNotIn("sighting", request["params"])
        self.assertEqual(request["skill"], "MOVE_TO")

    def test_window_rejects_duplicate_and_wrong_time_samples(self):
        samples = [dict(observation_id=i,simulation_seconds=.5 * (i + 1),
                        frame_file=f"frames/{i:03d}.png",ingest_status="not_attempted_no_sighting")
                   for i in range(4)]
        self.assertTrue(build_window_context(1,samples)["no_valid_sighting"])
        bad = copy.deepcopy(samples);bad[2]["observation_id"] = bad[1]["observation_id"]
        with self.assertRaises(ValueError):build_window_context(1,bad)
        bad = copy.deepcopy(samples);bad[2]["simulation_seconds"] = 1.6
        with self.assertRaises(ValueError):build_window_context(1,bad)
        with self.assertRaises(ValueError):build_window_context(1,samples[:3])

    def test_frame_ingest_uses_visual_memory_validator_and_preserves_old_time(self):
        engine = SemanticSkills(self.cfg)
        f = {**frame(), "simulation_seconds": 5.5, "file": "frames/011_05.50.png"}
        state = copy.deepcopy(VisionMatch(self.cfg).states[-1]);state["t"] = 5.5
        packet = public_packet(11,state,[f],self.cfg["start_b"])
        engine.observe_sighting({"time": 5.5, "feet": [400.0,250.0]},packet)
        self.assertEqual(engine.memory.track["seen_at"],5.5)
        self.assertEqual(engine.sequence,0)
        old = copy.deepcopy(engine.memory.track)
        f2 = {**f,"simulation_seconds": 6.0,"file": "frames/012_06.00.png"}
        state["t"] = 6.0
        packet2 = public_packet(12,state,[f2],self.cfg["start_b"])
        engine.observe_sighting(None,packet2)
        self.assertEqual(engine.memory.track,old)
        with self.assertRaisesRegex(ValueError,"input frame"):
            engine.observe_sighting({"time": 5.5,"feet": [400.0,250.0]},packet2)
        self.assertEqual(engine.memory.track,old)

    def test_windowed_summary_checks_timing_without_forcing_sighting(self):
        samples = [dict(observation_id=i,simulation_seconds=.5*i,
                        frame_file=f"frames/{i:03d}.png",ingest_status="not_attempted_no_sighting",
                        memory_check_passed=True) for i in range(17)]
        windows = [build_window_context(0,samples[:1])]+[
            build_window_context(i,samples[1+4*(i-1):1+4*i]) for i in (1,2,3,4)]
        records = [dict(observation_id=i,simulation_seconds=2.*i,
                        action_request={"skill":"MOVE_TO"},action_status="executed",
                        actual_state_change=i==0,memory_check_passed=True) for i in range(4)]
        records.append(dict(observation_id=4,simulation_seconds=8.0))
        result = summarize_skill_runtime_smoke(records,windows,samples)
        self.assertEqual(result["status"],"passed")
        self.assertEqual(result["checks"]["frame_sighting_ingests"],0)
        bad = copy.deepcopy(samples);bad[11]["simulation_seconds"] = 5.6
        self.assertEqual(summarize_skill_runtime_smoke(records,windows,bad)["status"],"partial")
        bad = copy.deepcopy(samples);bad[11]["reason"] = "adapter_failed: RuntimeError: unavailable"
        self.assertEqual(summarize_skill_runtime_smoke(records,windows,bad)["status"],"partial")
        bad_windows = copy.deepcopy(windows)
        bad_windows[3]["frames"][2]["frame_file"] = "frames/stale.png"
        self.assertFalse(summarize_skill_runtime_smoke(records,bad_windows,samples)
                         ["checks"]["window_refs_match_samples"])


if __name__ == "__main__":
    unittest.main()
