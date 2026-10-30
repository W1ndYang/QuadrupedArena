"""CPU-only contracts for the first Oracle state abstraction slice."""

from dataclasses import fields, replace
from collections import Counter
import copy
import math
import subprocess
import sys
import unittest

from adaptive_motion import load_config, visible_observation
from oracle_state_v1 import (
    LimitedObservationInput, Region, RegionCatalog, _aim_bin, _attack_geometry, _boundary,
    _cooldown, _deadline, _dwell, _memory_age, _relation, _visual_bin,
    action_catalog, action_template, capture_reference, encode_full, encode_limited,
    limited_from_public, scene_profile_id,
)
from oracle_state_v1_smoke import encode_boundary
from semantic_skills import SemanticSkills, VisualMemory
from trial_motion import Planner, clearance
from trial_rules import covers_at
from vision_core import CameraRig, VisionMatch, public_packet


def fixture():
    cfg = load_config("configs/trial_rotating.json")
    cfg["start_a"] = [-4., -1.5]
    cfg["start_b"] = [-1., 3.25]
    match = VisionMatch(cfg)
    engine = SemanticSkills(cfg)
    catalog = RegionCatalog.from_config(cfg)
    frame = {"simulation_seconds": 0., "camera_world_yaw_degrees": 0.,
             "camera_eye": [-3.64, -1.5, .6], "file": "frame.png"}
    packet = public_packet(0, match.states[-1], [frame], cfg["start_b"])
    packet["skills"] = engine.public(0)
    return cfg, match, engine, catalog, packet


class OracleStateV1Tests(unittest.TestCase):
    def test_region_catalog_uses_configured_geometry_and_planner(self):
        cfg = load_config("configs/trial_rotating.json")
        catalog = RegionCatalog.from_config(cfg)
        self.assertIsNone(Region("unverified", "corridor", "example_only", "unverified").adjacent_pre_collapse)
        self.assertEqual(catalog.coverage, "planner_free_space")
        self.assertEqual({r.role for r in catalog.regions}, {"target_zone", "cover_zone", "open_zone"})
        self.assertNotIn("corridor", {r.role for r in catalog.regions})
        for phase, t in (("pre_collapse", 0.), ("post_collapse", 20.)):
            planner = Planner(cfg, covers_at(cfg, t))
            for key in ("start_a", "start_b"):
                self.assertIsNotNone(catalog.locate(cfg[key], phase))
            for point in cfg["arena"]["task_points"]:
                self.assertEqual(catalog.locate(point["center"], phase), "target_" + point["name"])
                # The target circle owns its edge even when a cover band overlaps.
                edge = (point["center"][0] + point["radius"], point["center"][1])
                self.assertEqual(catalog.locate(edge, phase), "target_" + point["name"])
            for cover in planner.covers:
                center = cover["center"]
                self.assertIsNone(catalog.locate(center, phase))
                free_band_nodes = [n for n in planner.free_nodes
                                   if catalog.locate((n[0]*planner.grid, n[1]*planner.grid), phase)
                                   == "cover_" + cover["name"]]
                self.assertTrue(free_band_nodes, cover["name"])
                xy = tuple(v*planner.grid for v in min(free_band_nodes))
                self.assertEqual(catalog.cover_relation(xy, phase), "in_cover_zone")
                self.assertGreaterEqual(clearance(xy, cover), cfg["planner_clearance"] - 1e-10)
            overlap_nodes = 0
            for node in planner.free_nodes:
                xy = (node[0]*planner.grid, node[1]*planner.grid)
                if any(math.dist(xy, p["center"]) <= p["radius"] + 1e-10
                       for p in cfg["arena"]["task_points"]):
                    continue
                candidates = sorted((clearance(xy, cover), cover["name"]) for cover in planner.covers
                                    if clearance(xy, cover) <= catalog.cover_band_width + 1e-10)
                if len(candidates) > 1:
                    overlap_nodes += 1
                    self.assertEqual(catalog.locate(xy, phase), "cover_" + candidates[0][1])
            self.assertGreater(overlap_nodes, 0)
            for start in (cfg["start_a"], cfg["start_b"]):
                for point in cfg["arena"]["task_points"]:
                    route = planner.route(start, point["center"])
                    self.assertIsNotNone(route)
                    previous = start
                    for endpoint in route:
                        for u in (0., .25, .5, .75, 1.):
                            xy = tuple((1-u)*a + u*b for a, b in zip(previous, endpoint))
                            self.assertIsNotNone(catalog.locate(xy, phase), (phase, xy))
                        previous = endpoint
            for region in catalog.regions:
                self.assertIsNotNone(catalog.adjacent(region.region_id, phase))
                for neighbor in catalog.adjacent(region.region_id, phase):
                    self.assertIn(region.region_id, catalog.adjacent(neighbor, phase))
            self.assertIsNone(catalog.locate((100., 100.), phase))
            self.assertEqual(catalog.cover_relation((100., 100.), phase), "unknown")
        self.assertNotEqual(catalog.adjacent("cover_High3", "pre_collapse"),
                            catalog.adjacent("cover_High3", "post_collapse"))

    def test_real_boundary_capture_and_stability(self):
        cfg, match, engine, catalog, packet = fixture()
        before = (match.tick, match.t, match.ref.score, match.ref.generation,
                  tuple(match.ref.aim), tuple(match.ref.threshold))
        snap = capture_reference(match, engine, CameraRig())
        first = encode_full(snap, cfg, catalog)
        self.assertEqual(first.status, "valid")
        self.assertEqual(first, encode_full(snap, cfg, catalog))
        self.assertEqual(before, (match.tick, match.t, match.ref.score, match.ref.generation,
                                  tuple(match.ref.aim), tuple(match.ref.threshold)))
        self.assertNotIn("threshold", {f.name for f in fields(snap)})
        self.assertNotIn("rng", {f.name for f in fields(snap)})
        self.assertEqual(first.fields, tuple(sorted(first.fields)))
        self.assertNotIn("score", dict(first.fields))
        reordered = dict(reversed(list(cfg.items())))
        self.assertEqual(first.state_id, encode_full(snap, reordered, RegionCatalog.from_config(reordered)).state_id)
        script = ("from test_oracle_state_v1 import fixture; "
                  "from oracle_state_v1 import capture_reference,encode_full; "
                  "from vision_core import CameraRig; "
                  "c,m,e,r,p=fixture(); print(encode_full(capture_reference(m,e,CameraRig()),c,r).state_id)")
        child = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
        self.assertEqual(first.state_id, child)

    def test_limited_whitelist_and_id_separation(self):
        cfg, match, engine, catalog, packet = fixture()
        limited_names = {f.name for f in fields(LimitedObservationInput)}
        for forbidden in ("b", "human", "ref", "rng", "threshold", "candidates", "raw_pixel"):
            self.assertNotIn(forbidden, limited_names)
        obs = limited_from_public(packet, (None,), cfg)
        state = encode_limited(obs, cfg, catalog)
        self.assertEqual(state.status, "valid")
        self.assertEqual(state, encode_limited(obs, cfg, catalog))
        self.assertEqual(dict(state.fields)["window.valid_sighting_mask"], "0")
        self.assertEqual(dict(state.fields)["window.latest_sighting_region"], "unknown")
        self.assertEqual(dict(state.fields)["A.visual_memory_source"], "known_initial")
        self.assertNotIn("B.hp", dict(state.fields))
        self.assertNotIn("own_previous_action_outcome", dict(state.fields))
        altered = copy.deepcopy(packet)
        altered["hidden_b_xy"] = [999, 999]
        altered["candidates"] = [{"true_xy": [999, 999]}]
        altered["frames"][0]["raw_pixel"] = [1, 2]
        self.assertEqual(state, encode_limited(limited_from_public(altered, (None,), cfg), cfg, catalog))
        match.ref.hp[1] = 0
        match.b.xy[:] = [3., 3.]
        match.human.state = "ELIMINATED"
        self.assertEqual(state, encode_limited(limited_from_public(packet, (None,), cfg), cfg, catalog))
        full = encode_full(capture_reference(match, engine, CameraRig()), cfg, catalog)
        self.assertNotEqual(state.state_id, full.state_id)
        self.assertNotEqual(state.state_id, encode_limited(obs, cfg, catalog, version="oracle_state_v2").state_id)
        cfg2 = copy.deepcopy(cfg)
        cfg2["seed"] += 1
        self.assertNotEqual(scene_profile_id(cfg), scene_profile_id(cfg2))
        with self.assertRaises(ValueError):
            encode_limited(obs, cfg2, RegionCatalog.from_config(cfg2))

    def test_limited_visual_region_respects_public_uncertainty_and_hidden_truth(self):
        cfg, match, engine, catalog, packet = fixture()
        initial = limited_from_public(packet, (None,), cfg)
        self.assertEqual(dict(encode_limited(initial, cfg, catalog).fields)["A.visual_track_region"], "target_P2")
        self.assertIsNone(catalog.locate_uncertain(tuple(cfg["start_b"]), .6, "pre_collapse"))
        self.assertIsNone(catalog.locate_uncertain((-1. + .5, 3.25), .1, "pre_collapse"))
        self.assertEqual(catalog.locate_uncertain((-4., .5), .6, "pre_collapse"), "open_traversable")
        public_packet_with_track = copy.deepcopy(packet)
        memory = public_packet_with_track["skills"]["visual_memory"]
        memory.update(xy=[-4., .5], seen_at=0., source="rgb_foot_pixel",
                      base_uncertainty=.6, age=0., uncertainty_radius=.6, aim_usable=True)
        observed = limited_from_public(public_packet_with_track, (None,), cfg)
        limited = encode_limited(observed, cfg, catalog)
        self.assertEqual(dict(limited.fields)["A.visual_track_region"], "open_traversable")
        self.assertEqual(dict(encode_limited(replace(observed, memory_xy=(-1., 3.25)), cfg, catalog).fields)
                         ["A.visual_track_region"], "unknown")
        self.assertEqual(dict(encode_limited(replace(observed, memory_uncertainty_radius=2.), cfg, catalog).fields)
                         ["A.visual_track_region"], "unknown")
        self.assertEqual(dict(encode_limited(replace(observed, memory_xy=None, memory_seen_at=None,
                                                     memory_uncertainty_radius=None), cfg, catalog).fields)
                         ["A.visual_track_region"], "unknown")
        self.assertEqual(dict(encode_limited(replace(observed, memory_xy=None, memory_expired=True,
                                                     memory_uncertainty_radius=None), cfg, catalog).fields)
                         ["A.visual_track_region"], "unknown")
        expired = replace(observed, t=6., observation_id=3,
                          frames=tuple(replace(observed.frames[0], time=t) for t in (4.5, 5., 5.5, 6.)),
                          previous_action_id=action_catalog(cfg)[0].template_id,
                          memory_xy=None, memory_seen_at=0., memory_source=None,
                          memory_expired=True, memory_uncertainty_radius=None)
        self.assertEqual(encode_limited(expired, cfg, catalog).status, "valid")
        self.assertEqual(dict(encode_limited(expired, cfg, catalog).fields)["A.visual_track_region"], "unknown")
        # A feet pixel projecting to the P2 centre still cannot certify the
        # entire fresh 0.6 m sighting disk inside P2's 0.55 m radius.
        centre_frame = replace(observed.frames[0], camera_eye=(-1.881407077, 3.25, .6),
                               feet=(384., 431.))
        self.assertEqual(dict(encode_limited(replace(observed, frames=(centre_frame,)), cfg, catalog).fields)
                         ["window.latest_sighting_region"], "unknown")
        full_before = encode_full(capture_reference(match, engine, CameraRig()), cfg, catalog)
        match.b.xy[:] = [3.7, 1.5]
        match.human.state = "ELIMINATED"
        self.assertEqual(limited, encode_limited(limited_from_public(public_packet_with_track, (None,), cfg), cfg, catalog))
        full_after = encode_full(capture_reference(match, engine, CameraRig()), cfg, catalog)
        self.assertNotEqual(full_before.state_id, full_after.state_id)
        self.assertEqual(dict(full_after.fields)["B.region_id"], "cover_High2")

    def test_fixed_reference_eta_and_limited_motion_isolation(self):
        cfg, match, engine, catalog, packet = fixture()
        original = encode_full(capture_reference(match, engine, CameraRig()), cfg, catalog)
        for skill, params in (("MOVE_TO", {"target": "P2", "risk": "fast"}),
                              ("CAPTURE_POINT", {"risk": "safe"}),
                              ("OBSERVE", {"direction": "west"})):
            action_template(skill, params, cfg)
            engine.intent = skill
            engine.params = params
            changed = encode_full(capture_reference(match, engine, CameraRig()), cfg, catalog)
            self.assertEqual(original.state_id, changed.state_id)
            self.assertEqual(dict(original.fields)["A.target_eta_bin"], dict(changed.fields)["A.target_eta_bin"])
            self.assertEqual(dict(original.fields)["A.active_target_relation"],
                             dict(changed.fields)["A.active_target_relation"])
        engine.memory.track = {"xy": [0., 0.], "seen_at": 0.,
                               "source": "rgb_foot_pixel", "base_uncertainty": .6}
        different_memory = encode_full(capture_reference(match, engine, CameraRig()), cfg, catalog)
        self.assertEqual(dict(original.fields)["A.target_eta_bin"],
                         dict(different_memory.fields)["A.target_eta_bin"])
        self.assertEqual(dict(original.fields)["A.active_target_relation"],
                         dict(different_memory.fields)["A.active_target_relation"])
        obs = replace(limited_from_public(packet, (None,), cfg), own_speed=.3)
        limited = encode_limited(obs, cfg, catalog)
        self.assertEqual(limited.status, "valid")
        self.assertEqual(dict(limited.fields)["A.motion_bin"], "moving")
        match.a.v = .3
        forward_full = encode_full(capture_reference(match, engine, CameraRig()), cfg, catalog)
        match.a.v = -.3
        reversed_full = encode_full(capture_reference(match, engine, CameraRig()), cfg, catalog)
        self.assertNotEqual(forward_full.state_id, reversed_full.state_id)
        self.assertEqual(limited, encode_limited(obs, cfg, catalog))

    def test_read_only_boundary_diagnostic(self):
        cfg, match, engine, catalog, packet = fixture()
        before = (match.tick, match.t, tuple(match.ref.hp), tuple(match.ref.threshold),
                  engine.sequence, len(match.decisions))
        record = encode_boundary(match, engine, CameraRig(), packet, (None,), cfg, catalog)
        self.assertEqual(record["full"]["status"], "valid")
        self.assertEqual(record["limited"]["status"], "valid")
        self.assertIn("B.region_id", record["full"])
        self.assertNotIn("B.region_id", record["limited"])
        self.assertEqual(record["limited"]["A.visual_track_region"],
                         dict(encode_limited(limited_from_public(packet, (None,), cfg), cfg, catalog).fields)
                         ["A.visual_track_region"])
        self.assertEqual(record["frame_count"], 1)
        self.assertEqual(record["render_sync_quality"], "blocked")
        self.assertNotIn("threshold", str(record))
        self.assertEqual(before, (match.tick, match.t, tuple(match.ref.hp), tuple(match.ref.threshold),
                                  engine.sequence, len(match.decisions)))
        with self.assertRaises(ValueError):
            encode_boundary(match, engine, CameraRig(), packet, (None,), cfg, catalog,
                            render_sync_quality="verified")

    def test_cpu_match_decision_trajectory_without_rgb(self):
        """Actual Match advance/Skill commit; metadata is a CPU fixture, not Isaac RGB."""
        cfg, match, engine, catalog, _ = fixture()
        rig = CameraRig()
        previous_action = None
        records = []

        def frame():
            yaw = match.a.yaw
            return {"simulation_seconds": match.t,
                    "camera_world_yaw_degrees": math.degrees(yaw),
                    "camera_eye": [match.a.xy[0] + .36*math.cos(yaw),
                                   match.a.xy[1] + .36*math.sin(yaw), .6],
                    "file": f"cpu_fixture_no_rgb_{match.t:.2f}"}

        window = [frame()]
        for index in range(4):
            packet = public_packet(index, match.states[-1], window, cfg["start_b"])
            packet["skills"] = engine.public(match.t)
            before = (match.tick, match.t, tuple(match.ref.hp), match.ref.generation, engine.sequence)
            record = encode_boundary(match, engine, rig, packet, tuple(None for _ in window),
                                     cfg, catalog, previous_action, "not_applicable")
            records.append(record)
            self.assertEqual(before, (match.tick, match.t, tuple(match.ref.hp), match.ref.generation,
                                      engine.sequence))
            self.assertEqual(record["full"]["status"], "valid")
            self.assertEqual(record["limited"]["status"], "valid")
            self.assertEqual(record["frame_count"], 1 if index == 0 else 4)
            if index == 3:
                break
            request = {"observation_id": index, "skill": "OBSERVE",
                       "params": {"direction": "north"}}
            action, route, _, _ = engine.commit(request, packet)
            match.compile(action)
            match.apply(action, route)
            previous_action = action_template("OBSERVE", request["params"], cfg).template_id
            window = []
            for _ in range(4):
                match.advance(.5)
                window.append(frame())
        self.assertEqual([r["simulation_seconds"] for r in records], [0., 2., 4., 6.])
        self.assertEqual([r["full"]["decision_step"] for r in records], [0, 1, 2, 3])
        self.assertEqual([r["limited"]["decision_step"] for r in records], [0, 1, 2, 3])
        self.assertNotEqual(records[0]["full"]["state_id"], records[3]["full"]["state_id"])
        self.assertEqual(match.ref.active_target, 1)  # Real Referee hit refresh rotated P1 to P2.
        self.assertTrue(any(event["type"] == "target_invalidated" and event.get("reason") == "hit"
                            for event in match.ref.events))

    def test_normal_spawn_and_moving_match_boundary(self):
        cfg = load_config("configs/trial_rotating.json")
        match, engine, rig = VisionMatch(cfg), SemanticSkills(cfg), CameraRig()
        catalog = RegionCatalog.from_config(cfg)

        def frame():
            return {"simulation_seconds": match.t, "camera_world_yaw_degrees": math.degrees(match.a.yaw),
                    "camera_eye": [float(match.a.xy[0]), float(match.a.xy[1]), .6],
                    "file": f"cpu_fixture_no_rgb_{match.t:.2f}"}

        initial = public_packet(0, match.states[-1], [frame()], cfg["start_b"])
        initial["skills"] = engine.public(0)
        first = encode_boundary(match, engine, rig, initial, (None,), cfg, catalog,
                                render_sync_quality="not_applicable")
        self.assertEqual(first["full"]["status"], "valid")
        self.assertEqual(first["limited"]["status"], "valid")
        request = {"observation_id": 0, "skill": "MOVE_TO", "params": {"target": "P2", "risk": "fast"}}
        action, route, _, _ = engine.commit(request, initial)
        match.apply(action, route)
        window = []
        for _ in range(4):
            match.advance(.5)
            window.append(frame())
        self.assertGreater(abs(match.a.v), 0)
        packet = public_packet(1, match.states[-1], window, cfg["start_b"])
        packet["skills"] = engine.public(match.t)
        previous_action = action_template(request["skill"], request["params"], cfg).template_id
        second = encode_boundary(match, engine, rig, packet, (None,)*4, cfg, catalog,
                                 previous_action, "not_applicable")
        self.assertEqual(second["full"]["status"], "valid")
        self.assertEqual(second["limited"]["status"], "valid")
        self.assertEqual(second["frame_count"], 4)
        limited = encode_limited(limited_from_public(packet, (None,)*4, cfg, previous_action), cfg, catalog)
        self.assertEqual(dict(limited.fields)["A.motion_bin"], "moving")
        self.assertNotIn("own_previous_action_outcome", dict(limited.fields))

    def test_invalid_limited_inputs(self):
        cfg, _, _, catalog, packet = fixture()
        obs = limited_from_public(packet, (None,), cfg)
        with self.assertRaises(TypeError):
            encode_limited(object(), cfg, catalog)
        for bad in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                encode_limited(replace(obs, own_speed=bad), cfg, catalog)
        moving = encode_limited(replace(obs, own_speed=.2), cfg, catalog)
        self.assertEqual(moving.status, "valid")
        self.assertEqual(dict(moving.fields)["A.motion_bin"], "moving")
        with self.assertRaises(ValueError):
            encode_limited(replace(obs, own_hp=4), cfg, catalog)
        with self.assertRaises(ValueError):
            encode_limited(replace(obs, previous_action_id="B_HP=0"), cfg, catalog)
        with self.assertRaises(ValueError):
            encode_limited(replace(obs, finished="fictional_victory"), cfg, catalog)
        with self.assertRaises(ValueError):
            limited_from_public(packet, ({"time": 0., "feet": [384, 300], "true_xy": [1, 2]},), cfg)
        with self.assertRaises(KeyError):
            limited_from_public({"self_state": {}}, (None,), cfg)

    def test_time_boundaries(self):
        self.assertEqual(_boundary(0, 40, False), 0)
        self.assertEqual(_boundary(38, 40, False), 19)
        with self.assertRaises(ValueError):
            _boundary(40, 40, False)
        self.assertEqual(_boundary(40, 40, True), 19)
        self.assertEqual(_cooldown(0), "ready")
        self.assertEqual(_cooldown(1e-9), "ready")
        self.assertEqual(_cooldown(1.1e-9), "recovers_within_window")
        self.assertEqual(_cooldown(2), "recovers_within_window")
        self.assertEqual(_cooldown(2.0001), "unavailable_next_window")
        self.assertEqual(_deadline(1.9999), "lt_2s")
        self.assertEqual(_deadline(2), "2_to_4s")
        self.assertEqual(_deadline(4), "2_to_4s")
        self.assertEqual(_deadline(4.0001), "gt_4s")
        self.assertEqual(_dwell(True, .9999), "inside_under_1s")
        self.assertEqual(_dwell(True, 1), "scoring_eligible")
        self.assertEqual(_relation(.55, .55, 0), "capture_edge")
        self.assertEqual(_relation(.55-.0401, .55, 0), "inside_capture")
        self.assertEqual(_relation(.55-.04, .55, 0), "capture_edge")
        self.assertEqual(_aim_bin(0, True, 1), "newly_valid")
        self.assertEqual(_aim_bin(.02, True, 1), "under_half_tau")
        self.assertEqual(_aim_bin(.5, True, 1), "half_to_tau")
        self.assertEqual(_aim_bin(1, True, 1), "tau_or_more")
        self.assertEqual(_aim_bin(.25, True, .5), "half_to_tau")
        self.assertEqual(_aim_bin(.5, True, .5), "tau_or_more")
        self.assertEqual(_aim_bin(5, False, 1), "invalid")
        self.assertEqual(_visual_bin("rgb_foot_pixel", 2), "fresh_aim_usable")
        self.assertEqual(_visual_bin("rgb_foot_pixel", 2.0001), "stale_retained")
        self.assertEqual(_visual_bin("rgb_foot_pixel", 4), "stale_retained")
        self.assertEqual(_visual_bin("rgb_foot_pixel", 4.0001), "expired")
        self.assertEqual(_memory_age(4), "2_to_4")
        self.assertEqual(_memory_age(4.0001), "gt_4_or_none")
        self.assertEqual(_visual_bin(None, None), "none_or_initial_prior")
        memory = VisualMemory([0, 0])
        memory.track = {"xy": [0, 0], "seen_at": 0., "source": "rgb_foot_pixel", "base_uncertainty": .6}
        self.assertTrue(memory.belief(2)["aim_usable"])
        self.assertFalse(memory.belief(2.0001)["aim_usable"])
        self.assertIsNotNone(memory.belief(4))
        self.assertIsNone(memory.belief(4.0001))
        cfg, match, engine, catalog, _ = fixture()
        snap = capture_reference(match, engine, CameraRig())
        at_four = replace(snap, t=6., tick=300, human_memory=snap.a.xy, human_seen_at=2.)
        after_four = replace(snap, t=8., tick=400, human_memory=snap.a.xy, human_seen_at=3.98)
        self.assertEqual(dict(encode_full(at_four, cfg, catalog).fields)["B.memory_presence"], "active")
        self.assertEqual(dict(encode_full(at_four, cfg, catalog).fields)["B.memory_age_bin"], "2_to_4s")
        self.assertEqual(dict(encode_full(after_four, cfg, catalog).fields)["B.memory_presence"], "expired")

    def test_cadence_map_and_terminal(self):
        cfg, match, engine, catalog, packet = fixture()
        obs = limited_from_public(packet, (None,), cfg)
        self.assertEqual(dict(encode_limited(obs, cfg, catalog).fields)["map_phase"], "pre_collapse")
        with self.assertRaises(ValueError):
            encode_limited(replace(obs, t=2, observation_id=1), cfg, catalog)
        frames = tuple(replace(obs.frames[0], time=t) for t in (.5, 1., 1.5, 2.))
        obs2 = replace(obs, t=2., observation_id=1, frames=frames,
                       previous_action_id=action_catalog(cfg)[0].template_id)
        self.assertEqual(encode_limited(obs2, cfg, catalog).status, "valid")
        self.assertEqual(encode_limited(replace(obs2, previous_action_id=None), cfg, catalog).status, "blocked")
        with self.assertRaises(ValueError):
            encode_limited(replace(obs2, frames=frames[:-1] + (replace(frames[-1], time=1.9),)), cfg, catalog)
        fields20 = dict(encode_limited(replace(obs2, t=20, observation_id=10,
                                               frames=tuple(replace(f, time=t) for f, t in zip(frames, (18.5, 19, 19.5, 20)))),
                                       cfg, catalog).fields)
        self.assertEqual(fields20["map_phase"], "post_collapse")
        terminal = encode_limited(replace(obs, t=40., finished="time_limit"), cfg, catalog)
        self.assertEqual(terminal.status, "terminal")
        self.assertIsNotNone(terminal.state_id)
        default_cfg = load_config("configs/trial_rotating.json")
        default_match = VisionMatch(default_cfg)
        initial = encode_full(capture_reference(default_match, SemanticSkills(default_cfg), CameraRig()),
                              default_cfg, RegionCatalog.from_config(default_cfg))
        self.assertEqual(initial.status, "valid")
        self.assertIsNotNone(initial.state_id)
        outside = replace(capture_reference(default_match, SemanticSkills(default_cfg), CameraRig()),
                          a=replace(capture_reference(default_match, SemanticSkills(default_cfg), CameraRig()).a,
                                    xy=(100., 100.)))
        blocked = encode_full(outside, default_cfg, RegionCatalog.from_config(default_cfg))
        self.assertEqual(blocked.status, "blocked")
        self.assertIsNone(blocked.state_id)

    def test_geometry_and_policy_boundaries(self):
        cfg, match, engine, catalog, packet = fixture()
        snap = capture_reference(match, engine, CameraRig())
        self.assertEqual(_attack_geometry(cfg, snap.a, snap.b, 0, 0, 0, False), "disabled")
        self.assertEqual(_attack_geometry(cfg, snap.a, snap.b, 0, 0, .01, True), "cooldown")
        self.assertEqual(_attack_geometry(cfg, snap.a, replace(snap.b, xy=(snap.a.xy[0]+8.0001, snap.a.xy[1])),
                                          0, 0, 0, True), "invalid_range")
        self.assertNotEqual(_attack_geometry(cfg, snap.a, replace(snap.b, xy=(snap.a.xy[0]+8, snap.a.xy[1])),
                                             0, 0, 0, True), "invalid_range")
        at_aim_edge = (snap.a.xy[0] + 4*math.cos(math.pi/6), snap.a.xy[1] + 4*math.sin(math.pi/6))
        beyond_aim_edge = (snap.a.xy[0] + 4*math.cos(math.pi/6 + .001),
                           snap.a.xy[1] + 4*math.sin(math.pi/6 + .001))
        self.assertNotEqual(_attack_geometry(cfg, snap.a, replace(snap.b, xy=at_aim_edge),
                                             0, 0, 0, True), "invalid_angle")
        self.assertEqual(_attack_geometry(cfg, snap.a, replace(snap.b, xy=beyond_aim_edge),
                                          0, 0, 0, True), "invalid_angle")
        no_cover = []
        cfg["policy"]["human_sensor_range"] = 8
        own = [0., 0.]
        at_edge = [8*math.cos(math.radians(60)), 8*math.sin(math.radians(60))]
        # Sensor FOV uses its own configured half-angle, independent of attack angle.
        cfg["policy"]["human_sensor_half_angle_degrees"] = 60
        self.assertIsNotNone(visible_observation(cfg, own, 0, at_edge, no_cover, 0))
        outside = [8*math.cos(math.radians(60.01)), 8*math.sin(math.radians(60.01))]
        self.assertIsNone(visible_observation(cfg, own, 0, outside, no_cover, 0))

    def test_action_catalog(self):
        cfg, match, _, _, packet = fixture()
        actions = action_catalog(cfg)
        self.assertEqual([p["name"] for p in cfg["arena"]["task_points"]], ["P1", "P2"])
        covers = [c["name"] for c in cfg["arena"]["covers"]]
        self.assertEqual(covers, ["High1", "High2", "High3", "High4",
                                  "Low1", "Low2", "Low3", "Low4"])
        self.assertEqual(len(actions), 44)
        self.assertEqual(Counter(a.skill for a in actions),
                         {"MOVE_TO": 6, "CAPTURE_POINT": 6, "TAKE_COVER": 18,
                          "AIM_AT_TRACK": 2, "OBSERVE": 12})
        self.assertEqual(len({a.template_id for a in actions}), len(actions))
        expected = set()
        for risk in ("safe", "fast"):
            for target in ("active", "P1", "P2"):
                for skill in ("MOVE_TO", "CAPTURE_POINT"):
                    expected.add(action_template(skill, {"target": target, "risk": risk}, cfg))
            for cover in (None, *covers):
                params = {"risk": risk} if cover is None else {"cover": cover, "risk": risk}
                expected.add(action_template("TAKE_COVER", params, cfg))
            expected.add(action_template("AIM_AT_TRACK", {"risk": risk}, cfg))
            for direction in ("scan", "threat", "north", "south", "east", "west"):
                expected.add(action_template("OBSERVE", {"direction": direction, "risk": risk}, cfg))
        self.assertEqual(set(actions), expected)
        for skill in ("MOVE_TO", "CAPTURE_POINT", "TAKE_COVER", "AIM_AT_TRACK", "OBSERVE"):
            self.assertEqual(action_template(skill, {}, cfg),
                             action_template(skill, {"risk": "safe"}, cfg))
            self.assertNotEqual(action_template(skill, {}, cfg),
                                action_template(skill, {"risk": "fast"}, cfg))
        for skill in ("MOVE_TO", "CAPTURE_POINT"):
            self.assertEqual(action_template(skill, {}, cfg),
                             action_template(skill, {"target": "active"}, cfg))
        self.assertEqual(action_template("OBSERVE", {}, cfg),
                         action_template("OBSERVE", {"direction": "scan"}, cfg))
        for direction in ("scan", "threat", "north", "south", "east", "west"):
            self.assertNotEqual(action_template("OBSERVE", {"direction": direction}, cfg),
                                action_template("OBSERVE", {"direction": direction, "risk": "fast"}, cfg))
        self.assertEqual(action_template("CAPTURE_POINT", {"target": "P1", "risk": "fast"}, cfg).template_id,
                         'CAPTURE_POINT:[["risk","fast"],["target","P1"]]')
        self.assertEqual(action_template("AIM_AT_TRACK", {}, cfg).template_id,
                         'AIM_AT_TRACK:[["risk","safe"]]')
        self.assertEqual(action_template("OBSERVE", {"risk": "fast", "direction": "west"}, cfg),
                         action_template("OBSERVE", {"direction": "west", "risk": "fast"}, cfg))
        self.assertEqual(action_template("TAKE_COVER", {}, cfg).params, (("risk", "safe"),))
        request = actions[0].request(0)
        match_tick = match.tick
        # Actual API validates the generated high-level request without commit.
        SemanticSkills(cfg).validate(request, packet)
        self.assertEqual(match.tick, match_tick)
        for skill, params in (("MOVE", {"goal": [0, 0]}), ("MOVE_TO", {"waypoint": [0, 0]}),
                              ("OBSERVE", {"direction": "up"}),
                              ("TAKE_COVER", {"cover": "fictional"}), ("AIM_AT_TRACK", {"sighting": None}),
                              ("MOVE_TO", {"risk": "unsafe"})):
            with self.subTest(skill=skill, params=params), self.assertRaises(ValueError):
                action_template(skill, params, cfg)


if __name__ == "__main__":
    unittest.main()
