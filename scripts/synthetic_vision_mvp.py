"""Simulator-derived foot sighting for the formal vision scene.

The decision core is Isaac-free. ``observe_current_scene`` reads live USD and
PhysX state only in explicit synthetic vision modes.
"""

import copy
import math
import random

from isaacsim_minimal_geometry_test import project_camera_point
from formal_geometry_diagnostic import _foot_candidates, _query_ray
from tactical_logic import block
from trial_rules import covers_at


DEFAULT_PIXEL_NOISE_STD = 1.0  # Engineering initial value, not camera calibration.
DEFAULT_MISS_PROBABILITY = 0.0  # The one-shot smoke check should not randomly miss.
DEFAULT_SEED = 20261028
ENDPOINT_TOLERANCE_M = 0.05


def visibility_from_ray(target_distance, cover, ray, target_link):
    """Classify the camera-to-foot segment, allowing a clear segment with no hit.

    The formal G1 visual meshes have no CollisionAPI. A no-hit PhysX query can
    therefore support an unobstructed segment when the authored cover test is
    also clear; a missing or failed query cannot.
    """
    if cover is not None:
        return {"status": "occluded", "reason": "authored_cover_intersection",
                "blocker": cover[1]}
    if not isinstance(target_distance, (int, float)) or not math.isfinite(target_distance) or target_distance <= 0:
        return {"status": "unknown", "reason": "invalid_target_distance"}
    if not isinstance(ray, dict) or ray.get("error"):
        return {"status": "unknown", "reason": "ray_query_unavailable_or_failed"}
    if ray.get("hit") is False:
        return {"status": "visible", "reason": "clear_physx_segment_and_authored_covers"}
    if ray.get("hit") is not True:
        return {"status": "unknown", "reason": "ray_hit_state_unavailable"}
    distance, path = ray.get("distance"), ray.get("collision")
    if not isinstance(distance, (int, float)) or not math.isfinite(distance) or not isinstance(path, str) or not path.startswith("/"):
        return {"status": "unknown", "reason": "ray_hit_details_unavailable"}
    if path == target_link or path.startswith(target_link + "/"):
        if abs(distance - target_distance) <= ENDPOINT_TOLERANCE_M:
            return {"status": "visible", "reason": "target_endpoint_hit"}
        return {"status": "unknown", "reason": "target_hit_away_from_foot_endpoint"}
    if path.startswith("/World/Actors/B/"):
        return {"status": "unknown", "reason": "target_actor_self_hit"}
    if distance < target_distance - ENDPOINT_TOLERANCE_M:
        return {"status": "occluded", "reason": "nearer_physx_collision", "blocker": path}
    return {"status": "unknown", "reason": "non_target_hit_near_foot_endpoint"}


def build_noisy_sighting(candidates, packet, frame, width, height, seed=DEFAULT_SEED,
                         pixel_noise_std=DEFAULT_PIXEL_NOISE_STD,
                         miss_probability=DEFAULT_MISS_PROBABILITY):
    """Build exactly the Skill's {time, feet} shape, or no sighting."""
    if type(seed) is not int or type(pixel_noise_std) not in (int, float) or not math.isfinite(pixel_noise_std) or pixel_noise_std < 0:
        raise ValueError("Invalid synthetic sighting seed or pixel noise")
    if type(miss_probability) not in (int, float) or not math.isfinite(miss_probability) or not 0 <= miss_probability <= 1:
        raise ValueError("Invalid synthetic sighting miss probability")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("Invalid captured image dimensions")
    frames = packet.get("frames")
    if not isinstance(frames, list) or sum(item == frame for item in frames) != 1:
        raise ValueError("Sighting must reference exactly one packet frame")
    time = frame.get("simulation_seconds")
    if type(time) not in (int, float) or not math.isfinite(time) or time != packet.get("simulation_seconds"):
        raise ValueError("Frame and packet simulation time differ")
    record = {"schema": "synthetic_vision_mvp.v1", "source": "simulator_geometry_truth",
              "frame_file": frame.get("file"), "simulation_time_seconds": time,
              "same_frame_verified": False, "sync_integration_status": "blocked",
              "seed": seed, "pixel_noise_std": pixel_noise_std,
              "miss_probability": miss_probability, "image_size": [width, height],
              "candidates": candidates, "selected_link": None,
              "raw_pixel": None, "noisy_pixel": None,
              "sighting": None, "status": "not_emitted", "reason": "no_visible_foot"}
    visible = [item for item in candidates if item.get("projection", {}).get("in_frame") is True
               and item.get("visibility", {}).get("status") == "visible"]
    if not visible:
        return record
    chosen = visible[0]
    record["selected_link"] = chosen.get("link_prim")
    u, v = chosen["projection"]["pixel"]
    if not all(type(value) in (int, float) and math.isfinite(value) for value in (u, v)) or not (0 <= u < width and 0 <= v < height):
        record["reason"] = "invalid_projected_pixel"
        return record
    record["raw_pixel"] = [float(u), float(v)]
    rng = random.Random(seed)
    if rng.random() < miss_probability:
        record["reason"] = "synthetic_miss"
        return record
    noisy = [float(u + rng.gauss(0, pixel_noise_std)), float(v + rng.gauss(0, pixel_noise_std))]
    record["noisy_pixel"] = noisy
    if not all(math.isfinite(value) for value in noisy) or not (0 <= noisy[0] < width and 0 <= noisy[1] < height):
        record["reason"] = "noisy_pixel_out_of_frame"
        return record
    record.update(sighting={"time": float(time), "feet": noisy},
                  status="emitted", reason=None)
    return record


def handoff_to_skill(record, engine, packet):
    """Use the existing OBSERVE request and validator/commit boundary."""
    result = dict(record)
    if record.get("sighting") is None:
        result["skill_handoff"] = {"status": "not_attempted_no_sighting",
                                   "reason": record.get("reason")}
        return result
    request = {"observation_id": packet["observation_id"], "skill": "OBSERVE",
               "params": {"sighting": record["sighting"]},
               "reason": "Simulator-derived synthetic foot pixel MVP"}
    result["skill_request"] = request
    try:
        prepared = engine.validate(request, packet)
    except Exception as exc:
        result["skill_handoff"] = {"status": "rejected_by_validator",
                                   "reason": f"{type(exc).__name__}: {exc}"}
        return result
    result["skill_handoff"] = {"status": "validated",
                               "prepared_world_xy_m": list(prepared.track["xy"])}
    try:
        action, route, camera_command, skill_status = engine.commit(request, packet)
    except Exception as exc:
        result["skill_handoff"] = {"status": "commit_failed",
                                   "reason": f"{type(exc).__name__}: {exc}"}
        return result
    result["skill_handoff"] = {"status": "accepted_and_committed",
                               "memory_world_xy_m": list(engine.memory.track["xy"]),
                               "memory_source_tag": engine.memory.track["source"],
                               "action": action, "route_count": len(route),
                               "camera_command": camera_command,
                               "skill_status": skill_status}
    return result


VALIDATION_CASES = ("clear_initial", "high2_occluded", "moved_clear")


def validation_pose(index, state, cfg):
    """Author three finite diagnostic poses without changing match configuration."""
    if index not in range(len(VALIDATION_CASES)):
        raise ValueError("Unknown synthetic validation case")
    posed = copy.deepcopy(state)
    posed["speed"] = [0.0, 0.0]
    posed["xy"][1] = list(cfg["start_b"])
    if index == 1:
        posed["xy"][0] = [0.3, 1.8]
        posed["yaw"][0] = math.atan2(posed["xy"][1][1] - 1.8,
                                       posed["xy"][1][0] - 0.3)
    else:
        posed["xy"][0] = [4.8, -1.0]
        posed["yaw"][0] = math.pi / 2
        if index == 2:
            posed["xy"][1] = [cfg["start_b"][0] - 0.4,
                               cfg["start_b"][1] - 0.4]
    return {"name": VALIDATION_CASES[index], "state": posed,
            "authored_actor_xy": copy.deepcopy(posed["xy"]),
            "reason": ("Previously checked clear A view" if index == 0 else
                       "A view through existing High2 cover" if index == 1 else
                       "B moved 0.4 m west and south; A returns to clear view")}


def run_validation_cases(engine, sample_case, on_case=None):
    """Call a fresh scene sampler three times, then assess Skill memory changes."""
    records = []
    for index, name in enumerate(VALIDATION_CASES):
        before = copy.deepcopy(engine.memory.track)
        entry = {"case": name, "case_index": index, "start_status": "started",
                 "status": "running",
                 "memory_before": before, "same_frame_verified": False,
                 "sync_integration_status": "blocked"}
        if on_case is not None:
            on_case(records + [copy.deepcopy(entry)])
        try:
            sample = sample_case(index, name)
            frame, packet, synthetic = sample["frame"], sample["packet"], sample["synthetic"]
            entry.update({key: value for key, value in sample.items()
                          if key not in ("frame", "packet", "synthetic")})
            entry.update(frame=frame, observation_id=packet["observation_id"],
                         simulation_seconds=packet["simulation_seconds"])
            if (packet["observation_id"] != index or packet["frames"] != [frame] or
                    frame["simulation_seconds"] != packet["simulation_seconds"] or
                    (records and frame["simulation_seconds"] <= records[-1].get("simulation_seconds", -math.inf))):
                raise ValueError("Non-unique or mismatched current observation frame/time")
            entry["synthetic"] = handoff_to_skill(synthetic, engine, packet)
            after = copy.deepcopy(engine.memory.track)
            entry["memory_after"] = after
            handoff = entry["synthetic"]["skill_handoff"]["status"]
            candidates = synthetic.get("candidates", [])
            if index == 1:
                in_frame = [c for c in candidates if c.get("projection", {}).get("in_frame") is True]
                occluded = bool(in_frame) and all(
                    c.get("visibility", {}).get("status") == "occluded" and
                    "raycast" in c for c in in_frame)
                ok = (occluded and synthetic.get("sighting") is None and
                      handoff == "not_attempted_no_sighting" and after == before)
                entry["checks"] = {"in_frame_occlusion": occluded,
                                   "no_sighting": synthetic.get("sighting") is None,
                                   "no_commit_or_memory_refresh": after == before}
                entry["no_sighting_reason"] = synthetic.get("reason")
            else:
                sighting = synthetic.get("sighting")
                visible = any(c.get("projection", {}).get("in_frame") is True and
                              c.get("visibility", {}).get("status") == "visible" for c in candidates)
                ok = (visible and sighting is not None and
                      sighting["time"] == frame["simulation_seconds"] and
                      handoff == "accepted_and_committed" and
                      after.get("seen_at") == frame["simulation_seconds"] and
                      after.get("feet") == sighting["feet"])
                entry["checks"] = {"in_frame_visible": visible,
                                   "fresh_sighting": sighting is not None and
                                   sighting["time"] == frame["simulation_seconds"],
                                   "memory_updated": after != before and
                                   after.get("seen_at") == frame["simulation_seconds"]}
                if index == 2:
                    first = records[0].get("synthetic", {})
                    changed = (entry.get("authored_actor_xy", [None, None])[1] !=
                               records[1].get("authored_actor_xy", [None, None])[1] and
                               synthetic.get("raw_pixel") != first.get("raw_pixel"))
                    entry["checks"]["target_and_raw_projection_changed"] = changed
                    ok = ok and changed
            entry["status"] = "passed" if ok else "partial"
            if not ok:
                entry["reason"] = "case_acceptance_checks_failed"
        except Exception as exc:
            entry["status"] = "partial"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["memory_after"] = copy.deepcopy(engine.memory.track)
        entry["end_status"] = entry["status"]
        records.append(entry)
        if on_case is not None:
            on_case(records)
    return {"status": "passed" if all(r["status"] == "passed" for r in records) else "partial",
            "mode": "synthetic_vision_mvp_validation", "source": "simulator_geometry_truth",
            "sync_integration_status": "blocked", "same_frame_verified": False,
            "cases": records}


def observe_current_scene(stage, camera, packet, frame, cfg, image_size,
                          seed=DEFAULT_SEED, pixel_noise_std=DEFAULT_PIXEL_NOISE_STD,
                          miss_probability=DEFAULT_MISS_PROBABILITY):
    """Read live camera, G1 mesh vertices and cover colliders in the formal stage."""
    from pxr import Gf, UsdGeom
    import omni.physx

    width, height = image_size
    cache = UsdGeom.XformCache()
    world = cache.GetLocalToWorldTransform(camera.GetPrim())
    inverse = world.GetInverse()
    eye_point = world.Transform(Gf.Vec3d(0, 0, 0))
    eye = [float(eye_point[i]) for i in range(3)]
    intrinsics = (float(camera.GetFocalLengthAttr().Get()),
                  float(camera.GetHorizontalApertureAttr().Get()),
                  float(camera.GetVerticalApertureAttr().Get()))
    near, far = (float(value) for value in camera.GetClippingRangeAttr().Get())
    try:
        query = omni.physx.get_physx_scene_query_interface()
    except Exception:
        query = None  # _query_ray records an unavailable query for every foot.
    covers = covers_at(cfg, packet["simulation_seconds"])
    candidates = []
    for foot in _foot_candidates(stage, eye, cache, Gf, UsdGeom):
        item = {key: foot.get(key) for key in ("link_prim", "visual_prim", "world_m",
                                               "local_anchor_definition", "error") if key in foot}
        if "world_m" not in foot:
            item.update(projection={"in_frame": False, "reason": "foot_mesh_unavailable"},
                        visibility={"status": "unknown", "reason": "foot_mesh_unavailable"})
            candidates.append(item)
            continue
        local_point = inverse.Transform(Gf.Vec3d(*foot["world_m"]))
        local = [float(local_point[i]) for i in range(3)]
        item["camera_local_m"] = local
        try:
            projection = project_camera_point(local, width, height, *intrinsics)
            if not near <= projection["depth_m"] <= far:
                projection.update(in_frame=False, reason="outside_camera_clipping")
        except (TypeError, ValueError) as exc:
            projection = {"in_frame": False, "pixel": None,
                          "reason": f"{type(exc).__name__}: {exc}"}
        item["projection"] = projection
        if not projection["in_frame"]:
            item["visibility"] = {"status": "unknown", "reason": "foot_out_of_camera_frustum"}
        else:
            cover = block(eye, foot["world_m"], covers)
            ray = _query_ray(eye, foot["world_m"], query)
            item["raycast"] = ray
            item["visibility"] = visibility_from_ray(ray["target_distance_m"], cover, ray,
                                                     foot["link_prim"])
            item["analytic_cover"] = None if cover is None else cover[1]
        candidates.append(item)
    record = build_noisy_sighting(candidates, packet, frame, width, height, seed,
                                  pixel_noise_std, miss_probability)
    record["camera"] = {"prim_path": str(camera.GetPath()), "eye_world_m": eye,
                        "intrinsics_mm": list(intrinsics), "clipping_m": [near, far],
                        "convention": "USD +X right, +Y up, -Z forward; image top-left"}
    record["visibility_method"] = "authored current covers plus PhysX closest-hit segment"
    return record
