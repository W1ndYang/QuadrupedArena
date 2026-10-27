"""Bounded formal-scene geometry diagnostic; never submits a Skill request.

The pure classifiers are importable without Isaac Sim. ``run`` is called only
after vision_trial has built its normal courtyard, actors, camera and RGB sensor.
"""

import copy
import json
import math
from pathlib import Path
import zipfile

from isaacsim_minimal_geometry_test import project_camera_point


WIDTH, HEIGHT = 768, 432
FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")


def diagnostic_mode_error(args):
    if not args.geometry_diagnostic:
        return None
    if not args.vision_trial:
        return "--geometry-diagnostic requires --vision-trial"
    if any((args.vision_smoke, args.semantic_skills, args.deepseek,
            args.qwen_vision, args.assessment, args.g1b1_visible_spawn)):
        return "--geometry-diagnostic cannot be combined with smoke, Skill, planner, assessment or G1-B.1 modes"
    return None


def project_local_candidate(local, intrinsics, clipping):
    """Reuse G1-A pinhole math, adding clipping and explicit status."""
    try:
        depth = -float(local[2])
        if not math.isfinite(depth):
            raise ValueError("Non-finite camera depth")
        if depth <= 0:
            return {"status": "behind_camera", "pixel": None, "depth_m": depth}
        result = project_camera_point(local, WIDTH, HEIGHT, *intrinsics)
        if not all(math.isfinite(v) for v in (*result["pixel"], result["depth_m"])):
            raise ValueError("Non-finite projected pixel or depth")
        near, far = (float(v) for v in clipping)
        if not all(math.isfinite(v) for v in (near, far)) or not 0 < near < far:
            raise ValueError("Invalid clipping range")
        if not near <= depth <= far:
            return {"status": "out_of_clipping", **result}
        return {"status": "in_frame" if result["in_frame"] else "out_of_frame", **result}
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        return {"status": "invalid", "pixel": None, "error": str(exc)}


def classify_visibility(target_distance, blocker, hit, target_foot_path=None,
                        target_collision_covered=False):
    """A known cover can block; a missing hit never proves the foot visible."""
    if blocker is not None:
        return {"status": "occluded", "reason": "known_cover_intersection",
                "blocker": blocker[1]}
    if not isinstance(target_distance, (int, float)) or not math.isfinite(target_distance) or target_distance <= 0:
        return {"status": "unknown", "reason": "invalid_target_distance"}
    if not isinstance(hit, dict) or hit.get("error"):
        return {"status": "unknown", "reason": "query_failed_or_unavailable"}
    if not hit.get("hit"):
        return {"status": "unknown", "reason": "no_raycast_hit"}
    path = str(hit.get("collision") or "")
    distance = hit.get("distance")
    if not isinstance(distance, (int, float)) or not math.isfinite(distance):
        return {"status": "unknown", "reason": "hit_distance_unavailable"}
    if not path.startswith("/"):
        return {"status": "unknown", "reason": "hit_prim_unavailable"}
    if distance < target_distance - 0.02 and not (target_foot_path and path.startswith(target_foot_path + "/")):
        return {"status": "occluded", "reason": "nearer_non_target_hit", "blocker": path}
    if (target_collision_covered and target_foot_path
            and path.startswith(target_foot_path + "/")
            and abs(distance - target_distance) <= 0.02):
        return {"status": "visible_candidate", "reason": "target_foot_hit"}
    return {"status": "unknown", "reason": "target_geometry_uncovered_or_unverified"}


def record_is_structurally_same_frame(record):
    return (record.get("frame_index") == record.get("frame", {}).get("frame_index")
            and record.get("simulation_seconds") == record.get("frame", {}).get("simulation_seconds")
            and record.get("frame", {}).get("rgb_file")
            and record.get("camera", {}).get("world_transform_rows")
            and all(item.get("frame_index") == record.get("frame_index")
                    for item in record.get("feet", [])))


def _rows(matrix):
    return [[float(matrix[i][j]) for j in range(4)] for i in range(4)]


def _vec(value):
    return [float(value[i]) for i in range(3)]


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _scenarios(state, cfg):
    clear = copy.deepcopy(state)
    clear["xy"][0] = [4.8, -1.0]
    clear["yaw"][0] = math.pi / 2
    clear["speed"] = [0.0, 0.0]
    blocked = copy.deepcopy(state)
    blocked["xy"][0] = [0.3, 1.8]
    blocked["yaw"][0] = math.atan2(cfg["start_b"][1] - 1.8, cfg["start_b"][0] - 0.3)
    blocked["speed"] = [0.0, 0.0]
    return (("expected_clear", clear), ("expected_high2_occluded", blocked))


def _foot_candidates(stage, eye, cache, Gf, UsdGeom):
    feet = []
    for link in FOOT_LINKS:
        link_path = "/World/Actors/B/" + link
        link_prim = stage.GetPrimAtPath(link_path)
        item = {"link_prim": link_path, "source": "g1_visual_mesh_vertex",
                "verification": "mesh_vertex_candidate_unverified",
                "collision_geometry_available": False}
        if not link_prim.IsValid():
            item["error"] = "foot_link_prim_missing"
            feet.append(item)
            continue
        item["ankle_link_origin_world_m"] = _vec(cache.GetLocalToWorldTransform(link_prim).Transform(Gf.Vec3d(0, 0, 0)))
        vertices = []
        for child in link_prim.GetChildren():
            mesh = UsdGeom.Mesh(child)
            if not mesh:
                continue
            points = mesh.GetPointsAttr().Get()
            if not points:
                continue
            world = cache.GetLocalToWorldTransform(child)
            for index, point in enumerate(points):
                p = world.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                xyz = _vec(p)
                if all(math.isfinite(v) for v in xyz):
                    vertices.append((xyz, str(child.GetPath()), index, _vec(point)))
        if not vertices:
            item["error"] = "foot_visual_vertices_missing"
            feet.append(item)
            continue
        minimum_z = min(v[0][2] for v in vertices)
        sole_vertices = [v for v in vertices if v[0][2] <= minimum_z + 0.005]
        chosen = min(sole_vertices, key=lambda v: (math.dist(v[0], eye), v[1], v[2]))
        item.update(world_m=chosen[0], visual_prim=chosen[1], vertex_index=chosen[2],
                    local_vertex_m=chosen[3], local_anchor_definition="lowest 5 mm world-Z mesh vertices; nearest camera vertex",
                    mesh_vertex_count=len(vertices), minimum_mesh_z_m=minimum_z)
        feet.append(item)
    return feet


def _query_ray(eye, point, query):
    vector = [point[i] - eye[i] for i in range(3)]
    distance = math.sqrt(sum(v * v for v in vector))
    result = {"method": "omni.physx scene_query.raycast_closest",
              "origin_world_m": eye, "target_world_m": point,
              "target_distance_m": distance, "query_distance_m": distance,
              "collision_coverage": "arena floor/covers authored; G1 visual meshes have no CollisionAPI in Actor"}
    if not math.isfinite(distance) or distance <= 0:
        result["error"] = "invalid_ray_distance"
        return result
    result["direction_world"] = [v / distance for v in vector]
    if query is None:
        result["error"] = "scene_query_unavailable"
        return result
    try:
        raw = query.raycast_closest(tuple(eye), tuple(result["direction_world"]), distance)
        if not hasattr(raw, "get"):
            result["error"] = "uninterpretable_query_result"
            return result
        result["hit"] = bool(raw.get("hit", False))
        if result["hit"]:
            result["collision"] = str(raw.get("collision", ""))
            position = raw.get("position")
            if position is not None:
                result["position_world_m"] = _vec(position)
                result["distance"] = math.dist(eye, result["position_world_m"])
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def run(output_dir, state, cfg, stage, camera, rgb, app, pose_scene):
    """Capture two deterministic states in the already-built formal scene."""
    import numpy as np
    from pxr import Gf, UsdGeom
    from PIL import Image, ImageDraw
    from tactical_logic import block

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    records = []
    summary = {"mode": "formal_geometry_diagnostic", "status": "partial",
               "runtime_verified": False, "same_frame_verified": False,
               "samples_requested": 2, "samples_saved": 0, "counts": {},
               "blocked_reasons": [], "artifacts": [], "errors": []}
    readme = ("Formal vision_trial scene diagnostic. Pixels are geometry projections, "
              "not verified RGB foot annotations. No Skill request was submitted.\n"
              "Inspect diagnostic_records.json and both raw/overlay images.\n")
    (output / "README.txt").write_text(readme)
    query = None
    query_error = None
    try:
        import omni.physx
        query = omni.physx.get_physx_scene_query_interface()
    except Exception as exc:
        query_error = f"{type(exc).__name__}: {exc}"
    try:
        for index, (name, sample) in enumerate(_scenarios(state, cfg)):
            pose_scene(sample)
            for _ in range(12):
                app.update()
            pixels = np.asarray(rgb.get_data())
            if pixels.ndim != 3 or pixels.shape[:2] != (HEIGHT, WIDTH) or pixels.shape[2] < 3 or pixels[..., :3].std() < 1:
                raise RuntimeError(f"Missing or blank RGB in diagnostic frame {index}: {pixels.shape}")
            raw_name = f"frames/frame_{index:03d}_rgb.png"
            overlay_name = f"frames/frame_{index:03d}_overlay.png"
            image = Image.fromarray(pixels[..., :3].astype(np.uint8).copy())
            image.save(output / raw_name)
            cache = UsdGeom.XformCache()
            camera_world = cache.GetLocalToWorldTransform(camera.GetPrim())
            inverse = camera_world.GetInverse()
            eye = _vec(camera_world.Transform(Gf.Vec3d(0, 0, 0)))
            intrinsics = [float(camera.GetFocalLengthAttr().Get()),
                          float(camera.GetHorizontalApertureAttr().Get()),
                          float(camera.GetVerticalApertureAttr().Get())]
            clipping = [float(v) for v in camera.GetClippingRangeAttr().Get()]
            feet = _foot_candidates(stage, eye, cache, Gf, UsdGeom)
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            for foot in feet:
                foot["frame_index"] = index
                if "world_m" not in foot:
                    foot["projection"] = {"status": "invalid", "error": foot["error"]}
                    foot["visibility"] = {"status": "unknown", "reason": "footpoint_unavailable"}
                    continue
                local = inverse.Transform(Gf.Vec3d(*foot["world_m"]))
                foot["camera_local_m"] = _vec(local)
                foot["projection"] = project_local_candidate(foot["camera_local_m"], intrinsics, clipping)
                ray = _query_ray(eye, foot["world_m"], query)
                if query_error:
                    ray["query_initialization_error"] = query_error
                foot["raycast"] = ray
                cover = block(eye, foot["world_m"], cfg["arena"]["covers"])
                foot["analytic_cover"] = None if cover is None else {"name": cover[1], "hit_world_m": cover[2]}
                foot["visibility"] = classify_visibility(ray["target_distance_m"], cover, ray,
                                                          foot["link_prim"], False)
                projected = foot["projection"]
                if projected["status"] == "in_frame":
                    u, v = projected["pixel"]
                    color = (255, 65, 65) if foot["visibility"]["status"] == "occluded" else (255, 225, 0)
                    draw.line(((u - 9, v), (u + 9, v)), fill=color, width=2)
                    draw.line(((u, v - 9), (u, v + 9)), fill=color, width=2)
                    draw.text((min(u + 11, WIDTH - 100), max(v - 15, 0)), foot["link_prim"].split("/")[-1], fill=color)
            draw.text((8, 8), f"{name}: geometry candidates only", fill=(255, 255, 255))
            draw_image.save(output / overlay_name)
            record = {"frame_index": index, "scenario": name,
                      "simulation_seconds": float(sample["t"]),
                      "frame": {"frame_index": index, "simulation_seconds": float(sample["t"]),
                                "rgb_file": raw_name, "overlay_file": overlay_name,
                                "rgb_shape": list(pixels.shape), "authored_state": "updated_before_capture",
                                "timestamp_source": "VisionMatch state t; two authored snapshots may share t"},
                      "camera": {"prim_path": str(camera.GetPath()), "world_transform_rows": _rows(camera_world),
                                 "eye_world_m": eye, "focal_length_mm": intrinsics[0],
                                 "horizontal_aperture_mm": intrinsics[1], "vertical_aperture_mm": intrinsics[2],
                                 "clipping_m": clipping, "resolution": [WIDTH, HEIGHT],
                                 "convention": "USD camera +X right, +Y up, -Z forward; image top-left origin"},
                      "actor_state": {"a_xy": sample["xy"][0], "a_yaw_radians": sample["yaw"][0],
                                      "b_xy": sample["xy"][1], "b_yaw_radians": sample["yaw"][1]},
                      "feet": feet, "same_frame_verified": False,
                      "same_frame_reason": "RGB and authored USD transforms share capture call; renderer/PhysX snapshot synchronization unverified",
                      "sighting_candidate": None,
                      "sighting_reason": "foot anchor, RGB pixel and query coverage not runtime verified"}
            if not record_is_structurally_same_frame(record):
                raise RuntimeError("Internal frame record mismatch")
            records.append(record)
            summary["samples_saved"] += 1
        if not any(rec["scenario"] == "expected_clear" and any(
                f["visibility"]["status"] == "visible_candidate" for f in rec["feet"]) for rec in records):
            summary["blocked_reasons"].append("clear sample visibility unverified: G1 foot visuals lack query collision coverage")
        if not any(rec["scenario"] == "expected_high2_occluded" and any(
                f["visibility"]["status"] == "occluded" for f in rec["feet"]) for rec in records):
            summary["blocked_reasons"].append("occluded sample not confirmed by cover/query")
        summary["status"] = "blocked" if summary["blocked_reasons"] else "unknown"
    except Exception as exc:
        summary["status"] = "failed"
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        statuses = [foot["visibility"]["status"] for rec in records for foot in rec["feet"]]
        summary["counts"] = {kind: statuses.count(kind) for kind in ("visible_candidate", "occluded", "unknown")}
        _write_json(output / "diagnostic_records.json", records)
        summary["artifacts"] = ["README.txt", "summary.json", "diagnostic_records.json"] + [
            rec["frame"][key] for rec in records for key in ("rgb_file", "overlay_file")]
        summary["artifacts"].append("diagnostic_bundle.zip")
        _write_json(output / "summary.json", summary)
        with zipfile.ZipFile(output / "diagnostic_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in summary["artifacts"]:
                if name != "diagnostic_bundle.zip" and (output / name).is_file():
                    archive.write(output / name, name)
    return summary
