"""Small Isaac Sim camera/projection/PhysX visibility experiment.

Run on an Isaac Sim 5.1 Ubuntu host. ``--self-test`` uses only the Python stdlib.
G1-B emits experimental sighting candidates; it does not call a Skill or load
arena assets.
"""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import traceback

from synthetic_sighting import SCHEMA, build_sighting_candidate


WIDTH, HEIGHT = 768, 432
CAMERA_PATH = "/World/Camera"
TARGET_PATH = "/World/Target"
WALL_PATH = "/World/Wall"
EYE = (0.0, -4.0, 1.35)
LOOK_AT = (0.0, 1.5, 1.0)
FOOT_POINT = (0.0, 1.34, 0.08)
# The endpoints lie inside the target collider so a nearest-hit ray can hit its
# front surface before reaching them. All three scenes use these same samples.
BODY_SAMPLES = {
    "head": (0.0, 1.50, 1.65),
    "torso": (0.0, 1.50, 1.05),
    "left": (-0.35, 1.50, 1.05),
    "right": (0.35, 1.50, 1.05),
    "lower": (0.0, 1.50, 0.35),
}
SCENARIOS = {
    "clear": {"center": (4.0, -0.2, 1.10), "size": (0.40, 0.25, 2.20)},
    "partial": {"center": (-0.30, -0.2, 1.10), "size": (0.46, 0.25, 2.20)},
    "full": {"center": (0.0, -0.2, 1.10), "size": (2.40, 0.25, 2.20)},
}


def project_camera_point(point, width, height, focal_mm, aperture_h_mm, aperture_v_mm):
    """USD camera local +X right, +Y up, -Z forward; image origin top left."""
    x, y, z = (float(v) for v in point)
    if not all(math.isfinite(v) for v in (x, y, z, focal_mm, aperture_h_mm, aperture_v_mm)):
        raise ValueError("Non-finite projection input")
    if width <= 0 or height <= 0 or min(focal_mm, aperture_h_mm, aperture_v_mm) <= 0:
        raise ValueError("Invalid image size or camera intrinsics")
    depth = -z
    if depth <= 0:
        raise ValueError("Point is on or behind the USD camera plane")
    u = width / 2 + focal_mm / aperture_h_mm * width * x / depth
    v = height / 2 - focal_mm / aperture_v_mm * height * y / depth
    return {"pixel": [u, v], "depth_m": depth,
            "in_frame": 0 <= u < width and 0 <= v < height}


def self_test():
    intrinsics = (WIDTH, HEIGHT, 12.0, 24.0, 13.5)
    center = project_camera_point((0, 0, -2), *intrinsics)
    right = project_camera_point((0.5, 0, -2), *intrinsics)
    above = project_camera_point((0, 0.5, -2), *intrinsics)
    assert center["pixel"] == [WIDTH / 2, HEIGHT / 2]
    assert center["depth_m"] == 2 and center["in_frame"]
    assert right["pixel"][0] == WIDTH / 2 + 96
    assert above["pixel"][1] == HEIGHT / 2 - 96
    assert not project_camera_point((4, 0, -2), *intrinsics)["in_frame"]
    for point in ((0, 0, 0), (0, 0, 1)):
        try:
            project_camera_point(point, *intrinsics)
        except ValueError:
            pass
        else:
            raise AssertionError("Behind-camera point was accepted")
    visible = {"verdict": "visible"}
    candidate = build_sighting_candidate(center, visible, 0.1, "clear_rgb.png", WIDTH, HEIGHT)
    assert candidate["status"] == "emitted"
    assert candidate["payload"] == {"time": 0.1, "feet": [384.0, 216.0]}
    assert set(candidate["payload"]) == {"time", "feet"}
    assert candidate["skill_compatible"] is False
    assert candidate["integration_status"] == "not_connected"
    assert build_sighting_candidate(center, {"verdict": "blocked"}, 0.1,
                                    "x.png", WIDTH, HEIGHT)["reason"] == "anchor_occluded"
    assert build_sighting_candidate(center, {"verdict": "unknown"}, 0.1,
                                    "x.png", WIDTH, HEIGHT)["reason"] == "anchor_visibility_unknown"
    assert build_sighting_candidate(project_camera_point((4, 0, -2), *intrinsics),
                                    visible, 0.1, "x.png", WIDTH, HEIGHT)["reason"] == "projection_out_of_frame"
    for pixel in (None, [float("nan"), 216], [384, float("inf")]):
        invalid = build_sighting_candidate({"pixel": pixel, "depth_m": 2.0,
                                            "in_frame": True},
                                           visible, 0.1, "x.png", WIDTH, HEIGHT)
        assert invalid["payload"] is None
        json.dumps(invalid, allow_nan=False)
    assert build_sighting_candidate({"pixel": [384, 216], "in_frame": True},
                                    visible, 0.1, "x.png", WIDTH, HEIGHT)["reason"] == "projection_invalid_depth"
    assert build_sighting_candidate(center, visible, 0.1, None,
                                    WIDTH, HEIGHT)["reason"] == "frame_unavailable"
    assert build_sighting_candidate(center, visible, None, "x.png", WIDTH, HEIGHT)["payload"] is None
    assert json.dumps(candidate, sort_keys=True, allow_nan=False) == json.dumps(
        build_sighting_candidate(center, visible, 0.1, "clear_rgb.png", WIDTH, HEIGHT),
        sort_keys=True, allow_nan=False)
    print("SELF_TEST passed: projection and G1-B candidate logic/serialization")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run_isaac(output, result):
    # The package entry point is an Ubuntu Isaac Sim 5.1 assumption. All Kit,
    # Omniverse and USD imports occur only after SimulationApp has started.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "enable_cameras": True})
    sim = None
    rgb = depth = product = None
    candidates = {"schema": SCHEMA, "integration_status": "not_connected", "scenarios": {}}
    candidates_path = output / "sighting_candidates.json"
    result["synthetic_sighting"] = {"file": str(candidates_path), "schema": SCHEMA,
                                    "emitted_count": 0, "not_emitted_count": 0}
    try:
        import numpy as np
        import omni.physx
        import omni.replicator.core as rep
        import isaaclab.sim as sim_utils
        from pxr import Gf, UsdGeom, UsdPhysics
        from PIL import Image, ImageDraw

        result["version"] = {"isaacsim_package": None, "isaaclab_package": None,
                             "target": "Isaac Sim 5.1.0.0"}
        try:
            result["version"]["isaacsim_package"] = importlib.metadata.version("isaacsim")
        except importlib.metadata.PackageNotFoundError:
            result["version"]["isaacsim_package"] = "metadata_unavailable"
        try:
            result["version"]["isaaclab_package"] = importlib.metadata.version("isaaclab")
        except importlib.metadata.PackageNotFoundError:
            pass
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=1 / 60, device="cuda:0", enable_scene_query_support=True,
            use_fabric=False))
        stage = sim.stage
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene").CreateGravityMagnitudeAttr(9.81)

        def cube(path, center, size, color, collide=True):
            root = UsdGeom.Xform.Define(stage, path)
            translate = root.AddTranslateOp()
            translate.Set(Gf.Vec3d(*center))
            geom = UsdGeom.Cube.Define(stage, path + "/Geometry")
            geom.CreateSizeAttr(1.0)
            scale = geom.AddScaleOp()
            scale.Set(Gf.Vec3d(*size))
            geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            if collide:
                UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            return translate, scale

        cube("/World/Ground", (0, 0, -0.05), (12, 12, 0.1), (0.35, 0.38, 0.42))
        cube("/World/Target", (0, 1.50, 0.95), (0.90, 0.32, 1.80), (0.12, 0.45, 0.85))
        # Visual marker on the front near the foot; no collision, so it cannot
        # mask the body in the PhysX query.
        cube("/World/FootMarker", (0, 1.29, 0.08), (0.10, 0.04, 0.10),
             (0.95, 0.80, 0.05), collide=False)
        wall_translate, wall_scale = cube(WALL_PATH, **SCENARIOS["clear"],
                                          color=(0.85, 0.18, 0.12))
        light = sim_utils.DomeLightCfg(intensity=1500, color=(1.0, 1.0, 1.0))
        light.func("/World/Light", light)

        camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
        camera.CreateFocalLengthAttr(12.0)
        camera.CreateHorizontalApertureAttr(24.0)
        camera.CreateVerticalApertureAttr(13.5)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.04, 100.0))
        # USD GetLookAt returns world->camera. Its inverse respects the USD
        # convention (-Z forward, +Y up) without guessing a quaternion.
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*EYE), Gf.Vec3d(*LOOK_AT), Gf.Vec3d(0, 0, 1)).GetInverse())
        product = rep.create.render_product(CAMERA_PATH, (WIDTH, HEIGHT))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([product])
        rep.orchestrator.set_capture_on_play(False)
        try:
            depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
            depth.attach([product])
            result["depth"] = {"status": "attached_unverified"}
        except Exception as exc:
            result["depth"] = {"status": "unavailable", "reason": str(exc)}
            depth = None

        world = UsdGeom.XformCache().GetLocalToWorldTransform(camera.GetPrim())
        inverse = world.GetInverse()
        actual_eye = world.Transform(Gf.Vec3d(0, 0, 0))
        actual_forward = world.Transform(Gf.Vec3d(0, 0, -1)) - actual_eye
        toward_target = Gf.Vec3d(*LOOK_AT) - actual_eye
        unit_forward = actual_forward.GetNormalized()
        unit_target = toward_target.GetNormalized()
        forward_alignment = sum(float(unit_forward[i] * unit_target[i]) for i in range(3))
        if forward_alignment < 0.99:
            raise RuntimeError(f"USD camera does not face target: dot={forward_alignment:.6f}")
        local_foot = inverse.Transform(Gf.Vec3d(*FOOT_POINT))
        focal = float(camera.GetFocalLengthAttr().Get())
        aperture_h = float(camera.GetHorizontalApertureAttr().Get())
        aperture_v = float(camera.GetVerticalApertureAttr().Get())
        try:
            projected = project_camera_point(tuple(local_foot), WIDTH, HEIGHT,
                                             focal, aperture_h, aperture_v)
        except ValueError as exc:
            projected = {"pixel": None, "in_frame": False, "error": str(exc)}
        projection_ok = projected["in_frame"]
        result["camera"] = {
            "prim_path": CAMERA_PATH, "image_width": WIDTH, "image_height": HEIGHT,
            "world_transform_rows": [[float(world[i][j]) for j in range(4)] for i in range(4)],
            "eye_world_m": [float(v) for v in actual_eye],
            "look_at_world_m": list(LOOK_AT),
            "forward_alignment": forward_alignment,
            "focal_length_mm": focal, "horizontal_aperture_mm": aperture_h,
            "vertical_aperture_mm": aperture_v,
            "convention": "USD camera -Z forward, +Y up; pixel origin top-left",
        }
        result["projection"] = {"foot_world_m": list(FOOT_POINT),
                                "foot_camera_m": [float(v) for v in local_foot], **projected}
        query = omni.physx.get_physx_scene_query_interface()
        for name, layout in SCENARIOS.items():
            wall_translate.Set(Gf.Vec3d(*layout["center"]))
            wall_scale.Set(Gf.Vec3d(*layout["size"]))
            # Reset/rebuild the static collision scene after each authored edit.
            # Whether this refreshes queries on the target Ubuntu stack is part
            # of the experiment; failures remain visible in result.json.
            sim.reset()
            for _ in range(4):
                sim.step(render=True)
            # Capture-on-play is disabled above. Physics/render steps alone do
            # not request an Annotator frame; Isaac Sim 5.1's standalone
            # Replicator example explicitly steps the orchestrator here.
            capture = {"attempts": 1, "rt_subframes": 4,
                       "delta_time": 0.0, "pause_timeline": False}
            try:
                rep.orchestrator.step(rt_subframes=4, delta_time=0.0,
                                      pause_timeline=False)
                capture["status"] = "completed"
            except Exception as exc:
                capture.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            rays = []
            for label, point in BODY_SAMPLES.items():
                vector = np.asarray(point) - np.asarray(tuple(actual_eye))
                distance = float(np.linalg.norm(vector))
                direction = tuple((vector / distance).tolist())
                hit = query.raycast_closest(tuple(actual_eye), direction, distance + 0.20)
                has_hit = bool(hit.get("hit", False))
                collider = str(hit.get("collision", "")) if has_hit else None
                if collider == TARGET_PATH + "/Geometry":
                    verdict = "visible"
                elif collider == WALL_PATH + "/Geometry":
                    verdict = "blocked"
                else:
                    verdict = "unknown"  # including no hit or unexpected prim
                position = hit.get("position") if has_hit else None
                rays.append({"sample": label, "target_world_m": list(point),
                             "query_distance_m": distance + 0.20,
                             "verdict": verdict, "hit": has_hit,
                             "collider": collider,
                             "hit_position_world_m": None if position is None else [float(v) for v in position]})
            counts = {kind: sum(ray["verdict"] == kind for ray in rays)
                      for kind in ("visible", "blocked", "unknown")}
            # Independent foot anchor: it never contributes to the original
            # five body sample counts or visible_ratio.
            anchor_vector = np.asarray(FOOT_POINT) - np.asarray(tuple(actual_eye))
            anchor_distance = float(np.linalg.norm(anchor_vector))
            anchor = {"target_world_m": list(FOOT_POINT),
                      "query_distance_m": anchor_distance + 0.20,
                      "verdict": "unknown", "hit": False,
                      "collider": None, "hit_position_world_m": None}
            try:
                anchor_hit = query.raycast_closest(
                    tuple(actual_eye), tuple((anchor_vector / anchor_distance).tolist()),
                    anchor_distance + 0.20)
                if not hasattr(anchor_hit, "get"):
                    anchor["reason"] = "uninterpretable_query_result"
                else:
                    anchor["hit"] = bool(anchor_hit.get("hit", False))
                    if anchor["hit"]:
                        collider = anchor_hit.get("collision")
                        collider_path = None if collider is None else str(collider)
                        if not collider_path or not collider_path.startswith("/"):
                            anchor["reason"] = "missing_collider"
                        else:
                            anchor["collider"] = collider_path
                            anchor["verdict"] = (
                                "visible" if anchor["collider"] == TARGET_PATH + "/Geometry"
                                else "blocked")
                        position = anchor_hit.get("position")
                        if position is not None:
                            anchor["hit_position_world_m"] = [float(v) for v in position]
                    else:
                        anchor["reason"] = "no_hit"
            except Exception as exc:
                anchor.update(verdict="unknown", reason=f"{type(exc).__name__}: {exc}")
            item = {"wall": {"prim_path": WALL_PATH + "/Geometry", **layout},
                    "capture": capture,
                    "sample_count": len(rays),
                    "visible_count": counts["visible"],
                    "blocked_count": counts["blocked"],
                    "unknown_count": counts["unknown"],
                    "visible_ratio": counts["visible"] / len(rays),
                    "rays": rays, "sighting_anchor_ray": anchor}
            result["scenarios"][name] = item

            rgb_shape = None
            try:
                pixels = np.asarray(rgb.get_data())
                rgb_shape = list(pixels.shape)
                if pixels.ndim != 3 or pixels.shape[:2] != (HEIGHT, WIDTH) or pixels.shape[2] < 3:
                    raise RuntimeError(f"Unexpected RGB shape {pixels.shape}")
                if capture["status"] != "completed":
                    raise RuntimeError("Replicator capture step failed: " + capture["reason"])
                if np.std(pixels[..., :3]) < 1:
                    raise RuntimeError("Blank RGB frame")
                original = Image.fromarray(pixels[..., :3].astype(np.uint8).copy())
                raw_path = output / f"{name}_rgb.png"
                overlay_path = output / f"{name}_projection.png"
                original.save(raw_path)
                marked = original.copy()
                draw = ImageDraw.Draw(marked)
                if projected["pixel"] is not None:
                    u, v = projected["pixel"]
                    draw.line(((u - 8, v), (u + 8, v)), fill=(255, 255, 0), width=2)
                    draw.line(((u, v - 8), (u, v + 8)), fill=(255, 255, 0), width=2)
                marked.save(overlay_path)
                item["rgb"] = {"status": "saved", "capture_attempts": 1,
                               "array_shape": rgb_shape, "raw": str(raw_path),
                               "projection_overlay": str(overlay_path)}
            except Exception as exc:
                item["rgb"] = {"status": "failed", "capture_attempts": 1,
                               "array_shape": rgb_shape, "reason": str(exc)}
            if depth is not None:
                depth_shape = None
                try:
                    values = np.asarray(depth.get_data())
                    depth_shape = list(values.shape)
                    if values.shape[:2] != (HEIGHT, WIDTH):
                        raise RuntimeError(f"Unexpected depth shape {values.shape}")
                    if capture["status"] != "completed":
                        raise RuntimeError("Replicator capture step failed: " + capture["reason"])
                    path = output / f"{name}_depth.npy"
                    np.save(path, values)
                    item["depth"] = {"status": "saved", "capture_attempts": 1,
                                     "array_shape": depth_shape, "file": str(path),
                                     "units": "distance_to_camera (Isaac annotator)"}
                except Exception as exc:
                    item["depth"] = {"status": "failed", "capture_attempts": 1,
                                     "array_shape": depth_shape, "reason": str(exc)}

            # Isaac Lab 2.3 inherits the Isaac Sim 5.1 SimulationContext
            # current_time property. Read it after this frame's capture; never
            # derive a timestamp from the scenario name or step count.
            frame_file = item["rgb"]["raw"] if item["rgb"]["status"] == "saved" else None
            simulation_time = None
            if frame_file is not None:
                try:
                    simulation_time = sim.current_time
                except Exception as exc:
                    item["sighting_time_error"] = f"{type(exc).__name__}: {exc}"
            candidate = build_sighting_candidate(projected, anchor, simulation_time,
                                                 frame_file, WIDTH, HEIGHT)
            candidates["scenarios"][name] = candidate
            key = "emitted_count" if candidate["status"] == "emitted" else "not_emitted_count"
            result["synthetic_sighting"][key] += 1

        clear = result["scenarios"]["clear"]
        partial = result["scenarios"]["partial"]
        full = result["scenarios"]["full"]
        ray_ok = (clear["visible_count"] == clear["sample_count"]
                  and 0 < partial["visible_count"] < partial["sample_count"]
                  and full["visible_count"] == 0
                  and full["blocked_count"] == full["sample_count"]
                  and all(x["unknown_count"] == 0 for x in (clear, partial, full)))
        rgb_ok = all(x.get("rgb", {}).get("status") == "saved"
                     for x in (clear, partial, full))
        sighting_ok = (candidates["scenarios"]["clear"]["status"] == "emitted"
                       and candidates["scenarios"]["full"]["status"] == "not_emitted")
        if depth is not None:
            failed_depth = [(name, scene["depth"]["reason"])
                            for name, scene in result["scenarios"].items()
                            if scene.get("depth", {}).get("status") != "saved"]
            result["depth"] = ({"status": "failed", "reason":
                                 f"{failed_depth[0][0]}: {failed_depth[0][1]}"}
                               if failed_depth else {"status": "saved"})
        result["checks"] = {"projection": projection_ok, "physx_raycast": ray_ok,
                            "rgb": rgb_ok, "depth_optional": True,
                            "synthetic_sighting": sighting_ok}
        if not projection_ok:
            result["errors"].append("Projected foot point is outside the camera frame or invalid")
        if not ray_ok:
            result["errors"].append("PhysX raycast classifications did not match clear/partial/full expectations")
        if not rgb_ok:
            result["errors"].append("One or more RGB captures failed")
        if not sighting_ok:
            result["errors"].append("Clear did not emit a sighting or full emitted one")
        result["status"] = "passed" if projection_ok and ray_ok and rgb_ok and sighting_ok else "failed"
    except Exception as exc:
        result["status"] = "failed"
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        if result["depth"].get("status") == "attached_unverified":
            result["depth"] = {"status": "failed", "reason": "Run ended before depth summary"}
        (output / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        # Persist before close: some Kit shutdown paths terminate the process.
        write_json(candidates_path, candidates)
        write_json(output / "result.json", result)
        if rgb is not None and product is not None:
            try:
                rgb.detach([product.path])
            except Exception:
                pass
        if depth is not None and product is not None:
            try:
                depth.detach([product.path])
            except Exception:
                pass
        if product is not None:
            try:
                product.destroy()
            except Exception:
                pass
        if sim is not None:
            try:
                sim.clear_all_callbacks()
                sim.clear_instance()
                sim.stop()
            except Exception:
                pass
        app.close(wait_for_replicator=False, skip_cleanup=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Pure Python projection and sighting checks; no Isaac imports")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/quadrupedarena_geometry_test"))
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "failed", "version": {"target": "Isaac Sim 5.1.0.0"},
              "camera": None, "projection": None, "scenarios": {},
              "depth": {"status": "not_attempted"}, "checks": {}, "errors": [],
              "api_assumptions": ["from isaacsim import SimulationApp",
                                  "Static collision refresh via SimulationContext.reset",
                                  "Replicator distance_to_camera annotator is optional"]}
    try:
        run_isaac(output, result)
    except Exception as exc:
        result["status"] = "failed"
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        (output / "error.log").write_text(traceback.format_exc())
    finally:
        write_json(output / "result.json", result)
    print(f"GEOMETRY_TEST {result['status']} | {output / 'result.json'}")
    if result["errors"]:
        print("Reason: " + result["errors"][0])
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
