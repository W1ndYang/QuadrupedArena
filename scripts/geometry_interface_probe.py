"""Read-only annotator probe for the existing formal vision-trial scene.

The parsing helpers are Isaac-free. Runtime collection deliberately never turns
an annotator result into a visibility or same-frame verification claim.
"""

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import zipfile

import numpy as np

from formal_geometry_diagnostic import (
    FOOT_LINKS, HEIGHT, WIDTH, _foot_candidates, _rows, _scenarios, _vec,
    project_local_candidate,
)


SEGMENTATION = ("instance_id_segmentation", "instance_segmentation", "semantic_segmentation")
DEPTH = ("distance_to_camera", "distance_to_image_plane")
CANDIDATES = SEGMENTATION + DEPTH
PIXEL_DEPTH_TOLERANCE_M = 0.03
PIXEL_SAMPLING_CONVENTION = "truncate nonnegative projected (u,v) toward zero; index array[v,u]"


def interface_probe_mode_error(args):
    if not args.geometry_interface_probe:
        return None
    if not args.vision_trial:
        return "--geometry-interface-probe requires --vision-trial"
    if any((args.geometry_diagnostic, args.vision_smoke, args.semantic_skills,
            args.deepseek, args.qwen_vision, args.assessment, args.g1b1_visible_spawn)):
        return ("--geometry-interface-probe cannot be combined with geometry diagnostic, "
                "smoke, Skill, planner, assessment or G1-B.1 modes")
    if args.diagnostic_output_dir:
        return "--diagnostic-output-dir is only for --geometry-diagnostic"
    return None


def short_error(exc):
    return {"type": type(exc).__name__, "message": str(exc)[:500]}


def split_result(raw):
    """Keep metadata separate; never coerce a colored preview into an ID mask."""
    if raw is None:
        return None, None, "NoneType"
    if isinstance(raw, dict):
        return raw.get("data"), raw.get("info"), "dict"
    return raw, None, type(raw).__name__


def safe_json(value, depth=0):
    """Bound arbitrary metadata while preserving small id-to-label mappings."""
    if depth > 12:
        return {"truncated": "nesting"}
    if value is None or isinstance(value, (str, bool, int)):
        return value if not isinstance(value, str) else value[:500]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return safe_json(value.item(), depth + 1)
    if isinstance(value, dict):
        return {str(k)[:120]: safe_json(v, depth + 1)
                for k, v in list(value.items())[:2000]}
    if isinstance(value, (list, tuple)):
        return [safe_json(v, depth + 1) for v in value[:100]]
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def numeric_array(data):
    if data is None:
        return None
    try:
        array = np.asarray(data)
    except (TypeError, ValueError):
        return None
    return array if array.dtype.kind in "biuf" else None


def summarize_result(raw, kind, resolution=(WIDTH, HEIGHT)):
    data, info, result_type = split_result(raw)
    result = {"return_type": result_type, "info_type": type(info).__name__,
              "info": safe_json(info), "data_type": type(data).__name__}
    array = numeric_array(data)
    if array is None:
        result.update(array_available=False, issue="missing_or_non_numeric_array")
        return result, None
    result.update(array_available=True, shape=list(array.shape), dtype=str(array.dtype),
                  resolution_matches=bool(array.ndim >= 2 and array.shape[:2] == (resolution[1], resolution[0])))
    if kind in SEGMENTATION:
        result["integer_id_mask"] = bool(array.ndim == 2 and array.dtype.kind in "iu")
        if result["integer_id_mask"]:
            ids, counts = np.unique(array, return_counts=True)
            result.update(unique_count=int(len(ids)),
                          sample_values=[{"id": int(i), "pixels": int(c)}
                                         for i, c in zip(ids[:12], counts[:12])])
        else:
            result["issue"] = "not_2d_integer_id_mask"
    elif kind in DEPTH:
        finite = np.isfinite(array)
        values = array[finite]
        result.update(finite_count=int(finite.sum()), nan_count=int(np.isnan(array).sum()),
                      pos_inf_count=int(np.isposinf(array).sum()),
                      neg_inf_count=int(np.isneginf(array).sum()),
                      zero_count=int(np.count_nonzero(array == 0)),
                      finite_min=float(values.min()) if values.size else None,
                      finite_max=float(values.max()) if values.size else None,
                      depth_semantics_confirmed=False,
                      units=None, distance_definition=None, invalid_value_definition=None)
    return result, array


def _absolute_paths(value):
    if isinstance(value, str):
        return [value] if value.startswith("/") else []
    if isinstance(value, dict):
        return [p for v in value.values() for p in _absolute_paths(v)]
    if isinstance(value, (tuple, list)):
        return [p for v in value for p in _absolute_paths(v)]
    return []


def id_to_paths(info):
    """Accept only an explicit ID mapping and absolute prim paths in its values."""
    if not isinstance(info, dict):
        return {}, "mapping_unavailable"
    mapping = info.get("idToLabels")
    if not isinstance(mapping, dict):
        return {}, "mapping_unavailable"
    parsed = {}
    for key, label in mapping.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        paths = sorted(set(_absolute_paths(label)))
        if paths:
            parsed[index] = paths
    return parsed, "parsed" if parsed else "mapping_unavailable"


def mask_mapping(array, info, foot_visuals):
    """Exact visual-prim matches only; an actor or ankle parent never counts."""
    paths, parse_status = id_to_paths(info)
    result = {"status": parse_status, "id_to_paths": {str(k): v for k, v in paths.items()},
              "feet": {side: {"ids": [], "pixels": 0, "bbox_xyxy": None}
                       for side in FOOT_LINKS}}
    if array is None or array.ndim != 2 or array.dtype.kind not in "iu":
        result["status"] = "not_2d_integer_id_mask"
        return result
    all_visuals = {p for visual_paths in foot_visuals.values() for p in visual_paths}
    actor_parent = any(any((p == "/World/Actors/B" or p.startswith("/World/Actors/B/"))
                           and p not in all_visuals for p in path_list)
                       for path_list in paths.values())
    found = False
    for side, visuals in foot_visuals.items():
        # An ID carrying paths for both feet or for a parent is ambiguous.
        ids = [i for i, path_list in paths.items() if path_list and all(p in visuals for p in path_list)]
        mask = np.isin(array, ids) if ids else None
        count = int(np.count_nonzero(mask)) if mask is not None else 0
        bbox = None
        if count:
            yy, xx = np.nonzero(mask)
            bbox = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
        result["feet"][side] = {"ids": sorted(ids), "pixels": count, "bbox_xyxy": bbox}
        found |= bool(ids)
    if found:
        result["status"] = "foot_visual_mapping_found"
    elif actor_parent:
        result["status"] = "actor_level_only"
    elif paths:
        result["status"] = "mapping_without_unique_foot_visual"
    return result


def projected_pixel_index(projection, width=WIDTH, height=HEIGHT):
    """R/G1-A use top-left (u,v); integer conversion follows the prior probe check."""
    if not isinstance(projection, dict) or projection.get("status") != "in_frame":
        return None, "projection_not_in_frame"
    pixel = projection.get("pixel")
    if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
        return None, "projected_pixel_missing"
    try:
        u, v = (float(pixel[0]), float(pixel[1]))
    except (TypeError, ValueError, OverflowError):
        return None, "projected_pixel_invalid"
    if not math.isfinite(u) or not math.isfinite(v):
        return None, "projected_pixel_nonfinite"
    if not (0 <= u < width and 0 <= v < height):
        return None, "projected_pixel_out_of_bounds"
    return (int(u), int(v)), None


def candidate_pixel_result(foot, arrays, segmentation_info,
                           depth_semantics_confirmed=False,
                           tolerance_m=PIXEL_DEPTH_TOLERANCE_M):
    """Audit a mesh-vertex pixel; unconfirmed depth keeps final status unknown."""
    projection = foot.get("projection", {})
    pixel = projection.get("pixel") if isinstance(projection, dict) else None
    result = {"candidate_link_prim": foot.get("link_prim"),
              "candidate_visual_prim": foot.get("visual_prim"),
              "candidate_world_m": foot.get("world_m"),
              "projected_pixel_float_uv": pixel,
              "pixel_sampling_convention": PIXEL_SAMPLING_CONVENTION,
              "sampled_pixel_uv": None, "array_index_vu": None,
              "in_frame": False, "segmentation_id": None,
              "segmentation_prim_path": None,
              "segmentation_matches_candidate_visual": None,
              "distance_to_camera_sample": None,
              "distance_to_image_plane_sample": None,
              "predicted_ray_distance_m": foot.get("ray_distance_m"),
              "predicted_image_plane_distance_m": foot.get("image_plane_distance_m"),
              "distance_to_camera_delta_m": None,
              "distance_to_image_plane_delta_m": None,
              "depth_tolerance_m": tolerance_m,
              "depth_semantics_confirmed": bool(depth_semantics_confirmed),
              "candidate_pixel_status": "unknown", "heuristic_unverified": None,
              "status_reason": None, "is_validated_sole_contact": False,
              "scope": "candidate mesh vertex pixel relation only"}
    def unknown(reason):
        result["status_reason"] = reason
        return result
    if not isinstance(tolerance_m, (int, float)) or not math.isfinite(tolerance_m) or tolerance_m <= 0:
        return unknown("invalid_depth_tolerance")
    index, problem = projected_pixel_index(projection)
    if problem:
        return unknown(problem)
    u, v = index
    result.update(sampled_pixel_uv=[u, v], array_index_vu=[v, u], in_frame=True)
    if not foot.get("visual_prim"):
        return unknown("candidate_visual_prim_unavailable")
    names = ("instance_id_segmentation", "distance_to_camera", "distance_to_image_plane")
    resolved = {name: numeric_array(arrays.get(name)) for name in names}
    if any(value is None for value in resolved.values()):
        return unknown("required_array_unavailable")
    if any(value.shape != (HEIGHT, WIDTH) for value in resolved.values()):
        return unknown("array_shape_mismatch")
    mask = resolved["instance_id_segmentation"]
    if mask.dtype.kind not in "iu":
        return unknown("segmentation_not_integer_id_mask")
    identity = int(mask[v, u])
    result["segmentation_id"] = identity
    paths, mapping_status = id_to_paths(segmentation_info)
    path_list = paths.get(identity, [])
    if mapping_status != "parsed" or len(path_list) != 1:
        return unknown("segmentation_id_unmapped_or_ambiguous")
    path = path_list[0]
    result["segmentation_prim_path"] = path
    result["segmentation_matches_candidate_visual"] = path == foot["visual_prim"]
    measured = {}
    invalid_depth = False
    for name, key in (("distance_to_camera", "distance_to_camera_sample"),
                      ("distance_to_image_plane", "distance_to_image_plane_sample")):
        value = float(resolved[name][v, u])
        result[key] = value if math.isfinite(value) else str(value)
        if not math.isfinite(value) or value <= 0:
            invalid_depth = True
        measured[name] = value
    if invalid_depth:
        return unknown("invalid_depth_sample")
    try:
        ray = float(foot["ray_distance_m"])
        plane = float(foot["image_plane_distance_m"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return unknown("predicted_distance_unavailable")
    if not all(math.isfinite(x) and x > 0 for x in (ray, plane)):
        return unknown("predicted_distance_invalid")
    ray_delta = measured["distance_to_camera"] - ray
    plane_delta = measured["distance_to_image_plane"] - plane
    result.update(distance_to_camera_delta_m=ray_delta,
                  distance_to_image_plane_delta_m=plane_delta)
    if path == foot["visual_prim"] and abs(ray_delta) <= tolerance_m and abs(plane_delta) <= tolerance_m:
        hypothesis = "visible_candidate_pixel"
    elif (path != foot["visual_prim"]
          and not foot["visual_prim"].startswith(path + "/")
          and not path.startswith(foot["visual_prim"] + "/")
          and ray_delta < -tolerance_m and plane_delta < -tolerance_m):
        hypothesis = "occluded_candidate_pixel"
    else:
        return unknown("id_and_depth_do_not_jointly_support_candidate_status")
    if not depth_semantics_confirmed:
        result["heuristic_unverified"] = hypothesis
        return unknown("depth_semantics_unconfirmed")
    result["candidate_pixel_status"] = hypothesis
    result["status_reason"] = "exact_prim_and_two_depth_checks" if path == foot["visual_prim"] else "known_other_prim_and_two_nearer_depths"
    return result


def frame_evidence(outputs, rgb_shape):
    """Tokens and dimensions are observations, never an implicit sync contract."""
    tokens = {}
    sizes_match = True
    for name, item in outputs.items():
        info = item.get("info")
        if isinstance(info, dict):
            tokens[name] = {k: safe_json(info[k]) for k in ("frameId", "frame_id", "frame", "timestamp")
                            if k in info}
        else:
            tokens[name] = {}
        if item.get("array_available") and not item.get("resolution_matches"):
            sizes_match = False
    known = [json.dumps(token, sort_keys=True) for token in tokens.values() if token]
    return {"tokens": tokens, "rgb_token_available": False,
            "tokens_present_for_all": False,
            "tokens_equal_if_present": bool(known) and len(set(known)) == 1,
            "dimensions_match_rgb": sizes_match and rgb_shape[:2] == [HEIGHT, WIDTH],
            "same_frame_verified": False,
            "reason": "renderer synchronization contract not established"}


def artifact_list(output):
    return sorted(str(p.relative_to(output)) for p in output.rglob("*")
                  if p.is_file() and p.name != "diagnostic_bundle.zip")


def attach_candidates(registry, product):
    statuses = {}
    attached = {}
    for name in CANDIDATES:
        status = {"create": "not_attempted", "attach": "not_attempted",
                  "reads": {}, "detach": "not_attempted",
                  "requested_init_params": {"colorize": False} if name in SEGMENTATION else {}}
        statuses[name] = status
        try:
            if name in SEGMENTATION:
                try:
                    annotator = registry.get_annotator(name, init_params={"colorize": False})
                except TypeError as exc:
                    status["init_params_error"] = short_error(exc)
                    annotator = registry.get_annotator(name)
                    status["init_params_fallback"] = "without_colorize_request"
            else:
                annotator = registry.get_annotator(name)
            status["create"] = "ok"
        except Exception as exc:
            status.update(create="failed", create_error=short_error(exc))
            continue
        try:
            annotator.attach([product])
            status["attach"] = "ok"
            attached[name] = annotator
        except Exception as exc:
            status.update(attach="failed", attach_error=short_error(exc))
            try:
                # Each newly created annotator is used only by this probe. Isaac
                # Sim 5.1's Replicator example detaches such instances with no
                # argument; passing [HydraTexture] failed in the Ubuntu run.
                status["detach_argument"] = {"kind": "none", "value": None}
                annotator.detach()
                status["detach"] = "ok_after_attach_failure"
            except Exception as cleanup_exc:
                status["detach"] = "failed_after_attach_failure"
                status["detach_error"] = short_error(cleanup_exc)
    return statuses, attached


def detach_candidates(attached, statuses, product_path):
    """Detach every probe-owned annotator; one failure never masks another."""
    for name, annotator in attached.items():
        status = statuses[name]
        status["detach_argument"] = {"kind": "none", "value": None,
                                     "attached_render_product_path": product_path}
        try:
            annotator.detach()
            status["detach"] = "ok"
        except Exception as exc:
            status.update(detach="failed", detach_error=short_error(exc))


def collect_candidate(annotator, name, visuals, output, scenario, frame_index):
    try:
        raw = annotator.get_data()
        item, array = summarize_result(raw, name)
        _, info, _ = split_result(raw)
        if name in SEGMENTATION:
            item["mapping"] = mask_mapping(array, info, visuals)
            if not item.get("resolution_matches"):
                item["mapping"]["status"] = "resolution_mismatch_or_missing"
                for foot in item["mapping"]["feet"].values():
                    foot.update(ids=[], pixels=0, bbox_xyxy=None)
        if array is not None:
            path = f"frames/{scenario}_{name}.npy"
            np.save(output / path, array, allow_pickle=False)
            item["raw_file"] = path
            item["sha256"] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        item["frame_index"] = frame_index
        return item, "ok" if array is not None else "no_numeric_array"
    except Exception as exc:
        return {"error": short_error(exc), "frame_index": frame_index,
                "array_available": False}, "failed"


def probe_execution_status(samples_saved, statuses):
    if samples_saved == 0:
        return "blocked"
    failed = any(a["create"] != "ok" or a["attach"] != "ok"
                 or len(a["reads"]) != samples_saved
                 or any(v != "ok" for v in a["reads"].values()) or a["detach"] != "ok"
                 for a in statuses.values())
    return "partial" if failed or samples_saved < 2 else "complete"


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _foot_visuals(stage, UsdGeom):
    result = {}
    for side in FOOT_LINKS:
        prim = stage.GetPrimAtPath("/World/Actors/B/" + side)
        result[side] = sorted(str(child.GetPath()) for child in prim.GetChildren()
                              if prim.IsValid() and child.IsA(UsdGeom.Mesh)) if prim.IsValid() else []
    return result


def _scene_snapshot(stage, camera, Gf, UsdGeom, visuals=None):
    cache = UsdGeom.XformCache()
    result = {"camera_world_transform_rows": None, "actor_roots": {},
              "tracked_prims": {}}
    tracked = [str(camera.GetPath())]
    for side in FOOT_LINKS:
        tracked.append("/World/Actors/B/" + side)
        tracked.extend((visuals or {}).get(side, []))
    for path in tracked:
        try:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                result["tracked_prims"][path] = {"status": "missing", "reason": "prim_not_found"}
                continue
            rows = _rows(cache.GetLocalToWorldTransform(prim))
            result["tracked_prims"][path] = {"status": "ok", "world_transform_rows": rows}
            if path == str(camera.GetPath()):
                result["camera_world_transform_rows"] = rows
        except Exception as exc:
            result["tracked_prims"][path] = {"status": "failed", "error": short_error(exc)}
    for name in ("A", "B"):
        try:
            prim = stage.GetPrimAtPath("/World/Actors/" + name)
            if prim.IsValid():
                result["actor_roots"][name] = _rows(cache.GetLocalToWorldTransform(prim))
        except Exception as exc:
            result["actor_roots"][name] = {"status": "failed", "error": short_error(exc)}
    return result


def _candidate_refs(stage, camera, Gf, UsdGeom):
    cache = UsdGeom.XformCache()
    camera_world = cache.GetLocalToWorldTransform(camera.GetPrim())
    inverse = camera_world.GetInverse()
    eye = _vec(camera_world.Transform(Gf.Vec3d(0, 0, 0)))
    intrinsics = [float(camera.GetFocalLengthAttr().Get()),
                  float(camera.GetHorizontalApertureAttr().Get()),
                  float(camera.GetVerticalApertureAttr().Get())]
    clipping = [float(v) for v in camera.GetClippingRangeAttr().Get()]
    feet = _foot_candidates(stage, eye, cache, Gf, UsdGeom)
    for foot in feet:
        if "world_m" not in foot:
            continue
        local = _vec(inverse.Transform(Gf.Vec3d(*foot["world_m"])))
        projection = project_local_candidate(local, intrinsics, clipping)
        foot.update(camera_local_m=local, projection=projection,
                    reference_only="candidate geometry, not a visibility result")
        if projection["status"] == "in_frame":
            foot["ray_distance_m"] = math.dist(local, (0, 0, 0))
            foot["image_plane_distance_m"] = -local[2]
    return feet


def run(output_dir, state, cfg, stage, camera, product, rgb, rep, app, pose_scene):
    """Capture two authored poses; keep failures and raw numeric buffers."""
    from pxr import Gf, UsdGeom
    from PIL import Image

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    (output / "README.txt").write_text(
        "Formal scene read-only annotator probe. No visibility or same-frame claim.\n"
        "Inspect summary.json, probe_records.json, raw RGB and available .npy buffers.\n")
    env = {"python": platform.python_version(), "isaacsim": _version("isaacsim"),
           "isaaclab": _version("isaaclab"), "omni_replicator_core": _version("omni.replicator.core"),
           "kit": _version("omni.kit.app"), "camera_prim": str(camera.GetPath()),
           "replicator_module_version": safe_json(getattr(rep, "__version__", None)),
           "render_product_type": type(product).__name__,
           "render_product_path": str(getattr(product, "path", "")) or None,
           "render_product_repr": repr(product)[:300],
           "resolution": [WIDTH, HEIGHT], "version_source": "importlib.metadata; null means unavailable",
           "annotator_enumeration": "not_attempted",
           "annotator_names_tested": list(CANDIDATES)}
    try:
        getter = getattr(rep.annotators, "get_registered_annotators", None)
        if callable(getter):
            names = getter()
            env["annotator_enumeration"] = "ok"
            env["registered_annotator_names"] = [str(name) for name in names]
        else:
            env["annotator_enumeration"] = "api_unavailable"
    except Exception as exc:
        env["annotator_enumeration"] = "failed"
        env["annotator_enumeration_error"] = short_error(exc)
    try:
        import omni.kit.app
        kit_app = omni.kit.app.get_app()
        env["kit_app_version"] = str(kit_app.get_app_version()) if hasattr(kit_app, "get_app_version") else None
    except Exception as exc:
        env["kit_version_error"] = short_error(exc)
    try:
        import isaacsim
        env["isaacsim_module_version"] = safe_json(getattr(isaacsim, "__version__", None))
    except Exception as exc:
        env["isaacsim_module_error"] = short_error(exc)
    visuals = _foot_visuals(stage, UsdGeom)
    summary = {"mode": "geometry_interface_probe", "status": "failed", "probe_status": "blocked",
               "runtime_verified": False, "same_frame_verified": False,
               "same_capture_cycle_evidence": False, "environment": env,
               "candidate_foot_visuals": visuals, "samples_requested": 2, "samples_saved": 0,
               "annotators": {}, "foot_level_mapping_found": False,
               "depth_semantics_confirmed": False,
               "pixel_depth_tolerance_m": PIXEL_DEPTH_TOLERANCE_M,
               "blocked_reasons": [], "errors": [],
               "artifacts": [], "missing_artifacts": []}
    attached = {}
    records = []
    fatal = None
    try:
        summary["annotators"], attached = attach_candidates(rep.AnnotatorRegistry, product)
        for index, (scenario, sample) in enumerate(_scenarios(state, cfg)):
            rec = {"frame_index": index, "scenario": scenario,
                   "simulation_seconds": float(sample["t"]), "same_frame_verified": False,
                   "actor_state": {"xy": sample["xy"], "yaw_radians": sample["yaw"]},
                   "capture_sequence": ["snapshot_before", "pose_scene", "12 x app.update",
                                        "rgb.get_data", "annotator.get_data each", "snapshot_after"],
                   "app_update_count": 12, "orchestrator_step_count": 0,
                   "before": _scene_snapshot(stage, camera, Gf, UsdGeom, visuals),
                   "outputs": {}}
            records.append(rec)
            pose_scene(sample)
            rec["after_pose"] = _scene_snapshot(stage, camera, Gf, UsdGeom, visuals)
            for _ in range(12):
                app.update()
            pixels = np.asarray(rgb.get_data())
            if pixels.ndim != 3 or pixels.shape[:2] != (HEIGHT, WIDTH) or pixels.shape[2] < 3 or pixels[..., :3].std() < 1:
                raise RuntimeError(f"Missing or blank RGB in probe frame {index}: {pixels.shape}")
            rgb_file = f"frames/{scenario}_rgb.png"
            Image.fromarray(pixels[..., :3].astype(np.uint8).copy()).save(output / rgb_file)
            rec["rgb"] = {"file": rgb_file, "shape": list(pixels.shape), "dtype": str(pixels.dtype),
                          "sha256": hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest(),
                          "render_product": env["render_product_path"]}
            rec["feet"] = _candidate_refs(stage, camera, Gf, UsdGeom)
            for name, annotator in attached.items():
                item, read_status = collect_candidate(annotator, name, visuals, output, scenario, index)
                rec["outputs"][name] = item
                summary["annotators"][name]["reads"][scenario] = read_status
                if name in SEGMENTATION:
                    summary["foot_level_mapping_found"] |= item.get("mapping", {}).get("status") == "foot_visual_mapping_found"
            sampled_arrays = {}
            for name in ("instance_id_segmentation",) + DEPTH:
                raw_file = rec["outputs"].get(name, {}).get("raw_file")
                if raw_file:
                    try:
                        sampled_arrays[name] = np.load(output / raw_file, allow_pickle=False)
                    except Exception as exc:
                        rec["outputs"][name]["candidate_sample_load_error"] = short_error(exc)
            seg_info = rec["outputs"].get("instance_id_segmentation", {}).get("info")
            rec["candidate_pixels"] = [candidate_pixel_result(foot, sampled_arrays, seg_info,
                                        depth_semantics_confirmed=summary["depth_semantics_confirmed"])
                                       for foot in rec["feet"]]
            rec["after_capture"] = _scene_snapshot(stage, camera, Gf, UsdGeom, visuals)
            rec["frame_evidence"] = frame_evidence(rec["outputs"], rec["rgb"]["shape"])
            summary["samples_saved"] += 1
        summary["status"] = "passed"
    except Exception as exc:
        fatal = exc
        summary["errors"].append(short_error(exc))
    finally:
        detach_candidates(attached, summary["annotators"], env["render_product_path"])
        summary["detach_counts"] = {kind: sum(a["detach"] == kind for a in summary["annotators"].values())
                                    for kind in ("ok", "failed", "not_attempted", "failed_after_attach_failure",
                                                 "ok_after_attach_failure")}
        statuses = [item["candidate_pixel_status"] for rec in records
                    for item in rec.get("candidate_pixels", [])]
        hypotheses = [item["heuristic_unverified"] for rec in records
                      for item in rec.get("candidate_pixels", []) if item["heuristic_unverified"]]
        summary["candidate_pixel_counts"] = {kind: statuses.count(kind) for kind in
                                              ("visible_candidate_pixel", "occluded_candidate_pixel", "unknown")}
        summary["heuristic_unverified_counts"] = {kind: hypotheses.count(kind) for kind in
                                                   ("visible_candidate_pixel", "occluded_candidate_pixel")}
        summary["link_transform_status"] = {rec["scenario"]: {moment: {
            "ok": sum(item["status"] == "ok" for item in rec.get(moment, {}).get("tracked_prims", {}).values()),
            "missing_or_failed": [path for path, item in rec.get(moment, {}).get("tracked_prims", {}).items()
                                  if item["status"] != "ok"]}
            for moment in ("before", "after_pose", "after_capture")}
            for rec in records}
        if summary["samples_saved"] != 2:
            summary["blocked_reasons"].append("two_rgb_poses_not_captured")
        if not summary["foot_level_mapping_found"]:
            summary["blocked_reasons"].append("foot_visual_id_mapping_unverified")
        if summary["detach_counts"]["failed"] or summary["detach_counts"]["failed_after_attach_failure"]:
            summary["blocked_reasons"].append("probe_annotator_detach_failed")
        summary["blocked_reasons"].append("depth_semantics_and_renderer_sync_unverified")
        summary["probe_status"] = probe_execution_status(summary["samples_saved"], summary["annotators"])
        if summary["status"] == "passed" and summary["samples_saved"] != 2:
            summary["status"] = "failed"
        if len(records) == 2 and all("rgb" in r for r in records):
            rgb_changed = records[0]["rgb"]["sha256"] != records[1]["rgb"]["sha256"]
            available = set(records[0]["outputs"]) & set(records[1]["outputs"])
            changed = bool(available) and all(records[0]["outputs"][n].get("sha256") != records[1]["outputs"][n].get("sha256")
                                              and records[0]["outputs"][n].get("sha256")
                                              and records[1]["outputs"][n].get("sha256") for n in available)
            summary["state_switch"] = {"rgb_changed": rgb_changed,
                                       "available_annotators": sorted(available),
                                       "all_available_buffers_changed": bool(changed)}
            summary["same_capture_cycle_evidence"] = bool(rgb_changed and changed)
        (output / "probe_records.json").write_text(json.dumps(safe_json(records), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        summary["missing_artifacts"] = [f"frames/{scenario}_rgb.png" for scenario, _ in _scenarios(state, cfg)
                                        if not (output / f"frames/{scenario}_rgb.png").exists()]
        summary["missing_artifacts"] += [f"frames/{scenario}_{name}.npy"
                                         for scenario, _ in _scenarios(state, cfg) for name in CANDIDATES
                                         if not (output / f"frames/{scenario}_{name}.npy").exists()]
        summary["artifacts"] = artifact_list(output) + ["summary.json", "diagnostic_bundle.zip"]
        (output / "summary.json").write_text(json.dumps(safe_json(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        with zipfile.ZipFile(output / "diagnostic_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in artifact_list(output):
                bundle.write(output / name, name)
    if fatal:
        raise fatal
    return summary
