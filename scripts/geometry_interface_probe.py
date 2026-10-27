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
                annotator.detach([product])
                status["detach"] = "ok_after_attach_failure"
            except Exception as cleanup_exc:
                status["detach_error"] = short_error(cleanup_exc)
    return statuses, attached


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


def _scene_snapshot(stage, camera, Gf, UsdGeom):
    cache = UsdGeom.XformCache()
    result = {"camera_world_transform_rows": _rows(cache.GetLocalToWorldTransform(camera.GetPrim())),
              "actor_roots": {}}
    for name in ("A", "B"):
        prim = stage.GetPrimAtPath("/World/Actors/" + name)
        if prim.IsValid():
            result["actor_roots"][name] = _rows(cache.GetLocalToWorldTransform(prim))
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
               "depth_semantics_confirmed": False, "blocked_reasons": [], "errors": [],
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
                   "before": _scene_snapshot(stage, camera, Gf, UsdGeom),
                   "outputs": {}}
            records.append(rec)
            pose_scene(sample)
            rec["after_pose"] = _scene_snapshot(stage, camera, Gf, UsdGeom)
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
            rec["after_capture"] = _scene_snapshot(stage, camera, Gf, UsdGeom)
            rec["frame_evidence"] = frame_evidence(rec["outputs"], rec["rgb"]["shape"])
            summary["samples_saved"] += 1
        summary["status"] = "passed"
    except Exception as exc:
        fatal = exc
        summary["errors"].append(short_error(exc))
    finally:
        for name, annotator in attached.items():
            try:
                annotator.detach([product])
                summary["annotators"][name]["detach"] = "ok"
            except Exception as exc:
                summary["annotators"][name].update(detach="failed", detach_error=short_error(exc))
        if summary["samples_saved"] != 2:
            summary["blocked_reasons"].append("two_rgb_poses_not_captured")
        if not summary["foot_level_mapping_found"]:
            summary["blocked_reasons"].append("foot_visual_id_mapping_unverified")
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
