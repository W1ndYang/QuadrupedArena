"""Isaac-free checks for the formal scene's read-only interface probe."""

import argparse
import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from geometry_interface_probe import (
    CANDIDATES, DEPTH, _scene_snapshot, alignment_status, artifact_list, attach_candidates,
    candidate_pixel_result, depth_invalid_reason,
    collect_candidate, detach_candidates, frame_evidence, projected_pixel_index,
    id_to_paths, interface_probe_mode_error, mask_mapping, probe_execution_status,
    safe_json, summarize_result, sync_capture_step, valid_depth_mask,
)


LEFT = "/World/Actors/B/left_ankle_roll_link/visual0"
RIGHT = "/World/Actors/B/right_ankle_roll_link/visual3"
VISUALS = {"left_ankle_roll_link": [LEFT], "right_ankle_roll_link": [RIGHT]}


class FakeAnnotator:
    def __init__(self, value=None, attach_error=False, read_error=False, detach_error=False):
        self.value, self.attach_error, self.read_error = value, attach_error, read_error
        self.detach_error = detach_error
        self.detached = False

    def attach(self, products):
        if self.attach_error:
            raise RuntimeError("attach unavailable")

    def get_data(self):
        if self.read_error:
            raise RuntimeError("read unavailable")
        return self.value

    def detach(self):
        self.detached = True
        if self.detach_error:
            raise RuntimeError("detach unavailable")


class FakeRegistry:
    def __init__(self, annotators):
        self.annotators = annotators

    def get_annotator(self, name, **kwargs):
        item = self.annotators.get(name)
        if item is None:
            raise LookupError(name)
        return item


class ProbeTests(unittest.TestCase):
    def test_cli_conflicts_and_isolated_branch(self):
        base = dict(geometry_interface_probe=True, vision_trial=True,
                    geometry_diagnostic=False, vision_smoke=False, semantic_skills=False,
                    deepseek=False, qwen_vision=None, assessment=False,
                    g1b1_visible_spawn=False, diagnostic_output_dir=None)
        self.assertIsNone(interface_probe_mode_error(argparse.Namespace(**base)))
        self.assertIn("requires", interface_probe_mode_error(argparse.Namespace(**(base | {"vision_trial": False}))))
        for key in ("geometry_diagnostic", "vision_smoke", "semantic_skills", "deepseek",
                    "qwen_vision", "assessment", "g1b1_visible_spawn", "diagnostic_output_dir"):
            with self.subTest(key=key):
                value = "min" if key == "qwen_vision" else Path("x") if key == "diagnostic_output_dir" else True
                self.assertIsNotNone(interface_probe_mode_error(argparse.Namespace(**(base | {key: value}))))
        self.assertIsNone(interface_probe_mode_error(argparse.Namespace(**(base | {"geometry_interface_probe": False}))))
        source = Path(__file__).with_name("vision_trial.py").read_text()
        tree = ast.parse(source)
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        branches = [n for n in ast.walk(main) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Attribute) and n.test.attr == "geometry_interface_probe"]
        self.assertTrue(any(any(isinstance(child, ast.Return) for child in n.body) for n in branches))
        self.assertLess(source.index("if args.geometry_interface_probe:", source.index("def main()")),
                        source.index("window=[capture()]"))

    def test_array_summaries_and_missing_values(self):
        mask = np.array([[0, 2], [2, 3]], dtype=np.uint32)
        item, array = summarize_result({"data": mask, "info": {"idToLabels": {"2": LEFT}}},
                                       "instance_id_segmentation", (2, 2))
        self.assertEqual(array.shape, (2, 2))
        self.assertEqual(item["unique_count"], 3)
        self.assertTrue(item["resolution_matches"])
        self.assertEqual(item["dtype"], "uint32")
        color, _ = summarize_result(np.zeros((2, 2, 4), np.uint8), "semantic_segmentation", (2, 2))
        self.assertFalse(color["integer_id_mask"])
        absent, arr = summarize_result(None, "distance_to_camera")
        self.assertIsNone(arr)
        self.assertEqual(absent["issue"], "missing_or_non_numeric_array")
        weird, arr = summarize_result({"data": object(), "info": {"unknown": object()}}, "instance_segmentation")
        self.assertIsNone(arr)
        json.dumps(safe_json(weird), allow_nan=False)

    def test_depth_stats_and_unknown_semantics(self):
        depth = np.array([[0, 1, np.nan], [np.inf, -np.inf, -2]], dtype=np.float32)
        item, _ = summarize_result(depth, "distance_to_camera", (3, 2))
        self.assertEqual((item["finite_count"], item["nan_count"], item["pos_inf_count"],
                          item["neg_inf_count"], item["zero_count"]), (3, 1, 1, 1, 1))
        self.assertEqual((item["finite_min"], item["finite_max"]), (-2.0, 1.0))
        self.assertEqual((item["negative_finite_count"], item["finite_positive_count"]), (1, 1))
        self.assertTrue(np.isposinf(depth[1, 0]))
        self.assertFalse(item["depth_semantics_confirmed"])
        self.assertIsNone(item["distance_definition"])

    def test_depth_validity_and_independent_raw_buffers(self):
        values = np.array([1.25, 0, -0.5, np.nan, np.inf, -np.inf], np.float32)
        self.assertEqual(valid_depth_mask(values).tolist(), [True, False, False, False, False, False])
        self.assertEqual([depth_invalid_reason(float(v)) for v in values],
                         [None, "zero", "negative_finite", "nan", "positive_infinity", "negative_infinity"])
        other = np.array([[0, 2]], dtype=np.float32)
        first, first_raw = summarize_result(values.reshape(2, 3), DEPTH[0], (3, 2))
        second, second_raw = summarize_result(other, DEPTH[1], (2, 1))
        self.assertEqual((first["pos_inf_count"], first["finite_positive_count"]), (1, 1))
        self.assertEqual((second["pos_inf_count"], second["finite_positive_count"]), (0, 1))
        self.assertTrue(np.isposinf(first_raw[1, 1]))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "frames").mkdir()
            item, status = collect_candidate(FakeAnnotator(values.reshape(2, 3)), DEPTH[0],
                                             VISUALS, out, "raw", 0)
            self.assertEqual(status, "ok")
            loaded = np.load(out / item["raw_file"], allow_pickle=False)
            self.assertEqual(loaded.dtype, values.dtype)
            self.assertTrue(np.isposinf(loaded[1, 1]))
            json.dumps(safe_json(item), allow_nan=False)

    def test_mapping_exact_foot_and_actor_only(self):
        mask = np.array([[0, 7, 8], [7, 8, 8]], dtype=np.uint32)
        info = {"idToLabels": {7: {"primPath": LEFT}, 8: RIGHT}}
        mapping = mask_mapping(mask, info, VISUALS)
        self.assertEqual(mapping["status"], "foot_visual_mapping_found")
        self.assertEqual(mapping["feet"]["left_ankle_roll_link"],
                         {"ids": [7], "pixels": 2, "bbox_xyxy": [0, 0, 1, 1]})
        self.assertEqual(mapping["feet"]["right_ankle_roll_link"]["pixels"], 3)
        actor = mask_mapping(mask, {"idToLabels": {7: "/World/Actors/B"}}, VISUALS)
        self.assertEqual(actor["status"], "actor_level_only")
        self.assertEqual(actor["feet"]["left_ankle_roll_link"]["pixels"], 0)
        self.assertEqual(mask_mapping(mask, {}, VISUALS)["status"], "mapping_unavailable")
        self.assertEqual(mask_mapping(np.zeros((2, 2, 4), np.uint8), info, VISUALS)["status"],
                         "not_2d_integer_id_mask")
        self.assertEqual(id_to_paths({"idToLabels": {"bad": LEFT, "7": {"class": "robot"}}})[1],
                         "mapping_unavailable")

    def test_frame_evidence_never_claims_sync(self):
        outputs = {"instance_id_segmentation": {"info": {"frameId": 5},
                                                 "array_available": True, "resolution_matches": True},
                   "distance_to_camera": {"info": {"frameId": 6},
                                          "array_available": True, "resolution_matches": True}}
        evidence = frame_evidence(outputs, [432, 768, 4])
        self.assertFalse(evidence["tokens_equal_if_present"])
        self.assertFalse(evidence["same_frame_verified"])
        self.assertFalse(evidence["rgb_token_available"])
        outputs["distance_to_camera"]["info"] = {}
        outputs["distance_to_camera"]["resolution_matches"] = False
        evidence = frame_evidence(outputs, [432, 768, 4])
        self.assertFalse(evidence["dimensions_match_rgb"])
        self.assertFalse(evidence["tokens_present_for_all"])

    def test_attach_and_read_failures_are_isolated(self):
        good = FakeAnnotator(np.array([[1, 2]], dtype=np.uint32))
        bad_attach = FakeAnnotator(attach_error=True)
        bad_read = FakeAnnotator(read_error=True)
        statuses, attached = attach_candidates(FakeRegistry({CANDIDATES[0]: good,
                                                              CANDIDATES[1]: bad_attach,
                                                              CANDIDATES[3]: bad_read}), "rp")
        self.assertEqual(statuses[CANDIDATES[0]]["attach"], "ok")
        self.assertEqual(statuses[CANDIDATES[1]]["attach"], "failed")
        self.assertTrue(bad_attach.detached)
        self.assertEqual(statuses[CANDIDATES[2]]["create"], "failed")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "frames").mkdir()
            item, state = collect_candidate(good, CANDIDATES[0], VISUALS, output, "clear", 0)
            self.assertEqual(state, "ok")
            self.assertTrue((output / item["raw_file"]).exists())
            failed, state = collect_candidate(bad_read, CANDIDATES[3], VISUALS, output, "clear", 0)
            self.assertEqual(state, "failed")
            self.assertEqual(failed["error"]["type"], "RuntimeError")
            missing, state = collect_candidate(FakeAnnotator(None), CANDIDATES[4], VISUALS, output, "clear", 0)
            self.assertEqual(state, "no_numeric_array")
            self.assertIn("frames/clear_instance_id_segmentation.npy", artifact_list(output))
            json.dumps(safe_json({"items": [item, failed, missing]}), allow_nan=False)
        self.assertEqual(set(attached), {CANDIDATES[0], CANDIDATES[3]})
        detach_candidates(attached, statuses, "/Render/Probe")
        self.assertEqual(statuses[CANDIDATES[0]]["detach"], "ok")
        self.assertEqual(statuses[CANDIDATES[0]]["detach_argument"]["kind"], "none")
        self.assertEqual(statuses[CANDIDATES[0]]["detach_argument"]["attached_render_product_path"], "/Render/Probe")
        for name in attached:
            statuses[name]["reads"] = {"clear": "ok", "occluded": "ok"}
        self.assertEqual(probe_execution_status(2, statuses), "partial")
        self.assertEqual(probe_execution_status(0, statuses), "blocked")
        self.assertEqual(probe_execution_status(2, {CANDIDATES[0]: statuses[CANDIDATES[0]]}), "complete")

    def test_detach_failure_does_not_stop_other_cleanup(self):
        first = FakeAnnotator(detach_error=True)
        second = FakeAnnotator()
        statuses = {"a": {"detach": "not_attempted"}, "b": {"detach": "not_attempted"}}
        detach_candidates({"a": first, "b": second}, statuses, "/Render/Probe")
        self.assertTrue(first.detached and second.detached)
        self.assertEqual(statuses["a"]["detach"], "failed")
        self.assertEqual(statuses["a"]["detach_error"]["type"], "RuntimeError")
        self.assertEqual(statuses["b"]["detach"], "ok")

    def test_pixel_sampling_and_candidate_statuses(self):
        foot = {"link_prim": "/World/Actors/B/left_ankle_roll_link", "visual_prim": LEFT,
                "world_m": [1, 2, 0], "projection": {"status": "in_frame", "pixel": [2.9, 1.4]},
                "ray_distance_m": 3.0, "image_plane_distance_m": 2.9}
        arrays = {"instance_id_segmentation": np.zeros((432, 768), np.uint32),
                  "distance_to_camera": np.full((432, 768), 3.0, np.float32),
                  "distance_to_image_plane": np.full((432, 768), 2.9, np.float32)}
        arrays["instance_id_segmentation"][1, 2] = 7
        info = {"idToLabels": {"7": LEFT, "8": "/World/Covers/High2/Visual"}}
        result = candidate_pixel_result(foot, arrays, info, depth_semantics_confirmed=True)
        self.assertEqual(result["candidate_pixel_status"], "visible_candidate_pixel")
        self.assertEqual(result["sampled_pixel_uv"], [2, 1])
        self.assertEqual(result["array_index_vu"], [1, 2])
        self.assertFalse(result["is_validated_sole_contact"])
        self.assertEqual(projected_pixel_index({"status": "in_frame", "pixel": [767.9, 431.9]})[0], (767, 431))
        self.assertEqual(projected_pixel_index({"status": "in_frame", "pixel": [-0.1, 1]})[1],
                         "projected_pixel_out_of_bounds")
        self.assertEqual(projected_pixel_index({"status": "behind_camera", "pixel": [2, 1]})[1],
                         "projection_not_in_frame")
        arrays["instance_id_segmentation"][1, 2] = 8
        arrays["distance_to_camera"][1, 2] = 2.0
        arrays["distance_to_image_plane"][1, 2] = 1.95
        blocked = candidate_pixel_result(foot, arrays, info, depth_semantics_confirmed=True)
        self.assertEqual(blocked["candidate_pixel_status"], "occluded_candidate_pixel")
        self.assertEqual(blocked["segmentation_prim_path"], "/World/Covers/High2/Visual")
        self.assertLess(blocked["distance_to_camera_delta_m"], -0.03)
        arrays["distance_to_camera"][1, 2] = 3.0
        self.assertEqual(candidate_pixel_result(foot, arrays, info, True)["candidate_pixel_status"], "unknown")
        arrays["distance_to_camera"][1, 2] = 2.0
        uncertain = candidate_pixel_result(foot, arrays, info)
        self.assertEqual(uncertain["candidate_pixel_status"], "unknown")
        self.assertEqual(uncertain["heuristic_unverified"], "occluded_candidate_pixel")
        self.assertEqual(uncertain["status_reason"], "depth_semantics_unconfirmed")
        for name in DEPTH:
            for bad, reason in ((np.nan, "nan"), (np.inf, "positive_infinity"),
                                (-np.inf, "negative_infinity"), (0, "zero"),
                                (-1, "negative_finite")):
                arrays[name][1, 2] = bad
                invalid = candidate_pixel_result(foot, arrays, info, True)
                self.assertEqual(invalid["candidate_pixel_status"], "unknown")
                self.assertEqual(invalid["status_reason"], "invalid_depth_sample")
                self.assertEqual(invalid["invalid_depth"][name]["reason"], reason)
                self.assertEqual(invalid["invalid_depth"][name]["raw_value"],
                                 str(bad) if not np.isfinite(bad) else bad)
                arrays[name][1, 2] = 2.0 if name == DEPTH[0] else 1.95
        arrays["distance_to_camera"][1, 2] = 2.0
        self.assertEqual(candidate_pixel_result(foot, arrays, {}, True)["candidate_pixel_status"], "unknown")
        parent_info = {"idToLabels": {"8": "/World/Actors/B/left_ankle_roll_link"}}
        self.assertEqual(candidate_pixel_result(foot, arrays, parent_info, True)["candidate_pixel_status"], "unknown")
        short = dict(arrays, distance_to_camera=np.ones((1, 1), np.float32))
        self.assertEqual(candidate_pixel_result(foot, short, info, True)["status_reason"], "array_shape_mismatch")
        json.dumps(safe_json(result), allow_nan=False)

    def test_sync_step_success_failure_and_conservative_alignment(self):
        class Orchestrator:
            def __init__(self, fail=False): self.fail, self.calls = fail, []
            def step(self, **kwargs):
                self.calls.append(kwargs)
                if self.fail: raise RuntimeError("render failed")
        record = {"sync_step_attempted": False, "sync_step_succeeded": False,
                  "sync_step_error": None, "orchestrator_step_count": 0}
        class Rep:
            def __init__(self, orchestrator): self.orchestrator = orchestrator
        good = Orchestrator()
        sync_capture_step(Rep(good), record)
        self.assertEqual(good.calls, [{"wait_for_render": True}])
        self.assertTrue(record["sync_step_attempted"] and record["sync_step_succeeded"])
        self.assertIsNone(record["sync_step_error"])
        self.assertEqual(record["orchestrator_step_count"], 1)
        bad = Orchestrator(fail=True)
        failed = dict(record, sync_step_attempted=False, sync_step_succeeded=False,
                      sync_step_error=None, orchestrator_step_count=0)
        with self.assertRaisesRegex(RuntimeError, "render failed"):
            sync_capture_step(Rep(bad), failed)
        self.assertEqual(bad.calls, [{"wait_for_render": True}])
        self.assertFalse(failed["sync_step_succeeded"])
        self.assertEqual(failed["sync_step_error"]["type"], "RuntimeError")
        self.assertEqual(failed["orchestrator_step_count"], 1)
        unavailable = dict(failed, sync_step_attempted=False, sync_step_error=None,
                           orchestrator_step_count=0)
        with self.assertRaises(AttributeError):
            sync_capture_step(object(), unavailable)
        self.assertEqual(unavailable["sync_step_error"]["type"], "AttributeError")
        self.assertFalse(unavailable["sync_step_succeeded"])

        def sample(hash_suffix):
            return {"sync_step_succeeded": True,
                    "capture_cycle_evidence": {"same_render_product": True, "single_sync_step": True,
                                               "consecutive_reads_without_update": True},
                    "frame_evidence": {"dimensions_match_rgb": True,
                                       "frame_token_status": "unavailable", "same_frame_verified": False},
                    "outputs": {name: {"array_available": True, "sha256": name + hash_suffix}
                                for name in CANDIDATES}}
        samples = [sample("a"), sample("b")]
        switch = {"rgb_changed": True}
        self.assertEqual(alignment_status(samples, switch, ("instance_id_segmentation",) + DEPTH),
                         "empirically_consistent")
        self.assertEqual(alignment_status(samples, switch, CANDIDATES), "empirically_consistent")
        self.assertFalse(any(s["frame_evidence"]["same_frame_verified"] for s in samples))
        samples[0]["capture_cycle_evidence"]["same_render_product"] = False
        self.assertEqual(alignment_status(samples, switch, CANDIDATES), "unverified")
        samples[0]["capture_cycle_evidence"]["same_render_product"] = True
        samples[0]["sync_step_succeeded"] = False
        self.assertEqual(alignment_status(samples, switch, CANDIDATES), "unverified")

    def test_tracked_link_visual_transforms_and_failure_isolation(self):
        camera_path = "/World/OnboardCamera"
        left_link = "/World/Actors/B/left_ankle_roll_link"
        right_link = "/World/Actors/B/right_ankle_roll_link"
        class Prim:
            def __init__(self, path, valid=True): self.path, self.valid = path, valid
            def IsValid(self): return self.valid
        class Camera:
            def GetPath(self): return camera_path
            def GetPrim(self): return Prim(camera_path)
        class Stage:
            def GetPrimAtPath(self, path): return Prim(path, path != right_link)
        class Cache:
            def GetLocalToWorldTransform(self, prim):
                if prim.path == left_link: raise RuntimeError("transform unavailable")
                return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0], [4.0, 3.0, 2.0, 1.0]]
        class Geom:
            XformCache = Cache
        snap = _scene_snapshot(Stage(), Camera(), None, Geom, VISUALS)
        self.assertEqual(snap["tracked_prims"][LEFT]["status"], "ok")
        self.assertEqual(snap["tracked_prims"][left_link]["status"], "failed")
        self.assertEqual(snap["tracked_prims"][right_link]["status"], "missing")
        self.assertEqual(snap["tracked_prims"][camera_path]["status"], "ok")
        json.dumps(safe_json(snap), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
