"""Isaac-free checks for the formal scene's read-only interface probe."""

import argparse
import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from geometry_interface_probe import (
    CANDIDATES, artifact_list, attach_candidates, collect_candidate, frame_evidence,
    id_to_paths, interface_probe_mode_error, mask_mapping, probe_execution_status,
    safe_json, summarize_result,
)


LEFT = "/World/Actors/B/left_ankle_roll_link/visual0"
RIGHT = "/World/Actors/B/right_ankle_roll_link/visual3"
VISUALS = {"left_ankle_roll_link": [LEFT], "right_ankle_roll_link": [RIGHT]}


class FakeAnnotator:
    def __init__(self, value=None, attach_error=False, read_error=False):
        self.value, self.attach_error, self.read_error = value, attach_error, read_error
        self.detached = False

    def attach(self, products):
        if self.attach_error:
            raise RuntimeError("attach unavailable")

    def get_data(self):
        if self.read_error:
            raise RuntimeError("read unavailable")
        return self.value

    def detach(self, products):
        self.detached = True


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
        depth = np.array([[0, 1, np.nan], [np.inf, -np.inf, 3]], dtype=np.float32)
        item, _ = summarize_result(depth, "distance_to_camera", (3, 2))
        self.assertEqual((item["finite_count"], item["nan_count"], item["pos_inf_count"],
                          item["neg_inf_count"], item["zero_count"]), (3, 1, 1, 1, 1))
        self.assertEqual((item["finite_min"], item["finite_max"]), (0.0, 3.0))
        self.assertFalse(item["depth_semantics_confirmed"])
        self.assertIsNone(item["distance_definition"])

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
        for name in attached:
            statuses[name]["reads"] = {"clear": "ok", "occluded": "ok"}
            statuses[name]["detach"] = "ok"
        self.assertEqual(probe_execution_status(2, statuses), "partial")
        self.assertEqual(probe_execution_status(0, statuses), "blocked")
        self.assertEqual(probe_execution_status(2, {CANDIDATES[0]: statuses[CANDIDATES[0]]}), "complete")


if __name__ == "__main__":
    unittest.main()
