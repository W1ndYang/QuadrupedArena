"""CPU contracts for Pilot replay, action isolation, analysis and packaging."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from adaptive_motion import load_config
from oracle_pilot import analyze_actions, bootstrap_ci, discover, main, package, pilot_actions, resolve_config
from oracle_pilot_discovery import (DISCOVERY_SCHEMA, analyze_records, build_prefix_library,
                                    boundary_evidence, reference_diff, reference_summaries)
from oracle_pilot_runtime import run, task_action_id
from oracle_state_v1 import action_catalog, capture_reference
from semantic_skills import SemanticSkills
from vision_core import CameraRig, VisionMatch


class PilotTests(unittest.TestCase):
    def test_config_uses_actual_44_templates(self):
        config = resolve_config(Path("configs/oracle_coarse_pilot.json"))
        self.assertEqual(len(config["action_catalog"]), 44)
        self.assertEqual(len({x["action_id"] for x in config["action_catalog"]}), 44)
        self.assertEqual(config["continuation_action_id"],
                         next(x["action_id"] for x in config["action_catalog"] if
                              x["skill"] == "CAPTURE_POINT" and x["params"] ==
                              {"target": "active", "risk": "fast"}))
        self.assertFalse(set(config["training_synthetic_seeds"]) & set(config["test_synthetic_seeds"]))
        library = config["prefix_library"]
        self.assertEqual(library["schema"], DISCOVERY_SCHEMA)
        self.assertEqual(len(library["prefixes"]), 24)
        self.assertEqual(len({tuple(x["prefix_action_ids"]) for x in library["prefixes"]}), 24)
        self.assertEqual(len({x["prefix_action_ids"][-1] for x in library["prefixes"]}), 1)
        self.assertTrue(all(len(x["prefix_action_ids"]) == 4 for x in library["prefixes"]))

    def test_policy_selection_only_uses_limited_id(self):
        task = {"mode": "eval_policy", "policy_table": {"limited_A": "action_A"},
                "continuation_action_id": "fallback"}
        self.assertEqual(task_action_id(task, 3, "limited_A"), ("action_A", "table"))
        self.assertEqual(task_action_id(task, 3, "other"), ("fallback", "fallback"))
        self.assertIsNone(bootstrap_ci([1]))
        self.assertEqual(bootstrap_ci([2, 2, 2]), [2, 2])

    def test_fresh_cpu_prefix_replay_matches_limited_reference_and_score(self):
        config = resolve_config(Path("configs/oracle_coarse_pilot.json"))
        cfg = load_config("configs/trial_rotating.json")
        cfg["seed"] = config["game_seed"]
        task = {"mode": "preflight", "game_seed": cfg["seed"],
                "synthetic_seed": config["training_synthetic_seeds"][0],
                "pixel_noise_std": 1.0, "miss_probability": 0.0,
                "continuation_action_id": config["continuation_action_id"], "max_steps": 1}

        def one(root):
            match, engine, rig = VisionMatch(cfg), SemanticSkills(cfg), CameraRig()
            image = Image.new("RGB", (768, 432), "black")
            count = 0
            def capture():
                nonlocal count
                state = match.states[-1]
                name = f"frames/{count:03d}_{state['t']:.2f}.png"
                (root / "frames").mkdir(parents=True, exist_ok=True)
                image.save(root / name)
                count += 1
                return {"file": name, "simulation_seconds": state["t"],
                        "own_xy": state["xy"][0], "body_yaw_degrees": 0.,
                        "camera_pan_degrees": rig.pan, "camera_world_yaw_degrees": 0.,
                        "camera_eye": [state["xy"][0][0], state["xy"][0][1], .6],
                        "horizontal_fov_degrees": 90.}
            with patch("oracle_pilot_runtime.observe_current_scene", return_value={
                "status": "not_emitted", "reason": "no_visible_foot", "sighting": None,
                "candidates": [], "source": "simulator_geometry_truth"}):
                result = run(root, task, cfg, match, engine, rig, None, None, capture)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["boundaries"], 2)
            self.assertEqual(result["decisions"], 1)
            return json.loads((root / "boundaries.json").read_text())

        with tempfile.TemporaryDirectory() as tmp:
            a = one(Path(tmp) / "first")
            b = one(Path(tmp) / "second")
            self.assertEqual([(x["limited"]["state_id"], x["reference_digest_1e6"], x["score"])
                              for x in a],
                             [(x["limited"]["state_id"], x["reference_digest_1e6"], x["score"])
                              for x in b])

    def test_branch_checks_every_prefix_boundary_before_candidate(self):
        config = resolve_config(Path("configs/oracle_coarse_pilot.json"))
        cfg = load_config("configs/trial_rotating.json")
        cfg["seed"] = config["game_seed"]
        action_id = config["continuation_action_id"]

        def one(root, task):
            match, engine, rig = VisionMatch(cfg), SemanticSkills(cfg), CameraRig()
            image = Image.new("RGB", (768, 432), "black")
            count = 0
            def capture():
                nonlocal count
                state = match.states[-1]
                name = f"frames/{count:03d}_{state['t']:.2f}.png"
                (root / "frames").mkdir(parents=True, exist_ok=True)
                image.save(root / name)
                count += 1
                return {"file": name, "simulation_seconds": state["t"],
                        "own_xy": state["xy"][0], "body_yaw_degrees": 0.,
                        "camera_pan_degrees": rig.pan, "camera_world_yaw_degrees": 0.,
                        "camera_eye": [state["xy"][0][0], state["xy"][0][1], .6],
                        "horizontal_fov_degrees": 90.}
            with patch("oracle_pilot_runtime.observe_current_scene", return_value={
                "status": "not_emitted", "reason": "no_visible_foot", "sighting": None,
                "candidates": [], "source": "simulator_geometry_truth"}):
                result = run(root, task, cfg, match, engine, rig, None, None, capture)
            return result, json.loads((root / "boundaries.json").read_text())

        base = {"game_seed": cfg["seed"], "synthetic_seed": config["discovery_synthetic_seed"],
                "pixel_noise_std": 1.0, "miss_probability": 0.0,
                "continuation_action_id": action_id}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, original = one(root / "source", {**base, "mode": "discover", "max_steps": 1,
                                                "prefix_action_ids": [action_id]})
            expected = [boundary_evidence(x) for x in original]
            task = {**base, "mode": "branch", "target_step": 1, "prefix_action_ids": [action_id],
                    "candidate_action_id": action_id, "horizon_seconds": 2,
                    "expected_state_id": original[1]["limited"]["state_id"],
                    "expected_reference_digest": original[1]["replay_reference_digest"],
                    "expected_boundaries": expected}
            good, _ = one(root / "good", task)
            self.assertEqual(good["status"], "passed")
            bad = json.loads(json.dumps(task))
            bad["expected_boundaries"][0]["score"] = 999
            failed, _ = one(root / "bad", bad)
            self.assertEqual(failed["stop_reason"], "replay_mismatch")
            self.assertEqual(failed["decisions"], 0)
            self.assertEqual(failed["replay_mismatch_detail"]["boundary_step"], 0)

    def test_common_action_and_archive_are_computed_from_completed_branches(self):
        config = resolve_config(Path("configs/oracle_coarse_pilot.json"))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "tasks").mkdir()
            for sample, values in {"x1": [("A", 1.0), ("B", 2.0)],
                                   "x2": [("A", 3.0), ("B", 1.0)]}.items():
                for action, score in values:
                    key = sample + action
                    raw = output / "raw" / key
                    raw.mkdir(parents=True)
                    (raw / "decisions.json").write_text(json.dumps([{
                        "validator_status": "accepted", "execution_status": "completed"}]))
                    (output / "tasks" / f"{key}.json").write_text(json.dumps({
                        "task_key": key, "raw_dir": str(raw), "status": "passed",
                        "task": {"mode": "branch", "source_sha256": config["source_sha256"],
                                 "sample_id": sample, "target_step": 0,
                                 "expected_state_id": "state_s", "candidate_action_id": action},
                        "result": {"official_delta_score": score}}))
            summary = analyze_actions(output, config)
            self.assertEqual(summary[0]["common_action"], "A")
            self.assertEqual(summary[0]["mean_regret"], .5)
            self.assertFalse(json.loads((output / "analysis" / "policy.json").read_text())["table"])
            archive = package(output)
            self.assertTrue(archive.exists())
            self.assertGreater(json.loads((output / "manifest.json").read_text())["file_count"], 0)

    def test_discovery_distinguishes_physical_and_camera_only_differences(self):
        config = resolve_config(Path("configs/oracle_coarse_pilot.json"))
        cfg = load_config("configs/trial_rotating.json")
        cfg["seed"] = config["game_seed"]
        library = build_prefix_library(cfg, action_catalog(cfg))
        match, engine, rig = VisionMatch(cfg), SemanticSkills(cfg), CameraRig()
        from dataclasses import asdict
        base = asdict(capture_reference(match, engine, rig))
        physical = json.loads(json.dumps(base))
        physical["a"]["xy"][0] += 0.01
        camera = json.loads(json.dumps(base))
        camera["camera_pan"] = 10.0
        self.assertTrue(reference_diff(base, physical)["qualifies"])
        self.assertFalse(reference_diff(base, camera)["qualifies"])
        self.assertTrue(reference_diff(base, camera)["observation_memory_only"])
        self.assertNotEqual(reference_summaries(base)["dynamics_digest"],
                            reference_summaries(physical)["dynamics_digest"])
        self.assertEqual(reference_summaries(base)["dynamics_digest"],
                         reference_summaries(camera)["dynamics_digest"])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            records = []
            for index, snapshot in enumerate((base, physical, camera)):
                plan = library["prefixes"][index]
                key = f"task{index}"
                raw = output / "raw" / key
                (raw / "diagnostics").mkdir(parents=True)
                boundaries = []
                for step in range(5):
                    (raw / "diagnostics" / f"reference_{step:03d}.json").write_text(json.dumps({"snapshot": snapshot}))
                    boundaries.append({"limited": {"status": "valid", "state_id": f"state_{step}",
                                                  "canonical_state": f"canonical_{step}",
                                                  "fields": {"decision_step": step, "A.region_id": "open"}},
                                       "full_diagnostic_file": f"diagnostics/reference_{step:03d}.json",
                                       "reference_digest_1e6": f"ref_{index}",
                                       "replay_reference_digest": f"ref_{index}",
                                       "score": 0.0, "skill_status": {"state": "IDLE"},
                                       "visual_memory": {}, "finished": None})
                (raw / "boundaries.json").write_text(json.dumps(boundaries))
                (raw / "decisions.json").write_text(json.dumps([
                    {"action_id": aid, "validator_status": "accepted"}
                    for aid in plan["prefix_action_ids"]]))
                records.append({"task_key": key, "task": {"mode": "discover", "source_sha256": config["source_sha256"],
                                "prefix_id": plan["prefix_id"], "prefix_library_hash": library["hash"],
                                "prefix_action_ids": plan["prefix_action_ids"],
                                "prefix_actions": plan["actions"],
                                "game_seed": config["game_seed"],
                                "synthetic_seed": config["discovery_synthetic_seed"]},
                                "raw_dir": str(raw), "status": "passed", "wall_seconds": 1.0,
                                "result": {"stop_reason": "planned_boundary"}})
            samples, near = analyze_records(records, library, {**config, "discovery_batch": 1})
            self.assertEqual(samples["status"], "ready")
            self.assertEqual(samples["qualifying_group_count"], 5)
            self.assertEqual(len(samples["groups"][0]["samples"]), 2)
            self.assertEqual(samples["groups"][0]["available_distinct_full_states"], 2)
            self.assertEqual(near["exact_qualifying_groups"], 5)

    def test_old_samples_are_rejected_before_discovery_or_action_matrix(self):
        config = resolve_config(Path("configs/oracle_coarse_pilot.json"))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "samples.json").write_text(json.dumps({"status": "reduced_incomplete", "groups": []}))
            with self.assertRaisesRegex(ValueError, "old discovery data"):
                main(["discover", "--output", str(output)])
            (output / "preflight.json").write_text(json.dumps({"status": "passed", "source_sha256": config["source_sha256"]}))
            (output / "smoke.json").write_text(json.dumps({"status": "passed", "source_sha256": config["source_sha256"]}))
            with self.assertRaisesRegex(ValueError, "old samples"):
                pilot_actions(output, config, 1, False)
            (output / "samples.json").write_text(json.dumps({
                "discovery_schema": DISCOVERY_SCHEMA,
                "source_sha256": config["source_sha256"],
                "prefix_library_hash": config["prefix_library"]["hash"], "groups": []}))
            with self.assertRaisesRegex(ValueError, "No qualified exact-ID sample group"):
                pilot_actions(output, config, 44, False)

    def test_discover_schedules_distinct_fixed_seed_prefixes_without_branches(self):
        config = resolve_config(Path("configs/oracle_coarse_pilot.json"))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "preflight.json").write_text(json.dumps({"status": "passed", "source_sha256": config["source_sha256"]}))
            (output / "smoke.json").write_text(json.dumps({"status": "passed", "source_sha256": config["source_sha256"]}))
            called = []
            def fake_run_task(root, task, **_):
                called.append(task)
                return {"task_key": str(len(called)), "task": task, "raw_dir": str(output / "missing"),
                        "status": "partial", "result": {"stop_reason": "action_error"}, "wall_seconds": 0.0}
            with patch("oracle_pilot.run_task", side_effect=fake_run_task):
                result = discover(output, config, max_runs=2)
            self.assertEqual(result["attempted_prefixes"], 2)
            self.assertEqual(result["failed_prefixes"], 2)
            self.assertEqual({x["mode"] for x in called}, {"discover"})
            self.assertEqual(len({x["prefix_id"] for x in called}), 2)
            self.assertEqual({x["synthetic_seed"] for x in called}, {config["discovery_synthetic_seed"]})
            self.assertTrue(all(len(x["prefix_action_ids"]) == 4 for x in called))
            self.assertTrue((output / "near_miss_report.md").exists())
            self.assertTrue((output / "prefix_library.json").exists())

    def test_four_window_formal_prefix_produces_cpu_discovery_evidence(self):
        config = resolve_config(Path("configs/oracle_coarse_pilot.json"))
        cfg = load_config("configs/trial_rotating.json")
        cfg["seed"] = config["game_seed"]
        plan = config["prefix_library"]["prefixes"][0]
        task = {"mode": "discover", "game_seed": config["game_seed"],
                "source_sha256": config["source_sha256"],
                "synthetic_seed": config["discovery_synthetic_seed"],
                "pixel_noise_std": 1.0, "miss_probability": 0.0,
                "continuation_action_id": config["continuation_action_id"],
                "max_steps": 4, "prefix_id": plan["prefix_id"],
                "prefix_library_hash": config["prefix_library"]["hash"],
                "prefix_action_ids": plan["prefix_action_ids"],
                "prefix_actions": plan["actions"]}
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            match, engine, rig = VisionMatch(cfg), SemanticSkills(cfg), CameraRig()
            image = Image.new("RGB", (32, 32), "black")
            counter = 0
            def capture():
                nonlocal counter
                state = match.states[-1]
                filename = f"frames/{counter:03d}.png"
                (raw / "frames").mkdir(parents=True, exist_ok=True)
                image.save(raw / filename)
                counter += 1
                return {"file": filename, "simulation_seconds": state["t"],
                        "own_xy": state["xy"][0], "body_yaw_degrees": 0.,
                        "camera_pan_degrees": rig.pan, "camera_world_yaw_degrees": 0.,
                        "camera_eye": [state["xy"][0][0], state["xy"][0][1], .6],
                        "horizontal_fov_degrees": 90.}
            with patch("oracle_pilot_runtime.observe_current_scene", return_value={
                "status": "not_emitted", "reason": "no_visible_foot", "sighting": None,
                "candidates": [], "source": "simulator_geometry_truth"}):
                result = run(raw, task, cfg, match, engine, rig, None, None, capture)
            self.assertEqual(result["stop_reason"], "planned_boundary")
            self.assertEqual(result["decisions"], 4)
            decisions = json.loads((raw / "decisions.json").read_text())
            self.assertEqual([d["action_id"] for d in decisions], plan["prefix_action_ids"])
            boundaries = json.loads((raw / "boundaries.json").read_text())
            for boundary in boundaries:
                limited = boundary["limited"]
                self.assertEqual(limited["state_id"].split(":")[-1],
                                 hashlib.sha256(limited["canonical_state"].encode()).hexdigest())
            record = {"task_key": "cpu_fixture", "task": task, "raw_dir": str(raw),
                      "status": "passed", "wall_seconds": 0.0, "result": result}
            samples, _ = analyze_records([record], config["prefix_library"],
                                         {**config, "discovery_batch": 1})
            self.assertEqual(samples["completed_prefixes"], 1)
            self.assertEqual(samples["boundary_visits"], 5)
            self.assertEqual(samples["qualifying_group_count"], 0)


if __name__ == "__main__":
    unittest.main()
