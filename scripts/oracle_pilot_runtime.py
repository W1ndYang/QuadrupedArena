"""One finite Isaac pilot episode. The outer collector starts a fresh process per task.

Only a limited state ID may select a policy action. Full snapshots are written under
diagnostics after selection and never passed to the policy function.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from oracle_state_v1 import (RegionCatalog, action_catalog, capture_reference,
                             encode_full, encode_limited, limited_from_public)
from semantic_skills import SemanticSkills
from synthetic_vision_mvp import observe_current_scene, smoke_observation
from vision_core import public_packet


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def reference_digest(snapshot):
    """Tolerance-level replay fingerprint; not a complete simulator checkpoint."""
    def normalize(value):
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [normalize(v) for v in value]
        return value
    data = json.dumps(normalize(asdict(snapshot)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def task_action_id(task, step, state_id):
    """Selection sees only the supplied limited ID and explicit experiment plan."""
    mode = task["mode"]
    if mode in ("preflight", "discover", "branch"):
        prefix = task.get("prefix_action_ids", [])
        if step < len(prefix):
            return prefix[step], "prefix"
        if mode == "branch" and step == task["target_step"]:
            return task["candidate_action_id"], "candidate"
        return task["continuation_action_id"], "continuation"
    if mode == "eval_policy":
        action = task.get("policy_table", {}).get(state_id)
        return (action, "table") if action is not None else (task["continuation_action_id"], "fallback")
    if mode == "eval_baseline":
        return task["continuation_action_id"], "baseline"
    raise ValueError("Unknown pilot task mode")


def classify_action_error(exc):
    message = str(exc)
    if isinstance(exc, (KeyError, TypeError)):
        return "catalog_or_interface_error"
    if isinstance(exc, ValueError):
        if any(word in message for word in ("TARGET_STALE", "No route", "not reachable", "Unsafe", "Invalid target")):
            return "not_applicable_in_state"
        return "catalog_or_interface_error"
    return "rollout_runtime_error"


def run(root, task, cfg, match, engine, rig, stage, camera, capture):
    root = Path(root)
    catalog = RegionCatalog.from_config(cfg)
    actions = {item.template_id: item for item in action_catalog(cfg)}
    if len(actions) != 44:
        raise ValueError(f"Expected complete comparison catalogue, got {len(actions)}")
    for key in ("continuation_action_id", "candidate_action_id"):
        if key in task and task[key] not in actions:
            raise ValueError(f"Task {key} is not in the actual catalogue")
    for item in task.get("prefix_action_ids", []):
        if item not in actions:
            raise ValueError("Task prefix action is not in catalogue")
    for item in task.get("policy_table", {}).values():
        if item not in actions:
            raise ValueError("Policy action is not in catalogue")
    if task["mode"] == "branch" and len(task.get("prefix_action_ids", [])) != task["target_step"]:
        raise ValueError("Branch prefix length must equal target_step")
    if task["mode"] == "branch" and task["horizon_seconds"] % 2:
        raise ValueError("Horizon must be an integer number of 2 s decisions")

    started = time.monotonic()
    frames = []
    samples = []
    boundaries = []
    decisions = []
    prev_action = None

    def sample(frame):
        sample_id = len(samples)
        packet = public_packet(sample_id, match.states[-1], [frame], cfg["start_b"])
        packet["skills"] = engine.public(match.t)
        with Image.open(root / frame["file"]) as image:
            size = image.size
        synthetic = smoke_observation(packet, frame, lambda p, f: observe_current_scene(
            stage, camera, p, f, cfg, size, seed=task["synthetic_seed"] + sample_id,
            pixel_noise_std=task["pixel_noise_std"], miss_probability=task["miss_probability"]))
        if str(synthetic.get("reason", "")).startswith("adapter_failed:"):
            raise RuntimeError(synthetic["reason"])
        sighting = synthetic.get("sighting")
        ingest = "not_attempted_no_sighting"
        if sighting is not None:
            try:
                engine.observe_sighting(sighting, packet)
                ingest = "accepted"
            except Exception as exc:
                ingest = f"rejected:{type(exc).__name__}:{exc}"
        record = {"sample_id": sample_id, "frame_file": frame["file"],
                  "simulation_seconds": frame["simulation_seconds"],
                  "sighting": sighting, "ingest_status": ingest,
                  "memory_after": engine.public(match.t)["visual_memory"],
                  "synthetic_file": f"diagnostics/synthetic_{sample_id:03d}.json"}
        dump(root / record["synthetic_file"], synthetic)
        samples.append(record)
        return record

    window = [capture()]
    frames.append(window[-1])
    window_samples = [sample(window[-1])]
    stop_reason = None
    while True:
        step = len(decisions)
        packet = public_packet(step, match.states[-1], window, cfg["start_b"])
        packet["skills"] = engine.public(match.t)
        accepted = tuple(s["sighting"] if s["ingest_status"] == "accepted" else None for s in window_samples)
        limited = encode_limited(limited_from_public(packet, accepted, cfg, prev_action), cfg, catalog)
        reference = capture_reference(match, engine, rig)
        full = encode_full(reference, cfg, catalog)
        boundary = {"observation_id": step, "simulation_seconds": match.t,
                    "frame_files": [f["file"] for f in window],
                    "frame_times": [f["simulation_seconds"] for f in window],
                    "limited": {"status": limited.status, "reason": limited.reason,
                                "state_id": limited.state_id, "fields": dict(limited.fields)},
                    "full_diagnostic_file": f"diagnostics/reference_{step:03d}.json",
                    "reference_digest_1e6": reference_digest(reference),
                    "score": match.ref.score, "hp": list(match.ref.hp),
                    "skill_status": copy.deepcopy(engine.status), "finished": match.ref.finished}
        dump(root / boundary["full_diagnostic_file"],
             {"snapshot": asdict(reference), "full": {"status": full.status,
              "reason": full.reason, "state_id": full.state_id, "fields": dict(full.fields)}})
        boundaries.append(boundary)
        dump(root / "boundaries.json", boundaries)
        if task["mode"] == "branch" and step == task["target_step"]:
            if limited.state_id != task["expected_state_id"]:
                stop_reason = "replay_mismatch"
                break
            if task.get("expected_full_id") and full.state_id != task["expected_full_id"]:
                stop_reason = "full_replay_mismatch"
                break
            if task.get("expected_reference_digest") and boundary["reference_digest_1e6"] != task["expected_reference_digest"]:
                stop_reason = "reference_replay_mismatch"
                break
        if match.ref.finished:
            stop_reason = "terminal"
            break
        if task["mode"] in ("preflight", "discover") and step >= task["max_steps"]:
            stop_reason = "planned_boundary"
            break
        if task["mode"] == "branch" and match.t >= task["target_step"] * 2 + task["horizon_seconds"] - 1e-8:
            stop_reason = "horizon"
            break
        if task["mode"].startswith("eval_") and match.t >= cfg["duration"] - 1e-8:
            stop_reason = "duration"
            break
        if limited.status != "valid":
            stop_reason = "limited_state_unavailable"
            break

        action_id, source = task_action_id(task, step, limited.state_id)
        template = actions[action_id]
        request = template.request(step)
        decision = {"observation_id": step, "simulation_seconds": match.t,
                    "state_id": limited.state_id, "action_id": action_id,
                    "skill": template.skill, "params": dict(template.params),
                    "source": source, "score_before": match.ref.score,
                    "wall_started_seconds": time.monotonic() - started,
                    "validator_status": "not_attempted", "execution_status": "not_started"}
        decisions.append(decision)
        dump(root / "decisions.json", decisions)
        try:
            engine.validate(request, packet)
            decision["validator_status"] = "accepted"
            action, route, camera_command, skill_status = engine.commit(request, packet)
            match.compile(action)
            previous = np.asarray(packet["self_state"]["xy"])
            for p in route:
                if not engine.base.segment(previous, p):
                    raise ValueError("Unsafe compiled Skill route")
                previous = p
            match.apply(action, route)
            decision["compiled_action"] = action
            decision["compiled_skill"] = skill_status
            decision["route_count"] = len(route)
            decision["execution_status"] = "started"
        except Exception as exc:
            decision["error_class"] = classify_action_error(exc)
            decision["error"] = f"{type(exc).__name__}: {exc}"
            decision["traceback"] = traceback.format_exc()
            stop_reason = "action_error"
            dump(root / "decisions.json", decisions)
            break

        prev_action = action_id
        window = []
        window_samples = []
        for micro in range(1, 101):
            own = public_packet(step, match.states[-1], [], cfg["start_b"])
            own["skills"] = engine.public(match.t)
            interruption = engine.tick(own)
            if interruption:
                match.apply(*interruption)
                decision.setdefault("interruptions", []).append({"time": match.t, "state": engine.status.get("state")})
            else:
                for key in ("attack", "facing"):
                    match.active[key] = engine.action[key]
            camera_command = engine.camera(own)
            match.advance(cfg["dt"])
            rig.step(camera_command, micro * cfg["dt"], cfg["dt"])
            if abs(match.t - cfg["collapse"]["time"]) < 1e-8 and not match.ref.finished:
                action, route, camera_command, _ = engine.plan(public_packet(
                    step, match.states[-1], [], cfg["start_b"]))
                match.apply(action, route)
            if micro % 25 == 0 or match.ref.finished:
                frame = capture()
                frames.append(frame)
                window.append(frame)
                window_samples.append(sample(frame))
            if match.ref.finished:
                break
        decision["microsteps"] = micro
        decision["score_after"] = match.ref.score
        decision["delta_score"] = round(match.ref.score - decision["score_before"], 6)
        decision["finished"] = match.ref.finished
        decision["skill_status_after"] = copy.deepcopy(engine.status)
        decision["execution_status"] = ("interrupted" if decision.get("interruptions") else
                                        "completed" if engine.status.get("state") in ("HOLDING_POINT", "AT_DESTINATION") else
                                        "accepted_but_blocked" if engine.status.get("state") in
                                        ("UNSAFE_ROUTE", "NO_ROUTE", "NO_SAFE_COVER", "TARGET_STALE") else "started")
        decision["wall_elapsed_seconds"] = time.monotonic() - started - decision["wall_started_seconds"]
        dump(root / "decisions.json", decisions)

    result = {"status": "passed" if stop_reason in ("terminal", "horizon", "planned_boundary", "duration") else "partial",
              "mode": "oracle_coarse_pilot_" + task["mode"], "stop_reason": stop_reason,
              "simulation_seconds": match.t, "official_score": match.ref.score,
              "official_delta_score": round(match.ref.score -
                  (boundaries[task["target_step"]]["score"] if task["mode"] == "branch" and
                   len(boundaries) > task["target_step"] else boundaries[0]["score"]), 6),
              "opponent_official_score": None, "official_score_difference": None,
              "official_winner": None, "terminal_reason": match.ref.finished,
              "decisions": len(decisions), "boundaries": len(boundaries),
              "wall_seconds": time.monotonic() - started,
              "same_frame_verified": False, "render_sync_quality": "blocked",
              "sighting_source": "simulator_geometry_truth",
              "state_schema_version": limited.version,
              "action_space_version": "comparison_v1_candidate", "action_catalog_count": len(actions)}
    dump(root / "pilot_result.json", result)
    dump(root / "result.json", result)
    dump(root / "frame_index.json", frames)
    dump(root / "synthetic_samples.json", samples)
    return result
