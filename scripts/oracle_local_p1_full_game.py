"""Run the frozen P1 local Oracle and its fixed baseline to the 40 s game end.

The previous 20 s paired bundle is the immutable plan. Each new Isaac task
replays the same four-action prefix, then applies the same one-bit policy or
baseline through the game's actual terminal boundary. No policy is retrained.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import zipfile

from adaptive_motion import load_config
from arena_paths import PROJECT, workspace_root
from oracle_local_p1_v1 import VERSION as PRIOR_VERSION, task_outcome
from oracle_local_state_v1 import p1_west_boundary
from oracle_pilot import DEFAULT_CONFIG, atomic_json, digest, read, resolve_config, run_task
from oracle_pilot_v21 import package_artifacts, sha256_file

VERSION = "oracle_local_p1_full_game_v1"
DEFAULT_PRIOR = workspace_root() / "experiments/oracle-local-p1-v1/oracle_local_p1_v1_bundle.zip"
DEFAULT_OUTPUT = workspace_root() / "experiments/oracle-local-p1-full-game"
KEY_MODES = ("oracle_local_eval_prefix", "oracle_local_baseline_prefix")
REPLAY_FIELDS = ("simulation_seconds", "score", "hp", "finished", "collapsed",
                 "replay_reference_digest", "skill_status", "visual_memory")


def checked_prior(path, config):
    """Verify every ZIP member and reconstruct the four frozen paired tasks."""
    path = Path(path).expanduser().resolve()
    cfg = load_config(PROJECT / "configs" / config["rules_config"])
    cfg["seed"] = config["game_seed"]
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None or any(
                name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("Prior local Oracle ZIP has duplicate, unsafe, or corrupt members")
        manifest = json.loads(archive.read("manifest.json"))
        entries = manifest["files"]
        if manifest["file_count"] != len(entries) or len(names) != len(entries) + 2:
            raise ValueError("Prior local Oracle ZIP manifest count mismatch")
        expected = {}
        for entry in entries:
            member = entry["path"]
            payload = archive.read(member)
            if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"Prior local Oracle ZIP checksum mismatch: {member}")
            expected[member] = entry["sha256"]
        if set(names) != set(expected) | {"manifest.json", "SHA256SUMS"}:
            raise ValueError("Prior local Oracle ZIP has unlisted members")
        checks = dict(line.split("  ", 1)[::-1] for line in archive.read("SHA256SUMS").decode().splitlines())
        if checks != expected:
            raise ValueError("Prior local Oracle SHA256SUMS mismatch")
        summary = json.loads(archive.read("run_summary.json"))
        paired = json.loads(archive.read("paired_validation.json"))
        policy = json.loads(archive.read("local_policy.json"))
        provenance = json.loads(archive.read("experiment_provenance.json"))
        if (summary.get("status") != "passed" or summary.get("tasks") != 8 or
                paired.get("status") != "passed" or len(paired.get("pairs", [])) != 4 or
                policy.get("version") != PRIOR_VERSION or
                policy.get("source_sha256") != provenance.get("source_sha256") or
                policy.get("feature_threshold_x") != p1_west_boundary(cfg) or
                policy.get("fallback_action_id") != config["continuation_action_id"] or
                set(policy.get("local_policy_table", {})) != {"p1_west_edge", "p1_other_exit"} or
                config["game_seed"] != provenance.get("game_seed") or
                config["discovery_synthetic_seed"] != provenance.get("synthetic_seed") or
                cfg["duration"] != 40.0):
            raise ValueError("Prior policy, matched pairs, or 40 s formal rules disagree")
        pairs = []
        seen_keys = set()
        for pair in paired["pairs"]:
            tasks = {}
            for mode, key_field in zip(KEY_MODES, ("policy_task_key", "baseline_task_key")):
                key = pair[key_field]
                if key in seen_keys:
                    raise ValueError(f"Duplicate prior task key: {key}")
                seen_keys.add(key)
                record = json.loads(archive.read(f"tasks/{key}.json"))
                task = record["task"]
                boundaries = json.loads(archive.read(f"raw/{key}/boundaries.json"))
                decisions = json.loads(archive.read(f"raw/{key}/decisions.json"))
                if (record["status"] != "passed" or record["task_key"] != key or
                        task["mode"] != mode or task["sample_id"] != pair["sample_id"] or
                        task["prefix_id"] != pair["prefix_id"] or
                        task["source_sha256"] != provenance["source_sha256"] or
                        task["max_steps"] != 10 or len(task["prefix_action_ids"]) != 4 or
                        len(task["expected_boundaries"]) != 5 or
                        task["target_state_id"] != policy["base_r2l_state_id"] or
                        task["feature_threshold_x"] != policy["feature_threshold_x"] or
                        task["continuation_action_id"] != policy["fallback_action_id"] or
                        len(boundaries) < 6 or len(decisions) < 5 or
                        decisions[4]["action_id"] != pair[("policy_action_id" if mode == KEY_MODES[0]
                                                             else "baseline_action_id")] or
                        record["result"]["evaluation_delta_score"] != pair[("policy_official_score_delta"
                                                                              if mode == KEY_MODES[0]
                                                                              else "baseline_official_score_delta")]):
                    raise ValueError(f"Prior paired task is inconsistent: {key}")
                tasks[mode] = {"task": task, "boundaries": boundaries, "decisions": decisions,
                               "result": record["result"], "task_key": key}
            policy_task, baseline_task = (tasks[mode]["task"] for mode in KEY_MODES)
            for field in ("game_seed", "synthetic_seed", "pixel_noise_std", "miss_probability",
                          "prefix_action_ids", "expected_boundaries", "target_state_id"):
                if policy_task[field] != baseline_task[field]:
                    raise ValueError(f"Prior pair does not share {field}: {pair['sample_id']}")
            if (pair["local_bin"] not in policy["local_policy_table"] or
                    policy_task["local_policy_table"] != policy["local_policy_table"] or
                    baseline_task["local_policy_table"] != {} or
                    pair["policy_action_id"] != policy["local_policy_table"][pair["local_bin"]] or
                    pair["baseline_action_id"] != policy["fallback_action_id"] or
                    pair["paired_advantage"] != round(pair["policy_official_score_delta"] -
                                                      pair["baseline_official_score_delta"], 6)):
                raise ValueError(f"Prior policy selection disagrees with evidence: {pair['sample_id']}")
            pairs.append({"pair": pair, "tasks": tasks})
        if len(seen_keys) != 8 or paired["heldout_sample_id"] != pairs[-1]["pair"]["sample_id"]:
            raise ValueError("Prior four-pair train/heldout split is incomplete")
    return {"path": path, "sha256": sha256_file(path), "summary": summary,
            "paired": paired, "policy": policy, "provenance": provenance,
            "pairs": pairs, "game_duration_seconds": cfg["duration"]}


def full_game_task(prior_task, config, prior_sha256):
    task = copy.deepcopy(prior_task)
    task.update(experiment_version=VERSION, source_sha256=config["source_sha256"],
                max_steps=20, full_game_duration_seconds=40.0,
                frozen_prior_bundle_sha256=prior_sha256)
    return task


def replay_overlap(prior_boundaries, current_boundaries, prior_decisions, current_decisions):
    """Compare the entire previously observed 0..20 s trace without wall-clock fields."""
    mismatches = []
    if len(current_boundaries) < len(prior_boundaries):
        mismatches.append({"kind": "missing_boundaries", "expected": len(prior_boundaries),
                           "actual": len(current_boundaries)})
    for step, previous in enumerate(prior_boundaries[:len(current_boundaries)]):
        current = current_boundaries[step]
        for field in REPLAY_FIELDS:
            if previous.get(field) != current.get(field):
                mismatches.append({"kind": "boundary", "step": step, "field": field})
        if previous["limited"]["state_id"] != current["limited"]["state_id"]:
            mismatches.append({"kind": "boundary", "step": step, "field": "limited.state_id"})
    if len(current_decisions) < len(prior_decisions):
        mismatches.append({"kind": "missing_decisions", "expected": len(prior_decisions),
                           "actual": len(current_decisions)})
    for step, previous in enumerate(prior_decisions[:len(current_decisions)]):
        current = current_decisions[step]
        for field in ("action_id", "source", "validator_status", "execution_status",
                      "compiled_action", "delta_score"):
            if previous.get(field) != current.get(field):
                mismatches.append({"kind": "decision", "step": step, "field": field})
    return {"matched": not mismatches, "prior_boundaries": len(prior_boundaries),
            "current_boundaries": len(current_boundaries), "mismatches": mismatches}


def full_game_outcome(record, prior_item, policy):
    result = record.get("result") or {}
    if record["status"] != "passed" or result.get("stop_reason") != "terminal" or result.get(
            "terminal_reason") not in ("time_limit", "quadruped_eliminated", "opponent_eliminated",
                                       "simultaneous_elimination"):
        raise RuntimeError(f"Full game did not reach a formal terminal: {record['task_key']} {result.get('stop_reason')}")
    item = task_outcome(record, policy)
    if item["score_at_8s"] != prior_item["boundaries"][4]["score"]:
        raise RuntimeError(f"Full game 8 s score changed: {record['task_key']}")
    raw = Path(record["raw_dir"])
    boundaries = read(raw / "boundaries.json")
    decisions = read(raw / "decisions.json")
    overlap = replay_overlap(prior_item["boundaries"], boundaries, prior_item["decisions"], decisions)
    reached_20 = next((boundary for boundary in boundaries if abs(boundary["simulation_seconds"] - 20.0) < 1e-8), None)
    item.update(official_full_game_score=result["official_score"],
                full_game_terminal_reason=result["terminal_reason"],
                full_game_terminal_seconds=result["simulation_seconds"],
                score_at_20s=reached_20["score"] if reached_20 else None,
                previous_20s_overlap=overlap)
    return item


def run_full_game(prior, config, output, max_new_tasks=8, retry_failed=False):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"version": VERSION, "source_sha256": config["source_sha256"],
                  "frozen_prior_bundle_sha256": prior["sha256"],
                  "prior_policy_source_sha256": prior["provenance"]["source_sha256"],
                  "game_seed": config["game_seed"],
                  "synthetic_seed": config["discovery_synthetic_seed"],
                  "game_duration_seconds": prior["game_duration_seconds"]}
    provenance_path = output / "experiment_provenance.json"
    if provenance_path.exists() and read(provenance_path) != provenance:
        raise ValueError("Full-game output belongs to another source, policy, or prior bundle")
    atomic_json(provenance_path, provenance)
    atomic_json(output / "frozen_local_policy.json", prior["policy"])
    atomic_json(output / "prior_paired_validation.json", prior["paired"])
    atomic_json(output / "prior_run_summary.json", prior["summary"])
    atomic_json(output / "resolved_config.json", config)
    results = []
    budget = 0
    for pair_data in prior["pairs"]:
        pair = pair_data["pair"]
        for mode in KEY_MODES:
            prior_item = pair_data["tasks"][mode]
            task = full_game_task(prior_item["task"], config, prior["sha256"])
            key = digest(task)[:20]
            task_path = output / "tasks" / f"{key}.json"
            if not task_path.exists() or (retry_failed and read(task_path)["status"] != "passed"):
                if budget >= max_new_tasks:
                    atomic_json(output / "capture_progress.json", {"stage": "incomplete",
                                "completed_tasks": len(results), "new_tasks_this_run": budget})
                    return {"version": VERSION, "status": "incomplete", "completed_tasks": len(results)}
                budget += 1
            record = run_task(output, task, retry_failed=retry_failed)
            item = full_game_outcome(record, prior_item, prior["policy"])
            results.append(item)
            atomic_json(output / "capture_progress.json", {"stage": "running",
                        "completed_tasks": len(results), "new_tasks_this_run": budget,
                        "last_task_wall_seconds": record["wall_seconds"]})
            print(f"full_game {len(results)}/8 {pair['prefix_id']} {mode} "
                  f"final_score={item['official_full_game_score']} "
                  f"terminal={item['full_game_terminal_reason']}", flush=True)
    pairs = []
    for pair_data in prior["pairs"]:
        source = pair_data["pair"]
        policy_item = next(item for item in results if item["sample_id"] == source["sample_id"] and
                           item["mode"] == KEY_MODES[0])
        baseline_item = next(item for item in results if item["sample_id"] == source["sample_id"] and
                             item["mode"] == KEY_MODES[1])
        if (policy_item["local_bin"] != baseline_item["local_bin"] or
                policy_item["score_at_8s"] != baseline_item["score_at_8s"]):
            raise RuntimeError(f"Full-game pair differs before policy selection: {source['sample_id']}")
        pairs.append({"sample_id": source["sample_id"], "prefix_id": source["prefix_id"],
                      "local_bin": policy_item["local_bin"],
                      "policy_action_id": policy_item["action_id"],
                      "baseline_action_id": baseline_item["action_id"],
                      "policy_final_official_score": policy_item["official_full_game_score"],
                      "baseline_final_official_score": baseline_item["official_full_game_score"],
                      "paired_final_score_advantage": round(policy_item["official_full_game_score"] -
                                                            baseline_item["official_full_game_score"], 6),
                      "policy_terminal_reason": policy_item["full_game_terminal_reason"],
                      "baseline_terminal_reason": baseline_item["full_game_terminal_reason"],
                      "policy_terminal_seconds": policy_item["full_game_terminal_seconds"],
                      "baseline_terminal_seconds": baseline_item["full_game_terminal_seconds"],
                      "prior_20s_advantage": source["paired_advantage"],
                      "policy_score_at_20s": policy_item["score_at_20s"],
                      "baseline_score_at_20s": baseline_item["score_at_20s"],
                      "policy_overlap_match": policy_item["previous_20s_overlap"]["matched"],
                      "baseline_overlap_match": baseline_item["previous_20s_overlap"]["matched"],
                      "policy_task_key": policy_item["task_key"],
                      "baseline_task_key": baseline_item["task_key"]})
    training = pairs[:3]
    heldout = pairs[3]
    comparison = {"version": VERSION, "status": "passed", "game_duration_seconds": 40.0,
                  "pairs": pairs, "training_mean_final_score_advantage": statistics.mean(
                      item["paired_final_score_advantage"] for item in training),
                  "heldout_final_score_advantage": heldout["paired_final_score_advantage"],
                  "heldout_sample_id": heldout["sample_id"],
                  "all_four_mean_final_score_advantage": statistics.mean(
                      item["paired_final_score_advantage"] for item in pairs),
                  "all_prior_20s_overlaps_match": all(item["previous_20s_overlap"]["matched"] for item in results)}
    atomic_json(output / "full_game_paired_validation.json", comparison)
    atomic_json(output / "full_game_rollouts.json", results)
    summary = {"version": VERSION, "status": "passed", "completed_tasks": len(results),
               "all_prior_20s_overlaps_match": comparison["all_prior_20s_overlaps_match"],
               "heldout_final_score_advantage": comparison["heldout_final_score_advantage"],
               "all_four_mean_final_score_advantage": comparison["all_four_mean_final_score_advantage"],
               "full_game_local_gain_observed": comparison["heldout_final_score_advantage"] > 0 and
               comparison["training_mean_final_score_advantage"] > 0,
               "total_task_wall_seconds": round(sum(item["wall_seconds"] for item in results), 3)}
    atomic_json(output / "run_summary.json", summary)
    atomic_json(output / "capture_progress.json", {"stage": "complete", "completed_tasks": len(results),
                "new_tasks_this_run": budget})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-bundle", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tasks", type=int, default=8)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    config = resolve_config(DEFAULT_CONFIG)
    prior = checked_prior(args.prior_bundle, config)
    output = args.output.expanduser().resolve()
    if output == prior["path"] or output in prior["path"].parents:
        parser.error("--output must be separate from the prior ZIP")
    output.mkdir(parents=True, exist_ok=True)
    invocation_path = output / "invocations.json"
    invocations = read(invocation_path) if invocation_path.exists() else []
    invocations.append({"utc_started": datetime.now(timezone.utc).isoformat(),
                        "command": ["bash", "scripts/run_oracle_pilot_ubuntu.sh", "oracle_local_full_game",
                                    *(argv if argv is not None else sys.argv[1:])]})
    atomic_json(invocation_path, invocations)
    started = time.monotonic()
    try:
        result = run_full_game(prior, config, output, args.max_new_tasks, args.retry_failed)
    except Exception as exc:
        atomic_json(output / "run_failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                  "elapsed_wall_seconds": round(time.monotonic() - started, 3)})
        result = {"version": VERSION, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if result["status"] == "passed":
        (output / "run_failure.json").unlink(missing_ok=True)
    result["invocation_wall_seconds"] = round(time.monotonic() - started, 3)
    archive = package_artifacts(output, "oracle_local_p1_full_game_bundle.zip",
                                {"version": VERSION, "source_sha256": config["source_sha256"],
                                 "frozen_prior_bundle_sha256": prior["sha256"]})
    result["bundle"] = str(archive)
    result["bundle_sha256"] = sha256_file(archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
