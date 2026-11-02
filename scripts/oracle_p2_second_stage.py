"""Evaluate one formal action at the verified 20 s P2 state, then finish game.

The previous full-game local-Oracle bundle freezes the entire 0..20 s path.
Training uses apv2_01 only; apv2_02 is a held-out prefix. Candidate actions
are evaluated on final official score, never on millisecond survival ties.
"""
from __future__ import annotations

import argparse
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
from oracle_local_p1_v1 import formal_action
from oracle_pilot import DEFAULT_CONFIG, atomic_json, digest, read, resolve_config, run_task, task_base
from oracle_pilot_discovery import boundary_evidence
from oracle_pilot_runtime import strategy_action_executable
from oracle_pilot_v21 import package_artifacts, sha256_file

VERSION = "oracle_p2_second_stage_v1"
DEFAULT_PRIOR = workspace_root() / "experiments/oracle-local-p1-full-game/oracle_local_p1_full_game_bundle.zip"
DEFAULT_OUTPUT = workspace_root() / "experiments/oracle-p2-second-stage"
TARGET_STEP = 10
MAX_STEPS = 20
REQUESTS = (
    ("CAPTURE_POINT", {"target": "active", "risk": "fast"}),
    ("TAKE_COVER", {"risk": "safe"}),
    ("TAKE_COVER", {"risk": "fast"}),
    ("TAKE_COVER", {"cover": "High1", "risk": "safe"}),
    ("TAKE_COVER", {"cover": "High3", "risk": "safe"}),
    ("OBSERVE", {"direction": "scan", "risk": "safe"}),
    ("OBSERVE", {"direction": "threat", "risk": "safe"}),
)


def checked_full_game_bundle(path, config):
    path = Path(path).expanduser().resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None or any(
                name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("Full-game bundle has duplicate, unsafe, or corrupt members")
        manifest = json.loads(archive.read("manifest.json"))
        expected = {}
        for item in manifest["files"]:
            data = archive.read(item["path"])
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Full-game bundle checksum mismatch: {item['path']}")
            expected[item["path"]] = item["sha256"]
        sums = dict(line.split("  ", 1)[::-1] for line in archive.read("SHA256SUMS").decode().splitlines())
        if (manifest["file_count"] != len(expected) or sums != expected or
                set(names) != set(expected) | {"manifest.json", "SHA256SUMS"}):
            raise ValueError("Full-game bundle manifest disagrees with ZIP members")
        summary = json.loads(archive.read("run_summary.json"))
        paired = json.loads(archive.read("full_game_paired_validation.json"))
        policy = json.loads(archive.read("frozen_local_policy.json"))
        provenance = json.loads(archive.read("experiment_provenance.json"))
        if (summary.get("status") != "passed" or summary.get("completed_tasks") != 8 or
                not summary.get("all_prior_20s_overlaps_match") or paired.get("status") != "passed" or
                not paired.get("all_prior_20s_overlaps_match") or len(paired.get("pairs", [])) != 4 or
                paired["game_duration_seconds"] != 40.0 or
                provenance["source_sha256"] != manifest["source_sha256"] or
                provenance["game_seed"] != config["game_seed"] or
                provenance["synthetic_seed"] != config["discovery_synthetic_seed"] or
                policy["fallback_action_id"] != config["continuation_action_id"]):
            raise ValueError("Full-game bundle is not the verified frozen local policy run")
        west = [item for item in paired["pairs"] if item["local_bin"] == "p1_west_edge"]
        if len(west) != 2 or west[0]["prefix_id"].split("_")[1] != "01" or west[1]["prefix_id"].split("_")[1] != "02":
            raise ValueError("Expected the training and held-out west-edge prefixes")
        samples = []
        for pair in west:
            key = pair["policy_task_key"]
            record = json.loads(archive.read(f"tasks/{key}.json"))
            boundaries = json.loads(archive.read(f"raw/{key}/boundaries.json"))
            decisions = json.loads(archive.read(f"raw/{key}/decisions.json"))
            if (record["status"] != "passed" or record["task_key"] != key or
                    record["task"]["mode"] != "oracle_local_eval_prefix" or
                    record["task"]["sample_id"] != pair["sample_id"] or
                    record["task"]["prefix_id"] != pair["prefix_id"] or
                    record["task"]["source_sha256"] != provenance["source_sha256"] or
                    record["task"]["game_seed"] != config["game_seed"] or
                    record["task"]["synthetic_seed"] != config["discovery_synthetic_seed"] or
                    record["task"]["pixel_noise_std"] != config["pixel_noise_std"] or
                    record["task"]["miss_probability"] != config["miss_probability"] or
                    record["task"]["continuation_action_id"] != config["continuation_action_id"] or
                    len(boundaries) < 12 or len(decisions) < TARGET_STEP or
                    abs(boundaries[TARGET_STEP]["simulation_seconds"] - 20.0) > 1e-8 or
                    boundaries[TARGET_STEP]["score"] != pair["policy_score_at_20s"] or
                    record["result"]["official_score"] != pair["policy_final_official_score"] or
                    boundaries[TARGET_STEP]["limited"]["status"] != "valid" or
                    boundaries[TARGET_STEP]["limited"]["fields"]["A.region_id"] != "target_P2" or
                    boundaries[TARGET_STEP]["limited"]["fields"]["A.hp"] != 1 or
                    not all(strategy_action_executable(decision) for decision in decisions[:TARGET_STEP])):
                raise ValueError(f"West-edge source run is incomplete: {key}")
            samples.append({"sample_id": pair["sample_id"], "prefix_id": pair["prefix_id"],
                            "synthetic_seed": record["task"]["synthetic_seed"],
                            "prefix_action_ids": [decision["action_id"] for decision in decisions[:TARGET_STEP]],
                            "expected_boundaries": [{**boundary_evidence(boundary),
                                                     "collapsed": boundary["collapsed"]}
                                                    for boundary in boundaries[:TARGET_STEP + 1]],
                            "target_state_id": boundaries[TARGET_STEP]["limited"]["state_id"],
                            "score_at_20s": boundaries[TARGET_STEP]["score"],
                            "prior_final_score": pair["policy_final_official_score"],
                            "prior_task_key": key})
        if samples[0]["target_state_id"] != samples[1]["target_state_id"]:
            raise ValueError("Training and held-out P2 states are not the same R2-L state")
    return {"path": path, "sha256": sha256_file(path), "summary": summary,
            "paired": paired, "policy": policy, "provenance": provenance,
            "training": samples[0], "heldout": samples[1],
            "target_state_id": samples[0]["target_state_id"]}


def candidate_catalog(config):
    candidates = [formal_action(config, skill, params) for skill, params in REQUESTS]
    if len(candidates) != len(set(candidates)) or candidates[0] != config["continuation_action_id"]:
        raise ValueError("The P2 shortlist must contain unique formal actions and baseline first")
    return candidates


def branch_task(config, prior, sample, action_id):
    task = task_base(config, "oracle_second_stage", sample["synthetic_seed"])
    task.update(experiment_version=VERSION, sample_id=sample["sample_id"],
                prefix_id=sample["prefix_id"], prefix_action_ids=sample["prefix_action_ids"],
                expected_boundaries=sample["expected_boundaries"], target_step=TARGET_STEP,
                expected_state_id=sample["target_state_id"], candidate_action_id=action_id,
                max_steps=MAX_STEPS, full_game_duration_seconds=40.0,
                frozen_full_game_bundle_sha256=prior["sha256"])
    return task


def branch_outcome(record, sample):
    raw = Path(record["raw_dir"])
    result = record.get("result") or {}
    boundaries = read(raw / "boundaries.json") if (raw / "boundaries.json").exists() else []
    decisions = read(raw / "decisions.json") if (raw / "decisions.json").exists() else []
    if len(boundaries) < TARGET_STEP + 1 or len(decisions) < TARGET_STEP + 1:
        raise RuntimeError(f"P2 task lacks the strict 20 s replay: {record['task_key']}")
    if result.get("stop_reason") in ("replay_mismatch", "full_replay_mismatch", "reference_replay_mismatch"):
        raise RuntimeError(f"P2 task did not reproduce its 20 s source: {record['task_key']}")
    boundary = boundaries[TARGET_STEP]
    decision = decisions[TARGET_STEP]
    if (abs(boundary["simulation_seconds"] - 20.0) > 1e-8 or
            boundary["limited"]["state_id"] != sample["target_state_id"] or
            boundary["score"] != sample["score_at_20s"] or
            decision["action_id"] != record["task"]["candidate_action_id"] or
            decision["source"] != "second_stage_candidate"):
        raise RuntimeError(f"P2 task changed the intervention state or action: {record['task_key']}")
    completed = (record["status"] == "passed" and result.get("stop_reason") == "terminal" and
                 result.get("terminal_reason") in ("time_limit", "quadruped_eliminated"))
    executable = completed and strategy_action_executable(decision)
    if not completed and not (result.get("stop_reason") == "action_error" and
                              decision.get("failure_stage") in ("validate", "compile_and_apply") and
                              decision.get("error_class") == "not_applicable_in_state"):
        raise RuntimeError(f"Unclassified P2 candidate failure: {record['task_key']} {result.get('stop_reason')}")
    return {"sample_id": sample["sample_id"], "prefix_id": sample["prefix_id"],
            "action_id": record["task"]["candidate_action_id"],
            "status": "executable" if executable else "accepted_but_blocked" if completed else "not_applicable",
            "selectable": bool(executable), "validator_status": decision["validator_status"],
            "execution_status": decision["execution_status"],
            "compiled_action_skill": (decision.get("compiled_action") or {}).get("skill"),
            "compiled_skill_state": (decision.get("compiled_skill") or {}).get("state"),
            "error_class": decision.get("error_class"), "error": decision.get("error"),
            "score_at_20s": sample["score_at_20s"],
            "observed_final_score_diagnostic": result.get("official_score"),
            "observed_delta_score_diagnostic": result.get("evaluation_delta_score"),
            "final_official_score": result.get("official_score") if executable else None,
            "official_score_after_20s": result.get("evaluation_delta_score") if executable else None,
            "terminal_reason": result.get("terminal_reason") if executable else None,
            "terminal_seconds_diagnostic": result.get("simulation_seconds"),
            "task_key": record["task_key"], "wall_seconds": record["wall_seconds"]}


def choose_training_action(outcomes, candidates, baseline):
    if len(outcomes) != len(candidates) or len({x["action_id"] for x in outcomes}) != len(candidates):
        raise ValueError("The P2 shortlist is incomplete")
    values = {item["action_id"]: item for item in outcomes}
    if not values[baseline]["selectable"]:
        raise ValueError("The continuation baseline is not executable at the P2 state")
    eligible = [item for item in outcomes if item["selectable"]]
    best = sorted(eligible, key=lambda item: (-item["final_official_score"],
                                               item["action_id"] != baseline, item["action_id"]))[0]
    return {"selected_action_id": best["action_id"],
            "baseline_action_id": baseline,
            "training_selected_final_score": best["final_official_score"],
            "training_baseline_final_score": values[baseline]["final_official_score"],
            "training_advantage": round(best["final_official_score"] -
                                        values[baseline]["final_official_score"], 6),
            "selection_rule": "Highest final official score among executable actions; exact ties keep baseline; survival diagnostic only",
            "candidate_action_ids": candidates,
            "candidate_outcomes": outcomes}


def run_study(prior, config, output, max_new_tasks=9, retry_failed=False):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = candidate_catalog(config)
    provenance = {"version": VERSION, "source_sha256": config["source_sha256"],
                  "frozen_full_game_bundle_sha256": prior["sha256"],
                  "game_seed": config["game_seed"],
                  "synthetic_seed": config["discovery_synthetic_seed"],
                  "target_step": TARGET_STEP, "target_state_id": prior["target_state_id"],
                  "candidate_action_ids": candidates}
    provenance_path = output / "experiment_provenance.json"
    if provenance_path.exists() and read(provenance_path) != provenance:
        raise ValueError("P2 output belongs to another source, prior bundle, or shortlist")
    atomic_json(provenance_path, provenance)
    atomic_json(output / "resolved_config.json", config)
    atomic_json(output / "frozen_prior_policy.json", prior["policy"])
    atomic_json(output / "source_full_game_summary.json", prior["summary"])
    atomic_json(output / "source_full_game_pairs.json", prior["paired"])
    budget = 0

    def execute(sample, action_id, completed_count):
        nonlocal budget
        task = branch_task(config, prior, sample, action_id)
        key = digest(task)[:20]
        task_path = output / "tasks" / f"{key}.json"
        if not task_path.exists() or (retry_failed and read(task_path)["status"] != "passed"):
            if budget >= max_new_tasks:
                return None
            budget += 1
        record = run_task(output, task, retry_failed=retry_failed)
        item = branch_outcome(record, sample)
        atomic_json(output / "capture_progress.json", {"stage": "running",
                    "completed_tasks": completed_count + 1, "new_tasks_this_run": budget,
                    "last_task_wall_seconds": record["wall_seconds"]})
        print(f"p2_second_stage completed={completed_count + 1} {sample['prefix_id']} "
              f"{action_id} selectable={item['selectable']} final={item['final_official_score']}", flush=True)
        return item

    training = []
    for action_id in candidates:
        item = execute(prior["training"], action_id, len(training))
        if item is None:
            atomic_json(output / "capture_progress.json", {"stage": "incomplete",
                        "completed_tasks": len(training), "new_tasks_this_run": budget})
            return {"version": VERSION, "status": "incomplete", "completed_tasks": len(training)}
        training.append(item)
    atomic_json(output / "training_outcomes.json", training)
    model = choose_training_action(training, candidates, config["continuation_action_id"])
    model.update(version=VERSION, target_step=TARGET_STEP, target_state_id=prior["target_state_id"],
                 training_sample_id=prior["training"]["sample_id"],
                 heldout_sample_id=prior["heldout"]["sample_id"],
                 full_game_input_sha256=prior["sha256"])
    atomic_json(output / "p2_local_model.json", model)
    heldout_actions = [config["continuation_action_id"]]
    if model["selected_action_id"] != heldout_actions[0]:
        heldout_actions.append(model["selected_action_id"])
    heldout = []
    for action_id in heldout_actions:
        item = execute(prior["heldout"], action_id, len(training) + len(heldout))
        if item is None:
            atomic_json(output / "capture_progress.json", {"stage": "incomplete",
                        "completed_tasks": len(training) + len(heldout), "new_tasks_this_run": budget})
            return {"version": VERSION, "status": "incomplete",
                    "completed_tasks": len(training) + len(heldout)}
        heldout.append(item)
    baseline = heldout[0]
    selected = heldout[-1]
    if not baseline["selectable"] or not selected["selectable"]:
        raise RuntimeError("Selected P2 action was not executable in the held-out prefix")
    comparison = {"version": VERSION, "status": "passed", "training": {
                      "sample_id": prior["training"]["sample_id"],
                      "selected_action_id": model["selected_action_id"],
                      "selected_final_score": model["training_selected_final_score"],
                      "baseline_final_score": model["training_baseline_final_score"],
                      "paired_advantage": model["training_advantage"]},
                  "heldout": {"sample_id": prior["heldout"]["sample_id"],
                              "selected_action_id": model["selected_action_id"],
                              "selected_final_score": selected["final_official_score"],
                              "baseline_final_score": baseline["final_official_score"],
                              "paired_advantage": round(selected["final_official_score"] -
                                                        baseline["final_official_score"], 6)},
                  "research_success": model["training_advantage"] > 0 and
                                      selected["final_official_score"] > baseline["final_official_score"]}
    atomic_json(output / "heldout_validation.json", comparison)
    summary = {"version": VERSION, "status": "passed", "training_candidates": len(training),
               "heldout_tasks": len(heldout), "selected_action_id": model["selected_action_id"],
               "training_advantage": model["training_advantage"],
               "heldout_advantage": comparison["heldout"]["paired_advantage"],
               "research_success": comparison["research_success"],
               "total_task_wall_seconds": round(sum(x["wall_seconds"] for x in training + heldout), 3)}
    atomic_json(output / "run_summary.json", summary)
    atomic_json(output / "capture_progress.json", {"stage": "complete",
                "completed_tasks": len(training) + len(heldout), "new_tasks_this_run": budget})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-bundle", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tasks", type=int, default=9)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    config = resolve_config(DEFAULT_CONFIG)
    prior = checked_full_game_bundle(args.prior_bundle, config)
    output = args.output.expanduser().resolve()
    if output == prior["path"] or output in prior["path"].parents:
        parser.error("--output must be separate from the frozen source bundle")
    output.mkdir(parents=True, exist_ok=True)
    invocation_path = output / "invocations.json"
    invocations = read(invocation_path) if invocation_path.exists() else []
    invocations.append({"utc_started": datetime.now(timezone.utc).isoformat(),
                        "command": ["bash", "scripts/run_oracle_pilot_ubuntu.sh", "oracle_p2_second_stage",
                                    *(argv if argv is not None else sys.argv[1:])]})
    atomic_json(invocation_path, invocations)
    started = time.monotonic()
    try:
        result = run_study(prior, config, output, args.max_new_tasks, args.retry_failed)
    except Exception as exc:
        atomic_json(output / "run_failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                  "elapsed_wall_seconds": round(time.monotonic() - started, 3)})
        result = {"version": VERSION, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if result["status"] == "passed":
        (output / "run_failure.json").unlink(missing_ok=True)
    result["invocation_wall_seconds"] = round(time.monotonic() - started, 3)
    archive = package_artifacts(output, "oracle_p2_second_stage_bundle.zip",
                                {"version": VERSION, "source_sha256": config["source_sha256"],
                                 "frozen_full_game_bundle_sha256": prior["sha256"]})
    result["bundle"] = str(archive)
    result["bundle_sha256"] = sha256_file(archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
