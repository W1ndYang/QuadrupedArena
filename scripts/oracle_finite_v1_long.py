"""Extend the first Oracle experiment through delayed scoring and a held-out state.

The original 44-action two-second transition model remains immutable. This
small follow-up tests a fixed formal-action shortlist from 8 to at most 20 s,
with the same formal CAPTURE_POINT continuation after the first action. Three
real states train the local choice; a fourth qualified state is held out.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import zipfile

from arena_paths import PROJECT, workspace_root
from oracle_finite_v1 import DEFAULT_ANALYSIS, DEFAULT_REPLAY_BUNDLE, load_inputs
from oracle_pilot import DEFAULT_CONFIG, atomic_json, digest, read, resolve_config, run_task, task_base
from oracle_pilot_runtime import strategy_action_executable
from oracle_pilot_v21 import package_artifacts, sha256_file

VERSION = "oracle_finite_v1_delayed_score_followup"
DEFAULT_PRIOR = workspace_root() / "experiments/oracle-finite-v1/oracle_finite_v1_bundle.zip"
DEFAULT_OUTPUT = workspace_root() / "experiments/oracle-finite-v1-long"
HORIZON_STEP = 10  # 20 simulation seconds; the first decision is the 8 s boundary.
REQUESTS = (
    ("CAPTURE_POINT", {"target": "active", "risk": "fast"}),
    ("CAPTURE_POINT", {"target": "active", "risk": "safe"}),
    ("MOVE_TO", {"target": "active", "risk": "fast"}),
    ("MOVE_TO", {"target": "active", "risk": "safe"}),
    ("TAKE_COVER", {"risk": "safe"}),
    ("TAKE_COVER", {"risk": "fast"}),
    ("TAKE_COVER", {"cover": "High1", "risk": "safe"}),
    ("TAKE_COVER", {"cover": "High3", "risk": "safe"}),
    ("OBSERVE", {"direction": "north", "risk": "fast"}),
    ("OBSERVE", {"direction": "scan", "risk": "safe"}),
)


def checked_prior(path, inputs):
    path = Path(path).expanduser().resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None or any(
                name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("Prior Oracle bundle has duplicate, unsafe, or corrupt members")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["file_count"] != len(manifest["files"]):
            raise ValueError("Prior Oracle manifest count mismatch")
        expected = {}
        for item in manifest["files"]:
            payload = archive.read(item["path"])
            if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"Prior Oracle file checksum mismatch: {item['path']}")
            expected[item["path"]] = item["sha256"]
        checks = dict(line.split("  ", 1)[::-1] for line in archive.read("SHA256SUMS").decode().splitlines())
        if checks != expected:
            raise ValueError("Prior Oracle SHA256SUMS mismatch")
        model = json.loads(archive.read("transition_model.json"))
        summary = json.loads(archive.read("run_summary.json"))
        evaluation = json.loads(archive.read("evaluation_comparison.json"))
    if (summary.get("status") != "passed" or summary.get("completed_tasks") != 182 or
            model.get("root_state_id") != inputs["group"]["state_id"] or
            model.get("transition_count") != 176 or
            model.get("corrected_samples_sha256") != inputs["corrected_samples_sha256"] or
            model.get("replay_bundle_sha256") != inputs["replay_bundle_sha256"] or
            evaluation.get("valid_pairs") != 3):
        raise ValueError("Prior model does not match the verified first batch")
    return {"model": model, "summary": summary, "evaluation": evaluation,
            "bundle_sha256": sha256_file(path)}


def shortlist(config):
    catalog = {(item["skill"], tuple(sorted(item["params"].items()))): item["action_id"]
               for item in config["action_catalog"]}
    actions = [catalog[(skill, tuple(sorted(params.items())))] for skill, params in REQUESTS]
    if len(actions) != len(set(actions)) or config["continuation_action_id"] != actions[0]:
        raise ValueError("Shortlist must be ten unique formal actions with the fixed baseline first")
    return actions


def heldout_sample(inputs):
    group = inputs["group"]
    selected = {item["sample_id"] for item in group["samples"]}
    remaining = [item for item in group["candidate_samples"] if item["sample_id"] not in selected]
    if len(remaining) != 1 or remaining[0]["prefix_id"] not in inputs["prefixes"]:
        raise ValueError("Expected exactly one replay-verified held-out candidate")
    return remaining[0]


def task_for(config, inputs, prior, sample, action_id):
    prefix = inputs["prefixes"][sample["prefix_id"]]
    baseline = action_id == config["continuation_action_id"]
    mode = "oracle_eval_baseline_prefix" if baseline else "oracle_eval_prefix"
    task = task_base(config, mode, sample["synthetic_seed"])
    task.update(experiment_version=VERSION, sample_id=sample["sample_id"],
                prefix_id=sample["prefix_id"], prefix_action_ids=prefix["prefix_action_ids"],
                expected_boundaries=prefix["expected_boundaries"], max_steps=HORIZON_STEP,
                candidate_action_id_long=action_id,
                prior_bundle_sha256=prior["bundle_sha256"],
                corrected_samples_sha256=inputs["corrected_samples_sha256"])
    if not baseline:
        task["policy_table"] = {"4": {inputs["group"]["state_id"]: action_id}}
    return task


def outcome_from_record(record, action_id):
    raw = Path(record["raw_dir"])
    result = record.get("result") or {}
    if result.get("stop_reason") in ("replay_mismatch", "full_replay_mismatch", "reference_replay_mismatch"):
        raise RuntimeError(f"Strict prefix replay mismatch: {record['task_key']}")
    boundaries = read(raw / "boundaries.json") if (raw / "boundaries.json").exists() else []
    decisions = read(raw / "decisions.json") if (raw / "decisions.json").exists() else []
    if len(boundaries) < 5 or len(decisions) < 5:
        raise RuntimeError(f"Incomplete long rollout: {record['task_key']}")
    first = decisions[4]
    error = first.get("error_class")
    if record["status"] != "passed" and error != "not_applicable_in_state":
        raise RuntimeError(f"Unclassified long-rollout failure: {record['task_key']}: {error}")
    executable = record["status"] == "passed" and strategy_action_executable(first)
    transitions = []
    for step in range(4, min(len(decisions), len(boundaries) - 1)):
        decision = decisions[step]
        now, following = boundaries[step], boundaries[step + 1]
        transitions.append({"decision_step": step,
                            "state_id": now["limited"]["state_id"],
                            "canonical_state": now["limited"]["canonical_state"],
                            "action_id": decision["action_id"],
                            "reward_delta_score": decision.get("delta_score"),
                            "next_state_id": following["limited"]["state_id"],
                            "next_canonical_state": following["limited"]["canonical_state"],
                            "validator_status": decision["validator_status"],
                            "execution_status": decision["execution_status"],
                            "terminal": bool(following["finished"]),
                            "terminal_reason": following["finished"],
                            "sample_id": record["task"]["sample_id"],
                            "initial_action_id": action_id, "task_key": record["task_key"]})
    return {"sample_id": record["task"]["sample_id"],
            "prefix_id": record["task"]["prefix_id"],
            "initial_action_id": action_id, "executable": executable,
            "validator_status": first["validator_status"],
            "execution_status": first["execution_status"],
            "first_error_class": error, "first_error": first.get("error"),
            "score_at_8s": boundaries[4]["score"],
            "final_official_score": result.get("official_score"),
            "official_score_delta": result.get("evaluation_delta_score") if executable else None,
            "survival_seconds": result.get("simulation_seconds"),
            "terminal_reason": result.get("terminal_reason"),
            "stop_reason": result.get("stop_reason"),
            "status": record["status"], "task_key": record["task_key"],
            "wall_seconds": record["wall_seconds"],
            "transitions": transitions}


def solve_rollouts(outcomes, actions, samples, baseline_action):
    by_action = {action: [x for x in outcomes if x["initial_action_id"] == action] for action in actions}
    rows = []
    for action in actions:
        values = by_action[action]
        eligible = len(values) == len(samples) and len({x["sample_id"] for x in values}) == len(samples) and all(
            x["executable"] for x in values)
        rows.append({"action_id": action, "common_executable": eligible,
                     "sample_count": len(values),
                     "score_deltas": [x["official_score_delta"] for x in values],
                     "survival_seconds": [x["survival_seconds"] for x in values],
                     "mean_score_delta": statistics.mean(x["official_score_delta"] for x in values) if eligible else None,
                     "mean_survival_seconds": statistics.mean(x["survival_seconds"] for x in values) if eligible else None,
                     "terminal_reasons": [x["terminal_reason"] for x in values]})
    eligible = [x for x in rows if x["common_executable"]]
    if not eligible:
        raise ValueError("No shortlisted formal action is executable in all three root samples")
    best = sorted(eligible, key=lambda x: (-x["mean_score_delta"],
                                           x["action_id"] != baseline_action, x["action_id"]))[0]
    baseline = next(x for x in rows if x["action_id"] == baseline_action)
    return {"version": VERSION, "selected_action_id": best["action_id"],
            "baseline_action_id": baseline_action,
            "selected_mean_score_delta": best["mean_score_delta"],
            "baseline_mean_score_delta": baseline["mean_score_delta"],
            "training_mean_advantage": (best["mean_score_delta"] - baseline["mean_score_delta"]
                                        if baseline["mean_score_delta"] is not None else None),
            "shortlist_size": len(actions), "training_samples": len(samples),
            "horizon_start_seconds": 8, "horizon_end_seconds": 20,
            "continuation_action_id": baseline_action,
            "selection_rule": "mean official score gain; exact score ties retain the fixed baseline, then action ID; survival is diagnostic only",
            "action_values": rows}


def execute(output, task, records, budget, *, retry_failed=False):
    key = digest(task)[:20]
    existing = output / "tasks" / f"{key}.json"
    if not existing.exists():
        if budget["used"] >= budget["max"]:
            return None
        budget["used"] += 1
    elif retry_failed and read(existing)["status"] != "passed":
        if budget["used"] >= budget["max"]:
            return None
        budget["used"] += 1
    record = run_task(output, task, retry_failed=retry_failed)
    records.append(record)
    return record


def run_followup(inputs, prior, config, output, max_new_tasks, retry_failed):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    actions = shortlist(config)
    training = inputs["group"]["samples"]
    heldout = heldout_sample(inputs)
    provenance = {"version": VERSION, "source_sha256": config["source_sha256"],
                  "prior_bundle_sha256": prior["bundle_sha256"],
                  "replay_bundle_sha256": inputs["replay_bundle_sha256"],
                  "corrected_samples_sha256": inputs["corrected_samples_sha256"],
                  "game_seed": config["game_seed"], "synthetic_seed": config["discovery_synthetic_seed"],
                  "shortlist": actions, "max_steps": HORIZON_STEP}
    prior_file = output / "experiment_provenance.json"
    if prior_file.exists() and read(prior_file) != provenance:
        raise ValueError("Long rollout output belongs to a different source or input")
    atomic_json(prior_file, provenance)
    atomic_json(output / "resolved_config.json", config)
    atomic_json(output / "prior_run_summary.json", prior["summary"])
    evidence = output / "input_evidence"
    evidence.mkdir(exist_ok=True)
    shutil.copy2(inputs["analysis_dir"] / "samples_corrected.json", evidence / "samples_corrected.json")
    shutil.copy2(inputs["analysis_dir"] / "replay_plan.json", evidence / "replay_plan.json")
    atomic_json(evidence / "replay_validation.json", inputs["replay"])
    atomic_json(evidence / "prior_transition_model.json", prior["model"])
    atomic_json(evidence / "prior_evaluation_comparison.json", prior["evaluation"])
    atomic_json(output / "prior_model_summary.json", {k: prior["model"].get(k) for k in
        ("root_state_id", "transition_count", "root_actions_with_multiple_successors",
         "root_actions_with_different_rewards", "source_sha256")})
    budget = {"used": 0, "max": max_new_tasks}
    records = []
    outcomes = []
    for sample in training:
        for action_id in actions:
            task = task_for(config, inputs, prior, sample, action_id)
            record = execute(output, task, records, budget, retry_failed=retry_failed)
            if record is None:
                atomic_json(output / "capture_progress.json", {"stage": "training_incomplete",
                            "completed_training_rollouts": len(outcomes), "new_tasks": budget["used"]})
                return {"status": "incomplete", "stage": "training", "completed": len(outcomes)}
            outcome = outcome_from_record(record, action_id)
            outcomes.append(outcome)
            atomic_json(output / "capture_progress.json", {"stage": "training",
                        "completed_training_rollouts": len(outcomes), "new_tasks": budget["used"],
                        "last_task_wall_seconds": record["wall_seconds"]})
            print(f"long_oracle training={len(outcomes)}/{len(training)*len(actions)} "
                  f"action={action_id} sample={sample['prefix_id']} "
                  f"gain={outcome['official_score_delta']} t={outcome['survival_seconds']}", flush=True)
    atomic_json(output / "training_rollouts.json", outcomes)
    policy = solve_rollouts(outcomes, actions, training, config["continuation_action_id"])
    policy["root_state_id"] = inputs["group"]["state_id"]
    policy["source_sha256"] = config["source_sha256"]
    policy["prior_bundle_sha256"] = prior["bundle_sha256"]
    policy["policy_table"] = {"4": {inputs["group"]["state_id"]: policy["selected_action_id"]}}
    atomic_json(output / "long_policy.json", policy)
    transition_rows = [transition for outcome in outcomes for transition in outcome["transitions"]]
    model = {"version": VERSION, "source_sha256": config["source_sha256"],
             "prior_bundle_sha256": prior["bundle_sha256"],
             "root_state_id": inputs["group"]["state_id"],
             "decision_period_seconds": 2, "horizon_start_seconds": 8, "horizon_end_seconds": 20,
             "action_catalog": config["action_catalog"], "root_candidate_action_ids": actions,
             "continuation_action_id": config["continuation_action_id"],
             "root_action_outcomes": [{k: value for k, value in outcome.items() if k != "transitions"}
                                      for outcome in outcomes],
             "transitions": transition_rows,
             "distinct_state_time_keys": len({(x["decision_step"],x["state_id"]) for x in transition_rows}),
             "transition_count": len(transition_rows),
             "approximation": "All 44 formal actions retain their 2 s evidence in the prior model. Ten selected formal actions get 8-to-20 s rollouts on three hidden root states; all later decisions use the fixed baseline."}
    atomic_json(output / "long_transition_model.json", model)
    policy["long_model_sha256"] = sha256_file(output / "long_transition_model.json")
    atomic_json(output / "long_policy.json", policy)
    checks = []
    for action_id in (policy["selected_action_id"], config["continuation_action_id"]):
        if any(x["initial_action_id"] == action_id for x in checks):
            continue
        task = task_for(config, inputs, prior, heldout, action_id)
        task["heldout_validation"] = True
        task["training_policy_sha256"] = sha256_file(output / "long_policy.json")
        record = execute(output, task, records, budget, retry_failed=retry_failed)
        if record is None:
            atomic_json(output / "capture_progress.json", {"stage": "heldout_incomplete",
                        "completed_training_rollouts": len(outcomes),
                        "completed_heldout_rollouts": len(checks), "new_tasks": budget["used"]})
            return {"status": "incomplete", "stage": "heldout", "completed": len(outcomes)+len(checks)}
        outcome = outcome_from_record(record, action_id)
        checks.append(outcome)
        print(f"long_oracle heldout={len(checks)}/2 action={action_id} "
              f"gain={outcome['official_score_delta']} t={outcome['survival_seconds']}", flush=True)
    selected = checks[0]
    baseline = next((x for x in checks if x["initial_action_id"] == config["continuation_action_id"]),
                    selected if policy["selected_action_id"] == config["continuation_action_id"] else None)
    advantage = (selected["official_score_delta"] - baseline["official_score_delta"]
                 if baseline and selected["executable"] and baseline["executable"] else None)
    validation = {"version": VERSION, "heldout_sample_id": heldout["sample_id"],
                  "heldout_prefix_id": heldout["prefix_id"], "selected_action_id": policy["selected_action_id"],
                  "baseline_action_id": config["continuation_action_id"],
                  "selected": selected, "baseline": baseline,
                  "paired_official_score_advantage": advantage,
                  "same_action_control": policy["selected_action_id"] == config["continuation_action_id"],
                  "independent_heldout_tasks": len(checks),
                  "status": "passed" if advantage is not None else "partial",
                  "same_training_samples": False, "same_game_seed": True,
                  "note": "Held-out real state under the same full R2-L ID; same fixed game and synthetic-vision seeds."}
    atomic_json(output / "heldout_validation.json", validation)
    summary = {"version": VERSION, "status": validation["status"],
               "training_rollouts": len(outcomes), "heldout_rollouts": len(checks),
               "long_transition_count": model["transition_count"],
               "long_state_time_keys": model["distinct_state_time_keys"],
               "selected_action_id": policy["selected_action_id"],
               "training_mean_advantage": policy["training_mean_advantage"],
               "heldout_advantage": advantage,
               "mean_training_score_delta": policy["selected_mean_score_delta"],
               "mean_baseline_score_delta": policy["baseline_mean_score_delta"],
               "total_task_wall_seconds": round(sum(x["wall_seconds"] for x in records), 3)}
    atomic_json(output / "run_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--replay-bundle", type=Path, default=DEFAULT_REPLAY_BUNDLE)
    parser.add_argument("--prior-bundle", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tasks", type=int, default=32,
                        help="Maximum new Isaac tasks in this invocation; rerun to resume")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    config = resolve_config(DEFAULT_CONFIG)
    inputs = load_inputs(args.analysis_dir, args.replay_bundle, config)
    prior = checked_prior(args.prior_bundle, inputs)
    actions = shortlist(config)
    print(f"validated_prior=182/182 root={inputs['group']['state_id'][-12:]} "
          f"training={3*len(actions)} heldout=1..2 max_new={args.max_new_tasks}", flush=True)
    output = args.output.expanduser().resolve()
    if output == inputs["analysis_dir"] or inputs["analysis_dir"] in output.parents:
        parser.error("--output must be separate from corrected analysis")
    output.mkdir(parents=True, exist_ok=True)
    invocations_path = output / "invocations.json"
    invocations = read(invocations_path) if invocations_path.exists() else []
    invocations.append({"utc_started": datetime.now(timezone.utc).isoformat(),
                        "command": ["bash", "scripts/run_oracle_pilot_ubuntu.sh", "finite_oracle_long", *sys.argv[1:]]})
    atomic_json(invocations_path, invocations)
    started = time.monotonic()
    try:
        result = run_followup(inputs, prior, config, output, args.max_new_tasks, args.retry_failed)
    except Exception as exc:
        atomic_json(output / "run_failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                  "elapsed_wall_seconds": round(time.monotonic()-started, 3)})
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    result["invocation_wall_seconds"] = round(time.monotonic()-started, 3)
    archive = package_artifacts(output, "oracle_finite_v1_long_bundle.zip",
                                {"version": VERSION, "source_sha256": config["source_sha256"],
                                 "prior_bundle_sha256": prior["bundle_sha256"]})
    result["bundle"] = str(archive)
    result["bundle_sha256"] = sha256_file(archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
