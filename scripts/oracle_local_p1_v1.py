"""One-bit observable P1 exit split, delayed-reward solve, and paired Isaac check.

The feature is defined from public self_state.xy and map-known P1 geometry.
No R1 position, hidden sample ID, or reference digest enters action selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import zipfile

from adaptive_motion import load_config
from arena_paths import PROJECT, workspace_root
from oracle_finite_v1 import DEFAULT_ANALYSIS, DEFAULT_REPLAY_BUNDLE, load_inputs
from oracle_finite_v1_long import DEFAULT_PRIOR as FIRST_BUNDLE, checked_prior, heldout_sample
from oracle_local_state_v1 import VERSION as FEATURE_VERSION, p1_exit_state, p1_west_boundary
from oracle_pilot import DEFAULT_CONFIG, atomic_json, digest, read, resolve_config, run_task, task_base
from oracle_pilot_runtime import strategy_action_executable
from oracle_pilot_v21 import package_artifacts, sha256_file

VERSION = "oracle_local_p1_v1_delayed_reward"
DEFAULT_LONG = workspace_root() / "experiments/oracle-finite-v1-long/oracle_finite_v1_long_bundle.zip"
DEFAULT_OUTPUT = workspace_root() / "experiments/oracle-local-p1-v1"
HORIZON_STEP = 10


def checked_long_bundle(path, inputs, prior):
    path = Path(path).expanduser().resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None or any(
                name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("Long rollout bundle has duplicate, unsafe, or corrupt members")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["file_count"] != len(manifest["files"]):
            raise ValueError("Long rollout manifest count mismatch")
        expected = {}
        for item in manifest["files"]:
            data = archive.read(item["path"])
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Long rollout file checksum mismatch: {item['path']}")
            expected[item["path"]] = item["sha256"]
        checks = dict(line.split("  ", 1)[::-1] for line in archive.read("SHA256SUMS").decode().splitlines())
        if checks != expected:
            raise ValueError("Long rollout SHA256SUMS mismatch")
        summary = json.loads(archive.read("run_summary.json"))
        outcomes = json.loads(archive.read("training_rollouts.json"))
        heldout = json.loads(archive.read("heldout_validation.json"))
        provenance = json.loads(archive.read("experiment_provenance.json"))
        if (summary.get("status") != "passed" or summary.get("training_rollouts") != 30 or
                summary.get("heldout_rollouts") != 2 or len(outcomes) != 30 or
                provenance["prior_bundle_sha256"] != prior["bundle_sha256"] or
                provenance["corrected_samples_sha256"] != inputs["corrected_samples_sha256"] or
                heldout["heldout_sample_id"] != heldout_sample(inputs)["sample_id"]):
            raise ValueError("Long rollout data do not match the verified P1 group")
        return {"path": path, "sha256": sha256_file(path), "summary": summary,
                "outcomes": outcomes, "heldout": heldout, "archive": zipfile.ZipFile(path)}


def public_xy_at_8(archive, task_key):
    frames = json.loads(archive.read(f"raw/{task_key}/frame_index.json"))
    matches = [item["own_xy"] for item in frames if abs(item["simulation_seconds"] - 8.0) < 1e-8]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one public 8 s own position in {task_key}")
    return matches[0]


def formal_action(config, skill, params):
    selected = [item["action_id"] for item in config["action_catalog"]
                if item["skill"] == skill and item["params"] == params]
    if len(selected) != 1:
        raise ValueError("The proposed action is not a unique member of the formal 44-item catalog")
    return selected[0]


def build_local_model(inputs, prior, long_data, config):
    cfg = load_config(PROJECT / "configs" / config["rules_config"])
    cfg["seed"] = config["game_seed"]
    baseline = config["continuation_action_id"]
    scan = formal_action(config, "OBSERVE", {"direction": "scan", "risk": "safe"})
    action_ids = (baseline, scan)
    outcomes = long_data["outcomes"]
    by_sample_action = {(item["sample_id"], item["initial_action_id"]): item for item in outcomes}
    training = []
    for sample in inputs["group"]["samples"]:
        values = []
        public_positions = []
        for action_id in action_ids:
            item = by_sample_action.get((sample["sample_id"], action_id))
            if item is None or not item["executable"]:
                raise ValueError(f"Missing executable delayed reward: {sample['sample_id']} {action_id}")
            key = item["task_key"]
            decisions = json.loads(long_data["archive"].read(f"raw/{key}/decisions.json"))
            if not strategy_action_executable(decisions[4]):
                raise ValueError(f"Prior action only compiled to a blocked no-op: {key}")
            xy = public_xy_at_8(long_data["archive"], key)
            public_positions.append(xy)
            values.append({"action_id": action_id, "official_score_delta": item["official_score_delta"],
                           "terminal_reason": item["terminal_reason"],
                           "survival_seconds": item["survival_seconds"], "source_task_key": key})
        if any(xy != public_positions[0] for xy in public_positions[1:]):
            raise ValueError("Same prefix has inconsistent public 8 s positions")
        local = p1_exit_state(inputs["group"]["state_id"], 4, public_positions[0], cfg,
                              inputs["group"]["state_id"])
        training.append({"sample_id": sample["sample_id"], "prefix_id": sample["prefix_id"],
                         "public_own_xy_diagnostic": public_positions[0],
                         "local_state": local, "action_outcomes": values})
    by_bin = defaultdict(list)
    for sample in training:
        by_bin[sample["local_state"]["bin"]].append(sample)
    if set(by_bin) != {"p1_west_edge", "p1_other_exit"}:
        raise ValueError("The map-defined one-bit feature did not split the training states")
    policy = {}
    values_by_bin = {}
    for label, samples in sorted(by_bin.items()):
        scores = {action: statistics.mean(next(x["official_score_delta"] for x in item["action_outcomes"]
                                               if x["action_id"] == action) for item in samples)
                  for action in action_ids}
        policy[label] = sorted(action_ids, key=lambda action: (-scores[action], action != baseline, action))[0]
        values_by_bin[label] = {"sample_count": len(samples), "action_mean_score_delta": scores,
                                "selected_action_id": policy[label]}
    training_pairs = []
    for item in training:
        selected = policy[item["local_state"]["bin"]]
        selected_score = next(x["official_score_delta"] for x in item["action_outcomes"]
                              if x["action_id"] == selected)
        baseline_score = next(x["official_score_delta"] for x in item["action_outcomes"]
                              if x["action_id"] == baseline)
        training_pairs.append({"sample_id": item["sample_id"], "bin": item["local_state"]["bin"],
                               "policy_score_delta": selected_score, "baseline_score_delta": baseline_score,
                               "paired_advantage": round(selected_score - baseline_score, 6)})
    heldout = heldout_sample(inputs)
    heldout_baseline = long_data["heldout"]["baseline"]
    heldout_xy = public_xy_at_8(long_data["archive"], heldout_baseline["task_key"])
    heldout_local = p1_exit_state(inputs["group"]["state_id"], 4, heldout_xy, cfg,
                                  inputs["group"]["state_id"])
    model = {"version": VERSION, "feature_version": FEATURE_VERSION,
             "base_r2l_state_id": inputs["group"]["state_id"],
             "feature_rule": "At the 8 s target_P1 state, own public x < P1.center_x - P1.radius/2",
             "threshold_x": p1_west_boundary(cfg),
             "feature_source": "public_packet.self_state.xy and map-known P1 geometry",
             "decision_step": 4, "horizon_start_seconds": 8, "horizon_end_seconds": 20,
             "formal_action_catalog_count": len(config["action_catalog"]),
             "compared_action_ids": list(action_ids),
             "baseline_continuation_action_id": baseline,
             "training_samples": training, "values_by_bin": values_by_bin,
             "training_pairs": training_pairs,
             "training_mean_advantage": statistics.mean(x["paired_advantage"] for x in training_pairs),
             "heldout_sample_id": heldout["sample_id"],
             "heldout_prefix_id": heldout["prefix_id"],
             "heldout_public_own_xy_diagnostic": heldout_xy,
             "heldout_predicted_bin": heldout_local["bin"],
             "heldout_outcome_used_for_training": False,
             "prior_two_second_model_canonical_digest": sha256_file_from_data(prior["model"]),
             "long_bundle_sha256": long_data["sha256"],
             "source_sha256": config["source_sha256"],
             "limitation": "One map-defined binary state and two formal root actions; continuation is fixed baseline. No full 44-action long-horizon optimality claim."}
    policy_file = {"version": VERSION, "feature_version": FEATURE_VERSION,
                   "base_r2l_state_id": inputs["group"]["state_id"],
                   "feature_threshold_x": p1_west_boundary(cfg),
                   "local_policy_table": policy,
                   "fallback_action_id": baseline,
                   "target_decision_step": 4,
                   "training_mean_advantage": model["training_mean_advantage"],
                   "heldout_predicted_bin": heldout_local["bin"],
                   "selection_rule": "Higher 8-to-20 s official score delta within each observable bin; exact ties retain baseline",
                   "source_sha256": config["source_sha256"],
                   "long_bundle_sha256": long_data["sha256"]}
    return model, policy_file


def sha256_file_from_data(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def evaluation_task(config, inputs, policy, sample, baseline_mode):
    prefix = inputs["prefixes"][sample["prefix_id"]]
    mode = "oracle_local_baseline_prefix" if baseline_mode else "oracle_local_eval_prefix"
    task = task_base(config, mode, sample["synthetic_seed"])
    task.update(experiment_version=VERSION, sample_id=sample["sample_id"],
                prefix_id=sample["prefix_id"], prefix_action_ids=prefix["prefix_action_ids"],
                expected_boundaries=prefix["expected_boundaries"], max_steps=HORIZON_STEP,
                feature_version=FEATURE_VERSION, feature_threshold_x=policy["feature_threshold_x"],
                target_state_id=policy["base_r2l_state_id"],
                local_policy_table=policy["local_policy_table"] if not baseline_mode else {},
                local_policy_sha256=sha256_file_from_data(policy),
                corrected_samples_sha256=inputs["corrected_samples_sha256"])
    return task


def task_outcome(record, policy):
    result = record.get("result") or {}
    if record["status"] != "passed" or result.get("stop_reason") not in (
            "terminal", "planned_boundary"):
        raise RuntimeError(f"Local policy task did not complete: {record['task_key']} {result.get('stop_reason')}")
    raw = Path(record["raw_dir"])
    decisions = read(raw / "decisions.json")
    boundaries = read(raw / "boundaries.json")
    if len(decisions) < 5 or len(boundaries) < 6:
        raise RuntimeError(f"Local policy task has no post-action boundary: {record['task_key']}")
    local = boundaries[4].get("local_policy_state")
    if (not local or local["version"] != FEATURE_VERSION or
            local["base_state_id"] != policy["base_r2l_state_id"] or
            local["decision_step"] != 4 or
            local["threshold_x"] != policy["feature_threshold_x"]):
        raise RuntimeError(f"Local state missing or changed: {record['task_key']}")
    first = decisions[4]
    expected = (policy["fallback_action_id"] if record["task"]["mode"] == "oracle_local_baseline_prefix"
                else policy["local_policy_table"][local["bin"]])
    expected_source = ("baseline" if record["task"]["mode"] == "oracle_local_baseline_prefix"
                       else "local_table:" + local["bin"])
    if (first["action_id"] != expected or first["source"] != expected_source or
            first.get("local_policy_state") != local or not strategy_action_executable(first)):
        raise RuntimeError(f"Local policy action was not the selected executable action: {record['task_key']}")
    return {"sample_id": record["task"]["sample_id"], "prefix_id": record["task"]["prefix_id"],
            "mode": record["task"]["mode"], "local_bin": local["bin"],
            "action_id": first["action_id"], "action_source": first["source"],
            "score_at_8s": boundaries[4]["score"],
            "official_score_delta": result["evaluation_delta_score"],
            "final_official_score": result["official_score"],
            "survival_seconds": result["simulation_seconds"],
            "terminal_reason": result["terminal_reason"],
            "task_key": record["task_key"], "wall_seconds": record["wall_seconds"]}


def run_experiment(inputs, long_data, config, output, max_new_tasks=8, retry_failed=False):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, policy = build_local_model(inputs, long_data["prior"], long_data, config)
    provenance = {"version": VERSION, "source_sha256": config["source_sha256"],
                  "long_bundle_sha256": long_data["sha256"],
                  "corrected_samples_sha256": inputs["corrected_samples_sha256"],
                  "game_seed": config["game_seed"], "synthetic_seed": config["discovery_synthetic_seed"]}
    existing = output / "experiment_provenance.json"
    if existing.exists() and read(existing) != provenance:
        raise ValueError("Local policy output belongs to another source or input")
    atomic_json(existing, provenance)
    atomic_json(output / "local_model.json", model)
    atomic_json(output / "local_policy.json", policy)
    atomic_json(output / "resolved_config.json", config)
    evidence = output / "input_evidence"
    evidence.mkdir(exist_ok=True)
    shutil.copy2(inputs["analysis_dir"] / "samples_corrected.json", evidence / "samples_corrected.json")
    shutil.copy2(inputs["analysis_dir"] / "replay_plan.json", evidence / "replay_plan.json")
    atomic_json(evidence / "long_run_summary.json", long_data["summary"])
    budget = 0
    results = []
    samples = [*inputs["group"]["samples"], heldout_sample(inputs)]
    for sample in samples:
        for baseline_mode in (False, True):
            task = evaluation_task(config, inputs, policy, sample, baseline_mode)
            key = digest(task)[:20]
            path = output / "tasks" / f"{key}.json"
            new_or_retry = not path.exists() or (retry_failed and read(path)["status"] != "passed")
            if new_or_retry:
                if budget >= max_new_tasks:
                    atomic_json(output / "capture_progress.json", {"stage": "incomplete",
                                "completed_tasks": len(results), "new_tasks_this_run": budget})
                    return {"status": "incomplete", "completed_tasks": len(results)}
                budget += 1
            record = run_task(output, task, retry_failed=retry_failed)
            item = task_outcome(record, policy)
            results.append(item)
            atomic_json(output / "capture_progress.json", {"stage": "running",
                        "completed_tasks": len(results), "new_tasks_this_run": budget,
                        "last_task_wall_seconds": record["wall_seconds"]})
            print(f"local_oracle {len(results)}/8 {sample['prefix_id']} {item['mode']} "
                  f"bin={item['local_bin']} gain={item['official_score_delta']}", flush=True)
    pairs = []
    for sample in samples:
        a = next(x for x in results if x["sample_id"] == sample["sample_id"] and x["mode"] == "oracle_local_eval_prefix")
        b = next(x for x in results if x["sample_id"] == sample["sample_id"] and x["mode"] == "oracle_local_baseline_prefix")
        if a["local_bin"] != b["local_bin"]:
            raise RuntimeError(f"Pair local state mismatch: {sample['sample_id']}")
        pairs.append({"sample_id": sample["sample_id"], "prefix_id": sample["prefix_id"],
                      "local_bin": a["local_bin"], "policy_action_id": a["action_id"],
                      "baseline_action_id": b["action_id"],
                      "policy_official_score_delta": a["official_score_delta"],
                      "baseline_official_score_delta": b["official_score_delta"],
                      "paired_advantage": round(a["official_score_delta"] - b["official_score_delta"], 6),
                      "policy_terminal_reason": a["terminal_reason"],
                      "baseline_terminal_reason": b["terminal_reason"],
                      "policy_task_key": a["task_key"], "baseline_task_key": b["task_key"]})
    training = pairs[:3]
    heldout = pairs[3]
    comparison = {"version": VERSION, "pairs": pairs,
                  "training_mean_advantage": statistics.mean(x["paired_advantage"] for x in training),
                  "heldout_advantage": heldout["paired_advantage"],
                  "heldout_sample_id": heldout["sample_id"],
                  "heldout_success": heldout["paired_advantage"] > 0,
                  "all_four_mean_advantage": statistics.mean(x["paired_advantage"] for x in pairs),
                  "status": "passed", "same_game_and_vision_seed": True}
    atomic_json(output / "paired_validation.json", comparison)
    summary = {"version": VERSION, "status": "passed", "tasks": len(results),
               "local_state_bins": len(model["values_by_bin"]),
               "model_training_mean_advantage": model["training_mean_advantage"],
               "real_training_mean_advantage": comparison["training_mean_advantage"],
               "real_heldout_advantage": comparison["heldout_advantage"],
               "all_four_mean_advantage": comparison["all_four_mean_advantage"],
               "research_success": comparison["heldout_success"] and comparison["training_mean_advantage"] > 0,
               "total_task_wall_seconds": round(sum(x["wall_seconds"] for x in results), 3)}
    atomic_json(output / "run_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--replay-bundle", type=Path, default=DEFAULT_REPLAY_BUNDLE)
    parser.add_argument("--first-bundle", type=Path, default=FIRST_BUNDLE)
    parser.add_argument("--long-bundle", type=Path, default=DEFAULT_LONG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tasks", type=int, default=8)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    config = resolve_config(DEFAULT_CONFIG)
    inputs = load_inputs(args.analysis_dir, args.replay_bundle, config)
    prior = checked_prior(args.first_bundle, inputs)
    long_data = checked_long_bundle(args.long_bundle, inputs, prior)
    long_data["prior"] = prior
    output = args.output.expanduser().resolve()
    if output == inputs["analysis_dir"] or inputs["analysis_dir"] in output.parents:
        parser.error("--output must be separate from corrected analysis")
    output.mkdir(parents=True, exist_ok=True)
    invocation_path = output / "invocations.json"
    invocations = read(invocation_path) if invocation_path.exists() else []
    invocations.append({"utc_started": datetime.now(timezone.utc).isoformat(),
                        "command": ["bash", "scripts/run_oracle_pilot_ubuntu.sh", "oracle_local_p1", *sys.argv[1:]]})
    atomic_json(invocation_path, invocations)
    started = time.monotonic()
    try:
        result = run_experiment(inputs, long_data, config, output, args.max_new_tasks, args.retry_failed)
    except Exception as exc:
        atomic_json(output / "run_failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                  "elapsed_wall_seconds": round(time.monotonic() - started, 3)})
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        long_data["archive"].close()
    result["invocation_wall_seconds"] = round(time.monotonic() - started, 3)
    archive = package_artifacts(output, "oracle_local_p1_v1_bundle.zip",
                                {"version": VERSION, "source_sha256": config["source_sha256"],
                                 "long_bundle_sha256": long_data["sha256"]})
    result["bundle"] = str(archive)
    result["bundle_sha256"] = sha256_file(archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
