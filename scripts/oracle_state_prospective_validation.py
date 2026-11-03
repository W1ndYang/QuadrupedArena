"""Prospectively test a frozen local Oracle on an unused synthetic vision seed.

Four previously chosen action prefixes are replayed with a new seed. The 8 s
state is discovered first, then eligible exact-root prefixes receive paired
baseline/scan full-game runs. Policy actions are computed from the frozen
cross-validated fit without consulting the new outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from arena_paths import workspace_root
from finite_state_optimizer import feature_bin
from oracle_auto_local_solver import optimize
from oracle_pilot import DEFAULT_CONFIG, atomic_json, digest, read, resolve_config, run_task, task_base
from oracle_pilot_discovery import boundary_evidence
from oracle_pilot_v21 import package_artifacts, sha256_file
from oracle_state_refinement_actions import action_outcome, checked_discovery
from oracle_state_refinement_solver import (checked_actions, checked_support, model_action,
                                            solve)

VERSION = "oracle_state_prospective_validation_v1"
DEFAULT_FULL = workspace_root() / "experiments/oracle-local-p1-full-game/oracle_local_p1_full_game_bundle.zip"
DEFAULT_P2 = workspace_root() / "experiments/oracle-p2-second-stage/oracle_p2_second_stage_bundle.zip"
DEFAULT_DISCOVERY = (workspace_root() / "experiments/oracle-state-refinement-discovery/"
                     "oracle_state_refinement_discovery_bundle.zip")
DEFAULT_ACTIONS = (workspace_root() / "experiments/oracle-state-refinement-actions/"
                   "oracle_state_refinement_actions_bundle.zip")
DEFAULT_SUPPORT = (workspace_root() / "experiments/oracle-state-refinement-support/"
                   "oracle_state_refinement_support_bundle.zip")
DEFAULT_OUTPUT = workspace_root() / "experiments/oracle-state-prospective-validation"


def frozen_experiment(config, full_path, p2_path, discovery_path, actions_path, support_path):
    prior = optimize(config, full_path, p2_path)
    discovery = checked_discovery(discovery_path, config)
    actions = checked_actions(actions_path, discovery, config)
    support = checked_support(support_path, prior, full_path, config)
    solved = solve(config, prior, discovery, actions, support)
    fit = solved["cross_validated_fit"]
    cv = solved["cross_validation"]
    if (not solved["complete_action_support"] or fit["selected_bits"] != 1 or
            fit["predicate"]["id"] != "P1:y:+0.5r" or
            config["test_synthetic_seeds"][0] in config["training_synthetic_seeds"]):
        raise ValueError("Cross-validated policy or held-out seed is not the frozen candidate")
    return {"fit": fit, "cross_validation": cv, "prior_fit": prior["state_fit"],
            "root_state_id": fit["base_state_id"], "scan_action_id": discovery["scan_action_id"],
            "baseline_action_id": config["continuation_action_id"],
            "synthetic_seed": config["test_synthetic_seeds"][0],
            "plans": discovery["plans"], "input_bundle_sha256": solved["input_bundle_sha256"]}


def discovery_task(config, frozen, plan):
    task = task_base(config, "discover", frozen["synthetic_seed"])
    task.update(experiment_version=VERSION, prefix_id=plan["prefix_id"],
                prefix_action_ids=plan["prefix_action_ids"], max_steps=4,
                frozen_fit_sha256=digest(frozen["fit"]))
    return task


def discovered_state(record, frozen):
    raw = Path(record["raw_dir"])
    result = record.get("result") or {}
    boundaries = read(raw / "boundaries.json") if (raw / "boundaries.json").exists() else []
    frames = read(raw / "frame_index.json") if (raw / "frame_index.json").exists() else []
    if (record["status"] != "passed" or result.get("stop_reason") != "planned_boundary" or
            len(boundaries) != 5):
        raise RuntimeError(f"Prospective prefix did not reach verified 8 s: {record['task_key']}")
    xy = [frame["own_xy"] for frame in frames if abs(frame["simulation_seconds"] - 8.0) < 1e-8]
    if len(xy) != 1:
        raise RuntimeError(f"Prospective prefix has no unique public 8 s position: {record['task_key']}")
    boundary = boundaries[4]
    fields = boundary["limited"]["fields"]
    eligible = (boundary["limited"]["status"] == "valid" and not boundary["finished"] and
                boundary["limited"]["state_id"] == frozen["root_state_id"] and
                fields.get("target.active") == "P2" and fields.get("A.region_id") == "target_P1" and
                fields.get("A.hp", 0) > 0)
    return {"prefix_id": record["task"]["prefix_id"], "discovery_task_key": record["task_key"],
            "own_xy": xy[0], "score_at_8s": boundary["score"],
            "r2l_state_id": boundary["limited"]["state_id"],
            "eligible_exact_root": bool(eligible),
            "frozen_predicted_action_id": (model_action(frozen["fit"],
                {"own_xy": xy[0]}) if eligible else frozen["baseline_action_id"]),
            "old_x_predicted_action_id": (model_action(frozen["prior_fit"],
                {"own_xy": xy[0]}) if eligible else frozen["baseline_action_id"]),
            "expected_boundaries": [{**boundary_evidence(b), "collapsed": b["collapsed"]}
                                    for b in boundaries]}


def paired_task(config, frozen, state, prefix_action_ids, action):
    baseline = frozen["baseline_action_id"]
    mode = "oracle_local_baseline_prefix" if action == baseline else "oracle_local_eval_prefix"
    task = task_base(config, mode, frozen["synthetic_seed"])
    task.update(experiment_version=VERSION, prefix_id=state["prefix_id"],
                sample_id=state["discovery_task_key"] + "_04",
                prefix_action_ids=prefix_action_ids,
                expected_boundaries=state["expected_boundaries"], max_steps=20,
                target_state_id=frozen["root_state_id"],
                feature_threshold_x=frozen["prior_fit"]["predicate"]["threshold"],
                local_policy_table=({"p1_west_edge": action, "p1_other_exit": action}
                                    if action != baseline else {}),
                frozen_fit_sha256=digest(frozen["fit"]),
                source_discovery_task_key=state["discovery_task_key"])
    return task


def compare_prospective(frozen, discovered, outcomes):
    by_key = {(row["prefix_id"], row["action_id"]): row for row in outcomes}
    pairs = []
    for state in discovered:
        if not state["eligible_exact_root"]:
            continue
        baseline, scan = (by_key[(state["prefix_id"], action)] for action in
                          (frozen["baseline_action_id"], frozen["scan_action_id"]))
        if baseline["local_bin"] != scan["local_bin"]:
            raise ValueError(f"Prospective pair disagrees on local state: {state['prefix_id']}")
        selectable = baseline["selectable"] and scan["selectable"]
        predicted = state["frozen_predicted_action_id"]
        policy_outcome = baseline if predicted == frozen["baseline_action_id"] else scan
        pairs.append({"prefix_id": state["prefix_id"], "public_own_xy": state["own_xy"],
                      "frozen_predicted_action_id": predicted,
                      "old_x_predicted_action_id": state["old_x_predicted_action_id"],
                      "baseline_outcome": baseline, "scan_outcome": scan,
                      "both_actions_selectable": bool(selectable),
                      "frozen_policy_advantage": (round(policy_outcome["official_return"] -
                                                        baseline["official_return"], 6)
                                                  if selectable else None),
                      "all_scan_advantage": (round(scan["official_return"] -
                                                   baseline["official_return"], 6)
                                             if selectable else None)})
    valid = [pair for pair in pairs if pair["both_actions_selectable"]]
    return {"version": VERSION, "frozen_before_new_seed_outcomes": True,
            "synthetic_seed": frozen["synthetic_seed"],
            "discovered_prefixes": len(discovered), "eligible_exact_root": len(pairs),
            "selectable_pairs": len(valid), "pairs": pairs,
            "mean_frozen_policy_advantage": (sum(p["frozen_policy_advantage"] for p in valid) / len(valid)
                                              if valid else None),
            "mean_all_scan_advantage": (sum(p["all_scan_advantage"] for p in valid) / len(valid)
                                        if valid else None),
            "evaluation_limit": "One unseen synthetic seed on previously selected action prefixes; not a new game seed or random scene sample"}


def run_validation(config, frozen, output, max_new_tasks=12):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"version": VERSION, "source_sha256": config["source_sha256"],
                  "input_bundle_sha256": frozen["input_bundle_sha256"],
                  "cross_validated_fit_sha256": digest(frozen["fit"]),
                  "game_seed": config["game_seed"], "prospective_synthetic_seed": frozen["synthetic_seed"],
                  "prefix_ids": [p["prefix_id"] for p in frozen["plans"]]}
    path = output / "experiment_provenance.json"
    if path.exists() and read(path) != provenance:
        raise ValueError("Prospective output belongs to another frozen policy or source")
    atomic_json(path, provenance)
    atomic_json(output / "resolved_config.json", config)
    atomic_json(output / "frozen_state_fit.json", frozen["fit"])
    atomic_json(output / "frozen_cross_validation.json", frozen["cross_validation"])
    atomic_json(output / "frozen_prefix_plans.json", frozen["plans"])
    budget = 0
    discovered, outcomes = [], []
    for plan in frozen["plans"]:
        task = discovery_task(config, frozen, plan)
        key = digest(task)[:20]
        record_path = output / "tasks" / f"{key}.json"
        if not record_path.exists() or read(record_path)["status"] != "passed":
            if budget >= max_new_tasks:
                atomic_json(output / "capture_progress.json", {"stage": "incomplete",
                            "completed_discovery": len(discovered), "completed_paired_tasks": len(outcomes),
                            "new_tasks_this_run": budget})
                return {"version": VERSION, "status": "incomplete",
                        "completed_discovery": len(discovered), "completed_paired_tasks": len(outcomes)}
            budget += 1
        record = run_task(output, task, retry_failed=True)
        state = discovered_state(record, frozen)
        discovered.append(state)
        print(f"prospective_discover {len(discovered)}/4 {plan['prefix_id']} "
              f"eligible={state['eligible_exact_root']}", flush=True)
        if state["eligible_exact_root"]:
            fake_discovery = {"fit": {"base_state_id": frozen["root_state_id"],
                                      "baseline_action_id": frozen["baseline_action_id"],
                                      "predicate": {"threshold": frozen["prior_fit"]["predicate"]["threshold"]}}}
            for action in (frozen["baseline_action_id"], frozen["scan_action_id"]):
                task = paired_task(config, frozen, state, plan["prefix_action_ids"], action)
                key = digest(task)[:20]
                record_path = output / "tasks" / f"{key}.json"
                if not record_path.exists() or read(record_path)["status"] != "passed":
                    if budget >= max_new_tasks:
                        atomic_json(output / "capture_progress.json", {"stage": "incomplete",
                                    "completed_discovery": len(discovered),
                                    "completed_paired_tasks": len(outcomes), "new_tasks_this_run": budget})
                        return {"version": VERSION, "status": "incomplete",
                                "completed_discovery": len(discovered),
                                "completed_paired_tasks": len(outcomes)}
                    budget += 1
                record = run_task(output, task, retry_failed=True)
                outcome = action_outcome(record, action, state, fake_discovery)
                outcomes.append(outcome)
                print(f"prospective_pair {len(outcomes)} {plan['prefix_id']} "
                      f"{'baseline' if action == frozen['baseline_action_id'] else 'scan'} "
                      f"selectable={outcome['selectable']}", flush=True)
        atomic_json(output / "capture_progress.json", {"stage": "running",
                    "completed_discovery": len(discovered), "completed_paired_tasks": len(outcomes),
                    "new_tasks_this_run": budget, "last_task_wall_seconds": record["wall_seconds"]})
    comparison = compare_prospective(frozen, discovered, outcomes)
    atomic_json(output / "discovered_states.json", discovered)
    atomic_json(output / "paired_outcomes.json", outcomes)
    atomic_json(output / "prospective_validation.json", comparison)
    summary = {"version": VERSION, "status": "passed", "discovered_prefixes": len(discovered),
               "eligible_exact_root": comparison["eligible_exact_root"],
               "selectable_pairs": comparison["selectable_pairs"],
               "mean_frozen_policy_advantage": comparison["mean_frozen_policy_advantage"],
               "mean_all_scan_advantage": comparison["mean_all_scan_advantage"]}
    atomic_json(output / "run_summary.json", summary)
    atomic_json(output / "capture_progress.json", {"stage": "complete",
                "completed_discovery": len(discovered), "completed_paired_tasks": len(outcomes),
                "new_tasks_this_run": budget})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-game-bundle", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--p2-bundle", type=Path, default=DEFAULT_P2)
    parser.add_argument("--discovery-bundle", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--actions-bundle", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--support-bundle", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tasks", type=int, default=12)
    args = parser.parse_args(argv)
    if args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    config = resolve_config(DEFAULT_CONFIG)
    frozen = frozen_experiment(config, args.full_game_bundle, args.p2_bundle,
                               args.discovery_bundle, args.actions_bundle, args.support_bundle)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    invocation_path = output / "invocations.json"
    invocations = read(invocation_path) if invocation_path.exists() else []
    invocations.append({"utc_started": datetime.now(timezone.utc).isoformat(),
                        "command": ["bash", "scripts/run_oracle_pilot_ubuntu.sh", "oracle_state_prospective",
                                    *(argv if argv is not None else sys.argv[1:])]})
    atomic_json(invocation_path, invocations)
    started = time.monotonic()
    try:
        result = run_validation(config, frozen, output, args.max_new_tasks)
    except Exception as exc:
        atomic_json(output / "run_failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                  "elapsed_wall_seconds": round(time.monotonic() - started, 3)})
        result = {"version": VERSION, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if result["status"] == "passed":
        (output / "run_failure.json").unlink(missing_ok=True)
    result["invocation_wall_seconds"] = round(time.monotonic() - started, 3)
    archive = package_artifacts(output, "oracle_state_prospective_validation_bundle.zip",
                                {"version": VERSION, "source_sha256": config["source_sha256"],
                                 "cross_validated_fit_sha256": digest(frozen["fit"])})
    result["bundle"] = str(archive)
    result["bundle_sha256"] = sha256_file(archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
