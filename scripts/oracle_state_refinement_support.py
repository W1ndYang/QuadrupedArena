"""Complete missing root-action support on two earlier P1/P2 training prefixes.

Each task strictly replays its original 0..8 s prefix, tries OBSERVE(scan),
and follows the fixed baseline to the formal terminal. Partial tasks retry on
the next invocation; a blocked action remains diagnostic evidence only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import zipfile

from arena_paths import workspace_root
from oracle_auto_local_solver import optimize
from oracle_local_p1_v1 import formal_action
from oracle_pilot import DEFAULT_CONFIG, atomic_json, digest, read, resolve_config, run_task, task_base
from oracle_pilot_v21 import package_artifacts, sha256_file
from oracle_state_refinement_actions import action_outcome

VERSION = "oracle_state_refinement_support_v1"
DEFAULT_FULL = workspace_root() / "experiments/oracle-local-p1-full-game/oracle_local_p1_full_game_bundle.zip"
DEFAULT_P2 = workspace_root() / "experiments/oracle-p2-second-stage/oracle_p2_second_stage_bundle.zip"
DEFAULT_OUTPUT = workspace_root() / "experiments/oracle-state-refinement-support"


def support_plans(prior, full_path, config):
    baseline = config["continuation_action_id"]
    scan = formal_action(config, "OBSERVE", {"direction": "scan", "risk": "safe"})
    target = prior["state_fit"]["base_state_id"]
    threshold = prior["state_fit"]["predicate"]["threshold"]
    paired = {item["sample_id"]: item for item in prior["training_rows"]}
    plans = []
    with zipfile.ZipFile(full_path) as archive:
        source_pairs = json.loads(archive.read("full_game_paired_validation.json"))["pairs"]
        for pair in source_pairs:
            row = paired.get(pair["sample_id"])
            if row is None or scan in row["returns"]:
                continue
            key = pair["baseline_task_key"]
            record = json.loads(archive.read(f"tasks/{key}.json"))
            task = record["task"]
            boundaries = json.loads(archive.read(f"raw/{key}/boundaries.json"))
            if (record["status"] != "passed" or task["mode"] != "oracle_local_baseline_prefix" or
                    task["sample_id"] != row["sample_id"] or task["prefix_id"] != row["prefix_id"] or
                    task["target_state_id"] != target or task["feature_threshold_x"] != threshold or
                    task["continuation_action_id"] != baseline or len(task["expected_boundaries"]) != 5 or
                    boundaries[4]["limited"]["state_id"] != target or
                    boundaries[4]["score"] != row["score_at_8s"]):
                raise ValueError(f"Original baseline task is inconsistent: {key}")
            plans.append({"sample_id": row["sample_id"], "prefix_id": row["prefix_id"],
                          "old_baseline_task_key": key, "prefix_action_ids": task["prefix_action_ids"],
                          "expected_boundaries": task["expected_boundaries"],
                          "score_at_8s": row["score_at_8s"], "own_xy": row["own_xy"]})
    if len(plans) != 2:
        raise ValueError("Expected exactly two original training prefixes without scan support")
    return {"plans": plans, "scan_action_id": scan, "target_state_id": target,
            "feature_threshold_x": threshold, "baseline_action_id": baseline}


def support_task(config, prior, support, plan):
    task = task_base(config, "oracle_local_eval_prefix", config["discovery_synthetic_seed"])
    task.update(experiment_version=VERSION, sample_id=plan["sample_id"],
                prefix_id=plan["prefix_id"], prefix_action_ids=plan["prefix_action_ids"],
                expected_boundaries=plan["expected_boundaries"], max_steps=20,
                target_state_id=support["target_state_id"],
                feature_threshold_x=support["feature_threshold_x"],
                local_policy_table={"p1_west_edge": support["scan_action_id"],
                                    "p1_other_exit": support["scan_action_id"]},
                source_full_game_sha256=prior["full_game_bundle_sha256"])
    return task


def run_collection(config, prior, support, output, max_new_tasks=2):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"version": VERSION, "source_sha256": config["source_sha256"],
                  "full_game_bundle_sha256": prior["full_game_bundle_sha256"],
                  "p2_bundle_sha256": prior["p2_bundle_sha256"],
                  "game_seed": config["game_seed"], "synthetic_seed": config["discovery_synthetic_seed"]}
    path = output / "experiment_provenance.json"
    if path.exists() and read(path) != provenance:
        raise ValueError("Support output belongs to another source or input")
    atomic_json(path, provenance)
    atomic_json(output / "resolved_config.json", config)
    atomic_json(output / "frozen_support_plan.json", support)
    budget = 0
    outcomes = []
    fake_discovery = {"fit": {"base_state_id": support["target_state_id"],
                              "baseline_action_id": support["baseline_action_id"],
                              "predicate": {"threshold": support["feature_threshold_x"]}}}
    for plan in support["plans"]:
        task = support_task(config, prior, support, plan)
        key = digest(task)[:20]
        record_path = output / "tasks" / f"{key}.json"
        if not record_path.exists() or read(record_path)["status"] != "passed":
            if budget >= max_new_tasks:
                atomic_json(output / "capture_progress.json", {"stage": "incomplete",
                            "completed_tasks": len(outcomes), "remaining_tasks": 2 - len(outcomes),
                            "new_tasks_this_run": budget})
                return {"version": VERSION, "status": "incomplete", "completed_tasks": len(outcomes)}
            budget += 1
        record = run_task(output, task, retry_failed=True)
        outcome = action_outcome(record, support["scan_action_id"], plan, fake_discovery)
        outcome["sample_id"] = plan["sample_id"]
        outcomes.append(outcome)
        atomic_json(output / "capture_progress.json", {"stage": "running",
                    "completed_tasks": len(outcomes), "remaining_tasks": 2 - len(outcomes),
                    "new_tasks_this_run": budget, "last_task_wall_seconds": record["wall_seconds"]})
        print(f"state_support {len(outcomes)}/2 {plan['prefix_id']} "
              f"selectable={outcome['selectable']}", flush=True)
    atomic_json(output / "support_outcomes.json", outcomes)
    summary = {"version": VERSION, "status": "passed", "completed_tasks": 2,
               "selectable_outcomes": sum(x["selectable"] for x in outcomes),
               "total_task_wall_seconds": round(sum(x["wall_seconds"] for x in outcomes), 3)}
    atomic_json(output / "run_summary.json", summary)
    atomic_json(output / "capture_progress.json", {"stage": "complete", "completed_tasks": 2,
                "remaining_tasks": 0, "new_tasks_this_run": budget})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-game-bundle", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--p2-bundle", type=Path, default=DEFAULT_P2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tasks", type=int, default=2)
    args = parser.parse_args(argv)
    if args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    config = resolve_config(DEFAULT_CONFIG)
    prior = optimize(config, args.full_game_bundle, args.p2_bundle)
    support = support_plans(prior, args.full_game_bundle, config)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    invocations_path = output / "invocations.json"
    invocations = read(invocations_path) if invocations_path.exists() else []
    invocations.append({"utc_started": datetime.now(timezone.utc).isoformat(),
                        "command": ["bash", "scripts/run_oracle_pilot_ubuntu.sh", "oracle_state_support",
                                    *(argv if argv is not None else sys.argv[1:])]})
    atomic_json(invocations_path, invocations)
    started = time.monotonic()
    try:
        result = run_collection(config, prior, support, output, args.max_new_tasks)
    except Exception as exc:
        atomic_json(output / "run_failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                  "elapsed_wall_seconds": round(time.monotonic() - started, 3)})
        result = {"version": VERSION, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if result["status"] == "passed":
        (output / "run_failure.json").unlink(missing_ok=True)
    result["invocation_wall_seconds"] = round(time.monotonic() - started, 3)
    archive = package_artifacts(output, "oracle_state_refinement_support_bundle.zip",
                                {"version": VERSION, "source_sha256": config["source_sha256"],
                                 "full_game_bundle_sha256": prior["full_game_bundle_sha256"]})
    result["bundle"] = str(archive)
    result["bundle_sha256"] = sha256_file(archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
