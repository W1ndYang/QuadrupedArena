"""Resumable paired validation across three fresh formal game seeds.

The policy and root-state signature are frozen from earlier evidence. Each
game seed uses the same held-out synthetic vision stream, discovers the 8 s
state, then measures baseline and scan only for a
compatible observable root. Outcomes are reported by seed and prefix; this
collector does not refit the policy from its own results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from arena_paths import workspace_root
from oracle_pilot import DEFAULT_CONFIG, atomic_json, digest, read, resolve_config
from oracle_pilot_v21 import package_artifacts, sha256_file
from oracle_state_prospective_validation import (DEFAULT_ACTIONS, DEFAULT_DISCOVERY, DEFAULT_FULL,
                                                  DEFAULT_P2, DEFAULT_SUPPORT, frozen_experiment,
                                                  run_validation)

VERSION = "oracle_state_seed_sweep_v1"
DEFAULT_OUTPUT = workspace_root() / "experiments/oracle-state-seed-sweep"


def seed_plan(config):
    # Hold the vision seed fixed so game-seed variation is the experimental axis.
    game_seeds = [config["game_seed"] + offset for offset in (2, 3, 4)]
    synthetic_seed = config["test_synthetic_seeds"][0]
    if (len(set(game_seeds)) != 3 or synthetic_seed in config["training_synthetic_seeds"]):
        raise ValueError("Seed sweep requires three new game seeds and one held-out vision seed")
    return [{"game_seed": game, "synthetic_seed": synthetic_seed} for game in game_seeds]


def aggregate_completed(output, plan):
    by_prefix = defaultdict(list)
    seeds = []
    for item in plan:
        path = output / f"game_{item['game_seed']}" / "prospective_validation.json"
        result = read(path)
        if (result["version"] != VERSION or result["synthetic_seed"] != item["synthetic_seed"] or
                result["discovered_prefixes"] != 4):
            raise ValueError(f"Completed seed report differs from frozen plan: {item}")
        seeds.append({"game_seed": item["game_seed"], "synthetic_seed": item["synthetic_seed"],
                      "eligible_exact_root": result["eligible_exact_root"],
                      "eligible_compatible_root": result["eligible_compatible_root"],
                      "selectable_pairs": result["selectable_pairs"],
                      "mean_frozen_policy_advantage": result["mean_frozen_policy_advantage"],
                      "mean_all_scan_advantage": result["mean_all_scan_advantage"]})
        for pair in result["pairs"]:
            if pair["both_actions_selectable"]:
                by_prefix[pair["prefix_id"]].append({"game_seed": item["game_seed"],
                    "frozen_policy_advantage": pair["frozen_policy_advantage"],
                    "all_scan_advantage": pair["all_scan_advantage"],
                    "public_own_xy": pair["public_own_xy"]})
    prefix_rows = []
    all_valid = []
    for prefix_id, observations in sorted(by_prefix.items()):
        all_valid.extend(observations)
        prefix_rows.append({"prefix_id": prefix_id, "paired_seeds": len(observations),
                            "mean_frozen_policy_advantage": sum(
                                row["frozen_policy_advantage"] for row in observations) / len(observations),
                            "mean_all_scan_advantage": sum(
                                row["all_scan_advantage"] for row in observations) / len(observations),
                            "observations": observations})
    return {"version": VERSION, "seed_results": seeds,
            "paired_seed_prefix_visits": len(all_valid), "prefix_results": prefix_rows,
            "mean_frozen_policy_advantage": (sum(row["frozen_policy_advantage"] for row in all_valid) /
                                              len(all_valid) if all_valid else None),
            "mean_all_scan_advantage": (sum(row["all_scan_advantage"] for row in all_valid) /
                                        len(all_valid) if all_valid else None),
            "interpretation_limit": "Repeated prefixes across game seeds share geometry and are clustered observations, not independent position samples"}


def run_sweep(config, frozen, plan, output, max_new_tasks=12):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"version": VERSION, "source_sha256": config["source_sha256"],
                  "input_bundle_sha256": frozen["input_bundle_sha256"],
                  "frozen_fit_sha256": digest(frozen["fit"]),
                  "root_state_signature_sha256": digest(frozen["root_state_signature"]),
                  "seed_plan": plan,
                  "prefix_ids": [item["prefix_id"] for item in frozen["plans"]]}
    path = output / "experiment_provenance.json"
    if path.exists() and read(path) != provenance:
        raise ValueError("Seed sweep output belongs to another source or frozen plan")
    atomic_json(path, provenance)
    atomic_json(output / "frozen_state_fit.json", frozen["fit"])
    atomic_json(output / "seed_plan.json", plan)
    remaining = max_new_tasks
    seed_progress = []
    for item in plan:
        seed_frozen = {**frozen, "version": VERSION,
                       "synthetic_seed": item["synthetic_seed"],
                       "allow_rule_fingerprint_transfer": True}
        runtime_config = {**config, "game_seed": item["game_seed"]}
        seed_output = output / f"game_{item['game_seed']}"
        result = run_validation(runtime_config, seed_frozen, seed_output,
                                max_new_tasks=remaining)
        progress = read(seed_output / "capture_progress.json")
        remaining -= progress["new_tasks_this_run"]
        seed_progress.append({"game_seed": item["game_seed"],
                              "synthetic_seed": item["synthetic_seed"],
                              "status": result["status"],
                              "discovered_prefixes": result.get("discovered_prefixes",
                                                              result.get("completed_discovery")),
                              "selectable_pairs": result.get("selectable_pairs")})
        atomic_json(output / "capture_progress.json", {"stage": "running" if result["status"] == "passed"
                    else "incomplete", "seed_progress": seed_progress,
                    "new_tasks_this_run": max_new_tasks - remaining,
                    "remaining_seeds": len(plan) - sum(x["status"] == "passed" for x in seed_progress)})
        if result["status"] != "passed":
            return {"version": VERSION, "status": "incomplete",
                    "completed_seeds": sum(x["status"] == "passed" for x in seed_progress),
                    "new_tasks_this_run": max_new_tasks - remaining}
    aggregate = aggregate_completed(output, plan)
    atomic_json(output / "seed_sweep_analysis.json", aggregate)
    summary = {"version": VERSION, "status": "passed", "completed_seeds": len(plan),
               "paired_seed_prefix_visits": aggregate["paired_seed_prefix_visits"],
               "mean_frozen_policy_advantage": aggregate["mean_frozen_policy_advantage"],
               "mean_all_scan_advantage": aggregate["mean_all_scan_advantage"]}
    atomic_json(output / "run_summary.json", summary)
    atomic_json(output / "capture_progress.json", {"stage": "complete", "seed_progress": seed_progress,
                "new_tasks_this_run": max_new_tasks - remaining, "remaining_seeds": 0})
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
    plan = seed_plan(config)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    invocation_path = output / "invocations.json"
    invocations = read(invocation_path) if invocation_path.exists() else []
    invocations.append({"utc_started": datetime.now(timezone.utc).isoformat(),
                        "command": ["bash", "scripts/run_oracle_pilot_ubuntu.sh", "oracle_state_seed_sweep",
                                    *(argv if argv is not None else sys.argv[1:])]})
    atomic_json(invocation_path, invocations)
    started = time.monotonic()
    try:
        result = run_sweep(config, frozen, plan, output, args.max_new_tasks)
    except Exception as exc:
        atomic_json(output / "run_failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                  "elapsed_wall_seconds": round(time.monotonic() - started, 3)})
        result = {"version": VERSION, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if result["status"] == "passed":
        (output / "run_failure.json").unlink(missing_ok=True)
    result["invocation_wall_seconds"] = round(time.monotonic() - started, 3)
    archive = package_artifacts(output, "oracle_state_seed_sweep_bundle.zip",
                                {"version": VERSION, "source_sha256": config["source_sha256"],
                                 "frozen_fit_sha256": digest(frozen["fit"])})
    result["bundle"] = str(archive)
    result["bundle_sha256"] = sha256_file(archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
