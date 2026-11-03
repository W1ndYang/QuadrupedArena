"""Resume a 10 s paired action probe from verified full-game baseline runs.

The source bundle supplies 8 s prefixes and complete baseline trajectories.
The probe follows the same baseline action at 8 s, verifies the resulting
10 s state exactly, changes only the step-5 action to scan, and continues the
formal game to termination. Blocked actions remain in the record but do not
enter policy comparisons.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

from arena_paths import workspace_root
from oracle_pilot import DEFAULT_CONFIG, atomic_json, digest, read, resolve_config, run_task, task_base
from oracle_pilot_discovery import boundary_evidence
from oracle_pilot_runtime import strategy_action_executable
from oracle_pilot_v21 import package_artifacts, sha256_file
from oracle_state_seed_analysis import checked_bundle

VERSION = "oracle_late_decision_probe_v1"
DEFAULT_BASELINE = (workspace_root() / "experiments/oracle-state-seed-holdout/"
                    "oracle_state_seed_sweep_bundle.zip")
DEFAULT_OUTPUT = workspace_root() / "experiments/oracle-late-decision-probe"
TARGET_STEP = 5


def contexts_from_bundle(path, config):
    frozen, pairs, bundle_sha = checked_bundle(path)
    with zipfile.ZipFile(path) as archive:
        provenance = json.loads(archive.read("experiment_provenance.json"))
        seed_plan = json.loads(archive.read("seed_plan.json"))
        if (provenance["source_sha256"] != config["source_sha256"] or
                sorted(pairs) != [2026102112, 2026102113, 2026102114] or
                {item["synthetic_seed"] for item in seed_plan} != {20261101}):
            raise ValueError("The late probe requires the verified, unchanged forward sweep")
        contexts = []
        for seed in sorted(pairs):
            states = {item["prefix_id"]: item for item in
                      json.loads(archive.read(f"game_{seed}/discovered_states.json"))}
            for pair in pairs[seed]:
                prefix_id = pair["prefix_id"]
                state = states[prefix_id]
                base = pair["baseline_outcome"]
                key = base["task_key"]
                root = f"game_{seed}/raw/{key}/"
                boundaries = json.loads(archive.read(root + "boundaries.json"))
                decisions = json.loads(archive.read(root + "decisions.json"))
                task_record = json.loads(archive.read(f"game_{seed}/tasks/{key}.json"))
                task = task_record["task"]
                if (len(boundaries) <= TARGET_STEP or len(decisions) <= TARGET_STEP or
                        not base["selectable"] or
                        task["game_seed"] != seed or task["synthetic_seed"] != 20261101 or
                        task["prefix_id"] != prefix_id or
                        task["expected_boundaries"] != state["expected_boundaries"] or
                        boundaries[TARGET_STEP]["limited"]["status"] != "valid" or
                        boundaries[TARGET_STEP]["finished"] is not None or
                        decisions[4]["action_id"] != frozen["baseline_action_id"] or
                        decisions[TARGET_STEP]["action_id"] != frozen["baseline_action_id"]):
                    raise ValueError(f"Baseline cannot support a 10 s branch: {seed}/{prefix_id}")
                contexts.append({"game_seed": seed, "synthetic_seed": task["synthetic_seed"],
                                 "prefix_id": prefix_id, "prefix_action_ids": task["prefix_action_ids"],
                                 "expected_boundaries": state["expected_boundaries"],
                                 "baseline_task_key": key,
                                 "target_state_id": boundaries[TARGET_STEP]["limited"]["state_id"],
                                 "expected_target_boundary": {**boundary_evidence(boundaries[TARGET_STEP]),
                                                              "collapsed": boundaries[TARGET_STEP]["collapsed"]},
                                 "score_at_target": boundaries[TARGET_STEP]["score"],
                                 "baseline_final_score": base["observed_final_score_diagnostic"]})
    scans = {pair["scan_outcome"]["action_id"] for visit in pairs.values() for pair in visit}
    if len(scans) != 1 or len(contexts) != 12:
        raise ValueError("The forward sweep does not contain 12 paired contexts")
    return contexts, frozen["baseline_action_id"], next(iter(scans)), bundle_sha


def late_task(config, context, scan_action_id, baseline_bundle_sha):
    runtime_config = {**config, "game_seed": context["game_seed"]}
    task = task_base(runtime_config, "oracle_eval_prefix", context["synthetic_seed"])
    task.update(experiment_version=VERSION, prefix_id=context["prefix_id"],
                sample_id=context["baseline_task_key"] + "_05",
                source_baseline_bundle_sha256=baseline_bundle_sha,
                source_baseline_task_key=context["baseline_task_key"],
                prefix_action_ids=context["prefix_action_ids"],
                expected_boundaries=context["expected_boundaries"],
                max_steps=20,
                policy_table={str(TARGET_STEP): {context["target_state_id"]: scan_action_id}})
    return task


def checked_outcome(record, context, baseline_action_id, scan_action_id):
    raw = Path(record["raw_dir"])
    result = record.get("result") or {}
    boundaries = read(raw / "boundaries.json") if (raw / "boundaries.json").exists() else []
    decisions = read(raw / "decisions.json") if (raw / "decisions.json").exists() else []
    frames = read(raw / "frame_index.json") if (raw / "frame_index.json").exists() else []
    target_xy = [frame["own_xy"] for frame in frames
                 if abs(frame["simulation_seconds"] - TARGET_STEP * 2.0) < 1e-8]
    complete_game = (result.get("stop_reason") == "terminal" or
                     (result.get("stop_reason") in ("horizon", "duration") and
                      result.get("simulation_seconds", 0) >= 40.0 - 1e-8))
    if (record["status"] != "passed" or not complete_game or
            len(boundaries) <= TARGET_STEP or len(decisions) <= TARGET_STEP or
            len(target_xy) != 1 or
            decisions[4]["action_id"] != baseline_action_id or
            decisions[TARGET_STEP]["action_id"] != scan_action_id or
            decisions[TARGET_STEP]["source"] != "oracle_table" or
            boundaries[TARGET_STEP]["limited"]["state_id"] != context["target_state_id"]):
        raise ValueError(f"10 s action or replay diverged: {record['task_key']}")
    actual = {**boundary_evidence(boundaries[TARGET_STEP]),
              "collapsed": boundaries[TARGET_STEP]["collapsed"]}
    if actual != context["expected_target_boundary"]:
        raise ValueError(f"10 s pre-action boundary differs from baseline: {record['task_key']}")
    decision = decisions[TARGET_STEP]
    selectable = strategy_action_executable(decision)
    return {"game_seed": context["game_seed"], "prefix_id": context["prefix_id"],
            "baseline_task_key": context["baseline_task_key"], "scan_task_key": record["task_key"],
            "target_state_id": context["target_state_id"],
            "public_target_fields": boundaries[TARGET_STEP]["limited"]["fields"],
            "public_own_xy": target_xy[0],
            "public_hp": boundaries[TARGET_STEP]["hp"][0],
            "score_at_target": context["score_at_target"],
            "baseline_final_score": context["baseline_final_score"],
            "baseline_official_return": round(context["baseline_final_score"] -
                                              context["score_at_target"], 6),
            "scan_final_score": result["official_score"],
            "validator_status": decision["validator_status"],
            "execution_status": decision["execution_status"],
            "compiled_action_skill": (decision.get("compiled_action") or {}).get("skill"),
            "compiled_skill_state": (decision.get("compiled_skill") or {}).get("state"),
            "scan_selectable": selectable,
            "scan_official_return": (round(result["official_score"] -
                                           context["score_at_target"], 6)
                                     if selectable else None),
            "official_scan_advantage": (round(result["official_score"] -
                                              context["baseline_final_score"], 6)
                                         if selectable else None),
            "terminal_reason": result["terminal_reason"]}


def run_probe(config, contexts, baseline_action_id, scan_action_id, bundle_sha,
              output, max_new_tasks):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"version": VERSION, "source_sha256": config["source_sha256"],
                  "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "baseline_bundle_sha256": bundle_sha, "target_step": TARGET_STEP,
                  "baseline_action_id": baseline_action_id, "scan_action_id": scan_action_id,
                  "contexts": [{"game_seed": c["game_seed"], "prefix_id": c["prefix_id"],
                                "baseline_task_key": c["baseline_task_key"],
                                "target_state_id": c["target_state_id"]} for c in contexts]}
    provenance_path = output / "experiment_provenance.json"
    if provenance_path.exists() and read(provenance_path) != provenance:
        raise ValueError("Output directory belongs to another late probe")
    atomic_json(provenance_path, provenance)
    atomic_json(output / "probe_plan.json", contexts)
    new_tasks = 0
    outcomes = []
    for context in contexts:
        task = late_task(config, context, scan_action_id, bundle_sha)
        key = digest(task)[:20]
        record_path = output / "tasks" / f"{key}.json"
        if not record_path.exists() or read(record_path)["status"] != "passed":
            if new_tasks >= max_new_tasks:
                atomic_json(output / "capture_progress.json", {"stage": "incomplete",
                            "completed_tasks": len(outcomes), "remaining_tasks": len(contexts) - len(outcomes),
                            "new_tasks_this_run": new_tasks})
                return {"version": VERSION, "status": "incomplete",
                        "completed_tasks": len(outcomes), "new_tasks_this_run": new_tasks}
            new_tasks += 1
        record = run_task(output, task, retry_failed=True)
        outcome = checked_outcome(record, context, baseline_action_id, scan_action_id)
        outcomes.append(outcome)
        atomic_json(output / "capture_progress.json", {"stage": "running",
                    "completed_tasks": len(outcomes), "remaining_tasks": len(contexts) - len(outcomes),
                    "new_tasks_this_run": new_tasks})
        print(f"late_probe {len(outcomes)}/{len(contexts)} seed={context['game_seed']} "
              f"prefix={context['prefix_id']} selectable={outcome['scan_selectable']}", flush=True)
    atomic_json(output / "late_action_outcomes.json", outcomes)
    valid = [row for row in outcomes if row["scan_selectable"]]
    summary = {"version": VERSION, "status": "passed", "completed_tasks": len(outcomes),
               "selectable_pairs": len(valid),
               "mean_scan_advantage": (sum(row["official_scan_advantage"] for row in valid) / len(valid)
                                       if valid else None)}
    atomic_json(output / "run_summary.json", summary)
    atomic_json(output / "capture_progress.json", {"stage": "complete",
                "completed_tasks": len(outcomes), "remaining_tasks": 0,
                "new_tasks_this_run": new_tasks})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bundle", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tasks", type=int, default=6)
    args = parser.parse_args(argv)
    if args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    config = resolve_config(DEFAULT_CONFIG)
    contexts, baseline, scan, bundle_sha = contexts_from_bundle(args.baseline_bundle, config)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    invocation_path = output / "invocations.json"
    invocations = read(invocation_path) if invocation_path.exists() else []
    invocations.append({"utc_started": datetime.now(timezone.utc).isoformat(),
                        "command": ["bash", "scripts/run_oracle_pilot_ubuntu.sh", "oracle_late_decision_probe",
                                    *(argv if argv is not None else sys.argv[1:])]})
    atomic_json(invocation_path, invocations)
    started = time.monotonic()
    try:
        result = run_probe(config, contexts, baseline, scan, bundle_sha,
                           output, args.max_new_tasks)
    except Exception as exc:
        atomic_json(output / "run_failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                  "elapsed_wall_seconds": round(time.monotonic() - started, 3)})
        result = {"version": VERSION, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if result["status"] == "passed":
        (output / "run_failure.json").unlink(missing_ok=True)
    result["invocation_wall_seconds"] = round(time.monotonic() - started, 3)
    archive = package_artifacts(output, "oracle_late_decision_probe_bundle.zip",
                                {"version": VERSION, "source_sha256": config["source_sha256"],
                                 "baseline_bundle_sha256": bundle_sha})
    result["bundle"] = str(archive)
    result["bundle_sha256"] = sha256_file(archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
