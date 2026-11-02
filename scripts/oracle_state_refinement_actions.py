"""Paired full-game action outcomes for discovery-selected P1/P2 prefixes.

The state predicates and their predictions are frozen before outcome capture.
This collector measures both root actions; it does not select a new policy.
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

from adaptive_motion import load_config
from arena_paths import PROJECT, workspace_root
from finite_state_optimizer import feature_bin, geometry_predicates
from oracle_local_state_v1 import p1_west_boundary
from oracle_pilot import DEFAULT_CONFIG, atomic_json, digest, read, resolve_config, run_task, task_base
from oracle_pilot_discovery import boundary_evidence
from oracle_pilot_runtime import strategy_action_executable
from oracle_pilot_v21 import package_artifacts, sha256_file

VERSION = "oracle_state_refinement_actions_v1"
DEFAULT_DISCOVERY = (workspace_root() / "experiments/oracle-state-refinement-discovery/"
                     "oracle_state_refinement_discovery_bundle.zip")
DEFAULT_OUTPUT = workspace_root() / "experiments/oracle-state-refinement-actions"


def checked_discovery(path, config):
    path = Path(path).expanduser().resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or archive.testzip() is not None or
                any(name.startswith("/") or ".." in Path(name).parts for name in names)):
            raise ValueError("Discovery ZIP has duplicate, unsafe, or corrupt members")
        manifest = json.loads(archive.read("manifest.json"))
        entries = manifest["files"]
        if manifest["file_count"] != len(entries):
            raise ValueError("Discovery manifest count mismatch")
        checks = {}
        for entry in entries:
            payload = archive.read(entry["path"])
            actual = hashlib.sha256(payload).hexdigest()
            if len(payload) != entry["bytes"] or actual != entry["sha256"]:
                raise ValueError(f"Discovery member checksum mismatch: {entry['path']}")
            checks[entry["path"]] = actual
        sums = dict(line.split("  ", 1)[::-1] for line in archive.read("SHA256SUMS").decode().splitlines())
        if (len(checks) != len(entries) or sums != checks or
                set(names) != set(checks) | {"manifest.json", "SHA256SUMS"}):
            raise ValueError("Discovery archive membership disagrees with manifest")
        provenance = json.loads(archive.read("experiment_provenance.json"))
        summary = json.loads(archive.read("run_summary.json"))
        analysis = json.loads(archive.read("state_discovery_analysis.json"))
        fit = json.loads(archive.read("frozen_state_fit.json"))
        if (summary.get("status") != "passed" or summary.get("completed_prefixes") != 12 or
                analysis.get("version") != "oracle_state_refinement_discovery_v1" or
                provenance["source_sha256"] != manifest["source_sha256"] or
                provenance["game_seed"] != config["game_seed"] or
                provenance["synthetic_seed"] != config["discovery_synthetic_seed"] or
                fit["base_state_id"] != analysis["root_r2l_state_id"] or
                fit["selected_bits"] != 1 or fit["baseline_action_id"] != config["continuation_action_id"]):
            raise ValueError("Discovery analysis, frozen fit, or experiment context changed")
        tied = analysis["equivalent_predicates"]
        geometry = {item["id"]: item for item in geometry_predicates(
            load_config(PROJECT / "configs" / config["rules_config"])["arena"]["task_points"])}
        if ([item["predicate"]["id"] for item in tied] !=
                ["P1:x:-0.5r", "P1:y:+0.5r"] or fit["predicate"] != tied[0]["predicate"] or
                any(item["predicate"] != geometry[item["predicate"]["id"]] for item in tied) or
                fit["predicate"]["threshold"] != p1_west_boundary(
                    load_config(PROJECT / "configs" / config["rules_config"]))):
            raise ValueError("The frozen competing state predicates changed")
        west_side = tied[0]["training_west_side"]
        scan = fit["leaves"][west_side]["action_id"]
        if (fit["leaves"]["high" if west_side == "low" else "low"]["action_id"] !=
                fit["baseline_action_id"] or
                not any(item["action_id"] == scan and item["skill"] == "OBSERVE" and
                        item["params"] == {"direction": "scan", "risk": "safe"}
                        for item in config["action_catalog"])):
            raise ValueError("Frozen action semantics changed")
        relevant = [row for row in analysis["rows"] if row.get("relevant_p1_to_p2") and
                    row.get("exact_prior_root_state")]
        if (len(relevant) != 4 or sum(row["predicate_disagreement"] for row in relevant) != 1 or
                relevant[0]["priority_rank"] != 0):
            raise ValueError("Expected one decisive prefix and three same-root coverage prefixes")
        plans = []
        for row in relevant:
            key = row["task_key"]
            record = json.loads(archive.read(f"tasks/{key}.json"))
            boundaries = json.loads(archive.read(f"raw/{key}/boundaries.json"))
            frames = json.loads(archive.read(f"raw/{key}/frame_index.json"))
            xy = [item["own_xy"] for item in frames if abs(item["simulation_seconds"] - 8) < 1e-8]
            if (record["task_key"] != key or record["status"] != "passed" or
                    record["task"]["mode"] != "discover" or
                    record["task"]["prefix_id"] != row["prefix_id"] or
                    record["task"]["synthetic_seed"] != provenance["synthetic_seed"] or
                    record["task"]["game_seed"] != provenance["game_seed"] or
                    record["task"]["source_sha256"] != provenance["source_sha256"] or
                    len(boundaries) != 5 or len(xy) != 1 or xy[0] != row["public_own_xy"] or
                    boundaries[4]["limited"]["state_id"] != fit["base_state_id"] or
                    boundaries[4]["score"] != row["official_score_at_8s"] or
                    any((feature_bin(xy[0], item["predicate"]) == item["training_west_side"]) !=
                        row["predicate_west_like"][item["predicate"]["id"]] for item in tied)):
                raise ValueError(f"Discovery row disagrees with raw evidence: {key}")
            plans.append({"prefix_id": row["prefix_id"], "discovery_task_key": key,
                          "prefix_action_ids": record["task"]["prefix_action_ids"],
                          "expected_boundaries": [{**boundary_evidence(b), "collapsed": b["collapsed"]}
                                                  for b in boundaries],
                          "own_xy": xy[0], "score_at_8s": boundaries[4]["score"],
                          "predicate_west_like": row["predicate_west_like"],
                          "predicate_disagreement": row["predicate_disagreement"]})
    return {"path": path, "sha256": sha256_file(path), "provenance": provenance,
            "fit": fit, "tied": tied, "scan_action_id": scan, "plans": plans}


def evaluation_task(config, discovery, plan, action):
    baseline = config["continuation_action_id"]
    mode = "oracle_local_baseline_prefix" if action == baseline else "oracle_local_eval_prefix"
    task = task_base(config, mode, config["discovery_synthetic_seed"])
    task.update(experiment_version=VERSION, prefix_id=plan["prefix_id"],
                sample_id=plan["discovery_task_key"] + "_04", source_discovery_sha256=discovery["sha256"],
                prefix_action_ids=plan["prefix_action_ids"], expected_boundaries=plan["expected_boundaries"],
                max_steps=20, target_state_id=discovery["fit"]["base_state_id"],
                feature_threshold_x=discovery["fit"]["predicate"]["threshold"],
                local_policy_table=({"p1_west_edge": action, "p1_other_exit": action}
                                    if action != baseline else {}))
    return task


def action_outcome(record, expected_action, plan, discovery):
    raw = Path(record["raw_dir"])
    decisions = read(raw / "decisions.json") if (raw / "decisions.json").exists() else []
    boundaries = read(raw / "boundaries.json") if (raw / "boundaries.json").exists() else []
    result = record.get("result") or {}
    if (record["status"] != "passed" or result.get("stop_reason") not in ("terminal", "planned_boundary") or
            len(boundaries) < 6 or len(decisions) < 5 or
            boundaries[4]["limited"]["state_id"] != discovery["fit"]["base_state_id"] or
            boundaries[4]["score"] != plan["score_at_8s"] or
            decisions[4]["action_id"] != expected_action):
        raise RuntimeError(f"Action task failed or strict replay differed: {record['task_key']}")
    first = decisions[4]
    local = boundaries[4].get("local_policy_state") or {}
    wanted_source = ("baseline" if expected_action == discovery["fit"]["baseline_action_id"] else
                     "local_table:" + local.get("bin", ""))
    if (first["source"] != wanted_source or first.get("local_policy_state") != local or
            local.get("base_state_id") != discovery["fit"]["base_state_id"] or
            local.get("decision_step") != 4 or
            local.get("threshold_x") != discovery["fit"]["predicate"]["threshold"]):
        raise RuntimeError(f"Root decision did not follow the frozen action plan: {record['task_key']}")
    selectable = strategy_action_executable(first)
    return {"prefix_id": plan["prefix_id"], "task_key": record["task_key"],
            "action_id": expected_action, "action_source": first["source"],
            "local_bin": local["bin"], "validator_status": first["validator_status"],
            "execution_status": first["execution_status"],
            "compiled_action_skill": (first.get("compiled_action") or {}).get("skill"),
            "compiled_skill_state": (first.get("compiled_skill") or {}).get("state"),
            "selectable": selectable, "observed_final_score_diagnostic": result["official_score"],
            "official_return": round(result["official_score"] - plan["score_at_8s"], 6)
                               if selectable else None,
            "terminal_reason": result["terminal_reason"], "wall_seconds": record["wall_seconds"]}


def compare_pairs(discovery, outcomes):
    by_prefix = {(x["prefix_id"], x["action_id"]): x for x in outcomes}
    baseline, scan = discovery["fit"]["baseline_action_id"], discovery["scan_action_id"]
    rows = []
    for plan in discovery["plans"]:
        a, b = (by_prefix[(plan["prefix_id"], action)] for action in (baseline, scan))
        if a["local_bin"] != b["local_bin"]:
            raise ValueError(f"Paired local state differs: {plan['prefix_id']}")
        advantage = (round(b["official_return"] - a["official_return"], 6)
                     if a["selectable"] and b["selectable"] else None)
        winner = (scan if advantage is not None and advantage > 0 else
                  baseline if advantage is not None else None)
        predictions = {item["predicate"]["id"]:
                       scan if plan["predicate_west_like"][item["predicate"]["id"]] else baseline
                       for item in discovery["tied"]}
        rows.append({"prefix_id": plan["prefix_id"], "discovery_task_key": plan["discovery_task_key"],
                     "public_own_xy": plan["own_xy"], "predicate_disagreement": plan["predicate_disagreement"],
                     "frozen_predictions": predictions, "baseline_outcome": a, "scan_outcome": b,
                     "scan_minus_baseline_official_return": advantage,
                     "empirical_winner_with_baseline_tie_break": winner,
                     "prediction_correct": {name: prediction == winner if winner is not None else None
                                            for name, prediction in predictions.items()}})
    return {"version": VERSION, "frozen_before_outcomes": True,
            "official_score_tie_rule": "retain baseline; survival time is not a tie breaker",
            "blocked_action_rule": "retain diagnostic record, exclude from policy value",
            "pairs": rows, "decisive_pair": rows[0],
            "selectable_pairs": sum(row["empirical_winner_with_baseline_tie_break"] is not None for row in rows)}


def run_collection(config, discovery, output, max_new_tasks=8):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"version": VERSION, "source_sha256": config["source_sha256"],
                  "discovery_bundle_sha256": discovery["sha256"],
                  "frozen_fit_sha256": digest(discovery["fit"]),
                  "game_seed": config["game_seed"], "synthetic_seed": config["discovery_synthetic_seed"]}
    path = output / "experiment_provenance.json"
    if path.exists() and read(path) != provenance:
        raise ValueError("Output belongs to another source or frozen experiment")
    atomic_json(path, provenance)
    atomic_json(output / "resolved_config.json", config)
    atomic_json(output / "frozen_state_fit.json", discovery["fit"])
    atomic_json(output / "frozen_experiment_plan.json", {"plans": discovery["plans"],
                "competing_predicates": discovery["tied"],
                "candidate_action_ids": [config["continuation_action_id"], discovery["scan_action_id"]]})
    budget = 0
    outcomes = []
    for plan in discovery["plans"]:
        for action in (config["continuation_action_id"], discovery["scan_action_id"]):
            task = evaluation_task(config, discovery, plan, action)
            key = digest(task)[:20]
            record_path = output / "tasks" / f"{key}.json"
            if not record_path.exists() or read(record_path)["status"] != "passed":
                if budget >= max_new_tasks:
                    atomic_json(output / "capture_progress.json", {"stage": "incomplete",
                                "completed_tasks": len(outcomes), "new_tasks_this_run": budget,
                                "remaining_tasks": 8 - len(outcomes)})
                    return {"version": VERSION, "status": "incomplete", "completed_tasks": len(outcomes)}
                budget += 1
            record = run_task(output, task, retry_failed=True)
            outcome = action_outcome(record, action, plan, discovery)
            outcomes.append(outcome)
            atomic_json(output / "capture_progress.json", {"stage": "running",
                        "completed_tasks": len(outcomes), "new_tasks_this_run": budget,
                        "remaining_tasks": 8 - len(outcomes), "last_task_wall_seconds": record["wall_seconds"]})
            print(f"state_actions {len(outcomes)}/8 {plan['prefix_id']} "
                  f"{'baseline' if action == config['continuation_action_id'] else 'scan'} "
                  f"selectable={outcome['selectable']}", flush=True)
    paired = compare_pairs(discovery, outcomes)
    atomic_json(output / "action_outcomes.json", outcomes)
    atomic_json(output / "paired_model_challenge.json", paired)
    summary = {"version": VERSION, "status": "passed", "completed_tasks": len(outcomes),
               "selectable_pairs": paired["selectable_pairs"],
               "decisive_pair_advantage": paired["decisive_pair"]["scan_minus_baseline_official_return"],
               "total_task_wall_seconds": round(sum(x["wall_seconds"] for x in outcomes), 3)}
    atomic_json(output / "run_summary.json", summary)
    atomic_json(output / "capture_progress.json", {"stage": "complete", "completed_tasks": 8,
                "new_tasks_this_run": budget, "remaining_tasks": 0})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-bundle", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tasks", type=int, default=8)
    args = parser.parse_args(argv)
    if args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    config = resolve_config(DEFAULT_CONFIG)
    discovery = checked_discovery(args.discovery_bundle, config)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    invocations_path = output / "invocations.json"
    invocations = read(invocations_path) if invocations_path.exists() else []
    invocations.append({"utc_started": datetime.now(timezone.utc).isoformat(),
                        "command": ["bash", "scripts/run_oracle_pilot_ubuntu.sh", "oracle_state_actions",
                                    *(argv if argv is not None else sys.argv[1:])]})
    atomic_json(invocations_path, invocations)
    started = time.monotonic()
    try:
        result = run_collection(config, discovery, output, args.max_new_tasks)
    except Exception as exc:
        atomic_json(output / "run_failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                  "elapsed_wall_seconds": round(time.monotonic() - started, 3)})
        result = {"version": VERSION, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if result["status"] == "passed":
        (output / "run_failure.json").unlink(missing_ok=True)
    result["invocation_wall_seconds"] = round(time.monotonic() - started, 3)
    archive = package_artifacts(output, "oracle_state_refinement_actions_bundle.zip",
                                {"version": VERSION, "source_sha256": config["source_sha256"],
                                 "discovery_bundle_sha256": discovery["sha256"]})
    result["bundle"] = str(archive)
    result["bundle_sha256"] = sha256_file(archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
