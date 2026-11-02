"""Small, resumable finite-horizon Oracle experiment on one verified R2-L group.

One command collects 3 x 44 root actions, expands one reached successor with
44 actions, solves a two-decision empirical model, and runs three paired
prefix-matched Oracle/baseline episodes. Isaac is launched only by ``run``.
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

from arena_paths import PROJECT, workspace_root
from oracle_pilot import (DEFAULT_CONFIG, atomic_json, digest, read,
                          resolve_config, run_task, task_base)
from oracle_pilot_discovery import boundary_evidence
from oracle_pilot_v21 import package_artifacts, sha256_file

VERSION = "oracle_finite_v1_two_decision"
DEFAULT_ANALYSIS = PROJECT / "docs/codex_report/oracle_pilot_v21_corrected_analysis"
DEFAULT_OUTPUT = workspace_root() / "experiments/oracle-finite-v1"
DEFAULT_REPLAY_BUNDLE = PROJECT / "docs/codex_report/oracle_pilot_v21_replay_validation/replay_validation_bundle.zip"
REQUIRED_REPLAY_FIELDS = ("state_id", "canonical_state", "replay_reference_digest",
                          "score", "skill_status", "visual_memory", "finished", "collapsed")


def checked_replay_bundle(path):
    """Verify the returned Ubuntu bundle instead of trusting a summary alone."""
    path = Path(path).expanduser().resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("Replay bundle has duplicate names or CRC errors")
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("Replay bundle has an unsafe member path")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["file_count"] != len(manifest["files"]):
            raise ValueError("Replay manifest count mismatch")
        expected = {}
        for item in manifest["files"]:
            payload = archive.read(item["path"])
            if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"Replay bundle checksum mismatch: {item['path']}")
            expected[item["path"]] = item["sha256"]
        checksums = dict(line.split("  ", 1)[::-1]
                         for line in archive.read("SHA256SUMS").decode().splitlines())
        if checksums != expected:
            raise ValueError("Replay SHA256SUMS differs from manifest")
        summary = json.loads(archive.read("replay_validation.json"))
        records = [json.loads(archive.read(name)) for name in names
                   if name.startswith("tasks/") and name.endswith(".json")]
    if (summary.get("status") != "passed" or summary.get("passed_prefixes") != 4 or
            summary.get("covered_samples") != 6 or len(records) != 4 or
            any(item.get("status") != "passed" for item in records)):
        raise ValueError("All four independent prefix replays must have passed")
    return summary, {item["task"]["prefix_id"]: item for item in records}, sha256_file(path)


def load_inputs(analysis_dir, replay_bundle, config):
    analysis_dir = Path(analysis_dir).expanduser().resolve()
    samples_path = analysis_dir / "samples_corrected.json"
    plan = read(analysis_dir / "replay_plan.json")
    samples = read(samples_path)
    replay, records, replay_sha = checked_replay_bundle(replay_bundle)
    if (sha256_file(samples_path) != plan["corrected_samples_sha256"] or
            replay["corrected_samples_sha256"] != plan["corrected_samples_sha256"] or
            replay["input_zip_sha256"] != plan["input_zip_sha256"] or
            replay["replay_source_sha256"] != plan["analysis_source_sha256"] or
            plan["prefix_library_hash"] != config["prefix_library"]["hash"] or
            plan["game_seed"] != config["game_seed"] or
            plan["synthetic_seed"] != config["discovery_synthetic_seed"] or
            samples["analysis_version"] != plan["analysis_version"] or
            samples["qualifying_group_count"] != len(samples["groups"])):
        raise ValueError("Corrected samples, replay bundle, or formal configuration disagree")
    prefix_map = {item["prefix_id"]: item for item in plan["prefixes"]}
    if set(prefix_map) != set(records):
        raise ValueError("Replay task records differ from the corrected four-prefix plan")
    for prefix_id, item in records.items():
        if (item["task"].get("source_sha256") != replay["replay_source_sha256"] or
                item["task"].get("prefix_action_ids") != prefix_map[prefix_id]["prefix_action_ids"] or
                item["task"].get("expected_boundaries") != prefix_map[prefix_id]["expected_boundaries"] or
                len(prefix_map[prefix_id]["expected_boundaries"]) != 5 or
                any(not all(field in boundary for field in REQUIRED_REPLAY_FIELDS)
                    for boundary in prefix_map[prefix_id]["expected_boundaries"]) or
                item["result"].get("stop_reason") != "planned_boundary" or
                item["result"].get("boundaries") != 5):
            raise ValueError(f"Replay evidence is incomplete: {prefix_id}")
    groups = [group for group in samples["groups"] if len(group["samples"]) == 3]
    if not groups:
        raise ValueError("No verified three-state collision group exists")
    group = max(groups, key=lambda item: (item["available_distinct_full_states"],
                                          item["samples"][0]["target_step"]))
    if group["samples"][0]["target_step"] != 4:
        raise ValueError("The first finite Oracle experiment expects an 8-second state")
    if any(sample["prefix_id"] not in records or sample["state_id"] != group["state_id"]
           for sample in group["samples"]):
        raise ValueError("Selected samples lack verified replay prefixes")
    return {"analysis_dir": analysis_dir, "samples": samples, "plan": plan,
            "group": group, "prefixes": prefix_map, "replay": replay,
            "replay_bundle_sha256": replay_sha,
            "corrected_samples_sha256": plan["corrected_samples_sha256"]}


def branch_task(config, inputs, sample, action_id, *, prefix_actions=None,
                expected_boundaries=None, target_step=None):
    step = sample["target_step"] if target_step is None else target_step
    prefix = sample["prefix_action_ids"] if prefix_actions is None else prefix_actions
    expected = sample["replay_boundaries"] if expected_boundaries is None else expected_boundaries
    task = task_base(config, "branch", sample["synthetic_seed"])
    task.update(experiment_version=VERSION, sample_id=sample["sample_id"],
                target_step=step, prefix_action_ids=prefix,
                expected_state_id=expected[-1]["state_id"],
                expected_reference_digest=expected[-1]["replay_reference_digest"],
                expected_boundaries=expected, candidate_action_id=action_id,
                horizon_seconds=2, input_zip_sha256=inputs["plan"]["input_zip_sha256"],
                corrected_samples_sha256=inputs["corrected_samples_sha256"],
                replay_bundle_sha256=inputs["replay_bundle_sha256"])
    return task


def read_branch_outcome(record):
    task = record["task"]
    raw = Path(record["raw_dir"])
    decisions = read(raw / "decisions.json") if (raw / "decisions.json").exists() else []
    boundaries = read(raw / "boundaries.json") if (raw / "boundaries.json").exists() else []
    step = task["target_step"]
    decision = decisions[step] if len(decisions) > step else None
    result = record.get("result") or {}
    if result.get("stop_reason") in ("replay_mismatch", "full_replay_mismatch", "reference_replay_mismatch"):
        raise RuntimeError(f"Strict prefix replay failed in task {record['task_key']}")
    if not decision:
        raise RuntimeError(f"No candidate decision in task {record['task_key']}")
    error_class = decision.get("error_class")
    if record["status"] != "passed" and error_class != "not_applicable_in_state":
        raise RuntimeError(f"Unclassified branch failure in task {record['task_key']}: {error_class}")
    available = (record["status"] == "passed" and decision["validator_status"] == "accepted" and
                 len(boundaries) > step)
    successor = boundaries[-1] if available else None
    return {"sample_id": task["sample_id"], "state_id": task["expected_state_id"],
            "decision_step": step, "action_id": task["candidate_action_id"],
            "available": bool(available), "validator_status": decision["validator_status"],
            "execution_status": decision["execution_status"],
            "error_class": error_class, "error": decision.get("error"),
            "reward_delta_score": decision.get("delta_score") if available else None,
            "next_state_id": successor["limited"]["state_id"] if successor else None,
            "next_state_status": successor["limited"]["status"] if successor else None,
            "next_canonical_state": successor["limited"]["canonical_state"] if successor else None,
            "terminal": bool(successor["finished"]) if successor else None,
            "terminal_reason": successor["finished"] if successor else None,
            "stop_reason": result.get("stop_reason"), "task_key": record["task_key"]}


def select_expansion(root_outcomes, samples, catalog):
    """Expand the anchor sample's successor under the best common immediate action."""
    by_action = defaultdict(dict)
    for outcome in root_outcomes:
        by_action[outcome["action_id"]][outcome["sample_id"]] = outcome
    anchor = samples[0]["sample_id"]
    ranked = []
    for action in catalog:
        values = by_action[action["action_id"]]
        if len(values) != len(samples) or not all(x["available"] for x in values.values()):
            continue
        first = values[anchor]
        if first["terminal"] or first["next_state_status"] != "valid" or not first["next_state_id"]:
            continue
        mean_reward = statistics.mean(x["reward_delta_score"] for x in values.values())
        ranked.append((mean_reward, action["action_id"], first))
    if not ranked:
        return None
    _, chosen_action, chosen = sorted(ranked, key=lambda item: (-item[0], item[1]))[0]
    return {"source_sample_id": anchor, "source_action_id": chosen_action,
            "successor_state_id": chosen["next_state_id"],
            "source_task_key": chosen["task_key"],
            "selection": "highest mean immediate official reward among common available actions; anchor sample"}


def solve_model(model, *, baseline_action_id):
    """Two-layer backward induction; unexpanded successors have zero known continuation value."""
    layers = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for outcome in model["transitions"]:
        layers[outcome["decision_step"]][outcome["state_id"]][outcome["action_id"]].append(outcome)
    value = {}
    table = {}
    diagnostics = []
    for step in sorted(layers, reverse=True):
        table[str(step)] = {}
        for state_id, actions in sorted(layers[step].items()):
            expected_count = model["sample_counts"][f"{step}:{state_id}"]
            candidates = []
            for action in model["action_catalog"]:
                action_id = action["action_id"]
                outcomes = actions.get(action_id, [])
                common = (len(outcomes) == expected_count and
                          len({x["sample_id"] for x in outcomes}) == expected_count and
                          all(x["available"] for x in outcomes))
                if not common:
                    diagnostics.append({"decision_step": step, "state_id": state_id,
                                        "action_id": action_id, "common_available": False,
                                        "observed_samples": len(outcomes),
                                        "failure_count": sum(not x["available"] for x in outcomes)})
                    continue
                scores = [x["reward_delta_score"] +
                          (0.0 if x["terminal"] else value.get((step + 1, x["next_state_id"]), 0.0))
                          for x in outcomes]
                average = statistics.mean(scores)
                diagnostics.append({"decision_step": step, "state_id": state_id,
                                    "action_id": action_id, "common_available": True,
                                    "observed_samples": len(outcomes),
                                    "mean_immediate_reward": statistics.mean(x["reward_delta_score"] for x in outcomes),
                                    "estimated_value": average,
                                    "distinct_successors": len({x["next_state_id"] for x in outcomes}),
                                    "reward_range": [min(x["reward_delta_score"] for x in outcomes),
                                                     max(x["reward_delta_score"] for x in outcomes)]})
                candidates.append((average, action_id))
            if not candidates:
                if step == model["root_decision_step"]:
                    raise ValueError(f"No common available formal action in {step}:{state_id}")
                continue
            best_value, best_action = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
            value[(step, state_id)] = best_value
            table[str(step)][state_id] = best_action
    root_step = model["root_decision_step"]
    root_state = model["root_state_id"]
    root_options = {x["action_id"]: x for x in diagnostics
                    if x["decision_step"] == root_step and x["state_id"] == root_state}
    baseline = root_options.get(baseline_action_id)
    return {"version": VERSION, "policy_table": table, "root_state_id": root_state,
            "root_decision_step": root_step,
            "root_action_id": table[str(root_step)][root_state],
            "root_estimated_value": value[(root_step, root_state)],
            "baseline_action_id": baseline_action_id,
            "baseline_estimated_value": baseline.get("estimated_value") if baseline else None,
            "model_value_advantage": (value[(root_step, root_state)] - baseline["estimated_value"]
                                      if baseline and baseline.get("estimated_value") is not None else None),
            "state_time_coverage": sum(len(layer) for layer in table.values()),
            "horizon_decisions": 2 if sum(len(layer) for layer in table.values()) > 1 else 1,
            "unknown_successor_value": 0.0,
            "selection_rule": "max empirical mean official score delta over commonly available actions; tie by action ID",
            "action_diagnostics": diagnostics}


def model_from_outcomes(config, inputs, root, expansion, child):
    group = inputs["group"]
    transitions = root + child
    root_by_action = defaultdict(list)
    for item in root:
        root_by_action[item["action_id"]].append(item)
    heterogeneity = {action_id: {"different_successors": len({x["next_state_id"] for x in values
                                                              if x["available"]}) > 1,
                                 "different_rewards": len({x["reward_delta_score"] for x in values
                                                           if x["available"]}) > 1,
                                 "unavailable_samples": [x["sample_id"] for x in values if not x["available"]]}
                     for action_id, values in root_by_action.items()}
    states = [{"decision_step": 4, "state_id": group["state_id"],
               "canonical_state": group["canonical_state"], "sample_count": 3}]
    if expansion is not None:
        states.append({"decision_step": 5, "state_id": expansion["successor_state_id"],
                       "canonical_state": next(x["next_canonical_state"] for x in root
                                               if x["task_key"] == expansion["source_task_key"]),
                       "sample_count": 1})
    for outcome in transitions:
        if outcome["next_state_id"] and all(s["state_id"] != outcome["next_state_id"] or
                                             s["decision_step"] != outcome["decision_step"] + 1 for s in states):
            states.append({"decision_step": outcome["decision_step"] + 1,
                           "state_id": outcome["next_state_id"],
                           "canonical_state": outcome["next_canonical_state"],
                           "sample_count": 0, "expanded": False})
    return {"version": VERSION, "source_sha256": config["source_sha256"],
            "original_input_zip_sha256": inputs["plan"]["input_zip_sha256"],
            "corrected_samples_sha256": inputs["corrected_samples_sha256"],
            "replay_bundle_sha256": inputs["replay_bundle_sha256"],
            "scene": config["scene"], "rules_config": config["rules_config"],
            "opponent_policy": config["opponent_policy"], "game_seed": config["game_seed"],
            "synthetic_seed": config["discovery_synthetic_seed"],
            "action_space_version": config["action_catalog_version"],
            "action_catalog": config["action_catalog"],
            "root_decision_step": 4, "root_state_id": group["state_id"],
            "sample_counts": {f"4:{group['state_id']}": 3, **(
                {f"5:{expansion['successor_state_id']}": 1} if expansion is not None else {})},
            "expansion": expansion, "states": states, "transitions": transitions,
            "transition_count": len(transitions),
            "root_action_heterogeneity": heterogeneity,
            "root_actions_with_multiple_successors": sum(x["different_successors"] for x in heterogeneity.values()),
            "root_actions_with_different_rewards": sum(x["different_rewards"] for x in heterogeneity.values()),
            "unavailable_transition_count": sum(not x["available"] for x in transitions),
            "nonterminal_successor_states": len({x["next_state_id"] for x in transitions
                                                  if x["available"] and not x["terminal"]}),
            "approximation": ("Three equally weighted hidden root samples; one expanded successor sample; "
                              "unexpanded successors have zero measured continuation value." if expansion is not None else
                              "Three equally weighted hidden root samples; no common available action reached a valid nonterminal successor, so this is one-step only.")}


def progress(output, stage, completed, budget, records):
    values = [x["wall_seconds"] for x in records if x.get("wall_seconds") is not None]
    data = {"version": VERSION, "stage": stage, "completed_tasks": completed,
            "new_tasks_this_invocation": budget["used"], "max_new_tasks": budget["max"],
            "median_task_wall_seconds": statistics.median(values) if values else None,
            "estimated_182_tasks_wall_seconds": round(statistics.median(values) * 182, 1) if values else None}
    atomic_json(output / "capture_progress.json", data)
    print(f"finite_oracle stage={stage} completed={completed}/182 "
          f"new_this_run={budget['used']}/{budget['max']} "
          f"median_task_s={data['median_task_wall_seconds']}", flush=True)
    return data


def execute_task(output, task, budget, records, *, retry_failed=False):
    key = digest(task)[:20]
    path = output / "tasks" / f"{key}.json"
    if not path.exists():
        if budget["used"] >= budget["max"]:
            return None
        budget["used"] += 1
    elif retry_failed and read(path)["status"] != "passed":
        old = read(path)
        retry_needed = True
        if task["mode"] == "branch":
            try:
                read_branch_outcome(old)
            except RuntimeError:
                pass
            else:
                retry_needed = False
        if retry_needed:
            if budget["used"] >= budget["max"]:
                return None
            budget["used"] += 1
        else:
            retry_failed = False
    record = run_task(output, task, retry_failed=retry_failed)
    records.append(record)
    return record


def run_batch(inputs, output, config, max_new_tasks=182, retry_failed=False):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"version": VERSION, "source_sha256": config["source_sha256"],
                  "input_zip_sha256": inputs["plan"]["input_zip_sha256"],
                  "corrected_samples_sha256": inputs["corrected_samples_sha256"],
                  "replay_bundle_sha256": inputs["replay_bundle_sha256"],
                  "game_seed": config["game_seed"], "synthetic_seed": config["discovery_synthetic_seed"],
                  "command": "bash scripts/run_oracle_pilot_ubuntu.sh finite_oracle_run"}
    prior = output / "experiment_provenance.json"
    if prior.exists() and read(prior) != provenance:
        raise ValueError("Output directory belongs to another source or input; use a new --output")
    atomic_json(prior, provenance)
    evidence = output / "input_evidence"
    evidence.mkdir(exist_ok=True)
    shutil.copy2(inputs["analysis_dir"] / "samples_corrected.json", evidence / "samples_corrected.json")
    shutil.copy2(inputs["analysis_dir"] / "replay_plan.json", evidence / "replay_plan.json")
    atomic_json(evidence / "replay_validation.json", inputs["replay"])
    atomic_json(output / "resolved_config.json", config)
    budget = {"used": 0, "max": max_new_tasks}
    records = []
    samples = inputs["group"]["samples"]
    catalog = config["action_catalog"]
    root = []
    for sample in samples:
        for action in catalog:
            task = branch_task(config, inputs, sample, action["action_id"])
            record = execute_task(output, task, budget, records, retry_failed=retry_failed)
            if record is None:
                return progress(output, "root_incomplete", len(root), budget, records)
            root.append(read_branch_outcome(record))
            progress(output, "root", len(root), budget, records)
    atomic_json(output / "root_transitions.json", root)
    expansion = select_expansion(root, samples, catalog)
    child = []
    if expansion is not None:
        source_record = next(x for x in records if x["task_key"] == expansion["source_task_key"])
        source_boundaries = read(Path(source_record["raw_dir"]) / "boundaries.json")
        if len(source_boundaries) != 6:
            raise ValueError("Expansion source must contain all six boundaries")
        expected = []
        for boundary in source_boundaries:
            evidence = boundary_evidence(boundary)
            evidence["collapsed"] = boundary["collapsed"]
            expected.append(evidence)
        anchor = samples[0]
        for action in catalog:
            task = branch_task(config, inputs, anchor, action["action_id"],
                               prefix_actions=anchor["prefix_action_ids"] + [expansion["source_action_id"]],
                               expected_boundaries=expected, target_step=5)
            record = execute_task(output, task, budget, records, retry_failed=retry_failed)
            if record is None:
                atomic_json(output / "expansion_plan.json", expansion)
                return progress(output, "successor_incomplete", len(root) + len(child), budget, records)
            child.append(read_branch_outcome(record))
            progress(output, "successor", len(root) + len(child), budget, records)
    atomic_json(output / "expansion_plan.json", expansion)
    model = model_from_outcomes(config, inputs, root, expansion, child)
    atomic_json(output / "transition_model.json", model)
    policy = solve_model(model, baseline_action_id=config["continuation_action_id"])
    policy.update(source_sha256=config["source_sha256"],
                  model_sha256=sha256_file(output / "transition_model.json"),
                  corrected_samples_sha256=inputs["corrected_samples_sha256"])
    atomic_json(output / "oracle_policy.json", policy)
    policy_sha = sha256_file(output / "oracle_policy.json")
    evaluations = []
    for sample in samples:
        prefix = inputs["prefixes"][sample["prefix_id"]]
        for mode in ("oracle_eval_prefix", "oracle_eval_baseline_prefix"):
            task = task_base(config, mode, sample["synthetic_seed"])
            task.update(experiment_version=VERSION, sample_id=sample["sample_id"],
                        prefix_id=sample["prefix_id"], prefix_action_ids=prefix["prefix_action_ids"],
                        expected_boundaries=prefix["expected_boundaries"], max_steps=6,
                        policy_sha256=policy_sha, model_sha256=policy["model_sha256"],
                        corrected_samples_sha256=inputs["corrected_samples_sha256"])
            if mode == "oracle_eval_prefix":
                task["policy_table"] = policy["policy_table"]
            record = execute_task(output, task, budget, records, retry_failed=retry_failed)
            if record is None:
                return progress(output, "evaluation_incomplete", len(root) + len(child) + len(evaluations), budget, records)
            result = record.get("result") or {}
            if result.get("stop_reason") == "replay_mismatch":
                raise RuntimeError(f"Evaluation prefix replay mismatch: {record['task_key']}")
            raw = Path(record["raw_dir"])
            decisions = read(raw / "decisions.json") if (raw / "decisions.json").exists() else []
            evaluations.append({"sample_id": sample["sample_id"], "prefix_id": sample["prefix_id"],
                                "mode": mode, "task_key": record["task_key"],
                                "status": record["status"], "stop_reason": result.get("stop_reason"),
                                "score_delta_from_8s": result.get("evaluation_delta_score"),
                                "score_at_end": result.get("official_score"),
                                "policy_actions": [x["action_id"] for x in decisions[4:]],
                                "policy_sources": [x["source"] for x in decisions[4:]],
                                "wall_seconds": record["wall_seconds"]})
            progress(output, "evaluation", len(root) + len(child) + len(evaluations), budget, records)
    pairs = []
    for sample in samples:
        actual = next(x for x in evaluations if x["sample_id"] == sample["sample_id"] and x["mode"] == "oracle_eval_prefix")
        baseline = next(x for x in evaluations if x["sample_id"] == sample["sample_id"] and x["mode"] == "oracle_eval_baseline_prefix")
        pairs.append({"sample_id": sample["sample_id"],
                      "oracle_delta": actual["score_delta_from_8s"],
                      "baseline_delta": baseline["score_delta_from_8s"],
                      "paired_advantage": (round(actual["score_delta_from_8s"] - baseline["score_delta_from_8s"], 6)
                                           if actual["status"] == baseline["status"] == "passed" else None),
                      "oracle_table_hits": actual["policy_sources"].count("oracle_table")})
    valid = [x["paired_advantage"] for x in pairs if x["paired_advantage"] is not None]
    comparison = {"version": VERSION, "pairs": pairs, "episodes": evaluations,
                  "valid_pairs": len(valid), "mean_paired_advantage": statistics.mean(valid) if valid else None,
                  "status": "passed" if len(valid) == 3 and any(x["oracle_table_hits"] for x in pairs) else "partial"}
    atomic_json(output / "evaluation_comparison.json", comparison)
    result = progress(output, "complete", len(root) + len(child) + len(evaluations), budget, records)
    result.update(status=comparison["status"], model_state_time_coverage=policy["state_time_coverage"],
                  policy_horizon_decisions=policy["horizon_decisions"],
                  root_action_id=policy["root_action_id"],
                  root_actions_with_multiple_successors=model["root_actions_with_multiple_successors"],
                  root_actions_with_different_rewards=model["root_actions_with_different_rewards"],
                  model_value_advantage=policy["model_value_advantage"],
                  actual_mean_paired_advantage=comparison["mean_paired_advantage"],
                  output=str(output))
    atomic_json(output / "run_summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--replay-bundle", type=Path, default=DEFAULT_REPLAY_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tasks", type=int, default=182,
                        help="Per invocation cap; rerun the same command to resume")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    config = resolve_config(DEFAULT_CONFIG)
    inputs = load_inputs(args.analysis_dir, args.replay_bundle, config)
    replay_times = [item["wall_seconds"] for item in inputs["replay"]["prefix_results"]]
    estimate = round(statistics.median(replay_times) * 182 / 60, 1)
    print(f"validated replay=4/4 samples=6/6 root_state={inputs['group']['state_id'][-12:]} "
          f"planned_max_tasks=182 replay_based_estimate_minutes={estimate}", flush=True)
    output = args.output.expanduser().resolve()
    if output == inputs["analysis_dir"] or inputs["analysis_dir"] in output.parents:
        parser.error("--output must be separate from the corrected analysis directory")
    output.mkdir(parents=True, exist_ok=True)
    invocation_path = output / "invocations.json"
    invocations = read(invocation_path) if invocation_path.exists() else []
    invocations.append({"utc_started": datetime.now(timezone.utc).isoformat(),
                        "command": ["bash", "scripts/run_oracle_pilot_ubuntu.sh", "finite_oracle_run", *sys.argv[1:]],
                        "max_new_tasks": args.max_new_tasks, "retry_failed": args.retry_failed})
    atomic_json(invocation_path, invocations)
    started = time.monotonic()
    try:
        result = run_batch(inputs, output, config, args.max_new_tasks, args.retry_failed)
    except Exception as exc:
        atomic_json(output / "run_failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                  "elapsed_wall_seconds": round(time.monotonic() - started, 3)})
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    result["invocation_wall_seconds"] = round(time.monotonic() - started, 3)
    archive = package_artifacts(output, "oracle_finite_v1_bundle.zip",
                                {"version": VERSION, "source_sha256": config["source_sha256"],
                                 "input_zip_sha256": inputs["plan"]["input_zip_sha256"],
                                 "replay_bundle_sha256": inputs["replay_bundle_sha256"]})
    result["bundle"] = str(archive)
    result["bundle_sha256"] = sha256_file(archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
