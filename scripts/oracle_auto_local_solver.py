"""Rebuild the local Oracle by automated one-bit refinement and backward DP.

This is an offline evidence adapter. The optimizer itself is domain-independent
(`finite_state_optimizer.py`); this file verifies QuadrupedArena bundles and
converts measured official-score trajectories into a finite-horizon model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import zipfile

from adaptive_motion import load_config
from arena_paths import PROJECT
from finite_state_optimizer import (feature_bin, fit_one_bit_state, geometry_predicates,
                                    refined_state_key, solve_empirical_horizon)
from oracle_p2_second_stage import checked_full_game_bundle
from oracle_pilot import DEFAULT_CONFIG, atomic_json, resolve_config
from oracle_pilot_runtime import strategy_action_executable
from oracle_pilot_v21 import sha256_file

VERSION = "oracle_auto_local_solver_v1"
DEFAULT_FULL = PROJECT / "ubuntu_logs/oracle_local_p1_full_game_bundle.zip"
DEFAULT_P2 = PROJECT / "ubuntu_logs/oracle_p2_second_stage_bundle.zip"
DEFAULT_OUTPUT = PROJECT / "docs/codex_report/oracle_auto_local_solver_v1"


def checked_p2_bundle(path, full_sha256, config):
    path = Path(path).expanduser().resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None or any(
                name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("P2 bundle has duplicate, unsafe, or corrupt members")
        manifest = json.loads(archive.read("manifest.json"))
        expected = {}
        for item in manifest["files"]:
            data = archive.read(item["path"])
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"P2 bundle checksum mismatch: {item['path']}")
            expected[item["path"]] = item["sha256"]
        sums = dict(line.split("  ", 1)[::-1] for line in archive.read("SHA256SUMS").decode().splitlines())
        if (manifest["file_count"] != len(expected) or sums != expected or
                set(names) != set(expected) | {"manifest.json", "SHA256SUMS"}):
            raise ValueError("P2 manifest disagrees with ZIP members")
        provenance = json.loads(archive.read("experiment_provenance.json"))
        summary = json.loads(archive.read("run_summary.json"))
        outcomes = json.loads(archive.read("training_outcomes.json"))
        if (summary.get("status") != "passed" or summary.get("training_candidates") != len(outcomes) or
                provenance["frozen_full_game_bundle_sha256"] != full_sha256 or
                provenance["source_sha256"] != manifest["source_sha256"] or
                provenance["game_seed"] != config["game_seed"] or
                provenance["synthetic_seed"] != config["discovery_synthetic_seed"] or
                provenance["candidate_action_ids"] != [x["action_id"] for x in outcomes] or
                summary["selected_action_id"] not in provenance["candidate_action_ids"]):
            raise ValueError("P2 training data do not match the frozen full-game source")
        for outcome in outcomes:
            key = outcome["task_key"]
            record = json.loads(archive.read(f"tasks/{key}.json"))
            decisions = json.loads(archive.read(f"raw/{key}/decisions.json"))
            boundaries = json.loads(archive.read(f"raw/{key}/boundaries.json"))
            if (record["task_key"] != key or record["task"]["mode"] != "oracle_second_stage" or
                    record["task"]["candidate_action_id"] != outcome["action_id"] or
                    record["task"]["sample_id"] != outcome["sample_id"] or
                    record["task"]["source_sha256"] != provenance["source_sha256"] or
                    len(decisions) < 11 or len(boundaries) < 11 or
                    decisions[10]["action_id"] != outcome["action_id"] or
                    boundaries[10]["limited"]["state_id"] != provenance["target_state_id"] or
                    boundaries[10]["score"] != outcome["score_at_20s"] or
                    decisions[10]["validator_status"] != outcome["validator_status"] or
                    decisions[10]["execution_status"] != outcome["execution_status"] or
                    (record["result"] or {}).get("official_score") != outcome["observed_final_score_diagnostic"] or
                    (record["status"] == "passed" and strategy_action_executable(decisions[10])) !=
                    outcome["selectable"]):
                raise ValueError(f"P2 action outcome disagrees with raw task evidence: {key}")
    return {"path": path, "sha256": sha256_file(path), "provenance": provenance,
            "summary": summary, "outcomes": outcomes}


def public_xy_at_8(archive, key):
    frames = json.loads(archive.read(f"raw/{key}/frame_index.json"))
    positions = [item["own_xy"] for item in frames if abs(item["simulation_seconds"] - 8.0) < 1e-8]
    if len(positions) != 1:
        raise ValueError(f"Expected exactly one public 8 s position in {key}")
    return positions[0]


def root_rows(full_path, verified, baseline_action_id):
    rows = []
    with zipfile.ZipFile(full_path) as archive:
        for index, pair in enumerate(verified["paired"]["pairs"]):
            policy_key, baseline_key = pair["policy_task_key"], pair["baseline_task_key"]
            policy_boundaries = json.loads(archive.read(f"raw/{policy_key}/boundaries.json"))
            baseline_boundaries = json.loads(archive.read(f"raw/{baseline_key}/boundaries.json"))
            policy_decisions = json.loads(archive.read(f"raw/{policy_key}/decisions.json"))
            baseline_decisions = json.loads(archive.read(f"raw/{baseline_key}/decisions.json"))
            own_xy = public_xy_at_8(archive, policy_key)
            if (own_xy != public_xy_at_8(archive, baseline_key) or
                    policy_boundaries[4]["score"] != baseline_boundaries[4]["score"] or
                    policy_boundaries[4]["limited"]["state_id"] !=
                    baseline_boundaries[4]["limited"]["state_id"] or
                    pair["baseline_action_id"] != baseline_action_id or
                    not strategy_action_executable(policy_decisions[4]) or
                    not strategy_action_executable(baseline_decisions[4])):
                raise ValueError(f"Root pair does not have a valid common 8 s state: {pair['sample_id']}")
            start_score = policy_boundaries[4]["score"]
            returns = {baseline_action_id: pair["baseline_final_official_score"] - start_score}
            if pair["policy_action_id"] != baseline_action_id:
                returns[pair["policy_action_id"]] = pair["policy_final_official_score"] - start_score
            score_at_20 = next((x["score"] for x in policy_boundaries if
                                abs(x["simulation_seconds"] - 20.0) < 1e-8), None)
            rows.append({"sample_id": pair["sample_id"], "prefix_id": pair["prefix_id"],
                         "split": "heldout" if index == len(verified["paired"]["pairs"]) - 1 else "training",
                         "base_state_id": policy_boundaries[4]["limited"]["state_id"],
                         "own_xy": own_xy, "score_at_8s": start_score,
                         "score_at_20s": score_at_20,
                         "returns": returns,
                         "policy_action_id": pair["policy_action_id"],
                         "policy_final_score": pair["policy_final_official_score"],
                         "baseline_final_score": pair["baseline_final_official_score"]})
    if len(rows) != 4 or len({row["base_state_id"] for row in rows}) != 1:
        raise ValueError("Expected four samples in one root R2-L collision state")
    return rows


def optimize(config, full_path, p2_path, *, min_leaf_samples=1, complexity_penalty=0.0):
    verified = checked_full_game_bundle(full_path, config)
    p2 = checked_p2_bundle(p2_path, verified["sha256"], config)
    cfg = load_config(PROJECT / "configs" / config["rules_config"])
    baseline = config["continuation_action_id"]
    rows = root_rows(full_path, verified, baseline)
    training, heldout = rows[:3], rows[3]
    fit = fit_one_bit_state(training, geometry_predicates(cfg["arena"]["task_points"]), baseline,
                            min_leaf_samples=min_leaf_samples,
                            complexity_penalty=complexity_penalty)
    p2_state_id = p2["provenance"]["target_state_id"]
    if (p2["provenance"]["target_step"] != 10 or
            {x["sample_id"] for x in p2["outcomes"]} != {training[0]["sample_id"]} or
            p2["outcomes"][0]["action_id"] != baseline or
            training[0]["score_at_20s"] != p2["outcomes"][0]["score_at_20s"]):
        raise ValueError("P2 interventions do not continue the training root state")
    root_groups = defaultdict(list)
    for row in training:
        root_groups[refined_state_key(row["base_state_id"], row["own_xy"], fit)].append(row)
    states = [{"step": 4, "state_key": key,
               "sample_ids": [row["sample_id"] for row in group]}
              for key, group in root_groups.items()]
    states.append({"step": 10, "state_key": p2_state_id,
                   "sample_ids": [training[0]["sample_id"]]})
    transitions = []
    for outcome in p2["outcomes"]:
        observed = outcome["observed_delta_score_diagnostic"]
        transitions.append({"step": 10, "state_key": p2_state_id,
                            "action_id": outcome["action_id"],
                            "sample_id": outcome["sample_id"],
                            "reward": observed if observed is not None else 0.0,
                            "selectable": outcome["selectable"],
                            "terminal": True, "next_step": None, "next_state_key": None})
    scan_action = training[0]["policy_action_id"]
    for row in training:
        state_key = refined_state_key(row["base_state_id"], row["own_xy"], fit)
        for action_id, full_return in row["returns"].items():
            continues_to_p2 = (row["sample_id"] == training[0]["sample_id"] and
                               action_id == scan_action and action_id != baseline)
            if continues_to_p2 and row["score_at_20s"] is None:
                raise ValueError("The scanned training trajectory has no P2 boundary")
            transitions.append({"step": 4, "state_key": state_key,
                                "action_id": action_id, "sample_id": row["sample_id"],
                                "reward": (row["score_at_20s"] - row["score_at_8s"]
                                           if continues_to_p2 else full_return),
                                "selectable": True, "terminal": not continues_to_p2,
                                "next_step": 10 if continues_to_p2 else None,
                                "next_state_key": p2_state_id if continues_to_p2 else None})
    solved = solve_empirical_horizon(states, transitions, baseline)
    policy = {(item["step"], item["state_key"]): item["action_id"] for item in solved["values"]}
    root_key = refined_state_key(heldout["base_state_id"], heldout["own_xy"], fit)
    heldout_action = policy[(4, root_key)]
    heldout_observed = heldout["returns"].get(heldout_action)
    p2_action = policy[(10, p2_state_id)]
    heldout_supported = heldout_observed is not None and p2_action == baseline
    validation = {"heldout_sample_id": heldout["sample_id"],
                  "predicted_root_state_key": root_key,
                  "predicted_root_action_id": heldout_action,
                  "predicted_p2_action_id": p2_action,
                  "measured_policy_final_return_from_8s": heldout_observed if heldout_supported else None,
                  "measured_baseline_final_return_from_8s": heldout["returns"][baseline],
                  "measured_advantage": (heldout_observed - heldout["returns"][baseline]
                                         if heldout_supported else None),
                  "supported_by_frozen_heldout_rollout": heldout_supported,
                  "heldout_outcomes_used_for_fit_or_dp": False}
    return {"version": VERSION, "source_sha256": config["source_sha256"],
            "full_game_bundle_sha256": verified["sha256"], "p2_bundle_sha256": p2["sha256"],
            "training_rows": training, "heldout_row_diagnostic": heldout,
            "state_fit": fit, "states": states, "transitions": transitions,
            "solved": solved, "heldout_validation": validation,
            "scope": "Exact empirical optimum among measured, common executable actions and known continuations; not global simulator optimality"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-game-bundle", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--p2-bundle", type=Path, default=DEFAULT_P2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-leaf-samples", type=int, default=1)
    parser.add_argument("--complexity-penalty", type=float, default=0.0,
                        help="Official-score units charged for one extra state bit")
    args = parser.parse_args(argv)
    config = resolve_config(DEFAULT_CONFIG)
    result = optimize(config, args.full_game_bundle, args.p2_bundle,
                      min_leaf_samples=args.min_leaf_samples,
                      complexity_penalty=args.complexity_penalty)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("state_fit.json", result["state_fit"]),
                        ("finite_horizon_states.json", result["states"]),
                        ("finite_horizon_transitions.json", result["transitions"]),
                        ("optimized_policy.json", result["solved"]),
                        ("heldout_validation.json", result["heldout_validation"]),
                        ("experiment_summary.json", {k: result[k] for k in
                                                     ("version", "source_sha256", "full_game_bundle_sha256",
                                                      "p2_bundle_sha256", "scope")})):
        atomic_json(output / name, value)
    print(json.dumps({"status": "passed", "output": str(output),
                      "predicate": result["state_fit"]["predicate"],
                      "policy": result["solved"]["values"],
                      "heldout": result["heldout_validation"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
