"""Verify paired action evidence, refit observable state, and solve local DP.

The decisive prefix is evaluated against the *previously frozen* model before
it is added to training. Model selection then uses measured official returns,
with a leave-one-new-prefix-out diagnostic for the small sample regime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from adaptive_motion import load_config
from arena_paths import PROJECT
from finite_state_optimizer import (feature_bin, fit_one_bit_state, geometry_predicates,
                                    refined_state_key, solve_empirical_horizon)
from oracle_auto_local_solver import DEFAULT_FULL, DEFAULT_P2, optimize
from oracle_pilot import DEFAULT_CONFIG, atomic_json, digest, resolve_config
from oracle_pilot_runtime import strategy_action_executable
from oracle_pilot_v21 import sha256_file
from oracle_state_refinement_actions import (DEFAULT_DISCOVERY, action_outcome, checked_discovery,
                                              compare_pairs, evaluation_task)

VERSION = "oracle_state_refinement_solver_v1"
DEFAULT_ACTIONS = PROJECT / "ubuntu_logs/oracle_state_refinement_actions_bundle.zip"
DEFAULT_OUTPUT = PROJECT / "docs/codex_report/oracle_state_refinement_solver_v1"


def checked_actions(path, discovery, config):
    path = Path(path).expanduser().resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or archive.testzip() is not None or
                any(name.startswith("/") or ".." in Path(name).parts for name in names)):
            raise ValueError("Action ZIP has duplicate, unsafe, or corrupt members")
        manifest = json.loads(archive.read("manifest.json"))
        checks = {}
        for entry in manifest["files"]:
            payload = archive.read(entry["path"])
            actual = hashlib.sha256(payload).hexdigest()
            if len(payload) != entry["bytes"] or actual != entry["sha256"]:
                raise ValueError(f"Action ZIP member checksum mismatch: {entry['path']}")
            checks[entry["path"]] = actual
        sums = dict(line.split("  ", 1)[::-1] for line in archive.read("SHA256SUMS").decode().splitlines())
        if (manifest["file_count"] != len(checks) or len(checks) != len(manifest["files"]) or
                sums != checks or set(names) != set(checks) | {"manifest.json", "SHA256SUMS"}):
            raise ValueError("Action ZIP manifest or SHA256SUMS mismatch")
        provenance = json.loads(archive.read("experiment_provenance.json"))
        summary = json.loads(archive.read("run_summary.json"))
        plan = json.loads(archive.read("frozen_experiment_plan.json"))
        fit = json.loads(archive.read("frozen_state_fit.json"))
        outcomes = json.loads(archive.read("action_outcomes.json"))
        paired = json.loads(archive.read("paired_model_challenge.json"))
        stored_config = json.loads(archive.read("resolved_config.json"))
        baseline, scan = config["continuation_action_id"], discovery["scan_action_id"]
        if (summary.get("status") != "passed" or summary.get("completed_tasks") != 8 or
                provenance["discovery_bundle_sha256"] != discovery["sha256"] or
                provenance["source_sha256"] != manifest["source_sha256"] or
                provenance["frozen_fit_sha256"] != digest(discovery["fit"]) or
                provenance["game_seed"] != config["game_seed"] or
                provenance["synthetic_seed"] != config["discovery_synthetic_seed"] or
                stored_config["game_seed"] != config["game_seed"] or
                stored_config["action_catalog"] != config["action_catalog"] or
                stored_config["rules_config"] != config["rules_config"] or
                fit != discovery["fit"] or plan["plans"] != discovery["plans"] or
                plan["competing_predicates"] != discovery["tied"] or
                plan["candidate_action_ids"] != [baseline, scan]):
            raise ValueError("Action experiment differs from frozen discovery or formal rules")
        by_key = {(row["prefix_id"], row["action_id"]): row for row in outcomes}
        if len(by_key) != len(outcomes) or len(outcomes) != 8:
            raise ValueError("Action outcomes do not contain eight unique tasks")
        for item in discovery["plans"]:
            for action in (baseline, scan):
                row = by_key[(item["prefix_id"], action)]
                key = row["task_key"]
                record = json.loads(archive.read(f"tasks/{key}.json"))
                task = record["task"]
                decisions = json.loads(archive.read(f"raw/{key}/decisions.json"))
                boundaries = json.loads(archive.read(f"raw/{key}/boundaries.json"))
                frames = json.loads(archive.read(f"raw/{key}/frame_index.json"))
                actual_task = evaluation_task(config, discovery, item, action)
                # Source fingerprints change when offline tools evolve. Compare all
                # simulator-relevant fields and the recorded historical fingerprint.
                actual_task["source_sha256"] = provenance["source_sha256"]
                if (record["task_key"] != key or digest(task)[:20] != key or task != actual_task or
                        record["status"] != "passed" or len(decisions) < 5 or len(boundaries) < 6 or
                        boundaries[4]["limited"]["state_id"] != discovery["fit"]["base_state_id"] or
                        boundaries[4]["score"] != item["score_at_8s"] or
                        [f["own_xy"] for f in frames if abs(f["simulation_seconds"] - 8) < 1e-8] !=
                        [item["own_xy"]] or
                        record["result"]["official_score"] != row["observed_final_score_diagnostic"] or
                        record["result"]["terminal_reason"] != row["terminal_reason"] or
                        strategy_action_executable(decisions[4]) != row["selectable"] or
                        decisions[4]["validator_status"] != row["validator_status"] or
                        decisions[4]["execution_status"] != row["execution_status"] or
                        (decisions[4].get("compiled_action") or {}).get("skill") != row["compiled_action_skill"] or
                        (decisions[4].get("compiled_skill") or {}).get("state") != row["compiled_skill_state"] or
                        (round(record["result"]["official_score"] - item["score_at_8s"], 6)
                         if row["selectable"] else None) != row["official_return"]):
                    raise ValueError(f"Action row disagrees with raw simulator evidence: {key}")
                if (decisions[4]["action_id"] != action or
                        decisions[4]["source"] != row["action_source"] or
                        boundaries[4]["local_policy_state"]["bin"] != row["local_bin"]):
                    raise ValueError(f"Action was not actually selected at 8 s: {key}")
        if compare_pairs(discovery, outcomes) != paired or summary["selectable_pairs"] != paired["selectable_pairs"]:
            raise ValueError("Paired challenge differs from verified raw outcomes")
    return {"path": path, "sha256": sha256_file(path), "summary": summary,
            "outcomes": outcomes, "paired": paired, "source_sha256": provenance["source_sha256"]}


def measured_rows(discovery, actions):
    rows = []
    for pair in actions["paired"]["pairs"]:
        plan = next(p for p in discovery["plans"] if p["prefix_id"] == pair["prefix_id"])
        returns = {}
        for outcome in (pair["baseline_outcome"], pair["scan_outcome"]):
            if outcome["selectable"]:
                returns[outcome["action_id"]] = outcome["official_return"]
        rows.append({"sample_id": plan["discovery_task_key"] + "_04", "prefix_id": plan["prefix_id"],
                     "base_state_id": discovery["fit"]["base_state_id"], "own_xy": plan["own_xy"],
                     "score_at_8s": plan["score_at_8s"], "returns": returns})
    return rows


def model_action(fit, row):
    label = "all" if fit["selected_bits"] == 0 else feature_bin(row["own_xy"], fit["predicate"])
    return fit["leaves"][label]["action_id"]


def solve(config, prior, discovery, actions, *, min_leaf_samples=2, complexity_penalty=0.0):
    if (discovery["provenance"]["full_game_bundle_sha256"] != prior["full_game_bundle_sha256"] or
            discovery["provenance"]["p2_bundle_sha256"] != prior["p2_bundle_sha256"]):
        raise ValueError("Discovery state selection came from different full-game or P2 evidence")
    cfg = load_config(PROJECT / "configs" / config["rules_config"])
    predicates = geometry_predicates(cfg["arena"]["task_points"])
    baseline = config["continuation_action_id"]
    old_training, old_holdout = prior["training_rows"], prior["heldout_row_diagnostic"]
    new_rows = measured_rows(discovery, actions)
    if (len(old_training) != 3 or len(new_rows) != 4 or
            old_holdout["sample_id"] in {r["sample_id"] for r in old_training + new_rows} or
            not all(baseline in row["returns"] for row in new_rows)):
        raise ValueError("Training, heldout, or paired baseline support changed")
    original_fit = prior["state_fit"]
    challenge = new_rows[0]
    frozen_action = model_action(original_fit, challenge)
    if frozen_action != actions["paired"]["decisive_pair"]["frozen_predictions"][original_fit["predicate"]["id"]]:
        raise ValueError("Frozen model challenge no longer matches original fit")
    training = old_training + new_rows
    fit = fit_one_bit_state(training, predicates, baseline,
                            min_leaf_samples=min_leaf_samples, complexity_penalty=complexity_penalty)
    p2_state = next(x for x in prior["states"] if x["step"] == 10)
    p2_transitions = [edge for edge in prior["transitions"] if edge["step"] == 10]
    if (len(p2_transitions) < 2 or
            next(x for x in prior["solved"]["values"] if x["step"] == 10)["action_id"] != baseline):
        raise ValueError("Known P2 continuation no longer retains the baseline on official-score ties")
    states = []
    grouped = {}
    for row in training:
        key = refined_state_key(row["base_state_id"], row["own_xy"], fit)
        grouped.setdefault(key, []).append(row["sample_id"])
    states.extend({"step": 4, "state_key": key, "sample_ids": sample_ids}
                  for key, sample_ids in sorted(grouped.items()))
    states.append(p2_state)
    transitions = list(p2_transitions)
    for row in training:
        state_key = refined_state_key(row["base_state_id"], row["own_xy"], fit)
        for action, official_return in row["returns"].items():
            continues = (row["sample_id"] == old_training[0]["sample_id"] and
                         action == old_training[0]["policy_action_id"] and action != baseline)
            if continues and row["score_at_20s"] is None:
                raise ValueError("Original delayed-reward transition lacks a 20 s boundary")
            transitions.append({"step": 4, "state_key": state_key, "action_id": action,
                                "sample_id": row["sample_id"],
                                "reward": (row["score_at_20s"] - row["score_at_8s"]
                                           if continues else official_return),
                                "selectable": True, "terminal": not continues,
                                "next_step": 10 if continues else None,
                                "next_state_key": p2_state["state_key"] if continues else None})
    solved = solve_empirical_horizon(states, transitions, baseline)
    holdout_action = model_action(fit, old_holdout)
    holdout = {"sample_id": old_holdout["sample_id"], "prefix_id": old_holdout["prefix_id"],
               "predicted_action_id": holdout_action,
               "predicted_official_return": old_holdout["returns"].get(holdout_action),
               "baseline_official_return": old_holdout["returns"][baseline],
               "paired_advantage": (old_holdout["returns"].get(holdout_action) -
                                    old_holdout["returns"][baseline]
                                    if holdout_action in old_holdout["returns"] else None),
               "used_for_fit_or_dp": False,
               "does_not_distinguish_x_from_y": True}
    leave_one_out = []
    for held in new_rows:
        train = old_training + [row for row in new_rows if row["sample_id"] != held["sample_id"]]
        loo_fit = fit_one_bit_state(train, predicates, baseline,
                                    min_leaf_samples=min_leaf_samples, complexity_penalty=complexity_penalty)
        selected = model_action(loo_fit, held)
        leave_one_out.append({"heldout_prefix_id": held["prefix_id"],
                              "selected_predicate_id": (loo_fit["predicate"] or {}).get("id"),
                              "predicted_action_id": selected,
                              "paired_advantage": (held["returns"].get(selected) - held["returns"][baseline]
                                                   if selected in held["returns"] else None),
                              "oracle_action_id_diagnostic": sorted(
                                  held["returns"], key=lambda action: (-round(held["returns"][action], 9),
                                                                       action != baseline, action))[0]})
    sensitivity = []
    for support in (1, 2, 3):
        for penalty in (0.0, 0.25, 0.5, 0.75, 1.0):
            tested = fit_one_bit_state(training, predicates, baseline,
                                       min_leaf_samples=support, complexity_penalty=penalty)
            sensitivity.append({"min_leaf_samples": support, "complexity_penalty": penalty,
                                "selected_bits": tested["selected_bits"],
                                "predicate_id": (tested["predicate"] or {}).get("id"),
                                "selected_mean_return": tested["selected_mean_return"],
                                "selected_advantage": tested["selected_advantage"]})
    return {"version": VERSION, "input_bundle_sha256": {"full_game": prior["full_game_bundle_sha256"],
                                                          "p2": prior["p2_bundle_sha256"],
                                                          "discovery": discovery["sha256"],
                                                          "actions": actions["sha256"]},
            "frozen_challenge": {"prefix_id": challenge["prefix_id"],
                                 "frozen_predicted_action": frozen_action,
                                 "observed_best_action": actions["paired"]["decisive_pair"][
                                     "empirical_winner_with_baseline_tie_break"],
                                 "paired_scan_advantage": actions["paired"]["decisive_pair"][
                                     "scan_minus_baseline_official_return"]},
            "training_rows": training, "prior_heldout_validation": holdout,
            "fit": fit, "states": states, "transitions": transitions, "solved": solved,
            "new_prefix_leave_one_out": leave_one_out, "sensitivity": sensitivity,
            "scope": "Empirical one-bit state and fully supported actions, fixed continuation except the measured P2 branch"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-game-bundle", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--p2-bundle", type=Path, default=DEFAULT_P2)
    parser.add_argument("--discovery-bundle", type=Path, default=PROJECT / "ubuntu_logs/oracle_state_refinement_discovery_bundle.zip")
    parser.add_argument("--actions-bundle", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-leaf-samples", type=int, default=2)
    parser.add_argument("--complexity-penalty", type=float, default=0.0)
    args = parser.parse_args(argv)
    config = resolve_config(DEFAULT_CONFIG)
    prior = optimize(config, args.full_game_bundle, args.p2_bundle)
    discovery = checked_discovery(args.discovery_bundle, config)
    actions = checked_actions(args.actions_bundle, discovery, config)
    result = solve(config, prior, discovery, actions,
                   min_leaf_samples=args.min_leaf_samples,
                   complexity_penalty=args.complexity_penalty)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("analysis_summary.json", {k: result[k] for k in
                                ("version", "input_bundle_sha256", "frozen_challenge", "scope")}),
                        ("training_rows.json", result["training_rows"]),
                        ("state_fit.json", result["fit"]),
                        ("finite_horizon_states.json", result["states"]),
                        ("finite_horizon_transitions.json", result["transitions"]),
                        ("optimized_policy.json", result["solved"]),
                        ("prior_heldout_validation.json", result["prior_heldout_validation"]),
                        ("new_prefix_leave_one_out.json", result["new_prefix_leave_one_out"]),
                        ("sensitivity.json", result["sensitivity"])):
        atomic_json(output / name, value)
    print(json.dumps({"status": "passed", "output": str(output),
                      "selected_predicate": (result["fit"]["predicate"] or {}).get("id"),
                      "policy": result["solved"]["values"],
                      "heldout_advantage": result["prior_heldout_validation"]["paired_advantage"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
