"""Fit and validate a one-bit local policy from paired seed-sweep outcomes.

Each distinct action prefix contributes one training row. Repeated game seeds
are averaged within that row, then held out by game seed for cross-validation.
Only two measured, executable actions are considered.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

from arena_paths import PROJECT
from finite_state_optimizer import feature_bin, fit_one_bit_state, geometry_predicates

DEFAULT_BUNDLE = PROJECT / "ubuntu_logs/oracle_state_seed_sweep_bundle.zip"
DEFAULT_CONFIG = PROJECT / "configs/trial_v3.json"
DEFAULT_OUTPUT = PROJECT / "docs/codex_report/oracle_state_seed_sweep_analysis_v1"


def checked_bundle(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or archive.testzip() is not None or
                any(name.startswith("/") or ".." in Path(name).parts for name in names)):
            raise ValueError("Bundle contains duplicate, unsafe, or corrupt members")
        manifest = json.loads(archive.read("manifest.json"))
        checks = {}
        for entry in manifest["files"]:
            payload = archive.read(entry["path"])
            actual = hashlib.sha256(payload).hexdigest()
            if actual != entry["sha256"] or len(payload) != entry["bytes"]:
                raise ValueError(f"Bundle member checksum mismatch: {entry['path']}")
            checks[entry["path"]] = actual
        sums = dict(line.split("  ", 1)[::-1] for line in
                    archive.read("SHA256SUMS").decode().splitlines())
        if (manifest["file_count"] != len(checks) or len(checks) != len(manifest["files"]) or
                checks != sums or set(names) != set(checks) | {"manifest.json", "SHA256SUMS"}):
            raise ValueError("Bundle manifest or SHA256SUMS mismatch")
        progress = json.loads(archive.read("capture_progress.json"))
        plan = json.loads(archive.read("seed_plan.json"))
        summary = json.loads(archive.read("run_summary.json"))
        if (progress["stage"] != "complete" or summary["status"] != "passed" or
                summary["completed_seeds"] != len(plan) or len(plan) < 2 or
                len({item["game_seed"] for item in plan}) != len(plan)):
            raise ValueError("Seed sweep is incomplete or its plan is inconsistent")
        frozen = json.loads(archive.read("frozen_state_fit.json"))
        seed_pairs = {}
        prefix_ids = None
        for item in plan:
            seed = item["game_seed"]
            result = json.loads(archive.read(f"game_{seed}/prospective_validation.json"))
            if (result["synthetic_seed"] != item["synthetic_seed"] or
                    result["selectable_pairs"] != len(result["pairs"]) or
                    result["eligible_compatible_root"] != len(result["pairs"]) or
                    not all(pair["both_actions_selectable"] for pair in result["pairs"])):
                raise ValueError(f"Incomplete executable pairs for game seed {seed}")
            current_ids = {pair["prefix_id"] for pair in result["pairs"]}
            if (len(current_ids) != len(result["pairs"]) or
                    (prefix_ids is not None and current_ids != prefix_ids)):
                raise ValueError("Game seeds do not share the same unique prefix set")
            prefix_ids = current_ids
            for pair in result["pairs"]:
                base = pair["baseline_outcome"]
                scan = pair["scan_outcome"]
                predicted = pair["frozen_predicted_action_id"]
                predicted_gain = (scan["official_return"] - base["official_return"]
                                  if predicted == scan["action_id"] else 0.0)
                if (base["action_id"] != frozen["baseline_action_id"] or
                        scan["action_id"] == base["action_id"] or
                        predicted not in (base["action_id"], scan["action_id"]) or
                        abs(pair["all_scan_advantage"] -
                            (scan["official_return"] - base["official_return"])) > 1e-8 or
                        abs(pair["frozen_policy_advantage"] - predicted_gain) > 1e-8):
                    raise ValueError(f"Invalid paired advantage for {seed}/{pair['prefix_id']}")
                for label in ("baseline_outcome", "scan_outcome"):
                    outcome = pair[label]
                    key = outcome["task_key"]
                    root = f"game_{seed}/raw/{key}/"
                    raw = json.loads(archive.read(root + "result.json"))
                    decisions = json.loads(archive.read(root + "decisions.json"))
                    frames = json.loads(archive.read(root + "frame_index.json"))
                    at_eight = [frame["own_xy"] for frame in frames
                                if abs(frame["simulation_seconds"] - 8.0) < 1e-8]
                    if (raw["status"] != "passed" or
                            abs(raw["evaluation_delta_score"] - outcome["official_return"]) > 1e-8 or
                            raw["official_score"] != outcome["observed_final_score_diagnostic"] or
                            at_eight != [pair["public_own_xy"]] or len(decisions) < 5 or
                            decisions[4]["action_id"] != outcome["action_id"] or
                            decisions[4]["validator_status"] != "accepted" or
                            decisions[4]["execution_status"] not in
                            ("started", "completed", "interrupted") or
                            (decisions[4].get("compiled_action") or {}).get("skill") !=
                            outcome["compiled_action_skill"] or
                            outcome["compiled_action_skill"] not in ("MOVE", "SCAN") or
                            (decisions[4].get("compiled_skill") or {}).get("state") !=
                            outcome["compiled_skill_state"] or not outcome["selectable"]):
                        raise ValueError(f"Summary disagrees with raw outcome {seed}/{key}")
            seed_pairs[seed] = result["pairs"]
    return frozen, seed_pairs, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped_rows(seed_pairs, seeds, base_state_id):
    grouped = {}
    for seed in seeds:
        for pair in seed_pairs[seed]:
            prefix_id = pair["prefix_id"]
            baseline, scan = pair["baseline_outcome"], pair["scan_outcome"]
            row = grouped.setdefault(prefix_id, {"sample_id": prefix_id,
                "base_state_id": base_state_id, "own_xy": pair["public_own_xy"],
                "returns": {baseline["action_id"]: [], scan["action_id"]: []}})
            if row["own_xy"] != pair["public_own_xy"]:
                raise ValueError("A prefix changed observable position across seeds")
            row["returns"][baseline["action_id"]].append(baseline["official_return"])
            row["returns"][scan["action_id"]].append(scan["official_return"])
    return [{**row, "returns": {action: statistics.mean(values)
                                for action, values in row["returns"].items()}}
            for _, row in sorted(grouped.items())]


def policy_action(fit, own_xy):
    label = "all" if fit["predicate"] is None else feature_bin(own_xy, fit["predicate"])
    return fit["leaves"][label]["action_id"]


def max_margin_equivalent_fit(fit, rows, predicates):
    """Resolve equal-value, equal-partition predicates by sample clearance.

    The underlying optimizer's deterministic first-predicate rule is retained
    in ``fit`` for audit. This deployment tie break only swaps predicates that
    give the same low/high assignment on every unique training position.
    """
    if fit["predicate"] is None:
        return copy.deepcopy(fit), []
    original = fit["predicate"]
    eligible = {item["predicate_id"]: item for item in fit["candidate_diagnostics"]
                if item["eligible"]}
    candidates = []
    for predicate in predicates:
        item = eligible.get(predicate["id"])
        if (item is None or abs(item["penalized_objective"] -
                                fit["selected_penalized_objective"]) > 1e-9 or
                any(feature_bin(row["own_xy"], predicate) !=
                    feature_bin(row["own_xy"], original) for row in rows)):
            continue
        index = 0 if predicate["axis"] == "x" else 1
        clearance = min(abs(row["own_xy"][index] - predicate["threshold"]) for row in rows)
        candidates.append({"predicate_id": predicate["id"],
                           "minimum_training_clearance": clearance})
    if not candidates:
        raise ValueError("Optimizer's selected predicate is missing from equivalent candidates")
    chosen = sorted(candidates, key=lambda item: (-item["minimum_training_clearance"],
                                                   item["predicate_id"]))[0]
    deployment = copy.deepcopy(fit)
    deployment["predicate"] = next(item for item in predicates if item["id"] == chosen["predicate_id"])
    return deployment, candidates


def evaluate(fit, pairs, baseline_action_id):
    values = []
    for pair in pairs:
        chosen = policy_action(fit, pair["public_own_xy"])
        baseline = pair["baseline_outcome"]["official_return"]
        if chosen == baseline_action_id:
            values.append(0.0)
        elif chosen == pair["scan_outcome"]["action_id"]:
            values.append(pair["scan_outcome"]["official_return"] - baseline)
        else:
            raise ValueError(f"Unmeasured policy action: {chosen}")
    return statistics.mean(values)


def analyze(frozen, seed_pairs, predicates, *, min_leaf_samples=2, complexity_penalty=0.1):
    seeds = sorted(seed_pairs)
    baseline = frozen["baseline_action_id"]
    rows = grouped_rows(seed_pairs, seeds, frozen["base_state_id"])
    fit = fit_one_bit_state(rows, predicates, baseline,
                            min_leaf_samples=min_leaf_samples,
                            complexity_penalty=complexity_penalty)
    deployment_fit, equivalent_predicates = max_margin_equivalent_fit(fit, rows, predicates)
    per_seed = [{"game_seed": seed, "frozen_advantage": statistics.mean(
                     pair["frozen_policy_advantage"] for pair in seed_pairs[seed]),
                 "zero_bit_all_scan_advantage": statistics.mean(
                     pair["all_scan_advantage"] for pair in seed_pairs[seed]),
                 "refit_advantage": evaluate(deployment_fit, seed_pairs[seed], baseline)}
                for seed in seeds]
    folds = []
    for held in seeds:
        train = [seed for seed in seeds if seed != held]
        train_rows = grouped_rows(seed_pairs, train, frozen["base_state_id"])
        held_fit = fit_one_bit_state(train_rows, predicates, baseline,
                                     min_leaf_samples=min_leaf_samples,
                                     complexity_penalty=complexity_penalty)
        held_fit, _ = max_margin_equivalent_fit(held_fit, train_rows, predicates)
        folds.append({"held_out_game_seed": held, "train_game_seeds": train,
                      "selected_bits": held_fit["selected_bits"],
                      "predicate_id": (held_fit["predicate"] or {}).get("id"),
                      "leaf_actions": {label: leaf["action_id"] for label, leaf in held_fit["leaves"].items()},
                      "held_out_advantage": evaluate(held_fit, seed_pairs[held], baseline)})
    return {"game_seeds": seeds, "distinct_prefixes": len(rows),
            "seed_prefix_pairs": sum(len(pairs) for pairs in seed_pairs.values()),
            "training_rows": rows, "fit": fit, "deployment_fit": deployment_fit,
            "equivalent_predicates": equivalent_predicates, "per_seed": per_seed,
            "leave_one_game_seed_out": folds,
            "mean_cross_validated_advantage": statistics.mean(
                fold["held_out_advantage"] for fold in folds),
            "interpretation_limit": "Refit performance is in-sample on repeated positions; cross-validation holds out game seeds, not new positions."}


def evaluate_forward(candidate_plan, frozen, seed_pairs):
    """Evaluate an already frozen policy on entirely new game seeds."""
    if (sorted(seed_pairs) != candidate_plan["forward_game_seeds"] or
            frozen["base_state_id"] != candidate_plan["one_bit_fit"]["base_state_id"] or
            frozen["baseline_action_id"] != candidate_plan["baseline_action_id"]):
        raise ValueError("Forward bundle differs from the frozen candidate plan")
    rows = []
    for seed in sorted(seed_pairs):
        pairs = seed_pairs[seed]
        if (sorted(pair["prefix_id"] for pair in pairs) != candidate_plan["prefix_ids"] or
                {pair["scan_outcome"]["action_id"] for pair in pairs} !=
                {candidate_plan["zero_bit_action_id"]}):
            raise ValueError("Forward seed changed the evaluated prefixes or action")
        baseline = candidate_plan["baseline_action_id"]
        rows.append({"game_seed": seed,
                     "baseline_advantage": 0.0,
                     "zero_bit_all_scan_advantage": statistics.mean(
                         pair["scan_outcome"]["official_return"] -
                         pair["baseline_outcome"]["official_return"] for pair in pairs),
                     "one_bit_advantage": evaluate(candidate_plan["one_bit_fit"], pairs, baseline)})
    return {"training_bundle_sha256": candidate_plan["training_bundle_sha256"],
            "forward_game_seeds": candidate_plan["forward_game_seeds"],
            "per_seed": rows,
            "mean_zero_bit_advantage": statistics.mean(
                row["zero_bit_all_scan_advantage"] for row in rows),
            "mean_one_bit_advantage": statistics.mean(row["one_bit_advantage"] for row in rows),
            "selection_note": "Forward policy is frozen from training; no forward reward was used to select its state or action."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-bundle", type=Path,
                        help="Optional complete forward sweep; evaluated without refitting")
    args = parser.parse_args(argv)
    frozen, seed_pairs, bundle_sha = checked_bundle(args.bundle)
    config = json.loads(args.config.read_text())
    predicates = geometry_predicates(config["arena"]["task_points"])
    result = analyze(frozen, seed_pairs, predicates)
    result["bundle_sha256"] = bundle_sha
    scan_actions = {pair["scan_outcome"]["action_id"]
                    for pairs in seed_pairs.values() for pair in pairs}
    if len(scan_actions) != 1:
        raise ValueError("The scan action changed across game seeds")
    with zipfile.ZipFile(args.bundle) as archive:
        synthetic_seeds = {item["synthetic_seed"] for item in
                           json.loads(archive.read("seed_plan.json"))}
    if len(synthetic_seeds) != 1:
        raise ValueError("The training sweep changed synthetic seeds")
    forward_seeds = list(range(max(seed_pairs) + 1, max(seed_pairs) + 4))
    candidate_plan = {"training_bundle_sha256": bundle_sha,
                      "training_game_seeds": sorted(seed_pairs),
                      "forward_game_seeds": forward_seeds,
                      "forward_synthetic_seed": next(iter(synthetic_seeds)),
                      "prefix_ids": sorted({pair["prefix_id"] for pair in next(iter(seed_pairs.values()))}),
                      "baseline_action_id": frozen["baseline_action_id"],
                      "zero_bit_action_id": next(iter(scan_actions)),
                      "one_bit_fit": result["deployment_fit"],
                      "selection_note": "The one-bit fit is in-sample; all three candidates remain frozen for forward comparison."}
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "candidate_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != candidate_plan:
        raise ValueError("Existing frozen candidate plan differs from this analysis")
    (output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    plan_path.write_text(json.dumps(candidate_plan, ensure_ascii=False, indent=2) + "\n")
    if args.holdout_bundle is not None:
        holdout_frozen, holdout_pairs, holdout_sha = checked_bundle(args.holdout_bundle)
        with zipfile.ZipFile(args.holdout_bundle) as archive:
            holdout_synthetic = {item["synthetic_seed"] for item in
                                 json.loads(archive.read("seed_plan.json"))}
        if holdout_synthetic != {candidate_plan["forward_synthetic_seed"]}:
            raise ValueError("Forward sweep changed the frozen synthetic seed")
        forward = evaluate_forward(candidate_plan, holdout_frozen, holdout_pairs)
        forward["holdout_bundle_sha256"] = holdout_sha
        (output / "forward_validation.json").write_text(
            json.dumps(forward, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output / "analysis.json"),
                      "selected_bits": result["fit"]["selected_bits"],
                      "predicate": result["deployment_fit"]["predicate"],
                      "mean_cross_validated_advantage": result["mean_cross_validated_advantage"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
