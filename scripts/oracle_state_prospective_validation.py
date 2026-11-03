"""Prospectively test a frozen local Oracle on an unused synthetic vision seed.

Four previously chosen action prefixes are replayed with a new seed. The 8 s
state is discovered first, then eligible exact-root prefixes receive paired
baseline/scan full-game runs. Policy actions are computed from the frozen
cross-validated fit without consulting the new outcomes.
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
from finite_state_optimizer import feature_bin
from oracle_auto_local_solver import optimize
from oracle_pilot import DEFAULT_CONFIG, atomic_json, digest, read, resolve_config, run_task, task_base
from oracle_pilot_discovery import boundary_evidence
from oracle_pilot_v21 import package_artifacts, sha256_file
from oracle_state_refinement_actions import action_outcome, checked_discovery
from oracle_state_refinement_solver import (checked_actions, checked_support, model_action,
                                            solve)

VERSION = "oracle_state_prospective_validation_v1"
NEW_GAME_VERSION = "oracle_state_new_game_validation_v1"
NEW_GAME_TRANSFER_VERSION = "oracle_state_new_game_validation_v2"
DEFAULT_FULL = workspace_root() / "experiments/oracle-local-p1-full-game/oracle_local_p1_full_game_bundle.zip"
DEFAULT_P2 = workspace_root() / "experiments/oracle-p2-second-stage/oracle_p2_second_stage_bundle.zip"
DEFAULT_DISCOVERY = (workspace_root() / "experiments/oracle-state-refinement-discovery/"
                     "oracle_state_refinement_discovery_bundle.zip")
DEFAULT_ACTIONS = (workspace_root() / "experiments/oracle-state-refinement-actions/"
                   "oracle_state_refinement_actions_bundle.zip")
DEFAULT_SUPPORT = (workspace_root() / "experiments/oracle-state-refinement-support/"
                   "oracle_state_refinement_support_bundle.zip")
DEFAULT_OUTPUT = workspace_root() / "experiments/oracle-state-prospective-validation"
DEFAULT_NEW_GAME_DISCOVERY = (workspace_root() / "experiments/oracle-state-new-game-validation/"
                              "oracle_state_new_game_validation_bundle.zip")


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
    root_canonical = discovery["plans"][0]["expected_boundaries"][4]["canonical_state"]
    root_signature = state_signature(root_canonical)
    if any(state_signature(plan["expected_boundaries"][4]["canonical_state"]) != root_signature
           for plan in discovery["plans"]):
        raise ValueError("Original exact-root samples do not share observable state fields")
    return {"version": VERSION, "fit": fit, "cross_validation": cv, "prior_fit": prior["state_fit"],
            "root_state_id": fit["base_state_id"], "scan_action_id": discovery["scan_action_id"],
            "baseline_action_id": config["continuation_action_id"],
            "root_state_signature": root_signature, "allow_rule_fingerprint_transfer": False,
            "synthetic_seed": config["test_synthetic_seeds"][0],
            "plans": discovery["plans"], "input_bundle_sha256": solved["input_bundle_sha256"]}


def state_signature(canonical_state):
    """Only the rule fingerprint may differ when game seed is the sole rule change."""
    fields = json.loads(canonical_state)
    if not isinstance(fields, list) or len(fields) != 5 or fields[2:4] != ["limited", "valid"]:
        raise ValueError("Unexpected limited canonical state schema")
    return {"schema_version": fields[0], "view": fields[2],
            "status": fields[3], "fields": fields[4]}


def discovery_task(config, frozen, plan):
    task = task_base(config, "discover", frozen["synthetic_seed"])
    task.update(experiment_version=frozen["version"], prefix_id=plan["prefix_id"],
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
    return state_from_evidence(record["task"]["prefix_id"], record["task_key"],
                               boundaries, frames, frozen)


def state_from_evidence(prefix_id, task_key, boundaries, frames, frozen):
    xy = [frame["own_xy"] for frame in frames if abs(frame["simulation_seconds"] - 8.0) < 1e-8]
    if len(xy) != 1:
        raise RuntimeError(f"Prospective prefix has no unique public 8 s position: {task_key}")
    boundary = boundaries[4]
    fields = boundary["limited"]["fields"]
    if (boundary["limited"]["status"] == "valid" and
            dict(state_signature(boundary["limited"]["canonical_state"])["fields"]) != fields):
        raise ValueError("Canonical limited-state fields disagree with the public boundary")
    relevant = (boundary["limited"]["status"] == "valid" and not boundary["finished"] and
                fields.get("target.active") == "P2" and fields.get("A.region_id") == "target_P1" and
                fields.get("A.hp", 0) > 0)
    exact = relevant and boundary["limited"]["state_id"] == frozen["root_state_id"]
    compatible = relevant and state_signature(boundary["limited"]["canonical_state"]) == frozen[
        "root_state_signature"]
    transferable = bool(exact or (frozen["allow_rule_fingerprint_transfer"] and compatible))
    return {"prefix_id": prefix_id, "discovery_task_key": task_key,
            "own_xy": xy[0], "score_at_8s": boundary["score"],
            "r2l_state_id": boundary["limited"]["state_id"],
            "eligible_exact_root": bool(exact), "compatible_root_fields": bool(compatible),
            "eligible_transfer_root": transferable,
            "frozen_predicted_action_id": (model_action(frozen["fit"],
                {"own_xy": xy[0]}) if transferable else frozen["baseline_action_id"]),
            "old_x_predicted_action_id": (model_action(frozen["prior_fit"],
                {"own_xy": xy[0]}) if transferable else frozen["baseline_action_id"]),
            "expected_boundaries": [{**boundary_evidence(b), "collapsed": b["collapsed"]}
                                    for b in boundaries]}


def checked_reused_new_game_discovery(path, frozen, runtime_config):
    """Import completed 8 s states; never reuse prior action outcomes."""
    path = Path(path).expanduser().resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or archive.testzip() is not None or
                any(name.startswith("/") or ".." in Path(name).parts for name in names)):
            raise ValueError("Prior new-game ZIP has duplicate, unsafe, or corrupt members")
        manifest = json.loads(archive.read("manifest.json"))
        checks = {}
        for entry in manifest["files"]:
            payload = archive.read(entry["path"])
            actual = hashlib.sha256(payload).hexdigest()
            if len(payload) != entry["bytes"] or actual != entry["sha256"]:
                raise ValueError(f"Prior new-game member checksum mismatch: {entry['path']}")
            checks[entry["path"]] = actual
        sums = dict(line.split("  ", 1)[::-1] for line in archive.read("SHA256SUMS").decode().splitlines())
        if (manifest["file_count"] != len(checks) or len(checks) != len(manifest["files"]) or
                sums != checks or set(names) != set(checks) | {"manifest.json", "SHA256SUMS"}):
            raise ValueError("Prior new-game archive manifest mismatch")
        provenance = json.loads(archive.read("experiment_provenance.json"))
        summary = json.loads(archive.read("run_summary.json"))
        old_states = json.loads(archive.read("discovered_states.json"))
        old_fit = json.loads(archive.read("frozen_state_fit.json"))
        old_config = json.loads(archive.read("resolved_config.json"))
        old_config.pop("source_sha256", None)
        expected_config = dict(runtime_config)
        expected_config.pop("source_sha256", None)
        if (provenance["version"] != NEW_GAME_VERSION or summary["version"] != NEW_GAME_VERSION or
                summary["status"] != "passed" or summary["discovered_prefixes"] != 4 or
                summary["eligible_exact_root"] != 0 or summary["selectable_pairs"] != 0 or
                provenance["source_sha256"] != manifest["source_sha256"] or
                provenance["input_bundle_sha256"] != frozen["input_bundle_sha256"] or
                provenance["cross_validated_fit_sha256"] != digest(frozen["fit"]) or
                provenance["game_seed"] != runtime_config["game_seed"] or
                provenance["prospective_synthetic_seed"] != frozen["synthetic_seed"] or
                provenance["prefix_ids"] != [p["prefix_id"] for p in frozen["plans"]] or
                old_fit != frozen["fit"] or old_config != expected_config or len(old_states) != 4):
            raise ValueError("Prior new-game discovery has different seeds, model, source or plan")
        states = {}
        evidence = {}
        for plan, old_state in zip(frozen["plans"], old_states):
            key = old_state["discovery_task_key"]
            members = [f"tasks/{key}.json", f"raw/{key}/boundaries.json",
                       f"raw/{key}/frame_index.json", f"raw/{key}/decisions.json"]
            for member in members:
                evidence[member] = archive.read(member)
            record = json.loads(evidence[members[0]])
            task = record["task"]
            boundaries = json.loads(evidence[members[1]])
            frames = json.loads(evidence[members[2]])
            decisions = json.loads(evidence[members[3]])
            if (record["task_key"] != key or digest(task)[:20] != key or
                    record["status"] != "passed" or
                    record["result"]["stop_reason"] != "planned_boundary" or
                    task["mode"] != "discover" or task["prefix_id"] != plan["prefix_id"] or
                    task["prefix_action_ids"] != plan["prefix_action_ids"] or
                    task["game_seed"] != runtime_config["game_seed"] or
                    task["synthetic_seed"] != frozen["synthetic_seed"] or
                    task["source_sha256"] != provenance["source_sha256"] or
                    len(boundaries) != 5 or len(decisions) != 4 or
                    any(decision["action_id"] != task["prefix_action_ids"][i] or
                        decision["validator_status"] != "accepted"
                        for i, decision in enumerate(decisions))):
                raise ValueError(f"Prior new-game discovery task differs from raw evidence: {key}")
            state = state_from_evidence(plan["prefix_id"], key, boundaries, frames, frozen)
            for field in ("prefix_id", "own_xy", "score_at_8s", "r2l_state_id",
                          "expected_boundaries"):
                if state[field] != old_state[field]:
                    raise ValueError(f"Prior new-game discovery summary differs: {key} {field}")
            states[plan["prefix_id"]] = state
    return {"path": path, "sha256": sha256_file(path), "states": states, "evidence": evidence}


def paired_task(config, frozen, state, prefix_action_ids, action):
    baseline = frozen["baseline_action_id"]
    mode = "oracle_local_baseline_prefix" if action == baseline else "oracle_local_eval_prefix"
    task = task_base(config, mode, frozen["synthetic_seed"])
    task.update(experiment_version=frozen["version"], prefix_id=state["prefix_id"],
                sample_id=state["discovery_task_key"] + "_04",
                prefix_action_ids=prefix_action_ids,
                expected_boundaries=state["expected_boundaries"], max_steps=20,
                target_state_id=state["r2l_state_id"],
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
        if not state.get("eligible_transfer_root", state["eligible_exact_root"]):
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
    return {"version": frozen["version"], "frozen_before_new_seed_outcomes": True,
            "synthetic_seed": frozen["synthetic_seed"],
            "discovered_prefixes": len(discovered),
            "eligible_exact_root": sum(state["eligible_exact_root"] for state in discovered),
            "eligible_compatible_root": len(pairs),
            "selectable_pairs": len(valid), "pairs": pairs,
            "mean_frozen_policy_advantage": (sum(p["frozen_policy_advantage"] for p in valid) / len(valid)
                                              if valid else None),
            "mean_all_scan_advantage": (sum(p["all_scan_advantage"] for p in valid) / len(valid)
                                        if valid else None),
            "evaluation_limit": ("One new game seed and synthetic seed on previously selected action prefixes; canonical fields matched with seed-only rule fingerprint transfer"
                                 if frozen["version"] == NEW_GAME_TRANSFER_VERSION else
                                 "One new game seed and synthetic seed on previously selected action prefixes; not random prefix sampling"
                                 if frozen["version"] == NEW_GAME_VERSION else
                                 "One unseen synthetic seed on previously selected action prefixes; not a new game seed or random scene sample")}


def run_validation(config, frozen, output, max_new_tasks=12, reused_discovery=None):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"version": frozen["version"], "source_sha256": config["source_sha256"],
                  "input_bundle_sha256": frozen["input_bundle_sha256"],
                  "cross_validated_fit_sha256": digest(frozen["fit"]),
                  "root_state_signature_sha256": digest(frozen["root_state_signature"]),
                  "rule_transfer": "game_seed_only" if frozen["allow_rule_fingerprint_transfer"] else "none",
                  "reused_discovery_bundle_sha256": (reused_discovery["sha256"]
                                                      if reused_discovery else None),
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
    atomic_json(output / "frozen_root_state_signature.json", frozen["root_state_signature"])
    if reused_discovery:
        for member, payload in reused_discovery["evidence"].items():
            destination = output / "reused_discovery_evidence" / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
    budget = 0
    discovered, outcomes = [], []
    for plan in frozen["plans"]:
        if reused_discovery:
            state = reused_discovery["states"][plan["prefix_id"]]
            last_wall_seconds = 0.0
        else:
            task = discovery_task(config, frozen, plan)
            key = digest(task)[:20]
            record_path = output / "tasks" / f"{key}.json"
            if not record_path.exists() or read(record_path)["status"] != "passed":
                if budget >= max_new_tasks:
                    atomic_json(output / "capture_progress.json", {"stage": "incomplete",
                                "completed_discovery": len(discovered), "completed_paired_tasks": len(outcomes),
                                "new_tasks_this_run": budget})
                    return {"version": frozen["version"], "status": "incomplete",
                            "completed_discovery": len(discovered), "completed_paired_tasks": len(outcomes)}
                budget += 1
            record = run_task(output, task, retry_failed=True)
            state = discovered_state(record, frozen)
            last_wall_seconds = record["wall_seconds"]
        discovered.append(state)
        print(f"prospective_discover {len(discovered)}/4 {plan['prefix_id']} "
              f"eligible={state['eligible_transfer_root']}", flush=True)
        if state["eligible_transfer_root"]:
            fake_discovery = {"fit": {"base_state_id": state["r2l_state_id"],
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
                        return {"version": frozen["version"], "status": "incomplete",
                                "completed_discovery": len(discovered),
                                "completed_paired_tasks": len(outcomes)}
                    budget += 1
                record = run_task(output, task, retry_failed=True)
                outcome = action_outcome(record, action, state, fake_discovery)
                outcomes.append(outcome)
                last_wall_seconds = record["wall_seconds"]
                print(f"prospective_pair {len(outcomes)} {plan['prefix_id']} "
                      f"{'baseline' if action == frozen['baseline_action_id'] else 'scan'} "
                      f"selectable={outcome['selectable']}", flush=True)
        atomic_json(output / "capture_progress.json", {"stage": "running",
                    "completed_discovery": len(discovered), "completed_paired_tasks": len(outcomes),
                    "new_tasks_this_run": budget, "last_task_wall_seconds": last_wall_seconds})
    comparison = compare_prospective(frozen, discovered, outcomes)
    atomic_json(output / "discovered_states.json", discovered)
    atomic_json(output / "paired_outcomes.json", outcomes)
    atomic_json(output / "prospective_validation.json", comparison)
    summary = {"version": frozen["version"], "status": "passed", "discovered_prefixes": len(discovered),
               "eligible_exact_root": comparison["eligible_exact_root"],
               "eligible_compatible_root": comparison["eligible_compatible_root"],
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
    parser.add_argument("--validation-game-seed", type=int)
    parser.add_argument("--validation-synthetic-seed", type=int)
    parser.add_argument("--reuse-discovery-bundle", type=Path)
    parser.add_argument("--max-new-tasks", type=int, default=12)
    args = parser.parse_args(argv)
    if args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    training_config = resolve_config(DEFAULT_CONFIG)
    frozen = frozen_experiment(training_config, args.full_game_bundle, args.p2_bundle,
                               args.discovery_bundle, args.actions_bundle, args.support_bundle)
    if args.validation_game_seed is not None:
        if args.validation_game_seed == training_config["game_seed"]:
            parser.error("--validation-game-seed must differ from training game seed")
        synthetic_seed = (args.validation_synthetic_seed if args.validation_synthetic_seed is not None
                          else training_config["test_synthetic_seeds"][1])
        if synthetic_seed in training_config["training_synthetic_seeds"]:
            parser.error("Validation synthetic seed must be outside training seeds")
        frozen["version"] = (NEW_GAME_TRANSFER_VERSION if args.reuse_discovery_bundle
                             else NEW_GAME_VERSION)
        frozen["synthetic_seed"] = synthetic_seed
        frozen["allow_rule_fingerprint_transfer"] = True
        config = {**training_config, "game_seed": args.validation_game_seed}
    else:
        if args.validation_synthetic_seed is not None or args.reuse_discovery_bundle is not None:
            parser.error("New seeds and reused discovery require --validation-game-seed")
        config = training_config
    reused_discovery = (checked_reused_new_game_discovery(args.reuse_discovery_bundle, frozen, config)
                        if args.reuse_discovery_bundle else None)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    invocation_path = output / "invocations.json"
    invocations = read(invocation_path) if invocation_path.exists() else []
    invocations.append({"utc_started": datetime.now(timezone.utc).isoformat(),
                        "command": ["bash", "scripts/run_oracle_pilot_ubuntu.sh",
                                    "oracle_state_new_game" if frozen["version"] == NEW_GAME_VERSION else
                                    "oracle_state_prospective",
                                    *(argv if argv is not None else sys.argv[1:])]})
    atomic_json(invocation_path, invocations)
    started = time.monotonic()
    try:
        result = run_validation(config, frozen, output, args.max_new_tasks, reused_discovery)
    except Exception as exc:
        atomic_json(output / "run_failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                  "elapsed_wall_seconds": round(time.monotonic() - started, 3)})
        result = {"version": frozen["version"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if result["status"] == "passed":
        (output / "run_failure.json").unlink(missing_ok=True)
    result["invocation_wall_seconds"] = round(time.monotonic() - started, 3)
    filename = ("oracle_state_new_game_validation_v2_bundle.zip"
                if frozen["version"] == NEW_GAME_TRANSFER_VERSION else
                "oracle_state_new_game_validation_bundle.zip"
                if frozen["version"] == NEW_GAME_VERSION else
                "oracle_state_prospective_validation_bundle.zip")
    archive = package_artifacts(output, filename,
                                {"version": frozen["version"], "source_sha256": config["source_sha256"],
                                 "cross_validated_fit_sha256": digest(frozen["fit"])})
    result["bundle"] = str(archive)
    result["bundle_sha256"] = sha256_file(archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
