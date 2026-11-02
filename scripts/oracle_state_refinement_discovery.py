"""Collect new 8 s prefixes selected by automatic state-refinement diagnostics.

The second half of the existing formal prefix library is run once per prefix.
Public positions are ranked by whether equal-value one-bit predicates disagree.
No action outcome is inferred from this discovery-only stage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from adaptive_motion import load_config
from arena_paths import PROJECT, workspace_root
from finite_state_optimizer import feature_bin, geometry_predicates
from oracle_auto_local_solver import optimize
from oracle_p2_second_stage import DEFAULT_PRIOR as DEFAULT_FULL
from oracle_pilot import DEFAULT_CONFIG, atomic_json, digest, read, resolve_config, run_task, task_base
from oracle_pilot_v21 import package_artifacts, sha256_file

VERSION = "oracle_state_refinement_discovery_v1"
DEFAULT_P2 = workspace_root() / "experiments/oracle-p2-second-stage/oracle_p2_second_stage_bundle.zip"
DEFAULT_OUTPUT = workspace_root() / "experiments/oracle-state-refinement-discovery"


def equivalent_predicates(fit, predicates, reference_xy):
    """Find best-scoring splits and align each side with the training west leaf."""
    if fit["selected_bits"] != 1:
        raise ValueError("Current discovery requires a selected one-bit state model")
    lookup = {item["id"]: item for item in predicates}
    selected_value = fit["selected_mean_return"]
    results = []
    for item in fit["candidate_diagnostics"]:
        if item.get("eligible") and abs(item["mean_return"] - selected_value) < 1e-9:
            predicate = lookup[item["predicate_id"]]
            results.append({"predicate": predicate,
                            "training_west_side": feature_bin(reference_xy, predicate)})
    if len(results) < 2:
        raise ValueError("No competing equal-value state predicates remain to disambiguate")
    return results


def analyze_boundary(prefix_id, record, tied, root_state_id):
    raw = Path(record["raw_dir"])
    boundaries = read(raw / "boundaries.json") if (raw / "boundaries.json").exists() else []
    frames = read(raw / "frame_index.json") if (raw / "frame_index.json").exists() else []
    if (record["status"] != "passed" or (record.get("result") or {}).get("stop_reason") != "planned_boundary" or
            len(boundaries) != 5):
        return {"prefix_id": prefix_id, "task_key": record["task_key"],
                "status": "incomplete", "stop_reason": (record.get("result") or {}).get("stop_reason")}
    boundary = boundaries[4]
    positions = [item["own_xy"] for item in frames if abs(item["simulation_seconds"] - 8.0) < 1e-8]
    if len(positions) != 1:
        raise RuntimeError(f"Discovery prefix has no unique public 8 s position: {prefix_id}")
    fields = boundary["limited"]["fields"]
    relevant = (boundary["limited"]["status"] == "valid" and not boundary["finished"] and
                fields.get("target.active") == "P2" and fields.get("A.region_id") == "target_P1" and
                fields.get("A.hp", 0) > 0)
    labels = {item["predicate"]["id"]: feature_bin(positions[0], item["predicate"]) ==
              item["training_west_side"] for item in tied}
    disagreement = len(set(labels.values())) > 1
    exact = boundary["limited"]["state_id"] == root_state_id
    priority = (0 if relevant and exact and disagreement else
                1 if relevant and disagreement else
                2 if relevant and exact else
                3 if relevant else 4)
    return {"prefix_id": prefix_id, "task_key": record["task_key"], "status": "completed",
            "r2l_state_id": boundary["limited"]["state_id"],
            "public_own_xy": positions[0], "official_score_at_8s": boundary["score"],
            "hp_at_8s": boundary["hp"][0], "region_id": fields.get("A.region_id"),
            "active_target": fields.get("target.active"), "relevant_p1_to_p2": bool(relevant),
            "exact_prior_root_state": exact, "predicate_west_like": labels,
            "predicate_disagreement": disagreement, "priority_rank": priority}


def run_discovery(config, model, output, max_new_tasks=12, retry_failed=False):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    library = config["prefix_library"]
    plans = library["prefixes"][12:24]
    if len(plans) != 12 or library["prefix_count"] != 24:
        raise ValueError("Expected twelve previously unrun formal discovery prefixes")
    root_id = model["state_fit"]["base_state_id"]
    cfg = load_config(PROJECT / "configs" / config["rules_config"])
    predicates = geometry_predicates(cfg["arena"]["task_points"])
    tied = equivalent_predicates(model["state_fit"], predicates, model["training_rows"][0]["own_xy"])
    provenance = {"version": VERSION, "source_sha256": config["source_sha256"],
                  "full_game_bundle_sha256": model["full_game_bundle_sha256"],
                  "p2_bundle_sha256": model["p2_bundle_sha256"],
                  "prefix_library_hash": library["hash"], "discovery_prefix_ids": [p["prefix_id"] for p in plans],
                  "game_seed": config["game_seed"],
                  "synthetic_seed": config["discovery_synthetic_seed"]}
    existing = output / "experiment_provenance.json"
    if existing.exists() and read(existing) != provenance:
        raise ValueError("Discovery output belongs to another source or input")
    atomic_json(existing, provenance)
    atomic_json(output / "resolved_config.json", config)
    atomic_json(output / "frozen_state_fit.json", model["state_fit"])
    atomic_json(output / "equivalent_predicates.json", tied)
    budget = 0
    rows = []
    total_wall_seconds = 0.0
    for plan in plans:
        task = task_base(config, "discover", config["discovery_synthetic_seed"])
        task.update(experiment_version=VERSION, prefix_id=plan["prefix_id"],
                    prefix_action_ids=plan["prefix_action_ids"], max_steps=4,
                    source_model_full_game_sha256=model["full_game_bundle_sha256"])
        key = digest(task)[:20]
        path = output / "tasks" / f"{key}.json"
        if not path.exists() or (retry_failed and read(path)["status"] != "passed"):
            if budget >= max_new_tasks:
                atomic_json(output / "capture_progress.json", {"stage": "incomplete",
                            "completed_tasks": len(rows), "new_tasks_this_run": budget})
                return {"version": VERSION, "status": "incomplete", "completed_tasks": len(rows)}
            budget += 1
        record = run_task(output, task, retry_failed=retry_failed)
        total_wall_seconds += record["wall_seconds"]
        row = analyze_boundary(plan["prefix_id"], record, tied, root_id)
        rows.append(row)
        atomic_json(output / "capture_progress.json", {"stage": "running",
                    "completed_tasks": len(rows), "new_tasks_this_run": budget,
                    "last_task_wall_seconds": record["wall_seconds"]})
        print(f"state_discovery {len(rows)}/12 {plan['prefix_id']} "
              f"status={row['status']} priority={row.get('priority_rank')}", flush=True)
    ranked = sorted(rows, key=lambda item: (item.get("priority_rank", 5), item["prefix_id"]))
    analysis = {"version": VERSION, "root_r2l_state_id": root_id,
                "equivalent_predicates": tied,
                "rank_rule": "Exact-root predicate disagreement, then nearby P1-to-P2 disagreement, then other relevant states",
                "rows": ranked,
                "recommended_followup_prefix_ids": [item["prefix_id"] for item in ranked
                                                  if item.get("priority_rank", 5) <= 1],
                "outcome_values_collected": False}
    atomic_json(output / "state_discovery_analysis.json", analysis)
    summary = {"version": VERSION, "status": "passed", "attempted_tasks": len(rows),
               "completed_prefixes": sum(item["status"] == "completed" for item in rows),
               "relevant_p1_to_p2": sum(item.get("relevant_p1_to_p2", False) for item in rows),
               "predicate_disagreement_relevant": len(analysis["recommended_followup_prefix_ids"]),
               "total_task_wall_seconds": round(total_wall_seconds, 3)}
    atomic_json(output / "run_summary.json", summary)
    atomic_json(output / "capture_progress.json", {"stage": "complete", "completed_tasks": len(rows),
                "new_tasks_this_run": budget})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-game-bundle", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--p2-bundle", type=Path, default=DEFAULT_P2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tasks", type=int, default=12)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    config = resolve_config(DEFAULT_CONFIG)
    model = optimize(config, args.full_game_bundle, args.p2_bundle)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    invocation_path = output / "invocations.json"
    invocations = read(invocation_path) if invocation_path.exists() else []
    invocations.append({"utc_started": datetime.now(timezone.utc).isoformat(),
                        "command": ["bash", "scripts/run_oracle_pilot_ubuntu.sh", "oracle_state_discovery",
                                    *(argv if argv is not None else sys.argv[1:])]})
    atomic_json(invocation_path, invocations)
    started = time.monotonic()
    try:
        result = run_discovery(config, model, output, args.max_new_tasks, args.retry_failed)
    except Exception as exc:
        atomic_json(output / "run_failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                  "elapsed_wall_seconds": round(time.monotonic() - started, 3)})
        result = {"version": VERSION, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if result["status"] == "passed":
        (output / "run_failure.json").unlink(missing_ok=True)
    result["invocation_wall_seconds"] = round(time.monotonic() - started, 3)
    archive = package_artifacts(output, "oracle_state_refinement_discovery_bundle.zip",
                                {"version": VERSION, "source_sha256": config["source_sha256"],
                                 "full_game_bundle_sha256": model["full_game_bundle_sha256"],
                                 "p2_bundle_sha256": model["p2_bundle_sha256"]})
    result["bundle"] = str(archive)
    result["bundle_sha256"] = sha256_file(archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
