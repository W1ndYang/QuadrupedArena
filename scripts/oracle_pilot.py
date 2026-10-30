"""Resumable Ubuntu collector for the coarse-state Oracle feasibility Pilot.

Each Isaac branch is a fresh process. There is no claim of a restorable Isaac
snapshot; verified prefix replay is the only branch construction mechanism.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

from adaptive_motion import load_config
from arena_paths import PROJECT, workspace_root
from oracle_state_v1 import ACTION_SPACE_VERSION, action_catalog, action_template

DEFAULT_CONFIG = PROJECT / "configs/oracle_coarse_pilot.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def resolve_config(path):
    raw = read(path)
    if raw["rules_config"] != "trial_rotating.json" or raw["scene"] != "formal_courtyard":
        raise ValueError("Pilot runtime currently supports only the formal rotating courtyard")
    cfg = load_config(PROJECT / "configs" / raw["rules_config"])
    cfg["seed"] = raw["game_seed"]
    catalog = action_catalog(cfg)
    if len(catalog) != 44 or ACTION_SPACE_VERSION != "comparison_v1_candidate":
        raise ValueError("Current action catalogue is not the required 44-item candidate")
    ids = {a.template_id for a in catalog}
    def exact(request):
        item = action_template(request["skill"], request["params"], cfg)
        if item.template_id not in ids:
            raise ValueError("Request is not in the actual catalogue")
        return item.template_id
    raw["continuation_action_id"] = exact(raw["continuation_request"])
    raw["smoke_action_ids"] = [exact(x) for x in raw["smoke_requests"]]
    raw["discovery_action_ids"] = [exact(x) for x in raw["discovery_requests"]]
    raw["action_catalog_version"] = ACTION_SPACE_VERSION
    raw["action_catalog"] = [{"action_id": a.template_id, "skill": a.skill,
                              "params": dict(a.params)} for a in catalog]
    sources = ["run.sh", "scripts/launch.py", "scripts/vision_trial.py",
               "scripts/oracle_pilot.py", "scripts/oracle_pilot_runtime.py",
               "scripts/oracle_state_v1.py", "scripts/semantic_skills.py",
               "scripts/synthetic_vision_mvp.py", "scripts/vision_core.py",
               "scripts/llm_trial.py", "scripts/trial_rules.py",
               "scripts/adaptive_motion.py", "configs/trial_rotating.json",
               "configs/trial_adaptive.json", "configs/trial_v3.json",
               "configs/oracle_coarse_pilot.json"]
    raw["source_sha256"] = digest({name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
                                    for name in sources})
    if raw["candidate_action_duration"] != 2 or raw["total_rollout_horizon"] < 2 or raw["total_rollout_horizon"] % 2:
        raise ValueError("Pilot runtime requires one 2 s candidate action and 2 s horizon increments")
    if set(raw["training_synthetic_seeds"]) & set(raw["test_synthetic_seeds"]):
        raise ValueError("Training and test synthetic seeds overlap")
    if raw["baseline_list"] != ["fixed_capture_active_fast"]:
        raise ValueError("Only the implemented same-interface fixed baseline may be claimed")
    return raw


def task_base(config, mode, synthetic_seed):
    return {"mode": mode, "game_seed": config["game_seed"],
            "source_sha256": config["source_sha256"],
            "synthetic_seed": synthetic_seed,
            "pixel_noise_std": config["pixel_noise_std"],
            "miss_probability": config["miss_probability"],
            "continuation_action_id": config["continuation_action_id"]}


def run_task(output, task, *, retry_failed=False, force=False):
    key = digest(task)[:20]
    record_path = output / "tasks" / f"{key}.json"
    if record_path.exists() and not force:
        old = read(record_path)
        if old["status"] == "passed" or (old["status"] == "partial" and not retry_failed):
            return old
    if record_path.exists():
        history = output / "history"
        history.mkdir(parents=True, exist_ok=True)
        index = 1 + len(list(history.glob(f"{key}_*.json")))
        shutil.copy2(record_path, history / f"{key}_{index:03d}.json")
        prior_raw = output / "raw" / key
        if prior_raw.exists():
            prior_raw.rename(history / f"{key}_{index:03d}_raw")
    task_file = output / "task_specs" / f"{key}.json"
    atomic_json(task_file, task)
    command = ["bash", str(PROJECT / "run.sh"), "--vision-trial", "--semantic-skills",
               "--oracle-pilot-task", str(task_file), "--headless"]
    started = time.monotonic()
    try:
        proc = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True,
                              timeout=1400, check=False)
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        code = 124
    match = re.search(r"^Run: (.+)$", stdout, re.MULTILINE)
    run_dir = Path(match.group(1)).resolve() if match else None
    raw = output / "raw" / key
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "launcher_stdout.txt").write_text(stdout)
    (raw / "launcher_stderr.txt").write_text(stderr)
    result = None
    if run_dir and run_dir.is_dir():
        for filename in ("pilot_result.json", "result.json", "boundaries.json", "decisions.json",
                         "synthetic_samples.json", "frame_index.json", "protocol.json", "runtime.log", "process.json"):
            source = run_dir / filename
            if source.is_file():
                shutil.copy2(source, raw / filename)
        source = run_dir / "environment.json"
        if source.is_file():
            shutil.copy2(source, raw / "environment.json")
        for source in sorted((run_dir / "diagnostics").glob("reference_*.json")):
            dest = raw / "diagnostics" / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        frames = []
        for source in sorted((run_dir / "frames").glob("*.png")):
            frames.append({"file": source.name, "bytes": source.stat().st_size,
                           "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
            if task["mode"] in ("preflight", "discover") and len(frames) <= 5:
                dest = raw / "frames" / source.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        atomic_json(raw / "frame_hashes.json", frames)
        result_file = run_dir / "pilot_result.json"
        if result_file.exists():
            result = read(result_file)
    record = {"task_key": key, "task": task, "run_dir": str(run_dir) if run_dir else None,
              "raw_dir": str(raw), "launcher_exit_code": code,
              "wall_seconds": round(time.monotonic() - started, 3),
              "result": result,
              "status": "passed" if code == 0 and result and result["status"] == "passed" else "partial"}
    atomic_json(record_path, record)
    rebuild_jsonl(output)
    return record


def all_records(output):
    return [read(p) for p in sorted((output / "tasks").glob("*.json"))]


def rebuild_jsonl(output):
    records = [x for x in all_records(output) if x["task"]["mode"] == "branch"]
    dest = output / "branches.jsonl"
    temp = dest.with_name(dest.name + ".tmp")
    with temp.open("w") as stream:
        for item in records:
            raw = Path(item["raw_dir"])
            decisions = read(raw / "decisions.json") if (raw / "decisions.json").exists() else []
            boundaries = read(raw / "boundaries.json") if (raw / "boundaries.json").exists() else []
            target = item["task"]["target_step"]
            candidate = decisions[target] if len(decisions) > target else None
            start = boundaries[target] if len(boundaries) > target else None
            row = {"experiment_id": output.name, "run_id": item["task_key"],
                   "sample_id": item["task"].get("sample_id"),
                   "r2l_state_id": item["task"]["expected_state_id"],
                   "state_fields": start["limited"]["fields"] if start else None,
                   "r2f_diagnostic_file": start["full_diagnostic_file"] if start else None,
                   "game_seed": item["task"]["game_seed"],
                   "synthetic_seed": item["task"]["synthetic_seed"],
                   "rules_config": "trial_rotating.json", "opponent_policy": "HumanPolicy",
                   "continuation_policy": item["task"]["continuation_action_id"],
                   "action_space_version": ACTION_SPACE_VERSION,
                   "action_id": item["task"]["candidate_action_id"],
                   "candidate": candidate, "official_outcome": item["result"],
                   "next_state_id": boundaries[-1]["limited"]["state_id"] if boundaries else None,
                   "launcher_exit_code": item["launcher_exit_code"],
                   "status": item["status"]}
            stream.write(canonical(row) + "\n")
    temp.replace(dest)


def replay_signature(record):
    raw = Path(record["raw_dir"])
    if record["status"] != "passed" or not (raw / "boundaries.json").exists():
        return None
    return [{"t": b["simulation_seconds"], "id": b["limited"]["state_id"],
             "reference": b["reference_digest_1e6"], "score": b["score"],
             "skill": b["skill_status"], "finished": b["finished"]}
            for b in read(raw / "boundaries.json")]


def preflight(output, config):
    task = task_base(config, "preflight", config["training_synthetic_seeds"][0])
    task["max_steps"] = 1
    first = run_task(output, {**task, "repeat": 1}, force=True)
    second = run_task(output, {**task, "repeat": 2}, force=True)
    a, b = replay_signature(first), replay_signature(second)
    check = {"status": "passed" if a is not None and a == b else "blocked",
             "source_sha256": config["source_sha256"],
             "method": "fresh_process_same_seed_and_prefix_replay_not_snapshot_restore",
             "first_task_key": first["task_key"], "second_task_key": second["task_key"],
             "first_signature": a, "second_signature": b,
             "difference": None if a == b and a is not None else "R2-L/reference/score/Skill/terminal mismatch or failed run",
             "snapshot_restore_available": False, "same_frame_verified": False,
             "render_sync_quality": "blocked"}
    atomic_json(output / "preflight.json", check)
    return check


def smoke(output, config):
    if (not (output / "preflight.json").exists() or
            read(output / "preflight.json")["status"] != "passed" or
            read(output / "preflight.json").get("source_sha256") != config["source_sha256"]):
        raise ValueError("Run and pass preflight before smoke")
    first = next(x for x in all_records(output) if x["task"]["mode"] == "preflight" and
                 x["task"]["repeat"] == 1 and x["task"].get("source_sha256") == config["source_sha256"])
    start = read(Path(first["raw_dir"]) / "boundaries.json")[0]
    results = []
    for action_id in config["smoke_action_ids"]:
        task = task_base(config, "branch", config["training_synthetic_seeds"][0])
        task.update(sample_id="preflight_t0", target_step=0, prefix_action_ids=[],
                    expected_state_id=start["limited"]["state_id"],
                    expected_reference_digest=start["reference_digest_1e6"],
                    candidate_action_id=action_id,
                    horizon_seconds=config["total_rollout_horizon"])
        results.append(run_task(output, task))
    passed = [x for x in results if x["status"] == "passed"]
    per = [x["wall_seconds"] for x in passed]
    estimate = {"status": "passed" if len(passed) == len(results) else "partial",
                "source_sha256": config["source_sha256"],
                "branch_wall_seconds": per, "median_branch_wall_seconds": statistics.median(per) if per else None,
                "estimated_44_wall_seconds": statistics.median(per) * 44 if per else None,
                "estimated_528_wall_seconds": statistics.median(per) * 528 if per else None,
                "estimated_20_paired_eval_wall_seconds": statistics.median(per) * 40 * 5 if per else None,
                "note": "Last estimate extrapolates 8 s branches to 40 s episodes; startup and terminal timing may differ.",
                "gpu_memory_peak_mib": None, "cpu_memory_peak_mib": None,
                "task_keys": [x["task_key"] for x in results]}
    atomic_json(output / "smoke.json", estimate)
    return estimate


def discover(output, config, max_runs):
    records = []
    for synthetic_seed in config["training_synthetic_seeds"]:
        for action_id in config["discovery_action_ids"]:
            if len(records) >= max_runs:
                break
            task = task_base(config, "discover", synthetic_seed)
            task.update(max_steps=config["discovery_steps"], continuation_action_id=action_id)
            records.append(run_task(output, task))
    groups = defaultdict(list)
    for record in records:
        if record["status"] != "passed":
            continue
        raw = Path(record["raw_dir"])
        boundaries = read(raw / "boundaries.json")
        decisions = read(raw / "decisions.json")
        for step, boundary in enumerate(boundaries):
            if boundary["limited"]["status"] != "valid":
                continue
            prefix = [d["action_id"] for d in decisions[:step]]
            reference = read(raw / boundary["full_diagnostic_file"])["snapshot"]
            physical = {key: value for key, value in reference.items() if key not in
                        ("visual_track", "visual_seen_at", "visual_source")}
            sample = {"sample_id": record["task_key"] + f"_{step:02d}",
                      "state_id": boundary["limited"]["state_id"],
                      "fields": boundary["limited"]["fields"],
                      "full_diagnostic_file": str(raw / boundary["full_diagnostic_file"]),
                      "reference_digest_1e6": boundary["reference_digest_1e6"],
                      "physical_reference_digest": digest(physical),
                      "game_seed": record["task"]["game_seed"],
                      "synthetic_seed": record["task"]["synthetic_seed"],
                      "target_step": step, "prefix_action_ids": prefix,
                      "source_run": record["task_key"], "source_frequency": None}
            groups[sample["state_id"]].append(sample)
    selected = []
    total_boundary_visits = sum(len(v) for v in groups.values())
    def category(fields):
        if fields.get("A.visual_memory_bin") == "fresh_aim_usable":
            return "visual_threat"
        if fields.get("A.active_target_relation") in ("inside_capture", "capture_edge"):
            return "capture_or_dwell"
        if fields.get("own_skill_status") in ("unsafe_route", "no_route", "stopped", "emergency_cover"):
            return "replan_or_blocked"
        return "move_or_route"
    for state_id, candidates in groups.items():
        distinct = {}
        for item in candidates:
            distinct.setdefault(item["physical_reference_digest"], item)
        values = list(distinct.values())
        if len(values) >= 2:
            snapshots = [read(item["full_diagnostic_file"])["snapshot"] for item in values[:3]]
            differences = [key for key in snapshots[0] if
                           len({canonical(snapshot[key]) for snapshot in snapshots}) > 1]
            selected.append({"state_id": state_id, "fields": values[0]["fields"],
                             "category": category(values[0]["fields"]),
                             "offline_reference_differences": differences,
                             "observed_boundary_visits": len(candidates),
                             "observed_visit_fraction": len(candidates) / total_boundary_visits if total_boundary_visits else None,
                             "samples": values[:3], "available_distinct_full_states": len(values)})
    selected.sort(key=lambda x: (-len(x["samples"]), x["state_id"]))
    diverse = []
    seen_categories = set()
    for item in selected:
        if item["category"] not in seen_categories:
            diverse.append(item)
            seen_categories.add(item["category"])
    selected = (diverse + [item for item in selected if item not in diverse])[:4]
    result = {"status": "ready" if len(selected) == 4 and all(len(x["samples"]) == 3 for x in selected) else "reduced_incomplete",
              "source_sha256": config["source_sha256"],
              "requested_states": 4, "requested_samples_per_state": 3,
              "groups": selected, "discovery_runs": len(records),
              "note": "Only same canonical state IDs and distinct reference fingerprints are counted. No sample is invented."}
    atomic_json(output / "samples.json", result)
    return result


def pilot_actions(output, config, max_branches, retry_failed):
    if (not (output / "preflight.json").exists() or
            read(output / "preflight.json")["status"] != "passed" or
            read(output / "preflight.json").get("source_sha256") != config["source_sha256"]):
        raise ValueError("Verified replay preflight is required")
    if not (output / "smoke.json").exists() or read(output / "smoke.json").get("source_sha256") != config["source_sha256"]:
        raise ValueError("Run smoke and inspect throughput first")
    if not (output / "samples.json").exists() or read(output / "samples.json").get("source_sha256") != config["source_sha256"]:
        discover(output, config, config["max_discovery_runs"])
    samples = read(output / "samples.json")
    count = 0
    for group in samples["groups"]:
        for sample in group["samples"]:
            for action in config["action_catalog"]:
                task = task_base(config, "branch", sample["synthetic_seed"])
                task.update(sample_id=sample["sample_id"], target_step=sample["target_step"],
                            prefix_action_ids=sample["prefix_action_ids"],
                            expected_state_id=sample["state_id"],
                            expected_reference_digest=sample["reference_digest_1e6"],
                            candidate_action_id=action["action_id"],
                            horizon_seconds=config["total_rollout_horizon"])
                existing = output / "tasks" / f"{digest(task)[:20]}.json"
                if existing.exists():
                    prior = read(existing)
                    if prior["status"] == "passed" or (prior["status"] == "partial" and not retry_failed):
                        continue
                if count >= max_branches:
                    return count
                outcome = run_task(output, task, retry_failed=retry_failed)
                if outcome["result"] and outcome["result"].get("stop_reason") in (
                        "replay_mismatch", "full_replay_mismatch", "reference_replay_mismatch"):
                    raise RuntimeError("Systematic replay mismatch; branch batch stopped after preserving evidence")
                count += 1
    return count


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_actions(output, config):
    records = [x for x in all_records(output) if x["task"].get("source_sha256") == config["source_sha256"] and
               x["task"]["mode"] == "branch" and
               x["task"].get("sample_id") != "preflight_t0"]
    rows = []
    by_state = defaultdict(lambda: defaultdict(dict))
    for item in records:
        task = item["task"]
        raw = Path(item["raw_dir"])
        decisions = read(raw / "decisions.json") if (raw / "decisions.json").exists() else []
        d = decisions[task["target_step"]] if len(decisions) > task["target_step"] else None
        valid = bool(item["status"] == "passed" and d and d["validator_status"] == "accepted")
        value = item["result"]["official_delta_score"] if valid else None
        row = {"state_id": task["expected_state_id"], "sample_id": task["sample_id"],
               "action_id": task["candidate_action_id"], "available": int(valid),
               "official_delta_score": value, "execution_status": d["execution_status"] if d else "not_started",
               "error_class": d.get("error_class") if d else item["result"]["stop_reason"] if item["result"] else "launcher_failed",
               "task_key": item["task_key"]}
        rows.append(row)
        by_state[row["state_id"]][row["sample_id"]][row["action_id"]] = row
    write_csv(output / "analysis" / "action_value_matrix.csv", rows, list(rows[0]) if rows else
              ["state_id", "sample_id", "action_id", "available", "official_delta_score", "execution_status", "error_class", "task_key"])
    write_csv(output / "analysis" / "action_availability_matrix.csv",
              [{"state_id": r["state_id"], "sample_id": r["sample_id"], "action_id": r["action_id"],
                "available": r["available"], "execution_status": r["execution_status"],
                "error_class": r["error_class"]} for r in rows],
              ["state_id", "sample_id", "action_id", "available", "execution_status", "error_class"])
    ranks = []
    for state_id, samples in by_state.items():
        for sample_id, actions in samples.items():
            ordered = sorted((r for r in actions.values() if r["available"]),
                             key=lambda r: (-r["official_delta_score"], r["action_id"]))
            for rank, row in enumerate(ordered, 1):
                ranks.append({"state_id": state_id, "sample_id": sample_id,
                              "rank": rank, "action_id": row["action_id"],
                              "official_delta_score": row["official_delta_score"]})
    write_csv(output / "analysis" / "per_sample_action_rank.csv", ranks,
              ["state_id", "sample_id", "rank", "action_id", "official_delta_score"])
    summary = []
    table = {}
    for state_id, samples in by_state.items():
        ids = sorted(samples)
        if not ids:
            continue
        common = set.intersection(*(set(a for a, row in samples[s].items() if row["available"]) for s in ids))
        if not common:
            summary.append({"state_id": state_id, "samples": len(ids), "common_legal": 0,
                            "common_action": None, "mean_regret": None, "best_agreement": None})
            continue
        mean = {a: statistics.mean(samples[s][a]["official_delta_score"] for s in ids) for a in common}
        common_action = max(sorted(common), key=lambda a: mean[a])
        best = {s: max((r for r in samples[s].values() if r["available"]),
                       key=lambda r: r["official_delta_score"]) for s in ids}
        regrets = [best[s]["official_delta_score"] - samples[s][common_action]["official_delta_score"] for s in ids]
        summary.append({"state_id": state_id, "samples": len(ids), "common_legal": len(common),
                        "common_action": common_action, "mean_value": mean[common_action],
                        "mean_regret": statistics.mean(regrets),
                        "worst_common_value": min(samples[s][common_action]["official_delta_score"] for s in ids),
                        "best_agreement": sum(best[s]["action_id"] == common_action for s in ids) / len(ids),
                        "full_44_coverage": all(len(samples[s]) == 44 for s in ids)})
        if len(ids) >= 3 and all(len(samples[s]) == 44 for s in ids):
            table[state_id] = common_action
    write_csv(output / "analysis" / "common_actions.csv", summary,
              ["state_id", "samples", "common_legal", "common_action", "mean_value", "mean_regret",
               "worst_common_value", "best_agreement", "full_44_coverage"])
    policy = {"version": "coarse_pilot_candidate_v1", "action_space_version": ACTION_SPACE_VERSION,
              "source_sha256": config["source_sha256"],
              "game_seed_profile": config["game_seed"], "table": table,
              "fallback_action_id": config["continuation_action_id"],
              "training_synthetic_seeds": config["training_synthetic_seeds"],
              "status": "candidate_not_independently_validated"}
    atomic_json(output / "analysis" / "policy.json", policy)
    return summary


def bootstrap_ci(values):
    if len(values) < 2:
        return None
    rng = random.Random(20261030)
    means = sorted(statistics.mean(rng.choice(values) for _ in values) for _ in range(2000))
    return [means[50], means[1949]]


def policy_eval(output, config, max_pairs, retry_failed):
    policy_file = output / "analysis" / "policy.json"
    if not policy_file.exists():
        analyze_actions(output, config)
    policy = read(policy_file)
    if policy["source_sha256"] != config["source_sha256"] or policy["game_seed_profile"] != config["game_seed"]:
        raise ValueError("Policy source or game seed profile differs from current Pilot")
    if not policy["table"]:
        raise ValueError("No complete 44-action common policy state; evaluation would be baseline-only")
    seeds = config["test_synthetic_seeds"][:max_pairs]
    for seed in seeds:
        for mode in ("eval_policy", "eval_baseline"):
            task = task_base(config, mode, seed)
            if mode == "eval_policy":
                task["policy_table"] = policy["table"]
            run_task(output, task, retry_failed=retry_failed)
    rows = []
    lookup = {(x["task"]["synthetic_seed"], x["task"]["mode"]): x for x in all_records(output)
              if x["task"].get("source_sha256") == config["source_sha256"] and
              x["task"]["mode"] in ("eval_policy", "eval_baseline") and x["status"] == "passed"}
    for seed in seeds:
        a, b = lookup.get((seed, "eval_policy")), lookup.get((seed, "eval_baseline"))
        if a is None or b is None:
            continue
        raw = Path(a["raw_dir"])
        decisions = read(raw / "decisions.json")
        hits = sum(d["source"] == "table" for d in decisions)
        fallback = sum(d["source"] == "fallback" for d in decisions)
        rows.append({"synthetic_seed": seed, "pilot_score": a["result"]["official_score"],
                     "baseline_score": b["result"]["official_score"],
                     "paired_delta": round(a["result"]["official_score"] - b["result"]["official_score"], 6),
                     "table_hits": hits, "fallbacks": fallback,
                     "pilot_wall_seconds": a["wall_seconds"], "baseline_wall_seconds": b["wall_seconds"]})
    write_csv(output / "analysis" / "paired_eval.csv", rows,
              ["synthetic_seed", "pilot_score", "baseline_score", "paired_delta", "table_hits", "fallbacks",
               "pilot_wall_seconds", "baseline_wall_seconds"])
    vals = [r["paired_delta"] for r in rows]
    summary = {"paired_episodes": len(rows), "requested_pairs": len(seeds),
               "mean_delta": statistics.mean(vals) if vals else None,
               "median_delta": statistics.median(vals) if vals else None,
               "sample_sd": statistics.stdev(vals) if len(vals) > 1 else None,
               "bootstrap_95_ci": bootstrap_ci(vals),
               "wins": sum(x > 0 for x in vals), "ties": sum(x == 0 for x in vals),
               "losses": sum(x < 0 for x in vals),
               "table_hits": sum(r["table_hits"] for r in rows),
               "fallbacks": sum(r["fallbacks"] for r in rows),
               "limitation": "Independent synthetic vision seeds, fixed game seed because canonical R2-L profile ID includes cfg.seed."}
    atomic_json(output / "analysis" / "eval_summary.json", summary)
    return summary


def report(output, config):
    pre = read(output / "preflight.json") if (output / "preflight.json").exists() else None
    smoke_data = read(output / "smoke.json") if (output / "smoke.json").exists() else None
    samples = read(output / "samples.json") if (output / "samples.json").exists() else None
    common = analyze_actions(output, config)
    eval_data = read(output / "analysis" / "eval_summary.json") if (output / "analysis" / "eval_summary.json").exists() else None
    done = [x for x in all_records(output) if x["task"].get("source_sha256") == config["source_sha256"] and
            x["task"]["mode"] == "branch"]
    total = sum(x["status"] == "passed" for x in done)
    lines = ["# Oracle 粗粒度策略可行性 Pilot 运行汇总", "",
             f"- 结论：`INCONCLUSIVE`（只有完成完整动作矩阵、有效独立评估和充分统计后才可重新判定）",
             f"- 场景：`{config['scene']}`；规则：`{config['rules_config']}`；固定对手：`HumanPolicy`；动作目录：`{ACTION_SPACE_VERSION}` 44 项。",
             f"- 分支：{total}/{len(done)} 成功；目标上限 528。",
             f"- 重放预检：{pre['status'] if pre else '未运行'}；Smoke：{smoke_data['status'] if smoke_data else '未运行'}；样本组：{len(samples['groups']) if samples else 0}。",
             f"- 独立评估配对数：{eval_data['paired_episodes'] if eval_data else 0}；平均得分差：{eval_data['mean_delta'] if eval_data else '未计算'}。",
             "", "## 研究边界", "",
             "本汇总仅用官方本方 score 增量为价值；游戏没有对手官方 score、官方分差或胜负字段，均记为不可用。",
             "动作值是 2 秒候选动作加统一后续动作至 8 秒的有限时域经验值，不是最优 Q*。",
             "完整 Isaac/Skill/裁判状态不能用坐标或 R2-F 字典恢复；每个分支均重启并重放前缀。",
             "配置 seed 进入现有规范 state_id；独立评估只变化未参与场景指纹的合成视觉 seed，游戏 RNG seed 固定。这降低外推性。",
             "`render_sync_quality=blocked`，`same_frame_verified=false`。未调用 DeepSeek。",
             "", "## 状态动作公共选择", "",
             "| state_id | 样本 | 共同可用动作 | 公共动作 | 平均 regret | 完整 44 覆盖 |",
             "|---|---:|---:|---|---:|---|"]
    for item in common:
        lines.append(f"| `{item['state_id'][:28]}...` | {item['samples']} | {item['common_legal']} | `{item['common_action']}` | {item['mean_regret']} | {item.get('full_44_coverage',False)} |")
    lines += ["", "详见 `analysis/action_value_matrix.csv`、`analysis/common_actions.csv`、`analysis/policy.json`、`analysis/paired_eval.csv`，以及每个 task 的原始 JSON 与 `branches.jsonl`。", ""]
    (output / "analysis" / "report.md").write_text("\n".join(lines))


def package(output):
    files = [p for p in sorted(output.rglob("*")) if p.is_file() and p.name not in
             ("pilot_bundle.zip", "pilot_bundle.zip.tmp", "manifest.json", "SHA256SUMS")]
    entries = []
    for path in files:
        relative = str(path.relative_to(output))
        entries.append({"path": relative, "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"file_count": len(entries), "files": entries,
                "same_frame_verified": False, "render_sync_quality": "blocked"}
    atomic_json(output / "manifest.json", manifest)
    (output / "SHA256SUMS").write_text("".join(f"{e['sha256']}  {e['path']}\n" for e in entries))
    archive = output / "pilot_bundle.zip"
    temp = output / "pilot_bundle.zip.tmp"
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for path in files + [output / "manifest.json", output / "SHA256SUMS"]:
            z.write(path, path.relative_to(output))
    with zipfile.ZipFile(temp) as z:
        for entry in entries:
            if entry["path"] not in z.namelist():
                raise ValueError("Archive is missing a manifest file")
            data = z.read(entry["path"])
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("Archive checksum mismatch")
        if z.testzip() is not None:
            raise ValueError("Archive CRC failure")
    temp.replace(archive)
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "preflight", "smoke", "pilot_actions", "policy_eval", "retry_task", "report", "package"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=workspace_root() / "experiments/oracle-pilot/default")
    parser.add_argument("--max-branches", type=int, default=44)
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--task-key", help="For retry_task: exact task key printed in tasks/ and branches.jsonl")
    args = parser.parse_args(argv)
    config = resolve_config(args.config)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "resolved_config.json", config)
    if args.mode == "inspect":
        result = {"actions": len(config["action_catalog"]), "action_space_version": ACTION_SPACE_VERSION,
                  "output": str(output), "game_seed_profile": config["game_seed"],
                  "warning": "Changing game seed changes canonical R2-L state IDs; test seeds vary synthetic vision only."}
    elif args.mode == "preflight":
        result = preflight(output, config)
    elif args.mode == "smoke":
        result = smoke(output, config)
    elif args.mode == "pilot_actions":
        result = {"processed_branch_slots": pilot_actions(output, config, args.max_branches, args.retry_failed)}
    elif args.mode == "policy_eval":
        result = policy_eval(output, config, args.max_pairs, args.retry_failed)
    elif args.mode == "retry_task":
        if not args.task_key or not re.fullmatch(r"[0-9a-f]{20}", args.task_key):
            raise ValueError("retry_task requires --task-key with a 20-character hexadecimal key")
        source = output / "task_specs" / f"{args.task_key}.json"
        if not source.exists():
            raise ValueError("Unknown task key")
        task = read(source)
        if task.get("source_sha256") != config["source_sha256"]:
            raise ValueError("Cannot retry a task from another source version")
        result = run_task(output, task, force=True)
    else:
        result = {"mode": args.mode}
    report(output, config)
    archive = package(output)
    print(json.dumps({"result": result, "output": str(output), "zip": str(archive)}, ensure_ascii=False, indent=2))
    if args.mode in ("preflight", "smoke") and result.get("status") != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
