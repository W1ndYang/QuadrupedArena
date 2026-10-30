"""Versioned, exact-ID action-prefix discovery for the Oracle Pilot.

This module reads R1 only after a formal episode has completed. R1 never chooses
an action, a prefix, or a policy key. CPU analysis is diagnostic, not Isaac proof.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from oracle_state_v1 import action_template

DISCOVERY_SCHEMA = "action_prefix_discovery_v2"
PREFIX_LIBRARY_VERSION = "fixed_formal_prefixes_v2_1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def rounded(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return value


def request(skill, **params):
    return {"skill": skill, "params": params}


def prefix_requests():
    """Two fixed 12-prefix batches. The final formal OBSERVE clears one-step history."""
    observe = request("OBSERVE", direction="north", risk="safe")
    scan = request("OBSERVE", direction="scan", risk="safe")
    east = request("OBSERVE", direction="east", risk="safe")
    south = request("OBSERVE", direction="south", risk="safe")
    west = request("OBSERVE", direction="west", risk="safe")
    cover = request("TAKE_COVER", risk="safe")
    capture = request("CAPTURE_POINT", target="active", risk="safe")
    first = [
        [request("MOVE_TO", target="P2", risk="fast"), observe, observe, observe],
        [request("TAKE_COVER", cover="High3", risk="safe"), observe, observe, observe],
        [request("TAKE_COVER", cover="Low1", risk="safe"), observe, observe, observe],
        [request("TAKE_COVER", cover="Low4", risk="safe"), observe, observe, observe],
        [south, observe, observe, observe],
        [west, observe, observe, observe],
        [request("CAPTURE_POINT", target="active", risk="fast"), observe, cover, observe],
        [request("MOVE_TO", target="P1", risk="safe"), observe, cover, observe],
        [request("CAPTURE_POINT", target="P2", risk="safe"), east, observe, observe],
        [request("MOVE_TO", target="P2", risk="safe"), east, observe, observe],
        [request("TAKE_COVER", cover="High3", risk="fast"), capture, observe, observe],
        [request("TAKE_COVER", cover="Low1", risk="fast"), capture, observe, observe],
    ]
    second = [
        [request("TAKE_COVER", cover="High1", risk="safe"), observe, observe, observe],
        [request("TAKE_COVER", cover="Low3", risk="safe"), observe, observe, observe],
        [request("CAPTURE_POINT", target="P2", risk="fast"), observe, observe, observe],
        [request("MOVE_TO", target="P1", risk="fast"), observe, observe, observe],
        [east, observe, observe, observe],
        [scan, observe, observe, observe],
        [request("MOVE_TO", target="P2", risk="fast"), cover, observe, observe],
        [request("TAKE_COVER", cover="Low4", risk="safe"), cover, observe, observe],
        [capture, west, cover, observe],
        [request("MOVE_TO", target="P2", risk="safe"), west, cover, observe],
        [request("TAKE_COVER", cover="High3", risk="safe"), south, observe, observe],
        [request("TAKE_COVER", cover="Low1", risk="safe"), south, observe, observe],
    ]
    return first + second


def build_prefix_library(cfg, catalog):
    allowed = {item.template_id for item in catalog}
    plans = []
    for index, requests in enumerate(prefix_requests()):
        if len(requests) != 4:
            raise ValueError("Every discovery prefix must have four decision windows")
        ids = [action_template(item["skill"], item["params"], cfg).template_id for item in requests]
        if not set(ids) <= allowed:
            raise ValueError("Discovery prefix contains an action outside the formal 44-item catalog")
        plan = {"prefix_id": f"apv2_{index + 1:02d}_{digest(ids)[:10]}",
                "batch": 1 if index < 12 else 2,
                "prefix_action_ids": ids,
                "actions": [{"step": step, "action_id": action_id,
                             "skill": item["skill"], "params": item["params"]}
                            for step, (action_id, item) in enumerate(zip(ids, requests))]}
        plans.append(plan)
    if len(plans) != 24 or len({tuple(p["prefix_action_ids"]) for p in plans}) != 24:
        raise ValueError("Discovery library must contain 24 unique formal prefixes")
    return {"version": PREFIX_LIBRARY_VERSION, "schema": DISCOVERY_SCHEMA,
            "common_final_action_id": plans[0]["prefix_action_ids"][-1],
            "prefix_count": 24, "hash": digest(plans), "prefixes": plans}


FIELD_CLASSES = {
    "physical_motion": {"a", "b"},
    "game_dynamic": {"finished", "collapsed", "hp", "aim", "last_valid", "cooldown_until",
                     "active_target", "generation", "triggered_at", "expires_at", "inside", "dwell", "score"},
    "skill_executor": {"active_low_level", "active_attack_enabled", "skill_status",
                       "skill_intent", "skill_generation", "skill_last_hp"},
    "hidden_dynamics": {"human_state", "human_visible", "human_memory", "human_seen_at",
                        "patrol_index", "scan_until", "last_plan"},
    "observation_memory": {"visual_track", "visual_seen_at", "visual_source", "camera_pan"},
    "metadata": {"profile_id", "tick", "t"},
}
CLASS_BY_FIELD = {field: name for name, fields in FIELD_CLASSES.items() for field in fields}
QUALIFYING_CLASSES = ("physical_motion", "game_dynamic", "skill_executor", "hidden_dynamics")


def validate_reference_fields(snapshot):
    unknown = set(snapshot) - set(CLASS_BY_FIELD)
    if unknown:
        raise ValueError(f"Unclassified R1 reference fields: {sorted(unknown)}")


def reference_summaries(snapshot):
    validate_reference_fields(snapshot)
    normalized = rounded(snapshot)
    meaningful = {k: v for k, v in normalized.items() if CLASS_BY_FIELD[k] != "metadata"}
    dynamics = {k: v for k, v in meaningful.items() if CLASS_BY_FIELD[k] in QUALIFYING_CLASSES}
    return {"meaningful_state_digest": digest(meaningful),
            "dynamics_digest": digest(dynamics),
            "class_digests": {name: digest({k: v for k, v in normalized.items()
                                            if CLASS_BY_FIELD[k] == name})
                              for name in FIELD_CLASSES}}


def reference_diff(left, right):
    validate_reference_fields(left)
    validate_reference_fields(right)
    a, b = rounded(left), rounded(right)
    differences = {name: [] for name in FIELD_CLASSES}
    details = {}
    for key in sorted(set(a) | set(b)):
        if a.get(key) == b.get(key):
            continue
        category = CLASS_BY_FIELD[key]
        if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
            for subkey in sorted(set(a[key]) | set(b[key])):
                if a[key].get(subkey) != b[key].get(subkey):
                    path = f"{key}.{subkey}"
                    differences[category].append(path)
                    details[path] = {"left": a[key].get(subkey), "right": b[key].get(subkey)}
        else:
            differences[category].append(key)
            details[key] = {"left": a.get(key), "right": b.get(key)}
    qualified = any(differences[name] for name in QUALIFYING_CLASSES)
    only_observation = bool(differences["observation_memory"]) and not qualified
    return {"categories": differences, "values": details, "qualifies": qualified,
            "observation_memory_only": only_observation,
            "qualification": "hidden_dynamics_difference" if qualified and
                             differences["hidden_dynamics"] and not any(differences[n] for n in
                             ("physical_motion", "game_dynamic", "skill_executor")) else
                             "physical_or_task_difference" if qualified else
                             "observation_memory_only" if only_observation else "metadata_only"}


def boundary_evidence(boundary):
    return {"state_id": boundary["limited"]["state_id"],
            "canonical_state": boundary["limited"]["canonical_state"],
            "replay_reference_digest": boundary["replay_reference_digest"],
            "score": boundary["score"], "skill_status": boundary["skill_status"],
            "visual_memory": boundary["visual_memory"], "finished": boundary["finished"]}


def classify_state(fields):
    if fields.get("A.visual_memory_bin") == "fresh_aim_usable":
        return "visual_threat"
    if fields.get("A.active_target_relation") in ("inside_capture", "capture_edge"):
        return "capture_or_dwell"
    if fields.get("own_skill_status") in ("unsafe_route", "no_route", "stopped", "emergency_cover"):
        return "replan_or_blocked"
    return "move_or_route"


def analyze_records(records, library, config):
    """Return samples and near misses from immutable completed formal trajectories."""
    plans = {p["prefix_id"]: p for p in library["prefixes"]}
    snapshots = {}
    visits = []
    task_status = {}
    for record in records:
        task = record["task"]
        prefix_id = task["prefix_id"]
        if (task.get("mode") != "discover" or task.get("source_sha256") != config["source_sha256"] or
                task.get("game_seed") != config["game_seed"] or
                task.get("synthetic_seed") != config["discovery_synthetic_seed"] or
                task.get("prefix_library_hash") != library["hash"]):
            raise ValueError("Discovery record has a different source, seed, or library")
        if (prefix_id not in plans or task["prefix_action_ids"] != plans[prefix_id]["prefix_action_ids"] or
                task.get("prefix_actions") != plans[prefix_id]["actions"]):
            raise ValueError("Recorded prefix does not match the versioned library")
        raw = Path(record["raw_dir"])
        result = record.get("result") or {}
        complete = record["status"] == "passed" and result.get("stop_reason") == "planned_boundary"
        boundaries = json.loads((raw / "boundaries.json").read_text()) if (raw / "boundaries.json").exists() else []
        decisions = json.loads((raw / "decisions.json").read_text()) if (raw / "decisions.json").exists() else []
        complete &= len(boundaries) == 5 and len(decisions) == 4 and all(
            b["limited"]["status"] == "valid" for b in boundaries) and all(
            d.get("validator_status") == "accepted" and d.get("action_id") == task["prefix_action_ids"][i]
            for i, d in enumerate(decisions))
        visual_samples = json.loads((raw / "synthetic_samples.json").read_text()) if (raw / "synthetic_samples.json").exists() else []
        task_status[prefix_id] = {"task_key": record["task_key"], "status": "completed" if complete else "failed",
                                  "stop_reason": result.get("stop_reason"),
                                  "wall_seconds": record.get("wall_seconds"),
                                  "blocked_actions": sum(d.get("execution_status") == "accepted_but_blocked" for d in decisions),
                                  "visual_samples": len(visual_samples),
                                  "sightings": sum(s.get("sighting") is not None for s in visual_samples),
                                  "accepted_sightings": sum(s.get("ingest_status") == "accepted" for s in visual_samples),
                                  "failed_step": next((i for i, d in enumerate(decisions)
                                                       if d.get("validator_status") != "accepted"), None)}
        if not complete:
            continue
        replay = [boundary_evidence(b) for b in boundaries]
        for step, boundary in enumerate(boundaries):
            if boundary["limited"]["status"] != "valid":
                continue
            snapshot_file = raw / boundary["full_diagnostic_file"]
            snapshot = json.loads(snapshot_file.read_text())["snapshot"]
            summary = reference_summaries(snapshot)
            sample_id = record["task_key"] + f"_{step:02d}"
            snapshots[sample_id] = snapshot
            visits.append({"sample_id": sample_id, "prefix_id": prefix_id,
                           "source_run": record["task_key"], "target_step": step,
                           "state_id": boundary["limited"]["state_id"],
                           "canonical_state": boundary["limited"]["canonical_state"],
                           "fields": boundary["limited"]["fields"],
                           "full_diagnostic_file": str(snapshot_file),
                           "replay_reference_digest": boundary["replay_reference_digest"],
                           "reference_digest_1e6": boundary["reference_digest_1e6"],
                           **summary,
                           "game_seed": task["game_seed"], "synthetic_seed": task["synthetic_seed"],
                           "prefix_action_ids": task["prefix_action_ids"][:step],
                           "full_prefix_action_ids": task["prefix_action_ids"],
                           "replay_boundaries": replay[:step + 1],
                           "source_frequency": None})
    by_state = defaultdict(list)
    for visit in visits:
        by_state[(visit["state_id"], visit["canonical_state"])].append(visit)
    qualified_groups = []
    nonqualifying_exact = []
    for (state_id, canonical_state), candidates in by_state.items():
        distinct = {}
        for item in candidates:
            distinct.setdefault(item["meaningful_state_digest"], item)
        distinct_dynamics = {}
        for item in distinct.values():
            distinct_dynamics.setdefault(item["dynamics_digest"], item)
        values = list(distinct_dynamics.values())
        if len(values) >= 2:
            pair_diffs = []
            for i, left in enumerate(values[:3]):
                for right in values[i + 1:3]:
                    diff = reference_diff(snapshots[left["sample_id"]], snapshots[right["sample_id"]])
                    if not diff["qualifies"]:
                        raise ValueError("Dynamics digest differed without a qualifying field difference")
                    pair_diffs.append({"left": left["sample_id"], "right": right["sample_id"], **diff})
            qualified_groups.append({"state_id": state_id, "canonical_state": canonical_state,
                                     "fields": values[0]["fields"], "category": classify_state(values[0]["fields"]),
                                     "samples": values[:3], "available_distinct_full_states": len(values),
                                     "observed_boundary_visits": len(candidates),
                                     "pair_differences": pair_diffs})
        elif len(distinct) > 1:
            a, b = list(distinct.values())[:2]
            nonqualifying_exact.append({"state_id": state_id, "left": a["sample_id"],
                                        "right": b["sample_id"],
                                        "difference": reference_diff(snapshots[a["sample_id"]], snapshots[b["sample_id"]])})
    qualified_groups.sort(key=lambda x: (-len(x["samples"]), x["state_id"]))
    diverse, seen = [], set()
    for group in qualified_groups:
        if group["category"] not in seen:
            diverse.append(group)
            seen.add(group["category"])
    selected = (diverse + [g for g in qualified_groups if g not in diverse])[:4]
    for group in selected:
        group["observed_visit_fraction"] = group["observed_boundary_visits"] / len(visits)

    near_pairs = []
    field_counts = Counter()
    promising_pairs = 0
    for i, left in enumerate(visits):
        for right in visits[i + 1:]:
            if left["prefix_id"] == right["prefix_id"] or left["target_step"] != right["target_step"]:
                continue
            if left["state_id"] == right["state_id"]:
                continue
            differing = sorted(k for k in set(left["fields"]) | set(right["fields"])
                               if left["fields"].get(k) != right["fields"].get(k))
            if not differing:
                continue
            rdiff = reference_diff(snapshots[left["sample_id"]], snapshots[right["sample_id"]])
            if len(differing) <= 3:
                field_counts.update(differing)
                promising_pairs += int(rdiff["qualifies"])
            near_pairs.append({"left": left["sample_id"], "right": right["sample_id"],
                               "boundary_step": left["target_step"], "left_state_id": left["state_id"],
                               "right_state_id": right["state_id"], "differing_fields": differing,
                               "reference_difference": rdiff})
    near_pairs.sort(key=lambda p: (len(p["differing_fields"]), p["boundary_step"], p["left"], p["right"]))
    attempted = len(records)
    samples = {"discovery_schema": DISCOVERY_SCHEMA, "source_sha256": config["source_sha256"],
               "prefix_library_version": library["version"], "prefix_library_hash": library["hash"],
               "scene": config["scene"], "rules_config": config["rules_config"],
               "opponent_policy": config["opponent_policy"], "game_seed": config["game_seed"],
               "synthetic_seed": config["discovery_synthetic_seed"],
               "planned_prefixes": len([p for p in library["prefixes"] if p["batch"] <= config["discovery_batch"]]),
               "attempted_prefixes": attempted,
               "completed_prefixes": sum(s["status"] == "completed" for s in task_status.values()),
               "failed_prefixes": sum(s["status"] == "failed" for s in task_status.values()),
               "visual_samples": sum(s["visual_samples"] for s in task_status.values()),
               "sightings": sum(s["sightings"] for s in task_status.values()),
               "accepted_sightings": sum(s["accepted_sightings"] for s in task_status.values()),
               "boundary_visits": len(visits), "unique_state_ids": len({x["state_id"] for x in visits}),
               "qualifying_group_count": len(qualified_groups), "groups": selected,
               "status": "ready" if qualified_groups else "reduced_incomplete",
               "prefix_results": task_status, "same_frame_verified": False,
               "render_sync_quality": "blocked"}
    near = {"discovery_schema": DISCOVERY_SCHEMA, "prefix_library_hash": library["hash"],
            "exact_qualifying_groups": len(qualified_groups),
            "nonqualifying_exact_match_count": len(nonqualifying_exact),
            "nonqualifying_exact_matches": nonqualifying_exact[:20],
            "near_pair_count": len(near_pairs), "closest_pairs": near_pairs[:30],
            "top_blocking_fields": field_counts.most_common(20),
            "promising_near_pair_count": promising_pairs,
            "expansion_recommended": not qualified_groups and promising_pairs > 0}
    return samples, near


def near_miss_markdown(near):
    lines = ["# Oracle Pilot v2 近似碰撞诊断", "",
             f"- 合格精确碰撞组：{near['exact_qualifying_groups']}",
             f"- 不合格精确匹配（仅观测/元数据差异）：{near['nonqualifying_exact_match_count']}",
             f"- 不同 ID 的近似候选对：{near['near_pair_count']}",
             f"- 有依据扩展第二批：{'是' if near['expansion_recommended'] else '否'}", "",
             "近似匹配只用于诊断，不是正式样本，也不能进入动作价值矩阵。", "",
             "## 常见阻碍字段", "",
             "| R2-L 字段 | ≤3 字段差异的候选对数 |", "|---|---:|"]
    lines += [f"| `{field}` | {count} |" for field, count in near["top_blocking_fields"]]
    lines += ["", "`decision_step` 是决策边界标识；`own_previous_action` 是完整的最近一次动作模板；视觉 mask 是实际窗口观测。是否调整这些状态字段需另立研究任务，本轮不改编码。", "", "## 最接近的状态对", "",
              "| 边界 | 样本对 | 不同的 R2-L 字段 |", "|---:|---|---|"]
    for pair in near["closest_pairs"][:12]:
        lines.append(f"| {pair['boundary_step']} | `{pair['left']}` / `{pair['right']}` | " +
                     ", ".join(f"`{x}`" for x in pair["differing_fields"]) + " |")
    lines.append("")
    return "\n".join(lines)
