"""Small, domain-independent empirical state refinement and finite-horizon DP.

Rows contain only observations available to a policy and measured official
returns. Unmeasured or non-executable actions have no estimated value. The
solver never fills an unknown successor with zero or ranks survival on ties.
"""
from __future__ import annotations

from collections import defaultdict
import math
import statistics


def geometry_predicates(task_points):
    """Generate deterministic one-bit tests from known map geometry."""
    predicates = []
    for point in task_points:
        name, center, radius = point["name"], point["center"], point["radius"]
        if len(center) != 2 or radius <= 0:
            raise ValueError("Task point needs a 2D center and positive radius")
        for axis, coordinate in (("x", center[0]), ("y", center[1])):
            for offset in (-0.5, 0.0, 0.5):
                predicates.append({"id": f"{name}:{axis}:{offset:+.1f}r",
                                   "point": name, "axis": axis, "radius_fraction": offset,
                                   "threshold": coordinate + offset * radius})
    return predicates


def feature_bin(own_xy, predicate):
    if not isinstance(own_xy, (list, tuple)) or len(own_xy) != 2:
        raise ValueError("A public two-coordinate own_xy is required")
    coordinate = own_xy[0 if predicate["axis"] == "x" else 1]
    if not isinstance(coordinate, (int, float)) or not math.isfinite(coordinate):
        raise ValueError("Public own_xy must be finite")
    return "low" if coordinate < predicate["threshold"] else "high"


def _best_action(rows, baseline_action_id):
    sample_ids = {row["sample_id"] for row in rows}
    if len(sample_ids) != len(rows):
        raise ValueError("Each training state needs distinct replay samples")
    actions = set.intersection(*(set(row["returns"]) for row in rows))
    eligible = {action: statistics.mean(row["returns"][action] for row in rows)
                for action in actions if all(row["returns"][action] is not None for row in rows)}
    if baseline_action_id not in eligible:
        raise ValueError("The baseline needs measured, executable returns in every leaf")
    action = sorted(eligible, key=lambda item: (-round(eligible[item], 9),
                                                item != baseline_action_id, item))[0]
    return {"action_id": action, "mean_return": eligible[action],
            "eligible_action_means": eligible, "sample_ids": sorted(sample_ids)}


def fit_one_bit_state(rows, predicates, baseline_action_id, *, min_leaf_samples=1,
                      complexity_penalty=0.0):
    """Maximize observed official return with at most one geometric state bit.

    Predicate order breaks equal-value split ties. A no-split model wins exact
    ties, so an extra bit is used only when it improves measured value.
    """
    if not rows or min_leaf_samples < 1 or complexity_penalty < 0:
        raise ValueError("Training rows, positive support, and nonnegative complexity penalty are required")
    base_state_ids = {row["base_state_id"] for row in rows}
    if len(base_state_ids) != 1:
        raise ValueError("Fit one colliding base state at a time")
    baseline = _best_action(rows, baseline_action_id)
    candidates = [{"predicate": None, "leaves": {"all": baseline},
                   "mean_return": baseline["mean_return"],
                   "objective": baseline["mean_return"], "bits": 0}]
    diagnostics = []
    for predicate in predicates:
        groups = {label: [row for row in rows if feature_bin(row["own_xy"], predicate) == label]
                  for label in ("low", "high")}
        if min(len(group) for group in groups.values()) < min_leaf_samples:
            diagnostics.append({"predicate_id": predicate["id"], "eligible": False,
                                "reason": "leaf_support", "leaf_counts": {k: len(v) for k, v in groups.items()}})
            continue
        try:
            leaves = {label: _best_action(group, baseline_action_id) for label, group in groups.items()}
        except ValueError as exc:
            diagnostics.append({"predicate_id": predicate["id"], "eligible": False,
                                "reason": str(exc), "leaf_counts": {k: len(v) for k, v in groups.items()}})
            continue
        mean_return = sum(len(groups[label]) * leaves[label]["mean_return"] for label in groups) / len(rows)
        candidates.append({"predicate": predicate, "leaves": leaves,
                           "mean_return": mean_return,
                           "objective": mean_return - complexity_penalty, "bits": 1})
        diagnostics.append({"predicate_id": predicate["id"], "eligible": True,
                            "mean_return": mean_return,
                            "penalized_objective": mean_return - complexity_penalty,
                            "leaf_counts": {k: len(v) for k, v in groups.items()},
                            "chosen_actions": {k: v["action_id"] for k, v in leaves.items()}})
    selected = sorted(enumerate(candidates), key=lambda item: (-round(item[1]["objective"], 9),
                                                               item[1]["bits"], item[0]))[0][1]
    return {"base_state_id": next(iter(base_state_ids)),
            "baseline_action_id": baseline_action_id,
            "no_split_mean_return": baseline["mean_return"],
            "selected_mean_return": selected["mean_return"],
            "selected_advantage": selected["mean_return"] - baseline["mean_return"],
            "selected_penalized_objective": selected["objective"],
            "complexity_penalty_per_bit": complexity_penalty,
            "minimum_leaf_samples": min_leaf_samples,
            "selected_bits": selected["bits"],
            "predicate": selected["predicate"], "leaves": selected["leaves"],
            "candidate_diagnostics": diagnostics,
            "selection_rule": "Max measured official return minus per-bit penalty; one-bit budget; no split wins exact ties"}


def refined_state_key(base_state_id, own_xy, fit):
    predicate = fit["predicate"]
    return (base_state_id if predicate is None else
            f"{base_state_id}|{predicate['id']}:{feature_bin(own_xy, predicate)}")


def solve_empirical_horizon(states, transitions, baseline_action_id):
    """Backward induction over fully supported empirical finite-horizon edges.

    Each state specifies its required sample IDs. An action is eligible only
    with one executable outcome per required sample and known continuation
    values for every nonterminal successor.
    """
    required = {}
    for item in states:
        key = (item["step"], item["state_key"])
        samples = item["sample_ids"]
        if key in required or not samples or len(samples) != len(set(samples)):
            raise ValueError("State keys and required sample IDs must be unique")
        required[key] = set(samples)
    grouped = defaultdict(lambda: defaultdict(list))
    for edge in transitions:
        key = (edge["step"], edge["state_key"])
        if key not in required:
            raise ValueError(f"Transition refers to an unknown state: {key}")
        if not isinstance(edge["reward"], (int, float)) or not math.isfinite(edge["reward"]):
            raise ValueError("Transition reward must be finite official score delta")
        if not edge["terminal"] and edge["next_step"] <= edge["step"]:
            raise ValueError("Nonterminal edges must advance the decision step")
        grouped[key][edge["action_id"]].append(edge)
    values, policy, diagnostics = {}, {}, []
    for key in sorted(required, reverse=True):
        candidates = []
        for action_id, outcomes in sorted(grouped[key].items()):
            samples = [edge["sample_id"] for edge in outcomes]
            if len(samples) != len(set(samples)) or set(samples) != required[key] or not all(
                    edge["selectable"] for edge in outcomes):
                diagnostics.append({"step": key[0], "state_key": key[1], "action_id": action_id,
                                    "eligible": False, "reason": "missing_or_blocked_sample"})
                continue
            if any(not edge["terminal"] and (edge["next_step"], edge["next_state_key"]) not in values
                   for edge in outcomes):
                diagnostics.append({"step": key[0], "state_key": key[1], "action_id": action_id,
                                    "eligible": False, "reason": "unexpanded_successor"})
                continue
            q = statistics.mean(edge["reward"] +
                                (0.0 if edge["terminal"] else
                                 values[(edge["next_step"], edge["next_state_key"])])
                                for edge in outcomes)
            candidates.append((q, action_id))
            diagnostics.append({"step": key[0], "state_key": key[1], "action_id": action_id,
                                "eligible": True, "estimated_official_return": q,
                                "sample_count": len(outcomes)})
        if not candidates:
            raise ValueError(f"No fully supported executable action for {key}")
        value, action = sorted(candidates, key=lambda item: (-round(item[0], 9),
                                                              item[1] != baseline_action_id,
                                                              item[1]))[0]
        values[key], policy[key] = value, action
    return {"values": [{"step": key[0], "state_key": key[1], "official_return": values[key],
                         "action_id": policy[key]} for key in sorted(values)],
            "action_diagnostics": diagnostics,
            "selection_rule": "Empirical backward induction; full sample support and known successors; baseline wins exact ties"}
