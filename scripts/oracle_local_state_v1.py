"""One explicit, observable binary refinement for the 8 s P1-to-P2 case.

This leaves the global R2-L encoder and its state IDs untouched. The input is
the robot's own position from ``public_packet()['self_state']`` and map-known
P1 geometry; no opponent or reference snapshot is used by the policy.
"""
from __future__ import annotations

import math

VERSION = "oracle_local_p1_exit_v1"


def p1_west_boundary(cfg):
    point = next((p for p in cfg["arena"]["task_points"] if p["name"] == "P1"), None)
    if point is None or point["radius"] <= 0:
        raise ValueError("P1 geometry is required for the local Oracle state")
    return point["center"][0] - point["radius"] / 2


def p1_exit_state(base_state_id, decision_step, own_xy, cfg, target_state_id):
    """Return a one-bit local state only at the verified 8 s collision ID."""
    if decision_step != 4 or base_state_id != target_state_id:
        return None
    if (not isinstance(own_xy, (tuple, list)) or len(own_xy) != 2 or
            any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in own_xy)):
        raise ValueError("Local Oracle requires finite public self_state.xy")
    threshold = p1_west_boundary(cfg)
    label = "p1_west_edge" if own_xy[0] < threshold else "p1_other_exit"
    return {"version": VERSION, "base_state_id": base_state_id,
            "decision_step": decision_step, "bin": label,
            "state_key": f"{VERSION}:{base_state_id}:{label}",
            "threshold_x": threshold}
