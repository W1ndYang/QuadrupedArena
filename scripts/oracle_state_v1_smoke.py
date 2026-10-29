"""Optional read-only Oracle state diagnostic at an existing vision decision boundary.

The caller supplies the already captured public packet and each frame's accepted
{time, feet}/null result. This module does not render, submit a Skill, or step a match.
"""

from oracle_state_v1 import (
    RegionCatalog, capture_reference, encode_full, encode_limited, limited_from_public,
)


def _summary(state):
    fields = dict(state.fields)
    summary = {
        "status": state.status,
        "reason": state.reason,
        "state_id": state.state_id,
        "decision_step": fields.get("decision_step"),
        "map_phase": fields.get("map_phase"),
        "A.region_id": fields.get("A.region_id"),
    }
    if state.view == "full":
        summary["B.region_id"] = fields.get("B.region_id")
    elif state.view == "limited":
        summary["A.visual_track_region"] = fields.get("A.visual_track_region")
    return summary


def encode_boundary(match, engine, rig, packet, accepted_sightings, cfg,
                    catalog: RegionCatalog, previous_action_id=None,
                    render_sync_quality="blocked"):
    """Return bounded state metadata only; no hidden snapshot or diagnostics in output."""
    if render_sync_quality not in ("blocked", "not_applicable"):
        raise ValueError("Current sync quality cannot be asserted verified here")
    if not isinstance(accepted_sightings, tuple):
        raise ValueError("Accepted sightings must be a tuple in packet frame order")
    full = encode_full(capture_reference(match, engine, rig), cfg, catalog)
    limited = encode_limited(limited_from_public(packet, accepted_sightings, cfg, previous_action_id),
                             cfg, catalog)
    return {
        "observation_id": packet["observation_id"],
        "simulation_seconds": packet["simulation_seconds"],
        "frame_count": len(accepted_sightings),
        "render_sync_quality": render_sync_quality,
        "full": _summary(full),
        "limited": _summary(limited),
    }
