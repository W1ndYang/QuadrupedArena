"""Pure Python G1-B candidate builder for directly visible foot pixels.

The payload follows ``semantic_skills.VisualMemory.update``: exactly ``time``
and ``feet``. The surrounding record is experimental and is not a Skill call.
"""

import math


SCHEMA = "g1b.synthetic_sighting.v1"


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _safe(value):
    """Keep diagnostics serializable even for failed non-finite inputs."""
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or type(value) in (str, int, float, bool):
        return value
    return str(value)


def build_sighting_candidate(projection, anchor_ray, simulation_time, frame_file,
                             width, height):
    """Return a candidate record, conservatively omitting uncertain sightings.

    ``simulation_time`` must be read from the simulator for the captured frame;
    ``frame_file`` must identify an RGB frame that was actually saved. Neither
    is inferred from a scenario name or a fixed physics step count.
    """
    diagnostics = {
        "projection": _safe(projection),
        "sighting_anchor_ray": _safe(anchor_ray),
        "simulation_time_seconds": _safe(simulation_time),
        "frame_file": _safe(frame_file),
    }
    # The payload has the validator's two-field shape, but this isolated scene
    # has no production input packet or matching Skill camera calibration.
    record = {"status": "not_emitted", "reason": None, "payload": None,
              "diagnostics": diagnostics, "skill_compatible": False,
              "integration_status": "not_connected"}

    if not isinstance(projection, dict) or projection.get("error"):
        record["reason"] = "projection_failed"
        return record
    pixel = projection.get("pixel")
    if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
        record["reason"] = "projection_missing_pixel"
        return record
    if not all(_finite_number(value) for value in pixel):
        record["reason"] = "projection_invalid_pixel"
        return record
    if not _finite_number(projection.get("depth_m")) or projection["depth_m"] <= 0:
        record["reason"] = "projection_invalid_depth"
        return record
    u, v = pixel
    if (not _finite_number(width) or not _finite_number(height)
            or width <= 0 or height <= 0):
        record["reason"] = "projection_invalid_frame"
        return record
    if projection.get("in_frame") is not True or not (0 <= u < width and 0 <= v < height):
        record["reason"] = "projection_out_of_frame"
        return record

    verdict = anchor_ray.get("verdict") if isinstance(anchor_ray, dict) else None
    if verdict != "visible":
        record["reason"] = ("anchor_occluded" if verdict == "blocked"
                            else "anchor_visibility_unknown")
        return record
    if not isinstance(frame_file, str) or not frame_file:
        record["reason"] = "frame_unavailable"
        return record
    if not _finite_number(simulation_time) or simulation_time < 0:
        record["reason"] = "simulation_time_unavailable"
        return record

    record.update(status="emitted", reason=None,
                  payload={"time": float(simulation_time), "feet": [float(u), float(v)]})
    return record
