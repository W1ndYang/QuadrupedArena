"""Pure, conservative Oracle v1 state abstraction for the existing match.

This module never steps the match.  In particular, limited inputs contain no
reference to Match, Referee, HumanPolicy, or synthetic diagnostic records.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from numbers import Real
from typing import Any

import numpy as np

from adaptive_motion import visible_observation
from semantic_skills import SemanticSkills
from tactical_logic import block, wrap
from trial_motion import Planner, clearance
from trial_rules import covers_at


SCHEMA_VERSION = "oracle_state_v1_1"
# Existing VisualMemory.update assigns this radius to a fresh rgb_foot_pixel sighting.
RGB_FOOT_PIXEL_BASE_UNCERTAINTY = 0.6
VIEWS = ("full", "limited")
SKILLS = ("MOVE_TO", "CAPTURE_POINT", "TAKE_COVER", "AIM_AT_TRACK", "OBSERVE")
DIRECTIONS = ("scan", "threat", "north", "south", "east", "west")
FINISH_REASONS = ("time_limit", "quadruped_eliminated", "all_targets_invalid")


def _number(value: Any, name: str, low: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if low is not None and value < low:
        raise ValueError(f"{name} must be >= {low}")
    return value


def _xy(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a pair")
    return (_number(value[0], name), _number(value[1], name))


def _hp(value: Any, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer HP in 0..{maximum}")
    return value


def scene_profile_id(cfg: dict) -> str:
    """Conservative fingerprint: even presentation-only config changes split IDs."""
    try:
        data = json.dumps(cfg, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Config must be finite, JSON-serializable data") from exc
    return "trial_rotating_sha256_" + hashlib.sha256(data.encode()).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class Region:
    region_id: str
    role: str
    semantic_name: str
    geometry_kind: str
    center: tuple[float, float] | None = None
    radius: float | None = None
    size: tuple[float, float] | None = None
    map_phase_valid: str = "both"
    # A phase-specific None means unverified; () means verified no adjacency.
    adjacent_pre_collapse: tuple[str, ...] | None = None
    adjacent_post_collapse: tuple[str, ...] | None = None
    related_target: str | None = None
    related_cover: str | None = None


@dataclass(frozen=True, slots=True)
class RegionCatalog:
    profile_id: str
    regions: tuple[Region, ...]
    cover_band_width: float
    planners: tuple[Planner, Planner] = field(repr=False, compare=False)
    coverage: str = "planner_free_space"

    @classmethod
    def from_config(cls, cfg: dict) -> RegionCatalog:
        points = cfg["arena"]["task_points"]
        covers = cfg["arena"]["covers"]
        band = _number(cfg["planner_clearance"], "planner clearance", 0) + _number(cfg["planner_grid"], "planner grid", 0)
        if band <= 0:
            raise ValueError("Cover band must be positive")
        regions = tuple(Region("target_" + p["name"], "target_zone", p["name"] + "_inner",
                               "target_circle", _xy(p["center"], "target center"),
                               _number(p["radius"], "radius", 0), related_target=p["name"])
                        for p in points) + tuple(
            Region("cover_" + c["name"], "cover_zone", c["name"] + "_planner_band",
                   "free_cover_band", _xy(c["center"], "cover center"),
                   size=_xy(c["size"][:2], "cover size"), related_cover=c["name"])
            for c in covers) + (Region("open_traversable", "open_zone", "planner_free_remainder",
                                      "planner_free_remainder"),)
        ids = [r.region_id for r in regions]
        if len(set(ids)) != len(ids) or any(r.radius <= 0 for r in regions if r.role == "target_zone"):
            raise ValueError("Invalid or duplicate regions")
        targets = [r for r in regions if r.role == "target_zone"]
        for i, a in enumerate(targets):
            for b in targets[i + 1:]:
                if math.dist(a.center, b.center) <= a.radius + b.radius:
                    raise ValueError("Overlapping target regions")
        pre = Planner(cfg, covers_at(cfg, 0))
        post = Planner(cfg, covers_at(cfg, cfg["collapse"]["time"]))
        catalog = cls(scene_profile_id(cfg), regions, band, (pre, post))
        adjacency = []
        for phase, planner in (("pre_collapse", pre), ("post_collapse", post)):
            if not cls._free_graph_connected(planner):
                raise ValueError("Planner free graph is disconnected; one open remainder would be misleading")
            if not all(planner.free(np.asarray(p["center"], float)) for p in points):
                raise ValueError("Target center is outside Planner free space")
            if not all(planner.free(np.asarray(cfg[k], float)) for k in ("start_a", "start_b")):
                raise ValueError("Formal spawn is outside Planner free space")
            adjacency.append(catalog._graph_adjacency(phase, planner))
        if any(set(item) != set(ids) for item in adjacency):
            raise ValueError("Region has no Planner node in one map phase")
        enriched = tuple(replace(r, adjacent_pre_collapse=tuple(sorted(adjacency[0][r.region_id])),
                                 adjacent_post_collapse=tuple(sorted(adjacency[1][r.region_id])))
                         for r in regions)
        return replace(catalog, regions=enriched)

    @staticmethod
    def _free_graph_connected(planner: Planner) -> bool:
        nodes = planner.free_nodes
        if not nodes:
            return False
        seen = {min(nodes)}
        pending = list(seen)
        while pending:
            x, y = pending.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                neighbor = (x + dx, y + dy)
                if neighbor not in nodes or neighbor in seen:
                    continue
                if dx and dy and ((x + dx, y) not in nodes or (x, y + dy) not in nodes):
                    continue
                seen.add(neighbor)
                pending.append(neighbor)
        return len(seen) == len(nodes)

    def _graph_adjacency(self, phase: str, planner: Planner) -> dict[str, set[str]]:
        labels = {node: self.locate((node[0] * planner.grid, node[1] * planner.grid), phase)
                  for node in planner.free_nodes}
        adjacency = {r.region_id: set() for r in self.regions}
        for node, a in labels.items():
            x, y = node
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                other = (x + dx, y + dy)
                if other not in labels or (dx and dy and ((x + dx, y) not in labels or (x, y + dy) not in labels)):
                    continue
                b = labels[other]
                if a == b:
                    continue
                p = np.asarray(node, float) * planner.grid
                q = np.asarray(other, float) * planner.grid
                if planner.segment(p, q):
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        return adjacency

    def locate(self, xy: tuple[float, float], phase: str) -> str | None:
        if phase not in ("pre_collapse", "post_collapse"):
            raise ValueError("Invalid map phase")
        xy = _xy(xy.tolist() if isinstance(xy, np.ndarray) else xy, "region point")
        for region in self.regions:
            if region.role == "target_zone" and math.dist(xy, region.center) <= region.radius + 1e-10:
                return region.region_id
        planner = self.planners[0 if phase == "pre_collapse" else 1]
        if not planner.free(np.asarray(xy, float)):
            return None
        nearby = sorted((clearance(xy, cover), cover["name"]) for cover in planner.covers)
        if nearby and nearby[0][0] <= self.cover_band_width + 1e-10:
            return "cover_" + nearby[0][1]
        return "open_traversable"

    def locate_uncertain(self, xy: tuple[float, float], radius: float, phase: str) -> str | None:
        """Certify one label for the entire estimate disk, else return unknown."""
        xy = _xy(xy, "uncertain region point")
        radius = _number(radius, "uncertainty radius", 0)
        if radius == 0:
            return self.locate(xy, phase)
        if phase not in ("pre_collapse", "post_collapse"):
            raise ValueError("Invalid map phase")
        targets = [region for region in self.regions if region.role == "target_zone"]
        for region in targets:
            distance = math.dist(xy, region.center)
            if distance + radius <= region.radius:
                return region.region_id
            if distance - radius <= region.radius + 1e-10:
                return None
        planner = self.planners[0 if phase == "pre_collapse" else 1]
        if abs(xy[0]) + radius > 5.55 or abs(xy[1]) + radius > 3.55:
            return None
        # Rectangle clearance is 1-Lipschitz: a radius-r disk can lower any
        # clearance by at most r. This also covers the reserved collapse sweep.
        if any(clearance(xy, obstacle) - radius < planner.margin
               for obstacle in planner.obstacles):
            return None
        nearby = sorted((clearance(xy, cover), cover["name"]) for cover in planner.covers)
        if not nearby or nearby[0][0] - radius > self.cover_band_width:
            return "open_traversable"
        nearest_distance, nearest_name = nearby[0]
        if nearest_distance + radius <= self.cover_band_width and (
                len(nearby) == 1 or nearby[1][0] - nearest_distance > 2 * radius):
            return "cover_" + nearest_name
        return None

    def adjacent(self, region_id: str, phase: str) -> tuple[str, ...] | None:
        if phase not in ("pre_collapse", "post_collapse"):
            raise ValueError("Invalid map phase")
        region = next((r for r in self.regions if r.region_id == region_id), None)
        if region is None:
            raise ValueError("Unknown region")
        return region.adjacent_pre_collapse if phase == "pre_collapse" else region.adjacent_post_collapse

    def cover_relation(self, xy: tuple[float, float], phase: str) -> str:
        if self.locate(xy, phase) is None:
            return "unknown"
        planner = self.planners[0 if phase == "pre_collapse" else 1]
        nearest = min((clearance(xy, cover) for cover in planner.covers), default=float("inf"))
        if nearest <= self.cover_band_width + 1e-10:
            return "in_cover_zone"
        if nearest <= self.cover_band_width + planner.grid + 1e-10:
            return "adjacent_to_cover"
        return "exposed_or_other"


@dataclass(frozen=True, slots=True)
class MoverSnapshot:
    xy: tuple[float, float]
    yaw: float
    signed_v: float
    route: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class ReferenceSnapshot:
    """R1 read-only boundary copy. RNG and hidden threshold are deliberately absent."""

    profile_id: str
    tick: int
    t: float
    finished: str | None
    collapsed: bool
    a: MoverSnapshot
    b: MoverSnapshot
    hp: tuple[int, int]
    aim: tuple[float, float]
    last_valid: tuple[bool, bool]
    cooldown_until: tuple[float, float]
    human_state: str
    human_visible: bool
    human_memory: tuple[float, float] | None
    human_seen_at: float | None
    patrol_index: int
    scan_until: float | None
    last_plan: float
    active_target: int
    generation: int
    triggered_at: float | None
    expires_at: float | None
    inside: tuple[bool, ...]
    dwell: tuple[float, ...]
    score: float
    active_low_level: str | None
    active_attack_enabled: bool
    skill_status: str
    skill_intent: str | None
    skill_generation: int | None
    skill_last_hp: int | None
    visual_track: tuple[float, float] | None
    visual_seen_at: float | None
    visual_source: str | None
    camera_pan: float


def capture_reference(match: Any, engine: Any, rig: Any) -> ReferenceSnapshot:
    """Read existing live objects at a boundary; no timeline or RNG access."""
    ref, human = match.ref, match.human
    track = engine.memory.track

    def mover(m: Any) -> MoverSnapshot:
        return MoverSnapshot(_xy(m.xy.tolist(), "mover.xy"), _number(m.yaw, "yaw"),
                             _number(m.v, "v"), tuple(_xy(p.tolist(), "route point") for p in m.route))

    return ReferenceSnapshot(
        scene_profile_id(match.cfg), match.tick, _number(match.t, "t", 0), ref.finished,
        bool(match.collapsed), mover(match.a), mover(match.b), tuple(ref.hp),
        tuple(ref.aim), tuple(ref.last_valid), tuple(ref.cooldown_until), human.state,
        bool(human.visible), None if human.memory is None else _xy(human.memory.tolist(), "human.memory"),
        human.seen_at, human.patrol_index, human.scan_until, human.last_plan,
        ref.active_target, ref.generation, ref.triggered_at, ref.expires_at,
        tuple(ref.inside), tuple(ref.dwell), ref.score,
        None if match.active is None else match.active["skill"],
        True if match.active is None else bool(match.active.get("attack", True)), match.skill_status,
        engine.intent, engine.generation, engine.hp,
        None if "xy" not in track else _xy(track["xy"], "visual track"),
        track.get("seen_at"), track.get("source"), _number(rig.pan, "camera pan"),
    )


@dataclass(frozen=True, slots=True)
class FrameSighting:
    time: float
    feet: tuple[float, float] | None
    camera_world_yaw_degrees: float
    camera_eye: tuple[float, float, float]
    file: str


@dataclass(frozen=True, slots=True)
class LimitedObservationInput:
    """Whitelisted primitives only. A Match/Referee cannot fit this type."""

    profile_id: str
    observation_id: int
    t: float
    own_xy: tuple[float, float]
    own_yaw: float
    own_speed: float
    own_hp: int
    own_cooldown: float
    own_skill_status: str
    score: float
    active_target: int
    generation: int
    target_remaining: float | None
    dwell: tuple[float, ...]
    finished: str | None
    memory_xy: tuple[float, float] | None
    memory_seen_at: float | None
    memory_source: str | None
    memory_expired: bool
    memory_uncertainty_radius: float | None
    skill_status: str
    frames: tuple[FrameSighting, ...]
    previous_action_id: str | None = None


def limited_from_public(packet: dict, sightings: tuple[dict | None, ...], cfg: dict,
                        previous_action_id: str | None = None) -> LimitedObservationInput:
    """Explicit whitelist projection. Never accepts a synthetic diagnostic record."""
    if not isinstance(packet, dict) or not isinstance(sightings, tuple):
        raise ValueError("Expected public packet and tuple of compliant sightings")
    own, public, skill = packet["self_state"], packet["public_state"], packet["skills"]
    frames = packet["frames"]
    if len(frames) != len(sightings):
        raise ValueError("One sighting/null is required per frame")
    result = []
    for frame, sighting in zip(frames, sightings):
        t = _number(frame["simulation_seconds"], "frame time", 0)
        if sighting is not None:
            if not isinstance(sighting, dict) or set(sighting) != {"time", "feet"}:
                raise ValueError("Only {time, feet} sighting is permitted")
            if _number(sighting["time"], "sighting time") != t:
                raise ValueError("Sighting/frame time mismatch")
            feet = _xy(sighting["feet"], "feet")
            if not (0 <= feet[0] < 768 and 0 <= feet[1] < 432):
                raise ValueError("Feet outside RGB frame")
        else:
            feet = None
        eye = frame["camera_eye"]
        if not isinstance(eye, (list, tuple)) or len(eye) != 3:
            raise ValueError("camera_eye must be 3D")
        result.append(FrameSighting(t, feet, _number(frame["camera_world_yaw_degrees"], "camera yaw"),
                                    tuple(_number(v, "camera eye") for v in eye), str(frame["file"])))
    memory = skill["visual_memory"]
    expired = memory.get("status") == "expired"
    if expired:
        if set(memory) != {"status", "last_seen_at", "current_position"} or memory["current_position"] != "unknown":
            raise ValueError("Invalid expired visual memory")
        mxy, mseen, msource, mradius = None, _number(memory["last_seen_at"], "last_seen_at", 0), None, None
    else:
        allowed = {"xy", "seen_at", "source", "base_uncertainty", "age", "uncertainty_radius", "aim_usable", "frame", "feet"}
        if set(memory) - allowed or not {"xy", "seen_at", "source"} <= set(memory):
            raise ValueError("Invalid public visual memory")
        mxy, mseen, msource = _xy(memory["xy"], "memory.xy"), _number(memory["seen_at"], "seen_at", 0), memory["source"]
        mradius = (_number(memory["uncertainty_radius"], "memory.uncertainty_radius", 0)
                   if "uncertainty_radius" in memory else None)
    status = skill["skill_status"]
    if not isinstance(status, dict) or not isinstance(status.get("state"), str):
        raise ValueError("Invalid public Skill status")
    return LimitedObservationInput(
        scene_profile_id(cfg), packet["observation_id"], _number(packet["simulation_seconds"], "time", 0),
        _xy(own["xy"], "own.xy"), _number(own["yaw"], "own.yaw"),
        _number(own["speed"], "own.speed", 0), own["hp"], _number(own["cooldown"], "own.cooldown", 0),
        own["skill_status"], _number(public["score"], "score", 0), public["active_target"],
        public["target_generation"], public["target_remaining"], tuple(public["dwell"]), public["finished"],
        mxy, mseen, msource, expired, mradius, status["state"], tuple(result), previous_action_id,
    )


@dataclass(frozen=True, slots=True)
class EncodedState:
    version: str
    profile_id: str
    view: str
    status: str  # valid | terminal | blocked
    fields: tuple[tuple[str, str | int], ...]
    state_id: str | None
    reason: str | None = None


def _result(profile: str, view: str, status: str, fields: dict[str, str | int],
            reason: str | None = None, version: str = SCHEMA_VERSION) -> EncodedState:
    if view not in VIEWS or not isinstance(version, str) or not version:
        raise ValueError("Invalid state metadata")
    ordered = tuple(sorted(fields.items()))
    if status == "blocked":
        return EncodedState(version, profile, view, status, ordered, None, reason)
    canonical = json.dumps([version, profile, view, status, ordered], separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return EncodedState(version, profile, view, status, ordered, f"{version}:{view}:{digest}", reason)


def _boundary(t: float, duration: float, terminal: bool) -> int:
    t = _number(t, "t", 0)
    if t > duration + 1e-8:
        raise ValueError("Time beyond duration")
    if not terminal and (t >= duration or not math.isclose(t / 2, round(t / 2), abs_tol=1e-8)):
        raise ValueError("Nonterminal state must be on a 2-second decision boundary")
    return min(19, int(t // 2))


def _cooldown(remaining: float) -> str:
    remaining = _number(remaining, "cooldown remaining")
    return "ready" if remaining <= 1e-9 else "recovers_within_window" if remaining <= 2 else "unavailable_next_window"


def _deadline(remaining: float | None) -> str:
    if remaining is None:
        return "not_triggered"
    remaining = _number(remaining, "target remaining", 0)
    return "lt_2s" if remaining < 2 else "2_to_4s" if remaining <= 4 else "gt_4s"


def _memory_age(age: float | None) -> str:
    if age is None or age > 4:
        return "gt_4_or_none"
    age = _number(age, "visual memory age", 0)
    for end, label in ((.5, "0_to_0.5"), (1, "0.5_to_1"), (1.5, "1_to_1.5"), (2, "1.5_to_2")):
        if age <= end:
            return label
    return "2_to_4"


def _visual_bin(source: str | None, age: float | None) -> str:
    if source is None and age is None:
        return "none_or_initial_prior"
    if age is None or age > 4:
        return "expired"
    if source == "known_initial":
        return "none_or_initial_prior"
    if source != "rgb_foot_pixel":
        raise ValueError("Invalid visual source")
    return "fresh_aim_usable" if age <= 2 else "stale_retained"


def _dwell(inside: bool, seconds: float) -> str:
    seconds = _number(seconds, "dwell", 0)
    return "outside" if not inside else "inside_under_1s" if seconds < 1 else "scoring_eligible"


def _relation(distance: float, radius: float, eta: float | None) -> str:
    if distance < radius - .04:
        return "inside_capture"
    if distance <= radius + 1e-10:
        return "capture_edge"
    if eta is None:
        return "unreachable"
    return "near" if eta <= 2 else "far"


def _eta_bin(inside: bool, eta: float | None) -> str:
    if inside:
        return "inside"
    if eta is None:
        return "no_route"
    return "le_1_window" if eta <= 2 else "1_to_2_windows" if eta <= 4 else "gt_2_windows"


def _target_geometry(cfg: dict, t: float, index: int, xy: tuple[float, float], yaw: float,
                     speed: float) -> tuple[float, float, float | None]:
    """Fixed reference route to center, with no risk mode, visual track, or candidate Skill."""
    points = cfg["arena"]["task_points"]
    if type(index) is not int or not 0 <= index < len(points):
        raise ValueError("Invalid active target")
    point = points[index]
    distance = math.dist(xy, point["center"])
    planner = Planner(cfg, covers_at(cfg, t))
    route = planner.route(np.array(xy), np.array(point["center"], float))
    if route is None:
        eta = None
    else:
        own = {"xy": xy, "yaw": yaw, "speed": abs(speed)}
        eta = SemanticSkills(cfg).eta(own, route, False)
    return distance, point["radius"], eta


def _heading(yaw: float, start: tuple[float, float], target: tuple[float, float], name: str) -> str:
    angle = abs(wrap(math.atan2(target[1] - start[1], target[0] - start[0]) - yaw))
    return "toward" if angle <= math.pi / 6 else "lateral" if angle <= math.pi / 2 else "away"


def _common(cfg: dict, catalog: RegionCatalog, profile: str, t: float, finished: str | None,
            own_xy: tuple[float, float], own_yaw: float, own_speed: float, own_hp: int,
            cooldown: float, active: int, remaining: float | None, dwell: tuple[float, ...],
            mem_xy: tuple[float, float] | None, mem_seen: float | None, mem_source: str | None,
            own_skill_status: str, signed_speed_known: bool = True) -> tuple[dict[str, str | int], list[str]]:
    if catalog.profile_id != profile or profile != scene_profile_id(cfg):
        raise ValueError("Scene profile/catalog mismatch")
    step = _boundary(t, cfg["duration"], finished is not None)
    _hp(own_hp, "A HP", cfg["initial_hp"])
    if type(active) is not int or not 0 <= active < len(cfg["arena"]["task_points"]):
        raise ValueError("Invalid target")
    if len(dwell) != len(cfg["arena"]["task_points"]):
        raise ValueError("Invalid dwell array")
    for value in dwell:
        _number(value, "dwell", 0)
    if mem_source not in (None, "known_initial", "rgb_foot_pixel"):
        raise ValueError("Invalid visual memory source")
    if mem_seen is not None and (mem_seen > t or mem_seen < 0):
        raise ValueError("Invalid visual memory timestamp")
    phase = "post_collapse" if t >= cfg["collapse"]["time"] else "pre_collapse"
    rid = catalog.locate(own_xy, phase)
    distance, radius, eta = _target_geometry(cfg, t, active, own_xy, own_yaw, own_speed)
    age = None if mem_seen is None else t - mem_seen
    mem_rid = None if mem_xy is None or age is None or age > 4 else catalog.locate(mem_xy, phase)
    fields: dict[str, str | int] = {
        "decision_step": step, "map_phase": phase, "A.region_id": rid or "unknown",
        "A.active_target_relation": _relation(distance, radius, eta),
        "A.target_eta_bin": _eta_bin(distance <= radius + 1e-10, eta),
        "A.heading_to_target": _heading(own_yaw, own_xy, tuple(cfg["arena"]["task_points"][active]["center"]), "A"),
        "A.motion_bin": "safety_stopped" if own_skill_status == "safety_stop" else
                        "stopped" if own_speed == 0 else "moving" if not signed_speed_known else
                        "moving_forward" if own_speed > 0 else "moving_reverse",
        "A.cover_relation": catalog.cover_relation(own_xy, phase),
        "A.hp": own_hp, "A.cooldown_bin": _cooldown(cooldown),
        "target.active": cfg["arena"]["task_points"][active]["name"],
        "target.deadline_phase": _deadline(remaining),
        "target.dwell_bin": _dwell(distance <= radius + 1e-10, dwell[active]),
        "A.visual_memory_bin": _visual_bin(mem_source, age),
        "A.visual_track_region": mem_rid or "unknown",
        "A.visual_memory_age_bin": _memory_age(age),
        "A.visual_memory_source": mem_source or "none",
    }
    missing = [] if rid else ["A.region_id:outside_verified_region_catalog"]
    return fields, missing


def _aim_bin(age: float, valid: bool, tau: float) -> str:
    _number(age, "aim age", 0)
    tau = _number(tau, "aim tau", 0)
    if tau == 0:
        raise ValueError("Aim tau must be positive")
    if not valid:
        return "invalid"
    if age == 0:
        return "newly_valid"
    if age < tau / 2:
        return "under_half_tau"
    if age < tau:
        return "half_to_tau"
    return "tau_or_more"


def _attack_geometry(cfg: dict, shooter: MoverSnapshot, target: MoverSnapshot,
                     shooter_index: int, t: float, cooldown_until: float, enabled: bool) -> str:
    if not enabled:
        return "disabled"
    if t < cooldown_until - 1e-9:
        return "cooldown"
    d = np.array(target.xy) - shooter.xy
    distance = float(np.linalg.norm(d))
    if distance > cfg["range_m"] or distance < 1e-8:
        return "invalid_range"
    if abs(wrap(math.atan2(d[1], d[0]) - shooter.yaw)) > math.radians(cfg["aim_half_angle_degrees"]) + 1e-10:
        return "invalid_angle"
    heights = cfg["representative_heights"]
    return "blocked_los" if block((*shooter.xy, heights[shooter_index]),
                                  (*target.xy, heights[1 - shooter_index]), covers_at(cfg, t)) else "valid_now"


def encode_full(snapshot: ReferenceSnapshot, cfg: dict, catalog: RegionCatalog,
                version: str = SCHEMA_VERSION) -> EncodedState:
    if type(snapshot) is not ReferenceSnapshot:
        raise TypeError("Full encoder requires ReferenceSnapshot")
    t = _number(snapshot.t, "t", 0)
    if catalog.profile_id != snapshot.profile_id or snapshot.profile_id != scene_profile_id(cfg):
        raise ValueError("Scene profile/catalog mismatch")
    if type(snapshot.tick) is not int or not math.isclose(t, snapshot.tick * cfg["dt"], abs_tol=1e-8):
        raise ValueError("Tick/time mismatch")
    if snapshot.finished is not None:
        if snapshot.finished not in FINISH_REASONS:
            raise ValueError("Unknown terminal reason")
        _boundary(t, cfg["duration"], True)
        return _result(snapshot.profile_id, "full", "terminal", {"reason": snapshot.finished}, version=version)
    active = snapshot.active_target
    remaining = None if snapshot.expires_at is None else max(0., snapshot.expires_at - t)
    fields, missing = _common(cfg, catalog, snapshot.profile_id, t, None, snapshot.a.xy,
                              _number(snapshot.a.yaw, "A yaw"), _number(snapshot.a.signed_v, "A speed"),
                              snapshot.hp[0], snapshot.cooldown_until[0] - t, active, remaining,
                              snapshot.dwell, snapshot.visual_track, snapshot.visual_seen_at,
                              snapshot.visual_source, snapshot.skill_status)
    _hp(snapshot.hp[1], "B HP", cfg["initial_hp"])
    phase = fields["map_phase"]
    b_rid = catalog.locate(snapshot.b.xy, phase)
    if b_rid is None:
        missing.append("B.region_id:outside_verified_region_catalog")
    b_distance, b_radius, b_eta = _target_geometry(cfg, t, active, snapshot.b.xy, snapshot.b.yaw, snapshot.b.signed_v)
    sensor = visible_observation(cfg, np.array(snapshot.b.xy), snapshot.b.yaw, np.array(snapshot.a.xy), covers_at(cfg, t), t)
    b_mem_age = None if snapshot.human_seen_at is None else t - snapshot.human_seen_at
    if b_mem_age is not None and b_mem_age < 0:
        raise ValueError("B memory timestamp in future")
    b_mem_region = None if snapshot.human_memory is None else catalog.locate(snapshot.human_memory, phase)
    if snapshot.human_state not in ("PATROL", "PATROL_SCAN", "VISIBLE_AIM", "VISIBLE_COOLDOWN",
                                    "SEARCH_LAST_SEEN", "SEARCH_SCAN", "ELIMINATED"):
        raise ValueError("Unknown HumanPolicy state")
    policy_mode = {"PATROL": "patrol", "PATROL_SCAN": "scan", "VISIBLE_AIM": "aiming_or_engaged",
                   "VISIBLE_COOLDOWN": "aiming_or_engaged", "SEARCH_LAST_SEEN": "tracking_last_seen",
                   "SEARCH_SCAN": "tracking_last_seen", "ELIMINATED": "disabled"}[snapshot.human_state]
    fields.update({
        "A.aim_age_bin": _aim_bin(snapshot.aim[0], snapshot.last_valid[0], cfg["tau"][0]),
        "B.region_id": b_rid or "unknown",
        "B.active_target_relation": _relation(b_distance, b_radius, b_eta),
        "B.cover_relation": catalog.cover_relation(snapshot.b.xy, phase),
        "B.heading_relation": {"toward": "toward_A", "lateral": "lateral_to_A",
                               "away": "away_from_A"}[_heading(snapshot.b.yaw, snapshot.b.xy, snapshot.a.xy, "B")],
        "B.motion_bin": "stopped" if snapshot.b.signed_v == 0 else
                        "patrol_move" if policy_mode == "patrol" else
                        "search_move" if policy_mode == "tracking_last_seen" else "other_move",
        "B.policy_mode": policy_mode, "B.hp": snapshot.hp[1],
        "B.cooldown_bin": _cooldown(snapshot.cooldown_until[1] - t),
        "B.aim_age_bin": _aim_bin(snapshot.aim[1], snapshot.last_valid[1], cfg["tau"][1]),
        "B.memory_presence": "none" if snapshot.human_memory is None else
                             "expired" if b_mem_age is None or b_mem_age > cfg["policy"]["human_memory_seconds"] else "active",
        "B.last_seen_A_region": b_mem_region or "unknown",
        "B.memory_age_bin": "expired_or_none" if b_mem_age is None or b_mem_age > 4 else
                            "0_to_2s" if b_mem_age <= 2 else "2_to_4s",
        "B.patrol_phase": snapshot.patrol_index,
        "B.scan_phase_bin": "not_scanning" if snapshot.scan_until is None or snapshot.scan_until <= t else
                            "scan_near_end" if snapshot.scan_until - t <= .5 else "scanning",
        "B.replan_age_bin": "due_or_overdue" if t - snapshot.last_plan >= 1 else "recently_planned",
        "target.generation_relation": "unbound" if snapshot.skill_generation is None else
                                      "bound_current" if snapshot.skill_generation == snapshot.generation else "bound_old",
        "pair_range_bin": "within_8m" if math.dist(snapshot.a.xy, snapshot.b.xy) <= cfg["range_m"] else "beyond_8m",
        "B_sensor_relation_to_A": "sensor_visible" if sensor is not None else "sensor_not_visible",
        "B_attack_geometry": _attack_geometry(cfg, snapshot.b, snapshot.a, 1, t,
                                               snapshot.cooldown_until[1], snapshot.human_visible and min(snapshot.hp) > 0),
        "A_attack_geometry_truth": _attack_geometry(cfg, snapshot.a, snapshot.b, 0, t,
                                                    snapshot.cooldown_until[0], snapshot.active_attack_enabled and min(snapshot.hp) > 0),
        "relative_bearing_class": "front_cone" if abs(wrap(math.atan2(snapshot.b.xy[1]-snapshot.a.xy[1],
                                            snapshot.b.xy[0]-snapshot.a.xy[0])-snapshot.a.yaw)) <= math.pi/6 else
                                  "side_cone" if abs(wrap(math.atan2(snapshot.b.xy[1]-snapshot.a.xy[1],
                                            snapshot.b.xy[0]-snapshot.a.xy[0])-snapshot.a.yaw)) <= math.pi/3 else "rear_or_unknown",
    })
    if type(snapshot.patrol_index) is not int or not 0 <= snapshot.patrol_index < len(cfg["policy"]["human_patrol"]):
        raise ValueError("Invalid patrol index")
    if missing:
        return _result(snapshot.profile_id, "full", "blocked", fields, ";".join(missing), version)
    return _result(snapshot.profile_id, "full", "valid", fields, version=version)


def _skill_status(status: str) -> str:
    if status in ("RUNNING", "HOLDING_POINT", "AT_DESTINATION", "IDLE"):
        return "running"
    if status in ("TARGET_EXPIRED", "TARGET_INACTIVE"):
        return "target_expired"
    if status == "NO_ROUTE":
        return "no_route"
    if status in ("UNSAFE_ROUTE", "NO_SAFE_COVER"):
        return "unsafe_route"
    if status.startswith("EMERGENCY_"):
        return "emergency_cover"
    if status in ("BLOCKED_SAFETY_STOP", "TARGET_STALE"):
        return "stopped"
    return "other_status"


def encode_limited(obs: LimitedObservationInput, cfg: dict, catalog: RegionCatalog,
                   version: str = SCHEMA_VERSION) -> EncodedState:
    if type(obs) is not LimitedObservationInput:
        raise TypeError("Limited encoder requires LimitedObservationInput, not Match/R1")
    t = _number(obs.t, "time", 0)
    if catalog.profile_id != obs.profile_id or obs.profile_id != scene_profile_id(cfg):
        raise ValueError("Scene profile/catalog mismatch")
    if obs.finished is not None:
        if obs.finished not in FINISH_REASONS:
            raise ValueError("Unknown terminal reason")
        _boundary(t, cfg["duration"], True)
        return _result(obs.profile_id, "limited", "terminal", {"reason": obs.finished}, version=version)
    if type(obs.observation_id) is not int or obs.observation_id != int(t // 2):
        raise ValueError("Observation ID/time mismatch")
    expected = 1 if t == 0 else 4
    if len(obs.frames) != expected:
        raise ValueError("Expected bootstrap frame or four fresh half-second frames")
    times = [f.time for f in obs.frames]
    expected_times = [0.] if t == 0 else [t - 1.5, t - 1, t - .5, t]
    if any(not math.isclose(a, b, abs_tol=1e-8) for a, b in zip(times, expected_times)):
        raise ValueError("Frame cadence mismatch")
    if obs.memory_expired and obs.memory_xy is not None:
        raise ValueError("Expired memory cannot expose xy")
    if obs.memory_seen_at is not None and obs.memory_seen_at > t:
        raise ValueError("Future visual memory")
    fields, missing = _common(cfg, catalog, obs.profile_id, t, None, _xy(obs.own_xy, "own.xy"),
                              _number(obs.own_yaw, "own.yaw"), _number(obs.own_speed, "own.speed", 0),
                              obs.own_hp, obs.own_cooldown, obs.active_target,
                              obs.target_remaining, obs.dwell, obs.memory_xy,
                              obs.memory_seen_at, obs.memory_source, obs.own_skill_status,
                              signed_speed_known=False)
    phase = fields["map_phase"]
    if (obs.memory_xy is not None and obs.memory_seen_at is not None
            and t - obs.memory_seen_at <= 4 and obs.memory_uncertainty_radius is not None):
        fields["A.visual_track_region"] = (catalog.locate_uncertain(
            obs.memory_xy, obs.memory_uncertainty_radius, phase) or "unknown")
    else:
        fields["A.visual_track_region"] = "unknown"
    mask = "".join("1" if f.feet is not None else "0" for f in obs.frames)
    valid = [f for f in obs.frames if f.feet is not None]
    from semantic_skills import ground_point
    regions = []
    for f in valid:
        frame = {"camera_world_yaw_degrees": f.camera_world_yaw_degrees, "camera_eye": f.camera_eye}
        try:
            xy = _xy(ground_point(frame, list(f.feet)), "projected foot")
        except ValueError:
            # A syntactically valid foot ray is not guaranteed to hit valid ground.
            regions.append(None)
        else:
            regions.append(catalog.locate_uncertain(xy, RGB_FOOT_PIXEL_BASE_UNCERTAINTY, phase))
    latest = valid[-1] if valid else None
    latest_region = regions[-1] if regions else None
    latest_age = None if latest is None else t - latest.time
    trend = "unknown"
    if len(regions) >= 2 and regions[-1] is not None and regions[-2] is not None:
        trend = "roughly_stable" if regions[-1] == regions[-2] else "moving_between_regions"
    if obs.previous_action_id is not None and obs.previous_action_id not in {a.template_id for a in action_catalog(cfg)}:
        raise ValueError("Previous action ID must be a valid action template")
    if t > 0 and obs.previous_action_id is None:
        missing.append("own_previous_action:missing_history")
    status = _skill_status(obs.skill_status)
    fields.update({
        "window.latest_sighting_region": latest_region or "unknown",
        "window.latest_sighting_age_bin": _memory_age(latest_age),
        "window.valid_sighting_mask": mask,
        "window.inter_sighting_trend": trend,
        "window.no_sighting_count": len(obs.frames) - len(valid),
        "own_skill_status": status,
        "own_previous_action": obs.previous_action_id or "none",
    })
    # Window outcome cannot be reconstructed from the current Skill status.
    # Keep it in future transition audit labels, not in this state ID.
    if missing:
        return _result(obs.profile_id, "limited", "blocked", fields, ";".join(missing), version)
    return _result(obs.profile_id, "limited", "valid", fields, version=version)


@dataclass(frozen=True, slots=True)
class ActionTemplate:
    skill: str
    params: tuple[tuple[str, str], ...]

    @property
    def template_id(self) -> str:
        return self.skill + ":" + json.dumps(self.params, separators=(",", ":"), ensure_ascii=False)

    def request(self, observation_id: int) -> dict:
        if type(observation_id) is not int or observation_id < 0:
            raise ValueError("Invalid observation_id")
        return {"observation_id": observation_id, "skill": self.skill, "params": dict(self.params)}


def action_template(skill: str, params: dict, cfg: dict) -> ActionTemplate:
    """Validate finite high-level templates only; sighting is observation data."""
    if skill not in SKILLS or not isinstance(params, dict):
        raise ValueError("Only five formal Semantic Skills are supported")
    if "sighting" in params:
        raise ValueError("A sighting is observation data, never an action template")
    allowed = {"risk"}
    if skill in ("MOVE_TO", "CAPTURE_POINT"):
        allowed.add("target")
    if skill == "TAKE_COVER":
        allowed.add("cover")
    if skill == "OBSERVE":
        allowed.add("direction")
    if set(params) - allowed:
        raise ValueError("Unknown action parameter")
    risk = params.get("risk", "safe")
    if risk not in ("safe", "fast"):
        raise ValueError("Invalid risk")
    result: dict[str, str] = {}
    if skill in ("MOVE_TO", "CAPTURE_POINT"):
        target = params.get("target", "active")
        if target not in ("active",) + tuple(p["name"] for p in cfg["arena"]["task_points"]):
            raise ValueError("Invalid target")
        if skill == "CAPTURE_POINT" and target != "active":
            raise ValueError("CAPTURE_POINT template only targets the active point")
        result["target"] = target
        result["risk"] = risk
    elif skill == "TAKE_COVER":
        if "cover" in params:
            if params["cover"] not in tuple(c["name"] for c in cfg["arena"]["covers"]):
                raise ValueError("Invalid cover")
            result["cover"] = params["cover"]
        result["risk"] = risk
    elif skill == "OBSERVE":
        direction = params.get("direction", "scan")
        if direction not in DIRECTIONS:
            raise ValueError("Invalid direction")
        result["direction"] = direction
    # AIM_AT_TRACK has no parameter that changes its compiled action.
    return ActionTemplate(skill, tuple(sorted(result.items())))


def action_catalog(cfg: dict) -> tuple[ActionTemplate, ...]:
    candidates = []
    for risk in ("safe", "fast"):
        candidates.append(action_template("CAPTURE_POINT", {"target": "active", "risk": risk}, cfg))
        for target in ("active",) + tuple(p["name"] for p in cfg["arena"]["task_points"]):
            candidates.append(action_template("MOVE_TO", {"target": target, "risk": risk}, cfg))
        candidates.append(action_template("TAKE_COVER", {"risk": risk}, cfg))
        for cover in cfg["arena"]["covers"]:
            candidates.append(action_template("TAKE_COVER", {"cover": cover["name"], "risk": risk}, cfg))
    candidates.append(action_template("AIM_AT_TRACK", {}, cfg))
    for direction in DIRECTIONS:
        candidates.append(action_template("OBSERVE", {"direction": direction}, cfg))
    return tuple(candidates)
