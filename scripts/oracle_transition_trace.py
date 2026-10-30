"""Read-only Oracle transition contract for the existing synthetic Skill smoke.

This module never advances a Match, commits a Skill, or samples a referee RNG.
The trace is internal audit data; only its limited state subobjects have R2-L
permissions. It is not a policy input or an empirical transition model.
"""

from __future__ import annotations

import copy
import math
from collections import Counter


TRACE_VERSION = "oracle_transition_trace_v1"
STATE_KEYS = ("status", "reason", "state_id", "decision_step", "map_phase", "A.region_id")
REF_EVENT_TYPES = ("hit", "eliminated", "finish", "target_invalidated", "target_retired",
                   "target_activated", "collapse", "scoring_started")


def _state_view(boundary: dict, view: str) -> dict:
    source = boundary.get(view) or {}
    result = {key: source.get(key) for key in STATE_KEYS}
    if view == "full":
        result["B.region_id"] = source.get("B.region_id")
    else:
        result["A.visual_track_region"] = source.get("A.visual_track_region")
    return result


def _boundary(boundary: dict) -> dict:
    full = boundary.get("full") or {}
    decision_step = full.get("decision_step")
    # Terminal encoders intentionally contain only a terminal reason; the
    # existing observation index still identifies the closing decision step.
    if decision_step is None and full.get("status") == "terminal":
        decision_step = boundary.get("observation_id")
    return {
        "observation_id": boundary.get("observation_id"),
        "simulation_seconds": boundary.get("simulation_seconds"),
        "decision_step": decision_step,
        "frame_count": boundary.get("frame_count"),
        "map_phase": full.get("map_phase"),
        "full": _state_view(boundary, "full"),
        "limited": _state_view(boundary, "limited"),
        "render_sync_quality": boundary.get("render_sync_quality"),
    }


def referee_snapshot(match) -> dict:
    """Copy only existing referee values; no threshold, RNG, or timeline read."""
    ref = match.ref
    hp = tuple(ref.hp)
    return {
        "simulation_seconds": match.t,
        "score": ref.score,
        "A_hp": hp[0], "B_hp": hp[1],
        "A_alive": hp[0] > 0, "B_alive": hp[1] > 0,
        "active_target": ref.active_target,
        "target_generation": ref.generation,
        "dwell": list(ref.dwell),
        "finished": ref.finished,
        "referee_event_cursor": len(ref.events),
        "match_event_cursor": len(match.events),
    }


def _event_summary(match, start: dict) -> dict:
    referee = Counter(event.get("type") for event in match.ref.events[start["referee_event_cursor"]:]
                      if event.get("type") in REF_EVENT_TYPES)
    match_events = Counter(event.get("type") for event in match.events[start["match_event_cursor"]:]
                           if event.get("type") == "skill_cancelled")
    return {"referee_event_counts": dict(sorted(referee.items())),
            "match_event_counts": dict(sorted(match_events.items()))}


def _action(decision: dict | None, cfg: dict) -> dict:
    raw = None if decision is None else {key: copy.deepcopy(decision[key]) for key in
                                        ("skill", "params", "reason") if key in decision}
    compiled_action = None if decision is None else copy.deepcopy(decision.get("compiled_action"))
    compiled_skill = None if decision is None else copy.deepcopy(decision.get("compiled_skill"))
    summary = None if not isinstance(compiled_action, dict) else {
        key: copy.deepcopy(compiled_action[key]) for key in
        ("skill", "goal", "facing", "attack", "reverse", "speed_limit", "target_generation")
        if key in compiled_action}
    mapped, action_id, normalized, reason = False, None, None, None
    if not isinstance(raw, dict) or not isinstance(raw.get("skill"), str) or not isinstance(raw.get("params"), dict):
        reason = "missing_or_invalid_skill_request"
    else:
        try:
            # Import lazily so the collector can validate an archive without NumPy.
            from oracle_state_v1 import action_catalog, action_template
            template = action_template(raw["skill"], raw["params"], cfg)
            if template not in action_catalog(cfg):
                reason = "not_in_finite_action_catalog"
            else:
                mapped, action_id = True, template.template_id
                normalized = {"skill": template.skill, "params": dict(template.params)}
        except (ValueError, TypeError, KeyError) as exc:
            reason = f"action_template_rejected:{type(exc).__name__}:{exc}"
    return {"raw_skill_request": raw, "compiled_skill": compiled_skill,
            "compiled_action": compiled_action, "compiled_action_summary": summary,
            "oracle_action_id": action_id, "normalized_oracle_action": normalized,
            "action_mapping_status": "mapped" if mapped else "unmapped",
            "action_mapping_reason": reason}


def _alignment(start: dict, end: dict, terminal: bool) -> dict:
    errors = []
    a, b = start.get("observation_id"), end.get("observation_id")
    if type(a) is not int or type(b) is not int or b != a + 1:
        errors.append("observation_id_not_adjacent")
    step_a, step_b = start.get("decision_step"), end.get("decision_step")
    if type(step_a) is not int or type(step_b) is not int or step_b != step_a + 1:
        errors.append("decision_step_not_adjacent")
    t0, t1 = start.get("simulation_seconds"), end.get("simulation_seconds")
    elapsed = None
    if (type(t0) not in (int, float) or type(t1) not in (int, float) or
            not math.isfinite(t0) or not math.isfinite(t1) or t1 <= t0):
        errors.append("simulation_time_not_increasing")
    else:
        elapsed = round(t1 - t0, 8)
        if elapsed > 2. + 1e-8 or (not terminal and not math.isclose(elapsed, 2., abs_tol=1e-8)):
            errors.append("unexpected_decision_interval")
    return {"valid": not errors, "errors": errors, "actual_elapsed_seconds": elapsed}


class TransitionTrace:
    def __init__(self, trajectory_id: str, cfg: dict, synthetic_seed: int):
        from oracle_state_v1 import SCHEMA_VERSION, scene_profile_id
        self.trajectory_id = trajectory_id
        self.cfg = cfg
        self.state_schema_version = SCHEMA_VERSION
        self.profile_id = scene_profile_id(cfg)
        self.seed = cfg["seed"]
        self.synthetic_seed = synthetic_seed
        self.boundaries: list[dict] = []
        self.transitions: list[dict] = []
        self.pending: dict | None = None
        self.finalized = False

    def record_boundary(self, boundary: dict, match, *, same_frame_verified: bool = False):
        if same_frame_verified or boundary.get("render_sync_quality") != "blocked":
            raise ValueError("Current trace must preserve blocked render synchronization")
        current = _boundary(boundary)
        if current["map_phase"] is None and current["full"]["status"] == "terminal":
            current["map_phase"] = ("post_collapse" if current["simulation_seconds"] >=
                                    self.cfg["collapse"]["time"] else "pre_collapse")
        audit = referee_snapshot(match)
        if not math.isclose(current["simulation_seconds"], audit["simulation_seconds"], abs_tol=1e-8):
            raise ValueError("Oracle boundary and referee snapshot have different simulation times")
        if self.pending is not None:
            self._close(current, audit, match, same_frame_verified)
        self.boundaries.append(current)
        return current

    def record_action(self, decision: dict | None, match):
        if self.pending is not None:
            raise ValueError("Previous action has no successor boundary")
        if not self.boundaries:
            raise ValueError("Action requires a current Oracle boundary")
        current = self.boundaries[-1]
        audit = referee_snapshot(match)
        if not math.isclose(current["simulation_seconds"], audit["simulation_seconds"], abs_tol=1e-8):
            raise ValueError("Action is not at the current simulation boundary")
        self.pending = {"start": copy.deepcopy(current), "audit_start": audit,
                        "action": _action(decision, self.cfg),
                        "action_observation_id": None if decision is None else decision.get("observation_id"),
                        "action_simulation_seconds": None if decision is None else decision.get("simulation_seconds"),
                        "execution": {"status": "requested_not_committed", "replacements": [],
                                      "end_active_low_level_action": None}}

    def record_committed_action(self, decision: dict, match):
        if self.pending is None:
            raise ValueError("No requested action to commit")
        if decision.get("observation_id") != self.pending["start"]["observation_id"]:
            raise ValueError("Committed action changed observation ID")
        self.pending["action"] = _action(decision, self.cfg)
        self.pending["action_observation_id"] = decision.get("observation_id")
        self.pending["action_simulation_seconds"] = decision.get("simulation_seconds")
        self.pending["execution"]["status"] = "committed_waiting_execution"

    def reject_action(self, stage: str, reason: str):
        if self.pending is not None:
            self.pending["execution"].update(status="not_applied", failure_stage=stage,
                                              rejection_reason=reason)
            self.finalized = True

    def note_replacement(self, simulation_seconds: float, kind: str):
        if self.pending is not None:
            self.pending["execution"]["replacements"].append({"simulation_seconds": simulation_seconds,
                                                                  "kind": kind})

    def record_execution(self, match, *, steps_executed: int | None = None,
                         smoke_action_status: str | None = None):
        if self.pending is None:
            return
        self.pending["execution"].update(status="observed", steps_executed=steps_executed,
                                         smoke_action_status=smoke_action_status,
                                         match_skill_status=match.skill_status,
                                         end_active_low_level_action=copy.deepcopy(match.active))

    def _close(self, end: dict, audit_end: dict, match, same_frame_verified: bool):
        pending = self.pending
        start, audit_start, action = pending["start"], pending["audit_start"], pending["action"]
        terminal = audit_end["finished"] is not None
        alignment = _alignment(start, end, terminal)
        missing = []
        for view in ("full", "limited"):
            for label, boundary in (("start", start), ("next", end)):
                state = boundary[view]
                if state["status"] not in ("valid", "terminal") or not state["state_id"]:
                    missing.append(f"{label}.{view}.state_id:{state['status'] or 'missing'}")
        if action["raw_skill_request"] is None:
            missing.append("action.raw_skill_request:missing")
        if action["compiled_action"] is None:
            missing.append("action.compiled_action:missing")
        if action["compiled_skill"] is None:
            missing.append("action.compiled_skill:missing")
        if action["action_mapping_status"] != "mapped":
            missing.append("action.oracle_action_id:unmapped")
        if pending["execution"]["status"] != "observed":
            missing.append("action.execution:not_observed")
        if pending["action_observation_id"] != start["observation_id"]:
            missing.append("action.observation_id:mismatch")
        if pending["action_simulation_seconds"] != start["simulation_seconds"]:
            missing.append("action.simulation_seconds:mismatch")
        if not alignment["valid"]:
            missing.extend("alignment:" + error for error in alignment["errors"])
        delta_score = round(audit_end["score"] - audit_start["score"], 6)
        events = _event_summary(match, audit_start)
        components = {"delta_score": delta_score,
                      "A_hp_lost": max(0, audit_start["A_hp"] - audit_end["A_hp"]),
                      "B_hp_lost": max(0, audit_start["B_hp"] - audit_end["B_hp"]),
                      "elimination_events": events["referee_event_counts"].get("eliminated", 0),
                      "terminal_reason": audit_end["finished"]}
        execution = pending["execution"]
        execution["interruption_or_replan_observed"] = bool(
            execution["replacements"] or events["match_event_counts"].get("skill_cancelled", 0))
        record = {"trajectory_id": self.trajectory_id, "transition_index": len(self.transitions),
                  "trace_version": TRACE_VERSION, "state_schema_version": self.state_schema_version,
                  "scene_profile_id": self.profile_id, "config_fingerprint": self.profile_id,
                  "run_seed": self.seed, "synthetic_seed": self.synthetic_seed,
                  "collection_policy_id": "smoke_test_driver_fixed_sequence",
                  "start": start, "action": action, "action_observation_id": pending["action_observation_id"],
                  "next": copy.deepcopy(end), "actual_elapsed_seconds": alignment["actual_elapsed_seconds"],
                  "time_alignment": alignment, "audit": {"start": audit_start, "end": audit_end,
                                                     "event_summary": events, "execution": execution},
                  "reward_components": components, "reward": delta_score,
                  "scalar_reward_status": "defined_as_delta_score",
                  "terminal": terminal, "terminal_reason": audit_end["finished"],
                  "state_action_next_complete": not missing, "missing_or_quality_reasons": missing,
                  "render_sync_quality": start["render_sync_quality"],
                  "same_frame_verified": same_frame_verified,
                  "data_source": "existing_live_match_boundaries_and_referee",
                  "data_quality_flags": ["render_sync_quality_blocked", "fixed_smoke_driver",
                                         "contract_validation_only"],
                  "sample_use": "contract_validation_only"}
        self.transitions.append(record)
        self.pending = None

    def finalize(self):
        self.finalized = True

    def payload(self) -> dict:
        unclosed = [] if self.pending is None else [{
            "trajectory_id": self.trajectory_id,
            "observation_id": self.pending["start"]["observation_id"],
            "simulation_seconds": self.pending["start"]["simulation_seconds"],
            "action": copy.deepcopy(self.pending["action"]),
            "execution": copy.deepcopy(self.pending["execution"]),
            "reason": ("action_not_applied" if self.pending["execution"]["status"] == "not_applied"
                       else "no_successor_boundary" if self.finalized else "awaiting_successor_boundary"),
            "next": None}]
        mapped = sum(item["action"]["action_mapping_status"] == "mapped" for item in self.transitions)
        mapped += sum(item["action"]["action_mapping_status"] == "mapped" for item in unclosed)
        action_count = len(self.transitions) + len(unclosed)
        quality = sorted({reason for item in self.transitions for reason in item["missing_or_quality_reasons"]})
        if unclosed:
            quality.append("unclosed_action:" + unclosed[0]["reason"])
        summary = {"boundary_count": len(self.boundaries), "complete_transition_count": len(self.transitions),
                   "unclosed_action_count": len(unclosed), "mapped_action_count": mapped,
                   "unmapped_action_count": action_count - mapped,
                   "time_alignment_valid": all(item["time_alignment"]["valid"] for item in self.transitions),
                   "contract_complete_count": sum(item["state_action_next_complete"] for item in self.transitions),
                   "missing_or_quality_reasons": quality,
                   "data_quality_flags": ["render_sync_quality_blocked", "fixed_smoke_driver",
                                          "contract_validation_only"]}
        return {"trace_version": TRACE_VERSION, "trajectory_id": self.trajectory_id,
                "state_schema_version": self.state_schema_version, "scene_profile_id": self.profile_id,
                "config_fingerprint": self.profile_id, "run_seed": self.seed,
                "synthetic_seed": self.synthetic_seed,
                "collection_policy_id": "smoke_test_driver_fixed_sequence",
                "render_sync_quality": "blocked", "same_frame_verified": False,
                "data_source": "existing_live_match_boundaries_and_referee",
                "sample_use": "contract_validation_only", "scalar_reward_definition": "delta_score",
                "summary": summary, "transitions": copy.deepcopy(self.transitions),
                "unclosed_actions": unclosed}


def validate_trace_payload(payload: dict) -> dict:
    """Standard-library structural check shared with the upload collector."""
    if not isinstance(payload, dict) or payload.get("trace_version") != TRACE_VERSION:
        raise ValueError("Invalid transition trace version")
    transitions, unclosed, summary = (payload.get("transitions"), payload.get("unclosed_actions"),
                                      payload.get("summary"))
    if not isinstance(transitions, list) or not isinstance(unclosed, list) or not isinstance(summary, dict):
        raise ValueError("Missing transition trace sections")
    if (summary.get("complete_transition_count") != len(transitions) or
            summary.get("unclosed_action_count") != len(unclosed)):
        raise ValueError("Transition trace counts disagree with records")
    if payload.get("render_sync_quality") != "blocked" or payload.get("same_frame_verified") is not False:
        raise ValueError("Transition trace has invalid render synchronization claim")
    mapped = 0
    alignments = []
    previous_next = None
    for index, item in enumerate(transitions):
        if not isinstance(item, dict) or item.get("transition_index") != index:
            raise ValueError("Transition index is missing or out of order")
        if item.get("render_sync_quality") != payload.get("render_sync_quality") or item.get("same_frame_verified") is not False:
            raise ValueError("Transition sync quality is inconsistent")
        start, end = item.get("start"), item.get("next")
        if not isinstance(start, dict) or not isinstance(end, dict):
            raise ValueError("Transition boundary is missing")
        if previous_next is not None and start.get("observation_id") != previous_next:
            raise ValueError("Transition chain skips or duplicates a boundary")
        previous_next = end.get("observation_id")
        alignment = _alignment(start, end, bool(item.get("terminal")))
        if item.get("time_alignment") != alignment or item.get("actual_elapsed_seconds") != alignment["actual_elapsed_seconds"]:
            raise ValueError("Transition time alignment does not match boundary metadata")
        alignments.append(alignment["valid"])
        action = item.get("action")
        if not isinstance(action, dict) or action.get("action_mapping_status") not in ("mapped", "unmapped"):
            raise ValueError("Transition action mapping is missing")
        mapped += action["action_mapping_status"] == "mapped"
        if action["action_mapping_status"] == "mapped" and not action.get("oracle_action_id"):
            raise ValueError("Mapped action lacks Oracle action ID")
        for boundary in (start, end):
            limited = boundary.get("limited")
            if not isinstance(limited, dict) or set(limited) - (set(STATE_KEYS) | {"A.visual_track_region"}):
                raise ValueError("Hidden or unsupported field leaked into limited state")
    for item in unclosed:
        if not isinstance(item, dict) or item.get("next") is not None or not isinstance(item.get("action"), dict):
            raise ValueError("Unclosed action has a fabricated successor or no action record")
        mapped += item["action"].get("action_mapping_status") == "mapped"
    if (summary.get("mapped_action_count") != mapped or
            summary.get("unmapped_action_count") != len(transitions) + len(unclosed) - mapped or
            summary.get("time_alignment_valid") != all(alignments)):
        raise ValueError("Transition summary disagrees with action or alignment records")
    return {"found": True, **summary, "parse_error": None}
