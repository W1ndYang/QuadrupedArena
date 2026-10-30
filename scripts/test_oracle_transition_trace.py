"""CPU-only contract tests for the optional Oracle transition trace."""

import copy
from types import SimpleNamespace
import unittest

from adaptive_motion import load_config
from oracle_state_v1 import action_template
from oracle_state_v1_smoke import encode_boundary
from oracle_transition_trace import TransitionTrace, validate_trace_payload
from test_oracle_state_v1 import fixture
from vision_core import CameraRig, public_packet


def boundary(index, t, *, full_status="valid", limited_status="valid", full_id=None, limited_id=None):
    return {"observation_id": index, "simulation_seconds": t, "frame_count": 1 if index == 0 else 4,
            "render_sync_quality": "blocked",
            "full": {"status": full_status, "reason": "region_unverified" if full_status == "blocked" else None,
                     "state_id": full_id if full_id is not None else (None if full_status == "blocked" else f"f{index}"),
                     "decision_step": index, "map_phase": "pre_collapse",
                     "A.region_id": "open_traversable", "B.region_id": "cover_High2"},
            "limited": {"status": limited_status, "reason": "region_unverified" if limited_status == "blocked" else None,
                        "state_id": limited_id if limited_id is not None else (None if limited_status == "blocked" else f"l{index}"),
                        "decision_step": index, "map_phase": "pre_collapse",
                        "A.region_id": "open_traversable", "A.visual_track_region": "unknown",
                        "B.region_id": "SECRET_MUST_NOT_COPY"}}


def fake_match(t=0., score=0., hp=(3, 3), finished=None):
    referee = SimpleNamespace(score=score, hp=list(hp), active_target=0, generation=1,
                              dwell=[0., 0.], finished=finished, events=[])
    return SimpleNamespace(t=t, ref=referee, events=[], skill_status="RUNNING",
                           active={"skill": "HOLD", "attack": False})


def decision(index=0, t=0., skill="MOVE_TO", params=None):
    return {"observation_id": index, "simulation_seconds": t, "skill": skill,
            "params": {"target": "P2", "risk": "fast"} if params is None else params,
            "reason": "Finite smoke_test_driver, not an LLM policy",
            "compiled_skill": {"skill": skill, "state": "RUNNING", "route": [[1., 2.]]},
            "compiled_action": {"skill": "MOVE", "goal": [1., 2.], "attack": False}}


class OracleTransitionTraceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config("configs/trial_rotating.json")
        self.cfg["seed"] = 2026102107

    def make_trace(self):
        return TransitionTrace("cpu_fixture", self.cfg, 20261028)

    def test_adjacent_boundaries_action_alignment_and_scalar_score_reward(self):
        match, trace = fake_match(), self.make_trace()
        trace.record_boundary(boundary(0, 0.), match)
        trace.record_action(decision(), match)
        match.t, match.ref.score = 2., 1.3
        trace.record_execution(match, steps_executed=100, smoke_action_status="executed")
        trace.record_boundary(boundary(1, 2.), match)
        payload = trace.payload()
        self.assertEqual(payload["summary"]["complete_transition_count"], 1)
        self.assertEqual(payload["summary"]["unclosed_action_count"], 0)
        record = payload["transitions"][0]
        self.assertEqual((record["start"]["observation_id"], record["action_observation_id"],
                          record["next"]["observation_id"]), (0, 0, 1))
        self.assertEqual((record["start"]["simulation_seconds"], record["next"]["simulation_seconds"],
                          record["actual_elapsed_seconds"]), (0., 2., 2.))
        self.assertTrue(record["state_action_next_complete"])
        self.assertEqual(record["reward_components"]["delta_score"], 1.3)
        self.assertEqual(record["reward"], 1.3)
        self.assertEqual(record["scalar_reward_status"], "defined_as_delta_score")
        self.assertEqual(record["audit"]["start"]["score"], 0.)
        self.assertEqual(record["audit"]["end"]["score"], 1.3)
        self.assertEqual(record["render_sync_quality"], "blocked")
        self.assertFalse(record["same_frame_verified"])
        self.assertNotIn("B.region_id", record["start"]["limited"])
        self.assertNotIn("B.region_id", record["next"]["limited"])
        self.assertEqual(validate_trace_payload(payload)["complete_transition_count"], 1)

    def test_five_smoke_like_boundaries_close_four_actions_without_offset(self):
        match, trace = fake_match(), self.make_trace()
        for index in range(5):
            match.t = 2. * index
            trace.record_boundary(boundary(index, match.t), match)
            if index < 4:
                trace.record_action(decision(index=index, t=match.t), match)
                trace.record_execution(match, steps_executed=100, smoke_action_status="executed")
        trace.finalize()
        payload = trace.payload()
        self.assertEqual(payload["summary"]["boundary_count"], 5)
        self.assertEqual(payload["summary"]["complete_transition_count"], 4)
        self.assertEqual(payload["summary"]["unclosed_action_count"], 0)
        self.assertEqual(payload["summary"]["mapped_action_count"], 4)
        self.assertEqual([(item["start"]["observation_id"], item["action_observation_id"],
                           item["next"]["observation_id"]) for item in payload["transitions"]],
                         [(0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 3, 4)])
        self.assertTrue(payload["summary"]["time_alignment_valid"])
        self.assertEqual(validate_trace_payload(payload)["complete_transition_count"], 4)

    def test_action_mapping_keeps_raw_and_compiled_forms_separate(self):
        match, trace = fake_match(), self.make_trace()
        trace.record_boundary(boundary(0, 0.), match)
        request = decision(skill="OBSERVE", params={"direction": "north"})
        trace.record_action(request, match)
        action = trace.payload()["unclosed_actions"][0]["action"]
        self.assertEqual(action["raw_skill_request"]["params"], {"direction": "north"})
        self.assertEqual(action["oracle_action_id"],
                         action_template("OBSERVE", {"direction": "north"}, self.cfg).template_id)
        self.assertEqual(action["normalized_oracle_action"],
                         {"skill": "OBSERVE", "params": {"direction": "north"}})
        self.assertEqual(action["compiled_action"]["skill"], "MOVE")
        self.assertEqual(action["compiled_skill"]["skill"], "OBSERVE")
        request["params"]["direction"] = "south"
        self.assertEqual(action["raw_skill_request"]["params"], {"direction": "north"})

    def test_unmapped_action_and_no_successor_are_explicit(self):
        match, trace = fake_match(), self.make_trace()
        trace.record_boundary(boundary(0, 0.), match)
        trace.record_action(decision(params={"target": "P2", "risk": "fast", "sighting": {"time": 0, "feet": [1, 2]}}), match)
        trace.finalize()
        payload = trace.payload()
        self.assertEqual(payload["summary"]["complete_transition_count"], 0)
        self.assertEqual(payload["summary"]["unclosed_action_count"], 1)
        self.assertEqual(payload["summary"]["unmapped_action_count"], 1)
        self.assertEqual(payload["unclosed_actions"][0]["reason"], "no_successor_boundary")
        self.assertIsNone(payload["unclosed_actions"][0]["next"])
        self.assertIsNone(payload["unclosed_actions"][0]["action"]["oracle_action_id"])
        self.assertIn("sighting", payload["unclosed_actions"][0]["action"]["raw_skill_request"]["params"])

    def test_rejected_request_remains_unclosed_with_failure_stage(self):
        match, trace = fake_match(), self.make_trace()
        trace.record_boundary(boundary(0, 0.), match)
        trace.record_action(decision(), match)
        trace.reject_action("skill_validation", "ValueError: invalid request")
        pending = trace.payload()["unclosed_actions"][0]
        self.assertEqual(pending["reason"], "action_not_applied")
        self.assertEqual(pending["execution"]["failure_stage"], "skill_validation")
        self.assertIsNone(pending["next"])

    def test_missing_action_blocked_state_and_index_mismatch_keep_reasons(self):
        match, trace = fake_match(), self.make_trace()
        trace.record_boundary(boundary(0, 0., limited_status="blocked"), match)
        trace.record_action(None, match)
        match.t = 2.
        trace.record_boundary(boundary(2, 2.), match)
        record = trace.payload()["transitions"][0]
        self.assertFalse(record["state_action_next_complete"])
        self.assertFalse(record["time_alignment"]["valid"])
        self.assertIn("start.limited.state_id:blocked", record["missing_or_quality_reasons"])
        self.assertIn("action.raw_skill_request:missing", record["missing_or_quality_reasons"])
        self.assertIn("alignment:observation_id_not_adjacent", record["missing_or_quality_reasons"])

    def test_terminal_damage_events_and_early_interval(self):
        match, trace = fake_match(), self.make_trace()
        trace.record_boundary(boundary(0, 0.), match)
        trace.record_action(decision(), match)
        match.t, match.ref.score, match.ref.hp = 1.5, 0.4, [0, 2]
        match.ref.finished = "quadruped_eliminated"
        match.ref.events.extend([{"type": "hit", "threshold": 999, "hazard": 999},
                                 {"type": "eliminated", "actor": 0},
                                 {"type": "finish", "reason": "quadruped_eliminated"}])
        match.events.append({"type": "skill_cancelled", "reason": "target_expired"})
        trace.note_replacement(1., "skill_tick_replacement")
        trace.record_execution(match, steps_executed=75, smoke_action_status="executed")
        trace.record_boundary(boundary(1, 1.5, full_status="terminal", limited_status="terminal"), match)
        record = trace.payload()["transitions"][0]
        self.assertEqual(record["actual_elapsed_seconds"], 1.5)
        self.assertTrue(record["terminal"])
        self.assertEqual(record["terminal_reason"], "quadruped_eliminated")
        self.assertEqual(record["reward"], .4)
        self.assertEqual(record["reward_components"]["A_hp_lost"], 3)
        self.assertEqual(record["reward_components"]["B_hp_lost"], 1)
        self.assertEqual(record["reward_components"]["elimination_events"], 1)
        self.assertTrue(record["audit"]["execution"]["interruption_or_replan_observed"])
        self.assertNotIn("threshold", str(record))
        self.assertNotIn("hazard", str(record))

    def test_read_only_trace_does_not_step_commit_or_draw_rng(self):
        match, trace = fake_match(), self.make_trace()
        def forbidden(*_args, **_kwargs):
            raise AssertionError("transition recorder changed the match")
        match.advance = forbidden
        match.apply = forbidden
        match.commit = forbidden
        match.ref.rng = forbidden
        before = copy.deepcopy((match.t, match.ref.score, match.ref.hp, match.ref.events, match.events))
        trace.record_boundary(boundary(0, 0.), match)
        trace.record_action(decision(), match)
        trace.record_execution(match)
        trace.finalize()
        self.assertEqual(before, (match.t, match.ref.score, match.ref.hp, match.ref.events, match.events))

    def test_one_real_cpu_match_interval_uses_existing_boundaries(self):
        cfg, match, engine, catalog, packet = fixture()
        rig = CameraRig()
        trace = TransitionTrace("cpu_match_interval", cfg, 20261028)
        first = encode_boundary(match, engine, rig, packet, (None,), cfg, catalog)
        before = (match.tick, match.t, match.ref.score, len(match.decisions))
        trace.record_boundary(first, match)
        self.assertEqual(before, (match.tick, match.t, match.ref.score, len(match.decisions)))
        request = {"observation_id": 0, "skill": "OBSERVE", "params": {"direction": "north"}}
        trace.record_action({**request, "simulation_seconds": 0.}, match)
        action, route, _, compiled = engine.commit(request, packet)
        match.compile(action)
        match.apply(action, route)
        trace.record_committed_action({**request, "simulation_seconds": 0., "compiled_action": action,
                                       "compiled_skill": compiled}, match)
        frames = []
        for t in (.5, 1., 1.5, 2.):
            match.advance(.5)
            frames.append({"simulation_seconds": t, "camera_world_yaw_degrees": 0.,
                           "camera_eye": [-3.64, -1.5, .6], "file": f"cpu_no_rgb_{t:.1f}.png"})
        next_packet = public_packet(1, match.states[-1], frames, cfg["start_b"])
        next_packet["skills"] = engine.public(match.t)
        previous_id = action_template("OBSERVE", {"direction": "north"}, cfg).template_id
        second = encode_boundary(match, engine, rig, next_packet, (None,) * 4, cfg,
                                 catalog, previous_id)
        trace.record_execution(match, steps_executed=100, smoke_action_status="executed")
        at_boundary = (match.tick, match.t, match.ref.score, len(match.decisions))
        trace.record_boundary(second, match)
        self.assertEqual(at_boundary, (match.tick, match.t, match.ref.score, len(match.decisions)))
        record = trace.payload()["transitions"][0]
        self.assertEqual(record["action"]["oracle_action_id"], previous_id)
        self.assertEqual(record["next"]["full"]["state_id"], second["full"]["state_id"])
        self.assertEqual(record["next"]["limited"]["state_id"], second["limited"]["state_id"])
        self.assertEqual(record["reward"], round(match.ref.score - before[2], 6))


if __name__ == "__main__":
    unittest.main()
