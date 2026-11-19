"""Matched-input prediction, direct Skill decisions and hard numerical checks."""
import time

from formal_two_stage import FIRST_POINTS, SECOND_POINTS
from llm_decision_core import DecisionOrchestrator, PolicyAbstention
from llm_motion_client import SYSTEM_PROMPT
from llm_robot_rollout import geometry, prediction_horizon
from llm_study_protocol import (DIRECT_PROMPT, DIRECT_PROMPT_VERSION, digest, messages,
                                parse_direct, shared_packet)
from online_env import ACTION_NAMES


MATCHED_VERSION = 'guard-motion-full-public-v1'
MATCHED_PROMPT = SYSTEM_PROMPT + """
The user packet uses the same complete public state schema as the direct Skill
task. In this packet map.edges is keyed by edge ID. Use only directed edges
with guard in traversable_by and with both endpoints in current_track.region.
Find the latest guard measurement for the current track at latest_measurement_at.
Use decision.motion_window for game_time, the latest measurement and horizons.
Output map_id from map.map_id and track_id from decision.current_track.id.
"""


def decision_packet(request, history, semantic_map, rules, track, config, horizon=None):
    packet = shared_packet(request, history, semantic_map, rules, track)
    packet['decision']['skill_state'] = dict(boundary_waits=0, retreats=0) | request['skill_state']
    if track and track['observations']:
        latest = track['observations'][-1]['time']
        age = request['time']-latest
        if age <= 4.:
            if horizon is None:
                horizon = prediction_horizon(request['public'], request['stage'])
            packet['decision']['motion_window'] = dict(
                game_time=request['time'], latest_measurement_at=latest,
                measurement_age_seconds=age, horizon_seconds=horizon,
                path_coverage_seconds=horizon+age)
    packet['decision']['limits'] = dict(
        additional_peek_allowed=(
            request['public']['peek_count'] < config.get('max_peeks_per_stage', 3)
            and request['skill_state'].get('observation_window_elapsed',
                  request['public']['stage_elapsed_seconds']) < config.get('max_observation_seconds', 12.)
            and any(request['public']['action_mask'][2:4])),
        post_gate_waits_left=max(0, 2-request['skill_state'].get('boundary_waits', 0)),
        post_gate_retreat_available=request['skill_state'].get('retreats', 0) < 1)
    return packet


def live_packet(orchestrator, request, context, track, horizon=None):
    history = [row for prior in context.requests for row in prior['observations']]
    return decision_packet(request, history, orchestrator.map, orchestrator.rules, track,
                           orchestrator.config, horizon)


class MatchedPrediction(DecisionOrchestrator):
    def _motion_messages(self, reasoning_request, request, context, track):
        packet = live_packet(self, request, context, track, reasoning_request['horizon_seconds'])
        return messages(MATCHED_PROMPT, packet), MATCHED_VERSION, digest(MATCHED_PROMPT)


def hard_rejection(skill, request, legal, limits, geometry_check):
    if skill not in legal:
        return 'skill_mask'
    if skill.startswith('PEEK_') and not limits['additional_peek_allowed']:
        return 'observation_budget'
    public = request['public']
    if skill.startswith('SMOKE_') and not public['smoke_available']:
        return 'smoke_inventory'
    if public['action_mask'][7]:
        if skill == 'WAIT' and not limits['post_gate_waits_left']:
            return 'post_gate_wait_budget'
        if skill == 'TAKE_COVER' and not limits['post_gate_retreat_available']:
            return 'post_gate_retreat_budget'
    if not geometry_check or skill == 'WAIT':
        return None
    stage = request['stage']
    cover = (3. if stage else -8., 0.)
    planner = geometry()[2]
    origin = public['own_xy']
    if skill == 'TAKE_COVER':
        target = cover
    elif skill.startswith('PEEK_'):
        points = SECOND_POINTS if stage else FIRST_POINTS
        target = points['CENTER_S_PEEK' if skill.endswith('SOUTH') else 'CENTER_N_PEEK']
        if planner.route(target, cover) is None:
            return 'peek_return_route_unavailable'
    elif skill == 'NEXT_COVER_OR_GOAL':
        previous = public['last_action']
        if stage and previous not in (5, 6, 8, 9):
            # A WAIT loses the action ID of the selected gate. Let the existing
            # executor use its retained gate instead of guessing a destination.
            return None
        target = (10., -1.6 if previous in (5, 8) else 1.6) if stage else (3.01, 0.)
    else:
        target = (6. if stage else -5., -1.6 if 'SOUTH' in skill else 1.6)
    if planner.route(origin, target) is None:
        return 'public_geometry_route_unavailable'
    return None


def screen(parsed, request, decision, enabled):
    skill = parsed['skill']
    if skill == 'UNKNOWN':
        return dict(status='abstain', proposed=skill, executed=None, rejection=None,
                    fallback=None)
    legal, limits = decision['allowed_skills'], decision['limits']
    rejection = hard_rejection(skill, request, legal, limits, geometry_check=enabled)
    if rejection is None:
        return dict(status='execute', proposed=skill, executed=skill, rejection=None,
                    fallback=None)
    fallback = None
    if enabled:
        for candidate in ('TAKE_COVER', 'WAIT'):
            if hard_rejection(candidate, request, legal, limits, geometry_check=True) is None:
                fallback = candidate
                break
    return dict(status='execute' if fallback else 'rejected', proposed=skill,
                executed=fallback, rejection=rejection, fallback=fallback)


class DirectDecision(DecisionOrchestrator):
    def decide(self, request):
        started = time.perf_counter()
        context, cached = self._begin_decision(request)
        if cached is not None:
            return cached
        request_id = request['request_id']
        track = context.track_for(request['stage'])
        if request['public']['peek_count'] == 0 and not request['public']['action_mask'][7]:
            first = self._observe_action(request, track, 'initial_observation')
            return self._record(context, request_id, dict(
                first, status='skill', llm_called=False,
                timing=dict(api_seconds=0., local_seconds=time.perf_counter()-started)))
        used = sum(prior['request_id'] in context.llm_messages and
                   prior['stage'] == request['stage'] for prior in context.requests)
        if used >= self.config.get('direct_call_cap_per_stage', 12):
            raise PolicyAbstention('Direct model decision budget exhausted')
        packet = live_packet(self, request, context, track)
        exchange = messages(DIRECT_PROMPT, packet)
        context.llm_messages[request_id] = exchange
        api_start = time.perf_counter()
        inferred = self.reasoner.infer(exchange)
        api_seconds = time.perf_counter() - api_start
        context.llm_attempts[request_id] = inferred
        enabled = self.config['study_architecture'] == 'D3'
        # D3 records an out-of-mask proposal and may recover. A structurally
        # invalid response is neither parsed into a Skill nor repaired.
        parsed = parse_direct(inferred['text'], list(ACTION_NAMES[1:]))
        checked = screen(parsed, request, packet['decision'], enabled)
        reasoning = dict(raw_request=exchange, raw_response=inferred, direct=parsed,
                         safety_check=checked, prompt_version=DIRECT_PROMPT_VERSION,
                         prompt_digest=digest(DIRECT_PROMPT), api_seconds=api_seconds)
        if checked['executed'] is None:
            context.llm_attempts[request_id] = dict(inferred, direct_result=reasoning['direct'],
                                                    safety_check=checked)
            if checked['status'] == 'abstain':
                raise PolicyAbstention('Direct model returned UNKNOWN')
            raise PolicyAbstention(f"Direct proposal rejected: {checked['rejection']}")
        action = ACTION_NAMES.index(checked['executed'])
        return self._record(context, request_id, dict(
            status='skill', action=action, llm_called=True, reasoning=reasoning,
            timing=dict(api_seconds=api_seconds,
                        local_seconds=time.perf_counter()-started-api_seconds)))


def make_orchestrator(reasoner, config):
    name = config.get('study_architecture')
    if name == 'D1_matched':
        return MatchedPrediction(reasoner, config)
    if name in ('D2', 'D3'):
        return DirectDecision(reasoner, config)
    if name is not None:
        raise ValueError(f'Unknown research architecture: {name}')
    return DecisionOrchestrator(reasoner, config)
