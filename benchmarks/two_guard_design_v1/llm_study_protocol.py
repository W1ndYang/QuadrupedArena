"""Versioned, public-only inputs and structured outputs for the LLM study."""
import hashlib
import json
import math

from llm_semantic import ModelOutputError
from online_env import ACTION_NAMES


DIRECT_VERSION = 'skill-decision-v1'
DIRECT_PROMPT_VERSION = 'public-skill-decision-v2'
BEHAVIOR_VERSION = 'guard-behavior-v1'
BEHAVIOR_PROMPT_VERSION = 'public-guard-behavior-v2'
DIRECT_PROMPT = """Choose the next single Skill in the two-stage simulated tag game.
Return a JSON object with exactly schema, skill, reason, uncertainty.
These four keys must be at the top level. Begin with { and end with }.
Do not add Markdown fences, prose outside JSON, or a wrapper object.
schema is skill-decision-v1. skill is an exact ID from allowed_skills, or
UNKNOWN if you cannot responsibly choose. reason and uncertainty are short strings.
Use the map, public rules, robot state, resources and legal observations.
PEEK_SOUTH or PEEK_NORTH selects continued observation and its direction.
SMOKE_SOUTH or SMOKE_NORTH explicitly spends smoke. Gate skills without smoke
preserve it. WAIT is one existing one-second Option; the next boundary replans.
Choose one Skill only, with no extra parameters or continuous control.
When limits are supplied, respect the remaining peek, wait and retreat budgets.
Treat missing evidence and geometric paths as uncertain. Do not assume private
waypoints, an unseen guard location, or future observations.
"""
BEHAVIOR_PROMPT = """Infer evidence for behavior in the current guard region.
Return one JSON object with exactly schema, facts, assessments.
These three keys must be at the top level. Begin with { and end with }.
Do not add Markdown fences, prose outside JSON, or a wrapper object.
schema is guard-behavior-v1. facts is a list (at most six) of objects
with exactly observation_time and detail; use only supplied observation times.
assessments is a list with one row for each kind: periodicity, region_alternation,
repeated_dwell, patrol_phase, change_from_pattern. Each row has exactly kind,
status (SUPPORTED, CONTRADICTED or UNKNOWN), evidence_times (an array of supplied
observation times), inference (short string), and uncertainty (short string).
Claim only what the evidence supports. General public rules can constrain an
inference but are insufficient to establish this episode's private patrol phase.
No detection, discontinuous observation, stationary position and endpoint dwell
are distinct. Use UNKNOWN when competing explanations cannot be separated.
"""
BEHAVIOR_KINDS = {'periodicity', 'region_alternation', 'repeated_dwell',
                  'patrol_phase', 'change_from_pattern'}

SKILL_RULES = {
    'objective': 'Reach the goal in two stages before the game clock expires. '
                 'Being continuously marked for the rule threshold ends the run.',
    'handoff': 'Cross the first gate and reach the second cover to enter stage 1. '
               'The second guard patrols from t=0 and is not observed in stage 0.',
    'options': {
        'TAKE_COVER': 'Return to the cover for the current stage.',
        'PEEK_SOUTH': 'Peek toward the south and return to cover.',
        'PEEK_NORTH': 'Peek toward the north and return to cover.',
        'WAIT': 'Wait up to one game second; fresh detection can return early.',
        'SOUTH_GATE': 'Move to the south gate without using smoke.',
        'NORTH_GATE': 'Move to the north gate without using smoke.',
        'NEXT_COVER_OR_GOAL': 'Continue past the gate to second cover or goal.',
        'SMOKE_SOUTH': 'Use one smoke resource and move toward the south gate.',
        'SMOKE_NORTH': 'Use one smoke resource and move toward the north gate.',
    },
}


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(',', ':'))


def digest(value):
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def allowed_skills(request):
    mask = request['public']['action_mask']
    if len(mask) != len(ACTION_NAMES) or any(type(bit) is not bool for bit in mask):
        raise ValueError('Invalid public Skill mask')
    return [name for i, name in enumerate(ACTION_NAMES) if i and mask[i]]


def shared_packet(request, history, semantic_map, rules, track=None):
    if any(row['time'] > request['time'] + 1e-5 for row in history):
        raise ValueError('Future observation in study packet')
    public = request['public']
    if (not isinstance(public['own_xy'], list) or
            len(public['own_xy']) != 2 or
            any(type(x) not in (float, int) or not math.isfinite(x) for x in public['own_xy'])):
        raise ValueError('Invalid public own position')
    numbers = ('own_yaw', 'own_speed', 'remaining_seconds', 'stage_elapsed_seconds', 'peek_count')
    if any(type(public[k]) not in (float, int) or not math.isfinite(public[k]) for k in numbers):
        raise ValueError('Invalid numeric public state')
    if (type(public['smoke_available']) is not bool or
            not isinstance(public['skill_feedback'], str) or
            public['last_action'] is not None and type(public['last_action']) is not int):
        raise ValueError('Invalid typed public state')
    fields = {'current_option', 'plan_request_id', 'waits_done', 'recovery',
              'observation_window_elapsed', 'boundary_waits', 'retreats'}
    if (request['skill_state'].keys()-fields or any(
            type(v) not in (bool, float, int, str, type(None))
            for v in request['skill_state'].values())):
        raise ValueError('Unexpected public Skill state')
    if any(type(row['observer_yaw']) not in (float, int) or
           not math.isfinite(row['observer_yaw']) for row in history):
        raise ValueError('Invalid observer angle')
    return dict(map=semantic_map, rules=rules, skill_rules=SKILL_RULES,
                decision=dict(request_id=request['request_id'],
                              game_time=request['time'], stage=request['stage'],
                              decision_type=request['decision_type'],
                              current_robot_state=public,
                              skill_state=request['skill_state'],
                              observations=history,
                              allowed_skills=allowed_skills(request),
                              current_track=public_track(track)))


def public_track(track):
    if track is None:
        return None
    return {key: track[key] for key in
            ('id', 'region', 'estimated_velocity', 'association_uncertainty')}


def messages(prompt, packet):
    return [dict(role='system', content=prompt),
            dict(role='user', content=stable_json(packet))]


def strict_json(text):
    def unique_pairs(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ModelOutputError(f'Duplicate JSON property: {key}')
            obj[key] = value
        return obj

    def invalid_number(value):
        raise ModelOutputError(f'Invalid JSON constant: {value}')

    try:
        return json.loads(text, object_pairs_hook=unique_pairs,
                          parse_constant=invalid_number)
    except (TypeError, ValueError) as exc:
        raise ModelOutputError(str(exc)) from exc


def _fields(value, fields):
    if not isinstance(value, dict) or value.keys() != set(fields.split()):
        raise ModelOutputError(f'Expected object fields: {fields}')


def _short_text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise ModelOutputError('Expected a nonempty short explanation')


def parse_direct(text, legal):
    result = strict_json(text)
    _fields(result, 'schema skill reason uncertainty')
    if result['schema'] != DIRECT_VERSION or result['skill'] not in [*legal, 'UNKNOWN']:
        raise ModelOutputError('Unexpected direct decision schema or illegal Skill')
    _short_text(result['reason'])
    _short_text(result['uncertainty'])
    return result


def parse_behavior(text, history):
    result = strict_json(text)
    _fields(result, 'schema facts assessments')
    if result['schema'] != BEHAVIOR_VERSION:
        raise ModelOutputError('Unexpected behavior schema')
    times = {round(row['time'], 6) for row in history}

    def check_time(time):
        if (type(time) not in (int, float) or abs(time) > 100000 or
                not math.isfinite(time) or round(time, 6) not in times):
            raise ModelOutputError('Evidence timestamp absent from legal observation history')

    if not isinstance(result['facts'], list) or len(result['facts']) > 6:
        raise ModelOutputError('Expected up to six observations')
    for fact in result['facts']:
        _fields(fact, 'observation_time detail')
        check_time(fact['observation_time'])
        _short_text(fact['detail'])
    rows = result['assessments']
    if not isinstance(rows, list) or len(rows) != len(BEHAVIOR_KINDS):
        raise ModelOutputError('Expected one assessment per behavior kind')
    kinds = set()
    for row in rows:
        _fields(row, 'kind status evidence_times inference uncertainty')
        if (not isinstance(row['kind'], str) or row['kind'] not in BEHAVIOR_KINDS
                or row['kind'] in kinds):
            raise ModelOutputError('Unexpected or duplicate behavior kind')
        kinds.add(row['kind'])
        if row['status'] not in ('SUPPORTED', 'CONTRADICTED', 'UNKNOWN'):
            raise ModelOutputError('Unexpected behavior status')
        refs = row['evidence_times']
        if not isinstance(refs, list) or len(refs) > 12:
            raise ModelOutputError('Expected up to twelve evidence times')
        for stamp in refs:
            check_time(stamp)
        if row['status'] != 'UNKNOWN' and not refs:
            raise ModelOutputError('A conclusion must cite legal observations')
        _short_text(row['inference'])
        _short_text(row['uncertainty'])
    return result
