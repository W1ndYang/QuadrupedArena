"""Decision protocol between the quadruped agent and a language model (breach-protocol-v1).

The model sees: public rules, the public map (walls, goal, named points, optional play
routes), and per decision a packet with ONLY the quadruped's legal information. It answers
with a short plan of skills. Nothing here can read the referee or the human's hidden state.
"""
import json
import math

from .. import PROTOCOL_VERSION, RULES_VERSION

MAX_STEPS = 5
ON_CONTACT = ('retreat', 'continue')
STYLES = ('hold', 'standoff', 'hunt', 'unknown')

SKILL_DOC = {
    'MOVE': ('to: point; via: [points] (<=6); on_contact: retreat|continue (default retreat)',
             'Path to a point. retreat ends the skill when the human is seen within 9 m and the route ahead '
             'stays exposed (crossing a door view at speed is tolerated), then the agent retreats to cover.'),
    'HOLD': ('seconds: 0..10; on_contact: retreat|continue; until: "contact" (optional)',
             'Stand still and watch (face the human if visible).'),
    'SMOKE_RUSH': ('throw_from: point; smoke_at: point or [x,y]; approach_via: [points]; run_via: [points]',
                   'Go to throw_from (retreats on contact), stop, throw the single smoke at smoke_at '
                   '(<=6 m, clear line at 1 m height), then commit to the goal through run_via. '
                   'A rejected throw ends the skill as failed and keeps the smoke.'),
    'FEINT': ('at: point; approach_via: [points]; show_s: 0..6; leave_inside_m: 8.5..12; staging: point; relocate_via: [points]',
              'Go to a point the human can see from far away, stay visible up to show_s (leave early if the '
              'human closes inside leave_inside_m), then relocate out of sight to staging.'),
    'DASH': ('via: [points]', 'Committed run to the goal without smoke.'),
    'TAG': ('max_s: 0.5..4', 'Stop and aim at the visible human; a 1.0 s lock stuns him 1 s (4 s cooldown). '
                             'Does not win the match.'),
    'RETREAT': ('(none)', 'Run to the nearest named point hidden from the human\'s last legal position.'),
}


def _round(v):
    return [round(float(x), 2) for x in v]


def map_brief(game_map, hints='routes'):
    """Public map description for the system prompt. hints: none | routes | full."""
    m = game_map
    walls = [dict(name=c['name'], x=[round(c['center'][0] - c['size'][0] / 2, 2), round(c['center'][0] + c['size'][0] / 2, 2)],
                  y=[round(c['center'][1] - c['size'][1] / 2, 2), round(c['center'][1] + c['size'][1] / 2, 2)],
                  h=c['size'][2]) for c in m.covers]
    points = {k: dict(xy=_round(v), kind=m.kinds[k], label=m.labels[k]) for k, v in m.points.items()}
    brief = dict(id=m.id, title=m.title, summary=m.data.get('summary', ''), arena_m=list(m.size),
                 goal=dict(center=_round(m.goal), radius=m.goal_radius), walls=walls, points=points,
                 spawns=dict(quadruped=_round(m.spawns['quadruped']['xy']), human_start=_round(m.spawns['human']['xy'])))
    if hints in ('routes', 'full'):
        plays = json.loads(json.dumps(m.plays))
        if hints == 'routes':
            for t in plays.get('throws', {}).values():
                t.pop('smoke_hints', None)
            for L in plays.get('lures', {}).values():
                L.pop('commit_m', None)
                L.pop('chase', None)
        brief['plays'] = plays
    return brief


STRATEGY = """Read the human from your legal sightings: guarding (on the goal or a stand-off spot) versus hunting
(left the goal). Guarding usually loses to a smoke placed to cut HIS line of sight to your final run; hunting
usually loses to showing yourself on one side, relocating unseen and entering from the other side once he is
far out. Placing the smoke on the goal only blinds a guard standing on the goal.
"""


def system_prompt(game_map, rules, hints='routes', language='zh', strategy=True):
    r = rules.raw
    skills = '\n'.join(f'- {k}({v[0]}): {v[1]}' for k, v in SKILL_DOC.items())
    mb = json.dumps(map_brief(game_map, hints), ensure_ascii=False, separators=(',', ':'))
    lang = '用中文写 assessment。' if language == 'zh' else 'Write the assessment in English.'
    return f"""You command the QUADRUPED (attacker) in Breach Arena ({RULES_VERSION}, {PROTOCOL_VERSION}).
Win: your body centre enters the goal circle (r={game_map.goal_radius} m) before {r['duration_s']} s.
Lose: the human keeps you in sight for {r['tag']['seconds']} s continuous aim (<= {r['tag']['range_m']} m, he must be
almost still, after ~0.2 s reaction) - one tag ends the match; or time runs out.
You: upright {r['quadruped']['postures']['upright']['speed']} m/s; human {r['human']['speed']} m/s. You see all around;
he also detects all around but must turn to aim. Low walls are {0.7} m: you (0.40 m) are hidden right behind them,
he (eye 1.05 m) sees over them at distance. You have ONE smoke: throw <= {r['smoke']['throw_range_m']} m, clear line at
1 m height, {r['smoke']['cast_s']} s standing cast, active {r['smoke']['fuse_s']} s later for {r['smoke']['duration_s']} s,
radius {r['smoke']['radius_m']} m, blocks all sight through it.

The match does NOT pause while you think. Your plan is applied when it arrives; until then the current skill
keeps running (or you hold still). Prefer plans that stay safe for a few seconds of delay.
{STRATEGY if strategy else ''}
Skills (plan = 1..{MAX_STEPS} steps executed in order; points are names from the map or [x,y]):
{skills}

Map (public, both sides know it):
{mb}

Reply with ONE JSON object only:
{{"observation_id": <int from the packet>,
  "plan": [{{"skill": "MOVE", "params": {{"to": "F_W", "via": ["W_PLAZA_N"], "on_contact": "retreat"}}}}],
  "assessment": {{"human_style": "hold|standoff|hunt|unknown", "evidence": "...", "decision_summary": "...",
                  "risk": "..."}}}}
{lang} Keep assessment fields under 60 words each."""


def compact_packet(packet, game_map):
    """User-message JSON: the agent's legal packet, trimmed for tokens."""
    obs = packet['observation']
    contacts = packet.get('contacts', [])
    return dict(
        observation_id=packet['observation_id'], trigger=packet['trigger'], t=obs['t'], remaining_s=obs['remaining'],
        me=obs['self'], human_visible_now=obs['opponent'], human_last_known=obs['memory'],
        sightings=[dict(t=c['t'], human=c['xy'], me=c.get('me')) for c in contacts[-12:]],
        smokes=obs['smokes'], current_skill=packet.get('current'), queued=packet.get('queued'),
        recent_skills=[dict(skill=r['skill'], status=r['status'], reason=r['reason'], ended=r.get('ended'))
                       for r in packet.get('recent', [])],
        earlier_rounds=packet.get('history', []))


class PlanError(ValueError):
    pass


def _point(spec, game_map, field):
    try:
        return game_map.resolve(spec)
    except (KeyError, ValueError) as e:
        raise PlanError(f'{field}: {e}') from None


def _points(lst, game_map, field, limit=6):
    if lst is None:
        return []
    if not isinstance(lst, list) or len(lst) > limit:
        raise PlanError(f'{field} must be a list of at most {limit} points')
    for i, p in enumerate(lst):
        _point(p, game_map, f'{field}[{i}]')
    return lst


def _num(params, key, lo, hi, default=None):
    v = params.get(key, default)
    if v is None:
        raise PlanError(f'{key} is required')
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi or math.isnan(v):
        raise PlanError(f'{key} must be a number in [{lo}, {hi}]')
    return float(v)


def validate_plan(plan, game_map):
    """Return a cleaned plan or raise PlanError with a message the model can act on."""
    if not isinstance(plan, list) or not 1 <= len(plan) <= MAX_STEPS:
        raise PlanError(f'plan must be a list of 1..{MAX_STEPS} steps')
    out = []
    for i, step in enumerate(plan):
        if not isinstance(step, dict) or step.get('skill') not in SKILL_DOC:
            raise PlanError(f'step {i}: unknown skill {step.get("skill") if isinstance(step, dict) else step!r}')
        name, p = step['skill'], dict(step.get('params') or {})
        oc = p.get('on_contact')
        if oc is not None and oc not in ON_CONTACT:
            raise PlanError(f'step {i}: on_contact must be one of {ON_CONTACT}')
        if name == 'MOVE':
            _point(p.get('to'), game_map, f'step {i} to')
            p['via'] = _points(p.get('via'), game_map, f'step {i} via')
        elif name == 'HOLD':
            p['seconds'] = _num(p, 'seconds', 0, 10, 2.0)
            if p.get('until') not in (None, 'contact'):
                raise PlanError(f'step {i}: until must be "contact" or omitted')
        elif name == 'SMOKE_RUSH':
            _point(p.get('throw_from'), game_map, f'step {i} throw_from')
            _point(p.get('smoke_at'), game_map, f'step {i} smoke_at')
            p['approach_via'] = _points(p.get('approach_via'), game_map, f'step {i} approach_via')
            p['run_via'] = _points(p.get('run_via'), game_map, f'step {i} run_via')
        elif name == 'FEINT':
            _point(p.get('at'), game_map, f'step {i} at')
            _point(p.get('staging'), game_map, f'step {i} staging')
            p['approach_via'] = _points(p.get('approach_via'), game_map, f'step {i} approach_via')
            p['relocate_via'] = _points(p.get('relocate_via'), game_map, f'step {i} relocate_via')
            p['show_s'] = _num(p, 'show_s', 0, 6, 3.0)
            p['leave_inside_m'] = _num(p, 'leave_inside_m', 8.5, 12, 9.5)
        elif name == 'DASH':
            p['via'] = _points(p.get('via'), game_map, f'step {i} via')
        elif name == 'TAG':
            p['max_s'] = _num(p, 'max_s', 0.5, 4, 2.0)
        elif name == 'RETREAT':
            p = {}
        out.append(dict(skill=name, params=p))
    return out


def parse_response(text, packet_id, game_map):
    """Model text -> (plan, assessment). Raises PlanError."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        start, end = (text or '').find('{'), (text or '').rfind('}')
        if start < 0 or end <= start:
            raise PlanError('response is not a JSON object') from None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            raise PlanError('response is not valid JSON') from None
    if not isinstance(data, dict):
        raise PlanError('response must be a JSON object')
    if data.get('observation_id') != packet_id:
        raise PlanError(f'observation_id must be {packet_id}')
    plan = validate_plan(data.get('plan'), game_map)
    a = data.get('assessment') if isinstance(data.get('assessment'), dict) else {}
    a = {k: str(v)[:600] for k, v in a.items() if k in ('human_style', 'evidence', 'decision_summary', 'risk')}
    if a.get('human_style') not in STYLES:
        a['human_style'] = 'unknown'
    return plan, a
