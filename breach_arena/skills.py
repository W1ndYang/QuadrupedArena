"""Quadruped skills: closed-loop executors driven only by the quadruped's legal observation.

A skill turns one observation into one engine command per tick and reports
``status`` = running | done | interrupted | failed with a short ``reason``.
Skills know the public map (named points, walls) but not the human's hidden state.
"""
import math

from .geometry import line_of_sight

TOL = 0.15          # arrival tolerance at a skill target (m)
VIA_TOL = 0.5       # an intermediate via point counts as passed inside this radius
CONTACT_M = 9.0     # 'retreat' guard: opponent visible inside tag range + 1 m
GLIMPSE_S = 0.3     # ... for at least this long


class Skill:
    name = 'SKILL'

    def __init__(self, **params):
        self.params = params
        self.status, self.reason = 'running', ''
        self.started = None
        self.phase = ''

    def label(self):
        return self.name + (f':{self.phase}' if self.phase else '')

    def finish(self, status, reason=''):
        self.status, self.reason = status, reason

    def report(self):
        return dict(skill=self.name, params=self.params, status=self.status, reason=self.reason,
                    phase=self.phase, started=self.started)

    def update(self, obs, ctx):
        raise NotImplementedError


def _contact(obs, ctx=None, route=None, radius=CONTACT_M):
    """Retreat guard. Fires when the human is visible inside `radius` for GLIMPSE_S and the route
    ahead would keep the quadruped exposed (a door's view cone crossed at speed is ignored;
    a tag needs 1.2 s of continuous sight), or immediately inside 4 m."""
    o = obs['opponent']
    if o is None:
        return False
    me = obs['self']['xy']
    d = math.dist(o['xy'], me)
    if d <= 4.0:
        return True
    age = obs.get('contact_age', 1.0)
    if d > radius or age < GLIMPSE_S - 1e-9:
        return False
    if ctx is None or route is None:
        return True
    return age + ctx.time_to_hide(me, route.lookahead(me), o['xy']) > 0.9


class _Route:
    """Shared path following with via bookkeeping."""

    def __init__(self, ctx, to, via):
        self.to = ctx.map.resolve(to)
        self.via = [ctx.map.resolve(v) for v in via or []]

    def command(self, obs, mode, posture=None, fire=False, look=None):
        me = obs['self']['xy']
        while self.via and math.dist(me, self.via[0]) <= VIA_TOL:
            self.via.pop(0)
        return dict(goal=list(self.to), via=[list(v) for v in self.via], look=look, fire=fire,
                    posture=posture, mode=mode)

    def arrived(self, obs):
        return not self.via and math.dist(obs['self']['xy'], self.to) <= TOL

    def lookahead(self, me):
        return [tuple(me)] + [tuple(v) for v in self.via] + [tuple(self.to)]


class Move(Skill):
    """MOVE to a named point through optional via points.

    on_contact: 'continue' keeps going; 'retreat' ends the skill (status interrupted) when the
    human is visible within 9 m so the agent can pick cover.
    """
    name = 'MOVE'

    def update(self, obs, ctx):
        if self.started is None:
            self.started = obs['t']
            self.m = _Route(ctx, self.params['to'], self.params.get('via'))
        if self.params.get('on_contact', 'retreat') == 'retreat' and _contact(obs, ctx, self.m):
            self.finish('interrupted', 'contact')
            return ctx.stand(obs, self.label())
        if self.m.arrived(obs):
            self.finish('done', 'arrived')
            return ctx.stand(obs, self.label())
        return self.m.command(obs, self.label(), posture=self.params.get('posture', 'upright'))


class Hold(Skill):
    """HOLD in place for `seconds`; optionally end early when the human is seen (until='contact')."""
    name = 'HOLD'

    def update(self, obs, ctx):
        if self.started is None:
            self.started = obs['t']
        if self.params.get('until') == 'contact' and obs['opponent'] is not None:
            self.finish('done', 'contact')
        elif self.params.get('on_contact') == 'retreat' and _contact(obs):
            self.finish('interrupted', 'contact')
        elif obs['t'] - self.started >= self.params.get('seconds', 2.0) - 1e-9:
            self.finish('done', 'timer')
        cmd = ctx.stand(obs, self.label(), posture=self.params.get('posture'))
        if obs['opponent'] is not None:
            cmd['look'] = obs['opponent']['xy']
        return cmd


class SmokeRush(Skill):
    """SMOKE_RUSH: approach a throw spot, throw at `smoke_at`, then run to the goal via `run_via`.

    Approach phase retreats on contact; once the smoke is out the run is committed.
    """
    name = 'SMOKE_RUSH'

    def update(self, obs, ctx):
        p = self.params
        if self.started is None:
            self.started = obs['t']
            self.phase = 'approach'
            self.m = _Route(ctx, p['throw_from'], p.get('approach_via'))
            self.landing = list(ctx.map.resolve(p['smoke_at']))
        for n in obs['notices']:
            if n['type'] == 'throw_rejected':
                self.finish('failed', 'throw rejected: ' + n['reason'])
                return ctx.stand(obs, self.label())
            if n['type'] == 'smoke_released':
                self.phase = 'run'
                self.m = _Route(ctx, 'GOAL', p.get('run_via'))
        if self.phase == 'approach':
            if _contact(obs, ctx, self.m):
                self.finish('interrupted', 'contact before throw')
                return ctx.stand(obs, self.label())
            if self.m.arrived(obs):
                self.phase = 'throw'
            else:
                return self.m.command(obs, self.label(), posture='upright')
        if self.phase == 'throw':
            if obs['self']['smoke_left'] <= 0 and not obs['self']['casting']:
                self.finish('failed', 'no smoke left')
                return ctx.stand(obs, self.label())
            else:
                cmd = ctx.stand(obs, self.label())
                if not obs['self']['casting']:
                    cmd['throw'] = self.landing
                return cmd
        return self.m.command(obs, self.label(), posture='upright')


class Dash(Skill):
    """DASH: committed run to the goal via `via` (no smoke)."""
    name = 'DASH'

    def update(self, obs, ctx):
        if self.started is None:
            self.started = obs['t']
            self.m = _Route(ctx, 'GOAL', self.params.get('via'))
        return self.m.command(obs, self.label(), posture='upright')


class Feint(Skill):
    """FEINT: go to a point the human can see (out of tag range), show for `show_s` or until the
    human is seen moving toward it, then relocate out of sight to `staging` via `relocate_via`."""
    name = 'FEINT'

    def update(self, obs, ctx):
        p = self.params
        if self.started is None:
            self.started = obs['t']
            self.phase = 'go'
            self.m = _Route(ctx, p['at'], p.get('approach_via'))
            self.shown_at = None
        if self.phase == 'go':
            if self.m.arrived(obs):
                self.phase, self.shown_at = 'show', obs['t']
            else:
                return self.m.command(obs, self.label(), posture='upright')
        if self.phase == 'show':
            o = obs['opponent']
            close = o is not None and math.dist(o['xy'], obs['self']['xy']) <= p.get('leave_inside_m', 9.5)
            if close or obs['t'] - self.shown_at >= p.get('show_s', 3.0) - 1e-9:
                self.phase = 'relocate'
                self.m = _Route(ctx, p['staging'], p.get('relocate_via'))
            else:
                cmd = ctx.stand(obs, self.label())
                cmd['look'] = o['xy'] if o else list(ctx.map.goal)
                return cmd
        if self.m.arrived(obs):
            self.finish('done', 'at staging')
            return ctx.stand(obs, self.label())
        return self.m.command(obs, self.label(), posture='upright')


class Tag(Skill):
    """TAG: stop and aim at the visible human to stun him (1 s lock, 4 s cooldown)."""
    name = 'TAG'

    def update(self, obs, ctx):
        if self.started is None:
            self.started = obs['t']
        if any(n['type'] == 'you_tagged_human' for n in obs['notices']):
            self.finish('done', 'stunned human')
        o = obs['opponent']
        if o is None:
            self.finish('failed', 'no visible human')
        elif obs['t'] - self.started > self.params.get('max_s', 2.0):
            self.finish('failed', 'timer')
        cmd = ctx.stand(obs, self.label())
        if o is not None:
            cmd.update(look=o['xy'], fire=True)
        return cmd


class Retreat(Skill):
    """RETREAT to the nearest named point hidden from the human's last legal position."""
    name = 'RETREAT'

    def update(self, obs, ctx):
        if self.started is None:
            self.started = obs['t']
            threat = (obs['opponent'] or obs['memory'] or {}).get('xy')
            target = ctx.hidden_point(obs['self']['xy'], threat)
            self.params = dict(self.params, to=target)
            self.m = _Route(ctx, target, None)
        if self.m.arrived(obs):
            self.finish('done', 'in cover')
            return ctx.stand(obs, self.label())
        return self.m.command(obs, self.label(), posture='upright')


SKILLS = {cls.name: cls for cls in (Move, Hold, SmokeRush, Dash, Feint, Tag, Retreat)}


def build(step):
    """{'skill': NAME, 'params': {...}} -> Skill instance (params validated by protocol)."""
    return SKILLS[step['skill']](**step.get('params', {}))


class SkillContext:
    """Public knowledge for skills: the map and the quadruped's own planner."""

    def __init__(self, game_map, rules, planner, obstacles):
        self.map, self.rules, self.planner, self.obstacles = game_map, rules, planner, obstacles

    def stand(self, obs, mode, posture=None):
        return dict(goal=None, via=[], look=None, fire=False, posture=posture, mode=mode)

    def time_to_hide(self, me, polyline, threat, speed=3.2, horizon=1.2, dt=0.05):
        """Seconds until a point moving along polyline at `speed` leaves the threat's sight."""
        eye = self.rules.human.eye_height_m
        h = self.rules.quadruped.postures.upright.height_m
        t, seg, pos = 0.0, 0, list(polyline[0])
        while t <= horizon:
            if not line_of_sight(threat, eye, pos, h, self.obstacles):
                return t
            step = speed * dt
            while step > 1e-9 and seg < len(polyline) - 1:
                a, b = pos, polyline[seg + 1]
                L = math.dist(a, b)
                if step >= L:
                    pos, step, seg = list(b), step - L, seg + 1
                else:
                    pos = [a[0] + (b[0] - a[0]) * step / L, a[1] + (b[1] - a[1]) * step / L]
                    step = 0.0
            t += dt
        return float('inf')

    def hidden_point(self, me, threat):
        names = [k for k, v in self.map.kinds.items() if v in ('cover', 'lane', 'staging', 'throw', 'feint')]
        if threat is None:
            return min(names, key=lambda k: math.dist(me, self.map.points[k]))
        quad_h = self.rules.quadruped.postures.upright.height_m
        eye = self.rules.human.eye_height_m
        hidden = [k for k in names if not line_of_sight(threat, eye, self.map.points[k], quad_h, self.obstacles)
                  and math.dist(self.map.points[k], threat) > 3.0]
        pool = hidden or names
        return min(pool, key=lambda k: math.dist(me, self.map.points[k]) - 0.3 * math.dist(threat, self.map.points[k]))
