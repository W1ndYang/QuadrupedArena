"""Deterministic deciders (no API): a fixed script, and a read-and-counter baseline.

ReadAndCounter is the reference policy for the intended loop:
  scout from the feint point -> classify the human from legal sightings ->
  guard (hold / stand-off / unseen) => SMOKE_RUSH with the map's path smoke
  hunt                               => relocate out of sight, wait for commit, DASH
It uses only the packet built by QuadAgent (legal observation + contact log) and the public map.
"""
import math

from .geometry import boxes, clear_ray, line_of_sight
from .nav import Planner, path_length


class _Sync:
    def __init__(self):
        self.ready = []

    def poll(self, sim_t):
        out, self.ready = self.ready, []
        return out

    def answer(self, packet, plan, assessment):
        self.ready.append(dict(observation_id=packet['observation_id'], plan=plan, assessment=assessment,
                               latency_s=0.0))


class ScriptedDecider(_Sync):
    """Returns `plan` once, then holds. Used to force a specific play in tests and demos."""

    def __init__(self, plan, then=None):
        super().__init__()
        self.plans = [plan] + ([then] if then else [])

    def submit(self, packet):
        plan = self.plans.pop(0) if self.plans else [dict(skill='HOLD', params=dict(seconds=1.0))]
        self.answer(packet, plan, dict(decision_summary='scripted'))


def classify(contacts, goal, now):
    """Human style from legal sightings: hold | standoff | hunt | unknown (+ last sighting).

    hunt: seen 5 m or more from the goal, or still moving and has moved at least 0.6 m toward
    the quadruped while 2 m or more from the goal. hold: last seen within 1.5 m of the goal and
    not moving away. Otherwise standoff (settled somewhere near the goal).
    """
    recent = [c for c in contacts if now - c['t'] <= 6.0]
    if not recent:
        return 'unknown', None
    last = recent[-1]
    d_last = math.dist(last['xy'], goal)
    prior = [c for c in recent if 0.25 <= last['t'] - c['t'] <= 1.0]
    speed = (math.dist(prior[-1]['xy'], last['xy']) / (last['t'] - prior[-1]['t'])) if prior else 0.0
    toward = 0.0
    for c in recent:
        if last['t'] - c['t'] >= 0.75:
            me = c.get('me') or last.get('me')
            if me is None:
                continue
            ux, uy = me[0] - c['xy'][0], me[1] - c['xy'][1]
            n = math.hypot(ux, uy) or 1.0
            toward = max(toward, ((last['xy'][0] - c['xy'][0]) * ux + (last['xy'][1] - c['xy'][1]) * uy) / n)
    if d_last >= 5.0 or (d_last >= 2.0 and toward >= 0.6 and speed >= 0.3):
        return 'hunt', last
    if d_last <= 1.5:
        return 'hold', last
    return 'standoff', last


class ReadAndCounter(_Sync):
    def __init__(self, feint=None):
        super().__init__()
        self.feint = feint
        self.stage = 'scout'
        self.read = None
        self.flanks = 0
        self.failed = set()

    def attach(self, game_map, rules):
        self.map, self.rules = game_map, rules
        self.obstacles = boxes(game_map.covers)
        self.nav = Planner(game_map.covers, game_map.size, rules.planner.clearance_m, rules.planner.grid_m)
        self.feint = self.feint or game_map.names('feint')[0]

    def plays(self, key):
        return self.map.plays.get(key, {})

    def hidden_from(self, xy, name):
        return not line_of_sight(xy, self.rules.human.eye_height_m, self.map.points[name],
                                 self.rules.quadruped.postures.upright.height_m, self.obstacles)

    def submit(self, packet):
        obs = packet['observation']
        now = obs['t']
        style, last = classify(packet['contacts'], self.map.goal, now)
        if self.stage == 'scout':
            self.stage = 'read'
            approach = self.plays('approaches').get(self.feint, [])
            plan = [dict(skill='MOVE', params=dict(to=self.feint, via=approach, on_contact='retreat')),
                    dict(skill='HOLD', params=dict(seconds=2.5, on_contact='retreat'))]
            return self.answer(packet, plan, dict(decision_summary=f'scout from {self.feint} to read the human'))
        for rep in packet.get('recent', []):
            if rep['skill'] == 'SMOKE_RUSH' and rep['status'] != 'done':
                self.failed.add(rep['params']['throw_from'])
        if obs['self']['smoke_left'] == 0:
            door = self.nearest_door(obs['self']['xy'])
            return self.answer(packet, [dict(skill='DASH', params=dict(via=[door]))],
                               dict(decision_summary='smoke spent: commit through nearest door'))
        if self.stage == 'read' and style in ('hunt', 'standoff') and last is not None \
                and math.dist(last['xy'], self.map.goal) < 5.0 and now - last['t'] < 1.0 and packet['trigger'].startswith('plan finished'):
            self.stage = 'reread'
            return self.answer(packet, [dict(skill='HOLD', params=dict(seconds=1.5, on_contact='retreat'))],
                               dict(decision_summary=f'reading {style} at {last["xy"]}; watch 1.5 s more before committing'))
        self.read = style
        if style == 'hunt' and self.flanks == 0:
            self.flanks += 1
            return self.answer(packet, *self.flank(obs, last))
        return self.answer(packet, *self.smoke(obs, style, last))

    def nearest_door(self, xy):
        return min(self.map.names('door'), key=lambda d: math.dist(xy, self.map.points[d]))

    def exposure_m(self, path, observer, smoke):
        """Longest continuous visible stretch (m) of a polyline from observer with one smoke."""
        eye = self.rules.human.eye_height_m
        h = self.rules.quadruped.postures.upright.height_m
        sm = self.rules.smoke
        cloud = [dict(center=smoke, radius=sm.radius_m, height=sm.height_m)] if smoke else []
        best = run = 0.0
        for a, b in zip(path, path[1:]):
            n = max(1, int(math.dist(a, b) / 0.2))
            for k in range(n):
                q = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                if math.dist(q, self.map.goal) <= self.map.goal_radius:
                    return best
                seen = math.dist(q, observer) <= self.rules.tag.range_m and line_of_sight(observer, eye, q, h, self.obstacles, cloud)
                run = run + math.dist(a, b) / n if seen else 0.0
                best = max(best, run)
        return best

    def screen_landing(self, throw, observers, hint):
        """Landing within throw range that minimises the worst exposure of the run to `observers`."""
        T = self.map.points[throw]
        run = [T] + (self.nav.plan_via(T, [self.map.resolve(v) for v in self.plays('throws')[throw]['run_via']]
                                       + [self.map.goal]) or [self.map.goal])
        sm = self.rules.smoke
        best = None
        step = 0.5
        n = int(sm.throw_range_m / step)
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                q = (T[0] + i * step, T[1] + j * step)
                if math.dist(q, T) > sm.throw_range_m or not clear_ray((*T, sm.throw_height_m), (*q, sm.throw_height_m), self.obstacles):
                    continue
                worst = max(self.exposure_m(run, o, q) for o in observers)
                key = (round(worst, 1), math.dist(q, hint))
                if best is None or key < best[0]:
                    best = (key, [round(q[0], 2), round(q[1], 2)])
        return best[1], best[0][0]

    def smoke(self, obs, style, last):
        throws = self.plays('throws')
        ref = last['xy'] if last else self.map.goal
        order = self.map.data.get('rule_planner', {}).get('throw_preference', list(throws))
        if self.failed and all(t in self.failed for t in order):
            self.failed.clear()
            return ([dict(skill='HOLD', params=dict(seconds=3.0, on_contact='retreat'))],
                    dict(decision_summary='every throw spot was contested; wait in cover for the human to move'))
        pick = None
        for name in order:
            if name not in self.failed and self.hidden_from(ref, name) and math.dist(ref, self.map.points[name]) >= 4.0:
                pick = name
                break
        pick = pick or next(t for t in order if t not in self.failed)
        hint = throws[pick]['smoke_hints']['path']
        landing, worst = hint, None
        if last is not None:
            landing, worst = self.screen_landing(pick, [tuple(last['xy']), self.map.goal], hint)
        plan = [dict(skill='SMOKE_RUSH', params=dict(throw_from=pick, smoke_at=landing,
                                                     approach_via=self.plays('approaches').get(pick, []),
                                                     run_via=throws[pick]['run_via']))]
        why = (f'human reads as {style}; smoke from {pick} at {landing} screens the run from the last sighting '
               f'and from the goal' + (f' (worst visible stretch {worst} m)' if worst is not None else ' (map path hint)'))
        return plan, dict(observation_evidence=f'last legal sighting {ref}', decision_summary=why)

    def flank(self, obs, last):
        lures = self.plays('lures')
        here = obs['self']['xy']
        options = []
        for name, L in lures.items():
            if not self.hidden_from(last['xy'], L['staging']):
                continue
            options.append((L['commit_m'], math.dist(here, self.map.points[L['feint']]), name))
        if not options:
            return self.smoke(obs, 'hunt-but-no-hidden-staging', last)
        _, _, name = min(options)
        L = lures[name]
        staging = L['staging']
        route = self.nav.plan_via(here, [self.map.resolve(v) for v in L['relocate_via']] + [self.map.points[staging]])
        travel = path_length(here, route) / 2.9 + 0.4 if route else 6.0
        d0 = math.dist(last['xy'], self.map.goal)
        commit_at = last['t'] + 1.2 + max(0.0, L['commit_m'] + 0.5 - d0) / self.rules.human.speed
        wait = max(0.0, commit_at - (obs['t'] + travel))
        plan = [dict(skill='MOVE', params=dict(to=staging, via=L['relocate_via'], on_contact='retreat')),
                dict(skill='HOLD', params=dict(seconds=round(wait, 2), on_contact='retreat')),
                dict(skill='DASH', params=dict(via=self.plays('stagings')[staging]['dash_via']))]
        why = (f'human left the goal (last seen {d0:.1f} m out at t={last["t"]}); relocate to {staging} '
               f'unseen, wait {wait:.1f} s for him to pass {L["commit_m"]} m, then dash')
        return plan, dict(observation_evidence=f'hunter at {last["xy"]}', decision_summary=why)
