"""Quadruped controller: runs skill plans and asks a decider for new plans without pausing the match.

Deciders implement ``submit(request)`` and ``poll(sim_t) -> list of responses``. A synchronous
decider (rule planner) answers in the same tick; an asynchronous one (LLM) answers later while
the match keeps running. While a request is in flight the agent keeps executing the current
skill, or a fallback (hold in place; retreat if the human is close).
"""
import copy
import math

from . import skills as S
from .geometry import boxes
from .nav import Planner

CONTACT_LOG_S = 0.25      # sample spacing of the legal contact log
ERROR_BACKOFF_S = 2.0     # after an invalid / failed response wait this long before asking again


class QuadAgent:
    def __init__(self, decider, history=None):
        self.decider = decider
        self.history = history or []          # summaries of earlier rounds in a series
        self.plan, self.skill = [], None
        self.contacts = []                    # legal sightings [{t, xy, yaw, stunned}]
        self.reports = []                     # finished skill reports
        self.log = []                         # decision requests / responses
        self.pending = None
        self.retry_at = 0.0
        self.wanted = 'start'
        self.next_id = 0
        self.last_obs = None

    def attach(self, game_map, rules):
        self.map, self.rules = game_map, rules
        self.ctx = S.SkillContext(game_map, rules, Planner(game_map.covers, game_map.size,
                                                           rules.planner.clearance_m, rules.planner.grid_m),
                                  boxes(game_map.covers))
        if hasattr(self.decider, 'attach'):
            self.decider.attach(game_map, rules)

    # ------------------------------------------------------------------ decisions
    def request(self, obs, trigger):
        rid = self.next_id
        self.next_id += 1
        packet = dict(observation_id=rid, trigger=trigger, observation=copy.deepcopy(obs),
                      contacts=self.contacts[-24:], current=self.skill.report() if self.skill else None,
                      queued=[copy.deepcopy(s) for s in self.plan], recent=self.reports[-6:],
                      history=self.history)
        self.pending = dict(id=rid, sent=obs['t'], trigger=trigger)
        self.log.append(dict(type='request', id=rid, t=obs['t'], trigger=trigger))
        self.decider.submit(packet)

    def install(self, resp, obs):
        entry = dict(type='response', id=resp['observation_id'], t=obs['t'],
                     sent=self.pending['sent'] if self.pending else None, latency_s=resp.get('latency_s'),
                     wall_s=resp.get('wall_s'), plan=resp.get('plan'), assessment=resp.get('assessment'),
                     error=resp.get('error'), usage=resp.get('usage'))
        self.log.append(entry)
        self.pending = None
        if resp.get('error') or not resp.get('plan'):
            self.retry_at = obs['t'] + ERROR_BACKOFF_S
            if self.skill is None and not self.plan:
                self.wanted = self.wanted or 'previous response rejected: ' + str(resp.get('error'))[:120]
            return
        self.plan = [copy.deepcopy(s) for s in resp['plan']]
        self.skill = None
        self.wanted = None

    # ------------------------------------------------------------------ per tick
    def _poll(self, obs):
        for resp in self.decider.poll(obs['t']):
            if self.pending and resp['observation_id'] == self.pending['id']:
                self.install(resp, obs)
            else:
                self.log.append(dict(type='dropped_response', id=resp.get('observation_id'), t=obs['t']))

    def decide(self, obs):
        self.last_obs = obs
        o = obs['opponent']
        if o is None:
            self.visible_since = None
        elif getattr(self, 'visible_since', None) is None:
            self.visible_since = obs['t']
        obs['contact_age'] = 0.0 if o is None else round(obs['t'] - self.visible_since, 3)
        if o is not None and (not self.contacts or obs['t'] - self.contacts[-1]['t'] >= CONTACT_LOG_S - 1e-9):
            self.contacts.append(dict(t=obs['t'], xy=o['xy'], yaw=o['yaw'], stunned=o.get('stunned', False),
                                      me=obs['self']['xy']))
        self._poll(obs)
        for _ in range(4):
            if self.skill is None and self.plan:
                self.skill = S.build(self.plan.pop(0))
            if self.skill is None and self.wanted and self.pending is None and obs['t'] >= self.retry_at - 1e-9:
                self.request(obs, self.wanted)
                self.wanted = None
                self._poll(obs)                      # synchronous deciders answer in the same tick
                continue
            if self.skill is None:
                return self.fallback(obs)
            cmd = self.skill.update(obs, self.ctx)
            if self.skill.status == 'running':
                return cmd
            rep = self.skill.report()
            rep['ended'] = obs['t']
            self.reports.append(rep)
            self.log.append(dict(type='skill_end', t=obs['t'], **rep))
            self.skill = None
            if rep['status'] != 'done':
                self.plan = []
                self.wanted = f"{rep['skill']} {rep['status']}: {rep['reason']}"
                if rep['reason'].startswith('contact'):
                    self.skill = S.Retreat()
            elif not self.plan:
                done = f"plan finished ({rep['skill']}: {rep['reason']})"
                self.wanted = f'{self.wanted}; then {done}' if self.wanted else done
        return self.fallback(obs)

    def fallback(self, obs):
        o = obs['opponent']
        if o is not None and math.dist(o['xy'], obs['self']['xy']) <= S.CONTACT_M:
            self.skill = S.Retreat()
            return self.skill.update(obs, self.ctx)
        cmd = self.ctx.stand(obs, 'WAIT_DECISION')
        if o is not None:
            cmd['look'] = o['xy']
        return cmd

    def round_summary(self, result):
        """Short legal-information summary passed to the next round of a series."""
        seen = self.contacts
        return dict(result=result, skills=[dict(skill=r['skill'], status=r['status'], reason=r['reason'],
                                                params=r['params']) for r in self.reports],
                    human_sightings=[dict(t=c['t'], xy=c['xy']) for c in seen[::max(1, len(seen) // 12)]][:12])
