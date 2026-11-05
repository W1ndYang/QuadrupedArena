"""Isolated rule ablation: attacker must reach goal; tag provides a short window.

The previous 392 games and production referee are unchanged. This subclass adds
visible disabled status and own cooldown, both legally observable, to policies.
"""
import copy
import datetime

from run_study import R, read, dump, HEADINGS, DEFENDERS, DEFAULT_RULES
from fixed_goal_probe import Probe, PublicController, TagClock, save_match


class GoalOnlyClock(TagClock):
    def __init__(self, seconds=1., stun=1., cooldown=4.):
        super().__init__(seconds)
        self.time = 0.
        self.stun = stun
        self.cooldown = cooldown
        self.disabled_until = 0.
        self.cooldown_until = [0., 0.]

    def step(self, dt, valid, goal=False, timeout=False):
        self.time += dt
        valid = [valid[0] and self.time+1e-9 >= self.cooldown_until[0],
                 valid[1] and self.time+1e-9 >= self.disabled_until]
        _, _, hits = super().step(dt, valid, False, False)
        if hits[0]:
            self.disabled_until = self.time+self.stun
            self.cooldown_until[0] = self.time+self.cooldown
            self.age = [0., 0.]
            self.previous = [False, False]
        if goal and hits[1]:
            return 'draw', 'simultaneous', hits
        if hits[1]:
            return 'defender', 'tag', hits
        if goal:
            return 'attacker', 'goal', hits
        if timeout:
            return 'defender', 'timeout', hits
        return None, None, hits


class GoalOnlyController(PublicController):
    def decide(self, packet):
        if self.actor == 0:
            if packet['self'].get('tag_cooldown_remaining', 0) > 1e-8:
                return dict(mode='ADVANCE', route=self.route)
            if packet['visible'] is not None and packet['visible'].get('disabled_remaining', 0) > 1e-8:
                return dict(mode='ADVANCE', route=self.route)
        return super().decide(packet)


class GoalOnlyProbe(Probe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clock = GoalOnlyClock()
        for i, old in enumerate(self.controllers):
            self.controllers[i] = GoalOnlyController(old.kind, old.actor, old.route, old.home, old.patrol, old.rules)

    def observation(self, i, t):
        packet = super().observation(i, t)
        packet['self']['tag_cooldown_remaining'] = max(0., self.clock.cooldown_until[i]-t)
        if i == 0 and packet['visible'] is not None:
            packet['visible']['disabled_remaining'] = max(0., self.clock.disabled_until-t)
        return packet

    def execute(self, i, command, t):
        if i == 1 and t < self.clock.disabled_until-1e-9:
            self.stop(i)
            return self.actor[i].yaw, False, 'DISABLED'
        return super().execute(i, command, t)


def main():
    layout = next(q for q in read(R/'geometry_results.json') if q['id'] == 'E')
    base = read(R/'source_snapshot/configs/trial_a391_feedback_fix.json')
    base['start_b'] = layout['selected_guard']['position']
    rules = copy.deepcopy(DEFAULT_RULES)
    rules.update(version='fixed-goal-probe-v0.1-goal-only-ablation',
                 win_condition='attacker_goal_only; defender_tag_or_timeout', human_stun_seconds=1., tag_cooldown_seconds=4.)
    dump(R/'goal_only_protocol.json', dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         trigger='55/60 revised-spawn E cases ended on tag, only5/60 on goal; median4.78s. The proposed tag-win shortcut may overshadow task completion.',
         change='Only attacker goal entry wins. An attacker tag disables defender for1s; attacker has4s tag cooldown. Defender tag still ends the round.',
         observability='Visible disabled status and own cooldown added to public packet; no hidden position added.',
         execution='On disabled defender or own cooldown, the baseline resumes its already selected route. No extra route optimization.',
         policies=['contact_tag', 'wide_tag'], defenders=DEFENDERS, headings=HEADINGS, cases=24,
         limitation='Changes terminal condition and necessary stun/cooldown response together; an exploratory rule-set comparison, not a single-parameter causal claim.'))
    results = []
    for ap in ('contact_tag', 'wide_tag'):
        for dp in DEFENDERS:
            for h, yaw in enumerate(HEADINGS):
                position = layout['center'] if dp == 'on_goal' else layout['selected_guard']['position']
                identifier = f'goal_only_E_{ap}_{dp}_{h}_upright'
                route = [layout['via'], layout['center']] if ap == 'wide_tag' else [layout['center']]
                probe = GoalOnlyProbe(base, rules, layout['center'], route, position, dp, ap, yaw=yaw, patrol=layout['patrol'])
                result = probe.run(identifier)
                result['summary'].update(layout='E', condition='goal_only',
                     human_tags=sum(0 in e['actors'] for e in result['events'] if e['type'] == 'tag'))
                result['summary']['human_disabled_seconds'] = 1.
                save_match(R/'matches', result)
                results.append(result['summary'])
                print(identifier, result['summary']['winner'], result['summary']['reason'], result['summary']['seconds'], 'tags', result['summary']['human_tags'], flush=True)
    dump(R/'goal_only_results.json', results)


if __name__ == '__main__':
    main()
