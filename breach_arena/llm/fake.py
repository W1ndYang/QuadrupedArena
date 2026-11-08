"""Offline 'model' for tests and dry runs: answers protocol packets with the rule baseline.

It reads only the compact user packet (what a real model would see) and replies in the
protocol's JSON format, so prompt building, validation, latency and logging are exercised
without network access or cost.
"""
from ..planner_rule import ReadAndCounter


class RulePolicyModel:
    def __init__(self, game_map, rules):
        self.policy = ReadAndCounter()
        self.policy.attach(game_map, rules)

    def __call__(self, user):
        obs = dict(t=user['t'], remaining=user['remaining_s'], self=user['me'], opponent=user['human_visible_now'],
                   memory=user['human_last_known'], smokes=user['smokes'], notices=[])
        packet = dict(observation_id=user['observation_id'], trigger=user['trigger'], observation=obs,
                      contacts=[dict(t=s['t'], xy=s['human'], me=s.get('me')) for s in user['sightings']],
                      recent=[dict(r, params=r.get('params', {})) for r in user['recent_skills']],
                      current=user['current_skill'], history=user['earlier_rounds'])
        self.policy.submit(packet)
        out = self.policy.poll(user['t'])[0]
        a = out['assessment']
        return dict(observation_id=user['observation_id'], plan=out['plan'],
                    assessment=dict(human_style=self.policy.read or 'unknown',
                                    evidence=a.get('observation_evidence', ''),
                                    decision_summary=a.get('decision_summary', ''), risk='offline rule model'))
