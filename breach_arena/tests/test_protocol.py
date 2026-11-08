import json
import unittest

from breach_arena.agent import QuadAgent
from breach_arena.config import GameMap, load_rules
from breach_arena.engine import Match
from breach_arena.humans import make_bot
from breach_arena.llm.broker import LLMDecider
from breach_arena.llm.client import FakeClient
from breach_arena.llm.fake import RulePolicyModel
from breach_arena.llm.protocol import PlanError, parse_response, system_prompt, validate_plan

M = GameMap.load('BA-1')


class Validation(unittest.TestCase):
    def test_good_plan(self):
        plan = validate_plan([dict(skill='MOVE', params=dict(to='F_W', via=['W_PLAZA_N'])),
                              dict(skill='SMOKE_RUSH', params=dict(throw_from='T_N', smoke_at=[8.6, 2.2],
                                                                   run_via=['N_DOOR_OUT', 'N_DOOR'])),
                              dict(skill='HOLD', params=dict(seconds=2))], M)
        self.assertEqual([s['skill'] for s in plan], ['MOVE', 'SMOKE_RUSH', 'HOLD'])

    def test_rejections(self):
        bad = [
            [dict(skill='TELEPORT', params={})],
            [dict(skill='MOVE', params=dict(to='NOWHERE'))],
            [dict(skill='MOVE', params=dict(to='F_W', on_contact='fight'))],
            [dict(skill='HOLD', params=dict(seconds=99))],
            [dict(skill='SMOKE_RUSH', params=dict(throw_from='T_N', smoke_at='x'))],
            [dict(skill='HOLD', params={})] * 6,
            [],
        ]
        for p in bad:
            with self.assertRaises(PlanError):
                validate_plan(p, M)

    def test_parse_response(self):
        text = 'reasoning... {"observation_id": 3, "plan": [{"skill": "DASH", "params": {"via": ["N_DOOR"]}}], ' \
               '"assessment": {"human_style": "hunt", "decision_summary": "go"}} trailing'
        plan, a = parse_response(text, 3, M)
        self.assertEqual(plan[0]['skill'], 'DASH')
        self.assertEqual(a['human_style'], 'hunt')
        with self.assertRaises(PlanError):
            parse_response(text, 4, M)
        with self.assertRaises(PlanError):
            parse_response('no json here', 3, M)


class PromptContents(unittest.TestCase):
    def test_prompt_keeps_test_and_bot_data_out(self):
        r = load_rules()
        for hints in ('none', 'routes', 'full'):
            p = system_prompt(M, r, hints)
            for secret in ('cycle_checks', 'human_bots', 'hunt_route', 'rule_planner', 'throw_preference'):
                self.assertNotIn(secret, p)
            self.assertIn('SMOKE_RUSH', p)
            self.assertEqual('smoke_hints' in p, hints == 'full')
            self.assertEqual('"plays"' in p, hints != 'none')
        self.assertNotIn('Guarding usually loses', system_prompt(M, r, strategy=False))


class LegalPackets(unittest.TestCase):
    def test_model_only_sees_legal_detections(self):
        r = load_rules()
        packets = []
        model = RulePolicyModel(M, r)

        def respond(user):
            packets.append(user)
            return model(user)
        dec = LLMDecider(FakeClient(respond), timing='fixed:1.0')
        match = Match(M, r, QuadAgent(dec), make_bot('hunt'), record_every=1)
        match.run()
        self.assertGreaterEqual(len(packets), 2)
        by_t = {s['t']: s for s in match.trace}
        for p in packets:
            json.dumps(p)
            state = by_t[p['t']]
            self.assertEqual(p['human_visible_now'] is not None, state['visible'][0])
            if p['human_visible_now'] is not None:
                self.assertEqual(p['human_visible_now']['xy'], state['xy'][1])


if __name__ == '__main__':
    unittest.main()
