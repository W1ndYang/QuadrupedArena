"""Decisions arrive while the match keeps running (no pause), with recorded latency."""
import time
import unittest

from breach_arena.agent import QuadAgent
from breach_arena.config import GameMap, load_rules
from breach_arena.engine import Match
from breach_arena.humans import make_bot
from breach_arena.llm.broker import LLMDecider
from breach_arena.llm.client import APIError, FakeClient
from breach_arena.llm.fake import RulePolicyModel

M, R = GameMap.load('BA-1'), load_rules()


def play(decider, human='hold'):
    agent = QuadAgent(decider)
    match = Match(M, R, agent, make_bot(human))
    return match, agent, match.run()


class Timing(unittest.TestCase):
    def test_fixed_latency_lands_later_in_match_time(self):
        dec = LLMDecider(FakeClient(RulePolicyModel(M, R)), timing='fixed:2.0')
        match, agent, res = play(dec)
        self.assertGreaterEqual(len(dec.calls), 2)
        for c in dec.calls:
            self.assertAlmostEqual(c['applied_t'] - c['sent_t'], 2.0, delta=0.021)
        # the quadruped was waiting (not paused): the human kept acting during the first 2 s
        self.assertGreater(match.trace[int(2.0 / 0.04)]['t'], 1.9)
        self.assertEqual(res['winner'], 'quadruped')

    def test_virtual_latency_equals_measured_wall_time(self):
        dec = LLMDecider(FakeClient(RulePolicyModel(M, R), delay_s=0.25), timing='virtual')
        play(dec)
        for c in dec.calls:
            self.assertGreaterEqual(c['latency_s'] + 0.021, c['wall_s'])
            self.assertLess(c['latency_s'] - c['wall_s'], 0.08)

    def test_pause_mode_applies_on_the_same_tick(self):
        dec = LLMDecider(FakeClient(RulePolicyModel(M, R)), timing='pause')
        play(dec)
        self.assertTrue(all(c['latency_s'] == 0.0 for c in dec.calls))


class Failures(unittest.TestCase):
    def test_invalid_plan_is_recorded_and_re_requested(self):
        n = {'calls': 0}
        good = RulePolicyModel(M, R)

        def flaky(user):
            n['calls'] += 1
            if n['calls'] == 1:
                return dict(observation_id=user['observation_id'], plan=[dict(skill='FLY', params={})])
            return good(user)
        dec = LLMDecider(FakeClient(flaky), timing='fixed:0.5')
        match, agent, res = play(dec)
        self.assertIn('invalid plan', dec.calls[0]['error'])
        self.assertGreaterEqual(len(dec.calls), 2)
        self.assertIsNotNone(res)

    def test_api_error_does_not_crash_the_match(self):
        class Broken:
            def public_config(self):
                return {}

            def complete(self, messages):
                raise APIError('HTTP 500')
        dec = LLMDecider(Broken(), timing='fixed:0.2')
        started = time.perf_counter()
        _, agent, res = play(dec)
        self.assertEqual(res['reason'], 'timeout')
        self.assertTrue(all('APIError' in c['error'] for c in dec.calls))
        sent = [c['sent_t'] for c in dec.calls]
        self.assertTrue(all(b - a >= 2.0 - 1e-6 for a, b in zip(sent, sent[1:])), 'failed calls must back off 2 s')
        self.assertLess(time.perf_counter() - started, 60)


if __name__ == '__main__':
    unittest.main()
