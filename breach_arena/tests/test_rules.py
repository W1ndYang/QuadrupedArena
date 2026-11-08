import math
import unittest

from breach_arena.config import load_rules
from breach_arena.engine import Match

from .helpers import Script, aim_at_visible, arena, idle, wall


def run(m, quad, human, duration=6.0, **ov):
    r = load_rules(overrides={'duration_s': duration, **ov})
    match = Match(m, r, quad, human, record_every=1)
    return match, match.run()


class TagRules(unittest.TestCase):
    def test_human_tags_after_one_second_of_lock(self):
        match, res = run(arena(), Script(idle), Script(aim_at_visible))
        self.assertEqual((res['winner'], res['reason']), ('human', 'tag'))
        self.assertGreaterEqual(res['t'], 1.0)
        self.assertLessEqual(res['t'], 1.06)

    def test_no_tag_beyond_range(self):
        _, res = run(arena(quad=(-8.5, 0.0)), Script(idle), Script(aim_at_visible), duration=3.0)
        self.assertEqual(res['reason'], 'timeout')

    def test_moving_shooter_cannot_tag(self):
        def walk_and_fire(obs):
            o = obs['opponent']
            return dict(goal=[0.0, 6.0] if obs['self']['xy'][1] < 5.9 else [0.0, -6.0], look=o['xy'] if o else None, fire=True)
        _, res = run(arena(), Script(idle), Script(walk_and_fire), duration=3.0)
        self.assertEqual(res['reason'], 'timeout')

    def test_wall_blocks_tag(self):
        m = arena(covers=[wall('W', -3.2, -2.8, -3, 3)])
        _, res = run(m, Script(idle), Script(aim_at_visible), duration=3.0)
        self.assertEqual(res['reason'], 'timeout')

    def test_quadruped_tag_stuns_and_does_not_end(self):
        match, res = run(arena(), Script(aim_at_visible), Script(idle), duration=3.0)
        tags = [e for e in match.events if e['type'] == 'tag']
        self.assertEqual(tags[0]['shooter'], 0)
        self.assertAlmostEqual(tags[0]['stun_until'] - tags[0]['t'], 1.0, places=6)
        self.assertEqual(res['reason'], 'timeout')
        stunned = [s for s in match.trace if s['stunned']]
        self.assertTrue(stunned)


class GoalAndSmoke(unittest.TestCase):
    def test_goal_wins(self):
        quad = Script(lambda o: dict(goal=[10.0, 6.0], look=None, fire=False))
        _, res = run(arena(human=(11.0, -7.0), human_yaw=-90), quad, Script(idle), duration=10.0)
        self.assertEqual((res['winner'], res['reason']), ('quadruped', 'goal'))

    def test_smoke_timing_and_sight_block(self):
        thrown = {}

        def quad(obs):
            cmd = dict(goal=None, look=None, fire=False)
            if not thrown:
                thrown['x'] = True
                cmd['throw'] = [-3.0, 0.0]
            return cmd
        match, res = run(arena(), Script(quad), Script(aim_at_visible), duration=4.0)
        rel = next(e for e in match.events if e['type'] == 'smoke_released')
        self.assertAlmostEqual(rel['t'], 0.36, delta=0.03)           # 0.35 s standing cast
        self.assertAlmostEqual(rel['active_at'] - rel['t'], 0.4, places=6)
        self.assertAlmostEqual(rel['expires_at'] - rel['active_at'], 5.0, places=6)
        self.assertEqual(res['reason'], 'timeout')                    # lock never completes: cloud between
        after = [s for s in match.trace if rel['active_at'] + 0.02 <= s['t'] < rel['expires_at']]
        self.assertTrue(after and not any(s['visible'][1] for s in after))

    def test_throw_out_of_range_or_through_high_wall_is_rejected_and_kept(self):
        for covers, target in (((), [1.0, 0.0]), ([wall('W', -4.2, -3.8, -3, 3)], [-2.0, 0.0])):
            sent = []

            def quad(obs, target=target, sent=sent):
                cmd = dict(goal=None, look=None, fire=False)
                if not sent:
                    sent.append(1)
                    cmd['throw'] = target
                return cmd
            match, _ = run(arena(covers=covers, human=(6.0, 6.0)), Script(quad), Script(idle), duration=1.0)
            self.assertTrue(any(e['type'] == 'throw_rejected' for e in match.events))
            self.assertEqual(match.inventory, 1)

    def test_throw_over_low_wall_allowed(self):
        sent = []

        def quad(obs):
            cmd = dict(goal=None, look=None, fire=False)
            if not sent:
                sent.append(1)
                cmd['throw'] = [-2.0, 0.0]
            return cmd
        match, _ = run(arena(covers=[wall('L', -4.2, -3.8, -3, 3, 0.7)], human=(6.0, 6.0)), Script(quad), Script(idle), duration=1.0)
        self.assertTrue(any(e['type'] == 'smoke_released' for e in match.events))


class Legality(unittest.TestCase):
    def test_observations_hide_opponent_behind_walls_and_memory_expires(self):
        m = arena(covers=[wall('W', -3.2, -2.8, -3, 3)], quad=(-6.0, 0.0), human=(0.0, 0.0))
        q, h = Script(idle), Script(idle)
        run(m, q, h, duration=2.0)
        self.assertTrue(all(o['opponent'] is None and o['memory'] is None for o in q.seen + h.seen))

    def test_bodies_keep_minimum_separation(self):
        quad = Script(lambda o: dict(goal=[0.0, 0.0], look=None, fire=False))
        match, _ = run(arena(quad=(-3.0, 0.0), human=(0.0, 0.0), human_yaw=90), quad, Script(idle), duration=2.0)
        self.assertGreaterEqual(match.min_separation, 0.7 - 1e-6)


if __name__ == '__main__':
    unittest.main()
